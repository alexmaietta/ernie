# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import math
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import traceback
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np

from ernie_bench import __version__, profile, reference
from ernie_bench.backends import BACKENDS, create, known
from ernie_bench.backends.base import OUTPUT_FORMATS, BackendInfo
from ernie_bench.datasets import build_scene
from ernie_bench.results import RunWriter, Sample
from ernie_bench.scene import Scene
from ernie_bench.config import Capture, Measurement, Profile, SceneConfig, Suite, Workload
from ernie_bench.stats import describe
from ernie_bench.system import cuda, load, machine, perf, process, tuning
from ernie_bench.system.monitor import Monitor


PROTOCOL = 1
COLD_CACHES = {
    "XDG_CACHE_HOME": "xdg",
    "ERNIE_BENCH_CACHE": "backend",
    "WARP_CACHE_PATH": "warp",
    "__GL_SHADER_DISK_CACHE_PATH": "driver",
    "MADRONA_MWGPU_KERNEL_CACHE": "madrona/kernels",
    "MADRONA_BVH_KERNEL_CACHE": "madrona/bvh",
}
ROOT = Path(__file__).resolve().parents[1]
GATED = "harness cuda events, stream held until render returns"
UNGATED = "harness cuda events without a gate, because render waited on its own stream, so launch gaps count"
DROP_FAILED = "could not drop the page cache before cold runs; allow sudo -n scripts/system-prep drop-caches in sudoers"


class Unsupported(Exception):
    pass


def capture_images(backend, scene: Scene, workload: Workload, capture: Capture, directory: Path) -> None:
    outputs = [output for output in capture.outputs or workload.outputs if output in workload.outputs]
    images = list(range(workload.image_count)) if capture.images == "all" else [i for i in capture.images if i < workload.image_count]
    directory.mkdir(parents=True, exist_ok=True)
    for step in capture.sequence:
        backend.update(scene.state(step * workload.dt))
        backend.render()
        backend.sync()
        frames = backend.readback()
        arrays = {}
        for output in outputs:
            array = np.asarray(frames[output])
            channels, dtype = OUTPUT_FORMATS[output]
            expected = (workload.image_count, workload.height, workload.width, channels)
            if array.shape != expected or array.dtype != dtype:
                raise ValueError(f"{output} readback is {array.dtype}{list(array.shape)}, expected {np.dtype(dtype)}{list(expected)}")
            arrays[output] = array[images]
        np.savez(directory / f"step-{step:05d}.npz", **arrays)
    (directory / "index.json").write_text(json.dumps({"steps": list(capture.sequence), "images": images, "outputs": outputs}))


def warm_up(frame, measurement: Measurement) -> list[Sample]:
    times, converged = [], False
    window = measurement.warmup_window
    start = perf_counter_ns()
    while True:
        times.append(frame("warmup"))
        elapsed = (perf_counter_ns() - start) / 1e9
        if len(times) >= max(measurement.warmup_min_iterations, 2 * window) and elapsed >= measurement.warmup_min_seconds:
            previous, latest = np.array(times[-2 * window : -window]), np.array(times[-window:])
            tolerance = measurement.warmup_cv * latest.mean()
            converged = latest.std() <= tolerance and previous.std() <= tolerance and abs(previous.mean() - latest.mean()) <= tolerance
        if converged or elapsed >= measurement.warmup_max_seconds:
            break
    return [
        Sample("warmup", "iterations", len(times)),
        Sample("warmup", "seconds", elapsed),
        Sample("warmup", "converged", float(converged)),
    ]


def measure_steady(frame, opening, record, measurement: Measurement) -> list[Sample]:
    samples, clean_frames, clean, repeat, relative_ci = [], [], 0, 0, math.inf
    while repeat < measurement.max_repeats:
        opened, before = opening()
        frames = [frame("steady", repeat, iteration) for iteration in range(measurement.iterations)]
        disturbed = any(gpu["disturbed"] for gpu in record("steady", repeat, opened, before))
        samples.append(Sample("steady", "disturbed", float(disturbed), repeat))
        repeat += 1
        if not disturbed:
            clean += 1
            clean_frames += frames
            relative_ci = describe(clean_frames, measurement.confidence).relative_ci
            if clean >= measurement.repeats and relative_ci <= measurement.target_ci:
                break
    return samples + [
        Sample("result", "repeats", repeat),
        Sample("result", "clean_repeats", clean),
        Sample("result", "relative_ci", relative_ci),
    ]


def measure(
    name: str,
    scene: Scene,
    workload: Workload,
    measurement: Measurement,
    monitor: Monitor,
    capture: Capture | None = None,
    capture_directory: Path | None = None,
    profiled_frames: int | None = None,
) -> tuple[list[Sample], BackendInfo]:
    try:
        counters = perf.Counters()
    except perf.PerfUnavailable:
        counters = None
    backend = None
    timer = None
    samples = []
    cpus = os.sched_getaffinity(0)

    def snapshot():
        return process.usage(), counters.read() if counters else None, load.cpu_times()

    def opening():
        return monitor.mark(), snapshot()

    def record(phase: str, repeat: int, opened: float, before) -> list[dict]:
        usage, counts, times = snapshot()
        gpus = monitor.window(opened, monitor.mark())
        used = process.difference(before[0], usage)
        samples.extend(Sample(phase, f"process.{key}", value, repeat) for key, value in used.items())
        samples.extend(Sample(phase, f"system.{key}", value, repeat) for key, value in load.difference(before[2], times, cpus, used["cpu_seconds"]).items())
        if counters:
            samples.extend(Sample(phase, f"perf.{key}", value, repeat) for key, value in perf.difference(before[1], counts).items())
        for gpu in gpus:
            samples.extend(
                Sample(phase, f"gpu{gpu['gpu']}.{key}", float(value), repeat)
                for key, value in gpu.items()
                if key not in ("gpu", "throttle_reasons")
            )
            samples.extend(Sample(phase, f"gpu{gpu['gpu']}.throttle.{reason}", 1.0, repeat) for reason in gpu["throttle_reasons"])
        return gpus

    try:
        start = perf_counter_ns()
        backend = create(name)
        samples.append(Sample("startup", "create_seconds", (perf_counter_ns() - start) / 1e9))
        info = backend.info()
        reasons = backend.capabilities().unsupported(workload)
        if reasons:
            raise Unsupported("; ".join(reasons))

        opened, before = opening()
        start = perf_counter_ns()
        backend.load(scene, workload)
        samples.append(Sample("load", "seconds", (perf_counter_ns() - start) / 1e9))
        record("load", -1, opened, before)

        stream = None if profiled_frames else backend.stream()
        if stream is not None:
            try:
                timer = cuda.Timer(stream)
            except cuda.CudaUnavailable as error:
                samples.append(Sample("startup", "device_timer_error", 1.0))
                info.details["device_timer"] = str(error)
        info.details.setdefault("device_timer", GATED if timer else "none")
        step = 0

        def frame(phase: str, repeat: int = -1, iteration: int = -1) -> float:
            nonlocal step
            state = scene.state(step * workload.dt)
            step += 1
            switches = resource.getrusage(resource.RUSAGE_THREAD).ru_nivcsw
            t0 = perf_counter_ns()
            backend.update(state)
            t1 = perf_counter_ns()
            timed = timer is not None and phase != "first_frame"
            if timed:
                timer.begin(watched=phase == "warmup")
            backend.render()
            if timed:
                timer.end()
            backend.sync()
            t2 = perf_counter_ns()
            preempted = resource.getrusage(resource.RUSAGE_THREAD).ru_nivcsw - switches
            if timed and timer.blocked and info.details["device_timer"] != UNGATED:
                samples.append(Sample(phase, "device_timer_blocked", 1.0))
                info.details["device_timer"] = UNGATED
            if phase != "warmup":
                samples.append(Sample(phase, "update_seconds", (t1 - t0) / 1e9, repeat, iteration))
                samples.append(Sample(phase, "render_seconds", (t2 - t1) / 1e9, repeat, iteration))
                samples.append(Sample(phase, "preempted", float(preempted), repeat, iteration))
                if timed:
                    metric = "render_device_seconds" if timer.gated else "render_device_span_seconds"
                    samples.append(Sample(phase, metric, timer.seconds(), repeat, iteration))
            return (t2 - t0) / 1e9

        frame("first_frame")
        samples.extend(warm_up(frame, measurement))
        if profiled_frames:
            with cuda.profiled():
                for iteration in range(profiled_frames):
                    frame("profile", 0, iteration)
            return samples, info
        samples.extend(measure_steady(frame, opening, record, measurement))
        if capture:
            capture_images(backend, scene, workload, capture, capture_directory)
        return samples, info
    finally:
        if timer:
            timer.close()
        if backend:
            backend.close()
        if counters:
            counters.close()


def check(name: str, scene: Scene, workload: Workload, images: list[int]) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    backend = create(name)
    try:
        capabilities = backend.capabilities()
        workload = replace(
            workload,
            outputs=tuple(output for output in workload.outputs if output in capabilities.outputs),
            height=workload.width if capabilities.square_only else workload.height,
        )
        reasons = capabilities.unsupported(workload)
        if reasons:
            raise Unsupported("; ".join(reasons))
        backend.load(scene, workload)
        state = scene.state(0.0)
        backend.update(state)
        backend.render()
        backend.sync()
        frames = {output: array[images] for output, array in backend.readback().items()}
    finally:
        backend.close()
    depth, classes, rgb = reference.render(scene, state, workload.width, workload.height, images, workload.tier if "rgb" in workload.outputs else None)
    scores = reference.agreement(depth, classes, frames, reference_rgb=rgb)
    return scores, frames | {"workload": workload, "reference_rgb": rgb}, depth, classes


def attempt(
    name: str,
    scene: Scene,
    workload: Workload,
    measurement: Measurement,
    monitor: Monitor,
    capture: Capture | None = None,
    capture_directory: Path | None = None,
    profiled_frames: int | None = None,
) -> dict:
    try:
        samples, info = measure(name, scene, workload, measurement, monitor, capture, capture_directory, profiled_frames)
    except Unsupported as error:
        return {"status": "unsupported", "reason": str(error)}
    except Exception as error:
        return {"status": "error", "reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    result = {"status": "ok", "samples": samples, "info": asdict(info)}
    if capture:
        result["captures"] = str(capture_directory)
    return result


def environment(name: str, root: Path = ROOT) -> str:
    python = root / "envs" / name.partition(":")[0].split("_")[0] / ".venv" / "bin" / "python"
    return str(python) if python.exists() else sys.executable


def run_worker(
    name: str,
    spec: SceneConfig,
    workload: Workload,
    measurement: Measurement,
    locked: dict,
    directory: Path,
    label: str,
    capture: Capture | None = None,
    capture_directory: Path | None = None,
    profiler: Profile | None = None,
) -> dict:
    if workload.sm_percent < 100 and not tuning.mps_running():
        return {"status": "unsupported", "reason": "MPS is not running, run: sudo scripts/system-prep apply --mps"}
    job_path, result_path, log_path = (directory / f"{label}.{suffix}" for suffix in ("job.json", "result.json", "log"))
    report = directory / f"{label}{profile.SUFFIXES[profiler.tool]}" if profiler else None
    spawned = perf_counter()
    job_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "backend": name,
                "scene": asdict(spec),
                "workload": asdict(workload),
                "measurement": asdict(measurement),
                "locked": {str(gpu): clock for gpu, clock in locked.items()},
                "spawned": spawned,
                "result": str(result_path),
                "capture": asdict(capture) if capture else None,
                "captures": str(capture_directory) if capture_directory else None,
                "profiled_frames": profiler.frames if profiler else None,
            }
        )
    )
    variables = dict(os.environ)
    if workload.sm_percent < 100:
        variables["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(workload.sm_percent)
    cache = directory / f"{label}.cache"
    dropped = None
    if measurement.caches == "cold":
        variables |= {name: str(cache / folder) for name, folder in COLD_CACHES.items()}
        dropped = subprocess.run(["sudo", "-n", str(ROOT / "scripts" / "system-prep"), "drop-caches"], capture_output=True).returncode == 0
    prefix = profile.command(profiler.tool, report) if profiler else []
    try:
        with open(log_path, "w") as log:
            worker = subprocess.Popen(
                [*prefix, environment(name), "-m", "ernie_bench.worker", str(job_path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=variables,
                cwd=directory,
            )
            try:
                returncode = worker.wait(timeout=measurement.timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
                return {"status": "timeout", "reason": f"killed after {measurement.timeout_seconds:g} s", "log": str(log_path)}
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    tail = log_path.read_text(errors="replace")[-2000:]
    if not result_path.exists():
        return {"status": "error", "reason": f"worker exited with {returncode} and no result", "returncode": returncode, "log": str(log_path), "log_tail": tail}
    result = json.loads(result_path.read_text())
    if result.get("protocol") != PROTOCOL:
        return {"status": "error", "reason": f"worker speaks protocol {result.get('protocol')}, expected {PROTOCOL}", "log": str(log_path)}
    result["samples"] = [Sample(*sample) for sample in result.get("samples", [])]
    if profiler and result["status"] == "ok":
        try:
            result["samples"] += profile.collect(profiler.tool, report)
            result["report"] = str(report)
        except (OSError, subprocess.CalledProcessError) as error:
            result |= {"status": "error", "reason": f"could not read the {profiler.tool} report: {error}"}
    if dropped is not None:
        result["page_cache_dropped"] = dropped
    return result | {"returncode": returncode, "log": str(log_path)}


def _repository() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def start_run(
    name: str, described: dict, root: Path, log, cpus: str | None, nice: int | None, realtime: int | None = None, mode: str = "timing"
) -> tuple[RunWriter, dict]:
    warnings = []
    if cpus is not None:
        tuning.pin(cpus)
    if nice is not None and (warning := tuning.prioritise(nice)):
        warnings.append(warning)
    if realtime is not None and (warning := tuning.schedule_realtime(realtime)):
        warnings.append(warning)
    state = tuning.current()
    warnings += tuning.check(state)
    try:
        perf.Counters().close()
    except perf.PerfUnavailable as error:
        warnings.append(f"hardware counters are off: {error}; sudo scripts/system-prep apply enables them")
    for warning in warnings:
        log(f"warning: {warning}")

    system_prep = state["system_prep"] or {}
    locked = {int(system_prep["GPU"]): float(system_prep["APPLIED_GPU_CLOCK"])} if system_prep.get("APPLIED_GPU_CLOCK") else {}

    created = datetime.now()
    writer = RunWriter(
        root,
        {
            "run_id": f"{created:%Y%m%d-%H%M%S}-{name}",
            "created": created.isoformat(),
            "mode": mode,
            "status": "running",
            "ernie_bench": __version__,
            "repository": _repository(),
            "machine": machine.snapshot(),
            "tuning": state,
            "warnings": warnings,
            "processes_start": load.top_processes(),
            "suite": described,
            "backends": {},
            "outcomes": [],
        },
    )
    return writer, locked


def finish_run(writer: RunWriter) -> Path:
    writer.manifest["tuning_end"] = tuning.current()
    writer.manifest["processes_end"] = load.top_processes()
    writer.manifest["status"] = "complete"
    writer.save_manifest()
    return writer.directory


def run_suite(
    suite: Suite,
    root: Path,
    log=print,
    cpus: str | None = None,
    nice: int | None = None,
    isolation: str = "process",
    realtime: int | None = None,
    profiler: Profile | None = None,
) -> Path:
    if isolation not in ("process", "inline"):
        raise ValueError(f"unknown isolation {isolation!r}, expected process or inline")
    if isolation == "inline" and suite.measurement.caches == "cold":
        raise ValueError("cold caches need process isolation")
    if isolation == "inline" and profiler:
        raise ValueError("profiling needs process isolation")
    if isolation == "inline" and any(workload.sm_percent < 100 for workload in suite.points()):
        raise ValueError("limiting SMs with MPS needs process isolation")
    unknown = {name for name in suite.backends if not known(name)}
    if unknown:
        raise ValueError(f"unknown backends {sorted(unknown)}, expected one of {sorted(BACKENDS)}")

    described = asdict(suite) | ({"profile": asdict(profiler)} if profiler else {})
    writer, locked = start_run(suite.name, described, root, log, cpus, nice, realtime, "profile" if profiler else "timing")
    manifest = writer.manifest
    order = random.Random(suite.seed)
    workers = writer.directory / "workers"
    workers.mkdir()

    with ExitStack() as stack:
        monitor = stack.enter_context(Monitor(suite.measurement.monitor_interval, locked)) if isolation == "inline" else None
        for spec in suite.scenes:
            scene, built_for = None, None
            for workload in suite.scene_points(spec):
                if isolation == "inline" and built_for != (workload.worlds, workload.cameras_per_world):
                    built_for = (workload.worlds, workload.cameras_per_world)
                    scene = build_scene(spec, *built_for)
                for process_index in range(1 if profiler else suite.measurement.processes):
                    backends = list(suite.backends)
                    order.shuffle(backends)
                    for name in backends:
                        label = f"{len(manifest['outcomes']):05d}-{name.replace(':', '.')}"
                        capture = suite.capture if process_index == 0 and not profiler else None
                        captures = writer.directory / "captures" / label if capture else None
                        if isolation == "inline":
                            result = attempt(name, scene, workload, suite.measurement, monitor, capture, captures)
                        else:
                            result = run_worker(name, spec, workload, suite.measurement, locked, workers, label, capture, captures, profiler)
                        outcome = {"backend": name, "scene": spec.name, "workload": asdict(workload), "process": process_index}
                        outcome |= {key: value for key, value in result.items() if key not in ("samples", "info")}
                        if result["status"] == "ok":
                            writer.write(name, spec.name, workload, result["samples"], process_index)
                            manifest["backends"][name] = result["info"]
                        manifest["outcomes"].append(outcome)
                        if outcome.get("page_cache_dropped") is False and DROP_FAILED not in manifest["warnings"]:
                            manifest["warnings"].append(DROP_FAILED)
                            log(f"warning: {DROP_FAILED}")
                        writer.save_manifest()
                        log(
                            f"{outcome['status']:<11} {name:<8} {spec.name} worlds={workload.worlds} "
                            f"cams={workload.cameras_per_world} {workload.width}x{workload.height} "
                            f"process={process_index} {outcome.get('reason', '')}"
                        )

    return finish_run(writer)
