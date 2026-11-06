# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import random
import re
import resource
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from ernie_bench.harness import ROOT, finish_run, start_run
from ernie_bench.results import Sample
from ernie_bench.config import Workload
from ernie_bench.system.monitor import Monitor

MADRONA = ROOT / "external" / "madrona_benchmark"
CACHE = ROOT / "data" / "madrona-cache"
PROGRAMS = {"hideseek": "hideseek_headless", "habitat": "habitat_headless"}
VIEWS_PER_WORLD = {"hideseek": 6, "habitat": 1}
TIERS = {"rt": "direct", "rast": "unlit"}
STEP_TIME = re.compile(r"Average (?:total )?step time: ([0-9.]+) ms")


@dataclass(frozen=True)
class PaperGrid:
    scene: str = "hideseek"
    worlds: tuple[int, ...] = (256, 1024, 4096)
    resolutions: tuple[int, ...] = (64, 128, 256)
    modes: tuple[str, ...] = ("rt", "rast")
    steps: int = 200
    processes: int = 3
    timeout_seconds: float = 900.0
    seed: int = 0

    def __post_init__(self):
        if self.scene not in PROGRAMS:
            raise ValueError(f"unknown madrona scene {self.scene!r}, expected one of {sorted(PROGRAMS)}")
        unknown = set(self.modes) - set(TIERS)
        if unknown:
            raise ValueError(f"unknown madrona modes {sorted(unknown)}, expected rt or rast")


def run_program(grid: PaperGrid, worlds: int, resolution: int, mode: str, log_path: Path) -> dict:
    program = MADRONA / "build" / PROGRAMS[grid.scene]
    variables = dict(os.environ) | {
        "MADRONA_MWGPU_KERNEL_CACHE": str(CACHE / "kernels"),
        "MADRONA_BVH_KERNEL_CACHE": str(CACHE / "bvh"),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = perf_counter()
    with open(log_path, "w") as log:
        try:
            completed = subprocess.run(
                [str(program), str(worlds), str(grid.steps), mode, str(resolution), str(resolution)],
                cwd=MADRONA,
                env=variables,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=grid.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "reason": f"killed after {grid.timeout_seconds:g} s", "log": str(log_path)}
    wall = perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    output = log_path.read_text(errors="replace")
    found = STEP_TIME.search(output)
    if completed.returncode != 0 or not found:
        return {"status": "error", "reason": f"{program.name} exited with {completed.returncode}", "returncode": completed.returncode, "log": str(log_path), "log_tail": output[-2000:]}
    step = float(found.group(1)) / 1000
    return {
        "status": "ok",
        "returncode": 0,
        "log": str(log_path),
        "step_seconds": step,
        "wall_seconds": wall,
        "startup_seconds": wall - step * grid.steps,
        "cpu_seconds": (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
        "peak_rss_bytes": after.ru_maxrss * 1024,
    }


def run_paper(grid: PaperGrid, root: Path, log=print, cpus: str | None = None, nice: int | None = None, realtime: int | None = None) -> Path:
    program = MADRONA / "build" / PROGRAMS[grid.scene]
    if not program.exists():
        raise FileNotFoundError(f"{program} is missing, build external/madrona_benchmark first")
    writer, locked = start_run(f"madrona-paper-{grid.scene}", asdict(grid), root, log, cpus, nice, realtime)
    manifest = writer.manifest
    logs = writer.directory / "workers"
    logs.mkdir()
    points = [(worlds, resolution, mode) for worlds in grid.worlds for resolution in grid.resolutions for mode in grid.modes]
    order = random.Random(grid.seed)
    with Monitor(0.05, locked) as monitor:
        for process in range(grid.processes):
            order.shuffle(points)
            for worlds, resolution, mode in points:
                backend = f"madrona_{mode}_paper"
                workload = Workload(
                    name=f"madrona-paper-{grid.scene}",
                    worlds=worlds,
                    cameras_per_world=VIEWS_PER_WORLD[grid.scene],
                    width=resolution,
                    height=resolution,
                    outputs=("rgb",),
                    tier=TIERS[mode],
                )
                label = f"{len(manifest['outcomes']):05d}-{backend}"
                opened = monitor.mark()
                result = run_program(grid, worlds, resolution, mode, logs / f"{label}.log")
                gpus = monitor.window(opened, monitor.mark())
                outcome = {"backend": backend, "scene": grid.scene, "workload": asdict(workload), "process": process} | result
                if result["status"] == "ok":
                    disturbed = any(gpu["disturbed"] for gpu in gpus)
                    samples = [
                        Sample("startup", "process_seconds", result["startup_seconds"]),
                        Sample("steady", "render_seconds", result["step_seconds"], 0, 0),
                        Sample("steady", "disturbed", float(disturbed), 0),
                        Sample("steady", "process.cpu_seconds", result["cpu_seconds"], 0),
                        Sample("steady", "process.peak_rss_bytes", result["peak_rss_bytes"], 0),
                    ]
                    for gpu in gpus:
                        samples += [
                            Sample("steady", f"gpu{gpu['gpu']}.{key}", float(value), 0)
                            for key, value in gpu.items()
                            if key not in ("gpu", "throttle_reasons", "energy_joules")
                        ]
                        samples += [Sample("steady", f"gpu{gpu['gpu']}.throttle.{reason}", 1.0, 0) for reason in gpu["throttle_reasons"]]
                    writer.write(backend, grid.scene, workload, samples, process)
                    manifest["backends"][backend] = {"name": backend, "version": "madrona_benchmark", "device": "cuda:0", "details": {"program": PROGRAMS[grid.scene], "steps": grid.steps}}
                manifest["outcomes"].append(outcome)
                writer.save_manifest()
                log(f"{outcome['status']:<9} {backend:<18} worlds={worlds:<5} {resolution}x{resolution} process={process} {result.get('reason', '')}")
    return finish_run(writer)
