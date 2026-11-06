# SPDX-License-Identifier: BSD-3-Clause
import mmap
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from conftest import MEASUREMENT, SPEC, ScriptedBackend, measure, quiet, spin, values

from ernie_bench import harness
from ernie_bench.analysis import summarize_run
from ernie_bench.cli import main
from ernie_bench.results import RunWriter, Sample
from ernie_bench.config import Suite, Workload
from ernie_bench.system import load, machine, perf
from ernie_bench.system.monitor import Monitor

MB = 1 << 20
SORTABLE = np.random.default_rng(0).random(MB)
SHORT = replace(MEASUREMENT, iterations=10, repeats=2, max_repeats=2)

try:
    perf.Counters().close()
    PERF = None
except perf.PerfUnavailable as error:
    PERF = str(error)

needs_perf = pytest.mark.skipif(PERF is not None, reason=f"hardware counters unavailable: {PERF}")


class Sleeper(ScriptedBackend):
    def render(self):
        super().render()
        time.sleep(0.002)


class Helper(ScriptedBackend):
    def load(self, scene, workload):
        super().load(scene, workload)
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.work)
        self.worker.start()

    def work(self):
        while not self.stop.is_set():
            np.sort(SORTABLE)

    def close(self):
        self.stop.set()
        self.worker.join()
        super().close()


class Hoarder(ScriptedBackend):
    def load(self, scene, workload):
        super().load(scene, workload)
        self.scene_memory = np.ones(64 * MB, np.uint8)
        self.leak = []

    def render(self):
        super().render()
        self.leak.append(np.ones(MB, np.uint8))


class Toucher(ScriptedBackend):
    def render(self):
        super().render()
        region = mmap.mmap(-1, 16 * MB)
        pages = np.frombuffer(region, np.uint8)
        pages[:: os.sysconf("SC_PAGE_SIZE")] = 1
        del pages
        region.close()


class Accountant(ScriptedBackend):
    def __init__(self, threaded=False, **costs):
        super().__init__(**costs)
        self.threaded = threaded
        self.cpu = []

    def render(self):
        started = time.thread_time()
        super().render()
        if self.threaded:
            worker = threading.Thread(target=self.work)
            worker.start()
            worker.join()
        self.cpu.append(time.thread_time() - started + getattr(self, "worker_cpu", 0.0))
        self.worker_cpu = 0.0

    def work(self):
        started = time.thread_time()
        spin(self.cost["render"])
        self.worker_cpu = time.thread_time() - started


class Strider(ScriptedBackend):
    def __init__(self, size, **costs):
        super().__init__(**costs)
        self.data = np.ones(size // 8, np.int64)
        self.order = np.random.default_rng(0).permutation(len(self.data))[: 1 << 18]

    def render(self):
        super().render()
        self.data[self.order].sum()


def steady(samples, metric):
    return values(samples, "steady", metric)


@pytest.fixture
def two_cpus():
    original = os.sched_getaffinity(0)
    performance = machine.parse_cpu_list(machine.read("/sys/devices/cpu_core/cpus")) or sorted(original)
    chosen = set(performance[:2])
    if len(chosen) < 2:
        pytest.skip("needs two cpus of one core type")
    os.sched_setaffinity(0, chosen)
    yield chosen
    os.sched_setaffinity(0, original)


@pytest.fixture
def one_cpu():
    original = os.sched_getaffinity(0)
    cpu = max(original)
    os.sched_setaffinity(0, {cpu})
    yield cpu
    os.sched_setaffinity(0, original)


@pytest.mark.parametrize("threaded", [False, True], ids=["one thread", "two threads"])
def test_cpu_time_matches_what_the_backend_consumed(scene, monkeypatch, threaded):
    backend = Accountant(threaded=threaded, render=0.002)
    samples = measure(backend, scene, SHORT, monkeypatch=monkeypatch)
    consumed = np.array(backend.cpu[-SHORT.repeats * SHORT.iterations :]).reshape(SHORT.repeats, SHORT.iterations).sum(axis=1)
    counted = steady(samples, "process.cpu_seconds")
    assert (counted >= consumed).all()
    assert (counted <= consumed + 0.02).all()


def test_cpu_utilization_separates_busy_idle_and_threaded_backends(scene, monkeypatch):
    busy = steady(measure(ScriptedBackend(render=0.002), scene, SHORT, monkeypatch=monkeypatch), "process.cpu_utilization")
    idle = steady(measure(Sleeper(render=0.0002), scene, SHORT, monkeypatch=monkeypatch), "process.cpu_utilization")
    threaded = steady(measure(Helper(render=0.002), scene, SHORT, monkeypatch=monkeypatch), "process.cpu_utilization")
    assert np.median(busy) > 2 * np.median(idle)
    assert np.median(threaded) > np.median(busy) + 0.05


def test_sleeping_backends_show_voluntary_switches(scene, monkeypatch):
    samples = measure(Sleeper(), scene, SHORT, monkeypatch=monkeypatch)
    frames = SHORT.iterations
    assert (steady(samples, "process.voluntary_switches") >= frames).all()


def test_threads_created_by_the_backend_are_counted(scene, monkeypatch):
    plain = steady(measure(ScriptedBackend(), scene, SHORT, monkeypatch=monkeypatch), "process.threads")
    helped = steady(measure(Helper(), scene, SHORT, monkeypatch=monkeypatch), "process.threads")
    assert (helped == plain.max() + 1).all()


def test_scene_memory_and_leaks_are_measured(scene, monkeypatch):
    samples = measure(Hoarder(), scene, SHORT, monkeypatch=monkeypatch)
    assert values(samples, "load", "process.rss_growth_bytes")[0] == pytest.approx(64 * MB, rel=0.1)
    growth = steady(samples, "process.rss_growth_bytes")
    assert growth == pytest.approx([SHORT.iterations * MB] * len(growth), rel=0.15)
    assert (steady(samples, "process.peak_rss_bytes") >= steady(samples, "process.rss_bytes")).all()


def test_page_faults_follow_fresh_allocations(scene, monkeypatch):
    touching = steady(measure(Toucher(), scene, SHORT, monkeypatch=monkeypatch), "process.minor_faults")
    still = steady(measure(ScriptedBackend(), scene, SHORT, monkeypatch=monkeypatch), "process.minor_faults")
    pages = 16 * MB // os.sysconf("SC_PAGE_SIZE")
    assert (touching >= 0.9 * pages * SHORT.iterations).all()
    assert (still < 0.05 * pages * SHORT.iterations).all()


def test_preemption_is_flagged_and_does_not_bias_quiet_frames(scene, monkeypatch, one_cpu):
    hog = subprocess.Popen(
        [sys.executable, "-c", f"import os, time\nos.sched_setaffinity(0, {{{one_cpu}}})\nend = time.time() + 30\nwhile time.time() < end: pass"]
    )
    try:
        time.sleep(0.1)
        samples = measure(ScriptedBackend(render=0.0005), scene, replace(SHORT, iterations=200), monkeypatch=monkeypatch)
    finally:
        hog.kill()
        hog.wait()
    preempted = steady(samples, "preempted")
    assert preempted.mean() > 0.01
    assert (steady(samples, "process.involuntary_switches") > 0).all()
    assert (steady(samples, "system.foreign_cpu_seconds") > 0).all()
    assert steady(samples, "system.benchmark_cpus_utilization").min() > 0.9
    assert 0.0005 - 1e-6 <= np.median(quiet(samples, "render_seconds")) <= 0.0005 + 2e-4


@needs_perf
def test_cache_misses_follow_the_working_set(scene, monkeypatch, one_cpu):
    small = measure(Strider(64 * 1024), scene, SHORT, monkeypatch=monkeypatch)
    large = measure(Strider(256 * MB), scene, SHORT, monkeypatch=monkeypatch)
    assert np.median(steady(large, "perf.llc_misses")) > 10 * np.median(steady(small, "perf.llc_misses"))
    assert np.median(steady(large, "perf.cache_misses")) > 10 * np.median(steady(small, "perf.cache_misses"))
    assert (steady(small, "perf.cpu_migrations") == 0).all()


@needs_perf
def test_hardware_counters_include_backend_threads(scene, monkeypatch, two_cpus):
    plain = steady(measure(ScriptedBackend(render=0.002), scene, SHORT, monkeypatch=monkeypatch), "perf.cycles")
    helped = steady(measure(Helper(render=0.002), scene, SHORT, monkeypatch=monkeypatch), "perf.cycles")
    assert np.median(helped) > 1.3 * np.median(plain)


def test_counters_report_why_they_are_unavailable():
    if PERF is None:
        pytest.skip("hardware counters are available here")
    assert "perf_event_paranoid" in PERF


def test_system_load_difference_separates_foreign_time():
    start = {0: (100, 1000), 1: (100, 1000), 2: (0, 1000)}
    end = {0: (190, 1100), 1: (150, 1100), 2: (100, 1100)}
    measured = load.difference(start, end, {0, 1}, own_cpu_seconds=60 / load.TICKS)
    assert measured["cpu_utilization"] == pytest.approx(240 / 300)
    assert measured["benchmark_cpus_utilization"] == pytest.approx(140 / 200)
    assert measured["foreign_cpu_seconds"] == pytest.approx(80 / load.TICKS)


@pytest.mark.scans_processes
def test_top_processes_finds_a_busy_neighbour():
    hog = subprocess.Popen([sys.executable, "-c", "import time\nend = time.time() + 10\nwhile time.time() < end: pass"])
    try:
        time.sleep(0.1)
        busiest = load.top_processes(interval=0.5, count=50)
    finally:
        hog.kill()
        hog.wait()
    neighbour = next(p for p in busiest if p["pid"] == hog.pid)
    assert 5 < neighbour["cpu_percent"] <= 105


@pytest.mark.scans_processes
def test_top_processes_arithmetic(monkeypatch):
    snapshots = iter([{1: ("a", 100), 2: ("b", 50), 3: ("gone", 10)}, {1: ("a", 150), 2: ("b", 50), 4: ("new", 99)}])
    clock = iter([10.0, 10.5])
    monkeypatch.setattr(load, "_process_ticks", lambda: next(snapshots))
    monkeypatch.setattr(load, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(load, "sleep", lambda seconds: None)
    assert load.top_processes() == [{"pid": 1, "name": "a", "cpu_percent": 100 * 50 / load.TICKS / 0.5}]


def test_gpu_memory_is_sampled_when_a_gpu_is_present():
    with Monitor(interval=0.01) as monitor:
        if not monitor.gpu_count:
            pytest.skip("no NVML GPU")
        windows = monitor.window(monitor.mark(), monitor.mark())
    assert all(window["memory_used_bytes_max"] > 0 for window in windows)
    assert all(not np.isnan(window["energy_joules"]) for window in windows)


def test_counter_aggregation(tmp_path):
    workload = Workload(worlds=2)
    samples = []
    for repeat, (switches, rss, utilization, instructions, cycles, misses, references) in enumerate(
        [(10, 100, 0.5, 300, 100, 5, 50), (30, 300, 0.9, 500, 250, 15, 50), (999, 999, 9.0, 1, 1, 1, 1)]
    ):
        for iteration in range(5):
            samples.append(Sample("steady", "render_seconds", 0.01, repeat, iteration))
        samples += [
            Sample("steady", "disturbed", float(repeat == 2), repeat),
            Sample("steady", "process.voluntary_switches", switches, repeat),
            Sample("steady", "process.rss_bytes", rss, repeat),
            Sample("steady", "process.cpu_utilization", utilization, repeat),
            Sample("steady", "perf.instructions", instructions, repeat),
            Sample("steady", "perf.cycles", cycles, repeat),
            Sample("steady", "perf.cache_misses", misses, repeat),
            Sample("steady", "perf.cache_references", references, repeat),
            Sample("steady", "gpu0.utilization_mean", utilization * 100, repeat),
        ]
    samples.append(Sample("load", "process.rss_growth_bytes", 4096))
    run = RunWriter(tmp_path, {"run_id": "run", "outcomes": []})
    run.write("alpha", "s", workload, samples)

    _, (record,) = summarize_run(run.directory)
    counters = record["counters"]
    assert counters["process.voluntary_switches"] == pytest.approx(40 / 10)
    assert counters["process.rss_bytes"] == 300
    assert counters["process.cpu_utilization"] == pytest.approx(0.7)
    assert counters["gpu0.utilization_mean"] == pytest.approx(70)
    assert counters["perf.ipc"] == pytest.approx(800 / 350)
    assert counters["perf.cache_miss_rate"] == pytest.approx(20 / 100)
    assert record["load"]["process.rss_growth_bytes"] == 4096


def test_summary_counters_table_prints(tmp_path, scripted, capsys):
    scripted(alpha={"render": 0.001})
    directory = harness.run_suite(Suite("t", ("alpha",), (SPEC,), Workload(), SHORT), tmp_path, log=lambda _: None, isolation="inline")
    main(["summary", str(directory), "--counters"])
    out = capsys.readouterr().out
    assert "cpu%" in out and "fgn ms/f" in out
    assert "busiest other processes at start" in out


def test_scene_state_stays_on_one_core():
    from ernie_bench.datasets import build_scene

    scene = build_scene(SPEC, 256, 2)
    before = os.times()
    opened = time.perf_counter()
    for step in range(200):
        scene.state(step / 30)
    after = os.times()
    used = (after.user - before.user) + (after.system - before.system)
    assert used / (time.perf_counter() - opened) < 1.2
