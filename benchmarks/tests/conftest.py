# SPDX-License-Identifier: BSD-3-Clause
from time import perf_counter_ns

import numpy as np
import pytest

from ernie_bench import harness
from ernie_bench.backends import BACKENDS
from ernie_bench.backends.base import BackendInfo, Capabilities
from ernie_bench.datasets import build_scene
from ernie_bench.config import OUTPUTS, TIERS, Measurement, SceneSpec, Workload
from ernie_bench.system.monitor import Monitor

SPEC = SceneSpec("hide_and_seek", "hide_and_seek")
WORKLOAD = Workload(worlds=4, cameras_per_world=2, width=16, height=16)
MEASUREMENT = Measurement(
    warmup_min_iterations=3,
    warmup_min_seconds=0,
    warmup_max_seconds=0.5,
    warmup_window=3,
    warmup_cv=0.2,
    iterations=20,
    repeats=3,
    max_repeats=3,
)
TOLERANCE = 2e-4
SINGLE_SHOT = 0.02


def spin(seconds: float) -> None:
    end = perf_counter_ns() + int(seconds * 1e9)
    while perf_counter_ns() < end:
        pass


class ScriptedBackend:
    def __init__(self, load=0.0, update=0.0, render=0.0, sync=0.0, settle=0.0, slow_frames=0, slow=0.0, jitter=0.0, fail_on_frame=None, tiers=TIERS):
        self.cost = {"load": load, "update": update, "render": render, "sync": sync, "settle": settle}
        self.slow_frames, self.slow, self.jitter = slow_frames, slow, jitter
        self.fail_on_frame = fail_on_frame
        self.tiers = frozenset(tiers)
        self.calls, self.states, self.loads = [], [], []
        self.frames = 0
        self._submitted = 0

    def info(self):
        self.calls.append("info")
        return BackendInfo("scripted", "0", "cpu")

    def capabilities(self):
        self.calls.append("capabilities")
        return Capabilities(self.tiers, frozenset(OUTPUTS))

    def load(self, scene, workload):
        self.calls.append("load")
        self.loads.append((scene, workload))
        spin(self.cost["load"])

    def update(self, state):
        self.calls.append("update")
        self.states.append(state)
        spin(self.cost["update"])

    def render(self):
        self.calls.append("render")
        if self.frames == self.fail_on_frame:
            raise RuntimeError("render failed")
        extra = self.slow if self.frames < self.slow_frames else 0.0
        extra += self.jitter if self.frames % 2 else 0.0
        spin(self.cost["render"] + extra)
        self._submitted = perf_counter_ns()
        self.frames += 1

    def sync(self):
        self.calls.append("sync")
        settle = self._submitted + int(self.cost["settle"] * 1e9)
        while perf_counter_ns() < settle:
            pass
        spin(self.cost["sync"])

    def readback(self):
        self.calls.append("readback")
        return {}

    def stream(self):
        return None

    def close(self):
        self.calls.append("close")


@pytest.fixture
def scripted(monkeypatch):
    created = []

    def install(**costs):
        def create(name):
            backend = ScriptedBackend(**costs.get(name, {}))
            created.append((name, backend))
            return backend

        monkeypatch.setattr(harness, "create", create)
        for name in costs:
            monkeypatch.setitem(BACKENDS, name, ("scripted", {}))
        return created

    return install


def values(samples, phase, metric):
    return np.array([s.value for s in samples if s.phase == phase and s.metric == metric])


@pytest.fixture(scope="session")
def scene():
    return build_scene(SPEC, WORKLOAD.worlds, WORKLOAD.cameras_per_world)


def measure(backend, scene, measurement=MEASUREMENT, monitor_interval=0.001, monkeypatch=None):
    monkeypatch.setattr(harness, "create", lambda name: backend)
    with Monitor(monitor_interval) as monitor:
        samples, _ = harness.measure("scripted", scene, WORKLOAD, measurement, monitor)
    return samples


def costs_at_least(measured, cost, slack=TOLERANCE):
    assert cost - 1e-6 <= measured <= cost + slack


def never_under_and_best_is_exact(frames, cost):
    assert frames.min() >= cost - 1e-6
    assert frames.min() <= cost + TOLERANCE


def quiet(samples, metric, phase="steady"):
    preempted = {(s.repeat, s.iteration) for s in samples if s.phase == phase and s.metric == "preempted" and s.value}
    return np.array([s.value for s in samples if s.phase == phase and s.metric == metric and (s.repeat, s.iteration) not in preempted])


@pytest.fixture(autouse=True)
def skip_process_scan(request, monkeypatch):
    if "scans_processes" not in request.keywords:
        monkeypatch.setattr("ernie_bench.system.load.top_processes", lambda: [])


@pytest.fixture(autouse=True)
def ignore_system_clock_lock(monkeypatch, tmp_path_factory):
    monkeypatch.setattr("ernie_bench.system.tuning.SYSTEM_PREP_STATE", tmp_path_factory.mktemp("system-prep") / "absent.env")
