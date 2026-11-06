# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import replace

import numpy as np
import pytest
from conftest import MEASUREMENT, SINGLE_SHOT, SPEC, TOLERANCE, WORKLOAD, ScriptedBackend, costs_at_least, measure, never_under_and_best_is_exact, spin, values

from ernie_bench import harness
from ernie_bench.datasets import build_scene
from ernie_bench.results import read_run
from ernie_bench.config import Measurement, SceneSpec, Suite, Workload
from ernie_bench.system.monitor import Monitor

def test_each_phase_reports_its_own_cost(scene, monkeypatch):
    backend = ScriptedBackend(load=0.005, update=0.001, render=0.002, sync=0.0005)
    samples = measure(backend, scene, monkeypatch=monkeypatch)
    costs_at_least(values(samples, "load", "seconds")[0], 0.005, SINGLE_SHOT)
    never_under_and_best_is_exact(values(samples, "steady", "update_seconds"), 0.001)
    never_under_and_best_is_exact(values(samples, "steady", "render_seconds"), 0.0025)


def test_asynchronous_work_is_counted_through_sync(scene, monkeypatch):
    backend = ScriptedBackend(settle=0.003)
    samples = measure(backend, scene, monkeypatch=monkeypatch)
    never_under_and_best_is_exact(values(samples, "steady", "render_seconds"), 0.003)
    never_under_and_best_is_exact(values(samples, "steady", "update_seconds"), 0.0)


def test_scene_state_generation_is_not_timed(scene, monkeypatch):
    class SlowStateScene:
        def __init__(self, scene):
            self.scene = scene

        def state(self, time):
            spin(0.003)
            return self.scene.state(time)

    samples = measure(ScriptedBackend(render=0.001), SlowStateScene(scene), monkeypatch=monkeypatch)
    never_under_and_best_is_exact(values(samples, "steady", "update_seconds"), 0.0)
    never_under_and_best_is_exact(values(samples, "steady", "render_seconds"), 0.001)


def test_first_frame_and_warmup_stay_out_of_steady_state(scene, monkeypatch):
    backend = ScriptedBackend(render=0.001, slow_frames=6, slow=0.01)
    samples = measure(backend, scene, monkeypatch=monkeypatch)
    costs_at_least(values(samples, "first_frame", "render_seconds")[0], 0.011, SINGLE_SHOT)
    assert values(samples, "warmup", "iterations")[0] >= 5 + 2 * MEASUREMENT.warmup_window
    assert values(samples, "warmup", "converged")[0] == 1 or values(samples, "warmup", "seconds")[0] >= MEASUREMENT.warmup_max_seconds
    assert values(samples, "steady", "render_seconds").max() < 0.005


def test_warmup_gives_up_at_its_time_cap(scene, monkeypatch):
    backend = ScriptedBackend(render=0.0002, jitter=0.002)
    measurement = replace(MEASUREMENT, warmup_cv=0.01, warmup_max_seconds=0.05)
    samples = measure(backend, scene, measurement, monkeypatch=monkeypatch)
    assert values(samples, "warmup", "converged")[0] == 0
    costs_at_least(values(samples, "warmup", "seconds")[0], 0.05, SINGLE_SHOT)


def test_monitor_does_not_inflate_frame_times(scene, monkeypatch):
    idle = measure(ScriptedBackend(render=0.002), scene, monitor_interval=10.0, monkeypatch=monkeypatch)
    busy = measure(ScriptedBackend(render=0.002), scene, monitor_interval=0.001, monkeypatch=monkeypatch)
    never_under_and_best_is_exact(values(busy, "steady", "render_seconds"), 0.002)
    idle_typical = np.percentile(values(idle, "steady", "render_seconds"), 25)
    busy_typical = np.percentile(values(busy, "steady", "render_seconds"), 25)
    assert busy_typical == pytest.approx(idle_typical, abs=TOLERANCE)


def test_backend_sees_the_protocol_in_order(scene, monkeypatch):
    backend = ScriptedBackend()
    samples = measure(backend, scene, monkeypatch=monkeypatch)
    frames = 1 + int(values(samples, "warmup", "iterations")[0]) + len(values(samples, "steady", "render_seconds"))
    assert backend.calls == ["info", "capabilities", "load"] + ["update", "render", "sync"] * frames + ["close"]


def test_states_advance_one_step_per_frame(scene, monkeypatch):
    backend = ScriptedBackend()
    measure(backend, scene, monkeypatch=monkeypatch)
    times = [state.time for state in backend.states]
    assert times == pytest.approx([step * WORKLOAD.dt for step in range(len(times))])
    for state in backend.states[:: max(1, len(times) // 5)]:
        expected = scene.state(state.time)
        assert np.array_equal(state.transform, expected.transform)
        assert np.array_equal(state.camera, expected.camera)


def test_backend_writing_into_state_cannot_corrupt_the_scene(scene, monkeypatch):
    before = scene.state(0.0)
    instances = scene.instances.transform.copy()
    cameras = scene.cameras.local.copy()

    class Vandal(ScriptedBackend):
        def update(self, state):
            super().update(state)
            state.transform[:] = 0
            state.camera[:] = 0
            state.instance[:] = 0

    measure(Vandal(), scene, monkeypatch=monkeypatch)
    after = scene.state(0.0)
    assert np.array_equal(after.transform, before.transform)
    assert np.array_equal(after.camera, before.camera)
    assert np.array_equal(after.instance, before.instance)
    assert np.array_equal(scene.instances.transform, instances)
    assert np.array_equal(scene.cameras.local, cameras)


def test_close_runs_when_the_backend_fails(scene, monkeypatch):
    backend = ScriptedBackend(fail_on_frame=3)
    with pytest.raises(RuntimeError, match="render failed"):
        measure(backend, scene, monkeypatch=monkeypatch)
    assert backend.calls[-1] == "close"


def test_unsupported_workload_never_loads(scene, monkeypatch):
    backend = ScriptedBackend(tiers=("depth",))
    with pytest.raises(harness.Unsupported):
        measure(backend, scene, monkeypatch=monkeypatch)
    assert backend.calls == ["info", "capabilities", "close"]


SWEEP_SUITE = Suite(
    "sweep",
    ("alpha", "beta"),
    (SPEC, replace(SPEC, name="other", seed=7)),
    Workload(),
    replace(MEASUREMENT, iterations=2, repeats=1, max_repeats=1),
    {"worlds": [1, 3], "cameras_per_world": [1, 2], "resolution": [[8, 8], [16, 12]], "outputs": [["rgb"], ["depth"]]},
)


def test_every_point_runs_once_with_a_matching_scene(tmp_path, scripted):
    created = scripted(alpha={}, beta={})
    manifest, _ = read_run(harness.run_suite(SWEEP_SUITE, tmp_path, log=lambda _: None, isolation="inline"))

    expected = len(SWEEP_SUITE.scenes) * len(SWEEP_SUITE.points()) * len(SWEEP_SUITE.backends)
    assert len(manifest["outcomes"]) == len(created) == expected
    assert {o["status"] for o in manifest["outcomes"]} == {"ok"}

    seen = set()
    for (name, backend), outcome in zip(created, manifest["outcomes"]):
        (scene, workload), = backend.loads
        assert outcome["backend"] == name
        assert outcome["workload"] == {**outcome["workload"], "worlds": workload.worlds, "width": workload.width}
        assert scene.name == outcome["scene"]
        assert scene.world_count == workload.worlds
        assert len(scene.cameras) == workload.image_count
        seen.add((name, scene.name, workload))
    assert len(seen) == expected


def test_backend_order_is_seeded_and_varies_between_points(tmp_path, scripted):
    scripted(alpha={}, beta={})
    first, _ = read_run(harness.run_suite(SWEEP_SUITE, tmp_path / "a", log=lambda _: None, isolation="inline"))
    second, _ = read_run(harness.run_suite(SWEEP_SUITE, tmp_path / "b", log=lambda _: None, isolation="inline"))
    order = [o["backend"] for o in first["outcomes"]]
    assert order == [o["backend"] for o in second["outcomes"]]
    leaders = set(order[0::2])
    assert leaders == {"alpha", "beta"}


def test_a_failing_backend_does_not_stop_the_others(tmp_path, scripted):
    scripted(alpha={"fail_on_frame": 0}, beta={})
    manifest, table = read_run(harness.run_suite(SWEEP_SUITE, tmp_path, log=lambda _: None, isolation="inline"))
    statuses = {o["backend"]: set() for o in manifest["outcomes"]}
    for outcome in manifest["outcomes"]:
        statuses[outcome["backend"]].add(outcome["status"])
    assert statuses == {"alpha": {"error"}, "beta": {"ok"}}
    assert "render failed" in manifest["outcomes"][0]["reason"] or "render failed" in manifest["outcomes"][1]["reason"]
    assert set(table["backend"].to_pylist()) == {"beta"}
