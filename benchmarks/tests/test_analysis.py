# SPDX-License-Identifier: BSD-3-Clause
import math
from dataclasses import replace

import numpy as np
import pytest

from ernie_bench import harness
from ernie_bench.analysis import summarize_run
from ernie_bench.results import RunWriter, Sample, read_run
from ernie_bench.config import Measurement, SceneSpec, Suite, Workload

WORKLOAD = Workload(worlds=8, cameras_per_world=2, width=32, height=16, outputs=("rgb", "depth"), tier="direct", spp=4)


def writer(tmp_path):
    return RunWriter(tmp_path, {"run_id": "run", "outcomes": []})


def steady(frames, disturbed=(), energy=None):
    samples = []
    for repeat, repeat_frames in enumerate(frames):
        for iteration, (update, render) in enumerate(repeat_frames):
            samples.append(Sample("steady", "update_seconds", update, repeat, iteration))
            samples.append(Sample("steady", "render_seconds", render, repeat, iteration))
        samples.append(Sample("steady", "disturbed", float(repeat in disturbed), repeat))
        if energy:
            samples.append(Sample("steady", "gpu0.energy_joules", energy[repeat], repeat))
    return samples


def test_rows_round_trip_exactly(tmp_path):
    run = writer(tmp_path)
    awkward = [1e-9, 1 / 3, 123456.789, math.inf, 0.0]
    first = [Sample("steady", "render_seconds", value, 0, i) for i, value in enumerate(awkward)]
    second = [Sample("load", "seconds", 2.5)]
    run.write("alpha", "hide_and_seek", WORKLOAD, first)
    run.write("beta", "hide_and_seek", replace(WORKLOAD, worlds=1), second)

    _, table = read_run(run.directory)
    rows = sorted(table.to_pylist(), key=lambda r: (r["backend"], r["iteration"]))
    assert len(rows) == 6
    assert [r["value"] for r in rows[:5]] == awkward
    assert rows[0] | {"value": None, "iteration": None} == {
        "run_id": "run",
        "backend": "alpha",
        "scene": "hide_and_seek",
        "workload": rows[0]["workload"],
        "worlds": 8,
        "cameras_per_world": 2,
        "width": 32,
        "height": 16,
        "outputs": "rgb,depth",
        "tier": "direct",
        "spp": 4,
        "sm_percent": 100,
        "process": 0,
        "phase": "steady",
        "metric": "render_seconds",
        "repeat": 0,
        "iteration": None,
        "value": None,
    }
    assert rows[5]["worlds"] == 1 and rows[5]["value"] == 2.5


def test_summary_matches_hand_computed_statistics(tmp_path):
    run = writer(tmp_path)
    clean = [[(0.05, 0.15), (0.05, 0.25), (0.05, 0.20), (0.05, 0.30)], [(0.05, 0.10), (0.05, 0.20), (0.05, 0.35), (0.05, 0.15)]]
    disturbed = [[(5.0, 5.0)] * 4]
    run.write("alpha", "s", WORKLOAD, steady([clean[0], disturbed[0], clean[1]], disturbed={1}, energy=[10.0, 999.0, 30.0]))

    _, (record,) = summarize_run(run.directory)
    frames = np.array([u + r for repeat in clean for u, r in repeat])
    assert record["repeats"] == 3
    assert record["clean_repeats"] == 2
    assert record["frames"] == 8
    assert record["median_seconds"] == pytest.approx(np.median(frames))
    assert record["iqr_seconds"] == pytest.approx(np.percentile(frames, 75) - np.percentile(frames, 25))
    assert record["ci_low_seconds"] <= record["median_seconds"] <= record["ci_high_seconds"]
    assert record["images_per_second"] == pytest.approx(16 / np.median(frames))
    assert record["images_per_joule"] == pytest.approx(16 * 8 / 40.0)


def test_points_that_differ_only_in_outputs_stay_separate(tmp_path):
    run = writer(tmp_path)
    run.write("alpha", "s", replace(WORKLOAD, outputs=("rgb",)), steady([[(0.0, 0.001)] * 4]))
    run.write("alpha", "s", replace(WORKLOAD, outputs=("depth",)), steady([[(0.0, 0.009)] * 4]))
    _, records = summarize_run(run.directory)
    medians = {r["outputs"]: r["median_seconds"] for r in records}
    assert medians == {"rgb": pytest.approx(0.001), "depth": pytest.approx(0.009)}


def test_energy_is_withheld_for_short_measurements(tmp_path):
    run = writer(tmp_path)
    run.write("alpha", "s", WORKLOAD, steady([[(0.0, 0.01)] * 4], energy=[5.0]))
    _, (record,) = summarize_run(run.directory)
    assert record["images_per_joule"] is None


def test_every_repeat_disturbed_reports_no_statistics(tmp_path):
    run = writer(tmp_path)
    run.write("alpha", "s", WORKLOAD, steady([[(0.0, 0.01)] * 4, [(0.0, 0.01)] * 4], disturbed={0, 1}))
    _, (record,) = summarize_run(run.directory)
    assert record["frames"] == 0
    assert "median_seconds" not in record


def test_drift_is_zero_when_stable_and_large_when_still_warming(tmp_path):
    run = writer(tmp_path)
    run.write("stable", "s", WORKLOAD, steady([[(0.0, 0.01)] * 8]))
    run.write("warming", "s", WORKLOAD, steady([[(0.0, 0.02)] * 4 + [(0.0, 0.01)] * 4]))
    _, records = summarize_run(run.directory)
    drift = {r["backend"]: r["drift"] for r in records}
    assert drift["stable"] == 0
    assert drift["warming"] == pytest.approx(1.0 / 1.5)


SUITE = Suite(
    "end_to_end",
    ("alpha", "beta"),
    (SceneSpec("hide_and_seek", "hide_and_seek"),),
    Workload(width=8, height=8),
    Measurement(warmup_min_iterations=3, warmup_min_seconds=0, warmup_max_seconds=0.2, warmup_window=3, warmup_cv=0.2, iterations=10, repeats=3, max_repeats=3),
    {"worlds": [2, 6]},
)


def test_summary_reproduces_known_backend_costs(tmp_path, scripted):
    scripted(alpha={"render": 0.002}, beta={"update": 0.001, "settle": 0.003})
    _, records = summarize_run(harness.run_suite(SUITE, tmp_path, log=lambda _: None, isolation="inline"))
    expected = {"alpha": 0.002, "beta": 0.004}
    assert len(records) == 4
    for record in records:
        cost = expected[record["backend"]]
        assert cost - 1e-6 <= record["min_seconds"] <= cost + 2e-4
        assert record["median_seconds"] >= cost - 1e-6
        if record["quiet_median_seconds"] is not None:
            assert record["quiet_median_seconds"] >= cost - 1e-6
        assert record["images_per_second"] == pytest.approx(record["worlds"] / record["median_seconds"])


def test_leftover_warmup_shows_up_as_drift(tmp_path, scripted):
    scripted(alpha={"render": 0.001, "slow_frames": 25, "slow": 0.004})
    eager = replace(SUITE.measurement, warmup_cv=1e9, warmup_max_seconds=10.0)
    _, (record, _) = summarize_run(harness.run_suite(replace(SUITE, backends=("alpha",), measurement=eager), tmp_path, log=lambda _: None, isolation="inline"))
    assert record["drift"] > 0.5
