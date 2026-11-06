# SPDX-License-Identifier: BSD-3-Clause
import math
from dataclasses import replace

import pyarrow.compute as pc
import pytest

from ernie_bench import harness
from ernie_bench.backends.base import Capabilities
from ernie_bench.backends.null import NullBackend
from ernie_bench.cli import main
from ernie_bench.harness import run_suite
from ernie_bench.results import read_run
from ernie_bench.config import Measurement, SceneSpec, Suite, Workload

BROKEN = "ernie_bench.backends.null:Missing"

MEASUREMENT = Measurement(
    warmup_min_iterations=2,
    warmup_min_seconds=0,
    warmup_max_seconds=0.05,
    warmup_window=2,
    iterations=4,
    repeats=3,
    max_repeats=5,
    target_ci=1.0,
)

SUITE = Suite(
    "test",
    ("null", BROKEN),
    (SceneSpec("hide_and_seek", "hide_and_seek"),),
    Workload(),
    MEASUREMENT,
    {"worlds": [1, 8], "resolution": [[16, 16]]},
)


class FakeMonitor:
    def __init__(self, interval, locked):
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def mark(self):
        return 0.0

    def window(self, start, end):
        self.calls += 1
        reasons = ["hw_thermal"] if self.calls == 2 else []
        return [{"gpu": 0, "energy_joules": 2.0, "throttle_reasons": reasons, "disturbed": bool(reasons)}]


def metric(table, name):
    return table.filter(pc.equal(table["metric"], name))["value"].to_pylist()


def run(tmp_path, suite=SUITE, isolation="process"):
    return read_run(run_suite(suite, tmp_path, log=lambda _: None, isolation=isolation))


def test_run_records_rows_outcomes_and_machine(tmp_path):
    manifest, table = run(tmp_path)

    assert manifest["status"] == "complete"
    assert {"cpu", "memory", "os", "gpus", "packages"} <= manifest["machine"].keys()
    assert manifest["tuning"]["process"]["affinity"]
    statuses = {(o["backend"], o["workload"]["worlds"]): o["status"] for o in manifest["outcomes"]}
    assert statuses[("null", 1)] == statuses[("null", 8)] == "ok"
    assert statuses[(BROKEN, 1)] == "error"

    repeats = sum(metric(table, "repeats"))
    assert len(metric(table, "update_seconds")) == 2 + repeats * 4
    assert set(table["backend"].to_pylist()) == {"null"}
    assert set(table["phase"].to_pylist()) == {"startup", "load", "first_frame", "warmup", "steady", "result"}


def test_repeats_stop_at_the_minimum_once_the_ci_is_tight(tmp_path):
    _, table = run(tmp_path, replace(SUITE, backends=("null",), measurement=replace(MEASUREMENT, target_ci=math.inf)))
    assert metric(table, "repeats") == [3, 3]


def test_repeats_run_to_the_cap_when_the_ci_never_tightens(tmp_path):
    _, table = run(tmp_path, replace(SUITE, backends=("null",), measurement=replace(MEASUREMENT, target_ci=0.0)))
    assert metric(table, "repeats") == [5, 5]


def test_disturbed_repeats_do_not_count(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "Monitor", FakeMonitor)
    manifest, table = run(tmp_path, replace(SUITE, backends=("null",), sweep={}), "inline")
    assert metric(table, "repeats") == [4]
    assert metric(table, "clean_repeats") == [3]
    assert metric(table, "disturbed") == [1, 0, 0, 0]
    assert metric(table, "gpu0.throttle.hw_thermal") == [1]


def test_backends_without_a_stream_report_no_device_time(tmp_path):
    manifest, table = run(tmp_path, replace(SUITE, backends=("null",), sweep={}), "inline")
    assert metric(table, "render_device_seconds") == []
    assert manifest["backends"]["null"]["details"]["device_timer"] == "none"


def test_sm_limits_need_mps_and_a_cuda_backend(tmp_path, monkeypatch):
    sweep = {"sm_percent": [50]}
    with pytest.raises(ValueError, match="MPS"):
        run(tmp_path, replace(SUITE, backends=("null",), sweep=sweep), "inline")
    monkeypatch.setattr(harness.tuning, "mps_running", lambda: False)
    manifest, _ = run(tmp_path / "off", replace(SUITE, backends=("null",), sweep=sweep))
    assert [o["reason"] for o in manifest["outcomes"]] == ["MPS is not running, run: sudo scripts/system-prep apply --mps"]
    assert Capabilities(frozenset({"unlit"}), frozenset({"rgb"})).unsupported(Workload(sm_percent=50)) == ["MPS cannot limit its SMs"]
    assert Capabilities(frozenset({"unlit"}), frozenset({"rgb"}), sm_limit=True).unsupported(Workload(sm_percent=50)) == []


def test_unsupported_workloads_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(NullBackend, "capabilities", lambda self: Capabilities(frozenset({"depth"}), frozenset({"depth"})))
    manifest, table = run(tmp_path, replace(SUITE, backends=("null",), sweep={}), "inline")
    assert manifest["outcomes"][0]["status"] == "unsupported"
    assert len(table) == 0


def test_max_repeats_below_repeats_is_rejected():
    with pytest.raises(ValueError, match="max_repeats"):
        Measurement(repeats=5, max_repeats=2)


def test_cli_scene_summary_and_system(tmp_path, capsys, monkeypatch):
    main(["scene", "hide_and_seek", "--worlds", "4", "--cameras", "2"])
    assert "instances  76" in capsys.readouterr().out

    monkeypatch.setattr(harness, "Monitor", FakeMonitor)
    directory = run_suite(replace(SUITE, sweep={}), tmp_path, log=lambda _: None, isolation="inline")
    main(["summary", str(directory)])
    out = capsys.readouterr().out
    assert "null" in out
    assert " 3/4 " in out
    assert f"error: {BROKEN}" in out

    main(["system"])
    assert '"machine"' in capsys.readouterr().out

    report = tmp_path / "report.html"
    main(["report", str(directory), "-o", str(report)])
    html = report.read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert "images per second by worlds" in html
    assert "time to the first image" in html
