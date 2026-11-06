# SPDX-License-Identifier: BSD-3-Clause
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest
from conftest import SPEC

from ernie_bench import harness, worker
from ernie_bench.analysis import summarize_run
from ernie_bench.results import read_run
from ernie_bench.config import Measurement, Suite, Workload

TESTS = Path(__file__).resolve().parent
MEASUREMENT = Measurement(
    warmup_min_iterations=3,
    warmup_min_seconds=0,
    warmup_max_seconds=0.2,
    warmup_window=3,
    warmup_cv=0.2,
    iterations=20,
    repeats=2,
    max_repeats=2,
    timeout_seconds=60,
)


@pytest.fixture(autouse=True)
def importable_test_backends(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(TESTS), os.environ.get("PYTHONPATH")])))


def suite(*backends, measurement=MEASUREMENT, **changes):
    return Suite("workers", backends, (SPEC,), Workload(width=8, height=8), measurement, **changes)


def run(tmp_path, test_suite):
    directory = harness.run_suite(test_suite, tmp_path, log=lambda _: None)
    return directory, *read_run(directory)


def outcomes(manifest, backend):
    return [o for o in manifest["outcomes"] if o["backend"] == backend]


def metric(table, name, backend=None):
    rows = table.filter(pc.equal(table["metric"], name))
    if backend:
        rows = rows.filter(pc.equal(rows["backend"], backend))
    return np.array(rows["value"].to_pylist())


def test_fresh_processes_per_point_with_startup_costs(tmp_path):
    directory, manifest, table = run(tmp_path, suite("null", sweep={"worlds": [1, 4]}, measurement=replace(MEASUREMENT, processes=2)))
    assert [(o["workload"]["worlds"], o["process"], o["status"]) for o in manifest["outcomes"]] == [
        (1, 0, "ok"), (1, 1, "ok"), (4, 0, "ok"), (4, 1, "ok")
    ]
    assert set(table["process"].to_pylist()) == {0, 1}
    assert all(Path(o["log"]).exists() and o["returncode"] == 0 for o in manifest["outcomes"])
    assert ((metric(table, "interpreter_seconds") > 0) & (metric(table, "interpreter_seconds") < 10)).all()
    assert (metric(table, "harness_import_seconds") > 0).all()
    assert len(metric(table, "create_seconds")) == 4

    _, records = summarize_run(directory)
    assert [r["processes"] for r in records] == [2, 2]
    assert all(r["repeats"] == 4 and r["process_spread"] >= 0 for r in records)
    assert all(r["startup"]["interpreter_seconds"] > 0 for r in records)


def test_worker_timing_is_as_exact_as_inline(tmp_path):
    _, _, table = run(tmp_path, suite("worker_backends:Spinner"))
    render = metric(table, "render_seconds")
    assert render.min() >= 0.002 - 1e-6
    assert render.min() <= 0.002 + 2e-4


def test_a_crashing_backend_is_contained(tmp_path):
    _, manifest, table = run(tmp_path, suite("worker_backends:Crasher", "null"))
    crashed, = outcomes(manifest, "worker_backends:Crasher")
    assert crashed["status"] == "error"
    assert crashed["returncode"] == -6
    assert "no result" in crashed["reason"]
    assert outcomes(manifest, "null")[0]["status"] == "ok"
    assert set(table["backend"].to_pylist()) == {"null"}


def test_a_hanging_backend_is_killed_at_its_timeout(tmp_path):
    started = time.perf_counter()
    _, manifest, _ = run(tmp_path, suite("worker_backends:Hanger", measurement=replace(MEASUREMENT, timeout_seconds=3)))
    assert time.perf_counter() - started < 15
    hung, = manifest["outcomes"]
    assert hung["status"] == "timeout"
    job = str(Path(hung["log"]).with_suffix(".job.json"))
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                assert job not in (entry / "cmdline").read_bytes().decode(errors="replace")
            except OSError:
                pass


def test_backend_output_goes_to_the_log_and_not_the_protocol(tmp_path):
    _, manifest, table = run(tmp_path, suite("worker_backends:Chatty"))
    chatty, = manifest["outcomes"]
    assert chatty["status"] == "ok"
    log = Path(chatty["log"]).read_text()
    assert "chatty stdout" in log and "chatty stderr" in log
    assert len(metric(table, "render_seconds")) == MEASUREMENT.repeats * MEASUREMENT.iterations + 1


def test_backend_construction_is_timed(tmp_path):
    _, _, table = run(tmp_path, suite("worker_backends:SlowStart"))
    created, = metric(table, "create_seconds")
    assert 0.3 <= created < 0.5


def test_backend_exceptions_come_back_with_a_traceback(tmp_path):
    _, manifest, _ = run(tmp_path, suite("worker_backends:Raiser"))
    raised, = manifest["outcomes"]
    assert raised["status"] == "error"
    assert raised["reason"] == "ValueError: bad scene"
    assert "worker_backends.py" in raised["traceback"]


def test_backend_environments_are_picked_by_name(tmp_path):
    python = tmp_path / "envs" / "madrona" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assert harness.environment("madrona", tmp_path) == str(python)
    assert harness.environment("newton", tmp_path) == sys.executable


def test_protocol_mismatch_is_reported(tmp_path):
    job = tmp_path / "job.json"
    result = tmp_path / "result.json"
    job.write_text(json.dumps({"protocol": harness.PROTOCOL + 1, "result": str(result)}))
    worker.main(str(job))
    answer = json.loads(result.read_text())
    assert answer["status"] == "error" and "protocol" in answer["reason"]


@pytest.mark.parametrize("caches", ["warm", "cold"])
def test_cold_caches_get_a_fresh_directory_that_is_removed(tmp_path, caches, monkeypatch):
    for name in ("XDG_CACHE_HOME", "WARP_CACHE_PATH", "__GL_SHADER_DISK_CACHE_PATH", "ERNIE_BENCH_CACHE"):
        monkeypatch.delenv(name, raising=False)
    directory, manifest, _ = run(tmp_path, suite("worker_backends:CacheReporter", measurement=replace(MEASUREMENT, caches=caches)))
    details = {name: value for name, value in manifest["backends"]["worker_backends:CacheReporter"]["details"].items() if name != "device_timer"}
    if caches == "warm":
        assert set(details.values()) == {None}
    else:
        roots = {Path(value).parent for value in details.values()}
        assert len(roots) == 1 and roots.pop().parent == directory / "workers"
        assert not list((directory / "workers").glob("*.cache"))


def test_cold_caches_refuse_inline_runs(tmp_path):
    with pytest.raises(ValueError, match="process isolation"):
        harness.run_suite(suite("null", measurement=replace(MEASUREMENT, caches="cold")), tmp_path, log=lambda _: None, isolation="inline")


def test_workers_run_inside_the_run_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory, manifest, _ = run(tmp_path / "runs", suite("worker_backends:Litterer"))
    assert manifest["outcomes"][0]["status"] == "ok"
    assert (directory / "workers" / "litter.txt").exists()
    assert not (tmp_path / "litter.txt").exists()
