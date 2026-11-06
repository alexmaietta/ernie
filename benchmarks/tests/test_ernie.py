# SPDX-License-Identifier: BSD-3-Clause
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import SPEC

from ernie_bench import harness
from ernie_bench.analysis import summarize_run
from ernie_bench.results import read_run
from ernie_bench.config import Capture, Measurement, Suite, Workload

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "envs" / "ernie" / ".venv" / "bin" / "python"
IMPORTS = PYTHON.exists() and subprocess.run([str(PYTHON), "-c", "import ernie"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not IMPORTS, reason="envs/ernie is missing or ernie-renderer does not import")
MEASUREMENT = Measurement(warmup_min_iterations=3, warmup_min_seconds=0, warmup_max_seconds=0.5, warmup_window=3, warmup_cv=0.5, iterations=10, repeats=2, max_repeats=2)


@pytest.mark.parametrize("backend", ["ernie_hardware", "ernie_software"])
@pytest.mark.parametrize("tier", ["unlit", "direct", "pathtraced"])
def test_ernie_matches_the_reference_ray_caster(backend, tier):
    result = subprocess.run(
        [sys.executable, "-m", "ernie_bench.cli", "check", backend, "--tier", tier, "--worlds", "3", "--cameras", "2", "--images", "6"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout
    if tier in ("unlit", "direct"):
        assert '"rgb_agreement"' in result.stdout


def test_ernie_runs_through_workers_with_captures(tmp_path):
    workload = Workload(worlds=4, cameras_per_world=2, width=32, height=24, outputs=("rgb", "depth", "segmentation"), tier="direct")
    suite = Suite("ernie", ("ernie",), (SPEC,), workload, MEASUREMENT, capture=Capture(steps=(0, 15), images=(0, 7)))
    directory = harness.run_suite(suite, tmp_path, log=lambda _: None)
    manifest, _ = read_run(directory)
    outcome, = manifest["outcomes"]
    assert outcome["status"] == "ok", outcome
    assert manifest["backends"]["ernie"]["details"]["traversal"] in ("HARDWARE", "SOFTWARE")
    _, (record,) = summarize_run(directory)
    assert record["median_seconds"] > 0 and record["startup"]["create_seconds"] > 0
    saved = np.load(Path(outcome["captures"]) / "step-00015.npz")
    assert saved["rgb"].shape == (2, 24, 32, 3) and np.isfinite(saved["rgb"]).all() and saved["rgb"].max() > 0
    assert np.isfinite(saved["depth"]).any()
    assert set(np.unique(saved["segmentation"])) - {0} <= set(range(1, 7))


@pytest.mark.parametrize("backend, traversal", [("ernie_hardware", "HARDWARE"), ("ernie_software", "SOFTWARE")])
def test_ernie_modes_are_recorded(tmp_path, backend, traversal):
    workload = Workload(worlds=2, width=16, height=16, tier="unlit")
    directory = harness.run_suite(Suite("mode", (backend,), (SPEC,), workload, MEASUREMENT), tmp_path, log=lambda _: None)
    manifest, _ = read_run(directory)
    assert manifest["outcomes"][0]["status"] == "ok"
    details = manifest["backends"][backend]["details"]
    assert details["traversal"] == traversal and details["sampling"] == traversal


def test_ernie_refuses_a_mode_the_device_lacks(tmp_path):
    workload = Workload(worlds=1, width=16, height=16, tier="unlit")
    directory = harness.run_suite(Suite("mode", ("ernie_cuda_hardware",), (SPEC,), workload, MEASUREMENT), tmp_path, log=lambda _: None)
    manifest, _ = read_run(directory)
    assert manifest["outcomes"][0]["status"] == "error"
    assert "hardware ray traversal" in manifest["outcomes"][0]["reason"]
