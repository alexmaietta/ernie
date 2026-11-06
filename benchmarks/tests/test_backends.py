# SPDX-License-Identifier: BSD-3-Clause
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import SPEC

from ernie_bench import harness
from ernie_bench.backends.base import class_labels
from ernie_bench.results import read_run
from ernie_bench.config import Capture, Measurement, Suite, Workload

ROOT = Path(__file__).resolve().parents[1]
HALF_PIXEL_OFFSET = {"madrona_rast"}
COLOR_MATCHES = {("newton", "unlit"), ("newton", "direct"), ("madrona_rast", "unlit")}
MEASUREMENT = Measurement(warmup_min_iterations=3, warmup_min_seconds=0, warmup_max_seconds=2.0, warmup_window=3, warmup_cv=0.5, iterations=10, repeats=2, max_repeats=2)
TIERS = {
    "newton": ["unlit", "direct"],
    "ovrtx": ["direct"],
    "ovrtx_tiled": ["direct"],
    "madrona_rt": ["direct"],
    "madrona_rast": ["unlit"],
}


def id_map(entries):
    records, text = b"", b""
    offset = 24 * len(entries)
    for semantic_id, label in entries:
        encoded = label.encode()
        records += struct.pack("<6I", semantic_id, 0, 0, 0, len(encoded), offset + len(text))
        text += encoded
    return np.frombuffer(records + text + struct.pack("<I", len(entries)), np.uint8)


@pytest.mark.parametrize(
    "entries",
    [
        [(0, "BACKGROUND"), (1, "UNLABELLED"), (2, "class: wall;"), (3, "class: box;")],
        [(2, "class: wall; "), (3, "class: box; "), (7, "class: hider; ")],
    ],
    ids=["ernie", "ovrtx"],
)
def test_semantic_id_maps_decode_to_class_indices(entries):
    labels = class_labels(id_map(entries), np.array([0, 1, 2, 3, 7, 99]), ("floor", "wall", "box", "ramp", "hider"))
    assert labels[2] == 2 and labels[3] == 3
    assert labels[0] == labels[1] == labels[5] == 0
    assert labels[4] == (5 if (7, "class: hider; ") in entries else 0)


@pytest.mark.parametrize("backend, tier", [(backend, tier) for backend, tiers in TIERS.items() for tier in tiers])
def test_backend_matches_the_reference_ray_caster(backend, tier):
    if not (ROOT / "envs" / backend.split("_")[0] / ".venv").exists():
        pytest.skip(f"no envs/{backend.split('_')[0]}")
    result = subprocess.run(
        [sys.executable, "-m", "ernie_bench.cli", "check", backend, "--tier", tier, "--worlds", "3", "--cameras", "2", "--width", "64", "--height", "64", "--images", "6"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    scores = json.loads(result.stdout[result.stdout.index("{") : result.stdout.rindex("}") + 1])
    assert scores["hit_agreement"] >= 0.98, scores
    if backend in HALF_PIXEL_OFFSET:
        assert scores["depth_median_error"] <= 0.01, scores
    else:
        assert scores["depth_agreement"] >= 0.98, scores
    assert scores.get("segmentation_agreement", 1.0) >= 0.97, scores
    if (backend, tier) in COLOR_MATCHES:
        assert scores["rgb_agreement"] >= 0.95, scores


@pytest.mark.parametrize("backend", sorted(TIERS))
def test_backend_runs_through_workers_with_captures(tmp_path, backend):
    if not (ROOT / "envs" / backend.split("_")[0] / ".venv").exists():
        pytest.skip(f"no envs/{backend.split('_')[0]}")
    madrona = backend.startswith("madrona")
    outputs = ("rgb", "depth") if madrona else ("rgb", "depth", "segmentation")
    workload = Workload(worlds=3, cameras_per_world=2, width=32, height=32 if madrona else 24, outputs=outputs, tier=TIERS[backend][-1])
    suite = Suite("backend", (backend,), (SPEC,), workload, MEASUREMENT, capture=Capture(steps=(0, 15), images=(0, 5)))
    directory = harness.run_suite(suite, tmp_path, log=lambda _: None)
    manifest, _ = read_run(directory)
    outcome, = manifest["outcomes"]
    assert outcome["status"] == "ok", outcome
    saved = np.load(Path(outcome["captures"]) / "step-00015.npz")
    assert saved["rgb"].shape == (2, workload.height, 32, 3) and np.isfinite(saved["rgb"]).all() and saved["rgb"].max() > 0
    assert np.isfinite(saved["depth"]).any()
    if "segmentation" in outputs:
        assert set(np.unique(saved["segmentation"])) - {0} <= set(range(1, 7))
