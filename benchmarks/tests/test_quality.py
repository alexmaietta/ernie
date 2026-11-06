# SPDX-License-Identifier: BSD-3-Clause
import contextlib
import types

import numpy as np
import pytest

from ernie_bench import quality


def image(seed=0, shape=(24, 32, 3)):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def test_identical_colors_score_perfectly():
    expected = image()
    scores = quality.color_metrics(expected, expected)
    assert scores["psnr"] == float("inf")
    assert scores["ssim"] == pytest.approx(1.0)
    assert scores["mean_absolute_error"] == 0
    assert scores["flip"] is None or scores["flip"] == pytest.approx(0.0, abs=1e-6)


def test_lpips_compares_srgb_images_scaled_to_plus_minus_one(monkeypatch):
    seen = []

    def network(test, expected):
        seen.append((test, expected))
        return np.abs(test - expected)

    monkeypatch.setitem(__import__("sys").modules, "torch", types.SimpleNamespace(from_numpy=lambda array: array, no_grad=contextlib.nullcontext))
    monkeypatch.setattr(quality, "lpips_network", lambda: network)
    expected = image()
    assert quality.color_metrics(expected, expected)["lpips"] == 0.0
    test, _ = seen[0]
    assert test.shape == (1, 3, 24, 32)
    assert -1 <= test.min() and test.max() <= 1


def test_color_metrics_get_worse_with_more_error():
    expected = image()
    mild = quality.color_metrics(np.clip(expected + 0.01, 0, 1), expected)
    harsh = quality.color_metrics(np.clip(expected + 0.2, 0, 1), expected)
    assert mild["psnr"] > harsh["psnr"]
    assert mild["ssim"] > harsh["ssim"]
    assert mild["mean_absolute_error"] < harsh["mean_absolute_error"]
    assert quality.color_metrics(expected * 0 + 0.5, expected * 0 + 0.6)["psnr"] == pytest.approx(20.0, abs=1e-4)


def test_depth_metrics():
    expected = np.array([[[1.0], [2.0], [np.inf], [4.0]]])
    test = np.array([[[1.1], [2.0], [np.inf], [np.inf]]])
    scores = quality.depth_metrics(test, expected)
    assert scores["hit_agreement"] == 0.75
    assert scores["absolute_relative"] == pytest.approx(0.05)
    assert scores["rmse"] == pytest.approx(np.sqrt(0.01 / 2))


def test_segmentation_mean_iou():
    expected = np.array([[[1], [1], [2], [2]]])
    assert quality.segmentation_metrics(expected, expected)["mean_iou"] == 1.0
    assert quality.segmentation_metrics(np.array([[[1], [2], [2], [2]]]), expected)["mean_iou"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_quality_against_a_reference_backend_in_the_same_run(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    from conftest import SPEC

    from ernie_bench import harness
    from ernie_bench.config import Capture, Measurement, Suite, Workload

    tests = Path(__file__).resolve().parent
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(tests), os.environ.get("PYTHONPATH")])))
    measurement = Measurement(warmup_min_iterations=2, warmup_min_seconds=0, warmup_max_seconds=0.1, warmup_window=2, warmup_cv=0.5, iterations=3, repeats=1, max_repeats=1)
    workload = Workload(worlds=2, cameras_per_world=2, width=16, height=12, outputs=("rgb", "depth", "segmentation"))
    suite = Suite("quality", ("worker_backends:Painter", "worker_backends:PainterCopy"), (SPEC,), workload, measurement, capture=Capture(steps=(0, 4), images=(0, 3)))
    directory = harness.run_suite(suite, tmp_path, log=lambda _: None)
    summary = quality.summarize(quality.evaluate([directory], "worker_backends:Painter"))
    by_output = {row["output"]: row for row in summary}
    assert set(by_output) == {"rgb", "depth", "segmentation"}
    assert all(row["backend"] == "worker_backends:PainterCopy" and row["samples"] == 4 for row in summary)
    assert by_output["rgb"]["psnr"] is None and by_output["rgb"]["ssim"] == pytest.approx(1.0)
    assert by_output["depth"]["absolute_relative"] == 0 and by_output["segmentation"]["mean_iou"] == 1.0
