# SPDX-License-Identifier: BSD-3-Clause
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest
from conftest import SPEC
from worker_backends import paint

from ernie_bench import harness, media
from ernie_bench.cli import main
from ernie_bench.results import read_run
from ernie_bench.config import Capture, Measurement, Suite, Workload

TESTS = Path(__file__).resolve().parent
MEASUREMENT = Measurement(
    warmup_min_iterations=3, warmup_min_seconds=0, warmup_max_seconds=0.1, warmup_window=3, warmup_cv=0.5, iterations=5, repeats=1, max_repeats=1
)
WORKLOAD = Workload(worlds=3, cameras_per_world=2, width=24, height=16, outputs=("rgb", "depth", "segmentation"))


@pytest.fixture(autouse=True)
def importable_test_backends(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(TESTS), os.environ.get("PYTHONPATH")])))


def run(tmp_path, *backends, capture, isolation="process", **changes):
    measurement = changes.pop("measurement", MEASUREMENT)
    suite = Suite("media", backends, (SPEC,), WORKLOAD, measurement, capture=capture, **changes)
    directory = harness.run_suite(suite, tmp_path, log=lambda _: None, isolation=isolation)
    return directory, *read_run(directory)


@pytest.mark.parametrize("isolation", ["process", "inline"])
def test_capture_saves_exactly_the_requested_images_and_steps(tmp_path, isolation):
    capture = Capture(steps=(0, 7, 30), images=(1, 4), outputs=("rgb", "segmentation"))
    _, manifest, table = run(tmp_path, "worker_backends:Painter", capture=capture, isolation=isolation)
    outcome, = manifest["outcomes"]
    directory = Path(outcome["captures"])
    assert json.loads((directory / "index.json").read_text()) == {"steps": [0, 7, 30], "images": [1, 4], "outputs": ["rgb", "segmentation"]}
    for step in (0, 7, 30):
        saved = np.load(directory / f"step-{step:05d}.npz")
        expected = paint(WORKLOAD, step * WORKLOAD.dt)
        assert sorted(saved.files) == ["rgb", "segmentation"]
        for output in saved.files:
            assert np.array_equal(saved[output], expected[output][[1, 4]])
    rendered = table.filter(pc.equal(table["metric"], "render_seconds"))
    assert len(rendered) == 1 + MEASUREMENT.repeats * MEASUREMENT.iterations


def test_capture_all_images_of_a_sequence(tmp_path):
    _, manifest, _ = run(tmp_path, "worker_backends:Painter", capture=Capture(frames=4, images="all"))
    directory = Path(manifest["outcomes"][0]["captures"])
    assert sorted(path.name for path in directory.glob("*.npz")) == [f"step-{step:05d}.npz" for step in range(4)]
    assert np.load(directory / "step-00003.npz")["rgb"].shape == (WORKLOAD.image_count, 16, 24, 3)


def test_only_the_first_process_captures(tmp_path):
    _, manifest, _ = run(tmp_path, "worker_backends:Painter", capture=Capture(), sweep={}, measurement=replace(MEASUREMENT, processes=2))
    assert [bool(o.get("captures")) for o in manifest["outcomes"]] == [True, False]


def test_malformed_readback_is_a_backend_error(tmp_path):
    _, manifest, _ = run(tmp_path, "worker_backends:Squashed", capture=Capture())
    outcome, = manifest["outcomes"]
    assert outcome["status"] == "error"
    assert "readback is float32[6, 8, 24, 3], expected float32[6, 16, 24, 3]" in outcome["reason"]


def test_rgb_display_uses_srgb_and_auto_exposure():
    linear = np.zeros((1, 2, 2, 3), np.float32)
    linear[0, 0, 0] = 0.5
    linear[0, 1, 1] = 1.0
    shown = media.displayable("rgb", linear, exposure=1.0)
    assert shown[0, 0, 0].tolist() == [188, 188, 188]
    assert shown[0, 1, 1].tolist() == [255, 255, 255]
    assert shown[0, 0, 1].tolist() == [0, 0, 0]
    bright = media.displayable("rgb", linear * 40)
    assert np.array_equal(bright, media.displayable("rgb", linear))


def test_depth_display_is_brighter_when_nearer_and_black_when_invalid():
    depth = np.array([[[[1.0], [5.0], [np.inf], [0.0]]]], np.float32)
    shown = media.displayable("depth", depth)[0, 0, :, 0]
    assert shown[0] > shown[1] > 0
    assert shown[2] == shown[3] == 0


def test_segmentation_display_separates_ids_and_blanks_background():
    ids = np.array([[[[0], [1], [2], [1]]]], np.int32)
    shown = media.displayable("segmentation", ids)[0, 0]
    assert shown[0].tolist() == [0, 0, 0]
    assert shown[1].tolist() == shown[3].tolist() != shown[2].tolist()


def test_grid_places_labelled_tiles_under_a_title():
    red = np.full((40, 60, 3), (255, 0, 0), np.uint8)
    blue = np.full((40, 60, 3), (0, 0, 255), np.uint8)
    plain = media.grid([red, blue], gap=4)
    assert plain.shape == (40 + 8, 2 * 60 + 12, 3)
    assert plain[4, 4].tolist() == [255, 0, 0] and plain[4, 68].tolist() == [0, 0, 255]
    titled = media.grid([red, blue], ["ernie", "madrona"], "a title", gap=4)
    assert titled.shape[0] > plain.shape[0]
    assert titled.shape[1] == plain.shape[1]
    assert (titled[: titled.shape[0] - plain.shape[0]] != media.BACKGROUND).any()


def test_movies_round_trip_through_ffmpeg(tmp_path):
    frames = [np.full((30, 41, 3), value, np.uint8) for value in (0, 128, 255)]
    path = media.write_movie(tmp_path / "ramp.mp4", frames, fps=10)
    decoded = list(media.read_movie(path))
    assert len(decoded) == 3
    assert decoded[0].shape == (30, 42, 3)
    assert [round(float(frame[:, :41].mean())) for frame in decoded] == pytest.approx([0, 128, 255], abs=3)


def test_cli_exports_images_movies_and_backend_panels(tmp_path, capsys):
    directory, manifest, _ = run(tmp_path, "worker_backends:Painter", "null", capture=Capture(frames=3, images=(0, 5), outputs=("rgb", "depth")))
    main(["images", str(directory)])
    pngs = list((directory / "media" / "images").rglob("*.png"))
    assert len(pngs) == 2 * 3 * 2 * 2
    factor = media.MIN_TILE // 24
    assert media.read_image(pngs[0]).shape[:2] == (16 * factor, 24 * factor)

    main(["movie", str(directory)])
    assert len(list((directory / "media" / "movies").glob("*.mp4"))) == 2 * 2 * 2

    main(["panel", str(directory), "--output", "rgb", "--image", "5"])
    panels = sorted((directory / "media" / "panels").glob("*.png"))
    assert len(panels) == 3
    panel = media.read_image(panels[0])
    assert panel.shape[1] > 2 * 24 * factor

    main(["panel", str(directory), "--output", "depth", "--movie"])
    assert len(list((directory / "media" / "panels").glob("*depth*.mp4"))) == 2
    capsys.readouterr()


def test_montage_merges_images_folders_and_videos(tmp_path):
    still = media.save_image(tmp_path / "still.png", np.full((20, 30, 3), 200, np.uint8))
    folder = tmp_path / "frames"
    for step in range(4):
        media.save_image(folder / f"{step:03d}.png", np.full((10, 10, 3), step * 60, np.uint8))
    clip = media.write_movie(tmp_path / "clip.mp4", [np.full((16, 16, 3), 90, np.uint8)] * 2, fps=5)

    main(["montage", str(still), str(folder), str(clip), "--label", "rtx", "--label", "ernie", "--label", "newton", "--title", "compare", "-o", str(tmp_path / "out.mp4"), "--fps", "5"])
    merged = list(media.read_movie(tmp_path / "out.mp4"))
    assert len(merged) == 4

    main(["montage", str(still), str(still), "-o", str(tmp_path / "pair.png")])
    pair = media.read_image(tmp_path / "pair.png")
    assert pair.shape[1] > 2 * pair.shape[0] - 100
