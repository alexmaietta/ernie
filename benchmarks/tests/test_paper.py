# SPDX-License-Identifier: BSD-3-Clause
import stat
from dataclasses import replace

import pytest

from ernie_bench import paper
from ernie_bench.analysis import summarize_run
from ernie_bench.results import read_run

FAKE = """#!/usr/bin/env bash
worlds=$1 steps=$2 mode=$3
case "$FAKE_BEHAVIOUR" in
    crash) echo "boom" >&2; exit 3 ;;
    hang) sleep 60 ;;
    silent) echo "nothing useful" ;;
    *) echo "FPS 1234.5"; if [[ $mode == rt ]]; then echo "Average step time: 20.000000 ms"; else echo "Average step time: 5.000000 ms"; fi ;;
esac
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    root = tmp_path / "madrona"
    program = root / "build" / "hideseek_headless"
    program.parent.mkdir(parents=True)
    program.write_text(FAKE)
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(paper, "MADRONA", root)
    monkeypatch.setattr(paper, "CACHE", tmp_path / "cache")
    return tmp_path / "runs"


GRID = paper.PaperGrid(worlds=(4, 16), resolutions=(32,), steps=10, processes=2, timeout_seconds=3)


def test_paper_runs_record_step_times_per_fresh_process(fake):
    directory = paper.run_paper(GRID, fake, log=lambda _: None)
    manifest, _ = read_run(directory)
    assert len(manifest["outcomes"]) == 2 * 2 * 2
    assert {o["status"] for o in manifest["outcomes"]} == {"ok"}
    _, records = summarize_run(directory)
    medians = {(r["backend"], r["worlds"]): r["median_seconds"] for r in records}
    assert medians == {
        ("madrona_rast_paper", 4): pytest.approx(0.005),
        ("madrona_rast_paper", 16): pytest.approx(0.005),
        ("madrona_rt_paper", 4): pytest.approx(0.020),
        ("madrona_rt_paper", 16): pytest.approx(0.020),
    }
    for record in records:
        assert record["cameras_per_world"] == 6 and record["processes"] == 2
        assert record["images_per_second"] == pytest.approx(record["worlds"] * 6 / record["median_seconds"])
        assert record["tier"] == ("direct" if record["backend"] == "madrona_rt_paper" else "unlit")


@pytest.mark.parametrize("behaviour, status", [("crash", "error"), ("silent", "error"), ("hang", "timeout")])
def test_paper_runs_contain_failures(fake, monkeypatch, behaviour, status):
    monkeypatch.setenv("FAKE_BEHAVIOUR", behaviour)
    directory = paper.run_paper(replace(GRID, worlds=(4,), modes=("rt",), processes=1), fake, log=lambda _: None)
    manifest, table = read_run(directory)
    outcome, = manifest["outcomes"]
    assert outcome["status"] == status
    assert len(table) == 0


def test_paper_grid_rejects_unknown_scenes_and_modes():
    with pytest.raises(ValueError, match="scene"):
        paper.PaperGrid(scene="procthor")
    with pytest.raises(ValueError, match="modes"):
        paper.PaperGrid(modes=("pathtrace",))
