# SPDX-License-Identifier: BSD-3-Clause
import os
import subprocess
import sys
import textwrap

import pytest

import ernie
import ernie.demo
from conftest import REQUIRE_VALIDATION

DEMOS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "demos",
                                      "python"))


def demo_files():
    if not os.path.isdir(DEMOS):
        return []
    return [entry.path for entry in ernie.demo.discover_demos(DEMOS)]


@pytest.mark.parametrize("path", demo_files(), ids=os.path.basename)
def test_demo_runs_its_checks_without_a_viewer(path, renderer):
    command = [sys.executable, path, "--viewer", "null", "--test"]
    if REQUIRE_VALIDATION:
        command.append("--validation")
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(ernie.__file__)), os.environ.get("PYTHONPATH", "")]))
    result = subprocess.run(command, cwd=DEMOS, env=environment, capture_output=True, text=True,
                            timeout=600)
    if result.returncode != 0 and "No module named 'warp'" in result.stderr:
        pytest.skip("needs Warp")
    if result.returncode != 0 and "needs CUDA interop" in result.stderr:
        pytest.skip("needs CUDA interop")
    assert result.returncode == 0, result.stdout + result.stderr


def test_parser_has_the_common_options():
    args = ernie.demo.create_parser().parse_args(["--benchmark", "--num-frames", "7"])
    assert args.benchmark is None and args.num_frames == 7
    assert args.viewer == "window" and not args.headless and not args.test
    assert ernie.demo.create_parser().parse_args(["--benchmark", "2.5"]).benchmark == 2.5


def test_entries_take_category_and_name_from_the_file_name(tmp_path):
    (tmp_path / "demo_lighting_dome_daylight.py").write_text("class Demo(object):\n    pass\n")
    (tmp_path / "demo_support.py").write_text("")
    (tmp_path / "demo_other.py").write_text("print('no Demo class')\n")
    entries = ernie.demo.discover_demos(str(tmp_path))
    assert [entry.label for entry in entries] == ["lighting: dome daylight"]


DEMO_SOURCE = """\
import ernie
import ernie.demo


class Demo(ernie.demo.Demo):
    num_frames = 2

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.scene.define("/World/{name}", ernie.Cube)
        self.scene.overlay("{name}").label((0, 0, 0), "{name}")
        if viewer is not None:
            viewer.add_panel("{name} panel")
            viewer.on_frame(lambda v, info: None)
            viewer.lighting = ernie.app.Lighting.CAMERA
"""


def test_browser_switches_demos_and_removes_what_the_last_one_added(tmp_path):
    app = pytest.importorskip("ernie.app", reason="Ernie was built without its app",
                              exc_type=ImportError)
    for name in ("A", "B"):
        source = textwrap.dedent(DEMO_SOURCE).format(name=name)
        (tmp_path / f"demo_test_{name.lower()}.py").write_text(source)
    try:
        raw = app.Viewer(ernie.Scene(), hidden=True, width=480, height=320)
    except RuntimeError as error:
        pytest.skip(f"No Vulkan device: {error}")
    viewer = ernie.demo.DemoViewer(raw)
    args = ernie.demo.create_parser().parse_args(["--headless"])
    demo_a = ernie.demo._load_demo_class(str(tmp_path / "demo_test_a.py"))(viewer, args)
    viewer._adopt(demo_a)
    assert raw.lighting == app.Lighting.CAMERA  # re-applied after the session reset
    browser = ernie.demo.DemoBrowser(viewer, demo_a, args)
    assert [entry.name for entry in browser.entries] == ["a", "b"]

    browser._choose(1)
    demo_b, paused = browser.update(demo_a, False)
    assert type(demo_b).__module__.endswith("demo_test_b") and not paused
    scene = raw.scene
    assert scene.has("/World/B") and not scene.has("/World/A")
    assert scene.has(app.Viewer.perspective_camera)
    assert "A" not in scene.overlay_names() and "B" in scene.overlay_names()
    with pytest.raises(ValueError):
        raw.panel_visible("A panel")
    assert raw.panel_visible("B panel") and raw.panel_visible("Demos")
    raw.step()

    browser.reset_requested = True
    demo_b2, _ = browser.update(demo_b, False)
    assert demo_b2 is not demo_b and demo_b2.frame == 0 and scene.has("/World/B")
    viewer._release()
    raw.remove_panel("Demos")
    raw.close()
