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
    result = subprocess.run(command, cwd=DEMOS, env=demo_environment(), capture_output=True,
                            text=True, timeout=600)
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


def test_demos_package_lists_the_shipped_demos():
    result = subprocess.run([sys.executable, "-m", "ernie.demos", "--list"], capture_output=True,
                            text=True, timeout=120, env=demo_environment())
    assert result.returncode == 0, result.stderr
    names = result.stdout.split()
    assert "viewer_sun_panel" in names and "lighting_illuminants" in names


def demo_environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(ernie.__file__)), os.environ.get("PYTHONPATH", "")]))


DEMO_SOURCE = """\
import ernie
import ernie.demo


class Demo(ernie.demo.Demo):
    num_frames = 2

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.steps = 0
        self.on_step = None
        self.scene.define("/World/{name}", ernie.Cube)
        self.scene.define("/World/Camera", ernie.Camera)
        self.scene.overlay("{name}").label((0, 0, 0), "{name}")
        if viewer is not None:
            viewer.add_panel("{name} panel")
            viewer.on_frame(lambda v, info: None)
            viewer.lighting = ernie.app.Lighting.CAMERA

    def step(self):
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self)

    def test_final(self):
        assert self.steps == self.frame
"""


def write_demos(directory):
    for name in ("A", "B"):
        source = textwrap.dedent(DEMO_SOURCE).format(name=name)
        (directory / f"demo_test_{name.lower()}.py").write_text(source)


def test_python_demo_runs_on_the_cpp_runner_without_a_viewer(tmp_path, renderer):
    write_demos(tmp_path)
    demo_class = ernie.demo.load_demo_class(str(tmp_path / "demo_test_a.py"))
    args = demo_class.create_parser().parse_args(["--viewer", "null", "--test",
                                                  "--num-frames", "4"])
    demo = demo_class(None, args)
    assert demo.num_frames == 2 and demo.frame == 0
    assert ernie.demo.run(demo, args) is demo
    assert demo.steps == 4 and demo.frame == 4
    assert demo.image is not None and demo.image.shape[-1] == 4


def test_a_test_run_of_a_demo_without_checks_fails(renderer):
    class Unchecked(ernie.demo.Demo):
        num_frames = 1

        def __init__(self, viewer, args):
            super().__init__(viewer, args)
            self.scene.define("/World/Camera", ernie.Camera)

    args = Unchecked.create_parser().parse_args(["--viewer", "null", "--test"])
    with pytest.raises(Exception, match="no checks"):
        ernie.demo.run(Unchecked(None, args), args)


def test_switching_demos_removes_what_the_last_one_added(tmp_path):
    app = pytest.importorskip("ernie.app", reason="Ernie was built without its app",
                              exc_type=ImportError)
    write_demos(tmp_path)
    args = ernie.demo.create_parser().parse_args(["--headless", "--num-frames", "2"])
    try:
        runner = app.DemoRunner(ernie.demo.options_from_args(args))
    except RuntimeError as error:
        pytest.skip(f"No Vulkan device: {error}")
    demo_class = ernie.demo.load_demo_class(str(tmp_path / "demo_test_a.py"))
    infos, current = ernie.demo.catalog(demo_class, args)
    assert [info.label for info in infos] == ["test: a", "test: b"] and current == 0
    runner.set_catalog(infos, current)
    viewer = runner.viewer
    demo_a = demo_class(viewer, args)
    demo_a.on_step = lambda demo: runner.request_switch(1) if demo.frame == 1 else None

    last = runner.run(demo_a)
    assert type(last).__module__.endswith("demo_test_b") and last.frame == 2
    scene = viewer.scene
    assert scene.has("/World/B") and not scene.has("/World/A")
    assert scene.has(app.Viewer.perspective_camera)
    assert "A" not in scene.overlay_names() and "B" in scene.overlay_names()
    with pytest.raises(ValueError):
        viewer.panel_visible("A panel")
    assert viewer.lighting == app.Lighting.CAMERA  # B's own choice, kept over the reset
