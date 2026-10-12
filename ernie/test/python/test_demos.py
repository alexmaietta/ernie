# SPDX-License-Identifier: BSD-3-Clause
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from conftest import REQUIRE_VALIDATION

import ernie
import ernie.demo

#: The repository's demos:
#: demos/<category>/demo_<category>_<name>.{cpp,py}.
DEMOS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "demos")
)

#: Categories with Python demos only, which show Python's GPU interop.
PYTHON_ONLY = {"interop"}

#: Python demos whose C++ twin is still to be written; empty once the
#: demos are all ported.
AWAITING_TWINS = set()

#: Smaller batches than the demos' benchmark sizes, which software
#: Vulkan in CI renders slowly.
SMALL_BATCHES = {
    "demo_batched_cameras": ["--views", "64"],
    "demo_interop_robot_arms": ["--worlds", "16"],
    "demo_batched_gpu_poses": ["--views", "32"],
}


def python_demos():
    return (
        ernie.demo.discover_demos(DEMOS)
        if os.path.isdir(DEMOS)
        else []
    )


def cpp_demos():
    app = getattr(ernie._ernie, "app", None)
    return (
        [info.id for info in app.registered_demos()]
        if app is not None
        else []
    )


def demo_environment():
    return dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                os.path.dirname(os.path.dirname(ernie.__file__)),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    )


def run(command, cwd=None):
    if REQUIRE_VALIDATION:
        command = [*command, "--validation"]
    result = subprocess.run(
        command,
        cwd=cwd,
        env=demo_environment(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if (
        result.returncode != 0
        and "No module named 'warp'" in result.stderr
    ):
        pytest.skip("needs Warp")
    if (
        result.returncode != 0
        and "needs CUDA interop" in result.stderr
    ):
        pytest.skip("needs CUDA interop")
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def demo_ids():
    """Every demo, in either language, or only ERNIE_TEST_DEMO's when
    it is set (ctest runs each demo as a test of its own)."""
    ids = sorted(
        set(cpp_demos()) | {entry.id for entry in python_demos()}
    )
    only = os.environ.get("ERNIE_TEST_DEMO")
    return [demo for demo in ids if only in (None, "", demo)]


@pytest.mark.parametrize("demo", demo_ids())
def test_demo_passes_its_checks_and_twins_render_the_same(
    demo, renderer, tmp_path
):
    """Each language's demo runs once without a viewer: its checks
    (--test), and its last frame written to compare with its twin's."""
    common = [
        "--viewer",
        "null",
        "--test",
        *SMALL_BATCHES.get(demo, []),
    ]
    outputs = []
    if demo in cpp_demos():
        output = tmp_path / "cpp.png"
        run(
            [
                sys.executable,
                "-m",
                "ernie",
                "--demo",
                demo,
                *common,
                "--output",
                str(output),
            ]
        )
        outputs.append(output)
    entry = next((e for e in python_demos() if e.id == demo), None)
    if entry is not None:
        output = tmp_path / "python.png"
        run(
            [
                sys.executable,
                entry.path,
                *common,
                "--output",
                str(output),
            ],
            cwd=os.path.dirname(entry.path),
        )
        outputs.append(output)
    if len(outputs) < 2:
        return
    pillow = pytest.importorskip("PIL.Image")
    cpp, python = (
        np.asarray(pillow.open(path))[..., :3].astype(int)
        for path in outputs
    )
    assert cpp.shape == python.shape
    assert np.abs(cpp - python).mean() < 1.0, (
        "the twins should render the same image"
    )


def test_every_demo_has_a_twin_in_the_other_language():
    python = {
        entry.id
        for entry in python_demos()
        if entry.category not in PYTHON_ONLY
        and entry.id not in AWAITING_TWINS
    }
    cpp = set(cpp_demos())
    assert python == cpp, (
        f"Python only: {sorted(python - cpp)}; "
        f"C++ only: {sorted(cpp - python)}"
    )


def test_ernie_program_lists_its_demos_and_renders_a_scene_file(
    tmp_path, renderer
):
    result = run([sys.executable, "-m", "ernie", "--list"])
    assert result.stdout.split() == cpp_demos()
    scene = ernie.Scene()
    scene.define("/World/Ball", ernie.Sphere, radius=1.0)
    scene.define("/World/Sun", ernie.DistantLight)
    scene.define("/World/Camera", ernie.Camera).set(
        ernie.tokens.TRANSFORM,
        ernie.look_at_transform((0, -6, 2), (0, 0, 0)),
    )
    ernie.save_scene(scene, str(tmp_path / "ball.json"))
    run(
        [
            sys.executable,
            "-m",
            "ernie",
            str(tmp_path / "ball.json"),
            "--output",
            str(tmp_path / "ball.png"),
        ]
    )
    assert (tmp_path / "ball.png").stat().st_size > 100


def test_demos_package_lists_the_shipped_demos():
    result = run([sys.executable, "-m", "ernie.demos", "--list"])
    names = result.stdout.split()
    assert (
        "demo_viewer_custom_panel" in names
        and "demo_lighting_illuminants" in names
    )


def test_parser_has_the_common_options():
    args = ernie.demo.create_parser().parse_args(
        ["--benchmark", "--num-frames", "7"]
    )
    assert args.benchmark is None and args.num_frames == 7
    assert (
        args.viewer == "window"
        and not args.headless
        and not args.test
    )
    assert (
        ernie.demo.create_parser()
        .parse_args(["--benchmark", "2.5"])
        .benchmark
        == 2.5
    )


def test_entries_take_category_and_name_from_their_directory_and_file(
    tmp_path,
):
    (tmp_path / "lighting").mkdir()
    (
        tmp_path / "lighting" / "demo_lighting_dome_daylight.py"
    ).write_text("class Demo(object):\n    pass\n")
    # A file of the former form, `<name>.py`, takes the same kind of id.
    (tmp_path / "lighting" / "spot.py").write_text(
        "class Demo(object):\n    pass\n"
    )
    (tmp_path / "lighting" / "_helpers.py").write_text(
        "class Demo(object):\n    pass\n"
    )
    (tmp_path / "lighting" / "notes.py").write_text(
        "print('no Demo class')\n"
    )
    (tmp_path / "top_level.py").write_text(
        "class Demo(object):\n    pass\n"
    )
    entries = ernie.demo.discover_demos(str(tmp_path))
    assert [(entry.id, entry.label) for entry in entries] == [
        ("demo_lighting_dome_daylight", "lighting: dome daylight"),
        ("demo_lighting_spot", "lighting: spot"),
    ]


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


def write_demos(root):
    """Two demos, test/a.py and test/b.py, under `root`; returns
    test/a.py's path."""
    (root / "test").mkdir()
    for name in ("A", "B"):
        source = textwrap.dedent(DEMO_SOURCE).format(name=name)
        (root / "test" / f"{name.lower()}.py").write_text(source)
    return str(root / "test" / "a.py")


def test_python_demo_runs_on_the_cpp_runner_without_a_viewer(
    tmp_path, renderer
):
    demo_class = ernie.demo.load_demo_class(write_demos(tmp_path))
    args = demo_class.create_parser().parse_args(
        ["--viewer", "null", "--test", "--num-frames", "4"]
    )
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

    args = Unchecked.create_parser().parse_args(
        ["--viewer", "null", "--test"]
    )
    with pytest.raises(Exception, match="no checks"):
        ernie.demo.run(Unchecked(None, args), args)


def test_switching_demos_removes_what_the_last_one_added(tmp_path):
    app = pytest.importorskip(
        "ernie.app",
        reason="Ernie was built without its app",
        exc_type=ImportError,
    )
    path = write_demos(tmp_path)
    args = ernie.demo.create_parser().parse_args(
        ["--headless", "--num-frames", "2"]
    )
    try:
        runner = app.DemoRunner(ernie.demo.options_from_args(args))
    except RuntimeError as error:
        pytest.skip(f"No Vulkan device: {error}")
    demo_class = ernie.demo.load_demo_class(path)
    infos, current = ernie.demo.catalog(demo_class, args)
    assert [info.label for info in infos] == [
        "test: a",
        "test: b",
    ] and current == 0
    runner.set_catalog(infos, current)
    viewer = runner.viewer
    demo_a = demo_class(viewer, args)
    demo_a.on_step = lambda demo: (
        runner.request_switch(1) if demo.frame == 1 else None
    )

    last = runner.run(demo_a)
    assert (
        type(last).__module__.endswith("test_b") and last.frame == 2
    )
    scene = viewer.scene
    assert scene.has("/World/B") and not scene.has("/World/A")
    assert scene.has(app.Viewer.perspective_camera)
    assert (
        "A" not in scene.overlay_names()
        and "B" in scene.overlay_names()
    )
    with pytest.raises(ValueError):
        viewer.panel_visible("A panel")
    assert (
        viewer.lighting == app.Lighting.CAMERA
    )  # B's own choice, kept over the reset
