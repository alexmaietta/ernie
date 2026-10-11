# SPDX-License-Identifier: BSD-3-Clause
"""Demos: a base class for Ernie's demos and the loop that runs them, in a window, headless or
through a bare renderer, with a Demos panel to switch between them.

A demo subclasses Demo, builds its scene into `self.scene` in `__init__`, advances it in `step`
and checks itself in `test_final`. Its scene has a Camera of its own, even when the demo is meant
for the viewer (which frames the scene with its own camera): without a viewer (`--viewer null`,
`--benchmark`) and in tests (`render_image`) the scene renders through it::

    import ernie
    import ernie.demo
    from ernie import tokens

    class Demo(ernie.demo.Demo):
        def __init__(self, viewer, args):
            super().__init__(viewer, args)
            self.ball = self.scene.define("/World/Ball", ernie.Sphere, radius=0.5)
            self.scene.define("/World/Sun", ernie.DistantLight)
            self.scene.define("/World/Camera", ernie.Camera).set(
                tokens.TRANSFORM, ernie.look_at_transform((0, -5, 2), (0, 0, 0)))

        def step(self):
            self.ball.set(translate=(0, 0, self.time))  # moved by self.frame_dt each step

        def test_final(self):
            assert self.render_image()[..., :3].any()

    if __name__ == "__main__":
        ernie.demo.main(Demo)

`main(Demo)` is `init(Demo.create_parser())` then `run(Demo(viewer, args), args)`, as Newton's
examples run. The common options:

- `--viewer window` (the default) opens the viewer, with a Demos panel listing the demos beside
  this one to switch to, pause, step and reset; `--headless` renders the viewer without a window
  (no display needed); `--viewer null` has no viewer and renders the scene's RenderSettings
  through an `ernie.Renderer` each frame.
- `--num-frames N` stops after N frames (in a window, runs until it closes without it).
- `--benchmark [SECONDS]` forces `--viewer null`, and reports frames per second after a few
  frames' warm-up, stopping after SECONDS or `--num-frames`, whichever comes first.
- `--test` calls the demo's `test_post_step` after every step and `test_final` at the end; one of
  them must exist.
- `--output PATH` writes the last frame (or what `save_output` makes) as PNG.
- `--paused`, `--device`, `--renderer`, `--validation` and `--vsync`.

The base class and the loop are C++'s (`ernie.app.Demo`, `ernie.app.DemoRunner`; ernie/app/demo.h),
which C++ demos derive from and run the same way; this module adds argparse options, NumPy images
and the demos found beside a Python one. In the viewer, a demo uses it as any application does
(`add_panel`, `on_frame`, `active_camera` ...); what it adds is removed (it registers in a scope
of its own, `Viewer.scope`), and the viewport's display settings are restored, when the Demos
panel switches away.

References:
- Newton's examples (`newton.examples`: `Example`, `create_parser`, `init`, `run` and the
  example browser), https://github.com/newton-physics/newton
"""

import argparse
import importlib.util
import os
import re
import struct
import sys
import zlib

import numpy as np

from . import _ernie

if not hasattr(_ernie, "app"):
    raise ImportError("This build of Ernie has no app; configure it with -DERNIE_BUILD_APP=ON")

_app = _ernie.app

#: Arguments every demo shares, kept when the Demos panel switches demos.
COMMON_ARGS = ("viewer", "headless", "device", "renderer", "validation", "vsync", "paused")


class Demo(_app.Demo):
    """A demo: a scene built in `__init__`, advanced by `step`, rendered by `render`.

    Attributes:
        viewer: The viewer the demo shows in, or None under `--viewer null`.
        args: The parsed arguments (`create_parser`).
        scene: The Scene to build into: the viewer's, or a new one without a viewer.
        renderer: An `ernie.Renderer` of the demo's own, opened on first use.
        frame: Steps taken so far.
        time: Demo seconds at the start of this step.
        frame_dt: Seconds this step advances: the time since the last frame in a window
            (at most `max_frame_dt`), else `fixed_dt`.
        image: The last LdrColor rendered without a viewer, uint8 (rows, cols, 4).
    """

    #: Frames to run without a window when `--num-frames` is not given; a demo may set its own
    #: in `__init__`, from its options.
    num_frames = 100
    #: Seconds each step advances without a window, for repeatable runs.
    fixed_dt = 1.0 / 60.0
    #: Longest step in a window, so a stall does not throw the simulation.
    max_frame_dt = 1.0 / 30.0
    #: Window title; "Ernie demos" when empty.
    title = ""

    def __init__(self, viewer, args):
        super().__init__(viewer, options_from_args(args))
        self.viewer = viewer
        self.args = args
        self.frame_dt = self.fixed_dt
        self.image = None

    @classmethod
    def create_parser(cls):
        """The common options; override and add a demo's own to the parser this returns."""
        return create_parser(cls)

    @classmethod
    def description(cls):
        """The first paragraph of the demo's module docstring, on one line."""
        module = sys.modules.get(cls.__module__)
        doc = (module.__doc__ if module else None) or cls.__doc__ or ""
        return " ".join(doc.strip().split("\n\n")[0].split())

    def render(self):
        """Draws a frame: through the viewer, or with `renderer` into `image` without one."""
        if self.viewer is not None:
            self.viewer.step()
        else:
            self.image = np.asarray(self.renderer.render(self.scene))

    def render_image(self):
        """The scene's LdrColor as its RenderSettings (or its first Camera) describe, uint8
        (rows, cols, 4): the last frame without a viewer, else rendered now with `renderer`, for
        tests. Not the viewport: the viewer's own camera is not the scene's."""
        if self.viewer is None and self.image is not None:
            return self.image
        return np.asarray(self.renderer.render(self.scene))

    def save_output(self, path):
        """Writes `--output`: the viewport's next frame in a viewer, else the last frame."""
        if self.viewer is not None:
            self.viewer.capture(path)
            self.viewer.step()
        else:
            write_png(path, self.render_image()[..., :3])

    def map(self, path, attribute, stream=None):
        """A CUDA array over the renderer's own copy of an attribute, to write in place and then
        `scene.unmap`: the viewer's (`Viewer.map`) or `renderer`'s (`Renderer.map`)."""
        if self.viewer is not None:
            return self.viewer.map(path, attribute, stream=stream)
        return self.renderer.map(self.scene, path, attribute, stream=stream)


def create_parser(demo_class=Demo):
    """An argument parser with the options every demo shares; `Demo.create_parser` calls it."""
    parser = argparse.ArgumentParser(description=demo_class.description() or None,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--viewer", choices=["window", "null"], default="window",
                        help="'window' shows the viewer; 'null' renders the scene's "
                        "RenderSettings with a bare Renderer and no viewer")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False,
                        help="render the viewer without a window (no display needed)")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="frames to run; a window runs until closed, else "
                        f"{demo_class.num_frames} for this demo")
    parser.add_argument("--benchmark", type=float, default=False, nargs="?", const=None,
                        metavar="SECONDS",
                        help="report frames per second, rendering with --viewer null; stop "
                        "after SECONDS, if given, or --num-frames")
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False,
                        help="run the demo's test_post_step and test_final checks")
    parser.add_argument("--output", metavar="PATH", help="write the last frame here as PNG")
    parser.add_argument("--paused", action=argparse.BooleanOptionalAction, default=False,
                        help="start paused (the Demos panel resumes)")
    parser.add_argument("--device", default=None, help="GPU to prefer, by part of its name")
    parser.add_argument("--renderer", default="raster", help="render backend")
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction, default=False,
                        help="enable the Vulkan validation layer where installed")
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=False,
                        help="wait for the display's refresh")
    return parser


def default_args(parser):
    """The parser's defaults, for a demo the Demos panel opens."""
    return parser.parse_args([])


def options_from_args(args):
    """The `ernie.app.DemoOptions` the parsed arguments of `create_parser` describe."""
    options = _app.DemoOptions()
    options.viewer = args.viewer == "window" and args.benchmark is False
    options.headless = args.headless
    options.num_frames = args.num_frames
    options.benchmark = args.benchmark is not False
    options.benchmark_seconds = args.benchmark or None
    options.test = args.test
    options.output = args.output or ""
    options.paused = args.paused
    options.device = args.device or ""
    options.renderer = args.renderer
    options.validation = args.validation
    options.vsync = args.vsync
    return options


def init(parser=None, title="Ernie demos"):
    """Parses the command line and opens the viewer it asks for: `(viewer, args)`, with viewer an
    `ernie.app.Viewer` on a new Scene, or None under `--viewer null` and `--benchmark`. The
    runner (`ernie.app.DemoRunner`) rides along in the arguments for `run`."""
    parser = parser or create_parser()
    args = parser.parse_args()
    if args.benchmark is not False:
        args.viewer = "null"
    args.runner = _app.DemoRunner(options_from_args(args), title)
    return args.runner.viewer, args


def run(demo, args):
    """Steps and renders `demo` until its window closes or it has run `--num-frames`, then writes
    `--output`, runs its final test and reports the benchmark. In a window the Demos panel lists
    the demo files beside this one. Returns the demo running at the end."""
    runner = getattr(args, "runner", None)
    if runner is None:
        runner = args.runner = _app.DemoRunner(options_from_args(args))
    if runner.viewer is not None:
        runner.set_catalog(*catalog(type(demo), args))
    return runner.run(demo)


def main(demo_class):
    """Runs `demo_class` from the command line: `run(demo_class(*init(...)), args)`."""
    viewer, args = init(demo_class.create_parser(), title=demo_class.title or "Ernie demos")
    demo = demo_class(viewer, args)
    return run(demo, args)


def catalog(demo_class, args):
    """The Demos panel's entries for a run of `demo_class`, `(infos, current)`: the demo files
    beside its module, or the class alone when it is not among them."""
    path = _module_file(demo_class)
    entries = discover_demos(os.path.dirname(path)) if path else []
    current = next((i for i, entry in enumerate(entries)
                    if os.path.exists(entry.path) and os.path.samefile(entry.path, path)), None)
    infos = [_app.DemoInfo(entry.category, entry.name, "", _creator(entry.path, args))
             for entry in entries]
    if current is None:
        infos.insert(0, _app.DemoInfo("this", demo_class.__name__, demo_class.description(),
                                      _creator(demo_class, args)))
        current = 0
    return infos, current


class DemoEntry:
    """A demo file: its path, category and name from `demo_<category>_<name>.py`."""

    def __init__(self, path):
        self.path = path
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_", 2)
        self.category = parts[1] if len(parts) > 1 else "other"
        self.name = parts[2].replace("_", " ") if len(parts) > 2 else stem

    @property
    def label(self):
        return f"{self.category}: {self.name}"


def discover_demos(directory):
    """The `demo_*.py` files in `directory` that define a Demo class, by category and name."""
    entries = []
    for filename in sorted(os.listdir(directory)):
        if not (filename.startswith("demo_") and filename.endswith(".py")):
            continue
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as file:
            if re.search(r"^class Demo\(", file.read(), re.MULTILINE):
                entries.append(DemoEntry(path))
    return entries


def load_demo_class(path):
    """Imports the demo file at `path` afresh (its directory first on sys.path, for the helpers
    beside it) and returns its Demo class."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    name = "ernie_demo_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Demo


def write_png(path, image):
    """Writes an (rows, cols, 3 or 4) uint8 image as PNG with only the standard library."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    rows, cols, channels = image.shape
    colour_type = {3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + image[r].tobytes() for r in range(rows))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", cols, rows, 8, colour_type, 0, 0, 0)
    with open(path, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                   chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _creator(source, args):
    """create(viewer, options) for the Demos panel: builds the demo of the file (or class)
    `source` with its parser's defaults and the run's shared arguments. It holds those values,
    not `args`, which holds the runner that holds it."""
    shared = {name: getattr(args, name) for name in COMMON_ARGS}

    def create(viewer, options):
        demo_class = load_demo_class(source) if isinstance(source, str) else source
        demo_args = default_args(demo_class.create_parser())
        for name, value in shared.items():
            setattr(demo_args, name, value)
        return demo_class(viewer, demo_args)

    return create


def _module_file(cls):
    module = sys.modules.get(cls.__module__)
    path = getattr(module, "__file__", None)
    return os.path.abspath(path) if path else None


__all__ = [
    "COMMON_ARGS",
    "Demo",
    "DemoEntry",
    "catalog",
    "create_parser",
    "default_args",
    "discover_demos",
    "init",
    "load_demo_class",
    "main",
    "options_from_args",
    "run",
    "write_png",
]
