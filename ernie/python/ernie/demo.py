# SPDX-License-Identifier: BSD-3-Clause
"""Demos: a base class for Ernie's demos and the loop that runs them, in a window, headless or
through a bare renderer, with a Demos panel to switch between them.

A demo subclasses Demo, builds its scene into `self.scene` in `__init__`, advances it in `step`
and checks itself in `test_final`::

    import ernie.demo
    from ernie import tokens

    class Demo(ernie.demo.Demo):
        def __init__(self, viewer, args):
            super().__init__(viewer, args)
            self.scene.define("/World/Ball", tokens.SPHERE)

        def step(self):
            ...  # move things by self.frame_dt

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

In the viewer, a demo uses it as any application does (`add_panel`, `on_frame`, `active_camera`
...); what it adds is removed, and what it sets is restored, when the Demos panel switches away.

References:
- Newton's examples (`newton.examples`: `Example`, `create_parser`, `init`, `run` and the
  example browser), https://github.com/newton-physics/newton
"""

import argparse
import gc
import importlib.util
import os
import re
import struct
import sys
import time
import traceback
import zlib

import numpy as np

from . import Path, Renderer, Scene

#: Arguments every demo shares, kept when the Demos panel switches demos.
COMMON_ARGS = ("viewer", "headless", "device", "renderer", "validation", "vsync", "paused")

#: Frames a benchmark runs before it starts timing.
BENCHMARK_WARMUP = 5


class Demo:
    """A demo: a scene built in `__init__`, advanced by `step`, rendered by `render`.

    Attributes:
        viewer: The viewer the demo shows in, or None under `--viewer null`.
        args: The parsed arguments (`create_parser`).
        scene: The Scene to build into: the viewer's, or a new one without a viewer.
        frame: Steps taken so far.
        time: Demo seconds at the start of this step.
        frame_dt: Seconds this step advances: the time since the last frame in a window
            (at most `max_frame_dt`), else `fixed_dt`.
        image: The last LdrColor rendered without a viewer, uint8 (rows, cols, 4).
    """

    #: Frames to run without a window when `--num-frames` is not given.
    num_frames = 100
    #: Seconds each step advances without a window, for repeatable runs.
    fixed_dt = 1.0 / 60.0
    #: Longest step in a window, so a stall does not throw the simulation.
    max_frame_dt = 1.0 / 30.0
    #: Window title; the module's first docstring line when empty.
    title = ""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.scene = viewer.scene if viewer is not None else Scene()
        self.frame = 0
        self.time = 0.0
        self.frame_dt = self.fixed_dt
        self.image = None
        self._renderer = None

    @classmethod
    def create_parser(cls):
        """The common options; override and add a demo's own to the parser this returns."""
        return create_parser(cls)

    @classmethod
    def description(cls):
        """The first paragraph of the demo's module docstring, on one line."""
        doc = sys.modules[cls.__module__].__doc__ or cls.__doc__ or ""
        return " ".join(doc.strip().split("\n\n")[0].split())

    @property
    def renderer(self):
        """An `ernie.Renderer` of the demo's own, opened on first use with the run's device,
        backend and validation."""
        if self._renderer is None:
            self._renderer = Renderer(backend=self.args.renderer, validation=self.args.validation,
                                      device=self.args.device or "")
        return self._renderer

    def step(self):
        """Advances the scene by `frame_dt`; nothing for a still scene."""

    def render(self):
        """Draws a frame: through the viewer, or with `renderer` into `image` without one."""
        if self.viewer is not None:
            self.viewer.step()
        else:
            self.image = np.asarray(self.renderer.render(self.scene))

    def render_image(self):
        """The scene's LdrColor as its RenderSettings describe, uint8 (rows, cols, 4): the last
        frame without a viewer, else rendered now with `renderer`, for tests."""
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

    def close(self):
        """Releases what the demo holds beyond the scene and viewer, before another replaces it."""


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


def init(parser=None, title="Ernie demos"):
    """Parses the command line and opens the viewer it asks for: `(viewer, args)`, with viewer a
    `DemoViewer` over an `ernie.app.Viewer` on a new Scene, or None under `--viewer null`."""
    parser = parser or create_parser()
    args = parser.parse_args()
    if args.benchmark is not False:
        args.viewer = "null"
    if args.viewer == "null":
        return None, args
    from .app import Viewer

    viewer = Viewer(Scene(), title=title, hidden=args.headless, validation=args.validation,
                    device=args.device or "", renderer=args.renderer, vsync=args.vsync)
    return DemoViewer(viewer), args


def run(demo, args):
    """Steps and renders `demo` until its window closes or it has run `--num-frames`, then writes
    `--output`, runs its final test and reports the benchmark."""
    viewer = demo.viewer
    test_post_step = args.test and hasattr(demo, "test_post_step")
    test_final = args.test and hasattr(demo, "test_final")
    if args.test and not (test_post_step or test_final):
        raise NotImplementedError(f"{type(demo).__name__} has no test_final or test_post_step")
    num_frames = args.num_frames
    if num_frames is None and (viewer is None or args.headless):
        num_frames = type(demo).num_frames
    benchmark = args.benchmark is not False
    deadline = args.benchmark if benchmark else None

    browser = None
    if viewer is not None:
        viewer._adopt(demo)
        if not args.headless and not args.test:
            browser = DemoBrowser(viewer, demo, args)

    paused = args.paused and viewer is not None and not args.headless
    timed_frames = 0
    timer_start = None
    last = time.perf_counter()
    while viewer is None or viewer.is_open:
        if browser is not None:
            demo, paused = browser.update(demo, paused)
            if demo is None:
                viewer.step()
                continue
        if num_frames is not None and demo.frame >= num_frames:
            break
        now = time.perf_counter()
        realtime = viewer is not None and not args.headless
        demo.frame_dt = min(now - last, demo.max_frame_dt) if realtime else demo.fixed_dt
        last = now
        stepping = not paused or (browser is not None and browser.take_single_step())
        if stepping:
            demo.step()
            if test_post_step:
                demo.test_post_step()
        demo.render()
        if stepping:
            demo.frame += 1
            demo.time += demo.frame_dt
        if benchmark:
            if demo.frame == BENCHMARK_WARMUP:
                timer_start = time.perf_counter()
            elif timer_start is not None:
                timed_frames += 1
                if deadline is not None and time.perf_counter() - timer_start >= deadline:
                    break

    if args.output:
        demo.save_output(args.output)
        print(f"wrote {args.output}")
    if test_final:
        demo.test_final()
    if benchmark:
        if timer_start is None:
            print(f"Benchmark: too few frames to time (warm-up is {BENCHMARK_WARMUP})")
        else:
            elapsed = time.perf_counter() - timer_start
            print(f"Benchmark: {_format_fps(timed_frames / max(elapsed, 1e-9))} FPS "
                  f"({timed_frames} frames in {elapsed:.2f} s)")
    if viewer is not None:
        viewer.close()
    return demo


def main(demo_class):
    """Runs `demo_class` from the command line: `run(demo_class(*init(...)), args)`."""
    viewer, args = init(demo_class.create_parser(), title=demo_class.title or "Ernie demos")
    demo = demo_class(viewer, args)
    return run(demo, args)


class DemoViewer:
    """An `ernie.app.Viewer` as a demo sees it: everything passes through to the viewer (`raw`),
    while the panels, callbacks and viewport items a demo adds, and the viewer settings it
    assigns, are noted, so that switching demos removes and restores them."""

    _REGISTRATIONS = ("on_frame", "on_pre_render", "on_post_render", "on_selection",
                      "add_viewport_toggle", "add_hud_item", "add_viewport_menu_item")
    _RESTORED = ("viewport_display", "exposure", "exposure_compensation", "wireframe",
                 "view_var", "lighting", "gizmo")

    def __init__(self, viewer):
        object.__setattr__(self, "raw", viewer)
        object.__setattr__(self, "_defaults", {name: getattr(viewer, name)
                                               for name in self._RESTORED})
        object.__setattr__(self, "_callbacks", [])
        object.__setattr__(self, "_items", [])
        object.__setattr__(self, "_panels", [])
        object.__setattr__(self, "_assigned", {})

    def __getattr__(self, name):
        attribute = getattr(self.raw, name)
        if name in self._REGISTRATIONS:
            noted = self._callbacks if name.startswith("on_") else self._items

            def register(*args, **kwargs):
                handle = attribute(*args, **kwargs)
                noted.append(handle)
                return handle

            return register
        if name == "add_panel":
            def add_panel(title, *args, **kwargs):
                panel = attribute(title, *args, **kwargs)
                self._panels.append(title)
                return panel

            return add_panel
        return attribute

    def __setattr__(self, name, value):
        setattr(self.raw, name, value)
        self._assigned[name] = value

    def _adopt(self, demo):
        """Starts the session over on the demo's scene, then applies the viewer settings the demo
        assigned while building it (its camera, say), which the reset would otherwise undo."""
        self.raw.reset_session()
        for name, value in self._assigned.items():
            setattr(self.raw, name, value)
        title = demo.title or type(demo).description() or type(demo).__name__
        self.raw.log(f"Demo: {title}")

    def _release(self):
        """Removes what the last demo added and restores what it assigned."""
        for handle in self._callbacks:
            self.raw.remove_callback(handle)
        for handle in self._items:
            self.raw.remove_viewport_item(handle)
        for title in self._panels:
            self.raw.remove_panel(title)
        self._callbacks.clear()
        self._items.clear()
        self._panels.clear()
        self._assigned.clear()
        for name, value in self._defaults.items():
            setattr(self.raw, name, value)


class DemoBrowser:
    """The viewer's Demos panel: picks a demo among the `demo_*.py` files beside the running one,
    and pauses, steps and resets it. Switching clears the scene of everything but the viewer's
    own prims and overlays and builds the next demo into it."""

    def __init__(self, viewer, demo, args):
        from .app import Dock

        self.viewer = viewer
        self.args = args
        self.entries = discover_demos(os.path.dirname(_module_file(type(demo))))
        self.current = _module_file(type(demo))
        self.current_class = type(demo)
        self.switch_to = None
        self.reset_requested = False
        self.single_step = False
        self.paused = None

        panel = viewer.raw.add_panel("Demos", Dock.LEFT)
        self.panel = panel
        labels = [entry.label for entry in self.entries]
        selected = next((i for i, entry in enumerate(self.entries)
                         if os.path.samefile(entry.path, self.current)), 0)
        if labels:
            self.combo = panel.combo("Demo", labels, selected, on_change=self._choose)
        else:
            panel.text("No demos beside this one")
        panel.separator()
        self.pause_box = panel.checkbox("Paused", args.paused, on_change=self._pause)
        panel.button("Step", lambda: setattr(self, "single_step", True))
        panel.button("Reset", lambda: setattr(self, "reset_requested", True))
        self.status = panel.text("")
        panel.separator()
        self.about = panel.text(type(demo).description())

    def _choose(self, index):
        self.switch_to = self.entries[index]

    def _pause(self, paused):
        self.paused = paused

    def take_single_step(self):
        stepping, self.single_step = self.single_step, False
        return stepping

    def update(self, demo, paused):
        """Applies the panel's requests before a frame; returns the demo to run (None if loading
        one failed) and whether it is paused."""
        if self.paused is not None:
            paused, self.paused = self.paused, None
        if self.switch_to is not None:
            entry, self.switch_to = self.switch_to, None
            demo = self._load(demo, lambda: _load_demo_class(entry.path), entry.path)
        elif self.reset_requested:
            self.reset_requested = False
            demo = self._load(demo, lambda: self.current_class, self.current)
        if demo is not None:
            self.panel.set(self.status, f"frame {demo.frame}, {demo.time:.2f} s"
                           + ("  (paused)" if paused else ""))
        return demo, paused

    def _load(self, old, load_class, path):
        viewer = self.viewer
        if old is not None:
            old.close()
        viewer._release()
        clear_scene(viewer.scene)
        old = None
        gc.collect()
        try:
            demo_class = load_class()
            args = default_args(demo_class.create_parser())
            for name in COMMON_ARGS:
                setattr(args, name, getattr(self.args, name))
            demo = demo_class(viewer, args)
            viewer._adopt(demo)
        except (Exception, SystemExit) as error:  # noqa: BLE001 - shown in the Log panel
            viewer.raw.log(f"Could not load {os.path.basename(path)}: {error}", "error")
            traceback.print_exc()
            viewer._release()
            clear_scene(viewer.scene)
            viewer.raw.reset_session()
            self.panel.set(self.about, "")
            return None
        self.current, self.current_class = path, demo_class
        self.panel.set(self.about, demo_class.description())
        return demo


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


def clear_scene(scene):
    """Removes every prim but the viewer's own (under /__session) and every overlay but the
    viewer's own (named with a leading "__")."""
    from .app import Viewer

    session = Viewer.session_root
    for child in scene.children(Path("/")):
        if child != session:
            scene.remove(child)
    for name in scene.overlay_names():
        if not name.startswith("__"):
            scene.remove_overlay(name)


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


def _module_file(cls):
    return os.path.abspath(sys.modules[cls.__module__].__file__)


def _load_demo_class(path):
    """Imports the demo file at `path` afresh (its directory first on sys.path, for the helpers
    beside it) and returns its Demo class."""
    directory = os.path.dirname(path)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    name = "ernie_demo_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Demo


def _format_fps(fps):
    return f"{fps:.1f}" if fps >= 10 else f"{fps:.2f}" if fps >= 1 else f"{fps:.3f}"


__all__ = [
    "COMMON_ARGS",
    "Demo",
    "DemoBrowser",
    "DemoViewer",
    "clear_scene",
    "create_parser",
    "default_args",
    "discover_demos",
    "init",
    "main",
    "run",
    "write_png",
]
