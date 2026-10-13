# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import os
import queue
import re
import struct
import sys
import threading
import urllib.request
import zlib

import numpy as np

from . import _ernie

if not hasattr(_ernie, "app"):
    raise ImportError(
        "This build of Ernie has no app; configure it with "
        "-DERNIE_BUILD_APP=ON"
    )

_app = _ernie.app

#: Arguments every demo shares, kept when the Demos panel switches
#: demos.
COMMON_ARGS = (
    "viewer",
    "headless",
    "device",
    "renderer",
    "validation",
    "vsync",
    "paused",
)


class Demo(_app.Demo):
    """A demo: a scene built in `__init__`, advanced by `step`,
    rendered by `render`.

    Attributes:
        viewer: The viewer the demo shows in, or None under `--viewer
            null`.
        args: The parsed arguments (`create_parser`).
        scene: The Scene to build into: the viewer's, or a new one
            without a viewer.
        renderer: An `ernie.Renderer` of the demo's own, opened on
            first use.
        frame: Steps taken so far.
        time: Demo seconds at the start of this step.
        frame_dt: Seconds this step advances: the time since the last
            frame in a window (at most `max_frame_dt`), else
            `fixed_dt`.
        image: The last LdrColor rendered without a viewer, uint8
            (rows, cols, 4).
    """

    #: Frames to run without a window when `--num-frames` is not
    #: given; a demo may set its own in `__init__`, from its options.
    num_frames = 100
    #: Seconds each step advances without a window, for repeatable
    #: runs.
    fixed_dt = 1.0 / 60.0
    #: Longest step in a window, so a stall does not throw the
    #: simulation.
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
        """The common options; override and add a demo's own to the
        parser this returns."""
        return create_parser(cls)

    @classmethod
    def description(cls):
        """The first paragraph of the demo's module docstring, on one
        line."""
        module = sys.modules.get(cls.__module__)
        doc = (
            (module.__doc__ if module else None) or cls.__doc__ or ""
        )
        return " ".join(doc.strip().split("\n\n")[0].split())

    def render(self):
        """Draws a frame: through the viewer, or with `renderer` into
        `image` without one; a benchmark without checks renders
        asynchronously, reading nothing back (`_app.Demo.render`)."""
        if self.viewer is not None:
            self.viewer.step()
        elif self.args.benchmark is not False and not self.args.test:
            _app.Demo.render(self)
        else:
            self.image = np.asarray(self.renderer.render(self.scene))

    def render_image(self):
        """The scene's LdrColor as its RenderSettings (or its first
        Camera) describe, uint8 (rows, cols, 4): the last frame
        without a viewer, else rendered now with `renderer`, for
        tests. Not the viewport: the viewer's own camera is not the
        scene's."""
        if self.viewer is None and self.image is not None:
            return self.image
        return np.asarray(self.renderer.render(self.scene))

    def save_output(self, path):
        """Writes `--output`: the viewport's next frame in a viewer,
        else the last frame."""
        if self.viewer is not None:
            self.viewer.capture(path)
            self.viewer.step()
        else:
            write_png(path, self.render_image()[..., :3])

    def map(self, path, attribute, stream=None):
        """A CUDA array over the renderer's own copy of an attribute,
        to write in place and then `scene.unmap`: the viewer's
        (`Viewer.map`) or `renderer`'s (`Renderer.map`)."""
        if self.viewer is not None:
            return self.viewer.map(path, attribute, stream=stream)
        return self.renderer.map(
            self.scene, path, attribute, stream=stream
        )


def create_parser(demo_class=Demo):
    """An argument parser with the options every demo shares;
    `Demo.create_parser` calls it."""
    parser = argparse.ArgumentParser(
        description=demo_class.description() or None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--viewer",
        choices=["window", "null"],
        default="window",
        help="'window' shows the viewer; 'null' renders the scene's "
        "RenderSettings with a bare Renderer and no viewer",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="render the viewer without a window (no display needed)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="frames to run; a window runs until closed, else "
        f"{demo_class.num_frames} for this demo",
    )
    parser.add_argument(
        "--benchmark",
        type=float,
        default=False,
        nargs="?",
        const=None,
        metavar="SECONDS",
        help="report frames per second, rendering with --viewer "
        "null; stop after SECONDS, if given, or --num-frames",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print the profiler's GPU and CPU zones, nested and "
        "averaged, at the end",
    )
    parser.add_argument(
        "--profile-json",
        metavar="PATH",
        help="also write them as JSON (implies --profile)",
    )
    parser.add_argument(
        "--profile-trace",
        metavar="PATH",
        help="also write a Chrome trace for Perfetto or "
        "chrome://tracing (implies --profile)",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        metavar="N",
        help="first frames the profile leaves out (default 5); 0 "
        "keeps the scene's loading (implies --profile)",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the demo's test_post_step and test_final checks",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the last frame here as PNG",
    )
    parser.add_argument(
        "--paused",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="start paused (the Demos panel resumes)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="GPU to prefer, by part of its name",
    )
    parser.add_argument(
        "--renderer", default="raster", help="render backend"
    )
    parser.add_argument(
        "--validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the Vulkan validation layer where installed",
    )
    parser.add_argument(
        "--vsync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="wait for the display's refresh",
    )
    return parser


def default_args(parser):
    """The parser's defaults, for a demo the Demos panel opens."""
    return parser.parse_args([])


def options_from_args(args):
    """The `ernie.app.DemoOptions` the parsed arguments of
    `create_parser` describe."""
    options = _app.DemoOptions()
    options.viewer = (
        args.viewer == "window" and args.benchmark is False
    )
    options.headless = args.headless
    options.num_frames = args.num_frames
    options.benchmark = args.benchmark is not False
    options.benchmark_seconds = args.benchmark or None
    options.profile_json = args.profile_json or ""
    options.profile_trace = args.profile_trace or ""
    if args.profile_warmup is not None:
        options.profile_warmup = args.profile_warmup
    options.profile = bool(
        args.profile
        or args.profile_json
        or args.profile_trace
        or args.profile_warmup is not None
    )
    options.test = args.test
    options.output = args.output or ""
    options.paused = args.paused
    options.device = args.device or ""
    options.renderer = args.renderer
    options.validation = args.validation
    options.vsync = args.vsync
    return options


def init(parser=None, title="Ernie demos"):
    """Parses the command line and opens the viewer it asks for:
    `(viewer, args)`, with viewer an `ernie.app.Viewer` on a new
    Scene, or None under `--viewer null` and `--benchmark`. The runner
    (`ernie.app.DemoRunner`) rides along in the arguments for
    `run`."""
    parser = parser or create_parser()
    args = parser.parse_args()
    if args.benchmark is not False:
        args.viewer = "null"
    args.runner = _app.DemoRunner(options_from_args(args), title)
    return args.runner.viewer, args


def run(demo, args):
    """Steps and renders `demo` until its window closes or it has run
    `--num-frames`, then writes `--output`, runs its final test and
    reports the benchmark. In a window the Demos panel lists the demo
    files beside this one. Returns the demo running at the end."""
    runner = getattr(args, "runner", None)
    if runner is None:
        runner = args.runner = _app.DemoRunner(
            options_from_args(args)
        )
    if runner.viewer is not None:
        runner.set_catalog(*catalog(type(demo), args))
    return runner.run(demo)


def main(demo_class):
    """Runs `demo_class` from the command line:
    `run(demo_class(*init(...)), args)`."""
    viewer, args = init(
        demo_class.create_parser(),
        title=demo_class.title or "Ernie demos",
    )
    demo = demo_class(viewer, args)
    return run(demo, args)


def catalog(demo_class, args):
    """The Demos panel's entries for a run of `demo_class`, `(infos,
    current)`: the demos of every category beside its own
    (`demos/<category>/demo_<category>_<name>.py`), or the class alone
    when it is not among them."""
    path = _module_file(demo_class)
    entries = (
        discover_demos(os.path.dirname(os.path.dirname(path)))
        if path
        else []
    )
    current = next(
        (
            i
            for i, entry in enumerate(entries)
            if os.path.exists(entry.path)
            and os.path.samefile(entry.path, path)
        ),
        None,
    )
    infos = [
        _app.DemoInfo(
            entry.category, entry.name, "", _creator(entry.path, args)
        )
        for entry in entries
    ]
    if current is None:
        infos.insert(
            0,
            _app.DemoInfo(
                "this",
                demo_class.__name__,
                demo_class.description(),
                _creator(demo_class, args),
            ),
        )
        current = 0
    return infos, current


class DemoEntry:
    """A demo file, `<category>/demo_<category>_<name>.py`: its path,
    category, name and id (its file's stem, `demo_<category>_<name>`,
    as `--demo` and `python -m ernie.demos` take it). A file of the
    former `<category>/<name>.py` form gets the same id."""

    def __init__(self, path):
        self.path = path
        self.category = os.path.basename(
            os.path.dirname(os.path.abspath(path))
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        prefix = f"demo_{self.category}_"
        short = (
            stem[len(prefix) :] if stem.startswith(prefix) else stem
        )
        self.name = short.replace("_", " ")
        self.id = f"{prefix}{short}"
        #: The former id, `<category>/<name>`, which still finds it.
        self.path_id = f"{self.category}/{short}"

    @property
    def label(self):
        return f"{self.category}: {self.name}"


def discover_demos(root):
    """The demos under `root`, `<category>/demo_<category>_<name>.py`
    files that define a Demo class, by category and name."""
    entries = []
    for category in sorted(os.listdir(root)):
        directory = os.path.join(root, category)
        if category.startswith((".", "_")) or not os.path.isdir(
            directory
        ):
            continue
        for filename in sorted(os.listdir(directory)):
            if filename.startswith("_") or not filename.endswith(
                ".py"
            ):
                continue
            path = os.path.join(directory, filename)
            with open(path, encoding="utf-8") as file:
                if re.search(
                    r"^class Demo\(", file.read(), re.MULTILINE
                ):
                    entries.append(DemoEntry(path))
    return entries


def load_demo_class(path):
    """Imports the demo file at `path` afresh (its directory first on
    sys.path, for the helpers beside it) and returns its Demo
    class."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    stem = os.path.splitext(os.path.basename(path))[0]
    name = f"ernie_demo_{os.path.basename(directory)}_{stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not a Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Demo


def cache_dir():
    """Where the demos keep data they fetch: $ERNIE_CACHE, else
    $XDG_CACHE_HOME/ernie, else ~/.cache/ernie (C++
    `demos::CacheDir`)."""
    if os.environ.get("ERNIE_CACHE"):
        return os.environ["ERNIE_CACHE"]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "ernie")


def can_download():
    """Whether the demos may fetch data: $ERNIE_OFFLINE unset."""
    return not os.environ.get("ERNIE_OFFLINE")


def _open(request, report=print):
    """`urlopen(request)`'s status, headers and body, tried up to three
    times: servers drop connections mid-transfer now and then. Failed
    attempts go to `report`."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(
                request, timeout=120
            ) as response:
                return (
                    response.status,
                    response.headers,
                    response.read(),
                )
        except OSError as error:
            if attempt == 2:
                raise
            report(
                f"Fetching {request.full_url} failed: {error}; trying again"
            )


def _require_downloads():
    if not can_download():
        raise RuntimeError(
            "The demos cannot fetch data: $ERNIE_OFFLINE is set"
        )


def download(url, path, report=print):
    """Downloads `url` to `path`, its directory made as needed,
    through `path` + ".part", renamed when complete; failed attempts
    that are tried again go to `report`."""
    _require_downloads()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "ernie-demos"}
    )
    _, _, body = _open(request, report)
    with open(path + ".part", "wb") as f:
        f.write(body)
    os.replace(path + ".part", path)


class AsyncDownload:
    """`download` on a thread of its own, so a demo runs while its
    data comes: its progress and outcome are messages for the demo to
    show as it steps (`take_messages`, such as into the viewer's
    log)."""

    def __init__(self, url, path):
        self._messages = queue.Queue()
        self._messages.put((f"Fetching {url}", False))
        self.done = False
        self.succeeded = False

        def run():
            try:
                download(
                    url,
                    path,
                    lambda text: self._messages.put((text, False)),
                )
                self._messages.put(
                    (f"Fetched {os.path.basename(path)}", False)
                )
                self.succeeded = True
            except (OSError, RuntimeError) as error:
                self._messages.put((str(error), True))
            self.done = True

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def take_messages(self):
        """The (text, is_error) messages since the last call."""
        messages = []
        while not self._messages.empty():
            messages.append(self._messages.get_nowait())
        return messages


def fetch_ranges(url, ranges):
    """The bytes of `ranges` [(first, last), ...] of `url`, each
    inclusive, in one request (HTTP range requests; several come back
    as multipart/byteranges, RFC 9110 section 14.6)."""
    _require_downloads()
    header = "bytes=" + ",".join(f"{a}-{b}" for a, b in ranges)
    request = urllib.request.Request(
        url, headers={"Range": header, "User-Agent": "ernie-demos"}
    )
    status, headers, body = _open(request)
    if status != 206:
        raise RuntimeError(f"{url} ignored the range request")
    kind = headers.get("Content-Type", "")
    if len(ranges) == 1:
        return [body]
    match = re.search(r"boundary=(.+)", kind)
    if match is None:
        raise RuntimeError(f"{url} sent no multipart boundary")
    boundary = match.group(1).strip('"')
    parts = {}
    for part in body.split(b"--" + boundary.encode()):
        head, sep, data = part.partition(b"\r\n\r\n")
        found = re.search(rb"Content-Range: bytes (\d+)-(\d+)", head)
        if not sep or not found:
            continue
        first, last = int(found.group(1)), int(found.group(2))
        parts[first] = data[: last - first + 1]
    result = [parts.get(a, b"") for a, _ in ranges]
    if any(len(r) != b - a + 1 for r, (a, b) in zip(result, ranges)):
        raise RuntimeError(f"{url} sent incomplete ranges")
    return result


def write_png(path, image):
    """Writes an (rows, cols, 3 or 4) uint8 image as PNG with only the
    standard library."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    rows, cols, channels = image.shape
    colour_type = {3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + image[r].tobytes() for r in range(rows))

    def chunk(kind, data):
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(
        ">IIBBBBB", cols, rows, 8, colour_type, 0, 0, 0
    )
    with open(path, "wb") as file:
        file.write(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b"")
        )


def side_by_side(images, gap=4):
    """uint8 (rows, cols, channels) images of one height next to each
    other with black gaps."""
    height = images[0].shape[0]
    parts = []
    for i, image in enumerate(images):
        if i > 0:
            parts.append(
                np.zeros(
                    (height, gap, image.shape[2]), dtype=np.uint8
                )
            )
        parts.append(np.asarray(image, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def srgb_to_linear(rgb):
    """An sRGB-encoded colour in [0, 1] to linear."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(
        rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
    )


def define_planet_surface(scene, path, radius, albedo, material):
    """A smooth planet of `radius` metres at `path`: a planet Terrain of
    flat cube faces, its surface at the world's z = 0 (its centre
    `radius` below the origin), drawn with a Material of normal albedo
    `albedo` (linear RGB) at `material`. The ground under an atmosphere,
    which is the air alone (as the C++ demos' `DefinePlanetSurface`)."""
    from . import Material, Terrain

    scene.define(
        path,
        Terrain,
        heights=np.zeros((6, 65, 65), dtype=np.float32),
        planet_radius=radius,
        material_binding=material,
        translate=(0.0, 0.0, -radius),
    )
    scene.define(material, Material, albedo=tuple(albedo))


def mean_linear_rgb(image):
    """The mean colour of a uint8 (rows, cols, 3 or 4) image, decoded
    from sRGB to linear."""
    return srgb_to_linear(
        np.asarray(image)[..., :3].reshape(-1, 3).mean(axis=0) / 255.0
    )


def heatmap(values, low, high):
    """`values` from `low` (blue) through cyan, green and yellow to
    `high` (red), as RGB uint8."""
    t = np.clip(
        (np.asarray(values, dtype=np.float64) - low)
        / max(high - low, 1e-12),
        0.0,
        1.0,
    )
    stops = np.array(
        [
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ],
        dtype=np.float64,
    )
    position = t * (len(stops) - 1)
    index = np.minimum(position.astype(int), len(stops) - 2)
    fraction = (position - index)[..., None]
    return (
        (1.0 - fraction) * stops[index] + fraction * stops[index + 1]
    ).astype(np.uint8)


def _creator(source, args):
    """create(viewer, options) for the Demos panel: builds the demo of
    the file (or class) `source` with its parser's defaults and the
    run's shared arguments. It holds those values, not `args`, which
    holds the runner that holds it."""
    shared = {name: getattr(args, name) for name in COMMON_ARGS}

    def create(viewer, options):
        demo_class = (
            load_demo_class(source)
            if isinstance(source, str)
            else source
        )
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
    "cache_dir",
    "can_download",
    "catalog",
    "create_parser",
    "default_args",
    "discover_demos",
    "download",
    "fetch_ranges",
    "heatmap",
    "init",
    "load_demo_class",
    "main",
    "mean_linear_rgb",
    "options_from_args",
    "run",
    "side_by_side",
    "srgb_to_linear",
    "write_png",
]
