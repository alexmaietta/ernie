# SPDX-License-Identifier: BSD-3-Clause
"""A Newton viewer whose 3D view Ernie renders: ``ViewerErnie`` is
``newton.viewer.ViewerGL`` (its window, GUI, camera controls,
picking and the examples' panels) with the scene drawn by Ernie, from
Newton's camera, in place of its OpenGL pass.

What Newton logs is mirrored into an ``ernie.Scene`` (your own, or a
new one under Earth's sky and Sun): its shapes, tessellated into meshes
(``log_mesh``) and posed in batches (``log_instances``), become a
PointInstancer per batch under ``root``, their transforms split on the
GPU from the Warp arrays Newton passes, coloured per instance by the
instancer's ``primvars:displayColor``, linearised on the GPU. Ground
planes become GroundPlanes with
Newton's checker, lines a scene overlay, points Points prims. Each
frame Ernie renders the scene into a CUDA image, which ViewerGL shows
as its fullscreen image (CUDA-OpenGL interop, no host copy).

References:
    Newton's viewer API, ``newton.viewer.ViewerBase`` and
    ``ViewerGL`` (newton/_src/viewer/viewer.py, viewer_gl.py).
"""

from __future__ import annotations

import atexit
import gc
import re
import weakref
from collections.abc import Callable
from typing import Any

import newton
import numpy as np
import warp as wp
from newton.viewer import ViewerGL

import ernie
from ernie import tokens

# Newton's checker squares: albedo and 0.7 x it
# (newton/_src/viewer/gl/shaders.py).
_CHECKER_DARKEN = 0.7

# Newton's sky as a dome's linear RGB, from its horizon to its zenith
# (newton/_src/viewer/gl/opengl.py's sRGB), and the illuminance
# setting the dome's brightness behind the scene at the view's
# exposure. Newton's ambient light, neutral as the sky's colour is
# not, and its luminance (cd/m^2) beside the Sun's.
_SKY_LOWER = tuple((c / 255.0) ** 2.2 for c in (40.0, 44.0, 55.0))
_SKY_UPPER = tuple((c / 255.0) ** 2.2 for c in (68.0, 161.0, 255.0))
_SKY_LUX = 28000.0
_AMBIENT = (0.8, 0.8, 0.85)
_AMBIENT_LUMINANCE = 1500.0
# Minimal shading's fill, which stands in for the sky's light on what
# the sun leaves: stronger, and the sky's blue, as ViewerGL's ambient.
_MINIMAL_AMBIENT = (0.55, 0.7, 1.0)
_MINIMAL_AMBIENT_LUMINANCE = 12000.0


def _sky_gradient(rows: int = 64, columns: int = 8) -> np.ndarray:
    """Newton's sky as a dome's latitude-longitude texture, linear
    RGB: its lower colour at and below the horizon, blending to its
    upper by the sine of the elevation, as Newton's sky shader."""
    elevation = np.pi / 2 - np.pi * (np.arange(rows) + 0.5) / rows
    height = np.maximum(np.sin(elevation), 0.0)[:, None]
    rgb = (
        (1.0 - height) * np.asarray(_SKY_LOWER)
        + height * np.asarray(_SKY_UPPER)
    ).astype(np.float32)
    return np.ascontiguousarray(
        np.broadcast_to(rgb[:, None, :], (rows, columns, 3))
    )


@wp.kernel(enable_backward=False)
def _split_transforms(
    xforms: wp.array(dtype=wp.transform),
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.vec4),
):
    i = wp.tid()
    xform = xforms[i]
    positions[i] = wp.transform_get_translation(xform)
    q = wp.transform_get_rotation(xform)
    orientations[i] = wp.vec4(q[0], q[1], q[2], q[3])  # x, y, z, w


@wp.kernel(enable_backward=False)
def _linear_colors(
    srgb: wp.array(dtype=wp.vec3),
    linear: wp.array(dtype=wp.vec3),
):
    # Newton's sRGB colours in linear RGB, as its shader takes them;
    # one colour for all broadcasts.
    i = wp.tid()
    c = srgb[wp.min(i, srgb.shape[0] - 1)]
    linear[i] = wp.vec3(
        wp.pow(c[0], 2.2), wp.pow(c[1], 2.2), wp.pow(c[2], 2.2)
    )


@wp.kernel(enable_backward=False)
def _rotate_points(
    points: wp.array(dtype=wp.vec3),
    rotation: wp.mat33,
    turned: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    turned[i] = rotation * points[i]


@wp.kernel(enable_backward=False)
def _diameters(
    radii: wp.array(dtype=wp.float32),
    widths: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    widths[i] = 2.0 * radii[i]


def _linear(rgb: Any) -> tuple[float, float, float]:
    """Newton's sRGB colour in linear RGB, as its shader takes it."""
    return tuple(float(c) ** 2.2 for c in rgb)


def _identifier(name: str) -> str:
    """A prim name for a Newton path such as ``/model/shapes/box``."""
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", name.strip("/"))
    if not identifier or identifier[0].isdigit():
        identifier = "N" + identifier
    return identifier


def _up_rotation(
    up_axis: int,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """The rotation taking a model's up axis to Ernie's +Z, as a
    matrix and as ``rotate`` (XYZ Euler degrees)."""
    if up_axis == 1:  # Y up: +90 degrees about X
        return np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64
        ), (90.0, 0.0, 0.0)
    if up_axis == 0:  # X up: -90 degrees about Y
        return np.array(
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]], np.float64
        ), (0.0, -90.0, 0.0)
    return np.eye(3), (0.0, 0.0, 0.0)


# The view's shadings, as Ernie's minimal modes.
_SHADING = {"minimal": "diffuseGlossyEmission", "full": "off"}


class _Mesh:
    """A mesh Newton logged, and its own prim once shown."""

    def __init__(self) -> None:
        self.points: wp.array | None = None
        self.normals: wp.array | None = None
        self.indices: wp.array | None = None  # triangles, flat
        self.counts: np.ndarray | None = None  # all 3, on the CPU
        self.source: tuple[int, int] | None = (
            None  # indices' (ptr, len)
        )
        self.color = (0.7, 0.7, 0.7)
        self.prim: ernie.Mesh | None = None


class _Surface:
    """A particle surface Newton extracted (`ViewerErnie.
    log_particle_surface`), drawn from the first of its buffers' entries:
    a power of two above those in use lately, as Newton's capacity is
    often far larger."""

    def __init__(self, prim: ernie.Mesh) -> None:
        self.prim = prim
        self.surface: Any = None  # its ExtractionMesh, last logged
        self.points = 0  # entries drawn of its vertices
        self.indices = 0  # and of its indices
        # Counts on their way to the host, oldest first, and buffers
        # free for more.
        self.readbacks: list[tuple[wp.array, wp.Event]] = []
        self.free: list[wp.array] = []

    def fit(self, points: int, indices: int, surface: Any) -> None:
        """Grows the entries drawn to hold `points` and `indices`, with
        room to spare, within the buffers."""

        def room(used: int, least: int) -> int:
            return 1 << max(least, int(1.25 * used)).bit_length()

        if points > self.points:
            self.points = min(
                room(points, 1024), surface.vertices.shape[0]
            )
        if indices > self.indices:
            self.indices = min(
                3 * room(indices // 3, 2048), surface.indices.shape[0]
            )

    def read(self, counts: wp.array, stream: wp.Stream) -> None:
        """Fits the counts read back frames ago, and reads this
        frame's without waiting for them."""
        while self.readbacks and self.readbacks[0][1].is_complete:
            host, _ = self.readbacks.pop(0)
            self.fit(
                int(host.numpy()[0]),
                int(host.numpy()[1]),
                self.surface,
            )
            self.free.append(host)
        if len(self.readbacks) < 3:
            host = (
                self.free.pop()
                if self.free
                else wp.empty(
                    counts.shape[0],
                    dtype=wp.int32,
                    device="cpu",
                    pinned=True,
                )
            )
            wp.copy(host, counts, stream=stream)
            self.readbacks.append((host, stream.record_event()))


# Surfaces extracted into preallocated buffers, by their vertex
# buffer's address, as `ExtractionMesh.to_arrays` last saw them
# (`track_particle_surfaces`).
_extractions: weakref.WeakValueDictionary[int, Any] = (
    weakref.WeakValueDictionary()
)


def _surface_counts(surface: Any) -> wp.array | None:
    """The device counts of a surface Newton extracted into
    preallocated buffers: vertices, indices, then an overflow flag;
    None for one of exact-sized buffers."""
    counts = getattr(surface, "_counts", None)
    if getattr(surface, "_exact", True) or counts is None:
        return None
    return counts


def track_particle_surfaces() -> None:
    """Has ``ViewerErnie.log_mesh`` recognise the arrays of a surface
    ``newton.geometry.ParticleSurface`` extracted into preallocated
    buffers: ``ExtractionMesh.to_arrays`` records its surface, so the
    exact-length slices an example logs are drawn from the buffers at
    their capacity (``log_particle_surface``), not re-uploaded as
    their size changes."""
    try:
        from newton.geometry import ParticleSurface  # noqa: PLC0415
    except ImportError:
        return
    extraction = getattr(ParticleSurface, "ExtractionMesh", None)
    if extraction is None or getattr(
        extraction, "_ernie_tracked", False
    ):
        return
    to_arrays = extraction.to_arrays

    def tracked(self):
        if (
            self.vertices is not None
            and _surface_counts(self) is not None
        ):
            _extractions[self.vertices.ptr] = self
        return to_arrays(self)

    extraction.to_arrays = tracked
    extraction._ernie_tracked = True


class _Batch:
    """A PointInstancer of a batch of instances, with its prototype
    and per-instance arrays."""

    def __init__(self, instancer: ernie.PointInstancer) -> None:
        self.instancer = instancer
        self.mesh = ""
        self.prototype: ernie.Mesh | None = None
        self.positions: wp.array | None = None
        self.orientations: wp.array | None = None
        self.colors: wp.array | None = None  # linear RGB
        self.ones: wp.array | None = None


class ViewerErnie(ViewerGL):
    """``ViewerGL`` with its 3D view rendered by Ernie, taking
    ViewerGL's arguments::

        viewer = ViewerErnie()
        viewer.set_model(model)
        while viewer.is_running():
            if viewer.should_step():
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0
            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()

    Args:
        scene: The Ernie scene to add the model to, with your own
            prims (terrain, lights, atmosphere) and its own lighting;
            a new one under Earth's sky and Sun when None.
        scene_setup: Called with the scene after each ``set_model``,
            to add prims or set Ernie attributes on the model's.
        root: The prim the model goes under, turned so the model's
            up axis is Ernie's +Z.
        sky: For a new scene, "gradient", Newton's sky as a dome
            lighting the scene, or "atmosphere", Earth's, physical and
            slower.
        bands: Spectral bands the view renders with: 3 for an
            interactive view, Ernie's 16 for a sensor's.
        shading: "minimal", diffuse colours lit by the sun with its
            shadows, the ambient and the sky, with highlights and
            emission (Ernie's minimal mode, ``diffuseGlossyEmission``,
            about four times faster), or "full", Ernie's spectral
            shading and sensor.
        render_scale: The view's resolution against the window's,
            which shows it stretched: below 1 for speed.
        pipelined: Whether each frame shows the last frame's render
            while rendering its own, so rendering overlaps the next
            frame's simulation, a frame behind.
        **kwargs: ViewerGL's arguments (width, height, vsync,
            headless, paused, num_frames, ...).
    """

    def __init__(
        self,
        *args: Any,
        scene: ernie.Scene | None = None,
        scene_setup: Callable[[ernie.Scene], None] | None = None,
        root: str = "/World/Newton",
        bands: int = 16,
        shading: str = "minimal",
        sky: str = "gradient",
        render_scale: float = 1.0,
        pipelined: bool = True,
        **kwargs: Any,
    ) -> None:
        self._ernie_ready = (
            False  # ViewerGL.__init__ logs nothing yet
        )
        super().__init__(*args, **kwargs)
        lit = scene is not None
        self._own_scene = scene is None
        self._own_lighting = False  # until _define_view lights it
        self.ernie_scene = (
            scene if scene is not None else ernie.Scene()
        )
        self.ernie_renderer = ernie.Renderer()
        self.scene_setup = scene_setup
        self.root = root
        self._rotation = np.eye(3)
        self._meshes: dict[str, _Mesh] = {}
        self._surfaces: dict[str, _Surface] = {}
        track_particle_surfaces()
        self._batches: dict[str, _Batch] = {}
        self._planes: dict[str, list[ernie.GroundPlane]] = {}
        self._points: dict[
            str, tuple[ernie.Points, wp.array | None]
        ] = {}
        self._overlays: set[str] = set()
        self._visible: dict[str, bool] = {}
        self._mirrored: set[str] = set()  # batches logged this frame
        # Each wireframe's (vertex data, world matrix), logged apart.
        self._wireframes: dict[str, tuple[Any, np.ndarray]] = {}
        self.render_scale = render_scale
        self.pipelined = pipelined
        self._shading = "full"  # the settings' default until set
        self._mode = "off"  # the minimal mode the settings hold
        # The render under way, and the image last shown, kept until
        # the next is.
        self._pending: Any = None
        self._shown: Any = None
        self._define_view(lit, bands, sky)
        self.shading = shading
        self._ernie_ready = True
        # Released before the interpreter tears Warp down, even when an
        # example fails before closing the viewer.
        release = weakref.WeakMethod(self._release_ernie)
        atexit.register(lambda: (method := release()) and method())

    @property
    def shading(self) -> str:
        """How the view is shaded: "minimal" or "full" (the
        constructor's ``shading``)."""
        return self._shading

    @shading.setter
    def shading(self, shading: str) -> None:
        if shading not in _SHADING:
            raise ValueError(
                f"shading must be one of {', '.join(_SHADING)}, not {shading!r}"
            )
        self._shading = shading
        self._apply_shading()

    def _apply_shading(self) -> None:
        """Shades the frame as asked, but fully while the scene holds
        what minimal shading does not draw yet (point clouds such as
        Newton's particles, terrain or planets)."""
        mode = _SHADING[self._shading]
        if mode != "off":
            scene = self.ernie_scene
            visible = any(
                self._visible.get(str(path), True)
                for path in scene.prims("Points")
            )
            if visible or scene.prims("Terrain"):
                mode = "off"
        if mode == self._mode:
            return
        self._mode = mode
        self._settings.set("ernie:minimal:mode", mode)
        if self._own_lighting:
            minimal = mode != "off"
            self._settings.ambient = (
                _MINIMAL_AMBIENT_LUMINANCE
                if minimal
                else _AMBIENT_LUMINANCE
            )
            self._settings.ambient_color = (
                _MINIMAL_AMBIENT if minimal else _AMBIENT
            )

    def close(self) -> None:
        """Closes the window, first releasing what Ernie holds of
        Newton's arrays (`_release_ernie`)."""
        self._release_ernie()
        super().close()

    def _release_ernie(self) -> None:
        """Releases Ernie's renderer and the prims and overlays holding
        Newton's arrays, and the scene if the viewer made it: Ernie
        frees a Warp array through Warp, which the interpreter tears
        down before it collects what is left. On close, or at exit
        when an example fails before closing."""
        if not getattr(self, "_ernie_ready", False):
            return
        self._ernie_ready = False
        self._pending = self._shown = None
        self.ernie_renderer = None
        for name in self._overlays:
            self.ernie_scene.remove_overlay(name)
        if self.ernie_scene.has(self.root):
            self.ernie_scene.remove(self.root)
        self._meshes.clear()
        self._surfaces.clear()
        self._batches.clear()
        self._points.clear()
        self._wireframes.clear()
        if self._own_scene:
            self.ernie_scene = None
        gc.collect()

    def _define_view(self, lit: bool, bands: int, sky: str) -> None:
        """The camera and render settings Ernie renders Newton's view
        with, and for a new scene the Sun and a sky: Newton's gradient
        or Earth's atmosphere."""
        scene = self.ernie_scene
        # Exposed for sunlight (sunny 16: f/16, 1/100 s, ISO 100); a
        # metered exposure keys to Newton's dark ground and burns out
        # the shapes.
        self._camera = scene.define(
            "/NewtonView/Camera",
            ernie.Camera,
            exposure_mode="manual",
            exposure_f_stop=16.0,
            exposure_time=0.01,
            exposure_iso=100.0,
            spectral_bands=bands,
        )
        self._settings = scene.define(
            "/NewtonView/Settings",
            ernie.RenderSettings,
            camera=self._camera.path,
            overlays="*",
        )
        self._wireframe = ""  # the settings' wireframe mode, once set
        if lit:
            return
        if sky == "atmosphere":
            # Earth's sky: its air, ozone and haze lit by the Sun
            # above them, with its measured spectrum and width.
            scene.define(
                "/World/Air", ernie.Atmosphere, preset=tokens.EARTH
            )
        else:
            # Newton's sky: a dome of its gradient behind the scene,
            # lighting none of it, and as Newton a neutral ambient
            # light filling the shadows.
            dome = scene.define(
                "/World/Sky",
                ernie.DomeLight,
                photometric_illuminance=_SKY_LUX,
                shadow_enable=False,
            )
            dome.set("collection:lightLink:includeRoot", False)
            scene.write(dome.path, "ernie:texture", _sky_gradient())
            self._settings.dome_light_mode = "approximate"
            self._settings.ambient = _AMBIENT_LUMINANCE
            self._settings.ambient_color = _AMBIENT
            self._own_lighting = True
        scene.define(
            "/World/Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.4, -0.6, 0.7)),
        )

    # The frame: Ernie renders the view, ViewerGL shows it under its
    # GUI.

    @property
    def _stream(self) -> int | None:
        return (
            self.device.stream.cuda_stream
            if self.device.is_cuda
            else None
        )

    def end_frame(self) -> None:
        if not self.renderer.has_exit():
            self.log_image("ernie", self._render(), fullscreen=True)
        super().end_frame()

    def _render(self) -> wp.array:
        """The image to show: with `pipelined`, the last frame's
        render, once this frame's has started; else this frame's."""
        self._aim_view()
        self._apply_shading()
        operation = self.ernie_renderer.step_async(
            self.ernie_scene, settings=self._settings.path
        )
        shown = operation
        if self.pipelined:
            shown, self._pending = self._pending, operation
            if shown is None:  # the first frame
                shown = operation
        # Kept until the next image is shown: the GL copy of it may
        # still be queued.
        self._shown = self._image_of(shown)
        return self._shown

    def _image_of(self, operation: Any) -> wp.array:
        """A render's LdrColor in CUDA memory, ready on Warp's
        stream."""
        path = str(self._settings.path)
        frame = operation.outputs()[path].frames[0]
        image = frame.render_vars[f"{path}/LdrColor"].map(
            device=ernie.Device.CUDA, sync_stream=self._stream
        )
        return wp.from_dlpack(image)

    def _aim_view(self) -> None:
        """The view's camera on Newton's, at its window's size times
        `render_scale`, and its wireframe as the GUI asks."""
        width = max(
            1, round(self.renderer._screen_width * self.render_scale)
        )
        height = max(
            1, round(self.renderer._screen_height * self.render_scale)
        )
        camera = self.camera
        eye = self._rotation @ np.asarray(camera.pos, np.float64)
        front = self._rotation @ np.asarray(camera.get_front())
        up = self._rotation @ np.asarray(camera.get_up())
        self._camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, eye + front, up),
        )
        aperture = 24.0  # mm; the focal length gives Newton's fov
        self._camera.vertical_aperture = aperture
        self._camera.horizontal_aperture = aperture * width / height
        self._camera.focal_length = (
            0.5 * aperture / np.tan(np.radians(camera.fov) / 2.0)
        )
        self._camera.clipping_range = (camera.near, camera.far)
        self._settings.resolution = (width, height)
        # The GUI's Wireframe checkbox: edges over the shading.
        wireframe = (
            "overlay" if self.renderer.draw_wireframe else "off"
        )
        if wireframe != self._wireframe:
            self._wireframe = wireframe
            self._settings.wireframe = wireframe

    # The model.

    def clear_model(self) -> None:
        if getattr(self, "_ernie_ready", False):
            scene = self.ernie_scene
            scene.remove(ernie.Path(self.root))
            for name in self._overlays:
                scene.remove_overlay(name)
            for collection in (
                self._meshes,
                self._batches,
                self._planes,
                self._points,
                self._visible,
                self._wireframes,
            ):
                collection.clear()
            self._overlays.clear()
        super().clear_model()

    def set_model(self, model, *args: Any, **kwargs: Any) -> None:
        if model is not None:  # before the shapes' meshes, logged now
            self._rotation, rotate = _up_rotation(int(model.up_axis))
            self.ernie_scene.define(
                self.root, ernie.Xform, rotate=rotate
            )
        super().set_model(model, *args, **kwargs)
        if model is not None and self.scene_setup is not None:
            self.scene_setup(self.ernie_scene)

    # Geometry, logged to ViewerGL and mirrored in the Ernie scene.

    def _write(self, prim: Any, attribute: str, array: Any) -> None:
        cuda = isinstance(array, wp.array) and array.device.is_cuda
        self.ernie_scene.write(
            prim.path,
            attribute,
            array,
            stream=self._stream if cuda else None,
        )

    def _show(self, prim: Any, visible: bool) -> None:
        """Sets the prim's visibility when it changes."""
        path = str(prim.path)
        if self._visible.get(path) != visible:
            self._visible[path] = visible
            prim.visibility = "inherited" if visible else "invisible"

    def _write_mesh(
        self, mesh: _Mesh, prim: ernie.Mesh, topology: bool = True
    ) -> None:
        self._write(prim, "points", mesh.points)
        if topology:
            if mesh.counts is not None:
                self._write(prim, "faceVertexCounts", mesh.counts)
            self._write(prim, "faceVertexIndices", mesh.indices)
        if mesh.normals is not None:
            self._write(prim, "normals", mesh.normals)

    def log_mesh(
        self,
        name: str,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture: Any = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        dynamic: bool = False,
        opacity: float | None = None,
    ) -> None:
        super().log_mesh(
            name,
            points,
            indices,
            normals,
            uvs,
            texture,
            hidden,
            backface_culling,
            color,
            roughness,
            metallic,
            dynamic,
            opacity,
        )
        if not self._ernie_ready:
            return
        # A particle surface's slices: drawn from its whole buffers.
        surface = (
            _extractions.get(points.ptr)
            if isinstance(points, wp.array) and points.ptr
            else None
        )
        if surface is not None and surface.vertices.ptr == points.ptr:
            self.log_particle_surface(
                name,
                surface,
                hidden=hidden,
                color=color,
                used=(len(points), len(indices)),
            )
            return
        if self._qualify(name) in self._surfaces:
            # Logged empty, as examples do when it has no triangles.
            self._show(
                self._surfaces[self._qualify(name)].prim, False
            )
            if hidden:
                return
        name = self._qualify(name)
        mesh = self._meshes.setdefault(name, _Mesh())
        # Triangles written as Newton's array, without counts on the
        # GPU; again when its buffer changes, or every time for
        # dynamic topology.
        source = (indices.ptr, len(indices))
        topology = mesh.source != source or dynamic
        if topology:
            mesh.source = source
            mesh.indices = indices.flatten()
            mesh.counts = (
                None
                if indices.device.is_cuda
                else np.full(len(mesh.indices) // 3, 3, np.int32)
            )
        mesh.points, mesh.normals = points, normals
        if color is not None:
            mesh.color = _linear(color)
        if mesh.prim is None and not hidden:
            mesh.prim = self.ernie_scene.define(
                f"{self.root}/Meshes/{_identifier(name)}", ernie.Mesh
            )
            mesh.prim.display_color = mesh.color
            topology = True
        if mesh.prim is not None:
            self._show(mesh.prim, not hidden)
            self._write_mesh(mesh, mesh.prim, topology)
        for batch in self._batches.values():
            if batch.mesh == name and batch.prototype is not None:
                self._write_mesh(mesh, batch.prototype, topology)

    def log_particle_surface(
        self,
        name: str,
        surface: Any,
        *,
        hidden: bool = False,
        color: tuple[float, float, float] | None = None,
        used: tuple[int, int] | None = None,
    ) -> None:
        """Draws a surface ``newton.geometry.ParticleSurface`` extracted
        into preallocated buffers (``max_grid_cells`` set), its
        ``ExtractionMesh``, from those buffers at their capacity: the
        vertex and index counts stay on the GPU as the mesh's
        ``ernie:activeCounts``, so nothing waits for the GPU and nothing
        is re-uploaded as the surface changes size. Its bounds are
        measured on the GPU.

        Only the first of the buffers' entries are copied each frame: a
        power of two above the counts lately, which ``used`` (vertices
        and indices) gives when known on the host, else the counts read
        back frames late. Until it grows, triangles past it are not
        drawn.

        Raises:
            ValueError: for a surface of exact-sized buffers.
        """
        if not self._ernie_ready:
            return
        counts = _surface_counts(surface)
        if counts is None:
            raise ValueError(
                "log_particle_surface needs a surface extracted into "
                "preallocated buffers (ParticleSurface(max_grid_cells=...))"
            )
        name = self._qualify(name)
        state = self._surfaces.get(name)
        if state is None:
            prim = self.ernie_scene.define(
                f"{self.root}/Surfaces/{_identifier(name)}",
                ernie.Mesh,
            )
            prim.display_color = _linear(color or (0.7, 0.7, 0.7))
            state = self._surfaces[name] = _Surface(prim)
        elif color is not None:
            state.prim.display_color = _linear(color)
        self._show(state.prim, not hidden)
        if hidden or surface.vertices is None:
            return
        state.surface = surface
        if used is not None:
            state.fit(*used, surface)
        elif counts.device.is_cuda:
            if state.points == 0:
                # Its size, once, before any counts come back.
                host = counts.numpy()
                state.fit(int(host[0]), int(host[1]), surface)
            state.read(counts, self.device.stream)
        else:
            state.fit(
                int(counts.numpy()[0]),
                int(counts.numpy()[1]),
                surface,
            )
        self._write(
            state.prim, "points", surface.vertices[: state.points]
        )
        self._write(
            state.prim,
            "faceVertexIndices",
            surface.indices[: state.indices],
        )
        if surface.normals is not None:
            self._write(
                state.prim, "normals", surface.normals[: state.points]
            )
        self._write(state.prim, "ernie:activeCounts", counts)

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array | None,
        scales: wp.array | None,
        colors: wp.array | None,
        materials: wp.array | None,
        hidden: bool = False,
        opacities: wp.array | None = None,
    ) -> None:
        super().log_instances(
            name,
            mesh,
            xforms,
            scales,
            colors,
            materials,
            hidden=hidden,
            opacities=opacities,
        )
        if self._ernie_ready:
            self._mirror_instances(
                self._qualify(name),
                self._qualify(mesh),
                xforms,
                scales,
                colors,
                materials,
                hidden,
            )

    def log_state(self, state) -> None:
        """ViewerGL's own, then the shape batches its packed CUDA path
        posed without ``log_instances``, mirrored from their world
        transforms as ViewerBase computes them."""
        changed = self.model_changed
        self._mirrored.clear()
        super().log_state(state)
        if self.model is None:
            return
        hidden_layer = self._layer_force_hidden()
        for shapes in self._shape_instances.values():
            name = self._qualify(shapes.name)
            # Capsules are cylinders and caps, through log_instances.
            if (
                name in self._mirrored
                or shapes.geo_type == newton.GeoType.CAPSULE
            ):
                continue
            visible = (
                self._should_show_shape(
                    shapes.flags, shapes.static, shapes.geo_type
                )
                and not hidden_layer
            )
            if visible:
                shapes.update(
                    state,
                    world_offsets=self.world_offsets,
                    layer_xform=self.layer.xform,
                )
            new = (
                name not in self._batches and name not in self._planes
            )
            self._mirror_instances(
                name,
                self._qualify(shapes.mesh),
                shapes.world_xforms,
                shapes.scales,
                shapes.colors if changed or new else None,
                shapes.materials if changed or new else None,
                not visible,
            )

    def _mirror_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array | None,
        scales: wp.array | None,
        colors: wp.array | None,
        materials: wp.array | None,
        hidden: bool,
    ) -> None:
        """A batch of instances in the Ernie scene."""
        self._mirrored.add(name)
        count = len(xforms) if xforms is not None else 0
        if name in self._plane_batches():
            self._log_planes(name, xforms, colors, materials, hidden)
            return
        batch = self._batches.get(name)
        if batch is None:
            batch = _Batch(
                self.ernie_scene.define(
                    f"{self.root}/Instancers/{_identifier(name)}",
                    ernie.PointInstancer,
                )
            )
            self._batches[name] = batch
        self._show(batch.instancer, not hidden and count > 0)
        if count == 0:
            return
        device = xforms.device
        if batch.positions is None or len(batch.positions) != count:
            batch.positions = wp.empty(count, wp.vec3, device=device)
            batch.orientations = wp.empty(
                count, wp.vec4, device=device
            )
            batch.colors = wp.full(
                count,
                wp.vec3(*self._meshes[mesh].color),
                device=device,
            )
            batch.ones = wp.full(
                count, wp.vec3(1.0, 1.0, 1.0), device=device
            )
            self._write(
                batch.instancer, "primvars:displayColor", batch.colors
            )
        if batch.mesh != mesh:
            batch.mesh = mesh
            self._prototype(batch)
        if colors is not None:  # only when they change
            wp.launch(
                _linear_colors,
                count,
                inputs=[colors],
                outputs=[batch.colors],
                device=device,
            )
            self._write(
                batch.instancer, "primvars:displayColor", batch.colors
            )
        wp.launch(
            _split_transforms,
            count,
            inputs=[xforms],
            outputs=[batch.positions, batch.orientations],
            device=xforms.device,
        )
        instancer = batch.instancer
        self._write(instancer, "positions", batch.positions)
        self._write(instancer, "orientations", batch.orientations)
        self._write(
            instancer,
            "scales",
            scales if scales is not None else batch.ones,
        )

    def _prototype(self, batch: _Batch) -> None:
        """The batch's prototype: a Mesh sharing the logged mesh's
        arrays, coloured per instance by the instancer."""
        if batch.prototype is None:
            batch.prototype = self.ernie_scene.define(
                f"{batch.instancer.path}/Prototype", ernie.Mesh
            )
            batch.instancer.prototypes = [batch.prototype]
        self._write_mesh(self._meshes[batch.mesh], batch.prototype)

    def _plane_batches(self) -> set[str]:
        """The qualified names of the model's batches of planes."""
        return {
            self._qualify(shapes.name)
            for shapes in self._shape_instances.values()
            if shapes.geo_type == newton.GeoType.PLANE
        }

    def _log_planes(
        self,
        name: str,
        xforms: wp.array | None,
        colors: wp.array | None,
        materials: wp.array | None,
        hidden: bool,
    ) -> None:
        """Ground planes as GroundPlanes, infinite like Newton's, with
        its checker; posed when Newton passes their colours (on a new
        model), as they do not move."""
        planes = self._planes.setdefault(name, [])
        for plane in planes:
            self._show(plane, not hidden)
        if colors is None or xforms is None:
            return
        poses = xforms.numpy().reshape(-1, 7)
        rgb = colors.numpy().reshape(-1, 3)
        checker = (
            materials.numpy().reshape(-1, 4)[:, 2] > 0.0
            if materials is not None
            else np.zeros(len(poses), bool)
        )
        for i, pose in enumerate(poses):
            if i == len(planes):
                planes.append(
                    self.ernie_scene.define(
                        f"{self.root}/Planes/{_identifier(name)}_{i}",
                        ernie.GroundPlane,
                    )
                )
            plane = planes[i]
            plane.set_xform(
                translate=tuple(float(v) for v in pose[:3]),
                orient=ernie.Quat.from_xyzw(
                    *(float(v) for v in pose[3:])
                ),
            )
            # Newton's ground colour as albedo, not linearised: about
            # 1% albedo otherwise, near black under sunlight, where
            # ViewerGL shows it mid grey.
            colour = tuple(
                float(c) for c in rgb[min(i, len(rgb) - 1)]
            )
            darken = (
                _CHECKER_DARKEN
                if checker[min(i, len(checker) - 1)]
                else 1.0
            )
            plane.checker = True  # its colours are the plane's albedo
            plane.checker_color0 = colour
            plane.checker_color1 = tuple(darken * c for c in colour)
            self._show(plane, not hidden)

    def log_lines(
        self,
        name: str,
        starts: wp.array | None,
        ends: wp.array | None,
        colors: Any,
        width: float = 0.01,
        hidden: bool = False,
    ) -> None:
        super().log_lines(name, starts, ends, colors, width, hidden)
        if self._ernie_ready:
            self._overlay_segments(name, starts, ends, colors, hidden)

    def log_arrows(
        self,
        name: str,
        starts: wp.array | None,
        ends: wp.array | None,
        colors: Any,
        width: float = 0.01,
        hidden: bool = False,
    ) -> None:
        super().log_arrows(name, starts, ends, colors, width, hidden)
        if self._ernie_ready:
            self._overlay_segments(
                name, starts, ends, colors, hidden, arrows=True
            )

    def _overlay_segments(
        self,
        name: str,
        starts: wp.array | None,
        ends: wp.array | None,
        colors: Any,
        hidden: bool,
        arrows: bool = False,
    ) -> None:
        """Lines or arrows from Newton's model frame, in a scene
        overlay of their own, redrawn each time they are logged: one
        batch drawn from Newton's arrays as they are (`Overlay.
        segments` and `arrows`), turned on the GPU for a model not
        Z-up."""
        overlay = self._overlay(name)
        if (
            hidden
            or starts is None
            or ends is None
            or len(starts) == 0
        ):
            return
        starts, ends = self._in_world(starts), self._in_world(ends)
        style: dict[str, Any] = {}
        rgb = None
        if isinstance(colors, wp.array):
            rows = (
                len(colors)
                if colors.dtype == wp.vec3
                else len(colors) // 3
            )
            if rows == len(starts):
                rgb = colors
            else:  # one colour for all
                style["color"] = tuple(
                    float(c) for c in colors.numpy().reshape(-1)[:3]
                )
        elif colors is not None:
            style["color"] = tuple(float(c) for c in colors)
        draw = overlay.arrows if arrows else overlay.segments
        draw(
            starts,
            ends,
            colors=rgb,
            stream=self._stream if starts.device.is_cuda else None,
            **style,
        )

    def _in_world(self, points: wp.array) -> wp.array:
        """Model-frame points in the world, turned on their device
        unless the model is Z-up."""
        if np.array_equal(self._rotation, np.eye(3)):
            return points
        turned = wp.empty(len(points), wp.vec3, device=points.device)
        wp.launch(
            _rotate_points,
            len(points),
            inputs=[points, wp.mat33(*self._rotation.reshape(-1))],
            outputs=[turned],
            device=points.device,
        )
        return turned

    def _overlay(self, name: str) -> Any:
        """The scene overlay of a Newton batch of lines, emptied."""
        overlay_name = "newton" + self._qualify(name)
        self._overlays.add(overlay_name)
        return self.ernie_scene.overlay(overlay_name).clear()

    def log_wireframe_shape(
        self,
        name: str,
        vertex_data: np.ndarray | None,
        world_matrix: np.ndarray | None,
        hidden: bool = False,
    ) -> None:
        """A shape's edges, (position, colour) vertex pairs in its
        frame, placed by a column-major 4 x 4 world matrix; either
        None keeps the last."""
        super().log_wireframe_shape(
            name, vertex_data, world_matrix, hidden
        )
        if not self._ernie_ready:
            return
        name = self._qualify(name)
        last = self._wireframes.get(name, (None, np.eye(4)))
        data = vertex_data if vertex_data is not None else last[0]
        matrix = (
            np.asarray(world_matrix, np.float64)
            .reshape(-1)
            .reshape(4, 4, order="F")
            if world_matrix is not None
            else last[1]
        )
        self._wireframes[name] = (data, matrix)
        overlay = self._overlay(name)
        if hidden or data is None or len(data) == 0:
            return
        data = np.asarray(data, np.float64).reshape(-1, 6)
        points = data[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
        points = points @ self._rotation.T
        colours = data[0::2, 3:]
        overlay.lines(points.reshape(-1, 2, 3), colours)

    def log_points(
        self,
        name: str,
        points: wp.array | None,
        radii: wp.array | float | None = None,
        colors: Any = None,
        hidden: bool = False,
    ) -> None:
        super().log_points(name, points, radii, colors, hidden)
        if not self._ernie_ready:
            return
        name = self._qualify(name)
        entry = self._points.get(name)
        if entry is None:
            prim = self.ernie_scene.define(
                f"{self.root}/Points/{_identifier(name)}",
                ernie.Points,
            )
            entry = (prim, None)
        prim, widths = entry
        count = len(points) if points is not None else 0
        self._show(prim, not hidden and count > 0)
        if count == 0:
            self._points[name] = entry
            return
        self._write(prim, "points", points)
        if isinstance(radii, wp.array):
            if widths is None or len(widths) != len(radii):
                widths = wp.empty(
                    len(radii), wp.float32, radii.device
                )
            wp.launch(
                _diameters,
                len(radii),
                inputs=[radii],
                outputs=[widths],
                device=radii.device,
            )
            self._write(prim, "widths", widths)
        elif radii is not None:
            prim.widths = [2.0 * float(radii)]
        if isinstance(colors, wp.array):
            if len(colors) == 1:
                prim.display_color = _linear(
                    colors.numpy().reshape(3)
                )
            else:
                self._write(prim, "primvars:displayColor", colors)
        elif colors is not None:
            prim.display_color = _linear(colors)
        self._points[name] = (prim, widths)
