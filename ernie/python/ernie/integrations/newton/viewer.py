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

import re
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
        **kwargs: ViewerGL's arguments (width, height, vsync,
            headless, paused, num_frames, ...).
    """

    def __init__(
        self,
        *args: Any,
        scene: ernie.Scene | None = None,
        scene_setup: Callable[[ernie.Scene], None] | None = None,
        root: str = "/World/Newton",
        **kwargs: Any,
    ) -> None:
        self._ernie_ready = (
            False  # ViewerGL.__init__ logs nothing yet
        )
        super().__init__(*args, **kwargs)
        lit = scene is not None
        self.ernie_scene = (
            scene if scene is not None else ernie.Scene()
        )
        self.ernie_renderer = ernie.Renderer()
        self.scene_setup = scene_setup
        self.root = root
        self._rotation = np.eye(3)
        self._meshes: dict[str, _Mesh] = {}
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
        self._define_view(lit)
        self._ernie_ready = True

    def _define_view(self, lit: bool) -> None:
        """The camera and render settings Ernie renders Newton's view
        with, and Earth's sky and Sun for a new scene."""
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
        # Earth's sky: its air, ozone and haze lit by the Sun above
        # them, with its measured spectrum and width.
        scene.define(
            "/World/Air", ernie.Atmosphere, preset=tokens.EARTH
        )
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
        width = self.renderer._screen_width
        height = self.renderer._screen_height
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
        image = self.ernie_renderer.render(
            self.ernie_scene, self._settings.path, device="cuda"
        )
        return wp.from_dlpack(image)

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
