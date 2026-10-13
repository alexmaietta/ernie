# SPDX-License-Identifier: BSD-3-Clause
"""A simulation baked into time samples, then played on the timeline:
boxes thrown up and bouncing (a PointInstancer's positions and
orientations), a flag waving (a subdivision surface's points), a camera
zooming (its focal length and transform) and a light pulsing (its
intensity). Warp integrates on CUDA where it can, and every frame's
arrays go into the scene as samples on the GPU; the renderer blends the
two either side of the time code there, so scrubbing costs no copies."""

import math

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

FRAMES = 120  # time codes 0 to FRAMES, 30 a second
SUBSTEPS = 4
BODIES = 48
HALF = 0.1  # the boxes' half size, metres
GRAVITY = 9.81
FLAG = (24, 12)  # quads along and up the flag
FLAG_ORIGIN = (-2.5, 1.5, 1.0)  # its pole's foot of cloth, metres


def hash01(i, k):
    """A float in [0, 1) from two ints, as the C++ twin's."""
    x = (i * 7919 + k * 104729) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x / 4294967296.0


def launch():
    """Each box's starting position, velocity and spin (float32)."""
    p, v, w = [], [], []
    for i in range(BODIES):
        a = 2.0 * math.pi * hash01(i, 0)
        speed = 1.0 + 2.0 * hash01(i, 1)
        up = 3.0 + 3.0 * hash01(i, 2)
        p.append(
            (
                0.2 * math.cos(a),
                0.2 * math.sin(a),
                0.6 + 0.05 * (i % 8),
            )
        )
        v.append((speed * math.cos(a), speed * math.sin(a), up))
        w.append(
            tuple(8.0 * (hash01(i, 3 + c) - 0.5) for c in range(3))
        )
    return (np.array(x, np.float32) for x in (p, v, w))


@wp.kernel
def step(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.vec4),
    spins: wp.array(dtype=wp.vec3),
    dt: float,
    restitution: float,
):
    # Ballistic with a bouncing floor; the spin turns the box (as the
    # C++ twin's `Integrate`).
    i = wp.tid()
    v = velocities[i]
    v = wp.vec3(v[0], v[1], v[2] - 9.81 * dt)
    p = positions[i] + v * dt
    w = spins[i]
    if p[2] < 0.1:
        p = wp.vec3(p[0], p[1], 0.1)
        if v[2] < 0.0:
            v = wp.vec3(0.8 * v[0], 0.8 * v[1], -restitution * v[2])
            w = 0.8 * w
    q = orientations[i]
    h = 0.5 * dt
    x = q[0] + h * (w[0] * q[3] + w[1] * q[2] - w[2] * q[1])
    y = q[1] + h * (w[1] * q[3] + w[2] * q[0] - w[0] * q[2])
    z = q[2] + h * (w[2] * q[3] + w[0] * q[1] - w[1] * q[0])
    s = q[3] + h * (-w[0] * q[0] - w[1] * q[1] - w[2] * q[2])
    n = wp.sqrt(x * x + y * y + z * z + s * s)
    positions[i] = p
    velocities[i] = v
    orientations[i] = wp.vec4(x / n, y / n, z / n, s / n)
    spins[i] = w


@wp.kernel
def wave(
    rest: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    t: float,
):
    # The flag's cloth at time t (seconds), fixed at its pole (x = 0).
    i = wp.tid()
    r = rest[i]
    u = r[0] / 2.0
    y = 0.18 * u * wp.sin(5.0 * r[0] - 7.0 * t) + 0.05 * u * wp.sin(
        4.0 * r[2] + 5.0 * t
    )
    points[i] = wp.vec3(r[0], y, r[2])


def flag_mesh():
    """The flag's rest points (x along, z up), 2 by 1 metres, and its quads."""
    nx, nz = FLAG
    points = [
        (2.0 * i / nx, 0.0, 1.0 * j / nz)
        for j in range(nz + 1)
        for i in range(nx + 1)
    ]
    indices = []
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            indices += [a, a + 1, a + nx + 2, a + nx + 1]
    return np.array(points, np.float32), np.array(indices, np.int32)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        stage = ernie.Stage.at(scene, "/")
        stage.start_time_code = 0.0
        stage.end_time_code = float(FRAMES)
        stage.time_codes_per_second = 30.0
        stage.frames_per_second = 30.0
        world.define(
            "Sun", ernie.DistantLight, angle=0.53, shadow_enable=True
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (2.0, -3.0, 4.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            ),
        )
        self.beacon = world.define(
            "Beacon", ernie.SphereLight, radius=0.1
        )
        self.beacon.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            ),
        )
        scene.define(
            "/Looks/Floor", ernie.Material, albedo=(0.5, 0.5, 0.5)
        )
        scene.define(
            "/Looks/Box", ernie.Material, albedo=(0.8, 0.35, 0.1)
        )
        scene.define(
            "/Looks/Flag", ernie.Material, albedo=(0.1, 0.3, 0.8)
        )
        world.define(
            "Floor",
            ernie.GroundPlane,
            checker=True,
            checker_size=1.0,
            material_binding="/Looks/Floor",
        )
        bodies = world.define("Bodies", ernie.PointInstancer)
        bodies.define(
            "Box",
            ernie.Cube,
            size=2.0 * HALF,
            material_binding="/Looks/Box",
        )
        scene.set(
            bodies.path, tokens.PROTOTYPES, [bodies.path.child("Box")]
        )
        world.define(
            "Pole",
            ernie.Cylinder,
            radius=0.03,
            height=2.2,
            material_binding="/Looks/Floor",
        ).set(translate=(FLAG_ORIGIN[0], FLAG_ORIGIN[1], 1.1))
        rest, indices = flag_mesh()
        self.flag = world.define(
            "Flag",
            ernie.Mesh,
            subdivision_scheme="catmullClark",
            material_binding="/Looks/Flag",
        )
        self.flag.set(translate=FLAG_ORIGIN)
        self.flag.write(tokens.POINTS, rest)
        self.flag.write(
            tokens.FACE_VERTEX_COUNTS,
            np.full(len(indices) // 4, 4, np.int32),
        )
        self.flag.write(tokens.FACE_VERTEX_INDICES, indices)
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            clipping_range=(0.05, 100.0),
            sensor_noise=False,
            exposure_mode=tokens.AUTO,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(960, 540),
            ambient=0.1,
        )
        self.restitution = 0.6
        self.bake()

        # Re-bakes the simulation with the boxes' bounce.
        self.widgets.slider(
            "Restitution",
            0.1,
            0.9,
            self.restitution,
            on_change=self.rebake,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.play()

    def rebake(self, restitution):
        self.restitution = restitution
        self.bake()

    def bake(self):
        """Simulates every frame and writes them as time samples: the
        arrays as Warp made them, on the GPU where it runs."""
        wp.init()
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = self.scene
        bodies = ernie.Path("/World/Bodies")
        rest, _ = flag_mesh()
        with wp.ScopedDevice(device):
            p, v, w = launch()
            positions = wp.array(p, dtype=wp.vec3)
            velocities = wp.array(v, dtype=wp.vec3)
            spins = wp.array(w, dtype=wp.vec3)
            orientations = wp.array(
                np.tile(
                    np.array([0, 0, 0, 1], np.float32), (BODIES, 1)
                ),
                dtype=wp.vec4,
            )
            flag_rest = wp.array(rest, dtype=wp.vec3)
            stream = (
                wp.get_stream(device) if device != "cpu" else None
            )
            dt = 1.0 / (30.0 * SUBSTEPS)
            for frame in range(FRAMES + 1):
                if frame > 0:
                    for _ in range(SUBSTEPS):
                        wp.launch(
                            step,
                            BODIES,
                            inputs=[
                                positions,
                                velocities,
                                orientations,
                                spins,
                                dt,
                                self.restitution,
                            ],
                        )
                points = wp.empty(len(rest), dtype=wp.vec3)
                wp.launch(
                    wave,
                    len(rest),
                    inputs=[flag_rest, points, frame / 30.0],
                )
                time = float(frame)
                # Each frame's own copies, kept by the scene as samples.
                scene.write(
                    bodies,
                    tokens.POSITIONS,
                    wp.clone(positions),
                    stream=stream,
                    time=time,
                )
                scene.write(
                    bodies,
                    tokens.ORIENTATIONS,
                    wp.clone(orientations),
                    stream=stream,
                    time=time,
                )
                scene.write(
                    self.flag.path,
                    tokens.POINTS,
                    points,
                    stream=stream,
                    time=time,
                )
            wp.synchronize_device(device)
        # Values sampled sparsely, interpolated between: the camera
        # orbits and zooms in, the beacon pulses.
        for frame in range(0, FRAMES + 1, 10):
            t = frame / FRAMES
            angle = 0.35 + 0.5 * t
            eye = (
                8.0 * math.cos(angle),
                -8.0 * math.sin(angle),
                3.0 - 0.8 * t,
            )
            self.camera.set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    eye, (0.0, 0.3, 0.6), (0.0, 0.0, 1.0)
                ),
                time=float(frame),
            )
            self.camera.set(
                tokens.FOCAL_LENGTH,
                18.0 + 22.0 * t,
                time=float(frame),
            )
            self.beacon.set(
                tokens.INTENSITY,
                40.0 if (frame // 10) % 2 == 0 else 5.0,
                time=float(frame),
            )

    def render_at(self, times):
        """The scene at each time code, RGBA uint8; leaves the time as it
        was."""
        stage = ernie.Stage.at(self.scene, "/")
        current = stage.time_code
        images = []
        for t in times:
            stage.time_code = float(t)
            images.append(np.array(self.renderer.render(self.scene)))
        stage.time_code = current
        return images

    def save_output(self, path):
        """Four moments of the simulation side by side."""
        write_png(
            path, side_by_side(self.render_at([0, 20, 60, FRAMES]))
        )

    def test_final(self):
        scene = self.scene
        animated = {str(p) for p in scene.animated_prims()}
        print("animated:", sorted(animated))
        for prim in (
            "/World/Bodies",
            "/World/Flag",
            "/World/Camera",
            "/World/Beacon",
        ):
            assert prim in animated, f"{prim} should be sampled"
        assert (
            len(scene.sample_times("/World/Bodies", tokens.POSITIONS))
            == FRAMES + 1
        )
        # Values between samples interpolate.
        focal = scene.get_at(
            self.camera.path, tokens.FOCAL_LENGTH, 5.0
        )
        assert abs(focal - (18.0 + 22.0 * 5.0 / FRAMES)) < 1e-6, focal
        start, early, middle = self.render_at([0, 12, 60])
        changed = (
            np.abs(start.astype(int) - middle.astype(int)).max(-1)
            > 16
        ).mean()
        print(
            f"pixels changed by frame 60: {changed:.3f}; mean {middle[..., :3].mean():.1f}"
        )
        assert changed > 0.02, "the scene should move"
        assert (
            min(im[..., :3].mean() for im in (start, early, middle))
            > 5.0
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
