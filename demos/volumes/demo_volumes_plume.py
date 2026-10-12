# SPDX-License-Identifier: BSD-3-Clause
"""A rocket engine's plume in sunlight: hot gas glowing as a black body
at the nozzle, cooling to soot that shadows and dims what is behind. A
Warp kernel writes its soot and temperature on the GPU each frame,
DenseFields the renderer reads from CUDA without a copy to the host."""

from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear, write_png

# The plume's hottest, at the nozzle.
PEAK_KELVIN = 3000.0
NX, NY, NZ = 48, 48, 120
VOXEL = 0.1
PLUME = ernie.Path("/World/Plume")


@wp.func
@no_type_check
def fade(t: float):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@wp.func
@no_type_check
def gradient(hash: int, x: float, y: float, z: float):
    h = hash & 15
    u = y
    if h < 8:
        u = x
    v = z
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    if (h & 1) != 0:
        u = -u
    if (h & 2) != 0:
        v = -v
    return u + v


@wp.func
@no_type_check
def mix(t: float, a: float, b: float):
    return a + t * (b - a)


@wp.func
@no_type_check
def noise(p: wp.vec3, perm: wp.array(dtype=wp.int32)):
    """Perlin's improved noise, as ernie.fbm_noise's (GradientNoise)."""
    fx = wp.floor(p[0])
    fy = wp.floor(p[1])
    fz = wp.floor(p[2])
    x = int(fx) & 255
    y = int(fy) & 255
    z = int(fz) & 255
    gx = p[0] - fx
    gy = p[1] - fy
    gz = p[2] - fz
    u = fade(gx)
    v = fade(gy)
    w = fade(gz)
    a = perm[x] + y
    aa = perm[a] + z
    ab = perm[a + 1] + z
    b = perm[x + 1] + y
    ba = perm[b] + z
    bb = perm[b + 1] + z
    return mix(
        w,
        mix(
            v,
            mix(
                u,
                gradient(perm[aa], gx, gy, gz),
                gradient(perm[ba], gx - 1.0, gy, gz),
            ),
            mix(
                u,
                gradient(perm[ab], gx, gy - 1.0, gz),
                gradient(perm[bb], gx - 1.0, gy - 1.0, gz),
            ),
        ),
        mix(
            v,
            mix(
                u,
                gradient(perm[aa + 1], gx, gy, gz - 1.0),
                gradient(perm[ba + 1], gx - 1.0, gy, gz - 1.0),
            ),
            mix(
                u,
                gradient(perm[ab + 1], gx, gy - 1.0, gz - 1.0),
                gradient(perm[bb + 1], gx - 1.0, gy - 1.0, gz - 1.0),
            ),
        ),
    )


@wp.func
@no_type_check
def fbm(p: wp.vec3, octaves: int, perm: wp.array(dtype=wp.int32)):
    """Fractional Brownian motion, as ernie.fbm_noise's."""
    total = wp.float32(0.0)
    weight = wp.float32(0.0)
    amplitude = wp.float32(1.0)
    frequency = wp.float32(1.0)
    for _ in range(octaves):
        total += amplitude * noise(p * frequency, perm)
        weight += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return total / weight


@wp.kernel
@no_type_check
def plume(
    soot: wp.array3d(dtype=float),
    heat: wp.array3d(dtype=float),
    perm: wp.array(dtype=wp.int32),
    time: float,
):
    """A rocket engine's plume `time` seconds in, from a nozzle 1.2 m
    across at the origin down -z for 12 m, in 0.1 m voxels: widening as
    it goes, hottest at the nozzle with shock diamonds, cooling to soot
    downstream; turbulence streams down it and the core flickers. As the
    C++ twin's shader."""
    i, j, k = wp.tid()
    px = -2.4 + 0.1 * (float(i) + 0.5)
    py = -2.4 + 0.1 * (float(j) + 0.5)
    pz = -12.0 + 0.1 * (float(k) + 0.5)
    s = wp.max(-pz, 0.0)
    radius = 0.6 + 0.12 * s
    core = wp.exp(-2.0 * (px * px + py * py) / (radius * radius))
    fade_in = wp.clamp(s / 0.3, 0.0, 1.0) * wp.clamp(
        (12.0 - s) / 2.0, 0.0, 1.0
    )
    turbulence = 0.5 + 0.5 * fbm(
        wp.vec3(1.5 * px, 1.5 * py, 0.8 * (pz + 20.0 * time)), 3, perm
    )
    smoke = (
        core
        * (0.8 + 0.4 * (1.0 - wp.exp(-s / 4.0)))
        * wp.exp(-s / 14.0)
        * fade_in
        * (0.4 + 1.2 * turbulence)
    )
    diamonds = 1.0 + 0.35 * wp.cos(2.0 * wp.pi * s / 1.4) * wp.exp(
        -s / 3.0
    )
    flicker = 1.0 + 0.08 * wp.sin(2.0 * wp.pi * 12.0 * time)
    hot = (
        core
        * diamonds
        * wp.exp(-s / 5.0)
        * fade_in
        * flicker
        * (0.85 + 0.3 * turbulence)
    )
    soot[i, j, k] = wp.where(smoke > 1.0e-3, smoke, 0.0)
    heat[i, j, k] = wp.where(hot > 1.0e-3, hot, 0.0)


def mean_in(image, left, right, top, bottom):
    """The image's mean linear RGB over columns [left, right) and rows
    [top, bottom)."""
    pixels = np.asarray(image)[top:bottom, left:right, :3]
    return srgb_to_linear(pixels.reshape(-1, 3).mean(axis=0) / 255.0)


class Demo(ernie.demo.Demo):
    num_frames = 12

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("This demo needs CUDA interop")
        self.device = "cuda:0"
        world = self.scene.define("/World", ernie.Xform)
        # The rocket above its nozzle at the origin, 1.2 m across.
        rocket = world.define("Rocket", ernie.Xform)
        rocket.define(
            "Body",
            ernie.Cylinder,
            radius=0.6,
            height=6.0,
            translate=(0.0, 0.0, 3.4),
            material_binding="/World/Looks/Paint",
        )
        rocket.define(
            "Nose",
            ernie.Cone,
            radius=0.6,
            height=1.6,
            translate=(0.0, 0.0, 7.2),
            material_binding="/World/Looks/Paint",
        )
        rocket.define(
            "Nozzle",
            ernie.Cone,
            radius=0.4,
            height=0.8,
            translate=(0.0, 0.0, 0.4),
            material_binding="/World/Looks/Metal",
        )
        self.scene.define(
            "/World/Looks/Paint",
            ernie.Material,
            albedo=(0.7, 0.7, 0.7),
        )
        self.scene.define(
            "/World/Looks/Metal",
            ernie.Material,
            albedo=(0.25, 0.25, 0.25),
        )

        # The plume's soot and temperature, written on the GPU each frame
        # into dense fields whose first cell is at (-2.4, -2.4, -12) m and
        # 0.1 m apart.
        plume_prim = world.define(
            "Plume",
            ernie.Volume,
            material_binding="/World/Looks/Exhaust",
        )
        for name in ("Density", "Temperature"):
            plume_prim.define(
                name,
                ernie.DenseField,
                voxel_size=VOXEL,
                translate=(-2.35, -2.35, -11.95),
            )
        plume_prim.set_field("density", PLUME.child("Density"))
        plume_prim.set_field(
            "temperature", PLUME.child("Temperature")
        )
        with wp.ScopedDevice(self.device):
            self.soot = wp.zeros((NX, NY, NZ), dtype=float)
            self.heat = wp.zeros((NX, NY, NZ), dtype=float)
            self.perm = wp.array(
                ernie.noise_permutation(5).astype(np.int32),
                dtype=wp.int32,
            )
        self.burn(0.0)
        # Soot: mostly absorbing, so it glows as a black body where hot
        # (Kirchhoff) and scatters a little sunlight.
        self.exhaust = self.scene.define(
            "/World/Looks/Exhaust",
            ernie.Material,
            density=1.5,
            volume_albedo=(0.3, 0.3, 0.3),
            anisotropy=0.4,
            temperature_scale=PEAK_KELVIN,
        )

        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            shadow_enable=True,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-0.5, -1.0, 0.6)),
        )
        # Daylight exposure, from the side 40 m off.
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=35.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 500.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -40.0, -1.0), (0.0, 0.0, -1.5), (0.0, 0.0, 1.0)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(320, 480),
            ambient=0.0,
        )

        self.widgets.slider(
            "Peak temperature (K)",
            800.0,
            3500.0,
            PEAK_KELVIN,
            on_change=self.set_peak_temperature,
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def step(self):
        self.burn(self.time + self.frame_dt)

    def burn(self, time):
        """The plume's fields at `time`, written on the GPU."""
        stream = wp.get_stream(self.device)
        wp.launch(
            plume,
            dim=(NX, NY, NZ),
            inputs=[self.soot, self.heat, self.perm, time],
            device=self.device,
            stream=stream,
        )
        self.scene.write(
            PLUME.child("Density"),
            tokens.DENSE_VALUES,
            self.soot,
            stream=stream,
        )
        self.scene.write(
            PLUME.child("Temperature"),
            tokens.DENSE_VALUES,
            self.heat,
            stream=stream,
        )

    def set_peak_temperature(self, kelvin):
        self.exhaust.temperature_scale = kelvin

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        image = np.array(self.renderer.render(self.scene))
        rows, cols = image.shape[:2]
        # The brightest spot down the plume's axis (a shock diamond), and
        # the empty sky beside it.
        core = max(
            (
                mean_in(
                    image, cols // 2 - 4, cols // 2 + 4, row, row + 8
                )
                for row in range(rows // 4, rows - 7, 4)
            ),
            key=lambda spot: spot[0],
        )
        sky = mean_in(image, 0, 16, rows - 32, rows)
        print(
            f"Plume core {core[0]:.4f} {core[1]:.4f} {core[2]:.4f}, "
            f"sky {sky[0]:.4f} {sky[1]:.4f} {sky[2]:.4f}"
        )
        assert sky.sum() < 0.01, "space is black"
        assert core[0] > 0.2, "the hot core glows"
        assert core[0] > core[2], (
            "the glow is warm, as a black body's"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
