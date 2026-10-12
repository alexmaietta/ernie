# SPDX-License-Identifier: BSD-3-Clause
"""A lander hovering over the Moon, its exhaust blowing a sheet of dust
outward: a low sun lights the dust, which shades the ground under it. A
Warp kernel writes the dust, streaming out from under the engine, on
the GPU each frame, a DenseField the renderer reads from CUDA without a
copy to the host."""

import math
from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear, write_png

DENSITY = 0.3
NX, NY, NZ = 192, 192, 24
VOXEL = 0.25
DUST = ernie.Path("/World/Dust")


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
def dust(
    out: wp.array3d(dtype=float),
    perm: wp.array(dtype=wp.int32),
    time: float,
):
    """The dust's density `time` seconds in, as the C++ twin's shader:
    over a 48 m square, 6 m deep, in 0.25 m voxels, a sheet streaming
    out from under the engine at 8 m/s, streaked along its rays and
    billowing, thinning with height and distance."""
    i, j, k = wp.tid()
    px = -24.0 + 0.25 * (float(i) + 0.5)
    py = -24.0 + 0.25 * (float(j) + 0.5)
    pz = 0.25 * (float(k) + 0.5)
    r = wp.sqrt(px * px + py * py)
    c = wp.where(r > 0.0, px / r, 1.0)
    s = wp.where(r > 0.0, py / r, 0.0)
    # Out from 1.5 m, fading over 10 m; thicker further out.
    radial = wp.clamp((r - 1.5) / 2.0, 0.0, 1.0) * wp.exp(-r / 10.0)
    vertical = wp.exp(-pz / (0.2 + 0.05 * r))
    # Streaks along the rays, and billows, streaming out.
    travelled = 8.0 * time
    streaks = 0.5 + 0.5 * fbm(
        wp.vec3(6.0 * c, 6.0 * s, 0.15 * (r - travelled)), 3, perm
    )
    billows = 0.5 + 0.5 * fbm(
        wp.vec3(
            0.6 * (px - travelled * c) + 17.0,
            0.6 * (py - travelled * s) + 17.0,
            0.6 * pz + 17.0,
        ),
        4,
        perm,
    )
    value = (
        radial
        * vertical
        * wp.clamp(2.5 * streaks - 1.0, 0.0, 1.0)
        * (0.5 + billows)
    )
    out[i, j, k] = wp.where(value > 1.0e-3, value, 0.0)


def mean_rows(image, top, bottom):
    """The image's mean linear RGB over rows [top, bottom)."""
    pixels = np.asarray(image)[top:bottom, :, :3]
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
        world.define(
            "Ground",
            ernie.GroundPlane,
            material_binding="/World/Looks/Regolith",
        )
        self.scene.define(
            "/World/Looks/Regolith",
            ernie.Material,
            albedo=(0.12, 0.12, 0.12),
        )
        # The lander 4 m up: its body, legs and engine bell.
        lander = world.define("Lander", ernie.Xform)
        lander.define(
            "Body",
            ernie.Cube,
            size=1.0,
            translate=(0.0, 0.0, 5.5),
            scale=(3.0, 3.0, 2.0),
            material_binding="/World/Looks/Foil",
        )
        for leg in range(4):
            angle = math.pi * (0.25 + 0.5 * leg)
            lander.define(
                f"Leg{leg}",
                ernie.Cylinder,
                radius=0.08,
                height=2.0,
                translate=(
                    2.0 * math.cos(angle),
                    2.0 * math.sin(angle),
                    4.0,
                ),
                material_binding="/World/Looks/Metal",
            )
        lander.define(
            "Bell",
            ernie.Cone,
            radius=0.5,
            height=0.8,
            translate=(0.0, 0.0, 4.1),
            material_binding="/World/Looks/Metal",
        )
        self.scene.define(
            "/World/Looks/Foil",
            ernie.Material,
            albedo=(0.35, 0.28, 0.12),
        )
        self.scene.define(
            "/World/Looks/Metal",
            ernie.Material,
            albedo=(0.3, 0.3, 0.3),
        )

        # The dust, written on the GPU each frame, its first cell at
        # (-23.875, -23.875, 0.125) m: grains that scatter about half the
        # light, mostly back towards the sun, as regolith does (Hapke's
        # double Henyey-Greenstein lobes).
        dust_prim = world.define(
            "Dust", ernie.Volume, material_binding="/World/Looks/Dust"
        )
        dust_prim.define(
            "Density",
            ernie.DenseField,
            voxel_size=VOXEL,
            translate=(-23.875, -23.875, 0.125),
        )
        dust_prim.set_field("density", DUST.child("Density"))
        with wp.ScopedDevice(self.device):
            self.dust = wp.zeros((NX, NY, NZ), dtype=float)
            self.perm = wp.array(
                ernie.noise_permutation(11).astype(np.int32),
                dtype=wp.int32,
            )
        self.blow(0.0)
        self.dust_look = self.scene.define(
            "/World/Looks/Dust",
            ernie.Material,
            density=DENSITY,
            volume_albedo=(0.55, 0.55, 0.55),
            phase_function=tokens.DOUBLE_HENYEY_GREENSTEIN,
        )
        self.scene.set("/World/Looks/Dust", tokens.PHASE_B, 0.3)
        self.scene.set("/World/Looks/Dust", tokens.PHASE_C, 0.6)

        # A low sun from the left.
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            shadow_enable=True,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-1.0, -0.7, 0.3)),
        )
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 250.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (8.0, -22.0, 2.5), (0.0, 0.0, 2.5), (0.0, 0.0, 1.0)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(480, 320),
            ambient=0.0,
        )

        self.widgets.slider(
            "Dust density (1/m)",
            0.0,
            2.0,
            DENSITY,
            on_change=self.set_density,
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def step(self):
        self.blow(self.time + self.frame_dt)

    def blow(self, time):
        """The dust at `time`, written on the GPU."""
        stream = wp.get_stream(self.device)
        wp.launch(
            dust,
            dim=(NX, NY, NZ),
            inputs=[self.dust, self.perm, time],
            device=self.device,
            stream=stream,
        )
        self.scene.write(
            DUST.child("Density"),
            tokens.DENSE_VALUES,
            self.dust,
            stream=stream,
        )

    def set_density(self, density):
        self.dust_look.density = density

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        dusty = np.array(self.renderer.render(self.scene))
        self.scene.set(
            "/World/Dust", tokens.VISIBILITY, tokens.INVISIBLE
        )
        clear = np.array(self.renderer.render(self.scene))
        self.scene.set(
            "/World/Dust", tokens.VISIBILITY, tokens.INHERITED
        )
        rows = dusty.shape[0]
        # The sky above, and the ground beyond the lander, seen through
        # the dust or without it.
        sky = mean_rows(dusty, 0, rows // 8)
        band = (rows // 2 + rows // 16, rows // 2 + rows // 8)
        with_dust = mean_rows(dusty, *band)
        without = mean_rows(clear, *band)
        print(
            f"Sky {sky.sum():.4f}, ground band with dust "
            f"{with_dust[0]:.4f} {with_dust[1]:.4f} {with_dust[2]:.4f}, "
            f"without {without[0]:.4f} {without[1]:.4f} {without[2]:.4f}"
        )
        assert sky.sum() < 0.01, "the Moon's sky is black"
        assert abs(with_dust.sum() - without.sum()) > 0.01, (
            "the dust shows over the ground"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
