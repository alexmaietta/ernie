# SPDX-License-Identifier: BSD-3-Clause
"""A dust devil on Mars, seen from a rover: a hollow column of dust 120 m
tall, widening as it rises, swirling faster near its core and lifting a
skirt of dust at its foot, lit by the sun and the dusty sky. A Warp
kernel writes its dust on the GPU each frame, a DenseField the renderer
reads from CUDA without a copy to the host.

References:
- M. Balme and R. Greeley, "Dust devils on Earth and Mars", Reviews of
  Geophysics 44(3), RG3003, 2006 (their hollow, widening columns and
  skirts, tens to hundreds of metres tall on Mars).
"""

from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

NX, NY, NZ = 64, 64, 256
VOXEL = 0.625
DENSITY = 0.2
# The Sun's irradiance at Mars's mean distance, 1.5237 au.
MARS_IRRADIANCE = 1361.0 / (1.5237 * 1.5237)
DEVIL = ernie.Path("/World/Devil")


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
def devil(
    out: wp.array3d(dtype=float),
    perm: wp.array(dtype=wp.int32),
    time: float,
):
    """The dust devil's dust `time` seconds in, as the C++ twin's
    shader: in 0.625 m voxels from (-20, -20, 0) m, a hollow wall about
    an axis that wanders, turning faster towards the core and rising."""
    i, j, k = wp.tid()
    px = -20.0 + 0.625 * (float(i) + 0.5)
    py = -20.0 + 0.625 * (float(j) + 0.5)
    pz = 0.625 * (float(k) + 0.5)
    # Its axis wanders, more so higher up.
    qx = px - 1.5 * wp.sin(0.3 * time + 0.02 * pz)
    qy = py - 1.5 * wp.cos(0.23 * time + 0.015 * pz)
    r = wp.sqrt(qx * qx + qy * qy)
    radius = 4.0 + 0.06 * pz
    # Turning faster towards the core, and rising.
    spin = 1.2 * time * radius / wp.max(r, 0.5 * radius)
    c = wp.cos(spin)
    s = wp.sin(spin)
    swirled = wp.vec3(
        c * qx + s * qy, c * qy - s * qx, pz - 6.0 * time
    )
    turbulence = 0.5 + 0.5 * fbm(
        wp.cw_mul(swirled, wp.vec3(0.25, 0.25, 0.08)), 4, perm
    )
    ring = (r - radius) / (0.35 * radius)
    inside = r / radius
    wall = wp.exp(-ring * ring) + 0.25 * wp.exp(-inside * inside)
    column = (
        wall
        * wp.exp(-pz / 70.0)
        * wp.clamp((128.0 - pz) / 20.0, 0.0, 1.0)
    )
    foot = r / 9.0
    skirt = 1.5 * wp.exp(-pz / 2.0) * wp.exp(-foot * foot)
    dust = (column + skirt) * wp.clamp(
        1.6 * turbulence - 0.2, 0.0, 2.0
    )
    out[i, j, k] = wp.where(dust > 1.0e-3, dust, 0.0)


class Demo(ernie.demo.Demo):
    num_frames = 6

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("This demo needs CUDA interop")
        self.device = "cuda:0"
        world = self.scene.define("/World", ernie.Xform)
        # Mars's soil and air in a clear season, the sun 25 degrees up and
        # 50 to the view's left: the dust lit from behind, scattering
        # forward, clear of the sky's glow about the sun.
        ernie.demo.define_planet_surface(
            self.scene,
            "/World/Mars",
            3389.5e3,
            (0.33, 0.22, 0.14),
            "/World/Looks/Soil",
        )
        world.define(
            "Air",
            ernie.Atmosphere,
            preset=tokens.MARS,
            dust_optical_depth=0.3,
            ground_albedo=0.25,
        )
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=MARS_IRRADIANCE,
            physical_illuminant=tokens.SUN,
            angle=0.35,
            shadow_enable=True,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.265, 0.867, 0.423)),
        )

        # The dust, written on the GPU each frame, its first cell at
        # (-19.6875, -19.6875, 0.3125) m.
        devil_prim = world.define(
            "Devil",
            ernie.Volume,
            material_binding="/World/Looks/Dust",
        )
        devil_prim.define(
            "Density",
            ernie.DenseField,
            voxel_size=VOXEL,
            translate=(-19.6875, -19.6875, 0.3125),
        )
        devil_prim.set_field("density", DEVIL.child("Density"))
        with wp.ScopedDevice(self.device):
            self.dust = wp.zeros((NX, NY, NZ), dtype=float)
            self.perm = wp.array(
                ernie.noise_permutation(7).astype(np.int32),
                dtype=wp.int32,
            )
        self.blow(0.0)
        # Martian dust: reddish, scattering forward.
        self.look = self.scene.define(
            "/World/Looks/Dust",
            ernie.Material,
            density=DENSITY,
            volume_albedo=(0.9, 0.75, 0.6),
            anisotropy=0.6,
        )

        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            sensor_noise=False,
            exposure_time=1.0 / 125.0,
            exposure_iso=125.0,
            exposure_f_stop=11.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-160.0, -70.0, 2.5),
                (0.0, 0.0, 35.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(360, 480),
            ambient=0.0,
        )

        self.widgets.slider(
            "Dust density",
            0.0,
            0.5,
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
            devil,
            dim=(NX, NY, NZ),
            inputs=[self.dust, self.perm, time],
            device=self.device,
            stream=stream,
        )
        self.scene.write(
            DEVIL.child("Density"),
            tokens.DENSE_VALUES,
            self.dust,
            stream=stream,
        )

    def set_density(self, density):
        self.look.density = density

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        # The devil against none: its dust shows against the sky.
        with_dust = np.asarray(
            self.renderer.render(self.scene)
        ).astype(float)
        self.look.density = 0.0
        without = np.asarray(self.renderer.render(self.scene)).astype(
            float
        )
        self.look.density = DENSITY
        difference = float(
            np.abs(with_dust - without)[..., :3].mean()
        )
        print(f"The dust devil changes the image by {difference:.2f}")
        assert difference > 1.0, "the dust devil shows"


if __name__ == "__main__":
    ernie.demo.main(Demo)
