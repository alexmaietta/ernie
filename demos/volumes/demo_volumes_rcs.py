# SPDX-License-Identifier: BSD-3-Clause
"""A spacecraft's reaction control thrusters firing short pulses in
sunlight: each puff of exhaust spreads in a wide cone, a shell of gas
racing out after the valve closes, a hot flash at the nozzle while it
fires. A Warp kernel writes the gas on the GPU each frame, DenseFields
the renderer reads from CUDA without a copy to the host.

References:
- G. A. Simons, "Effect of Nozzle Boundary Layers on Rocket Exhaust
  Plumes", AIAA Journal 10(11), 1972, 1534-1535 (a plume's density
  falling as the square of the distance and a power of the cosine off
  its axis).
"""

import math
from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

NX, NY, NZ = 72, 72, 56
VOXEL = 0.125
# Each thruster's pulse, s, of every period.
PULSE = 0.08
DENSITY = 20.0
GAS = ernie.Path("/World/Gas")


@wp.kernel
@no_type_check
def rcs(
    gas_out: wp.array3d(dtype=float),
    heat_out: wp.array3d(dtype=float),
    time: float,
    pulse: float,
):
    """The thrusters' exhaust `time` seconds in, as the C++ twin's
    shader: eight thrusters in four quads around the body's top, each
    pulsing `pulse` s in turn every 1.2 s, the gas leaving at 4 m/s."""
    i, j, k = wp.tid()
    p = wp.vec3(
        -4.5 + 0.125 * (float(i) + 0.5),
        -4.5 + 0.125 * (float(j) + 0.5),
        -2.0 + 0.125 * (float(k) + 0.5),
    )
    gas = wp.float32(0.0)
    heat = wp.float32(0.0)
    for t in range(8):
        # Quad t / 2 around the body; its radial thruster, then its roll
        # one.
        angle = 0.5 * wp.pi * float(t // 2)
        out = wp.vec3(wp.cos(angle), wp.sin(angle), 0.0)
        axis = out
        if t % 2 == 1:
            axis = wp.vec3(-out[1], out[0], 0.0)
        to = p - (1.15 * out + wp.vec3(0.0, 0.0, 1.5))
        r = wp.max(wp.length(to), 0.05)
        c = wp.dot(to, axis) / r
        if c > 0.0:
            # Firing when the gas here left the nozzle, its pulse's edges
            # soft.
            left = time - r / 4.0 - float(t) * 1.2 / 8.0
            phase = left - 1.2 * wp.floor(left / 1.2)
            on = wp.clamp(phase / 0.01, 0.0, 1.0) * wp.clamp(
                (pulse - phase) / 0.01, 0.0, 1.0
            )
            c2 = c * c
            c4 = c2 * c2
            c8 = c4 * c4
            spread = c8 * c8
            near = 0.25 / wp.max(r, 0.25)
            gas += on * spread * near * near
            heat = wp.max(heat, on * spread * wp.exp(-r / 0.15))
    gas_out[i, j, k] = wp.where(gas > 1.0e-4, gas, 0.0)
    heat_out[i, j, k] = wp.where(heat > 1.0e-4, heat, 0.0)


class Demo(ernie.demo.Demo):
    num_frames = 30

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("This demo needs CUDA interop")
        self.device = "cuda:0"
        world = self.scene.define("/World", ernie.Xform)
        # The spacecraft, 2 m across, its thruster quads around its top.
        craft = world.define("Craft", ernie.Xform)
        craft.define(
            "Body",
            ernie.Cylinder,
            radius=1.0,
            height=4.0,
            material_binding="/World/Looks/Paint",
        )
        for q in range(4):
            angle = 0.5 * math.pi * q
            craft.define(
                f"Quad{q}",
                ernie.Sphere,
                radius=0.15,
                translate=(
                    1.15 * math.cos(angle),
                    1.15 * math.sin(angle),
                    1.5,
                ),
                material_binding="/World/Looks/Metal",
            )
        self.scene.define(
            "/World/Looks/Paint",
            ernie.Material,
            albedo=(0.8, 0.8, 0.8),
        )
        self.scene.define(
            "/World/Looks/Metal",
            ernie.Material,
            albedo=(0.3, 0.3, 0.3),
        )

        # The exhaust and its heat, written on the GPU each frame.
        gas = world.define(
            "Gas",
            ernie.Volume,
            material_binding="/World/Looks/Exhaust",
        )
        for name in ("Density", "Temperature"):
            gas.define(
                name,
                ernie.DenseField,
                voxel_size=VOXEL,
                translate=(-4.4375, -4.4375, -1.9375),
            )
        gas.set_field("density", GAS.child("Density"))
        gas.set_field("temperature", GAS.child("Temperature"))
        with wp.ScopedDevice(self.device):
            self.gas = wp.zeros((NX, NY, NZ), dtype=float)
            self.heat = wp.zeros((NX, NY, NZ), dtype=float)
        self.pulse = PULSE
        self.fire(0.0)
        # Frozen propellant droplets: bright, scattering forward, and a
        # black-body glow where hot.
        self.exhaust = self.scene.define(
            "/World/Looks/Exhaust",
            ernie.Material,
            density=DENSITY,
            volume_albedo=(0.9, 0.9, 0.9),
            anisotropy=0.3,
            temperature_scale=2500.0,
        )

        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            shadow_enable=True,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.6, -0.5, 0.4)),
        )
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=28.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 500.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (9.0, -9.0, 4.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)
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
            "Pulse (ms)",
            20.0,
            300.0,
            1000.0 * PULSE,
            on_change=self.set_pulse,
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def step(self):
        self.fire(self.time + self.frame_dt)

    def fire(self, time):
        """The exhaust at `time`, written on the GPU."""
        stream = wp.get_stream(self.device)
        wp.launch(
            rcs,
            dim=(NX, NY, NZ),
            inputs=[self.gas, self.heat, time, self.pulse],
            device=self.device,
            stream=stream,
        )
        self.scene.write(
            GAS.child("Density"),
            tokens.DENSE_VALUES,
            self.gas,
            stream=stream,
        )
        self.scene.write(
            GAS.child("Temperature"),
            tokens.DENSE_VALUES,
            self.heat,
            stream=stream,
        )

    def set_pulse(self, ms):
        self.pulse = 1e-3 * ms

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        # The gas against none of it: what it scatters and glows.
        def total():
            image = np.asarray(self.renderer.render(self.scene))
            return float(image[..., :3].astype(np.float64).sum())

        with_gas = total()
        self.exhaust.density = 0.0
        without = total()
        self.exhaust.density = DENSITY
        print(
            f"Exhaust adds {with_gas - without:.0f} to {without:.0f}"
        )
        assert with_gas > 1.02 * without, "the exhaust shows"


if __name__ == "__main__":
    ernie.demo.main(Demo)
