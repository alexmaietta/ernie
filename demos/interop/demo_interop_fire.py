# SPDX-License-Identifier: BSD-3-Clause
"""A campfire simulated by Warp on the GPU each step, as NVIDIA Flow's fire:
fuel burns where it is hot, the burning heats the gas and makes soot, and the
gas rises, swirls and cools. Its soot and temperature, written as
DenseFields from CUDA, are scattered into the renderer's grids on the GPU
with no copy to the host: the soot glows as a black body where hot and
shadows itself and the ground. The C++ twin runs the same solver in Vulkan
compute shaders (RenderDelegate::SetFieldWriter).

References:
- NVIDIA Flow 2 (PhysX SDK, flow/source/nvflow/shaders/AdvectionCommon.hlsli):
  its combustion model and constants, MacCormack advection.
- J. Stam, "Stable Fluids", SIGGRAPH 1999.
  https://doi.org/10.1145/311535.311548
- R. Fedkiw, J. Stam and H. W. Jensen, "Visual Simulation of Smoke",
  SIGGRAPH 2001 (buoyancy, vorticity confinement).
  https://doi.org/10.1145/383259.383260
- A. Selle, R. Fedkiw, B. Kim, Y. Liu and J. Rossignac, "An Unconditionally
  Stable MacCormack Method", Journal of Scientific Computing 35, 2008."""

from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens

NX, NY, NZ = 64, 64, 128
VOXEL = 0.03  # m: a 1.9 m by 1.9 m by 3.8 m domain.
JACOBI = 40
FIRE = ernie.Path("/World/Fire")
# Flow's fire: temperature over 0 to about 1, which the renderer takes from
# the ambient air's 300 K to about 2100 K.
IGNITION = 0.05
BURN_PER_TEMPERATURE = 4.0
FUEL_PER_BURN = 0.25
TEMPERATURE_PER_BURN = 5.0
SMOKE_PER_BURN = 1.0
COOLING = 5.0  # /s
BUOYANCY = 25.0  # m/s^2 per unit temperature
VORTICITY = 0.04
KELVIN_SCALE = 2000.0


@wp.func
@no_type_check
def mix(a: float, b: float, t: float):
    return a + (b - a) * t


@wp.func
@no_type_check
def at(f: wp.array3d(dtype=float), i: int, j: int, k: int):
    """`f` at cell (i, j, k), clamped into the grid."""
    return f[
        wp.clamp(i, 0, NX - 1),
        wp.clamp(j, 0, NY - 1),
        wp.clamp(k, 0, NZ - 1),
    ]


@wp.func
@no_type_check
def sample(f: wp.array3d(dtype=float), x: float, y: float, z: float):
    """`f` trilinear at (x, y, z), in cells."""
    fx = wp.floor(x)
    fy = wp.floor(y)
    fz = wp.floor(z)
    i = int(fx)
    j = int(fy)
    k = int(fz)
    u = x - fx
    v = y - fy
    w = z - fz
    c00 = mix(at(f, i, j, k), at(f, i + 1, j, k), u)
    c10 = mix(at(f, i, j + 1, k), at(f, i + 1, j + 1, k), u)
    c01 = mix(at(f, i, j, k + 1), at(f, i + 1, j, k + 1), u)
    c11 = mix(at(f, i, j + 1, k + 1), at(f, i + 1, j + 1, k + 1), u)
    return mix(mix(c00, c10, v), mix(c01, c11, v), w)


@wp.func
@no_type_check
def lowest(f: wp.array3d(dtype=float), x: float, y: float, z: float):
    """The least of `f`'s eight values around (x, y, z)."""
    i = int(wp.floor(x))
    j = int(wp.floor(y))
    k = int(wp.floor(z))
    m = at(f, i, j, k)
    for c in range(1, 8):
        m = wp.min(
            m,
            at(
                f, i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)
            ),
        )
    return m


@wp.func
@no_type_check
def highest(f: wp.array3d(dtype=float), x: float, y: float, z: float):
    """The greatest of `f`'s eight values around (x, y, z)."""
    i = int(wp.floor(x))
    j = int(wp.floor(y))
    k = int(wp.floor(z))
    m = at(f, i, j, k)
    for c in range(1, 8):
        m = wp.max(
            m,
            at(
                f, i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)
            ),
        )
    return m


@wp.func
@no_type_check
def hash01(x: int, y: int, z: int, n: int):
    """A value in [0, 1) from integers, as the C++ twin's."""
    h = (
        wp.uint32(x) * wp.uint32(73856093)
        ^ wp.uint32(y) * wp.uint32(19349663)
        ^ wp.uint32(z) * wp.uint32(83492791)
        ^ wp.uint32(n) * wp.uint32(2654435761)
    )
    h = h ^ (h >> wp.uint32(13))
    h = h * wp.uint32(0x5BD1E995)
    h = h ^ (h >> wp.uint32(15))
    return float(h & wp.uint32(0xFFFFFF)) / 16777216.0


@wp.kernel
@no_type_check
def burn(
    heat: wp.array3d(dtype=float),
    fuel: wp.array3d(dtype=float),
    smoke: wp.array3d(dtype=float),
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    frame: int,
    dt: float,
):
    """The fire's bed, flickering, feeds fuel and heat; where it is hot
    enough the fuel burns, heating the gas and making soot; the gas cools
    and the hot rises (Flow's combustion)."""
    i, j, k = wp.tid()
    dx = float(i - NX // 2)
    dy = float(j - NY // 2)
    dz = float(k - 3) * 2.0
    r = wp.sqrt(dx * dx + dy * dy + dz * dz)
    t = heat[i, j, k]
    f = fuel[i, j, k]
    if r < 9.0:
        # Flickering in patches of the bed, changing every few steps.
        flicker = hash01(i // 4, j // 4, k // 3, frame // 4)
        w = (1.0 - r / 9.0) * (0.4 + 0.6 * flicker)
        t = mix(t, 1.0, wp.min(10.0 * dt, 1.0) * w)
        f = mix(f, 0.8, wp.min(2.0 * dt, 1.0) * w)
        vx[i, j, k] = (
            vx[i, j, k]
            + 1.5
            * (hash01(i // 4, j // 4, k // 4, frame // 4) - 0.5)
            * w
        )
        vy[i, j, k] = (
            vy[i, j, k]
            + 1.5
            * (hash01(i // 4 + 9, j // 4, k // 4, frame // 4) - 0.5)
            * w
        )
    burnt = 0.0
    if t >= IGNITION and f > 0.0:
        burnt = wp.min(
            BURN_PER_TEMPERATURE * (wp.min(t, 1.0) - IGNITION) * dt,
            f / FUEL_PER_BURN,
        )
    f = f - FUEL_PER_BURN * burnt
    t = wp.min(t + TEMPERATURE_PER_BURN * burnt, 1.0)
    t = t - COOLING * dt * t
    heat[i, j, k] = t
    fuel[i, j, k] = f
    # Soot fades out towards the domain's sides, so it leaves rather than
    # piling up against them.
    edge = wp.min(
        wp.min(float(i), float(NX - 1 - i)),
        wp.min(float(j), float(NY - 1 - j)),
    )
    fade = wp.clamp(edge / 8.0, 0.0, 1.0)
    smoke[i, j, k] = (smoke[i, j, k] + SMOKE_PER_BURN * burnt) * mix(
        0.9, 1.0, fade
    )
    vz[i, j, k] = vz[i, j, k] + BUOYANCY * t * dt


@wp.kernel
@no_type_check
def curl(
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
):
    """The flow's vorticity, in cells."""
    i, j, k = wp.tid()
    wx[i, j, k] = 0.5 * (
        at(vz, i, j + 1, k)
        - at(vz, i, j - 1, k)
        - at(vy, i, j, k + 1)
        + at(vy, i, j, k - 1)
    )
    wy[i, j, k] = 0.5 * (
        at(vx, i, j, k + 1)
        - at(vx, i, j, k - 1)
        - at(vz, i + 1, j, k)
        + at(vz, i - 1, j, k)
    )
    wz[i, j, k] = 0.5 * (
        at(vy, i + 1, j, k)
        - at(vy, i - 1, j, k)
        - at(vx, i, j + 1, k)
        + at(vx, i, j - 1, k)
    )


@wp.func
@no_type_check
def spin(
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
):
    return wp.length(
        wp.vec3(at(wx, i, j, k), at(wy, i, j, k), at(wz, i, j, k))
    )


@wp.kernel
@no_type_check
def confine(
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    strength: float,
):
    """Vorticity confinement: the flow's small swirls given back what the
    grid's numerical diffusion takes."""
    i, j, k = wp.tid()
    n = 0.5 * wp.vec3(
        spin(wx, wy, wz, i + 1, j, k) - spin(wx, wy, wz, i - 1, j, k),
        spin(wx, wy, wz, i, j + 1, k) - spin(wx, wy, wz, i, j - 1, k),
        spin(wx, wy, wz, i, j, k + 1) - spin(wx, wy, wz, i, j, k - 1),
    )
    length = wp.length(n)
    if length > 1.0e-6:
        force = strength * wp.cross(
            n / length, wp.vec3(wx[i, j, k], wy[i, j, k], wz[i, j, k])
        )
        vx[i, j, k] = vx[i, j, k] + force[0]
        vy[i, j, k] = vy[i, j, k] + force[1]
        vz[i, j, k] = vz[i, j, k] + force[2]


@wp.kernel
@no_type_check
def divergence(
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    div[i, j, k] = 0.5 * (
        at(vx, i + 1, j, k)
        - at(vx, i - 1, j, k)
        + at(vy, i, j + 1, k)
        - at(vy, i, j - 1, k)
        + at(vz, i, j, k + 1)
        - at(vz, i, j, k - 1)
    )


@wp.kernel
@no_type_check
def jacobi(
    pressure: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    out[i, j, k] = (
        at(pressure, i + 1, j, k)
        + at(pressure, i - 1, j, k)
        + at(pressure, i, j + 1, k)
        + at(pressure, i, j - 1, k)
        + at(pressure, i, j, k + 1)
        + at(pressure, i, j, k - 1)
        - div[i, j, k]
    ) / 6.0


@wp.kernel
@no_type_check
def project(
    pressure: wp.array3d(dtype=float),
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
):
    """The flow less the pressure's gradient, so it swirls rather than
    piles up; the ground stops it."""
    i, j, k = wp.tid()
    vx[i, j, k] = vx[i, j, k] - 0.5 * (
        at(pressure, i + 1, j, k) - at(pressure, i - 1, j, k)
    )
    vy[i, j, k] = vy[i, j, k] - 0.5 * (
        at(pressure, i, j + 1, k) - at(pressure, i, j - 1, k)
    )
    vz[i, j, k] = vz[i, j, k] - 0.5 * (
        at(pressure, i, j, k + 1) - at(pressure, i, j, k - 1)
    )
    if k == 0:
        vz[i, j, k] = wp.max(vz[i, j, k], 0.0)


@wp.kernel
@no_type_check
def trace(
    source: wp.array3d(dtype=float),
    target: wp.array3d(dtype=float),
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    cells: float,
):
    """Semi-Lagrangian: each cell takes what was where the flow comes from,
    `cells` the step's cells per m/s (negative, where it goes)."""
    i, j, k = wp.tid()
    target[i, j, k] = sample(
        source,
        float(i) - vx[i, j, k] * cells,
        float(j) - vy[i, j, k] * cells,
        float(k) - vz[i, j, k] * cells,
    )


@wp.kernel
@no_type_check
def correct(
    source: wp.array3d(dtype=float),
    forward: wp.array3d(dtype=float),
    back: wp.array3d(dtype=float),
    target: wp.array3d(dtype=float),
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    cells: float,
    blend: float,
    keep: float,
):
    """MacCormack: the semi-Lagrangian result less `blend` of its error
    (half of what tracing it back misses), within the values it came from
    so it stays stable; then `keep` of it."""
    i, j, k = wp.tid()
    x = float(i) - vx[i, j, k] * cells
    y = float(j) - vy[i, j, k] * cells
    z = float(k) - vz[i, j, k] * cells
    value = forward[i, j, k] + blend * 0.5 * (
        source[i, j, k] - back[i, j, k]
    )
    value = wp.clamp(
        value, lowest(source, x, y, z), highest(source, x, y, z)
    )
    target[i, j, k] = keep * value


class Demo(ernie.demo.Demo):
    # Headless, while the twins' solvers still agree: past about 60 steps
    # their float rounding grows into a different fire.
    num_frames = 45

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("This demo needs CUDA interop")
        self.device = "cuda:0"
        define_scene(self.scene)
        # The solver's grids on the GPU; the fire's fields take its soot
        # and temperature as they are, every step.
        with wp.ScopedDevice(self.device):
            self.f = {
                name: wp.zeros((NX, NY, NZ), dtype=float)
                for name in (
                    "heat",
                    "fuel",
                    "smoke",
                    "vx",
                    "vy",
                    "vz",
                    "wx",
                    "wy",
                    "wz",
                    "pressure",
                    "div",
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                    "g",
                )
            }
        self.publish(wp.get_stream(self.device))
        if viewer is not None:
            viewer.active_camera = "/World/Camera"

    def publish(self, stream):
        """The fields' values as they are now, on the GPU."""
        for name, field in (("smoke", "Density"), ("heat", "Heat")):
            self.scene.write(
                FIRE.child(field),
                tokens.DENSE_VALUES,
                self.f[name],
                stream=stream,
            )

    def swap(self, a, b):
        self.f[a], self.f[b] = self.f[b], self.f[a]

    def maccormack(
        self, names, velocity, outputs, blend, keep, launch
    ):
        """Carries fields `names` along the flow `velocity` (field names),
        into `outputs`, MacCormack's way."""
        f = self.f
        cells = self.frame_dt / VOXEL
        v = [f[n] for n in velocity]
        for name, out, k in zip(names, outputs, keep):
            wp.launch(
                trace, inputs=[f[name], f["d"], *v, cells], **launch
            )
            wp.launch(
                trace, inputs=[f["d"], f["e"], *v, -cells], **launch
            )
            wp.launch(
                correct,
                inputs=[
                    f[name],
                    f["d"],
                    f["e"],
                    f[out],
                    *v,
                    cells,
                    blend,
                    k,
                ],
                **launch,
            )

    def step(self):
        stream = wp.get_stream(self.device)
        launch = dict(
            dim=(NX, NY, NZ), device=self.device, stream=stream
        )
        f = self.f
        dt = self.frame_dt
        wp.launch(
            burn,
            inputs=[
                f["heat"],
                f["fuel"],
                f["smoke"],
                f["vx"],
                f["vy"],
                f["vz"],
                self.frame + 1,
                dt,
            ],
            **launch,
        )
        wp.launch(
            curl,
            inputs=[
                f["vx"],
                f["vy"],
                f["vz"],
                f["wx"],
                f["wy"],
                f["wz"],
            ],
            **launch,
        )
        wp.launch(
            confine,
            inputs=[
                f["wx"],
                f["wy"],
                f["wz"],
                f["vx"],
                f["vy"],
                f["vz"],
                VORTICITY,
            ],
            **launch,
        )
        wp.launch(
            divergence,
            inputs=[f["vx"], f["vy"], f["vz"], f["div"]],
            **launch,
        )
        for _ in range(JACOBI):
            wp.launch(
                jacobi,
                inputs=[f["pressure"], f["div"], f["a"]],
                **launch,
            )
            self.swap("pressure", "a")
        wp.launch(
            project,
            inputs=[f["pressure"], f["vx"], f["vy"], f["vz"]],
            **launch,
        )
        # The flow carried by itself, from its values before.
        self.maccormack(
            ("vx", "vy", "vz"),
            ("vx", "vy", "vz"),
            ("a", "b", "c"),
            0.5,
            (0.999,) * 3,
            launch,
        )
        self.swap("vx", "a")
        self.swap("vy", "b")
        self.swap("vz", "c")
        # Soot fades slowly; fuel and heat carried as they are.
        self.maccormack(
            ("heat", "fuel", "smoke"),
            ("vx", "vy", "vz"),
            ("a", "b", "c"),
            0.9,
            (1.0, 1.0, 0.99),
            launch,
        )
        self.swap("heat", "a")
        self.swap("fuel", "b")
        self.swap("smoke", "c")
        self.publish(stream)

    def save_output(self, path):
        ernie.demo.write_png(
            path, np.array(self.renderer.render(self.scene))
        )

    def test_final(self):
        smoke = self.f["smoke"].numpy()
        heat = self.f["heat"].numpy()
        heights = np.arange(NZ) * VOXEL
        rise = (smoke.sum(axis=(0, 1)) * heights).sum() / smoke.sum()
        print(
            f"soot's mean height {rise:.2f} m, hottest {heat.max():.2f}"
        )
        assert rise > 0.3, "the soot should rise"
        assert heat.max() > 0.5, "the fire should be hot"
        image = np.array(self.renderer.render(self.scene)).astype(
            np.int32
        )
        warm = (image[..., 0] - image[..., 2] > 40) & (
            image[..., 0] > 120
        )
        print(f"glowing pixels {100 * warm.mean():.2f}%")
        assert warm.mean() > 0.005, "the flames should glow"


def define_scene(scene):
    """The ground at dusk, the fire's light, its soot and the camera, as the
    C++ twin's."""
    world = scene.define("/World", ernie.Xform)
    world.define(
        "Ground",
        ernie.GroundPlane,
        material_binding="/World/Looks/Earth",
    )
    scene.define(
        "/World/Looks/Earth", ernie.Material, albedo=(0.1, 0.08, 0.06)
    )
    world.define(
        "Sun",
        ernie.DistantLight,
        irradiance=5.0,
        physical_illuminant=tokens.SUN,
    ).set(
        tokens.TRANSFORM,
        ernie.distant_light_transform((-1.0, 0.4, 0.12)),
    )
    world.define(
        "Glow",
        ernie.SphereLight,
        radius=0.3,
        color=(1.0, 0.5, 0.15),
        intensity=60.0,
        translate=(0.0, 0.0, 0.25),
    )
    fire = world.define(
        "Fire", ernie.Volume, material_binding="/World/Looks/Soot"
    )
    # The grid's first cell at the domain's corner, its floor on the ground.
    low = (-NX * VOXEL / 2.0, -NY * VOXEL / 2.0, 0.0)
    for name in ("Density", "Heat"):
        fire.define(
            name, ernie.DenseField, voxel_size=VOXEL, translate=low
        )
    fire.set_field("density", FIRE.child("Density"))
    fire.set_field("temperature", FIRE.child("Heat"))
    # Soot: absorbing more than it scatters, so it glows as a black body
    # where hot and is dark grey where it has cooled.
    scene.define(
        "/World/Looks/Soot",
        ernie.Material,
        density=2.0,
        volume_albedo=(0.5, 0.5, 0.5),
        anisotropy=0.3,
        temperature=300.0,
        temperature_scale=KELVIN_SCALE,
        scattering_octaves=3,
    )
    world.define(
        "Camera",
        ernie.Camera,
        focal_length=28.0,
        sensor_noise=False,
        exposure_iso=800.0,
        exposure_f_stop=4.0,
        exposure_time=1.0 / 60.0,
    ).set(
        tokens.TRANSFORM,
        ernie.look_at_transform((0.0, -6.0, 1.4), (0.0, 0.0, 1.3)),
    )
    scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        camera=ernie.Path("/World/Camera"),
        resolution=(480, 480),
        ambient=0.0,
    )


if __name__ == "__main__":
    ernie.demo.main(Demo)
