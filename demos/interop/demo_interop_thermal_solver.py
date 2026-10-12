# SPDX-License-Identifier: BSD-3-Clause
"""A thermal solver's write-back: an avionics board whose mesh points are the
nodes of a nodal network (conduction to its neighbours, three chips heating in
turn, radiation from both faces to the cold surroundings), stepped every frame
and seen through an 8 to 14 um camera. The solver runs in NumPy, whose host
arrays the renderer uploads in place, or in Warp on the GPU, whose CUDA arrays
it copies on the device with no trip through the host; switch between them
as it runs.

References:
- D. G. Gilmore (ed.), "Spacecraft Thermal Control Handbook", 2nd ed., The
  Aerospace Press, 2002 (nodal networks: capacitance, conductance and
  radiation per node).
- F. P. Incropera et al., "Fundamentals of Heat and Mass Transfer", 7th ed.,
  Wiley, 2011, section 5.10 (explicit finite differences and their stable
  step).
"""

from typing import no_type_check

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

try:
    import warp as wp
except ImportError:  # the CPU solver alone
    wp = None

N = 64  # nodes along each side of the plate
SIDE = 0.5  # m
DX = SIDE / (N - 1)
THICKNESS = 0.0016  # m: a circuit board
CONDUCTIVITY = 15.0  # W/(m K), in its plane, its copper layers'
DENSITY = 1900.0  # kg/m^3
SPECIFIC_HEAT = 1100.0  # J/(kg K)
EMISSIVITY = 0.9  # both faces, its paint's
SIGMA = 5.670374e-8  # W/(m^2 K^4)
SINK = 250.0  # K: what the plate sees around it
START = 290.0  # K
DT = 1.0  # s: below the explicit limit C / 4G, 2.2 s
SUBSTEPS = 10  # 10 s of the plate's time a frame
PERIOD = 600.0  # s: the chips' duty cycle
CAPACITY = DENSITY * SPECIFIC_HEAT * THICKNESS * DX * DX  # J/K per node
CONDUCTANCE = CONDUCTIVITY * THICKNESS  # W/K between neighbours
RADIATING = 2.0 * EMISSIVITY * SIGMA * DX * DX  # W/K^4 per node
SINK4 = SINK**4
STEP = DT / CAPACITY  # K per joule
# The chips: their nodes' first row and column, size, and power in watts.
CHIPS = [(14, 12, 8, 18.0), (36, 20, 10, 30.0), (20, 42, 6, 12.0)]
PLATE = ernie.Path("/World/Plate")


def heaters(time, on=True):
    """Watts into each node at `time`: each chip on half its period, in
    turn."""
    power = np.zeros((N, N), dtype=np.float32)
    if not on:
        return power
    for k, (i, j, size, watts) in enumerate(CHIPS):
        phase = (time / PERIOD + k / len(CHIPS)) % 1.0
        if phase < 0.5:
            power[i : i + size, j : j + size] = watts / (size * size)
    return power


def step_cpu(kelvin, power):
    """One explicit step of the network: each node's capacity takes the
    heat its four neighbours conduct in (none across the plate's edges), its
    chips' and what it radiates away."""
    padded = np.pad(kelvin, 1, mode="edge")
    neighbours = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    flow = (
        np.float32(CONDUCTANCE) * (neighbours - np.float32(4.0) * kelvin)
        + power
        - np.float32(RADIATING) * (kelvin * kelvin * kelvin * kelvin - np.float32(SINK4))
    )
    return (kelvin + np.float32(STEP) * flow).astype(np.float32)


if wp is not None:

    @wp.kernel
    @no_type_check
    def step_gpu(
        kelvin: wp.array2d(dtype=float),
        power: wp.array2d(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        """`step_cpu`, a thread a node."""
        i, j = wp.tid()
        t = kelvin[i, j]
        neighbours = (
            kelvin[wp.max(i - 1, 0), j]
            + kelvin[wp.min(i + 1, N - 1), j]
            + kelvin[i, wp.max(j - 1, 0)]
            + kelvin[i, wp.min(j + 1, N - 1)]
        )
        flow = (
            float(CONDUCTANCE) * (neighbours - 4.0 * t)
            + power[i, j]
            - float(RADIATING) * (t * t * t * t - float(SINK4))
        )
        out[i, j] = t + float(STEP) * flow


def plate_mesh():
    """The plate standing upright, its points the nodes (node (i, j) at i
    along x and j up), two triangles a cell."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    points = np.stack(
        [i * DX - 0.5 * SIDE, np.zeros_like(i, dtype=float), 0.3 + j * DX],
        axis=-1,
    ).reshape(-1, 3)
    corner = (np.arange(N - 1)[:, None] * N + np.arange(N - 1)[None, :]).ravel()
    quads = np.stack([corner, corner + N, corner + N + 1, corner + 1], axis=-1)
    return points, quads


class Demo(ernie.demo.Demo):
    num_frames = 40
    fixed_dt = 1.0 / 30.0

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.gpu = wp is not None and self._cuda()
        self.heating = True
        self.time = 0.0
        self.kelvin = np.full((N, N), START, dtype=np.float32)
        if self.gpu:
            self.device = "cuda:0"
            with wp.ScopedDevice(self.device):
                self.state = [wp.array(self.kelvin), wp.zeros((N, N), dtype=float)]
                self.power = wp.zeros((N, N), dtype=float)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = world.define("Looks", ernie.Xform)
        concrete = looks.define(
            "Concrete", ernie.Material, reflectance_library="concrete"
        )
        paint = looks.define(
            "Paint",
            ernie.Material,
            reflectance_library="black_paint",
            thermal_emissivity=[8000.0, EMISSIVITY, 14000.0, EMISSIVITY],
        )
        world.define("Ground", ernie.GroundPlane, material_binding=concrete.path)
        points, quads = plate_mesh()
        scene.define(
            PLATE,
            ernie.Mesh,
            points=points,
            face_vertex_counts=np.full(len(quads), 4),
            face_vertex_indices=quads.ravel(),
            material_binding=paint.path,
        )
        # A stand under it, at the air's temperature.
        world.define(
            "Stand",
            ernie.Cube,
            size=1.0,
            material_binding=concrete.path,
            translate=(0.0, 0.0, 0.15),
            scale=(0.3, 0.1, 0.3),
        )
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            horizontal_aperture=20.955,
            vertical_aperture=20.955 * 0.75,
            clipping_range=(0.05, 100.0),
            sensor_noise=False,
            spectral_range=(8000.0, 14000.0),
            spectral_bands=6,
            sensor_type="mono",
            sensor_qe_mono=[8000.0, 0.8, 14000.0, 0.8],
            sensor_full_well=1e12,
            exposure_mode=tokens.AUTO,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.25, -0.95, 0.7), (0.0, 0.0, 0.52), (0.0, 0.0, 1.0)
            ),
        )
        variables = []
        for name in ("LdrColor", "Temperature"):
            var = scene.define(f"/Render/Thermal/{name}", ernie.RenderVar)
            var.source_name = name
            variables.append(var.path)
        scene.define(
            "/Render/Thermal",
            ernie.RenderProduct,
            camera=camera,
            resolution=(400, 300),
            ordered_vars=variables,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(400, 300),
            ambient=0.0,
            thermal_enable=True,
            thermal_default_temperature=285.0,
            thermal_background_temperature=SINK,
        )
        self.publish()

        def solver(index):
            gpu = index == 1
            if gpu and wp is None:
                return
            if gpu != self.gpu:
                self.kelvin = self.read()
                self.gpu = gpu
                if gpu:
                    self.device = "cuda:0"
                    with wp.ScopedDevice(self.device):
                        self.state = [
                            wp.array(self.kelvin),
                            wp.zeros((N, N), dtype=float),
                        ]
                        self.power = wp.zeros((N, N), dtype=float)

        self.widgets.combo(
            "Solver",
            ["CPU (NumPy)", "GPU (Warp)"],
            1 if self.gpu else 0,
            on_change=solver,
        )
        self.widgets.checkbox(
            "Chips on", True, on_change=lambda on: setattr(self, "heating", on)
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    @staticmethod
    def _cuda():
        wp.init()
        return wp.is_cuda_available()

    def read(self):
        """The nodes' temperatures, on the host."""
        return self.state[0].numpy() if self.gpu else self.kelvin

    def publish(self):
        """Writes the nodes' temperatures back to the plate's points: a
        host array, or the CUDA array as it is on its stream."""
        if self.gpu:
            self.scene.write(
                PLATE,
                tokens.THERMAL_TEMPERATURES,
                self.state[0].flatten(),
                stream=wp.get_stream(self.device),
            )
        else:
            self.scene.write(
                PLATE, tokens.THERMAL_TEMPERATURES, self.kelvin.ravel()
            )

    def step(self):
        for _ in range(SUBSTEPS):
            power = heaters(self.time, self.heating)
            if self.gpu:
                self.power.assign(power)
                wp.launch(
                    step_gpu,
                    dim=(N, N),
                    inputs=[self.state[0], self.power, self.state[1]],
                    device=self.device,
                )
                self.state.reverse()
            else:
                self.kelvin = step_cpu(self.kelvin, power)
            self.time += DT
        self.publish()

    def render_all(self):
        """LdrColor, uint8 (rows, cols, 4), and the Temperature var in
        kelvin (rows, cols)."""
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Thermal"]
        )
        frame = outputs["/Render/Thermal"].frames[0]
        ldr = np.array(frame.render_vars["/Render/Thermal/LdrColor"].map())
        kelvin = np.array(frame.render_vars["/Render/Thermal/Temperature"].map())
        return ldr, kelvin.reshape(ldr.shape[0], ldr.shape[1])

    def save_output(self, path):
        """The thermal camera's image and the Temperature var side by
        side."""
        ldr, kelvin = self.render_all()
        ernie.demo.write_png(
            path,
            ernie.demo.side_by_side(
                [ldr[..., :3], ernie.demo.heatmap(kelvin, 270.0, 360.0)]
            ),
        )

    def test_final(self):
        nodes = self.read()
        print(
            f"after {self.time:.0f} s: nodes {nodes.min():.1f} to "
            f"{nodes.max():.1f} K"
        )
        assert nodes.max() > START + 20.0, "the chips should heat the plate"
        assert nodes.min() < START, "the plate's far corners should cool"
        # The two solvers agree, a step from the same state.
        power = heaters(self.time)
        cpu = step_cpu(nodes, power)
        if wp is not None and self._cuda():
            with wp.ScopedDevice("cuda:0"):
                out = wp.zeros((N, N), dtype=float)
                wp.launch(
                    step_gpu,
                    dim=(N, N),
                    inputs=[wp.array(nodes), wp.array(power), out],
                )
                assert np.allclose(out.numpy(), cpu, atol=1e-3), (
                    "the CPU and GPU solvers should agree"
                )
        # What the renderer shows is what the solver wrote.
        ldr, kelvin = self.render_all()
        hottest = np.unravel_index(np.argmax(kelvin), kelvin.shape)
        shown = kelvin[kelvin > 0]
        print(
            f"Temperature var: {shown.min():.1f} to {shown.max():.1f} K; "
            f"hottest pixel at {hottest}"
        )
        assert abs(float(kelvin.max()) - float(nodes.max())) < 2.0, (
            "the hottest pixel should read the hottest node"
        )
        brightness = ldr[..., :3].astype(np.float64).mean(axis=-1)
        assert brightness[hottest] > np.median(brightness) + 20.0, (
            "the hot spot should glow at 8 to 14 um"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
