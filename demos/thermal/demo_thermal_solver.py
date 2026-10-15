# SPDX-License-Identifier: BSD-3-Clause
"""A thermal solver's write-back: an avionics board whose mesh points are the
nodes of a nodal network (conduction to its neighbours, three chips heating in
turn, radiation from both faces to the cold surroundings), stepped every frame
and seen through an 8 to 14 um camera. The solver runs in NumPy, whose host
arrays the renderer uploads in place, or in Warp on the GPU, whose CUDA arrays
it copies on the device with no trip through the host; switch between them
as it runs. Beside it an avionics box is a coarser network, a node a panel
and one for its electronics: its points map to their panel's node once, and
the solver then writes its seven nodes alone.

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
CAPACITY = (
    DENSITY * SPECIFIC_HEAT * THICKNESS * DX * DX
)  # J/K per node
CONDUCTANCE = CONDUCTIVITY * THICKNESS  # W/K between neighbours
RADIATING = 2.0 * EMISSIVITY * SIGMA * DX * DX  # W/K^4 per node
SINK4 = SINK**4
STEP = DT / CAPACITY  # K per joule
# The chips: their nodes' first row and column, size, and power in watts.
CHIPS = [(14, 12, 8, 18.0), (36, 20, 10, 30.0), (20, 42, 6, 12.0)]
PLATE = ernie.Path("/World/Plate")
# The avionics box: six aluminium panels (+x, -x, +y, -y, +z, -z) and the
# electronics inside, a lumped node each, each panel's points mapped to its.
BOX = ernie.Path("/World/Box")
BOX_SIDE = 0.2  # m
BOX_CENTER = (0.45, 0.1, 0.25)
FACES = 6
PANEL_CAPACITY = 100.0  # J/K: 1 mm of aluminium
INTERIOR_CAPACITY = 400.0  # J/K
MOUNTING = 2.0  # W/K from the electronics to each panel
BOX_RADIATING = (
    EMISSIVITY * SIGMA * BOX_SIDE * BOX_SIDE
)  # W/K^4 a panel
ELECTRONICS = 80.0  # W
SUNLIGHT = 20.0  # W the top panel absorbs


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
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    flow = (
        np.float32(CONDUCTANCE)
        * (neighbours - np.float32(4.0) * kelvin)
        + power
        - np.float32(RADIATING)
        * (kelvin * kelvin * kelvin * kelvin - np.float32(SINK4))
    )
    return (kelvin + np.float32(STEP) * flow).astype(np.float32)


def box_power(on=True):
    """Watts into the box's nodes: sunlight on its top panel, and its
    electronics' when on."""
    power = np.zeros(FACES + 1, dtype=np.float32)
    power[4] = SUNLIGHT
    power[FACES] = ELECTRONICS if on else 0.0
    return power


def box_step_cpu(kelvin, power):
    """One explicit step of the box's network: the panels conduct to the
    electronics and radiate away."""
    panels = kelvin[:FACES]
    mounted = np.float32(MOUNTING) * (kelvin[FACES] - panels)
    flow = np.empty_like(kelvin)
    flow[:FACES] = (
        mounted
        + power[:FACES]
        - np.float32(BOX_RADIATING) * (panels**4 - np.float32(SINK4))
    )
    flow[FACES] = power[FACES] - mounted.sum()
    capacity = np.array(
        [PANEL_CAPACITY] * FACES + [INTERIOR_CAPACITY], np.float32
    )
    return (kelvin + np.float32(DT) * flow / capacity).astype(
        np.float32
    )


if wp is not None:

    @wp.kernel
    @no_type_check
    def box_step_gpu(
        kelvin: wp.array(dtype=float),
        power: wp.array(dtype=float),
        out: wp.array(dtype=float),
    ):
        """`box_step_cpu`, a thread a node."""
        i = wp.tid()
        t = kelvin[i]
        if i < FACES:
            flow = (
                float(MOUNTING) * (kelvin[FACES] - t)
                + power[i]
                - float(BOX_RADIATING)
                * (t * t * t * t - float(SINK4))
            )
            out[i] = t + float(DT) * flow / float(PANEL_CAPACITY)
        else:
            heat = power[i]
            for k in range(FACES):
                heat -= float(MOUNTING) * (t - kelvin[k])
            out[i] = t + float(DT) * heat / float(INTERIOR_CAPACITY)

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
        [
            i * DX - 0.5 * SIDE,
            np.zeros_like(i, dtype=float),
            0.3 + j * DX,
        ],
        axis=-1,
    ).reshape(-1, 3)
    corner = (
        np.arange(N - 1)[:, None] * N + np.arange(N - 1)[None, :]
    ).ravel()
    quads = np.stack(
        [corner, corner + N, corner + N + 1, corner + 1], axis=-1
    )
    return points, quads


def box_mesh():
    """The box's six panels, four points each of their own, and each point's
    node: its panel's."""
    half = 0.5 * BOX_SIDE
    points, ids = [], []
    for axis in range(3):
        normal = np.eye(3)[axis]
        u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
        for sign in (1.0, -1.0):
            if sign < 0.0:  # counter-clockwise seen from outside
                u, v = v, u
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                points.append(
                    np.array(BOX_CENTER)
                    + half * (sign * normal + a * u + b * v)
                )
                ids.append(2 * axis + (0 if sign > 0.0 else 1))
    return (
        np.array(points),
        np.arange(4 * FACES),
        np.array(ids, dtype=np.int32),
    )


class Demo(ernie.demo.Demo):
    num_frames = 40
    fixed_dt = 1.0 / 30.0

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.gpu = wp is not None and self._cuda()
        self.heating = True
        self.time = 0.0
        self.kelvin = np.full((N, N), START, dtype=np.float32)
        self.box = np.full(FACES + 1, START, dtype=np.float32)
        if self.gpu:
            self.upload()
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
            thermal_emissivity=[
                8000.0,
                EMISSIVITY,
                14000.0,
                EMISSIVITY,
            ],
        )
        world.define(
            "Ground",
            ernie.GroundPlane,
            material_binding=concrete.path,
        )
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
        # The box on a plinth beside it, its panels' points mapped to their
        # nodes once; the solver then writes the nodes alone.
        points, indices, ids = box_mesh()
        scene.define(
            BOX,
            ernie.Mesh,
            points=points,
            face_vertex_counts=np.full(FACES, 4),
            face_vertex_indices=indices,
            material_binding=paint.path,
        )
        scene.write(BOX, tokens.THERMAL_NODE_IDS, ids)
        world.define(
            "Plinth",
            ernie.Cube,
            size=1.0,
            material_binding=concrete.path,
            translate=(BOX_CENTER[0], BOX_CENTER[1], 0.075),
            scale=(0.16, 0.16, 0.15),
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
                (0.45, -1.15, 0.75),
                (0.12, 0.0, 0.45),
                (0.0, 0.0, 1.0),
            ),
        )
        variables = []
        for name in ("LdrColor", "Temperature"):
            var = scene.define(
                f"/Render/Thermal/{name}", ernie.RenderVar
            )
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
                self.kelvin, self.box = self.read()
                self.gpu = gpu
                if gpu:
                    self.upload()

        self.widgets.combo(
            "Solver",
            ["CPU (NumPy)", "GPU (Warp)"],
            1 if self.gpu else 0,
            on_change=solver,
        )
        self.widgets.checkbox(
            "Chips on",
            True,
            on_change=lambda on: setattr(self, "heating", on),
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    @staticmethod
    def _cuda():
        wp.init()
        return wp.is_cuda_available()

    def upload(self):
        """Moves both networks' state to the GPU."""
        self.device = "cuda:0"
        with wp.ScopedDevice(self.device):
            self.state = [
                wp.array(self.kelvin),
                wp.zeros((N, N), dtype=float),
            ]
            self.power = wp.zeros((N, N), dtype=float)
            self.box_state = [
                wp.array(self.box),
                wp.zeros(FACES + 1, dtype=float),
            ]
            self.box_heat = wp.zeros(FACES + 1, dtype=float)

    def read(self):
        """The plate's and the box's node temperatures, on the host."""
        if self.gpu:
            return self.state[0].numpy(), self.box_state[0].numpy()
        return self.kelvin, self.box

    def publish(self):
        """Writes the nodes' temperatures back: the plate's to its points,
        the box's to its nodes, which its points map to. Host arrays, or the
        CUDA arrays as they are on their stream."""
        if self.gpu:
            stream = wp.get_stream(self.device)
            self.scene.write(
                PLATE,
                tokens.THERMAL_TEMPERATURES,
                self.state[0].flatten(),
                stream=stream,
            )
            self.scene.write(
                BOX,
                tokens.THERMAL_NODE_TEMPERATURES,
                self.box_state[0],
                stream=stream,
            )
        else:
            self.scene.write(
                PLATE,
                tokens.THERMAL_TEMPERATURES,
                self.kelvin.ravel(),
            )
            self.scene.write(
                BOX, tokens.THERMAL_NODE_TEMPERATURES, self.box
            )

    def step(self):
        box_heat = box_power(self.heating)
        for _ in range(SUBSTEPS):
            power = heaters(self.time, self.heating)
            if self.gpu:
                self.power.assign(power)
                self.box_heat.assign(box_heat)
                wp.launch(
                    step_gpu,
                    dim=(N, N),
                    inputs=[self.state[0], self.power, self.state[1]],
                    device=self.device,
                )
                wp.launch(
                    box_step_gpu,
                    dim=FACES + 1,
                    inputs=[
                        self.box_state[0],
                        self.box_heat,
                        self.box_state[1],
                    ],
                    device=self.device,
                )
                self.state.reverse()
                self.box_state.reverse()
            else:
                self.kelvin = step_cpu(self.kelvin, power)
                self.box = box_step_cpu(self.box, box_heat)
            self.time += DT
        self.publish()

    def render_all(self):
        """LdrColor, uint8 (rows, cols, 4), and the Temperature var in
        kelvin (rows, cols)."""
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Thermal"]
        )
        frame = outputs["/Render/Thermal"].frames[0]
        ldr = np.array(
            frame.render_vars["/Render/Thermal/LdrColor"].map()
        )
        kelvin = np.array(
            frame.render_vars["/Render/Thermal/Temperature"].map()
        )
        return ldr, kelvin.reshape(ldr.shape[0], ldr.shape[1])

    def save_output(self, path):
        """The thermal camera's image and the Temperature var side by
        side."""
        ldr, kelvin = self.render_all()
        ernie.demo.write_png(
            path,
            ernie.demo.side_by_side(
                [
                    ldr[..., :3],
                    ernie.demo.heatmap(kelvin, 270.0, 360.0),
                ]
            ),
        )

    def test_final(self):
        nodes, box = self.read()
        print(
            f"after {self.time:.0f} s: nodes {nodes.min():.1f} to "
            f"{nodes.max():.1f} K; box panels {np.round(box[:FACES], 1)}, "
            f"electronics {box[FACES]:.1f} K"
        )
        assert nodes.max() > START + 20.0, (
            "the chips should heat the plate"
        )
        assert nodes.min() < START, (
            "the plate's far corners should cool"
        )
        assert box[4] > box[:4].max() + 2.0, (
            "the sunlit top should be warmest"
        )
        assert box[:4].min() > START + 10.0, (
            "the electronics should heat it"
        )
        # The two solvers agree, a step from the same state.
        power = heaters(self.time)
        cpu = step_cpu(nodes, power)
        box_cpu = box_step_cpu(box, box_power())
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
                box_out = wp.zeros(FACES + 1, dtype=float)
                wp.launch(
                    box_step_gpu,
                    dim=FACES + 1,
                    inputs=[
                        wp.array(box),
                        wp.array(box_power()),
                        box_out,
                    ],
                )
                assert np.allclose(
                    box_out.numpy(), box_cpu, atol=1e-3
                ), "the CPU and GPU box networks should agree"
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
        # The box's top and front panels each read their node, flat.
        for face in (4, 3):
            panel = np.count_nonzero(
                np.abs(kelvin - box[face]) < 0.05
            )
            print(
                f"panel {face}: {panel} pixels at {box[face]:.2f} K"
            )
            assert panel > 200, (
                "a panel's pixels should read its node"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
