# SPDX-License-Identifier: BSD-3-Clause
"""Physics debug drawing: projectiles with drag bouncing on the ground,
a chain of springs and a spinning box, drawn over with trails coloured
by speed, dashed predictions, force arrows, fading contacts, x-ray
bounds, springs coloured by strain, frames, arcs and curves, labels
and numbers that follow the bodies, and a legend."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

PROJECTILES = 3
MASSES = 4
GRAVITY = 9.81
RADIUS = 0.25
DRAG = 0.02  # quadratic, per metre
RESTITUTION = 0.6
FRICTION = 0.8  # tangential speed kept per bounce
TRAIL = 120
SPRING = 60.0
DAMPING = 0.8
REST = 0.8
SPIN = 2.0  # radians per second
PIVOT = np.array([4.0, 0.0, 6.0])
BOX_AT = np.array([0.0, 4.0, 1.5])
SPIN_AXIS = np.array([0.3, 0.2, 1.0]) / math.sqrt(0.09 + 0.04 + 1.0)

SLOW = (0.2, 0.5, 1.0, 1.0)
FAST = (1.0, 0.25, 0.1, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
VELOCITY = (0.3, 1.0, 0.4, 1.0)
DRAG_COLOR = (1.0, 0.6, 0.1, 1.0)
CONTACT = (1.0, 0.9, 0.2, 1.0)
BOUNDS = (0.3, 0.9, 1.0, 0.7)
SPIN_COLOR = (1.0, 0.6, 0.2, 0.35)


def mix(a, b, t):
    """`a` to `b` by `t`, alpha included."""
    t = min(max(t, 0.0), 1.0)
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def spin_quat(angle):
    """The box's orientation as w, x, y, z."""
    s = math.sin(0.5 * angle)
    return (math.cos(0.5 * angle), *(s * SPIN_AXIS))


def box_transform(angle):
    w, x, y, z = spin_quat(angle)
    m = np.eye(4)
    # Rows of the rotation, column vectors as Ernie's transforms.
    m[:3, :3] = [
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
        ],
        [
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
        ],
        [
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
    ]
    m[:3, 3] = BOX_AT
    return m


class Projectile:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.wait = 0.0  # seconds until launch
        self.age = 0.0
        self.launches = 0
        self.trail = []  # positions with speeds


class Demo(ernie.demo.Demo):
    num_frames = 360

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_size=1.0,
        )
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.4, -0.6, 0.7)),
        )
        camera = world.define(
            "Camera", ernie.Camera, focal_length=20.0
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (6.0, -17.0, 8.0), (0.0, 1.0, 1.5)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(960, 540),
            overlays="*",
        )
        # A wall the predictions and bounds show through.
        world.define(
            "Wall",
            ernie.Cube,
            display_color=(0.5, 0.5, 0.55),
            translate=(-1.0, 1.0, 1.0),
            scale=(0.15, 1.5, 1.0),
        )
        self.box = world.define(
            "Box", ernie.Cube, size=0.6, display_color=(0.8, 0.3, 0.8)
        )

        # Labels and speeds posed under the bodies: added once.
        labels = scene.overlay("labels")
        self.projectiles = []
        self.balls = []
        for k in range(PROJECTILES):
            ball = world.define(
                f"Projectile{k}",
                ernie.Sphere,
                radius=RADIUS,
                display_color=(0.9, 0.35 + 0.2 * k, 0.2),
            )
            self.balls.append(ball)
            self.projectiles.append(Projectile())
            self.launch(k)
            self.projectiles[k].wait = 0.6 * k
            labels.label(
                (0.0, 0.0, RADIUS),
                f"p{k}",
                anchor=str(ball.path),
                depth="always",
            )
        self.speeds = np.zeros(PROJECTILES, np.float32)
        below = np.zeros((PROJECTILES, 3), np.float32)
        below[:, 2] = -3.0 * RADIUS
        labels.numbers(
            self.speeds,
            points=below,
            suffix=" m/s",
            decimals=1,
            anchors=[str(ball.path) for ball in self.balls],
            color=(0.6, 0.9, 1.0),
            depth="always",
            size=11,
        )
        self.mass_prims = [
            world.define(
                f"Mass{m}",
                ernie.Sphere,
                radius=0.2,
                display_color=(0.85, 0.85, 0.9),
            )
            for m in range(MASSES)
        ]
        self.masses = [
            PIVOT + np.array([REST * (m + 1), 0.0, 0.0])
            for m in range(MASSES)
        ]
        self.mass_velocities = [np.zeros(3) for _ in range(MASSES)]
        labels.label(
            (0.0, 0.0, 0.2),
            "bob",
            anchor=str(self.mass_prims[-1].path),
            depth="always",
        )
        labels.label(
            (0.0, 0.0, 0.4),
            "spinner",
            anchor=str(self.box.path),
            depth="always",
        )
        self.contacts = []  # [point, age]
        self.total_contacts = 0
        self.spin = 0.0
        self.place()
        self.draw()

    def launch(self, k):
        p = self.projectiles[k]
        n = p.launches + k
        speed = 11.0 + 1.5 * k
        elevation = math.radians(50.0 + 8.0 * (n % 3))
        azimuth = math.radians(10.0 - 12.0 * k)
        p.position = np.array([-8.0, -1.5 + 1.5 * k, RADIUS])
        p.velocity = speed * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        p.wait = 0.5
        p.age = 0.0
        p.trail = []
        p.launches += 1

    def step(self):
        dt = self.frame_dt
        for k, p in enumerate(self.projectiles):
            if p.wait > 0.0:
                p.wait -= dt
                continue
            speed = length(p.velocity)
            p.velocity = (
                p.velocity
                + (
                    np.array([0.0, 0.0, -GRAVITY])
                    - DRAG * speed * p.velocity
                )
                * dt
            )
            p.position = p.position + p.velocity * dt
            if p.position[2] < RADIUS:
                p.position[2] = RADIUS
                if p.velocity[2] < -0.5:
                    self.contacts.append(
                        [
                            np.array(
                                [p.position[0], p.position[1], 0.0]
                            ),
                            0.0,
                        ]
                    )
                    self.total_contacts += 1
                p.velocity[2] = max(-p.velocity[2] * RESTITUTION, 0.0)
                p.velocity[0] *= FRICTION
                p.velocity[1] *= FRICTION
            p.trail.append((p.position.copy(), length(p.velocity)))
            if len(p.trail) > TRAIL:
                p.trail.pop(0)
            p.age += dt
            if p.age > 6.0 or (
                length(p.velocity) < 0.4 and p.position[2] <= RADIUS
            ):
                self.launch(k)
        for contact in self.contacts:
            contact[1] += dt
        self.contacts = [c for c in self.contacts if c[1] <= 1.0]
        # The chain, semi-implicit Euler in substeps.
        substeps = 4
        h = dt / substeps
        for _ in range(substeps):
            forces = [
                np.array([0.0, 0.0, -GRAVITY]) for _ in range(MASSES)
            ]
            for m in range(MASSES):
                start = PIVOT if m == 0 else self.masses[m - 1]
                start_velocity = (
                    np.zeros(3)
                    if m == 0
                    else self.mass_velocities[m - 1]
                )
                d = self.masses[m] - start
                stretch = length(d)
                direction = d / stretch
                relative = self.mass_velocities[m] - start_velocity
                closing = (
                    relative[0] * direction[0]
                    + relative[1] * direction[1]
                    + relative[2] * direction[2]
                )
                f = (
                    -(SPRING * (stretch - REST) + DAMPING * closing)
                    * direction
                )
                forces[m] = forces[m] + f
                if m > 0:
                    forces[m - 1] = forces[m - 1] - f
            for m in range(MASSES):
                self.mass_velocities[m] = (
                    self.mass_velocities[m] + forces[m] * h
                )
                self.masses[m] = (
                    self.masses[m] + self.mass_velocities[m] * h
                )
        self.spin += SPIN * dt
        self.place()
        self.draw()

    def place(self):
        for k, (ball, p) in enumerate(
            zip(self.balls, self.projectiles)
        ):
            ball.set(translate=tuple(p.position))
            self.speeds[k] = length(p.velocity)
        self.scene.overlay("labels").touch()
        for prim, mass in zip(self.mass_prims, self.masses):
            prim.set(translate=tuple(mass))
        self.box.set(
            translate=tuple(BOX_AT),
            orient=ernie.Quat(
                *(float(c) for c in spin_quat(self.spin))
            ),
        )

    def draw(self):
        scene = self.scene
        # Trails: a segment per step, coloured by speed, fading with
        # age.
        trails = scene.overlay("trails")
        trails.clear()
        for p in self.projectiles:
            if len(p.trail) < 2:
                continue
            ends = []
            colors = []
            for i in range(1, len(p.trail)):
                ends += [p.trail[i - 1][0], p.trail[i][0]]
                color = mix(SLOW, FAST, p.trail[i][1] / 15.0)
                colors.append((*color[:3], i / len(p.trail)))
            trails.lines(np.array(ends), colors=colors, width=3)
        # Predictions without drag: dashed through walls, a cross
        # every quarter second.
        predicted = scene.overlay("predicted")
        predicted.clear()
        for p in self.projectiles:
            if p.wait > 0.0 or p.position[2] <= RADIUS:
                continue
            points = []
            for i in range(61):
                t = 0.05 * i
                q = (
                    p.position
                    + p.velocity * t
                    + np.array([0.0, 0.0, -0.5 * GRAVITY * t * t])
                )
                points.append(q)
                if i % 5 == 0 and i > 0:
                    predicted.marker(
                        q,
                        shape="cross",
                        size=8,
                        color=CONTACT,
                        depth="xray",
                    )
                if q[2] < RADIUS:
                    break
            predicted.polyline(
                points, dash=6, color=(1, 1, 1, 0.8), depth="xray"
            )
        # Velocity, drag and gravity on each projectile, velocity on
        # each mass.
        vectors = scene.overlay("vectors")
        vectors.clear()
        for p in self.projectiles:
            if p.wait > 0.0:
                continue
            vectors.arrow(
                p.position,
                p.position + 0.15 * p.velocity,
                color=VELOCITY,
                width=2.5,
            )
            vectors.arrow(
                p.position,
                p.position
                - 0.5 * DRAG * length(p.velocity) * p.velocity,
                color=DRAG_COLOR,
                width=2.5,
            )
            vectors.arrow(
                p.position,
                p.position + np.array([0.0, 0.0, -0.8]),
                color=FAST,
                width=2.5,
            )
        for mass, velocity in zip(self.masses, self.mass_velocities):
            vectors.arrow(
                mass,
                mass + 0.15 * velocity,
                color=VELOCITY,
                width=2.5,
            )
        # Contacts: rings growing and fading over a second, over
        # everything, with their normals.
        contacts = scene.overlay("contacts")
        contacts.clear()
        for at, age in self.contacts:
            color = (*CONTACT[:3], 1.0 - age)
            contacts.marker(
                at,
                shape="ring",
                size=10.0 + 40.0 * age,
                color=color,
                depth="always",
            )
            contacts.arrow(
                at,
                at + np.array([0.0, 0.0, 1.0 - age]),
                color=color,
                width=2,
                depth="always",
            )
        # Bounds in x-ray: boxes around the projectiles, the box's own
        # turned with it, spheres around the masses.
        bounds = scene.overlay("bounds")
        bounds.clear()
        bound = dict(width=1.5, color=BOUNDS, depth="xray")
        for p in self.projectiles:
            bounds.box(
                p.position - RADIUS, p.position + RADIUS, **bound
            )
        transform = box_transform(self.spin)
        bounds.box(
            (-0.3,) * 3, (0.3,) * 3, transform=transform, **bound
        )
        bound["color"] = (*BOUNDS[:3], 0.4)
        for mass in self.masses:
            bounds.sphere(mass, 0.3, **bound)
        # Springs: blue squeezed, white at rest, red stretched, wider
        # the more strained; and a rope curved through the chain.
        springs = scene.overlay("springs")
        springs.clear()
        for m in range(MASSES):
            start = PIVOT if m == 0 else self.masses[m - 1]
            strain = (length(self.masses[m] - start) - REST) / REST
            color = (
                mix(WHITE, FAST, 4.0 * strain)
                if strain > 0.0
                else mix(WHITE, SLOW, -4.0 * strain)
            )
            springs.line(
                start,
                self.masses[m],
                color=color,
                width=min(2.0 + 40.0 * abs(strain), 8.0),
            )
        springs.curve(
            PIVOT,
            self.masses[0],
            self.masses[2],
            self.masses[3],
            color=(0.8, 0.8, 1.0, 0.5),
            dash=3,
        )
        springs.marker(PIVOT, shape="diamond", size=10)
        # The box's frame, its spin axis and the turn it makes in half
        # a second, as a filled arc.
        frames = scene.overlay("frames")
        frames.clear()
        frames.axes(transform, 0.8)
        start = transform[:3, 0]
        frames.arc(
            BOX_AT,
            SPIN_AXIS,
            start,
            0.9,
            0.5 * SPIN,
            filled=True,
            color=SPIN_COLOR,
        )
        opaque = (*SPIN_COLOR[:3], 1.0)
        frames.arc(
            BOX_AT,
            SPIN_AXIS,
            start,
            0.9,
            0.5 * SPIN,
            color=opaque,
            width=2.5,
        )
        frames.arrow(
            BOX_AT,
            BOX_AT + 0.6 * SPIN * SPIN_AXIS,
            color=opaque,
            width=2.5,
        )
        self.draw_hud()

    def draw_hud(self):
        hud = self.scene.overlay("hud")
        hud.clear()
        launches = sum(p.launches for p in self.projectiles)
        hud.text(
            f"physics debug draw\nt = {self.frame * self.frame_dt:.2f} s\n"
            f"launches {launches}  contacts {self.total_contacts}",
            offset=(12, 12),
            size=14,
        )
        # The legend: a swatch and a name a row, down the left side.
        legend = [
            (FAST, "trail, by speed"),
            (WHITE, "prediction, dashed x-ray"),
            (VELOCITY, "velocity"),
            (DRAG_COLOR, "drag"),
            (CONTACT, "contact"),
            (BOUNDS, "bounds, x-ray"),
            (SLOW, "spring squeezed"),
            (SPIN_COLOR, "spin"),
        ]
        # Each item's top left corner at its place.
        place = dict(at=(0, 0.3), align=(0, 0))
        hud.rect(
            (196, 18 * len(legend) + 8),
            offset=(6, -6),
            color=(0, 0, 0, 0.5),
            **place,
        )
        for i, (color, name) in enumerate(legend):
            y = 18.0 * i
            hud.rect(
                (22, 6), offset=(12, y + 3), color=color[:3], **place
            )
            hud.text(
                name,
                offset=(42, y),
                size=12,
                background=(0, 0, 0, 0),
                **place,
            )

    def test_final(self):
        assert self.total_contacts > 2, "projectiles should bounce"
        assert self.projectiles[0].launches > 1, (
            "projectiles should launch again once settled"
        )
        for mass in self.masses:
            assert math.isfinite(mass[2]) and mass[2] < PIVOT[2], (
                "the chain should hang below its pivot"
            )
        for layer in (
            "trails",
            "predicted",
            "vectors",
            "bounds",
            "springs",
            "frames",
            "hud",
        ):
            assert not self.scene.overlay(layer).empty, (
                f"the {layer} layer should draw"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
