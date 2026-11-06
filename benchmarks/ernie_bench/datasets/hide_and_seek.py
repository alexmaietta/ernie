# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ernie_bench.datasets import shapes
from ernie_bench.scene import Cameras, DistantLight, Instances, Material, LissajousMotion, Scene, compose
from ernie_bench.config import SceneSpec

DEFAULTS = {
    "arena_size": 20.0,
    "wall_height": 2.0,
    "wall_thickness": 0.2,
    "door_width": 1.5,
    "hiders": 2,
    "seekers": 2,
    "boxes": 4,
    "ramps": 2,
    "agent_amplitude": [1.0, 4.0],
    "agent_frequency": [0.3, 0.8],
    "prop_amplitude": [0.2, 1.0],
    "prop_frequency": [0.05, 0.15],
    "fov_y": 60.0,
    "sun_direction": [-0.3, -0.2, -0.93],
    "sun_irradiance": float(np.pi),
    "ambient": 0.0,
    "near": 0.05,
    "far": 200.0,
    "eye_height": 0.8,
    "eye_forward": 0.3,
}

CLASSES = ("floor", "wall", "box", "ramp", "hider", "seeker")
FLOOR, WALL, BOX, RAMP, HIDER, SEEKER = range(len(CLASSES))
PLANE_MESH, BOX_MESH, RAMP_MESH, AGENT_MESH = range(4)

MATERIALS = [
    Material((0.55, 0.55, 0.50)),
    Material((0.75, 0.72, 0.68)),
    Material((0.80, 0.55, 0.25)),
    Material((0.35, 0.55, 0.35)),
    Material((0.20, 0.35, 0.85)),
    Material((0.85, 0.25, 0.20)),
]

LOOK_ALONG_X = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], np.float32
)


class Group(NamedTuple):
    mesh: int
    kind: int
    position: np.ndarray
    yaw: np.ndarray
    scale: np.ndarray


def _wall_segments(rng, start, end, door):
    door_center = rng.uniform(start + door, end - door)
    a = np.stack([start, door_center + door / 2], axis=1)
    b = np.stack([door_center - door / 2, end], axis=1)
    return (a + b) / 2, b - a


def _scattered(rng, mesh, kind, worlds, count, half, scale):
    position = np.zeros((worlds, count, 3))
    position[..., :2] = rng.uniform(-half + 1, half - 1, (worlds, count, 2))
    return Group(mesh, kind, position, rng.uniform(0, 2 * np.pi, (worlds, count)), scale)


def build(spec: SceneSpec, worlds: int, cameras_per_world: int) -> Scene:
    unknown = spec.params.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown hide_and_seek params {sorted(unknown)}")
    p = DEFAULTS | spec.params
    rng = np.random.default_rng(spec.seed)
    size, height, thickness, door = p["arena_size"], p["wall_height"], p["wall_thickness"], p["door_width"]
    half = size / 2
    full = np.full(worlds, half)

    outer = np.array([[0, -half, 0], [0, half, 0], [-half, 0, 0], [half, 0, 0]])
    outer_scale = np.array([[size + thickness, thickness, height]] * 2 + [[thickness, size + thickness, height]] * 2)

    split_x = rng.uniform(-0.2 * size, 0.2 * size, worlds)
    split_y = rng.uniform(-0.2 * size, 0.2 * size, worlds)
    vertical_center, vertical_length = _wall_segments(rng, -full, full, door)
    horizontal_center, horizontal_length = _wall_segments(rng, split_x, full, door)
    vertical = np.stack([np.repeat(split_x[:, None], 2, 1), vertical_center, np.zeros((worlds, 2))], -1)
    horizontal = np.stack([horizontal_center, np.repeat(split_y[:, None], 2, 1), np.zeros((worlds, 2))], -1)
    vertical_scale = np.stack([np.full((worlds, 2), thickness), vertical_length, np.full((worlds, 2), height)], -1)
    horizontal_scale = np.stack([horizontal_length, np.full((worlds, 2), thickness), np.full((worlds, 2), height)], -1)

    box_scale = np.concatenate(
        [rng.uniform(0.6, 1.4, (worlds, p["boxes"], 2)), np.ones((worlds, p["boxes"], 1))], -1
    )
    groups = [
        Group(PLANE_MESH, FLOOR, np.zeros((worlds, 1, 3)), np.zeros((worlds, 1)), np.broadcast_to([size, size, 1.0], (worlds, 1, 3))),
        Group(BOX_MESH, WALL, np.broadcast_to(outer, (worlds, 4, 3)), np.zeros((worlds, 4)), np.broadcast_to(outer_scale, (worlds, 4, 3))),
        Group(BOX_MESH, WALL, vertical, np.zeros((worlds, 2)), vertical_scale),
        Group(BOX_MESH, WALL, horizontal, np.zeros((worlds, 2)), horizontal_scale),
        _scattered(rng, BOX_MESH, BOX, worlds, p["boxes"], half, box_scale),
        _scattered(rng, RAMP_MESH, RAMP, worlds, p["ramps"], half, np.broadcast_to([2.0, 1.2, 0.8], (worlds, p["ramps"], 3))),
        _scattered(rng, AGENT_MESH, HIDER, worlds, p["hiders"], half, np.broadcast_to([0.5, 0.5, 1.0], (worlds, p["hiders"], 3))),
        _scattered(rng, AGENT_MESH, SEEKER, worlds, p["seekers"], half, np.broadcast_to([0.5, 0.5, 1.0], (worlds, p["seekers"], 3))),
    ]

    per_world = sum(g.position.shape[1] for g in groups)
    kind = np.concatenate([np.full(g.position.shape[1], g.kind) for g in groups])
    mesh = np.concatenate([np.full(g.position.shape[1], g.mesh) for g in groups])
    position = np.concatenate([g.position for g in groups], axis=1).reshape(-1, 3)
    yaw = np.concatenate([g.yaw for g in groups], axis=1).reshape(-1)
    scale = np.concatenate([g.scale for g in groups], axis=1).reshape(-1, 3)
    instance_kind = np.tile(kind, worlds)

    moving = np.flatnonzero(np.isin(instance_kind, [BOX, RAMP, HIDER, SEEKER]))
    agent = np.isin(instance_kind[moving], [HIDER, SEEKER])
    center = position[moving, :2]
    amplitude = np.where(
        agent[:, None],
        rng.uniform(*p["agent_amplitude"], (len(moving), 2)),
        rng.uniform(*p["prop_amplitude"], (len(moving), 2)),
    )
    motion = LissajousMotion(
        instance=moving,
        center=center,
        amplitude=np.clip(amplitude, 0, half - 0.75 - np.abs(center)),
        frequency=np.where(
            agent[:, None],
            rng.uniform(*p["agent_frequency"], (len(moving), 2)),
            rng.uniform(*p["prop_frequency"], (len(moving), 2)),
        ),
        phase=rng.uniform(0, 2 * np.pi, (len(moving), 2)),
        height=position[moving, 2],
        scale=scale[moving],
    )

    agents = np.flatnonzero(np.isin(kind, [HIDER, SEEKER]))
    if cameras_per_world and not len(agents):
        raise ValueError("hide_and_seek cameras ride on agents, but the scene has none")
    slot = np.arange(cameras_per_world)
    parent = (np.arange(worlds)[:, None] * per_world + agents[slot % len(agents)]).reshape(-1)
    heading = (slot // len(agents)) * np.pi / 2
    eye = np.column_stack([p["eye_forward"] * np.cos(heading), p["eye_forward"] * np.sin(heading), np.full(cameras_per_world, p["eye_height"])])
    mount = compose(eye, heading, np.ones((cameras_per_world, 3))) @ LOOK_ALONG_X

    scene = Scene(
        name=spec.name,
        world_count=worlds,
        meshes=[shapes.plane(), shapes.box(), shapes.ramp(), shapes.cylinder()],
        materials=MATERIALS,
        classes=CLASSES,
        instances=Instances(
            world=np.repeat(np.arange(worlds, dtype=np.int32), per_world),
            mesh=np.tile(mesh, worlds).astype(np.int32),
            material=instance_kind.astype(np.int32),
            semantic=instance_kind.astype(np.int32),
            transform=compose(position, yaw, scale),
        ),
        cameras=Cameras(
            world=np.repeat(np.arange(worlds, dtype=np.int32), cameras_per_world),
            parent=parent.astype(np.int64),
            local=np.tile(mount, (worlds, 1, 1)),
            fov_y=np.full(worlds * cameras_per_world, p["fov_y"], np.float32),
            near=np.full(worlds * cameras_per_world, p["near"], np.float32),
            far=np.full(worlds * cameras_per_world, p["far"], np.float32),
        ),
        lights=[DistantLight(direction=tuple(np.asarray(p["sun_direction"]) / np.linalg.norm(p["sun_direction"])), irradiance=p["sun_irradiance"])],
        motion=motion,
        ambient=p["ambient"],
    )
    scene.instances.transform[moving] = scene.state(0.0).transform
    return scene
