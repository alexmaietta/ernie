# SPDX-License-Identifier: BSD-3-Clause
"""Benchmarks batched cameras for reinforcement learning: one environment ("world") per view,
each a clone of the first, with camera poses that change every step on the GPU, and reports the
time per step and views rendered per second.

    PYTHONPATH=build/python python demos/python/demo_rl_batch_camera.py [views ...]

Each environment holds a few boxes and spheres over a shared ground under a sun with shadows. The
views orbit their environment, a step's poses arrive as a CUDA array, and the observations stay
on the GPU (`BatchCamera.step`), the way a training loop consumes them with torch.from_dlpack.
Spectral bands (`tokens.SPECTRAL_BANDS`) set most of the shading and sensor cost; RGB
observations need few.
"""

import sys
import time

import numpy as np

import ernie
from ernie import tokens
from ernie.batch import BatchCamera

SIZE = 64
STEPS = 60


def build_scene(worlds, bands):
    scene = ernie.Scene()
    scene.define("/World/Ground", ernie.GroundPlane)
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    env0 = scene.define("/Envs/Env0", ernie.Xform)
    rng = np.random.default_rng(7)
    for i in range(6):
        shape = ernie.Cube if i % 2 == 0 else ernie.Sphere
        env0.define(f"Thing{i}", shape, display_color=tuple(rng.uniform(0.2, 1.0, 3)),
                    translate=(*rng.uniform(-3.0, 3.0, 2), 0.5))
    for world in range(worlds):
        env = env0 if world == 0 else ernie.Xform(scene.clone("/Envs/Env0", f"/Envs/Env{world}"))
        env.world = world
    scene.define("/World/Camera", ernie.Camera, focal_length=18.0, spectral_bands=bands)
    return scene


def orbit(views, angle):
    """One pose per view, circling its environment from its own heading."""
    poses = np.empty((views, 4, 4), np.float32)
    for v in range(views):
        heading = angle + 2.0 * np.pi * v / views
        eye = (8.0 * np.cos(heading), 8.0 * np.sin(heading), 5.0)
        poses[v] = ernie.look_at_transform(eye, (0.0, 0.0, 0.5), up=(0.0, 0.0, 1.0))
    return poses


def benchmark(renderer, views, bands):
    scene = build_scene(views, bands)
    camera = BatchCamera(renderer, scene, "/World/Camera", views, resolution=(SIZE, SIZE))
    camera.set_worlds(np.arange(views, dtype=np.int32))
    poses = [renderer.cuda_upload(orbit(views, 0.03 * s)) for s in range(STEPS)]
    for p in poses[:5]:  # warm up: uploads, pipelines, allocations
        camera.step(p)
    start = time.perf_counter()
    observations = None
    for p in poses:
        observations = camera.step(p)
    renderer.cuda_download(observations["LdrColor"])  # the last step has finished
    per_step = (time.perf_counter() - start) / STEPS
    return per_step


def main():
    counts = [int(a) for a in sys.argv[1:]] or [256, 1024, 4096]
    renderer = ernie.Renderer()
    if not renderer.cuda_available:
        print("This demo needs CUDA interop")
        return
    print(f"{renderer.device_name}, {SIZE} x {SIZE} LdrColor observations")
    for bands in (3, 16):
        for views in counts:
            per_step = benchmark(renderer, views, bands)
            print(
                f"{views:6d} views, {bands:2d} bands: {per_step * 1e3:7.2f} ms per step, "
                f"{views / per_step:10,.0f} views per second"
            )


if __name__ == "__main__":
    main()
