"""Performance regression check: steps a fixed set of batched scenes the way a training loop does
(poses on the GPU, observations mapped to CUDA, no host waits) and reports the median time per
step of each, against baselines recorded per GPU.

    PYTHONPATH=build/python python tools/benchmark.py            # report
    PYTHONPATH=build/python python tools/benchmark.py --check    # fail on regressions
    PYTHONPATH=build/python python tools/benchmark.py --update   # record this GPU's baseline

Baselines live in tools/benchmark_baselines.json by device name; a GPU without one only reports.
Profile a regression with a Tracy build (README, Profiling). Needs CUDA interop.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import ernie
from ernie import tokens
from ernie.batch import BatchCamera

BASELINES = Path(__file__).with_name("benchmark_baselines.json")
SIZE = 64


def orbit(views, angle, radius, height, target):
    poses = np.empty((views, 4, 4), np.float32)
    for v in range(views):
        heading = angle + 2.0 * np.pi * v / views
        eye = (radius * np.cos(heading), radius * np.sin(heading), height)
        poses[v] = ernie.look_at_transform(eye, target, up=(0.0, 0.0, 1.0))
    return poses


def instanced_worlds(views, bands):
    """16 worlds of 200 instanced spheres each, a box and a ground, under a sun with shadows."""
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    box = np.eye(4)
    box[2, 3] = 1.0
    scene.define("/World/Box", tokens.CUBE).set(tokens.SIZE, 2.0).set(tokens.TRANSFORM, box)
    rng = np.random.default_rng(1)
    for w in range(16):
        scene.define(f"/Envs/E{w}", tokens.XFORM).set(tokens.WORLD, w)
        positions = rng.uniform(-8, 8, (200, 3)).astype(np.float32) * [1, 1, 0.2] + [0, 0, 1]
        scene.define(f"/Envs/E{w}/Balls", tokens.POINT_INSTANCER).write(
            tokens.POSITIONS, positions.astype(np.float32)
        ).set(tokens.PROTOTYPES, [ernie.Path(f"/Envs/E{w}/Balls/Ball")])
        scene.define(f"/Envs/E{w}/Balls/Ball", tokens.SPHERE).set(tokens.GEOM_RADIUS, 0.3)
    scene.define("/World/Camera", tokens.CAMERA).set(tokens.FOCAL_LENGTH, 18.0).set(
        tokens.SPECTRAL_BANDS, bands
    )
    return scene, np.arange(views, dtype=np.int32) % 16, (10.0, 6.0, (0.0, 0.0, 1.0))


def cloned_worlds(views, bands):
    """A world per view, each a clone of six boxes and spheres, over a shared ground."""
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    scene.define("/Envs/Env0", tokens.XFORM)
    rng = np.random.default_rng(7)
    for i in range(6):
        prim = scene.define(f"/Envs/Env0/Thing{i}", tokens.CUBE if i % 2 == 0 else tokens.SPHERE)
        transform = np.eye(4)
        transform[:3, 3] = (*rng.uniform(-3.0, 3.0, 2), 0.5)
        prim.set(tokens.TRANSFORM, transform)
    for world in range(views):
        if world > 0:
            scene.clone("/Envs/Env0", f"/Envs/Env{world}")
        scene.set(f"/Envs/Env{world}", tokens.WORLD, world)
    scene.define("/World/Camera", tokens.CAMERA).set(tokens.FOCAL_LENGTH, 18.0).set(
        tokens.SPECTRAL_BANDS, bands
    )
    return scene, np.arange(views, dtype=np.int32), (8.0, 5.0, (0.0, 0.0, 0.5))


SCENARIOS = {
    "instanced worlds, 256 views, 16 bands": (instanced_worlds, 256, 16),
    "instanced worlds, 1024 views, 16 bands": (instanced_worlds, 1024, 16),
    "cloned worlds, 1024 views, 3 bands": (cloned_worlds, 1024, 3),
    "cloned worlds, 4096 views, 3 bands": (cloned_worlds, 4096, 3),
}


def run(renderer, build, views, bands, windows=5, steps=30):
    scene, worlds, (radius, height, target) = build(views, bands)
    camera = BatchCamera(renderer, scene, "/World/Camera", views, resolution=(SIZE, SIZE))
    camera.set_worlds(worlds)
    poses = [renderer.cuda_upload(orbit(views, 0.03 * s, radius, height, target)) for s in range(16)]
    for s in range(10):  # warm up: pipelines, allocations, the first syncs
        camera.step(poses[s % len(poses)])
    times = []
    for _ in range(windows):
        start = time.perf_counter()
        for s in range(steps):
            observations = camera.step(poses[s % len(poses)])
        renderer.cuda_download(observations["LdrColor"])  # the window's steps have finished
        times.append((time.perf_counter() - start) / steps * 1e3)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail on regressions past tolerance")
    parser.add_argument("--update", action="store_true", help="record this GPU's baseline")
    parser.add_argument("--tolerance", type=float, default=0.15, help="allowed slowdown (0.15)")
    parser.add_argument("--only", help="run the scenarios whose names contain this")
    args = parser.parse_args()

    renderer = ernie.Renderer()
    if not renderer.cuda_available:
        print("The benchmark needs CUDA interop")
        return 2
    device = renderer.device_name
    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}
    baseline = baselines.get(device, {})
    print(f"{device}, {SIZE} x {SIZE} LdrColor, median ms per step")
    results = {}
    regressions = []
    for name, (build, views, bands) in SCENARIOS.items():
        if args.only and args.only not in name:
            continue
        ms = run(renderer, build, views, bands)
        results[name] = round(ms, 3)
        line = f"  {name:40} {ms:8.2f} ms ({views / ms * 1e3:9,.0f} views/s)"
        if name in baseline:
            change = ms / baseline[name] - 1.0
            line += f"   baseline {baseline[name]:8.2f} ms, {change:+6.1%}"
            if change > args.tolerance:
                regressions.append(name)
                line += "  REGRESSION"
        print(line, flush=True)
    if args.update:
        baselines[device] = {**baseline, **results}
        BASELINES.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
        print(f"Recorded the baseline of {device} in {BASELINES}")
    if not baseline and not args.update:
        print(f"No baseline for {device}: record one with --update")
    if args.check and regressions:
        print(f"{len(regressions)} regression(s) past {args.tolerance:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
