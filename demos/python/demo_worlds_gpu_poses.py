# SPDX-License-Identifier: BSD-3-Clause
"""Steps a batch of cameras whose poses live in CUDA memory, as a reinforcement learning loop
whose simulator keeps its state on the GPU, and times a step with the poses kept on the GPU
against downloading them first.

    PYTHONPATH=build/python python demos/python/demo_worlds_gpu_poses.py [views] [steps]

Poses written as a contiguous float32 CUDA array (any DLPack array: PyTorch, JAX, Warp, CuPy;
here `Renderer.cuda_upload` stands in for the simulator's) are copied device to device and each
view's matrices are computed on the GPU, so a step never waits for the host to read them: the CPU
prepares the frame while the simulator's queued GPU work still runs, which downloading the poses
would have to wait for. Stepping with `step_async` and mapping the outputs to CUDA without
waiting lets the CPU prepare the next step while the GPU renders this one.
Setting the camera's `poses_device` (`ernie:poses:device`) to "host" downloads them instead,
which lets CPU stages see exact poses (see `tokens.POSES_DEVICE`).
"""

import sys
import time

import numpy as np

import ernie
from ernie import tokens

SIZE = 64


def build_scene(views):
    scene = ernie.Scene()
    world = scene.define("/World", ernie.Xform)
    world.define("Ground", ernie.GroundPlane)
    world.define("Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    world.define("Box", ernie.Cube, size=2.0, translate=(0.0, 0.0, 1.0))
    camera = world.define("Camera", ernie.Camera, focal_length=18.0, poses=orbit(views, 0.0))
    scene.define("/Render/Batch", ernie.RenderProduct, camera=camera, resolution=(SIZE, SIZE))
    return scene, camera


def orbit(views, angle):
    """One camera per view around the box, each at its own heading, turned by `angle`."""
    poses = []
    for v in range(views):
        heading = angle + 2.0 * np.pi * v / views
        eye = (10.0 * np.cos(heading), 10.0 * np.sin(heading), 6.0)
        poses.append(ernie.look_at_transform(eye, (0.0, 0.0, 1.0), up=(0.0, 0.0, 1.0)))
    return np.asarray(poses, np.float32)


def main():
    views = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    renderer = ernie.Renderer()
    if not renderer.cuda_available:
        print("This demo needs CUDA interop")
        return
    scene, camera = build_scene(views)
    # The simulator's poses for every step, already on the GPU, and a stand-in for its physics:
    # a large device-to-device copy queued on the same stream before each step's poses.
    on_gpu = [renderer.cuda_upload(orbit(views, 0.05 * s)) for s in range(steps)]
    state = [renderer.cuda_upload(np.zeros(64 << 20, np.float32)) for _ in range(2)]
    def time_steps(device, physics):
        camera.poses_device = device
        start = time.perf_counter()
        for poses in on_gpu:
            if physics:
                renderer.cuda_copy(state[0], state[1])
            camera.write(tokens.POSES, poses)
            # Submit and take the outputs without waiting: the CUDA mapping orders on the GPU,
            # and the next step is prepared while this one renders.
            step = renderer.step_async(scene, render_products=["/Render/Batch"])
            product = step.outputs()["/Render/Batch"]
            product.frames[0].render_vars["/Render/Batch/LdrColor"].map(device=ernie.Device.CUDA)
        return (time.perf_counter() - start) / steps * 1e3

    time_steps(tokens.GPU, False)  # warm up
    for physics in (False, True):
        times = {device: time_steps(device, physics) for device in (tokens.GPU, tokens.HOST)}
        print(
            f"{views} views{', after queued physics' if physics else ''}: "
            f"{times[tokens.GPU]:.2f} ms per step with poses on the GPU, "
            f"{times[tokens.HOST]:.2f} ms downloading them"
        )

if __name__ == "__main__":
    main()
