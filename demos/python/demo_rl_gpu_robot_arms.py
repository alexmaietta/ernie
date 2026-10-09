"""Robot arms in thousands of worlds, posed on the GPU: a Warp kernel computes every arm's forward
kinematics into link positions and quaternions, and Ernie renders one view per world from those
arrays without the poses ever coming to the host, as a reinforcement learning loop whose
simulator keeps its state on the GPU.

    PYTHONPATH=build/python python demos/python/demo_rl_gpu_robot_arms.py [worlds] [--capture arms.png]

Each world ("environment") holds a three-link arm over a shared ground. The links are Xform prims
whose transform ops (`xformOp:translate`, `xformOp:orient`) are written as rows of two CUDA arrays
for all links at once (`Scene.write` of many paths); the boxes under them are posed by the GPU
between their world's transform and their own (`tokens.XFORM_EXTENT` bounds each link so nothing
is copied to the host). The cameras orbit their worlds, their poses also computed by a kernel as
positions and quaternions (`ernie:poses` as (views, 7)). Needs Warp (`pip install warp-lang`).
"""

import argparse
import time

import numpy as np
import warp as wp

import ernie
from demo_support import write_png
from ernie import tokens
from ernie.batch import BatchCamera

SIZE = 64
LINKS = 3
LENGTHS = (1.2, 1.0, 0.6)
STEPS = 120


@wp.kernel
def arm_poses(
    t: float,
    lengths: wp.vec3,
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
):
    """Forward kinematics of each world's arm: a yaw at the base, then three pitch joints."""
    world = wp.tid()
    phase = float(world) * 0.37
    up = wp.vec3(0.0, 0.0, 1.0)
    side = wp.vec3(0.0, 1.0, 0.0)
    frame = wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_from_axis_angle(up, 0.8 * t + phase))
    pitch0 = -0.7 - 0.4 * wp.sin(1.3 * t + phase)
    pitch1 = 1.1 + 0.5 * wp.sin(1.9 * t + 2.0 * phase)
    pitch2 = 0.7 * wp.sin(2.7 * t + phase)

    frame = wp.transform_multiply(frame, wp.transform(wp.vec3(), wp.quat_from_axis_angle(side, pitch0)))
    positions[world * 3] = wp.transform_get_translation(frame)
    orientations[world * 3] = wp.transform_get_rotation(frame)

    step = wp.transform(wp.vec3(lengths[0], 0.0, 0.0), wp.quat_from_axis_angle(side, pitch1))
    frame = wp.transform_multiply(frame, step)
    positions[world * 3 + 1] = wp.transform_get_translation(frame)
    orientations[world * 3 + 1] = wp.transform_get_rotation(frame)

    step = wp.transform(wp.vec3(lengths[1], 0.0, 0.0), wp.quat_from_axis_angle(side, pitch2))
    frame = wp.transform_multiply(frame, step)
    positions[world * 3 + 2] = wp.transform_get_translation(frame)
    orientations[world * 3 + 2] = wp.transform_get_rotation(frame)


@wp.kernel
def camera_poses(t: float, views: int, poses: wp.array2d(dtype=float)):
    """A view per world circling it: position, then the quaternion (x, y, z, w) of a camera
    looking down its -z at the arm, +y up."""
    view = wp.tid()
    heading = 0.2 * t + 6.2831853 * float(view) / float(views)
    eye = wp.vec3(5.0 * wp.cos(heading), 5.0 * wp.sin(heading), 3.0)
    forward = wp.normalize(wp.vec3(0.0, 0.0, 1.2) - eye)
    right = wp.normalize(wp.cross(forward, wp.vec3(0.0, 0.0, 1.0)))
    up = wp.cross(right, forward)
    rotation = wp.mat33(
        right[0], up[0], -forward[0], right[1], up[1], -forward[1], right[2], up[2], -forward[2]
    )
    q = wp.quat_from_matrix(rotation)
    poses[view, 0] = eye[0]
    poses[view, 1] = eye[1]
    poses[view, 2] = eye[2]
    poses[view, 3] = q[0]
    poses[view, 4] = q[1]
    poses[view, 5] = q[2]
    poses[view, 6] = q[3]


def build_scene(worlds):
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    scene.define("/Envs/Env0/Base", tokens.CYLINDER).set(tokens.GEOM_RADIUS, 0.35).set(
        tokens.GEOM_HEIGHT, 0.6
    ).set_xform(translate=(0.0, 0.0, 0.3)).set(tokens.DISPLAY_COLOR, (0.3, 0.3, 0.35))
    colors = [(0.9, 0.5, 0.1), (0.2, 0.6, 0.9), (0.9, 0.9, 0.9)]
    parent = "/Envs/Env0"
    for k in range(LINKS):
        # Each link is posed in its world's frame (not its parent link's), as simulators report
        # bodies, so the links are siblings.
        link = scene.define(f"{parent}/Link{k}", tokens.XFORM)
        link.set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT])
        link.set(tokens.XFORM_EXTENT, [-3.5, -3.5, -1.0, 3.5, 3.5, 4.0])
        scene.define(f"{parent}/Link{k}/Shape", tokens.CUBE).set_xform(
            translate=(LENGTHS[k] / 2, 0.0, 0.0), scale=(LENGTHS[k], 0.18, 0.18)
        ).set(tokens.DISPLAY_COLOR, colors[k])
    links = []
    for world in range(worlds):
        env = f"/Envs/Env{world}"
        if world > 0:
            scene.clone("/Envs/Env0", env)
        scene.set(env, tokens.WORLD, world)
        links += [f"{env}/Link{k}" for k in range(LINKS)]
    scene.define("/World/Camera", tokens.CAMERA).set(tokens.FOCAL_LENGTH, 18.0).set(
        tokens.SPECTRAL_BANDS, 3
    )
    # Looked up once and kept, as every step writes all of them.
    return scene, ernie.PrimSet(links)


def contact_sheet(images, columns=4):
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * SIZE, columns * SIZE, 4), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        sheet[r * SIZE : (r + 1) * SIZE, c * SIZE : (c + 1) * SIZE] = image
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("worlds", nargs="?", type=int, default=1024)
    parser.add_argument("--capture", help="write the first 16 worlds' last views here")
    args = parser.parse_args()

    renderer = ernie.Renderer()
    if not renderer.cuda_available:
        print("This demo needs CUDA interop")
        return
    wp.init()
    device = wp.get_device("cuda:0")
    stream = wp.get_stream(device)

    scene, links = build_scene(args.worlds)
    positions = wp.zeros(len(links), dtype=wp.vec3, device=device)
    orientations = wp.zeros(len(links), dtype=wp.quat, device=device)
    poses = wp.zeros((args.worlds, 7), dtype=float, device=device)
    camera = BatchCamera(renderer, scene, "/World/Camera", args.worlds, resolution=(SIZE, SIZE))
    camera.set_worlds(np.arange(args.worlds, dtype=np.int32))

    def step(t):
        wp.launch(arm_poses, args.worlds, [t, wp.vec3(*LENGTHS), positions, orientations], device=device)
        wp.launch(camera_poses, args.worlds, [t, args.worlds, poses], device=device)
        # The kernels' arrays as they are, on the stream they were written on.
        scene.write(links, tokens.TRANSLATE, positions, stream=stream.cuda_stream)
        scene.write(links, tokens.ORIENT, orientations, stream=stream.cuda_stream)
        return camera.step(poses, stream=stream.cuda_stream)

    for s in range(5):  # warm up: pipelines, allocations
        step(s / 30.0)
    start = time.perf_counter()
    for s in range(STEPS):
        observations = step((5 + s) / 30.0)
    images = renderer.cuda_download(observations["LdrColor"])
    per_step = (time.perf_counter() - start) / STEPS
    print(
        f"{args.worlds} worlds of {LINKS}-link arms ({len(links)} links posed on the GPU): "
        f"{per_step * 1e3:.2f} ms per step, {args.worlds / per_step:,.0f} views per second"
    )
    if args.capture:
        write_png(args.capture, contact_sheet(np.asarray(images)[:16]))
        print(f"Wrote {args.capture}")


if __name__ == "__main__":
    main()
