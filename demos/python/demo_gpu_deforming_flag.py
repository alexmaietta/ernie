"""A flag waving on its pole, its mesh deformed every frame by a Warp kernel on the GPU.

    PYTHONPATH=build/python python demos/python/demo_gpu_deforming_flag.py [--capture flag.png]

Needs Warp (`uv pip install warp-lang`) and CUDA interop. A Warp kernel writes the flag's points
and vertex colours into Warp arrays each frame, and `scene.write` hands Ernie those arrays on the
kernel's stream: they are copied device to device into the renderer's own buffers after the
kernel, and nothing passes through host memory. The flag is authored with normals, so Ernie
recomputes them on the GPU from the moved points (area-weighted around each vertex), and its
`extent` bounds every pose, so Ernie need not read the points back to bound them either.
"""

import argparse

import numpy as np
import warp as wp

import ernie
from ernie import tokens

COLUMNS = 160  # vertices along the flag, from the pole
ROWS = 96  # vertices up the flag
WIDTH = 4.0  # metres
HEIGHT = 2.4
FOOT = 2.0  # height of the flag's lower edge


@wp.kernel
def wave(points: wp.array(dtype=wp.vec3), colors: wp.array(dtype=wp.vec3), time: float):
    """The flag at `time`: travelling waves growing from the pole, and its stripes shaded by
    them."""
    tid = wp.tid()
    i = tid % COLUMNS
    j = tid // COLUMNS
    u = float(i) / float(COLUMNS - 1)  # 0 at the pole
    v = float(j) / float(ROWS - 1)  # 0 at the foot
    phase = 7.0 * u - 4.0 * time + 0.9 * v
    gust = 0.75 + 0.25 * wp.sin(0.7 * time)
    amplitude = 0.8 * u * gust
    y = amplitude * (wp.sin(phase) + 0.35 * wp.sin(2.3 * phase + 1.7 * v + 1.3 * time))
    # Folds pull the free edge in and let it sag a little.
    x = u * WIDTH * (1.0 - 0.05 * u * gust * wp.cos(phase) * wp.cos(phase))
    z = FOOT + v * HEIGHT - 0.12 * u * u * (1.0 - gust)
    points[tid] = wp.vec3(x, y, z)
    # Three stripes, lighter on the crests.
    stripe = wp.vec3(0.05, 0.12, 0.6)
    if v > 0.333:
        stripe = wp.vec3(0.85, 0.85, 0.8)
    if v > 0.667:
        stripe = wp.vec3(0.7, 0.05, 0.04)
    colors[tid] = stripe * (0.8 + 0.2 * wp.sin(phase))


def flag_triangles():
    """Two triangles per grid quad, counter-clockwise seen from -y."""
    i, j = np.meshgrid(np.arange(COLUMNS - 1), np.arange(ROWS - 1))
    a = (j * COLUMNS + i).ravel()
    b, c, d = a + 1, a + COLUMNS + 1, a + COLUMNS
    return np.stack([a, b, c, a, c, d], axis=1).reshape(-1).astype(np.int32)


def build_scene(points, colors):
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE).set(tokens.CHECKER, True).set(
        tokens.CHECKER_SIZE, 1.0)
    scene.define("/World/Pole", tokens.CYLINDER).set(tokens.GEOM_RADIUS, 0.05).set(
        tokens.GEOM_HEIGHT, FOOT + HEIGHT + 0.2).set(
        tokens.TRANSFORM, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5 * (FOOT + HEIGHT + 0.2)],
                           [0, 0, 0, 1]]).set(
        tokens.DISPLAY_COLOR, (0.7, 0.7, 0.7))
    indices = flag_triangles()
    count = COLUMNS * ROWS
    normals = np.tile(np.array([0.0, -1.0, 0.0], np.float32), (count, 1))
    flag = ernie.Path("/World/Flag")
    scene.define(flag, tokens.MESH).write(tokens.FACE_VERTEX_COUNTS,
                                          np.full(indices.size // 3, 3, np.int32)).write(
        tokens.FACE_VERTEX_INDICES, indices).write(tokens.NORMALS, normals).set(
        # Every pose the kernel can give, so bounds never need the points on the host.
        tokens.EXTENT, [-0.1, -1.2, FOOT - 0.3, WIDTH + 0.1, 1.2, FOOT + HEIGHT + 0.1])
    scene.write(flag, tokens.POINTS, points)
    scene.write(flag, tokens.DISPLAY_COLOR, colors)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((-0.5, -0.8, 0.6)))
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((8.5, -4.5, 4.2), (1.6, 0.0, 2.9))).set(
        tokens.FOCAL_LENGTH, 30.0)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera"))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", help="render without a window and save the image here")
    args = parser.parse_args()

    wp.init()
    device = "cuda:0"
    count = COLUMNS * ROWS
    points = wp.zeros(count, dtype=wp.vec3, device=device)
    colors = wp.zeros(count, dtype=wp.vec3, device=device)
    stream = wp.get_stream(device)
    wp.launch(wave, dim=count, inputs=[points, colors, 0.0], device=device)
    scene = build_scene(points, colors)

    from ernie.app import Viewer
    viewer = Viewer(scene, title="Ernie: a flag deformed by a Warp kernel",
                    hidden=args.capture is not None)
    clock = {"time": 0.0}

    def update(v, info):
        clock["time"] += min(info.delta, 1.0 / 30.0) if args.capture is None else 1.0 / 60.0
        wp.launch(wave, dim=count, inputs=[points, colors, clock["time"]], device=device)
        # Ernie copies the arrays after the kernel, on its stream; nothing waits on the host.
        v.scene.write(flag_path, tokens.POINTS, points, stream=stream)
        v.scene.write(flag_path, tokens.DISPLAY_COLOR, colors, stream=stream)

    flag_path = ernie.Path("/World/Flag")
    viewer.active_camera = ernie.Path("/World/Camera")
    if args.capture is not None:
        display = viewer.viewport_display
        display.lights = display.grid = display.cameras = False
        viewer.viewport_display = display
    viewer.on_frame(update)
    if args.capture is None:
        viewer.run()
        return
    for _ in range(60):
        viewer.step()
    viewer.capture(args.capture)
    viewer.step()


if __name__ == "__main__":
    main()
