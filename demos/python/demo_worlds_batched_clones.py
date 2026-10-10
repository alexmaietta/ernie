# SPDX-License-Identifier: BSD-3-Clause
"""Renders 16 environments ("worlds") in one batch, as a reinforcement learning loop renders one
camera per environment, and saves the views as a 4 x 4 contact sheet.

    PYTHONPATH=build/python python demos/python/demo_worlds_batched_clones.py [out.png]

Each environment is a clone of the first, tagged with its world: the environments overlap at the
origin, yet every view sees only its own world plus what no world claims (the ground and the
sun). One camera carries a pose and a world per view; a simulation updates them every step by
writing new arrays. Map the output with `device=ernie.Device.CUDA` to keep the images on the GPU
for PyTorch (`torch.from_dlpack`).
"""

import sys

import numpy as np

import ernie
from ernie import tokens

WORLDS, SIZE, COLUMNS = 16, 128, 4


def build_scene():
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    # One environment, then clones of it; each world moves and recolours its box.
    scene.define("/Envs/Env0", tokens.XFORM)
    scene.define("/Envs/Env0/Box", tokens.CUBE)
    for world in range(WORLDS):
        env = f"/Envs/Env{world}"
        if world > 0:
            scene.clone("/Envs/Env0", env)
        scene.set(env, tokens.WORLD, world)
        t = world / WORLDS
        offset = (3.0 * np.cos(2 * np.pi * t), 3.0 * np.sin(2 * np.pi * t), 1.0)
        transform = np.eye(4)
        transform[:3, 3] = offset
        scene.set(f"{env}/Box", tokens.TRANSFORM, transform)
        scene.set(f"{env}/Box", tokens.DISPLAY_COLOR, (t, 0.4, 1.0 - t))

    # One camera, one pose and one world per view.
    pose = ernie.look_at_transform((0.0, -12.0, 8.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    scene.define("/World/Camera", tokens.CAMERA).set(tokens.FOCAL_LENGTH, 18.0)
    scene.write("/World/Camera", tokens.POSES, np.stack([pose] * WORLDS))
    scene.write("/World/Camera", tokens.WORLDS, np.arange(WORLDS, dtype=np.int32))
    scene.define("/Render/Batch", tokens.RENDER_PRODUCT).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")
    ).set(tokens.RESOLUTION, (SIZE, SIZE))
    return scene


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_worlds_batched_clones.png"
    renderer = ernie.Renderer()
    product = renderer.step(build_scene(), render_products=["/Render/Batch"])["/Render/Batch"]
    views = product.frames[0].render_vars["/Render/Batch/LdrColor"].map()  # (16, 128, 128, 4)
    rows = WORLDS // COLUMNS
    sheet = views.reshape(rows, COLUMNS, SIZE, SIZE, 4).transpose(0, 2, 1, 3, 4)
    sheet = sheet.reshape(rows * SIZE, COLUMNS * SIZE, 4)
    try:
        from PIL import Image

        Image.fromarray(np.ascontiguousarray(sheet)).save(out)
        print(f"Rendered {WORLDS} worlds in one step into {out}")
    except ImportError:
        np.save(out.rsplit(".", 1)[0] + ".npy", sheet)
        print(f"Rendered {WORLDS} worlds in one step (install Pillow to save a PNG)")


if __name__ == "__main__":
    main()
