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
    scene.define("/World/Ground", ernie.GroundPlane)
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    # One environment, then clones of it; each world moves and recolours its box.
    env0 = scene.define("/Envs/Env0", ernie.Xform)
    env0.define("Box", ernie.Cube)
    for world in range(WORLDS):
        env = env0 if world == 0 else ernie.Xform(scene.clone("/Envs/Env0", f"/Envs/Env{world}"))
        env.world = world
        t = world / WORLDS
        box = ernie.Cube(scene.get_prim(f"{env.path}/Box"))
        box.set(translate=(3.0 * np.cos(2 * np.pi * t), 3.0 * np.sin(2 * np.pi * t), 1.0),
                display_color=(t, 0.4, 1.0 - t))

    # One camera, one pose and one world per view.
    pose = ernie.look_at_transform((0.0, -12.0, 8.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    camera = scene.define("/World/Camera", ernie.Camera, focal_length=18.0,
                          poses=np.stack([pose] * WORLDS),
                          worlds=np.arange(WORLDS, dtype=np.int32))
    scene.define("/Render/Batch", ernie.RenderProduct, camera=camera, resolution=(SIZE, SIZE))
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
