# SPDX-License-Identifier: BSD-3-Clause
"""Renders 16 environments ("worlds") in one batch, as a reinforcement learning loop renders one
camera per environment, and saves the views as a 4 x 4 contact sheet.

    PYTHONPATH=build/python python demos/python/demo_worlds_batched_clones.py
    PYTHONPATH=build/python python demos/python/demo_worlds_batched_clones.py --viewer null \
        --output sheet.png

Each environment is a clone of the first, tagged with its world: the environments overlap at the
origin, yet every view sees only its own world plus what no world claims (the ground and the
sun). One camera carries a pose and a world per view; a simulation updates them every step by
writing new arrays. Map the output with `device=ernie.Device.CUDA` to keep the images on the GPU
for PyTorch (`torch.from_dlpack`). The viewer, and the scene's own camera, see every world at
once: the boxes in a ring.
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

WORLDS, SIZE, COLUMNS = 16, 128, 4


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        scene.define("/World/Ground", ernie.GroundPlane)
        scene.define("/World/Sun", ernie.DistantLight).set(
            tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7)))
        # One environment, then clones of it; each world moves and recolours its box.
        env0 = scene.define("/Envs/Env0", ernie.Xform)
        env0.define("Box", ernie.Cube)
        for world in range(WORLDS):
            env = env0 if world == 0 else ernie.Xform(scene.clone("/Envs/Env0",
                                                                  f"/Envs/Env{world}"))
            env.world = world
            t = world / WORLDS
            box = ernie.Cube.at(env, "Box")
            box.set(translate=(3.0 * np.cos(2 * np.pi * t), 3.0 * np.sin(2 * np.pi * t), 1.0),
                    display_color=(t, 0.4, 1.0 - t))

        # An overview of every world, for the viewer's absence (--viewer null) and tests.
        pose = ernie.look_at_transform((0.0, -12.0, 8.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
        overview = scene.define("/World/Overview", ernie.Camera, focal_length=18.0)
        overview.set(tokens.TRANSFORM, pose)
        scene.define("/Render/Settings", ernie.RenderSettings, camera=overview,
                     resolution=(640, 480))

        # One camera, one pose and one world per view.
        camera = scene.define("/World/Camera", ernie.Camera, focal_length=18.0,
                              poses=np.stack([pose] * WORLDS),
                              worlds=np.arange(WORLDS, dtype=np.int32))
        scene.define("/Render/Batch", ernie.RenderProduct, camera=camera, resolution=(SIZE, SIZE))
        self.views = None

    def step(self):
        # The scene is still: render the batch once, all 16 views in one step.
        if self.views is None:
            product = self.renderer.step(self.scene, render_products=["/Render/Batch"])
            frame = product["/Render/Batch"].frames[0]
            self.views = np.array(frame.render_vars["/Render/Batch/LdrColor"].map())
            print(f"Rendered {WORLDS} worlds in one step")

    def contact_sheet(self):
        rows = WORLDS // COLUMNS
        sheet = self.views.reshape(rows, COLUMNS, SIZE, SIZE, 4).transpose(0, 2, 1, 3, 4)
        return sheet.reshape(rows * SIZE, COLUMNS * SIZE, 4)

    def save_output(self, path):
        ernie.demo.write_png(path, self.contact_sheet()[..., :3])

    def test_final(self):
        assert self.views is not None and self.views.shape == (WORLDS, SIZE, SIZE, 4)
        views = self.views[..., :3].astype(np.float32)
        assert all(view.std() > 1.0 for view in views), "every view should see its box"
        # The views share a pose, so where one differs from the rest (their median, the ground)
        # it sees its own box; a view of every world would match the median there instead.
        background = np.median(views, axis=0)
        masks = np.abs(views - background).max(axis=-1) > 24.0
        counts = masks.sum(axis=(1, 2))
        assert (counts > 20).all(), f"some view misses its box: {counts}"
        # One box and its shadow each, not the ring: about the same size in every view.
        assert counts.max() < 3 * np.median(counts), f"a view sees other worlds: {counts}"
        # Each sees its own box: redder and less blue with the world, as coloured.
        redness = [views[w][masks[w]][:, 0].mean() - views[w][masks[w]][:, 2].mean()
                   for w in range(WORLDS)]
        assert np.corrcoef(np.arange(WORLDS), redness)[0, 1] > 0.9, redness
        assert self.render_image()[..., :3].std() > 1.0, "the overview should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
