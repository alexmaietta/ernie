# SPDX-License-Identifier: BSD-3-Clause
"""A Warp kernel drawing an animated Julia set straight into a viewer
panel's texture, with no copy at all (Viewer.map_image)."""

import math
from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens

ITERATIONS = 96


@wp.kernel
@no_type_check
def julia(image: wp.array3d[wp.uint8], c: wp.vec2, zoom: float):
    """The Julia set of z^2 + c over the image, coloured by how fast
    each point escapes."""
    y, x = wp.tid()
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    zx = (float(x) / cols - 0.5) * 3.0 * cols / (rows * zoom)
    zy = (float(y) / rows - 0.5) * 3.0 / zoom
    escaped = int(ITERATIONS)
    for i in range(ITERATIONS):
        if zx * zx + zy * zy > 4.0:
            escaped = i
            break
        t = zx * zx - zy * zy + c[0]
        zy = 2.0 * zx * zy + c[1]
        zx = t
    s = float(escaped) / float(ITERATIONS)
    image[y, x, 0] = wp.uint8(int(255.0 * wp.sqrt(s)))
    image[y, x, 1] = wp.uint8(int(255.0 * s * s))
    image[y, x, 2] = wp.uint8(
        int(255.0 * (0.5 + 0.5 * wp.sin(6.0 * s)))
    )
    image[y, x, 3] = wp.uint8(255)


class Demo(ernie.demo.Demo):
    num_frames = 60

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument(
            "--size",
            type=int,
            default=512,
            help="the image's side, in pixels",
        )
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("This demo needs CUDA interop")
        self.size = args.size
        self.device = "cuda:0"
        world = self.scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, checker=True)
        world.define("Sun", ernie.DistantLight)
        world.define("Camera", ernie.Camera).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -8.0, 3.0), (0.0, 0.0, 0.0)
            ),
        )
        # Without a viewer, an image of the demo's own to draw into.
        self.own = (
            None
            if viewer is not None
            else wp.zeros(
                (self.size, self.size, 4),
                dtype=wp.uint8,
                device=self.device,
            )
        )
        self.zoom = 1.0
        self.widgets.slider(
            "Zoom",
            0.5,
            4.0,
            self.zoom,
            on_change=lambda value: setattr(self, "zoom", value),
        )

    def step(self):
        angle = 0.3 * self.time
        c = wp.vec2(
            0.7885 * math.cos(angle), 0.7885 * math.sin(angle)
        )
        stream = wp.get_stream(self.device)
        if self.viewer is not None:
            # The texture itself, written in place: no copy.
            mapped = self.viewer.map_image(
                "julia",
                (self.size, self.size, 4),
                np.uint8,
                stream=stream,
            )
            image = wp.from_dlpack(mapped)
            wp.launch(
                julia,
                dim=(self.size, self.size),
                inputs=[image, c, self.zoom],
                device=self.device,
                stream=stream,
            )
            self.viewer.unmap_image("julia", stream=stream)
        else:
            assert self.own is not None
            wp.launch(
                julia,
                dim=(self.size, self.size),
                inputs=[self.own, c, self.zoom],
                device=self.device,
                stream=stream,
            )
            self.log_image("julia", self.own, stream=stream)

    def test_final(self):
        image = self.logged_image("julia")
        if image is None:  # drawn in the viewer
            return
        image = np.asarray(self.renderer.cuda_download(image))
        assert image.shape == (self.size, self.size, 4)
        assert image[..., :3].std() > 10.0, (
            "the Julia set should not be flat"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
