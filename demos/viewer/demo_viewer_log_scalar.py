# SPDX-License-Identifier: BSD-3-Clause
"""A bouncing ball whose height, speed and energy are logged each step
as live plots (log_scalar)."""

import math

import ernie
import ernie.demo
from ernie import tokens

GRAVITY = 9.81
RADIUS = 0.5
DROP = 4.0  # metres, the ball's lowest point
RESTITUTION = 0.8


class Demo(ernie.demo.Demo):
    num_frames = 240

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_size=1.0,
        )
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.4, -0.6, 0.7)),
        )
        world.define("Camera", ernie.Camera, focal_length=24.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -12.0, 4.0), (0.0, 0.0, 2.0)
            ),
        )
        self.ball = world.define(
            "Ball",
            ernie.Sphere,
            radius=RADIUS,
            display_color=(0.9, 0.35, 0.2),
        )
        self.height = DROP
        self.velocity = 0.0
        self.place()
        self.widgets.slider(
            "Restitution",
            0.1,
            1.0,
            RESTITUTION,
            on_change=lambda value: setattr(
                self, "restitution", value
            ),
        )
        self.widgets.button("Drop again", self.drop)
        self.restitution = RESTITUTION

    def drop(self):
        self.height = DROP
        self.velocity = 0.0

    def place(self):
        self.ball.set(translate=(0.0, 0.0, self.height + RADIUS))

    def energy(self):
        """The ball's energy per unit mass as a fraction of the
        drop's."""
        return (GRAVITY * self.height + 0.5 * self.velocity**2) / (
            GRAVITY * DROP
        )

    def step(self):
        # Semi-implicit Euler, the ball bouncing off the ground with
        # the restitution.
        self.velocity -= GRAVITY * self.frame_dt
        self.height += self.velocity * self.frame_dt
        if self.height < 0.0:
            self.height = 0.0
            self.velocity = -self.velocity * self.restitution
        self.place()
        self.log_scalar("ball/height", self.height)
        self.log_scalar("ball/speed", abs(self.velocity))
        self.log_scalar("ball/energy", self.energy())

    def test_final(self):
        heights = self.logged_scalars("ball/height")
        energies = self.logged_scalars("ball/energy")
        assert len(heights) == self.frame, "a height every step"
        assert min(heights) >= 0.0 and max(heights) <= DROP, (
            "the ball stays above the ground"
        )
        assert (
            sum(
                1
                for a, b in zip(heights, heights[1:])
                if a > 0.0 and b == 0.0
            )
            >= 2
        ), "the ball should bounce at least twice"
        assert energies[-1] < 0.75 * energies[0], (
            "each bounce should lose energy"
        )
        assert all(math.isfinite(e) for e in energies)


if __name__ == "__main__":
    ernie.demo.main(Demo)
