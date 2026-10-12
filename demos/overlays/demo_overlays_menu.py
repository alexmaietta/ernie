# SPDX-License-Identifier: BSD-3-Clause
"""A heads-up menu of buttons over bouncing balls: pause, reset, add a
ball and toggle trails and labels, each button lit while hovered."""

import ernie
import ernie.demo
from ernie import tokens

MAX_BALLS = 8
GRAVITY = 9.81
RADIUS = 0.4
TRAIL = 60
WIDTH, HEIGHT = 640, 360

# The menu's buttons, by id.
PAUSE, RESET, ADD, TRAILS, LABELS = range(1, 6)
BUTTON = (150.0, 26.0)


def button_offset(row):
    """Where button `row` of the menu sits on the image."""
    return (16.0, 16.0 + 34.0 * row)


class Ball:
    def __init__(self, prim, x, height):
        self.prim = prim
        self.x = x
        self.height = height
        self.velocity = 0.0
        self.trail = []

    def position(self):
        return (self.x, 0.0, self.height + RADIUS)


class Demo(ernie.demo.Demo):
    num_frames = 120

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
        camera = world.define(
            "Camera", ernie.Camera, focal_length=24.0
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (2.0, -13.0, 4.0), (2.0, 0.0, 1.5)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(WIDTH, HEIGHT),
            overlays="*",
        )
        self.balls = []
        self.next = 0
        self.paused = False
        self.reset()
        # Clicks on the menu's buttons, in the viewer.
        if viewer is not None:
            viewer.add_overlay_click_handler(
                lambda v, overlay, id: (
                    overlay == "menu" and self.click(id)
                )
            )

    def step(self):
        if not self.paused:
            for ball in self.balls:
                ball.velocity -= GRAVITY * self.frame_dt
                ball.height += ball.velocity * self.frame_dt
                if ball.height < 0.0:
                    ball.height = 0.0
                    ball.velocity = -0.9 * ball.velocity
                ball.trail.append(ball.position())
                if len(ball.trail) > TRAIL:
                    ball.trail.pop(0)
        self.draw()

    def click(self, id):
        scene = self.scene
        if id == PAUSE:
            self.paused = not self.paused
        elif id == RESET:
            self.reset()
        elif id == ADD:
            self.add_ball()
        elif id == TRAILS:
            scene.overlay("trails").visible = not scene.overlay(
                "trails"
            ).visible
        elif id == LABELS:
            scene.overlay("labels").visible = not scene.overlay(
                "labels"
            ).visible
        self.draw()

    def reset(self):
        for ball in self.balls:
            self.scene.remove(ball.prim.path)
        self.balls = []
        labels = self.scene.overlay("labels")
        labels.clear()
        labels.visible = True
        self.scene.overlay("trails").visible = True
        self.paused = False
        for _ in range(3):
            self.add_ball()
        self.draw()

    def add_ball(self):
        if len(self.balls) >= MAX_BALLS:
            return
        i = len(self.balls)
        prim = self.scene.define(
            f"/World/Ball{self.next}",
            ernie.Sphere,
            radius=RADIUS,
            display_color=(0.9, 0.3 + 0.08 * i, 0.2),
        )
        self.next += 1
        # A label that follows the ball.
        self.scene.overlay("labels").label(
            (0.0, 0.0, RADIUS),
            f"ball {i}",
            anchor=str(prim.path),
            depth="always",
        )
        self.balls.append(Ball(prim, -1.0 + 1.2 * i, 1.0 + 0.6 * i))

    def draw(self):
        scene = self.scene
        for ball in self.balls:
            ball.prim.set(translate=ball.position())
        trails = scene.overlay("trails")
        trails.clear()
        for ball in self.balls:
            if len(ball.trail) > 1:
                trails.polyline(
                    ball.trail, color=(0.3, 0.7, 1.0, 0.8)
                )
        # The menu: a button a row, lit while hovered.
        menu = scene.overlay("menu")
        hovered = menu.hovered
        menu.clear()
        menu.hovered = hovered
        on = {True: "on", False: "off"}
        labels = [
            "Resume" if self.paused else "Pause",
            "Reset",
            "Add ball",
            f"Trails: {on[scene.overlay('trails').visible]}",
            f"Labels: {on[scene.overlay('labels').visible]}",
        ]
        for row, label in enumerate(labels):
            menu.button(
                label,
                offset=button_offset(row),
                align=(0, 0),
                box=BUTTON,
                id=row + PAUSE,
                size=13,
                background=(0.1, 0.12, 0.16, 0.85),
                hover=(0.25, 0.45, 0.8, 0.95),
            )
        status = f"{len(self.balls)} balls" + (
            ", paused" if self.paused else ""
        )
        menu.text(status, at=(1, 0), offset=(-16, 16), size=13)

    def test_final(self):
        # Each button is hit where it is drawn, and a click on it does
        # what it says.
        menu = self.scene.overlay("menu")
        extent = (WIDTH, HEIGHT)
        for id in range(PAUSE, LABELS + 1):
            x, y = button_offset(id - PAUSE)
            centre = (x + 0.5 * BUTTON[0], y + 0.5 * BUTTON[1])
            assert menu.screen_item_at(centre, extent) == id, id
        balls = len(self.balls)
        x, y = button_offset(ADD - PAUSE)
        self.click(menu.screen_item_at((x + 1, y + 1), extent))
        assert len(self.balls) == balls + 1, "Add ball adds one"
        height = self.balls[0].height
        self.click(PAUSE)
        self.step()
        assert self.balls[0].height == height, "paused balls stay put"
        self.click(TRAILS)
        assert not self.scene.overlay("trails").visible
        self.click(RESET)
        assert len(self.balls) == 3 and not self.paused, (
            "reset restarts"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
