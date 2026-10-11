# Demos

The demos in `demos/python` show Ernie's features one at a time: lighting in physical units, real orbital cameras, terrain and planets, GPU interop, batched rendering for reinforcement learning. Each is one file, `demo_<category>_<description>.py`, and runs the same way, after Newton's examples.

## Running a demo

```sh
PYTHONPATH=build/python python demos/python/demo_lighting_illuminants.py
```

This opens the viewer on the demo's scene, with a Demos panel on the left. Its menu lists every demo beside this one, by category; choosing one clears the scene and builds that demo in the same window. The panel also pauses, single-steps and resets the demo.

Every demo takes these options:

| Option | Effect |
| --- | --- |
| `--viewer window` | Opens the viewer (the default). |
| `--viewer null` | No viewer: each frame renders the scene's RenderSettings (or its first camera) with an `ernie.Renderer`. |
| `--headless` | Renders the viewer without a window, overlays and all; no display needed. |
| `--num-frames N` | Stops after N frames. Without it a window runs until closed, and the other modes run the demo's own count. |
| `--benchmark [SECONDS]` | Implies `--viewer null` and prints frames per second after a few frames' warm-up, stopping after SECONDS or `--num-frames`. |
| `--test` | Runs the demo's checks: `test_post_step` after every step and `test_final` at the end. |
| `--output PATH` | Writes the last frame, or the demo's own composite, as PNG. |
| `--paused` | Starts paused; the Demos panel resumes. |
| `--device NAME` | Prefers the GPU whose name contains NAME. |
| `--renderer NAME` | The render backend. |
| `--validation` | Enables the Vulkan validation layer where installed. |
| `--vsync` | Waits for the display's refresh. |

A demo's own options follow these; `--help` lists them all. For example:

```sh
python demos/python/demo_lighting_illuminants.py --viewer null --output illuminants.png
python demos/python/demo_overlays_debug_draw.py --headless --output overlays.png
python demos/python/demo_viewer_sun_panel.py --benchmark --num-frames 300
python demos/python/demo_overlays_debug_draw.py --viewer null --test
```

The Python tests run every demo with `--viewer null --test` (`ernie/test/python/test_demos.py`).

## Writing a demo

A demo subclasses `ernie.demo.Demo`. It builds its scene into `self.scene` in `__init__`, moves it in `step` by `self.frame_dt` seconds, and checks itself in `test_final`:

```python
import ernie
import ernie.demo
from ernie import tokens


class Demo(ernie.demo.Demo):
    num_frames = 120  # frames to run without a window

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, checker=True)
        self.ball = world.define("Ball", ernie.Sphere, radius=0.5, translate=(0, 0, 0.5))
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7)))
        world.define("Camera", ernie.Camera).set(
            tokens.TRANSFORM, ernie.look_at_transform((0, -8, 3), (0, 0, 0.5)))
        if viewer is not None:
            panel = viewer.add_panel("Ball")
            panel.slider("Radius", 0.1, 2.0, 0.5,
                         on_change=lambda value: self.ball.set(radius=value))

    def step(self):
        self.ball.set(translate=(2.0 * self.time, 0, 0.5))

    def test_final(self):
        assert self.render_image()[..., :3].std() > 1.0


if __name__ == "__main__":
    ernie.demo.main(Demo)
```

What to know:

- `viewer` is `None` under `--viewer null`, so put panels and other viewer calls under `if viewer is not None`. Give the scene a camera of its own: without a viewer, and in tests, the scene renders through it.
- In a window `frame_dt` is the real time since the last frame (at most `max_frame_dt`). Without one it is `fixed_dt`, so runs repeat exactly. `self.time` is the demo's clock at the start of the step.
- A demo uses the viewer as any application does: `add_panel`, `on_frame`, `add_hud_item`, `active_camera`, `viewport_display`. When the Demos panel switches away, what the demo added is removed and what it set is restored.
- Add your own options by overriding the class method `create_parser`, adding to what `super().create_parser()` returns. Give every option a default: the Demos panel opens demos with their defaults.
- Override `save_output(path)` for a composite `--output` (images side by side, a contact sheet), and `render` when the per-frame render should differ, as for batch renders through `self.renderer`. `self.map(path, attribute, stream)` maps an attribute through the viewer or the renderer, whichever draws the demo, for in-place GPU writes. `close()` runs when the demo ends or is switched away.
- Name the file `demo_<category>_<description>.py` and put it with the others, and the Demos panel lists it under its category.

`ernie.demo.init` and `ernie.demo.run` are `main`'s two halves, for a script that builds its own parser or runs a demo it constructs itself.
