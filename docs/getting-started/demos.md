# Demos

Ernie's demos show its features one at a time: lighting in physical units, real orbital cameras, terrain and planets, GPU interop, batched rendering for reinforcement learning. They live in `demos/<category>/`, each a pair, `<name>.cpp` and `<name>.py`: the same scene and checks in each language, rendering the same image. Only `demos/interop/`, which shows Python's GPU interop (Warp, DLPack, CUDA streams), is Python alone. A demo is named `<category>/<name>`, such as `lighting/illuminants`. They run the same way, after Newton's examples.

## Running a demo

The `ernie` program, which `pip install ernie-renderer` puts on the path, runs the C++ demos:

```sh
ernie                                  # the viewer on the default scene, with the Demos panel
ernie --demo lighting/illuminants      # the viewer with a demo running
ernie --list                           # every demo's name
ernie scene.json                       # the viewer on a scene file (ernie.save_scene's JSON)
ernie scene.json --output scene.png    # renders a scene file without a window
```

`python -m ernie.demos` runs the Python demos, which the package ships as `ernie.demos`:

```sh
python -m ernie.demos                         # viewer/sun_panel
python -m ernie.demos --list
python -m ernie.demos lighting/illuminants    # one by name, with its options after it
```

From a checkout, run the files themselves (`PYTHONPATH=build/python python demos/lighting/illuminants.py`), or a C++ demo's own program, which `-DERNIE_BUILD_DEMOS=ON` builds as `demo_<category>_<name>` (`build/bin/demo_lighting_illuminants`).

Each opens the viewer with a Demos panel on the left. Its menu lists the demos in the same language, by category; choosing one clears the scene and builds that demo in the same window. The panel also pauses, single-steps and resets the demo.

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
ernie --demo lighting/illuminants --viewer null --output illuminants.png
python -m ernie.demos lighting/illuminants --viewer null --output illuminants.png
python -m ernie.demos viewer/sun_panel --headless --test
ernie --demo viewer/sun_panel --benchmark --num-frames 300
```

The Python tests (`ernie/test/python/test_demos.py`) run every demo in both languages with `--viewer null --test`, check that each has its twin, and check that the twins' `--output` images match.

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
- A demo uses the viewer as any application does: `add_panel`, `on_frame`, `add_hud_item`, `active_camera`, `viewport_display`. A demo registers in a scope of its own (`Viewer.scope`), so when the Demos panel switches away, what it added is removed, and the viewport's display settings are restored. A camera, lighting or RenderSettings it chooses is kept over the viewer's reset of its session.
- Add your own options by overriding the class method `create_parser`, adding to what `super().create_parser()` returns. Give every option a default: the Demos panel opens demos with their defaults.
- Override `save_output(path)` for a composite `--output` (images side by side, a contact sheet), and `render` when the per-frame render should differ, as for batch renders through `self.renderer`. `self.map(path, attribute, stream)` maps an attribute through the viewer or the renderer, whichever draws the demo, for in-place GPU writes. `close()` runs when the demo ends or is switched away.
- Put it in `demos/<category>/<name>.py` beside its C++ twin, `<name>.cpp`, and the Demos panel lists it under its category. The image helpers both languages share are `ernie.demo`'s (`side_by_side`, `srgb_to_linear`, `mean_linear_rgb`, `heatmap`, `write_png`) and `demos/util.h`'s.

`ernie.demo.init` and `ernie.demo.run` are `main`'s two halves, for a script that builds its own parser or runs a demo it constructs itself.

## Demos in your own project

The base class and the loop are C++ (`ernie::app::Demo` and `ernie::app::DemoRunner`, in `ernie/app/demo.h`), and `ernie.demo.Demo` derives from them through Python bindings. So a project of your own, in Python or in C++, writes demos the same way, outside Ernie's repository, and runs them with the same options, the same `--test` checks and the same Demos panel, which lists the demo files beside the one that runs.

In C++, derive from `ernie::app::Demo`, override what you need, and hand the demos to `DemoMain`, which parses the shared options (leaving the others in `DemoOptions::extra`), opens the viewer and runs the first; its Demos panel lists them all:

```cpp
#include <ernie/app/demo.h>
#include <ernie/scene/geometry_schemas.h>

class BallDemo : public ernie::app::Demo {
  public:
    BallDemo(ernie::app::Viewer* viewer, const ernie::app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        ernie::Sphere::Define(GetScene(), "/World/Ball").SetRadius(0.5);
        // ... a sun and a camera, and a panel when GetViewer() is not null
    }

    void Step() override {
        ernie::SetXform(GetScene(), "/World/Ball", {.translate = glm::dvec3(2.0 * time, 0.0, 0.5)});
    }

    void TestFinal() override {
        if (RenderImage().Empty()) {
            throw std::runtime_error("nothing rendered");
        }
    }
};

int main(int argc, char** argv) {
    return ernie::app::DemoMain(argc, argv,
                                {ernie::app::MakeDemoInfo<BallDemo>("example", "ball")});
}
```

The C++ names follow Python's: `Step`, `Render`, `SaveOutput`, `TestPostStep`, `TestFinal` and `Close` to override; `GetScene`, `GetViewer` (null without one), `GetRenderer`, `RenderImage`, `frame`, `time` and `frameDt`; and `numFrames`, `fixedDt`, `maxFrameDt` and `description` to set. `WritePng` writes an image.

Ernie's own C++ demos register themselves instead, with `ERNIE_REGISTER_DEMO(Class, "category", "name", "description")` at the end of their file, and `RegisteredDemos()` lists them; an application linking its demos' object files can do the same and pass `RegisteredDemos()` to `DemoMain`.
