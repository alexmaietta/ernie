// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/app/viewer.h"
#include "ernie/scene/scene.h"
#include "ernie/session/renderer.h"
#include "ernie/util/tensor.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

/// The options every demo takes (`ParseDemoOptions`), as Newton's examples share theirs.
struct DemoOptions {
    /// Shows the viewer (`--viewer window`); false renders the scene's RenderSettings, or its
    /// first camera, with a `Renderer` and no viewer (`--viewer null`).
    bool viewer = true;
    bool headless = false;  ///< The viewer without a window (`--headless`): no display needed.
    /// Frames to run (`--num-frames`); without it a window runs until closed, and the other
    /// modes run the demo's own count (`Demo::NumFrames`).
    std::optional<std::uint64_t> numFrames;
    /// Reports frames per second after a few frames' warm-up (`--benchmark`), without a viewer.
    bool benchmark = false;
    /// Stops a benchmark after this many seconds (`--benchmark SECONDS`), or at `numFrames`.
    std::optional<double> benchmarkSeconds;
    /// Runs the demo's checks (`--test`): `Demo::TestPostStep` after every step and
    /// `Demo::TestFinal` at the end.
    bool test = false;
    std::filesystem::path output;     ///< Where to write the last frame as PNG (`--output`).
    bool paused = false;              ///< Starts paused, in a window (`--paused`).
    std::string device;               ///< GPU to prefer, by part of its name (`--device`).
    std::string renderer = "raster";  ///< Render backend (`--renderer`).
    bool validation = false;          ///< The Vulkan validation layer, where installed.
    bool vsync = false;               ///< Waits for the display's refresh (`--vsync`).
    bool help = false;                ///< `-h` or `--help` was given.
    bool list = false;                ///< `--list` was given: print the demos and exit.
    std::string demo;                 ///< `--demo NAME`: the demo to start with.
    /// The arguments not among the above, in order, for the demo's own options.
    std::vector<std::string> extra;

    /// Whether the demo shows in a window: a viewer, not headless.
    bool Windowed() const {
        return viewer && !headless;
    }
};

/// Parses the shared options from `args` (the command line without the program name), leaving
/// the others in `DemoOptions::extra`. `--benchmark` implies `--viewer null`.
/// @throws std::invalid_argument for a bad or missing value.
DemoOptions ParseDemoOptions(std::span<const std::string> args);

/// The shared options' help, one per line, for a demo's `--help`.
std::string_view DemoOptionsHelp();

/// A demo: a scene built in the constructor, advanced by `Step`, drawn by `Render` and checked by
/// `TestFinal`, run by `DemoRunner` in a window, headless or without a viewer. Ernie's own
/// demos derive from it, in C++ and in Python (`ernie.demo.Demo`), as may any application's.
///
/// The scene is the viewer's, or one of the demo's own without a viewer. Give it a camera of its
/// own, even for a demo meant for the viewer (which frames the scene with its own camera):
/// without a viewer, and in tests (`RenderImage`), the scene renders through it.
///
/// In the viewer a demo uses it as any application does (`Viewer::AddPanel`, `OnFrame`,
/// `SetActiveCamera`...); when the Demos panel switches away, what it added is removed (in the
/// demo's scope, `Viewer::SetScope`) and the viewport's display settings are restored.
///
/// References:
/// - Newton's examples (`newton.examples`: `Example`, `create_parser`, `init`, `run` and the
///   example browser), https://github.com/newton-physics/newton
class Demo {
  public:
    /// `viewer` is null without one.
    Demo(Viewer* viewer, const DemoOptions& options);
    virtual ~Demo();
    Demo(const Demo&) = delete;
    Demo& operator=(const Demo&) = delete;

    /// Advances the scene by `frameDt` seconds; nothing by default, for a still scene.
    virtual void Step();

    /// Draws a frame: the viewer's next, or without one the scene through `GetRenderer` into
    /// `Image`.
    virtual void Render();

    /// Writes `--output`: the viewport's next frame in a viewer, else the scene's LdrColor
    /// (`RenderImage`) as PNG.
    /// @throws std::runtime_error if the file cannot be written.
    virtual void SaveOutput(const std::filesystem::path& path);

    /// Checks after every step, under `--test`. A demo overrides this, `TestFinal` or both;
    /// the runner fails a test run of a demo that overrides neither.
    virtual void TestPostStep();

    /// Checks at the end of a `--test` run; throw to fail.
    virtual void TestFinal();

    /// Releases what the demo holds beyond the scene and viewer, when it ends or the Demos panel
    /// replaces it; for timing reports and the like.
    virtual void Close();

    /// Draws the demo's own controls with Dear ImGui (`#include <imgui.h>`) in the Demos panel's
    /// Options section each frame, after its `Widgets`, as Newton's examples' `gui(ui)` does;
    /// nothing by default. Runs only in a viewer.
    virtual void Gui();

    /// The demo's controls in the Demos panel's Options section, as widgets (sliders, checkboxes,
    /// combos) rather than ImGui calls, as Python demos add theirs (`widgets`). Add them in the
    /// constructor, with or without a viewer; the runner drops them, with their callbacks, when
    /// the demo ends or is replaced.
    WidgetPanel& Widgets();

    /// `Widgets` shared, for holders that may outlive the demo, such as its Python wrapper.
    std::shared_ptr<WidgetPanel> SharedWidgets();

    /// Frames to run without a window when `--num-frames` is not given: `numFrames`.
    virtual std::uint64_t NumFrames() const;
    /// Seconds a step advances without a window, for runs that repeat: `fixedDt`.
    virtual double FixedDt() const;
    /// The longest step in a window, so a stall does not throw the demo: `maxFrameDt`.
    virtual double MaxFrameDt() const;
    /// One line on what the demo shows, for the Demos panel and the log.
    virtual std::string Description() const;

    /// The scene to build into: the viewer's, or the demo's own without a viewer.
    Scene& GetScene();
    /// The demo's own scene, shared, without a viewer (for holders that may outlive the demo,
    /// such as its Python wrapper); null with a viewer, whose owner holds the scene.
    std::shared_ptr<Scene> OwnScene();
    /// The viewer, or null without one.
    Viewer* GetViewer();
    const DemoOptions& Options() const;

    /// A renderer of the demo's own, opened on first use with the run's device, backend and
    /// validation: for rendering without a viewer, batches and tests.
    Renderer& GetRenderer();

    /// `GetRenderer` shared, for holders that may outlive the demo, such as its Python wrapper.
    std::shared_ptr<Renderer> SharedRenderer();

    /// The last frame rendered without a viewer (`Render`), LdrColor uint8 (rows, cols, 4);
    /// empty in a viewer.
    const Tensor& Image() const;

    /// The scene's LdrColor as its RenderSettings (or its first camera) describe, uint8 (rows,
    /// cols, 4): the last frame without a viewer, else rendered now with `GetRenderer`. Not the
    /// viewport's image: the viewer's camera is not the scene's.
    Tensor RenderImage();

    /// Whether a `--test` run found no checks: neither `TestPostStep` nor `TestFinal` is
    /// overridden. Meaningful once `TestFinal` has run.
    bool Untested() const;

    std::uint64_t frame = 0;  ///< Steps taken so far.
    double time = 0.0;        ///< Demo seconds at the start of this step.
    /// Seconds this step advances: in a window the time since the last frame (at most
    /// `MaxFrameDt`), else `FixedDt`.
    double frameDt = 1.0 / 60.0;

    std::uint64_t numFrames = 100;   ///< `NumFrames`, for demos to set.
    double fixedDt = 1.0 / 60.0;     ///< `FixedDt`, for demos to set.
    double maxFrameDt = 1.0 / 30.0;  ///< `MaxFrameDt`, for demos to set.
    std::string description;         ///< `Description`, for demos to set.

  private:
    Viewer* viewer_;
    DemoOptions options_;
    std::shared_ptr<Scene> ownScene_;
    std::shared_ptr<Renderer> renderer_;
    std::shared_ptr<WidgetPanel> widgets_;
    Tensor image_;
    bool untestedSteps_ = false;
    bool untestedFinal_ = false;
};

/// A demo the runner can build, for the Demos panel and `DemoMain`.
struct DemoInfo {
    std::string category;     ///< Such as "lighting", from demo_<category>_<name>.
    std::string name;         ///< Such as "dome daylight".
    std::string description;  ///< One line, for the panel; the demo's own when empty.
    /// Builds the demo for a viewer (null without one) and the run's options; the options'
    /// `extra` holds no demo options when the Demos panel builds it, so give each a default.
    std::function<std::shared_ptr<Demo>(Viewer*, const DemoOptions&)> create;

    /// "category: name", as the panel lists it.
    std::string Label() const;

    /// "category/name", as `--demo` and `--list` name it, with underscores for spaces.
    std::string Id() const;
};

/// A `DemoInfo` for the demo class `T`, constructed from (Viewer*, const DemoOptions&).
template <typename T>
DemoInfo MakeDemoInfo(std::string category, std::string name, std::string description = {}) {
    return {std::move(category), std::move(name), std::move(description),
            [](Viewer* viewer, const DemoOptions& options) -> std::shared_ptr<Demo> {
                return std::make_shared<T>(viewer, options);
            }};
}

/// Writes `image`, uint8 (rows, cols, 3 or 4) on the CPU, as PNG.
/// @throws std::invalid_argument for another shape or type, or std::runtime_error if the file
/// cannot be written.
void WritePng(const std::filesystem::path& path, const Tensor& image);

/// The demos registered with `ERNIE_REGISTER_DEMO`, sorted by category then name: Ernie's own
/// (the ernie_demos library, in the `ernie` program and the Python module) and any an
/// application links in.
std::vector<DemoInfo> RegisteredDemos();

/// Registers a demo at static initialization, for `RegisteredDemos`.
struct DemoRegistration {
    explicit DemoRegistration(DemoInfo info);
};

/// Registers the demo class `Class` as `category`/`name`, such as "lighting"/"illuminants", with
/// a one-line description, in a source file of its own. Link the object file itself (an object
/// library), not through a static archive, which would drop it as unreferenced.
#define ERNIE_REGISTER_DEMO(Class, category, name, description)                                    \
    static const ::ernie::app::DemoRegistration ernie_demo_registration_##Class(                   \
        ::ernie::app::MakeDemoInfo<Class>(category, name, description))

/// Finds the demo `id` among `demos`: "category/name" (such as "lighting/illuminants"), or the
/// panel's label "category: name"; underscores and spaces match each other. Returns its index.
std::optional<std::size_t> FindDemo(std::span<const DemoInfo> demos, std::string_view id);

/// Runs demos as Newton's examples run: opens the viewer the options ask for (or none), runs a
/// demo until its window closes or it has run its frames, then writes `--output`, runs its
/// final test and reports the benchmark. In a window, a Demos panel switches between the
/// catalogue's demos in the same viewer, and pauses, steps and resets the running one.
class DemoRunner {
  public:
    /// Opens a viewer titled `title` on a scene of the runner's own, unless the options ask for
    /// none (`viewer` false or `benchmark`); demos built from now on register in a scope of
    /// their own (`Viewer::SetScope`).
    /// @throws std::runtime_error if the viewer cannot open.
    explicit DemoRunner(DemoOptions options, const std::string& title = "Ernie demos");
    ~DemoRunner();
    DemoRunner(const DemoRunner&) = delete;
    DemoRunner& operator=(const DemoRunner&) = delete;

    /// The viewer, or null without one.
    Viewer* GetViewer();
    const DemoOptions& Options() const;

    /// The demos the Demos panel lists, and which of them runs first (its selection).
    void SetCatalog(std::vector<DemoInfo> demos, std::size_t current = 0);
    std::span<const DemoInfo> Catalog() const;

    /// Builds the catalogue's demo `index` for this runner's viewer and options.
    std::shared_ptr<Demo> Create(std::size_t index);

    /// Switches to the catalogue's demo `index` before the next frame, as the Demos panel's menu
    /// does: removes what the running demo added to the viewer, empties the scene and builds the
    /// new demo into it. With a viewer only.
    void RequestSwitch(std::size_t index);

    /// Builds the running demo afresh before the next frame, as the Demos panel's Reset does.
    void RequestReset();

    /// Runs `demo`, built for this runner's viewer, as the class describes; returns the demo
    /// running at the end (the Demos panel may have switched it), or null if the last switch
    /// failed. Closes the viewer.
    /// @throws what the demo throws, and std::logic_error for a `--test` run of a demo without
    /// checks.
    std::shared_ptr<Demo> Run(std::shared_ptr<Demo> demo);

  private:
    struct State;
    std::unique_ptr<State> state_;
};

/// A demo program's `main`: parses the command line (`ParseDemoOptions`), then prints the help
/// (`--help`) or the demos (`--list`), or runs `demos[start]` (or the one `--demo NAME` names)
/// with all of `demos` in the Demos panel. Returns the exit status, printing any error.
int DemoMain(int argc, char** argv, std::vector<DemoInfo> demos, std::size_t start = 0);

}  // namespace ernie::app
