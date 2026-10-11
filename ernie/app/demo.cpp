// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/demo.h"

#include "ernie/app/panel.h"

#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ernie::app {

namespace {

/// Frames a benchmark runs before it starts timing.
constexpr std::uint64_t kBenchmarkWarmup = 5;

constexpr std::string_view kHelp =
    R"(  --viewer window|null   'window' shows the viewer (the default); 'null' renders the scene's
                         RenderSettings with a bare Renderer and no viewer
  --headless             render the viewer without a window (no display needed)
  --num-frames N         frames to run; a window runs until closed, else the demo's own count
  --benchmark [SECONDS]  report frames per second, rendering with --viewer null; stop after
                         SECONDS, if given, or --num-frames
  --test                 run the demo's checks (TestPostStep and TestFinal)
  --output PATH          write the last frame here as PNG
  --paused               start paused (the Demos panel resumes)
  --device NAME          GPU to prefer, by part of its name
  --renderer NAME        render backend (default: raster)
  --validation           enable the Vulkan validation layer where installed
  --vsync                wait for the display's refresh
  --demo NAME            the demo to start with
  --list                 print the demos and exit
  -h, --help             show this help and exit
)";

std::string FormatFps(double fps) {
    char text[32];
    std::snprintf(text, sizeof(text), fps >= 10.0 ? "%.1f" : fps >= 1.0 ? "%.2f" : "%.3f", fps);
    return text;
}

double ParseNumber(const std::string& option, const std::string& text) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size() || text.empty()) {
        throw std::invalid_argument(option + " needs a number, not '" + text + "'");
    }
    return value;
}

bool IsNumber(const std::string& text) {
    try {
        std::size_t used = 0;
        std::stod(text, &used);
        return used == text.size();
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

DemoOptions ParseDemoOptions(std::span<const std::string> args) {
    DemoOptions options;
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string arg = args[i];
        std::optional<std::string> inlineValue;
        if (const std::size_t equals = arg.find('='); arg.starts_with("--") && equals != arg.npos) {
            inlineValue = arg.substr(equals + 1);
            arg.resize(equals);
        }
        const auto value = [&]() -> std::string {
            if (inlineValue) {
                return *inlineValue;
            }
            if (i + 1 >= args.size()) {
                throw std::invalid_argument(arg + " needs a value");
            }
            return args[++i];
        };
        // Flags with --no- forms, as argparse's BooleanOptionalAction.
        const auto flag = [&](std::string_view name, bool& field) {
            if (arg == "--" + std::string(name)) {
                field = true;
                return true;
            }
            if (arg == "--no-" + std::string(name)) {
                field = false;
                return true;
            }
            return false;
        };
        if (arg == "-h" || arg == "--help") {
            options.help = true;
        } else if (arg == "--list") {
            options.list = true;
        } else if (arg == "--viewer") {
            const std::string mode = value();
            if (mode != "window" && mode != "null") {
                throw std::invalid_argument("--viewer is 'window' or 'null', not '" + mode + "'");
            }
            options.viewer = mode == "window";
        } else if (arg == "--num-frames") {
            const double frames = ParseNumber(arg, value());
            if (frames < 0.0 || frames != static_cast<double>(static_cast<std::uint64_t>(frames))) {
                throw std::invalid_argument("--num-frames needs a whole number of frames");
            }
            options.numFrames = static_cast<std::uint64_t>(frames);
        } else if (arg == "--benchmark") {
            options.benchmark = true;
            if (inlineValue) {
                options.benchmarkSeconds = ParseNumber(arg, *inlineValue);
            } else if (i + 1 < args.size() && IsNumber(args[i + 1])) {
                options.benchmarkSeconds = ParseNumber(arg, args[++i]);
            }
        } else if (arg == "--output") {
            options.output = value();
        } else if (arg == "--device") {
            options.device = value();
        } else if (arg == "--renderer") {
            options.renderer = value();
        } else if (arg == "--demo") {
            options.demo = value();
        } else if (!flag("headless", options.headless) && !flag("test", options.test) &&
                   !flag("paused", options.paused) && !flag("validation", options.validation) &&
                   !flag("vsync", options.vsync)) {
            options.extra.push_back(args[i]);
        }
    }
    if (options.benchmark) {
        options.viewer = false;
    }
    return options;
}

std::string_view DemoOptionsHelp() {
    return kHelp;
}

Demo::Demo(Viewer* viewer, const DemoOptions& options) : viewer_(viewer), options_(options) {
    if (viewer_ == nullptr) {
        ownScene_ = std::make_shared<Scene>();
    }
}

Demo::~Demo() = default;

void Demo::Step() {}

void Demo::Render() {
    if (viewer_ != nullptr) {
        viewer_->Step();
    } else {
        image_ = GetRenderer().Render(GetScene());
    }
}

void Demo::SaveOutput(const std::filesystem::path& path) {
    if (viewer_ != nullptr) {
        viewer_->Capture(path);
        viewer_->Step();
        return;
    }
    const Tensor image = RenderImage();
    const std::vector<std::int64_t>& shape = image.Shape();
    if (shape.size() != 3 || shape[2] != 4) {
        throw std::runtime_error("The scene's LdrColor is not an (rows, cols, 4) image");
    }
    const int rows = static_cast<int>(shape[0]);
    const int cols = static_cast<int>(shape[1]);
    if (stbi_write_png(path.string().c_str(), cols, rows, 4, image.Data(), cols * 4) == 0) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

void Demo::TestPostStep() {
    untestedSteps_ = true;
}

void Demo::TestFinal() {
    untestedFinal_ = true;
}

void Demo::Close() {}

std::uint64_t Demo::NumFrames() const {
    return numFrames;
}

double Demo::FixedDt() const {
    return fixedDt;
}

double Demo::MaxFrameDt() const {
    return maxFrameDt;
}

std::string Demo::Description() const {
    return description;
}

Scene& Demo::GetScene() {
    return viewer_ != nullptr ? viewer_->GetScene() : *ownScene_;
}

std::shared_ptr<Scene> Demo::OwnScene() {
    return ownScene_;
}

Viewer* Demo::GetViewer() {
    return viewer_;
}

const DemoOptions& Demo::Options() const {
    return options_;
}

Renderer& Demo::GetRenderer() {
    return *SharedRenderer();
}

std::shared_ptr<Renderer> Demo::SharedRenderer() {
    if (!renderer_) {
        renderer_ = std::make_shared<Renderer>(
            RendererOptions{options_.renderer, options_.device, options_.validation});
    }
    return renderer_;
}

const Tensor& Demo::Image() const {
    return image_;
}

Tensor Demo::RenderImage() {
    if (viewer_ == nullptr && !image_.Empty()) {
        return image_;
    }
    return GetRenderer().Render(GetScene());
}

bool Demo::Untested() const {
    return untestedFinal_ && (untestedSteps_ || frame == 0);
}

std::string DemoInfo::Label() const {
    return category + ": " + name;
}

struct DemoRunner::State {
    DemoOptions options;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<Viewer> viewer;
    std::vector<DemoInfo> catalog;
    std::size_t current = 0;
    /// The scope the running demo registers in.
    std::uint64_t scope = 0;

    // What the viewport showed before any demo, restored when the Demos panel switches.
    ViewportDisplay display;
    Viewer::Exposure exposure = Viewer::Exposure::Default;
    double compensation = 0.0;
    std::optional<Wireframe> wireframe;
    std::string viewVar;
    Viewer::Gizmo gizmo = Viewer::Gizmo::None;
    bool gizmoLocal = false;

    // The Demos panel and what it asked for during the last frame.
    WidgetPanel* panel = nullptr;
    WidgetPanel::Id status = 0;
    WidgetPanel::Id about = 0;
    std::optional<std::size_t> switchTo;
    std::optional<bool> pausedRequest;
    bool resetRequested = false;
    bool singleStep = false;

    void RestoreDisplay() {
        viewer->SetViewportDisplay(display);
        viewer->SetExposure(exposure);
        viewer->SetExposureCompensation(compensation);
        viewer->SetWireframe(wireframe);
        viewer->SetViewVar(viewVar);
        viewer->SetGizmo(gizmo);
        viewer->SetGizmoLocal(gizmoLocal);
    }

    void Announce(Demo& demo) {
        const std::string description = DescriptionOf(demo);
        viewer->Log(LogLevel::Info, "Demo: " + (description.empty() ? "untitled" : description));
        if (panel != nullptr) {
            panel->Set(about, description);
        }
    }

    std::string DescriptionOf(Demo& demo) const {
        std::string description = demo.Description();
        if (description.empty() && current < catalog.size()) {
            description = catalog[current].description;
        }
        return description;
    }

    void AddPanel(Demo& demo) {
        const std::uint64_t previous = viewer->Scope();
        viewer->SetScope(0);
        panel = &viewer->AddWidgetPanel("Demos", Dock::Left);
        if (catalog.empty()) {
            panel->Text("No other demos");
        } else {
            std::vector<std::string> labels;
            for (const DemoInfo& info : catalog) {
                labels.push_back(info.Label());
            }
            panel->Combo("Demo", std::move(labels), current,
                         [this](std::size_t index) { switchTo = index; });
        }
        panel->Separator();
        panel->Checkbox("Paused", options.paused, [this](bool on) { pausedRequest = on; });
        panel->Button("Step", [this] { singleStep = true; });
        panel->Button("Reset", [this] { resetRequested = true; });
        status = panel->Text("");
        panel->Separator();
        about = panel->Text(DescriptionOf(demo));
        viewer->SetScope(previous);
    }

    /// Replaces `old` with a new build of the catalogue's demo `index`; null, with the error in
    /// the log, if it cannot be built.
    std::shared_ptr<Demo> Load(std::shared_ptr<Demo> old, std::size_t index) {
        if (old) {
            old->Close();
        }
        viewer->RemoveScope(scope);
        RestoreDisplay();
        viewer->ClearScene();
        old.reset();
        scope = viewer->NewScope();
        viewer->SetScope(scope);
        DemoOptions demoOptions = options;
        demoOptions.extra.clear();
        try {
            std::shared_ptr<Demo> demo = catalog.at(index).create(viewer.get(), demoOptions);
            viewer->ResetSession();
            current = index;
            Announce(*demo);
            return demo;
        } catch (const std::exception& error) {
            viewer->Log(LogLevel::Error,
                        "Could not load " + catalog.at(index).Label() + ": " + error.what());
            std::fprintf(stderr, "Could not load %s: %s\n", catalog.at(index).Label().c_str(),
                         error.what());
            viewer->RemoveScope(scope);
            viewer->ClearScene();
            viewer->ResetSession();
            if (panel != nullptr) {
                panel->Set(about, std::string());
            }
            return nullptr;
        }
    }

    /// Removes the running demo's registrations and the panel, which hold the demo (and in
    /// Python, through it the viewer).
    void Release() {
        if (!viewer) {
            return;
        }
        viewer->RemoveScope(scope);
        if (panel != nullptr) {
            viewer->RemovePanel("Demos");
            panel = nullptr;
        }
        viewer->SetScope(0);
    }
};

DemoRunner::DemoRunner(DemoOptions options, const std::string& title)
    : state_(std::make_unique<State>()) {
    State& s = *state_;
    if (options.benchmark) {
        options.viewer = false;
    }
    s.options = std::move(options);
    if (!s.options.viewer) {
        return;
    }
    s.scene = std::make_unique<Scene>();
    ViewerOptions viewerOptions;
    viewerOptions.title = title;
    viewerOptions.hidden = s.options.headless;
    viewerOptions.validation = s.options.validation;
    viewerOptions.device = s.options.device;
    viewerOptions.renderer = s.options.renderer;
    viewerOptions.vsync = s.options.vsync;
    s.viewer = std::make_unique<Viewer>(*s.scene, viewerOptions);
    s.display = s.viewer->GetViewportDisplay();
    s.exposure = s.viewer->GetExposure();
    s.compensation = s.viewer->ExposureCompensation();
    s.wireframe = s.viewer->GetWireframe();
    s.viewVar = s.viewer->ViewVar();
    s.gizmo = s.viewer->GetGizmo();
    s.gizmoLocal = s.viewer->GizmoLocal();
    s.scope = s.viewer->NewScope();
    s.viewer->SetScope(s.scope);
}

DemoRunner::~DemoRunner() {
    state_->Release();
}

Viewer* DemoRunner::GetViewer() {
    return state_->viewer.get();
}

const DemoOptions& DemoRunner::Options() const {
    return state_->options;
}

void DemoRunner::SetCatalog(std::vector<DemoInfo> demos, std::size_t current) {
    if (!demos.empty() && current >= demos.size()) {
        throw std::out_of_range("The current demo is not in the catalogue");
    }
    state_->catalog = std::move(demos);
    state_->current = current;
}

std::span<const DemoInfo> DemoRunner::Catalog() const {
    return state_->catalog;
}

std::shared_ptr<Demo> DemoRunner::Create(std::size_t index) {
    State& s = *state_;
    s.current = index;
    return s.catalog.at(index).create(s.viewer.get(), s.options);
}

void DemoRunner::RequestSwitch(std::size_t index) {
    State& s = *state_;
    if (!s.viewer) {
        throw std::logic_error("Only a runner with a viewer switches demos");
    }
    if (index >= s.catalog.size()) {
        throw std::out_of_range("No demo " + std::to_string(index) + " in the catalogue");
    }
    s.switchTo = index;
}

void DemoRunner::RequestReset() {
    State& s = *state_;
    if (!s.viewer || s.catalog.empty()) {
        throw std::logic_error("Only a runner with a viewer and a catalogue resets demos");
    }
    s.resetRequested = true;
}

std::shared_ptr<Demo> DemoRunner::Run(std::shared_ptr<Demo> demo) {
    using Clock = std::chrono::steady_clock;
    State& s = *state_;
    const DemoOptions& o = s.options;
    Viewer* viewer = s.viewer.get();
    if (!demo) {
        throw std::invalid_argument("Run needs a demo");
    }
    try {
        if (viewer != nullptr) {
            viewer->ResetSession();
            if (o.Windowed() && !o.test) {
                s.AddPanel(*demo);
            }
            s.Announce(*demo);
        }
        bool paused = o.paused && o.Windowed();
        std::optional<Clock::time_point> timerStart;
        std::uint64_t timedFrames = 0;
        Clock::time_point last = Clock::now();
        while (viewer == nullptr || viewer->IsOpen()) {
            if (s.pausedRequest) {
                paused = *s.pausedRequest;
                s.pausedRequest.reset();
            }
            if (s.switchTo || s.resetRequested) {
                const std::size_t index = s.switchTo.value_or(s.current);
                s.switchTo.reset();
                s.resetRequested = false;
                demo = s.Load(std::move(demo), index);
            }
            if (!demo) {
                viewer->Step();
                continue;
            }
            if (s.panel != nullptr) {
                char status[96];
                std::snprintf(status, sizeof(status), "frame %llu, %.2f s%s",
                              static_cast<unsigned long long>(demo->frame), demo->time,
                              paused ? "  (paused)" : "");
                s.panel->Set(s.status, std::string(status));
            }
            const std::optional<std::uint64_t> limit =
                o.numFrames ? o.numFrames
                            : (o.Windowed() ? std::nullopt
                                            : std::optional<std::uint64_t>(demo->NumFrames()));
            if (limit && demo->frame >= *limit) {
                break;
            }
            const Clock::time_point now = Clock::now();
            demo->frameDt = o.Windowed()
                                ? std::min(std::chrono::duration<double>(now - last).count(),
                                           demo->MaxFrameDt())
                                : demo->FixedDt();
            last = now;
            bool stepping = !paused;
            if (paused && s.singleStep) {
                stepping = true;
                s.singleStep = false;
            }
            if (stepping) {
                demo->Step();
                if (o.test) {
                    demo->TestPostStep();
                }
            }
            demo->Render();
            if (stepping) {
                ++demo->frame;
                demo->time += demo->frameDt;
            }
            if (o.benchmark) {
                if (demo->frame == kBenchmarkWarmup) {
                    timerStart = Clock::now();
                } else if (timerStart) {
                    ++timedFrames;
                    if (o.benchmarkSeconds &&
                        std::chrono::duration<double>(Clock::now() - *timerStart).count() >=
                            *o.benchmarkSeconds) {
                        break;
                    }
                }
            }
        }

        if (!o.output.empty() && demo) {
            demo->SaveOutput(o.output);
            std::printf("wrote %s\n", o.output.string().c_str());
        }
        if (o.test && demo) {
            demo->TestFinal();
            if (demo->Untested()) {
                throw std::logic_error(
                    "The demo has no checks: override TestFinal or TestPostStep (test_final or "
                    "test_post_step in Python)");
            }
        }
        if (o.benchmark) {
            if (!timerStart) {
                std::printf("Benchmark: too few frames to time (warm-up is %llu)\n",
                            static_cast<unsigned long long>(kBenchmarkWarmup));
            } else {
                const double elapsed =
                    std::chrono::duration<double>(Clock::now() - *timerStart).count();
                std::printf(
                    "Benchmark: %s FPS (%llu frames in %.2f s)\n",
                    FormatFps(static_cast<double>(timedFrames) / std::max(elapsed, 1e-9)).c_str(),
                    static_cast<unsigned long long>(timedFrames), elapsed);
            }
        }
        std::fflush(stdout);
        if (demo) {
            demo->Close();
        }
    } catch (...) {
        s.Release();
        throw;
    }
    if (viewer != nullptr) {
        viewer->Close();
    }
    s.Release();
    return demo;
}

int DemoMain(int argc, char** argv, std::vector<DemoInfo> demos, std::size_t start) {
    const std::string program =
        argc > 0 ? std::filesystem::path(argv[0]).filename().string() : std::string("demo");
    DemoOptions options;
    try {
        options = ParseDemoOptions(std::vector<std::string>(argv + std::min(argc, 1), argv + argc));
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "%s: %s\n\nUsage: %s [options]\n\n%s", program.c_str(), error.what(),
                     program.c_str(), std::string(kHelp).c_str());
        return 2;
    }
    if (!options.demo.empty()) {
        const auto named = [&](const DemoInfo& info) {
            std::string spaced = options.demo;
            std::replace(spaced.begin(), spaced.end(), '_', ' ');
            return info.name == spaced || info.Label() == options.demo ||
                   info.category + " " + info.name == spaced;
        };
        const auto found = std::find_if(demos.begin(), demos.end(), named);
        if (found == demos.end()) {
            std::fprintf(stderr, "%s: no demo named '%s' (--list lists them)\n", program.c_str(),
                         options.demo.c_str());
            return 2;
        }
        start = static_cast<std::size_t>(found - demos.begin());
    }
    if (options.help) {
        std::printf("Usage: %s [options]\n\n", program.c_str());
        if (start < demos.size() && !demos[start].description.empty()) {
            std::printf("%s\n\n", demos[start].description.c_str());
        }
        std::printf("Options:\n%s", std::string(kHelp).c_str());
        return 0;
    }
    if (options.list) {
        for (const DemoInfo& info : demos) {
            std::printf("%s\n", info.Label().c_str());
        }
        return 0;
    }
    if (start >= demos.size()) {
        std::fprintf(stderr, "%s: no demo to run\n", program.c_str());
        return 2;
    }
    try {
        DemoRunner runner(options);
        runner.SetCatalog(std::move(demos), start);
        runner.Run(runner.Create(start));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: %s\n", program.c_str(), error.what());
        return 1;
    }
    return 0;
}

}  // namespace ernie::app
