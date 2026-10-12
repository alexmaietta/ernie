// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/demo.h"

#include "ernie/app/panel.h"
#include "ernie/util/profile_report.h"
#include "ernie/util/profiler.h"

#include <imgui.h>
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ernie::app {

namespace {

/// Frames a benchmark runs before it starts timing.
constexpr std::uint64_t kBenchmarkWarmup = 5;

/// Frames a profile keeps (the last of a longer run), and those it
/// leaves out of its averages at the start, which compile shaders and
/// upload the scene.
constexpr std::size_t kProfileHistory = 10000;
constexpr std::size_t kProfileWarmup = 5;

/// Prints the profiler's zones over the frames recorded, after the
/// warm-up, and writes them where `options` asks.
void ReportProfile(const DemoOptions& options) {
    profiler::SetEnabled(false);
    std::vector<profiler::Frame> frames = profiler::Frames();
    if (frames.size() > kProfileWarmup) {
        frames.erase(frames.begin(), frames.begin() + kProfileWarmup);
    }
    // Nor the newest, which the device may still be running.
    frames.resize(frames.size() -
                  std::min(frames.size(), profiler::kFramesInFlight));
    const std::vector<std::string> threads = profiler::ThreadNames();
    const profiler::Report report =
        profiler::Summarize(frames, threads);
    std::printf("Profile: %zu frames (%zu with GPU zones), means per "
                "frame\n\n%s",
                report.cpuFrames, report.gpuFrames,
                profiler::ReportText(report, {}).c_str());
    const auto write = [](const std::filesystem::path& path,
                          const std::string& text) {
        std::ofstream file(path, std::ios::binary);
        file << text;
        if (!file) {
            throw std::runtime_error("Could not write " +
                                     path.string());
        }
        std::printf("wrote %s\n", path.string().c_str());
    };
    if (!options.profileJson.empty()) {
        write(options.profileJson, profiler::ReportJson(report));
    }
    if (!options.profileTrace.empty()) {
        write(options.profileTrace,
              profiler::TraceJson(frames, threads));
    }
}

constexpr std::string_view kHelp =
    R"(  --viewer window|null   'window' shows the viewer (the default); 'null' renders the scene's
                         RenderSettings with a bare Renderer and no viewer
  --headless             render the viewer without a window (no display needed)
  --num-frames N         frames to run; a window runs until closed, else the demo's own count
  --benchmark [SECONDS]  report frames per second, rendering with --viewer null; stop after
                         SECONDS, if given, or --num-frames
  --profile              print the profiler's GPU and CPU zones, nested and averaged, at the end
  --profile-json PATH    also write them as JSON (implies --profile)
  --profile-trace PATH   also write a Chrome trace for Perfetto or chrome://tracing
                         (implies --profile)
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
    std::snprintf(text, sizeof(text),
                  fps >= 10.0  ? "%.1f"
                  : fps >= 1.0 ? "%.2f"
                               : "%.3f",
                  fps);
    return text;
}

double ParseNumber(const std::string& option,
                   const std::string& text) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size() || text.empty()) {
        throw std::invalid_argument(
            option + " needs a number, not '" + text + "'");
    }
    return value;
}

bool IsNumber(const std::string& text) {
    try {
        std::size_t used = 0;
        static_cast<void>(
            std::stod(text, &used));  // only whether it parses whole
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
        if (const std::size_t equals = arg.find('=');
            arg.starts_with("--") && equals != arg.npos) {
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
        // Flags with --no- forms, as argparse's
        // BooleanOptionalAction.
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
                throw std::invalid_argument(
                    "--viewer is 'window' or 'null', not '" + mode +
                    "'");
            }
            options.viewer = mode == "window";
        } else if (arg == "--num-frames") {
            const double frames = ParseNumber(arg, value());
            if (frames < 0.0 ||
                frames != static_cast<double>(
                              static_cast<std::uint64_t>(frames))) {
                throw std::invalid_argument(
                    "--num-frames needs a whole number of frames");
            }
            options.numFrames = static_cast<std::uint64_t>(frames);
        } else if (arg == "--benchmark") {
            options.benchmark = true;
            if (inlineValue) {
                options.benchmarkSeconds =
                    ParseNumber(arg, *inlineValue);
            } else if (i + 1 < args.size() && IsNumber(args[i + 1])) {
                options.benchmarkSeconds =
                    ParseNumber(arg, args[++i]);
            }
        } else if (arg == "--profile-json") {
            options.profileJson = value();
            options.profile = true;
        } else if (arg == "--profile-trace") {
            options.profileTrace = value();
            options.profile = true;
        } else if (arg == "--output") {
            options.output = value();
        } else if (arg == "--device") {
            options.device = value();
        } else if (arg == "--renderer") {
            options.renderer = value();
        } else if (arg == "--demo") {
            options.demo = value();
        } else if (!flag("headless", options.headless) &&
                   !flag("test", options.test) &&
                   !flag("profile", options.profile) &&
                   !flag("paused", options.paused) &&
                   !flag("validation", options.validation) &&
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

Demo::Demo(Viewer* viewer, const DemoOptions& options)
    : viewer_(viewer), options_(options) {
    if (viewer_ == nullptr) {
        ownScene_ = std::make_shared<Scene>();
    }
}

Demo::~Demo() = default;

void Demo::Step() {}

void Demo::Render() {
    if (viewer_ != nullptr) {
        viewer_->Step();
        return;
    }
    if (!options_.benchmark || options_.test) {
        image_ = GetRenderer().Render(GetScene());
        return;
    }
    // A benchmark measures rendering alone: frames in flight
    // together, as a viewer keeps them, and none read back to the
    // host.
    constexpr std::size_t kInFlight = 2;
    running_.push_back(GetRenderer().StepAsync(GetScene()));
    for (const auto& [path, output] : running_.back().Outputs()) {
        for (const auto& [name, var] : output.vars) {
            lastExtent_ = {var.Width(), var.Height()};
        }
    }
    while (running_.size() > kInFlight) {
        running_.front().Wait();
        running_.pop_front();
    }
}

std::pair<std::uint32_t, std::uint32_t> Demo::FinishFrames() {
    for (StepOperation& step : running_) {
        step.Wait();
    }
    running_.clear();
    return lastExtent_;
}

void Demo::SaveOutput(const std::filesystem::path& path) {
    if (viewer_ != nullptr) {
        viewer_->Capture(path);
        viewer_->Step();
        return;
    }
    WritePng(path, RenderImage());
}

void WritePng(const std::filesystem::path& path,
              const Tensor& image) {
    const std::vector<std::int64_t>& shape = image.Shape();
    if (shape.size() != 3 || (shape[2] != 3 && shape[2] != 4) ||
        image.Type() != DType::UInt8 ||
        image.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "WritePng takes a uint8 (rows, cols, 3 or 4) CPU image");
    }
    const int rows = static_cast<int>(shape[0]);
    const int cols = static_cast<int>(shape[1]);
    const int channels = static_cast<int>(shape[2]);
    if (stbi_write_png(path.string().c_str(), cols, rows, channels,
                       image.Data(), cols * channels) == 0) {
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

void Demo::Gui() {}

WidgetPanel& Demo::Widgets() {
    return *SharedWidgets();
}

std::shared_ptr<WidgetPanel> Demo::SharedWidgets() {
    if (!widgets_) {
        widgets_ =
            std::make_shared<WidgetPanel>("Options", Dock::Floating);
    }
    return widgets_;
}

void Demo::LogScalar(std::string_view name, double value, bool clear,
                     int smoothing) {
    std::vector<double>& samples = loggedScalars_[std::string(name)];
    if (clear) {
        samples.clear();
    }
    samples.push_back(value);
    if (viewer_ != nullptr) {
        viewer_->LogScalar(name, value, clear, smoothing);
    }
}

void Demo::LogArray(std::string_view name, const Tensor& array,
                    std::uintptr_t stream) {
    loggedArrays_[std::string(name)] = array;
    if (viewer_ != nullptr) {
        viewer_->LogArray(name, array, stream);
    }
}

void Demo::LogImage(std::string_view name, const Tensor& image,
                    std::uintptr_t stream) {
    loggedImages_[std::string(name)] = image;
    if (viewer_ != nullptr) {
        viewer_->LogImage(name, image, stream);
    }
}

std::vector<double> Demo::LoggedScalars(std::string_view name) const {
    const auto found = loggedScalars_.find(name);
    return found != loggedScalars_.end() ? found->second
                                         : std::vector<double>{};
}

Tensor Demo::LoggedArray(std::string_view name) const {
    const auto found = loggedArrays_.find(name);
    return found != loggedArrays_.end() ? found->second : Tensor{};
}

Tensor Demo::LoggedImage(std::string_view name) const {
    const auto found = loggedImages_.find(name);
    return found != loggedImages_.end() ? found->second : Tensor{};
}

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
        renderer_ = std::make_shared<Renderer>(RendererOptions{
            options_.renderer, options_.device, options_.validation});
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

std::string DemoInfo::Id() const {
    std::string id = "demo_" + category + "_" + name;
    std::replace(id.begin(), id.end(), ' ', '_');
    return id;
}

namespace {

std::vector<DemoInfo>& Registry() {
    static std::vector<DemoInfo> registry;
    return registry;
}

}  // namespace

DemoRegistration::DemoRegistration(DemoInfo info) {
    Registry().push_back(std::move(info));
}

std::vector<DemoInfo> RegisteredDemos() {
    std::vector<DemoInfo> demos = Registry();
    std::sort(demos.begin(), demos.end(),
              [](const DemoInfo& a, const DemoInfo& b) {
                  return std::tie(a.category, a.name) <
                         std::tie(b.category, b.name);
              });
    return demos;
}

std::optional<std::size_t> FindDemo(std::span<const DemoInfo> demos,
                                    std::string_view id) {
    std::string wanted(id);
    std::replace(wanted.begin(), wanted.end(), ' ', '_');
    for (std::size_t i = 0; i < demos.size(); ++i) {
        std::string label = demos[i].Label();
        std::replace(label.begin(), label.end(), ' ', '_');
        // The former "category/name" ids too.
        std::string path = demos[i].category + "/" + demos[i].name;
        std::replace(path.begin(), path.end(), ' ', '_');
        if (demos[i].Id() == wanted || label == wanted ||
            path == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

struct DemoRunner::State {
    DemoOptions options;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<Viewer> viewer;
    std::vector<DemoInfo> catalog;
    std::size_t current = 0;
    /// The scope the running demo registers in.
    std::uint64_t scope = 0;

    // What the viewport showed before any demo, restored when the
    // Demos panel switches.
    ViewportDisplay display;
    Viewer::Exposure exposure = Viewer::Exposure::Default;
    double compensation = 0.0;
    std::optional<Wireframe> wireframe;
    std::string viewVar;
    Viewer::Gizmo gizmo = Viewer::Gizmo::None;
    bool gizmoLocal = false;

    // The Demos panel, the demo it shows, and what it asked for
    // during the last frame.
    Panel* panel = nullptr;
    Demo* running = nullptr;
    std::string about;  ///< The running demo's description.
    bool paused = false;
    std::optional<std::size_t> switchTo;
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
        running = &demo;
        about = DescriptionOf(demo);
        viewer->Log(LogLevel::Info,
                    "Demo: " + (about.empty() ? "untitled" : about));
    }

    std::string DescriptionOf(Demo& demo) const {
        std::string description = demo.Description();
        if (description.empty() && current < catalog.size()) {
            description = catalog[current].description;
        }
        return description;
    }

    void AddPanel() {
        const std::uint64_t previous = viewer->Scope();
        viewer->SetScope(0);
        // In the upper right with the Stage, the Stage staying in
        // front.
        panel =
            &viewer->AddPanel("Demos", Dock::Right,
                              [this](Viewer& v) { DrawPanel(v); });
        if (viewer->FindPanel("Stage") != nullptr) {
            viewer->FocusPanel("Stage");
        }
        viewer->SetScope(previous);
    }

    /// The Demos panel, after Newton's example browser: a menu of the
    /// demos with a node per category, the running demo's controls
    /// and description, and its own options.
    void DrawPanel(Viewer& v) {
        // A collapsing header per category, the catalogue listing
        // each category's demos together; the running demo's opens
        // first.
        for (std::size_t first = 0; first < catalog.size();) {
            const std::string& category = catalog[first].category;
            std::size_t end = first;
            while (end < catalog.size() &&
                   catalog[end].category == category) {
                ++end;
            }
            ImGui::SetNextItemOpen(current >= first && current < end,
                                   ImGuiCond_Appearing);
            // Ids apart from the labels: a demo may share its
            // category's name (batched's "cameras").
            ImGui::PushID(("category/" + category).c_str());
            if (ImGui::CollapsingHeader(category.c_str())) {
                ImGui::Indent();
                for (std::size_t i = first; i < end; ++i) {
                    ImGui::PushID(catalog[i].Id().c_str());
                    if (ImGui::Selectable(catalog[i].name.c_str(),
                                          i == current &&
                                              running != nullptr)) {
                        switchTo = i;
                    }
                    ImGui::PopID();
                }
                ImGui::Unindent();
            }
            ImGui::PopID();
            first = end;
        }
        ImGui::SeparatorText(running != nullptr &&
                                     current < catalog.size()
                                 ? catalog[current].Label().c_str()
                                 : "No demo");
        ImGui::Checkbox("Paused", &paused);
        ImGui::SameLine();
        if (ImGui::Button("Step")) {
            singleStep = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            resetRequested = true;
        }
        if (running == nullptr) {
            return;
        }
        ImGui::SameLine();
        ImGui::TextDisabled(
            "frame %llu, %.2f s",
            static_cast<unsigned long long>(running->frame),
            running->time);
        if (!about.empty()) {
            ImGui::TextWrapped("%s", about.c_str());
        }
        ImGui::SeparatorText("Options");
        ImGui::PushID("options");
        running->Widgets().Draw(v);
        running->Gui();
        ImGui::PopID();
    }

    /// Replaces `old` with a new build of the catalogue's demo
    /// `index`; null, with the error in the log, if it cannot be
    /// built.
    std::shared_ptr<Demo> Load(std::shared_ptr<Demo> old,
                               std::size_t index) {
        running = nullptr;
        if (old) {
            old->Close();
            // Its widgets' callbacks may hold it (in Python, a cycle
            // through C++).
            old->Widgets().Clear();
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
            std::shared_ptr<Demo> demo =
                catalog.at(index).create(viewer.get(), demoOptions);
            viewer->ResetSession();
            current = index;
            Announce(*demo);
            return demo;
        } catch (const std::exception& error) {
            viewer->Log(LogLevel::Error,
                        "Could not load " +
                            catalog.at(index).Label() + ": " +
                            error.what());
            std::fprintf(stderr, "Could not load %s: %s\n",
                         catalog.at(index).Label().c_str(),
                         error.what());
            viewer->RemoveScope(scope);
            viewer->ClearScene();
            viewer->ResetSession();
            about.clear();
            return nullptr;
        }
    }

    /// Removes the running demo's registrations and the panel, which
    /// hold the demo (and in Python, through it the viewer).
    void Release() {
        running = nullptr;
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
    viewerOptions.profile = s.options.profile;
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

void DemoRunner::SetCatalog(std::vector<DemoInfo> demos,
                            std::size_t current) {
    if (!demos.empty() && current >= demos.size()) {
        throw std::out_of_range(
            "The current demo is not in the catalogue");
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
        throw std::logic_error(
            "Only a runner with a viewer switches demos");
    }
    if (index >= s.catalog.size()) {
        throw std::out_of_range("No demo " + std::to_string(index) +
                                " in the catalogue");
    }
    s.switchTo = index;
}

void DemoRunner::RequestReset() {
    State& s = *state_;
    if (!s.viewer || s.catalog.empty()) {
        throw std::logic_error("Only a runner with a viewer and a "
                               "catalogue resets demos");
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
            // The Demos panel, also in a hidden viewer, whose window
            // captures show it; not in test runs, which check one
            // demo.
            if (!o.test) {
                s.AddPanel();
            }
            s.Announce(*demo);
        }
        s.paused = o.paused && o.Windowed();
        if (o.profile) {
            profiler::Clear();
            profiler::SetHistory(kProfileHistory);
            profiler::SetEnabled(true);
        }
        std::optional<Clock::time_point> timerStart;
        std::uint64_t timedFrames = 0;
        Clock::time_point last = Clock::now();
        while (viewer == nullptr || viewer->IsOpen()) {
            if (s.switchTo || s.resetRequested) {
                const std::size_t index =
                    s.switchTo.value_or(s.current);
                s.switchTo.reset();
                s.resetRequested = false;
                demo = s.Load(std::move(demo), index);
            }
            if (!demo) {
                viewer->Step();
                continue;
            }
            const std::optional<std::uint64_t> limit =
                o.numFrames
                    ? o.numFrames
                    : (o.Windowed() ? std::nullopt
                                    : std::optional<std::uint64_t>(
                                          demo->NumFrames()));
            if (limit && demo->frame >= *limit) {
                break;
            }
            // A viewer starts the profiler's frames itself.
            if (o.profile && viewer == nullptr) {
                profiler::BeginFrame();
            }
            const Clock::time_point now = Clock::now();
            demo->frameDt =
                o.Windowed() ? std::min(std::chrono::duration<double>(
                                            now - last)
                                            .count(),
                                        demo->MaxFrameDt())
                             : demo->FixedDt();
            last = now;
            bool stepping = !s.paused;
            if (s.paused && s.singleStep) {
                stepping = true;
                s.singleStep = false;
            }
            // A window's sensor clock stands still while paused, but
            // for a step's worth at each single step.
            if (viewer != nullptr && o.Windowed()) {
                viewer->SetClockPaused(s.paused);
                if (s.paused && stepping) {
                    viewer->StepClock(demo->frameDt);
                }
            }
            if (stepping) {
                demo->Step();
                if (o.test) {
                    demo->TestPostStep();
                }
                // Headless, the sensor clock keeps the demo's time, a
                // frame ending where its step does; a window's viewer
                // runs it with its frames.
                if (!o.Windowed()) {
                    demo->GetRenderer().Delegate().AdvanceTime(
                        demo->frameDt);
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
                        std::chrono::duration<double>(Clock::now() -
                                                      *timerStart)
                                .count() >= *o.benchmarkSeconds) {
                        break;
                    }
                }
            }
        }
        // A benchmark's frames still running finish before its clock
        // stops.
        std::pair<std::uint32_t, std::uint32_t> extent{0, 0};
        Clock::time_point timerEnd = Clock::now();
        if (demo) {
            extent = demo->FinishFrames();
            timerEnd = Clock::now();
        }

        if (!o.output.empty() && demo) {
            demo->SaveOutput(o.output);
            std::printf("wrote %s\n", o.output.string().c_str());
        }
        if (o.profile) {
            ReportProfile(o);
        }
        if (o.test && demo) {
            demo->TestFinal();
            if (demo->Untested()) {
                throw std::logic_error(
                    "The demo has no checks: override TestFinal or "
                    "TestPostStep (test_final or "
                    "test_post_step in Python)");
            }
        }
        if (o.benchmark) {
            if (!timerStart) {
                std::printf("Benchmark: too few frames to time "
                            "(warm-up is %llu)\n",
                            static_cast<unsigned long long>(
                                kBenchmarkWarmup));
            } else {
                const double elapsed = std::chrono::duration<double>(
                                           timerEnd - *timerStart)
                                           .count();
                std::string size;
                if (extent.first > 0) {
                    size = ", " + std::to_string(extent.first) +
                           " x " + std::to_string(extent.second);
                }
                std::printf(
                    "Benchmark: %s FPS (%llu frames in %.2f s%s)\n",
                    FormatFps(static_cast<double>(timedFrames) /
                              std::max(elapsed, 1e-9))
                        .c_str(),
                    static_cast<unsigned long long>(timedFrames),
                    elapsed, size.c_str());
            }
        }
        std::fflush(stdout);
        if (demo) {
            demo->Close();
            // Its widgets' callbacks may hold it (in Python, a cycle
            // through C++).
            demo->Widgets().Clear();
        }
    } catch (...) {
        if (demo) {
            demo->Widgets().Clear();
        }
        s.Release();
        throw;
    }
    if (viewer != nullptr) {
        viewer->Close();
    }
    s.Release();
    return demo;
}

int DemoMain(int argc, char** argv, std::vector<DemoInfo> demos,
             std::size_t start) {
    const std::string program =
        argc > 0 ? std::filesystem::path(argv[0]).filename().string()
                 : std::string("demo");
    DemoOptions options;
    try {
        options = ParseDemoOptions(std::vector<std::string>(
            argv + std::min(argc, 1), argv + argc));
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "%s: %s\n\nUsage: %s [options]\n\n%s",
                     program.c_str(), error.what(), program.c_str(),
                     std::string(kHelp).c_str());
        return 2;
    }
    if (!options.demo.empty()) {
        const std::optional<std::size_t> found =
            FindDemo(demos, options.demo);
        if (!found) {
            std::fprintf(
                stderr,
                "%s: no demo named '%s' (--list lists them)\n",
                program.c_str(), options.demo.c_str());
            return 2;
        }
        start = *found;
    }
    if (options.help) {
        std::printf("Usage: %s [options]\n\n", program.c_str());
        if (start < demos.size() &&
            !demos[start].description.empty()) {
            std::printf("%s\n\n", demos[start].description.c_str());
        }
        std::printf("Options:\n%s", std::string(kHelp).c_str());
        return 0;
    }
    if (options.list) {
        for (const DemoInfo& info : demos) {
            std::printf("%s\n", info.Id().c_str());
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
        std::fprintf(stderr, "%s: %s\n", program.c_str(),
                     error.what());
        return 1;
    }
    return 0;
}

}  // namespace ernie::app
