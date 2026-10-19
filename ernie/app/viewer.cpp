// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer.cpp
/// @brief The viewer's set-up and frame loop, and its command line.
#include "ernie/app/viewer.h"

#include "ernie/app/capture.h"
#include "ernie/io/image.h"
#include "ernie/scene/json.h"
#include "ernie/scene/scene.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/process.h"
#include "ernie/util/profiler.h"

#include <imgui.h>

#include <charconv>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie::app {
namespace {

/// @return The presenter: a window's, made before the device context as its instance extensions
/// come from the window system, or an offscreen image's.
template <typename MakeContext>
std::unique_ptr<Presenter> MakePresenter(const ViewerOptions& options, MakeContext&& makeContext) {
    if (options.offscreen) {
        DeviceContext& context = makeContext(std::vector<std::string>());
        ERNIE_ZONE("Create the offscreen image");
        return std::make_unique<OffscreenPresenter>(context, options.size);
    }
#if ERNIE_VIEWER_WINDOW
    std::unique_ptr<Window> window;
    {
        ERNIE_ZONE("Open the window");
        window = std::make_unique<Window>(options.title, options.size);
    }
    ApplyTheme(options.shell.theme, window->ContentScale());
    DeviceContext& context = makeContext(Window::InstanceExtensions());
    ERNIE_ZONE("Set up presenting to the window");
    return std::make_unique<WindowPresenter>(context, std::move(window), options.vsync);
#else
    throw std::runtime_error(
        "This ernie was built without a window system; --offscreen draws without one");
#endif
}

/// @return Where the viewer keeps compiled pipelines between runs; empty when caching is off.
std::filesystem::path PipelineCachePath() {
    const std::optional<std::filesystem::path> folder =
        CacheDirectory("ERNIE_PIPELINE_CACHE", "pipelines");
    return folder ? *folder / "viewer.bin" : std::filesystem::path();
}

template <typename Number> Number ParseNumber(std::string_view text, std::string_view option) {
    Number value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string(option) + " takes a number, not " +
                                    std::string(text));
    }
    return value;
}

glm::uvec2 ParseSize(std::string_view text) {
    const std::size_t x = text.find('x');
    if (x == std::string_view::npos) {
        throw std::invalid_argument("--size takes WIDTHxHEIGHT, not " + std::string(text));
    }
    const glm::uvec2 size(ParseNumber<std::uint32_t>(text.substr(0, x), "--size"),
                          ParseNumber<std::uint32_t>(text.substr(x + 1), "--size"));
    if (size.x == 0 || size.y == 0) {
        throw std::invalid_argument("--size needs a positive width and height");
    }
    return size;
}

}  // namespace

Viewer::Viewer(Scene& scene, const ViewerOptions& options)
    : options_(options), scene_(scene), imgui_(options.shell.theme) {
    if (options.size.x == 0 || options.size.y == 0) {
        throw std::invalid_argument("The viewer needs a positive size");
    }
    presenter_ = MakePresenter(options, [&](std::vector<std::string> extensions) -> DeviceContext& {
        ERNIE_ZONE("Create the device");
        context_ = std::make_unique<DeviceContext>(
            DeviceContextOptions{.validation = options.validation,
                                 .device = options.device,
                                 .instanceExtensions = std::move(extensions),
                                 .pipelineCache = PipelineCachePath()});
        return *context_;
    });
    renderer_ =
        std::make_unique<ImGuiRenderer>(*context_, presenter_->Format(), presenter_->ImageCount());
    shell_ = std::make_unique<Shell>(scene, context_.get(), options.shell);
    shell_->Frame();
    {
        ERNIE_ZONE("Set up the device scene");
        arenas_ = std::make_unique<Arenas>(*context_, MemoryUse::Device, "scene columns",
                                           std::uint64_t{16} << 20);
        const ShaderCompiler compiler;
        deviceScene_ = std::make_unique<DeviceScene>(*context_, *arenas_, scene, compiler);
        if (deviceScene_->Tracer() != nullptr) {
            heap_ = std::make_unique<BindlessHeap>(*context_);
            camera_ = std::make_unique<CameraPass>(*context_, compiler, *heap_);
            // On one queue, a trace and the UI run in turn whatever the order: tracing apart
            // would only show each image a frame later.
            async_ = options.asyncRender && &context_->Compute() != &context_->Main();
            if (async_) {
                deviceScene_->SetQueue(context_->Compute());
            }
        }
        sampler_ = std::make_unique<Sampler>(
            *context_, SamplerDescription{.filter = Filter::Nearest, .address = Address::Clamp},
            "viewport");
    }
    profiler::SetThreadName("Main");
}

Viewer::~Viewer() {
    // Frames and traces in flight sample the viewport and read the frame's columns.
    Quietly([this] {
        for (const Queue* queue : context_->Queues()) {
            queue->Last().Wait();
        }
    });
}

void Viewer::Run() {
    while (Step()) {
    }
}

void Viewer::RunFrames(std::uint64_t count) {
    for (std::uint64_t frame = 0; frame < count && !presenter_->CloseRequested() && shell_->Open();
         ++frame) {
        Step();
    }
}

Tensor Viewer::ReadImage(const DeviceImage& image, const Ticket& drawn) {
    const glm::uvec2 size(image.Description().extent);
    const std::uint64_t bytes = std::uint64_t{4} * size.x * size.y;
    const Buffer staging(*context_, bytes, MemoryUse::Readback, "Image readback");
    context_->Main()
        .Submit(
            [&](VkCommandBuffer commands) {
                const VkBufferImageCopy region{
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                    .imageExtent = {size.x, size.y, 1}};
                vkCmdCopyImageToBuffer(commands, image.Handle(), VK_IMAGE_LAYOUT_GENERAL,
                                       staging.Handle(), 1, &region);
            },
            std::span(&drawn, 1))
        .Wait();
    staging.Invalidate(0, bytes);
    const std::span<const std::byte> texels = staging.Mapped().first(bytes);
    return Tensor::Copy(
        std::span(reinterpret_cast<const std::uint8_t*>(texels.data()), texels.size()),
        {size.y, size.x, 4});
}

void Viewer::SaveScreenshot(const std::filesystem::path& path) {
    const OffscreenPresenter* offscreen = Offscreen();
    if (offscreen == nullptr) {
        throw std::logic_error("Screenshots are of offscreen frames");
    }
    WritePng(path, ReadImage(offscreen->Image(), offscreen->Drawn()));
}

std::filesystem::path Viewer::CapturePath(std::string_view extension, std::int64_t frame) const {
    CaptureNameFields fields = shell_->CaptureFields();
    fields.frame = frame;
    const CaptureSettings& settings = shell_->Capture();
    std::filesystem::create_directories(settings.folder);
    return UniquePath(settings.folder /
                      (ExpandCaptureName(settings.name, fields) + std::string(extension)));
}

void Viewer::Capture(const Ticket& drawn) {
    if (shell_->TakeScreenshot()) {
        try {
            const std::filesystem::path path = CapturePath(".png", 0);
            WritePng(path, ReadImage(*slots_[shown_].image, drawn));
            shell_->AddRecentCapture(path);
        } catch (const std::exception& error) {
            shell_->Log().Error(std::string("Capturing: ") + error.what());
        }
    }
    if (recording_) {
        RecordFrame(drawn);
    }
}

void Viewer::StartSequence(const SequenceRequest& request) {
    const std::filesystem::path folder = CapturePath("", 0);
    std::filesystem::create_directories(folder);
    const auto frames =
        static_cast<std::int64_t>(std::floor((request.end - request.start) / request.every)) + 1;
    recording_ = Recording{.request = request,
                           .folder = folder,
                           .next = request.start,
                           .total = std::max<std::int64_t>(frames, 1)};
    shell_->Timeline().Pause();
}

void Viewer::RecordFrame(const Ticket& drawn) {
    Recording& recording = *recording_;
    std::array<char, 32> name{};
    std::snprintf(name.data(), name.size(), "frame_%04lld.png",
                  static_cast<long long>(recording.done));
    WritePng(recording.folder / name.data(), ReadImage(*slots_[shown_].image, drawn));
    ++recording.done;
    recording.next += recording.request.every;
    if (recording.done >= recording.total || shell_->stopRecording) {
        FinishSequence();
    }
}

void Viewer::FinishSequence() {
    const Recording recording = *recording_;
    recording_.reset();
    shell_->stopRecording = false;
    shell_->AddRecentCapture(recording.folder);
    if (recording.request.video == VideoFormat::None) {
        return;
    }
    const std::filesystem::path video =
        UniquePath(recording.folder.string() + VideoExtension(recording.request.video));
    const std::vector<std::string> command =
        EncodeCommand(FindProgram("ffmpeg"), recording.folder / "frame_%04d.png",
                      shell_->Timeline().FramesPerSecond() / recording.request.every, video,
                      recording.request.video);
    const bool keepFrames = recording.request.keepFrames;
    // Encoding takes a while: the viewer keeps running meanwhile.
    encoding_ =
        std::async(std::launch::async, [command, video, folder = recording.folder, keepFrames] {
            const bool encoded = RunProcess(command) == 0;
            if (encoded && !keepFrames) {
                std::filesystem::remove_all(folder);
            }
            return encoded ? video : std::filesystem::path();
        });
}

void Viewer::FollowRecording() {
    if (encoding_.valid() &&
        encoding_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const std::filesystem::path video = encoding_.get();
        if (video.empty()) {
            shell_->Log().Error("ffmpeg could not encode the sequence");
        } else {
            shell_->AddRecentCapture(video);
        }
    }
    if (const std::optional<SequenceRequest> request = shell_->TakeSequence()) {
        try {
            StartSequence(*request);
        } catch (const std::exception& error) {
            shell_->Log().Error(std::string("Recording: ") + error.what());
        }
    }
    if (recording_) {
        // Each frame renders the next time code of the sequence.
        shell_->Timeline().Seek(scene_, recording_->next);
    }
    shell_->SetRecording(recording_ ? std::optional(std::pair(recording_->done, recording_->total))
                                    : std::nullopt,
                         encoding_.valid());
}

bool Viewer::Step() {
    profiler::BeginFrame();
    ERNIE_FRAME_MARK("Viewer frame");
    {
        ERNIE_ZONE("Poll events");
        presenter_->PollEvents();
    }
    if (presenter_->CloseRequested() || !shell_->Open()) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    const double delta = std::chrono::duration<double>(now - last_).count();
    last_ = now;

    {
        ERNIE_ZONE("Begin the UI frame");
        presenter_->BeginUiFrame(delta);
        renderer_->NewFrame();
        ImGui::NewFrame();
    }
    FollowRecording();
    shell_->Draw();
    {
        // The menus' edits resolve before the device scene follows them.
        ERNIE_ZONE("Update the scene");
        scene_.Update();
    }
    {
        ERNIE_ZONE("Show the viewport");
        ShowViewport();
    }
    {
        ERNIE_ZONE("Build the UI's draw data");
        ImGui::Render();
    }

    std::optional<FrameTarget> target;
    {
        ERNIE_ZONE("Acquire an image");
        target = presenter_->Acquire();
    }
    if (target) {
        ImDrawData& data = *ImGui::GetDrawData();
        {
            // The backend cycles a vertex buffer per image, which the frame that last drew with
            // it must have finished with.
            ERNIE_ZONE("Wait for a frame in flight");
            while (inFlight_.size() >= presenter_->ImageCount()) {
                inFlight_.front().Wait();
                inFlight_.pop_front();
            }
        }
        renderer_->UpdateTextures(data);
        Ticket drawn;
        {
            ERNIE_ZONE("Run the frame");
            if (tracingNow_) {
                deviceScene_->Follow(shell_->Camera().Eye());
                const std::size_t slot = *tracingNow_;
                slots_[slot].traced = TraceViewport(slot);
                // The UI shows it this frame, waiting for it, or once it has finished.
                if (slot == shown_) {
                    traced_ = true;
                } else {
                    tracing_.push_back(slot);
                }
            } else if (!camera_ && !ecoPaused_) {
                // The device scene still follows the scene, with nothing to trace.
                deviceScene_->Step([](graph::Graph&) {}, recorded_);
            }
            drawn = DrawUi(data, *target);
            slots_[shown_].sampled = drawn;
        }
        inFlight_.push_back(drawn);
        if (const std::optional<Shell::Pick> pick = shell_->TakePick()) {
            ERNIE_ZONE("Pick");
            Pick(*pick, drawn);
        }
        {
            ERNIE_ZONE("Capture");
            Capture(drawn);
        }
        ERNIE_ZONE("Present");
        presenter_->Present(drawn);
    }
    {
        ERNIE_ZONE("Free retired objects");
        context_->Collect();
    }
    ++frames_;
    return shell_->Open() && (options_.frames == 0 || frames_ < options_.frames);
}

void Viewer::ShowViewport() {
    const auto [position, size] = shell_->Viewport();
    const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
    const glm::uvec2 pixels(std::max(1.0f, size.x * scale.x), std::max(1.0f, size.y * scale.y));
    if (!slots_[0].image || glm::uvec2(slots_[0].image->Description().extent) != pixels) {
        ResizeViewport(pixels);
    }
    ecoPaused_ = EcoPause();
    shell_->SetEcoPaused(ecoPaused_);
    ChooseSlots();
    ImGui::GetBackgroundDrawList()->AddImage(slots_[shown_].texture, position,
                                             {position.x + size.x, position.y + size.y});
}

void Viewer::ChooseSlots() {
    tracingNow_.reset();
    if (!camera_) {
        return;
    }
    // Traces finish in order on their queue: the newest finished one is shown.
    while (!tracing_.empty() && slots_[tracing_.front()].traced.Done()) {
        shown_ = tracing_.front();
        tracing_.pop_front();
        traced_ = true;
    }
    if (ecoPaused_) {
        return;
    }
    // A trace takes a slot neither shown nor tracing, so their images stay whole.
    std::size_t next = 0;
    while (next == shown_ || std::ranges::find(tracing_, next) != tracing_.end()) {
        ++next;
    }
    const bool capturing = recording_ || shell_->ScreenshotRequested();
    if (!async_ || capturing || !traced_) {
        // This frame's own image, which the UI waits for; older traces, still running, show
        // nothing newer.
        tracingNow_ = next;
        shown_ = next;
        tracing_.clear();
    } else if (tracing_.size() < kSlots - 1) {
        tracingNow_ = next;
    }
}

void Viewer::ResizeViewport(glm::uvec2 pixels) {
    ERNIE_ZONE("Resize the viewport");
    // Frames and traces in flight may still use the old images.
    for (const Queue* queue : context_->Queues()) {
        queue->Last().Wait();
    }
    tracing_.clear();
    traced_ = false;
    for (ViewportSlot& slot : slots_) {
        slot.traced = {};
        slot.sampled = {};
        if (slot.image) {
            renderer_->RemoveTexture(slot.texture);
            if (heap_) {
                heap_->RemoveStorageImage(slot.storage);
            }
        }
        slot.image = std::make_unique<DeviceImage>(
            *context_,
            ImageDescription{.format = ImageFormat::Rgba8Unorm,
                             .extent = {pixels, 1},
                             .usage = ImageUsage::Sampled | ImageUsage::Storage},
            "Viewport");
        slot.texture = renderer_->AddTexture(slot.image->View(), sampler_->Handle());
        if (heap_) {
            slot.storage = heap_->AddStorageImage(*slot.image);
            slot.prims = std::make_unique<DeviceColumn>(*arenas_, "viewport prims", DType::UInt32,
                                                        std::vector<std::int64_t>{},
                                                        std::uint64_t{pixels.x} * pixels.y);
            if (!slot.pose) {
                slot.pose = std::make_unique<Buffer>(*context_, kPoseBytes, MemoryUse::Upload,
                                                     "viewport pose");
            }
        }
    }
    // The recorded frames name the old images.
    ++recorded_;
}

void Viewer::Pick(const Shell::Pick& pick, const Ticket& drawn) {
    const ViewportSlot& shown = slots_[shown_];
    if (!shown.prims) {
        return;
    }
    const DeviceColumn& prims = *shown.prims;
    const glm::uvec2 size(shown.image->Description().extent);
    const glm::uvec2 pixel = glm::min(glm::uvec2(pick.at * glm::vec2(size)), size - 1u);
    const std::uint64_t at = std::uint64_t{pixel.y} * size.x + pixel.x;
    Readback readback(*context_, context_->Main(), prims.Storage(),
                      prims.Range().offset + at * sizeof(std::uint32_t), sizeof(std::uint32_t),
                      std::span(&drawn, 1));
    const std::uint32_t slot = readback.Values<std::uint32_t>().front();
    const std::optional<PrimId> prim = scene_.Store().PrimAtSlot(slot);
    shell_->Picked(prim ? std::optional(scene_.PrimPath(*prim)) : std::nullopt, pick.toggle);
}

bool Viewer::EcoPause() {
    const OrbitCamera& orbit = shell_->Camera();
    const ViewState now{.version = scene_.Store().Version(),
                        .camera = orbit.Transform(),
                        .fov = orbit.verticalFov,
                        .size = glm::uvec2(slots_[0].image->Description().extent)};
    stillFrames_ = lastView_ == now ? stillFrames_ + 1 : 0;
    lastView_ = now;
    const ViewportDisplay& display = shell_->Display();
    return display.eco &&
           stillFrames_ >= static_cast<std::uint64_t>(std::max(display.ecoFrames, 1));
}

Ticket Viewer::TraceViewport(std::size_t index) {
    const OrbitCamera& orbit = shell_->Camera();
    ViewportSlot& slot = slots_[index];
    const glm::uvec2 size(slot.image->Description().extent);
    const RecordedView view{.size = size,
                            .fov = orbit.verticalFov,
                            .background = glm::vec3(shell_->GetTheme().viewport)};
    if (recordedView_ != view) {
        recordedView_ = view;
        ++recorded_;
    }
    {
        // Written in place from the host once the slot's last trace, which read it, is done:
        // the submission makes it visible to this one.
        ERNIE_ZONE("Write the viewport pose");
        if (slot.traced.timeline != nullptr) {
            slot.traced.Wait();
        }
        const glm::dmat4 relative =
            glm::translate(glm::dmat4(1.0), -deviceScene_->Transforms().Origin()) *
            orbit.Transform();
        const std::array<glm::vec4, 3> rows = ToAffineRows(relative);
        std::memcpy(slot.pose->Mapped().data(), rows.data(), kPoseBytes);
        slot.pose->Flush(0, kPoseBytes);
    }
    // After the UI frames that sampled the slot's last image.
    const std::span<const Ticket> sampled =
        slot.sampled.timeline != nullptr ? std::span(&slot.sampled, 1) : std::span<const Ticket>();
    return deviceScene_->Step(
        [&](graph::Graph& graph) {
            CameraOutputs outputs{.colour = graph.Import(*slot.image, "viewport"),
                                  .colourIndex = slot.storage,
                                  .background = view.background,
                                  .prims = graph.Import(*slot.prims)};
            camera_->Add(graph,
                         {.cameraToWorld = orbit.Transform(),
                          .size = size,
                          .lens = MakePinhole(orbit.verticalFov, size),
                          .pose = graph.Import(*slot.pose, 0, kPoseBytes, "viewport pose")},
                         outputs);
        },
        recorded_ * kSlots + index, sampled);
}

Ticket Viewer::DrawUi(ImDrawData& data, const FrameTarget& target) {
    graph::Graph graph(*context_, deviceScene_->Pool());
    const ViewportSlot& shown = slots_[shown_];
    graph::Image viewport = graph.Import(*shown.image, "viewport");
    if (!camera_) {
        graph.Add(
            "Viewport", graph::Kind::Copy,
            [&](graph::PassBuilder& pass) { viewport = pass.Write(viewport, graph::Use::CopyDst); },
            [this, viewport](graph::PassContext& pass) {
                pass.Clear(viewport, shell_->GetTheme().viewport);
            });
    }
    graph.Add(
        "UI", graph::Kind::Raster,
        [&](graph::PassBuilder& pass) {
            pass.Read(viewport, graph::Use::Sampled);
            // It draws into the presenter's image, outside the graph.
            pass.NeverCull();
        },
        [&](graph::PassContext& pass) {
            ERNIE_ZONE("Record the UI");
            // Panels and the viewport's image cover every pixel.
            renderer_->Record(pass.commands.Handle(), data, target,
                              glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        });
    // The shown trace has finished unless this frame waits for its own; waiting on it either
    // way makes its writes visible across queues.
    const std::span<const Ticket> traced =
        shown.traced.timeline != nullptr ? std::span(&shown.traced, 1) : std::span<const Ticket>();
    return graph.Execute(context_->Main(), traced, target.semaphores);
}

std::string_view ViewUsage() {
    return "ernie [view] [SCENE.json] [--offscreen] [--frames N] [--size WxH] [--device NAME]\n"
           "             [--validation] [--vsync] [--profile] [--screenshot FILE.png]";
}

ViewCommand ParseViewCommand(std::span<const std::string_view> arguments) {
    ViewCommand command;
    ViewerOptions& viewer = command.viewer;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view argument = arguments[i];
        const auto value = [&]() {
            if (i + 1 >= arguments.size()) {
                throw std::invalid_argument(std::string(argument) + " needs a value");
            }
            return arguments[++i];
        };
        if (argument == "--offscreen") {
            viewer.offscreen = true;
        } else if (argument == "--frames") {
            viewer.frames = ParseNumber<std::uint64_t>(value(), argument);
        } else if (argument == "--size") {
            viewer.size = ParseSize(value());
        } else if (argument == "--device") {
            viewer.device = value();
        } else if (argument == "--validation") {
            viewer.validation = true;
        } else if (argument == "--vsync") {
            viewer.vsync = true;
        } else if (argument == "--profile") {
            viewer.shell.profile = true;
        } else if (argument == "--screenshot") {
            viewer.screenshot = value();
            viewer.offscreen = true;
        } else if (argument.starts_with("-")) {
            throw std::invalid_argument("Unknown option " + std::string(argument));
        } else if (command.scene) {
            throw std::invalid_argument("The viewer opens one scene");
        } else {
            command.scene = std::filesystem::path(argument);
        }
    }
    if (command.scene) {
        viewer.title = "Ernie - " + command.scene->filename().string();
    }
    return command;
}

int RunView(const ViewCommand& command) {
    Scene scene;
    {
        ERNIE_ZONE("Load the scene");
        if (command.scene) {
            LoadSceneFile(scene, *command.scene);
        }
        scene.Update();
    }
    Viewer viewer(scene, command.viewer);
    if (command.scene) {
        viewer.GetShell().Log().Info("Opened " + command.scene->string());
    }
    viewer.Run();
    if (!command.viewer.screenshot.empty()) {
        viewer.SaveScreenshot(command.viewer.screenshot);
    }
    return 0;
}

}  // namespace ernie::app
