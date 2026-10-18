// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer.cpp
/// @brief The viewer's set-up and frame loop, and its command line.
#include "ernie/app/viewer.h"

#include "ernie/scene/json.h"
#include "ernie/scene/scene.h"
#include "ernie/util/profiler.h"

#include <imgui.h>

#include <charconv>
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

Viewer::Viewer(const Scene& scene, const ViewerOptions& options)
    : options_(options), imgui_(options.shell.theme) {
    if (options.size.x == 0 || options.size.y == 0) {
        throw std::invalid_argument("The viewer needs a positive size");
    }
    presenter_ = MakePresenter(options, [&](std::vector<std::string> extensions) -> DeviceContext& {
        ERNIE_ZONE("Create the device");
        context_ = std::make_unique<DeviceContext>(
            DeviceContextOptions{.validation = options.validation,
                                 .device = options.device,
                                 .instanceExtensions = std::move(extensions)});
        return *context_;
    });
    renderer_ =
        std::make_unique<ImGuiRenderer>(*context_, presenter_->Format(), presenter_->ImageCount());
    shell_ = std::make_unique<Shell>(scene, context_.get(), options.shell);
    shell_->Frame();
    profiler::SetThreadName("Main");
}

Viewer::~Viewer() = default;

void Viewer::Run() {
    while (Step()) {
    }
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
    shell_->Draw();
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
            ERNIE_ZONE("Submit the UI");
            drawn = context_->Main().Submit(
                [&](VkCommandBuffer commands) {
                    ERNIE_ZONE("Record the UI");
                    renderer_->Record(commands, data, *target, shell_->GetTheme().viewport);
                },
                {}, target->semaphores);
        }
        inFlight_.push_back(drawn);
        ERNIE_ZONE("Present");
        presenter_->Present(drawn);
    }
    context_->Collect();
    ++frames_;
    return shell_->Open() && (options_.frames == 0 || frames_ < options_.frames);
}

std::string_view ViewUsage() {
    return "ernie [view] [SCENE.json] [--offscreen] [--frames N] [--size WxH] [--device NAME]\n"
           "             [--validation] [--vsync] [--profile]";
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
    return 0;
}

}  // namespace ernie::app
