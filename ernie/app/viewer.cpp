// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer.cpp
/// @brief The viewer's set-up and frame loop.
#include "ernie/app/viewer.h"

#include "ernie/app/imgui_renderer.h"
#include "ernie/app/imgui_session.h"
#include "ernie/app/offscreen_presenter.h"
#include "ernie/device/context.h"
#include "ernie/util/profiler.h"

#if ERNIE_VIEWER_WINDOW
#include "ernie/app/window_presenter.h"
#endif

#include <imgui.h>

#include <stdexcept>

namespace ernie::app {
namespace {

/// @return The presenter: a window's, made before the device context as its instance extensions
/// come from the window system, or an offscreen image's.
template <typename MakeContext>
std::unique_ptr<Presenter> MakePresenter(const ViewerOptions& options, MakeContext&& makeContext) {
    if (options.offscreen) {
        DeviceContext& context = makeContext(std::vector<std::string>());
        return std::make_unique<OffscreenPresenter>(context, options.size);
    }
#if ERNIE_VIEWER_WINDOW
    auto window = std::make_unique<Window>(options.title, options.size);
    ApplyTheme(options.shell.theme, window->ContentScale(), ImGui::GetStyle());
    DeviceContext& context = makeContext(Window::InstanceExtensions());
    return std::make_unique<WindowPresenter>(context, std::move(window), options.vsync);
#else
    throw std::runtime_error(
        "This ernie was built without a window system; --offscreen draws without one");
#endif
}

}  // namespace

Viewer::Viewer(const Scene& scene, const ViewerOptions& options)
    : options_(options), imgui_(std::make_unique<ImGuiSession>(options.shell.theme)) {
    if (options.size.x == 0 || options.size.y == 0) {
        throw std::invalid_argument("The viewer needs a positive size");
    }
    presenter_ = MakePresenter(options, [&](std::vector<std::string> extensions) -> DeviceContext& {
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

const OffscreenPresenter* Viewer::Offscreen() const {
    return dynamic_cast<const OffscreenPresenter*>(presenter_.get());
}

void Viewer::Run() {
    while (Step()) {
    }
}

bool Viewer::Step() {
    presenter_->PollEvents();
    if (presenter_->CloseRequested() || !shell_->Open()) {
        return false;
    }
    profiler::BeginFrame();
    const auto now = std::chrono::steady_clock::now();
    const double delta = std::chrono::duration<double>(now - last_).count();
    last_ = now;

    presenter_->BeginUiFrame(delta);
    renderer_->NewFrame();
    ImGui::NewFrame();
    shell_->Draw();
    ImGui::Render();

    if (const std::optional<FrameTarget> target = presenter_->Acquire()) {
        ERNIE_ZONE("Viewer draw");
        ImDrawData& data = *ImGui::GetDrawData();
        // The backend cycles a vertex buffer per image, which the frame that last drew with it
        // must have finished with.
        while (inFlight_.size() >= presenter_->ImageCount()) {
            inFlight_.front().Wait();
            inFlight_.pop_front();
        }
        renderer_->UpdateTextures(data);
        const Ticket drawn = context_->Main().Submit([&](VkCommandBuffer commands) {
            renderer_->Record(commands, data, *target, shell_->GetTheme().viewport);
        });
        inFlight_.push_back(drawn);
        presenter_->Present(drawn);
    }
    context_->Collect();
    ++frames_;
    return shell_->Open() && (options_.frames == 0 || frames_ < options_.frames);
}

}  // namespace ernie::app
