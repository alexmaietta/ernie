// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <exception>
#include <format>
#include <memory>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <imgui.h>

#include "ernie/app/viewer/panels/panels.h"
#include "ernie/render/render_output.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/rotation.h"
#include "ernie/xr/rig.h"
#include "ernie/xr/session.h"

namespace ernie::app {
namespace {

constexpr const char* kViewport = "XR";
constexpr const char* kRigPath = "/__ernie_xr";

struct Desk {
    Double3 head{0, 0, 1.6};
    double yaw = 90;
    double pitch = 0;
    std::set<std::string> keys;
    bool trigger = false;
    bool looking = false;
    float lastX = 0;
    float lastY = 0;
    double lastTime = 0;
};

Double3 Looking(double yaw, double pitch) {
    const double y = yaw * std::numbers::pi / 180;
    const double p = pitch * std::numbers::pi / 180;
    return {std::cos(p) * std::cos(y), std::cos(p) * std::sin(y), std::sin(p)};
}

void WalkDesk(Desk& desk, double time, xr::SimulatedInput& input) {
    const double dt = desk.lastTime > 0 ? time - desk.lastTime : 0;
    desk.lastTime = time;
    const Double3 ahead = Looking(desk.yaw, 0);
    const Double3 right{ahead.y, -ahead.x, 0};
    const auto held = [&](const char* key) { return desk.keys.contains(key) ? 1.0 : 0.0; };
    const double forward = held("W") - held("S");
    const double sideways = held("D") - held("A");
    desk.head.x += 1.5 * dt * (ahead.x * forward + right.x * sideways);
    desk.head.y += 1.5 * dt * (ahead.y * forward + right.y * sideways);
    const Quatd look = LookAlong(Looking(desk.yaw, desk.pitch));
    input.head = {desk.head, look, true};
    for (const xr::Side side : {xr::Side::Left, xr::Side::Right}) {
        xr::Controller& controller = input.controllers[uint32_t(side)];
        const double offset = side == xr::Side::Left ? -0.2 : 0.2;
        controller.active = true;
        controller.pose = {{desk.head.x, desk.head.y, desk.head.z}, look, true};
        const Double3 at = Rotate(look, {offset, -0.3, -0.3});
        controller.pose.position.x += at.x;
        controller.pose.position.y += at.y;
        controller.pose.position.z += at.z;
        controller.aim = controller.pose;
    }
    if (desk.trigger) {
        input.Press(xr::Side::Right, "trigger/value", 1);
    }
    const float stickX = float(held("RightArrow") - held("LeftArrow"));
    const float stickY = float(held("UpArrow") - held("DownArrow"));
    if (stickX != 0) {
        input.Press(xr::Side::Left, "thumbstick/x", stickX);
    }
    if (stickY != 0) {
        input.Press(xr::Side::Left, "thumbstick/y", stickY);
    }
    if (held("Q") != held("E")) {
        input.Press(xr::Side::Right, "thumbstick/x", float(held("E") - held("Q")));
    }
}

void MarkSessionPrims(Scene& scene, const Prim& prim) {
    scene.MarkSession(prim);
    for (const Prim& child : prim.Children()) {
        MarkSessionPrims(scene, child);
    }
}

class XrMode : public RenderDelegate, public std::enable_shared_from_this<XrMode> {
  public:
    XrMode(Viewer& viewer, Ref<Scene> scene) : m_viewer(viewer), m_scene(std::move(scene)) {}

    std::string_view Name() const override {
        return "XR";
    }

    bool Running() const {
        return m_session != nullptr;
    }

    void Start() {
        m_error.clear();
        try {
            xr::SessionDesc desc;
            desc.application = "Ernie viewer";
            desc.backend = m_backend;
            desc.resolutionScale = m_resolutionScale;
            desc.depthLayer = m_depthLayer;
            desc.blendMode = m_blend;
            desc.simulated.realTime = true;
            const std::shared_ptr<Desk> desk = m_desk;
            desc.simulated.script = [desk](uint64_t, double time, xr::SimulatedInput& input) {
                WalkDesk(*desk, time, input);
            };
            m_session = std::make_unique<xr::Session>(desc);
            m_rig.emplace(m_scene, *m_session, xr::RigDesc{.path = kRigPath});
            MarkSessionPrims(*m_scene, m_rig->Root());
            m_renderer = std::make_shared<Renderer>(m_scene);
            m_viewer.AddViewport(
                {.name = kViewport, .delegate = shared_from_this(), .hud = true, .toolbar = false});
            if (m_backend == xr::Backend::Simulated) {
                Watch();
            }
        } catch (const std::exception& error) {
            m_error = error.what();
            Stop();
        }
    }

    void Stop() {
        for (const uint64_t handler : m_handlers) {
            m_viewer.RemoveHandler(handler);
        }
        m_handlers.clear();
        m_renderer.reset();
        m_rig.reset();
        m_session.reset();
        if (const std::optional<Prim> rig = m_scene->Find(kRigPath)) {
            m_scene->Remove(*rig);
        }
    }

    std::optional<RenderFrame> Render(const RenderRequest&) override {
        if (!m_session) {
            return std::nullopt;
        }
        try {
            if (!m_session->Running()) {
                Stop();
                return std::nullopt;
            }
            return Step();
        } catch (const std::exception& error) {
            m_error = error.what();
            Stop();
            return std::nullopt;
        }
    }

    void Draw() {
        if (Running()) {
            DrawRunning();
            return;
        }
        const std::vector<std::string> viewports = m_viewer.Viewports();
        if (std::ranges::find(viewports, kViewport) != viewports.end()) {
            m_viewer.RemoveViewport(kViewport);
        }
        const char* backends[] = {"Simulated", "OpenXR"};
        int backend = m_backend == xr::Backend::Simulated ? 0 : 1;
        if (ImGui::Combo("Backend", &backend, backends, 2)) {
            m_backend = backend == 0 ? xr::Backend::Simulated : xr::Backend::OpenXR;
        }
        ImGui::SliderFloat("Resolution scale", &m_resolutionScale, 0.1f, 1.0f, "%.2f");
        ImGui::Checkbox("Depth layer", &m_depthLayer);
        const char* blends[] = {"Opaque", "Additive", "Passthrough"};
        int blend = int(m_blend);
        if (ImGui::Combo("Blend", &blend, blends, 3)) {
            m_blend = xr::BlendMode(blend);
        }
        if (ImGui::Button("Start")) {
            Start();
        }
        if (m_backend == xr::Backend::Simulated) {
            ImGui::TextWrapped("In the XR viewport, WASD walks, right-drag looks, the left button "
                               "aims and teleports, the arrow keys move and Q and E turn.");
        }
        if (!m_error.empty()) {
            ImGui::TextWrapped("%s", m_error.c_str());
        }
    }

  private:
    std::optional<RenderFrame> Step() {
        const xr::Frame frame = m_session->BeginFrame();
        m_rig->Update(frame);
        const xr::Controller& right = frame.Controller(xr::Side::Right);
        const xr::Controller& left = frame.Controller(xr::Side::Left);
        const bool aiming = right.active && right.trigger > 0.5f;
        if (m_aiming && !aiming) {
            m_rig->TeleportTo(frame, m_rig->Teleport(xr::Side::Right));
        }
        m_aiming = aiming;
        if (left.active) {
            m_rig->Move(frame, left.thumbstick);
        }
        const float turn = right.active ? right.thumbstick.x : 0;
        if (std::abs(turn) > 0.6f && !m_turned) {
            m_rig->SnapTurn(frame, turn > 0 ? -1 : 1);
        }
        m_turned = std::abs(turn) > 0.6f;
        xr::RigVisuals visuals = m_visuals;
        visuals.teleport = m_aiming;
        m_rig->Draw(*m_renderer->GetOverlays(), frame, visuals);
        if (!frame.shouldRender) {
            m_session->EndFrame(frame, RenderOutput());
            return std::nullopt;
        }
        RenderOutput output = m_renderer->Step(m_rig->Product(), frame.deltaTime);
        m_session->EndFrame(frame, output);
        const FrameOutput& eyes = output[m_rig->Product()].frames.back();
        RenderFrame shown;
        shown.image = eyes.RenderVar(m_rig->ColorVar()).tensor.Select(0);
        shown.ready = output.Done();
        if (!eyes.overlay.Empty()) {
            shown.overlay = eyes.overlay.Select(0);
        }
        return shown;
    }

    void Watch() {
        const std::shared_ptr<Desk> desk = m_desk;
        m_handlers.push_back(m_viewer.AddKeyHandler([desk](const KeyEvent& event) {
            if (event.action == InputAction::Press) {
                desk->keys.insert(event.key);
            } else if (event.action == InputAction::Release) {
                desk->keys.erase(event.key);
            }
            return false;
        }));
        m_handlers.push_back(m_viewer.AddMouseHandler([desk](const MouseEvent& event) {
            const bool ours = event.inViewport && event.viewport == kViewport;
            if (event.action == InputAction::Press && ours && event.button == 0) {
                desk->trigger = true;
            } else if (event.action == InputAction::Press && ours && event.button == 1) {
                desk->looking = true;
            } else if (event.action == InputAction::Release && event.button == 0) {
                desk->trigger = false;
            } else if (event.action == InputAction::Release && event.button == 1) {
                desk->looking = false;
            } else if (event.action == InputAction::Move && desk->looking) {
                desk->yaw -= 0.2 * (event.x - desk->lastX);
                desk->pitch = std::clamp(desk->pitch - 0.2 * (event.y - desk->lastY), -85.0, 85.0);
            }
            desk->lastX = event.x;
            desk->lastY = event.y;
            return ours && event.button >= 0;
        }));
    }

    void DrawRunning() {
        const xr::SessionStatus status = m_session->Status();
        const xr::SessionStats stats = m_session->Stats();
        ImGui::Text("%s", status.runtime.c_str());
        ImGui::Text("%s, %s", status.system.c_str(), status.state.c_str());
        ImGui::Text("%u x %u per eye at %.0f Hz", status.resolution.x, status.resolution.y,
                    status.refreshRate);
        ImGui::Text("%llu frames, %llu presented, %llu late, %llu missed",
                    static_cast<unsigned long long>(stats.frames),
                    static_cast<unsigned long long>(stats.presented),
                    static_cast<unsigned long long>(stats.late),
                    static_cast<unsigned long long>(stats.missed));
        ImGui::Text("CPU %.2f ms, p95 %.2f", stats.cpu.mean * 1e3, stats.cpu.p95 * 1e3);
        ImGui::Text("GPU %.2f ms, p95 %.2f", stats.gpu.mean * 1e3, stats.gpu.p95 * 1e3);
        ImGui::Text("Motion to photon %.2f ms", stats.motionToPhoton.mean * 1e3);
        ImGui::Separator();
        ImGui::Checkbox("Controllers", &m_visuals.controllers);
        ImGui::Checkbox("Hands", &m_visuals.hands);
        ImGui::Checkbox("Lasers", &m_visuals.lasers);
        ImGui::Checkbox("Play area", &m_visuals.playArea);
        for (const float rate : status.refreshRates) {
            ImGui::SameLine();
            if (ImGui::SmallButton(std::format("{:.0f} Hz", rate).c_str())) {
                m_session->SetRefreshRate(rate);
            }
        }
        if (ImGui::Button("Stop")) {
            Stop();
        }
    }

    Viewer& m_viewer;
    Ref<Scene> m_scene;
    xr::Backend m_backend = xr::Backend::Simulated;
    float m_resolutionScale = 0.5f;
    bool m_depthLayer = true;
    xr::BlendMode m_blend = xr::BlendMode::Opaque;
    xr::RigVisuals m_visuals;
    std::shared_ptr<Desk> m_desk = std::make_shared<Desk>();
    std::unique_ptr<xr::Session> m_session;
    std::optional<xr::Rig> m_rig;
    std::shared_ptr<Renderer> m_renderer;
    std::vector<uint64_t> m_handlers;
    std::string m_error;
    bool m_aiming = false;
    bool m_turned = false;
};

}

void AddXrPanel(Viewer& viewer, Ref<Scene> scene) {
    const auto mode = std::make_shared<XrMode>(viewer, std::move(scene));
    viewer.AddPanel("XR", Dock::Right, [mode] { mode->Draw(); }, false);
}

}
