// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/xr/session.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include "ernie/render/render_output.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/xr/poses.h"
#include "ernie/xr/rig.h"
#include "ernie/xr/rig_state.h"
#include "ernie/xr/runtime.h"
#include "ernie/xr/stats.h"

namespace ernie::xr {
namespace {

constexpr size_t kFramesInFlight = 2;
constexpr float kReleaseMargin = 0.1f;
constexpr float kFocalLength = 50;

float InputValue(const Controller& controller, Input input) {
    if (!controller.active) {
        return 0;
    }
    const Float2 stick = controller.thumbstick;
    switch (input) {
    case Input::Trigger:
        return controller.trigger;
    case Input::Grip:
        return controller.grip;
    case Input::Primary:
        return controller.buttons.primary ? 1.0f : 0.0f;
    case Input::Secondary:
        return controller.buttons.secondary ? 1.0f : 0.0f;
    case Input::ThumbstickClick:
        return controller.buttons.thumbstick ? 1.0f : 0.0f;
    case Input::Menu:
        return controller.buttons.menu ? 1.0f : 0.0f;
    case Input::ThumbstickUp:
        return std::max(stick.y, 0.0f);
    case Input::ThumbstickDown:
        return std::max(-stick.y, 0.0f);
    case Input::ThumbstickLeft:
        return std::max(-stick.x, 0.0f);
    case Input::ThumbstickRight:
        return std::max(stick.x, 0.0f);
    }
    return 0;
}

std::vector<uint32_t> Hands(HandRole role, Side dominant) {
    const uint32_t main = uint32_t(dominant);
    switch (role) {
    case HandRole::Left:
        return {0};
    case HandRole::Right:
        return {1};
    case HandRole::Dominant:
        return {main};
    case HandRole::Offhand:
        return {1 - main};
    case HandRole::Either:
        return {0, 1};
    }
    return {};
}

Backend ChooseBackend(Backend backend) {
    const char* value = std::getenv("ERNIE_XR");
    const std::string_view chosen = value != nullptr ? value : "";
    if (chosen == "sim" || chosen == "simulated") {
        return Backend::Simulated;
    }
    if (chosen == "openxr") {
        return Backend::OpenXR;
    }
    return backend;
}

void Smooth(Pose& smoothed, const Pose& target, double share) {
    if (!target.valid || !smoothed.valid || share >= 1) {
        smoothed = target;
        return;
    }
    smoothed.position = Add(smoothed.position, Scale({target.position.x - smoothed.position.x,
                                                      target.position.y - smoothed.position.y,
                                                      target.position.z - smoothed.position.z},
                                                     share));
    const Quatd a = smoothed.orientation;
    const Quatd b = target.orientation;
    const double sign = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w < 0 ? -1.0 : 1.0;
    smoothed.orientation =
        Normalize({a.x + share * (sign * b.x - a.x), a.y + share * (sign * b.y - a.y),
                   a.z + share * (sign * b.z - a.z), a.w + share * (sign * b.w - a.w)});
}

}

class SessionImpl {
  public:
    explicit SessionImpl(const SessionDesc& desc) : m_desc(desc) {
        m_desc.backend = ChooseBackend(desc.backend);
        const float scale = m_desc.resolutionScale;
        m_desc.resolutionScale = std::isfinite(scale) ? std::clamp(scale, 0.1f, 2.0f) : 1.0f;
        m_runtime = m_desc.backend == Backend::Simulated ? CreateSimulatedRuntime(m_desc)
                                                         : CreateOpenXrRuntime(m_desc);
    }

    ~SessionImpl() {
        m_stats.reset();
        m_inFlight.clear();
        m_rig.reset();
        m_runtime->Stop();
    }

    Ref<device::Device> CreateDevice(device::DeviceDesc desc) {
        if (m_device) {
            throw Error("the session has its device already");
        }
        m_runtime->ConfigureDevice(desc);
        m_device = device::Device::Create(desc);
        return m_device;
    }

    Uint2 Resolution() const {
        const Uint2 recommended = m_runtime->Recommended();
        const float scale = m_desc.resolutionScale;
        return {std::max(1u, uint32_t(std::lround(float(recommended.x) * scale))),
                std::max(1u, uint32_t(std::lround(float(recommended.y) * scale)))};
    }

    void Adopt(const std::shared_ptr<RigState>& rig) {
        if (!m_started) {
            if (!m_device && m_desc.backend == Backend::Simulated) {
                m_device = Ref<device::Device>(&rig->scene->GetDevice());
            } else if (!m_device) {
                CreateDevice({});
            }
            m_runtime->Start(*m_device);
            m_started = true;
        }
        SessionStatus status;
        m_runtime->FillStatus(status);
        rig->playArea = status.playArea;
        m_rig = rig;
    }

    Frame BeginFrame() {
        ERNIE_ZONE("XR/BeginFrame");
        Frame frame;
        frame.index = m_begun++;
        frame.features = m_runtime->Features();
        FrameClock clock;
        const double start = HostSeconds();
        if (m_desc.backend == Backend::Simulated) {
            ERNIE_ZONE("XR/Wait GPU");
            while (m_inFlight.size() >= kFramesInFlight ||
                   (!m_inFlight.empty() && m_inFlight.front().Ready())) {
                m_inFlight.front().Wait();
                m_inFlight.pop_front();
            }
        }
        m_runtime->BeginFrame(frame, clock);
        frame.dominantHand = m_desc.dominantHand;
        FilterPoses(frame);
        ReadActions(frame);
        m_stats->Begin(frame, clock, (clock.poses > 0 ? clock.poses : HostSeconds()) - start);
        return frame;
    }

    void EndFrame(const Frame& frame, const RenderOutput& output) {
        ERNIE_ZONE("XR/EndFrame");
        if (!m_rig) {
            throw Error("EndFrame needs a rig");
        }
        if (!frame.shouldRender) {
            m_stats->End(frame, nullptr);
            m_runtime->EndFrame(frame, nullptr);
            m_stats->SetMetrics(m_runtime->Metrics(m_stats->Stats()));
            return;
        }
        if (!output.Contains(m_rig->product)) {
            throw Error("the output has no frame of the rig's product {}", m_rig->product);
        }
        const ProductOutput& product = output[m_rig->product];
        if (product.frames.empty()) {
            throw Error("the rig's product {} rendered no frame", m_rig->product);
        }
        const FrameOutput& eyeFrame = product.frames.back();
        device::Tensor color = eyeFrame.RenderVar(m_rig->colorVar).tensor;
        device::Operation ready = output.Done();
        if (!eyeFrame.overlay.Empty()) {
            color = CompositeOverlay(color, eyeFrame.overlay, ready, &ready);
        }
        const Span<const int64_t> shape = color.Shape();
        if (shape.size() != 4 || shape[0] != 2) {
            throw Error("the rig's color isn't two eyes' images");
        }
        m_stats->End(frame, &ready);
        m_inFlight.push_back(ready);
        const EyeImages eyes{color,
                             m_rig->depthVar.empty() ? device::Tensor()
                                                     : eyeFrame.RenderVar(m_rig->depthVar).tensor,
                             ready};
        m_runtime->EndFrame(frame, &eyes);
        ++m_presented;
        m_stats->SetMetrics(m_runtime->Metrics(m_stats->Stats()));
    }

    SessionStatus Status() const {
        SessionStatus status;
        status.backend = m_desc.backend;
        m_runtime->FillStatus(status);
        status.resolution = m_runtime->Recommended();
        status.features = m_runtime->Features();
        status.framesBegun = m_begun;
        status.framesPresented = m_presented;
        status.depthLayer = m_runtime->DepthLayer();
        status.blendMode = m_runtime->Blend();
        return status;
    }

    void PushBindings(std::string_view name, const std::vector<ActionBinding>& bindings) {
        std::lock_guard lock(m_mutex);
        for (auto& [layer, bound] : m_layers) {
            if (layer == name) {
                bound = bindings;
                return;
            }
        }
        m_layers.emplace_back(std::string(name), bindings);
    }

    void PopBindings(std::string_view name) {
        std::lock_guard lock(m_mutex);
        std::erase_if(m_layers, [&](const auto& layer) { return layer.first == name; });
    }

    std::vector<std::string> BindingLayers() const {
        std::lock_guard lock(m_mutex);
        std::vector<std::string> names;
        for (const auto& layer : m_layers) {
            names.push_back(layer.first);
        }
        return names;
    }

    void SetPoseFilter(Side hand, const PoseFilter& filter) {
        std::lock_guard lock(m_mutex);
        m_filters[uint32_t(hand)] = filter;
    }

    PoseFilter GetPoseFilter(Side hand) const {
        std::lock_guard lock(m_mutex);
        return m_filters[uint32_t(hand)];
    }

    SessionDesc m_desc;
    std::unique_ptr<Runtime> m_runtime;
    Ref<device::Device> m_device;
    std::unique_ptr<FrameStats> m_stats = std::make_unique<FrameStats>();

  private:
    void FilterPoses(Frame& frame) {
        std::array<PoseFilter, 2> filters;
        {
            std::lock_guard lock(m_mutex);
            filters = m_filters;
        }
        for (uint32_t hand = 0; hand < 2; ++hand) {
            Controller& controller = frame.controllers[hand];
            controller.rawPose = controller.pose;
            controller.rawAim = controller.aim;
            if (!controller.active) {
                m_smoothed[hand] = {};
                continue;
            }
            const PoseFilter& filter = filters[hand];
            const double share =
                filter.smoothing > 0 ? 1 - std::exp(-frame.deltaTime / filter.smoothing) : 1.0;
            Smooth(m_smoothed[hand][0], Offset(controller.pose, filter.offset), share);
            Smooth(m_smoothed[hand][1], Offset(controller.aim, filter.offset), share);
            controller.pose = m_smoothed[hand][0];
            controller.aim = m_smoothed[hand][1];
        }
    }

    void ReadActions(Frame& frame) {
        std::vector<std::pair<std::string, std::vector<ActionBinding>>> layers;
        {
            std::lock_guard lock(m_mutex);
            layers = m_layers;
        }

        struct Reading {
            std::string action;
            float value = 0;
            bool above = false;
            bool kept = false;
            Side hand = Side::Right;
        };

        std::vector<std::string> claimed;
        std::vector<Reading> readings;
        for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) {
            std::vector<std::string> named;
            for (const ActionBinding& binding : layer->second) {
                if (std::ranges::find(claimed, binding.action) != claimed.end()) {
                    continue;
                }
                named.push_back(binding.action);
                auto found = std::ranges::find(readings, binding.action, &Reading::action);
                if (found == readings.end()) {
                    readings.push_back({.action = binding.action});
                    found = readings.end() - 1;
                }
                for (const uint32_t hand : Hands(binding.hand, frame.dominantHand)) {
                    if (!binding.profile.empty() && frame.profiles[hand] != binding.profile) {
                        continue;
                    }
                    const float value = InputValue(frame.controllers[hand], binding.input);
                    const bool above = value >= binding.threshold;
                    if (value > found->value || (above && !found->above)) {
                        found->value = std::max(found->value, value);
                        found->hand = Side(hand);
                    }
                    found->above = found->above || above;
                    found->kept = found->kept || value >= binding.threshold - kReleaseMargin;
                }
            }
            claimed.insert(claimed.end(), named.begin(), named.end());
        }
        for (const Reading& reading : readings) {
            ActionState state;
            state.action = reading.action;
            state.value = std::min(reading.value, 1.0f);
            const auto previous = m_held.find(reading.action);
            const bool wasHeld = previous != m_held.end() && previous->second.first;
            state.held = wasHeld ? reading.kept : reading.above;
            state.pressed = state.held && !wasHeld;
            state.released = !state.held && wasHeld;
            state.hand =
                state.held || previous == m_held.end() ? reading.hand : previous->second.second;
            if (state.pressed || state.released) {
                frame.events.push_back({reading.action, state.hand, state.pressed});
            }
            m_held[reading.action] = {state.held, state.hand};
            frame.actions.push_back(state);
        }
        for (auto& [action, held] : m_held) {
            if (held.first &&
                std::ranges::find(readings, action, &Reading::action) == readings.end()) {
                held.first = false;
                frame.events.push_back({action, held.second, false});
            }
        }
    }

    bool m_started = false;
    std::shared_ptr<RigState> m_rig;
    std::deque<device::Operation> m_inFlight;
    uint64_t m_begun = 0;
    uint64_t m_presented = 0;
    mutable std::mutex m_mutex;
    std::vector<std::pair<std::string, std::vector<ActionBinding>>> m_layers;
    std::map<std::string, std::pair<bool, Side>, std::less<>> m_held;
    std::array<PoseFilter, 2> m_filters;
    std::array<std::array<Pose, 2>, 2> m_smoothed;
};

Session::Session(const SessionDesc& desc) : m_impl(std::make_unique<SessionImpl>(desc)) {}

Session::~Session() = default;

Ref<device::Device> Session::CreateDevice(device::DeviceDesc desc) {
    return m_impl->CreateDevice(std::move(desc));
}

Ref<device::Device> Session::GetDevice() const {
    return m_impl->m_device;
}

bool Session::Running() {
    return m_impl->m_runtime->Running();
}

Frame Session::BeginFrame() {
    return m_impl->BeginFrame();
}

void Session::EndFrame(const Frame& frame, const RenderOutput& output) {
    m_impl->EndFrame(frame, output);
}

void Session::RequestExit() {
    m_impl->m_runtime->RequestExit();
}

bool Session::SetRefreshRate(float hz) {
    return m_impl->m_runtime->SetRefreshRate(hz);
}

void Session::PushBindings(std::string_view name, const std::vector<ActionBinding>& bindings) {
    m_impl->PushBindings(name, bindings);
}

void Session::PopBindings(std::string_view name) {
    m_impl->PopBindings(name);
}

std::vector<std::string> Session::BindingLayers() const {
    return m_impl->BindingLayers();
}

void Session::SetPoseFilter(Side hand, const PoseFilter& filter) {
    m_impl->SetPoseFilter(hand, filter);
}

PoseFilter Session::GetPoseFilter(Side hand) const {
    return m_impl->GetPoseFilter(hand);
}

device::Tensor Session::Presented() const {
    return m_impl->m_runtime->Presented();
}

device::Tensor Session::PresentedDepth() const {
    return m_impl->m_runtime->PresentedDepth();
}

SessionStatus Session::Status() const {
    return m_impl->Status();
}

SessionStats Session::Stats() const {
    return m_impl->m_stats->Stats();
}

void Session::WriteTimings(const std::filesystem::path& path) const {
    m_impl->m_stats->Write(path);
}

const SessionDesc& Session::GetDesc() const {
    return m_impl->m_desc;
}

Rig::Rig(Ref<Scene> scene, Session& session, const RigDesc& desc)
    : m_state(std::make_shared<RigState>()) {
    SessionImpl& impl = *session.m_impl;
    RigState& state = *m_state;
    const std::string& path = desc.path;
    const Float2 planes{impl.m_desc.nearPlane, impl.m_desc.farPlane};
    state.scene = scene;
    state.recommended = impl.Resolution();
    state.root = scene->Define<Xform>(path);
    for (std::optional<Prim> parent = state.root.Parent(); parent; parent = parent->Parent()) {
        if (parent->IsA<Xformable>()) {
            state.parentPath = parent->Path();
            break;
        }
    }
    state.head = scene->Define<Xform>(path + "/Head");
    state.leftEye = scene->Define<Camera>(path + "/LeftEye",
                                          {.focalLength = kFocalLength, .clippingRange = planes});
    state.rightEye = scene->Define<Camera>(path + "/RightEye",
                                           {.focalLength = kFocalLength, .clippingRange = planes});
    state.leftHand = scene->Define<Xform>(path + "/LeftHand");
    state.rightHand = scene->Define<Xform>(path + "/RightHand");
    scene->Define<Scope>(path + "/Render");
    state.colorVar = path + "/Render/LdrColor";
    scene->Define<RenderVar>(state.colorVar, {.sourceName = "LdrColor"});
    state.vars = {state.colorVar};
    if (impl.m_runtime->DepthLayer()) {
        state.depthVar = path + "/Render/NormalizedDepth";
        scene->Define<RenderVar>(state.depthVar, {.sourceName = "NormalizedDepth"});
        state.vars.push_back(state.depthVar);
    }
    for (const std::string& source : desc.extraVars) {
        const std::string var = path + "/Render/" + source;
        scene->Define<RenderVar>(var, {.sourceName = source});
        state.vars.push_back(var);
    }
    state.product = path + "/Render/Eyes";
    scene->Define<RenderProduct>(
        state.product,
        {.camera = {path + "/LeftEye", path + "/RightEye"},
         .resolution = state.recommended,
         .aspectRatioConformPolicy = AspectRatioConformPolicy::AdjustPixelAspectRatio,
         .orderedVars = state.vars,
         .overlays = desc.overlays});
    impl.Adopt(m_state);
}

}
