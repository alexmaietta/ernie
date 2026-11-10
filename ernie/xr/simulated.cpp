// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <thread>
#include <vector>

#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/xr/poses.h"
#include "ernie/xr/profiles.h"
#include "ernie/xr/runtime.h"

namespace ernie::xr {
namespace {

uint32_t Bit(Feature feature) {
    return 1u << uint32_t(feature);
}

uint32_t OfferedFeatures(const SimulatedOffers& offers) {
    return (offers.eyeGaze ? Bit(Feature::EyeGaze) : 0) |
           (offers.handTracking ? Bit(Feature::HandTracking) : 0) |
           (offers.refreshRates ? Bit(Feature::DisplayRefreshRate) : 0) |
           (offers.performanceMetrics ? Bit(Feature::PerformanceMetrics) : 0);
}

const char* StateName(SessionState state) {
    switch (state) {
    case SessionState::Idle:
        return "idle";
    case SessionState::Ready:
        return "ready";
    case SessionState::Synchronized:
        return "synchronized";
    case SessionState::Visible:
        return "visible";
    case SessionState::Focused:
        return "focused";
    case SessionState::Stopping:
        return "stopping";
    case SessionState::LossPending:
        return "loss pending";
    case SessionState::Exiting:
        return "exiting";
    default:
        return "unknown";
    }
}

Quatd Ahead() {
    return LookAlong({0, 1, 0}, {0, 0, 1});
}

class SimulatedRuntime final : public Runtime {
  public:
    explicit SimulatedRuntime(const SessionDesc& desc)
        : m_desc(desc.simulated),
          m_depthLayer(desc.depthLayer && desc.simulated.offers.depthLayer) {
        const SimulatedOffers& offers = m_desc.offers;
        const bool offered = desc.blendMode == BlendMode::Opaque ||
                             (desc.blendMode == BlendMode::Additive && offers.additive) ||
                             (desc.blendMode == BlendMode::AlphaBlend && offers.alphaBlend);
        m_blend = offered ? desc.blendMode : BlendMode::Opaque;
        const SimulatedProfile& profile = m_desc.profile;
        if (profile.refreshRates.empty() || profile.resolution.x == 0 ||
            profile.resolution.y == 0) {
            throw Error("a simulated profile needs a resolution and a refresh rate");
        }
        m_rates = offers.refreshRates ? profile.refreshRates
                                      : std::vector<float>{profile.refreshRates.front()};
        float rate = desc.refreshRate > 0 ? desc.refreshRate : profile.refreshRates.front();
        if (std::ranges::find(m_rates, rate) == m_rates.end()) {
            if (offers.refreshRates) {
                throw Error("the {} has no {} Hz rate", profile.name, rate);
            }
            rate = m_rates.front();
        }
        m_rate = rate;
        for (Feature feature : desc.features) {
            m_features |= Bit(feature) & OfferedFeatures(offers);
        }
        m_state = m_desc.skipStartup ? SessionState::Focused : SessionState::Idle;
    }

    void ConfigureDevice(device::DeviceDesc&) override {}

    void Start(device::Device& device) override {
        m_device = Ref<device::Device>(&device);
    }

    void Stop() override {
        for (const Copying& copying : m_copying) {
            copying.done.Wait();
        }
        m_copying.clear();
        m_presented = {};
        m_presentedDepth = {};
        m_device = {};
    }

    bool Running() override {
        const SessionState state = m_state;
        return state != SessionState::Stopping && state != SessionState::LossPending &&
               state != SessionState::Exiting;
    }

    bool SetRefreshRate(float hz) override {
        if (std::ranges::find(m_rates, hz) == m_rates.end()) {
            return false;
        }
        m_rate = hz;
        return true;
    }

    void BeginFrame(Frame& frame, FrameClock& clock) override {
        const int64_t period = int64_t(std::llround(1e9 / m_rate.load()));
        const int64_t last = m_displayTime;
        clock.period = double(period) * 1e-9;
        if (m_desc.realTime) {
            m_displayTime = WaitForVsync(period);
            clock.display = m_start + double(m_displayTime) * 1e-9;
        } else {
            m_displayTime += period;
            clock.display = HostSeconds() + clock.period;
        }
        clock.missed = last > 0 ? uint32_t((m_displayTime - last) / period - 1) : 0;
        frame.displayTime = m_displayTime;
        frame.deltaTime = double(last > 0 ? m_displayTime - last : period) * 1e-9;
        clock.poses = HostSeconds();

        const SimulatedProfile& profile = m_desc.profile;
        const double height = m_desc.userHeight > 0 ? m_desc.userHeight : profile.eyeHeight;
        const double separation = m_desc.ipd > 0 ? m_desc.ipd : profile.eyeSeparation;
        SimulatedInput input;
        input.profile = profile.interactionProfile;
        input.head = {{0, 0, height}, Ahead(), true};
        if (m_desc.placeControllers) {
            for (uint32_t hand = 0; hand < 2; ++hand) {
                Controller& controller = input.controllers[hand];
                controller.active = true;
                controller.pose = {{hand == 0 ? -0.2 : 0.2, 0.3, 0.6 * height}, Ahead(), true};
                controller.aim = controller.pose;
            }
        }
        if (m_desc.script) {
            ERNIE_ZONE("XR/Script");
            m_desc.script(frame.index, double(m_displayTime) * 1e-9, input);
        }
        if (input.skipRefreshes > 0) {
            m_displayTime += int64_t(input.skipRefreshes) * period;
            clock.missed += input.skipRefreshes;
            frame.displayTime = m_displayTime;
            frame.deltaTime += double(input.skipRefreshes) * clock.period;
            if (m_desc.realTime) {
                clock.display = m_start + double(m_displayTime) * 1e-9;
            }
        }
        if (input.lateBy > 0) {
            clock.display = clock.poses - input.lateBy;
        }
        Advance(input);
        frame.state = m_state;
        frame.recentered = input.recenter;
        frame.shouldRender = m_state == SessionState::Visible || m_state == SessionState::Focused;
        frame.head = input.head;
        for (uint32_t eye = 0; eye < 2; ++eye) {
            const double side = eye == 0 ? -0.5 : 0.5;
            View& view = frame.views[eye];
            view.pose = input.head;
            view.pose.position =
                Add(input.head.position, Rotate(input.head.orientation, {side * separation, 0, 0}));
            view.fov = profile.fov[eye];
        }
        if (m_state != SessionState::Focused) {
            return;
        }
        if (frame.Has(Feature::EyeGaze)) {
            frame.gaze = input.gaze;
        }
        frame.controllers = input.controllers;
        const InteractionProfile* interaction = FindProfile(input.profile);
        for (uint32_t hand = 0; hand < 2; ++hand) {
            Controller& controller = frame.controllers[hand];
            if (interaction != nullptr && !input.components[hand].empty()) {
                controller.active = true;
                ApplyComponents(*interaction, hand, input.components[hand], controller);
            }
            if (controller.active) {
                frame.profiles[hand] = input.profile;
            }
        }
        if (frame.Has(Feature::HandTracking)) {
            frame.hands = input.hands;
        }
    }

    void EndFrame(const Frame&, const EyeImages* eyes) override {
        if (eyes == nullptr) {
            return;
        }
        ERNIE_ZONE("XR/Present copy");
        Keep(eyes->color, m_presented, eyes->ready, "xr presented");
        if (!eyes->depth.Empty()) {
            Keep(eyes->depth, m_presentedDepth, eyes->ready, "xr presented depth");
        }
    }

    void RequestExit() override {
        m_exit = true;
    }

    bool DepthLayer() const override {
        return m_depthLayer;
    }

    BlendMode Blend() const override {
        return m_blend;
    }

    Uint2 Recommended() const override {
        return m_desc.profile.resolution;
    }

    uint32_t Features() const override {
        return m_features;
    }

    void FillStatus(SessionStatus& status) const override {
        status.runtime = "Ernie simulated headset";
        status.system = m_desc.profile.name;
        status.state = StateName(m_state);
        status.refreshRate = m_rate.load();
        status.refreshRates = m_rates;
        status.playArea = m_desc.offers.stageBounds ? m_desc.playArea : Float2{0, 0};
    }

    RuntimeMetrics Metrics(const SessionStats& stats) override {
        RuntimeMetrics metrics;
        metrics.available = (m_features & Bit(Feature::PerformanceMetrics)) != 0;
        if (!metrics.available) {
            return metrics;
        }
        for (auto timing = stats.recent.rbegin(); timing != stats.recent.rend(); ++timing) {
            if (timing->gpuDone && timing->rendered) {
                metrics.appCpu = timing->cpu;
                metrics.appGpu = timing->gpu;
                metrics.motionToPhoton = timing->motionToPhoton;
                break;
            }
        }
        metrics.droppedFrames = uint32_t(stats.missed);
        const double period = stats.period > 0 ? stats.period : 1.0 / m_rate.load();
        metrics.gpuUtilization = float(std::min(1.0, metrics.appGpu / period));
        metrics.cpuUtilization = float(std::min(1.0, metrics.appCpu / period));
        return metrics;
    }

    device::Tensor Presented() const override {
        return m_presented;
    }

    device::Tensor PresentedDepth() const override {
        return m_presentedDepth;
    }

  private:
    void Keep(const device::Tensor& source, device::Tensor& kept, const device::Operation& ready,
              const char* label) {
        const std::vector<int64_t> shape(source.Shape().begin(), source.Shape().end());
        if (kept.Empty() || kept.Type() != source.Type() ||
            !std::ranges::equal(kept.Shape(), shape)) {
            kept = m_device->CreateTensor(source.Type(), shape, label);
        }
        std::erase_if(m_copying, [](const Copying& copying) { return copying.done.Ready(); });
        m_copying.push_back({m_device->Copy(source, kept, ready), source});
    }

    struct Copying {
        device::Operation done;
        device::Tensor source;
    };

    void Advance(const SimulatedInput& input) {
        if (m_state == SessionState::Exiting) {
            return;
        }
        if (m_state == SessionState::Stopping || m_state == SessionState::LossPending) {
            m_state = SessionState::Exiting;
            return;
        }
        if (input.loseSession) {
            m_state = SessionState::LossPending;
            return;
        }
        if (m_exit) {
            m_state = SessionState::Stopping;
            return;
        }
        if (m_state < SessionState::Synchronized) {
            m_state = SessionState(uint32_t(m_state.load()) + 1);
            return;
        }
        if (m_state == SessionState::Synchronized && !m_desc.skipStartup && !m_started) {
            m_state = SessionState::Visible;
            return;
        }
        m_started = true;
        m_state = input.visible ? SessionState::Focused : SessionState::Synchronized;
    }

    int64_t WaitForVsync(int64_t period) {
        ERNIE_ZONE("XR/WaitFrame");
        if (m_start == 0) {
            m_start = HostSeconds();
        }
        const int64_t now = int64_t((HostSeconds() - m_start) * 1e9);
        int64_t display = (now + period + period - 1) / period * period;
        display = std::max(display, m_displayTime + period);
        const double wait = m_start + double(display - period) * 1e-9 - HostSeconds();
        if (wait > 0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(wait));
        }
        return display;
    }

    SimulatedDesc m_desc;
    bool m_depthLayer = true;
    BlendMode m_blend = BlendMode::Opaque;
    std::atomic<float> m_rate{0};
    std::vector<float> m_rates;
    uint32_t m_features = 0;
    std::atomic<SessionState> m_state{SessionState::Idle};
    bool m_started = false;
    bool m_exit = false;
    Ref<device::Device> m_device;
    int64_t m_displayTime = 0;
    device::Tensor m_presented;
    device::Tensor m_presentedDepth;
    std::vector<Copying> m_copying;
    double m_start = 0;
};

std::array<Double3, kHandJointCount> HandShape(HandPose pose, bool left) {
    std::array<Double3, kHandJointCount> joints{};
    joints[1] = {0, 0, 0.07};
    const double lengths[4] = {0.035, 0.03, 0.022, 0.02};
    double bend[5] = {0, 0, 0, 0, 0};
    if (pose == HandPose::Fist) {
        bend[0] = 0.7, bend[1] = bend[2] = bend[3] = bend[4] = 1;
    } else if (pose == HandPose::Point) {
        bend[0] = 0.7, bend[2] = bend[3] = bend[4] = 1;
    } else if (pose == HandPose::Pinch) {
        bend[1] = 0.45;
    }
    const double side = left ? -1.0 : 1.0;
    for (uint32_t finger = 0; finger < 5; ++finger) {
        const double x = side * (finger == 0 ? -0.035 : -0.03 + 0.02 * double(finger - 1));
        Double3 at{x, 0, finger == 0 ? 0.02 : -0.03};
        const uint32_t first = finger == 0 ? 2 : 6 + 5 * (finger - 1);
        const uint32_t count = finger == 0 ? 4 : 5;
        double angle = 0;
        for (uint32_t k = 0; k < count; ++k) {
            joints[first + k] = at;
            if (k + 1 < count) {
                angle += bend[finger] * 0.5;
                at = Add(at,
                         Scale({0, -std::sin(angle), -std::cos(angle)}, lengths[std::min(k, 3u)]));
            }
        }
    }
    if (pose == HandPose::Pinch) {
        joints[uint32_t(HandJoint::ThumbTip)] = joints[uint32_t(HandJoint::IndexTip)];
    }
    return joints;
}

}

SimulatedProfile SimulatedProfile::Quest3() {
    SimulatedProfile profile;
    profile.name = "Meta Quest 3 (simulated)";
    profile.resolution = {2064, 2208};
    profile.fov[0] = {-0.9425f, 0.7330f, 0.8203f, -0.8901f};
    profile.fov[1] = {-0.7330f, 0.9425f, 0.8203f, -0.8901f};
    profile.eyeSeparation = 0.063f;
    profile.eyeHeight = 1.6f;
    profile.refreshRates = {90, 72, 120};
    return profile;
}

void SimulatedInput::Press(Side hand, std::string_view component, float value) {
    const uint32_t side = uint32_t(hand);
    const InteractionProfile* interaction = FindProfile(profile);
    if (interaction == nullptr || !HasComponent(*interaction, side, component)) {
        throw Error("the {} profile has no {} {}", profile.empty() ? "simulated" : profile,
                    side == 0 ? "left" : "right", component);
    }
    components[side][std::string(component)] = value;
}

void SimulatedInput::PoseHand(Side hand, HandPose pose) {
    const uint32_t side = uint32_t(hand);
    const Controller& controller = controllers[side];
    Pose base = head;
    if (controller.active && controller.pose.valid) {
        base = controller.pose;
    } else {
        base.position =
            Add(head.position, Rotate(head.orientation, {side == 0 ? -0.15 : 0.15, -0.25, -0.3}));
    }
    const std::array<Double3, kHandJointCount> shape = HandShape(pose, side == 0);
    HandJoints& joints = hands[side];
    joints.active = true;
    for (uint32_t joint = 0; joint < kHandJointCount; ++joint) {
        JointPose& placed = joints.joints[joint];
        placed.pose = {Add(base.position, Rotate(base.orientation, shape[joint])), base.orientation,
                       true};
        placed.radius = joint >= 2 ? 0.008f : 0.02f;
    }
}

std::unique_ptr<Runtime> CreateSimulatedRuntime(const SessionDesc& desc) {
    return std::make_unique<SimulatedRuntime>(desc);
}

}
