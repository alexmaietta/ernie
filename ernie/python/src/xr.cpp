// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "bindings.h"
#include "ernie/xr/rig.h"
#include "ernie/xr/session.h"
#include "values.h"

namespace nb = nanobind;

namespace nanobind::detail {

template <class T, size_t N, class Item> struct ValueCaster {
    NB_TYPE_CASTER(T, const_name("tuple"))

    bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
        if (!PySequence_Check(src.ptr()) || PySequence_Size(src.ptr()) != Py_ssize_t(N)) {
            return false;
        }
        Item items[N];
        for (size_t i = 0; i < N; ++i) {
            object item = steal(PySequence_GetItem(src.ptr(), Py_ssize_t(i)));
            make_caster<Item> caster;
            if (!item.is_valid() || !caster.from_python(item, cast_flags::convert, nullptr)) {
                PyErr_Clear();
                return false;
            }
            items[i] = caster.operator cast_t<Item>();
        }
        value = Make(items);
        return true;
    }

    static handle from_cpp(const T& v, rv_policy, cleanup_list*) noexcept {
        Item items[N];
        Split(v, items);
        object tuple = steal(PyTuple_New(Py_ssize_t(N)));
        for (size_t i = 0; i < N; ++i) {
            PyTuple_SET_ITEM(tuple.ptr(), Py_ssize_t(i), cast(items[i]).release().ptr());
        }
        return tuple.release();
    }

    static T Make(const Item (&items)[N]) {
        if constexpr (N == 2) {
            return T{items[0], items[1]};
        } else if constexpr (N == 3) {
            return T{items[0], items[1], items[2]};
        } else {
            return T{items[0], items[1], items[2], items[3]};
        }
    }

    static void Split(const T& v, Item (&items)[N]) {
        if constexpr (N == 2) {
            items[0] = v.x, items[1] = v.y;
        } else if constexpr (N == 3) {
            items[0] = v.x, items[1] = v.y, items[2] = v.z;
        } else {
            items[0] = v.x, items[1] = v.y, items[2] = v.z, items[3] = v.w;
        }
    }
};

template <> struct type_caster<ernie::Float2> : ValueCaster<ernie::Float2, 2, float> {};

template <> struct type_caster<ernie::Uint2> : ValueCaster<ernie::Uint2, 2, uint32_t> {};

template <> struct type_caster<ernie::Float3> : ValueCaster<ernie::Float3, 3, float> {};

template <> struct type_caster<ernie::Double3> : ValueCaster<ernie::Double3, 3, double> {};

template <> struct type_caster<ernie::Float4> : ValueCaster<ernie::Float4, 4, float> {};

template <> struct type_caster<ernie::Quatd> : ValueCaster<ernie::Quatd, 4, double> {};

}

namespace ernie::xr {
namespace {

using python::ScenePrim;

ScenePrim Wrap(const Prim& prim) {
    return ScenePrim{Ref<Scene>(&prim.GetScene()), prim};
}

std::vector<Prim> Unwrap(const std::vector<ScenePrim>& prims) {
    std::vector<Prim> unwrapped;
    for (const ScenePrim& prim : prims) {
        unwrapped.push_back(prim.prim);
    }
    return unwrapped;
}

std::vector<ScenePrim> Wrap(const std::vector<Prim>& prims) {
    std::vector<ScenePrim> wrapped;
    for (const Prim& prim : prims) {
        wrapped.push_back(Wrap(prim));
    }
    return wrapped;
}

void BindFrame(nb::module_& module) {
    nb::enum_<Side>(module, "Side").value("LEFT", Side::Left).value("RIGHT", Side::Right);
    nb::enum_<Feature>(module, "Feature")
        .value("EYE_GAZE", Feature::EyeGaze)
        .value("HAND_TRACKING", Feature::HandTracking)
        .value("DISPLAY_REFRESH_RATE", Feature::DisplayRefreshRate)
        .value("PERFORMANCE_METRICS", Feature::PerformanceMetrics);
    nb::enum_<SessionState>(module, "SessionState")
        .value("UNKNOWN", SessionState::Unknown)
        .value("IDLE", SessionState::Idle)
        .value("READY", SessionState::Ready)
        .value("SYNCHRONIZED", SessionState::Synchronized)
        .value("VISIBLE", SessionState::Visible)
        .value("FOCUSED", SessionState::Focused)
        .value("STOPPING", SessionState::Stopping)
        .value("LOSS_PENDING", SessionState::LossPending)
        .value("EXITING", SessionState::Exiting);
    nb::enum_<HandJoint> joints(module, "HandJoint");
    const char* names[] = {"PALM",
                           "WRIST",
                           "THUMB_METACARPAL",
                           "THUMB_PROXIMAL",
                           "THUMB_DISTAL",
                           "THUMB_TIP",
                           "INDEX_METACARPAL",
                           "INDEX_PROXIMAL",
                           "INDEX_INTERMEDIATE",
                           "INDEX_DISTAL",
                           "INDEX_TIP",
                           "MIDDLE_METACARPAL",
                           "MIDDLE_PROXIMAL",
                           "MIDDLE_INTERMEDIATE",
                           "MIDDLE_DISTAL",
                           "MIDDLE_TIP",
                           "RING_METACARPAL",
                           "RING_PROXIMAL",
                           "RING_INTERMEDIATE",
                           "RING_DISTAL",
                           "RING_TIP",
                           "LITTLE_METACARPAL",
                           "LITTLE_PROXIMAL",
                           "LITTLE_INTERMEDIATE",
                           "LITTLE_DISTAL",
                           "LITTLE_TIP"};
    for (uint32_t joint = 0; joint < kHandJointCount; ++joint) {
        joints.value(names[joint], HandJoint(joint));
    }

    nb::class_<Pose>(module, "Pose")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Pose* pose, Double3 position, Quatd orientation, bool valid) {
                new (pose) Pose{position, orientation, valid};
            },
            nb::arg("position"), nb::arg("orientation") = Quatd{}, nb::arg("valid") = true)
        .def_rw("position", &Pose::position)
        .def_rw("orientation", &Pose::orientation)
        .def_rw("valid", &Pose::valid);
    nb::class_<Fov>(module, "Fov")
        .def(nb::init<>())
        .def_rw("left", &Fov::left)
        .def_rw("right", &Fov::right)
        .def_rw("up", &Fov::up)
        .def_rw("down", &Fov::down);
    nb::class_<View>(module, "View")
        .def(nb::init<>())
        .def_rw("pose", &View::pose)
        .def_rw("fov", &View::fov);
    nb::class_<Buttons>(module, "Buttons")
        .def(nb::init<>())
        .def_rw("primary", &Buttons::primary)
        .def_rw("secondary", &Buttons::secondary)
        .def_rw("thumbstick", &Buttons::thumbstick)
        .def_rw("menu", &Buttons::menu);
    nb::class_<Controller>(module, "Controller")
        .def(nb::init<>())
        .def_rw("active", &Controller::active)
        .def_rw("pose", &Controller::pose)
        .def_rw("aim", &Controller::aim)
        .def_rw("raw_pose", &Controller::rawPose)
        .def_rw("raw_aim", &Controller::rawAim)
        .def_rw("trigger", &Controller::trigger)
        .def_rw("grip", &Controller::grip)
        .def_rw("thumbstick", &Controller::thumbstick)
        .def_rw("buttons", &Controller::buttons);
    nb::class_<JointPose>(module, "JointPose")
        .def(nb::init<>())
        .def_rw("pose", &JointPose::pose)
        .def_rw("radius", &JointPose::radius);
    nb::class_<HandJoints>(module, "HandJoints")
        .def(nb::init<>())
        .def_rw("active", &HandJoints::active)
        .def_rw("joints", &HandJoints::joints)
        .def("__getitem__", [](const HandJoints& hand, HandJoint joint) { return hand[joint]; });
    nb::class_<HapticPulse>(module, "HapticPulse")
        .def(nb::init<>())
        .def_rw("hand", &HapticPulse::hand)
        .def_rw("amplitude", &HapticPulse::amplitude)
        .def_rw("duration", &HapticPulse::duration)
        .def_rw("frequency", &HapticPulse::frequency);
    nb::class_<ActionState>(module, "ActionState")
        .def_ro("action", &ActionState::action)
        .def_ro("value", &ActionState::value)
        .def_ro("held", &ActionState::held)
        .def_ro("pressed", &ActionState::pressed)
        .def_ro("released", &ActionState::released)
        .def_ro("hand", &ActionState::hand);
    nb::class_<ActionEvent>(module, "ActionEvent")
        .def_ro("action", &ActionEvent::action)
        .def_ro("hand", &ActionEvent::hand)
        .def_ro("pressed", &ActionEvent::pressed);
    nb::class_<Frame>(module, "Frame")
        .def_ro("index", &Frame::index)
        .def_ro("display_time", &Frame::displayTime)
        .def_ro("delta_time", &Frame::deltaTime)
        .def_ro("should_render", &Frame::shouldRender)
        .def_ro("views", &Frame::views)
        .def_ro("head", &Frame::head)
        .def_ro("gaze", &Frame::gaze)
        .def_ro("controllers", &Frame::controllers)
        .def_ro("hands", &Frame::hands)
        .def_ro("actions", &Frame::actions)
        .def_ro("events", &Frame::events)
        .def_ro("profiles", &Frame::profiles)
        .def_ro("dominant_hand", &Frame::dominantHand)
        .def_ro("state", &Frame::state)
        .def_ro("recentered", &Frame::recentered)
        .def("eye", &Frame::Eye, nb::arg("eye"))
        .def("controller", &Frame::Controller, nb::arg("hand"))
        .def("hand", &Frame::Hand, nb::arg("hand"))
        .def("haptics", &Frame::Haptics, nb::arg("hand"), nb::arg("amplitude"),
             nb::arg("duration") = 0.0, nb::arg("frequency") = 0.0f)
        .def("pressed", &Frame::Pressed, nb::arg("action"))
        .def("released", &Frame::Released, nb::arg("action"))
        .def("held", &Frame::Held, nb::arg("action"))
        .def("value", &Frame::Value, nb::arg("action"))
        .def("has", &Frame::Has, nb::arg("feature"));
}

void BindSession(nb::module_& module) {
    nb::enum_<Backend>(module, "Backend")
        .value("OPENXR", Backend::OpenXR)
        .value("SIMULATED", Backend::Simulated);
    nb::enum_<BlendMode>(module, "BlendMode")
        .value("OPAQUE", BlendMode::Opaque)
        .value("ADDITIVE", BlendMode::Additive);
    nb::enum_<ReferenceSpace>(module, "ReferenceSpace")
        .value("STAGE", ReferenceSpace::Stage)
        .value("LOCAL", ReferenceSpace::Local);
    nb::enum_<Input>(module, "Input")
        .value("TRIGGER", Input::Trigger)
        .value("GRIP", Input::Grip)
        .value("PRIMARY", Input::Primary)
        .value("SECONDARY", Input::Secondary)
        .value("THUMBSTICK_CLICK", Input::ThumbstickClick)
        .value("MENU", Input::Menu)
        .value("THUMBSTICK_UP", Input::ThumbstickUp)
        .value("THUMBSTICK_DOWN", Input::ThumbstickDown)
        .value("THUMBSTICK_LEFT", Input::ThumbstickLeft)
        .value("THUMBSTICK_RIGHT", Input::ThumbstickRight);
    nb::enum_<HandRole>(module, "HandRole")
        .value("LEFT", HandRole::Left)
        .value("RIGHT", HandRole::Right)
        .value("DOMINANT", HandRole::Dominant)
        .value("OFFHAND", HandRole::Offhand)
        .value("EITHER", HandRole::Either);
    nb::enum_<HandPose>(module, "HandPose")
        .value("OPEN", HandPose::Open)
        .value("PINCH", HandPose::Pinch)
        .value("POINT", HandPose::Point)
        .value("FIST", HandPose::Fist);

    nb::class_<ActionBinding>(module, "ActionBinding")
        .def(
            "__init__",
            [](ActionBinding* binding, std::string action, Input input, HandRole hand,
               float threshold, std::string profile) {
                new (binding)
                    ActionBinding{std::move(action), input, hand, threshold, std::move(profile)};
            },
            nb::arg("action"), nb::arg("input") = Input::Trigger,
            nb::arg("hand") = HandRole::Dominant, nb::arg("threshold") = 0.5f,
            nb::arg("profile") = "")
        .def_rw("action", &ActionBinding::action)
        .def_rw("input", &ActionBinding::input)
        .def_rw("hand", &ActionBinding::hand)
        .def_rw("threshold", &ActionBinding::threshold)
        .def_rw("profile", &ActionBinding::profile);
    nb::class_<PoseFilter>(module, "PoseFilter")
        .def(nb::init<>())
        .def_rw("offset", &PoseFilter::offset)
        .def_rw("smoothing", &PoseFilter::smoothing);
    nb::class_<SimulatedProfile>(module, "SimulatedProfile")
        .def(nb::init<>())
        .def_static("quest3", &SimulatedProfile::Quest3)
        .def_rw("name", &SimulatedProfile::name)
        .def_rw("interaction_profile", &SimulatedProfile::interactionProfile)
        .def_rw("resolution", &SimulatedProfile::resolution)
        .def_rw("fov", &SimulatedProfile::fov)
        .def_rw("eye_separation", &SimulatedProfile::eyeSeparation)
        .def_rw("eye_height", &SimulatedProfile::eyeHeight)
        .def_rw("refresh_rates", &SimulatedProfile::refreshRates);
    nb::class_<SimulatedInput>(module, "SimulatedInput")
        .def_rw("head", &SimulatedInput::head)
        .def_rw("gaze", &SimulatedInput::gaze)
        .def_rw("controllers", &SimulatedInput::controllers)
        .def_rw("hands", &SimulatedInput::hands)
        .def_rw("visible", &SimulatedInput::visible)
        .def_rw("profile", &SimulatedInput::profile)
        .def_rw("lose_session", &SimulatedInput::loseSession)
        .def_rw("recenter", &SimulatedInput::recenter)
        .def_rw("skip_refreshes", &SimulatedInput::skipRefreshes)
        .def_rw("late_by", &SimulatedInput::lateBy)
        .def("press", &SimulatedInput::Press, nb::arg("hand"), nb::arg("component"),
             nb::arg("value") = 1.0f)
        .def("pose_hand", &SimulatedInput::PoseHand, nb::arg("hand"), nb::arg("pose"));
    nb::class_<SimulatedOffers>(module, "SimulatedOffers")
        .def(nb::init<>())
        .def_rw("stage_bounds", &SimulatedOffers::stageBounds)
        .def_rw("refresh_rates", &SimulatedOffers::refreshRates)
        .def_rw("depth_layer", &SimulatedOffers::depthLayer)
        .def_rw("additive", &SimulatedOffers::additive)
        .def_rw("performance_metrics", &SimulatedOffers::performanceMetrics)
        .def_rw("hand_tracking", &SimulatedOffers::handTracking)
        .def_rw("eye_gaze", &SimulatedOffers::eyeGaze);
    nb::class_<SimulatedDesc>(module, "SimulatedDesc")
        .def(nb::init<>())
        .def_rw("profile", &SimulatedDesc::profile)
        .def_rw("real_time", &SimulatedDesc::realTime)
        .def_prop_rw(
            "script", [](const SimulatedDesc& desc) { return desc.script; },
            [](SimulatedDesc& desc, nb::object script) {
                if (script.is_none()) {
                    desc.script = nullptr;
                    return;
                }
                auto held = std::make_shared<nb::object>(std::move(script));
                desc.script = [held](uint64_t frame, double time, SimulatedInput& input) {
                    nb::gil_scoped_acquire acquire;
                    (*held)(frame, time, nb::cast(&input, nb::rv_policy::reference));
                };
            })
        .def_rw("play_area", &SimulatedDesc::playArea)
        .def_rw("offers", &SimulatedDesc::offers)
        .def_rw("skip_startup", &SimulatedDesc::skipStartup)
        .def_rw("user_height", &SimulatedDesc::userHeight)
        .def_rw("ipd", &SimulatedDesc::ipd)
        .def_rw("place_controllers", &SimulatedDesc::placeControllers);
    nb::class_<SessionDesc>(module, "SessionDesc")
        .def(nb::init<>())
        .def_rw("application", &SessionDesc::application)
        .def_rw("backend", &SessionDesc::backend)
        .def_rw("reference_space", &SessionDesc::referenceSpace)
        .def_rw("depth_layer", &SessionDesc::depthLayer)
        .def_rw("near_plane", &SessionDesc::nearPlane)
        .def_rw("far_plane", &SessionDesc::farPlane)
        .def_rw("blend_mode", &SessionDesc::blendMode)
        .def_rw("refresh_rate", &SessionDesc::refreshRate)
        .def_rw("resolution_scale", &SessionDesc::resolutionScale)
        .def_rw("features", &SessionDesc::features)
        .def_rw("dominant_hand", &SessionDesc::dominantHand)
        .def_rw("simulated", &SessionDesc::simulated);
    nb::class_<SessionStatus>(module, "SessionStatus")
        .def_ro("backend", &SessionStatus::backend)
        .def_ro("runtime", &SessionStatus::runtime)
        .def_ro("system", &SessionStatus::system)
        .def_ro("state", &SessionStatus::state)
        .def_ro("resolution", &SessionStatus::resolution)
        .def_ro("refresh_rate", &SessionStatus::refreshRate)
        .def_ro("refresh_rates", &SessionStatus::refreshRates)
        .def_ro("features", &SessionStatus::features)
        .def_ro("frames_begun", &SessionStatus::framesBegun)
        .def_ro("frames_presented", &SessionStatus::framesPresented)
        .def_ro("play_area", &SessionStatus::playArea)
        .def_ro("depth_layer", &SessionStatus::depthLayer)
        .def_ro("blend_mode", &SessionStatus::blendMode);
    nb::class_<FrameTiming>(module, "FrameTiming")
        .def_ro("index", &FrameTiming::index)
        .def_ro("display_time", &FrameTiming::displayTime)
        .def_ro("wait", &FrameTiming::wait)
        .def_ro("cpu", &FrameTiming::cpu)
        .def_ro("gpu", &FrameTiming::gpu)
        .def_ro("pose_to_submit", &FrameTiming::poseToSubmit)
        .def_ro("motion_to_photon", &FrameTiming::motionToPhoton)
        .def_ro("lateness", &FrameTiming::lateness)
        .def_ro("missed", &FrameTiming::missed)
        .def_ro("rendered", &FrameTiming::rendered)
        .def_ro("late", &FrameTiming::late)
        .def_ro("gpu_done", &FrameTiming::gpuDone);
    nb::class_<TimingSummary>(module, "TimingSummary")
        .def_ro("mean", &TimingSummary::mean)
        .def_ro("p95", &TimingSummary::p95)
        .def_ro("max", &TimingSummary::max);
    nb::class_<RuntimeMetrics>(module, "RuntimeMetrics")
        .def_ro("available", &RuntimeMetrics::available)
        .def_ro("app_cpu", &RuntimeMetrics::appCpu)
        .def_ro("app_gpu", &RuntimeMetrics::appGpu)
        .def_ro("compositor_cpu", &RuntimeMetrics::compositorCpu)
        .def_ro("compositor_gpu", &RuntimeMetrics::compositorGpu)
        .def_ro("motion_to_photon", &RuntimeMetrics::motionToPhoton)
        .def_ro("dropped_frames", &RuntimeMetrics::droppedFrames)
        .def_ro("cpu_utilization", &RuntimeMetrics::cpuUtilization)
        .def_ro("gpu_utilization", &RuntimeMetrics::gpuUtilization);
    nb::class_<SessionStats>(module, "SessionStats")
        .def_ro("period", &SessionStats::period)
        .def_ro("estimated", &SessionStats::estimated)
        .def_ro("frames", &SessionStats::frames)
        .def_ro("presented", &SessionStats::presented)
        .def_ro("not_rendered", &SessionStats::notRendered)
        .def_ro("late", &SessionStats::late)
        .def_ro("missed", &SessionStats::missed)
        .def_ro("wait", &SessionStats::wait)
        .def_ro("cpu", &SessionStats::cpu)
        .def_ro("gpu", &SessionStats::gpu)
        .def_ro("pose_to_submit", &SessionStats::poseToSubmit)
        .def_ro("motion_to_photon", &SessionStats::motionToPhoton)
        .def_ro("recent", &SessionStats::recent)
        .def_ro("runtime_metrics", &SessionStats::runtimeMetrics);

    nb::class_<Session>(module, "Session")
        .def(nb::init<const SessionDesc&>(), nb::arg("desc") = SessionDesc{},
             nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_device",
            [](Session& session, std::string adapter, bool validation) {
                return session.CreateDevice(
                    {.adapter = std::move(adapter), .validation = validation});
            },
            nb::kw_only(), nb::arg("adapter") = "", nb::arg("validation") = false,
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("device", &Session::GetDevice)
        .def_prop_ro("running", &Session::Running, nb::call_guard<nb::gil_scoped_release>())
        .def("begin_frame", &Session::BeginFrame, nb::call_guard<nb::gil_scoped_release>())
        .def("end_frame", &Session::EndFrame, nb::arg("frame"), nb::arg("output"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("request_exit", &Session::RequestExit)
        .def("set_refresh_rate", &Session::SetRefreshRate, nb::arg("hz"))
        .def("push_bindings", &Session::PushBindings, nb::arg("name"), nb::arg("bindings"))
        .def("pop_bindings", &Session::PopBindings, nb::arg("name"))
        .def_prop_ro("binding_layers", &Session::BindingLayers)
        .def("set_pose_filter", &Session::SetPoseFilter, nb::arg("hand"), nb::arg("filter"))
        .def("pose_filter", &Session::GetPoseFilter, nb::arg("hand"))
        .def_prop_ro("presented", &Session::Presented)
        .def_prop_ro("presented_depth", &Session::PresentedDepth)
        .def_prop_ro("status", &Session::Status)
        .def_prop_ro("stats", &Session::Stats)
        .def("write_timings", &Session::WriteTimings, nb::arg("path"));
}

void BindRig(nb::module_& module) {
    nb::enum_<MoveMode>(module, "MoveMode")
        .value("SMOOTH", MoveMode::Smooth)
        .value("GROUNDED", MoveMode::Grounded)
        .value("FLY", MoveMode::Fly);
    nb::enum_<Anchor>(module, "Anchor")
        .value("ORIGIN", Anchor::Origin)
        .value("CAMERA", Anchor::Camera)
        .value("PRIM", Anchor::Prim)
        .value("FOLLOW", Anchor::Follow)
        .value("FOLLOW_SMOOTHED", Anchor::FollowSmoothed);

    nb::class_<RigDesc>(module, "RigDesc")
        .def(nb::init<>())
        .def_rw("path", &RigDesc::path)
        .def_rw("extra_vars", &RigDesc::extraVars)
        .def_rw("overlays", &RigDesc::overlays);
    nb::class_<RigVisuals>(module, "RigVisuals")
        .def(nb::init<>())
        .def_rw("controllers", &RigVisuals::controllers)
        .def_rw("hands", &RigVisuals::hands)
        .def_rw("lasers", &RigVisuals::lasers)
        .def_rw("teleport", &RigVisuals::teleport)
        .def_rw("play_area", &RigVisuals::playArea)
        .def_rw("color", &RigVisuals::color)
        .def_rw("invalid_color", &RigVisuals::invalidColor)
        .def_rw("layer", &RigVisuals::layer);
    nb::class_<PointerDesc>(module, "PointerDesc")
        .def(nb::init<>())
        .def_rw("max_distance", &PointerDesc::maxDistance)
        .def_rw("arc_speed", &PointerDesc::arcSpeed)
        .def_rw("gravity", &PointerDesc::gravity)
        .def_rw("arc_segments", &PointerDesc::arcSegments)
        .def_rw("arc_time", &PointerDesc::arcTime)
        .def_rw("max_slope_degrees", &PointerDesc::maxSlopeDegrees)
        .def_rw("wait", &PointerDesc::wait)
        .def_prop_rw(
            "ignore", [](const PointerDesc& desc) { return Wrap(desc.ignore); },
            [](PointerDesc& desc, const std::vector<ScenePrim>& prims) {
                desc.ignore = Unwrap(prims);
            });
    nb::class_<PointerHit>(module, "PointerHit")
        .def(nb::init<>())
        .def_rw("active", &PointerHit::active)
        .def_rw("hit", &PointerHit::hit)
        .def_rw("valid", &PointerHit::valid)
        .def_rw("position", &PointerHit::position)
        .def_rw("normal", &PointerHit::normal)
        .def_rw("local_position", &PointerHit::localPosition)
        .def_rw("distance", &PointerHit::distance)
        .def_rw("prim", &PointerHit::prim)
        .def_rw("path", &PointerHit::path)
        .def_rw("frame", &PointerHit::frame);
    nb::class_<LocomotionDesc>(module, "LocomotionDesc")
        .def(nb::init<>())
        .def_rw("snap_degrees", &LocomotionDesc::snapDegrees)
        .def_rw("back_step", &LocomotionDesc::backStep)
        .def_rw("move_mode", &LocomotionDesc::moveMode)
        .def_rw("speed", &LocomotionDesc::speed)
        .def_rw("fly_speed", &LocomotionDesc::flySpeed)
        .def_rw("floor_aligned", &LocomotionDesc::floorAligned)
        .def_prop_rw(
            "bodies", [](const LocomotionDesc& desc) { return Wrap(desc.bodies); },
            [](LocomotionDesc& desc, const std::vector<ScenePrim>& prims) {
                desc.bodies = Unwrap(prims);
            });
    nb::class_<AnchorDesc>(module, "AnchorDesc")
        .def(
            "__init__",
            [](AnchorDesc* desc, Anchor mode, std::string target, float smoothing,
               std::optional<double> fixedHeight) {
                new (desc) AnchorDesc{mode, std::move(target), smoothing, fixedHeight};
            },
            nb::arg("mode") = Anchor::Origin, nb::arg("target") = "", nb::arg("smoothing") = 0.3f,
            nb::arg("fixed_height") = nb::none())
        .def_rw("mode", &AnchorDesc::mode)
        .def_rw("target", &AnchorDesc::target)
        .def_rw("smoothing", &AnchorDesc::smoothing)
        .def_rw("fixed_height", &AnchorDesc::fixedHeight);

    nb::class_<Rig>(module, "Rig")
        .def(nb::init<Ref<Scene>, Session&, const RigDesc&>(), nb::arg("scene"), nb::arg("session"),
             nb::arg("desc") = RigDesc{}, nb::keep_alive<1, 3>())
        .def_prop_ro("root", [](const Rig& rig) { return Wrap(rig.Root()); })
        .def_prop_ro("head", [](const Rig& rig) { return Wrap(rig.Head()); })
        .def(
            "eye", [](const Rig& rig, Side eye) { return Wrap(rig.Eye(eye)); }, nb::arg("eye"))
        .def(
            "hand", [](const Rig& rig, Side hand) { return Wrap(rig.Hand(hand)); }, nb::arg("hand"))
        .def_prop_ro("product", &Rig::Product)
        .def_prop_ro("color_var", &Rig::ColorVar)
        .def_prop_ro("vars", &Rig::Vars)
        .def_prop_ro("recommended_resolution", &Rig::RecommendedResolution)
        .def("update", &Rig::Update, nb::arg("frame"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_rw("root_pose", &Rig::RootPose, &Rig::SetRootPose)
        .def("world_pose", &Rig::WorldPose, nb::arg("pose"))
        .def_prop_rw("pointers", &Rig::GetPointers, &Rig::SetPointers)
        .def("laser", &Rig::Laser, nb::arg("hand"), nb::call_guard<nb::gil_scoped_release>())
        .def("teleport", &Rig::Teleport, nb::arg("hand"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_rw("locomotion", &Rig::GetLocomotion, &Rig::SetLocomotion)
        .def("teleport_to", &Rig::TeleportTo, nb::arg("frame"), nb::arg("hit"))
        .def("step_back", &Rig::StepBack, nb::arg("frame"))
        .def("snap_turn", &Rig::SnapTurn, nb::arg("frame"), nb::arg("direction"))
        .def("move", &Rig::Move, nb::arg("frame"), nb::arg("stick"), nb::arg("rise") = 0.0f)
        .def_prop_rw("user_scale", &Rig::UserScale, &Rig::SetUserScale)
        .def("draw", &Rig::Draw, nb::arg("overlays"), nb::arg("frame"),
             nb::arg("visuals") = RigVisuals{})
        .def_prop_rw("anchor", &Rig::GetAnchor, &Rig::SetAnchor)
        .def_prop_ro("body", [](const Rig& rig) -> std::optional<ScenePrim> {
            const std::optional<Prim> body = rig.Body();
            return body ? std::optional(Wrap(*body)) : std::nullopt;
        });
}

}
}

NB_MODULE(_xr, module) {
    nb::module_::import_("ernie._ernie");
    ernie::xr::BindFrame(module);
    ernie::xr::BindSession(module);
    ernie::xr::BindRig(module);
}
