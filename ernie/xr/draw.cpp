// SPDX-License-Identifier: BSD-3-Clause
#include <vector>

#include "ernie/util/profiler.h"
#include "ernie/xr/poses.h"
#include "ernie/xr/rig_state.h"

namespace ernie::xr {
namespace {

constexpr HandJoint kBones[][2] = {
    {HandJoint::Wrist, HandJoint::ThumbMetacarpal},
    {HandJoint::ThumbMetacarpal, HandJoint::ThumbProximal},
    {HandJoint::ThumbProximal, HandJoint::ThumbDistal},
    {HandJoint::ThumbDistal, HandJoint::ThumbTip},
    {HandJoint::Wrist, HandJoint::IndexMetacarpal},
    {HandJoint::IndexMetacarpal, HandJoint::IndexProximal},
    {HandJoint::IndexProximal, HandJoint::IndexIntermediate},
    {HandJoint::IndexIntermediate, HandJoint::IndexDistal},
    {HandJoint::IndexDistal, HandJoint::IndexTip},
    {HandJoint::Wrist, HandJoint::MiddleMetacarpal},
    {HandJoint::MiddleMetacarpal, HandJoint::MiddleProximal},
    {HandJoint::MiddleProximal, HandJoint::MiddleIntermediate},
    {HandJoint::MiddleIntermediate, HandJoint::MiddleDistal},
    {HandJoint::MiddleDistal, HandJoint::MiddleTip},
    {HandJoint::Wrist, HandJoint::RingMetacarpal},
    {HandJoint::RingMetacarpal, HandJoint::RingProximal},
    {HandJoint::RingProximal, HandJoint::RingIntermediate},
    {HandJoint::RingIntermediate, HandJoint::RingDistal},
    {HandJoint::RingDistal, HandJoint::RingTip},
    {HandJoint::Wrist, HandJoint::LittleMetacarpal},
    {HandJoint::LittleMetacarpal, HandJoint::LittleProximal},
    {HandJoint::LittleProximal, HandJoint::LittleIntermediate},
    {HandJoint::LittleIntermediate, HandJoint::LittleDistal},
    {HandJoint::LittleDistal, HandJoint::LittleTip},
};

Float3 Narrow(Double3 v) {
    return {float(v.x), float(v.y), float(v.z)};
}

std::vector<Float3> Narrow(const std::vector<Double3>& points) {
    std::vector<Float3> narrowed;
    narrowed.reserve(points.size());
    for (const Double3& point : points) {
        narrowed.push_back(Narrow(point));
    }
    return narrowed;
}

}

void Rig::Draw(Overlays& overlays, const Frame& frame, const RigVisuals& visuals) {
    ERNIE_ZONE("XR/Draw");
    RigState& state = *m_state;
    OverlayStyle style;
    style.color = visuals.color;
    style.anchor = state.root.Index();
    style.pickable = false;
    OverlayStyle invalid = style;
    invalid.color = visuals.invalidColor;
    Overlay overlay;
    for (const Side side : {Side::Left, Side::Right}) {
        const Controller& controller = frame.Controller(side);
        if (visuals.controllers && controller.active && controller.pose.valid) {
            overlay.Box(Narrow(controller.pose.position), {0.03f, 0.05f, 0.08f}, style,
                        controller.pose.orientation);
            if (controller.aim.valid) {
                overlay.Frame(Narrow(controller.aim.position), controller.aim.orientation, 0.1f,
                              style);
            }
        }
        const HandJoints& hand = frame.Hand(side);
        if (visuals.hands && hand.active) {
            std::vector<Float3> joints;
            for (const JointPose& joint : hand.joints) {
                if (joint.pose.valid) {
                    joints.push_back(Narrow(joint.pose.position));
                }
            }
            overlay.Markers(joints, style);
            std::vector<Float3> bones;
            for (const auto& [from, to] : kBones) {
                if (hand[from].pose.valid && hand[to].pose.valid) {
                    bones.push_back(Narrow(hand[from].pose.position));
                    bones.push_back(Narrow(hand[to].pose.position));
                }
            }
            overlay.Lines(bones, style);
        }
        if (visuals.lasers && controller.active) {
            const PointerHit laser = Laser(side);
            if (laser.active && laser.path.size() >= 2) {
                overlay.Polyline(Narrow(laser.path), false, laser.valid ? style : invalid);
            }
        }
    }
    if (visuals.teleport) {
        const PointerHit arc = Teleport(Side::Right);
        if (arc.active && arc.path.size() >= 2) {
            const OverlayStyle& used = arc.valid ? style : invalid;
            overlay.Polyline(Narrow(arc.path), false, used);
            if (arc.hit) {
                overlay.Circle(Narrow(arc.localPosition), {0, 0, 1}, 0.3f, used);
            }
        }
    }
    if (visuals.playArea && state.playArea.x > 0 && state.playArea.y > 0) {
        overlay.Rectangle({0, 0, 0}, {state.playArea.x / 2, 0, 0}, {0, state.playArea.y / 2, 0},
                          style);
    }
    overlays.Set(visuals.layer, std::move(overlay));
}

}
