// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/xr/rig.h"

#include <cmath>
#include <utility>

#include "ernie/util/profiler.h"
#include "ernie/xr/rig_state.h"

namespace ernie::xr {
namespace {

void Place(const Prim& prim, const Pose& pose) {
    if (pose.valid) {
        prim.Set(Xformable::kTranslate, pose.position);
        prim.Set(Xformable::kOrient, pose.orientation);
    }
}

void Aim(const CameraPrim& eye, const View& view) {
    Place(eye, view.pose);
    const float focal = eye.FocalLength();
    const float left = std::tan(view.fov.left);
    const float right = std::tan(view.fov.right);
    const float up = std::tan(view.fov.up);
    const float down = std::tan(view.fov.down);
    eye.SetHorizontalAperture(focal * (right - left));
    eye.SetVerticalAperture(focal * (up - down));
    eye.SetHorizontalApertureOffset(focal * (right + left) / 2);
    eye.SetVerticalApertureOffset(focal * (up + down) / 2);
}

}

XformPrim Rig::Root() const {
    return m_state->root;
}

XformPrim Rig::Head() const {
    return m_state->head;
}

CameraPrim Rig::Eye(Side eye) const {
    return eye == Side::Left ? m_state->leftEye : m_state->rightEye;
}

XformPrim Rig::Hand(Side hand) const {
    return hand == Side::Left ? m_state->leftHand : m_state->rightHand;
}

const std::string& Rig::Product() const {
    return m_state->product;
}

const std::string& Rig::ColorVar() const {
    return m_state->colorVar;
}

const std::vector<std::string>& Rig::Vars() const {
    return m_state->vars;
}

Uint2 Rig::RecommendedResolution() const {
    return m_state->recommended;
}

void Rig::Update(const Frame& frame) {
    ERNIE_ZONE("XR/Rig update");
    RigState& state = *m_state;
    Place(state.head, frame.head);
    Aim(state.leftEye, frame.Eye(Side::Left));
    Aim(state.rightEye, frame.Eye(Side::Right));
    for (const Side side : {Side::Left, Side::Right}) {
        const Controller& controller = frame.Controller(side);
        if (controller.active) {
            Place(Hand(side), controller.pose);
        }
    }
    state.lastFrame = frame;
    state.MirrorParent();
    state.Land();
    const bool any = state.wanted[kLaser][0] || state.wanted[kLaser][1] ||
                     state.wanted[kTeleport][0] || state.wanted[kTeleport][1];
    if (any) {
        if (state.pending && state.pending->Done()) {
            state.latest = std::move(state.pending);
        }
        state.pending = state.Queue(frame, state.wanted);
    }
}

void Rig::SetRootPose(const Pose& pose) {
    RigState& state = *m_state;
    state.rootPose = pose;
    state.rootPose.valid = true;
    state.ground.reset();
    state.WriteRoot();
}

Pose Rig::RootPose() const {
    return m_state->rootPose;
}

}
