// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <vector>

#include "ernie/util/error.h"
#include "ernie/xr/poses.h"
#include "ernie/xr/rig_state.h"

namespace ernie::xr {
namespace {

Double3 Heading(const Pose& head) {
    const Double3 forward = Rotate(head.orientation, {0, 0, -1});
    const double size = std::sqrt(forward.x * forward.x + forward.y * forward.y);
    return size > 1e-4 ? Double3{forward.x / size, forward.y / size, 0} : Double3{0, 1, 0};
}

Double3 HeadPosition(const Frame& frame) {
    return frame.head.valid ? frame.head.position : Double3{0, 0, 0};
}

}

void RigState::WriteRoot() {
    root.SetTranslate(rootPose.position);
    root.SetOrient(rootPose.orientation);
    const float scale = float(userScale);
    root.SetScale({scale, scale, scale});
}

void RigState::Offset(Double3 move) {
    rootPose.position =
        Add(rootPose.position, Scale(Rotate(rootPose.orientation, move), userScale));
    WriteRoot();
}

void RigState::Land() {
    if (!ground || !ground->hits.done.Ready()) {
        return;
    }
    const GroundCast cast = *ground;
    ground.reset();
    if (cast.hits.hit.ToHost<uint32_t>()[0] == 0) {
        return;
    }
    const double t = cast.hits.distance.ToHost<float>()[0];
    Offset({0, 0, cast.origin.z - t * cast.reach});
}

void Rig::SetLocomotion(const LocomotionDesc& desc) {
    m_state->locomotion = desc;
}

const LocomotionDesc& Rig::GetLocomotion() const {
    return m_state->locomotion;
}

bool Rig::TeleportTo(const Frame& frame, const PointerHit& hit) {
    if (!hit.valid) {
        return false;
    }
    const Double3 head = HeadPosition(frame);
    m_state->ground.reset();
    m_state->Offset(
        {hit.localPosition.x - head.x, hit.localPosition.y - head.y, hit.localPosition.z});
    return true;
}

void Rig::StepBack(const Frame& frame) {
    m_state->Offset(Scale(Heading(frame.head), -m_state->locomotion.backStep));
}

void Rig::SnapTurn(const Frame& frame, int direction) {
    RigState& state = *m_state;
    const double degrees = (direction < 0 ? -1.0 : 1.0) * state.locomotion.snapDegrees;
    const Double3 head = HeadPosition(frame);
    const Double3 pivot{head.x, head.y, 0};
    const Quatd turned = Multiply(state.rootPose.orientation, FromAxisAngle({0, 0, 1}, degrees));
    const Double3 before = Rotate(state.rootPose.orientation, pivot);
    const Double3 after = Rotate(turned, pivot);
    state.rootPose.position =
        Add(state.rootPose.position,
            Scale({before.x - after.x, before.y - after.y, before.z - after.z}, state.userScale));
    state.rootPose.orientation = turned;
    state.WriteRoot();
}

void Rig::Move(const Frame& frame, Float2 stick, float rise) {
    RigState& state = *m_state;
    state.Land();
    const LocomotionDesc& desc = state.locomotion;
    const double seconds = frame.deltaTime;
    const Double3 forward = Heading(frame.head);
    const Double3 right{forward.y, -forward.x, 0};
    Double3 move;
    if (desc.moveMode == MoveMode::Fly) {
        const Double3 ahead =
            desc.floorAligned ? forward : Rotate(frame.head.orientation, {0, 0, -1});
        move = Scale(Add(Add(Scale(right, stick.x), Scale(ahead, stick.y)), {0, 0, rise}),
                     desc.flySpeed * seconds);
    } else {
        move = Scale(Add(Scale(right, stick.x), Scale(forward, stick.y)), desc.speed * seconds);
    }
    if (move.x != 0 || move.y != 0 || move.z != 0) {
        state.Offset(move);
    }
    if (desc.moveMode == MoveMode::Grounded && frame.head.valid && !state.ground) {
        const Double3 head = frame.head.position;
        const double reach = head.z + 1;
        GroundCast cast;
        cast.origin = head;
        cast.reach = reach;
        cast.hits = state.Cast({float(head.x), float(head.y), float(head.z)}, {0, 0, -float(reach)},
                               RayOutput::Distance);
        state.ground = cast;
    }
}

void Rig::SetUserScale(double scale) {
    if (!(scale > 0)) {
        throw Error("a user scale must be above 0, not {}", scale);
    }
    RigState& state = *m_state;
    const Double3 head = state.lastFrame ? HeadPosition(*state.lastFrame) : Double3{0, 0, 0};
    const Double3 floor = Rotate(state.rootPose.orientation, {head.x, head.y, 0});
    state.rootPose.position = Add(state.rootPose.position, Scale(floor, state.userScale - scale));
    state.userScale = scale;
    state.WriteRoot();
}

double Rig::UserScale() const {
    return m_state->userScale;
}

}
