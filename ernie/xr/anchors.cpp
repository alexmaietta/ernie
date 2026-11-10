// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

#include "ernie/util/error.h"
#include "ernie/xr/poses.h"
#include "ernie/xr/rig_state.h"

namespace ernie::xr {
namespace {

double HeadingDegrees(Double3 forward) {
    return std::atan2(forward.y, forward.x) * 180 / std::numbers::pi;
}

Quatd Level(Quatd rotation) {
    return FromAxisAngle({0, 0, 1}, HeadingDegrees(Rotate(rotation, {1, 0, 0})));
}

bool Follows(Anchor mode) {
    return mode == Anchor::Prim || mode == Anchor::Follow || mode == Anchor::FollowSmoothed;
}

}

std::optional<std::string> RigState::FramePath() const {
    if (body) {
        return body->Path();
    }
    if (Follows(anchor.mode) && anchorTarget) {
        return anchorTarget->Path();
    }
    return parentPath;
}

bool RigState::FrameYaw() const {
    return !body && (anchor.mode == Anchor::Follow || anchor.mode == Anchor::FollowSmoothed);
}

void RigState::AdoptFrame(const WorldTransform& world) {
    const auto axis = [](Float3 v) {
        const double length = std::sqrt(double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
        return length > 0 ? Double3{v.x / length, v.y / length, v.z / length} : Double3{0, 0, 0};
    };
    frameRotation = FromAxes(axis(world.x), axis(world.y), axis(world.z));
    if (FrameYaw()) {
        frameRotation = Level(frameRotation);
    }
    framePosition = {double(world.high.x) + world.low.x, double(world.high.y) + world.low.y,
                     double(world.high.z) + world.low.z};
}

void RigState::MirrorFrame() {
    const std::optional<std::string> path = FramePath();
    if (!path) {
        frameRotation = {0, 0, 0, 1};
        framePosition = {0, 0, 0};
        readingFrame.clear();
        return;
    }
    if (!readingFrame.empty() && frameRead.Ready()) {
        if (readingFrame == *path) {
            WorldTransform world;
            std::memcpy(&world, frameRead.Bytes().data(), sizeof(world));
            AdoptFrame(world);
        }
        readingFrame.clear();
    }
    if (readingFrame.empty() && scene->Published() > 0) {
        frameRead = scene->ReadWorldTransformsAsync(std::vector<std::string>{*path});
        readingFrame = *path;
    }
}

void RigState::ReadFrame() {
    readingFrame.clear();
    const std::optional<std::string> path = FramePath();
    if (!path) {
        frameRotation = {0, 0, 0, 1};
        framePosition = {0, 0, 0};
        return;
    }
    scene->Publish().Wait();
    AdoptFrame(scene->WorldTransforms(std::vector<std::string>{*path}).front());
}

void RigState::Follow(const std::optional<Prim>& target, bool yaw, bool scaled, float lag) {
    if (!target) {
        if (root.HasApi<FollowAPI>()) {
            scene->Unapply<FollowAPI>(Prim(root));
        }
        return;
    }
    if (!root.HasApi<FollowAPI>()) {
        scene->Apply<FollowAPI>(Prim(root));
    }
    root.Set(FollowAPI::kFollowTarget, std::vector<std::string>{target->Path()});
    root.Set(FollowAPI::kFollowRotate, yaw ? FollowRotate::Yaw : FollowRotate::Inherit);
    root.Set(FollowAPI::kFollowInheritScale, scaled);
    root.Set(FollowAPI::kFollowPositionLag, Float3{lag, lag, lag});
    root.Set(FollowAPI::kFollowRotationLag, lag);
}

void RigState::LandOnCamera(const Frame& frame) {
    cameraPending = false;
    scene->Publish().Wait();
    const WorldTransform camera =
        scene->WorldTransforms(std::vector<std::string>{anchorTarget->Path()}).front();
    const Double3 eye{double(camera.high.x) + camera.low.x, double(camera.high.y) + camera.low.y,
                      double(camera.high.z) + camera.low.z};
    const Double3 looking{-double(camera.z.x), -double(camera.z.y), -double(camera.z.z)};
    const Double3 tracked = frame.head.valid ? frame.head.position : Double3{0, 0, 0};
    const double headHeading =
        frame.head.valid ? HeadingDegrees(Rotate(frame.head.orientation, {0, 0, -1})) : 90;
    const Quatd turn = FromAxisAngle({0, 0, 1}, HeadingDegrees(looking) - headHeading);
    const Double3 world = Add(eye, Scale(Rotate(turn, tracked), -userScale));
    const Quatd into = Conjugate(frameRotation);
    rootPose.orientation = Multiply(into, turn);
    rootPose.position = Rotate(into, Add(world, Scale(framePosition, -1)));
    WriteRoot();
}

void RigState::HoldHeight(const Frame& frame) {
    if (anchor.fixedHeight && frame.head.valid) {
        rootPose.position.z = *anchor.fixedHeight - userScale * frame.head.position.z;
        WriteRoot();
    }
}

std::optional<Prim> RigState::HitBody(const PointerHit& hit) const {
    if (hit.prim == kNoPrim || locomotion.bodies.empty()) {
        return std::nullopt;
    }
    for (std::optional<Prim> at = scene->ByIndex(hit.prim); at; at = at->Parent()) {
        for (const Prim& candidate : locomotion.bodies) {
            if (candidate.Index() == at->Index()) {
                return candidate;
            }
        }
    }
    return std::nullopt;
}

void RigState::Reframe(const Frame& frame, const PointerHit& hit, const std::optional<Prim>& onto) {
    const double heading = HeadingDegrees(Rotate(WorldRotation(), {1, 0, 0}));
    body = onto;
    ground.reset();
    if (onto) {
        Follow(onto, false, false, 0);
    } else {
        Follow(Follows(anchor.mode) ? anchorTarget : std::nullopt, anchor.mode != Anchor::Prim,
               anchor.mode == Anchor::Prim,
               anchor.mode == Anchor::FollowSmoothed ? anchor.smoothing : 0.0f);
    }
    ReadFrame();
    const Double3 tracked = frame.head.valid ? frame.head.position : Double3{0, 0, 0};
    rootPose.orientation =
        FromAxisAngle({0, 0, 1}, heading - HeadingDegrees(Rotate(frameRotation, {1, 0, 0})));
    const Double3 landing =
        Rotate(Conjugate(frameRotation), Add(hit.position, Scale(framePosition, -1)));
    rootPose.position =
        Add(landing, Scale(Rotate(rootPose.orientation, {tracked.x, tracked.y, 0}), -userScale));
    WriteRoot();
}

void Rig::SetAnchor(const AnchorDesc& desc) {
    RigState& state = *m_state;
    std::optional<Prim> target;
    if (desc.mode != Anchor::Origin) {
        if (desc.target.empty()) {
            throw Error("an anchor other than Origin needs a target");
        }
        target = state.scene->Find(desc.target);
        if (!target || !target->IsA<Xformable>()) {
            throw Error("there's no Xformable at {} to anchor to", desc.target);
        }
    }
    state.anchor = desc;
    state.anchorTarget = target;
    state.body.reset();
    state.ground.reset();
    state.Follow(Follows(desc.mode) ? target : std::nullopt, desc.mode != Anchor::Prim,
                 desc.mode == Anchor::Prim,
                 desc.mode == Anchor::FollowSmoothed ? desc.smoothing : 0.0f);
    state.rootPose = {{0, 0, 0}, {0, 0, 0, 1}, true};
    state.WriteRoot();
    state.ReadFrame();
    state.cameraPending = desc.mode == Anchor::Camera;
    if (state.lastFrame) {
        if (state.cameraPending) {
            state.LandOnCamera(*state.lastFrame);
        }
        state.HoldHeight(*state.lastFrame);
    }
}

const AnchorDesc& Rig::GetAnchor() const {
    return m_state->anchor;
}

std::optional<Prim> Rig::Body() const {
    return m_state->body;
}

}
