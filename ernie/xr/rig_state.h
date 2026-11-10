// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/render/ray_cast/ray_cast.h"
#include "ernie/scene/scene.h"
#include "ernie/xr/frame.h"
#include "ernie/xr/rig.h"

namespace ernie::xr {

constexpr uint32_t kLaser = 0;
constexpr uint32_t kTeleport = 1;
constexpr uint32_t kPointerKinds = 2;

struct PointerBatch {
    uint64_t frame = 0;
    std::array<bool, 2> active{};
    int32_t first[kPointerKinds][2] = {{-1, -1}, {-1, -1}};
    std::vector<Double3> points[kPointerKinds][2];
    RayHits hits;
    bool cast = false;
    bool read = false;
    PointerHit results[kPointerKinds][2];

    bool Done() const {
        return !cast || hits.done.Ready();
    }
};

struct GroundCast {
    double reach = 0;
    Double3 origin;
    RayHits hits;
};

struct RigState {
    Ref<Scene> scene;
    XformPrim root;
    XformPrim head;
    CameraPrim leftEye;
    CameraPrim rightEye;
    XformPrim leftHand;
    XformPrim rightHand;
    std::string product;
    std::string colorVar;
    std::string depthVar;
    std::vector<std::string> vars;
    Uint2 recommended;
    Pose rootPose{{0, 0, 0}, {0, 0, 0, 1}, true};
    double userScale = 1;

    std::optional<std::string> parentPath;
    Quatd frameRotation;
    Double3 framePosition;
    device::Download frameRead;
    std::string readingFrame;

    AnchorDesc anchor;
    std::optional<Prim> anchorTarget;
    std::optional<Prim> body;
    bool cameraPending = false;

    std::unique_ptr<RayCaster> caster;
    PointerDesc pointers;
    LocomotionDesc locomotion;
    bool wanted[kPointerKinds][2] = {};
    std::optional<Frame> lastFrame;
    std::optional<PointerBatch> pending;
    std::optional<PointerBatch> latest;
    std::optional<GroundCast> ground;

    std::optional<std::string> FramePath() const;
    bool FrameYaw() const;
    void AdoptFrame(const WorldTransform& world);
    void MirrorFrame();
    void ReadFrame();
    void Follow(const std::optional<Prim>& target, bool yaw, bool scaled, float lag);
    void LandOnCamera(const Frame& frame);
    void HoldHeight(const Frame& frame);
    std::optional<Prim> HitBody(const PointerHit& hit) const;
    void Reframe(const Frame& frame, const PointerHit& hit, const std::optional<Prim>& onto);
    Quatd WorldRotation() const;
    RayHits Cast(const std::vector<float>& origins, const std::vector<float>& steps,
                 RayOutput outputs);
    PointerBatch Queue(const Frame& frame, const bool (&kinds)[kPointerKinds][2]);
    void Resolve(PointerBatch& batch) const;
    PointerHit Pointer(uint32_t kind, Side hand);
    void WriteRoot();
    void Offset(Double3 move);
    void Land();
};

}
