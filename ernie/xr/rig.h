// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ernie/device/ref.h"
#include "ernie/overlay/overlay.h"
#include "ernie/scene/scene.h"
#include "ernie/stage/schema/ids.h"
#include "ernie/stage/schema/types.h"
#include "ernie/xr/frame.h"
#include "ernie/xr_export.h"

namespace ernie::xr {

struct RigDesc {
    std::string path = "/XR";
    std::vector<std::string> extraVars;
    std::vector<std::string> overlays = {"*"};
};

struct RigVisuals {
    bool controllers = true;
    bool hands = true;
    bool lasers = true;
    bool teleport = false;
    bool playArea = true;
    Float4 color = {0.3f, 0.75f, 1, 0.9f};
    Float4 invalidColor = {0.95f, 0.3f, 0.25f, 0.9f};
    std::string layer = "xr";
};

struct PointerDesc {
    float maxDistance = 30;
    float arcSpeed = 7;
    float gravity = 9.81f;
    uint32_t arcSegments = 24;
    float arcTime = 2;
    float maxSlopeDegrees = 35;
    bool wait = true;
    std::vector<Prim> ignore;
};

struct PointerHit {
    bool active = false;
    bool hit = false;
    bool valid = false;
    Double3 position;
    Double3 normal;
    Double3 localPosition;
    double distance = 0;
    uint32_t prim = kNoPrim;
    std::vector<Double3> path;
    uint64_t frame = 0;
};

enum class MoveMode : uint32_t { Smooth, Grounded, Fly };

struct LocomotionDesc {
    double snapDegrees = 30;
    double backStep = 1;
    MoveMode moveMode = MoveMode::Smooth;
    double speed = 1.5;
    double flySpeed = 3;
    bool floorAligned = true;
    std::vector<Prim> bodies;
};

enum class Anchor : uint32_t { Origin, Camera, Prim, Follow, FollowSmoothed };

struct AnchorDesc {
    Anchor mode = Anchor::Origin;
    std::string target;
    float smoothing = 0.3f;
    std::optional<double> fixedHeight;
};

class Session;
struct RigState;

class ERNIE_XR_API Rig {
  public:
    Rig(Ref<Scene> scene, Session& session, const RigDesc& desc = {});

    XformPrim Root() const;
    XformPrim Head() const;
    CameraPrim Eye(Side eye) const;
    XformPrim Hand(Side hand) const;
    const std::string& Product() const;
    const std::string& ColorVar() const;
    const std::vector<std::string>& Vars() const;
    Uint2 RecommendedResolution() const;

    void Update(const Frame& frame);
    void SetRootPose(const Pose& pose);
    Pose RootPose() const;
    Pose WorldPose(const Pose& pose) const;

    void SetPointers(const PointerDesc& desc);
    const PointerDesc& GetPointers() const;
    PointerHit Laser(Side hand);
    PointerHit Teleport(Side hand);

    void SetLocomotion(const LocomotionDesc& desc);
    const LocomotionDesc& GetLocomotion() const;
    bool TeleportTo(const Frame& frame, const PointerHit& hit);
    void StepBack(const Frame& frame);
    void SnapTurn(const Frame& frame, int direction);
    void Move(const Frame& frame, Float2 stick, float rise = 0);
    void SetUserScale(double scale);
    double UserScale() const;

    void Draw(Overlays& overlays, const Frame& frame, const RigVisuals& visuals = {});

    void SetAnchor(const AnchorDesc& desc);
    const AnchorDesc& GetAnchor() const;
    std::optional<Prim> Body() const;

  private:
    std::shared_ptr<RigState> m_state;
};

}
