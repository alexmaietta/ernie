// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ernie/device/ref.h"
#include "ernie/scene/scene.h"
#include "ernie/stage/schema/types.h"
#include "ernie/xr/frame.h"
#include "ernie/xr_export.h"

namespace ernie::xr {

struct RigDesc {
    std::string path = "/XR";
    std::vector<std::string> extraVars;
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

  private:
    std::shared_ptr<RigState> m_state;
};

}
