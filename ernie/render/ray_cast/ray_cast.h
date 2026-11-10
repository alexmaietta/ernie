// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/scene/prim.h"
#include "ernie/stage/schema/ids.h"

namespace ernie {

class Scene;

enum class RayOutput : uint32_t {
    None = 0,
    Distance = 1u << 0,
    Prim = 1u << 1,
    Normal = 1u << 2,
    Position = 1u << 3,
    InstanceId = 1u << 4,
};

constexpr RayOutput operator|(RayOutput a, RayOutput b) {
    return RayOutput(uint32_t(a) | uint32_t(b));
}

constexpr bool operator&(RayOutput a, RayOutput b) {
    return (uint32_t(a) & uint32_t(b)) != 0;
}

enum class RayCastMode : uint8_t { Closest, Any };

struct ERNIE_API Rays {
    device::Tensor origins;
    device::Tensor directions;
    device::Tensor tMins;
    device::Tensor tMaxes;
    device::Tensor worlds;
    float minDistance = 0;
    float maxDistance = std::numeric_limits<float>::infinity();
    uint32_t world = kGlobalWorld;
    std::optional<Prim> space;
};

struct ERNIE_API RayHits {
    device::Tensor hit;
    device::Tensor distance;
    device::Tensor prim;
    device::Tensor normal;
    device::Tensor position;
    device::Tensor instance;
    device::Operation done;
};

struct ERNIE_API RayCastDesc {
    RayCastMode mode = RayCastMode::Closest;
    RayOutput outputs = RayOutput::Distance | RayOutput::Prim;
    std::string label = "RayCast";
    std::vector<Prim> ignore;
};

class ERNIE_API RayCaster {
  public:
    explicit RayCaster(Ref<Scene> scene);
    ~RayCaster();
    RayCaster(RayCaster&&) noexcept;
    RayCaster& operator=(RayCaster&&) noexcept;

    device::Operation Cast(const Rays& rays, RayHits& hits, const RayCastDesc& desc = {},
                           Stream stream = Stream::None());
    RayHits Cast(const Rays& rays, const RayCastDesc& desc = {});
    Scene& GetScene() const;

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

}
