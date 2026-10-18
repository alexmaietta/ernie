// SPDX-License-Identifier: BSD-3-Clause
/// @file shared.h
/// @brief What C++ sees of the types shaders share with it. A shader file's shared part, before
/// its `#ifndef __cplusplus` code, is read by both languages: Slang's scalar types are glm's here,
/// laid out alike as Slang lays out data reached through `Ref` (packed, scalar alignment).
#pragma once

#include <glm/glm.hpp>

#include <cstdint>

namespace ernie::shaders {

using uint = std::uint32_t;
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
using std::uint64_t;

/// @brief A buffer a shader reads or writes `T`s of, as ref.slang's `Ref`: 8 bytes, a device
/// address on Vulkan (an arena index and offset on WebGPU), 0 for none.
template <typename T> struct Ref {
    std::uint64_t handle = 0;
    bool operator==(const Ref&) const = default;
};

}  // namespace ernie::shaders
