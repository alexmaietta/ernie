// SPDX-License-Identifier: BSD-3-Clause
/// @file random.h
/// @brief Deterministic random streams shared by every render
/// backend.
#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string_view>

/// Deterministic random streams shared by every render backend.
///
/// Every random decision a renderer makes (sensor noise, and the
/// samples of ray and path tracers) derives from three numbers: a
/// seed the scene authors, a stream naming what is rendered (a render
/// product, `HashStream` its path) and the frame number. The same
/// scene, settings and frame then render the same image on any
/// machine, in any order, whatever else renders alongside.
///
/// Reference: M. Jarzynski and M. Olano, "Hash Functions for GPU
/// Rendering", Journal of Computer Graphics Techniques 9(3):21-38,
/// 2020 (pcg3d, pcg4d).
namespace ernie {

/// The pcg3d hash: three well-mixed 32-bit words from three. Shaders
/// implement the same function.
inline constexpr glm::uvec3 Pcg3d(glm::uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

/// The pcg4d hash: four well-mixed 32-bit words from four. Shaders implement the same function.
inline constexpr glm::uvec4 Pcg4d(glm::uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> 16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

/// A stable 32-bit stream id for `name`, such as a render product's
/// path: its 32-bit FNV-1a hash.
inline constexpr std::uint32_t HashStream(std::string_view name) {
    std::uint32_t hash = 2166136261u;
    for (const char c : name) {
        hash = (hash ^ static_cast<std::uint8_t>(c)) * 16777619u;
    }
    return hash;
}

/// The seed of frame `frame` of stream `stream` under the authored
/// `seed`, from which a frame's temporal noise and samples derive.
/// Distinct frames, streams or seeds give unrelated seeds.
inline constexpr std::uint32_t FrameSeed(std::uint32_t seed, std::uint32_t stream,
                                         std::uint64_t frame) {
    const glm::uvec3 h = Pcg3d(glm::uvec3(seed, stream ^ static_cast<std::uint32_t>(frame >> 32u),
                                          static_cast<std::uint32_t>(frame)));
    return h.x ^ h.y ^ h.z;
}

/// Identifies one sample: which frame (`FrameSeed`), view, pixel and sample index it is.
struct SampleKey {
    std::uint32_t frameSeed = 0;
    std::uint32_t view = 0;
    std::uint32_t pixel = 0;
    std::uint32_t sample = 0;
};

/// 32 random bits for dimension `dimension` of a sample. A counter-based stream: any sample's
/// numbers can be regenerated without the ones before it, in any order, on the host or the GPU.
inline constexpr std::uint32_t SampleBits(const SampleKey& key, std::uint32_t dimension) {
    const glm::uvec4 h =
        Pcg4d(glm::uvec4(key.pixel, key.sample, dimension, key.view ^ key.frameSeed));
    return h.x ^ Pcg3d(glm::uvec3(h.y, h.z, key.frameSeed)).x;
}

/// A uniform number in [0, 1) for dimension `dimension` of a sample, with 24 random bits.
inline constexpr float SampleUniform(const SampleKey& key, std::uint32_t dimension) {
    return static_cast<float>(SampleBits(key, dimension) >> 8u) * (1.0f / 16777216.0f);
}

}  // namespace ernie
