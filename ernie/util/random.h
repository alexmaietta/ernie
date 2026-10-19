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

/// @return `x`'s bits in reverse order.
inline constexpr std::uint32_t ReverseBits(std::uint32_t x) {
    x = ((x >> 1u) & 0x55555555u) | ((x & 0x55555555u) << 1u);
    x = ((x >> 2u) & 0x33333333u) | ((x & 0x33333333u) << 2u);
    x = ((x >> 4u) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4u);
    x = ((x >> 8u) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8u);
    return (x >> 16u) | (x << 16u);
}

/// @return `x` permuted as an Owen scramble does, keyed by `seed`: each bit flipped by a hash of
/// the bits below it, in the reversed order `NestedUniformScramble` hands it.
inline constexpr std::uint32_t LaineKarrasPermutation(std::uint32_t x, std::uint32_t seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

/// @return `x` Owen-scrambled under `seed`: each bit flipped by a hash of the bits above it.
inline constexpr std::uint32_t NestedUniformScramble(std::uint32_t x, std::uint32_t seed) {
    return ReverseBits(LaineKarrasPermutation(ReverseBits(x), seed));
}

/// @return `value` folded into a 32-bit `seed`.
inline constexpr std::uint32_t CombineSeed(std::uint32_t seed, std::uint32_t value) {
    return seed ^ (value + (seed << 6u) + (seed >> 2u));
}

/// The Sobol sequence's direction numbers for its first two dimensions, van der Corput's and Joe
/// and Kuo's first: together a (0,2)-sequence, stratified in every pair of grid sizes.
inline constexpr std::uint32_t kSobolDirections[2][32] = {
    {0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000,
     0x01000000, 0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000,
     0x00020000, 0x00010000, 0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800,
     0x00000400, 0x00000200, 0x00000100, 0x00000080, 0x00000040, 0x00000020, 0x00000010,
     0x00000008, 0x00000004, 0x00000002, 0x00000001},
    {0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000,
     0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000,
     0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800,
     0xcc00cc00, 0xaa00aa00, 0xff00ff00, 0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
     0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff}};

/// @return Point `index`'s coordinate `dimension` (0 or 1) of the Sobol sequence, as 32 bits.
inline constexpr std::uint32_t SobolBits(std::uint32_t index, std::uint32_t dimension) {
    std::uint32_t bits = 0;
    for (std::uint32_t bit = 0; index != 0; ++bit, index >>= 1u) {
        if ((index & 1u) != 0) {
            bits ^= kSobolDirections[dimension][bit];
        }
    }
    return bits;
}

/// A low-discrepancy number in [0, 1) for dimension `dimension` of a sample, with 24 bits: an
/// Owen-scrambled, shuffled Sobol sequence over the pixel's samples. Dimensions come in pairs,
/// each Sobol's first two dimensions under a scramble of its own, so a pixel's first 2^k samples
/// stratify every pair as a (0,2)-sequence does, and a feature's dimensions keep their samples
/// when others are added. Shaders implement the same function.
///
/// References:
/// - B. Burley, "Practical Hash-based Owen Scrambling", JCGT 9(4), 2020.
/// - M. Pharr, W. Jakob and G. Humphreys, "Physically Based Rendering", 4th edition, 2023
///   (`PaddedSobolSampler`).
inline constexpr float SampleSobol(const SampleKey& key, std::uint32_t dimension) {
    const std::uint32_t pair = dimension / 2u;
    const std::uint32_t seed = Pcg4d(glm::uvec4(key.pixel, pair, key.view, key.frameSeed)).x;
    const std::uint32_t index = NestedUniformScramble(key.sample, seed);
    const std::uint32_t bits =
        NestedUniformScramble(SobolBits(index, dimension % 2u), CombineSeed(seed, dimension % 2u));
    return static_cast<float>(bits >> 8u) * (1.0f / 16777216.0f);
}

/// @brief SplitMix64's finaliser: every input bit affects every output bit.
/// @param z The bits.
/// @return Them mixed.
inline constexpr std::uint64_t Mix64(std::uint64_t z) {
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31u);
}

/// @brief Folds a value into a hash, as for keys of several fields.
/// @param seed  The hash so far.
/// @param value The next field.
/// @return The combined hash.
inline constexpr std::uint64_t HashCombine(std::uint64_t seed, std::uint64_t value) {
    return Mix64(seed + 0x9e3779b97f4a7c15ull + value);
}

/// A sequential generator for host-side Monte Carlo work such as baking tables: SplitMix64, with
/// full double precision. Use `SampleBits` instead for anything the GPU must reproduce.
///
/// Reference: G. L. Steele, D. Lea and C. H. Flood, "Fast Splittable Pseudorandom Number
/// Generators", OOPSLA 2014.
class SplitMix64 {
  public:
    /// @brief Starts a stream.
    /// @param seed The seed; different seeds give unrelated streams.
    explicit constexpr SplitMix64(std::uint64_t seed) : state_(seed) {}

    /// @brief Draws the next 64 random bits.
    /// @return The bits.
    constexpr std::uint64_t NextBits() {
        return Mix64(state_ += 0x9e3779b97f4a7c15ull);
    }

    /// @brief Draws a uniform number.
    /// @return A double in [0, 1) with 53 random bits.
    constexpr double NextUniform() {
        return static_cast<double>(NextBits() >> 11u) * 0x1.0p-53;
    }

  private:
    std::uint64_t state_;
};

}  // namespace ernie
