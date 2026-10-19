// SPDX-License-Identifier: BSD-3-Clause
/// @file random.slang
/// @brief Deterministic random numbers keyed by sample identity, as util/random.h computes them on
/// the host, bit for bit: hashed uniform numbers, and an Owen-scrambled Sobol sequence over a
/// pixel's samples. Any sample's numbers are regenerated without the ones before it, in any
/// order, as a backward pass replays a path.
///
///     import random;
///     SampleKey key = { frameSeed, view, pixel, sample };
///     float2 jitter = float2(SampleSobol(key, kPixelDimension),
///                            SampleSobol(key, kPixelDimension + 1));
///
/// References:
/// - M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering", JCGT 9(3), 2020.
/// - B. Burley, "Practical Hash-based Owen Scrambling", JCGT 9(4), 2020.
/// - M. Pharr, W. Jakob and G. Humphreys, "Physically Based Rendering", 4th edition, 2023
///   (`PaddedSobolSampler`).

/// Each feature's dimensions, fixed so adding one reshuffles none of the others. Each pair is
/// stratified on its own: the pixel's position, the lens's, then time with the band.
static const uint kPixelDimension = 0;
static const uint kLensDimension = 2;
static const uint kTimeDimension = 4;
static const uint kBandDimension = 5;

/// Identifies one sample: which frame (`FrameSeed` on the host), view, pixel and sample it is.
struct SampleKey {
    uint frameSeed;
    uint view;
    uint pixel;
    uint sample;
};

/// @return The pcg3d hash of `v`.
uint3 Pcg3d(uint3 v) {
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

/// @return The pcg4d hash of `v`.
uint4 Pcg4d(uint4 v) {
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

/// @return 32 random bits for dimension `dimension` of a sample.
uint SampleBits(SampleKey key, uint dimension) {
    uint4 h = Pcg4d(uint4(key.pixel, key.sample, dimension, key.view ^ key.frameSeed));
    return h.x ^ Pcg3d(uint3(h.y, h.z, key.frameSeed)).x;
}

/// @return A uniform number in [0, 1) for dimension `dimension` of a sample, with 24 random bits.
float SampleUniform(SampleKey key, uint dimension) {
    return float(SampleBits(key, dimension) >> 8u) * (1.0 / 16777216.0);
}

/// @return `x` permuted as an Owen scramble does, keyed by `seed`, in reversed bit order.
uint LaineKarrasPermutation(uint x, uint seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

/// @return `x` Owen-scrambled under `seed`.
uint NestedUniformScramble(uint x, uint seed) {
    return reversebits(LaineKarrasPermutation(reversebits(x), seed));
}

/// @return `value` folded into a 32-bit `seed`.
uint CombineSeed(uint seed, uint value) {
    return seed ^ (value + (seed << 6u) + (seed >> 2u));
}

/// The Sobol sequence's direction numbers for its first two dimensions: a (0,2)-sequence.
static const uint kSobolDirections[2][32] = {
    { 0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000,
      0x01000000, 0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000,
      0x00020000, 0x00010000, 0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800,
      0x00000400, 0x00000200, 0x00000100, 0x00000080, 0x00000040, 0x00000020, 0x00000010,
      0x00000008, 0x00000004, 0x00000002, 0x00000001 },
    { 0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000,
      0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000,
      0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800,
      0xcc00cc00, 0xaa00aa00, 0xff00ff00, 0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
      0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff }
};

/// @return Point `index`'s coordinate `dimension` (0 or 1) of the Sobol sequence, as 32 bits.
uint SobolBits(uint index, uint dimension) {
    uint bits = 0;
    for (uint bit = 0; index != 0; ++bit, index >>= 1u) {
        if ((index & 1u) != 0) {
            bits ^= kSobolDirections[dimension][bit];
        }
    }
    return bits;
}

/// @return A low-discrepancy number in [0, 1) for dimension `dimension` of a sample, with 24
/// bits: an Owen-scrambled, shuffled Sobol sequence over the pixel's samples, each pair of
/// dimensions Sobol's first two under a scramble of its own.
float SampleSobol(SampleKey key, uint dimension) {
    uint pair = dimension / 2u;
    uint seed = Pcg4d(uint4(key.pixel, pair, key.view, key.frameSeed)).x;
    uint index = NestedUniformScramble(key.sample, seed);
    uint bits = NestedUniformScramble(SobolBits(index, dimension % 2u),
                                      CombineSeed(seed, dimension % 2u));
    return float(bits >> 8u) * (1.0 / 16777216.0);
}
