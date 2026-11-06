// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

static const uint kMaxGroupThreads = 256;
#if ERNIE_NARROW_WAVES
static const uint kMaxGroupWaves = kMaxGroupThreads / 4;
#else
static const uint kMaxGroupWaves = kMaxGroupThreads / 8;
#endif
static const uint kRowThreads = 16;

groupshared uint s_group[kMaxGroupThreads + kMaxGroupThreads / kRowThreads];
groupshared uint s_rows[kMaxGroupThreads / kRowThreads + 1];

uint Padded(uint thread) {
    return thread + thread / kRowThreads;
}

interface ICombine {
    static uint Identity();
    static uint Combine(uint a, uint b);
    static uint Wave(uint value);
}

struct SumCombine : ICombine {
    static uint Identity() { return 0; }
    static uint Combine(uint a, uint b) { return a + b; }
    static uint Wave(uint value) { return WaveActiveSum(value); }
}

struct MinCombine : ICombine {
    static uint Identity() { return 0xFFFFFFFFu; }
    static uint Combine(uint a, uint b) { return min(a, b); }
    static uint Wave(uint value) { return WaveActiveMin(value); }
}

uint GroupCombine<C : ICombine>(uint thread, uint threads, uint value) {
    uint combined = C.Identity();
#if ERNIE_HIP
    s_group[Padded(thread)] = value;
    GroupMemoryBarrierWithGroupSync();

    const uint rows = threads / kRowThreads;
    if (thread < rows) {
        uint row = C.Identity();
        for (uint i = 0; i < kRowThreads; ++i) {
            row = C.Combine(row, s_group[Padded(thread * kRowThreads + i)]);
        }
        s_rows[thread] = row;
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        for (uint row = 0; row < rows; ++row) {
            combined = C.Combine(combined, s_rows[row]);
        }
    }
#else
    const uint lanes = LaneCount();
    const uint waved = C.Wave(value);
    if (LaneIndex() == 0) {
        s_group[thread / lanes] = waved;
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        for (uint wave = 0; wave < (threads + lanes - 1) / lanes; ++wave) {
            combined = C.Combine(combined, s_group[wave]);
        }
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    return combined;
}

uint GroupExclusiveSum(uint thread, uint threads, uint value, out uint total) {
#if ERNIE_HIP
    s_group[Padded(thread)] = value;
    GroupMemoryBarrierWithGroupSync();

    const uint rows = threads / kRowThreads;
    if (thread < rows) {
        uint sum = 0;
        for (uint i = 0; i < kRowThreads; ++i) {
            const uint at = Padded(thread * kRowThreads + i);
            const uint word = s_group[at];
            s_group[at] = sum;
            sum += word;
        }
        s_rows[thread] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        uint sum = 0;
        for (uint row = 0; row < rows; ++row) {
            const uint rowSum = s_rows[row];
            s_rows[row] = sum;
            sum += rowSum;
        }
        s_rows[rows] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    total = s_rows[rows];
    const uint exclusive = s_group[Padded(thread)] + s_rows[thread / kRowThreads];
#else
    const uint lanes = LaneCount();
    const uint before = WavePrefixSum(value);
    if (LaneIndex() == lanes - 1) {
        s_group[thread / lanes] = before + value;
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        uint sum = 0;
        for (uint wave = 0; wave < (threads + lanes - 1) / lanes; ++wave) {
            const uint waveSum = s_group[wave];
            s_group[wave] = sum;
            sum += waveSum;
        }
        s_group[kMaxGroupWaves] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    total = s_group[kMaxGroupWaves];
    const uint exclusive = s_group[thread / lanes] + before;
#endif
    GroupMemoryBarrierWithGroupSync();
    return exclusive;
}
