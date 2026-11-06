// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.algorithms.kernels.group;

#include "ernie/algorithms/kernels/radix_sort.h"

using namespace ernie;

static const uint kSplitBits = 4;
static const uint kSplitBins = 1 << kSplitBits;
static const uint kCountBits = 16;
static const uint kCountMask = (1u << kCountBits) - 1;
static const uint kCountWords = kSplitBins / (32 / kCountBits);

groupshared uint64_t s_keys[kSortTileItems];
groupshared uint s_places[kSortTileItems];
groupshared uint s_counts[kCountWords * kSortThreads];
groupshared uint s_bins[kDigitBins];

uint LiveCount(SortParams p) {
    return min(p.deviceCount[0], p.count);
}

uint Digit(uint64_t key, uint shift, uint mask) {
    return uint(key >> shift) & mask;
}

uint64_t LoadKey(uint* keys, uint index, uint words) {
    if (words == 1) {
        return keys[index];
    }
    return uint64_t(keys[2 * index]) | (uint64_t(keys[2 * index + 1]) << 32);
}

void StoreKey(uint* keys, uint index, uint words, uint64_t key) {
    if (words == 1) {
        keys[index] = uint(key);
    } else {
        keys[2 * index] = uint(key);
        keys[2 * index + 1] = uint(key >> 32);
    }
}

[shader("compute")]
[numthreads(kSortThreads, 1, 1)]
void CountDigits(uniform SortParams p) {
    const uint tile = LinearGroupIndex();
    if (tile >= p.tiles) {
        return;
    }

    const uint thread = WaveOrderedThreadIndex();
    const uint count = LiveCount(p);
    s_bins[thread] = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < kSortItems; ++i) {
        const uint item = tile * kSortTileItems + i * kSortThreads + thread;
        if (item < count) {
            InterlockedAdd(s_bins[Digit(LoadKey(p.keys, item, p.keyWords), p.shift, p.mask)], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    p.counts[thread * p.tiles + tile] = s_bins[thread];
}

uint CountWord(uint bin, uint thread) {
    return (bin % kCountWords) * kSortThreads + thread;
}

uint CountShift(uint bin) {
    return (bin / kCountWords) * kCountBits;
}

uint ReadCount(uint bin, uint thread) {
    return (s_counts[CountWord(bin, thread)] >> CountShift(bin)) & kCountMask;
}

void Split(uint thread, uint shift, uint mask, uint bit, uint valid) {
    [ForceUnroll]
    for (uint w = 0; w < kCountWords; ++w) {
        s_counts[w * kSortThreads + thread] = 0;
    }

    uint64_t keys[kSortItems];
    uint places[kSortItems];
    uint bins[kSortItems];
    uint before[kSortItems];
    [ForceUnroll]
    for (uint i = 0; i < kSortItems; ++i) {
        keys[i] = s_keys[thread * kSortItems + i];
        places[i] = s_places[thread * kSortItems + i];
        const uint digit = places[i] < valid ? Digit(keys[i], shift, mask) : kDigitBins - 1;
        bins[i] = (digit >> bit) % kSplitBins;
        before[i] = ReadCount(bins[i], thread);
        s_counts[CountWord(bins[i], thread)] += 1u << CountShift(bins[i]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint words[kCountWords];
    uint sum = 0;
    [ForceUnroll]
    for (uint w = 0; w < kCountWords; ++w) {
        words[w] = s_counts[thread * kCountWords + w];
        sum += words[w];
    }
    uint total;
    uint prefix = GroupExclusiveSum(thread, kSortThreads, sum, total);
    [ForceUnroll]
    for (uint w = 0; w < kCountWords; ++w) {
        s_counts[thread * kCountWords + w] = prefix;
        prefix += words[w];
    }
    GroupMemoryBarrierWithGroupSync();

    uint to[kSortItems];
    [ForceUnroll]
    for (uint i = 0; i < kSortItems; ++i) {
        const uint lowerBinItems = bins[i] >= kCountWords ? total & kCountMask : 0;
        to[i] = ReadCount(bins[i], thread) + before[i] + lowerBinItems;
    }
    GroupMemoryBarrierWithGroupSync();

    [ForceUnroll]
    for (uint i = 0; i < kSortItems; ++i) {
        s_keys[to[i]] = keys[i];
        s_places[to[i]] = places[i];
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(kSortThreads, 1, 1)]
void Scatter(uniform SortParams p) {
    const uint tile = LinearGroupIndex();
    const uint start = tile * kSortTileItems;
    const uint count = LiveCount(p);
    if (start >= count) {
        return;
    }

    const uint thread = WaveOrderedThreadIndex();
    const uint valid = min(kSortTileItems, count - start);
    s_bins[thread] = 0;
    GroupMemoryBarrierWithGroupSync();

    [ForceUnroll]
    for (uint i = 0; i < kSortItems; ++i) {
        const uint place = i * kSortThreads + thread;
        const uint64_t key = place < valid ? LoadKey(p.keys, start + place, p.keyWords) : 0;
        s_keys[place] = key;
        s_places[place] = place;
        if (place < valid) {
            InterlockedAdd(s_bins[Digit(key, p.shift, p.mask)], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint total;
    const uint inTile = GroupExclusiveSum(thread, kSortThreads, s_bins[thread], total);
    s_bins[thread] = p.counts[thread * p.tiles + tile] - inTile;

    [ForceUnroll]
    for (uint bit = 0; bit < kDigitBits; bit += kSplitBits) {
        Split(thread, p.shift, p.mask, bit, valid);
    }

    [ForceUnroll]
    for (uint i = 0; i < kSortItems; ++i) {
        const uint place = i * kSortThreads + thread;
        if (place < valid) {
            const uint64_t key = s_keys[place];
            const uint to = s_bins[Digit(key, p.shift, p.mask)] + place;
            StoreKey(p.keysOut, to, p.keyWords, key);
            p.valuesOut[to] = p.values[start + s_places[place]];
        }
    }
}
