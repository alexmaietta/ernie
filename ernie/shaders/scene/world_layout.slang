// SPDX-License-Identifier: BSD-3-Clause
/// @file world_layout.slang
/// @brief The top levels' records sorted by world on the device, from each instance row's world:
/// `Clear` empties every record, `Scatter` puts each row in its world after the shared ones, and
/// `FillShared` copies the shared rows into every world. Each world's records start `stride`
/// apart; within a world, its own rows come in no fixed order. The parameters are shared with C++
/// (instances.cpp).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/shared.h"
namespace ernie::shaders {
#else
import buffer_ref;
#endif

/// What a record no instance fills holds.
static const uint kNoSource = 0xffffffff;

struct LayoutParameters {
    /// Each row's world, negative for every world.
    BufferRef<int> worlds;
    /// Each world's rows placed so far, then the shared rows'.
    BufferRef<uint> cursors;
    /// The rows in every world.
    BufferRef<uint> sharedRows;
    /// Each record's instance row, world after world.
    BufferRef<uint> sources;
    uint rows;
    uint worldCount;
    uint stride;
    uint sharedCount;
    /// The threads dispatched, each taking every `threads`th item.
    uint threads;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

[shader("compute")]
[numthreads(64, 1, 1)]
void Clear(uint3 id : SV_DispatchThreadID, uniform LayoutParameters parameters) {
    for (uint i = id.x; i < parameters.worldCount * parameters.stride; i += parameters.threads) {
        parameters.sources[i] = kNoSource;
    }
    for (uint i = id.x; i <= parameters.worldCount; i += parameters.threads) {
        parameters.cursors[i] = 0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void Scatter(uint3 id : SV_DispatchThreadID, uniform LayoutParameters parameters) {
    for (uint row = id.x; row < parameters.rows; row += parameters.threads) {
        int world = parameters.worlds[row];
        if (world < 0) {
            parameters.sharedRows[parameters.cursors.AtomicAdd(parameters.worldCount, 1)] = row;
        } else if (uint(world) < parameters.worldCount) {
            uint slot = parameters.sharedCount + parameters.cursors.AtomicAdd(uint(world), 1);
            parameters.sources[uint(world) * parameters.stride + slot] = row;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void FillShared(uint3 id : SV_DispatchThreadID, uniform LayoutParameters parameters) {
    uint count = parameters.worldCount * parameters.sharedCount;
    for (uint i = id.x; i < count; i += parameters.threads) {
        uint world = i / parameters.sharedCount;
        uint shared = i % parameters.sharedCount;
        parameters.sources[world * parameters.stride + shared] = parameters.sharedRows[shared];
    }
}

#endif
