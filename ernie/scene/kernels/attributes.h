// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/changes.h"

namespace ernie {

struct CellsParams {
    StampTarget target;
    Ptr<uint64_t> column;
    Ptr<uint> cells;
    Ptr<uint> rows;
    Ptr<uint64_t> mask;
    uint count;
    uint words;
    uint firstRow;
    uint padding;
};

ERNIE_CHECK_SIZE(CellsParams, 80);

struct CopyRun {
    uint64_t dst;
    uint64_t src;
    uint first;
    uint padding;
};

ERNIE_CHECK_SIZE(CopyRun, 24);

struct CopyWordsParams {
    Ptr<CopyRun> runs;
    uint runCount;
    uint total;
};

ERNIE_CHECK_SIZE(CopyWordsParams, 16);

struct CloneRowsParams {
    Ptr<uint64_t> column;
    Ptr<uint> sources;
    uint first;
    uint count;
    uint copies;
    uint words;
};

ERNIE_CHECK_SIZE(CloneRowsParams, 32);

static const uint kMaxFillWords = 16;

struct FillCellsParams {
    Ptr<uint64_t> column;
    uint first;
    uint count;
    uint words;
    uint padding;
    uint cell[kMaxFillWords];
};

ERNIE_CHECK_SIZE(FillCellsParams, 88);

struct PoseParams {
    StampTarget translateStamps;
    StampTarget orientStamps;
    Ptr<uint64_t> translate;
    Ptr<uint64_t> orient;
    Ptr<uint> rows;
    Ptr<uint> values;
    Ptr<uint> rotations;
    Ptr<uint> sources;
    Ptr<uint64_t> mask;
    uint count;
    uint doubles;
    uint order;
    uint firstRow;
};

ERNIE_CHECK_SIZE(PoseParams, 136);

static const uint kPoseWords = 7;

static const uint kCellsGroup = 64;

#ifdef __SLANG__

Ptr<uint> CellWords(Ptr<uint64_t> pages, uint row, uint words) {
    return Ptr<uint>(pages[row >> kPageRowsLog2]) + (row & (kPageRows - 1)) * words;
}

#endif

}
