// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/stage/columns/kernels/cells.h"
#include "ernie/stage/columns/kernels/doubles.h"

using namespace ernie;

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void CopyWords(uniform CopyWordsParams p) {
    const uint thread = LinearThreadIndex();
    if (thread >= p.total) {
        return;
    }
    uint low = 0;
    uint high = p.runCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (p.runs[middle].first <= thread) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const CopyRun run = p.runs[low];
    const uint word = thread - run.first;
    Ptr<uint>(run.dst)[word] = Ptr<uint>(run.src)[word];
}

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void ScatterCells(uniform CellsParams p) {
    const uint thread = LinearThreadIndex();
    if (thread >= p.count * p.words) {
        return;
    }
    const uint item = thread / p.words;
    const uint word = thread - item * p.words;
    const uint row = ViewRow(p.rows, p.firstRow, item);
    if (row == kNoRow || !Selected(p.mask, item)) {
        return;
    }
    CellWords(p.column, row, p.words)[word] = p.cells[thread];
    if (word == 0) {
        StampRow(p.target, row);
    }
}

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void GatherCells(uniform CellsParams p) {
    const uint thread = LinearThreadIndex();
    if (thread >= p.count * p.words) {
        return;
    }
    const uint item = thread / p.words;
    const uint word = thread - item * p.words;
    const uint row = ViewRow(p.rows, p.firstRow, item);
    if (row == kNoRow) {
        return;
    }
    p.cells[thread] = CellWords(p.column, row, p.words)[word];
}

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void CloneRows(uniform CloneRowsParams p) {
    const uint thread = LinearThreadIndex();
    if (thread >= p.copies * p.count * p.words) {
        return;
    }
    const uint word = thread % p.words;
    const uint item = thread / p.words;
    const uint source = item % p.count;
    const uint copy = item / p.count;
    CellWords(p.column, p.first + copy * p.count + source, p.words)[word] =
        CellWords(p.column, p.sources[source], p.words)[word];
}

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void FillCells(uniform FillCellsParams p) {
    const uint thread = LinearThreadIndex();
    if (thread >= p.count * p.words) {
        return;
    }
    const uint item = thread / p.words;
    const uint word = thread - item * p.words;
    CellWords(p.column, p.first + item, p.words)[word] = p.cell[word];
}

Ptr<uint> PoseWords(PoseParams p, uint row, uint component) {
    if (p.order != 0 && component >= 3) {
        component = component == 3 ? 6 : component - 1;
    }
    return component < 3 ? CellWords(p.translate, row, 6) + component * 2
                         : CellWords(p.orient, row, 8) + (component - 3) * 2;
}

uint2 PoseInput(PoseParams p, uint item, uint component) {
    const uint source = p.sources != nullptr ? p.sources[item] : item;
    Ptr<uint> values = p.values;
    uint at = source * kPoseWords + component;
    if (p.rotations != nullptr) {
        at = component < 3 ? source * 3 + component : source * 4 + component - 3;
        values = component < 3 ? p.values : p.rotations;
    }
    return p.doubles != 0 ? uint2(values[at * 2], values[at * 2 + 1])
                          : DoubleBits(asfloat(values[at]));
}

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void WritePoses(uniform PoseParams p) {
    const uint item = LinearThreadIndex();
    if (item >= p.count || !Selected(p.mask, item)) {
        return;
    }
    const uint row = ViewRow(p.rows, p.firstRow, item);
    if (row == kNoRow) {
        return;
    }
    for (uint component = 0; component < kPoseWords; ++component) {
        const uint2 words = PoseInput(p, item, component);
        Ptr<uint> cell = PoseWords(p, row, component);
        cell[0] = words.x;
        cell[1] = words.y;
    }
    StampRow(p.translateStamps, row);
    StampRow(p.orientStamps, row);
}

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void ReadPoses(uniform PoseParams p) {
    const uint item = LinearThreadIndex();
    if (item >= p.count) {
        return;
    }
    const uint row = ViewRow(p.rows, p.firstRow, item);
    for (uint component = 0; component < kPoseWords; ++component) {
        const uint at = item * kPoseWords + component;
        uint2 words = uint2(0, 0);
        if (row != kNoRow) {
            Ptr<uint> cell = PoseWords(p, row, component);
            words = uint2(cell[0], cell[1]);
        }
        if (p.doubles != 0) {
            p.values[at * 2] = words.x;
            p.values[at * 2 + 1] = words.y;
        } else {
            p.values[at] = asuint(Narrow(words.x, words.y));
        }
    }
}

[shader("compute")]
[numthreads(kCellsGroup, 1, 1)]
void GatherRows(uniform GatherRowsParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const uint row = p.rows[index];
    for (uint word = 0; word < p.words; ++word) {
        p.out[index * p.words + word] = row == kNoRow ? 0 : p.source[row * p.words + word];
    }
}
