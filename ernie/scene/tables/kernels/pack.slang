// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/pack.h"

using namespace ernie;

[shader("compute")]
[numthreads(kPackGroup, 1, 1)]
void PackWords(uniform PackParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const PackEntry entry = p.entries[index];
    if (entry.pages == 0) {
        p.words[entry.destination] = entry.value;
        return;
    }
    const Ptr<uint64_t> pages = Ptr<uint64_t>(entry.pages);
    const Ptr<uint> source = Ptr<uint>(pages[entry.row >> kPageRowsLog2]) +
                             (entry.row & (kPageRows - 1)) * entry.words;
    for (uint word = 0; word < entry.words; ++word) {
        p.words[entry.destination + word] = source[word];
    }
}
