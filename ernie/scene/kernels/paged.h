// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

static const uint kPageRowsLog2 = 8;
static const uint kPageRows = 1u << kPageRowsLog2;

#ifdef __SLANG__

struct Paged<T> {
    Ptr<uint64_t> pages;

    Ptr<T> Address(uint row) {
        return Ptr<T>(pages[row >> kPageRowsLog2]) + (row & (kPageRows - 1));
    }

    __subscript(uint row)->T {
        get {
            return Address(row)[0];
        }
        set {
            Address(row)[0] = newValue;
        }
    }
}

#else

template <class T> struct Paged {
    Ptr<uint64_t> pages;
};

#endif

}
