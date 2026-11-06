// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/program.h"
#include "ernie/util/error.h"

namespace ernie::device {

template <class T>
Ptr<T> RequireRange(const Program& program, const char* what, BufferRange range, uint64_t count) {
    program.GetDevice().Require(range.GetBuffer(), what);
    if (range.Bytes() < count * sizeof(T)) {
        throw Error("{} needs {} bytes of buffer \"{}\"; the range it was given has {}", what,
                    count * sizeof(T), range.GetBuffer().Label(), range.Bytes());
    }
    return {range.Address()};
}

}
