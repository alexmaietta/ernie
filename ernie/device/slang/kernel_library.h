// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <span>

namespace ernie {

struct EmbeddedKernel {
    const char* path;
    const unsigned char* bytes;
    size_t size;
};

struct KernelLibrary {
    std::span<const EmbeddedKernel> files;
    std::span<const KernelLibrary* const> uses;
};

}
