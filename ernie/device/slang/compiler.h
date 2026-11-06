// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/kernels/device.h"
#include "ernie/device/program.h"
#include "ernie/device/slang/kernel_library.h"
#include "ernie/util/disk_cache.h"

namespace ernie::device {

inline constexpr const char* kPreludeHeader = "ernie_prelude.h";

using SourceFiles = std::map<std::string, std::string, std::less<>>;

struct KernelShape {
    Size3 groupSize;
    uint32_t paramBytes = 0;
    bool foldable = true;
};

struct KernelCode {
    std::string code;
    KernelShape shape;
    std::vector<std::string> strings;
    std::vector<std::pair<std::string, std::string>> files;
    bool debugInfo = false;
    bool cached = false;

    std::string Encode() const;

    static std::optional<KernelCode> Decode(std::string_view entry);
};

bool HasModule(const KernelLibrary* library, const SourceFiles& sources, std::string_view name);

KernelCode Compile(Backend backend, Traversal traversal, Sampling sampling, bool debugInfo,
                   const KernelLibrary* library, const SourceFiles& sources,
                   std::span<const std::string> modules, std::string_view entry,
                   std::string_view label, const DiskCache& cache, bool narrowWaves = false);

}

namespace ernie {

extern const KernelLibrary kDeviceKernels;

}
