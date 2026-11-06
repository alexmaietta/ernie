// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/export.h"
#include "ernie/util/version.h"

namespace ernie::device {

enum class Backend : uint8_t {
    Vulkan,
    Cuda,
    Hip,
    Metal,
    Cpu,
};

ERNIE_API std::string_view BackendName(Backend backend);

enum class Traversal : uint8_t {
    Auto,
    Hardware,
    Software,
};

enum class Sampling : uint8_t {
    Auto,
    Hardware,
    Software,
};

enum class AdapterType : uint8_t { Discrete, Integrated, Virtual, Cpu, Other };

struct Features {
    bool hardwareTraversal = false;
    bool unifiedMemory = false;
    bool hostMappedMemory = false;
    bool virtualMemory = false;
    bool externalMemory = false;
    bool hardwareTextures = false;
    bool blockCompression = false;
    bool anisotropicFiltering = false;
    bool float64 = false;
    bool atomicFloat32Add = false;
    bool atomicInt64 = false;
    bool shaderClock = false;
    bool kernelStatistics = false;
    bool presentation = false;
    uint32_t waveSizeMin = 32;
    uint32_t waveSizeMax = 32;
};

struct Limits {
    uint64_t maxBufferBytes = 0;
    uint32_t maxGroupSize = 0;
    uint32_t maxParamBytes = 0;
    uint64_t maxReservedBytes = 0;
    uint32_t maxTextures = 0;
    float maxAnisotropy = 1;
};

struct AdapterInfo {
    Backend backend = Backend::Vulkan;
    std::string name;
    std::string alias;
    uint32_t ordinal = 0;
    AdapterType type = AdapterType::Other;
    std::array<uint8_t, 16> uuid{};
    std::string architecture;
    Version apiVersion;
    Version driverVersion;
    uint64_t memoryBytes = 0;
    Features features;
    Limits limits;
    std::string unsupported;
};

ERNIE_API std::vector<AdapterInfo> EnumerateAdapters();

}
