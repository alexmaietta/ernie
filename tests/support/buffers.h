// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"

namespace ernie::test {

template <class T> std::vector<T> Random(size_t count, uint32_t seed) {
    std::mt19937_64 random(seed);
    std::vector<T> values(count);
    for (T& value : values) {
        value = T(random());
    }
    return values;
}

template <class T>
Ref<device::Buffer> Upload(device::Device& device, const std::vector<T>& values) {
    const uint64_t bytes = values.size() * sizeof(T);
    const Ref<device::Buffer> buffer = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(bytes, 16), .usage = device::BufferUsage::AccelBuildInput});
    if (bytes > 0) {
        device.GetQueue().Write(buffer->Slice(0, bytes), values.data());
    }
    return buffer;
}

inline std::vector<std::byte> ReadBytes(device::Device& device, device::BufferRange range) {
    std::vector<std::byte> bytes(range.Bytes());
    device.GetQueue().Read(range, bytes.data());
    return bytes;
}

template <class T>
std::vector<T> Download(device::Device& device, device::Buffer& buffer, size_t count) {
    std::vector<T> values(count);
    if (count > 0) {
        device.GetQueue().Read(buffer.Slice(0, count * sizeof(T)), values.data());
    }
    return values;
}

}
