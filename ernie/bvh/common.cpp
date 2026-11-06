// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/common.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <variant>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/util/log.h"

namespace ernie::bvh {

using device::Buffer;
using device::BufferRange;
using device::Device;
using device::Traversal;

const char* FormatName(Traversal format) {
    return format == Traversal::Hardware ? "hardware" : "software";
}

uint32_t Kind(const Geometry& geometry) {
    return std::holds_alternative<device::AccelBoxes>(geometry) ? kBvhBoxes : kBvhTriangles;
}

uint32_t PrimitiveCount(const Geometry& geometry) {
    return std::visit([](const auto& held) { return held.count; }, geometry);
}

device::AccelInput ToAccelInput(const Geometry& geometry) {
    return std::visit([](const auto& held) -> device::AccelInput { return held; }, geometry);
}

std::vector<Ref<Buffer>> GeometryBuffers(const Geometry& geometry) {
    std::vector<Ref<Buffer>> buffers;
    device::ForEachBuffer(geometry,
                          [&](BufferRange range) { buffers.emplace_back(&range.GetBuffer()); });
    return buffers;
}

uint64_t StorageBytes(uint32_t count, uint32_t kind) {
    return BvhLeavesOffset(count) + uint64_t(count) * BvhRecordWords(kind) * 16;
}

uint64_t Total(std::span<const uint32_t> capacities) {
    return std::accumulate(capacities.begin(), capacities.end(), uint64_t(0));
}

std::vector<uint32_t> Offsets(std::span<const uint32_t> capacities) {
    std::vector<uint32_t> offsets = {0};
    for (const uint32_t capacity : capacities) {
        offsets.push_back(offsets.back() + capacity);
    }
    return offsets;
}

Buffer& Reserve(Device& device, Ref<Buffer>& buffer, const device::BufferDesc& desc) {
    if (!buffer || buffer->Bytes() < desc.bytes) {
        Log(LogLevel::Debug, "{} grows from {} to {} bytes", desc.label,
            buffer ? buffer->Bytes() : 0, desc.bytes);
        buffer = device.CreateBuffer(desc);
    }
    return *buffer;
}

void RequireInstances(const char* what, const Tlas& tlas, BufferRange instances) {
    const uint64_t total = Total(tlas.Capacities());
    const uint64_t bytes = total * sizeof(InstanceRecord);
    if (instances.Bytes() < bytes) {
        throw Error("{} of a TLAS of {} instances from {} bytes; it takes {}", what, total,
                    instances.Bytes(), bytes);
    }
}

std::vector<uint32_t> SelectWorlds(const char* what, const Tlas& tlas,
                                   std::span<const uint32_t> worlds) {
    if (worlds.empty()) {
        std::vector<uint32_t> all(tlas.Count());
        std::iota(all.begin(), all.end(), 0u);
        return all;
    }
    std::vector<bool> seen(tlas.Count(), false);
    for (const uint32_t world : worlds) {
        if (world >= tlas.Count()) {
            throw Error("{} of world {} of a TLAS of {} worlds", what, world, tlas.Count());
        }
        if (seen[world]) {
            throw Error("{} of world {} twice", what, world);
        }
        seen[world] = true;
    }
    return {worlds.begin(), worlds.end()};
}

void RequireRefittable(const char* what, const device::AccelInput& input) {
    if (!device::Refittable(input)) {
        throw Error("{} of a BLAS that wasn't made refittable", what);
    }
}

Ref<Buffer> Tables::Upload(Device& device, std::string label) const {
    return device.CreateBuffer({.label = std::move(label)}, std::span<const std::byte>(m_bytes));
}

}
