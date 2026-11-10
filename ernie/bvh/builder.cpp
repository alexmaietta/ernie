// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/builder.h"

#include <memory>
#include <utility>
#include <vector>

#include "ernie/bvh/common.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie::bvh {

using device::Device;
using device::Traversal;

Tlas::Tlas(Traversal format, std::span<const uint32_t> capacities)
    : m_format(format), m_capacities(capacities.begin(), capacities.end()) {
    if (capacities.empty()) {
        throw Error("CreateTlas of no worlds; a TLAS has at least one");
    }
    Log(LogLevel::Debug, "created a {} TLAS of {} worlds holding up to {} instances",
        FormatName(format), capacities.size(), Total(capacities));
}

uint64_t Tlas::Handle(uint32_t world) const {
    if (world >= Count()) {
        throw Error("World {} of a TLAS of {} worlds", world, Count());
    }
    return m_handles[world];
}

void Tlas::SetHandles(Device& device, std::vector<uint64_t> handles) {
    m_handles = std::move(handles);
    m_table = device.CreateBuffer(
        {.bytes = m_handles.size() * sizeof(uint64_t), .label = "tlas handles"});
    device.GetQueue().Write(*m_table, m_handles.data());
}

void Builder::SubmitRefit(device::CommandEncoder&& encoder, Tlas& tlas,
                          device::BufferRange instances, std::optional<device::BufferRange> counts,
                          std::span<const uint32_t> worlds) {
    Refit(encoder, tlas, instances, counts, worlds);
    m_device.GetQueue().Submit(encoder.Finish());
}

std::unique_ptr<Builder> Builder::Create(Device& device, Traversal format) {
    if (format == Traversal::Auto) {
        format = device.GetTraversal();
    }
    if (format == Traversal::Hardware && device.GetTraversal() != Traversal::Hardware) {
        throw UnsupportedError("Builder of hardware acceleration structures on {}, which "
                               "traverses rays in software",
                               device.Adapter().alias);
    }
    Log(LogLevel::Debug, "created a {} BVH builder for {}", FormatName(format),
        device.Adapter().alias);
    if (format == Traversal::Hardware) {
        return CreateHardwareBuilder(device);
    }
    return CreateSoftwareBuilder(device);
}

}
