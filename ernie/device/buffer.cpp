// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/buffer.h"

#include "ernie/device/device.h"
#include "ernie/util/error.h"

namespace ernie::device {

Buffer::Buffer(Device& device, const BufferDesc& desc)
    : Resource(device, desc.label), m_bytes(desc.bytes), m_memory(desc.memory),
      m_usage(desc.usage) {}

Buffer::~Buffer() {
    if (m_counted) {
        GetDevice().CountMemory(*this, -1);
    }
}

bool Buffer::FullyBacked() const {
    return !Virtual() || !GetDevice().Adapter().features.virtualMemory;
}

ExternalAddress Buffer::External(DLDeviceType api) {
    const AdapterInfo& adapter = GetDevice().Adapter();
    if ((adapter.backend == Backend::Cuda && api == kDLCUDA) ||
        (adapter.backend == Backend::Hip && api == kDLROCM)) {
        return {{api, int32_t(adapter.ordinal)}, Address()};
    }
    throw UnsupportedError("Buffer \"{}\" on {} isn't shared with device type {}", Label(),
                           adapter.alias, int(api));
}

BufferRange Buffer::Slice(uint64_t offset, uint64_t bytes) {
    return BufferRange(*this).Slice(offset, bytes);
}

BufferRange BufferRange::Slice(uint64_t offset, uint64_t bytes) const {
    if (offset > m_bytes || (bytes != kWholeBuffer && bytes > m_bytes - offset)) {
        throw Error("bytes {} to {} of buffer \"{}\" are past the end of its range, bytes {} to {}",
                    m_offset + offset,
                    m_offset + (bytes == kWholeBuffer ? m_bytes : offset + bytes),
                    m_buffer->Label(), m_offset, m_offset + m_bytes);
    }
    return {*m_buffer, m_offset + offset, bytes == kWholeBuffer ? m_bytes - offset : bytes};
}

}
