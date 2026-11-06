// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <dlpack/dlpack.h>

#include "ernie/device/device.h"
#include "ernie/device/resource.h"
#include "ernie/export.h"

namespace ernie::device {

class Device;

enum class Memory : uint8_t {
    Device,
    Upload,
    Readback,
    Shared,
    Virtual,
};

enum class BufferUsage : uint32_t {
    None = 0,
    Indirect = 1u << 0,
    AccelBuildInput = 1u << 1,
    AccelStorage = 1u << 2,
    Interop = 1u << 3,
};

constexpr BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return BufferUsage(uint32_t(a) | uint32_t(b));
}

constexpr bool Has(BufferUsage usage, BufferUsage bits) {
    return (uint32_t(usage) & uint32_t(bits)) != 0;
}

struct BufferDesc {
    uint64_t bytes = 0;
    Memory memory = Memory::Device;
    BufferUsage usage = BufferUsage::None;
    std::string label = "buffer";
};

constexpr uint64_t kBufferAlignment = 256;

constexpr uint64_t kWholeBuffer = ~uint64_t(0);

class BufferRange;

struct ExternalAddress {
    DLDevice device{};
    uint64_t address = 0;
};

class ERNIE_API Buffer : public Resource {
  public:
    virtual ExternalAddress External(DLDeviceType api);

    ~Buffer() override;

    uint64_t Bytes() const {
        return m_bytes;
    }

    uint64_t Address() const {
        return m_address;
    }

    void* Mapped() const {
        return m_mapped;
    }

    Memory GetMemory() const {
        return m_memory;
    }

    BufferUsage Usage() const {
        return m_usage;
    }

    bool Virtual() const {
        return m_memory == Memory::Virtual;
    }

    uint64_t PageBytes() const {
        return m_pageBytes;
    }

    bool FullyBacked() const;

    BufferRange Slice(uint64_t offset, uint64_t bytes = kWholeBuffer);

  protected:
    Buffer(Device& device, const BufferDesc& desc);

    uint64_t m_bytes;
    uint64_t m_address = 0;
    void* m_mapped = nullptr;
    uint64_t m_pageBytes = 0;

  private:
    friend class Device;

    Memory m_memory;
    BufferUsage m_usage;
    bool m_counted = false;
};

class BufferRange {
  public:
    BufferRange(Buffer& buffer) : m_buffer(&buffer), m_bytes(buffer.Bytes()) {}

    Buffer& GetBuffer() const {
        return *m_buffer;
    }

    uint64_t Offset() const {
        return m_offset;
    }

    uint64_t Bytes() const {
        return m_bytes;
    }

    uint64_t Address() const {
        return m_buffer->Address() + m_offset;
    }

    ERNIE_API BufferRange Slice(uint64_t offset, uint64_t bytes = kWholeBuffer) const;

  private:
    friend class Buffer;

    BufferRange(Buffer& buffer, uint64_t offset, uint64_t bytes)
        : m_buffer(&buffer), m_offset(offset), m_bytes(bytes) {}

    Buffer* m_buffer;
    uint64_t m_offset = 0;
    uint64_t m_bytes = 0;
};

template <class T> Ref<Buffer> Device::CreateBuffer(BufferDesc desc, Span<const T> values) {
    return CreateBuffer(std::move(desc), AsBytes(values));
}

template <class T> Ref<Buffer> Device::CreateBuffer(BufferDesc desc, std::span<const T> values) {
    return CreateBuffer(std::move(desc), Span<const T>(values));
}

template <class T> Ref<Buffer> Device::CreateBuffer(BufferDesc desc, const std::vector<T>& values) {
    return CreateBuffer(std::move(desc), Span<const T>(values));
}

}
