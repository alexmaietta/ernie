// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>

#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/export.h"
#include "ernie/util/span.h"

namespace ernie::device {

class Buffer;
class Device;
class Queue;

struct ERNIE_API Operation {
    Queue* queue = nullptr;
    uint64_t value = 0;

    bool Ready() const;

    void Wait() const;

    void Wait(Stream stream) const;
};

class ERNIE_API Download : public Operation {
  public:
    Download();
    ~Download();
    Download(const Download& other);
    Download(Download&& other) noexcept;
    Download& operator=(const Download& other);
    Download& operator=(Download&& other) noexcept;

    Span<const std::byte> Bytes() const;

  private:
    friend class Queue;

    Ref<Device> m_device;
    Ref<Buffer> m_staging;
    uint64_t m_offset = 0;
    uint64_t m_bytes = 0;
};

}
