// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <optional>

#include "ernie/algorithms/scan.h"
#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"

namespace ernie::device {
class Device;
}

namespace ernie::algorithms {

class RadixSort {
  public:
    static constexpr uint32_t kMaxItems = Scan::kMaxItems;

    RadixSort(device::Device& device, uint32_t capacity, uint32_t keyBytes = 8);

    void Sort(device::CommandEncoder& encoder, device::BufferRange keys, device::BufferRange values,
              uint32_t count, std::optional<uint32_t> keyBits = std::nullopt) const;

    void Sort(device::CommandEncoder& encoder, device::BufferRange keys, device::BufferRange values,
              device::BufferRange count, uint32_t capacity,
              std::optional<uint32_t> keyBits = std::nullopt) const;

  private:
    void Record(device::CommandEncoder& encoder, device::BufferRange keys,
                device::BufferRange values, uint32_t count, Ptr<uint32_t> liveCount,
                std::optional<uint32_t> keyBits) const;

    uint32_t m_capacity;
    uint32_t m_keyWords;
    Ref<device::Program> m_program;
    device::Kernel* m_countDigits;
    device::Kernel* m_scatter;
    Scan m_scan;
    Ref<device::Buffer> m_keys;
    Ref<device::Buffer> m_values;
    Ref<device::Buffer> m_counts;
    Ref<device::Buffer> m_unlimitedCount;
};

}
