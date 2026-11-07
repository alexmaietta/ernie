// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/operation.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/catalog/prims.h"

namespace ernie {

std::vector<uint32_t> TableRows(const Prims& prims, std::span<const uint32_t> indices,
                                SchemaId table);

class Readback {
  public:
    explicit Readback(device::Device& device) : m_device(device) {}

    device::Download ReadAsync(const Ref<device::Buffer>& source, uint32_t stride,
                               std::span<const uint32_t> rows);

    template <class T>
    std::vector<T> Read(const Ref<device::Buffer>& source, std::span<const uint32_t> rows) {
        static_assert(sizeof(T) % sizeof(uint32_t) == 0);
        std::vector<T> values(rows.size());
        const device::Download download = ReadAsync(source, sizeof(T), rows);
        download.Wait();
        if (download.Bytes().size() == values.size() * sizeof(T) && !values.empty()) {
            std::memcpy(values.data(), download.Bytes().data(), values.size() * sizeof(T));
        }
        return values;
    }

  private:
    device::Device& m_device;
    Ref<device::Program> m_program;
};

}
