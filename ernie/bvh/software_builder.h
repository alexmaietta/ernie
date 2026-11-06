// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "ernie/bvh/builder.h"
#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"

namespace ernie::device {
class Device;
}

namespace ernie::bvh {

class SoftwareBlas final : public Blas {
  public:
    static constexpr device::Traversal kFormat = device::Traversal::Software;

    SoftwareBlas(device::Device& device, const Geometry& geometry);

    uint64_t Handle() const override {
        return m_storage->Address();
    }

    device::Buffer& Storage() const {
        return *m_storage;
    }

    const Geometry& GetGeometry() const {
        return m_geometry;
    }

  private:
    Geometry m_geometry;
    std::vector<Ref<device::Buffer>> m_buffers;
    Ref<device::Buffer> m_storage;
};

class SoftwareTlas final : public Tlas {
  public:
    static constexpr device::Traversal kFormat = device::Traversal::Software;

    SoftwareTlas(device::Device& device, std::span<const uint32_t> capacities);

    device::Buffer& Storage() const {
        return *m_storage;
    }

    device::BufferRange World(uint32_t world) const;

  private:
    Ref<device::Buffer> m_storage;
};

}
