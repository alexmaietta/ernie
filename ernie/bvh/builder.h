// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "ernie/bvh/kernels/bvh.h"
#include "ernie/device/accel.h"
#include "ernie/device/adapter.h"
#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/ref.h"
#include "ernie/export.h"

namespace ernie::device {
class Device;
}

namespace ernie::bvh {

using Geometry = std::variant<device::AccelTriangles, device::AccelBoxes>;

class ERNIE_API Blas : public RefCounted {
  public:
    device::Traversal Format() const {
        return m_format;
    }

    virtual uint64_t Handle() const = 0;

  protected:
    explicit Blas(device::Traversal format) : m_format(format) {}

    ~Blas() override = default;

  private:
    device::Traversal m_format;
};

class ERNIE_API Tlas : public RefCounted {
  public:
    device::Traversal Format() const {
        return m_format;
    }

    uint32_t Count() const {
        return uint32_t(m_capacities.size());
    }

    std::span<const uint32_t> Capacities() const {
        return m_capacities;
    }

    uint64_t Handle(uint32_t world = 0) const;

    device::Buffer& Handles() const {
        return *m_table;
    }

  protected:
    Tlas(device::Traversal format, std::span<const uint32_t> capacities);
    ~Tlas() override = default;

    void SetHandles(device::Device& device, std::vector<uint64_t> handles);

  private:
    device::Traversal m_format;
    std::vector<uint32_t> m_capacities;
    std::vector<uint64_t> m_handles;
    Ref<device::Buffer> m_table;
};

class ERNIE_API Builder {
  public:
    static std::unique_ptr<Builder> Create(device::Device& device,
                                           device::Traversal format = device::Traversal::Auto);

    virtual ~Builder() = default;
    Builder(const Builder&) = delete;
    Builder& operator=(const Builder&) = delete;

    virtual device::Traversal Format() const = 0;

    virtual void Prepare() {}

    virtual Ref<Blas> CreateBlas(const Geometry& geometry) = 0;
    virtual Ref<Tlas> CreateTlas(std::span<const uint32_t> capacities) = 0;

    virtual void Build(device::CommandEncoder& encoder, std::span<const Ref<Blas>> blases) = 0;
    virtual void Refit(device::CommandEncoder& encoder, std::span<const Ref<Blas>> blases) = 0;
    virtual void Build(device::CommandEncoder& encoder, Tlas& tlas, device::BufferRange instances,
                       std::optional<device::BufferRange> counts = std::nullopt,
                       std::span<const uint32_t> worlds = {}) = 0;
    virtual void Refit(device::CommandEncoder& encoder, Tlas& tlas, device::BufferRange instances,
                       std::optional<device::BufferRange> counts = std::nullopt,
                       std::span<const uint32_t> worlds = {}) = 0;

  protected:
    explicit Builder(device::Device& device) : m_device(device) {}

    device::Device& m_device;
};

}
