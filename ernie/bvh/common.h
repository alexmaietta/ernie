// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "ernie/bvh/builder.h"
#include "ernie/device/accel.h"
#include "ernie/device/buffer.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/util/error.h"

namespace ernie::bvh {

std::unique_ptr<Builder> CreateSoftwareBuilder(device::Device& device);
std::unique_ptr<Builder> CreateHardwareBuilder(device::Device& device);

uint32_t Kind(const Geometry& geometry);
uint32_t PrimitiveCount(const Geometry& geometry);
device::AccelInput ToAccelInput(const Geometry& geometry);
std::vector<Ref<device::Buffer>> GeometryBuffers(const Geometry& geometry);
uint64_t StorageBytes(uint32_t count, uint32_t kind);

uint64_t Total(std::span<const uint32_t> capacities);
std::vector<uint32_t> Offsets(std::span<const uint32_t> capacities);

device::Buffer& Reserve(device::Device& device, Ref<device::Buffer>& buffer,
                        const device::BufferDesc& desc);

void RequireInstances(const char* what, const Tlas& tlas, device::BufferRange instances);
std::vector<uint32_t> SelectWorlds(const char* what, const Tlas& tlas,
                                   std::span<const uint32_t> worlds);
void RequireRefittable(const char* what, const device::AccelInput& input);

const char* FormatName(device::Traversal format);

template <class Concrete, class Structure>
const Concrete& RequireFormat(const char* what, const Structure& structure) {
    if (structure.Format() != Concrete::kFormat) {
        throw Error("{} of a {} {} by a builder of {} ones", what, FormatName(structure.Format()),
                    std::derived_from<Concrete, Blas> ? "BLAS" : "TLAS",
                    FormatName(Concrete::kFormat));
    }
    return static_cast<const Concrete&>(structure);
}

class Tables {
  public:
    template <class T> uint64_t Add(const std::vector<T>& values) {
        const uint64_t at = RoundUp(m_bytes.size(), alignof(T));
        m_bytes.resize(at + values.size() * sizeof(T));
        std::memcpy(m_bytes.data() + at, values.data(), values.size() * sizeof(T));
        return at;
    }

    Ref<device::Buffer> Upload(device::Device& device, std::string label) const;

  private:
    std::vector<std::byte> m_bytes;
};

}
