// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>

#include <dlpack/dlpack.h>

#include "ernie/device/operation.h"
#include "ernie/device/stream.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/scene/kernels/schema/schema.h"

namespace ernie {

class ViewState;

enum class MapAccess : uint8_t { Read, Write, ReadWrite };

class ERNIE_API Mapping {
  public:
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& other) noexcept;
    Mapping& operator=(Mapping&& other) noexcept;
    ~Mapping();

    bool Open() const {
        return m_open;
    }

    MapAccess Access() const {
        return m_access;
    }

    bool Aliased() const {
        return m_aliased;
    }

    const device::Tensor& Values() const {
        return m_values;
    }

    const device::Tensor& Rows() const {
        return m_rows;
    }

    const device::Tensor& Offsets() const {
        return m_offsets;
    }

    const device::Operation& Ready() const {
        return m_ready;
    }

    DLTensor DLPack() const {
        return m_values.DLPack();
    }

    device::Operation Close(Stream producer = Stream::None(), const DLTensor* mask = nullptr);

  private:
    friend class UntypedAttribute;

    void Release() noexcept;

    std::shared_ptr<ViewState> m_view;
    ColumnRef m_column;
    MapAccess m_access = MapAccess::Read;
    device::Tensor m_values;
    device::Tensor m_rows;
    device::Tensor m_offsets;
    device::Operation m_ready;
    bool m_aliased = false;
    bool m_open = false;
};

}
