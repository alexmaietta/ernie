// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>

#include <dlpack/dlpack.h>

#include "ernie/device/operation.h"
#include "ernie/device/stream.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/scene/access/attribute.h"

namespace ernie {

class ERNIE_API Poses {
  public:
    explicit Poses(std::shared_ptr<ViewState> view);

    uint32_t Size() const {
        return m_translate.Size();
    }

    device::Operation Write(const DLTensor& source, Stream producer = Stream::None(),
                            QuaternionOrder order = QuaternionOrder::Xyzw,
                            const DLTensor* rows = nullptr, const DLTensor* mask = nullptr);
    device::Operation Write(const DLTensor& translation, const DLTensor& rotation,
                            Stream producer = Stream::None(),
                            QuaternionOrder order = QuaternionOrder::Xyzw,
                            const DLTensor* rows = nullptr, const DLTensor* mask = nullptr);
    device::Operation Write(const device::Tensor& source, Stream producer = Stream::None(),
                            QuaternionOrder order = QuaternionOrder::Xyzw,
                            const device::Tensor& rows = {}, const device::Tensor& mask = {});
    device::Operation Write(const device::Tensor& translation, const device::Tensor& rotation,
                            Stream producer = Stream::None(),
                            QuaternionOrder order = QuaternionOrder::Xyzw,
                            const device::Tensor& rows = {}, const device::Tensor& mask = {});
    device::Tensor Read(device::DType type = device::DType::Float32,
                        Stream consumer = Stream::Host(),
                        QuaternionOrder order = QuaternionOrder::Xyzw) const;

  private:
    device::Operation Dispatch(const device::Tensor& values, const device::Tensor* rotations,
                               Stream producer, QuaternionOrder order, const device::Tensor& rows,
                               const device::Tensor& mask);

    UntypedAttribute m_translate;
    UntypedAttribute m_orient;
};

}
