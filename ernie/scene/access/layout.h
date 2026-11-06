// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string_view>

#include <dlpack/dlpack.h>

#include "ernie/device/device.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/scene/kernels/schema/schema.h"

namespace ernie {

struct ColumnLayout {
    DLDataType dtype;
    uint32_t lanes;
    device::DType type;
};

ERNIE_API ColumnLayout Layout(ColumnType type);

inline bool OnHost(DLDevice device) {
    return device.device_type == kDLCPU || device.device_type == kDLCUDAHost ||
           device.device_type == kDLROCMHost || device.device_type == kDLCUDAManaged;
}

void CheckReadable(device::Device& device, DLDevice tensor);
void WaitOnHost(device::Device& device, Stream producer);
uint32_t TensorRows(const DLTensor& tensor, const ColumnLayout& layout, std::string_view field);
uint32_t TensorItems(const DLTensor& tensor, const ColumnLayout& layout, uint32_t prims,
                     std::string_view field);
const std::byte* TensorBytes(const DLTensor& tensor);

}
