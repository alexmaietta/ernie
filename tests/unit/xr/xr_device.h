// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <optional>

#include "ernie/device/device.h"
#include "ernie/util/error.h"

namespace ernie::xr::test {

inline device::Device* RenderDevice() {
    static const std::optional<Ref<device::Device>> device =
        []() -> std::optional<Ref<device::Device>> {
        try {
            return device::Device::Create();
        } catch (const UnsupportedError&) {
            return std::nullopt;
        }
    }();
    return device ? device->Get() : nullptr;
}

}
