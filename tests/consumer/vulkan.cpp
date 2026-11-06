// SPDX-License-Identifier: BSD-3-Clause
#include <cstdio>
#include <exception>

#include "ernie/device/vulkan/native.h"
#include "ernie/ernie.h"

namespace device = ernie::device;

int main() {
    try {
        for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
            if (adapter.backend != device::Backend::Vulkan || !adapter.unsupported.empty()) {
                continue;
            }
            device::DeviceDesc desc;
            desc.adapter = adapter.alias;
            const ernie::Ref<device::Device> created = device::Device::Create(desc);
            const device::vulkan::Native native = device::vulkan::NativeHandles(*created);
            if (native.device == VK_NULL_HANDLE || native.getInstanceProcAddr == nullptr) {
                std::fprintf(stderr, "%s has no native Vulkan handles\n", adapter.alias.c_str());
                return 1;
            }
            device::vulkan::Exclusive(*created, [](VkQueue) {});
            std::printf("%s shares its Vulkan device\n", adapter.alias.c_str());
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
