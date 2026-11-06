// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <imgui.h>

#include "ernie/device/device.h"
#include "ernie/device/operation.h"
#include "ernie/device/tensor.h"

struct GLFWwindow;

namespace ernie::app {

struct Extent {
    uint32_t width = 0;
    uint32_t height = 0;
};

class Presenter {
  public:
    virtual ~Presenter() = default;

    virtual bool Begin() = 0;
    virtual Extent Size() const = 0;
    virtual void Viewport(ImDrawList& list, const device::Tensor& pixels, ImVec2 min,
                          ImVec2 max) = 0;
    virtual void Image(ImDrawList& list, const device::Tensor& pixels, ImVec2 min, ImVec2 max,
                       float opacity) = 0;
    virtual device::Operation End(Span<const device::Operation> after) = 0;
    virtual std::vector<uint8_t> ReadPixels(const device::Operation& after) = 0;
};

std::unique_ptr<Presenter> CreateVulkanPresenter(device::Device& device, GLFWwindow* window,
                                                 Extent size, bool vsync);

}
