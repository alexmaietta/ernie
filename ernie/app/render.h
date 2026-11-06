// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>

#include "ernie/app/cli.h"
#include "ernie/device/ref.h"

namespace ernie::app {

Ref<Scene> DefaultScene(device::Device& device, uint32_t worlds, bool waitForKernels);

std::string ChooseProduct(Scene& scene, const Command& command);

int RenderFrames(const Ref<Scene>& scene, const std::string& product, const Command& command);

int Benchmark(const Ref<Scene>& scene, const std::string& product, const Command& command);

}
