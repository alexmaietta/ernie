// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "ernie/bvh/kernels/bvh.h"

namespace ernie::bvh {

using Triangle = BvhCorners;

std::vector<std::byte> BuildReference(std::span<const Triangle> triangles);

std::vector<std::byte> BuildReference(std::span<const BvhBox> boxes);

std::vector<std::byte> BuildReference(std::span<const InstanceRecord> instances,
                                      std::span<const BvhBox> bounds);

}
