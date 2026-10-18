// SPDX-License-Identifier: BSD-3-Clause
/// @file triangulation.cpp
/// @brief Fan triangulation of USD polygons.
#include "ernie/geometry/triangulation.h"

#include <algorithm>
#include <stdexcept>

namespace ernie {

std::vector<std::uint32_t> TriangulateFaces(std::span<const std::int64_t> faceVertexCounts,
                                            std::span<const std::int64_t> faceVertexIndices,
                                            std::size_t points) {
    std::vector<std::uint32_t> triangles;
    std::size_t next = 0;
    for (const std::int64_t count : faceVertexCounts) {
        if (count < 3) {
            throw std::invalid_argument("Every face needs at least 3 vertices");
        }
        const auto size = static_cast<std::size_t>(count);
        if (next + size > faceVertexIndices.size()) {
            throw std::invalid_argument("faceVertexCounts add up to more than faceVertexIndices");
        }
        const std::span<const std::int64_t> face = faceVertexIndices.subspan(next, size);
        if (std::ranges::any_of(face, [&](std::int64_t index) {
                return index < 0 || static_cast<std::size_t>(index) >= points;
            })) {
            throw std::invalid_argument("faceVertexIndices has an index out of range");
        }
        for (std::size_t k = 1; k + 1 < size; ++k) {
            for (const std::int64_t corner : {face[0], face[k], face[k + 1]}) {
                triangles.push_back(static_cast<std::uint32_t>(corner));
            }
        }
        next += size;
    }
    if (next != faceVertexIndices.size()) {
        throw std::invalid_argument("faceVertexCounts add up to fewer than faceVertexIndices");
    }
    return triangles;
}

}  // namespace ernie
