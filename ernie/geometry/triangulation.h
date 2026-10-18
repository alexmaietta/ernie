// SPDX-License-Identifier: BSD-3-Clause
/// @file triangulation.h
/// @brief USD polygons as triangles, for bottom levels built over meshes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ernie {

/// @brief Triangulates USD polygons into fans.
/// @param faceVertexCounts  Vertices a face.
/// @param faceVertexIndices Point indices, face by face.
/// @param points            How many points the indices may name.
/// @return Three point indices a triangle.
/// @throws std::invalid_argument for a face of under 3 vertices, counts that do not add up to the
/// indices, or an index out of range.
std::vector<std::uint32_t> TriangulateFaces(std::span<const std::int64_t> faceVertexCounts,
                                            std::span<const std::int64_t> faceVertexIndices,
                                            std::size_t points);

}  // namespace ernie
