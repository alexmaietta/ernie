// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ernie {

/// Triangle mesh geometry in a mesh's own frame.
struct MeshData {
    std::vector<float> points;           ///< x, y, z per vertex.
    std::vector<std::uint32_t> indices;  ///< Three vertex indices per triangle.
    std::vector<float> normals;          ///< Empty, or x, y, z per vertex.
    std::vector<float> colors;           ///< Empty, or linear r, g, b per vertex.
    /// Empty, or texture coordinates s, t per triangle corner (per entry of `indices`), so
    /// face-varying coordinates need no split vertices.
    std::vector<float> uvs;
};

/// Where each corner of `TriangulateFaces`' triangles comes from: its face and its position in
/// `faceVertexIndices`, for primvars with uniform and faceVarying interpolation.
struct TriangleCorners {
    std::vector<std::uint32_t> faces;
    std::vector<std::uint32_t> faceVertices;
};

/// The corners of the fans `TriangulateFaces` makes of faces of `faceVertexCounts` (unchecked:
/// call after it).
TriangleCorners TriangulatedCorners(std::span<const std::int64_t> faceVertexCounts);

/// Triangulates polygons given as USD `faceVertexCounts` and `faceVertexIndices` into fans.
/// @throws std::invalid_argument if a face has fewer than 3 vertices, the counts do not add up to
/// the number of indices, or an index is negative or not below `pointCount`.
std::vector<std::uint32_t> TriangulateFaces(std::span<const std::int64_t> faceVertexCounts,
                                            std::span<const std::int64_t> faceVertexIndices,
                                            std::size_t pointCount);

/// Per-instance arrays of a point instancer, as UsdGeomPointInstancer defines them.
struct InstanceData {
    std::vector<float> positions;  ///< x, y, z per instance.
    std::vector<float>
        orientations;           ///< Empty, or a quaternion x, y, z, w (real last) per instance.
    std::vector<float> scales;  ///< Empty, or x, y, z per instance.
};

}  // namespace ernie
