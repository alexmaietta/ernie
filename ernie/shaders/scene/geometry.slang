// SPDX-License-Identifier: BSD-3-Clause
/// @file geometry.slang
/// @brief Geometry as kernels read it: its kinds and records, shared with C++ (geometry.h).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/shared.h"
namespace ernie::shaders {
#else
import ref;
#endif

/// The kinds of geometry: triangles, then the unit shapes procedural primitives stand for.
static const uint kTriangles = 0;
static const uint kSphere = 1;
static const uint kCube = 2;
static const uint kCylinder = 3;
static const uint kCone = 4;
static const uint kCapsule = 5;
static const uint kPlane = 6;

/// A geometry as kernels read it, by geometry id: an instance's custom index.
struct GeometryRecord {
    uint kind;
    /// A capsule's segment half-length over its radius.
    float halfLength;
    uint vertexCount;
    uint triangleCount;
    /// Triangles' x, y, z positions and their corners' indices, three a triangle.
    Ref<float> vertices;
    Ref<uint> indices;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

/// Writes a triangle's corners, in the geometry's frame, to `corners`.
void Corners(GeometryRecord record, uint triangle, out float3 corners[3]) {
    for (uint i = 0; i < 3; ++i) {
        uint at = 3 * record.indices[3 * triangle + i];
        corners[i] = float3(record.vertices[at], record.vertices[at + 1], record.vertices[at + 2]);
    }
}

#endif
