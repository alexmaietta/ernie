// SPDX-License-Identifier: BSD-3-Clause
/// @file geometry.slang
/// @brief Geometry as kernels read it, as geometry.h lays it out; a test holds the two together.

/// `GeometryKind`.
public static const uint kTriangles = 0;
public static const uint kSphere = 1;
public static const uint kCube = 2;
public static const uint kCylinder = 3;
public static const uint kCone = 4;
public static const uint kCapsule = 5;
public static const uint kPlane = 6;

/// A geometry's record, as the C++ `GeometryRecord`: an instance's custom index finds it.
public struct GeometryRecord {
    public uint kind;
    public float halfLength;
    public uint vertexCount;
    public uint triangleCount;
    public float* vertices;
    public uint* indices;
};

/// Writes a triangle's corners, in the geometry's frame, to `corners`.
public void Corners(GeometryRecord record, uint triangle, out float3 corners[3]) {
    uint* at = record.indices + 3 * triangle;
    for (uint i = 0; i < 3; ++i) {
        float* v = record.vertices + 3 * at[i];
        corners[i] = float3(v[0], v[1], v[2]);
    }
}
