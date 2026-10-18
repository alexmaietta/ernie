// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.cpp
/// @brief The tracing modules: the geometry kinds generated from `GeometryKind`, then the shapes
/// and ray tracer embedded from their .slang files.
#include "ernie/render/ray_tracer.h"

#include "ernie/render/geometry.h"
#include "ernie/render/shaders/ray_tracer.h"
#include "ernie/render/shaders/shapes.h"

#include <array>
#include <string>

namespace ernie {
namespace {

/// The geometry kinds as the shaders name them, from `GeometryKind`.
ShaderModule KindsModule() {
    std::string source;
    const auto kind = [&](const char* name, GeometryKind value) {
        source += std::string("public static const uint ") + name + " = " +
                  std::to_string(static_cast<std::uint32_t>(value)) + ";\n";
    };
    kind("kTriangles", GeometryKind::Triangles);
    kind("kSphere", GeometryKind::Sphere);
    kind("kCube", GeometryKind::Cube);
    kind("kCylinder", GeometryKind::Cylinder);
    kind("kCone", GeometryKind::Cone);
    kind("kCapsule", GeometryKind::Capsule);
    kind("kPlane", GeometryKind::Plane);
    return {"geometry_kinds", std::move(source)};
}

}  // namespace

std::span<const ShaderModule> RayTracerModules() {
    static const std::array modules{KindsModule(), shaders::kShapes, shaders::kRayTracer};
    return modules;
}

}  // namespace ernie
