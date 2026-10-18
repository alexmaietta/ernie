// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.h
/// @brief The Slang modules every tracing kernel imports: `IRayTracer`, the one interface passes
/// trace through whatever backend answers it, its hardware ray-query backend, and the exact
/// intersections of the implicit shapes.
///
///     import ray_tracer;
///     HardwareRayQuery tracer = { parameters.scene, parameters.records };
///     Hit hit = tracer.TraceClosest(ray, 0xff);
///     if (hit.found) { float3 n = WorldNormal(hit, parameters.records); }
#pragma once

#include "ernie/device/shader_compiler.h"

#include <span>

namespace ernie {

/// @return The modules, in import order, a tracing kernel compiles with: the geometry kinds,
/// `shapes` (unit shapes intersected in their own frame) and `ray_tracer`.
std::span<const ShaderModule> RayTracerModules();

}  // namespace ernie
