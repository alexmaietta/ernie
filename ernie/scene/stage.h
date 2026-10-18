// SPDX-License-Identifier: BSD-3-Clause
/// @file stage.h
/// @brief UsdStage's metadata, with Ernie's scene-wide settings.
///
/// References:
/// - Pixar, UsdStage, https://openusd.org/release/api/class_usd_stage.html
/// - Pixar, "Encoding Stage UpAxis" and "Encoding Stage Linear Units",
///   https://openusd.org/release/api/group___usd_geom_up_axis__group.html
#pragma once

#include <string>

namespace ernie {

enum class UpAxis { Y, Z };

/// @brief What a scene says about itself as a whole. Ernie draws in metres with z up, the
/// fallbacks here; a scene from USD records its own units, which importers convert.
struct Stage {
    UpAxis upAxis = UpAxis::Z;
    double metersPerUnit = 1.0;
    double kilogramsPerUnit = 1.0;
    /// The root prim a reference to the scene brings in, by name.
    std::string defaultPrim;

    bool operator==(const Stage&) const = default;
};

/// @throws std::invalid_argument for units that aren't positive and finite, or a default prim
/// that isn't a prim name.
void Validate(const Stage& stage);

}  // namespace ernie
