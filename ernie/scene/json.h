// SPDX-License-Identifier: BSD-3-Clause
/// @file json.h
/// @brief Scenes loaded from and saved to JSON.
///
/// A scene file maps prim paths to prims:
///
///     {
///       "$schema": "https://ernie.dev/schema/ernie-scene.schema.json",
///       "prims": {
///         "/World": {"type": "Xform", "xform": {"translate": [0, 0, 1]}},
///         "/World/Rock": {"type": "Sphere", "apiSchemas": ["MaterialBindingAPI"],
///                         "radius": 0.5, "material:binding": "/Looks/Basalt"}
///       }
///     }
///
/// - `"type"` names a prim's concrete type; without one a new prim is typeless and an existing one
///   keeps its type. `"apiSchemas"` lists the API schemas it applies.
/// - `"xform"` sets its transform ops (`SetXform`): `translate`, `rotate` (degrees) with
///   `rotationOrder`, `orient` ([w, x, y, z]) and `scale`.
/// - Every other key is an attribute by its USD name, its value read as the attribute's declared
///   type: vectors as arrays, quaternions as [w, x, y, z], matrices as 4 rows (column vectors,
///   translation in the last column), relationships as path strings, tensors as arrays nested by
///   their shape. Undeclared attributes take the type their JSON suggests.
/// - `{"$array": i}` in place of a value is `arrays[i]`, shared rather than copied: how Python
///   hands over its arrays.
///
/// `SaveScene` writes every prim back in path order, attributes by name, so loading the text into
/// an empty scene rebuilds it. GPU tensors and non-finite numbers cannot be written.
#pragma once

#include "ernie/util/tensor.h"

#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace ernie {

class Scene;

/// The JSON Schema scene files name in `"$schema"`.
inline constexpr std::string_view kSceneSchemaUrl =
    "https://ernie.dev/schema/ernie-scene.schema.json";

/// @brief Builds the prims `json` describes into `scene`.
/// @throws std::invalid_argument for text that isn't JSON, a description of the wrong shape, an
/// `"$array"` index out of range, or a value the scene rejects, naming the prim and the key.
void LoadScene(Scene& scene, std::string_view json, std::span<const Tensor> arrays = {});
/// @throws std::invalid_argument if the file cannot be read, or as `LoadScene`.
void LoadSceneFile(Scene& scene, const std::filesystem::path& path);

/// @return The scene as JSON, with 2-space indents.
/// @throws std::invalid_argument for a GPU tensor or a non-finite number, naming the prim and
/// attribute.
std::string SaveScene(const Scene& scene);
/// @throws std::invalid_argument as `SaveScene`, or if the file cannot be written.
void SaveSceneFile(const Scene& scene, const std::filesystem::path& path);

}  // namespace ernie
