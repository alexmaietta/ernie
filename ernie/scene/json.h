// SPDX-License-Identifier: BSD-3-Clause
/// @file json.h
/// @brief Scenes loaded from and saved to JSON.
///
/// A scene file maps prim paths to prims:
///
///     {
///       "$schema": "https://alexmaietta.github.io/ernie/schema/ernie-scene.schema.json",
///       "prims": {
///         "/World": {"typeName": "Xform", "xform": {"translate": [0, 0, 1]}},
///         "/World/Rock": {"typeName": "Sphere", "apiSchemas": ["MaterialBindingAPI"],
///                         "radius": 0.5, "material:binding": "/Looks/Basalt"}
///       }
///     }
///
/// - `"typeName"` names a prim's concrete type, as USD's metadata does; without one a new prim is
///   typeless and an existing one keeps its type. `"apiSchemas"` lists the API schemas it applies.
///   Neither is a USD attribute name (BasisCurves has a `type`).
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
#include <vector>

namespace ernie {

class Scene;

/// The JSON Schema scene files name in `"$schema"`, published with the documentation site.
inline constexpr std::string_view kSceneSchemaUrl =
    "https://alexmaietta.github.io/ernie/schema/ernie-scene.schema.json";

/// @brief Builds the prims `json` describes into `scene`.
/// @throws std::invalid_argument for text that isn't JSON, a description of the wrong shape, an
/// `"$array"` index out of range, or a value the scene rejects, naming the prim and the key.
void LoadScene(Scene& scene, std::string_view json, std::span<const Tensor> arrays = {});
/// @throws std::invalid_argument if the file cannot be read, or as `LoadScene`.
void LoadSceneFile(Scene& scene, const std::filesystem::path& path);

/// @brief A problem loading met: the prim and key it is at (either may be empty), and what is
/// wrong.
struct SceneProblem {
    std::string path;
    std::string key;
    std::string what;

    /// @return "'/World/Rock' radius: must be at least 0".
    std::string Message() const;
};
/// @brief Loads as `LoadScene` does, recording every problem and going on past it.
/// @return The problems, in file order; empty if the scene loaded whole.
std::vector<SceneProblem> LoadSceneChecked(Scene& scene, std::string_view json,
                                           std::span<const Tensor> arrays = {});

/// @return The scene as JSON, with 2-space indents.
/// @throws std::invalid_argument for a GPU tensor or a non-finite number, naming the prim and
/// attribute.
std::string SaveScene(const Scene& scene);
/// @throws std::invalid_argument as `SaveScene`, or if the file cannot be written.
void SaveSceneFile(const Scene& scene, const std::filesystem::path& path);

}  // namespace ernie
