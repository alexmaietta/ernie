// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <filesystem>
#include <string_view>

/// Scenes described in JSON, the format of Python's `ernie.load_scene`:
///
///     {
///       "/World": {"type": "Xform",
///         "Rock": {"type": "Sphere", "radius": 0.5, "translate": [0, 0, 1]},
///         "Sun": {"type": "DistantLight", "angle": 0.53, "inputs:intensity": 2.0}},
///       "Render/Settings": {"type": "RenderSettings", "camera": "/World/Camera",
///                           "resolution": [640, 480]}
///     }
///
/// - The top level maps prim paths (absolute, or relative to the root) to prims.
/// - A prim is an object: `"type"` names its type (`Scene::Define`; registered or not). Without
///   one, a new prim is untyped and an existing prim keeps its type and attributes. Keys whose
///   values are objects are its children, by name; the other keys are its attributes, by the
///   Python name of one its type declares (`AttributeDef::pyName`, "cell_size") or by USD name
///   ("inputs:intensity").
/// - `translate` (3 numbers), `rotate` (3 Euler angles in degrees), `rotation_order` ("XYZ", ...,
///   "ZYX"), `orient` (a quaternion as [w, x, y, z], the real part first as USD writes one and as
///   Python's `Quat(w, x, y, z)` takes it) and `scale` (one number or 3) set its transform ops
///   (`SetXform`); `transform` sets `xformOp:transform` to a 4 x 4 matrix given as 4 rows of 4
///   numbers (math convention: column vectors, translation in the last column).
/// - Values convert as Python's do: booleans, integers (32-bit), other numbers as doubles,
///   strings, arrays of strings, arrays of 2 numbers (`ivec2` when both are integers), 3 or 4 as
///   vectors, other lengths as float arrays, and 4 x 4 nested arrays as matrices. A declared
///   attribute's value is then converted to its type (`AttributeDef::Coerce`: a string to a
///   relationship target, say); one declared as a tensor (`points`, `heights`) takes a possibly
///   nested array as a CPU tensor, int32 for one of whole numbers (`AttributeMeta::integer`:
///   indices, counts) and float32 for any other, whatever the numbers look like, written with
///   `Scene::Write`. Values are checked as `Scene::Set` checks them (`Validate`).
namespace ernie {

class Scene;

/// Builds the prims `json` describes into `scene` (see above).
/// @throws std::invalid_argument for text that is not JSON (with its line and column), a
/// description of the wrong shape, or a value `scene` rejects, naming the prim and the key.
void LoadScene(Scene& scene, std::string_view json);

/// `LoadScene` with the text of the file at `path`.
/// @throws std::invalid_argument if the file cannot be read, naming it, or as `LoadScene`.
void LoadSceneFile(Scene& scene, const std::filesystem::path& path);

}  // namespace ernie
