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
///   `Scene::Write`. A declared 4-vector takes its 4 numbers as doubles, and a declared
///   `AttributeMeta::lenient` attribute keeps a value its type cannot convert. On any other
///   attribute, nested arrays other than 4 x 4 (such as a per-vertex `primvars:displayColor`, N
///   rows of 3) are a float32 tensor; on an undeclared one, 4 numbers for an `xformOp:orient` (or
///   `xformOp:orient:<suffix>`) are a quaternion as [w, x, y, z]. Values are checked as `Scene::Set`
///   checks them (`Validate`).
///
/// `SaveScene` writes the format back, so that `LoadScene` of its text into an empty scene
/// rebuilds the scene: every prim but the viewer's (under `tokens::kSessionRoot`), roots by path
/// at the top level and the others nested as children by name, `"type"` first when typed, then
/// the attributes by USD name in name order (transform ops as `xformOp:*` and `xformOpOrder`
/// attributes, not the `translate`-style keys), then the children in path order. Values are
/// written as `LoadScene` reads them back: `Path`s and lists of them as strings, matrices as 4
/// rows, quaternions as [w, x, y, z], floats in their shortest form that reads back as the same
/// float, and tensors as arrays nested by their shape. Limits:
/// - large tensors make large files (a number per element, a line per row), and NaN or infinite
///   numbers cannot be written (JSON has none);
/// - declared attributes come back as their declared types; undeclared ones as `LoadScene` reads
///   the JSON: `Path`s as strings, 2 to 4 numbers as vectors (a float array or a rank-1 tensor
///   of that length among them, and a `dvec4` or a quaternion off `xformOp:orient` as a float
///   `vec4`), other rank-1 arrays as float arrays, nested arrays as float32 tensors (any element
///   type), and a 4 x 4 tensor as a matrix;
/// - declared tensors come back int32 (`AttributeMeta::integer`) or float32 whatever their element
///   type; an empty tensor comes back without its shape and a rank-0 one with shape [1];
/// - an empty list of strings or paths comes back as an empty float array (declared relationship
///   lists convert it back), and unset (`Value::Empty`) values are left out;
/// - overlays (`Scene::GetOverlay`) are not prims and are not written;
/// - GPU tensors are not copied to the host: `SaveScene` throws for them.
namespace ernie {

class Scene;

/// Builds the prims `json` describes into `scene` (see above).
/// @throws std::invalid_argument for text that is not JSON (with its line and column), a
/// description of the wrong shape, or a value `scene` rejects, naming the prim and the key.
void LoadScene(Scene& scene, std::string_view json);

/// `LoadScene` with the text of the file at `path`.
/// @throws std::invalid_argument if the file cannot be read, naming it, or as `LoadScene`.
void LoadSceneFile(Scene& scene, const std::filesystem::path& path);

/// The JSON description of `scene` (see above), pretty-printed with 2-space indents: objects one
/// key per line, arrays of numbers or strings on one line, nested arrays a row per line.
/// @throws std::invalid_argument naming the prim and the attribute for a GPU tensor, a NaN or
/// infinite number, an attribute whose name `LoadScene` would read as something else (`type`, a
/// transform key such as `scale`, or another declared attribute's Python name), or a child named as an
/// attribute of its parent.
std::string SaveScene(const Scene& scene);

/// Writes `SaveScene(scene)` to the file at `path`, replacing it.
/// @throws std::invalid_argument as `SaveScene`, or if the file cannot be written, naming it.
void SaveSceneFile(const Scene& scene, const std::filesystem::path& path);

}  // namespace ernie
