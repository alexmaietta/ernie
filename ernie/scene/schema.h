// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/camera.h"
#include "ernie/scene/scene.h"

#include <glm/glm.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// Helpers for authoring and reading the schemas in `tokens.h`.
///
/// The readers accept any numeric type an author might use for an
/// attribute (an `int`, `float` or `double` number; a `vec` or `dvec`
/// vector), since bindings such as Python's produce doubles.
namespace ernie {

/// The attribute as a number, or `fallback` if it is unset or not
/// numeric.
double GetNumber(const Scene& scene, const Path& path,
                 std::string_view attribute, double fallback);

/// The attribute as a bool (from a bool or an int), or `fallback`.
bool GetBool(const Scene& scene, const Path& path,
             std::string_view attribute, bool fallback);

/// The attribute as a 2-vector (from `vec2`, `dvec2` or `ivec2`), or
/// `fallback`.
glm::dvec2 GetVec2(const Scene& scene, const Path& path,
                   std::string_view attribute, glm::dvec2 fallback);

/// The attribute as a 3-vector (from `vec3` or `dvec3`), or
/// `fallback`.
glm::dvec3 GetVec3(const Scene& scene, const Path& path,
                   std::string_view attribute, glm::dvec3 fallback);

/// The attribute as a list of numbers, or nullopt if it is unset or
/// not numeric: a float array, or a 2-, 3- or 4-vector (`vec`, `dvec`
/// or `ivec`), which is how short lists arrive from bindings such as
/// Python's (a list of four numbers is a `vec4`), or a single number.
std::optional<std::vector<float>>
GetFloats(const Scene& scene, const Path& path,
          std::string_view attribute);

/// The attribute as a token or string, or `fallback`.
std::string GetToken(const Scene& scene, const Path& path,
                     std::string_view attribute,
                     std::string_view fallback);

/// The attribute as a relationship target, or the empty path.
Path GetTarget(const Scene& scene, const Path& path,
               std::string_view attribute);

/// The source of a UsdShade connection: a Shader prim and one of its
/// outputs, e.g. "outputs:rgb".
struct Connection {
    Path prim;
    std::string output;
};

/// The connection of input `input` (e.g. "inputs:diffuseColor") of
/// the prim at `path`: its `<input>.connect` attribute, the source's
/// property path as text ("/Looks/M/Tex.outputs:rgb"), or nullopt
/// when it has none.
/// @throws std::invalid_argument if the text is not a prim path, a
/// dot and a property name.
std::optional<Connection> GetConnection(const Scene& scene,
                                        const Path& path,
                                        std::string_view input);

/// Connects input `input` of the prim at `path` to output `output` of
/// the prim at `source`.
void Connect(Scene& scene, const Path& path, std::string_view input,
             const Path& source, std::string_view output);

/// Camera-to-world transform of a USD camera at `eye` looking at
/// `target`: the camera looks down its -z axis with +y up, so `up`
/// ends up in the camera's y-z plane.
/// @throws std::invalid_argument if `eye` equals `target` or `up` is
/// parallel to the view.
glm::dmat4 LookAtTransform(glm::dvec3 eye, glm::dvec3 target,
                           glm::dvec3 up);

/// Transform of a DistantLight shining from `towardsLight`: its +z
/// axis points along it, so the light travels down -z as UsdLux
/// defines.
/// @throws std::invalid_argument if `towardsLight` is zero.
glm::dmat4 DistantLightTransform(glm::dvec3 towardsLight);

/// Unit vector from the scene towards a DistantLight: the +z axis of
/// its world transform.
glm::dvec3 TowardsLight(const Scene& scene, const Path& light);

/// Authors `camera` as the Camera prim at `path`: its transform,
/// `projection`, `focalLength`, apertures (the horizontal one from
/// the image's aspect) and their offsets, `clippingRange`, `exposure`
/// and the `exposure:*` attributes, `ernie:camera:scan` and the
/// pushbroom's line offset, and `ernie:spectral:range` and
/// `ernie:spectral:bands`. Its lens and sensor are not written.
/// `CameraParamsOf` reads it back.
void SetCamera(Scene& scene, const Path& path,
               const CameraParams& camera);

}  // namespace ernie
