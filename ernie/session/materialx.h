// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

/// MaterialX materials: a Material's MaterialX surface (a network of
/// Shader prims of MaterialX node definitions on
/// outputs:mtlx:surface, or a material of a .mtlx document it names)
/// as a MaterialX document, and the Slang that MaterialX's shader
/// generation writes for it. MaterialX itself stays inside Ernie:
/// only plain types cross.
///
/// References:
/// - Academy Software Foundation, "MaterialX Specification" 1.39,
///   https://materialx.org/Specification.html
/// - Academy Software Foundation, MaterialX 1.39.5:
///   MaterialXGenSlang.
namespace ernie {

/// A Material's MaterialX surface as a document.
struct MaterialXSource {
    std::string document;  ///< The MaterialX document, as XML.
    /// The element of it to render, by name path: a material (a
    /// surfacematerial node), or a shader.
    std::string element;
    /// Where its relative file names resolve from: the .mtlx file's
    /// folder; empty for a network.
    std::filesystem::path base;
};

/// The MaterialX surface of the Material at `material`: its network
/// on outputs:mtlx:surface, else the document ernie:materialx:file
/// names (its ernie:materialx:element, or its first renderable
/// element); none without either.
/// @throws std::invalid_argument for a node of a definition MaterialX
/// lacks, a connection to a prim that is not a Shader, a file that
/// cannot be read, an element it lacks, or a document MaterialX finds
/// invalid (with its message).
std::optional<MaterialXSource> MaterialXOf(const Scene& scene,
                                           const Path& material);

/// The elements of the MaterialX document `file` there are to render
/// (materials, shaders and outputs), by name path, in document order.
/// @throws std::invalid_argument for a file that cannot be read.
std::vector<std::string>
MaterialXElements(const std::filesystem::path& file);

/// The folder of MaterialX's data libraries (stdlib, pbrlib, bxdf,
/// lights and their Slang implementations), which generation reads:
/// $ERNIE_MATERIALX_LIBRARIES, else the build's.
std::filesystem::path MaterialXLibraries();

/// A value a MaterialX shader reads from its material's uniform
/// buffer.
struct MaterialXUniform {
    std::string variable;  ///< Its name in the shader.
    /// Its MaterialX type: float, integer, boolean, vector2, vector3,
    /// vector4, color3, color4, matrix33 or matrix44.
    std::string type;
    /// Its byte offset in the buffer, which is laid out scalar (each
    /// value's floats or ints one after another, 4-byte aligned;
    /// booleans as ints; matrices row after row).
    std::uint32_t offset = 0;
    /// The input it comes from, by MaterialX name path
    /// ("<node>/<input>"); empty for none.
    std::string path;
    std::string value;  ///< Its value, as MaterialX writes it.
};

/// A texture a MaterialX shader samples, by its slot in the
/// material's texture table.
struct MaterialXTexture {
    std::string variable;  ///< Its name in the shader.
    std::uint32_t slot = 0;
    std::string file;  ///< Its file (filename value), resolved.
    std::string path;  ///< The input it comes from, by name path.
    /// How its image node addresses it beyond [0, 1] along u and v
    /// (MaterialX's uaddressmode and vaddressmode: 0 constant,
    /// 1 clamp, 2 periodic, 3 mirror) and filters it (filtertype:
    /// 0 closest, 1 linear, 2 cubic).
    int uAddress = 2;
    int vAddress = 2;
    int filter = 1;
};

/// A MaterialX surface's shading as Slang for Ernie's shading pass:
/// `void ernieMaterialX(out float3 color, out float3 transparency)`,
/// which evaluates its graph for the pixel being shaded through the
/// functions below (which the shading pass defines before it), with
/// MaterialX's own closures and nodes:
///
/// - ernieMxPositionWorld(), ernieMxNormalWorld(),
///   ernieMxTangentWorld(), ernieMxBitangentWorld() and their
///   *Object() versions (float3); ernieMxTexcoord(int set) (float2);
///   ernieMxColor(int set) (float4); ernieMxGeomPropFloat(int id),
///   ...Float2, ...Float3, ...Float4 and ...Int, the primvars
///   `geomProps` names by id; ernieMxViewPosition() (float3),
///   ernieMxTime() and ernieMxFrame() (float).
/// - ernieMxLoadFloat(uint offset), ...Int, ...Float2, ...Float3,
///   ...Float4, ...Float3x3 and ...Float4x4: the uniform buffer's
///   values.
/// - ernieMxSample(uint slot, float2 uv),
///   ernieMxSampleLevel(uint slot, float2 uv, float lod),
///   ernieMxSampleGrad(uint slot, float2 uv, float2 dx, float2 dy)
///   (float4) and ernieMxTextureSize(uint slot, int level) (int2):
///   the texture table's.
/// - ernieMxLightCount() (int); ernieMxLightDirection(int light)
///   (float3, towards it; zero for one that cannot light the pixel);
///   ernieMxLightResponse(int light, float3 response): the surface's
///   response to a unit of the light's radiant intensity, which the
///   shading pass scales by the light's spectrum, distance and
///   shadow.
/// - ernieMxEnvironmentRadiance(float3 direction, float alpha) and
///   ernieMxEnvironmentIrradiance(float3 normal) (float3): the light
///   from all around (dome, sky and ambient light), as a GGX lobe of
///   roughness alpha about `direction` and as cosine-weighted
///   irradiance over pi.
///
/// `color` is what leaves the surface towards the eye, but for the
/// lights' share (reported per light): its emission and environment
/// light. Colours are linear Rec. 709, as MaterialX's are.
struct MaterialXShader {
    std::string code;
    std::vector<MaterialXUniform> uniforms;
    std::uint32_t uniformBytes = 0;
    std::vector<MaterialXTexture> textures;
    std::vector<std::string> geomProps;  ///< Primvar names, by id.
    /// Whether it is a surface (lit by its closures) rather than a
    /// pattern or value shown as it is.
    bool surface = false;
    /// Whether it may be transparent (MaterialX's
    /// isTransparentSurface).
    bool transparent = false;
};

/// Generates `source`'s element's shading for Ernie.
/// @throws std::invalid_argument when MaterialX cannot generate it
/// (with its message).
MaterialXShader
GenerateMaterialXShader(const MaterialXSource& source);

}  // namespace ernie
