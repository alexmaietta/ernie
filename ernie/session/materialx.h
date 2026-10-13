// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

#include <filesystem>
#include <optional>
#include <string>

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

/// The folder of MaterialX's data libraries (stdlib, pbrlib, bxdf,
/// lights and their Slang implementations), which generation reads:
/// $ERNIE_MATERIALX_LIBRARIES, else the build's.
std::filesystem::path MaterialXLibraries();

/// The Slang of a MaterialX surface, as MaterialX's Slang generator
/// writes it: a vertex and a pixel stage.
struct MaterialXSlang {
    std::string vertex;
    std::string pixel;
};

/// Generates `source`'s element's Slang.
/// @throws std::invalid_argument when MaterialX cannot generate it
/// (with its message).
MaterialXSlang GenerateMaterialXSlang(const MaterialXSource& source);

}  // namespace ernie
