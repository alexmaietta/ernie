// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ernie {

/// A variant selection made before a USD stage is translated: the
/// variant set `set` of the USD prim `prim` set to `selection`.
struct UsdVariantSelection {
    std::string prim;
    std::string set;
    std::string selection;
};

struct UsdImportOptions {
    std::vector<UsdVariantSelection> variants;
    /// The time code to read; the stage's default time without one.
    std::optional<double> time;
    /// Translate prims of purpose "proxy" and "guide" too.
    bool proxy = false;
    bool guide = false;
};

struct UsdImportResult {
    /// The prim the file was imported as.
    Path root;
    /// The Ernie prim each translated USD prim became, by USD path.
    std::unordered_map<std::string, Path> prims;
    /// What the importer skipped or approximated.
    std::vector<std::string> warnings;
    /// The file's authored time code range (startTimeCode and
    /// endTimeCode), if any; it also becomes the scene's Stage range
    /// when the scene has none (end not after start).
    std::optional<glm::dvec2> timeCodeRange;
    double timeCodesPerSecond = 24.0;
    double framesPerSecond = 24.0;
};

/// Imports the USD file `file` (.usd, .usda, .usdc or .usdz) into
/// `scene` at `root`: OpenUSD composes the stage (sublayers,
/// references, payloads, inherits, variants), then each prim becomes
/// the Ernie prim of the same name and type. The file's default prim
/// becomes `root` itself, as a USD reference to the file would place
/// it; without one, its root prims go under `root`, an Xform. The
/// root's transform converts the file's `upAxis` and `metersPerUnit`
/// into the scene's.
///
/// Translated: Xform and Scope, Mesh (welding faceVarying normals
/// into vertices; leftHanded faces reversed), the Cube, Sphere,
/// Cylinder, Cone, Capsule and Plane shapes, Points, PointInstancer,
/// cameras, UsdLux lights, and UsdPreviewSurface materials (bound
/// through `ComputeBoundMaterial`), with materials of other render
/// contexts (OmniPBR's MDL, say) approximated by UsdPreviewSurface.
/// Instances are expanded. UsdSkel skeletons, their animations
/// sampled at each time code, blend shapes and meshes' bindings
/// become Ernie's own
/// (`skel_schemas.h`), the meshes at rest for the renderer to pose.
/// Other prim types become Xforms, with a warning.
///
/// OpenUSD lives in a plugin, `libernie_usd`, loaded on first use:
/// the one beside Ernie's library or Python module, or the one the
/// environment variable ERNIE_USD_PLUGIN names.
/// @throws std::runtime_error if the plugin is missing or cannot load
///   (another OpenUSD already in the process, say), or if the file
///   cannot be opened.
UsdImportResult ImportUsd(Scene& scene, const Path& root,
                          const std::filesystem::path& file,
                          const UsdImportOptions& options = {});

/// Whether the USD importer plugin can be loaded; when not, why in
/// `reason`.
bool UsdImportAvailable(std::string* reason = nullptr);

}  // namespace ernie
