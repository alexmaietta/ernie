// SPDX-License-Identifier: BSD-3-Clause
//
// libernie_usd: composes a USD stage with OpenUSD, reads it through
// UsdImaging's Hydra scene indices, and authors Ernie prims through
// the Sink in plugin_api.h. The only code that includes pxr headers.
//
// The scene indices resolve what USD leaves to renderers: native and
// point instancing (as instancers of shared prototypes), material
// bindings (collections, purposes, inheritance), inherited primvars
// and purposes, extents, and skinning where UsdSkelImaging is built.
//
// References:
//   OpenUSD API: https://openusd.org/release/api/index.html
//   Hydra 2.0 scene indices:
//     https://openusd.org/release/api/_page__hydra__getting__started__guide.html
//   UsdImaging's scene index chain (UsdImagingCreateSceneIndices):
//     https://openusd.org/release/api/sceneIndices_8h.html
//   OmniPBR's inputs:
//   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniPBR.html

#include "ernie/usd/plugin_api.h"

#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/cameraSchema.h>
#include <pxr/imaging/hd/capsuleSchema.h>
#include <pxr/imaging/hd/coneSchema.h>
#include <pxr/imaging/hd/cubeSchema.h>
#include <pxr/imaging/hd/cylinderSchema.h>
#include <pxr/imaging/hd/flatteningSceneIndex.h>
#include <pxr/imaging/hd/geomSubsetSchema.h>
#include <pxr/imaging/hd/instanceSchema.h>
#include <pxr/imaging/hd/instancerTopologySchema.h>
#include <pxr/imaging/hd/lightSchema.h>
#include <pxr/imaging/hd/materialBindingSchema.h>
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/materialConnectionSchema.h>
#include <pxr/imaging/hd/materialNetworkSchema.h>
#include <pxr/imaging/hd/materialNodeParameterSchema.h>
#include <pxr/imaging/hd/materialNodeSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/meshSchema.h>
#include <pxr/imaging/hd/meshTopologySchema.h>
#include <pxr/imaging/hd/planeSchema.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/purposeSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/sceneIndexPrimView.h>
#include <pxr/imaging/hd/sphereSchema.h>
#include <pxr/imaging/hd/subdivisionTagsSchema.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/packageUtils.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/collectionAPI.h>
#include <pxr/usd/usd/editContext.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/variantSets.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/pointBased.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdRender/product.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/usd/usdRender/var.h>
#include <pxr/usd/usdSkel/animQuery.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/binding.h>
#include <pxr/usd/usdSkel/blendShape.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/inbetweenShape.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdSkel/utils.h>
#include <pxr/usd/usdVol/fieldAsset.h>
#include <pxr/usd/usdVol/openVDBAsset.h>
#include <pxr/usd/usdVol/volume.h>
#include <pxr/usdImaging/usdImaging/flattenedDataSourceProviders.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

using ernie::usd_plugin::Request;
using ernie::usd_plugin::Sink;

#define ERNIE_USD_STRINGIFY2(x) #x
#define ERNIE_USD_STRINGIFY(x) ERNIE_USD_STRINGIFY2(x)

// ---------------------------------------------------------------------
// Values

/// `value` as numbers for Sink::SetNumbers, or nothing when it is not
/// a number, vector or matrix.
std::optional<std::vector<double>> NumbersOf(const VtValue& value) {
    std::vector<double> out;
    if (value.IsHolding<double>()) {
        out = {value.UncheckedGet<double>()};
    } else if (value.IsHolding<float>()) {
        out = {static_cast<double>(value.UncheckedGet<float>())};
    } else if (value.IsHolding<GfHalf>()) {
        out = {static_cast<double>(
            static_cast<float>(value.UncheckedGet<GfHalf>()))};
    } else if (value.IsHolding<GfVec2f>()) {
        const GfVec2f v = value.UncheckedGet<GfVec2f>();
        out = {v[0], v[1]};
    } else if (value.IsHolding<GfVec2d>()) {
        const GfVec2d v = value.UncheckedGet<GfVec2d>();
        out = {v[0], v[1]};
    } else if (value.IsHolding<GfVec2i>()) {
        const GfVec2i v = value.UncheckedGet<GfVec2i>();
        out = {static_cast<double>(v[0]), static_cast<double>(v[1])};
    } else if (value.IsHolding<GfVec3f>()) {
        const GfVec3f v = value.UncheckedGet<GfVec3f>();
        out = {v[0], v[1], v[2]};
    } else if (value.IsHolding<GfVec3d>()) {
        const GfVec3d v = value.UncheckedGet<GfVec3d>();
        out = {v[0], v[1], v[2]};
    } else if (value.IsHolding<GfVec3h>()) {
        const GfVec3h v = value.UncheckedGet<GfVec3h>();
        out = {static_cast<float>(v[0]), static_cast<float>(v[1]),
               static_cast<float>(v[2])};
    } else if (value.IsHolding<GfVec4f>()) {
        const GfVec4f v = value.UncheckedGet<GfVec4f>();
        out = {v[0], v[1], v[2], v[3]};
    } else if (value.IsHolding<GfVec4d>()) {
        const GfVec4d v = value.UncheckedGet<GfVec4d>();
        out = {v[0], v[1], v[2], v[3]};
    } else if (value.IsHolding<GfMatrix4d>()) {
        // GfMatrix4d's rows (row vectors) are glm's columns (column
        // vectors): the same 16 doubles.
        const double* data = value.UncheckedGet<GfMatrix4d>().data();
        out.assign(data, data + 16);
    } else {
        return std::nullopt;
    }
    return out;
}

/// The path an asset value resolves to, or as authored when it does
/// not resolve.
std::string AssetText(const SdfAssetPath& asset) {
    const std::string& resolved = asset.GetResolvedPath();
    return resolved.empty() ? asset.GetAssetPath() : resolved;
}

/// A data source's value now, empty without one.
VtValue ValueOf(const HdDataSourceBaseHandle& source) {
    const HdSampledDataSourceHandle sampled =
        HdSampledDataSource::Cast(source);
    return sampled ? sampled->GetValue(0.0f) : VtValue();
}

template <typename T>
std::optional<T> TypedOf(const HdDataSourceBaseHandle& source) {
    const VtValue value = ValueOf(source);
    if (value.IsHolding<T>()) {
        return value.UncheckedGet<T>();
    }
    return std::nullopt;
}

GfMatrix4d CorrectionOf(const UsdStageRefPtr& stage,
                        const Request& request) {
    const double meters = UsdGeomGetStageMetersPerUnit(stage);
    const TfToken up = UsdGeomGetStageUpAxis(stage);
    GfMatrix4d correction(1.0);
    const double scale = meters / request.metersPerUnit;
    if (std::abs(scale - 1.0) > 1e-12) {
        correction.SetScale(scale);
    }
    const bool fileY = up == UsdGeomTokens->y;
    const bool sceneY = request.upAxis == 'Y';
    if (fileY != sceneY) {
        // Y up to Z up turns +90 degrees about X (Y onto Z); back is
        // -90.
        GfMatrix4d rotate(1.0);
        rotate.SetRotate(
            GfRotation(GfVec3d(1.0, 0.0, 0.0), fileY ? 90.0 : -90.0));
        correction = correction * rotate;
    }
    return correction;
}

// ---------------------------------------------------------------------
// Meshes

/// A primvar flattened (de-indexed) to one value per element of its
/// interpolation.
struct Flat {
    std::vector<float> values;
    std::size_t width = 0;
    TfToken interpolation;
    bool Valid() const {
        return width != 0;
    }
    const float* At(std::size_t element) const {
        return values.data() + element * width;
    }
};

template <typename T, std::size_t N>
Flat FlattenOf(const VtArray<T>& values,
               const TfToken& interpolation) {
    Flat flat;
    flat.width = N;
    flat.interpolation = interpolation;
    flat.values.reserve(values.size() * N);
    for (const T& value : values) {
        for (std::size_t c = 0; c < N; ++c) {
            flat.values.push_back(static_cast<float>(value[c]));
        }
    }
    return flat;
}

/// A primvar of float2 or float3 values (vertex or face corner
/// normals, texture coordinates, colours), de-indexed by Hydra.
Flat FlatPrimvar(const HdPrimvarSchema& primvar) {
    if (!primvar) {
        return Flat{};
    }
    const VtValue value = ValueOf(primvar.GetFlattenedPrimvarValue());
    const TfToken interpolation =
        primvar.GetInterpolation()
            ? primvar.GetInterpolation()->GetTypedValue(0.0f)
            : HdPrimvarSchemaTokens->constant;
    if (value.IsHolding<VtVec2fArray>()) {
        return FlattenOf<GfVec2f, 2>(
            value.UncheckedGet<VtVec2fArray>(), interpolation);
    }
    if (value.IsHolding<VtVec2dArray>()) {
        return FlattenOf<GfVec2d, 2>(
            value.UncheckedGet<VtVec2dArray>(), interpolation);
    }
    if (value.IsHolding<VtVec3fArray>()) {
        return FlattenOf<GfVec3f, 3>(
            value.UncheckedGet<VtVec3fArray>(), interpolation);
    }
    if (value.IsHolding<VtVec3dArray>()) {
        return FlattenOf<GfVec3d, 3>(
            value.UncheckedGet<VtVec3dArray>(), interpolation);
    }
    return Flat{};
}

/// A mesh's source arrays, flattened.
struct MeshSource {
    VtVec3fArray points;
    VtIntArray counts;
    VtIntArray indices;
    bool leftHanded = false;
    Flat normals;
    Flat color;
    std::vector<std::pair<std::string, Flat>> uvs;
    /// First corner of each face.
    std::vector<std::size_t> starts;
};

/// What Ernie's Mesh takes: points with one normal each, faces, and
/// texture coordinates per face corner.
struct MeshOut {
    std::vector<float> points;
    std::vector<float> normals;
    std::vector<float> colors;
    std::vector<std::int32_t> counts;
    std::vector<std::int32_t> indices;
    std::vector<std::pair<std::string, std::vector<float>>> uvs;
    /// The output point each source point became first, or -1.
    std::vector<std::int32_t> vertexOf;
    /// The source point of each output point.
    std::vector<std::int32_t> sourceOf;
};

/// Index of the element of `flat` a corner reads: by point, face or
/// corner, as its interpolation says.
std::size_t ElementOf(const Flat& flat, std::size_t point,
                      std::size_t face, std::size_t corner) {
    if (flat.interpolation == HdPrimvarSchemaTokens->vertex ||
        flat.interpolation == HdPrimvarSchemaTokens->varying) {
        return point;
    }
    if (flat.interpolation == HdPrimvarSchemaTokens->uniform) {
        return face;
    }
    if (flat.interpolation == HdPrimvarSchemaTokens->faceVarying) {
        return corner;
    }
    return 0;  // constant
}

bool PerCorner(const Flat& flat) {
    return flat.Valid() &&
           (flat.interpolation ==
                HdPrimvarSchemaTokens->faceVarying ||
            flat.interpolation == HdPrimvarSchemaTokens->uniform);
}

/// The faces `faces` of `source` as an Ernie mesh: points split where
/// a point's corners have different normals or colours (so hard edges
/// stay hard), the rest shared; faces reversed when left-handed;
/// texture coordinates per corner.
MeshOut BuildMesh(const MeshSource& source,
                  const std::vector<std::size_t>& faces) {
    MeshOut out;
    out.vertexOf.assign(source.points.size(), -1);
    struct Key {
        std::size_t point;
        std::array<std::uint32_t, 6> bits;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const {
            std::size_t h = std::hash<std::size_t>()(key.point);
            for (const std::uint32_t b : key.bits) {
                h ^= std::hash<std::uint32_t>()(b) +
                     0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
    std::unordered_map<Key, std::int32_t, KeyHash> vertices;
    const auto bitsOf = [](const Flat& flat, std::size_t element,
                           std::array<std::uint32_t, 6>& bits,
                           std::size_t offset) {
        for (std::size_t c = 0; c < flat.width; ++c) {
            std::memcpy(&bits[offset + c], flat.At(element) + c,
                        sizeof(float));
        }
    };
    const bool normals = source.normals.Valid();
    const bool colors =
        source.color.Valid() &&
        source.color.interpolation != HdPrimvarSchemaTokens->constant;
    std::vector<std::vector<float>> uvs(source.uvs.size());
    for (const std::size_t face : faces) {
        const std::size_t start = source.starts[face];
        const auto count =
            static_cast<std::size_t>(source.counts[face]);
        out.counts.push_back(static_cast<std::int32_t>(count));
        for (std::size_t k = 0; k < count; ++k) {
            // Left-handed faces keep their first corner and reverse
            // the rest, so that they wind as right-handed ones.
            const std::size_t local =
                source.leftHanded && k != 0 ? count - k : k;
            const std::size_t corner = start + local;
            const auto point =
                static_cast<std::size_t>(source.indices[corner]);
            Key key{point, {}};
            if (normals) {
                bitsOf(source.normals,
                       ElementOf(source.normals, point, face, corner),
                       key.bits, 0);
            }
            if (colors) {
                bitsOf(source.color,
                       ElementOf(source.color, point, face, corner),
                       key.bits, 3);
            }
            auto [it, added] = vertices.try_emplace(
                key,
                static_cast<std::int32_t>(out.points.size() / 3));
            if (added) {
                if (out.vertexOf[point] < 0) {
                    out.vertexOf[point] = it->second;
                }
                out.sourceOf.push_back(
                    static_cast<std::int32_t>(point));
                const GfVec3f& p = source.points[point];
                out.points.insert(out.points.end(),
                                  {p[0], p[1], p[2]});
                if (normals) {
                    const float* n = source.normals.At(ElementOf(
                        source.normals, point, face, corner));
                    out.normals.insert(out.normals.end(), n, n + 3);
                }
                if (colors) {
                    const float* c = source.color.At(
                        ElementOf(source.color, point, face, corner));
                    out.colors.insert(out.colors.end(), c, c + 3);
                }
            }
            out.indices.push_back(it->second);
            for (std::size_t u = 0; u < source.uvs.size(); ++u) {
                const Flat& uv = source.uvs[u].second;
                const float* t =
                    uv.At(ElementOf(uv, point, face, corner));
                uvs[u].insert(uvs[u].end(), t, t + 2);
            }
        }
    }
    for (std::size_t u = 0; u < source.uvs.size(); ++u) {
        out.uvs.emplace_back(source.uvs[u].first, std::move(uvs[u]));
    }
    return out;
}

/// Whether the primvar `name` holds texture coordinates: its role
/// says so, or it is named as UV sets usually are.
bool IsTexCoords(const std::string& name,
                 const HdPrimvarSchema& primvar) {
    if (primvar.GetRole() &&
        primvar.GetRole()->GetTypedValue(0.0f) ==
            HdPrimvarRoleTokens->textureCoordinate) {
        return true;
    }
    return name == "st" || name.starts_with("st") || name == "uv" ||
           name.starts_with("UV") || name.starts_with("uv") ||
           name == "map1";
}

/// Decomposes a row-vector transform without shear into translation,
/// rotation and scale.
void Decompose(const GfMatrix4d& matrix, GfVec3f& translation,
               GfQuatf& rotation, GfVec3f& scale) {
    translation = GfVec3f(matrix.ExtractTranslation());
    GfMatrix4d rows = matrix;
    for (int r = 0; r < 3; ++r) {
        const GfVec3d row(matrix[r][0], matrix[r][1], matrix[r][2]);
        const double length = row.GetLength();
        scale[r] = static_cast<float>(length);
        for (int c = 0; c < 3; ++c) {
            rows[r][c] = length > 0.0 ? matrix[r][c] / length : 0.0;
        }
    }
    rotation = GfQuatf(rows.ExtractRotationQuat());
}

// UsdLux inputs Ernie's lights take, as UsdImaging names a light's
// parameters (the attributes without "inputs:"), and the two that are
// not inputs.
const std::array<const char*, 20> kLightInputs = {
    "intensity",
    "exposure",
    "color",
    "colorTemperature",
    "enableColorTemperature",
    "normalize",
    "radius",
    "width",
    "height",
    "length",
    "angle",
    "texture:file",
    "texture:format",
    "shaping:focus",
    "shaping:focusTint",
    "shaping:cone:angle",
    "shaping:cone:softness",
    "shaping:ies:file",
    "shaping:ies:angleScale",
    "shaping:ies:normalize"};
const std::array<const char*, 2> kLightAttributes = {"treatAsPoint",
                                                     "treatAsLine"};

// ---------------------------------------------------------------------
// The importer

class Importer {
  public:
    Importer(const Request& request, Sink& sink)
        : request_(request), sink_(sink), root_(request.root),
          time_(request.hasTime ? UsdTimeCode(request.time)
                                : UsdTimeCode::EarliestTime()) {}

    void Run() {
        TfErrorMark mark;
        stage_ = UsdStage::Open(request_.file, UsdStage::LoadAll);
        if (!stage_) {
            std::string message =
                std::string("Cannot open the USD file ") +
                request_.file;
            for (const TfError& error : mark) {
                message += ": " + error.GetCommentary();
            }
            mark.Clear();
            throw std::runtime_error(message);
        }
        mark.Clear();
        ApplyVariants();
        CollectSkinning();
        if (stage_->HasAuthoredTimeCodeRange()) {
            sink_.SetTimeRange(stage_->GetStartTimeCode(),
                               stage_->GetEndTimeCode(),
                               stage_->GetTimeCodesPerSecond(),
                               stage_->GetFramesPerSecond());
        }

        // UsdImaging's chain (instancing, material binding
        // resolution, plugins such as skinning), then flattening of
        // what Ernie does not inherit itself: purposes, material
        // bindings and primvars.
        UsdImagingCreateSceneIndicesInfo info;
        info.stage = stage_;
        info.addDrawModeSceneIndex = false;
        const UsdImagingSceneIndices indices =
            UsdImagingCreateSceneIndices(info);
        indices.stageSceneIndex->SetTime(time_);
        input_ = indices.finalSceneIndex;
        const HdContainerDataSourceHandle providers =
            UsdImagingFlattenedDataSourceProviders();
        flat_ = HdFlatteningSceneIndex::New(
            input_,
            HdRetainedContainerDataSource::New(
                HdPurposeSchema::GetSchemaToken(),
                providers->Get(HdPurposeSchema::GetSchemaToken()),
                HdMaterialBindingsSchema::GetSchemaToken(),
                providers->Get(
                    HdMaterialBindingsSchema::GetSchemaToken()),
                HdPrimvarsSchema::GetSchemaToken(),
                providers->Get(HdPrimvarsSchema::GetSchemaToken())));

        const GfMatrix4d correction = CorrectionOf(stage_, request_);
        const UsdPrim defaultPrim = stage_->GetDefaultPrim();
        if (defaultPrim) {
            default_ = defaultPrim.GetPath();
            correction_ = correction;
            TranslateTree(default_);
        } else {
            Call(root_, [&] {
                DefinePrim(root_.c_str(), "Xform");
                SetMatrix(root_, correction);
            });
            for (const UsdPrim& child :
                 stage_->GetPseudoRoot().GetChildren()) {
                TranslateTree(child.GetPath());
            }
        }
        if (request_.renderSettings) {
            RenderPrims();
        }
        LightLinks();
        SampledAttributes();
        if (!mark.IsClean()) {
            for (const TfError& error : mark) {
                Warn("USD: " + error.GetCommentary());
            }
        }
        mark.Clear();
    }

  private:
    void ApplyVariants() {
        if (request_.variantCount == 0) {
            return;
        }
        const UsdEditContext context(stage_,
                                     stage_->GetSessionLayer());
        for (std::size_t i = 0; i < request_.variantCount; ++i) {
            const auto& v = request_.variants[i];
            const UsdPrim prim =
                stage_->GetPrimAtPath(SdfPath(v.prim));
            if (!prim) {
                Warn(std::string("No prim ") + v.prim +
                     " to select the variant " + v.set + "=" +
                     v.selection + " on");
                continue;
            }
            UsdVariantSet set =
                prim.GetVariantSets().GetVariantSet(v.set);
            if (!set.IsValid() ||
                !set.HasAuthoredVariant(v.selection)) {
                Warn(std::string(v.prim) + " has no variant " +
                     v.set + "=" + v.selection);
                continue;
            }
            set.SetVariantSelection(v.selection);
        }
    }

    /// The Ernie path of the scene index prim at `path`: the default
    /// prim is the root; prims outside it (materials bound from
    /// elsewhere, instancing's prototypes) go under the root's
    /// "_external".
    std::string Map(const SdfPath& path) const {
        if (default_.IsEmpty()) {
            return root_ + path.GetString();
        }
        if (path == default_) {
            return root_;
        }
        if (path.HasPrefix(default_)) {
            return root_ +
                   path.ReplacePrefix(default_,
                                      SdfPath::AbsoluteRootPath())
                       .GetString();
        }
        return root_ + "/_external" + path.GetString();
    }

    void Warn(const std::string& message) {
        if (warned_.insert(message).second) {
            sink_.Warn(message.c_str());
        }
    }

    /// Runs `f`, reporting what Ernie rejects as a warning about
    /// `path`.
    template <typename F> bool Call(const std::string& path, F&& f) {
        try {
            f();
            return true;
        } catch (const std::exception& e) {
            Warn(path + ": " + e.what());
            return false;
        }
    }

    /// Defines the Ernie prim `path`, and first the root's
    /// "_external" when it is under it: prims outside the default
    /// prim are in the stage's frame, so it undoes the default prim's
    /// own transform.
    void DefinePrim(const char* path, const char* type) {
        const std::string external = root_ + "/_external";
        if (!default_.IsEmpty() && !external_ &&
            std::string_view(path).starts_with(external)) {
            defined_.insert(external);
            external_ = true;
            sink_.Define(external.c_str(), "Xform");
            const HdXformSchema xform = HdXformSchema::GetFromParent(
                input_->GetPrim(default_).dataSource);
            if (xform.GetMatrix()) {
                SetMatrix(external, xform.GetMatrix()
                                        ->GetTypedValue(0.0f)
                                        .GetInverse());
            }
        }
        sink_.Define(path, type);
        defined_.insert(path);
    }

    void SetMatrix(const std::string& path,
                   const GfMatrix4d& matrix) {
        sink_.SetNumbers(path.c_str(), "xformOp:transform",
                         matrix.data(), 16);
    }

    /// The file the asset value `asset` of the prim `path` names, for
    /// Ernie to read: its resolved path, unpacked first into the
    /// package cache when it is inside a package (a .usdz's texture or
    /// volume).
    std::string Asset(const std::string& path,
                      const SdfAssetPath& asset) {
        const std::string text = AssetText(asset);
        return ArIsPackageRelativePath(text) ? Unpacked(path, text)
                                             : text;
    }

    /// The file inside a package at the package-relative path
    /// `packaged`, unpacked into the package cache once per version
    /// of the package, or `packaged` itself when it cannot be.
    std::string Unpacked(const std::string& path,
                         const std::string& packaged) {
        if (const auto it = unpacked_.find(packaged);
            it != unpacked_.end()) {
            return it->second;
        }
        const auto fail = [&](const std::string& why) {
            Warn(path + ": " + packaged + " is inside a package; " +
                 why);
            unpacked_.emplace(packaged, packaged);
            return packaged;
        };
        if (request_.packageCache == nullptr) {
            return fail(
                "Ernie has no cache folder to unpack it into");
        }
        const std::shared_ptr<ArAsset> asset =
            ArGetResolver().OpenAsset(ArResolvedPath(packaged));
        const std::shared_ptr<const char> bytes =
            asset ? asset->GetBuffer() : nullptr;
        if (!bytes) {
            return fail("it cannot be read");
        }
        // A folder per package version: its path, size and time.
        const std::string package =
            ArSplitPackageRelativePathOuter(packaged).first;
        std::error_code error;
        const auto size = std::filesystem::file_size(package, error);
        const auto time =
            std::filesystem::last_write_time(package, error);
        const std::size_t key = std::hash<std::string>{}(
            packaged + "|" + std::to_string(size) + "|" +
            std::to_string(time.time_since_epoch().count()));
        const std::filesystem::path folder =
            std::filesystem::path(request_.packageCache) /
            TfStringPrintf("%016zx", key);
        const std::filesystem::path file =
            folder /
            std::filesystem::path(
                ArSplitPackageRelativePathInner(packaged).second)
                .filename();
        if (std::filesystem::file_size(file, error) !=
            asset->GetSize()) {
            std::filesystem::create_directories(folder, error);
            // Written aside and renamed, so a reader never sees half.
            const std::filesystem::path partial =
                file.string() + ".partial";
            {
                std::ofstream out(partial, std::ios::binary);
                out.write(bytes.get(), static_cast<std::streamsize>(
                                           asset->GetSize()));
                if (!out) {
                    return fail("it cannot be written to " +
                                folder.string());
                }
            }
            std::filesystem::rename(partial, file, error);
            if (error) {
                return fail("it cannot be written to " +
                            folder.string());
            }
        }
        unpacked_.emplace(packaged, file.string());
        return file.string();
    }

    /// Sets the attribute `name` on `path` to `value` if it is of a
    /// kind Ernie holds as a single value.
    void SetValue(const std::string& path, const std::string& name,
                  const VtValue& value) {
        if (value.IsHolding<bool>()) {
            sink_.SetBool(path.c_str(), name.c_str(),
                          value.UncheckedGet<bool>());
        } else if (value.IsHolding<int>()) {
            sink_.SetInt(path.c_str(), name.c_str(),
                         value.UncheckedGet<int>());
        } else if (value.IsHolding<TfToken>()) {
            sink_.SetToken(path.c_str(), name.c_str(),
                           value.UncheckedGet<TfToken>().GetText());
        } else if (value.IsHolding<std::string>()) {
            sink_.SetToken(path.c_str(), name.c_str(),
                           value.UncheckedGet<std::string>().c_str());
        } else if (value.IsHolding<SdfAssetPath>()) {
            const std::string text =
                Asset(path, value.UncheckedGet<SdfAssetPath>());
            sink_.SetToken(path.c_str(), name.c_str(), text.c_str());
        } else if (const auto numbers = NumbersOf(value)) {
            sink_.SetNumbers(path.c_str(), name.c_str(),
                             numbers->data(), numbers->size());
        } else if (value.IsHolding<VtFloatArray>()) {
            const VtFloatArray& a =
                value.UncheckedGet<VtFloatArray>();
            sink_.WriteFloats(path.c_str(), name.c_str(), a.cdata(),
                              a.size(), 1);
        } else if (!value.IsEmpty()) {
            Warn(path + ": " + name + " is a " + value.GetTypeName() +
                 ", which Ernie does not take");
        }
    }

    /// Translates the prims under and at `top` (once).
    void TranslateTree(const SdfPath& top) {
        if (!trees_.insert(top).second) {
            return;
        }
        const HdSceneIndexPrimView view(flat_, top);
        for (auto it = view.begin(); it != view.end(); ++it) {
            if (!Translate(*it)) {
                it.SkipDescendants();
            }
        }
    }

    TfToken PurposeOf(const HdContainerDataSourceHandle& flat) const {
        const HdPurposeSchema purpose =
            HdPurposeSchema::GetFromParent(flat);
        return purpose.GetPurpose()
                   ? purpose.GetPurpose()->GetTypedValue(0.0f)
                   : TfToken();
    }

    /// Translates one prim; false when its children are to be
    /// skipped.
    bool Translate(const SdfPath& source) {
        const HdSceneIndexPrim prim = flat_->GetPrim(source);
        const TfToken& type = prim.primType;
        if (type == HdPrimTypeTokens->material ||
            type == HdPrimTypeTokens->geomSubset) {
            return false;  // With what binds them, and in their mesh.
        }
        // Non-imageable stage prims (physics joints and the like) and
        // render settings are not drawn.
        if (type.IsEmpty()) {
            if (const UsdPrim usd = stage_->GetPrimAtPath(source);
                usd && !usd.GetTypeName().IsEmpty() &&
                !usd.IsA<UsdGeomImageable>()) {
                return false;
            }
        } else if (TfStringStartsWith(type.GetString(), "render") ||
                   type == HdPrimTypeTokens->coordSys) {
            return false;
        }
        // Blend shapes go with their meshes (`WriteBlendShapes`);
        // skeletons are posed whatever their purpose (UsdSkelImaging
        // makes them guides).
        if (type == TfToken("skelBlendShape") ||
            type == TfToken("skelAnimation") ||
            type == HdPrimTypeTokens->extComputation) {
            return false;
        }
        const UsdPrim usd = stage_->GetPrimAtPath(source);
        const bool skeleton = usd && usd.IsA<UsdSkelSkeleton>();
        const TfToken purpose =
            skeleton ? TfToken() : PurposeOf(prim.dataSource);
        if ((purpose == UsdGeomTokens->guide && !request_.guide) ||
            (purpose == UsdGeomTokens->proxy && !request_.proxy)) {
            return false;
        }
        const std::string path = Map(source);
        const std::string ernieType = ErnieTypeOf(type, source);
        if (!Call(path, [&] {
                DefinePrim(path.c_str(), ernieType.c_str());
            })) {
            return false;
        }
        Mapped(source, path);
        const HdSceneIndexPrim local = input_->GetPrim(source);
        Transform(source, local.dataSource, path);
        Visibility(local.dataSource, path);
        TimeSamples(source, path);
        const HdContainerDataSourceHandle& ds = prim.dataSource;
        // A native instance: drawn by the instancer UsdImaging
        // gathered it into (outside the default prim), which takes
        // its place.
        if (const HdInstanceSchema instance =
                HdInstanceSchema::GetFromParent(ds);
            instance.GetInstancer()) {
            TranslateTree(
                instance.GetInstancer()->GetTypedValue(0.0f));
        }
        if (ernieType == "Skeleton") {
            TranslateSkeleton(
                UsdSkelSkeleton(stage_->GetPrimAtPath(source)), path);
            return false;
        }
        if (type == HdPrimTypeTokens->mesh) {
            TranslateMesh(source, ds, path);
        } else if (type == HdPrimTypeTokens->points) {
            TranslatePoints(ds, path);
        } else if (type == HdPrimTypeTokens->instancer) {
            TranslateInstancer(ds, path);
        } else if (type == HdPrimTypeTokens->camera) {
            TranslateCamera(ds, path);
        } else if (HdPrimTypeIsLight(type)) {
            TranslateLight(ds, path);
            if (usd && usd.HasAPI<UsdLuxLightAPI>()) {
                lights_.emplace_back(source, path);
            }
        } else if (ernieType == "Volume") {
            TranslateVolume(UsdVolVolume(usd), ds, path);
        } else if (ernieType == "OpenVDBAsset") {
            TranslateField(UsdVolOpenVDBAsset(usd), path);
        } else if (type == HdPrimTypeTokens->cube ||
                   type == HdPrimTypeTokens->sphere ||
                   type == HdPrimTypeTokens->cylinder ||
                   type == HdPrimTypeTokens->cone ||
                   type == HdPrimTypeTokens->capsule ||
                   type == HdPrimTypeTokens->plane) {
            TranslateShape(type, ds, path);
        }
        return true;
    }

    std::string ErnieTypeOf(const TfToken& type,
                            const SdfPath& source) {
        // UsdSkelImaging draws skeletons as guide meshes; Ernie poses
        // its own.
        if (const UsdPrim usd = stage_->GetPrimAtPath(source)) {
            if (usd.IsA<UsdSkelSkeleton>()) {
                return "Skeleton";
            }
            if (usd.IsA<UsdSkelRoot>()) {
                return "SkelRoot";
            }
            if (usd.IsA<UsdVolVolume>()) {
                return "Volume";
            }
            if (usd.IsA<UsdVolOpenVDBAsset>()) {
                return "OpenVDBAsset";
            }
        }
        static const std::unordered_map<TfToken, std::string,
                                        TfToken::HashFunctor>
            kTypes{
                {HdPrimTypeTokens->mesh, "Mesh"},
                {HdPrimTypeTokens->points, "Points"},
                {HdPrimTypeTokens->instancer, "PointInstancer"},
                {HdPrimTypeTokens->camera, "Camera"},
                {HdPrimTypeTokens->cube, "Cube"},
                {HdPrimTypeTokens->sphere, "Sphere"},
                {HdPrimTypeTokens->cylinder, "Cylinder"},
                {HdPrimTypeTokens->cone, "Cone"},
                {HdPrimTypeTokens->capsule, "Capsule"},
                {HdPrimTypeTokens->plane, "Plane"},
                {HdPrimTypeTokens->distantLight, "DistantLight"},
                {HdPrimTypeTokens->domeLight, "DomeLight"},
                {HdPrimTypeTokens->sphereLight, "SphereLight"},
                {HdPrimTypeTokens->rectLight, "RectLight"},
                {HdPrimTypeTokens->diskLight, "DiskLight"},
                {HdPrimTypeTokens->cylinderLight, "CylinderLight"}};
        if (const auto it = kTypes.find(type); it != kTypes.end()) {
            return it->second;
        }
        if (!type.IsEmpty()) {
            Warn("Prims of type " + type.GetString() +
                 " are not translated yet; they become Xforms "
                 "(first: " +
                 source.GetString() + ")");
        }
        return "Xform";
    }

    void Transform(const SdfPath& source,
                   const HdContainerDataSourceHandle& local,
                   const std::string& path) {
        const HdXformSchema xform =
            HdXformSchema::GetFromParent(local);
        GfMatrix4d matrix(1.0);
        if (xform.GetMatrix()) {
            matrix = xform.GetMatrix()->GetTypedValue(0.0f);
        }
        if (xform.GetResetXformStack() &&
            xform.GetResetXformStack()->GetTypedValue(0.0f)) {
            // Ernie has no !resetXformStack!: the same world
            // transform, relative to the parent's.
            matrix =
                matrix * WorldOf(source.GetParentPath()).GetInverse();
        }
        if (source == default_) {
            matrix = matrix * correction_;
        }
        if (matrix != GfMatrix4d(1.0)) {
            Call(path, [&] { SetMatrix(path, matrix); });
        }
    }

    /// Records that the USD prim `source` became the Ernie prim
    /// `path`.
    void Mapped(const SdfPath& source, const std::string& path) {
        sink_.MapSource(source.GetText(), path.c_str());
        mapped_.emplace_back(source, path);
    }

    /// Sets `value` as the time sample at `time` of `name` on `path`,
    /// if it is of a kind Ernie holds as a single value.
    bool SetValueAt(const std::string& path, const std::string& name,
                    double time, const VtValue& value) {
        const char* p = path.c_str();
        const char* n = name.c_str();
        if (value.IsHolding<bool>()) {
            const double number =
                value.UncheckedGet<bool>() ? 1.0 : 0.0;
            sink_.SetNumbersAt(p, n, time, &number, 1);
        } else if (value.IsHolding<int>()) {
            const double number = value.UncheckedGet<int>();
            sink_.SetNumbersAt(p, n, time, &number, 1);
        } else if (value.IsHolding<TfToken>()) {
            sink_.SetTokenAt(p, n, time,
                             value.UncheckedGet<TfToken>().GetText());
        } else if (value.IsHolding<std::string>()) {
            sink_.SetTokenAt(
                p, n, time,
                value.UncheckedGet<std::string>().c_str());
        } else if (value.IsHolding<SdfAssetPath>()) {
            // A file per frame, such as a volume's grids.
            sink_.SetTokenAt(
                p, n, time,
                Asset(path, value.UncheckedGet<SdfAssetPath>())
                    .c_str());
        } else if (const auto numbers = NumbersOf(value)) {
            sink_.SetNumbersAt(p, n, time, numbers->data(),
                               numbers->size());
        } else if (value.IsHolding<VtVec3fArray>() &&
                   value.UncheckedGet<VtVec3fArray>().size() == 1) {
            // A constant primvar, such as displayColor.
            const GfVec3f& v = value.UncheckedGet<VtVec3fArray>()[0];
            const double numbers[] = {v[0], v[1], v[2]};
            sink_.SetNumbersAt(p, n, time, numbers, 3);
        } else if (value.IsHolding<VtFloatArray>()) {
            const VtFloatArray& a =
                value.UncheckedGet<VtFloatArray>();
            const std::size_t shape[] = {a.size()};
            sink_.WriteFloatArrayAt(p, n, time, a.cdata(), shape, 1);
        } else {
            return false;
        }
        return true;
    }

    /// Every other time-sampled attribute of the translated prims
    /// that their Ernie prims declare (a camera's focal length, a
    /// light's intensity, a shader's inputs), as time samples;
    /// transforms, visibility, points and skeletal animation come
    /// their own ways, and arrays per element (normals, primvars of
    /// many values) are read at one time.
    void SampledAttributes() {
        static const std::set<std::string> kHandled{
            "points",     "normals",          "visibility",
            "extent",     "faceVertexCounts", "faceVertexIndices",
            "velocities", "accelerations",    "xformOpOrder"};
        for (const auto& [source, path] : mapped_) {
            const UsdPrim prim = stage_->GetPrimAtPath(source);
            // A SkelAnimation's samples come in as its arrays over
            // `ernie:times` (UsdSkelAnimQuery).
            if (!prim || prim.IsA<UsdSkelAnimation>()) {
                continue;
            }
            for (const UsdAttribute& attribute :
                 prim.GetAttributes()) {
                const std::string name =
                    attribute.GetName().GetString();
                if (kHandled.contains(name) ||
                    TfStringStartsWith(name, "xformOp:") ||
                    TfStringStartsWith(name, "skel:") ||
                    TfStringStartsWith(name, "primvars:skel:")) {
                    continue;
                }
                std::vector<double> times;
                if (!attribute.GetTimeSamples(&times) ||
                    times.size() < 2) {
                    continue;
                }
                if (!sink_.Declares(path.c_str(), name.c_str())) {
                    Warn(path + ": its animated " + name +
                         " has no Ernie attribute; it is not "
                         "animated");
                    continue;
                }
                Call(path, [&] {
                    for (const double time : times) {
                        VtValue value;
                        attribute.Get(&value, time);
                        if (!SetValueAt(path, name, time, value)) {
                            Warn(path + ": " + name + " is a " +
                                 value.GetTypeName() +
                                 ", whose samples Ernie does not "
                                 "take");
                            return;
                        }
                    }
                });
            }
        }
    }

    /// The prim's sampled local transform and visibility (as the
    /// stage composes them, through UsdGeom) as Ernie's time samples.
    void TimeSamples(const SdfPath& source, const std::string& path) {
        const UsdPrim prim = stage_->GetPrimAtPath(source);
        if (!prim) {
            return;  // Hydra's own prims, such as gathered
                     // prototypes.
        }
        if (const UsdGeomXformable xformable(prim); xformable) {
            std::vector<double> times;
            if (xformable.GetTimeSamples(&times) &&
                times.size() > 1) {
                Call(path, [&] {
                    for (const double time : times) {
                        GfMatrix4d matrix(1.0);
                        bool resets = false;
                        xformable.GetLocalTransformation(
                            &matrix, &resets, time);
                        if (resets) {
                            Warn(
                                path +
                                ": its animated transform resets the "
                                "transform stack; it is read at one "
                                "time");
                            return;
                        }
                        if (source == default_) {
                            matrix = matrix * correction_;
                        }
                        sink_.SetNumbersAt(path.c_str(),
                                           "xformOp:transform", time,
                                           matrix.data(), 16);
                    }
                });
            }
        }
        if (const UsdGeomImageable imageable(prim); imageable) {
            std::vector<double> times;
            const UsdAttribute visibility =
                imageable.GetVisibilityAttr();
            if (visibility.GetTimeSamples(&times) &&
                times.size() > 1) {
                Call(path, [&] {
                    for (const double time : times) {
                        TfToken token;
                        visibility.Get(&token, time);
                        sink_.SetTokenAt(
                            path.c_str(), "visibility", time,
                            token == UsdGeomTokens->invisible
                                ? "invisible"
                                : "inherited");
                    }
                });
            }
        }
    }

    /// A mesh's sampled points (a point cache) at `out`'s points, as
    /// time samples.
    void PointSamples(const SdfPath& source, const std::string& path,
                      const MeshOut& out) {
        const UsdGeomPointBased points(stage_->GetPrimAtPath(source));
        std::vector<double> times;
        if (!points ||
            !points.GetPointsAttr().GetTimeSamples(&times) ||
            times.size() < 2) {
            return;
        }
        Call(path, [&] {
            std::vector<float> values(3 * out.sourceOf.size());
            for (const double time : times) {
                VtVec3fArray at;
                points.GetPointsAttr().Get(&at, time);
                if (at.size() != out.vertexOf.size()) {
                    Warn(path + ": its points change in number over "
                                "time; they "
                                "are read at one time");
                    return;
                }
                for (std::size_t v = 0; v < out.sourceOf.size();
                     ++v) {
                    const GfVec3f& p =
                        at[static_cast<std::size_t>(out.sourceOf[v])];
                    values[3 * v] = p[0];
                    values[3 * v + 1] = p[1];
                    values[3 * v + 2] = p[2];
                }
                const std::size_t shape[] = {out.sourceOf.size(), 3};
                sink_.WriteFloatArrayAt(path.c_str(), "points", time,
                                        values.data(), shape, 2);
            }
        });
    }

    /// The world transform of the scene index prim at `path`, from
    /// its ancestors' local ones.
    GfMatrix4d WorldOf(const SdfPath& path) const {
        GfMatrix4d world(1.0);
        for (SdfPath p = path;
             !p.IsEmpty() && p != SdfPath::AbsoluteRootPath();
             p = p.GetParentPath()) {
            const HdXformSchema xform = HdXformSchema::GetFromParent(
                input_->GetPrim(p).dataSource);
            if (xform.GetMatrix()) {
                world =
                    world * xform.GetMatrix()->GetTypedValue(0.0f);
            }
            if (xform.GetResetXformStack() &&
                xform.GetResetXformStack()->GetTypedValue(0.0f)) {
                break;
            }
        }
        return world;
    }

    void Visibility(const HdContainerDataSourceHandle& local,
                    const std::string& path) {
        const HdVisibilitySchema visibility =
            HdVisibilitySchema::GetFromParent(local);
        if (visibility.GetVisibility() &&
            !visibility.GetVisibility()->GetTypedValue(0.0f)) {
            Call(path, [&] {
                sink_.SetToken(path.c_str(), "visibility",
                               "invisible");
            });
        }
    }

    /// A constant displayColor, which shapes take as their colour.
    void ConstantColor(const HdContainerDataSourceHandle& ds,
                       const std::string& path) {
        const Flat color = FlatPrimvar(
            HdPrimvarsSchema::GetFromParent(ds).GetPrimvar(
                HdTokens->displayColor));
        if (color.width == 3 && !color.values.empty()) {
            Call(path, [&] {
                sink_.WriteFloats(path.c_str(),
                                  "primvars:displayColor",
                                  color.values.data(), 1, 3);
            });
        }
    }

    void TranslateShape(const TfToken& type,
                        const HdContainerDataSourceHandle& ds,
                        const std::string& path) {
        const auto number =
            [&](const char* name,
                const HdDoubleDataSourceHandle& value) {
                if (value) {
                    const double v = value->GetTypedValue(0.0f);
                    Call(path, [&] {
                        sink_.SetNumbers(path.c_str(), name, &v, 1);
                    });
                }
            };
        const auto token = [&](const char* name,
                               const HdTokenDataSourceHandle& value) {
            if (value) {
                Call(path, [&] {
                    sink_.SetToken(
                        path.c_str(), name,
                        value->GetTypedValue(0.0f).GetText());
                });
            }
        };
        if (type == HdPrimTypeTokens->cube) {
            number("size", HdCubeSchema::GetFromParent(ds).GetSize());
        } else if (type == HdPrimTypeTokens->sphere) {
            number("radius",
                   HdSphereSchema::GetFromParent(ds).GetRadius());
        } else if (type == HdPrimTypeTokens->cylinder) {
            const HdCylinderSchema s =
                HdCylinderSchema::GetFromParent(ds);
            number("height", s.GetHeight());
            number("radius", s.GetRadius());
            token("axis", s.GetAxis());
        } else if (type == HdPrimTypeTokens->cone) {
            const HdConeSchema s = HdConeSchema::GetFromParent(ds);
            number("height", s.GetHeight());
            number("radius", s.GetRadius());
            token("axis", s.GetAxis());
        } else if (type == HdPrimTypeTokens->capsule) {
            const HdCapsuleSchema s =
                HdCapsuleSchema::GetFromParent(ds);
            number("height", s.GetHeight());
            number("radius", s.GetRadius());
            token("axis", s.GetAxis());
        } else if (type == HdPrimTypeTokens->plane) {
            const HdPlaneSchema s = HdPlaneSchema::GetFromParent(ds);
            number("width", s.GetWidth());
            number("length", s.GetLength());
            token("axis", s.GetAxis());
        }
        ConstantColor(ds, path);
        Bind(ds, path);
    }

    void TranslateCamera(const HdContainerDataSourceHandle& ds,
                         const std::string& path) {
        const HdCameraSchema camera =
            HdCameraSchema::GetFromParent(ds);
        // Hydra holds apertures and focal length in scene units; USD
        // (and Ernie) in tenths of them.
        const auto tenths = [&](const char* name,
                                const HdFloatDataSourceHandle& value,
                                double unit) {
            if (value) {
                const double v =
                    static_cast<double>(value->GetTypedValue(0.0f)) /
                    unit;
                Call(path, [&] {
                    sink_.SetNumbers(path.c_str(), name, &v, 1);
                });
            }
        };
        tenths("focalLength", camera.GetFocalLength(),
               GfCamera::FOCAL_LENGTH_UNIT);
        tenths("horizontalAperture", camera.GetHorizontalAperture(),
               GfCamera::APERTURE_UNIT);
        tenths("verticalAperture", camera.GetVerticalAperture(),
               GfCamera::APERTURE_UNIT);
        tenths("horizontalApertureOffset",
               camera.GetHorizontalApertureOffset(),
               GfCamera::APERTURE_UNIT);
        tenths("verticalApertureOffset",
               camera.GetVerticalApertureOffset(),
               GfCamera::APERTURE_UNIT);
        tenths("fStop", camera.GetFStop(), 1.0);
        tenths("focusDistance", camera.GetFocusDistance(), 1.0);
        if (camera.GetProjection()) {
            Call(path, [&] {
                sink_.SetToken(path.c_str(), "projection",
                               camera.GetProjection()
                                   ->GetTypedValue(0.0f)
                                   .GetText());
            });
        }
        if (camera.GetClippingRange()) {
            const GfVec2f range =
                camera.GetClippingRange()->GetTypedValue(0.0f);
            const double values[] = {range[0], range[1]};
            Call(path, [&] {
                sink_.SetNumbers(path.c_str(), "clippingRange",
                                 values, 2);
            });
        }
    }

    void TranslateLight(const HdContainerDataSourceHandle& ds,
                        const std::string& path) {
        const HdContainerDataSourceHandle light =
            HdContainerDataSource::Cast(
                ds->Get(HdLightSchemaTokens->light));
        if (!light) {
            return;
        }
        for (const char* name : kLightInputs) {
            const VtValue value = ValueOf(light->Get(TfToken(name)));
            if (!value.IsEmpty()) {
                Call(path, [&] {
                    SetValue(path, std::string("inputs:") + name,
                             value);
                });
            }
        }
        for (const char* name : kLightAttributes) {
            const VtValue value = ValueOf(light->Get(TfToken(name)));
            if (!value.IsEmpty()) {
                Call(path, [&] { SetValue(path, name, value); });
            }
        }
    }

    /// A UsdVol Volume: its `field:<name>` relationships to its
    /// field prims (translated wherever they are) and its material.
    void TranslateVolume(const UsdVolVolume& volume,
                         const HdContainerDataSourceHandle& ds,
                         const std::string& path) {
        for (const auto& [name, field] : volume.GetFieldPaths()) {
            TranslateTree(field);
            const std::string target = Map(field);
            const char* targets[] = {target.c_str()};
            Call(path, [&] {
                sink_.SetTargets(
                    path.c_str(),
                    ("field:" + name.GetString()).c_str(), targets, 1,
                    false);
            });
        }
        Bind(ds, path);
    }

    /// An OpenVDBAsset field: which grid of which file.
    void TranslateField(const UsdVolOpenVDBAsset& field,
                        const std::string& path) {
        for (const UsdAttribute& attribute :
             {field.GetFilePathAttr(), field.GetFieldNameAttr(),
              field.GetFieldIndexAttr(), field.GetFieldDataTypeAttr(),
              field.GetVectorDataRoleHintAttr(),
              field.GetFieldClassAttr()}) {
            VtValue value;
            if (attribute.HasAuthoredValue() &&
                attribute.Get(&value, time_)) {
                Call(path, [&] {
                    SetValue(path, attribute.GetName().GetString(),
                             value);
                });
            }
        }
    }

    /// Each light's light and shadow link collections, once every
    /// prim they may name is translated: their includes and excludes
    /// as Ernie paths, those outside the import dropped.
    void LightLinks() {
        std::unordered_map<SdfPath, std::string, SdfPath::Hash> ernie;
        for (const auto& [source, path] : mapped_) {
            ernie.emplace(source, path);
        }
        const auto targetOf =
            [&](const SdfPath& target) -> std::optional<std::string> {
            if (const auto it = ernie.find(target);
                it != ernie.end()) {
                return it->second;
            }
            if (default_.IsEmpty() || target.HasPrefix(default_)) {
                return Map(target);
            }
            return std::nullopt;
        };
        for (const auto& [source, path] : lights_) {
            const UsdLuxLightAPI light(stage_->GetPrimAtPath(source));
            for (const auto& [collection, link] :
                 {std::pair(light.GetLightLinkCollectionAPI(),
                            "collection:lightLink:"),
                  std::pair(light.GetShadowLinkCollectionAPI(),
                            "collection:shadowLink:")}) {
                if (!collection) {
                    continue;
                }
                const std::string prefix = link;
                if (const UsdAttribute expression =
                        collection.GetMembershipExpressionAttr();
                    expression && expression.HasAuthoredValue()) {
                    Warn(path + ": its " + prefix +
                         " membership expression is not supported; "
                         "only its includes and excludes link");
                }
                bool includeRoot = true;
                if (const UsdAttribute attribute =
                        collection.GetIncludeRootAttr();
                    attribute.HasAuthoredValue() &&
                    attribute.Get(&includeRoot)) {
                    Call(path, [&] {
                        sink_.SetBool(
                            path.c_str(),
                            (prefix + "includeRoot").c_str(),
                            includeRoot);
                    });
                }
                for (const auto& [relationship, name] :
                     {std::pair(collection.GetIncludesRel(),
                                "includes"),
                      std::pair(collection.GetExcludesRel(),
                                "excludes")}) {
                    SdfPathVector targets;
                    if (!relationship ||
                        !relationship.GetTargets(&targets) ||
                        targets.empty()) {
                        continue;
                    }
                    std::vector<std::string> paths;
                    for (const SdfPath& target : targets) {
                        if (const auto mapped = targetOf(target)) {
                            paths.push_back(*mapped);
                        } else {
                            Warn(path + ": its " + prefix + name +
                                 " target " + target.GetString() +
                                 " is outside the import; it is "
                                 "dropped");
                        }
                    }
                    std::vector<const char*> texts;
                    for (const std::string& p : paths) {
                        texts.push_back(p.c_str());
                    }
                    Call(path, [&] {
                        sink_.SetTargets(
                            path.c_str(), (prefix + name).c_str(),
                            texts.data(), texts.size(), true);
                    });
                }
            }
        }
    }

    /// The stage's RenderSettings, RenderProducts and RenderVars,
    /// wherever they are (usually /Render, outside the default prim),
    /// with the attributes and relationships Ernie's declare; the
    /// cameras they name are translated too.
    void RenderPrims() {
        for (const UsdPrim& prim : stage_->Traverse()) {
            const char* type =
                prim.IsA<UsdRenderSettings>()  ? "RenderSettings"
                : prim.IsA<UsdRenderProduct>() ? "RenderProduct"
                : prim.IsA<UsdRenderVar>()     ? "RenderVar"
                                               : nullptr;
            if (type == nullptr) {
                continue;
            }
            const std::string path = Map(prim.GetPath());
            if (!Call(path,
                      [&] { DefinePrim(path.c_str(), type); })) {
                continue;
            }
            Mapped(prim.GetPath(), path);
            for (const UsdAttribute& attribute :
                 prim.GetAuthoredAttributes()) {
                std::string name = attribute.GetName().GetString();
                VtValue value;
                if (!sink_.Declares(path.c_str(), name.c_str()) ||
                    !attribute.Get(&value, time_)) {
                    continue;
                }
                if (name == "sourceName") {
                    value = RenderVarName(path, value);
                }
                Call(path, [&] { SetValue(path, name, value); });
            }
            for (const UsdRelationship& relationship :
                 prim.GetAuthoredRelationships()) {
                const std::string name =
                    relationship.GetName().GetString();
                SdfPathVector targets;
                if (!sink_.Declares(path.c_str(), name.c_str()) ||
                    !relationship.GetForwardedTargets(&targets)) {
                    continue;
                }
                std::vector<std::string> paths;
                for (const SdfPath& target : targets) {
                    // Cameras outside the default prim come in too.
                    if (const UsdPrim t =
                            stage_->GetPrimAtPath(target);
                        t && t.IsA<UsdGeomCamera>()) {
                        TranslateTree(target);
                    }
                    paths.push_back(Map(target));
                }
                std::vector<const char*> texts;
                for (const std::string& p : paths) {
                    texts.push_back(p.c_str());
                }
                Call(path, [&] {
                    sink_.SetTargets(path.c_str(), name.c_str(),
                                     texts.data(), texts.size(),
                                     true);
                });
            }
        }
    }

    /// A RenderVar's source as Ernie names its render vars: USD's
    /// usual names (Hydra's AOVs and RenderMan's) for the ones Ernie
    /// has, others as they are.
    VtValue RenderVarName(const std::string& path,
                          const VtValue& value) {
        std::string name;
        if (value.IsHolding<std::string>()) {
            name = value.UncheckedGet<std::string>();
        } else if (value.IsHolding<TfToken>()) {
            name = value.UncheckedGet<TfToken>().GetString();
        } else {
            return value;
        }
        static const std::unordered_map<std::string, std::string>
            kNames{{"color", "HdrColor"},
                   {"Ci", "HdrColor"},
                   {"rgba", "HdrColor"},
                   {"depth", "DistanceToImagePlane"},
                   {"z", "DistanceToImagePlane"},
                   {"normal", "Normal"},
                   {"Neye", "Normal"},
                   {"N", "Normal"},
                   {"primId", "PrimId"},
                   {"instanceId", "InstanceId"},
                   {"motion", "MotionVector"},
                   {"dPdtime", "MotionVector"}};
        if (const auto it = kNames.find(name); it != kNames.end()) {
            return VtValue(it->second);
        }
        if (name.find_first_of("<>'.") != std::string::npos) {
            Warn(path + ": its source " + name +
                 " is a light path expression; Ernie renders its "
                 "colour (HdrColor)");
            return VtValue(std::string("HdrColor"));
        }
        return value;
    }

    void TranslatePoints(const HdContainerDataSourceHandle& ds,
                         const std::string& path) {
        const HdPrimvarsSchema primvars =
            HdPrimvarsSchema::GetFromParent(ds);
        const auto points = TypedOf<VtVec3fArray>(
            primvars.GetPrimvar(HdTokens->points).GetPrimvarValue());
        const auto widths = TypedOf<VtFloatArray>(
            primvars.GetPrimvar(HdTokens->widths).GetPrimvarValue());
        Call(path, [&] {
            if (points) {
                sink_.WriteFloats(
                    path.c_str(), "points",
                    reinterpret_cast<const float*>(points->cdata()),
                    points->size(), 3);
            }
            if (widths && !widths->empty()) {
                sink_.WriteFloats(path.c_str(), "widths",
                                  widths->cdata(), widths->size(), 1);
            }
        });
        ConstantColor(ds, path);
        Bind(ds, path);
    }

    /// An instancer (a PointInstancer's, or native instances'
    /// gathered by UsdImaging): instances of its prototypes,
    /// translated too.
    void TranslateInstancer(const HdContainerDataSourceHandle& ds,
                            const std::string& path) {
        const HdInstancerTopologySchema topology =
            HdInstancerTopologySchema::GetFromParent(ds);
        const VtArray<SdfPath> prototypes =
            topology.GetPrototypes()
                ? topology.GetPrototypes()->GetTypedValue(0.0f)
                : VtArray<SdfPath>();
        // Each instance's prototype, from each prototype's instances.
        std::vector<std::int32_t> protoIndices;
        const auto lists = topology.GetInstanceIndices();
        for (std::size_t p = 0; p < lists.GetNumElements(); ++p) {
            const HdIntArrayDataSourceHandle list =
                lists.GetElement(p);
            if (!list) {
                continue;
            }
            for (const int instance : list->GetTypedValue(0.0f)) {
                if (instance < 0) {
                    continue;
                }
                if (static_cast<std::size_t>(instance) >=
                    protoIndices.size()) {
                    protoIndices.resize(
                        static_cast<std::size_t>(instance) + 1, 0);
                }
                protoIndices[static_cast<std::size_t>(instance)] =
                    static_cast<std::int32_t>(p);
            }
        }
        const std::size_t count = protoIndices.size();
        std::vector<float> positions(3 * count, 0.0f);
        std::vector<float> orientations;
        std::vector<float> scales;
        const HdPrimvarsSchema primvars =
            HdPrimvarsSchema::GetFromParent(ds);
        const auto value = [&](const TfToken& name) {
            return ValueOf(
                primvars.GetPrimvar(name).GetFlattenedPrimvarValue());
        };
        if (const VtValue transforms =
                value(HdInstancerTokens->instanceTransforms);
            transforms.IsHolding<VtMatrix4dArray>()) {
            const VtMatrix4dArray& m =
                transforms.UncheckedGet<VtMatrix4dArray>();
            orientations.assign(4 * count, 0.0f);
            scales.assign(3 * count, 1.0f);
            for (std::size_t i = 0; i < count && i < m.size(); ++i) {
                GfVec3f t, s;
                GfQuatf q;
                Decompose(m[i], t, q, s);
                const GfVec3f im = q.GetImaginary();
                std::copy_n(t.data(), 3, &positions[3 * i]);
                std::copy_n(s.data(), 3, &scales[3 * i]);
                orientations[4 * i] = im[0];
                orientations[4 * i + 1] = im[1];
                orientations[4 * i + 2] = im[2];
                orientations[4 * i + 3] = q.GetReal();
            }
        }
        if (const VtValue t =
                value(HdInstancerTokens->instanceTranslations);
            t.IsHolding<VtVec3fArray>()) {
            const VtVec3fArray& v = t.UncheckedGet<VtVec3fArray>();
            for (std::size_t i = 0; i < count && i < v.size(); ++i) {
                std::copy_n(v[i].data(), 3, &positions[3 * i]);
            }
        }
        if (const VtValue r =
                value(HdInstancerTokens->instanceRotations);
            !r.IsEmpty()) {
            orientations.assign(4 * count, 0.0f);
            const auto write = [&](std::size_t i, const GfVec3d& im,
                                   double real) {
                orientations[4 * i] = static_cast<float>(im[0]);
                orientations[4 * i + 1] = static_cast<float>(im[1]);
                orientations[4 * i + 2] = static_cast<float>(im[2]);
                orientations[4 * i + 3] = static_cast<float>(real);
            };
            if (r.IsHolding<VtQuathArray>()) {
                const VtQuathArray& q =
                    r.UncheckedGet<VtQuathArray>();
                for (std::size_t i = 0; i < count && i < q.size();
                     ++i) {
                    write(i, GfVec3d(q[i].GetImaginary()),
                          q[i].GetReal());
                }
            } else if (r.IsHolding<VtQuatfArray>()) {
                const VtQuatfArray& q =
                    r.UncheckedGet<VtQuatfArray>();
                for (std::size_t i = 0; i < count && i < q.size();
                     ++i) {
                    write(i, GfVec3d(q[i].GetImaginary()),
                          q[i].GetReal());
                }
            }
        }
        if (const VtValue s =
                value(HdInstancerTokens->instanceScales);
            s.IsHolding<VtVec3fArray>()) {
            const VtVec3fArray& v = s.UncheckedGet<VtVec3fArray>();
            scales.assign(3 * count, 1.0f);
            for (std::size_t i = 0; i < count && i < v.size(); ++i) {
                std::copy_n(v[i].data(), 3, &scales[3 * i]);
            }
        }
        std::vector<std::string> targets;
        for (const SdfPath& prototype : prototypes) {
            targets.push_back(Map(prototype));
        }
        std::vector<const char*> texts;
        for (const std::string& target : targets) {
            texts.push_back(target.c_str());
        }
        Call(path, [&] {
            const char* p = path.c_str();
            sink_.WriteFloats(p, "positions", positions.data(), count,
                              3);
            if (!orientations.empty()) {
                sink_.WriteFloats(p, "orientations",
                                  orientations.data(), count, 4);
            }
            if (!scales.empty()) {
                sink_.WriteFloats(p, "scales", scales.data(), count,
                                  3);
            }
            sink_.WriteInts(p, "protoIndices", protoIndices.data(),
                            count);
            sink_.SetTargets(p, "prototypes", texts.data(),
                             texts.size(), true);
        });
        if (const VtBoolArray mask =
                topology.GetMask()
                    ? topology.GetMask()->GetTypedValue(0.0f)
                    : VtBoolArray();
            std::ranges::any_of(mask,
                                [](bool shown) { return !shown; })) {
            Warn(path +
                 ": instance masks (invisibleIds) are not supported; "
                 "every instance is drawn");
        }
        for (const SdfPath& prototype : prototypes) {
            TranslateTree(prototype);
        }
    }

    void TranslateMesh(const SdfPath& sourcePath,
                       const HdContainerDataSourceHandle& ds,
                       const std::string& path) {
        const HdMeshSchema mesh = HdMeshSchema::GetFromParent(ds);
        const HdMeshTopologySchema topology = mesh.GetTopology();
        const HdPrimvarsSchema primvars =
            HdPrimvarsSchema::GetFromParent(ds);
        MeshSource source;
        if (const auto points = TypedOf<VtVec3fArray>(
                primvars.GetPrimvar(HdTokens->points)
                    .GetPrimvarValue())) {
            source.points = *points;
        }
        // A skinned mesh's rest points, which Ernie skins itself (in
        // place of UsdSkelImaging's posed ones).
        const auto skinned = skinning_.find(sourcePath);
        if (skinned != skinning_.end()) {
            VtVec3fArray rest;
            if (UsdGeomPointBased(stage_->GetPrimAtPath(sourcePath))
                    .GetPointsAttr()
                    .Get(&rest, time_)) {
                source.points = rest;
            }
        }
        if (topology.GetFaceVertexCounts()) {
            source.counts =
                topology.GetFaceVertexCounts()->GetTypedValue(0.0f);
        }
        if (topology.GetFaceVertexIndices()) {
            source.indices =
                topology.GetFaceVertexIndices()->GetTypedValue(0.0f);
        }
        source.leftHanded = topology.GetOrientation() &&
                            topology.GetOrientation()->GetTypedValue(
                                0.0f) == HdTokens->leftHanded;
        std::size_t corners = 0;
        for (const int count : source.counts) {
            source.starts.push_back(corners);
            corners += static_cast<std::size_t>(std::max(count, 0));
        }
        if (corners != source.indices.size()) {
            Warn(path +
                 ": faceVertexCounts and faceVertexIndices disagree; "
                 "the mesh is skipped");
            return;
        }
        for (const int index : source.indices) {
            if (index < 0 || static_cast<std::size_t>(index) >=
                                 source.points.size()) {
                Warn(
                    path +
                    ": a face vertex index is out of range; the mesh "
                    "is skipped");
                return;
            }
        }
        const TfToken scheme =
            mesh.GetSubdivisionScheme()
                ? mesh.GetSubdivisionScheme()->GetTypedValue(0.0f)
                : TfToken("catmullClark");
        const bool subdivided = scheme != TfToken("none");
        // A subdivision surface's normals are the surface's (USD
        // ignores authored ones); colours that would split its points
        // (tearing it) become one. A skinned mesh's normals are
        // recomputed as it moves, so its points stay welded.
        if (!subdivided && skinned == skinning_.end()) {
            source.normals =
                FlatPrimvar(primvars.GetPrimvar(HdTokens->normals));
            if (source.normals.width != 3) {
                source.normals = Flat{};
            }
        }
        source.color =
            FlatPrimvar(primvars.GetPrimvar(HdTokens->displayColor));
        if (source.color.width != 3) {
            source.color = Flat{};
        }
        if (subdivided && PerCorner(source.color) &&
            !source.color.values.empty()) {
            source.color.values.resize(3);
            source.color.interpolation =
                HdPrimvarSchemaTokens->constant;
        }
        for (const TfToken& name : primvars.GetPrimvarNames()) {
            if (name == HdTokens->points ||
                name == HdTokens->normals ||
                name == HdTokens->displayColor) {
                continue;
            }
            const HdPrimvarSchema primvar = primvars.GetPrimvar(name);
            if (IsTexCoords(name.GetString(), primvar)) {
                Flat uv = FlatPrimvar(primvar);
                if (uv.width == 2) {
                    source.uvs.emplace_back(name.GetString(),
                                            std::move(uv));
                }
            }
        }
        CheckSizes(path, source);

        // Subsets bound to materials other than the mesh's become
        // child meshes; the faces in none stay on this one.
        const SdfPath meshMaterial = MaterialOf(ds);
        std::vector<bool> taken(source.counts.size(), false);
        for (const SdfPath& child :
             flat_->GetChildPrimPaths(sourcePath)) {
            const HdSceneIndexPrim subset = flat_->GetPrim(child);
            if (subset.primType != HdPrimTypeTokens->geomSubset) {
                continue;
            }
            const HdGeomSubsetSchema schema =
                HdGeomSubsetSchema::GetFromParent(subset.dataSource);
            const SdfPath material = MaterialOf(subset.dataSource);
            if (!schema.GetIndices() || material == meshMaterial ||
                (schema.GetType() &&
                 schema.GetType()->GetTypedValue(0.0f) !=
                     HdGeomSubsetSchemaTokens->typeFaceSet)) {
                continue;
            }
            std::vector<std::size_t> list;
            for (const int face :
                 schema.GetIndices()->GetTypedValue(0.0f)) {
                if (face >= 0 &&
                    static_cast<std::size_t>(face) < taken.size() &&
                    !taken[static_cast<std::size_t>(face)]) {
                    taken[static_cast<std::size_t>(face)] = true;
                    list.push_back(static_cast<std::size_t>(face));
                }
            }
            if (list.empty()) {
                continue;
            }
            const std::string childPath = Map(child);
            if (!Call(childPath, [&] {
                    DefinePrim(childPath.c_str(), "Mesh");
                })) {
                continue;
            }
            Mapped(child, childPath);
            const MeshOut out = BuildMesh(source, list);
            WriteMesh(childPath, out, source);
            if (skinned != skinning_.end()) {
                WriteSkinning(childPath, out, skinned->second);
            } else {
                PointSamples(sourcePath, childPath, out);
            }
            if (subdivided) {
                WriteSubdivision(childPath, mesh, scheme, out, list);
                Warn(path +
                     ": its material subsets are refined apart, so "
                     "their borders are the surface's boundaries");
            }
            BindTo(material, childPath);
        }
        std::vector<std::size_t> rest;
        for (std::size_t face = 0; face < taken.size(); ++face) {
            if (!taken[face]) {
                rest.push_back(face);
            }
        }
        if (!rest.empty()) {
            const MeshOut out = BuildMesh(source, rest);
            WriteMesh(path, out, source);
            if (skinned != skinning_.end()) {
                WriteSkinning(path, out, skinned->second);
            } else {
                PointSamples(sourcePath, path, out);
            }
            if (subdivided) {
                WriteSubdivision(path, mesh, scheme, out, rest);
            }
            BindTo(meshMaterial, path);
        }
    }

    /// A mesh bound to a skeleton, and how.
    struct Skinned {
        SdfPath skeleton;
        UsdSkelSkinningQuery query;
    };

    /// The skinned meshes under the stage's SkelRoots, with UsdSkel's
    /// resolution of their bindings, before the tree is translated.
    void CollectSkinning() {
        const Usd_PrimFlagsPredicate predicate =
            UsdTraverseInstanceProxies(UsdPrimDefaultPredicate);
        const UsdPrimRange range(stage_->GetPseudoRoot(), predicate);
        for (auto it = range.begin(); it != range.end(); ++it) {
            if (!it->IsA<UsdSkelRoot>()) {
                continue;
            }
            it.PruneChildren();
            const UsdSkelRoot root(*it);
            skelCache_.Populate(root, predicate);
            std::vector<UsdSkelBinding> bindings;
            skelCache_.ComputeSkelBindings(root, &bindings,
                                           predicate);
            for (const UsdSkelBinding& binding : bindings) {
                for (const UsdSkelSkinningQuery& query :
                     binding.GetSkinningTargets()) {
                    if (query.HasJointInfluences() ||
                        query.HasBlendShapes()) {
                        skinning_.emplace(
                            query.GetPrim().GetPath(),
                            Skinned{binding.GetSkeleton()
                                        .GetPrim()
                                        .GetPath(),
                                    query});
                    }
                }
            }
        }
    }

    void SetTokenList(const std::string& path, const char* attribute,
                      const VtTokenArray& tokens) {
        std::vector<const char*> texts;
        texts.reserve(tokens.size());
        for (const TfToken& token : tokens) {
            texts.push_back(token.GetText());
        }
        sink_.SetTokens(path.c_str(), attribute, texts.data(),
                        texts.size());
    }

    /// Row-vector USD matrices as Ernie's (joints, 4, 4):
    /// column-vector ones, each stored row by row.
    void WriteMatrices(const std::string& path, const char* attribute,
                       const VtMatrix4dArray& matrices) {
        std::vector<float> values;
        values.reserve(16 * matrices.size());
        for (const GfMatrix4d& m : matrices) {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    values.push_back(static_cast<float>(m[c][r]));
                }
            }
        }
        const std::size_t shape[] = {matrices.size(), 4, 4};
        sink_.WriteFloatArray(path.c_str(), attribute, values.data(),
                              shape, 3);
    }

    void TranslateSkeleton(const UsdSkelSkeleton& skel,
                           const std::string& path) {
        const UsdSkelSkeletonQuery query =
            skelCache_.GetSkelQuery(skel);
        if (!query) {
            Warn(path + ": its skeleton is invalid; it is not posed");
            return;
        }
        Call(path, [&] {
            const VtTokenArray& joints = query.GetJointOrder();
            SetTokenList(path, "joints", joints);
            VtMatrix4dArray bind;
            VtMatrix4dArray rest;
            if (skel.GetBindTransformsAttr().Get(&bind) &&
                bind.size() == joints.size()) {
                WriteMatrices(path, "bindTransforms", bind);
            } else {
                Warn(path + ": it needs a bind transform per joint");
            }
            if (query.ComputeJointLocalTransforms(&rest, time_,
                                                  true) &&
                rest.size() == joints.size()) {
                WriteMatrices(path, "restTransforms", rest);
            }
            if (const UsdSkelAnimQuery& anim = query.GetAnimQuery()) {
                const std::string animation =
                    AnimationPath(anim, path);
                const char* target = animation.c_str();
                sink_.SetTargets(path.c_str(), "skel:animationSource",
                                 &target, 1, false);
            }
        });
    }

    /// The Ernie SkelAnimation of `anim`, translated once: where its
    /// USD path maps if its parent was translated, else under the
    /// first skeleton it poses.
    std::string AnimationPath(const UsdSkelAnimQuery& anim,
                              const std::string& skeleton) {
        const SdfPath source = anim.GetPrim().GetPath();
        if (const auto it = animations_.find(source);
            it != animations_.end()) {
            return it->second;
        }
        std::string path = Map(source);
        if (!defined_.contains(Map(source.GetParentPath()))) {
            path = skeleton + "/" + source.GetName();
        }
        animations_.emplace(source, path);
        Call(path, [&] {
            DefinePrim(path.c_str(), "SkelAnimation");
            Mapped(source, path);
            TranslateAnimation(anim, path);
        });
        return path;
    }

    /// An animation's joint transforms and blend shape weights at
    /// each of its time samples.
    void TranslateAnimation(const UsdSkelAnimQuery& anim,
                            const std::string& path) {
        const VtTokenArray joints = anim.GetJointOrder();
        const VtTokenArray shapes = anim.GetBlendShapeOrder();
        SetTokenList(path, "joints", joints);
        SetTokenList(path, "blendShapes", shapes);
        std::vector<double> times;
        std::vector<double> weightTimes;
        anim.GetJointTransformTimeSamples(&times);
        anim.GetBlendShapeWeightTimeSamples(&weightTimes);
        times.insert(times.end(), weightTimes.begin(),
                     weightTimes.end());
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()),
                    times.end());
        const bool sampled = !times.empty();
        std::vector<UsdTimeCode> codes;
        for (const double t : times) {
            codes.emplace_back(t);
        }
        if (!sampled) {
            codes.push_back(time_);
        }
        std::vector<float> translations;
        std::vector<float> rotations;
        std::vector<float> scales;
        std::vector<float> weights;
        bool transforms = !joints.empty();
        bool weighted = !shapes.empty();
        for (const UsdTimeCode& code : codes) {
            VtVec3fArray t;
            VtQuatfArray r;
            VtVec3hArray s;
            if (transforms &&
                anim.ComputeJointLocalTransformComponents(&t, &r, &s,
                                                          code) &&
                t.size() == joints.size() &&
                r.size() == joints.size() &&
                s.size() == joints.size()) {
                for (std::size_t j = 0; j < joints.size(); ++j) {
                    const GfVec3f& im = r[j].GetImaginary();
                    translations.insert(translations.end(),
                                        {t[j][0], t[j][1], t[j][2]});
                    rotations.insert(
                        rotations.end(),
                        {im[0], im[1], im[2], r[j].GetReal()});
                    scales.insert(scales.end(),
                                  {static_cast<float>(s[j][0]),
                                   static_cast<float>(s[j][1]),
                                   static_cast<float>(s[j][2])});
                }
            } else {
                transforms = false;
            }
            VtFloatArray w;
            if (weighted && anim.ComputeBlendShapeWeights(&w, code) &&
                w.size() == shapes.size()) {
                weights.insert(weights.end(), w.begin(), w.end());
            } else {
                weighted = false;
            }
        }
        const char* p = path.c_str();
        const std::size_t n = codes.size();
        if (sampled) {
            std::vector<float> codesOut(times.begin(), times.end());
            sink_.WriteFloats(p, "ernie:times", codesOut.data(), n,
                              1);
        }
        if (transforms) {
            const std::size_t vec3[] = {n, joints.size(), 3};
            const std::size_t vec4[] = {n, joints.size(), 4};
            sink_.WriteFloatArray(p, "translations",
                                  translations.data(), vec3, 3);
            sink_.WriteFloatArray(p, "rotations", rotations.data(),
                                  vec4, 3);
            sink_.WriteFloatArray(p, "scales", scales.data(), vec3,
                                  3);
        } else if (!joints.empty()) {
            Warn(path + ": its joint transforms are incomplete; the "
                        "skeleton rests");
        }
        if (weighted) {
            sink_.WriteFloats(p, "blendShapeWeights", weights.data(),
                              n, shapes.size());
        }
    }

    /// `out`'s binding to its skeleton: its influences per output
    /// point (normalized, as UsdSkel expects), and its blend shapes.
    void WriteSkinning(const std::string& path, const MeshOut& out,
                       const Skinned& skinned) {
        const UsdSkelSkinningQuery& query = skinned.query;
        Call(path, [&] {
            const char* p = path.c_str();
            const std::string skeleton = Map(skinned.skeleton);
            const char* target = skeleton.c_str();
            sink_.SetTargets(p, "skel:skeleton", &target, 1, false);
            if (VtTokenArray order; query.GetJointOrder(&order)) {
                SetTokenList(path, "skel:joints", order);
            }
            const GfMatrix4d bind = query.GetGeomBindTransform(time_);
            sink_.SetNumbers(p, "primvars:skel:geomBindTransform",
                             bind.data(), 16);
            sink_.SetToken(p, "skel:skinningMethod",
                           query.GetSkinningMethod().GetText());
            VtIntArray indices;
            VtFloatArray weights;
            if (query.HasJointInfluences() &&
                query.ComputeJointInfluences(&indices, &weights,
                                             time_)) {
                const auto n = static_cast<std::size_t>(
                    query.GetNumInfluencesPerComponent());
                UsdSkelNormalizeWeights(weights, static_cast<int>(n));
                if (query.IsRigidlyDeformed()) {
                    const std::size_t shape[] = {1, n};
                    sink_.WriteIntArray(p,
                                        "primvars:skel:jointIndices",
                                        indices.cdata(), shape, 2);
                    sink_.WriteFloatArray(
                        p, "primvars:skel:jointWeights",
                        weights.cdata(), shape, 2);
                } else {
                    std::vector<std::int32_t> pointIndices;
                    std::vector<float> pointWeights;
                    for (const std::int32_t source : out.sourceOf) {
                        const auto first =
                            static_cast<std::size_t>(source) * n;
                        pointIndices.insert(pointIndices.end(),
                                            indices.cbegin() + first,
                                            indices.cbegin() + first +
                                                n);
                        pointWeights.insert(pointWeights.end(),
                                            weights.cbegin() + first,
                                            weights.cbegin() + first +
                                                n);
                    }
                    const std::size_t shape[] = {out.sourceOf.size(),
                                                 n};
                    sink_.WriteIntArray(
                        p, "primvars:skel:jointIndices",
                        pointIndices.data(), shape, 2);
                    sink_.WriteFloatArray(
                        p, "primvars:skel:jointWeights",
                        pointWeights.data(), shape, 2);
                }
            }
        });
        if (query.HasBlendShapes()) {
            WriteBlendShapes(path, out, query);
        }
    }

    /// `out`'s blend shapes as BlendShape prims under it, their
    /// offsets at its points (a source point split into several moves
    /// them all).
    void WriteBlendShapes(const std::string& path, const MeshOut& out,
                          const UsdSkelSkinningQuery& query) {
        VtTokenArray names;
        SdfPathVector targets;
        query.GetBlendShapeOrder(&names);
        query.GetBlendShapeTargetsRel().GetTargets(&targets);
        if (names.size() != targets.size()) {
            Warn(path + ": its blend shapes and their targets "
                        "disagree; they are skipped");
            return;
        }
        std::vector<std::vector<std::int32_t>> outputs(
            out.vertexOf.size());
        for (std::size_t v = 0; v < out.sourceOf.size(); ++v) {
            outputs[static_cast<std::size_t>(out.sourceOf[v])]
                .push_back(static_cast<std::int32_t>(v));
        }
        std::vector<std::string> shapePaths;
        for (const SdfPath& target : targets) {
            const std::string shapePath =
                path + "/" + target.GetName();
            shapePaths.push_back(shapePath);
            const UsdSkelBlendShape shape(
                stage_->GetPrimAtPath(target));
            if (!shape) {
                Warn(path + ": its blend shape " +
                     target.GetString() + " is missing");
                continue;
            }
            VtIntArray sourceIndices;
            shape.GetPointIndicesAttr().Get(&sourceIndices);
            // Each offset at the output points of its source point.
            std::vector<std::int32_t> pointIndices;
            std::vector<std::size_t> from;
            VtVec3fArray offsets;
            shape.GetOffsetsAttr().Get(&offsets);
            for (std::size_t k = 0; k < offsets.size(); ++k) {
                const int source =
                    sourceIndices.empty()
                        ? static_cast<int>(k)
                        : (k < sourceIndices.size() ? sourceIndices[k]
                                                    : -1);
                if (source < 0 || static_cast<std::size_t>(source) >=
                                      outputs.size()) {
                    continue;
                }
                for (const std::int32_t v :
                     outputs[static_cast<std::size_t>(source)]) {
                    pointIndices.push_back(v);
                    from.push_back(k);
                }
            }
            const auto gather = [&](const VtVec3fArray& values) {
                std::vector<float> gathered;
                for (const std::size_t k : from) {
                    const GfVec3f value =
                        k < values.size() ? values[k] : GfVec3f(0.0f);
                    gathered.insert(gathered.end(),
                                    {value[0], value[1], value[2]});
                }
                return gathered;
            };
            Call(shapePath, [&] {
                DefinePrim(shapePath.c_str(), "BlendShape");
                Mapped(target, shapePath);
                const char* p = shapePath.c_str();
                const std::vector<float> moved = gather(offsets);
                sink_.WriteFloats(p, "offsets", moved.data(),
                                  from.size(), 3);
                sink_.WriteInts(p, "pointIndices",
                                pointIndices.data(),
                                pointIndices.size());
                std::vector<float> inbetweenWeights;
                std::vector<float> inbetweenOffsets;
                for (const UsdSkelInbetweenShape& inbetween :
                     shape.GetInbetweens()) {
                    float weight = 0.0f;
                    VtVec3fArray values;
                    if (inbetween.GetWeight(&weight) &&
                        inbetween.GetOffsets(&values)) {
                        inbetweenWeights.push_back(weight);
                        const std::vector<float> at = gather(values);
                        inbetweenOffsets.insert(
                            inbetweenOffsets.end(), at.begin(),
                            at.end());
                    }
                }
                if (!inbetweenWeights.empty()) {
                    sink_.WriteFloats(p, "ernie:inbetweenWeights",
                                      inbetweenWeights.data(),
                                      inbetweenWeights.size(), 1);
                    const std::size_t shape3[] = {
                        inbetweenWeights.size(), from.size(), 3};
                    sink_.WriteFloatArray(p, "ernie:inbetweenOffsets",
                                          inbetweenOffsets.data(),
                                          shape3, 3);
                }
            });
        }
        Call(path, [&] {
            SetTokenList(path, "skel:blendShapes", names);
            std::vector<const char*> texts;
            for (const std::string& shapePath : shapePaths) {
                texts.push_back(shapePath.c_str());
            }
            sink_.SetTargets(path.c_str(), "skel:blendShapeTargets",
                             texts.data(), texts.size(), true);
        });
    }

    /// Drops primvars whose size does not fit their interpolation.
    void CheckSizes(const std::string& path, MeshSource& source) {
        const auto expected = [&](const Flat& flat) -> std::size_t {
            if (flat.interpolation == HdPrimvarSchemaTokens->vertex ||
                flat.interpolation ==
                    HdPrimvarSchemaTokens->varying) {
                return source.points.size();
            }
            if (flat.interpolation ==
                HdPrimvarSchemaTokens->uniform) {
                return source.counts.size();
            }
            if (flat.interpolation ==
                HdPrimvarSchemaTokens->faceVarying) {
                return source.indices.size();
            }
            return 1;
        };
        const auto check = [&](Flat& flat, const std::string& what) {
            if (flat.Valid() &&
                flat.values.size() / flat.width < expected(flat)) {
                Warn(path + ": " + what +
                     " has too few values for its " +
                     flat.interpolation.GetString() +
                     " interpolation; it is dropped");
                flat = Flat{};
            }
        };
        check(source.normals, "normals");
        check(source.color, "displayColor");
        for (auto& [name, uv] : source.uvs) {
            check(uv, "primvars:" + name);
        }
        std::erase_if(source.uvs, [](const auto& uv) {
            return !uv.second.Valid();
        });
    }

    void WriteMesh(const std::string& path, const MeshOut& out,
                   const MeshSource& source) {
        Call(path, [&] {
            const char* p = path.c_str();
            sink_.WriteFloats(p, "points", out.points.data(),
                              out.points.size() / 3, 3);
            sink_.WriteInts(p, "faceVertexCounts", out.counts.data(),
                            out.counts.size());
            sink_.WriteInts(p, "faceVertexIndices",
                            out.indices.data(), out.indices.size());
            if (!out.normals.empty()) {
                sink_.WriteFloats(p, "normals", out.normals.data(),
                                  out.normals.size() / 3, 3);
            }
            if (!out.colors.empty()) {
                sink_.WriteFloats(p, "primvars:displayColor",
                                  out.colors.data(),
                                  out.colors.size() / 3, 3);
            } else if (source.color.Valid()) {
                sink_.WriteFloats(p, "primvars:displayColor",
                                  source.color.values.data(), 1, 3);
            }
            for (const auto& [name, values] : out.uvs) {
                const std::string attribute = "primvars:" + name;
                sink_.WriteFloats(p, attribute.c_str(), values.data(),
                                  values.size() / 2, 2);
                const std::string interpolation =
                    attribute + ":interpolation";
                sink_.SetToken(p, interpolation.c_str(),
                               "faceVarying");
            }
        });
    }

    /// The subdivision tags of `mesh` on the Ernie mesh `path`, `out`
    /// made of `faces` of it: point and face indices remapped, and
    /// the creases and corners at points it lacks dropped.
    void WriteSubdivision(const std::string& path,
                          const HdMeshSchema& mesh,
                          const TfToken& scheme, const MeshOut& out,
                          const std::vector<std::size_t>& faces) {
        const HdSubdivisionTagsSchema tags =
            mesh.GetSubdivisionTags();
        const auto token = [](const HdTokenDataSourceHandle& value,
                              const char* fallback) {
            return value ? value->GetTypedValue(0.0f)
                         : TfToken(fallback);
        };
        const auto ints =
            [](const HdIntArrayDataSourceHandle& value) {
                return value ? value->GetTypedValue(0.0f)
                             : VtIntArray();
            };
        const auto floats =
            [](const HdFloatArrayDataSourceHandle& value) {
                return value ? value->GetTypedValue(0.0f)
                             : VtFloatArray();
            };
        const TfToken boundary =
            token(tags.GetInterpolateBoundary(), "edgeAndCorner");
        const TfToken linear = token(
            tags.GetFaceVaryingLinearInterpolation(), "cornersPlus1");
        const TfToken triangles =
            token(tags.GetTriangleSubdivisionRule(), "catmullClark");
        const VtIntArray creases = ints(tags.GetCreaseIndices());
        const VtIntArray lengths = ints(tags.GetCreaseLengths());
        const VtFloatArray sharpnesses =
            floats(tags.GetCreaseSharpnesses());
        const VtIntArray corners = ints(tags.GetCornerIndices());
        const VtFloatArray cornerSharpnesses =
            floats(tags.GetCornerSharpnesses());
        const VtIntArray holes =
            ints(mesh.GetTopology().GetHoleIndices());
        const auto vertex = [&](int index) {
            return index >= 0 && static_cast<std::size_t>(index) <
                                     out.vertexOf.size()
                       ? out.vertexOf[static_cast<std::size_t>(index)]
                       : -1;
        };
        bool dropped = false;
        std::vector<std::int32_t> outCreases, outLengths;
        std::vector<float> outSharpnesses;
        const bool perChain = sharpnesses.size() == lengths.size();
        std::size_t first = 0;
        std::size_t edge = 0;
        for (std::size_t chain = 0; chain < lengths.size(); ++chain) {
            const auto length =
                static_cast<std::size_t>(std::max(lengths[chain], 0));
            std::vector<std::int32_t> mapped;
            for (std::size_t k = 0;
                 k < length && first + k < creases.size(); ++k) {
                if (const int v = vertex(creases[first + k]);
                    v >= 0) {
                    mapped.push_back(v);
                }
            }
            const bool sharp =
                perChain ? chain < sharpnesses.size()
                         : edge + length - 1 <= sharpnesses.size();
            if (length >= 2 && mapped.size() == length && sharp) {
                outCreases.insert(outCreases.end(), mapped.begin(),
                                  mapped.end());
                outLengths.push_back(
                    static_cast<std::int32_t>(length));
                if (perChain) {
                    outSharpnesses.push_back(sharpnesses[chain]);
                } else {
                    outSharpnesses.insert(
                        outSharpnesses.end(),
                        sharpnesses.begin() +
                            static_cast<std::ptrdiff_t>(edge),
                        sharpnesses.begin() +
                            static_cast<std::ptrdiff_t>(edge +
                                                        length - 1));
                }
            } else {
                dropped = true;
            }
            first += length;
            edge += length > 0 ? length - 1 : 0;
        }
        std::vector<std::int32_t> outCorners;
        std::vector<float> outCornerSharpnesses;
        for (std::size_t i = 0;
             i < corners.size() && i < cornerSharpnesses.size();
             ++i) {
            if (const int v = vertex(corners[i]); v >= 0) {
                outCorners.push_back(v);
                outCornerSharpnesses.push_back(cornerSharpnesses[i]);
            } else {
                dropped = true;
            }
        }
        std::unordered_map<std::size_t, std::int32_t> faceAt;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            faceAt.emplace(faces[i], static_cast<std::int32_t>(i));
        }
        std::vector<std::int32_t> outHoles;
        for (const int hole : holes) {
            if (const auto it =
                    faceAt.find(static_cast<std::size_t>(hole));
                hole >= 0 && it != faceAt.end()) {
                outHoles.push_back(it->second);
            }
        }
        Call(path, [&] {
            const char* p = path.c_str();
            sink_.SetToken(p, "subdivisionScheme", scheme.GetText());
            sink_.SetToken(p, "interpolateBoundary",
                           boundary.GetText());
            sink_.SetToken(p, "faceVaryingLinearInterpolation",
                           linear.GetText());
            sink_.SetToken(p, "triangleSubdivisionRule",
                           triangles.GetText());
            if (!outLengths.empty()) {
                sink_.WriteInts(p, "creaseIndices", outCreases.data(),
                                outCreases.size());
                sink_.WriteInts(p, "creaseLengths", outLengths.data(),
                                outLengths.size());
                sink_.WriteFloats(p, "creaseSharpnesses",
                                  outSharpnesses.data(),
                                  outSharpnesses.size(), 1);
            }
            if (!outCorners.empty()) {
                sink_.WriteInts(p, "cornerIndices", outCorners.data(),
                                outCorners.size());
                sink_.WriteFloats(p, "cornerSharpnesses",
                                  outCornerSharpnesses.data(),
                                  outCornerSharpnesses.size(), 1);
            }
            if (!outHoles.empty()) {
                sink_.WriteInts(p, "holeIndices", outHoles.data(),
                                outHoles.size());
            }
        });
        if (dropped) {
            Warn(path + ": creases or corners outside its faces, or "
                        "of mismatched "
                        "sharpnesses, are dropped");
        }
    }

    // -----------------------------------------------------------------
    // Materials

    /// The material a prim is bound to (resolved by UsdImaging,
    /// inherited by flattening): its full binding, else its
    /// all-purpose one.
    static SdfPath MaterialOf(const HdContainerDataSourceHandle& ds) {
        const HdMaterialBindingsSchema bindings =
            HdMaterialBindingsSchema::GetFromParent(ds);
        for (const TfToken& purpose :
             {TfToken("full"),
              HdMaterialBindingsSchemaTokens->allPurpose}) {
            const HdMaterialBindingSchema binding =
                bindings.GetMaterialBinding(purpose);
            if (binding.GetPath()) {
                const SdfPath path =
                    binding.GetPath()->GetTypedValue(0.0f);
                if (!path.IsEmpty()) {
                    return path;
                }
            }
        }
        return SdfPath();
    }

    void Bind(const HdContainerDataSourceHandle& ds,
              const std::string& path) {
        BindTo(MaterialOf(ds), path);
    }

    void BindTo(const SdfPath& material, const std::string& path) {
        if (material.IsEmpty()) {
            return;
        }
        const std::string target = TranslateMaterial(material);
        if (target.empty()) {
            return;
        }
        const char* targets[] = {target.c_str()};
        Call(path, [&] {
            sink_.SetTargets(path.c_str(), "material:binding",
                             targets, 1, false);
        });
    }

    /// Translates the material at `source` once; its Ernie path, or
    /// empty when it cannot be.
    std::string TranslateMaterial(const SdfPath& source) {
        if (const auto it = materials_.find(source);
            it != materials_.end()) {
            return it->second;
        }
        materials_[source] = "";
        const HdSceneIndexPrim prim = flat_->GetPrim(source);
        const HdMaterialSchema material =
            HdMaterialSchema::GetFromParent(prim.dataSource);
        const std::string path = Map(source);
        if (!material || !Call(path, [&] {
                DefinePrim(path.c_str(), "Material");
            })) {
            return "";
        }
        Mapped(source, path);
        // UsdPreviewSurface (the universal render context) first,
        // then MDL, approximated.
        const HdMaterialNetworkSchema universal =
            material.GetMaterialNetwork(
                HdMaterialSchemaTokens->universalRenderContext);
        const HdMaterialConnectionSchema surface =
            universal.GetTerminals().Get(
                HdMaterialTerminalTokens->surface);
        const HdMaterialConnectionSchema mdl =
            material.GetMaterialNetwork(TfToken("mdl"))
                .GetTerminals()
                .Get(HdMaterialTerminalTokens->surface);
        if (const std::string node = NodeName(surface);
            !node.empty() &&
            IdOf(universal, node) == "UsdPreviewSurface") {
            const std::string shader =
                TranslateNode(universal, node, source);
            if (!shader.empty()) {
                ConnectSurface(
                    path, shader + ".outputs:" + OutputName(surface));
            }
        } else if (const std::string mdlNode = NodeName(mdl);
                   !mdlNode.empty()) {
            ApproximateMdl(path,
                           material.GetMaterialNetwork(TfToken("mdl"))
                               .GetNodes()
                               .Get(TfToken(mdlNode)));
        } else {
            Warn(source.GetString() +
                 " has no UsdPreviewSurface or MDL surface; its "
                 "prims keep their displayColor");
        }
        materials_[source] = path;
        return path;
    }

    static std::string
    NodeName(const HdMaterialConnectionSchema& connection) {
        return connection && connection.GetUpstreamNodePath()
                   ? connection.GetUpstreamNodePath()
                         ->GetTypedValue(0.0f)
                         .GetString()
                   : std::string();
    }

    static std::string
    OutputName(const HdMaterialConnectionSchema& connection) {
        return connection.GetUpstreamNodeOutputName()
                   ? connection.GetUpstreamNodeOutputName()
                         ->GetTypedValue(0.0f)
                         .GetString()
                   : std::string("surface");
    }

    static std::string IdOf(const HdMaterialNetworkSchema& network,
                            const std::string& node) {
        const HdMaterialNodeSchema schema =
            network.GetNodes().Get(TfToken(node));
        return schema.GetNodeIdentifier()
                   ? schema.GetNodeIdentifier()
                         ->GetTypedValue(0.0f)
                         .GetString()
                   : std::string();
    }

    void ConnectSurface(const std::string& material,
                        const std::string& target) {
        Call(material, [&] {
            sink_.SetToken(material.c_str(),
                           "outputs:surface.connect", target.c_str());
        });
    }

    /// Translates the network node `name` (a shader's path, or one
    /// relative to its `material`) and those upstream of it, its
    /// parameters and connections as they are; its Ernie path.
    std::string TranslateNode(const HdMaterialNetworkSchema& network,
                              const std::string& name,
                              const SdfPath& material) {
        SdfPath source(name);
        if (!source.IsAbsolutePath()) {
            source = material.AppendPath(source);
        }
        if (const auto it = shaders_.find(source);
            it != shaders_.end()) {
            return it->second;
        }
        const std::string path = Map(source);
        shaders_[source] = path;
        if (!Call(path,
                  [&] { DefinePrim(path.c_str(), "Shader"); })) {
            return "";
        }
        Mapped(source, path);
        const HdMaterialNodeSchema node =
            network.GetNodes().Get(TfToken(name));
        const std::string id = IdOf(network, name);
        Call(path, [&] {
            sink_.SetToken(path.c_str(), "info:id", id.c_str());
        });
        const auto parameters = node.GetParameters();
        for (const TfToken& parameter : parameters.GetNames()) {
            const HdMaterialNodeParameterSchema value =
                parameters.Get(parameter);
            const VtValue v = ValueOf(value.GetValue());
            Call(path, [&] {
                SetValue(path, "inputs:" + parameter.GetString(), v);
            });
        }
        const auto connections = node.GetInputConnections();
        for (const TfToken& input : connections.GetNames()) {
            const auto list = connections.Get(input);
            if (list.GetNumElements() == 0) {
                continue;
            }
            const HdMaterialConnectionSchema connection =
                list.GetElement(0);
            const std::string upstream = NodeName(connection);
            if (upstream.empty()) {
                continue;
            }
            const std::string from =
                TranslateNode(network, upstream, material);
            if (from.empty()) {
                continue;
            }
            const std::string target =
                from + ".outputs:" + OutputName(connection);
            const std::string attribute =
                "inputs:" + input.GetString() + ".connect";
            Call(path, [&] {
                sink_.SetToken(path.c_str(), attribute.c_str(),
                               target.c_str());
            });
        }
        return path;
    }

    // -----------------------------------------------------------------
    // MDL, approximated by UsdPreviewSurface until Ernie compiles MDL

    /// The UsdPreviewSurface approximating an MDL material: OmniPBR's
    /// and OmniSurface's main inputs (base colour or texture,
    /// roughness, metallic, normal map, emission, opacity), else the
    /// MDL's defaults with a warning.
    void ApproximateMdl(const std::string& material,
                        const HdMaterialNodeSchema& mdl) {
        std::string module;
        if (const HdContainerDataSourceHandle info =
                mdl.GetNodeTypeInfo()) {
            if (const auto sub = TypedOf<TfToken>(info->Get(
                    TfToken("mdl:sourceAsset:subIdentifier")))) {
                module = sub->GetString();
            }
        }
        const auto parameters = mdl.GetParameters();
        const auto input = [&](const char* name) {
            return ValueOf(parameters.Get(TfToken(name)).GetValue());
        };
        const auto number =
            [&](const char* name) -> std::optional<double> {
            const auto n = NumbersOf(input(name));
            return n && n->size() == 1
                       ? std::optional<double>((*n)[0])
                       : std::nullopt;
        };
        const auto flag = [&](const char* name) {
            const VtValue v = input(name);
            return (v.IsHolding<bool>() && v.UncheckedGet<bool>()) ||
                   (v.IsHolding<int>() && v.UncheckedGet<int>() != 0);
        };
        const auto color =
            [&](const char* name) -> std::optional<GfVec3d> {
            const auto n = NumbersOf(input(name));
            return n && n->size() == 3
                       ? std::optional<GfVec3d>(
                             GfVec3d((*n)[0], (*n)[1], (*n)[2]))
                       : std::nullopt;
        };
        const auto file =
            [&](const char* name) -> std::optional<std::string> {
            const VtValue v = input(name);
            if (!v.IsHolding<SdfAssetPath>() ||
                v.UncheckedGet<SdfAssetPath>()
                    .GetAssetPath()
                    .empty()) {
                return std::nullopt;
            }
            return Asset(material, v.UncheckedGet<SdfAssetPath>());
        };

        const std::string surface = material + "/ErniePreviewSurface";
        if (!Call(surface,
                  [&] { DefinePrim(surface.c_str(), "Shader"); })) {
            return;
        }
        const char* s = surface.c_str();
        sink_.SetToken(s, "info:id", "UsdPreviewSurface");
        ConnectSurface(material, surface + ".outputs:surface");
        std::string reader;
        const auto st = [&]() -> const std::string& {
            if (reader.empty()) {
                reader = material + "/ErnieStReader";
                Call(reader, [&] {
                    DefinePrim(reader.c_str(), "Shader");
                    sink_.SetToken(reader.c_str(), "info:id",
                                   "UsdPrimvarReader_float2");
                    sink_.SetToken(reader.c_str(), "inputs:varname",
                                   "st");
                });
            }
            return reader;
        };
        const auto texture = [&](const char* name,
                                 const std::string& path,
                                 const char* colorSpace,
                                 const char* output,
                                 const char* target) {
            const std::string tex = material + "/" + name;
            Call(tex, [&] {
                DefinePrim(tex.c_str(), "Shader");
                sink_.SetToken(tex.c_str(), "info:id",
                               "UsdUVTexture");
                sink_.SetToken(tex.c_str(), "inputs:file",
                               path.c_str());
                sink_.SetToken(tex.c_str(), "inputs:sourceColorSpace",
                               colorSpace);
                const std::string reads = st() + ".outputs:result";
                sink_.SetToken(tex.c_str(), "inputs:st.connect",
                               reads.c_str());
                const std::string from = tex + "." + output;
                const std::string connect =
                    std::string(target) + ".connect";
                sink_.SetToken(s, connect.c_str(), from.c_str());
            });
            return tex;
        };
        const auto setNumber = [&](const char* name, double value) {
            Call(surface,
                 [&] { sink_.SetNumbers(s, name, &value, 1); });
        };
        const auto setColor = [&](const char* name,
                                  const GfVec3d& value) {
            Call(surface,
                 [&] { sink_.SetNumbers(s, name, value.data(), 3); });
        };

        const bool omniSurface = module.starts_with("OmniSurface");
        if (module != "OmniPBR" && module != "OmniPBR_ClearCoat" &&
            !omniSurface) {
            Warn(material + ": MDL material " +
                 (module.empty() ? std::string("(no subIdentifier)")
                                 : module) +
                 " is approximated by its base colour; Ernie does "
                 "not compile MDL yet");
        }
        if (omniSurface) {
            setColor("inputs:diffuseColor",
                     color("diffuse_reflection_color")
                             .value_or(GfVec3d(0.8)) *
                         number("diffuse_reflection_weight")
                             .value_or(0.8));
            if (const auto f =
                    file("diffuse_reflection_color_image")) {
                texture("ErnieDiffuseTexture", *f, "sRGB",
                        "outputs:rgb", "inputs:diffuseColor");
            }
            setNumber("inputs:roughness",
                      number("specular_reflection_roughness")
                          .value_or(0.2));
            setNumber("inputs:metallic",
                      number("metalness").value_or(0.0));
            if (flag("enable_emission")) {
                setColor(
                    "inputs:emissiveColor",
                    color("emission_color").value_or(GfVec3d(1.0)) *
                        number("emission_weight").value_or(1.0));
            }
            return;
        }
        // OmniPBR and anything else that names inputs the same way.
        GfVec3d base =
            color("diffuse_color_constant").value_or(GfVec3d(0.2));
        if (const auto tint = color("diffuse_tint")) {
            base = GfCompMult(base, *tint);
        }
        setColor("inputs:diffuseColor", base);
        if (const auto f = file("diffuse_texture")) {
            texture("ErnieDiffuseTexture", *f, "sRGB", "outputs:rgb",
                    "inputs:diffuseColor");
        }
        setNumber(
            "inputs:roughness",
            number("reflection_roughness_constant").value_or(0.5));
        if (const auto f = file("reflectionroughness_texture")) {
            texture("ErnieRoughnessTexture", *f, "raw", "outputs:r",
                    "inputs:roughness");
        }
        setNumber("inputs:metallic",
                  number("metallic_constant").value_or(0.0));
        if (const auto f = file("metallic_texture")) {
            texture("ErnieMetallicTexture", *f, "raw", "outputs:r",
                    "inputs:metallic");
        }
        if (const auto f = file("normalmap_texture")) {
            const std::string tex =
                texture("ErnieNormalTexture", *f, "raw",
                        "outputs:rgb", "inputs:normal");
            Call(tex, [&] {
                const double scale[] = {2.0, 2.0, 2.0, 1.0};
                const double bias[] = {-1.0, -1.0, -1.0, 0.0};
                sink_.SetNumbers(tex.c_str(), "inputs:scale", scale,
                                 4);
                sink_.SetNumbers(tex.c_str(), "inputs:bias", bias, 4);
            });
        }
        if (flag("enable_emission")) {
            setColor("inputs:emissiveColor",
                     color("emissive_color").value_or(GfVec3d(1.0)));
        }
        if (flag("enable_opacity")) {
            setNumber("inputs:opacity",
                      number("opacity_constant").value_or(1.0));
        }
    }

    const Request& request_;
    Sink& sink_;
    std::string root_;
    UsdTimeCode time_ = UsdTimeCode::EarliestTime();
    UsdStageRefPtr stage_;
    HdSceneIndexBaseRefPtr input_;
    HdSceneIndexBaseRefPtr flat_;
    SdfPath default_;
    bool external_ = false;
    GfMatrix4d correction_ = GfMatrix4d(1.0);
    std::set<SdfPath> trees_;
    std::unordered_set<std::string>
        defined_;  ///< Ernie prims defined.
    /// Each translated USD prim and its Ernie prim (`Mapped`).
    std::vector<std::pair<SdfPath, std::string>> mapped_;
    UsdSkelCache skelCache_;
    std::unordered_map<SdfPath, Skinned, SdfPath::Hash> skinning_;
    std::unordered_map<SdfPath, std::string, SdfPath::Hash>
        animations_;
    std::unordered_map<SdfPath, std::string, SdfPath::Hash>
        materials_;
    std::unordered_map<SdfPath, std::string, SdfPath::Hash> shaders_;
    /// Lights whose link collections wait for every prim to be
    /// translated (`LightLinks`).
    std::vector<std::pair<SdfPath, std::string>> lights_;
    /// Files unpacked from packages by their package-relative paths.
    std::unordered_map<std::string, std::string> unpacked_;
    std::unordered_set<std::string> warned_;
};

void CopyError(const std::string& message, char* error,
               std::size_t errorSize) {
    if (error == nullptr || errorSize == 0) {
        return;
    }
    const std::size_t n = std::min(message.size(), errorSize - 1);
    std::memcpy(error, message.data(), n);
    error[n] = '\0';
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) std::uint32_t
ernie_usd_abi_version() {
    return ernie::usd_plugin::kAbiVersion;
}

__attribute__((visibility("default"))) const char*
ernie_usd_version() {
    return ERNIE_USD_STRINGIFY(PXR_MAJOR_VERSION) "." ERNIE_USD_STRINGIFY(
        PXR_MINOR_VERSION) "." ERNIE_USD_STRINGIFY(PXR_PATCH_VERSION);
}

__attribute__((visibility("default"))) bool
ernie_usd_import(const ernie::usd_plugin::Request* request,
                 ernie::usd_plugin::Sink* sink, char* error,
                 std::size_t errorSize) {
    if (request == nullptr || sink == nullptr ||
        request->abiVersion != ernie::usd_plugin::kAbiVersion) {
        CopyError("The USD importer was called with another ABI",
                  error, errorSize);
        return false;
    }
    try {
        Importer(*request, *sink).Run();
        return true;
    } catch (const std::exception& e) {
        CopyError(e.what(), error, errorSize);
    } catch (...) {
        CopyError("The USD importer failed", error, errorSize);
    }
    return false;
}

}  // extern "C"
