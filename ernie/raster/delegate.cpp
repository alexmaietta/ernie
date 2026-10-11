// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/wide_view.h"
#include "ernie/session/product_frame.h"
#include "ernie/util/profile.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

/// The vars the view holds for a chain writing `aovs` AOVs, with
/// MotionVector when `motion` (it is shown): every var (empty)
/// without either, else every var and the first `aovs` AOV slots.
std::vector<std::string_view> ViewHeldVars(std::size_t aovs,
                                           bool motion) {
    if (aovs == 0 && !motion) {
        return {};
    }
    // Every var (the opt-in MotionVector aside unless shown) and the
    // slots.
    std::vector<std::string_view> vars = RequiredVars({});
    if (motion) {
        vars.push_back(render_vars::kMotionVector);
    }
    vars.insert(vars.end(), kAovSlotNames.begin(),
                kAovSlotNames.begin() +
                    static_cast<std::ptrdiff_t>(aovs));
    return vars;
}

float Number(const Scene& scene, const Path& path,
             std::string_view attribute, float fallback) {
    return static_cast<float>(
        GetNumber(scene, path, attribute, fallback));
}

/// How 8-bit values of `reader`'s image decode: sRGB unless it says
/// raw.
ImageEncoding EncodingOf(const TextureReader& reader) {
    return reader.colorSpace == TextureColorSpace::Raw
               ? ImageEncoding::Linear
               : ImageEncoding::Srgb;
}

/// What a texture's upload depends on: its source (and for a tensor,
/// the scene and its version, since another scene's prim at the same
/// path holds another one), colour space, wrapping, channel, and the
/// scale and bias that normal maps and one-channel textures apply
/// before upload.
std::string TextureKey(const Scene& scene,
                       const TextureReader& reader) {
    std::string key = reader.Source();
    if (!reader.image.empty()) {
        key += "@" + std::to_string(scene.Id()) + ":" +
               std::to_string(scene.AttributeVersion(
                   Path(reader.image), tokens::kShaderTexture));
    }
    key += "|" + std::to_string(static_cast<int>(reader.colorSpace)) +
           "|" + std::to_string(static_cast<int>(reader.wrapS)) +
           "|" + std::to_string(static_cast<int>(reader.wrapT)) +
           "|" + std::to_string(reader.channel);
    for (int c = 0; c < 4; ++c) {
        key += "|" + std::to_string(reader.scale[c]) + "|" +
               std::to_string(reader.bias[c]);
    }
    return key;
}

/// The image `reader` reads: its prim's `ernie:texture` tensor or its
/// file.
/// @throws std::invalid_argument for a tensor that is no image,
/// std::runtime_error for a file that cannot be read.
Image TextureImageOf(const Scene& scene,
                     const TextureReader& reader) {
    if (!reader.image.empty()) {
        const Tensor* tensor =
            Shader(scene, Path(reader.image)).GetTexture();
        if (tensor == nullptr) {
            throw std::invalid_argument(
                "'" + reader.image + "' has no " +
                std::string(tokens::kShaderTexture));
        }
        return ImageOfTensor(*tensor, EncodingOf(reader));
    }
    return LoadImageFile(reader.file, EncodingOf(reader));
}

std::invalid_argument UploadError(const Path& path,
                                  std::string_view attribute,
                                  const std::string& why) {
    return std::invalid_argument("Cannot upload " +
                                 std::string(attribute) + " of '" +
                                 path.String() + "': " + why);
}

/// The Material's albedo map `map` as a float32 raster with its grid.
RasterView RasterOf(const ernie::Material& material,
                    const Tensor& map) {
    if (map.Type() != DType::Float32 || map.Rank() != 2 ||
        map.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "expected a float32 (rows, cols) CPU tensor");
    }
    RasterView view;
    view.data = map.As<float>();
    view.height = static_cast<std::size_t>(map.Shape()[0]);
    view.width = static_cast<std::size_t>(map.Shape()[1]);
    const glm::dvec2 corner = material.GetAlbedoMapOrigin();
    view.originX = corner.x;
    view.originY = corner.y;
    view.cellSize = material.GetAlbedoMapCellSize();
    if (material.HasAlbedoMapNoData()) {
        view.noData =
            static_cast<float>(material.GetAlbedoMapNoData());
    }
    return view;
}

/// The Terrain's heights as a float32 raster with its grid.
RasterView RasterOf(const Terrain& terrain, const Tensor& heights) {
    if (heights.Type() != DType::Float32 || heights.Rank() != 2 ||
        heights.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "expected a float32 (rows, cols) CPU tensor");
    }
    RasterView view;
    view.data = heights.As<float>();
    view.height = static_cast<std::size_t>(heights.Shape()[0]);
    view.width = static_cast<std::size_t>(heights.Shape()[1]);
    const glm::dvec2 corner = terrain.GetOrigin();
    view.originX = corner.x;
    view.originY = corner.y;
    view.cellSize = terrain.GetCellSize();
    if (terrain.HasNoData()) {
        view.noData = static_cast<float>(terrain.GetNoData());
    }
    return view;
}

TerrainColorMode ColorModeOf(const std::string& token) {
    if (token == tokens::kHillshade) {
        return TerrainColorMode::Hillshade;
    }
    if (token == tokens::kColormap) {
        return TerrainColorMode::Colormap;
    }
    return TerrainColorMode::ShadedColormap;
}

/// A CPU array of `components`-vectors as floats, shaped (n,
/// components) or flat.
std::vector<float> FloatsOf(const Tensor& tensor,
                            std::size_t components,
                            std::string_view what) {
    const bool shaped =
        tensor.Rank() == 2 &&
        tensor.Shape()[1] == static_cast<std::int64_t>(components);
    if (tensor.Location().kind != DeviceKind::Cpu ||
        !(tensor.Rank() == 1 || shaped) ||
        static_cast<std::size_t>(tensor.Elements()) % components !=
            0) {
        throw std::invalid_argument(
            std::string(what) + " must be a CPU array of " +
            std::to_string(components) + "-vectors");
    }
    const Tensor packed = tensor.Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    if (packed.Type() == DType::Float32) {
        return {packed.As<float>(), packed.As<float>() + count};
    }
    if (packed.Type() == DType::Float64) {
        return {packed.As<double>(), packed.As<double>() + count};
    }
    throw std::invalid_argument(std::string(what) +
                                " must be float32 or float64");
}

/// The prim's attribute as an array: a written tensor or, for small
/// arrays, a float array value.
std::optional<std::vector<float>>
FloatAttribute(const Scene& scene, const Path& path,
               std::string_view attribute, std::size_t components) {
    if (const Tensor* tensor = scene.Read(path, attribute)) {
        return FloatsOf(*tensor, components, attribute);
    }
    return std::nullopt;
}

/// The constant colour of a Gprim's `primvars:displayColor`: a colour
/// value or a one-row array, else the declared fallback (colours per
/// vertex are its array, `ReadDisplayColor`).
glm::vec3 ConstantDisplayColor(const Gprim& gprim) {
    if (const Tensor* tensor = gprim.ReadDisplayColor();
        tensor != nullptr && tensor->Elements() == 3 &&
        tensor->Location().kind == DeviceKind::Cpu) {
        const std::vector<float> color =
            FloatsOf(*tensor, 3, tokens::kDisplayColor);
        return glm::vec3(color[0], color[1], color[2]);
    }
    return glm::vec3(gprim.GetDisplayColor());
}

/// The instance op of transform op `op` (`xformOp:translate`...),
/// composed on the GPU.
InstanceOpKind InstanceOpKindOf(std::string_view op) {
    using namespace tokens;
    constexpr std::pair<std::string_view, InstanceOpKind> kKinds[] = {
        {kTranslate, InstanceOpKind::Translate},
        {kScale, InstanceOpKind::Scale},
        {kOrient, InstanceOpKind::Orient},
        {kRotateX, InstanceOpKind::RotateX},
        {kRotateY, InstanceOpKind::RotateY},
        {kRotateZ, InstanceOpKind::RotateZ},
        {kRotateXYZ, InstanceOpKind::RotateXYZ},
        {kRotateXZY, InstanceOpKind::RotateXZY},
        {kRotateYXZ, InstanceOpKind::RotateYXZ},
        {kRotateYZX, InstanceOpKind::RotateYZX},
        {kRotateZXY, InstanceOpKind::RotateZXY},
        {kRotateZYX, InstanceOpKind::RotateZYX},
        {kTransform, InstanceOpKind::Transform}};
    for (const auto& [name, kind] : kKinds) {
        if (name == op) {
            return kind;
        }
    }
    throw std::invalid_argument(std::string(op) +
                                " is not composed on the GPU");
}

/// `between` as an instance parent's translate, rotation and scale.
/// @throws std::invalid_argument if it shears, which they cannot
/// hold.
void SetBetween(InstanceParent& parent, const glm::dmat4& between) {
    glm::dvec3 scale(glm::length(glm::dvec3(between[0])),
                     glm::length(glm::dvec3(between[1])),
                     glm::length(glm::dvec3(between[2])));
    if (glm::determinant(glm::dmat3(between)) < 0.0) {
        scale.x = -scale.x;
    }
    const glm::dmat3 rotation(glm::dvec3(between[0]) / scale.x,
                              glm::dvec3(between[1]) / scale.y,
                              glm::dvec3(between[2]) / scale.z);
    const glm::dquat q = glm::normalize(glm::quat_cast(rotation));
    const glm::dmat4 rebuilt =
        glm::translate(glm::dmat4(1.0), glm::dvec3(between[3])) *
        glm::dmat4(glm::mat3_cast(q)) *
        glm::scale(glm::dmat4(1.0), scale);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            if (std::abs(rebuilt[c][r] - between[c][r]) >
                1e-6 * std::max(1.0, std::abs(between[c][r]))) {
                throw std::invalid_argument(
                    "the transforms between it and the GPU-posed "
                    "prim "
                    "above it shear, which GPU poses cannot hold");
            }
        }
    }
    parent.translate = glm::vec3(between[3]);
    parent.rotation = glm::vec4(q.x, q.y, q.z, q.w);
    parent.scale = glm::vec3(scale);
}

/// Whether `change` is to a camera or render prim, which the frame
/// resolves anew anyway.
bool IsViewChange(const Scene& scene, const Change& change) {
    if (Any(change.bits, Dirty::Removed) || !scene.Has(change.path)) {
        return false;
    }
    const std::string_view type = scene.TypeOf(change.path);
    return type == tokens::kCamera ||
           type == tokens::kRenderSettings ||
           type == tokens::kRenderProduct ||
           type == tokens::kRenderVar;
}

/// A GPU-posed prim's `ernie:xform:extent`: six numbers as a float
/// array value or a CPU array.
std::optional<Bounds3> XformExtentOf(const Scene& scene,
                                     const Path& path) {
    std::vector<float> box;
    if (const Value* value = scene.Get(path, tokens::kXformExtent)) {
        if (const auto* floats = value->Get<std::vector<float>>()) {
            box = *floats;
        }
    } else if (const Tensor* tensor =
                   scene.Read(path, tokens::kXformExtent);
               tensor != nullptr &&
               tensor->Location().kind == DeviceKind::Cpu) {
        box = FloatsOf(*tensor, 3, tokens::kXformExtent);
    }
    if (box.size() != 6) {
        return std::nullopt;
    }
    return Bounds3({box[0], box[1], box[2]},
                   {box[3], box[4], box[5]});
}

std::string Describe(const Path& path, std::string_view what,
                     const std::exception& error) {
    return "'" + path.String() + "' " + std::string(what) + ": " +
           error.what();
}

/// `shape` averaged over each of `bands` (eight samples per band)
/// times `scale`.
template <typename Shape>
std::vector<float> BinShape(const Shape& shape,
                            const SpectralBands& bands,
                            double scale) {
    std::vector<float> values(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        double sum = 0.0;
        for (int k = 0; k < 8; ++k) {
            sum += shape(bands.Lower(i) +
                         (k + 0.5) * bands.Width() / 8.0);
        }
        values[static_cast<std::size_t>(i)] =
            static_cast<float>(scale * sum / 8.0);
    }
    return values;
}

}  // namespace

GroundParams GroundParamsOf(const Scene& scene, const Path& ground) {
    const GroundPlane plane(scene, ground);
    GroundParams params;
    params.height = static_cast<float>(plane.GetHeight());
    params.checker = plane.GetChecker();
    params.checkerSize = static_cast<float>(plane.GetCheckerSize());
    if (!(params.checkerSize > 0.0f) ||
        !std::isfinite(params.checkerSize)) {
        throw std::invalid_argument(
            "GroundPlane '" + ground.String() +
            "' needs a positive checker size");
    }
    params.checkerColors[0] = glm::vec3(plane.GetCheckerColor0());
    params.checkerColors[1] = glm::vec3(plane.GetCheckerColor1());
    return params;
}

TerrainParams TerrainParamsOf(const Scene& scene,
                              const RenderPrims& prims) {
    using namespace tokens;
    TerrainParams params;
    if (!prims.terrain.IsEmpty()) {
        params.exaggeration = static_cast<float>(
            Terrain(scene, prims.terrain).GetExaggeration());
    }
    const RenderSettings settings =
        RenderSettings::OrFallbacks(scene, prims.settings);
    params.colorMode = ColorModeOf(settings.GetColorMode());
    LodSettings& lod = params.lod;
    lod.enabled = settings.GetLodEnable();
    lod.detail = static_cast<float>(settings.GetLodDetail());
    lod.morphStart = static_cast<float>(settings.GetLodMorphStart());
    lod.freeze = settings.GetLodFreeze();
    lod.showLevels = settings.GetLodShowLevels();
    return params;
}

RasterDelegate::RasterDelegate(const VulkanContext& context,
                               VkFormat presentFormat,
                               BrdfRegistry brdfs)
    : context_(context),
      renderer_(context, presentFormat, std::move(brdfs)),
      terrain_(renderer_), planet_(renderer_), meshes_(renderer_),
      ground_(renderer_), points_(renderer_) {
    points_.SetAnchors(&meshes_);
}

RasterDelegate::~RasterDelegate() {
    context_.WaitForSubmissions();
}

const RendererInfo& RasterDelegate::Info() const {
    static const RendererInfo info = [] {
        RendererInfo raster{"raster", false, {}};
        for (const std::string_view name : kRenderVarNames) {
            if (RasterRenderer::CanPresent(name)) {
                raster.viewVars.emplace_back(name);
            }
        }
        return raster;
    }();
    return info;
}

std::optional<Bounds3> RasterDelegate::SceneBounds() const {
    std::optional<Bounds3> bounds;
    for (const std::optional<Bounds3>& drawable :
         {terrain_.Bounds(), planet_.Bounds(), meshes_.Bounds(),
          points_.Bounds()}) {
        if (!drawable) {
            continue;
        }
        bounds =
            bounds
                ? Bounds3{glm::min(bounds->first, drawable->first),
                          glm::max(bounds->second, drawable->second)}
                : *drawable;
    }
    // The drawables' bounds are relative to the render origin.
    if (bounds) {
        const glm::vec3 origin(origin_);
        bounds =
            Bounds3{bounds->first + origin, bounds->second + origin};
    }
    return bounds;
}

void RasterDelegate::AddSensorFilter(SensorFilterModel model) {
    renderer_.AddSensorFilter(std::move(model));
}

void RasterDelegate::AddLightPattern(LightPatternModel model) {
    renderer_.AddLightPattern(std::move(model));
}

const BrdfRegistry& RasterDelegate::Brdfs() const {
    return renderer_.Brdfs();
}

namespace {

/// The render targets of a frame's images: one set, or one per chunk
/// of a batch's views.
std::vector<RenderTargets*> TargetsOf(FrameImages& images) {
    std::vector<RenderTargets*> targets;
    if (auto* chunked = dynamic_cast<ChunkedFrameImages*>(&images)) {
        for (const std::unique_ptr<FrameImages>& chunk :
             chunked->Chunks()) {
            targets.push_back(
                static_cast<RenderTargets*>(chunk.get()));
        }
    } else {
        targets.push_back(static_cast<RenderTargets*>(&images));
    }
    return targets;
}

}  // namespace

RenderStats RasterDelegate::Stats() const {
    RenderStats stats;
    stats.gpuBytes =
        renderer_.GpuBytes() + terrain_.GpuBytes() +
        planet_.GpuBytes() +
        (viewTargets_ != nullptr ? viewTargets_->GpuBytes() : 0);
    for (const auto& [product, frames] : frames_) {
        for (const std::shared_ptr<ProductFrame>& frame : frames) {
            if (const auto* minimal =
                    dynamic_cast<const MinimalTargets*>(
                        frame->images.get())) {
                stats.gpuBytes += minimal->GpuBytes();
                continue;
            }
            for (const RenderTargets* targets :
                 TargetsOf(*frame->images)) {
                stats.gpuBytes += targets->GpuBytes();
            }
        }
    }
    stats.triangles = terrain_.TriangleCount() +
                      planet_.TriangleCount() +
                      meshes_.TriangleCount();
    stats.counters.emplace_back(raster_stats::kTerrainPatches,
                                terrain_.PatchCount() +
                                    planet_.PatchCount());
    stats.counters.emplace_back(
        raster_stats::kTerrainLodLevels,
        std::max(terrain_.Lod().Levels(), planet_.Lod().Levels()));
    stats.counters.emplace_back(raster_stats::kMeshDraws,
                                meshes_.DrawCount(0));
    stats.counters.emplace_back(raster_stats::kMeshDrawCalls,
                                meshes_.DrawCalls(0));
    return stats;
}

RasterDelegate::Key RasterDelegate::KeyOf(
    const Scene& scene, const Path& path,
    std::initializer_list<std::string_view> attributes) const {
    Key key{path, {}};
    key.second[0] = sceneId_;
    std::size_t i = 1;
    for (const std::string_view attribute : attributes) {
        key.second.at(i++) = scene.AttributeVersion(path, attribute);
    }
    return key;
}

std::optional<std::uint32_t>
RasterDelegate::MaterialId(const Path& path) const {
    const auto it = materials_.find(path);
    return it != materials_.end() ? std::optional(it->second.id)
                                  : std::nullopt;
}

std::uint32_t RasterDelegate::MaterialOf(const Scene& scene,
                                         const Path& prim) const {
    return MaterialId(BoundMaterial(scene, prim))
        .value_or(RasterRenderer::kDefaultMaterial);
}

void RasterDelegate::SyncMaterials(const Scene& scene) {
    ERNIE_ZONE("Sync materials");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    for (const Path& path : scene.Prims(kMaterial)) {
        seen.insert(path);
        auto [it, added] = materials_.try_emplace(path);
        SyncedMaterial& material = it->second;
        if (added) {
            material.id = renderer_.AddMaterial();
        }
        try {
            SyncMaterial(scene, path, material);
        } catch (const std::invalid_argument& error) {
            errors.emplace_back(error.what());
        }
    }
    for (auto it = materials_.begin(); it != materials_.end();) {
        if (seen.contains(it->first)) {
            ++it;
            continue;
        }
        renderer_.RemoveMaterial(it->second.id);
        it = materials_.erase(it);
    }
    if (!errors.empty()) {
        throw std::invalid_argument(errors.front());
    }
}

void RasterDelegate::SyncMaterial(const Scene& scene,
                                  const Path& path,
                                  SyncedMaterial& material) {
    using namespace tokens;
    const MaterialSettings settings =
        MaterialSettingsOf(scene, path, renderer_.Brdfs());
    renderer_.SetMaterial(material.id, settings);
    RasterMaterial& gpu = renderer_.Material(material.id);
    // The textures of a UsdPreviewSurface: albedo, normal map,
    // roughness and metallic.
    const std::array<const std::optional<TextureReader>*, 5> readers{
        &settings.albedoTexture, &settings.normalTexture,
        &settings.roughnessTexture, &settings.metallicTexture,
        &settings.opacityTexture};
    constexpr std::array<std::string_view, 5> kInputs{
        kDiffuseColor, kNormalInput, kRoughness, kMetallic, kOpacity};
    for (std::size_t slot = 0; slot < readers.size(); ++slot) {
        const std::optional<TextureReader>& reader = *readers[slot];
        const std::string key =
            reader ? TextureKey(scene, *reader) : std::string();
        if (material.textures[slot] == key) {
            continue;
        }
        material.textures[slot] = key;
        try {
            const std::optional<Image> image =
                reader ? std::optional(TextureImageOf(scene, *reader))
                       : std::nullopt;
            const Image* data = image ? &*image : nullptr;
            const TextureReader wrap =
                reader.value_or(TextureReader());
            if (slot == 0) {
                gpu.SetAlbedoTexture(data, wrap);
            } else if (slot == 1) {
                gpu.SetNormalTexture(data, wrap);
            } else {
                gpu.SetScalarTexture(
                    static_cast<RasterMaterial::Scalar>(slot - 2),
                    data, wrap);
            }
        } catch (const std::exception& error) {
            throw UploadError(path, kInputs[slot], error.what());
        }
    }

    const Key albedoKey =
        KeyOf(scene, path,
              {kAlbedoMap, kAlbedoMapOrigin, kAlbedoMapCellSize,
               kAlbedoMapNoData});
    const ernie::Material prim(scene, path);
    if (material.albedoMap != albedoKey) {
        material.albedoMap = albedoKey;
        try {
            if (const Tensor* tensor = prim.GetAlbedoMap()) {
                const Tensor albedo = tensor->Contiguous();
                gpu.SetAlbedoMap(RasterOf(prim, albedo));
            } else if (gpu.HasAlbedoMap()) {
                gpu.ClearAlbedoMap();
            }
        } catch (const std::invalid_argument& error) {
            throw UploadError(path, kAlbedoMap, error.what());
        }
    }
    for (const bool normal : {false, true}) {
        const std::string_view attribute =
            normal ? kDetailNormal : kDetailAlbedo;
        std::optional<Key>& cached =
            normal ? material.detailNormal : material.detailAlbedo;
        const Key key = KeyOf(scene, path, {attribute});
        if (cached == key) {
            continue;
        }
        cached = key;
        try {
            std::optional<Image> image;
            if (const Tensor* tensor = normal
                                           ? prim.GetDetailNormal()
                                           : prim.GetDetailAlbedo()) {
                image = ImageOfTensor(*tensor,
                                      normal ? ImageEncoding::Linear
                                             : ImageEncoding::Srgb);
            }
            if (normal) {
                gpu.SetDetailNormal(image ? &*image : nullptr);
            } else {
                gpu.SetDetailAlbedo(image ? &*image : nullptr);
            }
        } catch (const std::invalid_argument& error) {
            throw UploadError(path, attribute, error.what());
        }
    }
}

void RasterDelegate::SyncHeightfield(const Scene& scene,
                                     const Path& terrain) {
    ERNIE_ZONE("Sync heightfield");
    using namespace tokens;
    // `::ernie::Terrain`, not this class's `Terrain()`.
    using TerrainView = ::ernie::Terrain;
    const std::optional<TerrainView> prim =
        terrain.IsEmpty()
            ? std::nullopt
            : std::optional<TerrainView>(TerrainView(scene, terrain));
    const Tensor* written = prim ? prim->GetHeights() : nullptr;
    if (written == nullptr) {
        if (heightfield_) {
            terrain_.ClearHeightfield();
            planet_.ClearHeights();
            heightfield_.reset();
        }
        return;
    }
    const Key key =
        KeyOf(scene, terrain,
              {kHeights, kOrigin, kCellSize, kNoData, kPlanetRadius});
    if (heightfield_ == key) {
        return;
    }
    heightfield_ = key;
    if (prim->HasPlanetRadius()) {
        try {
            if (terrain_.HasHeightfield()) {
                terrain_.ClearHeightfield();
            }
            const Tensor heights = written->Contiguous();
            if (heights.Type() != DType::Float32 ||
                heights.Rank() != 3 ||
                heights.Shape()[0] != CubeSphereLod::kFaces ||
                heights.Location().kind != DeviceKind::Cpu) {
                throw std::invalid_argument(
                    "a planet's heights must be a float32 (6, n + 1, "
                    "n + 1) CPU tensor");
            }
            std::array<RasterView, CubeSphereLod::kFaces> faces;
            const auto rows =
                static_cast<std::size_t>(heights.Shape()[1]);
            const auto cols =
                static_cast<std::size_t>(heights.Shape()[2]);
            for (std::size_t face = 0; face < faces.size(); ++face) {
                faces[face].data =
                    heights.As<float>() + face * rows * cols;
                faces[face].width = cols;
                faces[face].height = rows;
                if (prim->HasNoData()) {
                    faces[face].noData =
                        static_cast<float>(prim->GetNoData());
                }
            }
            planet_.SetHeights(prim->GetPlanetRadius(), faces);
        } catch (const std::invalid_argument& error) {
            throw UploadError(terrain, kHeights, error.what());
        }
        return;
    }
    if (planet_.HasHeights()) {
        planet_.ClearHeights();
    }
    // Heights written on a GPU stay there: the terrain copies them
    // device to device and rebuilds what depends on the cells that
    // changed on the GPU.
    if (IsGpu(written->Location().kind)) {
        try {
            if (!terrainOnGpu_) {
                terrain_.SetComputeInterop(&RequireCompute());
                terrainOnGpu_ = true;
            }
            DeviceHeightfield field;
            field.heights = *written;
            const glm::dvec2 corner = prim->GetOrigin();
            field.originX = corner.x;
            field.originY = corner.y;
            field.cellSize = prim->GetCellSize();
            if (prim->HasNoData()) {
                field.noData = static_cast<float>(prim->GetNoData());
            }
            // Four numbers, which the registry has no type for.
            if (const Value* value =
                    scene.Get(terrain, kHeightsDirty)) {
                // A vec4, or four numbers.
                glm::vec4 dirty(-1.0f);
                if (const auto* vector = value->Get<glm::vec4>()) {
                    dirty = *vector;
                } else if (const auto* list =
                               value->Get<std::vector<float>>();
                           list != nullptr && list->size() == 4) {
                    dirty = {(*list)[0], (*list)[1], (*list)[2],
                             (*list)[3]};
                }
                if (glm::any(glm::lessThan(dirty, glm::vec4(0.0f)))) {
                    throw std::invalid_argument(
                        std::string(kHeightsDirty) +
                        " must be four non-negative numbers");
                }
                field.dirty = std::array<std::uint32_t, 4>{
                    static_cast<std::uint32_t>(dirty.x),
                    static_cast<std::uint32_t>(dirty.y),
                    static_cast<std::uint32_t>(dirty.z),
                    static_cast<std::uint32_t>(dirty.w)};
            }
            field.stream = scene.WriteStream(terrain, kHeights);
            terrain_.SetHeightfield(field);
        } catch (const std::invalid_argument& error) {
            throw UploadError(terrain, kHeights, error.what());
        }
        return;
    }
    try {
        const Tensor heights = written->Contiguous();
        terrain_.SetHeightfield(RasterOf(*prim, heights));
    } catch (const std::invalid_argument& error) {
        throw UploadError(terrain, kHeights, error.what());
    }
}

namespace {

/// Whether `camera` is a Camera exposed automatically.
bool ExposesAutomatically(const Scene& scene, const Path& camera) {
    return !camera.IsEmpty() &&
           scene.TypeOf(camera) == tokens::kCamera &&
           Camera(scene, camera).GetExposureMode() == tokens::kAuto;
}

}  // namespace

void RasterDelegate::ApplyAutoExposure(const Scene& scene,
                                       const Path& stateKey,
                                       FrameParams& frame) {
    // The linear value an average scene is keyed to, as the viewer's.
    constexpr double kMiddleGrey = 0.18;
    if (!ExposesAutomatically(scene, prims_.camera)) {
        autoStops_.erase(stateKey);
        return;
    }
    // Metered every frame under its key, for the frames after it;
    // batches of views meter none and keep their camera's exposure.
    const std::string key = stateKey.String();
    frame.meter = true;
    renderer_.SetMeterKey(key);
    const auto stops = autoStops_.find(stateKey);
    if (const std::optional<double> luminance =
            renderer_.MeteredLuminance(key)) {
        // The metered HdrColor is the scene's luminance in cd/m^2,
        // before any exposure.
        const double target = std::log2(
            kMiddleGrey /
            (*luminance * frame.camera.LinearExposureScale()));
        const double frames =
            Camera(scene, prims_.camera).GetExposureAdaptation();
        if (stops == autoStops_.end() || !(frames > 0.0)) {
            autoStops_[stateKey] = target;
        } else {
            stops->second += (target - stops->second) *
                             (1.0 - std::exp(-1.0 / frames));
        }
    }
    if (const auto found = autoStops_.find(stateKey);
        found != autoStops_.end()) {
        frame.exposure *=
            static_cast<float>(std::exp2(found->second));
    }
}

bool RasterDelegate::NeedsMetering(
    const Scene& scene, const RenderPrims& base,
    std::span<const RenderProductSpec> specs) const {
    for (const RenderProductSpec& spec : specs) {
        const Path& camera =
            spec.camera.IsEmpty() ? base.camera : spec.camera;
        if (ExposesAutomatically(scene, camera) &&
            !renderer_.MeteredLuminance(spec.product.String())) {
            return true;
        }
    }
    return false;
}

void RasterDelegate::ApplyDomeDirections(FrameParams& frame) const {
    if (!frame.dome || frame.domeMode != DomeLightMode::Directions ||
        !frame.dome->shadow || !domeLightingImage_) {
        return;
    }
    const std::size_t free =
        RasterRenderer::kMaxLights -
        std::min(frame.lights.size(), RasterRenderer::kMaxLights);
    const std::size_t count =
        std::min<std::size_t>(frame.domeDirections, free);
    if (count == 0) {
        return;
    }
    if (count != domeDirectionCount_) {
        domeDirections_ =
            DomeDirectionsOf(*domeLightingImage_, count);
        domeDirectionCount_ = count;
    }
    const glm::dmat3 rotation =
        glm::transpose(glm::dmat3(frame.dome->toDome));
    for (const DomeDirection& direction : domeDirections_) {
        // As the dome's light: its spectral radiance times the
        // region's colour, upsampled as the map is, times its RGB
        // irradiance.
        const glm::dvec3 rgb = direction.irradiance;
        const double peak = std::max({rgb.r, rgb.g, rgb.b});
        if (!(peak > 0.0)) {
            continue;
        }
        const std::vector<float> shape =
            RgbToSpectrum(rgb / peak).Bin(frame.camera.bands);
        LightParams light;
        light.type = LightType::Distant;
        light.direction =
            glm::vec3(glm::normalize(rotation * direction.direction));
        light.spectrum.assign(frame.dome->radiance.size(), 0.0f);
        for (std::size_t b = 0;
             b < light.spectrum.size() && b < shape.size(); ++b) {
            light.spectrum[b] = static_cast<float>(
                frame.dome->radiance[b] * shape[b] * peak);
        }
        light.shadows = true;
        light.domeDiffuse = true;
        frame.lights.push_back(std::move(light));
    }
    frame.domeDirectionsLit = true;
}

void RasterDelegate::ApplyDomeSun(FrameParams& frame) const {
    if (!domeSun_ || !frame.dome ||
        (frame.domeMode != DomeLightMode::Full &&
         frame.domeMode != DomeLightMode::Directions) ||
        !frame.dome->shadow || domeLightingMap_.Empty()) {
        return;
    }
    // The dome's frame turned into the world, as `DomeParamsOf`
    // turned its map.
    const glm::dmat3 rotation =
        glm::transpose(glm::dmat3(frame.dome->toDome));
    frame.dome->sh = DomeIrradianceSh(domeLightingMap_, rotation);
    // The sun's spectral irradiance: the dome's spectral radiance
    // times the sun's colour, upsampled as the dome's map is, times
    // its RGB irradiance.
    const glm::dvec3 rgb = domeSun_->irradiance;
    const double peak = std::max({rgb.r, rgb.g, rgb.b});
    const std::vector<float> shape =
        RgbToSpectrum(rgb / peak).Bin(frame.camera.bands);
    std::vector<float> spectrum(frame.dome->radiance.size(), 0.0f);
    for (std::size_t b = 0; b < spectrum.size() && b < shape.size();
         ++b) {
        spectrum[b] = static_cast<float>(frame.dome->radiance[b] *
                                         shape[b] * peak);
    }
    const glm::vec3 towards(
        glm::normalize(rotation * domeSun_->direction));
    if (!frame.sun) {
        frame.sun = true;
        frame.sunDirection = towards;
        frame.sunAngle = static_cast<float>(domeSun_->angle);
        frame.sunIrradiance = std::move(spectrum);
        frame.shadows.enabled = true;
        return;
    }
    LightParams sun;
    sun.type = LightType::Distant;
    sun.direction = towards;
    sun.spectrum = std::move(spectrum);
    sun.shadows = true;
    frame.lights.push_back(std::move(sun));
}

void RasterDelegate::SyncSky(const Scene& scene,
                             const RenderPrims& prims) {
    ERNIE_ZONE("Sync sky");
    using namespace tokens;
    const std::optional<Key> map =
        prims.dome.IsEmpty() ? std::nullopt
                             : std::optional(KeyOf(scene, prims.dome,
                                                   {kLightTexture}));
    // The full and directions dome lighting modes, with the dome's
    // shadows on, light its map's sun on its own.
    const std::string mode =
        RenderSettings::OrFallbacks(scene, prims.settings)
            .GetDomeLightMode();
    const bool wantSun =
        !prims.dome.IsEmpty() &&
        (mode == "full" || mode == "directions") &&
        GetBool(scene, prims.dome, kShadowEnable, true);
    if (map != domeMap_ || wantSun != domeSunWanted_) {
        domeMap_ = map;
        domeSunWanted_ = wantSun;
        domeSun_.reset();
        domeLightingMap_ = Tensor();
        domeLightingImage_.reset();
        domeDirections_.clear();
        domeDirectionCount_ = 0;
        try {
            const std::optional<Image> image =
                prims.dome.IsEmpty() ? std::nullopt
                                     : DomeMapOf(scene, prims.dome);
            const std::optional<DomeSunExtraction> extraction =
                wantSun && image ? ExtractDomeSun(*image)
                                 : std::nullopt;
            if (wantSun) {
                // A uniform map stands for a dome without one.
                domeLightingImage_ =
                    extraction ? extraction->lighting
                    : image    ? *image
                               : Image(32, 16, glm::vec4(1.0f));
            }
            if (extraction) {
                domeSun_ = extraction->sun;
                const Image& lit = extraction->lighting;
                const auto* values =
                    reinterpret_cast<const float*>(lit.pixels.data());
                domeLightingMap_ = Tensor::Copy<float>(
                    std::span(values, lit.pixels.size() * 4),
                    {static_cast<std::int64_t>(lit.height),
                     static_cast<std::int64_t>(lit.width), 4});
            }
            renderer_.SetDomeMap(image ? &*image : nullptr,
                                 extraction ? &extraction->lighting
                                            : nullptr);
        } catch (const std::invalid_argument& error) {
            renderer_.SetDomeMap(nullptr);
            throw UploadError(prims.dome, kLightTexture,
                              error.what());
        }
    }
    const std::optional<Key> stars =
        prims.stars.IsEmpty()
            ? std::nullopt
            : std::optional(KeyOf(
                  scene, prims.stars,
                  {kStarsRightAscension, kStarsDeclination,
                   kStarsDirection, kStarsMagnitude, kStarsColorIndex,
                   kStarsTemperature, kStarsMagnitudeLimit,
                   kStarsZeroPoint, kExposure}));
    if (stars != stars_) {
        stars_ = stars;
        try {
            renderer_.SetStars(prims.stars.IsEmpty()
                                   ? std::vector<Star>()
                                   : StarsOf(scene, prims.stars));
        } catch (const std::invalid_argument& error) {
            renderer_.SetStars({});
            throw UploadError(prims.stars, kStarsMagnitude,
                              error.what());
        }
    }
}

ComputeInterop* RasterDelegate::Compute(std::string* why) {
    if (!computeTried_) {
        computeTried_ = true;
        compute_ = ComputeInterop::Create(context_, &computeWhy_);
    }
    if (compute_ == nullptr && why != nullptr) {
        *why = computeWhy_;
    }
    return compute_.get();
}

ComputeInterop& RasterDelegate::RequireCompute() {
    std::string why;
    ComputeInterop* compute = Compute(&why);
    if (compute == nullptr) {
        throw std::invalid_argument(
            "GPU arrays need compute interop, which is "
            "unavailable: " +
            why);
    }
    if (!meshesOnGpu_) {
        meshes_.SetComputeInterop(compute);
        points_.SetComputeInterop(compute);
        meshesOnGpu_ = true;
    }
    return *compute;
}

void RasterDelegate::SetMeshPoints(const Scene& scene,
                                   const Path& path,
                                   std::uint32_t handle) {
    const Tensor* tensor = scene.Read(path, tokens::kPoints);
    if (tensor != nullptr && IsGpu(tensor->Location().kind)) {
        RequireCompute();
        meshes_.SetPoints(handle, *tensor, ExtentOf(scene, path),
                          scene.WriteStream(path, tokens::kPoints));
        return;
    }
    const std::optional<std::vector<float>> points =
        FloatAttribute(scene, path, tokens::kPoints, 3);
    meshes_.SetPoints(handle, points.value_or(std::vector<float>()));
}

void RasterDelegate::SetMeshAttributes(const Scene& scene,
                                       const Path& path,
                                       std::uint32_t handle) {
    const Tensor* normals = scene.Read(path, tokens::kNormals);
    if (normals != nullptr && IsGpu(normals->Location().kind)) {
        RequireCompute();
        meshes_.SetNormals(handle, *normals,
                           scene.WriteStream(path, tokens::kNormals));
    }
    const Tensor* colors = Gprim(scene, path).ReadDisplayColor();
    if (colors != nullptr && IsGpu(colors->Location().kind)) {
        RequireCompute();
        meshes_.SetColors(
            handle, *colors,
            scene.WriteStream(path, tokens::kDisplayColor));
    }
}

void RasterDelegate::SyncMeshes(const Scene& scene) {
    ERNIE_ZONE("Sync meshes");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    std::vector<Path> paths;
    for (const std::string_view type :
         {kMesh, kCube, kSphere, kCylinder, kCone, kCapsule,
          kPlane}) {
        const std::vector<Path> ofType = scene.Prims(type);
        paths.insert(paths.end(), ofType.begin(), ofType.end());
    }
    for (const Path& path : paths) {
        seen.insert(path);
        Key key =
            KeyOf(scene, path,
                  {kFaceVertexCounts, kFaceVertexIndices, kNormals,
                   kDisplayColor, kSize, kGeomRadius, kGeomHeight,
                   kGeomWidth, kGeomLength, kAxis});
        // Normals and colours written on the GPU update in place like
        // points, rather than adding the mesh again (key slots 3 and
        // 4).
        const auto onGpu = [&](std::string_view attribute) {
            const Tensor* tensor = scene.Read(path, attribute);
            return tensor != nullptr &&
                   IsGpu(tensor->Location().kind);
        };
        const bool gpuNormals = onGpu(kNormals);
        const bool gpuColors = onGpu(kDisplayColor);
        key.second[3] = gpuNormals ? 1 : key.second[3];
        key.second[4] = gpuColors ? 1 : key.second[4];
        const std::array<std::uint64_t, 3> versions{
            scene.AttributeVersion(path, kPoints),
            gpuNormals ? scene.AttributeVersion(path, kNormals) : 0,
            gpuColors ? scene.AttributeVersion(path, kDisplayColor)
                      : 0};
        const auto it = syncedMeshes_.find(path);
        if (it != syncedMeshes_.end() && it->second.key == key) {
            SyncedMesh& synced = it->second;
            if (synced.versions == versions) {
                continue;
            }
            if (synced.handle != 0) {
                try {
                    if (synced.versions[0] != versions[0]) {
                        SetMeshPoints(scene, path, synced.handle);
                    }
                    if (synced.versions[1] != versions[1] ||
                        synced.versions[2] != versions[2]) {
                        SetMeshAttributes(scene, path, synced.handle);
                    }
                    synced.versions = versions;
                    continue;
                } catch (const std::invalid_argument&) {
                }
            }
        }
        SyncedMesh synced;
        synced.key = key;
        synced.versions = versions;
        if (it != syncedMeshes_.end() && it->second.handle != 0) {
            meshes_.RemoveMesh(it->second.handle);
        }
        try {
            MeshData data = MeshDataOf(scene, path);
            synced.hasColors = !data.colors.empty();
            synced.tint = ConstantDisplayColor(Gprim(scene, path));
            synced.handle = meshes_.AddMesh(std::move(data));
            if (onGpu(kPoints)) {
                SetMeshPoints(scene, path, synced.handle);
            }
            if (gpuNormals || gpuColors) {
                SetMeshAttributes(scene, path, synced.handle);
            }
        } catch (const std::invalid_argument& error) {
            if (synced.handle != 0) {
                meshes_.RemoveMesh(synced.handle);
                synced.handle = 0;
            }
            errors.push_back(Describe(path, "mesh", error));
        }
        syncedMeshes_[path] = synced;
    }
    for (auto it = syncedMeshes_.begin();
         it != syncedMeshes_.end();) {
        if (seen.contains(it->first)) {
            ++it;
            continue;
        }
        if (it->second.handle != 0) {
            meshes_.RemoveMesh(it->second.handle);
        }
        it = syncedMeshes_.erase(it);
    }
    if (!errors.empty()) {
        throw std::invalid_argument("Cannot upload " +
                                    errors.front());
    }
}

void RasterDelegate::SyncInstancers(const Scene& scene) {
    ERNIE_ZONE("Sync instancers");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    for (const Path& path : scene.Prims(kPointInstancer)) {
        seen.insert(path);
        const Key key =
            KeyOf(scene, path, {kPositions, kOrientations, kScales});
        const std::optional<GpuAnchor> anchor =
            GpuAnchorOf(scene, path);
        auto [it, added] = instancers_.try_emplace(path);
        SyncedInstancer& synced = it->second;
        const std::uint32_t anchorInstancer =
            anchor ? anchor->instancer : 0;
        const std::uint32_t anchorRow = anchor ? anchor->row : 0;
        const glm::dmat4 anchorBelow =
            anchor ? anchor->below : glm::dmat4(1.0);
        if (!added && synced.key == key &&
            synced.anchorInstancer == anchorInstancer &&
            synced.anchorRow == anchorRow &&
            synced.anchorBelow == anchorBelow) {
            continue;
        }
        synced.key = key;
        synced.anchorInstancer = anchorInstancer;
        synced.anchorRow = anchorRow;
        synced.anchorBelow = anchorBelow;
        try {
            const PointInstancer instancer(scene, path);
            const Tensor* positions = instancer.GetPositions();
            if (anchor && positions != nullptr) {
                // Under a GPU-posed prim: its instances composed on
                // the GPU under that prim's instance, then the
                // transforms from it down to the instancer.
                ComputeInterop& compute = RequireCompute();
                const std::optional<Bounds3> extent =
                    XformExtentOf(scene, anchor->at);
                if (!extent) {
                    throw std::invalid_argument(
                        "its instances are posed under GPU-posed '" +
                        anchor->at.String() + "', which needs '" +
                        std::string(kXformExtent) + "'");
                }
                ComposedInstanceData data;
                const auto op = [&](std::string_view attribute,
                                    InstanceOpKind kind,
                                    std::int64_t width) {
                    const Tensor* tensor =
                        scene.Read(path, attribute);
                    if (tensor != nullptr &&
                        IsGpu(tensor->Location().kind)) {
                        data.ops.push_back({kind, *tensor});
                    } else if (const std::optional<std::vector<float>>
                                   values = FloatAttribute(
                                       scene, path, attribute,
                                       width)) {
                        const auto rows = static_cast<std::int64_t>(
                                              values->size()) /
                                          width;
                        data.ops.push_back(
                            {kind, compute.Upload(Tensor::Copy<float>(
                                       *values, {rows, width}))});
                    }
                };
                op(kPositions, InstanceOpKind::Translate, 3);
                op(kOrientations, InstanceOpKind::Orient, 4);
                op(kScales, InstanceOpKind::Scale, 3);
                InstanceParent parent;
                parent.instancer = anchor->instancer;
                parent.row = anchor->row;
                SetBetween(parent, anchor->below);
                data.parents.assign(
                    static_cast<std::size_t>(
                        data.ops.front().rows.Shape()[0]),
                    parent);
                data.extent = *extent;
                data.stream = scene.WriteStream(path, kPositions);
                if (synced.handle == 0) {
                    synced.handle = meshes_.AddInstancer({});
                }
                meshes_.SetInstances(synced.handle, data);
            } else if (positions != nullptr &&
                       IsGpu(positions->Location().kind)) {
                ComputeInterop& compute = RequireCompute();
                const auto onDevice = [&](const Tensor* tensor) {
                    if (tensor == nullptr) {
                        return Tensor();
                    }
                    return IsGpu(tensor->Location().kind)
                               ? *tensor
                               : compute.Upload(*tensor);
                };
                DeviceInstanceData data;
                data.positions = *positions;
                data.orientations =
                    onDevice(instancer.GetOrientations());
                data.scales = onDevice(instancer.GetScales());
                data.extent = ExtentOf(scene, path);
                data.stream = scene.WriteStream(path, kPositions);
                if (synced.handle == 0) {
                    synced.handle = meshes_.AddInstancer({});
                }
                meshes_.SetInstances(synced.handle, data);
            } else {
                InstanceData data = InstanceDataOf(scene, path);
                if (synced.handle == 0) {
                    synced.handle =
                        meshes_.AddInstancer(std::move(data));
                } else {
                    meshes_.SetInstances(synced.handle,
                                         std::move(data));
                }
            }
        } catch (const std::invalid_argument& error) {
            if (synced.handle != 0) {
                meshes_.RemoveInstancer(synced.handle);
                synced.handle = 0;
            }
            errors.push_back(Describe(path, "instancer", error));
        }
    }
    for (auto it = instancers_.begin(); it != instancers_.end();) {
        if (seen.contains(it->first)) {
            ++it;
            continue;
        }
        if (it->second.handle != 0) {
            meshes_.RemoveInstancer(it->second.handle);
        }
        it = instancers_.erase(it);
    }
    if (!errors.empty()) {
        throw std::invalid_argument("Cannot upload " +
                                    errors.front());
    }
}

void RasterDelegate::SyncGpuXforms(const Scene& scene) {
    ERNIE_ZONE("Sync GPU transforms");
    using namespace tokens;
    struct Member {
        Path path;
        GpuXform pose;
    };
    // Prims sharing arrays (one `Scene::Write` of many paths), then
    // each group by its first prim.
    std::map<std::vector<const void*>, std::vector<Member>> byArrays;
    std::vector<std::string> errors;
    for (const Path& path : scene.Prims()) {
        try {
            std::optional<GpuXform> pose = GpuXformOf(scene, path);
            if (!pose || pose->ops.empty()) {
                continue;
            }
            std::vector<const void*> arrays;
            for (const auto& [op, whole] : pose->ops) {
                arrays.push_back(whole.Data());
            }
            byArrays[arrays].push_back({path, std::move(*pose)});
        } catch (const std::invalid_argument& error) {
            errors.emplace_back(error.what());
        }
    }
    std::map<GpuXformKey, std::vector<Member>> groups;
    for (auto& [arrays, members] : byArrays) {
        const auto first =
            std::min_element(members.begin(), members.end(),
                             [](const Member& a, const Member& b) {
                                 return a.pose.row < b.pose.row;
                             });
        const std::int64_t rows =
            first->pose.ops.front().second.Shape()[0];
        groups[{first->path, rows}] = std::move(members);
    }

    // Each GPU-posed prim's group and row, then the GPU-posed prim it
    // is posed under, if any: the nearest above it whose transform
    // reaches it.
    std::unordered_map<Path, std::pair<GpuXformGroup*, std::uint32_t>>
        posedAt;
    for (const auto& [key, members] : groups) {
        GpuXformGroup& group = gpuXformGroups_[key];
        if (group.handle == 0) {
            group.handle = meshes_.AddInstancer({});
        }
        for (const Member& member : members) {
            posedAt[member.path] = {
                &group, static_cast<std::uint32_t>(member.pose.row)};
        }
    }
    const auto parentOf =
        [&](const Member& member) -> std::optional<Path> {
        if (member.pose.reset) {
            return std::nullopt;
        }
        for (Path at = member.path.Parent();
             !at.IsEmpty() && !at.IsRoot(); at = at.Parent()) {
            if (posedAt.contains(at)) {
                return at;
            }
            if (scene.Has(at) && ResetsXformStack(scene, at)) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    };
    // The prim whose world transform a GPU-posed prim's draws are
    // under: its outermost GPU-posed ancestor's parent, or none for
    // the world.
    std::unordered_map<Path, std::optional<Path>> frames;
    std::unordered_map<Path, const Member*> byPath;
    for (const auto& [key, members] : groups) {
        for (const Member& member : members) {
            byPath[member.path] = &member;
        }
    }
    const std::function<std::optional<Path>(const Member&)> frameOf =
        [&](const Member& member) -> std::optional<Path> {
        if (const auto known = frames.find(member.path);
            known != frames.end()) {
            return known->second;
        }
        std::optional<Path> frame;
        if (const std::optional<Path> parent = parentOf(member)) {
            frame = frameOf(*byPath.at(*parent));
        } else if (!member.pose.reset &&
                   !member.path.Parent().IsRoot()) {
            frame = member.path.Parent();
        }
        frames[member.path] = frame;
        return frame;
    };

    for (auto it = gpuXformGroups_.begin();
         it != gpuXformGroups_.end();) {
        if (groups.contains(it->first)) {
            ++it;
            continue;
        }
        if (it->second.handle != 0) {
            meshes_.RemoveInstancer(it->second.handle);
        }
        it = gpuXformGroups_.erase(it);
    }

    gpuPosed_.clear();
    for (const auto& [key, members] : groups) {
        try {
            GpuXformGroup& group = gpuXformGroups_[key];
            const GpuXform& first = members.front().pose;
            // The group's bounds when every prim gives its own
            // (`ernie:xform:extent`, in the frame of its draws);
            // otherwise the drawable copies the arrays to the host to
            // bound them, which poses composed on the GPU cannot be.
            std::uint64_t version = members.size();
            std::optional<Bounds3> extent;
            bool bounded = true;
            bool composed = false;
            std::vector<InstanceParent> parents(
                static_cast<std::size_t>(key.second));
            for (const Member& member : members) {
                if (member.pose.wFirst != first.wFirst) {
                    throw std::invalid_argument(
                        "its prims, written as rows of the same "
                        "arrays, must share one '" +
                        std::string(kQuaternionOrder) + "'");
                }
                for (std::size_t k = 0; k < member.pose.ops.size();
                     ++k) {
                    if (k >= first.ops.size() ||
                        member.pose.ops[k].first !=
                            first.ops[k].first) {
                        throw std::invalid_argument(
                            "its prims, written as rows of the same "
                            "arrays, must share one "
                            "xformOpOrder");
                    }
                    version += scene.AttributeVersion(
                        member.path, member.pose.ops[k].first);
                }
                for (const std::string_view attribute :
                     {kXformOpOrder, kXformExtent,
                      kQuaternionOrder}) {
                    version += scene.AttributeVersion(member.path,
                                                      attribute);
                }
                if (const std::optional<Bounds3> box =
                        XformExtentOf(scene, member.path)) {
                    extent = extent ? Bounds3(glm::min(extent->first,
                                                       box->first),
                                              glm::max(extent->second,
                                                       box->second))
                                    : *box;
                } else {
                    bounded = false;
                }
                composed = composed || member.pose.composed;
                // Posed under another GPU-posed prim: that one's
                // instance, then the transforms of the prims between,
                // from its child down to this prim's parent.
                if (const std::optional<Path> above =
                        parentOf(member)) {
                    composed = true;
                    InstanceParent& parent =
                        parents[static_cast<std::size_t>(
                            member.pose.row)];
                    const auto [aboveGroup, aboveRow] =
                        posedAt.at(*above);
                    parent.instancer = aboveGroup->handle;
                    parent.row = aboveRow;
                    glm::dmat4 between(1.0);
                    for (Path at = member.path.Parent(); at != *above;
                         at = at.Parent()) {
                        between = scene.LocalTransform(at) * between;
                        version += scene.OwnTransformVersion(at);
                    }
                    SetBetween(parent, between);
                }
            }
            if (composed && !bounded) {
                throw std::invalid_argument(
                    "prims whose poses are composed on the GPU "
                    "(Euler rotations, matrices, scales "
                    "of one number, ops out of translate, orient, "
                    "scale order, or GPU-posed prims "
                    "under others) need '" +
                    std::string(kXformExtent) + "' on each");
            }
            group.members = members.size();
            group.rows = key.second;
            group.extent = bounded ? extent : std::nullopt;
            group.wFirst = first.wFirst;
            group.composed = composed;
            group.parents = composed ? std::move(parents)
                                     : std::vector<InstanceParent>{};
            group.ops.clear();
            for (const auto& [op, whole] : first.ops) {
                group.ops.push_back(op);
            }
            // Versions count within one scene: another scene's prims
            // at the same paths upload.
            if (version != group.version || group.version == 0 ||
                group.scene != scene.Id()) {
                UploadGpuXformGroup(group, first, key.second);
                group.version = version;
                group.scene = scene.Id();
            }
            for (const Member& member : members) {
                gpuPosed_[member.path] = {
                    &group, group.handle,
                    static_cast<std::uint32_t>(member.pose.row),
                    member.pose.reset, frameOf(member)};
            }
        } catch (const std::invalid_argument& error) {
            errors.push_back(Describe(members.front().path,
                                      "GPU transforms", error));
        }
    }
    if (!errors.empty()) {
        throw std::invalid_argument("Cannot upload " +
                                    errors.front());
    }
}

void RasterDelegate::UploadGpuXformGroup(GpuXformGroup& group,
                                         const GpuXform& pose,
                                         std::int64_t rows) {
    ComputeInterop& compute = RequireCompute();
    if (group.composed) {
        ComposedInstanceData data;
        for (const auto& [op, whole] : pose.ops) {
            const InstanceOpKind kind = InstanceOpKindOf(op);
            data.ops.push_back(
                {kind, whole,
                 kind == InstanceOpKind::Orient && pose.wFirst});
        }
        data.parents = group.parents;
        data.extent = *group.extent;
        data.stream = pose.stream;
        meshes_.SetInstances(group.handle, data);
        return;
    }
    DeviceInstanceData data;
    if (pose.translate.Empty()) {
        if (group.zeros.Empty() || group.zeros.Shape()[0] != rows) {
            group.zeros = compute.Upload(
                Tensor::Zeros(DType::Float32, {rows, 3}));
        }
        data.positions = group.zeros;
    } else {
        data.positions = pose.translate;
    }
    data.orientations = pose.orient;
    data.scales = pose.scale;
    data.extent = group.extent;
    data.stream = pose.stream;
    data.orientationsWFirst = pose.wFirst;
    if (group.handle == 0) {
        group.handle = meshes_.AddInstancer({});
    }
    meshes_.SetInstances(group.handle, data);
}

std::optional<RasterDelegate::GpuAnchor>
RasterDelegate::GpuAnchorOf(const Scene& scene,
                            const Path& path) const {
    if (gpuPosed_.empty()) {
        return std::nullopt;
    }
    for (Path at = path; !at.IsEmpty() && !at.IsRoot();
         at = at.Parent()) {
        if (const auto posed = gpuPosed_.find(at);
            posed != gpuPosed_.end()) {
            GpuAnchor anchor;
            anchor.at = at;
            anchor.instancer = posed->second.instancer;
            anchor.row = posed->second.row;
            if (posed->second.frame) {
                anchor.frame =
                    scene.WorldTransform(*posed->second.frame);
            }
            for (Path below = path; below != at;
                 below = below.Parent()) {
                anchor.below =
                    scene.LocalTransform(below) * anchor.below;
            }
            return anchor;
        }
        if (scene.Has(at) && ResetsXformStack(scene, at)) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

HostArrayReader RasterDelegate::HostArrays(const Scene& scene) const {
    return [this, &scene](const Path& path,
                          std::string_view attribute,
                          const Tensor& device) {
        HostArray& kept = hostArrays_[{path, std::string(attribute)}];
        const std::uint64_t version =
            scene.AttributeVersion(path, attribute);
        if (kept.host.Empty() || kept.scene != scene.Id() ||
            kept.version != version) {
            if (compute_ == nullptr) {
                throw std::invalid_argument(
                    "'" + std::string(attribute) + "' of '" +
                    path.String() +
                    "' on a GPU needs compute interop");
            }
            kept.host = compute_->Download(
                device, scene.WriteStream(path, attribute));
            kept.scene = scene.Id();
            kept.version = version;
        }
        return kept.host;
    };
}

std::vector<AnchoredView>
RasterDelegate::AnchoredViewsOf(const Scene& scene,
                                std::span<const Path> cameras) const {
    std::vector<std::optional<GpuAnchor>> anchors;
    for (const Path& camera : cameras) {
        anchors.push_back(GpuAnchorOf(scene, camera));
    }
    if (std::none_of(anchors.begin(), anchors.end(),
                     [](const std::optional<GpuAnchor>& anchor) {
                         return anchor.has_value();
                     })) {
        return {};
    }
    std::vector<AnchoredView> views;
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const Path& camera = cameras[i];
        if (scene.Read(camera, tokens::kPoses) != nullptr) {
            throw std::invalid_argument(
                "Camera '" + camera.String() +
                "' renders with cameras under GPU-posed prims, so it "
                "takes no '" +
                std::string(tokens::kPoses) + "'");
        }
        AnchoredView view;
        if (!anchors[i]) {
            view.frame = scene.WorldTransform(camera);
        } else {
            // Its frame's world transform, its anchor's pose, then
            // the camera's transforms below.
            view.instancer = anchors[i]->instancer;
            view.row = anchors[i]->row;
            view.frame = anchors[i]->frame;
            view.below = anchors[i]->below;
        }
        views.push_back(view);
    }
    return views;
}

bool RasterDelegate::RefreshGpuXforms(
    const Scene& scene, std::span<const Change> changes) {
    ERNIE_ZONE("Refresh GPU transforms");
    // Thousands of prims change every step: each costs a few lookups,
    // and only one prim of each group is checked in full.
    struct Touched {
        const Path* path =
            nullptr;  ///< A prim of the group the changes rewrote.
        std::array<const void*, 3> arrays{};
        std::size_t rewritten = 0;
    };
    std::map<GpuXformGroup*, Touched> touched;
    const auto posedAncestor = [this](const Path& path) {
        for (Path at = path.Parent(); !at.IsEmpty() && !at.IsRoot();
             at = at.Parent()) {
            if (gpuPosed_.contains(at)) {
                return true;
            }
        }
        return false;
    };
    for (const Change& change : changes) {
        const auto posed = gpuPosed_.find(change.path);
        if (posed != gpuPosed_.end()) {
            // A prim posed on the GPU, its rows rewritten under the
            // same order.
            GpuXformGroup& group = *posed->second.group;
            if (change.bits != Dirty::Transform ||
                scene.AttributeVersion(change.path,
                                       tokens::kXformOpOrder) >
                    syncedVersion_) {
                return false;
            }
            Touched& seen = touched[&group];
            for (std::size_t op = 0; op < group.ops.size(); ++op) {
                const std::pair<Tensor, std::int64_t>* row =
                    scene.RowOf(change.path, group.ops[op]);
                if (row == nullptr ||
                    row->second != posed->second.row) {
                    return false;
                }
                if (seen.path == nullptr) {
                    seen.arrays[op] = row->first.Data();
                } else if (seen.arrays[op] != row->first.Data()) {
                    return false;  // the group split across arrays
                }
            }
            if (seen.path == nullptr) {
                seen.path = &change.path;
            }
            ++seen.rewritten;
        } else if (change.bits == Dirty::Transform &&
                   posedAncestor(change.path) &&
                   scene.OwnTransformVersion(change.path) <=
                       syncedVersion_) {
            // Below a prim posed on the GPU, moved only with it.
        } else if (!IsViewChange(scene, change)) {
            return false;
        }
    }
    // Every prim of each touched group rewritten, from arrays of the
    // group's rows (checked in full for one prim): its draws stand as
    // they are and only its arrays change.
    std::vector<std::pair<GpuXformGroup*, GpuXform>> uploads;
    try {
        for (const auto& [group, seen] : touched) {
            std::optional<GpuXform> pose =
                GpuXformOf(scene, *seen.path);
            if (seen.rewritten != group->members || !pose ||
                pose->wFirst != group->wFirst || pose->ops.empty()) {
                return false;
            }
            // Every op's array has the group's rows (`GpuXformOf`);
            // composed poses fill only ops.
            if (pose->ops.front().second.Shape()[0] != group->rows) {
                return false;
            }
            uploads.emplace_back(group, std::move(*pose));
        }
    } catch (const std::invalid_argument&) {
        return false;  // the full sync reports it
    }
    for (auto& [group, pose] : uploads) {
        UploadGpuXformGroup(*group, pose, group->rows);
    }
    return true;
}

Tensor RasterDelegate::MapArray(Scene& scene, const Path& path,
                                std::string_view attribute,
                                std::uintptr_t stream) {
    using namespace tokens;
    RequireCompute();
    Sync(scene);
    if (!scene.Has(path)) {
        throw std::invalid_argument(
            "Cannot map " + std::string(attribute) + " of '" +
            path.String() + "', which does not exist");
    }
    if (attribute.starts_with("xformOp:") &&
        gpuPosed_.contains(path)) {
        return MapArray(scene, PrimSet({path}), attribute, stream);
    }
    const std::string_view type = scene.TypeOf(path);
    const auto unsynced = [&] {
        return std::invalid_argument(
            "Cannot map " + std::string(attribute) + " of '" +
            path.String() + "': it has not been uploaded");
    };
    Tensor view;
    if (type == kMesh && attribute == kPoints) {
        const auto it = syncedMeshes_.find(path);
        if (it == syncedMeshes_.end() || it->second.handle == 0) {
            throw unsynced();
        }
        view = meshes_.MapPoints(it->second.handle, stream);
    } else if (type == kPointInstancer &&
               (attribute == kPositions ||
                attribute == kOrientations || attribute == kScales)) {
        const auto it = instancers_.find(path);
        if (it == instancers_.end() || it->second.handle == 0) {
            throw unsynced();
        }
        const std::size_t array = attribute == kPositions      ? 0
                                  : attribute == kOrientations ? 1
                                                               : 2;
        view = meshes_.MapInstances(it->second.handle, array, stream);
    } else if (type == kGeomPoints && attribute == kPoints) {
        const auto it = syncedPoints_.find(path);
        if (it == syncedPoints_.end() || it->second.handle == 0) {
            throw unsynced();
        }
        view = points_.MapPositions(it->second.handle, stream);
    } else if (type == kTerrain && attribute == kHeights &&
               path == ResolveRenderPrims(scene, {}).terrain) {
        view = terrain_.MapHeights(stream);
    } else if (type == kCamera && attribute == kPoses) {
        // The pose buffer the views read, shaped as the poses
        // written; a frame finds them there and copies nothing
        // (`StepAsync`).
        const Tensor* written = GpuPosesOf(scene, path);
        if (written == nullptr) {
            throw std::invalid_argument(
                "Cannot map the poses of '" + path.String() +
                "': write them first as a contiguous float32 GPU "
                "array");
        }
        if (PosesAreQuaternions(*written) &&
            QuaternionsWFirst(scene, path)) {
            throw std::invalid_argument("Cannot map the poses of '" +
                                        path.String() +
                                        "' with quaternions w first");
        }
        ComputeInterop& compute = RequireCompute();
        const auto views =
            static_cast<std::uint32_t>(written->Shape()[0]);
        view = compute.View(renderer_.EnsurePoseBuffer(views),
                            DType::Float32, written->Shape());
        // Nothing writes it until the frames already submitted have
        // read it.
        compute.WaitForVulkan(stream);
        if (view.Data() != written->Data()) {
            compute.Copy(*written, view, stream);
        }
    } else {
        throw std::invalid_argument(
            "Cannot map " + std::string(attribute) + " of '" +
            path.String() +
            "': only a Mesh's points, a PointInstancer's "
            "arrays, a Points prim's points, the Terrain's heights "
            "and a "
            "Camera's poses map");
    }
    scene.Write(path, attribute, view, stream);
    return view;
}

Tensor RasterDelegate::MapArray(Scene& scene, const PrimSet& prims,
                                std::string_view attribute,
                                std::uintptr_t stream) {
    RequireCompute();
    Sync(scene);
    const std::vector<Path>& paths = prims.Paths();
    const auto refuse = [&](const std::string& why) {
        return std::invalid_argument(
            "Cannot map " + std::string(attribute) + " of " +
            (paths.empty() ? "no prims"
                           : "'" + paths.front().String() + "'...") +
            ": " + why);
    };
    if (paths.empty()) {
        throw refuse("give the prims of a group posed on the GPU");
    }
    const auto first = gpuPosed_.find(paths.front());
    if (first == gpuPosed_.end()) {
        throw refuse("its transform ops are not rows of a GPU array "
                     "posing it on the GPU");
    }
    GpuXformGroup& group = *first->second.group;
    // Exactly the group's prims, prim i at row i, so the tensor
    // written back is the whole array.
    if (paths.size() != group.members ||
        static_cast<std::int64_t>(paths.size()) != group.rows) {
        throw refuse(
            "give every prim the array's rows pose, in row order");
    }
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const auto posed = gpuPosed_.find(paths[i]);
        if (posed == gpuPosed_.end() ||
            posed->second.group != &group || posed->second.row != i) {
            throw refuse("'" + paths[i].String() + "' is not row " +
                         std::to_string(i) + " of the same array");
        }
    }
    const auto op =
        std::find(group.ops.begin(), group.ops.end(), attribute);
    if (op == group.ops.end()) {
        throw refuse("the prims have no such transform op");
    }
    Tensor view;
    if (group.composed) {
        view = meshes_.MapComposedOp(
            group.handle,
            static_cast<std::size_t>(op - group.ops.begin()), stream);
    } else {
        const InstanceOpKind kind = InstanceOpKindOf(*op);
        if (kind == InstanceOpKind::Orient && group.wFirst) {
            throw refuse("orientations map with w last");
        }
        const std::size_t array =
            kind == InstanceOpKind::Translate ? 0
            : kind == InstanceOpKind::Orient  ? 1
                                              : 2;
        view = meshes_.MapInstances(group.handle, array, stream);
    }
    scene.Write(prims, attribute, view, stream);
    return view;
}

void RasterDelegate::SyncPoints(const Scene& scene) {
    ERNIE_ZONE("Sync points");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    for (const Path& path : scene.Prims(kGeomPoints)) {
        seen.insert(path);
        const Key key = KeyOf(
            scene, path, {kPoints, kWidths, kDisplayColor, kExtent});
        auto [it, added] = syncedPoints_.try_emplace(path);
        SyncedPoints& synced = it->second;
        if (!added && synced.key == key) {
            continue;
        }
        synced.key = key;
        try {
            const ernie::Points cloud(scene, path);
            // A width for all from one value (a number, or an array
            // of one written or not), else one per point: a float
            // array, or a written array (`tensor`, `ReadWidths`).
            const auto widthsOf =
                [&](float& width,
                    std::vector<float>& perPoint) -> const Tensor* {
                std::vector<float> widths = cloud.GetWidths();
                width = widths.size() == 1 ? widths.front() : 1.0f;
                if (widths.size() > 1) {
                    perPoint = std::move(widths);
                }
                const Tensor* tensor = cloud.ReadWidths();
                if (tensor != nullptr && tensor->Elements() == 1 &&
                    tensor->Location().kind == DeviceKind::Cpu) {
                    width = FloatsOf(*tensor, 1, kWidths).front();
                    return nullptr;
                }
                return tensor;
            };
            const Tensor* points = cloud.GetPoints();
            if (points != nullptr && IsGpu(points->Location().kind)) {
                ComputeInterop& compute = RequireCompute();
                const auto onDevice = [&](const Tensor* tensor) {
                    return tensor == nullptr ? Tensor()
                           : IsGpu(tensor->Location().kind)
                               ? *tensor
                               : compute.Upload(*tensor);
                };
                DevicePointsData data;
                data.points = *points;
                std::vector<float> perPoint;
                const Tensor* widths = widthsOf(data.width, perPoint);
                data.widths =
                    widths == nullptr && !perPoint.empty()
                        ? compute.Upload(Tensor::Copy<float>(
                              perPoint, {static_cast<std::int64_t>(
                                            perPoint.size())}))
                        : onDevice(widths);
                const Tensor* colors = cloud.ReadDisplayColor();
                if (colors != nullptr &&
                    colors->Elements() == points->Elements()) {
                    data.colors = onDevice(colors);
                }
                data.extent = ExtentOf(scene, path);
                data.stream = scene.WriteStream(path, kPoints);
                if (synced.handle == 0) {
                    synced.handle = points_.Add({});
                }
                points_.Set(synced.handle, data);
            } else {
                PointsData data;
                data.points = FloatAttribute(scene, path, kPoints, 3)
                                  .value_or(std::vector<float>());
                if (const Tensor* widths =
                        widthsOf(data.width, data.widths)) {
                    data.widths = FloatsOf(*widths, 1, kWidths);
                }
                if (const Tensor* colors = cloud.ReadDisplayColor();
                    colors != nullptr &&
                    static_cast<std::size_t>(colors->Elements()) ==
                        data.points.size() &&
                    data.points.size() > 3) {
                    data.colors = FloatsOf(*colors, 3, kDisplayColor);
                }
                if (synced.handle == 0) {
                    synced.handle = points_.Add(std::move(data));
                } else {
                    points_.Set(synced.handle, std::move(data));
                }
            }
        } catch (const std::invalid_argument& error) {
            if (synced.handle != 0) {
                points_.Remove(synced.handle);
                synced.handle = 0;
            }
            errors.push_back(Describe(path, "points", error));
        }
    }
    for (auto it = syncedPoints_.begin();
         it != syncedPoints_.end();) {
        if (seen.contains(it->first)) {
            ++it;
            continue;
        }
        if (it->second.handle != 0) {
            points_.Remove(it->second.handle);
        }
        it = syncedPoints_.erase(it);
    }
    if (!errors.empty()) {
        throw std::invalid_argument("Cannot upload " +
                                    errors.front());
    }
}

std::vector<PointsItem>
RasterDelegate::PointsItems(const Scene& scene) const {
    std::vector<PointsItem> items;
    for (const auto& [path, synced] : syncedPoints_) {
        if (synced.handle == 0 || !scene.IsVisible(path)) {
            continue;
        }
        PointsItem item;
        item.points = synced.handle;
        if (const std::optional<GpuAnchor> anchor =
                GpuAnchorOf(scene, path)) {
            // Under a prim posed on the GPU: posed under its instance
            // on the GPU, bounded by its extent wherever it goes.
            const std::optional<Bounds3> extent =
                XformExtentOf(scene, anchor->at);
            if (!extent) {
                throw std::invalid_argument(
                    "Cannot draw points '" + path.String() +
                    "': GPU-posed prims with points " +
                    "under them need '" +
                    std::string(tokens::kXformExtent) + "' ('" +
                    anchor->at.String() + "' has none)");
            }
            item.transform = anchor->frame;
            item.anchorInstancer = anchor->instancer;
            item.anchorRow = anchor->row;
            item.below = anchor->below;
            item.anchorBounds = *extent;
        } else {
            item.transform = scene.WorldTransform(path);
        }
        item.material = MaterialOf(scene, path);
        item.tint = ConstantDisplayColor(Gprim(scene, path));
        item.primId = PrimIdOf(path);
        item.world = WorldOf(scene, path);
        items.push_back(item);
    }
    return items;
}

void RasterDelegate::Sync(const Scene& scene) {
    ERNIE_ZONE("Sync");
    if (scene.Id() != sceneId_) {
        sceneId_ = scene.Id();
        cursor_.version = scene.Version();
    } else {
        const std::vector<Change> changes = scene.Changes(cursor_);
        // Changes only to cameras and render prims, such as a batch's
        // poses and worlds every step, leave the synced materials,
        // terrain, meshes and draws as they are (lights and settings
        // resolve every frame).
        if (std::all_of(changes.begin(), changes.end(),
                        [&scene](const Change& change) {
                            return IsViewChange(scene, change);
                        })) {
            return;
        }
        // So does a simulation step that only rewrites the arrays of
        // prims posed on the GPU: the drawable takes the new arrays.
        if (RefreshGpuXforms(scene, changes)) {
            renderer_.MarkCastersChanged();
            return;
        }
    }
    // What casts shadows may have moved: the terrain's horizon over
    // it follows.
    renderer_.MarkCastersChanged();
    std::vector<std::string> errors;
    const auto attempt = [&errors](auto&& step) {
        try {
            step();
        } catch (const std::invalid_argument& error) {
            errors.emplace_back(error.what());
        }
    };
    attempt([&] { SyncMaterials(scene); });
    attempt([&] {
        SyncHeightfield(scene, ResolveRenderPrims(scene, {}).terrain);
    });
    attempt([&] { SyncMeshes(scene); });
    // GPU-posed prims first: instancers under them compose under
    // their instances.
    attempt([&] { SyncGpuXforms(scene); });
    attempt([&] { SyncInstancers(scene); });
    attempt([&] { SyncPoints(scene); });
    attempt([&] { points_.SetItems(PointsItems(scene)); });
    // The draws, so the meshes' bounds are known as soon as the scene
    // syncs.
    attempt([&] { meshes_.SetItems(Items(scene)); });
    syncedVersion_ = scene.Version();
    if (!errors.empty()) {
        std::string message = errors.front();
        for (std::size_t i = 1; i < errors.size(); ++i) {
            message += "; " + errors[i];
        }
        throw std::invalid_argument(message);
    }
}

std::vector<MeshItem>
RasterDelegate::Items(const Scene& scene) const {
    ERNIE_ZONE("Sync items");
    std::set<Path> prototypes;
    for (const auto& [path, synced] : instancers_) {
        for (const Path& prototype :
             PointInstancer(scene, path).GetPrototypes()) {
            prototypes.insert(prototype);
        }
    }
    const auto isPrototype = [&prototypes](const Path& path) {
        return std::any_of(prototypes.begin(), prototypes.end(),
                           [&path](const Path& prototype) {
                               return path.HasPrefix(prototype);
                           });
    };
    const auto shade = [this, &scene](const Path& mesh,
                                      const SyncedMesh& synced,
                                      MeshItem& item) {
        item.material = MaterialOf(scene, mesh);
        const bool bound =
            item.material != RasterRenderer::kDefaultMaterial;
        item.tint = bound ? glm::vec3(1.0f) : synced.tint;
        item.vertexColors = !bound;
    };

    std::vector<MeshItem> items;
    for (const auto& [path, synced] : syncedMeshes_) {
        if (synced.handle == 0 || isPrototype(path) ||
            !scene.IsVisible(path)) {
            continue;
        }
        MeshItem item;
        item.mesh = synced.handle;
        if (const std::optional<GpuAnchor> anchor =
                GpuAnchorOf(scene, path)) {
            // Under a prim posed on the GPU: an instance of its group
            // (composed under any GPU-posed prims above it), between
            // the world transform of its frame (on the CPU) and the
            // transforms from it down to the mesh (the prototype).
            item.transform = anchor->frame;
            item.prototype = glm::mat4(anchor->below);
            item.instancer = anchor->instancer;
            item.instances = {anchor->row};
        } else {
            item.transform = scene.WorldTransform(path);
        }
        item.primId = PrimIdOf(path);
        item.world = WorldOf(scene, path);
        shade(path, synced, item);
        items.push_back(std::move(item));
    }
    for (const auto& [path, synced] : instancers_) {
        if (synced.handle == 0 || !scene.IsVisible(path)) {
            continue;
        }
        const PointInstancer instancer(scene, path);
        const std::vector<Path> targets = instancer.GetPrototypes();
        std::vector<std::vector<std::uint32_t>> lists =
            InstancesByPrototype(scene, path, targets.size());
        const Tensor* positions = instancer.GetPositions();
        const std::vector<std::int32_t> worlds = InstanceWorldsOf(
            scene, path,
            positions != nullptr
                ? static_cast<std::size_t>(positions->Elements()) / 3
                : 0,
            HostArrays(scene));
        // Under a GPU-posed prim, its instances are composed under
        // that prim's instance (`SyncInstancers`), drawn in the frame
        // above it.
        const glm::dmat4 world =
            synced.anchorInstancer != 0
                ? GpuAnchorOf(scene, path).value_or(GpuAnchor{}).frame
                : scene.WorldTransform(path);
        for (std::size_t k = 0; k < targets.size(); ++k) {
            const auto mesh = syncedMeshes_.find(targets[k]);
            if (mesh == syncedMeshes_.end() ||
                mesh->second.handle == 0 || lists[k].empty()) {
                continue;
            }
            MeshItem item;
            item.mesh = mesh->second.handle;
            item.transform = world;
            item.prototype =
                glm::mat4(scene.LocalTransform(targets[k]));
            item.instancer = synced.handle;
            item.primId = PrimIdOf(path);
            // Instances sorted by world, so each view draws its
            // world's as one range.
            std::vector<std::uint32_t>& instances = lists[k];
            std::stable_sort(
                instances.begin(), instances.end(),
                [&worlds](std::uint32_t a, std::uint32_t b) {
                    return worlds[a] < worlds[b];
                });
            const bool oneWorld =
                worlds[instances.front()] == worlds[instances.back()];
            item.world = worlds[instances.front()];
            if (!oneWorld) {
                for (const std::uint32_t instance : instances) {
                    item.instanceWorlds.push_back(worlds[instance]);
                }
            }
            item.instances = std::move(instances);
            shade(targets[k], mesh->second, item);
            items.push_back(std::move(item));
        }
    }
    return items;
}

void RasterDelegate::Update(const Scene& scene, const Path& settings,
                            VkExtent2D extent) {
    // The caller records and submits the frame itself, so nothing may
    // still be running.
    context_.WaitForSubmissions();
    BeginFrame();
    window_ = TakeTimeWindow();
    anchoredViews_ = false;
    meshes_.SetAnchoredViews({});
    const RenderPrims prims = ResolveRenderPrims(scene, settings);
    // The view runs the first product's chain (`DefaultFilterChain`
    // without products), holding every var and the AOV slots of what
    // that chain writes.
    const std::vector<SensorFilterStep> filters =
        ResolveRenderProducts(scene, {}, settings).front().filters;
    viewAovs_ =
        filters.empty()
            ? std::vector<std::string>{}
            : AovOutputsOf(filters, renderer_.SensorFilters());
    UpdatePrims(scene, prims, extent,
                NextFrame(scene, prims.settings, prims.settings),
                prims.settings, filters, {}, {},
                ViewHeldVars(viewAovs_.size(), viewMotion_));
    SettleMotion();
}

void RasterDelegate::SettleMotion() {
    // The frame is prepared: what moved in it stands still unless
    // written or mapped again before the next, so writes between
    // frames count as the next one's motion.
    meshes_.SettleMotion();
    points_.SettleMotion();
}

std::vector<std::string> RasterDelegate::ViewVars() const {
    std::vector<std::string> vars = Info().viewVars;
    for (const std::string& aov : viewAovs_) {
        vars.push_back(std::string(kAovPrefix) + aov);
    }
    return vars;
}

void RasterDelegate::PrepareSensorState(const Path& key,
                                        VkExtent2D extent,
                                        std::uint32_t views) {
    // Passes the frame no longer has keep no state.
    std::erase_if(sensorStates_, [&](const auto& entry) {
        return entry.first.first == key &&
               entry.first.second >= renderer_.SensorPasses();
    });
    for (std::size_t pass = 0; pass < renderer_.SensorPasses();
         ++pass) {
        PrepareSensorState(key, pass, extent, views);
    }
}

void RasterDelegate::PrepareSensorState(const Path& key,
                                        std::size_t pass,
                                        VkExtent2D extent,
                                        std::uint32_t views) {
    const std::uint32_t slots = renderer_.SensorStateSlots(pass);
    if (slots == 0) {
        renderer_.SetSensorState(pass, nullptr, nullptr, false);
        sensorStates_.erase({key, pass});
        return;
    }
    const std::uint32_t layers = views * slots;
    if (layers > context_.Limits().maxImageArrayLayers) {
        throw std::invalid_argument(
            "Temporal sensor filters keep " + std::to_string(slots) +
            " state slots per view, too many for " +
            std::to_string(views) + " views on this device");
    }
    SensorState& state = sensorStates_[{key, pass}];
    const std::string chain = renderer_.Plan(pass).Key();
    const GpuImage* existing = state.images[0].get();
    if (existing == nullptr ||
        existing->Desc().width != extent.width ||
        existing->Desc().height != extent.height ||
        existing->Desc().layers != layers || state.chain != chain) {
        // A new history: the images and the chain they hold the state
        // of. Earlier frames may still use the old ones (the viewer's
        // included, which it submits itself): rare, so wait.
        if (existing != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        for (std::unique_ptr<GpuImage>& image : state.images) {
            image = std::make_unique<GpuImage>(
                context_,
                ImageDesc{extent.width, extent.height, layers, 1,
                          VK_FORMAT_R32G32B32A32_SFLOAT,
                          VK_IMAGE_USAGE_STORAGE_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT,
                          false, true});
        }
        state.next = 0;
        state.history = false;
        state.chain = chain;
    }
    renderer_.SetSensorState(pass, state.images[1 - state.next].get(),
                             state.images[state.next].get(),
                             state.history);
    // The next frame reads what this one writes. Frames run on the
    // device in submission order, and the sensor pass's barrier
    // orders this frame's writes before the next one's reads.
    state.next = 1 - state.next;
    state.history = true;
}

void RasterDelegate::UpdatePrims(
    const Scene& scene, const RenderPrims& prims, VkExtent2D extent,
    FrameId id, const Path& stateKey,
    std::span<const SensorFilterStep> filters,
    std::vector<glm::dmat4> poses, std::vector<std::int32_t> worlds,
    std::vector<std::string_view> vars,
    std::optional<glm::dmat4> posesParent,
    const std::optional<std::string>& overlays,
    std::vector<FrameParams::SensorChain> chains, MinimalMode minimal,
    std::optional<glm::vec3> minimalColor) {
    ERNIE_ZONE("UpdatePrims");
    Sync(scene);
    prims_ = prims;
    // Lights under prims posed on the GPU (a lamp on a robot's link):
    // posed there under their instances, the renderer fitting their
    // shadow views from those poses. The CPU plans them at their
    // frames without the instances' poses, an estimate for what does
    // not move with them.
    std::vector<AnchoredView> lightAnchors;
    std::vector<bool> anchoredLights(prims_.lights.size(), false);
    for (std::size_t i = 0; i < prims_.lights.size(); ++i) {
        if (const std::optional<GpuAnchor> anchor =
                GpuAnchorOf(scene, prims_.lights[i])) {
            prims_.lightTransforms[prims_.lights[i]] =
                anchor->frame * anchor->below;
            lightAnchors.push_back({anchor->instancer, anchor->row,
                                    anchor->frame, anchor->below});
            anchoredLights[i] = true;
        }
    }
    meshes_.SetAnchoredLights(std::move(lightAnchors));
    SyncSky(scene, prims_);
    FrameParams frame = [&] {
        ERNIE_ZONE("FrameParamsOf");
        return FrameParamsOf(scene, prims_, extent.width,
                             extent.height);
    }();
    for (std::size_t i = 0;
         i < frame.lights.size() && i < anchoredLights.size(); ++i) {
        frame.lights[i].posedOnGpu = anchoredLights[i];
    }
    ApplyDomeSun(frame);
    ApplyDomeDirections(frame);
    ApplyAutoExposure(scene, stateKey, frame);
    frame.filters.assign(filters.begin(), filters.end());
    // One chain needs no chains of its own: `filters` and `vars` run
    // as it (its filters, which are a var's own when every var has
    // the same).
    if (chains.size() == 1) {
        frame.filters = chains.front().filters;
    } else if (chains.size() > 1) {
        frame.chains = std::move(chains);
    }
    // Its span of the sensor clock, and the animation's time.
    frame.startTime = window_.start;
    frame.time = window_.end;
    const Stage stage(scene);
    frame.timeCode = stage.GetTimeCode();
    frame.timeCodesPerSecond = stage.GetTimeCodesPerSecond();
    frame.poses = std::move(poses);
    frame.posesOnGpu = posesParent.has_value();
    frame.posesQuaternions = frame.posesOnGpu && gpuPosesQuaternions_;
    frame.posesParent = posesParent.value_or(glm::dmat4(1.0));
    // The views' poses at the last frame of `stateKey`, for motion
    // blur: none for the first frame or another view count, which
    // hold the camera still.
    {
        std::vector<glm::dmat4> current =
            frame.poses.empty()
                ? std::vector<glm::dmat4>{frame.camera.transform}
                : frame.poses;
        std::vector<glm::dmat4>& last = lastPoses_[stateKey];
        if (last.size() == current.size() && !frame.posesOnGpu) {
            frame.previousPoses = std::move(last);
        }
        last = std::move(current);
    }
    // A floating origin: the GPU's coordinates are relative to a
    // point near the camera, which moves only when the camera has
    // travelled `kRebaseDistance` from it, so what is relative to it
    // is rebuilt rarely while float precision stays a fraction of a
    // millimetre near the camera.
    const glm::dvec3 eye(frame.poses.empty()
                             ? frame.camera.transform[3]
                             : frame.poses.front()[3]);
    if (glm::length(eye - origin_) > kRebaseDistance) {
        origin_ = eye;
    }
    frame.origin = origin_;
    if (anchoredViews_ && frame.posesOnGpu) {
        // Anchored views' poses are written relative to the origin
        // (`MeshDrawable`).
        frame.posesParent =
            glm::translate(glm::dmat4(1.0), frame.origin);
    }
    frame.vars = std::move(vars);
    // Without a batch's worlds, the single view sees its camera's
    // world.
    frame.worlds =
        worlds.empty() && !prims_.camera.IsEmpty()
            ? std::vector<std::int32_t>{WorldOf(scene, prims_.camera)}
            : std::move(worlds);
    frame.frame = id.frame;
    frame.stream = id.stream;
    TerrainParams terrain = TerrainParamsOf(scene, prims_);
    terrain.material = MaterialOf(scene, prims_.terrain);
    terrain.primId =
        prims_.terrain.IsEmpty() ? 0u : PrimIdOf(prims_.terrain);
    terrain_.SetParams(terrain);
    planet_.SetParams(terrain);
    ground_.SetEnabled(!prims_.ground.IsEmpty());
    if (!prims_.ground.IsEmpty()) {
        GroundParams ground = GroundParamsOf(scene, prims_.ground);
        ground.material = MaterialOf(scene, prims_.ground);
        ground.primId = PrimIdOf(prims_.ground);
        ground_.SetParams(ground);
    }
    // The highlighted prims and everything under them, as the prim
    // ids the frame draws. Each prefix's prims follow it in path
    // order: only theirs are visited.
    std::vector<std::uint32_t> highlighted;
    for (const Path& prefix : highlight_) {
        for (auto it = primIds_.lower_bound(prefix);
             it != primIds_.end() && it->first.HasPrefix(prefix);
             ++it) {
            highlighted.push_back(it->second);
        }
    }
    std::sort(highlighted.begin(), highlighted.end());
    highlighted.erase(
        std::unique(highlighted.begin(), highlighted.end()),
        highlighted.end());
    renderer_.SetHighlight(highlighted, highlightColour_);
    RasterDrawable* drawables[] = {&terrain_, &planet_, &meshes_,
                                   &ground_, &points_};
    if (overlays) {
        frame.overlays = *overlays;
    }
    // Minimal shading as the step resolved it for the product; views
    // draw the full pipeline.
    frame.minimal = minimal;
    if (minimalColor) {
        frame.minimalColor = *minimalColor;
    }
    std::vector<const Overlay*> drawn;
    if (!frame.overlays.empty()) {
        for (const std::string& name :
             SelectOverlays(frame.overlays, scene.OverlayNames())) {
            drawn.push_back(scene.FindOverlay(name));
        }
    }
    const auto views = static_cast<std::uint32_t>(
        std::max<std::size_t>(frame.poses.size(), 1));
    renderer_.Update(frame, drawables, drawn);
    PrepareSensorState(stateKey, extent, views);
}

std::uint32_t RasterDelegate::PrimIdOf(const Path& path) const {
    const auto [it, added] = primIds_.try_emplace(
        path, static_cast<std::uint32_t>(primPaths_.size() + 1));
    if (added) {
        primPaths_.push_back(path);
    }
    return it->second;
}

Path RasterDelegate::PrimPath(std::uint32_t id) const {
    return id >= 1 && id <= primPaths_.size() ? primPaths_[id - 1]
                                              : Path();
}

Path RasterDelegate::Pick(std::uint32_t x, std::uint32_t y) {
    GpuImage* ids = viewTargets_ != nullptr
                        ? viewTargets_->Var(render_vars::kPrimId)
                        : nullptr;
    if (ids == nullptr || x >= viewTargets_->Extent().width ||
        y >= viewTargets_->Extent().height) {
        return {};
    }
    // The caller's frames are submitted outside the context's
    // tracking.
    vkDeviceWaitIdle(context_.Device());
    GpuBuffer texel(context_, sizeof(std::uint32_t),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    BufferMemory::Readback);
    context_.Submit([&](VkCommandBuffer commands) {
        const VkImageLayout layout = ids->Layout();
        ids->Transition(commands,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                                   1};
        region.imageOffset = {static_cast<std::int32_t>(x),
                              static_cast<std::int32_t>(y), 0};
        region.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(commands, ids->Handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               texel.Handle(), 1, &region);
        ids->Transition(commands, layout);
    });
    std::uint32_t id = 0;
    texel.Read(std::as_writable_bytes(std::span(&id, 1)));
    return PrimPath(id);
}

std::vector<TimelinePoint> RasterDelegate::SubmitWaits() const {
    if (compute_ == nullptr ||
        compute_->VulkanWait().semaphore == VK_NULL_HANDLE) {
        return {};
    }
    return {compute_->VulkanWait()};
}

void RasterDelegate::Record(VkCommandBuffer commands,
                            const RenderingTarget& target,
                            std::string_view var) {
    const VkExtent2D extent = renderer_.Extent();
    if (viewTargets_ == nullptr ||
        viewTargets_->Extent().width != extent.width ||
        viewTargets_->Extent().height != extent.height ||
        viewTargets_->Bands() != renderer_.Bands().count ||
        viewTargets_->AovNames() != viewAovs_ ||
        viewTargets_->Faces() != renderer_.Faces() ||
        (viewTargets_->Var(render_vars::kMotionVector) != nullptr) !=
            viewMotion_) {
        if (extent.width == 0 || extent.height == 0) {
            BeginRendering(commands, target);
            vkCmdEndRendering(commands);
            return;
        }
        vkDeviceWaitIdle(context_.Device());
        const std::vector<std::string_view> vars =
            ViewHeldVars(viewAovs_.size(), viewMotion_);
        viewTargets_ = std::make_unique<RenderTargets>(
            context_, extent, renderer_.Bands().count, false, 1, vars,
            viewAovs_, renderer_.Faces());
    }
    const RasterDrawable* drawables[] = {
        &terrain_, &planet_, &meshes_, &ground_, &points_};
    renderer_.Record(commands, *viewTargets_, drawables, &target,
                     var);
}

std::shared_ptr<ProductFrame>
RasterDelegate::AcquireFrame(const Path& product, VkExtent2D extent,
                             int bands, std::uint32_t views,
                             bool batched,
                             std::span<const std::string_view> vars,
                             std::span<const std::string> aovs,
                             std::uint32_t faces, bool minimal) {
    ERNIE_ZONE("AcquireFrame");
    const std::vector<std::string_view> held =
        minimal ? std::vector<std::string_view>{}
                : RequiredVars(vars);
    const std::vector<std::string> minimalHeld =
        minimal ? MinimalTargets::Held(vars)
                : std::vector<std::string>{};
    const std::vector<std::string> aovNames(aovs.begin(), aovs.end());
    std::vector<std::shared_ptr<ProductFrame>>& frames =
        frames_[product];
    const bool shared = compute_ != nullptr;
    std::erase_if(
        frames, [&](const std::shared_ptr<ProductFrame>& frame) {
            bool differs =
                frame->images->Views() != views ||
                frame->batched != batched ||
                frame->images->Extent().width != extent.width ||
                frame->images->Extent().height != extent.height;
            if (const auto* own = dynamic_cast<const MinimalTargets*>(
                    frame->images.get())) {
                differs = differs || !minimal ||
                          own->Shared() != shared ||
                          own->Vars() != minimalHeld;
            } else {
                const RenderTargets& targets =
                    *TargetsOf(*frame->images).front();
                differs = differs || minimal ||
                          targets.Bands() != bands ||
                          targets.Shared() != shared ||
                          targets.Faces() != faces ||
                          targets.Vars() != held ||
                          targets.AovNames() != aovNames;
            }
            const bool stale = frame.use_count() == 1 &&
                               frame->arrayMaps == 0 && differs;
            if (stale) {
                // Earlier steps may still render into it, and maps
                // copy from it: rare, so wait for everything.
                context_.WaitForSubmissions();
            }
            return stale;
        });
    for (const std::shared_ptr<ProductFrame>& frame : frames) {
        if (frame.use_count() == 1 && frame->arrayMaps == 0) {
            return frame;
        }
    }
    auto frame = std::make_shared<ProductFrame>();
    frame->context = &context_;
    frame->compute = compute_.get();
    frame->linearPool = &linearPool_;
    frame->batched = batched;
    // A batch of more views than one set of targets holds renders
    // into consecutive chunks; a minimal frame's attachments hold a
    // chunk at a time and its outputs every view.
    const std::uint32_t chunk =
        RenderTargets::MaxViews(context_, bands, faces);
    if (minimal) {
        frame->images = std::make_unique<MinimalTargets>(
            context_, extent, views,
            RenderTargets::MaxViews(context_, 1, 1), vars, shared);
    } else if (views <= chunk) {
        frame->images = std::make_unique<RenderTargets>(
            context_, extent, bands, shared, views, held, aovNames,
            faces);
    } else {
        std::vector<std::unique_ptr<FrameImages>> chunks;
        for (std::uint32_t first = 0; first < views; first += chunk) {
            chunks.push_back(std::make_unique<RenderTargets>(
                context_, extent, bands, shared,
                std::min(chunk, views - first), held, aovNames,
                faces));
        }
        frame->images =
            std::make_unique<ChunkedFrameImages>(std::move(chunks));
    }
    frames.push_back(frame);
    return frame;
}

std::uint32_t RasterDelegate::BeginFrame() {
    const std::uint32_t slot = nextSlot_;
    nextSlot_ = (nextSlot_ + 1) % FrameUploads::kSlots;
    // Only the frame that last staged its uploads in this slot must
    // have finished: later ones keep running while this one is
    // prepared.
    {
        ERNIE_ZONE("Wait for the slot's frame");
        context_.WaitForSubmission(slotSubmissions_[slot]);
    }
    renderer_.BeginFrame(slot);
    return slot;
}

StepOperation
RasterDelegate::StepAsync(const Scene& scene,
                          std::span<const Path> products,
                          const Path& settings, double deltaTime) {
    ERNIE_ZONE("StepAsync");
    ComputeInterop* compute = Compute();
    const RenderPrims base = ResolveRenderPrims(scene, settings);
    const std::vector<RenderProductSpec> specs =
        ResolveRenderProducts(scene, products, settings);
    // The step's span of the sensor clock, which a frame metered
    // first leaves to it.
    AdvanceTime(deltaTime);
    const TimeWindow window = TakeTimeWindow();
    // A product exposed automatically with nothing metered yet (its
    // first frame) renders a frame to meter first, read as soon as
    // it finishes, so even a single frame is exposed for its scene.
    if (!metering_ && NeedsMetering(scene, base, specs)) {
        metering_ = true;
        try {
            StepAsync(scene, products, settings).Fetch();
        } catch (...) {
            metering_ = false;
            throw;
        }
        metering_ = false;
        context_.WaitForSubmissions();
        renderer_.ReadFinishedMeters();
    }
    window_ = window;
    RenderProductSetOutputs outputs;
    std::uint64_t submission = 0;
    for (const RenderProductSpec& spec : specs) {
        const std::uint32_t slot = BeginFrame();
        RenderPrims prims = base;
        if (!spec.camera.IsEmpty()) {
            prims.camera = spec.camera;
        }
        if (prims.camera.IsEmpty()) {
            throw std::invalid_argument("The scene has no Camera");
        }
        // The product's cameras, or the settings' one.
        const std::vector<Path> cameras =
            spec.cameras.empty() ? std::vector<Path>{prims.camera}
                                 : spec.cameras;
        // Cameras under prims posed on the GPU (a camera on a robot's
        // link): every view posed on the GPU under its anchor
        // (`MeshDrawable::SetAnchoredViews`), with the last ones
        // rendered as the CPU's estimates, or each view's pose
        // without its anchor's at first.
        Sync(scene);
        std::vector<AnchoredView> anchored =
            AnchoredViewsOf(scene, cameras);
        std::vector<glm::dmat4> estimates;
        if (!anchored.empty()) {
            CheckBatchCameras(scene, cameras, spec.resolution.width,
                              spec.resolution.height);
            estimates = renderer_.LastGpuPoses(
                static_cast<std::uint32_t>(anchored.size()));
            if (estimates.empty()) {
                for (const AnchoredView& view : anchored) {
                    estimates.push_back(view.frame * view.below);
                }
            }
            prims.cameraTransform = estimates.front();
        }
        anchoredViews_ = !anchored.empty();
        meshes_.SetAnchoredViews(std::move(anchored));
        const int bands =
            (prims.cameraTransform
                 ? CameraParamsOf(
                       scene, prims.camera, spec.resolution.width,
                       spec.resolution.height, *prims.cameraTransform)
                 : CameraParamsOf(scene, prims.camera,
                                  spec.resolution.width,
                                  spec.resolution.height))
                .bands.count;
        // Poses written as a GPU array stay on the GPU
        // (`GpuPosesOf`), copied device to device after the work
        // writing them, with the last ones rendered as the CPU's
        // estimates; or come to the host for the views' uniforms.
        const Tensor* gpuPoses =
            cameras.size() == 1 && !anchoredViews_
                ? GpuPosesOf(scene, cameras.front())
                : nullptr;
        std::optional<Tensor> hostPoses;
        std::vector<glm::dmat4> poses;
        if (anchoredViews_) {
            poses = estimates;
        } else if (gpuPoses != nullptr) {
            const auto count =
                static_cast<std::uint32_t>(gpuPoses->Shape()[0]);
            poses = renderer_.LastGpuPoses(count);
            if (poses.empty()) {
                const glm::dmat4 own =
                    CameraParamsOf(scene, cameras.front(),
                                   spec.resolution.width,
                                   spec.resolution.height)
                        .transform;
                poses.assign(count, own);
            }
        } else {
            if (cameras.size() == 1) {
                const Tensor* written =
                    scene.Read(cameras.front(), tokens::kPoses);
                if (written != nullptr &&
                    IsGpu(written->Location().kind)) {
                    hostPoses = RequireCompute().Download(
                        *written, scene.WriteStream(cameras.front(),
                                                    tokens::kPoses));
                }
            }
            poses = ViewPosesOf(scene, cameras, spec.resolution.width,
                                spec.resolution.height,
                                hostPoses ? &*hostPoses : nullptr);
        }
        const bool batched = !poses.empty();
        const auto views = static_cast<std::uint32_t>(
            std::max<std::size_t>(poses.size(), 1));
        std::vector<std::int32_t> worlds =
            ViewWorldsOf(scene, cameras, views, HostArrays(scene));
        // Minimal shading, the product's or else its settings'
        // (`tokens::kMinimalMode`): its vars come straight from the
        // raster pass, without sensor filters.
        const MinimalMode minimal = spec.minimal.value_or(
            MinimalModeOf(scene, base.settings, MinimalMode::Off));
        // Only the vars the product asks for, as the static names of
        // `kRenderVarNames`, and its AOVs ("aov:<output>") as the
        // slots its chains lay them in (`kAovSlotNames`), each from
        // its chain: the product's, or a var's own
        // (`RenderProductSpec::varFilters`).
        const SensorFilterRegistry& registry =
            renderer_.SensorFilters();
        std::vector<FrameParams::SensorChain> chains;
        std::vector<std::string> aovs;
        std::vector<std::vector<std::string>> chainAovs;
        std::vector<std::vector<std::string_view>> chainWritten;
        const auto chainOf =
            [&](const std::vector<SensorFilterStep>& steps) {
                const std::vector<SensorFilterStep> full =
                    steps.empty() ? StepsOf(DefaultFilterChain())
                                  : steps;
                for (std::size_t c = 0; c < chains.size(); ++c) {
                    if (chains[c].filters == full) {
                        return c;
                    }
                }
                FrameParams::SensorChain chain;
                chain.filters = full;
                chain.aovOffset =
                    static_cast<std::uint32_t>(aovs.size());
                chainAovs.push_back(AovOutputsOf(full, registry));
                for (const std::string& output : chainAovs.back()) {
                    if (std::find(aovs.begin(), aovs.end(), output) !=
                        aovs.end()) {
                        throw std::invalid_argument(
                            "Two of product '" +
                            spec.product.String() +
                            "'s chains write AOV '" + output + "'");
                    }
                    aovs.push_back(output);
                }
                if (aovs.size() > kMaxAovs) {
                    throw std::invalid_argument(
                        "Product '" + spec.product.String() +
                        "' writes " + "more than " +
                        std::to_string(kMaxAovs) + " AOVs");
                }
                chainWritten.push_back(VarsWrittenBy(full, registry));
                chains.push_back(std::move(chain));
                return chains.size() - 1;
            };
        if (minimal == MinimalMode::Off) {
            chainOf(spec.filters);  // the product's chain first
        }
        std::vector<std::string_view> vars;
        for (std::size_t v = 0; v < spec.vars.size(); ++v) {
            const auto& [path, name] = spec.vars[v];
            if (minimal != MinimalMode::Off) {
                if (!MinimalTargets::Holds(name)) {
                    throw std::invalid_argument(
                        "Render var '" + path.String() +
                        "' asks for '" + name +
                        "', which minimal shading does not render "
                        "(only LdrColor, "
                        "DistanceToImagePlane, DistanceToCamera, "
                        "Normal, PrimId and InstanceId)");
                }
                const auto known =
                    std::find(kRenderVarNames.begin(),
                              kRenderVarNames.end(), name);
                if (std::find(vars.begin(), vars.end(), *known) ==
                    vars.end()) {
                    vars.push_back(*known);
                }
                continue;
            }
            const std::optional<std::vector<SensorFilterStep>>& own =
                v < spec.varFilters.size() ? spec.varFilters[v]
                                           : std::nullopt;
            const std::size_t c = chainOf(own ? *own : spec.filters);
            const auto missing = [&] {
                return std::invalid_argument(
                    "Render var '" + path.String() + "' asks for '" +
                    name +
                    "', which its sensor filters do not write");
            };
            std::string_view held;
            const auto known = std::find(kRenderVarNames.begin(),
                                         kRenderVarNames.end(), name);
            if (known != kRenderVarNames.end()) {
                const std::vector<std::string_view>& written =
                    chainWritten[c];
                if (std::find(written.begin(), written.end(), name) ==
                    written.end()) {
                    throw missing();
                }
                held = *known;
            } else if (name.starts_with(kAovPrefix)) {
                const std::vector<std::string>& outputs =
                    chainAovs[c];
                if (std::find(outputs.begin(), outputs.end(),
                              name.substr(kAovPrefix.size())) ==
                    outputs.end()) {
                    throw missing();
                }
                const auto slot =
                    std::find(aovs.begin(), aovs.end(),
                              name.substr(kAovPrefix.size()));
                held = kAovSlotNames[static_cast<std::size_t>(
                    slot - aovs.begin())];
            } else {
                continue;
            }
            if (std::find(vars.begin(), vars.end(), held) !=
                vars.end()) {
                // The same var twice: the first chain asking for it
                // makes it.
                bool same = false;
                for (const std::string_view other : chains[c].vars) {
                    same = same || other == held;
                }
                if (!same) {
                    throw std::invalid_argument(
                        "Product '" + spec.product.String() +
                        "' asks for '" + name +
                        "' from two sensor filter chains");
                }
                continue;
            }
            vars.push_back(held);
            chains[c].vars.push_back(held);
        }
        // Automatic exposure meters HdrColor: held by the product's
        // chain if no var asks for it, though not output.
        if (minimal == MinimalMode::Off &&
            ExposesAutomatically(scene, prims.camera) &&
            std::find(vars.begin(), vars.end(),
                      render_vars::kHdrColor) == vars.end()) {
            const std::vector<std::string_view>& written =
                chainWritten.front();
            if (std::find(written.begin(), written.end(),
                          render_vars::kHdrColor) != written.end()) {
                vars.push_back(render_vars::kHdrColor);
                chains.front().vars.push_back(render_vars::kHdrColor);
            }
        }
        // Chains no var asks for do not run.
        std::erase_if(chains,
                      [](const FrameParams::SensorChain& chain) {
                          return chain.vars.empty();
                      });
        // A wide fisheye draws its views on cube faces, each a layer
        // of the raster images. Its lens alone decides, not its pose
        // (which may be on the GPU).
        const std::uint32_t faces =
            minimal == MinimalMode::Off
                ? static_cast<std::uint32_t>(CubeFacesOf(
                      CameraParamsOf(scene, cameras.front(),
                                     spec.resolution.width,
                                     spec.resolution.height,
                                     glm::dmat4(1.0))))
                : 1u;
        std::shared_ptr<ProductFrame> frame = AcquireFrame(
            spec.product, spec.resolution, bands, views, batched,
            vars, aovs, faces, minimal != MinimalMode::Off);
        std::optional<glm::dmat4> posesParent;
        if (gpuPoses != nullptr) {
            posesParent = PosesParentOf(scene, cameras.front());
        } else if (anchoredViews_) {
            // Their poses come relative to the frame's origin:
            // `UpdatePrims` puts them under it.
            posesParent = glm::dmat4(1.0);
        }
        gpuPosesQuaternions_ =
            gpuPoses != nullptr && PosesAreQuaternions(*gpuPoses);
        UpdatePrims(scene, prims, spec.resolution,
                    NextFrame(scene, base.settings, spec.product),
                    spec.product, spec.filters, std::move(poses),
                    std::move(worlds), std::move(vars), posesParent,
                    spec.overlays, std::move(chains), minimal,
                    spec.minimalColor);
        if (gpuPoses != nullptr) {
            // Once no submission reads the pose buffer (`UpdatePrims`
            // waits), on the writer's stream; the submission waits
            // for it (`SubmitWaits`).
            const std::uintptr_t stream =
                scene.WriteStream(cameras.front(), tokens::kPoses);
            ComputeInterop& compute = RequireCompute();
            if (gpuPoses->Data() ==
                compute
                    .View(renderer_.EnsurePoseBuffer(1),
                          DType::Float32, {1})
                    .Data()) {
                // Written in place into the pose buffer (`MapArray`):
                // only the writes to wait for.
                compute.SignalVulkan(stream);
            } else if (gpuPosesQuaternions_ &&
                       QuaternionsWFirst(scene, cameras.front())) {
                // Position, w, x, y, z into position, x, y, z, w.
                constexpr std::array<std::size_t, 7> kXyzwOfWxyz{
                    0, 1, 2, 4, 5, 6, 3};
                RequireCompute().CopyColumnsToBuffer(
                    *gpuPoses, kXyzwOfWxyz, renderer_.PoseBuffer(), 0,
                    stream);
            } else {
                RequireCompute().CopyToBuffer(
                    *gpuPoses, renderer_.PoseBuffer(), 0, stream);
            }
        }
        const std::vector<TimelinePoint> waits = SubmitWaits();
        std::vector<TimelinePoint> signals;
        if (compute != nullptr) {
            signals.push_back(compute->NextVulkanSignal());
        }
        auto* minimalTargets =
            dynamic_cast<MinimalTargets*>(frame->images.get());
        const std::vector<RenderTargets*> targets =
            minimalTargets != nullptr ? std::vector<RenderTargets*>{}
                                      : TargetsOf(*frame->images);
        const RasterDrawable* drawables[] = {
            &terrain_, &planet_, &meshes_, &ground_, &points_};
        // Every view of the product renders in this one submission,
        // sharing the shadow passes.
        ERNIE_ZONE_TEXT(spec.product.String());
        submission = context_.SubmitAsync(
            [&](VkCommandBuffer commands) {
                renderer_.RecordShadows(commands, drawables);
                if (minimalTargets != nullptr) {
                    renderer_.RecordMinimal(commands, *minimalTargets,
                                            drawables);
                    return;
                }
                std::uint32_t first = 0;
                for (RenderTargets* chunk : targets) {
                    renderer_.RecordViews(commands, *chunk, drawables,
                                          first);
                    first += chunk->Views();
                }
            },
            waits, signals);
        frame->submission = submission;
        slotSubmissions_[slot] = submission;
        RenderProductOutput& output = outputs[spec.product];
        output.product = spec.product;
        output.startTime = window.start;
        output.time = window.end;
        for (const auto& [path, name] : spec.vars) {
            output.vars.emplace(name,
                                RenderVarOutput(frame, path, name));
        }
    }
    SettleMotion();
    ERNIE_FRAME_MARK("Step");
    return StepOperation(context_, submission, std::move(outputs));
}

}  // namespace ernie
