// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"

#include "ernie/device/gpu_profile.h"

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
#include "ernie/util/log.h"
#include "ernie/util/profile.h"
#include "ernie/util/raster_source.h"
#include "ernie/util/texture_file.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

namespace ernie {

namespace {

/// Prim `path`'s own temperature (`tokens::kThermalTemperature`),
/// `kelvin`, for its drawable.
/// @throws std::invalid_argument below 0 K.
float CheckedTemperature(const Path& path, double kelvin) {
    if (!(kelvin >= 0.0)) {
        throw std::invalid_argument(
            "Prim '" + path.String() + "' has '" +
            std::string(tokens::kThermalTemperature) +
            "' below 0 K: " + std::to_string(kelvin));
    }
    return static_cast<float>(kelvin);
}

/// The vars the view holds for a chain writing `aovs` AOVs, with
/// opt-in var `optIn` (it is shown) unless empty: every var (empty)
/// without either, else every var and the first `aovs` AOV slots.
std::vector<std::string_view> ViewHeldVars(std::size_t aovs,
                                           std::string_view optIn) {
    if (aovs == 0 && optIn.empty()) {
        return {};
    }
    // Every var (the opt-in ones aside unless shown) and the slots.
    std::vector<std::string_view> vars = RequiredVars({});
    if (!optIn.empty()) {
        vars.push_back(optIn);
    }
    vars.insert(vars.end(), kAovSlotNames.begin(),
                kAovSlotNames.begin() +
                    static_cast<std::ptrdiff_t>(aovs));
    return vars;
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
std::string SourceKey(const Scene& scene,
                      const TextureReader& reader) {
    std::string key = reader.Source();
    if (!reader.image.empty()) {
        key += "@" + std::to_string(scene.Id()) + ":" +
               std::to_string(scene.AttributeVersion(
                   Path(reader.image), tokens::kShaderTexture));
    }
    return key;
}

std::string TextureKey(const Scene& scene,
                       const TextureReader& reader) {
    std::string key = SourceKey(scene, reader);
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

/// What `slot`'s upload of `reader`'s texture depends on, so
/// materials that agree on it share one
/// (`RasterDelegate::SharedTexture`): the source, a file's
/// modification time, the colour space, and for normal maps and
/// one-channel textures the scale, bias and channel applied before
/// upload. Wrapping is the sampler's, each material's own.
std::string UploadKey(const Scene& scene, MaterialTextureSlot slot,
                      const TextureReader& reader) {
    std::string key = SourceKey(scene, reader);
    if (reader.image.empty()) {
        std::error_code error;
        const auto time =
            std::filesystem::last_write_time(reader.file, error);
        key += "@" + std::to_string(
                         error ? 0 : time.time_since_epoch().count());
    }
    key += "|" + std::to_string(static_cast<int>(reader.colorSpace));
    const auto values = [&](int c) {
        key += "|" + std::to_string(reader.scale[c]) + "|" +
               std::to_string(reader.bias[c]);
    };
    if (slot == MaterialTextureSlot::Albedo) {
        key +=
            "|albedo" +
            std::to_string(static_cast<int>(reader.virtualTexture));
    } else if (slot == MaterialTextureSlot::Normal) {
        key += "|normal";
        for (int c = 0; c < 3; ++c) {
            values(c);
        }
    } else {
        const int channel = std::clamp(reader.channel, 0, 3);
        key += "|scalar" + std::to_string(channel);
        values(channel);
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

/// What `TextureCache` keys hold beyond the upload key: BC7 from the
/// GPU's encoder; bumped when its output changes.
constexpr std::string_view kCompressedKey = "|gpu-bc7-1";

/// How a texture `TextureCache` holds reads back as stored: in its
/// own format (the reader's colour space already applied), virtual as
/// `reader` says.
TextureReader CachedReader(const TextureReader& reader) {
    TextureReader cached;
    cached.virtualTexture = reader.virtualTexture;
    return cached;
}

/// An image file's texture for `slot` as `RasterDelegate::
/// UploadPrepared` takes it: from `cache` (when given) under `key`
/// (`UploadKey`), setting `cached`, else decoded and prepared. Any
/// thread may call it; it reads no scene.
/// @throws as `LoadImageFile` and `PrepareMaterialTexture`.
PreparedTexture LoadPrepared(MaterialTextureSlot slot,
                             const TextureReader& reader,
                             const TextureCache* cache,
                             const std::string& key,
                             std::optional<TextureData>& cached) {
    ERNIE_ZONE("Load texture");
    if (cache != nullptr) {
        if (std::optional<TextureData> stored =
                cache->Read(key + std::string(kCompressedKey))) {
            if ((stored->format == TextureFormat::Bc7 ||
                 stored->format == TextureFormat::Bc7Srgb) &&
                stored->layers == 1) {
                log::Detail("Texture cache: " + reader.file);
                cached = std::move(stored);
                return {};
            }
        }
    }
    return PrepareMaterialTexture(
        slot, LoadImageFile(reader.file, EncodingOf(reader)), reader);
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
    if (token == tokens::kColormap) {
        return TerrainColorMode::Colormap;
    }
    if (token == tokens::kShadedColormap) {
        return TerrainColorMode::ShadedColormap;
    }
    if (token == tokens::kHillshade) {
        return TerrainColorMode::Hillshade;
    }
    return TerrainColorMode::Shaded;
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
    const glm::dmat3 rotation{glm::dvec3(between[0]) / scale.x,
                              glm::dvec3(between[1]) / scale.y,
                              glm::dvec3(between[2]) / scale.z};
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

/// Whether `change` is to a camera or render prim, or to the stage's
/// metadata, which the frame resolves anew anyway.
bool IsViewChange(const Scene& scene, const Change& change);

/// Whether `rows` can be a posed group's instance channel of
/// `attribute` (`GpuXformGroup::Channels`): contiguous float32, a row
/// of 3 per instance for colours, of 1 for temperatures and
/// visibility.
bool IsInstanceChannel(const Tensor& rows, std::string_view attribute) {
    const std::int64_t width = attribute == tokens::kDisplayColor ? 3 : 1;
    return rows.Type() == DType::Float32 && rows.IsContiguous() &&
           rows.Rank() >= 1 && rows.Rank() <= 2 &&
           (rows.Rank() == 2 ? rows.Shape()[1] == width : width == 1);
}

bool IsViewChange(const Scene& scene, const Change& change) {
    // The stage's metadata on the root, such as the time code a
    // timeline plays every frame: read each frame, or not at all.
    if (change.path.IsRoot()) {
        return true;
    }
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

/// Whether the Mesh at `path` is a point cache the GPU blends: two or
/// more samples of `points`, float32 arrays of one size, all on the
/// CPU or all on the device.
bool IsPointCache(const Scene& scene, const Path& path) {
    if (scene.TypeOf(path) != tokens::kMesh) {
        return false;
    }
    const std::span<const TimeSample> samples =
        scene.SamplesOf(path, tokens::kPoints);
    if (samples.size() < 2 || !samples.front().isTensor) {
        return false;
    }
    return std::ranges::all_of(
        samples, [&](const TimeSample& sample) {
            return sample.tensor.Type() == DType::Float32 &&
                   sample.tensor.Location().kind ==
                       samples.front().tensor.Location().kind &&
                   sample.tensor.Elements() ==
                       samples.front().tensor.Elements();
        });
}

std::string Describe(const Path& path, std::string_view what,
                     const std::exception& error) {
    return "'" + path.String() + "' " + std::string(what) + ": " +
           error.what();
}

/// `shape` averaged over each of `bands` (eight samples per band), or
/// projected onto RGB mode's channels (`BandValue`), times `scale`.
template <typename Shape>
std::vector<float> BinShape(const Shape& shape,
                            const SpectralBands& bands,
                            double scale) {
    std::vector<float> values(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        if (bands.rgb) {
            values[static_cast<std::size_t>(i)] = static_cast<float>(
                scale * BandValue(bands, i, shape));
            continue;
        }
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

/// Whether `path` is a Sphere tessellated `adaptive`, which
/// `AdaptiveSpheres` draws rather than its mesh.
bool IsAdaptiveSphere(const Scene& scene, const Path& path) {
    return scene.TypeOf(path) == tokens::kSphere &&
           Sphere(scene, path).GetTessellation() ==
               tokens::kAdaptiveSphere;
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
        const Terrain terrain(scene, prims.terrain);
        params.exaggeration =
            static_cast<float>(terrain.GetExaggeration());
        params.colorMode = ColorModeOf(terrain.GetColorMode());
        params.hillshadeAzimuth =
            static_cast<float>(terrain.GetHillshadeAzimuth());
        params.hillshadeAltitude =
            static_cast<float>(terrain.GetHillshadeAltitude());
        params.planetCentre =
            glm::dvec3(scene.WorldTransform(prims.terrain)[3]);
        params.finestCell = terrain.GetFinestCell();
    }
    const RenderSettings settings =
        RenderSettings::OrFallbacks(scene, prims.settings);
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
      terrain_(renderer_), planet_(renderer_), spheres_(renderer_),
      meshes_(renderer_), ground_(renderer_), points_(renderer_) {
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
         {terrain_.Bounds(), planet_.BodyBounds(),
          spheres_.BodyBounds(), meshes_.Bounds(),
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

/// Where the scenes the delegate renders keep columns mapped on its
/// compute device (`Column::Map`): the arrays it draws from, written
/// in place (`MapArray`), else for prims it does not draw from such
/// arrays yet the rows as they are, there, until a sync does.
class DelegateColumns final : public ColumnStorage {
  public:
    explicit DelegateColumns(RasterDelegate& delegate)
        : delegate_(delegate) {}

    std::optional<Rows> Allocate(Scene& scene, const PrimSet& prims,
                                 std::string_view attribute, Device device,
                                 std::uintptr_t stream) override {
        ComputeInterop* compute = delegate_.Compute();
        if (compute == nullptr || device != compute->Location()) {
            return std::nullopt;
        }
        try {
            return Rows{delegate_.MapArray(scene, prims, attribute, stream),
                        true};
        } catch (const std::invalid_argument&) {
        }
        try {
            const Tensor host = scene.GetColumn(prims, attribute).Read();
            if (host.Location().kind != DeviceKind::Cpu) {
                return std::nullopt;
            }
            return Rows{compute->Upload(host.Contiguous()), false};
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        }
    }

  private:
    RasterDelegate& delegate_;
};

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
        planet_.GpuBytes() + spheres_.GpuBytes() +
        (viewTargets_ != nullptr ? viewTargets_->GpuBytes() : 0);
    // Material textures once each, however many materials share them.
    for (const auto& [key, cached] : sharedTextures_) {
        if (const auto texture = cached.lock()) {
            stats.gpuBytes += texture->image != nullptr
                                  ? texture->image->Bytes()
                                  : texture->virtualTexture->Bytes();
        }
    }
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
    stats.triangles =
        terrain_.TriangleCount() + planet_.TriangleCount() +
        spheres_.TriangleCount() + meshes_.TriangleCount();
    stats.counters.emplace_back(raster_stats::kTerrainPatches,
                                terrain_.PatchCount() +
                                    planet_.PatchCount() +
                                    spheres_.PatchCount());
    stats.counters.emplace_back(
        raster_stats::kTerrainLodLevels,
        std::max(terrain_.Lod().Levels(), planet_.Lod().Levels()));
    stats.counters.emplace_back(raster_stats::kMeshDraws,
                                meshes_.DrawCount(0));
    stats.counters.emplace_back(raster_stats::kMeshDrawCalls,
                                meshes_.DrawCalls(0));
    stats.counters.emplace_back(raster_stats::kTexturesLoading,
                                loadingTextures_.size() +
                                    compressingTextures_.size());
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
    const std::uint32_t id =
        MaterialId(BoundMaterial(scene, prim))
            .value_or(RasterRenderer::kDefaultMaterial);
    materialPrims_.try_emplace(id, prim);
    return id;
}

void RasterDelegate::SyncTemperatureAxes(const Scene& scene) {
    for (const auto& [path, material] : materials_) {
        if (!material.temperatureModel) {
            continue;
        }
        const auto prim = materialPrims_.find(material.id);
        if (prim == materialPrims_.end()) {
            continue;
        }
        // A prim gone or bound elsewhere since: the next one seen.
        if (!scene.Has(prim->second) ||
            BoundMaterial(scene, prim->second) != path) {
            materialPrims_.erase(prim);
            continue;
        }
        const glm::dvec3 axis(scene.WorldTransform(prim->second)[2]);
        const double length = glm::length(axis);
        if (length > 0.0) {
            renderer_.SetTemperatureAxis(material.id,
                                         glm::vec3(axis / length));
        }
    }
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
    std::erase_if(sharedTextures_, [](const auto& entry) {
        return entry.second.expired();
    });
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
    material.temperatureModel = !settings.temperatureModel.empty();
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
        material.pending[slot].clear();
        material.readers[slot] = reader.value_or(TextureReader());
        try {
            // The readers come in `MaterialTextureSlot`'s order.
            const auto textureSlot =
                static_cast<MaterialTextureSlot>(slot);
            std::shared_ptr<const MaterialTexture> texture =
                reader ? SharedTexture(scene, textureSlot, *reader,
                                       material.pending[slot])
                       : nullptr;
            // A texture still loading reads a placeholder meanwhile.
            gpu.SetTexture(textureSlot,
                           material.pending[slot].empty()
                               ? std::move(texture)
                               : LoadingTexture(textureSlot),
                           material.readers[slot],
                           !material.pending[slot].empty());
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
    // Its temperature texture: its thermal terminal's, one channel in
    // kelvin after the reader's scale and bias, else its own tensor.
    const Key temperatureKey =
        KeyOf(scene, path, {kThermalTemperatureTexture});
    const std::string temperatureReader =
        settings.temperatureTexture
            ? TextureKey(scene, *settings.temperatureTexture)
            : std::string();
    if (material.temperatureTexture != temperatureKey ||
        material.temperatureReader != temperatureReader) {
        material.temperatureTexture = temperatureKey;
        material.temperatureReader = temperatureReader;
        try {
            if (settings.temperatureTexture) {
                TextureReader reader = *settings.temperatureTexture;
                reader.colorSpace = TextureColorSpace::Raw;
                const Image image = TextureImageOf(scene, reader);
                const auto c =
                    static_cast<glm::length_t>(reader.channel);
                std::vector<float> kelvin;
                kelvin.reserve(image.pixels.size());
                for (const glm::vec4& texel : image.pixels) {
                    kelvin.push_back(texel[c] * reader.scale[c] +
                                     reader.bias[c]);
                }
                gpu.SetTemperatureTexture(kelvin, image.width,
                                          image.height);
            } else if (const Tensor* tensor =
                           prim.GetThermalTemperatureTexture()) {
                const Tensor kelvin = tensor->Contiguous();
                const std::vector<std::int64_t> shape =
                    kelvin.Shape();
                if (kelvin.Type() != DType::Float32 ||
                    (shape.size() != 2 &&
                     !(shape.size() == 3 && shape[2] == 1)) ||
                    shape[0] < 1 || shape[1] < 1) {
                    throw std::invalid_argument(
                        "needs float32 (rows, cols) or (rows, cols, "
                        "1)");
                }
                channel.height = static_cast<std::uint32_t>(shape[0]);
                channel.width = static_cast<std::uint32_t>(shape[1]);
                channel.values.assign(
                    values.As<float>(),
                    values.As<float>() +
                        static_cast<std::size_t>(channel.width) *
                            channel.height);
            }
        } catch (const std::invalid_argument& error) {
            throw UploadError(path, attribute, error.what());
        }
        return channel;
    };
    const Channel kelvin =
        load(settings.temperatureTexture,
             prim.GetThermalTemperatureTexture(),
             kThermalTemperatureTexture);
    const Channel inertia =
        load(settings.inertiaTexture, prim.GetThermalInertiaTexture(),
             kThermalInertiaTexture);
    try {
        if (!kelvin.values.empty() && !inertia.values.empty() &&
            (kelvin.width != inertia.width ||
             kelvin.height != inertia.height)) {
            throw std::invalid_argument(
                "its temperature and thermal inertia textures differ "
                "in size");
        }
        const Channel& size = kelvin.values.empty() ? inertia : kelvin;
        if (!size.values.empty() || gpu.HasThermalTexture()) {
            gpu.SetThermalTexture(kelvin.values, inertia.values,
                                  size.width, size.height);
        }
    } catch (const std::invalid_argument& error) {
        throw UploadError(path, kThermalInertiaTexture, error.what());
    }
}

std::shared_ptr<const MaterialTexture> RasterDelegate::SharedTexture(
    const Scene& scene, MaterialTextureSlot slot,
    const TextureReader& reader, std::string& pending) {
    ERNIE_ZONE("Shared texture");
    pending.clear();
    const std::string key = UploadKey(scene, slot, reader);
    std::weak_ptr<const MaterialTexture>& shared =
        sharedTextures_[key];
    if (auto texture = shared.lock()) {
        return texture;
    }
    std::shared_ptr<const MaterialTexture> texture;
    if (!reader.image.empty()) {
        // Tensors, which change as they please, upload uncompressed.
        texture = UploadMaterialTexture(
            context_, slot, TextureImageOf(scene, reader), reader);
    } else if (IsTextureFile(reader.file)) {
        // DDS and KTX2 files upload as stored, mapped (virtual
        // textures read their tiles as they are seen).
        texture = UploadMaterialTexture(
            context_, slot, LoadTextureFile(reader.file), reader);
    } else {
        if (const auto failed = failedTextures_.find(key);
            failed != failedTextures_.end()) {
            throw std::invalid_argument(failed->second);
        }
        // Albedo too large to decode whole (a planet's) bakes a band
        // at a time into the texture cache and reads as a virtual
        // texture of it, mapped: at once when it is cached.
        const bool baked = slot == MaterialTextureSlot::Albedo &&
                           IsBakedImageFile(reader.file);
        if (baked) {
            if (std::optional<TextureData> stored =
                    textureCache_.Read(key +
                                       std::string(kCompressedKey))) {
                texture = UploadMaterialTexture(context_, slot,
                                                std::move(*stored),
                                                CachedReader(reader));
                shared = texture;
                return texture;
            }
        }
        // Decoded and mipped, or its bake begun, off the main thread;
        // `FinishTextureLoads` uploads it.
        if (!loadingTextures_.contains(key) &&
            !compressingTextures_.contains(key)) {
            const TextureCache* cache =
                Encoder() != nullptr ? &textureCache_ : nullptr;
            // Bakes write beside the cache, else to a file of their
            // own.
            std::filesystem::path output;
            if (baked) {
                output =
                    textureCache_
                        .PartialPath(key +
                                     std::string(kCompressedKey))
                        .value_or(
                            std::filesystem::temp_directory_path() /
                            ("ernie_bake_" +
                             std::to_string(std::random_device{}()) +
                             ".dds"));
            }
            loadingTextures_.emplace(
                key,
                textureLoaders_.Submit([slot, reader, cache, key,
                                        output, encoder = Encoder()] {
                    LoadedTexture loaded;
                    loaded.reader = reader;
                    if (!output.empty()) {
                        loaded.bake = BakeAlbedo(reader.file, output,
                                                 reader, encoder);
                        loaded.output = output;
                        return loaded;
                    }
                    loaded.prepared = LoadPrepared(
                        slot, reader, cache, key, loaded.cached);
                    // An encoder there (and so a cache) or not.
                    if (!loaded.cached && cache != nullptr) {
                        loaded.compression.emplace(
                            std::move(loaded.prepared));
                    }
                    return loaded;
                }));
        }
        pending = key;
        return nullptr;
    }
    shared = texture;
    return texture;
}

void RasterDelegate::FinishTextureLoads(bool wait) {
    if (loadingTextures_.empty() && compressingTextures_.empty()) {
        return;
    }
    const std::vector<std::string> errors = ProgressTextureLoads(
        wait, wait ? ~std::uint64_t{0} : kCompressTexelsPerFrame);
    if (errors.empty()) {
        return;
    }
    if (wait) {
        std::string message = errors.front();
        for (std::size_t i = 1; i < errors.size(); ++i) {
            message += "; " + errors[i];
        }
        throw std::invalid_argument(message);
    }
    for (const std::string& message : errors) {
        log::Info(message);
    }
}

std::vector<std::string>
RasterDelegate::ProgressTextureLoads(bool waitLoads,
                                     std::uint64_t compressTexels) {
    ERNIE_ZONE("Finish texture loads");
    std::vector<std::string> errors;
    const auto fail = [&](const std::string& key,
                          const std::string& error) {
        failedTextures_[key] = error;
        DeliverTexture(key, nullptr, error, errors);
    };
    // Loads finished (all, waiting) go to compression, or straight to
    // the materials when cached or without an encoder.
    for (auto it = loadingTextures_.begin();
         it != loadingTextures_.end();) {
        if (!waitLoads && it->second.wait_for(std::chrono::seconds(
                              0)) != std::future_status::ready) {
            ++it;
            continue;
        }
        const std::string key = it->first;
        try {
            LoadedTexture loaded = it->second.get();
            if (loaded.compression || loaded.bake) {
                compressingTextures_.emplace(
                    key,
                    CompressingTexture{std::move(loaded.compression),
                                       std::move(loaded.bake),
                                       std::move(loaded.output),
                                       loaded.reader});
            } else {
                const std::shared_ptr<const MaterialTexture> texture =
                    UploadPrepared(
                        std::make_shared<const TextureData>(
                            loaded.cached
                                ? std::move(*loaded.cached)
                                : StoredOf(loaded.prepared)),
                        key, false);
                sharedTextures_[key] = texture;
                DeliverTexture(key, texture, {}, errors);
            }
        } catch (const std::exception& failure) {
            fail(key, failure.what());
        }
        it = loadingTextures_.erase(it);
    }
    // Compression, up to the budget; those done are cached, uploaded
    // and given to their materials.
    std::uint64_t budget = compressTexels;
    for (auto it = compressingTextures_.begin();
         it != compressingTextures_.end() && budget > 0;) {
        const std::string key = it->first;
        CompressingTexture& compressing = it->second;
        try {
            bool done = false;
            if (compressing.bake) {
                // Waiting, for the bake's bands to be read too.
                const std::uint64_t before =
                    compressing.bake->Encoded();
                done = compressing.bake->Step(budget, waitLoads);
                budget -= std::min(
                    budget, compressing.bake->Encoded() - before);
            } else {
                TextureCompression& compression =
                    *compressing.compression;
                const std::uint64_t before = compression.Remaining();
                done = compression.Step(*Encoder(), budget);
                budget -= std::min(budget,
                                   before - compression.Remaining());
            }
            if (!done) {
                ++it;
                continue;
            }
            const std::shared_ptr<const MaterialTexture> texture =
                compressing.bake
                    ? UploadBaked(compressing.output, key,
                                  compressing.reader)
                    : UploadPrepared(
                          std::make_shared<const TextureData>(
                              compressing.compression->TakeStored()),
                          key, true);
            sharedTextures_[key] = texture;
            DeliverTexture(key, texture, {}, errors);
        } catch (const std::exception& failure) {
            fail(key, failure.what());
        }
        it = compressingTextures_.erase(it);
    }
    return errors;
}

void RasterDelegate::DeliverTexture(
    const std::string& key,
    const std::shared_ptr<const MaterialTexture>& texture,
    const std::string& error, std::vector<std::string>& errors) {
    using namespace tokens;
    constexpr std::array<std::string_view, 5> kInputs{
        kDiffuseColor, kNormalInput, kRoughness, kMetallic, kOpacity};
    for (auto& [path, material] : materials_) {
        for (std::size_t slot = 0; slot < material.pending.size();
             ++slot) {
            if (material.pending[slot] != key) {
                continue;
            }
            material.pending[slot].clear();
            // A failed texture leaves the slot to the material's
            // constant, as no texture would.
            renderer_.Material(material.id)
                .SetTexture(static_cast<MaterialTextureSlot>(slot),
                            texture, material.readers[slot]);
            if (texture == nullptr) {
                errors.emplace_back(
                    UploadError(path, kInputs[slot], error).what());
            }
        }
    }
}

void RasterDelegate::EvictTextures(bool cache) {
    ERNIE_ZONE("Evict textures");
    // No frame in flight may read what goes.
    vkDeviceWaitIdle(context_.Device());
    if (cache) {
        const auto forget = [&](const std::string& key) {
            textureCache_.Remove(key + std::string(kCompressedKey));
        };
        for (const auto& [key, texture] : sharedTextures_) {
            forget(key);
        }
        for (const auto& [key, loading] : loadingTextures_) {
            forget(key);
        }
        for (const auto& [key, compressing] : compressingTextures_) {
            forget(key);
        }
    }
    for (auto& [path, material] : materials_) {
        RasterMaterial& gpu = renderer_.Material(material.id);
        for (std::size_t slot = 0; slot < material.textures.size();
             ++slot) {
            gpu.SetTexture(static_cast<MaterialTextureSlot>(slot),
                           nullptr, material.readers[slot]);
            material.textures[slot].clear();
            material.pending[slot].clear();
        }
    }
    // Loads under way finish on their threads, unread; bakes stop on
    // one (as their bands are read), removing their files.
    loadingTextures_.clear();
    for (auto& [key, compressing] : compressingTextures_) {
        if (compressing.bake) {
            textureLoaders_.Submit(
                [bake = std::move(compressing.bake)]() mutable {
                    bake.reset();
                });
        }
    }
    compressingTextures_.clear();
    failedTextures_.clear();
    sharedTextures_.clear();
    // The next sync syncs every material again, as for a new scene.
    sceneId_ = 0;
    settledVersion_ = {~0ull, ~0ull};
}

TextureWait RasterDelegate::WaitForTextures(const Scene& scene,
                                            double timeoutSeconds) {
    ERNIE_ZONE("Wait for textures");
    Sync(scene);
    using Clock = std::chrono::steady_clock;
    const bool forever = !(timeoutSeconds < 1e9);
    const Clock::time_point deadline =
        forever ? Clock::time_point::max()
                : Clock::now() +
                      std::chrono::duration_cast<Clock::duration>(
                          std::chrono::duration<double>(
                              std::max(timeoutSeconds, 0.0)));
    for (auto& [key, ready] : loadingTextures_) {
        if (!forever &&
            ready.wait_until(deadline) != std::future_status::ready) {
            break;
        }
        if (forever) {
            ready.wait();
        }
    }
    TextureWait result;
    result.failed = ProgressTextureLoads(false, ~std::uint64_t{0});
    // Bakes, waiting for their bands, some 16 frames' worth at a time
    // until the deadline.
    while (!compressingTextures_.empty() &&
           loadingTextures_.empty() && Clock::now() < deadline) {
        const std::vector<std::string> failed =
            ProgressTextureLoads(true, 16 * kCompressTexelsPerFrame);
        result.failed.insert(result.failed.end(), failed.begin(),
                             failed.end());
    }
    // A planet's heights file, until the deadline.
    if (pendingPlanet_ &&
        (forever || pendingPlanet_->load.wait_until(deadline) ==
                        std::future_status::ready)) {
        try {
            FinishPlanetLoad(true);
        } catch (const std::exception& error) {
            result.failed.emplace_back(error.what());
        }
    }
    result.loading = loadingTextures_.size() +
                     compressingTextures_.size() +
                     (pendingPlanet_ ? 1u : 0u);
    return result;
}

std::shared_ptr<const MaterialTexture>
RasterDelegate::LoadingTexture(MaterialTextureSlot slot) {
    if (slot != MaterialTextureSlot::Albedo) {
        return nullptr;
    }
    if (loadingAlbedo_ == nullptr) {
        // 8 x 8 cells of 8 texels, light and mid grey.
        constexpr std::uint32_t kSide = 64;
        Image checker(kSide, kSide);
        for (std::uint32_t y = 0; y < kSide; ++y) {
            for (std::uint32_t x = 0; x < kSide; ++x) {
                checker.At(x, y) = glm::vec4(
                    glm::vec3((x / 8 + y / 8) % 2 == 0 ? 0.6f : 0.3f),
                    1.0f);
            }
        }
        loadingAlbedo_ = UploadMaterialTexture(
            context_, slot, checker, TextureReader());
    }
    return loadingAlbedo_;
}

bool RasterDelegate::WaitsForTextures(const Scene& scene,
                                      const Path& settings) const {
    const std::string loading =
        settings.IsEmpty()
            ? std::string(tokens::kAuto)
            : RenderSettings(scene, settings).GetTextureLoading();
    if (loading == tokens::kAuto) {
        return !viewportFrame_;
    }
    if (loading != tokens::kComplete && loading != tokens::kAsync) {
        throw std::invalid_argument(
            "'" + std::string(tokens::kTextureLoading) + "' of '" +
            settings.String() +
            "' must be \"auto\", \"complete\" or \"async\"");
    }
    return loading == tokens::kComplete;
}

BlockEncoder* RasterDelegate::Encoder() {
    if (!encoderTried_) {
        encoderTried_ = true;
        if (context_.SupportsTextureCompressionBc()) {
            try {
                encoder_ = std::make_unique<GpuBlockEncoder>(
                    context_, renderer_.Compiler());
            } catch (const std::exception&) {
                // Textures upload uncompressed.
            }
        }
    }
    return encoder_.get();
}

std::shared_ptr<const MaterialTexture> RasterDelegate::UploadPrepared(
    std::shared_ptr<const TextureData> stored, const std::string& key,
    bool cache) {
    ERNIE_ZONE("Upload prepared texture");
    // Written to the cache by a loader thread, so this frame need not
    // wait for the disk.
    if (cache) {
        textureLoaders_.Submit(
            [&textureCache = textureCache_, stored,
             cacheKey = key + std::string(kCompressedKey)] {
                textureCache.Write(cacheKey, *stored);
            });
    }
    return UploadMaterialTexture(context_, *stored);
}

std::shared_ptr<const MaterialTexture>
RasterDelegate::UploadBaked(const std::filesystem::path& output,
                            const std::string& key,
                            const TextureReader& reader) {
    ERNIE_ZONE("Upload baked texture");
    const std::string cacheKey = key + std::string(kCompressedKey);
    std::optional<TextureData> stored;
    if (textureCache_.Enabled()) {
        if (textureCache_.Publish(output, cacheKey)) {
            stored = textureCache_.Read(cacheKey);
        }
        if (!stored) {
            throw std::runtime_error("Cannot cache the texture baked "
                                     "into " +
                                     output.string());
        }
    } else {
        // Without a cache the bake's own file, mapped, goes once read
        // (its mapping keeps it).
        stored = LoadTextureFile(output);
        std::error_code error;
        std::filesystem::remove(output, error);
    }
    return UploadMaterialTexture(
        context_, MaterialTextureSlot::Albedo, std::move(*stored),
        CachedReader(reader));
}

RasterDelegate::PlanetLoad
RasterDelegate::LoadPlanetFile(const std::filesystem::path& file,
                               const glm::dvec3& radii,
                               const PlanetHeightsOptions& options) {
    ERNIE_ZONE("Planet heights file");
    std::shared_ptr<const RasterSource> source = OpenRaster(file);
    const LatLongGrid grid =
        options.grid.value_or(source->Grid().value_or(
            LatLongGrid{-180.0, 90.0, 360.0 / source->Width(),
                        180.0 / source->Height()}));
    // The file's quads along a face, at the equator: faces of up to
    // 1024 (whole pages of 256), the rest of the depth paged.
    const double cells =
        90.0 / std::min(grid.degreesAcross, grid.degreesDown);
    const auto wanted = static_cast<std::uint32_t>(
        std::min(std::ceil(cells), double{1u << 20}));
    constexpr std::uint32_t kBaseCells = 1024;
    const std::uint32_t base =
        std::clamp(std::bit_ceil(std::max(wanted, 1u)),
                   PlanetPages::kPageCells, kBaseCells);
    std::uint32_t depth = 0;
    while ((std::uint64_t{base} << depth) < wanted) {
        ++depth;
    }
    PlanetLoad load;
    load.faces = PlanetFacesOf(*source, options, radii, base);
    load.source = std::move(source);
    load.base = base;
    load.depth = depth;
    return load;
}

void RasterDelegate::FinishPlanetLoad(bool wait) {
    if (!pendingPlanet_ ||
        (!wait && pendingPlanet_->load.wait_for(std::chrono::seconds(
                      0)) != std::future_status::ready)) {
        return;
    }
    PendingPlanet pending = std::move(*pendingPlanet_);
    pendingPlanet_.reset();
    try {
        PlanetLoad load = pending.load.get();
        const auto side = static_cast<std::size_t>(load.base) + 1;
        std::array<RasterView, CubeSphereLod::kFaces> views;
        for (std::size_t face = 0; face < views.size(); ++face) {
            views[face].data =
                load.faces.heights.data() + face * side * side;
            views[face].width = side;
            views[face].height = side;
        }
        planet_.SetHeights(pending.radii, views, load.faces.slopes,
                           load.faces.ranges);
        if (load.depth > 0) {
            planet_.SetPages(std::make_unique<PlanetPages>(
                context_, std::move(load.source), pending.options,
                pending.radii, load.base, load.depth));
        }
        log::Info("Planet heights from " + pending.file.string() +
                  " active: faces of " + std::to_string(load.base) +
                  " quads" +
                  (load.depth > 0
                       ? ", paged to " +
                             std::to_string(load.base << load.depth)
                       : std::string()));
    } catch (const std::exception& error) {
        const std::invalid_argument failure = UploadError(
            pending.terrain, tokens::kHeightsFile, error.what());
        if (wait) {
            throw failure;
        }
        log::Info(failure.what());
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
    // A planet's heights from a file: faces of up to 1024 quads read
    // off the frame (the planet drawn as it was meanwhile), finer
    // ones paged in as views need them.
    if (prim && !prim->GetHeightsFile().empty() &&
        (prim->HasPlanetRadius() || prim->HasPlanetRadii())) {
        const Key key =
            KeyOf(scene, terrain,
                  {kHeightsFile, kHeightsUnit, kHeightsReference,
                   kPlanetRadius, kPlanetRadii});
        if (heightfield_ == key) {
            return;
        }
        heightfield_ = key;
        if (terrain_.HasHeightfield()) {
            terrain_.ClearHeightfield();
        }
        PendingPlanet pending;
        pending.terrain = terrain;
        pending.file = prim->GetHeightsFile();
        pending.radii = prim->HasPlanetRadii()
                            ? prim->GetPlanetRadii()
                            : glm::dvec3(prim->GetPlanetRadius());
        pending.options = PlanetHeightsOptions{
            prim->GetHeightsUnit(), prim->GetHeightsReference(),
            std::nullopt};
        log::Info("Loading planet heights from " +
                  pending.file.string());
        pending.load = textureLoaders_.Submit(
            [file = pending.file, radii = pending.radii,
             options = pending.options] {
                return LoadPlanetFile(file, radii, options);
            });
        pendingPlanet_ = std::move(pending);
        return;
    }
    pendingPlanet_.reset();
    const Tensor* written = prim ? prim->GetHeights() : nullptr;
    if (written == nullptr) {
        if (heightfield_) {
            terrain_.ClearHeightfield();
            planet_.ClearHeights();
            heightfield_.reset();
        }
        return;
    }
    const Key key = KeyOf(scene, terrain,
                          {kHeights, kOrigin, kCellSize, kNoData,
                           kPlanetRadius, kPlanetRadii});
    if (heightfield_ == key) {
        return;
    }
    heightfield_ = key;
    if (prim->HasPlanetRadius() || prim->HasPlanetRadii()) {
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
            planet_.SetHeights(
                prim->HasPlanetRadii()
                    ? prim->GetPlanetRadii()
                    : glm::dvec3(prim->GetPlanetRadius()),
                faces);
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
    // How far each frame moves towards the metered exposure: the
    // camera's adaptation, over frames, or the viewer's own.
    std::optional<float> blend;
    if (ExposesAutomatically(scene, prims_.camera)) {
        const double frames =
            Camera(scene, prims_.camera).GetExposureAdaptation();
        blend =
            frames > 0.0
                ? static_cast<float>(1.0 - std::exp(-1.0 / frames))
                : 1.0f;
    } else if (viewportFrame_ && viewportAutoExposure_) {
        blend = *viewportAutoExposure_;
    }
    renderer_.SetAutoExposure(blend);
    if (!blend) {
        return;
    }
    // Metered every frame under its key, which the renderer's GPU
    // state for it adapts to, for the frames after it; batches of
    // views meter none and keep their camera's exposure.
    frame.meter = true;
    renderer_.SetMeterKey(stateKey.String());
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
                static_cast<double>(frame.dome->radiance[b] *
                                    shape[b]) *
                peak);
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
        spectrum[b] = static_cast<float>(
            static_cast<double>(frame.dome->radiance[b] * shape[b]) *
            peak);
    }
    const glm::vec3 towards(
        glm::normalize(rotation * domeSun_->direction));
    LightParams sun;
    sun.type = LightType::Distant;
    sun.direction = towards;
    sun.angle = static_cast<float>(domeSun_->angle);
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
        renderer_.SetOverlayComputeInterop(compute);
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

void RasterDelegate::SetMeshIndices(const Scene& scene,
                                    const Path& path,
                                    std::uint32_t handle) {
    RequireCompute();
    meshes_.SetIndices(
        handle, *scene.Read(path, tokens::kFaceVertexIndices),
        scene.WriteStream(path, tokens::kFaceVertexIndices));
}

void RasterDelegate::SetMeshActiveCounts(const Scene& scene,
                                         const Path& path,
                                         std::uint32_t handle) {
    const Tensor* counts = scene.Read(path, tokens::kActiveCounts);
    if (counts == nullptr) {
        meshes_.SetActiveCounts(handle, Tensor());
        return;
    }
    if (IsGpu(counts->Location().kind)) {
        RequireCompute();
    }
    meshes_.SetActiveCounts(
        handle, *counts,
        scene.WriteStream(path, tokens::kActiveCounts));
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
    if (colors != nullptr && IsGpu(colors->Location().kind) &&
        colors->Elements() != 3) {
        RequireCompute();
        meshes_.SetColors(
            handle, *colors,
            scene.WriteStream(path, tokens::kDisplayColor));
    }
}

void RasterDelegate::SetMeshTemperatures(const Scene& scene,
                                         const Path& path,
                                         std::uint32_t handle) {
    ERNIE_ZONE("Thermal point temperatures");
    const Tensor* kelvin =
        scene.Read(path, tokens::kThermalTemperatures);
    if (kelvin == nullptr || kelvin->Empty()) {
        meshes_.SetTemperatures(handle, std::span<const float>());
        return;
    }
    if (IsGpu(kelvin->Location().kind)) {
        RequireCompute();
        meshes_.SetTemperatures(
            handle, *kelvin,
            scene.WriteStream(path, tokens::kThermalTemperatures));
        return;
    }
    if (kelvin->Type() != DType::Float32 || kelvin->Rank() != 1) {
        throw std::invalid_argument(
            "'" + std::string(tokens::kThermalTemperatures) +
            "' needs float32 (n,)");
    }
    const Tensor host = kelvin->Contiguous();
    meshes_.SetTemperatures(
        handle, std::span(host.As<float>(),
                          static_cast<std::size_t>(host.Elements())));
}

void RasterDelegate::SetMeshNodes(const Scene& scene,
                                  const Path& path,
                                  std::uint32_t handle, bool ids) {
    ERNIE_ZONE("Thermal nodes");
    using tokens::kThermalNodeIds;
    using tokens::kThermalNodeTemperatures;
    if (ids) {
        const Tensor* nodes = scene.Read(path, kThermalNodeIds);
        std::vector<std::uint32_t> values;
        if (nodes != nullptr && !nodes->Empty()) {
            const Tensor host = nodes->Contiguous();
            if (host.Rank() != 1 || (host.Type() != DType::Int32 &&
                                     host.Type() != DType::UInt32)) {
                throw std::invalid_argument(
                    "'" + std::string(kThermalNodeIds) +
                    "' needs int32 (n,)");
            }
            const auto* raw = host.As<std::int32_t>();
            for (std::int64_t i = 0; i < host.Elements(); ++i) {
                if (host.Type() == DType::Int32 && raw[i] < 0) {
                    throw std::invalid_argument(
                        "'" + std::string(kThermalNodeIds) +
                        "' holds a negative node");
                }
                values.push_back(static_cast<std::uint32_t>(raw[i]));
            }
        }
        meshes_.SetNodeIds(handle, values);
    }
    const Tensor* kelvin = scene.Read(path, kThermalNodeTemperatures);
    if (kelvin == nullptr || kelvin->Empty()) {
        meshes_.SetNodeTemperatures(handle, std::span<const float>());
        return;
    }
    if (IsGpu(kelvin->Location().kind)) {
        RequireCompute();
        meshes_.SetNodeTemperatures(
            handle, *kelvin,
            scene.WriteStream(path, kThermalNodeTemperatures));
        return;
    }
    if (kelvin->Type() != DType::Float32 || kelvin->Rank() != 1) {
        throw std::invalid_argument(
            "'" + std::string(kThermalNodeTemperatures) +
            "' needs float32 (nodes,)");
    }
    const Tensor host = kelvin->Contiguous();
    meshes_.SetNodeTemperatures(
        handle, std::span(host.As<float>(),
                          static_cast<std::size_t>(host.Elements())));
}

std::optional<std::vector<Path>>
RasterDelegate::ChangedMeshes(const Scene& scene,
                              std::span<const Change> changes) const {
    using namespace tokens;
    // Skinned meshes are posed every sync whatever changed
    // (`PoseSkins`); a new refine level refines every surface.
    if (scene.AttributeVersion(Path::Root(),
                               kSubdivisionRefineLevel) !=
        subdivisionVersion_) {
        return std::nullopt;
    }
    std::vector<Path> meshes;
    for (const Change& change : changes) {
        const Path& path = change.path;
        if (path.IsRoot()) {
            continue;
        }
        const std::string_view type =
            scene.Has(path) ? scene.TypeOf(path) : std::string_view();
        // A synced mesh only moved or hidden (a character's root
        // motion, say) keeps its data: its draws follow (`Items`).
        constexpr auto kMoved = static_cast<std::uint32_t>(
            Dirty::Transform | Dirty::Visibility);
        if (syncedMeshes_.contains(path) &&
            (static_cast<std::uint32_t>(change.bits) & ~kMoved) ==
                0) {
            continue;
        }
        if (IsMeshType(type) || syncedMeshes_.contains(path)) {
            meshes.push_back(path);
        }
    }
    return meshes;
}

void RasterDelegate::SyncMeshes(const Scene& scene,
                                const std::vector<Path>* only) {
    ERNIE_ZONE("Sync meshes");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    std::vector<Path> paths;
    // Meshes replaced or gone, removed at once at the end: one at a
    // time, a new scene's sync took quadratic time in its meshes.
    std::vector<std::uint32_t> dropped;
    bool skinned = false;
    if (only != nullptr) {
        for (const Path& path : *only) {
            if (scene.Has(path) && IsMeshType(scene.TypeOf(path))) {
                paths.push_back(path);
            }
        }
    } else {
        for (const std::string_view type :
             {kMesh, kCube, kSphere, kCylinder, kCone, kCapsule,
              kPlane}) {
            const std::vector<Path> ofType = scene.Prims(type);
            paths.insert(paths.end(), ofType.begin(), ofType.end());
        }
    }
    subdivisionVersion_ =
        scene.AttributeVersion(Path::Root(), kSubdivisionRefineLevel);
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
        // A constant colour (a value, or one row, as a column's) only
        // tints the draws (`SyncedMesh::tint`, or a group's instance
        // colours): the mesh stays, as colours per vertex do not.
        const Tensor* colours = scene.Read(path, kDisplayColor);
        const bool constantColour =
            colours == nullptr || colours->Elements() == 3;
        const bool gpuColors = onGpu(kDisplayColor) && !constantColour;
        key.second[3] = gpuNormals ? 1 : key.second[3];
        key.second[4] = gpuColors ? 1 : key.second[4];
        if (constantColour) {
            key.second[4] = 0;
        }
        // A subdivision surface's level (its own or the stage's) and
        // tags (key slot 11). Its points refine again when they
        // change.
        const std::optional<int> subdivision =
            SubdivisionLevelOf(scene, path);
        if (subdivision) {
            std::uint64_t hash =
                static_cast<std::uint64_t>(*subdivision) + 1;
            for (const std::string_view attribute :
                 {kSubdivisionScheme, kInterpolateBoundary,
                  kFaceVaryingLinearInterpolation,
                  kTriangleSubdivisionRule, kCreaseIndices,
                  kCreaseLengths, kCreaseSharpnesses, kCornerIndices,
                  kCornerSharpnesses, kHoleIndices}) {
                hash = hash * 0x100000001b3ull ^
                       scene.AttributeVersion(path, attribute);
            }
            key.second[11] = hash;
        }
        // A skinned mesh is added again as its binding or its
        // skeleton's joints change (slot 11 too); it is posed on the
        // GPU every sync (`PoseSkins`).
        if (scene.TypeOf(path) == kMesh) {
            const Path skeleton = Mesh(scene, path).GetSkeleton();
            if (!skeleton.IsEmpty()) {
                skinned = true;
                std::uint64_t hash =
                    scene.AttributeVersion(skeleton, kJoints);
                for (const std::string_view attribute :
                     {kSkelSkeleton, kSkelJoints, kSkelJointIndices,
                      kSkelJointWeights, kSkelGeomBindTransform,
                      kSkelSkinningMethod, kSkelBlendShapes,
                      kSkelBlendShapeTargets}) {
                    hash = hash * 0x100000001b3ull ^
                           scene.AttributeVersion(path, attribute);
                }
                key.second[11] =
                    key.second[11] * 0x100000001b3ull ^ (hash + 1);
            }
        }
        // A point cache (sampled points) is blended on the GPU every
        // sync, and added again as its samples change (slot 11 too).
        const bool pointCache = IsPointCache(scene, path);
        if (pointCache) {
            skinned = true;
            std::uint64_t hash = 0;
            for (const TimeSample& sample :
                 scene.SamplesOf(path, kPoints)) {
                hash = hash * 0x100000001b3ull ^
                       reinterpret_cast<std::uintptr_t>(
                           sample.tensor.Data()) ^
                       std::bit_cast<std::uint64_t>(sample.time);
            }
            key.second[11] =
                key.second[11] * 0x100000001b3ull ^ (hash + 1);
        }
        // Triangles written on the GPU too, while their number holds
        // (key slot 2).
        const bool gpuIndices = onGpu(kFaceVertexIndices);
        if (gpuIndices) {
            key.second[2] = static_cast<std::uint64_t>(
                scene.Read(path, kFaceVertexIndices)->Elements());
        }
        // The points and triangles in use, read in place (slot 4),
        // and the temperatures, written each solver step (slot 5);
        // a thermal network's nodes (6) and their temperatures (7).
        const std::array<std::uint64_t, 8> versions{
            pointCache ? 0 : scene.AttributeVersion(path, kPoints),
            gpuNormals ? scene.AttributeVersion(path, kNormals) : 0,
            gpuColors ? scene.AttributeVersion(path, kDisplayColor)
                      : 0,
            gpuIndices
                ? scene.AttributeVersion(path, kFaceVertexIndices)
                : 0,
            scene.AttributeVersion(path, kActiveCounts),
            scene.AttributeVersion(path, kThermalTemperatures),
            scene.AttributeVersion(path, kThermalNodeIds),
            scene.AttributeVersion(path, kThermalNodeTemperatures)};
        const auto it = syncedMeshes_.find(path);
        if (it != syncedMeshes_.end() && it->second.key == key) {
            SyncedMesh& synced = it->second;
            synced.tint = ConstantDisplayColor(Gprim(scene, path));
            if (synced.versions == versions) {
                continue;
            }
            if (synced.handle != 0 &&
                !(subdivision && synced.versions[0] != versions[0])) {
                try {
                    if (synced.versions[3] != versions[3]) {
                        SetMeshIndices(scene, path, synced.handle);
                    }
                    if (synced.versions[4] != versions[4]) {
                        SetMeshActiveCounts(scene, path,
                                            synced.handle);
                    }
                    if (synced.versions[0] != versions[0]) {
                        SetMeshPoints(scene, path, synced.handle);
                    }
                    if (synced.versions[1] != versions[1] ||
                        synced.versions[2] != versions[2]) {
                        SetMeshAttributes(scene, path, synced.handle);
                    }
                    if (synced.versions[5] != versions[5]) {
                        SetMeshTemperatures(scene, path,
                                            synced.handle);
                    }
                    if (synced.versions[6] != versions[6] ||
                        synced.versions[7] != versions[7]) {
                        SetMeshNodes(scene, path, synced.handle,
                                     synced.versions[6] !=
                                         versions[6]);
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
            dropped.push_back(it->second.handle);
        }
        try {
            // A skinned mesh or point cache at rest, which the GPU
            // poses.
            std::optional<SkinBinding> binding =
                scene.TypeOf(path) == kMesh
                    ? SkinBindingOf(scene, path)
                    : std::nullopt;
            const bool deformed = binding.has_value() || pointCache;
            RefinementStencils stencils;
            MeshData data = deformed
                                ? MeshDataOf(scene, path,
                                             {.posed = false,
                                              .stencils = &stencils})
                                : MeshDataOf(scene, path);
            synced.hasColors = !data.colors.empty();
            synced.tint = ConstantDisplayColor(Gprim(scene, path));
            synced.handle = meshes_.AddMesh(std::move(data));
            if (deformed) {
                AddSkin(scene, path, std::move(binding),
                        std::move(stencils), synced);
            }
            // Before the points, which recompute normals from host
            // triangles otherwise.
            if (gpuIndices) {
                SetMeshIndices(scene, path, synced.handle);
            }
            if (scene.Read(path, kActiveCounts) != nullptr) {
                SetMeshActiveCounts(scene, path, synced.handle);
            }
            if (const auto writer = meshWriters_.find(path);
                writer != meshWriters_.end()) {
                meshes_.SetWriter(synced.handle,
                                  writer->second.writer,
                                  writer->second.bounds);
            }
            // A posed mesh's points come from its skin.
            if (onGpu(kPoints) && !deformed) {
                SetMeshPoints(scene, path, synced.handle);
            }
            if (gpuNormals || gpuColors) {
                SetMeshAttributes(scene, path, synced.handle);
            }
            if (scene.Read(path, kThermalTemperatures) != nullptr) {
                SetMeshTemperatures(scene, path, synced.handle);
            }
            if (scene.Read(path, kThermalNodeIds) != nullptr) {
                SetMeshNodes(scene, path, synced.handle, true);
            }
        } catch (const std::invalid_argument& error) {
            if (synced.handle != 0) {
                meshes_.RemoveMesh(synced.handle);
                synced.handle = 0;
            }
            errors.push_back(Describe(path, "mesh", error));
        }
        syncedMeshes_[path] = synced;
        remeshed_.insert(path);
    }
    // Those gone: of all, or of those named.
    const auto drop = [&](std::map<Path, SyncedMesh>::iterator it) {
        if (it->second.handle != 0) {
            dropped.push_back(it->second.handle);
        }
        remeshed_.insert(it->first);
        return syncedMeshes_.erase(it);
    };
    if (only != nullptr) {
        for (const Path& path : *only) {
            const auto it = syncedMeshes_.find(path);
            if (it != syncedMeshes_.end() && !seen.contains(path)) {
                drop(it);
            }
        }
    } else {
        for (auto it = syncedMeshes_.begin();
             it != syncedMeshes_.end();) {
            it = seen.contains(it->first) ? std::next(it) : drop(it);
        }
    }
    meshes_.RemoveMeshes(dropped);
    // A partial sync (`ChangedMeshes`) keeps the skins it did not
    // see.
    skinned_ = only != nullptr ? skinned_ || skinned : skinned;
    posedTimeCode_ = scene.AttributeVersion(Path::Root(), kTimeCode);
    // Every skinned mesh, changed or not, at the time code.
    if (skinned_) {
        PoseSkins(scene, errors);
    }
    if (!errors.empty()) {
        throw std::invalid_argument("Cannot upload " +
                                    errors.front());
    }
}

void RasterDelegate::AddSkin(const Scene& scene, const Path& path,
                             std::optional<SkinBinding> given,
                             RefinementStencils stencils,
                             SyncedMesh& synced) {
    // A point cache alone binds no joints.
    SkinBinding binding = given ? std::move(*given) : SkinBinding{};
    MeshDrawable::Skin skin;
    // Its control points, as the scene holds them: each sample of a
    // point cache, one after another.
    const auto append = [&](const Tensor& points,
                            std::uintptr_t stream) {
        // A device sample is copied in on the device; its bounds are
        // read once.
        const bool device = IsGpu(points.Location().kind);
        const Tensor packed =
            (device ? RequireCompute().Download(points) : points)
                .Contiguous();
        if (packed.Type() != DType::Float32) {
            throw std::invalid_argument(
                "a posed mesh's points must be float32");
        }
        const float* first = packed.As<float>();
        if (device) {
            skin.deviceRest.push_back(points);
            skin.restStream = stream;
        } else {
            skin.restPoints.insert(skin.restPoints.end(), first,
                                   first + packed.Elements());
        }
        Bounds3 box{glm::vec3(std::numeric_limits<float>::max()),
                    glm::vec3(std::numeric_limits<float>::lowest())};
        for (std::int64_t i = 0; i + 2 < packed.Elements(); i += 3) {
            const glm::vec3 p(first[i], first[i + 1], first[i + 2]);
            box.first = glm::min(box.first, p);
            box.second = glm::max(box.second, p);
        }
        synced.sampleBounds.push_back(box);
    };
    synced.sampleBounds.clear();
    const std::span<const TimeSample> samples =
        scene.SamplesOf(path, tokens::kPoints);
    if (IsPointCache(scene, path)) {
        for (const TimeSample& sample : samples) {
            append(sample.tensor, sample.stream);
        }
        skin.restSamples = samples.size();
    } else if (const Tensor* points =
                   scene.Read(path, tokens::kPoints)) {
        append(*points, scene.WriteStream(path, tokens::kPoints));
    }
    synced.deformed = true;
    const std::size_t controls =
        skin.deviceRest.empty()
            ? skin.restPoints.size() / 3 / skin.restSamples
            : static_cast<std::size_t>(
                  skin.deviceRest.front().Elements()) /
                  3;
    const std::size_t perPoint = binding.perPoint;
    if (perPoint == 0 || binding.constant) {
        skin.constantInfluences = true;
        skin.influenceStarts = {0,
                                static_cast<std::uint32_t>(perPoint)};
    } else {
        if (binding.jointIndices.size() != controls * perPoint) {
            throw std::invalid_argument("a skinned mesh needs joint "
                                        "influences for each point");
        }
        for (std::size_t p = 0; p <= controls; ++p) {
            skin.influenceStarts.push_back(
                static_cast<std::uint32_t>(p * perPoint));
        }
    }
    const std::size_t influences =
        skin.constantInfluences ? perPoint : controls * perPoint;
    std::set<std::uint32_t> joints;
    for (std::size_t k = 0; k < influences; ++k) {
        skin.influenceJoints.push_back(
            static_cast<std::uint32_t>(binding.jointIndices[k]));
        skin.influenceWeights.push_back(binding.jointWeights[k]);
        if (binding.jointWeights[k] != 0.0f) {
            joints.insert(skin.influenceJoints.back());
        }
    }
    skin.joints = binding.joints.size();
    skin.dualQuaternion =
        binding.method == SkinningMethod::DualQuaternion;
    // Blend shape offsets by point, each with its sub-shape.
    struct Entry {
        std::uint32_t point;
        std::uint32_t shape;
        glm::vec3 offset;
    };
    std::vector<Entry> entries;
    synced.shapeReach = 0.0f;
    for (std::size_t s = 0; s < binding.subShapes.size(); ++s) {
        const SkinBinding::SubShape& shape = binding.subShapes[s];
        float reach = 0.0f;
        for (std::size_t k = 0; 3 * k + 2 < shape.offsets.size();
             ++k) {
            const std::int64_t point =
                shape.pointIndices.empty()
                    ? static_cast<std::int64_t>(k)
                : k < shape.pointIndices.size()
                    ? shape.pointIndices[k]
                    : -1;
            if (point < 0 ||
                static_cast<std::size_t>(point) >= controls) {
                continue;
            }
            const glm::vec3 offset(shape.offsets[3 * k],
                                   shape.offsets[3 * k + 1],
                                   shape.offsets[3 * k + 2]);
            entries.push_back({static_cast<std::uint32_t>(point),
                               static_cast<std::uint32_t>(s),
                               offset});
            reach = std::max(reach, glm::length(offset));
        }
        synced.shapeReach += reach;
    }
    if (!entries.empty()) {
        std::stable_sort(entries.begin(), entries.end(),
                         [](const Entry& a, const Entry& b) {
                             return a.point < b.point;
                         });
        skin.shapeStarts.assign(controls + 1, 0u);
        for (const Entry& entry : entries) {
            ++skin.shapeStarts[entry.point + 1];
            skin.shapeOffsets.insert(
                skin.shapeOffsets.end(),
                {entry.offset.x, entry.offset.y, entry.offset.z});
            skin.shapeIndices.push_back(entry.shape);
        }
        for (std::size_t p = 0; p < controls; ++p) {
            skin.shapeStarts[p + 1] += skin.shapeStarts[p];
        }
    }
    skin.subShapes = binding.subShapes.size();
    skin.stencils = std::move(stencils);
    synced.skinJoints.assign(joints.begin(), joints.end());
    synced.restBounds = Bounds3{};
    if (controls != 0) {
        synced.restBounds = synced.sampleBounds.front();
        for (const Bounds3& box : synced.sampleBounds) {
            synced.restBounds.first =
                glm::min(synced.restBounds.first, box.first);
            synced.restBounds.second =
                glm::max(synced.restBounds.second, box.second);
        }
    }
    meshes_.SetSkin(synced.handle, std::move(skin));
    synced.skin =
        std::make_shared<const SkinBinding>(std::move(binding));
}

void RasterDelegate::PoseSkins(const Scene& scene,
                               std::vector<std::string>& errors) {
    ERNIE_ZONE("Pose skins");
    const double time = Stage(scene, Path::Root()).GetTimeCode();
    std::map<Path, SkeletonPose> skeletons;
    // World transforms, each prim's found once: a character's meshes
    // share their ancestors.
    std::map<Path, glm::dmat4> worlds;
    const std::function<glm::dmat4(const Path&)> WorldOf =
        [&](const Path& path) -> glm::dmat4 {
        if (path.IsRoot()) {
            return scene.LocalTransform(path);
        }
        if (const auto it = worlds.find(path); it != worlds.end()) {
            return it->second;
        }
        const glm::dmat4 local = scene.LocalTransform(path);
        const glm::dmat4 world = ResetsXformStack(scene, path)
                                     ? local
                                     : WorldOf(path.Parent()) * local;
        worlds.emplace(path, world);
        return world;
    };
    // A 3x4 transform's rows, as skinning.slang reads them.
    const auto rows = [](const glm::mat4& m,
                         std::vector<float>& out) {
        for (int r = 0; r < 3; ++r) {
            out.insert(out.end(),
                       {m[0][r], m[1][r], m[2][r], m[3][r]});
        }
    };
    const auto box = [](const glm::mat4& m, const Bounds3& b,
                        Bounds3& into) {
        for (int corner = 0; corner < 8; ++corner) {
            const glm::vec3 p(
                m * glm::vec4((corner & 1) ? b.second.x : b.first.x,
                              (corner & 2) ? b.second.y : b.first.y,
                              (corner & 4) ? b.second.z : b.first.z,
                              1.0f));
            into.first = glm::min(into.first, p);
            into.second = glm::max(into.second, p);
        }
    };
    for (auto& [path, synced] : syncedMeshes_) {
        if (!synced.deformed || synced.handle == 0) {
            continue;
        }
        try {
            const SkinBinding& binding = *synced.skin;
            SkinPose pose;
            if (!binding.skeleton.IsEmpty()) {
                auto skeleton = skeletons.find(binding.skeleton);
                if (skeleton == skeletons.end()) {
                    skeleton =
                        skeletons
                            .emplace(binding.skeleton,
                                     SkeletonPoseOf(scene,
                                                    binding.skeleton,
                                                    time))
                            .first;
                }
                pose = SkinPoseOf(WorldOf(path), binding,
                                  skeleton->second);
            }
            const glm::mat4 bind = binding.geomBindTransform;
            const glm::mat4 toMesh(pose.skelToMesh);
            MeshDrawable::SkinPose out;
            rows(bind, out.rows);
            rows(toMesh, out.rows);
            for (const glm::mat4& joint : pose.skinning) {
                if (binding.method == SkinningMethod::Linear) {
                    rows(toMesh * joint * bind, out.rows);
                } else {
                    const JointDualQuaternion dq = FactorJoint(joint);
                    out.rows.insert(out.rows.end(),
                                    {dq.real.x, dq.real.y, dq.real.z,
                                     dq.real.w, dq.dual.x, dq.dual.y,
                                     dq.dual.z, dq.dual.w});
                    for (int r = 0; r < 3; ++r) {
                        out.rows.insert(out.rows.end(),
                                        {dq.scale[0][r],
                                         dq.scale[1][r],
                                         dq.scale[2][r], 0.0f});
                    }
                }
            }
            out.subShapeWeights = pose.subShapeWeights;
            // A point cache's two samples either side of the time
            // code.
            Bounds3 restBounds = synced.restBounds;
            if (synced.sampleBounds.size() > 1) {
                if (const std::optional<SampleBracket> bracket =
                        scene.BracketAt(path, tokens::kPoints)) {
                    out.lower =
                        static_cast<std::uint32_t>(bracket->lower);
                    out.upper =
                        static_cast<std::uint32_t>(bracket->upper);
                    out.alpha = static_cast<float>(bracket->alpha);
                    const Bounds3& a =
                        synced.sampleBounds[bracket->lower];
                    const Bounds3& b =
                        synced.sampleBounds[bracket->upper];
                    restBounds = {glm::min(a.first, b.first),
                                  glm::max(a.second, b.second)};
                }
            }
            // Its rest box (widened by its blend shapes) carried by
            // each joint it follows: a box around it in any blend of
            // them.
            const glm::vec3 reach(synced.shapeReach);
            const Bounds3 rest{restBounds.first - reach,
                               restBounds.second + reach};
            out.bounds = {
                glm::vec3(std::numeric_limits<float>::max()),
                glm::vec3(std::numeric_limits<float>::lowest())};
            if (synced.skinJoints.empty()) {
                box(toMesh, rest, out.bounds);
            }
            for (const std::uint32_t j : synced.skinJoints) {
                box(toMesh * pose.skinning[j] * bind, rest,
                    out.bounds);
            }
            meshes_.SetSkinPose(synced.handle, std::move(out));
        } catch (const std::invalid_argument& error) {
            errors.push_back(Describe(path, "skinned mesh", error));
        }
    }
}

bool RasterDelegate::SyncSampledInstancer(const Scene& scene,
                                          const Path& path,
                                          SyncedInstancer& synced) {
    using namespace tokens;
    constexpr std::array<std::string_view, 3> kNames{
        kPositions, kOrientations, kScales};
    constexpr std::array<std::size_t, 3> kWidths{3, 4, 3};
    std::array<std::span<const TimeSample>, 3> samples;
    std::span<const TimeSample> timing;
    for (std::size_t k = 0; k < kNames.size(); ++k) {
        const std::span<const TimeSample> each =
            scene.SamplesOf(path, kNames[k]);
        if (each.size() < 2 || !each.front().isTensor) {
            continue;
        }
        if (!timing.empty() &&
            !std::ranges::equal(timing, each, {}, &TimeSample::time,
                                &TimeSample::time)) {
            return false;  // Arrays sampled at other times: on the
                           // CPU.
        }
        timing = each;
        samples[k] = each;
    }
    if (timing.empty() || GpuAnchorOf(scene, path)) {
        return false;
    }
    std::uint64_t key = 1469598103934665603ull;
    for (const std::span<const TimeSample>& each : samples) {
        for (const TimeSample& sample : each) {
            key = (key ^ reinterpret_cast<std::uintptr_t>(
                             sample.tensor.Data())) *
                      0x100000001b3ull ^
                  std::bit_cast<std::uint64_t>(sample.time);
        }
        key = key * 0x100000001b3ull + each.size();
    }
    if (synced.handle == 0 || synced.samplesKey != key) {
        // A sample's floats on the host: downloaded once when on the
        // device, for its bounds.
        const auto host = [&](const Tensor& tensor) {
            const Tensor packed =
                (IsGpu(tensor.Location().kind)
                     ? RequireCompute().Download(tensor)
                     : tensor)
                    .Contiguous();
            if (packed.Type() != DType::Float32) {
                throw std::invalid_argument(
                    "its sampled arrays must be float32");
            }
            return std::vector<float>(packed.As<float>(),
                                      packed.As<float>() +
                                          packed.Elements());
        };
        MeshDrawable::InstanceSamples out;
        InstanceData data;
        std::array<std::vector<float>*, 3> firsts{
            &data.positions, &data.orientations, &data.scales};
        std::array<std::vector<float>*, 3> hosts{
            &out.positions, &out.orientations, &out.scales};
        std::array<std::vector<Tensor>*, 3> devices{
            &out.devicePositions, &out.deviceOrientations,
            &out.deviceScales};
        std::array<std::size_t*, 3> counts{&out.positionSamples,
                                           &out.orientationSamples,
                                           &out.scaleSamples};
        synced.sampleBounds.clear();
        synced.sampleScale = 1.0f;
        for (std::size_t k = 0; k < kNames.size(); ++k) {
            if (samples[k].empty()) {
                // As it is, on the host.
                if (const std::optional<std::vector<float>> values =
                        FloatAttribute(scene, path, kNames[k],
                                       kWidths[k])) {
                    *firsts[k] = *values;
                }
                continue;
            }
            *counts[k] = samples[k].size();
            const bool device =
                IsGpu(samples[k].front().tensor.Location().kind);
            if (device) {
                out.stream = samples[k].front().stream;
            }
            for (const TimeSample& sample : samples[k]) {
                if (device) {
                    devices[k]->push_back(sample.tensor);
                }
                // Positions' bounds and the largest scale need them
                // on the host; the rest only the first.
                if (k == 1 && device && !firsts[k]->empty()) {
                    continue;
                }
                const std::vector<float> values = host(sample.tensor);
                if (firsts[k]->empty()) {
                    *firsts[k] = values;
                }
                if (!device) {
                    hosts[k]->insert(hosts[k]->end(), values.begin(),
                                     values.end());
                }
                if (k == 0) {
                    Bounds3 box{
                        glm::vec3(std::numeric_limits<float>::max()),
                        glm::vec3(
                            std::numeric_limits<float>::lowest())};
                    for (std::size_t i = 0; i + 2 < values.size();
                         i += 3) {
                        const glm::vec3 p(values[i], values[i + 1],
                                          values[i + 2]);
                        box.first = glm::min(box.first, p);
                        box.second = glm::max(box.second, p);
                    }
                    synced.sampleBounds.push_back(box);
                } else if (k == 2) {
                    for (const float s : values) {
                        synced.sampleScale =
                            std::max(synced.sampleScale, std::abs(s));
                    }
                }
            }
        }
        if (samples[0].empty()) {
            Bounds3 box{
                glm::vec3(std::numeric_limits<float>::max()),
                glm::vec3(std::numeric_limits<float>::lowest())};
            for (std::size_t i = 0; i + 2 < data.positions.size();
                 i += 3) {
                const glm::vec3 p(data.positions[i],
                                  data.positions[i + 1],
                                  data.positions[i + 2]);
                box.first = glm::min(box.first, p);
                box.second = glm::max(box.second, p);
            }
            synced.sampleBounds.push_back(box);
        }
        if (const std::optional<std::vector<float>> colors =
                FloatAttribute(scene, path, kDisplayColor, 3)) {
            data.colors = *colors;
        }
        if (synced.handle == 0) {
            synced.handle = meshes_.AddInstancer(std::move(data));
        } else {
            meshes_.SetInstances(synced.handle, std::move(data));
        }
        meshes_.SetInstanceSamples(synced.handle, std::move(out));
        synced.samplesKey = key;
        synced.key =
            Key{};  // The usual sync starts afresh without samples.
    }
    const SampleBracket bracket = BracketOf(timing, scene.TimeCode());
    const Bounds3& a = synced.sampleBounds[std::min(
        bracket.lower, synced.sampleBounds.size() - 1)];
    const Bounds3& b = synced.sampleBounds[std::min(
        bracket.upper, synced.sampleBounds.size() - 1)];
    meshes_.SetInstanceBlend(
        synced.handle, static_cast<std::uint32_t>(bracket.lower),
        static_cast<std::uint32_t>(bracket.upper),
        static_cast<float>(bracket.alpha),
        {glm::min(a.first, b.first), glm::max(a.second, b.second)},
        synced.sampleScale);
    return true;
}

void RasterDelegate::SyncInstancers(const Scene& scene) {
    ERNIE_ZONE("Sync instancers");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    sampledInstancers_ = false;
    for (const Path& path : scene.Prims(kPointInstancer)) {
        seen.insert(path);
        try {
            if (SyncSampledInstancer(scene, path,
                                     instancers_[path])) {
                sampledInstancers_ = true;
                continue;
            }
        } catch (const std::invalid_argument& error) {
            errors.push_back(Describe(path, "instancer", error));
            continue;
        }
        const Key key = KeyOf(
            scene, path,
            {kPositions, kOrientations, kScales, kDisplayColor});
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
                                       static_cast<std::size_t>(
                                           width))) {
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
                       std::ranges::any_of(
                           std::array{positions,
                                      instancer.GetOrientations(),
                                      instancer.GetScales(),
                                      instancer.GetDisplayColor()},
                           [](const Tensor* tensor) {
                               return tensor != nullptr &&
                                      IsGpu(tensor->Location().kind);
                           })) {
                // Any array on the GPU (such as colours alone) puts
                // them all there.
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
                data.positions = onDevice(positions);
                data.orientations =
                    onDevice(instancer.GetOrientations());
                data.scales = onDevice(instancer.GetScales());
                data.colors = onDevice(instancer.GetDisplayColor());
                data.extent = ExtentOf(scene, path);
                // Copies follow the stream the positions were written
                // on, or else another array on the GPU.
                for (const std::string_view attribute :
                     {kDisplayColor, kScales, kOrientations,
                      kPositions}) {
                    const Tensor* tensor =
                        scene.Read(path, attribute);
                    if (tensor != nullptr &&
                        IsGpu(tensor->Location().kind)) {
                        data.stream =
                            scene.WriteStream(path, attribute);
                    }
                }
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
    // The members' `attribute` as rows of one array (a column), each
    // at its pose's row: one of the group's instance channels.
    const auto columnChannel = [&scene](const std::vector<Member>& members,
                                        std::string_view attribute) {
        const Tensor* whole = nullptr;
        for (const Member& member : members) {
            const std::pair<Tensor, std::int64_t>* row =
                scene.RowOf(member.path, attribute);
            if (row == nullptr || row->second != member.pose.row ||
                (whole != nullptr && row->first.Data() != whole->Data())) {
                return Tensor();
            }
            whole = &row->first;
        }
        return whole != nullptr && IsInstanceChannel(*whole, attribute)
                   ? *whole
                   : Tensor();
    };
    // Prims sharing arrays (one `Scene::Write` of many paths), then
    // each group by its first prim.
    std::map<std::vector<const void*>, std::vector<Member>> byArrays;
    std::vector<std::string> errors;
    for (const Path& path : scene.Prims()) {
        try {
            std::optional<GpuXform> pose = PoseOf(scene, path);
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
                     {kXformOpOrder, kXformExtent, kQuaternionOrder,
                      kDisplayColor, kThermalTemperature, kVisibility}) {
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
            for (const auto& [attribute, channel] : group.Channels()) {
                *channel =
                    composed ? Tensor() : columnChannel(members, attribute);
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

std::optional<GpuXform> RasterDelegate::PoseOf(const Scene& scene,
                                               const Path& path) {
    std::optional<GpuXform> pose = GpuXformOf(scene, path);
    if (pose || Compute() == nullptr) {
        return pose;
    }
    // Rows of a column on the host: posed as GPU arrays are, uploaded
    // whole when written; prims mixing them with other ops stay posed
    // on the CPU.
    try {
        return GpuXformOf(scene, path, true);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

void RasterDelegate::UploadGpuXformGroup(GpuXformGroup& group,
                                         const GpuXform& given,
                                         std::int64_t rows) {
    ComputeInterop& compute = RequireCompute();
    // Rows of a column on the host go to the device whole, each op's
    // array once.
    GpuXform pose = given;
    for (auto& [op, whole] : pose.ops) {
        if (whole.Location().kind == DeviceKind::Cpu) {
            whole = compute.Upload(whole);
            (op == tokens::kTranslate ? pose.translate
             : op == tokens::kOrient  ? pose.orient
             : op == tokens::kScale   ? pose.scale
                                      : whole) = whole;
        }
    }
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
    // Its prims' colours, temperatures and visibility, columns of
    // them, as instance channels.
    const auto onDevice = [&compute](const Tensor& channel) {
        return channel.Empty() || channel.Location().kind != DeviceKind::Cpu
                   ? channel
                   : compute.Upload(channel);
    };
    data.colors = onDevice(group.colors);
    data.temperatures = onDevice(group.temperatures);
    data.visible = onDevice(group.visible);
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

glm::dmat4 RasterDelegate::HostPoseOf(const Scene& scene,
                                      const Path& path) {
    const std::optional<GpuAnchor> anchor = GpuAnchorOf(scene, path);
    if (!anchor) {
        return scene.WorldTransform(path);
    }
    // The anchor's own ops, its GPU rows copied to the host, under
    // its parent's pose (itself perhaps under GPU-posed prims).
    const Path& at = anchor->at;
    ComputeInterop& compute = RequireCompute();
    const GpuOpReader read = [&](std::string_view attribute,
                                 const Tensor& device) {
        return compute.Download(device,
                                scene.WriteStream(at, attribute));
    };
    const glm::dmat4 parent =
        ResetsXformStack(scene, at) || at.Parent().IsRoot()
            ? glm::dmat4(1.0)
            : HostPoseOf(scene, at.Parent());
    return parent * ComposeXformOps(scene, at, &read) * anchor->below;
}

OverlayAnchorPose
RasterDelegate::OverlayAnchorOf(const Scene& scene,
                                const Path& path) const {
    OverlayAnchor& anchor = overlayAnchors_[path];
    if (anchor.scene != scene.Id() ||
        anchor.version != scene.Version()) {
        anchor.scene = scene.Id();
        anchor.version = scene.Version();
        const bool exists = scene.Has(path);
        anchor.gpu = exists ? GpuAnchorOf(scene, path) : std::nullopt;
        anchor.world = exists && !anchor.gpu
                           ? scene.WorldTransform(path)
                           : glm::dmat4(1.0);
        anchor.index = exists ? WorldOf(scene, path) : -1;
    }
    if (anchor.gpu) {
        return {anchor.gpu->frame,
                meshes_.AnchorSlot(anchor.gpu->instancer,
                                   anchor.gpu->row),
                anchor.gpu->below,
                meshes_.AnchorOps(anchor.gpu->instancer),
                anchor.index};
    }
    OverlayAnchorPose pose;
    pose.frame = anchor.world;
    pose.world = anchor.index;
    return pose;
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
    const Scene& scene, std::span<const Change> changes,
    std::span<const ColumnChange> columns) {
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
    // Columns written whole: a group's transform op or constant
    // colours, its rows in the group's order (its first and last prims
    // checked), all of them.
    for (const ColumnChange& column : columns) {
        const std::vector<Path>& paths = *column.paths;
        const auto first = gpuPosed_.find(paths.front());
        const auto last = gpuPosed_.find(paths.back());
        if (first == gpuPosed_.end() || last == gpuPosed_.end() ||
            first->second.group != last->second.group) {
            return false;
        }
        GpuXformGroup& group = *first->second.group;
        if (group.members != paths.size() || first->second.row != 0 ||
            std::size_t{last->second.row} + 1 != paths.size() ||
            scene.AttributeVersion(paths.front(), tokens::kXformOpOrder) >
                syncedVersion_) {
            return false;
        }
        Touched& seen = touched[&group];
        // One of the group's instance channels, taken whole.
        const auto channels = group.Channels();
        const auto channel = std::ranges::find(
            channels, column.attribute,
            &std::pair<std::string_view, Tensor*>::first);
        if (channel != channels.end()) {
            const std::pair<Tensor, std::int64_t>* row =
                scene.RowOf(paths.front(), column.attribute);
            if (group.composed || row == nullptr ||
                row->first.Shape()[0] != group.rows ||
                !IsInstanceChannel(row->first, column.attribute)) {
                return false;
            }
            *channel->second = row->first;
            if (seen.path == nullptr) {
                seen.path = &paths.front();
            }
            seen.rewritten = std::max(seen.rewritten, paths.size());
            continue;
        }
        const auto op = std::ranges::find(group.ops, column.attribute);
        if (column.bits != Dirty::Transform || op == group.ops.end()) {
            return false;
        }
        const auto index =
            static_cast<std::size_t>(op - group.ops.begin());
        const std::pair<Tensor, std::int64_t>* row =
            scene.RowOf(paths.front(), column.attribute);
        if (row == nullptr || index >= seen.arrays.size()) {
            return false;
        }
        if (seen.path != nullptr && seen.arrays[index] != nullptr &&
            seen.arrays[index] != row->first.Data()) {
            return false;
        }
        seen.arrays[index] = row->first.Data();
        if (seen.path == nullptr) {
            seen.path = &paths.front();
        }
        seen.rewritten = std::max(seen.rewritten, paths.size());
    }
    // Every prim of each touched group rewritten, from arrays of the
    // group's rows (checked in full for one prim): its draws stand as
    // they are and only its arrays change.
    std::vector<std::pair<GpuXformGroup*, GpuXform>> uploads;
    try {
        for (const auto& [group, seen] : touched) {
            std::optional<GpuXform> pose = PoseOf(scene, *seen.path);
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

void RasterDelegate::SetMeshWriter(const Path& path,
                                   MeshWriter writer,
                                   std::optional<Bounds3> bounds) {
    const auto synced = syncedMeshes_.find(path);
    const std::uint32_t handle =
        synced != syncedMeshes_.end() ? synced->second.handle : 0;
    if (writer) {
        meshWriters_[path] = {writer, bounds};
    } else {
        meshWriters_.erase(path);
    }
    if (handle != 0) {
        meshes_.SetWriter(handle, std::move(writer), bounds);
    }
}

void RasterDelegate::SetFieldWriter(
    const Path& path, std::array<std::uint32_t, 3> shape,
    FieldWriter writer) {
    if (writer) {
        fieldWriters_.insert(path);
    } else {
        fieldWriters_.erase(path);
    }
    renderer_.SetVolumeFieldWriter(path.String(), shape,
                                   std::move(writer));
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
    } else if (type == kMesh && attribute == kFaceVertexIndices) {
        const auto it = syncedMeshes_.find(path);
        if (it == syncedMeshes_.end() || it->second.handle == 0) {
            throw unsynced();
        }
        view = meshes_.MapIndices(it->second.handle, stream);
    } else if (type == kPointInstancer &&
               (attribute == kPositions ||
                attribute == kOrientations || attribute == kScales ||
                attribute == kDisplayColor)) {
        const auto it = instancers_.find(path);
        if (it == instancers_.end() || it->second.handle == 0) {
            throw unsynced();
        }
        const std::size_t array = attribute == kPositions      ? 0
                                  : attribute == kOrientations ? 1
                                  : attribute == kScales       ? 2
                                                               : 3;
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
    } else if (type == kLaserLight &&
               attribute == kLaserPatternImageArray) {
        // The laser's region of the pattern arrays the shaders read;
        // a frame finds the image there and copies nothing.
        const Tensor* written = scene.Read(path, attribute);
        if (written == nullptr || written->Type() != DType::Float32 ||
            written->Rank() != 2 ||
            !IsGpu(written->Location().kind)) {
            throw std::invalid_argument(
                "Cannot map the pattern image of '" + path.String() +
                "': write it first as a float32 (rows, cols) GPU "
                "array");
        }
        ComputeInterop& compute = RequireCompute();
        const std::vector<std::int64_t> shape = written->Shape();
        const RasterRenderer::PatternArrayRegion region =
            renderer_.EnsurePatternArray(
                path.String(), static_cast<std::uint32_t>(shape[1]),
                static_cast<std::uint32_t>(shape[0]));
        view = compute.View(renderer_.PatternArrays(), DType::Float32,
                            shape, region.offset * sizeof(float));
        // Nothing writes it until the frames already submitted have
        // read it; it starts as the image written.
        compute.WaitForVulkan(stream);
        if (view.Data() != written->Data()) {
            compute.Copy(*written, view, stream);
        }
    } else {
        throw std::invalid_argument(
            "Cannot map " + std::string(attribute) + " of '" +
            path.String() +
            "': only a Mesh's points and GPU triangle indices, a "
            "PointInstancer's "
            "arrays, a Points prim's points, the Terrain's heights, "
            "a Camera's poses and a LaserLight's pattern image map");
    }
    scene.Write(path, attribute, view, stream);
    return view;
}

Tensor RasterDelegate::MapOverlay(Scene& scene,
                                  std::string_view overlay,
                                  std::size_t batch,
                                  std::string_view array,
                                  std::uintptr_t stream) {
    RequireCompute();
    if (scene.FindOverlay(overlay) == nullptr) {
        throw std::invalid_argument("The scene has no overlay '" +
                                    std::string(overlay) + "'");
    }
    Overlay& layer = scene.GetOverlay(overlay);
    Tensor view = renderer_.Overlays().MapBatchArray(layer, batch,
                                                     array, stream);
    layer.ReplaceBatchArray(batch, array, view);
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
    ERNIE_ZONE("Sync points items");
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
        item.temperature = CheckedTemperature(
            path, Gprim(scene, path).GetThermalTemperature());
        item.primId = PrimIdOf(path);
        item.linkGroup = links_.GroupOf(path);
        item.world = WorldOf(scene, path);
        items.push_back(item);
    }
    return items;
}

void RasterDelegate::SyncLinks(const Scene& scene) {
    ERNIE_ZONE("Sync light links");
    const RenderPrims prims = ResolveRenderPrims(scene, {});
    std::vector<Path> geometry;
    for (const auto& [path, synced] : syncedMeshes_) {
        geometry.push_back(path);
    }
    for (const auto& [path, synced] : instancers_) {
        geometry.push_back(path);
    }
    for (const auto& [path, synced] : syncedPoints_) {
        geometry.push_back(path);
    }
    for (const Path& path : {prims.terrain, prims.ground}) {
        if (!path.IsEmpty()) {
            geometry.push_back(path);
        }
    }
    // Volumes too: a light lights a volume and its shadows fall
    // through it only when its links hold it.
    for (const Path& path : scene.Prims(tokens::kVolume)) {
        geometry.push_back(path);
    }
    links_ = LinkGroupsOf(scene, prims.lights, geometry);
}

void RasterDelegate::Sync(const Scene& scene) {
    ERNIE_ZONE("Sync");
    // Columns the scene maps on the compute device live in the arrays
    // drawn from (`DelegateColumns`).
    if (columnStorage_ == nullptr) {
        columnStorage_ = std::make_shared<DelegateColumns>(*this);
    }
    scene.SetColumnStorage(columnStorage_);
    // What changed since the last sync; none for a new scene, which
    // syncs whole.
    std::optional<std::vector<Change>> changed;
    if (scene.Id() != sceneId_) {
        sceneId_ = scene.Id();
        cursor_.version = scene.Version();
    } else {
        // Columns written whole (a simulation's bodies every step)
        // come whole, expanded only when a full sync needs their
        // prims.
        std::vector<ColumnChange> columns;
        const std::vector<Change>& changes =
            changed.emplace(scene.Changes(cursor_, columns));
        // Changes only to cameras and render prims, such as a batch's
        // poses and worlds every step, leave the synced materials,
        // terrain, meshes and draws as they are (lights and settings
        // resolve every frame); a new time code poses skinned meshes.
        const bool posing =
            (skinned_ || sampledInstancers_) &&
            scene.AttributeVersion(Path::Root(), tokens::kTimeCode) !=
                posedTimeCode_;
        const auto view = [&scene](const Change& change) {
            return IsViewChange(scene, change);
        };
        if (!posing && std::ranges::all_of(changes, view) &&
            std::ranges::all_of(
                columns, [&](const ColumnChange& column) {
                    return std::ranges::all_of(
                        *column.paths, [&](const Path& path) {
                            return view({path, column.bits});
                        });
                })) {
            return;
        }
        // So does a simulation step that only rewrites the arrays of
        // prims posed on the GPU: the drawable takes the new arrays.
        if (!posing && RefreshGpuXforms(scene, changes, columns)) {
            renderer_.MarkCastersChanged();
            return;
        }
        if (!columns.empty()) {
            *changed = scene.ExpandColumns(changes, columns);
        }
        // Posing moves meshes the changes do not name.
        if (posing) {
            changed.reset();
        }
    }
    // What casts shadows may have moved: the terrain's horizon over
    // it follows.
    renderer_.MarkCastersChanged();
    // Lasers' pattern arrays the scene no longer holds on the GPU
    // give their regions back; hidden ones keep theirs, where their
    // maps write.
    renderer_.ReleasePatternArrays([&scene](const std::string& key) {
        const Path path(key);
        if (!scene.Has(path)) {
            return false;
        }
        const Tensor* array =
            scene.Read(path, tokens::kLaserPatternImageArray);
        return array != nullptr && IsGpu(array->Location().kind);
    });
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
    // Only the meshes the changes name, unless they are most of them.
    std::optional<std::vector<Path>> changedMeshes;
    if (changed) {
        changedMeshes = ChangedMeshes(scene, *changed);
        if (changedMeshes &&
            2 * changedMeshes->size() > syncedMeshes_.size()) {
            changedMeshes.reset();
        }
    }
    attempt([&] {
        SyncMeshes(scene, changedMeshes ? &*changedMeshes : nullptr);
    });
    // GPU-posed prims first: instancers under them compose under
    // their instances. With none, only a changed prim's own transform
    // written as GPU arrays makes some, so the scene is not searched.
    const bool gpuXforms =
        !changed || !gpuPosed_.empty() ||
        std::ranges::any_of(*changed, [this, &scene](const Change& change) {
            if (!Any(change.bits, Dirty::Transform | Dirty::Added) ||
                !scene.Has(change.path)) {
                return false;
            }
            try {
                const std::optional<GpuXform> pose =
                    PoseOf(scene, change.path);
                return pose && !pose->ops.empty();
            } catch (const std::invalid_argument&) {
                return true;
            }
        });
    if (gpuXforms) {
        attempt([&] { SyncGpuXforms(scene); });
    }
    attempt([&] { SyncInstancers(scene); });
    attempt([&] { SyncPoints(scene); });
    // Light linking's groups of what is drawn, before the draws that
    // carry them.
    const bool linked = !links_.groups.empty();
    attempt([&] { SyncLinks(scene); });
    attempt([&] { points_.SetItems(PointsItems(scene)); });
    // The draws, so the meshes' bounds are known as soon as the scene
    // syncs: only those of the meshes changed, when the changes reach
    // no further and no light links group the draws.
    std::optional<std::set<Path>> touched;
    if (changed && !errors.empty()) {
        changed.reset();
    }
    if (changed && !linked && links_.groups.empty()) {
        touched = TouchedMeshes(scene, *changed);
        // Most of them, as a simulation writes every body: all at
        // once.
        if (touched && 2 * (touched->size() + remeshed_.size()) >
                           syncedMeshes_.size()) {
            touched.reset();
        }
    }
    attempt([&] {
        meshes_.SetItems(Items(scene, touched ? &*touched : nullptr));
    });
    syncedVersion_ = scene.Version();
    if (!errors.empty()) {
        std::string message = errors.front();
        for (std::size_t i = 1; i < errors.size(); ++i) {
            message += "; " + errors[i];
        }
        throw std::invalid_argument(message);
    }
}

std::optional<std::set<Path>>
RasterDelegate::TouchedMeshes(const Scene& scene,
                              std::span<const Change> changes) const {
    using namespace tokens;
    // Under a prim posed on the GPU, a mesh's draw follows its group
    // (`GpuAnchorOf`), which other prims' arrays change.
    if (!gpuPosed_.empty()) {
        return std::nullopt;
    }
    std::set<Path> touched;
    for (const Change& change : changes) {
        if (IsViewChange(scene, change)) {
            continue;
        }
        const Path& path = change.path;
        const std::string_view type =
            scene.Has(path) ? scene.TypeOf(path) : std::string_view();
        // A mesh's own draw, whatever changed, or one gone.
        if (IsMeshType(type) || syncedMeshes_.contains(path) ||
            remeshed_.contains(path)) {
            touched.insert(path);
            continue;
        }
        // Other prims change no mesh's draw but through the meshes'
        // own changes: what an ancestor's transform, visibility or
        // world (`AttributeMeta::inherited`), or a material's edits
        // (`AttributeMeta::targetDirty`), reach comes stamped. Only
        // an instancer's prototypes decide which meshes draw alone.
        if (type == kPointInstancer &&
            Any(change.bits,
                Dirty::Topology | Dirty::Added | Dirty::Removed)) {
            return std::nullopt;
        }
    }
    return touched;
}

std::vector<MeshItem>
RasterDelegate::Items(const Scene& scene,
                      const std::set<Path>* touched) {
    ERNIE_ZONE("Sync items");
    // Instancers' prototypes, and the meshes of each, which may be a
    // mesh or a subtree of them (as USD's are), by their nearest
    // prototype ancestor: found in full, or for the meshes touched.
    // The meshes whose draws are built again, besides all.
    const std::set<Path>* const rebuilt[] = {touched, &remeshed_};
    const auto prototypeOf = [this](const Path& mesh) {
        for (Path at = mesh; !at.IsEmpty(); at = at.Parent()) {
            if (prototypes_.contains(at)) {
                return at;
            }
            if (at == Path::Root()) {
                break;
            }
        }
        return Path();
    };
    if (touched == nullptr) {
        prototypes_.clear();
        prototypeMeshes_.clear();
        inPrototypes_.clear();
        for (const auto& [path, synced] : instancers_) {
            for (const Path& prototype :
                 PointInstancer(scene, path).GetPrototypes()) {
                prototypes_.insert(prototype);
            }
        }
        for (const auto& [path, synced] : syncedMeshes_) {
            if (const Path prototype = prototypeOf(path);
                !prototype.IsEmpty()) {
                prototypeMeshes_[prototype].push_back(path);
                inPrototypes_.insert(path);
            }
        }
    } else {
        for (const std::set<Path>* paths : rebuilt) {
            for (const Path& path : *paths) {
                const Path prototype = syncedMeshes_.contains(path)
                                           ? prototypeOf(path)
                                           : Path();
                if (prototype.IsEmpty() ==
                    !inPrototypes_.contains(path)) {
                    continue;
                }
                if (prototype.IsEmpty()) {
                    inPrototypes_.erase(path);
                    for (auto& [at, meshes] : prototypeMeshes_) {
                        std::erase(meshes, path);
                    }
                } else {
                    inPrototypes_.insert(path);
                    std::vector<Path>& meshes =
                        prototypeMeshes_[prototype];
                    meshes.insert(std::lower_bound(meshes.begin(),
                                                   meshes.end(),
                                                   path),
                                  path);
                }
            }
        }
    }
    const auto shade = [this, &scene](const Path& mesh,
                                      const SyncedMesh& synced,
                                      MeshItem& item) {
        item.material = MaterialOf(scene, mesh);
        const bool bound =
            item.material != RasterRenderer::kDefaultMaterial;
        item.tint = bound ? glm::vec3(1.0f) : synced.tint;
        item.vertexColors = !bound;
        item.temperature = CheckedTemperature(
            mesh, Gprim(scene, mesh).GetThermalTemperature());
    };

    // Each mesh's own draw, none when hidden or in a prototype.
    const auto build = [&](const Path& path, const SyncedMesh& synced)
        -> std::optional<MeshItem> {
        if (synced.handle == 0 || inPrototypes_.contains(path)) {
            return std::nullopt;
        }
        // Under a group whose visibility is a column, hidden on the GPU
        // (`GpuXformGroup::visible`), so drawn while what is above its
        // prim shows it.
        const std::optional<GpuAnchor> anchor = GpuAnchorOf(scene, path);
        const auto posed =
            anchor ? gpuPosed_.find(anchor->at) : gpuPosed_.end();
        const bool channelled = posed != gpuPosed_.end() &&
                                !posed->second.group->visible.Empty();
        if (channelled ? !anchor->at.Parent().IsEmpty() &&
                             !scene.IsVisible(anchor->at.Parent())
                       : !scene.IsVisible(path)) {
            return std::nullopt;
        }
        // Drawn by `spheres_` (`UpdatePrims`).
        if (IsAdaptiveSphere(scene, path)) {
            return std::nullopt;
        }
        MeshItem item;
        item.mesh = synced.handle;
        if (anchor) {
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
        item.linkGroup = links_.GroupOf(path);
        item.world = WorldOf(scene, path);
        shade(path, synced, item);
        return item;
    };
    if (touched == nullptr) {
        meshItems_.clear();
        for (const auto& [path, synced] : syncedMeshes_) {
            meshItems_.emplace(path, build(path, synced));
        }
    } else {
        for (const std::set<Path>* paths : rebuilt) {
            for (const Path& path : *paths) {
                const auto synced = syncedMeshes_.find(path);
                if (synced == syncedMeshes_.end()) {
                    meshItems_.erase(path);
                } else {
                    meshItems_[path] = build(path, synced->second);
                }
            }
        }
    }
    remeshed_.clear();
    std::vector<MeshItem> items;
    items.reserve(meshItems_.size());
    for (const auto& [path, item] : meshItems_) {
        if (item) {
            items.push_back(*item);
        }
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
            const auto meshes = prototypeMeshes_.find(targets[k]);
            if (meshes == prototypeMeshes_.end() ||
                lists[k].empty()) {
                continue;
            }
            // From the prototype's parent's frame, which instances
            // place, down to each of its meshes.
            const glm::dmat4 above = glm::inverse(
                scene.WorldTransform(targets[k].Parent()));
            for (const Path& meshPath : meshes->second) {
                const auto mesh = syncedMeshes_.find(meshPath);
                if (mesh == syncedMeshes_.end() ||
                    mesh->second.handle == 0 ||
                    !scene.IsVisible(meshPath)) {
                    continue;
                }
                MeshItem item;
                item.mesh = mesh->second.handle;
                item.transform = world;
                item.prototype =
                    glm::mat4(above * scene.WorldTransform(meshPath));
                item.instancer = synced.handle;
                item.primId = PrimIdOf(path);
                item.linkGroup = links_.GroupOf(path);
                // Instances sorted by world, so each view draws its
                // world's as one range.
                std::vector<std::uint32_t>& instances = lists[k];
                std::stable_sort(
                    instances.begin(), instances.end(),
                    [&worlds](std::uint32_t a, std::uint32_t b) {
                        return worlds[a] < worlds[b];
                    });
                const bool oneWorld = worlds[instances.front()] ==
                                      worlds[instances.back()];
                item.world = worlds[instances.front()];
                if (!oneWorld) {
                    for (const std::uint32_t instance : instances) {
                        item.instanceWorlds.push_back(
                            worlds[instance]);
                    }
                }
                item.instances = instances;
                shade(meshPath, mesh->second, item);
                items.push_back(std::move(item));
            }
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
    // The view runs the chain of a product looking through its camera
    // (`ViewChainProduct`), else the first product's
    // (`DefaultFilterChain` without products). It holds every var and
    // the AOV slots of what that chain writes.
    const Path chain =
        ViewChainProduct(scene, settings, prims.camera);
    const Path chosen[] = {chain};
    std::vector<SensorFilterStep> filters =
        ResolveRenderProducts(
            scene, chain.IsEmpty() ? std::span<const Path>() : chosen,
            settings)
            .front()
            .filters;
    // A stage (`SetViewStage`): the chain up to that filter, its last
    // domain what the vars show.
    if (!chain.IsEmpty() && !viewStage_.IsEmpty()) {
        std::vector<Path> staged =
            RenderProduct(scene, chain).GetFilters();
        if (const auto stage = std::ranges::find(staged, viewStage_);
            stage != staged.end()) {
            staged.erase(stage + 1, staged.end());
            filters = SensorFilterStepsOf(scene, staged);
        }
    }
    viewAovs_ =
        filters.empty()
            ? std::vector<std::string>{}
            : AovOutputsOf(filters, renderer_.SensorFilters());
    // The viewport's frame, which the viewer may expose itself.
    viewportFrame_ = true;
    try {
        UpdatePrims(scene, prims, extent,
                    NextFrame(scene, prims.settings, prims.settings),
                    prims.settings, filters, {}, {},
                    ViewHeldVars(viewAovs_.size(), viewOptIn_));
    } catch (...) {
        viewportFrame_ = false;
        throw;
    }
    viewportFrame_ = false;
    ReadPatternArrays();
    SettleMotion();
}

void RasterDelegate::ReadPatternArrays() {
    const std::vector<RasterRenderer::PatternArrayWrite>& writes =
        renderer_.PatternArrayWrites();
    if (writes.empty()) {
        return;
    }
    // Into their regions on the writers' streams, after the frames
    // already submitted; the frame waits for them (`SubmitWaits`).
    ComputeInterop& compute = RequireCompute();
    const std::shared_ptr<GpuBuffer>& arrays =
        renderer_.PatternArrays();
    for (const RasterRenderer::PatternArrayWrite& write : writes) {
        const Tensor region =
            compute.View(arrays, DType::Float32, write.array.Shape(),
                         write.offset);
        if (write.array.Data() == region.Data()) {
            // Written in place (`MapArray`): only the writes to wait
            // for.
            compute.SignalVulkan(write.stream);
        } else {
            compute.CopyToBuffer(write.array, *arrays, write.offset,
                                 write.stream);
        }
    }
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
                                        std::uint32_t views,
                                        double time) {
    // Passes the frame no longer has keep no state.
    std::erase_if(sensorStates_, [&](const auto& entry) {
        return entry.first.first == key &&
               entry.first.second >= renderer_.SensorPasses();
    });
    std::erase_if(sensorAgcs_, [&](const auto& entry) {
        return entry.first.first == key &&
               entry.first.second >= renderer_.SensorPasses();
    });
    for (std::size_t pass = 0; pass < renderer_.SensorPasses();
         ++pass) {
        PrepareSensorState(key, pass, extent, views, time);
    }
}

void RasterDelegate::PrepareSensorState(const Path& key,
                                        std::size_t pass,
                                        VkExtent2D extent,
                                        std::uint32_t views,
                                        double time) {
    // The automatic gain control's range, its halves taking turns:
    // empty (zeros) until a frame records one.
    SensorAgc& agc = sensorAgcs_[{key, pass}];
    if (agc.buffer == nullptr) {
        agc.buffer = std::make_unique<GpuBuffer>(
            context_, 4 * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Upload);
        const std::array<std::uint32_t, 4> empty{};
        agc.buffer->Write(std::as_bytes(std::span(empty)));
    }
    renderer_.SetSensorAgc(pass, agc.buffer.get(), agc.flip);
    agc.flip ^= 1u;
    const std::uint32_t slots = renderer_.SensorStateSlots(pass);
    if (slots == 0) {
        renderer_.SetSensorState(pass, nullptr, nullptr, false, 0.0);
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
    // The seconds since the history's frame: none without one, or as
    // time stands still.
    const double interval =
        state.history ? std::max(time - state.time, 0.0) : 0.0;
    state.time = time;
    renderer_.SetSensorState(pass, state.images[1 - state.next].get(),
                             state.images[state.next].get(),
                             state.history, interval);
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
    // Textures that finished loading take their placeholders' places,
    // and a planet's heights file once read its old heights'; a frame
    // that waits for all of them waits here, and for a planet's
    // height pages as it prepares.
    const bool waits = WaitsForTextures(scene, prims.settings);
    FinishTextureLoads(waits);
    FinishPlanetLoad(waits);
    planet_.SetWaitForPages(waits);
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
    // The viewport's own exposure stops, and its meter.
    if (viewportFrame_) {
        frame.exposure =
            static_cast<float>(std::exp2(viewportStops_));
        frame.meter = true;
    }
    for (std::size_t i = 0;
         i < frame.lights.size() && i < anchoredLights.size(); ++i) {
        frame.lights[i].posedOnGpu = anchoredLights[i];
    }
    // Light linking: each light's masks over the groups `Sync` gave
    // the geometry.
    for (std::size_t i = 0;
         i < frame.lights.size() && i < prims_.lights.size(); ++i) {
        frame.lights[i].lightMask =
            links_.LightMask(prims_.lights[i]);
        frame.lights[i].shadowMask =
            links_.ShadowMask(prims_.lights[i]);
    }
    for (VolumeParams& volume : frame.volumes) {
        volume.linkGroup = links_.GroupOf(Path(volume.prim));
        // Grids written on a compute device are copied from it.
        for (const auto* field :
             {&volume.density, &volume.temperature,
              &volume.emission}) {
            if (*field && !(*field)->buffer.Empty() &&
                (*field)->buffer.Location().kind != DeviceKind::Cpu) {
                renderer_.SetVolumeComputeInterop(&RequireCompute());
            }
        }
    }
    // The lights' poses at the last frame of `stateKey`, for motion
    // blur: none for a light's first frame or one posed on the GPU.
    {
        std::map<Path, LightParams::Pose>& last =
            lastLightPoses_[stateKey];
        std::map<Path, LightParams::Pose> current;
        for (std::size_t i = 0;
             i < frame.lights.size() && i < prims_.lights.size();
             ++i) {
            LightParams& light = frame.lights[i];
            const auto found = last.find(prims_.lights[i]);
            if (found != last.end() && !light.posedOnGpu) {
                light.previous = found->second;
            }
            current.emplace(prims_.lights[i],
                            LightParams::Pose{light.position,
                                              light.direction,
                                              light.tangent});
        }
        last = std::move(current);
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
    terrain.linkGroup = prims_.terrain.IsEmpty()
                            ? 0u
                            : links_.GroupOf(prims_.terrain);
    terrain.temperature =
        prims_.terrain.IsEmpty()
            ? 0.0f
            : CheckedTemperature(prims_.terrain,
                                 Gprim(scene, prims_.terrain)
                                     .GetThermalTemperature());
    terrain_.SetParams(terrain);
    planet_.SetParams(terrain);
    // Adaptive spheres, those under prims posed on the GPU at their
    // poses copied back (the LOD is chosen on the CPU). Invalid radii
    // are their meshes' errors.
    {
        std::vector<AdaptiveSphere> spheres;
        for (const Path& path : scene.Prims(tokens::kSphere)) {
            if (inPrototypes_.contains(path) ||
                !scene.IsVisible(path) ||
                !IsAdaptiveSphere(scene, path)) {
                continue;
            }
            const double radius = Sphere(scene, path).GetRadius();
            if (!(radius > 0.0) || !std::isfinite(radius)) {
                continue;
            }
            AdaptiveSphere sphere =
                AdaptiveSphere::Of(HostPoseOf(scene, path), radius);
            if (!(glm::all(glm::greaterThan(sphere.radii,
                                            glm::dvec3(0.0))) &&
                  std::isfinite(glm::length(sphere.radii)))) {
                continue;
            }
            sphere.material = MaterialOf(scene, path);
            sphere.primId = PrimIdOf(path);
            sphere.linkGroup = links_.GroupOf(path);
            sphere.temperature = CheckedTemperature(
                path, Gprim(scene, path).GetThermalTemperature());
            sphere.world = WorldOf(scene, path);
            spheres.push_back(sphere);
        }
        spheres_.SetSpheres(spheres, terrain.lod);
    }
    ground_.SetEnabled(!prims_.ground.IsEmpty());
    if (!prims_.ground.IsEmpty()) {
        GroundParams ground = GroundParamsOf(scene, prims_.ground);
        ground.material = MaterialOf(scene, prims_.ground);
        ground.primId = PrimIdOf(prims_.ground);
        ground.linkGroup = links_.GroupOf(prims_.ground);
        ground.temperature = CheckedTemperature(
            prims_.ground, GroundPlane(scene, prims_.ground)
                               .GetThermalTemperature());
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
    RasterDrawable* drawables[] = {&terrain_, &planet_, &spheres_,
                                   &meshes_,  &ground_, &points_};
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
    // Overlay batches with GPU arrays copy from the compute device.
    for (const Overlay* overlay : drawn) {
        if (overlay != nullptr &&
            std::ranges::any_of(
                overlay->Batches(), [](const OverlayBatch& batch) {
                    return IsGpu(batch.points.Location().kind);
                })) {
            RequireCompute();
            break;
        }
    }
    const auto views = static_cast<std::uint32_t>(
        std::max<std::size_t>(frame.poses.size(), 1));
    // Overlays posed under prims, this frame's poses.
    if (scene.Id() != overlayAnchorsScene_) {
        overlayAnchors_.clear();
        overlayAnchorsScene_ = scene.Id();
    }
    renderer_.Overlays().SetAnchorResolver(
        [this, &scene](const Path& path) {
            return OverlayAnchorOf(scene, path);
        },
        &meshes_);
    // Once every drawable has named its prims' materials.
    SyncTemperatureAxes(scene);
    renderer_.Update(frame, drawables, drawn);
    renderer_.Overlays().SetAnchorResolver({}, &meshes_);
    PrepareSensorState(stateKey, extent, views, frame.time);
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

std::optional<OverlayHit>
RasterDelegate::PickOverlay(const Scene& scene, std::uint32_t x,
                            std::uint32_t y) {
    if (viewTargets_ == nullptr) {
        return std::nullopt;
    }
    const std::optional<OverlayPass::Picked> picked =
        renderer_.PickOverlay(*viewTargets_, x, y);
    if (!picked) {
        return std::nullopt;
    }
    // The overlay by name, as the scene holds it.
    for (const std::string& name : scene.OverlayNames()) {
        if (scene.FindOverlay(name) == picked->overlay) {
            return OverlayHit{name, picked->batch, picked->item};
        }
    }
    return std::nullopt;
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
        viewTargetsOptIn_ != viewOptIn_) {
        if (extent.width == 0 || extent.height == 0) {
            BeginRendering(commands, target);
            vkCmdEndRendering(commands);
            return;
        }
        vkDeviceWaitIdle(context_.Device());
        const std::vector<std::string_view> vars =
            ViewHeldVars(viewAovs_.size(), viewOptIn_);
        viewTargets_ = std::make_unique<RenderTargets>(
            context_, extent, renderer_.Bands().count, false, 1, vars,
            viewAovs_, renderer_.Faces());
        viewTargetsOptIn_ = viewOptIn_;
    }
    const RasterDrawable* drawables[] = {
        &terrain_, &planet_, &spheres_, &meshes_, &ground_, &points_};
    // The frame's passes under one zone, as the profiler's tree shows
    // them.
    ERNIE_GPU_ZONE(context_, commands, "Rendering");
    renderer_.Record(commands, *viewTargets_, drawables, &target,
                     var);
}

std::shared_ptr<ProductFrame> RasterDelegate::AcquireFrame(
    const Path& product, VkExtent2D extent, int bands,
    std::uint32_t views, bool batched,
    std::span<const std::string_view> vars,
    std::span<const std::string> aovs, std::uint32_t faces,
    bool minimal, bool inPlace) {
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
    // The vars the product's frames hold as linear buffers too: those
    // mapped to linear GPU tensors before, which the sensor can write
    // so (`RenderTargets::LinearVars`).
    std::set<std::string, std::less<>>& linearVars =
        linearVars_[product];
    const auto linearize = [&](ProductFrame& frame) {
        if (minimal || compute_ == nullptr) {
            return;
        }
        frame.linearVars = &linearVars;
        // The entries (vars, then AOV slots) the targets hold whose
        // vars were mapped so, by any of their names.
        std::uint32_t heldEntries = 0;
        for (const std::string_view name : held) {
            if (const std::optional<std::size_t> entry =
                    RenderTargets::LinearEntry(name, aovNames)) {
                heldEntries |= 1u << *entry;
            }
        }
        // None with the product's outputs not in place: each map
        // copies.
        std::uint32_t wanted = 0;
        for (const std::string& name : linearVars) {
            if (const std::optional<std::size_t> entry =
                    RenderTargets::LinearEntry(name, aovNames);
                entry && inPlace) {
                wanted |= (1u << *entry) & heldEntries;
            }
        }
        const std::vector<RenderTargets*> targets =
            TargetsOf(*frame.images);
        const RenderTargets::LinearVars* current =
            targets.front()->GetLinearVars();
        if ((current != nullptr ? current->held : 0u) == wanted) {
            return;
        }
        // Earlier steps may still write the buffers it drops: rare
        // (a var's first linear map), so wait for everything.
        if (current != nullptr) {
            context_.WaitForSubmissions();
        }
        std::shared_ptr<RenderTargets::LinearVars> linear =
            wanted == 0 ? nullptr
                        : RenderTargets::MakeLinearVars(
                              context_, extent, views, bands, wanted);
        for (RenderTargets* chunk : targets) {
            chunk->SetLinearVars(linear);
        }
    };
    for (const std::shared_ptr<ProductFrame>& frame : frames) {
        if (frame.use_count() == 1 && frame->arrayMaps == 0) {
            linearize(*frame);
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
    linearize(*frame);
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
    // A frame that waits for its textures waits for the tiles of the
    // virtual ones it samples too: frames rendered beforehand, as the
    // metering one is, each serving every tile the last asked for,
    // until one asks for none; not again until the scene changes.
    if (!settling_ && !metering_) {
        const bool waits = WaitsForTextures(scene, base.settings);
        if (waits) {
            Sync(scene);
            FinishTextureLoads(true);
        }
        if (waits && renderer_.HasVirtualTextures() &&
            settledVersion_ !=
                std::pair(scene.Id(), scene.Version())) {
            ERNIE_ZONE("Settle virtual textures");
            settling_ = true;
            try {
                for (int pass = 0; pass < kSettlingPasses; ++pass) {
                    StepAsync(scene, products, settings).Fetch();
                    context_.WaitForSubmissions();
                    if (!renderer_.SettleVirtualTextures()) {
                        break;
                    }
                }
            } catch (...) {
                settling_ = false;
                throw;
            }
            settling_ = false;
            settledVersion_ = std::pair(scene.Id(), scene.Version());
        }
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
                const std::vector<std::string>& chainAov =
                    chainAovs[c];
                if (std::find(chainAov.begin(), chainAov.end(),
                              name.substr(kAovPrefix.size())) ==
                    chainAov.end()) {
                    throw missing();
                }
                const auto aovSlot =
                    std::find(aovs.begin(), aovs.end(),
                              name.substr(kAovPrefix.size()));
                held = kAovSlotNames[static_cast<std::size_t>(
                    aovSlot - aovs.begin())];
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
            vars, aovs, faces, minimal != MinimalMode::Off,
            scene.TypeOf(spec.product) != tokens::kRenderProduct ||
                RenderProduct(scene, spec.product)
                    .GetOutputsInPlace());
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
        ReadPatternArrays();
        if (gpuPoses != nullptr) {
            // Once no submission reads the pose buffer (`UpdatePrims`
            // waits), on the writer's stream; the submission waits
            // for it (`SubmitWaits`).
            const std::uintptr_t stream =
                scene.WriteStream(cameras.front(), tokens::kPoses);
            ComputeInterop& interop = RequireCompute();
            if (gpuPoses->Data() ==
                interop
                    .View(renderer_.EnsurePoseBuffer(1),
                          DType::Float32, {1})
                    .Data()) {
                // Written in place into the pose buffer (`MapArray`):
                // only the writes to wait for.
                interop.SignalVulkan(stream);
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
        const RasterDrawable* drawables[] = {&terrain_, &planet_,
                                             &spheres_, &meshes_,
                                             &ground_,  &points_};
        // Every view of the product renders in this one submission,
        // sharing the shadow passes.
        ERNIE_ZONE_TEXT(spec.product.String());
        submission = context_.SubmitAsync(
            [&](VkCommandBuffer commands) {
                ERNIE_GPU_ZONE(context_, commands, "Rendering");
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
