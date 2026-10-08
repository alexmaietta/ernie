#include "ernie/raster/delegate.h"

#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/product_frame.h"
#include "ernie/util/profile.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

/// The vars the view holds for a chain writing `aovs` AOVs: every var (empty) without AOVs, else
/// every var and the first `aovs` AOV slots.
std::vector<std::string_view> ViewHeldVars(std::size_t aovs) {
    if (aovs == 0) {
        return {};
    }
    std::vector<std::string_view> vars(kRenderVarNames.begin(), kRenderVarNames.end());
    vars.insert(vars.end(), kAovSlotNames.begin(),
                kAovSlotNames.begin() + static_cast<std::ptrdiff_t>(aovs));
    return vars;
}

float Number(const Scene& scene, const Path& path, std::string_view attribute, float fallback) {
    return static_cast<float>(GetNumber(scene, path, attribute, fallback));
}

std::invalid_argument UploadError(const Path& path, std::string_view attribute,
                                  const std::string& why) {
    return std::invalid_argument("Cannot upload " + std::string(attribute) + " of '" +
                                 path.String() + "': " + why);
}

/// `tensor` as a float32 raster with the grid in the prim's attributes.
RasterView RasterOf(const Scene& scene, const Path& path, const Tensor& tensor,
                    std::string_view origin, std::string_view cellSize, std::string_view noData) {
    if (tensor.Type() != DType::Float32 || tensor.Rank() != 2 ||
        tensor.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument("expected a float32 (rows, cols) CPU tensor");
    }
    RasterView view;
    view.data = tensor.As<float>();
    view.height = static_cast<std::size_t>(tensor.Shape()[0]);
    view.width = static_cast<std::size_t>(tensor.Shape()[1]);
    const glm::dvec2 corner = GetVec2(scene, path, origin, {0.0, 0.0});
    view.originX = corner.x;
    view.originY = corner.y;
    view.cellSize = GetNumber(scene, path, cellSize, 1.0);
    if (scene.Get(path, noData) != nullptr) {
        view.noData = static_cast<float>(GetNumber(scene, path, noData, 0.0));
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

/// A CPU array of `components`-vectors as floats, shaped (n, components) or flat.
std::vector<float> FloatsOf(const Tensor& tensor, std::size_t components, std::string_view what) {
    const bool shaped =
        tensor.Rank() == 2 && tensor.Shape()[1] == static_cast<std::int64_t>(components);
    if (tensor.Location().kind != DeviceKind::Cpu || !(tensor.Rank() == 1 || shaped) ||
        static_cast<std::size_t>(tensor.Elements()) % components != 0) {
        throw std::invalid_argument(std::string(what) + " must be a CPU array of " +
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
    throw std::invalid_argument(std::string(what) + " must be float32 or float64");
}

/// The prim's attribute as an array: a written tensor or, for small arrays, a float array value.
std::optional<std::vector<float>> FloatAttribute(const Scene& scene, const Path& path,
                                                 std::string_view attribute,
                                                 std::size_t components) {
    if (const Tensor* tensor = scene.Read(path, attribute)) {
        return FloatsOf(*tensor, components, attribute);
    }
    return std::nullopt;
}

/// Targets of a relationship holding one path or a list of them.
std::vector<Path> TargetsOf(const Scene& scene, const Path& path, std::string_view attribute) {
    const Value* value = scene.Get(path, attribute);
    if (value == nullptr) {
        return {};
    }
    if (const auto* list = value->Get<std::vector<Path>>()) {
        return *list;
    }
    if (const auto* single = value->Get<Path>()) {
        return {*single};
    }
    return {};
}

/// The constant colour of `primvars:displayColor`: a colour value or a one-row array.
std::optional<glm::vec3> ConstantDisplayColor(const Scene& scene, const Path& mesh) {
    if (scene.Get(mesh, tokens::kDisplayColor) != nullptr) {
        return glm::vec3(GetVec3(scene, mesh, tokens::kDisplayColor, glm::dvec3(0.5)));
    }
    if (const Tensor* tensor = scene.Read(mesh, tokens::kDisplayColor)) {
        if (tensor->Elements() == 3 && tensor->Location().kind == DeviceKind::Cpu) {
            const std::vector<float> color = FloatsOf(*tensor, 3, tokens::kDisplayColor);
            return glm::vec3(color[0], color[1], color[2]);
        }
    }
    return std::nullopt;
}

std::string Describe(const Path& path, std::string_view what, const std::exception& error) {
    return "'" + path.String() + "' " + std::string(what) + ": " + error.what();
}

/// `shape` averaged over each of `bands` (eight samples per band) times `scale`.
template <typename Shape>
std::vector<float> BinShape(const Shape& shape, const SpectralBands& bands, double scale) {
    std::vector<float> values(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        double sum = 0.0;
        for (int k = 0; k < 8; ++k) {
            sum += shape(bands.Lower(i) + (k + 0.5) * bands.Width() / 8.0);
        }
        values[static_cast<std::size_t>(i)] = static_cast<float>(scale * sum / 8.0);
    }
    return values;
}

}  // namespace

GroundParams GroundParamsOf(const Scene& scene, const Path& ground) {
    using namespace tokens;
    GroundParams params;
    params.height = Number(scene, ground, kGroundHeight, params.height);
    params.checker = GetBool(scene, ground, kChecker, params.checker);
    params.checkerSize = Number(scene, ground, kCheckerSize, params.checkerSize);
    if (!(params.checkerSize > 0.0f) || !std::isfinite(params.checkerSize)) {
        throw std::invalid_argument("GroundPlane '" + ground.String() +
                                    "' needs a positive checker size");
    }
    params.checkerColors[0] =
        glm::vec3(GetVec3(scene, ground, kCheckerColor0, glm::dvec3(params.checkerColors[0])));
    params.checkerColors[1] =
        glm::vec3(GetVec3(scene, ground, kCheckerColor1, glm::dvec3(params.checkerColors[1])));
    return params;
}

TerrainParams TerrainParamsOf(const Scene& scene, const RenderPrims& prims) {
    using namespace tokens;
    TerrainParams params;
    params.exaggeration = Number(scene, prims.terrain, kExaggeration, 1.0f);
    const Path& settings = prims.settings;
    params.colorMode = ColorModeOf(GetToken(scene, settings, kColorMode, kShadedColormap));
    LodSettings& lod = params.lod;
    lod.enabled = GetBool(scene, settings, kLodEnable, lod.enabled);
    lod.detail = Number(scene, settings, kLodDetail, lod.detail);
    lod.morphStart = Number(scene, settings, kLodMorphStart, lod.morphStart);
    lod.freeze = GetBool(scene, settings, kLodFreeze, lod.freeze);
    lod.showLevels = GetBool(scene, settings, kLodShowLevels, lod.showLevels);
    return params;
}

RasterDelegate::RasterDelegate(const VulkanContext& context, VkFormat presentFormat,
                               BrdfRegistry brdfs)
    : context_(context), renderer_(context, presentFormat, std::move(brdfs)), terrain_(renderer_),
      planet_(renderer_), meshes_(renderer_), ground_(renderer_), points_(renderer_) {}

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
         {terrain_.Bounds(), planet_.Bounds(), meshes_.Bounds(), points_.Bounds()}) {
        if (!drawable) {
            continue;
        }
        bounds = bounds ? Bounds3{glm::min(bounds->first, drawable->first),
                                  glm::max(bounds->second, drawable->second)}
                        : *drawable;
    }
    // The drawables' bounds are relative to the render origin.
    if (bounds) {
        const glm::vec3 origin(origin_);
        bounds = Bounds3{bounds->first + origin, bounds->second + origin};
    }
    return bounds;
}

void RasterDelegate::AddSensorFilter(SensorFilterModel model) {
    renderer_.AddSensorFilter(std::move(model));
}

const BrdfRegistry& RasterDelegate::Brdfs() const {
    return renderer_.Brdfs();
}

namespace {

/// The render targets of a frame's images: one set, or one per chunk of a batch's views.
std::vector<RenderTargets*> TargetsOf(FrameImages& images) {
    std::vector<RenderTargets*> targets;
    if (auto* chunked = dynamic_cast<ChunkedFrameImages*>(&images)) {
        for (const std::unique_ptr<FrameImages>& chunk : chunked->Chunks()) {
            targets.push_back(static_cast<RenderTargets*>(chunk.get()));
        }
    } else {
        targets.push_back(static_cast<RenderTargets*>(&images));
    }
    return targets;
}

}  // namespace

RenderStats RasterDelegate::Stats() const {
    RenderStats stats;
    stats.gpuBytes = renderer_.GpuBytes() + terrain_.GpuBytes() + planet_.GpuBytes() +
                     (viewTargets_ != nullptr ? viewTargets_->GpuBytes() : 0);
    for (const auto& [product, frames] : frames_) {
        for (const std::shared_ptr<ProductFrame>& frame : frames) {
            for (const RenderTargets* targets : TargetsOf(*frame->images)) {
                stats.gpuBytes += targets->GpuBytes();
            }
        }
    }
    stats.triangles = terrain_.TriangleCount() + planet_.TriangleCount() + meshes_.TriangleCount();
    stats.counters.emplace_back(raster_stats::kTerrainPatches,
                                terrain_.PatchCount() + planet_.PatchCount());
    stats.counters.emplace_back(raster_stats::kTerrainLodLevels,
                                std::max(terrain_.Lod().Levels(), planet_.Lod().Levels()));
    stats.counters.emplace_back(raster_stats::kMeshDraws, meshes_.DrawCount(0));
    stats.counters.emplace_back(raster_stats::kMeshDrawCalls, meshes_.DrawCalls(0));
    return stats;
}

RasterDelegate::Key
RasterDelegate::KeyOf(const Scene& scene, const Path& path,
                      std::initializer_list<std::string_view> attributes) const {
    Key key{path, {}};
    key.second[0] = sceneId_;
    std::size_t i = 1;
    for (const std::string_view attribute : attributes) {
        key.second.at(i++) = scene.AttributeVersion(path, attribute);
    }
    return key;
}

std::optional<std::uint32_t> RasterDelegate::MaterialId(const Path& path) const {
    const auto it = materials_.find(path);
    return it != materials_.end() ? std::optional(it->second.id) : std::nullopt;
}

std::uint32_t RasterDelegate::MaterialOf(const Scene& scene, const Path& prim) const {
    return MaterialId(BoundMaterial(scene, prim)).value_or(RasterRenderer::kDefaultMaterial);
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

void RasterDelegate::SyncMaterial(const Scene& scene, const Path& path, SyncedMaterial& material) {
    using namespace tokens;
    renderer_.SetMaterial(material.id, MaterialSettingsOf(scene, path, renderer_.Brdfs()));
    RasterMaterial& gpu = renderer_.Material(material.id);

    const Key albedoKey =
        KeyOf(scene, path, {kAlbedoMap, kAlbedoMapOrigin, kAlbedoMapCellSize, kAlbedoMapNoData});
    if (material.albedoMap != albedoKey) {
        material.albedoMap = albedoKey;
        try {
            if (const Tensor* tensor = scene.Read(path, kAlbedoMap)) {
                const Tensor albedo = tensor->Contiguous();
                gpu.SetAlbedoMap(RasterOf(scene, path, albedo, kAlbedoMapOrigin, kAlbedoMapCellSize,
                                          kAlbedoMapNoData));
            } else if (gpu.HasAlbedoMap()) {
                gpu.ClearAlbedoMap();
            }
        } catch (const std::invalid_argument& error) {
            throw UploadError(path, kAlbedoMap, error.what());
        }
    }
    for (const bool normal : {false, true}) {
        const std::string_view attribute = normal ? kDetailNormal : kDetailAlbedo;
        std::optional<Key>& cached = normal ? material.detailNormal : material.detailAlbedo;
        const Key key = KeyOf(scene, path, {attribute});
        if (cached == key) {
            continue;
        }
        cached = key;
        try {
            std::optional<Image> image;
            if (const Tensor* tensor = scene.Read(path, attribute)) {
                image =
                    ImageOfTensor(*tensor, normal ? ImageEncoding::Linear : ImageEncoding::Srgb);
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

void RasterDelegate::SyncHeightfield(const Scene& scene, const Path& terrain) {
    ERNIE_ZONE("Sync heightfield");
    using namespace tokens;
    if (terrain.IsEmpty() || scene.Read(terrain, kHeights) == nullptr) {
        if (heightfield_) {
            terrain_.ClearHeightfield();
            planet_.ClearHeights();
            heightfield_.reset();
        }
        return;
    }
    const Key key = KeyOf(scene, terrain, {kHeights, kOrigin, kCellSize, kNoData, kPlanetRadius});
    if (heightfield_ == key) {
        return;
    }
    heightfield_ = key;
    if (scene.Get(terrain, kPlanetRadius) != nullptr) {
        try {
            if (terrain_.HasHeightfield()) {
                terrain_.ClearHeightfield();
            }
            const Tensor heights = scene.Read(terrain, kHeights)->Contiguous();
            if (heights.Type() != DType::Float32 || heights.Rank() != 3 ||
                heights.Shape()[0] != CubeSphereLod::kFaces ||
                heights.Location().kind != DeviceKind::Cpu) {
                throw std::invalid_argument(
                    "a planet's heights must be a float32 (6, n + 1, n + 1) CPU tensor");
            }
            std::array<RasterView, CubeSphereLod::kFaces> faces;
            const auto rows = static_cast<std::size_t>(heights.Shape()[1]);
            const auto cols = static_cast<std::size_t>(heights.Shape()[2]);
            for (std::size_t face = 0; face < faces.size(); ++face) {
                faces[face].data = heights.As<float>() + face * rows * cols;
                faces[face].width = cols;
                faces[face].height = rows;
                if (scene.Get(terrain, kNoData) != nullptr) {
                    faces[face].noData =
                        static_cast<float>(GetNumber(scene, terrain, kNoData, 0.0));
                }
            }
            planet_.SetHeights(GetNumber(scene, terrain, kPlanetRadius, 0.0), faces);
        } catch (const std::invalid_argument& error) {
            throw UploadError(terrain, kHeights, error.what());
        }
        return;
    }
    if (planet_.HasHeights()) {
        planet_.ClearHeights();
    }
    // Heights written on a GPU stay there: the terrain copies them device to device and rebuilds
    // what depends on the cells that changed on the GPU.
    const Tensor* written = scene.Read(terrain, kHeights);
    if (IsGpu(written->Location().kind)) {
        try {
            if (!terrainOnGpu_) {
                terrain_.SetComputeInterop(&RequireCompute());
                terrainOnGpu_ = true;
            }
            DeviceHeightfield field;
            field.heights = *written;
            const glm::dvec2 corner = GetVec2(scene, terrain, kOrigin, {0.0, 0.0});
            field.originX = corner.x;
            field.originY = corner.y;
            field.cellSize = GetNumber(scene, terrain, kCellSize, 1.0);
            if (scene.Get(terrain, kNoData) != nullptr) {
                field.noData = static_cast<float>(GetNumber(scene, terrain, kNoData, 0.0));
            }
            if (const Value* value = scene.Get(terrain, kHeightsDirty)) {
                // A vec4, or four numbers.
                glm::vec4 dirty(-1.0f);
                if (const auto* vector = value->Get<glm::vec4>()) {
                    dirty = *vector;
                } else if (const auto* list = value->Get<std::vector<float>>();
                           list != nullptr && list->size() == 4) {
                    dirty = {(*list)[0], (*list)[1], (*list)[2], (*list)[3]};
                }
                if (glm::any(glm::lessThan(dirty, glm::vec4(0.0f)))) {
                    throw std::invalid_argument(std::string(kHeightsDirty) +
                                                " must be four non-negative numbers");
                }
                field.dirty = std::array<std::uint32_t, 4>{
                    static_cast<std::uint32_t>(dirty.x), static_cast<std::uint32_t>(dirty.y),
                    static_cast<std::uint32_t>(dirty.z), static_cast<std::uint32_t>(dirty.w)};
            }
            field.stream = scene.WriteStream(terrain, kHeights);
            terrain_.SetHeightfield(field);
        } catch (const std::invalid_argument& error) {
            throw UploadError(terrain, kHeights, error.what());
        }
        return;
    }
    try {
        const Tensor heights = scene.Read(terrain, kHeights)->Contiguous();
        terrain_.SetHeightfield(RasterOf(scene, terrain, heights, kOrigin, kCellSize, kNoData));
    } catch (const std::invalid_argument& error) {
        throw UploadError(terrain, kHeights, error.what());
    }
}

void RasterDelegate::SyncSky(const Scene& scene, const RenderPrims& prims) {
    ERNIE_ZONE("Sync sky");
    using namespace tokens;
    const std::optional<Key> map =
        prims.dome.IsEmpty() ? std::nullopt
                             : std::optional(KeyOf(scene, prims.dome, {kLightTexture}));
    if (map != domeMap_) {
        domeMap_ = map;
        try {
            const std::optional<Image> image =
                prims.dome.IsEmpty() ? std::nullopt : DomeMapOf(scene, prims.dome);
            renderer_.SetDomeMap(image ? &*image : nullptr);
        } catch (const std::invalid_argument& error) {
            renderer_.SetDomeMap(nullptr);
            throw UploadError(prims.dome, kLightTexture, error.what());
        }
    }
    const std::optional<Key> stars =
        prims.stars.IsEmpty()
            ? std::nullopt
            : std::optional(KeyOf(scene, prims.stars,
                                  {kStarsRightAscension, kStarsDeclination, kStarsDirection,
                                   kStarsMagnitude, kStarsColorIndex, kStarsTemperature,
                                   kStarsMagnitudeLimit, kStarsZeroPoint, kExposure}));
    if (stars != stars_) {
        stars_ = stars;
        try {
            renderer_.SetStars(prims.stars.IsEmpty() ? std::vector<Star>()
                                                     : StarsOf(scene, prims.stars));
        } catch (const std::invalid_argument& error) {
            renderer_.SetStars({});
            throw UploadError(prims.stars, kStarsMagnitude, error.what());
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
        throw std::invalid_argument("GPU arrays need compute interop, which is unavailable: " + why);
    }
    if (!meshesOnGpu_) {
        meshes_.SetComputeInterop(compute);
        points_.SetComputeInterop(compute);
        meshesOnGpu_ = true;
    }
    return *compute;
}

void RasterDelegate::SetMeshPoints(const Scene& scene, const Path& path, std::uint32_t handle) {
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

void RasterDelegate::SetMeshAttributes(const Scene& scene, const Path& path,
                                       std::uint32_t handle) {
    const Tensor* normals = scene.Read(path, tokens::kNormals);
    if (normals != nullptr && IsGpu(normals->Location().kind)) {
        RequireCompute();
        meshes_.SetNormals(handle, *normals, scene.WriteStream(path, tokens::kNormals));
    }
    const Tensor* colors = scene.Read(path, tokens::kDisplayColor);
    if (colors != nullptr && IsGpu(colors->Location().kind)) {
        RequireCompute();
        meshes_.SetColors(handle, *colors, scene.WriteStream(path, tokens::kDisplayColor));
    }
}

void RasterDelegate::SyncMeshes(const Scene& scene) {
    ERNIE_ZONE("Sync meshes");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    std::vector<Path> paths;
    for (const std::string_view type :
         {kMesh, kCube, kSphere, kCylinder, kCone, kCapsule, kPlane}) {
        const std::vector<Path> ofType = scene.Prims(type);
        paths.insert(paths.end(), ofType.begin(), ofType.end());
    }
    for (const Path& path : paths) {
        seen.insert(path);
        Key key = KeyOf(scene, path,
                        {kFaceVertexCounts, kFaceVertexIndices, kNormals, kDisplayColor, kSize,
                         kGeomRadius, kGeomHeight, kGeomWidth, kGeomLength, kAxis});
        // Normals and colours written on the GPU update in place like points, rather than
        // adding the mesh again (key slots 3 and 4).
        const auto onGpu = [&](std::string_view attribute) {
            const Tensor* tensor = scene.Read(path, attribute);
            return tensor != nullptr && IsGpu(tensor->Location().kind);
        };
        const bool gpuNormals = onGpu(kNormals);
        const bool gpuColors = onGpu(kDisplayColor);
        key.second[3] = gpuNormals ? 1 : key.second[3];
        key.second[4] = gpuColors ? 1 : key.second[4];
        const std::array<std::uint64_t, 3> versions{
            scene.AttributeVersion(path, kPoints),
            gpuNormals ? scene.AttributeVersion(path, kNormals) : 0,
            gpuColors ? scene.AttributeVersion(path, kDisplayColor) : 0};
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
                    if (synced.versions[1] != versions[1] || synced.versions[2] != versions[2]) {
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
            synced.tint = ConstantDisplayColor(scene, path).value_or(glm::vec3(0.5f));
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
    for (auto it = syncedMeshes_.begin(); it != syncedMeshes_.end();) {
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
        throw std::invalid_argument("Cannot upload " + errors.front());
    }
}

void RasterDelegate::SyncInstancers(const Scene& scene) {
    ERNIE_ZONE("Sync instancers");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    for (const Path& path : scene.Prims(kPointInstancer)) {
        seen.insert(path);
        const Key key = KeyOf(scene, path, {kPositions, kOrientations, kScales});
        auto [it, added] = instancers_.try_emplace(path);
        SyncedInstancer& synced = it->second;
        if (!added && synced.key == key) {
            continue;
        }
        synced.key = key;
        try {
            const Tensor* positions = scene.Read(path, kPositions);
            if (positions != nullptr && IsGpu(positions->Location().kind)) {
                ComputeInterop& compute = RequireCompute();
                const auto onDevice = [&](std::string_view attribute) {
                    const Tensor* tensor = scene.Read(path, attribute);
                    if (tensor == nullptr) {
                        return Tensor();
                    }
                    return IsGpu(tensor->Location().kind) ? *tensor : compute.Upload(*tensor);
                };
                DeviceInstanceData data;
                data.positions = *positions;
                data.orientations = onDevice(kOrientations);
                data.scales = onDevice(kScales);
                data.extent = ExtentOf(scene, path);
                data.stream = scene.WriteStream(path, kPositions);
                if (synced.handle == 0) {
                    synced.handle = meshes_.AddInstancer({});
                }
                meshes_.SetInstances(synced.handle, data);
            } else {
                InstanceData data = InstanceDataOf(scene, path);
                if (synced.handle == 0) {
                    synced.handle = meshes_.AddInstancer(std::move(data));
                } else {
                    meshes_.SetInstances(synced.handle, std::move(data));
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
        throw std::invalid_argument("Cannot upload " + errors.front());
    }
}

Tensor RasterDelegate::MapArray(Scene& scene, const Path& path, std::string_view attribute,
                                std::uintptr_t stream) {
    using namespace tokens;
    RequireCompute();
    Sync(scene);
    if (!scene.Has(path)) {
        throw std::invalid_argument("Cannot map " + std::string(attribute) + " of '" +
                                    path.String() + "', which does not exist");
    }
    const std::string_view type = scene.TypeOf(path);
    const auto unsynced = [&] {
        return std::invalid_argument("Cannot map " + std::string(attribute) + " of '" +
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
               (attribute == kPositions || attribute == kOrientations || attribute == kScales)) {
        const auto it = instancers_.find(path);
        if (it == instancers_.end() || it->second.handle == 0) {
            throw unsynced();
        }
        const std::size_t array = attribute == kPositions ? 0 : attribute == kOrientations ? 1 : 2;
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
    } else {
        throw std::invalid_argument("Cannot map " + std::string(attribute) + " of '" +
                                    path.String() + "': only a Mesh's points, a PointInstancer's "
                                    "arrays, a Points prim's points and the Terrain's heights map");
    }
    scene.Write(path, attribute, view, stream);
    return view;
}

void RasterDelegate::SyncPoints(const Scene& scene) {
    ERNIE_ZONE("Sync points");
    using namespace tokens;
    std::set<Path> seen;
    std::vector<std::string> errors;
    for (const Path& path : scene.Prims(kGeomPoints)) {
        seen.insert(path);
        const Key key = KeyOf(scene, path, {kPoints, kWidths, kDisplayColor, kExtent});
        auto [it, added] = syncedPoints_.try_emplace(path);
        SyncedPoints& synced = it->second;
        if (!added && synced.key == key) {
            continue;
        }
        synced.key = key;
        try {
            // A width for all from a value or a one-element array, else one per point.
            const auto widthsOf = [&](float& width) -> const Tensor* {
                width = static_cast<float>(GetNumber(scene, path, kWidths, 1.0));
                const Tensor* tensor = scene.Read(path, kWidths);
                if (tensor != nullptr && tensor->Elements() == 1 &&
                    tensor->Location().kind == DeviceKind::Cpu) {
                    width = FloatsOf(*tensor, 1, kWidths).front();
                    return nullptr;
                }
                return tensor;
            };
            const Tensor* points = scene.Read(path, kPoints);
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
                data.widths = onDevice(widthsOf(data.width));
                const Tensor* colors = scene.Read(path, kDisplayColor);
                if (colors != nullptr && colors->Elements() == points->Elements()) {
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
                data.points = FloatAttribute(scene, path, kPoints, 3).value_or(std::vector<float>());
                if (const Tensor* widths = widthsOf(data.width)) {
                    data.widths = FloatsOf(*widths, 1, kWidths);
                }
                if (const Tensor* colors = scene.Read(path, kDisplayColor);
                    colors != nullptr &&
                    static_cast<std::size_t>(colors->Elements()) == data.points.size() &&
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
    for (auto it = syncedPoints_.begin(); it != syncedPoints_.end();) {
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
        throw std::invalid_argument("Cannot upload " + errors.front());
    }
}

std::vector<PointsItem> RasterDelegate::PointsItems(const Scene& scene) const {
    std::vector<PointsItem> items;
    for (const auto& [path, synced] : syncedPoints_) {
        if (synced.handle == 0 || !scene.IsVisible(path)) {
            continue;
        }
        PointsItem item;
        item.points = synced.handle;
        item.transform = scene.WorldTransform(path);
        item.material = MaterialOf(scene, path);
        item.tint = ConstantDisplayColor(scene, path).value_or(glm::vec3(0.5f));
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
        // Changes only to cameras and render prims, such as a batch's poses and worlds every
        // step, leave the synced materials, terrain, meshes and draws as they are (lights and
        // settings resolve every frame).
        const auto view = [&scene](const Change& change) {
            if (Any(change.bits, Dirty::Removed) || !scene.Has(change.path)) {
                return false;
            }
            const std::string_view type = scene.TypeOf(change.path);
            return type == tokens::kCamera || type == tokens::kRenderSettings ||
                   type == tokens::kRenderProduct || type == tokens::kRenderVar;
        };
        if (std::all_of(changes.begin(), changes.end(), view)) {
            return;
        }
    }
    std::vector<std::string> errors;
    const auto attempt = [&errors](auto&& step) {
        try {
            step();
        } catch (const std::invalid_argument& error) {
            errors.emplace_back(error.what());
        }
    };
    attempt([&] { SyncMaterials(scene); });
    attempt([&] { SyncHeightfield(scene, ResolveRenderPrims(scene, {}).terrain); });
    attempt([&] { SyncMeshes(scene); });
    attempt([&] { SyncInstancers(scene); });
    attempt([&] { SyncPoints(scene); });
    attempt([&] { points_.SetItems(PointsItems(scene)); });
    // The draws, so the meshes' bounds are known as soon as the scene syncs.
    attempt([&] { meshes_.SetItems(Items(scene)); });
    if (!errors.empty()) {
        std::string message = errors.front();
        for (std::size_t i = 1; i < errors.size(); ++i) {
            message += "; " + errors[i];
        }
        throw std::invalid_argument(message);
    }
}

std::vector<MeshItem> RasterDelegate::Items(const Scene& scene) const {
    ERNIE_ZONE("Sync items");
    std::set<Path> prototypes;
    for (const auto& [path, synced] : instancers_) {
        for (const Path& prototype : TargetsOf(scene, path, tokens::kPrototypes)) {
            prototypes.insert(prototype);
        }
    }
    const auto isPrototype = [&prototypes](const Path& path) {
        return std::any_of(prototypes.begin(), prototypes.end(),
                           [&path](const Path& prototype) { return path.HasPrefix(prototype); });
    };
    const auto shade = [this, &scene](const Path& mesh, const SyncedMesh& synced, MeshItem& item) {
        item.material = MaterialOf(scene, mesh);
        const bool bound = item.material != RasterRenderer::kDefaultMaterial;
        item.tint = bound ? glm::vec3(1.0f) : synced.tint;
        item.vertexColors = !bound;
    };

    std::vector<MeshItem> items;
    for (const auto& [path, synced] : syncedMeshes_) {
        if (synced.handle == 0 || isPrototype(path) || !scene.IsVisible(path)) {
            continue;
        }
        MeshItem item;
        item.mesh = synced.handle;
        item.transform = scene.WorldTransform(path);
        item.primId = PrimIdOf(path);
        item.world = WorldOf(scene, path);
        shade(path, synced, item);
        items.push_back(std::move(item));
    }
    for (const auto& [path, synced] : instancers_) {
        if (synced.handle == 0 || !scene.IsVisible(path)) {
            continue;
        }
        const std::vector<Path> targets = TargetsOf(scene, path, tokens::kPrototypes);
        std::vector<std::vector<std::uint32_t>> lists =
            InstancesByPrototype(scene, path, targets.size());
        const Tensor* positions = scene.Read(path, tokens::kPositions);
        const std::vector<std::int32_t> worlds = InstanceWorldsOf(
            scene, path,
            positions != nullptr ? static_cast<std::size_t>(positions->Elements()) / 3 : 0);
        const glm::dmat4 world = scene.WorldTransform(path);
        for (std::size_t k = 0; k < targets.size(); ++k) {
            const auto mesh = syncedMeshes_.find(targets[k]);
            if (mesh == syncedMeshes_.end() || mesh->second.handle == 0 || lists[k].empty()) {
                continue;
            }
            MeshItem item;
            item.mesh = mesh->second.handle;
            item.transform = world;
            item.prototype = glm::mat4(scene.LocalTransform(targets[k]));
            item.instancer = synced.handle;
            item.primId = PrimIdOf(path);
            // Instances sorted by world, so each view draws its world's as one range.
            std::vector<std::uint32_t>& instances = lists[k];
            std::stable_sort(
                instances.begin(), instances.end(),
                [&worlds](std::uint32_t a, std::uint32_t b) { return worlds[a] < worlds[b]; });
            const bool oneWorld = worlds[instances.front()] == worlds[instances.back()];
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

void RasterDelegate::Update(const Scene& scene, const Path& settings, VkExtent2D extent) {
    // The caller records and submits the frame itself, so nothing may still be running.
    context_.WaitForSubmissions();
    BeginFrame();
    const RenderPrims prims = ResolveRenderPrims(scene, settings);
    // The view runs the first product's chain (`DefaultFilterChain` without products), holding
    // every var and the AOV slots of what that chain writes.
    const std::vector<SensorFilterStep> filters =
        ResolveRenderProducts(scene, {}, settings).front().filters;
    viewAovs_ = filters.empty() ? std::vector<std::string>{}
                                : AovOutputsOf(filters, renderer_.SensorFilters());
    UpdatePrims(scene, prims, extent, NextFrame(scene, prims.settings, prims.settings),
                prims.settings, filters, {}, {}, ViewHeldVars(viewAovs_.size()));
    SettleMotion();
}

void RasterDelegate::SettleMotion() {
    // The frame is prepared: what moved in it stands still unless written or mapped again
    // before the next, so writes between frames count as the next one's motion.
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

void RasterDelegate::PrepareSensorState(const Path& key, VkExtent2D extent, std::uint32_t views) {
    // Passes the frame no longer has keep no state.
    std::erase_if(sensorStates_, [&](const auto& entry) {
        return entry.first.first == key && entry.first.second >= renderer_.SensorPasses();
    });
    for (std::size_t pass = 0; pass < renderer_.SensorPasses(); ++pass) {
        PrepareSensorState(key, pass, extent, views);
    }
}

void RasterDelegate::PrepareSensorState(const Path& key, std::size_t pass, VkExtent2D extent,
                                        std::uint32_t views) {
    const std::uint32_t slots = renderer_.SensorStateSlots(pass);
    if (slots == 0) {
        renderer_.SetSensorState(pass, nullptr, nullptr, false);
        sensorStates_.erase({key, pass});
        return;
    }
    const std::uint32_t layers = views * slots;
    if (layers > context_.Limits().maxImageArrayLayers) {
        throw std::invalid_argument("Temporal sensor filters keep " + std::to_string(slots) +
                                    " state slots per view, too many for " +
                                    std::to_string(views) + " views on this device");
    }
    SensorState& state = sensorStates_[{key, pass}];
    const std::string chain = renderer_.Plan(pass).Key();
    const GpuImage* existing = state.images[0].get();
    if (existing == nullptr || existing->Desc().width != extent.width ||
        existing->Desc().height != extent.height || existing->Desc().layers != layers ||
        state.chain != chain) {
        // A new history: the images and the chain they hold the state of. Earlier frames may
        // still use the old ones (the viewer's included, which it submits itself): rare, so wait.
        if (existing != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        for (std::unique_ptr<GpuImage>& image : state.images) {
            image = std::make_unique<GpuImage>(
                context_, ImageDesc{extent.width, extent.height, layers, 1,
                                    VK_FORMAT_R32G32B32A32_SFLOAT,
                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false,
                                    true});
        }
        state.next = 0;
        state.history = false;
        state.chain = chain;
    }
    renderer_.SetSensorState(pass, state.images[1 - state.next].get(),
                             state.images[state.next].get(), state.history);
    // The next frame reads what this one writes. Frames run on the device in submission order,
    // and the sensor pass's barrier orders this frame's writes before the next one's reads.
    state.next = 1 - state.next;
    state.history = true;
}

void RasterDelegate::UpdatePrims(const Scene& scene, const RenderPrims& prims, VkExtent2D extent,
                                 FrameId id, const Path& stateKey,
                                 std::span<const SensorFilterStep> filters,
                                 std::vector<glm::dmat4> poses, std::vector<std::int32_t> worlds,
                                 std::vector<std::string_view> vars,
                                 std::optional<glm::dmat4> posesParent,
                                 const std::optional<std::string>& overlays,
                                 std::vector<FrameParams::SensorChain> chains) {
    ERNIE_ZONE("UpdatePrims");
    Sync(scene);
    prims_ = prims;
    SyncSky(scene, prims_);
    FrameParams frame = [&] {
        ERNIE_ZONE("FrameParamsOf");
        return FrameParamsOf(scene, prims_, extent.width, extent.height);
    }();
    frame.filters.assign(filters.begin(), filters.end());
    // One chain needs no chains of its own: `filters` and `vars` run as it (its filters, which
    // are a var's own when every var has the same).
    if (chains.size() == 1) {
        frame.filters = chains.front().filters;
    } else if (chains.size() > 1) {
        frame.chains = std::move(chains);
    }
    frame.poses = std::move(poses);
    frame.posesOnGpu = posesParent.has_value();
    frame.posesParent = posesParent.value_or(glm::dmat4(1.0));
    // The views' poses at the last frame of `stateKey`, for motion blur: none for the first frame
    // or another view count, which hold the camera still.
    {
        std::vector<glm::dmat4> current =
            frame.poses.empty() ? std::vector<glm::dmat4>{frame.camera.transform} : frame.poses;
        std::vector<glm::dmat4>& last = lastPoses_[stateKey];
        if (last.size() == current.size() && !frame.posesOnGpu) {
            frame.previousPoses = std::move(last);
        }
        last = std::move(current);
    }
    // A floating origin: the GPU's coordinates are relative to a point near the camera, which
    // moves only when the camera has travelled `kRebaseDistance` from it, so what is relative to
    // it is rebuilt rarely while float precision stays a fraction of a millimetre near the camera.
    const glm::dvec3 eye(frame.poses.empty() ? frame.camera.transform[3] : frame.poses.front()[3]);
    if (glm::length(eye - origin_) > kRebaseDistance) {
        origin_ = eye;
    }
    frame.origin = origin_;
    frame.vars = std::move(vars);
    // Without a batch's worlds, the single view sees its camera's world.
    frame.worlds = worlds.empty() && !prims_.camera.IsEmpty()
                       ? std::vector<std::int32_t>{WorldOf(scene, prims_.camera)}
                       : std::move(worlds);
    frame.frame = id.frame;
    frame.stream = id.stream;
    TerrainParams terrain = TerrainParamsOf(scene, prims_);
    terrain.material = MaterialOf(scene, prims_.terrain);
    terrain.primId = prims_.terrain.IsEmpty() ? 0u : PrimIdOf(prims_.terrain);
    terrain_.SetParams(terrain);
    planet_.SetParams(terrain);
    ground_.SetEnabled(!prims_.ground.IsEmpty());
    if (!prims_.ground.IsEmpty()) {
        GroundParams ground = GroundParamsOf(scene, prims_.ground);
        ground.material = MaterialOf(scene, prims_.ground);
        ground.primId = PrimIdOf(prims_.ground);
        ground_.SetParams(ground);
    }
    // The highlighted prims and everything under them, as the prim ids the frame draws.
    std::vector<std::uint32_t> highlighted;
    for (const auto& [path, primId] : primIds_) {
        if (std::any_of(highlight_.begin(), highlight_.end(),
                        [&path](const Path& prefix) { return path.HasPrefix(prefix); })) {
            highlighted.push_back(primId);
        }
    }
    renderer_.SetHighlight(highlighted, highlightColour_);
    RasterDrawable* drawables[] = {&terrain_, &planet_, &meshes_, &ground_, &points_};
    if (overlays) {
        frame.overlays = *overlays;
    }
    std::vector<const Overlay*> drawn;
    if (!frame.overlays.empty()) {
        for (const std::string& name : SelectOverlays(frame.overlays, scene.OverlayNames())) {
            drawn.push_back(scene.FindOverlay(name));
        }
    }
    const auto views = static_cast<std::uint32_t>(std::max<std::size_t>(frame.poses.size(), 1));
    renderer_.Update(frame, drawables, drawn);
    PrepareSensorState(stateKey, extent, views);
}

std::uint32_t RasterDelegate::PrimIdOf(const Path& path) const {
    const auto [it, added] =
        primIds_.try_emplace(path, static_cast<std::uint32_t>(primPaths_.size() + 1));
    if (added) {
        primPaths_.push_back(path);
    }
    return it->second;
}

Path RasterDelegate::PrimPath(std::uint32_t id) const {
    return id >= 1 && id <= primPaths_.size() ? primPaths_[id - 1] : Path();
}

Path RasterDelegate::Pick(std::uint32_t x, std::uint32_t y) {
    GpuImage* ids = viewTargets_ != nullptr ? viewTargets_->Var(render_vars::kPrimId) : nullptr;
    if (ids == nullptr || x >= viewTargets_->Extent().width ||
        y >= viewTargets_->Extent().height) {
        return {};
    }
    // The caller's frames are submitted outside the context's tracking.
    vkDeviceWaitIdle(context_.Device());
    GpuBuffer texel(context_, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    BufferMemory::Readback);
    context_.Submit([&](VkCommandBuffer commands) {
        const VkImageLayout layout = ids->Layout();
        ids->Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), 0};
        region.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(commands, ids->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               texel.Handle(), 1, &region);
        ids->Transition(commands, layout);
    });
    std::uint32_t id = 0;
    texel.Read(std::as_writable_bytes(std::span(&id, 1)));
    return PrimPath(id);
}

std::vector<TimelinePoint> RasterDelegate::SubmitWaits() const {
    if (compute_ == nullptr || compute_->VulkanWait().semaphore == VK_NULL_HANDLE) {
        return {};
    }
    return {compute_->VulkanWait()};
}

void RasterDelegate::Record(VkCommandBuffer commands, const RenderingTarget& target,
                            std::string_view var) {
    const VkExtent2D extent = renderer_.Extent();
    if (viewTargets_ == nullptr || viewTargets_->Extent().width != extent.width ||
        viewTargets_->Extent().height != extent.height ||
        viewTargets_->Bands() != renderer_.Bands().count || viewTargets_->AovNames() != viewAovs_) {
        if (extent.width == 0 || extent.height == 0) {
            BeginRendering(commands, target);
            vkCmdEndRendering(commands);
            return;
        }
        vkDeviceWaitIdle(context_.Device());
        const std::vector<std::string_view> vars = ViewHeldVars(viewAovs_.size());
        viewTargets_ = std::make_unique<RenderTargets>(context_, extent, renderer_.Bands().count,
                                                       false, 1, vars, viewAovs_);
    }
    const RasterDrawable* drawables[] = {&terrain_, &planet_, &meshes_, &ground_, &points_};
    renderer_.Record(commands, *viewTargets_, drawables, &target, var);
}

std::shared_ptr<ProductFrame> RasterDelegate::AcquireFrame(const Path& product, VkExtent2D extent,
                                                           int bands, std::uint32_t views,
                                                           bool batched,
                                                           std::span<const std::string_view> vars,
                                                           std::span<const std::string> aovs) {
    ERNIE_ZONE("AcquireFrame");
    const std::vector<std::string_view> held = RequiredVars(vars);
    const std::vector<std::string> aovNames(aovs.begin(), aovs.end());
    std::vector<std::shared_ptr<ProductFrame>>& frames = frames_[product];
    const bool shared = compute_ != nullptr;
    std::erase_if(frames, [&](const std::shared_ptr<ProductFrame>& frame) {
        const RenderTargets& targets = *TargetsOf(*frame->images).front();
        const bool stale = frame.use_count() == 1 && frame->arrayMaps == 0 &&
                           (targets.Extent().width != extent.width ||
                            targets.Extent().height != extent.height || targets.Bands() != bands ||
                            targets.Shared() != shared || frame->images->Views() != views ||
                            frame->batched != batched || targets.Vars() != held ||
                            targets.AovNames() != aovNames);
        if (stale) {
            // Earlier steps may still render into it, and maps copy from it: rare, so wait for
            // everything.
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
    // A batch of more views than one set of targets holds renders into consecutive chunks.
    const std::uint32_t chunk = RenderTargets::MaxViews(context_, bands);
    if (views <= chunk) {
        frame->images = std::make_unique<RenderTargets>(context_, extent, bands, shared, views,
                                                        held, aovNames);
    } else {
        std::vector<std::unique_ptr<FrameImages>> chunks;
        for (std::uint32_t first = 0; first < views; first += chunk) {
            chunks.push_back(std::make_unique<RenderTargets>(
                context_, extent, bands, shared, std::min(chunk, views - first), held, aovNames));
        }
        frame->images = std::make_unique<ChunkedFrameImages>(std::move(chunks));
    }
    frames.push_back(frame);
    return frame;
}

std::uint32_t RasterDelegate::BeginFrame() {
    const std::uint32_t slot = nextSlot_;
    nextSlot_ = (nextSlot_ + 1) % FrameUploads::kSlots;
    // Only the frame that last staged its uploads in this slot must have finished: later ones
    // keep running while this one is prepared.
    {
        ERNIE_ZONE("Wait for the slot's frame");
        context_.WaitForSubmission(slotSubmissions_[slot]);
    }
    renderer_.BeginFrame(slot);
    return slot;
}

StepOperation RasterDelegate::StepAsync(const Scene& scene, std::span<const Path> products,
                                        const Path& settings) {
    ERNIE_ZONE("StepAsync");
    ComputeInterop* compute = Compute();
    const RenderPrims base = ResolveRenderPrims(scene, settings);
    const std::vector<RenderProductSpec> specs = ResolveRenderProducts(scene, products, settings);
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
        const int bands =
            CameraOf(scene, prims.camera, spec.resolution.width, spec.resolution.height)
                .bands.count;
        // The product's cameras, or the settings' one.
        const std::vector<Path> cameras =
            spec.cameras.empty() ? std::vector<Path>{prims.camera} : spec.cameras;
        // Poses written as a GPU array stay on the GPU (`GpuPosesOf`), copied device to device
        // after the work writing them, with the last ones rendered as the CPU's estimates; or
        // come to the host for the views' uniforms.
        const Tensor* gpuPoses = cameras.size() == 1 ? GpuPosesOf(scene, cameras.front()) : nullptr;
        std::optional<Tensor> hostPoses;
        std::vector<glm::dmat4> poses;
        if (gpuPoses != nullptr) {
            const auto count = static_cast<std::uint32_t>(gpuPoses->Shape()[0]);
            poses = renderer_.LastGpuPoses(count);
            if (poses.empty()) {
                const glm::dmat4 own =
                    CameraOf(scene, cameras.front(), spec.resolution.width, spec.resolution.height)
                        .transform;
                poses.assign(count, own);
            }
        } else {
            if (cameras.size() == 1) {
                const Tensor* written = scene.Read(cameras.front(), tokens::kPoses);
                if (written != nullptr && IsGpu(written->Location().kind)) {
                    hostPoses = RequireCompute().Download(
                        *written, scene.WriteStream(cameras.front(), tokens::kPoses));
                }
            }
            poses = ViewPosesOf(scene, cameras, spec.resolution.width, spec.resolution.height,
                                hostPoses ? &*hostPoses : nullptr);
        }
        const bool batched = !poses.empty();
        const auto views = static_cast<std::uint32_t>(std::max<std::size_t>(poses.size(), 1));
        std::vector<std::int32_t> worlds = ViewWorldsOf(scene, cameras, views);
        // Only the vars the product asks for, as the static names of `kRenderVarNames`, and its
        // AOVs ("aov:<output>") as the slots its chains lay them in (`kAovSlotNames`), each from
        // its chain: the product's, or a var's own (`RenderProductSpec::varFilters`).
        const SensorFilterRegistry& registry = renderer_.SensorFilters();
        std::vector<FrameParams::SensorChain> chains;
        std::vector<std::string> aovs;
        std::vector<std::vector<std::string>> chainAovs;
        std::vector<std::vector<std::string_view>> chainWritten;
        const auto chainOf = [&](const std::vector<SensorFilterStep>& steps) {
            const std::vector<SensorFilterStep> full =
                steps.empty() ? StepsOf(DefaultFilterChain()) : steps;
            for (std::size_t c = 0; c < chains.size(); ++c) {
                if (chains[c].filters == full) {
                    return c;
                }
            }
            FrameParams::SensorChain chain;
            chain.filters = full;
            chain.aovOffset = static_cast<std::uint32_t>(aovs.size());
            chainAovs.push_back(AovOutputsOf(full, registry));
            for (const std::string& output : chainAovs.back()) {
                if (std::find(aovs.begin(), aovs.end(), output) != aovs.end()) {
                    throw std::invalid_argument("Two of product '" + spec.product.String() +
                                                "'s chains write AOV '" + output + "'");
                }
                aovs.push_back(output);
            }
            if (aovs.size() > kMaxAovs) {
                throw std::invalid_argument("Product '" + spec.product.String() + "' writes " +
                                            "more than " + std::to_string(kMaxAovs) + " AOVs");
            }
            chainWritten.push_back(VarsWrittenBy(full, registry));
            chains.push_back(std::move(chain));
            return chains.size() - 1;
        };
        chainOf(spec.filters);  // the product's chain first
        std::vector<std::string_view> vars;
        for (std::size_t v = 0; v < spec.vars.size(); ++v) {
            const auto& [path, name] = spec.vars[v];
            const std::optional<std::vector<SensorFilterStep>>& own =
                v < spec.varFilters.size() ? spec.varFilters[v] : std::nullopt;
            const std::size_t c = chainOf(own ? *own : spec.filters);
            const auto missing = [&] {
                return std::invalid_argument("Render var '" + path.String() + "' asks for '" +
                                             name +
                                             "', which its sensor filters do not write");
            };
            std::string_view held;
            const auto known = std::find(kRenderVarNames.begin(), kRenderVarNames.end(), name);
            if (known != kRenderVarNames.end()) {
                const std::vector<std::string_view>& written = chainWritten[c];
                if (std::find(written.begin(), written.end(), name) == written.end()) {
                    throw missing();
                }
                held = *known;
            } else if (name.starts_with(kAovPrefix)) {
                const std::vector<std::string>& outputs = chainAovs[c];
                if (std::find(outputs.begin(), outputs.end(), name.substr(kAovPrefix.size())) ==
                    outputs.end()) {
                    throw missing();
                }
                const auto slot =
                    std::find(aovs.begin(), aovs.end(), name.substr(kAovPrefix.size()));
                held = kAovSlotNames[static_cast<std::size_t>(slot - aovs.begin())];
            } else {
                continue;
            }
            if (std::find(vars.begin(), vars.end(), held) != vars.end()) {
                // The same var twice: the first chain asking for it makes it.
                bool same = false;
                for (const std::string_view other : chains[c].vars) {
                    same = same || other == held;
                }
                if (!same) {
                    throw std::invalid_argument("Product '" + spec.product.String() +
                                                "' asks for '" + name +
                                                "' from two sensor filter chains");
                }
                continue;
            }
            vars.push_back(held);
            chains[c].vars.push_back(held);
        }
        // Chains no var asks for do not run.
        std::erase_if(chains, [](const FrameParams::SensorChain& chain) {
            return chain.vars.empty();
        });
        std::shared_ptr<ProductFrame> frame =
            AcquireFrame(spec.product, spec.resolution, bands, views, batched, vars, aovs);
        std::optional<glm::dmat4> posesParent;
        if (gpuPoses != nullptr) {
            posesParent = PosesParentOf(scene, cameras.front());
        }
        UpdatePrims(scene, prims, spec.resolution, NextFrame(scene, base.settings, spec.product),
                    spec.product, spec.filters, std::move(poses), std::move(worlds),
                    std::move(vars), posesParent, spec.overlays, std::move(chains));
        if (gpuPoses != nullptr) {
            // Once no submission reads the pose buffer (`UpdatePrims` waits), on the writer's
            // stream; the submission waits for it (`SubmitWaits`).
            RequireCompute().CopyToBuffer(*gpuPoses, renderer_.PoseBuffer(), 0,
                                          scene.WriteStream(cameras.front(), tokens::kPoses));
        }
        const std::vector<TimelinePoint> waits = SubmitWaits();
        std::vector<TimelinePoint> signals;
        if (compute != nullptr) {
            signals.push_back(compute->NextVulkanSignal());
        }
        const std::vector<RenderTargets*> targets = TargetsOf(*frame->images);
        const RasterDrawable* drawables[] = {&terrain_, &planet_, &meshes_, &ground_, &points_};
        // Every view of the product renders in this one submission, sharing the shadow passes.
        ERNIE_ZONE_TEXT(spec.product.String());
        submission = context_.SubmitAsync(
            [&](VkCommandBuffer commands) {
                renderer_.RecordShadows(commands, drawables);
                std::uint32_t first = 0;
                for (RenderTargets* chunk : targets) {
                    renderer_.RecordViews(commands, *chunk, drawables, first);
                    first += chunk->Views();
                }
            },
            waits, signals);
        frame->submission = submission;
        slotSubmissions_[slot] = submission;
        RenderProductOutput& output = outputs[spec.product];
        output.product = spec.product;
        for (const auto& [path, name] : spec.vars) {
            output.vars.emplace(name, RenderVar(frame, path, name));
        }
    }
    SettleMotion();
    ERNIE_FRAME_MARK("Step");
    return StepOperation(context_, submission, std::move(outputs));
}

}  // namespace ernie
