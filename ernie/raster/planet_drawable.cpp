// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/planet_drawable.h"

#include "ernie/kernel/embedded_shaders.h"
#include "ernie/util/profile.h"

#include <glm/gtc/packing.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

constexpr std::size_t kInitialPatchCapacity = 256;

/// LOD levels the uniforms carry morph ranges for.
constexpr std::uint32_t kMorphLevels = 32;
static_assert(kMorphLevels >= CubeSphereLod::kMaxLevels);

struct PlanetUniforms {
    float body[4];
    std::uint32_t dims[4];
    float lodEye[4];
    float morph[kMorphLevels * 2];
    std::uint32_t ids[4];
    float hillshade[4];
    std::uint32_t refine[4];
    float shape[4];
    /// The heights' and gradients' bindless heap slots; unused.
    std::uint32_t textures[4];
};
static_assert(sizeof(PlanetUniforms) == 384,
              "PlanetUniforms must match planet.slang");

constexpr VkDeviceSize kUniformsStride = 512;

struct DrawConstants {
    float lightViewProj[16];
    std::uint32_t first;
    std::uint32_t instanceOffset;
    std::uint32_t indexOffset;
    std::uint32_t records;
    std::uint32_t shadowFromFrame;
    std::uint32_t view;
    std::uint32_t shadowLayer;
    std::uint32_t padding;
};
static_assert(sizeof(DrawConstants) == 96,
              "DrawConstants must match frame.slang");

constexpr VkShaderStageFlags kGraphics =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

/// Frame (0 and 1), material (5) and planet (10 its uniforms, which
/// name its heights and gradients in the bindless heap; 12 its LOD
/// patches; 14 their anchors) bindings; see planet.slang.
constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGraphics},
    {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
    {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphics},
};

std::vector<ShaderModuleSource>
PlanetModules(bool pushbroom = false) {
    return {{"brdf", std::string(BrdfSlang())},
            ProjectionModule(pushbroom),
            {"bindless", shaders::kBindlessSlang},
            {"frame", shaders::kFrameSlang},
            {"planet", shaders::kPlanetSlang},
            {"gbuffer", shaders::kGBufferSlang},
            {"surface", shaders::kSurfaceSlang}};
}

/// `frustum` with its points moved by `offset`: a point p is inside
/// the result where p - offset was inside `frustum`.
Frustum Shifted(Frustum frustum, const glm::vec3& offset) {
    for (glm::vec4& plane : frustum.planes) {
        plane.w -= glm::dot(glm::vec3(plane), offset);
    }
    return frustum;
}

/// A face's slopes along increasing column and row, per vertex:
/// central differences of the heights over the distance between the
/// neighbours' base points (one-sided at the face's edges),
/// and their mip chain packed as half floats, each level averaging 2
/// x 2 of the one before.
std::vector<std::vector<std::uint32_t>>
PackedFaceGradients(const CubeSphereLod& lod, std::uint32_t face,
                    const std::vector<float>& heights) {
    const std::size_t side = lod.FaceCells() + 1;
    const auto at = [&](std::size_t col, std::size_t row) {
        return heights[row * side + col];
    };
    const auto arc = [&](std::size_t c0, std::size_t r0,
                         std::size_t c1, std::size_t r1) {
        return glm::distance(lod.BasePoint(lod.VertexDirection(
                                 face, static_cast<std::uint32_t>(c0),
                                 static_cast<std::uint32_t>(r0))),
                             lod.BasePoint(lod.VertexDirection(
                                 face, static_cast<std::uint32_t>(c1),
                                 static_cast<std::uint32_t>(r1))));
    };
    std::size_t w = side;
    std::size_t h = side;
    std::vector<glm::vec2> level(side * side);
    for (std::size_t row = 0; row < side; ++row) {
        const std::size_t north = row > 0 ? row - 1 : row;
        const std::size_t south = row + 1 < side ? row + 1 : row;
        for (std::size_t col = 0; col < side; ++col) {
            const std::size_t west = col > 0 ? col - 1 : col;
            const std::size_t east = col + 1 < side ? col + 1 : col;
            const double du = (at(east, row) - at(west, row)) /
                              arc(west, row, east, row);
            const double dv = (at(col, south) - at(col, north)) /
                              arc(col, north, col, south);
            level[row * side + col] = glm::vec2(
                static_cast<float>(du), static_cast<float>(dv));
        }
    }
    std::vector<std::vector<std::uint32_t>> mips;
    while (true) {
        std::vector<std::uint32_t> packed(level.size());
        for (std::size_t i = 0; i < level.size(); ++i) {
            packed[i] = glm::packHalf2x16(level[i]);
        }
        mips.push_back(std::move(packed));
        if (w == 1 && h == 1) {
            return mips;
        }
        const std::size_t nw = std::max<std::size_t>(w / 2, 1);
        const std::size_t nh = std::max<std::size_t>(h / 2, 1);
        std::vector<glm::vec2> next(nw * nh);
        for (std::size_t row = 0; row < nh; ++row) {
            const std::size_t r0 = std::min(2 * row, h - 1);
            const std::size_t r1 = std::min(2 * row + 1, h - 1);
            for (std::size_t col = 0; col < nw; ++col) {
                const std::size_t c0 = std::min(2 * col, w - 1);
                const std::size_t c1 = std::min(2 * col + 1, w - 1);
                next[row * nw + col] =
                    0.25f * (level[r0 * w + c0] + level[r0 * w + c1] +
                             level[r1 * w + c0] + level[r1 * w + c1]);
            }
        }
        level = std::move(next);
        w = nw;
        h = nh;
    }
}

/// The refinement levels that bring `lod`'s quads (at the face
/// centres, where they are largest) down to `finestCell` world units,
/// as deep as `CubeSphereLod::SetRefinement` allows; none for 0.
std::uint32_t RefinementFor(const CubeSphereLod& lod,
                            double finestCell) {
    if (!(finestCell > 0.0) || lod.Empty()) {
        return 0;
    }
    const double dataCell = lod.Radius() * 0.5 * std::numbers::pi /
                            static_cast<double>(lod.FaceCells());
    const double wanted = std::ceil(std::log2(dataCell / finestCell));
    std::uint32_t deepest =
        CubeSphereLod::kMaxLevels - (lod.Levels() - lod.Refinement());
    while (deepest > 0 && (static_cast<std::uint64_t>(lod.FaceCells())
                           << deepest) > (std::uint64_t{1} << 30)) {
        --deepest;
    }
    return wanted <= 0.0 ? 0u
                         : static_cast<std::uint32_t>(std::min(
                               wanted, static_cast<double>(deepest)));
}

}  // namespace

PlanetDrawable::PlanetDrawable(RasterRenderer& renderer)
    : renderer_(renderer),
      layout_(renderer.Context(), kBindings, sizeof(DrawConstants)),
      uniforms_(renderer.Context(), kUniformsStride,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                BufferMemory::Device) {
    std::vector<ShaderModuleSource> modules = PlanetModules();
    modules.push_back({"planet_shadow", shaders::kPlanetShadowSlang});
    const std::vector<std::uint32_t> shadowVertex =
        renderer_.Compiler().CompileSpirv(modules, "shadowVertex");
    GraphicsPipelineDesc shadow;
    shadow.vertexSpirv = shadowVertex;
    shadow.depthFormat = VK_FORMAT_D32_SFLOAT;
    shadow.depthBias = true;
    shadow.depthClamp = true;
    shadowPipeline_ = std::make_unique<GraphicsPipeline>(
        renderer_.Context(), layout_, shadow);
    CreateGBufferPipeline(false);
    ReservePatches(kInitialPatchCapacity);
}

PlanetDrawable::~PlanetDrawable() {
    vkDeviceWaitIdle(renderer_.Context().Device());
}

void PlanetDrawable::CreateGBufferPipeline(bool pushbroom) {
    std::vector<ShaderModuleSource> modules =
        PlanetModules(pushbroom);
    modules.push_back({"planet_color", shaders::kPlanetColorSlang});
    const std::vector<std::uint32_t> vertex =
        renderer_.Compiler().CompileSpirv(modules, "vertexMain");
    const std::vector<std::uint32_t> fragment =
        renderer_.Compiler().CompileSpirv(modules, "fragmentMain");
    GraphicsPipelineDesc desc;
    desc.vertexSpirv = vertex;
    desc.fragmentSpirv = fragment;
    desc.colorFormats.assign(RenderTargets::kGBufferFormats.begin(),
                             RenderTargets::kGBufferFormats.end());
    desc.depthFormat = RenderTargets::kDepthFormat;
    desc.depthCompare = RenderTargets::kDepthCompare;
    gbufferPipelines_[pushbroom ? 1 : 0] =
        std::make_unique<GraphicsPipeline>(renderer_.Context(),
                                           layout_, desc);
}

void PlanetDrawable::ReservePatches(std::size_t count) {
    if (patchBuffer_ != nullptr &&
        count * sizeof(LodPatch) <= patchBuffer_->Size()) {
        return;
    }
    const std::size_t capacity = std::max(
        count, patchBuffer_ != nullptr
                   ? static_cast<std::size_t>(
                         2 * patchBuffer_->Size() / sizeof(LodPatch))
                   : std::size_t{0});
    vkDeviceWaitIdle(renderer_.Context().Device());
    patchBuffer_ = std::make_unique<GpuBuffer>(
        renderer_.Context(), capacity * sizeof(LodPatch),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    anchorBuffer_ = std::make_unique<GpuBuffer>(
        renderer_.Context(), capacity * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
}

void PlanetDrawable::ClearHeights() {
    vkDeviceWaitIdle(renderer_.Context().Device());
    heights_.reset();
    gradients_.reset();
    lod_ = CubeSphereLod();
    patches_.clear();
    anchors_.clear();
    mainRanges_.clear();
    cascadeRanges_.clear();
    triangles_ = 0;
    prepared_ = false;
}

void PlanetDrawable::SetHeights(
    double radius,
    std::span<const RasterView, CubeSphereLod::kFaces> faces) {
    SetHeights(glm::dvec3(radius), faces);
}

void PlanetDrawable::SetHeights(
    const glm::dvec3& radii,
    std::span<const RasterView, CubeSphereLod::kFaces> faces) {
    CubeSphereLod lod(std::max({radii.x, radii.y, radii.z}), faces,
                      kLodLeafCells);
    lod.SetRadii(radii);
    const std::uint32_t side = lod.FaceCells() + 1;
    if (side > renderer_.Context().Limits().maxImageDimension2D) {
        throw std::invalid_argument(
            "Planet faces exceed the device's maximum texture size");
    }
    std::vector<std::vector<float>> packed(CubeSphereLod::kFaces);
    std::vector<std::vector<std::vector<std::uint32_t>>> gradients;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        const RasterView& raster = faces[face];
        std::vector<float>& values = packed[face];
        values.resize(static_cast<std::size_t>(side) * side);
        for (std::uint32_t row = 0; row < side; ++row) {
            for (std::uint32_t col = 0; col < side; ++col) {
                const float h = raster(col, row);
                values[static_cast<std::size_t>(row) * side + col] =
                    raster.IsValid(h) ? h : 0.0f;
            }
        }
        gradients.push_back(PackedFaceGradients(lod, face, values));
    }

    vkDeviceWaitIdle(renderer_.Context().Device());
    ClearHeights();
    auto heights = std::make_unique<GpuImage>(
        renderer_.Context(),
        ImageDesc{side, side, CubeSphereLod::kFaces, 1,
                  VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT});
    std::vector<std::span<const std::byte>> heightLayers;
    for (const std::vector<float>& values : packed) {
        heightLayers.push_back(std::as_bytes(std::span(values)));
    }
    heights->Upload(heightLayers);
    const auto mips =
        static_cast<std::uint32_t>(gradients.front().size());
    auto gradientImage = std::make_unique<GpuImage>(
        renderer_.Context(),
        ImageDesc{side, side, CubeSphereLod::kFaces, mips,
                  VK_FORMAT_R16G16_SFLOAT,
                  VK_IMAGE_USAGE_SAMPLED_BIT});
    std::vector<std::span<const std::byte>> gradientLayers;
    for (const auto& face : gradients) {
        for (const std::vector<std::uint32_t>& mip : face) {
            gradientLayers.push_back(std::as_bytes(std::span(mip)));
        }
    }
    gradientImage->Upload(gradientLayers);
    lod_ = std::move(lod);
    const glm::vec2 range = lod_.HeightRange();
    minHeight_ = range.x <= range.y ? range.x : 0.0f;
    maxHeight_ = range.x <= range.y ? range.y : 0.0f;
    heights_ = std::move(heights);
    gradients_ = std::move(gradientImage);
}

std::uint64_t PlanetDrawable::GpuBytes() const {
    std::uint64_t bytes = uniforms_.Size() + patchBuffer_->Size() +
                          anchorBuffer_->Size();
    if (heights_ != nullptr) {
        bytes += heights_->Bytes() + gradients_->Bytes();
    }
    return bytes;
}

std::optional<Bounds3> PlanetDrawable::BodyBounds() const {
    if (!HasHeights()) {
        return std::nullopt;
    }
    const double r = lod_.Radius() +
                     static_cast<double>(
                         std::max(minHeight_ * params_.exaggeration,
                                  maxHeight_ * params_.exaggeration));
    return Bounds3{glm::vec3(glm::dvec3(-r) - Offset()),
                   glm::vec3(glm::dvec3(r) - Offset())};
}

// From radius e the ground at radius `high` can be seen up to the
// tangent to the lowest sphere, `low`, and beyond it as far again as
// that ground rises over it: sqrt(e^2 - low^2) + sqrt(high^2 -
// low^2).
std::optional<Bounds3> PlanetDrawable::Bounds() const {
    if (!HasHeights()) {
        return std::nullopt;
    }
    const double h0 = static_cast<double>(
        std::min(minHeight_ * params_.exaggeration,
                 maxHeight_ * params_.exaggeration));
    const double h1 = static_cast<double>(
        std::max(minHeight_ * params_.exaggeration,
                 maxHeight_ * params_.exaggeration));
    const glm::dvec3& radii = lod_.Radii();
    const double low =
        std::max(0.0, std::min({radii.x, radii.y, radii.z}) +
                          std::min(h0, 0.0));
    const double high = lod_.Radius() + std::max(h1, 0.0);
    // The origin follows the camera within `kRebaseReach`.
    constexpr double kRebaseReach = 1024.0;
    std::vector<glm::dvec3> eyes{Offset()};
    eyes.insert(eyes.end(), lodEyes_.begin(), lodEyes_.end());
    glm::dvec3 lowest(std::numeric_limits<double>::max());
    glm::dvec3 highest(std::numeric_limits<double>::lowest());
    for (const glm::dvec3& eye : eyes) {
        const double e = glm::length(eye);
        const double reach =
            std::sqrt(std::max(0.0, e * e - low * low)) +
            std::sqrt(high * high - low * low) + kRebaseReach;
        lowest = glm::min(lowest, eye - reach);
        highest = glm::max(highest, eye + reach);
    }
    lowest = glm::max(lowest, glm::dvec3(-high));
    highest = glm::min(highest, glm::dvec3(high));
    return Bounds3{glm::vec3(lowest - Offset()),
                   glm::vec3(highest - Offset())};
}

void PlanetDrawable::Select(const glm::dvec3& eye,
                            std::span<const LodRange> ranges,
                            const Frustum* frustum) {
    const std::size_t first = patches_.size();
    lod_.Select(eye, ranges, frustum, params_.exaggeration, true,
                patches_);
    for (std::size_t i = first; i < patches_.size(); ++i) {
        const LodPatch& patch = patches_[i];
        const glm::dvec3 anchor =
            lod_.BasePoint(lod_.FineDirection(
                patch.face, patch.originX, patch.originY)) -
            Offset();
        anchors_.emplace_back(glm::vec3(anchor), 0.0f);
    }
}

std::vector<LodRange>
PlanetDrawable::ShadowRanges(std::span<const LodRange> ranges,
                             const ShadowCascade& view) const {
    // A level-L patch's quads are 2^(L + 1) fine cells across, a fine
    // cell about a quarter of the circumference over `FineCells()`:
    // levels with quads finer than a texel of the view are never
    // subdivided into, so a cascade a kilometre a texel draws
    // kilometre quads, not the eye's metre ones.
    std::vector<LodRange> out(ranges.begin(), ranges.end());
    const double fineCell = lod_.Radius() * 0.5 * std::numbers::pi /
                            static_cast<double>(lod_.FineCells());
    const double texel = view.texelWorldSize;
    if (texel <= 0.0 || fineCell <= 0.0) {
        return out;
    }
    const double coarsest =
        std::floor(std::log2(texel / fineCell)) - 1.0;
    const auto finest = static_cast<std::size_t>(
        std::clamp(coarsest, 0.0, static_cast<double>(out.size())));
    for (std::size_t level = 0; level < finest; ++level) {
        out[level].morphEnd = -1.0f;
    }
    return out;
}

void PlanetDrawable::Prepare(
    const FrameParams& frame,
    std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Planet prepare");
    prepared_ = false;
    pushbroom_ = frame.camera.pushbroom;
    if (gbufferPipelines_[pushbroom_ ? 1 : 0] == nullptr) {
        CreateGBufferPipeline(pushbroom_);
    }
    if (!HasHeights()) {
        return;
    }
    const std::uint32_t material =
        renderer_.HasMaterial(params_.material)
            ? params_.material
            : RasterRenderer::kDefaultMaterial;
    lod_.SetRefinement(RefinementFor(lod_, params_.finestCell));
    PlanetUniforms uniforms{};
    uniforms.refine[0] = lod_.Refinement();
    // (Radius() / a_i)^2: the ellipsoid relative to the sphere of the
    // largest radius, which `body.x` holds.
    for (int i = 0; i < 3; ++i) {
        const double ratio = lod_.Radius() / lod_.Radii()[i];
        uniforms.shape[i] = static_cast<float>(ratio * ratio);
    }
    uniforms.body[0] = static_cast<float>(lod_.Radius());
    uniforms.body[1] = params_.exaggeration;
    uniforms.body[2] = minHeight_;
    uniforms.body[3] = maxHeight_;
    uniforms.dims[0] = lod_.FaceCells();
    uniforms.dims[1] = static_cast<std::uint32_t>(params_.colorMode);
    uniforms.dims[2] = params_.lod.showLevels ? 1u : 0u;
    uniforms.dims[3] = lod_.PatchCells();
    // The hillshade's light in each point's frame (east, north, up).
    const glm::vec3 towards = HillshadeDirection(params_);
    uniforms.hillshade[0] = towards.x;
    uniforms.hillshade[1] = towards.y;
    uniforms.hillshade[2] = towards.z;

    const LodSettings& lod = params_.lod;
    const std::size_t views = frame.Views();
    if (!(lod.freeze && frozen_) || lodEyes_.size() != views) {
        lodEyes_.clear();
        lodFrusta_.clear();
        for (std::size_t v = 0; v < views; ++v) {
            const CameraView view = frame.View(v);
            lodEyes_.push_back(Offset() + glm::dvec3(view.eye));
            lodFrusta_.push_back(
                Shifted(Frustum::FromMatrix(view.ViewProjection()),
                        -view.eye));
        }
    }
    frozen_ = lod.freeze;
    std::vector<LodRange> ranges;
    firstRange_ = 0.0;
    if (lod.enabled) {
        const float morphStart =
            std::clamp(lod.morphStart, 0.05f, 0.95f);
        const double leafWidth =
            lod_.Radius() * 0.5 * std::numbers::pi * kLodLeafCells /
            static_cast<double>(lod_.FineCells());
        firstRange_ = std::max(
            static_cast<double>(lod.detail) * leafWidth,
            lod_.MinFirstRange(params_.exaggeration, morphStart));
        ranges = lod_.Ranges(firstRange_, morphStart);
    }
    for (std::uint32_t level = 0; level < kMorphLevels; ++level) {
        float start = std::numeric_limits<float>::max();
        float slope = 0.0f;
        if (level < ranges.size() &&
            std::isfinite(ranges[level].morphEnd)) {
            start = ranges[level].morphStart;
            slope = 1.0f / (ranges[level].morphEnd - start);
        }
        uniforms.morph[2 * level] = start;
        uniforms.morph[2 * level + 1] = slope;
    }
    uniforms.ids[0] = material;
    uniforms.ids[1] = params_.primId;
    uniforms.ids[2] = params_.linkGroup;
    uniforms.ids[3] = std::bit_cast<std::uint32_t>(
        frame.thermal.enable ? params_.temperature : 0.0f);
    // The heights and gradients in the bindless heap, in fresh slots:
    // the last ones retire once no frame in flight reads them.
    BindlessHeap& heap = renderer_.Context().Bindless();
    heapSlots_.clear();
    for (const GpuImage* image : {heights_.get(), gradients_.get()}) {
        heapSlots_.push_back(heap.AddImage(
            BindlessHeap::Kind::Texture2DArray, image->View()));
        uniforms.textures[heapSlots_.size() - 1] =
            heapSlots_.back().Index();
    }
    const VkDeviceSize bytes = views * kUniformsStride;
    if (uniforms_.Size() < bytes) {
        vkDeviceWaitIdle(renderer_.Context().Device());
        uniforms_ = GpuBuffer(renderer_.Context(), bytes,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              BufferMemory::Device);
    }
    for (std::size_t v = 0; v < views; ++v) {
        const glm::vec3 eye(lodEyes_[v] - Offset());
        uniforms.lodEye[0] = eye.x;
        uniforms.lodEye[1] = eye.y;
        uniforms.lodEye[2] = eye.z;
        renderer_.Uploads().Write(
            uniforms_, std::as_bytes(std::span(&uniforms, 1)),
            v * kUniformsStride);
    }

    patches_.clear();
    anchors_.clear();
    mainRanges_.clear();
    for (std::size_t v = 0; v < views; ++v) {
        const auto first =
            static_cast<std::uint32_t>(patches_.size());
        if (ranges.empty()) {
            if (v == 0) {
                lod_.SelectAll(patches_);
                for (const LodPatch& patch : patches_) {
                    const glm::dvec3 anchor =
                        lod_.BasePoint(lod_.FineDirection(
                            patch.face, patch.originX,
                            patch.originY)) -
                        Offset();
                    anchors_.emplace_back(glm::vec3(anchor), 0.0f);
                }
            }
            mainRanges_.push_back(
                {0, static_cast<std::uint32_t>(patches_.size())});
            continue;
        }
        Select(lodEyes_[v], ranges,
               frame.posesOnGpu ? nullptr : &lodFrusta_[v]);
        mainRanges_.push_back(
            {first,
             static_cast<std::uint32_t>(patches_.size()) - first});
    }
    triangles_ = 0;
    for (const DrawRange& range : mainRanges_) {
        triangles_ += lod_.TriangleCount(
            std::span(patches_).subspan(range.first, range.count));
    }
    cascadeRanges_.assign(cascades.size(), DrawRange{});
    for (std::size_t c = 0; c < cascades.size(); ++c) {
        if (ranges.empty()) {
            cascadeRanges_[c] = mainRanges_.front();
            continue;
        }
        const auto first =
            static_cast<std::uint32_t>(patches_.size());
        const glm::vec3 eye(lodEyes_.front() - Offset());
        // A view the GPU poses, or moving across the shutter, draws
        // every patch: its matrix here is only an estimate, or one of
        // its sub-frames'.
        const ShadowCascade& cascade = cascades[c];
        const Frustum light = Shifted(
            Frustum::FromMatrix(cascade.lightViewProjection), -eye);
        const bool culls = !cascade.moving && !cascade.gpuFitted;
        Select(lodEyes_.front(), ShadowRanges(ranges, cascade),
               culls ? &light : nullptr);
        cascadeRanges_[c] = {
            first,
            static_cast<std::uint32_t>(patches_.size()) - first};
    }
    if (!patches_.empty()) {
        ReservePatches(patches_.size());
        renderer_.Uploads().Write(*patchBuffer_,
                                  std::as_bytes(std::span(patches_)));
        renderer_.Uploads().Write(*anchorBuffer_,
                                  std::as_bytes(std::span(anchors_)));
    }
    prepared_ = true;
}

std::vector<DescriptorResource>
PlanetDrawable::Resources(const RasterFrame& frame,
                          std::size_t view) const {
    std::vector<DescriptorResource> resources = {frame.resources,
                                                 frame.shadows};
    renderer_.MaterialOrDefault(params_.material)
        .AppendResources(resources);
    resources.push_back(DescriptorResource::Of(
        uniforms_, view * kUniformsStride, sizeof(PlanetUniforms)));
    resources.push_back(DescriptorResource::Of(*patchBuffer_));
    resources.push_back(DescriptorResource::Of(*anchorBuffer_));
    return resources;
}

void PlanetDrawable::DrawPatches(VkCommandBuffer commands,
                                 DrawRange range,
                                 const glm::mat4& light,
                                 const RasterFrame& frame,
                                 bool shadow) const {
    if (range.count == 0) {
        return;
    }
    DrawConstants constants{};
    std::memcpy(constants.lightViewProj, glm::value_ptr(light),
                sizeof(constants.lightViewProj));
    constants.first = range.first;
    constants.shadowFromFrame =
        shadow && frame.shadowFromFrame ? 1u : 0u;
    constants.view = frame.view;
    constants.shadowLayer = shadow ? frame.shadowLayer : 0u;
    layout_.PushConstants(commands, &constants, sizeof(constants));
    const std::uint32_t cells = lod_.PatchCells();
    vkCmdDraw(commands, 6 * cells * cells, range.count, 0, 0);
}

void PlanetDrawable::RecordShadow(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::size_t cascade, const glm::mat4& lightViewProjection,
    const ShadowCasters& casters) const {
    if (!prepared_ || cascade >= cascadeRanges_.size() ||
        !casters.shared || !casters.Casts(params_.linkGroup)) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      shadowPipeline_->Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 Resources(frame, 0));
    DrawPatches(commands, cascadeRanges_[cascade],
                lightViewProjection, frame, true);
}

void PlanetDrawable::RecordColor(VkCommandBuffer commands,
                                 const RasterFrame& frame,
                                 std::size_t view,
                                 DrawPhase phase) const {
    if (!prepared_ || view >= mainRanges_.size() ||
        phase == DrawPhase::Occludees) {
        return;
    }
    vkCmdBindPipeline(
        commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
        gbufferPipelines_[pushbroom_ ? 1 : 0]->Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 Resources(frame, view));
    DrawPatches(commands, mainRanges_[view], glm::mat4(1.0f), frame);
}

}  // namespace ernie
