// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/volume_pass.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/render_targets.h"
#include "ernie/scene/light_pattern.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/profile.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ernie {

namespace {

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

constexpr DescriptorBinding kBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {11, VK_DESCRIPTOR_TYPE_SAMPLER, kCompute},
    {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {13, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {14, VK_DESCRIPTOR_TYPE_SAMPLER, kCompute},
    {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    // light_pattern's pattern data.
    {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

constexpr std::uint32_t kNoGrid = 0xFFFFFFFFu;
// NanoVDB grids start 32-byte aligned.
constexpr std::uint64_t kGridAlignment = 32;

// volume.slang's structs.
struct FieldGpu {
    float toIndex[3][4];
    std::uint32_t grid[4];
};

struct VolumeGpu {
    float toLocal[3][4];
    float boundsMin[4];
    float boundsMax[4];
    FieldGpu density;
    FieldGpu temperature;
    FieldGpu emission;
    float params[4];
    std::uint32_t info[4];
};
static_assert(sizeof(VolumeGpu) == 304,
              "VolumeGpu must match volume.slang");

struct VolumeMaterialGpu {
    float albedo[SpectralBands::kMaxBands];
    float emission[SpectralBands::kMaxBands];
    float params[4];
    float phase[4];
};
static_assert(sizeof(VolumeMaterialGpu) == 544,
              "VolumeMaterialGpu must match volume.slang");

struct VolumeConstants {
    std::uint32_t layout[4];
    std::uint32_t offsets[4];
    std::uint32_t flags[4];
    std::uint32_t lights[4];
};

// volume.slang's phase functions.
float PhaseOf(const std::string& phase) {
    if (phase == tokens::kDoubleHenyeyGreenstein) {
        return 1.0f;
    }
    if (phase == tokens::kRayleigh) {
        return 2.0f;
    }
    if (phase == tokens::kIsotropic) {
        return 3.0f;
    }
    return 0.0f;
}

std::uint32_t Groups(std::uint32_t size) {
    return (size + 7) / 8;
}

// The first three rows of affine `m`.
void Rows(const glm::dmat4& m, float rows[3][4]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            rows[r][c] = static_cast<float>(m[c][r]);
        }
    }
}

// The bounds of box (`min`, `max`) through affine `m`.
void Transform(const glm::dmat4& m, glm::dvec3& min,
               glm::dvec3& max) {
    const glm::dvec3 lo = min;
    const glm::dvec3 hi = max;
    min = glm::dvec3(std::numeric_limits<double>::infinity());
    max = -min;
    for (int corner = 0; corner < 8; ++corner) {
        const glm::dvec3 p{(corner & 1) != 0 ? hi.x : lo.x,
                           (corner & 2) != 0 ? hi.y : lo.y,
                           (corner & 4) != 0 ? hi.z : lo.z};
        const glm::dvec3 q(m * glm::dvec4(p, 1.0));
        min = glm::min(min, q);
        max = glm::max(max, q);
    }
}

// Whether box (`min`, `max`) may show in clip space `viewProj`: no
// frustum plane has all its corners outside.
bool InFrustum(const glm::mat4& viewProj, const glm::dvec3& min,
               const glm::dvec3& max) {
    std::array<int, 6> outside{};
    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec4 clip =
            viewProj * glm::vec4((corner & 1) != 0 ? max.x : min.x,
                                 (corner & 2) != 0 ? max.y : min.y,
                                 (corner & 4) != 0 ? max.z : min.z,
                                 1.0f);
        outside[0] += clip.x < -clip.w ? 1 : 0;
        outside[1] += clip.x > clip.w ? 1 : 0;
        outside[2] += clip.y < -clip.w ? 1 : 0;
        outside[3] += clip.y > clip.w ? 1 : 0;
        outside[4] += clip.z < 0.0f ? 1 : 0;
        outside[5] += clip.z > clip.w ? 1 : 0;
    }
    return std::none_of(outside.begin(), outside.end(),
                        [](int n) { return n == 8; });
}

// The integral of the CIE y function over 360-830 nm by 1 nm
// midpoints, as the renderer scales emission (cd/m^2 per unit).
double LuminanceIntegral() {
    static const double integral = [] {
        double sum = 0.0;
        for (double w = 360.5; w < 830.0; w += 1.0) {
            sum += CieXyz(w).y;
        }
        return sum;
    }();
    return integral;
}

VolumeMaterialGpu MaterialOf(const VolumeMaterialParams& params,
                             const SpectralBands& bands) {
    VolumeMaterialGpu material{};
    const std::vector<float> albedo =
        RgbToSpectrum(glm::clamp(params.albedo, 0.0, 1.0)).Bin(bands);
    std::copy(albedo.begin(), albedo.end(), material.albedo);
    // Emission as a material's: its colour upsampled on its largest
    // component's scale, under the equal-energy illuminant of
    // 1 cd/m^2 per unit.
    const glm::dvec3 emission = glm::max(params.emission, 0.0);
    const double peak =
        std::max({emission.r, emission.g, emission.b});
    if (peak > 0.0) {
        const std::vector<float> emitted =
            RgbToSpectrum(emission / peak).Bin(bands);
        const double perNit =
            1.0 / (kLuminousEfficacy * LuminanceIntegral());
        for (std::size_t b = 0; b < emitted.size(); ++b) {
            material.emission[b] = static_cast<float>(
                static_cast<double>(emitted[b]) * peak * perNit);
        }
    }
    material.params[0] = static_cast<float>(params.anisotropy);
    material.params[1] =
        static_cast<float>(std::max(params.blackbody, 0.0));
    material.params[2] = peak > 0.0 ? 1.0f : 0.0f;
    material.params[3] = std::any_of(albedo.begin(), albedo.end(),
                                     [](float a) { return a > 0.0f; })
                             ? 1.0f
                             : 0.0f;
    material.phase[0] = PhaseOf(params.phase);
    material.phase[1] =
        static_cast<float>(std::clamp(params.b, 0.0, 0.99));
    material.phase[2] =
        static_cast<float>(std::clamp(params.c, -1.0, 1.0));
    return material;
}

std::unique_ptr<GpuBuffer> Grow(const VulkanContext& context,
                                std::unique_ptr<GpuBuffer> buffer,
                                VkDeviceSize bytes) {
    if (buffer == nullptr || buffer->Size() < bytes) {
        if (buffer != nullptr) {
            vkDeviceWaitIdle(context.Device());
        }
        buffer = std::make_unique<GpuBuffer>(
            context, std::max<VkDeviceSize>(bytes, 16),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            BufferMemory::Device);
    }
    return buffer;
}

}  // namespace

VolumePass::VolumePass(const VulkanContext& context,
                       const ShaderCompiler& compiler)
    : context_(context), compiler_(compiler),
      layout_(context, kBindings, sizeof(VolumeConstants)) {
    pool_ = Grow(context_, nullptr, kGridAlignment);
}

const VolumePass::LoadedGrid*
VolumePass::Load(const VolumeFieldParams& field,
                 const std::string& prim) {
    const std::filesystem::path path(field.file);
    std::error_code error;
    const auto written =
        std::filesystem::last_write_time(path, error);
    if (error) {
        throw std::runtime_error("Volume " + prim + ": cannot read " +
                                 field.file);
    }
    const GridKey key{std::filesystem::absolute(path).string(),
                      field.gridName, field.gridIndex};
    auto found = grids_.find(key);
    if (found == grids_.end() || found->second.written != written) {
        const std::vector<NanoVdbGrid> grids = ReadNanoVdb(path);
        const NanoVdbGrid* grid =
            FindNanoVdbGrid(grids, field.gridName, field.gridIndex);
        if (grid == nullptr) {
            throw std::runtime_error(
                "Volume " + prim + ": " + field.file +
                " has no grid " +
                (field.gridName.empty()
                     ? std::string("#")
                     : "'" + field.gridName + "' #") +
                std::to_string(field.gridIndex));
        }
        LoadedGrid loaded;
        loaded.grid = *grid;
        loaded.written = written;
        found = grids_.insert_or_assign(key, std::move(loaded)).first;
        poolStale_ = true;
    }
    found->second.used = true;
    return &found->second;
}

void VolumePass::RebuildPool() {
    std::uint64_t bytes = 0;
    for (auto& [key, loaded] : grids_) {
        loaded.offset = bytes;
        bytes += (loaded.grid.buffer.size() + kGridAlignment - 1) /
                 kGridAlignment * kGridAlignment;
    }
    // PNanoVDB's 32-bit addresses on the GPU.
    if (bytes >= (std::uint64_t{1} << 32)) {
        throw std::runtime_error(
            "Volumes' grids take 4 GiB or more; Ernie holds less");
    }
    std::vector<std::uint8_t> pool(std::max<std::uint64_t>(bytes, 16),
                                   0);
    for (const auto& [key, loaded] : grids_) {
        std::memcpy(pool.data() + loaded.offset,
                    loaded.grid.buffer.data(),
                    loaded.grid.buffer.size());
    }
    vkDeviceWaitIdle(context_.Device());
    pool_ = std::make_unique<GpuBuffer>(
        context_, pool.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        BufferMemory::Device);
    pool_->Write(std::as_bytes(std::span(pool)));
    poolBytes_ = bytes;
    poolStale_ = false;
}

void VolumePass::Set(std::span<const VolumeParams> volumes,
                     const glm::dvec3& origin,
                     const SpectralBands& bands,
                     std::span<const std::int32_t> viewWorlds,
                     std::span<const glm::mat4> viewProjections,
                     std::uint64_t frame, FrameUploads& uploads) {
    ERNIE_ZONE("Volumes");
    seed_ = static_cast<std::uint32_t>(frame * 2654435761u);
    views_ = static_cast<std::uint32_t>(viewWorlds.size());
    for (auto& [key, loaded] : grids_) {
        loaded.used = false;
    }
    // Each volume's fields, loading the grids not held yet.
    struct Fields {
        const LoadedGrid* density = nullptr;
        const LoadedGrid* temperature = nullptr;
        const LoadedGrid* emission = nullptr;
    };
    std::vector<Fields> fields(volumes.size());
    for (std::size_t i = 0; i < volumes.size(); ++i) {
        const VolumeParams& volume = volumes[i];
        if (volume.density) {
            fields[i].density = Load(*volume.density, volume.prim);
        }
        if (volume.temperature) {
            fields[i].temperature =
                Load(*volume.temperature, volume.prim);
        }
        if (volume.emission) {
            fields[i].emission = Load(*volume.emission, volume.prim);
        }
    }
    // Grids no volume names any more leave the pool.
    for (auto it = grids_.begin(); it != grids_.end();) {
        if (!it->second.used) {
            it = grids_.erase(it);
            poolStale_ = true;
        } else {
            ++it;
        }
    }
    if (poolStale_) {
        RebuildPool();
    }

    std::vector<VolumeMaterialGpu> materials;
    std::vector<VolumeGpu> instances;
    // Each instance's world and its bounds relative to the origin.
    std::vector<std::int32_t> worlds;
    std::vector<std::pair<glm::dvec3, glm::dvec3>> boxes;
    for (std::size_t i = 0; i < volumes.size(); ++i) {
        const VolumeParams& volume = volumes[i];
        const Fields& held = fields[i];
        const std::pair<const LoadedGrid*, const VolumeFieldParams*>
            all[] = {{held.density,
                      volume.density ? &*volume.density : nullptr},
                     {held.temperature, volume.temperature
                                            ? &*volume.temperature
                                            : nullptr},
                     {held.emission,
                      volume.emission ? &*volume.emission : nullptr}};
        // The volume's local bounds: its fields' active values.
        glm::dvec3 lo(std::numeric_limits<double>::infinity());
        glm::dvec3 hi = -lo;
        for (const auto& [loaded, field] : all) {
            if (loaded == nullptr || loaded->grid.Empty()) {
                continue;
            }
            glm::dvec3 min;
            glm::dvec3 max;
            loaded->grid.GridBounds(min, max);
            Transform(field->toVolume, min, max);
            lo = glm::min(lo, min);
            hi = glm::max(hi, max);
        }
        if (glm::any(glm::greaterThan(lo, hi))) {
            continue;
        }
        const auto material =
            static_cast<std::uint32_t>(materials.size());
        materials.push_back(MaterialOf(volume.material, bands));

        VolumeGpu gpu{};
        gpu.boundsMin[0] = static_cast<float>(lo.x);
        gpu.boundsMin[1] = static_cast<float>(lo.y);
        gpu.boundsMin[2] = static_cast<float>(lo.z);
        gpu.boundsMax[0] = static_cast<float>(hi.x);
        gpu.boundsMax[1] = static_cast<float>(hi.y);
        gpu.boundsMax[2] = static_cast<float>(hi.z);
        FieldGpu* const slots[] = {&gpu.density, &gpu.temperature,
                                   &gpu.emission};
        for (std::size_t f = 0; f < 3; ++f) {
            const auto& [loaded, field] = all[f];
            FieldGpu& slot = *slots[f];
            slot.grid[0] = kNoGrid;
            if (loaded == nullptr) {
                continue;
            }
            Rows(glm::inverse(loaded->grid.indexToGrid) *
                     glm::inverse(field->toVolume),
                 slot.toIndex);
            slot.grid[0] = static_cast<std::uint32_t>(loaded->offset);
            slot.grid[1] =
                static_cast<std::uint32_t>(loaded->grid.type);
        }
        gpu.params[0] = static_cast<float>(volume.material.density);
        gpu.params[1] =
            static_cast<float>(volume.material.temperature);
        gpu.params[2] =
            static_cast<float>(volume.material.temperatureScale);
        gpu.info[0] = material;

        // The step: a fraction of the voxels of its densest field
        // (the density's, else the first), as placed in the world.
        const auto& [stepGrid, stepField] =
            held.density != nullptr    ? all[0]
            : held.emission != nullptr ? all[2]
                                       : all[1];
        const glm::dmat4 voxelToLocal =
            stepField->toVolume * stepGrid->grid.indexToGrid;
        const std::vector<glm::dmat4> placements =
            volume.instances.empty()
                ? std::vector<glm::dmat4>{volume.transform}
                : volume.instances;
        for (std::size_t k = 0; k < placements.size(); ++k) {
            const glm::dmat4& transform = placements[k];
            const std::int32_t world =
                k < volume.instanceWorlds.size()
                    ? volume.instanceWorlds[k]
                    : volume.world;
            const glm::dmat4 toVoxel = transform * voxelToLocal;
            const double voxel =
                std::min({glm::length(glm::dvec3(toVoxel[0])),
                          glm::length(glm::dvec3(toVoxel[1])),
                          glm::length(glm::dvec3(toVoxel[2]))});
            VolumeGpu placed = gpu;
            Rows(glm::inverse(transform) *
                     glm::translate(glm::dmat4(1.0), origin),
                 placed.toLocal);
            placed.params[3] = static_cast<float>(
                std::max(voxel, 1e-9) / kStepsPerVoxel);
            placed.info[1] = static_cast<std::uint32_t>(world);
            placed.info[2] = volume.linkGroup;
            instances.push_back(placed);
            worlds.push_back(world);
            glm::dvec3 min = lo;
            glm::dvec3 max = hi;
            Transform(glm::translate(glm::dmat4(1.0), -origin) *
                          transform,
                      min, max);
            boxes.emplace_back(min, max);
        }
    }
    active_ = !instances.empty();
    if (!active_) {
        return;
    }

    // Each view's volumes: those it marches, its world's and the
    // shared ones in its frustum; then those that shadow them, its
    // world's and the shared ones wherever they are (a cloud above
    // the frame shades the dust in it).
    std::vector<std::uint32_t> runs(
        4 * std::max<std::size_t>(views_, 1), 0);
    std::vector<std::uint32_t> lists;
    for (std::uint32_t v = 0; v < views_; ++v) {
        const std::int32_t world = viewWorlds[v];
        const auto inWorld = [&](std::size_t i) {
            return worlds[i] == -1 || world == -1 ||
                   worlds[i] == world;
        };
        runs[4 * v] = static_cast<std::uint32_t>(lists.size());
        for (std::size_t i = 0; i < instances.size(); ++i) {
            if (!inWorld(i) ||
                (v < viewProjections.size() &&
                 !InFrustum(viewProjections[v], boxes[i].first,
                            boxes[i].second))) {
                continue;
            }
            lists.push_back(static_cast<std::uint32_t>(i));
        }
        runs[4 * v + 1] =
            static_cast<std::uint32_t>(lists.size()) - runs[4 * v];
        runs[4 * v + 2] = static_cast<std::uint32_t>(lists.size());
        for (std::size_t i = 0; i < instances.size(); ++i) {
            if (inWorld(i)) {
                lists.push_back(static_cast<std::uint32_t>(i));
            }
        }
        runs[4 * v + 3] = static_cast<std::uint32_t>(lists.size()) -
                          runs[4 * v + 2];
    }
    if (lists.empty()) {
        lists.push_back(0);
    }
    const auto upload = [&](std::unique_ptr<GpuBuffer>& buffer,
                            std::span<const std::byte> bytes) {
        buffer = Grow(context_, std::move(buffer), bytes.size());
        uploads.Write(*buffer, bytes);
    };
    upload(volumes_, std::as_bytes(std::span(instances)));
    upload(materials_, std::as_bytes(std::span(materials)));
    upload(runs_, std::as_bytes(std::span(runs)));
    upload(lists_, std::as_bytes(std::span(lists)));
}

void VolumePass::Record(VkCommandBuffer commands,
                        RenderTargets& targets,
                        const GpuBuffer& shadingUniforms,
                        const Lights& lights,
                        const ViewLayout& layout,
                        std::uint32_t firstView, std::uint32_t width,
                        std::uint32_t height, bool pushbroom,
                        bool sky) const {
    if (!active_) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Volumes");
    const std::size_t slot = pushbroom ? 1 : 0;
    std::unique_ptr<ComputePipeline>& pipeline = pipelines_[slot];
    const std::size_t patterns = lights.patterns->Models().size();
    if (pipeline == nullptr || pipelinePatterns_[slot] != patterns) {
        // The light patterns as shading takes them (`lightPattern`),
        // for lasers' beams in the medium.
        std::vector<ShaderModuleSource> modules =
            LightPatternModules(*lights.patterns);
        modules.insert(
            modules.end(),
            {{"projection", pushbroom
                                ? shaders::kProjectionPushbroomSlang
                                : shaders::kProjectionPinholeSlang},
             {"brdf", shaders::kBrdfSlang},
             {"atmosphere", shaders::kAtmosphereSlang},
             {"volume", std::string(shaders::kPNanoVdbPreludeSlang) +
                            shaders::kPNanoVdbHeader + "\n" +
                            shaders::kVolumeSlang}});
        if (pipeline != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        pipeline = std::make_unique<ComputePipeline>(
            context_, layout_,
            compiler_.CompileSpirv(modules, "volumeMain"));
        pipelinePatterns_[slot] = patterns;
    }
    const auto sampled = [](const GpuImage& image) {
        return DescriptorResource::Of(
            image.View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    const DescriptorResource resources[] = {
        DescriptorResource::Of(shadingUniforms),
        sampled(targets.Depth()),
        sampled(targets.GBuffer(2)),
        DescriptorResource::Of(targets.RasterRadiance().View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(*pool_),
        DescriptorResource::Of(*volumes_),
        DescriptorResource::Of(*materials_),
        DescriptorResource::Of(*runs_),
        DescriptorResource::Of(*lists_),
        DescriptorResource::Of(*lights.table),
        DescriptorResource::Of(lights.shadowMaps->View()),
        DescriptorResource::Of(*lights.shadowSampler),
        DescriptorResource::Of(*lights.shadowViews),
        sampled(*lights.transmittance),
        DescriptorResource::Of(*lights.tableSampler),
        DescriptorResource::Of(*lights.patternNorms),
        DescriptorResource::Of(*lights.timing),
        DescriptorResource::Of(*lights.profiles),
        DescriptorResource::Of(*lights.patternData)};
    VolumeConstants constants{};
    constants.layout[0] = firstView;
    constants.layout[1] = layout.stride;
    constants.layout[2] = kMaxSteps;
    constants.layout[3] = seed_;
    constants.offsets[0] = layout.invViewProj;
    constants.offsets[1] = layout.strip;
    constants.offsets[2] = layout.range;
    constants.offsets[3] = layout.size;
    constants.flags[0] = sky ? 1u : 0u;
    constants.flags[1] =
        targets.Stokes() ? targets.StokesOffset() : 0u;
    constants.lights[0] = layout.view;
    constants.lights[1] = layout.lightShadow;
    constants.lights[2] = layout.atmosphere;
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline->Handle());
    layout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    layout_.PushConstants(commands, &constants, sizeof(constants));
    vkCmdDispatch(commands, Groups(width), Groups(height),
                  targets.RasterLayers());
}

}  // namespace ernie
