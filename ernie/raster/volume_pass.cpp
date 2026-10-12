// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/volume_pass.h"

#include "ernie/device/compute_interop.h"
#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/drawable.h"
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
    // The shading pass's resources (ShadingResources).
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    // The grids, which PNanoVDB reads as a StructuredBuffer<uint>.
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    // The pass's own buffers (VolumeResources).
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCompute},
    // light_pattern's pattern data.
    {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

// The volumes' shadow maps' pass (volume.slang's `volumeShadowMain`):
// the grids, the pass's buffers and the maps it writes.
constexpr DescriptorBinding kShadowBindings[] = {
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCompute},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

// volume.slang's VolumeResources.
struct VolumeResources {
    VkDeviceAddress volumes;
    VkDeviceAddress materials;
    VkDeviceAddress runs;
    VkDeviceAddress lists;
    VkDeviceAddress lights;
    VkDeviceAddress lightViews;
    VkDeviceAddress lightViewWorlds;
    std::uint32_t volumeCount;
    std::uint32_t shadowSize;
    std::uint32_t lightCount;
    std::uint32_t unused;
};
static_assert(sizeof(VolumeResources) == 72,
              "VolumeResources must match volume.slang");

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
    float scattering[4];
};
static_assert(sizeof(VolumeMaterialGpu) == 560,
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
    material.phase[3] =
        static_cast<float>(std::max(params.skyLight, 0.0));
    // Wrenninge's octaves, a at most b so that each order carries no
    // more energy than its extinction lets through.
    const double extinction =
        std::clamp(params.octaveExtinction, 0.0, 1.0);
    material.scattering[0] =
        static_cast<float>(std::clamp(params.octaves, 1, 8));
    material.scattering[1] = static_cast<float>(
        std::clamp(params.octaveAttenuation, 0.0, extinction));
    material.scattering[2] = static_cast<float>(extinction);
    material.scattering[3] =
        static_cast<float>(std::clamp(params.octavePhase, 0.0, 1.0));
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

// `count` bytes of contiguous `tensor` from byte `offset`, a view on
// its device.
Tensor BytesOf(const Tensor& tensor, std::uint64_t offset,
               std::uint64_t count) {
    return Tensor(static_cast<std::uint8_t*>(tensor.Data()) + offset,
                  DType::UInt8, {static_cast<std::int64_t>(count)},
                  {}, tensor.Location());
}

}  // namespace

VolumePass::VolumePass(const VulkanContext& context,
                       const ShaderCompiler& compiler)
    : context_(context), compiler_(compiler),
      layout_(context, kBindings, sizeof(VolumeConstants)),
      shadowLayout_(context, kShadowBindings) {
    pool_ = Grow(context_, nullptr, kGridAlignment);
    resources_ = std::make_unique<GpuBuffer>(
        context_, sizeof(VolumeResources),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, BufferMemory::Device);
}

const VolumePass::LoadedGrid*
VolumePass::Load(const VolumeFieldParams& field,
                 const std::string& prim) {
    if (!field.array.empty()) {
        // A NanoVDBField's array, read again when written again.
        const GridKey key{field.array, field.gridName, field.gridIndex,
                          true};
        auto found = grids_.find(key);
        if (found == grids_.end() ||
            found->second.version != field.version) {
            ERNIE_ZONE("Volume grid array");
            const auto missing = [&] {
                return std::runtime_error(
                    "Volume " + prim + ": " + field.array +
                    " has no grid " +
                    (field.gridName.empty()
                         ? std::string("#")
                         : "'" + field.gridName + "' #") +
                    std::to_string(field.gridIndex));
            };
            LoadedGrid loaded;
            loaded.version = field.version;
            if (field.buffer.Location().kind == DeviceKind::Cpu) {
                const Tensor bytes = field.buffer.Contiguous();
                const std::vector<NanoVdbGrid> grids =
                    NanoVdbGridsOf(std::span(
                        static_cast<const std::uint8_t*>(
                            bytes.Data()),
                        static_cast<std::size_t>(bytes.Elements()) *
                            SizeOf(bytes.Type())));
                const NanoVdbGrid* grid = FindNanoVdbGrid(
                    grids, field.gridName, field.gridIndex);
                if (grid == nullptr) {
                    throw missing();
                }
                loaded.grid = *grid;
            } else {
                // On a compute device: only each grid's head comes to
                // the host, to find the grid and read its header; the
                // grid is copied into the pool there.
                if (compute_ == nullptr) {
                    throw std::runtime_error("Volume " + prim + ": " +
                                             field.array +
                                             " is on a device, which "
                                             "needs compute interop");
                }
                const std::uint64_t total =
                    static_cast<std::uint64_t>(
                        field.buffer.Elements()) *
                    SizeOf(field.buffer.Type());
                std::uint64_t offset = 0;
                int matched = 0;
                bool held = false;
                while (!held && offset < total) {
                    const Tensor head = compute_->Download(
                        BytesOf(
                            field.buffer, offset,
                            std::min<std::uint64_t>(kNanoVdbHeadBytes,
                                                    total - offset)),
                        field.stream);
                    NanoVdbGrid grid = NanoVdbGridHead(std::span(
                        head.As<std::uint8_t>(),
                        static_cast<std::size_t>(head.Elements())));
                    if (offset + grid.size > total) {
                        throw std::runtime_error(
                            "Volume " + prim + ": " + field.array +
                            "'s grid runs past its array");
                    }
                    if ((field.gridName.empty() ||
                         grid.name == field.gridName) &&
                        matched++ == field.gridIndex) {
                        loaded.grid = std::move(grid);
                        loaded.device = field.buffer;
                        loaded.deviceOffset = offset;
                        loaded.stream = field.stream;
                        held = true;
                    }
                    offset += grid.size;
                }
                if (!held) {
                    throw missing();
                }
            }
            // A device grid written again at its size is copied over
            // itself, the pool kept (`Set`).
            const bool inPlace =
                found != grids_.end() && !loaded.device.Empty() &&
                !found->second.device.Empty() &&
                found->second.grid.size == loaded.grid.size;
            if (inPlace) {
                loaded.offset = found->second.offset;
                loaded.copy = true;
            } else {
                poolStale_ = true;
            }
            found =
                grids_.insert_or_assign(key, std::move(loaded)).first;
        }
        found->second.used = true;
        return &found->second;
    }
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
        ERNIE_ZONE("Volume grid read");
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

void VolumePass::CopyDeviceGrid(LoadedGrid& loaded) {
    ERNIE_ZONE("Volume grid copy");
    compute_->CopyToBuffer(
        BytesOf(loaded.device, loaded.deviceOffset, loaded.grid.size),
        *pool_, loaded.offset, loaded.stream);
    loaded.copy = false;
}

void VolumePass::RebuildPool() {
    ERNIE_ZONE("Volume grid pool");
    std::uint64_t bytes = 0;
    bool device = false;
    for (auto& [key, loaded] : grids_) {
        loaded.offset = bytes;
        const std::uint64_t size = loaded.device.Empty()
                                       ? loaded.grid.buffer.size()
                                       : loaded.grid.size;
        bytes += (size + kGridAlignment - 1) / kGridAlignment *
                 kGridAlignment;
        device = device || !loaded.device.Empty();
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
    // Shared with the compute device when it copies grids in.
    pool_ = std::make_unique<GpuBuffer>(
        context_, pool.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device ? BufferMemory::Shared : BufferMemory::Device);
    pool_->Write(std::as_bytes(std::span(pool)));
    for (auto& [key, loaded] : grids_) {
        if (!loaded.device.Empty()) {
            CopyDeviceGrid(loaded);
        }
    }
    poolBytes_ = bytes;
    poolStale_ = false;
}

void VolumePass::Set(std::span<const VolumeParams> volumes,
                     const glm::dvec3& origin,
                     const SpectralBands& bands,
                     std::span<const std::int32_t> viewWorlds,
                     std::span<const glm::mat4> viewProjections,
                     std::span<const std::int32_t> lightViewWorlds,
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
    {
        ERNIE_ZONE("Volume fields");
        for (std::size_t i = 0; i < volumes.size(); ++i) {
            const VolumeParams& volume = volumes[i];
            if (volume.density) {
                fields[i].density =
                    Load(*volume.density, volume.prim);
            }
            if (volume.temperature) {
                fields[i].temperature =
                    Load(*volume.temperature, volume.prim);
            }
            if (volume.emission) {
                fields[i].emission =
                    Load(*volume.emission, volume.prim);
            }
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
    for (auto& [key, loaded] : grids_) {
        if (loaded.copy) {
            CopyDeviceGrid(loaded);
        }
    }

    std::vector<VolumeMaterialGpu> materials;
    std::vector<VolumeGpu> instances;
    // Each instance's world and its bounds relative to the origin.
    std::vector<std::int32_t> worlds;
    std::vector<std::pair<glm::dvec3, glm::dvec3>> boxes;
    {
        ERNIE_ZONE("Volume instances");
        for (std::size_t i = 0; i < volumes.size(); ++i) {
            const VolumeParams& volume = volumes[i];
            const Fields& held = fields[i];
            const std::pair<const LoadedGrid*,
                            const VolumeFieldParams*>
                all[] = {
                    {held.density,
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
                slot.grid[0] =
                    static_cast<std::uint32_t>(loaded->offset);
                slot.grid[1] =
                    static_cast<std::uint32_t>(loaded->grid.type);
            }
            gpu.params[0] =
                static_cast<float>(volume.material.density);
            gpu.params[1] =
                static_cast<float>(volume.material.temperature);
            gpu.params[2] =
                static_cast<float>(volume.material.temperatureScale);
            gpu.info[0] = material;

            // The step: a fraction of the voxels of its densest field
            // (the density's, else the first), as placed in the
            // world.
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
    {
        ERNIE_ZONE("Volume view lists");
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
                static_cast<std::uint32_t>(lists.size()) -
                runs[4 * v];
            runs[4 * v + 2] =
                static_cast<std::uint32_t>(lists.size());
            for (std::size_t i = 0; i < instances.size(); ++i) {
                if (inWorld(i)) {
                    lists.push_back(static_cast<std::uint32_t>(i));
                }
            }
            runs[4 * v + 3] =
                static_cast<std::uint32_t>(lists.size()) -
                runs[4 * v + 2];
        }
    }
    if (lists.empty()) {
        lists.push_back(0);
    }
    const auto upload = [&](std::unique_ptr<GpuBuffer>& buffer,
                            std::span<const std::byte> bytes) {
        ERNIE_ZONE("Volume upload");
        buffer = Grow(context_, std::move(buffer), bytes.size());
        uploads.Write(*buffer, bytes);
    };
    upload(volumes_, std::as_bytes(std::span(instances)));
    upload(materials_, std::as_bytes(std::span(materials)));
    upload(runs_, std::as_bytes(std::span(runs)));
    upload(lists_, std::as_bytes(std::span(lists)));
    const std::int32_t none = ShadowCasters::kNoWorld;
    upload(lightViewWorlds_,
           lightViewWorlds.empty()
               ? std::as_bytes(std::span(&none, 1))
               : std::as_bytes(lightViewWorlds));
    // The passes read them by address (VolumeResources, `Prepare`).
    addresses_ = {volumes_->Address(), materials_->Address(),
                  runs_->Address(), lists_->Address(),
                  lightViewWorlds_->Address()};
    volumeCount_ = static_cast<std::uint32_t>(instances.size());
}

void VolumePass::Prepare(VkCommandBuffer commands,
                         const GpuBuffer& lights,
                         const GpuBuffer& lightViews,
                         std::uint32_t lightCount,
                         const LightPatternRegistry& patterns,
                         const GpuBuffer& patternData,
                         std::uint32_t shadowViews) const {
    if (!active_) {
        return;
    }
    ERNIE_ZONE("Record volume shadows");
    ERNIE_GPU_ZONE(context_, commands, "Volume shadows");
    VolumeResources resources{};
    resources.volumes = addresses_[0];
    resources.materials = addresses_[1];
    resources.runs = addresses_[2];
    resources.lists = addresses_[3];
    resources.lights = lights.Address();
    resources.lightViews = lightViews.Address();
    resources.lightViewWorlds = addresses_[4];
    resources.volumeCount = volumeCount_;
    resources.shadowSize = kShadowSize;
    resources.lightCount = lightCount;
    // After whatever read the last frame's, before the passes reading
    // these.
    VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    before.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    before.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
    before.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &before;
    vkCmdPipelineBarrier2(commands, &dependency);
    vkCmdUpdateBuffer(commands, resources_->Handle(), 0,
                      sizeof(resources), &resources);
    VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    after.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    after.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    after.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    after.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
    dependency.pMemoryBarriers = &after;
    vkCmdPipelineBarrier2(commands, &dependency);

    shadowViews_ = shadowViews;
    if (shadowViews == 0) {
        return;
    }
    if (shadowMap_ == nullptr ||
        shadowMap_->Desc().layers != 3 * shadowViews) {
        if (shadowMap_ != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        shadowMap_ = std::make_unique<GpuImage>(
            context_,
            ImageDesc{kShadowSize, kShadowSize, 3 * shadowViews, 1,
                      VK_FORMAT_R32G32B32A32_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT |
                          VK_IMAGE_USAGE_SAMPLED_BIT,
                      false, true});
    }
    if (shadowPipeline_ == nullptr ||
        shadowPipelinePatterns_ != patterns.Models().size()) {
        ERNIE_ZONE("Volume shadow pipeline");
        std::vector<ShaderModuleSource> modules =
            LightPatternModules(patterns);
        modules.insert(
            modules.end(),
            {{"projection", shaders::kProjectionPinholeSlang},
             {"brdf", shaders::kBrdfSlang},
             {"bindless", shaders::kBindlessSlang},
             {"atmosphere", shaders::kAtmosphereSlang},
             {"volume", std::string(shaders::kPNanoVdbPreludeSlang) +
                            shaders::kPNanoVdbHeader + "\n" +
                            shaders::kVolumeSlang}});
        if (shadowPipeline_ != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        shadowPipeline_ = std::make_unique<ComputePipeline>(
            context_, shadowLayout_,
            compiler_.CompileSpirv(modules, "volumeShadowMain"));
        shadowPipelinePatterns_ = patterns.Models().size();
    }
    shadowMap_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    const DescriptorResource resourcesOf[] = {
        DescriptorResource::Of(*pool_),
        DescriptorResource::Of(*resources_),
        DescriptorResource::Of(shadowMap_->View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(patternData)};
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      shadowPipeline_->Handle());
    shadowLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                       resourcesOf);
    vkCmdDispatch(commands, Groups(kShadowSize), Groups(kShadowSize),
                  shadowViews);
    shadowMap_->Transition(commands,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VolumePass::Record(VkCommandBuffer commands,
                        RenderTargets& targets,
                        const Shading& shading,
                        const ViewLayout& layout,
                        std::uint32_t firstView, std::uint32_t width,
                        std::uint32_t height, bool pushbroom,
                        bool sky) const {
    if (!active_) {
        return;
    }
    ERNIE_ZONE("Record volumes");
    ERNIE_GPU_ZONE(context_, commands, "Volumes");
    const std::size_t slot = pushbroom ? 1 : 0;
    std::unique_ptr<ComputePipeline>& pipeline = pipelines_[slot];
    const std::size_t patterns = shading.patterns->Models().size();
    if (pipeline == nullptr || pipelinePatterns_[slot] != patterns) {
        ERNIE_ZONE("Volume pipeline");
        // The light patterns as shading takes them (`lightPattern`),
        // for lasers' beams in the medium.
        std::vector<ShaderModuleSource> modules =
            LightPatternModules(*shading.patterns);
        modules.insert(
            modules.end(),
            {{"projection", pushbroom
                                ? shaders::kProjectionPushbroomSlang
                                : shaders::kProjectionPinholeSlang},
             {"brdf", shaders::kBrdfSlang},
             {"bindless", shaders::kBindlessSlang},
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
        DescriptorResource::Of(*shading.resources),
        sampled(targets.Depth()),
        sampled(targets.GBuffer(2)),
        DescriptorResource::Of(targets.RasterRadiance().View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(*pool_),
        DescriptorResource::Of(*resources_),
        DescriptorResource::Of(*shading.patternData)};
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
