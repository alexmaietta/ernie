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

// The dense grids' scatter (volume.slang's `denseScatterMain`): the
// pool it writes and the values it reads.
constexpr DescriptorBinding kScatterBindings[] = {
    {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

// The lighting caches' pass (volume.slang's `lightCacheMain`): the
// grids and the pass's buffers.
constexpr DescriptorBinding kCacheBindings[] = {
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCompute},
};

// What decides whether the caches and shadow maps are built again
// (volume.slang's `volumeChangesMain`): the pass's buffers.
constexpr DescriptorBinding kChangeBindings[] = {
    {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCompute},
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
    VkDeviceAddress lightCache;
    std::uint32_t volumeCount;
    std::uint32_t shadowSize;
    std::uint32_t lightCount;
    std::uint32_t cacheSlots;
    VkDeviceAddress previous;
    VkDeviceAddress rebuild;
    std::uint32_t materialCount;
    std::uint32_t shadowViewCount;
    std::uint32_t unused[2];
};
static_assert(sizeof(VolumeResources) == 112,
              "VolumeResources must match volume.slang");

constexpr std::uint32_t kNoGrid = 0xFFFFFFFFu;
// Volumes' lighting caches (volume.slang's `lightCacheMain`): none
// for an instance, voxels of its densest field per cell along each
// axis, and the most cells along one.
constexpr std::uint32_t kNoCache = 0xFFFFFFFFu;
constexpr double kCacheVoxels = 2.0;
constexpr std::uint32_t kMaxCacheCells = 128;
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
    float toWorld[3][4];
    std::uint32_t cache[4];
};
static_assert(sizeof(VolumeGpu) == 368,
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

// volume.slang's modules, with the light patterns as shading takes
// them (`lightPattern`), for lasers' beams in the medium, for a
// pinhole or a pushbroom camera, its march for `groups` band groups
// (`VOLUME_GROUPS`; 0 for its default).
std::vector<ShaderModuleSource>
VolumeModules(const LightPatternRegistry& patterns, bool pushbroom,
              std::uint32_t groups = 0) {
    std::vector<ShaderModuleSource> modules =
        LightPatternModules(patterns);
    modules.insert(
        modules.end(),
        {{"projection", pushbroom ? shaders::kProjectionPushbroomSlang
                                  : shaders::kProjectionPinholeSlang},
         {"brdf", shaders::kBrdfSlang},
         {"bindless", shaders::kBindlessSlang},
         {"atmosphere", shaders::kAtmosphereSlang},
         {"virtual_shadow", shaders::kVirtualShadowSlang},
         {"volume", (groups > 0 ? "#define VOLUME_GROUPS " +
                                      std::to_string(groups) + "\n"
                                : std::string()) +
                        shaders::kPNanoVdbPreludeSlang +
                        shaders::kPNanoVdbHeader + "\n" +
                        shaders::kVolumeSlang}});
    return modules;
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
      shadowLayout_(context, kShadowBindings),
      scatterLayout_(context, kScatterBindings,
                     sizeof(VolumeConstants)),
      cacheLayout_(context, kCacheBindings),
      changeLayout_(context, kChangeBindings,
                    sizeof(VolumeConstants)) {
    pool_ = Grow(context_, nullptr, kGridAlignment);
    resources_ = std::make_unique<GpuBuffer>(
        context_, sizeof(VolumeResources),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, BufferMemory::Device);
    rebuild_ = std::make_unique<GpuBuffer>(
        context_, 6 * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        BufferMemory::Device);
}

const VolumePass::LoadedGrid*
VolumePass::Load(const VolumeFieldParams& field,
                 const std::string& prim) {
    if (field.dense) {
        // A DenseField's values: on the CPU, a sparse grid of them,
        // built again when written again; on a device, a grid of
        // every voxel of their shape, built once, which the GPU fills
        // from them each time they are written (`StageDenseGrids`,
        // `Prepare`).
        // Or, without values, a writer's each frame
        // (`SetFieldWriter`).
        const auto writer = field.buffer.Empty()
                                ? writers_.find(field.array)
                                : writers_.end();
        if (field.buffer.Empty() && writer == writers_.end()) {
            return nullptr;
        }
        glm::uvec3 shape(0);
        if (writer != writers_.end()) {
            const auto& [x, y, z] = writer->second.first;
            shape = glm::uvec3(x, y, z);
        } else {
            const std::vector<std::int64_t>& dims =
                field.buffer.Shape();
            shape = glm::uvec3(dims[0], dims[1], dims[2]);
        }
        const bool written = writer != writers_.end();
        const GridKey key{field.array, "", 0, true};
        auto found = grids_.find(key);
        if (found == grids_.end() ||
            found->second.version != field.version ||
            found->second.voxelSize != field.voxelSize ||
            found->second.shape != shape ||
            (found->second.writer.empty() == written)) {
            ERNIE_ZONE("Volume dense grid");
            LoadedGrid loaded;
            loaded.version = field.version;
            loaded.dense = true;
            loaded.voxelSize = field.voxelSize;
            loaded.shape = shape;
            // On the CPU, on a device or from a writer alike: a grid
            // of every voxel of the shape, kept while it is, which
            // the GPU fills from the values (uploaded, copied or
            // written).
            {
                if (!written &&
                    field.buffer.Location().kind != DeviceKind::Cpu &&
                    compute_ == nullptr) {
                    throw std::runtime_error("Volume " + prim + ": " +
                                             field.array +
                                             " is on a device, which "
                                             "needs compute interop");
                }
                // The grid of every voxel kept while the shape is.
                const bool same =
                    found != grids_.end() &&
                    found->second.layout.leafBytes != 0 &&
                    found->second.shape == shape &&
                    found->second.voxelSize == field.voxelSize;
                if (same) {
                    loaded.grid = std::move(found->second.grid);
                    loaded.layout = found->second.layout;
                    loaded.offset = found->second.offset;
                } else {
                    loaded.grid =
                        NanoVdbGridsOf(
                            DenseNanoVdbTopology(
                                glm::ivec3(shape), field.voxelSize,
                                glm::dvec3(0.0), "", loaded.layout))
                            .front();
                    poolStale_ = true;
                }
                loaded.device = field.buffer;
                loaded.stream = field.stream;
                loaded.writer = written ? field.array : std::string();
                loaded.scatter = true;
            }
            found =
                grids_.insert_or_assign(key, std::move(loaded)).first;
        }
        // A writer writes every frame.
        if (written) {
            found->second.scatter = true;
        }
        found->second.used = true;
        return &found->second;
    }
    if (!field.array.empty()) {
        // A NanoVDBField's array, read again when written again.
        const GridKey key{field.array, field.gridName,
                          field.gridIndex, true};
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
                // The same array written again in place (its memory
                // and size the same), as a simulation steps its
                // grids: their layout kept, its heads are not read
                // again, so the step is one copy on the device with
                // no wait.
                if (found != grids_.end() &&
                    !found->second.device.Empty() &&
                    found->second.device.Data() ==
                        field.buffer.Data() &&
                    found->second.device.Elements() ==
                        field.buffer.Elements() &&
                    found->second.device.Type() ==
                        field.buffer.Type()) {
                    loaded.grid = found->second.grid;
                    loaded.device = field.buffer;
                    loaded.deviceOffset = found->second.deviceOffset;
                    loaded.stream = field.stream;
                    held = true;
                }
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
        // A device NanoVDBField's grid comes from the device; a dense
        // grid's topology from the host, its values scattered in.
        const std::uint64_t size = loaded.grid.buffer.empty()
                                       ? loaded.grid.size
                                       : loaded.grid.buffer.size();
        bytes += (size + kGridAlignment - 1) / kGridAlignment *
                 kGridAlignment;
        device = device || (!loaded.device.Empty() && !loaded.dense);
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
        if (loaded.dense && loaded.layout.leafBytes != 0) {
            loaded.scatter = true;
        } else if (!loaded.device.Empty()) {
            CopyDeviceGrid(loaded);
        }
    }
    poolBytes_ = bytes;
    poolStale_ = false;
}

void VolumePass::StageDenseGrids(FrameUploads& uploads) {
    scatters_.clear();
    std::uint64_t floats = 0;
    // Shared with the compute device when it copies values in.
    bool shared = false;
    for (const auto& [key, loaded] : grids_) {
        if (loaded.scatter) {
            floats += static_cast<std::uint64_t>(loaded.shape.x) *
                      loaded.shape.y * loaded.shape.z;
            shared = shared || (loaded.writer.empty() &&
                                loaded.device.Location().kind !=
                                    DeviceKind::Cpu);
        }
    }
    if (floats == 0) {
        return;
    }
    ERNIE_ZONE("Volume dense grids");
    if (staging_ == nullptr || staging_->Size() < 4 * floats ||
        (shared && !stagingShared_)) {
        if (staging_ != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        staging_ = std::make_unique<GpuBuffer>(
            context_, 4 * floats,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            shared ? BufferMemory::Shared : BufferMemory::Device);
        stagingShared_ = shared;
    }
    std::uint64_t at = 0;
    for (auto& [key, loaded] : grids_) {
        if (!loaded.scatter) {
            continue;
        }
        Scatter scatter;
        if (!loaded.writer.empty()) {
            scatter.writer = writers_.at(loaded.writer).second;
        } else if (loaded.device.Location().kind == DeviceKind::Cpu) {
            // Uploaded with the frame's other uploads.
            const Tensor values = loaded.device.Contiguous();
            uploads.Write(
                *staging_,
                std::as_bytes(std::span(
                    values.As<float>(),
                    static_cast<std::size_t>(values.Elements()))),
                4 * at);
        } else {
            compute_->CopyToBuffer(loaded.device, *staging_, 4 * at,
                                   loaded.stream);
        }
        scatter.values = static_cast<std::uint32_t>(at);
        scatter.firstLeaf = static_cast<std::uint32_t>(
            (loaded.offset + loaded.layout.firstLeaf) / 4);
        scatter.leafWords = loaded.layout.leafBytes / 4;
        scatter.valuesWord = loaded.layout.valuesOffset / 4;
        scatter.shape = loaded.shape;
        scatter.leaves = glm::uvec3(loaded.layout.leaves);
        scatters_.push_back(std::move(scatter));
        at += static_cast<std::uint64_t>(loaded.shape.x) *
              loaded.shape.y * loaded.shape.z;
        loaded.scatter = false;
    }
}

void VolumePass::SetFieldWriter(const std::string& path,
                                std::array<std::uint32_t, 3> shape,
                                FieldWriter writer) {
    if (writer) {
        writers_[path] = {shape, std::move(writer)};
    } else {
        writers_.erase(path);
    }
}

void VolumePass::Set(std::span<const VolumeParams> volumes,
                     const glm::dvec3& origin,
                     const SpectralBands& bands,
                     std::span<const std::int32_t> viewWorlds,
                     std::span<const glm::mat4> viewProjections,
                     std::span<const std::int32_t> lightViewWorlds,
                     const VolumeSettings& settings,
                     std::uint64_t frame, FrameUploads& uploads) {
    ERNIE_ZONE("Volumes");
    seed_ = static_cast<std::uint32_t>(frame * 2654435761u);
    bandGroups_ = static_cast<std::uint32_t>(
        std::clamp<std::size_t>((bands.count + 3) / 4, 1, 16));
    if (!(settings == settings_)) {
        hostChanged_ = 3;
    }
    settings_ = settings;
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
        hostChanged_ = 3;
    }
    for (auto& [key, loaded] : grids_) {
        if (loaded.copy) {
            CopyDeviceGrid(loaded);
            hostChanged_ = 3;
        }
    }
    StageDenseGrids(uploads);
    if (!scatters_.empty()) {
        hostChanged_ = 3;
    }

    std::vector<VolumeMaterialGpu> materials;
    std::vector<VolumeGpu> instances;
    // Each instance's world and its bounds relative to the origin.
    std::vector<std::int32_t> worlds;
    std::vector<std::pair<glm::dvec3, glm::dvec3>> boxes;
    // The lighting caches' cells, all and the most of one instance.
    std::uint64_t cacheCells = 0;
    std::uint64_t cacheLargest = 0;
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
                    std::max(voxel, 1e-9) / settings_.stepsPerVoxel);
                placed.info[1] = static_cast<std::uint32_t>(world);
                placed.info[2] = volume.linkGroup;
                Rows(glm::translate(glm::dmat4(1.0), -origin) *
                         transform,
                     placed.toWorld);
                // Its lighting cache: a cell per `kCacheVoxels`
                // voxels of its densest field across its bounds, its
                // cells from `cacheCells`.
                placed.cache[3] = kNoCache;
                if (settings_.lightCache) {
                    std::uint64_t cells = 1;
                    for (int a = 0; a < 3; ++a) {
                        const double extent = hi[a] - lo[a];
                        const double size =
                            glm::length(glm::dvec3(voxelToLocal[a]));
                        const auto n =
                            static_cast<std::uint32_t>(std::clamp(
                                std::ceil(extent /
                                          (size * kCacheVoxels)),
                                2.0, double(kMaxCacheCells)));
                        placed.cache[a] = n;
                        cells *= n;
                    }
                    placed.cache[3] =
                        static_cast<std::uint32_t>(cacheCells);
                    cacheCells += cells;
                    cacheLargest = std::max(cacheLargest, cells);
                }
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
    bounds_.reset();
    boxes_.clear();
    for (const auto& [min, max] : boxes) {
        const Bounds3 box{glm::vec3(min), glm::vec3(max)};
        boxes_.push_back(box);
        bounds_ = bounds_
                      ? Bounds3{glm::min(bounds_->first, box.first),
                                glm::max(bounds_->second, box.second)}
                      : box;
    }
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
    upload(lightViewWorlds_, lightViewWorlds.empty()
                                 ? std::as_bytes(std::span(&none, 1))
                                 : std::as_bytes(lightViewWorlds));
    // The passes read them by address (VolumeResources, `Prepare`).
    addresses_ = {volumes_->Address(), materials_->Address(),
                  runs_->Address(), lists_->Address(),
                  lightViewWorlds_->Address()};
    volumeCount_ = static_cast<std::uint32_t>(instances.size());
    materialCount_ = static_cast<std::uint32_t>(materials.size());
    cacheCells_ = cacheCells;
    cacheLargest_ = cacheLargest;
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
    if (!scatters_.empty()) {
        // The dense grids' values written on the device, scattered
        // into their leaves in the pool before anything samples it.
        ERNIE_ZONE("Record dense grids");
        ERNIE_GPU_ZONE(context_, commands, "Volume dense grids");
        if (scatterPipeline_ == nullptr ||
            scatterPipelinePatterns_ != patterns.Models().size()) {
            ERNIE_ZONE("Volume scatter pipeline");
            if (scatterPipeline_ != nullptr) {
                vkDeviceWaitIdle(context_.Device());
            }
            scatterPipeline_ = std::make_unique<ComputePipeline>(
                context_, scatterLayout_,
                compiler_.CompileSpirv(VolumeModules(patterns, false),
                                       "denseScatterMain"));
            scatterPipelinePatterns_ = patterns.Models().size();
        }
        // The writers' values first, into their places in `staging_`.
        bool wrote = false;
        for (const Scatter& scatter : scatters_) {
            if (!scatter.writer) {
                continue;
            }
            ERNIE_ZONE("Volume field writer");
            FieldWriteTarget target;
            target.values = staging_->Address() + 4 * scatter.values;
            target.buffer = staging_->Handle();
            target.offset = 4 * VkDeviceSize{scatter.values};
            target.shape = {scatter.shape.x, scatter.shape.y,
                            scatter.shape.z};
            scatter.writer(commands, target);
            wrote = true;
        }
        if (wrote) {
            VkMemoryBarrier2 values{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            values.srcStageMask =
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            values.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            values.dstStageMask =
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            values.dstAccessMask =
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo written{
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            written.memoryBarrierCount = 1;
            written.pMemoryBarriers = &values;
            vkCmdPipelineBarrier2(commands, &written);
        }
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          scatterPipeline_->Handle());
        const DescriptorResource buffers[] = {
            DescriptorResource::Of(*pool_),
            DescriptorResource::Of(*staging_)};
        scatterLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            buffers);
        for (const Scatter& scatter : scatters_) {
            VolumeConstants constants{};
            constants.layout[0] = scatter.shape.x;
            constants.layout[1] = scatter.shape.y;
            constants.layout[2] = scatter.shape.z;
            constants.layout[3] = scatter.values;
            constants.offsets[0] = scatter.firstLeaf;
            constants.offsets[1] = scatter.leafWords;
            constants.offsets[2] = scatter.valuesWord;
            constants.offsets[3] = scatter.leaves.y;
            constants.flags[0] = scatter.leaves.z;
            scatterLayout_.PushConstants(commands, &constants,
                                         sizeof(constants));
            vkCmdDispatch(commands, (scatter.shape.x + 3) / 4,
                          (scatter.shape.y + 3) / 4,
                          (scatter.shape.z + 3) / 4);
        }
        scatters_.clear();
        VkMemoryBarrier2 written{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        written.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        written.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        written.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        written.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &written;
        vkCmdPipelineBarrier2(commands, &dependency);
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
    resources.shadowSize = ShadowSize();
    resources.lightCount = lightCount;
    // The lighting caches: per instance, a slot for each light and,
    // with the sky's light, two for it from above and the ground's
    // from below.
    const std::uint32_t slots =
        lightCount + (settings_.skyLightSteps > 0 ? 2 : 0);
    if (cacheCells_ > 0) {
        const VkDeviceSize bytes = 4 * cacheCells_ * slots;
        if (cache_ == nullptr || cache_->Size() < bytes) {
            if (cache_ != nullptr) {
                vkDeviceWaitIdle(context_.Device());
            }
            cache_ = std::make_unique<GpuBuffer>(
                context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
            hostChanged_ |= 1;
        }
        resources.lightCache = cache_->Address();
        resources.cacheSlots = slots;
    }
    shadowViews_ = shadowViews;
    const std::uint32_t size = ShadowSize();
    if (shadowViews > 0 &&
        (shadowMap_ == nullptr ||
         shadowMap_->Desc().layers != 3 * shadowViews ||
         shadowMap_->Desc().width != size)) {
        if (shadowMap_ != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        shadowMap_ = std::make_unique<GpuImage>(
            context_, ImageDesc{size, size, 3 * shadowViews, 1,
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT,
                                false, true});
        hostChanged_ |= 2;
    }
    // What the caches and maps read, for the next frame to compare
    // with: laid out again, and so built again, when a count changes.
    const std::array<std::uint32_t, 4> counts{
        volumeCount_, materialCount_, lightCount, shadowViews};
    if (counts != compared_) {
        compared_ = counts;
        hostChanged_ = 3;
    }
    const VkDeviceSize compared =
        volumeCount_ * sizeof(VolumeGpu) +
        materialCount_ * sizeof(VolumeMaterialGpu) + lights.Size() +
        shadowViews * (sizeof(glm::mat4) + sizeof(std::int32_t));
    if (previous_ == nullptr || previous_->Size() < compared) {
        if (previous_ != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        previous_ = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(compared, 16),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
        hostChanged_ = 3;
    }
    resources.previous = previous_->Address();
    resources.rebuild = rebuild_->Address();
    resources.materialCount = materialCount_;
    resources.shadowViewCount = shadowViews;
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

    {
        // Whether to build the caches and maps again, on the GPU:
        // after the last frame's comparison and the dispatches it
        // sized.
        ERNIE_ZONE("Record volume changes");
        VkMemoryBarrier2 last{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        last.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        last.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                             VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        last.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        last.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        dependency.pMemoryBarriers = &last;
        vkCmdPipelineBarrier2(commands, &dependency);
        if (changePipeline_ == nullptr ||
            changePipelinePatterns_ != patterns.Models().size()) {
            ERNIE_ZONE("Volume change pipeline");
            if (changePipeline_ != nullptr) {
                vkDeviceWaitIdle(context_.Device());
            }
            changePipeline_ = std::make_unique<ComputePipeline>(
                context_, changeLayout_,
                compiler_.CompileSpirv(VolumeModules(patterns, false),
                                       "volumeChangesMain"));
            changePipelinePatterns_ = patterns.Models().size();
        }
        VolumeConstants constants{};
        if (cacheCells_ > 0) {
            constants.layout[0] =
                static_cast<std::uint32_t>((cacheLargest_ + 63) / 64);
            constants.layout[1] = slots;
            constants.layout[2] = volumeCount_;
        }
        constants.layout[3] = hostChanged_;
        constants.offsets[0] = Groups(size);
        constants.offsets[1] = Groups(size);
        constants.offsets[2] = shadowViews;
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          changePipeline_->Handle());
        const DescriptorResource buffers[] = {
            DescriptorResource::Of(*resources_)};
        changeLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                           buffers);
        changeLayout_.PushConstants(commands, &constants,
                                    sizeof(constants));
        vkCmdDispatch(commands, 1, 1, 1);
        hostChanged_ = 0;
        VkMemoryBarrier2 sized{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        sized.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        sized.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        sized.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        sized.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        dependency.pMemoryBarriers = &sized;
        vkCmdPipelineBarrier2(commands, &dependency);
    }

    if (cacheCells_ > 0) {
        // Each cell's optical depth towards each light and up and
        // down, once for every view and step to read; none when what
        // it reads is as last frame's.
        ERNIE_ZONE("Record volume lighting caches");
        ERNIE_GPU_ZONE(context_, commands, "Volume lighting caches");
        if (cachePipeline_ == nullptr ||
            cachePipelinePatterns_ != patterns.Models().size()) {
            ERNIE_ZONE("Volume cache pipeline");
            if (cachePipeline_ != nullptr) {
                vkDeviceWaitIdle(context_.Device());
            }
            cachePipeline_ = std::make_unique<ComputePipeline>(
                context_, cacheLayout_,
                compiler_.CompileSpirv(VolumeModules(patterns, false),
                                       "lightCacheMain"));
            cachePipelinePatterns_ = patterns.Models().size();
        }
        const DescriptorResource buffers[] = {
            DescriptorResource::Of(*pool_),
            DescriptorResource::Of(*resources_)};
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          cachePipeline_->Handle());
        cacheLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          buffers);
        vkCmdDispatchIndirect(commands, rebuild_->Handle(), 0);
        VkMemoryBarrier2 built{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        built.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        built.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        built.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        built.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        dependency.pMemoryBarriers = &built;
        vkCmdPipelineBarrier2(commands, &dependency);
    }

    if (shadowViews == 0) {
        return;
    }
    if (shadowPipeline_ == nullptr ||
        shadowPipelinePatterns_ != patterns.Models().size()) {
        ERNIE_ZONE("Volume shadow pipeline");
        if (shadowPipeline_ != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        shadowPipeline_ = std::make_unique<ComputePipeline>(
            context_, shadowLayout_,
            compiler_.CompileSpirv(VolumeModules(patterns, false),
                                   "volumeShadowMain"));
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
    // None when what the maps read is as last frame's.
    vkCmdDispatchIndirect(commands, rebuild_->Handle(),
                          3 * sizeof(std::uint32_t));
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
    // Compiled for the frame's band groups, its arrays of them in
    // registers.
    if (pipeline == nullptr || pipelinePatterns_[slot] != patterns ||
        pipelineGroups_[slot] != bandGroups_) {
        ERNIE_ZONE("Volume pipeline");
        if (pipeline != nullptr) {
            vkDeviceWaitIdle(context_.Device());
        }
        pipeline = std::make_unique<ComputePipeline>(
            context_, layout_,
            compiler_.CompileSpirv(VolumeModules(*shading.patterns,
                                                 pushbroom,
                                                 bandGroups_),
                                   "volumeMain"));
        pipelinePatterns_[slot] = patterns;
        pipelineGroups_[slot] = bandGroups_;
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
    constants.layout[2] = settings_.maxSteps;
    constants.layout[3] = seed_;
    constants.offsets[0] = layout.invViewProj;
    constants.offsets[1] = layout.strip;
    constants.offsets[2] = layout.range;
    constants.offsets[3] = layout.size;
    constants.flags[0] = sky ? 1u : 0u;
    constants.flags[1] =
        targets.Stokes() ? targets.StokesOffset() : 0u;
    // The steps towards the lights, the sky and through the air, and
    // the most octaves, a byte each.
    constants.flags[2] =
        settings_.shadowSteps | (settings_.skyLightSteps << 8) |
        (settings_.airSteps << 16) | (settings_.maxOctaves << 24);
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
