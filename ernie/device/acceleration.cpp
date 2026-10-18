// SPDX-License-Identifier: BSD-3-Clause
/// @file acceleration.cpp
/// @brief Sizing acceleration structures, keeping them in a pool, and recording their builds,
/// compacted-size queries and compaction.
#include "ernie/device/acceleration.h"

#include "ernie/device/context.h"

#include <stdexcept>
#include <utility>

namespace ernie {
namespace {

/// The alignment the specification requires of an acceleration structure's offset.
constexpr std::uint64_t kStructureAlignment = 256;
/// The alignment instance arrays need.
constexpr std::uint64_t kInstanceAlignment = 16;

VkDeviceAddress DeviceAddress(const BufferView& view) {
    return view.buffer == nullptr ? 0 : view.buffer->Address() + view.offset;
}

void RequireRayQueries(const DeviceContext& context) {
    if (!context.Capabilities().rayQuery) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          context.Info().name + " has no hardware acceleration structures");
    }
}

VkBuildAccelerationStructureFlagsKHR ToVk(const BuildFlags& flags) {
    VkBuildAccelerationStructureFlagsKHR vk =
        flags.fastTrace ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                        : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    if (flags.compactable) {
        vk |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }
    if (flags.updatable) {
        vk |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    return vk;
}

VkAccelerationStructureTypeKHR ToVk(AccelerationKind kind) {
    return kind == AccelerationKind::Bottom ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
                                            : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
}

VkAccelerationStructureGeometryKHR Geometry(const TriangleGeometry& mesh) {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = DeviceAddress(mesh.vertices);
    triangles.vertexStride = mesh.stride;
    triangles.maxVertex = mesh.vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = DeviceAddress(mesh.indices);
    VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
    geometry.geometry.triangles = triangles;
    return geometry;
}

VkAccelerationStructureGeometryKHR Geometry(const BoxGeometry& boxes) {
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
    aabbs.data.deviceAddress = DeviceAddress(boxes.boxes);
    aabbs.stride = boxes.stride;
    // Opaque: a candidate is committed by the shader that intersects it exactly.
    VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
    geometry.geometry.aabbs = aabbs;
    return geometry;
}

VkAccelerationStructureGeometryKHR Geometry(const BufferView& instances) {
    VkAccelerationStructureGeometryInstancesDataKHR data{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    data.data.deviceAddress = DeviceAddress(instances);
    VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR};
    geometry.geometry.instances = data;
    return geometry;
}

/// A bottom level's geometries and their primitive counts.
struct BottomInput {
    std::vector<VkAccelerationStructureGeometryKHR> geometry;
    std::vector<std::uint32_t> primitives;
};

BottomInput Input(std::span<const TriangleGeometry> triangles, std::span<const BoxGeometry> boxes) {
    if (triangles.empty() == boxes.empty()) {
        throw std::invalid_argument("A bottom level holds triangles or boxes, one kind only");
    }
    BottomInput input;
    for (const TriangleGeometry& mesh : triangles) {
        if (mesh.vertexCount == 0 || mesh.triangleCount == 0) {
            throw std::invalid_argument("A bottom level's triangles need vertices");
        }
        input.geometry.push_back(Geometry(mesh));
        input.primitives.push_back(mesh.triangleCount);
    }
    for (const BoxGeometry& box : boxes) {
        input.geometry.push_back(Geometry(box));
        input.primitives.push_back(box.count);
    }
    return input;
}

VkAccelerationStructureBuildGeometryInfoKHR
BuildInfo(AccelerationKind kind, const BuildFlags& flags,
          std::span<const VkAccelerationStructureGeometryKHR> geometry) {
    return {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = ToVk(kind),
            .flags = ToVk(flags),
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = static_cast<std::uint32_t>(geometry.size()),
            .pGeometries = geometry.data()};
}

AccelerationSizes Query(const DeviceContext& context, AccelerationKind kind,
                        const BuildFlags& flags,
                        std::span<const VkAccelerationStructureGeometryKHR> geometry,
                        std::span<const std::uint32_t> primitives) {
    const VkAccelerationStructureBuildGeometryInfoKHR build = BuildInfo(kind, flags, geometry);
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(context.Device(),
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                                            primitives.data(), &sizes);
    return {.kind = kind,
            .flags = flags,
            .storage = sizes.accelerationStructureSize,
            .buildScratch = sizes.buildScratchSize,
            .updateScratch = flags.updatable ? sizes.updateScratchSize : 0,
            .primitives = {primitives.begin(), primitives.end()}};
}

/// Checks a build holds no more geometries or primitives than its target was sized for.
void CheckFits(const AccelerationStructure& target, std::span<const std::uint32_t> primitives) {
    const std::vector<std::uint32_t>& sized = target.Sizes().primitives;
    bool fits = primitives.size() <= sized.size();
    for (std::size_t i = 0; fits && i < primitives.size(); ++i) {
        fits = primitives[i] <= sized[i];
    }
    if (!fits) {
        throw std::invalid_argument("A build of " + target.Name() +
                                    " holds more than it was sized for");
    }
}

std::uint64_t QueryScratchAlignment(const DeviceContext& context) {
    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 all{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                    .pNext = &properties};
    vkGetPhysicalDeviceProperties2(context.Physical(), &all);
    return std::max<std::uint64_t>(properties.minAccelerationStructureScratchOffsetAlignment, 1);
}

/// @return The pool the builds' targets share, which sets their scratch alignment.
const AccelerationPool& BuildPool(std::span<const BottomBuild> bottoms,
                                  std::span<const TopBuild> tops) {
    const AccelerationStructure* first = bottoms.empty() ? tops[0].target : bottoms[0].target;
    if (first == nullptr) {
        throw std::invalid_argument("A build has no target");
    }
    return first->Pool();
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t Scratch(const AccelerationStructure& target, bool update) {
    return update ? target.Sizes().updateScratch : target.Sizes().buildScratch;
}

void CheckTarget(const AccelerationStructure* target, AccelerationKind kind, bool update) {
    if (target == nullptr || target->Kind() != kind) {
        throw std::invalid_argument("A build's target is missing or of the other kind");
    }
    if (update && !target->Sizes().flags.updatable) {
        throw std::invalid_argument(target->Name() + " was not built to be updated");
    }
}

}  // namespace

AccelerationSizes Sizes(const DeviceContext& context, std::span<const TriangleGeometry> triangles,
                        BuildFlags flags) {
    RequireRayQueries(context);
    const BottomInput input = Input(triangles, {});
    return Query(context, AccelerationKind::Bottom, flags, input.geometry, input.primitives);
}

AccelerationSizes Sizes(const DeviceContext& context, std::span<const BoxGeometry> boxes,
                        BuildFlags flags) {
    RequireRayQueries(context);
    const BottomInput input = Input({}, boxes);
    return Query(context, AccelerationKind::Bottom, flags, input.geometry, input.primitives);
}

AccelerationSizes Sizes(const DeviceContext& context, std::uint32_t capacity, BuildFlags flags) {
    RequireRayQueries(context);
    const VkAccelerationStructureGeometryKHR instances = Geometry(BufferView{});
    return Query(context, AccelerationKind::Top, flags, std::span(&instances, 1),
                 std::span(&capacity, 1));
}

AccelerationPool::AccelerationPool(DeviceContext& context, std::string_view name,
                                   std::uint64_t arenaSize)
    : context_(context), name_(name), arenas_(context, MemoryUse::Device, name, arenaSize),
      scratchAlignment_(QueryScratchAlignment(context)) {
    RequireRayQueries(context);
}

BufferRange AccelerationPool::Allocate(AccelerationKind kind, std::uint64_t bytes) {
    const BufferRange range = arenas_.Allocate(bytes, kStructureAlignment);
    if (kind == AccelerationKind::Bottom) {
        usage_.bottomBytes += bytes;
        ++usage_.bottoms;
    } else {
        usage_.topBytes += bytes;
        ++usage_.tops;
    }
    return range;
}

void AccelerationPool::Free(AccelerationKind kind, const BufferRange& range) {
    arenas_.Free(range);
    if (kind == AccelerationKind::Bottom) {
        usage_.bottomBytes -= range.size;
        --usage_.bottoms;
    } else {
        usage_.topBytes -= range.size;
        --usage_.tops;
    }
}

AccelerationStructure::AccelerationStructure(AccelerationPool& pool, const AccelerationSizes& sizes,
                                             std::string_view name)
    : pool_(pool), sizes_(sizes), name_(name), range_(pool.Allocate(sizes.kind, sizes.storage)) {
    DeviceContext& context = pool.Context();
    const VkAccelerationStructureCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = Storage().Handle(),
        .offset = range_.offset,
        .size = sizes.storage,
        .type = ToVk(sizes.kind)};
    const VkResult created =
        vkCreateAccelerationStructureKHR(context.Device(), &info, nullptr, &structure_);
    if (created < VK_SUCCESS) {
        pool.Free(sizes.kind, range_);
    }
    CheckVk(created, [&] { return "Creating acceleration structure " + name_; });
    context.Name(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
                 reinterpret_cast<std::uint64_t>(structure_), name);
    const VkAccelerationStructureDeviceAddressInfoKHR address{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = structure_};
    address_ = vkGetAccelerationStructureDeviceAddressKHR(context.Device(), &address);
}

AccelerationStructure::~AccelerationStructure() {
    // Work submitted so far may still trace it, or build a top level that refers to it.
    DeviceContext& context = pool_.Context();
    context.Retire([device = context.Device(), structure = structure_] {
        vkDestroyAccelerationStructureKHR(device, structure, nullptr);
    });
    pool_.Free(sizes_.kind, range_);
}

const Buffer& AccelerationStructure::Storage() const {
    return pool_.arenas_.Storage(range_);
}

std::uint64_t ScratchBytes(std::span<const BottomBuild> bottoms, std::span<const TopBuild> tops) {
    if (bottoms.empty() && tops.empty()) {
        return 0;
    }
    const std::uint64_t alignment = BuildPool(bottoms, tops).ScratchAlignment();
    // One alignment's slack places the first build's scratch, wherever `scratch` starts.
    std::uint64_t bytes = alignment;
    for (const BottomBuild& build : bottoms) {
        bytes += AlignUp(Scratch(*build.target, build.update), alignment);
    }
    for (const TopBuild& build : tops) {
        bytes += AlignUp(Scratch(*build.target, build.update), alignment);
    }
    return bytes;
}

void RecordBuilds(VkCommandBuffer commands, std::span<const BottomBuild> bottoms,
                  std::span<const TopBuild> tops, BufferView scratch) {
    if (bottoms.empty() && tops.empty()) {
        return;
    }
    const AccelerationPool& pool = BuildPool(bottoms, tops);
    const std::uint64_t alignment = pool.ScratchAlignment();
    VkDeviceAddress next = AlignUp(DeviceAddress(scratch), alignment);
    std::vector<BottomInput> inputs;
    std::vector<VkAccelerationStructureGeometryKHR> instances;
    inputs.reserve(bottoms.size());
    instances.reserve(tops.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
    std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> ranges;
    const auto add = [&](const AccelerationStructure& target, bool update,
                         std::span<const VkAccelerationStructureGeometryKHR> geometry,
                         std::span<const std::uint32_t> primitives) {
        VkAccelerationStructureBuildGeometryInfoKHR info =
            BuildInfo(target.Kind(), target.Sizes().flags, geometry);
        info.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                           : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.srcAccelerationStructure = update ? target.Handle() : VK_NULL_HANDLE;
        info.dstAccelerationStructure = target.Handle();
        info.scratchData.deviceAddress = next;
        next += AlignUp(Scratch(target, update), alignment);
        infos.push_back(info);
        std::vector<VkAccelerationStructureBuildRangeInfoKHR>& range = ranges.emplace_back();
        for (const std::uint32_t count : primitives) {
            range.push_back({.primitiveCount = count});
        }
    };
    for (const BottomBuild& build : bottoms) {
        CheckTarget(build.target, AccelerationKind::Bottom, build.update);
        const BottomInput& input = inputs.emplace_back(Input(build.triangles, build.boxes));
        CheckFits(*build.target, input.primitives);
        add(*build.target, build.update, input.geometry, input.primitives);
    }
    for (const TopBuild& build : tops) {
        CheckTarget(build.target, AccelerationKind::Top, build.update);
        if (DeviceAddress(build.instances) % kInstanceAlignment != 0) {
            throw std::invalid_argument("A top level's instances need 16-byte alignment");
        }
        CheckFits(*build.target, std::span(&build.count, 1));
        const VkAccelerationStructureGeometryKHR& geometry =
            instances.emplace_back(Geometry(build.instances));
        add(*build.target, build.update, std::span(&geometry, 1), std::span(&build.count, 1));
    }
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pointers;
    for (const auto& range : ranges) {
        pointers.push_back(range.data());
    }
    vkCmdBuildAccelerationStructuresKHR(commands, static_cast<std::uint32_t>(infos.size()),
                                        infos.data(), pointers.data());
}

CompactionQuery::CompactionQuery(DeviceContext& context, std::uint32_t capacity)
    : device_(context.Device()), capacity_(capacity) {
    const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                     .queryType =
                                         VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                                     .queryCount = capacity};
    CheckVk(vkCreateQueryPool(device_, &info, nullptr, &pool_), "Creating a compaction query");
}

CompactionQuery::~CompactionQuery() {
    vkDestroyQueryPool(device_, pool_, nullptr);
}

void CompactionQuery::Record(VkCommandBuffer commands,
                             std::span<const AccelerationStructure* const> built) {
    if (built.size() > capacity_) {
        throw std::invalid_argument("A compaction query holds " + std::to_string(capacity_) +
                                    " sizes, not " + std::to_string(built.size()));
    }
    std::vector<VkAccelerationStructureKHR> handles;
    for (const AccelerationStructure* structure : built) {
        if (!structure->Sizes().flags.compactable) {
            throw std::invalid_argument(structure->Name() + " was not built to be compacted");
        }
        handles.push_back(structure->Handle());
    }
    count_ = static_cast<std::uint32_t>(handles.size());
    vkCmdResetQueryPool(commands, pool_, 0, capacity_);
    if (count_ > 0) {
        vkCmdWriteAccelerationStructuresPropertiesKHR(
            commands, count_, handles.data(),
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, pool_, 0);
    }
}

std::optional<std::vector<std::uint64_t>> CompactionQuery::Sizes() const {
    // Each size followed by its availability.
    std::vector<std::uint64_t> results(2 * std::size_t{count_});
    const VkResult result =
        count_ == 0
            ? VK_SUCCESS
            : vkGetQueryPoolResults(device_, pool_, 0, count_,
                                    results.size() * sizeof(std::uint64_t), results.data(),
                                    2 * sizeof(std::uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result < VK_SUCCESS) {
        CheckVk(result, "Reading compacted sizes");
    }
    std::vector<std::uint64_t> sizes;
    for (std::uint32_t i = 0; i < count_; ++i) {
        if (results[2 * i + 1] == 0) {
            return std::nullopt;
        }
        sizes.push_back(results[2 * i]);
    }
    return sizes;
}

void RecordCompaction(VkCommandBuffer commands, const AccelerationStructure& source,
                      const AccelerationStructure& compacted) {
    const VkCopyAccelerationStructureInfoKHR copy{
        .sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
        .src = source.Handle(),
        .dst = compacted.Handle(),
        .mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR};
    vkCmdCopyAccelerationStructureKHR(commands, &copy);
}

}  // namespace ernie
