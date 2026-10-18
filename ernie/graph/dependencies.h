// SPDX-License-Identifier: BSD-3-Clause
/// @file dependencies.h
/// @brief The frame graph's dependency analysis: the synchronization2 barriers between passes in
/// their order, computed on the host and tested without a device.
///
/// Images stay in the general layout, which the bindless heap's descriptors name, except while a
/// raster pass uses them as attachments. Buffers are ranges of arenas reached by address, so
/// their dependencies are recorded as memory barriers.
///
/// References:
/// - H.-K. Arntzen, "Render graphs and Vulkan — a deep dive", 2017.
///   https://themaister.net/blog/2017/08/15/render-graphs-and-vulkan-a-deep-dive/
/// - Khronos, "Synchronization Examples".
///   https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
#pragma once

#include "ernie/device/vulkan.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ernie::graph {

/// @brief How a pass uses a resource; with the pass's `Kind`, it decides stages, accesses and
/// image layouts.
enum class Use : std::uint8_t {
    /// Sampled through a texture descriptor.
    Sampled,
    /// Read as a storage image or buffer.
    StorageRead,
    /// Written as a storage image or buffer.
    StorageWrite,
    /// Read and written as a storage image or buffer.
    StorageReadWrite,
    /// A colour attachment, cleared or overwritten.
    ColorWrite,
    /// A colour attachment, loaded or blended.
    ColorReadWrite,
    /// A depth attachment, cleared or overwritten.
    DepthWrite,
    /// A depth attachment, loaded and written.
    DepthReadWrite,
    /// A depth attachment, tested only.
    DepthRead,
    /// The source of a copy or blit.
    CopySrc,
    /// The destination of a copy, blit or fill.
    CopyDst,
    /// A uniform buffer.
    Uniform,
    /// Indirect draw or dispatch arguments.
    Indirect,
    /// Read by the host once the graph's ticket is reached.
    HostRead,
    /// Read by an acceleration structure build: geometry, instances, or the structures a
    /// top-level one refers to.
    BuildInput,
    /// A build's scratch memory, read and written.
    BuildScratch,
    /// The acceleration structure a build writes.
    BuildOutput,
    /// An acceleration structure that ray queries or ray tracing shaders trace.
    Trace,
};

/// @brief What a pass records, deciding its stages.
enum class Kind : std::uint8_t {
    /// Dispatches.
    Compute,
    /// Draws within dynamic rendering.
    Raster,
    /// Copies, blits and fills.
    Copy,
    /// Acceleration structure builds.
    Build,
    /// Ray tracing pipeline traces.
    RayTracing,
};

/// @brief How a shader reaches a resource. The graph treats all alike; the distinction
/// documents accesses no descriptor shows.
enum class Reach : std::uint8_t {
    Descriptor,
    /// An index into the bindless heap.
    Bindless,
    /// A buffer device address.
    Address,
};

/// @brief A range of array layers; mips are always whole.
struct Layers {
    std::uint32_t first = 0;
    /// The layer count; `kAll` for the rest of the image.
    std::uint32_t count = kAll;
    static constexpr std::uint32_t kAll = ~0u;
};

/// @brief Where a resource's state comes from when a graph starts.
enum class Origin : std::uint8_t {
    /// Owned outside the graph; written by anything earlier.
    Imported,
    /// A new transient: nothing touched it.
    Fresh,
    /// A pooled transient an earlier graph used.
    Recycled,
};

/// @brief A resource as dependency analysis sees it.
struct DependencyResource {
    std::string name;
    bool image = false;
    Origin origin = Origin::Fresh;
};

/// @brief One declared access of a pass.
struct DependencyAccess {
    std::uint32_t resource = 0;
    Use use = Use::Sampled;
    Layers layers;
    Reach reach = Reach::Descriptor;
};

/// @brief A pass as dependency analysis sees it.
struct DependencyPass {
    std::string name;
    Kind kind = Kind::Compute;
    std::vector<DependencyAccess> accesses;
};

/// @brief The stages, accesses and layout of one use.
struct UseState {
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    /// The image layout; undefined for buffers.
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool writes = false;
    bool reads = false;
};

/// @brief A barrier for one resource: an image barrier for an image, folded into the pass's
/// memory barrier for a buffer.
struct Barrier {
    std::uint32_t resource = 0;
    VkPipelineStageFlags2 srcStages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 dstAccess = VK_ACCESS_2_NONE;
    /// An image's layouts before and after; equal for no transition.
    VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    /// The image layers covered; whole for a layout change.
    Layers layers;
};

/// @brief The barriers recorded together before a pass, or after the last.
struct Dependency {
    std::vector<Barrier> images;
    std::vector<Barrier> buffers;

    bool Empty() const {
        return images.empty() && buffers.empty();
    }
};

/// @brief The dependencies of a whole graph.
struct PassDependencies {
    /// The dependency before each pass, in order.
    std::vector<Dependency> passes;
    /// After the last pass: imported images back in the general layout, and imported writes
    /// visible to later commands.
    Dependency after;
    /// Problems found; the result is unusable unless empty.
    std::vector<std::string> errors;
};

/// @return The stages, accesses and layout of a use; no stages when `kind` cannot make it.
UseState ToState(Kind kind, Use use, bool image);

/// @brief The dependency before each pass, in order: a memory barrier from the last write for a
/// read, an execution dependency from earlier reads for a write, and an image barrier for a
/// layout change. Reads already made visible get none.
/// @return The dependencies, with errors for reads of unwritten transients, uses a resource
/// cannot have and conflicting layouts.
PassDependencies ComputeDependencies(std::span<const DependencyResource> resources,
                                     std::span<const DependencyPass> passes);

/// @return A use's name, such as "StorageWrite".
const char* ToString(Use use);

}  // namespace ernie::graph
