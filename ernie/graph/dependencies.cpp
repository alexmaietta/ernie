// SPDX-License-Identifier: BSD-3-Clause
/// @file dependencies.cpp
/// @brief Uses as stages, accesses and layouts, and the barriers a pass order needs.
#include "ernie/graph/dependencies.h"

#include <algorithm>
#include <cstddef>

namespace ernie::graph {
namespace {

constexpr VkAccessFlags2 kWriteAccess =
    VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

/// A pass's accesses to one resource, merged.
struct Merged {
    std::uint32_t resource = 0;
    UseState state;
    Layers layers;
    bool valid = true;
};

/// What a resource's earlier accesses left to synchronize with.
struct ResourceState {
    VkPipelineStageFlags2 writeStages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 writeAccess = VK_ACCESS_2_NONE;
    /// Stages that read since the last write.
    VkPipelineStageFlags2 readStages = VK_PIPELINE_STAGE_2_NONE;
    /// The stage and access pairs the last write is visible to.
    std::vector<UseState> visible;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    /// Whether it holds content, and whether this graph wrote it.
    bool written = false;
    bool writtenHere = false;
    bool touched = false;
};

VkPipelineStageFlags2 ShaderStages(Kind kind) {
    switch (kind) {
    case Kind::Compute:
        return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    case Kind::Raster:
        // Vertex shaders sample too, such as terrain heights.
        return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    case Kind::RayTracing:
        return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    case Kind::Copy:
    case Kind::Build:
        break;
    }
    return VK_PIPELINE_STAGE_2_NONE;
}

std::uint64_t LayerEnd(Layers layers) {
    return layers.count == Layers::kAll ? ~std::uint64_t{0}
                                        : std::uint64_t{layers.first} + layers.count;
}

Layers Union(Layers a, Layers b) {
    const std::uint32_t first = std::min(a.first, b.first);
    const std::uint64_t end = std::max(LayerEnd(a), LayerEnd(b));
    if (end == ~std::uint64_t{0}) {
        return {first, Layers::kAll};
    }
    return {first, static_cast<std::uint32_t>(end - first)};
}

/// The layout two uses in one pass share, or undefined if none.
VkImageLayout Combine(VkImageLayout a, VkImageLayout b) {
    if (a == b) {
        return a;
    }
    // Attachments that are read as depth, and sampled too, share the read-only depth layout.
    const auto depthRead = [](VkImageLayout l) {
        return l == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL || l == VK_IMAGE_LAYOUT_GENERAL;
    };
    if (depthRead(a) && depthRead(b)) {
        return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    }
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

bool Covers(const std::vector<UseState>& visible, const UseState& use) {
    return std::ranges::any_of(visible, [&](const UseState& seen) {
        return (use.stages & ~seen.stages) == 0 && (use.access & ~seen.access) == 0;
    });
}

std::vector<Merged> Merge(const DependencyPass& pass, std::span<const DependencyResource> resources,
                          std::vector<std::string>& errors) {
    std::vector<Merged> merged;
    for (const DependencyAccess& access : pass.accesses) {
        if (access.resource >= resources.size()) {
            errors.push_back("pass '" + pass.name + "' uses an unknown resource");
            continue;
        }
        const DependencyResource& resource = resources[access.resource];
        const UseState state = ToState(pass.kind, access.use, resource.image);
        if (state.stages == VK_PIPELINE_STAGE_2_NONE) {
            errors.push_back("pass '" + pass.name + "' cannot use " +
                             (resource.image ? "image '" : "buffer '") + resource.name + "' as " +
                             ToString(access.use));
            continue;
        }
        const auto it = std::ranges::find(merged, access.resource, &Merged::resource);
        if (it == merged.end()) {
            merged.push_back({access.resource, state, access.layers});
            continue;
        }
        it->state.stages |= state.stages;
        it->state.access |= state.access;
        it->state.writes = it->state.writes || state.writes;
        it->state.reads = it->state.reads || state.reads;
        it->layers = Union(it->layers, access.layers);
        if (resource.image) {
            const VkImageLayout layout = Combine(it->state.layout, state.layout);
            if (layout == VK_IMAGE_LAYOUT_UNDEFINED && it->valid) {
                errors.push_back("pass '" + pass.name + "' uses image '" + resource.name +
                                 "' in conflicting layouts");
                it->valid = false;
            }
            it->state.layout = layout;
        }
    }
    return merged;
}

void Initialize(ResourceState& state, const DependencyResource& resource) {
    switch (resource.origin) {
    case Origin::Imported:
        // Anything before the graph may have written it.
        state.writeStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        state.writeAccess = VK_ACCESS_2_MEMORY_WRITE_BIT;
        state.written = true;
        break;
    case Origin::Recycled:
        // An earlier graph's reads may still be running.
        state.readStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        break;
    case Origin::Fresh:
        break;
    }
}

void AddBarrier(Dependency& batch, const DependencyResource& resource, std::uint32_t index,
                VkPipelineStageFlags2 srcStages, VkAccessFlags2 srcAccess, const Merged& use,
                VkImageLayout oldLayout) {
    Barrier barrier{.resource = index,
                    .srcStages = srcStages,
                    .srcAccess = srcAccess,
                    .dstStages = use.state.stages,
                    .dstAccess = use.state.access};
    if (resource.image) {
        barrier.oldLayout = oldLayout;
        barrier.newLayout = use.state.layout;
        // Layer ranges narrow memory barriers only; layouts change for the whole image.
        barrier.layers = oldLayout == use.state.layout ? use.layers : Layers{};
        batch.images.push_back(barrier);
    } else {
        batch.buffers.push_back(barrier);
    }
}

void Analyze(const Merged& use, const DependencyResource& resource, ResourceState& state,
             Dependency& before) {
    const std::uint32_t index = use.resource;
    state.touched = true;
    const bool layoutChange = resource.image && use.state.layout != state.layout;
    if (layoutChange) {
        AddBarrier(before, resource, index, state.writeStages | state.readStages, state.writeAccess,
                   use, state.layout);
        state.layout = use.state.layout;
    } else if (use.state.writes) {
        const VkPipelineStageFlags2 src = state.writeStages | state.readStages;
        if (src != VK_PIPELINE_STAGE_2_NONE) {
            AddBarrier(before, resource, index, src, state.writeAccess, use, state.layout);
        }
    } else if (!Covers(state.visible, use.state)) {
        if (state.writeStages != VK_PIPELINE_STAGE_2_NONE) {
            AddBarrier(before, resource, index, state.writeStages, state.writeAccess, use,
                       state.layout);
        }
        state.visible.push_back(use.state);
    }
    if (use.state.writes) {
        state.writeStages = use.state.stages;
        state.writeAccess = use.state.access & kWriteAccess;
        state.readStages = VK_PIPELINE_STAGE_2_NONE;
        state.visible.clear();
        state.written = true;
        state.writtenHere = true;
    } else {
        if (layoutChange) {
            // The transition is a write ordered before these stages and already available.
            state.writeStages = use.state.stages;
            state.writeAccess = VK_ACCESS_2_NONE;
            state.readStages = VK_PIPELINE_STAGE_2_NONE;
            state.visible.assign(1, use.state);
        }
        state.readStages |= use.state.stages;
    }
}

void Finish(std::span<const DependencyResource> resources, std::span<const ResourceState> states,
            Dependency& after) {
    for (std::size_t i = 0; i < resources.size(); ++i) {
        const DependencyResource& resource = resources[i];
        const ResourceState& state = states[i];
        if (resource.origin != Origin::Imported || !state.touched) {
            continue;
        }
        const auto index = static_cast<std::uint32_t>(i);
        Merged use{.resource = index,
                   .state = {.stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             .access = kAnyAccess,
                             .layout = VK_IMAGE_LAYOUT_GENERAL}};
        if (resource.image && state.layout != VK_IMAGE_LAYOUT_GENERAL) {
            // Back in the general layout, as every import enters.
            AddBarrier(after, resource, index, state.writeStages | state.readStages,
                       state.writeAccess, use, state.layout);
        } else if (state.writtenHere && state.writeAccess != VK_ACCESS_2_NONE) {
            if (!resource.image) {
                // Readbacks: the host reads once the ticket is reached.
                use.state.stages |= VK_PIPELINE_STAGE_2_HOST_BIT;
                use.state.access |= VK_ACCESS_2_HOST_READ_BIT;
            }
            AddBarrier(after, resource, index, state.writeStages, state.writeAccess, use,
                       state.layout);
        }
    }
}

}  // namespace

UseState ToState(Kind kind, Use use, bool image) {
    const VkPipelineStageFlags2 shader = ShaderStages(kind);
    const VkPipelineStageFlags2 transfer = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    const VkPipelineStageFlags2 build = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    const VkPipelineStageFlags2 depth =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    const bool raster = kind == Kind::Raster;
    const VkImageLayout general = image ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    UseState s;
    switch (use) {
    case Use::Sampled:
        if (image) {
            s = {shader, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, general, false, true};
        }
        break;
    case Use::StorageRead:
        s = {shader, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, general, false, true};
        break;
    case Use::StorageWrite:
        s = {shader, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, general, true, false};
        break;
    case Use::StorageReadWrite:
        s = {shader, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
             general, true, true};
        break;
    case Use::ColorWrite:
        if (image && raster) {
            return {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, true, false};
        }
        return {};
    case Use::ColorReadWrite:
        if (image && raster) {
            return {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, true, true};
        }
        return {};
    case Use::DepthWrite:
        if (image && raster) {
            return {depth, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, true, false};
        }
        return {};
    case Use::DepthReadWrite:
        if (image && raster) {
            return {depth,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, true, true};
        }
        return {};
    case Use::DepthRead:
        if (image && raster) {
            return {depth, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, false, true};
        }
        return {};
    case Use::CopySrc:
        return {transfer, VK_ACCESS_2_TRANSFER_READ_BIT, general, false, true};
    case Use::CopyDst:
        return {transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT, general, true, false};
    case Use::Uniform:
        if (!image) {
            s = {shader, VK_ACCESS_2_UNIFORM_READ_BIT, general, false, true};
        }
        break;
    case Use::Indirect:
        if (!image && (kind == Kind::Compute || raster || kind == Kind::RayTracing)) {
            return {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                    general, false, true};
        }
        return {};
    case Use::HostRead:
        if (!image) {
            return {VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT, general, false, true};
        }
        return {};
    case Use::BuildInput:
        if (!image && kind == Kind::Build) {
            return {build,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                    general, false, true};
        }
        return {};
    case Use::BuildScratch:
        // Read and written within the build; what it held before never matters.
        if (!image && kind == Kind::Build) {
            return {build,
                    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                    general, true, false};
        }
        return {};
    case Use::BuildOutput:
        if (!image && kind == Kind::Build) {
            return {build, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, general, true, false};
        }
        return {};
    case Use::Trace:
        if (!image) {
            s = {shader, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, general, false, true};
        }
        break;
    }
    // Shader uses need a pass that has shaders.
    if (shader == VK_PIPELINE_STAGE_2_NONE) {
        return {};
    }
    return s;
}

PassDependencies ComputeDependencies(std::span<const DependencyResource> resources,
                                     std::span<const DependencyPass> passes) {
    PassDependencies dependencies;
    std::vector<ResourceState> states(resources.size());
    for (std::size_t i = 0; i < resources.size(); ++i) {
        Initialize(states[i], resources[i]);
    }
    dependencies.passes.resize(passes.size());
    for (std::size_t p = 0; p < passes.size(); ++p) {
        const DependencyPass& pass = passes[p];
        for (const Merged& use : Merge(pass, resources, dependencies.errors)) {
            const DependencyResource& resource = resources[use.resource];
            ResourceState& state = states[use.resource];
            if (use.state.reads && !state.written) {
                dependencies.errors.push_back("pass '" + pass.name + "' reads '" + resource.name +
                                              "', which no earlier pass writes and which is "
                                              "not imported");
            }
            if (use.valid) {
                Analyze(use, resource, state, dependencies.passes[p]);
            }
        }
    }
    Finish(resources, states, dependencies.after);
    return dependencies;
}

const char* ToString(Use use) {
    switch (use) {
    case Use::Sampled:
        return "Sampled";
    case Use::StorageRead:
        return "StorageRead";
    case Use::StorageWrite:
        return "StorageWrite";
    case Use::StorageReadWrite:
        return "StorageReadWrite";
    case Use::ColorWrite:
        return "ColorWrite";
    case Use::ColorReadWrite:
        return "ColorReadWrite";
    case Use::DepthWrite:
        return "DepthWrite";
    case Use::DepthReadWrite:
        return "DepthReadWrite";
    case Use::DepthRead:
        return "DepthRead";
    case Use::CopySrc:
        return "CopySrc";
    case Use::CopyDst:
        return "CopyDst";
    case Use::Uniform:
        return "Uniform";
    case Use::Indirect:
        return "Indirect";
    case Use::HostRead:
        return "HostRead";
    case Use::BuildInput:
        return "BuildInput";
    case Use::BuildScratch:
        return "BuildScratch";
    case Use::BuildOutput:
        return "BuildOutput";
    case Use::Trace:
        return "Trace";
    }
    return "?";
}

}  // namespace ernie::graph
