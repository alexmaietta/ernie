// SPDX-License-Identifier: BSD-3-Clause
/// @file graph.cpp
/// @brief Declaring, culling, backing and recording the frame graph's passes.
#include "ernie/graph/graph.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string_view>

#ifndef ERNIE_GRAPH_FULL_BARRIERS
#define ERNIE_GRAPH_FULL_BARRIERS 0
#endif

namespace ernie::graph {
namespace {

bool FullBarriersByDefault() {
#if ERNIE_GRAPH_FULL_BARRIERS
    return true;
#else
    const char* value = std::getenv("ERNIE_GRAPH_FULL_BARRIERS");
    return value != nullptr && std::string_view(value) == "1";
#endif
}

bool Writes(Use use) {
    switch (use) {
    case Use::StorageWrite:
    case Use::StorageReadWrite:
    case Use::ColorWrite:
    case Use::ColorReadWrite:
    case Use::DepthWrite:
    case Use::DepthReadWrite:
    case Use::CopyDst:
    case Use::BuildScratch:
    case Use::BuildOutput:
        return true;
    default:
        return false;
    }
}

bool Reads(Use use) {
    return !Writes(use) || use == Use::StorageReadWrite || use == Use::ColorReadWrite ||
           use == Use::DepthReadWrite;
}

ImageUsage NeededUsage(Use use) {
    switch (use) {
    case Use::Sampled:
        return ImageUsage::Sampled;
    case Use::StorageRead:
    case Use::StorageWrite:
    case Use::StorageReadWrite:
        return ImageUsage::Storage;
    case Use::ColorWrite:
    case Use::ColorReadWrite:
    case Use::DepthWrite:
    case Use::DepthReadWrite:
    case Use::DepthRead:
        return ImageUsage::Attachment;
    default:
        // Every image takes transfer usage.
        return ImageUsage{};
    }
}

}  // namespace

std::uint32_t PassBuilder::Declare(std::uint32_t id, std::uint32_t version, Shape shape, Use use,
                                   Layers layers, Reach reach, bool write) {
    const auto resourceShape = [](const Graph::Resource& resource) {
        return resource.image         ? Shape::Image
               : resource.Structure() ? Shape::Acceleration
                                      : Shape::Buffer;
    };
    if (id >= graph_.resources_.size() || resourceShape(graph_.resources_[id]) != shape) {
        static constexpr const char* kShapes[] = {"image", "buffer", "acceleration structure"};
        throw std::invalid_argument(std::string("An invalid ") +
                                    kShapes[static_cast<std::size_t>(shape)] + " handle");
    }
    Graph::Resource& resource = graph_.resources_[id];
    Graph::Pass& pass = graph_.passes_[pass_];
    if (Writes(use) != write) {
        throw std::invalid_argument("Pass '" + pass.name + "' declares " + ToString(use) + " on '" +
                                    resource.name + "' as a " + (write ? "write" : "read"));
    }
    // Structures are built and traced; the memory they live in is no plain buffer, and the other
    // way round.
    const bool structureUse = use == Use::BuildOutput || use == Use::Trace;
    if (structureUse != (shape == Shape::Acceleration) && use != Use::BuildInput) {
        throw std::invalid_argument(
            "Pass '" + pass.name + "' declares " + ToString(use) + " on '" + resource.name +
            "', which " +
            (structureUse ? "is no acceleration structure" : "is an acceleration structure"));
    }
    if (version > resource.version) {
        throw GraphError("Pass '" + pass.name + "' uses a version of '" + resource.name +
                         "' not written yet");
    }
    pass.accesses.push_back({id, use, layers, reach});
    if (write) {
        ++resource.version;
    }
    return resource.version;
}

Image PassBuilder::Read(Image image, Use use, Layers layers) {
    Declare(image.id, image.version, Shape::Image, use, layers, Reach::Descriptor, false);
    return image;
}

Image PassBuilder::Write(Image image, Use use, Layers layers) {
    return {image.id,
            Declare(image.id, image.version, Shape::Image, use, layers, Reach::Descriptor, true)};
}

Buffer PassBuilder::Read(Buffer buffer, Use use) {
    Declare(buffer.id, buffer.version, Shape::Buffer, use, {}, Reach::Descriptor, false);
    return buffer;
}

Buffer PassBuilder::Write(Buffer buffer, Use use) {
    return {buffer.id,
            Declare(buffer.id, buffer.version, Shape::Buffer, use, {}, Reach::Descriptor, true)};
}

Image PassBuilder::ReadBindless(Image image, Use use, Layers layers) {
    Declare(image.id, image.version, Shape::Image, use, layers, Reach::Bindless, false);
    return image;
}

Buffer PassBuilder::ReadAddress(Buffer buffer, Use use) {
    Declare(buffer.id, buffer.version, Shape::Buffer, use, {}, Reach::Address, false);
    return buffer;
}

Buffer PassBuilder::WriteAddress(Buffer buffer, Use use) {
    return {buffer.id,
            Declare(buffer.id, buffer.version, Shape::Buffer, use, {}, Reach::Address, true)};
}

Acceleration PassBuilder::Build(Acceleration structure) {
    return {structure.id, Declare(structure.id, structure.version, Shape::Acceleration,
                                  Use::BuildOutput, {}, Reach::Address, true)};
}

Acceleration PassBuilder::Read(Acceleration structure, Use use) {
    if (use != Use::Trace && use != Use::BuildInput) {
        throw std::invalid_argument(std::string("An acceleration structure is traced or read by "
                                                "builds, not used as ") +
                                    ToString(use));
    }
    Declare(structure.id, structure.version, Shape::Acceleration, use, {}, Reach::Address, false);
    return structure;
}

void PassBuilder::NeverCull() {
    graph_.passes_[pass_].neverCull = true;
}

void PassContext::Check(std::uint32_t id) const {
    const Graph::Pass& pass = graph_.passes_[pass_];
    if (std::ranges::none_of(
            pass.accesses, [&](const DependencyAccess& access) { return access.resource == id; })) {
        throw std::logic_error(
            "Pass '" + pass.name + "' did not declare '" +
            (id < graph_.resources_.size() ? graph_.resources_[id].name : std::string("?")) + "'");
    }
}

DeviceImage& PassContext::Get(Image image) const {
    Check(image.id);
    return *graph_.resources_[image.id].deviceImage;
}

BufferSlice PassContext::Get(Buffer buffer) const {
    Check(buffer.id);
    return graph_.resources_[buffer.id].slice;
}

const AccelerationStructure& PassContext::Get(Acceleration structure) const {
    Check(structure.id);
    const Graph::Resource& resource = graph_.resources_[structure.id];
    if (resource.acceleration == nullptr) {
        throw std::logic_error("'" + resource.name + "' is a pool of structures, not one");
    }
    return *resource.acceleration;
}

VkImageLayout PassContext::Layout(Image image) const {
    Check(image.id);
    const Graph::Pass& pass = graph_.passes_[pass_];
    const auto access = std::ranges::find(pass.accesses, image.id, &DependencyAccess::resource);
    return ToState(pass.kind, access->use, true).layout;
}

void PassContext::Clear(Image image, const glm::vec4& colour) const {
    const VkClearColorValue clear{.float32 = {colour.r, colour.g, colour.b, colour.a}};
    const VkImageSubresourceRange all{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
    vkCmdClearColorImage(commands.Handle(), Get(image).Handle(), Layout(image), &clear, 1, &all);
}

DeviceContext& PassContext::Context() const {
    return graph_.context_;
}

void PassContext::Keep(std::shared_ptr<const void> object) const {
    graph_.kept_.push_back(std::move(object));
}

Graph::Graph(DeviceContext& context, TransientPool& pool)
    : context_(context), pool_(pool), fullBarriers_(FullBarriersByDefault()) {}

Graph::~Graph() {
    ReleaseLeases();
}

std::uint32_t Graph::AddResource(Resource resource) {
    resources_.push_back(std::move(resource));
    return static_cast<std::uint32_t>(resources_.size() - 1);
}

Image Graph::Import(DeviceImage& image, std::string_view name) {
    const auto found = std::ranges::find(resources_, &image, &Resource::deviceImage);
    if (found != resources_.end()) {
        return {static_cast<std::uint32_t>(found - resources_.begin()), found->version};
    }
    return {AddResource({.name = std::string(name),
                         .image = true,
                         .imported = true,
                         .deviceImage = &image,
                         .origin = Origin::Imported}),
            0};
}

Buffer Graph::Import(const DeviceColumn& column, std::string_view name) {
    return Import(column.Storage(), column.Range().offset, column.Bytes(),
                  name.empty() ? column.Name() : name);
}

Acceleration Graph::Import(const AccelerationStructure& structure) {
    const auto found = std::ranges::find(resources_, &structure, &Resource::acceleration);
    if (found != resources_.end()) {
        return {static_cast<std::uint32_t>(found - resources_.begin()), found->version};
    }
    return {AddResource(
                {.name = structure.Name(),
                 .imported = true,
                 .slice = {&structure.Storage(), structure.Range().offset, structure.Range().size},
                 .acceleration = &structure,
                 .origin = Origin::Imported}),
            0};
}

Acceleration Graph::Import(const AccelerationPool& pool) {
    const auto found = std::ranges::find(resources_, &pool, &Resource::structures);
    if (found != resources_.end()) {
        return {static_cast<std::uint32_t>(found - resources_.begin()), found->version};
    }
    return {AddResource({.name = pool.Name(),
                         .imported = true,
                         .structures = &pool,
                         .origin = Origin::Imported}),
            0};
}

Buffer Graph::Import(const ernie::Buffer& buffer, std::uint64_t offset, std::uint64_t size,
                     std::string_view name) {
    const auto found = std::ranges::find_if(resources_, [&](const Resource& resource) {
        return resource.imported && !resource.image && !resource.Structure() &&
               resource.slice.buffer == &buffer && resource.slice.offset == offset &&
               resource.slice.size == size;
    });
    if (found != resources_.end()) {
        return {static_cast<std::uint32_t>(found - resources_.begin()), found->version};
    }
    return {AddResource({.name = std::string(name),
                         .imported = true,
                         .slice = {&buffer, offset, size},
                         .origin = Origin::Imported}),
            0};
}

Image Graph::Create(const ImageDescription& description, std::string_view name) {
    return {AddResource({.name = std::string(name), .image = true, .description = description}), 0};
}

Buffer Graph::Create(std::uint64_t bytes, std::string_view name) {
    return {AddResource({.name = std::string(name), .bytes = bytes}), 0};
}

std::uint32_t Graph::NewPass(std::string_view name, Kind kind) {
    if (compiled_) {
        throw std::logic_error("Passes cannot be added after Compile");
    }
    passes_.push_back({.name = std::string(name), .kind = kind, .scope = scope_});
    return static_cast<std::uint32_t>(passes_.size() - 1);
}

Graph::ScopeGuard::~ScopeGuard() {
    graph_.scope_ = graph_.scopes_[graph_.scope_].parent;
}

Graph::ScopeGuard Graph::Scope(std::string_view name) {
    const std::uint32_t depth = scope_ == kNoScope ? 0 : scopes_[scope_].depth + 1;
    scopes_.push_back({.name = std::string(name), .parent = scope_, .depth = depth});
    scope_ = static_cast<std::uint32_t>(scopes_.size() - 1);
    return ScopeGuard(*this);
}

std::vector<std::uint32_t> Graph::LivePasses() const {
    // Walk back from the passes that must run, keeping passes that write what a kept pass reads.
    std::vector<bool> needed(resources_.size(), false);
    std::vector<bool> keep(passes_.size(), false);
    for (std::size_t p = passes_.size(); p-- > 0;) {
        const Pass& pass = passes_[p];
        const bool live =
            pass.neverCull || std::ranges::any_of(pass.accesses, [&](const DependencyAccess& a) {
                return Writes(a.use) && (resources_[a.resource].imported || needed[a.resource]);
            });
        if (!live) {
            continue;
        }
        keep[p] = true;
        for (const DependencyAccess& access : pass.accesses) {
            if (Reads(access.use)) {
                needed[access.resource] = true;
            }
        }
    }
    std::vector<std::uint32_t> live;
    for (std::size_t p = 0; p < passes_.size(); ++p) {
        if (keep[p]) {
            live.push_back(static_cast<std::uint32_t>(p));
        }
    }
    return live;
}

void Graph::Lease(const std::vector<std::uint32_t>& live) {
    std::vector<bool> used(resources_.size(), false);
    std::vector<ImageUsage> usage(resources_.size(), ImageUsage{});
    for (const std::uint32_t p : live) {
        for (const DependencyAccess& access : passes_[p].accesses) {
            used[access.resource] = true;
            usage[access.resource] = usage[access.resource] | NeededUsage(access.use);
        }
    }
    for (std::size_t i = 0; i < resources_.size(); ++i) {
        Resource& resource = resources_[i];
        if (resource.imported || !used[i]) {
            continue;
        }
        if (resource.image) {
            ImageDescription description = resource.description;
            description.usage = description.usage | usage[i];
            const TransientPool::ImageLease lease = pool_.AcquireImage(description);
            resource.deviceImage = lease.image;
            resource.origin = lease.recycled ? Origin::Recycled : Origin::Fresh;
        } else {
            // Freed ranges come back only once the device is done with them.
            resource.range = pool_.AcquireBuffer(resource.bytes);
            resource.slice = {&pool_.Buffers().Storage(resource.range), resource.range.offset,
                              resource.bytes};
            resource.origin = Origin::Fresh;
        }
        leased_ = true;
    }
}

void Graph::ReleaseLeases() {
    if (!leased_) {
        return;
    }
    for (Resource& resource : resources_) {
        if (resource.imported) {
            continue;
        }
        if (resource.deviceImage != nullptr) {
            pool_.Release(*resource.deviceImage);
            resource.deviceImage = nullptr;
        }
        if (resource.slice.buffer != nullptr) {
            pool_.Release(resource.range);
            resource.slice = {};
        }
    }
    leased_ = false;
}

void Graph::Compile() {
    if (compiled_) {
        return;
    }
    ERNIE_ZONE("Compile the frame graph");
    live_ = LivePasses();
    Lease(live_);
    std::vector<DependencyResource> resources;
    resources.reserve(resources_.size());
    for (const Resource& resource : resources_) {
        resources.push_back({resource.name, resource.image, resource.origin});
    }
    std::vector<DependencyPass> passes;
    passes.reserve(live_.size());
    for (const std::uint32_t p : live_) {
        passes.push_back({passes_[p].name, passes_[p].kind, passes_[p].accesses});
    }
    dependencies_ = ComputeDependencies(resources, passes);
    if (!dependencies_.errors.empty()) {
        ReleaseLeases();
        std::string message = "The frame graph cannot run:";
        for (const std::string& error : dependencies_.errors) {
            message += "\n  " + error;
        }
        throw GraphError(message);
    }
    compiled_ = true;
}

void Graph::Record(VkCommandBuffer commands, const Dependency& dependency) const {
    if (dependency.Empty()) {
        return;
    }
    std::vector<VkImageMemoryBarrier2> images;
    images.reserve(dependency.images.size());
    for (const Barrier& b : dependency.images) {
        const Layers layers = b.layers;
        images.push_back({.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                          .srcStageMask = b.srcStages,
                          .srcAccessMask = b.srcAccess,
                          .dstStageMask = b.dstStages,
                          .dstAccessMask = b.dstAccess,
                          .oldLayout = b.oldLayout,
                          .newLayout = b.newLayout,
                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .image = resources_[b.resource].deviceImage->Handle(),
                          .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                               .levelCount = VK_REMAINING_MIP_LEVELS,
                                               .baseArrayLayer = layers.first,
                                               .layerCount = layers.count == Layers::kAll
                                                                 ? VK_REMAINING_ARRAY_LAYERS
                                                                 : layers.count}});
    }
    // Buffers are ranges reached by address: their dependencies are one memory barrier.
    VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    for (const Barrier& b : dependency.buffers) {
        memory.srcStageMask |= b.srcStages;
        memory.srcAccessMask |= b.srcAccess;
        memory.dstStageMask |= b.dstStages;
        memory.dstAccessMask |= b.dstAccess;
    }
    const VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .memoryBarrierCount = dependency.buffers.empty() ? 0u : 1u,
                                .pMemoryBarriers = &memory,
                                .imageMemoryBarrierCount =
                                    static_cast<std::uint32_t>(images.size()),
                                .pImageMemoryBarriers = images.data()};
    vkCmdPipelineBarrier2(commands, &info);
}

void Graph::RecordPass(VkCommandBuffer commands, std::uint32_t order, std::uint32_t depth) const {
    const Pass& pass = passes_[live_[order]];
    // Scopes open outermost first, each zone living on the stack of the call nested in it.
    std::uint32_t scope = pass.scope;
    while (scope != kNoScope && scopes_[scope].depth > depth) {
        scope = scopes_[scope].parent;
    }
    if (scope != kNoScope && scopes_[scope].depth == depth) {
        ERNIE_GPU_ZONE(context_, commands, scopes_[scope].name.c_str());
        RecordPass(commands, order, depth + 1);
        return;
    }
    ERNIE_ZONE("Record a pass");
    ERNIE_ZONE_TEXT(pass.name);
    ERNIE_GPU_ZONE(context_, commands, pass.name.c_str());
    Record(commands, dependencies_.passes[order]);
    if (fullBarriers_) {
        ernie::Barrier(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, kAnyAccess);
    }
    PassContext context(*this, live_[order]);
    context.commands = commands;
    if (pass.run) {
        pass.run(context);
    }
}

Ticket Graph::Execute(Queue& queue, std::span<const Ticket> waits, BinarySemaphores binary) {
    if (executed_) {
        throw std::logic_error("A frame graph executes once");
    }
    Compile();
    executed_ = true;
    // Transient images new to the pool are made ready on the main queue.
    std::vector<Ticket> after(waits.begin(), waits.end());
    for (const Resource& resource : resources_) {
        if (resource.image && resource.deviceImage != nullptr) {
            after.push_back(resource.deviceImage->Ready());
        }
    }
    Ticket ticket;
    if (flushEachPass_) {
        // The first submission waits on the binary semaphore; the last signals its own.
        for (std::uint32_t order = 0; order < live_.size(); ++order) {
            ticket = queue.Submit([&](VkCommandBuffer commands) { RecordPass(commands, order); },
                                  after, {.wait = std::exchange(binary.wait, VK_NULL_HANDLE)});
            ticket.Wait();
        }
    }
    ticket = queue.Submit(
        [&](VkCommandBuffer commands) {
            if (!flushEachPass_) {
                for (std::uint32_t order = 0; order < live_.size(); ++order) {
                    RecordPass(commands, order);
                }
            }
            Record(commands, dependencies_.after);
        },
        after, binary);
    ReleaseLeases();
    for (std::shared_ptr<const void>& object : kept_) {
        context_.Retire(std::move(object));
    }
    kept_.clear();
    return ticket;
}

std::vector<std::string> Graph::ExecutedPasses() const {
    std::vector<std::string> names;
    for (const std::uint32_t p : live_) {
        names.push_back(passes_[p].name);
    }
    return names;
}

std::string Graph::Dot() const {
    std::vector<bool> live(passes_.size(), !compiled_);
    for (const std::uint32_t p : live_) {
        live[p] = true;
    }
    std::ostringstream dot;
    dot << "digraph FrameGraph {\n  rankdir=LR;\n";
    for (std::size_t r = 0; r < resources_.size(); ++r) {
        const Resource& resource = resources_[r];
        const char* shape = resource.image         ? "ellipse"
                            : resource.Structure() ? "hexagon"
                                                   : "cylinder";
        dot << "  r" << r << " [shape=" << shape << ", label=\"" << resource.name << "\""
            << (resource.imported ? ", style=bold" : "") << "];\n";
    }
    for (std::size_t p = 0; p < passes_.size(); ++p) {
        const Pass& pass = passes_[p];
        dot << "  p" << p << " [shape=box, label=\"" << pass.name << "\""
            << (live[p] ? "" : ", style=dashed") << "];\n";
        for (const DependencyAccess& access : pass.accesses) {
            if (Writes(access.use)) {
                dot << "  p" << p << " -> r" << access.resource;
            } else {
                dot << "  r" << access.resource << " -> p" << p;
            }
            dot << " [label=\"" << ToString(access.use) << "\""
                << (access.reach == Reach::Descriptor ? "" : ", style=dotted") << "];\n";
        }
    }
    dot << "}\n";
    return dot.str();
}

void AddUploadPass(Graph& graph, std::string_view name, Uploader& uploader,
                   std::span<Buffer> targets) {
    if (!uploader.Staged()) {
        return;
    }
    graph.Add(
        name, Kind::Copy,
        [&](PassBuilder& pass) {
            for (Buffer& target : targets) {
                if (target.Valid()) {
                    target = pass.Write(target, Use::CopyDst);
                }
            }
        },
        [&uploader](PassContext& pass) {
            for (std::shared_ptr<const ernie::Buffer>& staging : uploader.Record(pass.commands)) {
                pass.Keep(std::move(staging));
            }
        });
}

}  // namespace ernie::graph
