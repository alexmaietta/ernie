// SPDX-License-Identifier: BSD-3-Clause
/// @file graph.h
/// @brief The frame graph: passes declare the images and buffers they read and write, and the
/// graph culls what nothing needs, backs transients with pooled memory and records the passes in
/// order with the barriers between them.
///
///     graph::Graph graph(context, pool);
///     const graph::Buffer poses = graph.Import(poseColumn, "poses");
///     graph::Image radiance = graph.Create(description, "radiance");
///     graph.Add("Shade", graph::Kind::Compute,
///         [&](graph::PassBuilder& pass) {
///             pass.ReadAddress(poses);
///             radiance = pass.Write(radiance);
///         },
///         [=](graph::PassContext& pass) {
///             shade.Dispatch(pass.commands, groups, Parameters{pass.Get(poses).Address(), ...});
///         });
///     const Ticket done = graph.Execute(context.Main());
///
/// Setup runs inside `Add`; run callbacks run in `Execute`. Passes run in the order they were
/// added. Imports enter and leave in the general layout. Bindless and device-address reads are
/// invisible to the graph, so passes declare them (`ReadBindless`, `ReadAddress`).
/// ERNIE_GRAPH_FULL_BARRIERS=1 adds a full barrier before every pass, to rule out a missing one.
///
/// References:
/// - Y. O'Donnell, "FrameGraph: Extensible Rendering Architecture in Frostbite", GDC 2017.
/// - Epic Games, Unreal Engine's Render Dependency Graph (RenderCore/RenderGraphBuilder.cpp).
/// - Google, Filament's FrameGraph. https://github.com/google/filament
/// - H.-K. Arntzen, "Render graphs and Vulkan — a deep dive", 2017.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/image.h"
#include "ernie/graph/blackboard.h"
#include "ernie/graph/dependencies.h"
#include "ernie/graph/pool.h"

#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ernie::graph {

/// @brief A graph that cannot run: a pass reads what nothing wrote, or uses a resource in a way
/// it cannot.
class GraphError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// @brief A graph's image; each write makes a new version.
struct Image {
    std::uint32_t id = ~0u;
    /// Writes declared before this handle was made.
    std::uint32_t version = 0;
    bool Valid() const {
        return id != ~0u;
    }
};

/// @brief A graph's buffer; each write makes a new version.
struct Buffer {
    std::uint32_t id = ~0u;
    std::uint32_t version = 0;
    bool Valid() const {
        return id != ~0u;
    }
};

/// @brief A buffer resource as a pass records with it: a range of a device buffer.
struct BufferSlice {
    const ernie::Buffer* buffer = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    VkDeviceAddress Address() const {
        return buffer->Address() + offset;
    }
};

class Graph;

/// @brief Declares one pass's accesses during setup.
class PassBuilder {
  public:
    /// @return The same handle.
    /// @throws std::invalid_argument for a use that writes.
    Image Read(Image image, Use use = Use::Sampled, Layers layers = {});
    /// @brief Declares a write, or a read and write.
    /// @return The new version.
    /// @throws std::invalid_argument for a use that only reads.
    Image Write(Image image, Use use = Use::StorageWrite, Layers layers = {});
    Buffer Read(Buffer buffer, Use use = Use::StorageRead);
    Buffer Write(Buffer buffer, Use use = Use::StorageWrite);
    /// @brief Declares a read through the bindless heap.
    Image ReadBindless(Image image, Use use = Use::Sampled, Layers layers = {});
    /// @brief Declares a read through a device address.
    Buffer ReadAddress(Buffer buffer, Use use = Use::StorageRead);
    /// @brief Declares a write through a device address.
    Buffer WriteAddress(Buffer buffer, Use use = Use::StorageReadWrite);
    /// @brief Keeps the pass though nothing reads what it writes, as for one the host reads.
    void NeverCull();

  private:
    friend class Graph;
    PassBuilder(Graph& graph, std::uint32_t pass) : graph_(graph), pass_(pass) {}

    std::uint32_t Declare(std::uint32_t id, std::uint32_t version, bool image, Use use,
                          Layers layers, Reach reach, bool write);

    Graph& graph_;
    std::uint32_t pass_;
};

/// @brief What a pass's run callback records with.
class PassContext {
  public:
    VkCommandBuffer commands = VK_NULL_HANDLE;

    /// @throws std::logic_error unless the pass declared it.
    DeviceImage& Get(Image image) const;
    BufferSlice Get(Buffer buffer) const;
    /// @return The image's layout during the pass, for attachments.
    VkImageLayout Layout(Image image) const;
    DeviceContext& Context() const;

  private:
    friend class Graph;
    PassContext(const Graph& graph, std::uint32_t pass) : graph_(graph), pass_(pass) {}

    void Check(std::uint32_t id) const;

    const Graph& graph_;
    std::uint32_t pass_;
};

/// @brief Passes in order and the resources they use, built for one frame: set up, compiled
/// and executed once.
class Graph {
  public:
    /// @param pool Where transients come from; outlives the graph.
    Graph(DeviceContext& context, TransientPool& pool);
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    /// @brief Uses an image owned outside the graph; importing it again returns the same
    /// resource. The image outlives `Execute`'s work.
    /// @return The handle of its latest version.
    Image Import(DeviceImage& image, std::string_view name);
    /// @brief Uses a column's rows, as kernels, CUDA or Warp write them.
    Buffer Import(const DeviceColumn& column, std::string_view name);
    /// @brief Uses a range of a buffer owned outside the graph.
    Buffer Import(const ernie::Buffer& buffer, std::uint64_t offset, std::uint64_t size,
                  std::string_view name);
    /// @brief Declares an image living for one execution; the usage its passes declare is
    /// added to `description`'s.
    Image Create(const ImageDescription& description, std::string_view name);
    /// @brief Declares a device-local buffer living for one execution.
    Buffer Create(std::uint64_t bytes, std::string_view name);

    /// @brief Adds a pass after the others, running `setup` now.
    /// @param setup Called with a `PassBuilder&` to declare accesses.
    /// @param run Called with a `PassContext&` in `Execute`.
    /// @throws std::logic_error after `Compile`.
    template <typename Setup, typename Run>
    void Add(std::string_view name, Kind kind, Setup&& setup, Run&& run) {
        PassBuilder builder(*this, NewPass(name, kind));
        std::forward<Setup>(setup)(builder);
        passes_.back().run = std::forward<Run>(run);
    }

    /// @brief A named group of the passes added while it lives, labelled as one region.
    class [[nodiscard]] Zone {
      public:
        ~Zone();
        Zone(const Zone&) = delete;
        Zone& operator=(const Zone&) = delete;

      private:
        friend class Graph;
        explicit Zone(Graph& graph) : graph_(graph) {}
        Graph& graph_;
    };
    /// @brief Opens a scope for the passes added until the zone is destroyed; scopes nest.
    Zone Scope(std::string_view name);

    /// @return Handles that passes share by type.
    Blackboard& Board() {
        return board_;
    }

    /// @brief Drops passes nothing needs, backs transients and computes the barriers.
    /// `Execute` calls it.
    /// @throws GraphError if a pass reads what nothing wrote, or uses a resource as it cannot.
    void Compile();
    /// @brief Records every kept pass in order with its barriers, submits them on `queue` after
    /// `waits`, and gives transients back to the pool.
    /// @return The ticket of the graph's work, which CUDA can wait on too.
    /// @throws GraphError as `Compile`; std::logic_error if executed already.
    Ticket Execute(Queue& queue, std::span<const Ticket> waits = {});

    /// @brief Adds a full barrier before every pass; on by default with
    /// ERNIE_GRAPH_FULL_BARRIERS=1.
    void SetFullBarriers(bool on) {
        fullBarriers_ = on;
    }
    /// @brief Submits each pass on its own and waits for it, so a device fault names its pass.
    void SetFlushEachPass(bool on) {
        flushEachPass_ = on;
    }

    /// @return The dependencies `Compile` computed, for the passes it kept.
    const PassDependencies& Dependencies() const {
        return dependencies_;
    }
    /// @return The names of the passes `Compile` kept, in order.
    std::vector<std::string> ExecutedPasses() const;
    /// @return The graph in Graphviz's dot language.
    std::string Dot() const;

  private:
    friend class PassBuilder;
    friend class PassContext;

    struct Resource {
        std::string name;
        bool image = false;
        bool imported = false;
        DeviceImage* deviceImage = nullptr;
        BufferSlice slice;
        /// A transient's description, or size, and its leased range.
        ImageDescription description;
        std::uint64_t bytes = 0;
        BufferRange range;
        std::uint32_t version = 0;
        Origin origin = Origin::Fresh;
    };
    struct Pass {
        std::string name;
        Kind kind = Kind::Compute;
        std::vector<DependencyAccess> accesses;
        std::function<void(PassContext&)> run;
        /// The scopes it was added in, outermost first.
        std::vector<std::string> scopes;
        bool neverCull = false;
    };

    std::uint32_t NewPass(std::string_view name, Kind kind);
    std::uint32_t AddResource(Resource resource);
    std::vector<std::uint32_t> LivePasses() const;
    void Lease(const std::vector<std::uint32_t>& live);
    void ReleaseLeases();
    void Record(VkCommandBuffer commands, const Dependency& dependency) const;
    /// Records a pass with its labels and barriers.
    void RecordPass(VkCommandBuffer commands, std::uint32_t order) const;

    DeviceContext& context_;
    TransientPool& pool_;
    std::vector<Resource> resources_;
    std::vector<Pass> passes_;
    std::vector<std::string> scopes_;
    Blackboard board_;
    /// The passes kept, in order, and their dependencies.
    std::vector<std::uint32_t> live_;
    PassDependencies dependencies_;
    bool compiled_ = false;
    bool executed_ = false;
    bool leased_ = false;
    bool fullBarriers_ = false;
    bool flushEachPass_ = false;
};

}  // namespace ernie::graph
