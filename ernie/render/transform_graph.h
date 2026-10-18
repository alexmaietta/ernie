// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph.h
/// @brief World transforms composed on the device: each prim a node at its id's slot, one small
/// kernel per level of the hierarchy composing its nodes from their parents', and transform ops
/// written in place by CUDA, Warp or kernels evaluated on the device without the host.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/staging.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/xform.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief The scene's transform hierarchy on the device. A prim's node is its id's slot, so the
/// stable id finds its transform with no table: `Worlds` holds a 3x4 row-major matrix a slot
/// (VkTransformMatrixKHR's layout). Each node keeps its parent's slot, none at a root or under
/// `!resetXformStack!`.
///
/// Local transforms are the host's, uploaded when their ops are authored, unless a prim's op order
/// uses an attribute bound to a device column (`Bind`). Then the prim's ops become a short program
/// the device evaluates each update: the bound ops read their rows in place, as CUDA, Warp or
/// kernels wrote them, and runs of the host's other ops are folded into one matrix each.
///
/// Updates follow the scene's changes and cost what changed: added and removed prims are found by
/// replaying the buckets' row logs, and a prim defined between a prim and its descendants takes
/// them over by walking only that subtree.
class TransformGraph {
  public:
    static constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
    /// The most ops a program holds: device ops and the folded host ops between them.
    static constexpr std::uint32_t kMaxOps = 8;
    /// A transform's top three rows, as the device keeps them.
    using Rows12 = std::array<float, 12>;
    /// One op of a program, as the device reads it.
    struct DeviceOp {
        std::uint32_t kind = 0;
        /// Bit 0 inverts; bits 8 to 15 count the floats the op reads.
        std::uint32_t flags = 0;
        VkDeviceAddress source = 0;
        Rows12 constant{};
        bool operator==(const DeviceOp&) const = default;
    };
    using Program = std::array<DeviceOp, kMaxOps>;

    TransformGraph(DeviceContext& context, Arenas& arenas, const Scene& scene,
                   const ShaderCompiler& compiler);
    ~TransformGraph();
    TransformGraph(const TransformGraph&) = delete;
    TransformGraph& operator=(const TransformGraph&) = delete;

    /// @brief Evaluates the bucket's `attribute` op on the device from `column`, a row a prim as
    /// `SceneColumns` keeps it: translate and Euler rotations read three floats, scale one or
    /// three, rotateX, rotateY and rotateZ one angle in degrees, orient four (x, y, z, w) and
    /// transform sixteen (column-major).
    /// @throws std::invalid_argument when `attribute` is no transform op, the column does not hold
    /// the bucket's rows of floats in the op's shape, or the attribute is bound already.
    void Bind(BucketId bucket, std::string_view attribute, const DeviceColumn& column);
    /// @brief Follows the scene's changes since the last update and composes every world
    /// transform, after `waits`: the tickets of the bound columns' writers and their sync.
    /// @return The ticket passes wait on before reading `Worlds`.
    /// @throws std::invalid_argument when a prim's program needs more than `kMaxOps` ops.
    Ticket Update(std::span<const Ticket> waits = {});

    /// @return Each slot's world transform, three float4 rows a slot.
    const DeviceColumn& Worlds() const {
        return *worlds_;
    }
    /// @return The prim's node, its slot, or none when it is not in the graph yet.
    std::optional<std::uint32_t> Node(PrimId prim) const;
    /// @return The number of levels, each one dispatch.
    std::size_t Levels() const {
        return levels_.size();
    }

  private:
    struct Binding {
        BucketId bucket = 0;
        AttributeId attribute = 0;
        const DeviceColumn* column = nullptr;
        /// The column's address its programs point into; a column that moves rewrites them.
        VkDeviceAddress address = 0;
    };
    struct NodeState {
        PrimId prim;
        bool live = false;
        std::uint32_t parent = kNone;
        std::uint32_t level = 0;
        /// Where the node is in its level's list, its parent's children and the programmed list.
        std::uint32_t position = kNone;
        std::uint32_t siblingPosition = kNone;
        std::uint32_t programPosition = kNone;
        std::vector<std::uint32_t> children;
        /// Its order's ops, parsed when the order changes.
        std::vector<XformOp> ops;
        bool resets = false;
    };
    /// Slots in a device list, kept dense by swap-removal, its changed entries uploaded in runs.
    struct SlotList {
        std::vector<std::uint32_t> slots;
        std::unique_ptr<DeviceColumn> device;
        std::vector<std::uint32_t> dirty;
    };
    /// Slots of a slot array changed since its last upload, each once.
    struct DirtySlots {
        std::vector<std::uint32_t> slots;
        std::vector<bool> marked;
        void Mark(std::uint32_t slot) {
            if (slot >= marked.size()) {
                marked.resize(slot + 1, false);
            }
            if (!marked[slot]) {
                marked[slot] = true;
                slots.push_back(slot);
            }
        }
    };

    void Replay(const ChangeSet& changes);
    /// Refreshes the nodes whose transform values the host authored, reparenting those whose
    /// order changed.
    void RefreshAuthored(const ChangeSet& changes);
    void AddNode(PrimId prim);
    void RemoveNode(std::uint32_t slot);
    /// Hangs a node under `parent` at the level below it, its subtree following.
    void Attach(std::uint32_t slot, std::uint32_t parent);
    void Detach(std::uint32_t slot);
    void SetLevel(std::uint32_t slot, std::uint32_t level);
    /// @return The slot of the node's nearest ancestor prim, none at a root or a stack reset.
    std::uint32_t ParentSlot(std::uint32_t slot) const;
    /// Hands descendants whose nearest ancestor prim is now `slot`'s over to it.
    void AdoptDescendants(std::uint32_t slot);
    /// Parses a node's op order.
    void ParseOps(std::uint32_t slot);
    /// Recomposes a node's local transform or program from its ops.
    void Refresh(std::uint32_t slot);
    /// Reparses and refreshes every node of a bucket.
    void RefreshBucket(BucketId bucket);
    void EnsureSlots(std::uint32_t slots);
    std::uint32_t Insert(SlotList& list, std::uint32_t slot);
    /// Takes a slot out of a list it is in at its `position` member, swap-removing.
    /// @return Where it was.
    std::uint32_t Unlist(SlotList& list, std::uint32_t NodeState::* position, std::uint32_t slot);
    /// Takes a node's program out, the last program moving into its place.
    void Unprogram(std::uint32_t slot);
    const Binding* FindBinding(BucketId bucket, std::optional<Token> attribute) const;
    /// A prim's transform as the host composes it, or as the program the device evaluates.
    struct Built {
        bool onDevice = false;
        Rows12 local{};
        Program program{};
        std::uint32_t length = 0;
    };
    Built BuildProgram(std::uint32_t slot) const;
    void UploadDirty();
    void RecordDispatches(VkCommandBuffer commands) const;

    DeviceContext& context_;
    Arenas& arenas_;
    const Scene& scene_;
    Uploader uploader_;
    ComputeKernel local_;
    ComputeKernel compose_;
    std::uint64_t cursor_ = 0;
    std::vector<Binding> bindings_;
    /// Each bucket's prims by row as the graph last saw them, to replay row logs on.
    std::vector<std::vector<PrimId>> rows_;
    std::vector<NodeState> nodes_;
    std::vector<SlotList> levels_;
    SlotList programmed_;
    // Host copies of the slot arrays and the programmed list's programs, and the entries changed
    // since the last upload.
    std::vector<std::uint32_t> parents_;
    std::vector<Rows12> locals_;
    std::vector<Program> programs_;
    std::vector<std::uint32_t> lengths_;
    DirtySlots dirtyParents_;
    DirtySlots dirtyLocals_;
    DirtySlots dirtyPrograms_;
    std::unique_ptr<DeviceColumn> parentColumn_;
    std::unique_ptr<DeviceColumn> localColumn_;
    std::unique_ptr<DeviceColumn> programColumn_;
    std::unique_ptr<DeviceColumn> lengthColumn_;
    std::unique_ptr<DeviceColumn> worlds_;
};

}  // namespace ernie
