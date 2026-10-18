// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_graph.h
/// @brief World transforms composed on the device: every prim a node, ordered by depth, so one
/// small kernel per level of the hierarchy composes its nodes from their parents'. Poses written
/// in place by CUDA, Warp or kernels become world transforms without the host.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/staging.h"
#include "ernie/scene/scene.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

/// @brief The scene's transform hierarchy on the device. Each prim is a node with its parent's
/// node (none at a root or `!resetXformStack!`) and a local transform; worlds are composed level
/// by level, as 3x4 row-major matrices (VkTransformMatrixKHR's layout).
///
/// Local transforms come from the host, uploaded only when their ops are authored, except in
/// buckets whose poses are bound: there `before * translate * orient * after` is composed on the
/// device from the bucket's `xformOp:translate` and `xformOp:orient` device columns, as written
/// in place each step, with `before` and `after` the ops around them that the host authors.
///
/// Structural changes rebuild the graph on the host, at a cost in prims; per-step work is on the
/// device and costs no host transfers.
class TransformGraph {
  public:
    static constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
    /// A transform's top three rows, as the device keeps them.
    using Rows12 = std::array<float, 12>;

    TransformGraph(DeviceContext& context, Arenas& arenas, const Scene& scene,
                   const ShaderCompiler& compiler);
    ~TransformGraph();
    TransformGraph(const TransformGraph&) = delete;
    TransformGraph& operator=(const TransformGraph&) = delete;

    /// @brief Composes the bucket's local transforms from its pose columns: `translate` with three
    /// floats a row and `orient` with four (x, y, z, w), as `SceneColumns` keeps them. Prims whose
    /// `xformOpOrder` has no `xformOp:translate` right before `xformOp:orient` keep their host
    /// transforms.
    /// @throws std::invalid_argument unless the columns have the bucket's rows and those shapes.
    void BindPoses(BucketId bucket, const DeviceColumn& translate, const DeviceColumn& orient);
    /// @brief Follows the scene's changes since the last update and composes every world
    /// transform, after `waits`: the tickets of the pose columns' writers and their sync.
    /// @return The ticket passes wait on before reading `Worlds`.
    Ticket Update(std::span<const Ticket> waits = {});

    /// @return Each node's world transform, three float4 rows a node.
    const DeviceColumn& Worlds() const {
        return *worlds_;
    }
    /// @return The node of each prim id slot, `kNone` for free slots: the table writers and passes
    /// use to find a prim's transform from its stable id.
    const DeviceColumn& SlotNodes() const {
        return *slotNodes_;
    }
    /// @return The prim's node, or none when it is not in the graph yet.
    std::optional<std::uint32_t> Node(PrimId prim) const;
    /// @return The number of levels, each one dispatch.
    std::size_t Levels() const {
        return levels_.size();
    }

  private:
    struct Poses {
        BucketId bucket = 0;
        const DeviceColumn* translate = nullptr;
        const DeviceColumn* orient = nullptr;
        /// The node of each bucket row, `kNone` for rows composed on the host.
        std::unique_ptr<DeviceColumn> nodes;
    };
    struct Level {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    void Rebuild();
    /// Uploads a node's host-authored transforms: its local, or the ops around its pose.
    void UploadLocal(PrimId prim, std::uint32_t node, const Poses* poses);
    const Poses* FindPoses(BucketId bucket) const;
    /// Uploads a node's rows of `column` unless they are what was uploaded last.
    void Write(DeviceColumn& column, std::vector<Rows12>& uploaded, std::uint32_t node,
               const Rows12& rows);

    DeviceContext& context_;
    Arenas& arenas_;
    const Scene& scene_;
    Uploader uploader_;
    ComputeKernel pose_;
    ComputeKernel compose_;
    std::uint64_t cursor_ = 0;
    bool built_ = false;
    std::vector<Poses> poses_;
    std::vector<Level> levels_;
    /// Host copies of the slot table and each node's prim.
    std::vector<std::uint32_t> slotNode_;
    std::vector<PrimId> nodePrim_;
    /// What the host last uploaded of each node's transforms.
    std::vector<Rows12> uploadedLocals_;
    std::vector<Rows12> uploadedBefore_;
    std::vector<Rows12> uploadedAfter_;
    std::unique_ptr<DeviceColumn> parents_;
    std::unique_ptr<DeviceColumn> locals_;
    std::unique_ptr<DeviceColumn> before_;
    std::unique_ptr<DeviceColumn> after_;
    std::unique_ptr<DeviceColumn> worlds_;
    std::unique_ptr<DeviceColumn> slotNodes_;
};

}  // namespace ernie
