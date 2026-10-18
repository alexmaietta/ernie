// SPDX-License-Identifier: BSD-3-Clause
/// @file scene_columns.h
/// @brief Scene columns whose source of truth is the device: per-step values, such as the poses
/// of thousands of worlds, written in place by kernels, CUDA or Warp, and bound by passes as they
/// are. The host scene keeps the structure and the authored values; per-step data never passes
/// through it.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/staging.h"
#include "ernie/graph/graph.h"
#include "ernie/scene/scene.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ernie {

/// @brief How a host column's rows look on the device: floats for the scene's doubles.
struct DeviceElement {
    DType type = DType::Float32;
    std::vector<std::int64_t> shape;
};

/// @return The device element of a column type: Double becomes one float, Vec2 to Vec4 that many,
/// Quat four (x, y, z, w), Matrix sixteen in glm's column-major order, Int an int32 and Bool a
/// byte. Variable-length types have none.
std::optional<DeviceElement> ToDeviceElement(ColumnType type);

/// @brief Device columns that follow their scene buckets' rows. Each starts from the scene's
/// values, then the device owns them: when prims are removed, the bucket's swap-removes are
/// replayed as one batch of row copies on the device, rows added on the host start from the
/// host's values, and values authored on the host overwrite those rows. Rows the device wrote and
/// the host only moved keep the device's values.
class SceneColumns {
  public:
    /// @param arenas Where the columns live: `MemoryUse::Shared` arenas to share them with CUDA.
    SceneColumns(DeviceContext& context, Arenas& arenas, const Scene& scene);
    ~SceneColumns();
    SceneColumns(const SceneColumns&) = delete;
    SceneColumns& operator=(const SceneColumns&) = delete;

    /// @brief Puts a bucket's column of `attribute` on the device, from the scene's values.
    /// @throws std::invalid_argument when the bucket has no such column, or its type has no
    /// device element.
    const DeviceColumn& Bind(BucketId bucket, std::string_view attribute);
    /// @brief Puts a bucket's column of `attribute` on the device in `column`, such as a CUDA,
    /// PyTorch or Warp tensor wrapped in place (`CudaInterop::Wrap`): its values are what the
    /// device holds, not the scene's, and values the host authors later still overwrite them.
    /// @throws std::invalid_argument unless `column` has the bucket's rows and the attribute's
    /// device element, or when the attribute is bound already.
    const DeviceColumn& Bind(BucketId bucket, std::string_view attribute,
                             std::unique_ptr<DeviceColumn> column);
    /// @return The bucket's device column of `attribute`, or null when it is not bound.
    const DeviceColumn* Find(BucketId bucket, std::string_view attribute) const;
    /// @brief Follows the scene's changes since the last sync, after `waits`.
    /// @return The ticket passes wait on before reading the columns.
    Ticket Sync(std::span<const Ticket> waits = {});
    /// @brief Follows the scene's changes as a pass of a graph run on the main queue: rows move
    /// and host values land before later passes that import the columns read them.
    void AddSync(graph::Graph& graph);

  private:
    struct Binding {
        BucketId bucket = 0;
        AttributeId attribute = 0;
        std::uint64_t cursor = 0;
        std::unique_ptr<DeviceColumn> column;
        /// The device bytes of what the host last gave each row, to tell host edits apart.
        std::vector<std::byte> shadow;
    };

    /// A bucket's host column of an attribute, and its device element.
    struct Host {
        AttributeId attribute = 0;
        const Column* column = nullptr;
        DeviceElement element;
        std::uint32_t rows = 0;
    };

    /// @throws std::invalid_argument when the bucket has no such column, or it has no device
    /// element.
    Host HostColumn(BucketId bucket, std::string_view attribute) const;
    /// Where each row changed since a cursor has its values from: a row before, or the host.
    using Origins = std::unordered_map<std::uint32_t, std::uint32_t>;

    static Origins RowOrigins(const ChangeSet& changes, BucketId bucket);
    /// @return The row `row`'s values come from; rows that did not change keep their own.
    static std::uint32_t Origin(const Origins& origins, std::uint32_t row);
    /// @return The device copies that move the kept rows to their new places, moving the shadow
    /// alike; rows new from the host are added to `hostRows`.
    std::vector<VkBufferCopy> MoveRows(Binding& binding, const Origins& origins, std::uint32_t rows,
                                       std::vector<std::uint32_t>& hostRows);
    /// Adds the rows whose host values were authored anew to `hostRows`, taking their values
    /// into the shadow.
    void FindAuthoredRows(const ChangeSet& changes, Binding& binding, const Origins& origins,
                          std::uint32_t rows, std::vector<std::uint32_t>& hostRows);
    /// Uploads the binding's shadow at sorted `rows`.
    void Upload(const Binding& binding, std::span<const std::uint32_t> rows);
    /// Follows the changes on the host: resizes the columns, plans the row moves and stages the
    /// host's rows.
    /// @return What the moves and uploads wait on.
    std::vector<Ticket> Prepare(std::span<const Ticket> waits);
    /// Records the planned moves, then the staged uploads.
    void Record(VkCommandBuffer commands,
                const std::function<void(std::shared_ptr<const void>)>& keep);
    const Binding* FindBinding(BucketId bucket, AttributeId attribute) const;

    DeviceContext& context_;
    Arenas& arenas_;
    const Scene& scene_;
    Uploader uploader_;
    std::vector<Binding> bindings_;
    /// Authored ranges' device bytes, compared with the shadow.
    std::vector<std::byte> scratch_;
    /// Each column's row copies, from the rows before the sync to the rows after it.
    std::vector<std::pair<const DeviceColumn*, std::vector<VkBufferCopy>>> moves_;
};

}  // namespace ernie
