// SPDX-License-Identifier: BSD-3-Clause
/// @file scene_columns.h
/// @brief Scene columns whose source of truth is the device: per-step values, such as the poses
/// of thousands of worlds, written in place by kernels, CUDA or Warp, and bound by passes as they
/// are. The host scene keeps the structure and the authored values; per-step data never passes
/// through it.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/staging.h"
#include "ernie/scene/scene.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
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
    /// Uploads rows [begin, end) of the binding's shadow.
    void Upload(const Binding& binding, std::uint32_t begin, std::uint32_t end);
    /// Uploads sorted rows, a run at a time.
    void UploadRows(Binding& binding, std::span<const std::uint32_t> rows);
    const Binding* FindBinding(BucketId bucket, AttributeId attribute) const;

    DeviceContext& context_;
    Arenas& arenas_;
    const Scene& scene_;
    Uploader uploader_;
    std::vector<Binding> bindings_;
};

}  // namespace ernie
