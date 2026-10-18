// SPDX-License-Identifier: BSD-3-Clause
/// @file scene_columns.cpp
/// @brief Replaying bucket row changes on device columns, and converting host rows to floats.
#include "ernie/render/scene_columns.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace ernie {
namespace {

/// Where a row's values come from after a sync: a row of the device column before it, or the
/// host.
constexpr std::uint32_t kFromHost = std::numeric_limits<std::uint32_t>::max();

/// Appends one host value as the device element's bytes.
template <typename T> void AppendDevice(const T& value, std::vector<std::byte>& bytes) {
    const auto append = [&](const auto* data, std::size_t count) {
        const auto* first = reinterpret_cast<const std::byte*>(data);
        bytes.insert(bytes.end(), first, first + count * sizeof(*data));
    };
    if constexpr (std::is_same_v<T, double>) {
        const float converted = static_cast<float>(value);
        append(&converted, 1);
    } else if constexpr (std::is_same_v<T, glm::dquat>) {
        const glm::vec4 converted(value.x, value.y, value.z, value.w);
        append(&converted.x, 4);
    } else if constexpr (std::is_same_v<T, glm::dmat4>) {
        const glm::mat4 converted(value);
        append(&converted[0][0], 16);
    } else if constexpr (std::is_same_v<T, glm::dvec2> || std::is_same_v<T, glm::dvec3> ||
                         std::is_same_v<T, glm::dvec4>) {
        const glm::vec<T::length(), float> converted(value);
        append(&converted.x, T::length());
    } else {
        append(&value, 1);
    }
}

/// The rows' device bytes, converted from the host column.
std::vector<std::byte> DeviceBytes(const ColumnValues& values, std::uint32_t begin,
                                   std::uint32_t end) {
    std::vector<std::byte> bytes;
    WithColumnType(values.Type(), [&]<typename T>(std::type_identity<T>) {
        if constexpr (std::is_trivially_copyable_v<T> && !std::is_same_v<T, Token> &&
                      !std::is_same_v<T, Target>) {
            const std::vector<T>& rows = static_cast<const TypedColumn<T>&>(values).Values();
            for (std::uint32_t row = begin; row < end; ++row) {
                AppendDevice(rows[row], bytes);
            }
        }
    });
    return bytes;
}

}  // namespace

std::optional<DeviceElement> ToDeviceElement(ColumnType type) {
    switch (type) {
    case ColumnType::Bool:
        return DeviceElement{DType::UInt8, {}};
    case ColumnType::Int:
        return DeviceElement{DType::Int32, {}};
    case ColumnType::Double:
        return DeviceElement{DType::Float32, {}};
    case ColumnType::Vec2:
        return DeviceElement{DType::Float32, {2}};
    case ColumnType::Vec3:
        return DeviceElement{DType::Float32, {3}};
    case ColumnType::Vec4:
    case ColumnType::Quat:
        return DeviceElement{DType::Float32, {4}};
    case ColumnType::Matrix:
        return DeviceElement{DType::Float32, {4, 4}};
    default:
        return std::nullopt;
    }
}

SceneColumns::SceneColumns(DeviceContext& context, Arenas& arenas, const Scene& scene)
    : context_(context), arenas_(arenas), scene_(scene), uploader_(context, context.Main()) {}

SceneColumns::~SceneColumns() = default;

const DeviceColumn& SceneColumns::Bind(BucketId bucket, std::string_view attribute) {
    const PrimStore& store = scene_.Store();
    const std::optional<Token> token = store.Tokens().Find(attribute);
    const Column* column = token ? store.GetBucket(bucket).Find(*token) : nullptr;
    if (column == nullptr) {
        throw std::invalid_argument("The bucket has no column " + std::string(attribute));
    }
    if (const Binding* bound = FindBinding(bucket, *token)) {
        return *bound->column;
    }
    const std::optional<DeviceElement> element = ToDeviceElement(column->values->Type());
    if (!element) {
        throw std::invalid_argument(std::string(attribute) +
                                    " holds values of no fixed size, which stay on the host");
    }
    const std::uint32_t rows = store.GetBucket(bucket).Rows();
    Binding& binding = bindings_.emplace_back(Binding{
        bucket, *token, store.Version(),
        std::make_unique<DeviceColumn>(arenas_, attribute, element->type, element->shape, rows),
        DeviceBytes(*column->values, 0, rows)});
    Upload(binding, 0, rows);
    return *binding.column;
}

const DeviceColumn* SceneColumns::Find(BucketId bucket, std::string_view attribute) const {
    const std::optional<Token> token = scene_.Store().Tokens().Find(attribute);
    const Binding* binding = token ? FindBinding(bucket, *token) : nullptr;
    return binding ? binding->column.get() : nullptr;
}

const SceneColumns::Binding* SceneColumns::FindBinding(BucketId bucket,
                                                       AttributeId attribute) const {
    for (const Binding& binding : bindings_) {
        if (binding.bucket == bucket && binding.attribute == attribute) {
            return &binding;
        }
    }
    return nullptr;
}

void SceneColumns::Upload(const Binding& binding, std::uint32_t begin, std::uint32_t end) {
    if (begin == end) {
        return;
    }
    const DeviceColumn& device = *binding.column;
    const std::uint64_t rowBytes = device.RowBytes();
    uploader_.Write(device.Storage(), device.Range().offset + begin * rowBytes,
                    std::span<const std::byte>(binding.shadow)
                        .subspan(begin * rowBytes, (end - begin) * rowBytes));
}

void SceneColumns::UploadRows(Binding& binding, std::span<const std::uint32_t> rows) {
    std::size_t first = 0;
    for (std::size_t i = 1; i <= rows.size(); ++i) {
        if (i == rows.size() || rows[i] != rows[i - 1] + 1) {
            Upload(binding, rows[first], rows[i - 1] + 1);
            first = i;
        }
    }
}

Ticket SceneColumns::Sync(std::span<const Ticket> waits) {
    const PrimStore& store = scene_.Store();
    std::vector<Ticket> after(waits.begin(), waits.end());
    // Values staged by `Bind` land before any column moves to a larger range.
    after.push_back(uploader_.Flush(waits));
    // Each binding's row copies, from the rows before the sync to the rows after it.
    std::vector<std::pair<const DeviceColumn*, std::vector<VkBufferCopy>>> moves;
    for (Binding& binding : bindings_) {
        const ChangeSet changes = store.ChangesSince(binding.cursor);
        binding.cursor = changes.version;
        DeviceColumn& column = *binding.column;
        const std::uint64_t rowBytes = column.RowBytes();
        // Where each changed row's values come from; rows not in it stay where they are.
        std::unordered_map<std::uint32_t, std::uint32_t> origin;
        const auto from = [&](std::uint32_t row) {
            const auto found = origin.find(row);
            return found == origin.end() ? row : found->second;
        };
        for (const ChangeSet::Rows& entry : changes.rows) {
            if (entry.bucket != binding.bucket) {
                continue;
            }
            const StructuralChange& change = entry.change;
            switch (change.kind) {
            case StructuralChange::Kind::Added:
                for (std::uint32_t row = change.row; row < change.row + change.count; ++row) {
                    origin[row] = kFromHost;
                }
                break;
            case StructuralChange::Kind::Removed:
                // Unless it was the last, a Moved change brings the last row here next.
                origin.erase(change.row);
                break;
            case StructuralChange::Kind::Moved:
                origin[change.row] = from(change.from);
                origin.erase(change.from);
                break;
            case StructuralChange::Kind::Reset:
                origin.clear();
                for (std::uint32_t row = 0; row < change.count; ++row) {
                    origin[row] = kFromHost;
                }
                break;
            }
        }
        const std::uint32_t rows = store.GetBucket(binding.bucket).Rows();
        std::vector<VkBufferCopy> copies;
        std::vector<std::uint32_t> hostRows;
        binding.shadow.resize(std::max<std::uint64_t>(binding.shadow.size(), rows * rowBytes));
        for (const auto& [row, source] : origin) {
            if (row >= rows) {
                continue;
            }
            if (source == kFromHost) {
                hostRows.push_back(row);
            } else if (source != row) {
                // Every source is a row removed or past the end, never another copy's target.
                copies.push_back({source * rowBytes, row * rowBytes, rowBytes});
                std::memcpy(binding.shadow.data() + row * rowBytes,
                            binding.shadow.data() + source * rowBytes, rowBytes);
            }
        }
        binding.shadow.resize(rows * rowBytes);
        if (!copies.empty()) {
            moves.emplace_back(&column, std::move(copies));
        }
        if (rows > column.Rows()) {
            after.push_back(column.Resize(rows, context_.Main(), after));
        } else {
            column.Resize(rows, context_.Main());
        }

        // Values authored on the host: rows that differ from what the host last gave. Changes
        // come by chunk, so rows the device wrote are compared, not taken on trust.
        const ColumnValues& host = *store.GetBucket(binding.bucket).Find(binding.attribute)->values;
        for (const ChangeSet::Values& values : changes.values) {
            if (values.bucket != binding.bucket || values.attribute != binding.attribute) {
                continue;
            }
            for (const RowRange& range : values.rows) {
                for (std::uint32_t row = range.begin; row < std::min(range.end, rows); ++row) {
                    if (from(row) != kFromHost &&
                        !std::ranges::equal(DeviceBytes(host, row, row + 1),
                                            std::span<const std::byte>(binding.shadow)
                                                .subspan(row * rowBytes, rowBytes))) {
                        hostRows.push_back(row);
                    }
                }
            }
        }
        std::ranges::sort(hostRows);
        for (const std::uint32_t row : hostRows) {
            const std::vector<std::byte> bytes = DeviceBytes(host, row, row + 1);
            std::memcpy(binding.shadow.data() + row * rowBytes, bytes.data(), rowBytes);
        }
        UploadRows(binding, hostRows);
    }

    Ticket moved;
    if (!moves.empty()) {
        // The sources are rows no copy writes: every move reads a row past the end or one that
        // was removed, so one batch of regions per column is safe.
        moved = context_.Main().Submit(
            [&](VkCommandBuffer commands) {
                const VkAccessFlags2 any =
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                Barrier(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, any,
                        VK_PIPELINE_STAGE_2_COPY_BIT, any);
                for (auto& [column, regions] : moves) {
                    for (VkBufferCopy& region : regions) {
                        region.srcOffset += column->Range().offset;
                        region.dstOffset += column->Range().offset;
                    }
                    vkCmdCopyBuffer(commands, column->Storage().Handle(),
                                    column->Storage().Handle(),
                                    static_cast<std::uint32_t>(regions.size()), regions.data());
                }
                Barrier(commands, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, any);
            },
            after);
    }
    const Ticket uploaded = uploader_.Flush(after);
    return uploaded.timeline ? uploaded : (moved.timeline ? moved : context_.Main().Last());
}

}  // namespace ernie
