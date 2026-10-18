// SPDX-License-Identifier: BSD-3-Clause
/// @file vol.cpp
/// @brief Volumes' and fields' registry entries and extents: fields from their grids' headers,
/// volumes from their fields.
#include "ernie/scene/schemas/vol.h"

#include "ernie/io/nanovdb.h"

#include <glm/glm.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

/// @return The bounds of the grid a field selects by name and index from the headers `read`
/// returns; empty when it has none, or its source is missing or unreadable (the renderer reports
/// those).
template <typename Read> Box SelectedGrid(const Read& read, std::string_view name, int index) {
    try {
        const std::vector<NanoVdbGrid> heads = read();
        const NanoVdbGrid* grid = FindNanoVdbGrid(heads, name, index);
        return grid != nullptr ? grid->GridBounds() : Box();
    } catch (const std::runtime_error&) {
        return {};
    }
}

/// File fields from the file's grid headers, without reading voxels.
void FileFieldExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto path = rows.Column<Token>(FieldAsset::kFilePath);
    const auto name = rows.Column<Token>(FieldAsset::kFieldName);
    const auto index = rows.Column<std::int32_t>(FieldAsset::kFieldIndex);
    ComputeRows(range, out, [&](std::uint32_t row) {
        return SelectedGrid([&] { return ReadNanoVdbHeads(std::string(rows.Text(path[row]))); },
                            rows.Text(name[row]), index[row]);
    });
}

/// In-memory NanoVDB fields from their grids' headers; none for grids still on a device.
void NanoVdbFieldExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto buffer = rows.Column<Tensor>(NanoVDBField::kBuffer);
    const auto name = rows.Column<Token>(NanoVDBField::kFieldName);
    const auto index = rows.Column<std::int32_t>(NanoVDBField::kFieldIndex);
    ComputeRows(range, out, [&](std::uint32_t row) {
        const Tensor& grids = buffer[row];
        if (grids.Empty() || IsGpu(grids.Location().kind) || !grids.IsContiguous()) {
            return Box();
        }
        const std::span<const std::uint8_t> bytes(static_cast<const std::uint8_t*>(grids.Data()),
                                                  static_cast<std::size_t>(grids.Elements()) *
                                                      ByteSize(grids.Type()));
        return SelectedGrid([&] { return ParseNanoVdbHeads(bytes); }, rows.Text(name[row]),
                            index[row]);
    });
}

/// Dense fields from their shape: values at the voxel size times their indices, grown by a voxel
/// for what trilinear sampling reaches.
void DenseFieldExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto values = rows.Column<Tensor>(DenseField::kValues);
    const auto voxel = rows.Column<double>(DenseField::kVoxelSize);
    ComputeRows(range, out, [&](std::uint32_t row) {
        const Tensor& grid = values[row];
        if (grid.Rank() != 3) {
            return Box();
        }
        const glm::dvec3 shape(grid.Shape()[0], grid.Shape()[1], grid.Shape()[2]);
        return Box{glm::dvec3(-voxel[row]), shape * voxel[row]};
    });
}

/// Volumes from their fields' extents, each brought into the volume's space.
void VolumeExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const Bucket& bucket = rows.store.GetBucket(rows.bucket);
    ComputeRows(range, out, [&](std::uint32_t row) {
        Box box;
        for (const Column& column : bucket.columns) {
            if (column.values->Type() != ColumnType::Relationship || !column.authored.Test(row) ||
                !rows.Text(column.attribute).starts_with(tokens::kFieldPrefix)) {
                continue;
            }
            const auto field =
                rows.Resolve(rows.store.Read<Target>(rows.bucket, column.attribute)[row]);
            if (const std::optional<Box> extent =
                    field ? rows.ExtentInRowSpace(*field, row) : std::nullopt) {
                box.Extend(*extent);
            }
        }
        return box;
    });
}

}  // namespace

Volume& Volume::SetField(std::string_view name, const Path& field) {
    Edit().Set(Id(), std::string(tokens::kFieldPrefix) + std::string(name), field);
    return *this;
}

ERNIE_DEFINE_SCHEMA(Volume, Concrete, ComputedFromOtherPrims(VolumeExtent), ERNIE_NO_ATTRIBUTES,
                    .openNamespaces = {tokens::kFieldPrefix})
ERNIE_DEFINE_SCHEMA(FieldBase, Abstract, kInheritExtent, ERNIE_NO_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(FieldAsset, Abstract, ComputedExtent(FileFieldExtent),
                    ERNIE_FIELD_ASSET_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(OpenVDBAsset, Concrete, kInheritExtent, ERNIE_OPENVDB_ASSET_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(NanoVDBField, Concrete, ComputedExtent(NanoVdbFieldExtent),
                    ERNIE_NANOVDB_FIELD_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(DenseField, Concrete, ComputedExtent(DenseFieldExtent),
                    ERNIE_DENSE_FIELD_ATTRIBUTES)

}  // namespace ernie
