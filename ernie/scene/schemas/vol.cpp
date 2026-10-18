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

/// @return The bounds of the grid a field selects by name and index; empty if it has none.
Box SelectedGrid(std::span<const NanoVdbGrid> grids, std::string_view name, int index) {
    const NanoVdbGrid* grid = FindNanoVdbGrid(grids, name, index);
    return grid != nullptr ? grid->GridBounds() : Box();
}

/// File fields from the file's grid headers, without reading voxels.
void FileFieldExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto path = rows.Column<Token>(FieldAsset::kFilePath);
    const auto name = rows.Column<Token>(FieldAsset::kFieldName);
    const auto index = rows.Column<std::int32_t>(FieldAsset::kFieldIndex);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        Box& box = out[row - range.begin] = Box();
        try {
            const std::vector<NanoVdbGrid> heads =
                ReadNanoVdbHeads(std::string(rows.Text(path[row])));
            box = SelectedGrid(heads, rows.Text(name[row]), index[row]);
        } catch (const std::runtime_error&) {
            // A missing or unreadable file bounds nothing; the renderer reports it.
        }
    }
}

/// In-memory NanoVDB fields from their grids' headers; none for grids still on a device.
void NanoVdbFieldExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto buffer = rows.Column<Tensor>(NanoVDBField::kBuffer);
    const auto name = rows.Column<Token>(NanoVDBField::kFieldName);
    const auto index = rows.Column<std::int32_t>(NanoVDBField::kFieldIndex);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        Box& box = out[row - range.begin] = Box();
        const Tensor& grids = buffer[row];
        if (grids.Empty() || IsGpu(grids.Location().kind) || !grids.IsContiguous()) {
            continue;
        }
        const std::span<const std::uint8_t> bytes(static_cast<const std::uint8_t*>(grids.Data()),
                                                  static_cast<std::size_t>(grids.Elements()) *
                                                      ByteSize(grids.Type()));
        try {
            box = SelectedGrid(ParseNanoVdbHeads(bytes), rows.Text(name[row]), index[row]);
        } catch (const std::runtime_error&) {
        }
    }
}

/// Dense fields from their shape: values at the voxel size times their indices, grown by a voxel
/// for what trilinear sampling reaches.
void DenseFieldExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto values = rows.Column<Tensor>(DenseField::kValues);
    const auto voxel = rows.Column<double>(DenseField::kVoxelSize);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        const Tensor& grid = values[row];
        out[row - range.begin] =
            grid.Rank() == 3
                ? Box{glm::dvec3(-voxel[row]),
                      glm::dvec3(grid.Shape()[0], grid.Shape()[1], grid.Shape()[2]) * voxel[row]}
                : Box();
    }
}

/// Volumes from their fields' extents, each brought into the volume's space.
void VolumeExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const Bucket& bucket = rows.store.GetBucket(rows.bucket);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        Box& box = out[row - range.begin] = Box();
        const glm::dmat4 toVolume = glm::inverse(rows.scene->WorldTransform(bucket.prims[row]));
        for (const Column& column : bucket.columns) {
            if (column.values->Type() != ColumnType::Relationship || !column.authored.Test(row) ||
                !rows.Text(column.attribute).starts_with(tokens::kFieldPrefix)) {
                continue;
            }
            const Target target = rows.store.Read<Target>(rows.bucket, column.attribute)[row];
            const std::optional<PrimId> field =
                target.path != kNoPath ? rows.store.Find(target.path) : std::nullopt;
            const std::optional<Box> extent = field ? rows.scene->Extent(*field) : std::nullopt;
            if (extent) {
                box.Extend(extent->Transformed(toVolume * rows.scene->WorldTransform(*field)));
            }
        }
    }
}

}  // namespace

Volume& Volume::SetField(std::string_view name, const Path& field) {
    Edit().Set(Id(), std::string(tokens::kFieldPrefix) + std::string(name), field);
    return *this;
}

ERNIE_DEFINE_SCHEMA(Volume, tokens::kVolume, &Gprim::Definition(), Concrete,
                    ComputedFromOtherPrims(VolumeExtent),
                    "A volume, bounded by and rendered from the fields it names.",
                    ERNIE_NO_ATTRIBUTES, .openNamespaces = {tokens::kFieldPrefix})
ERNIE_DEFINE_SCHEMA(FieldBase, "FieldBase", &Xformable::Definition(), Abstract, kInheritExtent,
                    "A field's frame and grid, which volumes name.", ERNIE_NO_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(FieldAsset, "FieldAsset", &FieldBase::Definition(), Abstract,
                    ComputedExtent(FileFieldExtent), "A field read from a file.",
                    ERNIE_FIELD_ASSET_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(OpenVDBAsset, tokens::kOpenVdbAsset, &FieldAsset::Definition(), Concrete,
                    kInheritExtent, "An OpenVDB grid, read once converted to NanoVDB.",
                    ERNIE_OPENVDB_ASSET_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(NanoVDBField, tokens::kNanoVdbField, &FieldBase::Definition(), Concrete,
                    ComputedExtent(NanoVdbFieldExtent),
                    "NanoVDB grids in memory, written each step by a simulation.",
                    ERNIE_NANOVDB_FIELD_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(DenseField, tokens::kDenseField, &FieldBase::Definition(), Concrete,
                    ComputedExtent(DenseFieldExtent),
                    "A dense grid of values, as solvers write them.", ERNIE_DENSE_FIELD_ATTRIBUTES)

}  // namespace ernie
