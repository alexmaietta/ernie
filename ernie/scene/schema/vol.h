// SPDX-License-Identifier: BSD-3-Clause
/// @file vol.h
/// @brief UsdVol's volumes and fields, with Ernie's in-memory NanoVDB and dense fields.
///
/// References:
/// - Pixar, UsdVol, https://openusd.org/release/api/usd_vol_page_front.html
#pragma once

#include "ernie/scene/schema/geom.h"

// clang-format off
#define ERNIE_FIELD_SELECTION_ATTRIBUTES(X, where)                                                 \
    X(std::string, FieldName, "fieldName", "", "The grid's name in the " where "; unset, any grid.", \
      .group = "Field", .dirty = Dirty::Points)                                                    \
    X(int, FieldIndex, "fieldIndex", 0,                                                            \
      "Which of the " where "'s grids of that name, or of all its grids without one; 0 the first.", \
      .group = "Field", .lower = AtLeast(0.0), .dirty = Dirty::Points)

#define ERNIE_FIELD_ASSET_ATTRIBUTES(X)                                                            \
    X(std::string, FilePath, "filePath", "", "The grid's file: NanoVDB, uncompressed.",            \
      .group = "Field", .asset = true, .dirty = Dirty::Points)                                     \
    ERNIE_FIELD_SELECTION_ATTRIBUTES(X, "file")                                                    \
    X(std::string, FieldDataType, "fieldDataType", "",                                             \
      "The grid's value type, for reference; Ernie reads it from the file.", .group = "Field")     \
    X(std::string, VectorDataRoleHint, "vectorDataRoleHint", "None", "What a vector grid holds.",  \
      .group = "Field", .options = {"None", "Point", "Normal", "Vector", "Color"})

#define ERNIE_OPENVDB_ASSET_ATTRIBUTES(X)                                                          \
    X(std::string, FieldClass, "fieldClass", "",                                                   \
      "The grid's class, for reference: Ernie renders fog volumes, not level sets.",               \
      .group = "Field", .options = {"levelSet", "fogVolume", "staggered", "unknown"})

#define ERNIE_NANOVDB_FIELD_ATTRIBUTES(X)                                                          \
    X(Tensor, Buffer, "ernie:buffer", Tensor(),                                                    \
      "NanoVDB grids back to back as NanoVDB keeps them in memory (Warp's Volume.array()), on "    \
      "the host or a device.",                                                                     \
      .group = "Field", .bytes = true, .dirty = Dirty::Points)                                     \
    ERNIE_FIELD_SELECTION_ATTRIBUTES(X, "buffer")

#define ERNIE_DENSE_FIELD_ATTRIBUTES(X)                                                            \
    X(Tensor, Values, "ernie:values", Tensor(),                                                    \
      "float32 (x, y, z) values, z fastest, on the host or a device; 0 is empty.",                 \
      .group = "Field", .ranks = {3}, .dirty = Dirty::Points)                                      \
    X(double, VoxelSize, "ernie:voxelSize", 1.0,                                                   \
      "Metres between values: value (i, j, k) lies at the voxel size times (i, j, k).",            \
      .group = "Field", .lower = Above(0.0), .finite = true, .dirty = Dirty::Points)
// clang-format on

namespace ernie {

ERNIE_SCHEMA_ACCESSORS(Volume, Gprim, ERNIE_NO_ATTRIBUTES);

class Volume : public VolumeAccessors<Volume, Gprim> {
    ERNIE_SCHEMA_BODY(Volume, Gprim,
                      "A volume, bounded by and rendered from the fields its field:<name> "
                      "relationships name.")
    ERNIE_SCHEMA_DEFINE(Volume)
    /// Names a field of the volume, as `field:<name>`.
    Volume& SetField(std::string_view name, const Path& field);
};

ERNIE_SCHEMA(FieldBase, Xformable, ERNIE_NO_ATTRIBUTES,
             "A field's frame and grid, which volumes name.");
ERNIE_SCHEMA(FieldAsset, FieldBase, ERNIE_FIELD_ASSET_ATTRIBUTES, "A field read from a file.");
ERNIE_CONCRETE_SCHEMA(OpenVDBAsset, FieldAsset, ERNIE_OPENVDB_ASSET_ATTRIBUTES,
                      "An OpenVDB grid, read once converted to NanoVDB.");
ERNIE_CONCRETE_SCHEMA(NanoVDBField, FieldBase, ERNIE_NANOVDB_FIELD_ATTRIBUTES,
                      "NanoVDB grids in memory, written each step by a simulation.");
ERNIE_CONCRETE_SCHEMA(DenseField, FieldBase, ERNIE_DENSE_FIELD_ATTRIBUTES,
                      "A dense grid of values, as solvers write them.");

}  // namespace ernie
