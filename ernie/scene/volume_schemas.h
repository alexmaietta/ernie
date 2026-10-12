// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <string>
#include <string_view>
#include <vector>

/// Volumes as reflected schemas (`prim_type.h`), after UsdVol: a
/// Volume names its fields with `field:<name>` relationships (an open
/// namespace), and each field is a prim of its own, such as an
/// OpenVDBAsset naming a grid of a NanoVDB file. How the fields shade
/// is the Volume's bound Material's volume inputs
/// (`material_schemas.h`).
///
/// Reference: Pixar, UsdVol,
/// https://openusd.org/release/api/usd_vol_page_front.html

// clang-format off
#define ERNIE_FIELD_ASSET_ATTRIBUTES(X)                                                            \
    X(std::string, FilePath, ::ernie::tokens::kFilePath, "",                                       \
      (::ernie::AttributeMeta{.doc = "The grid's file: NanoVDB (.nvdb), uncompressed; convert "    \
                                     ".vdb files first (nanovdb_convert).",                        \
                              .group = "Field", .asset = true,                                     \
                              .dirty = ::ernie::Dirty::Points}))                                   \
    X(std::string, FieldName, ::ernie::tokens::kFieldName, "",                                     \
      (::ernie::AttributeMeta{.doc = "The grid's name in the file; unset, any grid.",              \
                              .group = "Field", .dirty = ::ernie::Dirty::Points}))                 \
    X(int, FieldIndex, ::ernie::tokens::kFieldIndex, 0,                                            \
      (::ernie::AttributeMeta{.doc = "Which of the file's grids of that name (or of all its "      \
                                     "grids, without a name), 0 the first.",                       \
                              .group = "Field", .min = 0.0, .max = 1e6,                            \
                              .dirty = ::ernie::Dirty::Points}))                                   \
    X(std::string, FieldDataType, ::ernie::tokens::kFieldDataType, "",                             \
      (::ernie::AttributeMeta{.doc = "The grid's value type, for reference; Ernie reads it from "  \
                                     "the file.",                                                  \
                              .group = "Field"}))                                                  \
    X(std::string, VectorDataRoleHint, ::ernie::tokens::kVectorDataRoleHint, "None",               \
      (::ernie::AttributeMeta{.doc = "What a vector grid holds.", .group = "Field",                \
                              .options = {"None", "Point", "Normal", "Vector", "Color"}}))

#define ERNIE_OPENVDB_ASSET_ATTRIBUTES(X)                                                          \
    X(std::string, FieldClass, ::ernie::tokens::kFieldClass, "",                                   \
      (::ernie::AttributeMeta{.doc = "The grid's class, for reference: a fog volume (densities) "  \
                                     "Ernie renders; a level set it does not.",                    \
                              .group = "Field",                                                    \
                              .options = {"", "levelSet", "fogVolume", "staggered", "unknown"}}))

#define ERNIE_NANOVDB_FIELD_ATTRIBUTES(X)                                                          \
    X(::ernie::Tensor, Buffer, ::ernie::tokens::kNanoVdbBuffer, nullptr,                           \
      (::ernie::AttributeMeta{.doc = "One or more NanoVDB grids back to back as NanoVDB keeps "    \
                                     "them in memory (Warp's Volume.array()), in an array of any " \
                                     "type, on the CPU or a CUDA device.",                         \
                              .group = "Field", .dirty = ::ernie::Dirty::Points}))                 \
    X(std::string, FieldName, ::ernie::tokens::kFieldName, "",                                     \
      (::ernie::AttributeMeta{.doc = "The grid's name in the buffer; unset, any grid.",            \
                              .group = "Field", .dirty = ::ernie::Dirty::Points}))                 \
    X(int, FieldIndex, ::ernie::tokens::kFieldIndex, 0,                                            \
      (::ernie::AttributeMeta{.doc = "Which of the buffer's grids of that name (or of all its "    \
                                     "grids, without a name), 0 the first.",                       \
                              .group = "Field", .min = 0.0, .max = 1e6,                            \
                              .dirty = ::ernie::Dirty::Points}))

#define ERNIE_FIELD_BASE_ATTRIBUTES(X)
#define ERNIE_VOLUME_ATTRIBUTES(X)

#define ERNIE_FIELD_BASE_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_FIELD_BASE_ATTRIBUTES(X)
#define ERNIE_FIELD_ASSET_ALL(X) ERNIE_FIELD_BASE_ALL(X) ERNIE_FIELD_ASSET_ATTRIBUTES(X)
#define ERNIE_OPENVDB_ASSET_ALL(X) ERNIE_FIELD_ASSET_ALL(X) ERNIE_OPENVDB_ASSET_ATTRIBUTES(X)
#define ERNIE_NANOVDB_FIELD_ALL(X) ERNIE_FIELD_BASE_ALL(X) ERNIE_NANOVDB_FIELD_ATTRIBUTES(X)
#define ERNIE_VOLUME_ALL(X) ERNIE_GPRIM_ALL(X) ERNIE_VOLUME_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// A field of values over space, which Volumes name; its transform
/// places the field in its Volume (UsdVol's FieldBase).
class FieldBase : public Imageable {
    ERNIE_SCHEMA_BODY(FieldBase, Imageable, ERNIE_FIELD_BASE_ALL)
};

/// A field read from a file (UsdVol's FieldAsset).
class FieldAsset : public FieldBase {
    ERNIE_SCHEMA_BODY(FieldAsset, FieldBase, ERNIE_FIELD_ASSET_ALL)
};

/// A grid of a VDB file. Ernie reads NanoVDB files (.nvdb) as they
/// are, the GPU walking the grid's tree in place.
class OpenVDBAsset : public FieldAsset {
    ERNIE_SCHEMA_BODY(OpenVDBAsset, FieldAsset,
                      ERNIE_OPENVDB_ASSET_ALL)
    ERNIE_SCHEMA_DEFINE(OpenVDBAsset)
};

/// Ernie's field of NanoVDB grids written as an array instead of a
/// file, such as a simulation's each step (Warp's `Volume.array()`):
/// one of its grids, read as an OpenVDBAsset's.
class NanoVDBField : public FieldBase {
    ERNIE_SCHEMA_BODY(NanoVDBField, FieldBase, ERNIE_NANOVDB_FIELD_ALL)
    ERNIE_SCHEMA_DEFINE(NanoVDBField)
};

/// A participating medium: fields, each named by a relationship
/// `field:<name>` to a field prim, shaded by the bound Material's
/// volume inputs (density, albedo, emission, temperature). Its
/// transform places its fields, each also through its own.
class Volume : public Gprim {
    ERNIE_SCHEMA_BODY(Volume, Gprim, ERNIE_VOLUME_ALL)
    ERNIE_SCHEMA_DEFINE(Volume)

    /// Names `field` as the field `name` ("density").
    Volume& SetField(std::string_view name, const Path& field);
    /// The field named `name`, or the empty path.
    Path GetField(std::string_view name) const;
    /// Whether it names a field `name`.
    bool HasField(std::string_view name) const;
    /// Removes the field `name`.
    Volume& ClearField(std::string_view name);
    /// The names of its fields, sorted.
    std::vector<std::string> FieldNames() const;
};

/// The relationship naming a Volume's field `name`: "field:<name>".
std::string FieldRelationship(std::string_view name);

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> VolumeSchemaTypes();

}  // namespace ernie
