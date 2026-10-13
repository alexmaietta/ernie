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
                                     "type, on the CPU or a CUDA device. Written again in the "    \
                                     "same device memory, as a simulation steps its grids, "       \
                                     "their layout is taken as kept; a new array is read anew.",   \
                              .group = "Field", .dirty = ::ernie::Dirty::Points}))                 \
    X(std::string, FieldName, ::ernie::tokens::kFieldName, "",                                     \
      (::ernie::AttributeMeta{.doc = "The grid's name in the buffer; unset, any grid.",            \
                              .group = "Field", .dirty = ::ernie::Dirty::Points}))                 \
    X(int, FieldIndex, ::ernie::tokens::kFieldIndex, 0,                                            \
      (::ernie::AttributeMeta{.doc = "Which of the buffer's grids of that name (or of all its "    \
                                     "grids, without a name), 0 the first.",                       \
                              .group = "Field", .min = 0.0, .max = 1e6,                            \
                              .dirty = ::ernie::Dirty::Points}))

#define ERNIE_DENSE_FIELD_ATTRIBUTES(X)                                                            \
    X(::ernie::Tensor, Values, ::ernie::tokens::kDenseValues, nullptr,                             \
      (::ernie::AttributeMeta{.doc = "float32 (x, y, z) values, z fastest, on the CPU or a CUDA "  \
                                     "device; 0 is empty. Written again on the device at its "     \
                                     "shape, as a solver steps, it is copied into place on the "   \
                                     "GPU with no work on the CPU.",                               \
                              .group = "Field", .dirty = ::ernie::Dirty::Points}))                 \
    X(double, VoxelSize, ::ernie::tokens::kVoxelSize, 1.0,                                         \
      (::ernie::AttributeMeta{.doc = "World units between values: value (i, j, k) is at the "      \
                                     "voxel size times (i, j, k) in the field's frame.",           \
                              .group = "Field", .min = 1e-9, .max = 1e9,                           \
                              .dirty = ::ernie::Dirty::Points}))

#define ERNIE_CLOUD_LAYERS_FIELD_ATTRIBUTES(X)                                                     \
    X(double, Coverage, ::ernie::tokens::kCloudCoverage, 0.3,                                      \
      (::ernie::AttributeMeta{.doc = "How much of the sky the layers cover, over the weather map: " \
                                     "0 clear, 1 overcast.",                                       \
                              .group = "Clouds", .min = 0.0, .max = 1.0}))                         \
    X(double, Extent, ::ernie::tokens::kCloudExtent, 20000.0,                                      \
      (::ernie::AttributeMeta{.doc = "The layers' reach across the field's frame (world units) "    \
                                     "from its origin in x and y, z up: the volume's bounds.",     \
                              .group = "Clouds", .min = 1.0}))                                     \
    X(double, StepSize, ::ernie::tokens::kCloudStepSize, 50.0,                                     \
      (::ernie::AttributeMeta{.doc = "World units the march steps through the layers as a grid's "  \
                                     "voxel, before the volume's steps per voxel.",                \
                              .group = "Clouds", .min = 1e-3}))                                    \
    X(double, WeatherTileSize, ::ernie::tokens::kCloudWeatherTileSize, 100000.0,                   \
      (::ernie::AttributeMeta{.doc = "World units across the repeating weather map: where clouds " \
                                     "form, each layer from its channel.",                         \
                              .group = "Clouds", .min = 1e-3}))                                    \
    X(glm::dvec2, WeatherOffset, ::ernie::tokens::kCloudWeatherOffset, glm::dvec2(0.0),             \
      (::ernie::AttributeMeta{.doc = "The weather map's offset (world units), as wind moves it.",  \
                              .group = "Clouds"}))                                                 \
    X(double, ShapeTileSize, ::ernie::tokens::kCloudShapeTileSize, 3333.0,                         \
      (::ernie::AttributeMeta{.doc = "World units across the repeating Perlin-Worley shapes.",      \
                              .group = "Clouds", .min = 1e-3}))                                    \
    X(double, DetailTileSize, ::ernie::tokens::kCloudDetailTileSize, 166.7,                        \
      (::ernie::AttributeMeta{.doc = "World units across the repeating Worley details that erode " \
                                     "the shapes' edges.",                                         \
                              .group = "Clouds", .min = 1e-3}))                                    \
    X(double, TurbulenceTileSize, ::ernie::tokens::kCloudTurbulenceTileSize, 5000.0,               \
      (::ernie::AttributeMeta{.doc = "World units across the repeating curl-noise turbulence.",     \
                              .group = "Clouds", .min = 1e-3}))                                    \
    X(double, TurbulenceDisplacement, ::ernie::tokens::kCloudTurbulenceDisplacement, 350.0,        \
      (::ernie::AttributeMeta{.doc = "World units the turbulence displaces the shapes, most at "   \
                                     "their bottoms.",                                             \
                              .group = "Clouds", .min = 0.0}))                                     \
    X(std::vector<float>, LayerAltitudes, ::ernie::tokens::kCloudLayerAltitudes,                   \
      (std::vector<float>{750.0f, 1000.0f, 7500.0f}),                                              \
      (::ernie::AttributeMeta{.doc = "Each layer's bottom (z, world units), up to four layers: "   \
                                     "three-geospatial's low and mid cumulus and cirrus.",         \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerHeights, ::ernie::tokens::kCloudLayerHeights,                       \
      (std::vector<float>{650.0f, 1200.0f, 500.0f}),                                               \
      (::ernie::AttributeMeta{.doc = "Each layer's height (world units); 0 none.",                 \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerDensities, ::ernie::tokens::kCloudLayerDensities,                   \
      (std::vector<float>{0.2f, 0.2f, 0.003f}),                                                    \
      (::ernie::AttributeMeta{.doc = "Each layer's density scale, its value at its densest, "      \
                                     "times the material's density.",                              \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerShapeAmounts, ::ernie::tokens::kCloudLayerShapeAmounts,             \
      (std::vector<float>{1.0f, 1.0f, 0.4f}),                                                      \
      (::ernie::AttributeMeta{.doc = "How much each layer's shape noise carves it.",               \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerDetailAmounts, ::ernie::tokens::kCloudLayerDetailAmounts,           \
      (std::vector<float>{1.0f, 1.0f, 0.0f}),                                                      \
      (::ernie::AttributeMeta{.doc = "How much each layer's detail noise erodes its edges.",       \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerWeatherExponents, ::ernie::tokens::kCloudLayerWeatherExponents,     \
      (std::vector<float>{1.0f, 1.0f, 1.0f}),                                                      \
      (::ernie::AttributeMeta{.doc = "Each layer's weather map's value to this power.",            \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerShapeBiases, ::ernie::tokens::kCloudLayerShapeBiases,               \
      (std::vector<float>{0.35f, 0.35f, 0.35f}),                                                   \
      (::ernie::AttributeMeta{.doc = "Each layer's height fraction to this power before its "      \
                                     "rounding: below 1, flatter bottoms and rounder tops.",       \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerCoverageFilterWidths,                                               \
      ::ernie::tokens::kCloudLayerCoverageFilterWidths, (std::vector<float>{0.6f, 0.6f, 0.5f}),    \
      (::ernie::AttributeMeta{.doc = "Each layer's coverage's softness, over its weather.",        \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerDensityProfiles, ::ernie::tokens::kCloudLayerDensityProfiles,       \
      (std::vector<float>{0.0f, 0.0f, 0.75f, 0.25f, 0.0f, 0.0f, 0.75f, 0.25f, 0.0f, 0.0f, 0.75f,   \
                          0.25f}),                                                                 \
      (::ernie::AttributeMeta{.doc = "Each layer's a, b, c and d: its density times a e^(b h) + c " \
                                     "h + d over its height fraction h, denser at its top.",       \
                              .group = "Clouds"}))                                                 \
    X(std::vector<float>, LayerChannels, ::ernie::tokens::kCloudLayerChannels,                     \
      (std::vector<float>{0.0f, 1.0f, 2.0f}),                                                      \
      (::ernie::AttributeMeta{.doc = "Each layer's weather map channel: 0 low cumulus cells, 1 "    \
                                     "mid, 2 high streaks, 3 everywhere.",                         \
                              .group = "Clouds"}))

#define ERNIE_FIELD_BASE_ATTRIBUTES(X)
#define ERNIE_VOLUME_ATTRIBUTES(X)

#define ERNIE_FIELD_BASE_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_FIELD_BASE_ATTRIBUTES(X)
#define ERNIE_FIELD_ASSET_ALL(X) ERNIE_FIELD_BASE_ALL(X) ERNIE_FIELD_ASSET_ATTRIBUTES(X)
#define ERNIE_OPENVDB_ASSET_ALL(X) ERNIE_FIELD_ASSET_ALL(X) ERNIE_OPENVDB_ASSET_ATTRIBUTES(X)
#define ERNIE_NANOVDB_FIELD_ALL(X) ERNIE_FIELD_BASE_ALL(X) ERNIE_NANOVDB_FIELD_ATTRIBUTES(X)
#define ERNIE_DENSE_FIELD_ALL(X) ERNIE_FIELD_BASE_ALL(X) ERNIE_DENSE_FIELD_ATTRIBUTES(X)
#define ERNIE_CLOUD_LAYERS_FIELD_ALL(X)                                                            \
    ERNIE_FIELD_BASE_ALL(X) ERNIE_CLOUD_LAYERS_FIELD_ATTRIBUTES(X)
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
    ERNIE_SCHEMA_BODY(NanoVDBField, FieldBase,
                      ERNIE_NANOVDB_FIELD_ALL)
    ERNIE_SCHEMA_DEFINE(NanoVDBField)
};

/// Ernie's field of a dense array of values, such as a grid solver's
/// smoke or dust each step (Warp's arrays): sampled as a NanoVDB grid
/// of every voxel, which a device array fills in place on the GPU.
class DenseField : public FieldBase {
    ERNIE_SCHEMA_BODY(DenseField, FieldBase, ERNIE_DENSE_FIELD_ALL)
    ERNIE_SCHEMA_DEFINE(DenseField)
};

/// Ernie's procedural cloud layers: up to four layers over a
/// repeating weather map, carved from tileable noise at each sample on
/// the GPU (three-geospatial's clouds, after Horizon Zero Dawn's), in
/// the field's frame with z up; a Volume's density, its material's
/// density scaling it.
class CloudLayersField : public FieldBase {
    ERNIE_SCHEMA_BODY(CloudLayersField, FieldBase,
                      ERNIE_CLOUD_LAYERS_FIELD_ALL)
    ERNIE_SCHEMA_DEFINE(CloudLayersField)
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
