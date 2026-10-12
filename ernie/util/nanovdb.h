// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// A NanoVDB grid's value type, as `PNANOVDB_GRID_TYPE_*` numbers it.
/// Volumes read the scalar types and the float vectors; other types
/// load but are not sampled.
enum class NanoVdbType : std::uint32_t {
    Unknown = 0,
    Float = 1,
    Double = 2,
    Vec3f = 6,
    Half = 9,
    Fp4 = 13,
    Fp8 = 14,
    Fp16 = 15,
    FpN = 16,
    Vec4f = 17,
};

/// The name `PNANOVDB_GRID_TYPE_*` gives a grid type, lower case
/// ("float", "fp16", "vec3f"), or "type <n>" for others.
std::string NanoVdbTypeName(NanoVdbType type);

/// One sparse grid as NanoVDB lays it out: a buffer the GPU reads in
/// place (PNanoVDB's tree walk), and what Ernie needs of its header.
///
/// Index space puts a value at each integer coordinate; the grid's
/// map takes index coordinates to the grid's own frame (OpenVDB's
/// "world", which a Volume prim's transform then places).
///
/// References:
/// - K. Museth, "NanoVDB: A GPU-Friendly and Portable VDB Data
///   Structure For Real-Time Rendering And Simulation", ACM SIGGRAPH
///   2021 Talks. https://doi.org/10.1145/3450623.3464653
/// - K. Museth, "VDB: High-Resolution Sparse Volumes with Dynamic
///   Topology", ACM Transactions on Graphics 32(3), 27, 2013.
///   https://doi.org/10.1145/2487228.2487235
/// - NanoVDB's file layout, nanovdb/io/IO.h and FileMetaData in
///   nanovdb/NanoVDB.h, OpenVDB 13.0.0.
struct NanoVdbGrid {
    std::string name;
    NanoVdbType type = NanoVdbType::Unknown;
    /// Index coordinates to the grid's frame, from the grid's map
    /// (doubles).
    glm::dmat4 indexToGrid{1.0};
    /// The active values' bounds in index space, inclusive; empty
    /// (min > max) for a grid without active values.
    glm::ivec3 indexMin{0};
    glm::ivec3 indexMax{-1};
    /// The size of a voxel in the grid's frame.
    glm::dvec3 voxelSize{1.0};
    /// Whether the grid carries its values' statistics (NanoVDB's
    /// HAS_MIN_MAX flag; Warp's grids do not).
    bool hasStatistics = false;
    /// With them, the smallest and largest active value of a scalar
    /// grid (the root's; for vector grids, of their lengths); else 0.
    double minValue = 0.0;
    double maxValue = 0.0;
    /// The grid itself, from its GridData header on, padded to a
    /// multiple of 4 bytes.
    std::vector<std::uint8_t> buffer;

    /// Whether the grid holds active values.
    bool Empty() const;
    /// Whether volumes sample it as one value per voxel.
    bool Scalar() const;
    /// Whether volumes sample it as a 3-vector per voxel.
    bool Vector() const;
    /// The active bounds in the grid's frame: the index bounds grown
    /// by a voxel on every side (what trilinear sampling reaches),
    /// through `indexToGrid`.
    void GridBounds(glm::dvec3& min, glm::dvec3& max) const;
};

/// Every grid in a NanoVDB file (.nvdb), in file order: a file of
/// segments (header, per-grid metadata and names, then the grids)
/// or a raw grid buffer.
/// @throws std::runtime_error if the file cannot be read, is not a
/// NanoVDB file, has a major version other than 32, or is compressed
/// (zip or blosc); write such files uncompressed (`nanovdb_convert`
/// without `-z`/`-b`, or Warp's `Volume.save_to_nvdb`).
std::vector<NanoVdbGrid>
ReadNanoVdb(const std::filesystem::path& path);

/// The grids of a raw NanoVDB buffer, one or more grids back to back
/// as NanoVDB keeps them in memory (Warp's `Volume.array()`).
/// @throws std::runtime_error if the buffer does not start with a
/// NanoVDB grid of major version 32.
std::vector<NanoVdbGrid>
NanoVdbGridsOf(std::span<const std::uint8_t> bytes);

/// The grid UsdVol's OpenVDBAsset names: the `index`th grid (0 the
/// first) named `name`, or the `index`th grid of the file when `name`
/// is empty; nullptr if there is none.
const NanoVdbGrid* FindNanoVdbGrid(std::span<const NanoVdbGrid> grids,
                                   std::string_view name,
                                   int index = 0);

}  // namespace ernie
