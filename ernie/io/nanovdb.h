// SPDX-License-Identifier: BSD-3-Clause
/// @file nanovdb.h
/// @brief NanoVDB grids: reading .nvdb/.vdb files and building grids.
#pragma once

#include "ernie/geometry/bounds.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief A NanoVDB grid's value type, numbered as
/// `PNANOVDB_GRID_TYPE_*`.
/// Volumes sample scalars and float vectors; other types load only.
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

/// @brief One sparse grid in NanoVDB layout, with its header's facts.
/// The GPU walks the buffer in place (PNanoVDB). The grid's map
/// takes index space to the grid frame (OpenVDB's "world"), which a
/// Volume prim's transform places.
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
    /// The grid's name.
    std::string name;
    /// The value type.
    NanoVdbType type = NanoVdbType::Unknown;
    /// Index coordinates to the grid frame, from the grid's map.
    glm::dmat4 indexToGrid{1.0};
    /// Lower inclusive index bound of active values; min > max when
    /// there are none.
    glm::ivec3 indexMin{0};
    /// Upper inclusive index bound of active values.
    glm::ivec3 indexMax{-1};
    /// A voxel's size in the grid frame.
    glm::dvec3 voxelSize{1.0};
    /// Whether the grid has value statistics (HAS_MIN_MAX; Warp's
    /// grids do not).
    bool hasStatistics = false;
    /// Smallest active value (vector length for vector grids); 0
    /// without statistics.
    double minValue = 0.0;
    /// Largest active value (vector length for vector grids); 0
    /// without statistics.
    double maxValue = 0.0;
    /// The grid from its GridData header on, padded to 8 bytes;
    /// empty when only the head was read (`NanoVdbGridHead`).
    std::vector<std::uint8_t> buffer;
    /// The grid's size in bytes, per its header.
    std::uint64_t size = 0;

    /// @return Whether the grid has no active values.
    bool Empty() const;
    /// @return Whether volumes sample one value per voxel.
    bool Scalar() const;
    /// @return Whether volumes sample a 3-vector per voxel.
    bool Vector() const;
    /// @return Active bounds in the grid frame, grown by a voxel on each side for trilinear
    /// sampling; the grid's origin when it has no active voxels.
    Box GridBounds() const;
};

/// @brief Reads every grid of a NanoVDB file, in file order.
/// Accepts segmented files or a raw grid buffer.
/// @param path The .nvdb file.
/// @return The grids.
/// @throws std::runtime_error If unreadable, not NanoVDB, not major
///         version 32, or zip/blosc compressed (save uncompressed:
///         `nanovdb_convert` without `-z`/`-b`).
std::vector<NanoVdbGrid> ReadNanoVdb(const std::filesystem::path& path);

/// @brief Reads the headers of a NanoVDB file's grids (names, types, bounds and transforms)
/// without their voxels, as `NanoVdbGridHead` does for each grid.
/// @throws std::runtime_error As `ReadNanoVdb`.
std::vector<NanoVdbGrid> ReadNanoVdbHeads(const std::filesystem::path& path);

/// @brief Splits a raw in-memory NanoVDB buffer into its grids.
/// @param bytes Grids back to back, as Warp's `Volume.array()`.
/// @return The grids.
/// @throws std::runtime_error If it does not start with a version 32
///         NanoVDB grid.
std::vector<NanoVdbGrid> ParseNanoVdbGrids(std::span<const std::uint8_t> bytes);
/// @brief Reads the headers of a raw buffer's grids, without copying their voxels.
/// @throws std::runtime_error As `ParseNanoVdbGrids`.
std::vector<NanoVdbGrid> ParseNanoVdbHeads(std::span<const std::uint8_t> bytes);

/// Bytes at a grid's start holding its header, tree and root, root
/// statistics included.
inline constexpr std::size_t kNanoVdbHeadBytes = 1024;

/// @brief Reads a grid's header without its buffer, for grids on a
/// device of which only the head reaches the host.
/// @param head The first `kNanoVdbHeadBytes`, or fewer for small
///             grids.
/// @return The grid; its `size` locates the next grid.
/// @throws std::runtime_error If `head` is not a version 32 NanoVDB
///         grid or ends before its root.
NanoVdbGrid NanoVdbGridHead(std::span<const std::uint8_t> head);

/// @brief Finds the grid a UsdVol OpenVDBAsset names.
/// @param grids The file's grids.
/// @param name  The grid name; empty matches any.
/// @param index Which match, 0 the first.
/// @return The grid, or null if there is none.
const NanoVdbGrid* FindNanoVdbGrid(std::span<const NanoVdbGrid> grids, std::string_view name,
                                   int index = 0);

/// @brief Builds an in-memory NanoVDB float grid (a fog volume) from
/// dense values, active where not 0, with bounds and min/max.
/// @param values    C-order (x, y, z) values, z fastest.
/// @param shape     The voxel counts along x, y and z.
/// @param voxelSize The voxel size in the grid frame; positive.
/// @param origin    Where index (0, 0, 0) lies in the grid frame.
/// @param name      The grid's name.
/// @return The grid buffer, for a NanoVDBField's array.
/// @throws std::invalid_argument If `values` does not match `shape`
///         or `voxelSize` is not positive.
///
/// References:
/// - K. Museth, "NanoVDB: A GPU-Friendly and Portable VDB Data
///   Structure For Real-Time Rendering And Simulation", ACM SIGGRAPH
///   2021 Talks. https://doi.org/10.1145/3450623.3464653
/// - PNanoVDB.h, OpenVDB 13.0.0: the layout and its setters.
std::vector<std::uint8_t> DenseNanoVdbGrid(std::span<const float> values, const glm::ivec3& shape,
                                           double voxelSize, const glm::dvec3& origin,
                                           std::string_view name = "");

/// @brief Where a `DenseNanoVdbTopology`'s values go.
/// Leaf (a, b, c) starts at firstLeaf + ((a leaves.y + b) leaves.z +
/// c) leafBytes; voxel (i, j, k) is at valuesOffset + 4 ((i & 7) 64
/// + (j & 7) 8 + (k & 7)) within it.
struct DenseNanoVdbLayout {
    /// Byte offset of the first leaf in the grid.
    std::uint64_t firstLeaf = 0;
    /// Bytes per leaf.
    std::uint32_t leafBytes = 0;
    /// Byte offset of a leaf's values.
    std::uint32_t valuesOffset = 0;
    /// Leaves (8^3 blocks) along x, y and z.
    glm::ivec3 leaves{0};
};

/// @brief As `DenseNanoVdbGrid` with every voxel active and 0, for
/// values written in place later (a DenseField on the GPU).
/// No min and max.
/// @param shape     The voxel counts along x, y and z.
/// @param voxelSize The voxel size in the grid frame; positive.
/// @param origin    Where index (0, 0, 0) lies in the grid frame.
/// @param name      The grid's name.
/// @param layout    Receives where the values go.
/// @return The grid buffer.
/// @throws std::invalid_argument For a negative shape or a voxel
///         size that is not positive.
std::vector<std::uint8_t> DenseNanoVdbTopology(const glm::ivec3& shape, double voxelSize,
                                               const glm::dvec3& origin, std::string_view name,
                                               DenseNanoVdbLayout& layout);

}  // namespace ernie
