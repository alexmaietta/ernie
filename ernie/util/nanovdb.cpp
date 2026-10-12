// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/nanovdb.h"

#define PNANOVDB_C
#include <PNanoVDB.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace ernie {

namespace {

constexpr std::uint64_t kMagicNumber =
    0x304244566f6e614eULL;  // NanoVDB0
constexpr std::uint64_t kMagicGrid =
    0x314244566f6e614eULL;  // NanoVDB1
constexpr std::uint64_t kMagicFile =
    0x324244566f6e614eULL;  // NanoVDB2
constexpr std::uint32_t kMajorVersion = 32;
// sizeof(FileHeader) and sizeof(FileMetaData), with the offsets
// read from the metadata (nanovdb/NanoVDB.h).
constexpr std::size_t kFileHeaderSize = 16;
constexpr std::size_t kMetaDataSize = 176;
constexpr std::size_t kMetaGridSize = 0;
constexpr std::size_t kMetaFileSize = 8;
constexpr std::size_t kMetaNameSize = 136;

template <typename T>
T Load(std::span<const std::uint8_t> bytes, std::size_t offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("NanoVDB data ends early");
    }
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

std::uint32_t MajorOf(std::uint32_t version) {
    return version >> 21;
}

// An IEEE 754 half's bits (PNanoVDB's C reader returns them raw).
double HalfToFloat(short bits) {
    const auto h = static_cast<std::uint16_t>(bits);
    const int exponent = (h >> 10) & 0x1f;
    const int mantissa = h & 0x3ff;
    double value = 0.0;
    if (exponent == 0) {
        value = std::ldexp(mantissa, -24);
    } else if (exponent == 31) {
        value = mantissa == 0
                    ? std::numeric_limits<double>::infinity()
                    : std::numeric_limits<double>::quiet_NaN();
    } else {
        value = std::ldexp(mantissa + 1024, exponent - 25);
    }
    return (h & 0x8000) != 0 ? -value : value;
}

void CheckVersion(std::uint32_t version) {
    if (MajorOf(version) != kMajorVersion) {
        throw std::runtime_error("NanoVDB major version " +
                                 std::to_string(MajorOf(version)) +
                                 " is not supported (Ernie reads " +
                                 std::to_string(kMajorVersion) + ")");
    }
}

// The header fields Ernie keeps, read through PNanoVDB's own
// accessors from a copy of the grid aligned for them.
NanoVdbGrid GridOf(std::span<const std::uint8_t> bytes) {
    NanoVdbGrid grid;
    grid.buffer.assign(bytes.begin(), bytes.end());
    grid.buffer.resize((bytes.size() + 7) / 8 * 8, 0);
    // PNanoVDB reads 64-bit words, so read from an aligned copy.
    std::vector<std::uint64_t> words(grid.buffer.size() / 8);
    std::memcpy(words.data(), grid.buffer.data(), grid.buffer.size());
    const pnanovdb_buf_t buf = pnanovdb_make_buf(
        reinterpret_cast<std::uint32_t*>(words.data()),
        words.size() * 2);
    const pnanovdb_grid_handle_t handle{pnanovdb_address_null()};
    // Each grid of a multi-grid buffer becomes its own buffer.
    pnanovdb_grid_set_grid_index(buf, handle, 0);
    pnanovdb_grid_set_grid_count(buf, handle, 1);
    std::memcpy(grid.buffer.data(), words.data(), grid.buffer.size());

    char name[257] = {};
    for (std::uint32_t i = 0; i < 64; ++i) {
        const std::uint32_t word =
            pnanovdb_grid_get_grid_name(buf, handle, i);
        std::memcpy(name + 4 * i, &word, 4);
    }
    grid.name = name;
    grid.type = static_cast<NanoVdbType>(
        pnanovdb_grid_get_grid_type(buf, handle));

    const pnanovdb_map_handle_t map =
        pnanovdb_grid_get_map(buf, handle);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            grid.indexToGrid[column][row] = pnanovdb_map_get_matd(
                buf, map,
                static_cast<std::uint32_t>(3 * row + column));
        }
        grid.indexToGrid[3][row] = pnanovdb_map_get_vecd(
            buf, map, static_cast<std::uint32_t>(row));
    }
    for (std::uint32_t i = 0; i < 3; ++i) {
        grid.voxelSize[static_cast<int>(i)] =
            pnanovdb_grid_get_voxel_size(buf, handle, i);
    }

    const pnanovdb_tree_handle_t tree =
        pnanovdb_grid_get_tree(buf, handle);
    const pnanovdb_root_handle_t root =
        pnanovdb_tree_get_root(buf, tree);
    const pnanovdb_coord_t lo = pnanovdb_root_get_bbox_min(buf, root);
    const pnanovdb_coord_t hi = pnanovdb_root_get_bbox_max(buf, root);
    grid.indexMin = {lo.x, lo.y, lo.z};
    grid.indexMax = {hi.x, hi.y, hi.z};
    grid.hasStatistics = (pnanovdb_grid_get_flags(buf, handle) &
                          PNANOVDB_GRID_FLAGS_HAS_MIN_MAX) != 0;
    if (grid.Empty() || !grid.hasStatistics) {
        return grid;
    }

    const pnanovdb_grid_type_t type =
        pnanovdb_grid_get_grid_type(buf, handle);
    const pnanovdb_address_t min =
        pnanovdb_root_get_min_address(type, buf, root);
    const pnanovdb_address_t max =
        pnanovdb_root_get_max_address(type, buf, root);
    switch (grid.type) {
    case NanoVdbType::Float:
    case NanoVdbType::Fp4:
    case NanoVdbType::Fp8:
    case NanoVdbType::Fp16:
    case NanoVdbType::FpN:
        grid.minValue = pnanovdb_read_float(buf, min);
        grid.maxValue = pnanovdb_read_float(buf, max);
        break;
    case NanoVdbType::Double:
        grid.minValue = pnanovdb_read_double(buf, min);
        grid.maxValue = pnanovdb_read_double(buf, max);
        break;
    case NanoVdbType::Half:
        grid.minValue = HalfToFloat(pnanovdb_read_half(buf, min));
        grid.maxValue = HalfToFloat(pnanovdb_read_half(buf, max));
        break;
    case NanoVdbType::Vec3f: {
        // NanoVDB orders vectors by length.
        const pnanovdb_vec3_t a = pnanovdb_read_vec3(buf, min);
        const pnanovdb_vec3_t b = pnanovdb_read_vec3(buf, max);
        grid.minValue = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        grid.maxValue = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
        break;
    }
    default:
        break;
    }
    return grid;
}

// The grids of a raw buffer: each GridData header gives its grid's
// size.
std::vector<NanoVdbGrid>
RawGrids(std::span<const std::uint8_t> bytes) {
    std::vector<NanoVdbGrid> grids;
    std::size_t offset = 0;
    while (offset + PNANOVDB_GRID_SIZE <= bytes.size()) {
        const auto magic = Load<std::uint64_t>(bytes, offset);
        if (magic != kMagicNumber && magic != kMagicGrid) {
            throw std::runtime_error(
                "NanoVDB grid signature not found");
        }
        CheckVersion(Load<std::uint32_t>(
            bytes, offset + PNANOVDB_GRID_OFF_VERSION));
        const auto size = Load<std::uint64_t>(
            bytes, offset + PNANOVDB_GRID_OFF_GRID_SIZE);
        if (size < PNANOVDB_GRID_SIZE ||
            offset + size > bytes.size()) {
            throw std::runtime_error(
                "NanoVDB grid size runs past its data");
        }
        grids.push_back(GridOf(bytes.subspan(offset, size)));
        offset += size;
    }
    if (grids.empty()) {
        throw std::runtime_error("NanoVDB data holds no grid");
    }
    return grids;
}

}  // namespace

std::string NanoVdbTypeName(NanoVdbType type) {
    switch (type) {
    case NanoVdbType::Unknown:
        return "unknown";
    case NanoVdbType::Float:
        return "float";
    case NanoVdbType::Double:
        return "double";
    case NanoVdbType::Vec3f:
        return "vec3f";
    case NanoVdbType::Half:
        return "half";
    case NanoVdbType::Fp4:
        return "fp4";
    case NanoVdbType::Fp8:
        return "fp8";
    case NanoVdbType::Fp16:
        return "fp16";
    case NanoVdbType::FpN:
        return "fpn";
    case NanoVdbType::Vec4f:
        return "vec4f";
    }
    return "type " + std::to_string(static_cast<std::uint32_t>(type));
}

bool NanoVdbGrid::Empty() const {
    return glm::any(glm::greaterThan(indexMin, indexMax));
}

bool NanoVdbGrid::Scalar() const {
    switch (type) {
    case NanoVdbType::Float:
    case NanoVdbType::Half:
    case NanoVdbType::Fp4:
    case NanoVdbType::Fp8:
    case NanoVdbType::Fp16:
    case NanoVdbType::FpN:
        return true;
    default:
        return false;
    }
}

bool NanoVdbGrid::Vector() const {
    return type == NanoVdbType::Vec3f;
}

void NanoVdbGrid::GridBounds(glm::dvec3& min, glm::dvec3& max) const {
    min = glm::dvec3(std::numeric_limits<double>::infinity());
    max = -min;
    if (Empty()) {
        min = max = glm::dvec3(indexToGrid[3]);
        return;
    }
    const glm::dvec3 lo = glm::dvec3(indexMin) - 1.0;
    const glm::dvec3 hi = glm::dvec3(indexMax) + 1.0;
    for (int corner = 0; corner < 8; ++corner) {
        const glm::dvec3 index{(corner & 1) != 0 ? hi.x : lo.x,
                               (corner & 2) != 0 ? hi.y : lo.y,
                               (corner & 4) != 0 ? hi.z : lo.z};
        const glm::dvec3 point =
            glm::dvec3(indexToGrid * glm::dvec4(index, 1.0));
        min = glm::min(min, point);
        max = glm::max(max, point);
    }
}

std::vector<NanoVdbGrid>
NanoVdbGridsOf(std::span<const std::uint8_t> bytes) {
    return RawGrids(bytes);
}

std::vector<NanoVdbGrid>
ReadNanoVdb(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open NanoVDB file " +
                                 path.string());
    }
    const std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    const std::span<const std::uint8_t> data(bytes);
    const std::string where = " in " + path.string();
    try {
        const auto magic = Load<std::uint64_t>(data, 0);
        if (magic == kMagicGrid) {
            return RawGrids(data);
        }
        if (magic != kMagicNumber && magic != kMagicFile) {
            throw std::runtime_error("not a NanoVDB file");
        }
        // A file of segments; a raw buffer written before NanoVDB
        // 32.6 also starts with NanoVDB0 but holds the grid version
        // where a segment's header holds its own.
        if (MajorOf(Load<std::uint32_t>(data, 8)) != kMajorVersion &&
            MajorOf(Load<std::uint32_t>(
                data, PNANOVDB_GRID_OFF_VERSION)) == kMajorVersion) {
            return RawGrids(data);
        }
        std::vector<NanoVdbGrid> grids;
        std::size_t offset = 0;
        while (offset < data.size()) {
            CheckVersion(Load<std::uint32_t>(data, offset + 8));
            const auto count = Load<std::uint16_t>(data, offset + 12);
            const auto codec = Load<std::uint16_t>(data, offset + 14);
            if (codec != 0) {
                throw std::runtime_error(
                    std::string("compressed (") +
                    (codec == 1 ? "zip" : "blosc") +
                    "); write it uncompressed");
            }
            offset += kFileHeaderSize;
            std::vector<std::uint64_t> sizes;
            for (std::uint16_t i = 0; i < count; ++i) {
                const auto gridSize =
                    Load<std::uint64_t>(data, offset + kMetaGridSize);
                const auto fileSize =
                    Load<std::uint64_t>(data, offset + kMetaFileSize);
                if (gridSize != fileSize) {
                    throw std::runtime_error("grid sizes disagree");
                }
                sizes.push_back(gridSize);
                offset +=
                    kMetaDataSize +
                    Load<std::uint32_t>(data, offset + kMetaNameSize);
            }
            for (const std::uint64_t size : sizes) {
                if (offset + size > data.size()) {
                    throw std::runtime_error(
                        "grid runs past the file's end");
                }
                std::vector<NanoVdbGrid> one =
                    RawGrids(data.subspan(offset, size));
                grids.insert(grids.end(),
                             std::make_move_iterator(one.begin()),
                             std::make_move_iterator(one.end()));
                offset += size;
            }
            // The next segment, if any, starts with its own magic.
            if (offset < data.size()) {
                const auto next = Load<std::uint64_t>(data, offset);
                if (next != kMagicNumber && next != kMagicFile) {
                    break;
                }
            }
        }
        return grids;
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(
            "NanoVDB: " + std::string(error.what()) + where);
    }
}

const NanoVdbGrid* FindNanoVdbGrid(std::span<const NanoVdbGrid> grids,
                                   std::string_view name, int index) {
    // An index into the caller's grids, which cppcheck follows where
    // a range over the span would look like a temporary to it.
    int seen = 0;
    for (std::size_t i = 0; i < grids.size(); ++i) {
        if (!name.empty() && grids[i].name != name) {
            continue;
        }
        if (seen++ == index) {
            return grids.data() + i;
        }
    }
    return nullptr;
}

}  // namespace ernie
