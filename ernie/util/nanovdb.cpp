// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/nanovdb.h"

#define PNANOVDB_C
#include <PNanoVDB.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

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

// The header fields Ernie keeps of the grid `bytes` starts (all of
// it, or its head), read through PNanoVDB's own accessors from a copy
// aligned for them.
NanoVdbGrid HeaderOf(std::span<const std::uint8_t> bytes) {
    NanoVdbGrid grid;
    // PNanoVDB reads 64-bit words, so read from an aligned copy.
    std::vector<std::uint64_t> words((bytes.size() + 7) / 8, 0);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    const pnanovdb_buf_t buf = pnanovdb_make_buf(
        reinterpret_cast<std::uint32_t*>(words.data()),
        words.size() * 2);
    const pnanovdb_grid_handle_t handle{pnanovdb_address_null()};
    // What PNanoVDB reads at `address`, `size` bytes, must be there.
    const auto within = [&](pnanovdb_address_t address,
                            std::size_t size) {
        return address.byte_offset + size <= bytes.size();
    };
    grid.size =
        Load<std::uint64_t>(bytes, PNANOVDB_GRID_OFF_GRID_SIZE);

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
    if (!within(root.address, PNANOVDB_ROOT_BASE_SIZE)) {
        throw std::runtime_error(
            "NanoVDB grid's root lies past the bytes read");
    }
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
    if (!within(min, 12) || !within(max, 12)) {
        grid.hasStatistics = false;
        return grid;
    }
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

// The grid `bytes` holds, with its own buffer.
NanoVdbGrid GridOf(std::span<const std::uint8_t> bytes) {
    NanoVdbGrid grid = HeaderOf(bytes);
    // Each grid of a multi-grid buffer becomes its own buffer.
    std::vector<std::uint64_t> words((bytes.size() + 7) / 8, 0);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    const pnanovdb_buf_t buf = pnanovdb_make_buf(
        reinterpret_cast<std::uint32_t*>(words.data()),
        words.size() * 2);
    const pnanovdb_grid_handle_t handle{pnanovdb_address_null()};
    pnanovdb_grid_set_grid_index(buf, handle, 0);
    pnanovdb_grid_set_grid_count(buf, handle, 1);
    grid.buffer.resize(words.size() * 8);
    std::memcpy(grid.buffer.data(), words.data(), grid.buffer.size());
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

NanoVdbGrid NanoVdbGridHead(std::span<const std::uint8_t> head) {
    const auto magic = Load<std::uint64_t>(head, 0);
    if (magic != kMagicNumber && magic != kMagicGrid) {
        throw std::runtime_error("NanoVDB grid signature not found");
    }
    CheckVersion(
        Load<std::uint32_t>(head, PNANOVDB_GRID_OFF_VERSION));
    NanoVdbGrid grid = HeaderOf(head);
    if (grid.size < PNANOVDB_GRID_SIZE) {
        throw std::runtime_error("NanoVDB grid size is too small");
    }
    return grid;
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

namespace {

// `DenseNanoVdbGrid`, or with `everyVoxel` `DenseNanoVdbTopology` (no
// `values`, every voxel of the shape active), filling `layout` when
// given.
std::vector<std::uint8_t>
BuildDense(std::span<const float> values, const glm::ivec3& shape,
           double voxelSize, const glm::dvec3& origin,
           std::string_view name, bool everyVoxel,
           DenseNanoVdbLayout* layout) {
    if (glm::any(glm::lessThan(shape, glm::ivec3(0))) ||
        (!everyVoxel &&
         values.size() != static_cast<std::size_t>(shape.x) *
                              static_cast<std::size_t>(shape.y) *
                              static_cast<std::size_t>(shape.z))) {
        throw std::invalid_argument(
            "DenseNanoVdbGrid: values do not fill the shape");
    }
    if (!(voxelSize > 0.0)) {
        throw std::invalid_argument(
            "DenseNanoVdbGrid: the voxel size must be positive");
    }
    constexpr pnanovdb_grid_type_t kType = PNANOVDB_GRID_TYPE_FLOAT;
    const auto constant = [](auto member) {
        return pnanovdb_grid_type_constants[kType].*member;
    };
    using Constants = pnanovdb_grid_type_constants_t;
    const auto at = [&](int i, int j, int k) {
        return values.empty()
                   ? 0.0f
                   : values[(static_cast<std::size_t>(i) * shape.y +
                             j) *
                                shape.z +
                            k];
    };
    // The leaves (8^3) holding active values, then the lower (128^3)
    // and upper (4096^3) nodes above them, each keyed by its origin.
    const auto originOf = [](const glm::ivec3& ijk, int size) {
        return ijk & glm::ivec3(~(size - 1));
    };
    const auto less = [](const glm::ivec3& a, const glm::ivec3& b) {
        return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
    };
    std::map<glm::ivec3, std::size_t, decltype(less)> leafIndex(less);
    std::vector<glm::ivec3> leaves;
    for (int x = 0; x < shape.x; x += 8) {
        for (int y = 0; y < shape.y; y += 8) {
            for (int z = 0; z < shape.z; z += 8) {
                bool active = everyVoxel;
                for (int i = x;
                     i < std::min(x + 8, shape.x) && !active; ++i) {
                    for (int j = y;
                         j < std::min(y + 8, shape.y) && !active;
                         ++j) {
                        for (int k = z; k < std::min(z + 8, shape.z);
                             ++k) {
                            if (at(i, j, k) != 0.0f) {
                                active = true;
                                break;
                            }
                        }
                    }
                }
                if (active) {
                    leafIndex.emplace(glm::ivec3(x, y, z),
                                      leaves.size());
                    leaves.emplace_back(x, y, z);
                }
            }
        }
    }
    std::map<glm::ivec3, std::size_t, decltype(less)> lowerIndex(
        less);
    std::vector<glm::ivec3> lowers;
    std::map<glm::ivec3, std::size_t, decltype(less)> upperIndex(
        less);
    std::vector<glm::ivec3> uppers;
    for (const glm::ivec3& leaf : leaves) {
        const glm::ivec3 lower = originOf(leaf, 128);
        if (lowerIndex.emplace(lower, lowers.size()).second) {
            lowers.push_back(lower);
        }
        const glm::ivec3 upper = originOf(leaf, 4096);
        if (upperIndex.emplace(upper, uppers.size()).second) {
            uppers.push_back(upper);
        }
    }

    // The layout: grid, tree, root and its tiles, then the nodes by
    // level, uppers first, each 32-byte aligned.
    const std::uint64_t tree = PNANOVDB_GRID_SIZE;
    const std::uint64_t root = tree + PNANOVDB_TREE_SIZE;
    const std::uint64_t firstUpper =
        root + constant(&Constants::root_size) +
        uppers.size() * constant(&Constants::root_tile_size);
    const std::uint64_t firstLower =
        firstUpper + uppers.size() * constant(&Constants::upper_size);
    const std::uint64_t firstLeaf =
        firstLower + lowers.size() * constant(&Constants::lower_size);
    const std::uint64_t size =
        firstLeaf + leaves.size() * constant(&Constants::leaf_size);
    std::vector<std::uint64_t> words(size / 8, 0);
    const pnanovdb_buf_t buf = pnanovdb_make_buf(
        reinterpret_cast<std::uint32_t*>(words.data()),
        words.size() * 2);
    const auto address = [](std::uint64_t offset) {
        return pnanovdb_address_t{static_cast<std::uint32_t>(offset)};
    };
    const auto coord = [](const glm::ivec3& c) {
        return pnanovdb_coord_t{c.x, c.y, c.z};
    };

    // Each node's active bounds and min and max, filled leaves up.
    struct Stats {
        glm::ivec3 low{std::numeric_limits<int>::max()};
        glm::ivec3 high{std::numeric_limits<int>::min()};
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        void Add(const Stats& other) {
            low = glm::min(low, other.low);
            high = glm::max(high, other.high);
            min = std::min(min, other.min);
            max = std::max(max, other.max);
        }
    };
    std::vector<Stats> lowerStats(lowers.size());
    std::vector<Stats> upperStats(uppers.size());
    Stats rootStats;
    std::uint64_t voxels = 0;
    for (std::size_t n = 0; n < leaves.size(); ++n) {
        const glm::ivec3& origin8 = leaves[n];
        const pnanovdb_leaf_handle_t leaf{
            address(firstLeaf + n * constant(&Constants::leaf_size))};
        Stats stats;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                for (int k = 0; k < 8; ++k) {
                    const glm::ivec3 ijk =
                        origin8 + glm::ivec3(i, j, k);
                    if (glm::any(glm::greaterThanEqual(ijk, shape))) {
                        continue;
                    }
                    const float value = at(ijk.x, ijk.y, ijk.z);
                    if (value == 0.0f && !everyVoxel) {
                        continue;
                    }
                    const auto bit = static_cast<std::uint32_t>(
                        (i << 6) | (j << 3) | k);
                    pnanovdb_leaf_set_value_mask(buf, leaf, bit, 1);
                    pnanovdb_write_float(
                        buf,
                        pnanovdb_leaf_get_table_address(kType, buf,
                                                        leaf, bit),
                        value);
                    stats.low = glm::min(stats.low, ijk);
                    stats.high = glm::max(stats.high, ijk);
                    stats.min = std::min(stats.min, value);
                    stats.max = std::max(stats.max, value);
                    ++voxels;
                }
            }
        }
        const pnanovdb_coord_t low = coord(stats.low);
        pnanovdb_leaf_set_bbox_min(buf, leaf, &low);
        const glm::ivec3 dif = stats.high - stats.low;
        pnanovdb_leaf_set_bbox_dif_and_flags(
            buf, leaf,
            static_cast<std::uint32_t>(dif.x | (dif.y << 8) |
                                       (dif.z << 16)));
        pnanovdb_write_float(
            buf,
            pnanovdb_address_offset(
                leaf.address, constant(&Constants::leaf_off_min)),
            stats.min);
        pnanovdb_write_float(
            buf,
            pnanovdb_address_offset(
                leaf.address, constant(&Constants::leaf_off_max)),
            stats.max);
        // Its lower node's child.
        const std::size_t l = lowerIndex.at(originOf(origin8, 128));
        const pnanovdb_lower_handle_t lower{address(
            firstLower + l * constant(&Constants::lower_size))};
        const pnanovdb_coord_t at8 = coord(origin8);
        const std::uint32_t slot =
            pnanovdb_lower_coord_to_offset(&at8);
        pnanovdb_lower_set_child_mask(buf, lower, slot, 1);
        pnanovdb_lower_set_child(kType, buf, lower, slot, leaf);
        lowerStats[l].Add(stats);
    }
    for (std::size_t l = 0; l < lowers.size(); ++l) {
        const pnanovdb_lower_handle_t lower{address(
            firstLower + l * constant(&Constants::lower_size))};
        const Stats& stats = lowerStats[l];
        const pnanovdb_coord_t low = coord(stats.low);
        const pnanovdb_coord_t high = coord(stats.high);
        pnanovdb_lower_set_bbox_min(buf, lower, &low);
        pnanovdb_lower_set_bbox_max(buf, lower, &high);
        pnanovdb_write_float(
            buf,
            pnanovdb_address_offset(
                lower.address, constant(&Constants::lower_off_min)),
            stats.min);
        pnanovdb_write_float(
            buf,
            pnanovdb_address_offset(
                lower.address, constant(&Constants::lower_off_max)),
            stats.max);
        const std::size_t u =
            upperIndex.at(originOf(lowers[l], 4096));
        const pnanovdb_upper_handle_t upper{address(
            firstUpper + u * constant(&Constants::upper_size))};
        const pnanovdb_coord_t at128 = coord(lowers[l]);
        const std::uint32_t slot =
            pnanovdb_upper_coord_to_offset(&at128);
        pnanovdb_upper_set_child_mask(buf, upper, slot, 1);
        pnanovdb_upper_set_child(kType, buf, upper, slot, lower);
        upperStats[u].Add(stats);
    }
    const pnanovdb_root_handle_t rootHandle{address(root)};
    for (std::size_t u = 0; u < uppers.size(); ++u) {
        const pnanovdb_upper_handle_t upper{address(
            firstUpper + u * constant(&Constants::upper_size))};
        const Stats& stats = upperStats[u];
        const pnanovdb_coord_t low = coord(stats.low);
        const pnanovdb_coord_t high = coord(stats.high);
        pnanovdb_upper_set_bbox_min(buf, upper, &low);
        pnanovdb_upper_set_bbox_max(buf, upper, &high);
        pnanovdb_write_float(
            buf,
            pnanovdb_address_offset(
                upper.address, constant(&Constants::upper_off_min)),
            stats.min);
        pnanovdb_write_float(
            buf,
            pnanovdb_address_offset(
                upper.address, constant(&Constants::upper_off_max)),
            stats.max);
        const pnanovdb_root_tile_handle_t tile{
            address(root + constant(&Constants::root_size) +
                    u * constant(&Constants::root_tile_size))};
        const pnanovdb_coord_t at4096 = coord(uppers[u]);
        pnanovdb_root_tile_set_key(buf, tile,
                                   pnanovdb_coord_to_key(&at4096));
        pnanovdb_root_set_child(kType, buf, rootHandle, tile, upper);
        rootStats.Add(stats);
    }

    // The root, tree and grid headers.
    if (!leaves.empty()) {
        const pnanovdb_coord_t low = coord(rootStats.low);
        const pnanovdb_coord_t high = coord(rootStats.high);
        pnanovdb_root_set_bbox_min(buf, rootHandle, &low);
        pnanovdb_root_set_bbox_max(buf, rootHandle, &high);
        pnanovdb_write_float(buf,
                             pnanovdb_address_offset(
                                 rootHandle.address,
                                 constant(&Constants::root_off_min)),
                             rootStats.min);
        pnanovdb_write_float(buf,
                             pnanovdb_address_offset(
                                 rootHandle.address,
                                 constant(&Constants::root_off_max)),
                             rootStats.max);
    } else {
        // No active values: an empty box, min past max.
        const pnanovdb_coord_t low{1, 1, 1};
        const pnanovdb_coord_t high{0, 0, 0};
        pnanovdb_root_set_bbox_min(buf, rootHandle, &low);
        pnanovdb_root_set_bbox_max(buf, rootHandle, &high);
    }
    pnanovdb_root_set_tile_count(
        buf, rootHandle, static_cast<std::uint32_t>(uppers.size()));

    const pnanovdb_tree_handle_t treeHandle{address(tree)};
    pnanovdb_tree_set_node_offset_leaf(
        buf, treeHandle, leaves.empty() ? 0 : firstLeaf - tree);
    pnanovdb_tree_set_node_offset_lower(
        buf, treeHandle, lowers.empty() ? 0 : firstLower - tree);
    pnanovdb_tree_set_node_offset_upper(
        buf, treeHandle, uppers.empty() ? 0 : firstUpper - tree);
    pnanovdb_tree_set_node_offset_root(buf, treeHandle, root - tree);
    pnanovdb_tree_set_node_count_leaf(
        buf, treeHandle, static_cast<std::uint32_t>(leaves.size()));
    pnanovdb_tree_set_node_count_lower(
        buf, treeHandle, static_cast<std::uint32_t>(lowers.size()));
    pnanovdb_tree_set_node_count_upper(
        buf, treeHandle, static_cast<std::uint32_t>(uppers.size()));
    pnanovdb_tree_set_voxel_count(buf, treeHandle, voxels);

    const pnanovdb_grid_handle_t grid{address(0)};
    pnanovdb_grid_set_magic(buf, grid, PNANOVDB_MAGIC_GRID);
    pnanovdb_grid_set_version(
        buf, grid,
        pnanovdb_make_version(PNANOVDB_MAJOR_VERSION_NUMBER,
                              PNANOVDB_MINOR_VERSION_NUMBER,
                              PNANOVDB_PATCH_VERSION_NUMBER));
    // A topology's values come later: no min and max for them.
    pnanovdb_grid_set_flags(
        buf, grid,
        PNANOVDB_GRID_FLAGS_HAS_BBOX |
            PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST |
            (values.empty() ? 0u : PNANOVDB_GRID_FLAGS_HAS_MIN_MAX));
    pnanovdb_grid_set_grid_index(buf, grid, 0);
    pnanovdb_grid_set_grid_count(buf, grid, 1);
    pnanovdb_grid_set_grid_size(buf, grid, size);
    char gridName[256] = {};
    std::memcpy(gridName, name.data(),
                std::min<std::size_t>(name.size(), 255));
    for (std::uint32_t i = 0; i < 64; ++i) {
        std::uint32_t word = 0;
        std::memcpy(&word, gridName + 4 * i, 4);
        pnanovdb_grid_set_grid_name(buf, grid, i, word);
    }
    const pnanovdb_map_handle_t map{address(PNANOVDB_GRID_OFF_MAP)};
    for (std::uint32_t row = 0; row < 3; ++row) {
        for (std::uint32_t column = 0; column < 3; ++column) {
            const double scale = row == column ? voxelSize : 0.0;
            const double inverse =
                row == column ? 1.0 / voxelSize : 0.0;
            pnanovdb_map_set_matf(buf, map, 3 * row + column,
                                  static_cast<float>(scale));
            pnanovdb_map_set_invmatf(buf, map, 3 * row + column,
                                     static_cast<float>(inverse));
            pnanovdb_map_set_matd(buf, map, 3 * row + column, scale);
            pnanovdb_map_set_invmatd(buf, map, 3 * row + column,
                                     inverse);
        }
        pnanovdb_map_set_vecf(buf, map, row,
                              static_cast<float>(origin[row]));
        pnanovdb_map_set_vecd(buf, map, row, origin[row]);
    }
    pnanovdb_map_set_taperf(buf, map, 0, 1.0f);
    pnanovdb_map_set_taperd(buf, map, 0, 1.0);
    if (!leaves.empty()) {
        for (int a = 0; a < 3; ++a) {
            pnanovdb_grid_set_world_bbox(
                buf, grid, static_cast<std::uint32_t>(a),
                origin[a] + voxelSize * rootStats.low[a]);
            pnanovdb_grid_set_world_bbox(
                buf, grid, static_cast<std::uint32_t>(a + 3),
                origin[a] + voxelSize * (rootStats.high[a] + 1));
        }
    }
    for (std::uint32_t a = 0; a < 3; ++a) {
        pnanovdb_grid_set_voxel_size(buf, grid, a, voxelSize);
    }
    pnanovdb_grid_set_grid_class(buf, grid,
                                 PNANOVDB_GRID_CLASS_FOG_VOLUME);
    pnanovdb_grid_set_grid_type(buf, grid, kType);

    if (layout != nullptr) {
        layout->firstLeaf = firstLeaf;
        layout->leafBytes = constant(&Constants::leaf_size);
        layout->valuesOffset = constant(&Constants::leaf_off_table);
        layout->leaves = (shape + 7) / 8;
    }
    std::vector<std::uint8_t> bytes(size);
    std::memcpy(bytes.data(), words.data(), size);
    return bytes;
}

}  // namespace

std::vector<std::uint8_t>
DenseNanoVdbGrid(std::span<const float> values,
                 const glm::ivec3& shape, double voxelSize,
                 const glm::dvec3& origin, std::string_view name) {
    return BuildDense(values, shape, voxelSize, origin, name, false,
                      nullptr);
}

std::vector<std::uint8_t>
DenseNanoVdbTopology(const glm::ivec3& shape, double voxelSize,
                     const glm::dvec3& origin, std::string_view name,
                     DenseNanoVdbLayout& layout) {
    return BuildDense({}, shape, voxelSize, origin, name, true,
                      &layout);
}

}  // namespace ernie
