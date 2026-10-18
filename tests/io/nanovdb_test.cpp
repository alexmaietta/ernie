// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/io/nanovdb.h"
#include "ernie/util/file.h"

#include "tests/common.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

// fixtures/volumes/make_nvdb.py: 16^3 ones in 0.1 m voxels from (-0.75, -0.75, -0.75) m, as
// Warp's Volume.save_to_nvdb writes them.
std::filesystem::path Cube() {
    return test::FixturePath("volumes/cube.nvdb");
}

std::vector<char> Bytes(const std::filesystem::path& path) {
    const std::string text = ReadTextFile(path);
    return {text.begin(), text.end()};
}

std::filesystem::path Write(const std::string& name, const std::vector<char>& bytes) {
    return test::WriteFile(test::TempPath(name), {bytes.data(), bytes.size()});
}

TEST(NanoVdb, ReadsWarpsFile) {
    const std::vector<NanoVdbGrid> grids = ReadNanoVdb(Cube());
    ASSERT_EQ(grids.size(), 1u);
    const NanoVdbGrid& grid = grids[0];
    EXPECT_EQ(grid.type, NanoVdbType::Float);
    EXPECT_TRUE(grid.Scalar());
    EXPECT_FALSE(grid.Vector());
    EXPECT_FALSE(grid.Empty());
    EXPECT_EQ(grid.buffer.size() % 8, 0u);

    // Index (i, j, k) sits at -0.75 + 0.1 (i, j, k) m, to the float
    // precision Warp writes its map in.
    constexpr double kFloat = 1e-7;
    const glm::dvec3 origin(grid.indexToGrid * glm::dvec4(0, 0, 0, 1));
    const glm::dvec3 step(grid.indexToGrid * glm::dvec4(2, 4, 6, 1));
    EXPECT_NEAR(origin.x, -0.75, kFloat);
    EXPECT_NEAR(origin.z, -0.75, kFloat);
    EXPECT_NEAR(step.x, -0.55, kFloat);
    EXPECT_NEAR(step.y, -0.35, kFloat);
    EXPECT_NEAR(step.z, -0.15, kFloat);
    EXPECT_NEAR(grid.voxelSize.y, 0.1, kFloat);

    EXPECT_EQ(grid.indexMin, glm::ivec3(0));
    EXPECT_EQ(grid.indexMax, glm::ivec3(15));
    // Warp writes no statistics.
    EXPECT_FALSE(grid.hasStatistics);
    EXPECT_EQ(grid.maxValue, 0.0);

    // Grown by a voxel each side, what trilinear sampling reaches.
    const Box bounds = grid.GridBounds();
    EXPECT_NEAR(bounds.min.x, -0.85, kFloat);
    EXPECT_NEAR(bounds.max.y, 0.85, kFloat);
}

TEST(NanoVdb, RawBufferMatchesTheFile) {
    const NanoVdbGrid file = ReadNanoVdb(Cube())[0];
    const std::vector<NanoVdbGrid> raw = ParseNanoVdbGrids(file.buffer);
    ASSERT_EQ(raw.size(), 1u);
    EXPECT_EQ(raw[0].type, file.type);
    EXPECT_EQ(raw[0].indexMin, file.indexMin);
    EXPECT_EQ(raw[0].indexMax, file.indexMax);
    EXPECT_EQ(raw[0].maxValue, file.maxValue);
    EXPECT_EQ(raw[0].buffer, file.buffer);

    // A file holding just the raw grid reads the same.
    const std::vector<char> bytes(file.buffer.begin(), file.buffer.end());
    const std::vector<NanoVdbGrid> read = ReadNanoVdb(Write("ball_raw.nvdb", bytes));
    ASSERT_EQ(read.size(), 1u);
    EXPECT_EQ(read[0].buffer, file.buffer);
}

TEST(NanoVdb, HeadsOfAFileMatchItsGridsWithoutTheirVoxels) {
    const NanoVdbGrid full = ReadNanoVdb(Cube())[0];
    const std::vector<NanoVdbGrid> heads = ReadNanoVdbHeads(Cube());
    ASSERT_EQ(heads.size(), 1u);
    EXPECT_TRUE(heads[0].buffer.empty());
    EXPECT_EQ(heads[0].name, full.name);
    EXPECT_EQ(heads[0].GridBounds(), full.GridBounds());
}

TEST(NanoVdb, AGridsHeadHoldsItsHeader) {
    const NanoVdbGrid file = ReadNanoVdb(Cube())[0];
    const std::span<const std::uint8_t> bytes(file.buffer);
    const NanoVdbGrid head =
        NanoVdbGridHead(bytes.first(std::min(bytes.size(), kNanoVdbHeadBytes)));
    EXPECT_TRUE(head.buffer.empty());
    EXPECT_EQ(head.size, file.size);
    EXPECT_LE(head.size, file.buffer.size());
    EXPECT_EQ(head.name, file.name);
    EXPECT_EQ(head.type, file.type);
    EXPECT_EQ(head.indexToGrid, file.indexToGrid);
    EXPECT_EQ(head.indexMin, file.indexMin);
    EXPECT_EQ(head.indexMax, file.indexMax);
    EXPECT_EQ(head.hasStatistics, file.hasStatistics);
    EXPECT_EQ(head.maxValue, file.maxValue);
    // The GridData header alone, 672 bytes.
    EXPECT_THROW(NanoVdbGridHead(bytes.first(672)), std::runtime_error)
        << "the root lies past a bare header";
}

TEST(NanoVdb, BuildsAGridFromDenseValues) {
    // 20 x 10 x 140 voxels (two lower nodes along z), a ramp in a box
    // [2, 17] x [1, 8] x [3, 130], 0 elsewhere.
    const glm::ivec3 shape(20, 10, 140);
    std::vector<float> values(20 * 10 * 140, 0.0f);
    for (std::size_t i = 2; i <= 17; ++i) {
        for (std::size_t j = 1; j <= 8; ++j) {
            for (std::size_t k = 3; k <= 130; ++k) {
                values[(i * 10 + j) * 140 + k] = 0.5f + 0.01f * static_cast<float>(k);
            }
        }
    }
    const std::vector<std::uint8_t> bytes =
        DenseNanoVdbGrid(values, shape, 0.25, glm::dvec3(1.0, 2.0, 3.0), "density");
    const std::vector<NanoVdbGrid> grids = ParseNanoVdbGrids(bytes);
    ASSERT_EQ(grids.size(), 1u);
    const NanoVdbGrid& grid = grids[0];
    EXPECT_EQ(grid.name, "density");
    EXPECT_EQ(grid.type, NanoVdbType::Float);
    EXPECT_EQ(grid.size, bytes.size());
    EXPECT_EQ(grid.indexMin, glm::ivec3(2, 1, 3));
    EXPECT_EQ(grid.indexMax, glm::ivec3(17, 8, 130));
    EXPECT_TRUE(grid.hasStatistics);
    EXPECT_FLOAT_EQ(static_cast<float>(grid.minValue), 0.53f);
    EXPECT_FLOAT_EQ(static_cast<float>(grid.maxValue), 1.8f);
    EXPECT_EQ(grid.voxelSize, glm::dvec3(0.25));
    EXPECT_EQ(grid.indexToGrid * glm::dvec4(4.0, 0.0, 0.0, 1.0), glm::dvec4(2.0, 2.0, 3.0, 1.0));

    const std::vector<std::uint8_t> empty =
        DenseNanoVdbGrid(std::vector<float>(8, 0.0f), glm::ivec3(2), 1.0, glm::dvec3(0.0));
    EXPECT_TRUE(ParseNanoVdbGrids(empty).front().Empty());
    EXPECT_THROW(DenseNanoVdbGrid(values, glm::ivec3(2), 1.0, glm::dvec3(0.0)),
                 std::invalid_argument);
}

TEST(NanoVdb, ADenseTopologyHoldsEveryVoxelOfItsShape) {
    // Every voxel of a 20 x 10 x 12 box active (and 0), in 3 x 2 x 2
    // leaves after the grid's nodes; the GPU tests write and sample
    // its values.
    const glm::ivec3 shape(20, 10, 12);
    DenseNanoVdbLayout layout;
    const std::vector<std::uint8_t> bytes =
        DenseNanoVdbTopology(shape, 0.5, glm::dvec3(0.0), "density", layout);
    EXPECT_EQ(layout.leaves, glm::ivec3(3, 2, 2));
    EXPECT_EQ(bytes.size(), layout.firstLeaf + 12 * std::uint64_t{layout.leafBytes});
    const NanoVdbGrid topology = ParseNanoVdbGrids(bytes).front();
    EXPECT_EQ(topology.indexMin, glm::ivec3(0));
    EXPECT_EQ(topology.indexMax, shape - 1);
    EXPECT_FALSE(topology.hasStatistics);
}

TEST(NanoVdb, FindsGridsByNameAndIndex) {
    std::vector<NanoVdbGrid> grids(3);
    grids[0].name = "density";
    grids[1].name = "temperature";
    grids[2].name = "density";
    EXPECT_EQ(FindNanoVdbGrid(grids, "density"), &grids[0]);
    EXPECT_EQ(FindNanoVdbGrid(grids, "density", 1), &grids[2]);
    EXPECT_EQ(FindNanoVdbGrid(grids, "temperature"), &grids[1]);
    EXPECT_EQ(FindNanoVdbGrid(grids, "", 2), &grids[2]);
    EXPECT_EQ(FindNanoVdbGrid(grids, "flame"), nullptr);
    EXPECT_EQ(FindNanoVdbGrid(grids, "density", 2), nullptr);
}

TEST(NanoVdb, RejectsOtherAndCompressedFiles) {
    EXPECT_THROW(ReadNanoVdb(Write("not.nvdb", std::vector<char>(64, 'x'))), std::runtime_error);
    EXPECT_THROW(ReadNanoVdb(test::TempPath("missing.nvdb")), std::runtime_error);

    // The segment header's codec (bytes 14-15): 1 is zip.
    std::vector<char> bytes = Bytes(Cube());
    bytes[14] = 1;
    try {
        ReadNanoVdb(Write("zip.nvdb", bytes));
        FAIL() << "a compressed file read";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("zip"), std::string::npos);
    }
}

}  // namespace
}  // namespace ernie
