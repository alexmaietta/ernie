// SPDX-License-Identifier: BSD-3-Clause
/// @file vol_test.cpp
/// @brief Tests of volumes' and fields' extents, so any volume can be framed.
#include "ernie/io/nanovdb.h"
#include "ernie/scene/schemas/vol.h"
#include "ernie/scene/xform.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <vector>

namespace ernie {
namespace {

using test::BoxNear;

constexpr double kFloat = 1e-6;

/// The fixture's 16^3 cube of 0.1 m voxels from -0.75 m, grown by a voxel: +-0.85 m.
OpenVDBAsset CubeField(Scene& scene, const Path& path) {
    return OpenVDBAsset::Define(scene, path)
        .SetFilePath(test::FixturePath("volumes/cube.nvdb").string());
}

TEST(Volume, AFileFieldIsBoundedByItsGridHeader) {
    Scene scene;
    const OpenVDBAsset field = CubeField(scene, "/Smoke/density");
    EXPECT_TRUE(BoxNear(*scene.Extent(field.Id()), glm::dvec3(-0.85), glm::dvec3(0.85), kFloat));
    const OpenVDBAsset missing =
        OpenVDBAsset::Define(scene, "/Missing").SetFilePath("/nowhere.nvdb");
    EXPECT_TRUE(scene.Extent(missing.Id())->Empty()) << "bounds nothing, throws nothing";
}

TEST(Volume, AVolumeIsBoundedByItsFieldsInItsOwnSpace) {
    Scene scene;
    Volume smoke = Volume::Define(scene, "/World/Smoke");
    SetXform(scene, "/World/Smoke", {.translate = glm::dvec3(10.0, 0.0, 0.0)});
    CubeField(scene, "/World/Smoke/density");
    SetXform(scene, "/World/Smoke/density", {.translate = glm::dvec3(0.0, 0.0, 2.0)});
    smoke.SetField("density", "/World/Smoke/density");
    EXPECT_TRUE(
        BoxNear(*scene.Extent(smoke.Id()), {-0.85, -0.85, 1.15}, {0.85, 0.85, 2.85}, kFloat));
    EXPECT_TRUE(BoxNear(scene.WorldBounds("/World/Smoke"), {9.15, -0.85, 1.15}, {10.85, 0.85, 2.85},
                        kFloat))
        << "framing the volume frames its field";
}

TEST(Volume, FieldsElsewhereComeThroughTheTransformsBetween) {
    Scene scene;
    Volume smoke = Volume::Define(scene, "/World/Smoke");
    SetXform(scene, "/World/Smoke", {.scale = glm::dvec3(2.0)});
    CubeField(scene, "/Library/density");
    SetXform(scene, "/Library/density", {.translate = glm::dvec3(4.0, 0.0, 0.0)});
    smoke.SetField("density", "/Library/density");
    EXPECT_TRUE(
        BoxNear(*scene.Extent(smoke.Id()), {1.575, -0.425, -0.425}, {2.425, 0.425, 0.425}, kFloat));

    SetXform(scene, "/Library/density", {.translate = glm::dvec3(0.0)});
    scene.Update();
    EXPECT_TRUE(BoxNear(*scene.Extent(smoke.Id()), glm::dvec3(-0.425), glm::dvec3(0.425), kFloat))
        << "a volume follows its fields' edits";
}

TEST(Volume, InMemoryAndDenseFieldsAreBoundedFromTheirArrays) {
    Scene scene;
    const std::vector<std::uint8_t> grid =
        DenseNanoVdbGrid(std::vector<float>(8 * 8 * 8, 1.0f), glm::ivec3(8), 0.5, glm::dvec3(1.0));
    const NanoVDBField nano =
        NanoVDBField::Define(scene, "/Nano")
            .SetBuffer(Tensor::Copy<std::uint8_t>(grid, {static_cast<std::int64_t>(grid.size())}));
    EXPECT_TRUE(BoxNear(*scene.Extent(nano.Id()), glm::dvec3(0.5), glm::dvec3(5.0), kFloat))
        << "indices 0 to 7 at 0.5 m from 1 m, grown by a voxel";

    const DenseField dense = DenseField::Define(scene, "/Dense")
                                 .SetValues(Tensor::Zeros(DType::Float32, {4, 5, 6}))
                                 .SetVoxelSize(0.5);
    EXPECT_TRUE(BoxNear(*scene.Extent(dense.Id()), glm::dvec3(-0.5), {2.0, 2.5, 3.0}));
}

}  // namespace
}  // namespace ernie
