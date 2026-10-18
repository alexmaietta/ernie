// SPDX-License-Identifier: BSD-3-Clause
/// @file geometry_test.cpp
/// @brief Tests of the geometry pool: meshes uploaded as triangles, implicit shapes sharing unit
/// geometries with their sizes in shape transforms, and edits costing what they change.
#include "ernie/render/geometry.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

bool Has(const std::vector<GeometryId>& ids, GeometryId id) {
    return std::ranges::find(ids, id) != ids.end();
}

/// A quad of four points and one face.
void WriteQuad(Scene& scene, PrimId mesh, float z) {
    const std::vector<float> points{0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z};
    scene.Write(mesh, "points", Tensor::Copy(std::span<const float>(points), {4, 3}));
    scene.Write(mesh, "faceVertexCounts",
                Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{4}), {1}));
    scene.Write(
        mesh, "faceVertexIndices",
        Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{0, 1, 2, 3}), {4}));
}

class Geometries : public ::testing::Test {
  protected:
    void SetUp() override {
        context_ = test::MakeContext();
        if (!context_) {
            GTEST_SKIP() << "No Vulkan device";
        }
        arenas_ = std::make_unique<Arenas>(*context_, MemoryUse::Device, "geometry", 1 << 16);
        pool_ = std::make_unique<graph::TransientPool>(*context_);
        mesh_ = scene_.Define("/Quad", "Mesh");
        WriteQuad(scene_, mesh_, 0.0f);
        ball_ = scene_.Define("/Ball", "Sphere");
        scene_.Set(ball_, "radius", 2.0);
        pebble_ = scene_.Define("/Pebble", "Sphere");
        pill_ = scene_.Define("/Pill", "Capsule");
        scene_.Set(pill_, "radius", 0.5);
        scene_.Set(pill_, "height", 2.0);
        wall_ = scene_.Define("/Wall", "Plane");
        scene_.Set(wall_, "axis", std::string("X"));
        scene_.Set(wall_, "width", 4.0);
        scene_.Set(wall_, "length", 6.0);
        geometry_ = std::make_unique<GeometryPool>(*context_, *arenas_, scene_);
    }
    void TearDown() override {
        if (context_) {
            context_->WaitIdle();
            ExpectClean(*context_);
        }
    }

    /// @return The update's changes, with its uploads run.
    GeometryPool::Changes Update() {
        GeometryPool::Changes changes = geometry_->Update();
        graph::Graph graph(*context_, *pool_);
        geometry_->AddUploads(graph);
        graph.Execute(context_->Main()).Wait();
        return changes;
    }

    std::vector<std::uint32_t> ReadWords(const BufferRange& range) {
        Readback readback(*context_, context_->Main(), arenas_->Storage(range), range.offset,
                          range.size);
        std::vector<std::uint32_t> words(range.size / 4);
        std::memcpy(words.data(), readback.Bytes().data(), range.size);
        return words;
    }

    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<graph::TransientPool> pool_;
    Scene scene_;
    PrimId mesh_, ball_, pebble_, pill_, wall_;
    std::unique_ptr<GeometryPool> geometry_;
};

TEST_F(Geometries, MeshesBecomeTrianglesAndShapesShareUnitGeometry) {
    const GeometryPool::Changes changes = Update();
    EXPECT_EQ(changes.placed.size(), 5u) << "each prim once";
    EXPECT_TRUE(changes.refitted.empty()) << "a new mesh's own values are no edit";
    const PrimGeometry quad = *geometry_->Find(mesh_);
    const Geometry& triangles = geometry_->Get(quad.geometry);
    EXPECT_EQ(triangles.kind, GeometryKind::Triangles);
    EXPECT_EQ(triangles.vertexCount, 4u);
    EXPECT_EQ(triangles.triangleCount, 2u) << "the quad as a fan";
    EXPECT_EQ(ReadWords(triangles.indices), (std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3}));

    // Kernels find a geometry's kind, counts and data by its id.
    const DeviceColumn& records = geometry_->Records();
    const std::vector<std::uint32_t> words = ReadWords(
        {records.Range().arena, records.Range().offset + quad.geometry * sizeof(GeometryRecord),
         sizeof(GeometryRecord)});
    GeometryRecord record;
    std::memcpy(static_cast<void*>(&record), words.data(), sizeof(record));
    EXPECT_EQ(record.kind, static_cast<std::uint32_t>(GeometryKind::Triangles));
    EXPECT_EQ(record.triangleCount, 2u);
    EXPECT_EQ(record.indices, arenas_->Address(triangles.indices));

    const PrimGeometry ball = *geometry_->Find(ball_);
    EXPECT_EQ(geometry_->Find(pebble_)->geometry, ball.geometry) << "one unit sphere for both";
    EXPECT_EQ(geometry_->Get(ball.geometry).kind, GeometryKind::Sphere);
    EXPECT_EQ(ball.shape[0][0], 2.0) << "its radius in its shape transform";
    EXPECT_EQ(geometry_->Find(pebble_)->shape[0][0], 1.0) << "the fallback radius";

    const Geometry& capsule = geometry_->Get(geometry_->Find(pill_)->geometry);
    EXPECT_EQ(capsule.kind, GeometryKind::Capsule);
    EXPECT_FLOAT_EQ(capsule.halfLength, 2.0f) << "half its height over its radius";

    // An X-facing plane: its normal along x, its width along z and its length along y.
    const glm::dmat4 wall = geometry_->Find(wall_)->shape;
    EXPECT_EQ(glm::dvec3(wall * glm::dvec4(0, 0, 1, 0)), glm::dvec3(1, 0, 0));
    EXPECT_EQ(glm::dvec3(wall * glm::dvec4(0, 1, 0, 0)), glm::dvec3(0, 0, 2));
    EXPECT_EQ(glm::dvec3(wall * glm::dvec4(1, 0, 0, 0)), glm::dvec3(0, 3, 0));
    EXPECT_FALSE(geometry_->Find(scene_.Define("/Group", "Xform"))) << "no geometry";
}

TEST_F(Geometries, EditsCostWhatTheyChange) {
    Update();
    const GeometryId quad = geometry_->Find(mesh_)->geometry;
    const GeometryId capsule = geometry_->Find(pill_)->geometry;

    // Points moved in place: a refit.
    const std::vector<float> lifted{0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    scene_.Write(mesh_, "points", Tensor::Copy(std::span<const float>(lifted), {4, 3}));
    GeometryPool::Changes changes = Update();
    EXPECT_TRUE(Has(changes.refitted, quad));
    EXPECT_TRUE(changes.rebuilt.empty());

    // A new face: a rebuild.
    scene_.Write(mesh_, "faceVertexCounts",
                 Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{3, 3}), {2}));
    scene_.Write(
        mesh_, "faceVertexIndices",
        Tensor::Copy(std::span<const std::int32_t>(std::vector<std::int32_t>{0, 1, 2, 0, 2, 3}),
                     {6}));
    changes = Update();
    EXPECT_TRUE(Has(changes.rebuilt, quad));

    // A larger sphere: a new shape transform, no geometry work.
    scene_.Set(ball_, "radius", 5.0);
    changes = Update();
    ASSERT_EQ(changes.placed.size(), 1u);
    EXPECT_EQ(changes.placed[0].shape[0][0], 5.0);
    EXPECT_TRUE(changes.rebuilt.empty() && changes.refitted.empty());

    // A mesh recoloured: a new colour, no geometry work.
    scene_.Set(mesh_, "primvars:displayColor", glm::dvec3(1, 0, 0));
    changes = Update();
    ASSERT_EQ(changes.placed.size(), 1u);
    EXPECT_EQ(changes.placed[0].colour, glm::vec3(1, 0, 0));
    EXPECT_TRUE(changes.rebuilt.empty() && changes.refitted.empty());

    // A capsule taken out frees its own geometry; the shared sphere stays.
    scene_.Remove("/Pill");
    changes = Update();
    EXPECT_EQ(changes.removed.size(), 1u);
    EXPECT_TRUE(Has(changes.freed, capsule));
}

}  // namespace
}  // namespace ernie
