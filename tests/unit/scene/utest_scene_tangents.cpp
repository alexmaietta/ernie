// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/builders/tangents.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/bvh/builder.h"
#include "ernie/device/device.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

TangentMesh Quad(std::vector<Float2> st) {
    TangentMesh mesh;
    mesh.points = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    mesh.indices = {0, 1, 2, 3};
    mesh.counts = {4};
    mesh.normals = std::vector<Float3>(4, Float3{0, 0, 1});
    mesh.st = std::move(st);
    return mesh;
}

void ExpectTangents(const std::vector<Float4>& tangents, Float4 expected) {
    ASSERT_FALSE(tangents.empty());
    for (const Float4& t : tangents) {
        EXPECT_NEAR(t.x, expected.x, 1e-5f);
        EXPECT_NEAR(t.y, expected.y, 1e-5f);
        EXPECT_NEAR(t.z, expected.z, 1e-5f);
        EXPECT_EQ(t.w, expected.w);
    }
}

TEST(TangentTest, MikkTSpaceFollowsTheTextureAxes) {
    ExpectTangents(MikkTangents(Quad({{0, 0}, {1, 0}, {1, 1}, {0, 1}})), {1, 0, 0, 1});
    ExpectTangents(MikkTangents(Quad({{0, 0}, {0, -1}, {1, -1}, {1, 0}})), {0, 1, 0, 1});
    ExpectTangents(MikkTangents(Quad({{1, 0}, {0, 0}, {0, 1}, {1, 1}})), {-1, 0, 0, -1});

    TangentMesh pentagon;
    for (uint32_t i = 0; i < 5; ++i) {
        const float angle = 2 * 3.14159265f * float(i) / 5;
        pentagon.points.push_back({std::cos(angle), std::sin(angle), 0});
        pentagon.st.push_back({std::cos(angle), std::sin(angle)});
        pentagon.indices.push_back(int32_t(i));
    }
    pentagon.counts = {5};
    pentagon.normals = std::vector<Float3>(5, Float3{0, 0, 1});
    const std::vector<Float4> fanned = MikkTangents(pentagon);
    ASSERT_EQ(fanned.size(), 5u);
    for (const Float4& t : fanned) {
        EXPECT_NEAR(std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z), 1.0f, 1e-4f);
    }
    EXPECT_TRUE(MikkTangents(Quad({{0, 0}, {1, 0}})).empty());
}

const std::vector<Float2> kSt = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

Mesh::Desc TexturedQuad() {
    return {.primvars = {Primvar::Create("st", Interpolation::FaceVarying, kSt)},
            .points = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}},
            .faceVertexIndices = {0, 1, 2, 3},
            .faceVertexCounts = {4},
            .subdivisionScheme = SubdivisionScheme::None};
}

struct Scene {
    Store& store;
    Catalog& catalog;
    Timeline& timeline;
    Tangents& tangents;
    Ordinal ordinal = 0;

    void Publish() {
        timeline.Publish(++ordinal).Wait();
    }

    ArrayRef Cell(uint32_t prim) {
        return tangents.Record(catalog.GetPrims().Row(prim, SchemaId::Mesh)).tangents;
    }

    std::vector<Float4> Read(uint32_t prim) {
        return Items<Float4>(store.Arrays(ColumnType::IntArray).Read(Cell(prim)));
    }

    void GivePoints(uint32_t prim, const ArrayRef& array, const std::vector<Float3>& points) {
        store.Arrays(ColumnType::Float3Array).Write(array, std::as_bytes(std::span(points)));
        store.SetArray(kPointBasedPoints, catalog.GetPrims().Row(prim, SchemaId::PointBased),
                       array);
    }

    ArrayRef Reuse(const ArrayRef& freed) {
        ArrayHeap& float3s = store.Arrays(ColumnType::Float3Array);
        std::vector<ArrayRef> spare;
        ArrayRef reused = float3s.Allocate(freed.count);
        while (reused.address != freed.address && spare.size() < 64) {
            spare.push_back(reused);
            reused = float3s.Allocate(freed.count);
        }
        for (const ArrayRef& array : spare) {
            float3s.Free(array);
        }
        return reused;
    }

    uint32_t Follows(uint32_t prim) {
        return tangents.Record(catalog.GetPrims().Row(prim, SchemaId::Mesh)).primvar;
    }
};

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        Store store(device);
        Catalog catalog(store);
        Timeline timeline(catalog);
        const std::unique_ptr<bvh::Builder> builder = bvh::Builder::Create(device);
        Geometry geometry(catalog, *builder);
        Tangents tangents(catalog, geometry);
        timeline.Add(geometry);
        timeline.Add(tangents);
        Scene scene{store, catalog, timeline, tangents};
        test(scene);
    });
}

TEST(TangentTest, MeshesWithStGetTangentsPerFaceVertex) {
    ForEachScene([](Scene& s) {
        Mesh::Desc plain = TexturedQuad();
        plain.primvars.clear();
        const uint32_t with =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/With"}, TexturedQuad()).firstPrim;
        const uint32_t without =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/Without"}, plain).firstPrim;
        s.Publish();
        EXPECT_EQ(s.Cell(with).count, 16u);
        EXPECT_NE(s.Cell(with).address, 0u);
        EXPECT_EQ(s.Follows(with), s.catalog.GetTokens().Find("st").id);
        EXPECT_EQ(s.Cell(without).count, 0u);
        EXPECT_EQ(s.Follows(without), 0u);
    });
}

TEST(TangentTest, ClonesShareOneTangentSet) {
    ForEachScene([](Scene& s) {
        const uint32_t quad =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/Quad"}, TexturedQuad()).firstPrim;
        const PrimBlock copies = s.catalog.Clone(quad, std::vector<std::string>{"/A", "/B", "/C"});
        s.Publish();
        EXPECT_EQ(s.tangents.Stats().meshes, 4u);
        EXPECT_EQ(s.tangents.Stats().sets, 1u);
        EXPECT_EQ(s.tangents.Stats().computed, 1u);
        EXPECT_EQ(s.Cell(copies.Prim(2, 0)).address, s.Cell(quad).address);
    });
}

const std::vector<Float3> kFlat = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
const std::vector<Float3> kTurned = {{0, 0, 0}, {0, 1, 0}, {-1, 1, 0}, {-1, 0, 0}};

TEST(TangentTest, PointsReusingAFreedAddressGetTheirOwnTangents) {
    ForEachScene([](Scene& s) {
        const uint32_t quad =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/Quad"}, TexturedQuad()).firstPrim;
        const PrimBlock copies = s.catalog.Clone(quad, std::vector<std::string>{"/A", "/B"});
        const uint32_t first = copies.Prim(0, 0);
        const uint32_t second = copies.Prim(1, 0);
        ArrayHeap& float3s = s.store.Arrays(ColumnType::Float3Array);
        const ArrayRef old = float3s.Allocate(4);
        s.GivePoints(second, old, kFlat);
        s.GivePoints(first, float3s.Allocate(4), kFlat);
        s.Publish();

        s.GivePoints(second, float3s.Allocate(4), kFlat);
        const ArrayRef reused = s.Reuse(old);
        ASSERT_EQ(reused.address, old.address);
        s.GivePoints(first, reused, kTurned);
        s.Publish();

        EXPECT_EQ(s.tangents.Stats().computed, 2u);
        ExpectTangents(s.Read(first), {0, 1, 0, 1});
        ExpectTangents(s.Read(second), {1, 0, 0, 1});
    });
}

TEST(TangentTest, AMeshAddedOnARemovedMeshsArraysGetsItsOwnTangents) {
    ForEachScene([](Scene& s) {
        const uint32_t quad =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/Quad"}, TexturedQuad()).firstPrim;
        const uint32_t gone = s.catalog.Clone(quad, std::vector<std::string>{"/Gone"}).firstPrim;
        const ArrayRef old = s.store.Arrays(ColumnType::Float3Array).Allocate(4);
        s.GivePoints(gone, old, kFlat);
        s.Publish();

        s.catalog.Remove(gone);
        const uint32_t added = s.catalog.Clone(quad, std::vector<std::string>{"/Added"}).firstPrim;
        const ArrayRef reused = s.Reuse(old);
        ASSERT_EQ(reused.address, old.address);
        s.GivePoints(added, reused, kTurned);
        s.Publish();

        ExpectTangents(s.Read(added), {0, 1, 0, 1});
        ExpectTangents(s.Read(quad), {1, 0, 0, 1});
    });
}

TEST(TangentTest, RemovingStRemovesTheTangents) {
    ForEachScene([](Scene& s) {
        const uint32_t quad =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/Quad"}, TexturedQuad()).firstPrim;
        s.Publish();
        s.store.SetArray(kGprimPrimvars, s.catalog.GetPrims().Row(quad, SchemaId::Gprim), {});
        s.Publish();
        EXPECT_EQ(s.Cell(quad).count, 0u);
        EXPECT_EQ(s.Follows(quad), 0u);
        EXPECT_EQ(s.tangents.Stats().sets, 1u);
    });
}

TEST(TangentTest, SharedPointsMovedInPlaceRecomputeOnce) {
    ForEachScene([](Scene& s) {
        const uint32_t quad =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/Quad"}, TexturedQuad()).firstPrim;
        const PrimBlock copy = s.catalog.Clone(quad, std::vector<std::string>{"/Copy"});
        s.Publish();
        const uint32_t row = s.catalog.GetPrims().Row(quad, SchemaId::PointBased);
        const ArrayRef points = s.store.Array(kPointBasedPoints, row);
        const std::vector<Float3> turned = {{0, 0, 0}, {0, 1, 0}, {-1, 1, 0}, {-1, 0, 0}};
        s.store.Arrays(ColumnType::Float3Array).Write(points, std::as_bytes(std::span(turned)));
        s.store.SetArray(kPointBasedPoints, row, points);
        s.Publish();
        EXPECT_EQ(s.tangents.Stats().computed + s.tangents.Stats().deformed, 1u);
        EXPECT_EQ(s.tangents.Stats().sets, 1u);
        EXPECT_EQ(s.Cell(copy.Prim(0, 0)).address, s.Cell(quad).address);
    });
}

TEST(TangentTest, TransformsAloneComputeNothing) {
    ForEachScene([](Scene& s) {
        const uint32_t quad =
            s.catalog.Define<Mesh>(std::vector<std::string>{"/Quad"}, TexturedQuad()).firstPrim;
        s.Publish();
        const uint32_t runs = s.tangents.Runs();
        View(s.catalog, SchemaId::Xformable, std::vector<uint32_t>{quad})
            .Get<Double3>(kXformableTranslate)
            .Set(Double3{1, 0, 0});
        s.Publish();
        EXPECT_EQ(s.tangents.Runs(), runs);
    });
}

}
}
