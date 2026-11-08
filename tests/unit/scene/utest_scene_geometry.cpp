// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/bvh/builder.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/builders/meshes.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

const std::vector<Float3> kQuad = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};

struct Scene {
    device::Device& device;
    Store& store;
    Catalog& catalog;
    Timeline& timeline;
    Geometry& geometry;
    Ordinal ordinal = 0;

    uint32_t Quad(const std::string& path) {
        return catalog
            .Define<Mesh>(
                std::vector<std::string>{path},
                {.points = kQuad, .faceVertexIndices = {0, 1, 2, 3}, .faceVertexCounts = {4}})
            .firstPrim;
    }

    void Publish() {
        timeline.Publish(++ordinal).Wait();
    }

    uint32_t PointsRow(uint32_t prim) {
        return catalog.GetPrims().Row(prim, SchemaId::PointBased);
    }

    const MeshRecord& Record(uint32_t prim) {
        return geometry.Record(catalog.GetPrims().Row(prim, SchemaId::Mesh));
    }

    template <class T> std::vector<T> Derived(uint32_t prim, ArrayRef MeshRecord::* field) {
        const ArrayRef array = Record(prim).*field;
        const ColumnType type =
            sizeof(T) == sizeof(Float3) ? ColumnType::Float3Array : ColumnType::IntArray;
        return array.count == 0 ? std::vector<T>{} : Items<T>(store.Arrays(type).Read(array));
    }

    std::vector<Float3> Normals(uint32_t prim) {
        return Derived<Float3>(prim, &MeshRecord::normals);
    }

    uint64_t BlasCell(uint32_t prim) {
        return Record(prim).blas;
    }
};

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        if (device.Adapter().backend == device::Backend::Hip) {
            return;
        }
        Store store(device);
        Catalog catalog(store);
        Timeline timeline(catalog);
        const std::unique_ptr<bvh::Builder> builder = bvh::Builder::Create(device);
        Geometry geometry(catalog, *builder);
        timeline.Add(geometry);
        Scene scene{device, store, catalog, timeline, geometry};
        test(scene);
    });
}

TEST(GeometryTest, AMeshGetsABlasTablesAndSmoothNormals) {
    ForEachScene([](Scene& s) {
        const uint32_t prim = s.Quad("/Quad");
        s.Publish();
        const std::optional<MeshShape> shape = s.geometry.Find(prim);
        ASSERT_TRUE(shape);
        EXPECT_EQ(shape->triangles, 2u);
        EXPECT_EQ(s.BlasCell(prim), shape->blas->Handle());
        for (const Float3& normal : s.Normals(prim)) {
            EXPECT_NEAR(normal.z, 1.0f, 1e-6f);
        }
        EXPECT_EQ(s.geometry.Stats().built, 1u);
        EXPECT_EQ(s.geometry.Stats().triangulated, 1u);
    });
}

TEST(GeometryTest, FacesAreFannedOnTheDeviceAndInvalidOnesSkipped) {
    ForEachScene([](Scene& s) {
        const uint32_t mixed = s.catalog
                                   .Define<Mesh>(std::vector<std::string>{"/Mixed"},
                                                 {.points = std::vector<Float3>(8),
                                                  .faceVertexIndices = {0, 1, 2, 1, 3, 4, 2, 5, 6,
                                                                        0, -1, 2, 4, 3, 5, 6, 7},
                                                  .faceVertexCounts = {3, 4, 2, 3, 5}})
                                   .firstPrim;
        const uint32_t overrun = s.catalog
                                     .Define<Mesh>(std::vector<std::string>{"/Overrun"},
                                                   {.points = kQuad,
                                                    .faceVertexIndices = {0, 1, 2, 3, 0},
                                                    .faceVertexCounts = {4, 3}})
                                     .firstPrim;
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().triangulated, 2u);

        EXPECT_EQ(s.geometry.Find(mixed)->triangles, 6u);
        EXPECT_EQ(s.Derived<int32_t>(mixed, &MeshRecord::faces),
                  (std::vector<int32_t>{0, 1, 1, 4, 4, 4}));
        EXPECT_EQ(
            s.Derived<int32_t>(mixed, &MeshRecord::corners),
            (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 3, 5, 6, 12, 13, 14, 12, 14, 15, 12, 15, 16}));
        EXPECT_EQ(s.Derived<int32_t>(mixed, &MeshRecord::starts),
                  (std::vector<int32_t>{0, 1, 4, 6, 8, 13, 15, 17, 18}));
        EXPECT_EQ(s.Derived<int32_t>(mixed, &MeshRecord::adjacency),
                  (std::vector<int32_t>{0, 0, 1, 2, 0, 2, 1, 3, 1, 2, 3, 4, 5, 3, 4, 4, 5, 5}));

        EXPECT_EQ(s.geometry.Find(overrun)->triangles, 2u);
        EXPECT_EQ(s.Derived<int32_t>(overrun, &MeshRecord::faces), (std::vector<int32_t>{0, 0}));
        EXPECT_EQ(s.Derived<int32_t>(overrun, &MeshRecord::starts),
                  (std::vector<int32_t>{0, 2, 3, 5, 6}));
        EXPECT_EQ(s.Derived<int32_t>(overrun, &MeshRecord::adjacency),
                  (std::vector<int32_t>{0, 1, 0, 0, 1, 1}));
    });
}

TEST(GeometryTest, AMeshIndexingPastItsPointsHasNoShape) {
    ForEachScene([](Scene& s) {
        const uint32_t prim =
            s.catalog
                .Define<Mesh>(
                    std::vector<std::string>{"/Short"},
                    {.points = kQuad, .faceVertexIndices = {0, 1, 4}, .faceVertexCounts = {3}})
                .firstPrim;
        s.Publish();
        EXPECT_FALSE(s.geometry.Find(prim));
        EXPECT_EQ(s.BlasCell(prim), 0u);
        EXPECT_TRUE(s.Derived<int32_t>(prim, &MeshRecord::corners).empty());
    });
}

TEST(GeometryTest, ClonesInEveryWorldShareOneBlas) {
    ForEachScene([](Scene& s) {
        s.catalog.Define(SchemaId::World, std::vector<std::string>{"/Env0"});
        const uint32_t quad = s.Quad("/Env0/Quad");
        std::vector<std::string> targets;
        for (uint32_t i = 1; i < 8; ++i) {
            targets.push_back(fmt::format("/Env{}", i));
        }
        s.catalog.Clone(s.catalog.Find("/Env0"), targets);
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().meshes, 8u);
        EXPECT_EQ(s.geometry.Stats().shapes, 1u);
        EXPECT_EQ(s.geometry.Stats().built, 1u);
        const uint64_t blas = s.geometry.Find(quad)->blas->Handle();
        for (uint32_t i = 1; i < 8; ++i) {
            const uint32_t clone = s.catalog.Find(fmt::format("/Env{}/Quad", i));
            EXPECT_EQ(s.BlasCell(clone), blas);
        }
        EXPECT_EQ(s.geometry.Find(quad)->users, 8u);
    });
}

TEST(GeometryTest, PointsMovedInPlaceRefitAndRecomputeNormals) {
    ForEachScene([](Scene& s) {
        const uint32_t prim = s.Quad("/Quad");
        s.Publish();
        const uint64_t blas = s.geometry.Find(prim)->blas->Handle();
        const ArrayRef points = s.store.Array(kPointBasedPoints, s.PointsRow(prim));
        const std::vector<Float3> tilted = {{0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}};
        s.store.Arrays(ColumnType::Float3Array).Write(points, std::as_bytes(std::span(tilted)));
        s.store.SetArray(kPointBasedPoints, s.PointsRow(prim), points);
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().refit, 1u);
        EXPECT_EQ(s.geometry.Stats().built, 0u);
        EXPECT_EQ(s.geometry.Find(prim)->blas->Handle(), blas);
        const float d = 1.0f / std::sqrt(2.0f);
        for (const Float3& normal : s.Normals(prim)) {
            EXPECT_NEAR(normal.x, -d, 1e-5f);
            EXPECT_NEAR(normal.z, d, 1e-5f);
        }
    });
}

TEST(GeometryTest, AClonesOwnPointsGiveItItsOwnShape) {
    ForEachScene([](Scene& s) {
        const uint32_t quad = s.Quad("/Quad");
        const PrimBlock copies = s.catalog.Clone(quad, std::vector<std::string>{"/A", "/B"});
        s.Publish();
        ArrayHeap& float3s = s.store.Arrays(ColumnType::Float3Array);
        const ArrayRef own = float3s.Allocate(4);
        float3s.Write(own, std::as_bytes(std::span(kQuad)));
        s.store.SetArray(kPointBasedPoints, s.PointsRow(copies.Prim(0, 0)), own);
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().shapes, 2u);
        EXPECT_EQ(s.geometry.Stats().built, 1u);
        EXPECT_EQ(s.geometry.Stats().triangulated, 0u);
        EXPECT_EQ(s.geometry.Find(quad)->users, 2u);
        EXPECT_NE(s.BlasCell(copies.Prim(0, 0)), s.BlasCell(quad));
        EXPECT_EQ(s.BlasCell(copies.Prim(1, 0)), s.BlasCell(quad));
    });
}

TEST(GeometryTest, PointsReusingAFreedAddressGetTheirOwnShape) {
    ForEachScene([](Scene& s) {
        const uint32_t quad = s.Quad("/Quad");
        const PrimBlock copies = s.catalog.Clone(quad, std::vector<std::string>{"/A", "/B"});
        const uint32_t a = copies.Prim(0, 0);
        const uint32_t b = copies.Prim(1, 0);
        ArrayHeap& float3s = s.store.Arrays(ColumnType::Float3Array);
        const auto give = [&](uint32_t prim, const ArrayRef& array,
                              const std::vector<Float3>& points) {
            float3s.Write(array, std::as_bytes(std::span(points)));
            s.store.SetArray(kPointBasedPoints, s.PointsRow(prim), array);
        };
        const ArrayRef old = float3s.Allocate(4);
        give(a, old, kQuad);
        give(b, float3s.Allocate(4), kQuad);
        s.Publish();

        give(a, float3s.Allocate(4), kQuad);
        std::vector<ArrayRef> spare;
        ArrayRef reused = float3s.Allocate(4);
        while (reused.address != old.address && spare.size() < 64) {
            spare.push_back(reused);
            reused = float3s.Allocate(4);
        }
        ASSERT_EQ(reused.address, old.address);
        for (const ArrayRef& array : spare) {
            float3s.Free(array);
        }
        const std::vector<Float3> tilted = {{0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}};
        give(b, reused, tilted);
        s.Publish();

        EXPECT_EQ(s.geometry.Stats().built, 2u);
        EXPECT_NE(s.BlasCell(a), s.BlasCell(b));
        const float d = 1.0f / std::sqrt(2.0f);
        for (const Float3& normal : s.Normals(b)) {
            EXPECT_NEAR(normal.x, -d, 1e-5f);
            EXPECT_NEAR(normal.z, d, 1e-5f);
        }
        for (const Float3& normal : s.Normals(a)) {
            EXPECT_NEAR(normal.z, 1.0f, 1e-5f);
        }
    });
}

TEST(GeometryTest, NewTopologyRebuilds) {
    ForEachScene([](Scene& s) {
        const uint32_t prim = s.Quad("/Quad");
        s.Publish();
        ArrayHeap& ints = s.store.Arrays(ColumnType::IntArray);
        const std::vector<int32_t> counts = {3, 3};
        const std::vector<int32_t> indices = {0, 1, 2, 0, 2, 3};
        const ArrayRef faces = ints.Allocate(2);
        ints.Write(faces, std::as_bytes(std::span(counts)));
        const ArrayRef corners = ints.Allocate(6);
        ints.Write(corners, std::as_bytes(std::span(indices)));
        const uint32_t row = s.catalog.GetPrims().Row(prim, SchemaId::Mesh);
        s.store.SetArray(kMeshFaceVertexCounts, row, faces);
        s.store.SetArray(kMeshFaceVertexIndices, row, corners);
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().built, 1u);
        EXPECT_EQ(s.geometry.Stats().triangulated, 1u);
        EXPECT_EQ(s.geometry.Stats().topologies, 1u);
        EXPECT_EQ(s.geometry.Find(prim)->triangles, 2u);
    });
}

TEST(GeometryTest, RemovingTheLastUserFreesTheShape) {
    ForEachScene([](Scene& s) {
        const uint32_t a = s.Quad("/A");
        const PrimBlock b = s.catalog.Clone(a, std::vector<std::string>{"/B"});
        s.Publish();
        s.catalog.Remove(a);
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().shapes, 1u);
        EXPECT_EQ(s.geometry.Find(b.Prim(0, 0))->users, 1u);
        s.catalog.Remove(b.Prim(0, 0));
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().shapes, 0u);
        EXPECT_EQ(s.geometry.Stats().topologies, 0u);
        EXPECT_EQ(s.geometry.Stats().meshes, 0u);
    });
}

TEST(GeometryTest, MeshesWithoutFacesHaveNoBlas) {
    ForEachScene([](Scene& s) {
        const uint32_t empty =
            s.catalog.Define(SchemaId::Mesh, std::vector<std::string>{"/Empty"}).firstPrim;
        const uint32_t loose =
            s.catalog
                .Define<Mesh>(
                    std::vector<std::string>{"/Loose"},
                    {.points = kQuad, .faceVertexIndices = {0, 9, 2}, .faceVertexCounts = {3}})
                .firstPrim;
        s.Publish();
        EXPECT_FALSE(s.geometry.Find(empty));
        EXPECT_FALSE(s.geometry.Find(loose));
        EXPECT_EQ(s.BlasCell(empty), 0u);
        EXPECT_EQ(s.BlasCell(loose), 0u);
    });
}

TEST(GeometryTest, UnrelatedWritesBuildNothing) {
    ForEachScene([](Scene& s) {
        s.Quad("/Quad");
        s.catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Ball"});
        s.Publish();
        const uint64_t submitted = s.device.GetQueue().Submitted();
        s.Publish();
        EXPECT_EQ(s.device.GetQueue().Submitted(), submitted);
        s.catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Other"});
        s.Publish();
        EXPECT_EQ(s.geometry.Stats().built, 0u);
        EXPECT_EQ(s.geometry.Stats().refit, 0u);
    });
}

}
}
