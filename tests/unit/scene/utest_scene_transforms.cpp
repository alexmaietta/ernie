// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/transforms.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

struct World {
    ColumnStore& store;
    Structure& structure;
    Publisher& publisher;
    TransformTable& hierarchy;
    Ordinal ordinal = 0;

    uint32_t Define(SchemaId schema, const std::string& path) {
        return structure.Define(schema, std::vector<std::string>{path}).firstPrim;
    }

    template <class T> uint32_t Define(const std::string& path, const typename T::Desc& desc) {
        return structure.Define<T>(std::vector<std::string>{path}, desc).firstPrim;
    }

    View One(uint32_t prim, SchemaId schema = SchemaId::Imageable) {
        return View(structure, schema, std::vector<uint32_t>{prim});
    }

    void Publish() {
        publisher.Publish(++ordinal).Wait();
    }

    WorldTransform Transform(uint32_t prim) {
        const std::vector<uint32_t> prims = {prim};
        return store
            .ReadRows<WorldTransform>(hierarchy.Transforms(),
                                      structure.GetPrims().Rows(prims, SchemaId::Xformable))
            .front();
    }

    std::vector<double> Translation(uint32_t prim) {
        const WorldTransform world = Transform(prim);
        return {double(world.high.x) + double(world.low.x),
                double(world.high.y) + double(world.low.y),
                double(world.high.z) + double(world.low.z)};
    }

    ResolvedImageable Resolved(uint32_t prim) {
        const std::vector<uint32_t> prims = {prim};
        return store
            .ReadRows<ResolvedImageable>(hierarchy.Imageables(),
                                         structure.GetPrims().Rows(prims, SchemaId::Imageable))
            .front();
    }
};

template <class Test> void ForEachWorld(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        Publisher publisher(structure);
        TransformTable hierarchy(structure);
        publisher.Add(hierarchy);
        World world{store, structure, publisher, hierarchy};
        test(world);
    });
}

void ExpectNear(const std::vector<double>& actual, const std::vector<double>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5) << i;
    }
}

TEST(TransformsTest, ChildrenComposeWithTheirParents) {
    ForEachWorld([](World& w) {
        const double s = std::sqrt(0.5);
        const uint32_t parent =
            w.Define<Xform>("/Parent", {.translate = {1, 0, 0}, .orient = {0, 0, s, s}});
        const uint32_t child = w.Define<Sphere>("/Parent/Child", {.translate = {0, 2, 0}});
        const uint32_t loose = w.Define<Sphere>("/Loose", {.translate = {0, 0, 5}});
        w.Publish();
        ExpectNear(w.Translation(parent), {1, 0, 0});
        ExpectNear(w.Translation(child), {-1, 0, 0});
        ExpectNear(w.Translation(loose), {0, 0, 5});
        const WorldTransform world = w.Transform(child);
        EXPECT_NEAR(world.x.y, 1.0f, 1e-6f);
    });
}

TEST(TransformsTest, ResettingTheStackIgnoresAncestors) {
    ForEachWorld([](World& w) {
        w.Define<Xform>("/Parent", {.translate = {10, 0, 0}, .scale = {2, 2, 2}});
        const uint32_t body =
            w.Define<Xform>("/Parent/Body", {.translate = {0, 1, 0}, .resetXformStack = true});
        const uint32_t part = w.Define<Sphere>("/Parent/Body/Part", {.translate = {0, 0, 1}});
        w.Publish();
        ExpectNear(w.Translation(body), {0, 1, 0});
        ExpectNear(w.Translation(part), {0, 1, 1});
    });
}

TEST(TransformsTest, MovingAParentMovesItsSubtreeAndNothingElse) {
    ForEachWorld([](World& w) {
        const uint32_t parent = w.Define(SchemaId::Xform, "/Parent");
        const uint32_t child = w.Define<Sphere>("/Parent/Child", {.translate = {0, 1, 0}});
        const uint32_t other = w.Define<Sphere>("/Other", {.translate = {0, 0, 3}});
        w.Publish();
        w.One(parent, SchemaId::Xformable).Get<Double3>(kXformableTranslate).Set(Double3{5, 0, 0});
        w.Publish();
        ExpectNear(w.Translation(child), {5, 1, 0});
        ExpectNear(w.Translation(other), {0, 0, 3});
    });
}

TEST(TransformsTest, OnlyDirtyWorldsAreDispatched) {
    ForEachWorld([](World& w) {
        std::vector<uint32_t> balls;
        for (uint32_t i = 0; i < 3; ++i) {
            w.Define(SchemaId::World, fmt::format("/Env{}", i));
            balls.push_back(w.Define(SchemaId::Sphere, fmt::format("/Env{}/Ball", i)));
        }
        w.Define(SchemaId::Sphere, "/Sun");
        w.Publish();
        EXPECT_EQ(w.hierarchy.Dispatched(), 7u);

        w.One(balls[1], SchemaId::Xformable)
            .Get<Double3>(kXformableTranslate)
            .Set(Double3{0, 4, 0});
        w.Publish();
        EXPECT_EQ(w.hierarchy.Dispatched(), 2u);
        ExpectNear(w.Translation(balls[1]), {0, 4, 0});

        const uint32_t dispatched = w.hierarchy.Dispatched();
        w.One(balls[2], SchemaId::Sphere).Get<float>(kSphereRadius).Set(2.0f);
        w.Publish();
        EXPECT_EQ(w.hierarchy.Dispatched(), dispatched);

        w.One(w.structure.Find("/Sun"), SchemaId::Xformable)
            .Get<Double3>(kXformableTranslate)
            .Set(Double3{0, 0, 1});
        w.Publish();
        EXPECT_EQ(w.hierarchy.Dispatched(), 7u);
    });
}

TEST(TransformsTest, VisibilityPurposeAndMaterialAreInherited) {
    ForEachWorld([](World& w) {
        const uint32_t look = w.Define(SchemaId::UsdPreviewSurface, "/Look");
        const uint32_t other = w.Define(SchemaId::UsdPreviewSurface, "/Other");
        const uint32_t parent = w.Define<Xform>("/Parent", {.materialBinding = {"/Look"},
                                                            .visibility = Visibility::Invisible,
                                                            .purpose = Purpose::Guide});
        const uint32_t child = w.Define(SchemaId::Sphere, "/Parent/Child");
        const uint32_t own = w.Define<Sphere>("/Parent/Own", {.materialBinding = {"/Other"}});
        const uint32_t free = w.Define(SchemaId::Sphere, "/Free");
        w.Publish();
        const uint32_t invisible = uint32_t(Visibility::Invisible);
        EXPECT_EQ(w.Resolved(child).visibility, invisible);
        EXPECT_EQ(w.Resolved(free).visibility, uint32_t(Visibility::Inherited));
        EXPECT_EQ(w.Resolved(child).purpose, uint32_t(Purpose::Guide));
        EXPECT_EQ(w.Resolved(child).material, look);
        EXPECT_EQ(w.Resolved(own).material, other);
        EXPECT_EQ(w.Resolved(free).material, kNoPrim);

        w.One(parent).Get<Visibility>(kImageableVisibility).Set(Visibility::Inherited);
        w.Publish();
        EXPECT_EQ(w.Resolved(child).visibility, uint32_t(Visibility::Inherited));

        w.structure.Remove(look);
        w.Publish();
        EXPECT_EQ(w.Resolved(child).material, kNoPrim);
    });
}

TEST(TransformsTest, TranslationKeepsDoublePrecision) {
    ForEachWorld([](World& w) {
        w.Define<Xform>("/Far", {.translate = {1e7 + 0.25, -3e6, 0}});
        const uint32_t near = w.Define<Sphere>("/Far/Near", {.translate = {0.125, 0.5, 0}});
        w.Publish();
        const std::vector<double> translation = w.Translation(near);
        EXPECT_EQ(translation[0], 1e7 + 0.375);
        EXPECT_EQ(translation[1], -3e6 + 0.5);
    });
}

TEST(TransformsTest, NothingWrittenRunsNothing) {
    ForEachWorld([](World& w) {
        w.Define(SchemaId::Sphere, "/Ball");
        w.Publish();
        EXPECT_EQ(w.hierarchy.Dispatched(), 1u);
        w.Define(SchemaId::Scope, "/Empty");
        w.Publish();
        EXPECT_EQ(w.hierarchy.Dispatched(), 2u);
        const uint64_t ordinal = w.ordinal;
        w.Publish();
        EXPECT_EQ(w.ordinal, ordinal + 1);
        EXPECT_EQ(w.hierarchy.Dispatched(), 2u);
    });
}

}
}
