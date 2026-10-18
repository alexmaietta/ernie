// SPDX-License-Identifier: BSD-3-Clause
/// @file storage_test.cpp
/// @brief Tests of the columnar prim store: buckets, rows, values and change sets.
#include "ernie/scene/storage.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

/// A store with a bucket of spheres whose radius falls back to 1.
struct Spheres {
    PrimStore store;
    AttributeId radius = store.Tokens().Intern("radius");
    AttributeId color = store.Tokens().Intern("color");
    BucketId bucket = store.AddBucket(
        {store.Tokens().Intern("Sphere"), {}},
        std::vector<ColumnLayout>{ColumnLayout::Of<double>(radius, 1.0),
                                  ColumnLayout::Of<glm::dvec3>(color, glm::dvec3(0.5))});

    PrimId Define(const std::string& path) {
        return store.Define(store.Paths().Intern(Path(path)), bucket);
    }
};

TEST(PrimStore, DefinesFindsAndRemovesPrims) {
    Spheres s;
    const PrimId a = s.Define("/World/A");
    const PrimId b = s.Define("/World/B");
    EXPECT_EQ(s.store.PrimCount(), 2u);
    EXPECT_EQ(s.store.Find(s.store.Paths().Intern("/World/B")), b);
    EXPECT_EQ(s.Define("/World/A"), a) << "defining again in the same bucket returns the prim";
    EXPECT_EQ(s.store.PrimRow(b), 1u);

    s.store.Remove(a);
    EXPECT_FALSE(s.store.Alive(a));
    EXPECT_EQ(s.store.PrimRow(b), 0u) << "the last row moves into the removed one's place";
    EXPECT_FALSE(s.store.Find(s.store.Paths().Intern("/World/A")));
    EXPECT_THROW(s.store.PrimRow(a), std::invalid_argument);

    const PrimId c = s.Define("/World/C");
    EXPECT_EQ(c.slot, a.slot) << "slots are reused";
    EXPECT_NE(c.generation, a.generation) << "but an old handle stays dead";
    EXPECT_FALSE(s.store.Alive(a));
}

TEST(PrimStore, MovingToAnotherBucketKeepsTheIdAndSharedAuthoredValues) {
    Spheres s;
    const AttributeId height = s.store.Tokens().Intern("height");
    const BucketId cylinders =
        s.store.AddBucket({s.store.Tokens().Intern("Cylinder"), {}},
                          std::vector<ColumnLayout>{ColumnLayout::Of<double>(s.radius, 1.0),
                                                    ColumnLayout::Of<double>(height, 2.0)});
    const PrimId a = s.Define("/A");
    const PrimId b = s.Define("/B");
    s.store.Set(a, s.radius, 3.0);
    s.store.Set(a, s.color, glm::dvec3(1.0));
    const std::uint64_t cursor = s.store.Version();

    s.store.Move(a, cylinders);
    EXPECT_TRUE(s.store.Alive(a));
    EXPECT_EQ(s.store.PrimBucket(a), cylinders);
    EXPECT_EQ(*s.store.Get<double>(a, s.radius), 3.0) << "both buckets have radius";
    EXPECT_EQ(s.store.Get<double>(a, height), nullptr) << "unauthored in its new bucket";
    EXPECT_EQ(s.store.PrimRow(b), 0u) << "the old bucket's last row filled the gap";

    const ChangeSet changes = s.store.ChangesSince(cursor);
    ASSERT_EQ(changes.rows.size(), 3u) << "removed and moved in one bucket, added in the other";
    EXPECT_EQ(changes.rows[0].bucket, s.bucket);
    EXPECT_EQ(changes.rows[0].change.kind, StructuralChange::Kind::Removed);
    EXPECT_EQ(changes.rows[1].change.kind, StructuralChange::Kind::Moved);
    EXPECT_EQ(changes.rows[2].bucket, cylinders);
    EXPECT_EQ(changes.rows[2].change.kind, StructuralChange::Kind::Added);
}

TEST(PrimStore, AMovedRowIsReportedChangedAtItsNewPlace) {
    Spheres s;
    std::vector<PrimId> prims;
    for (int i = 0; i < 100; ++i) {
        prims.push_back(s.Define("/P" + std::to_string(i)));
    }
    const std::uint64_t cursor = s.store.Version();
    s.store.Set(prims[99], s.radius, 5.0);
    s.store.Remove(prims[3]);
    ASSERT_EQ(s.store.PrimRow(prims[99]), 3u);
    const ChangeSet changes = s.store.ChangesSince(cursor);
    bool reported = false;
    for (const ChangeSet::Values& values : changes.values) {
        for (const RowRange range : values.rows) {
            reported |= values.attribute == s.radius && range.begin <= 3 && 3 < range.end;
        }
    }
    EXPECT_TRUE(reported) << "the edit made at row 99 now lives at row 3";
}

TEST(PrimStore, UnauthoredRowsHoldTheFallbackForWholeColumnReads) {
    Spheres s;
    const PrimId a = s.Define("/A");
    s.Define("/B");
    EXPECT_EQ(s.store.Get<double>(a, s.radius), nullptr) << "unauthored";
    s.store.Set(a, s.radius, 2.0);
    EXPECT_EQ(*s.store.Get<double>(a, s.radius), 2.0);
    EXPECT_EQ(s.store.Read<double>(s.bucket, s.radius)[1], 1.0)
        << "the fallback, as the GPU reads it";

    s.store.Clear(a, s.radius);
    EXPECT_EQ(s.store.Get<double>(a, s.radius), nullptr);
    EXPECT_EQ(s.store.Read<double>(s.bucket, s.radius)[0], 1.0);
    EXPECT_THROW(s.store.Set(a, s.radius, glm::dvec3(1.0)), std::invalid_argument)
        << "another type";
}

TEST(PrimStore, CustomAttributesAddColumns) {
    Spheres s;
    const PrimId a = s.Define("/A");
    const PrimId b = s.Define("/B");
    const AttributeId note = s.store.Tokens().Intern("ernie:note");
    s.store.Set(b, note, std::int32_t{7});
    EXPECT_EQ(*s.store.Get<std::int32_t>(b, note), 7);
    EXPECT_EQ(s.store.Get<std::int32_t>(a, note), nullptr);
    EXPECT_EQ(s.store.Read<std::int32_t>(s.bucket, note).size(), 2u);
}

TEST(PrimStore, RemovingMovesTheLastRowsValuesAndAuthoredBits) {
    Spheres s;
    const PrimId a = s.Define("/A");
    s.Define("/B");
    const PrimId c = s.Define("/C");
    s.store.Set(c, s.radius, 3.0);
    s.store.Remove(a);
    EXPECT_EQ(s.store.PrimRow(c), 0u);
    EXPECT_EQ(*s.store.Get<double>(c, s.radius), 3.0);
    EXPECT_EQ(s.store.Read<double>(s.bucket, s.radius).size(), 2u);
}

TEST(PrimStore, BucketsKeyOnTypeAndSortedApiSchemas) {
    PrimStore store;
    const Token light = store.Tokens().Intern("SphereLight");
    const Token shaping = store.Tokens().Intern("ShapingAPI");
    const Token spectral = store.Tokens().Intern("ErnieSpectralLightAPI");
    const BucketId a = store.AddBucket({light, {shaping, spectral}}, {});
    EXPECT_EQ(store.AddBucket({light, {spectral, shaping, shaping}}, {}), a);
    EXPECT_NE(store.AddBucket({light, {}}, {}), a);
    const PrimId prim = store.Define(store.Paths().Intern("/L"), a);
    EXPECT_THROW(store.Define(store.Paths().Intern("/L"), store.AddBucket({light, {}}, {})),
                 std::invalid_argument);
    EXPECT_EQ(store.PrimBucket(prim), a);
}

TEST(PrimStore, ChangeSetsReportOnlyWhatChangedSinceTheCursor) {
    Spheres s;
    std::vector<PathId> paths;
    for (int i = 0; i < 200; ++i) {
        paths.push_back(s.store.Paths().Intern(Path("/P" + std::to_string(i))));
    }
    const std::vector<PrimId> prims = s.store.DefineMany(paths, s.bucket);
    ChangeSet changes = s.store.ChangesSince(0);
    ASSERT_EQ(changes.rows.size(), 1u);
    EXPECT_EQ(changes.rows[0].change.kind, StructuralChange::Kind::Added);
    EXPECT_EQ(changes.rows[0].change.count, 200u);
    EXPECT_TRUE(changes.values.empty());

    std::uint64_t cursor = changes.version;
    EXPECT_TRUE(s.store.ChangesSince(cursor).rows.empty());
    EXPECT_TRUE(s.store.ChangesSince(cursor).values.empty());

    s.store.Set(prims[130], s.radius, 2.0);
    changes = s.store.ChangesSince(cursor);
    ASSERT_EQ(changes.values.size(), 1u);
    EXPECT_EQ(changes.values[0].attribute, s.radius);
    EXPECT_EQ(changes.values[0].rows, (std::vector<RowRange>{{128, 192}})) << "the row's chunk";

    cursor = changes.version;
    s.store.Write<glm::dvec3>(s.bucket, s.color);
    changes = s.store.ChangesSince(cursor);
    ASSERT_EQ(changes.values.size(), 1u);
    EXPECT_EQ(changes.values[0].rows, (std::vector<RowRange>{{0, 200}})) << "chunks merge";

    cursor = changes.version;
    s.store.Remove(prims[5]);
    changes = s.store.ChangesSince(cursor);
    ASSERT_EQ(changes.rows.size(), 2u);
    EXPECT_EQ(changes.rows[0].change.kind, StructuralChange::Kind::Removed);
    EXPECT_EQ(changes.rows[1].change.kind, StructuralChange::Kind::Moved);
    EXPECT_EQ(changes.rows[1].change.from, 199u);
    EXPECT_EQ(changes.rows[1].change.row, 5u);
}

TEST(PrimStore, ReadersTooFarBehindResyncTheBucket) {
    Spheres s;
    const PathId path = s.store.Paths().Intern("/P");
    for (int i = 0; i < 70'000; ++i) {
        s.store.Remove(s.store.Define(path, s.bucket));
    }
    const ChangeSet changes = s.store.ChangesSince(0);
    EXPECT_EQ(changes.resets, std::vector<BucketId>{s.bucket});
    EXPECT_TRUE(changes.rows.empty());
    EXPECT_FALSE(s.store.ChangesSince(s.store.Version() - 2).rows.empty()) << "recent changes stay";
}

}  // namespace
}  // namespace ernie
