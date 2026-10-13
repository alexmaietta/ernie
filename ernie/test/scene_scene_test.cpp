// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <array>

using ernie::Change;
using ernie::ChangeCursor;
using ernie::Dirty;
using ernie::Path;
using ernie::Prim;
using ernie::Scene;
using ernie::Tensor;
using ernie::Value;
namespace tokens = ernie::tokens;

namespace {

Dirty BitsOf(const std::vector<Change>& changes, const Path& path) {
    for (const Change& change : changes) {
        if (change.path == path) {
            return change.bits;
        }
    }
    return Dirty::None;
}

}  // namespace

TEST(Value, HoldsTypedValuesAndTokens) {
    EXPECT_TRUE(Value().Empty());
    EXPECT_EQ(*Value(2.5f).Get<float>(), 2.5f);
    EXPECT_EQ(Value(2.5f).Get<double>(), nullptr);
    EXPECT_EQ(*Value("lambert").Get<std::string>(), "lambert");
    EXPECT_EQ(*Value(tokens::kInvisible).Get<std::string>(),
              "invisible");
    EXPECT_TRUE(Value(Path("/Looks/Rock")).Is<Path>());
    EXPECT_TRUE(Value(glm::ivec2(640, 480)).Is<glm::ivec2>());
    EXPECT_TRUE(
        Value(std::vector<float>{1.0f}).Is<std::vector<float>>());
    EXPECT_EQ(Value(std::vector<Path>{"/A", "/B"})
                  .Get<std::vector<Path>>()
                  ->size(),
              2u);
    EXPECT_EQ(Value(glm::dvec2(1.0, 2.0)),
              Value(glm::dvec2(1.0, 2.0)));
    EXPECT_NE(Value(1), Value(1.0));
}

TEST(Scene, DirtyBitsFollowAttributeNames) {
    EXPECT_EQ(ernie::DirtyOf(tokens::kTransform), Dirty::Transform);
    EXPECT_EQ(ernie::DirtyOf(tokens::kVisibility), Dirty::Visibility);
    EXPECT_EQ(ernie::DirtyOf(tokens::kMaterialBinding),
              Dirty::Material);
    EXPECT_EQ(ernie::DirtyOf(tokens::kFaceVertexIndices),
              Dirty::Topology);
    EXPECT_EQ(ernie::DirtyOf(tokens::kProtoIndices), Dirty::Topology);
    EXPECT_EQ(ernie::DirtyOf(tokens::kHeights), Dirty::Points);
    EXPECT_EQ(ernie::DirtyOf(tokens::kPositions), Dirty::Points);
    EXPECT_EQ(ernie::DirtyOf(tokens::kDisplayColor), Dirty::Primvars);
    EXPECT_EQ(ernie::DirtyOf(tokens::kIntensity), Dirty::Params);
    EXPECT_TRUE(Any(Dirty::Points | Dirty::Params, Dirty::Params));
    EXPECT_FALSE(Any(Dirty::Points, Dirty::Params | Dirty::Added));
}

TEST(Scene, DefinesPrimsWithTypelessAncestors) {
    Scene scene;
    const Prim sun =
        scene.Define("/World/Lights/Sun", tokens::kDistantLight);
    EXPECT_TRUE(sun.IsValid());
    EXPECT_EQ(sun.Type(), "DistantLight");
    EXPECT_EQ(scene.TypeOf("/World/Lights"), "");
    EXPECT_EQ(scene.TypeOf("/"), "");
    EXPECT_EQ(scene.Prims(),
              (std::vector<Path>{"/World", "/World/Lights",
                                 "/World/Lights/Sun"}));
    EXPECT_EQ(scene.Prims(tokens::kDistantLight),
              (std::vector<Path>{"/World/Lights/Sun"}));

    scene.Define("/World/Terrain", tokens::kTerrain);
    scene.Define("/World/Lights/Fill", tokens::kSphereLight);
    EXPECT_EQ(scene.Children("/World"),
              (std::vector<Path>{"/World/Lights", "/World/Terrain"}));
    EXPECT_EQ(scene.Children("/"), (std::vector<Path>{"/World"}));
    EXPECT_FALSE(scene.GetPrim("/Nope").IsValid());
    EXPECT_FALSE(Prim().IsValid());
    EXPECT_THROW(Prim().Type(), std::invalid_argument);
    EXPECT_THROW(scene.Define(Path::Root(), "Xform"),
                 std::invalid_argument);
    EXPECT_THROW(scene.Define(Path(), "Xform"),
                 std::invalid_argument);
    EXPECT_THROW(scene.TypeOf("/Nope"), std::invalid_argument);
}

TEST(Scene, PrimsDefineChildrenAtRelativePaths) {
    Scene scene;
    const Prim world = scene.Define("/World", tokens::kXform);
    const Prim rock = world.Define("Rock", tokens::kSphere);
    EXPECT_EQ(rock.GetPath(), Path("/World/Rock"));
    EXPECT_EQ(rock.Type(), "Sphere");
    const Prim arm = world.Define("Rig/Arm", tokens::kCube);
    EXPECT_EQ(arm.GetPath(), Path("/World/Rig/Arm"));
    EXPECT_EQ(scene.TypeOf("/World/Rig"), "");
    EXPECT_THROW(world.Define("", tokens::kCube),
                 std::invalid_argument);
    EXPECT_THROW(world.Define("Rig/", tokens::kCube),
                 std::invalid_argument);
    EXPECT_THROW(world.Define("/Rock", tokens::kCube),
                 std::invalid_argument);
    EXPECT_THROW(world.Define("../Rock", tokens::kCube),
                 std::invalid_argument);
    EXPECT_THROW(Prim().Define("Rock", tokens::kCube),
                 std::invalid_argument);

    EXPECT_EQ(world.GetPrim("Rig/Arm").GetPath(),
              Path("/World/Rig/Arm"));
    EXPECT_FALSE(world.GetPrim("Missing").IsValid());
    EXPECT_FALSE(Prim().GetPrim("Rock").IsValid());
    EXPECT_THROW(world.GetPrim("../Rock"), std::invalid_argument);
}

TEST(Scene, SetsAndGetsValuesByUsdName) {
    Scene scene;
    Prim sun = scene.Define("/Sun", tokens::kDistantLight);
    sun.Set(tokens::kIntensity, 3.0f).Set(tokens::kAngle, 0.53f);
    EXPECT_EQ(sun.Get<float>(tokens::kIntensity, 1.0f), 3.0f);
    EXPECT_EQ(sun.Get<float>(tokens::kColor, 1.0f), 1.0f) << "unset";
    EXPECT_EQ(sun.Get<double>(tokens::kIntensity, 7.0), 7.0)
        << "other type";
    EXPECT_EQ(Prim().Get<float>(tokens::kIntensity, 2.0f), 2.0f);
    EXPECT_EQ(Prim().Get(tokens::kIntensity), nullptr);
    EXPECT_EQ(Prim().Read(tokens::kHeights), nullptr);
    EXPECT_EQ(scene.Get("/Nope", tokens::kIntensity), nullptr);
    EXPECT_EQ(scene.Attributes("/Sun"),
              (std::vector<std::string>{"inputs:angle",
                                        "inputs:intensity"}));
    EXPECT_THROW(scene.Set("/Nope", tokens::kIntensity, 1.0f),
                 std::invalid_argument);
    EXPECT_THROW(Prim().Set(tokens::kIntensity, 1.0f),
                 std::invalid_argument);

    EXPECT_TRUE(scene.Clear("/Sun", tokens::kAngle));
    EXPECT_FALSE(scene.Clear("/Sun", tokens::kAngle));
    EXPECT_EQ(sun.Get(tokens::kAngle), nullptr);
}

TEST(Scene, ValuesAndTensorsReplaceEachOther) {
    Scene scene;
    Prim terrain = scene.Define("/Terrain", tokens::kTerrain);
    const std::array<float, 4> heights{1, 2, 3, 4};
    terrain.Write(tokens::kHeights,
                  Tensor::Copy<float>(heights, {2, 2}));
    ASSERT_NE(terrain.Read(tokens::kHeights), nullptr);
    EXPECT_EQ(terrain.Read(tokens::kHeights)->As<float>()[3], 4.0f);
    EXPECT_EQ(terrain.Get(tokens::kHeights), nullptr);
    EXPECT_EQ(scene.Read("/Nope", tokens::kHeights), nullptr);

    terrain.Set(tokens::kHeights, 5.0f);
    EXPECT_EQ(terrain.Read(tokens::kHeights), nullptr);
    terrain.Write(tokens::kHeights,
                  Tensor::Copy<float>(heights, {4}));
    EXPECT_EQ(terrain.Get(tokens::kHeights), nullptr);
    terrain.Write(tokens::kHeights,
                  Tensor::Copy<float>(heights, {2, 2}));
    EXPECT_EQ(terrain.Read(tokens::kHeights)->Rank(), 2u);

    EXPECT_TRUE(scene.Clear("/Terrain", tokens::kHeights));
    EXPECT_EQ(terrain.Read(tokens::kHeights), nullptr);
    EXPECT_THROW(scene.Write("/Nope", tokens::kHeights, Tensor()),
                 std::invalid_argument);
    EXPECT_THROW(Prim().Write(tokens::kHeights, Tensor()),
                 std::invalid_argument);
}

TEST(Scene, BulkWritesShareOneTensorAcrossPrims) {
    Scene scene;
    const std::vector<Path> particles{
        "/Particles/p0", "/Particles/p1", "/Particles/p2"};
    for (const Path& path : particles) {
        scene.Define(path, tokens::kXform);
    }
    const std::array<float, 9> positions{0, 0, 0, 1, 1, 1, 2, 2, 2};
    const Tensor rows = Tensor::Copy<float>(positions, {3, 3});
    scene.Write(particles, tokens::kPositions, rows);
    const Tensor* p2 =
        scene.Read("/Particles/p2", tokens::kPositions);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(p2->Data(), rows.As<float>() + 6)
        << "a view, not a copy";
    EXPECT_EQ(p2->Shape(), (std::vector<std::int64_t>{3}));

    const std::vector<Path> missing{"/Particles/p0", "/Nope",
                                    "/Particles/p2"};
    EXPECT_THROW(scene.Write(missing, tokens::kVelocities, rows),
                 std::invalid_argument);
    EXPECT_EQ(scene.Read("/Particles/p0", tokens::kVelocities),
              nullptr)
        << "nothing written";
    EXPECT_THROW(scene.Write(std::span(particles).first(2),
                             tokens::kPositions, rows),
                 std::invalid_argument);
    EXPECT_THROW(scene.Write(particles, tokens::kPositions, Tensor()),
                 std::invalid_argument);
}

TEST(Scene, ComposesTransformsAndVisibilityDownTheTree) {
    Scene scene;
    scene.Define("/World", tokens::kXform)
        .Set(tokens::kTransform,
             glm::translate(glm::dmat4(1.0),
                            glm::dvec3(10.0, 0.0, 0.0)));
    scene.Define("/World/Rover", tokens::kXform)
        .Set(tokens::kTransform,
             glm::scale(glm::dmat4(1.0), glm::dvec3(2.0)));
    const glm::dvec4 origin = scene.WorldTransform("/World/Rover") *
                              glm::dvec4(1.0, 0.0, 0.0, 1.0);
    EXPECT_EQ(origin, glm::dvec4(12.0, 0.0, 0.0, 1.0));
    EXPECT_EQ(scene.LocalTransform("/"), glm::dmat4(1.0));
    EXPECT_THROW(scene.WorldTransform("/Nope"),
                 std::invalid_argument);

    EXPECT_TRUE(scene.IsVisible("/World/Rover"));
    scene.Set("/World", tokens::kVisibility, tokens::kInvisible);
    EXPECT_FALSE(scene.IsVisible("/World/Rover"));
    scene.Set("/World", tokens::kVisibility, tokens::kInherited);
    EXPECT_TRUE(scene.IsVisible("/World/Rover"));
    EXPECT_THROW(scene.IsVisible("/Nope"), std::invalid_argument);
}

TEST(Scene, ChangesReportEachPrimOncePerCursor) {
    Scene scene;
    ChangeCursor cursor;
    scene.Define("/World/Sun", tokens::kDistantLight);
    std::vector<Change> changes = scene.Changes(cursor);
    ASSERT_EQ(changes.size(), 2u);
    EXPECT_EQ(changes[0].path, Path("/World"));
    EXPECT_EQ(changes[0].bits, Dirty::Added);
    EXPECT_EQ(changes[1].bits, Dirty::Added);
    EXPECT_EQ(cursor.version, scene.Version());
    EXPECT_TRUE(scene.Changes(cursor).empty());

    scene.Set("/World/Sun", tokens::kIntensity, 2.0f);
    scene.Set("/World/Sun", tokens::kMaterialBinding,
              Path("/Looks/Rock"));
    changes = scene.Changes(cursor);
    ASSERT_EQ(changes.size(), 1u);
    EXPECT_EQ(changes[0].bits, Dirty::Params | Dirty::Material);

    scene.Define("/World/Sun", tokens::kDistantLight);
    EXPECT_TRUE(scene.Changes(cursor).empty())
        << "redefining with the same type is a no-op";
}

TEST(Scene, AddedSubsumesOtherBitsForNewPrims) {
    Scene scene;
    ChangeCursor cursor;
    scene.Define("/Terrain", tokens::kTerrain)
        .Set(tokens::kCellSize, 5.0);
    EXPECT_EQ(BitsOf(scene.Changes(cursor), "/Terrain"),
              Dirty::Added);
}

TEST(Scene, TransformAndVisibilityEditsDirtyDescendants) {
    Scene scene;
    scene.Define("/World/Rover/Wheel", tokens::kMesh);
    scene.Define("/Worlds", tokens::kXform);
    ChangeCursor cursor;
    scene.Changes(cursor);

    scene.Set("/World", tokens::kTransform, glm::dmat4(2.0));
    const std::vector<Change> moved = scene.Changes(cursor);
    EXPECT_EQ(BitsOf(moved, "/World/Rover/Wheel"), Dirty::Transform);
    EXPECT_EQ(BitsOf(moved, "/World/Rover"), Dirty::Transform);
    EXPECT_EQ(BitsOf(moved, "/Worlds"), Dirty::None)
        << "a sibling, not a descendant";

    scene.Set("/World/Rover", tokens::kVisibility,
              tokens::kInvisible);
    scene.Write("/World/Rover/Wheel", tokens::kPoints,
                Tensor::Zeros(ernie::DType::Float32, {3, 3}));
    const std::vector<Change> hidden = scene.Changes(cursor);
    EXPECT_EQ(BitsOf(hidden, "/World/Rover/Wheel"),
              Dirty::Visibility | Dirty::Points);
    EXPECT_EQ(BitsOf(hidden, "/World"), Dirty::None);
}

TEST(Scene, DeclaredInheritanceAndRelationshipsReachDependents) {
    Scene scene;
    scene.Define("/World/Rover/Wheel", tokens::kMesh);
    scene.Define("/World/Rock", tokens::kMesh);
    scene.Define("/Looks/Grey", tokens::kMaterial);
    scene.Define("/Looks/Grey/Shader", tokens::kShader);
    scene.Set("/World/Rock", tokens::kMaterialBinding, Path("/Looks/Grey"));
    scene.Define("/World/Skin", tokens::kMesh)
        .Set(tokens::kSkelSkeleton, Path("/World/Skel"));
    scene.Define("/World/Skel", tokens::kSkeleton)
        .Set(tokens::kSkelAnimationSource, Path("/World/Anim"));
    scene.Define("/World/Anim", tokens::kSkelAnimation);
    ChangeCursor cursor;
    scene.Changes(cursor);

    // An inherited attribute (ernie:world) stamps the descendants.
    scene.Set("/World/Rover", tokens::kWorld, 3);
    const std::vector<Change> world = scene.Changes(cursor);
    EXPECT_EQ(BitsOf(world, "/World/Rover/Wheel"), Dirty::Params);
    EXPECT_EQ(BitsOf(world, "/World/Rock"), Dirty::None);

    // A change under a material reaches the prims bound to it.
    scene.Set("/Looks/Grey/Shader", "inputs:roughness", 0.5);
    const std::vector<Change> shaded = scene.Changes(cursor);
    EXPECT_EQ(BitsOf(shaded, "/World/Rock"), Dirty::Material);
    EXPECT_EQ(BitsOf(shaded, "/World/Rover/Wheel"), Dirty::None);

    // And on through a chain: the animation, its skeleton, the mesh.
    scene.Set("/World/Anim", "user:speed", 2.0);
    const std::vector<Change> posed = scene.Changes(cursor);
    EXPECT_EQ(BitsOf(posed, "/World/Skel"), Dirty::Points);
    EXPECT_EQ(BitsOf(posed, "/World/Skin"), Dirty::Points);

    // Rebinding moves what reaches it; removing the target reaches
    // those still bound.
    scene.Set("/World/Rock", tokens::kMaterialBinding, Path("/Looks/Red"));
    scene.Changes(cursor);
    scene.Set("/Looks/Grey", "user:note", 1.0);
    EXPECT_EQ(BitsOf(scene.Changes(cursor), "/World/Rock"), Dirty::None);
    scene.Set("/World/Rover/Wheel", tokens::kMaterialBinding,
              Path("/Looks/Grey"));
    scene.Changes(cursor);
    scene.Remove("/Looks/Grey");
    EXPECT_EQ(BitsOf(scene.Changes(cursor), "/World/Rover/Wheel"),
              Dirty::Material);
}

TEST(Scene, ColumnsWriteManyPrimsWhole) {
    Scene scene;
    for (const char* path : {"/A", "/B", "/C"}) {
        scene.Define(path, tokens::kXform);
        ernie::SetXform(scene, path, {.translate = {0, 0, 0}});
    }
    scene.Define("/A/Mesh", tokens::kMesh);
    const ernie::PrimSet bodies({"/A", "/B", "/C"});
    const auto rows = [](float x) {
        const std::array<float, 9> values{x, 0, 0, x + 1, 0, 0,
                                          x + 2, 0, 0};
        return Tensor::Copy<float>(values, {3, 3});
    };
    ChangeCursor cursor;
    scene.Write(bodies, tokens::kTranslate, rows(1));
    scene.Changes(cursor);

    // Written again: whole, its rows read as they are now.
    scene.Write(bodies, tokens::kTranslate, rows(10));
    EXPECT_FLOAT_EQ(scene.Read("/B", tokens::kTranslate)->As<float>()[0],
                    11.0f);
    EXPECT_DOUBLE_EQ(scene.WorldTransform("/A/Mesh")[3][0], 10.0);
    EXPECT_EQ(scene.AttributeVersion("/C", tokens::kTranslate),
              scene.Version());
    const std::vector<Change> expanded = scene.Changes(cursor);
    EXPECT_EQ(BitsOf(expanded, "/B"), Dirty::Transform);
    EXPECT_EQ(BitsOf(expanded, "/A/Mesh"), Dirty::Transform);

    // To a consumer taking columns whole: one change, expandable.
    scene.Write(bodies, tokens::kTranslate, rows(20));
    std::vector<ernie::ColumnChange> columns;
    const std::vector<Change> rest = scene.Changes(cursor, columns);
    EXPECT_TRUE(rest.empty());
    ASSERT_EQ(columns.size(), 1u);
    EXPECT_EQ(columns[0].paths->size(), 3u);
    EXPECT_EQ(columns[0].bits, Dirty::Transform);
    EXPECT_EQ(BitsOf(scene.ExpandColumns(rest, columns), "/A/Mesh"),
              Dirty::Transform);
    scene.Unmap(bodies, tokens::kTranslate);
    columns.clear();
    scene.Changes(cursor, columns);
    EXPECT_EQ(columns.size(), 1u) << "unmapped whole";

    // A copy keeps its own rows.
    const Scene copy = scene;
    scene.Write(bodies, tokens::kTranslate, rows(30));
    EXPECT_FLOAT_EQ(copy.Read("/A", tokens::kTranslate)->As<float>()[0],
                    20.0f);

    // One prim written alone ends the column, its rows kept and none
    // of its changes lost; the next write binds anew.
    scene.Set("/B", tokens::kTranslate, glm::dvec3(5.0, 0.0, 0.0));
    EXPECT_FLOAT_EQ(scene.Read("/A", tokens::kTranslate)->As<float>()[0],
                    30.0f);
    columns.clear();
    const std::vector<Change> broken = scene.Changes(cursor, columns);
    EXPECT_TRUE(columns.empty());
    EXPECT_EQ(BitsOf(broken, "/C"), Dirty::Transform);
    scene.Write(bodies, tokens::kTranslate, rows(40));
    scene.Write(bodies, tokens::kTranslate, rows(50));
    EXPECT_FLOAT_EQ(scene.Read("/B", tokens::kTranslate)->As<float>()[0],
                    51.0f);
}

TEST(Scene, ColumnHandlesWriteAndMapOnAnyDevice) {
    Scene scene;
    for (const char* path : {"/A", "/B"}) {
        scene.Define(path, tokens::kXform);
    }
    ernie::SetXform(scene, "/A", {.translate = {1, 2, 3}});
    ernie::SetXform(scene, "/B", {.translate = {4, 5, 6}});
    ernie::Column poses =
        scene.GetColumn(ernie::PrimSet({"/A", "/B"}), tokens::kTranslate);
    // Read before written: the prims' values, gathered.
    const Tensor before = poses.Read();
    ASSERT_EQ(before.Shape(), (std::vector<std::int64_t>{2, 3}));
    EXPECT_FLOAT_EQ(before.As<float>()[4], 5.0f);

    // Mapped on the host: gathered, bound, written in place.
    ChangeCursor cursor;
    const ernie::Device cpu{};
    Tensor rows = poses.Map(cpu);
    scene.Changes(cursor);
    rows.As<float>()[0] = 10.0f;
    poses.Unmap();
    std::vector<ernie::ColumnChange> columns;
    EXPECT_TRUE(scene.Changes(cursor, columns).empty());
    EXPECT_EQ(columns.size(), 1u);
    EXPECT_FLOAT_EQ(scene.Read("/A", tokens::kTranslate)->As<float>()[0],
                    10.0f);
    EXPECT_EQ(poses.Map(cpu).Data(), rows.Data()) << "mapped again";
    // Nothing holds them on a GPU.
    EXPECT_THROW(poses.Map({ernie::DeviceKind::Cuda, 0}),
                 std::invalid_argument);

    // A consumer's storage, used once offered and asked for no more.
    struct Storage : ernie::ColumnStorage {
        int asked = 0;
        Tensor rows = Tensor::Zeros(ernie::DType::Float32, {2, 3});
        std::optional<Rows> Allocate(Scene&, const ernie::PrimSet&,
                                     std::string_view, ernie::Device,
                                     std::uintptr_t) override {
            ++asked;
            return Rows{rows};
        }
    };
    const auto storage = std::make_shared<Storage>();
    scene.SetColumnStorage(storage);
    EXPECT_EQ(poses.Map(cpu).Data(), storage->rows.Data());
    EXPECT_EQ(poses.Map(cpu).Data(), storage->rows.Data());
    EXPECT_EQ(storage->asked, 1);
    EXPECT_EQ(poses.Read().Data(), storage->rows.Data());
    // Rows written take over from it; a copy of the scene has none.
    const Tensor own = Tensor::Zeros(ernie::DType::Float32, {2, 3});
    poses.Write(own);
    EXPECT_EQ(poses.Read().Data(), own.Data());
    Scene copy = scene;
    EXPECT_THROW(copy.GetColumn(ernie::PrimSet({"/A", "/B"}),
                                tokens::kTranslate)
                     .Map({ernie::DeviceKind::Cuda, 0}),
                 std::invalid_argument);
}

TEST(Scene, ColumnsOfAnyNumberReadAsTheirPrimsValues) {
    Scene scene;
    for (const char* path : {"/A", "/A/Child", "/B"}) {
        scene.Define(path, tokens::kCube);
    }
    const ernie::PrimSet cubes({"/A", "/B"});
    // A number, a vector and visibility, each a column.
    const std::array<float, 2> kelvin{300.0f, 450.0f};
    scene.GetColumn(cubes, tokens::kThermalTemperature)
        .Write(Tensor::Copy<float>(kelvin, {2}));
    const std::array<float, 6> colours{1, 0, 0, 0, 0, 1};
    scene.GetColumn(cubes, tokens::kDisplayColor)
        .Write(Tensor::Copy<float>(colours, {2, 3}));
    const std::array<float, 2> shown{0.0f, 1.0f};
    scene.GetColumn(cubes, tokens::kVisibility)
        .Write(Tensor::Copy<float>(shown, {2}));
    EXPECT_DOUBLE_EQ(ernie::GetNumber(scene, "/B",
                                      tokens::kThermalTemperature, 0.0),
                     450.0);
    EXPECT_EQ(ernie::GetVec3(scene, "/B", tokens::kDisplayColor,
                             glm::dvec3(0.5)),
              glm::dvec3(0.0, 0.0, 1.0));
    EXPECT_FALSE(scene.IsVisible("/A"));
    EXPECT_FALSE(scene.IsVisible("/A/Child")) << "hiding its descendants";
    EXPECT_TRUE(scene.IsVisible("/B"));
    // Written again whole, read anew.
    const std::array<float, 2> swapped{1.0f, 0.0f};
    scene.GetColumn(cubes, tokens::kVisibility)
        .Write(Tensor::Copy<float>(swapped, {2}));
    EXPECT_TRUE(scene.IsVisible("/A/Child"));
    EXPECT_FALSE(scene.IsVisible("/B"));
}

TEST(Scene, RemovalsAndRetypesAreReported) {
    Scene scene;
    scene.Define("/World/Rover/Wheel", tokens::kMesh);
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kIntensity, 2.0f);
    ChangeCursor cursor;
    scene.Changes(cursor);

    EXPECT_TRUE(scene.Remove("/World/Rover"));
    EXPECT_FALSE(scene.Remove("/World/Rover"));
    EXPECT_FALSE(scene.Has("/World/Rover/Wheel"));
    EXPECT_THROW(scene.Remove(Path::Root()), std::invalid_argument);
    std::vector<Change> changes = scene.Changes(cursor);
    ASSERT_EQ(changes.size(), 2u);
    EXPECT_EQ(changes[0].path, Path("/World/Rover"));
    EXPECT_EQ(changes[0].bits, Dirty::Removed);
    EXPECT_EQ(changes[1].bits, Dirty::Removed);

    scene.Define("/World/Sun", tokens::kSphereLight);
    EXPECT_EQ(scene.Get("/World/Sun", tokens::kIntensity), nullptr)
        << "attributes dropped";
    EXPECT_EQ(BitsOf(scene.Changes(cursor), "/World/Sun"),
              Dirty::Removed | Dirty::Added);

    scene.Define("/World/Rover", tokens::kXform);
    EXPECT_EQ(BitsOf(scene.Changes(cursor), "/World/Rover"),
              Dirty::Added);
}

TEST(Scene, CursorsAreIndependent) {
    Scene scene;
    ChangeCursor early;
    scene.Define("/A", tokens::kXform);
    ChangeCursor late;
    scene.Changes(late);
    scene.Set("/A", tokens::kTransform, glm::dmat4(1.0));
    EXPECT_EQ(BitsOf(scene.Changes(late), "/A"), Dirty::Transform);
    EXPECT_EQ(BitsOf(scene.Changes(early), "/A"), Dirty::Added);
}

TEST(Scene, IdsAreUniqueAndRenewedByCopies) {
    Scene a;
    Scene b;
    EXPECT_NE(a.Id(), b.Id());
    a.Define("/A", tokens::kXform);
    Scene copy = a;
    EXPECT_NE(copy.Id(), a.Id());
    EXPECT_TRUE(copy.Has("/A"));
    const std::uint64_t before = b.Id();
    b = a;
    EXPECT_NE(b.Id(), before);
    EXPECT_NE(b.Id(), a.Id());
    const std::uint64_t moved = copy.Id();
    const Scene target = std::move(copy);
    EXPECT_EQ(target.Id(), moved);
}

TEST(Scene, CloneCopiesASubtreeAndMovesItsInnerRelationships) {
    Scene scene;
    scene.Define("/Looks/Shared", "Material");
    scene.Define("/Env", "Xform").Set("ernie:world", 0);
    scene.Define("/Env/Rocks", "PointInstancer")
        .Write("positions", Tensor::Copy<float>(
                                std::vector<float>{1, 2, 3}, {1, 3}))
        .Set("prototypes", std::vector<Path>{"/Env/Rocks/Rock"});
    scene.Define("/Env/Rocks/Rock", "Mesh")
        .Set("material:binding", Path("/Looks/Shared"));

    scene.Clone("/Env", "/Envs/Env1");
    ASSERT_TRUE(scene.Has("/Envs/Env1/Rocks/Rock"));
    EXPECT_EQ(scene.TypeOf("/Envs/Env1/Rocks"), "PointInstancer");
    EXPECT_EQ(scene.Get<int>("/Envs/Env1", "ernie:world", -1), 0);
    EXPECT_EQ(scene.Get<std::vector<Path>>("/Envs/Env1/Rocks",
                                           "prototypes", {}),
              std::vector<Path>{"/Envs/Env1/Rocks/Rock"})
        << "relationships inside the source move with it";
    EXPECT_EQ(scene.Get<Path>("/Envs/Env1/Rocks/Rock",
                              "material:binding", Path()),
              Path("/Looks/Shared"))
        << "relationships outside stay";
    EXPECT_EQ(scene.Read("/Envs/Env1/Rocks", "positions")->Data(),
              scene.Read("/Env/Rocks", "positions")->Data())
        << "tensors are shared";

    EXPECT_THROW(scene.Clone("/Missing", "/Copy"),
                 std::invalid_argument);
    EXPECT_THROW(scene.Clone("/Env", "/Envs/Env1"),
                 std::invalid_argument)
        << "exists";
    EXPECT_THROW(scene.Clone("/Env", "/Env/Inner"),
                 std::invalid_argument)
        << "inside";
}

TEST(Scene, CloneMovesShaderConnectionsInsideTheSource) {
    // A material's network: the surface reads its own texture, which
    // reads coordinates from outside the material.
    Scene scene;
    scene.Define("/Looks/M", "Material")
        .Set("outputs:surface.connect",
             std::string("/Looks/M/Surface.outputs:surface"));
    scene.Define("/Looks/M/Surface", "Shader")
        .Set("inputs:diffuseColor.connect",
             std::string("/Looks/M/Tex.outputs:rgb"));
    scene.Define("/Looks/M/Tex", "Shader")
        .Set("inputs:st.connect",
             std::string("/Looks/St.outputs:result"));

    scene.Clone("/Looks/M", "/Looks/N");
    EXPECT_EQ(scene.Get<std::string>("/Looks/N",
                                     "outputs:surface.connect", ""),
              "/Looks/N/Surface.outputs:surface");
    EXPECT_EQ(scene.Get<std::string>("/Looks/N/Surface",
                                     "inputs:diffuseColor.connect",
                                     ""),
              "/Looks/N/Tex.outputs:rgb");
    EXPECT_EQ(scene.Get<std::string>("/Looks/N/Tex",
                                     "inputs:st.connect", ""),
              "/Looks/St.outputs:result")
        << "connections outside stay";
}

TEST(Scene, MappedArraysAreWrittenInPlaceAndCommittedByUnmap) {
    Scene scene;
    scene.Define("/Rocks", "PointInstancer")
        .Write("positions", Tensor::Copy<float>(
                                std::vector<float>{1, 2, 3}, {1, 3}));
    EXPECT_EQ(scene.WriteStream("/Rocks", "positions"), 1u)
        << "the legacy stream by default";
    ChangeCursor cursor;
    scene.Changes(cursor);
    Tensor positions = scene.Map("/Rocks", "positions");
    positions.As<float>()[0] = 7.0f;
    EXPECT_EQ(scene.Read("/Rocks", "positions")->As<float>()[0], 7.0f)
        << "the same memory";
    EXPECT_TRUE(scene.Changes(cursor).empty())
        << "unseen until unmapped";
    scene.Unmap("/Rocks", "positions", 0x1234);
    const std::vector<Change> changes = scene.Changes(cursor);
    ASSERT_EQ(changes.size(), 1u);
    EXPECT_EQ(changes[0].path, Path("/Rocks"));
    EXPECT_EQ(scene.WriteStream("/Rocks", "positions"), 0x1234u);
    EXPECT_THROW(scene.Map("/Rocks", "missing"),
                 std::invalid_argument);
    scene.Set("/Rocks", "ernie:world", 1);
    EXPECT_THROW(scene.Map("/Rocks", "ernie:world"),
                 std::invalid_argument)
        << "a value";
    EXPECT_THROW(scene.Unmap("/Rocks", "missing"),
                 std::invalid_argument);
}
