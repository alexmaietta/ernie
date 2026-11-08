// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/primvar.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/tangents.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie {
namespace {

std::vector<std::string> Paths(std::string_view prefix, uint32_t count) {
    std::vector<std::string> paths;
    for (uint32_t i = 0; i < count; ++i) {
        paths.push_back(fmt::format("{}{}", prefix, i));
    }
    return paths;
}

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        const Ref<Scene> scene = Scene::Create(device);
        test(*scene);
    });
}

TEST(SceneTest, PrimsAreDefinedFoundAndRemoved) {
    ForEachScene([](Scene& scene) {
        const Prim robot = scene.Define<Xform>("/Robot");
        const Prim ball = scene.Define<Sphere>("/Robot/Ball", {.radius = 2});
        EXPECT_TRUE(ball.Valid());
        EXPECT_EQ(ball.Path(), "/Robot/Ball");
        EXPECT_EQ(ball.Name(), "Ball");
        EXPECT_EQ(ball.Schema(), SchemaId::Sphere);
        EXPECT_TRUE(ball.IsA<Gprim>());
        EXPECT_EQ(ball.Parent(), robot);
        EXPECT_EQ(robot.Children(), std::vector<Prim>{ball});
        EXPECT_EQ(scene.Find("/Robot/Ball"), ball);
        EXPECT_EQ(scene.Select<Sphere>("/Robot/*").Get<float>(kSphereRadius).Get(),
                  std::vector<float>{2});
        scene.Remove(robot);
        EXPECT_FALSE(ball.Valid());
        EXPECT_FALSE(scene.Find("/Robot"));
        EXPECT_EQ(scene.PrimCount(), 0u);
    });
}

TEST(SceneTest, StructureVersionsChangeWithPrimsNotValues) {
    ForEachScene([](Scene& scene) {
        uint64_t version = scene.StructureVersion();
        const auto changed = [&] {
            const uint64_t now = scene.StructureVersion();
            const bool moved = now != version;
            version = now;
            return moved;
        };
        const Prim robot = scene.Define<Xform>("/Robot");
        EXPECT_TRUE(changed());
        scene.Define<Sphere>("/Robot/Ball");
        EXPECT_TRUE(changed());
        scene.Select<Sphere>("/Robot/*").Get<float>(kSphereRadius).Set(3.0f);
        EXPECT_FALSE(changed());
        scene.Clone(robot, std::vector<std::string>{"/Copy"});
        EXPECT_TRUE(changed());
        scene.Remove(robot);
        EXPECT_TRUE(changed());
    });
}

TEST(SceneTest, ABatchIsOneDenseView) {
    ForEachScene([](Scene& scene) {
        const std::vector<std::string> paths = Paths("/Ball", 500);
        const PrimRange balls = scene.Define<Sphere>(paths, {.radius = 0.5f});
        EXPECT_EQ(balls.Size(), 500u);
        const View view = scene.Select(balls, SchemaId::Sphere);
        EXPECT_TRUE(view.Dense(SchemaId::Sphere));
        EXPECT_EQ(view.Get<float>(kSphereRadius).Get(), std::vector<float>(500, 0.5f));
        EXPECT_EQ(scene.Select("/Ball*").PrimCount(), 500u);
        const PrimRange plain = scene.Define(Paths("/Box", 3), SchemaId::Cube);
        EXPECT_EQ(scene.ByIndex(plain.Index(2, 0)).Path(), "/Box2");
    });
}

TEST(SceneTest, ClonesFillNewWorlds) {
    ForEachScene([](Scene& scene) {
        const Prim env = scene.Define<World>("/Env0");
        scene.Define<Sphere>("/Env0/Ball");
        const PrimRange copies = scene.Clone(env, Paths("/Copy", 3));
        EXPECT_EQ(copies.copies, 3u);
        EXPECT_EQ(copies.count, 2u);
        EXPECT_EQ(copies.firstWorld, 1u);
        EXPECT_EQ(scene.WorldCount(), 4u);
        EXPECT_EQ(scene.ByIndex(copies.Index(2, 1)).World(), 3u);
    });
}

TEST(SceneTest, SnapshotsCarryEverySystemsOutput) {
    ForEachScene([](Scene& scene) {
        scene.Define<World>("/Env0");
        scene.Define<Sphere>("/Env0/Ball");
        const ernie::Token taxonomy = scene.Token("class");
        scene.Require(std::vector<ernie::Token>{taxonomy});
        EXPECT_THROW(scene.Acquire(), Error);
        scene.Publish().Wait();
        EXPECT_EQ(scene.Published(), 1u);
        const Snapshot snapshot = scene.Acquire(Stream::Host());
        EXPECT_EQ(snapshot.ordinal, 1u);
        EXPECT_NE(snapshot.tables, 0u);
        const PinnedTables& pinned = PinnedTables::From(snapshot);
        EXPECT_EQ(snapshot.tables, pinned.buffer->Address());
        EXPECT_NE(pinned.Header<RenderTables>().instances.address, 0u);
        EXPECT_NE(pinned.Header<RenderTables>().lights.address, 0u);
        EXPECT_NE(pinned.Header<RenderTables>().origins.address, 0u);
        EXPECT_EQ(pinned.Header<RenderTables>().semantics.taxonomyCount, 1u);
        EXPECT_EQ(pinned.Header<RenderTables>().worldCount, 1u);
        EXPECT_TRUE(snapshot.ready.Ready());
        scene.Release(snapshot);
        scene.Publish({.ordinal = 10}).Wait();
        EXPECT_EQ(scene.Published(), 10u);
    });
}

TEST(SceneTest, PosesMoveWhatTheSceneRenders) {
    ForEachScene([](Scene& scene) {
        const PrimRange balls = scene.Define<Sphere>(Paths("/Ball", 2));
        scene.Publish().Wait();
        std::vector<float> poses = {1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 0, 0, 0, 1};
        int64_t shape[] = {2, 7};
        DLTensor tensor{};
        tensor.data = poses.data();
        tensor.device = {kDLCPU, 0};
        tensor.ndim = 2;
        tensor.dtype = {kDLFloat, 32, 1};
        tensor.shape = shape;
        const View view = scene.Select(balls, SchemaId::Sphere);
        view.WritePoses(tensor);
        scene.Publish().Wait();
        const std::vector<WorldTransform> worlds = scene.WorldTransforms(Paths("/Ball", 2));
        EXPECT_FLOAT_EQ(worlds[1].high.x + worlds[1].low.x, 4.0f);
        EXPECT_FLOAT_EQ(worlds[0].high.z + worlds[0].low.z, 3.0f);
    });
}

TEST(SceneTest, AttachedTranslatesMoveWhatTheSceneRenders) {
    ForEachScene([](Scene& scene) {
        const PrimRange balls = scene.Define<Sphere>(Paths("/Ball", 2), {}, {.attachable = true});
        scene.Publish().Wait();
        struct Owned {
            DLManagedTensor managed{};
            std::vector<double> values = {1, 2, 3, 4, 5, 6};
            int64_t shape[2] = {2, 3};
        };
        Owned* owned = new Owned;
        owned->managed.manager_ctx = owned;
        owned->managed.dl_tensor = {.data = owned->values.data(),
                                    .device = {kDLCPU, 0},
                                    .ndim = 2,
                                    .dtype = {kDLFloat, 64, 1},
                                    .shape = owned->shape};
        owned->managed.deleter = [](DLManagedTensor* managed) {
            delete static_cast<Owned*>(managed->manager_ctx);
        };
        const View view = scene.Select(balls, SchemaId::Sphere);
        UntypedAttribute translate = view.Get(kXformableTranslate);
        translate.Attach(&owned->managed);
        scene.Publish().Wait();
        std::vector<WorldTransform> worlds = scene.WorldTransforms(Paths("/Ball", 2));
        EXPECT_FLOAT_EQ(worlds[1].high.y + worlds[1].low.y, 5.0f);

        owned->values[4] = 50;
        translate.MarkChanged();
        scene.Publish().Wait();
        worlds = scene.WorldTransforms(Paths("/Ball", 2));
        EXPECT_FLOAT_EQ(worlds[1].high.y + worlds[1].low.y, 50.0f);
        translate.Detach();
    });
}

TEST(SceneTest, APrimGetsAndSetsItsOwnAttributes) {
    ForEachScene([](Scene& scene) {
        const Prim ball = scene.Define<Sphere>("/Ball", {.radius = 2});
        EXPECT_EQ(ball.Get(Sphere::kRadius), 2.0f);
        ball.Set(Sphere::kRadius, 3.0f);
        EXPECT_EQ(ball.Get(Sphere::kRadius), 3.0f);
        EXPECT_EQ(ball.Get<float>("radius"), 3.0f);
        EXPECT_THROW(ball.Get<int32_t>("radius"), Error);

        const Prim mesh = scene.Define<Mesh>("/Mesh");
        const std::vector<Float3> points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        mesh.Set(Mesh::kPoints, points);
        EXPECT_EQ(mesh.Get(Mesh::kPoints), points);
        EXPECT_THROW(ball.Get(Mesh::kPoints), Error);
        scene.Remove(ball);
        EXPECT_THROW(ball.Get(Sphere::kRadius), Error);
    });
}

TEST(SceneTest, TypedPrimsReadAndWriteEveryKindOfField) {
    ForEachScene([](Scene& scene) {
        const SpherePrim ball = scene.Define<Sphere>("/Ball", {.radius = 2});
        EXPECT_EQ(ball.Radius(), 2.0f);
        ball.SetRadius(3);
        EXPECT_EQ(ball.Get(Sphere::kRadius), 3.0f);
        ball.Set(Sphere::kRadius, 4.0);
        EXPECT_EQ(ball.Radius(), 4.0f);
        ball.SetTranslate({1, 2, 3});
        EXPECT_EQ(ball.Translate(), (Double3{1, 2, 3}));
        ball.SetVisibility(Visibility::Invisible);
        EXPECT_EQ(ball.Visibility(), Visibility::Invisible);
        ball.SetDoubleSided(true);
        EXPECT_TRUE(ball.DoubleSided());
        scene.Apply<SemanticsLabelsAPI>(ball, "class", {.semanticsLabels = {"rock"}});
        EXPECT_EQ(ball.Instances<SemanticsLabelsAPI>(), std::vector<std::string>{"class"});
        EXPECT_EQ(ball.Get(SemanticsLabelsAPI::kSemanticsLabels, "class"),
                  std::vector<std::string>{"rock"});
        ball.Set(SemanticsLabelsAPI::kSemanticsLabels, "class", {"boulder"});
        EXPECT_EQ(ball.Get(SemanticsLabelsAPI::kSemanticsLabels, "class"),
                  std::vector<std::string>{"boulder"});

        const UsdPreviewSurfacePrim look = scene.Define<UsdPreviewSurface>("/Look");
        EXPECT_EQ(look.Id(), "UsdPreviewSurface");
        const UsdUVTexturePrim albedo = scene.Define<UsdUVTexture>("/Albedo");
        albedo.SetFile("albedo.png");
        EXPECT_EQ(albedo.File(), "albedo.png");

        const MeshPrim mesh = scene.Define<Mesh>("/Mesh");
        const std::vector<Float3> points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        mesh.SetPoints(points);
        EXPECT_EQ(mesh.Points(), points);
        mesh.SetFaceVertexIndices({0, 1, 2});
        EXPECT_EQ(mesh.FaceVertexIndices(), (std::vector<int32_t>{0, 1, 2}));
        mesh.SetMaterialBinding({"/Look"});
        EXPECT_EQ(mesh.MaterialBinding(), std::vector<std::string>{"/Look"});
        mesh.SetPrimvars({Primvar::Create("heat", Interpolation::Vertex, points)});
        ASSERT_EQ(mesh.Primvars().size(), 1u);
        EXPECT_EQ(mesh.Primvars()[0].name, "heat");

        scene.Publish().Wait();
        const Box box = scene.PrimBounds(std::vector<std::string>{ball.Path()}).front();
        EXPECT_NEAR(box.lower.x, -3.0f, 1e-4f);
        EXPECT_NEAR(box.upper.z, 7.0f, 1e-4f);

        EXPECT_NO_THROW(XformablePrim{ball});
        EXPECT_NO_THROW(MaterialBindingAPIPrim{ball});
        EXPECT_THROW(MeshPrim{ball}, Error);
        EXPECT_EQ(GprimPrim(ball).DoubleSided(), true);
    });
}

TEST(SceneTest, TypedViewsHandOutTypedAttributes) {
    ForEachScene([](Scene& scene) {
        scene.Define<Sphere>(Paths("/Ball", 3), {.radius = 1});
        const SphereView balls = scene.Select<Sphere>("/Ball*");
        EXPECT_EQ(balls.PrimCount(), 3u);
        balls.Radius().Set(std::vector<float>{1, 2, 3});
        EXPECT_EQ(balls.Radius().Get(), (std::vector<float>{1, 2, 3}));
        EXPECT_EQ(balls.Get(Sphere::kRadius).Get(), (std::vector<float>{1, 2, 3}));
        balls.Visibility().Set(Visibility::Invisible);
        EXPECT_EQ(balls.Visibility().Get(), std::vector<Visibility>(3, Visibility::Invisible));

        scene.Define<Mesh>("/Mesh", {.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}});
        const MeshView meshes = scene.Select<Mesh>("/Mesh");
        EXPECT_EQ(meshes.Points().Get(0).size(), 3u);
        EXPECT_TRUE(meshes.MaterialBinding().GetTargets(0).empty());
    });
}

TEST(SceneTest, MarkChangedingAttachedPointsRefitsTheirBlas) {
    ForEachScene([](Scene& scene) {
        const std::vector<Float3> quad = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        const PrimRange quads = scene.Define<Mesh>(
            Paths("/Quad", 2),
            {.points = quad, .faceVertexIndices = {0, 1, 2, 3}, .faceVertexCounts = {4}});
        scene.Publish().Wait();
        struct Owned {
            DLManagedTensor managed{};
            std::vector<float> values;
            int64_t shape[3] = {2, 4, 3};
        };
        Owned* owned = new Owned;
        for (uint32_t i = 0; i < 2; ++i) {
            for (const Float3& point : quad) {
                owned->values.insert(owned->values.end(), {point.x, point.y, point.z});
            }
        }
        owned->managed.manager_ctx = owned;
        owned->managed.dl_tensor = {.data = owned->values.data(),
                                    .device = {kDLCPU, 0},
                                    .ndim = 3,
                                    .dtype = {kDLFloat, 32, 1},
                                    .shape = owned->shape};
        owned->managed.deleter = [](DLManagedTensor* managed) {
            delete static_cast<Owned*>(managed->manager_ctx);
        };
        UntypedAttribute points = scene.Select(quads, SchemaId::Mesh).Get(kPointBasedPoints);
        points.Attach(&owned->managed);
        scene.Publish().Wait();
        const GeometryStats& stats = SceneImpl::Get(scene).GetMeshes().Stats();
        EXPECT_EQ(stats.built, 2u);

        owned->values[2] = 1;
        points.MarkChanged();
        scene.Publish().Wait();
        EXPECT_EQ(stats.built, 0u);
        EXPECT_EQ(stats.refit, 2u);
        points.Detach();
    });
}

TEST(SceneTest, TargetsTokensAndPrimvarsAreSetByValue) {
    ForEachScene([](Scene& scene) {
        const Prim mesh = scene.Define<Mesh>("/Mesh");
        const View view = scene.Select(std::vector<std::string>{"/Mesh"});
        UntypedAttribute binding = view.Get("materialBinding");
        binding.SetTargets(0, {"/Later"});
        EXPECT_TRUE(binding.GetTargets(0).empty());
        scene.Define<Material>("/Later");
        EXPECT_EQ(binding.GetTargets(0), std::vector<std::string>{"/Later"});

        const std::vector<Prim> meshes = scene.Handles({view.Prims()[0], 1, 1, kGlobalWorld, 0});
        scene.Apply(meshes, SchemaId::SemanticsLabelsAPI, "class");
        const View labels = scene.Select("/Mesh", SchemaId::SemanticsLabelsAPI, "class");
        labels.Get("semanticsLabels").SetTokens(0, {"box"});
        EXPECT_EQ(labels.Get("semanticsLabels").GetTokens(0), std::vector<std::string>{"box"});

        const std::vector<float> heat = {1, 2, 3};
        const std::vector<Primvar> primvars = {
            Primvar::Create("heat", Interpolation::Vertex, heat)};
        view.Get("primvars").SetPrimvars(0, primvars);
        const std::vector<Primvar> read = view.Get("primvars").GetPrimvars(0);
        ASSERT_EQ(read.size(), 1u);
        EXPECT_EQ(read[0].name, "heat");
        EXPECT_EQ(read[0].Values<float>(), heat);
        EXPECT_THROW(view.Get("points").SetTokens(0, {"a"}), Error);
        EXPECT_TRUE(mesh.Valid());
    });
}

TEST(SceneTest, DeformedMeshesRecomputeTangentsOnTheDevice) {
    ForEachScene([](Scene& scene) {
        constexpr uint32_t kSide = 5;
        std::vector<Float3> points;
        std::vector<Float2> st;
        for (uint32_t y = 0; y < kSide; ++y) {
            for (uint32_t x = 0; x < kSide; ++x) {
                points.push_back({float(x), float(y), 0});
                st.push_back({float(x) / (kSide - 1), float(y) / (kSide - 1)});
            }
        }
        std::vector<int32_t> indices;
        std::vector<int32_t> counts;
        for (uint32_t y = 0; y + 1 < kSide; ++y) {
            for (uint32_t x = 0; x + 1 < kSide; ++x) {
                const int32_t a = int32_t(y * kSide + x);
                indices.insert(indices.end(),
                               {a, a + 1, a + int32_t(kSide) + 1, a + int32_t(kSide)});
                counts.push_back(4);
            }
        }
        const Prim grid = scene.Define<Mesh>(
            "/Grid", {.primvars = {Primvar::Create("st", Interpolation::Vertex, st)},
                      .points = points,
                      .faceVertexIndices = indices,
                      .faceVertexCounts = counts,
                      .subdivisionScheme = SubdivisionScheme::None});
        scene.Publish().Wait();
        const View view = scene.Select(grid);
        view.Get("points").Map(MapAccess::Write).Close();
        scene.Publish().Wait();

        for (Float3& point : points) {
            point.z = 0.4f * std::sin(point.x) + 0.3f * std::cos(0.7f * point.y);
        }
        Mapping mapping = view.Get("points").Map(MapAccess::Write);
        const device::Tensor& values = mapping.Values();
        scene.GetDevice().GetQueue().Write(
            values.GetBuffer().Slice(values.Offset(), points.size() * sizeof(Float3)),
            points.data());
        mapping.Close();
        scene.Publish().Wait();
        const TangentStats& stats = SceneImpl::Get(scene).GetTangents().Stats();
        EXPECT_EQ(stats.deformed, 1u);
        EXPECT_EQ(stats.computed, 0u);

        TangentMesh deformed;
        deformed.points = points;
        deformed.indices = indices;
        deformed.counts = counts;
        deformed.normals = SmoothCornerNormals(points, indices, counts);
        deformed.st = *CornerValues<Float2>(st, {}, Interpolation::Vertex, indices, counts);
        const std::vector<Float4> expected = MikkTangents(deformed);
        SceneImpl& impl = SceneImpl::Get(scene);
        const uint32_t meshRow = impl.GetStructure().GetPrims().Row(grid.Index(), SchemaId::Mesh);
        const std::vector<Float4> found =
            Items<Float4>(impl.GetColumnStore()
                              .Arrays(ColumnType::IntArray)
                              .Read(impl.GetTangents().Record(meshRow).tangents));
        ASSERT_EQ(found.size(), expected.size());
        for (uint32_t corner = 0; corner < found.size(); ++corner) {
            const float agree = found[corner].x * expected[corner].x +
                                found[corner].y * expected[corner].y +
                                found[corner].z * expected[corner].z;
            EXPECT_GT(agree, 0.995f) << corner;
            EXPECT_EQ(found[corner].w, expected[corner].w) << corner;
        }
    });
}

TEST(SceneTest, BadClonesChangeNothing) {
    ForEachScene([](Scene& scene) {
        scene.Define<World>("/env_0");
        scene.Define<Xform>("/env_0/Robot");
        scene.Define<Sphere>("/env_0/Robot/Ball");
        scene.Define<World>("/other");
        const Prim source = *scene.Find("/env_0");
        const uint32_t prims = scene.PrimCount();
        const auto clone = [&](std::vector<std::string> targets) { scene.Clone(source, targets); };
        EXPECT_THROW(clone({"/env_0"}), Error);
        EXPECT_THROW(clone({"/env_0/Inner"}), Error);
        EXPECT_THROW(clone({"/missing/env"}), Error);
        EXPECT_THROW(clone({"/a", "/a"}), Error);
        EXPECT_THROW(clone({"/b", "/other/nested"}), Error);
        EXPECT_THROW(scene.Define<World>("/other/inner"), Error);
        EXPECT_EQ(scene.PrimCount(), prims);
        EXPECT_FALSE(scene.Find("/b"));
        EXPECT_EQ(scene.WorldCount(), 2u);

        scene.Define<Xform>("/group");
        scene.Clone(*scene.Find("/env_0/Robot"), std::vector<std::string>{"/group/Robot"});
        EXPECT_EQ(scene.Find("/group/Robot/Ball")->World(), kGlobalWorld);
        scene.Clone(*scene.Find("/env_0/Robot"), std::vector<std::string>{"/other/Robot"});
        EXPECT_EQ(scene.Find("/other/Robot/Ball")->World(), 1u);
    });
}

TEST(SceneTest, TargetsInsideTheSourceFollowEachCopy) {
    ForEachScene([](Scene& scene) {
        scene.Define<Camera>("/Overview");
        scene.Define<World>("/env_0");
        scene.Define<Camera>("/env_0/Cam", {.clippingRange = {0.1f, 50}});
        scene.Define<RenderVar>("/env_0/Color");
        scene.Define<RenderProduct>("/env_0/Product", {.camera = {"/env_0/Cam", "/Overview"},
                                                       .orderedVars = {"/env_0/Color"}});
        scene.Clone(*scene.Find("/env_0"), std::vector<std::string>{"/env_1", "/env_2"});
        const View products = scene.Select("/env_*/Product");
        for (uint32_t env = 0; env < 3; ++env) {
            const std::string root = "/env_" + std::to_string(env);
            EXPECT_EQ(products.Get("camera").GetTargets(env),
                      (std::vector<std::string>{root + "/Cam", "/Overview"}));
        }
        EXPECT_EQ(scene.Select("/env_*/Cam").Get<Float2>("clippingRange").Get(),
                  (std::vector<Float2>(3, Float2{0.1f, 50})));
    });
}

TEST(SceneTest, OverridesWinUntilClearedAndKeepAuthoredWrites) {
    ForEachScene([](Scene& scene) {
        const Prim ball = scene.Define<Sphere>("/Ball", {.radius = 1});
        const auto radius = [&] { return scene.Select(ball).Get(Sphere::kRadius).Get().front(); };
        const auto authored = [&] {
            float value = 0;
            const std::vector<std::byte> bytes = scene.ReadAuthored(ball, Sphere::kRadius.Column());
            std::memcpy(&value, bytes.data(), sizeof(value));
            return value;
        };
        Overrides weaker = scene.CreateOverrides();
        Overrides stronger = scene.CreateOverrides();
        weaker.Set(ball, Sphere::kRadius, 2.0f);
        EXPECT_EQ(radius(), 2.0f);
        EXPECT_TRUE(scene.Overridden(ball, Sphere::kRadius.Column()));
        stronger.Set(ball, Sphere::kRadius, 3.0f);
        EXPECT_EQ(radius(), 3.0f);
        scene.Select(ball).Get(Sphere::kRadius).Set(4.0f);
        EXPECT_EQ(radius(), 3.0f);
        EXPECT_EQ(authored(), 4.0f);
        std::vector<float> written = {7.0f};
        int64_t shape[] = {1};
        DLTensor host{};
        host.data = written.data();
        host.device = {kDLCPU, 0};
        host.ndim = 1;
        host.dtype = {kDLFloat, 32, 1};
        host.shape = shape;
        scene.Select(ball).Get(Sphere::kRadius).Write(scene.GetDevice().Import(host)).Wait();
        EXPECT_EQ(radius(), 3.0f);
        EXPECT_EQ(authored(), 7.0f);
        stronger.Clear();
        EXPECT_EQ(radius(), 2.0f);
        {
            Overrides moved = std::move(weaker);
            EXPECT_TRUE(weaker.Empty());
        }
        EXPECT_EQ(radius(), 7.0f);
        EXPECT_FALSE(scene.Overridden(ball, Sphere::kRadius.Column()));

        Overrides gone = scene.CreateOverrides();
        gone.Set(ball, Sphere::kRadius, 5.0f);
        scene.Remove(ball);
        const Prim again = scene.Define<Sphere>("/Ball", {.radius = 6});
        EXPECT_FALSE(scene.Overridden(again, Sphere::kRadius.Column()));
        gone.Clear();
        EXPECT_EQ(scene.Select(again).Get(Sphere::kRadius).Get().front(), 6.0f);
    });
}

TEST(SceneTest, SessionPrimsMarkTheirSubtree) {
    ForEachScene([](Scene& scene) {
        const Prim root = scene.Define<Scope>("/__session");
        const Prim child = scene.Define<Camera>("/__session/Perspective");
        const Prim user = scene.Define<Camera>("/Camera");
        EXPECT_FALSE(child.IsSession());
        scene.MarkSession(root);
        EXPECT_TRUE(root.IsSession());
        EXPECT_TRUE(child.IsSession());
        EXPECT_FALSE(user.IsSession());
    });
}

TEST(SceneTest, MappedPrimvarsWriteEachPrimsValuesInPlace) {
    ForEachScene([](Scene& scene) {
        const std::vector<Float3> colors = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        const std::vector<float> opacities = {0.25f, 0.5f, 0.75f};
        scene.Define<World>("/env_0");
        scene.Define<Points>(
            "/env_0/Cloud",
            {.primvars = {Primvar::Create("displayColor", Interpolation::Vertex, colors),
                          Primvar::Create("displayOpacity", Interpolation::Vertex, opacities)},
             .points = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}});
        scene.Clone(*scene.Find("/env_0"), std::vector<std::string>{"/env_1", "/env_2"});
        const View clouds =
            scene.Select(std::vector<std::string>{"/env_0/Cloud", "/env_1/Cloud", "/env_2/Cloud"});
        UntypedAttribute primvars = clouds.Get("primvars");
        const auto write = [&](float base) {
            Mapping mapping = primvars.MapPrimvar("displayColor", MapAccess::Write);
            const device::Tensor& values = mapping.Values();
            EXPECT_EQ(std::vector<int64_t>(values.Shape().begin(), values.Shape().end()),
                      (std::vector<int64_t>{3, 3, 3}));
            for (uint32_t w = 0; w < 3; ++w) {
                const std::vector<Float3> tint(3, Float3{base + float(w), 0, 0});
                scene.GetDevice().GetQueue().Write(
                    values.GetBuffer().Slice(values.Offset() +
                                                 uint64_t(w) * values.Strides()[0] * sizeof(float),
                                             sizeof(Float3) * 3),
                    tint.data());
            }
            mapping.Close();
            scene.Publish().Wait();
            for (uint32_t w = 0; w < 3; ++w) {
                const std::vector<Primvar> found = primvars.GetPrimvars(w);
                ASSERT_EQ(found.size(), 2u);
                for (const Primvar& primvar : found) {
                    if (primvar.name == "displayColor") {
                        EXPECT_EQ(primvar.Values<Float3>(),
                                  std::vector<Float3>(3, Float3{base + float(w), 0, 0}));
                    } else {
                        EXPECT_EQ(primvar.Values<float>(), opacities);
                    }
                }
            }
        };
        write(10);
        write(20);

        Mapping opacity = primvars.MapPrimvar("displayOpacity", MapAccess::Read);
        EXPECT_EQ(
            std::vector<int64_t>(opacity.Values().Shape().begin(), opacity.Values().Shape().end()),
            (std::vector<int64_t>{3, 3}));
        opacity.Close();
        EXPECT_THROW(primvars.MapPrimvar("rest", MapAccess::Write), Error);
        EXPECT_THROW(clouds.Get("points").MapPrimvar("displayColor", MapAccess::Write), Error);

        scene.Define<Points>(
            "/Other",
            {.primvars = {Primvar::Create("rest", Interpolation::Vertex, colors),
                          Primvar::Create("displayOpacity", Interpolation::Vertex, opacities)}});
        scene.Define<Points>(
            "/Short",
            {.primvars = {Primvar::Create("displayColor", Interpolation::Vertex, colors)}});
        EXPECT_THROW(scene.Select(std::vector<std::string>{"/env_0/Cloud", "/Other"})
                         .Get("primvars")
                         .MapPrimvar("displayColor", MapAccess::Write),
                     Error);
        EXPECT_THROW(scene.Select(std::vector<std::string>{"/env_0/Cloud", "/Short"})
                         .Get("primvars")
                         .MapPrimvar("displayColor", MapAccess::Write),
                     Error);
    });
}

}
}
