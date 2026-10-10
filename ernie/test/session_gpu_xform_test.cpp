// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

using ernie::Device;
using ernie::DeviceKind;
using ernie::DType;
using ernie::GpuXform;
using ernie::Path;
using ernie::Scene;
using ernie::Tensor;
namespace tokens = ernie::tokens;

namespace {

// Tensors that claim CUDA memory: the rules below read only their metadata, never the memory.
std::array<float, 64> gMemory{};

Tensor Cuda(std::vector<std::int64_t> shape, DType type = DType::Float32) {
    return Tensor(gMemory.data(), type, std::move(shape), {}, Device{DeviceKind::Cuda, 0});
}

std::vector<std::string> Order(std::initializer_list<std::string_view> ops) {
    return {ops.begin(), ops.end()};
}

/// Links /Env<i>/Link with `ops` as their order.
std::vector<Path> Links(Scene& scene, int count, std::vector<std::string> ops) {
    std::vector<Path> links;
    for (int i = 0; i < count; ++i) {
        links.emplace_back("/Env" + std::to_string(i) + "/Link");
        ernie::Xform::Define(scene, links.back());
        scene.Set(links.back(), tokens::kXformOpOrder, ops);
    }
    return links;
}

std::string ErrorOf(const Scene& scene, const Path& path) {
    try {
        ernie::GpuXformOf(scene, path);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return "no error";
}

}  // namespace

// --- Rows -------------------------------------------------------------------------------------

TEST(SceneRows, WritesOfManyPathsRememberTheTensorAndRow) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 3, Order({tokens::kTranslate}));
    const Tensor rows = Cuda({3, 3});
    scene.Write(links, tokens::kTranslate, rows, 7);
    for (std::int64_t i = 0; i < 3; ++i) {
        const auto* row = scene.RowOf(links[static_cast<std::size_t>(i)], tokens::kTranslate);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->first.Data(), rows.Data());
        EXPECT_EQ(row->first.Shape(), rows.Shape());
        EXPECT_EQ(row->second, i);
        EXPECT_EQ(scene.WriteStream(links[static_cast<std::size_t>(i)], tokens::kTranslate), 7u);
    }
    EXPECT_EQ(scene.RowOf(links[0], tokens::kOrient), nullptr);
    EXPECT_EQ(scene.RowOf(Path("/Missing"), tokens::kTranslate), nullptr);
}

TEST(SceneRows, OwnTransformVersionsIgnoreAncestors) {
    Scene scene;
    ernie::Xform::Define(scene, "/A/B");
    EXPECT_EQ(scene.OwnTransformVersion(Path("/A/B")), 0u);
    scene.Set(Path("/A"), tokens::kTranslate, glm::dvec3(1.0));
    EXPECT_EQ(scene.OwnTransformVersion(Path("/A/B")), 0u) << "the parent moved, not B";
    EXPECT_EQ(scene.OwnTransformVersion(Path("/A")), scene.Version());
    scene.Set(Path("/A/B"), tokens::kXformOpOrder, Order({tokens::kTranslate}));
    EXPECT_EQ(scene.OwnTransformVersion(Path("/A/B")), scene.Version());
    scene.Set(Path("/A/B"), tokens::kDisplayColor, glm::vec3(1.0f));
    EXPECT_LT(scene.OwnTransformVersion(Path("/A/B")), scene.Version()) << "not a transform op";
    EXPECT_EQ(scene.OwnTransformVersion(Path("/Missing")), 0u);
}

TEST(SceneRows, OtherWritesForgetTheRow) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 3, Order({tokens::kTranslate}));
    scene.Write(links, tokens::kTranslate, Cuda({3, 3}));
    scene.Set(links[0], tokens::kTranslate, glm::dvec3(1.0));
    scene.Write(links[1], tokens::kTranslate, Cuda({3}));
    scene.Clear(links[2], tokens::kTranslate);
    for (const Path& link : links) {
        EXPECT_EQ(scene.RowOf(link, tokens::kTranslate), nullptr) << link.String();
    }
}

// --- Which prims are posed on the GPU ---------------------------------------------------------

TEST(GpuXform, CpuOpsAreNoGpuPose) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 2, Order({tokens::kTranslate}));
    scene.Set(links[0], tokens::kTranslate, glm::dvec3(1.0));
    EXPECT_FALSE(ernie::GpuXformOf(scene, links[0]).has_value());
    EXPECT_FALSE(ernie::GpuXformOf(scene, links[1]).has_value()) << "no value at all";
    ernie::Xform::Define(scene, "/Plain");
    EXPECT_FALSE(ernie::GpuXformOf(scene, Path("/Plain")).has_value());
}

TEST(GpuXform, RowsOfTranslateOrientAndScalePoseOnTheGpu) {
    Scene scene;
    const std::vector<Path> links =
        Links(scene, 4, Order({tokens::kTranslate, tokens::kOrient, tokens::kScale}));
    const Tensor translate = Cuda({4, 3});
    const Tensor orient = Cuda({4, 4});
    const Tensor scale = Cuda({4, 3});
    scene.Write(links, tokens::kTranslate, translate, 5);
    scene.Write(links, tokens::kOrient, orient, 5);
    scene.Write(links, tokens::kScale, scale, 5);
    const std::optional<GpuXform> pose = ernie::GpuXformOf(scene, links[2]);
    ASSERT_TRUE(pose.has_value());
    EXPECT_FALSE(pose->reset);
    EXPECT_EQ(pose->row, 2);
    EXPECT_EQ(pose->translate.Shape(), translate.Shape());
    EXPECT_EQ(pose->orient.Shape(), orient.Shape());
    EXPECT_EQ(pose->scale.Shape(), scale.Shape());
    EXPECT_EQ(pose->stream, 5u);
}

TEST(GpuXform, AnyOrderedSubsetAndAResetPose) {
    Scene scene;
    const std::vector<Path> links =
        Links(scene, 2, Order({tokens::kResetXformStack, tokens::kOrient}));
    scene.Write(links, tokens::kOrient, Cuda({2, 4}));
    const std::optional<GpuXform> pose = ernie::GpuXformOf(scene, links[1]);
    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->reset);
    EXPECT_TRUE(pose->translate.Empty());
    EXPECT_FALSE(pose->orient.Empty());
    EXPECT_TRUE(pose->scale.Empty());
}

TEST(GpuXform, UnsupportedPosesSayWhy) {
    const auto with = [](std::vector<std::string> order, auto write) {
        Scene scene;
        const std::vector<Path> links = Links(scene, 2, std::move(order));
        write(scene, links);
        return ErrorOf(scene, links[0]);
    };
    // Suffixed and inverted ops are composed on the CPU only.
    EXPECT_NE(with(Order({tokens::kTranslate, "xformOp:translate:pivot"}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                   })
                  .find("suffixed and inverted ops"),
              std::string::npos);
    EXPECT_NE(with(Order({tokens::kTranslate, "!invert!xformOp:translate"}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                   })
                  .find("suffixed and inverted ops"),
              std::string::npos);
    // Widths that fit no op.
    EXPECT_NE(with(Order({tokens::kRotateXYZ}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kRotateXYZ, Cuda({2, 4}));
                   })
                  .find("(n, 3)"),
              std::string::npos);
    EXPECT_NE(with(Order({tokens::kTransform}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kTransform, Cuda({2, 12}));
                   })
                  .find("(n, 16)"),
              std::string::npos);
    // An op of the order on the CPU while another is on the GPU.
    EXPECT_NE(with(Order({tokens::kTranslate, tokens::kOrient}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                       s.Set(l[0], tokens::kOrient, glm::dquat(1, 0, 0, 0));
                   })
                  .find("written for many prims"),
              std::string::npos);
    // A GPU array written to one prim rather than as rows of many.
    EXPECT_NE(with(Order({tokens::kTranslate}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l[0], tokens::kTranslate, Cuda({3}));
                   })
                  .find("written for many prims"),
              std::string::npos);
    // Wrong element type or row width.
    EXPECT_NE(with(Order({tokens::kTranslate}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kTranslate, Cuda({2, 3}, DType::Float64));
                   })
                  .find("float32"),
              std::string::npos);
    EXPECT_NE(with(Order({tokens::kOrient}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kOrient, Cuda({2, 3}));
                   })
                  .find("(n, 4)"),
              std::string::npos);
    // Arrays of different lengths, or rows that do not line up.
    EXPECT_NE(with(Order({tokens::kTranslate, tokens::kOrient}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                       const std::vector<Path> three{l[0], l[1], Path("/Env0")};
                       s.Write(three, tokens::kOrient, Cuda({3, 4}));
                   })
                  .find("same number of rows"),
              std::string::npos);
    EXPECT_NE(with(Order({tokens::kTranslate, tokens::kOrient}),
                   [](Scene& s, const std::vector<Path>& l) {
                       s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                       const std::vector<Path> swapped{l[1], l[0]};
                       s.Write(swapped, tokens::kOrient, Cuda({2, 4}));
                   })
                  .find("same row"),
              std::string::npos);
}

TEST(GpuXform, OtherOpsAndOrdersAreComposedOnTheGpu) {
    const auto pose = [](std::vector<std::string> order, auto write) {
        Scene scene;
        const std::vector<Path> links = Links(scene, 2, std::move(order));
        write(scene, links);
        return *ernie::GpuXformOf(scene, links[1]);
    };
    // Euler rotations, rotations about one axis, matrices, uniform scales, other orders.
    const GpuXform euler = pose(Order({tokens::kTranslate, tokens::kRotateZYX}),
                                [](Scene& s, const std::vector<Path>& l) {
                                    s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                                    s.Write(l, tokens::kRotateZYX, Cuda({2, 3}));
                                });
    EXPECT_TRUE(euler.composed);
    ASSERT_EQ(euler.ops.size(), 2u);
    EXPECT_EQ(euler.ops[1].first, tokens::kRotateZYX);
    EXPECT_TRUE(euler.translate.Empty()) << "composed poses keep their ops, not instance arrays";
    EXPECT_EQ(euler.row, 1);
    EXPECT_TRUE(pose(Order({tokens::kRotateZ}), [](Scene& s, const std::vector<Path>& l) {
                    s.Write(l, tokens::kRotateZ, Cuda({2, 1}));
                }).composed);
    EXPECT_TRUE(pose(Order({tokens::kScale}), [](Scene& s, const std::vector<Path>& l) {
                    s.Write(l, tokens::kScale, Cuda({2, 1}));
                }).composed);
    EXPECT_TRUE(pose(Order({tokens::kScale, tokens::kTranslate}),
                     [](Scene& s, const std::vector<Path>& l) {
                         s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                         s.Write(l, tokens::kScale, Cuda({2, 3}));
                     })
                    .composed);
    // Translate, orient and scale in that order stay instance arrays.
    const GpuXform plain = pose(Order({tokens::kTranslate, tokens::kScale}),
                                [](Scene& s, const std::vector<Path>& l) {
                                    s.Write(l, tokens::kTranslate, Cuda({2, 3}));
                                    s.Write(l, tokens::kScale, Cuda({2, 3}));
                                });
    EXPECT_FALSE(plain.composed);
    EXPECT_FALSE(plain.translate.Empty());
    EXPECT_FALSE(plain.scale.Empty());
}

TEST(GpuXform, ATransformArrayAloneIsComposed) {
    Scene scene;
    const std::vector<Path> links{Path("/A"), Path("/B")};
    for (const Path& link : links) {
        ernie::Xform::Define(scene, link);
    }
    scene.Write(links, tokens::kTransform, Cuda({2, 16}));
    const std::optional<GpuXform> pose = ernie::GpuXformOf(scene, links[0]);
    ASSERT_TRUE(pose.has_value());
    EXPECT_TRUE(pose->composed);
    ASSERT_EQ(pose->ops.size(), 1u);
    EXPECT_EQ(pose->ops[0].first, tokens::kTransform);
}

// --- Anchors ----------------------------------------------------------------------------------

TEST(GpuXform, AnchorsAreTheNearestGpuPoseAbove) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 2, Order({tokens::kTranslate}));
    scene.Write(links, tokens::kTranslate, Cuda({2, 3}));
    ernie::Cube::Define(scene, "/Env0/Link/Visuals/Box");
    EXPECT_EQ(ernie::GpuXformAnchorOf(scene, Path("/Env0/Link/Visuals/Box")), links[0]);
    EXPECT_EQ(ernie::GpuXformAnchorOf(scene, links[1]), links[1]) << "a prim anchors itself";
    EXPECT_FALSE(ernie::GpuXformAnchorOf(scene, Path("/Env0")).has_value());
}

TEST(GpuXform, AResetBelowTheAnchorCutsItOff) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 1, Order({tokens::kTranslate}));
    scene.Write(links, tokens::kTranslate, Cuda({1, 3}));
    ernie::Cube::Define(scene, "/Env0/Link/Free/Box");
    scene.Set(Path("/Env0/Link/Free"), tokens::kXformOpOrder, Order({tokens::kResetXformStack}));
    EXPECT_FALSE(ernie::GpuXformAnchorOf(scene, Path("/Env0/Link/Free/Box")).has_value());
}

// --- Camera poses as positions and quaternions -------------------------------------------------

TEST(CameraPoses, RowsOfPositionsAndQuaternionsMatchMatrices) {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Rig/Camera");
    scene.Set(Path("/World/Rig"), tokens::kTranslate, glm::dvec3(100.0, 0.0, 0.0));
    scene.Set(Path("/World/Rig"), tokens::kXformOpOrder, Order({tokens::kTranslate}));
    const glm::dquat q = glm::angleAxis(glm::radians(30.0), glm::normalize(glm::dvec3(1, 2, 3)));
    const std::array<float, 14> rows{1.0f, 2.0f, 3.0f, float(q.x), float(q.y), float(q.z), float(q.w),
                                     -4.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    scene.Write(Path("/World/Rig/Camera"), tokens::kPoses, Tensor::Copy<float>(rows, {2, 7}));
    const Path camera("/World/Rig/Camera");
    const std::vector<glm::dmat4> poses = ernie::ViewPosesOf(scene, std::span(&camera, 1), 64, 48);
    ASSERT_EQ(poses.size(), 2u);
    glm::dmat4 first(glm::mat3_cast(q));
    first[3] = glm::dvec4(101.0, 2.0, 3.0, 1.0);  // after the parent's translate
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(poses[0][c][r], first[c][r], 1e-6) << c << ", " << r;
        }
    }
    EXPECT_EQ(glm::dvec3(poses[1][3]), glm::dvec3(96.0, 0.0, 9.0));
    EXPECT_TRUE(ernie::PosesAreQuaternions(*scene.Read(camera, tokens::kPoses)));
    EXPECT_FALSE(ernie::PosesAreQuaternions(Cuda({2, 4, 4})));
    EXPECT_FALSE(ernie::PosesAreQuaternions(Cuda({2, 16})));
}

TEST(CameraPoses, OtherShapesAreRefused) {
    Scene scene;
    const Path camera("/Camera");
    ernie::Camera::Define(scene, camera);
    scene.Write(camera, tokens::kPoses, Tensor::Zeros(DType::Float32, {2, 6}));
    EXPECT_THROW(ernie::ViewPosesOf(scene, std::span(&camera, 1), 64, 48), std::invalid_argument);
    scene.Write(camera, tokens::kPoses, Cuda({2, 7}));
    EXPECT_NE(ernie::GpuPosesOf(scene, camera), nullptr) << "GPU rows of seven stay on the GPU";
}

// --- PrimSets ---------------------------------------------------------------------------------

TEST(PrimSets, WriteRowsLikePathsAndKeepThePrims) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 3, Order({tokens::kTranslate}));
    const ernie::PrimSet prims(links);
    EXPECT_EQ(prims.Size(), 3u);
    const std::array<float, 9> values{0, 0, 0, 1, 1, 1, 2, 2, 2};
    for (int step = 0; step < 3; ++step) {
        const Tensor rows = Tensor::Copy<float>(values, {3, 3});
        scene.Write(prims, tokens::kTranslate, rows, 4);
        for (std::size_t i = 0; i < links.size(); ++i) {
            const auto* row = scene.RowOf(links[i], tokens::kTranslate);
            ASSERT_NE(row, nullptr);
            EXPECT_EQ(row->first.Data(), rows.Data()) << "step " << step;
            EXPECT_EQ(row->second, static_cast<std::int64_t>(i));
            EXPECT_EQ(scene.WriteStream(links[i], tokens::kTranslate), 4u);
            EXPECT_EQ(glm::dvec3(scene.LocalTransform(links[i])[3]), glm::dvec3(double(i)));
        }
    }
}

TEST(PrimSets, LookPrimsUpAgainAfterRemovalsOrInAnotherScene) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 2, Order({tokens::kTranslate}));
    const ernie::PrimSet prims(links);
    scene.Write(prims, tokens::kTranslate, Tensor::Zeros(DType::Float32, {2, 3}));
    // Removing and redefining a prim: the set finds the new one.
    scene.Remove(links[1]);
    EXPECT_THROW(scene.Write(prims, tokens::kTranslate, Tensor::Zeros(DType::Float32, {2, 3})),
                 std::invalid_argument);
    ernie::Xform::Define(scene, links[1]);
    scene.Set(links[1], tokens::kXformOpOrder, Order({tokens::kTranslate}));
    const std::array<float, 6> moved{0, 0, 0, 5, 5, 5};
    scene.Write(prims, tokens::kTranslate, Tensor::Copy<float>(moved, {2, 3}));
    EXPECT_EQ(glm::dvec3(scene.LocalTransform(links[1])[3]), glm::dvec3(5.0));
    // Another scene with the same paths.
    Scene other;
    Links(other, 2, Order({tokens::kTranslate}));
    other.Write(prims, tokens::kTranslate, Tensor::Copy<float>(moved, {2, 3}));
    EXPECT_EQ(glm::dvec3(other.LocalTransform(links[1])[3]), glm::dvec3(5.0));
    // A copied scene has prims of its own.
    Scene copy = scene;
    const std::array<float, 6> again{0, 0, 0, 7, 7, 7};
    copy.Write(prims, tokens::kTranslate, Tensor::Copy<float>(again, {2, 3}));
    EXPECT_EQ(glm::dvec3(copy.LocalTransform(links[1])[3]), glm::dvec3(7.0));
    EXPECT_EQ(glm::dvec3(scene.LocalTransform(links[1])[3]), glm::dvec3(5.0));
}

TEST(PrimSets, RefuseRowsOfAnotherCountAndWriteNothingForAMissingPrim) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 2, Order({tokens::kTranslate}));
    EXPECT_THROW(scene.Write(ernie::PrimSet(links), tokens::kTranslate,
                             Tensor::Zeros(DType::Float32, {3, 3})),
                 std::invalid_argument);
    const ernie::PrimSet withMissing({links[0], Path("/Missing")});
    EXPECT_THROW(scene.Write(withMissing, tokens::kTranslate, Tensor::Zeros(DType::Float32, {2, 3})),
                 std::invalid_argument);
    EXPECT_EQ(scene.Read(links[0], tokens::kTranslate), nullptr) << "nothing was written";
}

// --- Quaternion order --------------------------------------------------------------------------

TEST(QuaternionOrder, OrientArraysMayPutWFirst) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 2, Order({tokens::kOrient}));
    const glm::dquat q = glm::angleAxis(glm::radians(50.0), glm::normalize(glm::dvec3(2, -1, 3)));
    const std::array<float, 8> wxyz{float(q.w), float(q.x), float(q.y), float(q.z), 1, 0, 0, 0};
    scene.Write(links, tokens::kOrient, Tensor::Copy<float>(wxyz, {2, 4}));
    for (const Path& link : links) {
        scene.Set(link, tokens::kQuaternionOrder, tokens::kWxyz);
    }
    const glm::dmat4 local = scene.LocalTransform(links[0]);
    const glm::dmat4 expected(glm::mat3_cast(q));
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(local[c][r], expected[c][r], 1e-6);
        }
    }
    EXPECT_EQ(scene.LocalTransform(links[1]), glm::dmat4(1.0)) << "w = 1 first is the identity";
    // A quaternion value is the same whatever the order says.
    scene.Set(links[1], tokens::kOrient, q);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(scene.LocalTransform(links[1])[c][r], expected[c][r], 1e-12);
        }
    }
}

TEST(QuaternionOrder, GpuPosesAndCamerasCarryIt) {
    Scene scene;
    const std::vector<Path> links = Links(scene, 2, Order({tokens::kOrient}));
    scene.Write(links, tokens::kOrient, Cuda({2, 4}));
    EXPECT_FALSE(ernie::GpuXformOf(scene, links[0])->wFirst);
    scene.Set(links[0], tokens::kQuaternionOrder, tokens::kWxyz);
    EXPECT_TRUE(ernie::GpuXformOf(scene, links[0])->wFirst);
    scene.Set(links[0], tokens::kQuaternionOrder, "zyxw");
    EXPECT_THROW(ernie::GpuXformOf(scene, links[0]), std::invalid_argument);

    const Path camera("/Camera");
    ernie::Camera::Define(scene, camera);
    scene.Set(camera, tokens::kQuaternionOrder, tokens::kWxyz);
    const glm::dquat q = glm::angleAxis(glm::radians(20.0), glm::dvec3(0, 0, 1));
    const std::array<float, 7> row{1, 2, 3, float(q.w), float(q.x), float(q.y), float(q.z)};
    scene.Write(camera, tokens::kPoses, Tensor::Copy<float>(row, {1, 7}));
    const std::vector<glm::dmat4> poses = ernie::ViewPosesOf(scene, std::span(&camera, 1), 64, 48);
    ASSERT_EQ(poses.size(), 1u);
    EXPECT_NEAR(poses[0][0][1], std::sin(glm::radians(20.0)), 1e-6);
    EXPECT_NEAR(poses[0][0][0], std::cos(glm::radians(20.0)), 1e-6);
    EXPECT_EQ(glm::dvec3(poses[0][3]), glm::dvec3(1, 2, 3));
}
