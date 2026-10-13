// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/skel_schemas.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/skinning.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

constexpr int kSegments = 10;

/// A camera looking down at a red bar along x, 20 long and 4 wide, in
/// quads, lit from above.
Scene BarScene() {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera")
        .SetFocalLength(18.0);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 40.0},
                                     {0.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform,
             ernie::DistantLightTransform({0.0, 0.0, 1.0}));
    ernie::RenderSettings::Define(scene, "/Render/Settings")
        .SetResolution(glm::dvec2(kSize, kSize))
        .SetAmbient(0.0);
    ernie::Material::Define(scene, "/Looks/Red")
        .SetAlbedo(glm::dvec3(1.0, 0.0, 0.0));
    std::vector<float> points;
    for (int i = 0; i <= kSegments; ++i) {
        const float x =
            -10.0f + 20.0f * static_cast<float>(i) / kSegments;
        points.insert(points.end(), {x, -2.0f, 0.0f, x, 2.0f, 0.0f});
    }
    std::vector<std::int32_t> counts(kSegments, 4);
    std::vector<std::int32_t> indices;
    for (int i = 0; i < kSegments; ++i) {
        indices.insert(indices.end(),
                       {2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1});
    }
    ernie::Mesh::Define(scene, "/World/Bar")
        .SetPoints(
            Tensor::Copy<float>(points, {2 * (kSegments + 1), 3}))
        .SetFaceVertexCounts(
            Tensor::Copy<std::int32_t>(counts, {kSegments}))
        .SetFaceVertexIndices(
            Tensor::Copy<std::int32_t>(indices, {4 * kSegments}))
        .SetMaterialBinding("/Looks/Red");
    return scene;
}

/// Binds the bar to a skeleton whose elbow, at the bar's middle,
/// turns 60 degrees about z from time code 0 to 10; points ramp from
/// the root to the elbow across the middle.
void Skin(Scene& scene, const std::string& method) {
    const std::vector<float> identity = {1, 0, 0, 0, 0, 1, 0, 0,
                                         0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<float> binds = identity;
    binds.insert(binds.end(), identity.begin(), identity.end());
    ernie::Skeleton::Define(scene, "/World/Skel")
        .SetJoints({"Root", "Root/Elbow"})
        .SetBindTransforms(Tensor::Copy<float>(binds, {2, 4, 4}))
        .SetRestTransforms(Tensor::Copy<float>(binds, {2, 4, 4}))
        .SetAnimationSource("/World/Skel/Bend");
    const glm::quat bent =
        glm::angleAxis(glm::radians(60.0f), glm::vec3(0, 0, 1));
    ernie::SkelAnimation::Define(scene, "/World/Skel/Bend")
        .SetJoints({"Root", "Root/Elbow"})
        .SetTimes(
            Tensor::Copy<float>(std::vector<float>{0.0f, 10.0f}, {2}))
        .SetTranslations(Tensor::Copy<float>(
            std::vector<float>(12, 0.0f), {2, 2, 3}))
        .SetRotations(Tensor::Copy<float>(
            std::vector<float>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                               bent.x, bent.y, bent.z, bent.w},
            {2, 2, 4}))
        .SetScales(Tensor::Copy<float>(std::vector<float>(12, 1.0f),
                                       {2, 2, 3}));
    std::vector<std::int32_t> joints;
    std::vector<float> weights;
    for (int i = 0; i <= kSegments; ++i) {
        const float x =
            -10.0f + 20.0f * static_cast<float>(i) / kSegments;
        const float elbow = std::clamp((x + 4.0f) / 8.0f, 0.0f, 1.0f);
        for (int side = 0; side < 2; ++side) {
            joints.insert(joints.end(), {0, 1});
            weights.insert(weights.end(), {1.0f - elbow, elbow});
        }
    }
    const std::int64_t n = 2 * (kSegments + 1);
    ernie::Mesh(scene, "/World/Bar")
        .SetSkeleton("/World/Skel")
        .SetJointIndices(Tensor::Copy<std::int32_t>(joints, {n, 2}))
        .SetJointWeights(Tensor::Copy<float>(weights, {n, 2}))
        .SetSkinningMethod(method);
}

}  // namespace

// A point cache (time-sampled points) blends on the GPU as the scene
// blends it on the CPU, plain or as a subdivision surface's control
// points, posed again as the time code moves.
TEST_F(RasterDelegateTest, PointCachesBlendOnTheGpuAsOnTheCpu) {
    for (const std::string scheme : {"none", "catmullClark"}) {
        Scene cached = BarScene();
        ernie::Mesh(cached, "/World/Bar")
            .SetSubdivisionScheme(scheme);
        const Tensor flat =
            *cached.Read("/World/Bar", tokens::kPoints);
        std::vector<float> raised(flat.As<float>(),
                                  flat.As<float>() + flat.Elements());
        for (std::size_t i = 0; i < raised.size(); i += 3) {
            raised[i + 1] += 0.4f * raised[i];  // Sheared along y.
        }
        cached.WriteAt("/World/Bar", tokens::kPoints, 0.0, flat);
        cached.WriteAt("/World/Bar", tokens::kPoints, 10.0,
                       Tensor::Copy<float>(raised, flat.Shape()));
        RasterDelegate delegate(*context_);
        std::vector<std::uint8_t> previous;
        for (const double time : {0.0, 4.0, 10.0}) {
            ernie::Stage(cached, Path::Root()).SetTimeCode(time);
            const std::vector<std::uint8_t> gpu =
                Bytes(delegate.Render(cached));
            Scene reference = BarScene();
            ernie::Mesh(reference, "/World/Bar")
                .SetSubdivisionScheme(scheme)
                .SetPoints(
                    *cached.Read("/World/Bar", tokens::kPoints));
            RasterDelegate cpu(*context_);
            EXPECT_LE(
                BytesDiffering(gpu, Bytes(cpu.Render(reference))),
                gpu.size() / 200)
                << scheme << " at " << time;
            if (!previous.empty()) {
                EXPECT_GT(BytesDiffering(gpu, previous),
                          gpu.size() / 200)
                    << scheme << ": the bar should shear";
            }
            previous = gpu;
        }
    }
}

// The GPU poses a skinned mesh as the CPU reference does: linearly
// and as dual quaternions, a subdivision surface refined after, posed
// again as the time code moves.
TEST_F(RasterDelegateTest, SkinnedMeshesPoseOnTheGpuAsOnTheCpu) {
    for (const std::string method :
         {"classicLinear", "dualQuaternion"}) {
        for (const std::string scheme : {"none", "catmullClark"}) {
            Scene skinned = BarScene();
            ernie::Mesh(skinned, "/World/Bar")
                .SetSubdivisionScheme(scheme);
            Skin(skinned, method);
            RasterDelegate delegate(*context_);
            std::vector<std::uint8_t> previous;
            for (const double time : {0.0, 10.0}) {
                ernie::Stage(skinned, Path::Root()).SetTimeCode(time);
                const std::vector<std::uint8_t> gpu =
                    Bytes(delegate.Render(skinned));

                // The same bar with the CPU's posed points.
                const auto binding =
                    ernie::SkinBindingOf(skinned, "/World/Bar");
                ASSERT_TRUE(binding.has_value());
                const ernie::SkinPose pose = ernie::SkinPoseOf(
                    skinned, "/World/Bar", *binding,
                    ernie::SkeletonPoseOf(skinned, "/World/Skel",
                                          time));
                const Tensor* rest =
                    skinned.Read("/World/Bar", tokens::kPoints);
                const std::vector<float> posed = ernie::SkinnedPoints(
                    *binding, pose,
                    std::span(
                        rest->Contiguous().As<float>(),
                        static_cast<std::size_t>(rest->Elements())));
                Scene reference = BarScene();
                ernie::Mesh(reference, "/World/Bar")
                    .SetSubdivisionScheme(scheme)
                    .SetPoints(
                        Tensor::Copy<float>(posed, rest->Shape()));
                RasterDelegate cpu(*context_);
                const std::vector<std::uint8_t> expected =
                    Bytes(cpu.Render(reference));
                EXPECT_LE(BytesDiffering(gpu, expected),
                          gpu.size() / 200)
                    << method << " " << scheme << " at " << time;
                if (!previous.empty()) {
                    EXPECT_GT(BytesDiffering(gpu, previous),
                              gpu.size() / 100)
                        << method << " " << scheme
                        << ": the bar should bend";
                }
                previous = gpu;
            }
        }
    }
}
