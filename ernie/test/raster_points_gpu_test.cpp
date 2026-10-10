// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute_interop.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using ernie::Path;
using ernie::RasterDelegate;
using ernie::Scene;
using ernie::Tensor;
using namespace ernie::testing;
namespace tokens = ernie::tokens;

namespace {

constexpr std::uint32_t kPixels = 64;

/// Three spheres in a row along x, 4 m across, 40 m below a camera looking straight down, and
/// render vars of each kind through /Render/Out.
Scene PointsScene() {
    Scene scene;
    ernie::Camera::Define(scene, "/World/Camera").SetFocalLength(25.0).SetSensorNoise(false);
    scene.Set("/World/Camera", tokens::kTransform,
              ernie::LookAtTransform({0.0, 0.0, 40.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, ernie::DistantLightTransform({0.3, 0.2, 1.0}));
    const std::array<float, 9> points{-8, 0, 0, 0, 0, 0, 8, 0, 0};
    const std::array<float, 9> colors{1, 0, 0, 0, 1, 0, 0, 0, 1};
    ernie::Points::Define(scene, "/World/Grains")
        .SetPoints(Tensor::Copy<float>(points, {3, 3}))
        .SetWidths({4.0f});
    // A colour per point: an array, where the schema's displayColor is one colour.
    scene.Write("/World/Grains", tokens::kDisplayColor, Tensor::Copy<float>(colors, {3, 3}));
    std::vector<Path> vars;
    for (const std::string_view name : {"LdrColor", "DistanceToCamera", "InstanceId", "PrimId"}) {
        const Path var("/Render/Vars/" + std::string(name));
        ernie::RenderVar::Define(scene, var).SetSourceName(name);
        vars.push_back(var);
    }
    ernie::RenderProduct::Define(scene, "/Render/Out")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kPixels, kPixels));
    scene.Set("/Render/Out", tokens::kOrderedVars, vars);
    return scene;
}

/// Each var of one step of /Render/Out, as floats.
std::map<std::string, std::vector<float>> Step(RasterDelegate& delegate, const Scene& scene) {
    const Path product[] = {"/Render/Out"};
    auto outputs = delegate.Step(scene, product).at("/Render/Out").vars;
    std::map<std::string, std::vector<float>> values;
    for (auto& [name, var] : outputs) {
        const Tensor tensor = var.Map(ernie::MapDevice::Cpu).tensor;
        std::vector<float>& out = values[name];
        out.resize(static_cast<std::size_t>(tensor.Elements()));
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = tensor.Type() == ernie::DType::UInt8    ? float(tensor.As<std::uint8_t>()[i])
                     : tensor.Type() == ernie::DType::UInt32 ? float(tensor.As<std::uint32_t>()[i])
                                                             : tensor.As<float>()[i];
        }
    }
    return values;
}

/// The pixel of /World/Camera showing world point (x, y, 0), and its index.
std::size_t PixelOf(double x, double y) {
    // 25 mm over the default 15.29 mm aperture: 1.635 focal lengths per image height.
    const double focal = 25.0 / 15.2908 * kPixels;
    const auto col = static_cast<std::size_t>(kPixels / 2.0 + x / 40.0 * focal);
    const auto row = static_cast<std::size_t>(kPixels / 2.0 - y / 40.0 * focal);
    return row * kPixels + col;
}

}  // namespace

TEST_F(RasterDelegateTest, PointsDrawAsSpheresWithTheirIdsColoursAndDepth) {
    RasterDelegate delegate(*context_);
    Scene scene = PointsScene();
    auto vars = Step(delegate, scene);
    const std::vector<float>& distance = vars.at("DistanceToCamera");
    // The sphere's top, 2 m above its centre, faces the camera.
    EXPECT_NEAR(distance[PixelOf(0.0, 0.0)], 38.0f, 0.05f);
    EXPECT_NEAR(distance[PixelOf(8.0, 0.0)], std::sqrt(8.0f * 8.0f + 40.0f * 40.0f) - 2.0f, 0.3f);
    EXPECT_TRUE(std::isinf(distance[PixelOf(4.0, 6.0)])) << "background between the spheres";
    // Their silhouettes: about pi r^2 pixels each, r = 2 m seen from about 40 m.
    std::size_t covered = 0;
    for (const float d : distance) {
        covered += std::isfinite(d) ? 1u : 0u;
    }
    const double radius = 2.0 / 40.0 * 25.0 / 15.2908 * kPixels;
    EXPECT_NEAR(double(covered), 3.0 * M_PI * radius * radius, 0.15 * 3.0 * M_PI * radius * radius);
    // Each point's index is its InstanceId, and every point the prim's PrimId.
    const std::vector<float>& instance = vars.at("InstanceId");
    EXPECT_EQ(instance[PixelOf(-8.0, 0.0)], 0.0f);
    EXPECT_EQ(instance[PixelOf(0.0, 0.0)], 1.0f);
    EXPECT_EQ(instance[PixelOf(8.0, 0.0)], 2.0f);
    const std::vector<float>& prim = vars.at("PrimId");
    EXPECT_NE(prim[PixelOf(0.0, 0.0)], 0.0f);
    EXPECT_EQ(prim[PixelOf(0.0, 0.0)], prim[PixelOf(8.0, 0.0)]);
    // Their own colours.
    const std::vector<float>& ldr = vars.at("LdrColor");
    const std::size_t red = 4 * PixelOf(-8.0, 0.0);
    const std::size_t blue = 4 * PixelOf(8.0, 0.0);
    EXPECT_GT(ldr[red], 4.0f * ldr[red + 2]);
    EXPECT_GT(ldr[blue + 2], 4.0f * ldr[blue]);

    // Wider points.
    ernie::Points(scene, "/World/Grains").SetWidths({6.0f});
    EXPECT_NEAR(Step(delegate, scene).at("DistanceToCamera")[PixelOf(0.0, 0.0)], 37.0f, 0.05f);
}

TEST_F(RasterDelegateTest, PointsFromCudaMatchTheHostsAndBlurAsTheyMove) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    Scene host = PointsScene();
    RasterDelegate reference(*context_);
    const auto expected = Step(reference, host);
    Scene device = PointsScene();
    for (const std::string_view attribute : {tokens::kPoints, tokens::kDisplayColor}) {
        device.Write("/World/Grains", attribute,
                     cuda->Upload(*device.Read("/World/Grains", attribute)));
    }
    device.Set("/World/Grains", tokens::kExtent,
               std::vector<float>{-10.0f, -2.0f, -2.0f, 10.0f, 2.0f, 2.0f});
    // With the same bounds the shadows fit the same: the same image.
    EXPECT_EQ(Step(delegate, device).at("LdrColor"), expected.at("LdrColor"));

    // The middle sphere moves 6 m along y over a frame with the shutter open: its colour smears
    // along the way, and less of it stays where it started.
    ernie::Camera(device, "/World/Camera").SetShutterOpen(-1.0).SetShutterSamples(8);
    const auto moved = [&](float y) {
        const std::array<float, 9> points{-8, 0, 0, 0, y, 0, 8, 0, 0};
        ernie::Points(device, "/World/Grains")
            .SetPoints(cuda->Upload(Tensor::Copy<float>(points, {3, 3})));
    };
    moved(0.0f);
    Step(delegate, device);
    moved(6.0f);
    const auto blurred = Step(delegate, device).at("LdrColor");
    const std::size_t middle = 4 * PixelOf(0.0, 3.0) + 1;
    EXPECT_GT(blurred[middle], 20.0f) << "green between where it was and is";
    EXPECT_LT(blurred[4 * PixelOf(0.0, 0.0) + 1],
              expected.at("LdrColor")[4 * PixelOf(0.0, 0.0) + 1])
        << "dimmer where it started";
}
