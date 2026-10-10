// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute_interop.h"
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/frame_params.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <cmath>
#include <cstdint>
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

/// MeshScene without noise, rendering HdrColor and LdrColor through /Render/Out.
Scene MotionScene() {
    Scene scene = MeshScene();
    ernie::Camera(scene, "/World/Camera").SetSensorNoise(false);
    std::vector<Path> vars;
    for (const std::string_view name : {"HdrColor", "LdrColor"}) {
        const Path var("/Render/Vars/" + std::string(name));
        ernie::RenderVar::Define(scene, var).SetSourceName(name);
        vars.push_back(var);
    }
    ernie::RenderProduct::Define(scene, "/Render/Out")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(kSize, kSize));
    scene.Set("/Render/Out", tokens::kOrderedVars, vars);
    return scene;
}

/// One step of /Render/Out: render var `var` as floats.
std::vector<float> Step(RasterDelegate& delegate, const Scene& scene, const std::string& var) {
    const Path product[] = {"/Render/Out"};
    const Tensor tensor =
        delegate.Step(scene, product).at("/Render/Out").vars.at(var).Map(ernie::MapDevice::Cpu).tensor;
    std::vector<float> values(static_cast<std::size_t>(tensor.Elements()));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = tensor.Type() == ernie::DType::UInt8
                        ? static_cast<float>(tensor.As<std::uint8_t>()[i])
                        : tensor.As<float>()[i];
    }
    return values;
}

/// The largest difference between `a` and `b` over the largest magnitude of `b`.
float RelativeError(const std::vector<float>& a, const std::vector<float>& b) {
    float error = 0.0f;
    float scale = 1e-12f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        error = std::max(error, std::abs(a[i] - b[i]));
        scale = std::max(scale, std::abs(b[i]));
    }
    return error / scale;
}

void SetShutter(Scene& scene, double open, double close, int samples, double rolling = 0.0) {
    ernie::Camera(scene, "/World/Camera")
        .SetShutterOpen(open)
        .SetShutterClose(close)
        .SetShutterSamples(samples)
        .SetShutterRolling(rolling);
}

glm::dmat4 CameraAt(double x) {
    return ernie::LookAtTransform({x, 0.0, 60.0}, {x, 0.0, 0.0}, {0.0, 1.0, 0.0});
}

/// The weights a global shutter over the whole frame gives `count` sub-frames: the trapezoid
/// rule, halves at the ends.
std::vector<double> TrapezoidWeights(int count) {
    std::vector<double> weights(static_cast<std::size_t>(count), 1.0 / (count - 1));
    weights.front() *= 0.5;
    weights.back() *= 0.5;
    return weights;
}

/// Expects a step from `move(0)` to `move(1)` with the shutter open over the whole frame to render
/// the shutter's weighting of sharp steps at each sub-frame's `move(s)`, and the step after it,
/// without a move, to render sharp again.
void ExpectMotionBlur(RasterDelegate& delegate, Scene& scene,
                      const std::function<void(double)>& move, int samples) {
    SetShutter(scene, 0.0, 0.0, samples);
    const std::vector<double> weights = TrapezoidWeights(samples);
    std::vector<float> expected;
    for (int k = 0; k < samples; ++k) {
        move(k / double(samples - 1));
        const std::vector<float> sharp = Step(delegate, scene, "HdrColor");
        expected.resize(sharp.size(), 0.0f);
        for (std::size_t i = 0; i < sharp.size(); ++i) {
            expected[i] += static_cast<float>(weights[static_cast<std::size_t>(k)]) * sharp[i];
        }
    }
    const std::vector<float> sharp = Step(delegate, scene, "HdrColor");
    SetShutter(scene, -1.0, 0.0, samples);
    move(0.0);
    Step(delegate, scene, "HdrColor");
    move(1.0);
    const std::vector<float> blurred = Step(delegate, scene, "HdrColor");
    EXPECT_LT(RelativeError(blurred, expected), 2e-3f);
    EXPECT_GT(RelativeError(blurred, sharp), 0.05f) << "blurred";
    EXPECT_LT(RelativeError(Step(delegate, scene, "HdrColor"), sharp), 1e-5f)
        << "standing still once it stops";
}

}  // namespace

TEST_F(RasterDelegateTest, InstancesAndDeformingPointsBlurFromHostAndCudaArrays) {
    RasterDelegate delegate(*context_);
    std::string why;
    ernie::ComputeInterop* cuda = delegate.Compute(&why);
    Scene scene = MotionScene();
    // The rocks: the middle one slides 6 m towards -x.
    const auto rocks = [&](double s, bool device) {
        const std::array<float, 9> positions{12, -10, 0, static_cast<float>(12.0 - 6.0 * s),
                                             0,  0,   12, 10, 0};
        const Tensor host = Tensor::Copy<float>(positions, {3, 3});
        ernie::PointInstancer(scene, "/World/Rocks")
            .SetPositions(device ? cuda->Upload(host) : host);
    };
    // The quad: its points stretched up to twice as wide.
    const auto quad = [&](double s, bool device) {
        const auto w = static_cast<float>(2.0 * (1.0 + s));
        const std::array<float, 12> points{-w, -2, 0, w, -2, 0, w, 2, 0, -w, 2, 0};
        const Tensor host = Tensor::Copy<float>(points, {4, 3});
        ernie::Mesh(scene, "/World/Quad").SetPoints(device ? cuda->Upload(host) : host);
    };
    {
        SCOPED_TRACE("instances on the host");
        ExpectMotionBlur(delegate, scene, [&](double s) { rocks(s, false); }, 4);
    }
    {
        SCOPED_TRACE("points on the host");
        ExpectMotionBlur(delegate, scene, [&](double s) { quad(s, false); }, 4);
    }
    if (cuda == nullptr) {
        GTEST_SKIP() << "No CUDA interop: " << why;
    }
    scene.Set("/World/Rocks", tokens::kExtent,
              std::vector<float>{-2.0f, -12.0f, -1.0f, 14.0f, 12.0f, 1.0f});
    scene.Set("/World/Quad", tokens::kExtent,
              std::vector<float>{-4.0f, -2.0f, 0.0f, 4.0f, 2.0f, 0.0f});
    {
        SCOPED_TRACE("instances on CUDA");
        ExpectMotionBlur(delegate, scene, [&](double s) { rocks(s, true); }, 4);
    }
    {
        SCOPED_TRACE("points on CUDA");
        ExpectMotionBlur(delegate, scene, [&](double s) { quad(s, true); }, 4);
    }
    {
        // Written in place each step through a map: the last step's arrays stay for the blur.
        SCOPED_TRACE("instances mapped");
        ExpectMotionBlur(
            delegate, scene,
            [&](double s) {
                const Tensor view = delegate.MapArray(scene, "/World/Rocks", tokens::kPositions);
                const std::array<float, 9> positions{
                    12, -10, 0, static_cast<float>(12.0 - 6.0 * s), 0, 0, 12, 10, 0};
                cuda->Copy(cuda->Upload(Tensor::Copy<float>(positions, {3, 3})), view);
                scene.Unmap("/World/Rocks", tokens::kPositions);
            },
            4);
    }
}

TEST_F(RasterDelegateTest, InstancersLeavingTheViewBlurFromWhereTheyWere) {
    // The rocks' instancer slides 30 m along +x, out of the view (about 25 m to its edge): GPU
    // culling must keep its instances for the sub-frames that still see them.
    RasterDelegate delegate(*context_);
    Scene scene = MotionScene();
    ExpectMotionBlur(
        delegate, scene,
        [&](double s) {
            scene.Set("/World/Rocks", tokens::kTransform,
                      glm::translate(glm::dmat4(1.0), glm::dvec3(30.0 * s, 0.0, 0.0)));
        },
        5);
}

TEST_F(RasterDelegateTest, MotionVectorsFollowTheCameraAndWhatMoves) {
    RasterDelegate delegate(*context_);
    Scene scene = MotionScene();
    const Path var("/Render/Vars/MotionVector");
    ernie::RenderVar::Define(scene, var).SetSourceName("MotionVector");
    scene.Set("/Render/Out", tokens::kOrderedVars,
              std::vector<Path>{"/Render/Vars/HdrColor", var});
    const double focal =
        ernie::CameraParamsOf(scene, "/World/Camera", kSize, kSize).FocalLengthPixels();
    const auto at = [](const std::vector<float>& motion, std::size_t x, std::size_t y) {
        return glm::vec2(motion[(y * kSize + x) * 2], motion[(y * kSize + x) * 2 + 1]);
    };
    const std::size_t centre = kSize / 2;

    // A first frame has nothing to move from.
    scene.Set("/World/Camera", tokens::kTransform, CameraAt(0.0));
    const std::vector<float> first = Step(delegate, scene, "MotionVector");
    ASSERT_EQ(first.size(), std::size_t{kSize} * kSize * 2) << "(rows, cols, 2)";
    for (const float m : first) {
        ASSERT_NEAR(m, 0.0f, 1e-3f) << "but for the reprojection's rounding";
    }

    // The camera 1 m along +x: what it sees 60 m away moves f / 60 pixels the other way.
    scene.Set("/World/Camera", tokens::kTransform, CameraAt(1.0));
    const std::vector<float> panned = Step(delegate, scene, "MotionVector");
    const float shift = static_cast<float>(-focal / 60.0);
    // The red quad, near x = -12 m (left of centre).
    const auto quadX = static_cast<std::size_t>(centre - 12.0 * focal / 60.0);
    EXPECT_NEAR(at(panned, quadX, centre).x, shift, 0.05f);
    EXPECT_NEAR(at(panned, quadX, centre).y, 0.0f, 0.05f);
    EXPECT_NEAR(at(panned, centre, centre).x, 0.0f, 1e-3f) << "the sky, far away, barely moves";

    // Still again, the middle rock slides 2 m along -y (down the image): only it moves.
    Step(delegate, scene, "MotionVector");
    const std::array<float, 9> positions{12, -10, 0, 12, -2, 0, 12, 10, 0};
    ernie::PointInstancer(scene, "/World/Rocks")
        .SetPositions(Tensor::Copy<float>(positions, {3, 3}));
    const std::vector<float> slid = Step(delegate, scene, "MotionVector");
    const auto rockX = static_cast<std::size_t>(centre + 11.0 * focal / 60.0);
    const auto rockY = static_cast<std::size_t>(centre + 2.0 * focal / 60.0);
    EXPECT_NEAR(at(slid, rockX, rockY).y, static_cast<float>(2.0 * focal / 60.0), 0.05f)
        << "down the image";
    EXPECT_NEAR(at(slid, rockX, rockY).x, 0.0f, 0.05f);
    EXPECT_LT(glm::length(at(slid, quadX, centre)), 1e-3f) << "the quad stood still";
}

TEST_F(RasterDelegateTest, MovingCastersShadowEachSubframeFromWhereTheyAre) {
    RasterDelegate delegate(*context_);
    Scene scene = MotionScene();
    // A ground the raised quad shadows under a low sun, the quad sliding across it.
    ernie::GroundPlane::Define(scene, "/World/Ground").SetHeight(-6.0);
    scene.Set("/World/Sun", tokens::kTransform, ernie::DistantLightTransform({0.6, 0.2, 0.8}));
    ernie::DistantLight(scene, "/World/Sun").SetShadowEnable(true);
    scene.Remove("/World/Rocks");
    ExpectMotionBlur(
        delegate, scene,
        [&](double s) {
            scene.Set("/World/Quad", tokens::kTransform,
                      glm::translate(glm::dmat4(1.0), glm::dvec3(-12.0 + 10.0 * s, 0.0, 0.0)));
        },
        5);
}

TEST_F(RasterDelegateTest, CameraMotionBlurAveragesTheViewsAcrossTheShutter) {
    RasterDelegate delegate(*context_);
    Scene scene = MotionScene();
    constexpr int kSamples = 5;
    const glm::dmat4 from = CameraAt(0.0);
    const glm::dmat4 to = CameraAt(6.0);
    // Sharp frames at each sub-frame's pose, summed as the shutter weighs them.
    const std::vector<double> weights = TrapezoidWeights(kSamples);
    std::vector<float> expected;
    for (int k = 0; k < kSamples; ++k) {
        scene.Set("/World/Camera", tokens::kTransform,
                  ernie::InterpolatePose(from, to, k / double(kSamples - 1)));
        const std::vector<float> sharp = Step(delegate, scene, "HdrColor");
        expected.resize(sharp.size(), 0.0f);
        for (std::size_t i = 0; i < sharp.size(); ++i) {
            expected[i] += static_cast<float>(weights[static_cast<std::size_t>(k)]) * sharp[i];
        }
    }
    const std::vector<float> still = Step(delegate, scene, "HdrColor");

    // The shutter open over the whole frame, from the last frame's pose to this one's.
    SetShutter(scene, -1.0, 0.0, kSamples);
    scene.Set("/World/Camera", tokens::kTransform, from);
    Step(delegate, scene, "HdrColor");  // moved back from the last reference's pose
    EXPECT_LT(RelativeError(Step(delegate, scene, "HdrColor"), Step(delegate, scene, "HdrColor")),
              1e-6f)
        << "a camera standing still renders the same frame";
    scene.Set("/World/Camera", tokens::kTransform, to);
    const std::vector<float> blurred = Step(delegate, scene, "HdrColor");
    EXPECT_LT(RelativeError(blurred, expected), 2e-3f);
    EXPECT_GT(RelativeError(blurred, still), 0.1f) << "blurred";
}

TEST_F(RasterDelegateTest, ObjectMotionBlurFollowsTheMovesBetweenSteps) {
    RasterDelegate delegate(*context_);
    Scene scene = MotionScene();
    constexpr int kSamples = 4;
    const auto quadAt = [&](double x) {
        scene.Set("/World/Quad", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0), glm::dvec3(x, 0.0, 0.0)));
    };
    const std::vector<double> weights = TrapezoidWeights(kSamples);
    std::vector<float> expected;
    for (int k = 0; k < kSamples; ++k) {
        quadAt(-12.0 + 6.0 * k / (kSamples - 1));
        const std::vector<float> sharp = Step(delegate, scene, "HdrColor");
        expected.resize(sharp.size(), 0.0f);
        for (std::size_t i = 0; i < sharp.size(); ++i) {
            expected[i] += static_cast<float>(weights[static_cast<std::size_t>(k)]) * sharp[i];
        }
    }
    SetShutter(scene, -1.0, 0.0, kSamples);
    quadAt(-12.0);
    Step(delegate, scene, "HdrColor");
    quadAt(-6.0);
    EXPECT_LT(RelativeError(Step(delegate, scene, "HdrColor"), expected), 2e-3f);
    // The next step without a move stands still again: the quad sharp where it stopped.
    const std::vector<float> stopped = Step(delegate, scene, "HdrColor");
    SetShutter(scene, 0.0, 0.0, kSamples);
    EXPECT_LT(RelativeError(stopped, Step(delegate, scene, "HdrColor")), 1e-5f);
}

TEST_F(RasterDelegateTest, RollingShutterExposesEachRowAtItsOwnTime) {
    RasterDelegate delegate(*context_);
    Scene scene = MotionScene();
    constexpr int kSamples = 9;
    const glm::dmat4 from = CameraAt(0.0);
    const glm::dmat4 to = CameraAt(8.0);
    std::vector<std::vector<float>> sharp;
    for (int k = 0; k < kSamples; ++k) {
        scene.Set("/World/Camera", tokens::kTransform,
                  ernie::InterpolatePose(from, to, k / double(kSamples - 1)));
        sharp.push_back(Step(delegate, scene, "HdrColor"));
    }
    // Instantaneous rows read out over the frame, the top row at the last frame's time.
    SetShutter(scene, 0.0, 0.0, kSamples, 1.0);
    scene.Set("/World/Camera", tokens::kTransform, from);
    Step(delegate, scene, "HdrColor");
    scene.Set("/World/Camera", tokens::kTransform, to);
    const std::vector<float> rolled = Step(delegate, scene, "HdrColor");
    // Each row interpolates the two sub-frames around its time.
    const double step = 1.0 / (kSamples - 1);
    float worst = 0.0f;
    float scale = 1e-12f;
    for (std::size_t y = 0; y < kSize; ++y) {
        const double t = -(1.0 - (static_cast<double>(y) + 0.5) / kSize);
        for (std::size_t x = 0; x < kSize; ++x) {
            for (std::size_t c = 0; c < 4; ++c) {
                const std::size_t i = (y * kSize + x) * 4 + c;
                double value = 0.0;
                for (int k = 0; k < kSamples; ++k) {
                    const double w = std::max(0.0, 1.0 - std::abs(t - (-1.0 + k * step)) / step);
                    value += w * static_cast<double>(sharp[static_cast<std::size_t>(k)][i]);
                }
                worst = std::max(worst, std::abs(rolled[i] - static_cast<float>(value)));
                scale = std::max(scale, std::abs(rolled[i]));
            }
        }
    }
    EXPECT_LT(worst / scale, 2e-3f);
    EXPECT_GT(RelativeError(rolled, sharp.back()), 0.1f) << "the top rows lag behind";
}

TEST_F(RasterDelegateTest, MotionBlurBlendsTheClearColourByCoverage) {
    RasterDelegate delegate(*context_);
    Scene scene = MotionScene();
    ernie::RenderSettings(scene, "/Render/Settings").SetClearColor(glm::dvec3(0.0, 0.0, 1.0));
    SetShutter(scene, -1.0, 0.0, 6);
    scene.Set("/World/Camera", tokens::kTransform, CameraAt(0.0));
    Step(delegate, scene, "LdrColor");
    scene.Set("/World/Camera", tokens::kTransform, CameraAt(6.0));
    const std::vector<float> ldr = Step(delegate, scene, "LdrColor");
    // A corner far from every prim shows the clear colour, as without motion.
    EXPECT_EQ(ldr[0], 0.0f);
    EXPECT_EQ(ldr[1], 0.0f);
    EXPECT_EQ(ldr[2], 255.0f);
    // Some pixels partly covered over the exposure mix the quad's red with the blue.
    std::size_t mixed = 0;
    for (std::size_t i = 0; i < ldr.size(); i += 4) {
        mixed += ldr[i] > 20.0f && ldr[i + 2] > 20.0f ? 1u : 0u;
    }
    EXPECT_GT(mixed, 0u);
}
