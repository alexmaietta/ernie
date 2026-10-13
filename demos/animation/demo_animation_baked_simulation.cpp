// SPDX-License-Identifier: BSD-3-Clause
// A simulation baked into time samples, then played on the timeline:
// boxes thrown up and bouncing (a PointInstancer's positions and
// orientations), a flag waving (a subdivision surface's points), a
// camera zooming (its focal length and transform) and a light pulsing
// (its intensity). The renderer blends the two samples either side of
// the time code on the GPU, from samples it uploads once.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kFrames =
    120;  ///< Time codes 0 to kFrames, 30 a second.
constexpr int kSubsteps = 4;
constexpr int kBodies = 48;
constexpr float kHalf = 0.1f;  ///< The boxes' half size, metres.
constexpr int kFlagX = 24;     ///< Quads along the flag.
constexpr int kFlagZ = 12;     ///< Quads up it.
const glm::dvec3 kFlagOrigin(-2.5, 1.5, 1.0);
constexpr double kPi = 3.14159265358979323846;

/// A float in [0, 1) from two ints, as the Python twin's.
double Hash01(std::uint32_t i, std::uint32_t k) {
    std::uint32_t x = i * 7919u + k * 104729u;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return static_cast<double>(x) / 4294967296.0;
}

struct Body {
    glm::vec3 p;
    glm::vec3 v;
    glm::vec4 q{0.0f, 0.0f, 0.0f, 1.0f};  ///< x, y, z, w.
    glm::vec3 w;
};

/// Ballistic with a bouncing floor; the spin turns the box (as the
/// Python twin's kernel).
void Integrate(Body& b, float dt, float restitution) {
    glm::vec3 v(b.v.x, b.v.y, b.v.z - 9.81f * dt);
    glm::vec3 p = b.p + v * dt;
    glm::vec3 w = b.w;
    if (p.z < 0.1f) {
        p.z = 0.1f;
        if (v.z < 0.0f) {
            v = glm::vec3(0.8f * v.x, 0.8f * v.y, -restitution * v.z);
            w = 0.8f * w;
        }
    }
    const glm::vec4 q = b.q;
    const float h = 0.5f * dt;
    const float x = q.x + h * (w.x * q.w + w.y * q.z - w.z * q.y);
    const float y = q.y + h * (w.y * q.w + w.z * q.x - w.x * q.z);
    const float z = q.z + h * (w.z * q.w + w.x * q.y - w.y * q.x);
    const float s = q.w + h * (-w.x * q.x - w.y * q.y - w.z * q.z);
    const float n = std::sqrt(x * x + y * y + z * z + s * s);
    b.p = p;
    b.v = v;
    b.q = glm::vec4(x / n, y / n, z / n, s / n);
    b.w = w;
}

/// The flag's rest points (x along, z up), 2 by 1 metres.
std::vector<float> FlagRest() {
    std::vector<float> points;
    for (int j = 0; j <= kFlagZ; ++j) {
        for (int i = 0; i <= kFlagX; ++i) {
            points.insert(points.end(),
                          {2.0f * static_cast<float>(i) / kFlagX,
                           0.0f,
                           1.0f * static_cast<float>(j) / kFlagZ});
        }
    }
    return points;
}

class BakedSimulationDemo : public app::Demo {
  public:
    BakedSimulationDemo(app::Viewer* viewer,
                        const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Stage(scene, Path::Root())
            .SetStartTimeCode(0.0)
            .SetEndTimeCode(kFrames)
            .SetTimeCodesPerSecond(30.0)
            .SetFramesPerSecond(30.0);
        DistantLight::Define(scene, "/World/Sun")
            .SetAngle(0.53)
            .SetShadowEnable(true)
            .SetTransform(LookAtTransform(
                {2.0, -3.0, 4.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
        SphereLight::Define(scene, "/World/Beacon")
            .SetRadius(0.1)
            .SetTransform(LookAtTransform(
                {0.0, 0.0, 3.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}));
        Material::Define(scene, "/Looks/Floor")
            .SetAlbedo(glm::dvec3(0.5));
        Material::Define(scene, "/Looks/Box")
            .SetAlbedo({0.8, 0.35, 0.1});
        Material::Define(scene, "/Looks/Flag")
            .SetAlbedo({0.1, 0.3, 0.8});
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerSize(1.0)
            .SetMaterialBinding(Path("/Looks/Floor"));
        PointInstancer::Define(scene, "/World/Bodies")
            .SetPrototypes(
                std::vector<Path>{Path("/World/Bodies/Box")});
        Cube::Define(scene, "/World/Bodies/Box")
            .SetSize(2.0 * kHalf)
            .SetMaterialBinding(Path("/Looks/Box"));
        Cylinder::Define(scene, "/World/Pole")
            .SetRadius(0.03)
            .SetHeight(2.2)
            .SetMaterialBinding(Path("/Looks/Floor"));
        XformOps pole;
        pole.translate =
            glm::dvec3(kFlagOrigin.x, kFlagOrigin.y, 1.1);
        SetXform(scene, Path("/World/Pole"), pole);
        const std::vector<float> rest = FlagRest();
        std::vector<std::int32_t> indices;
        for (int j = 0; j < kFlagZ; ++j) {
            for (int i = 0; i < kFlagX; ++i) {
                const int a = j * (kFlagX + 1) + i;
                indices.insert(
                    indices.end(),
                    {a, a + 1, a + kFlagX + 2, a + kFlagX + 1});
            }
        }
        const auto quads =
            static_cast<std::int64_t>(indices.size() / 4);
        Mesh::Define(scene, "/World/Flag")
            .SetSubdivisionScheme(std::string("catmullClark"))
            .SetMaterialBinding(Path("/Looks/Flag"))
            .SetPoints(Tensor::Copy<float>(
                rest,
                {static_cast<std::int64_t>(rest.size() / 3), 3}))
            .SetFaceVertexCounts(Tensor::Copy<std::int32_t>(
                std::vector<std::int32_t>(
                    static_cast<std::size_t>(quads), 4),
                {quads}))
            .SetFaceVertexIndices(Tensor::Copy<std::int32_t>(
                indices,
                {static_cast<std::int64_t>(indices.size())}));
        XformOps flag;
        flag.translate = kFlagOrigin;
        SetXform(scene, Path("/World/Flag"), flag);
        Camera::Define(scene, "/World/Camera")
            .SetClippingRange({0.05, 100.0})
            .SetSensorNoise(false)
            .SetExposureMode(std::string(tokens::kAuto));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({960, 540})
            .SetAmbient(0.1);
        Bake();

        // Re-bakes the simulation with the boxes' bounce.
        Widgets().Slider("Restitution", 0.1, 0.9, restitution_,
                         [this](double restitution) {
                             restitution_ =
                                 static_cast<float>(restitution);
                             Bake();
                         });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            viewer->GetTimeline().Play();
        }
    }

    /// Four moments of the simulation side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(
            path, demos::SideBySide(RenderAt({0, 20, 60, kFrames})));
    }

    void TestFinal() override {
        const Scene& scene = GetScene();
        std::set<std::string> animated;
        for (const Path& path : scene.AnimatedPrims()) {
            animated.insert(path.String());
        }
        for (const char* prim : {"/World/Bodies", "/World/Flag",
                                 "/World/Camera", "/World/Beacon"}) {
            Check(animated.contains(prim),
                  std::string(prim) + " should be sampled");
        }
        Check(
            scene.SamplesOf(Path("/World/Bodies"), tokens::kPositions)
                    .size() == kFrames + 1,
            "every frame should be a sample");
        // Values between samples interpolate.
        const double focal = *scene
                                  .GetAt(Path("/World/Camera"),
                                         tokens::kFocalLength, 5.0)
                                  .Get<double>();
        Check(std::abs(focal - (18.0 + 22.0 * 5.0 / kFrames)) < 1e-6,
              "the focal length should interpolate");
        const std::vector<Tensor> images = RenderAt({0, 12, 60});
        const double changed = Changed(images[0], images[2]);
        std::printf("pixels changed by frame 60: %.3f; mean %.1f\n",
                    changed, MeanRgb(images[2]));
        Check(changed > 0.02, "the scene should move");
        for (const Tensor& image : images) {
            Check(MeanRgb(image) > 5.0, "every moment should be lit");
        }
    }

  private:
    /// Simulates every frame and writes them as time samples; values
    /// sampled sparsely interpolate between.
    void Bake() {
        Scene& scene = GetScene();
        std::vector<Body> bodies(kBodies);
        for (int i = 0; i < kBodies; ++i) {
            const auto u = static_cast<std::uint32_t>(i);
            const double a = 2.0 * kPi * Hash01(u, 0);
            const double speed = 1.0 + 2.0 * Hash01(u, 1);
            const double up = 3.0 + 3.0 * Hash01(u, 2);
            Body& b = bodies[static_cast<std::size_t>(i)];
            b.p = glm::vec3(static_cast<float>(0.2 * std::cos(a)),
                            static_cast<float>(0.2 * std::sin(a)),
                            static_cast<float>(0.6 + 0.05 * (i % 8)));
            b.v = glm::vec3(static_cast<float>(speed * std::cos(a)),
                            static_cast<float>(speed * std::sin(a)),
                            static_cast<float>(up));
            for (int c = 0; c < 3; ++c) {
                b.w[c] = static_cast<float>(
                    8.0 *
                    (Hash01(u, 3 + static_cast<std::uint32_t>(c)) -
                     0.5));
            }
        }
        const std::vector<float> rest = FlagRest();
        const auto points =
            static_cast<std::int64_t>(rest.size() / 3);
        const float dt = 1.0f / (30.0f * kSubsteps);
        for (int frame = 0; frame <= kFrames; ++frame) {
            if (frame > 0) {
                for (int s = 0; s < kSubsteps; ++s) {
                    for (Body& b : bodies) {
                        Integrate(b, dt, restitution_);
                    }
                }
            }
            std::vector<float> positions, orientations;
            for (const Body& b : bodies) {
                positions.insert(positions.end(),
                                 {b.p.x, b.p.y, b.p.z});
                orientations.insert(orientations.end(),
                                    {b.q.x, b.q.y, b.q.z, b.q.w});
            }
            // The flag's cloth at this frame, fixed at its pole (x =
            // 0).
            const float t = static_cast<float>(frame) / 30.0f;
            std::vector<float> wave(rest.size());
            for (std::size_t i = 0; i < rest.size(); i += 3) {
                const float x = rest[i];
                const float z = rest[i + 2];
                const float u = x / 2.0f;
                wave[i] = x;
                wave[i + 1] =
                    0.18f * u * std::sin(5.0f * x - 7.0f * t) +
                    0.05f * u * std::sin(4.0f * z + 5.0f * t);
                wave[i + 2] = z;
            }
            const double time = frame;
            scene.WriteAt(
                Path("/World/Bodies"), tokens::kPositions, time,
                Tensor::Copy<float>(positions, {kBodies, 3}));
            scene.WriteAt(
                Path("/World/Bodies"), tokens::kOrientations, time,
                Tensor::Copy<float>(orientations, {kBodies, 4}));
            scene.WriteAt(Path("/World/Flag"), tokens::kPoints, time,
                          Tensor::Copy<float>(wave, {points, 3}));
        }
        // The camera orbits and zooms in, the beacon pulses.
        for (int frame = 0; frame <= kFrames; frame += 10) {
            const double t = static_cast<double>(frame) / kFrames;
            const double angle = 0.35 + 0.5 * t;
            const glm::dvec3 eye(8.0 * std::cos(angle),
                                 -8.0 * std::sin(angle),
                                 3.0 - 0.8 * t);
            scene.SetAt(Path("/World/Camera"), tokens::kTransform,
                        frame,
                        LookAtTransform(eye, {0.0, 0.3, 0.6},
                                        {0.0, 0.0, 1.0}));
            scene.SetAt(Path("/World/Camera"), tokens::kFocalLength,
                        frame, 18.0 + 22.0 * t);
            scene.SetAt(Path("/World/Beacon"), tokens::kIntensity,
                        frame, (frame / 10) % 2 == 0 ? 40.0 : 5.0);
        }
    }

    /// The scene at each time code, RGBA uint8; leaves the time as it
    /// was.
    std::vector<Tensor> RenderAt(const std::vector<double>& times) {
        Stage stage(GetScene(), Path::Root());
        const double current = stage.GetTimeCode();
        std::vector<Tensor> images;
        for (const double t : times) {
            stage.SetTimeCode(t);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        stage.SetTimeCode(current);
        return images;
    }

    /// The mean of an RGBA uint8 image's colour channels.
    static double MeanRgb(const Tensor& image) {
        const Tensor pixels = image.Contiguous();
        const auto* bytes =
            static_cast<const std::uint8_t*>(pixels.Data());
        const auto count =
            static_cast<std::size_t>(pixels.Elements() / 4);
        double sum = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            sum += bytes[4 * i] + bytes[4 * i + 1] + bytes[4 * i + 2];
        }
        return sum / (3.0 * static_cast<double>(count));
    }

    /// The share of pixels whose colour changed by more than 16
    /// levels.
    static double Changed(const Tensor& a, const Tensor& b) {
        const Tensor pa = a.Contiguous();
        const Tensor pb = b.Contiguous();
        const auto* x = static_cast<const std::uint8_t*>(pa.Data());
        const auto* y = static_cast<const std::uint8_t*>(pb.Data());
        const auto count =
            static_cast<std::size_t>(pa.Elements() / 4);
        std::size_t changed = 0;
        for (std::size_t i = 0; i < count; ++i) {
            int most = 0;
            for (std::size_t c = 0; c < 3; ++c) {
                most = std::max(most, std::abs(int(x[4 * i + c]) -
                                               int(y[4 * i + c])));
            }
            changed += most > 16 ? 1 : 0;
        }
        return static_cast<double>(changed) /
               static_cast<double>(count);
    }

    float restitution_ = 0.6f;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    BakedSimulationDemo, "animation", "baked_simulation",
    "A simulation baked into time samples and played on the "
    "timeline, blended on the GPU.");
