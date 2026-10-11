// SPDX-License-Identifier: BSD-3-Clause
// Structured light: a 450 nm projector shows a six-bit Gray code, one
// bit plane per frame of its sequence (`ernie:laser:sequence:count`),
// to a mono camera beside it. Each frame of the sensor clock
// shows the next plane; the six frames decode to the projector column
// each pixel sees, from which a scanner would triangulate depth.
//
// References:
// - J. Salvi, J. Pages and J. Batlle, "Pattern codification
// strategies
//   in structured light systems", Pattern Recognition 37(4), 2004.
// - J. L. Posdamer and M. D. Altschuler, "Surface measurement by
//   space-encoded projected beam systems", CGIP 18(1), 1982.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kBits = 6;
constexpr int kColumns = 1 << kBits;
constexpr double kPeriod = 1.0 / 15.0;  // s each plane shows
constexpr double kExposure = 1.0 / 60.0;

/// The Gray code's bit planes, most significant first, stacked top to
/// bottom in one image of `kColumns` x 4 texels each; returns its
/// path in the demos' cache.
std::filesystem::path GrayCode() {
    constexpr std::int64_t kRows = 4;
    Tensor image =
        Tensor::Zeros(DType::UInt8, {kBits * kRows, kColumns, 3});
    auto* data = static_cast<std::uint8_t*>(image.Data());
    for (int plane = 0; plane < kBits; ++plane) {
        for (int col = 0; col < kColumns; ++col) {
            const int gray = col ^ (col >> 1);
            const bool on = ((gray >> (kBits - 1 - plane)) & 1) != 0;
            for (std::int64_t row = 0; row < kRows; ++row) {
                for (int k = 0; k < 3; ++k) {
                    data[((plane * kRows + row) * kColumns + col) *
                             3 +
                         k] = on ? 255 : 0;
                }
            }
        }
    }
    const std::filesystem::path path =
        demos::CacheDir() / "laser_gray_code.png";
    std::filesystem::create_directories(path.parent_path());
    app::WritePng(path, image);
    return path;
}

void Box(Scene& scene, const Path& path, glm::dvec3 centre,
         glm::dvec3 size, glm::dvec3 colour) {
    Cube::Define(scene, path).SetSize(1.0).SetDisplayColor(colour);
    SetXform(scene, path, {.translate = centre, .scale = size});
}

class StructuredLightDemo : public app::Demo {
  public:
    StructuredLightDemo(app::Viewer* viewer,
                        const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = kBits;
        fixedDt = kPeriod;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Floor")
            .SetChecker(true)
            .SetCheckerColor0(glm::dvec3(0.6))
            .SetCheckerColor1(glm::dvec3(0.55));
        Box(scene, "/World/Wall", {0.0, 1.2, 1.0}, {3.0, 0.1, 2.0},
            glm::dvec3(0.7));
        Box(scene, "/World/Block", {-0.35, 0.4, 0.2}, {0.4, 0.4, 0.4},
            glm::dvec3(0.7));
        Sphere::Define(scene, "/World/Ball")
            .SetRadius(0.25)
            .SetDisplayColor(glm::dvec3(0.7));
        SetXform(scene, "/World/Ball",
                 {.translate = glm::dvec3(0.4, 0.3, 0.25)});
        // The projector: 40 mW at 450 nm, a DLP scanner's blue, its
        // six planes in turn, each shown for 1/15 s.
        LaserLight::Define(scene, "/World/Projector")
            .SetWavelength(450.0)
            .SetPower(0.04)
            .SetWaistRadius(glm::dvec2(0.001))
            .SetPattern("image")
            .SetPatternImage(GrayCode().string())
            .SetPatternFieldOfView({50.0, 40.0})
            .SetSequenceCount(kBits)
            .SetSequencePeriod(kPeriod)
            .SetTransform(LookAtTransform(
                {0.3, -1.8, 1.2}, {0.0, 0.6, 0.4}, {0.0, 0.0, 1.0}));
        // A mono camera 30 cm to the projector's left, exposing 1/60
        // s within each plane.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(20.0)
            .SetSensorNoise(false)
            .SetSensorType("mono")
            .SetExposureTime(kExposure)
            .SetExposureIso(3200.0)
            .SetExposureFStop(2.0)
            .SetTransform(LookAtTransform(
                {0.0, -1.8, 1.2}, {0.0, 0.6, 0.4}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({320, 240})
            .SetAmbient(0.0);
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    /// The six planes as the camera saw them, then the decoded
    /// columns as grey, side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        std::vector<Tensor> images = Planes();
        images.push_back(Decoded(images));
        app::WritePng(path, demos::SideBySide(images));
    }

    void TestFinal() override {
        const std::vector<Tensor> planes = Planes();
        const Tensor decoded = Decoded(planes);
        // Along the middle row, the columns the projector lit, left
        // to right: increasing, as the projector's x is the camera's.
        const auto rows = decoded.Shape()[0];
        const auto cols = decoded.Shape()[1];
        const auto channels = decoded.Shape()[2];
        const auto* data =
            static_cast<const std::uint8_t*>(decoded.Data());
        std::vector<int> seen;
        for (std::int64_t c = 0; c < cols; ++c) {
            const std::uint8_t value =
                data[((rows / 2) * cols + c) * channels];
            if (data[((rows / 2) * cols + c) * channels + 1] != 0) {
                seen.push_back(value * (kColumns - 1) / 255);
            }
        }
        int forward = 0;
        int backward = 0;
        for (std::size_t i = 1; i < seen.size(); ++i) {
            forward += seen[i] > seen[i - 1] ? 1 : 0;
            backward += seen[i] < seen[i - 1] ? 1 : 0;
        }
        const auto distinct = static_cast<int>(
            std::set<int>(seen.begin(), seen.end()).size());
        std::printf(
            "middle row: %zu pixels lit, %d columns, %d steps "
            "up and %d down\n",
            seen.size(), distinct, forward, backward);
        Check(distinct >= kColumns / 2,
              "the middle row should see most of the projector's "
              "columns");
        Check(backward * 5 < forward,
              "the decoded columns should rise from left to right");
    }

  private:
    /// The camera's frame in the middle of each plane's showing,
    /// uint8 (rows, cols, 4).
    std::vector<Tensor> Planes() {
        RenderDelegate& delegate = GetRenderer().Delegate();
        const double now = delegate.Time();
        std::vector<Tensor> images;
        for (int plane = 0; plane < kBits; ++plane) {
            delegate.ResetTime((plane + 0.75) * kPeriod);
            images.push_back(GetRenderer().Render(GetScene()));
        }
        delegate.ResetTime(now);
        return images;
    }

    /// Each pixel's projector column from the planes: a bit set where
    /// a plane is brighter than half the pixel's brightest, the Gray
    /// code to binary, as opaque grey (R = G = B = column / 63 x 255)
    /// with G 0 where nothing lights the pixel.
    static Tensor Decoded(const std::vector<Tensor>& planes) {
        const auto rows = planes[0].Shape()[0];
        const auto cols = planes[0].Shape()[1];
        const auto channels = planes[0].Shape()[2];
        Tensor out = Tensor::Zeros(DType::UInt8, {rows, cols, 4});
        auto* data = static_cast<std::uint8_t*>(out.Data());
        for (std::int64_t i = 0; i < rows * cols; ++i) {
            data[i * 4 + 3] = 255;
            int brightest = 0;
            for (const Tensor& plane : planes) {
                brightest = std::max(
                    brightest,
                    static_cast<int>(static_cast<const std::uint8_t*>(
                        plane.Data())[i * channels]));
            }
            if (brightest < 32) {
                continue;
            }
            int gray = 0;
            for (const Tensor& plane : planes) {
                const int value = static_cast<const std::uint8_t*>(
                    plane.Data())[i * channels];
                gray = (gray << 1) | (2 * value > brightest ? 1 : 0);
            }
            int column = 0;
            for (int g = gray; g != 0; g >>= 1) {
                column ^= g;
            }
            const auto grey = static_cast<std::uint8_t>(
                column * 255 / (kColumns - 1));
            data[i * 4] = grey;
            data[i * 4 + 1] = std::max<std::uint8_t>(grey, 1);
            data[i * 4 + 2] = grey;
        }
        return out;
    }
};

}  // namespace

ERNIE_REGISTER_DEMO(
    StructuredLightDemo, "lasers", "structured_light",
    "A Gray-code projector sequence decoded to projector columns.");
