// SPDX-License-Identifier: BSD-3-Clause
// NASA's Blue Marble, a 21600 x 10800 image of the Earth, on a globe:
// fetched once into the cache, loaded off the main thread (a grey
// checker meanwhile), compressed to BC7 on the GPU and streamed as a
// virtual texture, its tiles sharpening as the camera comes down.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kEarthRadius = 6371.0088e3;

// Blue Marble Next Generation, August 2004, with topography and
// bathymetry (NASA Earth Observatory, public domain).
constexpr const char* kUrl =
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/"
    "world.topo.bathy.200408.3x21600x10800.jpg";

const std::array<std::string, 3> kLoading = {"auto", "async",
                                             "complete"};

/// The point at `latitude` and `longitude` (degrees, east of +x) on a
/// sphere of radius `radius` about the origin, z north.
glm::dvec3 At(double latitude, double longitude, double radius) {
    const double phi = latitude * kPi / 180.0;
    const double lambda = longitude * kPi / 180.0;
    return radius * glm::dvec3(std::cos(phi) * std::cos(lambda),
                               std::cos(phi) * std::sin(lambda),
                               std::sin(phi));
}

/// A stand-in Earth for the checks, which run offline: a lat-long
/// image wider than a virtual texture's side (so it streams as the
/// real one does), each quarter of longitude its own colour, west to
/// east red, green, blue and yellow, darker towards the poles.
std::filesystem::path StandInEarth() {
    constexpr std::int64_t kCols = 8400;
    constexpr std::int64_t kRows = 4200;
    constexpr std::array<std::array<std::uint8_t, 3>, 4> kQuarters = {
        {{200, 40, 40},
         {40, 200, 40},
         {40, 40, 200},
         {200, 200, 40}}};
    std::vector<std::uint8_t> texels(kCols * kRows * 4);
    for (std::int64_t row = 0; row < kRows; ++row) {
        const double shade =
            1.0 -
            0.5 * std::abs(static_cast<double>(row) / kRows - 0.5);
        for (std::int64_t col = 0; col < kCols; ++col) {
            const auto& colour =
                kQuarters[static_cast<std::size_t>(col * 4 / kCols)];
            std::uint8_t* texel = &texels[static_cast<std::size_t>(
                (row * kCols + col) * 4)];
            for (std::size_t c = 0; c < 3; ++c) {
                texel[c] =
                    static_cast<std::uint8_t>(colour[c] * shade);
            }
            texel[3] = 255;
        }
    }
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "ernie_demo_blue_marble_stand_in.png";
    app::WritePng(
        path, Tensor::Copy<std::uint8_t>(texels, {kRows, kCols, 4}));
    return path;
}

class BlueMarbleDemo : public app::Demo {
  public:
    BlueMarbleDemo(app::Viewer* viewer,
                   const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The globe: a cube sphere with texture coordinates in
        // latitude and longitude, ocean blue until its image comes.
        Sphere::Define(scene, "/World/Earth")
            .SetRadius(kEarthRadius)
            .SetTessellation(std::string(tokens::kCubeSphere))
            .SetTessellationLevel(192)
            .SetMaterialBinding("/World/Looks/Earth");
        Material::Define(scene, "/World/Looks/Earth");
        Shader::Define(scene, "/World/Looks/Earth/Surface")
            .SetInfoId(tokens::kUsdPreviewSurface)
            .SetDiffuseColor(glm::dvec3(0.02, 0.05, 0.12))
            .SetRoughness(0.8);
        Connect(scene, "/World/Looks/Earth", tokens::kSurfaceOutput,
                "/World/Looks/Earth/Surface", tokens::kSurfaceOutput);
        Shader::Define(scene, "/World/Looks/Earth/St")
            .SetInfoId(tokens::kUsdPrimvarReaderFloat2)
            .SetVarname("st");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth))
            .SetAnchor(std::string(tokens::kAnchorCentre));
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53)
            .SetShadowEnable(false)
            .SetTransform(DistantLightTransform(At(10.0, 30.0, 1.0)));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetClippingRange(glm::dvec2(
                1000.0, std::numeric_limits<double>::infinity()))
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 500.0);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({1280, 720})
            .SetAmbient(0.0);

        // The image: the stand-in for the checks; else the cached
        // Blue Marble, fetched in the background the first time.
        image_ = demos::CacheDir() / "blue_marble" /
                 "world.topo.bathy.200408.3x21600x10800.jpg";
        if (options.test) {
            UseImage(StandInEarth());
        } else if (std::filesystem::exists(image_)) {
            UseImage(image_);
        } else if (demos::CanDownload()) {
            download_ =
                std::make_unique<demos::AsyncDownload>(kUrl, image_);
            progress_ = Widgets().Progress("Download");
        } else {
            Report("Blue Marble is not cached and cannot be fetched",
                   true);
        }

        Widgets().Slider("Altitude (km)", 200.0, 40000.0, altitude_,
                         [this](double km) {
                             altitude_ = km;
                             Place();
                         });
        Widgets().Slider("Latitude", -80.0, 80.0, latitude_,
                         [this](double degrees) {
                             latitude_ = degrees;
                             Place();
                         });
        Widgets().Slider("Longitude", -180.0, 180.0, longitude_,
                         [this](double degrees) {
                             longitude_ = degrees;
                             Place();
                         });
        Widgets().Combo("Texture loading",
                        std::vector<std::string>(kLoading.begin(),
                                                 kLoading.end()),
                        0, [this](std::size_t index) {
                            RenderSettings(GetScene(),
                                           "/Render/Settings")
                                .SetTextureLoading(kLoading[index]);
                        });
        // Magenta while loading; green to red by how coarse its tiles
        // are.
        Widgets().Checkbox(
            "Texture loading view", false, [this](bool on) {
                RenderSettings(GetScene(), "/Render/Settings")
                    .SetTextureLoadingDebug(on);
            });
        // Back to a first load: the GPU's textures go, and with the
        // cache's copies decoding and compression start over too.
        Widgets().Button("Evict textures", [this] { Evict(false); });
        Widgets().Button("Evict textures and cache",
                         [this] { Evict(true); });
        Place();
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            // The Sun's gizmo would sit at the Earth's centre.
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            viewer->SetViewportDisplay(display);
        }
    }

    void Step() override {
        // The download's news in the log, and the image once fetched.
        if (download_ != nullptr) {
            for (const auto& message : download_->TakeMessages()) {
                Report(message.text, message.error);
            }
            Widgets().Set(progress_, download_->Fraction());
            Widgets().Set(progress_, download_->Status());
            if (download_->Done()) {
                if (download_->Succeeded()) {
                    UseImage(image_);
                }
                download_.reset();
                Widgets().Remove(progress_);
            }
        }
    }

    void TestFinal() override {
        // Over the blue and the green quarters of the stand-in, from
        // 20 000 km: the globe shows that quarter's colour.
        altitude_ = 20000.0;
        latitude_ = 0.0;
        const auto centre = [this](double longitude) {
            longitude_ = longitude;
            Place();
            const Tensor image = GetRenderer().Render(GetScene());
            const std::int64_t rows = image.Shape()[0];
            const std::int64_t cols = image.Shape()[1];
            const auto* p =
                static_cast<const std::uint8_t*>(image.Data()) +
                ((rows / 2) * cols + cols / 2) * 4;
            return glm::dvec3(p[0], p[1], p[2]);
        };
        const glm::dvec3 blue = centre(45.0);
        const glm::dvec3 green = centre(-45.0);
        std::printf("Centre over 45 E: %.0f %.0f %.0f; 45 W: %.0f "
                    "%.0f %.0f\n",
                    blue.r, blue.g, blue.b, green.r, green.g,
                    green.b);
        Check(blue.b > blue.r && blue.b > blue.g,
              "the blue quarter shows east of the meridian");
        Check(green.g > green.r && green.g > green.b,
              "the green quarter shows west of it");
    }

  private:
    /// Textures the globe with `path`, read as sRGB colour.
    void UseImage(const std::filesystem::path& path) {
        Scene& scene = GetScene();
        Shader::Define(scene, "/World/Looks/Earth/Image")
            .SetInfoId(tokens::kUsdUVTexture)
            .SetSourceColorSpace("sRGB")
            .SetFile(path.string());
        Connect(scene, "/World/Looks/Earth/Image", tokens::kSt,
                "/World/Looks/Earth/St", "outputs:result");
        Connect(scene, "/World/Looks/Earth/Surface",
                tokens::kDiffuseColor, "/World/Looks/Earth/Image",
                "outputs:rgb");
    }

    /// The camera `altitude_` over `latitude_` and `longitude_`,
    /// looking down at the globe's centre, north up.
    void Place() {
        const glm::dvec3 eye =
            At(latitude_, longitude_, kEarthRadius + altitude_ * 1e3);
        Camera(GetScene(), "/World/Camera")
            .SetTransform(LookAtTransform(eye, {0.0, 0.0, 0.0},
                                          {0.0, 0.0, 1.0}));
    }

    /// Drops the textures (`RenderDelegate::EvictTextures`) of the
    /// viewport's renderer, or the demo's own without a viewer.
    void Evict(bool cache) {
        if (app::Viewer* viewer = GetViewer()) {
            viewer->Delegate().EvictTextures(cache);
        } else {
            GetRenderer().Delegate().EvictTextures(cache);
        }
        Report(cache ? "Textures and their cache evicted"
                     : "Textures evicted",
               false);
    }

    /// A message in the viewer's log, else printed.
    void Report(const std::string& text, bool error) {
        if (app::Viewer* viewer = GetViewer()) {
            viewer->Log(error ? app::LogLevel::Error
                              : app::LogLevel::Info,
                        text);
        } else {
            std::printf("%s\n", text.c_str());
        }
    }

    std::filesystem::path image_;
    std::unique_ptr<demos::AsyncDownload> download_;
    app::WidgetPanel::Id progress_ = 0;
    double altitude_ = 12000.0;  ///< km
    double latitude_ = 20.0;
    double longitude_ = 10.0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    BlueMarbleDemo, "textures", "blue_marble",
    "NASA's 21600 x 10800 Blue Marble on a globe, streamed "
    "as a virtual texture after loading off the main "
    "thread.");
