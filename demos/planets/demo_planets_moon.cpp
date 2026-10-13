// SPDX-License-Identifier: BSD-3-Clause
// The whole Moon from LRO's data: LOLA's heights paged in from a
// latitude-longitude file as the view needs them, and LROC's colour
// mosaic baked a band at a time into the texture cache and wrapped by
// latitude and longitude. The panel picks each dataset's resolution,
// fetching it once into the cache; a low sun shows the relief.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kMoonRadius = 1737400.0;

/// A dataset: its files' URLs (the first is the one read; a PDS
/// label's image follows it), and its values' unit and reference.
struct Dataset {
    const char* label;
    std::vector<std::string> urls;
    double unit = 1.0;
    double reference = 0.0;
};

// NASA SVS's CGI Moon Kit (record 4720): LOLA heights in km and LROC
// WAC colour, both from 180 W; and LOLA's 128 pixel a degree grid
// from the PDS Geosciences Node, half metres about the Moon's radius.
constexpr const char* kKit =
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/";
constexpr const char* kLola =
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/cylindrical/img/";

const std::array<Dataset, 3> kHeights = {{
    {"LOLA 16 ppd (66 MB)",
     {std::string(kKit) + "ldem_16.tif"},
     1000.0},
    {"LOLA 64 ppd (1.1 GB)",
     {std::string(kKit) + "ldem_64.tif"},
     1000.0},
    {"LOLA 128 ppd (2.1 GB)",
     {std::string(kLola) + "ldem_128.lbl",
      std::string(kLola) + "ldem_128.img"},
     1.0,
     kMoonRadius},
}};

const std::array<Dataset, 3> kColours = {{
    {"LROC 8k (51 MB)",
     {std::string(kKit) + "lroc_color_poles_8k.tif"}},
    {"LROC 16k (187 MB)",
     {std::string(kKit) + "lroc_color_poles_16k.tif"}},
    {"LROC 27k (518 MB)",
     {std::string(kKit) + "lroc_color_poles.tif"}},
}};

/// The unit vector at `latitude` and `longitude` (degrees, east of
/// +x), z north.
glm::dvec3 Direction(double latitude, double longitude) {
    const double phi = latitude * kPi / 180.0;
    const double lambda = longitude * kPi / 180.0;
    return {std::cos(phi) * std::cos(lambda),
            std::cos(phi) * std::sin(lambda), std::sin(phi)};
}

/// Where dataset file `url` is cached.
std::filesystem::path CachedPath(const std::string& url) {
    return demos::CacheDir() / "moon" /
           std::filesystem::path(url).filename();
}

/// A stand-in Moon for the checks, which run offline: heights in a
/// PDS image as LOLA's 128 ppd grid holds them (a little finer than a
/// face of 1024 quads, so a depth is paged), rolling with a 40 km
/// crater at 43 S 11 W; and colour by quarter of longitude, west to
/// east red, green, blue and yellow.
std::pair<std::filesystem::path, std::filesystem::path>
StandInMoon() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "ernie_demo_moon";
    std::filesystem::create_directories(dir);
    constexpr std::int64_t kCols = 4608;
    constexpr std::int64_t kRows = 2304;
    const glm::dvec3 crater = Direction(-43.3, -11.2);
    std::vector<std::int16_t> samples(kCols * kRows);
    for (std::int64_t row = 0; row < kRows; ++row) {
        const double latitude =
            90.0 - (static_cast<double>(row) + 0.5) * 180.0 / kRows;
        for (std::int64_t col = 0; col < kCols; ++col) {
            const double longitude =
                (static_cast<double>(col) + 0.5) * 360.0 / kCols;
            const glm::dvec3 d = Direction(latitude, longitude);
            // A bowl 2 km deep within a rim 500 m high.
            const double r =
                std::acos(std::min(1.0, glm::dot(d, crater))) *
                kMoonRadius / 20000.0;
            const double bowl =
                r < 1.0 ? -2000.0 * (1.0 - r * r)
                        : 500.0 * std::exp(-4.0 * (r - 1.0));
            const double metres =
                1500.0 * std::sin(3.0 * latitude * kPi / 180.0) *
                    std::cos(2.0 * longitude * kPi / 180.0) +
                bowl;
            samples[static_cast<std::size_t>(row * kCols + col)] =
                static_cast<std::int16_t>(std::lround(2.0 * metres));
        }
    }
    const std::filesystem::path heights = dir / "stand_in.lbl";
    std::ofstream(dir / "stand_in.img", std::ios::binary)
        .write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * 2));
    std::ofstream(heights)
        << "PDS_VERSION_ID = PDS3\n^IMAGE = \"stand_in.img\"\n"
           "OBJECT = IMAGE\n  LINES = 2304\n  LINE_SAMPLES = 4608\n"
           "  SAMPLE_TYPE = LSB_INTEGER\n  SAMPLE_BITS = 16\n"
           "  SCALING_FACTOR = 0.5\n  OFFSET = 1737400\n"
           "END_OBJECT = IMAGE\nOBJECT = IMAGE_MAP_PROJECTION\n"
           "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
           "  MAP_RESOLUTION = 12.8\n  CENTER_LATITUDE = 0\n"
           "  MAXIMUM_LATITUDE = 90\n  WESTERNMOST_LONGITUDE = 0\n"
           "END_OBJECT = IMAGE_MAP_PROJECTION\nEND\n";

    constexpr std::int64_t kWidth = 2048;
    constexpr std::int64_t kHeight = 1024;
    constexpr std::array<std::array<std::uint8_t, 3>, 4> kQuarters = {
        {{200, 40, 40},
         {40, 200, 40},
         {40, 40, 200},
         {200, 200, 40}}};
    std::vector<std::uint8_t> texels(kWidth * kHeight * 4);
    for (std::int64_t row = 0; row < kHeight; ++row) {
        for (std::int64_t col = 0; col < kWidth; ++col) {
            const auto& colour =
                kQuarters[static_cast<std::size_t>(col * 4 / kWidth)];
            std::uint8_t* texel =
                &texels[static_cast<std::size_t>(row * kWidth + col) *
                        4];
            texel[0] = colour[0];
            texel[1] = colour[1];
            texel[2] = colour[2];
            texel[3] = 255;
        }
    }
    const std::filesystem::path colour = dir / "stand_in.png";
    app::WritePng(colour, Tensor::Copy<std::uint8_t>(
                              texels, {kHeight, kWidth, 4}));
    return {heights, colour};
}

/// A dataset's files fetched one after another into the cache, with a
/// progress bar while they come; `onDone` once all are there.
class Fetch {
  public:
    Fetch(app::WidgetPanel& widgets, const Dataset& dataset,
          std::string label, std::function<void()> onDone,
          std::function<void(const std::string&, bool)> report)
        : widgets_(widgets), onDone_(std::move(onDone)),
          report_(std::move(report)) {
        for (const std::string& url : dataset.urls) {
            if (!std::filesystem::exists(CachedPath(url))) {
                urls_.push_back(url);
            }
        }
        progress_ = widgets_.Progress(std::move(label));
    }
    ~Fetch() {
        widgets_.Remove(progress_);
    }
    Fetch(const Fetch&) = delete;
    Fetch& operator=(const Fetch&) = delete;

    /// Moves the fetch on; true once it has ended.
    bool Step() {
        if (download_ == nullptr && next_ < urls_.size()) {
            download_ = std::make_unique<demos::AsyncDownload>(
                urls_[next_], CachedPath(urls_[next_]));
        }
        if (download_ == nullptr) {
            onDone_();
            return true;
        }
        for (const auto& message : download_->TakeMessages()) {
            report_(message.text, message.error);
        }
        widgets_.Set(progress_, download_->Fraction());
        widgets_.Set(progress_, download_->Status());
        if (!download_->Done()) {
            return false;
        }
        const bool succeeded = download_->Succeeded();
        download_.reset();
        ++next_;
        return !succeeded;
    }

  private:
    app::WidgetPanel& widgets_;
    std::vector<std::string> urls_;
    std::size_t next_ = 0;
    std::unique_ptr<demos::AsyncDownload> download_;
    app::WidgetPanel::Id progress_ = 0;
    std::function<void()> onDone_;
    std::function<void(const std::string&, bool)> report_;
};

class MoonDemo : public app::Demo {
  public:
    MoonDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // A smooth Moon until its heights come.
        Terrain::Define(scene, "/World/Moon")
            .SetHeights(Tensor::Copy<float>(
                std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
            .SetPlanetRadius(kMoonRadius)
            .SetMaterialBinding("/World/Looks/Regolith");
        Material::Define(scene, "/World/Looks/Regolith");
        Shader::Define(scene, "/World/Looks/Regolith/Surface")
            .SetInfoId(tokens::kUsdPreviewSurface)
            .SetDiffuseColor(glm::dvec3(0.12))
            .SetRoughness(1.0);
        Connect(
            scene, "/World/Looks/Regolith", tokens::kSurfaceOutput,
            "/World/Looks/Regolith/Surface", tokens::kSurfaceOutput);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetClippingRange(glm::dvec2(
                10.0, std::numeric_limits<double>::infinity()))
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(11.0)
            .SetExposureTime(1.0 / 125.0);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({1280, 720})
            .SetAmbient(0.0);

        if (options.test) {
            const auto [heights, colour] = StandInMoon();
            UseHeights(heights, 1.0, kMoonRadius);
            UseColour(colour);
        } else if (demos::CanDownload() || Cached(kHeights[0])) {
            SelectHeights(0);
            SelectColour(0);
        } else {
            Report(
                "The Moon's data is not cached and cannot be fetched",
                true);
        }

        std::vector<std::string> heights;
        std::vector<std::string> colours;
        for (std::size_t i = 0; i < kHeights.size(); ++i) {
            heights.emplace_back(kHeights[i].label);
            colours.emplace_back(kColours[i].label);
        }
        Widgets().Combo("Heights", heights, 0,
                        [this](std::size_t i) { SelectHeights(i); });
        Widgets().Combo("Colour", colours, 0,
                        [this](std::size_t i) { SelectColour(i); });
        Widgets().Slider("Exaggeration", 1.0, 10.0, 1.0,
                         [this](double scale) {
                             Terrain(GetScene(), "/World/Moon")
                                 .SetExaggeration(scale);
                         });
        const auto slider = [this](const char* label, double min,
                                   double max, double& value) {
            Widgets().Slider(label, min, max, value,
                             [this, &value](double v) {
                                 value = v;
                                 Place();
                             });
        };
        slider("Altitude (km)", 1.0, 5000.0, altitude_);
        slider("Latitude", -89.0, 89.0, latitude_);
        slider("Longitude", -180.0, 180.0, longitude_);
        slider("Heading", 0.0, 360.0, heading_);
        slider("Tilt", 0.0, 85.0, tilt_);
        slider("Sun elevation", 0.0, 90.0, sunElevation_);
        slider("Sun azimuth", 0.0, 360.0, sunAzimuth_);
        Place();
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void Step() override {
        for (std::unique_ptr<Fetch>* fetch : {&heights_, &colour_}) {
            if (*fetch != nullptr && (*fetch)->Step()) {
                fetch->reset();
            }
        }
    }

    void TestFinal() override {
        // From 5000 km over the equator straight down: the stand-in's
        // blue quarter east of the meridian, its green one west.
        const auto centre = [this](double longitude) {
            altitude_ = 5000.0;
            latitude_ = 0.0;
            longitude_ = longitude;
            tilt_ = 0.0;
            sunElevation_ = 90.0;
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

        // The crater from 60 km under a sun 8 degrees up in the east:
        // its eastern wall in shadow, the plain around it lit.
        altitude_ = 60.0;
        latitude_ = -40.5;
        longitude_ = -11.2;
        heading_ = 180.0;
        tilt_ = 55.0;
        sunElevation_ = 8.0;
        sunAzimuth_ = 90.0;
        Place();
        const Tensor image = GetRenderer().Render(GetScene());
        const std::int64_t pixels =
            image.Shape()[0] * image.Shape()[1];
        const auto* p =
            static_cast<const std::uint8_t*>(image.Data());
        std::int64_t shadowed = 0;
        std::int64_t lit = 0;
        for (std::int64_t i = 0; i < pixels; ++i) {
            const int g = p[i * 4 + 1];
            shadowed += g < 16 ? 1 : 0;
            lit += g > 40 ? 1 : 0;
        }
        const double shadow = static_cast<double>(shadowed) /
                              static_cast<double>(pixels);
        const double light =
            static_cast<double>(lit) / static_cast<double>(pixels);
        std::printf("Crater view: %.1f%% in shadow, %.1f%% lit\n",
                    100.0 * shadow, 100.0 * light);
        Check(shadow > 0.005,
              "the crater's eastern wall is in shadow");
        Check(light > 0.5, "the plain around it is lit");
    }

  private:
    /// Whether every file of `dataset` is cached.
    static bool Cached(const Dataset& dataset) {
        for (const std::string& url : dataset.urls) {
            if (!std::filesystem::exists(CachedPath(url))) {
                return false;
            }
        }
        return true;
    }

    /// Reads heights dataset `index`, fetching it first if need be.
    void SelectHeights(std::size_t index) {
        const Dataset& dataset = kHeights[index];
        heights_ = std::make_unique<Fetch>(
            Widgets(), dataset, "Heights download",
            [this, &dataset] {
                UseHeights(CachedPath(dataset.urls.front()),
                           dataset.unit, dataset.reference);
            },
            [this](const std::string& text, bool error) {
                Report(text, error);
            });
    }

    /// Wraps colour dataset `index` over the Moon, fetching it first
    /// if need be.
    void SelectColour(std::size_t index) {
        const Dataset& dataset = kColours[index];
        colour_ = std::make_unique<Fetch>(
            Widgets(), dataset, "Colour download",
            [this, &dataset] {
                UseColour(CachedPath(dataset.urls.front()));
            },
            [this](const std::string& text, bool error) {
                Report(text, error);
            });
    }

    void UseHeights(const std::filesystem::path& file, double unit,
                    double reference) {
        Terrain(GetScene(), "/World/Moon")
            .SetHeightsFile(file.string())
            .SetHeightsUnit(unit)
            .SetHeightsReference(reference);
    }

    /// The Moon's colour from `file`, in latitude and longitude, its
    /// display stretch scaled to about the regolith's albedo.
    void UseColour(const std::filesystem::path& file) {
        Scene& scene = GetScene();
        Shader::Define(scene, "/World/Looks/Regolith/Colour")
            .SetInfoId(tokens::kUsdUVTexture)
            .SetSourceColorSpace("sRGB")
            .SetFile(file.string());
        scene.Set("/World/Looks/Regolith/Colour",
                  tokens::kShaderScale,
                  glm::dvec4(0.3, 0.3, 0.3, 1.0));
        Connect(scene, "/World/Looks/Regolith/Surface",
                tokens::kDiffuseColor, "/World/Looks/Regolith/Colour",
                "outputs:rgb");
    }

    /// The camera `altitude_` km over `latitude_` and `longitude_`,
    /// facing `heading_` (clockwise from north) `tilt_` degrees up
    /// from straight down; the sun at `sunElevation_` and
    /// `sunAzimuth_` there.
    void Place() {
        const glm::dvec3 up = Direction(latitude_, longitude_);
        const glm::dvec3 east =
            glm::normalize(glm::cross(glm::dvec3(0.0, 0.0, 1.0), up));
        const glm::dvec3 north = glm::cross(up, east);
        const auto towards = [&](double azimuth, double elevation) {
            const double a = azimuth * kPi / 180.0;
            const double e = elevation * kPi / 180.0;
            return std::cos(e) *
                       (std::cos(a) * north + std::sin(a) * east) +
                   std::sin(e) * up;
        };
        const glm::dvec3 eye = up * (kMoonRadius + altitude_ * 1e3);
        const glm::dvec3 forward = towards(heading_, tilt_ - 90.0);
        Camera(GetScene(), "/World/Camera")
            .SetTransform(LookAtTransform(eye, eye + forward, up));
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(DistantLightTransform(
                towards(sunAzimuth_, sunElevation_)));
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

    std::unique_ptr<Fetch> heights_;
    std::unique_ptr<Fetch> colour_;
    double altitude_ = 60.0;  ///< km
    double latitude_ = -40.5;
    double longitude_ = -11.2;
    double heading_ = 180.0;
    double tilt_ = 55.0;
    double sunElevation_ = 8.0;
    double sunAzimuth_ = 90.0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    MoonDemo, "planets", "moon",
    "The whole Moon from LRO: LOLA's heights paged in and LROC's "
    "colour baked into the texture cache, each at the resolution the "
    "panel picks.");
