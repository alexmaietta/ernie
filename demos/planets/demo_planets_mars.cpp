// SPDX-License-Identifier: BSD-3-Clause
// The whole of Mars from MGS's laser altimeter: MOLA's heights paged
// in from a latitude-longitude file (four of them joined at 64 pixels
// a degree) over the ellipsoid, under its dusty air; its colour plain
// or Viking's global mosaic, baked a band at a time into the texture
// cache. The panel picks each dataset, fetching it once into the
// cache; a low sun shows the relief.

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
// The IAU's ellipsoid of Mars: equatorial and polar radii.
constexpr double kEquator = 3396190.0;
constexpr double kPole = 3376200.0;
// The Sun's irradiance at Mars's mean distance, 1.5237 au.
constexpr double kIrradiance = 1361.0 / (1.5237 * 1.5237);

/// A dataset: its files' URLs; the file read (the first's, else a
/// `.mosaic` of `rows` written beside them); none for no files.
struct Dataset {
    const char* label;
    std::vector<std::string> urls;
    const char* mosaic = nullptr;
    std::vector<std::string> rows;
};

// MGS MOLA's Mission Experiment Gridded Data Records (PDS Geosciences
// Node): heights in metres above the areoid, big-endian, from 0 E;
// and USGS's Viking Orbiter global colour mosaic, 925 m a pixel, from
// 180 W.
constexpr const char* kMegdr =
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/"
    "mgsl_300x/";
constexpr const char* kViking =
    "https://planetarymaps.usgs.gov/mosaic/"
    "Mars_Viking_ClrMosaic_global_925m.tif";

/// MEGDR file `name`'s label and image in folder `folder`.
std::vector<std::string> Megdr(const std::string& folder,
                               const std::string& name) {
    return {std::string(kMegdr) + folder + "/" + name + ".lbl",
            std::string(kMegdr) + folder + "/" + name + ".img"};
}

std::vector<std::string>
Joined(std::vector<std::vector<std::string>> parts) {
    std::vector<std::string> all;
    for (std::vector<std::string>& part : parts) {
        all.insert(all.end(), part.begin(), part.end());
    }
    return all;
}

const std::array<Dataset, 3> kHeights = {{
    {"MOLA 16 ppd (33 MB)", Megdr("meg016", "megt90n000eb")},
    {"MOLA 32 ppd (133 MB)", Megdr("meg032", "megt90n000fb")},
    {"MOLA 64 ppd (531 MB)",
     Joined({Megdr("meg064", "megt90n000gb"),
             Megdr("meg064", "megt90n180gb"),
             Megdr("meg064", "megt00n000gb"),
             Megdr("meg064", "megt00n180gb")}),
     "megt_64.mosaic",
     {"megt90n000gb.lbl megt90n180gb.lbl",
      "megt00n000gb.lbl megt00n180gb.lbl"}},
}};

const std::array<Dataset, 2> kColours = {{
    {"Plain", {}},
    {"Viking colour (798 MB)", {kViking}},
}};

/// The unit vector at `latitude` and `longitude` (degrees, east of
/// +x), z north.
glm::dvec3 Direction(double latitude, double longitude) {
    const double phi = latitude * kPi / 180.0;
    const double lambda = longitude * kPi / 180.0;
    return {std::cos(phi) * std::cos(lambda),
            std::cos(phi) * std::sin(lambda), std::sin(phi)};
}

/// Where file `name` (a URL's last part) is cached.
std::filesystem::path CachedPath(const std::string& name) {
    return demos::CacheDir() / "mars" /
           std::filesystem::path(name).filename();
}

/// The file a dataset is read from.
std::filesystem::path ReadPath(const Dataset& dataset) {
    return CachedPath(dataset.mosaic != nullptr
                          ? dataset.mosaic
                          : dataset.urls.front());
}

/// A stand-in Mars for the checks, which run offline: heights in a
/// big-endian PDS image as MOLA's grids hold them (a little finer
/// than a face of 1024 quads, so a depth is paged), with a 25 km high
/// shield volcano 300 km across at 18.65 N 133.8 W; and colour by
/// quarter of longitude, west to east red, green, blue and yellow.
std::pair<std::filesystem::path, std::filesystem::path>
StandInMars() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "ernie_demo_mars";
    std::filesystem::create_directories(dir);
    constexpr std::int64_t kCols = 4608;
    constexpr std::int64_t kRows = 2304;
    const glm::dvec3 summit = Direction(18.65, -133.8);
    std::vector<std::uint8_t> samples(kCols * kRows * 2);
    for (std::int64_t row = 0; row < kRows; ++row) {
        const double latitude =
            90.0 - (static_cast<double>(row) + 0.5) * 180.0 / kRows;
        for (std::int64_t col = 0; col < kCols; ++col) {
            const double longitude =
                (static_cast<double>(col) + 0.5) * 360.0 / kCols;
            const double r =
                std::acos(std::min(
                    1.0, glm::dot(Direction(latitude, longitude),
                                  summit))) *
                kEquator / 150000.0;
            const double metres =
                2000.0 * std::sin(2.0 * latitude * kPi / 180.0) +
                (r < 1.0 ? 25000.0 * (1.0 - r) : 0.0);
            const auto value = static_cast<std::uint16_t>(
                static_cast<std::int16_t>(std::lround(metres)));
            const auto at =
                static_cast<std::size_t>(row * kCols + col) * 2;
            samples[at] = static_cast<std::uint8_t>(value >> 8);
            samples[at + 1] = static_cast<std::uint8_t>(value & 0xff);
        }
    }
    const std::filesystem::path heights = dir / "stand_in.lbl";
    std::ofstream(dir / "stand_in.img", std::ios::binary)
        .write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size()));
    std::ofstream(heights)
        << "PDS_VERSION_ID = PDS3\n^IMAGE = \"stand_in.img\"\n"
           "OBJECT = IMAGE\n  LINES = 2304\n  LINE_SAMPLES = 4608\n"
           "  SAMPLE_TYPE = MSB_INTEGER\n  SAMPLE_BITS = 16\n"
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
        widgets_.Set(progress_, std::to_string(next_ + 1) + " of " +
                                    std::to_string(urls_.size()) +
                                    ": " + download_->Status());
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

class MarsDemo : public app::Demo {
  public:
    MarsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // A smooth Mars until its heights come.
        Terrain::Define(scene, "/World/Mars")
            .SetHeights(Tensor::Copy<float>(
                std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
            .SetPlanetRadii(glm::dvec3(kEquator, kEquator, kPole))
            .SetMaterialBinding("/World/Looks/Soil");
        // Reddish soil, albedo about Mars's mean 0.25.
        Material::Define(scene, "/World/Looks/Soil");
        Shader::Define(scene, "/World/Looks/Soil/Surface")
            .SetInfoId(tokens::kUsdPreviewSurface)
            .SetDiffuseColor(glm::dvec3(0.33, 0.22, 0.14))
            .SetRoughness(1.0);
        Connect(scene, "/World/Looks/Soil", tokens::kSurfaceOutput,
                "/World/Looks/Soil/Surface", tokens::kSurfaceOutput);
        // Carbon dioxide and dust (feldspar with hematite).
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kMars))
            .SetAnchor(std::string(tokens::kAnchorCentre))
            .SetDustOpticalDepth(0.5)
            .SetGroundAlbedo(0.25);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(kIrradiance)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.35);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetClippingRange(glm::dvec2(
                10.0, std::numeric_limits<double>::infinity()))
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 125.0);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({1280, 720})
            .SetAmbient(0.0);

        if (options.test) {
            const auto [heights, colour] = StandInMars();
            UseHeights(heights);
            UseColour(colour);
        } else if (demos::CanDownload() ||
                   std::filesystem::exists(ReadPath(kHeights[0]))) {
            SelectHeights(0);
        } else {
            Report(
                "Mars's heights are not cached and cannot be fetched",
                true);
        }

        std::vector<std::string> heights;
        for (const Dataset& dataset : kHeights) {
            heights.emplace_back(dataset.label);
        }
        std::vector<std::string> colours;
        for (const Dataset& dataset : kColours) {
            colours.emplace_back(dataset.label);
        }
        Widgets().Combo("Heights", heights, 0,
                        [this](std::size_t i) { SelectHeights(i); });
        Widgets().Combo("Colour", colours, 0,
                        [this](std::size_t i) { SelectColour(i); });
        Widgets().Slider("Exaggeration", 1.0, 10.0, 1.0,
                         [this](double scale) {
                             Terrain(GetScene(), "/World/Mars")
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
        slider("Altitude (km)", 1.0, 10000.0, altitude_);
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
        // From 8000 km over the equator straight down: the stand-in's
        // blue quarter east of the meridian, its green one west.
        const auto centre = [this](double longitude) {
            altitude_ = 8000.0;
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

        // The volcano from 400 km south under a sun 10 degrees up in
        // the east: its western flank in shadow, the plains lit.
        Place(DefaultView());
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
        std::printf("Volcano view: %.1f%% in shadow, %.1f%% lit\n",
                    100.0 * shadow, 100.0 * light);
        Check(shadow > 0.005,
              "the volcano's western flank is in shadow");
        Check(light > 0.4, "the plains are lit");
    }

  private:
    /// A place of the camera and the sun.
    struct View {
        double altitude;  ///< km
        double latitude;
        double longitude;
        double heading;
        double tilt;
        double sunElevation;
        double sunAzimuth;
    };

    /// Olympus Mons from 300 km up and 400 km south, the sun low in
    /// the east.
    static View DefaultView() {
        return {300.0, 12.0, -133.8, 0.0, 55.0, 10.0, 90.0};
    }

    /// Reads heights dataset `index`, fetching it first if need be.
    void SelectHeights(std::size_t index) {
        const Dataset& dataset = kHeights[index];
        heights_ = std::make_unique<Fetch>(
            Widgets(), dataset, "Heights download",
            [this, &dataset] {
                if (dataset.mosaic != nullptr) {
                    std::ofstream mosaic(ReadPath(dataset));
                    for (const std::string& row : dataset.rows) {
                        mosaic << row << "\n";
                    }
                }
                UseHeights(ReadPath(dataset));
            },
            [this](const std::string& text, bool error) {
                Report(text, error);
            });
    }

    /// Colours Mars plain, or with colour dataset `index`, fetching
    /// it first if need be.
    void SelectColour(std::size_t index) {
        const Dataset& dataset = kColours[index];
        if (dataset.urls.empty()) {
            colour_.reset();
            GetScene().Clear("/World/Looks/Soil/Surface",
                             std::string(tokens::kDiffuseColor) +
                                 std::string(tokens::kConnectSuffix));
            return;
        }
        colour_ = std::make_unique<Fetch>(
            Widgets(), dataset, "Colour download",
            [this, &dataset] { UseColour(ReadPath(dataset)); },
            [this](const std::string& text, bool error) {
                Report(text, error);
            });
    }

    void UseHeights(const std::filesystem::path& file) {
        Terrain(GetScene(), "/World/Mars")
            .SetHeightsFile(file.string())
            .SetHeightsUnit(1.0)
            .SetHeightsReference(0.0);
    }

    /// Mars's colour from `file`, in latitude and longitude, its
    /// display stretch scaled to about the soil's albedo.
    void UseColour(const std::filesystem::path& file) {
        Scene& scene = GetScene();
        Shader::Define(scene, "/World/Looks/Soil/Colour")
            .SetInfoId(tokens::kUsdUVTexture)
            .SetSourceColorSpace("sRGB")
            .SetFile(file.string());
        scene.Set("/World/Looks/Soil/Colour", tokens::kShaderScale,
                  glm::dvec4(0.6, 0.6, 0.6, 1.0));
        Connect(scene, "/World/Looks/Soil/Surface",
                tokens::kDiffuseColor, "/World/Looks/Soil/Colour",
                "outputs:rgb");
    }

    void Place(const View& view) {
        altitude_ = view.altitude;
        latitude_ = view.latitude;
        longitude_ = view.longitude;
        heading_ = view.heading;
        tilt_ = view.tilt;
        sunElevation_ = view.sunElevation;
        sunAzimuth_ = view.sunAzimuth;
        Place();
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
        const glm::dvec3 eye = up * (kEquator + altitude_ * 1e3);
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
    double altitude_ = DefaultView().altitude;
    double latitude_ = DefaultView().latitude;
    double longitude_ = DefaultView().longitude;
    double heading_ = DefaultView().heading;
    double tilt_ = DefaultView().tilt;
    double sunElevation_ = DefaultView().sunElevation;
    double sunAzimuth_ = DefaultView().sunAzimuth;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    MarsDemo, "planets", "mars",
    "The whole of Mars from MOLA: its heights paged in under its "
    "dusty "
    "air, plain or in Viking's colour baked into the texture cache, "
    "each dataset fetched as the panel picks it.");
