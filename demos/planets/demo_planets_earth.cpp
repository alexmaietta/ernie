// SPDX-License-Identifier: BSD-3-Clause
// The whole Earth: an elevation model's heights paged in over the
// WGS84 ellipsoid under its air, coloured by NASA's Blue Marble, one
// image or its eight 21600 x 21600 tiles joined (86400 x 43200, past
// a GPU image's side), baked a band at a time to BC7 on the GPU and
// streamed as a virtual texture. The panel picks each dataset,
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

#include <algorithm>
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
// WGS84's ellipsoid: equatorial and polar radii.
constexpr double kEquator = 6378137.0;
constexpr double kPole = 6356752.314245;

/// A dataset: its files' URLs; the file read (the first's, else a
/// `.mosaic` of `rows` written beside them); and its heights' unit.
struct Dataset {
    const char* label;
    std::vector<std::string> urls;
    const char* mosaic = nullptr;
    std::vector<std::string> rows;
    double unit = 1.0;
};

// NASA Earth Observatory's Blue Marble Next Generation (August 2004,
// with topography and bathymetry) and its GEBCO land relief (0 to
// 6400 m in 8 bits); NOAA NCEI's ETOPO 2022 (land and sea floor in
// metres).
constexpr const char* kRecord =
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/";
constexpr const char* kEtopo =
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/";

/// Blue Marble's tiles, A to D west to east from 180 W, 1 the
/// northern row.
std::vector<std::string> Tiles() {
    std::vector<std::string> urls;
    for (const char* tile :
         {"A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"}) {
        urls.push_back(
            std::string(kRecord) +
            "73776/world.topo.bathy.200408.3x21600x21600." + tile +
            ".jpg");
    }
    return urls;
}

const std::array<Dataset, 3> kHeights = {{
    {"GEBCO land, 8-bit (18 MB)",
     {std::string(kRecord) +
      "73934/gebco_08_rev_elev_21600x10800.png"},
     nullptr,
     {},
     6400.0 / 255.0},
    {"ETOPO 2022 60\" (466 MB)",
     {std::string(kEtopo) + "60s/60s_surface_elev_gtif/"
                            "ETOPO_2022_v1_60s_N90W180_surface.tif"}},
    {"ETOPO 2022 30\" (1.6 GB)",
     {std::string(kEtopo) + "30s/30s_surface_elev_gtif/"
                            "ETOPO_2022_v1_30s_N90W180_surface.tif"}},
}};

const std::array<Dataset, 2> kColours = {{
    {"Blue Marble 21600 (27 MB)",
     {std::string(kRecord) +
      "73776/world.topo.bathy.200408.3x21600x10800.jpg"}},
    {"Blue Marble 8 tiles, 86400 (420 MB)",
     Tiles(),
     "world.topo.bathy.200408.3x21600x21600.mosaic",
     {"world.topo.bathy.200408.3x21600x21600.A1.jpg "
      "world.topo.bathy.200408.3x21600x21600.B1.jpg "
      "world.topo.bathy.200408.3x21600x21600.C1.jpg "
      "world.topo.bathy.200408.3x21600x21600.D1.jpg",
      "world.topo.bathy.200408.3x21600x21600.A2.jpg "
      "world.topo.bathy.200408.3x21600x21600.B2.jpg "
      "world.topo.bathy.200408.3x21600x21600.C2.jpg "
      "world.topo.bathy.200408.3x21600x21600.D2.jpg"}},
}};

/// The unit vector at `latitude` and `longitude` (degrees, east of
/// +x), z north.
glm::dvec3 Direction(double latitude, double longitude) {
    const double phi = latitude * kPi / 180.0;
    const double lambda = longitude * kPi / 180.0;
    return {std::cos(phi) * std::cos(lambda),
            std::cos(phi) * std::sin(lambda), std::sin(phi)};
}

/// Where file `name` (a URL's last part) is cached: Blue Marble's
/// beside the earlier demo's.
std::filesystem::path CachedPath(const std::string& name) {
    const std::string file =
        std::filesystem::path(name).filename().string();
    return demos::CacheDir() /
           (file.starts_with("world.topo") ? "blue_marble"
                                           : "earth") /
           file;
}

/// The file a dataset is read from.
std::filesystem::path ReadPath(const Dataset& dataset) {
    return CachedPath(dataset.mosaic != nullptr
                          ? dataset.mosaic
                          : dataset.urls.front());
}

/// A stand-in Earth for the checks, which run offline: heights in a
/// PDS image (a little finer than a face of 1024 quads, so a depth is
/// paged) with a mountain 8 km high and 60 km across at 28 N 87 E;
/// and colour as Blue Marble's tiles, eight joined (a `.mosaic`),
/// each quarter of longitude its own colour, west to east red, green,
/// blue and yellow.
std::pair<std::filesystem::path, std::filesystem::path>
StandInEarth() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "ernie_demo_earth";
    std::filesystem::create_directories(dir);
    constexpr std::int64_t kCols = 4608;
    constexpr std::int64_t kRows = 2304;
    const glm::dvec3 summit = Direction(28.0, 87.0);
    std::vector<std::int16_t> samples(kCols * kRows);
    for (std::int64_t row = 0; row < kRows; ++row) {
        const double latitude =
            90.0 - (static_cast<double>(row) + 0.5) * 180.0 / kRows;
        for (std::int64_t col = 0; col < kCols; ++col) {
            const double longitude =
                -180.0 +
                (static_cast<double>(col) + 0.5) * 360.0 / kCols;
            const double r =
                std::acos(std::min(
                    1.0, glm::dot(Direction(latitude, longitude),
                                  summit))) *
                kEquator / 30000.0;
            samples[static_cast<std::size_t>(row * kCols + col)] =
                static_cast<std::int16_t>(
                    std::lround(r < 1.0 ? 8000.0 * (1.0 - r) : 0.0));
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
           "END_OBJECT = IMAGE\nOBJECT = IMAGE_MAP_PROJECTION\n"
           "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
           "  MAP_RESOLUTION = 12.8\n  CENTER_LATITUDE = 0\n"
           "  MAXIMUM_LATITUDE = 90\n  WESTERNMOST_LONGITUDE = -180\n"
           "END_OBJECT = IMAGE_MAP_PROJECTION\nEND\n";

    constexpr std::int64_t kSide = 1050;
    constexpr std::array<std::array<std::uint8_t, 3>, 4> kQuarters = {
        {{200, 40, 40},
         {40, 200, 40},
         {40, 40, 200},
         {200, 200, 40}}};
    std::ofstream mosaic(dir / "stand_in.mosaic");
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 4; ++column) {
            const auto& colour =
                kQuarters[static_cast<std::size_t>(column)];
            std::vector<std::uint8_t> tile(kSide * kSide * 4);
            for (std::size_t i = 0; i < tile.size(); i += 4) {
                tile[i] = colour[0];
                tile[i + 1] = colour[1];
                tile[i + 2] = colour[2];
                tile[i + 3] = 255;
            }
            const std::string name = "tile" + std::to_string(row) +
                                     std::to_string(column) + ".png";
            app::WritePng(dir / name, Tensor::Copy<std::uint8_t>(
                                          tile, {kSide, kSide, 4}));
            mosaic << name << (column < 3 ? " " : "\n");
        }
    }
    return {heights, dir / "stand_in.mosaic"};
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

class EarthDemo : public app::Demo {
  public:
    EarthDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // A smooth Earth until its heights come.
        Terrain::Define(scene, "/World/Earth")
            .SetHeights(Tensor::Copy<float>(
                std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
            .SetPlanetRadii(glm::dvec3(kEquator, kEquator, kPole))
            .SetMaterialBinding("/World/Looks/Earth");
        // Ocean blue until its colour comes.
        Material::Define(scene, "/World/Looks/Earth");
        Shader::Define(scene, "/World/Looks/Earth/Surface")
            .SetInfoId(tokens::kUsdPreviewSurface)
            .SetDiffuseColor(glm::dvec3(0.02, 0.05, 0.12))
            .SetRoughness(1.0);
        Connect(scene, "/World/Looks/Earth", tokens::kSurfaceOutput,
                "/World/Looks/Earth/Surface", tokens::kSurfaceOutput);
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth))
            .SetAnchor(std::string(tokens::kAnchorCentre));
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
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 500.0);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({1280, 720})
            .SetAmbient(0.0);

        if (options.test) {
            const auto [heights, colour] = StandInEarth();
            UseHeights(heights, 1.0);
            UseColour(colour);
        } else if (demos::CanDownload() ||
                   std::filesystem::exists(ReadPath(kHeights[0]))) {
            SelectHeights(0);
            SelectColour(0);
        } else {
            Report("The Earth's data is not cached and cannot be "
                   "fetched",
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
                             Terrain(GetScene(), "/World/Earth")
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
        slider("Altitude (km)", 1.0, 40000.0, altitude_);
        slider("Latitude", -89.0, 89.0, latitude_);
        slider("Longitude", -180.0, 180.0, longitude_);
        slider("Heading", 0.0, 360.0, heading_);
        slider("Tilt", 0.0, 85.0, tilt_);
        slider("Sun elevation", 0.0, 90.0, sunElevation_);
        slider("Sun azimuth", 0.0, 360.0, sunAzimuth_);
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
        for (std::unique_ptr<Fetch>* fetch : {&heights_, &colour_}) {
            if (*fetch != nullptr && (*fetch)->Step()) {
                fetch->reset();
            }
        }
    }

    void TestFinal() override {
        // From 20 000 km over the equator straight down: the
        // stand-in's blue quarter east of the meridian, its green one
        // west.
        const auto centre = [this](double longitude) {
            Place({20000.0, 0.0, longitude, 0.0, 0.0, 90.0, 0.0});
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

        // The mountain from 40 km up and 50 km south, the sun low in
        // the east: its western slopes in shadow, darker than the
        // plains around it by more than the air's light fills in.
        Place({40.0, 27.55, 87.0, 0.0, 55.0, 12.0, 100.0});
        const Tensor image = GetRenderer().Render(GetScene());
        const std::int64_t pixels =
            image.Shape()[0] * image.Shape()[1];
        const auto* p =
            static_cast<const std::uint8_t*>(image.Data());
        std::vector<int> brightness(static_cast<std::size_t>(pixels));
        for (std::int64_t i = 0; i < pixels; ++i) {
            brightness[static_cast<std::size_t>(i)] =
                p[i * 4] + p[i * 4 + 1] + p[i * 4 + 2];
        }
        std::sort(brightness.begin(), brightness.end());
        const double median = brightness[brightness.size() / 2];
        const double darkest = brightness[brightness.size() / 200];
        std::printf("Mountain view: darkest %.0f, median %.0f\n",
                    darkest, median);
        Check(median > 60.0, "the plains are lit");
        Check(darkest < 0.7 * median,
              "the mountain's western slopes are in shadow");
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

    /// The Himalaya from 150 km up and 250 km south of Everest, the
    /// sun low in the east.
    static View DefaultView() {
        return {150.0, 25.8, 87.0, 0.0, 60.0, 12.0, 100.0};
    }

    /// Reads heights dataset `index`, fetching it first if need be.
    void SelectHeights(std::size_t index) {
        const Dataset& dataset = kHeights[index];
        heights_ = std::make_unique<Fetch>(
            Widgets(), dataset, "Heights download",
            [this, &dataset] {
                UseHeights(ReadPath(dataset), dataset.unit);
            },
            [this](const std::string& text, bool error) {
                Report(text, error);
            });
    }

    /// Colours the Earth with colour dataset `index`, fetching it
    /// first if need be.
    void SelectColour(std::size_t index) {
        const Dataset& dataset = kColours[index];
        colour_ = std::make_unique<Fetch>(
            Widgets(), dataset, "Colour download",
            [this, &dataset] {
                if (dataset.mosaic != nullptr) {
                    std::ofstream mosaic(ReadPath(dataset));
                    for (const std::string& row : dataset.rows) {
                        mosaic << row << "\n";
                    }
                }
                UseColour(ReadPath(dataset));
            },
            [this](const std::string& text, bool error) {
                Report(text, error);
            });
    }

    void UseHeights(const std::filesystem::path& file, double unit) {
        Terrain(GetScene(), "/World/Earth")
            .SetHeightsFile(file.string())
            .SetHeightsUnit(unit)
            .SetHeightsReference(0.0);
    }

    /// The Earth's colour from `file`, in latitude and longitude.
    void UseColour(const std::filesystem::path& file) {
        Scene& scene = GetScene();
        Shader::Define(scene, "/World/Looks/Earth/Colour")
            .SetInfoId(tokens::kUsdUVTexture)
            .SetSourceColorSpace("sRGB")
            .SetFile(file.string());
        Connect(scene, "/World/Looks/Earth/Surface",
                tokens::kDiffuseColor, "/World/Looks/Earth/Colour",
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

ERNIE_REGISTER_DEMO(EarthDemo, "planets", "earth",
                    "The whole Earth: an elevation model's heights "
                    "paged in under its "
                    "air, coloured by Blue Marble (one image or its "
                    "eight tiles), each "
                    "dataset fetched as the panel picks it.");
