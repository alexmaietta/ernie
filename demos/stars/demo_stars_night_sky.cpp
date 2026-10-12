// SPDX-License-Identifier: BSD-3-Clause
// Orion over Kitt Peak on a January night through the Earth's air:
// the stars coming out at dusk, high at night, dimmed and reddened
// setting, and without the air. The Bright Star Catalogue once
// fetched; offline, Orion's brightest among made-up stars.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/util/astronomy.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr std::size_t kWidth = 720;
constexpr std::size_t kHeight = 480;
constexpr double kLatitude = 31.96;  // Kitt Peak, degrees
constexpr double kLongitude = -111.6;
constexpr double kOrionRa = 83.8;  // degrees
constexpr double kOrionDec = -1.2;
constexpr const char* kBscUrl =
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-out.max=unlimited"
    "&-source=V/50/catalog&-out=_RAJ2000,_DEJ2000,Vmag,B-V";

/// Stars as rows of right ascension, declination, V and B-V.
using StarTable = std::vector<std::array<float, 4>>;

// Orion's brightest stars and neighbours (J2000 degrees, V, B-V).
const StarTable kBrightStars = {
    {88.7929f, 7.4071f, 0.42f, 1.85f},      // Betelgeuse
    {78.6345f, -8.2016f, 0.13f, -0.03f},    // Rigel
    {81.2828f, 6.3497f, 1.64f, -0.22f},     // Bellatrix
    {84.0534f, -1.2019f, 1.69f, -0.18f},    // Alnilam
    {85.1897f, -1.9426f, 1.77f, -0.21f},    // Alnitak
    {83.0017f, -0.2991f, 2.23f, -0.22f},    // Mintaka
    {86.9391f, -9.6696f, 2.09f, -0.17f},    // Saiph
    {83.7845f, 9.9342f, 3.39f, -0.16f},     // Meissa
    {83.8583f, -5.9099f, 2.77f, -0.24f},    // Hatysa
    {101.2872f, -16.7161f, -1.46f, 0.00f},  // Sirius
    {114.8255f, 5.2250f, 0.34f, 0.42f},     // Procyon
    {68.9802f, 16.5093f, 0.87f, 1.54f},     // Aldebaran
    {79.1723f, 45.9980f, 0.08f, 0.80f},     // Capella
    {113.6494f, 31.8883f, 1.58f, 0.03f},    // Castor
    {116.3290f, 28.0262f, 1.14f, 1.00f},    // Pollux
};

/// A view: hours UT on 16 January 2026, the sun's elevation, the
/// exposure and whether the air shows.
struct View {
    const char* name;
    double hours;
    double sun;
    double seconds;
    double iso;
    bool air;
};

const std::array<View, 5> kViews = {{
    {"Civil dusk, Orion rising", 1.0, -4.0, 1.0 / 15.0, 100.0, true},
    {"Nautical dusk", 1.5, -10.0, 2.0, 400.0, true},
    {"Night, Orion high", 4.5, -45.0, 15.0, 1600.0, true},
    {"Orion setting", 10.5, -45.0, 15.0, 1600.0, true},
    {"Without the air", 4.5, -45.0, 15.0, 1600.0, false},
}};

/// Orion's brightest among made-up stars to V 6.5.
StarTable OfflineStars() {
    StarTable table = kBrightStars;
    const StarCatalogue made = SyntheticStars(9000, 3);
    for (std::size_t i = 0; i < made.magnitude.size(); ++i) {
        table.push_back({made.rightAscension[i], made.declination[i],
                         made.magnitude[i], made.colorIndex[i]});
    }
    return table;
}

/// A VizieR table of right ascension, declination, V and B-V, rows
/// without a magnitude left out.
StarTable ReadVizier(const std::filesystem::path& path) {
    std::ifstream file(path);
    Check(file.good(), "cannot read " + path.string());
    StarTable table;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::array<std::string, 4> fields;
        std::istringstream in(line);
        for (std::string& field : fields) {
            std::getline(in, field, '\t');
        }
        try {
            const float ra = std::stof(fields[0]);
            const float dec = std::stof(fields[1]);
            const float v = std::stof(fields[2]);
            float bv = 0.65f;
            try {
                bv = std::stof(fields[3]);
            } catch (const std::exception&) {
            }
            table.push_back({ra, dec, v, bv});
        } catch (const std::exception&) {
            // Headers, units and blank magnitudes.
        }
    }
    return table;
}

/// A unit vector at `azimuth` (from north, east positive) and
/// `elevation`, degrees: x east, y north, z up.
glm::dvec3 Towards(double azimuth, double elevation) {
    const double a = glm::radians(azimuth);
    const double e = glm::radians(elevation);
    return {std::cos(e) * std::sin(a), std::cos(e) * std::cos(a),
            std::sin(e)};
}

/// Each pixel of a uint8 RGBA image decoded to linear colour.
std::vector<glm::dvec3> Linear(const Tensor& image) {
    const auto* px = static_cast<const std::uint8_t*>(image.Data());
    std::vector<glm::dvec3> out(kWidth * kHeight);
    for (std::size_t p = 0; p < out.size(); ++p) {
        out[p] = demos::SrgbToLinear(
            glm::dvec3(px[p * 4], px[p * 4 + 1], px[p * 4 + 2]) /
            255.0);
    }
    return out;
}

/// The stars seen: pixels no dimmer than their eight neighbours and
/// brighter by `threshold` (linear) than any three pixels away.
std::vector<std::size_t>
StarPeaks(const std::vector<glm::dvec3>& linear,
          double threshold = 0.02) {
    std::vector<double> lum(linear.size());
    for (std::size_t p = 0; p < lum.size(); ++p) {
        lum[p] =
            glm::dot(linear[p], glm::dvec3(0.2126, 0.7152, 0.0722));
    }
    std::vector<std::size_t> peaks;
    for (std::size_t y = 3; y + 3 < kHeight; ++y) {
        for (std::size_t x = 3; x + 3 < kWidth; ++x) {
            const double here = lum[y * kWidth + x];
            bool peak = true;
            double ring = 0.0;
            for (std::size_t v = y - 3; v <= y + 3; ++v) {
                for (std::size_t u = x - 3; u <= x + 3; ++u) {
                    const std::size_t d = std::max(
                        v > y ? v - y : y - v, u > x ? u - x : x - u);
                    const double there = lum[v * kWidth + u];
                    if (d == 1) {
                        peak = peak && here >= there;
                    } else if (d == 3) {
                        ring = std::max(ring, there);
                    }
                }
            }
            if (peak && here > ring + threshold) {
                peaks.push_back(y * kWidth + x);
            }
        }
    }
    return peaks;
}

class NightSkyDemo : public app::Demo {
  public:
    NightSkyDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 1;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        demos::DefinePlanetSurface(scene, "/World/Earth", 6371.0088e3,
                                   glm::dvec3(0.15, 0.13, 0.11),
                                   "/World/Looks/Ground");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kEarth));
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.53);
        StarField::Define(scene, "/World/Stars");
        SetStars(OfflineStars());
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetHorizontalAperture(36.0)
            .SetVerticalAperture(24.0)
            .SetExposureFStop(2.8)
            .SetSensorNoise(false);
        // A little blur and glare, so bright stars read as brighter.
        const std::pair<const char*, std::map<std::string, float>>
            steps[] = {{"psf", {{"blur", 0.9f}, {"maxRadius", 4.0f}}},
                       {"cie", {}},
                       {"glare", {{"strength", 0.05f}}},
                       {"exposure", {}}};
        const Path product("/Render/Product");
        std::vector<Path> filters;
        for (std::size_t i = 0; i < std::size(steps); ++i) {
            const Path path = product.Child("Chain").Child(
                "Step" + std::to_string(i));
            SensorFilterPrim::Define(scene, path)
                .SetFilterType(steps[i].first);
            for (const auto& [name, value] : steps[i].second) {
                scene.Set(path, "inputs:" + name, value);
            }
            filters.push_back(path);
        }
        RenderProduct::Define(scene, product)
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetFilters(filters);
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetProducts({product})
            .SetAmbient(0.0);
        Show(0);

        std::vector<std::string> names;
        for (const View& view : kViews) {
            names.emplace_back(view.name);
        }
        Widgets().Combo("View", std::move(names), 0,
                        [this](std::size_t index) {
                            Show(static_cast<int>(index));
                        });
        Widgets().Slider("Hours (UT)", 0.0, 14.0, kViews[0].hours,
                         [this](double hours) { SkyAt(hours); });
        Widgets().Slider("Sun elevation", -45.0, 5.0, kViews[0].sun,
                         [this](double degrees) { SunAt(degrees); });
        Widgets().Checkbox("Air", true,
                           [this](bool on) { ShowAir(on); });
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
        }

        // The Bright Star Catalogue when cached; else fetched in the
        // background, but not for the checks, which run offline.
        catalogue_ = demos::CacheDir() / "stars" / "bsc5.tsv";
        if (!options.test) {
            if (std::filesystem::exists(catalogue_)) {
                Load();
            } else if (demos::CanDownload()) {
                download_ = std::make_unique<demos::AsyncDownload>(
                    kBscUrl, catalogue_);
            }
        }
    }

    void Step() override {
        if (download_ != nullptr) {
            for (const auto& message : download_->TakeMessages()) {
                Report(message.text, message.error);
            }
            if (download_->Done()) {
                if (download_->Succeeded()) {
                    Load();
                }
                download_.reset();
            }
        }
    }

    /// The views side by side.
    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, demos::SideBySide(RenderAll()));
    }

    void TestFinal() override {
        const std::vector<Tensor> images = RenderAll();
        std::vector<std::vector<glm::dvec3>> linear;
        std::vector<std::size_t> stars;
        for (std::size_t i = 0; i < images.size(); ++i) {
            linear.push_back(Linear(images[i]));
            stars.push_back(StarPeaks(linear.back()).size());
            std::printf("%s: %zu stars\n", kViews[i].name,
                        stars.back());
        }
        Check(stars[0] < stars[1] && stars[1] < stars[2],
              "the stars come out at dusk");

        // Each view's stars: summed light and colour above the sky.
        const auto starlight = [&](std::size_t index) {
            glm::dvec3 sky;
            for (int c = 0; c < 3; ++c) {
                std::vector<double> channel;
                for (const glm::dvec3& p : linear[index]) {
                    channel.push_back(p[c]);
                }
                const auto middle =
                    channel.begin() +
                    static_cast<std::ptrdiff_t>(channel.size() / 2);
                std::nth_element(channel.begin(), middle,
                                 channel.end());
                sky[c] = *middle;
            }
            glm::dvec3 sum(0.0);
            for (const std::size_t p : StarPeaks(linear[index])) {
                sum +=
                    glm::max(linear[index][p] - sky, glm::dvec3(0.0));
            }
            return sum;
        };
        const glm::dvec3 high = starlight(2);
        const glm::dvec3 low = starlight(3);
        const glm::dvec3 bare = starlight(4);
        std::printf(
            "starlight high %.2f %.2f %.2f, setting %.2f %.2f "
            "%.2f, without air %.2f %.2f %.2f\n",
            high.r, high.g, high.b, low.r, low.g, low.b, bare.r,
            bare.g, bare.b);
        Check(bare.r + bare.g + bare.b > high.r + high.g + high.b,
              "the air dims the stars");
        Check(low.r / low.b > high.r / high.b,
              "stars low in the sky are redder");
    }

  private:
    void SetStars(const StarTable& table) {
        const auto column = [&](std::size_t c) {
            std::vector<float> values;
            values.reserve(table.size());
            for (const auto& row : table) {
                values.push_back(row[c]);
            }
            return Tensor::Copy<float>(
                values, {static_cast<std::int64_t>(values.size())});
        };
        StarField(GetScene(), "/World/Stars")
            .SetRightAscension(column(0))
            .SetDeclination(column(1))
            .SetMagnitude(column(2))
            .SetColorIndex(column(3));
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

    void Load() {
        try {
            const StarTable table = ReadVizier(catalogue_);
            SetStars(table);
            Report("Loaded " + std::to_string(table.size()) +
                       " Bright Star Catalogue stars",
                   false);
        } catch (const std::exception& error) {
            Report(error.what(), true);
        }
    }

    void ShowAir(bool on) {
        Atmosphere(GetScene(), "/World/Air")
            .SetVisibility(std::string(on ? tokens::kInherited
                                          : tokens::kInvisible));
    }

    void SunAt(double elevation) {
        DistantLight(GetScene(), "/World/Sun")
            .SetTransform(
                DistantLightTransform(Towards(245.0, elevation)));
    }

    /// The sky at `hours` UT, the camera following Orion but kept
    /// high enough to show the horizon.
    void SkyAt(double hours) {
        const glm::dmat4 sky = CelestialToLocal(
            kLatitude, kLongitude, JulianDate(2026, 1, 16, hours));
        StarField(GetScene(), "/World/Stars").SetTransform(sky);
        const double ra = glm::radians(kOrionRa);
        const double dec = glm::radians(kOrionDec);
        const glm::dvec3 d =
            glm::dmat3(sky) * glm::dvec3(std::cos(dec) * std::cos(ra),
                                         std::cos(dec) * std::sin(ra),
                                         std::sin(dec));
        const double azimuth = glm::degrees(std::atan2(d.x, d.y));
        const double elevation =
            std::max(glm::degrees(std::asin(d.z)), 22.0);
        const glm::dvec3 eye(0.0, 0.0, 2.0);
        Camera(GetScene(), "/World/Camera")
            .SetTransform(LookAtTransform(
                eye, eye + Towards(azimuth, elevation),
                {0.0, 0.0, 1.0}));
    }

    void Show(int index) {
        view_ = index;
        const View& view = kViews[static_cast<std::size_t>(index)];
        Camera(GetScene(), "/World/Camera")
            .SetExposureTime(view.seconds)
            .SetExposureIso(view.iso);
        ShowAir(view.air);
        SunAt(view.sun);
        SkyAt(view.hours);
    }

    /// Every view, uint8 RGBA; leaves the view as it was.
    std::vector<Tensor> RenderAll() {
        const int current = view_;
        std::vector<Tensor> images;
        for (std::size_t i = 0; i < kViews.size(); ++i) {
            Show(static_cast<int>(i));
            images.push_back(GetRenderer().Render(GetScene()));
        }
        Show(current);
        return images;
    }

    std::filesystem::path catalogue_;
    std::unique_ptr<demos::AsyncDownload> download_;
    int view_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    NightSkyDemo, "stars", "night_sky",
    "Orion over Kitt Peak through the Earth's air: the stars coming "
    "out at dusk, high at night, dimmed and reddened setting, and "
    "without the air.");
