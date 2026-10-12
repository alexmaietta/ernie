// SPDX-License-Identifier: BSD-3-Clause
// The IAU's names of lunar features (USGS Gazetteer of Planetary
// Nomenclature) as labels on the Moon, chosen each frame by a level
// of detail: facing the camera, near enough, large enough on screen,
// the largest first, at most so many. Offline, a few well-known
// features.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kRadius = 1737400.0;  // metres
constexpr std::uint32_t kWidth = 960;
constexpr std::uint32_t kHeight = 540;
constexpr double kFocal = 35.0;  // mm, over a 36 mm aperture
constexpr double kAperture = 36.0;
constexpr const char* kUrl =
    "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/"
    "MOON_nomenclature_center_pts.zip";

struct Feature {
    std::string name;
    double lat = 0.0;       // degrees north
    double lon = 0.0;       // degrees east
    double diameter = 0.0;  // km
    std::string type;
};

/// Well-known features, for running offline.
std::vector<Feature> FallbackFeatures() {
    return {
        {"Oceanus Procellarum", 20.67, 303.32, 2592.2, "Oceanus"},
        {"Mare Imbrium", 34.72, 345.09, 1145.5, "Mare"},
        {"Mare Tranquillitatis", 8.35, 30.83, 875.7, "Mare"},
        {"Mare Serenitatis", 27.29, 18.36, 674.3, "Mare"},
        {"Mare Crisium", 16.18, 59.10, 555.9, "Mare"},
        {"Mare Nectaris", -15.19, 34.60, 339.4, "Mare"},
        {"Clavius", -58.62, 345.27, 230.8, "Crater"},
        {"Plato", 51.62, 350.62, 100.7, "Crater"},
        {"Copernicus", 9.62, 339.92, 96.1, "Crater"},
        {"Tycho", -43.30, 348.78, 85.3, "Crater"},
        {"Aristarchus", 23.73, 312.51, 40.0, "Crater"},
        {"Kepler", 8.12, 321.99, 29.5, "Crater"},
    };
}

/// The Gazetteer's features: the shapefile's attribute table (.dbf),
/// stored uncompressed in the zip as the USGS publishes it.
std::vector<Feature>
ReadNomenclature(const std::filesystem::path& zip) {
    std::ifstream file(zip, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)),
                            {});
    const auto u16 = [&](std::size_t at) {
        return static_cast<std::uint32_t>(
            static_cast<unsigned char>(bytes[at]) |
            (static_cast<unsigned char>(bytes[at + 1]) << 8));
    };
    const auto u32 = [&](std::size_t at) {
        return u16(at) | (u16(at + 2) << 16);
    };
    // The central directory, from its end record, to the .dbf entry.
    const std::size_t end = bytes.rfind("PK\x05\x06");
    Check(end != std::string::npos, "the nomenclature is no zip");
    std::size_t entry = u32(end + 16);
    std::size_t dbf = 0;
    for (std::uint32_t e = 0; e < u16(end + 10); ++e) {
        const std::string name =
            bytes.substr(entry + 46, u16(entry + 28));
        if (name.ends_with(".dbf")) {
            Check(u16(entry + 10) == 0, "the table is compressed");
            const std::size_t local = u32(entry + 42);
            dbf = local + 30 + u16(local + 26) + u16(local + 28);
        }
        entry +=
            46 + u16(entry + 28) + u16(entry + 30) + u16(entry + 32);
    }
    Check(dbf != 0, "the nomenclature has no table");
    // dBase: the record count, header and record sizes, then a field
    // descriptor of 32 bytes each.
    const std::uint32_t records = u32(dbf + 4);
    const std::uint32_t header = u16(dbf + 8);
    const std::uint32_t length = u16(dbf + 10);
    struct Field {
        std::size_t offset = 0;
        std::size_t size = 0;
    };
    std::vector<std::pair<std::string, Field>> fields;
    std::size_t offset = 1;  // past the deletion flag
    for (std::size_t at = dbf + 32; bytes[at] != '\x0d'; at += 32) {
        // Up to 11 bytes, ended by a 0 if shorter.
        std::string name = bytes.substr(at, 11);
        name.resize(name.find('\0') == std::string::npos
                        ? 11
                        : name.find('\0'));
        const std::size_t size =
            static_cast<unsigned char>(bytes[at + 16]);
        fields.emplace_back(name, Field{offset, size});
        offset += size;
    }
    const auto field = [&](std::string_view name) {
        for (const auto& [n, f] : fields) {
            if (n == name) {
                return f;
            }
        }
        throw std::runtime_error("no field " + std::string(name));
    };
    const Field name = field("clean_name");
    const Field type = field("type");
    const Field lat = field("center_lat");
    const Field lon = field("center_lon");
    const Field diameter = field("diameter");
    std::vector<Feature> features;
    for (std::uint32_t r = 0; r < records; ++r) {
        const std::size_t base =
            dbf + header + std::size_t{r} * length;
        const auto text = [&](const Field& f) {
            std::string value = bytes.substr(base + f.offset, f.size);
            value.erase(value.find_last_not_of(' ') + 1);
            return value;
        };
        Feature feature;
        feature.name = text(name);
        feature.type = text(type);
        feature.lat = std::stod(text(lat));
        feature.lon = std::stod(text(lon));
        feature.diameter = std::stod(text(diameter));
        features.push_back(std::move(feature));
    }
    return features;
}

/// The unit vector out of the Moon at a feature: x at 0 east, z
/// north.
glm::dvec3 Direction(double lat, double lon) {
    const double phi = glm::radians(lat);
    const double lambda = glm::radians(lon);
    return {std::cos(phi) * std::cos(lambda),
            std::cos(phi) * std::sin(lambda), std::sin(phi)};
}

/// A feature type's colour: seas, craters, mountains, lines, others.
glm::vec4 TypeColor(const std::string& type) {
    for (const char* sea :
         {"Mare", "Oceanus", "Lacus", "Sinus", "Palus"}) {
        if (type.starts_with(sea)) {
            return {0.55f, 0.75f, 1.0f, 1.0f};
        }
    }
    if (type.starts_with("Crater")) {
        return {1.0f, 1.0f, 1.0f, 1.0f};
    }
    for (const char* high : {"Mons", "Promontorium"}) {
        if (type.starts_with(high)) {
            return {1.0f, 0.7f, 0.35f, 1.0f};
        }
    }
    for (const char* line :
         {"Rima", "Rupes", "Dorsum", "Vallis", "Catena"}) {
        if (type.starts_with(line)) {
            return {0.55f, 1.0f, 0.55f, 1.0f};
        }
    }
    if (type.starts_with("Satellite")) {
        return {0.75f, 0.75f, 0.75f, 1.0f};
    }
    return {1.0f, 0.55f, 0.9f, 1.0f};
}

/// The level of detail's options.
struct Lod {
    double maxLabels = 60.0;
    double minPixels = 12.0;    // a feature's diameter on screen
    double maxDistance = 6000;  // km from the camera
    bool satellites = false;    // lettered satellite craters
    bool outlines = true;
};

class MoonNamesDemo : public app::Demo {
  public:
    MoonNamesDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 240;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        Sphere::Define(scene, "/World/Moon")
            .SetRadius(kRadius)
            .SetDisplayColor({0.55, 0.55, 0.55});
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.6, -0.5, 0.3}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(kFocal)
            .SetHorizontalAperture(kAperture)
            .SetVerticalAperture(kAperture * kHeight / kWidth)
            .SetClippingRange(glm::dvec2(1000.0, 1.0e8));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(kWidth, kHeight))
            .SetOverlays("*");
        // In the viewer the camera is free, starting 2500 km over
        // Oceanus Procellarum, near enough for names; the checks
        // follow a descent.
        free_ = viewer != nullptr && !options.test;
        if (viewer != nullptr) {
            Pose(0.0);
            viewer->SetActiveCamera("/World/Camera");
            if (free_) {
                viewer->LookAt((kRadius + 2.5e6) *
                                   Direction(10.0, 315.0),
                               glm::dvec3(0.0));
            }
            // The Sun's gizmo would sit at the Moon's centre.
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            viewer->SetViewportDisplay(display);
        }
        // The Gazetteer when cached; else a few features while it is
        // fetched in the background, but not for the checks, which
        // run offline.
        zip_ = demos::CacheDir() / "moon" /
               "MOON_nomenclature_center_pts.zip";
        features_ = FallbackFeatures();
        if (std::filesystem::exists(zip_) && !options.test) {
            Load();
        } else if (!options.test && demos::CanDownload()) {
            download_ =
                std::make_unique<demos::AsyncDownload>(kUrl, zip_);
        }
        Widgets().Slider("Max labels", 1, 300, lod_.maxLabels,
                         [this](double v) { lod_.maxLabels = v; });
        Widgets().Slider("Min pixels", 1, 100, lod_.minPixels,
                         [this](double v) { lod_.minPixels = v; });
        Widgets().Slider("Max distance (km)", 100, 10000,
                         lod_.maxDistance,
                         [this](double v) { lod_.maxDistance = v; });
        Widgets().Checkbox("Satellite craters", lod_.satellites,
                           [this](bool on) { lod_.satellites = on; });
        Widgets().Checkbox("Outlines", lod_.outlines,
                           [this](bool on) { lod_.outlines = on; });
        Place();
    }

    void Step() override {
        // The download's news in the viewer's log, and its names once
        // fetched.
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
        Place();
    }

    void TestFinal() override {
        for (const std::size_t count : shown_) {
            Check(count > 0 && count <= static_cast<std::size_t>(
                                            lod_.maxLabels),
                  "labels each frame, no more than allowed");
        }
        Check(smallest_.back() < smallest_.front(),
              "smaller features named as the camera comes down");
    }

  private:
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

    /// The Gazetteer's names in place of the few built in.
    void Load() {
        try {
            features_ = ReadNomenclature(zip_);
            Report("Loaded " + std::to_string(features_.size()) +
                       " lunar feature names",
                   false);
        } catch (const std::exception& error) {
            Report(error.what(), true);
        }
    }

    /// The camera at `t` of its descent: from four radii out, looking
    /// at the Moon, to a low orbit past Copernicus, looking back down
    /// at Kepler.
    void Pose(double t) {
        const double distance = kRadius * (4.0 - 2.8 * t);
        const double lat = 10.0 + 5.0 * t;
        const double lon = 300.0 + 35.0 * t;
        const glm::dvec3 eye = distance * Direction(lat, lon);
        const glm::dvec3 target =
            t * kRadius * Direction(lat - 5.0, lon - 13.0);
        GetScene().Set("/World/Camera", tokens::kTransform,
                       LookAtTransform(eye, target, {0.0, 0.0, 1.0}));
    }

    /// Moves the camera down its path, unless free, and labels what
    /// it sees.
    void Place() {
        Scene& scene = GetScene();
        if (!free_) {
            Pose(std::min(static_cast<double>(frame) / 239.0, 1.0));
        }
        const glm::dmat4 pose = scene.WorldTransform("/World/Camera");
        const glm::dvec3 eye = glm::dvec3(pose[3]);
        const double distance = glm::length(eye);
        // The camera's axes (right, up, forward), to keep to what it
        // sees.
        const glm::dvec3 right = glm::dvec3(pose[0]);
        const glm::dvec3 up = glm::dvec3(pose[1]);
        const glm::dvec3 forward = -glm::dvec3(pose[2]);
        // Each candidate's size on screen, largest first.
        const double focal = kWidth * kFocal / kAperture;  // pixels
        std::vector<std::pair<double, std::size_t>> candidates;
        for (std::size_t i = 0; i < features_.size(); ++i) {
            const Feature& f = features_[i];
            if (!lod_.satellites && f.type.starts_with("Satellite")) {
                continue;
            }
            const glm::dvec3 n = Direction(f.lat, f.lon);
            const glm::dvec3 toEye = eye - kRadius * n;
            const double range = glm::length(toEye);
            // Facing the camera, above its horizon, near enough.
            if (glm::dot(n, toEye) < 0.1 * range ||
                range > 1000.0 * lod_.maxDistance) {
                continue;
            }
            // In view.
            const glm::dvec3 d = -toEye;
            const double depth = glm::dot(d, forward);
            if (depth <= 0.0 ||
                std::abs(glm::dot(d, right)) * focal >
                    0.5 * kWidth * depth ||
                std::abs(glm::dot(d, up)) * focal >
                    0.5 * kHeight * depth) {
                continue;
            }
            const double pixels = focal * 1000.0 * f.diameter / range;
            if (pixels >= lod_.minPixels) {
                candidates.emplace_back(pixels, i);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      return a.first != b.first ? a.first > b.first
                                                : a.second < b.second;
                  });
        candidates.resize(
            std::min(candidates.size(),
                     static_cast<std::size_t>(lod_.maxLabels)));
        Overlay& names = scene.GetOverlay("names").Clear();
        Overlay& outlines = scene.GetOverlay("outlines").Clear();
        for (const auto& [pixels, i] : candidates) {
            const Feature& f = features_[i];
            const glm::dvec3 n = Direction(f.lat, f.lon);
            OverlayStyle label;
            label.color = TypeColor(f.type);
            label.size = static_cast<float>(
                std::clamp(10.0 + pixels / 40.0, 10.0, 18.0));
            label.depth = OverlayDepth::Always;
            // A dark halo keeps names legible over bright ground.
            label.background.a = 0.8f;
            label.halo = 1.5f;
            names.Label(kRadius * n, f.name, label);
            if (lod_.outlines) {
                // The feature's rim traced on the surface.
                const double angle =
                    0.5 * f.diameter * 1000.0 / kRadius;
                const glm::dvec3 u = glm::normalize(glm::cross(
                    std::abs(n.z) < 0.9 ? glm::dvec3(0, 0, 1)
                                        : glm::dvec3(1, 0, 0),
                    n));
                const glm::dvec3 v = glm::cross(n, u);
                std::vector<glm::dvec3> rim;
                for (int k = 0; k < 48; ++k) {
                    const double a =
                        2.0 * std::numbers::pi * k / 48.0;
                    rim.push_back(
                        1.0005 * kRadius *
                        (std::cos(angle) * n +
                         std::sin(angle) *
                             (std::cos(a) * u + std::sin(a) * v)));
                }
                OverlayStyle edge;
                edge.color = label.color;
                edge.color.a = 0.6f;
                edge.width = 1.5f;
                outlines.Polyline(rim, edge, true);
            }
        }
        shown_.push_back(candidates.size());
        double smallest = std::numeric_limits<double>::infinity();
        for (const auto& [pixels, i] : candidates) {
            smallest = std::min(smallest, features_[i].diameter);
        }
        smallest_.push_back(smallest);
        // The HUD: what the level of detail shows, and its options.
        Overlay& hud = scene.GetOverlay("hud").Clear();
        char line[256];
        std::snprintf(line, sizeof(line),
                      "Lunar nomenclature (IAU, USGS)\n"
                      "%zu of %zu names, altitude %.0f km\n"
                      "max %d labels, min %.0f px, max %.0f km%s",
                      candidates.size(), features_.size(),
                      (distance - kRadius) / 1000.0,
                      static_cast<int>(lod_.maxLabels),
                      lod_.minPixels, lod_.maxDistance,
                      lod_.satellites ? ", satellites" : "");
        OverlayStyle text;
        text.size = 13.0f;
        hud.Text({.at = {1.0f, 1.0f}, .offset = {-12.0f, -12.0f}},
                 line, text);
    }

    std::vector<Feature> features_;
    std::filesystem::path zip_;
    std::unique_ptr<demos::AsyncDownload> download_;
    Lod lod_;
    bool free_ = false;  // the viewer moves the camera
    std::vector<std::size_t> shown_;
    std::vector<double> smallest_;  // km, per frame
};

}  // namespace

ERNIE_REGISTER_DEMO(MoonNamesDemo, "overlays", "moon names",
                    "The IAU's lunar feature names as labels on the "
                    "Moon, chosen by a level of detail as the camera "
                    "comes down.");
