// SPDX-License-Identifier: BSD-3-Clause
// The Sun, its planets and the Moon at their true sizes and distances
// on a chosen date (no motion), lit by the Sun as a sphere light,
// their orbits drawn as overlays split finely where they pass the
// eye. Each body is an adaptive sphere, finer towards the eye, and
// wears a latitude-longitude map streamed as a virtual texture:
// NASA's where one is small enough, else Solar System Scope's (CC BY
// 4.0, from NASA's data); the Sun's own generated, emitted through
// its material. The panel picks the date and a body to go to.
//
// References:
// - E. M. Standish, "Keplerian Elements for Approximate Positions of
//   the Major Planets", JPL Solar System Dynamics (1800 to 2050).
// - B. A. Archinal et al., "Report of the IAU Working Group on
//   Cartographic Coordinates and Rotational Elements: 2015",
//   Celestial Mechanics and Dynamical Astronomy 130:22, 2018.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "util.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr double kPi = std::numbers::pi;
constexpr double kDegree = kPi / 180.0;
constexpr double kAu = 1.495978707e11;
// The ecliptic's obliquity at J2000, which turns the IAU's equatorial
// poles into the ecliptic frame the scene is in (z its north).
constexpr double kObliquity = 23.43928 * kDegree;
// The Sun's mean radiance as luminance: its disc's 1.6e9 cd/m^2.
constexpr double kSunLuminance = 1.6e9;

/// A body's mean orbital elements at J2000 and their rates per Julian
/// century (Standish): semi-major axis (au), eccentricity,
/// inclination, mean longitude, longitude of perihelion and of the
/// ascending node (degrees).
struct Elements {
    std::array<double, 6> at;
    std::array<double, 6> rate;
};

/// A body: its size and shape, its orbit about the Sun (none for the
/// Sun and the Moon), its north pole's right ascension and
/// declination and prime meridian's angle (degrees, and degrees a
/// day) as the IAU gives them, its map, and a colour for it and its
/// orbit until then.
struct Body {
    const char* name;
    double equator;  ///< Metres.
    double pole;
    std::optional<Elements> orbit;
    double poleRa;
    double poleDec;
    double w0;
    double wRate;
    const char* url;  ///< Empty for the Sun's, made here.
    const char* folder;
    glm::dvec3 colour;
};

constexpr const char* kScope =
    "https://www.solarsystemscope.com/textures/download/";

const std::array<Body, 10> kBodies = {{
    {"Sun",
     695.7e6,
     695.7e6,
     std::nullopt,
     286.13,
     63.87,
     84.176,
     14.1844,
     "",
     "solar_system",
     {1.0, 0.8, 0.5}},
    {"Mercury",
     2440.5e3,
     2438.3e3,
     Elements{{0.38709927, 0.20563593, 7.00497902, 252.25032350,
               77.45779628, 48.33076593},
              {0.00000037, 0.00001906, -0.00594749, 149472.67411175,
               0.16047689, -0.12534081}},
     281.0103,
     61.4155,
     329.5988,
     6.1385108,
     "8k_mercury.jpg",
     "solar_system",
     {0.6, 0.6, 0.6}},
    {"Venus",
     6051.8e3,
     6051.8e3,
     Elements{{0.72333566, 0.00677672, 3.39467605, 181.97909950,
               131.60246718, 76.67984255},
              {0.00000390, -0.00004107, -0.00078890, 58517.81538729,
               0.00268329, -0.27769418}},
     272.76,
     67.16,
     160.20,
     -1.4813688,
     "4k_venus_atmosphere.jpg",
     "solar_system",
     {0.9, 0.8, 0.6}},
    {"Earth",
     6378.137e3,
     6356.752e3,
     Elements{{1.00000261, 0.01671123, -0.00001531, 100.46457166,
               102.93768193, 0.0},
              {0.00000562, -0.00004392, -0.01294668, 35999.37244981,
               0.32327364, 0.0}},
     0.0,
     90.0,
     190.147,
     360.9856235,
     "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/"
     "world.topo.bathy.200408.3x21600x10800.jpg",
     "blue_marble",
     {0.2, 0.4, 0.9}},
    {"Moon",
     1738.1e3,
     1736.0e3,
     std::nullopt,
     269.9949,
     66.5392,
     38.3213,
     13.17635815,
     "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/"
     "lroc_color_poles_8k.tif",
     "moon",
     {0.7, 0.7, 0.7}},
    {"Mars",
     3396.2e3,
     3376.2e3,
     Elements{{1.52371034, 0.09339410, 1.84969142, -4.55343205,
               -23.94362959, 49.55953891},
              {0.00001847, 0.00007882, -0.00813131, 19140.30268499,
               0.44441088, -0.29257343}},
     317.269202,
     54.432516,
     176.049863,
     350.891982443297,
     "8k_mars.jpg",
     "solar_system",
     {0.8, 0.4, 0.2}},
    {"Jupiter",
     71492e3,
     66854e3,
     Elements{{5.20288700, 0.04838624, 1.30439695, 34.39644051,
               14.72847983, 100.47390909},
              {-0.00011607, -0.00013253, -0.00183714, 3034.74612775,
               0.21252668, 0.20469106}},
     268.056595,
     64.495303,
     284.95,
     870.5360000,
     "https://assets.science.nasa.gov/content/dam/science/psd/"
     "photojournal/pia/pia07/pia07782/PIA07782.tif",
     "solar_system",
     {0.8, 0.7, 0.5}},
    {"Saturn",
     60268e3,
     54364e3,
     Elements{{9.53667594, 0.05386179, 2.48599187, 49.95424423,
               92.59887831, 113.66242448},
              {-0.00125060, -0.00050991, 0.00193609, 1222.49362201,
               -0.41897216, -0.28867794}},
     40.589,
     83.537,
     38.90,
     810.7939024,
     "8k_saturn.jpg",
     "solar_system",
     {0.9, 0.8, 0.6}},
    {"Uranus",
     25559e3,
     24973e3,
     Elements{{19.18916464, 0.04725744, 0.77263783, 313.23810451,
               170.95427630, 74.01692503},
              {-0.00196176, -0.00004397, -0.00242939, 428.48202785,
               0.40805281, 0.04240589}},
     257.311,
     -15.175,
     203.81,
     -501.1600928,
     "2k_uranus.jpg",
     "solar_system",
     {0.6, 0.8, 0.9}},
    {"Neptune",
     24764e3,
     24341e3,
     Elements{{30.06992276, 0.00859048, 1.77004347, -55.12002969,
               44.96476227, 131.78422574},
              {0.00026291, 0.00005105, 0.00035372, 218.45945325,
               -0.32241464, -0.00508664}},
     299.36,
     43.46,
     249.978,
     541.1397757,
     "2k_neptune.jpg",
     "solar_system",
     {0.3, 0.4, 0.9}},
}};

// Saturn's rings: their radii, and a map of their colour and opacity
// from the inside outwards.
constexpr double kRingInner = 74500e3;
constexpr double kRingOuter = 140220e3;
constexpr const char* kRingMap = "8k_saturn_ring_alpha.png";

constexpr std::size_t kSun = 0;
constexpr std::size_t kEarth = 3;
constexpr std::size_t kMoon = 4;
constexpr std::size_t kSaturn = 7;

/// A map's URL: Solar System Scope's by file name, else as given.
std::string UrlOf(const std::string& map) {
    return map.starts_with("https://") ? map
                                       : std::string(kScope) + map;
}

/// Where map `url` is cached, in `folder` (the other demos' for their
/// bodies, shared).
std::filesystem::path CachedPath(const std::string& url,
                                 const char* folder) {
    return demos::CacheDir() / folder /
           std::filesystem::path(url).filename();
}

/// Point (x, y) of an orbit's plane, its periapsis along x, in the
/// ecliptic frame: turned by the argument of periapsis `omega`, the
/// inclination `i` and the node's longitude `node` (radians).
glm::dvec3 Ecliptic(double x, double y, double omega, double i,
                    double node) {
    const double co = std::cos(omega), so = std::sin(omega);
    const double cn = std::cos(node), sn = std::sin(node);
    const double ci = std::cos(i), si = std::sin(i);
    return {
        (co * cn - so * sn * ci) * x + (-so * cn - co * sn * ci) * y,
        (co * sn + so * cn * ci) * x + (-so * sn + co * cn * ci) * y,
        (so * si) * x + (co * si) * y};
}

/// An orbit at `centuries` past J2000: where it lies at eccentric
/// anomaly `e` (radians), and the body's own eccentric anomaly then.
struct Orbit {
    double a = 0.0;  ///< Metres.
    double e = 0.0;
    double i = 0.0;
    double omega = 0.0;
    double node = 0.0;
    double anomaly = 0.0;

    Orbit(const Elements& elements, double centuries) {
        std::array<double, 6> v{};
        for (std::size_t k = 0; k < 6; ++k) {
            v[k] = elements.at[k] + elements.rate[k] * centuries;
        }
        a = v[0] * kAu;
        e = v[1];
        i = v[2] * kDegree;
        node = v[5] * kDegree;
        omega = (v[4] - v[5]) * kDegree;
        // Kepler's equation, M = E - e sin E, by Newton's method.
        const double m =
            std::remainder((v[3] - v[4]) * kDegree, 2.0 * kPi);
        anomaly = m + e * std::sin(m);
        for (int step = 0; step < 8; ++step) {
            anomaly -= (anomaly - e * std::sin(anomaly) - m) /
                       (1.0 - e * std::cos(anomaly));
        }
    }

    glm::dvec3 At(double eccentric) const {
        return Ecliptic(a * (std::cos(eccentric) - e),
                        a * std::sqrt(1.0 - e * e) *
                            std::sin(eccentric),
                        omega, i, node);
    }
};

/// The Moon's place about the Earth `days` past J2000: a mean
/// circular orbit 384 400 km out, 5.145 degrees to the ecliptic, its
/// node regressing; and that orbit at argument of latitude `u`.
struct MoonOrbit {
    double node;
    double latitude;

    explicit MoonOrbit(double days)
        : node((125.08 - 0.052954 * days) * kDegree),
          latitude((218.316 + 13.176396 * days - 125.08 +
                    0.052954 * days) *
                   kDegree) {}

    glm::dvec3 At(double u) const {
        return Ecliptic(384400e3 * std::cos(u),
                        384400e3 * std::sin(u), 0.0, 5.145 * kDegree,
                        node);
    }
};

/// Body `body`'s frame `days` past J2000, its radius at x and y and
/// pole at z: its x towards its prime meridian, z its north pole,
/// turned from the IAU's equatorial frame into the ecliptic's.
glm::dmat3 Frame(const Body& body, double days) {
    const double ra = body.poleRa * kDegree;
    const double dec = body.poleDec * kDegree;
    const glm::dvec3 pole(std::cos(dec) * std::cos(ra),
                          std::cos(dec) * std::sin(ra),
                          std::sin(dec));
    // The ascending node of its equator on the Earth's.
    const glm::dvec3 node(-std::sin(ra), std::cos(ra), 0.0);
    const double w = std::remainder(
        (body.w0 + body.wRate * days) * kDegree, 2.0 * kPi);
    const glm::dvec3 x =
        std::cos(w) * node + std::sin(w) * glm::cross(pole, node);
    const auto ecliptic = [](const glm::dvec3& v) {
        return glm::dvec3(
            v.x,
            v.y * std::cos(kObliquity) + v.z * std::sin(kObliquity),
            -v.y * std::sin(kObliquity) + v.z * std::cos(kObliquity));
    };
    return {ecliptic(x), ecliptic(glm::cross(pole, x)),
            ecliptic(pole)};
}

/// The Sun's surface, made here: granules over a brighter background
/// and a few dark spots, latitude by longitude.
std::filesystem::path SunMap() {
    const std::filesystem::path path =
        demos::CacheDir() / "solar_system" / "sun_map.png";
    if (std::filesystem::exists(path)) {
        return path;
    }
    constexpr std::int64_t kCols = 2048;
    constexpr std::int64_t kRows = 1024;
    std::mt19937 random(1);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> value(static_cast<std::size_t>(kCols * kRows),
                              0.0);
    // Granulation: noise of a few octaves, each cell a random level
    // blended across.
    for (int octave = 0; octave < 4; ++octave) {
        const std::int64_t cells = 64 << octave;
        std::vector<double> lattice(
            static_cast<std::size_t>((cells + 1) * (cells / 2 + 1)));
        for (double& v : lattice) {
            v = uniform(random);
        }
        for (std::int64_t row = 0; row < kRows; ++row) {
            const double y = static_cast<double>(row) *
                             static_cast<double>(cells / 2) / kRows;
            for (std::int64_t col = 0; col < kCols; ++col) {
                const double x = static_cast<double>(col) *
                                 static_cast<double>(cells) / kCols;
                const auto x0 = static_cast<std::int64_t>(x);
                const auto y0 = static_cast<std::int64_t>(y);
                const double fx = x - static_cast<double>(x0);
                const double fy = y - static_cast<double>(y0);
                const auto at = [&](std::int64_t i, std::int64_t j) {
                    return lattice[static_cast<std::size_t>(
                        j * (cells + 1) + i)];
                };
                value[static_cast<std::size_t>(row * kCols + col)] +=
                    std::ldexp(1.0, -octave) *
                    ((1 - fx) * (1 - fy) * at(x0, y0) +
                     fx * (1 - fy) * at(x0 + 1, y0) +
                     (1 - fx) * fy * at(x0, y0 + 1) +
                     fx * fy * at(x0 + 1, y0 + 1));
            }
        }
    }
    // Sunspots in the active latitudes.
    std::vector<glm::dvec3> spots;
    for (int s = 0; s < 6; ++s) {
        spots.emplace_back(
            uniform(random) * kCols,
            kRows * (0.5 + (uniform(random) < 0.5 ? -1 : 1) *
                               (0.06 + 0.1 * uniform(random))),
            6.0 + 10.0 * uniform(random));
    }
    std::vector<std::uint8_t> texels(
        static_cast<std::size_t>(kCols * kRows * 4));
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t col = 0; col < kCols; ++col) {
            double level =
                0.75 + 0.2 * (value[static_cast<std::size_t>(
                                  row * kCols + col)] /
                                  1.875 -
                              0.5);
            for (const glm::dvec3& spot : spots) {
                const double d =
                    glm::length(glm::dvec2(static_cast<double>(col),
                                           static_cast<double>(row)) -
                                glm::dvec2(spot));
                level *=
                    1.0 - 0.7 * std::exp(-d * d / (spot.z * spot.z));
            }
            const auto c = [&](double scale) {
                return static_cast<std::uint8_t>(
                    std::clamp(level * scale * 255.0, 0.0, 255.0));
            };
            std::uint8_t* texel =
                &texels[static_cast<std::size_t>(row * kCols + col) *
                        4];
            texel[0] = c(1.0);
            texel[1] = c(0.8);
            texel[2] = c(0.45);
            texel[3] = 255;
        }
    }
    std::filesystem::create_directories(path.parent_path());
    app::WritePng(
        path, Tensor::Copy<std::uint8_t>(texels, {kRows, kCols, 4}));
    return path;
}

/// Points along an orbit's closed curve `at(t)`, t from 0 to 2 pi,
/// split finely where it passes `eye`: until a piece's middle strays
/// from its chord by under a ten-thousandth of its distance from the
/// eye, so the line meets the body there however close the eye is.
template <typename At>
std::vector<glm::dvec3> OrbitPoints(const At& at,
                                    const glm::dvec3& eye) {
    constexpr int kStart = 256;
    std::vector<glm::dvec3> points;
    const auto refine = [&](auto&& self, double t0, double t1,
                            const glm::dvec3& p0,
                            const glm::dvec3& p1, int depth) -> void {
        const double tm = 0.5 * (t0 + t1);
        const glm::dvec3 pm = at(tm);
        const double sag = glm::length(pm - 0.5 * (p0 + p1));
        if (depth < 40 && sag > 1e-4 * glm::length(pm - eye)) {
            self(self, t0, tm, p0, pm, depth + 1);
            self(self, tm, t1, pm, p1, depth + 1);
            return;
        }
        points.push_back(p1);
    };
    glm::dvec3 previous = at(0.0);
    points.push_back(previous);
    for (int k = 1; k <= kStart; ++k) {
        const double t = 2.0 * kPi * k / kStart;
        const glm::dvec3 next = at(t);
        refine(refine, 2.0 * kPi * (k - 1) / kStart, t, previous,
               next, 0);
        previous = next;
    }
    points.pop_back();  // The first again.
    return points;
}

class SolarSystemDemo : public app::Demo {
  public:
    SolarSystemDemo(app::Viewer* viewer,
                    const app::DemoOptions& options)
        : Demo(viewer, options), test_(options.test) {
        numFrames = 3;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        for (std::size_t i = 0; i < kBodies.size(); ++i) {
            const Body& body = kBodies[i];
            const std::string name = body.name;
            const Path look("/World/Looks/" + name);
            Sphere::Define(scene, "/World/" + name)
                .SetRadius(body.equator)
                .SetTessellation(std::string(tokens::kAdaptiveSphere))
                .SetMaterialBinding(look);
            Material::Define(scene, look);
            Shader::Define(scene, look.String() + "/Surface")
                .SetInfoId(tokens::kUsdPreviewSurface)
                .SetDiffuseColor(i == kSun ? glm::dvec3(0.0)
                                           : body.colour)
                .SetEmissiveColor(i == kSun
                                      ? kSunLuminance * body.colour
                                      : glm::dvec3(0.0))
                .SetRoughness(1.0);
            Connect(scene, look, tokens::kSurfaceOutput,
                    look.String() + "/Surface",
                    tokens::kSurfaceOutput);
            Shader::Define(scene, look.String() + "/St")
                .SetInfoId(tokens::kUsdPrimvarReaderFloat2)
                .SetVarname("st");
        }
        DefineRings(scene);
        // The Sun's light: a sphere of its radius and radiance.
        SphereLight::Define(scene, "/World/SunLight")
            .SetRadius(kBodies[kSun].equator)
            .SetIntensity(kSunLuminance)
            .SetShadowEnable(false);
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
            .SetAmbient(0.0)
            .SetOverlays("*");

        // Maps: the Sun's made here, the rest fetched in turn, each
        // worn as it comes.
        UseMap(kSun, SunMap());
        if (!test_ && demos::CanDownload()) {
            for (std::size_t i = 0; i < kBodies.size(); ++i) {
                if (*kBodies[i].url != '\0') {
                    maps_.push_back(i);
                }
            }
            maps_.push_back(kBodies.size());  // The rings'.
            progress_ = Widgets().Progress("Maps");
        }

        Widgets().Slider("Year", 1900.0, 2100.0, year_,
                         [this](double y) {
                             year_ = y;
                             Place();
                         });
        std::vector<std::string> targets;
        for (const Body& body : kBodies) {
            targets.emplace_back(body.name);
        }
        targets.emplace_back("Inner planets");
        targets.emplace_back("Whole system");
        Widgets().Combo("Go to", targets, kEarth,
                        [this](std::size_t i) { GoTo(i); });
        // Stops past the exposure for the sunlight where the camera
        // goes.
        Widgets().Slider("Exposure (stops)", -10.0, 10.0, 0.0,
                         [this](double stops) {
                             stops_ = stops;
                             Expose();
                         });
        Widgets().Checkbox("Orbits", true, [this](bool on) {
            GetScene().GetOverlay("orbits").SetVisible(on);
        });
        Place();
        if (viewer != nullptr) {
            viewer->SetActiveCamera("/World/Camera");
            app::ViewportDisplay display =
                viewer->GetViewportDisplay();
            display.lights = false;
            viewer->SetViewportDisplay(display);
        }
        GoTo(kEarth);
    }

    void Step() override {
        StepMaps();
        // Orbits drawn again when the eye has moved.
        const glm::dvec3 eye = Eye();
        if (glm::length(eye - drawnFrom_) > 1e-3 * Nearest(eye)) {
            DrawOrbits();
        }
    }

    void TestFinal() override {
        // The Earth from the side facing the Sun: lit, not black.
        GoTo(kEarth);
        const auto centre = [this] {
            const Tensor image = GetRenderer().Render(GetScene());
            const std::int64_t rows = image.Shape()[0];
            const std::int64_t cols = image.Shape()[1];
            const auto* p =
                static_cast<const std::uint8_t*>(image.Data()) +
                ((rows / 2) * cols + cols / 2) * 4;
            return glm::dvec3(p[0], p[1], p[2]);
        };
        const glm::dvec3 earth = centre();
        // The Sun, exposed for its own light: its surface's map
        // shows, not one white.
        GoTo(kSun);
        const Tensor sun = GetRenderer().Render(GetScene());
        const std::int64_t rows = sun.Shape()[0];
        const std::int64_t cols = sun.Shape()[1];
        const auto* p = static_cast<const std::uint8_t*>(sun.Data());
        int lowest = 255;
        int highest = 0;
        for (std::int64_t row = rows / 2 - 20; row < rows / 2 + 20;
             ++row) {
            for (std::int64_t col = cols / 2 - 20;
                 col < cols / 2 + 20; ++col) {
                const int r = p[(row * cols + col) * 4];
                lowest = std::min(lowest, r);
                highest = std::max(highest, r);
            }
        }
        // The whole system: orbits drawn across the black.
        GoTo(kBodies.size() + 1);
        const Tensor whole = GetRenderer().Render(GetScene());
        std::int64_t drawn = 0;
        const auto* q =
            static_cast<const std::uint8_t*>(whole.Data());
        for (std::int64_t i = 0;
             i < whole.Shape()[0] * whole.Shape()[1]; ++i) {
            drawn +=
                q[i * 4] + q[i * 4 + 1] + q[i * 4 + 2] > 60 ? 1 : 0;
        }
        std::printf(
            "Earth %.0f %.0f %.0f; Sun %d to %d; whole system "
            "%lld pixels drawn\n",
            earth.r, earth.g, earth.b, lowest, highest,
            static_cast<long long>(drawn));
        Check(earth.r + earth.g + earth.b > 60.0,
              "the Earth's day side is lit by the Sun");
        Check(highest > 40 && highest - lowest > 10,
              "the Sun's surface varies over its map");
        Check(drawn > 1000, "the orbits are drawn");
    }

  private:
    /// Saturn's rings: a flat annulus in its equator, its texture
    /// coordinates' s running outwards, its map's alpha its opacity.
    void DefineRings(Scene& scene) {
        constexpr int kSegments = 512;
        std::vector<double> points;
        std::vector<float> st;
        std::vector<std::int64_t> counts;
        std::vector<std::int64_t> indices;
        for (int k = 0; k <= kSegments; ++k) {
            const double angle = 2.0 * kPi * k / kSegments;
            for (const double radius : {kRingInner, kRingOuter}) {
                points.insert(points.end(),
                              {radius * std::cos(angle),
                               radius * std::sin(angle), 0.0});
                st.insert(st.end(),
                          {radius == kRingInner ? 0.0f : 1.0f,
                           static_cast<float>(k) / kSegments});
            }
            // Both windings, seen from either side.
            if (k < kSegments) {
                counts.insert(counts.end(), {4, 4});
                indices.insert(indices.end(),
                               {2 * k, 2 * k + 1, 2 * k + 3,
                                2 * k + 2, 2 * k, 2 * k + 2,
                                2 * k + 3, 2 * k + 1});
            }
        }
        const auto vertices =
            static_cast<std::int64_t>(points.size() / 3);
        Mesh::Define(scene, "/World/Rings")
            .SetPoints(Tensor::Copy<double>(points, {vertices, 3}))
            .SetFaceVertexCounts(Tensor::Copy<std::int64_t>(
                counts, {static_cast<std::int64_t>(counts.size())}))
            .SetFaceVertexIndices(Tensor::Copy<std::int64_t>(
                indices, {static_cast<std::int64_t>(indices.size())}))
            .SetMaterialBinding("/World/Looks/Rings");
        scene.Write("/World/Rings", "primvars:st",
                    Tensor::Copy<float>(st, {vertices, 2}));
        Material::Define(scene, "/World/Looks/Rings");
        Shader::Define(scene, "/World/Looks/Rings/Surface")
            .SetInfoId(tokens::kUsdPreviewSurface)
            .SetDiffuseColor(glm::dvec3(0.6, 0.55, 0.45))
            .SetRoughness(1.0);
        Connect(scene, "/World/Looks/Rings", tokens::kSurfaceOutput,
                "/World/Looks/Rings/Surface", tokens::kSurfaceOutput);
        Shader::Define(scene, "/World/Looks/Rings/St")
            .SetInfoId(tokens::kUsdPrimvarReaderFloat2)
            .SetVarname("st");
    }

    /// Fetches the next map, wearing each as it comes.
    void StepMaps() {
        if (download_ == nullptr) {
            while (!maps_.empty()) {
                const std::size_t i = maps_.front();
                const std::string url = i < kBodies.size()
                                            ? UrlOf(kBodies[i].url)
                                            : UrlOf(kRingMap);
                const std::filesystem::path path = CachedPath(
                    url, i < kBodies.size() ? kBodies[i].folder
                                            : "solar_system");
                if (std::filesystem::exists(path)) {
                    UseMap(i, path);
                    maps_.erase(maps_.begin());
                    continue;
                }
                download_ =
                    std::make_unique<demos::AsyncDownload>(url, path);
                downloading_ = path;
                break;
            }
            if (download_ == nullptr) {
                if (progress_ != 0) {
                    Widgets().Remove(progress_);
                    progress_ = 0;
                }
                return;
            }
        }
        for (const auto& message : download_->TakeMessages()) {
            Report(message.text, message.error);
        }
        Widgets().Set(progress_, download_->Fraction());
        Widgets().Set(progress_, std::to_string(maps_.size()) +
                                     " left: " + download_->Status());
        if (download_->Done()) {
            if (download_->Succeeded()) {
                UseMap(maps_.front(), downloading_);
            }
            download_.reset();
            maps_.erase(maps_.begin());
        }
    }

    /// Body `i` wears map `path` (the rings' for `kBodies.size()`);
    /// the Sun's emitted.
    void UseMap(std::size_t i, const std::filesystem::path& path) {
        Scene& scene = GetScene();
        const std::string look =
            "/World/Looks/" + std::string(i < kBodies.size()
                                              ? kBodies[i].name
                                              : "Rings");
        Shader::Define(scene, look + "/Map")
            .SetInfoId(tokens::kUsdUVTexture)
            .SetSourceColorSpace("sRGB")
            .SetFile(path.string());
        Connect(scene, look + "/Map", tokens::kSt, look + "/St",
                "outputs:result");
        if (i == kSun) {
            scene.Set(look + "/Map", tokens::kShaderScale,
                      glm::dvec4(glm::dvec3(kSunLuminance), 1.0));
            Connect(scene, look + "/Surface", tokens::kEmissiveColor,
                    look + "/Map", "outputs:rgb");
            return;
        }
        Connect(scene, look + "/Surface", tokens::kDiffuseColor,
                look + "/Map", "outputs:rgb");
        if (i == kBodies.size()) {
            Connect(scene, look + "/Surface", tokens::kOpacity,
                    look + "/Map", "outputs:a");
            Shader(scene, look + "/Surface").SetOpacityThreshold(0.2);
        }
    }

    /// Days past J2000 at `year_`.
    double Days() const {
        return (year_ - 2000.0) * 365.25;
    }

    /// Body `i`'s centre on the date.
    glm::dvec3 Position(std::size_t i) const {
        if (i == kSun) {
            return glm::dvec3(0.0);
        }
        if (i == kMoon) {
            const MoonOrbit moon(Days());
            return Position(kEarth) + moon.At(moon.latitude);
        }
        const Orbit orbit(*kBodies[i].orbit, Days() / 36525.0);
        return orbit.At(orbit.anomaly);
    }

    /// Every body at its place and turn on the date, and the orbits.
    void Place() {
        Scene& scene = GetScene();
        for (std::size_t i = 0; i < kBodies.size(); ++i) {
            const Body& body = kBodies[i];
            const glm::dmat3 axes = Frame(body, Days());
            glm::dmat4 transform(axes);
            transform[3] = glm::dvec4(Position(i), 1.0);
            transform = glm::scale(
                transform,
                glm::dvec3(1.0, 1.0, body.pole / body.equator));
            Sphere(scene, "/World/" + std::string(body.name))
                .SetTransform(transform);
            if (i == kSaturn) {
                glm::dmat4 rings(axes);
                rings[3] = transform[3];
                Mesh(scene, "/World/Rings").SetTransform(rings);
            }
        }
        DrawOrbits();
    }

    /// The orbits about the Sun, and the Moon's about the Earth, and
    /// a label beside each body.
    void DrawOrbits() {
        Overlay& overlay = GetScene().GetOverlay("orbits");
        overlay.Clear();
        const glm::dvec3 eye = Eye();
        drawnFrom_ = eye;
        for (std::size_t i = 0; i < kBodies.size(); ++i) {
            const Body& body = kBodies[i];
            OverlayStyle style;
            style.color = glm::vec4(glm::vec3(body.colour), 0.7f);
            style.width = 1.5f;
            if (body.orbit) {
                const Orbit orbit(*body.orbit, Days() / 36525.0);
                overlay.Polyline(
                    OrbitPoints([&](double t) { return orbit.At(t); },
                                eye),
                    style, true);
            } else if (i == kMoon) {
                const MoonOrbit moon(Days());
                const glm::dvec3 earth = Position(kEarth);
                overlay.Polyline(
                    OrbitPoints(
                        [&](double t) { return earth + moon.At(t); },
                        eye),
                    style, true);
            }
            OverlayStyle label;
            label.color = glm::vec4(1.0f);
            label.size = 13.0f;
            label.depth = OverlayDepth::Always;
            overlay.Label(Position(i), body.name, label);
        }
    }

    /// Goes to body `i` (its day side, four radii out), or for the
    /// two after the bodies the inner planets or the whole system
    /// from above.
    void GoTo(std::size_t i) {
        glm::dvec3 eye;
        glm::dvec3 target(0.0);
        if (i < kBodies.size()) {
            target = Position(i);
            const double radius =
                i == kSaturn ? kRingOuter : kBodies[i].equator;
            // Between the body and the Sun, a little above the
            // ecliptic.
            const glm::dvec3 sunward = i == kSun
                                           ? glm::dvec3(1.0, 0.0, 0.0)
                                           : glm::normalize(-target);
            const glm::dvec3 side = glm::normalize(
                glm::cross(glm::dvec3(0, 0, 1), sunward));
            eye = target + 4.0 * radius *
                               glm::normalize(sunward + 0.6 * side +
                                              glm::dvec3(0, 0, 0.3));
        } else {
            const double reach =
                i == kBodies.size() ? 2.0 * kAu : 35.0 * kAu;
            eye = glm::dvec3(0.0, -0.6 * reach, reach);
        }
        if (app::Viewer* viewer = GetViewer()) {
            viewer->LookAt(eye, target);
        } else {
            Camera(GetScene(), "/World/Camera")
                .SetTransform(
                    LookAtTransform(eye, target, {0.0, 0.0, 1.0}));
        }
        // Exposed for the sunlight there: it falls as the square of
        // the distance from the Sun; the Sun itself 16 stops darker
        // than the Earth's daylight.
        sunlight_ = i == kSun ? std::exp2(16.0)
                    : i < kBodies.size()
                        ? std::pow(glm::length(target) / kAu, -2.0)
                        : 1.0;
        Expose();
        DrawOrbits();
    }

    /// The camera's exposure: the Earth's daylight's (f/8, 1/500 s at
    /// ISO 100) for `sunlight_` times as much light, `stops_` more.
    void Expose() {
        Camera(GetScene(), "/World/Camera")
            .SetExposureTime(std::exp2(stops_) / (500.0 * sunlight_));
    }

    /// The camera's place.
    glm::dvec3 Eye() {
        const glm::dmat4 transform =
            GetScene().Get(Path("/World/Camera"), tokens::kTransform,
                           glm::dmat4(1.0));
        return glm::dvec3(transform[3]);
    }

    /// The distance from `eye` to the nearest body's surface, at
    /// least a kilometre.
    double Nearest(const glm::dvec3& eye) const {
        double nearest = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < kBodies.size(); ++i) {
            nearest =
                std::min(nearest, glm::length(eye - Position(i)) -
                                      kBodies[i].equator);
        }
        return std::max(nearest, 1000.0);
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

    bool test_ = false;
    double stops_ = 0.0;
    double sunlight_ = 1.0;
    double year_ = 2026.8;
    std::vector<std::size_t> maps_;
    std::unique_ptr<demos::AsyncDownload> download_;
    std::filesystem::path downloading_;
    app::WidgetPanel::Id progress_ = 0;
    glm::dvec3 drawnFrom_{std::numeric_limits<double>::max()};
};

}  // namespace

ERNIE_REGISTER_DEMO(
    SolarSystemDemo, "planets", "solar_system",
    "The Sun, planets and Moon at their true sizes and distances on "
    "a "
    "chosen date, lit by the Sun, orbits drawn, each body mapped and "
    "streamed as a virtual texture.");
