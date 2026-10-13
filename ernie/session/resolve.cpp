// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/resolve.h"

#include "ernie/geometry/ellipsoid.h"
#include "ernie/geometry/primitives.h"
#include "ernie/geometry/sphere_tessellation.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/wide_view.h"
#include "ernie/scene/xform.h"
#include "ernie/session/volumes.h"
#include "ernie/util/astronomy.h"
#include "ernie/util/atmosphere.h"
#include "ernie/util/ies.h"
#include "ernie/util/reflectance_library.h"
#include "ernie/util/solar.h"
#include "ernie/util/spectrum.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

bool InSession(const Path& path) {
    static const Path root{std::string(tokens::kSessionRoot)};
    return path.HasPrefix(root);
}

/// The first prim of `type` outside the session root, or the empty
/// path.
Path FirstOutsideSession(const Scene& scene, std::string_view type) {
    for (const Path& path : scene.Prims(type)) {
        if (!InSession(path)) {
            return path;
        }
    }
    return {};
}

Path FirstVisible(const Scene& scene, std::string_view type) {
    for (const Path& path : scene.Prims(type)) {
        if (scene.IsVisible(path)) {
            return path;
        }
    }
    return {};
}

float Number(const Scene& scene, const Path& path,
             std::string_view attribute, float fallback) {
    return static_cast<float>(
        GetNumber(scene, path, attribute, fallback));
}

/// A CPU array of `components`-vectors as floats, shaped (n,
/// components) or flat.
std::vector<float> FloatsOf(const Tensor& tensor,
                            std::size_t components,
                            std::string_view what) {
    const bool shaped =
        tensor.Rank() == 2 &&
        tensor.Shape()[1] == static_cast<std::int64_t>(components);
    if (tensor.Location().kind != DeviceKind::Cpu ||
        !(tensor.Rank() == 1 || shaped) ||
        static_cast<std::size_t>(tensor.Elements()) % components !=
            0) {
        throw std::invalid_argument(
            std::string(what) + " must be a CPU array of " +
            std::to_string(components) + "-vectors");
    }
    const Tensor packed = tensor.Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    if (packed.Type() == DType::Float32) {
        return {packed.As<float>(), packed.As<float>() + count};
    }
    if (packed.Type() == DType::Float64) {
        return {packed.As<double>(), packed.As<double>() + count};
    }
    throw std::invalid_argument(std::string(what) +
                                " must be float32 or float64");
}

/// A CPU array of integers, of any shape.
std::vector<std::int64_t> IntsOf(const Tensor& tensor,
                                 std::string_view what) {
    if (tensor.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(std::string(what) +
                                    " must be a CPU array");
    }
    const Tensor packed = tensor.Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    switch (packed.Type()) {
    case DType::Int32:
        return {packed.As<std::int32_t>(),
                packed.As<std::int32_t>() + count};
    case DType::Int64:
        return {packed.As<std::int64_t>(),
                packed.As<std::int64_t>() + count};
    case DType::UInt32:
        return {packed.As<std::uint32_t>(),
                packed.As<std::uint32_t>() + count};
    default:
        throw std::invalid_argument(
            std::string(what) + " must be int32, int64 or uint32");
    }
}

/// The prim's attribute as an array: a written tensor or, for small
/// arrays, a float array value.
std::optional<std::vector<float>>
FloatAttribute(const Scene& scene, const Path& path,
               std::string_view attribute, std::size_t components) {
    if (const Tensor* tensor = scene.Read(path, attribute)) {
        return FloatsOf(*tensor, components, attribute);
    }
    return std::nullopt;
}

/// The `ernie:spectrum` pairs of a light, if it has them.
std::optional<std::vector<float>> SpectrumPairs(const Light& light) {
    return light.HasSpectrum() ? std::optional(light.GetSpectrum())
                               : std::nullopt;
}

/// UsdLux's `inputs:intensity` times 2^`inputs:exposure`.
double LightIntensity(const Light& light) {
    return light.GetIntensity() * std::exp2(light.GetExposure());
}

/// `LightIntensity`, divided by `size` when the light normalizes
/// (UsdLux's `inputs:normalize`: its world surface area, or a distant
/// light's projected solid angle); a size of 0, a point's, leaves it.
///
/// References:
/// - Pixar, UsdLux LightAPI, inputs:normalize,
///   https://openusd.org/dev/user_guides/schemas/usdLux/LightAPI.html
double LightIntensity(const Light& light, double size) {
    const double intensity = LightIntensity(light);
    return light.GetNormalize() && size > 0.0 ? intensity / size
                                              : intensity;
}

/// The black body temperature of a light with
/// `inputs:enableColorTemperature`, else 0.
double ColorTemperature(const Light& light) {
    return light.GetEnableColorTemperature()
               ? light.GetColorTemperature()
               : 0.0;
}

/// `shape` averaged over each of `bands` (eight samples per band), or
/// projected onto RGB mode's channels (`BandValue`), times `scale`.
template <typename Shape>
std::vector<float> BinShape(const Shape& shape,
                            const SpectralBands& bands,
                            double scale) {
    std::vector<float> values(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        if (bands.rgb) {
            values[static_cast<std::size_t>(i)] = static_cast<float>(
                scale * BandValue(bands, i, shape));
            continue;
        }
        double sum = 0.0;
        for (int k = 0; k < 8; ++k) {
            sum += shape(bands.Lower(i) +
                         (k + 0.5) * bands.Width() / 8.0);
        }
        values[static_cast<std::size_t>(i)] =
            static_cast<float>(scale * sum / 8.0);
    }
    return values;
}

/// Whether `nm` is within 380 to 780 nm, where Omniverse defines its
/// "white" and "blackbody" illuminants (at 5 nm intervals); they emit
/// nothing outside it.
bool InIlluminantRange(double nm) {
    return nm >= 380.0 && nm <= 780.0;
}

/// The relative shape of a light's spectrum over wavelength in nm
/// (`tokens::kPhysicalIlluminant`): its illuminant, else a black body
/// at the enabled colour temperature, else flat; times the shape of
/// `inputs:color` (upsampled with `RgbToSigmoid`). The "white" and
/// "blackbody" illuminants span 380 to 780 nm as Omniverse's do;
/// unset, the flat or black body shape spans the whole spectral
/// range, for sensors beyond the visible.
/// @throws std::invalid_argument for an unknown illuminant, a
/// "custom" one without pairs, or illuminant D outside 4000 to 25000
/// K.
///
/// References:
/// - NVIDIA, Omniverse lighting, PhysicalLightIlluminantAPI,
///   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
std::function<double(double)> IlluminantShape(const Light& light) {
    using namespace tokens;
    const std::string illuminant = light.GetPhysicalIlluminant();
    const double kelvin = light.GetColorTemperature();
    std::function<double(double)> base;
    if (illuminant.empty()) {
        const double temperature = ColorTemperature(light);
        base = [temperature](double w) {
            return temperature > 0.0 ? PlanckRadiance(w, temperature)
                                     : 1.0;
        };
    } else if (illuminant == kWhite) {
        base = [](double w) {
            return InIlluminantRange(w) ? 1.0 : 0.0;
        };
    } else if (illuminant == kBlackbody) {
        if (!(kelvin > 0.0)) {
            throw std::invalid_argument(
                "Light '" + light.GetPath().String() +
                "' needs a positive colour temperature");
        }
        base = [kelvin](double w) {
            return InIlluminantRange(w) ? PlanckRadiance(w, kelvin)
                                        : 0.0;
        };
    } else if (illuminant == kIlluminantD) {
        CieDaylight(560.0, kelvin);  // checks the temperature
        base = [kelvin](double w) { return CieDaylight(w, kelvin); };
    } else if (illuminant == kSun) {
        base = [](double w) { return SolarSpectralIrradiance()(w); };
    } else if (illuminant == kCustom) {
        const std::vector<float> pairs =
            light.GetPhysicalCustomIlluminant();
        if (pairs.size() < 4) {
            throw std::invalid_argument(
                "Light '" + light.GetPath().String() +
                "' has a custom illuminant but no '" +
                std::string(kPhysicalCustomIlluminant) + "' pairs");
        }
        const auto spectrum =
            std::make_shared<Spectrum>(Spectrum::FromPairs(pairs));
        const double low = spectrum->Wavelengths().front();
        const double high = spectrum->Wavelengths().back();
        base = [spectrum, low, high](double w) {
            return w < low || w > high
                       ? 0.0
                       : std::max((*spectrum)(w), 0.0);
        };
    } else {
        throw std::invalid_argument(
            "Light '" + light.GetPath().String() +
            "' has unknown illuminant '" + illuminant + "'");
    }
    const glm::dvec3 color = light.GetColor();
    if (color == glm::dvec3(1.0)) {
        return base;
    }
    const SigmoidSpectrum tint = RgbToSigmoid(color);
    return [base, tint](double w) { return base(w) * tint(w); };
}

/// How a sphere, rect or disk light's emission falls off around its
/// axis: its cone (spheres) and its focus.
struct EmissionShape {
    double coneAngle = 180.0;  ///< Degrees.
    double coneSoftness = 0.0;
    double focus = 0.0;
    double tintLuminance =
        0.0;  ///< Of the focus tint, what the focus falloff fades to.

    /// The emission at cosine `c` off the axis over that along it:
    /// the cone's smoothstep in the cosine (as the shading's) times
    /// lerp(tint, 1, |c|^focus).
    double operator()(double c) const {
        double cone = 1.0;
        if (coneAngle < 180.0) {
            const double angle =
                glm::radians(std::max(coneAngle, 0.0));
            const double outer = std::cos(angle);
            const double inner = std::cos(
                angle * (1.0 - std::clamp(coneSoftness, 0.0, 1.0)));
            const double t = std::clamp(
                (c - outer) / std::max(inner - outer, 1e-12), 0.0,
                1.0);
            cone = t * t * (3.0 - 2.0 * t);
        }
        const double focused =
            focus > 0.0 ? std::pow(std::abs(c), focus) : 1.0;
        return cone *
               (tintLuminance + (1.0 - tintLuminance) * focused);
    }
};

/// A light's IES profile grid (`LightParams::profile`) averaged
/// around its axis at cosine `c` off it, linear between rows.
double ProfileRing(const std::vector<float>& profile, double c) {
    constexpr int rows = LightParams::kProfileRows;
    constexpr int columns = LightParams::kProfileColumns;
    const double at =
        std::acos(std::clamp(c, -1.0, 1.0)) / M_PI * (rows - 1);
    const int row = std::min(static_cast<int>(at), rows - 2);
    const double t = at - row;
    double sum = 0.0;
    for (int j = 0; j < columns; ++j) {
        const double below = static_cast<double>(profile[
            static_cast<std::size_t>(row * columns + j)]);
        const double above = static_cast<double>(profile[
            static_cast<std::size_t>((row + 1) * columns + j)]);
        sum += (1.0 - t) * below + t * above;
    }
    return sum / columns;
}

/// The solid angle a light spreads its power over, so that a power
/// gives the same flux whatever its shape: for a sphere the integral
/// of `shape` over the sphere (4 pi unshaped), for a flat Lambertian
/// emitter of its cosine-weighted value over the hemisphere it faces
/// (pi unshaped), per unit area; times its IES `profile` when not
/// empty (in cd sr then). A focus tint counts by its luminance.
double EmittedSolidAngle(const EmissionShape& shape, bool flat,
                         const std::vector<float>& profile = {}) {
    constexpr int kSteps = 8192;
    const double low = flat ? 0.0 : -1.0;
    const double step = (1.0 - low) / kSteps;
    double sum = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const double c = low + (i + 0.5) * step;
        sum += (flat ? c : 1.0) * shape(c) *
               (profile.empty() ? 1.0 : ProfileRing(profile, c));
    }
    return std::max(2.0 * M_PI * sum * step, 1e-9);
}

/// The IES profile of `light` (`inputs:shaping:ies:file`) on
/// `LightParams::profile`'s grid, its vertical angle scaled by
/// `inputs:shaping:ies:angleScale` as UsdLux's ShapingAPI does, times
/// its tilt factor for the light's tilt in `world` (`IesTilt`); empty
/// without a file. Files are read once each, and again when they
/// change.
/// @throws std::runtime_error or std::invalid_argument for a file
/// that cannot be read.
///
/// References:
/// - Pixar, UsdLux ShapingAPI,
///   https://openusd.org/dev/user_guides/schemas/usdLux/ShapingAPI.html
std::vector<float> ProfileOf(const BoundableLight& light,
                             const glm::dmat4& world) {
    const std::string file = light.GetShapingIesFile();
    if (file.empty()) {
        return {};
    }
    struct Cached {
        std::filesystem::file_time_type written;
        std::shared_ptr<const IesProfile> profile;
    };
    static std::mutex mutex;
    static std::map<std::string, Cached> cache;
    const std::filesystem::file_time_type written =
        std::filesystem::last_write_time(file);
    std::shared_ptr<const IesProfile> profile;
    {
        const std::lock_guard lock(mutex);
        Cached& entry = cache[file];
        if (entry.profile == nullptr || entry.written != written) {
            entry = {written, std::make_shared<const IesProfile>(
                                  ReadIes(file))};
        }
        profile = entry.profile;
    }
    const double scale = light.GetShapingIesAngleScale();
    const double tilt = profile->tilt.Factor(
        profile->tilt.AngleOf(glm::normalize(glm::dvec3(world[0])),
                              glm::normalize(glm::dvec3(world[1])),
                              glm::normalize(glm::dvec3(world[2]))));
    constexpr int rows = LightParams::kProfileRows;
    constexpr int columns = LightParams::kProfileColumns;
    std::vector<float> grid(static_cast<std::size_t>(rows * columns));
    for (int i = 0; i < rows; ++i) {
        const double theta = 180.0 * i / (rows - 1);
        const double ies = scale > 0.0 ? theta / scale
                           : scale < 0.0
                               ? (theta - 180.0) / -scale + 180.0
                               : theta;
        for (int j = 0; j < columns; ++j) {
            grid[static_cast<std::size_t>(i * columns + j)] =
                static_cast<float>(
                    tilt * profile->Sample(ies, 360.0 * j / columns));
        }
    }
    return grid;
}

/// The profile along a light's axis (its nadir), else its brightest,
/// for an illuminance there.
double ProfileAxis(const std::vector<float>& profile) {
    const double axis = ProfileRing(profile, 1.0);
    return axis > 0.0 ? axis
                      : static_cast<double>(std::max(
                            *std::max_element(profile.begin(),
                                              profile.end()),
                            1e-9f));
}

/// A light's IES profile grid `theta` radians off its axis and `phi`
/// around it from its +x, bilinear (as the shading samples it).
double ProfileAt(const std::vector<float>& profile, double theta,
                 double phi) {
    constexpr int rows = LightParams::kProfileRows;
    constexpr int columns = LightParams::kProfileColumns;
    const double r = std::clamp(theta / M_PI, 0.0, 1.0) * (rows - 1);
    const int r0 = std::min(static_cast<int>(r), rows - 2);
    const double tr = r - r0;
    const double c = phi / (2.0 * M_PI) * columns;
    const int c0 =
        ((static_cast<int>(std::floor(c)) % columns) + columns) %
        columns;
    const int c1 = (c0 + 1) % columns;
    const double tc = c - std::floor(c);
    const auto at = [&](int row, int column) {
        return static_cast<double>(profile[static_cast<std::size_t>(
            row * columns + column)]);
    };
    return (1.0 - tr) * ((1.0 - tc) * at(r0, c0) + tc * at(r0, c1)) +
           tr * ((1.0 - tc) * at(r0 + 1, c0) + tc * at(r0 + 1, c1));
}

/// What a cylinder light's side spreads its power over, per unit of
/// its length times its diameter: the integral over directions of its
/// shape (about its -z axis) and profile times its side's projected
/// area there, which is sin of the angle to its length; pi^2
/// unshaped, so that times 2 r l it is pi times the side's area.
double CylinderEmittedSolidAngle(const EmissionShape& shape,
                                 const std::vector<float>& profile) {
    constexpr int kTheta = 512;
    constexpr int kPhi = 256;
    const double dTheta = M_PI / kTheta;
    const double dPhi = 2.0 * M_PI / kPhi;
    double sum = 0.0;
    for (int i = 0; i < kTheta; ++i) {
        const double theta = (i + 0.5) * dTheta;  // off the -z axis
        const double s = std::sin(theta);
        const double focused = shape(std::cos(theta));
        for (int j = 0; j < kPhi; ++j) {
            const double phi = (j + 0.5) * dPhi;
            const double along =
                s *
                std::cos(phi);  // its component along the length, +x
            const double projected =
                std::sqrt(std::max(1.0 - along * along, 0.0));
            sum += focused * projected * s *
                   (profile.empty() ? 1.0
                                    : ProfileAt(profile, theta, phi));
        }
    }
    return sum * dTheta * dPhi;
}

/// The shape of a sphere, rect or disk light's emission
/// (`EmissionShape`), with `coneAngle` and `coneSoftness` its cone's
/// (180 and 0 for none).
EmissionShape EmissionShapeOf(const BoundableLight& light,
                              double coneAngle, double coneSoftness) {
    const glm::dvec3 tint =
        glm::clamp(light.GetShapingFocusTint(), 0.0, 1.0);
    return {coneAngle, coneSoftness,
            std::max(light.GetShapingFocus(), 0.0),
            0.2126 * tint.r + 0.7152 * tint.g + 0.0722 * tint.b};
}

/// `light`'s focus and its tint per band of `bands`
/// (`LightParams::focus`), the tint upsampled as `inputs:color` is
/// (`RgbToSigmoid`); no tint when it is black.
void SetFocus(LightParams& light, const BoundableLight& emitter,
              const SpectralBands& bands) {
    light.focus =
        static_cast<float>(std::max(emitter.GetShapingFocus(), 0.0));
    const glm::dvec3 tint =
        glm::clamp(emitter.GetShapingFocusTint(), 0.0, 1.0);
    if (light.focus > 0.0f && tint != glm::dvec3(0.0)) {
        const SigmoidSpectrum shape = RgbToSigmoid(tint);
        light.focusTint =
            BinShape([&](double w) { return shape(w); }, bands, 1.0);
    }
}

/// Luminance of a Lambertian disk of `radius` that gives 1 lux to a
/// surface facing it on its axis `distance` away: the inverse of pi
/// R^2 / (R^2 + d^2), the disk's configuration factor times pi.
///
/// References:
/// - J. R. Howell, A Catalog of Radiation Heat Transfer Configuration
///   Factors, differential area to a parallel coaxial disk,
///   http://www.thermalradiation.net/indexCat.html
double DiskLuminancePerLux(double radius, double distance) {
    return (radius * radius + distance * distance) /
           (M_PI * radius * radius);
}

/// Luminance of a Lambertian `width` x `height` rectangle that gives
/// 1 lux to a surface facing it on its axis `distance` away: the
/// inverse of pi times four configuration factors to the quarter
/// rectangles, 2 (x / sqrt(1 + x^2) atan(y / sqrt(1 + x^2)) + the
/// same with x and y swapped) for x and y the half sizes over the
/// distance.
///
/// References:
/// - J. R. Howell, A Catalog of Radiation Heat Transfer Configuration
///   Factors, differential area to a parallel rectangle with a corner
///   above it, http://www.thermalradiation.net/indexCat.html
double RectLuminancePerLux(double width, double height,
                           double distance) {
    const double x = 0.5 * width / distance;
    const double y = 0.5 * height / distance;
    const double sx = std::sqrt(1.0 + x * x);
    const double sy = std::sqrt(1.0 + y * y);
    return 1.0 / (2.0 * (x / sx * std::atan(y / sx) +
                         y / sy * std::atan(x / sy)));
}

/// A photometric area light's luminous value (a luminous intensity
/// for a sphere, a luminance for a flat emitter), as Omniverse's
/// PhotometricAreaLightAPI gives it. With both an illuminance and its
/// distance above 0, the value lighting a surface facing it on its
/// axis that far away with that illuminance, `perLux(distance)`
/// giving the value per lux; otherwise its power in lumens over
/// `spread` (the cone's solid angle for a sphere, pi times the area
/// for a flat emitter), so it emits that power whatever its colour,
/// cone or focus.
///
/// References:
/// - NVIDIA, Omniverse lighting, PhotometricAreaLightAPI,
///   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
template <typename PerLux>
double PhotometricAreaValue(const BoundableLight& light,
                            double spread, const PerLux& perLux) {
    const double illuminance = light.GetPhotometricIlluminance();
    const double distance = light.GetPhotometricIlluminanceDistance();
    if (illuminance > 0.0 && distance > 0.0) {
        return illuminance * perLux(distance);
    }
    return light.GetPhotometricPower() / spread;
}

/// A light's spectrum in relative units, per band of `bands`: its
/// shape (`IlluminantShape`) scaled so that its luminous value (683
/// lm/W times its integral against the CIE y function) is `luminous`.
std::vector<float> RelativeSpectrum(const Light& light,
                                    const SpectralBands& bands,
                                    double luminous) {
    const std::function<double(double)> shape =
        IlluminantShape(light);
    // Midpoints of 1 nm steps, so narrow spectra integrate as wide as
    // they are.
    double luminance = 0.0;
    for (double w = 360.5; w < 830.0; w += 1.0) {
        luminance += shape(w) * CieXyz(w).y;
    }
    if (!(luminance > 0.0)) {
        throw std::invalid_argument(
            "Light '" + light.GetPath().String() +
            "' has no visible light to scale to a luminous value");
    }
    return BinShape(shape, bands,
                    luminous / (kLuminousEfficacy * luminance));
}

/// A physical lamp's spectrum per band (see `tokens::kLightPower`):
/// its power over `spread`, which is 4 pi sr for a sphere's radiant
/// intensity and pi times the area for an area light's radiance.
std::vector<float> LampSpectrum(const BoundableLight& light,
                                const SpectralBands& bands,
                                const std::vector<float>* pairs,
                                double spread) {
    const bool hasPower = light.HasPower();
    const double power = light.GetPower();
    if (power < 0.0) {
        throw std::invalid_argument("Light '" +
                                    light.GetPath().String() +
                                    "' has a negative power");
    }
    if (pairs != nullptr) {
        const Spectrum spectrum = Spectrum::FromPairs(*pairs);
        std::vector<float> binned = spectrum.Bin(bands);
        if (hasPower && !spectrum.Empty()) {
            const double low =
                std::max(360.0, spectrum.Wavelengths().front());
            const double high =
                std::min(830.0, spectrum.Wavelengths().back());
            const double total =
                high > low
                    ? spectrum.Average(low, high) * (high - low)
                    : 0.0;
            const double scale =
                total > 0.0 ? power / spread / total : 0.0;
            for (float& value : binned) {
                value = static_cast<float>(
                    static_cast<double>(value) * scale);
            }
        }
        return binned;
    }
    const double temperature = ColorTemperature(light);
    const auto shape = [temperature](double w) {
        return temperature > 0.0 ? PlanckRadiance(w, temperature)
                                 : 1.0;
    };
    double total = 0.0;
    for (double w = 360.5; w < 830.0; w += 1.0) {
        total += shape(w);
    }
    return BinShape(shape, bands, power / spread / total);
}

/// Real spherical harmonics up to order 2 at unit direction `d`, in
/// the order (0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0),
/// (2,1), (2,2).
std::array<double, 9> ShBasis(const glm::dvec3& d) {
    return {0.282095,
            0.488603 * d.y,
            0.488603 * d.z,
            0.488603 * d.x,
            1.092548 * d.x * d.y,
            1.092548 * d.y * d.z,
            0.315392 * (3.0 * d.z * d.z - 1.0),
            1.092548 * d.x * d.z,
            0.546274 * (d.x * d.x - d.y * d.y)};
}

}  // namespace

std::string_view WireframeToken(Wireframe mode) {
    switch (mode) {
    case Wireframe::Off:
        return tokens::kOff;
    case Wireframe::Overlay:
        return tokens::kOverlay;
    case Wireframe::HiddenLine:
        return tokens::kHiddenLine;
    }
    return tokens::kOff;
}

MinimalMode MinimalModeOf(const Scene& scene, const Path& prim,
                          MinimalMode fallback) {
    if (prim.IsEmpty() || !scene.Has(prim) ||
        scene.Get(prim, tokens::kMinimalMode) == nullptr) {
        return fallback;
    }
    const std::string token =
        RenderSettingsBase(scene, prim).GetMinimalMode();
    if (token == tokens::kOff) {
        return MinimalMode::Off;
    }
    if (token == tokens::kNone) {
        return MinimalMode::None;
    }
    if (token == tokens::kConstantDiffuse) {
        return MinimalMode::Constant;
    }
    if (token == tokens::kTexturedDiffuse) {
        return MinimalMode::Textured;
    }
    if (token == tokens::kDiffuseGlossyEmission) {
        return MinimalMode::Glossy;
    }
    throw std::invalid_argument(
        "'" + std::string(tokens::kMinimalMode) + "' of '" +
        prim.String() + "' is '" + token +
        "', not off, none, constantDiffuse, texturedDiffuse or "
        "diffuseGlossyEmission");
}

Wireframe WireframeOf(const Scene& scene, const Path& settings) {
    const Value* value =
        settings.IsEmpty() ? nullptr
                           : scene.Get(settings, tokens::kWireframe);
    if (value == nullptr) {
        return Wireframe::Off;
    }
    // A bool, as the viewport's toggle once wrote it: on is
    // "overlay".
    if (const bool* on = value->Get<bool>()) {
        return *on ? Wireframe::Overlay : Wireframe::Off;
    }
    if (const int* on = value->Get<int>()) {
        return *on != 0 ? Wireframe::Overlay : Wireframe::Off;
    }
    const std::string token =
        RenderSettings(scene, settings).GetWireframe();
    for (const Wireframe mode : {Wireframe::Off, Wireframe::Overlay,
                                 Wireframe::HiddenLine}) {
        if (token == WireframeToken(mode)) {
            return mode;
        }
    }
    throw std::invalid_argument(
        "'" + std::string(tokens::kWireframe) + "' of '" +
        settings.String() +
        "' must be \"off\", \"overlay\" or \"hiddenLine\"");
}

Occlusion OcclusionOf(const Scene& scene, const Path& settings) {
    const std::string token =
        RenderSettings::OrFallbacks(scene, settings).GetOcclusion();
    if (token == tokens::kOff) {
        return Occlusion::Off;
    }
    if (token == tokens::kOccluders) {
        return Occlusion::Occluders;
    }
    if (token == tokens::kTwoPass) {
        return Occlusion::TwoPass;
    }
    throw std::invalid_argument(
        "'" + std::string(tokens::kOcclusion) + "' of '" +
        settings.String() +
        "' must be \"off\", \"occluders\" or \"twoPass\"");
}

Culling CullingOf(const Scene& scene, const Path& settings) {
    const std::string token =
        RenderSettings::OrFallbacks(scene, settings).GetCulling();
    if (token == tokens::kOff) {
        return Culling::Off;
    }
    if (token == tokens::kCpu) {
        return Culling::Cpu;
    }
    if (token == tokens::kGpu) {
        return Culling::Gpu;
    }
    throw std::invalid_argument(
        "'" + std::string(tokens::kCulling) + "' of '" +
        settings.String() + "' must be \"off\", \"cpu\" or \"gpu\"");
}

std::optional<std::uint64_t> FrameOf(const Scene& scene,
                                     const Path& settings) {
    const Value* value = settings.IsEmpty()
                             ? nullptr
                             : scene.Get(settings, tokens::kFrame);
    if (value == nullptr) {
        return std::nullopt;
    }
    // As a double, not the view's int (`RenderSettings::GetFrame`),
    // to reject fractions and frames past an int's range; not a
    // number reads as -1, rejected below.
    const double frame =
        AttributeTraits<double>::From(*value).value_or(-1.0);
    if (!(frame >= 0.0) || frame != std::floor(frame) ||
        frame >= 0x1p64) {
        throw std::invalid_argument(
            "'" + std::string(tokens::kFrame) + "' of '" +
            settings.String() + "' must be a non-negative integer");
    }
    return static_cast<std::uint64_t>(frame);
}

Path BoundMaterial(const Scene& scene, const Path& prim) {
    const Path material =
        GetTarget(scene, prim, tokens::kMaterialBinding);
    if (scene.Has(material) &&
        scene.TypeOf(material) == tokens::kMaterial) {
        return material;
    }
    return {};
}

RenderPrims ResolveRenderPrims(const Scene& scene,
                               const Path& settings) {
    RenderPrims prims;
    if (!settings.IsEmpty()) {
        if (!scene.Has(settings) ||
            scene.TypeOf(settings) != tokens::kRenderSettings) {
            throw std::invalid_argument(
                "'" + settings.String() +
                "' is not a RenderSettings prim");
        }
        prims.settings = settings;
    } else {
        prims.settings =
            FirstOutsideSession(scene, tokens::kRenderSettings);
    }
    // The lights the settings name, when they name any (an empty list
    // naming none).
    std::optional<std::vector<Path>> chosen;
    if (!prims.settings.IsEmpty()) {
        const RenderSettings view(scene, prims.settings);
        const std::vector<Path> cameras = view.GetCamera();
        if (!cameras.empty()) {
            prims.camera = cameras.front();
        }
        if (view.HasLights()) {
            // Checked, so that a value of another type is an error
            // rather than no lights.
            const Value* value =
                scene.Get(prims.settings, tokens::kLightsRel);
            if (value == nullptr ||
                !(value->Is<Path>() ||
                  value->Is<std::vector<Path>>())) {
                throw std::invalid_argument(
                    "'" + std::string(tokens::kLightsRel) + "' of '" +
                    prims.settings.String() +
                    "' is not a relationship");
            }
            chosen = view.GetLights();
        }
    }
    if (prims.camera.IsEmpty()) {
        prims.camera = FirstOutsideSession(scene, tokens::kCamera);
    }
    prims.terrain = FirstVisible(scene, tokens::kTerrain);
    if (!prims.terrain.IsEmpty()) {
        prims.material = BoundMaterial(scene, prims.terrain);
    }
    std::vector<Path> candidates;
    if (chosen) {
        for (const Path& path : *chosen) {
            if (scene.Has(path) && scene.IsVisible(path)) {
                candidates.push_back(path);
            }
        }
    } else {
        for (const std::string_view type :
             {tokens::kDistantLight, tokens::kSphereLight,
              tokens::kRectLight, tokens::kDiskLight,
              tokens::kCylinderLight, tokens::kLaserLight,
              tokens::kDomeLight}) {
            for (const Path& path : scene.Prims(type)) {
                if (!InSession(path) && scene.IsVisible(path)) {
                    candidates.push_back(path);
                }
            }
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(
        std::unique(candidates.begin(), candidates.end()),
        candidates.end());
    for (const Path& path : candidates) {
        const std::string& type = scene.TypeOf(path);
        if (type == tokens::kDomeLight) {
            if (prims.dome.IsEmpty()) {
                prims.dome = path;
            }
        } else if (type == tokens::kDistantLight ||
                   type == tokens::kSphereLight ||
                   type == tokens::kRectLight ||
                   type == tokens::kDiskLight ||
                   type == tokens::kCylinderLight ||
                   type == tokens::kLaserLight) {
            prims.lights.push_back(path);
        }
    }
    prims.ground = FirstVisible(scene, tokens::kGroundPlane);
    prims.stars = FirstVisible(scene, tokens::kStarField);
    prims.atmosphere = FirstVisible(scene, tokens::kAtmosphere);
    for (const Path& path : scene.Prims(tokens::kCelestialBody)) {
        if (prims.bodies.size() < FrameParams::kMaxBodies &&
            scene.IsVisible(path)) {
            prims.bodies.push_back(path);
        }
    }
    return prims;
}

namespace {

void CheckPosesShape(const Tensor& tensor, const Path& camera) {
    const bool shaped =
        (tensor.Rank() == 3 && tensor.Shape()[1] == 4 &&
         tensor.Shape()[2] == 4) ||
        (tensor.Rank() == 2 && tensor.Shape()[1] == 16) ||
        (tensor.Rank() == 2 && tensor.Shape()[1] == 7);
    if (!shaped || tensor.Shape()[0] < 1) {
        throw std::invalid_argument("'" +
                                    std::string(tokens::kPoses) +
                                    "' of '" + camera.String() +
                                    "' must be shaped (views, 4, 4), "
                                    "or (views, 7) for positions "
                                    "and quaternions");
    }
}

}  // namespace

bool PosesAreQuaternions(const Tensor& poses) {
    return poses.Rank() == 2 && poses.Shape()[1] == 7;
}

namespace {}  // namespace

void CheckBatchCameras(const Scene& scene,
                       std::span<const Path> cameras,
                       std::uint32_t width, std::uint32_t height) {
    if (cameras.empty()) {
        return;
    }
    // Projections and bands only: the stand-in transform leaves them
    // alone.
    const glm::dmat4 any(1.0);
    const CameraParams first =
        CameraParamsOf(scene, cameras.front(), width, height, any);
    const glm::mat4 projection = first.RasterView().projection;
    for (const Path& path : cameras) {
        const CameraParams camera =
            CameraParamsOf(scene, path, width, height, any);
        const glm::mat4 other = camera.RasterView().projection;
        bool same = camera.bands == first.bands;
        for (int c = 0; c < 4 && same; ++c) {
            for (int r = 0; r < 4 && same; ++r) {
                same = std::abs(other[c][r] - projection[c][r]) <=
                       1e-5f *
                           std::max(1.0f, std::abs(projection[c][r]));
            }
        }
        if (!same) {
            throw std::invalid_argument(
                "Camera '" + path.String() +
                "' differs in projection or bands from '" +
                cameras.front().String() +
                "', so they cannot render as one batch");
        }
    }
}

glm::dmat4 PosesParentOf(const Scene& scene, const Path& camera) {
    const Path parent = camera.Parent();
    return parent.IsEmpty() || parent.IsRoot() || !scene.Has(parent)
               ? glm::dmat4(1.0)
               : scene.WorldTransform(parent);
}

const Tensor* GpuPosesOf(const Scene& scene, const Path& camera) {
    const Tensor* tensor = scene.Read(camera, tokens::kPoses);
    if (tensor == nullptr || !IsGpu(tensor->Location().kind)) {
        return nullptr;
    }
    CheckPosesShape(*tensor, camera);
    const std::string device = Camera(scene, camera).GetPosesDevice();
    if (device == tokens::kHost) {
        return nullptr;
    }
    if (device != tokens::kGpu) {
        throw std::invalid_argument(
            "'" + std::string(tokens::kPosesDevice) + "' of '" +
            camera.String() + "' must be \"gpu\" or \"host\"");
    }
    return tensor->Type() == DType::Float32 && tensor->IsContiguous()
               ? tensor
               : nullptr;
}

std::vector<glm::dmat4> ViewPosesOf(const Scene& scene,
                                    std::span<const Path> cameras,
                                    std::uint32_t width,
                                    std::uint32_t height,
                                    const Tensor* hostPoses) {
    using namespace tokens;
    for (const Path& camera : cameras) {
        if (!scene.Has(camera) || scene.TypeOf(camera) != kCamera) {
            throw std::invalid_argument("'" + camera.String() +
                                        "' is not a Camera prim");
        }
    }
    std::vector<glm::dmat4> poses;
    if (cameras.size() > 1) {
        CheckBatchCameras(scene, cameras, width, height);
        for (const Path& path : cameras) {
            poses.push_back(scene.WorldTransform(path));
        }
        return poses;
    }
    if (cameras.size() != 1) {
        return poses;
    }
    const Path& camera = cameras.front();
    const Tensor* tensor = hostPoses != nullptr
                               ? hostPoses
                               : Camera(scene, camera).GetPoses();
    if (tensor == nullptr) {
        return poses;
    }
    CheckPosesShape(*tensor, camera);
    if (tensor->Location().kind != DeviceKind::Cpu ||
        (tensor->Type() != DType::Float32 &&
         tensor->Type() != DType::Float64)) {
        throw std::invalid_argument(
            "'" + std::string(kPoses) + "' of '" + camera.String() +
            "' must be a CPU float32 or float64 array");
    }
    const Tensor packed = tensor->Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    std::vector<double> values;
    if (packed.Type() == DType::Float32) {
        values.assign(packed.As<float>(), packed.As<float>() + count);
    } else {
        values.assign(packed.As<double>(),
                      packed.As<double>() + count);
    }
    const glm::dmat4 parentWorld = PosesParentOf(scene, camera);
    if (PosesAreQuaternions(packed)) {
        // Rows of a position and a unit quaternion (x, y, z, qx, qy,
        // qz, qw, or with w first).
        const bool wFirst = QuaternionsWFirst(scene, camera);
        for (std::size_t first = 0; first < values.size();
             first += 7) {
            const double* row = values.data() + first;
            const glm::dquat q =
                wFirst ? glm::dquat(row[3], row[4], row[5], row[6])
                       : glm::dquat(row[6], row[3], row[4], row[5]);
            glm::dmat4 local(glm::mat3_cast(glm::normalize(q)));
            local[3] = glm::dvec4(row[0], row[1], row[2], 1.0);
            poses.push_back(parentWorld * local);
        }
        return poses;
    }
    for (std::size_t first = 0; first < values.size(); first += 16) {
        glm::dmat4 local(1.0);
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                local[col][row] =
                    values[first +
                           static_cast<std::size_t>(row * 4 + col)];
            }
        }
        poses.push_back(parentWorld * local);
    }
    return poses;
}

std::int32_t WorldOf(const Scene& scene, const Path& path) {
    for (Path at = path; !at.IsEmpty() && !at.IsRoot();
         at = at.Parent()) {
        if (scene.Has(at) &&
            scene.Get(at, tokens::kWorld) != nullptr) {
            return static_cast<std::int32_t>(
                GetNumber(scene, at, tokens::kWorld, -1.0));
        }
    }
    return -1;
}

namespace {

/// `tensor`, the `ernie:worlds` of `path`, if it has `count` of them,
/// as worlds.
std::optional<std::vector<std::int32_t>>
WorldsAttribute(const Path& path, const Tensor* tensor,
                std::size_t count, const HostArrayReader& read) {
    if (tensor == nullptr) {
        return std::nullopt;
    }
    std::optional<Tensor> host;
    if (tensor->Location().kind != DeviceKind::Cpu && read) {
        host = read(path, tokens::kWorlds, *tensor);
    }
    const std::vector<std::int64_t> values =
        IntsOf(host ? *host : *tensor, tokens::kWorlds);
    if (values.size() != count) {
        throw std::invalid_argument(
            "'" + std::string(tokens::kWorlds) + "' of '" +
            path.String() + "' has " + std::to_string(values.size()) +
            " worlds for " + std::to_string(count));
    }
    return std::vector<std::int32_t>(values.begin(), values.end());
}

}  // namespace

std::vector<std::int32_t> ViewWorldsOf(const Scene& scene,
                                       std::span<const Path> cameras,
                                       std::size_t views,
                                       const HostArrayReader& read) {
    std::vector<std::int32_t> worlds;
    if (cameras.size() > 1) {
        for (const Path& camera : cameras) {
            worlds.push_back(WorldOf(scene, camera));
        }
        return worlds;
    }
    if (cameras.empty()) {
        return std::vector<std::int32_t>(views, -1);
    }
    if (scene.Read(cameras.front(), tokens::kPoses) != nullptr) {
        if (auto perView = WorldsAttribute(
                cameras.front(),
                scene.Read(cameras.front(), tokens::kWorlds), views,
                read)) {
            return *perView;
        }
    }
    return std::vector<std::int32_t>(views,
                                     WorldOf(scene, cameras.front()));
}

std::vector<std::int32_t>
InstanceWorldsOf(const Scene& scene, const Path& instancer,
                 std::size_t count, const HostArrayReader& read) {
    if (auto perInstance = WorldsAttribute(
            instancer, PointInstancer(scene, instancer).GetWorlds(),
            count, read)) {
        return *perInstance;
    }
    return std::vector<std::int32_t>(count,
                                     WorldOf(scene, instancer));
}

namespace {

/// The sensor filter chain of `filters`, a product's or render var's
/// `ernie:filters`: each SensorFilter prim's type, with every number
/// or float array under its inputs:.
std::vector<SensorFilterStep>
FilterStepsOf(const Scene& scene, const std::vector<Path>& filters) {
    using namespace tokens;
    std::vector<SensorFilterStep> steps;
    for (const Path& filter : filters) {
        if (!scene.Has(filter) ||
            scene.TypeOf(filter) != kSensorFilter) {
            throw std::invalid_argument(
                "'" + filter.String() +
                "' is not a SensorFilter prim");
        }
        SensorFilterStep step;
        step.type = SensorFilterPrim(scene, filter).GetFilterType();
        if (step.type.empty()) {
            throw std::invalid_argument(
                "Sensor filter '" + filter.String() + "' has no '" +
                std::string(kFilterType) + "'");
        }
        constexpr std::string_view kInputs = "inputs:";
        for (const std::string& attribute :
             scene.Attributes(filter)) {
            if (!attribute.starts_with(kInputs)) {
                continue;
            }
            const std::string name = attribute.substr(kInputs.size());
            const Value* value = scene.Get(filter, attribute);
            const bool scalar = value != nullptr &&
                                (value->Get<double>() != nullptr ||
                                 value->Get<float>() != nullptr ||
                                 value->Get<int>() != nullptr ||
                                 value->Get<bool>() != nullptr);
            // Arrays, also from the vectors short lists arrive as
            // (`GetFloats`).
            if (std::optional<std::vector<float>> list =
                    scalar ? std::nullopt
                           : GetFloats(scene, filter, attribute)) {
                step.arrays[name] = std::move(*list);
            } else if (const Tensor* tensor =
                           scene.Read(filter, attribute)) {
                step.arrays[name] = FloatsOf(*tensor, 1, attribute);
            } else {
                step.inputs[name] = static_cast<float>(
                    GetNumber(scene, filter, attribute, 0.0));
            }
        }
        steps.push_back(std::move(step));
    }
    return steps;
}

}  // namespace

std::vector<RenderProductSpec>
ResolveRenderProducts(const Scene& scene,
                      std::span<const Path> products,
                      const Path& settings) {
    using namespace tokens;
    const Path resolved =
        ResolveRenderPrims(scene, settings).settings;
    const glm::dvec2 settingsResolution =
        RenderSettings::OrFallbacks(scene, resolved).GetResolution();
    std::vector<Path> paths(products.begin(), products.end());
    if (paths.empty() && !resolved.IsEmpty()) {
        paths = RenderSettings(scene, resolved).GetProducts();
    }
    const auto extentOf = [](glm::dvec2 resolution) {
        if (!(resolution.x >= 1.0) || !(resolution.y >= 1.0)) {
            throw std::invalid_argument(
                "Render resolutions must be positive");
        }
        return VkExtent2D{static_cast<std::uint32_t>(resolution.x),
                          static_cast<std::uint32_t>(resolution.y)};
    };
    std::vector<RenderProductSpec> specs;
    if (paths.empty()) {
        RenderProductSpec spec;
        spec.product = resolved.IsEmpty() ? Path::Root() : resolved;
        spec.resolution = extentOf(settingsResolution);
        spec.vars.emplace_back(spec.product.IsRoot()
                                   ? Path("/LdrColor")
                                   : spec.product.Child("LdrColor"),
                               std::string(kLdrColor));
        spec.varFilters.push_back(std::nullopt);
        specs.push_back(std::move(spec));
        return specs;
    }
    for (const Path& path : paths) {
        if (!scene.Has(path) ||
            scene.TypeOf(path) != kRenderProduct) {
            throw std::invalid_argument(
                "'" + path.String() +
                "' is not a RenderProduct prim");
        }
        const RenderProduct product(scene, path);
        RenderProductSpec spec;
        spec.product = path;
        // One camera, or a batch of them.
        spec.cameras = product.GetCamera();
        spec.camera =
            spec.cameras.empty() ? Path() : spec.cameras.front();
        // What the product leaves unset comes from its settings.
        spec.resolution =
            extentOf(product.HasResolution() ? product.GetResolution()
                                             : settingsResolution);
        if (product.HasOverlays()) {
            spec.overlays = product.GetOverlays();
        }
        if (product.HasMinimalMode()) {
            spec.minimal =
                MinimalModeOf(scene, path, MinimalMode::Off);
        }
        if (product.HasMinimalConstantColor()) {
            spec.minimalColor =
                glm::vec3(product.GetMinimalConstantColor());
        }
        bool ownChains = false;
        for (const Path& var : product.GetOrderedVars()) {
            if (!scene.Has(var) || scene.TypeOf(var) != kRenderVar) {
                throw std::invalid_argument(
                    "'" + var.String() + "' is not a RenderVar prim");
            }
            const RenderVar view(scene, var);
            spec.vars.emplace_back(var, view.GetSourceName());
            // A var with its own `ernie:filters` comes from that
            // chain, not the product's.
            if (view.HasFilters()) {
                spec.varFilters.push_back(
                    FilterStepsOf(scene, view.GetFilters()));
                ownChains = true;
            } else {
                spec.varFilters.push_back(std::nullopt);
            }
        }
        spec.filters = FilterStepsOf(scene, product.GetFilters());
        // The built-in filters decide which vars the chain writes. A
        // chain with custom filters (which may integrate in place of
        // qe or cie), or vars with chains of their own, are checked
        // by the renderer, which knows the custom filters
        // (`VarsWrittenBy` with its registry).
        std::vector<SensorFilter> builtins;
        bool custom = ownChains;
        for (const SensorFilterStep& step : spec.filters) {
            if (const std::optional<SensorFilter> builtin =
                    step.Builtin()) {
                builtins.push_back(*builtin);
            } else {
                custom = true;
            }
        }
        if (custom) {
            if (spec.vars.empty()) {
                spec.vars.emplace_back(path.Child("LdrColor"),
                                       std::string(kLdrColor));
                spec.varFilters.push_back(std::nullopt);
            }
            specs.push_back(std::move(spec));
            continue;
        }
        const std::vector<SensorFilter> chain = CheckFilterChain(
            spec.filters.empty() ? DefaultFilterChain() : builtins);
        const std::vector<std::string_view> written =
            VarsWrittenBy(chain);
        for (const auto& [var, name] : spec.vars) {
            // AOVs ("aov:<output>") come from custom filters, which
            // the renderer checks.
            if (name.starts_with(kAovPrefix) &&
                name.size() > kAovPrefix.size()) {
                continue;
            }
            const auto& names = kRenderVarNames;
            if (std::find(names.begin(), names.end(), name) ==
                names.end()) {
                throw std::invalid_argument(
                    "Render var '" + var.String() + "' asks for '" +
                    name + "', which Ernie does not render");
            }
            if (std::find(written.begin(), written.end(), name) ==
                written.end()) {
                throw std::invalid_argument(
                    "Render var '" + var.String() + "' asks for '" +
                    name +
                    "', which the product's sensor filters do not "
                    "write");
            }
        }
        if (spec.vars.empty()) {
            spec.vars.emplace_back(path.Child("LdrColor"),
                                   std::string(kLdrColor));
            spec.varFilters.push_back(std::nullopt);
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

std::vector<float> SunIrradianceOf(const Scene& scene,
                                   const Path& path,
                                   const SpectralBands& bands) {
    bands.Check();
    const DistantLight sun(scene, path);
    if (sun.IsPhotometric()) {
        return RelativeSpectrum(sun, bands,
                                sun.GetPhotometricIlluminance());
    }
    const double temperature = ColorTemperature(sun);
    const std::optional<std::vector<float>> pairs =
        SpectrumPairs(sun);
    if (sun.HasIrradiance() &&
        sun.GetPhysicalIlluminant() == tokens::kSun) {
        // The measured solar spectrum, scaled to the irradiance over
        // its whole range: the Sun's own at 1361 W/m^2, or at another
        // distance by the inverse square.
        std::vector<float> binned =
            SolarSpectralIrradiance().Bin(bands);
        const double scale =
            sun.GetIrradiance() / TotalSolarIrradiance();
        for (float& value : binned) {
            value = static_cast<float>(static_cast<double>(value) *
                                       scale);
        }
        return binned;
    }
    if (sun.HasIrradiance() && !pairs &&
        !sun.GetPhysicalIlluminant().empty()) {
        // Another illuminant's shape (`IlluminantShape`), scaled to
        // the irradiance over all it emits (1 nm steps to 3000 nm).
        const std::function<double(double)> shape =
            IlluminantShape(sun);
        double total = 0.0;
        for (double w = 0.5; w < 3000.0; w += 1.0) {
            total += shape(w);
        }
        if (!(total > 0.0)) {
            throw std::invalid_argument(
                "Light '" + path.String() +
                "' has an illuminant that emits nothing");
        }
        return BinShape(shape, bands, sun.GetIrradiance() / total);
    }
    if (sun.HasIrradiance() || pairs) {
        if (pairs) {
            const Spectrum spectrum = Spectrum::FromPairs(*pairs);
            std::vector<float> binned = spectrum.Bin(bands);
            if (sun.HasIrradiance() && !spectrum.Empty()) {
                const double total =
                    spectrum.Average(spectrum.Wavelengths().front(),
                                     spectrum.Wavelengths().back()) *
                    (spectrum.Wavelengths().back() -
                     spectrum.Wavelengths().front());
                const double scale = sun.GetIrradiance() / total;
                for (float& value : binned) {
                    value = static_cast<float>(
                        static_cast<double>(value) * scale);
                }
            }
            return binned;
        }
        return BlackbodyIrradiance(
            bands, temperature > 0.0 ? temperature : 5772.0,
            sun.GetIrradiance());
    }
    // A white surface facing it reads its intensity: pi times it in
    // lux. Normalized, as UsdLux's luminance over the disk's
    // projected solid angle, the intensity is the illuminance.
    return RelativeSpectrum(sun, bands,
                            (sun.GetNormalize() ? 1.0 : M_PI) *
                                LightIntensity(sun));
}

namespace {

/// A laser line's spectral power per band of `bands`, W nm^-1: a
/// Gaussian of FWHM `linewidth` nm around `wavelength` nm carrying
/// `power` W, each band's share integrated exactly (with erf), so a
/// line far narrower than a band keeps its power; a linewidth of 0
/// puts it all in the band holding the wavelength.
std::vector<float> LaserLine(const SpectralBands& bands,
                             double wavelength, double linewidth,
                             double power) {
    std::vector<float> values(static_cast<std::size_t>(bands.count),
                              0.0f);
    const double sigma =
        linewidth / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    const auto below = [&](double nm) {
        if (sigma <= 0.0) {
            return nm >= wavelength ? 1.0 : 0.0;
        }
        return 0.5 * std::erfc((wavelength - nm) /
                               (sigma * std::sqrt(2.0)));
    };
    for (int i = 0; i < bands.count; ++i) {
        const double share = below(bands.Lower(i) + bands.Width()) -
                             below(bands.Lower(i));
        values[static_cast<std::size_t>(i)] =
            static_cast<float>(power * share / bands.Width());
    }
    return values;
}

/// A laser's pattern image `file` (`LoadLightPatternImage`), or null
/// without one. Files are read once each, and again when they
/// change, so lasers sharing one share its texels.
/// @throws std::runtime_error or std::invalid_argument for a file
/// that cannot be read.
std::shared_ptr<const LightPatternImage>
PatternImageOf(const std::string& file) {
    if (file.empty()) {
        return nullptr;
    }
    struct Cached {
        std::filesystem::file_time_type written;
        std::shared_ptr<const LightPatternImage> image;
    };
    static std::mutex mutex;
    static std::map<std::string, Cached> cache;
    const std::filesystem::file_time_type written =
        std::filesystem::last_write_time(file);
    const std::lock_guard lock(mutex);
    Cached& entry = cache[file];
    if (entry.image == nullptr || entry.written != written) {
        entry = {written, std::make_shared<const LightPatternImage>(
                              LoadLightPatternImage(file))};
    }
    return entry.image;
}

/// The LaserLight at `path`'s pattern image array `array` on the host
/// (`LightPatternImageOf`), built again only when it is written
/// again.
std::shared_ptr<const LightPatternImage>
PatternImageOfArray(const Scene& scene, const Path& path,
                    const Tensor& array) {
    struct Cached {
        std::uint64_t version = 0;
        std::shared_ptr<const LightPatternImage> image;
    };
    static std::mutex mutex;
    static std::map<std::pair<std::uint64_t, std::string>, Cached>
        cache;
    const std::uint64_t version =
        scene.AttributeVersion(path, tokens::kLaserPatternImageArray);
    const std::lock_guard lock(mutex);
    Cached& entry = cache[{scene.Id(), path.String()}];
    if (entry.image == nullptr || entry.version != version) {
        entry = {version, std::make_shared<const LightPatternImage>(
                              LightPatternImageOf(array))};
    }
    return entry.image;
}

/// The LaserLight at `path` under transform `world` as a light: its
/// beam from the waist, divergence and power it gives (`LaserLight`),
/// its power through its aperture, and a shadow view along its beam.
/// `light`'s linear polarization: `degree` of its light, its electric
/// field `degrees` from its +x `x` about `travel`, the way its light
/// goes, counterclockwise looking along it.
void SetPolarization(LightParams& light, double degree,
                     double degrees, const glm::dvec3& travel,
                     const glm::dvec3& x) {
    light.polarization =
        static_cast<float>(std::clamp(degree, 0.0, 1.0));
    const double angle = glm::radians(degrees);
    // Looking along `travel` with x to the right, up is x cross
    // travel, where counterclockwise turns x first.
    const glm::dvec3 y = glm::cross(x, travel);
    light.polarizationAxis = glm::vec3(
        glm::normalize(std::cos(angle) * x + std::sin(angle) * y));
}

/// The pulses of the light at `path` (`tokens::kPulseRate`) into
/// `light`'s timing: a laser's shape and trigger beside the rate,
/// width and delay `LaserParamsOf` read; another light's all, its
/// intensity its level as a pulse starts.
/// @throws std::invalid_argument for a negative rate, decay or
/// jitter, pulses that do not fit their rate, or an unknown trigger.
void SetPulses(const Scene& scene, const Path& path,
               LightParams& light) {
    LightParams::Beam& beam = light.beam;
    std::string trigger;
    if (light.type == LightType::Laser) {
        const LaserLight laser(scene, path);
        beam.pulseDecay = laser.GetPulseDecay();
        beam.pulseJitter = laser.GetPulseJitter();
        trigger = laser.GetPulseTrigger();
    } else {
        const Light lamp(scene, path);
        beam.pulseRate = lamp.GetPulseRate();
        beam.pulseWidth = lamp.GetPulseWidth();
        beam.pulseDelay = lamp.GetPulseDelay();
        beam.pulseDecay = lamp.GetPulseDecay();
        beam.pulseJitter = lamp.GetPulseJitter();
        beam.pulsePeak = true;
        trigger = lamp.GetPulseTrigger();
    }
    beam.pulseOnExposure = trigger == tokens::kExposureTrigger;
    const bool fits = beam.pulseOnExposure ||
                      beam.pulseWidth * beam.pulseRate <= 1.0 + 1e-9;
    if ((trigger != tokens::kClock && !beam.pulseOnExposure) ||
        !(beam.pulseRate >= 0.0) || !std::isfinite(beam.pulseRate) ||
        (beam.Pulsed() && (!(beam.pulseWidth > 0.0) || !fits)) ||
        !std::isfinite(beam.pulseDelay) ||
        !(beam.pulseDecay >= 0.0) ||
        !std::isfinite(beam.pulseDecay) ||
        !(beam.pulseJitter >= 0.0) ||
        !std::isfinite(beam.pulseJitter)) {
        throw std::invalid_argument(
            "Light '" + path.String() +
            "' needs a pulse rate, decay and jitter of 0 or more, "
            "pulses that fit their rate, and a trigger of \"clock\" "
            "or \"exposure\"");
    }
}

LightParams LaserParamsOf(const Scene& scene, const Path& path,
                          const SpectralBands& bands,
                          const glm::dmat4& world) {
    const LaserLight laser(scene, path);
    LightParams light;
    light.type = LightType::Laser;
    light.shadows = laser.GetShadowEnable();
    light.position = glm::dvec3(world[3]);
    const glm::dvec3 axis = -glm::dvec3(world[2]);
    const glm::dvec3 xAxis(world[0]);
    if (glm::length(axis) == 0.0) {
        throw std::invalid_argument("Light '" + path.String() +
                                    "' has a degenerate transform");
    }
    const glm::dvec3 d = glm::normalize(axis);
    const glm::dvec3 x = xAxis - glm::dot(xAxis, d) * d;
    if (glm::length(x) == 0.0) {
        throw std::invalid_argument("Light '" + path.String() +
                                    "' has a degenerate transform");
    }
    light.direction = glm::vec3(d);
    light.tangent = glm::vec3(glm::normalize(x));
    SetPolarization(light, laser.GetPolarizationDegree(),
                    laser.GetPolarizationAngle(), d,
                    glm::normalize(x));
    const double wavelength = laser.GetWavelength();
    const glm::dvec2 waist = laser.GetWaistRadius();
    const glm::dvec2 quality = laser.GetBeamQuality();
    const double power = laser.GetPower();
    if (!(wavelength > 0.0) || !(waist.x > 0.0) || !(waist.y > 0.0) ||
        quality.x < 1.0 || quality.y < 1.0 || power < 0.0 ||
        laser.GetLinewidth() < 0.0) {
        throw std::invalid_argument(
            "LaserLight '" + path.String() +
            "' needs a positive wavelength and waist, M^2 of 1 or "
            "more, and no negative power or linewidth");
    }
    light.beam.waist = glm::vec2(waist);
    light.beam.wavelength = static_cast<float>(wavelength);
    light.beam.linewidth = static_cast<float>(laser.GetLinewidth());
    light.beam.speckle = laser.GetSpeckle();
    light.beam.rayleigh = glm::vec2(laser.RayleighRange());
    light.beam.waistDistance =
        static_cast<float>(laser.GetWaistDistance());
    // An aperture passes the beam's core: of a Gaussian of radii w
    // there, 1 - exp(-2 a^2 / (w_x w_y)) of its power.
    double through = power;
    if (const double a = laser.GetApertureRadius(); a > 0.0) {
        const glm::dvec2 z(-laser.GetWaistDistance());
        const glm::dvec2 r = glm::dvec2(light.beam.rayleigh);
        const glm::dvec2 w =
            waist * glm::sqrt(1.0 + (z / r) * (z / r));
        through *= 1.0 - std::exp(-2.0 * a * a / (w.x * w.y));
    }
    light.spectrum =
        LaserLine(bands, wavelength, laser.GetLinewidth(), through);
    // Its pattern (`LightPatternRegistry::WithBuiltins`), its data
    // and image. Patterns spread from the aperture, so their shadow
    // view is a perspective one over the pattern's angular extent
    // (`coneAngle`), its beamlets' spread added; a spot's looks along
    // its beam.
    LightParams::Beam& beam = light.beam;
    beam.pattern = laser.GetPattern();
    beam.efficiency = static_cast<float>(laser.GetEfficiency());
    beam.zeroOrder = static_cast<float>(laser.GetZeroOrder());
    beam.patternData = laser.GetPatternData();
    beam.patternImage = PatternImageOf(laser.GetPatternImage());
    // An image written as an array replaces the file's: on the GPU
    // the renderer reads it there, on the host it is built once a
    // change.
    if (const Tensor* array =
            scene.Read(path, tokens::kLaserPatternImageArray)) {
        if (IsGpu(array->Location().kind)) {
            if (array->Type() != DType::Float32 ||
                array->Rank() != 2 || !array->IsContiguous() ||
                array->Shape()[0] < 1 || array->Shape()[1] < 1 ||
                array->Shape()[0] > 0xffff ||
                array->Shape()[1] > 0xffff) {
                throw std::invalid_argument(
                    "LaserLight '" + path.String() +
                    "' needs its pattern image array as a contiguous "
                    "float32 (rows, cols) array, at most 65535 on a "
                    "side");
            }
            beam.patternImage = nullptr;
            beam.patternArray = *array;
            beam.patternArrayStream = scene.WriteStream(
                path, tokens::kLaserPatternImageArray);
            beam.patternArrayKey = path.String();
        } else {
            beam.patternImage =
                PatternImageOfArray(scene, path, *array);
        }
    }
    const glm::dvec2 spread = 2.0 * laser.Divergence();
    double extent = 0.0;  // half-angle, radians
    if (beam.pattern == tokens::kLine) {
        const double fan = glm::radians(laser.GetFanAngle());
        const bool gaussian =
            laser.GetLineProfile() == tokens::kGaussian;
        if (!(fan > 0.0) || !(fan < glm::radians(170.0)) ||
            (!gaussian && laser.GetLineProfile() != tokens::kFlat)) {
            throw std::invalid_argument(
                "LaserLight '" + path.String() +
                "' needs a fan angle above 0 and below 170 degrees "
                "and a "
                "line profile of \"flat\" or \"gaussian\"");
        }
        beam.patternParams = {static_cast<float>(0.5 * fan),
                              gaussian ? 1.0f : 0.0f};
        extent = 0.5 * fan + spread.x;
    } else if (beam.pattern == tokens::kGrid) {
        const int columns = laser.GetGridColumns();
        const int rows = laser.GetGridRows();
        const glm::dvec2 pitch = glm::radians(laser.GetGridPitch());
        if (columns < 1 || rows < 1 || !(pitch.x > 0.0) ||
            !(pitch.y > 0.0)) {
            throw std::invalid_argument(
                "LaserLight '" + path.String() +
                "' needs one or more grid columns and rows and a "
                "positive pitch");
        }
        beam.patternParams = {
            static_cast<float>(columns), static_cast<float>(rows),
            static_cast<float>(pitch.x), static_cast<float>(pitch.y)};
        const glm::dvec2 half =
            0.5 * glm::dvec2(columns - 1, rows - 1) * pitch + spread;
        extent = std::atan(glm::length(glm::tan(half)));
    } else if (beam.pattern == tokens::kDots) {
        const glm::dvec2 fov =
            glm::radians(laser.GetPatternFieldOfView());
        const int columns = laser.GetDotColumns();
        const int rows = laser.GetDotRows();
        const int tileColumns = laser.GetDotTileColumns();
        const int tileRows = laser.GetDotTileRows();
        const double jitter = laser.GetDotJitter();
        const int seed = laser.GetDotSeed();
        if (columns < 1 || rows < 1 || tileColumns < 1 ||
            tileRows < 1 || !(jitter >= 0.0) || !(jitter <= 1.0) ||
            seed < 0 || !(fov.x > 0.0) || !(fov.y > 0.0) ||
            !(fov.x < glm::radians(170.0)) ||
            !(fov.y < glm::radians(170.0))) {
            throw std::invalid_argument(
                "LaserLight '" + path.String() +
                "' needs one or more dots and tiles along each "
                "axis, a jitter from 0 to 1, a seed of 0 or more and "
                "a field of view above 0 and below 170 degrees");
        }
        const glm::dvec2 half = glm::tan(0.5 * fov);
        beam.patternParams = {static_cast<float>(columns),
                              static_cast<float>(rows),
                              static_cast<float>(tileColumns),
                              static_cast<float>(tileRows),
                              static_cast<float>(jitter),
                              static_cast<float>(half.x),
                              static_cast<float>(half.y),
                              static_cast<float>(seed)};
        extent = std::atan(glm::length(half)) +
                 std::max(spread.x, spread.y);
    } else if (beam.pattern == tokens::kScan) {
        const glm::dvec2 fov =
            glm::radians(laser.GetPatternFieldOfView());
        const glm::dvec2 frequencies = laser.GetScanFrequencies();
        if (!(fov.x >= 0.0) || !(fov.y >= 0.0) ||
            !(fov.x < glm::radians(170.0)) ||
            !(fov.y < glm::radians(170.0)) ||
            !(frequencies.x >= 0.0) || !(frequencies.y >= 0.0)) {
            throw std::invalid_argument(
                "LaserLight '" + path.String() +
                "' needs a scan's field of view of 0 or more and "
                "below "
                "170 degrees and frequencies of 0 or more");
        }
        const glm::dvec2 half = glm::tan(0.5 * fov);
        beam.patternParams = {
            static_cast<float>(half.x), static_cast<float>(half.y),
            static_cast<float>(frequencies.x),
            static_cast<float>(frequencies.y),
            static_cast<float>(glm::radians(laser.GetScanPhase()))};
        extent = std::atan(glm::length(half)) +
                 std::max(spread.x, spread.y);
    } else if (beam.pattern == tokens::kImage) {
        const glm::dvec2 fov =
            glm::radians(laser.GetPatternFieldOfView());
        if (!(fov.x > 0.0) || !(fov.y > 0.0) ||
            !(fov.x < glm::radians(170.0)) ||
            !(fov.y < glm::radians(170.0)) ||
            (beam.patternImage == nullptr &&
             beam.patternArray.Empty())) {
            throw std::invalid_argument(
                "LaserLight '" + path.String() +
                "' needs a pattern image and a field of view above 0 "
                "and below 170 degrees");
        }
        const glm::dvec2 half = glm::tan(0.5 * fov);
        beam.patternParams = {static_cast<float>(half.x),
                              static_cast<float>(half.y)};
        beam.extent = glm::vec2(half);
        extent = std::atan(glm::length(half)) +
                 std::max(spread.x, spread.y);
    } else if (beam.pattern != tokens::kSpot) {
        // A custom pattern, which the renderer looks up (and sizes
        // the shadow view for) by name.
        beam.patternParams = laser.GetPatternParams();
    }
    if (beam.efficiency < 0.0f || beam.zeroOrder < 0.0f ||
        beam.efficiency + beam.zeroOrder > 1.0f + 1e-6f) {
        throw std::invalid_argument(
            "LaserLight '" + path.String() +
            "' sends more than its power into its pattern: "
            "efficiency "
            "and zero order add to at most 1");
    }
    // Its pulses and sequence of patterns.
    beam.pulseRate = laser.GetPulseRate();
    beam.pulseWidth = laser.GetPulseWidth();
    beam.pulseDelay = laser.GetPulseDelay();
    beam.sequenceCount = laser.GetSequenceCount();
    beam.sequencePeriod = laser.GetSequencePeriod();
    beam.sequenceDelay = laser.GetSequenceDelay();
    if (!(beam.pulseRate >= 0.0) || !std::isfinite(beam.pulseRate) ||
        (beam.pulseRate > 0.0 &&
         (!(beam.pulseWidth > 0.0) ||
          beam.pulseWidth * beam.pulseRate > 1.0 + 1e-9)) ||
        !std::isfinite(beam.pulseDelay) || beam.sequenceCount < 1 ||
        !(beam.sequencePeriod > 0.0) ||
        !std::isfinite(beam.sequencePeriod) ||
        !std::isfinite(beam.sequenceDelay)) {
        throw std::invalid_argument(
            "LaserLight '" + path.String() +
            "' needs a pulse rate of 0 or more with pulses that fit "
            "it, one or more patterns in its sequence and a positive "
            "period");
    }
    if (!beam.IsSpot()) {
        light.coneAngle = static_cast<float>(
            std::min(glm::degrees(extent) + 1.0, 180.0));
    }
    return light;
}

}  // namespace

LightParams LightParamsOf(const Scene& scene, const Path& path,
                          const SpectralBands& bands) {
    const std::string_view type = scene.TypeOf(path);
    const bool known =
        type == tokens::kDistantLight ||
        type == tokens::kSphereLight || type == tokens::kRectLight ||
        type == tokens::kDiskLight ||
        type == tokens::kCylinderLight || type == tokens::kLaserLight;
    return LightParamsOf(scene, path, bands,
                         known ? scene.WorldTransform(path)
                               : glm::dmat4(1.0));
}

LightParams LightParamsOf(const Scene& scene, const Path& path,
                          const SpectralBands& bands,
                          const glm::dmat4& world) {
    using namespace tokens;
    bands.Check();
    LightParams light;
    const std::string_view type = scene.TypeOf(path);
    if (type == kDistantLight) {
        light.type = LightType::Distant;
        const DistantLight distant(scene, path);
        light.shadows = distant.GetShadowEnable();
        light.angle =
            std::max(static_cast<float>(distant.GetAngle()), 0.0f);
        light.direction =
            glm::vec3(glm::normalize(glm::dvec3(world[2])));
        light.spectrum = SunIrradianceOf(scene, path, bands);
        // Its irradiance over sunlight's wavelengths, 250 to 4000 nm.
        const SpectralBands broad{250.0, 4000.0,
                                  SpectralBands::kMaxBands};
        double total = 0.0;
        for (const float value :
             SunIrradianceOf(scene, path, broad)) {
            total += static_cast<double>(value) * broad.Width();
        }
        light.totalIrradiance = static_cast<float>(total);
        // Its light travels along its -z.
        const glm::dvec3 travel = -glm::dvec3(light.direction);
        const glm::dvec3 xAxis(world[0]);
        const glm::dvec3 across =
            xAxis - glm::dot(xAxis, travel) * travel;
        if (glm::length(across) > 0.0) {
            SetPolarization(light, distant.GetPolarizationDegree(),
                            distant.GetPolarizationAngle(), travel,
                            glm::normalize(across));
        }
        return light;
    }
    if (type == kLaserLight) {
        return LaserParamsOf(scene, path, bands, world);
    }
    if (type != kSphereLight && type != kRectLight &&
        type != kDiskLight && type != kCylinderLight) {
        throw std::invalid_argument(
            "'" + path.String() +
            "' is not a DistantLight, SphereLight, RectLight, "
            "DiskLight or "
            "CylinderLight");
    }
    const BoundableLight emitter(scene, path);
    light.shadows = emitter.GetShadowEnable();
    light.position = glm::dvec3(world[3]);
    const glm::dvec3 axis = -glm::dvec3(world[2]);
    const glm::dvec3 xAxis(world[0]);
    const glm::dvec3 yAxis(world[1]);
    if (glm::length(axis) == 0.0 || glm::length(xAxis) == 0.0 ||
        glm::length(yAxis) == 0.0) {
        throw std::invalid_argument("Light '" + path.String() +
                                    "' has a degenerate transform");
    }
    light.direction = glm::vec3(glm::normalize(axis));
    const glm::dvec3 d = glm::normalize(axis);
    const glm::dvec3 x = xAxis - glm::dot(xAxis, d) * d;
    if (glm::length(x) == 0.0) {
        throw std::invalid_argument("Light '" + path.String() +
                                    "' has a degenerate transform");
    }
    light.tangent = glm::vec3(glm::normalize(x));
    // About its axis, the way its light mostly goes (an area light's
    // field across that axis).
    SetPolarization(light, emitter.GetPolarizationDegree(),
                    emitter.GetPolarizationAngle(), d,
                    glm::normalize(x));
    SetFocus(light, emitter, bands);
    light.profile = ProfileOf(emitter, world);
    if (type == kCylinderLight) {
        // Lambertian over its side, 2 pi r l: unshaped, a power over
        // pi times that, and shaped by its focus and profile about
        // its -z axis as the other lights are. An illuminance on that
        // axis takes its silhouette as l by 2 r, close beyond a few
        // radii.
        const CylinderLight cylinder(scene, path);
        light.type = LightType::Cylinder;
        light.size = glm::vec2(
            cylinder.GetLength() * glm::length(xAxis), 0.0f);
        light.radius = static_cast<float>(cylinder.GetRadius() *
                                          glm::length(yAxis));
        const double length = light.size.x;
        const double radius = light.radius;
        const double area = 2.0 * M_PI * radius * length;
        if (!(area > 0.0)) {
            throw std::invalid_argument("Light '" + path.String() +
                                        "' has no area");
        }
        const double side = 2.0 * radius * length;
        const EmissionShape shape =
            EmissionShapeOf(emitter, 180.0, 0.0);
        const double plain =
            side * CylinderEmittedSolidAngle(shape, {});
        const double spread = light.profile.empty()
                                  ? plain
                                  : side * CylinderEmittedSolidAngle(
                                               shape, light.profile);
        const double onAxis =
            light.profile.empty() ? 1.0 : ProfileAxis(light.profile);
        const std::optional<std::vector<float>> pairs =
            SpectrumPairs(emitter);
        if (emitter.IsPhotometric()) {
            const auto perLux = [&](double distance) {
                return RectLuminancePerLux(length, 2.0 * radius,
                                           distance) /
                       onAxis;
            };
            light.spectrum = RelativeSpectrum(
                emitter, bands,
                PhotometricAreaValue(emitter, spread, perLux));
        } else if (emitter.HasPower() || pairs) {
            light.spectrum = LampSpectrum(
                emitter, bands, pairs ? &*pairs : nullptr, spread);
        } else {
            // With a profile, its candela times the intensity over
            // its side's projection 2 r l (a luminance per candela),
            // or normalized the luminance keeping its flux.
            const double luminance = LightIntensity(emitter, area);
            light.spectrum = RelativeSpectrum(
                emitter, bands,
                light.profile.empty() ? luminance
                : emitter.GetShapingIesNormalize()
                    ? luminance * plain / spread
                    : LightIntensity(emitter) / side);
        }
        return light;
    }
    if (type != kSphereLight) {
        double area = 0.0;
        if (type == kRectLight) {
            const RectLight rect(scene, path);
            light.type = LightType::Rect;
            light.size =
                glm::vec2(rect.GetWidth() * glm::length(xAxis),
                          rect.GetHeight() * glm::length(yAxis));
            area = static_cast<double>(light.size.x) *
                   static_cast<double>(light.size.y);
        } else {
            light.type = LightType::Disk;
            light.radius = static_cast<float>(
                DiskLight(scene, path).GetRadius() *
                glm::length(xAxis));
            const double radius = light.radius;
            area = M_PI * radius * radius;
        }
        if (!(area > 0.0)) {
            throw std::invalid_argument("Light '" + path.String() +
                                        "' has no area");
        }
        // A power spreads over the area times the focused (and
        // profiled) hemisphere, pi plain.
        const EmissionShape shape =
            EmissionShapeOf(emitter, 180.0, 0.0);
        const double plain = area * EmittedSolidAngle(shape, true);
        const double spread =
            light.profile.empty()
                ? plain
                : area *
                      EmittedSolidAngle(shape, true, light.profile);
        const double onAxis =
            light.profile.empty() ? 1.0 : ProfileAxis(light.profile);
        const std::optional<std::vector<float>> pairs =
            SpectrumPairs(emitter);
        if (emitter.IsPhotometric()) {
            const glm::dvec2 size(light.size);
            const double radius = light.radius;
            const auto perLux = [&](double distance) {
                return (type == kRectLight
                            ? RectLuminancePerLux(size.x, size.y,
                                                  distance)
                            : DiskLuminancePerLux(radius, distance)) /
                       onAxis;
            };
            light.spectrum = RelativeSpectrum(
                emitter, bands,
                PhotometricAreaValue(emitter, spread, perLux));
        } else if (emitter.HasPower() || pairs) {
            light.spectrum = LampSpectrum(
                emitter, bands, pairs ? &*pairs : nullptr, spread);
        } else {
            // With a profile, its candela times the intensity over
            // the area (a luminance per candela), or normalized the
            // luminance keeping the light's flux.
            const double luminance = LightIntensity(emitter, area);
            light.spectrum = RelativeSpectrum(
                emitter, bands,
                light.profile.empty() ? luminance
                : emitter.GetShapingIesNormalize()
                    ? luminance * plain / spread
                    : LightIntensity(emitter) / area);
        }
        return light;
    }
    const SphereLight sphere(scene, path);
    light.radius = sphere.GetTreatAsPoint()
                       ? 0.0f
                       : static_cast<float>(sphere.GetRadius());
    light.coneAngle =
        std::clamp(static_cast<float>(sphere.GetShapingConeAngle()),
                   0.0f, 180.0f);
    light.coneSoftness = std::clamp(
        static_cast<float>(sphere.GetShapingConeSoftness()), 0.0f,
        1.0f);
    if (light.radius < 0.0f) {
        throw std::invalid_argument("SphereLight '" + path.String() +
                                    "' has a negative radius");
    }
    // A power, photometric or radiant, spreads over the solid angle
    // under the cone, focus and profile, so narrowing any
    // concentrates the light rather than dimming it.
    const EmissionShape shape =
        EmissionShapeOf(sphere, light.coneAngle, light.coneSoftness);
    const double plain = EmittedSolidAngle(shape, false);
    const double spread =
        light.profile.empty()
            ? plain
            : EmittedSolidAngle(shape, false, light.profile);
    if (sphere.IsPhotometric()) {
        const double onAxis =
            light.profile.empty() ? 1.0 : ProfileAxis(light.profile);
        const auto perLux = [&](double distance) {
            return distance * distance / onAxis;
        };
        light.spectrum = RelativeSpectrum(
            sphere, bands,
            PhotometricAreaValue(sphere, spread, perLux));
        return light;
    }
    const std::optional<std::vector<float>> pairs =
        SpectrumPairs(sphere);
    if (sphere.HasPower() || pairs) {
        light.spectrum = LampSpectrum(
            sphere, bands, pairs ? &*pairs : nullptr, spread);
        return light;
    }
    const double radius = light.radius;
    const double area = radius > 0.0 ? M_PI * radius * radius : 1.0;
    // Normalized, over its surface area, 4 pi r^2: a luminous
    // intensity of a quarter of it.
    const double intensity =
        LightIntensity(sphere, 4.0 * M_PI * radius * radius) * area;
    // With a profile, its candela times the intensity, or normalized
    // the luminous intensity per candela keeping the light's flux.
    light.spectrum = RelativeSpectrum(
        sphere, bands,
        light.profile.empty()             ? intensity
        : sphere.GetShapingIesNormalize() ? intensity * plain / spread
                                          : LightIntensity(sphere));
    return light;
}

std::array<glm::vec3, 9>
DomeIrradianceSh(const Tensor& latlong, const glm::dmat3& rotation) {
    if (latlong.Location().kind != DeviceKind::Cpu ||
        latlong.Type() != DType::Float32 ||
        latlong.Shape().size() != 3 || latlong.Shape()[2] < 3 ||
        latlong.Shape()[0] < 1 || latlong.Shape()[1] < 1) {
        throw std::invalid_argument(
            "A dome texture must be a float32 (rows, cols, 3 or 4) "
            "array on the CPU");
    }
    const Tensor texels = latlong.Contiguous();
    const std::int64_t rows = texels.Shape()[0];
    const std::int64_t cols = texels.Shape()[1];
    const std::int64_t channels = texels.Shape()[2];
    const float* data = texels.As<float>();
    std::array<glm::dvec3, 9> radiance{};
    const double dTheta = M_PI / static_cast<double>(rows);
    const double dPhi = 2.0 * M_PI / static_cast<double>(cols);
    for (std::int64_t r = 0; r < rows; ++r) {
        const double theta = (static_cast<double>(r) + 0.5) * dTheta;
        const double solidAngle =
            (std::cos(static_cast<double>(r) * dTheta) -
             std::cos(static_cast<double>(r + 1) * dTheta)) *
            dPhi;
        for (std::int64_t c = 0; c < cols; ++c) {
            const double phi = (static_cast<double>(c) + 0.5) * dPhi;
            const glm::dvec3 local(std::sin(theta) * std::cos(phi),
                                   std::sin(theta) * std::sin(phi),
                                   std::cos(theta));
            const float* texel = data + (r * cols + c) * channels;
            const glm::dvec3 rgb(texel[0], texel[1], texel[2]);
            const std::array<double, 9> basis =
                ShBasis(rotation * local);
            for (std::size_t k = 0; k < 9; ++k) {
                radiance[k] += rgb * basis[k] * solidAngle;
            }
        }
    }
    constexpr std::array<double, 9> kBand{
        1.0,  2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25,
        0.25, 0.25,      0.25,      0.25};
    std::array<glm::vec3, 9> sh;
    for (std::size_t k = 0; k < 9; ++k) {
        sh[k] = glm::vec3(radiance[k] * kBand[k]);
    }
    return sh;
}

DomeParams DomeParamsOf(const Scene& scene, const Path& path,
                        const SpectralBands& bands) {
    using namespace tokens;
    bands.Check();
    if (scene.TypeOf(path) != kDomeLight) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not a DomeLight");
    }
    const DomeLight prim(scene, path);
    DomeParams dome;
    const glm::dmat4 world = scene.WorldTransform(path);
    const glm::dmat3 rotation(glm::normalize(glm::dvec3(world[0])),
                              glm::normalize(glm::dvec3(world[1])),
                              glm::normalize(glm::dvec3(world[2])));
    dome.toDome = glm::mat3(glm::transpose(rotation));
    dome.visible = prim.GetVisibleInPrimaryRay();
    dome.shadow = prim.GetShadowEnable();
    if (const Tensor* texture = prim.GetTexture()) {
        dome.sh = DomeIrradianceSh(*texture, rotation);
    } else {
        dome.sh[0] =
            glm::vec3(static_cast<float>(std::sqrt(4.0 * M_PI)));
    }
    if (prim.IsPhotometric()) {
        // An upward-facing surface receives pi times the luminance
        // times the map's irradiance there (1 for a uniform dome), as
        // its spherical harmonics give it.
        const std::array<double, 9> up = ShBasis({0.0, 0.0, 1.0});
        glm::dvec3 irradiance(0.0);
        for (std::size_t k = 0; k < up.size(); ++k) {
            irradiance += glm::dvec3(dome.sh[k]) * up[k];
        }
        const double luminance =
            glm::dot(irradiance, glm::dvec3(0.2126, 0.7152, 0.0722));
        if (!(luminance > 0.0)) {
            throw std::invalid_argument(
                "DomeLight '" + path.String() +
                "' lights no upward surface to give an illuminance");
        }
        dome.radiance = RelativeSpectrum(
            prim, bands,
            prim.GetPhotometricIlluminance() / (M_PI * luminance));
        return dome;
    }
    const std::optional<std::vector<float>> pairs =
        SpectrumPairs(prim);
    dome.radiance =
        pairs ? Spectrum::FromPairs(*pairs).Bin(bands)
              : RelativeSpectrum(prim, bands, LightIntensity(prim));
    return dome;
}

std::optional<DomeSunExtraction> ExtractDomeSun(const Image& map) {
    const auto rows = static_cast<std::int64_t>(map.height);
    const auto cols = static_cast<std::int64_t>(map.width);
    if (rows < 2 || cols < 2) {
        return std::nullopt;
    }
    const double dTheta = M_PI / static_cast<double>(rows);
    const double dPhi = 2.0 * M_PI / static_cast<double>(cols);
    const auto luminance = [&](std::int64_t r, std::int64_t c) {
        const glm::vec4& t = map.At(static_cast<std::uint32_t>(c),
                                    static_cast<std::uint32_t>(r));
        return 0.2126 * static_cast<double>(t.r) +
               0.7152 * static_cast<double>(t.g) +
               0.0722 * static_cast<double>(t.b);
    };
    const auto solidAngle = [&](std::int64_t r) {
        return (std::cos(static_cast<double>(r) * dTheta) -
                std::cos(static_cast<double>(r + 1) * dTheta)) *
               dPhi;
    };
    // The peak, the median and the whole map's irradiance-like total.
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(rows * cols));
    double total = 0.0;
    std::int64_t peakRow = 0;
    std::int64_t peakCol = 0;
    double peak = -1.0;
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            const double y = luminance(r, c);
            values.push_back(y);
            total += std::max(y, 0.0) * solidAngle(r);
            if (y > peak) {
                peak = y;
                peakRow = r;
                peakCol = c;
            }
        }
    }
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(
                                          values.size() / 2),
                     values.end());
    const double median = values[values.size() / 2];
    if (!(peak > 0.0) || !(total > 0.0) ||
        peak < 20.0 * std::max(median, 1e-12)) {
        return std::nullopt;
    }
    // The region: texels above the threshold connected to the peak
    // (wrapping around in azimuth).
    const double threshold = kDomeSunThreshold * peak;
    std::vector<std::uint8_t> inside(
        static_cast<std::size_t>(rows * cols), 0);
    std::vector<std::pair<std::int64_t, std::int64_t>> stack{
        {peakRow, peakCol}};
    std::vector<std::pair<std::int64_t, std::int64_t>> region;
    inside[static_cast<std::size_t>(peakRow * cols + peakCol)] = 1;
    double area = 0.0;
    const double maxArea =
        2.0 * M_PI * (1.0 - std::cos(glm::radians(2.5)));
    while (!stack.empty()) {
        const auto [r, c] = stack.back();
        stack.pop_back();
        region.emplace_back(r, c);
        area += solidAngle(r);
        if (area > maxArea) {
            return std::nullopt;  // too wide for a sun
        }
        for (const auto [dr, dc] :
             {std::pair{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
            const std::int64_t nr = r + dr;
            if (nr < 0 || nr >= rows) {
                continue;
            }
            const std::int64_t nc = ((c + dc) % cols + cols) % cols;
            const auto index =
                static_cast<std::size_t>(nr * cols + nc);
            if (inside[index] == 0 && luminance(nr, nc) > threshold) {
                inside[index] = 1;
                stack.emplace_back(nr, nc);
            }
        }
    }
    // The sky around it: the mean of the texels bordering the region.
    glm::dvec3 border(0.0);
    double borderCount = 0.0;
    for (const auto [r, c] : region) {
        for (const auto [dr, dc] :
             {std::pair{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
            const std::int64_t nr = r + dr;
            if (nr < 0 || nr >= rows) {
                continue;
            }
            const std::int64_t nc = ((c + dc) % cols + cols) % cols;
            if (inside[static_cast<std::size_t>(nr * cols + nc)] ==
                0) {
                border += glm::dvec3(
                    map.At(static_cast<std::uint32_t>(nc),
                           static_cast<std::uint32_t>(nr)));
                borderCount += 1.0;
            }
        }
    }
    const glm::dvec3 sky =
        borderCount > 0.0 ? border / borderCount : glm::dvec3(0.0);
    DomeSunExtraction extraction;
    extraction.lighting = map;
    glm::dvec3 direction(0.0);
    double energy = 0.0;
    for (const auto [r, c] : region) {
        glm::vec4& texel =
            extraction.lighting.At(static_cast<std::uint32_t>(c),
                                   static_cast<std::uint32_t>(r));
        const glm::dvec3 above =
            glm::max(glm::dvec3(texel) - sky, glm::dvec3(0.0));
        const double omega = solidAngle(r);
        extraction.sun.irradiance += above * omega;
        const double theta = (static_cast<double>(r) + 0.5) * dTheta;
        const double phi = (static_cast<double>(c) + 0.5) * dPhi;
        const double weight =
            (0.2126 * above.r + 0.7152 * above.g + 0.0722 * above.b) *
            omega;
        direction +=
            weight * glm::dvec3(std::sin(theta) * std::cos(phi),
                                std::sin(theta) * std::sin(phi),
                                std::cos(theta));
        energy += weight;
        texel = glm::vec4(glm::vec3(sky), texel.a);
    }
    if (energy < 0.05 * total || glm::length(direction) == 0.0) {
        return std::nullopt;  // a highlight, not what lights the
                              // scene
    }
    extraction.sun.direction = glm::normalize(direction);
    // The region's area as a disc's.
    extraction.sun.angle = glm::degrees(
        2.0 * std::acos(std::max(1.0 - area / (2.0 * M_PI), -1.0)));
    return extraction;
}

std::vector<DomeDirection> DomeDirectionsOf(const Image& map,
                                            std::size_t count) {
    const auto rows = static_cast<std::int64_t>(map.height);
    const auto cols = static_cast<std::int64_t>(map.width);
    if (rows < 1 || cols < 1 || count == 0) {
        return {};
    }
    const double dTheta = M_PI / static_cast<double>(rows);
    const double dPhi = 2.0 * M_PI / static_cast<double>(cols);
    const auto solidAngle = [&](std::int64_t r) {
        return (std::cos(static_cast<double>(r) * dTheta) -
                std::cos(static_cast<double>(r + 1) * dTheta)) *
               dPhi;
    };
    const auto texel = [&](std::int64_t r, std::int64_t c) {
        return glm::dvec3(map.At(static_cast<std::uint32_t>(c),
                                 static_cast<std::uint32_t>(r)));
    };
    // The energy's summed-area table: sums[(r, c)] covers rows < r
    // and columns < c.
    std::vector<double> sums(
        static_cast<std::size_t>((rows + 1) * (cols + 1)), 0.0);
    const auto sum = [&](std::int64_t r, std::int64_t c) -> double& {
        return sums[static_cast<std::size_t>(r * (cols + 1) + c)];
    };
    for (std::int64_t r = 0; r < rows; ++r) {
        double row = 0.0;
        for (std::int64_t c = 0; c < cols; ++c) {
            const glm::dvec3 v =
                glm::max(texel(r, c), glm::dvec3(0.0));
            row += (0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b) *
                   solidAngle(r);
            sum(r + 1, c + 1) = sum(r, c + 1) + row;
        }
    }
    struct Region {
        std::int64_t r0, r1, c0, c1;
    };
    const auto energyOf = [&](const Region& region) {
        return sum(region.r1, region.c1) - sum(region.r0, region.c1) -
               sum(region.r1, region.c0) + sum(region.r0, region.c0);
    };
    std::vector<Region> regions{{0, rows, 0, cols}};
    while (regions.size() < count) {
        // The most energetic region that can still be split.
        std::size_t best = regions.size();
        double most = 0.0;
        for (std::size_t i = 0; i < regions.size(); ++i) {
            const Region& region = regions[i];
            const double energy = energyOf(region);
            if ((region.r1 - region.r0 > 1 ||
                 region.c1 - region.c0 > 1) &&
                energy > most) {
                most = energy;
                best = i;
            }
        }
        if (best == regions.size()) {
            break;
        }
        const Region region = regions[best];
        const double middle = std::sin(
            0.5 * static_cast<double>(region.r0 + region.r1) *
            dTheta);
        const double width =
            static_cast<double>(region.c1 - region.c0) * dPhi *
            middle;
        const double height =
            static_cast<double>(region.r1 - region.r0) * dTheta;
        const bool acrossColumns =
            region.c1 - region.c0 > 1 &&
            (width >= height || region.r1 - region.r0 == 1);
        Region first = region;
        Region second = region;
        if (acrossColumns) {
            std::int64_t cut = region.c0 + 1;
            while (cut < region.c1 - 1 &&
                   energyOf(Region{region.r0, region.r1, region.c0,
                                   cut}) < 0.5 * most) {
                ++cut;
            }
            first.c1 = cut;
            second.c0 = cut;
        } else {
            std::int64_t cut = region.r0 + 1;
            while (cut < region.r1 - 1 &&
                   energyOf(Region{region.r0, cut, region.c0,
                                   region.c1}) < 0.5 * most) {
                ++cut;
            }
            first.r1 = cut;
            second.r0 = cut;
        }
        regions[best] = first;
        regions.push_back(second);
    }
    std::vector<DomeDirection> directions;
    for (const Region& region : regions) {
        DomeDirection out;
        glm::dvec3 direction(0.0);
        for (std::int64_t r = region.r0; r < region.r1; ++r) {
            const double theta =
                (static_cast<double>(r) + 0.5) * dTheta;
            const double omega = solidAngle(r);
            for (std::int64_t c = region.c0; c < region.c1; ++c) {
                const glm::dvec3 v =
                    glm::max(texel(r, c), glm::dvec3(0.0));
                const double phi =
                    (static_cast<double>(c) + 0.5) * dPhi;
                out.irradiance += v * omega;
                direction +=
                    (0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b) *
                    omega *
                    glm::dvec3(std::sin(theta) * std::cos(phi),
                               std::sin(theta) * std::sin(phi),
                               std::cos(theta));
            }
        }
        if (glm::length(direction) > 0.0) {
            out.direction = glm::normalize(direction);
            directions.push_back(out);
        }
    }
    return directions;
}

std::optional<Image> DomeMapOf(const Scene& scene, const Path& path) {
    const Tensor* texture = DomeLight(scene, path).GetTexture();
    if (texture == nullptr) {
        return std::nullopt;
    }
    try {
        return ImageOfTensor(*texture, ImageEncoding::Linear);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            "'" + std::string(tokens::kLightTexture) + "' of '" +
            path.String() + "': " + error.what());
    }
}

namespace {

/// The prim's float array `attribute`, `components` per element, from
/// a tensor or a float array value.
std::optional<std::vector<float>>
FloatArray(const Scene& scene, const Path& path,
           std::string_view attribute, std::size_t components) {
    if (std::optional<std::vector<float>> values =
            FloatAttribute(scene, path, attribute, components)) {
        return values;
    }
    if (std::optional<std::vector<float>> list =
            GetFloats(scene, path, attribute)) {
        if (list->size() % components != 0) {
            throw std::invalid_argument(
                "'" + std::string(attribute) + "' of '" +
                path.String() + "' needs " +
                std::to_string(components) + " values per element");
        }
        return list;
    }
    return std::nullopt;
}

/// The rotation part of the world transform of `path`, without scale.
glm::dmat3 WorldRotation(const Scene& scene, const Path& path) {
    const glm::dmat4 world = scene.WorldTransform(path);
    return glm::dmat3(glm::normalize(glm::dvec3(world[0])),
                      glm::normalize(glm::dvec3(world[1])),
                      glm::normalize(glm::dvec3(world[2])));
}

}  // namespace

std::vector<Star> StarsOf(const Scene& scene, const Path& path) {
    using namespace tokens;
    if (scene.TypeOf(path) != kStarField) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not a StarField");
    }
    // The star arrays are tensors or float array values, read by
    // name: `StarField`'s tensor accessors read tensors only. They
    // have no fallback.
    const auto array = [&](std::string_view attribute,
                           std::size_t components) {
        return FloatArray(scene, path, attribute, components);
    };
    const std::optional<std::vector<float>> magnitudes =
        array(kStarsMagnitude, 1);
    if (!magnitudes) {
        return {};
    }
    const std::size_t count = magnitudes->size();
    const auto check =
        [&](const std::optional<std::vector<float>>& values,
            std::string_view attribute, std::size_t components) {
            if (values && values->size() != count * components) {
                throw std::invalid_argument(
                    "'" + std::string(attribute) + "' of '" +
                    path.String() + "' has " +
                    std::to_string(values->size() / components) +
                    " stars where '" + std::string(kStarsMagnitude) +
                    "' has " + std::to_string(count));
            }
        };
    const std::optional<std::vector<float>> ra =
        array(kStarsRightAscension, 1);
    const std::optional<std::vector<float>> dec =
        array(kStarsDeclination, 1);
    const std::optional<std::vector<float>> direction =
        array(kStarsDirection, 3);
    const std::optional<std::vector<float>> temperature =
        array(kStarsTemperature, 1);
    const std::optional<std::vector<float>> colorIndex =
        array(kStarsColorIndex, 1);
    check(ra, kStarsRightAscension, 1);
    check(dec, kStarsDeclination, 1);
    check(direction, kStarsDirection, 3);
    check(temperature, kStarsTemperature, 1);
    check(colorIndex, kStarsColorIndex, 1);
    if (!(ra && dec) && !direction) {
        throw std::invalid_argument(
            "StarField '" + path.String() + "' needs '" +
            std::string(kStarsRightAscension) + "' and '" +
            std::string(kStarsDeclination) + "', or '" +
            std::string(kStarsDirection) + "'");
    }
    const StarField field(scene, path);
    const double zeroPoint =
        field.GetZeroPoint() * std::exp2(field.GetExposure());
    const double limit = field.GetMagnitudeLimit();
    std::vector<Star> stars;
    stars.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double magnitude = (*magnitudes)[i];
        if (!(magnitude <= limit)) {
            continue;
        }
        Star star;
        if (ra && dec) {
            star.rightAscension = static_cast<float>(
                glm::radians(static_cast<double>((*ra)[i])));
            star.declination = static_cast<float>(
                glm::radians(static_cast<double>((*dec)[i])));
        } else {
            const glm::dvec3 d = glm::normalize(glm::dvec3(
                (*direction)[3 * i], (*direction)[3 * i + 1],
                (*direction)[3 * i + 2]));
            star.rightAscension =
                static_cast<float>(std::atan2(d.y, d.x));
            star.declination = static_cast<float>(
                std::asin(std::clamp(d.z, -1.0, 1.0)));
        }
        const double kelvin =
            temperature  ? static_cast<double>((*temperature)[i])
            : colorIndex ? ColorIndexToTemperature((*colorIndex)[i])
                         : 5772.0;
        star.temperature = static_cast<float>(kelvin);
        star.irradiance555 =
            static_cast<float>(BlackbodySpectralIrradiance(
                555.0, kelvin,
                StarIlluminance(magnitude, zeroPoint)));
        stars.push_back(star);
    }
    return stars;
}

StarFieldParams StarFieldParamsOf(const Scene& scene,
                                  const Path& path) {
    if (scene.TypeOf(path) != tokens::kStarField) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not a StarField");
    }
    StarFieldParams field;
    field.toCatalogue =
        glm::mat3(glm::transpose(WorldRotation(scene, path)));
    return field;
}

double AtmospherePresetRadius(std::string_view preset) {
    // The IUGG mean radius of the Earth and the IAU mean radius of
    // Mars (Archinal et al. 2018).
    if (preset == tokens::kEarth) {
        return 6371.0088e3;
    }
    if (preset == tokens::kMars) {
        return 3389.5e3;
    }
    return 0.0;
}

AtmosphereParams
AtmosphereParamsOf(const Scene& scene, const Path& path,
                   const Path& terrain,
                   const std::optional<glm::dvec3>& eye) {
    if (scene.TypeOf(path) != tokens::kAtmosphere) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not an Atmosphere");
    }
    const Atmosphere prim(scene, path);
    AtmosphereParams params;
    params.preset = prim.GetPreset();
    const double presetRadius = AtmospherePresetRadius(params.preset);
    if (presetRadius <= 0.0) {
        throw std::invalid_argument(
            "Atmosphere '" + path.String() + "' has no preset '" +
            params.preset + "'; use 'earth' or 'mars'");
    }
    // Anchored on the ground, a planet Terrain's air, about the
    // Terrain's centre; anchored at the centre, its own planet's,
    // wherever the Terrain is (the Earth's air seen from a Terrain
    // Moon).
    const std::string anchor = prim.GetAnchor();
    if (anchor != tokens::kAnchorGround &&
        anchor != tokens::kAnchorCentre) {
        throw std::invalid_argument("Atmosphere '" + path.String() +
                                    "' has no anchor '" + anchor +
                                    "'; use 'ground' or 'centre'");
    }
    // A planet Terrain's radii: equal ones are a sphere's radius.
    glm::dvec3 radii(0.0);
    if (!terrain.IsEmpty() && anchor == tokens::kAnchorGround) {
        const Terrain planet(scene, terrain);
        radii = planet.HasPlanetRadii()
                    ? planet.GetPlanetRadii()
                    : glm::dvec3(planet.GetPlanetRadius());
        if (!(glm::all(glm::greaterThan(radii, glm::dvec3(0.0))))) {
            radii = glm::dvec3(0.0);
        }
    }
    const bool ellipsoid = radii.x != radii.y || radii.y != radii.z;
    const double planetTerrain = ellipsoid ? 0.0 : radii.x;
    const double radius = prim.GetPlanetRadius();
    params.planetRadius = radius > 0.0          ? radius
                          : planetTerrain > 0.0 ? planetTerrain
                                                : presetRadius;
    const glm::dvec3 local =
        anchor == tokens::kAnchorCentre
            ? glm::dvec3(0.0)
            : glm::dvec3(0.0, 0.0, -params.planetRadius);
    params.planetCentre =
        planetTerrain > 0.0 || ellipsoid
            ? glm::dvec3(scene.WorldTransform(terrain)[3])
            : glm::dvec3(scene.WorldTransform(path) *
                         glm::dvec4(local, 1.0));
    // On an ellipsoid, the sphere touching it under the eye.
    if (ellipsoid && eye) {
        const glm::dvec3 foot =
            GeodeticSurfacePoint(*eye - params.planetCentre, radii);
        params.planetCentre +=
            foot - GeodeticNormal(foot, radii) * params.planetRadius;
    }
    params.groundAlbedo =
        std::clamp(prim.GetGroundAlbedo(), 0.0, 1.0);
    params.hazeOpticalDepth =
        std::max(prim.GetHazeOpticalDepth(), 0.0);
    params.ozoneColumn = std::max(prim.GetOzoneColumn(), 0.0);
    params.sodiumColumn = std::max(prim.GetSodiumColumn(), 0.0);
    params.dustOpticalDepth =
        std::max(prim.GetDustOpticalDepth(), 0.0);
    params.cloudTop = std::max(prim.GetCloudTop(), 0.0);
    params.cloudCover = std::clamp(prim.GetCloudCover(), 0.0, 1.0);
    params.polarization =
        std::clamp(prim.GetPolarizationDegree(), 0.0, 1.0);
    return params;
}

std::optional<PlanetSphere> PlanetOf(const Scene& scene,
                                     const Path& path) {
    if (scene.TypeOf(path) != tokens::kAtmosphere) {
        return std::nullopt;
    }
    AtmosphereParams params;
    try {
        params = AtmosphereParamsOf(
            scene, path, FirstVisible(scene, tokens::kTerrain));
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
    // Both presets' air reaches the model's top altitude.
    return PlanetSphere{params.planetCentre, params.planetRadius,
                        params.planetRadius +
                            AtmosphereModel{}.topAltitude};
}

Path NearestAtmosphere(const Scene& scene, const glm::dvec3& point) {
    Path nearest;
    double best = 0.0;
    double bestRadius = 0.0;
    for (const Path& path : scene.Prims(tokens::kAtmosphere)) {
        if (!scene.IsVisible(path)) {
            continue;
        }
        const std::optional<PlanetSphere> planet =
            PlanetOf(scene, path);
        if (!planet) {
            continue;
        }
        const double distance = std::max(
            glm::length(point - planet->centre) - planet->top, 0.0);
        if (nearest.IsEmpty() || distance < best ||
            (distance == best && planet->radius < bestRadius)) {
            nearest = path;
            best = distance;
            bestRadius = planet->radius;
        }
    }
    return nearest;
}

std::optional<PlanetSphere> PlanetOf(const Scene& scene) {
    const Path atmosphere = FirstVisible(scene, tokens::kAtmosphere);
    if (atmosphere.IsEmpty()) {
        return std::nullopt;
    }
    return PlanetOf(scene, atmosphere);
}

BodyParams BodyParamsOf(const Scene& scene, const Path& path) {
    if (scene.TypeOf(path) != tokens::kCelestialBody) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not a CelestialBody");
    }
    const CelestialBody prim(scene, path);
    BodyParams params;
    params.centre = glm::dvec3(scene.WorldTransform(path)[3]);
    params.radius = std::max(prim.GetRadius(), 0.0);
    params.albedo = glm::clamp(glm::vec3(prim.GetAlbedo()),
                               glm::vec3(0.0f), glm::vec3(1.0f));
    return params;
}

double OverscanOf(const CameraParams& camera) {
    // A camera rendered on cube faces resamples their own projection,
    // not one pinhole image.
    if (camera.lens.IsPinhole() || NeedsCubeFaces(camera)) {
        return 1.0;
    }
    const double f = camera.FocalLengthPixels();
    const glm::dvec2 centre = camera.PrincipalPoint();
    const double w = camera.width;
    const double h = camera.height;
    if (camera.pushbroom) {
        // Only along the detector line, its rows being time: the
        // undistorted line's ends.
        const double line =
            camera.lineOffsetMm / camera.focalLengthMm;
        double overscan = 1.0;
        constexpr int kSamples = 256;
        for (int i = 0; i <= kSamples; ++i) {
            const double x = (w * i / kSamples - centre.x) / f;
            const double u = camera.lens.Undistort({x, line}).x;
            overscan =
                std::max(overscan,
                         u > 0.0 ? u * f / (w - centre.x)
                                 : -u * f / std::max(centre.x, 1e-9));
        }
        return overscan * 1.01;
    }
    double overscan = 1.0;
    const auto include = [&](double x, double y) {
        // A lens of rays: their pinhole points (`CameraRay`).
        glm::dvec2 u;
        if (camera.lens.rays != nullptr) {
            const glm::dvec3 ray = CameraRay(camera, {x, y});
            u = glm::dvec2(ray) / std::max(-ray.z, 1e-6);
        } else {
            u = camera.lens.Undistort(
                {(x - centre.x) / f, (centre.y - y) / f});
        }
        const double right = w - centre.x;
        const double up = centre.y;
        const double down = h - centre.y;
        if (u.x > 0.0) {
            overscan = std::max(overscan, u.x * f / right);
        } else if (u.x < 0.0) {
            overscan = std::max(overscan, -u.x * f / centre.x);
        }
        if (u.y > 0.0) {
            overscan = std::max(overscan, u.y * f / up);
        } else if (u.y < 0.0) {
            overscan = std::max(overscan, -u.y * f / down);
        }
    };
    constexpr int kSamples = 64;
    for (int i = 0; i <= kSamples; ++i) {
        const double t = static_cast<double>(i) / kSamples;
        include(t * w, 0.0);
        include(t * w, h);
        include(0.0, t * h);
        include(w, t * h);
    }
    return overscan * 1.01;
}

FrameParams FrameParamsOf(const Scene& scene,
                          const RenderPrims& prims,
                          std::uint32_t width, std::uint32_t height) {
    using namespace tokens;
    if (prims.camera.IsEmpty()) {
        throw std::invalid_argument("The scene has no Camera");
    }
    FrameParams params;
    params.camera =
        prims.cameraTransform
            ? CameraParamsOf(scene, prims.camera, width, height,
                             *prims.cameraTransform)
            : CameraParamsOf(scene, prims.camera, width, height);
    params.overscan = OverscanOf(params.camera);
    const Path& settings = prims.settings;
    // Without a RenderSettings prim, every option is its declared
    // fallback.
    const RenderSettings options =
        RenderSettings::OrFallbacks(scene, settings);
    params.ambient =
        std::max(static_cast<float>(options.GetAmbient()), 0.0f);
    params.ambientColor = glm::max(
        glm::vec3(options.GetAmbientColor()), glm::vec3(0.0f));
    const std::string domeMode = options.GetDomeLightMode();
    if (domeMode == "approximate") {
        params.domeMode = DomeLightMode::Approximate;
    } else if (domeMode == "full") {
        params.domeMode = DomeLightMode::Full;
    } else if (domeMode == "limited") {
        params.domeMode = DomeLightMode::Limited;
    } else if (domeMode == "directions") {
        params.domeMode = DomeLightMode::Directions;
    } else {
        throw std::invalid_argument(
            "'" + settings.String() + "' " +
            std::string(kDomeLightMode) + " is '" + domeMode +
            "', not approximate, full, directions or limited");
    }
    params.domeShadowRadius = std::max(
        static_cast<float>(options.GetDomeShadowRadius()), 1e-6f);
    params.domeDirections = static_cast<std::uint32_t>(
        std::clamp(options.GetDomeDirections(), 1, 1024));
    params.wireframe = WireframeOf(scene, settings);
    params.textureLoadingDebug = options.GetTextureLoadingDebug();
    params.overlays = options.GetOverlays();
    params.minimal = MinimalModeOf(scene, settings, MinimalMode::Off);
    params.minimalColor =
        glm::vec3(options.GetMinimalConstantColor());
    const std::string precision = options.GetSpectralPrecision();
    if (precision != tokens::kFloat32 &&
        precision != tokens::kFloat16) {
        throw std::invalid_argument(
            "'" + std::string(tokens::kSpectralPrecision) + "' of '" +
            settings.String() + "' is '" + precision +
            "', not float32 or float16");
    }
    params.halfRadiance = precision == tokens::kFloat16;
    params.exposure = static_cast<float>(
        std::exp2(options.GetExposureCompensation()));
    params.meter = options.GetMeter();
    params.culling = CullingOf(scene, settings);
    params.occlusion = OcclusionOf(scene, settings);
    params.meshLodError =
        std::max(static_cast<float>(options.GetLodMeshError()), 0.0f);
    params.clearColor = glm::vec3(options.GetClearColor());
    VolumeSettings& volumeSettings = params.volumeSettings;
    volumeSettings.stepsPerVoxel =
        std::clamp(options.GetVolumeStepsPerVoxel(), 0.1, 8.0);
    const auto count = [](int value, int low, int high) {
        return static_cast<std::uint32_t>(
            std::clamp(value, low, high));
    };
    volumeSettings.maxSteps =
        count(options.GetVolumeMaxSteps(), 1, 4096);
    volumeSettings.shadowSteps =
        count(options.GetVolumeShadowSteps(), 1, 64);
    volumeSettings.skyLightSteps =
        count(options.GetVolumeSkyLightSteps(), 0, 64);
    volumeSettings.airSteps =
        count(options.GetVolumeAirSteps(), 0, 128);
    volumeSettings.maxOctaves =
        count(options.GetVolumeMaxOctaves(), 1, 8);
    volumeSettings.shadowResolution =
        count(options.GetVolumeShadowResolution(), 16, 2048);
    volumeSettings.lightCache = options.GetVolumeLightCache();
    volumeSettings.temporal = options.GetVolumeTemporal();
    volumeSettings.temporalAnimated =
        options.GetVolumeTemporalAnimated();
    volumeSettings.resolutionScale =
        options.GetVolumeResolutionScale() < 0.75 ? 0.5 : 1.0;
    volumeSettings.collectors =
        options.GetVolumeMultipleScattering() == tokens::kCollectors;
    volumeSettings.temporalBlock =
        options.GetVolumeTemporalBlock() >= 3 ? 4u : 2u;
    volumeSettings.stepsPerPixel =
        std::max(options.GetVolumeStepsPerPixel(), 0.0);
    volumeSettings.thinOpticalDepth =
        std::max(options.GetVolumeThinOpticalDepth(), 0.0);
    ShadowSettings& shadows = params.shadows;
    shadows.cascades = options.GetShadowCascades();
    shadows.resolution =
        static_cast<std::uint32_t>(options.GetShadowResolution());
    shadows.worldResolution = static_cast<std::uint32_t>(
        options.GetShadowWorldResolution());
    const std::string worlds = options.GetShadowWorlds();
    if (worlds == kSeparate) {
        shadows.worlds = WorldShadows::Separate;
    } else if (worlds == kShared) {
        shadows.worlds = WorldShadows::Shared;
    } else if (worlds == kMerged) {
        shadows.worlds = WorldShadows::Merged;
    } else {
        throw std::invalid_argument(
            "'" + std::string(kShadowWorlds) + "' of '" +
            settings.String() +
            "' must be \"separate\", \"shared\" or \"merged\"");
    }
    const std::string fit = options.GetShadowWorldFit();
    if (fit != kScene && fit != kWorldFit) {
        throw std::invalid_argument(
            "'" + std::string(kShadowWorldFit) + "' of '" +
            settings.String() + "' must be \"scene\" or \"world\"");
    }
    shadows.worldFit = fit == kWorldFit ? WorldShadowFit::World
                                        : WorldShadowFit::Scene;
    shadows.worldMargin = std::max(
        static_cast<float>(options.GetShadowWorldMargin()), 0.0f);
    shadows.softness =
        static_cast<float>(options.GetShadowSoftness());
    shadows.normalOffset =
        static_cast<float>(options.GetShadowNormalOffset());
    shadows.splitExponent = std::max(
        static_cast<float>(options.GetShadowSplitExponent()), 1.0f);
    shadows.showCascades = options.GetShadowShowCascades();
    if (!prims.dome.IsEmpty()) {
        params.dome =
            DomeParamsOf(scene, prims.dome, params.camera.bands);
    }
    ThermalSettings& thermal = params.thermal;
    thermal.enable = options.GetThermalEnable();
    thermal.defaultTemperature = static_cast<float>(
        std::max(options.GetThermalDefaultTemperature(), 0.0));
    thermal.backgroundTemperature = static_cast<float>(
        std::max(options.GetThermalBackgroundTemperature(), 0.0));
    // The background, a black body all around: radiance added to the
    // dome's, or a uniform dome of its own (sh of a map of 1).
    if (thermal.enable && thermal.backgroundTemperature > 0.0f) {
        if (!params.dome) {
            params.dome.emplace();
            params.dome->radiance.assign(
                static_cast<std::size_t>(params.camera.bands.count),
                0.0f);
            params.dome->sh[0] =
                glm::vec3(static_cast<float>(std::sqrt(4.0 * M_PI)));
        }
        const std::vector<float> glow = ThermalEmission(
            params.camera.bands, thermal.backgroundTemperature,
            [](double) { return 1.0; });
        for (std::size_t b = 0; b < glow.size(); ++b) {
            params.dome->radiance[b] += glow[b];
        }
    }
    if (!prims.stars.IsEmpty()) {
        params.stars = StarFieldParamsOf(scene, prims.stars);
    }
    // Of several Atmospheres, the one nearest the camera.
    if (!prims.atmosphere.IsEmpty()) {
        const Path nearest = NearestAtmosphere(
            scene, glm::dvec3(params.camera.transform[3]));
        params.atmosphere = AtmosphereParamsOf(
            scene, nearest.IsEmpty() ? prims.atmosphere : nearest,
            prims.terrain, glm::dvec3(params.camera.transform[3]));
    }
    for (const Path& body : prims.bodies) {
        params.bodies.push_back(BodyParamsOf(scene, body));
    }
    params.volumes = VolumesOf(scene);
    for (const Path& light : prims.lights) {
        const auto posed = prims.lightTransforms.find(light);
        params.lights.push_back(
            posed != prims.lightTransforms.end()
                ? LightParamsOf(scene, light, params.camera.bands,
                                posed->second)
                : LightParamsOf(scene, light, params.camera.bands));
        params.lights.back().world = WorldOf(scene, light);
        SetPulses(scene, light, params.lights.back());
    }
    return params;
}

namespace {

/// The shader type `info:id` of the Shader at `path`, or "" when
/// there is none.
std::string ShaderIdOf(const Scene& scene, const Path& path) {
    if (!scene.Has(path) || scene.TypeOf(path) != tokens::kShader) {
        return {};
    }
    return Shader(scene, path).GetInfoId();
}

TextureWrap WrapOf(const std::string& wrap, const Path& texture,
                   std::string_view attribute) {
    if (wrap == "repeat" || wrap == "useMetadata") {
        return TextureWrap::Repeat;
    }
    if (wrap == "mirror") {
        return TextureWrap::Mirror;
    }
    if (wrap == "clamp") {
        return TextureWrap::Clamp;
    }
    if (wrap == "black") {
        return TextureWrap::Black;
    }
    throw std::invalid_argument(
        "'" + texture.String() + "' " + std::string(attribute) +
        " is '" + wrap +
        "', not repeat, mirror, clamp, black or useMetadata");
}

glm::vec4 Vec4Of(const Scene& scene, const Path& path,
                 std::string_view attribute, glm::vec4 fallback) {
    const Value* value = scene.Get(path, attribute);
    if (value == nullptr) {
        return fallback;
    }
    if (const auto* v = value->Get<glm::vec4>()) {
        return *v;
    }
    if (const auto* v = value->Get<glm::dvec4>()) {
        return glm::vec4(*v);
    }
    throw std::invalid_argument("'" + path.String() + "' " +
                                std::string(attribute) +
                                " must be a 4-vector");
}

/// The UsdUVTexture that input `input` of the UsdPreviewSurface at
/// `surface` connects to, with the UsdTransform2d and
/// UsdPrimvarReader_float2 feeding its coordinates. A `scalar` input
/// reads one channel (outputs:r, g, b or a), any other outputs:rgb.
TextureReader TextureReaderOf(const Scene& scene,
                              const Connection& connection,
                              std::string_view input,
                              const Path& surface,
                              bool scalar = false) {
    using namespace tokens;
    const Path& texture = connection.prim;
    const std::string where =
        "'" + surface.String() + "' " + std::string(input);
    if (ShaderIdOf(scene, texture) != kUsdUVTexture) {
        throw std::invalid_argument(where + " connects to '" +
                                    texture.String() +
                                    "', which is not a UsdUVTexture");
    }
    const Shader shader(scene, texture);
    TextureReader reader;
    if (scalar) {
        constexpr std::array<std::string_view, 4> kChannels{
            "outputs:r", "outputs:g", "outputs:b", "outputs:a"};
        const auto it = std::find(kChannels.begin(), kChannels.end(),
                                  connection.output);
        if (it == kChannels.end()) {
            throw std::invalid_argument(
                where + " reads " + connection.output +
                "; a scalar input reads outputs:r, g, b or a");
        }
        reader.channel = static_cast<int>(it - kChannels.begin());
    } else if (connection.output != "outputs:rgb") {
        throw std::invalid_argument(
            where + " reads " + connection.output +
            "; a colour or normal reads outputs:rgb");
    }
    if (shader.GetTexture() != nullptr) {
        reader.image = texture.String();
    } else {
        reader.file = shader.GetFile();
    }
    if (reader.Source().empty()) {
        throw std::invalid_argument("'" + texture.String() +
                                    "' has neither " +
                                    std::string(kFile) + " nor " +
                                    std::string(kShaderTexture));
    }
    reader.wrapS = WrapOf(shader.GetWrapS(), texture, kWrapS);
    reader.wrapT = WrapOf(shader.GetWrapT(), texture, kWrapT);
    const std::string space = shader.GetSourceColorSpace();
    if (space == "auto") {
        reader.colorSpace = TextureColorSpace::Auto;
    } else if (space == "raw") {
        reader.colorSpace = TextureColorSpace::Raw;
    } else if (space == "sRGB") {
        reader.colorSpace = TextureColorSpace::Srgb;
    } else {
        throw std::invalid_argument("'" + texture.String() + "' " +
                                    std::string(kSourceColorSpace) +
                                    " is '" + space +
                                    "', not auto, raw or sRGB");
    }
    const std::string virtualTexture = shader.GetVirtualTexture();
    if (virtualTexture == "auto") {
        reader.virtualTexture = TextureVirtual::Auto;
    } else if (virtualTexture == "on") {
        reader.virtualTexture = TextureVirtual::On;
    } else if (virtualTexture == "off") {
        reader.virtualTexture = TextureVirtual::Off;
    } else {
        throw std::invalid_argument("'" + texture.String() + "' " +
                                    std::string(kVirtualTexture) +
                                    " is '" + virtualTexture +
                                    "', not auto, on or off");
    }
    // UsdUVTexture's 4-vector scale, undeclared as it shares its name
    // with UsdTransform2d's.
    reader.scale = Vec4Of(scene, texture, kShaderScale, reader.scale);
    reader.bias = glm::vec4(shader.GetBias());
    if (const std::optional<Connection> st =
            GetConnection(scene, texture, kSt)) {
        Path source = st->prim;
        if (ShaderIdOf(scene, source) == kUsdTransform2d) {
            // UsdTransform2d's 2-vector inputs:scale shares its name
            // with UsdUVTexture's 4-vector one, so the Shader type
            // does not declare it.
            const Shader transform(scene, source);
            reader.uvScale = glm::vec2(GetVec2(
                scene, source, kShaderScale, glm::dvec2(1.0)));
            reader.uvRotation =
                static_cast<float>(transform.GetRotation());
            reader.uvTranslation =
                glm::vec2(transform.GetTranslation());
            const std::optional<Connection> in =
                GetConnection(scene, source, kIn);
            if (!in) {
                throw std::invalid_argument(
                    "'" + source.String() + "' has no " +
                    std::string(kIn) + " connection");
            }
            source = in->prim;
        }
        if (ShaderIdOf(scene, source) != kUsdPrimvarReaderFloat2) {
            throw std::invalid_argument(
                "'" + texture.String() + "' " + std::string(kSt) +
                " comes from '" + source.String() +
                "', not a UsdPrimvarReader_float2");
        }
        reader.uvSet = Shader(scene, source).GetVarname();
    }
    return reader;
}

/// The UsdPreviewSurface the Material at `material` connects its
/// surface to, or the empty path.
/// @throws std::invalid_argument if it connects to something else.
Path PreviewSurfaceOf(const Scene& scene, const Path& material) {
    using namespace tokens;
    const std::optional<Connection> surface =
        GetConnection(scene, material, kSurfaceOutput);
    if (!surface) {
        return {};
    }
    if (ShaderIdOf(scene, surface->prim) != kUsdPreviewSurface) {
        throw std::invalid_argument(
            "'" + material.String() + "' " +
            std::string(kSurfaceOutput) + " connects to '" +
            surface->prim.String() +
            "', which is not a UsdPreviewSurface");
    }
    return surface->prim;
}

/// The texture coordinate set meshes bound to `material` read: its
/// textures' (the first's), else "st".
std::string UvSetOf(const Scene& scene, const Path& material) {
    if (material.IsEmpty()) {
        return "st";
    }
    try {
        const Path surface = PreviewSurfaceOf(scene, material);
        if (surface.IsEmpty()) {
            return "st";
        }
        for (const std::string_view input :
             {tokens::kDiffuseColor, tokens::kNormalInput,
              tokens::kRoughness, tokens::kMetallic,
              tokens::kOpacity}) {
            if (const std::optional<Connection> texture =
                    GetConnection(scene, surface, input)) {
                const bool scalar = input != tokens::kDiffuseColor &&
                                    input != tokens::kNormalInput;
                return TextureReaderOf(scene, *texture, input,
                                       surface, scalar)
                    .uvSet;
            }
        }
    } catch (const std::invalid_argument&) {
        // The material's own sync reports what is wrong with it.
    }
    return "st";
}

/// Texture coordinates `primvars:<name>` of the Mesh at `mesh` per
/// corner of `data`'s triangles (`MeshData::uvs`), or empty without
/// them.
std::vector<float>
TexCoordsOf(const Scene& scene, const Path& mesh,
            const std::string& name, const MeshData& data,
            std::span<const std::int64_t> faceCounts,
            std::size_t faceVertexCount) {
    using namespace tokens;
    const std::string attribute = std::string(kPrimvarsPrefix) + name;
    const std::optional<std::vector<float>> values =
        FloatAttribute(scene, mesh, attribute, 2);
    if (!values || data.indices.empty()) {
        return {};
    }
    const std::size_t valueCount = values->size() / 2;
    std::vector<std::int64_t> lookup;
    const std::string indicesName =
        attribute + std::string(kIndicesSuffix);
    const Tensor* indices = scene.Read(mesh, indicesName);
    if (indices != nullptr) {
        lookup = IntsOf(*indices, indicesName);
        for (const std::int64_t index : lookup) {
            if (index < 0 ||
                static_cast<std::size_t>(index) >= valueCount) {
                throw std::invalid_argument(
                    indicesName + " has an index out of range");
            }
        }
    }
    const std::size_t elements =
        indices != nullptr ? lookup.size() : valueCount;
    const std::size_t points = data.points.size() / 3;
    std::string interpolation =
        GetToken(scene, mesh,
                 attribute + std::string(kInterpolationSuffix), "");
    if (interpolation.empty()) {
        interpolation = elements == faceVertexCount ? "faceVarying"
                        : elements == points        ? "vertex"
                        : elements == faceCounts.size() ? "uniform"
                        : elements == 1                 ? "constant"
                                                        : "";
    }
    const std::size_t expected =
        interpolation == "faceVarying" ? faceVertexCount
        : interpolation == "vertex"    ? points
        : interpolation == "uniform"   ? faceCounts.size()
        : interpolation == "constant"  ? 1
                                       : 0;
    if (expected == 0 || elements != expected) {
        throw std::invalid_argument(
            attribute + " of '" + mesh.String() + "' has " +
            std::to_string(elements) +
            " values, one per neither point, face vertex, face nor "
            "mesh" +
            (interpolation.empty() ? ""
                                   : " (" + interpolation + ")"));
    }
    const TriangleCorners corners = TriangulatedCorners(faceCounts);
    std::vector<float> uvs(2 * data.indices.size());
    for (std::size_t c = 0; c < data.indices.size(); ++c) {
        const std::size_t element =
            interpolation == "faceVarying" ? corners.faceVertices[c]
            : interpolation == "vertex"    ? data.indices[c]
            : interpolation == "uniform"   ? corners.faces[c]
                                           : 0;
        const auto value = static_cast<std::size_t>(
            indices != nullptr ? lookup[element]
                               : static_cast<std::int64_t>(element));
        uvs[2 * c] = (*values)[2 * value];
        uvs[2 * c + 1] = (*values)[2 * value + 1];
    }
    return uvs;
}

}  // namespace

MaterialSettings MaterialSettingsOf(const Scene& scene,
                                    const Path& material,
                                    const BrdfRegistry& brdfs) {
    using namespace tokens;
    MaterialSettings settings;
    // No material (an unbound prim's empty path) shades with the
    // defaults.
    if (material.IsEmpty() || !scene.Has(material)) {
        return settings;
    }
    const Material view(scene, material);
    // A UsdPreviewSurface network shades with the preview surface's
    // model, reading its inputs from the shader, unless the Material
    // names another model.
    const Path surface = PreviewSurfaceOf(scene, material);
    settings.brdf = surface.IsEmpty() || view.HasBrdf()
                        ? view.GetBrdf()
                        : "preview_surface";
    const BrdfModel* model = brdfs.Find(settings.brdf);
    if (model == nullptr) {
        throw std::invalid_argument("Material '" + material.String() +
                                    "' names unknown BRDF '" +
                                    settings.brdf + "'");
    }
    const bool preview =
        !surface.IsEmpty() && settings.brdf == "preview_surface";
    const Path& inputs = preview ? surface : material;
    for (const BrdfParam& param : model->params) {
        settings.brdfParams.push_back(Number(scene, inputs,
                                             "inputs:" + param.name,
                                             param.defaultValue));
    }
    if (preview) {
        const Shader shader(scene, surface);
        const auto index = [&](std::string_view name) {
            for (std::size_t i = 0; i < model->params.size(); ++i) {
                if (model->params[i].name == name) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        };
        // useSpecularWorkflow is USD's int, or a bool; specularColor
        // is one colour input over three parameters.
        settings.brdfParams[static_cast<std::size_t>(
            index("useSpecularWorkflow"))] =
            shader.GetUseSpecularWorkflow() ? 1.0f : 0.0f;
        const glm::dvec3 specular = shader.GetSpecularColor();
        for (int c = 0; c < 3; ++c) {
            settings.brdfParams[static_cast<std::size_t>(
                index("specularColorR") + c)] =
                static_cast<float>(specular[c]);
        }
        if (const std::optional<Connection> roughness =
                GetConnection(scene, surface, kRoughness)) {
            settings.roughnessTexture = TextureReaderOf(
                scene, *roughness, kRoughness, surface, true);
            settings.brdfParams[static_cast<std::size_t>(
                index("roughness"))] = 0.0f;
        }
        if (const std::optional<Connection> metallic =
                GetConnection(scene, surface, kMetallic)) {
            settings.metallicTexture = TextureReaderOf(
                scene, *metallic, kMetallic, surface, true);
            settings.metallicParam = index("metallic");
        }
    }
    settings.albedo = glm::vec3(view.GetAlbedo());
    settings.useAlbedoMap = view.GetAlbedoMapEnable();
    DetailSettings& detail = settings.detail;
    detail.tileSize = static_cast<float>(view.GetDetailTileSize());
    detail.albedoStrength =
        static_cast<float>(view.GetDetailAlbedoStrength());
    detail.normalStrength =
        static_cast<float>(view.GetDetailNormalStrength());
    detail.directXNormals = view.GetDetailDirectxNormals();
    detail.hexTiling = view.GetDetailHexTiling();
    detail.hexRotation =
        static_cast<float>(view.GetDetailHexRotation());
    detail.hexContrast =
        static_cast<float>(view.GetDetailHexContrast());
    detail.hexSharpness =
        static_cast<float>(view.GetDetailHexSharpness());

    // The network's diffuse colour (a constant or a texture) is the
    // albedo, and its normal a normal map, whichever model shades it.
    if (!surface.IsEmpty()) {
        const Shader shader(scene, surface);
        if (const std::optional<Connection> diffuse =
                GetConnection(scene, surface, kDiffuseColor)) {
            settings.albedoTexture = TextureReaderOf(
                scene, *diffuse, kDiffuseColor, surface);
            settings.albedo = glm::vec3(1.0f);
        } else {
            settings.albedo = glm::vec3(shader.GetDiffuseColor());
        }
        if (const std::optional<Connection> normal =
                GetConnection(scene, surface, kNormalInput)) {
            settings.normalTexture = TextureReaderOf(
                scene, *normal, kNormalInput, surface);
        }
        if (const std::optional<Connection> opacity =
                GetConnection(scene, surface, kOpacity)) {
            settings.opacityTexture = TextureReaderOf(
                scene, *opacity, kOpacity, surface, true);
        }
        settings.opacity = std::clamp(
            static_cast<float>(shader.GetOpacity()), 0.0f, 1.0f);
        settings.opacityThreshold = std::clamp(
            static_cast<float>(shader.GetOpacityThreshold()), 0.0f,
            1.0f);
    }
    // A measured reflectance spectrum replaces the albedo, whichever
    // model shades it.
    const std::vector<float> reflectance = view.GetReflectance();
    if (!reflectance.empty()) {
        const auto bad = [&](const std::string& why) {
            return std::invalid_argument(
                "Material '" + material.String() + "' has '" +
                std::string(kReflectance) + "' " + why);
        };
        try {
            settings.reflectance = Spectrum::FromPairs(reflectance);
        } catch (const std::invalid_argument& error) {
            throw bad(std::string("pairs that are not a spectrum: ") +
                      error.what());
        }
        for (const double value : settings.reflectance.Values()) {
            if (!(value >= 0.0 && value <= 1.0)) {
                throw bad("values outside [0, 1]");
            }
        }
    } else if (const std::string library =
                   view.GetReflectanceLibrary();
               !library.empty()) {
        // A measured spectrum Ernie embeds, by id.
        const MeasuredReflectance* measured =
            FindReflectance(library);
        if (measured == nullptr) {
            std::string ids;
            for (const MeasuredReflectance& entry :
                 ReflectanceLibrary()) {
                ids +=
                    (ids.empty() ? "" : ", ") + std::string(entry.id);
            }
            throw std::invalid_argument(
                "Material '" + material.String() + "' names '" +
                library + "' in '" +
                std::string(tokens::kReflectanceLibrary) +
                "', which is none of Ernie's measured "
                "reflectances: " +
                ids);
        }
        settings.reflectance = measured->spectrum;
    }
    // Emission, from the preview surface or else the Material itself,
    // whatever the model.
    const Path& emitter = surface.IsEmpty() ? material : surface;
    if (GetConnection(scene, emitter, kEmissiveColor)) {
        throw std::invalid_argument("'" + emitter.String() + "' " +
                                    std::string(kEmissiveColor) +
                                    " is connected; only a constant "
                                    "emissiveColor is supported");
    }
    const glm::dvec3 emission =
        surface.IsEmpty() ? view.GetEmissiveColor()
                          : Shader(scene, surface).GetEmissiveColor();
    settings.emission =
        glm::max(glm::vec3(emission), glm::vec3(0.0f));
    // Thermal emission, whatever the model.
    const double temperature = view.GetThermalTemperature();
    if (!(temperature >= 0.0)) {
        throw std::invalid_argument(
            "Material '" + material.String() + "' has '" +
            std::string(kThermalTemperature) +
            "' below 0 K: " + std::to_string(temperature));
    }
    settings.temperature = static_cast<float>(temperature);
    if (const std::vector<float> emissivity =
            view.GetThermalEmissivity();
        !emissivity.empty()) {
        const auto bad = [&](const std::string& why) {
            return std::invalid_argument(
                "Material '" + material.String() + "' has '" +
                std::string(kThermalEmissivity) + "' " + why);
        };
        try {
            settings.emissivity = Spectrum::FromPairs(emissivity);
        } catch (const std::invalid_argument& error) {
            throw bad(std::string("pairs that are not a spectrum: ") +
                      error.what());
        }
        for (const double value : settings.emissivity.Values()) {
            if (!(value >= 0.0 && value <= 1.0)) {
                throw bad("values outside [0, 1]");
            }
        }
    }
    settings.temperatureModel = view.GetThermalModel();
    settings.temperatureModelParams = view.GetThermalModelParams();
    settings.temperatureModelData = view.GetThermalModelData();
    settings.polarizationRetention = static_cast<float>(
        std::clamp(view.GetPolarizationRetention(), 0.0, 1.0));
    settings.speckleRoughness =
        static_cast<float>(std::max(view.GetSpeckleRoughness(), 0.0));
    return settings;
}

bool IsMeshType(std::string_view type) {
    using namespace tokens;
    return type == kMesh || type == kCube || type == kSphere ||
           type == kCylinder || type == kCone || type == kCapsule ||
           type == kPlane;
}

MeshData PrimitiveMeshOf(const Scene& scene, const Path& path) {
    using namespace tokens;
    const std::string_view type = scene.TypeOf(path);
    const auto f = [](double value) {
        return static_cast<float>(value);
    };
    const auto axisOf = [&path](const AxisShape& shape) {
        const std::string name = shape.GetAxis();
        const std::optional<Axis> axis = AxisNamed(name);
        if (!axis) {
            throw std::invalid_argument("'" + path.String() +
                                        "' has unknown axis '" +
                                        name + "'");
        }
        return *axis;
    };
    if (type == kCube) {
        return BoxMesh(glm::vec3(f(Cube(scene, path).GetSize())));
    }
    if (type == kSphere) {
        const Sphere sphere(scene, path);
        const std::string name = sphere.GetTessellation();
        const std::optional<SphereTessellation> tessellation =
            SphereTessellationNamed(name);
        if (!tessellation) {
            throw std::invalid_argument(
                "'" + path.String() + "' has unknown tessellation '" +
                name + "'; use uv, icosphere, cubeSphere or healpix");
        }
        return TessellatedSphere(
            f(sphere.GetRadius()), *tessellation,
            static_cast<std::uint32_t>(
                std::max(sphere.GetTessellationLevel(), 0)));
    }
    if (type == kCylinder) {
        const Cylinder cylinder(scene, path);
        return CylinderMesh(f(cylinder.GetRadius()),
                            f(cylinder.GetHeight()),
                            axisOf(cylinder));
    }
    if (type == kCone) {
        const Cone cone(scene, path);
        return ConeMesh(f(cone.GetRadius()), f(cone.GetHeight()),
                        axisOf(cone));
    }
    if (type == kCapsule) {
        const Capsule capsule(scene, path);
        return CapsuleMesh(f(capsule.GetRadius()),
                           f(capsule.GetHeight()), axisOf(capsule));
    }
    if (type == kPlane) {
        const Plane plane(scene, path);
        return PlaneMesh(f(plane.GetWidth()), f(plane.GetLength()),
                         axisOf(plane));
    }
    throw std::invalid_argument("'" + path.String() +
                                "' is not a UsdGeom primitive");
}

MeshData MeshDataOf(const Scene& scene, const Path& mesh) {
    using namespace tokens;
    if (scene.TypeOf(mesh) != kMesh &&
        IsMeshType(scene.TypeOf(mesh))) {
        return PrimitiveMeshOf(scene, mesh);
    }
    const Mesh view(scene, mesh);
    MeshData data;
    const Tensor* pointsTensor = view.GetPoints();
    if (pointsTensor != nullptr &&
        IsGpu(pointsTensor->Location().kind)) {
        if (pointsTensor->Elements() % 3 != 0) {
            throw std::invalid_argument("points must be 3-vectors");
        }
        data.points.assign(
            static_cast<std::size_t>(pointsTensor->Elements()), 0.0f);
    } else {
        data.points = FloatAttribute(scene, mesh, kPoints, 3)
                          .value_or(std::vector<float>());
    }
    const std::size_t count = data.points.size() / 3;
    const Tensor* counts = view.GetFaceVertexCounts();
    const Tensor* indices = view.GetFaceVertexIndices();
    std::vector<std::int64_t> faceCounts;
    std::size_t faceVertexCount = 0;
    const bool gpuIndices =
        indices != nullptr && IsGpu(indices->Location().kind);
    if (gpuIndices) {
        // Triangles written on the GPU: zeros hold their place until
        // the renderer copies them in, as points. Counts, if any, are
        // all 3.
        if (indices->Elements() % 3 != 0) {
            throw std::invalid_argument(
                "faceVertexIndices on the GPU must be triangles");
        }
        if (counts != nullptr) {
            const bool triangles =
                IsGpu(counts->Location().kind)
                    ? 3 * counts->Elements() == indices->Elements()
                    : std::ranges::all_of(
                          IntsOf(*counts, kFaceVertexCounts),
                          [](std::int64_t n) { return n == 3; });
            if (!triangles) {
                throw std::invalid_argument(
                    "faceVertexIndices on the GPU must be triangles");
            }
        }
        data.indices.assign(
            static_cast<std::size_t>(indices->Elements()), 0u);
    } else if ((counts == nullptr) != (indices == nullptr)) {
        throw std::invalid_argument(
            "needs both faceVertexCounts and faceVertexIndices");
    } else if (counts != nullptr) {
        faceCounts = IntsOf(*counts, kFaceVertexCounts);
        const std::vector<std::int64_t> faceIndices =
            IntsOf(*indices, kFaceVertexIndices);
        data.indices =
            TriangulateFaces(faceCounts, faceIndices, count);
        faceVertexCount = faceIndices.size();
    }
    // Normals and colours written on the GPU are zeros here, holding
    // their place until the renderer copies them in (as points).
    const Tensor* normalsTensor = view.GetNormals();
    if (normalsTensor != nullptr &&
        IsGpu(normalsTensor->Location().kind)) {
        if (static_cast<std::size_t>(normalsTensor->Elements()) ==
            data.points.size()) {
            data.normals.assign(data.points.size(), 0.0f);
        }
    } else if (std::optional<std::vector<float>> normals =
                   FloatAttribute(scene, mesh, kNormals, 3)) {
        if (normals->size() == data.points.size()) {
            data.normals = std::move(*normals);
        }
    }
    if (const Tensor* colors = view.ReadDisplayColor()) {
        if (static_cast<std::size_t>(colors->Elements()) ==
                data.points.size() &&
            count > 1) {
            data.colors =
                IsGpu(colors->Location().kind)
                    ? std::vector<float>(data.points.size(), 0.0f)
                    : FloatsOf(*colors, 3, kDisplayColor);
        }
    }
    // Without the triangles on the host, no texture coordinates.
    if (!gpuIndices) {
        data.uvs = TexCoordsOf(
            scene, mesh, UvSetOf(scene, view.GetMaterialBinding()),
            data, faceCounts, faceVertexCount);
    }
    return data;
}

InstanceData InstanceDataOf(const Scene& scene,
                            const Path& instancer) {
    using namespace tokens;
    InstanceData data;
    data.positions = FloatAttribute(scene, instancer, kPositions, 3)
                         .value_or(std::vector<float>());
    data.orientations =
        FloatAttribute(scene, instancer, kOrientations, 4)
            .value_or(std::vector<float>());
    data.scales = FloatAttribute(scene, instancer, kScales, 3)
                      .value_or(std::vector<float>());
    data.colors = FloatAttribute(scene, instancer, kDisplayColor, 3)
                      .value_or(std::vector<float>());
    return data;
}

std::optional<Bounds3> ExtentOf(const Scene& scene,
                                const Path& prim) {
    // A (2, 3) array written in place of the six numbers, on the CPU.
    const auto fromArray = [](const Tensor& tensor)
        -> std::optional<std::vector<float>> {
        if (tensor.Location().kind != DeviceKind::Cpu ||
            tensor.Elements() != 6) {
            return std::nullopt;
        }
        return FloatsOf(tensor, 3, tokens::kExtent);
    };
    std::vector<float> values;
    if (const PrimTypeDef* type =
            scene.Has(prim) ? PrimTypeOf(scene, prim) : nullptr;
        type != nullptr && type->IsA(Boundable::Definition())) {
        const Boundable boundable(scene, prim);
        if (const Tensor* tensor = boundable.ReadExtent()) {
            std::optional<std::vector<float>> read =
                fromArray(*tensor);
            if (!read) {
                return std::nullopt;
            }
            values = std::move(*read);
        } else {
            values = boundable.GetExtent();
        }
    } else if (const Tensor* tensor =
                   scene.Read(prim, tokens::kExtent)) {
        // A prim of a type without bounds of its own, such as an
        // untyped one, by name.
        std::optional<std::vector<float>> read = fromArray(*tensor);
        if (!read) {
            return std::nullopt;
        }
        values = std::move(*read);
    } else if (const Value* value =
                   scene.Get(prim, tokens::kExtent)) {
        // A prim of a type without bounds of its own, such as an
        // untyped one.
        if (const auto* floats = value->Get<std::vector<float>>()) {
            values = *floats;
        }
    }
    if (values.size() != 6) {
        return std::nullopt;
    }
    return Bounds3(glm::vec3(values[0], values[1], values[2]),
                   glm::vec3(values[3], values[4], values[5]));
}

std::optional<Bounds3> LocalBoundsOf(const Scene& scene,
                                     const Path& prim) {
    if (std::optional<Bounds3> extent = ExtentOf(scene, prim)) {
        return extent;
    }
    if (const std::optional<PlanetSphere> planet =
            PlanetOf(scene, prim)) {
        // The planet lies in world space, whatever the prim's
        // transform.
        const glm::vec3 centre(
            glm::inverse(scene.WorldTransform(prim)) *
            glm::dvec4(planet->centre, 1.0));
        const glm::vec3 reach(static_cast<float>(planet->top));
        return Bounds3{centre - reach, centre + reach};
    }
    if (!IsMeshType(scene.TypeOf(prim))) {
        return std::nullopt;
    }
    MeshData mesh;
    try {
        if (scene.TypeOf(prim) == tokens::kMesh) {
            const Tensor* points = Mesh(scene, prim).GetPoints();
            if (points == nullptr ||
                points->Location().kind != DeviceKind::Cpu) {
                return std::nullopt;
            }
            mesh.points = FloatsOf(*points, 3, tokens::kPoints);
        } else {
            mesh = PrimitiveMeshOf(scene, prim);
        }
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
    if (mesh.points.size() < 3) {
        return std::nullopt;
    }
    Bounds3 bounds{glm::vec3(std::numeric_limits<float>::max()),
                   glm::vec3(std::numeric_limits<float>::lowest())};
    for (std::size_t i = 0; i + 2 < mesh.points.size(); i += 3) {
        const glm::vec3 p(mesh.points[i], mesh.points[i + 1],
                          mesh.points[i + 2]);
        bounds.first = glm::min(bounds.first, p);
        bounds.second = glm::max(bounds.second, p);
    }
    return bounds;
}

std::vector<std::vector<std::uint32_t>>
InstancesByPrototype(const Scene& scene, const Path& instancer,
                     std::size_t prototypes) {
    std::vector<std::vector<std::uint32_t>> lists(prototypes);
    const PointInstancer view(scene, instancer);
    const Tensor* positions = view.GetPositions();
    const std::size_t count =
        positions != nullptr
            ? static_cast<std::size_t>(positions->Elements()) / 3
            : 0;
    if (prototypes == 0) {
        return lists;
    }
    const Tensor* protoIndices = view.GetProtoIndices();
    if (protoIndices == nullptr) {
        for (std::uint32_t i = 0; i < count; ++i) {
            lists[0].push_back(i);
        }
        return lists;
    }
    const std::vector<std::int64_t> indices =
        IntsOf(*protoIndices, tokens::kProtoIndices);
    for (std::size_t i = 0; i < std::min(count, indices.size());
         ++i) {
        if (indices[i] >= 0 &&
            static_cast<std::size_t>(indices[i]) < prototypes) {
            lists[static_cast<std::size_t>(indices[i])].push_back(
                static_cast<std::uint32_t>(i));
        }
    }
    return lists;
}

Image ImageOfTensor(const Tensor& tensor, ImageEncoding encoding) {
    const bool grey = tensor.Rank() == 2;
    const std::int64_t channels =
        grey ? 1 : (tensor.Rank() == 3 ? tensor.Shape()[2] : 0);
    const bool typed = tensor.Type() == DType::Float32 ||
                       tensor.Type() == DType::UInt8;
    if (!typed || (channels != 1 && channels != 3 && channels != 4) ||
        tensor.Location().kind != DeviceKind::Cpu ||
        tensor.Shape()[0] < 1 || tensor.Shape()[1] < 1) {
        throw std::invalid_argument(
            "expected a float32 or uint8 (rows, cols[, 1, 3 or 4]) "
            "CPU tensor");
    }
    const Tensor packed = tensor.Contiguous();
    Image image(static_cast<std::uint32_t>(tensor.Shape()[1]),
                static_cast<std::uint32_t>(tensor.Shape()[0]));
    const bool bytes = tensor.Type() == DType::UInt8;
    const auto* floats = bytes ? nullptr : packed.As<float>();
    const auto* uints = bytes ? packed.As<std::uint8_t>() : nullptr;
    const auto value = [&](std::size_t index, bool colour) {
        if (!bytes) {
            return floats[index];
        }
        const float stored =
            static_cast<float>(uints[index]) / 255.0f;
        return colour && encoding == ImageEncoding::Srgb
                   ? SrgbToLinear(stored)
                   : stored;
    };
    const auto stride = static_cast<std::size_t>(channels);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        glm::vec4& pixel = image.pixels[i];
        const std::size_t base = i * stride;
        if (channels == 1) {
            pixel = glm::vec4(glm::vec3(value(base, true)), 1.0f);
        } else {
            pixel = glm::vec4(
                value(base, true), value(base + 1, true),
                value(base + 2, true),
                channels == 4 ? value(base + 3, false) : 1.0f);
        }
    }
    return image;
}

std::optional<GpuXform> GpuXformOf(const Scene& scene,
                                   const Path& path) {
    using namespace tokens;
    const auto onGpu = [&](std::string_view attribute) {
        const Tensor* tensor = scene.Read(path, attribute);
        return tensor != nullptr && IsGpu(tensor->Location().kind);
    };
    const auto fail = [&](const std::string& why) {
        return std::invalid_argument("Cannot pose '" + path.String() +
                                     "' on the GPU: " + why);
    };
    // The order, or `xformOp:transform` alone without one.
    static const std::vector<std::string> kTransformAlone{
        std::string(kTransform)};
    const Value* orderValue = scene.Get(path, kXformOpOrder);
    const auto* order =
        orderValue != nullptr
            ? orderValue->Get<std::vector<std::string>>()
            : nullptr;
    if (order == nullptr) {
        if (!onGpu(kTransform)) {
            return std::nullopt;
        }
        order = &kTransformAlone;
    }
    if (std::none_of(order->begin(), order->end(),
                     [&](const std::string& entry) {
                         return entry.starts_with("xformOp:") &&
                                onGpu(entry);
                     })) {
        return std::nullopt;
    }

    // The widths each op's rows may have.
    const auto widths = [](std::string_view op)
        -> std::pair<std::int64_t, std::int64_t> {
        if (op == kOrient) {
            return {4, 4};
        }
        if (op == kScale) {
            return {3, 1};
        }
        if (op == kRotateX || op == kRotateY || op == kRotateZ) {
            return {1, 1};
        }
        if (op == kTransform) {
            return {16, 16};
        }
        if (op == kTranslate || op == kRotateXYZ ||
            op == kRotateXZY || op == kRotateYXZ ||
            op == kRotateYZX || op == kRotateZXY ||
            op == kRotateZYX) {
            return {3, 3};
        }
        return {0, 0};
    };

    GpuXform pose;
    std::size_t n = 0;
    if (!order->empty() && order->front() == kResetXformStack) {
        pose.reset = true;
        n = 1;
    }
    std::optional<std::int64_t> rows;
    std::optional<std::int64_t> row;
    for (; n < order->size(); ++n) {
        const std::string& op = (*order)[n];
        const auto [width, alternative] = widths(op);
        if (width == 0) {
            throw fail("its xformOpOrder lists " + op +
                       ": suffixed and inverted ops are "
                       "composed on the CPU only");
        }
        const std::pair<Tensor, std::int64_t>* source =
            scene.RowOf(path, op);
        if (!onGpu(op) || source == nullptr) {
            throw fail(
                op + " must be a row of a GPU array written for many "
                     "prims at once "
                     "(Scene::Write of paths), as its other ops are");
        }
        Tensor whole = source->first;
        if (op == kTransform && whole.Rank() == 3 &&
            whole.Shape()[1] == 4 && whole.Shape()[2] == 4 &&
            whole.IsContiguous()) {
            // (n, 4, 4) matrices as (n, 16) rows, the view keeping
            // the array alive.
            whole =
                Tensor(whole.Data(), whole.Type(),
                       {whole.Shape()[0], 16}, {}, whole.Location(),
                       std::make_shared<const Tensor>(whole));
        }
        if (whole.Type() != DType::Float32 || whole.Rank() != 2 ||
            !whole.IsContiguous() ||
            (whole.Shape()[1] != width &&
             whole.Shape()[1] != alternative)) {
            throw fail(op +
                       " must be rows of a contiguous float32 (n, " +
                       std::to_string(width) +
                       (op == kTransform ? ") or (n, 4, 4)" : ")") +
                       " array");
        }
        if (rows && *rows != whole.Shape()[0]) {
            throw fail("the arrays of its ops must have the same "
                       "number of rows");
        }
        if (row && *row != source->second) {
            throw fail(
                "its ops must be the same row of their arrays");
        }
        rows = whole.Shape()[0];
        row = source->second;
        pose.ops.emplace_back(op, whole);
        pose.stream = scene.WriteStream(path, op);
    }
    pose.row = row.value_or(0);
    pose.wFirst = QuaternionsWFirst(scene, path);

    // Translate, orient and scale (3 wide), each at most once in that
    // order, are instance arrays as they are; anything else is
    // composed on the GPU.
    int stage = 0;
    for (const auto& [op, whole] : pose.ops) {
        const int opStage = op == kTranslate ? 1
                            : op == kOrient  ? 2
                            : op == kScale   ? 3
                                             : 0;
        if (opStage == 0 || opStage <= stage ||
            (opStage == 3 && whole.Shape()[1] != 3)) {
            pose.composed = true;
            break;
        }
        stage = opStage;
    }
    if (!pose.composed) {
        for (const auto& [op, whole] : pose.ops) {
            (op == kTranslate ? pose.translate
             : op == kOrient  ? pose.orient
                              : pose.scale) = whole;
        }
    }
    return pose;
}

bool QuaternionsWFirst(const Scene& scene, const Path& path) {
    const Value* value = scene.Get(path, tokens::kQuaternionOrder);
    const auto* order =
        value != nullptr ? value->Get<std::string>() : nullptr;
    if (order == nullptr || *order == tokens::kXyzw) {
        return false;
    }
    if (*order == tokens::kWxyz) {
        return true;
    }
    throw std::invalid_argument(
        "'" + std::string(tokens::kQuaternionOrder) + "' of '" +
        path.String() + "' must be \"xyzw\" or \"wxyz\"");
}

std::optional<Path> GpuXformAnchorOf(const Scene& scene,
                                     const Path& path) {
    for (Path at = path; !at.IsEmpty() && !at.IsRoot();
         at = at.Parent()) {
        if (!scene.Has(at)) {
            continue;
        }
        if (GpuXformOf(scene, at)) {
            return at;
        }
        if (ResetsXformStack(scene, at)) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

}  // namespace ernie
