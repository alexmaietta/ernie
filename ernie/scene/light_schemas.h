// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <string>
#include <vector>

/// UsdLux's lights as reflected schemas (`prim_type.h`), with Ernie's physical and Omniverse's
/// photometric inputs, following UsdLux's hierarchy: every light is a `Light`; sphere, rect and
/// disk lights are `BoundableLightBase` (area lights, with a power), distant and dome lights
/// `NonboundableLightBase` (lights at infinity, with an illuminance). Each declaration's fallback
/// is the value Ernie renders an unset attribute with.
///
/// References:
/// - Pixar, UsdLux, https://openusd.org/release/api/usd_lux_page_front.html
/// - NVIDIA, Omniverse lighting, photometric units,
///   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html

// clang-format off
#define ERNIE_PHOTOMETRIC_SWITCH                                                                   \
    " Setting any photometric value puts the light in photometric units: they alone then set its " \
    "brightness."

#define ERNIE_LIGHT_ATTRIBUTES(X)                                                                  \
    X(double, Intensity, ::ernie::tokens::kIntensity, 1.0,                                         \
      (::ernie::AttributeMeta{.group = "Light", .max = 1e6}))                                      \
    X(double, Exposure, ::ernie::tokens::kExposure, 0.0,                                           \
      (::ernie::AttributeMeta{.doc = "Stops; scales the intensity by 2^exposure.",                 \
                              .group = "Light", .min = -16.0, .max = 16.0}))                       \
    X(glm::dvec3, Color, ::ernie::tokens::kColor, glm::dvec3(1.0),                                 \
      (::ernie::AttributeMeta{.label = "Colour", .doc = "Linear RGB tint.", .group = "Light",      \
                              .color = true}))                                                     \
    X(bool, EnableColorTemperature, ::ernie::tokens::kEnableColorTemperature, false,               \
      (::ernie::AttributeMeta{.label = "Use temperature", .group = "Light"}))                      \
    X(double, ColorTemperature, ::ernie::tokens::kColorTemperature, 6500.0,                        \
      (::ernie::AttributeMeta{.label = "Temperature", .doc = "Kelvin.", .group = "Light",          \
                              .min = 1000.0, .max = 20000.0}))                                     \
    X(bool, ShadowEnable, ::ernie::tokens::kShadowEnable, true,                                    \
      (::ernie::AttributeMeta{.label = "Shadows", .group = "Light"}))                              \
    X(std::string, PhysicalIlluminant, ::ernie::tokens::kPhysicalIlluminant, "",                   \
      (::ernie::AttributeMeta{                                                                     \
          .label = "Illuminant",                                                                   \
          .doc = "The spectrum's shape: \"white\" (flat), \"blackbody\" or \"illuminant D\" at "   \
                 "the temperature, or \"custom\" (the custom illuminant's pairs). Unset: a black " \
                 "body if the temperature is used, else flat.",                                    \
          .group = "Spectrum",                                                                     \
          .options = {::ernie::tokens::kWhite, ::ernie::tokens::kBlackbody,                        \
                      ::ernie::tokens::kIlluminantD, ::ernie::tokens::kCustom}}))                  \
    X(std::vector<float>, PhysicalCustomIlluminant, ::ernie::tokens::kPhysicalCustomIlluminant,    \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.doc = "(nm, value) pairs, wavelengths increasing, of a \"custom\" " \
                                     "illuminant's shape; zero outside them.",                     \
                              .group = "Spectrum"}))                                               \
    X(std::vector<float>, Spectrum, ::ernie::tokens::kLightSpectrum, std::vector<float>{},         \
      (::ernie::AttributeMeta{.doc = "(nm, value) pairs of a physical light's spectrum: W m^-2 "   \
                                     "nm^-1 for a distant light, W sr^-1 nm^-1 for a sphere, W "   \
                                     "m^-2 sr^-1 nm^-1 for a rect, disk or dome.",                 \
                              .group = "Spectrum"}))

#define ERNIE_BOUNDABLE_LIGHT_ATTRIBUTES(X)                                                        \
    X(double, Power, ::ernie::tokens::kLightPower, 0.0,                                            \
      (::ernie::AttributeMeta{.doc = "Radiant flux in W over 360 to 830 nm, in place of the "      \
                                     "intensity when set.",                                        \
                              .group = "Shape", .max = 1e7}))                                      \
    X(double, PhotometricPower, ::ernie::tokens::kPhotometricPower, 1600.0,                        \
      (::ernie::AttributeMeta{.label = "Power",                                                    \
                              .doc = "Luminous flux in lumens." ERNIE_PHOTOMETRIC_SWITCH,          \
                              .group = "Photometric", .max = 1e6}))                                \
    X(double, PhotometricIlluminance, ::ernie::tokens::kPhotometricIlluminance, 0.0,               \
      (::ernie::AttributeMeta{.label = "Illuminance",                                              \
                              .doc = "Lux on a surface facing it at the distance below, in place " \
                                     "of the power.",                                              \
                              .group = "Photometric", .max = 200000.0}))                           \
    X(double, PhotometricIlluminanceDistance, ::ernie::tokens::kPhotometricIlluminanceDistance,    \
      0.0,                                                                                         \
      (::ernie::AttributeMeta{.label = "At distance", .doc = "Metres; 0 uses the power.",          \
                              .group = "Photometric", .max = 1000.0}))

#define ERNIE_NONBOUNDABLE_LIGHT_ATTRIBUTES(X)                                                     \
    X(double, PhotometricIlluminance, ::ernie::tokens::kPhotometricIlluminance, 10000.0,           \
      (::ernie::AttributeMeta{.label = "Illuminance",                                              \
                              .doc = "Lux: on a surface facing a distant light, on an "            \
                                     "upward-facing surface under a dome." ERNIE_PHOTOMETRIC_SWITCH, \
                              .group = "Photometric", .max = 200000.0}))

#define ERNIE_DISTANT_LIGHT_ATTRIBUTES(X)                                                          \
    X(double, Angle, ::ernie::tokens::kAngle, 0.53,                                                \
      (::ernie::AttributeMeta{.doc = "Degrees the light subtends; widens penumbrae.",              \
                              .group = "Shape", .max = 30.0}))                                     \
    X(double, Irradiance, ::ernie::tokens::kLightIrradiance, 1361.0,                               \
      (::ernie::AttributeMeta{.doc = "W/m^2 at normal incidence; the Sun's by default. Set, or "   \
                                     "with a spectrum, the light is physical.",                    \
                              .group = "Shape", .max = 1e5}))

#define ERNIE_SPHERE_LIGHT_ATTRIBUTES(X)                                                           \
    X(double, Radius, ::ernie::tokens::kRadius, 0.5,                                               \
      (::ernie::AttributeMeta{.doc = "Metres.", .group = "Shape", .max = 1000.0}))                 \
    X(bool, TreatAsPoint, ::ernie::tokens::kTreatAsPoint, false,                                   \
      (::ernie::AttributeMeta{.label = "Point", .doc = "Light as a point: radius 0.",              \
                              .group = "Shape"}))                                                  \
    X(double, ShapingConeAngle, ::ernie::tokens::kShapingConeAngle, 180.0,                         \
      (::ernie::AttributeMeta{.label = "Cone angle", .doc = "Degrees off the axis it reaches; "    \
                                                            "180 unshaped.",                       \
                              .group = "Shape", .max = 180.0}))                                    \
    X(double, ShapingConeSoftness, ::ernie::tokens::kShapingConeSoftness, 0.0,                     \
      (::ernie::AttributeMeta{.label = "Cone softness",                                            \
                              .doc = "Fraction of the cone angle over which it fades.",            \
                              .group = "Shape", .max = 1.0}))

#define ERNIE_RECT_LIGHT_ATTRIBUTES(X)                                                             \
    X(double, Width, ::ernie::tokens::kWidth, 1.0,                                                 \
      (::ernie::AttributeMeta{.doc = "Metres along x.", .group = "Shape", .max = 1000.0}))         \
    X(double, Height, ::ernie::tokens::kHeight, 1.0,                                               \
      (::ernie::AttributeMeta{.doc = "Metres along y.", .group = "Shape", .max = 1000.0}))

#define ERNIE_DISK_LIGHT_ATTRIBUTES(X)                                                             \
    X(double, Radius, ::ernie::tokens::kRadius, 0.5,                                               \
      (::ernie::AttributeMeta{.doc = "Metres.", .group = "Shape", .max = 1000.0}))

#define ERNIE_DOME_LIGHT_ATTRIBUTES(X)                                                             \
    X(std::string, TextureFile, ::ernie::tokens::kTextureFile, "",                                 \
      (::ernie::AttributeMeta{.doc = "USD's map file; not read yet: write the map as a texture.",  \
                              .group = "Shape", .asset = true}))                                   \
    X(::ernie::Tensor, Texture, ::ernie::tokens::kLightTexture, nullptr,                           \
      (::ernie::AttributeMeta{.doc = "float32 (rows, cols, 3 or 4) linear RGB scaling its "        \
                                     "radiance, latitude-longitude around the dome's +z axis.",    \
                              .group = "Shape"}))                                                  \
    X(bool, VisibleInPrimaryRay, ::ernie::tokens::kVisibleInPrimaryRay, true,                      \
      (::ernie::AttributeMeta{.doc = "Shows behind the scene where camera rays meet nothing; "     \
                                     "off, it still lights the scene.",                            \
                              .group = "Shape"}))

// Every attribute of each type, its bases' first, for its accessors.
#define ERNIE_BOUNDABLE_LIGHT_ALL(X) ERNIE_LIGHT_ATTRIBUTES(X) ERNIE_BOUNDABLE_LIGHT_ATTRIBUTES(X)
#define ERNIE_NONBOUNDABLE_LIGHT_ALL(X)                                                            \
    ERNIE_LIGHT_ATTRIBUTES(X) ERNIE_NONBOUNDABLE_LIGHT_ATTRIBUTES(X)
#define ERNIE_DISTANT_LIGHT_ALL(X) ERNIE_NONBOUNDABLE_LIGHT_ALL(X) ERNIE_DISTANT_LIGHT_ATTRIBUTES(X)
#define ERNIE_DOME_LIGHT_ALL(X) ERNIE_NONBOUNDABLE_LIGHT_ALL(X) ERNIE_DOME_LIGHT_ATTRIBUTES(X)
#define ERNIE_SPHERE_LIGHT_ALL(X) ERNIE_BOUNDABLE_LIGHT_ALL(X) ERNIE_SPHERE_LIGHT_ATTRIBUTES(X)
#define ERNIE_RECT_LIGHT_ALL(X) ERNIE_BOUNDABLE_LIGHT_ALL(X) ERNIE_RECT_LIGHT_ATTRIBUTES(X)
#define ERNIE_DISK_LIGHT_ALL(X) ERNIE_BOUNDABLE_LIGHT_ALL(X) ERNIE_DISK_LIGHT_ATTRIBUTES(X)
// clang-format on

namespace ernie::schema {

/// Any light: UsdLux's LightAPI inputs, its spectrum's shape and Ernie's physical spectrum.
class Light : public SchemaView {
    ERNIE_SCHEMA_BODY(Light, SchemaView, ERNIE_LIGHT_ATTRIBUTES)

    /// Whether the light gives its brightness in photometric units: it has a `photometric:` value.
    bool IsPhotometric() const;
};

/// An area light (sphere, rect, disk): a radiant power, or a photometric power or illuminance.
class BoundableLight : public Light {
    ERNIE_SCHEMA_BODY(BoundableLight, Light, ERNIE_BOUNDABLE_LIGHT_ALL)
};

/// A light at infinity (distant, dome), whose photometric brightness is an illuminance.
class NonboundableLight : public Light {
    ERNIE_SCHEMA_BODY(NonboundableLight, Light, ERNIE_NONBOUNDABLE_LIGHT_ALL)
};

/// A light at infinity shining along the -z axis of its transform, such as the Sun.
class DistantLight : public NonboundableLight {
    ERNIE_SCHEMA_BODY(DistantLight, NonboundableLight, ERNIE_DISTANT_LIGHT_ALL)
    ERNIE_SCHEMA_DEFINE(DistantLight)
};

/// Light from every direction, optionally from a latitude-longitude map.
class DomeLight : public NonboundableLight {
    ERNIE_SCHEMA_BODY(DomeLight, NonboundableLight, ERNIE_DOME_LIGHT_ALL)
    ERNIE_SCHEMA_DEFINE(DomeLight)
};

/// A sphere (or point) light, optionally shaped to a cone around its -z axis.
class SphereLight : public BoundableLight {
    ERNIE_SCHEMA_BODY(SphereLight, BoundableLight, ERNIE_SPHERE_LIGHT_ALL)
    ERNIE_SCHEMA_DEFINE(SphereLight)
};

/// A rectangle in its xy plane emitting along -z.
class RectLight : public BoundableLight {
    ERNIE_SCHEMA_BODY(RectLight, BoundableLight, ERNIE_RECT_LIGHT_ALL)
    ERNIE_SCHEMA_DEFINE(RectLight)
};

/// A disk in its xy plane emitting along -z.
class DiskLight : public BoundableLight {
    ERNIE_SCHEMA_BODY(DiskLight, BoundableLight, ERNIE_DISK_LIGHT_ALL)
    ERNIE_SCHEMA_DEFINE(DiskLight)
};

}  // namespace ernie::schema
