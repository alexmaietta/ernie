// SPDX-License-Identifier: BSD-3-Clause
/// @file lux.h
/// @brief UsdLux's lights and their shaping and shadow API schemas.
///
/// References:
/// - Pixar, UsdLux, https://openusd.org/release/api/usd_lux_page_front.html
#pragma once

#include "ernie/scene/schemas/geom.h"

// clang-format off
#define ERNIE_LIGHT_LENGTH .group = "Light", .lower = AtLeast(0.0), .finite = true

/// LightAPI's attributes, which every light has as a built-in API schema.
#define ERNIE_LIGHT_API_ATTRIBUTES(X)                                                              \
    X(std::string, ShaderId, "light:shaderId", "", "The light's shader, as renderers name it.",    \
      .group = "Light")                                                                            \
    X(std::string, MaterialSyncMode, "light:materialSyncMode", "noMaterialResponse",               \
      "For geometry that emits: whether its bound material's glow tints the light, is lit "        \
      "independently, or is ignored.",                                                             \
      .group = "Light",                                                                            \
      .options = {"materialGlowTintsLight", "independent", "noMaterialResponse"})                 \
    X(double, Intensity, "inputs:intensity", 1.0, "Brightness, scaled by 2^exposure.",             \
      .group = "Light", .lower = AtLeast(0.0), .finite = true)                                     \
    X(double, Exposure, "inputs:exposure", 0.0, "Stops of brightness, as photographers count them.", \
      .group = "Light", .finite = true)                                                            \
    X(double, Diffuse, "inputs:diffuse", 1.0, "Multiplier of its diffuse response.",               \
      .group = "Light", .lower = AtLeast(0.0))                                                     \
    X(double, Specular, "inputs:specular", 1.0, "Multiplier of its specular response.",            \
      .group = "Light", .lower = AtLeast(0.0))                                                     \
    X(bool, Normalize, "inputs:normalize", false,                                                  \
      "Divides its power by its area, so resizing it leaves its brightness.", .group = "Light")    \
    X(glm::dvec3, Color, "inputs:color", glm::dvec3(1.0), "Linear RGB tint.", .group = "Light",    \
      .lower = AtLeast(0.0), .finite = true, .color = true)                                        \
    X(bool, EnableColorTemperature, "inputs:enableColorTemperature", false,                        \
      "Tints it by its colour temperature too.", .group = "Light")                                 \
    X(double, ColorTemperature, "inputs:colorTemperature", 6500.0, "Kelvin, a black body's.",      \
      .group = "Light", .lower = Above(0.0), .finite = true)                                       \
    X(double, PolarizationDegree, "ernie:polarization:degree", 0.0,                                \
      "The fraction of its light linearly polarized.", .label = "Polarized",                       \
      .group = "Polarization", .lower = AtLeast(0.0), .upper = AtMost(1.0))                        \
    X(double, PolarizationAngle, "ernie:polarization:angle", 0.0,                                  \
      "Degrees from the light's +x to its electric field.", .label = "Angle",                      \
      .group = "Polarization", .finite = true)                                                     \
    X(double, PulseRate, "ernie:pulse:rate", 0.0, "Pulses per second, a strobe's; 0 continuous.",  \
      .group = "Time", .lower = AtLeast(0.0), .finite = true)                                      \
    X(double, PulseWidth, "ernie:pulse:width", 0.0, "Seconds each pulse lasts.", .group = "Time",  \
      .lower = AtLeast(0.0), .finite = true)                                                       \
    X(double, PulseDelay, "ernie:pulse:delay", 0.0,                                                \
      "Seconds on the sensor clock, or after the exposure opens, a pulse starts.",                 \
      .group = "Time", .finite = true)                                                             \
    X(double, PulseDecay, "ernie:pulse:decay", 0.0,                                                \
      "Seconds its level takes to fall by e over a pulse, as a xenon flash's; 0 flat.",            \
      .group = "Time", .lower = AtLeast(0.0), .finite = true)                                      \
    X(double, PulseJitter, "ernie:pulse:jitter", 0.0,                                              \
      "Seconds each pulse may start early or late.", .group = "Time", .lower = AtLeast(0.0),       \
      .finite = true)                                                                              \
    X(Choice<PulseTrigger>, PulseTrigger, "ernie:pulse:trigger", PulseTrigger::Clock,              \
      "The sensor clock, or the camera's exposure: one pulse a frame, its delay after it opens.",  \
      .group = "Time")

#define ERNIE_BOUNDABLE_LIGHT_BASE_ATTRIBUTES(X)                                                   \
    X(double, Power, "ernie:power", 0.0,                                                           \
      "Radiant flux in watts, scaling the illuminant's spectral shape; authored, it sets the "     \
      "brightness unless photometric values are authored.",                                        \
      .group = "Light", .lower = AtLeast(0.0), .finite = true)

#define ERNIE_DISTANT_LIGHT_ATTRIBUTES(X)                                                          \
    X(double, Angle, "inputs:angle", 0.53, "Angular diameter in degrees; the Sun's is 0.53.",      \
      .group = "Light", .lower = AtLeast(0.0), .upper = AtMost(360.0))                             \
    X(double, Irradiance, "ernie:irradiance", 1361.0,                                              \
      "W m^-2 at normal incidence (the Sun's at 1 au), scaling the illuminant's spectral shape; "  \
      "authored, it sets the brightness unless photometric values are authored.",                 \
      .group = "Light", .lower = AtLeast(0.0), .finite = true)

#define ERNIE_DISK_LIGHT_ATTRIBUTES(X)                                                             \
    X(double, Radius, "inputs:radius", 0.5, "Metres; a disk facing -z.", ERNIE_LIGHT_LENGTH)

#define ERNIE_RECT_LIGHT_ATTRIBUTES(X)                                                             \
    X(double, Width, "inputs:width", 1.0, "Metres along x; a rectangle facing -z.",                \
      ERNIE_LIGHT_LENGTH)                                                                          \
    X(double, Height, "inputs:height", 1.0, "Metres along y.", ERNIE_LIGHT_LENGTH)                 \
    X(std::string, TextureFile, "inputs:texture:file", "", "An image it emits.", .group = "Light", \
      .asset = true)

#define ERNIE_SPHERE_LIGHT_ATTRIBUTES(X)                                                           \
    X(double, Radius, "inputs:radius", 0.5, "Metres.", ERNIE_LIGHT_LENGTH)                         \
    X(bool, TreatAsPoint, "treatAsPoint", false, "Lights as a point however large it is.",         \
      .group = "Light")

#define ERNIE_CYLINDER_LIGHT_ATTRIBUTES(X)                                                         \
    X(double, Length, "inputs:length", 1.0, "Metres along x.", ERNIE_LIGHT_LENGTH)                 \
    X(double, Radius, "inputs:radius", 0.5, "Metres.", ERNIE_LIGHT_LENGTH)                         \
    X(bool, TreatAsLine, "treatAsLine", false, "Lights as a line however thick it is.",            \
      .group = "Light")

#define ERNIE_DOME_LIGHT_ATTRIBUTES(X)                                                             \
    X(std::string, TextureFile, "inputs:texture:file", "", "The environment it emits.",            \
      .group = "Light", .asset = true)                                                             \
    X(std::string, TextureFormat, "inputs:texture:format", "automatic", "The texture's mapping.",  \
      .group = "Light",                                                                            \
      .options = {"automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"})      \
    X(Tensor, Texture, "ernie:texture", Tensor(),                                                  \
      "float32 (rows, columns, 3 or 4) linear RGB in memory, latitude-longitude about +z, in "     \
      "place of the file.",                                                                        \
      .group = "Light", .ranks = {3})                                                              \
    X(std::vector<Path>, Portals, "portals", std::vector<Path>{},                                  \
      "Portal lights its light enters through.", .group = "Light")                                 \
    X(double, GuideRadius, "guideRadius", 1.0e5, "Metres to the sphere drawn as its guide.",       \
      .group = "Light", .lower = Above(0.0))

#define ERNIE_PORTAL_LIGHT_ATTRIBUTES(X)                                                           \
    X(double, Width, "inputs:width", 1.0, "Metres along x.", ERNIE_LIGHT_LENGTH)                   \
    X(double, Height, "inputs:height", 1.0, "Metres along y.", ERNIE_LIGHT_LENGTH)

#define ERNIE_SHAPING_API_ATTRIBUTES(X)                                                            \
    X(double, ShapingFocus, "inputs:shaping:focus", 0.0, "Concentrates its light towards -z.",     \
      .group = "Shaping", .lower = AtLeast(0.0))                                                   \
    X(glm::dvec3, ShapingFocusTint, "inputs:shaping:focusTint", glm::dvec3(0.0),                   \
      "The colour of what the focus takes away.", .group = "Shaping", .color = true)               \
    X(double, ShapingConeAngle, "inputs:shaping:cone:angle", 90.0, "Degrees from -z it lights.",   \
      .group = "Shaping", .lower = AtLeast(0.0), .upper = AtMost(180.0))                           \
    X(double, ShapingConeSoftness, "inputs:shaping:cone:softness", 0.0,                            \
      "How gradually the cone's edge falls off.", .group = "Shaping", .lower = AtLeast(0.0))       \
    X(std::string, ShapingIesFile, "inputs:shaping:ies:file", "", "An IES LM-63 profile.",         \
      .group = "Shaping", .asset = true)                                                           \
    X(double, ShapingIesAngleScale, "inputs:shaping:ies:angleScale", 0.0,                          \
      "Widens or narrows the profile's angles.", .group = "Shaping")                               \
    X(bool, ShapingIesNormalize, "inputs:shaping:ies:normalize", false,                            \
      "Keeps the profile from changing the light's power.", .group = "Shaping")

// Omniverse's photometric and illuminant schemas (omni.usd.schema.physical_lighting). A light's
// spectral shape comes from its illuminant alone; its brightness from authored photometric values,
// else authored ernie:irradiance or ernie:power, else USD's intensity and exposure.
#define ERNIE_PHOTOMETRIC_SWITCH " Authored, it alone sets the light's brightness."

#define ERNIE_PHOTOMETRIC_AREA_LIGHT_API_ATTRIBUTES(X)                                             \
    X(double, PhotometricPower, "photometric:power", 1600.0,                                       \
      "Luminous flux in lumens, whatever the colour or cone." ERNIE_PHOTOMETRIC_SWITCH,           \
      .group = "Photometric", .lower = AtLeast(0.0), .finite = true)                               \
    X(double, PhotometricIlluminance, "photometric:illuminance", 0.0,                              \
      "Lux on a surface facing it on its axis at the distance below, in place of the power when "  \
      "both are above 0.",                                                                         \
      .group = "Photometric", .lower = AtLeast(0.0), .finite = true)                               \
    X(double, PhotometricIlluminanceDistance, "photometric:illuminance:distance", 0.0,             \
      "Metres at which the illuminance holds; 0 uses the power.", .group = "Photometric",          \
      .lower = AtLeast(0.0), .finite = true)

#define ERNIE_PHOTOMETRIC_ILLUMINANCE_ATTRIBUTES(X, where)                                         \
    X(double, PhotometricIlluminance, "photometric:illuminance", 10000.0,                          \
      "Lux " where "." ERNIE_PHOTOMETRIC_SWITCH, .group = "Photometric", .lower = AtLeast(0.0),    \
      .finite = true)

#define ERNIE_PHOTOMETRIC_DISTANT_LIGHT_API_ATTRIBUTES(X)                                          \
    ERNIE_PHOTOMETRIC_ILLUMINANCE_ATTRIBUTES(X, "on a surface facing the light")

#define ERNIE_PHOTOMETRIC_DOME_LIGHT_API_ATTRIBUTES(X)                                             \
    ERNIE_PHOTOMETRIC_ILLUMINANCE_ATTRIBUTES(X, "on an upward-facing surface under the dome")

#define ERNIE_PHYSICAL_LIGHT_ILLUMINANT_API_ATTRIBUTES(X)                                          \
    X(std::string, PhysicalIlluminant, "physical:illuminant", "",                                  \
      "The spectrum's shape: white (flat) or blackbody over 380 to 780 nm, illuminant D at the "   \
      "colour temperature, custom (the pairs below), or sun (TSIS-1, 115 nm to 200 um); unset, a " \
      "black body when the temperature is used, else flat.",                                       \
      .group = "Spectrum", .options = {"white", "blackbody", "illuminant D", "custom", "sun"})      \
    X(std::vector<float>, PhysicalCustomIlluminant, "physical:customIlluminant",                   \
      std::vector<float>{},                                                                        \
      "(nm, value) pairs, wavelengths increasing, of a custom illuminant's shape; 0 outside them.", \
      .group = "Spectrum")

#define ERNIE_SHADOW_API_ATTRIBUTES(X)                                                             \
    X(bool, ShadowEnable, "inputs:shadow:enable", true, "Whether it casts shadows.",               \
      .group = "Shadow")                                                                           \
    X(glm::dvec3, ShadowColor, "inputs:shadow:color", glm::dvec3(0.0), "The colour of shadows.",  \
      .group = "Shadow", .color = true)                                                            \
    X(double, ShadowDistance, "inputs:shadow:distance", -1.0,                                      \
      "Metres shadow rays reach; -1 for no limit.", .group = "Shadow")                             \
    X(double, ShadowFalloff, "inputs:shadow:falloff", -1.0,                                        \
      "Metres over which shadows fade near their limit; -1 for none.", .group = "Shadow")          \
    X(double, ShadowFalloffGamma, "inputs:shadow:falloffGamma", 1.0, "The fade's curve.",          \
      .group = "Shadow")
// clang-format on

namespace ernie {

/// What times a pulsing light's pulses.
enum class PulseTrigger { Clock, Exposure };

template <> struct ChoiceTokens<PulseTrigger> {
    static constexpr std::array<std::string_view, 2> kTokens{"clock", "exposure"};
};

// The light bases' views read LightAPI's attributes too, which every light has built in.
#define ERNIE_BOUNDABLE_LIGHT_VIEW(X)                                                              \
    ERNIE_LIGHT_API_ATTRIBUTES(X) ERNIE_BOUNDABLE_LIGHT_BASE_ATTRIBUTES(X)

ERNIE_API_SCHEMA(LightAPI, ERNIE_LIGHT_API_ATTRIBUTES,
                 "What every light has: brightness, colour, and Ernie's polarization and pulses.");
ERNIE_SCHEMA(BoundableLightBase, Boundable, ERNIE_BOUNDABLE_LIGHT_VIEW,
             "A light with a shape, bounded by it.");
ERNIE_SCHEMA(NonboundableLightBase, Xformable, ERNIE_LIGHT_API_ATTRIBUTES,
             "A light at infinity or around everything, so unbounded.");
ERNIE_CONCRETE_SCHEMA(DistantLight, NonboundableLightBase, ERNIE_DISTANT_LIGHT_ATTRIBUTES,
                      "A light from infinitely far along +z, such as the Sun's.");
ERNIE_CONCRETE_SCHEMA(DiskLight, BoundableLightBase, ERNIE_DISK_LIGHT_ATTRIBUTES,
                      "A disk emitting towards -z.");
ERNIE_CONCRETE_SCHEMA(RectLight, BoundableLightBase, ERNIE_RECT_LIGHT_ATTRIBUTES,
                      "A rectangle emitting towards -z.");
ERNIE_CONCRETE_SCHEMA(SphereLight, BoundableLightBase, ERNIE_SPHERE_LIGHT_ATTRIBUTES,
                      "A sphere emitting outwards.");
ERNIE_CONCRETE_SCHEMA(CylinderLight, BoundableLightBase, ERNIE_CYLINDER_LIGHT_ATTRIBUTES,
                      "A cylinder along x emitting outwards.");
ERNIE_CONCRETE_SCHEMA(DomeLight, NonboundableLightBase, ERNIE_DOME_LIGHT_ATTRIBUTES,
                      "Light from all around, from an environment image.");
ERNIE_CONCRETE_SCHEMA(PortalLight, BoundableLightBase, ERNIE_PORTAL_LIGHT_ATTRIBUTES,
                      "A rectangle a dome light's light comes in through, guiding its samples.");
ERNIE_API_SCHEMA(MeshLightAPI, ERNIE_NO_ATTRIBUTES,
                 "Makes a Mesh emit, its bound material's glow tinting the light.");
ERNIE_API_SCHEMA(VolumeLightAPI, ERNIE_NO_ATTRIBUTES,
                 "Makes a Volume emit, its bound material's glow tinting the light.");
ERNIE_API_SCHEMA(ShapingAPI, ERNIE_SHAPING_API_ATTRIBUTES,
                 "Focuses a light, limits it to a cone, or shapes it by an IES profile.");
ERNIE_API_SCHEMA(ShadowAPI, ERNIE_SHADOW_API_ATTRIBUTES, "Controls a light's shadows.");
ERNIE_API_SCHEMA(PhotometricAreaLightAPI, ERNIE_PHOTOMETRIC_AREA_LIGHT_API_ATTRIBUTES,
                 "Omniverse's photometric units for shaped lights: lumens, or lux at a distance.");
ERNIE_API_SCHEMA(PhotometricDistantLightAPI, ERNIE_PHOTOMETRIC_DISTANT_LIGHT_API_ATTRIBUTES,
                 "Omniverse's photometric units for distant lights: lux on a facing surface.");
ERNIE_API_SCHEMA(PhotometricDomeLightAPI, ERNIE_PHOTOMETRIC_DOME_LIGHT_API_ATTRIBUTES,
                 "Omniverse's photometric units for dome lights: lux on an upward-facing "
                 "surface.");
ERNIE_API_SCHEMA(PhysicalLightIlluminantAPI, ERNIE_PHYSICAL_LIGHT_ILLUMINANT_API_ATTRIBUTES,
                 "Omniverse's illuminant: the shape of a light's spectrum.");

}  // namespace ernie
