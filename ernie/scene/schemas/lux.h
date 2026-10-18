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

/// LightAPI's attributes, which every light has; DistantLight brightens the intensity's fallback.
#define ERNIE_LIGHT_ATTRIBUTES(X, intensity)                                                       \
    X(double, Intensity, "inputs:intensity", intensity, "Brightness, scaled by 2^exposure.",       \
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
      .group = "Light", .lower = Above(0.0), .finite = true)

#define ERNIE_DISTANT_LIGHT_ATTRIBUTES(X)                                                          \
    ERNIE_LIGHT_ATTRIBUTES(X, 50000.0)                                                             \
    X(double, Angle, "inputs:angle", 0.53, "Angular diameter in degrees; the Sun's is 0.53.",      \
      .group = "Light", .lower = AtLeast(0.0), .upper = AtMost(360.0))

#define ERNIE_DISK_LIGHT_ATTRIBUTES(X)                                                             \
    ERNIE_LIGHT_ATTRIBUTES(X, 1.0)                                                                 \
    X(double, Radius, "inputs:radius", 0.5, "Metres; a disk facing -z.", ERNIE_LIGHT_LENGTH)

#define ERNIE_RECT_LIGHT_ATTRIBUTES(X)                                                             \
    ERNIE_LIGHT_ATTRIBUTES(X, 1.0)                                                                 \
    X(double, Width, "inputs:width", 1.0, "Metres along x; a rectangle facing -z.",                \
      ERNIE_LIGHT_LENGTH)                                                                          \
    X(double, Height, "inputs:height", 1.0, "Metres along y.", ERNIE_LIGHT_LENGTH)                 \
    X(std::string, TextureFile, "inputs:texture:file", "", "An image it emits.", .group = "Light", \
      .asset = true)

#define ERNIE_SPHERE_LIGHT_ATTRIBUTES(X)                                                           \
    ERNIE_LIGHT_ATTRIBUTES(X, 1.0)                                                                 \
    X(double, Radius, "inputs:radius", 0.5, "Metres.", ERNIE_LIGHT_LENGTH)                         \
    X(bool, TreatAsPoint, "treatAsPoint", false, "Lights as a point however large it is.",         \
      .group = "Light")

#define ERNIE_CYLINDER_LIGHT_ATTRIBUTES(X)                                                         \
    ERNIE_LIGHT_ATTRIBUTES(X, 1.0)                                                                 \
    X(double, Length, "inputs:length", 1.0, "Metres along x.", ERNIE_LIGHT_LENGTH)                 \
    X(double, Radius, "inputs:radius", 0.5, "Metres.", ERNIE_LIGHT_LENGTH)                         \
    X(bool, TreatAsLine, "treatAsLine", false, "Lights as a line however thick it is.",            \
      .group = "Light")

#define ERNIE_DOME_LIGHT_ATTRIBUTES(X)                                                             \
    ERNIE_LIGHT_ATTRIBUTES(X, 1.0)                                                                 \
    X(std::string, TextureFile, "inputs:texture:file", "", "The environment it emits.",            \
      .group = "Light", .asset = true)                                                             \
    X(std::string, TextureFormat, "inputs:texture:format", "automatic", "The texture's mapping.",  \
      .group = "Light",                                                                            \
      .options = {"automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"})      \
    X(std::vector<Path>, Portals, "portals", std::vector<Path>{},                                  \
      "Portal lights its light enters through.", .group = "Light")                                 \
    X(double, GuideRadius, "guideRadius", 1.0e5, "Metres to the sphere drawn as its guide.",       \
      .group = "Light", .lower = Above(0.0))

#define ERNIE_PORTAL_LIGHT_ATTRIBUTES(X)                                                           \
    ERNIE_LIGHT_ATTRIBUTES(X, 1.0)                                                                 \
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

ERNIE_SCHEMA_ACCESSORS(BoundableLightBase, Boundable, ERNIE_NO_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(NonboundableLightBase, Xformable, ERNIE_NO_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(DistantLight, NonboundableLightBase, ERNIE_DISTANT_LIGHT_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(DiskLight, BoundableLightBase, ERNIE_DISK_LIGHT_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(RectLight, BoundableLightBase, ERNIE_RECT_LIGHT_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(SphereLight, BoundableLightBase, ERNIE_SPHERE_LIGHT_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(CylinderLight, BoundableLightBase, ERNIE_CYLINDER_LIGHT_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(DomeLight, NonboundableLightBase, ERNIE_DOME_LIGHT_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(PortalLight, BoundableLightBase, ERNIE_PORTAL_LIGHT_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(ShapingAPI, SchemaView, ERNIE_SHAPING_API_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(ShadowAPI, SchemaView, ERNIE_SHADOW_API_ATTRIBUTES);

/// A light with a shape, bounded by it.
class BoundableLightBase : public BoundableLightBaseAccessors<BoundableLightBase, Boundable> {
    ERNIE_SCHEMA_BODY(BoundableLightBase, Boundable)
};

/// A light at infinity or around everything, so unbounded.
class NonboundableLightBase
    : public NonboundableLightBaseAccessors<NonboundableLightBase, Xformable> {
    ERNIE_SCHEMA_BODY(NonboundableLightBase, Xformable)
};

/// A light from infinitely far along +z, such as the Sun's.
class DistantLight : public DistantLightAccessors<DistantLight, NonboundableLightBase> {
    ERNIE_SCHEMA_BODY(DistantLight, NonboundableLightBase)
    ERNIE_SCHEMA_DEFINE(DistantLight)
};

/// A disk emitting towards -z.
class DiskLight : public DiskLightAccessors<DiskLight, BoundableLightBase> {
    ERNIE_SCHEMA_BODY(DiskLight, BoundableLightBase)
    ERNIE_SCHEMA_DEFINE(DiskLight)
};

/// A rectangle emitting towards -z.
class RectLight : public RectLightAccessors<RectLight, BoundableLightBase> {
    ERNIE_SCHEMA_BODY(RectLight, BoundableLightBase)
    ERNIE_SCHEMA_DEFINE(RectLight)
};

/// A sphere emitting outwards.
class SphereLight : public SphereLightAccessors<SphereLight, BoundableLightBase> {
    ERNIE_SCHEMA_BODY(SphereLight, BoundableLightBase)
    ERNIE_SCHEMA_DEFINE(SphereLight)
};

/// A cylinder along x emitting outwards.
class CylinderLight : public CylinderLightAccessors<CylinderLight, BoundableLightBase> {
    ERNIE_SCHEMA_BODY(CylinderLight, BoundableLightBase)
    ERNIE_SCHEMA_DEFINE(CylinderLight)
};

/// Light from all around, from an environment image.
class DomeLight : public DomeLightAccessors<DomeLight, NonboundableLightBase> {
    ERNIE_SCHEMA_BODY(DomeLight, NonboundableLightBase)
    ERNIE_SCHEMA_DEFINE(DomeLight)
};

/// A rectangle a dome light's light comes in through, guiding its samples.
class PortalLight : public PortalLightAccessors<PortalLight, BoundableLightBase> {
    ERNIE_SCHEMA_BODY(PortalLight, BoundableLightBase)
    ERNIE_SCHEMA_DEFINE(PortalLight)
};

/// Focuses a light, limits it to a cone, or shapes it by an IES profile.
class ShapingAPI : public ShapingAPIAccessors<ShapingAPI, SchemaView> {
    ERNIE_API_SCHEMA_BODY(ShapingAPI)
};

/// Controls a light's shadows.
class ShadowAPI : public ShadowAPIAccessors<ShadowAPI, SchemaView> {
    ERNIE_API_SCHEMA_BODY(ShadowAPI)
};

}  // namespace ernie
