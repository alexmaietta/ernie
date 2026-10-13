// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/delegate.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "support/delegate_scenes.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace ernie::testing;

namespace {

class CascadeShadowTest : public VulkanTest {};

/// A sphere casting into the reference scenes.
struct Ball {
    glm::dvec3 centre;
    double radius;
};

/// The distance along unit `d` from `o` to where it leaves or enters
/// `ball` ahead, or a negative one where it misses it.
double Hit(const glm::dvec3& o, const glm::dvec3& d,
           const Ball& ball) {
    const glm::dvec3 oc = o - ball.centre;
    const double b = glm::dot(oc, d);
    const double c = glm::dot(oc, oc) - ball.radius * ball.radius;
    const double discriminant = b * b - c;
    if (discriminant < 0.0) {
        return -1.0;
    }
    const double root = std::sqrt(discriminant);
    return -b - root > 0.0 ? -b - root : -b + root;
}

/// A reference scene: balls on a grey floor at z = 0 under a point
/// sun from `toSun`, seen from `eye` towards `target` through a lens
/// of `focal` mm on a 36 mm sensor.
struct Reference {
    std::vector<Ball> balls;
    glm::dvec3 toSun{0.0, 0.0, 1.0};
    glm::dvec3 eye{0.0};
    glm::dvec3 target{0.0};
    double focal = 24.0;
    std::uint32_t width = 640;
    std::uint32_t height = 360;
    /// The floor an Earth-sized planet's, its top at z = 0, rather
    /// than a ground plane: compared within `reach` of the eye, where
    /// it is flat to a millimetre.
    bool planet = false;
    double reach = 1e9;
    /// The cascades' reach (`tokens::kShadowDistance`), 0 for as far
    /// as the view sees.
    double shadowDistance = 0.0;
};

/// How a scene's floor pixels shadow against the exact shadows of its
/// balls.
struct ShadowCheck {
    std::size_t floor = 0;     ///< Floor pixels compared.
    std::size_t shadowed = 0;  ///< Of them, those the balls shadow.
    std::size_t wrong = 0;     ///< Of them, lit or shadowed wrongly.
    /// Floor pixels within two of a shadow's edge, and the sum of
    /// their visibilities' differences from the exact answer at their
    /// centres: about a quarter of them for a sharp edge where it
    /// should be, more as it blurs, steps or wanders.
    std::size_t edge = 0;
    double edgeError = 0.0;
};

/// Renders `reference` through the sun's cascades and compares each
/// pixel that sees the floor with the exact answer: its ray met on
/// the floor, and the sun's ray from there tested against every ball.
/// Pixels within two of a change of answer, or of a ball or the sky,
/// are left out of `wrong`, as edges a texel wide.
ShadowCheck CheckShadows(const ernie::VulkanContext& context,
                         const Reference& reference) {
    using namespace ernie;
    namespace tokens = ernie::tokens;
    Scene scene;
    if (reference.planet) {
        constexpr double kEarth = 6371008.8;
        Terrain::Define(scene, "/World/Floor")
            .SetHeights(Tensor::Copy<float>(
                std::vector<float>(6 * 65 * 65, 0.0f), {6, 65, 65}))
            .SetPlanetRadius(kEarth)
            .SetMaterialBinding("/Looks/Grey");
        scene.Set("/World/Floor", tokens::kTransform,
                  glm::translate(glm::dmat4(1.0),
                                 glm::dvec3(0.0, 0.0, -kEarth)));
    } else {
        GroundPlane::Define(scene, "/World/Floor")
            .SetMaterialBinding("/Looks/Grey");
    }
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    for (std::size_t i = 0; i < reference.balls.size(); ++i) {
        const std::string path = "/World/Ball" + std::to_string(i);
        Sphere::Define(scene, path)
            .SetRadius(reference.balls[i].radius)
            .SetMaterialBinding("/Looks/Grey");
        scene.Set(path, tokens::kTransform,
                  glm::translate(glm::dmat4(1.0),
                                 reference.balls[i].centre));
    }
    const glm::dvec3 toSun = glm::normalize(reference.toSun);
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform(toSun));
    const double aspect = static_cast<double>(reference.height) /
                          static_cast<double>(reference.width);
    Camera::Define(scene, "/World/Camera")
        .SetSensorNoise(false)
        .SetFocalLength(reference.focal)
        .SetHorizontalAperture(36.0)
        .SetVerticalAperture(36.0 * aspect)
        .SetClippingRange(glm::dvec2(0.1, 1e5));
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform(reference.eye, reference.target,
                              {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(reference.width, reference.height))
        .SetAmbient(0.0)
        .SetShadowDistance(reference.shadowDistance);
    // Once earlier frames have measured what the view sees, which the
    // cascades' reach fits to (`RasterRenderer::FitCascadeDistance`).
    RasterDelegate delegate(context);
    Tensor image;
    for (int frame = 0; frame < 4; ++frame) {
        image = delegate.Render(scene);
    }
    const auto* pixels = image.As<std::uint8_t>();

    // The exact answers: 0 a ball or the sky, 1 lit floor, 2
    // shadowed.
    const std::size_t w = reference.width;
    const std::size_t h = reference.height;
    const glm::dvec3 forward =
        glm::normalize(reference.target - reference.eye);
    const glm::dvec3 right = glm::normalize(
        glm::cross(forward, glm::dvec3(0.0, 0.0, 1.0)));
    const glm::dvec3 up = glm::cross(right, forward);
    const double tanX = 18.0 / reference.focal;
    const double tanY = tanX * aspect;
    std::vector<int> exact(w * h, 0);
    for (std::size_t r = 0; r < h; ++r) {
        for (std::size_t c = 0; c < w; ++c) {
            const double x = ((static_cast<double>(c) + 0.5) /
                                  static_cast<double>(w) * 2.0 -
                              1.0) *
                             tanX;
            const double y =
                (1.0 - (static_cast<double>(r) + 0.5) /
                           static_cast<double>(h) * 2.0) *
                tanY;
            const glm::dvec3 d =
                glm::normalize(forward + x * right + y * up);
            if (d.z >= 0.0) {
                continue;
            }
            const double t = -reference.eye.z / d.z;
            if (t > reference.reach) {
                continue;
            }
            bool seesBall = false;
            for (const Ball& ball : reference.balls) {
                const double hit = Hit(reference.eye, d, ball);
                seesBall = seesBall || (hit > 0.0 && hit < t);
            }
            if (seesBall) {
                continue;
            }
            const glm::dvec3 p = reference.eye + d * t;
            bool shadowed = false;
            for (const Ball& ball : reference.balls) {
                shadowed = shadowed || Hit(p, toSun, ball) > 0.0;
            }
            exact[r * w + c] = shadowed ? 2 : 1;
        }
    }
    // The lit floor's level, and each compared pixel against half it.
    std::vector<int> lit;
    for (std::size_t i = 0; i < w * h; ++i) {
        if (exact[i] == 1) {
            lit.push_back(pixels[i * 4 + 1]);
        }
    }
    ShadowCheck check;
    if (lit.empty()) {
        return check;
    }
    std::nth_element(lit.begin(),
                     lit.begin() +
                         static_cast<std::ptrdiff_t>(lit.size() / 2),
                     lit.end());
    const int threshold = lit[lit.size() / 2] / 2;
    for (std::size_t r = 2; r + 2 < h; ++r) {
        for (std::size_t c = 2; c + 2 < w; ++c) {
            const int answer = exact[r * w + c];
            if (answer == 0) {
                continue;
            }
            bool edge = false;
            bool shadowEdge = false;
            for (std::size_t i = r - 2; i <= r + 2; ++i) {
                for (std::size_t j = c - 2; j <= c + 2; ++j) {
                    const int other = exact[i * w + j];
                    edge = edge || other != answer;
                    shadowEdge =
                        shadowEdge || (other != 0 && other != answer);
                }
            }
            if (shadowEdge) {
                const double seen = std::clamp(
                    pixels[(r * w + c) * 4 + 1] / (2.0 * threshold),
                    0.0, 1.0);
                ++check.edge;
                check.edgeError +=
                    std::abs(seen - (answer == 1 ? 1.0 : 0.0));
            }
            if (edge) {
                continue;
            }
            ++check.floor;
            check.shadowed += answer == 2 ? 1u : 0u;
            const bool dark = pixels[(r * w + c) * 4 + 1] < threshold;
            check.wrong += dark != (answer == 2) ? 1u : 0u;
        }
    }
    return check;
}

/// Expects `reference` through the cascades to have compared a
/// shadowed floor and got at most one pixel in `wrongIn` wrong, and
/// its shadows' edges within `edgeError` of the exact answer on
/// average.
void ExpectExact(const ernie::VulkanContext& context,
                 const Reference& reference,
                 std::size_t wrongIn = 300, double edgeError = 0.15) {
    const ShadowCheck check = CheckShadows(context, reference);
    ASSERT_GT(check.floor, 10000u);
    ASSERT_GT(check.shadowed, 10u);
    ASSERT_GT(check.edge, 100u);
    const std::string report = std::to_string(check.wrong) + " of " +
                               std::to_string(check.floor) +
                               " floor pixels wrong, edge error " +
                               std::to_string(check.edgeError) +
                               " over " + std::to_string(check.edge);
    EXPECT_LE(check.wrong * wrongIn, check.floor) << report;
    EXPECT_LE(check.edgeError,
              edgeError * static_cast<double>(check.edge))
        << report;
}

}  // namespace

// The reference scenes below hold the sun's cascades to the exact
// shadows of balls under a point sun, pixel for pixel away from their
// edges.

// A ball resting on the floor, its shadow up to where it touches.
TEST_F(CascadeShadowTest, ReferenceABallRestingOnTheFloor) {
    Reference reference;
    reference.balls = {{{0.0, 0.0, 1.0}, 1.0}};
    reference.toSun = {0.4, -0.6, 0.7};
    reference.eye = {-6.0, -8.0, 4.0};
    reference.target = {0.0, 0.0, 0.5};
    ExpectExact(*context_, reference);
}

// Seen at grazing angles from 1.5 m up, balls 10 to 80 m out, across
// cascades.
TEST_F(CascadeShadowTest, ReferenceBallsSeenAlongTheFloor) {
    Reference reference;
    reference.balls = {{{10.0, 1.0, 0.5}, 0.5},
                       {{30.0, -2.0, 1.0}, 1.0},
                       {{80.0, 4.0, 2.0}, 2.0}};
    reference.toSun = {-0.5, -0.6, 0.6};
    reference.eye = {0.0, 0.0, 1.5};
    reference.target = {40.0, 0.0, 0.0};
    ExpectExact(*context_, reference);
}

// As on an Earth-sized planet, whose horizon kilometres off the
// cascades would reach as far as the view sees: within 200 m, as
// Unreal's shadow distance, they keep a ball's shadow sharp.
TEST_F(CascadeShadowTest, ReferenceBallsSeenAlongAPlanet) {
    Reference reference;
    reference.balls = {{{10.0, 1.0, 0.5}, 0.5},
                       {{30.0, -2.0, 1.0}, 1.0},
                       {{80.0, 4.0, 2.0}, 2.0}};
    reference.toSun = {-0.5, -0.6, 0.6};
    reference.eye = {0.0, 0.0, 1.5};
    reference.target = {40.0, 0.0, 0.0};
    reference.planet = true;
    reference.reach = 150.0;
    reference.shadowDistance = 200.0;
    ExpectExact(*context_, reference);
}

// A ball resting on an Earth-sized planet, the cascades within 200 m.
TEST_F(CascadeShadowTest, ReferenceABallRestingOnAPlanet) {
    Reference reference;
    reference.balls = {{{0.0, 0.0, 1.0}, 1.0}};
    reference.toSun = {0.4, -0.6, 0.7};
    reference.eye = {-6.0, -8.0, 4.0};
    reference.target = {0.0, 0.0, 0.5};
    reference.planet = true;
    reference.reach = 150.0;
    reference.shadowDistance = 200.0;
    ExpectExact(*context_, reference);
}

// A sun 5 degrees up: long shadows across a steeply slanting light.
TEST_F(CascadeShadowTest, ReferenceALowSun) {
    Reference reference;
    reference.balls = {{{0.0, 0.0, 1.0}, 1.0}};
    reference.toSun = {std::cos(glm::radians(5.0)), 0.0,
                       std::sin(glm::radians(5.0))};
    reference.eye = {-4.0, -14.0, 6.0};
    reference.target = {-8.0, 0.0, 0.0};
    ExpectExact(*context_, reference);
}

// A long lens on a ball 600 m away.
TEST_F(CascadeShadowTest, ReferenceALongLensFarAway) {
    Reference reference;
    reference.balls = {{{600.0, 0.0, 3.0}, 2.0}};
    reference.toSun = {-0.3, -0.5, 0.8};
    reference.eye = {0.0, 0.0, 60.0};
    reference.target = {600.0, 0.0, 0.0};
    reference.focal = 400.0;
    ExpectExact(*context_, reference);
}

// Small balls seen from 20 m up, half a metre across.
TEST_F(CascadeShadowTest, ReferenceSmallBallsFromAbove) {
    Reference reference;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            reference.balls.push_back(
                {{-4.0 + 2.0 * i, -4.0 + 2.0 * j, 0.6 + 0.3 * i},
                 0.25});
        }
    }
    reference.toSun = {0.3, -0.4, 0.85};
    reference.eye = {0.0, -6.0, 20.0};
    reference.target = {0.0, 0.0, 0.0};
    reference.focal = 35.0;
    ExpectExact(*context_, reference);
}

// A ball floating 20 m over the floor, its shadow far from it.
TEST_F(CascadeShadowTest, ReferenceAHighBall) {
    Reference reference;
    reference.balls = {{{0.0, 0.0, 20.0}, 2.0}};
    reference.toSun = {0.2, 0.3, 0.9};
    reference.eye = {-10.0, -25.0, 15.0};
    reference.target = {-4.0, -6.0, 0.0};
    // Its shadow 25 m off, where the 5x5 filter's two and a half
    // texels either way span a pixel or more.
    ExpectExact(*context_, reference, 300, 0.18);
}

namespace {

/// Pixels of `image` (RGBA8) in a shadow's penumbra: green between a
/// quarter and three quarters of the image's brightest.
std::size_t PenumbraPixels(const ernie::Tensor& image) {
    const auto* pixels = image.As<std::uint8_t>();
    const auto count = static_cast<std::size_t>(image.Elements() / 4);
    int lit = 0;
    for (std::size_t i = 0; i < count; ++i) {
        lit = std::max<int>(lit, pixels[i * 4 + 1]);
    }
    std::size_t penumbra = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const int g = pixels[i * 4 + 1];
        penumbra += g > lit / 4 && g < 3 * lit / 4 ? 1u : 0u;
    }
    return penumbra;
}

/// A 2 m square roof `height` metres over a floor, under a sun of
/// angular diameter `angle` degrees straight above, seen from beside
/// and above it.
ernie::Tensor RoofShadow(ernie::RasterDelegate& delegate,
                         double height, double angle) {
    using namespace ernie;
    namespace tokens = ernie::tokens;
    Scene scene;
    GroundPlane::Define(scene, "/World/Floor")
        .SetMaterialBinding("/Looks/Grey");
    Material::Define(scene, "/Looks/Grey").SetAlbedo(glm::dvec3(0.5));
    DefineQuad(scene, "/World/Roof", 1.0f);
    scene.Set("/World/Roof", tokens::kTransform,
              glm::translate(glm::dmat4(1.0),
                             glm::dvec3(0.0, 0.0, height)));
    DistantLight::Define(scene, "/World/Sun").SetAngle(angle);
    scene.Set("/World/Sun", tokens::kTransform,
              DistantLightTransform({0.0, 0.0, 1.0}));
    Camera::Define(scene, "/World/Camera")
        .SetSensorNoise(false)
        .SetClippingRange(glm::dvec2(0.1, 1000.0));
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -6.0, height + 2.0},
                              {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution(glm::dvec2(96.0, 96.0))
        .SetAmbient(0.0);
    return delegate.Render(scene);
}

}  // namespace

// The sun's angular size softens the cascades' shadows: the penumbra
// widens with the caster's height over the receiver, and a point
// sun's stays a texel or two.
TEST_F(CascadeShadowTest, TheSunsSizeWidensThePenumbraWithHeight) {
    ernie::RasterDelegate delegate(*context_);
    const std::size_t low =
        PenumbraPixels(RoofShadow(delegate, 0.5, 5.0));
    const std::size_t high =
        PenumbraPixels(RoofShadow(delegate, 3.0, 5.0));
    const std::size_t point =
        PenumbraPixels(RoofShadow(delegate, 3.0, 0.0));
    EXPECT_GT(high, 2 * low);
    EXPECT_GT(high, 3 * point);
}
