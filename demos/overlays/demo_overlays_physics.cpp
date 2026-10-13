// SPDX-License-Identifier: BSD-3-Clause
// Physics debug drawing: projectiles with drag bouncing on the
// ground, a chain of springs and a spinning box, drawn over with
// trails coloured by speed, dashed predictions, force arrows, fading
// contacts, x-ray bounds, springs coloured by strain, frames, arcs
// and curves, labels and numbers that follow the bodies, and a
// legend.

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/overlay.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kProjectiles = 3;
constexpr int kMasses = 4;
constexpr double kGravity = 9.81;
constexpr double kRadius = 0.25;
constexpr double kDrag = 0.02;  // quadratic, per metre
constexpr double kRestitution = 0.6;
constexpr double kFriction = 0.8;  // tangential speed kept per bounce
constexpr std::size_t kTrail = 120;
constexpr double kSpring = 60.0;
constexpr double kDamping = 0.8;
constexpr double kRest = 0.8;
constexpr double kSpin = 2.0;  // radians per second
const glm::dvec3 kPivot{4.0, 0.0, 6.0};
const glm::dvec3 kBoxAt{0.0, 4.0, 1.5};
const glm::dvec3 kSpinAxis =
    glm::normalize(glm::dvec3(0.3, 0.2, 1.0));

/// `a` to `b` by `t`, alpha included.
glm::vec4 Mix(const glm::vec4& a, const glm::vec4& b, double t) {
    return a + (b - a) * static_cast<float>(std::clamp(t, 0.0, 1.0));
}

const glm::vec4 kSlow{0.2f, 0.5f, 1.0f, 1.0f};
const glm::vec4 kFast{1.0f, 0.25f, 0.1f, 1.0f};
const glm::vec4 kWhite{1.0f, 1.0f, 1.0f, 1.0f};
const glm::vec4 kVelocity{0.3f, 1.0f, 0.4f, 1.0f};
const glm::vec4 kDragColor{1.0f, 0.6f, 0.1f, 1.0f};
const glm::vec4 kContact{1.0f, 0.9f, 0.2f, 1.0f};
const glm::vec4 kBounds{0.3f, 0.9f, 1.0f, 0.7f};
const glm::vec4 kSpinColor{1.0f, 0.6f, 0.2f, 0.35f};

struct Projectile {
    glm::dvec3 position{0.0};
    glm::dvec3 velocity{0.0};
    double wait = 0.0;  // seconds until launch
    double age = 0.0;
    int launches = 0;
    std::vector<std::pair<glm::dvec3, double>> trail;  // with speeds
};

struct Contact {
    glm::dvec3 at{0.0};
    double age = 0.0;
};

class PhysicsDemo : public app::Demo {
  public:
    PhysicsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 360;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(1.0);
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(20.0)
            .SetTransform(LookAtTransform(
                {6.0, -17.0, 8.0}, {0.0, 1.0, 1.5}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution(glm::dvec2(960, 540))
            .SetOverlays("*");
        // A wall the predictions and bounds show through.
        Cube::Define(scene, "/World/Wall")
            .SetDisplayColor({0.5, 0.5, 0.55});
        SetXform(scene, "/World/Wall",
                 {.translate = glm::dvec3(-1.0, 1.0, 1.0),
                  .scale = glm::dvec3(0.15, 1.5, 1.0)});
        Cube::Define(scene, "/World/Box")
            .SetSize(0.6)
            .SetDisplayColor({0.8, 0.3, 0.8});

        // Labels and speeds posed under the bodies: added once.
        Overlay& labels = scene.GetOverlay("labels");
        OverlayStyle label;
        label.depth = OverlayDepth::Always;
        std::vector<Path> paths;
        for (int k = 0; k < kProjectiles; ++k) {
            const Path path("/World/Projectile" + std::to_string(k));
            Sphere::Define(scene, path)
                .SetRadius(kRadius)
                .SetDisplayColor({0.9, 0.35 + 0.2 * k, 0.2});
            const auto slot = static_cast<std::size_t>(k);
            projectiles_[slot].wait = 0.6 * k;
            Launch(k);
            projectiles_[slot].wait = 0.6 * k;
            label.anchor = path;
            labels.Label({0.0, 0.0, kRadius}, "p" + std::to_string(k),
                         label);
            paths.push_back(path);
        }
        OverlayBatch speeds;
        speeds.kind = OverlayBatchKind::Numbers;
        speeds.points =
            Tensor::Zeros(DType::Float32, {kProjectiles, 3});
        speeds.values = Tensor::Zeros(DType::Float32, {kProjectiles});
        speeds.format = {.suffix = " m/s", .decimals = 1};
        speeds.anchors = paths;
        speeds.style.color = {0.6f, 0.9f, 1.0f, 1.0f};
        speeds.style.depth = OverlayDepth::Always;
        speeds.style.size = 11.0f;
        auto* below = static_cast<float*>(speeds.points.Data());
        for (int k = 0; k < kProjectiles; ++k) {
            below[3 * k + 2] = static_cast<float>(-3.0 * kRadius);
        }
        labels.Batch(speeds);
        for (int m = 0; m < kMasses; ++m) {
            const Path path("/World/Mass" + std::to_string(m));
            Sphere::Define(scene, path)
                .SetRadius(0.2)
                .SetDisplayColor({0.85, 0.85, 0.9});
            masses_[static_cast<std::size_t>(m)] =
                kPivot + glm::dvec3(kRest * (m + 1), 0.0, 0.0);
        }
        label.anchor =
            Path("/World/Mass" + std::to_string(kMasses - 1));
        labels.Label({0.0, 0.0, 0.2}, "bob", label);
        label.anchor = Path("/World/Box");
        labels.Label({0.0, 0.0, 0.4}, "spinner", label);
        Place();
        Draw();
    }

    void Step() override {
        const double dt = frameDt;
        for (int k = 0; k < kProjectiles; ++k) {
            Projectile& p = projectiles_[static_cast<std::size_t>(k)];
            if (p.wait > 0.0) {
                p.wait -= dt;
                continue;
            }
            const double speed = glm::length(p.velocity);
            p.velocity += (glm::dvec3(0.0, 0.0, -kGravity) -
                           kDrag * speed * p.velocity) *
                          dt;
            p.position += p.velocity * dt;
            if (p.position.z < kRadius) {
                p.position.z = kRadius;
                if (p.velocity.z < -0.5) {
                    contacts_.push_back({glm::dvec3(
                        p.position.x, p.position.y, 0.0)});
                    ++totalContacts_;
                }
                p.velocity.z =
                    std::max(-p.velocity.z * kRestitution, 0.0);
                p.velocity.x *= kFriction;
                p.velocity.y *= kFriction;
            }
            p.trail.emplace_back(p.position, glm::length(p.velocity));
            if (p.trail.size() > kTrail) {
                p.trail.erase(p.trail.begin());
            }
            p.age += dt;
            if (p.age > 6.0 || (glm::length(p.velocity) < 0.4 &&
                                p.position.z <= kRadius)) {
                Launch(k);
            }
        }
        for (Contact& contact : contacts_) {
            contact.age += dt;
        }
        std::erase_if(contacts_,
                      [](const Contact& c) { return c.age > 1.0; });
        // The chain, semi-implicit Euler in substeps.
        constexpr int kSubsteps = 4;
        const double h = dt / kSubsteps;
        for (int s = 0; s < kSubsteps; ++s) {
            std::array<glm::dvec3, kMasses> forces;
            forces.fill(glm::dvec3(0.0, 0.0, -kGravity));
            for (std::size_t m = 0; m < kMasses; ++m) {
                const glm::dvec3 from =
                    m == 0 ? kPivot : masses_[m - 1];
                const glm::dvec3 fromVelocity =
                    m == 0 ? glm::dvec3(0.0) : massVelocities_[m - 1];
                const glm::dvec3 d = masses_[m] - from;
                const double length = glm::length(d);
                const glm::dvec3 dir = d / length;
                const double closing =
                    glm::dot(massVelocities_[m] - fromVelocity, dir);
                const glm::dvec3 f = -(kSpring * (length - kRest) +
                                       kDamping * closing) *
                                     dir;
                forces[m] += f;
                if (m > 0) {
                    forces[m - 1] -= f;
                }
            }
            for (std::size_t m = 0; m < kMasses; ++m) {
                massVelocities_[m] += forces[m] * h;
                masses_[m] += massVelocities_[m] * h;
            }
        }
        spin_ += kSpin * dt;
        Place();
        Draw();
    }

    void TestFinal() override {
        Check(totalContacts_ > 2, "projectiles should bounce");
        Check(projectiles_[0].launches > 1,
              "projectiles should launch again once settled");
        for (const glm::dvec3& mass : masses_) {
            Check(std::isfinite(mass.z) && mass.z < kPivot.z,
                  "the chain should hang below its pivot");
        }
        const Scene& scene = GetScene();
        for (const char* layer :
             {"trails", "predicted", "vectors", "bounds", "springs",
              "frames", "hud"}) {
            const Overlay* overlay = scene.FindOverlay(layer);
            Check(overlay != nullptr && !overlay->Empty(),
                  std::string("the ") + layer + " layer should draw");
        }
    }

  private:
    void Launch(int k) {
        Projectile& p = projectiles_[static_cast<std::size_t>(k)];
        const int n = p.launches + k;
        const double speed = 11.0 + 1.5 * k;
        const double elevation = glm::radians(50.0 + 8.0 * (n % 3));
        const double azimuth = glm::radians(10.0 - 12.0 * k);
        p.position = {-8.0, -1.5 + 1.5 * k, kRadius};
        p.velocity =
            speed *
            glm::dvec3(std::cos(elevation) * std::cos(azimuth),
                       std::cos(elevation) * std::sin(azimuth),
                       std::sin(elevation));
        p.wait = 0.5;
        p.age = 0.0;
        p.trail.clear();
        ++p.launches;
    }

    glm::dmat4 BoxTransform() const {
        glm::dmat4 m =
            glm::mat4_cast(glm::angleAxis(spin_, kSpinAxis));
        m[3] = glm::dvec4(kBoxAt, 1.0);
        return m;
    }

    void Place() {
        Scene& scene = GetScene();
        auto* speeds = static_cast<float*>(
            scene.GetOverlay("labels").Batches()[0].values.Data());
        for (std::size_t k = 0; k < kProjectiles; ++k) {
            SetXform(scene, "/World/Projectile" + std::to_string(k),
                     {.translate = projectiles_[k].position});
            speeds[k] = static_cast<float>(
                glm::length(projectiles_[k].velocity));
        }
        scene.GetOverlay("labels").Touch();
        for (std::size_t m = 0; m < kMasses; ++m) {
            SetXform(scene, "/World/Mass" + std::to_string(m),
                     {.translate = masses_[m]});
        }
        SetXform(scene, "/World/Box",
                 {.translate = kBoxAt,
                  .orient = glm::angleAxis(spin_, kSpinAxis)});
    }

    void Draw() {
        Scene& scene = GetScene();
        // Trails: a segment per step, coloured by speed, fading with
        // age.
        Overlay& trails = scene.GetOverlay("trails").Clear();
        OverlayStyle trail;
        trail.width = 3.0f;
        for (const Projectile& p : projectiles_) {
            std::vector<glm::dvec3> ends;
            std::vector<glm::vec4> colors;
            for (std::size_t i = 1; i < p.trail.size(); ++i) {
                ends.push_back(p.trail[i - 1].first);
                ends.push_back(p.trail[i].first);
                glm::vec4 color =
                    Mix(kSlow, kFast, p.trail[i].second / 15.0);
                color.a = static_cast<float>(i) /
                          static_cast<float>(p.trail.size());
                colors.push_back(color);
            }
            trails.Lines(ends, trail, colors);
        }
        // Predictions without drag: dashed through walls, a cross
        // every quarter second.
        Overlay& predicted = scene.GetOverlay("predicted").Clear();
        OverlayStyle path;
        path.dash = 6.0f;
        path.color = {1.0f, 1.0f, 1.0f, 0.8f};
        path.depth = OverlayDepth::XRay;
        OverlayStyle tick = path;
        tick.size = 8.0f;
        tick.color = kContact;
        for (const Projectile& p : projectiles_) {
            if (p.wait > 0.0 || p.position.z <= kRadius) {
                continue;
            }
            std::vector<glm::dvec3> points;
            for (int i = 0; i <= 60; ++i) {
                const double t = 0.05 * i;
                const glm::dvec3 q =
                    p.position + p.velocity * t +
                    glm::dvec3(0.0, 0.0, -0.5 * kGravity * t * t);
                points.push_back(q);
                if (i % 5 == 0 && i > 0) {
                    predicted.Marker(q, tick, MarkerShape::Cross);
                }
                if (q.z < kRadius) {
                    break;
                }
            }
            predicted.Polyline(points, path);
        }
        // Velocity, drag and gravity on each projectile, velocity on
        // each mass.
        Overlay& vectors = scene.GetOverlay("vectors").Clear();
        OverlayStyle arrow;
        arrow.width = 2.5f;
        for (const Projectile& p : projectiles_) {
            if (p.wait > 0.0) {
                continue;
            }
            arrow.color = kVelocity;
            vectors.Arrow(p.position, p.position + 0.15 * p.velocity,
                          arrow);
            arrow.color = kDragColor;
            vectors.Arrow(p.position,
                          p.position - 0.5 * kDrag *
                                           glm::length(p.velocity) *
                                           p.velocity,
                          arrow);
            arrow.color = kFast;
            vectors.Arrow(p.position,
                          p.position + glm::dvec3(0.0, 0.0, -0.8),
                          arrow);
        }
        arrow.color = kVelocity;
        for (std::size_t m = 0; m < kMasses; ++m) {
            vectors.Arrow(masses_[m],
                          masses_[m] + 0.15 * massVelocities_[m],
                          arrow);
        }
        // Contacts: rings growing and fading over a second, over
        // everything, with their normals.
        Overlay& contacts = scene.GetOverlay("contacts").Clear();
        for (const Contact& contact : contacts_) {
            OverlayStyle ring;
            ring.depth = OverlayDepth::Always;
            ring.color = kContact;
            ring.color.a = static_cast<float>(1.0 - contact.age);
            ring.size = static_cast<float>(10.0 + 40.0 * contact.age);
            contacts.Marker(contact.at, ring, MarkerShape::Ring);
            ring.width = 2.0f;
            contacts.Arrow(
                contact.at,
                contact.at + glm::dvec3(0.0, 0.0, 1.0 - contact.age),
                ring);
        }
        // Bounds in x-ray: boxes around the projectiles, the box's
        // own turned with it, spheres around the masses.
        Overlay& bounds = scene.GetOverlay("bounds").Clear();
        OverlayStyle bound;
        bound.width = 1.5f;
        bound.color = kBounds;
        bound.depth = OverlayDepth::XRay;
        for (const Projectile& p : projectiles_) {
            bounds.Box(p.position - kRadius, p.position + kRadius,
                       bound);
        }
        bounds.Box(glm::dvec3(-0.3), glm::dvec3(0.3), bound,
                   BoxTransform());
        bound.color.a = 0.4f;
        for (const glm::dvec3& mass : masses_) {
            bounds.Sphere(mass, 0.3, bound);
        }
        // Springs: blue squeezed, white at rest, red stretched, wider
        // the more strained; and a rope curved through the chain.
        Overlay& springs = scene.GetOverlay("springs").Clear();
        for (std::size_t m = 0; m < kMasses; ++m) {
            const glm::dvec3 from = m == 0 ? kPivot : masses_[m - 1];
            const double strain =
                (glm::distance(from, masses_[m]) - kRest) / kRest;
            OverlayStyle spring;
            spring.color = strain > 0.0
                               ? Mix(kWhite, kFast, 4.0 * strain)
                               : Mix(kWhite, kSlow, -4.0 * strain);
            spring.width = static_cast<float>(
                std::min(2.0 + 40.0 * std::abs(strain), 8.0));
            springs.Line(from, masses_[m], spring);
        }
        OverlayStyle rope;
        rope.color = {0.8f, 0.8f, 1.0f, 0.5f};
        rope.dash = 3.0f;
        springs.Curve(kPivot, masses_[0], masses_[2], masses_[3],
                      rope);
        OverlayStyle pivot;
        pivot.size = 10.0f;
        springs.Marker(kPivot, pivot, MarkerShape::Diamond);
        // The box's frame, its spin axis and the turn it makes in
        // half a second, as a filled arc.
        Overlay& frames = scene.GetOverlay("frames").Clear();
        const glm::dmat4 box = BoxTransform();
        frames.Axes(box, 0.8);
        OverlayStyle spin;
        spin.color = kSpinColor;
        frames.Arc(kBoxAt, kSpinAxis, glm::dvec3(box[0]), 0.9,
                   0.5 * kSpin, spin, true);
        spin.color.a = 1.0f;
        spin.width = 2.5f;
        frames.Arc(kBoxAt, kSpinAxis, glm::dvec3(box[0]), 0.9,
                   0.5 * kSpin, spin);
        frames.Arrow(kBoxAt, kBoxAt + 0.6 * kSpin * kSpinAxis, spin);
        DrawHud();
    }

    void DrawHud() {
        Overlay& hud = GetScene().GetOverlay("hud").Clear();
        OverlayStyle text;
        text.size = 14.0f;
        char line[160];
        int launches = 0;
        for (const Projectile& p : projectiles_) {
            launches += p.launches;
        }
        std::snprintf(line, sizeof(line),
                      "physics debug draw\nt = %.2f s\nlaunches %d  "
                      "contacts %d",
                      time, launches, totalContacts_);
        hud.Text({.offset = {12.0f, 12.0f}}, line, text);
        // The legend: a swatch and a name a row, down the left side.
        const std::pair<glm::vec4, const char*> legend[] = {
            {kFast, "trail, by speed"},
            {kWhite, "prediction, dashed x-ray"},
            {kVelocity, "velocity"},
            {kDragColor, "drag"},
            {kContact, "contact"},
            {kBounds, "bounds, x-ray"},
            {kSlow, "spring squeezed"},
            {kSpinColor, "spin"}};
        text.size = 12.0f;
        text.background.a = 0.0f;
        OverlayStyle panel;
        panel.color = {0.0f, 0.0f, 0.0f, 0.5f};
        // Each item's top left corner at its place.
        const auto at = [](float x, float y) {
            return ScreenPlacement{.at = {0.0f, 0.3f},
                                   .offset = {x, y},
                                   .align = glm::vec2(0.0f)};
        };
        hud.Rect(at(6.0f, -6.0f),
                 {196.0f, 18.0f * std::size(legend) + 8.0f}, panel);
        for (std::size_t i = 0; i < std::size(legend); ++i) {
            const float y = 18.0f * static_cast<float>(i);
            OverlayStyle swatch;
            swatch.color = legend[i].first;
            swatch.color.a = 1.0f;
            hud.Rect(at(12.0f, y + 3.0f), {22.0f, 6.0f}, swatch);
            hud.Text(at(42.0f, y), legend[i].second, text);
        }
    }

    std::array<Projectile, kProjectiles> projectiles_;
    std::vector<Contact> contacts_;
    int totalContacts_ = 0;
    std::array<glm::dvec3, kMasses> masses_{};
    std::array<glm::dvec3, kMasses> massVelocities_{};
    double spin_ = 0.0;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    PhysicsDemo, "overlays", "physics",
    "Physics debug drawing: trails, predictions, "
    "forces, contacts, bounds, springs and frames over "
    "a small simulation.");
