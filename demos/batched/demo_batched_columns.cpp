// SPDX-License-Identifier: BSD-3-Clause
// A field of cubes riding a travelling wave, every cube's pose and
// colour written each frame as two columns (one array each, mapped and
// written in place): one change per array however many cubes.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/glm.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kSide = 64;  // cubes along each side
constexpr double kSpacing = 1.2;  // m between cubes
constexpr double kAmplitude = 1.5;  // m
constexpr double kWavelength = 18.0;  // m
constexpr double kSpeed = 6.0;  // m/s

/// Cube `i`'s place at `t` seconds: on the grid, lifted by a ring wave
/// travelling out from the centre.
glm::dvec3 PlaceAt(int i, double t) {
    const double half = 0.5 * (kSide - 1) * kSpacing;
    const double x = (i % kSide) * kSpacing - half;
    const double y = (i / kSide) * kSpacing - half;
    const double r = std::hypot(x, y);
    const double phase =
        2.0 * std::numbers::pi * (r - kSpeed * t) / kWavelength;
    return {x, y, kAmplitude * std::sin(phase)};
}

/// A colour from blue (trough) to orange (crest) by height.
glm::dvec3 ColourAt(double z) {
    const double h = 0.5 + 0.5 * z / kAmplitude;
    return glm::mix(glm::dvec3(0.1, 0.3, 0.9), glm::dvec3(1.0, 0.5, 0.1),
                    h);
}

class ColumnsDemo : public app::Demo {
  public:
    ColumnsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.4, -0.5, 0.8}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(18.0)
            .SetTransform(LookAtTransform({0.0, -70.0, 45.0},
                                          {0.0, 0.0, 0.0},
                                          {0.0, 0.0, 1.0}));
        std::vector<Path> cubes;
        for (int i = 0; i < kSide * kSide; ++i) {
            const Path cube("/World/Field/Cube" + std::to_string(i));
            Cube::Define(scene, cube).SetSize(1.0);
            SetXform(scene, cube, {.translate = PlaceAt(i, 0.0)});
            cubes.push_back(cube);
        }
        // Each attribute of every cube as one column.
        const PrimSet field(cubes);
        poses_ = scene.GetColumn(field, tokens::kTranslate);
        colours_ = scene.GetColumn(field, tokens::kDisplayColor);
        colours_.Write(Tensor::Zeros(DType::Float32, {kSide * kSide, 3}));
        Place();
    }

    void Step() override {
        time_ += frameDt;
        Place();
    }

    void TestPostStep() override {
        // Rows read as each cube's own attribute.
        for (const int i : {0, kSide * kSide / 2 + 7, kSide * kSide - 1}) {
            const glm::dvec3 want = PlaceAt(i, time_);
            const glm::dvec3 at(GetScene().WorldTransform(
                "/World/Field/Cube" + std::to_string(i))[3]);
            Check(glm::distance(at, want) < 1e-4,
                  "each cube is where its row puts it");
        }
    }

    void TestFinal() override {
        ChangeCursor cursor{GetScene().Version()};
        Place();
        std::vector<ColumnChange> columns;
        const std::vector<Change> rest = GetScene().Changes(cursor, columns);
        Check(columns.size() == 2 && rest.empty(),
              "a frame's writes are two column changes, none per cube");
    }

  private:
    /// Every cube's row of both columns, written in place.
    void Place() {
        const Device host{};
        float* poses = poses_.Map(host).As<float>();
        float* colours = colours_.Map(host).As<float>();
        for (int i = 0; i < kSide * kSide; ++i) {
            const glm::dvec3 p = PlaceAt(i, time_);
            const glm::dvec3 c = ColourAt(p.z);
            for (int k = 0; k < 3; ++k) {
                poses[3 * i + k] = static_cast<float>(p[k]);
                colours[3 * i + k] = static_cast<float>(c[k]);
            }
        }
        poses_.Unmap();
        colours_.Unmap();
    }

    Column poses_;
    Column colours_;
    double time_ = 0.0;
};

}  // namespace

ERNIE_REGISTER_DEMO(ColumnsDemo, "batched", "columns",
                    "4096 cubes riding a wave, every pose and colour "
                    "written each frame as two columns.");
