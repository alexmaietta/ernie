// SPDX-License-Identifier: BSD-3-Clause
// Ripples spreading over a pond by the wave equation, the height
// field logged each step as a live heatmap (LogArray).

#include "ernie/app/demo.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <cmath>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kSize = 64;       // heights a side
constexpr double kCell = 0.25;  // metres between heights
constexpr double kCourant =
    0.2;  // c^2, in cells per step squared: stable below 0.5
constexpr double kDamping = 0.995;

class LogArrayDemo : public app::Demo {
  public:
    LogArrayDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 120;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        const double half = 0.5 * (kSize - 1) * kCell;
        Terrain::Define(scene, "/World/Pond")
            .SetCellSize(kCell)
            .SetOrigin({-half, half})
            .SetDisplayColor({0.25, 0.45, 0.7});
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.5, -0.4, 0.6}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(20.0)
            .SetTransform(LookAtTransform({0.0, -16.0, 11.0},
                                          {0.0, 0.0, 0.0},
                                          {0.0, 0.0, 1.0}));
        heights_.assign(kSize * kSize, 0.0);
        AddDrop(kSize / 2, kSize / 2);
        previous_ = heights_;
        WriteHeights();
        Widgets().Slider("Damping", 0.95, 1.0, kDamping,
                         [this](double value) { damping_ = value; });
        Widgets().Button(
            "Drop", [this] { AddDrop(kSize / 3, 2 * kSize / 3); });
    }

    void Step() override {
        std::vector<double> following(heights_.size(), 0.0);
        for (int r = 1; r < kSize - 1; ++r) {
            for (int c = 1; c < kSize - 1; ++c) {
                const double laplacian =
                    At(heights_, r + 1, c) + At(heights_, r - 1, c) +
                    At(heights_, r, c + 1) + At(heights_, r, c - 1) -
                    4.0 * At(heights_, r, c);
                following[Index(r, c)] =
                    (2.0 * At(heights_, r, c) - At(previous_, r, c) +
                     kCourant * laplacian) *
                    damping_;
            }
        }
        previous_ = std::move(heights_);
        heights_ = std::move(following);
        WriteHeights();
        LogArray("pond/heights",
                 Tensor::Copy<double>(heights_, {kSize, kSize}));
        std::vector<double> energy(kSize, 0.0);
        for (int r = 0; r < kSize; ++r) {
            for (int c = 0; c < kSize; ++c) {
                energy[static_cast<std::size_t>(r)] +=
                    At(heights_, r, c) * At(heights_, r, c);
            }
        }
        LogArray("pond/row energy",
                 Tensor::Copy<double>(energy, {kSize}));
    }

    void TestFinal() override {
        const Tensor heights = LoggedArray("pond/heights");
        const Tensor energy = LoggedArray("pond/row energy");
        Check(heights.Shape() ==
                      std::vector<std::int64_t>{kSize, kSize} &&
                  energy.Shape() == std::vector<std::int64_t>{kSize},
              "the field and the row energies should be logged");
        const auto* h = static_cast<const double*>(heights.Data());
        double largest = 0.0;
        for (int i = 0; i < kSize * kSize; ++i) {
            Check(std::isfinite(h[i]),
                  "the heights should stay finite");
            largest = std::max(largest, std::abs(h[i]));
        }
        // The drop spread: the centre fell, and ripples remain.
        Check(std::abs(h[Index(kSize / 2, kSize / 2)]) < 0.5,
              "the drop should have spread");
        Check(largest > 1e-3, "ripples should remain");
    }

  private:
    static std::size_t Index(int r, int c) {
        return static_cast<std::size_t>(r) * kSize +
               static_cast<std::size_t>(c);
    }
    static double At(const std::vector<double>& h, int r, int c) {
        return h[Index(r, c)];
    }

    /// Adds a Gaussian bump of a metre centred on (row, col), four
    /// cells across.
    void AddDrop(int row, int col) {
        constexpr double kWidth = 4.0;
        for (int r = 0; r < kSize; ++r) {
            for (int c = 0; c < kSize; ++c) {
                const double d2 = static_cast<double>(
                    (r - row) * (r - row) + (c - col) * (c - col));
                heights_[Index(r, c)] +=
                    std::exp(-d2 / (2.0 * kWidth * kWidth));
            }
        }
    }

    void WriteHeights() {
        std::vector<float> heights(heights_.begin(), heights_.end());
        Terrain(GetScene(), "/World/Pond")
            .SetHeights(Tensor::Copy<float>(heights, {kSize, kSize}));
    }

    std::vector<double> heights_;
    std::vector<double> previous_;
    double damping_ = kDamping;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    LogArrayDemo, "viewer", "log array",
    "Ripples on a pond, the height field logged as a live heatmap.");
