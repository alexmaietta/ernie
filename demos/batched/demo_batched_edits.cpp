// SPDX-License-Identifier: BSD-3-Clause
// Conway's Game of Life on 10,000 cubes: each generation recolours
// only the cells that flipped, one prim at a time, and lifts one row
// of them (its parent's transform), so a frame syncs only what
// changed.

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/xform.h"
#include "util.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr int kSide = 100;  // cells along each side
constexpr double kSpacing = 1.1;  // m between cells
const glm::dvec3 kAlive{0.2, 0.85, 0.3};
const glm::dvec3 kDead{0.12, 0.12, 0.14};

Path RowPath(int r) {
    return Path("/World/Board/Row" + std::to_string(r));
}

Path CellPath(int r, int c) {
    return RowPath(r).Child("Cell" + std::to_string(c));
}

class EditsDemo : public app::Demo {
  public:
    EditsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 60;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        DistantLight::Define(scene, "/World/Sun")
            .SetTransform(DistantLightTransform({0.3, -0.4, 0.9}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(18.0)
            .SetTransform(LookAtTransform({0.0, -95.0, 90.0},
                                          {0.0, 0.0, 0.0},
                                          {0.0, 0.0, 1.0}));
        // A fixed random soup, a third alive (PCG-style generator).
        std::uint64_t state = 0x853c49e6748fea9bull;
        alive_.resize(kSide * kSide);
        const double half = 0.5 * (kSide - 1) * kSpacing;
        for (int r = 0; r < kSide; ++r) {
            Xform::Define(scene, RowPath(r));
            SetXform(scene, RowPath(r), {.translate = {0, 0, 0}});
            for (int c = 0; c < kSide; ++c) {
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                const bool alive = (state >> 33) % 3 == 0;
                alive_[Index(r, c)] = alive;
                Cube::Define(scene, CellPath(r, c))
                    .SetSize(1.0)
                    .SetDisplayColor(alive ? kAlive : kDead);
                SetXform(scene, CellPath(r, c),
                         {.translate = {c * kSpacing - half,
                                        r * kSpacing - half, 0.0}});
            }
        }
    }

    void Step() override {
        Scene& scene = GetScene();
        // The next generation on a torus; only the flipped cells are
        // written.
        std::vector<bool> next(alive_.size());
        for (int r = 0; r < kSide; ++r) {
            for (int c = 0; c < kSide; ++c) {
                int around = 0;
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (dr != 0 || dc != 0) {
                            around += alive_[Index(r + dr, c + dc)] ? 1 : 0;
                        }
                    }
                }
                const bool was = alive_[Index(r, c)];
                const bool is = around == 3 || (was && around == 2);
                next[Index(r, c)] = is;
                if (is != was) {
                    Cube(scene, CellPath(r, c))
                        .SetDisplayColor(is ? kAlive : kDead);
                }
            }
        }
        alive_ = std::move(next);
        // A scanning row: the last one down, the next one up.
        SetXform(scene, RowPath(row_), {.translate = {0, 0, 0}});
        row_ = (row_ + 1) % kSide;
        SetXform(scene, RowPath(row_), {.translate = {0, 0, 1.0}});
    }

    void TestPostStep() override {
        const Scene& scene = GetScene();
        for (int r = 0; r < kSide; ++r) {
            for (int c = 0; c < kSide; ++c) {
                const glm::dvec3 colour =
                    Cube(scene, CellPath(r, c)).GetDisplayColor();
                Check(colour == (alive_[Index(r, c)] ? kAlive : kDead),
                      "every cell shows its state");
            }
        }
        Check(scene.WorldTransform(CellPath(row_, 0))[3][2] == 1.0,
              "the scanning row is lifted");
    }

  private:
    static std::size_t Index(int r, int c) {
        r = (r + kSide) % kSide;
        c = (c + kSide) % kSide;
        return static_cast<std::size_t>(r * kSide + c);
    }

    std::vector<bool> alive_;
    int row_ = 0;
};

}  // namespace

ERNIE_REGISTER_DEMO(EditsDemo, "batched", "edits",
                    "The Game of Life on 10,000 cubes: only the cells "
                    "that flip are written, one prim at a time.");
