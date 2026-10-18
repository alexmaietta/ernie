// SPDX-License-Identifier: BSD-3-Clause
/// @file extents.cpp
/// @brief Scene updates at 100,000 prims: the first, one edit's, and one inherited edit's.
#include "benchmarks/bench.h"
#include "ernie/scene/schema/geom.h"

#include <memory>
#include <string>

namespace ernie::bench {
namespace {

constexpr int kSpheres = 100'000;

std::unique_ptr<Scene> MakeSpheres() {
    auto scene = std::make_unique<Scene>();
    for (int i = 0; i < kSpheres; ++i) {
        scene->Define("/World/Ball_" + std::to_string(i), "Sphere");
    }
    return scene;
}

const bool kFirstUpdate = Register("scene.update.first_100k", [] {
    std::unique_ptr<Scene> scene;
    return Timing{Fastest([&] { scene = MakeSpheres(); }, [&] { scene->Update(); }), 10.0};
});

const bool kOneEdit = Register("scene.update.after_one_edit_of_100k", [] {
    std::unique_ptr<Scene> scene = MakeSpheres();
    scene->Update();
    const PrimId ball = *scene->Find("/World/Ball_54321");
    double radius = 1.0;
    const double ms = Fastest([&] { scene->Set(ball, Sphere::kRadius, radius += 1.0); },
                              [&] { scene->Update(); });
    return Timing{ms, 1.0};
});

const bool kInheritedEdit = Register("scene.update.after_hiding_the_parent_of_100k", [] {
    std::unique_ptr<Scene> scene = MakeSpheres();
    scene->Define("/World", "Xform");
    scene->Update();
    const PrimId world = *scene->Find("/World");
    bool hidden = false;
    const double ms = Fastest(
        [&] {
            hidden = !hidden;
            scene->Set(world, Imageable::kVisibility, hidden ? "invisible" : "inherited");
        },
        [&] { scene->Update(); });
    return Timing{ms, 10.0};
});

}  // namespace
}  // namespace ernie::bench
