// SPDX-License-Identifier: BSD-3-Clause
/// @file extents.cpp
/// @brief Cached extents at 100,000 prims: the first update, and one edit's.
#include "benchmarks/bench.h"
#include "ernie/scene/schemas/geom.h"

#include <memory>
#include <string>
#include <vector>

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

const bool kFirstUpdate = Register("scene.extents.first_update_100k", [] {
    std::unique_ptr<Scene> scene;
    return Timing{Fastest([&] { scene = MakeSpheres(); }, [&] { scene->UpdateExtents(); }), 10.0};
});

const bool kOneEdit = Register("scene.extents.update_after_one_edit_of_100k", [] {
    std::unique_ptr<Scene> scene = MakeSpheres();
    scene->UpdateExtents();
    const PrimId ball = *scene->Find("/World/Ball_54321");
    double radius = 1.0;
    const double ms = Fastest([&] { scene->Set(ball, Sphere::kRadius, radius += 1.0); },
                              [&] { scene->UpdateExtents(); });
    return Timing{ms, 1.0};
});

}  // namespace
}  // namespace ernie::bench
