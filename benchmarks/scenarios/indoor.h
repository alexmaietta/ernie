// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Procedural floor plans for the indoor scenarios: a grid of rooms
// with textured floors and walls, doorways between them.

#include "../bench.h"

#include <vector>

namespace ernie::bench::indoor {

constexpr float kWallHeight = 2.6f;

struct Room {
    glm::vec2 min;
    glm::vec2 max;

    glm::vec2 Centre() const {
        return (min + max) / 2.0f;
    }
};

/// A floor plan: 2 or 3 columns of 2 rows of rooms, 3 to 5 m each,
/// and its walls as segments, doorways left out.
struct Layout {
    std::vector<Room> rooms;
    std::vector<std::pair<glm::vec2, glm::vec2>> walls;
};

Layout MakeLayout(std::uint64_t seed);

/// The first of the shell's prototypes: the floors, one per floor
/// material, then the wall.
struct Shell {
    int firstFloor = 0;
    int floors = 0;
    int wall = 0;
};

/// Appends the shell's prototypes, defined under `root`, to
/// `prototypes`.
Shell DefineShell(Scene& scene, const Path& root,
                  std::vector<Path>& prototypes);

/// Places `layout`'s floors and walls in `world`.
void PlaceShell(const Layout& layout, const Shell& shell, int world,
                std::vector<Placement>& placements);

/// A random point in `room`, `margin` from its walls, at height `z`.
glm::vec3 PointIn(const Room& room, float margin, float z,
                  Random& random);

/// The pose of view `view` of a world of `layout` at `step`: in one
/// of its rooms at eye height, looking around.
glm::dmat4 ViewIn(const Layout& layout, int world, int view,
                  int step);

}  // namespace ernie::bench::indoor
