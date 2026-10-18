// SPDX-License-Identifier: BSD-3-Clause
/// @file stage.cpp
/// @brief Checks of a scene's stage metadata.
#include "ernie/scene/stage.h"

#include "ernie/scene/path.h"

#include <cmath>
#include <stdexcept>

namespace ernie {
namespace {

void RequireUnit(double perUnit, const char* name) {
    if (!(std::isfinite(perUnit) && perUnit > 0.0)) {
        throw std::invalid_argument(std::string(name) + " must be positive and finite");
    }
}

}  // namespace

void Validate(const Stage& stage) {
    RequireUnit(stage.metersPerUnit, "metersPerUnit");
    RequireUnit(stage.kilogramsPerUnit, "kilogramsPerUnit");
    if (!stage.defaultPrim.empty() && !Path::IsValidName(stage.defaultPrim)) {
        throw std::invalid_argument("defaultPrim '" + stage.defaultPrim + "' is not a prim name");
    }
}

}  // namespace ernie
