// SPDX-License-Identifier: BSD-3-Clause
/// @file navigation.cpp
/// @brief Mouse drags and the wheel as camera moves.
#include "ernie/app/navigation.h"

#include <imgui.h>

namespace ernie::app {
namespace {

/// Radians turned per pixel dragged.
constexpr double kOrbitRate = 0.005;

bool AnyButtonDown() {
    return ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
           ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
           ImGui::IsMouseDown(ImGuiMouseButton_Middle);
}

}  // namespace

bool Navigation::Update(OrbitCamera& camera, bool hovered, float height) {
    const ImGuiIO& io = ImGui::GetIO();
    if (!AnyButtonDown()) {
        dragging_ = false;
    } else if (hovered && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                           ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
                           ImGui::IsMouseClicked(ImGuiMouseButton_Middle))) {
        dragging_ = true;
    }
    const double dx = io.MouseDelta.x;
    const double dy = io.MouseDelta.y;
    bool moved = false;
    if (dragging_ && (dx != 0.0 || dy != 0.0)) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            camera.Orbit(-dx * kOrbitRate, dy * kOrbitRate);
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            camera.Pan(dx, dy, height);
        } else {
            camera.PanVertical(dx, dy, height);
        }
        moved = true;
    }
    if (hovered && io.MouseWheel != 0.0f) {
        camera.Zoom(io.MouseWheel);
        moved = true;
    }
    return moved;
}

}  // namespace ernie::app
