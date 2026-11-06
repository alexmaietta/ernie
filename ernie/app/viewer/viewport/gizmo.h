// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <imgui.h>

#include "ernie/app/viewer/shell/editor.h"
#include "ernie/app/viewer/shell/selection.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"

namespace ernie::app {

enum class GizmoMode : uint8_t { None, Translate, Rotate, Scale };

struct GizmoCamera {
    WorldTransform frame{};
    float focal = 0;
    float width = 0;
    float height = 0;
    float right = 0;
    float up = 0;
    bool orthographic = false;
    ImVec2 min;
    ImVec2 size;
};

struct GizmoRay {
    Double3 origin;
    Double3 direction;
};

GizmoRay CameraRay(const GizmoCamera& camera, double u, double v);
std::optional<ImVec2> ToScreen(const GizmoCamera& camera, Double3 point);

class Gizmo {
  public:
    Gizmo(Ref<Scene> scene, std::shared_ptr<Renderer> renderer, Editor* editor);
    ~Gizmo();
    Gizmo(const Gizmo&) = delete;
    Gizmo& operator=(const Gizmo&) = delete;

    void Update(const Selection& selection, const std::optional<GizmoCamera>& camera);
    bool Input(ImVec2 mouse, bool pressed, bool held);

    bool Active() const {
        return m_drag.has_value();
    }

    bool Hovered() const {
        return m_hovered != Handle::None;
    }

    GizmoMode mode = GizmoMode::Translate;
    bool local = false;
    bool snap = false;
    float moveStep = 0.1f;
    float turnStep = 15;
    float scaleStep = 0.1f;

  private:
    enum class Handle : uint8_t { None, X, Y, Z, YZ, XZ, XY, View, Centre };

    struct Pose {
        Double3 translate;
        Quatd orient;
        Float3 scale = {1, 1, 1};
    };

    struct Drag {
        Handle handle = Handle::None;
        ImVec2 mouse;
        Pose before;
        Double3 pivot;
        Double3 grabbed;
        double along = 0;
        double angle = 0;
        bool started = false;
    };

    void Read();
    std::array<Double3, 3> Axes(bool forScale) const;
    double Units() const;
    Handle Hit(ImVec2 mouse) const;
    void Move(ImVec2 mouse);
    void Write(const Pose& pose);
    void Draw();

    Ref<Scene> m_scene;
    std::shared_ptr<Renderer> m_renderer;
    Editor* m_editor = nullptr;
    std::string m_path;
    std::string m_parent;
    std::optional<GizmoCamera> m_camera;
    std::vector<device::Download> m_reads;
    std::optional<Pose> m_pose;
    std::optional<WorldTransform> m_world;
    std::optional<WorldTransform> m_parentWorld;
    Handle m_hovered = Handle::None;
    std::optional<Drag> m_drag;
    std::vector<float> m_drawn;
};

}
