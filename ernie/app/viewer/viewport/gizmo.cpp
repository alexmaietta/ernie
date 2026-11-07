// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewport/gizmo.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include "ernie/scene/rotation.h"

namespace ernie::app {
namespace {

constexpr std::string_view kGizmoOverlay = "viewer/gizmo";
constexpr float kHandlePoints = 90;
constexpr float kReachPoints = 8;
constexpr double kOrthographicUnits = 0.1;
constexpr double kDegrees = 180 / 3.14159265358979323846;
constexpr Float4 kAxisColors[] = {
    {0.86f, 0.22f, 0.2f, 1}, {0.36f, 0.8f, 0.24f, 1}, {0.26f, 0.48f, 0.96f, 1}};
constexpr Float4 kHotColor = {1.0f, 0.82f, 0.18f, 1};
constexpr Float4 kViewColor = {0.9f, 0.9f, 0.9f, 0.85f};

Double3 Add(Double3 a, Double3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Double3 Sub(Double3 a, Double3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Double3 Mul(Double3 a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}

double Dot(Double3 a, Double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Double3 Cross(Double3 a, Double3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Double3 Normalize(Double3 a) {
    const double length = std::sqrt(Dot(a, a));
    return length > 0 ? Mul(a, 1 / length) : Double3{};
}

Double3 Wide(Float3 a) {
    return {a.x, a.y, a.z};
}

Float3 Narrow(Double3 a) {
    return {float(a.x), float(a.y), float(a.z)};
}

Double3 Origin(const WorldTransform& frame) {
    return {double(frame.high.x) + frame.low.x, double(frame.high.y) + frame.low.y,
            double(frame.high.z) + frame.low.z};
}

Quatd Rotation(const WorldTransform& frame) {
    return ernie::Normalize(
        FromAxes(Normalize(Wide(frame.x)), Normalize(Wide(frame.y)), Normalize(Wide(frame.z))));
}

Double3 Solve(const WorldTransform& frame, Double3 v) {
    const Double3 a = Wide(frame.x);
    const Double3 b = Wide(frame.y);
    const Double3 c = Wide(frame.z);
    const double det = Dot(a, Cross(b, c));
    if (std::abs(det) < 1e-12) {
        return v;
    }
    return {Dot(v, Cross(b, c)) / det, Dot(a, Cross(v, c)) / det, Dot(a, Cross(b, v)) / det};
}

float SegmentDistance(ImVec2 point, ImVec2 a, ImVec2 b) {
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float squared = dx * dx + dy * dy;
    const float t =
        squared > 0
            ? std::clamp(((point.x - a.x) * dx + (point.y - a.y) * dy) / squared, 0.0f, 1.0f)
            : 0.0f;
    const float ex = point.x - (a.x + dx * t);
    const float ey = point.y - (a.y + dy * t);
    return std::sqrt(ex * ex + ey * ey);
}

bool Inside(ImVec2 point, const std::array<ImVec2, 4>& corners) {
    bool inside = false;
    for (size_t at = 0, before = 3; at < 4; before = at++) {
        const ImVec2 a = corners[at];
        const ImVec2 b = corners[before];
        if ((a.y > point.y) != (b.y > point.y) &&
            point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x) {
            inside = !inside;
        }
    }
    return inside;
}

double Snapped(double value, double step, bool snapping) {
    return snapping && step > 0 ? std::round(value / step) * step : value;
}

double ScreenAngle(ImVec2 mouse, ImVec2 centre) {
    return std::atan2(double(mouse.y - centre.y), double(mouse.x - centre.x));
}

}

std::optional<ImVec2> ToScreen(const GizmoCamera& camera, Double3 point) {
    const WorldTransform& frame = camera.frame;
    const Double3 offset = Sub(point, Origin(frame));
    const auto along = [&](Float3 axis) {
        const Double3 wide = Wide(axis);
        const double squared = Dot(wide, wide);
        return squared > 0 ? Dot(offset, wide) / squared : 0;
    };
    const double x = along(frame.x);
    const double y = along(frame.y);
    const double z = along(frame.z);
    double filmX = x / kOrthographicUnits;
    double filmY = y / kOrthographicUnits;
    if (!camera.orthographic) {
        if (z > -1e-6) {
            return std::nullopt;
        }
        filmX = x * camera.focal / -z;
        filmY = y * camera.focal / -z;
    }
    const double u = ((filmX - camera.right) / (0.5 * camera.width) + 1) / 2;
    const double v = (1 - (filmY - camera.up) / (0.5 * camera.height)) / 2;
    return ImVec2{camera.min.x + float(u) * camera.size.x, camera.min.y + float(v) * camera.size.y};
}

GizmoRay CameraRay(const GizmoCamera& camera, double u, double v) {
    const WorldTransform& frame = camera.frame;
    const double filmX = (2 * u - 1) * 0.5 * camera.width + camera.right;
    const double filmY = (1 - 2 * v) * 0.5 * camera.height + camera.up;
    const auto toWorld = [&](double x, double y, double z) {
        return Add(Add(Mul(Wide(frame.x), x), Mul(Wide(frame.y), y)), Mul(Wide(frame.z), z));
    };
    Double3 origin = Origin(frame);
    Double3 along;
    if (camera.orthographic) {
        origin = Add(origin, toWorld(filmX * kOrthographicUnits, filmY * kOrthographicUnits, 0));
        along = toWorld(0, 0, -1);
    } else {
        along = toWorld(filmX / camera.focal, filmY / camera.focal, -1);
    }
    return {origin, along};
}

Gizmo::Gizmo(Ref<Scene> scene, std::shared_ptr<Renderer> renderer, Editor* editor)
    : m_scene(std::move(scene)), m_renderer(std::move(renderer)), m_editor(editor) {}

Gizmo::~Gizmo() = default;

void Gizmo::Update(const Selection& selection, const std::optional<GizmoCamera>& camera) {
    m_camera = camera;
    std::string path;
    std::string parent;
    if (mode != GizmoMode::None && selection.paths.size() == 1) {
        if (const std::optional<Prim> prim = m_scene->Find(selection.paths.front());
            prim && prim->IsA(SchemaId::Xformable) && !prim->IsSession()) {
            path = prim->Path();
            if (const std::optional<Prim> above = prim->Parent();
                above && above->IsA(SchemaId::Xformable)) {
                parent = above->Path();
            }
        }
    }
    if (path != m_path || parent != m_parent) {
        m_path = std::move(path);
        m_parent = std::move(parent);
        m_reads.clear();
        m_pose.reset();
        m_world.reset();
        m_parentWorld.reset();
        m_hovered = Handle::None;
        m_drag.reset();
    }
    if (!m_drag) {
        Read();
    }
    Draw();
}

void Gizmo::Read() {
    if (m_path.empty()) {
        return;
    }
    if (!m_reads.empty()) {
        if (!std::ranges::all_of(m_reads,
                                 [](const device::Download& read) { return read.Ready(); })) {
            return;
        }
        const auto read = [&]<class T>(size_t index, std::optional<T>& value) {
            if (index < m_reads.size() && m_reads[index].Bytes().size() >= sizeof(T)) {
                T loaded;
                std::memcpy(&loaded, m_reads[index].Bytes().data(), sizeof(T));
                value = loaded;
            }
        };
        std::optional<Double3> translate;
        std::optional<Quatd> orient;
        std::optional<Float3> scale;
        read(0, translate);
        read(1, orient);
        read(2, scale);
        read(3, m_world);
        if (!m_parent.empty()) {
            read(4, m_parentWorld);
        }
        if (translate && orient && scale) {
            m_pose = Pose{*translate, *orient, *scale};
        }
        m_reads.clear();
    }
    const View prim = m_scene->Select(m_path);
    for (const ColumnRef column : {Xformable::kTranslate.Column(), Xformable::kOrient.Column(),
                                   Xformable::kScale.Column()}) {
        m_reads.push_back(prim.Get(column).ReadHostAsync());
    }
    m_reads.push_back(m_scene->ReadWorldTransformsAsync(std::vector<std::string>{m_path}));
    if (!m_parent.empty()) {
        m_reads.push_back(m_scene->ReadWorldTransformsAsync(std::vector<std::string>{m_parent}));
    }
}

std::array<Double3, 3> Gizmo::Axes(bool forScale) const {
    if ((local || forScale) && m_world) {
        return {Normalize(Wide(m_world->x)), Normalize(Wide(m_world->y)),
                Normalize(Wide(m_world->z))};
    }
    return {Double3{1, 0, 0}, Double3{0, 1, 0}, Double3{0, 0, 1}};
}

double Gizmo::Units() const {
    const GizmoCamera& camera = *m_camera;
    const double points = kHandlePoints * ImGui::GetStyle().FontScaleDpi;
    if (camera.orthographic) {
        return points * camera.width * kOrthographicUnits / std::max(camera.size.x, 1.0f);
    }
    const Double3 pivot = m_drag ? m_drag->pivot : Origin(*m_world);
    const Double3 forward = Normalize(Mul(Wide(camera.frame.z), -1));
    const double depth = std::max(Dot(Sub(pivot, Origin(camera.frame)), forward), 1e-3);
    return points * depth * camera.width / camera.focal / std::max(camera.size.x, 1.0f);
}

Gizmo::Handle Gizmo::Hit(ImVec2 mouse) const {
    if (!m_camera || !m_world || mode == GizmoMode::None) {
        return Handle::None;
    }
    const GizmoCamera& camera = *m_camera;
    const Double3 pivot = Origin(*m_world);
    const double length = Units();
    const std::array<Double3, 3> axes = Axes(mode == GizmoMode::Scale);
    const std::optional<ImVec2> centre = ToScreen(camera, pivot);
    if (!centre) {
        return Handle::None;
    }
    const float reach = kReachPoints * ImGui::GetStyle().FontScaleDpi;
    if (mode != GizmoMode::Rotate && SegmentDistance(mouse, *centre, *centre) < 1.25f * reach) {
        return Handle::Centre;
    }
    Handle best = Handle::None;
    float nearest = reach;
    const auto consider = [&](Handle handle, float distance) {
        if (distance < nearest) {
            nearest = distance;
            best = handle;
        }
    };
    constexpr Handle kAxes[] = {Handle::X, Handle::Y, Handle::Z};
    constexpr Handle kPlanes[] = {Handle::YZ, Handle::XZ, Handle::XY};
    for (size_t axis = 0; axis < 3; ++axis) {
        if (mode == GizmoMode::Rotate) {
            const Double3 u = axes[(axis + 1) % 3];
            const Double3 v = axes[(axis + 2) % 3];
            std::optional<ImVec2> previous;
            for (int step = 0; step <= 64; ++step) {
                const double angle = step * 2 * 3.14159265358979323846 / 64;
                const std::optional<ImVec2> point = ToScreen(
                    camera, Add(pivot, Mul(Add(Mul(u, std::cos(angle)), Mul(v, std::sin(angle))),
                                           0.85 * length)));
                if (point && previous) {
                    consider(kAxes[axis], SegmentDistance(mouse, *previous, *point));
                }
                previous = point;
            }
            continue;
        }
        if (const std::optional<ImVec2> tip =
                ToScreen(camera, Add(pivot, Mul(axes[axis], length)))) {
            consider(kAxes[axis], SegmentDistance(mouse, *centre, *tip));
        }
        if (mode == GizmoMode::Translate) {
            const Double3 u = axes[(axis + 1) % 3];
            const Double3 v = axes[(axis + 2) % 3];
            std::array<ImVec2, 4> corners;
            bool shown = true;
            const double near = 0.2 * length;
            const double far = 0.4 * length;
            const Double3 points[4] = {Add(pivot, Add(Mul(u, near), Mul(v, near))),
                                       Add(pivot, Add(Mul(u, far), Mul(v, near))),
                                       Add(pivot, Add(Mul(u, far), Mul(v, far))),
                                       Add(pivot, Add(Mul(u, near), Mul(v, far)))};
            for (size_t corner = 0; corner < 4; ++corner) {
                const std::optional<ImVec2> point = ToScreen(camera, points[corner]);
                shown = shown && point.has_value();
                corners[corner] = point.value_or(ImVec2{});
            }
            if (shown && Inside(mouse, corners)) {
                consider(kPlanes[axis], 0);
            }
        }
    }
    if (mode == GizmoMode::Rotate && best == Handle::None) {
        const float radius = 1.0f * kHandlePoints * ImGui::GetStyle().FontScaleDpi;
        const float distance = SegmentDistance(mouse, *centre, *centre);
        consider(Handle::View, std::abs(distance - radius));
    }
    return best;
}

bool Gizmo::Input(ImVec2 mouse, bool pressed, bool held) {
    if (m_drag) {
        if (held) {
            Move(mouse);
            return true;
        }
        const Pose before = m_drag->before;
        m_drag.reset();
        if (m_pose && m_editor) {
            const char* name = mode == GizmoMode::Translate ? "Move"
                               : mode == GizmoMode::Rotate  ? "Rotate"
                                                            : "Scale";
            m_editor->SetPose(name, m_path, before.translate, before.orient, before.scale,
                              m_pose->translate, m_pose->orient, m_pose->scale);
        }
        m_reads.clear();
        return true;
    }
    m_hovered = Hit(mouse);
    if (!pressed || m_hovered == Handle::None || !m_pose || !m_world || !m_camera) {
        return false;
    }
    Drag drag{.handle = m_hovered, .mouse = mouse, .before = *m_pose, .pivot = Origin(*m_world)};
    m_drag = drag;
    Move(mouse);
    return true;
}

void Gizmo::Move(ImVec2 mouse) {
    Drag& drag = *m_drag;
    const GizmoCamera& camera = *m_camera;
    const double u = (mouse.x - camera.min.x) / std::max(camera.size.x, 1.0f);
    const double v = (mouse.y - camera.min.y) / std::max(camera.size.y, 1.0f);
    const GizmoRay ray = CameraRay(camera, u, v);
    const std::array<Double3, 3> axes = Axes(mode == GizmoMode::Scale);
    const bool snapping = snap != ImGui::GetIO().KeyCtrl;
    const bool first = !drag.started;
    drag.started = true;
    const auto index = [](Handle handle) {
        switch (handle) {
        case Handle::X:
        case Handle::YZ:
            return 0;
        case Handle::Y:
        case Handle::XZ:
            return 1;
        default:
            return 2;
        }
    };
    const Double3 forward = Normalize(Mul(Wide(camera.frame.z), -1));
    Pose pose = drag.before;
    const std::optional<ImVec2> centre = ToScreen(camera, drag.pivot);

    if (mode == GizmoMode::Translate) {
        Double3 delta{};
        if (drag.handle == Handle::X || drag.handle == Handle::Y || drag.handle == Handle::Z) {
            const Double3 axis = axes[index(drag.handle)];
            const Double3 d = Normalize(ray.direction);
            const Double3 w = Sub(drag.pivot, ray.origin);
            const double b = Dot(axis, d);
            const double denominator = 1 - b * b;
            if (denominator < 1e-6) {
                return;
            }
            const double t = (b * Dot(d, w) - Dot(axis, w)) / denominator;
            if (first) {
                drag.along = t;
            }
            delta = Mul(axis, Snapped(t - drag.along, moveStep, snapping));
        } else {
            const Double3 normal =
                drag.handle == Handle::Centre ? forward : axes[index(drag.handle)];
            const double facing = Dot(ray.direction, normal);
            if (std::abs(facing) < 1e-9) {
                return;
            }
            const double s = Dot(Sub(drag.pivot, ray.origin), normal) / facing;
            const Double3 hit = Add(ray.origin, Mul(ray.direction, s));
            if (first) {
                drag.grabbed = hit;
            }
            delta = Sub(hit, drag.grabbed);
            if (snapping) {
                const std::array<Double3, 3> grid =
                    drag.handle == Handle::Centre
                        ? std::array<Double3, 3>{Double3{1, 0, 0}, Double3{0, 1, 0},
                                                 Double3{0, 0, 1}}
                        : axes;
                Double3 snapped{};
                for (size_t axis = 0; axis < 3; ++axis) {
                    if (drag.handle != Handle::Centre && axis == size_t(index(drag.handle))) {
                        continue;
                    }
                    snapped = Add(snapped,
                                  Mul(grid[axis], Snapped(Dot(delta, grid[axis]), moveStep, true)));
                }
                delta = snapped;
            }
        }
        const Double3 moved = m_parentWorld ? Solve(*m_parentWorld, delta) : delta;
        pose.translate = Add(drag.before.translate, moved);
        if (m_world) {
            m_world->high = Narrow(Add(drag.pivot, delta));
            m_world->low = {0, 0, 0};
        }
    } else if (mode == GizmoMode::Rotate) {
        if (!centre) {
            return;
        }
        const double angle = ScreenAngle(mouse, *centre);
        if (first) {
            drag.angle = angle;
        }
        const Double3 axis =
            drag.handle == Handle::View ? Mul(forward, -1) : axes[index(drag.handle)];
        double turned = angle - drag.angle;
        turned = std::remainder(turned, 2 * 3.14159265358979323846);
        turned = Snapped(turned, turnStep / kDegrees, snapping);
        const double sign = Dot(axis, forward) < 0 ? -1 : 1;
        const Quatd spin = FromAxisAngle(axis, sign * turned * kDegrees);
        const Quatd above = m_parentWorld ? Rotation(*m_parentWorld) : Quatd{};
        pose.orient = ernie::Normalize(
            Multiply(Multiply(Multiply(Conjugate(above), spin), above), drag.before.orient));
    } else {
        if (!centre) {
            return;
        }
        const float dx = mouse.x - drag.mouse.x;
        const float dy = mouse.y - drag.mouse.y;
        if (drag.handle == Handle::Centre) {
            const double factor =
                std::max(1 + Snapped((dx - dy) / (100.0 * ImGui::GetStyle().FontScaleDpi),
                                     scaleStep, snapping),
                         0.01);
            pose.scale = Narrow(Mul(Wide(drag.before.scale), factor));
        } else {
            const size_t axis = index(drag.handle);
            const std::optional<ImVec2> tip =
                ToScreen(camera, Add(drag.pivot, Mul(axes[axis], Units())));
            if (!tip) {
                return;
            }
            const float sx = tip->x - centre->x;
            const float sy = tip->y - centre->y;
            const float squared = std::max(sx * sx + sy * sy, 1.0f);
            const double factor = std::max(
                1 + Snapped(double(dx * sx + dy * sy) / squared, scaleStep, snapping), 0.01);
            Double3 scale = Wide(drag.before.scale);
            (axis == 0 ? scale.x : axis == 1 ? scale.y : scale.z) *= factor;
            pose.scale = Narrow(scale);
        }
    }
    Write(pose);
}

void Gizmo::Write(const Pose& pose) {
    const View prim = m_scene->Select(m_path);
    prim.Get(Xformable::kTranslate).Set(pose.translate);
    prim.Get(Xformable::kOrient).Set(pose.orient);
    prim.Get(Xformable::kScale).Set(pose.scale);
    m_pose = pose;
}

void Gizmo::Draw() {
    if (!m_renderer) {
        return;
    }
    if (m_path.empty() || !m_world || !m_camera || mode == GizmoMode::None) {
        if (!m_drawn.empty()) {
            m_drawn.clear();
            m_renderer->RemoveOverlay(kGizmoOverlay);
        }
        return;
    }
    const GizmoCamera& camera = *m_camera;
    const Double3 pivot = Origin(*m_world);
    const double length = Units();
    const std::array<Double3, 3> axes = Axes(mode == GizmoMode::Scale);
    const Handle hot = m_drag ? m_drag->handle : m_hovered;
    std::vector<float> key = {float(pivot.x), float(pivot.y), float(pivot.z), float(length),
                              float(mode),    float(hot),     float(local)};
    for (const Double3& axis : axes) {
        key.insert(key.end(), {float(axis.x), float(axis.y), float(axis.z)});
    }
    for (const Float3 axis : {camera.frame.x, camera.frame.y, camera.frame.z}) {
        key.insert(key.end(), {axis.x, axis.y, axis.z});
    }
    if (key == m_drawn) {
        return;
    }
    m_drawn = key;

    const Float3 centre = Narrow(pivot);
    const auto style = [&](Float4 color, bool hotHandle, float width = 2.5f) {
        return OverlayStyle{.color = hotHandle ? kHotColor : color,
                            .width = width,
                            .size = 10,
                            .marker = MarkerShape::Square,
                            .depth = OverlayDepth::Always,
                            .pickable = false};
    };
    constexpr Handle kAxes[] = {Handle::X, Handle::Y, Handle::Z};
    constexpr Handle kPlanes[] = {Handle::YZ, Handle::XZ, Handle::XY};
    Overlay overlay;
    for (size_t axis = 0; axis < 3; ++axis) {
        const Double3 tip = Add(pivot, Mul(axes[axis], length));
        const bool isHot = hot == kAxes[axis];
        switch (mode) {
        case GizmoMode::Translate: {
            overlay.Arrow(centre, Narrow(tip), style(kAxisColors[axis], isHot),
                          float(0.22 * length));
            const Double3 u = axes[(axis + 1) % 3];
            const Double3 v = axes[(axis + 2) % 3];
            const double near = 0.2 * length;
            const double far = 0.4 * length;
            const std::vector<Float3> square = {Narrow(Add(pivot, Add(Mul(u, near), Mul(v, near)))),
                                                Narrow(Add(pivot, Add(Mul(u, far), Mul(v, near)))),
                                                Narrow(Add(pivot, Add(Mul(u, far), Mul(v, far)))),
                                                Narrow(Add(pivot, Add(Mul(u, near), Mul(v, far))))};
            Float4 color = hot == kPlanes[axis] ? kHotColor : kAxisColors[axis];
            color.w = hot == kPlanes[axis] ? 0.7f : 0.35f;
            overlay.Polygon(square, style(color, false));
            break;
        }
        case GizmoMode::Rotate:
            overlay.Circle(centre, Narrow(axes[axis]), float(0.85 * length),
                           style(kAxisColors[axis], isHot));
            break;
        case GizmoMode::Scale:
            overlay.Line(centre, Narrow(tip), style(kAxisColors[axis], isHot));
            overlay.Marker(Narrow(tip), style(kAxisColors[axis], isHot));
            break;
        case GizmoMode::None:
            break;
        }
    }
    if (mode == GizmoMode::Rotate) {
        const Double3 toward = Normalize(Wide(camera.frame.z));
        overlay.Circle(centre, Narrow(toward), float(length),
                       style(kViewColor, hot == Handle::View, 1.5f));
    } else {
        OverlayStyle middle = style(kViewColor, hot == Handle::Centre);
        middle.size = 12;
        overlay.Marker(centre, middle);
    }
    m_renderer->SetOverlay(kGizmoOverlay, overlay);
}

}
