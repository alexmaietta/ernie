// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/prim_icons.h"

#include "ernie/scene/tokens.h"

#include <imgui.h>

#include <cmath>
#include <numbers>

namespace ernie::app {

namespace {

using namespace tokens;

ImU32 Rgb(int r, int g, int b, int a = 255) {
    return IM_COL32(r, g, b, a);
}

/// A point of the icon's square: (u, v) in [0, 1] from its top-left corner.
struct Frame {
    ImVec2 corner;
    float size;

    ImVec2 operator()(float u, float v) const {
        return {corner.x + u * size, corner.y + v * size};
    }

    float operator()(float length) const {
        return length * size;
    }
};

void Tripod(ImDrawList* draw, const Frame& at) {
    const float width = std::max(1.2f, at(0.09f));
    const ImVec2 centre = at(0.5f, 0.58f);
    draw->AddLine(centre, at(0.5f, 0.1f), Rgb(110, 200, 90), width);    // y, green
    draw->AddLine(centre, at(0.1f, 0.86f), Rgb(225, 80, 70), width);    // x, red
    draw->AddLine(centre, at(0.9f, 0.86f), Rgb(80, 140, 235), width);   // z, blue
}

void Folder(ImDrawList* draw, const Frame& at) {
    const ImU32 back = Rgb(74, 85, 88);
    const ImU32 front = Rgb(96, 108, 112);
    draw->AddRectFilled(at(0.08f, 0.2f), at(0.45f, 0.38f), back, at(0.06f));
    draw->AddRectFilled(at(0.08f, 0.3f), at(0.92f, 0.82f), front, at(0.08f));
}

/// A cube in three-quarter view, its faces lit from above-left.
void Cube(ImDrawList* draw, const Frame& at, float scale = 1.0f, ImVec2 offset = {0.0f, 0.0f}) {
    const auto p = [&](float u, float v) {
        return at(offset.x + 0.5f + (u - 0.5f) * scale, offset.y + 0.5f + (v - 0.5f) * scale);
    };
    const ImVec2 top[] = {p(0.5f, 0.1f), p(0.9f, 0.3f), p(0.5f, 0.5f), p(0.1f, 0.3f)};
    const ImVec2 left[] = {p(0.1f, 0.3f), p(0.5f, 0.5f), p(0.5f, 0.92f), p(0.1f, 0.72f)};
    const ImVec2 right[] = {p(0.5f, 0.5f), p(0.9f, 0.3f), p(0.9f, 0.72f), p(0.5f, 0.92f)};
    draw->AddConvexPolyFilled(top, 4, Rgb(150, 160, 164));
    draw->AddConvexPolyFilled(left, 4, Rgb(104, 114, 118));
    draw->AddConvexPolyFilled(right, 4, Rgb(74, 85, 88));
}

/// A shaded ball: a base colour and a highlight up and to the left.
void Ball(ImDrawList* draw, const Frame& at, ImU32 base, ImU32 middle, ImU32 highlight) {
    const ImVec2 centre = at(0.5f, 0.52f);
    draw->AddCircleFilled(centre, at(0.4f), base);
    draw->AddCircleFilled(at(0.44f, 0.44f), at(0.28f), middle);
    draw->AddCircleFilled(at(0.38f, 0.36f), at(0.11f), highlight);
}

void Cylinder(ImDrawList* draw, const Frame& at, bool cone) {
    const ImU32 side = Rgb(96, 108, 112);
    const ImU32 top = Rgb(150, 160, 164);
    if (cone) {
        draw->AddTriangleFilled(at(0.5f, 0.08f), at(0.88f, 0.8f), at(0.12f, 0.8f), side);
        draw->AddEllipseFilled(at(0.5f, 0.8f), ImVec2(at(0.38f), at(0.12f)), Rgb(74, 85, 88));
        return;
    }
    draw->AddRectFilled(at(0.15f, 0.26f), at(0.85f, 0.8f), side);
    draw->AddEllipseFilled(at(0.5f, 0.8f), ImVec2(at(0.35f), at(0.12f)), side);
    draw->AddEllipseFilled(at(0.5f, 0.26f), ImVec2(at(0.35f), at(0.12f)), top);
}

const ImU32 kLamp = IM_COL32(226, 220, 132, 255);

void Sun(ImDrawList* draw, const Frame& at) {
    const ImVec2 centre = at(0.5f, 0.5f);
    draw->AddCircleFilled(centre, at(0.2f), kLamp);
    for (int k = 0; k < 8; ++k) {
        const float a = static_cast<float>(k) * std::numbers::pi_v<float> / 4.0f;
        draw->AddLine(at(0.5f + 0.3f * std::cos(a), 0.5f + 0.3f * std::sin(a)),
                      at(0.5f + 0.44f * std::cos(a), 0.5f + 0.44f * std::sin(a)), kLamp,
                      std::max(1.0f, at(0.07f)));
    }
}

void Bulb(ImDrawList* draw, const Frame& at) {
    draw->AddCircleFilled(at(0.5f, 0.4f), at(0.28f), kLamp);
    draw->AddRectFilled(at(0.38f, 0.64f), at(0.62f, 0.9f), Rgb(150, 160, 164), at(0.04f));
}

void Panel(ImDrawList* draw, const Frame& at, bool disk) {
    if (disk) {
        draw->AddEllipseFilled(at(0.5f, 0.32f), ImVec2(at(0.38f), at(0.16f)), kLamp);
    } else {
        const ImVec2 quad[] = {at(0.22f, 0.16f), at(0.92f, 0.16f), at(0.78f, 0.46f),
                               at(0.08f, 0.46f)};
        draw->AddConvexPolyFilled(quad, 4, kLamp);
    }
    for (const float u : {0.3f, 0.5f, 0.7f}) {
        draw->AddLine(at(u, 0.56f), at(u, 0.9f), kLamp, std::max(1.0f, at(0.07f)));
    }
}

void Dome(ImDrawList* draw, const Frame& at) {
    draw->PathArcTo(at(0.5f, 0.72f), at(0.38f), std::numbers::pi_v<float>,
                    2.0f * std::numbers::pi_v<float>, 16);
    draw->PathFillConvex(kLamp);
    draw->AddLine(at(0.08f, 0.76f), at(0.92f, 0.76f), kLamp, std::max(1.0f, at(0.08f)));
}

/// A five-pointed star with a smaller one beside it.
void Stars(ImDrawList* draw, const Frame& at) {
    const auto star = [&](float u, float v, float outer) {
        ImVec2 points[10];
        for (int k = 0; k < 10; ++k) {
            const float a = -0.5f * std::numbers::pi_v<float> +
                            static_cast<float>(k) * std::numbers::pi_v<float> / 5.0f;
            const float r = k % 2 == 0 ? outer : 0.42f * outer;
            points[k] = at(u + r * std::cos(a), v + r * std::sin(a));
        }
        // Not convex: a fan of triangles from the centre.
        for (int k = 0; k < 10; ++k) {
            draw->AddTriangleFilled(at(u, v), points[k], points[(k + 1) % 10], kLamp);
        }
    };
    star(0.42f, 0.56f, 0.38f);
    star(0.8f, 0.2f, 0.16f);
}

void CameraIcon(ImDrawList* draw, const Frame& at) {
    const ImU32 body = Rgb(150, 160, 164);
    draw->AddRectFilled(at(0.08f, 0.32f), at(0.66f, 0.78f), body, at(0.08f));
    draw->AddTriangleFilled(at(0.66f, 0.55f), at(0.94f, 0.32f), at(0.94f, 0.78f), body);
    draw->AddCircleFilled(at(0.26f, 0.22f), at(0.1f), body);
    draw->AddCircleFilled(at(0.48f, 0.2f), at(0.12f), body);
}

void Plane(ImDrawList* draw, const Frame& at) {
    const ImVec2 quad[] = {at(0.3f, 0.3f), at(0.95f, 0.3f), at(0.7f, 0.78f), at(0.05f, 0.78f)};
    draw->AddConvexPolyFilled(quad, 4, Rgb(104, 114, 118));
    draw->AddLine(at(0.5f, 0.3f), at(0.38f, 0.78f), Rgb(150, 160, 164), 1.0f);
    draw->AddLine(at(0.18f, 0.54f), at(0.83f, 0.54f), Rgb(150, 160, 164), 1.0f);
}

void Mountain(ImDrawList* draw, const Frame& at) {
    draw->AddTriangleFilled(at(0.36f, 0.22f), at(0.72f, 0.84f), at(0.02f, 0.84f),
                            Rgb(120, 140, 96));
    draw->AddTriangleFilled(at(0.66f, 0.38f), at(0.98f, 0.84f), at(0.36f, 0.84f),
                            Rgb(150, 126, 92));
    draw->AddTriangleFilled(at(0.36f, 0.22f), at(0.45f, 0.38f), at(0.27f, 0.38f),
                            Rgb(230, 232, 234));
}

void Film(ImDrawList* draw, const Frame& at) {
    const ImU32 colour = Rgb(127, 166, 194);
    draw->AddRect(at(0.1f, 0.18f), at(0.9f, 0.82f), colour, at(0.06f), 0, std::max(1.2f, at(0.08f)));
    for (const float u : {0.24f, 0.42f, 0.6f, 0.78f}) {
        draw->AddRectFilled(at(u - 0.05f, 0.25f), at(u + 0.05f, 0.32f), colour);
        draw->AddRectFilled(at(u - 0.05f, 0.68f), at(u + 0.05f, 0.75f), colour);
    }
}

void Aperture(ImDrawList* draw, const Frame& at) {
    const ImU32 colour = Rgb(127, 166, 194);
    const ImVec2 centre = at(0.5f, 0.5f);
    draw->AddCircle(centre, at(0.38f), colour, 0, std::max(1.2f, at(0.08f)));
    for (int k = 0; k < 6; ++k) {
        const float a = static_cast<float>(k) * std::numbers::pi_v<float> / 3.0f;
        draw->AddLine(at(0.5f + 0.15f * std::cos(a), 0.5f + 0.15f * std::sin(a)),
                      at(0.5f + 0.38f * std::cos(a + 0.6f), 0.5f + 0.38f * std::sin(a + 0.6f)),
                      colour, 1.0f);
    }
}

void Untyped(ImDrawList* draw, const Frame& at) {
    draw->AddCircle(at(0.5f, 0.5f), at(0.24f), Rgb(110, 120, 124), 0, std::max(1.2f, at(0.08f)));
}

}  // namespace

void DrawPrimIcon(ImDrawList* draw, const ImVec2& corner, float size, std::string_view type) {
    const Frame at{corner, size};
    if (type == kXform) {
        Tripod(draw, at);
    } else if (type == kScope || type.empty()) {
        // Untyped prims group others, as folders.
        Folder(draw, at);
    } else if (type == kSphere || type == kCapsule) {
        Ball(draw, at, Rgb(74, 85, 88), Rgb(104, 114, 118), Rgb(170, 178, 182));
    } else if (type == kCylinder || type == kCone) {
        Cylinder(draw, at, type == kCone);
    } else if (type == kMesh || type == kCube || type == kVolume) {
        Cube(draw, at);
    } else if (type == kPlane || type == kGroundPlane) {
        Plane(draw, at);
    } else if (type == kPointInstancer) {
        Cube(draw, at, 0.55f, {-0.2f, 0.18f});
        Cube(draw, at, 0.55f, {0.2f, 0.18f});
        Cube(draw, at, 0.55f, {0.0f, -0.18f});
    } else if (type == kMaterial) {
        Ball(draw, at, Rgb(110, 92, 140), Rgb(150, 128, 178), Rgb(206, 198, 214));
    } else if (type == kDistantLight) {
        Sun(draw, at);
    } else if (type == kSphereLight) {
        Bulb(draw, at);
    } else if (type == kRectLight || type == kDiskLight) {
        Panel(draw, at, type == kDiskLight);
    } else if (type == kDomeLight) {
        Dome(draw, at);
    } else if (type == kStarField) {
        Stars(draw, at);
    } else if (type == kCamera) {
        CameraIcon(draw, at);
    } else if (type == kTerrain) {
        Mountain(draw, at);
    } else if (type == kRenderSettings || type == kRenderProduct || type == kRenderVar) {
        Film(draw, at);
    } else if (type == kSensorFilter) {
        Aperture(draw, at);
    } else {
        Untyped(draw, at);
    }
}

void DrawRendererIcon(ImDrawList* draw, const ImVec2& corner, float size) {
    const Frame at{corner, size};
    // A lit ball on the ground with its shadow: a rendered image.
    draw->AddEllipseFilled(at(0.58f, 0.86f), ImVec2(at(0.36f), at(0.1f)), Rgb(20, 22, 24, 200));
    Ball(draw, at, Rgb(52, 92, 122), Rgb(74, 133, 173), Rgb(214, 232, 244));
    draw->AddCircle(at(0.5f, 0.52f), at(0.4f), Rgb(150, 160, 164), 0, std::max(1.0f, at(0.05f)));
}

void PrimIcon(std::string_view type, float size) {
    const ImVec2 corner = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(size, size));
    DrawPrimIcon(ImGui::GetWindowDrawList(), corner, size, type);
}

}  // namespace ernie::app
