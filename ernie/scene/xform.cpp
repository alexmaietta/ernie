// SPDX-License-Identifier: BSD-3-Clause
/// @file xform.cpp
/// @brief Transform op composition, after UsdGeomXformable.
#include "ernie/scene/xform.h"

#include "ernie/scene/convert.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

constexpr std::array<std::array<int, 3>, 6> kAxes{
    {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
constexpr std::array<std::string_view, 6> kRotateOps{"xformOp:rotateXYZ", "xformOp:rotateXZY",
                                                     "xformOp:rotateYXZ", "xformOp:rotateYZX",
                                                     "xformOp:rotateZXY", "xformOp:rotateZYX"};

/// The axes an order turns about, first first.
std::array<int, 3> TurnAxes(RotationOrder order) {
    return kAxes[static_cast<std::size_t>(order)];
}

/// A right-handed turn about one axis, built from its cosine and sine so the axis stays exact.
glm::dmat3 AxisRotation(int axis, double degrees) {
    const double c = std::cos(glm::radians(degrees));
    const double s = std::sin(glm::radians(degrees));
    const int a = (axis + 1) % 3;
    const int b = (axis + 2) % 3;
    glm::dmat3 r(1.0);
    r[a][a] = c;
    r[a][b] = s;
    r[b][a] = -s;
    r[b][b] = c;
    return r;
}

/// One entry of `xformOpOrder`: its attribute, op type (`translate`, `rotateXYZ`) and inversion.
struct OpEntry {
    std::string_view attribute;
    std::string_view type;
    bool invert = false;
};

OpEntry ParseOp(const Path& path, std::string_view entry) {
    OpEntry op;
    if (entry.starts_with(tokens::kInvertPrefix)) {
        op.invert = true;
        entry.remove_prefix(tokens::kInvertPrefix.size());
    }
    if (!entry.starts_with(tokens::kXformOpPrefix)) {
        throw std::invalid_argument("xformOpOrder of '" + path.String() + "' lists '" +
                                    std::string(entry) + "', which is no xformOp");
    }
    op.attribute = entry;
    const std::string_view type = entry.substr(tokens::kXformOpPrefix.size());
    op.type = type.substr(0, type.find(':'));
    return op;
}

std::optional<RotationOrder> EulerOrder(std::string_view type) {
    for (std::size_t i = 0; i < kRotateOps.size(); ++i) {
        if (kRotateOps[i].substr(tokens::kXformOpPrefix.size()) == type) {
            return static_cast<RotationOrder>(i);
        }
    }
    return std::nullopt;
}

std::invalid_argument BadOp(const Path& path, const OpEntry& op, std::string_view need) {
    return std::invalid_argument(std::string(op.attribute) + " of '" + path.String() +
                                 "' must be " + std::string(need));
}

glm::dvec3 ToVec3(const Path& path, const OpEntry& op, const Value& value, bool scalar) {
    if (const auto vector = ToVector<3>(value)) {
        return *vector;
    }
    if (const auto number = ToNumber(value); scalar && number) {
        return glm::dvec3(*number);
    }
    throw BadOp(path, op, scalar ? "three numbers or one" : "three numbers");
}

/// The matrix of one op, inverted if the order says so; identity for an op without a value.
glm::dmat4 OpMatrix(const Scene& scene, PrimId prim, const OpEntry& op) {
    const Path path = scene.PrimPath(prim);
    const std::optional<RotationOrder> euler = EulerOrder(op.type);
    const bool single = op.type == "rotateX" || op.type == "rotateY" || op.type == "rotateZ";
    if (!euler && !single && op.type != "translate" && op.type != "scale" && op.type != "orient" &&
        op.type != "transform") {
        throw std::invalid_argument("'" + path.String() + "' has transform op " +
                                    std::string(op.attribute) + ", which Ernie does not know");
    }
    const std::optional<Value> value = scene.Get(prim, op.attribute);
    if (!value) {
        return glm::dmat4(1.0);
    }
    glm::dmat4 m(1.0);
    if (op.type == "translate") {
        m = glm::translate(m, ToVec3(path, op, *value, false));
    } else if (op.type == "scale") {
        m = glm::scale(m, ToVec3(path, op, *value, true));
    } else if (single) {
        const std::optional<double> degrees = ToNumber(*value);
        if (!degrees) {
            throw BadOp(path, op, "one angle in degrees");
        }
        m = glm::dmat4(AxisRotation(op.type.back() - 'X', *degrees));
    } else if (euler) {
        m = glm::dmat4(EulerRotation(ToVec3(path, op, *value, false), *euler));
    } else if (op.type == "orient") {
        const auto* q = value->Get<glm::dquat>();
        if (q == nullptr) {
            throw BadOp(path, op, "a quaternion");
        }
        m = glm::dmat4(glm::mat3_cast(glm::normalize(*q)));
    } else {
        const auto* matrix = value->Get<glm::dmat4>();
        if (matrix == nullptr) {
            throw BadOp(path, op, "a 4 x 4 matrix");
        }
        m = *matrix;
    }
    return op.invert ? glm::inverse(m) : m;
}

/// @return The prim's op order, or none when it has no `xformOpOrder`.
std::optional<std::vector<std::string>> OpOrder(const Scene& scene, PrimId prim) {
    const std::optional<Value> order = scene.Get(prim, tokens::kXformOpOrder);
    if (!order) {
        return std::nullopt;
    }
    return *order->Get<std::vector<std::string>>();
}

}  // namespace

std::string_view RotateOp(RotationOrder order) {
    return kRotateOps[static_cast<std::size_t>(order)];
}

glm::dmat3 EulerRotation(const glm::dvec3& degrees, RotationOrder order) {
    const auto [i, j, k] = TurnAxes(order);
    return AxisRotation(k, degrees[k]) * AxisRotation(j, degrees[j]) * AxisRotation(i, degrees[i]);
}

glm::dvec3 EulerAngles(const glm::dmat3& rotation, RotationOrder order) {
    const auto [i, j, k] = TurnAxes(order);
    // +1 for the cyclic orders (XYZ, YZX, ZXY), -1 for the others.
    const double s = (j == (i + 1) % 3) ? 1.0 : -1.0;
    const auto at = [&](int row, int col) { return rotation[col][row]; };
    const double sinMiddle = std::clamp(-s * at(k, i), -1.0, 1.0);
    double first = 0.0;
    double last = 0.0;
    if (std::abs(sinMiddle) < 1.0 - 1e-12) {
        first = std::atan2(s * at(k, j), at(k, k));
        last = std::atan2(s * at(j, i), at(i, i));
    } else {
        // Gimbal lock: the first and last axes line up, so the first takes the whole turn.
        first = std::atan2(-s * at(j, k), at(j, j));
    }
    glm::dvec3 degrees(0.0);
    degrees[i] = glm::degrees(first);
    // atan2 against the cosine stays accurate near lock, where asin magnifies rounding.
    degrees[j] = glm::degrees(std::atan2(sinMiddle, std::hypot(at(k, j), at(k, k))));
    degrees[k] = glm::degrees(last);
    return degrees;
}

void SetXform(Scene& scene, const Path& path, const XformOps& xform) {
    if (xform.rotate && xform.orient) {
        throw std::invalid_argument("An xform takes rotate or orient, not both");
    }
    const PrimId prim = scene.Require(path);
    std::vector<std::string> order;
    if (ResetsXformStack(scene, prim)) {
        order.emplace_back(tokens::kResetXformStack);
    }
    const auto add = [&](std::string_view op, const Value& value) {
        scene.Set(prim, op, value);
        order.emplace_back(op);
    };
    if (xform.translate) {
        add(tokens::kTranslate, *xform.translate);
    }
    if (xform.rotate) {
        add(RotateOp(xform.rotationOrder), *xform.rotate);
    }
    if (xform.orient) {
        add(tokens::kOrient, glm::normalize(*xform.orient));
    }
    if (xform.scale) {
        add(tokens::kScale, *xform.scale);
    }
    scene.Set(prim, tokens::kXformOpOrder, std::move(order));
}

glm::dmat4 ComposeXformOps(const Scene& scene, PrimId prim) {
    const Path path = scene.PrimPath(prim);
    const std::optional<std::vector<std::string>> order = OpOrder(scene, prim);
    if (!order) {
        return OpMatrix(scene, prim, ParseOp(path, tokens::kTransform));
    }
    glm::dmat4 local(1.0);
    for (const std::string& entry : *order) {
        if (entry != tokens::kResetXformStack) {
            local = local * OpMatrix(scene, prim, ParseOp(path, entry));
        }
    }
    return local;
}

bool ResetsXformStack(const Scene& scene, PrimId prim) {
    const std::optional<std::vector<std::string>> order = OpOrder(scene, prim);
    return order && !order->empty() && order->front() == tokens::kResetXformStack;
}

}  // namespace ernie
