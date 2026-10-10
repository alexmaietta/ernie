// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/xform.h"

#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace ernie {
namespace {

constexpr std::string_view kOpPrefix = "xformOp:";
constexpr std::string_view kInvertPrefix = "!invert!";

/// The axes of `order` in the order they apply.
std::array<int, 3> AxesOf(RotationOrder order) {
    switch (order) {
    case RotationOrder::XYZ:
        return {0, 1, 2};
    case RotationOrder::XZY:
        return {0, 2, 1};
    case RotationOrder::YXZ:
        return {1, 0, 2};
    case RotationOrder::YZX:
        return {1, 2, 0};
    case RotationOrder::ZXY:
        return {2, 0, 1};
    case RotationOrder::ZYX:
        return {2, 1, 0};
    }
    return {0, 1, 2};
}

/// The rotation by `radians` about axis `axis` (0, 1 or 2), right-handed.
glm::dmat3 AxisRotation(int axis, double radians) {
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    glm::dmat3 r(1.0);
    const int a = (axis + 1) % 3;
    const int b = (axis + 2) % 3;
    r[a][a] = c;
    r[a][b] = s;
    r[b][a] = -s;
    r[b][b] = c;
    return r;
}

std::optional<RotationOrder> EulerOrderOf(std::string_view type) {
    constexpr std::pair<std::string_view, RotationOrder> kOrders[] = {
        {"rotateXYZ", RotationOrder::XYZ}, {"rotateXZY", RotationOrder::XZY},
        {"rotateYXZ", RotationOrder::YXZ}, {"rotateYZX", RotationOrder::YZX},
        {"rotateZXY", RotationOrder::ZXY}, {"rotateZYX", RotationOrder::ZYX}};
    for (const auto& [name, order] : kOrders) {
        if (type == name) {
            return order;
        }
    }
    return std::nullopt;
}

/// One entry of `xformOpOrder`: the attribute it reads, its op type (`translate`, `rotateXYZ`...)
/// and whether it is inverted.
struct OpEntry {
    std::string attribute;
    std::string_view type;
    bool invert = false;
};

OpEntry ParseOp(const Path& path, std::string_view entry) {
    OpEntry op;
    if (entry.starts_with(kInvertPrefix)) {
        op.invert = true;
        entry.remove_prefix(kInvertPrefix.size());
    }
    if (!entry.starts_with(kOpPrefix)) {
        throw std::invalid_argument("xformOpOrder of '" + path.String() + "' lists '" +
                                    std::string(entry) + "', which is no xformOp");
    }
    op.attribute = std::string(entry);
    std::string_view type = std::string_view(op.attribute).substr(kOpPrefix.size());
    op.type = type.substr(0, type.find(':'));
    return op;
}

const std::vector<std::string>* OpOrderOf(const Scene& scene, const Path& path) {
    const Value* order = scene.Get(path, tokens::kXformOpOrder);
    if (order == nullptr) {
        return nullptr;
    }
    static const std::vector<std::string> kNoOps;
    const auto* tokens = order->Get<std::vector<std::string>>();
    if (const auto* numbers = order->Get<std::vector<float>>(); numbers && numbers->empty()) {
        return &kNoOps;  // An empty list, as Python passes one.
    }
    if (tokens == nullptr) {
        throw std::invalid_argument("xformOpOrder of '" + path.String() +
                                    "' must be a list of op names");
    }
    return tokens;
}

std::invalid_argument BadOp(const Path& path, std::string_view attribute, std::string_view need) {
    return std::invalid_argument(std::string(attribute) + " of '" + path.String() + "' must be " +
                                 std::string(need));
}

/// The numbers of an op's value or CPU tensor, or a GPU tensor `read` copies to the host, or none
/// if it has neither.
std::optional<std::vector<double>> NumbersOf(const Scene& scene, const Path& path,
                                             std::string_view attribute,
                                             const GpuOpReader* read) {
    if (const Value* value = scene.Get(path, attribute)) {
        return std::visit(
            [&](const auto& v) -> std::vector<double> {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float> ||
                              std::is_same_v<T, int>) {
                    return {static_cast<double>(v)};
                } else if constexpr (std::is_same_v<T, glm::vec3> ||
                                     std::is_same_v<T, glm::dvec3>) {
                    return {double(v.x), double(v.y), double(v.z)};
                } else if constexpr (std::is_same_v<T, glm::dquat>) {
                    return {v.x, v.y, v.z, v.w};
                } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                    std::vector<double> rows(16);
                    for (int row = 0; row < 4; ++row) {
                        for (int col = 0; col < 4; ++col) {
                            rows[static_cast<std::size_t>(4 * row + col)] = v[col][row];
                        }
                    }
                    return rows;
                } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                    return {v.begin(), v.end()};
                } else if constexpr (std::is_same_v<T, glm::vec4>) {
                    // Four numbers are ambiguous for a quaternion (w first or last); see kOrient.
                    throw BadOp(path, attribute, "a quaternion, not four numbers");
                } else {
                    throw BadOp(path, attribute, "numbers, a quaternion or a matrix");
                }
            },
            value->Raw());
    }
    const Tensor* tensor = scene.Read(path, attribute);
    if (tensor == nullptr) {
        return std::nullopt;
    }
    if (tensor->Location().kind != DeviceKind::Cpu && read == nullptr) {
        throw std::invalid_argument(
            std::string(attribute) + " of '" + path.String() +
            "' is a GPU array: the renderer poses the meshes under it on the GPU, but its "
            "transform cannot be read on the CPU (by points, instancers, lights or cameras "
            "under it, or tools)");
    }
    const Tensor host = tensor->Location().kind != DeviceKind::Cpu
                            ? (*read)(attribute, *tensor).Contiguous()
                            : tensor->Contiguous();
    const auto count = static_cast<std::size_t>(host.Elements());
    std::vector<double> numbers(count);
    if (host.Type() == DType::Float64) {
        std::copy_n(host.As<double>(), count, numbers.begin());
    } else if (host.Type() == DType::Float32) {
        std::copy_n(host.As<float>(), count, numbers.begin());
    } else {
        throw BadOp(path, attribute, "float32 or float64");
    }
    return numbers;
}

glm::dvec3 Vec3Of(const Path& path, const OpEntry& op, const std::vector<double>& n,
                  bool scalarAllowed) {
    if (n.size() == 3) {
        return {n[0], n[1], n[2]};
    }
    if (scalarAllowed && n.size() == 1) {
        return glm::dvec3(n[0]);
    }
    throw BadOp(path, op.attribute, scalarAllowed ? "three numbers or one" : "three numbers");
}

bool IsKnownOp(std::string_view type) {
    return type == "translate" || type == "scale" || type == "rotateX" || type == "rotateY" ||
           type == "rotateZ" || type == "orient" || type == "transform" ||
           EulerOrderOf(type).has_value();
}

/// The matrix of one op, inverted if the order says so.
glm::dmat4 OpMatrix(const Scene& scene, const Path& path, const OpEntry& op,
                    const GpuOpReader* read) {
    if (!IsKnownOp(op.type)) {
        throw std::invalid_argument("'" + path.String() + "' has transform op " + op.attribute +
                                    ", which Ernie does not know");
    }
    const std::optional<std::vector<double>> numbers = NumbersOf(scene, path, op.attribute, read);
    if (!numbers) {
        return glm::dmat4(1.0);
    }
    const std::vector<double>& n = *numbers;
    glm::dmat4 m(1.0);
    if (op.type == "translate") {
        m = glm::translate(glm::dmat4(1.0), Vec3Of(path, op, n, false));
    } else if (op.type == "scale") {
        m = glm::scale(glm::dmat4(1.0), Vec3Of(path, op, n, true));
    } else if (op.type == "rotateX" || op.type == "rotateY" || op.type == "rotateZ") {
        if (n.size() != 1) {
            throw BadOp(path, op.attribute, "one angle in degrees");
        }
        m = glm::dmat4(AxisRotation(op.type.back() - 'X', glm::radians(n[0])));
    } else if (const std::optional<RotationOrder> order = EulerOrderOf(op.type)) {
        m = glm::dmat4(EulerRotation(Vec3Of(path, op, n, false), *order));
    } else if (op.type == "orient") {
        if (n.size() != 4) {
            throw BadOp(path, op.attribute, "a quaternion");
        }
        // Arrays are x, y, z, w, or w first by the prim's `ernie:quaternionOrder`.
        const Value* order = scene.Get(path, tokens::kQuaternionOrder);
        const auto* token = order != nullptr ? order->Get<std::string>() : nullptr;
        const bool wFirst = scene.Get(path, op.attribute) == nullptr && token != nullptr &&
                            *token == tokens::kWxyz;
        const glm::dquat q =
            wFirst ? glm::dquat(n[0], n[1], n[2], n[3]) : glm::dquat(n[3], n[0], n[1], n[2]);
        m = glm::dmat4(glm::mat3_cast(glm::normalize(q)));
    } else {
        if (n.size() != 16) {
            throw BadOp(path, op.attribute, "a 4 x 4 matrix");
        }
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                m[col][row] = n[static_cast<std::size_t>(4 * row + col)];
            }
        }
    }
    return op.invert ? glm::inverse(m) : m;
}

bool Near(const glm::dmat4& a, const glm::dmat4& b) {
    double scale = 1.0;
    double error = 0.0;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            scale = std::max(scale, std::abs(a[col][row]));
            error = std::max(error, std::abs(a[col][row] - b[col][row]));
        }
    }
    return error <= 1e-9 * scale;
}

}  // namespace

std::string_view RotateOp(RotationOrder order) {
    switch (order) {
    case RotationOrder::XYZ:
        return tokens::kRotateXYZ;
    case RotationOrder::XZY:
        return tokens::kRotateXZY;
    case RotationOrder::YXZ:
        return tokens::kRotateYXZ;
    case RotationOrder::YZX:
        return tokens::kRotateYZX;
    case RotationOrder::ZXY:
        return tokens::kRotateZXY;
    case RotationOrder::ZYX:
        return tokens::kRotateZYX;
    }
    return tokens::kRotateXYZ;
}

glm::dmat3 EulerRotation(glm::dvec3 degrees, RotationOrder order) {
    const auto [i, j, k] = AxesOf(order);
    return AxisRotation(k, glm::radians(degrees[k])) * AxisRotation(j, glm::radians(degrees[j])) *
           AxisRotation(i, glm::radians(degrees[i]));
}

glm::dvec3 EulerAngles(const glm::dmat3& rotation, RotationOrder order) {
    const auto [i, j, k] = AxesOf(order);
    // +1 for the cyclic orders (XYZ, YZX, ZXY), -1 for the others.
    const double s = (j == (i + 1) % 3) ? 1.0 : -1.0;
    const auto at = [&](int row, int col) { return rotation[col][row]; };
    const double sinMiddle = std::clamp(-s * at(k, i), -1.0, 1.0);
    double first = 0.0;
    double last = 0.0;
    const double middle = std::asin(sinMiddle);
    if (std::abs(sinMiddle) < 1.0 - 1e-12) {
        first = std::atan2(s * at(k, j), at(k, k));
        last = std::atan2(s * at(j, i), at(i, i));
    } else {
        // Gimbal lock: the first and last axes line up; put the whole turn on the first.
        first = std::atan2(-s * at(j, k), at(j, j));
    }
    glm::dvec3 degrees(0.0);
    degrees[i] = glm::degrees(first);
    degrees[j] = glm::degrees(middle);
    degrees[k] = glm::degrees(last);
    return degrees;
}

void SetXform(Scene& scene, const Path& path, const XformOps& xform) {
    if (xform.rotate && xform.orient) {
        throw std::invalid_argument("An xform takes rotate or orient, not both");
    }
    std::vector<std::string> order;
    if (ResetsXformStack(scene, path)) {
        order.emplace_back(tokens::kResetXformStack);
    }
    if (xform.translate) {
        scene.Set(path, tokens::kTranslate, *xform.translate);
        order.emplace_back(tokens::kTranslate);
    }
    if (xform.rotate) {
        const std::string_view op = RotateOp(xform.rotationOrder);
        scene.Set(path, op, *xform.rotate);
        order.emplace_back(op);
    }
    if (xform.orient) {
        scene.Set(path, tokens::kOrient, glm::normalize(*xform.orient));
        order.emplace_back(tokens::kOrient);
    }
    if (xform.scale) {
        scene.Set(path, tokens::kScale, *xform.scale);
        order.emplace_back(tokens::kScale);
    }
    scene.Set(path, tokens::kXformOpOrder, std::move(order));
}

glm::dmat4 ComposeXformOps(const Scene& scene, const Path& path) {
    return ComposeXformOps(scene, path, nullptr);
}

glm::dmat4 ComposeXformOps(const Scene& scene, const Path& path, const GpuOpReader* read) {
    const std::vector<std::string>* order = OpOrderOf(scene, path);
    if (order == nullptr) {
        // Without an order, xformOp:transform alone, as a value or a CPU tensor.
        return OpMatrix(scene, path, ParseOp(path, tokens::kTransform), read);
    }
    glm::dmat4 local(1.0);
    for (const std::string& entry : *order) {
        if (entry == tokens::kResetXformStack) {
            continue;
        }
        local = local * OpMatrix(scene, path, ParseOp(path, entry), read);
    }
    return local;
}

bool ResetsXformStack(const Scene& scene, const Path& path) {
    const std::vector<std::string>* order = OpOrderOf(scene, path);
    return order != nullptr && !order->empty() && order->front() == tokens::kResetXformStack;
}

std::vector<std::pair<std::string, Value>> LocalTransformEdits(const Scene& scene, const Path& path,
                                                               const glm::dmat4& local) {
    const std::vector<std::string>* order = OpOrderOf(scene, path);
    if (order == nullptr) {
        return {{std::string(tokens::kTransform), local}};
    }
    const bool reset = ResetsXformStack(scene, path);
    const auto asTransform = [&]() -> std::vector<std::pair<std::string, Value>> {
        std::vector<std::string> only;
        if (reset) {
            only.emplace_back(tokens::kResetXformStack);
        }
        only.emplace_back(tokens::kTransform);
        return {{std::string(tokens::kXformOpOrder), std::move(only)},
                {std::string(tokens::kTransform), local}};
    };

    // Translate, then one rotate op or orient, then scale, each at most once and without suffixes
    // or inversion: what a decomposition can fill in.
    std::optional<std::string> translate;
    std::optional<std::string> rotate;
    std::optional<std::string> scale;
    int stage = 0;
    for (std::size_t n = reset ? 1 : 0; n < order->size(); ++n) {
        const OpEntry op = ParseOp(path, (*order)[n]);
        const bool plain = !op.invert && op.attribute.size() == kOpPrefix.size() + op.type.size();
        const bool isRotate = op.type == "orient" || EulerOrderOf(op.type).has_value();
        const int opStage = op.type == "translate" ? 1 : isRotate ? 2 : op.type == "scale" ? 3 : 0;
        if (!plain || opStage <= stage) {
            return asTransform();
        }
        stage = opStage;
        (opStage == 1 ? translate : opStage == 2 ? rotate : scale) = op.attribute;
    }

    // local = T * R * S, with a negative determinant taken as a mirrored x.
    glm::dvec3 axes[3] = {glm::dvec3(local[0]), glm::dvec3(local[1]), glm::dvec3(local[2])};
    glm::dvec3 scaling(glm::length(axes[0]), glm::length(axes[1]), glm::length(axes[2]));
    if (scaling.x == 0.0 || scaling.y == 0.0 || scaling.z == 0.0) {
        return asTransform();
    }
    if (glm::determinant(glm::dmat3(local)) < 0.0) {
        scaling.x = -scaling.x;
    }
    const glm::dmat3 rotation(axes[0] / scaling.x, axes[1] / scaling.y, axes[2] / scaling.z);
    const glm::dvec3 offset(local[3]);

    // The op values, and the matrix they make, rebuilt from those values: it must reproduce `local`
    // (no shear, and nothing the order has no op for).
    std::vector<std::pair<std::string, Value>> edits;
    glm::dmat4 rebuilt(1.0);
    if (translate) {
        edits.emplace_back(*translate, offset);
        rebuilt = glm::translate(rebuilt, offset);
    }
    if (rotate) {
        if (*rotate == tokens::kOrient) {
            const glm::dquat q = glm::normalize(glm::quat_cast(rotation));
            edits.emplace_back(*rotate, q);
            rebuilt = rebuilt * glm::dmat4(glm::mat3_cast(q));
        } else {
            const RotationOrder rotationOrder =
                *EulerOrderOf(std::string_view(*rotate).substr(kOpPrefix.size()));
            const glm::dvec3 degrees = EulerAngles(rotation, rotationOrder);
            edits.emplace_back(*rotate, degrees);
            rebuilt = rebuilt * glm::dmat4(EulerRotation(degrees, rotationOrder));
        }
    }
    if (scale) {
        edits.emplace_back(*scale, scaling);
        rebuilt = glm::scale(rebuilt, scaling);
    }
    if (!Near(rebuilt, local)) {
        return asTransform();
    }
    return edits;
}

bool IsXformAttribute(std::string_view attribute) {
    return attribute == tokens::kXformOpOrder || attribute.starts_with(kOpPrefix);
}

}  // namespace ernie
