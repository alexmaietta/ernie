// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/rotation.h"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "ernie/util/error.h"

namespace ernie {

namespace {

double Dot(Double3 a, Double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Double3 Cross(Double3 a, Double3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Double3 Unit(Double3 v) {
    const double length = std::sqrt(Dot(v, v));
    if (length == 0) {
        throw Error("A direction of zero length has no rotation");
    }
    return {v.x / length, v.y / length, v.z / length};
}

double Radians(double degrees) {
    return degrees * std::numbers::pi / 180;
}

double Degrees(double radians) {
    return radians * 180 / std::numbers::pi;
}

}

Quatd LookAlong(Double3 direction, Double3 up) {
    const Double3 z = Unit({-direction.x, -direction.y, -direction.z});
    Double3 axis = Unit(up);
    if (std::abs(Dot(z, axis)) > 0.999) {
        axis = std::abs(axis.y) < 0.999 ? Double3{0, 1, 0} : Double3{0, 0, 1};
    }
    const Double3 x = Unit(Cross(axis, z));
    return FromAxes(x, Cross(z, x), z);
}

Quatd LookAt(Double3 eye, Double3 target, Double3 up) {
    return LookAlong({target.x - eye.x, target.y - eye.y, target.z - eye.z}, up);
}

Quatd FromAxisAngle(Double3 axis, double degrees) {
    const Double3 n = Unit(axis);
    const double half = Radians(degrees) / 2;
    const double s = std::sin(half);
    return {n.x * s, n.y * s, n.z * s, std::cos(half)};
}

Quatd FromAzimuthElevation(double azimuth, double elevation) {
    const double a = Radians(azimuth);
    const double e = Radians(elevation);
    return LookAlong({-std::sin(a) * std::cos(e), -std::cos(a) * std::cos(e), -std::sin(e)});
}

Quatd FromAxes(Double3 x, Double3 y, Double3 z) {
    const double trace = x.x + y.y + z.z;
    if (trace > 0) {
        const double s = std::sqrt(trace + 1) * 2;
        return {(y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, s / 4};
    }
    if (x.x > y.y && x.x > z.z) {
        const double s = std::sqrt(1 + x.x - y.y - z.z) * 2;
        return {s / 4, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s};
    }
    if (y.y > z.z) {
        const double s = std::sqrt(1 + y.y - x.x - z.z) * 2;
        return {(y.x + x.y) / s, s / 4, (z.y + y.z) / s, (z.x - x.z) / s};
    }
    const double s = std::sqrt(1 + z.z - x.x - y.y) * 2;
    return {(z.x + x.z) / s, (z.y + y.z) / s, s / 4, (x.y - y.x) / s};
}

Quatd FromEulerDegrees(Double3 degrees) {
    const double cx = std::cos(Radians(degrees.x) / 2);
    const double sx = std::sin(Radians(degrees.x) / 2);
    const double cy = std::cos(Radians(degrees.y) / 2);
    const double sy = std::sin(Radians(degrees.y) / 2);
    const double cz = std::cos(Radians(degrees.z) / 2);
    const double sz = std::sin(Radians(degrees.z) / 2);
    return {sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz};
}

Double3 ToEulerDegrees(Quatd q) {
    const double sine = std::clamp(2 * (q.w * q.y - q.z * q.x), -1.0, 1.0);
    return {Degrees(std::atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))),
            Degrees(std::asin(sine)),
            Degrees(std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))};
}

Quatd Multiply(Quatd a, Quatd b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

Quatd Conjugate(Quatd rotation) {
    return {-rotation.x, -rotation.y, -rotation.z, rotation.w};
}

Quatd Normalize(Quatd q) {
    const double length = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return length > 0 ? Quatd{q.x / length, q.y / length, q.z / length, q.w / length} : Quatd{};
}

Double3 Rotate(Quatd rotation, Double3 vector) {
    const Double3 axis{rotation.x, rotation.y, rotation.z};
    const Double3 cross = Cross(axis, vector);
    const Double3 twice = Cross(axis, cross);
    return {vector.x + 2 * (rotation.w * cross.x + twice.x),
            vector.y + 2 * (rotation.w * cross.y + twice.y),
            vector.z + 2 * (rotation.w * cross.z + twice.z)};
}

}
