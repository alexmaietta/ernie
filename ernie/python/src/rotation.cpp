// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <tuple>

#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

#include "bindings.h"

namespace nb = nanobind;

namespace ernie::python {

namespace {

using Vector = std::array<double, 3>;
using Quaternion = std::tuple<double, double, double, double>;

Double3 ToDouble3(const Vector& v) {
    return {v[0], v[1], v[2]};
}

Quaternion ToTuple(const Quatd& q) {
    return {q.x, q.y, q.z, q.w};
}

}

void BindRotation(nb::module_& module) {
    module.def(
        "look_at",
        [](const Vector& eye, const Vector& target, const Vector& up) {
            return ToTuple(LookAt(ToDouble3(eye), ToDouble3(target), ToDouble3(up)));
        },
        nb::arg("eye"), nb::arg("target"), nb::arg("up") = Vector{0, 0, 1});
    module.def(
        "look_along",
        [](const Vector& direction, const Vector& up) {
            return ToTuple(LookAlong(ToDouble3(direction), ToDouble3(up)));
        },
        nb::arg("direction"), nb::arg("up") = Vector{0, 0, 1});
    module.def(
        "from_axis_angle",
        [](const Vector& axis, double degrees) {
            return ToTuple(FromAxisAngle(ToDouble3(axis), degrees));
        },
        nb::arg("axis"), nb::arg("degrees"));
    module.def(
        "from_azimuth_elevation",
        [](double azimuth, double elevation) {
            return ToTuple(FromAzimuthElevation(azimuth, elevation));
        },
        nb::arg("azimuth"), nb::arg("elevation"));
}

}
