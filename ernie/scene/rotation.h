// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/export.h"
#include "ernie/scene/kernels/schema/types.h"

namespace ernie {

ERNIE_API Quatd LookAlong(Double3 direction, Double3 up = {0, 0, 1});
ERNIE_API Quatd LookAt(Double3 eye, Double3 target, Double3 up = {0, 0, 1});
ERNIE_API Quatd FromAxisAngle(Double3 axis, double degrees);
ERNIE_API Quatd FromAzimuthElevation(double azimuth, double elevation);
ERNIE_API Quatd FromAxes(Double3 x, Double3 y, Double3 z);
ERNIE_API Quatd FromEulerDegrees(Double3 degrees);
ERNIE_API Double3 ToEulerDegrees(Quatd rotation);

ERNIE_API Quatd Multiply(Quatd a, Quatd b);
ERNIE_API Quatd Conjugate(Quatd rotation);
ERNIE_API Quatd Normalize(Quatd rotation);
ERNIE_API Double3 Rotate(Quatd rotation, Double3 vector);

}
