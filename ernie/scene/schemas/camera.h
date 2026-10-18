// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.h
/// @brief UsdGeomCamera: a view's projection, lens and exposure.
///
/// References:
/// - Pixar, UsdGeomCamera, https://openusd.org/release/api/class_usd_geom_camera.html
#pragma once

#include "ernie/scene/schemas/geom.h"

// clang-format off
#define ERNIE_LENS_LENGTH(X, Name, usdName, fallback, text)                                        \
    X(double, Name, usdName, fallback, text " in tenths of a scene unit (millimetres in metres).", \
      .group = "Lens", .lower = AtLeast(0.0), .finite = true)

#define ERNIE_CAMERA_ATTRIBUTES(X)                                                                 \
    X(std::string, Projection, "projection", "perspective", "How it projects the scene.",          \
      .group = "Lens", .options = {"perspective", "orthographic"})                                 \
    ERNIE_LENS_LENGTH(X, HorizontalAperture, "horizontalAperture", 20.955, "Film width")           \
    ERNIE_LENS_LENGTH(X, VerticalAperture, "verticalAperture", 15.2908, "Film height")             \
    X(double, HorizontalApertureOffset, "horizontalApertureOffset", 0.0,                           \
      "The film's offset from the lens axis along x, in the aperture's units.", .group = "Lens")   \
    X(double, VerticalApertureOffset, "verticalApertureOffset", 0.0,                               \
      "The film's offset from the lens axis along y.", .group = "Lens")                            \
    ERNIE_LENS_LENGTH(X, FocalLength, "focalLength", 50.0, "The lens's focal length")              \
    X(glm::dvec2, ClippingRange, "clippingRange", glm::dvec2(1.0, 1000000.0),                      \
      "Near and far distances in scene units.", .group = "Lens", .lower = Above(0.0))              \
    X(Tensor, ClippingPlanes, "clippingPlanes", Tensor(),                                          \
      "(n, 4) planes (a, b, c, d) keeping points where a x + b y + c z + d >= 0.",                 \
      .group = "Lens", .components = 4)                                                            \
    X(double, FStop, "fStop", 0.0, "The lens's f-number; 0 for a pinhole, all in focus.",          \
      .group = "Lens", .lower = AtLeast(0.0))                                                      \
    X(double, FocusDistance, "focusDistance", 0.0, "Scene units to the plane in focus.",           \
      .group = "Lens", .lower = AtLeast(0.0))                                                      \
    X(std::string, StereoRole, "stereoRole", "mono", "Which eye it is.", .group = "Lens",          \
      .options = {"mono", "left", "right"})                                                        \
    X(double, ShutterOpen, "shutter:open", 0.0, "Frames from the time code the shutter opens.",   \
      .group = "Shutter")                                                                          \
    X(double, ShutterClose, "shutter:close", 0.0, "Frames from the time code it closes.",          \
      .group = "Shutter")                                                                          \
    X(double, Exposure, "exposure", 0.0, "Stops of exposure added to the image.",                  \
      .group = "Exposure", .finite = true)                                                         \
    X(double, ExposureIso, "exposure:iso", 100.0, "The sensor's ISO speed.", .group = "Exposure",  \
      .lower = Above(0.0))                                                                         \
    X(double, ExposureTime, "exposure:time", 1.0, "Seconds of exposure.", .group = "Exposure",     \
      .lower = AtLeast(0.0))                                                                       \
    X(double, ExposureFStop, "exposure:fStop", 1.0, "The f-number exposure is computed with.",     \
      .group = "Exposure", .lower = Above(0.0))                                                    \
    X(double, ExposureResponsivity, "exposure:responsivity", 1.0, "The sensor's responsivity.",    \
      .group = "Exposure", .lower = AtLeast(0.0))
// clang-format on

namespace ernie {

ERNIE_SCHEMA_ACCESSORS(Camera, Xformable, ERNIE_CAMERA_ATTRIBUTES);

/// A view looking down -z, y up, with a lens and film.
class Camera : public CameraAccessors<Camera, Xformable> {
    ERNIE_SCHEMA_BODY(Camera, Xformable)
    ERNIE_SCHEMA_DEFINE(Camera)
};

}  // namespace ernie
