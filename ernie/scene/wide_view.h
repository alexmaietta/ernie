// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/camera.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <vector>

/// Rays of a lens in the camera's frame, and the cube faces a wide camera's views are rendered
/// on, for fields the single pinhole raster image cannot take (its tangent grows without bound
/// towards 90 degrees off axis).
///
/// The camera's frame is USD's: x right, y up, looking down -z. A cube face is a square pinhole
/// view of 90 degrees (plus a margin) along one of the camera's axes; the sensor follows each
/// pixel's lens ray to the face it leaves through. kernel/film/cube_faces.slang holds the same
/// conventions for the sensor kernel.
///
/// References:
/// - N. Greene, "Environment Mapping and Other Applications of World Projections", IEEE Computer
///   Graphics and Applications 6(11):21-29, 1986 (cube maps).
/// - Khronos, "Vulkan Specification", 16.5.4 "Cube Map Face Selection" (selection by the
///   largest-magnitude component).
namespace ernie {

/// The unit ray, in the camera's frame, that lands at distorted normalised image coordinates
/// `distorted` ((x, y) = ((u - cx) / f, (cy - v) / f), y up) through `lens`: through the pinhole
/// or the undone Brown-Conrady distortion for a perspective lens, and at the field angle a
/// fisheye's radius gives, which may pass 90 degrees.
glm::dvec3 LensRay(const LensModel& lens, glm::dvec2 distorted);

/// The faces of a wide camera's views: forward along the camera's -z, then its +x, -x, +y and
/// -y. Five leave out only the back face's pyramid, so they cover every ray within 125.3 degrees
/// of the axis (135 along the side faces' middles), any fisheye up to a 250 degree field.
enum class CubeFace { Front = 0, Right = 1, Left = 2, Up = 3, Down = 4 };
inline constexpr int kCubeFaces = 5;

/// Face `face`'s axes in the camera's frame (columns x, y, z): a camera on them looks down their
/// -z, as the camera itself does on the front face. Proper rotations.
glm::dmat3 CubeFaceAxes(CubeFace face);

/// The face whose view direction lies nearest unit ray `ray` (in the camera's frame): the
/// largest of -z, x, -x, y and -y.
CubeFace CubeFaceOf(const glm::dvec3& ray);

/// Where `ray` crosses face `face`'s image plane at unit distance: its tangent-plane coordinates
/// (x right, y up on the face), in [-1, 1] across the face's 90 degrees.
glm::dvec2 CubeFacePlane(CubeFace face, const glm::dvec3& ray);

/// The unit ray through face `face`'s tangent-plane coordinates `plane`.
glm::dvec3 CubeFaceRay(CubeFace face, glm::dvec2 plane);

/// The tangent of the half field a face's view covers: 1 (45 degrees) plus `marginPixels` of a
/// face `facePixels` across, so bilinear taps at a seam stay inside the face they sample.
double CubeFaceExtent(double facePixels, double marginPixels = 2.0);

/// Whether `camera` needs its views on cube faces: a fisheye or a lens of rays whose field
/// reaches past `kMaxSingleViewAngle` off axis somewhere in its image, where one pinhole raster
/// image would spread its pixels too thin at the centre.
bool NeedsCubeFaces(const Camera& camera);

/// The field angle (radians) past which a fisheye renders on cube faces: 60 degrees, where a
/// single raster image's centre is already twice as coarse as its sensor's (tan 60 / (pi / 3)).
inline constexpr double kMaxSingleViewAngle = 1.0471975511965976;

/// The largest field angle `camera` sees, at the corner of its image farthest from its principal
/// point (its edges' midpoints for a fisheye whose field folds no earlier).
double MaxFieldAngle(const Camera& camera);

/// The unit ray, in the camera's frame, of sensor position `pixel` of `camera` (pixels from the
/// top left corner, centres at + 0.5): its lens's ray table's, or `LensRay` of its normalised
/// image coordinates.
glm::dvec3 CameraRay(const Camera& camera, glm::dvec2 pixel);

}  // namespace ernie
