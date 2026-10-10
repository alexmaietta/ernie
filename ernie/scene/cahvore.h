// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/camera.h"
#include "ernie/scene/wide_view.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// A camera calibrated in JPL's CAHV family of models, as Mars rovers' cameras are (their PDS
/// labels' GEOMETRIC_CAMERA_MODEL, JPL's camera model files): CAHV, a linear (pinhole) model;
/// CAHVOR, with radial distortion about an optical axis; and CAHVORE, with a fish-eye term and an
/// entrance pupil that moves along the optical axis with the field angle, so its rays do not
/// meet at one point.
/// - C: the projection centre (for CAHVORE, the entrance pupil at zero field angle), in the
///   model's frame and units (metres for rovers).
/// - A: the unit axis perpendicular to the image plane, pointing into the scene.
/// - H, V: image x (right) and y (down) of point P are (P - C) . H / (P - C) . A and
///   (P - C) . V / (P - C) . A for a CAHV model: H and V hold the focal lengths in pixels and
///   the image centre.
/// - O: the unit optical axis about which the distortion is radial.
/// - R: radial distortion coefficients (rho0, rho1, rho2).
/// - E: the entrance pupil's movement coefficients (epsilon0 to epsilon2).
/// - The CAHVORE type: 1 perspective (linearity 1), 2 fish-eye (linearity 0), 3 general, with
///   its linearity between them.
/// Image coordinates are the models': (0, 0) is the centre of the upper-left pixel, x to the
/// right, y down.
///
/// References:
/// - Y. Yakimovsky and R. Cunningham, "A System for Extracting Three-Dimensional Measurements
///   from a Stereo Pair of TV Cameras", Computer Graphics and Image Processing 7(2):195-210,
///   1978 (CAHV).
/// - D. B. Gennery, "Least-Squares Camera Calibration Including Lens Distortion and Automatic
///   Editing of Calibration Points", in A. Gruen and T. Huang (eds.), Calibration and
///   Orientation of Cameras in Computer Vision, Springer, 2001, 123-136 (CAHVOR).
/// - D. B. Gennery, "Generalized Camera Calibration Including Fish-Eye Lenses", International
///   Journal of Computer Vision 68(3):239-266, 2006 (CAHVORE).
/// - T. Litwin's cmod library (cmod_cahv.c, cmod_cahvor.c, cmod_cahvore.c), as ported by
///   NASA-AMMOS, "CameraModelUtilsJS", https://github.com/NASA-AMMOS/CameraModelUtilsJS
///   (Apache-2.0): `ImageToRay` follows its cahvore-utilities/cahvore.js.
/// - PDS Imaging Node, "Geometric Camera Models" (MSL GEOMETRIC_CM.TXT), for the conventions.
/// - S. Aiazzi et al., "IRIS: High-fidelity Perception Sensor Modeling for Closed-Loop
///   Planetary Simulations", AIAA SciTech 2022, AIAA 2022-1433 (CAHVORE camera rays in a
///   planetary simulator).
struct CahvoreModel {
    enum class Kind { Cahv, Cahvor, Cahvore };
    Kind kind = Kind::Cahv;
    glm::dvec3 c{0.0};
    glm::dvec3 a{0.0, 0.0, 1.0};
    glm::dvec3 h{0.0};
    glm::dvec3 v{0.0};
    glm::dvec3 o{0.0, 0.0, 1.0};
    glm::dvec3 r{0.0};
    glm::dvec3 e{0.0};
    int pupilType = 1;       ///< CAHVORE's type: 1 perspective, 2 fish-eye, 3 general.
    double linearity = 1.0;  ///< CAHVORE type 3's.
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    /// CAHVORE's linearity for its type: 1 for perspective, 0 for fish-eye, else `linearity`.
    double Linearity() const;

    /// A projection ray: its origin (C, or for CAHVORE the entrance pupil's position for its
    /// field angle) and unit direction.
    struct Ray {
        glm::dvec3 origin{0.0};
        glm::dvec3 direction{0.0};
    };
    /// The ray through image point `image` (cmod's 2d_to_3d).
    /// @throws std::invalid_argument for a degenerate model or a point its distortion cannot
    /// undo.
    Ray ImageToRay(glm::dvec2 image) const;

    /// The image point of 3D point `point`: CAHV and CAHVOR in closed form (cmod's 3d_to_2d),
    /// CAHVORE by Newton's method on `ImageToRay` from its CAHV projection.
    glm::dvec2 PointToImage(const glm::dvec3& point) const;

    /// The model as Ernie renders it, in the model's frame (camera to frame).
    struct ErnieCamera {
        /// Its pose (`transform`), focal length, apertures and principal point, and the lens:
        /// CAHV and CAHVOR as perspective with Brown-Conrady terms, CAHVORE as a Kannala-Brandt
        /// fish-eye, each fitted to the model's rays over the image.
        Camera camera;
        /// The fit's largest departure from the model, pixels, over the image: from its
        /// distortion's optical axis O lying off the image plane's normal A (a tilt Ernie's
        /// lenses, centred on the normal, cannot take; pixels for long focal lengths), and any
        /// non-square or skewed pixels.
        double residualPixels = 0.0;
        /// Whether the model's image is a mirror of Ernie's (V running up against H x A): flip
        /// the rendered rows.
        bool rowsReversed = false;
        /// CAHVORE's largest entrance pupil shift over the image, in the model's units: Ernie's
        /// camera is central, so a point at distance d moves by up to about this / d radians.
        double pupilShiftMax = 0.0;
        /// The largest field angle over the image, radians.
        double fieldAngleMax = 0.0;
    };
    /// @param pixelPitchUm the detector's pixel pitch, which the models do not hold: Ernie's
    /// apertures and focal length are physical, for the camera equation.
    /// @throws std::invalid_argument without an image size or a positive pitch, or for a model
    /// whose image axes are more than a degree from perpendicular (not a physical camera).
    ErnieCamera ToCamera(double pixelPitchUm) const;

    /// The model's exact rays, every `step` pixels across `ToCamera`'s image, in its camera's
    /// frame (its rows flipped for a mirrored model): a lens that takes the model's distortion
    /// about its tilted optical axis, which `ToCamera`'s parametric lens leaves as a residual. The
    /// entrance pupil's movement is still left out.
    /// @throws std::invalid_argument as `ToCamera`, or for a step of 0.
    LensRayTable RayTable(std::uint32_t step = 8) const;
};

/// A named camera model in a frame, as camera model files list them.
struct NamedCahvore {
    std::string name;
    std::string frame;  ///< The frame its vectors are in, such as a rover's Body_RSM_EL.
    CahvoreModel model;
};

/// The models in JSON `json`: an array of {"name", "frame", "model": {"type": "cahv" | "cahvor" |
/// "cahvore", "C", "A", "H", "V", "O", "R", "E", "pupilType", "linearity", "width", "height"}},
/// the form of NASA-JPL's m2020-urdf-models (m2020-camera-models.json).
/// @throws std::invalid_argument for malformed JSON or a missing or mistyped field.
std::vector<NamedCahvore> ReadCahvoreModels(std::string_view json);

/// `ReadCahvoreModels` of the file at `path`.
/// @throws std::runtime_error if the file cannot be read, or as `ReadCahvoreModels`.
std::vector<NamedCahvore> ReadCahvoreModelsFile(const std::string& path);

}  // namespace ernie
