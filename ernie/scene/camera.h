#pragma once

#include "ernie/scene/scene.h"
#include "ernie/util/spectrum.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ernie {

/// The raster view of a camera for one frame, derived from a `Camera` by `Camera::RasterView`:
/// what the rasterizer's passes need.
struct CameraView {
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::vec3 eye{0.0f};
    glm::vec3 forward{0.0f, 1.0f, 0.0f};  ///< Unit viewing direction.
    float verticalFov = 0.785398f;        ///< Radians.
    float aspect = 1.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;

    glm::mat4 ViewProjection() const {
        return projection * view;
    }

    /// Whether the projection is orthographic: every ray runs along `forward`, and `verticalFov`
    /// is 0.
    bool IsOrthographic() const {
        return projection[3][3] == 1.0f;
    }
};

/// How a lens maps the field angle of a ray to its place in the image.
enum class LensProjection : std::uint8_t {
    /// Rectilinear: a ray at field angle theta lands at radius tan(theta), with Brown-Conrady
    /// distortion on top.
    Perspective,
    /// Kannala-Brandt fisheye: at radius theta_d = theta (1 + k1 theta^2 + k2 theta^4 +
    /// k3 theta^6 + k4 theta^8), with `fisheye` as k1 to k4; all zero is the equidistant
    /// projection.
    Fisheye,
};

/// A lens: its projection, natural vignetting and transmission. A perspective lens has radial and
/// tangential distortion (Brown-Conrady, in OpenCV's convention), which maps undistorted
/// normalised image coordinates (x, y) = (X/Z, Y/Z) to distorted ones:
///   x' = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
///   y' = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y
/// A fisheye lens maps them to theta_d / r (x, y), with theta = atan(r) (OpenCV's fisheye model).
///
/// The rasterizer draws a pinhole image and the sensor resamples it through the lens, so a
/// fisheye's field angle must stay below `kMaxFisheyeFieldAngle`.
///
/// References:
/// - D. C. Brown, "Decentering Distortion of Lenses", Photogrammetric Engineering 32(3):444-462,
///   1966.
/// - J. Kannala and S. S. Brandt, "A Generic Camera Model and Calibration Method for
///   Conventional, Wide-Angle, and Fish-Eye Lenses", IEEE TPAMI 28(8):1335-1340, 2006.
///   https://doi.org/10.1109/TPAMI.2006.153
/// - OpenCV, "Camera Calibration and 3D Reconstruction",
///   https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html, and "Fisheye camera model",
///   https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
struct LensModel {
    LensProjection projection = LensProjection::Perspective;
    double k1 = 0.0;
    double k2 = 0.0;
    double k3 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    std::array<double, 4> fisheye{};  ///< Kannala-Brandt k1 to k4.
    bool vignetting = false;    ///< Natural falloff of irradiance off axis (cos^4 for pinholes).
    double transmission = 1.0;  ///< Fraction of light the optics pass.

    /// The largest field angle a fisheye may see, in radians (80 degrees): the pinhole image the
    /// sensor resamples grows as its tangent.
    static constexpr double kMaxFisheyeFieldAngle = 1.3962634015954636;

    /// Whether the lens is a pinhole: perspective without distortion.
    bool IsPinhole() const {
        return projection == LensProjection::Perspective && !HasDistortion();
    }

    /// Whether Brown-Conrady distortion is set.
    bool HasDistortion() const {
        return k1 != 0.0 || k2 != 0.0 || k3 != 0.0 || p1 != 0.0 || p2 != 0.0;
    }

    /// Distorted normalised image coordinates of pinhole ones `undistorted`.
    glm::dvec2 Distort(glm::dvec2 undistorted) const;

    /// The inverse of `Distort`: pinhole normalised image coordinates of the ray landing at
    /// `distorted`. By fixed-point iteration for perspective lenses, accurate for moderate
    /// distortion; by Newton's method on the field angle for fisheyes.
    glm::dvec2 Undistort(glm::dvec2 distorted) const;

    /// The field angle in radians of the ray landing at normalised image coordinates `distorted`.
    double FieldAngle(glm::dvec2 distorted) const;

    /// The irradiance at normalised image coordinates `distorted` relative to the axis, from the
    /// aperture's foreshortening and the solid angle the image there sees:
    /// cos(theta) sin(theta) / (theta_d theta_d'), cos^4(theta) for a pinhole.
    double NaturalVignetting(glm::dvec2 distorted) const;
};

/// One channel of a sensor: its quantum efficiency (electrons per photon) over wavelength.
struct SensorChannel {
    std::string name;
    Spectrum quantumEfficiency;
};

/// An image sensor: channels with quantum efficiency curves, full well, noise, gain and ADC.
///
/// Electrons accumulate as photons times quantum efficiency, plus dark current; the readout adds
/// photon shot noise (Poisson), read noise (Gaussian), and per-pixel fixed patterns PRNU
/// (relative gain) and DSNU (offset); electrons clip at the full well, and the ADC converts them
/// to digital numbers with the ISO-dependent gain, adds the black level and clips to its bit
/// depth. The presets are generic curve shapes, not any particular sensor: replace the channels
/// with measured curves for a specific camera.
///
/// Reference: EMVA Standard 1288, "Standard for Characterization of Image Sensors and Cameras",
/// release 4.0, 2021 (the linear camera model).
struct SensorModel {
    std::vector<SensorChannel> channels;
    double fullWellElectrons = 20000.0;
    double readNoiseElectrons = 3.0;      ///< Standard deviation.
    double darkCurrentElectrons = 0.1;    ///< Per second.
    double prnu = 0.005;                  ///< Standard deviation of the relative gain.
    double dsnuElectrons = 0.5;           ///< Standard deviation of the per-pixel offset.
    double electronsPerDnAtIso100 = 0.0;  ///< 0 means full well over the ADC range.
    int adcBits = 12;
    double blackLevelDn = 0.0;
    bool noise = true;
    std::uint32_t seed = 0;

    /// Bayer-like red, green and blue channels behind an infrared cut filter.
    static SensorModel Rgb();

    /// One panchromatic channel with a front-illuminated silicon response, like the monochrome
    /// pushbroom cameras of lunar orbiters.
    static SensorModel Mono();

    /// Electrons per digital number at `iso`.
    double ElectronsPerDn(double iso) const;
};

/// Ernie's camera: a physically based pinhole camera for one image, with UsdGeomCamera's pose,
/// projection and exposure attributes, a spectral range, a lens and a sensor.
///
/// The pose `transform` maps camera to world coordinates; as in USD the camera looks down its -z
/// axis with +y up. Interactive controls such as `OrbitController` write it.
///
/// Focal length and apertures are read in millimetres. The image fits the vertical aperture:
/// pixels are square with a pitch of `verticalAperture / height`, and the horizontal aperture
/// follows from the image's aspect. Aperture offsets shift the principal point.
///
/// An `orthographic` camera follows UsdGeomCamera (and GfCamera): its apertures and their offsets
/// are in tenths of a scene unit, the size of the view in the scene, and it has no focal length;
/// every ray runs along the view direction. Its optics are telecentric, without fall-off off axis
/// or depth of field, and its pixels have the pitch `orthographicPixelPitchUm` on the sensor. Its
/// lens has no distortion. Distances to the camera are from its image plane. Terrain levels of
/// detail still follow the distance from the camera's position, so place it near what it sees.
///
/// Radiometry follows the camera equation: a scene radiance L reaches the sensor as irradiance
/// E = L pi t cos^4(theta) / (4 N^2) for an object far away, with transmission t, f-number N and
/// field angle theta; a pixel collects E times its area times the exposure time, converted to
/// photons and electrons per band. The photographic outputs instead scale luminance by USD's
/// linear exposure scale; with a one-channel (monochrome) sensor they are grey, weighting the
/// bands by that channel's quantum efficiency, scaled so a flat spectrum keeps its luminance.
///
/// References:
/// - B. K. P. Horn, "Robot Vision", MIT Press, 1986 (the image irradiance equation
///   E = L (pi / 4) (d / f)^2 cos^4(alpha), with N = f / d).
/// - Pixar, "UsdGeomCamera", https://openusd.org/release/api/class_usd_geom_camera.html
struct Camera {
    glm::dmat4 transform{1.0};  ///< Camera to world.
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    /// USD `projection` "orthographic": the apertures and offsets below are in tenths of a scene
    /// unit, and the focal length is unused.
    bool orthographic = false;
    double focalLengthMm = 50.0;
    double verticalApertureMm = 15.2908;
    double horizontalApertureOffsetMm = 0.0;
    double verticalApertureOffsetMm = 0.0;
    /// An orthographic camera's pixel pitch on its sensor, in micrometres, for the camera equation;
    /// a perspective camera's follows from its aperture.
    double orthographicPixelPitchUm = 5.0;
    double nearPlane = 1.0;
    double farPlane = 1e6;
    double exposureStops = 0.0;  ///< USD `exposure`: scales the photographic output by 2^stops.
    double exposureTime = 1.0;   ///< Seconds.
    double iso = 100.0;
    double fStop = 1.0;
    double responsivity = 1.0;
    /// USD `fStop`: the lens's f-number for depth of field; 0 is a pinhole, all in focus.
    double lensFStop = 0.0;
    double focusDistance = 0.0;  ///< USD `focusDistance`, scene units; 0 focuses at infinity.
    /// USD `shutter:open` and `shutter:close`: the interval each row integrates over, in frames
    /// relative to the frame's time, the last frame's at -1. Equal (the default) for no motion
    /// blur.
    double shutterOpen = 0.0;
    double shutterClose = 0.0;
    /// Sub-frames rendered across the shutter and the readout when either is open.
    int shutterSamples = 8;
    /// Rolling shutter readout: the top row's interval comes this many frames before the bottom
    /// row's, rows in between in proportion; 0 is a global shutter.
    double rollingShutter = 0.0;
    /// A pushbroom (line scan) camera (`ernie:camera:scan` "pushbroom"): one detector line along
    /// the image's x axis, `lineOffsetMm` off the principal point along y on the focal plane,
    /// read out once per image row as the camera moves. Row r (from the top) is exposed at the
    /// pose (r + 0.5) / height of the way from the last frame's pose to this frame's, so a frame
    /// is one strip of the scan. Its lens may distort radially (k1 to k3, with r^2 from the line's
    /// distance off the axis too), moving pixels along the line only; it has no tangential
    /// distortion, shutter interval or rolling readout of its own.
    ///
    /// Reference: R. Gupta and R. I. Hartley, "Linear Pushbroom Cameras", IEEE TPAMI 19(9):963-975,
    /// 1997.
    bool pushbroom = false;
    double lineOffsetMm = 0.0;
    SpectralBands bands;
    LensModel lens;
    SensorModel sensor = SensorModel::Rgb();

    /// The camera's position in the world.
    glm::dvec3 Eye() const {
        return glm::dvec3(transform[3]);
    }

    /// Unit viewing direction in the world, the transform's -z axis.
    glm::dvec3 Forward() const;

    /// Sets the vertical aperture so the image spans `radians` vertically at the focal length.
    void SetVerticalFov(double radians);

    /// The raster view: the inverse pose, and a projection with the focal length divided by
    /// `overscan` (so with lens distortion it covers the undistorted rays of every sensor pixel),
    /// the principal point and the clipping range. An orthographic camera's spans its apertures
    /// in the scene (`overscan` 1, as it has no distortion).
    CameraView RasterView(double overscan = 1.0) const;

    /// `RasterView` from camera-to-world `pose` in place of `transform`, such as one view of a
    /// batch, without copying the camera.
    CameraView RasterView(const glm::dmat4& pose, double overscan) const;

    /// USD's ratio between incident luminance and the normalised photographic output:
    /// responsivity * time * (iso / 100) * 2^exposure / fStop^2.
    double LinearExposureScale() const;

    /// Side of a pixel on the sensor, in metres.
    double PixelPitchMetres() const;

    /// Focal length in pixels, of a perspective camera.
    double FocalLengthPixels() const;

    /// An orthographic camera's side of a pixel in the scene, in scene units.
    double OrthographicPixelSize() const;

    /// The principal point in pixels, from the image's top-left corner.
    glm::dvec2 PrincipalPoint() const;

    /// The vertical field of view in radians.
    double VerticalFov() const;

    /// Irradiance at the sensor per unit scene radiance at normalised image coordinates
    /// (`x`, `y`) on the sensor: pi t cos^4(theta) / (4 N^2) for a pinhole, the lens's
    /// `NaturalVignetting` in place of cos^4, which is left out when vignetting is off.
    double IrradiancePerRadiance(glm::dvec2 normalised) const;

    /// Mean electrons a pixel at normalised coordinates `normalised` collects in channel
    /// `channel` from band radiances `radiance` (W m^-2 sr^-1 nm^-1, one per band), before dark
    /// current, noise and clipping.
    double Electrons(std::span<const float> radiance, std::size_t channel,
                     glm::dvec2 normalised = {0.0, 0.0}) const;

    /// Per-band factors f_i so that `Electrons` = sum_i radiance_i f_i at the image centre.
    std::vector<float> ElectronsPerRadiance(std::size_t channel) const;

    /// Per-band photons a pixel at the image centre collects over the exposure per unit spectral
    /// radiance (the camera equation: optics, pixel area, time and band width over the photon
    /// energy), before quantum efficiency: `ElectronsPerRadiance` without the channel's QE.
    std::vector<float> PhotonsPerRadiance() const;

    /// @throws std::invalid_argument for non-positive sizes, times, f-numbers, ISO or focal
    /// length, a bad clipping range, a degenerate transform, a bad spectral range, no sensor
    /// channels or more than four, or a shutter closing before it opens, a negative rolling
    /// readout or shutter samples outside 1 to `kMaxShutterSamples`, an orthographic camera with
    /// lens distortion or a fisheye or a non-positive pixel pitch, a pushbroom that is
    /// orthographic, has a fisheye or tangential distortion or more than one sub-frame, or a
    /// fisheye with
    /// Brown-Conrady distortion or a field reaching past `LensModel::kMaxFisheyeFieldAngle`.
    void Check() const;

    static constexpr int kMaxShutterSamples = 64;

    /// The sub-frames a frame renders: 1 with the shutter closed (open equal to close) and a
    /// global shutter, else `shutterSamples`, at least 2.
    int Subframes() const;

    /// The time of sub-frame `subframe` in frames relative to the frame's: evenly spaced from the
    /// top row's opening to the bottom row's closing, the last at `shutterClose`. Each row's
    /// exposure interpolates its interval linearly between them.
    double SubframeTime(int subframe) const;
};

/// The camera of the Camera prim at `path` for an image of `width` x `height`: its world
/// transform, and UsdGeomCamera's `focalLength`, `verticalAperture`, aperture offsets,
/// `clippingRange`, `exposure`, `exposure:time`, `exposure:iso`, `exposure:fStop` and
/// `exposure:responsivity`, and Ernie's `ernie:spectral:*`, `ernie:lens:*` and `ernie:sensor:*`
/// attributes (see `tokens.h`). Unset attributes keep `Camera`'s defaults.
/// @throws std::invalid_argument if the prim is not a Camera, its projection is not perspective,
/// or the result fails `Check`.
Camera CameraOf(const Scene& scene, const Path& path, std::uint32_t width, std::uint32_t height);

/// As `CameraOf`, with `transform` in place of the prim's world transform, such as an estimate for
/// a camera posed on the GPU, whose transform the CPU cannot read.
Camera CameraOf(const Scene& scene, const Path& path, std::uint32_t width, std::uint32_t height,
                const glm::dmat4& transform);

}  // namespace ernie
