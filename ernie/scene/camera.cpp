// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/camera.h"

#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

/// A Gaussian bump of height `peak` centred on `centre` nm with
/// standard deviation `width` nm, tabulated every 5 nm over 350-1100
/// nm.
Spectrum Bump(double centre, double width, double peak) {
    std::vector<double> wavelengths;
    std::vector<double> values;
    for (double w = 350.0; w <= 1100.0; w += 5.0) {
        const double t = (w - centre) / width;
        wavelengths.push_back(w);
        values.push_back(peak * std::exp(-0.5 * t * t));
    }
    return Spectrum(std::move(wavelengths), std::move(values));
}

}  // namespace

namespace {

/// The fisheye's radius theta_d of field angle `theta`, and its
/// derivative.
double FisheyeRadius(const std::array<double, 4>& k, double theta) {
    const double t2 = theta * theta;
    return theta *
           (1.0 +
            t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
}

double FisheyeSlope(const std::array<double, 4>& k, double theta) {
    const double t2 = theta * theta;
    return 1.0 + t2 * (3.0 * k[0] +
                       t2 * (5.0 * k[1] +
                             t2 * (7.0 * k[2] + t2 * 9.0 * k[3])));
}

/// The field angle of the fisheye ray landing at radius `rd`, by
/// Newton's method.
double FisheyeFieldAngle(const std::array<double, 4>& k, double rd) {
    double theta = rd;
    for (int i = 0; i < 20; ++i) {
        theta -=
            (FisheyeRadius(k, theta) - rd) / FisheyeSlope(k, theta);
    }
    return theta;
}

}  // namespace

glm::dvec2 LensModel::Distort(glm::dvec2 p) const {
    if (projection == LensProjection::Fisheye) {
        const double r = glm::length(p);
        return r > 0.0
                   ? p * (FisheyeRadius(fisheye, std::atan(r)) / r)
                   : p;
    }
    const double r2 = glm::dot(p, p);
    const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    return {p.x * radial + 2.0 * p1 * p.x * p.y +
                p2 * (r2 + 2.0 * p.x * p.x),
            p.y * radial + p1 * (r2 + 2.0 * p.y * p.y) +
                2.0 * p2 * p.x * p.y};
}

glm::dvec2 LensModel::Undistort(glm::dvec2 distorted) const {
    if (projection == LensProjection::Fisheye) {
        const double rd = glm::length(distorted);
        return rd > 0.0
                   ? distorted *
                         (std::tan(FisheyeFieldAngle(fisheye, rd)) /
                          rd)
                   : distorted;
    }
    glm::dvec2 p = distorted;
    for (int i = 0; i < 20; ++i) {
        p += distorted - Distort(p);
    }
    return p;
}

double LensModel::FieldAngle(glm::dvec2 distorted) const {
    if (projection == LensProjection::Fisheye) {
        return FisheyeFieldAngle(fisheye, glm::length(distorted));
    }
    return std::atan(glm::length(Undistort(distorted)));
}

double LensModel::NaturalVignetting(glm::dvec2 distorted) const {
    if (projection == LensProjection::Fisheye) {
        const double rd = glm::length(distorted);
        if (rd <= 0.0) {
            return 1.0;
        }
        const double theta = FisheyeFieldAngle(fisheye, rd);
        return std::abs(std::cos(theta)) * std::sin(theta) /
               (rd * FisheyeSlope(fisheye, theta));
    }
    const glm::dvec2 p = Undistort(distorted);
    const double cosine2 = 1.0 / (1.0 + glm::dot(p, p));
    return cosine2 * cosine2;
}

SensorModel SensorModel::Rgb() {
    SensorModel sensor;
    sensor.channels = {{"r", Bump(600.0, 32.0, 0.50)},
                       {"g", Bump(535.0, 36.0, 0.55)},
                       {"b", Bump(460.0, 28.0, 0.45)}};
    return sensor;
}

SensorModel SensorModel::Mono() {
    SensorModel sensor;
    sensor.channels = {
        {"mono", Spectrum({350.0, 400.0, 500.0, 600.0, 700.0, 800.0,
                           900.0, 1000.0, 1100.0},
                          {0.15, 0.35, 0.55, 0.62, 0.55, 0.40, 0.22,
                           0.06, 0.0})}};
    return sensor;
}

double SensorModel::ElectronsPerDn(double iso) const {
    const double maxDn = std::exp2(adcBits) - 1.0;
    const double atIso100 = electronsPerDnAtIso100 > 0.0
                                ? electronsPerDnAtIso100
                                : fullWellElectrons / maxDn;
    return atIso100 * 100.0 / iso;
}

double CameraParams::LinearExposureScale() const {
    return responsivity * exposureTime * (iso / 100.0) *
           std::exp2(exposureStops) / (fStop * fStop);
}

double CameraParams::PixelPitchMetres() const {
    return orthographic ? orthographicPixelPitchUm * 1e-6
                        : verticalApertureMm * 1e-3 / height;
}

double CameraParams::FocalLengthPixels() const {
    return focalLengthMm / (verticalApertureMm / height);
}

double CameraParams::OrthographicPixelSize() const {
    return 0.1 * verticalApertureMm / height;
}

glm::dvec2 CameraParams::PrincipalPoint() const {
    // Offsets share the apertures' units, millimetres or tenths of a
    // scene unit.
    const double pitch = verticalApertureMm / height;
    return {0.5 * width + horizontalApertureOffsetMm / pitch,
            0.5 * height - verticalApertureOffsetMm / pitch};
}

double CameraParams::VerticalFov() const {
    return 2.0 * std::atan(0.5 * verticalApertureMm / focalLengthMm);
}

double
CameraParams::IrradiancePerRadiance(glm::dvec2 normalised) const {
    double factor = M_PI * lens.transmission / (4.0 * fStop * fStop);
    // Telecentric optics see every pixel's scene head on.
    if (lens.vignetting && !orthographic) {
        factor *= lens.NaturalVignetting(normalised);
    }
    return factor;
}

std::vector<float>
CameraParams::ElectronsPerRadiance(std::size_t channel) const {
    const SensorChannel& sensorChannel = sensor.channels.at(channel);
    std::vector<float> factors = PhotonsPerRadiance();
    for (int i = 0; i < bands.count; ++i) {
        const double low = bands.Lower(i);
        const double qe = sensorChannel.quantumEfficiency.Average(
            low, low + bands.Width());
        factors[static_cast<std::size_t>(i)] = static_cast<float>(
            static_cast<double>(
                factors[static_cast<std::size_t>(i)]) *
            qe);
    }
    return factors;
}

std::vector<float> CameraParams::PhotonsPerRadiance() const {
    const double area = PixelPitchMetres() * PixelPitchMetres();
    const double optics = IrradiancePerRadiance({0.0, 0.0});
    std::vector<float> factors(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        const double photonsPerJoule =
            bands.Centre(i) * 1e-9 / (kPlanck * kSpeedOfLight);
        factors[static_cast<std::size_t>(i)] =
            static_cast<float>(optics * area * exposureTime *
                               bands.Width() * photonsPerJoule);
    }
    return factors;
}

double CameraParams::Electrons(std::span<const float> radiance,
                               std::size_t channel,
                               glm::dvec2 normalised) const {
    const std::vector<float> factors = ElectronsPerRadiance(channel);
    if (radiance.size() != factors.size()) {
        throw std::invalid_argument(
            "CameraParams::Electrons needs one radiance per band");
    }
    double electrons = 0.0;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        electrons += static_cast<double>(radiance[i]) *
                     static_cast<double>(factors[i]);
    }
    return electrons * IrradiancePerRadiance(normalised) /
           IrradiancePerRadiance({0.0, 0.0});
}

glm::dvec3 CameraParams::Forward() const {
    return glm::normalize(-glm::dvec3(transform[2]));
}

void CameraParams::SetVerticalFov(double radians) {
    verticalApertureMm =
        2.0 * focalLengthMm * std::tan(0.5 * radians);
}

CameraView CameraParams::RasterView(double overscan) const {
    return RasterView(transform, overscan);
}

CameraView CameraParams::RasterView(const glm::dmat4& pose,
                                    double overscan) const {
    const double w = width;
    const double h = height;
    const glm::dvec2 centre = PrincipalPoint();
    CameraView view;
    view.aspect = static_cast<float>(w / h);
    view.nearPlane = static_cast<float>(nearPlane);
    view.farPlane = static_cast<float>(farPlane);
    view.eye = glm::vec3(pose[3]);
    view.forward = glm::vec3(glm::normalize(-glm::dvec3(pose[2])));
    view.view = glm::mat4(glm::inverse(pose));
    if (orthographic) {
        // The image's edges in the view, the principal point on the
        // axis.
        const double size = OrthographicPixelSize();
        view.verticalFov = 0.0f;
        view.projection = glm::mat4(
            glm::orthoRH_ZO(-centre.x * size, (w - centre.x) * size,
                            -(h - centre.y) * size, centre.y * size,
                            nearPlane, farPlane));
        return view;
    }
    const double focal = FocalLengthPixels() / overscan;
    view.verticalFov =
        static_cast<float>(2.0 * std::atan(0.5 * h / focal));
    view.projection = glm::perspectiveRH_ZO(
        view.verticalFov, view.aspect, view.nearPlane, view.farPlane);
    view.projection[2][0] =
        static_cast<float>(1.0 - 2.0 * centre.x / w);
    view.projection[2][1] =
        static_cast<float>(2.0 * centre.y / h - 1.0);
    return view;
}

void CameraParams::Check() const {
    bands.Check();
    if (width == 0 || height == 0 || !(focalLengthMm > 0.0) ||
        !(verticalApertureMm > 0.0) || !(exposureTime > 0.0) ||
        !(iso > 0.0) || !(fStop > 0.0) || !(responsivity > 0.0)) {
        throw std::invalid_argument(
            "A camera needs positive image size, focal length, "
            "aperture, exposure time, ISO, f-number and "
            "responsivity");
    }
    if (!(nearPlane > 0.0) || !(farPlane > nearPlane)) {
        throw std::invalid_argument(
            "A camera needs a positive clipping range");
    }
    if (!(std::abs(glm::determinant(glm::dmat3(transform))) > 0.0)) {
        throw std::invalid_argument(
            "A camera needs an invertible transform");
    }
    if (sensor.channels.empty() || sensor.channels.size() > 4) {
        throw std::invalid_argument(
            "A sensor needs one to four channels");
    }
    if (sensor.adcBits < 1 || sensor.adcBits > 16 ||
        !(sensor.fullWellElectrons > 0.0) ||
        !(lens.transmission > 0.0)) {
        throw std::invalid_argument(
            "A sensor needs 1 to 16 ADC bits and a positive full "
            "well, "
            "and a lens a positive transmission");
    }
    if (!(shutterClose >= shutterOpen) || !(rollingShutter >= 0.0) ||
        shutterSamples < 1 || shutterSamples > kMaxShutterSamples) {
        throw std::invalid_argument(
            "A shutter needs close at or after open, a non-negative "
            "rolling readout and 1 to " +
            std::to_string(kMaxShutterSamples) + " samples");
    }
    if (pushbroom &&
        (orthographic ||
         lens.projection != LensProjection::Perspective ||
         Subframes() != 1)) {
        throw std::invalid_argument(
            "A pushbroom camera needs a perspective projection and "
            "no shutter interval or "
            "rolling readout: its rows are its readout");
    }
    if (orthographic &&
        (!lens.IsPinhole() || !(orthographicPixelPitchUm > 0.0))) {
        throw std::invalid_argument(
            "An orthographic camera needs a perspective lens without "
            "distortion and a positive "
            "pixel pitch");
    }
    if (!lens.chromaticScale.Empty()) {
        for (int i = 0; i < bands.count; ++i) {
            if (!(lens.chromaticScale.Average(
                      bands.Lower(i),
                      bands.Lower(i) + bands.Width()) > 0.0)) {
                throw std::invalid_argument(
                    "A lens's chromatic scale must be positive");
            }
        }
    }
    if (lens.rays != nullptr) {
        const LensRayTable& table = *lens.rays;
        if (table.width != width || table.height != height ||
            table.step == 0 ||
            table.columns !=
                (width + table.step - 1) / table.step + 1 ||
            table.rows !=
                (height + table.step - 1) / table.step + 1 ||
            table.rays.size() !=
                std::size_t(table.columns) * table.rows) {
            throw std::invalid_argument(
                "A lens's ray table needs the camera's image size "
                "and a ray every step pixels "
                "across it, corners included");
        }
        if (pushbroom) {
            throw std::invalid_argument(
                "A pushbroom camera's lens cannot be a ray table");
        }
        for (const glm::vec3& ray : table.rays) {
            if (!(std::acos(std::clamp(
                      -double(ray.z) / glm::length(glm::dvec3(ray)),
                      -1.0, 1.0)) <=
                  LensModel::kMaxFisheyeFieldAngle)) {
                throw std::invalid_argument(
                    "A lens's ray table reaches past " +
                    std::to_string(LensModel::kMaxFisheyeFieldAngle *
                                   180.0 / M_PI) +
                    " degrees off axis");
            }
        }
        return;
    }
    if (lens.projection == LensProjection::Fisheye) {
        if (lens.HasDistortion()) {
            throw std::invalid_argument(
                "A fisheye lens takes its distortion from its "
                "fisheye coefficients, not "
                "Brown-Conrady's");
        }
        // The widest ray is at a corner of the image; the fisheye's
        // radius must grow with the field angle up to it.
        const double f = FocalLengthPixels();
        const glm::dvec2 centre = PrincipalPoint();
        const double corner =
            glm::length(
                glm::dvec2(std::max(centre.x, width - centre.x),
                           std::max(centre.y, height - centre.y))) /
            f;
        double theta = 0.0;
        double reached = 0.0;
        constexpr int kSteps = 256;
        for (int i = 1; i <= kSteps && reached < corner; ++i) {
            theta = LensModel::kMaxFisheyeFieldAngle * i / kSteps;
            const double radius = FisheyeRadius(lens.fisheye, theta);
            if (!(FisheyeSlope(lens.fisheye, theta) > 0.0)) {
                break;
            }
            reached = radius;
        }
        if (reached < corner) {
            throw std::invalid_argument(
                "A fisheye camera's field reaches past " +
                std::to_string(LensModel::kMaxFisheyeFieldAngle *
                               180.0 / M_PI) +
                " degrees off axis, or its projection folds back; "
                "lengthen the focal length or "
                "shrink the aperture");
        }
    }
}

int CameraParams::Subframes() const {
    const bool moving =
        shutterClose > shutterOpen || rollingShutter > 0.0;
    return moving ? std::max(shutterSamples, 2) : 1;
}

double CameraParams::SubframeTime(int subframe) const {
    const int count = Subframes();
    const double start = shutterOpen - rollingShutter;
    return count == 1 ? shutterClose
                      : start + (shutterClose - start) * subframe /
                                    static_cast<double>(count - 1);
}

void SetSingletChromatic(LensModel& lens, double focalLengthMm,
                         double nd, double abbe, double lateral) {
    if (!(nd > 1.0) || !(abbe > 0.0) || !(focalLengthMm > 0.0) ||
        !(lateral >= 0.0) || !(lateral <= 1.0)) {
        throw std::invalid_argument(
            "A singlet's chromatic aberration needs an index above "
            "1, a "
            "positive Abbe number and focal length, and a lateral "
            "fraction from 0 to 1");
    }
    // The d, F and C lines (micrometres); Cauchy's n = A + B /
    // lambda^2 through them.
    constexpr double kD = 0.5875618;
    constexpr double kF = 0.4861327;
    constexpr double kC = 0.6562725;
    const double b =
        (nd - 1.0) / abbe / (1.0 / (kF * kF) - 1.0 / (kC * kC));
    const double a = nd - b / (kD * kD);
    std::vector<double> wavelengths;
    std::vector<double> scale;
    std::vector<double> focus;
    for (double nm = 300.0; nm <= 1100.0 + 1e-9; nm += 10.0) {
        const double um = nm * 1e-3;
        const double n = a + b / (um * um);
        const double ratio =
            (nd - 1.0) / (n - 1.0);  // f(lambda) / f_d
        wavelengths.push_back(nm);
        scale.push_back(1.0 + lateral * (ratio - 1.0));
        focus.push_back(focalLengthMm * (ratio - 1.0));
    }
    lens.chromaticScale = Spectrum(wavelengths, scale);
    lens.chromaticFocusMm = Spectrum(wavelengths, focus);
}

CameraParams CameraParamsOf(const Scene& scene, const Path& path,
                            std::uint32_t width,
                            std::uint32_t height) {
    if (!scene.Has(path) || scene.TypeOf(path) != tokens::kCamera) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not a Camera");
    }
    return CameraParamsOf(scene, path, width, height,
                          scene.WorldTransform(path));
}

CameraParams CameraParamsOf(const Scene& scene, const Path& path,
                            std::uint32_t width, std::uint32_t height,
                            const glm::dmat4& transform) {
    using namespace tokens;
    if (!scene.Has(path) || scene.TypeOf(path) != kCamera) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not a Camera");
    }
    const Camera prim(scene, path);
    const std::string projection = prim.GetProjection();
    if (projection != kPerspective && projection != kOrthographic) {
        throw std::invalid_argument(
            "Camera '" + path.String() + "' has projection '" +
            projection + "'; use perspective or orthographic");
    }
    CameraParams camera;
    camera.orthographic = projection == kOrthographic;
    camera.transform = transform;
    camera.width = width;
    camera.height = height;
    camera.focalLengthMm = prim.GetFocalLength();
    camera.verticalApertureMm = prim.GetVerticalAperture();
    camera.horizontalApertureOffsetMm =
        prim.GetHorizontalApertureOffset();
    camera.verticalApertureOffsetMm =
        prim.GetVerticalApertureOffset();
    const glm::dvec2 clipping = prim.GetClippingRange();
    camera.nearPlane = clipping.x;
    camera.farPlane = clipping.y;
    camera.exposureStops = prim.GetExposure();
    camera.exposureTime = prim.GetExposureTime();
    camera.iso = prim.GetExposureIso();
    camera.fStop = prim.GetExposureFStop();
    camera.responsivity = prim.GetExposureResponsivity();
    camera.lensFStop = std::max(prim.GetFStop(), 0.0);
    camera.focusDistance = std::max(prim.GetFocusDistance(), 0.0);
    camera.shutterOpen = prim.GetShutterOpen();
    camera.shutterClose = prim.GetShutterClose();
    camera.shutterSamples = prim.GetShutterSamples();
    camera.rollingShutter = prim.GetShutterRolling();
    const std::string scan = prim.GetScan();
    if (scan != kFrameScan && scan != kPushbroom) {
        throw std::invalid_argument("Camera '" + path.String() +
                                    "' has scan '" + scan +
                                    "'; use frame or pushbroom");
    }
    camera.pushbroom = scan == kPushbroom;
    camera.lineOffsetMm = prim.GetPushbroomLineOffset();

    const glm::dvec2 range = prim.GetSpectralRange();
    camera.bands = {range.x, range.y, prim.GetSpectralBands()};

    LensModel& lens = camera.lens;
    // Missing coefficients are 0.
    const std::vector<float> distortion = prim.GetLensDistortion();
    const auto at = [&distortion](std::size_t i) {
        return i < distortion.size() ? double(distortion[i]) : 0.0;
    };
    lens.k1 = at(0);
    lens.k2 = at(1);
    lens.k3 = at(2);
    lens.p1 = at(3);
    lens.p2 = at(4);
    const std::string lensProjection = prim.GetLensProjection();
    if (lensProjection == kFisheye) {
        lens.projection = LensProjection::Fisheye;
    } else if (lensProjection != kPerspective) {
        throw std::invalid_argument(
            "Camera '" + path.String() + "' has lens projection '" +
            lensProjection + "'; use perspective or fisheye");
    }
    const std::vector<float> fisheye = prim.GetLensFisheye();
    for (std::size_t i = 0; i < lens.fisheye.size(); ++i) {
        lens.fisheye[i] =
            i < fisheye.size() ? double(fisheye[i]) : 0.0;
    }
    lens.vignetting = prim.GetLensVignetting();
    lens.transmission = prim.GetLensTransmission();
    // Curves of (wavelength, value) pairs; none when unset.
    lens.chromaticScale =
        Spectrum::FromPairs(prim.GetLensChromaticScale());
    lens.chromaticFocusMm =
        Spectrum::FromPairs(prim.GetLensChromaticFocus());
    if (const Tensor* written = prim.GetLensRays()) {
        const Tensor rays = written->Contiguous();
        if (rays.Type() != DType::Float32 || rays.Rank() != 3 ||
            rays.Shape()[2] != 3 ||
            rays.Location().kind != DeviceKind::Cpu) {
            throw std::invalid_argument(
                "Camera '" + path.String() + "' " +
                std::string(kLensRays) +
                " must be a float32 (rows, columns, 3) CPU tensor");
        }
        auto table = std::make_shared<LensRayTable>();
        table->width = width;
        table->height = height;
        table->step =
            static_cast<std::uint32_t>(prim.GetLensRaysStep());
        table->rows = static_cast<std::uint32_t>(rays.Shape()[0]);
        table->columns = static_cast<std::uint32_t>(rays.Shape()[1]);
        const float* values = rays.As<float>();
        table->rays.resize(std::size_t(table->rows) * table->columns);
        for (std::size_t i = 0; i < table->rays.size(); ++i) {
            table->rays[i] = glm::vec3(
                values[3 * i], values[3 * i + 1], values[3 * i + 2]);
        }
        lens.rays = std::move(table);
    }

    const std::string type = prim.GetSensorType();
    if (type == "rgb") {
        camera.sensor = SensorModel::Rgb();
    } else if (type == "mono") {
        camera.sensor = SensorModel::Mono();
    } else {
        throw std::invalid_argument("Camera '" + path.String() +
                                    "' has sensor type '" + type +
                                    "'; use rgb or mono");
    }
    SensorModel& sensor = camera.sensor;
    for (SensorChannel& channel : sensor.channels) {
        if (const std::optional<std::vector<float>> pairs =
                prim.GetSensorQe(channel.name)) {
            channel.quantumEfficiency = Spectrum::FromPairs(*pairs);
        }
    }
    sensor.fullWellElectrons = prim.GetSensorFullWell();
    sensor.readNoiseElectrons = prim.GetSensorReadNoise();
    sensor.darkCurrentElectrons = prim.GetSensorDarkCurrent();
    sensor.prnu = prim.GetSensorPrnu();
    sensor.dsnuElectrons = prim.GetSensorDsnu();
    sensor.electronsPerDnAtIso100 = prim.GetSensorElectronsPerDn();
    sensor.adcBits = prim.GetSensorAdcBits();
    sensor.blackLevelDn = prim.GetSensorBlackLevel();
    camera.orthographicPixelPitchUm = prim.GetSensorPixelPitch();
    sensor.noise = prim.GetSensorNoise();
    sensor.seed = static_cast<std::uint32_t>(prim.GetSensorSeed());
    camera.Check();
    return camera;
}

}  // namespace ernie
