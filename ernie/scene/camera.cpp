#include "ernie/scene/camera.h"

#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

/// A Gaussian bump of height `peak` centred on `centre` nm with standard deviation `width` nm,
/// tabulated every 5 nm over 350-1100 nm.
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

const std::vector<float>* FloatArray(const Scene& scene, const Path& path,
                                     std::string_view attribute) {
    const Value* value = scene.Get(path, attribute);
    return value != nullptr ? value->Get<std::vector<float>>() : nullptr;
}

}  // namespace

namespace {

/// The fisheye's radius theta_d of field angle `theta`, and its derivative.
double FisheyeRadius(const std::array<double, 4>& k, double theta) {
    const double t2 = theta * theta;
    return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
}

double FisheyeSlope(const std::array<double, 4>& k, double theta) {
    const double t2 = theta * theta;
    return 1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])));
}

/// The field angle of the fisheye ray landing at radius `rd`, by Newton's method.
double FisheyeFieldAngle(const std::array<double, 4>& k, double rd) {
    double theta = rd;
    for (int i = 0; i < 20; ++i) {
        theta -= (FisheyeRadius(k, theta) - rd) / FisheyeSlope(k, theta);
    }
    return theta;
}

}  // namespace

glm::dvec2 LensModel::Distort(glm::dvec2 p) const {
    if (projection == LensProjection::Fisheye) {
        const double r = glm::length(p);
        return r > 0.0 ? p * (FisheyeRadius(fisheye, std::atan(r)) / r) : p;
    }
    const double r2 = glm::dot(p, p);
    const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    return {p.x * radial + 2.0 * p1 * p.x * p.y + p2 * (r2 + 2.0 * p.x * p.x),
            p.y * radial + p1 * (r2 + 2.0 * p.y * p.y) + 2.0 * p2 * p.x * p.y};
}

glm::dvec2 LensModel::Undistort(glm::dvec2 distorted) const {
    if (projection == LensProjection::Fisheye) {
        const double rd = glm::length(distorted);
        return rd > 0.0 ? distorted * (std::tan(FisheyeFieldAngle(fisheye, rd)) / rd) : distorted;
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
        return std::abs(std::cos(theta)) * std::sin(theta) / (rd * FisheyeSlope(fisheye, theta));
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
        {"mono", Spectrum({350.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0},
                          {0.15, 0.35, 0.55, 0.62, 0.55, 0.40, 0.22, 0.06, 0.0})}};
    return sensor;
}

double SensorModel::ElectronsPerDn(double iso) const {
    const double maxDn = std::exp2(adcBits) - 1.0;
    const double atIso100 =
        electronsPerDnAtIso100 > 0.0 ? electronsPerDnAtIso100 : fullWellElectrons / maxDn;
    return atIso100 * 100.0 / iso;
}

double Camera::LinearExposureScale() const {
    return responsivity * exposureTime * (iso / 100.0) * std::exp2(exposureStops) / (fStop * fStop);
}

double Camera::PixelPitchMetres() const {
    return orthographic ? orthographicPixelPitchUm * 1e-6 : verticalApertureMm * 1e-3 / height;
}

double Camera::FocalLengthPixels() const {
    return focalLengthMm / (verticalApertureMm / height);
}

double Camera::OrthographicPixelSize() const {
    return 0.1 * verticalApertureMm / height;
}

glm::dvec2 Camera::PrincipalPoint() const {
    // Offsets share the apertures' units, millimetres or tenths of a scene unit.
    const double pitch = verticalApertureMm / height;
    return {0.5 * width + horizontalApertureOffsetMm / pitch,
            0.5 * height - verticalApertureOffsetMm / pitch};
}

double Camera::VerticalFov() const {
    return 2.0 * std::atan(0.5 * verticalApertureMm / focalLengthMm);
}

double Camera::IrradiancePerRadiance(glm::dvec2 normalised) const {
    double factor = M_PI * lens.transmission / (4.0 * fStop * fStop);
    // Telecentric optics see every pixel's scene head on.
    if (lens.vignetting && !orthographic) {
        factor *= lens.NaturalVignetting(normalised);
    }
    return factor;
}

std::vector<float> Camera::ElectronsPerRadiance(std::size_t channel) const {
    const SensorChannel& sensorChannel = sensor.channels.at(channel);
    std::vector<float> factors = PhotonsPerRadiance();
    for (int i = 0; i < bands.count; ++i) {
        const double low = bands.Lower(i);
        const double qe = sensorChannel.quantumEfficiency.Average(low, low + bands.Width());
        factors[static_cast<std::size_t>(i)] =
            static_cast<float>(static_cast<double>(factors[static_cast<std::size_t>(i)]) * qe);
    }
    return factors;
}

std::vector<float> Camera::PhotonsPerRadiance() const {
    const double area = PixelPitchMetres() * PixelPitchMetres();
    const double optics = IrradiancePerRadiance({0.0, 0.0});
    std::vector<float> factors(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        const double photonsPerJoule = bands.Centre(i) * 1e-9 / (kPlanck * kSpeedOfLight);
        factors[static_cast<std::size_t>(i)] =
            static_cast<float>(optics * area * exposureTime * bands.Width() * photonsPerJoule);
    }
    return factors;
}

double Camera::Electrons(std::span<const float> radiance, std::size_t channel,
                         glm::dvec2 normalised) const {
    const std::vector<float> factors = ElectronsPerRadiance(channel);
    if (radiance.size() != factors.size()) {
        throw std::invalid_argument("Camera::Electrons needs one radiance per band");
    }
    double electrons = 0.0;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        electrons += static_cast<double>(radiance[i]) * static_cast<double>(factors[i]);
    }
    return electrons * IrradiancePerRadiance(normalised) / IrradiancePerRadiance({0.0, 0.0});
}

glm::dvec3 Camera::Forward() const {
    return glm::normalize(-glm::dvec3(transform[2]));
}

void Camera::SetVerticalFov(double radians) {
    verticalApertureMm = 2.0 * focalLengthMm * std::tan(0.5 * radians);
}

CameraView Camera::RasterView(double overscan) const {
    return RasterView(transform, overscan);
}

CameraView Camera::RasterView(const glm::dmat4& pose, double overscan) const {
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
        // The image's edges in the view, the principal point on the axis.
        const double size = OrthographicPixelSize();
        view.verticalFov = 0.0f;
        view.projection = glm::mat4(glm::orthoRH_ZO(-centre.x * size, (w - centre.x) * size,
                                                    -(h - centre.y) * size, centre.y * size,
                                                    nearPlane, farPlane));
        return view;
    }
    const double focal = FocalLengthPixels() / overscan;
    view.verticalFov = static_cast<float>(2.0 * std::atan(0.5 * h / focal));
    view.projection =
        glm::perspectiveRH_ZO(view.verticalFov, view.aspect, view.nearPlane, view.farPlane);
    view.projection[2][0] = static_cast<float>(1.0 - 2.0 * centre.x / w);
    view.projection[2][1] = static_cast<float>(2.0 * centre.y / h - 1.0);
    return view;
}

void Camera::Check() const {
    bands.Check();
    if (width == 0 || height == 0 || !(focalLengthMm > 0.0) || !(verticalApertureMm > 0.0) ||
        !(exposureTime > 0.0) || !(iso > 0.0) || !(fStop > 0.0) || !(responsivity > 0.0)) {
        throw std::invalid_argument("A camera needs positive image size, focal length, "
                                    "aperture, exposure time, ISO, f-number and responsivity");
    }
    if (!(nearPlane > 0.0) || !(farPlane > nearPlane)) {
        throw std::invalid_argument("A camera needs a positive clipping range");
    }
    if (!(std::abs(glm::determinant(glm::dmat3(transform))) > 0.0)) {
        throw std::invalid_argument("A camera needs an invertible transform");
    }
    if (sensor.channels.empty() || sensor.channels.size() > 4) {
        throw std::invalid_argument("A sensor needs one to four channels");
    }
    if (sensor.adcBits < 1 || sensor.adcBits > 16 || !(sensor.fullWellElectrons > 0.0) ||
        !(lens.transmission > 0.0)) {
        throw std::invalid_argument("A sensor needs 1 to 16 ADC bits and a positive full well, "
                                    "and a lens a positive transmission");
    }
    if (!(shutterClose >= shutterOpen) || !(rollingShutter >= 0.0) || shutterSamples < 1 ||
        shutterSamples > kMaxShutterSamples) {
        throw std::invalid_argument("A shutter needs close at or after open, a non-negative "
                                    "rolling readout and 1 to " +
                                    std::to_string(kMaxShutterSamples) + " samples");
    }
    if (pushbroom &&
        (orthographic || lens.projection != LensProjection::Perspective || lens.p1 != 0.0 ||
         lens.p2 != 0.0 || Subframes() != 1)) {
        throw std::invalid_argument(
            "A pushbroom camera needs a perspective projection, a lens with radial distortion "
            "at most, and no shutter interval or rolling readout: its rows are its readout");
    }
    if (orthographic && (!lens.IsPinhole() || !(orthographicPixelPitchUm > 0.0))) {
        throw std::invalid_argument(
            "An orthographic camera needs a perspective lens without distortion and a positive "
            "pixel pitch");
    }
    if (lens.rays != nullptr) {
        const LensRayTable& table = *lens.rays;
        if (table.width != width || table.height != height || table.step == 0 ||
            table.columns != (width + table.step - 1) / table.step + 1 ||
            table.rows != (height + table.step - 1) / table.step + 1 ||
            table.rays.size() != std::size_t(table.columns) * table.rows) {
            throw std::invalid_argument(
                "A lens's ray table needs the camera's image size and a ray every step pixels "
                "across it, corners included");
        }
        if (pushbroom) {
            throw std::invalid_argument("A pushbroom camera's lens cannot be a ray table");
        }
        for (const glm::vec3& ray : table.rays) {
            if (!(std::acos(std::clamp(-double(ray.z) / glm::length(glm::dvec3(ray)), -1.0, 1.0)) <=
                  LensModel::kMaxFisheyeFieldAngle)) {
                throw std::invalid_argument(
                    "A lens's ray table reaches past " +
                    std::to_string(LensModel::kMaxFisheyeFieldAngle * 180.0 / M_PI) +
                    " degrees off axis");
            }
        }
        return;
    }
    if (lens.projection == LensProjection::Fisheye) {
        if (lens.HasDistortion()) {
            throw std::invalid_argument(
                "A fisheye lens takes its distortion from its fisheye coefficients, not "
                "Brown-Conrady's");
        }
        // The widest ray is at a corner of the image; the fisheye's radius must grow with the
        // field angle up to it.
        const double f = FocalLengthPixels();
        const glm::dvec2 centre = PrincipalPoint();
        const double corner =
            glm::length(glm::dvec2(std::max(centre.x, width - centre.x),
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
                std::to_string(LensModel::kMaxFisheyeFieldAngle * 180.0 / M_PI) +
                " degrees off axis, or its projection folds back; lengthen the focal length or "
                "shrink the aperture");
        }
    }
}

int Camera::Subframes() const {
    const bool moving = shutterClose > shutterOpen || rollingShutter > 0.0;
    return moving ? std::max(shutterSamples, 2) : 1;
}

double Camera::SubframeTime(int subframe) const {
    const int count = Subframes();
    const double start = shutterOpen - rollingShutter;
    return count == 1 ? shutterClose
                      : start + (shutterClose - start) * subframe / static_cast<double>(count - 1);
}

Camera CameraOf(const Scene& scene, const Path& path, std::uint32_t width, std::uint32_t height) {
    if (!scene.Has(path) || scene.TypeOf(path) != tokens::kCamera) {
        throw std::invalid_argument("'" + path.String() + "' is not a Camera");
    }
    return CameraOf(scene, path, width, height, scene.WorldTransform(path));
}

Camera CameraOf(const Scene& scene, const Path& path, std::uint32_t width, std::uint32_t height,
                const glm::dmat4& transform) {
    using namespace tokens;
    if (!scene.Has(path) || scene.TypeOf(path) != kCamera) {
        throw std::invalid_argument("'" + path.String() + "' is not a Camera");
    }
    const std::string projection = GetToken(scene, path, kProjection, kPerspective);
    if (projection != kPerspective && projection != kOrthographic) {
        throw std::invalid_argument("Camera '" + path.String() + "' has projection '" +
                                    projection + "'; use perspective or orthographic");
    }
    Camera camera;
    camera.orthographic = projection == kOrthographic;
    camera.transform = transform;
    camera.width = width;
    camera.height = height;
    const auto number = [&](std::string_view attribute, double fallback) {
        return GetNumber(scene, path, attribute, fallback);
    };
    camera.focalLengthMm = number(kFocalLength, camera.focalLengthMm);
    camera.verticalApertureMm = number(kVerticalAperture, camera.verticalApertureMm);
    camera.horizontalApertureOffsetMm = number(kHorizontalApertureOffset, 0.0);
    camera.verticalApertureOffsetMm = number(kVerticalApertureOffset, 0.0);
    const glm::dvec2 clipping =
        GetVec2(scene, path, kClippingRange, {camera.nearPlane, camera.farPlane});
    camera.nearPlane = clipping.x;
    camera.farPlane = clipping.y;
    camera.exposureStops = number(kCameraExposure, 0.0);
    camera.exposureTime = number(kExposureTime, camera.exposureTime);
    camera.iso = number(kExposureIso, camera.iso);
    camera.fStop = number(kExposureFStop, camera.fStop);
    camera.responsivity = number(kExposureResponsivity, camera.responsivity);
    camera.lensFStop = std::max(number(kFStop, 0.0), 0.0);
    camera.focusDistance = std::max(number(kFocusDistance, 0.0), 0.0);
    camera.shutterOpen = number(kShutterOpen, camera.shutterOpen);
    camera.shutterClose = number(kShutterClose, camera.shutterClose);
    camera.shutterSamples = static_cast<int>(number(kShutterSamples, camera.shutterSamples));
    camera.rollingShutter = number(kShutterRolling, camera.rollingShutter);
    const std::string scan = GetToken(scene, path, kCameraScan, kFrameScan);
    if (scan != kFrameScan && scan != kPushbroom) {
        throw std::invalid_argument("Camera '" + path.String() + "' has scan '" + scan +
                                    "'; use frame or pushbroom");
    }
    camera.pushbroom = scan == kPushbroom;
    camera.lineOffsetMm = number(kPushbroomLineOffset, camera.lineOffsetMm);

    const glm::dvec2 range =
        GetVec2(scene, path, kSpectralRange, {camera.bands.minNm, camera.bands.maxNm});
    camera.bands = {range.x, range.y, static_cast<int>(number(kSpectralBands, camera.bands.count))};

    LensModel& lens = camera.lens;
    if (const std::vector<float>* k = FloatArray(scene, path, kLensDistortion)) {
        const auto at = [k](std::size_t i) { return i < k->size() ? double((*k)[i]) : 0.0; };
        lens.k1 = at(0);
        lens.k2 = at(1);
        lens.k3 = at(2);
        lens.p1 = at(3);
        lens.p2 = at(4);
    }
    const std::string lensProjection = GetToken(scene, path, kLensProjection, kPerspective);
    if (lensProjection == kFisheye) {
        lens.projection = LensProjection::Fisheye;
    } else if (lensProjection != kPerspective) {
        throw std::invalid_argument("Camera '" + path.String() + "' has lens projection '" +
                                    lensProjection + "'; use perspective or fisheye");
    }
    if (const std::vector<float>* k = FloatArray(scene, path, kLensFisheye)) {
        for (std::size_t i = 0; i < lens.fisheye.size(); ++i) {
            lens.fisheye[i] = i < k->size() ? double((*k)[i]) : 0.0;
        }
    }
    lens.vignetting = GetBool(scene, path, kLensVignetting, lens.vignetting);
    lens.transmission = number(kLensTransmission, lens.transmission);
    if (const Tensor* written = scene.Read(path, kLensRays)) {
        const Tensor rays = written->Contiguous();
        if (rays.Type() != DType::Float32 || rays.Rank() != 3 || rays.Shape()[2] != 3 ||
            rays.Location().kind != DeviceKind::Cpu) {
            throw std::invalid_argument("Camera '" + path.String() + "' " +
                                        std::string(kLensRays) +
                                        " must be a float32 (rows, columns, 3) CPU tensor");
        }
        auto table = std::make_shared<LensRayTable>();
        table->width = width;
        table->height = height;
        table->step = static_cast<std::uint32_t>(number(kLensRaysStep, 8.0));
        table->rows = static_cast<std::uint32_t>(rays.Shape()[0]);
        table->columns = static_cast<std::uint32_t>(rays.Shape()[1]);
        const float* values = rays.As<float>();
        table->rays.resize(std::size_t(table->rows) * table->columns);
        for (std::size_t i = 0; i < table->rays.size(); ++i) {
            table->rays[i] = glm::vec3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
        }
        lens.rays = std::move(table);
    }

    const std::string type = GetToken(scene, path, kSensorType, "rgb");
    if (type == "rgb") {
        camera.sensor = SensorModel::Rgb();
    } else if (type == "mono") {
        camera.sensor = SensorModel::Mono();
    } else {
        throw std::invalid_argument("Camera '" + path.String() + "' has sensor type '" + type +
                                    "'; use rgb or mono");
    }
    SensorModel& sensor = camera.sensor;
    for (SensorChannel& channel : sensor.channels) {
        const std::string attribute = std::string(kSensorQePrefix) + channel.name;
        if (const std::vector<float>* pairs = FloatArray(scene, path, attribute)) {
            channel.quantumEfficiency = Spectrum::FromPairs(*pairs);
        }
    }
    sensor.fullWellElectrons = number(kSensorFullWell, sensor.fullWellElectrons);
    sensor.readNoiseElectrons = number(kSensorReadNoise, sensor.readNoiseElectrons);
    sensor.darkCurrentElectrons = number(kSensorDarkCurrent, sensor.darkCurrentElectrons);
    sensor.prnu = number(kSensorPrnu, sensor.prnu);
    sensor.dsnuElectrons = number(kSensorDsnu, sensor.dsnuElectrons);
    sensor.electronsPerDnAtIso100 = number(kSensorGain, sensor.electronsPerDnAtIso100);
    sensor.adcBits = static_cast<int>(number(kSensorAdcBits, sensor.adcBits));
    sensor.blackLevelDn = number(kSensorBlackLevel, sensor.blackLevelDn);
    camera.orthographicPixelPitchUm = number(kSensorPixelPitch, camera.orthographicPixelPitchUm);
    sensor.noise = GetBool(scene, path, kSensorNoise, sensor.noise);
    sensor.seed = static_cast<std::uint32_t>(number(kSensorSeed, sensor.seed));
    camera.Check();
    return camera;
}

}  // namespace ernie
