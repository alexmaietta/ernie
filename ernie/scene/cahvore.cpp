// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/cahvore.h"

#include "ernie/util/least_squares.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ernie {

namespace {

constexpr double kEpsilon = 1e-15;

void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

// The CAHV image ray through `image` before distortion, and whether (V x H) . A is positive (cmod's
// direction convention).
glm::dvec3 CahvDirection(const CahvoreModel& m, glm::dvec2 image) {
    const glm::dvec3 f = m.v - image.y * m.a;
    const glm::dvec3 g = m.h - image.x * m.a;
    glm::dvec3 direction = glm::cross(f, g);
    const double length = glm::length(direction);
    Require(length > kEpsilon, "The CAHV model's ray through this point is degenerate");
    direction /= length;
    if (glm::dot(glm::cross(m.v, m.h), m.a) < 0.0) {
        direction = -direction;
    }
    return direction;
}

glm::dvec2 CahvProject(const CahvoreModel& m, const glm::dvec3& p) {
    const double alpha = glm::dot(p, m.a);
    return {glm::dot(p, m.h) / alpha, glm::dot(p, m.v) / alpha};
}

// CAHVORE's projection of direction `p` from C, its entrance pupil's shift left out: the field
// angle theta from O, chi = tan(L theta) / L for linearity L (theta for 0, sin for L < 0), then
// the distorted chi' = chi (1 + rho0 + rho1 chi^2 + rho2 chi^4) along the image ray O + chi' l.
glm::dvec2 CahvoreProjectCentral(const CahvoreModel& m, const glm::dvec3& p) {
    const double zeta = glm::dot(p, m.o);
    const glm::dvec3 lambda = p - zeta * m.o;
    const double lambdaLength = glm::length(lambda);
    const double theta = std::atan2(lambdaLength, zeta);
    const double l = m.Linearity();
    double chi = theta;
    if (l > kEpsilon) {
        chi = std::tan(l * theta) / l;
    } else if (l < -kEpsilon) {
        chi = std::sin(l * theta) / l;
    }
    const double chi2 = chi * chi;
    const double chiP = chi * (1.0 + m.r.x + chi2 * (m.r.y + chi2 * m.r.z));
    const glm::dvec3 unit = lambdaLength > kEpsilon ? lambda / lambdaLength : glm::dvec3(0.0);
    return CahvProject(m, m.o + chiP * unit);
}

}  // namespace

double CahvoreModel::Linearity() const {
    return pupilType == 1 ? 1.0 : pupilType == 2 ? 0.0 : linearity;
}

CahvoreModel::Ray CahvoreModel::ImageToRay(glm::dvec2 image) const {
    if (kind == Kind::Cahv) {
        return {c, CahvDirection(*this, image)};
    }
    if (kind == Kind::Cahvor) {
        // cmod_cahvor_2d_to_3d: the CAHV ray, its radial distortion removed by solving
        // k5 u^5 + k3 u^3 + k1 u = 1 for u by Newton's method.
        const glm::dvec3 rr = CahvDirection(*this, image);
        const double omega = glm::dot(rr, o);
        Require(omega * omega > kEpsilon, "The CAHVOR ray is perpendicular to its optical axis");
        const glm::dvec3 lambda = rr - omega * o;
        const double tau = glm::dot(lambda, lambda) / (omega * omega);
        const double k1 = 1.0 + r.x;
        const double k3 = r.y * tau;
        const double k5 = r.z * tau * tau;
        double u = 1.0 - (r.x + k3 + k5);
        for (int i = 0; i < 20; ++i) {
            const double u2 = u * u;
            const double poly = ((k5 * u2 + k3) * u2 + k1) * u - 1.0;
            const double deriv = (5.0 * k5 * u2 + 3.0 * k3) * u2 + k1;
            Require(deriv > kEpsilon, "The CAHVOR distortion is too negative here");
            const double du = poly / deriv;
            u -= du;
            if (std::abs(du) < 1e-8) {
                break;
            }
        }
        return {c, glm::normalize(rr - (1.0 - u) * lambda)};
    }
    // cmod_cahvore_2d_to_3d_general.
    const glm::dvec3 w = glm::cross(v - image.y * a, h - image.x * a);
    const double avh = glm::dot(a, glm::cross(v, h));
    Require(std::abs(avh) > kEpsilon, "The CAHVORE model's A, V and H are degenerate");
    const glm::dvec3 rp = w / avh;
    const double zetap = glm::dot(rp, o);
    Require(std::abs(zetap) > kEpsilon, "The CAHVORE ray is perpendicular to its optical axis");
    const glm::dvec3 lambdap = rp - zetap * o;
    const double chip = glm::length(lambdap) / zetap;
    if (chip < 1e-8) {
        return {c, o};
    }
    // chi from chi' = (1 + rho0) chi + rho1 chi^3 + rho2 chi^5, by Newton's method.
    double chi = chip;
    for (int n = 0; n < 100; ++n) {
        const double chi2 = chi * chi;
        const double deriv = 1.0 + r.x + 3.0 * r.y * chi2 + 5.0 * r.z * chi2 * chi2;
        Require(std::abs(deriv) > kEpsilon, "The CAHVORE distortion folds here");
        const double dchi =
            ((1.0 + r.x) * chi + r.y * chi2 * chi + r.z * chi2 * chi2 * chi - chip) / deriv;
        chi -= dchi;
        if (std::abs(dchi) < 1e-8) {
            break;
        }
    }
    // The incoming ray's field angle, and the entrance pupil's shift along O with it.
    const double l = Linearity();
    double theta = chi;
    if (l < -kEpsilon) {
        theta = std::asin(l * chi) / l;
    } else if (l > kEpsilon) {
        theta = std::atan(l * chi) / l;
    }
    const double theta2 = theta * theta;
    const double sine = std::sin(theta);
    Require(std::abs(sine) > kEpsilon, "The CAHVORE field angle is degenerate");
    const double shift = (theta / sine - 1.0) * (e.x + e.y * theta2 + e.z * theta2 * theta2);
    return {c + shift * o, glm::normalize(glm::normalize(lambdap) * sine + o * std::cos(theta))};
}

glm::dvec2 CahvoreModel::PointToImage(const glm::dvec3& point) const {
    const glm::dvec3 p = point - c;
    if (kind == Kind::Cahv) {
        return CahvProject(*this, p);
    }
    if (kind == Kind::Cahvor) {
        // cmod_cahvor_3d_to_2d: p moved off O by mu lambda, mu = rho0 + rho1 tau + rho2 tau^2.
        const double omega = glm::dot(p, o);
        const glm::dvec3 lambda = p - omega * o;
        const double tau = glm::dot(lambda, lambda) / (omega * omega);
        const double mu = r.x + tau * (r.y + tau * r.z);
        return CahvProject(*this, p + mu * lambda);
    }
    // CAHVORE: from the projection without the pupil's shift, Newton's method on the image point
    // whose ray passes through `point`, the miss measured as angles across the ray.
    glm::dvec2 image = CahvoreProjectCentral(*this, p);
    for (int iteration = 0; iteration < 50; ++iteration) {
        const Ray ray = ImageToRay(image);
        const glm::dvec3 helper =
            std::abs(ray.direction.x) < 0.9 ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 1.0, 0.0);
        const glm::dvec3 e1 = glm::normalize(glm::cross(ray.direction, helper));
        const glm::dvec3 e2 = glm::cross(ray.direction, e1);
        const auto miss = [&](glm::dvec2 at) {
            const Ray r2 = ImageToRay(at);
            const glm::dvec3 toPoint = glm::normalize(point - r2.origin);
            return glm::dvec2(glm::dot(toPoint - r2.direction, e1),
                              glm::dot(toPoint - r2.direction, e2));
        };
        const glm::dvec2 residual = miss(image);
        if (glm::length(residual) < 1e-14) {
            break;
        }
        constexpr double kStep = 1e-3;  // pixels
        const glm::dvec2 dx = (miss(image + glm::dvec2(kStep, 0.0)) - residual) / kStep;
        const glm::dvec2 dy = (miss(image + glm::dvec2(0.0, kStep)) - residual) / kStep;
        const glm::dmat2 jacobian(dx, dy);
        if (std::abs(glm::determinant(jacobian)) < 1e-300) {
            break;
        }
        const glm::dvec2 step = glm::inverse(jacobian) * residual;
        image -= step;
        if (glm::length(step) < 1e-10) {
            break;
        }
    }
    return image;
}

CahvoreModel::ErnieCamera CahvoreModel::ToCamera(double pixelPitchUm) const {
    Require(width > 0 && height > 0, "The camera model has no image size");
    Require(pixelPitchUm > 0.0, "The pixel pitch must be positive");
    // The CAHV image plane: H = hs H' + hc A and V = vs V' + vc A, with focal lengths hs and vs in
    // pixels and image centre (hc, vc).
    const glm::dvec3 axis = glm::normalize(a);
    const double hc = glm::dot(h, axis);
    const double vc = glm::dot(v, axis);
    const glm::dvec3 hPrime = h - hc * axis;
    const glm::dvec3 vPrime = v - vc * axis;
    const double hs = glm::length(hPrime);
    const double vs = glm::length(vPrime);
    Require(hs > kEpsilon && vs > kEpsilon, "The camera model's H or V lies along A");
    // A physical camera's image axes are all but perpendicular.
    const double skew = std::abs(glm::dot(hPrime / hs, vPrime / vs));
    if (skew > std::sin(M_PI / 180.0)) {
        throw std::invalid_argument("The camera model's image axes are " +
                                    std::to_string(std::acos(skew) * 180.0 / M_PI) +
                                    " degrees apart: not a physical camera's");
    }
    // Ernie's axes on the image plane's: z back from the scene (-A), x along H', y up. The
    // distortion is radial about O instead, a fraction of a degree off A: Ernie's lenses, centred
    // on the image plane's normal, take that tilt as a residual of pixels at long focal lengths.
    const glm::dvec3 x = hPrime / hs;
    const glm::dvec3 z = -axis;
    const glm::dvec3 y = glm::cross(z, x);

    ErnieCamera out;
    // A model whose V runs up the image against H x A (as cmod allows) is a mirror of Ernie's
    // camera: its rows are flipped.
    out.rowsReversed = glm::dot(y, -vPrime) < 0.0;
    Camera& camera = out.camera;
    const double pitch = pixelPitchUm * 1e-3;  // mm
    const double f = vs;                       // pixels
    camera.width = width;
    camera.height = height;
    camera.focalLengthMm = f * pitch;
    camera.verticalApertureMm = pitch * height;
    // Ernie's pixel of model image point `image`: centres at + 0.5, the rows flipped for a mirror.
    const auto ernieOf = [&](glm::dvec2 image) {
        return glm::dvec2(image.x + 0.5,
                          out.rowsReversed ? height - (image.y + 0.5) : image.y + 0.5);
    };
    const glm::dvec2 principal = ernieOf({hc, vc});
    camera.horizontalApertureOffsetMm = (principal.x - 0.5 * width) * pitch;
    camera.verticalApertureOffsetMm = (0.5 * height - principal.y) * pitch;
    camera.transform =
        glm::dmat4(glm::dvec4(x, 0.0), glm::dvec4(y, 0.0), glm::dvec4(z, 0.0), glm::dvec4(c, 1.0));
    camera.nearPlane = 0.01;
    camera.farPlane = 1e4;

    // The model's rays over the image, in Ernie's camera axes, against where they land.
    struct Sample {
        glm::dvec3 ray;
        glm::dvec2 distorted;  // Ernie's normalised image coordinates, y up
        glm::dvec2 pixel;      // Ernie's pixel coordinates
    };
    std::vector<Sample> samples;
    constexpr int kAcross = 40;
    constexpr int kDown = 30;
    for (int j = 0; j <= kDown; ++j) {
        for (int i = 0; i <= kAcross; ++i) {
            const glm::dvec2 image((width - 1.0) * i / kAcross, (height - 1.0) * j / kDown);
            const Ray ray = ImageToRay(image);
            const glm::dvec3 local(glm::dot(ray.direction, x), glm::dot(ray.direction, y),
                                   glm::dot(ray.direction, z));
            const glm::dvec2 pixel = ernieOf(image);
            samples.push_back(
                {local, glm::dvec2(pixel.x - principal.x, principal.y - pixel.y) / f, pixel});
            out.pupilShiftMax = std::max(out.pupilShiftMax, glm::length(ray.origin - c));
            out.fieldAngleMax =
                std::max(out.fieldAngleMax, std::acos(std::clamp(-local.z, -1.0, 1.0)));
        }
    }

    double scale = 1.0;
    if (kind == Kind::Cahvore) {
        // Kannala-Brandt: the distorted radius theta (a + b1 theta^2 + ... + b4 theta^8), a into
        // the focal length, k = b / a.
        camera.lens.projection = LensProjection::Fisheye;
        NormalEquations<5> fit;
        for (const Sample& s : samples) {
            const double theta = std::acos(std::clamp(-s.ray.z, -1.0, 1.0));
            const double t2 = theta * theta;
            fit.Add({theta, theta * t2, theta * t2 * t2, theta * t2 * t2 * t2,
                     theta * t2 * t2 * t2 * t2},
                    glm::length(s.distorted));
        }
        const std::array<double, 5> b = fit.Solve();
        scale = b[0] != 0.0 ? b[0] : 1.0;
        for (int k = 0; k < 4; ++k) {
            camera.lens.fisheye[static_cast<std::size_t>(k)] =
                b[static_cast<std::size_t>(k) + 1] / scale;
        }
    } else {
        // Brown-Conrady, radial and tangential, on the pinhole coordinates, as the ISD fits.
        NormalEquations<6> fit;
        const auto basis = [](glm::dvec2 u, int axisIndex) {
            const double r2 = glm::dot(u, u);
            const double value = u[axisIndex];
            const double crossTerm = 2.0 * u.x * u.y;
            const double own = r2 + 2.0 * value * value;
            return std::array<double, 6>{value,
                                         value * r2,
                                         value * r2 * r2,
                                         value * r2 * r2 * r2,
                                         axisIndex == 0 ? crossTerm : own,
                                         axisIndex == 0 ? own : crossTerm};
        };
        for (const Sample& s : samples) {
            const glm::dvec2 u = glm::dvec2(s.ray) / -s.ray.z;
            for (int axisIndex = 0; axisIndex < 2; ++axisIndex) {
                fit.Add(basis(u, axisIndex), s.distorted[axisIndex]);
            }
        }
        const std::array<double, 6> b = fit.Solve();
        scale = b[0] != 0.0 ? b[0] : 1.0;
        camera.lens.k1 = b[1] / scale;
        camera.lens.k2 = b[2] / scale;
        camera.lens.k3 = b[3] / scale;
        camera.lens.p1 = b[4] / scale;
        camera.lens.p2 = b[5] / scale;
    }
    camera.focalLengthMm = f * scale * pitch;

    // How far Ernie's camera puts each sample from the model's pixel.
    const double fitted = f * scale;
    for (const Sample& s : samples) {
        glm::dvec2 d;
        if (kind == Kind::Cahvore) {
            const double theta = std::acos(std::clamp(-s.ray.z, -1.0, 1.0));
            const double t2 = theta * theta;
            const std::array<double, 4>& k = camera.lens.fisheye;
            const double radius =
                theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
            const double across = glm::length(glm::dvec2(s.ray));
            d = across > kEpsilon ? glm::dvec2(s.ray) / across * radius : glm::dvec2(0.0);
        } else {
            d = camera.lens.Distort(glm::dvec2(s.ray) / -s.ray.z);
        }
        const glm::dvec2 pixel = principal + glm::dvec2(d.x, -d.y) * fitted;
        out.residualPixels = std::max(out.residualPixels, glm::length(pixel - s.pixel));
    }
    return out;
}

LensRayTable CahvoreModel::RayTable(std::uint32_t step) const {
    Require(step > 0, "A ray table's step must be positive");
    const ErnieCamera fit = ToCamera(1.0);
    const glm::dmat3 toCamera = glm::transpose(glm::dmat3(fit.camera.transform));
    LensRayTable table;
    table.width = width;
    table.height = height;
    table.step = step;
    table.columns = (width + step - 1) / step + 1;
    table.rows = (height + step - 1) / step + 1;
    for (std::uint32_t j = 0; j < table.rows; ++j) {
        for (std::uint32_t i = 0; i < table.columns; ++i) {
            // Ernie's pixel position, from the top left, to the model's image coordinates.
            const double px = double(i) * step;
            const double py = double(j) * step;
            const glm::dvec2 image(px - 0.5, (fit.rowsReversed ? height - py : py) - 0.5);
            table.rays.push_back(glm::vec3(toCamera * ImageToRay(image).direction));
        }
    }
    return table;
}

namespace {

glm::dvec3 VectorOf(const nlohmann::json& model, const char* name) {
    const auto it = model.find(name);
    if (it == model.end() || !it->is_array() || it->size() != 3) {
        throw std::invalid_argument(std::string("The camera model's ") + name +
                                    " is not a 3-vector");
    }
    return {(*it)[0].get<double>(), (*it)[1].get<double>(), (*it)[2].get<double>()};
}

}  // namespace

std::vector<NamedCahvore> ReadCahvoreModels(std::string_view text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(std::string("Malformed camera model JSON: ") + error.what());
    }
    if (!json.is_array()) {
        throw std::invalid_argument("Camera model JSON must be an array of named models");
    }
    std::vector<NamedCahvore> models;
    try {
        for (const nlohmann::json& entry : json) {
            NamedCahvore named;
            named.name = entry.at("name").get<std::string>();
            named.frame = entry.value("frame", std::string());
            const nlohmann::json& m = entry.at("model");
            const std::string type = m.at("type").get<std::string>();
            CahvoreModel& model = named.model;
            if (type == "cahv") {
                model.kind = CahvoreModel::Kind::Cahv;
            } else if (type == "cahvor") {
                model.kind = CahvoreModel::Kind::Cahvor;
            } else if (type == "cahvore") {
                model.kind = CahvoreModel::Kind::Cahvore;
            } else {
                throw std::invalid_argument("Camera model '" + named.name + "' has type '" + type +
                                            "'; use cahv, cahvor or cahvore");
            }
            model.c = VectorOf(m, "C");
            model.a = VectorOf(m, "A");
            model.h = VectorOf(m, "H");
            model.v = VectorOf(m, "V");
            if (model.kind != CahvoreModel::Kind::Cahv) {
                model.o = VectorOf(m, "O");
                model.r = VectorOf(m, "R");
            }
            if (model.kind == CahvoreModel::Kind::Cahvore) {
                model.e = VectorOf(m, "E");
                model.pupilType = m.value("pupilType", 3);
                model.linearity = m.value("linearity", 1.0);
            }
            model.width = m.at("width").get<std::uint32_t>();
            model.height = m.at("height").get<std::uint32_t>();
            models.push_back(std::move(named));
        }
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(
            std::string("A camera model has a missing or mistyped field: ") + error.what());
    }
    return models;
}

std::vector<NamedCahvore> ReadCahvoreModelsFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read the camera models '" + path + "'");
    }
    std::ostringstream text;
    text << file.rdbuf();
    return ReadCahvoreModels(text.str());
}

}  // namespace ernie
