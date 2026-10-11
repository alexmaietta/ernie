// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ernie::testing {

inline constexpr double kPi = 3.14159265358979323846;

/// One evaluation: cos(i), cos(e), cos(g), cos(psi) and the albedo
/// input.
struct Geometry {
    double mu0;
    double mu;
    double cosG;
    double cosPsi;
    glm::dvec3 albedo;
};

/// Grid of incidence, emission and azimuth angles with a coloured
/// albedo.
inline std::vector<Geometry> GeometryGrid(glm::dvec3 albedo = {
                                              0.12, 0.35, 0.8}) {
    std::vector<Geometry> grid;
    for (const double i : {0.0, 10.0, 30.0, 50.0, 70.0, 85.0}) {
        for (const double e : {0.0, 15.0, 30.0, 60.0, 80.0}) {
            for (const double psi : {0.0, 45.0, 90.0, 135.0, 180.0}) {
                const double ri = i * kPi / 180.0;
                const double re = e * kPi / 180.0;
                const double rp = psi * kPi / 180.0;
                grid.push_back(
                    {std::cos(ri), std::cos(re),
                     std::cos(ri) * std::cos(re) +
                         std::sin(ri) * std::sin(re) * std::cos(rp),
                     std::cos(rp), albedo});
            }
        }
    }
    return grid;
}

inline double HenyeyGreenstein(double xi, double cosG) {
    return (1.0 - xi * xi) /
           std::pow(1.0 + 2.0 * xi * cosG + xi * xi, 1.5);
}

inline double HapkeH(double w, double x) {
    const double gamma = std::sqrt(1.0 - w);
    const double r0 = (1.0 - gamma) / (1.0 + gamma);
    return 1.0 / (1.0 - w * x *
                            (r0 + 0.5 * (1.0 - 2.0 * r0 * x) *
                                      std::log((1.0 + x) / x)));
}

/// Reference Hapke model (see brdf_hapke.slang), written from the
/// published formulas in double precision with the shader's guards on
/// degenerate angles.
inline double HapkeReference(const Geometry& g, double w,
                             const std::vector<float>& values) {
    const std::vector<double> p(values.begin(), values.end());
    const double b = p[0];
    const double c = p[1];
    const double bs0 = p[2];
    const double hs = p[3];
    const double thetaBar = p[4] * kPi / 180.0;
    const double phi = std::clamp(p[5], 1e-6, 0.74);
    const double kArg = 1.209 * std::pow(phi, 2.0 / 3.0);
    const double k = -std::log(1.0 - kArg) / kArg;

    double mu0e = g.mu0;
    double mue = g.mu;
    double s = 1.0;
    if (thetaBar >= 1e-4) {
        const double tanT = std::tan(thetaBar);
        const double chi = 1.0 / std::sqrt(1.0 + kPi * tanT * tanT);
        const double i = std::max(std::acos(g.mu0), 1e-4);
        const double e = std::max(std::acos(g.mu), 1e-4);
        const double psi = std::min(
            std::acos(std::clamp(g.cosPsi, -1.0, 1.0)), kPi - 1e-4);
        const double f = std::exp(-2.0 * std::tan(psi / 2.0));
        const double sin2 = std::pow(std::sin(psi / 2.0), 2.0);
        auto e1 = [&](double x) {
            return std::exp(-2.0 / (kPi * tanT * std::tan(x)));
        };
        auto e2 = [&](double x) {
            return std::exp(-1.0 / (kPi * tanT * tanT * std::tan(x) *
                                    std::tan(x)));
        };
        auto eta = [&](double x) {
            return chi * (std::cos(x) +
                          std::sin(x) * tanT * e2(x) / (2.0 - e1(x)));
        };
        if (i <= e) {
            const double d = 2.0 - e1(e) - psi / kPi * e1(i);
            mu0e = chi *
                   (std::cos(i) +
                    std::sin(i) * tanT *
                        (std::cos(psi) * e2(e) + sin2 * e2(i)) / d);
            mue =
                chi * (std::cos(e) + std::sin(e) * tanT *
                                         (e2(e) - sin2 * e2(i)) / d);
            s = mue / eta(e) * g.mu0 / eta(i) * chi /
                (1.0 - f + f * chi * g.mu0 / eta(i));
        } else {
            const double d = 2.0 - e1(i) - psi / kPi * e1(e);
            mu0e =
                chi * (std::cos(i) + std::sin(i) * tanT *
                                         (e2(i) - sin2 * e2(e)) / d);
            mue = chi *
                  (std::cos(e) +
                   std::sin(e) * tanT *
                       (std::cos(psi) * e2(i) + sin2 * e2(e)) / d);
            s = mue / eta(e) * g.mu0 / eta(i) * chi /
                (1.0 - f + f * chi * g.mu / eta(e));
        }
    }
    const double cosG = g.cosG;
    const double gAngle = std::acos(std::clamp(cosG, -1.0, 1.0));
    const double phase =
        0.5 * (1.0 + c) * (1.0 - b * b) /
            std::pow(1.0 - 2.0 * b * cosG + b * b, 1.5) +
        0.5 * (1.0 - c) * (1.0 - b * b) /
            std::pow(1.0 + 2.0 * b * cosG + b * b, 1.5);
    const double shoe = bs0 / (1.0 + std::tan(gAngle / 2.0) / hs);
    const double multiple =
        HapkeH(w, mu0e / k) * HapkeH(w, mue / k) - 1.0;
    return k * w / 4.0 * mu0e / (mu0e + mue) *
           (phase * (1.0 + shoe) + multiple) * s;
}

/// The preview surface's I/F for one channel of base colour `albedo`,
/// written from explicit light, view and half vectors (glTF 2.0
/// Appendix B), with unresolved slope variance `variance`: parameters
/// roughness, metallic, ior, useSpecularWorkflow, specularColor (3),
/// clearcoat, clearcoatRoughness.
inline double PreviewSurfaceReference(const Geometry& g,
                                      double albedo,
                                      const std::vector<float>& p,
                                      double variance = 0.0) {
    const double sinI = std::sqrt(std::max(0.0, 1.0 - g.mu0 * g.mu0));
    const double sinE = std::sqrt(std::max(0.0, 1.0 - g.mu * g.mu));
    const double sinPsi =
        std::sqrt(std::max(0.0, 1.0 - g.cosPsi * g.cosPsi));
    const glm::dvec3 n(0.0, 0.0, 1.0);
    const glm::dvec3 l(sinI, 0.0, g.mu0);
    const glm::dvec3 v(sinE * g.cosPsi, sinE * sinPsi, g.mu);
    const glm::dvec3 h = glm::normalize(l + v);
    const double nh = glm::dot(n, h);
    const double vh = glm::dot(v, h);
    const auto lobe = [&](double roughness) {
        const double alpha = roughness * roughness;
        const double a2 =
            std::clamp(alpha * alpha + 2.0 * variance, 1e-6, 1.0);
        const double d =
            a2 / (kPi * std::pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
        const double vis =
            0.5 / (g.mu0 * std::sqrt(g.mu * g.mu * (1.0 - a2) + a2) +
                   g.mu * std::sqrt(g.mu0 * g.mu0 * (1.0 - a2) + a2));
        return d * vis;
    };
    const auto schlick = [&](double f0) {
        return f0 + (1.0 - f0) * std::pow(1.0 - vh, 5.0);
    };
    const double specular = lobe(p[0]);
    // Turquin's multiple-scattering compensation, 1 + f0 (1 / E - 1)
    // with E from Lazarov's fit at the lobe's (widened) roughness, as
    // the geometric mean over cos(i) and cos(e).
    const double alpha = static_cast<double>(p[0]) * p[0];
    const double lobeRoughness = std::pow(
        std::clamp(alpha * alpha + 2.0 * variance, 1e-6, 1.0), 0.25);
    const auto albedoE = [&](double mu) {
        const glm::dvec4 r =
            lobeRoughness * glm::dvec4(-1.0, -0.0275, -0.572, 0.022) +
            glm::dvec4(1.0, 0.0425, 1.04, -0.04);
        const double a004 =
            std::min(r.x * r.x, std::exp2(-9.28 * mu)) * r.x + r.y;
        return std::max((-1.04 * a004 + r.z) + (1.04 * a004 + r.w),
                        1e-3);
    };
    const auto compensation = [&](double f0) {
        return std::sqrt((1.0 + f0 * (1.0 / albedoE(g.mu0) - 1.0)) *
                         (1.0 + f0 * (1.0 / albedoE(g.mu) - 1.0)));
    };
    double f = 0.0;
    if (p[3] > 0.5) {
        const double f0 = (p[4] + p[5] + p[6]) / 3.0;
        const double fresnel = schlick(f0);
        f = (1.0 - fresnel) * albedo / kPi +
            fresnel * specular * compensation(f0);
    } else {
        const double r0 = std::pow((p[2] - 1.0) / (p[2] + 1.0), 2.0);
        const double dielectric =
            (1.0 - schlick(r0)) * albedo / kPi +
            schlick(r0) * specular * compensation(r0);
        const double metal =
            schlick(albedo) * specular * compensation(albedo);
        f = (1.0 - p[1]) * dielectric + p[1] * metal;
    }
    if (p[7] > 0.0) {
        const double coat = schlick(0.04) * p[7];
        f = f * (1.0 - coat) + coat * lobe(p[8]);
    }
    return kPi * g.mu0 * f;
}

}  // namespace ernie::testing
