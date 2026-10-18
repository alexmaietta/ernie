// SPDX-License-Identifier: BSD-3-Clause
/// @file scattering.cpp
/// @brief Rayleigh and Mie scattering, refractivities and ozone.
#include "ernie/physics/scattering.h"

#include "ernie/util/math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace ernie {

namespace {

using std::numbers::pi;

/// Inverse square of a wavelength in nm, in um^-2, the variable of
/// the dispersion formulas.
double InverseSquareMicrons(double wavelengthNm) {
    const double um = wavelengthNm * 1e-3;
    return 1.0 / (um * um);
}

/// Wiscombe's (1980) number of Mie series terms for size parameter
/// `x`.
int MieTerms(double x) {
    const double cubeRoot = std::cbrt(x);
    const double terms = x <= 8.0     ? x + 4.0 * cubeRoot + 1.0
                         : x < 4200.0 ? x + 4.05 * cubeRoot + 2.0
                                      : x + 4.0 * cubeRoot + 2.0;
    return std::max(1, static_cast<int>(terms));
}

/// The logarithmic derivative D_n(z) = psi_n'(z) / psi_n(z) of the
/// Riccati-Bessel function, by Lentz's continued fraction for the
/// ratio of spherical Bessel functions: D_n = -n / z + a_1 a_2 ...,
/// with partial terms a_j = (-1)^(j+1) (2n + 2j - 1) / z and
/// convergents taken as ratios of two running products.
std::complex<double> LentzLogDerivative(std::complex<double> z, int n) {
    using Complex = std::complex<double>;
    Complex inverse = 2.0 / z;
    const Complex alpha = (n + 0.5) * inverse;
    Complex term = -(n + 1.5) * inverse;
    Complex numerator = term + 1.0 / alpha;
    Complex denominator = term;
    Complex ratio = numerator / denominator;
    Complex product = alpha * ratio;
    while (std::abs(std::abs(ratio) - 1.0) > 1e-12) {
        term = inverse - term;
        numerator = 1.0 / numerator + term;
        denominator = 1.0 / denominator + term;
        ratio = numerator / denominator;
        inverse = -inverse;
        product *= ratio;
    }
    return -static_cast<double>(n) / z + product;
}

}  // namespace

double AirRefractivity(double wavelengthNm, double co2Fraction) {
    const double s = InverseSquareMicrons(wavelengthNm);
    const double n300 = (8060.51 + 2480990.0 / (132.274 - s) + 17455.7 / (39.32957 - s)) * 1e-8;
    return n300 * (1.0 + 0.54 * (co2Fraction - 0.0003));
}

double CarbonDioxideRefractivity(double wavelengthNm) {
    const double s = InverseSquareMicrons(wavelengthNm);
    return (22822.1 + 117.8 * s + 2406030.0 / (130.0 - s) + 15997.0 / (38.9 - s)) * 1e-8;
}

double AirKingFactor(double wavelengthNm, double co2Fraction) {
    const double s = InverseSquareMicrons(wavelengthNm);
    const double nitrogen = 1.034 + 3.17e-4 * s;
    const double oxygen = 1.096 + 1.385e-3 * s + 1.448e-4 * s * s;
    // Volume percentages of N2, O2 and Ar in dry air, and CO2's.
    const double co2 = co2Fraction * 100.0;
    return (78.084 * nitrogen + 20.946 * oxygen + 0.934 * 1.0 + co2 * kCarbonDioxideKingFactor) /
           (78.084 + 20.946 + 0.934 + co2);
}

double RayleighCrossSection(double wavelengthNm, double refractivity, double kingFactor) {
    const double lambda = wavelengthNm * 1e-9;
    const double n2 = (1.0 + refractivity) * (1.0 + refractivity);
    const double ratio = (n2 - 1.0) / (n2 + 2.0);
    const double lambda2 = lambda * lambda;
    return 24.0 * pi * pi * pi * ratio * ratio /
           (lambda2 * lambda2 * kStandardNumberDensity * kStandardNumberDensity) * kingFactor;
}

double RayleighPhase(double cosTheta, double kingFactor) {
    // F = (6 + 3 rho) / (6 - 7 rho) solved for the depolarisation
    // ratio rho.
    const double rho = 6.0 * (kingFactor - 1.0) / (3.0 + 7.0 * kingFactor);
    const double gamma = rho / (2.0 - rho);
    return 3.0 / (16.0 * pi) * ((1.0 + 3.0 * gamma) + (1.0 - gamma) * cosTheta * cosTheta) /
           (1.0 + 2.0 * gamma);
}

MieEfficiencies MieSphere(double sizeParameter, std::complex<double> refractiveIndex,
                          std::span<const double> cosines, std::span<double> phase) {
    using Complex = std::complex<double>;
    if (!phase.empty() && phase.size() != cosines.size()) {
        throw std::invalid_argument("MieSphere needs one phase value per cosine");
    }
    const double x = sizeParameter;
    const Complex m = refractiveIndex;
    const int terms = MieTerms(x);

    // The logarithmic derivative D_n(mx) by downward recurrence
    // (Bohren and Huffman, eq. 4.89), stable for absorbing spheres
    // where upward recurrence is not, started from its exact value
    // by Lentz's continued fraction: Bohren and Huffman's start at
    // 0 a few orders up loses digits for large |mx|.
    const Complex y = m * x;
    const std::size_t count = static_cast<std::size_t>(terms);
    std::vector<Complex> d(count + 1);
    std::vector<double> dx(count + 1);
    d[count] = LentzLogDerivative(y, terms);
    dx[count] = LentzLogDerivative(Complex(x), terms).real();
    for (std::size_t n = count; n > 0; --n) {
        const Complex nOverY = static_cast<double>(n) / y;
        d[n - 1] = nOverY - 1.0 / (d[n] + nOverY);
        const double nOverX = static_cast<double>(n) / x;
        dx[n - 1] = nOverX - 1.0 / (dx[n] + nOverX);
    }

    // Riccati-Bessel functions psi_n = x j_n(x) and chi_n = -x
    // y_n(x) from n = -1, 0. chi grows with n and recurs upward
    // stably, as does psi up to n = x; past that psi decays and
    // upward recurrence loses digits, so it follows from D_n(x)
    // instead: psi_n = psi_{n-1} / (D_n(x) + n / x).
    double psiPrev = std::cos(x);
    double psi = std::sin(x);
    double chiPrev = -std::sin(x);
    double chi = std::cos(x);
    Complex xiPrev(psi, -chi);

    // Angular functions pi_n and the amplitude functions S1, S2 per
    // cosine.
    const std::size_t angles = phase.empty() ? 0 : cosines.size();
    std::vector<double> piPrev(angles, 0.0);
    std::vector<double> piCurrent(angles, 1.0);
    std::vector<Complex> s1(angles, Complex(0.0));
    std::vector<Complex> s2(angles, Complex(0.0));

    double extinction = 0.0;
    double scattering = 0.0;
    double asymmetry = 0.0;
    Complex aPrev(0.0);
    Complex bPrev(0.0);
    for (std::size_t n = 1; n <= count; ++n) {
        const double fn = static_cast<double>(n);
        const double psiNext =
            fn <= x ? (2.0 * fn - 1.0) / x * psi - psiPrev : psi / (dx[n] + fn / x);
        const double chiNext = (2.0 * fn - 1.0) / x * chi - chiPrev;
        psiPrev = psi;
        psi = psiNext;
        chiPrev = chi;
        chi = chiNext;
        const Complex xi(psi, -chi);

        // Bohren and Huffman, eq. 4.88.
        const Complex da = d[n] / m + fn / x;
        const Complex db = d[n] * m + fn / x;
        const Complex a = (da * psi - psiPrev) / (da * xi - xiPrev);
        const Complex b = (db * psi - psiPrev) / (db * xi - xiPrev);

        extinction += (2.0 * fn + 1.0) * (a.real() + b.real());
        scattering += (2.0 * fn + 1.0) * (std::norm(a) + std::norm(b));
        asymmetry += (2.0 * fn + 1.0) / (fn * (fn + 1.0)) * (a * std::conj(b)).real();
        if (n > 1) {
            asymmetry +=
                (fn - 1.0) * (fn + 1.0) / fn * (aPrev * std::conj(a) + bPrev * std::conj(b)).real();
        }

        const double weight = (2.0 * fn + 1.0) / (fn * (fn + 1.0));
        for (std::size_t j = 0; j < angles; ++j) {
            const double mu = cosines[j];
            const double tau = fn * mu * piCurrent[j] - (fn + 1.0) * piPrev[j];
            s1[j] += weight * (a * piCurrent[j] + b * tau);
            s2[j] += weight * (a * tau + b * piCurrent[j]);
            const double piNext =
                ((2.0 * fn + 1.0) * mu * piCurrent[j] - (fn + 1.0) * piPrev[j]) / fn;
            piPrev[j] = piCurrent[j];
            piCurrent[j] = piNext;
        }

        xiPrev = xi;
        aPrev = a;
        bPrev = b;
    }

    MieEfficiencies result;
    result.extinction = 2.0 / (x * x) * extinction;
    result.scattering = 2.0 / (x * x) * scattering;
    result.asymmetry =
        result.scattering > 0.0 ? 4.0 / (x * x * result.scattering) * asymmetry : 0.0;
    // The unpolarised intensity (|S1|^2 + |S2|^2) / 2 integrates to
    // pi x^2 Q_sca over the sphere.
    const double normalisation = 2.0 * pi * x * x * result.scattering;
    for (std::size_t j = 0; j < angles; ++j) {
        phase[j] = normalisation > 0.0 ? (std::norm(s1[j]) + std::norm(s2[j])) / normalisation
                                       : kIsotropicPhase;
    }
    return result;
}

LogNormalMode LogNormalMode::FromEffective(double effectiveRadius, double effectiveVariance,
                                           double relativeNumber) {
    const double variance = std::log1p(effectiveVariance);
    return {effectiveRadius * std::exp(-2.5 * variance), std::sqrt(variance), relativeNumber};
}

double LogNormalMode::EffectiveRadius() const {
    return medianRadius * std::exp(2.5 * sigma * sigma);
}

double LogNormalMode::EffectiveVariance() const {
    return std::expm1(sigma * sigma);
}

RefractiveIndex::RefractiveIndex(std::complex<double> value) : wavelengths_{1.0}, values_{value} {}

RefractiveIndex::RefractiveIndex(std::vector<double> wavelengthsNm,
                                 std::vector<std::complex<double>> values)
    : wavelengths_(std::move(wavelengthsNm)), values_(std::move(values)) {
    if (wavelengths_.size() != values_.size()) {
        throw std::invalid_argument("A refractive index needs one value per wavelength");
    }
    for (std::size_t i = 1; i < wavelengths_.size(); ++i) {
        if (!(wavelengths_[i] > wavelengths_[i - 1])) {
            throw std::invalid_argument("Refractive index wavelengths must be strictly "
                                        "increasing");
        }
    }
}

std::complex<double> RefractiveIndex::operator()(double wavelengthNm) const {
    return Interpolate<std::complex<double>>(wavelengths_, values_, wavelengthNm);
}

std::complex<double> MaxwellGarnett(std::complex<double> matrix, std::complex<double> inclusion,
                                    double fraction) {
    const std::complex<double> em = matrix * matrix;
    const std::complex<double> ei = inclusion * inclusion;
    const std::complex<double> mixed =
        em * (ei + 2.0 * em + 2.0 * fraction * (ei - em)) / (ei + 2.0 * em - fraction * (ei - em));
    return std::sqrt(mixed);
}

ParticleOptics MieParticles(std::span<const LogNormalMode> modes,
                            std::complex<double> refractiveIndex, double wavelengthNm,
                            std::span<const double> cosines, int radiusSamples) {
    double totalNumber = 0.0;
    for (const LogNormalMode& mode : modes) {
        totalNumber += mode.relativeNumber;
    }
    if (modes.empty() || !(totalNumber > 0.0) || radiusSamples < 2) {
        throw std::invalid_argument("Mie particles need size modes with a positive number");
    }

    const double lambda = wavelengthNm * 1e-9;
    ParticleOptics result;
    result.phase.assign(cosines.size(), 0.0);
    std::vector<double> phase(cosines.size());
    for (const LogNormalMode& mode : modes) {
        const double fraction = mode.relativeNumber / totalNumber;
        const double mean = std::log(mode.medianRadius);
        // Numbers fall off as exp(-t^2/2) in t = (ln r - mean) /
        // sigma, and cross sections weight them by r^2, which moves
        // their mass up by 2 sigma^2.
        const double low = mean - 4.0 * mode.sigma;
        const double high = mean + 2.0 * mode.sigma * mode.sigma + 4.0 * mode.sigma;
        const int samples = mode.sigma > 0.0 ? radiusSamples : 1;
        double weightSum = 0.0;
        double modeScattering = 0.0;
        double modeAbsorption = 0.0;
        std::vector<double> modePhase(cosines.size(), 0.0);
        for (int i = 0; i < samples; ++i) {
            const double lnR = samples == 1 ? mean : low + (high - low) * i / (samples - 1);
            const double t = mode.sigma > 0.0 ? (lnR - mean) / mode.sigma : 0.0;
            const double trapezoid = (i == 0 || i == samples - 1) && samples > 1 ? 0.5 : 1.0;
            const double weight = trapezoid * std::exp(-0.5 * t * t);
            const double r = std::exp(lnR);
            const MieEfficiencies q =
                MieSphere(2.0 * pi * r / lambda, refractiveIndex, cosines, phase);
            const double area = pi * r * r;
            const double scattering = weight * q.scattering * area;
            modeScattering += scattering;
            modeAbsorption += weight * q.Absorption() * area;
            for (std::size_t j = 0; j < cosines.size(); ++j) {
                modePhase[j] += scattering * phase[j];
            }
            weightSum += weight;
        }
        result.scattering += fraction * modeScattering / weightSum;
        result.absorption += fraction * modeAbsorption / weightSum;
        for (std::size_t j = 0; j < cosines.size(); ++j) {
            result.phase[j] += fraction * modePhase[j] / weightSum;
        }
    }
    for (double& p : result.phase) {
        p = result.scattering > 0.0 ? p / result.scattering : kIsotropicPhase;
    }
    return result;
}

double OzoneCrossSection(double wavelengthNm) {
    // 10 nm averages starting at 360 nm (CosmoScout VR,
    // ozoneMode.cpp, MIT).
    static constexpr std::array<double, 48> kAverages = {
        1.18006e-27, 2.18205e-28, 2.81764e-28, 6.63629e-28, 1.52685e-27, 2.76259e-27, 5.51975e-27,
        8.45102e-27, 1.58232e-26, 2.31555e-26, 3.66625e-26, 4.92413e-26, 7.76088e-26, 9.02900e-26,
        1.48333e-25, 1.60547e-25, 2.14349e-25, 2.76161e-25, 3.09823e-25, 3.50934e-25, 4.27703e-25,
        4.68477e-25, 4.40965e-25, 4.71385e-25, 5.03275e-25, 4.31623e-25, 3.74999e-25, 3.22324e-25,
        2.66819e-25, 2.24367e-25, 1.85651e-25, 1.47571e-25, 1.21105e-25, 9.43730e-26, 7.46292e-26,
        6.57117e-26, 5.10619e-26, 4.14823e-26, 4.22622e-26, 3.23257e-26, 2.44425e-26, 2.79549e-26,
        2.52744e-26, 1.61447e-26, 1.45506e-26, 2.07028e-26, 1.37295e-26, 6.98672e-27};
    constexpr double kFirstNm = 360.0;
    constexpr double kBinNm = 10.0;
    const double lastNm = kFirstNm + kBinNm * kAverages.size();
    if (wavelengthNm < kFirstNm || wavelengthNm > lastNm) {
        return 0.0;
    }
    // Linear between bin centres, constant in the outer half bins.
    const double position = (wavelengthNm - kFirstNm) / kBinNm - 0.5;
    const double clamped = std::clamp(position, 0.0, static_cast<double>(kAverages.size() - 1));
    const std::size_t i = std::min(static_cast<std::size_t>(clamped), kAverages.size() - 2);
    const double t = clamped - static_cast<double>(i);
    return kAverages[i] + (kAverages[i + 1] - kAverages[i]) * t;
}

const RefractiveIndex& HematiteRefractiveIndex() {
    // CosmoScout VR,
    // scattering-table-generator/mie-settings/mars_realistic.json
    // (CC0): every 10 nm from 220 to 1000 nm, n and k.
    static constexpr std::array<std::array<double, 2>, 79> kTable = {{
        {1.349, 1.127}, {1.517, 1.145}, {1.635, 1.145}, {1.733, 1.149}, {1.828, 1.148},
        {1.917, 1.133}, {1.992, 1.107}, {2.049, 1.076}, {2.090, 1.053}, {2.134, 1.041},
        {2.185, 1.018}, {2.218, 0.979}, {2.219, 0.947}, {2.206, 0.951}, {2.211, 0.987},
        {2.243, 1.034}, {2.305, 1.075}, {2.387, 1.097}, {2.475, 1.096}, {2.560, 1.075},
        {2.638, 1.040}, {2.708, 0.993}, {2.766, 0.936}, {2.809, 0.873}, {2.837, 0.813},
        {2.858, 0.758}, {2.876, 0.707}, {2.890, 0.654}, {2.893, 0.602}, {2.887, 0.560},
        {2.881, 0.532}, {2.885, 0.512}, {2.901, 0.492}, {2.927, 0.460}, {2.951, 0.413},
        {2.963, 0.351}, {2.956, 0.286}, {2.931, 0.229}, {2.898, 0.186}, {2.862, 0.156},
        {2.830, 0.134}, {2.800, 0.119}, {2.775, 0.110}, {2.753, 0.103}, {2.736, 0.096},
        {2.719, 0.089}, {2.703, 0.083}, {2.688, 0.078}, {2.675, 0.075}, {2.662, 0.072},
        {2.652, 0.068}, {2.641, 0.066}, {2.631, 0.063}, {2.621, 0.061}, {2.612, 0.060},
        {2.604, 0.059}, {2.596, 0.058}, {2.589, 0.057}, {2.582, 0.057}, {2.575, 0.057},
        {2.570, 0.058}, {2.566, 0.059}, {2.562, 0.059}, {2.559, 0.059}, {2.555, 0.058},
        {2.552, 0.058}, {2.549, 0.057}, {2.547, 0.056}, {2.544, 0.054}, {2.541, 0.053},
        {2.537, 0.053}, {2.535, 0.052}, {2.533, 0.051}, {2.531, 0.051}, {2.529, 0.049},
        {2.527, 0.048}, {2.525, 0.046}, {2.522, 0.045}, {2.520, 0.043},
    }};
    static const RefractiveIndex index = [] {
        std::vector<double> wavelengths;
        std::vector<std::complex<double>> values;
        for (std::size_t i = 0; i < kTable.size(); ++i) {
            wavelengths.push_back(220.0 + 10.0 * static_cast<double>(i));
            values.emplace_back(kTable[i][0], kTable[i][1]);
        }
        return RefractiveIndex(std::move(wavelengths), std::move(values));
    }();
    return index;
}

}  // namespace ernie
