// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/sensor_filters.h"

#include "ernie/kernel/embedded_shaders.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ernie {

std::string_view NameOf(SensorFilter filter) {
    switch (filter) {
    case SensorFilter::Lens:
        return "lens";
    case SensorFilter::Qe:
        return "qe";
    case SensorFilter::Noise:
        return "noise";
    case SensorFilter::Adc:
        return "adc";
    case SensorFilter::Cie:
        return "cie";
    case SensorFilter::Exposure:
        return "exposure";
    }
    return "";
}

std::optional<SensorFilter> SensorFilterNamed(std::string_view name) {
    for (const SensorFilter filter : kSensorFilters) {
        if (NameOf(filter) == name) {
            return filter;
        }
    }
    return std::nullopt;
}

std::vector<SensorFilter> DefaultFilterChain() {
    return {kSensorFilters.begin(), kSensorFilters.end()};
}

std::vector<SensorFilter> CheckFilterChain(std::span<const SensorFilter> chain) {
    std::vector<SensorFilter> checked;
    if (chain.empty() || chain.front() != SensorFilter::Lens) {
        checked.push_back(SensorFilter::Lens);
    }
    const auto seen = [&checked](SensorFilter filter) {
        return std::find(checked.begin(), checked.end(), filter) != checked.end();
    };
    for (const SensorFilter filter : chain) {
        if (seen(filter)) {
            throw std::invalid_argument("Sensor filter '" + std::string(NameOf(filter)) +
                                        "' appears twice");
        }
        if ((filter == SensorFilter::Noise || filter == SensorFilter::Adc) &&
            !seen(SensorFilter::Qe)) {
            throw std::invalid_argument("Sensor filter '" + std::string(NameOf(filter)) +
                                        "' needs 'qe' before it");
        }
        if (filter == SensorFilter::Exposure && !seen(SensorFilter::Cie)) {
            throw std::invalid_argument("Sensor filter 'exposure' needs 'cie' before it");
        }
        checked.push_back(filter);
    }
    return checked;
}

FilterKind KindOf(SensorFilter filter) {
    switch (filter) {
    case SensorFilter::Lens:
        return FilterKind::Spatial;
    case SensorFilter::Qe:
    case SensorFilter::Cie:
        return FilterKind::Integrator;
    case SensorFilter::Noise:
    case SensorFilter::Adc:
    case SensorFilter::Exposure:
        return FilterKind::Pixel;
    }
    return FilterKind::Pixel;
}

namespace {

bool IsIdentifier(std::string_view name) {
    return !name.empty() &&
           (std::isalpha(static_cast<unsigned char>(name.front())) != 0 || name.front() == '_') &&
           std::all_of(name.begin(), name.end(), [](char ch) {
               return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
           });
}

}  // namespace

void SensorFilterRegistry::Add(SensorFilterModel model) {
    const std::string what = "Sensor filter '" + model.id + "'";
    if (!IsIdentifier(model.id)) {
        throw std::invalid_argument(what + ": its id must be an identifier");
    }
    if (SensorFilterNamed(model.id) || Find(model.id) != nullptr) {
        throw std::invalid_argument(what + " is already defined");
    }
    const bool suits =
        model.kind == FilterKind::Band    ? model.domain == FilterDomain::Spectrum
        : model.kind == FilterKind::Pixel ? model.domain != FilterDomain::Spectrum
        : model.kind == FilterKind::Integrator
            ? model.domain == FilterDomain::Electrons || model.domain == FilterDomain::Rgb
            : model.kind == FilterKind::Spatial;
    if (!suits) {
        throw std::invalid_argument(what + ": a custom filter is a band filter on the spectrum, "
                                           "a pixel filter on another domain, a spatial filter, "
                                           "or an integrator into electrons or linear colour");
    }
    if (model.params.size() > kMaxParams) {
        throw std::invalid_argument(what + " has more than " + std::to_string(kMaxParams) +
                                    " parameters");
    }
    std::size_t floats = 0;
    for (const SensorFilterParam& param : model.params) {
        if (param.size == 0) {
            throw std::invalid_argument(what + ": parameter '" + param.name + "' has size 0");
        }
        floats += param.size;
    }
    if (floats > kMaxParamFloats) {
        throw std::invalid_argument(what + "'s parameters hold more than " +
                                    std::to_string(kMaxParamFloats) + " floats");
    }
    for (std::size_t i = 0; i < model.params.size(); ++i) {
        const std::string& name = model.params[i].name;
        if (!IsIdentifier(name)) {
            throw std::invalid_argument(what + ": parameter '" + name + "' is not an identifier");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (model.params[j].name == name) {
                throw std::invalid_argument(what + ": parameter '" + name + "' repeats");
            }
        }
    }
    if (!IsIdentifier(model.typeName) || model.slang.find(model.typeName) == std::string::npos) {
        throw std::invalid_argument(what + ": its Slang must define type '" + model.typeName + "'");
    }
    if (model.pyramid &&
        (model.kind != FilterKind::Spatial || model.domain == FilterDomain::Spectrum)) {
        throw std::invalid_argument(what + ": only spatial filters on a domain after the "
                                           "spectrum read a pyramid");
    }
    if (model.states > kMaxStates) {
        throw std::invalid_argument(what + " keeps more than " + std::to_string(kMaxStates) +
                                    " state slots");
    }
    if (model.outputs.size() > kMaxAovs) {
        throw std::invalid_argument(what + " has more than " + std::to_string(kMaxAovs) +
                                    " outputs");
    }
    for (std::size_t i = 0; i < model.outputs.size(); ++i) {
        const std::string& name = model.outputs[i].name;
        if (!IsIdentifier(name)) {
            throw std::invalid_argument(what + ": output '" + name + "' is not an identifier");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (model.outputs[j].name == name) {
                throw std::invalid_argument(what + ": output '" + name + "' repeats");
            }
        }
    }
    models_.push_back(std::move(model));
}

const SensorFilterModel* SensorFilterRegistry::Find(std::string_view id) const {
    for (const SensorFilterModel& model : models_) {
        if (model.id == id) {
            return &model;
        }
    }
    return nullptr;
}

std::vector<float> ResampleCurve(std::span<const float> values, std::size_t size) {
    std::vector<float> out(size, values.empty() ? 0.0f : values.front());
    if (values.size() < 2 || size < 2) {
        return out;
    }
    for (std::size_t i = 0; i < size; ++i) {
        const double at = static_cast<double>(i) / static_cast<double>(size - 1) *
                          static_cast<double>(values.size() - 1);
        const std::size_t lower = std::min(static_cast<std::size_t>(at), values.size() - 2);
        const double f = at - static_cast<double>(lower);
        out[i] = static_cast<float>(static_cast<double>(values[lower]) * (1.0 - f) +
                                    static_cast<double>(values[lower + 1]) * f);
    }
    return out;
}

std::vector<SensorFilterModel> StandardSensorFilters() {
    using Kind = FilterKind;
    using Domain = FilterDomain;
    // A Gaussian QE curve at 10 nm from 360 to 830 nm, for `measuredQe`'s defaults: a generic
    // silicon sensor behind RGB dyes.
    const auto gaussian = [](double peakNm, double widthNm, double peak) {
        std::vector<float> curve(48);
        for (std::size_t i = 0; i < curve.size(); ++i) {
            const double nm = 360.0 + 10.0 * static_cast<double>(i);
            curve[i] = static_cast<float>(
                peak * std::exp(-0.5 * (nm - peakNm) * (nm - peakNm) / (widthNm * widthNm)));
        }
        return curve;
    };
    return {
        {"measuredQe",
         "Measured QE",
         "Electrons from red, green and blue QE curves (48 samples, 360 to 830 nm) in place of "
         "the camera's sensor model: an integrator, in place of qe.",
         {"EMVA Standard 1288 Release 4.0 Linear, 2021"},
         Kind::Integrator,
         Domain::Electrons,
         {{"red", 0.0f, "Red QE at 360 to 830 nm, 0 to 1", 48, gaussian(600.0, 35.0, 0.45)},
          {"green", 0.0f, "Green QE", 48, gaussian(530.0, 40.0, 0.55)},
          {"blue", 0.0f, "Blue QE", 48, gaussian(460.0, 30.0, 0.5)}},
         "MeasuredQe",
         shaders::kFilterMeasuredQeSlang},
        {"psf",
         "Point spread",
         "The lens's point-spread function: diffraction for the camera's f-number per band "
         "(a Gaussian fit to the Airy disc, sigma = 0.42 lambda N) with a Gaussian blur for "
         "aberrations; light is kept.",
         {"G. B. Airy, \"On the Diffraction of an Object-glass with Circular Aperture\", 1835",
          "B. Zhang, J. Zerubia and J.-C. Olivo-Marin, \"Gaussian approximations of fluorescence "
          "microscope point-spread function models\", Applied Optics 46(10), 2007"},
         Kind::Spatial,
         Domain::Spectrum,
         {{"blur", 0.0f, "Gaussian sigma in pixels added to diffraction's"},
          {"maxRadius", 12.0f, "The widest kernel in pixels, bounding the cost"}},
         "Psf",
         shaders::kFilterPsfSlang},
        {"defocus",
         "Depth of field",
         "Thin-lens defocus from the camera's fStop and focusDistance: each point a disc of the "
         "circle of confusion's size; pinhole cameras (fStop 0) are untouched.",
         {"M. Potmesil and I. Chakravarty, \"A lens and aperture camera model for synthetic "
          "image generation\", SIGGRAPH 1981",
          "J. Jimenez, \"Next Generation Post Processing in Call of Duty: Advanced Warfare\", "
          "SIGGRAPH 2014 course"},
         Kind::Spatial,
         Domain::Spectrum,
         {{"maxRadius", 8.0f, "The widest blur disc's radius in pixels, bounding the cost"}},
         "Defocus",
         shaders::kFilterDefocusSlang},
        {"bayer",
         "Bayer mosaic",
         "A Bayer colour filter array: each pixel of an RGB sensor keeps one channel's "
         "electrons.",
         {"B. E. Bayer, \"Color imaging array\", US patent 3,971,065, 1976"},
         Kind::Pixel,
         Domain::Electrons,
         {{"pattern", 0.0f, "0 RGGB, 1 BGGR, 2 GRBG, 3 GBRG"}},
         "Bayer",
         shaders::kFilterBayerSlang},
        {"demosaic",
         "Demosaic",
         "Full RGB digital numbers interpolated from a Bayer mosaic.",
         {"H. S. Malvar, L. He and R. Cutler, \"High-quality linear interpolation for "
          "demosaicing of Bayer-patterned color images\", IEEE ICASSP 2004"},
         Kind::Spatial,
         Domain::Digital,
         {{"pattern", 0.0f, "As the bayer filter's"},
          {"method", 1.0f, "0 bilinear, 1 Malvar-He-Cutler"}},
         "Demosaic",
         shaders::kFilterDemosaicSlang,
         {{"mosaic", "The raw mosaic before demosaicing: each pixel's own channel, the others 0"}}},
        {"blooming",
         "Blooming",
         "Charge beyond the full well spilling along columns, filling each neighbour to its full "
         "well, as in a CCD; antiBlooming drains a fraction of it instead.",
         {"J. R. Janesick, \"Scientific Charge-Coupled Devices\", SPIE Press, 2001"},
         Kind::Spatial,
         Domain::Electrons,
         {{"antiBlooming", 0.0f, "Fraction of the excess an overflow drain removes, 0 to 1"},
          {"maxLength", 64.0f, "The longest spill in pixels each way, bounding the cost"}},
         "Blooming",
         shaders::kFilterBloomingSlang,
         {{"overflow", "Electrons beyond the full well before spilling"}}},
        {"whiteBalance",
         "White balance",
         "Gains on linear colour's red, green and blue.",
         {"J. von Kries, 1905, in D. L. MacAdam (ed.), \"Sources of Color Science\", 1970"},
         Kind::Pixel,
         Domain::Rgb,
         {{"red", 1.0f, "Red gain"}, {"green", 1.0f, "Green gain"}, {"blue", 1.0f, "Blue gain"}},
         "WhiteBalance",
         shaders::kFilterWhiteBalanceSlang},
        {"colorMatrix",
         "Colour matrix",
         "A 3 x 3 colour correction matrix on linear colour, row by row.",
         {"M. S. Brown, \"Understanding the In-Camera Image Processing Pipeline for Computer "
          "Vision\", CVPR 2016 tutorial"},
         Kind::Pixel,
         Domain::Rgb,
         {{"m00", 1.0f, ""},
          {"m01", 0.0f, ""},
          {"m02", 0.0f, ""},
          {"m10", 0.0f, ""},
          {"m11", 1.0f, ""},
          {"m12", 0.0f, ""},
          {"m20", 0.0f, ""},
          {"m21", 0.0f, ""},
          {"m22", 1.0f, ""}},
         "ColorMatrix",
         shaders::kFilterColorMatrixSlang},
        {"toneCurve",
         "Tone curve",
         "Each display channel through a curve of 33 samples over 0 to 1 (identity by default).",
         {"M. S. Brown, \"Understanding the In-Camera Image Processing Pipeline for Computer "
          "Vision\", CVPR 2016 tutorial"},
         Kind::Pixel,
         Domain::Display,
         {{"curve", 0.0f, "Output at evenly spaced inputs from 0 to 1", 33, {0.0f, 1.0f}}},
         "ToneCurve",
         shaders::kFilterToneCurveSlang},
        {"spectralTransmission",
         "Spectral transmission",
         "Each band times a transmission curve of 48 samples, 360 to 830 nm every 10 nm (all 1 "
         "by default): an IR-cut, band-pass or ND filter, a lens's or window's transmission.",
         {},
         Kind::Band,
         Domain::Spectrum,
         {{"curve", 1.0f, "Transmission at evenly spaced wavelengths from 360 to 830 nm", 48}},
         "SpectralTransmission",
         shaders::kFilterSpectralTransmissionSlang},
        {"glare",
         "Glare",
         "A wide halo of scattered light around bright sources, its point-spread falling as a "
         "power of distance, summed over the image's pyramid; light is kept.",
         {"G. Spencer, P. Shirley, K. Zimmerman and D. P. Greenberg, \"Physically-Based Glare "
          "Effects for Digital Images\", SIGGRAPH 1995"},
         Kind::Spatial,
         Domain::Rgb,
         {{"strength", 0.04f, "Fraction of the light scattered into the halo, 0 to 1"},
          {"falloff", 3.0f, "The point-spread's power of distance, above 2"},
          {"levels", 8.0f, "Pyramid levels summed: the halo reaches about 2^levels pixels"}},
         "Glare",
         shaders::kFilterGlareSlang,
         {},
         0,
         true},
        {"flare",
         "Lens flare ghosts",
         "Ghosts of bright sources from reflections between lens surfaces: defocused images of "
         "the aperture (a disc, or a polygon of blades) on the line through the principal point, "
         "each at its own scale (negative mirrored), blur and coating tint; the light they take "
         "leaves the direct image. Arrays take one value per ghost. After qe, before noise.",
         {"M. Hullin, E. Eisemann, H.-P. Seidel and S. Lee, \"Physically-Based Real-Time Lens "
          "Flare Rendering\", SIGGRAPH 2011",
          "J. Chapman, \"Pseudo Lens Flare\", 2013",
          "W. J. Smith, \"Modern Optical Engineering\", 4th ed., 2008"},
         Kind::Spatial,
         Domain::Electrons,
         {{"strength", 1e-3f, "Share of the light each ghost takes, times its weight and tint"},
          {"ghosts", 4.0f, "Ghosts drawn, 0 to 8"},
          {"threshold", 0.0f, "Electrons above which a pixel makes ghosts"},
          {"blades", 6.0f, "The iris's blades, the ghosts' polygon; below 3 a disc"},
          {"rotation", 0.0f, "The polygon's rotation, degrees"},
          {"scales",
           0.0f,
           "Each ghost's scale about the principal point; negative mirrors it through",
           8,
           {-0.5f, -1.0f, -1.6f, 0.6f, -0.25f, -2.2f, 0.3f, -0.75f}},
          {"radii",
           0.0f,
           "Each ghost's blur radius, a fraction of the image's half diagonal",
           8,
           {0.03f, 0.06f, 0.1f, 0.04f, 0.02f, 0.15f, 0.05f, 0.08f}},
          {"weights",
           0.0f,
           "Each ghost's relative share",
           8,
           {1.0f, 0.7f, 0.5f, 0.6f, 1.2f, 0.3f, 0.8f, 0.5f}},
          {"tintRed",
           0.0f,
           "Each ghost's tint in red (also the NIR channel's)",
           8,
           {1.0f, 0.6f, 0.9f, 0.4f, 1.0f, 0.7f, 0.5f, 0.9f}},
          {"tintGreen",
           0.0f,
           "Each ghost's tint in green",
           8,
           {0.8f, 1.0f, 0.6f, 0.7f, 0.5f, 1.0f, 0.9f, 0.6f}},
          {"tintBlue",
           0.0f,
           "Each ghost's tint in blue",
           8,
           {0.4f, 0.9f, 1.0f, 1.0f, 0.6f, 0.5f, 1.0f, 0.8f}}},
         "Flare",
         shaders::kFilterFlareSlang,
         {},
         0,
         true},
        {"lag",
         "Image lag",
         "A fraction of each frame's charge read out with the next, so moving highlights trail "
         "ghosts (temporal).",
         {"J. R. Janesick, \"Photon Transfer\", SPIE Press, 2007"},
         Kind::Pixel,
         Domain::Electrons,
         {{"fraction", 0.05f, "Of each frame's charge read out with the next, 0 to 1"}},
         "Lag",
         shaders::kFilterLagSlang,
         {},
         1},
        {"temporalAverage",
         "Temporal average",
         "An exponential moving average of linear colour over the frames (temporal).",
         {"S. W. Roberts, \"Control chart tests based on geometric moving averages\", "
          "Technometrics 1(3), 1959"},
         Kind::Pixel,
         Domain::Rgb,
         {{"alpha", 0.2f, "Weight of the new frame, 0 to 1"}},
         "TemporalAverage",
         shaders::kFilterTemporalAverageSlang,
         {},
         1},
        {"compand",
         "Square-root companding",
         "Digital numbers compressed as the square root of their fraction of full scale, as "
         "LROC's cameras compress 12 bits to 8.",
         {"M. S. Robinson et al., \"Lunar Reconnaissance Orbiter Camera (LROC) Instrument "
          "Overview\", Space Science Reviews 150, 2010"},
         Kind::Pixel,
         Domain::Digital,
         {{"inputBits", 12.0f, "Bits of the ADC's numbers"},
          {"outputBits", 8.0f, "Bits after companding"}},
         "Compand",
         shaders::kFilterCompandSlang},
        {"detectorPattern",
         "Detector pattern",
         "A fixed gain and offset per detector of a line scanner, the same down its whole column: "
         "the along-track stripes of pushbroom images. Before noise.",
         {"H. Carfantan and J. Idier, \"Statistical Linear Destriping of Satellite-Based "
          "Pushbroom-Type Images\", IEEE TGRS 48(4), 2010"},
         Kind::Pixel,
         Domain::Electrons,
         {{"gain", 0.01f, "Standard deviation of each detector's relative gain"},
          {"offset", 2.0f, "Standard deviation of each detector's offset, electrons"},
          {"axis", 0.0f, "0: a detector per column (lines are rows), 1: per row"}},
         "DetectorPattern",
         shaders::kFilterDetectorPatternSlang},
        {"readoutTaps",
         "Readout taps",
         "Interleaved output amplifiers, pixel x on tap x mod taps, each with its own gain and "
         "offset, such as even and odd pixel offsets. After adc.",
         {"D. C. Humm et al., \"Flight Calibration of the LROC Narrow Angle Camera\", Space "
          "Science Reviews 200, 2016"},
         Kind::Pixel,
         Domain::Digital,
         {{"taps", 2.0f, "Interleaved taps, 1 to 4"},
          {"offset0", 0.0f, "Tap 0's offset (even pixels of two), DN"},
          {"offset1", 0.0f, "Tap 1's offset, DN"},
          {"offset2", 0.0f, "Tap 2's offset, DN"},
          {"offset3", 0.0f, "Tap 3's offset, DN"},
          {"gain0", 1.0f, "Tap 0's gain"},
          {"gain1", 1.0f, "Tap 1's gain"},
          {"gain2", 1.0f, "Tap 2's gain"},
          {"gain3", 1.0f, "Tap 3's gain"}},
         "ReadoutTaps",
         shaders::kFilterReadoutTapsSlang},
        {"tdi",
         "Time delay integration",
         "A TDI line scanner's stages summed along the track, each one line's exposure, smeared "
         "when the image drifts against the charge's clocking. Before noise.",
         {"G. Lepage, J. Bogaerts and G. Meynants, \"Time-Delay-Integration Architectures in "
          "CMOS Image Sensors\", IEEE Transactions on Electron Devices 56(11), 2009"},
         Kind::Spatial,
         Domain::Electrons,
         {{"stages", 32.0f, "Stages summed, 1 to 128"},
          {"drift", 0.0f, "Pixels the image drifts along the track per stage; 0 in step"},
          {"axis", 0.0f, "0: the track runs down the rows, 1: along the columns"}},
         "Tdi",
         shaders::kFilterTdiSlang},
        {"defectPixels",
         "Defect pixels",
         "Dead pixels with no response and hot pixels with a high dark current, the same pixels "
         "every frame. Before noise.",
         {"EMVA Standard 1288 Release 4.0 Linear, 2021"},
         Kind::Pixel,
         Domain::Electrons,
         {{"dead", 1e-4f, "Fraction of pixels with no response"},
          {"hot", 1e-4f, "Fraction of pixels with a high dark current"},
          {"hotRate", 1000.0f, "A hot pixel's typical dark current, electrons per second"}},
         "DefectPixels",
         shaders::kFilterDefectPixelsSlang},
        {"smear",
         "Frame-transfer smear",
         "Light collected while a shutterless frame-transfer CCD shifts its image into storage: "
         "each pixel gains its column's mean signal times the transfer time over the exposure. "
         "Before noise.",
         {"J. R. Janesick, \"Scientific Charge-Coupled Devices\", SPIE Press, 2001"},
         Kind::Spatial,
         Domain::Electrons,
         {{"transferTime", 1e-3f, "Seconds to shift the whole frame into storage"},
          {"axis", 0.0f, "0: charge moves along columns, 1: along rows"}},
         "Smear",
         shaders::kFilterSmearSlang},
        {"darkCurrent",
         "Thermal dark current",
         "Dark current doubling every few degrees with the sensor's temperature, each pixel's "
         "rate off the mean (DSNU). In place of the sensor's constant dark current; before "
         "noise.",
         {"EMVA Standard 1288 Release 4.0 Linear, 2021",
          "J. R. Janesick, \"Photon Transfer\", SPIE Press, 2007"},
         Kind::Pixel,
         Domain::Electrons,
         {{"temperature", 20.0f, "The sensor's temperature, degrees Celsius"},
          {"referenceTemperature", 20.0f, "Degrees Celsius"},
          {"referenceRate", 10.0f, "Mean dark current there, electrons per second"},
          {"doubling", 6.3f, "Degrees Celsius per doubling of the rate"},
          {"nonuniformity", 0.1f, "Standard deviation of each pixel's rate, relative"}},
         "DarkCurrent",
         shaders::kFilterDarkCurrentSlang},
        {"jpeg",
         "JPEG",
         "The display image through baseline JPEG (YCbCr, 8 x 8 DCT, standard quantisation "
         "tables at a quality, 4:4:4) and back. After exposure.",
         {"ITU-T Recommendation T.81, 1992", "E. Hamilton, \"JPEG File Interchange Format\", 1992"},
         Kind::Spatial,
         Domain::Display,
         {{"quality", 85.0f, "1 to 100, as encoders take it"}},
         "Jpeg",
         shaders::kFilterJpegSlang},
        {"compandTable",
         "Piecewise-linear companding",
         "Digital numbers through up to five linear segments, truncated, as LROC's cameras "
         "encode 12 bits in 8; by default the NAC's nominal table (code 0). After adc. Arrays "
         "take five values each.",
         {"M. S. Robinson et al., \"Lunar Reconnaissance Orbiter Camera (LROC) Instrument "
          "Overview\", Space Science Reviews 150, 2010, Appendix A"},
         Kind::Pixel,
         Domain::Digital,
         {{"segments", 5.0f, "Segments used, 1 to 5"},
          {"starts",
           0.0f,
           "Each segment's first input DN, increasing",
           5,
           {0.0f, 32.0f, 136.0f, 543.0f, 2207.0f}},
          {"slopes",
           0.0f,
           "Each segment's output DN per input DN",
           5,
           {0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f}},
          {"offsets",
           0.0f,
           "Each segment's output DN at input 0",
           5,
           {0.0f, 8.0f, 25.0f, 59.0f, 128.0f}}},
         "CompandTable",
         shaders::kFilterCompandTableSlang},
    };
}

SensorFilterRegistry SensorFilterRegistry::WithStandard() {
    SensorFilterRegistry registry;
    for (SensorFilterModel& model : StandardSensorFilters()) {
        registry.Add(std::move(model));
    }
    return registry;
}

std::vector<SensorFilterStep> StepsOf(std::span<const SensorFilter> chain) {
    std::vector<SensorFilterStep> steps;
    for (const SensorFilter filter : chain) {
        steps.push_back({std::string(NameOf(filter)), {}});
    }
    return steps;
}

std::vector<SensorFilterStep> CheckFilterChain(std::span<const SensorFilterStep> chain,
                                               const SensorFilterRegistry& registry) {
    // The built-in filters keep their own rules, a custom integrator standing for the built-in
    // whose domain it makes (`qe` for electrons, `cie` for linear colour); other custom filters
    // need their domain to exist.
    std::vector<SensorFilter> builtins;
    const auto integratorOf =
        [&registry](const SensorFilterStep& step) -> std::optional<SensorFilter> {
        const SensorFilterModel* model = registry.Find(step.type);
        if (model == nullptr || model->kind != FilterKind::Integrator) {
            return std::nullopt;
        }
        return model->domain == FilterDomain::Electrons ? SensorFilter::Qe : SensorFilter::Cie;
    };
    for (const SensorFilterStep& step : chain) {
        if (const std::optional<SensorFilter> builtin = step.Builtin()) {
            builtins.push_back(*builtin);
        } else if (registry.Find(step.type) == nullptr) {
            throw std::invalid_argument("Unknown sensor filter type '" + step.type + "'");
        } else if (const std::optional<SensorFilter> role = integratorOf(step)) {
            builtins.push_back(*role);
        }
    }
    CheckFilterChain(builtins);
    std::vector<SensorFilterStep> checked;
    if (chain.empty() || chain.front().Builtin() != SensorFilter::Lens) {
        checked.push_back({std::string(NameOf(SensorFilter::Lens)), {}});
    }
    const auto seen = [&checked, &integratorOf](SensorFilter filter) {
        return std::any_of(checked.begin(), checked.end(), [&](const SensorFilterStep& step) {
            return step.Builtin() == filter || integratorOf(step) == filter;
        });
    };
    for (const SensorFilterStep& step : chain) {
        if (step.Builtin() == SensorFilter::Lens && !checked.empty()) {
            throw std::invalid_argument("Sensor filter 'lens' must come first");
        }
        if (const SensorFilterModel* model = registry.Find(step.type);
            model != nullptr && model->kind != FilterKind::Integrator) {
            bool ready = true;
            switch (model->domain) {
            case FilterDomain::Spectrum:
                ready = !seen(SensorFilter::Qe) && !seen(SensorFilter::Cie);
                break;
            case FilterDomain::Electrons:
                ready = seen(SensorFilter::Qe);
                break;
            case FilterDomain::Digital:
                ready = seen(SensorFilter::Adc);
                break;
            case FilterDomain::Rgb:
                ready = seen(SensorFilter::Cie);
                break;
            case FilterDomain::Display:
                ready = seen(SensorFilter::Exposure);
                break;
            }
            if (!ready) {
                throw std::invalid_argument(
                    "Sensor filter '" + step.type +
                    (model->domain == FilterDomain::Spectrum
                         ? "' works on the bands, so it must come before 'qe' and 'cie'"
                         : "' runs before the filter producing what it works on"));
            }
        }
        checked.push_back(step);
    }
    return checked;
}

std::vector<std::string_view> VarsWrittenBy(std::span<const SensorFilterStep> chain,
                                            const SensorFilterRegistry& registry) {
    std::vector<SensorFilter> roles;
    for (const SensorFilterStep& step : CheckFilterChain(chain, registry)) {
        if (const std::optional<SensorFilter> builtin = step.Builtin()) {
            roles.push_back(*builtin);
        } else if (const SensorFilterModel* model = registry.Find(step.type);
                   model->kind == FilterKind::Integrator) {
            roles.push_back(model->domain == FilterDomain::Electrons ? SensorFilter::Qe
                                                                     : SensorFilter::Cie);
        }
    }
    return VarsWrittenBy(roles);
}

std::vector<std::string_view> VarsWrittenBy(std::span<const SensorFilter> chain) {
    std::vector<std::string_view> vars;
    for (const SensorFilter filter : chain) {
        switch (filter) {
        case SensorFilter::Lens:
            vars.insert(vars.end(),
                        {render_vars::kSpectralRadiance, render_vars::kDistanceToImagePlane,
                         render_vars::kDistanceToCamera, render_vars::kNormal, render_vars::kPrimId,
                         render_vars::kInstanceId, render_vars::kMotionVector});
            break;
        case SensorFilter::Qe:
            vars.push_back(render_vars::kElectrons);
            break;
        case SensorFilter::Noise:
            break;
        case SensorFilter::Adc:
            vars.push_back(render_vars::kRawDn);
            break;
        case SensorFilter::Cie:
            vars.push_back(render_vars::kHdrColor);
            break;
        case SensorFilter::Exposure:
            vars.push_back(render_vars::kLdrColor);
            break;
        }
    }
    return vars;
}

}  // namespace ernie

namespace ernie {

std::vector<std::string_view> RequiredVars(std::span<const std::string_view> requested) {
    if (requested.empty()) {
        // Every var but the opt-in MotionVector, which costs passes of its own.
        std::vector<std::string_view> every(kRenderVarNames.begin(), kRenderVarNames.end());
        std::erase(every, render_vars::kMotionVector);
        return every;
    }
    // The sensor computes every var from the raster radiance in one pass, keeping what a var is
    // made from in registers, so a frame holds just the requested vars, in binding order.
    std::vector<std::string_view> vars;
    for (const std::string_view name : kRenderVarNames) {
        if (std::find(requested.begin(), requested.end(), name) != requested.end()) {
            vars.push_back(name);
        }
    }
    for (const std::string_view slot : kAovSlotNames) {
        if (std::find(requested.begin(), requested.end(), slot) != requested.end()) {
            vars.push_back(slot);
        }
    }
    return vars;
}

std::vector<std::string> AovOutputsOf(std::span<const SensorFilterStep> chain,
                                      const SensorFilterRegistry& registry) {
    std::vector<std::string> outputs;
    for (const SensorFilterStep& step : CheckFilterChain(chain, registry)) {
        if (step.Builtin()) {
            continue;
        }
        for (const SensorFilterOutput& output : registry.Find(step.type)->outputs) {
            if (std::find(outputs.begin(), outputs.end(), output.name) != outputs.end()) {
                throw std::invalid_argument("Two sensor filters of the chain write AOV '" +
                                            output.name + "'");
            }
            outputs.push_back(output.name);
        }
    }
    if (outputs.size() > kMaxAovs) {
        throw std::invalid_argument("The sensor filter chain writes " +
                                    std::to_string(outputs.size()) + " AOVs, more than " +
                                    std::to_string(kMaxAovs));
    }
    return outputs;
}

}  // namespace ernie
