// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

#include <array>
#include <cstdint>

namespace ernie {

/// LROC's two Narrow Angle Cameras, side by side across the track.
enum class LrocNac {
    Left,   ///< NAC-L
    Right,  ///< NAC-R
};

/// Authors one of LROC's Narrow Angle Cameras as published, onto Camera `camera` and the
/// readout chain of RenderProduct `product`. The image's geometry (pose along the orbit, focal
/// length, detector line, summing, lens distortion and line time) comes from its ISD
/// (`IsdLineScanner::StripOf`, Python's `IsdLineScanner.author_strip`); this sets the rest:
/// - The sensor: one panchromatic channel over the 400-760 nm spectral response with the
///   detector's average quantum efficiency through it (> 60%: 0.6, its edges ramped over 10 nm;
///   the paper gives no curve), the gain, detector noise and full well of NAC-L or NAC-R, a
///   12-bit ADC and the telescope's f-number.
/// - The readout chain, SensorFilter prims under `product`: qe, noise, adc, readoutTaps (the
///   even and odd pixels' two analog chains, offsets 0 until calibrated) and compandTable (the
///   nominal 12- to 8-bit table, code 0), so RawDN holds the 8-bit numbers the NAC sends.
///
/// Reference: M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC) Instrument
/// Overview", Space Science Reviews 150:81-124, 2010: sections 3.1 and 4.1, Table 2 (focal
/// ratio, gain, detector noise, full well, spectral response, digitisation) and Appendix A
/// (bit-compression tables).
/// @throws std::invalid_argument if `camera` is not a Camera or `product` not a RenderProduct.
void AuthorLrocNac(Scene& scene, const Path& camera, const Path& product,
                   LrocNac which = LrocNac::Left);

/// The seven bands of LROC's Wide Angle Camera, by Robinson et al. 2010's band names: two
/// ultraviolet (read out 4 x 4 summed through the UV optics) and five visible.
enum class LrocWacBand {
    Uv320,
    Uv360,
    Vis415,
    Vis565,
    Vis605,
    Vis645,  ///< Also the monochrome mode's band.
    Vis690,
};

/// One WAC band's published values.
struct LrocWacBandSpec {
    double nameNm;        ///< Table 4's band name.
    double effectiveNm;   ///< Table 4's effective wavelength.
    double fwhmNm;        ///< Table 4's full width at half maximum.
    double responsivity;  ///< Table 4's responsivity coefficient, (W/m^3/sr) / (DN/s).
    /// Table 3: the visible optics' or the UV optics' (section 3.2.1's text gives 5.5 and 5.1
    /// for the lens designs; the table's are the system's).
    double fNumber;
    double pixelPitchUm;  ///< The 9 um pixels, 4 x 4 summed in the UV (section 3.2).
    /// The band's effective quantum efficiency, which with Ernie's camera equation turns a flat
    /// spectral radiance into Table 4's DN per second: gain h c / (RC FWHM pi / (4 N^2) A
    /// lambda_eff), the optics', filter's and fill factor's losses all in it.
    double EffectiveQe() const;
};

/// Band `band`'s values (Robinson et al. 2010, Tables 3 and 4).
LrocWacBandSpec LrocWacBandOf(LrocWacBand band);

/// The band whose Table 4 band name or effective wavelength is `nanometres` (605 or 604).
/// @throws std::invalid_argument for neither.
LrocWacBand LrocWacBandNamed(int nanometres);

/// Authors one band of LROC's Wide Angle Camera as published, onto Camera `camera` and the
/// readout chain of RenderProduct `product`. The image's geometry (each framelet's pose, focal
/// length, lens distortion and exposure) comes from its ISD (`IsdPushFrame::FrameletOf`,
/// Python's `IsdPushFrame.author_framelet`); this sets the rest:
/// - The sensor: one channel through the band, its passband Table 4's FWHM wide about the
///   effective wavelength (edges ramped over 2 nm about the half maxima, so its area is the
///   FWHM's) at the band's effective quantum efficiency (`LrocWacBandSpec::EffectiveQe`), the
///   lens's transmission 1 (in that efficiency); the KAI-1001's gain, read noise and full well;
///   an 11-bit ADC; the band's optics' f-number. The camera's spectral range is narrowed to the
///   passband, four bands across it.
/// - The readout chain, SensorFilter prims under `product`: qe, noise and adc, so RawDN holds
///   the 11-bit numbers before the WAC's companding to 8 bits, which the paper gives only as a
///   figure (Fig. 25).
///
/// Reference: M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC) Instrument
/// Overview", Space Science Reviews 150:81-124, 2010: section 3.2, Table 3 (focal ratios, gain,
/// read noise, full well, digitisation, summing) and Table 4 (bandpasses and responsivity).
/// @throws std::invalid_argument if `camera` is not a Camera or `product` not a RenderProduct.
void AuthorLrocWac(Scene& scene, const Path& camera, const Path& product,
                   LrocWacBand band = LrocWacBand::Vis605);

/// The Bayer layouts of the `bayer` and `demosaic` filters' `pattern`.
enum class BayerPattern { Rggb = 0, Bggr = 1, Grbg = 2, Gbrg = 3 };

/// The Mars 2020 rover Perseverance's Navcam pair (Maki et al. 2020, Table 2).
struct Mars2020Navcam {
    static constexpr double kFocalLengthMm = 19.1;
    static constexpr double kFNumber = 12.0;
    static constexpr double kBestFocusM = 3.5;
    static constexpr double kBaselineM = 0.424;
    /// The CMV-20000's 5120 x 3840 pixels of 6.4 um: a full frame of 32.77 x 24.58 mm.
    static constexpr std::uint32_t kWidth = 5120;
    static constexpr std::uint32_t kHeight = 3840;
    static constexpr double kPixelPitchUm = 6.4;
    /// Fields of view, degrees: horizontal, vertical and diagonal.
    static constexpr double kFieldOfViewDeg[3] = {96.0, 73.0, 120.0};
    static constexpr double kElectronsPerDn = 1.0 / 0.24;  ///< Conversion gain 0.24 DN/e-.
    static constexpr double kDarkNoiseElectrons = 8.0;     ///< Pixel dark noise, e- RMS.
    static constexpr double kFullWellElectrons = 15000.0;
    static constexpr int kAdcBits = 12;

    /// The lens's Kannala-Brandt k1 for the published fields of view: the equidistant (f-theta)
    /// projection, theta_d = theta (1 + k1 theta^2), fitted by least squares to the three, at
    /// the image's half widths over the focal length. The paper gives the calibrated model only
    /// as CAHVORE, without its coefficients.
    static double FisheyeK1();
};

/// Authors one of the Mars 2020 Navcams as published, onto Camera `camera` and the readout
/// chain of RenderProduct `product`:
/// - The camera: the f-theta lens (`Mars2020Navcam::FisheyeK1`, Ernie's fisheye) of 19.1 mm
///   over the full 32.77 x 24.58 mm frame, f/12, focused at 3.5 m. Render it at 5120 x 3840 for
///   the detector's own 6.4 um pixels; a smaller resolution bins them (as the cameras can
///   downsample), each pixel then wider and collecting more light.
/// - The sensor: RGB, the CMV-20000's gain, dark noise and full well, a 12-bit ADC. The paper
///   gives the detector's QE only as 64.5% at 550 nm (monochrome, with microlenses; its table
///   prints "55 nm"), with no colour responses, so the channels keep Ernie's generic RGB
///   curves.
/// - The readout chain, SensorFilter prims under `product`: qe, bayer (`pattern`; the paper
///   does not give the layout), noise, adc and demosaic (Malvar-He-Cutler), so RawDN holds the
///   12-bit numbers demosaiced; then cie, exposure and jpeg at `jpegQuality` for the 8-bit
///   colour images the rover sends JPEG-compressed (the paper gives the quality as
///   selectable). Fixed-pattern noise and dark current are not published; add `darkCurrent`
///   or `defectPixels` filters to model them.
///
/// Reference: J. N. Maki et al., "The Mars 2020 Engineering Cameras and Microphone on the
/// Perseverance Rover: A Next-Generation Imaging System for Mars Exploration", Space Science
/// Reviews 216:137, 2020, https://doi.org/10.1007/s11214-020-00765-9: Table 2 and sections on
/// the detector and flight software.
/// @throws std::invalid_argument if `camera` is not a Camera or `product` not a RenderProduct.
void AuthorMars2020Navcam(Scene& scene, const Path& camera, const Path& product,
                          BayerPattern pattern = BayerPattern::Rggb, double jpegQuality = 85.0);

/// The Navcam pair on stereo rig `rig` (an Xform, defined if missing, posed by the caller as the
/// cameras' shared frame, looking down its -z with +y up): Cameras `rig`/Left and `rig`/Right
/// 42.4 cm apart along its x, each authored as `AuthorMars2020Navcam` with products
/// `leftProduct` and `rightProduct`.
void AuthorMars2020NavcamPair(Scene& scene, const Path& rig, const Path& leftProduct,
                              const Path& rightProduct, BayerPattern pattern = BayerPattern::Rggb,
                              double jpegQuality = 85.0);

/// Perseverance's hazard cameras: two stereo pairs on the front of the rover body (looking down
/// past the front wheels) and one on the back.
enum class Mars2020Hazcams { Front, Rear };

/// The Hazcams' optics (Maki et al. 2020, Table 2), on the Navcams' camera body and detector
/// (`Mars2020Navcam`'s sensor values).
struct Mars2020Hazcam {
    static constexpr double kFocalLengthMm = 14.0;
    static constexpr double kFNumber = 12.0;
    static constexpr double kBestFocusM = 0.9;
    static constexpr double kFrontBaselineM = 0.248;
    static constexpr double kRearBaselineM = 0.934;
    /// Fields of view, degrees: horizontal, vertical and diagonal.
    static constexpr double kFieldOfViewDeg[3] = {136.0, 102.0, 172.0};

    /// The lens's Kannala-Brandt k1 for the published fields of view, as
    /// `Mars2020Navcam::FisheyeK1`. Its corners reach 86 degrees off axis, so it renders on cube
    /// faces (`NeedsCubeFaces`).
    static double FisheyeK1();
};

/// Authors one of the Mars 2020 Hazcams as published, as `AuthorMars2020Navcam` does a Navcam
/// but through the Hazcam's f-theta lens of 14.0 mm (`Mars2020Hazcam::FisheyeK1`), f/12,
/// focused at 0.9 m. The flight cameras' own calibrations (CAHVORE, `CahvoreModel`) see a little
/// less than the published fields (128.7 x 98.6 degrees for the front left); author one of them
/// with `CahvoreModel`'s ray table for a particular camera's exact geometry.
///
/// Reference: J. N. Maki et al., "The Mars 2020 Engineering Cameras and Microphone on the
/// Perseverance Rover", Space Science Reviews 216:137, 2020, Table 2.
/// @throws std::invalid_argument if `camera` is not a Camera or `product` not a RenderProduct.
void AuthorMars2020Hazcam(Scene& scene, const Path& camera, const Path& product,
                          BayerPattern pattern = BayerPattern::Rggb, double jpegQuality = 85.0);

/// A Hazcam pair on stereo rig `rig` (as `AuthorMars2020NavcamPair`): Cameras `rig`/Left and
/// `rig`/Right, 24.8 cm apart for the front pair or 93.4 cm for the rear, each authored as
/// `AuthorMars2020Hazcam`.
void AuthorMars2020HazcamPair(Scene& scene, const Path& rig, const Path& leftProduct,
                              const Path& rightProduct,
                              Mars2020Hazcams which = Mars2020Hazcams::Front,
                              BayerPattern pattern = BayerPattern::Rggb, double jpegQuality = 85.0);

/// The Mars 2020 rover Perseverance's Mastcam-Z: two zoom cameras whose filter wheels hold the
/// Bayer RGB's near-IR cut-off (filter 0), six narrow bands from 442 to 1022 nm (1 to 6) and a
/// neutral-density solar filter (7), the left camera's bands visible and the right's reaching
/// into the near infrared (Bell et al. 2021, Tables 3 and 5).
enum class MastcamZEye { Left, Right };

struct MastcamZ {
    /// The KAI-2020CM's 1600 x 1200 photoactive pixels of 7.4 um.
    static constexpr std::uint32_t kWidth = 1600;
    static constexpr std::uint32_t kHeight = 1200;
    static constexpr double kPixelPitchUm = 7.4;
    static constexpr double kElectronsPerDn = 15.6;
    static constexpr int kAdcBits = 11;
    /// Zoom: 26 mm at f/6.7 to 110 mm at f/9.5.
    static constexpr double kShortestFocalMm = 26.0;
    static constexpr double kLongestFocalMm = 110.0;
    static constexpr double kWidestFNumber = 6.7;
    static constexpr double kNarrowestFNumber = 9.5;
    static constexpr double kBaselineM = 0.244;
    static constexpr double kToeInDeg = 1.15;  ///< Each camera's, 2.3 degrees between them.
    /// The detector's peak quantum efficiency through its Bayer microlenses, ~40%.
    static constexpr double kPeakQe = 0.40;

    /// A filter's Table 3 effective wavelength and half width at half maximum, nm (for filter
    /// 0, the green Bayer channel's), and its neutral density (filter 7's; 0 otherwise).
    struct Filter {
        double effectiveNm;
        double hwhmNm;
        double neutralDensity;
    };
    /// @throws std::invalid_argument for a filter outside 0 to 7.
    static Filter FilterOf(MastcamZEye eye, int filter);
    /// Filter 0's red, green and blue Bayer channels (Table 3).
    static std::array<Filter, 3> BayerOf(MastcamZEye eye);
    /// The f-number at focal length `focalLengthMm`, linearly between the zoom's ends (the
    /// paper gives only those).
    static double FNumberAt(double focalLengthMm);
    static double FullWellElectrons(MastcamZEye eye);   ///< 21826 (left), 21818 (right).
    static double ReadNoiseElectrons(MastcamZEye eye);  ///< 22 (left), 21 (right).
};

/// Authors one Mastcam-Z camera through filter `filter` as published, onto Camera `camera` and
/// the readout chain of RenderProduct `product`:
/// - The camera: a pinhole of `focalLengthMm` (26 to 110) over the 11.84 x 8.88 mm detector at
///   its f-number (`MastcamZ::FNumberAt`).
/// - Filter 0: an RGB sensor whose channels are Gaussians at Table 3's Bayer effective
///   wavelengths and half widths, peaking at the detector's ~40%; the chain qe, bayer
///   (`pattern`; the paper does not give the layout), noise, adc, demosaic.
/// - Filters 1 to 7: one channel, the filter's passband (Table 3's 2 x HWHM wide about its
///   effective wavelength, ramped over 2 nm about the half maxima, times 10^-ND for filter 7)
///   on the detector's response, Ernie's generic silicon curve scaled to the ~40% peak (the
///   paper publishes no curve, and the Bayer microfilters' differences in a narrow band are
///   left out); the camera's spectral range narrowed to the passband, four bands across it; the
///   chain qe, noise, adc.
/// - The sensor: the KAI-2020CM's gain, read noise and full well for `eye`, an 11-bit ADC.
/// Scene radiance past 830 nm follows what the scene gives there: materials' RGB albedo holds
/// its 830 nm value, so give materials reflectance spectra for the near-IR bands.
///
/// Reference: J. F. Bell III et al., "The Mars 2020 Perseverance Rover Mast Camera Zoom
/// (Mastcam-Z) Multispectral, Stereoscopic Imaging Investigation", Space Science Reviews 217:24,
/// 2021, https://doi.org/10.1007/s11214-020-00755-x: Tables 3 and 5 and the focal plane section.
/// @throws std::invalid_argument if `camera` is not a Camera, `product` not a RenderProduct,
/// `filter` outside 0 to 7 or `focalLengthMm` outside the zoom.
void AuthorMastcamZ(Scene& scene, const Path& camera, const Path& product, MastcamZEye eye,
                    int filter, double focalLengthMm = MastcamZ::kShortestFocalMm,
                    BayerPattern pattern = BayerPattern::Rggb);

/// The Mastcam-Z pair on stereo rig `rig` (an Xform, defined if missing, looking down its -z with
/// +y up): Cameras `rig`/Left and `rig`/Right 24.4 cm apart along its x, each toed in 1.15
/// degrees, authored as `AuthorMastcamZ` through filters `leftFilter` and `rightFilter`.
void AuthorMastcamZPair(Scene& scene, const Path& rig, const Path& leftProduct,
                        const Path& rightProduct, int leftFilter, int rightFilter,
                        double focalLengthMm = MastcamZ::kShortestFocalMm,
                        BayerPattern pattern = BayerPattern::Rggb);

}  // namespace ernie
