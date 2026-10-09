#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

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
    double nameNm;           ///< Table 4's band name.
    double effectiveNm;      ///< Table 4's effective wavelength.
    double fwhmNm;           ///< Table 4's full width at half maximum.
    double responsivity;     ///< Table 4's responsivity coefficient, (W/m^3/sr) / (DN/s).
    /// Table 3: the visible optics' or the UV optics' (section 3.2.1's text gives 5.5 and 5.1
    /// for the lens designs; the table's are the system's).
    double fNumber;
    double pixelPitchUm;     ///< The 9 um pixels, 4 x 4 summed in the UV (section 3.2).
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

}  // namespace ernie
