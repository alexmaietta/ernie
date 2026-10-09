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

}  // namespace ernie
