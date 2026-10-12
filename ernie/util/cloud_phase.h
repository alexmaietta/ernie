// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/spectrum.h"

#include <vector>

namespace ernie {

/// The phase function of a cloud's water droplets in each of a
/// frame's bands, for volumes to read (volume.slang's `dropletAt`):
/// Mie scattering by a gamma distribution of droplet radii, its
/// narrow forward peak (within `kDropletForwardDegrees`) folded into
/// the extinction as unscattered light. The silver lining, the bright
/// sheen opposite the sun on cloud tops, the fogbow near 140 degrees
/// and the glory at 180 come from its lobes. Each band's row, for
/// single scattering, keeps the peak (as Bouthors et al. add it back
/// for light scattered once), over the extinction less its share; the
/// bands' mean, for light scattered many times, has it folded away.
///
/// References:
/// - A. Bouthors, F. Neyret and S. Lefebvre, "Real-time realistic
///   illumination and shading of stratiform clouds", Eurographics
///   Workshop on Natural Phenomena 2006 (the droplets' gamma
///   distribution, and the forward peak folded into the extinction,
///   their "modified Mie").
/// - A. Bouthors, F. Neyret, N. Max, E. Bruneton and C. Crassin,
///   "Interactive multiple anisotropic scattering in clouds", ACM
///   SIGGRAPH Symposium on Interactive 3D Graphics and Games 2008.
/// - J. E. Hansen and L. D. Travis, "Light scattering in planetary
///   atmospheres", Space Science Reviews 16(4):527-610, 1974, section
///   2.4 (distributions alike in effective radius and variance
///   scatter alike: the gamma's as a log-normal's).
///   https://doi.org/10.1007/BF00168069
struct DropletPhaseTable {
    /// Samples of each row, at scattering angles theta(u) = pi / 2
    /// (1 - cos pi u), u = i / (angles - 1): dense about the forward
    /// and backward directions, where the peaks are.
    int angles = 0;
    /// Rows of `angles` values (sr^-1): each band's, its peak kept,
    /// over 1 - its share (integrating to 1 / (1 - it)); then the
    /// bands' mean, its peak folded away (integrating to 1), for
    /// light scattered many times, whose colours blur.
    std::vector<float> values;
    /// The share of the light the forward peak scattered, the bands'
    /// mean, which the extinction loses (times 1 - it).
    double forwardFraction = 0.0;
};

/// The narrow forward peak's half angle folded into the extinction.
inline constexpr double kDropletForwardDegrees = 5.0;

/// The phase function of water droplets of effective radius
/// `effectiveRadiusUm` (micrometres) in a gamma distribution of width
/// `spread` (its gamma: effective variance 1 / (spread + 2)), at each
/// band's centre, sampled at `angles` angles.
DropletPhaseTable DropletPhase(double effectiveRadiusUm,
                               double spread,
                               const SpectralBands& bands,
                               int angles = 1024);

}  // namespace ernie
