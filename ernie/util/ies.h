// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <filesystem>
#include <string_view>
#include <vector>

namespace ernie {

/// A luminaire's photometric profile from an IES LM-63 file (type C photometry): its luminous
/// intensity in candela over the vertical angle theta from its nadir (0, straight down its axis)
/// to its zenith (180 degrees) and the horizontal angle phi around the axis. Files give one half,
/// quadrant or single plane of a symmetric luminaire, which `Sample` mirrors.
///
/// References:
/// - IES LM-63-19, Approved Method: IES Standard File Format for the Electronic Transfer of
///   Photometric Data and Related Information, Illuminating Engineering Society, 2019.
/// - I. Ashdown, "Parsing the IESNA LM-63 Photometric Data File", 1998,
///   http://lumen.iee.put.poznan.pl/kw/iesna.txt
struct IesProfile {
    std::vector<double> vertical;    ///< Degrees, increasing, within 0 to 180.
    std::vector<double> horizontal;  ///< Degrees, increasing, from 0 (or 90 for 90 to 270).
    /// Candela, `horizontal.size()` rows of `vertical.size()`, times the file's candela multiplier
    /// and ballast factor.
    std::vector<double> candela;

    /// Candela towards `theta` degrees from the nadir and `phi` degrees around the axis, linear in
    /// both between the file's angles, mirrored by the file's symmetry; 0 outside its vertical
    /// angles.
    double Sample(double theta, double phi) const;

    /// The luminous flux, lumens: the intensity integrated over the sphere.
    double Flux() const;
};

/// The profile of LM-63 `text` (any of the 1986 to 2019 versions). A TILT=INCLUDE block is
/// skipped; tilt files are not read.
/// @throws std::invalid_argument for text that is not an LM-63 file, photometry other than type
/// C, or angles out of order.
IesProfile ParseIes(std::string_view text);

/// `ParseIes` of the file at `path`.
/// @throws std::runtime_error if it cannot be read.
IesProfile ReadIes(const std::filesystem::path& path);

}  // namespace ernie
