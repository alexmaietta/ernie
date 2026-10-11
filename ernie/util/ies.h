// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

#include <filesystem>
#include <string_view>
#include <vector>

namespace ernie {

/// An IES file's photometry: the axis its angles turn about, in the
/// luminaire's frame (x its length and its 0 degree plane, y its
/// width, z up; it shines down -z).
enum class IesType {
    /// Type C: about the vertical axis z. The vertical angle runs
    /// from the nadir (0) to the zenith (180); the horizontal angle
    /// turns around z from x towards y. Most luminaires.
    C = 1,
    /// Type B: about the width y (floodlights). The vertical angle
    /// turns the plane through y, from the nadir towards x; the
    /// horizontal angle leaves that plane, towards -y; both -90 to
    /// 90.
    B = 2,
    /// Type A: about the length x (automotive lamps). The vertical
    /// angle turns the plane through x, from the nadir towards y; the
    /// horizontal angle leaves that plane, towards x.
    A = 3,
};

/// An IES file's tilt data (TILT=INCLUDE, or a tilt file): how its
/// lamp's output changes as the luminaire tilts, for lamps whose
/// output depends on their burning position (metal halide).
struct IesTilt {
    /// The lamp's position in the luminaire: 1 vertical (base up or
    /// down), 2 horizontal along the 0 degree plane (x), 3 horizontal
    /// along the 90 degree plane (y); 0 without tilt data.
    int geometry = 0;
    std::vector<double> angles;  ///< Degrees, increasing.
    std::vector<double>
        factors;  ///< Multiplying the candela at each angle.

    /// The luminaire's tilt in degrees for the lamp's geometry, from
    /// its axes in a z-up world (unit vectors): for a vertical lamp
    /// the angle of the luminaire's z from vertical, for a horizontal
    /// one the lamp axis's elevation above horizontal.
    double AngleOf(const glm::dvec3& x, const glm::dvec3& y,
                   const glm::dvec3& z) const;

    /// The factor at `degrees`, linear between the file's angles and
    /// held beyond them; past 90 degrees, data up to 90 mirrors about
    /// it. 1 without tilt data.
    double Factor(double degrees) const;
};

/// A luminaire's photometric profile from an IES LM-63 file: its
/// luminous intensity in candela over its file's angles (`IesType`),
/// which `Sample` reads in any direction. Files give a half, quadrant
/// or single plane of a symmetric luminaire, which `Sample` mirrors.
///
/// References:
/// - IES LM-63-19, Approved Method: IES Standard File Format for the
///   Electronic Transfer of Photometric Data and Related Information,
///   Illuminating Engineering Society, 2019.
/// - I. Ashdown, "Parsing the IESNA LM-63 Photometric Data File",
///   1998, http://lumen.iee.put.poznan.pl/kw/iesna.txt
/// - G. Ward, Radiance ies2rad, source.cal and tilt.cal (type B
///   coordinates and tilt angles),
///   https://github.com/NREL/Radiance/tree/master/src/cv
struct IesProfile {
    IesType type = IesType::C;
    std::vector<double> vertical;    ///< Degrees, increasing.
    std::vector<double> horizontal;  ///< Degrees, increasing.
    /// Candela, `horizontal.size()` rows of `vertical.size()`, times
    /// the file's candela multiplier and ballast factor.
    std::vector<double> candela;
    IesTilt tilt;

    /// Candela towards `theta` degrees from the nadir (-z) and `phi`
    /// degrees around it from x towards y, in whichever coordinates
    /// the file uses: linear in both of its angles between those it
    /// gives, mirrored by its symmetry; 0 outside its angles.
    double Sample(double theta, double phi) const;

    /// The luminous flux, lumens: the intensity integrated over the
    /// sphere.
    double Flux() const;

  private:
    /// Candela at the file's vertical angle `v` and horizontal angle
    /// `h`, in its coordinates.
    double Table(double v, double h) const;
};

/// The profile of LM-63 `text` (any of the 1986 to 2019 versions), of
/// type A, B or C, with its tilt data from a TILT=INCLUDE block or a
/// TILT=<file> named relative to `directory`.
/// @throws std::invalid_argument for text that is not an LM-63 file,
/// an unknown photometric type, or angles out of order;
/// std::runtime_error for a tilt file that cannot be read.
IesProfile ParseIes(std::string_view text,
                    const std::filesystem::path& directory = {});

/// `ParseIes` of the file at `path`, its tilt file beside it.
/// @throws std::runtime_error if it cannot be read.
IesProfile ReadIes(const std::filesystem::path& path);

}  // namespace ernie
