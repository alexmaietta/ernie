// SPDX-License-Identifier: BSD-3-Clause
/// @file ies.h
/// @brief IES LM-63 photometric profiles of luminaires.
#pragma once

#include <glm/glm.hpp>

#include <filesystem>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief The axis an IES file's angles turn about.
/// In the luminaire's frame: x its length and 0 degree plane, y its
/// width, z up; it shines down -z.
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

/// @brief How a lamp's output changes as its luminaire tilts.
/// From TILT=INCLUDE or a tilt file; for lamps sensitive to burning
/// position, such as metal halide.
struct IesTilt {
    /// Lamp position: 1 vertical, 2 horizontal along x (0 degree
    /// plane), 3 horizontal along y (90 degree plane); 0 for none.
    int geometry = 0;
    /// Tilt angles, in degrees, increasing.
    std::vector<double> angles;
    /// Candela multipliers at each angle.
    std::vector<double> factors;

    /// @brief The luminaire's tilt for the lamp's geometry.
    /// Vertical lamps: the luminaire z's angle from vertical.
    /// Horizontal lamps: the lamp axis's elevation.
    /// @param x The luminaire's x axis in a z-up world (unit).
    /// @param y The luminaire's y axis (unit).
    /// @param z The luminaire's z axis (unit).
    /// @return The tilt, in degrees.
    double Angle(const glm::dvec3& x, const glm::dvec3& y, const glm::dvec3& z) const;

    /// @brief The multiplier at a tilt.
    /// Linear between the file's angles, held beyond; data up to 90
    /// degrees mirrors past it.
    /// @param degrees The tilt, in degrees.
    /// @return The factor; 1 without tilt data.
    double Factor(double degrees) const;
};

/// @brief A luminaire's luminous intensity from an IES LM-63 file.
/// Files may give a half, quadrant or one plane of a symmetric
/// luminaire; `Sample` mirrors them.
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
    /// The photometric type.
    IesType type = IesType::C;
    /// Vertical angles, in degrees, increasing.
    std::vector<double> vertical;
    /// Horizontal angles, in degrees, increasing.
    std::vector<double> horizontal;
    /// Candela, `horizontal.size()` rows of `vertical.size()`, with
    /// the file's multiplier and ballast factor applied.
    std::vector<double> candela;
    /// The tilt data.
    IesTilt tilt;

    /// @brief Intensity in a direction, bilinear between the file's
    /// angles and mirrored by its symmetry.
    /// @param theta Degrees from the nadir (-z).
    /// @param phi   Degrees around the nadir, from x towards y.
    /// @return Candela; 0 outside the file's angles.
    double Sample(double theta, double phi) const;

    /// @return The luminous flux over the sphere, in lumens.
    double Flux() const;

  private:
    /// Candela at the file's vertical angle `v` and horizontal angle
    /// `h`, in its coordinates.
    double Table(double v, double h) const;
};

/// @brief Parses LM-63 text, versions 1986 to 2019, types A, B, C.
/// @param text      The file's contents.
/// @param directory Where a TILT=<file> is resolved.
/// @return The profile with its tilt data.
/// @throws std::invalid_argument If the text is not LM-63, has an
///         unknown type, or has angles out of order.
/// @throws std::runtime_error If a tilt file cannot be read.
IesProfile ParseIes(std::string_view text, const std::filesystem::path& directory = {});

/// @brief Reads and parses an IES file; tilt files resolve beside it.
/// @param path The file.
/// @return The profile.
/// @throws std::runtime_error If it cannot be read.
IesProfile ReadIes(const std::filesystem::path& path);

}  // namespace ernie
