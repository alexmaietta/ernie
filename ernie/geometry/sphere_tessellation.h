// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/geometry/mesh_data.h"

#include <cstdint>
#include <optional>
#include <string_view>

namespace ernie {

/// How a sphere is divided into triangles.
enum class SphereTessellation {
    /// Rings of latitude and meridians: 2 `level` slices by `level`
    /// stacks. Cells crowd at the poles.
    Uv,
    /// An icosahedron's faces divided `level` times in four, its
    /// vertices pushed onto the sphere: 20 4^level near-equal
    /// triangles, no poles.
    Icosphere,
    /// A cube's six faces in `level` x `level` quads, projected
    /// equiangularly as planet Terrains' faces are
    /// (`CubeSphereLod::FaceDirection`).
    CubeSphere,
    /// HEALPix's 12 `level`^2 equal-area pixels (`level` is N_side),
    /// each two triangles: the grid of all-sky and planetary maps
    /// such as HiPS.
    Healpix,
};

/// The tessellation named "uv", "icosphere", "cubeSphere" or
/// "healpix"; none for anything else.
std::optional<SphereTessellation>
SphereTessellationNamed(std::string_view name);

/// The level `TessellatedSphere` uses for 0: some 2,000 to 5,000
/// triangles each (24 stacks, 4 subdivisions, 16 quads a face edge,
/// N_side 16).
std::uint32_t DefaultSphereLevel(SphereTessellation tessellation);

/// A sphere of `radius` about the origin, its poles on z, with
/// outward normals, counter-clockwise triangles seen from outside,
/// and equirectangular texture coordinates per corner: s the
/// longitude east of +x over 360 degrees from -180, t the latitude
/// from the south pole (0) to the north (1), a triangle across the
/// antimeridian taking its corners' longitudes on one side, and a
/// corner on a pole the mean of the triangle's other two. Vertices on
/// shared edges are shared, so the mesh is closed. `level` 0 takes
/// the default.
///
/// References:
/// - K. M. Gorski, E. Hivon, A. J. Banday, B. D. Wandelt, F. K.
///   Hansen, M. Reinecke and M. Bartelmann, "HEALPix: A Framework for
///   High-Resolution Discretization and Fast Analysis of Data
///   Distributed on the Sphere", The Astrophysical Journal
///   622(2):759-771, 2005 (the pixel boundaries, from each base
///   pixel's local coordinates as healpix_base's `xyf2loc`).
///   https://doi.org/10.1086/427976
/// - J. R. Baumgardner and P. O. Frederickson, "Icosahedral
///   Discretization of the Two-Sphere", SIAM Journal on Numerical
///   Analysis 22(6):1107-1115, 1985.
///   https://doi.org/10.1137/0722066
/// - C. Ronchi, R. Iacono and P. S. Paolucci, "The 'Cubed Sphere'",
///   Journal of Computational Physics 124(1):93-114, 1996.
///   https://doi.org/10.1006/jcph.1996.0047
/// @throws std::invalid_argument for a radius that is not positive
/// and finite, or a level beyond the tessellation's limit (UV 2 to
/// 4096 stacks, icosphere 9 subdivisions, cube sphere 1024 and
/// HEALPix 1024).
MeshData TessellatedSphere(float radius,
                           SphereTessellation tessellation,
                           std::uint32_t level = 0);

}  // namespace ernie
