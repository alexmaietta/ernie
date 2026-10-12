// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/cloud_phase.h"

#include <cstdint>
#include <vector>

namespace ernie {

/// The scattering orders' sets the canonical transport keeps apart
/// (2, 3-4, 5-6, 7-8, 9-12, 13-18, 19-30, 31 and more), each with its
/// own collector: low orders stay anisotropic and near the view ray,
/// high ones diffuse and spread.
inline constexpr int kCloudOrderSets = 8;

/// The order set of light scattered `order` times (2 or more).
int CloudOrderSet(int order);

/// The canonical transport table's lattice. Lengths are optical
/// depths of the droplets' extinction less their forward peak
/// (`DropletPhase`).
struct CloudTransportShape {
    /// The light's cosines to the slab's normal: (i + 1) / count.
    int lightCosines = 8;
    /// The slab's optical thicknesses: 0.25 * 2^i.
    int thicknesses = 10;
    /// The view point's depths, as fractions of the thickness: (1 -
    /// cos pi u) / 2, u = i / (count - 1), dense near both faces.
    int depths = 9;
    /// Bins of the cosine of the light's direction of travel to the
    /// slab's inward normal, over [-1, 1] (negative going back up).
    int viewCosines = 16;
    /// Bins of its azimuth from the light's, over [0, pi]: symmetric
    /// about the light's plane.
    int azimuths = 8;

    double LightCosine(int i) const;
    double Thickness(int i) const;
    double DepthFraction(int i) const;
    /// The cosine at bin `i`'s centre.
    double ViewCosine(int i) const;
    /// The azimuth at bin `i`'s centre.
    double Azimuth(int i) const;
    /// Entries per order set.
    int Cells() const;
    /// Entry (light, thickness, depth, view cosine, azimuth)'s index
    /// within an order set.
    int Cell(int light, int thickness, int depth, int viewCosine,
             int azimuth) const;
};

/// Light transport in a plane-parallel slab of cloud lit from above
/// by a distant source, for each set of scattering orders (2 and
/// more): the radiance at a view point inside it in each direction of
/// travel, and where on the lit face that light entered, its
/// collector. Lit faces of any cloud, taken locally as such slabs,
/// read their multiple scattering from it.
///
/// References:
/// - A. Bouthors, F. Neyret, N. Max, E. Bruneton and C. Crassin,
///   "Interactive multiple anisotropic scattering in clouds", ACM
///   SIGGRAPH Symposium on Interactive 3D Graphics and Games 2008
///   (the canonical transport function and its collectors, by Monte
///   Carlo in slabs; stored here as a lattice rather than fitted).
/// - S. Chandrasekhar, "Radiative Transfer", Dover 1960
/// (plane-parallel
///   slabs).
struct CloudTransportTable {
    CloudTransportShape shape;
    /// Per order set, per cell: the radiance per unit irradiance
    /// normal to the light (sr^-1), then the collector's centre on
    /// the lit face from the view point's foot along the light's
    /// azimuth and across it (towards the view's side), and its
    /// spread (their standard deviation), in optical depths.
    std::vector<float> values;
    /// Light scattered once's radiance per cell, as `values`' first,
    /// to check against its analytic form.
    std::vector<float> single;

    /// Set `set`, cell `cell`'s `component` (0 radiance, 1 and 2 the
    /// collector's centre, 3 its spread).
    float At(int set, int cell, int component) const {
        return values[(static_cast<std::size_t>(set) * shape.Cells() +
                       cell) *
                          4 +
                      component];
    }
};

struct CloudTransportOptions {
    CloudTransportShape shape;
    /// Photons per slab (light cosine and thickness).
    int photons = 1 << 17;
    std::uint64_t seed = 1;
};

/// The canonical transport for droplets scattering as `phase`'s mean
/// row (its forward peak folded away), by Monte Carlo in each slab of
/// the lattice: photons enter the lit face and scatter without
/// absorption, their crossings of the depths' planes tallied by order
/// set and direction, with where each entered.
CloudTransportTable
SimulateCloudTransport(const DropletPhaseTable& phase,
                       const CloudTransportOptions& options = {});

/// The canonical transport for Bouthors et al.'s cloud droplets (an
/// effective radius of 7 um, gamma 2) at `options`, its radiance and
/// collectors (not `single`): read from Ernie's cache folder
/// (`ERNIE_CLOUD_CACHE`, else "clouds" in the user's cache folder),
/// else simulated, some seconds, and kept there.
CloudTransportTable
CachedCloudTransport(const CloudTransportOptions& options = {});

}  // namespace ernie
