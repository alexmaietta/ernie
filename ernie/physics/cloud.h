// SPDX-License-Identifier: BSD-3-Clause
/// @file cloud.h
/// @brief Clouds: droplet phase functions, multiple-scattering transport in slabs, and the noise
/// textures cloud layers are carved from.
#pragma once

#include "ernie/physics/spectrum.h"
#include "ernie/util/math.h"

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace ernie {

// ----------------------------------------------------------------------------------------------
// Droplet phase functions

/// @brief Cloud droplets' phase function in each band.
/// Mie scattering by gamma-distributed radii, read by volume.slang's `dropletAt`. Its lobes give
/// the silver lining, fogbow and glory. The narrow forward peak is folded into the extinction for
/// multiple scattering and kept for single scattering.
///
/// References:
/// - A. Bouthors, F. Neyret and S. Lefebvre, "Real-time realistic illumination and shading of
///   stratiform clouds", Eurographics Workshop on Natural Phenomena 2006 (the droplets' gamma
///   distribution, and the forward peak folded into the extinction, their "modified Mie").
/// - A. Bouthors, F. Neyret, N. Max, E. Bruneton and C. Crassin, "Interactive multiple
///   anisotropic scattering in clouds", ACM SIGGRAPH Symposium on Interactive 3D Graphics and
///   Games 2008.
/// - J. E. Hansen and L. D. Travis, "Light scattering in planetary atmospheres", Space Science
///   Reviews 16(4):527-610, 1974, section 2.4 (distributions alike in effective radius and
///   variance scatter alike: the gamma's as a log-normal's). https://doi.org/10.1007/BF00168069
struct DropletPhaseTable {
    /// Samples per row, at theta(u) = pi/2 (1 - cos pi u) for u = i / (angles - 1): dense near the
    /// forward and back peaks.
    std::size_t angles = 0;
    /// Rows of `angles` values (sr^-1). One per band with its peak kept, over 1 - forwardFraction;
    /// then the bands' mean with the peak folded away, for multiple scattering.
    std::vector<float> values;
    /// Share of light in the forward peak, averaged over bands; the extinction is scaled by 1
    /// minus it.
    double forwardFraction = 0.0;

    /// @brief The scattering angle of a sample.
    /// @param i The sample's index along a row.
    /// @return theta(i / (angles - 1)), in radians.
    double Angle(std::size_t i) const {
        return std::numbers::pi * CosineSpacing(i, angles);
    }
};

/// Half angle of the forward peak folded into the extinction, in degrees.
inline constexpr double kDropletForwardDegrees = 5.0;

/// @brief Computes droplets' phase function at each band's centre.
/// @param effectiveRadiusUm Effective droplet radius, in micrometres.
/// @param spread            The gamma distribution's width parameter; effective variance is
///                          1 / (spread + 2).
/// @param bands             The frame's bands.
/// @param angles            Samples per row.
/// @return The table.
DropletPhaseTable DropletPhase(double effectiveRadiusUm, double spread, const SpectralBands& bands,
                               std::size_t angles = 1024);

// ----------------------------------------------------------------------------------------------
// Transport in slabs

/// Scattering order sets kept apart, each with its own collector: 2, 3-4, 5-6, 7-8, 9-12, 13-18,
/// 19-30, 31+. Low orders stay near the view ray; high ones diffuse.
inline constexpr std::size_t kCloudOrderSets = 8;

/// @brief Finds the order set an order falls in.
/// @param order Times scattered, 2 or more.
/// @return The order set.
std::size_t CloudOrderSet(std::size_t order);

/// @brief The canonical transport table's lattice.
/// Lengths are optical depths of extinction less the forward peak (`DropletPhase`).
struct CloudTransportShape {
    /// Light cosines to the slab normal: (i + 1) / count.
    std::size_t lightCosines = 8;
    /// Slab optical thicknesses: 0.25 * 2^i.
    std::size_t thicknesses = 10;
    /// View depths as thickness fractions, (1 - cos pi u) / 2 with u = i / (count - 1): dense near
    /// both faces.
    std::size_t depths = 9;
    /// Bins of the travel direction's cosine to the inward normal, over [-1, 1]; negative goes
    /// back up.
    std::size_t viewCosines = 16;
    /// Bins of azimuth from the light's, over [0, pi]; symmetric about the light's plane.
    std::size_t azimuths = 8;

    /// @return Light cosine `i`.
    double LightCosine(std::size_t i) const;
    /// @return Optical thickness `i`.
    double Thickness(std::size_t i) const;
    /// @return Depth `i` as a fraction of the thickness.
    double DepthFraction(std::size_t i) const;
    /// @return The cosine at bin `i`'s centre.
    double ViewCosine(std::size_t i) const;
    /// @return The azimuth at bin `i`'s centre.
    double Azimuth(std::size_t i) const;
    /// @return Entries per order set.
    std::size_t Cells() const;
    /// @brief Indexes an entry within an order set.
    /// @param light      The light cosine index.
    /// @param thickness  The thickness index.
    /// @param depth      The depth index.
    /// @param viewCosine The view cosine bin.
    /// @param azimuth    The azimuth bin.
    /// @return The cell index.
    std::size_t Cell(std::size_t light, std::size_t thickness, std::size_t depth,
                     std::size_t viewCosine, std::size_t azimuth) const;
};

/// @brief Light transport in a plane-parallel cloud slab lit from above, per scattering order set.
/// Gives radiance at a point inside in each direction, and where on the lit face it entered (its
/// collector). Lit faces of any cloud, treated locally as slabs, read multiple scattering from it.
///
/// References:
/// - A. Bouthors, F. Neyret, N. Max, E. Bruneton and C. Crassin, "Interactive multiple
///   anisotropic scattering in clouds", ACM SIGGRAPH Symposium on Interactive 3D Graphics and
///   Games 2008 (the canonical transport function and its collectors, by Monte Carlo in slabs;
///   stored here as a lattice rather than fitted).
/// - S. Chandrasekhar, "Radiative Transfer", Dover 1960 (plane-parallel slabs).
struct CloudTransportTable {
    /// The lattice.
    CloudTransportShape shape;
    /// Four values per order set and cell: radiance per unit irradiance normal to the light
    /// (sr^-1); the collector's centre on the lit face along and across the light's azimuth; and
    /// its spread. Lengths in optical depths.
    std::vector<float> values;
    /// Single-scattering radiance per cell, to check against its analytic form.
    std::vector<float> single;

    /// @brief Reads one value.
    /// @param set       The order set.
    /// @param cell      The cell (`CloudTransportShape::Cell`).
    /// @param component 0 radiance, 1-2 collector centre, 3 spread.
    /// @return The value.
    float At(std::size_t set, std::size_t cell, std::size_t component) const {
        return values[(set * shape.Cells() + cell) * 4 + component];
    }
};

/// @brief How `SimulateCloudTransport` runs.
struct CloudTransportOptions {
    /// The lattice.
    CloudTransportShape shape;
    /// Photons per slab (light cosine and thickness).
    std::size_t photons = 1 << 17;
    /// The random seed.
    std::uint64_t seed = 1;
};

/// @brief Simulates the canonical transport by Monte Carlo.
/// Photons enter each slab's lit face and scatter without absorption; crossings of the depth
/// planes are tallied by order set and direction with their entry points.
/// @param phase   Droplets; the mean row (peak folded away) is used.
/// @param options The lattice and sampling.
/// @return The table.
CloudTransportTable SimulateCloudTransport(const DropletPhaseTable& phase,
                                           const CloudTransportOptions& options = {});

/// @brief The transport for Bouthors et al.'s droplets (7 um, gamma 2), cached between runs.
/// Read from `ERNIE_CLOUD_CACHE` or "clouds" in the user's cache folder; else simulated (some
/// seconds) and stored.
/// @param options The lattice and sampling.
/// @return The table, without `single`.
CloudTransportTable CachedCloudTransport(const CloudTransportOptions& options = {});

// ----------------------------------------------------------------------------------------------
// Noise textures

/// @brief The noise textures cloud layers are carved from.
/// Read by volume.slang's `cloudLayersDensity`. Values in [0, 1], x fastest; every texture tiles.
///
/// References:
/// - A. Schneider and N. Vos, "The Real-Time Volumetric Cloudscapes of Horizon: Zero Dawn",
///   SIGGRAPH 2015 Advances in Real-Time Rendering (Perlin-Worley shapes eroded by Worley
///   details).
/// - S. Hillaire, TileableVolumeNoise, 2017, MIT (https://github.com/sebh/TileableVolumeNoise):
///   the textures' recipes.
/// - takram, three-geospatial's clouds (packages/clouds/src/shaders: cloudShape,
///   cloudShapeDetail, localWeather and turbulence), 2024, MIT, Copyright (c) 2024 Shota
///   Matsuda: the frequencies and weights here.
struct CloudNoise {
    /// Side of the shape volume, in texels.
    std::size_t shapeSize = 128;
    /// Perlin-Worley shape: shapeSize^3 values.
    std::vector<float> shape;
    /// Side of the detail volume, in texels.
    std::size_t detailSize = 32;
    /// Worley detail: detailSize^3 values.
    std::vector<float> detail;
    /// Side of the weather map, in texels.
    std::size_t weatherSize = 512;
    /// weatherSize^2 texels of four channels: low, mid and high layers' coverage, then 1.
    std::vector<float> weather;
    /// Side of the turbulence map, in texels.
    std::size_t turbulenceSize = 128;
    /// turbulenceSize^2 texels of three channels: a unit curl direction mapped to [0, 1].
    std::vector<float> turbulence;
};

/// @brief Generates the cloud noise across CPU threads.
/// @param shapeSize      Side of the shape volume, in texels.
/// @param detailSize     Side of the detail volume, in texels.
/// @param weatherSize    Side of the weather map, in texels.
/// @param turbulenceSize Side of the turbulence map, in texels.
/// @return The textures.
CloudNoise MakeCloudNoise(std::size_t shapeSize = 128, std::size_t detailSize = 32,
                          std::size_t weatherSize = 512, std::size_t turbulenceSize = 128);

/// @brief `MakeCloudNoise`, cached between runs.
/// Read from `ERNIE_CLOUD_CACHE` or "clouds" in the user's cache folder; else generated (some
/// seconds) and stored there.
/// @return The textures.
CloudNoise CachedCloudNoise();

}  // namespace ernie
