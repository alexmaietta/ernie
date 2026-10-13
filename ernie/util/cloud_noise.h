// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <vector>

namespace ernie {

/// The tileable noise textures procedural cloud layers are carved
/// from (volume.slang's `cloudLayersDensity`): a Perlin-Worley shape
/// and a Worley detail, each repeating in all three axes, a local
/// weather map of the layers' coverage and a curl-noise turbulence,
/// both repeating across. Values over 0 to 1, x fastest.
///
/// References:
/// - A. Schneider and N. Vos, "The Real-Time Volumetric
///   Cloudscapes of Horizon: Zero Dawn", SIGGRAPH 2015 Advances in
///   Real-Time Rendering (Perlin-Worley shapes eroded by Worley
///   details).
/// - S. Hillaire, TileableVolumeNoise, 2017, MIT
///   (https://github.com/sebh/TileableVolumeNoise): the tileable
///   Worley noise and the textures' recipes.
/// - takram, three-geospatial's clouds (packages/clouds/src/shaders:
///   cloudShape, cloudShapeDetail, localWeather and turbulence),
///   2024, MIT, Copyright (c) 2024 Shota Matsuda: the frequencies and
///   weights here.
struct CloudNoise {
    std::size_t shapeSize = 128;
    /// shapeSize^3 values.
    std::vector<float> shape;
    std::size_t detailSize = 32;
    /// detailSize^3 values.
    std::vector<float> detail;
    std::size_t weatherSize = 512;
    /// weatherSize^2 texels of four channels (low, mid and high
    /// layers' coverage, then 1).
    std::vector<float> weather;
    std::size_t turbulenceSize = 128;
    /// turbulenceSize^2 texels of three channels, a unit curl's
    /// direction over 0 to 1.
    std::vector<float> turbulence;
};

/// Hillaire's tileable Worley noise at `p` (repeating over 0 to 1 in
/// each axis) with `cells` cells across: the squared distance to the
/// nearest cell's point, clamped to 1.
double TileableWorley(double x, double y, double z, double cells);

/// Periodic Perlin noise's fractal sum over `octaves` octaves, from
/// `frequency` (per axis) doubling, each half the last's weight, over
/// their weights' sum (about -1 to 1, not clamped).
double TileablePerlin(double x, double y, double z, double fx,
                      double fy, double fz, int octaves);

/// The cloud noise textures, generated on the CPU across its threads.
CloudNoise MakeCloudNoise();

/// `MakeCloudNoise`'s, read from Ernie's cache folder
/// (`ERNIE_CLOUD_CACHE`, else "clouds" in the user's cache folder),
/// else generated (some seconds) and kept there.
CloudNoise CachedCloudNoise();

}  // namespace ernie
