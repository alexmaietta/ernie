// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Small helpers the C++ demos share, as ernie.demo's helpers do the
// Python demos'.

#include "ernie/util/tensor.h"

#include <glm/vec3.hpp>

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie::demos {

/// uint8 (rows, cols, channels) images of one height and channel
/// count next to each other, with black gaps of `gap` columns.
Tensor SideBySide(std::span<const Tensor> images, int gap = 4);

/// `values` (rows, cols, as doubles) from `low` (blue) through cyan,
/// green and yellow to `high` (red), as a uint8 (rows, cols, 3)
/// image, as Python's `ernie.demo.heatmap`.
Tensor Heatmap(std::span<const double> values, std::int64_t rows,
               std::int64_t cols, double low, double high);

/// An sRGB-encoded colour in [0, 1] to linear.
glm::dvec3 SrgbToLinear(glm::dvec3 srgb);

/// The mean colour of a uint8 (rows, cols, 3 or 4) image, decoded
/// from sRGB to linear.
glm::dvec3 MeanLinearRgb(const Tensor& image);

/// The standard deviation of a uint8 image's colour channels (alpha
/// left out), for checking an image is not flat.
double ColourDeviation(const Tensor& image);

/// Where fetched data lives, as Python's `ernie.demo.cache_dir`:
/// $ERNIE_CACHE, else $XDG_CACHE_HOME/ernie, else ~/.cache/ernie.
std::filesystem::path CacheDir();

/// Whether the demos can fetch data: built with libcurl
/// (ERNIE_DEMO_DOWNLOADS) and $ERNIE_OFFLINE unset.
bool CanDownload();

/// Downloads `url` to `path`, its directory made as needed, through
/// `path` + ".part", renamed when complete.
/// @throws std::runtime_error if it cannot (`CanDownload`), or on a
/// failed request.
void Download(const std::string& url,
              const std::filesystem::path& path);

/// The bytes of `ranges` of `url`, each [first, last] inclusive, in
/// one request (HTTP range requests; several come back as
/// multipart/byteranges).
/// @throws std::runtime_error as `Download`, or if the server ignores
/// the ranges.
std::vector<std::string> FetchRanges(
    const std::string& url,
    std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);

/// Data file `name` the demos ship (such as LRO's ISDs): in
/// $ERNIE_DEMO_DATA, else the source tree's ISD fixtures, which the
/// Python package ships as ernie/demos/data.
/// @throws std::runtime_error if neither holds it.
std::filesystem::path DataFile(std::string_view name);

/// Throws std::runtime_error with `message` unless `condition` holds:
/// a demo's checks.
inline void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace ernie::demos
