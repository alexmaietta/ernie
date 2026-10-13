// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Small helpers the C++ demos share, as ernie.demo's helpers do the
// Python demos'.

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/util/tensor.h"

#include <glm/vec3.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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

/// A smooth planet of `radius` metres at `path`: a planet Terrain of
/// flat cube faces, its surface at the world's z = 0 (its centre
/// `radius` below the origin), drawn with a Material of normal albedo
/// `albedo` (linear RGB) at `material`. The ground under an
/// atmosphere, which is the air alone.
void DefinePlanetSurface(Scene& scene, const Path& path,
                         double radius, const glm::dvec3& albedo,
                         const Path& material);

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

/// How far a download has come, shared with the thread fetching it.
struct DownloadProgress {
    /// Bytes in hand, a resumed part's included.
    std::atomic<std::uint64_t> bytes = 0;
    /// Bytes in all; 0 until the server says.
    std::atomic<std::uint64_t> total = 0;
    /// Set to stop the download, keeping its part to resume.
    std::atomic<bool> cancel = false;
};

/// Downloads `url` to `path`, its directory made as needed: streamed
/// into `path` + ".part", renamed when complete. A part left by a
/// download stopped or failed before is resumed (an HTTP range
/// request; from the start if the server ignores it), and failed
/// attempts resume up to five times, going to `report` (else
/// printed). `progress`, when given, follows it and can cancel it.
/// @throws std::runtime_error if it cannot (`CanDownload`), on a
/// failed request, or when cancelled.
void Download(
    const std::string& url, const std::filesystem::path& path,
    const std::function<void(const std::string&)>& report = {},
    DownloadProgress* progress = nullptr);

/// `Download` on a thread of its own, so a demo runs while its data
/// comes: its progress and outcome are messages for the demo to show
/// as it steps (`TakeMessages`, such as into the viewer's log), and
/// its bytes so far for a progress bar (`Fraction`, `Status`).
class AsyncDownload {
  public:
    /// A message, and whether it is an error.
    struct Message {
        std::string text;
        bool error = false;
    };

    /// Starts fetching `url` to `path`.
    AsyncDownload(std::string url, std::filesystem::path path);
    /// Cancels the download (its part kept, to resume next time) and
    /// waits for it to stop.
    ~AsyncDownload();
    AsyncDownload(const AsyncDownload&) = delete;
    AsyncDownload& operator=(const AsyncDownload&) = delete;

    /// The messages since the last call.
    std::vector<Message> TakeMessages();
    /// Whether it has ended, and if so whether `path` holds the data.
    bool Done() const;
    bool Succeeded() const;

    /// Bytes in hand, and in all (0 until the server says).
    std::uint64_t Bytes() const {
        return progress_.bytes;
    }
    std::uint64_t Total() const {
        return progress_.total;
    }
    /// The fraction in hand, 0 to 1 (0 until the total is known).
    double Fraction() const;
    /// Megabytes in hand and in all, as "120 of 2,100 MB".
    std::string Status() const;
    /// Stops the download, keeping its part to resume.
    void Cancel() {
        progress_.cancel = true;
    }

  private:
    mutable std::mutex mutex_;
    std::vector<Message> messages_;
    bool done_ = false;
    bool succeeded_ = false;
    DownloadProgress progress_;
    std::thread thread_;
};

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
