// SPDX-License-Identifier: BSD-3-Clause
#include "util.h"

#include "ernie/scene/material_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/xform.h"

#ifdef ERNIE_DEMO_DOWNLOADS
#include <curl/curl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>

namespace ernie::demos {

namespace {

void CheckImage(const Tensor& image) {
    const std::vector<std::int64_t>& shape = image.Shape();
    if (shape.size() != 3 || image.Type() != DType::UInt8 ||
        image.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "Expected a uint8 (rows, cols, channels) CPU image");
    }
}

double Decode(double c) {
    return c <= 0.04045 ? c / 12.92
                        : std::pow((c + 0.055) / 1.055, 2.4);
}

#ifdef ERNIE_DEMO_DOWNLOADS
/// A GET of `url` with an optional Range header: the body, its
/// Content-Type and the HTTP status.
struct Response {
    std::string body;
    std::string contentType;
    long status = 0;
};

Response GetOnce(const std::string& url, const std::string& range) {
    static std::once_flag initialised;
    std::call_once(initialised,
                   [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        throw std::runtime_error("libcurl would not start");
    }
    Response response;
    const auto write = +[](char* data, std::size_t size,
                           std::size_t count, void* user) {
        static_cast<std::string*>(user)->append(data, size * count);
        return size * count;
    };
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "ernie-demos");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);
    if (!range.empty()) {
        curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
    }
    const CURLcode result = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status);
    const char* type = nullptr;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &type);
    if (type != nullptr) {
        response.contentType = type;
    }
    curl_easy_cleanup(curl);
    if (result != CURLE_OK) {
        throw std::runtime_error("Fetching " + url + " failed: " +
                                 curl_easy_strerror(result));
    }
    return response;
}

/// `GetOnce`, tried up to three times: servers drop connections
/// mid-transfer now and then. Failed attempts go to `report`, else
/// are printed.
Response
Get(const std::string& url, const std::string& range,
    const std::function<void(const std::string&)>& report = {}) {
    for (int attempt = 1;; ++attempt) {
        try {
            return GetOnce(url, range);
        } catch (const std::runtime_error& error) {
            if (attempt == 3) {
                throw;
            }
            const std::string message =
                std::string(error.what()) + "; trying again";
            if (report) {
                report(message);
            } else {
                std::printf("%s\n", message.c_str());
            }
        }
    }
}
#endif

void RequireDownloads() {
    if (!CanDownload()) {
        throw std::runtime_error(
            "The demos cannot fetch data: built without libcurl "
            "(ERNIE_DEMO_DOWNLOADS), or $ERNIE_OFFLINE is set");
    }
}

}  // namespace

Tensor Heatmap(std::span<const double> values, std::int64_t rows,
               std::int64_t cols, double low, double high) {
    constexpr double kStops[5][3] = {{0, 0, 255},
                                     {0, 255, 255},
                                     {0, 255, 0},
                                     {255, 255, 0},
                                     {255, 0, 0}};
    Tensor image = Tensor::Zeros(DType::UInt8, {rows, cols, 3});
    auto* data = static_cast<std::uint8_t*>(image.Data());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double t = std::clamp((values[i] - low) /
                                        std::max(high - low, 1e-12),
                                    0.0, 1.0);
        const double position = t * 4.0;
        const int index = std::min(static_cast<int>(position), 3);
        const double fraction = position - index;
        for (int c = 0; c < 3; ++c) {
            data[i * 3 + static_cast<std::size_t>(c)] =
                static_cast<std::uint8_t>(
                    (1.0 - fraction) * kStops[index][c] +
                    fraction * kStops[index + 1][c]);
        }
    }
    return image;
}

Tensor SideBySide(std::span<const Tensor> images, int gap) {
    if (images.empty()) {
        throw std::invalid_argument("SideBySide needs images");
    }
    for (const Tensor& image : images) {
        CheckImage(image);
    }
    const std::int64_t rows = images[0].Shape()[0];
    const std::int64_t channels = images[0].Shape()[2];
    std::int64_t cols =
        gap * static_cast<std::int64_t>(images.size() - 1);
    for (const Tensor& image : images) {
        if (image.Shape()[0] != rows ||
            image.Shape()[2] != channels) {
            throw std::invalid_argument(
                "SideBySide needs images of one height and channel "
                "count");
        }
        cols += image.Shape()[1];
    }
    Tensor result =
        Tensor::Zeros(DType::UInt8, {rows, cols, channels});
    auto* to = static_cast<std::uint8_t*>(result.Data());
    std::int64_t left = 0;
    for (const Tensor& image : images) {
        const std::int64_t width = image.Shape()[1];
        const auto* from =
            static_cast<const std::uint8_t*>(image.Data());
        for (std::int64_t row = 0; row < rows; ++row) {
            std::memcpy(to + (row * cols + left) * channels,
                        from + row * width * channels,
                        static_cast<std::size_t>(width * channels));
        }
        left += width + gap;
    }
    return result;
}

void DefinePlanetSurface(Scene& scene, const Path& path,
                         double radius, const glm::dvec3& albedo,
                         const Path& material) {
    constexpr std::int64_t kSide = 65;
    Terrain::Define(scene, path)
        .SetHeights(Tensor::Copy<float>(
            std::vector<float>(6 * kSide * kSide, 0.0f),
            {6, kSide, kSide}))
        .SetPlanetRadius(radius)
        .SetMaterialBinding(material);
    SetXform(scene, path,
             {.translate = glm::dvec3(0.0, 0.0, -radius)});
    Material::Define(scene, material).SetAlbedo(albedo);
}

glm::dvec3 SrgbToLinear(glm::dvec3 srgb) {
    return {Decode(srgb.x), Decode(srgb.y), Decode(srgb.z)};
}

glm::dvec3 MeanLinearRgb(const Tensor& image) {
    CheckImage(image);
    const std::int64_t pixels = image.Shape()[0] * image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    glm::dvec3 sum(0.0);
    for (std::int64_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            sum[c] += texels[i * channels + c];
        }
    }
    return SrgbToLinear(sum / (255.0 * static_cast<double>(pixels)));
}

double ColourDeviation(const Tensor& image) {
    CheckImage(image);
    const std::int64_t pixels = image.Shape()[0] * image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    double sum = 0.0;
    double squares = 0.0;
    for (std::int64_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double value = texels[i * channels + c];
            sum += value;
            squares += value * value;
        }
    }
    const double count = 3.0 * static_cast<double>(pixels);
    const double mean = sum / count;
    return std::sqrt(std::max(squares / count - mean * mean, 0.0));
}

std::filesystem::path CacheDir() {
    if (const char* cache = std::getenv("ERNIE_CACHE");
        cache != nullptr && *cache != '\0') {
        return cache;
    }
    std::filesystem::path base;
    if (const char* xdg = std::getenv("XDG_CACHE_HOME");
        xdg != nullptr && *xdg != '\0') {
        base = xdg;
    } else if (const char* home = std::getenv("HOME")) {
        base = std::filesystem::path(home) / ".cache";
    } else {
        base = std::filesystem::temp_directory_path();
    }
    return base / "ernie";
}

bool CanDownload() {
#ifdef ERNIE_DEMO_DOWNLOADS
    const char* offline = std::getenv("ERNIE_OFFLINE");
    return offline == nullptr || *offline == '\0';
#else
    return false;
#endif
}

void Download(const std::string& url,
              const std::filesystem::path& path,
              const std::function<void(const std::string&)>& report,
              DownloadProgress* progress) {
    RequireDownloads();
#ifdef ERNIE_DEMO_DOWNLOADS
    static std::once_flag initialised;
    std::call_once(initialised,
                   [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    std::filesystem::create_directories(path.parent_path());
    std::filesystem::path part = path;
    part += ".part";
    DownloadProgress own;
    DownloadProgress& state = progress != nullptr ? *progress : own;
    // One attempt, appending to the part from `have` bytes.
    struct Attempt {
        CURL* curl = nullptr;
        std::FILE* file = nullptr;
        std::filesystem::path part;
        std::uint64_t have = 0;
        bool checked = false;
        DownloadProgress* state = nullptr;
    };
    constexpr int kAttempts = 5;
    for (int attempt = 1;; ++attempt) {
        std::error_code error;
        const std::uintmax_t partBytes =
            std::filesystem::file_size(part, error);
        Attempt at;
        at.part = part;
        at.have = error ? 0 : partBytes;
        at.state = &state;
        at.file = std::fopen(part.string().c_str(),
                             at.have > 0 ? "ab" : "wb");
        if (at.file == nullptr) {
            throw std::runtime_error("Could not write " +
                                     part.string());
        }
        at.curl = curl_easy_init();
        if (at.curl == nullptr) {
            std::fclose(at.file);
            throw std::runtime_error("libcurl would not start");
        }
        // A server that ignores the range sends the whole file: the
        // part starts again.
        const auto write = +[](char* data, std::size_t size,
                               std::size_t count, void* user) {
            auto* a = static_cast<Attempt*>(user);
            if (!a->checked) {
                a->checked = true;
                long status = 0;
                curl_easy_getinfo(a->curl, CURLINFO_RESPONSE_CODE,
                                  &status);
                if (a->have > 0 && status != 206) {
                    a->file = std::freopen(a->part.string().c_str(),
                                           "wb", a->file);
                    a->have = 0;
                    if (a->file == nullptr) {
                        return std::size_t{0};
                    }
                }
            }
            return std::fwrite(data, size, count, a->file) * size;
        };
        const auto info =
            +[](void* user, curl_off_t total, curl_off_t now,
                curl_off_t, curl_off_t) {
                auto* a = static_cast<Attempt*>(user);
                a->state->bytes =
                    a->have + static_cast<std::uint64_t>(now);
                if (total > 0) {
                    a->state->total =
                        a->have + static_cast<std::uint64_t>(total);
                }
                return a->state->cancel ? 1 : 0;
            };
        curl_easy_setopt(at.curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(at.curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(at.curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(at.curl, CURLOPT_CONNECTTIMEOUT, 30L);
        // A transfer stalled below 1 kB/s for a minute is tried
        // again.
        curl_easy_setopt(at.curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
        curl_easy_setopt(at.curl, CURLOPT_LOW_SPEED_TIME, 60L);
        curl_easy_setopt(at.curl, CURLOPT_USERAGENT, "ernie-demos");
        curl_easy_setopt(at.curl, CURLOPT_WRITEFUNCTION, write);
        curl_easy_setopt(at.curl, CURLOPT_WRITEDATA, &at);
        curl_easy_setopt(at.curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(at.curl, CURLOPT_XFERINFOFUNCTION, info);
        curl_easy_setopt(at.curl, CURLOPT_XFERINFODATA, &at);
        if (at.have > 0) {
            curl_easy_setopt(at.curl, CURLOPT_RESUME_FROM_LARGE,
                             static_cast<curl_off_t>(at.have));
        }
        const CURLcode result = curl_easy_perform(at.curl);
        long status = 0;
        curl_easy_getinfo(at.curl, CURLINFO_RESPONSE_CODE, &status);
        curl_easy_cleanup(at.curl);
        const bool written =
            at.file != nullptr && std::fclose(at.file) == 0;
        if (result == CURLE_ABORTED_BY_CALLBACK) {
            throw std::runtime_error(
                "Stopped fetching " + url + " at " +
                std::to_string(state.bytes / 1000000) +
                " MB; it resumes next time");
        }
        // A part already whole: the range starts past the file's end.
        const bool whole = result == CURLE_HTTP_RETURNED_ERROR &&
                           status == 416 && at.have > 0;
        if ((result == CURLE_OK && written) || whole) {
            std::filesystem::rename(part, path);
            return;
        }
        const std::string message =
            "Fetching " + url + " failed: " +
            (written ? curl_easy_strerror(result)
                     : "could not write") +
            (attempt < kAttempts ? "; resuming" : "");
        if (attempt == kAttempts) {
            throw std::runtime_error(message);
        }
        if (report) {
            report(message);
        } else {
            std::printf("%s\n", message.c_str());
        }
    }
#else
    static_cast<void>(url);
    static_cast<void>(path);
    static_cast<void>(report);
    static_cast<void>(progress);
#endif
}

AsyncDownload::AsyncDownload(std::string url,
                             std::filesystem::path path) {
    messages_.push_back({"Fetching " + url});
    thread_ = std::thread([this, url = std::move(url),
                           path = std::move(path)] {
        const auto post = [this](std::string text, bool error) {
            const std::lock_guard lock(mutex_);
            messages_.push_back({std::move(text), error});
        };
        bool succeeded = false;
        try {
            Download(
                url, path,
                [&](const std::string& text) { post(text, false); },
                &progress_);
            post("Fetched " + path.filename().string(), false);
            succeeded = true;
        } catch (const std::exception& error) {
            post(error.what(), true);
        }
        const std::lock_guard lock(mutex_);
        done_ = true;
        succeeded_ = succeeded;
    });
}

AsyncDownload::~AsyncDownload() {
    Cancel();
    if (thread_.joinable()) {
        thread_.join();
    }
}

double AsyncDownload::Fraction() const {
    const std::uint64_t total = progress_.total;
    return total > 0
               ? std::min(1.0, static_cast<double>(progress_.bytes) /
                                   static_cast<double>(total))
               : 0.0;
}

std::string AsyncDownload::Status() const {
    // Thousands separated, as 2,100.
    const auto megabytes = [](std::uint64_t bytes) {
        std::string digits = std::to_string(bytes / 1000000);
        for (auto at = static_cast<std::ptrdiff_t>(digits.size()) - 3;
             at > 0; at -= 3) {
            digits.insert(static_cast<std::size_t>(at), ",");
        }
        return digits;
    };
    const std::uint64_t total = progress_.total;
    return megabytes(progress_.bytes) +
           (total > 0 ? " of " + megabytes(total) : std::string()) +
           " MB";
}

std::vector<AsyncDownload::Message> AsyncDownload::TakeMessages() {
    const std::lock_guard lock(mutex_);
    return std::exchange(messages_, {});
}

bool AsyncDownload::Done() const {
    const std::lock_guard lock(mutex_);
    return done_;
}

bool AsyncDownload::Succeeded() const {
    const std::lock_guard lock(mutex_);
    return succeeded_;
}

std::vector<std::string> FetchRanges(
    const std::string& url,
    std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    RequireDownloads();
    std::vector<std::string> parts;
#ifdef ERNIE_DEMO_DOWNLOADS
    if (ranges.empty()) {
        return parts;
    }
    std::string header;
    for (const auto& [first, last] : ranges) {
        header += (header.empty() ? "" : ",") +
                  std::to_string(first) + "-" + std::to_string(last);
    }
    Response response = Get(url, header);
    if (response.status != 206) {
        throw std::runtime_error(url + " ignored the range request");
    }
    if (ranges.size() == 1) {
        parts.push_back(std::move(response.body));
        return parts;
    }
    // multipart/byteranges (RFC 9110, section 14.6): parts between
    // "--boundary" lines, each with a Content-Range header.
    const std::string& type = response.contentType;
    const std::size_t at = type.find("boundary=");
    if (at == std::string::npos) {
        throw std::runtime_error(url + " sent no multipart boundary");
    }
    std::string boundary = type.substr(at + 9);
    boundary.erase(std::remove(boundary.begin(), boundary.end(), '"'),
                   boundary.end());
    const std::string delimiter = "--" + boundary;
    const std::string& body = response.body;
    std::map<std::uint64_t, std::string> byFirst;
    for (std::size_t start = body.find(delimiter);
         start != std::string::npos;) {
        const std::size_t headers = start + delimiter.size();
        const std::size_t end = body.find("\r\n\r\n", headers);
        if (end == std::string::npos) {
            break;
        }
        const std::string head = body.substr(headers, end - headers);
        const std::size_t range = head.find("bytes ");
        if (range != std::string::npos) {
            char* dash = nullptr;
            const std::uint64_t first =
                std::strtoull(head.c_str() + range + 6, &dash, 10);
            const std::uint64_t last =
                std::strtoull(dash + 1, nullptr, 10);
            byFirst[first] = body.substr(end + 4, last - first + 1);
        }
        start = body.find(delimiter, end);
    }
    for (const auto& [first, last] : ranges) {
        const auto part = byFirst.find(first);
        if (part == byFirst.end() ||
            part->second.size() != last - first + 1) {
            throw std::runtime_error(url + " sent incomplete ranges");
        }
        parts.push_back(std::move(part->second));
    }
#else
    static_cast<void>(url);
    static_cast<void>(ranges);
#endif
    return parts;
}

std::filesystem::path DataFile(std::string_view name) {
    std::vector<std::filesystem::path> directories;
    if (const char* data = std::getenv("ERNIE_DEMO_DATA")) {
        directories.emplace_back(data);
    }
#ifdef ERNIE_DEMO_DATA_DIR
    directories.emplace_back(ERNIE_DEMO_DATA_DIR);
#endif
    for (const std::filesystem::path& directory : directories) {
        const std::filesystem::path path = directory / name;
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    throw std::runtime_error(
        "The demos' data file " + std::string(name) +
        " is in neither $ERNIE_DEMO_DATA nor the source tree");
}

}  // namespace ernie::demos
