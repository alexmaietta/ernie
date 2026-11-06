// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "ernie/device/tensor.h"

namespace ernie::app {

enum class ImageFormat { Png, Jpeg, Exr };

enum class VideoFormat { None, Mp4, ProRes };

struct CaptureSettings {
    std::filesystem::path folder;
    std::string pattern = "{scene}_{date}_{time}";
    ImageFormat format = ImageFormat::Png;
    int frames = 120;
    double fps = 30;
    VideoFormat video = VideoFormat::Mp4;
    bool keepFrames = false;
};

std::string ExpandPattern(std::string_view pattern, std::string_view scene, std::string_view camera,
                          int64_t frame);

std::filesystem::path FreePath(const std::filesystem::path& path);

std::filesystem::path FindFfmpeg();

class Capture {
  public:
    Capture();
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    void Screenshot();
    void Record();
    void Stop();
    void Update(const device::Tensor* shown, uint64_t frame, std::string_view scene,
                std::string_view camera);

    bool Recording() const {
        return m_recording;
    }

    int Recorded() const {
        return m_recorded;
    }

    bool Encoding() const {
        return m_encoding;
    }

    const std::vector<std::filesystem::path>& Recent() const {
        return m_recent;
    }

    CaptureSettings settings;

  private:
    struct Job {
        device::Download read;
        device::DType type;
        std::vector<int64_t> shape;
        std::filesystem::path path;
    };

    void Write(const Job& job);
    void Encode();

    bool m_screenshot = false;
    bool m_recording = false;
    bool m_finishing = false;
    int m_recorded = 0;
    uint64_t m_lastFrame = 0;
    std::filesystem::path m_sequence;
    std::vector<Job> m_jobs;
    std::vector<std::filesystem::path> m_recent;
    std::thread m_encoder;
    std::atomic<bool> m_encoding = false;
};

}
