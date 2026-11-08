// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "ernie/app/viewer/panels/panels.h"

namespace ernie::app {
namespace {

constexpr size_t kRecentShown = 12;

std::string Labelled(const char* icon, const char* text) {
    return std::string(icon) + "  " + text;
}

void DrawSequence(Capture& capture) {
    CaptureSettings& settings = capture.settings;
    ImGui::SeparatorText("Sequence");
    const float field = 5.0f * ImGui::GetFontSize();
    ImGui::BeginDisabled(capture.Recording());
    ImGui::SetNextItemWidth(field);
    ImGui::InputInt("Frames", &settings.frames, 0);
    settings.frames = std::max(settings.frames, 1);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(field);
    ImGui::InputDouble("fps", &settings.fps, 0, 0, "%g");
    settings.fps = std::max(settings.fps, 1.0);
    ImGui::SetItemTooltip("The video's frame rate; recording takes each frame the viewport "
                          "renders");
    static const bool ffmpeg = !FindFfmpeg().empty();
    const bool encodable = ffmpeg && settings.format != ImageFormat::Exr;
    const char* videos[] = {"Frames only", "MP4 (H.264)", "ProRes 422 HQ (.mov)"};
    int video = encodable ? int(settings.video) : 0;
    ImGui::BeginDisabled(!encodable);
    ImGui::SetNextItemWidth(12.0f * ImGui::GetFontSize());
    if (ImGui::Combo("Video", &video, videos, int(std::size(videos)))) {
        settings.video = VideoFormat(video);
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(video == 0);
    ImGui::Checkbox("Keep frames", &settings.keepFrames);
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (!ffmpeg) {
        ImGui::TextDisabled("ffmpeg isn't on the PATH, so sequences stay frames");
    } else if (settings.format == ImageFormat::Exr) {
        ImGui::TextDisabled("EXR sequences stay frames");
    }
    ImGui::EndDisabled();
    if (capture.Recording()) {
        if (ImGui::Button(Labelled(icons::kSquare, "Stop").c_str())) {
            capture.Stop();
        }
        ImGui::SameLine();
        const std::string label =
            std::to_string(capture.Recorded()) + " / " + std::to_string(settings.frames);
        ImGui::ProgressBar(float(capture.Recorded()) / float(settings.frames), {-1.0f, 0.0f},
                           label.c_str());
        return;
    }
    ImGui::BeginDisabled(capture.Encoding());
    if (ImGui::Button(Labelled(icons::kVideo, "Record").c_str())) {
        capture.Record();
    }
    ImGui::EndDisabled();
    if (capture.Encoding()) {
        ImGui::SameLine();
        ImGui::TextDisabled("Encoding the last video...");
    }
}

void DrawCapture(Capture& capture) {
    CaptureSettings& settings = capture.settings;
    const float fields =
        -ImGui::CalcTextSize("Name pattern").x - ImGui::GetStyle().ItemInnerSpacing.x;
    std::string folder = settings.folder.string();
    ImGui::SetNextItemWidth(fields);
    if (ImGui::InputText("Folder", &folder)) {
        settings.folder = folder;
    }
    ImGui::SetNextItemWidth(fields);
    ImGui::InputText("Name pattern", &settings.pattern);
    ImGui::SetItemTooltip("{scene}, {camera}, {date}, {time} and {frame} fill in; existing "
                          "files are never overwritten");
    const char* formats[] = {"PNG", "JPEG", "EXR"};
    int format = int(settings.format);
    ImGui::SetNextItemWidth(8.0f * ImGui::GetFontSize());
    if (ImGui::Combo("Format", &format, formats, int(std::size(formats)))) {
        settings.format = ImageFormat(format);
    }
    ImGui::SetItemTooltip("EXR keeps float outputs' values");
    if (ImGui::Button(Labelled(icons::kDownload, "Screenshot").c_str())) {
        capture.Screenshot();
    }
    ImGui::SetItemTooltip("Save the active viewport's image (F10)");
    DrawSequence(capture);
    if (!capture.Recent().empty()) {
        ImGui::SeparatorText("Recent");
        for (size_t at = 0; at < capture.Recent().size() && at < kRecentShown; ++at) {
            const std::string path = capture.Recent()[at].string();
            if (ImGui::Selectable(path.c_str())) {
                ImGui::SetClipboardText(path.c_str());
            }
            ImGui::SetItemTooltip("Copy the path");
        }
    }
}

}

Panel CreateCapturePanel(Capture& capture) {
    return {"Capture", Dock::Bottom, [&capture] { DrawCapture(capture); }};
}

}
