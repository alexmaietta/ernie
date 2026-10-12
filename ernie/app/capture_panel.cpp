// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/capture.h"
#include "ernie/app/icons.h"
#include "ernie/app/menu.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/stage_schemas.h"

#include <imgui_internal.h>

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <string>

namespace ernie::app {

namespace {

class CapturePanel final : public Panel {
  public:
    CapturePanel() : Panel("Capture", Dock::Bottom) {}

    void Draw(Viewer& viewer) override {
        CaptureOptions& options = viewer.GetCaptureOptions();
        if (!ImGui::BeginTable("##capture", 2,
                               ImGuiTableFlags_SizingFixedFit)) {
            return;
        }
        ImGui::TableSetupColumn("name",
                                ImGuiTableColumnFlags_WidthFixed,
                                8.0f * ImGui::GetFontSize());
        ImGui::TableSetupColumn("value",
                                ImGuiTableColumnFlags_WidthStretch);
        Output(options);
        Image(options);
        Screenshot(viewer);
        Sequence(viewer);
        ImGui::EndTable();
        Recent(viewer);
    }

  private:
    static void Row(const char* label) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
    }

    void Output(CaptureOptions& options) {
        Row("Folder");
        ImGui::SetNextItemWidth(-1.0f);
        if (folder_.empty() && !options.folder.empty()) {
            folder_ = options.folder.string();
        }
        if (InputString("##folder", folder_)) {
            options.folder = folder_;
        }
        Row("Name");
        ImGui::SetNextItemWidth(-1.0f);
        InputString("##name", options.name);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "{scene}, {camera}, {date} and {time} fill in; a "
                "file never overwrites another");
        }
    }

    static void Screenshot(Viewer& viewer) {
        Row("Screenshot");
        if (ImGui::Button(
                (std::string(icons::kDownload) + "  Capture (F10)")
                    .c_str())) {
            viewer.Screenshot();
        }
    }

    /// Which render vars a capture writes, and at what size.
    static void Image(CaptureOptions& options) {
        Row("Render vars");
        std::vector<std::string>& vars = options.vars;
        std::string shown;
        for (const std::string& var : vars) {
            shown += (shown.empty() ? "" : ", ") + var;
        }
        ImGui::SetNextItemWidth(-1.0f);
        menu::NextMenuWide();
        if (ImGui::BeginCombo("##vars", shown.empty()
                                            ? "The viewport as shown"
                                            : shown.c_str())) {
            // Toggles stay open, so several change at once.
            ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups,
                                false);
            for (const std::string_view name : kRenderVarNames) {
                const auto found =
                    std::find(vars.begin(), vars.end(), name);
                bool on = found != vars.end();
                if (menu::Check(std::string(name).c_str(), &on)) {
                    if (on) {
                        vars.emplace_back(name);
                    } else {
                        vars.erase(found);
                    }
                }
            }
            ImGui::PopItemFlag();
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "LdrColor as PNG, the others as EXR (_<var>), "
                "rendered anew through the active camera; none "
                "writes the viewport's image as shown");
        }
        Row("Scale");
        const std::pair<double, const char*> scales[] = {
            {0.5, "50%"},
            {1.0, "100%"},
            {2.0, "200%"},
            {4.0, "400%"}};
        for (const auto& [scale, label] : scales) {
            if (ImGui::RadioButton(label, options.scale == scale)) {
                options.scale = scale;
            }
            ImGui::SameLine();
        }
        ImGui::TextDisabled("of the viewport's render resolution");
    }

    void Sequence(Viewer& viewer) {
        const Scene& scene = viewer.GetScene();
        const auto [start, end] = Timeline::Range(scene);
        if (!rangeSet_) {
            start_ = start;
            end_ = end;
        }
        Row("Frames");
        const float field = 5.0f * ImGui::GetFontSize();
        ImGui::SetNextItemWidth(field);
        if (ImGui::InputDouble("##from", &start_, 0.0, 0.0, "%g")) {
            rangeSet_ = true;
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("to");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(field);
        if (ImGui::InputDouble("##to", &end_, 0.0, 0.0, "%g")) {
            rangeSet_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Timeline Range")) {
            rangeSet_ = false;
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("every");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(field);
        ImGui::InputInt("##every", &every_);
        every_ = std::max(every_, 1);
        ImGui::SameLine();
        ImGui::TextDisabled(
            "at %g fps", Stage(scene).GetFramesPerSecond() / every_);

        Row("Video");
        const bool ffmpeg = !FindFfmpegCached().empty();
        const char* formats[] = {"Frames only", "MP4 (H.264)",
                                 "ProRes 422 HQ (.mov)"};
        ImGui::BeginDisabled(!ffmpeg);
        ImGui::SetNextItemWidth(14.0f * ImGui::GetFontSize());
        ImGui::Combo("##video", &video_, formats, 3);
        ImGui::SameLine();
        ImGui::BeginDisabled(video_ == 0);
        ImGui::Checkbox("Keep frames", &keepFrames_);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (!ffmpeg) {
            ImGui::SameLine();
            ImGui::TextDisabled(
                "ffmpeg is not on the PATH: frames only");
        }

        Row("");
        if (viewer.Recording()) {
            const auto [done, total] = viewer.RecordingProgress();
            if (ImGui::Button((std::string(icons::kSquare) + "  Stop")
                                  .c_str())) {
                viewer.StopRecording();
            }
            ImGui::SameLine();
            char label[64];
            std::snprintf(label, sizeof(label), "%lld / %lld",
                          static_cast<long long>(done),
                          static_cast<long long>(total));
            ImGui::ProgressBar(
                total > 0 ? static_cast<float>(done) / total : 0.0f,
                ImVec2(-1.0f, 0.0f), label);
            return;
        }
        ImGui::BeginDisabled(viewer.Encoding());
        if (ImGui::Button(
                (std::string(icons::kVideo) + "  Record").c_str())) {
            SequenceOptions sequence;
            sequence.start = start_;
            sequence.end = end_;
            sequence.every = every_;
            sequence.video = ffmpeg ? static_cast<VideoFormat>(video_)
                                    : VideoFormat::None;
            sequence.keepFrames = keepFrames_;
            try {
                viewer.RecordSequence(sequence);
            } catch (const std::exception& error) {
                viewer.Log(LogLevel::Error, error.what());
            }
        }
        ImGui::EndDisabled();
        if (viewer.Encoding()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Encoding the last video...");
        }
    }

    static void Recent(Viewer& viewer) {
        const auto& recent = viewer.RecentCaptures();
        if (recent.empty()) {
            return;
        }
        ImGui::SeparatorText("Recent");
        // Latest first; a click copies the path.
        for (auto it = recent.rbegin(); it != recent.rend(); ++it) {
            const std::string path = it->string();
            if (ImGui::Selectable(path.c_str())) {
                ImGui::SetClipboardText(path.c_str());
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Click to copy the path");
            }
        }
    }

    /// ffmpeg's path, looked up once.
    static const std::filesystem::path& FindFfmpegCached() {
        static const std::filesystem::path ffmpeg = FindFfmpeg();
        return ffmpeg;
    }

    std::string folder_;
    double start_ = 0.0;
    double end_ = 0.0;
    bool rangeSet_ = false;  ///< Chosen apart from the timeline's.
    int every_ = 1;
    int video_ = 1;
    bool keepFrames_ = true;
};

}  // namespace

std::unique_ptr<Panel> MakeCapturePanel() {
    return std::make_unique<CapturePanel>();
}

}  // namespace ernie::app
