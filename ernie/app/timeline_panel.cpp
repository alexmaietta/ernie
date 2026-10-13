// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/icons.h"
#include "ernie/app/menu.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace ernie::app {

namespace {

ImU32 Colour(const glm::vec4& c) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(c.r, c.g, c.b, c.a));
}

/// The smallest of 1, 2 and 5 times a power of ten time codes whose
/// ticks are at least `pixels` apart at `perCode` pixels a time code.
double TickStep(double perCode, double pixels) {
    const double least = pixels / std::max(perCode, 1e-9);
    const double decade =
        std::pow(10.0, std::floor(std::log10(least)));
    for (const double factor : {1.0, 2.0, 5.0, 10.0}) {
        if (factor * decade >= least) {
            return factor * decade;
        }
    }
    return 10.0 * decade;
}

class TimelinePanel final : public Panel {
  public:
    TimelinePanel() : Panel("Timeline", Dock::Timeline) {}

    void Draw(Viewer& viewer) override {
        Scene& scene = viewer.GetScene();
        Timeline& timeline = viewer.GetTimeline();
        const auto [start, end] = Timeline::Range(scene);
        const float unit = ImGui::GetFontSize();
        const float field = 4.0f * unit;
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        // An icon button: the widest icon and the frame's padding.
        float icon = 0.0f;
        for (const char* glyph :
             {icons::kSkipBack, icons::kStepBack, icons::kPlay,
              icons::kSquare, icons::kRepeat, icons::kSettings}) {
            icon = std::max(icon, ImGui::CalcTextSize(glyph).x);
        }
        const float button =
            icon + 2.0f * ImGui::GetStyle().FramePadding.x;

        // The range's start, the ruler, its end and the current time
        // code, then the transport, as Kit lays its timeline out.
        TimeField(viewer, "##start", tokens::kStartTimeCode, start,
                  field, "Start time code");
        ImGui::SameLine();
        const float right =
            2.0f * field + 8.0f * button + 10.0f * spacing;
        const float ruler = std::max(
            ImGui::GetContentRegionAvail().x - right, 4.0f * unit);
        Ruler(viewer, timeline, start, end, ruler);
        ImGui::SameLine();
        TimeField(viewer, "##end", tokens::kEndTimeCode, end, field,
                  "End time code");
        ImGui::SameLine();
        double now = Timeline::TimeCode(scene);
        ImGui::SetNextItemWidth(field);
        if (ImGui::InputDouble("##now", &now, 0.0, 0.0, "%g")) {
            timeline.Seek(scene, now);
        }
        Tooltip("Current time code");
        ImGui::SameLine();
        Transport(viewer, timeline);
    }

  private:
    static void Tooltip(const char* text) {
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", text);
        }
    }

    /// A field of one of the stage's time codes, edited through the
    /// viewer's undo history.
    static void TimeField(Viewer& viewer, const char* id,
                          std::string_view attribute, double value,
                          float width, const char* tooltip) {
        ImGui::SetNextItemWidth(width);
        ImGui::InputDouble(id, &value, 0.0, 0.0, "%g");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            viewer.Edit(Path::Root(), attribute, value);
        }
        Tooltip(tooltip);
    }

    /// The ruler across the range: ticks labelled at least five
    /// characters apart, the current time code's line, and a click or
    /// drag that pauses and scrubs, to whole frames.
    void Ruler(Viewer& viewer, Timeline& timeline, double start,
               double end, float width) {
        Scene& scene = viewer.GetScene();
        const Theme& theme = viewer.GetTheme();
        const float height = ImGui::GetFrameHeight();
        const ImVec2 low = ImGui::GetCursorScreenPos();
        const ImVec2 high(low.x + width, low.y + height);
        ImGui::InvisibleButton("##ruler", ImVec2(width, height));
        const double perCode =
            static_cast<double>(width) / (end - start);
        const double frame = Timeline::FrameTimeCodes(scene);
        if (ImGui::IsItemActive()) {
            timeline.Pause();
            const double at =
                start + static_cast<double>(
                            ImGui::GetIO().MousePos.x - low.x) /
                            perCode;
            const double snapped =
                start + std::round((at - start) / frame) * frame;
            timeline.Seek(scene, std::clamp(snapped, start, end));
        }
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(low, high, Colour(theme.field),
                            ImGui::GetStyle().FrameRounding);
        const float textWidth = ImGui::CalcTextSize("00000").x;
        const double major = TickStep(perCode, textWidth + 8.0f);
        const double minor = major / 5.0;
        const bool minors = minor * perCode >= 5.0;
        const ImU32 tick = Colour(theme.textMuted);
        for (double t = std::ceil(start / minor) * minor; t <= end;
             t += minor) {
            const float x =
                low.x + static_cast<float>((t - start) * perCode);
            const bool labelled =
                std::fabs(t / major - std::round(t / major)) < 1e-6;
            if (!labelled && !minors) {
                continue;
            }
            draw->AddLine(
                ImVec2(x,
                       high.y - (labelled ? 0.45f : 0.2f) * height),
                ImVec2(x, high.y), tick);
            if (labelled) {
                char label[32];
                std::snprintf(label, sizeof(label), "%g", t);
                draw->AddText(ImVec2(x + 2.0f, low.y), tick, label);
            }
        }
        const double now = Timeline::TimeCode(scene);
        if (now >= start && now <= end) {
            const float x =
                low.x + static_cast<float>((now - start) * perCode);
            const ImU32 accent = Colour(theme.accent);
            draw->AddLine(ImVec2(x, low.y), ImVec2(x, high.y), accent,
                          2.0f);
            const float half = 0.2f * height;
            draw->AddTriangleFilled(ImVec2(x - half, low.y),
                                    ImVec2(x + half, low.y),
                                    ImVec2(x, low.y + half), accent);
        }
        if (ImGui::IsItemHovered() && !ImGui::IsItemActive()) {
            ImGui::SetTooltip("Click or drag to scrub");
        }
    }

    /// First frame, a frame back, play or pause, stop, a frame on,
    /// last frame, loop, and the settings menu.
    static void Transport(Viewer& viewer, Timeline& timeline) {
        Scene& scene = viewer.GetScene();
        const auto [start, end] = Timeline::Range(scene);
        const auto icon = [](const char* glyph, const char* id,
                             const char* tooltip) {
            const bool clicked =
                ImGui::Button((std::string(glyph) + id).c_str());
            Tooltip(tooltip);
            ImGui::SameLine();
            return clicked;
        };
        if (icon(icons::kSkipBack, "##first", "First frame")) {
            timeline.Pause();
            timeline.Seek(scene, start);
        }
        if (icon(icons::kStepBack, "##back", "Previous frame")) {
            timeline.StepFrames(scene, -1);
        }
        if (timeline.IsPlaying()) {
            if (icon(icons::kPause, "##play", "Pause (Space)")) {
                timeline.Pause();
            }
        } else if (icon(icons::kPlay, "##play", "Play (Space)")) {
            timeline.Play();
        }
        if (icon(icons::kSquare, "##stop",
                 "Stop: back to the start")) {
            timeline.Stop(scene);
        }
        if (icon(icons::kStepForward, "##forward", "Next frame")) {
            timeline.StepFrames(scene, 1);
        }
        if (icon(icons::kSkipForward, "##last", "Last frame")) {
            timeline.Pause();
            timeline.Seek(scene, end);
        }
        // Loop lit while on.
        const bool looping = timeline.IsLooping();
        if (looping) {
            ImGui::PushStyleColor(
                ImGuiCol_Button,
                ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }
        const bool loop = ImGui::Button(
            (std::string(icons::kRepeat) + "##loop").c_str());
        if (looping) {
            ImGui::PopStyleColor();
        }
        Tooltip(looping ? "Looping: on" : "Looping: off");
        if (loop) {
            timeline.SetLooping(!looping);
        }
        ImGui::SameLine();
        if (ImGui::Button(
                (std::string(icons::kSettings) + "##settings")
                    .c_str())) {
            ImGui::OpenPopup("##timelineSettings");
        }
        Tooltip("Timeline settings");
        menu::NextMenuWide();
        if (ImGui::BeginPopup("##timelineSettings")) {
            Settings(viewer, timeline);
            ImGui::EndPopup();
        }
    }

    /// The frame rate (time codes and frames a second alike, so a
    /// time code is a frame) and playing every frame.
    static void Settings(Viewer& viewer, Timeline& timeline) {
        const Stage stage(viewer.GetScene());
        menu::Section("Frame Rate");
        const double rate = stage.GetFramesPerSecond();
        for (const double fps :
             {24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 120.0}) {
            char label[32];
            std::snprintf(label, sizeof(label), "%g fps", fps);
            if (menu::Option(label, rate == fps)) {
                viewer.Edit(Path::Root(), tokens::kTimeCodesPerSecond,
                            fps);
                viewer.Edit(Path::Root(), tokens::kFramesPerSecond,
                            fps);
            }
        }
        menu::Section("Playback");
        bool everyFrame = timeline.PlaysEveryFrame();
        if (menu::Check("Play Every Frame", &everyFrame)) {
            timeline.SetPlayEveryFrame(everyFrame);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("A frame each step however long it "
                              "takes, rather than real time");
        }
        bool looping = timeline.IsLooping();
        if (menu::Check("Loop", &looping)) {
            timeline.SetLooping(looping);
        }
    }
};

}  // namespace

std::unique_ptr<Panel> MakeTimelinePanel() {
    return std::make_unique<TimelinePanel>();
}

}  // namespace ernie::app
