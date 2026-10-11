// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/viewer.h"
#include "ernie/util/profiler.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

namespace {

using profiler::Frame;
using profiler::Zone;

double Ms(std::uint64_t ns) {
    return static_cast<double>(ns) * 1e-6;
}

/// A muted colour for a zone, the same for the same name, as Kit's
/// profiler colours its zones.
ImU32 ZoneColour(const char* name) {
    const std::size_t hash = std::hash<std::string_view>()(name);
    const float hue = static_cast<float>(hash % 360) / 360.0f;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    ImGui::ColorConvertHSVtoRGB(hue, 0.45f, 0.62f, r, g, b);
    return ImGui::GetColorU32(ImVec4(r, g, b, 1.0f));
}

/// Ernie's profiler (ernie/util/profiler.h) after Omniverse Kit's
/// profiler window: recording on and off, a bar per frame of its time
/// (click one to inspect it), a timeline of the frame's CPU zones per
/// thread and GPU zones (zoom with the wheel, pan by dragging), and
/// the zones' totals. Hidden, it stops recording; the viewer turns
/// recording off whenever the panel is not shown.
class ProfilerPanel final : public Panel {
  public:
    ProfilerPanel() : Panel("Profiler", Dock::Bottom) {
        SetVisible(false);
    }

    void Draw(Viewer& viewer) override {
        static_cast<void>(viewer);
        ImGui::Checkbox("Record", &record_);
        profiler::SetEnabled(record_);
        ImGui::SameLine();
        ImGui::Checkbox("Pause", &paused_);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            profiler::Clear();
            frames_.clear();
            selected_.reset();
        }
        if (!paused_) {
            frames_ = profiler::Frames();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%zu frames%s", frames_.size(),
                            record_ ? "" : "  (not recording zones)");
        const Frame* frame = Selected();
        DrawFrameGraph();
        if (frame == nullptr) {
            ImGui::TextDisabled("No frames yet.");
            return;
        }
        ImGui::Text(
            "Frame %llu: %.2f ms CPU, %zu CPU zones, %zu GPU zones",
            static_cast<unsigned long long>(frame->index),
            Ms(frame->end - frame->start), frame->cpu.size(),
            frame->gpu.size());
        if (!paused_) {
            ImGui::SameLine();
            ImGui::TextDisabled(
                "(pause, or click a frame, to zoom and pan)");
        }
        DrawTimeline(*frame,
                     0.55f * ImGui::GetContentRegionAvail().y);
        DrawTotals(*frame);
    }

  private:
    /// The frame inspected: the one clicked, else the newest whose
    /// GPU zones have arrived (the device runs a frame's commands a
    /// frame or two after the host records them).
    const Frame* Selected() {
        if (frames_.empty()) {
            return nullptr;
        }
        if (selected_) {
            for (const Frame& frame : frames_) {
                if (frame.index == *selected_) {
                    return &frame;
                }
            }
            selected_.reset();
        }
        for (auto frame = frames_.rbegin(); frame != frames_.rend();
             ++frame) {
            if (!frame->gpu.empty()) {
                return &*frame;
            }
        }
        return &frames_.back();
    }

    void DrawFrameGraph() {
        const float height = 64.0f;
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        ImGui::InvisibleButton("##frames",
                               ImVec2(std::max(width, 1.0f), height));
        const bool hovered = ImGui::IsItemHovered();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(
            origin, ImVec2(origin.x + width, origin.y + height),
            ImGui::GetColorU32(ImGuiCol_FrameBg));
        if (frames_.empty()) {
            return;
        }
        // Scaled to the frames' 95th percentile, so one slow frame
        // (the first, compiling shaders) does not flatten the rest;
        // longer bars clip at the top.
        std::vector<double> times;
        for (const Frame& frame : frames_) {
            times.push_back(Ms(frame.end - frame.start));
        }
        std::sort(times.begin(), times.end());
        const double longest =
            std::max(1000.0 / 30.0 * 1.1,
                     1.5 * times[std::min(times.size() - 1,
                                          times.size() * 95 / 100)]);
        // Guides at 60 and 30 frames per second.
        for (const double guide : {1000.0 / 60.0, 1000.0 / 30.0}) {
            if (guide <= longest) {
                const float y =
                    origin.y + height -
                    static_cast<float>(guide / longest) * height;
                draw->AddLine(
                    ImVec2(origin.x, y), ImVec2(origin.x + width, y),
                    ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.5f));
            }
        }
        const std::size_t count = std::max<std::size_t>(
            profiler::History(), frames_.size());
        const float bar = width / static_cast<float>(count);
        const float first = origin.x + width -
                            bar * static_cast<float>(frames_.size());
        const Frame* inspected = Selected();
        for (std::size_t i = 0; i < frames_.size(); ++i) {
            const Frame& frame = frames_[i];
            const double ms = Ms(frame.end - frame.start);
            const float x = first + bar * static_cast<float>(i);
            const float top =
                origin.y + height -
                static_cast<float>(std::min(ms / longest, 1.0)) *
                    height;
            const bool chosen = inspected != nullptr &&
                                frame.index == inspected->index;
            const bool under = hovered &&
                               ImGui::GetIO().MousePos.x >= x &&
                               ImGui::GetIO().MousePos.x < x + bar;
            const ImU32 colour =
                chosen ? ImGui::GetColorU32(
                             ImGuiCol_PlotHistogramHovered)
                       : ImGui::GetColorU32(
                             under ? ImGuiCol_ButtonHovered
                                   : ImGuiCol_PlotHistogram);
            draw->AddRectFilled(ImVec2(x, top),
                                ImVec2(x + std::max(bar - 1.0f, 1.0f),
                                       origin.y + height),
                                colour);
            if (under) {
                ImGui::SetTooltip(
                    "Frame %llu: %.2f ms",
                    static_cast<unsigned long long>(frame.index), ms);
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    selected_ = frame.index;
                    paused_ = true;  // keep it while inspecting
                    view_.reset();
                }
            }
        }
    }

    /// One track of zones laid out by depth from `top`; returns its
    /// height.
    float DrawTrack(ImDrawList* draw, const char* label,
                    const std::vector<const Zone*>& zones,
                    std::uint64_t base, float left, float right,
                    float top) {
        const float row = ImGui::GetTextLineHeight() + 4.0f;
        std::uint32_t depth = 0;
        for (const Zone* zone : zones) {
            depth = std::max(depth, zone->depth);
        }
        const float height =
            row * static_cast<float>(depth + 1) + 6.0f;
        draw->AddText(ImVec2(left - kLabelWidth + 6.0f, top + 3.0f),
                      ImGui::GetColorU32(ImGuiCol_TextDisabled),
                      label);
        draw->AddLine(ImVec2(left - kLabelWidth, top + height),
                      ImVec2(right, top + height),
                      ImGui::GetColorU32(ImGuiCol_Separator));
        const double span = view_->second - view_->first;
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        for (const Zone* zone : zones) {
            const double start =
                static_cast<double>(zone->start - base);
            const double end = static_cast<double>(zone->end - base);
            float x0 = left + static_cast<float>(
                                  (start - view_->first) / span) *
                                  (right - left);
            float x1 = left + static_cast<float>(
                                  (end - view_->first) / span) *
                                  (right - left);
            if (x1 < left || x0 > right) {
                continue;
            }
            x0 = std::max(x0, left);
            x1 = std::min(std::max(x1, x0 + 1.0f), right);
            const float y0 =
                top + 3.0f + row * static_cast<float>(zone->depth);
            const float y1 = y0 + row - 2.0f;
            draw->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1),
                                ZoneColour(zone->name), 2.0f);
            const float textWidth = ImGui::CalcTextSize(zone->name).x;
            if (x1 - x0 > textWidth + 6.0f) {
                draw->AddText(ImVec2(x0 + 3.0f, y0 + 1.0f),
                              IM_COL32(240, 240, 240, 255),
                              zone->name);
            }
            if (ImGui::IsWindowHovered() && mouse.x >= x0 &&
                mouse.x < x1 && mouse.y >= y0 && mouse.y < y1) {
                ImGui::SetTooltip("%s\n%.3f ms", zone->name,
                                  Ms(zone->end - zone->start));
            }
        }
        return height;
    }

    void DrawTimeline(const Frame& frame, float height) {
        if (!ImGui::BeginChild("##timeline",
                               ImVec2(0.0f, std::max(height, 80.0f)),
                               ImGuiChildFlags_Borders)) {
            ImGui::EndChild();
            return;
        }
        // The CPU tracks span the frame; the GPU's, its own clock
        // from its first zone.
        const double frameSpan =
            static_cast<double>(frame.end - frame.start);
        double gpuSpan = 0.0;
        std::uint64_t gpuBase = UINT64_MAX;
        for (const Zone& zone : frame.gpu) {
            gpuBase = std::min(gpuBase, zone.start);
        }
        for (const Zone& zone : frame.gpu) {
            gpuSpan = std::max(
                gpuSpan, static_cast<double>(zone.end - gpuBase));
        }
        // Live, the view fits each frame; paused, it keeps the zoom
        // and pan.
        if (!view_ || !paused_) {
            view_ = std::pair(
                0.0, std::max(std::max(frameSpan, gpuSpan), 1.0));
        }
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float left = origin.x + kLabelWidth;
        const float right =
            origin.x + ImGui::GetContentRegionAvail().x;
        ImDrawList* draw = ImGui::GetWindowDrawList();

        // Zoom about the mouse with the wheel, pan by dragging.
        const ImGuiIO& io = ImGui::GetIO();
        if (ImGui::IsWindowHovered() && right > left) {
            const double span = view_->second - view_->first;
            const double at =
                view_->first +
                std::clamp((io.MousePos.x - left) / (right - left),
                           0.0f, 1.0f) *
                    span;
            if (io.MouseWheel != 0.0f) {
                const double scale = std::pow(
                    0.85, static_cast<double>(io.MouseWheel));
                view_ = std::pair(at - (at - view_->first) * scale,
                                  at + (view_->second - at) * scale);
            }
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                const double shift =
                    -io.MouseDelta.x / (right - left) * span;
                view_ = std::pair(view_->first + shift,
                                  view_->second + shift);
            }
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                view_.reset();
                view_ = std::pair(
                    0.0, std::max(std::max(frameSpan, gpuSpan), 1.0));
            }
        }
        // A time ruler.
        const double span = view_->second - view_->first;
        const double step =
            std::pow(10.0, std::floor(std::log10(span / 5.0)));
        const float rulerHeight = ImGui::GetTextLineHeight() + 4.0f;
        for (double t = std::ceil(view_->first / step) * step;
             t <= view_->second; t += step) {
            const float x =
                left + static_cast<float>((t - view_->first) / span) *
                           (right - left);
            draw->AddLine(ImVec2(x, origin.y),
                          ImVec2(x, origin.y + 4.0f),
                          ImGui::GetColorU32(ImGuiCol_TextDisabled));
            char text[32];
            std::snprintf(text, sizeof(text), "%.3g ms", t * 1e-6);
            draw->AddText(ImVec2(x + 2.0f, origin.y),
                          ImGui::GetColorU32(ImGuiCol_TextDisabled),
                          text);
        }
        float top = origin.y + rulerHeight;
        // CPU tracks by thread.
        std::map<std::uint32_t, std::vector<const Zone*>> threads;
        for (const Zone& zone : frame.cpu) {
            threads[zone.thread].push_back(&zone);
        }
        const std::vector<std::string> names =
            profiler::ThreadNames();
        for (const auto& [thread, zones] : threads) {
            const std::string label =
                "CPU " + (thread < names.size()
                              ? names[thread]
                              : std::to_string(thread));
            top += DrawTrack(draw, label.c_str(), zones, frame.start,
                             left, right, top);
        }
        if (!frame.gpu.empty()) {
            std::vector<const Zone*> zones;
            for (const Zone& zone : frame.gpu) {
                zones.push_back(&zone);
            }
            top += DrawTrack(draw, "GPU", zones, gpuBase, left, right,
                             top);
        }
        ImGui::Dummy(ImVec2(right - origin.x, top - origin.y));
        ImGui::EndChild();
    }

    void DrawTotals(const Frame& frame) {
        struct Total {
            std::string name;
            const char* where;
            int calls = 0;
            double total = 0.0;
            double longest = 0.0;
        };
        std::map<std::pair<std::string, int>, Total> totals;
        const auto add = [&](const Zone& zone, int gpu) {
            Total& total = totals[{zone.name, gpu}];
            total.name = zone.name;
            total.where = gpu != 0 ? "GPU" : "CPU";
            total.calls += 1;
            const double ms = Ms(zone.end - zone.start);
            total.total += ms;
            total.longest = std::max(total.longest, ms);
        };
        for (const Zone& zone : frame.cpu) {
            add(zone, 0);
        }
        for (const Zone& zone : frame.gpu) {
            add(zone, 1);
        }
        std::vector<Total> rows;
        for (auto& [key, total] : totals) {
            rows.push_back(std::move(total));
        }
        constexpr ImGuiTableFlags kFlags =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable |
            ImGuiTableFlags_Resizable;
        if (!ImGui::BeginTable("##totals", 5, kFlags)) {
            return;
        }
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Zone",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(
            "Where", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn(
            "Calls", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn(
            "Total ms",
            ImGuiTableColumnFlags_WidthFixed |
                ImGuiTableColumnFlags_DefaultSort |
                ImGuiTableColumnFlags_PreferSortDescending,
            80.0f);
        ImGui::TableSetupColumn(
            "Max ms", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();
        if (const ImGuiTableSortSpecs* sort =
                ImGui::TableGetSortSpecs();
            sort != nullptr && sort->SpecsCount > 0) {
            const ImGuiTableColumnSortSpecs& spec = sort->Specs[0];
            const bool ascending =
                spec.SortDirection == ImGuiSortDirection_Ascending;
            std::sort(
                rows.begin(), rows.end(),
                [&](const Total& a, const Total& b) {
                    const auto key = [&](const Total& t) -> double {
                        switch (spec.ColumnIndex) {
                        case 2:
                            return t.calls;
                        case 4:
                            return t.longest;
                        default:
                            return t.total;
                        }
                    };
                    if (spec.ColumnIndex == 0 ||
                        spec.ColumnIndex == 1) {
                        const std::string ka =
                            spec.ColumnIndex == 0 ? a.name : a.where;
                        const std::string kb =
                            spec.ColumnIndex == 0 ? b.name : b.where;
                        return ascending ? ka < kb : ka > kb;
                    }
                    return ascending ? key(a) < key(b)
                                     : key(a) > key(b);
                });
        }
        for (const Total& row : rows) {
            ImGui::PushID(&row);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::ColorButton("##colour",
                               ImGui::ColorConvertU32ToFloat4(
                                   ZoneColour(row.name.c_str())),
                               ImGuiColorEditFlags_NoTooltip,
                               ImVec2(ImGui::GetTextLineHeight(),
                                      ImGui::GetTextLineHeight()));
            ImGui::SameLine();
            ImGui::TextUnformatted(row.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", row.where);
            ImGui::TableNextColumn();
            ImGui::Text("%d", row.calls);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", row.total);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", row.longest);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    static constexpr float kLabelWidth = 110.0f;
    bool record_ = true;
    bool paused_ = false;
    std::vector<Frame> frames_;
    std::optional<std::uint64_t> selected_;
    std::optional<std::pair<double, double>>
        view_;  ///< Nanoseconds from the frame's start.
};

}  // namespace

std::unique_ptr<Panel> MakeProfilerPanel() {
    return std::make_unique<ProfilerPanel>();
}

}  // namespace ernie::app
