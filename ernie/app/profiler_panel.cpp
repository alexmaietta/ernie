// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/builtin_panels.h"
#include "ernie/app/viewer.h"
#include "ernie/device/memory_stats.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/util/profile_report.h"
#include "ernie/util/profiler.h"

#include <imgui.h>

#if defined(__linux__)
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

namespace {

using profiler::Frame;
using profiler::Report;
using profiler::ReportFilter;
using profiler::ReportNode;
using profiler::ReportRow;
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

/// Writes `text` to `path` and says so in the viewer's log.
void SaveText(Viewer& viewer, const std::filesystem::path& path,
              const std::string& text, const char* what) {
    std::ofstream file(path, std::ios::binary);
    file << text;
    if (file) {
        viewer.Log(LogLevel::Info, std::string("Saved ") + what +
                                       " to " + path.string());
    } else {
        viewer.Log(LogLevel::Error, std::string("Could not save ") +
                                        what + " to " +
                                        path.string());
    }
}

/// The process's resident memory, where the platform tells.
std::optional<std::uint64_t> HostResidentBytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    std::uint64_t total = 0;
    std::uint64_t resident = 0;
    if (statm >> total >> resident) {
        return resident *
               static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    }
#endif
    return std::nullopt;
}

/// Ernie's profiler (ernie/util/profiler.h) after Omniverse Kit's
/// Profiler window. Above, CPU and GPU recording, a capture of the
/// next seconds to a Chrome trace (Perfetto, chrome://tracing),
/// filters (sort by time, a least time, how deep GPU and CPU zones
/// nest) and averaging, and exports of the zones as text, JSON or to
/// the log. Below, tabs: Zones, the GPU's and the CPU's zones nested
/// with their mean times and shares as Kit lists them; Timeline, a
/// bar per frame (click one to inspect it) over a timeline of its
/// zones (zoom with the wheel, pan by dragging); and Memory, the
/// device's heaps and Ernie's buffers and images by kind. The viewer
/// turns recording off whenever the panel is not shown.
class ProfilerPanel final : public Panel {
  public:
    ProfilerPanel() : Panel("Profiler", Dock::Bottom) {
        SetVisible(false);
    }

    void Draw(Viewer& viewer) override {
        const bool capturing = capture_.has_value();
        profiler::SetCpuEnabled(cpu_ || capturing);
        profiler::SetGpuEnabled(gpu_ || capturing);
        if (!paused_) {
            // Not the newest, whose GPU zones arrive in part while
            // the device runs them.
            frames_ = profiler::Frames();
            frames_.resize(
                frames_.size() -
                std::min(frames_.size(), profiler::kFramesInFlight));
        }
        if (capturing) {
            Capture(viewer);
        }
        DrawControls(viewer);
        if (ImGui::BeginTabBar("##profiler")) {
            if (ImGui::BeginTabItem("Zones")) {
                DrawZones();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Timeline")) {
                DrawFrames();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Memory")) {
                DrawMemory(viewer);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }

  private:
    /// A capture in progress: the frames that start before `end`.
    struct Capturing {
        std::uint64_t start = 0;
        std::uint64_t end = 0;
        std::map<std::uint64_t, Frame> frames;
    };

    void DrawControls(Viewer& viewer) {
        const float number = 70.0f * ImGui::GetFontSize() / 14.0f;
        ImGui::Checkbox("CPU profiler", &cpu_);
        ImGui::SetItemTooltip("Record the CPU's zones (ERNIE_ZONE).");
        ImGui::SameLine();
        ImGui::Checkbox("GPU profiler", &gpu_);
        ImGui::SetItemTooltip(
            "Record the GPU's zones (ERNIE_GPU_ZONE) with timestamp "
            "queries.");
        ImGui::SameLine();
        ImGui::Checkbox("Pause updates", &paused_);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            profiler::Clear();
            frames_.clear();
            selected_.reset();
        }
        ImGui::SameLine();
        if (!frames_.empty()) {
            ImGui::TextDisabled("%zu frames, %.1f fps",
                                frames_.size(),
                                1000.0 / std::max(Mean(), 1e-3));
        }

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Capture time (s)");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputFloat("##captureTime", &captureSeconds_, 0.0f,
                          0.0f, "%.1f");
        captureSeconds_ = std::clamp(captureSeconds_, 0.1f, 60.0f);
        ImGui::SameLine();
        ImGui::BeginDisabled(capture_.has_value());
        const bool key = !ImGui::GetIO().WantTextInput &&
                         ImGui::IsKeyPressed(ImGuiKey_F5, false);
        if (ImGui::Button("Capture (F5)") ||
            (key && !capture_.has_value())) {
            const std::uint64_t now = profiler::Now();
            capture_ = Capturing{now,
                                 now + static_cast<std::uint64_t>(
                                           captureSeconds_ * 1e9),
                                 {}};
        }
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Record CPU and GPU zones for the "
                              "capture time, then save "
                              "them as a Chrome trace for Perfetto "
                              "or chrome://tracing.");
        if (capture_) {
            ImGui::SameLine();
            ImGui::TextDisabled("Capturing... %.1f s",
                                static_cast<double>(profiler::Now() -
                                                    capture_->start) *
                                    1e-9);
        }

        ImGui::Checkbox("Sort by time", &filter_.byTime);
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Min time (ms)");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputDouble("##minTime", &filter_.minMs, 0.0, 0.0,
                           "%.3f");
        filter_.minMs = std::max(filter_.minMs, 0.0);
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("GPU depth");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputInt("##gpuDepth", &gpuDepth_, 0, 0);
        gpuDepth_ = std::clamp(gpuDepth_, 0, 32);
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("CPU depth");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputInt("##cpuDepth", &cpuDepth_, 0, 0);
        cpuDepth_ = std::clamp(cpuDepth_, 0, 32);
        ImGui::SameLine();
        ImGui::Checkbox("Average over (s)", &average_);
        ImGui::SetItemTooltip(
            "Average the zones over the frames of the last seconds; "
            "else show the newest frame, or the one clicked in the "
            "Timeline.");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::BeginDisabled(!average_);
        ImGui::InputFloat("##averageTime", &averageSeconds_, 0.0f,
                          0.0f, "%.2f");
        averageSeconds_ = std::clamp(averageSeconds_, 0.05f, 10.0f);
        ImGui::EndDisabled();

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Export:");
        ImGui::SameLine();
        if (ImGui::Button("Copy Text")) {
            ImGui::SetClipboardText(
                profiler::ReportText(report_, Filter()).c_str());
        }
        ImGui::SameLine();
        // The zones shown when the button is pressed.
        if (ImGui::Button("Save Text")) {
            viewer.OpenFileDialog(
                {"Save Profile",
                 true,
                 {{"Text", {".txt"}}},
                 "ernie_profile.txt"},
                [text = profiler::ReportText(report_, Filter())](
                    Viewer& v, const std::filesystem::path& path) {
                    SaveText(v, path, text, "the profile");
                });
        }
        ImGui::SameLine();
        if (ImGui::Button("Save JSON")) {
            viewer.OpenFileDialog(
                {"Save Profile",
                 true,
                 {{"JSON", {".json"}}},
                 "ernie_profile.json"},
                [text = profiler::ReportJson(report_)](
                    Viewer& v, const std::filesystem::path& path) {
                    SaveText(v, path, text, "the profile");
                });
        }
        ImGui::SameLine();
        if (ImGui::Button("Save to Log")) {
            viewer.Log(LogLevel::Info,
                       "Profile:\n" +
                           profiler::ReportText(report_, Filter()));
        }
    }

    /// Gathers the frames of the capture in progress, and once its
    /// time is up and the GPU's zones have had a moment to arrive,
    /// asks where to save it.
    void Capture(Viewer& viewer) {
        for (const Frame& frame : profiler::Frames()) {
            if (frame.start >= capture_->start &&
                frame.start < capture_->end) {
                capture_->frames[frame.index] = frame;
            }
        }
        if (profiler::Now() < capture_->end + 200'000'000) {
            return;
        }
        std::vector<Frame> frames;
        for (auto& [index, frame] : capture_->frames) {
            frames.push_back(std::move(frame));
        }
        capture_.reset();
        viewer.OpenFileDialog(
            {"Save Trace",
             true,
             {{"Chrome trace", {".json"}}},
             "ernie_trace.json"},
            [text = profiler::TraceJson(frames,
                                        profiler::ThreadNames())](
                Viewer& v, const std::filesystem::path& path) {
                SaveText(v, path, text, "the trace");
            });
    }

    /// The exports' filter: the deeper of the two depths (the CPU's
    /// zones sit under a node per thread).
    ReportFilter Filter() const {
        return {static_cast<std::uint32_t>(
                    std::max(gpuDepth_, cpuDepth_ + 1)),
                filter_.minMs, filter_.byTime};
    }

    /// The mean frame time of the frames kept, in milliseconds.
    double Mean() const {
        double total = 0.0;
        for (const Frame& frame : frames_) {
            total += Ms(frame.end - frame.start);
        }
        return frames_.empty()
                   ? 0.0
                   : total / static_cast<double>(frames_.size());
    }

    /// The frames the Zones tab averages: those of the last
    /// `averageSeconds_`, else the one inspected.
    std::vector<Frame> Averaged() {
        std::vector<Frame> frames;
        if (!average_) {
            if (const Frame* frame = Selected()) {
                frames.push_back(*frame);
            }
            return frames;
        }
        if (frames_.empty()) {
            return frames;
        }
        const std::uint64_t window =
            static_cast<std::uint64_t>(averageSeconds_ * 1e9);
        const std::uint64_t last = frames_.back().end;
        for (const Frame& frame : frames_) {
            if (frame.end + window >= last) {
                frames.push_back(frame);
            }
        }
        return frames;
    }

    void DrawZones() {
        report_ =
            profiler::Summarize(Averaged(), profiler::ThreadNames());
        constexpr ImGuiTableFlags kFlags =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
        if (!ImGui::BeginTable("##zones", 4, kFlags)) {
            return;
        }
        const float scale = ImGui::GetFontSize() / 14.0f;
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Duration (ms)",
                                ImGuiTableColumnFlags_WidthFixed,
                                95.0f * scale);
        ImGui::TableSetupColumn(
            "Calls", ImGuiTableColumnFlags_WidthFixed, 50.0f * scale);
        ImGui::TableSetupColumn("Percentage",
                                ImGuiTableColumnFlags_WidthFixed,
                                110.0f * scale);
        ImGui::TableHeadersRow();
        DrawSection("GPU", report_.gpu, report_.gpuMs,
                    report_.gpuFrames,
                    static_cast<std::uint32_t>(gpuDepth_));
        DrawSection("CPU", report_.cpu, report_.cpuMs,
                    report_.cpuFrames,
                    static_cast<std::uint32_t>(cpuDepth_) + 1);
        ImGui::EndTable();
    }

    /// One section of the Zones table: a row of its total heading its
    /// zones, nested (open and close them), siblings as they ran or
    /// longest first.
    void DrawSection(const char* title,
                     const std::vector<ReportNode>& nodes,
                     double totalMs, std::size_t frames,
                     std::uint32_t depth) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(
                                                 ImGuiCol_TextLink));
        const bool shown =
            ImGui::TreeNodeEx(title,
                              ImGuiTreeNodeFlags_SpanAllColumns |
                                  ImGuiTreeNodeFlags_DefaultOpen,
                              "%s", title);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", totalMs);
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        if (frames == 0) {
            ImGui::TextUnformatted(title[0] == 'G' && !gpu_
                                       ? "(not recording)"
                                       : "(no frames)");
        } else {
            ImGui::Text("%zu frame%s", frames,
                        frames == 1 ? "" : "s");
        }
        ImGui::PopStyleColor();
        if (!shown) {
            return;
        }
        const ReportFilter filter{depth, filter_.minMs,
                                  filter_.byTime};
        std::uint32_t open = 0;  // tree levels pushed
        std::uint32_t closed =
            UINT32_MAX;  // depth of a closed parent
        for (const ReportRow& row : profiler::Rows(nodes, filter)) {
            if (row.depth > closed) {
                continue;
            }
            closed = UINT32_MAX;
            for (; open > row.depth; --open) {
                ImGui::TreePop();
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_SpanAllColumns |
                ImGuiTreeNodeFlags_DefaultOpen;
            if (!row.parent) {
                flags |= ImGuiTreeNodeFlags_Leaf |
                         ImGuiTreeNodeFlags_NoTreePushOnOpen;
            }
            const bool expanded =
                ImGui::TreeNodeEx(row.node->name.c_str(), flags, "%s",
                                  row.node->name.c_str());
            if (row.parent) {
                if (expanded) {
                    ++open;
                } else {
                    closed = row.depth;
                }
            }
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", row.node->ms);
            ImGui::TableNextColumn();
            ImGui::Text("%.3g", row.node->calls);
            ImGui::TableNextColumn();
            char share[16];
            std::snprintf(share, sizeof(share), "%.2f%%",
                          row.node->percent);
            ImGui::ProgressBar(
                static_cast<float>(row.node->percent / 100.0),
                ImVec2(-1.0f, ImGui::GetTextLineHeight()), share);
        }
        for (; open > 0; --open) {
            ImGui::TreePop();
        }
        ImGui::TreePop();  // the section's
    }

    /// The Timeline tab: the frame graph, and the inspected frame's
    /// zones on a timeline.
    void DrawFrames() {
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
        DrawTimeline(*frame, ImGui::GetContentRegionAvail().y);
    }
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

    std::string Size(std::uint64_t bytes) const {
        char text[32];
        if (megabytes_) {
            std::snprintf(text, sizeof(text), "%.2f",
                          static_cast<double>(bytes) /
                              (1024.0 * 1024.0));
        } else {
            std::snprintf(text, sizeof(text), "%llu",
                          static_cast<unsigned long long>(bytes));
        }
        return text;
    }

    /// The Memory tab, as Kit's memory profiler: the host's resident
    /// memory, the device's heaps (what Ernie allocated, what the
    /// process uses, the budget), and Ernie's buffers and images by
    /// kind.
    void DrawMemory(Viewer& viewer) {
        ImGui::Checkbox("By category", &byCategory_);
        ImGui::SameLine();
        ImGui::Checkbox("Size in MB", &megabytes_);
        const char* unit = megabytes_ ? "MB" : "bytes";
        if (const std::optional<std::uint64_t> host =
                HostResidentBytes()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Host memory: %s %s resident",
                                Size(*host).c_str(), unit);
        }

        const VmaAllocator allocator = viewer.Context().Allocator();
        const VkPhysicalDeviceMemoryProperties* properties = nullptr;
        vmaGetMemoryProperties(allocator, &properties);
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(allocator, budgets.data());
        constexpr ImGuiTableFlags kHeapFlags =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
            ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("##heaps", 6, kHeapFlags)) {
            char header[32];
            ImGui::TableSetupColumn("Heap");
            for (const char* name :
                 {"Ernie's", "Process", "Budget", "Size"}) {
                std::snprintf(header, sizeof(header), "%s (%s)", name,
                              unit);
                ImGui::TableSetupColumn(header);
            }
            ImGui::TableSetupColumn("Used of budget");
            ImGui::TableHeadersRow();
            for (std::uint32_t heap = 0;
                 heap < properties->memoryHeapCount; ++heap) {
                const VmaBudget& budget = budgets[heap];
                const bool device =
                    (properties->memoryHeaps[heap].flags &
                     VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%u: %s", heap,
                            device ? "Device local" : "Host");
                for (const std::uint64_t bytes :
                     {budget.statistics.allocationBytes, budget.usage,
                      budget.budget,
                      static_cast<std::uint64_t>(
                          properties->memoryHeaps[heap].size)}) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(Size(bytes).c_str());
                }
                ImGui::TableNextColumn();
                const double used =
                    budget.budget > 0
                        ? static_cast<double>(budget.usage) /
                              static_cast<double>(budget.budget)
                        : 0.0;
                char share[16];
                std::snprintf(share, sizeof(share), "%.1f%%",
                              100.0 * used);
                ImGui::ProgressBar(
                    static_cast<float>(used),
                    ImVec2(-1.0f, ImGui::GetTextLineHeight()), share);
            }
            ImGui::EndTable();
        }

        std::vector<memory::Allocation> live = memory::Live();
        std::sort(live.begin(), live.end(),
                  [](const memory::Allocation& a,
                     const memory::Allocation& b) {
                      return a.bytes > b.bytes;
                  });
        struct Category {
            std::uint64_t bytes = 0;
            std::vector<const memory::Allocation*> allocations;
        };
        std::map<std::string, Category> categories;
        std::uint64_t total = 0;
        for (const memory::Allocation& allocation : live) {
            Category& category = categories[allocation.category];
            category.bytes += allocation.bytes;
            category.allocations.push_back(&allocation);
            total += allocation.bytes;
        }
        std::vector<std::pair<std::string, const Category*>> sorted;
        for (const auto& [name, category] : categories) {
            sorted.emplace_back(name, &category);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                      return a.second->bytes > b.second->bytes;
                  });

        constexpr ImGuiTableFlags kFlags =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
        if (!ImGui::BeginTable("##allocations", 3, kFlags)) {
            return;
        }
        const float scale = ImGui::GetFontSize() / 14.0f;
        char sizeHeader[32];
        std::snprintf(sizeHeader, sizeof(sizeHeader), "Size (%s)",
                      unit);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(
            "Count", ImGuiTableColumnFlags_WidthFixed, 60.0f * scale);
        ImGui::TableSetupColumn(sizeHeader,
                                ImGuiTableColumnFlags_WidthFixed,
                                110.0f * scale);
        ImGui::TableHeadersRow();
        const auto allocationRow = [&](const memory::Allocation& a,
                                       bool named) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TreeNodeEx(&a,
                              ImGuiTreeNodeFlags_Leaf |
                                  ImGuiTreeNodeFlags_NoTreePushOnOpen,
                              "%s%s%s", named ? a.category : "",
                              named ? ": " : "", a.detail.c_str());
            ImGui::TableNextColumn();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Size(a.bytes).c_str());
        };
        if (byCategory_) {
            for (const auto& [name, category] : sorted) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                const bool open = ImGui::TreeNodeEx(
                    name.c_str(), ImGuiTreeNodeFlags_SpanAllColumns);
                ImGui::TableNextColumn();
                ImGui::Text("%zu", category->allocations.size());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(Size(category->bytes).c_str());
                if (open) {
                    for (const memory::Allocation* allocation :
                         category->allocations) {
                        allocationRow(*allocation, false);
                    }
                    ImGui::TreePop();
                }
            }
        } else {
            for (const memory::Allocation& allocation : live) {
                allocationRow(allocation, true);
            }
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(
                                                 ImGuiCol_TextLink));
        ImGui::TextUnformatted("Total of Ernie's buffers and images");
        ImGui::TableNextColumn();
        ImGui::Text("%zu", live.size());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(Size(total).c_str());
        ImGui::PopStyleColor();
        ImGui::EndTable();
    }

    static constexpr float kLabelWidth = 110.0f;
    bool cpu_ = true;
    bool gpu_ = true;
    bool paused_ = false;
    float captureSeconds_ = 3.0f;
    std::optional<Capturing> capture_;
    ReportFilter filter_;  ///< Its least time and sorting.
    int gpuDepth_ = 8;
    int cpuDepth_ = 8;
    bool average_ = true;
    float averageSeconds_ = 1.0f;
    Report report_;  ///< What the Zones tab shows.
    bool byCategory_ = true;
    bool megabytes_ = true;
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
