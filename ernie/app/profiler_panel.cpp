// SPDX-License-Identifier: BSD-3-Clause
/// @file profiler_panel.cpp
/// @brief The Profiler panel, after Omniverse Kit's Profiler window: CPU and GPU zones averaged
/// into trees, a bar per frame, and exports as text, JSON and a Chrome trace.
#include "ernie/app/message_log.h"
#include "ernie/app/panels.h"
#include "ernie/util/file.h"
#include "ernie/util/profiler.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace ernie::app {
namespace {

using profiler::Frame;
using profiler::ReportFilter;
using profiler::ReportNode;
using profiler::ReportRow;

double Ms(std::uint64_t nanoseconds) {
    return static_cast<double>(nanoseconds) * 1e-6;
}

class ProfilerPanel final : public Panel {
  public:
    ProfilerPanel(MessageLog& log, bool alwaysRecord)
        : Panel("Profiler", Dock::Bottom, false), log_(log), alwaysRecord_(alwaysRecord) {}

    void Draw() override {
        profiler::SetCpuEnabled(cpu_ || alwaysRecord_);
        profiler::SetGpuEnabled(gpu_ || alwaysRecord_);
        if (!paused_) {
            // Not the newest, whose GPU zones may still be arriving.
            frames_ = profiler::Frames();
            frames_.resize(frames_.size() - std::min(frames_.size(), profiler::kFramesInFlight));
        }
        DrawControls();
        if (ImGui::BeginTabBar("##profiler")) {
            if (ImGui::BeginTabItem("Zones")) {
                DrawZones();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Frames")) {
                DrawFrameBars();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }

    /// Recording costs time, so it stops while nobody looks.
    void Undrawn() override {
        if (!alwaysRecord_) {
            profiler::SetEnabled(false);
        }
    }

  private:
    void DrawControls() {
        const float number = 5.0f * ImGui::GetFontSize();
        ImGui::Checkbox("CPU", &cpu_);
        ImGui::SetItemTooltip("Record the CPU's zones (ERNIE_ZONE).");
        ImGui::SameLine();
        ImGui::Checkbox("GPU", &gpu_);
        ImGui::SetItemTooltip("Record the GPU's zones (ERNIE_GPU_ZONE).");
        ImGui::SameLine();
        ImGui::Checkbox("Pause", &paused_);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            profiler::Clear();
            frames_.clear();
            selected_.reset();
        }
        if (!frames_.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%zu frames, %.1f fps", frames_.size(),
                                1000.0 / std::max(MeanMs(), 1e-3));
        }

        ImGui::Checkbox("Sort by time", &byTime_);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputDouble("Min ms", &minMs_, 0.0, 0.0, "%.3f");
        minMs_ = std::max(minMs_, 0.0);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputInt("Depth", &depth_, 0, 0);
        depth_ = std::clamp(depth_, 0, 32);
        ImGui::SameLine();
        ImGui::Checkbox("Average over (s)", &average_);
        ImGui::SetItemTooltip("Average the zones over the frames of the last seconds; else show "
                              "the newest frame, or the one clicked in Frames.");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::BeginDisabled(!average_);
        ImGui::InputDouble("##averageSeconds", &averageSeconds_, 0.0, 0.0, "%.2f");
        averageSeconds_ = std::clamp(averageSeconds_, 0.05, 10.0);
        ImGui::EndDisabled();

        if (ImGui::Button("Copy Text")) {
            ImGui::SetClipboardText(profiler::ReportText(report_, Filter()).c_str());
        }
        ImGui::SameLine();
        if (ImGui::Button("Save JSON")) {
            Save("ernie_profile.json", profiler::ReportJson(report_));
        }
        ImGui::SameLine();
        if (ImGui::Button("Save Trace")) {
            Save("ernie_trace.json", profiler::TraceJson(frames_, profiler::ThreadNames()));
        }
        ImGui::SetItemTooltip("The frames kept, as a Chrome trace for Perfetto or "
                              "chrome://tracing.");
    }

    void Save(const std::filesystem::path& file, const std::string& text) {
        const std::filesystem::path path = std::filesystem::absolute(file);
        try {
            WriteTextFile(path, text);
            log_.Info("Saved " + path.string());
        } catch (const std::exception& error) {
            log_.Error(error.what());
        }
    }

    ReportFilter Filter() const {
        return {static_cast<std::uint32_t>(depth_), minMs_, byTime_};
    }

    double MeanMs() const {
        double total = 0.0;
        for (const Frame& frame : frames_) {
            total += Ms(frame.end - frame.start);
        }
        return frames_.empty() ? 0.0 : total / static_cast<double>(frames_.size());
    }

    /// @return The frame inspected: the one clicked, else the newest with GPU zones, as the
    /// device runs a frame's commands a frame or two after the host records them.
    const Frame* Inspected() {
        if (selected_) {
            const auto found = std::ranges::find(frames_, *selected_, &Frame::index);
            if (found != frames_.end()) {
                return &*found;
            }
            selected_.reset();
        }
        const auto withGpu = std::ranges::find_if(frames_.rbegin(), frames_.rend(),
                                                  [](const Frame& f) { return !f.gpu.empty(); });
        if (withGpu != frames_.rend()) {
            return &*withGpu;
        }
        return frames_.empty() ? nullptr : &frames_.back();
    }

    /// @return The frames the Zones tab averages: those of the last `averageSeconds_`, else the
    /// one inspected.
    std::vector<Frame> Averaged() {
        if (!average_) {
            const Frame* frame = Inspected();
            return frame != nullptr ? std::vector<Frame>{*frame} : std::vector<Frame>{};
        }
        std::vector<Frame> frames;
        if (frames_.empty()) {
            return frames;
        }
        const auto window = static_cast<std::uint64_t>(averageSeconds_ * 1e9);
        for (const Frame& frame : frames_) {
            if (frame.end + window >= frames_.back().end) {
                frames.push_back(frame);
            }
        }
        return frames;
    }

    void DrawZones() {
        report_ = profiler::Summarize(Averaged(), profiler::ThreadNames());
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
        if (!ImGui::BeginTable("##zones", 4, kFlags)) {
            return;
        }
        const float em = ImGui::GetFontSize();
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Duration (ms)", ImGuiTableColumnFlags_WidthFixed, 7.0f * em);
        ImGui::TableSetupColumn("Calls", ImGuiTableColumnFlags_WidthFixed, 3.5f * em);
        ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthFixed, 8.0f * em);
        ImGui::TableHeadersRow();
        DrawSection("GPU", report_.gpu, report_.gpuMs, report_.gpuFrames, Filter());
        // The CPU's zones sit under a node per thread.
        ReportFilter cpu = Filter();
        ++cpu.depth;
        DrawSection("CPU", report_.cpu, report_.cpuMs, report_.cpuFrames, cpu);
        ImGui::EndTable();
    }

    /// One section of the Zones table: its total heading its nested zones.
    void DrawSection(const char* title, const std::vector<ReportNode>& nodes, double totalMs,
                     std::size_t frames, const ReportFilter& filter) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextLink));
        const bool shown = ImGui::TreeNodeEx(title, ImGuiTreeNodeFlags_SpanAllColumns |
                                                        ImGuiTreeNodeFlags_DefaultOpen);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", totalMs);
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        if (frames == 0) {
            ImGui::TextUnformatted("(no frames)");
        } else {
            ImGui::Text("%zu frame%s", frames, frames == 1 ? "" : "s");
        }
        ImGui::PopStyleColor();
        if (!shown) {
            return;
        }
        DrawRows(profiler::Rows(nodes, filter));
        ImGui::TreePop();
    }

    static void DrawRows(const std::vector<ReportRow>& rows) {
        constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t open = 0;        // tree levels pushed
        std::uint32_t closed = kNone;  // the depth of a closed parent
        for (const ReportRow& row : rows) {
            if (row.depth > closed) {
                continue;
            }
            closed = kNone;
            for (; open > row.depth; --open) {
                ImGui::TreePop();
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_DefaultOpen;
            if (!row.parent) {
                flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            }
            const bool expanded = ImGui::TreeNodeEx(row.node->name.c_str(), flags);
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
            std::array<char, 16> share{};
            std::snprintf(share.data(), share.size(), "%.2f%%", row.node->percent);
            ImGui::ProgressBar(static_cast<float>(row.node->percent / 100.0),
                               ImVec2(-1.0f, ImGui::GetTextLineHeight()), share.data());
        }
        for (; open > 0; --open) {
            ImGui::TreePop();
        }
    }

    /// A bar per frame, scaled to the frames' 95th percentile so one slow frame does not flatten
    /// the rest; clicking one inspects it.
    void DrawFrameBars() {
        const float height = 4.5f * ImGui::GetFontSize();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float width = std::max(ImGui::GetContentRegionAvail().x, 1.0f);
        ImGui::InvisibleButton("##frames", ImVec2(width, height));
        const bool hovered = ImGui::IsItemHovered();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height),
                            ImGui::GetColorU32(ImGuiCol_FrameBg));
        if (frames_.empty()) {
            ImGui::TextDisabled("No frames yet.");
            return;
        }
        std::vector<double> times;
        for (const Frame& frame : frames_) {
            times.push_back(Ms(frame.end - frame.start));
        }
        std::ranges::sort(times);
        const double longest = std::max(1000.0 / 30.0 * 1.1, 1.5 * times[times.size() * 95 / 100]);
        const auto heightOf = [&](double ms) {
            return static_cast<float>(std::min(ms / longest, 1.0)) * height;
        };
        // Guides at 60 and 30 frames per second.
        for (const double guide : {1000.0 / 60.0, 1000.0 / 30.0}) {
            const float y = origin.y + height - heightOf(guide);
            draw->AddLine(ImVec2(origin.x, y), ImVec2(origin.x + width, y),
                          ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.5f));
        }
        const std::size_t slots = std::max(profiler::History(), frames_.size());
        const float bar = width / static_cast<float>(slots);
        const float first = origin.x + width - bar * static_cast<float>(frames_.size());
        const Frame* inspected = Inspected();
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        for (std::size_t i = 0; i < frames_.size(); ++i) {
            const Frame& frame = frames_[i];
            const double ms = Ms(frame.end - frame.start);
            const float x = first + bar * static_cast<float>(i);
            const bool under = hovered && mouse.x >= x && mouse.x < x + bar;
            const ImGuiCol colour = inspected != nullptr && frame.index == inspected->index
                                        ? ImGuiCol_PlotHistogramHovered
                                        : (under ? ImGuiCol_ButtonHovered : ImGuiCol_PlotHistogram);
            draw->AddRectFilled(ImVec2(x, origin.y + height - heightOf(ms)),
                                ImVec2(x + std::max(bar - 1.0f, 1.0f), origin.y + height),
                                ImGui::GetColorU32(colour));
            if (under) {
                ImGui::SetTooltip("Frame %llu: %.2f ms",
                                  static_cast<unsigned long long>(frame.index), ms);
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    selected_ = frame.index;
                    paused_ = true;
                }
            }
        }
        if (inspected != nullptr) {
            ImGui::Text("Frame %llu: %.2f ms, %zu CPU zones, %zu GPU zones",
                        static_cast<unsigned long long>(inspected->index),
                        Ms(inspected->end - inspected->start), inspected->cpu.size(),
                        inspected->gpu.size());
        }
    }

    MessageLog& log_;
    bool alwaysRecord_;
    bool cpu_ = true;
    bool gpu_ = true;
    bool paused_ = false;
    bool byTime_ = false;
    double minMs_ = 0.0;
    int depth_ = 8;
    bool average_ = true;
    double averageSeconds_ = 1.0;
    std::vector<Frame> frames_;
    std::optional<std::uint64_t> selected_;
    profiler::Report report_;
};

}  // namespace

std::unique_ptr<Panel> MakeProfilerPanel(MessageLog& log, bool alwaysRecord) {
    return std::make_unique<ProfilerPanel>(log, alwaysRecord);
}

}  // namespace ernie::app
