// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>

#include "ernie/app/viewer/panels/panels.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie::app {
namespace {

using profiler::Frame;
using profiler::ReportNode;

constexpr size_t kHistory = 300;
constexpr size_t kFramesInFlight = 2;
constexpr ImGuiTableFlags kTable =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;

double FrameMs(const Frame& frame) {
    return double(frame.endNs - frame.startNs) * 1e-6;
}

ImU32 NameColor(std::string_view name) {
    uint32_t hash = 2166136261u;
    for (const char c : name) {
        hash = (hash ^ uint8_t(c)) * 16777619u;
    }
    float r = 0;
    float g = 0;
    float b = 0;
    ImGui::ColorConvertHSVtoRGB(float(hash % 360u) / 360.0f, 0.45f, 0.85f, r, g, b);
    return ImGui::GetColorU32(ImVec4(r, g, b, 1.0f));
}

void Save(std::string_view name, const std::string& text) {
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    const std::filesystem::path path =
        std::filesystem::absolute(std::format("ernie_{}_{:%Y%m%d_%H%M%S}.json", name, now));
    std::ofstream out(path, std::ios::binary);
    if (out.write(text.data(), std::streamsize(text.size()))) {
        Log(LogLevel::Info, "saved {}", path.string());
    } else {
        Log(LogLevel::Error, "couldn't write {}", path.string());
    }
}

class Profiler {
  public:
    Profiler() {
        profiler::SetHistory(kHistory);
    }

    void Draw() {
        if (!m_open) {
            m_open = true;
            if (!profiler::Enabled()) {
                profiler::SetEnabled(true);
                m_ownsRecording = true;
            }
        }
        if (!m_paused) {
            m_frames = profiler::Frames();
            m_frames.resize(m_frames.size() - std::min(m_frames.size(), kFramesInFlight));
            m_tracks = profiler::Tracks();
        }
        DrawControls();
        if (ImGui::BeginTabBar("##profiler")) {
            if (ImGui::BeginTabItem("Zones")) {
                DrawZones();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Frames")) {
                DrawFrameBars();
                if (const Frame* frame = Inspected()) {
                    DrawTracks(*frame);
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }

    void Hidden() {
        if (m_open && m_ownsRecording) {
            profiler::SetEnabled(false);
        }
        m_open = false;
        m_ownsRecording = false;
    }

  private:
    void DrawControls() {
        const float number = 5.0f * ImGui::GetFontSize();
        bool recording = profiler::Enabled();
        if (ImGui::Checkbox("Record", &recording)) {
            profiler::SetEnabled(recording);
            m_ownsRecording = recording;
        }
        ImGui::SetItemTooltip("Record the CPU's zones and the GPU's timestamps");
        ImGui::SameLine();
        ImGui::Checkbox("Pause", &m_paused);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            profiler::Clear();
            m_frames.clear();
            m_selected.reset();
        }
        if (!m_frames.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%zu frames, %.1f fps", m_frames.size(),
                                1000.0 / std::max(MeanMs(), 1e-3));
        }
        ImGui::Checkbox("Sort by time", &m_byTime);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputDouble("Min ms", &m_minMs, 0, 0, "%.3f");
        m_minMs = std::max(m_minMs, 0.0);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::InputInt("Depth", &m_depth, 0, 0);
        m_depth = std::clamp(m_depth, 0, 32);
        ImGui::SameLine();
        ImGui::Checkbox("Average over (s)", &m_average);
        ImGui::SetItemTooltip("Average the zones over the last seconds' frames; else show the "
                              "newest frame, or the one clicked in Frames");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(number);
        ImGui::BeginDisabled(!m_average);
        ImGui::InputDouble("##averageSeconds", &m_averageSeconds, 0, 0, "%.2f");
        m_averageSeconds = std::clamp(m_averageSeconds, 0.05, 10.0);
        ImGui::EndDisabled();
        const profiler::ReportFilter filter{uint32_t(m_depth), m_minMs, m_byTime};
        if (ImGui::Button("Copy Text")) {
            ImGui::SetClipboardText(profiler::ReportText(m_report, filter).c_str());
        }
        ImGui::SameLine();
        if (ImGui::Button("Save Timings")) {
            Save("timings", profiler::ReportJson(m_report));
        }
        ImGui::SetItemTooltip("The zone timings shown, as JSON");
        ImGui::SameLine();
        if (ImGui::Button("Save Trace")) {
            Save("trace", profiler::TraceJson(m_frames));
        }
        ImGui::SetItemTooltip("The frames kept, as a Chrome trace for Perfetto or "
                              "chrome://tracing");
    }

    double MeanMs() const {
        double total = 0;
        for (const Frame& frame : m_frames) {
            total += FrameMs(frame);
        }
        return m_frames.empty() ? 0 : total / double(m_frames.size());
    }

    bool HasGpuZones(const Frame& frame) const {
        return std::ranges::any_of(frame.zones, [&](const profiler::Zone& zone) {
            return zone.track < m_tracks.size() && m_tracks[zone.track].gpu;
        });
    }

    const Frame* Inspected() {
        if (m_selected) {
            const auto found = std::ranges::find(m_frames, *m_selected, &Frame::index);
            if (found != m_frames.end()) {
                return &*found;
            }
            m_selected.reset();
        }
        for (auto frame = m_frames.rbegin(); frame != m_frames.rend(); ++frame) {
            if (HasGpuZones(*frame)) {
                return &*frame;
            }
        }
        return m_frames.empty() ? nullptr : &m_frames.back();
    }

    std::vector<Frame> Averaged() {
        if (!m_average) {
            const Frame* frame = Inspected();
            return frame != nullptr ? std::vector<Frame>{*frame} : std::vector<Frame>{};
        }
        std::vector<Frame> frames;
        const auto window = uint64_t(m_averageSeconds * 1e9);
        for (const Frame& frame : m_frames) {
            if (frame.endNs + window >= m_frames.back().endNs) {
                frames.push_back(frame);
            }
        }
        return frames;
    }

    void DrawZones() {
        m_report = profiler::Summarize(Averaged());
        constexpr ImGuiTableFlags kFlags = kTable | ImGuiTableFlags_ScrollY;
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
        for (const profiler::ReportTrack& track : m_report.tracks) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextLink));
            const bool open =
                ImGui::TreeNodeEx(track.name.c_str(), ImGuiTreeNodeFlags_SpanAllColumns |
                                                          ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", track.ms);
            ImGui::TableNextColumn();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(track.gpu ? "GPU" : "CPU");
            ImGui::PopStyleColor();
            if (open) {
                DrawNodes(track.zones, 0);
                ImGui::TreePop();
            }
        }
        ImGui::EndTable();
    }

    void DrawNodes(const std::vector<ReportNode>& nodes, int depth) {
        if (depth >= m_depth) {
            return;
        }
        std::vector<const ReportNode*> shown;
        for (const ReportNode& node : nodes) {
            if (node.ms >= m_minMs) {
                shown.push_back(&node);
            }
        }
        if (m_byTime) {
            std::ranges::sort(
                shown, [](const ReportNode* a, const ReportNode* b) { return a->ms > b->ms; });
        }
        for (const ReportNode* node : shown) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool parent = !node->children.empty() && depth + 1 < m_depth;
            ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_DefaultOpen;
            if (!parent) {
                flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            }
            const bool open = ImGui::TreeNodeEx(node->name.c_str(), flags);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", node->ms);
            ImGui::TableNextColumn();
            ImGui::Text("%.3g", node->calls);
            ImGui::TableNextColumn();
            char share[16];
            std::snprintf(share, sizeof(share), "%.2f%%", node->percent);
            ImGui::ProgressBar(float(node->percent / 100), {-1, ImGui::GetTextLineHeight()}, share);
            if (parent && open) {
                DrawNodes(node->children, depth + 1);
                ImGui::TreePop();
            }
        }
    }

    void DrawFrameBars() {
        const float height = 4.5f * ImGui::GetFontSize();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float width = std::max(ImGui::GetContentRegionAvail().x, 1.0f);
        ImGui::InvisibleButton("##frames", {width, height});
        const bool hovered = ImGui::IsItemHovered();
        ImDrawList& draw = *ImGui::GetWindowDrawList();
        draw.AddRectFilled(origin, {origin.x + width, origin.y + height},
                           ImGui::GetColorU32(ImGuiCol_FrameBg));
        if (m_frames.empty()) {
            ImGui::TextDisabled(profiler::Enabled() ? "No frames yet" : "Not recording");
            return;
        }
        std::vector<double> times;
        for (const Frame& frame : m_frames) {
            times.push_back(FrameMs(frame));
        }
        std::ranges::sort(times);
        const double longest = std::max(1000.0 / 30.0 * 1.1, 1.5 * times[times.size() * 95 / 100]);
        const auto barHeight = [&](double ms) {
            return float(std::min(ms / longest, 1.0)) * height;
        };
        for (const double guide : {1000.0 / 60.0, 1000.0 / 30.0}) {
            const float y = origin.y + height - barHeight(guide);
            draw.AddLine({origin.x, y}, {origin.x + width, y},
                         ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.5f));
        }
        const float bar = width / float(std::max(kHistory, m_frames.size()));
        const float first = origin.x + width - bar * float(m_frames.size());
        const Frame* inspected = Inspected();
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        for (size_t i = 0; i < m_frames.size(); ++i) {
            const Frame& frame = m_frames[i];
            const double ms = FrameMs(frame);
            const float x = first + bar * float(i);
            const bool under = hovered && mouse.x >= x && mouse.x < x + bar;
            const ImGuiCol color = inspected != nullptr && frame.index == inspected->index
                                       ? ImGuiCol_PlotHistogramHovered
                                       : (under ? ImGuiCol_ButtonHovered : ImGuiCol_PlotHistogram);
            draw.AddRectFilled({x, origin.y + height - barHeight(ms)},
                               {x + std::max(bar - 1.0f, 1.0f), origin.y + height},
                               ImGui::GetColorU32(color));
            if (under) {
                ImGui::SetTooltip("Frame %llu: %.2f ms", (unsigned long long)frame.index, ms);
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    m_selected = frame.index;
                    m_paused = true;
                }
            }
        }
        if (inspected != nullptr) {
            ImGui::Text("Frame %llu: %.2f ms, %zu zones", (unsigned long long)inspected->index,
                        FrameMs(*inspected), inspected->zones.size());
        }
    }

    void DrawTracks(const Frame& frame) {
        uint64_t low = frame.startNs;
        uint64_t high = std::max(frame.endNs, frame.startNs + 1);
        for (const profiler::Zone& zone : frame.zones) {
            low = std::min(low, zone.startNs);
            high = std::max(high, zone.endNs);
        }
        const double total = double(high - low);
        const float font = ImGui::GetFontSize();
        const float lane = font + 4.0f;
        const float axis = font + 6.0f;
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 size = ImGui::GetContentRegionAvail();
        size = {std::max(size.x, 1.0f), std::max(size.y, 4.0f * lane)};
        ImGui::InvisibleButton("##tracks", size);
        const bool hovered = ImGui::IsItemHovered();
        const ImGuiIO& io = ImGui::GetIO();
        const float width = size.x;
        if (hovered && io.MouseWheel != 0) {
            const double at = (io.MousePos.x - origin.x) / width;
            const double time = m_viewStart + at * m_viewWidth;
            m_viewWidth = std::clamp(m_viewWidth * std::pow(0.8, io.MouseWheel), 1e-5, 1.0);
            m_viewStart = time - at * m_viewWidth;
        }
        if (ImGui::IsItemActive() && io.MouseDelta.x != 0) {
            m_viewStart -= io.MouseDelta.x / width * m_viewWidth;
        }
        if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            m_viewStart = 0;
            m_viewWidth = 1;
        }
        m_viewStart = std::clamp(m_viewStart, 0.0, 1.0 - m_viewWidth);
        const auto toX = [&](uint64_t ns) {
            const double t = double(ns - low) / total;
            return origin.x + float((t - m_viewStart) / m_viewWidth) * width;
        };

        ImDrawList& draw = *ImGui::GetWindowDrawList();
        draw.PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);
        draw.AddRectFilled(origin, {origin.x + size.x, origin.y + size.y},
                           ImGui::GetColorU32(ImGuiCol_FrameBg));
        const double shownMs = total * m_viewWidth * 1e-6;
        double step = std::pow(10.0, std::floor(std::log10(shownMs / 8.0)));
        step *= shownMs / step > 40 ? 5 : shownMs / step > 16 ? 2 : 1;
        const double firstMs = m_viewStart * total * 1e-6;
        const ImU32 muted = ImGui::GetColorU32(ImGuiCol_TextDisabled);
        for (double ms = std::ceil(firstMs / step) * step; ms <= firstMs + shownMs; ms += step) {
            const float x = toX(low + uint64_t(ms * 1e6));
            draw.AddLine({x, origin.y}, {x, origin.y + size.y},
                         ImGui::GetColorU32(ImGuiCol_Border, 0.5f));
            char label[32];
            std::snprintf(label, sizeof(label), "%g ms", ms);
            draw.AddText({x + 3.0f, origin.y + 2.0f}, muted, label);
        }

        std::map<std::pair<bool, uint32_t>, std::vector<const profiler::Zone*>> tracks;
        for (const profiler::Zone& zone : frame.zones) {
            const bool gpu = zone.track < m_tracks.size() && m_tracks[zone.track].gpu;
            tracks[{!gpu, zone.track}].push_back(&zone);
        }
        float y = origin.y + axis;
        const profiler::Zone* under = nullptr;
        for (const auto& [key, zones] : tracks) {
            const std::string name = key.second < m_tracks.size()
                                         ? m_tracks[key.second].name
                                         : "Track " + std::to_string(key.second);
            draw.AddText({origin.x + 4.0f, y}, muted, name.c_str());
            y += lane;
            uint32_t depth = 0;
            for (const profiler::Zone* zone : zones) {
                depth = std::max(depth, zone->depth);
                const float x0 = toX(zone->startNs);
                const float x1 = std::max(toX(zone->endNs), x0 + 1.0f);
                if (x1 < origin.x || x0 > origin.x + width) {
                    continue;
                }
                const float top = y + float(zone->depth) * lane;
                const ImVec2 a(x0, top);
                const ImVec2 b(x1, top + lane - 1.0f);
                const std::string_view zoneName = zone->id.Name();
                draw.AddRectFilled(a, b, NameColor(zoneName));
                if (x1 - x0 > 3.0f * font) {
                    draw.PushClipRect(a, b, true);
                    draw.AddText({std::max(x0, origin.x) + 3.0f, top + 2.0f},
                                 IM_COL32(20, 20, 20, 255), zoneName.data(),
                                 zoneName.data() + zoneName.size());
                    draw.PopClipRect();
                }
                if (hovered && io.MousePos.x >= a.x && io.MousePos.x < b.x &&
                    io.MousePos.y >= a.y && io.MousePos.y < b.y) {
                    under = zone;
                }
            }
            y += float(depth + 1) * lane + 4.0f;
        }
        draw.PopClipRect();
        if (under != nullptr) {
            const std::string name(under->id.Name());
            ImGui::SetTooltip("%s\n%.3f ms, at %.3f ms", name.c_str(),
                              double(under->endNs - under->startNs) * 1e-6,
                              double(under->startNs - low) * 1e-6);
        } else if (hovered) {
            ImGui::SetItemTooltip("Wheel to zoom, drag to pan, double click to show it all");
        }
    }

    bool m_open = false;
    bool m_ownsRecording = false;
    bool m_paused = false;
    bool m_byTime = false;
    double m_minMs = 0;
    int m_depth = 8;
    bool m_average = true;
    double m_averageSeconds = 1;
    double m_viewStart = 0;
    double m_viewWidth = 1;
    std::vector<Frame> m_frames;
    std::vector<profiler::Track> m_tracks;
    std::optional<uint64_t> m_selected;
    profiler::Report m_report;
};

}

Panel CreateProfilerPanel() {
    auto profiler = std::make_shared<Profiler>();
    return {"Profiler", Dock::Bottom, [profiler] { profiler->Draw(); }, false,
            [profiler] { profiler->Hidden(); }};
}

}
