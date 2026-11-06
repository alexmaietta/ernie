// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include "ernie/app/viewer/panels/panels.h"
#include "ernie/app/viewer/widgets/widgets.h"

namespace ernie::app {
namespace {

constexpr ImGuiTableFlags kTable =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;

void Bar(uint64_t used, uint64_t total, const char* tooltip) {
    const float share = total > 0 ? std::min(float(used) / float(total), 1.0f) : 0.0f;
    const std::string label = FormatBytes(used) + " / " + FormatBytes(total);
    ImGui::ProgressBar(share, {-1.0f, 0.0f}, label.c_str());
    ImGui::SetItemTooltip("%s", tooltip);
}

class Memory {
  public:
    explicit Memory(Ref<device::Device> device) : m_device(std::move(device)) {}

    void Draw() {
        const float em = ImGui::GetFontSize();
        const device::MemoryStats memory = m_device->MemoryUsage();

        ImGui::SeparatorText("Ernie");
        Bar(memory.deviceBytes, m_device->Adapter().memoryBytes,
            "Ernie's device memory against the adapter's");
        Row("Textures", FormatBytes(memory.textureBytes));
        Row("Upload", FormatBytes(memory.uploadBytes));
        Row("Readback", FormatBytes(memory.readbackBytes));
        Row("Shared", FormatBytes(memory.sharedBytes));
        Row("Reserved", FormatBytes(memory.reservedBytes));

        if (!memory.heaps.empty()) {
            ImGui::SeparatorText("Heaps");
            for (size_t i = 0; i < memory.heaps.size(); ++i) {
                const device::MemoryHeap& heap = memory.heaps[i];
                ImGui::PushID(int(i));
                ImGui::TextDisabled("%s heap %zu, %s", heap.deviceLocal ? "Device" : "Host", i,
                                    FormatBytes(heap.size).c_str());
                Bar(heap.usage, heap.budget,
                    "Every process's use of the heap against the budget the driver gives");
                ImGui::PopID();
            }
        }

        DrawTraffic();

        ImGui::SeparatorText("Allocations by Label");
        uint64_t allocated = 0;
        for (const device::MemoryUse& use : memory.uses) {
            allocated += use.bytes;
        }
        if (ImGui::BeginTable("##uses", 5, kTable | ImGuiTableFlags_ScrollY)) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Buffers", ImGuiTableColumnFlags_WidthFixed, 4.0f * em);
            ImGui::TableSetupColumn("Textures", ImGuiTableColumnFlags_WidthFixed, 4.0f * em);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 6.0f * em);
            ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthFixed, 8.0f * em);
            ImGui::TableHeadersRow();
            for (const device::MemoryUse& use : memory.uses) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(use.label.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%u", use.buffers);
                ImGui::TableNextColumn();
                ImGui::Text("%u", use.textures);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(FormatBytes(use.bytes).c_str());
                ImGui::TableNextColumn();
                const float part = allocated > 0 ? float(use.bytes) / float(allocated) : 0.0f;
                char label[16];
                std::snprintf(label, sizeof(label), "%.1f%%", 100.0f * part);
                ImGui::ProgressBar(part, {-1.0f, 0.0f}, label);
            }
            ImGui::EndTable();
        }
    }

  private:
    void DrawTraffic() {
        ImGui::SeparatorText("Traffic");
        const auto now = std::chrono::steady_clock::now();
        const device::TrafficStats traffic = m_device->Traffic();
        const double seconds = std::chrono::duration<double>(now - m_sampled).count();
        if (seconds >= 0.5) {
            if (m_sampled != std::chrono::steady_clock::time_point{}) {
                const device::TrafficStats delta = traffic - m_traffic;
                m_rates = {
                    double(delta.uploadBytes) / seconds, double(delta.readbackBytes) / seconds,
                    double(delta.inlineBytes) / seconds, double(delta.copyBytes) / seconds,
                    double(delta.submissions) / seconds, double(delta.hostWaits) / seconds};
            }
            m_traffic = traffic;
            m_sampled = now;
        }
        Row("Uploads", FormatBytes(uint64_t(m_rates[0])) + "/s");
        Row("Readbacks", FormatBytes(uint64_t(m_rates[1])) + "/s");
        Row("Inline", FormatBytes(uint64_t(m_rates[2])) + "/s");
        Row("Copies", FormatBytes(uint64_t(m_rates[3])) + "/s");
        Row("Submissions", std::format("{:.0f}/s", m_rates[4]));
        Row("Host waits", std::format("{:.0f}/s", m_rates[5]));
    }

    Ref<device::Device> m_device;
    device::TrafficStats m_traffic;
    std::chrono::steady_clock::time_point m_sampled;
    std::array<double, 6> m_rates{};
};

}

Panel CreateMemoryPanel(Ref<device::Device> device) {
    auto memory = std::make_shared<Memory>(std::move(device));
    return {"Memory", Dock::Bottom, [memory] { memory->Draw(); }, false};
}

}
