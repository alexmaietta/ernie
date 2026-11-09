// SPDX-License-Identifier: BSD-3-Clause
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "ernie/app/viewer/panels/panels.h"
#include "ernie/app/viewer/widgets/widgets.h"
#include "ernie/stage/schema/schema.h"
#include "ernie/util/log.h"

namespace ernie::app {
namespace {

constexpr size_t kLogKept = 10000;

const char* Prefix(LogLevel level) {
    switch (level) {
    case LogLevel::Debug:
        return "ernie: debug: ";
    case LogLevel::Warning:
        return "ernie: warning: ";
    case LogLevel::Error:
        return "ernie: error: ";
    default:
        return "ernie: ";
    }
}

void CountTypes(const Prim& prim, std::map<std::string, uint32_t>& types) {
    ++types[std::string(Info(prim.Schema()).name)];
    for (const Prim& child : prim.Children()) {
        CountTypes(child, types);
    }
}

class LogCapture {
  public:
    struct Entry {
        double time;
        LogLevel level;
        std::string message;
    };

    LogCapture() : echoed(GetLogLevel()) {
        SetLogSink([this](LogLevel level, std::string_view message) {
            if (level >= echoed) {
                std::fprintf(stderr, "%s%.*s\n", Prefix(level), int(message.size()),
                             message.data());
            }
            const std::lock_guard lock(mutex);
            entries.push_back({std::chrono::duration<double>(Clock::now() - m_start).count(), level,
                               std::string(message)});
            if (entries.size() > kLogKept) {
                entries.pop_front();
            }
        });
    }

    ~LogCapture() {
        SetLogSink(nullptr);
        SetLogLevel(echoed);
    }

    LogCapture(const LogCapture&) = delete;
    LogCapture& operator=(const LogCapture&) = delete;

    const LogLevel echoed;
    std::mutex mutex;
    std::deque<Entry> entries;

  private:
    using Clock = std::chrono::steady_clock;

    Clock::time_point m_start = Clock::now();
};

}

Panel CreateInfoPanel(Ref<Scene> scene, Ref<device::Device> device,
                      std::shared_ptr<RenderDelegate> delegate) {
    struct Types {
        uint64_t version = ~uint64_t(0);
        std::map<std::string, uint32_t> counts;
    };

    auto types = std::make_shared<Types>();
    return {"Info", Dock::Bottom, [scene, device, delegate, types] {
                if (!ImGui::BeginTable("##info", 3, ImGuiTableFlags_SizingStretchSame)) {
                    return;
                }
                ImGui::TableNextColumn();
                ImGui::SeparatorText("Scene");
                if (scene) {
                    Row("Prims", std::to_string(scene->PrimCount()));
                    Row("Worlds", std::to_string(scene->WorldCount()));
                    if (delegate) {
                        Row("Renderer", std::string(delegate->Name()));
                    }
                } else {
                    ImGui::TextDisabled("No scene");
                }

                ImGui::TableNextColumn();
                ImGui::SeparatorText("Prims by type");
                if (scene && types->version != scene->StructureVersion()) {
                    types->version = scene->StructureVersion();
                    types->counts.clear();
                    const View roots = scene->Select("/*");
                    for (const uint32_t root : roots.Prims()) {
                        CountTypes(scene->ByIndex(root), types->counts);
                    }
                }
                for (const auto& [type, count] : types->counts) {
                    Row(type.c_str(), std::to_string(count));
                }

                ImGui::TableNextColumn();
                ImGui::SeparatorText("Device");
                const device::AdapterInfo& adapter = device->Adapter();
                const device::MemoryStats memory = device->MemoryUsage();
                Row("GPU", adapter.name);
                Row("Backend", std::string(device::BackendName(adapter.backend)));
                Row("Device memory",
                    FormatBytes(memory.deviceBytes) + " of " + FormatBytes(adapter.memoryBytes));
                Row("Upload", FormatBytes(memory.uploadBytes));
                Row("Readback", FormatBytes(memory.readbackBytes));
                Row("Shared", FormatBytes(memory.sharedBytes));
                ImGui::EndTable();
            }};
}

Panel CreateLogPanel(const Theme& theme) {
    struct State {
        LogCapture capture;
        bool shown[4] = {GetLogLevel() == LogLevel::Debug, true, true, true};
        std::string filter;
    };

    auto state = std::make_shared<State>();
    return {"Log", Dock::Bottom, [state, &theme] {
                const char* levels[] = {"Debug", "Info", "Warnings", "Errors"};
                for (int level = 0; level < 4; ++level) {
                    if (ImGui::Checkbox(levels[level], &state->shown[level]) && level == 0) {
                        SetLogLevel(state->shown[0] ? LogLevel::Debug : state->capture.echoed);
                    }
                    ImGui::SameLine();
                }
                ImGui::SetNextItemWidth(14.0f * ImGui::GetFontSize());
                ImGui::InputTextWithHint("##filter", "Filter", &state->filter);
                ImGui::SameLine();
                const std::lock_guard lock(state->capture.mutex);
                if (ImGui::Button("Clear")) {
                    state->capture.entries.clear();
                }
                ImGui::SameLine();
                const bool copy = ImGui::Button("Copy");
                std::string copied;
                if (ImGui::BeginChild("##entries", {0, 0}, ImGuiChildFlags_Borders)) {
                    for (const LogCapture::Entry& entry : state->capture.entries) {
                        if (!state->shown[int(entry.level)] ||
                            (!state->filter.empty() &&
                             entry.message.find(state->filter) == std::string::npos)) {
                            continue;
                        }
                        ImGui::TextDisabled("%8.2f", entry.time);
                        ImGui::SameLine();
                        const ImVec4 color = entry.level == LogLevel::Error     ? theme.error
                                             : entry.level == LogLevel::Warning ? theme.warning
                                                                                : theme.text;
                        ImGui::PushStyleColor(ImGuiCol_Text, color);
                        ImGui::TextWrapped("%s", entry.message.c_str());
                        ImGui::PopStyleColor();
                        if (copy) {
                            copied += entry.message + "\n";
                        }
                    }
                    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) {
                        ImGui::SetScrollHereY(1.0f);
                    }
                }
                ImGui::EndChild();
                if (copy) {
                    ImGui::SetClipboardText(copied.c_str());
                }
            }};
}

}
