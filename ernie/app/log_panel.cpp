#include "ernie/app/builtin_panels.h"
#include "ernie/app/viewer.h"

#include <imgui.h>

#include <string>
#include <string_view>

namespace ernie::app {

namespace {

class LogPanel final : public Panel {
  public:
    LogPanel() : Panel("Log", Dock::Bottom) {}

    void Draw(Viewer& viewer) override {
        ImGui::Checkbox("Info", &show_[0]);
        ImGui::SameLine();
        ImGui::Checkbox("Warnings", &show_[1]);
        ImGui::SameLine();
        ImGui::Checkbox("Errors", &show_[2]);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200.0f);
        ImGui::InputTextWithHint("##filter", "Filter", filter_, sizeof(filter_));
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            viewer.ClearLog();
        }
        ImGui::SameLine();
        const bool copy = ImGui::Button("Copy");
        std::string copied;
        if (ImGui::BeginChild("##log", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders)) {
            const std::string_view filter(filter_);
            const ImVec4 colors[] = {ImGui::GetStyleColorVec4(ImGuiCol_Text),
                                     ImVec4(0.91f, 0.70f, 0.24f, 1.0f),
                                     ImVec4(0.89f, 0.34f, 0.32f, 1.0f)};
            for (const LogEntry& entry : viewer.LogEntries()) {
                const auto level = static_cast<std::size_t>(entry.level);
                if (!show_[level] ||
                    (!filter.empty() && entry.text.find(filter) == std::string::npos)) {
                    continue;
                }
                ImGui::TextDisabled("%8.2f", entry.time);
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, colors[level]);
                ImGui::TextWrapped("%s", entry.text.c_str());
                ImGui::PopStyleColor();
                if (copy) {
                    copied += entry.text + "\n";
                }
            }
            // Follow new messages while scrolled to the end.
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) {
                ImGui::SetScrollHereY(1.0f);
            }
        }
        ImGui::EndChild();
        if (copy) {
            ImGui::SetClipboardText(copied.c_str());
        }
    }

  private:
    bool show_[3] = {true, true, true};
    char filter_[128] = {};
};

}  // namespace

std::unique_ptr<Panel> MakeLogPanel() {
    return std::make_unique<LogPanel>();
}

}  // namespace ernie::app
