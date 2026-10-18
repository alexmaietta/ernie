// SPDX-License-Identifier: BSD-3-Clause
/// @file log_panel.cpp
/// @brief The Log panel.
#include "ernie/app/message_log.h"
#include "ernie/app/panels.h"
#include "ernie/app/theme.h"
#include "ernie/app/widgets.h"

#include <imgui.h>

#include <array>
#include <string>

namespace ernie::app {
namespace {

class LogPanel final : public Panel {
  public:
    LogPanel(MessageLog& log, const Theme& theme)
        : Panel("Log", Dock::Bottom), log_(log), theme_(theme) {}

    void Draw() override {
        ImGui::Checkbox("Info", &shown_[0]);
        ImGui::SameLine();
        ImGui::Checkbox("Warnings", &shown_[1]);
        ImGui::SameLine();
        ImGui::Checkbox("Errors", &shown_[2]);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(14.0f * ImGui::GetFontSize());
        ImGui::InputTextWithHint("##filter", "Filter", filter_.data(), filter_.size());
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            log_.Clear();
        }
        ImGui::SameLine();
        const bool copy = ImGui::Button("Copy");
        std::string copied;
        if (ImGui::BeginChild("##messages", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders)) {
            const std::string_view filter(filter_.data());
            for (const Message& message : log_.Messages()) {
                const auto severity = static_cast<std::size_t>(message.severity);
                if (!shown_[severity] ||
                    (!filter.empty() && message.text.find(filter) == std::string::npos)) {
                    continue;
                }
                ImGui::TextDisabled("%8.2f", message.time);
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, Colour(message.severity));
                ImGui::TextWrapped("%s", message.text.c_str());
                ImGui::PopStyleColor();
                if (copy) {
                    copied += message.text + "\n";
                }
            }
            // Follows new messages while scrolled to the end.
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
    ImVec4 Colour(Severity severity) const {
        switch (severity) {
        case Severity::Warning:
            return ToImVec4(theme_.warning);
        case Severity::Error:
            return ToImVec4(theme_.error);
        case Severity::Info:
            break;
        }
        return ImGui::GetStyleColorVec4(ImGuiCol_Text);
    }

    MessageLog& log_;
    const Theme& theme_;
    std::array<bool, 3> shown_{true, true, true};
    std::array<char, 128> filter_{};
};

}  // namespace

std::unique_ptr<Panel> MakeLogPanel(MessageLog& log, const Theme& theme) {
    return std::make_unique<LogPanel>(log, theme);
}

}  // namespace ernie::app
