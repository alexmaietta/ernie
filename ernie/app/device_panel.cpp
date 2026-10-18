// SPDX-License-Identifier: BSD-3-Clause
/// @file device_panel.cpp
/// @brief The Devices panel.
#include "ernie/app/panels.h"
#include "ernie/app/widgets.h"
#include "ernie/device/context.h"

#include <imgui.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ernie::app {
namespace {

std::string ApiVersion(const VulkanDeviceInfo& device) {
    return std::to_string(device.apiMajor) + "." + std::to_string(device.apiMinor);
}

/// @return The names of the capabilities a device has.
std::vector<const char*> CapabilityNames(const DeviceCapabilities& capabilities) {
    const std::pair<bool, const char*> all[] = {
        {capabilities.rayQuery, "ray queries"},
        {capabilities.externalMemory, "shared memory"},
        {capabilities.externalTimeline, "shared timelines"},
        {capabilities.shaderInt64, "64-bit integers"},
        {capabilities.shaderFloat16, "16-bit floats"},
        {capabilities.memoryBudget, "memory budget"},
        {capabilities.deviceFault, "fault reports"},
        {capabilities.calibratedTimestamps, "GPU zones"},
        {capabilities.present, "presenting"},
        {capabilities.sparseResidency, "sparse residency"},
    };
    std::vector<const char*> names;
    for (const auto& [has, name] : all) {
        if (has) {
            names.push_back(name);
        }
    }
    return names;
}

class DevicePanel final : public Panel {
  public:
    explicit DevicePanel(const DeviceContext* device)
        : Panel("Devices", Dock::RightLower), device_(device) {}

    void Draw() override {
        if (device_ != nullptr) {
            DrawCurrent(*device_);
        } else {
            ImGui::TextDisabled("No device in use.");
        }
        // Listed when first opened: listing makes a Vulkan instance.
        if (!ImGui::CollapsingHeader("Installed")) {
            return;
        }
        if (ImGui::Button("Refresh") || !installed_) {
            List();
        }
        if (!error_.empty()) {
            ImGui::TextWrapped("%s", error_.c_str());
        }
        DrawInstalled();
    }

  private:
    static void DrawCurrent(const DeviceContext& device) {
        const VulkanDeviceInfo& info = device.Info();
        ImGui::SeparatorText("In use");
        LabelValue("Name", info.name);
        LabelValue("Type", ToString(info.type));
        LabelValue("Vulkan", ApiVersion(info));
        LabelValue("UUID", info.uuid);
        std::string capabilities;
        for (const char* name : CapabilityNames(info.capabilities)) {
            capabilities += (capabilities.empty() ? "" : ", ") + std::string(name);
        }
        ImGui::TextUnformatted("Capabilities");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s", capabilities.empty() ? "none" : capabilities.c_str());
        ImGui::PopStyleColor();
        const Transfers moved = device.Transfers();
        LabelValue("Uploaded", FormatBytes(moved.uploaded));
        LabelValue("Read back", FormatBytes(moved.readBack));
    }

    void List() {
        try {
            installed_ = ListVulkanDevices();
            error_.clear();
        } catch (const DeviceError& error) {
            installed_ = std::vector<VulkanDeviceInfo>();
            error_ = error.what();
        }
    }

    void DrawInstalled() const {
        constexpr ImGuiTableFlags kFlags =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;
        if (!ImGui::BeginTable("##devices", 4, kFlags)) {
            return;
        }
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Vulkan");
        ImGui::TableSetupColumn("Supported");
        ImGui::TableHeadersRow();
        for (const VulkanDeviceInfo& device : *installed_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(device.name.c_str());
            Tooltip(device.uuid);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(ToString(device.type));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(ApiVersion(device).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(device.supported ? "yes" : "no");
        }
        ImGui::EndTable();
    }

    const DeviceContext* device_;
    std::optional<std::vector<VulkanDeviceInfo>> installed_;
    std::string error_;
};

}  // namespace

std::unique_ptr<Panel> MakeDevicePanel(const DeviceContext* device) {
    return std::make_unique<DevicePanel>(device);
}

}  // namespace ernie::app
