// SPDX-License-Identifier: BSD-3-Clause
/// @file ui.cpp
/// @brief The viewer's UI, in the sections of ui.h: the panels, the theme and widgets, the orbit
/// camera, the shell and its menus, layout, mouse controls and heads-up display, and the
/// presenters: an offscreen image, and GLFW's windows, routed through volk's Vulkan loader, with
/// their swapchains and their input through ImGui's GLFW backend.
#include "ernie/app/ui.h"

#include "ernie/app/create.h"
#include "ernie/device/context.h"
#include "ernie/scene/json.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/render.h"
#include "ernie/util/file.h"
#include "ernie/util/on_failure.h"
#include "ernie/util/profiler.h"
#include "ernie/util/text.h"
#include "ernie/util/version.h"

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui_internal.h>
#include <imgui_stdlib.h>

#if ERNIE_VIEWER_WINDOW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#endif

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace ernie::app {

// Panels

void MessageLog::Add(Severity severity, std::string text) {
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_;
    messages_.push_back({elapsed.count(), severity, std::move(text)});
    while (messages_.size() > limit_) {
        messages_.pop_front();
    }
}

std::unordered_set<Path> StageMatches(const Scene& scene, std::string_view filter) {
    std::unordered_set<Path> matches;
    if (filter.empty()) {
        return matches;
    }
    const std::string wanted = ToLower(filter);
    for (const Path& path : scene.Prims()) {
        if (ToLower(path.String()).find(wanted) == std::string::npos) {
            continue;
        }
        for (Path shown = path; !shown.IsEmpty() && !shown.IsRoot(); shown = shown.Parent()) {
            if (!matches.insert(shown).second) {
                break;
            }
        }
    }
    return matches;
}

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
                if (!shown_[static_cast<std::size_t>(message.severity)] ||
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

class StagePanel final : public Panel {
  public:
    StagePanel(const Scene& scene, Selection& selection)
        : Panel("Stage", Dock::Right), scene_(scene), selection_(selection) {}

    void Draw() override {
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::InputTextWithHint("##search", "Search", search_.data(), search_.size()) ||
            version_ != scene_.Version()) {
            matches_ = StageMatches(scene_, search_.data());
            version_ = scene_.Version();
        }
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                           ImGuiTableFlags_Resizable |
                                           ImGuiTableFlags_BordersInnerV;
        if (!ImGui::BeginTable("##prims", 2, kFlags)) {
            return;
        }
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed,
                                8.0f * ImGui::GetFontSize());
        ImGui::TableHeadersRow();
        DrawChildren(Path::Root());
        ImGui::EndTable();
    }

  private:
    bool Searching() const {
        return search_[0] != '\0';
    }

    void DrawChildren(const Path& parent) {
        for (const Path& child : scene_.Children(parent)) {
            if (!Searching() || matches_.contains(child)) {
                DrawPrim(child);
            }
        }
    }

    void DrawPrim(const Path& path) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const bool leaf = scene_.Children(path).empty();
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAllColumns |
                                   ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_OpenOnDoubleClick;
        if (leaf) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        }
        if (selection_.Contains(path)) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }
        // Searching opens the matches' ancestors.
        if (Searching()) {
            ImGui::SetNextItemOpen(true);
        }
        const std::string name(path.Name());
        const bool open = ImGui::TreeNodeEx(path.String().c_str(), flags, "%s", name.c_str());
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            if (ImGui::GetIO().KeyCtrl) {
                selection_.Toggle(path);
            } else {
                selection_.Set(path);
            }
        }
        TableTexts({scene_.TypeName(scene_.Require(path))});
        if (open && !leaf) {
            DrawChildren(path);
            ImGui::TreePop();
        }
    }

    const Scene& scene_;
    Selection& selection_;
    std::array<char, 128> search_{};
    std::unordered_set<Path> matches_;
    std::uint64_t version_ = 0;
};

/// @return The world box around the prims of `paths` the scene has.
Box WorldBounds(const Scene& scene, const std::vector<Path>& paths) {
    Box bounds;
    for (const Path& path : paths) {
        if (scene.Has(path)) {
            bounds.Extend(scene.WorldBounds(path));
        }
    }
    return bounds;
}

std::string VectorText(const glm::dvec3& vector) {
    std::array<char, 96> text{};
    std::snprintf(text.data(), text.size(), "%.4g, %.4g, %.4g", vector.x, vector.y, vector.z);
    return text.data();
}

std::string BoxText(const Box& box) {
    return box.Empty() ? "empty" : VectorText(box.min) + " to " + VectorText(box.max) + " m";
}

std::string_view TypeLabel(std::string_view type) {
    return type.empty() ? "untyped" : type;
}

std::string ValueText(const Value& value) {
    const auto numbers = [](const auto& components) {
        return Join(components, ", ", [](auto n) { return ToText(static_cast<double>(n)); });
    };
    const auto vector = [&]<glm::length_t N, typename T>(const glm::vec<N, T>& v) {
        return "(" + numbers(std::span(glm::value_ptr(v), static_cast<std::size_t>(N))) + ")";
    };
    return std::visit(
        [&]<typename T>(const T& data) -> std::string {
            if constexpr (std::is_same_v<T, std::monostate>) {
                return "none";
            } else if constexpr (std::is_same_v<T, bool>) {
                return data ? "true" : "false";
            } else if constexpr (std::is_arithmetic_v<T>) {
                return ToText(static_cast<double>(data));
            } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                return Join(std::array{data[0], data[1], data[2], data[3]}, " ", vector);
            } else if constexpr (std::is_same_v<T, glm::dquat>) {
                return vector(glm::dvec4(data.w, data.x, data.y, data.z));
            } else if constexpr (std::is_same_v<T, std::string>) {
                return data;
            } else if constexpr (std::is_same_v<T, Path>) {
                return data.String();
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                return "[" + numbers(data) + "]";
            } else if constexpr (std::is_same_v<T, std::vector<Path>>) {
                return "[" + Join(data, ", ", &Path::String) + "]";
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                return "[" + Join(data, ", ") + "]";
            } else {
                return vector(data);
            }
        },
        value.Raw());
}

std::string AttributeText(const Scene& scene, PrimId prim, const std::string& attribute) {
    const std::size_t samples = scene.Samples(prim, attribute).size();
    const std::string sampled = samples > 0 ? ", " + std::to_string(samples) + " samples" : "";
    if (const Tensor* array = scene.Read(prim, attribute)) {
        return "array of " + Join(array->Shape(), " x ", [](auto n) { return std::to_string(n); }) +
               ", " + FormatBytes(array->Bytes()) + (array->OnHost() ? "" : " on a device") +
               sampled;
    }
    const std::optional<Value> value = scene.Get(prim, attribute);
    return (value ? ValueText(*value) : "none") + sampled;
}

/// @brief After Omniverse Kit's Property window, read-only: the first selected prim's path, type,
/// API schemas and bounds, then its authored attributes.
class PropertiesPanel final : public Panel {
  public:
    PropertiesPanel(const Scene& scene, const Selection& selection)
        : Panel("Properties", Dock::RightLower), scene_(scene), selection_(selection) {}

    void Draw() override {
        if (selection_.Empty() || !scene_.Has(selection_.Paths().front())) {
            ImGui::TextDisabled("Select a prim in the Stage.");
            return;
        }
        const Path& path = selection_.Paths().front();
        const PrimId prim = scene_.Require(path);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##search", "Search", search_.data(), search_.size());
        LabelValue("Path", path.String());
        LabelValue("Type", TypeLabel(scene_.TypeName(prim)));
        if (const std::vector<std::string> apis = scene_.AppliedApis(prim); !apis.empty()) {
            LabelValue("API schemas", Join(apis, ", "));
        }
        LabelValue("World bounds", BoxText(scene_.WorldBounds(path)));
        if (const std::size_t selected = selection_.Paths().size(); selected > 1) {
            ImGui::TextDisabled("The first of %zu selected.", selected);
        }
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                           ImGuiTableFlags_Resizable |
                                           ImGuiTableFlags_BordersInnerV;
        if (!ImGui::BeginTable("##attributes", 2, kFlags)) {
            return;
        }
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Attribute");
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        const std::string wanted = ToLower(search_.data());
        for (const std::string& attribute : scene_.Attributes(path)) {
            if (ToLower(attribute).find(wanted) == std::string::npos) {
                continue;
            }
            ImGui::TableNextRow();
            TableTexts({attribute, AttributeText(scene_, prim, attribute)});
        }
        ImGui::EndTable();
    }

  private:
    const Scene& scene_;
    const Selection& selection_;
    std::array<char, 128> search_{};
};

/// @brief After Ernie 1's Info panel: the scene's scale and the device's use, in sections side by
/// side, as the bottom dock is wide and short.
class InfoPanel final : public Panel {
  public:
    InfoPanel(const Scene& scene, const DeviceContext* device)
        : Panel("Info", Dock::Bottom), scene_(scene), device_(device) {}

    void Draw() override {
        if (counted_ != scene_.Version()) {
            Count();
        }
        if (!ImGui::BeginTable("##info", 3, ImGuiTableFlags_SizingStretchSame)) {
            return;
        }
        ImGui::TableNextColumn();
        ImGui::SeparatorText("Scene");
        LabelValue("Prims", std::to_string(prims_));
        LabelValue("Bounds", BoxText(bounds_));
        LabelValue("Size", bounds_.Empty() ? "none" : VectorText(bounds_.max - bounds_.min) + " m");
        LabelValue("Time code", ToText(scene_.Time()));
        ImGui::TableNextColumn();
        ImGui::SeparatorText("Prims by type");
        for (const auto& [type, count] : types_) {
            LabelValue(TypeLabel(type), std::to_string(count));
        }
        ImGui::TableNextColumn();
        ImGui::SeparatorText("Device");
        DrawDevice();
        ImGui::EndTable();
    }

  private:
    void Count() {
        prims_ = 0;
        types_.clear();
        for (const Path& path : scene_.Prims()) {
            if (!path.IsRoot()) {
                ++prims_;
                ++types_[std::string(scene_.TypeName(scene_.Require(path)))];
            }
        }
        bounds_ = WorldBounds(scene_, scene_.Children(Path::Root()));
        counted_ = scene_.Version();
    }

    void DrawDevice() const {
        const float fps = ImGui::GetIO().Framerate;
        std::array<char, 48> frame{};
        std::snprintf(frame.data(), frame.size(), "%.2f ms, %.0f fps",
                      fps > 0.0f ? 1000.0 / fps : 0.0, static_cast<double>(fps));
        LabelValue("Frame", frame.data());
        if (device_ == nullptr) {
            ImGui::TextDisabled("No device in use.");
            return;
        }
        const Transfers moved = device_->Transfers();
        LabelValue("Uploaded", FormatBytes(moved.uploaded));
        LabelValue("Read back", FormatBytes(moved.readBack));
        LabelValue("Copied", FormatBytes(moved.copied));
        const VkPhysicalDeviceMemoryProperties* memory = nullptr;
        vmaGetMemoryProperties(device_->Allocator(), &memory);
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(device_->Allocator(), budgets.data());
        for (std::uint32_t heap = 0; heap < memory->memoryHeapCount; ++heap) {
            const VkMemoryHeapFlags flags = memory->memoryHeaps[heap].flags;
            const bool local = (flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            LabelValue("Heap " + std::to_string(heap) + (local ? ", device" : ", host"),
                       FormatBytes(budgets[heap].usage) + " of " +
                           FormatBytes(budgets[heap].budget));
        }
    }

    const Scene& scene_;
    const DeviceContext* device_;
    std::optional<std::uint64_t> counted_;
    std::size_t prims_ = 0;
    std::map<std::string, std::size_t> types_;
    Box bounds_;
};

std::string ApiVersion(const VulkanDeviceInfo& device) {
    return std::to_string(device.apiMajor) + "." + std::to_string(device.apiMinor);
}

/// @return The capabilities a device has, as a list of names.
std::string CapabilityNames(const DeviceCapabilities& capabilities) {
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
    std::vector<std::string_view> names;
    for (const auto& [has, name] : all) {
        if (has) {
            names.emplace_back(name);
        }
    }
    return names.empty() ? "none" : Join(names, ", ");
}

class DevicePanel final : public Panel {
  public:
    explicit DevicePanel(const DeviceContext* device)
        : Panel("Devices", Dock::Bottom), device_(device) {}

    void Draw() override {
        if (device_ != nullptr) {
            const VulkanDeviceInfo& info = device_->Info();
            ImGui::SeparatorText("In use");
            LabelValue("Name", info.name);
            LabelValue("Type", ToString(info.type));
            LabelValue("Vulkan", ApiVersion(info));
            LabelValue("UUID", info.uuid);
            LabelValue("Capabilities", CapabilityNames(info.capabilities));
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
            TableTexts({device.name});
            ImGui::SetItemTooltip("%s", device.uuid.c_str());
            TableTexts(
                {ToString(device.type), ApiVersion(device), device.supported ? "yes" : "no"});
        }
        ImGui::EndTable();
    }

    const DeviceContext* device_;
    std::optional<std::vector<VulkanDeviceInfo>> installed_;
    std::string error_;
};

using profiler::Frame;
using profiler::ReportFilter;
using profiler::ReportNode;
using profiler::ReportRow;

double FrameMs(const Frame& frame) {
    return static_cast<double>(frame.end - frame.start) * 1e-6;
}

/// @brief After Omniverse Kit's Profiler window: CPU and GPU zones averaged into trees, a bar per
/// frame, and exports as text, JSON and a Chrome trace.
class ProfilerPanel final : public Panel {
  public:
    ProfilerPanel(MessageLog& log, bool alwaysRecord)
        : Panel("Profiler", Dock::Bottom), log_(log), alwaysRecord_(alwaysRecord) {}

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
            total += FrameMs(frame);
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
            times.push_back(FrameMs(frame));
        }
        std::ranges::sort(times);
        const double longest = std::max(1000.0 / 30.0 * 1.1, 1.5 * times[times.size() * 95 / 100]);
        const auto barHeight = [&](double ms) {
            return static_cast<float>(std::min(ms / longest, 1.0)) * height;
        };
        // Guides at 60 and 30 frames per second.
        for (const double guide : {1000.0 / 60.0, 1000.0 / 30.0}) {
            const float y = origin.y + height - barHeight(guide);
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
            const double ms = FrameMs(frame);
            const float x = first + bar * static_cast<float>(i);
            const bool under = hovered && mouse.x >= x && mouse.x < x + bar;
            const ImGuiCol colour = inspected != nullptr && frame.index == inspected->index
                                        ? ImGuiCol_PlotHistogramHovered
                                        : (under ? ImGuiCol_ButtonHovered : ImGuiCol_PlotHistogram);
            draw->AddRectFilled(ImVec2(x, origin.y + height - barHeight(ms)),
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
                        static_cast<unsigned long long>(inspected->index), FrameMs(*inspected),
                        inspected->cpu.size(), inspected->gpu.size());
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

/// @brief A help window, centred when it opens, drawn while `open`; its close button clears it.
template <typename Contents> void HelpWindow(const char* title, bool& open, Contents&& contents) {
    if (!open) {
        return;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
                            ImVec2(0.5f, 0.5f));
    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDocking |
                                        ImGuiWindowFlags_AlwaysAutoResize |
                                        ImGuiWindowFlags_NoCollapse;
    if (ImGui::Begin(title, &open, kFlags)) {
        contents();
    }
    ImGui::End();
}

}  // namespace

std::unique_ptr<Panel> MakeLogPanel(MessageLog& log, const Theme& theme) {
    return std::make_unique<LogPanel>(log, theme);
}

std::unique_ptr<Panel> MakeStagePanel(const Scene& scene, Selection& selection) {
    return std::make_unique<StagePanel>(scene, selection);
}

std::unique_ptr<Panel> MakePropertiesPanel(const Scene& scene, const Selection& selection) {
    return std::make_unique<PropertiesPanel>(scene, selection);
}

std::unique_ptr<Panel> MakeInfoPanel(const Scene& scene, const DeviceContext* device) {
    return std::make_unique<InfoPanel>(scene, device);
}

std::unique_ptr<Panel> MakeDevicePanel(const DeviceContext* device) {
    return std::make_unique<DevicePanel>(device);
}

std::unique_ptr<Panel> MakeProfilerPanel(MessageLog& log, bool alwaysRecord) {
    return std::make_unique<ProfilerPanel>(log, alwaysRecord);
}

void DrawControlsWindow(bool& open) {
    constexpr std::pair<const char*, const char*> kControls[] = {
        {"Left drag", "Orbit the camera around its target"},
        {"Right drag", "Pan along the ground"},
        {"Middle drag", "Pan across the view"},
        {"Wheel", "Zoom towards the target"},
        {"F", "Frame the selection, or the whole scene"},
        {"Click in Stage", "Select a prim; Ctrl adds or removes it"},
        {"Ctrl+Q", "Quit"},
    };
    HelpWindow("Keyboard and Mouse", open, [&] {
        if (ImGui::BeginTable("##controls", 2,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
            for (const auto& [input, action] : kControls) {
                ImGui::TableNextRow();
                TableTexts({input, action});
            }
            ImGui::EndTable();
        }
    });
}

void DrawAboutWindow(bool& open, std::string_view device) {
    HelpWindow("About Ernie", open, [&] {
        ImGui::TextUnformatted(
            "Ernie: Efficient Renderer for Natural and Interplanetary Environments");
        LabelValue("Version", Version());
        LabelValue("Device", device.empty() ? std::string_view("none") : device);
        LabelValue("Dear ImGui", IMGUI_VERSION);
        ImGui::Separator();
        ImGui::TextUnformatted("BSD-3-Clause licence. https://github.com/alexmaietta/ernie");
    });
}

// Theme and widgets

namespace {

ImVec2 ToImVec2(const glm::vec2& size) {
    return {size.x, size.y};
}

glm::vec4 WithAlpha(const glm::vec4& colour, float alpha) {
    return {colour.r, colour.g, colour.b, alpha};
}

void SetMetrics(const Theme& theme, ImGuiStyle& style) {
    style.WindowPadding = {8.0f, 6.0f};
    style.FramePadding = ToImVec2(theme.framePadding);
    style.ItemSpacing = ToImVec2(theme.itemSpacing);
    style.ItemInnerSpacing = {4.0f, 3.0f};
    style.CellPadding = {4.0f, 2.0f};
    style.IndentSpacing = 14.0f;
    style.ScrollbarSize = 10.0f;
    style.GrabMinSize = 8.0f;
    style.WindowRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.FrameRounding = theme.rounding;
    style.PopupRounding = theme.rounding;
    style.ScrollbarRounding = theme.rounding;
    style.GrabRounding = theme.rounding;
    style.TabRounding = theme.rounding;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = theme.frameBorder;
    // Framed tabs, without a line over the shown one.
    style.TabBorderSize = theme.frameBorder;
    style.TabBarOverlineSize = 0.0f;
    style.DockingSeparatorSize = 1.0f;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.WindowTitleAlign = {0.0f, 0.5f};
    style.SeparatorTextBorderSize = 1.0f;
}

void SetColours(const Theme& theme, ImGuiStyle& style) {
    const auto set = [&](ImGuiCol index, const glm::vec4& colour) {
        style.Colors[index] = ToImVec4(colour);
    };
    const glm::vec4 accentDim = WithAlpha(theme.accent, 0.45f);
    const glm::vec4 accentFaint = WithAlpha(theme.accent, 0.25f);
    const glm::vec4 clear(0.0f);
    set(ImGuiCol_Text, theme.text);
    set(ImGuiCol_TextDisabled, theme.textMuted);
    set(ImGuiCol_WindowBg, theme.panel);
    set(ImGuiCol_ChildBg, clear);
    set(ImGuiCol_PopupBg, theme.popup);
    set(ImGuiCol_Border, theme.border);
    set(ImGuiCol_BorderShadow, clear);
    set(ImGuiCol_FrameBg, theme.field);
    set(ImGuiCol_FrameBgHovered, theme.fieldHovered);
    set(ImGuiCol_FrameBgActive, theme.fieldHovered);
    for (const ImGuiCol title : {ImGuiCol_TitleBg, ImGuiCol_TitleBgActive,
                                 ImGuiCol_TitleBgCollapsed, ImGuiCol_MenuBarBg}) {
        set(title, theme.background);
    }
    set(ImGuiCol_ScrollbarBg, clear);
    set(ImGuiCol_ScrollbarGrab, theme.fieldHovered);
    set(ImGuiCol_ScrollbarGrabHovered, theme.textMuted);
    set(ImGuiCol_ScrollbarGrabActive, theme.accent);
    // A white box with a black check; menus mark their items in the accent instead.
    set(ImGuiCol_CheckMark, theme.checkMark);
    set(ImGuiCol_CheckboxSelectedBg, theme.checkBox);
    set(ImGuiCol_SliderGrab, theme.buttonLit);
    set(ImGuiCol_SliderGrabActive, theme.text);
    set(ImGuiCol_Button, theme.button);
    set(ImGuiCol_ButtonHovered, theme.fieldHovered);
    set(ImGuiCol_ButtonActive, theme.buttonLit);
    // Selected and hovered rows of trees, lists and menus.
    set(ImGuiCol_Header, theme.selection);
    set(ImGuiCol_HeaderHovered, WithAlpha(theme.selection, 0.85f));
    set(ImGuiCol_HeaderActive, theme.selection);
    set(ImGuiCol_Separator, theme.separator);
    set(ImGuiCol_SeparatorHovered, accentDim);
    set(ImGuiCol_SeparatorActive, theme.accent);
    set(ImGuiCol_ResizeGrip, clear);
    set(ImGuiCol_ResizeGripHovered, accentDim);
    set(ImGuiCol_ResizeGripActive, theme.accent);
    set(ImGuiCol_InputTextCursor, theme.text);
    // Tabs on the darker bar, the shown one in the panel's colour.
    set(ImGuiCol_Tab, theme.background);
    set(ImGuiCol_TabHovered, theme.fieldHovered);
    set(ImGuiCol_TabSelected, theme.panel);
    set(ImGuiCol_TabSelectedOverline, theme.accent);
    set(ImGuiCol_TabDimmed, theme.background);
    set(ImGuiCol_TabDimmedSelected, theme.panel);
    set(ImGuiCol_TabDimmedSelectedOverline, WithAlpha(theme.accent, 0.0f));
    set(ImGuiCol_DockingPreview, accentDim);
    set(ImGuiCol_DockingEmptyBg, theme.background);
    set(ImGuiCol_PlotLines, theme.accent);
    set(ImGuiCol_PlotLinesHovered, theme.text);
    set(ImGuiCol_PlotHistogram, theme.accent);
    set(ImGuiCol_PlotHistogramHovered, theme.text);
    set(ImGuiCol_TableHeaderBg, theme.header);
    set(ImGuiCol_TableBorderStrong, theme.separator);
    set(ImGuiCol_TableBorderLight, theme.separator);
    set(ImGuiCol_TableRowBg, clear);
    set(ImGuiCol_TableRowBgAlt, WithAlpha(theme.text, 0.03f));
    set(ImGuiCol_TextLink, theme.accent);
    set(ImGuiCol_TextSelectedBg, accentFaint);
    set(ImGuiCol_TreeLines, theme.separator);
    set(ImGuiCol_DragDropTarget, theme.accent);
    set(ImGuiCol_DragDropTargetBg, accentFaint);
    set(ImGuiCol_UnsavedMarker, theme.warning);
    set(ImGuiCol_NavCursor, theme.accent);
    set(ImGuiCol_NavWindowingHighlight, theme.accent);
    set(ImGuiCol_NavWindowingDimBg, WithAlpha(theme.background, 0.6f));
    set(ImGuiCol_ModalWindowDimBg, WithAlpha(theme.background, 0.6f));
}

}  // namespace

void ApplyTheme(const Theme& theme, float contentScale) {
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    SetMetrics(theme, style);
    SetColours(theme, style);
    style.ScaleAllSizes(contentScale);
    style.FontSizeBase = theme.fontSize;
    style.FontScaleDpi = contentScale;
}

void LabelValue(std::string_view label, std::string_view value) {
    ImGui::TextUnformatted(label.data(), label.data() + label.size());
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(value.data(), value.data() + value.size());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void TableTexts(std::initializer_list<std::string_view> texts) {
    for (const std::string_view text : texts) {
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(text.data(), text.data() + text.size());
    }
}

std::string FormatBytes(std::uint64_t bytes) {
    constexpr std::array kUnits{"B", "KiB", "MiB", "GiB", "TiB"};
    auto value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < kUnits.size()) {
        value /= 1024.0;
        ++unit;
    }
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), unit == 0 ? "%.0f %s" : "%.1f %s", value, kUnits[unit]);
    return text.data();
}

// Orbit camera

namespace {

/// @return Metres per pixel at the target's depth in a view `height` pixels tall.
double MetresPerPixel(const OrbitCamera& camera, double height) {
    return 2.0 * camera.distance * std::tan(0.5 * camera.verticalFov) / std::max(height, 1.0);
}

}  // namespace

glm::dvec3 OrbitCamera::Eye() const {
    const double horizontal = std::cos(pitch) * distance;
    return target + glm::dvec3(std::sin(yaw) * horizontal, -std::cos(yaw) * horizontal,
                               std::sin(pitch) * distance);
}

glm::dmat4 OrbitCamera::Transform() const {
    return glm::inverse(glm::lookAtRH(Eye(), target, glm::dvec3(0.0, 0.0, 1.0)));
}

void OrbitCamera::Orbit(double deltaYaw, double deltaPitch) {
    yaw += deltaYaw;
    pitch = std::clamp(pitch + deltaPitch, kMinPitch, kMaxPitch);
}

void OrbitCamera::Pan(double dx, double dy, double height) {
    const glm::dvec3 right(std::cos(yaw), std::sin(yaw), 0.0);
    const glm::dvec3 forward(-std::sin(yaw), std::cos(yaw), 0.0);
    target += (-dx * right + dy * forward) * MetresPerPixel(*this, height);
}

void OrbitCamera::PanVertical(double dx, double dy, double height) {
    const glm::dvec3 right(std::cos(yaw), std::sin(yaw), 0.0);
    target += (-dx * right + dy * glm::dvec3(0.0, 0.0, 1.0)) * MetresPerPixel(*this, height);
}

void OrbitCamera::Zoom(double steps) {
    distance = std::clamp(distance * std::pow(0.9, steps), kMinDistance, kMaxDistance);
}

void OrbitCamera::Frame(const Box& box) {
    if (box.Empty()) {
        return;
    }
    target = box.Center();
    const double radius = std::max(0.5 * glm::length(box.max - box.min), kMinDistance);
    distance = std::clamp(1.1 * radius / std::sin(0.5 * verticalFov), kMinDistance, kMaxDistance);
}

// Shell

namespace {

/// Hashed alone, not on the ID stack, so it is the same outside a frame.
const ImGuiID kDockSpace = ImHashStr("ErnieDockSpace");

/// Radians the camera turns per pixel dragged.
constexpr double kOrbitRate = 0.005;

/// Seconds between refreshes of the heads-up display.
constexpr double kHudRefresh = 0.25;

constexpr std::array kMouseButtons{ImGuiMouseButton_Left, ImGuiMouseButton_Right,
                                   ImGuiMouseButton_Middle};

// Menu items as Omniverse Kit draws them: a left gutter holds a check on toggles in the accent
// colour (`ImGuiCol_TextLink`), shortcuts sit muted at the right, and menus open at least 17 lines
// wide.

/// Spaces before every label, where the markers go.
constexpr const char* kGutter = "    ";

/// @return Whether the plain item was clicked.
bool MenuAction(const char* label, const char* shortcut = nullptr, bool enabled = true) {
    return ImGui::MenuItem((std::string(kGutter) + label).c_str(), shortcut, false, enabled);
}

/// @brief A toggle, checked in the gutter when on; a click flips `on`.
void MenuCheck(const char* label, bool& on) {
    if (MenuAction(label)) {
        on = !on;
    }
    if (on) {
        const ImVec2 low = ImGui::GetItemRectMin();
        const ImVec2 high = ImGui::GetItemRectMax();
        const float size = 0.7f * ImGui::GetTextLineHeight();
        const ImVec2 centre(low.x + 0.5f * ImGui::CalcTextSize(kGutter).x, 0.5f * (low.y + high.y));
        ImGui::RenderCheckMark(ImGui::GetWindowDrawList(),
                               ImVec2(centre.x - 0.5f * size, centre.y - 0.5f * size),
                               ImGui::GetColorU32(ImGuiCol_TextLink), size);
    }
}

/// @brief A section heading inside a menu, muted, before its items.
void MenuSection(const char* label) {
    ImGui::TextDisabled("%s", label);
}

template <typename Items> void Menu(const char* name, Items&& items) {
    // A menu that stays closed leaves the constraint unused.
    ImGui::SetNextWindowSizeConstraints(ImVec2(17.0f * ImGui::GetFontSize(), 0.0f),
                                        ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::BeginMenu(name)) {
        items();
        ImGui::EndMenu();
    }
}

/// @brief Opens `target`, a URL or a folder, in the desktop's own application, in the background.
/// @return Whether that could start.
bool OpenExternally(const std::string& target) {
#if defined(_WIN32)
    const std::string command = "start \"\" \"" + target + "\"";
#elif defined(__APPLE__)
    const std::string command = "open '" + target + "' &";
#else
    const std::string command = "xdg-open '" + target + "' >/dev/null 2>&1 &";
#endif
    return std::system(command.c_str()) == 0;
}

constexpr const char* kDocumentation = "https://alexmaietta.github.io/ernie/";
constexpr const char* kRepository = "https://github.com/alexmaietta/ernie";

/// @return Whether a prim of `type` groups others: new prims go under a selected one.
bool IsGroup(std::string_view type) {
    return type == Xform::kName || type == Scope::kName;
}

/// @return A camera the scene's render settings name, for new render prims; empty for none.
Path SettingsCamera(const Scene& scene) {
    for (const Path& settings : scene.Prims(RenderSettings::kName)) {
        const Path camera = RenderSettings(scene, settings).GetCamera();
        if (!camera.IsEmpty()) {
            return camera;
        }
    }
    return {};
}

}  // namespace

Shell::Shell(Scene& scene, const DeviceContext* device, const ShellOptions& options)
    : scene_(scene), device_(device), theme_(options.theme), sizes_(options.docks) {
    // Stage above Properties on the right; Log, Info, Devices and Profiler tabbed at the bottom.
    panels_.push_back(MakeStagePanel(scene, selection_));
    panels_.push_back(MakePropertiesPanel(scene, selection_));
    panels_.push_back(MakeLogPanel(log_, theme_));
    panels_.push_back(MakeInfoPanel(scene, device));
    panels_.push_back(MakeDevicePanel(device));
    panels_.push_back(MakeProfilerPanel(log_, options.profile));
    if (options.profile) {
        profiler::SetEnabled(true);
    }
}

Panel* Shell::FindPanel(std::string_view title) const {
    const auto found = std::ranges::find(panels_, title, &Panel::Title);
    return found != panels_.end() ? found->get() : nullptr;
}

std::pair<ImVec2, ImVec2> Shell::Viewport() const {
    if (const ImGuiDockNode* centre = ImGui::DockBuilderGetCentralNode(kDockSpace)) {
        return {centre->Pos, centre->Size};
    }
    const ImGuiViewport* main = ImGui::GetMainViewport();
    return {main->WorkPos, main->WorkSize};
}

void Shell::Draw() {
    ERNIE_ZONE("Draw the UI");
    choices_ = {};
    {
        ERNIE_ZONE("Draw the menus");
        DrawMenuBar();
    }
    {
        ERNIE_ZONE("Lay out the docks");
        DrawDockSpace();
    }
    DrawPanels();
    {
        ERNIE_ZONE("Draw the windows");
        DrawControlsWindow(windows_.controls);
        DrawAboutWindow(windows_.about, device_ != nullptr ? device_->Info().name : "");
        if (windows_.demo) {
            ImGui::ShowDemoWindow(&windows_.demo);
        }
        DrawFileWindow();
    }
    TakeShortcuts();
    {
        ERNIE_ZONE("Move the camera");
        Navigate();
    }
    {
        ERNIE_ZONE("Draw the HUD");
        DrawHud();
    }
    Apply();
}

void Shell::DrawDockSpace() {
    if (!layoutBuilt_) {
        BuildLayout();
        layoutBuilt_ = true;
    }
    // The viewport shows through the central node.
    ImGui::DockSpaceOverViewport(kDockSpace, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);
}

void Shell::BuildLayout() {
    ImGui::DockBuilderRemoveNode(kDockSpace);
    ImGui::DockBuilderAddNode(kDockSpace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(kDockSpace, ImGui::GetMainViewport()->WorkSize);
    docks_.clear();
    // Hidden panels take no room; they dock where they ask once shown.
    const auto wants = [&](Dock dock) {
        return std::ranges::any_of(panels_, [dock](const std::unique_ptr<Panel>& panel) {
            return panel->Visible() && panel->DefaultDock() == dock;
        });
    };
    const auto split = [](ImGuiID& rest, ImGuiDir direction, float share) {
        return ImGui::DockBuilderSplitNode(rest, direction, std::clamp(share, 0.02f, 0.98f),
                                           nullptr, &rest);
    };
    ImGuiID centre = kDockSpace;
    if (wants(Dock::Left)) {
        docks_[Dock::Left] = split(centre, ImGuiDir_Left, sizes_.left);
    }
    // The right column: `Right` above `RightLower`, either taking all of it alone.
    const bool upper = wants(Dock::Right);
    const bool lower = wants(Dock::RightLower);
    if (upper || lower) {
        ImGuiID column = split(centre, ImGuiDir_Right, sizes_.right);
        if (upper && lower) {
            docks_[Dock::RightLower] = split(column, ImGuiDir_Down, sizes_.rightLower);
        }
        docks_[upper ? Dock::Right : Dock::RightLower] = column;
    }
    if (wants(Dock::Bottom)) {
        docks_[Dock::Bottom] = split(centre, ImGuiDir_Down, sizes_.bottom);
    }
    for (const std::unique_ptr<Panel>& panel : panels_) {
        const auto node = docks_.find(panel->DefaultDock());
        if (panel->Visible() && node != docks_.end()) {
            ImGui::DockBuilderDockWindow(panel->Title().c_str(), node->second);
        }
    }
    ImGui::DockBuilderFinish(kDockSpace);
    // A dock opens on the last window docked into it; the bottom one opens on the Log instead.
    const Panel* log = FindPanel("Log");
    selectLog_ = log != nullptr && log->Visible() && log->DefaultDock() == Dock::Bottom &&
                 docks_.contains(Dock::Bottom);
}

void Shell::SelectLogTab() {
    // A dock shows the focused window's tab, and ImGui focuses each window as it is made: once
    // every panel has been, the Log takes the focus.
    if (selectLog_ && ImGui::FindWindowByName("Log") != nullptr) {
        ImGui::SetWindowFocus("Log");
        selectLog_ = false;
    }
}

void Shell::DrawPanels() {
    ERNIE_ZONE("Draw the panels");
    for (const std::unique_ptr<Panel>& panel : panels_) {
        ERNIE_ZONE("Draw a panel");
        ERNIE_ZONE_TEXT(panel->Title());
        if (!panel->Visible()) {
            panel->Undrawn();
            continue;
        }
        if (const auto node = docks_.find(panel->DefaultDock()); node != docks_.end()) {
            ImGui::SetNextWindowDockID(node->second, ImGuiCond_FirstUseEver);
        }
        bool keep = true;
        if (ImGui::Begin(panel->Title().c_str(), &keep)) {
            panel->Draw();
        } else {
            panel->Undrawn();
        }
        ImGui::End();
        panel->SetVisible(keep);
    }
    SelectLogTab();
}

void Shell::Navigate() {
    const ImGuiIO& io = ImGui::GetIO();
    const auto [origin, size] = Viewport();
    // Over the viewport and no window: the dock space lets the mouse through its central node.
    const bool hovered = !io.WantCaptureMouse && io.MousePos.x >= origin.x &&
                         io.MousePos.y >= origin.y && io.MousePos.x < origin.x + size.x &&
                         io.MousePos.y < origin.y + size.y;
    const auto any = [](auto&& test) { return std::ranges::any_of(kMouseButtons, test); };
    if (!any([](ImGuiMouseButton button) { return ImGui::IsMouseDown(button); })) {
        dragging_ = false;
    } else if (hovered &&
               any([](ImGuiMouseButton button) { return ImGui::IsMouseClicked(button); })) {
        dragging_ = true;
    }
    const double dx = io.MouseDelta.x;
    const double dy = io.MouseDelta.y;
    if (dragging_ && (dx != 0.0 || dy != 0.0)) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            camera_.Orbit(-dx * kOrbitRate, dy * kOrbitRate);
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            camera_.Pan(dx, dy, size.y);
        } else {
            camera_.PanVertical(dx, dy, size.y);
        }
    }
    if (hovered && io.MouseWheel != 0.0f) {
        camera_.Zoom(io.MouseWheel);
    }
}

void Shell::DrawHud() {
    const auto [origin, size] = Viewport();
    const double now = ImGui::GetTime();
    if (hudRefreshed_ < 0.0 || now - hudRefreshed_ >= kHudRefresh) {
        const double fps = ImGui::GetIO().Framerate;
        const glm::dvec3 eye = camera_.Eye();
        const glm::dvec3& target = camera_.target;
        std::array<char, 256> text{};
        std::snprintf(text.data(), text.size(),
                      "%.0f fps, %.2f ms\n%.0f x %.0f\nEye %.3g, %.3g, %.3g m\n"
                      "Target %.3g, %.3g, %.3g m",
                      fps, fps > 0.0 ? 1000.0 / fps : 0.0, static_cast<double>(size.x),
                      static_cast<double>(size.y), eye.x, eye.y, eye.z, target.x, target.y,
                      target.z);
        hud_ = text.data();
        hudRefreshed_ = now;
    }
    // Behind every window.
    const ImVec2 padding = ImGui::GetStyle().WindowPadding;
    ImGui::GetBackgroundDrawList()->AddText(ImVec2(origin.x + padding.x, origin.y + padding.y),
                                            ImGui::GetColorU32(ToImVec4(theme_.hud)), hud_.c_str());
}

void Shell::Frame() {
    const std::vector<Path> framed =
        selection_.Empty() ? scene_.Children(Path::Root()) : selection_.Paths();
    camera_.Frame(WorldBounds(scene_, framed));
}

template <typename Edit> void Shell::Attempt(const char* what, Edit&& edit) {
    try {
        edit();
    } catch (const std::exception& error) {
        log_.Error(std::string(what) + ": " + error.what());
    }
}

void Shell::DrawMenuBar() {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }
    const bool selected = !selection_.Empty();
    Menu("File", [&] {
        windows_.open |= MenuAction("Open Scene...", "Ctrl+O");
        windows_.save |= MenuAction("Save Scene As...", "Ctrl+Shift+S");
        ImGui::Separator();
        choices_.quit |= MenuAction("Quit", "Ctrl+Q");
    });
    Menu("Edit", [&] {
        const std::string undo = history_.CanUndo() ? "Undo " + history_.UndoLabel() : "Undo";
        const std::string redo = history_.CanRedo() ? "Redo " + history_.RedoLabel() : "Redo";
        choices_.undo |= MenuAction(undo.c_str(), "Ctrl+Z", history_.CanUndo());
        choices_.redo |= MenuAction(redo.c_str(), "Ctrl+Shift+Z", history_.CanRedo());
        ImGui::Separator();
        choices_.duplicate |= MenuAction("Duplicate", "Ctrl+D", selected);
        choices_.remove |= MenuAction("Delete", "Del", selected);
        ImGui::Separator();
        choices_.frame |= MenuAction(selected ? "Frame Selection" : "Frame Scene", "F");
        choices_.clearSelection |= MenuAction("Clear Selection", nullptr, selected);
    });
    Menu("Create", [&] { DrawCreateMenu(); });
    Menu("Window", [&] {
        for (const std::unique_ptr<Panel>& panel : panels_) {
            bool visible = panel->Visible();
            MenuCheck(panel->Title().c_str(), visible);
            panel->SetVisible(visible);
        }
        ImGui::Separator();
        choices_.resetLayout |= MenuAction("Reset Layout");
        MenuCheck("ImGui Demo", windows_.demo);
    });
    Menu("Help", [&] { DrawHelpMenu(); });
    ImGui::EndMainMenuBar();
}

void Shell::DrawCreateMenu() {
    std::string_view group;
    for (const CreatableType& creatable : CreatableTypes()) {
        if (group != creatable.group) {
            if (!group.empty()) {
                ImGui::Separator();
            }
            group = creatable.group;
            MenuSection(creatable.group);
        }
        if (MenuAction(creatable.label)) {
            choices_.create = creatable.type;
        }
    }
}

void Shell::DrawHelpMenu() {
    const std::string docs = kDocumentation;
    const std::array<std::pair<const char*, std::string>, 5> links{{
        {"Documentation", docs},
        {"Viewer Guide", docs + "getting-started/viewer/"},
        {"Demos", docs + "getting-started/demos/"},
        {"Python API", docs + "api/python/scene/"},
        {"C++ API", docs + "api/cpp/"},
    }};
    const auto open = [&](const std::string& url) {
        if (!OpenExternally(url)) {
            log_.Error("Could not open " + url);
        }
    };
    for (const auto& [label, url] : links) {
        if (MenuAction(label)) {
            open(url);
        }
    }
    ImGui::Separator();
    if (MenuAction("Report an Issue")) {
        open(std::string(kRepository) + "/issues");
    }
    if (MenuAction("Source Code")) {
        open(kRepository);
    }
    ImGui::Separator();
    MenuCheck("Keyboard and Mouse", windows_.controls);
    MenuCheck("About Ernie", windows_.about);
}

void Shell::DrawFileWindow() {
    if (!windows_.open && !windows_.save) {
        return;
    }
    const char* title = windows_.open ? "Open Scene" : "Save Scene As";
    bool shown = true;
    ImGui::SetNextWindowSize(ImVec2(32.0f * ImGui::GetFontSize(), 0.0f), ImGuiCond_Appearing);
    if (ImGui::Begin(title, &shown, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking)) {
        ImGui::TextUnformatted("Scene file (.json):");
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        const bool entered =
            ImGui::InputText("##path", &filePath_, ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::Button(windows_.open ? "Open" : "Save") || entered) {
            if (windows_.open) {
                Attempt("Opening a scene", [&] { OpenScene(filePath_); });
            } else {
                Attempt("Saving the scene", [&] { SaveScene(filePath_); });
            }
            shown = false;
        }
        ImGui::SameLine();
        shown &= !ImGui::Button("Cancel");
    }
    ImGui::End();
    if (!shown) {
        windows_.open = windows_.save = false;
    }
}

void Shell::TakeShortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) {
        return;
    }
    const auto pressed = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, false); };
    const bool plain = !io.KeyCtrl && !io.KeyAlt;
    choices_.quit |= io.KeyCtrl && pressed(ImGuiKey_Q);
    choices_.frame |= plain && pressed(ImGuiKey_F);
    choices_.undo |= io.KeyCtrl && !io.KeyShift && pressed(ImGuiKey_Z);
    choices_.redo |= io.KeyCtrl && ((io.KeyShift && pressed(ImGuiKey_Z)) || pressed(ImGuiKey_Y));
    choices_.duplicate |= io.KeyCtrl && pressed(ImGuiKey_D);
    choices_.remove |= plain && pressed(ImGuiKey_Delete);
    windows_.open |= io.KeyCtrl && pressed(ImGuiKey_O);
    windows_.save |= io.KeyCtrl && io.KeyShift && pressed(ImGuiKey_S);
}

void Shell::Apply() {
    open_ = open_ && !choices_.quit;
    layoutBuilt_ = layoutBuilt_ && !choices_.resetLayout;
    if (choices_.clearSelection) {
        selection_.Clear();
    }
    if (choices_.undo) {
        Attempt("Undoing", [&] { history_.Undo(scene_); });
    }
    if (choices_.redo) {
        Attempt("Redoing", [&] { history_.Redo(scene_); });
    }
    if (choices_.duplicate) {
        Attempt("Duplicating", [&] { DuplicateSelection(); });
    }
    if (choices_.remove) {
        Attempt("Deleting", [&] { DeleteSelection(); });
    }
    if (!choices_.create.empty()) {
        Attempt("Creating", [&] { Create(choices_.create); });
    }
    // Undone prims leave the selection.
    std::vector<Path> kept;
    for (const Path& path : selection_.Paths()) {
        if (scene_.Find(path)) {
            kept.push_back(path);
        }
    }
    if (kept.size() != selection_.Paths().size()) {
        selection_.Clear();
        for (const Path& path : kept) {
            selection_.Toggle(path);
        }
    }
    if (choices_.frame) {
        Frame();
    }
}

Path Shell::Create(std::string_view type) {
    const CreatableType* creatable = FindCreatable(type);
    if (creatable == nullptr) {
        throw std::invalid_argument("The viewer cannot create a '" + std::string(type) + "'");
    }
    Path parent(creatable->parent);
    const std::optional<PrimId> selected =
        selection_.Empty() ? std::nullopt : scene_.Find(selection_.Paths().front());
    if (selected && IsGroup(scene_.TypeName(*selected))) {
        parent = selection_.Paths().front();
    }
    const Path path = FreeChildPath(scene_, parent, creatable->name);
    const std::optional<PrimId> parentPrim = scene_.Find(parent);
    // Authored in a scratch scene, then copied in as one step with any missing parents.
    Scene scratch;
    const CreateContext context{.target = camera_.target,
                                .scale = std::max(0.05 * camera_.distance, 1e-3),
                                .parentWorld = parentPrim ? scene_.WorldTransform(*parentPrim)
                                                          : glm::dmat4(1.0),
                                .view = camera_.Transform(),
                                .verticalFov = camera_.verticalFov,
                                .camera = SettingsCamera(scene_)};
    AuthorDefaultPrim(scratch, path, type, context);
    // Missing ancestors become Xforms, outermost first, in the same step.
    std::vector<Path> missing;
    for (Path at = parent; !at.IsRoot() && !scene_.Find(at); at = at.Parent()) {
        missing.insert(missing.begin(), at);
    }
    history_.BeginGroup();
    for (const Path& at : missing) {
        history_.Define(scene_, at, Xform::kName);
    }
    history_.Add(scene_, scratch, path);
    history_.EndGroup();
    selection_.Set(path);
    log_.Info("Created " + path.String());
    return path;
}

void Shell::DuplicateSelection() {
    std::vector<Path> copies;
    history_.BeginGroup();
    for (const Path& path : selection_.Paths()) {
        if (!scene_.Find(path)) {
            continue;
        }
        const Path copy = FreeChildPath(scene_, path.Parent(), path.Name());
        history_.Copy(scene_, path, copy);
        copies.push_back(copy);
    }
    history_.EndGroup();
    selection_.Clear();
    for (const Path& copy : copies) {
        selection_.Toggle(copy);
    }
}

void Shell::DeleteSelection() {
    history_.BeginGroup();
    for (const Path& path : selection_.Paths()) {
        history_.Remove(scene_, path);
    }
    history_.EndGroup();
    selection_.Clear();
}

void Shell::OpenScene(const std::filesystem::path& path) {
    // Read once aside, so a bad file leaves the scene as it was.
    Scene checked;
    LoadSceneFile(checked, path);
    for (const Path& root : scene_.Children(Path::Root())) {
        scene_.Remove(root);
    }
    LoadSceneFile(scene_, path);
    history_.Reset();
    selection_.Clear();
    log_.Info("Opened " + path.string());
    Frame();
}

void Shell::SaveScene(const std::filesystem::path& path) {
    SaveSceneFile(scene_, path);
    log_.Info("Saved " + path.string());
}

// Presenters

OffscreenPresenter::OffscreenPresenter(DeviceContext& context, glm::uvec2 extent)
    : image_(context,
             {.format = ImageFormat::Rgba8Unorm,
              .extent = {extent, 1},
              .usage = ImageUsage::Attachment | ImageUsage::Sampled},
             "Viewer frame"),
      drawn_(image_.Ready()) {}

void OffscreenPresenter::BeginUiFrame(double delta) {
    ImGuiIO& io = ImGui::GetIO();
    const glm::uvec3 extent = image_.Description().extent;
    io.DisplaySize = {static_cast<float>(extent.x), static_cast<float>(extent.y)};
    io.DeltaTime = static_cast<float>(std::max(delta, 1e-4));
}

#if ERNIE_VIEWER_WINDOW
namespace {

int glfwUsers = 0;

/// @return Whether GLFW is initialised, initialising it for one more user.
bool AcquireGlfw() {
    if (glfwUsers == 0) {
        glfwSetErrorCallback([](int code, const char* message) {
            log::Detail("GLFW error " + std::to_string(code) + ": " + message);
        });
        // GLFW finds Vulkan through the loader volk opened, as the device layer does.
        if (volkInitialize() != VK_SUCCESS) {
            return false;
        }
        glfwInitVulkanLoader(vkGetInstanceProcAddr);
        if (glfwInit() == GLFW_FALSE) {
            return false;
        }
    }
    ++glfwUsers;
    return true;
}

void ReleaseGlfw() {
    if (--glfwUsers == 0) {
        glfwTerminate();
    }
}

template <typename T, typename Query> std::vector<T> Enumerate(Query&& query) {
    std::uint32_t count = 0;
    query(&count, nullptr);
    std::vector<T> items(count);
    query(&count, items.data());
    return items;
}

VkSurfaceKHR CreateSurface(VkInstance instance, GLFWwindow* window) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    CheckVk(glfwCreateWindowSurface(instance, window, nullptr, &surface),
            "glfwCreateWindowSurface");
    return surface;
}

/// @return An 8-bit UNORM format: the UI's colours are sRGB-encoded already.
VkSurfaceFormatKHR ChooseFormat(VkPhysicalDevice physical, VkSurfaceKHR surface) {
    const std::vector<VkSurfaceFormatKHR> formats =
        Enumerate<VkSurfaceFormatKHR>([&](std::uint32_t* count, VkSurfaceFormatKHR* items) {
            vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, items);
        });
    if (formats.empty()) {
        throw DeviceError(DeviceErrorKind::Unsupported, "The window's surface has no formats");
    }
    for (const VkFormat wanted : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
        const auto found = std::ranges::find_if(formats, [&](const VkSurfaceFormatKHR& format) {
            return format.format == wanted &&
                   format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        if (found != formats.end()) {
            return *found;
        }
    }
    return formats.front();
}

VkPresentModeKHR ChoosePresentMode(VkPhysicalDevice physical, VkSurfaceKHR surface, bool vsync) {
    if (vsync) {
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    const std::vector<VkPresentModeKHR> modes =
        Enumerate<VkPresentModeKHR>([&](std::uint32_t* count, VkPresentModeKHR* items) {
            vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, count, items);
        });
    for (const VkPresentModeKHR wanted :
         {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
        if (std::ranges::find(modes, wanted) != modes.end()) {
            return wanted;
        }
    }
    // The one mode every surface supports.
    return VK_PRESENT_MODE_FIFO_KHR;
}

/// @return Whether the swapchain still matches the surface after an acquire or present.
bool Matches(VkResult result) {
    return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
}

}  // namespace

bool Window::Available() {
    if (!AcquireGlfw()) {
        return false;
    }
    const bool supported = glfwVulkanSupported() == GLFW_TRUE;
    ReleaseGlfw();
    return supported;
}

std::vector<std::string> Window::InstanceExtensions() {
    if (!AcquireGlfw()) {
        return {};
    }
    std::uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string> extensions(names, names + (names != nullptr ? count : 0));
    ReleaseGlfw();
    return extensions;
}

Window::Window(const std::string& title, glm::uvec2 size) {
    if (!AcquireGlfw()) {
        throw std::runtime_error("No display to open a window on; --offscreen draws without one");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(static_cast<int>(size.x), static_cast<int>(size.y), title.c_str(),
                               nullptr, nullptr);
    if (window_ == nullptr) {
        ReleaseGlfw();
        throw std::runtime_error("Could not open a window");
    }
}

Window::~Window() {
    glfwDestroyWindow(window_);
    ReleaseGlfw();
}

glm::uvec2 Window::FramebufferSize() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return {static_cast<std::uint32_t>(std::max(width, 0)),
            static_cast<std::uint32_t>(std::max(height, 0))};
}

float Window::ContentScale() const {
    float x = 1.0f;
    float y = 1.0f;
    glfwGetWindowContentScale(window_, &x, &y);
    return std::max(x, 1.0f);
}

WindowPresenter::WindowPresenter(DeviceContext& context, std::unique_ptr<Window> window, bool vsync)
    : context_(context), window_(std::move(window)),
      surface_(CreateSurface(context.Instance(), window_->Handle())), vsync_(vsync) {
    const OnFailure destroySurface(
        [&] { vkDestroySurfaceKHR(context_.Instance(), surface_, nullptr); });
    if (!context.Capabilities().present) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          "The device cannot present to a window; --offscreen draws without one");
    }
    VkBool32 presents = VK_FALSE;
    CheckVk(vkGetPhysicalDeviceSurfaceSupportKHR(context.Physical(), context.Main().Family(),
                                                 surface_, &presents),
            "vkGetPhysicalDeviceSurfaceSupportKHR");
    if (presents == VK_FALSE) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          "The device's main queue cannot present to the window");
    }
    format_ = ChooseFormat(context.Physical(), surface_);
    const OnFailure destroyHandles([&] {
        DestroyFrames();
        vkDestroySwapchainKHR(context_.Device(), swapchain_, nullptr);
    });
    CreateSwapchain(window_->FramebufferSize());
    ImGui_ImplGlfw_InitForVulkan(window_->Handle(), true);
}

WindowPresenter::~WindowPresenter() {
    ImGui_ImplGlfw_Shutdown();
    DestroyFrames();
    vkDestroySwapchainKHR(context_.Device(), swapchain_, nullptr);
    vkDestroySurfaceKHR(context_.Instance(), surface_, nullptr);
}

void WindowPresenter::CreateSwapchain(glm::uvec2 extent) {
    ERNIE_ZONE("Create the swapchain");
    VkSurfaceCapabilitiesKHR capabilities{};
    CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context_.Physical(), surface_, &capabilities),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        extent_ = capabilities.currentExtent;
    } else {
        extent_ = {std::clamp(extent.x, capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width),
                   std::clamp(extent.y, capabilities.minImageExtent.height,
                              capabilities.maxImageExtent.height)};
    }
    std::uint32_t count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        count = std::min(count, capabilities.maxImageCount);
    }
    const VkSwapchainKHR old = swapchain_;
    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface_,
        .minImageCount = count,
        .imageFormat = format_.format,
        .imageColorSpace = format_.colorSpace,
        .imageExtent = extent_,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = ChoosePresentMode(context_.Physical(), surface_, vsync_),
        .clipped = VK_TRUE,
        .oldSwapchain = old};
    CheckVk(vkCreateSwapchainKHR(context_.Device(), &info, nullptr, &swapchain_),
            "vkCreateSwapchainKHR");
    if (old != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(context_.Device(), old, nullptr);
    }
    images_ = Enumerate<VkImage>([&](std::uint32_t* n, VkImage* items) {
        vkGetSwapchainImagesKHR(context_.Device(), swapchain_, n, items);
    });
    for (const VkImage image : images_) {
        const VkImageViewCreateInfo view{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format_.format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        CheckVk(vkCreateImageView(context_.Device(), &view, nullptr, &views_.emplace_back()),
                "vkCreateImageView");
    }
    const VkSemaphoreCreateInfo semaphore{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    const auto create = [&](std::vector<VkSemaphore>& semaphores, std::size_t wanted) {
        while (semaphores.size() < wanted) {
            CheckVk(vkCreateSemaphore(context_.Device(), &semaphore, nullptr,
                                      &semaphores.emplace_back()),
                    "vkCreateSemaphore");
        }
    };
    create(acquiredSemaphores_, images_.size() + 1);
    create(drawnSemaphores_, images_.size());
    nextAcquired_ = 0;
}

void WindowPresenter::DestroyFrames() {
    // The queue's idle once its presents have waited on their semaphores too.
    Quietly([this] { context_.Main().Borrow([](VkQueue queue) { vkQueueWaitIdle(queue); }); });
    for (const VkImageView view : views_) {
        vkDestroyImageView(context_.Device(), view, nullptr);
    }
    for (const VkSemaphore semaphore : acquiredSemaphores_) {
        vkDestroySemaphore(context_.Device(), semaphore, nullptr);
    }
    for (const VkSemaphore semaphore : drawnSemaphores_) {
        vkDestroySemaphore(context_.Device(), semaphore, nullptr);
    }
    views_.clear();
    acquiredSemaphores_.clear();
    drawnSemaphores_.clear();
}

void WindowPresenter::PollEvents() {
    glfwPollEvents();
}

bool WindowPresenter::CloseRequested() const {
    return glfwWindowShouldClose(window_->Handle()) == GLFW_TRUE;
}

void WindowPresenter::BeginUiFrame(double /*delta*/) {
    // The backend measures time steps and sizes itself.
    ImGui_ImplGlfw_NewFrame();
}

std::optional<FrameTarget> WindowPresenter::Acquire() {
    const glm::uvec2 size = window_->FramebufferSize();
    if (size.x == 0 || size.y == 0) {
        return std::nullopt;
    }
    const glm::uvec2 extent(extent_.width, extent_.height);
    if (stale_ || size != extent) {
        ERNIE_ZONE("Resize the swapchain");
        DestroyFrames();
        CreateSwapchain(size);
        stale_ = false;
    }
    // Blocks only while every image is queued for the display, as vsync paces frames; the
    // device waits for the image itself.
    ERNIE_ZONE("Wait for a swapchain image");
    const VkSemaphore acquired = acquiredSemaphores_[nextAcquired_];
    std::uint32_t index = 0;
    const VkResult result = vkAcquireNextImageKHR(context_.Device(), swapchain_,
                                                  std::numeric_limits<std::uint64_t>::max(),
                                                  acquired, VK_NULL_HANDLE, &index);
    if (!Matches(result)) {
        if (result != VK_ERROR_OUT_OF_DATE_KHR) {
            CheckVk(result, "vkAcquireNextImageKHR");
        }
        stale_ = true;
        return std::nullopt;
    }
    nextAcquired_ = (nextAcquired_ + 1) % acquiredSemaphores_.size();
    acquired_ = index;
    return FrameTarget{images_[index],
                       views_[index],
                       glm::uvec2(extent_.width, extent_.height),
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                       {acquired, drawnSemaphores_[index]}};
}

void WindowPresenter::Present(Ticket /*drawn*/) {
    if (!acquired_) {
        return;
    }
    // The device waits for the frame's drawing; the host goes on to the next.
    const VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                .waitSemaphoreCount = 1,
                                .pWaitSemaphores = &drawnSemaphores_[*acquired_],
                                .swapchainCount = 1,
                                .pSwapchains = &swapchain_,
                                .pImageIndices = &*acquired_};
    const VkResult result = context_.Main().Present(info);
    if (!Matches(result) && result != VK_ERROR_OUT_OF_DATE_KHR) {
        CheckVk(result, "vkQueuePresentKHR");
    }
    stale_ = result != VK_SUCCESS;
    acquired_.reset();
}
#endif

}  // namespace ernie::app
