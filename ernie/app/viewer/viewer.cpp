// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <nlohmann/json.hpp>

#include "ernie/app/cli.h"
#include "ernie/app/viewer/panels/panels.h"
#include "ernie/app/viewer/presenter/presenter.h"
#include "ernie/app/viewer/shell/editor.h"
#include "ernie/app/viewer/shell/history.h"
#include "ernie/app/viewer/theme/theme.h"
#include "ernie/app/viewer/viewport/orbit_camera.h"
#include "ernie/app/viewer/viewport/viewport.h"
#include "ernie/app/viewer/widgets/widgets.h"
#include "ernie/ernie.h"
#include "ernie/io/image.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

extern "C" {
extern const unsigned char kErnieFont_inter[];
extern const std::size_t kErnieFont_inter_size;
extern const unsigned char kErnieFont_lucide[];
extern const std::size_t kErnieFont_lucide_size;
}

namespace ernie::app {
namespace {

constexpr const char* kBuiltinMenus[] = {"File", "Edit", "Create", "Window", "Help"};
constexpr int kLayoutVersion = 1;
constexpr const char* kLightingNames[] = {"stage", "camera", "default"};
constexpr const char* kWorldPickNames[] = {"all", "first", "random", "listed", "selected"};
constexpr const char* kWorldLayoutNames[] = {"grid", "overlap"};

template <size_t N> size_t NameIndex(const char* const (&names)[N], const nlohmann::json& value) {
    const auto found = std::ranges::find(names, value.get<std::string>());
    if (found == std::end(names)) {
        throw Error("A layout has no choice {}", value.dump());
    }
    return size_t(found - std::begin(names));
}

ImGuiKeyChord ParseChord(std::string_view text) {
    const auto same = [](std::string_view a, std::string_view b) {
        return std::ranges::equal(a, b, [](char x, char y) {
            return std::tolower(static_cast<unsigned char>(x)) ==
                   std::tolower(static_cast<unsigned char>(y));
        });
    };
    ImGuiKeyChord chord = 0;
    while (!text.empty()) {
        const size_t plus = text.find('+', 1);
        const std::string_view part = text.substr(0, plus);
        text = plus == std::string_view::npos ? std::string_view() : text.substr(plus + 1);
        if (same(part, "Ctrl") || same(part, "Control")) {
            chord |= ImGuiMod_Ctrl;
        } else if (same(part, "Shift")) {
            chord |= ImGuiMod_Shift;
        } else if (same(part, "Alt")) {
            chord |= ImGuiMod_Alt;
        } else if (same(part, "Super") || same(part, "Cmd")) {
            chord |= ImGuiMod_Super;
        } else {
            const std::string_view name = same(part, "Del")   ? "Delete"
                                          : same(part, "Esc") ? "Escape"
                                                              : part;
            ImGuiKey found = ImGuiKey_None;
            for (int key = ImGuiKey_NamedKey_BEGIN; key < ImGuiKey_NamedKey_END; ++key) {
                if (same(ImGui::GetKeyName(ImGuiKey(key)), name)) {
                    found = ImGuiKey(key);
                    break;
                }
            }
            if (found == ImGuiKey_None || !text.empty()) {
                return 0;
            }
            chord |= found;
        }
    }
    return chord;
}

void LoadFonts(const Theme& theme) {
    static const ImWchar kIcons[] = {icons::kFirst, icons::kLast, 0};
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig text;
    text.FontDataOwnedByAtlas = false;
    text.GlyphExcludeRanges = kIcons;
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(kErnieFont_inter),
                                   int(kErnieFont_inter_size), theme.fontSize, &text);
    ImFontConfig glyphs;
    glyphs.FontDataOwnedByAtlas = false;
    glyphs.MergeMode = true;
    glyphs.GlyphMinAdvanceX = theme.iconSize;
    glyphs.GlyphOffset = {0, 2};
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(kErnieFont_lucide),
                                   int(kErnieFont_lucide_size), theme.iconSize, &glyphs, kIcons);
}

Ref<device::Device> DisplayDevice(device::Device& render) {
    if (render.Adapter().backend == device::Backend::Vulkan && render.Presents()) {
        return Ref<device::Device>(&render);
    }
    for (const device::AdapterInfo& adapter : device::EnumerateAdapters()) {
        if (adapter.backend == device::Backend::Vulkan && adapter.unsupported.empty() &&
            adapter.features.presentation && adapter.uuid == render.Adapter().uuid) {
            return device::Device::Create({.adapter = adapter.alias, .presentation = true});
        }
    }
    return device::Device::Create({.backend = device::Backend::Vulkan, .presentation = true});
}

std::unique_ptr<Presenter> CreatePresenter(device::Device& device, GLFWwindow* window, Extent size,
                                           bool vsync) {
    if (device.Adapter().backend == device::Backend::Vulkan) {
        return CreateVulkanPresenter(device, window, size, vsync);
    }
    throw UnsupportedError("The viewer can't present from {} devices yet",
                           device::BackendName(device.Adapter().backend));
}

}

class ViewerImpl {
  public:
    ViewerImpl(Ref<device::Device> device, Ref<Scene> scene, const ViewerDesc& desc);
    ~ViewerImpl();
    ViewerImpl(const ViewerImpl&) = delete;
    ViewerImpl& operator=(const ViewerImpl&) = delete;

    bool Open() const;
    void Frame();

    std::vector<uint8_t> ReadPixels() {
        return presenter->ReadPixels(m_lastDraw);
    }

    void AddPanel(Panel panel) {
        panels.push_back(std::move(panel));
        m_layoutBuilt = false;
    }

    void ShowPanel(std::string_view title, bool shown) {
        for (Panel& panel : panels) {
            if (panel.title == title) {
                panel.visible = shown;
                return;
            }
        }
        throw Error("The viewer has no panel \"{}\"", title);
    }

    void Inspect(const std::string& path) {
        m_selection.Set(path);
        for (Panel& panel : panels) {
            if (panel.title == "Stage" || panel.title == "Properties") {
                panel.visible = true;
                ImGui::SetWindowFocus(panel.title.c_str());
            }
        }
    }

    void AddViewport(ViewportDesc desc, bool relayout = true) {
        if (desc.name.empty()) {
            throw Error("A viewport needs a name");
        }
        if (FindViewport(desc.name) != nullptr ||
            std::ranges::any_of(panels,
                                [&](const Panel& panel) { return panel.title == desc.name; })) {
            throw Error("The viewer already has a window named \"{}\"", desc.name);
        }
        m_viewports.push_back(std::make_unique<Viewport>(
            std::move(desc), m_scene, m_device, m_selection, m_theme,
            [this](const std::string& path) { Inspect(path); }, &m_editor, &m_sessions));
        ConnectOverlayEvents();
        m_layoutBuilt = m_layoutBuilt && !relayout;
    }

    void AddViewportBeside() {
        const Viewport* active = Active();
        std::string name;
        for (int number = 2; name.empty() || FindViewport(name) != nullptr; ++number) {
            name = fmt::format("Viewport {}", number);
        }
        const ImGuiWindow* window =
            active != nullptr ? ImGui::FindWindowByName(active->Name().c_str()) : nullptr;
        AddViewport({.name = name,
                     .product = active != nullptr ? active->Product() : m_desc.product,
                     .renderer = active != nullptr ? active->GetRenderer() : nullptr},
                    window == nullptr || window->DockId == 0);
        if (window != nullptr && window->DockId != 0) {
            ImGui::DockBuilderDockWindow(name.c_str(), window->DockId);
        }
    }

    nlohmann::json SaveLayout() const {
        nlohmann::json panelsShown = nlohmann::json::object();
        for (const Panel& panel : panels) {
            panelsShown[panel.title] = panel.visible;
        }
        nlohmann::json viewports = nlohmann::json::array();
        for (const auto& viewport : m_viewports) {
            nlohmann::json hud = nlohmann::json::array();
            for (size_t at = 0; at < viewport->hudItems.size(); ++at) {
                if (viewport->hudItems[at]) {
                    hud.push_back(kHudItemNames[at]);
                }
            }
            nlohmann::json shown = nullptr;
            if (viewport->ShowsWorlds()) {
                const WorldDisplay& worlds = viewport->Worlds();
                shown = {{"pick", kWorldPickNames[size_t(worlds.pick)]},
                         {"count", worlds.count},
                         {"seed", worlds.seed},
                         {"worlds", worlds.worlds},
                         {"global", worlds.global},
                         {"layout", kWorldLayoutNames[size_t(worlds.layout)]}};
                shown["spacing"] =
                    worlds.spacing ? nlohmann::json::array(
                                         {worlds.spacing->x, worlds.spacing->y, worlds.spacing->z})
                                   : nlohmann::json(nullptr);
            }
            viewports.push_back({{"name", viewport->Name()},
                                 {"lighting", kLightingNames[size_t(viewport->Lighting())]},
                                 {"worlds", shown},
                                 {"product", viewport->Product()},
                                 {"aov", viewport->Aov()},
                                 {"camera", viewport->CameraPath()},
                                 {"visible", viewport->visible},
                                 {"toolbar", viewport->toolbarShown},
                                 {"hud", viewport->hudShown},
                                 {"hudItems", hud}});
        }
        nlohmann::json layout = {
            {"version", kLayoutVersion}, {"gui", guiShown},
            {"menuBar", menuBarShown},   {"panels", panelsShown},
            {"viewports", viewports},    {"imgui", ImGui::SaveIniSettingsToMemory()}};
        if (m_window != nullptr) {
            int width = 0;
            int height = 0;
            glfwGetWindowSize(m_window, &width, &height);
            layout["window"] = {{"width", width}, {"height", height}, {"fullscreen", Fullscreen()}};
        }
        return layout;
    }

    void SaveLayoutFile(const std::filesystem::path& path) const {
        std::ofstream file(path);
        if (!file) {
            throw Error("Can't write the layout to {}", path.string());
        }
        file << SaveLayout().dump(2) << "\n";
    }

    void LoadLayoutFile(const std::filesystem::path& path) {
        std::ifstream file(path);
        if (!file) {
            throw Error("Can't read the layout {}", path.string());
        }
        const nlohmann::json layout = nlohmann::json::parse(file, nullptr, false);
        if (layout.is_discarded() || !layout.is_object()) {
            throw Error("{} isn't a viewer layout", path.string());
        }
        LoadLayout(layout);
    }

    void LoadLayout(const nlohmann::json& layout) {
        if (layout.value("version", 0) != kLayoutVersion) {
            throw Error("Viewer layouts are version {}, not {}", kLayoutVersion,
                        layout.value("version", 0));
        }
        guiShown = layout.value("gui", guiShown);
        menuBarShown = layout.value("menuBar", menuBarShown);
        if (const auto window = layout.find("window"); window != layout.end() && m_window) {
            SetWindowSize(window->value("width", 1280u), window->value("height", 720u));
            EnableFullscreen(window->value("fullscreen", false));
        }
        if (const auto shown = layout.find("panels"); shown != layout.end()) {
            for (Panel& panel : panels) {
                panel.visible = shown->value(panel.title, panel.visible);
            }
        }
        if (const auto viewports = layout.find("viewports");
            viewports != layout.end() && !viewports->empty()) {
            std::set<std::string> named;
            for (const nlohmann::json& entry : *viewports) {
                named.insert(entry.at("name").get<std::string>());
            }
            std::vector<std::string> unnamed;
            for (const auto& viewport : m_viewports) {
                if (!named.contains(viewport->Name())) {
                    unnamed.push_back(viewport->Name());
                }
            }
            for (const std::string& name : unnamed) {
                RemoveViewport(name);
            }
            const std::shared_ptr<Renderer> renderer =
                m_viewports.empty() ? nullptr : m_viewports.front()->GetRenderer();
            for (const nlohmann::json& entry : *viewports) {
                const std::string name = entry.at("name").get<std::string>();
                if (FindViewport(name) == nullptr) {
                    AddViewport({.name = name,
                                 .product = entry.value("product", std::string()),
                                 .aov = entry.value("aov", std::string("LdrColor")),
                                 .renderer = renderer});
                }
                Viewport& viewport = *FindViewport(name);
                viewport.visible = entry.value("visible", true);
                viewport.toolbarShown = entry.value("toolbar", true);
                viewport.hudShown = entry.value("hud", true);
                if (const auto hud = entry.find("hudItems"); hud != entry.end()) {
                    for (size_t at = 0; at < viewport.hudItems.size(); ++at) {
                        viewport.hudItems[at] =
                            std::ranges::find(*hud, kHudItemNames[at]) != hud->end();
                    }
                }
                const std::string aov = entry.value("aov", std::string());
                if (!aov.empty() && !viewport.Aov().empty()) {
                    viewport.SetAov(aov);
                }
                if (const auto shown = entry.find("worlds");
                    shown != entry.end() && shown->is_object() && viewport.ShowsWorlds()) {
                    WorldDisplay worlds;
                    worlds.pick = WorldPick(
                        NameIndex(kWorldPickNames, shown->value("pick", nlohmann::json("all"))));
                    worlds.count = shown->value("count", 0u);
                    worlds.seed = shown->value("seed", 0u);
                    worlds.worlds = shown->value("worlds", std::vector<std::string>());
                    worlds.global = shown->value("global", true);
                    worlds.layout = WorldLayout(NameIndex(
                        kWorldLayoutNames, shown->value("layout", nlohmann::json("grid"))));
                    if (const auto spacing = shown->find("spacing");
                        spacing != shown->end() && spacing->is_array() && spacing->size() == 3) {
                        worlds.spacing =
                            Double3{(*spacing)[0].get<double>(), (*spacing)[1].get<double>(),
                                    (*spacing)[2].get<double>()};
                    }
                    viewport.SetWorlds(std::move(worlds));
                }
                if (const auto lighting = entry.find("lighting");
                    lighting != entry.end() && viewport.ShowsWorlds()) {
                    viewport.SetLighting(LightingMode(NameIndex(kLightingNames, *lighting)));
                }
                const std::string camera = entry.value("camera", std::string());
                if (!camera.empty() && m_scene && m_scene->Find(camera)) {
                    viewport.LookThrough(camera);
                }
            }
        }
        if (const auto ini = layout.find("imgui"); ini != layout.end()) {
            m_pendingIni = ini->get<std::string>();
        }
    }

    void RemoveViewport(std::string_view name) {
        const auto found = std::ranges::find_if(
            m_viewports, [&](const auto& viewport) { return viewport->Name() == name; });
        if (found == m_viewports.end()) {
            throw Error("The viewer has no viewport \"{}\"", name);
        }
        for (const auto& viewport : m_viewports) {
            viewport->Follow(nullptr);
        }
        m_retiredViewports.push_back({std::move(*found), m_lastDraw});
        m_viewports.erase(found);
        if (m_active == name) {
            m_active.clear();
        }
        m_layoutBuilt = false;
    }

    std::vector<std::string> Viewports() const {
        std::vector<std::string> names;
        for (const auto& viewport : m_viewports) {
            names.push_back(viewport->Name());
        }
        return names;
    }

    Viewport* FindViewport(std::string_view name) const {
        for (const auto& viewport : m_viewports) {
            if (viewport->Name() == name) {
                return viewport.get();
            }
        }
        return nullptr;
    }

    Viewport& GetViewport(std::string_view name) const {
        if (name.empty()) {
            if (m_viewports.empty()) {
                throw Error("The viewer has no viewports");
            }
            return *m_viewports.front();
        }
        Viewport* found = FindViewport(name);
        if (found == nullptr) {
            throw Error("The viewer has no viewport \"{}\"", name);
        }
        return *found;
    }

    Viewport* Active() const {
        if (Viewport* active = FindViewport(m_active)) {
            return active;
        }
        return m_viewports.empty() ? nullptr : m_viewports.front().get();
    }

    struct MenuEntry {
        uint64_t id = 0;
        MenuItemDesc item;
        ImGuiKeyChord chord = 0;
    };

    struct KeyHandler {
        uint64_t id = 0;
        std::function<bool(const KeyEvent&)> handle;
    };

    struct MouseHandler {
        uint64_t id = 0;
        std::function<bool(const MouseEvent&)> handle;
    };

    struct OverlayHandler {
        uint64_t id = 0;
        std::function<bool(const OverlayEvent&)> handle;
    };

    std::vector<KeyHandler> keyHandlers;
    std::vector<MouseHandler> mouseHandlers;
    std::vector<OverlayHandler> overlayHandlers;

    void ConnectOverlayEvents() {
        for (const std::unique_ptr<Viewport>& viewport : m_viewports) {
            if (overlayHandlers.empty()) {
                viewport->overlayEvents = {};
                continue;
            }
            viewport->overlayEvents = [this](const OverlayEvent& event) {
                bool taken = false;
                const std::vector<OverlayHandler> handlers = overlayHandlers;
                for (const OverlayHandler& handler : handlers) {
                    taken = handler.handle(event) || taken;
                }
                return taken;
            };
        }
    }

    uint64_t nextHandler = 1;
    uint64_t nextOverlay = 1;
    bool keysTaken = false;
    bool guiShown = true;
    bool baseGuiShown = true;

    void DispatchInput() {
        keysTaken = false;
        const ImGuiIO& io = ImGui::GetIO();
        const Modifiers modifiers{io.KeyCtrl, io.KeyShift, io.KeyAlt, io.KeySuper};
        if (!keyHandlers.empty() && !io.WantTextInput) {
            for (int key = ImGuiKey_NamedKey_BEGIN; key < ImGuiKey_GamepadStart; ++key) {
                const ImGuiKey named = ImGuiKey(key);
                const bool first = ImGui::IsKeyPressed(named, false);
                const bool repeat = !first && ImGui::IsKeyPressed(named, true);
                const bool released = ImGui::IsKeyReleased(named);
                if (!first && !repeat && !released) {
                    continue;
                }
                const KeyEvent event{ImGui::GetKeyName(named),
                                     first    ? InputAction::Press
                                     : repeat ? InputAction::Repeat
                                              : InputAction::Release,
                                     modifiers};
                for (const KeyHandler& handler : keyHandlers) {
                    keysTaken = handler.handle(event) || keysTaken;
                }
            }
        }
        Viewport* hovered = nullptr;
        for (const auto& viewport : m_viewports) {
            viewport->BlockInput(false);
            if (viewport->visible && viewport->Hovered()) {
                hovered = viewport.get();
            }
        }
        if (mouseHandlers.empty()) {
            return;
        }
        bool taken = false;
        MouseEvent event;
        if (hovered != nullptr) {
            event.x = io.MousePos.x - hovered->Rect().x;
            event.y = io.MousePos.y - hovered->Rect().y;
            event.inViewport = true;
            event.viewport = hovered->Name();
        } else {
            event.x = io.MousePos.x;
            event.y = io.MousePos.y;
        }
        event.modifiers = modifiers;
        const auto send = [&](MouseEvent sent) {
            for (const MouseHandler& handler : mouseHandlers) {
                taken = handler.handle(sent) || taken;
            }
        };
        for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
            if (ImGui::IsMouseClicked(button)) {
                event.action = InputAction::Press;
                event.button = button;
                send(event);
            }
            if (ImGui::IsMouseReleased(button)) {
                event.action = InputAction::Release;
                event.button = button;
                send(event);
            }
        }
        event.button = -1;
        if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) {
            event.action = InputAction::Move;
            send(event);
        }
        if (io.MouseWheel != 0 || io.MouseWheelH != 0) {
            event.action = InputAction::Scroll;
            event.scrollX = io.MouseWheelH;
            event.scrollY = io.MouseWheel;
            send(event);
        }
        if (hovered != nullptr) {
            hovered->BlockInput(taken);
        }
    }

    void SetFontSize(float pixels) {
        if (pixels > 0) {
            ImGui::GetStyle().FontSizeBase = pixels;
        }
    }

    std::vector<MenuEntry> menuItems;
    uint64_t nextMenuItem = 1;
    bool menuBarShown = true;

    std::vector<GLFWmonitor*> Monitors() const {
        int count = 0;
        GLFWmonitor** monitors = m_glfw ? glfwGetMonitors(&count) : nullptr;
        return monitors ? std::vector<GLFWmonitor*>(monitors, monitors + count)
                        : std::vector<GLFWmonitor*>();
    }

    std::vector<Screen> Screens() const {
        std::vector<Screen> screens;
        GLFWmonitor* primary = m_glfw ? glfwGetPrimaryMonitor() : nullptr;
        for (GLFWmonitor* monitor : Monitors()) {
            Screen screen;
            const char* name = glfwGetMonitorName(monitor);
            screen.name = name ? name : "";
            glfwGetMonitorPos(monitor, &screen.x, &screen.y);
            if (const GLFWvidmode* mode = glfwGetVideoMode(monitor)) {
                screen.width = uint32_t(mode->width);
                screen.height = uint32_t(mode->height);
                screen.refreshRate = uint32_t(mode->refreshRate);
            }
            glfwGetMonitorContentScale(monitor, &screen.scale, nullptr);
            screen.primary = monitor == primary;
            screens.push_back(std::move(screen));
        }
        return screens;
    }

    GLFWmonitor* Monitor(int screen) const {
        const std::vector<GLFWmonitor*> monitors = Monitors();
        if (screen < 0 || size_t(screen) >= monitors.size()) {
            throw Error("No screen {}; {} are connected", screen, monitors.size());
        }
        return monitors[size_t(screen)];
    }

    void SetWindowPosition(int x, int y) {
        if (m_window != nullptr && !Fullscreen()) {
            glfwSetWindowPos(m_window, x, y);
        }
    }

    void HideWindow() {
        if (m_window != nullptr) {
            glfwHideWindow(m_window);
            glfwPollEvents();
        }
    }

    void SetWindowSize(uint32_t width, uint32_t height) {
        if (m_window != nullptr && !Fullscreen()) {
            glfwSetWindowSize(m_window, int(width), int(height));
        }
    }

    void SetOutputScreen(int screen) {
        if (m_window == nullptr) {
            return;
        }
        GLFWmonitor* monitor = Monitor(screen);
        m_screen = screen;
        if (Fullscreen()) {
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(m_window, monitor, 0, 0, mode->width, mode->height,
                                 mode->refreshRate);
            return;
        }
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
        glfwGetMonitorWorkarea(monitor, &x, &y, &width, &height);
        int windowWidth = 0;
        int windowHeight = 0;
        glfwGetWindowSize(m_window, &windowWidth, &windowHeight);
        glfwSetWindowPos(m_window, x + std::max(0, (width - windowWidth) / 2),
                         y + std::max(0, (height - windowHeight) / 2));
    }

    bool Fullscreen() const {
        return m_window != nullptr && glfwGetWindowMonitor(m_window) != nullptr;
    }

    void EnableFullscreen(bool fullscreen) {
        if (m_window == nullptr || fullscreen == Fullscreen()) {
            return;
        }
        if (fullscreen) {
            glfwGetWindowPos(m_window, &m_windowed[0], &m_windowed[1]);
            glfwGetWindowSize(m_window, &m_windowed[2], &m_windowed[3]);
            GLFWmonitor* monitor = m_screen >= 0 ? Monitor(m_screen) : glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(m_window, monitor, 0, 0, mode->width, mode->height,
                                 mode->refreshRate);
        } else {
            glfwSetWindowMonitor(m_window, nullptr, m_windowed[0], m_windowed[1], m_windowed[2],
                                 m_windowed[3], GLFW_DONT_CARE);
        }
    }

    std::set<std::string, std::less<>> hiddenMenus;
    std::set<std::pair<std::string, std::string>, std::less<>> hiddenItems;

    bool Shown(std::string_view menu) const {
        return !hiddenMenus.contains(menu);
    }

    bool ShownItem(std::string_view menu, std::string_view label) const {
        return !hiddenItems.contains(std::pair(std::string(menu), std::string(label)));
    }

    uint64_t AddMenuItem(MenuItemDesc item) {
        const ImGuiKeyChord chord = ParseChord(item.shortcut);
        menuItems.push_back({nextMenuItem, std::move(item), chord});
        return nextMenuItem++;
    }

    std::vector<Panel> panels;

    device::Device& GetDevice() {
        return *m_device;
    }

    const std::vector<std::string>& Selected() const {
        return m_selection.paths;
    }

    void Select(std::vector<std::string> paths) {
        m_selection.paths = std::move(paths);
    }

    void RecordEdit(Edit edit) {
        m_history.Push(std::move(edit));
    }

    bool UndoEdit() {
        return m_history.Undo();
    }

    bool RedoEdit() {
        return m_history.Redo();
    }

    std::unique_ptr<Presenter> presenter;
    ImGuiContext* imgui = nullptr;

  private:
    struct RetiredViewport {
        std::unique_ptr<Viewport> viewport;
        device::Operation drawn;
    };

    void Release() noexcept;
    void Lead();
    void DrawMenuBar();
    void DrawMenuItems(std::string_view menu, bool separated = true);
    void DrawDockSpace();
    void DrawCreateMenu();
    void DrawPanels();
    void TakeShortcuts();

    Ref<device::Device> m_device;
    Ref<Scene> m_scene;
    ViewerDesc m_desc;
    GLFWwindow* m_window = nullptr;
    bool m_glfw = false;
    device::Operation m_lastDraw;
    Theme m_theme;
    Selection m_selection;
    History m_history;
    Editor m_editor;
    FileDialog m_fileDialog;
    SessionWorlds m_sessions;
    std::vector<std::function<void()>> m_deferred;
    std::optional<std::string> m_pendingIni;
    Capture m_capture;
    std::vector<std::unique_ptr<Viewport>> m_viewports;
    std::vector<RetiredViewport> m_retiredViewports;
    std::string m_active;
    int m_screen = -1;
    std::array<int, 4> m_windowed{};
    std::array<ImGuiID, 5> m_docks{};
    bool m_layoutBuilt = false;
    bool m_showControls = false;
    bool m_showAbout = false;
    bool m_quit = false;
};

ViewerImpl::ViewerImpl(Ref<device::Device> device, Ref<Scene> scene, const ViewerDesc& desc)
    : m_device(std::move(device)), m_scene(std::move(scene)), m_desc(desc),
      m_editor(m_scene, m_selection, m_history) {
    menuBarShown = m_desc.menuBar;
    AddViewport({.name = "Viewport",
                 .product = m_desc.product,
                 .aov = m_desc.aov,
                 .delegate = m_desc.delegate,
                 .renderer = m_desc.renderer,
                 .hud = m_desc.hud,
                 .hudItems = m_desc.hudItems,
                 .toolbar = m_desc.toolbar,
                 .targetRenderFps = m_desc.targetRenderFps});
    if (m_desc.ui && m_desc.builtinPanels) {
        std::vector<Panel> builtins = {
            CreateInfoPanel(m_scene, m_device, m_viewports.front()->Delegate()),
            CreateLogPanel(m_theme), CreateProfilerPanel(), CreateMemoryPanel(m_device),
            CreateCapturePanel(m_capture)};
        if (m_scene) {
            builtins.push_back(CreateStagePanel(m_scene, m_selection, m_editor, m_theme));
            builtins.push_back(
                CreatePropertiesPanel(m_scene, m_selection, m_editor, m_fileDialog, m_theme));
        }
        for (Panel& panel : builtins) {
            panel.builtin = true;
            AddPanel(std::move(panel));
        }
    }
    IMGUI_CHECKVERSION();
    imgui = ImGui::CreateContext();
    ImGui::SetCurrentContext(imgui);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    if (m_desc.ui) {
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    }
    try {
        if (!desc.offscreen) {
            m_glfw = glfwInit() == GLFW_TRUE;
            if (!m_glfw) {
                throw Error("GLFW couldn't start, so no window opens");
            }
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            m_window = glfwCreateWindow(int(desc.width), int(desc.height), desc.title.c_str(),
                                        nullptr, nullptr);
            if (m_window == nullptr) {
                throw Error("GLFW couldn't open a window");
            }
            if (desc.screen >= 0) {
                SetOutputScreen(desc.screen);
            }
            if (desc.position) {
                glfwSetWindowPos(m_window, desc.position->x, desc.position->y);
            }
            if (desc.fullscreen) {
                EnableFullscreen(true);
            }
            ImGui_ImplGlfw_InitForOther(m_window, true);
        }
        float scale = 1;
        if (m_window != nullptr) {
            glfwGetWindowContentScale(m_window, &scale, nullptr);
        }
        ApplyTheme(m_theme, scale);
        LoadFonts(m_theme);
        SetFontSize(m_desc.fontSize);
        presenter = CreatePresenter(*m_device, m_window, {desc.width, desc.height}, desc.vsync);
        if (!desc.layout.empty()) {
            LoadLayoutFile(desc.layout);
        }
    } catch (...) {
        Release();
        throw;
    }
}

ViewerImpl::~ViewerImpl() {
    Release();
}

void ViewerImpl::Release() noexcept {
    if (imgui != nullptr) {
        ImGui::SetCurrentContext(imgui);
    }
    presenter.reset();
    if (m_window != nullptr) {
        ImGui_ImplGlfw_Shutdown();
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    if (m_glfw) {
        glfwTerminate();
        m_glfw = false;
    }
    if (imgui != nullptr) {
        ImGui::DestroyContext(imgui);
        imgui = nullptr;
    }
}

bool ViewerImpl::Open() const {
    return !m_quit && (m_window == nullptr || glfwWindowShouldClose(m_window) == GLFW_FALSE);
}

void ViewerImpl::Lead() {
    for (const auto& viewport : m_viewports) {
        if (viewport->visible && viewport->Interacted()) {
            m_active = viewport->Name();
        }
    }
    const Viewport* active = Active();
    for (const auto& viewport : m_viewports) {
        const Viewport* leader = nullptr;
        if (!viewport->Product().empty() && viewport->Delegate()) {
            for (const auto& other : m_viewports) {
                if (other->visible && other->Product() == viewport->Product() &&
                    (leader == nullptr || other.get() == active)) {
                    leader = other.get();
                }
            }
        }
        viewport->Follow(leader);
    }
}

void ViewerImpl::Frame() {
    ImGui::SetCurrentContext(imgui);
    profiler::BeginFrame();
    ERNIE_ZONE("Viewer/Frame");
    if (m_window != nullptr) {
        glfwPollEvents();
    }
    std::erase_if(m_retiredViewports,
                  [](const RetiredViewport& retired) { return retired.drawn.Ready(); });
    for (const std::function<void()>& deferred : std::exchange(m_deferred, {})) {
        deferred();
    }
    Lead();
    for (const auto& viewport : m_viewports) {
        if (viewport->visible) {
            viewport->Advance(m_lastDraw);
        }
    }
    if (!presenter->Begin()) {
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    if (m_window != nullptr) {
        ImGui_ImplGlfw_NewFrame();
    } else {
        const Extent size = presenter->Size();
        io.DisplaySize = {float(size.width), float(size.height)};
        io.DeltaTime = 1.0f / 60.0f;
    }
    if (m_pendingIni) {
        ImGui::ClearIniSettings();
        ImGui::LoadIniSettingsFromMemory(m_pendingIni->c_str(), m_pendingIni->size());
        m_pendingIni.reset();
        m_layoutBuilt = true;
    }
    ImGui::NewFrame();
    DispatchInput();
    const bool gui = m_desc.ui && guiShown;
    if (gui) {
        DrawMenuBar();
        DrawDockSpace();
    }
    std::vector<Viewport*> shown;
    for (const auto& viewport : m_viewports) {
        if (viewport->visible) {
            shown.push_back(viewport.get());
        }
    }
    const ImGuiViewport& main = *ImGui::GetMainViewport();
    for (size_t at = 0; at < shown.size(); ++at) {
        std::optional<ViewportRect> fixed;
        if (!gui) {
            const float width = main.Size.x / float(shown.size());
            fixed = ViewportRect{main.Pos.x + width * float(at), main.Pos.y, width, main.Size.y};
        }
        shown[at]->Draw(*presenter, fixed, gui && baseGuiShown);
    }
    if (gui) {
        DrawPanels();
        DrawControls(m_showControls, m_theme);
        DrawAbout(m_showAbout, *m_device);
        m_fileDialog.Draw();
    }
    if (m_desc.ui) {
        TakeShortcuts();
    }
    if (const Viewport* active = Active()) {
        const std::string& camera = active->CameraPath();
        m_capture.Update(active->Shown(), active->ShownFrame(),
                         m_desc.title.empty() ? "ernie" : m_desc.title,
                         camera.substr(camera.rfind('/') + 1));
    }
    ImGui::Render();
    std::vector<device::Operation> ready;
    for (const Viewport* viewport : shown) {
        ready.push_back(viewport->ShownReady());
    }
    m_lastDraw = presenter->End(ready);
}

void ViewerImpl::DrawCreateMenu() {
    std::array<std::vector<SchemaId>, std::size(kPrimGroupNames)> groups;
    for (uint32_t id = 0; id < SchemaCount(); ++id) {
        if (Info(SchemaId(id)).kind == SchemaKind::Concrete) {
            groups[size_t(Classify(SchemaId(id)))].push_back(SchemaId(id));
        }
    }
    const auto create = [&](SchemaId type) {
        const std::string_view name = Info(type).name;
        if (!ShownItem("Create", name)) {
            return;
        }
        const float icon = ImGui::GetFontSize();
        const std::string label = "      " + std::string(name);
        const bool clicked = ImGui::MenuItem(label.c_str());
        const ImVec2 corner = ImGui::GetItemRectMin();
        DrawPrimIcon(*ImGui::GetWindowDrawList(),
                     {corner.x, corner.y + 0.5f * (ImGui::GetItemRectSize().y - icon)}, icon, name);
        if (!clicked) {
            return;
        }
        const Viewport* active = Active();
        const OrbitCamera orbit = active ? active->Orbit() : OrbitCamera{};
        switch (Classify(type)) {
        case PrimGroup::Cameras:
            m_editor.Create(type, orbit.Eye(), orbit.Orient());
            return;
        case PrimGroup::Lights:
            m_editor.Create(type, {orbit.target.x, orbit.target.y, orbit.target.z + orbit.distance},
                            Quatd{});
            return;
        default:
            m_editor.Create(type, orbit.target, Quatd{});
            return;
        }
    };
    for (size_t group = 0; group < groups.size(); ++group) {
        std::vector<SchemaId>& members = groups[group];
        std::ranges::sort(members, {}, [](SchemaId id) { return Info(id).name; });
        if (members.empty()) {
            continue;
        }
        if (PrimGroup(group) == PrimGroup::Organize || members.size() == 1) {
            std::ranges::for_each(members, create);
            if (PrimGroup(group) == PrimGroup::Organize) {
                ImGui::Separator();
            }
        } else if (Shown(fmt::format("Create/{}", kPrimGroupNames[group])) &&
                   ImGui::BeginMenu(kPrimGroupNames[group])) {
            std::ranges::for_each(members, create);
            ImGui::EndMenu();
        }
    }
}

void ViewerImpl::DrawMenuBar() {
    if (!menuBarShown) {
        return;
    }
    std::vector<std::string> custom;
    for (const MenuEntry& entry : menuItems) {
        const std::string top = entry.item.menu.substr(0, entry.item.menu.find('/'));
        if (std::ranges::find(kBuiltinMenus, top) == std::end(kBuiltinMenus) &&
            std::ranges::find(custom, top) == custom.end()) {
            custom.push_back(top);
        }
    }
    const bool base = baseGuiShown;
    if ((!base && custom.empty()) || !ImGui::BeginMainMenuBar()) {
        return;
    }
    const auto item = [&](std::string_view menu, const char* label, const char* shortcut,
                          bool selected = false, bool enabled = true) {
        return ShownItem(menu, label) && ImGui::MenuItem(label, shortcut, selected, enabled);
    };
    const auto toggle = [&](std::string_view menu, const char* label, bool& value) {
        if (ShownItem(menu, label)) {
            ImGui::MenuItem(label, nullptr, &value);
        }
    };
    if (base && Shown("File") && ImGui::BeginMenu("File")) {
        if (item("File", "Screenshot", "F10")) {
            m_capture.Screenshot();
        }
        ImGui::Separator();
        DrawMenuItems("File", false);
        if (item("File", "Quit", "Ctrl+Q")) {
            m_quit = true;
        }
        ImGui::EndMenu();
    }
    if (base && m_scene && Shown("Edit") && ImGui::BeginMenu("Edit")) {
        const std::string* undo = m_history.NextUndo();
        const std::string* redo = m_history.NextRedo();
        if (ShownItem("Edit", "Undo") &&
            ImGui::MenuItem(undo ? ("Undo " + *undo).c_str() : "Undo", "Ctrl+Z", false, undo)) {
            m_history.Undo();
        }
        if (ShownItem("Edit", "Redo") && ImGui::MenuItem(redo ? ("Redo " + *redo).c_str() : "Redo",
                                                         "Ctrl+Shift+Z", false, redo)) {
            m_history.Redo();
        }
        ImGui::Separator();
        const bool selected = !m_selection.paths.empty();
        if (item("Edit", "Duplicate", "Ctrl+D", false, selected)) {
            m_editor.Duplicate();
        }
        if (item("Edit", "Delete", "Del", false, selected)) {
            m_editor.Delete();
        }
        if (item("Edit", "Clear Selection", nullptr, false, selected)) {
            m_selection.Clear();
        }
        DrawMenuItems("Edit");
        ImGui::EndMenu();
    }
    if (base && m_scene && Shown("Create") && ImGui::BeginMenu("Create")) {
        DrawCreateMenu();
        DrawMenuItems("Create");
        ImGui::EndMenu();
    }
    if (base && Shown("Window") && ImGui::BeginMenu("Window")) {
        if (ShownItem("Window", "Viewports") && ImGui::BeginMenu("Viewports")) {
            for (const auto& viewport : m_viewports) {
                ImGui::MenuItem(viewport->Name().c_str(), nullptr, &viewport->visible);
            }
            ImGui::Separator();
            if (m_scene && ImGui::MenuItem("Add Viewport")) {
                AddViewportBeside();
            }
            const Viewport* active = Active();
            const std::string remove =
                active != nullptr ? "Remove " + active->Name() : std::string("Remove Viewport");
            if (ImGui::MenuItem(remove.c_str(), nullptr, false,
                                active != nullptr && m_viewports.size() > 1)) {
                RemoveViewport(active->Name());
            }
            ImGui::EndMenu();
        }
        if (Viewport* active = Active()) {
            toggle("Window", "Toolbar", active->toolbarShown);
        }
        ImGui::Separator();
        for (Panel& panel : panels) {
            toggle("Window", panel.title.c_str(), panel.visible);
        }
        ImGui::Separator();
        if (ShownItem("Window", "Interface")) {
            ImGui::MenuItem("Interface", "Tab", &guiShown);
        }
        if (m_window != nullptr && ShownItem("Window", "Fullscreen")) {
            bool fullscreen = Fullscreen();
            if (ImGui::MenuItem("Fullscreen", nullptr, &fullscreen)) {
                EnableFullscreen(fullscreen);
            }
        }
        ImGui::Separator();
        const std::vector<FileFilter> layouts = {{"Viewer layouts", {".json"}}};
        if (item("Window", "Save Layout...", nullptr)) {
            m_fileDialog.Open({.title = "Save Layout", .save = true, .filters = layouts},
                              [this](const std::filesystem::path& path) {
                                  try {
                                      SaveLayoutFile(path);
                                  } catch (const std::exception& error) {
                                      Log(LogLevel::Error, "{}", error.what());
                                  }
                              });
        }
        if (item("Window", "Load Layout...", nullptr)) {
            m_fileDialog.Open({.title = "Load Layout", .filters = layouts},
                              [this](const std::filesystem::path& path) {
                                  m_deferred.push_back([this, path] {
                                      try {
                                          LoadLayoutFile(path);
                                      } catch (const std::exception& error) {
                                          Log(LogLevel::Error, "{}", error.what());
                                      }
                                  });
                              });
        }
        DrawMenuItems("Window");
        ImGui::EndMenu();
    }
    for (const std::string& menu : custom) {
        if (Shown(menu) && ImGui::BeginMenu(menu.c_str())) {
            DrawMenuItems(menu, false);
            ImGui::EndMenu();
        }
    }
    if (base && Shown("Help") && ImGui::BeginMenu("Help")) {
        toggle("Help", "Keyboard and Mouse", m_showControls);
        toggle("Help", "About Ernie", m_showAbout);
        DrawMenuItems("Help");
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
}

void ViewerImpl::DrawMenuItems(std::string_view menu, bool separated) {
    bool first = true;
    const auto separate = [&] {
        if (first && separated) {
            ImGui::Separator();
        }
        first = false;
    };
    std::vector<std::string> submenus;
    const std::string prefix = std::string(menu) + "/";
    for (const MenuEntry& entry : menuItems) {
        const MenuItemDesc& item = entry.item;
        if (item.menu.starts_with(prefix)) {
            const std::string sub =
                item.menu.substr(prefix.size(), item.menu.find('/', prefix.size()) - prefix.size());
            if (std::ranges::find(submenus, sub) == submenus.end()) {
                submenus.push_back(sub);
            }
            continue;
        }
        if (item.menu != menu || !ShownItem(menu, item.label)) {
            continue;
        }
        separate();
        const bool checked = item.checked && item.checked();
        const bool enabled = !item.enabled || item.enabled();
        if (ImGui::MenuItem(item.label.c_str(),
                            item.shortcut.empty() ? nullptr : item.shortcut.c_str(), checked,
                            enabled) &&
            item.action) {
            item.action();
        }
    }
    for (const std::string& sub : submenus) {
        if (!Shown(prefix + sub) || !ShownItem(menu, sub)) {
            continue;
        }
        separate();
        if (ImGui::BeginMenu(sub.c_str())) {
            DrawMenuItems(prefix + sub, false);
            ImGui::EndMenu();
        }
    }
}

void ViewerImpl::DrawDockSpace() {
    const ImGuiID dockspace = ImHashStr("ErnieDockSpace");
    if (!m_layoutBuilt) {
        ImGui::DockBuilderRemoveNode(dockspace);
        ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspace, ImGui::GetMainViewport()->WorkSize);
        const auto wants = [&](Dock dock) {
            return std::ranges::any_of(
                panels, [&](const Panel& panel) { return panel.visible && panel.dock == dock; });
        };
        ImGuiID center = dockspace;
        const std::pair<Dock, std::pair<ImGuiDir, float>> splits[] = {
            {Dock::Left, {ImGuiDir_Left, 0.2f}},
            {Dock::Right, {ImGuiDir_Right, 0.25f}},
            {Dock::Bottom, {ImGuiDir_Down, 0.25f}}};
        m_docks = {};
        for (const auto& [dock, split] : splits) {
            if (wants(dock)) {
                m_docks[size_t(dock)] = ImGui::DockBuilderSplitNode(center, split.first,
                                                                    split.second, nullptr, &center);
            }
        }
        if (wants(Dock::RightBottom)) {
            ImGuiID& right = m_docks[size_t(Dock::Right)];
            m_docks[size_t(Dock::RightBottom)] =
                right != 0
                    ? ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.55f, nullptr, &right)
                    : ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.25f, nullptr, &center);
        }
        std::vector<const Viewport*> shown;
        for (const auto& viewport : m_viewports) {
            if (viewport->visible) {
                shown.push_back(viewport.get());
            }
        }
        for (size_t at = 0; at < shown.size(); ++at) {
            ImGuiID node = center;
            if (at + 1 < shown.size()) {
                node = ImGui::DockBuilderSplitNode(
                    center, ImGuiDir_Left, 1.0f / float(shown.size() - at), nullptr, &center);
            }
            ImGui::DockBuilderDockWindow(shown[at]->Name().c_str(), node);
        }
        for (const Panel& panel : panels) {
            if (m_docks[size_t(panel.dock)] != 0) {
                ImGui::DockBuilderDockWindow(panel.title.c_str(), m_docks[size_t(panel.dock)]);
            }
        }
        ImGui::DockBuilderFinish(dockspace);
        m_layoutBuilt = true;
    }
    ImGui::DockSpaceOverViewport(dockspace, ImGui::GetMainViewport());
}

void ViewerImpl::DrawPanels() {
    for (Panel& panel : panels) {
        bool drawn = false;
        if (panel.visible && (baseGuiShown || !panel.builtin)) {
            if (m_docks[size_t(panel.dock)] != 0) {
                ImGui::SetNextWindowDockID(m_docks[size_t(panel.dock)], ImGuiCond_FirstUseEver);
            } else if (panel.dock == Dock::Floating) {
                const ImGuiViewport& main = *ImGui::GetMainViewport();
                ImGui::SetNextWindowPos({main.WorkPos.x + 0.5f * main.WorkSize.x,
                                         main.WorkPos.y + 0.5f * main.WorkSize.y},
                                        ImGuiCond_FirstUseEver, {0.5f, 0.5f});
                ImGui::SetNextWindowSize({24 * ImGui::GetFontSize(), 16 * ImGui::GetFontSize()},
                                         ImGuiCond_FirstUseEver);
            }
            if (ImGui::Begin(panel.title.c_str(), &panel.visible)) {
                ERNIE_ZONE("Viewer/Panel");
                panel.draw();
                drawn = true;
            }
            ImGui::End();
        }
        if (!drawn && panel.hidden) {
            panel.hidden();
        }
    }
}

void ViewerImpl::TakeShortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput || keysTaken) {
        return;
    }
    if (!io.KeyCtrl && !io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        guiShown = !guiShown;
    }
    for (const MenuEntry& entry : menuItems) {
        if (entry.chord != 0 && entry.item.action && ImGui::IsKeyChordPressed(entry.chord) &&
            (!entry.item.enabled || entry.item.enabled())) {
            entry.item.action();
        }
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
        m_quit = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F10, false)) {
        m_capture.Screenshot();
    }
    if (!m_scene) {
        return;
    }
    if (!io.KeyCtrl && !io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        if (Viewport* active = Active()) {
            active->Frame(!io.KeyShift);
        }
    }
    constexpr std::pair<ImGuiKey, GizmoMode> kGizmoKeys[] = {{ImGuiKey_Q, GizmoMode::None},
                                                             {ImGuiKey_W, GizmoMode::Translate},
                                                             {ImGuiKey_E, GizmoMode::Rotate},
                                                             {ImGuiKey_R, GizmoMode::Scale}};
    for (const auto& [key, mode] : kGizmoKeys) {
        if (!io.KeyCtrl && !io.KeyAlt && ImGui::IsKeyPressed(key, false)) {
            if (Viewport* active = Active(); active && active->GetGizmo()) {
                active->GetGizmo()->mode = mode;
            }
        }
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z, true)) {
        if (io.KeyShift) {
            m_history.Redo();
        } else {
            m_history.Undo();
        }
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Y, true)) {
        m_history.Redo();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_D, false)) {
        m_editor.Duplicate();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
        m_editor.Delete();
    }
}

Viewer::Viewer(Ref<device::Device> device, const ViewerDesc& desc)
    : m_impl(std::make_unique<ViewerImpl>(std::move(device), nullptr, desc)) {}

Viewer::Viewer(Ref<Scene> scene, const ViewerDesc& desc)
    : m_impl(std::make_unique<ViewerImpl>(DisplayDevice(scene->GetDevice()), scene, desc)) {}

Viewer::~Viewer() = default;

bool Viewer::Open() const {
    return m_impl->Open();
}

void Viewer::Frame() {
    m_impl->Frame();
}

void Viewer::Run() {
    while (Open()) {
        Frame();
    }
    m_impl->HideWindow();
}

void Viewer::Show(const device::Tensor& image, const device::Operation& ready,
                  std::string_view viewport) {
    m_impl->GetViewport(viewport).Show(image, ready);
}

void Viewer::AddViewport(ViewportDesc desc) {
    m_impl->AddViewport(std::move(desc));
}

void Viewer::RemoveViewport(std::string_view name) {
    m_impl->RemoveViewport(name);
}

std::vector<std::string> Viewer::Viewports() const {
    return m_impl->Viewports();
}

void Viewer::AddPanel(std::string title, Dock dock, std::function<void()> draw, bool visible) {
    m_impl->AddPanel({std::move(title), dock, std::move(draw), visible});
}

void Viewer::ShowPanel(std::string_view title, bool shown) {
    m_impl->ShowPanel(title, shown);
}

uint64_t Viewer::AddMenuItem(std::string menu, std::string label, std::function<void()> action,
                             std::string shortcut) {
    return m_impl->AddMenuItem(
        {std::move(menu), std::move(label), std::move(action), std::move(shortcut), {}, {}});
}

uint64_t Viewer::AddMenuItem(MenuItemDesc item) {
    return m_impl->AddMenuItem(std::move(item));
}

void Viewer::RemoveMenuItem(uint64_t id) {
    std::erase_if(m_impl->menuItems, [&](const auto& entry) { return entry.id == id; });
}

void Viewer::ShowMenuBar(bool shown) {
    m_impl->menuBarShown = shown;
}

void Viewer::ShowMenu(std::string_view menu, bool shown) {
    if (shown) {
        m_impl->hiddenMenus.erase(std::string(menu));
    } else {
        m_impl->hiddenMenus.emplace(menu);
    }
}

void Viewer::ShowMenuItem(std::string_view menu, std::string_view label, bool shown) {
    const std::pair<std::string, std::string> key(menu, label);
    if (shown) {
        m_impl->hiddenItems.erase(key);
    } else {
        m_impl->hiddenItems.insert(key);
    }
}

void Viewer::AddOverlay(std::function<void(const ViewportRect&)> draw, std::string_view viewport) {
    m_impl->GetViewport(viewport).AddOverlay(std::move(draw));
}

uint64_t Viewer::AddText(std::string text, const TextStyle& style, std::string_view viewport) {
    m_impl->GetViewport(viewport).AddText(m_impl->nextOverlay, std::move(text), style);
    return m_impl->nextOverlay++;
}

void Viewer::SetText(uint64_t id, std::string text) {
    for (const std::string& name : m_impl->Viewports()) {
        if (m_impl->GetViewport(name).SetText(id, text)) {
            return;
        }
    }
    throw Error("The viewer has no text {}", id);
}

uint64_t Viewer::AddImage(const device::Tensor& image, const Placement& placement, float opacity,
                          std::string_view viewport) {
    m_impl->GetViewport(viewport).AddImage(m_impl->nextOverlay, image, {}, placement, opacity);
    return m_impl->nextOverlay++;
}

uint64_t Viewer::AddImage(const std::filesystem::path& file, const Placement& placement,
                          float opacity, std::string_view viewport) {
    Rgba8Image loaded = LoadRgba8File(file);
    int64_t shape[] = {int64_t(loaded.height), int64_t(loaded.width), 4};
    DLTensor host{};
    host.data = loaded.texels.data();
    host.device = {kDLCPU, 0};
    host.ndim = 3;
    host.dtype = {kDLUInt, 8, 1};
    host.shape = shape;
    device::Device& device = m_impl->GetDevice();
    const device::Tensor pixels = device.Import(host, Stream::None(), file.filename().string());
    m_impl->GetViewport(viewport).AddImage(m_impl->nextOverlay, pixels,
                                           device.Follow(Stream::None()), placement, opacity);
    return m_impl->nextOverlay++;
}

void Viewer::RemoveOverlay(uint64_t id) {
    for (const std::string& name : m_impl->Viewports()) {
        m_impl->GetViewport(name).RemoveOverlay(id);
    }
}

void Viewer::RecordEdit(std::string name, std::function<void()> undo, std::function<void()> redo) {
    m_impl->RecordEdit({std::move(name), std::move(undo), std::move(redo)});
}

bool Viewer::Undo() {
    return m_impl->UndoEdit();
}

bool Viewer::Redo() {
    return m_impl->RedoEdit();
}

uint64_t Viewer::AddColorbar(const ColorbarDesc& desc, std::string_view viewport) {
    m_impl->GetViewport(viewport).AddColorbar(m_impl->nextOverlay, desc);
    return m_impl->nextOverlay++;
}

void Viewer::SetColorbar(uint64_t id, const ColorbarDesc& desc) {
    for (const std::string& name : m_impl->Viewports()) {
        if (m_impl->GetViewport(name).SetColorbar(id, desc)) {
            return;
        }
    }
    throw Error("The viewer has no colorbar {}", id);
}

void Viewer::ShowGui(bool shown) {
    m_impl->guiShown = shown;
}

void Viewer::ToggleGui() {
    m_impl->guiShown = !m_impl->guiShown;
}

bool Viewer::GuiShown() const {
    return m_impl->guiShown;
}

void Viewer::ShowBaseGui(bool shown) {
    m_impl->baseGuiShown = shown;
}

void Viewer::SetFontSize(float pixels) {
    m_impl->SetFontSize(pixels);
}

uint64_t Viewer::AddKeyHandler(std::function<bool(const KeyEvent&)> handler) {
    m_impl->keyHandlers.push_back({m_impl->nextHandler, std::move(handler)});
    return m_impl->nextHandler++;
}

uint64_t Viewer::AddMouseHandler(std::function<bool(const MouseEvent&)> handler) {
    m_impl->mouseHandlers.push_back({m_impl->nextHandler, std::move(handler)});
    return m_impl->nextHandler++;
}

uint64_t Viewer::AddOverlayHandler(std::function<bool(const OverlayEvent&)> handler) {
    m_impl->overlayHandlers.push_back({m_impl->nextHandler, std::move(handler)});
    m_impl->ConnectOverlayEvents();
    return m_impl->nextHandler++;
}

std::optional<Float2> Viewer::WorldToScreen(const Double3& point, std::string_view viewport) const {
    return m_impl->GetViewport(viewport).WorldToScreen(point);
}

std::optional<OverlayRay> Viewer::ScreenRay(Float2 at, std::string_view viewport) const {
    return m_impl->GetViewport(viewport).ScreenRay(at);
}

void Viewer::RemoveHandler(uint64_t id) {
    std::erase_if(m_impl->keyHandlers, [&](const auto& handler) { return handler.id == id; });
    std::erase_if(m_impl->mouseHandlers, [&](const auto& handler) { return handler.id == id; });
    std::erase_if(m_impl->overlayHandlers, [&](const auto& handler) { return handler.id == id; });
    m_impl->ConnectOverlayEvents();
}

void Viewer::ShowHud(bool shown, std::string_view viewport) {
    m_impl->GetViewport(viewport).hudShown = shown;
}

void Viewer::ShowHudItem(HudItem item, bool shown, std::string_view viewport) {
    m_impl->GetViewport(viewport).hudItems.at(size_t(item)) = shown;
}

void Viewer::AddHudItem(std::string label, std::function<std::string()> value,
                        std::string_view viewport) {
    m_impl->GetViewport(viewport).AddHudItem(std::move(label), std::move(value));
}

std::vector<Screen> Viewer::Screens() const {
    return m_impl->Screens();
}

void Viewer::SetWindowPosition(int x, int y) {
    m_impl->SetWindowPosition(x, y);
}

void Viewer::SetWindowSize(uint32_t width, uint32_t height) {
    m_impl->SetWindowSize(width, height);
}

void Viewer::SetOutputScreen(int screen) {
    m_impl->SetOutputScreen(screen);
}

void Viewer::EnableFullscreen(bool fullscreen) {
    m_impl->EnableFullscreen(fullscreen);
}

bool Viewer::Fullscreen() const {
    return m_impl->Fullscreen();
}

double Viewer::RenderFps(std::string_view viewport) const {
    return m_impl->GetViewport(viewport).RenderFps();
}

double Viewer::UiFps() const {
    if (m_impl->imgui == nullptr) {
        return 0.0;
    }
    ImGui::SetCurrentContext(m_impl->imgui);
    return double(ImGui::GetIO().Framerate);
}

void Viewer::SaveLayout(const std::filesystem::path& path) const {
    m_impl->SaveLayoutFile(path);
}

void Viewer::LoadLayout(const std::filesystem::path& path) {
    m_impl->LoadLayoutFile(path);
}

void Viewer::SetWorldDisplay(const WorldDisplay& display, std::string_view viewport) {
    m_impl->GetViewport(viewport).SetWorlds(display);
}

const WorldDisplay& Viewer::GetWorldDisplay(std::string_view viewport) const {
    return m_impl->GetViewport(viewport).Worlds();
}

void Viewer::SetLighting(LightingMode lighting, std::string_view viewport) {
    m_impl->GetViewport(viewport).SetLighting(lighting);
}

LightingMode Viewer::Lighting(std::string_view viewport) const {
    return m_impl->GetViewport(viewport).Lighting();
}

void Viewer::SetAov(std::string aov, std::string_view viewport) {
    m_impl->GetViewport(viewport).SetAov(aov);
}

const std::string& Viewer::Aov(std::string_view viewport) const {
    return m_impl->GetViewport(viewport).Aov();
}

void Viewer::SetAovColors(const ColorizeOptions& colors, std::string_view viewport) {
    m_impl->GetViewport(viewport).Display().colors = colors;
}

void Viewer::SetAovExposure(float stops, std::string_view viewport) {
    m_impl->GetViewport(viewport).Display().exposure = stops;
}

void Viewer::SetTargetRenderFps(double fps, std::string_view viewport) {
    m_impl->GetViewport(viewport).targetRenderFps = std::max(0.0, fps);
}

uint32_t Viewer::Width(std::string_view viewport) const {
    const Viewport& found = m_impl->GetViewport(viewport);
    return found.Size().width > 0 ? found.Size().width : m_impl->presenter->Size().width;
}

uint32_t Viewer::Height(std::string_view viewport) const {
    const Viewport& found = m_impl->GetViewport(viewport);
    return found.Size().height > 0 ? found.Size().height : m_impl->presenter->Size().height;
}

void Viewer::Select(std::vector<std::string> paths) {
    m_impl->Select(std::move(paths));
}

const std::vector<std::string>& Viewer::Selected() const {
    return m_impl->Selected();
}

const device::Tensor* Viewer::Shown(std::string_view viewport) const {
    return m_impl->GetViewport(viewport).Shown();
}

std::shared_ptr<Renderer> Viewer::GetRenderer(std::string_view viewport) const {
    return m_impl->GetViewport(viewport).GetRenderer();
}

std::vector<uint8_t> Viewer::ReadPixels() {
    return m_impl->ReadPixels();
}

}

extern "C" ERNIE_VIEWER_API int ErnieRunViewer(const ernie::app::Command* command,
                                               ernie::Scene* scene, const std::string* product,
                                               std::string* error) {
    try {
        ernie::app::Viewer viewer(ernie::Ref<ernie::Scene>(scene), {.product = *product,
                                                                    .aov = command->aov,
                                                                    .width = command->width,
                                                                    .height = command->height,
                                                                    .vsync = command->vsync,
                                                                    .layout = command->layout});
        viewer.Run();
        return 0;
    } catch (const std::exception& failure) {
        *error = failure.what();
        return 1;
    }
}
