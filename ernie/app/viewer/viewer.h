// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/operation.h"
#include "ernie/device/tensor.h"
#include "ernie/io/image.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/scene.h"
#include "ernie/viewer_export.h"

namespace ernie::app {

enum class Dock : uint8_t { Left, Right, RightBottom, Bottom, Floating };

enum class HudItem : uint8_t {
    Fps,
    UiFps,
    DeviceMemory,
    HostMemory,
    Resolution,
    Camera,
    Renderer,
    Selection,
};

enum class Anchor : uint8_t {
    TopLeft,
    Top,
    TopRight,
    Left,
    Center,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
};

struct Placement {
    Anchor anchor = Anchor::BottomLeft;
    float x = 0;
    float y = 0;
    float width = 0;
    float height = 0;
};

struct TextStyle {
    Placement placement;
    Float4 color{1, 1, 1, 1};
    float size = 0;
    bool shadow = true;
    bool background = false;
};

struct MenuItemDesc {
    std::string menu;
    std::string label;
    std::function<void()> action;
    std::string shortcut;
    std::function<bool()> checked;
    std::function<bool()> enabled;
};

struct WindowPosition {
    int x = 0;
    int y = 0;
};

struct Screen {
    std::string name;
    int x = 0;
    int y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t refreshRate = 0;
    float scale = 1;
    bool primary = false;
};

struct ColorbarDesc {
    Placement placement{Anchor::BottomRight, 16, 16, 20, 200};
    Colormap colormap = Colormap::Grayscale;
    float minimum = 0;
    float maximum = 1;
    bool logarithmic = false;
    bool reversed = false;
    std::string label;
    std::string units;
    uint32_t ticks = 5;
};

enum class LightingMode : uint8_t { Stage, Camera, Default };

enum class WorldPick : uint8_t { All, First, Random, Listed, Selected };

enum class WorldLayout : uint8_t { Grid, Overlap };

struct WorldDisplay {
    WorldPick pick = WorldPick::All;
    uint32_t count = 0;
    uint32_t seed = 0;
    std::vector<std::string> worlds;
    bool global = true;
    WorldLayout layout = WorldLayout::Grid;
    std::optional<Double3> spacing;
};

enum class InputAction : uint8_t { Press, Release, Repeat, Move, Scroll };

struct Modifiers {
    bool ctrl = false;
    bool shift = false;
    bool alt = false;
    bool super = false;
};

struct KeyEvent {
    std::string key;
    InputAction action = InputAction::Press;
    Modifiers modifiers;
};

enum class OverlayGesture : uint8_t { Hover, Leave, Press, Drag, Release, Click };

struct OverlayEvent {
    OverlayGesture gesture = OverlayGesture::Hover;
    std::string viewport;
    OverlayHit hit;
    OverlayRay ray;
    float x = 0;
    float y = 0;
    Modifiers modifiers;
};

struct MouseEvent {
    InputAction action = InputAction::Move;
    int button = -1;
    float x = 0;
    float y = 0;
    float scrollX = 0;
    float scrollY = 0;
    bool inViewport = false;
    std::string viewport;
    Modifiers modifiers;
};

struct ViewportRect {
    float x = 0;
    float y = 0;
    float width = 0;
    float height = 0;
};

struct RenderRequest {
    std::string_view product;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct RenderFrame {
    device::Tensor image;
    device::Operation ready;
    device::Tensor overlay;
};

class ERNIE_VIEWER_API RenderDelegate {
  public:
    virtual ~RenderDelegate() = default;
    virtual std::optional<RenderFrame> Render(const RenderRequest& request) = 0;
    virtual std::string_view Name() const = 0;
};

struct ViewportDesc {
    std::string name = "Viewport";
    std::string product;
    std::string aov = "LdrColor";
    std::shared_ptr<RenderDelegate> delegate;
    std::shared_ptr<Renderer> renderer;
    bool hud = true;
    std::vector<HudItem> hudItems = {HudItem::Fps,          HudItem::UiFps,
                                     HudItem::DeviceMemory, HudItem::HostMemory,
                                     HudItem::Resolution,   HudItem::Camera};
    bool toolbar = true;
    double targetRenderFps = 0;
};

struct ViewerDesc {
    std::string product;
    std::string aov = "LdrColor";
    std::string title = "Ernie";
    uint32_t width = 1280;
    uint32_t height = 720;
    bool offscreen = false;
    bool ui = true;
    bool vsync = false;
    std::shared_ptr<RenderDelegate> delegate;
    std::shared_ptr<Renderer> renderer;
    bool hud = true;
    bool toolbar = true;
    bool menuBar = true;
    std::optional<WindowPosition> position;
    int screen = -1;
    bool fullscreen = false;
    double targetRenderFps = 0;
    float fontSize = 0;
    std::vector<HudItem> hudItems = {HudItem::Fps,          HudItem::UiFps,
                                     HudItem::DeviceMemory, HudItem::HostMemory,
                                     HudItem::Resolution,   HudItem::Camera};
    bool builtinPanels = true;
    std::filesystem::path layout;
};

class ViewerImpl;

class ERNIE_VIEWER_API Viewer {
  public:
    explicit Viewer(Ref<device::Device> device, const ViewerDesc& desc = {});
    Viewer(Ref<Scene> scene, const ViewerDesc& desc);
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    bool Open() const;
    void Frame();
    void Run();

    void Show(const device::Tensor& image, const device::Operation& ready = {},
              std::string_view viewport = {});
    void AddViewport(ViewportDesc desc);
    void RemoveViewport(std::string_view name);
    std::vector<std::string> Viewports() const;
    void AddPanel(std::string title, Dock dock, std::function<void()> draw, bool visible = true);
    void ShowPanel(std::string_view title, bool shown);
    uint64_t AddMenuItem(std::string menu, std::string label, std::function<void()> action,
                         std::string shortcut = {});
    uint64_t AddMenuItem(MenuItemDesc item);
    void RemoveMenuItem(uint64_t id);
    void ShowMenuBar(bool shown);
    void ShowMenu(std::string_view menu, bool shown);
    void ShowMenuItem(std::string_view menu, std::string_view label, bool shown);
    void AddOverlay(std::function<void(const ViewportRect&)> draw, std::string_view viewport = {});
    uint64_t AddColorbar(const ColorbarDesc& desc, std::string_view viewport = {});
    void SetColorbar(uint64_t id, const ColorbarDesc& desc);
    uint64_t AddText(std::string text, const TextStyle& style = {}, std::string_view viewport = {});
    void SetText(uint64_t id, std::string text);
    uint64_t AddImage(const device::Tensor& image, const Placement& placement = {},
                      float opacity = 1, std::string_view viewport = {});
    uint64_t AddImage(const std::filesystem::path& file, const Placement& placement = {},
                      float opacity = 1, std::string_view viewport = {});
    void RemoveOverlay(uint64_t id);

    void ShowGui(bool shown);
    void ToggleGui();
    bool GuiShown() const;
    void ShowBaseGui(bool shown);
    void SetFontSize(float pixels);

    uint64_t AddBeginFrameHandler(std::function<void()> handler);
    uint64_t AddEndFrameHandler(std::function<void()> handler);
    uint64_t AddKeyHandler(std::function<bool(const KeyEvent&)> handler);
    uint64_t AddMouseHandler(std::function<bool(const MouseEvent&)> handler);
    uint64_t AddOverlayHandler(std::function<bool(const OverlayEvent&)> handler);
    std::optional<Float2> WorldToScreen(const Double3& point, std::string_view viewport = {}) const;
    std::optional<OverlayRay> ScreenRay(Float2 at, std::string_view viewport = {}) const;
    void RemoveHandler(uint64_t id);

    void ShowHud(bool shown, std::string_view viewport = {});
    void ShowHudItem(HudItem item, bool shown, std::string_view viewport = {});
    void AddHudItem(std::string label, std::function<std::string()> value,
                    std::string_view viewport = {});

    std::vector<Screen> Screens() const;
    void SetWindowPosition(int x, int y);
    void SetWindowSize(uint32_t width, uint32_t height);
    void SetOutputScreen(int screen);
    void EnableFullscreen(bool fullscreen);
    bool Fullscreen() const;

    double RenderFps(std::string_view viewport = {}) const;
    double UiFps() const;
    void SetTargetRenderFps(double fps, std::string_view viewport = {});

    uint32_t Width(std::string_view viewport = {}) const;
    uint32_t Height(std::string_view viewport = {}) const;
    const device::Tensor* Shown(std::string_view viewport = {}) const;
    std::shared_ptr<Renderer> GetRenderer(std::string_view viewport = {}) const;
    void SaveLayout(const std::filesystem::path& path) const;
    void LoadLayout(const std::filesystem::path& path);
    void SetWorldDisplay(const WorldDisplay& display, std::string_view viewport = {});
    const WorldDisplay& GetWorldDisplay(std::string_view viewport = {}) const;
    void SetLighting(LightingMode lighting, std::string_view viewport = {});
    LightingMode Lighting(std::string_view viewport = {}) const;
    void SetAov(std::string aov, std::string_view viewport = {});
    const std::string& Aov(std::string_view viewport = {}) const;
    void SetAovColors(const ColorizeOptions& colors, std::string_view viewport = {});
    void SetAovExposure(float stops, std::string_view viewport = {});
    const std::vector<std::string>& Selected() const;
    void Select(std::vector<std::string> paths);
    void RecordEdit(std::string name, std::function<void()> undo, std::function<void()> redo);
    bool Undo();
    bool Redo();
    std::vector<uint8_t> ReadPixels();

  private:
    std::unique_ptr<ViewerImpl> m_impl;
};

}
