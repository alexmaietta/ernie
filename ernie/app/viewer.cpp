// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer.h"

#include "ernie/app/builtin_panels.h"
#include "ernie/app/create.h"
#include "ernie/app/icons.h"
#include "ernie/app/prim_icons.h"
#include "ernie/app/session.h"
#include "ernie/device/gpu_profile.h"
#include "ernie/device/image.h"
#include "ernie/device/offscreen_target.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/swapchain.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/orbit_controller.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/scene_json.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/session/delegate.h"
#include "ernie/session/resolve.h"
#include "ernie/util/profile.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>
#include <implot.h>
#include <stb_image_write.h>

// After imgui.h, which it needs first.
#include <ImGuizmo.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(__linux__)
#include <unistd.h>
#endif

extern "C" {
extern const unsigned char kErnieFont_inter[];
extern const std::size_t kErnieFont_inter_size;
extern const unsigned char kErnieFont_lucide[];
extern const std::size_t kErnieFont_lucide_size;
}

namespace ernie::app {

namespace {

constexpr VkFormat kViewportFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr const char* kViewportTitle = "Viewport";
constexpr const char* kStatsTitle = "Stats";
constexpr const char* kDockSpace = "ErnieDockSpace";

/// The session's overlay outlining the selection; overlays named with
/// a leading "__" are the viewer's own, which the Overlays panel
/// hides.
constexpr const char* kSelectionOverlay = "__selection";
/// The session overlay of what the viewport's eye menu shows: light
/// and camera gizmos, the grid.
constexpr const char* kViewportOverlay = "__viewport";
/// Seconds between updates of the heads-up display's figures, so they
/// stay readable.
constexpr double kHudRefresh = 0.25;
/// Light gizmos' lamp yellow, selected and not.
constexpr glm::vec4 kLampColor{0.98f, 0.82f, 0.30f, 1.0f};
constexpr glm::vec4 kLampDimColor{0.78f, 0.68f, 0.36f, 0.75f};
/// Camera frusta's grey, and the grid's.
constexpr glm::vec4 kCameraGizmoColor{0.72f, 0.76f, 0.78f, 0.8f};
constexpr glm::vec4 kGridColor{0.55f, 0.58f, 0.60f, 0.35f};

/// Pixels the mouse may move between press and release for a click (a
/// pick) rather than a drag.
constexpr float kClickSlop = 3.0f;
constexpr std::size_t kFrameHistory = 240;

const Path& SessionSettingsPath() {
    static const Path path =
        SessionRootPath().Child("RenderSettings");
    return path;
}

/// GLFW is process-wide: initialised by the first visible viewer,
/// terminated with the last.
class GlfwLibrary {
  public:
    GlfwLibrary() {
        if (users_++ == 0) {
            glfwSetErrorCallback([](int code, const char* message) {
                std::fprintf(stderr, "GLFW %d: %s\n", code, message);
            });
            glfwInitVulkanLoader(LoadVulkanLoader());
            if (glfwInit() == GLFW_FALSE) {
                users_ = 0;
                throw std::runtime_error("Could not initialize GLFW");
            }
        }
    }

    ~GlfwLibrary() {
        if (--users_ == 0) {
            glfwTerminate();
        }
    }

    GlfwLibrary(const GlfwLibrary&) = delete;
    GlfwLibrary& operator=(const GlfwLibrary&) = delete;

  private:
    static inline int users_ = 0;
};

/// The command buffer, fence and acquire semaphore of the one frame
/// kept in flight.
struct FrameResources {
    const VulkanContext& gpu;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;

    explicit FrameResources(const VulkanContext& context)
        : gpu(context) {
        VkCommandPoolCreateInfo poolInfo{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = gpu.QueueFamily();
        CheckVk(vkCreateCommandPool(gpu.Device(), &poolInfo, nullptr,
                                    &pool),
                "vkCreateCommandPool");
        VkCommandBufferAllocateInfo allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        CheckVk(vkAllocateCommandBuffers(gpu.Device(), &allocate,
                                         &commands),
                "vkAllocateCommandBuffers");
        VkFenceCreateInfo fenceInfo{
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        CheckVk(
            vkCreateFence(gpu.Device(), &fenceInfo, nullptr, &done),
            "vkCreateFence");
        VkSemaphoreCreateInfo semaphoreInfo{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        CheckVk(vkCreateSemaphore(gpu.Device(), &semaphoreInfo,
                                  nullptr, &acquired),
                "vkCreateSemaphore");
    }

    ~FrameResources() {
        vkDeviceWaitIdle(gpu.Device());
        vkDestroySemaphore(gpu.Device(), acquired, nullptr);
        vkDestroyFence(gpu.Device(), done, nullptr);
        vkDestroyCommandPool(gpu.Device(), pool, nullptr);
    }

    FrameResources(const FrameResources&) = delete;
    FrameResources& operator=(const FrameResources&) = delete;

    void Wait() const {
        vkWaitForFences(gpu.Device(), 1, &done, VK_TRUE, UINT64_MAX);
    }
};

ImVec4 ToImVec4(const glm::vec4& colour) {
    return {colour.r, colour.g, colour.b, colour.a};
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!std::filesystem::is_regular_file(path) || !file) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return {std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()};
}

/// The last path component, for labels.
std::string Label(const Path& path) {
    return std::string(path.Name());
}

const char* LightingName(Lighting lighting) {
    switch (lighting) {
    case Lighting::Stage:
        return "Stage Lights";
    case Lighting::Camera:
        return "Camera Light";
    case Lighting::Default:
        return "Default Light";
    }
    return "";
}

/// Draws the icon glyph `icon` (one UTF-8 character) centred on
/// `centre` by its ink, not its advance, which icon fonts leave
/// uneven.
void DrawIconCentred(ImDrawList* draw, const char* icon,
                     ImVec2 centre, ImU32 colour) {
    unsigned int codepoint = 0;
    ImTextCharFromUtf8(&codepoint, icon, nullptr);
    const ImFontGlyph* glyph =
        ImGui::GetFontBaked()->FindGlyphNoFallback(
            static_cast<ImWchar>(codepoint));
    ImVec2 at;
    if (glyph != nullptr) {
        at = ImVec2(centre.x - 0.5f * (glyph->X0 + glyph->X1),
                    centre.y - 0.5f * (glyph->Y0 + glyph->Y1));
    } else {
        const ImVec2 size = ImGui::CalcTextSize(icon);
        at = ImVec2(centre.x - 0.5f * size.x,
                    centre.y - 0.5f * size.y);
    }
    draw->AddText(ImVec2(std::round(at.x), std::round(at.y)), colour,
                  icon);
}

/// A backend's name as menus show it: "raster" as "Raster".
std::string DisplayName(std::string name) {
    if (!name.empty()) {
        name[0] = static_cast<char>(
            std::toupper(static_cast<unsigned char>(name[0])));
    }
    return name;
}

/// `bytes` as Kit's heads-up display shows memory: GiB from one GiB,
/// else MiB.
std::string FormatBytes(std::uint64_t bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * kMiB;
    char text[32];
    const double value = static_cast<double>(bytes);
    if (value >= kGiB) {
        std::snprintf(text, sizeof(text), "%.2f GiB", value / kGiB);
    } else {
        std::snprintf(text, sizeof(text), "%.0f MiB", value / kMiB);
    }
    return text;
}

/// `count` with commas between thousands.
std::string FormatCount(std::uint64_t count) {
    std::string digits = std::to_string(count);
    for (std::ptrdiff_t at =
             static_cast<std::ptrdiff_t>(digits.size()) - 3;
         at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return digits;
}

/// The process's resident memory, where the platform tells.
std::optional<std::uint64_t> HostResidentBytes() {
#if defined(__linux__)
    // statm: total and resident pages.
    std::ifstream statm("/proc/self/statm");
    std::uint64_t total = 0;
    std::uint64_t resident = 0;
    if (statm >> total >> resident) {
        return resident *
               static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    }
#endif
    return std::nullopt;
}

const char* WireframeName(std::optional<Wireframe> mode) {
    if (!mode) {
        return "Wireframe";
    }
    switch (*mode) {
    case Wireframe::Off:
        return "Shaded";
    case Wireframe::Overlay:
        return "Wireframe Overlay";
    case Wireframe::HiddenLine:
        return "Hidden Line";
    }
    return "";
}

/// Built-in panel showing the device, frame times, memory and the
/// last render error.
class StatsPanel final : public Panel {
  public:
    StatsPanel() : Panel(kStatsTitle, Dock::Bottom) {}

    void Record(float milliseconds) {
        times_.push_back(milliseconds);
        if (times_.size() > kFrameHistory) {
            times_.pop_front();
        }
    }

    void Draw(Viewer& viewer) override;

  private:
    std::deque<float> times_;
};

/// Built-in panel listing the scene's overlays (debug drawing
/// layers), each with its item counts and toggles to show or clear
/// it.
class OverlaysPanel final : public Panel {
  public:
    OverlaysPanel() : Panel("Overlays", Dock::Bottom) {}

    void Draw(Viewer& viewer) override {
        Scene& scene = viewer.GetScene();
        const std::vector<std::string> names = scene.OverlayNames();
        if (std::none_of(names.begin(), names.end(),
                         [](const std::string& name) {
                             return !name.starts_with("__");
                         })) {
            ImGui::TextDisabled(
                "No overlays. Draw some with Scene::GetOverlay.");
            return;
        }
        for (const std::string& name : names) {
            if (name.starts_with("__")) {
                continue;
            }
            Overlay& overlay = scene.GetOverlay(name);
            ImGui::PushID(name.c_str());
            bool visible = overlay.IsVisible();
            if (ImGui::Checkbox(name.c_str(), &visible)) {
                overlay.SetVisible(visible);
            }
            ImGui::SameLine();
            ImGui::TextDisabled(
                "%zu lines, %zu triangles, %zu sprites",
                overlay.Segments().size(), overlay.Fills().size(),
                overlay.Sprites().size());
            ImGui::SameLine(ImGui::GetContentRegionMax().x -
                            ImGui::CalcTextSize("Clear").x -
                            ImGui::GetStyle().FramePadding.x * 2.0f);
            if (ImGui::SmallButton("Clear")) {
                overlay.Clear();
            }
            ImGui::PopID();
        }
    }
};

}  // namespace

struct Viewer::State {
    Viewer& self;
    Scene& scene;
    ViewerOptions options;

    std::optional<GlfwLibrary> glfw;
    GLFWwindow* window = nullptr;
    std::unique_ptr<VulkanContext> gpu;
    std::unique_ptr<Swapchain> swapchain;
    std::unique_ptr<OffscreenTarget> uiTarget;
    /// Two frames in flight: the host records one while the device
    /// draws the other. Each frame waits for the one two before it,
    /// which also finishes everything submitted earlier, such as the
    /// render delegate's frame in the upload slot it takes again (it
    /// alternates two per viewport frame).
    std::array<std::optional<FrameResources>, 2> frameResources;
    FrameResources* frame =
        nullptr;  ///< This frame's, from `frameResources`.
    std::size_t frameSlot = 0;
    ImGuiContext* imgui = nullptr;
    ImPlotContext* implot = nullptr;
    bool imguiVulkan = false;
    bool imguiGlfw = false;
    std::unique_ptr<RenderDelegate> delegate;
    std::string renderer;

    std::unique_ptr<OffscreenTarget> viewport;
    std::unique_ptr<Sampler> sampler;
    VkDescriptorSet viewportTexture = VK_NULL_HANDLE;
    VkExtent2D requestedViewport{640, 360};

    OrbitController controller;
    /// Which of the camera menu's orthographic views (Top, Front,
    /// Right) the perspective camera shows, or -1 for perspective.
    int orthographicView = -1;
    Path activeCamera;
    Path settingsSource;
    std::map<std::string, std::uint64_t> mirrored;
    std::string viewVar = "LdrColor";
    int viewBand = -1;
    Viewer::Exposure exposure = Viewer::Exposure::Camera;
    double compensation = 0.0;
    std::optional<double>
        autoStops;  ///< The automatic exposure's stops, as it adapts.

    bool AutoExposing() const {
        return exposure == Viewer::Exposure::Auto ||
               (exposure == Viewer::Exposure::Default &&
                activeCamera == PerspectiveCamera());
    }

    /// The stops of exposure the viewport adds to the camera's: the
    /// compensation, the scene settings' own, and when exposing
    /// automatically the stops that bring the metered luminance to
    /// middle grey, approached smoothly.
    double ExposureStops() {
        // Middle grey: the linear display value an average scene is
        // keyed to.
        constexpr double kMiddleGrey = 0.18;
        // Seconds the automatic exposure takes to cover most (63%) of
        // a change.
        constexpr double kAdaptation = 0.25;
        double stops = compensation + RenderSettings::OrFallbacks(
                                          scene, settingsSource)
                                          .GetExposureCompensation();
        if (!AutoExposing()) {
            autoStops.reset();
            return stops;
        }
        if (const std::optional<double> luminance =
                delegate->MeteredLuminance();
            luminance && scene.Has(activeCamera)) {
            const double scale =
                CameraParamsOf(scene, activeCamera, ViewportWidth(),
                               ViewportHeight())
                    .LinearExposureScale();
            const double target =
                std::log2(kMiddleGrey / (*luminance * scale));
            autoStops =
                autoStops ? *autoStops +
                                (target - *autoStops) *
                                    (1.0 - std::exp(-frameInfo.delta /
                                                    kAdaptation))
                          : target;
        }
        return stops + autoStops.value_or(0.0);
    }
    Lighting lighting = Lighting::Stage;
    std::optional<Wireframe> wireframe;
    ViewportDisplay display;
    std::vector<std::string>
        hudLines;  ///< The heads-up display, as last refreshed.
    double hudTime = -kHudRefresh;  ///< When `hudLines` was
                                    ///< refreshed, in viewer seconds.
    Theme theme;
    float contentScale = 1.0f;
    std::vector<Path> selection;
    EditHistory history;
    std::deque<LogEntry> log;
    Viewer::Gizmo gizmo = Viewer::Gizmo::Translate;
    bool gizmoLocal = false;
    bool gizmoDrawn = false;  ///< Whether the last frame drew the
                              ///< gizmo, so it may take the mouse.
    /// Prims' local bounds (`LocalBoundsOf`), with the attribute
    /// versions they were read at.
    std::map<Path, std::pair<std::uint64_t, std::optional<Bounds3>>>
        boundsCache;

    /// `LocalBoundsOf(path)`, read again only when its geometry
    /// attributes change.
    std::optional<Bounds3> LocalBounds(const Path& path) {
        std::uint64_t key =
            std::hash<std::string>()(scene.TypeOf(path));
        for (const std::string_view attribute :
             {tokens::kExtent, tokens::kPoints, tokens::kSize,
              tokens::kGeomRadius, tokens::kGeomHeight,
              tokens::kGeomWidth, tokens::kGeomLength,
              tokens::kAxis}) {
            key = key * 1099511628211ull +
                  scene.AttributeVersion(path, attribute);
        }
        auto& [cachedKey, bounds] = boundsCache[path];
        if (cachedKey != key || key == 0) {
            cachedKey = key;
            bounds = LocalBoundsOf(scene, path);
        }
        return bounds;
    }

    enum class Stage { Frame, Ui, PreRender, PostRender };
    // Each registration keeps the scope it was added in
    // (`Viewer::SetScope`).
    struct FrameCallback {
        std::uint64_t id;
        Stage stage;
        std::function<void(Viewer&, const FrameInfo&)> callback;
        std::uint64_t scope = 0;
    };
    struct SelectionCallback {
        std::uint64_t id;
        std::function<void(Viewer&, std::span<const Path>)> callback;
        std::uint64_t scope = 0;
    };
    struct MenuItem {
        std::string menu;
        std::string label;
        std::function<void(Viewer&)> action;
        std::uint64_t scope = 0;
    };
    std::vector<FrameCallback> frameCallbacks;
    std::vector<SelectionCallback> selectionCallbacks;
    std::uint64_t nextCallback = 1;
    /// The registration scope now (`Viewer::SetScope`), the next new
    /// one, and the scope of each added panel by title.
    std::uint64_t scope = 0;
    std::uint64_t nextScope = 1;
    std::map<std::string, std::uint64_t, std::less<>> panelScopes;
    /// What the application chose since the scene was last cleared
    /// (`ClearScene`), which `ResetSession` keeps.
    bool cameraChosen = false;
    bool lightingChosen = false;
    bool settingsChosen = false;
    std::vector<std::unique_ptr<Panel>> panels;
    FileDialog fileDialog;
    /// The scene file last opened or saved, where File > Save Scene
    /// As starts.
    std::filesystem::path scenePath;
    /// Panels to bring to the front (`FocusPanel`), and those drawn
    /// at least once.
    std::vector<std::string> focusRequests;
    std::set<std::string, std::less<>> drawnPanels;
    StatsPanel* stats = nullptr;
    Panel* profilerPanel = nullptr;
    bool showControls = false;  ///< Help > Keyboard and Mouse.
    bool showAbout = false;     ///< Help > About Ernie.
    std::size_t builtInPanels =
        0;  ///< The first panels, which removing only hides.
    std::vector<MenuItem> menuItems;
    struct ViewportToggle {
        std::uint64_t id;
        std::string label;
        bool shown;
        std::string overlay;
        std::function<void(Viewer&, bool)> changed;
        std::uint64_t scope = 0;
    };
    struct HudItem {
        std::uint64_t id;
        std::string label;
        std::function<std::string(Viewer&)> value;
        bool shown;
        std::uint64_t scope = 0;
    };
    struct ViewportMenuItem {
        std::uint64_t id;
        std::string label;
        std::function<void(Viewer&)> action;
        std::function<bool(const Viewer&)> checked;
        std::uint64_t scope = 0;
    };
    std::vector<ViewportToggle> viewportToggles;
    std::vector<HudItem> hudItems;
    std::vector<ViewportMenuItem> viewportMenuItems;

    /// The overlays the viewport draws: the session's selection and
    /// eye-menu overlays, those of applications' toggles while shown,
    /// and unless the eye menu hides them, the scene's others, or
    /// those the settings name when they do.
    std::string ShownOverlays(bool authored) const {
        std::set<std::string, std::less<>> toggled;
        std::string shown =
            std::string(kSelectionOverlay) + " " + kViewportOverlay;
        for (const ViewportToggle& toggle : viewportToggles) {
            if (toggle.overlay.empty()) {
                continue;
            }
            toggled.insert(toggle.overlay);
            if (toggle.shown) {
                shown += " " + toggle.overlay;
            }
        }
        if (!display.overlays) {
            return shown;
        }
        std::vector<std::string> names = scene.OverlayNames();
        if (authored) {
            // The settings' names, separated by commas or spaces,
            // with "*" for all.
            std::string list =
                RenderSettings::OrFallbacks(scene, settingsSource)
                    .GetOverlays();
            std::replace(list.begin(), list.end(), ',', ' ');
            std::istringstream words(list);
            std::vector<std::string> named(
                std::istream_iterator<std::string>(words), {});
            if (std::find(named.begin(), named.end(), "*") ==
                named.end()) {
                names = std::move(named);
            }
        }
        for (const std::string& name : names) {
            if (name != kSelectionOverlay &&
                name != kViewportOverlay && !toggled.contains(name)) {
                shown += " " + name;
            }
        }
        return shown;
    }

    std::uint64_t frames = 0;
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last = start;
    bool open = true;
    bool layoutReady = false;
    std::map<Dock, ImGuiID> dockNodes;
    bool userLayout = false;
    FrameInfo frameInfo;
    std::optional<std::string> pendingLayout;
    std::string renderError;

    struct CaptureRequest {
        std::filesystem::path path;
        std::string
            var;  ///< The var to write; empty for the shown one.
        bool rendered = false;
    };
    std::optional<CaptureRequest> capture;

    int framebufferWidth = 0;
    int framebufferHeight = 0;

    State(Viewer& viewer, Scene& s, const ViewerOptions& o)
        : self(viewer), scene(s), options(o), theme(o.theme) {}

    ~State() {
        if (gpu) {
            vkDeviceWaitIdle(gpu->Device());
        }
        if (viewportTexture != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(viewportTexture);
        }
        viewport.reset();
        sampler.reset();
        delegate.reset();
        if (imguiVulkan) {
            ImGui_ImplVulkan_Shutdown();
        }
        if (imguiGlfw) {
            ImGui_ImplGlfw_Shutdown();
        }
        if (implot != nullptr) {
            ImPlot::DestroyContext(implot);
        }
        if (imgui != nullptr) {
            ImGui::DestroyContext(imgui);
        }
        frame = nullptr;
        for (std::optional<FrameResources>& resources :
             frameResources) {
            resources.reset();
        }
        uiTarget.reset();
        swapchain.reset();
        gpu.reset();
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }

    State(const State&) = delete;
    State& operator=(const State&) = delete;

    void OpenWindow() {
        glfw.emplace();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window =
            glfwCreateWindow(static_cast<int>(options.width),
                             static_cast<int>(options.height),
                             options.title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            throw std::runtime_error("Could not create a window");
        }
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        glfwGetWindowContentScale(window, &scaleX, &scaleY);
        contentScale = std::max(scaleX, 1.0f);
    }

    void CreateGpu() {
        VulkanContextOptions gpuOptions;
        gpuOptions.validation = options.validation;
        gpuOptions.preferredDevice = options.device;
        if (window != nullptr) {
            std::uint32_t count = 0;
            const char** names =
                glfwGetRequiredInstanceExtensions(&count);
            if (names == nullptr) {
                throw std::runtime_error(
                    "GLFW cannot present with Vulkan on this system");
            }
            gpuOptions.instanceExtensions.assign(names,
                                                 names + count);
            gpuOptions.deviceExtensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        }
        gpu = VulkanContext::Create(gpuOptions);
        if (window != nullptr) {
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            const VkResult result = glfwCreateWindowSurface(
                gpu->Instance(), window, nullptr, &surface);
            CheckVk(result, "glfwCreateWindowSurface");
            glfwGetFramebufferSize(window, &framebufferWidth,
                                   &framebufferHeight);
            swapchain = std::make_unique<Swapchain>(
                *gpu, surface,
                static_cast<std::uint32_t>(
                    std::max(framebufferWidth, 1)),
                static_cast<std::uint32_t>(
                    std::max(framebufferHeight, 1)),
                options.vsync);
        } else {
            framebufferWidth = static_cast<int>(options.width);
            framebufferHeight = static_cast<int>(options.height);
            uiTarget = std::make_unique<OffscreenTarget>(
                *gpu, options.width, options.height);
        }
        for (std::optional<FrameResources>& resources :
             frameResources) {
            resources.emplace(*gpu);
        }
        frame = &*frameResources[0];
        delegate = CreateRenderDelegate(options.renderer, *gpu,
                                        kViewportFormat);
        renderer = options.renderer;
        sampler = std::make_unique<Sampler>(
            *gpu, SamplerDesc{VK_FILTER_LINEAR});
    }

    VkFormat UiFormat() const {
        return swapchain ? swapchain->Format()
                         : uiTarget->ColorFormat();
    }

    void CreateImGui() {
        IMGUI_CHECKVERSION();
        imgui = ImGui::CreateContext();
        implot = ImPlot::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable |
                          ImGuiConfigFlags_NavEnableKeyboard;
        LoadFonts();
        ApplyTheme(theme, contentScale, ImGui::GetStyle());
        if (window != nullptr) {
            ImGui_ImplGlfw_InitForVulkan(window, true);
            imguiGlfw = true;
        } else {
            io.DisplaySize =
                ImVec2(static_cast<float>(options.width),
                       static_cast<float>(options.height));
            io.BackendPlatformName = "ernie_hidden";
        }
        const VkFormat format = UiFormat();
        ImGui_ImplVulkan_InitInfo info{};
        info.ApiVersion = VK_API_VERSION_1_3;
        info.Instance = gpu->Instance();
        info.PhysicalDevice = gpu->PhysicalDevice();
        info.Device = gpu->Device();
        info.QueueFamily = gpu->QueueFamily();
        info.Queue = gpu->Queue();
        info.DescriptorPoolSize = 64;
        info.MinImageCount =
            swapchain ? swapchain->MinImageCount() : 2;
        info.ImageCount = swapchain ? swapchain->ImageCount() : 2;
        info.UseDynamicRendering = true;
        info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        info.PipelineInfoMain.PipelineRenderingCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        info.PipelineInfoMain.PipelineRenderingCreateInfo
            .colorAttachmentCount = 1;
        info.PipelineInfoMain.PipelineRenderingCreateInfo
            .pColorAttachmentFormats = &format;
        ImGui_ImplVulkan_Init(&info);
        imguiVulkan = true;
        if (!options.layout.empty()) {
            pendingLayout = ReadFile(options.layout);
        }
    }

    void LoadFonts() {
        ImGuiIO& io = ImGui::GetIO();
        static const ImWchar kIconRange[] = {
            static_cast<ImWchar>(icons::kFirst),
            static_cast<ImWchar>(icons::kLast), 0};
        ImFontConfig text;
        text.FontDataOwnedByAtlas = false;
        // Inter has glyphs of its own in the private use area, which
        // would hide the icons'.
        text.GlyphExcludeRanges = kIconRange;
        std::snprintf(text.Name, sizeof(text.Name), "Inter");
        io.Fonts->AddFontFromMemoryTTF(
            const_cast<unsigned char*>(kErnieFont_inter),
            static_cast<int>(kErnieFont_inter_size), theme.fontSize,
            &text);
        ImFontConfig iconConfig;
        iconConfig.FontDataOwnedByAtlas = false;
        iconConfig.MergeMode = true;
        iconConfig.GlyphMinAdvanceX = theme.iconSize;
        iconConfig.GlyphOffset = ImVec2(0.0f, 2.0f);
        std::snprintf(iconConfig.Name, sizeof(iconConfig.Name),
                      "Lucide");
        io.Fonts->AddFontFromMemoryTTF(
            const_cast<unsigned char*>(kErnieFont_lucide),
            static_cast<int>(kErnieFont_lucide_size), theme.iconSize,
            &iconConfig, kIconRange);
    }

    void CreateSession() {
        settingsSource = options.settings;
        if (!settingsSource.IsEmpty() &&
            (!scene.Has(settingsSource) ||
             scene.TypeOf(settingsSource) !=
                 tokens::kRenderSettings)) {
            throw std::invalid_argument(
                "'" + settingsSource.String() +
                "' is not a RenderSettings prim");
        }
        if (settingsSource.IsEmpty()) {
            settingsSource = FirstUserSettings();
        }
        activeCamera = PerspectiveCamera();
        lighting = HasStageLights(scene) ? Lighting::Stage
                                         : Lighting::Default;
        delegate->Sync(scene);
        FrameBounds(SceneBounds());
    }

    /// Starts over on the scene's present content, as on opening: no
    /// selection or undo steps, the first RenderSettings, the
    /// perspective view framing the scene, lighting by its lights.
    void ResetSession() {
        SetSelection({});
        history.Reset();
        if (!settingsChosen || !scene.Has(settingsSource)) {
            settingsSource = FirstUserSettings();
        }
        if (!lightingChosen) {
            lighting = HasStageLights(scene) ? Lighting::Stage
                                             : Lighting::Default;
        }
        delegate->Sync(scene);
        if (!cameraChosen || !scene.Has(activeCamera)) {
            orthographicView = -1;
            activeCamera = PerspectiveCamera();
            FrameBounds(SceneBounds());
        }
    }

    /// File > Open Scene or Save Scene As, once a file is chosen:
    /// logs what it did, or the error and brings the Log panel to the
    /// front.
    void SceneFileAction(const std::filesystem::path& path,
                         bool save) {
        try {
            if (save) {
                self.SaveScene(path);
            } else {
                self.OpenScene(path);
            }
            self.Log(LogLevel::Info, (save ? "Saved the scene to "
                                           : "Opened the scene ") +
                                         path.string());
        } catch (const std::exception& error) {
            self.Log(LogLevel::Error, error.what());
            self.FocusPanel("Log");
        }
    }

    Path FirstUserSettings() const {
        for (const Path& path :
             scene.Prims(tokens::kRenderSettings)) {
            if (!IsSessionPath(path)) {
                return path;
            }
        }
        return {};
    }

    std::optional<Bounds3> SceneBounds() const {
        return delegate->SceneBounds();
    }

    /// Aims the controller at `bounds` (a unit box at the origin
    /// without any) and writes the active camera.
    void FrameBounds(const std::optional<Bounds3>& bounds) {
        const Bounds3 box = bounds.value_or(
            Bounds3{glm::vec3(-1.0f), glm::vec3(1.0f)});
        controller.Frame(box.first, box.second);
        WriteActiveCamera();
    }

    void WriteActiveCamera() {
        if (activeCamera == PerspectiveCamera()) {
            // The camera's lens may have changed (the Camera panel):
            // framing follows it. An orthographic view's aperture is
            // its height, not a field of view.
            if (scene.Has(activeCamera) && orthographicView < 0) {
                const CameraParams camera =
                    CameraParamsOf(scene, activeCamera,
                                   ViewportWidth(), ViewportHeight());
                if (!camera.orthographic) {
                    controller.verticalFov =
                        static_cast<float>(camera.VerticalFov());
                }
            }
            AuthorControllerCamera(scene, activeCamera, controller,
                                   ViewportWidth(), ViewportHeight(),
                                   orthographicView >= 0);
        } else {
            scene.SetLocalTransform(
                activeCamera, LocalFromWorld(scene, activeCamera,
                                             controller.Transform()));
        }
    }

    std::uint32_t ViewportWidth() const {
        return viewport ? viewport->Extent().width
                        : requestedViewport.width;
    }

    std::uint32_t ViewportHeight() const {
        return viewport ? viewport->Extent().height
                        : requestedViewport.height;
    }

    void SetActive(const Path& camera) {
        activeCamera = camera;
        if (camera != PerspectiveCamera()) {
            const float fov = static_cast<float>(
                CameraParamsOf(scene, camera, ViewportWidth(),
                               ViewportHeight())
                    .VerticalFov());
            controller =
                ControllerFromTransform(scene.WorldTransform(camera),
                                        controller.distance, fov);
        } else if (scene.Has(camera)) {
            const CameraParams current = CameraParamsOf(
                scene, camera, ViewportWidth(), ViewportHeight());
            // An orthographic view's aperture is its height: the
            // controller keeps its own field of view.
            controller = ControllerFromTransform(
                current.transform, controller.distance,
                current.orthographic
                    ? controller.verticalFov
                    : static_cast<float>(current.VerticalFov()));
        }
    }

    /// Falls back to the session camera and the scene's first
    /// settings when the scene removed what the viewer was using.
    void ValidateSession() {
        if (!scene.Has(PerspectiveCamera())) {
            AuthorControllerCamera(scene, PerspectiveCamera(),
                                   controller, ViewportWidth(),
                                   ViewportHeight());
        }
        if (!scene.Has(activeCamera) ||
            scene.TypeOf(activeCamera) != tokens::kCamera) {
            activeCamera = PerspectiveCamera();
        }
        if (!settingsSource.IsEmpty() &&
            (!scene.Has(settingsSource) ||
             scene.TypeOf(settingsSource) !=
                 tokens::kRenderSettings)) {
            settingsSource = FirstUserSettings();
        }
    }

    void ResizeSwapchain(int width, int height) {
        framebufferWidth = width;
        framebufferHeight = height;
        if (width > 0 && height > 0) {
            swapchain->Resize(static_cast<std::uint32_t>(width),
                              static_cast<std::uint32_t>(height));
            ImGui_ImplVulkan_SetMinImageCount(
                swapchain->MinImageCount());
        }
    }

    void ResizeViewport() {
        const VkExtent2D want{std::max(requestedViewport.width, 1u),
                              std::max(requestedViewport.height, 1u)};
        if (viewport && viewport->Extent().width == want.width &&
            viewport->Extent().height == want.height) {
            return;
        }
        vkDeviceWaitIdle(gpu->Device());
        if (viewportTexture != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(viewportTexture);
            viewportTexture = VK_NULL_HANDLE;
        }
        viewport = std::make_unique<OffscreenTarget>(
            *gpu, want.width, want.height, kViewportFormat);
        viewportTexture = ImGui_ImplVulkan_AddTexture(
            sampler->Handle(), viewport->Color().View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (activeCamera == PerspectiveCamera()) {
            WriteActiveCamera();
        }
    }

    void SetSelection(std::vector<Path> paths) {
        if (paths == selection) {
            return;
        }
        selection = std::move(paths);
        const auto callbacks = selectionCallbacks;
        for (const SelectionCallback& entry : callbacks) {
            entry.callback(self, selection);
        }
    }

    void BuildDefaultLayout(ImGuiID dockSpace) {
        ImGui::DockBuilderRemoveNode(dockSpace);
        ImGui::DockBuilderAddNode(dockSpace,
                                  ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(
            dockSpace, ImGui::GetMainViewport()->WorkSize);
        dockNodes.clear();
        // Hidden panels take no room; they dock where they ask when
        // shown again.
        const auto wants = [&](Dock dock) {
            return std::any_of(panels.begin(), panels.end(),
                               [dock](const auto& panel) {
                                   return panel->IsVisible() &&
                                          panel->GetDock() == dock;
                               });
        };
        const auto share = [](float fraction) {
            return std::clamp(fraction, 0.02f, 0.98f);
        };
        const DockSizes& sizes = options.docks;
        ImGuiID center = dockSpace;
        if (wants(Dock::Left)) {
            dockNodes[Dock::Left] = ImGui::DockBuilderSplitNode(
                center, ImGuiDir_Left, share(sizes.left), nullptr,
                &center);
        }
        // The right column: `Right` above `RightLower`, either taking
        // all of it alone.
        if (wants(Dock::Right) || wants(Dock::RightLower)) {
            ImGuiID column = ImGui::DockBuilderSplitNode(
                center, ImGuiDir_Right, share(sizes.right), nullptr,
                &center);
            if (wants(Dock::Right) && wants(Dock::RightLower)) {
                dockNodes[Dock::RightLower] =
                    ImGui::DockBuilderSplitNode(
                        column, ImGuiDir_Down,
                        share(sizes.rightLower), nullptr, &column);
                dockNodes[Dock::Right] = column;
            } else {
                dockNodes[wants(Dock::Right) ? Dock::Right
                                             : Dock::RightLower] =
                    column;
            }
        }
        if (wants(Dock::Bottom)) {
            dockNodes[Dock::Bottom] = ImGui::DockBuilderSplitNode(
                center, ImGuiDir_Down, share(sizes.bottom), nullptr,
                &center);
        }
        ImGui::DockBuilderDockWindow(kViewportTitle, center);
        for (const auto& panel : panels) {
            const auto node = dockNodes.find(panel->GetDock());
            if (panel->IsVisible() && node != dockNodes.end()) {
                ImGui::DockBuilderDockWindow(panel->Title().c_str(),
                                             node->second);
            }
        }
        ImGui::DockBuilderFinish(dockSpace);
    }

    void DrawMenuBar() {
        if (!ImGui::BeginMainMenuBar()) {
            return;
        }
        std::vector<std::string> menus = {"File", "Edit", "Create",
                                          "Window"};
        for (const MenuItem& item : menuItems) {
            if (std::find(menus.begin(), menus.end(), item.menu) ==
                    menus.end() &&
                item.menu != "Help") {
                menus.push_back(item.menu);
            }
        }
        menus.push_back("Help");  // last, as in Kit
        for (const std::string& menu : menus) {
            if (!ImGui::BeginMenu(menu.c_str())) {
                continue;
            }
            if (menu == "Edit") {
                const std::string undo =
                    "Undo " + history.UndoLabel();
                const std::string redo =
                    "Redo " + history.RedoLabel();
                if (ImGui::MenuItem(undo.c_str(), "Ctrl+Z", false,
                                    history.CanUndo())) {
                    self.Undo();
                }
                if (ImGui::MenuItem(redo.c_str(), "Ctrl+Shift+Z",
                                    false, history.CanRedo())) {
                    self.Redo();
                }
                ImGui::Separator();
                const std::pair<const char*, Viewer::Gizmo> modes[] =
                    {{"No Gizmo", Viewer::Gizmo::None},
                     {"Translate", Viewer::Gizmo::Translate},
                     {"Rotate", Viewer::Gizmo::Rotate},
                     {"Scale", Viewer::Gizmo::Scale}};
                const char* keys[] = {"Q", "W", "E", "R"};
                for (std::size_t i = 0; i < std::size(modes); ++i) {
                    if (ImGui::MenuItem(modes[i].first, keys[i],
                                        gizmo == modes[i].second)) {
                        gizmo = modes[i].second;
                    }
                }
                ImGui::MenuItem("Gizmo in Local Axes", nullptr,
                                &gizmoLocal);
                ImGui::Separator();
                if (ImGui::MenuItem("Duplicate", "Ctrl+D", false,
                                    !selection.empty())) {
                    self.Duplicate(std::vector<Path>(selection));
                }
                if (ImGui::MenuItem("Delete", "Del", false,
                                    !selection.empty())) {
                    self.Delete(std::vector<Path>(selection));
                }
                if (ImGui::MenuItem("Frame Selection", "F", false,
                                    !selection.empty())) {
                    self.Frame(selection);
                }
            }
            if (menu == "File") {
                for (const bool save : {false, true}) {
                    if (ImGui::MenuItem(save ? "Save Scene As..."
                                             : "Open Scene...")) {
                        self.OpenFileDialog(
                            {save ? "Save Scene As" : "Open Scene",
                             save,
                             {{"Scenes", {".json"}}},
                             scenePath},
                            [this, save](
                                Viewer&,
                                const std::filesystem::path& path) {
                                SceneFileAction(path, save);
                            });
                    }
                }
                ImGui::Separator();
            }
            if (menu == "Help") {
                ImGui::MenuItem("Keyboard and Mouse", nullptr,
                                &showControls);
                ImGui::MenuItem("About Ernie", nullptr, &showAbout);
            }
            if (menu == "Create") {
                const char* group = nullptr;
                for (const CreatableType& creatable :
                     CreatableTypes()) {
                    if (group != nullptr &&
                        std::string_view(group) != creatable.group) {
                        ImGui::Separator();
                    }
                    group = creatable.group;
                    if (ImGui::MenuItem(creatable.label)) {
                        try {
                            self.Create(creatable.type);
                        } catch (const std::exception& error) {
                            self.Log(LogLevel::Error, error.what());
                        }
                    }
                }
            }
            if (menu == "Window") {
                for (const auto& panel : panels) {
                    bool visible = panel->IsVisible();
                    if (ImGui::MenuItem(panel->Title().c_str(),
                                        nullptr, &visible)) {
                        panel->SetVisible(visible);
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Reset Layout")) {
                    layoutReady = false;
                    userLayout = false;
                }
            }
            bool first = true;
            for (std::size_t i = 0; i < menuItems.size(); ++i) {
                if (menuItems[i].menu != menu) {
                    continue;
                }
                if (first && (menu == "Window" || menu == "File")) {
                    if (menu == "Window") {
                        ImGui::Separator();
                    }
                    first = false;
                }
                if (ImGui::MenuItem(menuItems[i].label.c_str())) {
                    const auto action = menuItems[i].action;
                    action(self);
                }
            }
            if (menu == "File") {
                if (!first) {
                    ImGui::Separator();
                }
                if (ImGui::MenuItem("Close", "Esc")) {
                    open = false;
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    void DrawViewport() {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                            ImVec2(0.0f, 0.0f));
        const bool visible =
            ImGui::Begin(kViewportTitle, nullptr,
                         ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleVar();
        if (!visible) {
            ImGui::End();
            return;
        }
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 size = ImGui::GetContentRegionAvail();
        requestedViewport = {
            static_cast<std::uint32_t>(std::max(size.x, 1.0f)),
            static_cast<std::uint32_t>(std::max(size.y, 1.0f))};
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(
            origin, ImVec2(origin.x + size.x, origin.y + size.y),
            ImGui::GetColorU32(ToImVec4(theme.background)));
        if (viewportTexture != VK_NULL_HANDLE) {
            draw->AddImage(
                ImTextureRef(
                    reinterpret_cast<ImTextureID>(viewportTexture)),
                origin, ImVec2(origin.x + size.x, origin.y + size.y));
        }
        // The gizmo takes the mouse over its handles and while
        // dragging; elsewhere the viewport navigates, and a click
        // picks.
        const bool gizmoBusy =
            gizmoDrawn && (ImGuizmo::IsOver() || ImGuizmo::IsUsing());
        bool hovered = false;
        bool active = false;
        if (!gizmoBusy) {
            ImGui::SetNextItemAllowOverlap();
            ImGui::InvisibleButton(
                "##viewport",
                ImVec2(std::max(size.x, 1.0f),
                       std::max(size.y, 1.0f)),
                ImGuiButtonFlags_MouseButtonLeft |
                    ImGuiButtonFlags_MouseButtonRight |
                    ImGuiButtonFlags_MouseButtonMiddle);
            hovered = ImGui::IsItemHovered();
            active = ImGui::IsItemActive();
            const ImGuiIO& io = ImGui::GetIO();
            if (ImGui::IsItemDeactivated() &&
                ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
                io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Left] <=
                    kClickSlop * kClickSlop) {
                PickAt(io.MousePos.x - origin.x,
                       io.MousePos.y - origin.y, io.KeyCtrl);
            }
        }
        HandleNavigation(hovered, active, size.y);
        DrawGizmo(origin, size);
        DrawToolbar(origin, size);
        DrawToolStrip(origin);
        DrawAxisIndicator(origin, size);
        if (!renderError.empty()) {
            // At the bottom left, beside the axis indicator.
            ImGui::SetCursorScreenPos(ImVec2(
                origin.x + 90.0f * contentScale,
                origin.y + size.y - 2.0f * ImGui::GetFrameHeight()));
            ImGui::TextColored(ToImVec4(theme.error), "%s",
                               renderError.c_str());
        }
        ImGui::End();
    }

    /// Kit's orientation indicator in the viewport's bottom left
    /// corner: the world's x, y and z axes as seen through the active
    /// camera, in red, green and blue with labelled tips, drawn back
    /// to front and dimmer when pointing away.
    void DrawAxisIndicator(const ImVec2& origin, const ImVec2& size) {
        if (!scene.Has(activeCamera)) {
            return;
        }
        const glm::dmat4 camera = scene.WorldTransform(activeCamera);
        // World directions in the camera's frame: x right, y up, z
        // towards the viewer.
        const glm::dmat3 toView = glm::transpose(
            glm::dmat3(glm::normalize(glm::dvec3(camera[0])),
                       glm::normalize(glm::dvec3(camera[1])),
                       glm::normalize(glm::dvec3(camera[2]))));
        const float length = 26.0f * contentScale;
        const ImVec2 centre(origin.x + 14.0f * contentScale + length,
                            origin.y + size.y - 14.0f * contentScale -
                                length);
        struct Axis {
            glm::dvec3 view;
            ImU32 colour;
            const char* label;
        };
        std::array<Axis, 3> axes{
            {{toView * glm::dvec3(1, 0, 0),
              IM_COL32(222, 74, 64, 255), "X"},
             {toView * glm::dvec3(0, 1, 0),
              IM_COL32(110, 190, 70, 255), "Y"},
             {toView * glm::dvec3(0, 0, 1),
              IM_COL32(70, 130, 230, 255), "Z"}}};
        std::sort(axes.begin(), axes.end(),
                  [](const Axis& a, const Axis& b) {
                      return a.view.z < b.view.z;
                  });
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddCircleFilled(
            centre, length + 10.0f * contentScale,
            ImGui::GetColorU32(ToImVec4(theme.overlay)), 32);
        for (const Axis& axis : axes) {
            const ImVec2 tip(
                centre.x + static_cast<float>(axis.view.x) * length,
                centre.y - static_cast<float>(axis.view.y) * length);
            // Axes pointing away from the viewer fade.
            const float alpha = axis.view.z < 0.0 ? 0.45f : 1.0f;
            const ImU32 colour = (axis.colour & ~IM_COL32_A_MASK) |
                                 (static_cast<ImU32>(alpha * 255.0f)
                                  << IM_COL32_A_SHIFT);
            draw->AddLine(centre, tip, colour, 2.0f * contentScale);
            draw->AddCircleFilled(tip, 7.0f * contentScale, colour,
                                  16);
            const ImVec2 text = ImGui::CalcTextSize(axis.label);
            draw->AddText(
                ImVec2(tip.x - 0.5f * text.x, tip.y - 0.5f * text.y),
                IM_COL32(20, 20, 20,
                         static_cast<int>(alpha * 255.0f)),
                axis.label);
        }
    }

    /// Selects what the viewport shows at (`x`, `y`) pixels: it
    /// alone, or toggled in the selection with `toggle`; the
    /// background clears the selection unless toggling.
    void PickAt(float x, float y, bool toggle) {
        if (x < 0.0f || y < 0.0f) {
            return;
        }
        const Path picked = self.Pick(static_cast<std::uint32_t>(x),
                                      static_cast<std::uint32_t>(y));
        if (!toggle) {
            SetSelection(picked.IsEmpty()
                             ? std::vector<Path>{}
                             : std::vector<Path>{picked});
            return;
        }
        if (picked.IsEmpty()) {
            return;
        }
        std::vector<Path> paths = selection;
        const auto found =
            std::find(paths.begin(), paths.end(), picked);
        if (found != paths.end()) {
            paths.erase(found);
        } else {
            paths.push_back(picked);
        }
        SetSelection(std::move(paths));
    }

    /// The transform gizmo on a single selected prim, in coordinates
    /// relative to the camera so it stays precise far from the world
    /// origin.
    void DrawGizmo(const ImVec2& origin, const ImVec2& size) {
        gizmoDrawn = false;
        if (gizmo == Viewer::Gizmo::None || selection.size() != 1) {
            return;
        }
        const Path& path = selection.front();
        if (!scene.Has(path) || path.IsRoot() ||
            IsSessionPath(path) || !scene.Has(activeCamera)) {
            return;
        }
        const CameraParams camera = CameraParamsOf(
            scene, activeCamera, ViewportWidth(), ViewportHeight());
        const glm::dvec3 eye(camera.transform[3]);
        glm::dmat4 orientation = camera.transform;
        orientation[3] = glm::dvec4(0.0, 0.0, 0.0, 1.0);
        const glm::mat4 view(glm::inverse(orientation));
        const glm::mat4 projection = camera.RasterView().projection;
        glm::dmat4 world = scene.WorldTransform(path);
        world[3] -= glm::dvec4(eye, 0.0);
        glm::mat4 model(world);
        ImGuizmo::SetOrthographic(camera.orthographic);
        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
        ImGuizmo::SetRect(origin.x, origin.y, size.x, size.y);
        const ImGuizmo::OPERATION operation =
            gizmo == Viewer::Gizmo::Rotate  ? ImGuizmo::ROTATE
            : gizmo == Viewer::Gizmo::Scale ? ImGuizmo::SCALE
                                            : ImGuizmo::TRANSLATE;
        if (ImGuizmo::Manipulate(
                glm::value_ptr(view), glm::value_ptr(projection),
                operation,
                gizmoLocal ? ImGuizmo::LOCAL : ImGuizmo::WORLD,
                glm::value_ptr(model))) {
            glm::dmat4 moved(model);
            moved[3] += glm::dvec4(eye, 0.0);
            self.EditLocalTransform(
                path, LocalFromWorld(scene, path, moved), true);
        }
        gizmoDrawn = true;
    }

    /// Draws the light at `path` (placed by `world`) as Kit's light
    /// manipulators show one: its emitting shape and the direction it
    /// shines, with a spot's cone, in `style`. Returns false for
    /// prims that are no light. Sizes for points and directions
    /// follow the view's distance.
    bool DrawLightGizmo(Overlay& overlay, const Path& path,
                        const glm::dmat4& world,
                        const OverlayStyle& style) {
        const std::string& type = scene.TypeOf(path);
        const bool sphere = type == tokens::kSphereLight;
        const bool rect = type == tokens::kRectLight;
        const bool disk = type == tokens::kDiskLight;
        const bool distant = type == tokens::kDistantLight;
        const bool cylinder = type == tokens::kCylinderLight;
        if (!sphere && !rect && !disk && !distant && !cylinder) {
            return false;
        }
        const double reach =
            0.12 * static_cast<double>(controller.distance);
        const glm::dvec3 centre(world[3]);
        const glm::dvec3 x = glm::normalize(glm::dvec3(world[0]));
        const glm::dvec3 y = glm::normalize(glm::dvec3(world[1]));
        // Lights shine along their -z axis.
        const glm::dvec3 down = -glm::normalize(glm::dvec3(world[2]));
        if (distant) {
            // Parallel rays through a disc facing the light.
            overlay.Circle(centre, down, 0.35 * reach, style);
            for (const glm::dvec2 offset :
                 {glm::dvec2(0.0), glm::dvec2(1.0, 0.0),
                  glm::dvec2(-1.0, 0.0), glm::dvec2(0.0, 1.0),
                  glm::dvec2(0.0, -1.0)}) {
                const glm::dvec3 start =
                    centre +
                    0.25 * reach * (offset.x * x + offset.y * y);
                overlay.Arrow(start, start + down * reach, style,
                              0.18 * reach);
            }
            return true;
        }
        if (sphere) {
            const SphereLight light(std::as_const(scene), path);
            const double radius =
                light.GetTreatAsPoint() ? 0.0 : light.GetRadius();
            overlay.Sphere(centre, std::max(radius, 0.05 * reach),
                           style);
            const double cone = light.GetShapingConeAngle();
            if (cone < 90.0) {
                // The cone's edge as rays and a ring at the gizmo's
                // reach.
                const double spread =
                    reach * std::tan(glm::radians(cone));
                const glm::dvec3 tip = centre + down * reach;
                overlay.Circle(tip, down, spread, style);
                for (int k = 0; k < 4; ++k) {
                    const double angle = k * glm::half_pi<double>();
                    overlay.Line(centre,
                                 tip + spread * (std::cos(angle) * x +
                                                 std::sin(angle) * y),
                                 style);
                }
            } else {
                overlay.Arrow(centre, centre + down * reach, style,
                              0.18 * reach);
            }
            return true;
        }
        if (cylinder) {
            // Its tube: a ring at each end and four lines along it,
            // scaled with the prim.
            const CylinderLight light(std::as_const(scene), path);
            const glm::dvec3 half =
                0.5 * light.GetLength() *
                glm::length(glm::dvec3(world[0])) * x;
            const double radius =
                light.GetRadius() * glm::length(glm::dvec3(world[1]));
            overlay.Circle(centre - half, x, radius, style);
            overlay.Circle(centre + half, x, radius, style);
            for (const glm::dvec3 side : {y, -y, down, -down}) {
                overlay.Line(centre - half + radius * side,
                             centre + half + radius * side, style);
            }
            return true;
        }
        // Rect and disk lights: their outline, scaled with the prim,
        // and their normal.
        if (rect) {
            const RectLight light(std::as_const(scene), path);
            const double w = 0.5 * light.GetWidth();
            const double h = 0.5 * light.GetHeight();
            overlay.Box({-w, -h, 0.0}, {w, h, 0.0}, style, world);
        } else {
            const double radius =
                DiskLight(std::as_const(scene), path).GetRadius() *
                glm::length(glm::dvec3(world[0]));
            overlay.Circle(centre, down, radius, style);
        }
        overlay.Arrow(centre, centre + down * reach, style,
                      0.18 * reach);
        return true;
    }

    /// Draws in the session's selection overlay what the outline
    /// (`SetHighlight`) cannot show: each selected light as its
    /// gizmo, and each camera's axes. Geometry is outlined.
    void UpdateSelectionOverlay() {
        Overlay& overlay = scene.GetOverlay(kSelectionOverlay);
        overlay.SetVisible(display.selection);
        if (selection.empty() && overlay.Empty()) {
            return;
        }
        overlay.Clear();
        OverlayStyle style;
        style.color = theme.accent;
        style.width = 2.0f * contentScale;
        style.depth = OverlayDepth::XRay;
        OverlayStyle lamp = style;
        lamp.color = kLampColor;
        lamp.width = 1.5f * contentScale;
        for (const Path& path : selection) {
            if (!scene.Has(path) || path.IsRoot()) {
                continue;
            }
            const glm::dmat4 world = scene.WorldTransform(path);
            if (DrawLightGizmo(overlay, path, world, lamp)) {
                continue;
            }
            if (scene.TypeOf(path) == tokens::kCamera) {
                overlay.Axes(world, 1.0, style);
            }
        }
    }

    /// Draws what the eye menu shows over the scene into the
    /// viewport's session overlay: the grid, every light's gizmo and
    /// every scene camera's frustum but the one looked through. The
    /// selection's outline draws selected lights brighter.
    void UpdateViewportOverlay() {
        Overlay& overlay = scene.GetOverlay(kViewportOverlay);
        if (!display.grid && !display.lights && !display.cameras) {
            if (!overlay.Empty()) {
                overlay.Clear();
            }
            return;
        }
        overlay.Clear();
        if (display.grid) {
            DrawGrid(overlay);
        }
        const auto selected = [&](const Path& path) {
            return display.selection &&
                   std::find(selection.begin(), selection.end(),
                             path) != selection.end();
        };
        OverlayStyle style;
        style.width = 1.0f * contentScale;
        style.depth = OverlayDepth::XRay;
        if (display.lights) {
            style.color = kLampDimColor;
            for (const std::string_view type :
                 {tokens::kDistantLight, tokens::kSphereLight,
                  tokens::kRectLight, tokens::kDiskLight,
                  tokens::kCylinderLight}) {
                for (const Path& path : scene.Prims(type)) {
                    if (!IsSessionPath(path) && !selected(path)) {
                        DrawLightGizmo(overlay, path,
                                       scene.WorldTransform(path),
                                       style);
                    }
                }
            }
        }
        if (display.cameras) {
            style.color = kCameraGizmoColor;
            for (const Path& path : scene.Prims(tokens::kCamera)) {
                if (!IsSessionPath(path) && path != activeCamera) {
                    DrawCameraFrustum(overlay, path, style);
                }
            }
        }
    }

    /// Kit's ground grid: lines a power of ten apart suiting the
    /// view's distance, around the point the view orbits, with the x
    /// axis in red and the y axis in green through the origin.
    void DrawGrid(Overlay& overlay) {
        constexpr int kLines = 20;  // to each side of the centre
        const double step =
            std::pow(10.0, std::floor(std::log10(std::max(
                               0.25 * controller.distance, 1e-3))));
        const glm::dvec2 centre =
            glm::round(glm::dvec2(controller.target) / step) * step;
        const double half = kLines * step;
        std::vector<glm::dvec3> endpoints;
        endpoints.reserve(8 * kLines + 4);
        for (int k = -kLines; k <= kLines; ++k) {
            const double x = centre.x + k * step;
            const double y = centre.y + k * step;
            // The axes draw in their own colours.
            if (std::abs(x) > 0.5 * step) {
                endpoints.emplace_back(x, centre.y - half, 0.0);
                endpoints.emplace_back(x, centre.y + half, 0.0);
            }
            if (std::abs(y) > 0.5 * step) {
                endpoints.emplace_back(centre.x - half, y, 0.0);
                endpoints.emplace_back(centre.x + half, y, 0.0);
            }
        }
        OverlayStyle style;
        style.color = kGridColor;
        style.width = 1.0f * contentScale;
        overlay.Lines(endpoints, style);
        style.width = 1.5f * contentScale;
        if (std::abs(centre.y) <= half) {
            style.color = {0.80f, 0.32f, 0.30f, 0.8f};
            overlay.Line({centre.x - half, 0.0, 0.0},
                         {centre.x + half, 0.0, 0.0}, style);
        }
        if (std::abs(centre.x) <= half) {
            style.color = {0.42f, 0.72f, 0.34f, 0.8f};
            overlay.Line({0.0, centre.y - half, 0.0},
                         {0.0, centre.y + half, 0.0}, style);
        }
    }

    /// The frustum of the camera at `path` as a pyramid from its eye,
    /// its size following the view's distance, with a triangle over
    /// its top edge showing which way is up.
    void DrawCameraFrustum(Overlay& overlay, const Path& path,
                           const OverlayStyle& style) {
        CameraParams camera;
        try {
            camera = CameraParamsOf(scene, path, ViewportWidth(),
                                    ViewportHeight());
        } catch (const std::exception&) {
            return;  // a camera that cannot resolve draws nothing
        }
        const double depth =
            0.06 * static_cast<double>(controller.distance);
        const double up =
            depth * std::tan(0.5 * camera.VerticalFov());
        const double side =
            up * camera.width / std::max(camera.height, 1u);
        const glm::dmat4& world = camera.transform;
        const auto at = [&](double x, double y, double z) {
            return glm::dvec3(world * glm::dvec4(x, y, z, 1.0));
        };
        // Cameras look down their -z axis with +y up.
        const glm::dvec3 eye = at(0.0, 0.0, 0.0);
        const glm::dvec3 corners[] = {
            at(-side, -up, -depth), at(side, -up, -depth),
            at(side, up, -depth), at(-side, up, -depth)};
        const glm::dvec3 endpoints[] = {
            eye,        corners[0], eye,        corners[1],
            eye,        corners[2], eye,        corners[3],
            corners[0], corners[1], corners[1], corners[2],
            corners[2], corners[3], corners[3], corners[0]};
        overlay.Lines(endpoints, style);
        const glm::dvec3 roof[] = {
            at(-0.5 * side, 1.15 * up, -depth),
            at(0.0, 1.6 * up, -depth),
            at(0.5 * side, 1.15 * up, -depth),
            at(-0.5 * side, 1.15 * up, -depth)};
        overlay.Polyline(roof, style);
    }

    void HandleNavigation(bool hovered, bool active, float height) {
        const ImGuiIO& io = ImGui::GetIO();
        bool moved = false;
        if (active &&
            ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            controller.Orbit(-io.MouseDelta.x * 0.005f,
                             io.MouseDelta.y * 0.005f);
            moved = true;
        }
        // Right drags along the ground, middle across the screen (up
        // and down along z).
        if (active &&
            ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
            controller.Pan(io.MouseDelta.x, io.MouseDelta.y, height);
            moved = true;
        }
        if (active &&
            ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f)) {
            controller.PanVertical(io.MouseDelta.x, io.MouseDelta.y,
                                   height);
            moved = true;
        }
        if (hovered && io.MouseWheel != 0.0f) {
            controller.Zoom(io.MouseWheel);
            moved = true;
        }
        if (moved &&
            (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f ||
             io.MouseWheel != 0.0f)) {
            WriteActiveCamera();
        }
        if (!io.WantTextInput &&
            (hovered || ImGui::IsWindowFocused()) && !io.KeyCtrl) {
            if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                self.Frame(selection);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
                CycleWireframe();
            }
            const std::pair<ImGuiKey, Viewer::Gizmo> gizmoKeys[] = {
                {ImGuiKey_Q, Viewer::Gizmo::None},
                {ImGuiKey_W, Viewer::Gizmo::Translate},
                {ImGuiKey_E, Viewer::Gizmo::Rotate},
                {ImGuiKey_R, Viewer::Gizmo::Scale}};
            for (const auto& [key, mode] : gizmoKeys) {
                if (ImGui::IsKeyPressed(key, false)) {
                    gizmo = mode;
                }
            }
        }
        if (!io.WantTextInput &&
            ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            open = false;
        }
    }

    /// Steps the viewport's wireframe from the mode it draws with to
    /// the next of off, overlay and hidden line.
    void CycleWireframe() {
        Wireframe current = Wireframe::Off;
        if (wireframe) {
            current = *wireframe;
        } else {
            try {
                current = WireframeOf(scene, SessionSettingsPath());
            } catch (const std::invalid_argument&) {
                // A bad token renders as an error; cycling starts
                // over from off.
            }
        }
        wireframe = static_cast<Wireframe>(
            (static_cast<std::uint32_t>(current) + 1) % 3);
    }

    /// The spectral band the viewport shows of SpectralRadiance: the
    /// mean, or one of the active camera's bands by its wavelengths.
    void DrawBandCombo(float width) {
        SpectralBands bands;
        try {
            bands = CameraParamsOf(scene, activeCamera, 16, 16).bands;
        } catch (const std::exception&) {
            return;
        }
        const auto label = [&](int band) {
            char text[64];
            if (band < 0) {
                std::snprintf(text, sizeof(text), "%.0f-%.0f nm mean",
                              bands.minNm, bands.maxNm);
            } else {
                std::snprintf(text, sizeof(text), "%.0f-%.0f nm",
                              bands.Lower(band),
                              bands.Lower(band) + bands.Width());
            }
            return std::string(text);
        };
        if (viewBand >= bands.count) {
            viewBand = -1;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(width);
        if (ImGui::BeginCombo("##band", label(viewBand).c_str())) {
            for (int band = -1; band < bands.count; ++band) {
                if (ImGui::Selectable(label(band).c_str(),
                                      band == viewBand)) {
                    viewBand = band;
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Spectral radiance as grey, at the "
                              "photographic image's exposure");
        }
    }

    /// Kit's vertical tool strip down the viewport's left: select (no
    /// gizmo), move, rotate and scale (Q, W, E, R), then world or
    /// local axes for the gizmo.
    void DrawToolStrip(const ImVec2& origin) {
        const float pad = 8.0f * contentScale;
        const float button = 1.15f * ImGui::GetFrameHeight();
        const float gap = 3.0f * contentScale;
        ImVec2 at(origin.x + pad,
                  origin.y + pad + 1.6f * ImGui::GetFrameHeight());
        // Each button placed in turn: the window's own cursor would
        // return to its left edge.
        const auto next = [&](float extra = 0.0f) {
            at.y += extra;
            ImGui::SetCursorScreenPos(at);
            at.y += button + gap;
        };
        ImDrawList* draw = ImGui::GetWindowDrawList();
        // A square button with its icon centred by the glyph's ink.
        const auto iconButton = [&](const char* id, const char* icon,
                                    bool on) {
            const ImVec2 corner = ImGui::GetCursorScreenPos();
            const bool pressed =
                ImGui::InvisibleButton(id, ImVec2(button, button));
            const bool hovered = ImGui::IsItemHovered();
            const glm::vec4 fill = on        ? theme.accent
                                   : hovered ? theme.fieldHovered
                                             : theme.overlay;
            draw->AddRectFilled(
                corner, ImVec2(corner.x + button, corner.y + button),
                ImGui::GetColorU32(ToImVec4(fill)),
                theme.pillRounding * contentScale);
            DrawIconCentred(draw, icon,
                            ImVec2(corner.x + 0.5f * button,
                                   corner.y + 0.5f * button),
                            ImGui::GetColorU32(ImGuiCol_Text));
            return pressed;
        };
        const struct {
            const char* id;
            const char* icon;
            Viewer::Gizmo mode;
            const char* tip;
        } tools[] = {{"##select", icons::kMousePointer,
                      Viewer::Gizmo::None, "Select (Q)"},
                     {"##move", icons::kMove,
                      Viewer::Gizmo::Translate, "Move (W)"},
                     {"##rotate", icons::kRotateCw,
                      Viewer::Gizmo::Rotate, "Rotate (E)"},
                     {"##scale", icons::kScaling,
                      Viewer::Gizmo::Scale, "Scale (R)"}};
        for (const auto& tool : tools) {
            next();
            if (iconButton(tool.id, tool.icon, gizmo == tool.mode)) {
                gizmo = tool.mode;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", tool.tip);
            }
        }
        next(2.0f * gap);
        if (iconButton("##axes",
                       gizmoLocal ? icons::kBox : icons::kGlobe,
                       gizmoLocal)) {
            gizmoLocal = !gizmoLocal;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(gizmoLocal ? "Gizmo in local axes"
                                         : "Gizmo in world axes");
        }
    }

    /// Kit's viewport settings menu: frame pacing, exposure, the
    /// perspective camera's lens and framing, then applications'
    /// items (`AddViewportMenuItem`).
    void DrawSettingsMenu() {
        ImGui::SeparatorText("Viewport");
        bool vsync = self.GetVsync();
        if (ImGui::MenuItem("Vsync", nullptr, &vsync)) {
            self.SetVsync(vsync);
        }
        ImGui::MenuItem("Gizmo in Local Axes", nullptr, &gizmoLocal);
        ImGui::SeparatorText("Exposure");
        // On and off in one: automatic, or the camera's own exposure.
        bool automatic = AutoExposing();
        if (ImGui::MenuItem("Auto Exposure", nullptr, &automatic)) {
            exposure = automatic ? Viewer::Exposure::Auto
                                 : Viewer::Exposure::Camera;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Key the scene's metered luminance to middle grey; "
                "off, the camera's "
                "own exposure (ISO, shutter, f-stop) applies");
        }
        ImGui::TextDisabled("Mode");
        const std::pair<Viewer::Exposure, const char*> modes[] = {
            {Viewer::Exposure::Default, "Default"},
            {Viewer::Exposure::Auto, "Automatic"},
            {Viewer::Exposure::Camera, "Camera's Own"}};
        for (const auto& [mode, name] : modes) {
            if (ImGui::MenuItem(name, nullptr, exposure == mode)) {
                exposure = mode;
            }
        }
        float stops = static_cast<float>(compensation);
        ImGui::SetNextItemWidth(160.0f * contentScale);
        if (ImGui::SliderFloat("Compensation", &stops, -6.0f, 6.0f,
                               "%+.1f EV")) {
            compensation = stops;
        }
        ImGui::SeparatorText("Camera");
        const bool perspective = activeCamera == PerspectiveCamera();
        ImGui::BeginDisabled(!perspective ||
                             !scene.Has(activeCamera));
        if (scene.Has(activeCamera)) {
            CameraParams lens =
                CameraParamsOf(scene, activeCamera, ViewportWidth(),
                               ViewportHeight());
            float fov =
                static_cast<float>(glm::degrees(lens.VerticalFov()));
            ImGui::SetNextItemWidth(160.0f * contentScale);
            if (ImGui::SliderFloat("Vertical FOV", &fov, 5.0f, 120.0f,
                                   "%.0f deg")) {
                // Through the focal length, as the Camera panel shows
                // it.
                lens.focalLengthMm =
                    lens.verticalApertureMm /
                    (2.0 *
                     std::tan(0.5 * glm::radians(
                                        static_cast<double>(fov))));
                self.Edit(activeCamera, tokens::kFocalLength,
                          lens.focalLengthMm, true);
                WriteActiveCamera();
            }
        }
        ImGui::EndDisabled();
        if (ImGui::MenuItem("Frame Selection", "F", false,
                            !selection.empty())) {
            self.Frame(selection);
        }
        if (ImGui::MenuItem("Frame All")) {
            self.Frame();
        }
        if (!viewportMenuItems.empty()) {
            ImGui::Separator();
            // Copied: an action may add or remove items.
            const std::vector<ViewportMenuItem> items =
                viewportMenuItems;
            for (const ViewportMenuItem& item : items) {
                const bool checked =
                    item.checked && item.checked(self);
                if (ImGui::MenuItem(item.label.c_str(), nullptr,
                                    checked)) {
                    item.action(self);
                }
            }
        }
    }

    /// Kit's eye menu: what the viewport draws over the scene,
    /// applications' toggles
    /// (`AddViewportToggle`), and the heads-up display's items.
    void DrawShowMenu() {
        // Toggles stay open, so several switch at once.
        ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
        ImGui::SeparatorText("Show By Type");
        ImGui::MenuItem("Lights", nullptr, &display.lights);
        ImGui::MenuItem("Cameras", nullptr, &display.cameras);
        ImGui::SeparatorText("Show");
        ImGui::MenuItem("Grid", nullptr, &display.grid);
        ImGui::MenuItem("Selection Outline", nullptr,
                        &display.selection);
        ImGui::MenuItem("Scene Overlays", nullptr, &display.overlays);
        if (!viewportToggles.empty()) {
            ImGui::Separator();
            for (std::size_t i = 0; i < viewportToggles.size(); ++i) {
                bool shown = viewportToggles[i].shown;
                if (ImGui::MenuItem(viewportToggles[i].label.c_str(),
                                    nullptr, &shown)) {
                    self.SetViewportToggleShown(viewportToggles[i].id,
                                                shown);
                }
            }
        }
        ImGui::Separator();
        if (ImGui::BeginMenu("Heads Up Display")) {
            ImGui::MenuItem("Show", nullptr, &display.hud);
            ImGui::Separator();
            ImGui::BeginDisabled(!display.hud);
            ImGui::MenuItem("FPS", nullptr, &display.hudFps);
            ImGui::MenuItem("Device Memory", nullptr,
                            &display.hudDeviceMemory);
            ImGui::MenuItem("Host Memory", nullptr,
                            &display.hudHostMemory);
            ImGui::MenuItem("Resolution", nullptr,
                            &display.hudResolution);
            ImGui::MenuItem("Exposure", nullptr,
                            &display.hudExposure);
            ImGui::MenuItem("Triangles", nullptr,
                            &display.hudTriangles);
            for (HudItem& item : hudItems) {
                ImGui::MenuItem(item.label.c_str(), nullptr,
                                &item.shown);
            }
            ImGui::EndDisabled();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Toolbar")) {
            ImGui::MenuItem("Renderer", nullptr,
                            &display.toolbarRenderer);
            ImGui::MenuItem("Render Var", nullptr,
                            &display.toolbarVar);
            ImGui::MenuItem("Camera", nullptr,
                            &display.toolbarCamera);
            ImGui::MenuItem("Lighting", nullptr,
                            &display.toolbarLighting);
            ImGui::EndMenu();
        }
        ImGui::PopItemFlag();
        // A switch shows in the heads-up display at once.
        hudTime = -kHudRefresh;
    }

    /// Kit's renderer menu: the backends, the rendering mode
    /// (wireframe) and the render settings.
    void DrawRendererMenu() {
        ImGui::SeparatorText("Renderer");
        for (const std::string& name : RenderBackends()) {
            if (ImGui::MenuItem(DisplayName(name).c_str(), nullptr,
                                name == renderer) &&
                name != renderer) {
                try {
                    self.SetRenderer(name);
                } catch (const std::exception& error) {
                    renderError = error.what();
                    self.Log(LogLevel::Error, error.what());
                }
            }
        }
        ImGui::SeparatorText("Rendering Mode");
        for (const std::optional<Wireframe> mode :
             {std::optional<Wireframe>(),
              std::optional(Wireframe::Off),
              std::optional(Wireframe::Overlay),
              std::optional(Wireframe::HiddenLine)}) {
            const char* name =
                mode ? WireframeName(mode) : "From Render Settings";
            if (ImGui::MenuItem(name, mode ? nullptr : "Z cycles",
                                mode == wireframe)) {
                wireframe = mode;
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Render Settings...")) {
            for (const auto& panel : panels) {
                if (panel->Title() == "Settings") {
                    panel->SetVisible(true);
                    ImGui::SetWindowFocus(panel->Title().c_str());
                }
            }
        }
    }

    /// Kit's camera menu: the viewer's perspective camera, its
    /// orthographic views from the top, front and right, the scene's
    /// cameras, and a new camera where the view is.
    void DrawCameraMenu() {
        const bool viewer = activeCamera == PerspectiveCamera();
        if (ImGui::MenuItem("Perspective", nullptr,
                            viewer && orthographicView < 0)) {
            orthographicView = -1;
            if (!viewer) {
                SetActive(PerspectiveCamera());
            }
            WriteActiveCamera();
        }
        const struct {
            const char* name;
            float yaw;
            float pitch;
        } views[] = {{"Top", 0.0f, OrbitController::kMaxPitch},
                     {"Front", 0.0f, 0.0f},
                     {"Right", glm::half_pi<float>(), 0.0f}};
        for (int v = 0; v < 3; ++v) {
            if (ImGui::MenuItem(views[v].name, nullptr,
                                viewer && orthographicView == v)) {
                if (!viewer) {
                    SetActive(PerspectiveCamera());
                }
                orthographicView = v;
                controller.yaw = views[v].yaw;
                controller.pitch = views[v].pitch;
                WriteActiveCamera();
            }
        }
        const std::vector<Path> cameras = ViewableCameras(scene);
        if (cameras.size() > 1) {
            ImGui::SeparatorText("Cameras");
            for (const Path& camera : cameras) {
                if (camera == PerspectiveCamera()) {
                    continue;
                }
                if (ImGui::MenuItem(Label(camera).c_str(), nullptr,
                                    camera == activeCamera)) {
                    SetActive(camera);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", camera.String().c_str());
                }
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Create Camera from View")) {
            Path path("/Camera");
            for (int k = 1; scene.Has(path); ++k) {
                path = Path("/Camera_" + std::to_string(k));
            }
            self.CameraFromView(path);
            SetSelection({path});
        }
    }

    /// Refreshes the heads-up display's lines a few times a second:
    /// Kit's frame rate, memory and resolution, and applications'
    /// items.
    void RefreshHud() {
        if (frameInfo.time - hudTime < kHudRefresh) {
            return;
        }
        hudTime = frameInfo.time;
        hudLines.clear();
        char line[192];
        if (display.hudFps) {
            const double fps = std::max(
                static_cast<double>(ImGui::GetIO().Framerate), 1e-3);
            std::snprintf(line, sizeof(line),
                          "FPS: %.1f, Frame time: %.2f ms", fps,
                          1000.0 / fps);
            hudLines.emplace_back(line);
        }
        if (display.hudDeviceMemory) {
            const auto [used, total] = DeviceMemory();
            hudLines.push_back(
                "Device Memory: " + FormatBytes(used) + " / " +
                FormatBytes(total) + " (renderer " +
                FormatBytes(delegate->Stats().gpuBytes) + ")");
        }
        if (display.hudHostMemory) {
            const std::optional<std::uint64_t> bytes =
                HostResidentBytes();
            hudLines.push_back(
                "Host Memory: " +
                (bytes ? FormatBytes(*bytes) : "unknown"));
        }
        if (display.hudResolution) {
            std::snprintf(line, sizeof(line), "Resolution: %u x %u",
                          ViewportWidth(), ViewportHeight());
            hudLines.emplace_back(line);
        }
        if (display.hudExposure) {
            if (const std::optional<double> ev = self.MeteredEv()) {
                std::snprintf(line, sizeof(line),
                              "Exposure: EV %.1f%s, %+.1f stops", *ev,
                              AutoExposing() ? " (auto)" : "",
                              autoStops.value_or(0.0) + compensation);
            } else {
                std::snprintf(line, sizeof(line),
                              "Exposure: not metered");
            }
            hudLines.emplace_back(line);
        }
        if (display.hudTriangles) {
            hudLines.push_back(
                "Triangles: " +
                FormatCount(delegate->Stats().triangles));
        }
        for (const HudItem& item : hudItems) {
            if (item.shown) {
                hudLines.push_back(item.label + ": " +
                                   item.value(self));
            }
        }
    }

    /// GPU memory of the device-local heaps: what is in use (the
    /// whole process's where the driver reports budgets,
    /// VK_EXT_memory_budget, else the viewer's own allocations) and
    /// the heaps' size.
    std::pair<std::uint64_t, std::uint64_t> DeviceMemory() const {
        const VkPhysicalDeviceMemoryProperties* properties = nullptr;
        vmaGetMemoryProperties(gpu->Allocator(), &properties);
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(gpu->Allocator(), budgets.data());
        std::uint64_t used = 0;
        std::uint64_t total = 0;
        for (std::uint32_t heap = 0;
             heap < properties->memoryHeapCount; ++heap) {
            if ((properties->memoryHeaps[heap].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                used += budgets[heap].usage;
                total += properties->memoryHeaps[heap].size;
            }
        }
        return {used, total};
    }

    /// The heads-up display as Kit draws it: lines of text on a
    /// translucent box whose top right corner is `topRight`. Returns
    /// the box's left edge (`topRight.x` without a box).
    float DrawHud(const ImVec2& topRight) {
        if (!display.hud) {
            return topRight.x;
        }
        RefreshHud();
        if (hudLines.empty()) {
            return topRight.x;
        }
        const float pad = 6.0f * contentScale;
        const float lineHeight = ImGui::GetTextLineHeight();
        float width = 0.0f;
        for (const std::string& text : hudLines) {
            width =
                std::max(width, ImGui::CalcTextSize(text.c_str()).x);
        }
        const ImVec2 corner(topRight.x - width - 2.0f * pad,
                            topRight.y);
        ImDrawList* draw = ImGui::GetWindowDrawList();
        ImVec4 box = ToImVec4(theme.overlay);
        box.w *= 0.85f;
        draw->AddRectFilled(
            corner,
            ImVec2(corner.x + width + 2.0f * pad,
                   corner.y + lineHeight * hudLines.size() +
                       2.0f * pad),
            ImGui::GetColorU32(box), 4.0f * contentScale);
        ImVec2 at(corner.x + pad, corner.y + pad);
        for (const std::string& text : hudLines) {
            // The label muted, its value bright.
            const std::size_t colon = text.find(": ");
            const std::string label = text.substr(
                0, colon == std::string::npos ? 0 : colon + 2);
            draw->AddText(
                at, ImGui::GetColorU32(ToImVec4(theme.textMuted)),
                label.c_str());
            draw->AddText(
                ImVec2(at.x + ImGui::CalcTextSize(label.c_str()).x,
                       at.y),
                ImGui::GetColorU32(ToImVec4(theme.text)),
                text.c_str() + label.size());
            at.y += lineHeight;
        }
        return corner.x;
    }

    /// The viewport's toolbar as Kit draws it, pills over the image:
    /// at the top left the settings menu, the renderer, the eye menu
    /// of what shows and the shown render var; the heads-up display
    /// in the top right corner, with the camera and the lighting to
    /// its left.
    void DrawToolbar(const ImVec2& origin, const ImVec2& size) {
        const float pad = 8.0f * contentScale;
        ImGui::SetCursorScreenPos(
            ImVec2(origin.x + pad, origin.y + pad));
        ImGui::PushStyleColor(ImGuiCol_FrameBg,
                              ToImVec4(theme.overlay));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered,
                              ToImVec4(theme.fieldHovered));
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ToImVec4(theme.overlay));
        ImGui::PushStyleColor(ImGuiCol_PopupBg,
                              ToImVec4(theme.panel));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,
                            theme.pillRounding * contentScale);
        ImGui::PushStyleVar(
            ImGuiStyleVar_FramePadding,
            ImVec2(9.0f * contentScale, 4.0f * contentScale));
        ImGui::PushStyleVar(
            ImGuiStyleVar_ItemSpacing,
            ImVec2(5.0f * contentScale, 4.0f * contentScale));
        // Pills fit their label, without an arrow.
        constexpr ImGuiComboFlags kPill =
            ImGuiComboFlags_WidthFitPreview |
            ImGuiComboFlags_NoArrowButton;
        const float comboWidth = 150.0f * contentScale;
        ImDrawList* draw = ImGui::GetWindowDrawList();
        const ImU32 textColor = ImGui::GetColorU32(ImGuiCol_Text);

        // The settings menu: a pill drawn with Kit's three bars.
        {
            const ImVec2 corner = ImGui::GetCursorScreenPos();
            const bool open =
                ImGui::BeginCombo("##settings", "   ", kPill);
            if (open) {
                DrawSettingsMenu();
                ImGui::EndCombo();
            }
            const float height = ImGui::GetFrameHeight();
            const float half = 0.22f * height;
            const ImVec2 centre(
                corner.x +
                    0.5f * (ImGui::CalcTextSize("   ").x +
                            2.0f * ImGui::GetStyle().FramePadding.x),
                corner.y + 0.5f * height);
            for (const float dy : {-0.6f, 0.0f, 0.6f}) {
                draw->AddLine(
                    ImVec2(centre.x - half, centre.y + dy * half),
                    ImVec2(centre.x + half, centre.y + dy * half),
                    textColor, 1.5f * contentScale);
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Viewport settings");
        }
        ImGui::SameLine();

        if (display.toolbarRenderer) {
            // Spaces where the drawn icon goes.
            const ImVec2 corner = ImGui::GetCursorScreenPos();
            const std::string rendererLabel =
                "      " + DisplayName(renderer);
            if (ImGui::BeginCombo("##renderer", rendererLabel.c_str(),
                                  kPill)) {
                DrawRendererMenu();
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Renderer, rendering mode and render settings");
            }
            const float icon = ImGui::GetTextLineHeight() * 1.15f;
            DrawRendererIcon(
                draw,
                ImVec2(corner.x + ImGui::GetStyle().FramePadding.x,
                       corner.y +
                           0.5f * (ImGui::GetFrameHeight() - icon)),
                icon);
            ImGui::SameLine();
        }

        if (ImGui::BeginCombo("##show", icons::kEye, kPill)) {
            DrawShowMenu();
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("What the viewport shows");
        }
        const std::string varLabel =
            std::string(icons::kImage) + "  " + viewVar;
        if (display.toolbarVar) {
            ImGui::SameLine();
        }
        if (display.toolbarVar &&
            ImGui::BeginCombo("##var", varLabel.c_str(), kPill)) {
            const std::vector<std::string>& shown =
                delegate->Info().viewVars;
            for (const std::string_view name : kRenderVarNames) {
                const bool supported =
                    std::find(shown.begin(), shown.end(), name) !=
                    shown.end();
                ImGui::BeginDisabled(!supported);
                if (ImGui::Selectable(std::string(name).c_str(),
                                      name == viewVar)) {
                    viewVar = name;
                }
                ImGui::EndDisabled();
                if (!supported &&
                    ImGui::IsItemHovered(
                        ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("The %s renderer cannot show "
                                      "%s in the viewport",
                                      renderer.c_str(),
                                      std::string(name).c_str());
                }
            }
            // The AOVs the sensor filter chain writes, below the
            // vars.
            bool aovs = false;
            for (const std::string& name : delegate->ViewVars()) {
                if (!name.starts_with(kAovPrefix)) {
                    continue;
                }
                if (!aovs) {
                    ImGui::SeparatorText("AOVs");
                    aovs = true;
                }
                if (ImGui::Selectable(name.c_str(),
                                      name == viewVar)) {
                    viewVar = name;
                }
            }
            ImGui::EndCombo();
        }
        if (display.toolbarVar &&
            viewVar == render_vars::kSpectralRadiance) {
            DrawBandCombo(comboWidth);
        }

        // The heads-up display in the top right corner, and to its
        // left the camera and the lighting, as Kit's "Perspective"
        // and "Stage Lights".
        const ImGuiStyle& style = ImGui::GetStyle();
        const float hudLeft =
            DrawHud(ImVec2(origin.x + size.x - pad, origin.y + pad));
        const std::string cameraLabel =
            std::string(icons::kCamera) + "  " + Label(activeCamera);
        const std::string lightLabel =
            std::string(icons::kSun) + "  " + LightingName(lighting);
        const auto pillWidth = [&](const std::string& label) {
            return ImGui::CalcTextSize(label.c_str()).x +
                   2.0f * style.FramePadding.x;
        };
        float left = hudLeft - style.ItemSpacing.x;
        if (display.toolbarLighting) {
            left -= pillWidth(lightLabel);
        }
        if (display.toolbarCamera) {
            const float gap =
                display.toolbarLighting ? style.ItemSpacing.x : 0.0f;
            left -= pillWidth(cameraLabel) + gap;
        }
        ImGui::SetCursorScreenPos(ImVec2(left, origin.y + pad));
        if (display.toolbarCamera) {
            if (ImGui::BeginCombo("##camera", cameraLabel.c_str(),
                                  kPill)) {
                DrawCameraMenu();
                ImGui::EndCombo();
            }
            ImGui::SameLine();
        }
        if (display.toolbarLighting &&
            ImGui::BeginCombo("##lighting", lightLabel.c_str(),
                              kPill)) {
            for (const Lighting mode :
                 {Lighting::Stage, Lighting::Camera,
                  Lighting::Default}) {
                if (ImGui::MenuItem(LightingName(mode), nullptr,
                                    mode == lighting)) {
                    lighting = mode;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(4);
    }

    /// Help's windows: the viewport's mouse and keyboard controls,
    /// and what Ernie is.
    void DrawHelpWindows() {
        if (showControls) {
            ImGui::SetNextWindowSize(
                ImVec2(440.0f * contentScale, 0.0f),
                ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Keyboard and Mouse", &showControls,
                             ImGuiWindowFlags_NoDocking |
                                 ImGuiWindowFlags_AlwaysAutoResize)) {
                const std::pair<const char*, const char*> controls[] =
                    {
                        {"Left drag", "Orbit around the target"},
                        {"Right drag", "Pan along the ground"},
                        {"Middle drag",
                         "Pan across the screen, up and down in z"},
                        {"Wheel", "Zoom towards the target"},
                        {"Click",
                         "Select; Ctrl+click adds or removes"},
                        {"F", "Frame the selection (everything "
                              "without one)"},
                        {"Q W E R", "Select, move, rotate, scale"},
                        {"Z", "Cycle wireframe: off, overlay, hidden "
                              "line"},
                        {"Del", "Delete the selection"},
                        {"Ctrl+D", "Duplicate the selection"},
                        {"F2", "Rename the selected prim (Stage)"},
                        {"Ctrl+Z / Ctrl+Shift+Z", "Undo / redo"},
                        {"Esc", "Close the viewer"},
                    };
                if (ImGui::BeginTable("##controls", 2,
                                      ImGuiTableFlags_RowBg)) {
                    for (const auto& [keys, action] : controls) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextColored(ToImVec4(theme.accent),
                                           "%s", keys);
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(action);
                    }
                    ImGui::EndTable();
                }
            }
            ImGui::End();
        }
        if (showAbout) {
            if (ImGui::Begin("About Ernie", &showAbout,
                             ImGuiWindowFlags_NoDocking |
                                 ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::TextUnformatted("Ernie");
                ImGui::TextDisabled("A spectral, physically based "
                                    "sensor renderer on Vulkan");
                ImGui::Separator();
                ImGui::Text("Device: %s", gpu->DeviceName().c_str());
                ImGui::Text("Renderer: %s", renderer.c_str());
                ImGui::Text("Dear ImGui %s", ImGui::GetVersion());
            }
            ImGui::End();
        }
    }

    void DrawPanels() {
        for (std::size_t i = 0; i < panels.size(); ++i) {
            Panel& panel = *panels[i];
            if (!panel.IsVisible()) {
                continue;
            }
            const auto node = dockNodes.find(panel.GetDock());
            if (node != dockNodes.end()) {
                ImGui::SetNextWindowDockID(node->second,
                                           ImGuiCond_FirstUseEver);
            }
            const auto focus =
                std::find(focusRequests.begin(), focusRequests.end(),
                          panel.Title());
            if (focus != focusRequests.end() &&
                drawnPanels.contains(panel.Title())) {
                ImGui::SetNextWindowFocus();
                focusRequests.erase(focus);
            }
            bool keepOpen = true;
            if (ImGui::Begin(panel.Title().c_str(), &keepOpen)) {
                panel.Draw(self);
            }
            ImGui::End();
            drawnPanels.insert(panel.Title());
            if (!keepOpen) {
                panel.SetVisible(false);
            }
        }
    }

    void BuildUi() {
        const ImGuiID dockSpace = ImGui::GetID(kDockSpace);
        if (!layoutReady) {
            BuildDefaultLayout(dockSpace);
            layoutReady = true;
        }
        DrawMenuBar();
        const ImGuiIO& io = ImGui::GetIO();
        if (!io.WantTextInput && io.KeyCtrl) {
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
                if (io.KeyShift) {
                    self.Redo();
                } else {
                    self.Undo();
                }
            } else if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) {
                self.Redo();
            }
        }
        if (!io.WantTextInput && io.KeyCtrl && !selection.empty() &&
            ImGui::IsKeyPressed(ImGuiKey_D, false)) {
            self.Duplicate(std::vector<Path>(selection));
        }
        if (!io.WantTextInput && !selection.empty() &&
            ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
            self.Delete(std::vector<Path>(selection));
        }
        ImGui::DockSpaceOverViewport(dockSpace,
                                     ImGui::GetMainViewport());
        DrawViewport();
        DrawPanels();
        DrawHelpWindows();
        fileDialog.Draw();
        RunCallbacks(Stage::Ui, frameInfo);
        // A drag ends once nothing is held, so the next edit is a new
        // undo step.
        if (!ImGui::IsAnyItemActive() && !ImGuizmo::IsUsing()) {
            history.Close();
        }
    }

    void RecordViewport(VkCommandBuffer commands) {
        const VkClearColorValue clear{{theme.background.r,
                                       theme.background.g,
                                       theme.background.b, 1.0f}};
        bool drawn = false;
        try {
            UpdateSessionLights(scene,
                                scene.WorldTransform(activeCamera));
            std::map<std::string, Value, std::less<>> overrides;
            overrides.emplace(tokens::kCameraRel, activeCamera);
            if (lighting == Lighting::Camera) {
                overrides.emplace(
                    tokens::kLightsRel,
                    std::vector<Path>{CameraLightPath()});
            } else if (lighting == Lighting::Default) {
                overrides.emplace(
                    tokens::kLightsRel,
                    std::vector<Path>{DefaultLightPath()});
            }
            if (wireframe) {
                overrides.emplace(
                    tokens::kWireframe,
                    std::string(WireframeToken(*wireframe)));
            }
            // The viewport favours sharp shadows over images that
            // repeat exactly, and shows every overlay, unless the
            // scene's settings choose.
            const auto authored = [&](std::string_view attribute) {
                return !settingsSource.IsEmpty() &&
                       scene.Get(settingsSource, attribute) !=
                           nullptr;
            };
            if (!authored(tokens::kShadowFit)) {
                overrides.emplace(tokens::kShadowFit,
                                  std::string(tokens::kVisible));
            }
            // Exposure: metered every frame, for automatic exposure
            // and the meter.
            overrides.emplace(tokens::kExposureCompensation,
                              ExposureStops());
            overrides.emplace(tokens::kMeter, true);
            // The eye menu's overlays and the selection's outline
            // always show, besides the overlays the settings choose.
            UpdateSelectionOverlay();
            UpdateViewportOverlay();
            // Selected geometry outlined as Kit does, from the
            // frame's prim ids.
            const std::vector<Path> outlined =
                display.selection ? selection : std::vector<Path>{};
            delegate->SetHighlight(outlined, theme.outline);
            overrides.emplace(
                tokens::kOverlays,
                ShownOverlays(authored(tokens::kOverlays)));
            MirrorSettings(scene, settingsSource,
                           SessionSettingsPath(), overrides,
                           mirrored);
            delegate->SetViewVar(capture && !capture->var.empty()
                                     ? capture->var
                                     : viewVar);
            delegate->Update(scene, SessionSettingsPath(),
                             viewport->Extent());
            const RenderingTarget target =
                viewport->Prepare(commands, clear);
            // An AOV the chain no longer writes shows as LdrColor.
            const std::vector<std::string> shown =
                delegate->ViewVars();
            if (std::find(shown.begin(), shown.end(), viewVar) ==
                shown.end()) {
                viewVar = std::string(render_vars::kLdrColor);
            }
            const std::string& var = capture && !capture->var.empty()
                                         ? capture->var
                                         : viewVar;
            delegate->SetViewBand(viewBand);
            delegate->Record(commands, target, var);
            renderError.clear();
            drawn = true;
        } catch (const std::invalid_argument& error) {
            if (renderError != error.what()) {
                self.Log(LogLevel::Error, error.what());
            }
            renderError = error.what();
        }
        if (!drawn) {
            const RenderingTarget target =
                viewport->Prepare(commands, clear);
            BeginRendering(commands, target);
            vkCmdEndRendering(commands);
        }
        viewport->Color().Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    void RecordUi(VkCommandBuffer commands, VkImage image,
                  VkImageView view, VkExtent2D extent) {
        if (swapchain) {
            TransitionImage(commands, image,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        } else {
            uiTarget->Color().Transition(
                commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
        RenderingTarget target;
        target.color = view;
        target.extent = extent;
        target.clearColor = {{theme.background.r, theme.background.g,
                              theme.background.b, 1.0f}};
        BeginRendering(commands, target);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                        commands);
        vkCmdEndRendering(commands);
        if (swapchain) {
            TransitionImage(commands, image,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
    }

    void Submit(VkCommandBuffer commands,
                const Swapchain::Frame* image) {
        std::vector<VkSemaphoreSubmitInfo> waits;
        if (image != nullptr) {
            VkSemaphoreSubmitInfo& wait = waits.emplace_back();
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            wait.semaphore = image->acquired;
            wait.stageMask =
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        for (const TimelinePoint& point : delegate->SubmitWaits()) {
            VkSemaphoreSubmitInfo& wait = waits.emplace_back();
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            wait.semaphore = point.semaphore;
            wait.value = point.value;
            wait.stageMask = point.stages;
        }
        VkSemaphoreSubmitInfo signal{
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        if (image != nullptr) {
            signal.semaphore = image->rendered;
            signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        VkCommandBufferSubmitInfo commandInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commands;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.waitSemaphoreInfoCount =
            static_cast<std::uint32_t>(waits.size());
        submit.pWaitSemaphoreInfos = waits.data();
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &commandInfo;
        submit.signalSemaphoreInfoCount = image != nullptr ? 1 : 0;
        submit.pSignalSemaphoreInfos = &signal;
        vkResetFences(gpu->Device(), 1, &frame->done);
        CheckVk(vkQueueSubmit2(gpu->Queue(), 1, &submit, frame->done),
                "vkQueueSubmit2");
    }

    void WriteCapture() {
        frame->Wait();
        const std::vector<std::uint8_t> texels =
            viewport->ReadColor();
        const VkExtent2D extent = viewport->Extent();
        const std::filesystem::path path = capture->path;
        capture.reset();
        if (stbi_write_png(
                path.string().c_str(), static_cast<int>(extent.width),
                static_cast<int>(extent.height), 4, texels.data(),
                static_cast<int>(extent.width * 4)) == 0) {
            throw std::runtime_error("Cannot write " + path.string());
        }
    }

    /// Runs the callbacks of `stage` on a copy of the list, so
    /// callbacks may add or remove callbacks.
    void RunCallbacks(Stage stage, const FrameInfo& info) {
        const auto callbacks = frameCallbacks;
        for (const FrameCallback& entry : callbacks) {
            if (entry.stage == stage) {
                entry.callback(self, info);
            }
        }
    }

    std::uint64_t AddCallback(
        Stage stage,
        std::function<void(Viewer&, const FrameInfo&)> callback) {
        const std::uint64_t id = nextCallback++;
        frameCallbacks.push_back(
            {id, stage, std::move(callback), scope});
        return id;
    }

    bool Step();
};

namespace {

void StatsPanel::Draw(Viewer& viewer) {
    const RenderStats stats = viewer.Stats();
    ImGui::TextDisabled("Device");
    ImGui::SameLine(100.0f);
    ImGui::TextUnformatted(viewer.Context().DeviceName().c_str());
    ImGui::TextDisabled("Renderer");
    ImGui::SameLine(100.0f);
    ImGui::TextUnformatted(viewer.Renderer().c_str());
    ImGui::TextDisabled("Frame");
    ImGui::SameLine(100.0f);
    const float framerate = ImGui::GetIO().Framerate;
    ImGui::Text("%.2f ms (%.0f fps)",
                framerate > 0.0f
                    ? 1000.0 / static_cast<double>(framerate)
                    : 0.0,
                static_cast<double>(framerate));
    const std::uint64_t bytes = stats.gpuBytes;
    ImGui::TextDisabled("GPU memory");
    ImGui::SameLine(100.0f);
    ImGui::Text("%.1f MiB",
                static_cast<double>(bytes) / (1024.0 * 1024.0));
    ImGui::TextDisabled("Triangles");
    ImGui::SameLine(100.0f);
    ImGui::Text("%llu",
                static_cast<unsigned long long>(stats.triangles));
    for (const auto& [name, value] : stats.counters) {
        ImGui::TextDisabled("%s", name.c_str());
        ImGui::SameLine(100.0f);
        ImGui::Text("%llu", static_cast<unsigned long long>(value));
    }
    std::vector<float> times(times_.begin(), times_.end());
    if (!times.empty() &&
        ImPlot::BeginPlot("##frametimes", ImVec2(-1.0f, 120.0f),
                          ImPlotFlags_NoLegend |
                              ImPlotFlags_NoMenus)) {
        ImPlot::SetupAxes(nullptr, "ms", ImPlotAxisFlags_NoTickLabels,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0,
                                static_cast<double>(kFrameHistory),
                                ImPlotCond_Always);
        ImPlot::PlotLine("frame", times.data(),
                         static_cast<int>(times.size()));
        ImPlot::EndPlot();
    }
}

}  // namespace

bool Viewer::State::Step() {
    if (!open) {
        return false;
    }
    if (window != nullptr) {
        glfwPollEvents();
        if (glfwWindowShouldClose(window) != 0) {
            open = false;
            return false;
        }
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            return true;
        }
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width != framebufferWidth ||
            height != framebufferHeight) {
            ResizeSwapchain(width, height);
        }
        if (width <= 0 || height <= 0) {
            return true;
        }
    }
    // A profiler frame per viewer frame, recording only while the
    // Profiler panel shows.
    if (!profilerPanel->IsVisible()) {
        profiler::SetEnabled(false);
    }
    profiler::BeginFrame();
    {
        ERNIE_ZONE("Wait for frame");
        frame->Wait();
    }
    ResizeViewport();
    if (capture) {
        capture->rendered = true;
    }

    const auto now = std::chrono::steady_clock::now();
    FrameInfo info;
    info.frame = frames;
    info.time = std::chrono::duration<double>(now - start).count();
    info.delta = std::chrono::duration<double>(now - last).count();
    last = now;
    frameInfo = info;
    RunCallbacks(Stage::Frame, info);
    if (!open) {
        return false;
    }
    ValidateSession();

    if (pendingLayout) {
        ImGui::LoadIniSettingsFromMemory(pendingLayout->c_str(),
                                         pendingLayout->size());
        pendingLayout.reset();
        layoutReady = true;
        userLayout = true;
    }
    ImGui_ImplVulkan_NewFrame();
    if (window != nullptr) {
        ImGui_ImplGlfw_NewFrame();
    } else {
        ImGui::GetIO().DeltaTime =
            static_cast<float>(std::max(info.delta, 1e-4));
    }
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
    ImGuiErrorRecoveryState recovery;
    ImGui::ErrorRecoveryStoreState(&recovery);
    try {
        ERNIE_ZONE("Viewer UI");
        BuildUi();
    } catch (...) {
        ImGuiIO& io = ImGui::GetIO();
        const bool asserting = io.ConfigErrorRecoveryEnableAssert;
        io.ConfigErrorRecoveryEnableAssert = false;
        ImGui::ErrorRecoveryTryToRecoverState(&recovery);
        io.ConfigErrorRecoveryEnableAssert = asserting;
        ImGui::EndFrame();
        throw;
    }
    ImGui::Render();
    if (stats != nullptr) {
        stats->Record(static_cast<float>(info.delta * 1000.0));
    }
    RunCallbacks(Stage::PreRender, info);
    ValidateSession();

    std::optional<Swapchain::Frame> image;
    if (swapchain) {
        image = swapchain->Acquire(frame->acquired);
        if (!image) {
            ResizeSwapchain(framebufferWidth, framebufferHeight);
            return open;
        }
    }

    VkCommandBuffer commands = frame->commands;
    vkResetCommandBuffer(commands, 0);
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commands, &begin);
    gpu->CollectGpuProfile(commands);
    {
        ERNIE_ZONE("Viewer viewport");
        RecordViewport(commands);
    }
    {
        ERNIE_ZONE("Viewer UI draw");
        ERNIE_GPU_ZONE(*gpu, commands, "UI");
        if (image) {
            RecordUi(commands, image->image, image->view,
                     swapchain->Extent());
        } else {
            RecordUi(commands, uiTarget->Color().Handle(),
                     uiTarget->Color().View(), uiTarget->Extent());
        }
    }
    vkEndCommandBuffer(commands);
    {
        ERNIE_ZONE("Submit");
        Submit(commands, image ? &*image : nullptr);
    }
    ++frames;
    ERNIE_FRAME_MARK("Viewer frame");
    ERNIE_ZONE("Present");
    if (image && !swapchain->Present(*image)) {
        ResizeSwapchain(framebufferWidth, framebufferHeight);
    }
    if (capture && capture->rendered) {
        WriteCapture();
    }
    frameSlot = (frameSlot + 1) % frameResources.size();
    frame = &*frameResources[frameSlot];
    RunCallbacks(Stage::PostRender, info);
    return open;
}

Viewer::Viewer(Scene& scene, const ViewerOptions& options)
    : state_(std::make_unique<State>(*this, scene, options)) {
    State& s = *state_;
    if (options.width == 0 || options.height == 0) {
        throw std::invalid_argument(
            "The viewer's window needs a positive size");
    }
    if (!options.hidden) {
        s.OpenWindow();
    }
    s.CreateGpu();
    s.CreateImGui();
    auto stats = std::make_unique<StatsPanel>();
    s.stats = stats.get();
    // Stage above Properties on the right, and Log, Stats and
    // Overlays at the bottom, opening on Stage, Properties and Log.
    s.panels.push_back(MakeStagePanel());
    s.panels.push_back(MakePropertiesPanel());
    s.panels.push_back(MakeSettingsPanel());
    s.panels.push_back(MakeCameraPanel());
    s.panels.push_back(std::move(stats));
    s.panels.push_back(std::make_unique<OverlaysPanel>());
    s.panels.push_back(MakeLogPanel());
    s.panels.push_back(MakeProfilerPanel());
    s.profilerPanel = s.panels.back().get();
    s.builtInPanels = s.panels.size();
    profiler::SetThreadName("Main");
    s.CreateSession();
}

Viewer::~Viewer() = default;

void Viewer::Run() {
    while (Step()) {
    }
}

bool Viewer::Step() {
    return state_->Step();
}

bool Viewer::IsOpen() const {
    return state_->open;
}

void Viewer::Close() {
    state_->open = false;
    if (state_->window != nullptr) {
        glfwSetWindowShouldClose(state_->window, GLFW_TRUE);
    }
}

std::uint64_t Viewer::OnFrame(
    std::function<void(Viewer&, const FrameInfo&)> callback) {
    return state_->AddCallback(State::Stage::Frame,
                               std::move(callback));
}

std::uint64_t Viewer::OnUi(
    std::function<void(Viewer&, const FrameInfo&)> callback) {
    return state_->AddCallback(State::Stage::Ui, std::move(callback));
}

std::uint64_t Viewer::OnPreRender(
    std::function<void(Viewer&, const FrameInfo&)> callback) {
    return state_->AddCallback(State::Stage::PreRender,
                               std::move(callback));
}

std::uint64_t Viewer::OnPostRender(
    std::function<void(Viewer&, const FrameInfo&)> callback) {
    return state_->AddCallback(State::Stage::PostRender,
                               std::move(callback));
}

std::uint64_t Viewer::OnSelection(
    std::function<void(Viewer&, std::span<const Path>)> callback) {
    const std::uint64_t id = state_->nextCallback++;
    state_->selectionCallbacks.push_back(
        {id, std::move(callback), state_->scope});
    return id;
}

void Viewer::RemoveCallback(std::uint64_t id) {
    std::erase_if(state_->frameCallbacks,
                  [id](const auto& entry) { return entry.id == id; });
    std::erase_if(state_->selectionCallbacks,
                  [id](const auto& entry) { return entry.id == id; });
}

Scene& Viewer::GetScene() {
    return state_->scene;
}

std::vector<std::string> Viewer::Renderers() {
    return RenderBackends();
}

const std::string& Viewer::Renderer() const {
    return state_->renderer;
}

void Viewer::SetRenderer(const std::string& name) {
    State& s = *state_;
    if (name == s.renderer) {
        return;
    }
    vkDeviceWaitIdle(s.gpu->Device());
    std::unique_ptr<RenderDelegate> next =
        CreateRenderDelegate(name, *s.gpu, kViewportFormat);
    s.delegate = std::move(next);
    s.renderer = name;
    const std::vector<std::string>& shown =
        s.delegate->Info().viewVars;
    if (std::find(shown.begin(), shown.end(), s.viewVar) ==
        shown.end()) {
        s.viewVar = std::string(render_vars::kLdrColor);
    }
}

const BrdfRegistry& Viewer::Brdfs() const {
    return state_->delegate->Brdfs();
}

RenderStats Viewer::Stats() const {
    return state_->delegate->Stats();
}

void Viewer::Sync() {
    state_->delegate->Sync(state_->scene);
}

void Viewer::ResetSession() {
    state_->ResetSession();
}

void Viewer::OpenScene(const std::filesystem::path& path) {
    {
        Scene check;
        LoadSceneFile(check, path);
        if (check.Has(SessionRoot())) {
            throw std::invalid_argument(
                path.string() +
                ": a scene file cannot describe the viewer's own "
                "prims, under " +
                SessionRoot().String());
        }
    }
    ClearScene();
    LoadSceneFile(state_->scene, path);
    state_->scenePath = path;
    ResetSession();
}

void Viewer::ClearScene() {
    State& s = *state_;
    s.SetSelection({});
    s.history.Reset();
    for (const Path& root : s.scene.Children(Path::Root())) {
        if (root != SessionRoot()) {
            s.scene.Remove(root);
        }
    }
    for (const std::string& name : s.scene.OverlayNames()) {
        if (!name.starts_with("__")) {
            s.scene.RemoveOverlay(name);
        }
    }
    s.orthographicView = -1;
    s.activeCamera = PerspectiveCamera();
    s.settingsSource = {};
    s.cameraChosen = s.lightingChosen = s.settingsChosen = false;
}

std::uint64_t Viewer::NewScope() {
    return state_->nextScope++;
}

std::uint64_t Viewer::Scope() const {
    return state_->scope;
}

void Viewer::SetScope(std::uint64_t scope) {
    state_->scope = scope;
}

void Viewer::RemoveScope(std::uint64_t scope) {
    if (scope == 0) {
        throw std::invalid_argument(
            "Scope 0 is the application's own and cannot be removed");
    }
    State& s = *state_;
    const auto in = [scope](const auto& item) {
        return item.scope == scope;
    };
    std::erase_if(s.frameCallbacks, in);
    std::erase_if(s.selectionCallbacks, in);
    std::erase_if(s.menuItems, in);
    std::erase_if(s.viewportToggles, in);
    std::erase_if(s.hudItems, in);
    std::erase_if(s.viewportMenuItems, in);
    s.hudTime = -kHudRefresh;
    std::vector<std::string> titles;
    for (const auto& [title, added] : s.panelScopes) {
        if (added == scope) {
            titles.push_back(title);
        }
    }
    for (const std::string& title : titles) {
        RemovePanel(title);
    }
}

void Viewer::SaveScene(const std::filesystem::path& path) {
    SaveSceneFile(state_->scene, path);
    state_->scenePath = path;
}

RenderDelegate& Viewer::Delegate() {
    return *state_->delegate;
}

VulkanContext& Viewer::Context() {
    return *state_->gpu;
}

const VulkanContext& Viewer::Context() const {
    return *state_->gpu;
}

const Path& Viewer::SessionRoot() {
    return SessionRootPath();
}

const Path& Viewer::PerspectiveCamera() {
    static const Path path = SessionRootPath().Child("Perspective");
    return path;
}

const std::vector<Path>& Viewer::Selection() const {
    return state_->selection;
}

void Viewer::SetSelection(std::vector<Path> paths) {
    state_->SetSelection(std::move(paths));
}

std::vector<Path> Viewer::Cameras() const {
    return ViewableCameras(state_->scene);
}

const Path& Viewer::ActiveCamera() const {
    return state_->activeCamera;
}

void Viewer::SetActiveCamera(Path camera) {
    const Scene& scene = state_->scene;
    if (!scene.Has(camera) ||
        scene.TypeOf(camera) != tokens::kCamera) {
        throw std::invalid_argument("'" + camera.String() +
                                    "' is not a Camera prim");
    }
    state_->SetActive(camera);
    state_->cameraChosen = true;
}

void Viewer::Frame(std::span<const Path> paths) {
    State& s = *state_;
    s.delegate->Sync(s.scene);
    std::optional<Bounds3> bounds;
    if (!paths.empty()) {
        std::vector<std::optional<Bounds3>> boxes;
        for (const Path& path : paths) {
            if (s.scene.Has(path)) {
                std::optional<Bounds3> extent = s.LocalBounds(path);
                if (extent) {
                    const glm::dmat4 world =
                        s.scene.WorldTransform(path);
                    Bounds3 box{
                        glm::vec3(std::numeric_limits<float>::max()),
                        glm::vec3(
                            std::numeric_limits<float>::lowest())};
                    for (int corner = 0; corner < 8; ++corner) {
                        const glm::dvec3 local(
                            (corner & 1) ? extent->second.x
                                         : extent->first.x,
                            (corner & 2) ? extent->second.y
                                         : extent->first.y,
                            (corner & 4) ? extent->second.z
                                         : extent->first.z);
                        const glm::vec3 point(world *
                                              glm::dvec4(local, 1.0));
                        box.first = glm::min(box.first, point);
                        box.second = glm::max(box.second, point);
                    }
                    boxes.emplace_back(box);
                }
            }
        }
        bounds = Union(boxes);
    }
    s.FrameBounds(bounds ? bounds : s.SceneBounds());
}

Path Viewer::CameraFromView(const Path& path) {
    State& s = *state_;
    if (IsSessionPath(path)) {
        throw std::invalid_argument(
            "'" + path.String() +
            "' is under the viewer's session root");
    }
    if (s.scene.Has(path)) {
        throw std::invalid_argument("'" + path.String() +
                                    "' already exists");
    }
    CameraParams camera =
        CameraParamsOf(s.scene, s.activeCamera, s.ViewportWidth(),
                       s.ViewportHeight());
    s.scene.Define(path, tokens::kCamera);
    camera.transform =
        LocalFromWorld(s.scene, path, camera.transform);
    SetCamera(s.scene, path, camera);
    return path;
}

int Viewer::ViewBand() const {
    return state_->viewBand;
}

void Viewer::SetViewBand(int band) {
    state_->viewBand = band < 0 ? -1 : band;
}

const std::string& Viewer::ViewVar() const {
    return state_->viewVar;
}

void Viewer::SetViewVar(std::string name) {
    const auto& names = kRenderVarNames;
    if (std::find(names.begin(), names.end(), name) == names.end() &&
        !name.starts_with(kAovPrefix)) {
        throw std::invalid_argument("'" + name +
                                    "' is not a render var");
    }
    // AOVs only once the view's chain writes them.
    const std::vector<std::string> shown =
        state_->delegate->ViewVars();
    if (std::find(shown.begin(), shown.end(), name) == shown.end()) {
        throw std::invalid_argument("The " + state_->renderer +
                                    " renderer cannot show '" + name +
                                    "' in the viewport");
    }
    state_->viewVar = std::move(name);
}

void Viewer::Log(LogLevel level, std::string text) {
    constexpr std::size_t kLogLimit = 5000;
    State& s = *state_;
    const double time =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - s.start)
            .count();
    s.log.push_back({time, level, std::move(text)});
    while (s.log.size() > kLogLimit) {
        s.log.pop_front();
    }
}

const std::deque<LogEntry>& Viewer::LogEntries() const {
    return state_->log;
}

void Viewer::ClearLog() {
    state_->log.clear();
}

void Viewer::Edit(const Path& path, std::string_view attribute,
                  Value value, bool open) {
    state_->history.Set(state_->scene, path, attribute,
                        std::move(value), open);
}

void Viewer::EditLocalTransform(const Path& path,
                                const glm::dmat4& local, bool open) {
    state_->history.Set(
        state_->scene, path,
        LocalTransformEdits(state_->scene, path, local), open);
}

void Viewer::ClearAttribute(const Path& path,
                            std::string_view attribute) {
    state_->history.Clear(state_->scene, path, attribute);
}

bool Viewer::Undo() {
    return state_->history.Undo(state_->scene);
}

bool Viewer::Redo() {
    return state_->history.Redo(state_->scene);
}

EditHistory& Viewer::History() {
    return state_->history;
}

Path Viewer::Create(std::string_view type) {
    State& s = *state_;
    const CreatableType* creatable = FindCreatable(type);
    if (creatable == nullptr) {
        throw std::invalid_argument("The viewer cannot create a '" +
                                    std::string(type) + "'");
    }
    // Under a selected group prim, else where the type usually goes.
    Path parent(creatable->parent);
    if (s.selection.size() == 1 &&
        !IsSessionPath(s.selection.front()) &&
        s.scene.Has(s.selection.front())) {
        const std::string& selected =
            s.scene.TypeOf(s.selection.front());
        if (selected == tokens::kXform ||
            selected == tokens::kScope || selected.empty()) {
            parent = s.selection.front();
        }
    }
    Path path = parent.Child(creatable->name);
    for (int n = 1; s.scene.Has(path); ++n) {
        path = parent.Child(std::string(creatable->name) +
                            std::to_string(n));
    }
    CreateContext context;
    context.target = glm::dvec3(s.controller.target);
    // Default sizes at the view's scale: a power of ten near a tenth
    // of its distance.
    context.scale = std::pow(
        10.0, std::round(std::log10(std::max(
                  static_cast<double>(s.controller.distance) / 10.0,
                  1e-3))));
    context.parentWorld = s.scene.Has(parent)
                              ? s.scene.WorldTransform(parent)
                              : glm::dmat4(1.0);
    context.camera =
        CameraParamsOf(s.scene, s.activeCamera, s.ViewportWidth(),
                       s.ViewportHeight());
    if (!IsSessionPath(s.activeCamera)) {
        context.activeCamera = s.activeCamera;
    }
    Scene scratch;
    AuthorDefaultPrim(scratch, path, type, context);
    s.history.BeginGroup();
    try {
        s.history.Add(s.scene, scratch, path);
        if (type == tokens::kMaterial) {
            for (const Path& selected : s.selection) {
                if (s.scene.Has(selected) &&
                    IsMeshType(s.scene.TypeOf(selected))) {
                    s.history.Set(s.scene, selected,
                                  tokens::kMaterialBinding, path);
                }
            }
        }
    } catch (...) {
        s.history.EndGroup();
        throw;
    }
    s.history.EndGroup();
    s.SetSelection({path});
    Log(LogLevel::Info, "Created " + path.String());
    return path;
}

void Viewer::Delete(std::span<const Path> paths) {
    State& s = *state_;
    const std::vector<Path> doomed(paths.begin(), paths.end());
    s.history.BeginGroup();
    for (const Path& path : doomed) {
        if (!IsSessionPath(path) && s.history.Remove(s.scene, path)) {
            Log(LogLevel::Info, "Deleted " + path.String());
        }
    }
    s.history.EndGroup();
    s.SetSelection({});
}

namespace {

/// Copies the prim at `from` of `scene` and its descendants, with
/// their attributes and tensors (shared), into `scratch` under `to`.
void CopySubtree(const Scene& scene, const Path& from, Scene& scratch,
                 const Path& to) {
    const std::string prefix = from.String();
    for (const Path& prim : scene.Prims()) {
        if (prim != from && !prim.HasPrefix(from)) {
            continue;
        }
        const Path target(to.String() +
                          prim.String().substr(prefix.size()));
        scratch.Define(target, scene.TypeOf(prim));
        for (const std::string& attribute : scene.Attributes(prim)) {
            if (const Value* value = scene.Get(prim, attribute)) {
                scratch.Set(target, attribute, *value);
            } else if (const Tensor* tensor =
                           scene.Read(prim, attribute)) {
                scratch.Write(target, attribute, *tensor);
            }
        }
    }
}

}  // namespace

std::vector<Path> Viewer::Duplicate(std::span<const Path> paths) {
    State& s = *state_;
    const std::vector<Path> originals(paths.begin(), paths.end());
    std::vector<Path> copies;
    s.history.BeginGroup();
    for (const Path& path : originals) {
        if (IsSessionPath(path) || path.IsRoot() ||
            !s.scene.Has(path)) {
            continue;
        }
        Path copy;
        for (int k = 1;; ++k) {
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), "_%02d", k);
            copy = path.Parent().Child(std::string(path.Name()) +
                                       suffix);
            if (!s.scene.Has(copy)) {
                break;
            }
        }
        Scene scratch;
        CopySubtree(s.scene, path, scratch, copy);
        s.history.Add(s.scene, scratch, copy);
        copies.push_back(copy);
        Log(LogLevel::Info,
            "Duplicated " + path.String() + " as " + copy.String());
    }
    s.history.EndGroup();
    if (!copies.empty()) {
        s.SetSelection(copies);
    }
    return copies;
}

Path Viewer::Rename(const Path& path, std::string_view name) {
    State& s = *state_;
    if (path.IsRoot() || IsSessionPath(path) || !s.scene.Has(path)) {
        throw std::invalid_argument("Cannot rename '" +
                                    path.String() + "'");
    }
    if (name == path.Name()) {
        return path;
    }
    const Path renamed =
        path.Parent().Child(name);  // validates the name
    if (s.scene.Has(renamed)) {
        throw std::invalid_argument("'" + renamed.String() +
                                    "' already exists");
    }
    Scene scratch;
    CopySubtree(s.scene, path, scratch, renamed);
    const bool wasSelected =
        std::find(s.selection.begin(), s.selection.end(), path) !=
        s.selection.end();
    s.history.BeginGroup();
    s.history.Add(s.scene, scratch, renamed);
    s.history.Remove(s.scene, path);
    s.history.EndGroup();
    if (wasSelected) {
        std::vector<Path> selection = s.selection;
        std::replace(selection.begin(), selection.end(), path,
                     renamed);
        s.SetSelection(std::move(selection));
    }
    Log(LogLevel::Info,
        "Renamed " + path.String() + " to " + renamed.String());
    return renamed;
}

Viewer::Gizmo Viewer::GetGizmo() const {
    return state_->gizmo;
}

void Viewer::SetGizmo(Gizmo gizmo) {
    state_->gizmo = gizmo;
}

bool Viewer::GizmoLocal() const {
    return state_->gizmoLocal;
}

void Viewer::SetGizmoLocal(bool local) {
    state_->gizmoLocal = local;
}

Path Viewer::Pick(std::uint32_t x, std::uint32_t y) {
    const Path picked = state_->delegate->Pick(x, y);
    return IsSessionPath(picked) ? Path() : picked;
}

Viewer::Exposure Viewer::GetExposure() const {
    return state_->exposure;
}

void Viewer::SetExposure(Exposure exposure) {
    state_->exposure = exposure;
}

double Viewer::ExposureCompensation() const {
    return state_->compensation;
}

void Viewer::SetExposureCompensation(double stops) {
    state_->compensation = stops;
}

std::optional<double> Viewer::MeteredEv() const {
    const std::optional<double> luminance =
        state_->delegate->MeteredLuminance();
    if (!luminance || !(*luminance > 0.0)) {
        return std::nullopt;
    }
    return std::log2(*luminance * 100.0 / 12.5);
}

bool Viewer::AutoExposing() const {
    return state_->AutoExposing();
}

bool Viewer::GetVsync() const {
    return state_->swapchain != nullptr && state_->swapchain->Vsync();
}

void Viewer::SetVsync(bool vsync) {
    if (state_->swapchain != nullptr) {
        state_->swapchain->SetVsync(vsync);
        ImGui_ImplVulkan_SetMinImageCount(
            state_->swapchain->MinImageCount());
    }
}

Lighting Viewer::GetLighting() const {
    return state_->lighting;
}

void Viewer::SetLighting(Lighting lighting) {
    state_->lighting = lighting;
    state_->lightingChosen = true;
}

std::optional<Wireframe> Viewer::GetWireframe() const {
    return state_->wireframe;
}

void Viewer::SetWireframe(std::optional<Wireframe> mode) {
    state_->wireframe = mode;
}

const Path& Viewer::Settings() const {
    return state_->settingsSource;
}

void Viewer::SetSettings(Path settings) {
    const Scene& scene = state_->scene;
    if (!scene.Has(settings) ||
        scene.TypeOf(settings) != tokens::kRenderSettings ||
        IsSessionPath(settings)) {
        throw std::invalid_argument("'" + settings.String() +
                                    "' is not a RenderSettings prim");
    }
    state_->settingsSource = std::move(settings);
    state_->settingsChosen = true;
}

void Viewer::Capture(const std::filesystem::path& path,
                     std::string var) {
    const std::vector<std::string> shown =
        state_->delegate->ViewVars();
    if (!var.empty() &&
        std::find(shown.begin(), shown.end(), var) == shown.end()) {
        throw std::invalid_argument("The " + state_->renderer +
                                    " renderer cannot show '" + var +
                                    "' in the viewport");
    }
    state_->capture =
        State::CaptureRequest{path, std::move(var), false};
}

void Viewer::CaptureWindow(const std::filesystem::path& path) {
    State& s = *state_;
    if (s.uiTarget == nullptr) {
        throw std::logic_error(
            "Only a hidden viewer can capture its window");
    }
    vkDeviceWaitIdle(s.gpu->Device());
    const std::vector<std::uint8_t> texels = s.uiTarget->ReadColor();
    const VkExtent2D extent = s.uiTarget->Extent();
    if (stbi_write_png(
            path.string().c_str(), static_cast<int>(extent.width),
            static_cast<int>(extent.height), 4, texels.data(),
            static_cast<int>(extent.width * 4)) == 0) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

Panel& Viewer::AddPanel(std::unique_ptr<Panel> panel) {
    if (!panel) {
        throw std::invalid_argument("AddPanel needs a panel");
    }
    for (const auto& existing : state_->panels) {
        if (existing->Title() == panel->Title()) {
            throw std::invalid_argument("A panel titled '" +
                                        panel->Title() + "' exists");
        }
    }
    if (panel->Title() == kViewportTitle) {
        throw std::invalid_argument("'Viewport' is reserved");
    }
    const Dock dock = panel->GetDock();
    state_->panelScopes[panel->Title()] = state_->scope;
    state_->panels.push_back(std::move(panel));
    if (!state_->userLayout && dock != Dock::Floating &&
        !state_->dockNodes.contains(dock)) {
        state_->layoutReady = false;
    }
    return *state_->panels.back();
}

Panel& Viewer::AddPanel(std::string title, Dock dock,
                        std::function<void(Viewer&)> draw) {
    return AddPanel(std::make_unique<FunctionPanel>(
        std::move(title), dock, std::move(draw)));
}

Panel* Viewer::FindPanel(std::string_view title) {
    for (const auto& panel : state_->panels) {
        if (panel->Title() == title) {
            return panel.get();
        }
    }
    return nullptr;
}

void Viewer::OpenFileDialog(
    FileDialogOptions options,
    std::function<void(Viewer&, const std::filesystem::path&)> done) {
    state_->fileDialog.Open(std::move(options),
                            [this, done = std::move(done)](
                                const std::filesystem::path& path) {
                                if (done) {
                                    done(*this, path);
                                }
                            });
}

FileDialog& Viewer::GetFileDialog() {
    return state_->fileDialog;
}

void Viewer::FocusPanel(std::string_view title) {
    Panel* panel = FindPanel(title);
    if (panel == nullptr) {
        throw std::invalid_argument("No panel titled '" +
                                    std::string(title) + "'");
    }
    panel->SetVisible(true);
    std::vector<std::string>& requests = state_->focusRequests;
    requests.erase(
        std::remove(requests.begin(), requests.end(), title),
        requests.end());
    requests.emplace_back(title);
}

WidgetPanel& Viewer::AddWidgetPanel(std::string title, Dock dock) {
    return static_cast<WidgetPanel&>(AddPanel(
        std::make_unique<WidgetPanel>(std::move(title), dock)));
}

void Viewer::RemovePanel(const std::string& title) {
    auto& panels = state_->panels;
    const auto found = std::find_if(
        panels.begin(), panels.end(),
        [&](const auto& panel) { return panel->Title() == title; });
    if (found == panels.end()) {
        throw std::invalid_argument("No panel titled '" + title +
                                    "'");
    }
    if (found - panels.begin() <
        static_cast<std::ptrdiff_t>(state_->builtInPanels)) {
        (*found)->SetVisible(false);
        return;
    }
    std::vector<std::string>& requests = state_->focusRequests;
    requests.erase(
        std::remove(requests.begin(), requests.end(), title),
        requests.end());
    state_->drawnPanels.erase(title);
    state_->panelScopes.erase(title);
    panels.erase(found);
}

void Viewer::AddMenuItem(const std::string& menu,
                         const std::string& label,
                         std::function<void(Viewer&)> action) {
    if (menu.empty() || label.empty()) {
        throw std::invalid_argument(
            "Menu items need a menu and a label");
    }
    state_->menuItems.push_back(
        {menu, label, std::move(action), state_->scope});
}

const ViewportDisplay& Viewer::GetViewportDisplay() const {
    return state_->display;
}

void Viewer::SetViewportDisplay(const ViewportDisplay& display) {
    state_->display = display;
    state_->hudTime = -kHudRefresh;
}

std::uint64_t Viewer::AddViewportToggle(
    std::string label, bool shown, std::string overlay,
    std::function<void(Viewer&, bool)> changed) {
    if (label.empty()) {
        throw std::invalid_argument("Viewport toggles need a label");
    }
    State& s = *state_;
    const std::uint64_t id = s.nextCallback++;
    s.viewportToggles.push_back({id, std::move(label), shown,
                                 std::move(overlay),
                                 std::move(changed), s.scope});
    return id;
}

namespace {

template <typename Items>
auto& FindViewportItem(Items& items, std::uint64_t id,
                       const char* kind) {
    const auto found = std::find_if(
        items.begin(), items.end(),
        [id](const auto& item) { return item.id == id; });
    if (found == items.end()) {
        throw std::invalid_argument("No " + std::string(kind) +
                                    " has id " + std::to_string(id));
    }
    return *found;
}

}  // namespace

bool Viewer::ViewportToggleShown(std::uint64_t id) const {
    return FindViewportItem(state_->viewportToggles, id,
                            "viewport toggle")
        .shown;
}

void Viewer::SetViewportToggleShown(std::uint64_t id, bool shown) {
    auto& toggle = FindViewportItem(state_->viewportToggles, id,
                                    "viewport toggle");
    if (toggle.shown == shown) {
        return;
    }
    toggle.shown = shown;
    if (toggle.changed) {
        // Copied: the callback may remove the toggle.
        const auto changed = toggle.changed;
        changed(*this, shown);
    }
}

std::uint64_t
Viewer::AddHudItem(std::string label,
                   std::function<std::string(Viewer&)> value,
                   bool shown) {
    if (label.empty() || !value) {
        throw std::invalid_argument(
            "Heads-up display items need a label and a value");
    }
    State& s = *state_;
    const std::uint64_t id = s.nextCallback++;
    s.hudItems.push_back(
        {id, std::move(label), std::move(value), shown, s.scope});
    s.hudTime = -kHudRefresh;
    return id;
}

bool Viewer::HudItemShown(std::uint64_t id) const {
    return FindViewportItem(state_->hudItems, id,
                            "heads-up display item")
        .shown;
}

void Viewer::SetHudItemShown(std::uint64_t id, bool shown) {
    FindViewportItem(state_->hudItems, id, "heads-up display item")
        .shown = shown;
    state_->hudTime = -kHudRefresh;
}

std::uint64_t Viewer::AddViewportMenuItem(
    std::string label, std::function<void(Viewer&)> action,
    std::function<bool(const Viewer&)> checked) {
    if (label.empty() || !action) {
        throw std::invalid_argument(
            "Viewport menu items need a label and an action");
    }
    State& s = *state_;
    const std::uint64_t id = s.nextCallback++;
    s.viewportMenuItems.push_back({id, std::move(label),
                                   std::move(action),
                                   std::move(checked), s.scope});
    return id;
}

void Viewer::RemoveViewportItem(std::uint64_t id) {
    State& s = *state_;
    const auto matches = [id](const auto& item) {
        return item.id == id;
    };
    std::erase_if(s.viewportToggles, matches);
    std::erase_if(s.hudItems, matches);
    std::erase_if(s.viewportMenuItems, matches);
    s.hudTime = -kHudRefresh;
}

void Viewer::SaveLayout(const std::filesystem::path& path) const {
    std::size_t size = 0;
    const char* data = ImGui::SaveIniSettingsToMemory(&size);
    std::ofstream file(path, std::ios::binary);
    if (!file.write(data, static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

void Viewer::LoadLayout(const std::filesystem::path& path) {
    state_->pendingLayout = ReadFile(path);
}

const Theme& Viewer::GetTheme() const {
    return state_->theme;
}

void Viewer::SetTheme(const Theme& theme) {
    state_->theme = theme;
    ApplyTheme(theme, state_->contentScale, ImGui::GetStyle());
}

}  // namespace ernie::app
