// SPDX-License-Identifier: BSD-3-Clause
/// @file ui.h
/// @brief The viewer's UI, in sections: the panels and what they share; the theme and widgets;
/// the orbit camera; the shell, which draws the menu bar, the docked panels around the viewport
/// and the heads-up display; and the presenters frames go to. The shell draws ImGui windows only,
/// so it runs without a device or a window.
#pragma once

#include "ernie/app/capture.h"
#include "ernie/app/demo.h"
#include "ernie/app/timeline.h"
#include "ernie/device/image.h"
#include "ernie/device/queue.h"
#include "ernie/geometry/bounds.h"
#include "ernie/scene/edit_history.h"
#include "ernie/scene/path.h"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

struct GLFWwindow;

namespace ernie {
class DeviceContext;
class Scene;
}  // namespace ernie

namespace ernie::app {

// Panels

struct Theme;

/// @brief Where a panel docks in the default layout.
enum class Dock : std::uint8_t {
    Left,
    /// The right column's upper part; all of it when nothing docks below.
    Right,
    /// The right column's lower part.
    RightLower,
    Bottom,
    /// Not docked: a free window.
    Floating,
};

/// @brief A dockable window. Panels take what they show and edit in their constructors, and draw
/// with Dear ImGui in `Draw`, between the window's `Begin` and `End`.
class Panel {
  public:
    /// @param title Its title, also its id in saved layouts.
    Panel(std::string title, Dock dock, bool visible = true)
        : title_(std::move(title)), dock_(dock), visible_(visible) {}
    virtual ~Panel() = default;
    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;

    const std::string& Title() const {
        return title_;
    }
    /// @return Where it docks in the default layout; a saved layout overrides it.
    Dock DefaultDock() const {
        return dock_;
    }
    bool Visible() const {
        return visible_;
    }
    void SetVisible(bool visible) {
        visible_ = visible;
    }

    virtual void Draw() = 0;
    /// @brief Called on frames it is not drawn: closed, or a tab behind another.
    virtual void Undrawn() {}

  private:
    std::string title_;
    Dock dock_;
    bool visible_;
};

/// @brief Selected prims in the order they were picked.
class Selection {
  public:
    const std::vector<Path>& Paths() const {
        return paths_;
    }
    bool Empty() const {
        return paths_.empty();
    }
    bool Contains(const Path& path) const {
        return std::ranges::find(paths_, path) != paths_.end();
    }
    /// @brief Selects `path` alone.
    void Set(const Path& path) {
        paths_ = {path};
    }
    /// @brief Adds `path`, or removes it when selected.
    void Toggle(const Path& path) {
        if (const auto found = std::ranges::find(paths_, path); found != paths_.end()) {
            paths_.erase(found);
        } else {
            paths_.push_back(path);
        }
    }
    void Clear() {
        paths_.clear();
    }

  private:
    std::vector<Path> paths_;
};

enum class Severity : std::uint8_t { Info, Warning, Error };

struct Message {
    /// Seconds since the log was made.
    double time = 0.0;
    Severity severity = Severity::Info;
    std::string text;
};

/// @brief The viewer's newest messages, oldest first, up to a limit, as the Log panel lists them.
class MessageLog {
  public:
    explicit MessageLog(std::size_t limit = 10'000) : limit_(limit) {}

    void Add(Severity severity, std::string text);
    void Info(std::string text) {
        Add(Severity::Info, std::move(text));
    }
    void Error(std::string text) {
        Add(Severity::Error, std::move(text));
    }
    void Clear() {
        messages_.clear();
    }
    const std::deque<Message>& Messages() const {
        return messages_;
    }

  private:
    std::size_t limit_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    std::deque<Message> messages_;
};

/// @return The Log panel over `log`, filtered by severity and text.
std::unique_ptr<Panel> MakeLogPanel(MessageLog& log, const Theme& theme);

/// @return The Stage panel: the scene's prims as a tree, picking the selection.
std::unique_ptr<Panel> MakeStagePanel(const Scene& scene, Selection& selection);

/// @return The prims whose path contains `filter`, ignoring case, and their ancestors: what the
/// Stage panel lists for it. Empty for an empty filter, which lists everything.
std::unordered_set<Path> StageMatches(const Scene& scene, std::string_view filter);

/// @return The Properties panel: the first selected prim's schemas and their attributes, edited
/// through `history` as one undo step an edit, a drag one step; failures go to `log`.
std::unique_ptr<Panel> MakePropertiesPanel(Scene& scene, const Selection& selection,
                                           EditHistory& history, MessageLog& log);

/// @return The Info panel: the scene's prims by type and bounds, and the frame time, the device's
/// memory budget and the bytes it has moved.
/// @param device The device in use; null for none.
std::unique_ptr<Panel> MakeInfoPanel(const Scene& scene, const DeviceContext* device);

/// @return The Devices panel: the device in use, or none, and every Vulkan device installed.
std::unique_ptr<Panel> MakeDevicePanel(const DeviceContext* device);

/// @return The Profiler panel over Ernie's zone profiler, saving exports to the working directory
/// and saying so in `log`.
/// @param alwaysRecord Whether zones record while the panel is hidden too.
std::unique_ptr<Panel> MakeProfilerPanel(MessageLog& log, bool alwaysRecord);

class Shell;

/// @return The Settings panel: a Render tab editing the scene's render settings, and a Viewer tab
/// of the viewport's own settings.
std::unique_ptr<Panel> MakeSettingsPanel(Scene& scene, Shell& shell);

/// @return The Demos panel: the registered demos by category, the running one's controls,
/// description and options.
std::unique_ptr<Panel> MakeDemosPanel(Shell& shell);

/// @return The Capture panel: where captures go and how they are named, the screenshot, image
/// sequences encoded into video, and the recent captures.
std::unique_ptr<Panel> MakeCapturePanel(Shell& shell);

/// @brief Draws the Keyboard and Mouse window while `open`; its close button clears it.
void DrawControlsWindow(bool& open);

/// @brief Draws the About Ernie window while `open`; its close button clears it.
/// @param device The name of the device in use; empty for none.
void DrawAboutWindow(bool& open, std::string_view device);

// Theme and widgets

/// @brief Colours and metrics of the viewer's UI: a classic scientific desktop look, after PANGU
/// v7's GUI, with grey-green windows, white sunken fields, one-pixel edges, square corners and a
/// navy accent. Colours are sRGB-encoded RGBA.
struct Theme {
    /// Tab bar, menu bar, behind the docks.
    glm::vec4 background{0.788f, 0.800f, 0.773f, 1.0f};
    /// Panel bodies.
    glm::vec4 panel{0.831f, 0.843f, 0.816f, 1.0f};
    /// Section bars, table headers.
    glm::vec4 header{0.776f, 0.788f, 0.761f, 1.0f};
    /// Input fields, combos.
    glm::vec4 field{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 fieldHovered{0.957f, 0.965f, 0.945f, 1.0f};
    glm::vec4 button{0.863f, 0.875f, 0.847f, 1.0f};
    /// A pressed or chosen button.
    glm::vec4 buttonLit{0.698f, 0.761f, 0.835f, 1.0f};
    /// Lines between areas.
    glm::vec4 separator{0.667f, 0.678f, 0.651f, 1.0f};
    /// Frames', tabs' and popups' edges.
    glm::vec4 border{0.553f, 0.565f, 0.537f, 1.0f};
    glm::vec4 text{0.0f, 0.0f, 0.0f, 1.0f};
    /// Labels, types and disabled text.
    glm::vec4 textMuted{0.333f, 0.345f, 0.325f, 1.0f};
    /// Selected and hovered rows.
    glm::vec4 selection{0.627f, 0.741f, 0.863f, 1.0f};
    /// Checked menu items, links, active tools.
    glm::vec4 accent{0.122f, 0.290f, 0.541f, 1.0f};
    glm::vec4 warning{0.910f, 0.698f, 0.235f, 1.0f};
    glm::vec4 error{0.886f, 0.341f, 0.318f, 1.0f};
    /// Menus and popups.
    glm::vec4 popup{0.831f, 0.843f, 0.816f, 1.0f};
    glm::vec4 checkBox{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 checkMark{0.0f, 0.0f, 0.0f, 1.0f};
    /// The viewport's colour where nothing is rendered.
    glm::vec4 viewport{0.18f, 0.19f, 0.20f, 1.0f};
    /// The viewport's toolbar pills and heads-up box, over the image, and their edges.
    glm::vec4 overlay{0.831f, 0.843f, 0.816f, 0.92f};
    glm::vec4 overlayBorder{0.553f, 0.565f, 0.537f, 1.0f};
    /// The pills' corner radius, in pixels at a content scale of 1.
    float pillRounding = 0.0f;
    /// Text size, in pixels at a content scale of 1.
    float fontSize = 14.0f;
    /// Corner radius of frames, tabs and popups, in pixels.
    float rounding = 0.0f;
    /// Fields' and buttons' edge width, in pixels at a content scale of 1.
    float frameBorder = 1.0f;
    /// Padding inside frames, in pixels.
    glm::vec2 framePadding{6.0f, 3.0f};
    /// Spacing between items, in pixels.
    glm::vec2 itemSpacing{6.0f, 4.0f};
};

/// @brief Resets the current ImGui style and writes a theme into it.
/// @param contentScale Scales sizes and fonts, for high-density displays.
void ApplyTheme(const Theme& theme, float contentScale = 1.0f);

inline ImVec4 ToImVec4(const glm::vec4& colour) {
    return {colour.r, colour.g, colour.b, colour.a};
}

/// @brief Draws a label and a value on one line, the value muted and wrapped to the window.
void LabelValue(std::string_view label, std::string_view value);

/// @brief Fills the next cells of the current table row with `texts`.
void TableTexts(std::initializer_list<std::string_view> texts);

/// @return Bytes as a short size with a binary unit, such as "1.5 GiB".
std::string FormatBytes(std::uint64_t bytes);

// Orbit camera

/// @brief The viewer's camera: an eye orbiting a target point in a z-up world, in double
/// precision as the scene's transforms are, so it orbits as smoothly far from the origin as at it.
/// Yaw turns about +z (0 looks along +y, positive turns clockwise seen from above); pitch is the
/// eye's elevation above the target's horizontal plane.
struct OrbitCamera {
    glm::dvec3 target{0.0};
    /// Radians.
    double yaw = 0.0;
    /// Radians, within [kMinPitch, kMaxPitch].
    double pitch = 0.6;
    /// Metres from the eye to the target, within [kMinDistance, kMaxDistance].
    double distance = 10.0;
    /// Radians.
    double verticalFov = 0.785398;

    /// Short of straight down and up, where the view's up vector degenerates.
    static constexpr double kMinPitch = -1.55;
    static constexpr double kMaxPitch = 1.55;
    /// A millimetre to about 100,000 light-years.
    static constexpr double kMinDistance = 1e-3;
    static constexpr double kMaxDistance = 1e21;

    glm::dvec3 Eye() const;
    /// @return The camera-to-world transform of the eye looking at the target, +z up.
    glm::dmat4 Transform() const;
    /// @brief Puts the eye where a camera is, looking where it looks, the target `distance` ahead;
    /// a roll about the view axis is dropped, as the orbit keeps +z up.
    void LookFrom(const glm::dmat4& cameraToWorld);

    /// @brief Turns by yaw and pitch radians, clamping pitch.
    void Orbit(double deltaYaw, double deltaPitch);
    /// @brief Moves the target parallel to the ground by a drag of (`dx`, `dy`) pixels in a view
    /// `height` pixels tall, so the ground at the target follows the cursor.
    void Pan(double dx, double dy, double height);
    /// @brief Moves the target across the view: sideways as `Pan`, and up and down along +z.
    void PanVertical(double dx, double dy, double height);
    /// @brief Moves closer by `steps` scroll steps; negative moves away.
    void Zoom(double steps);
    /// @brief Aims at the box's centre from the current direction, at a distance where its
    /// bounding sphere fits the vertical field of view with a small margin. An empty box leaves
    /// the camera as it is.
    void Frame(const Box& box);
};

// Shell

/// @brief The default layout's dock sizes, as fractions of the space the docks placed before them
/// leave (left, then the right column, then the bottom).
struct DockSizes {
    float left = 0.20f;
    float right = 0.26f;
    /// `Dock::RightLower`'s share of the right column's height.
    float rightLower = 0.45f;
    float bottom = 0.25f;
};

/// @brief What lights the viewport, as Kit's lighting menu offers: the scene's own lights, a light
/// from the eye, or a default sun.
enum class Lighting : std::uint8_t { Stage, Camera, Default };

/// @return The lighting's menu label.
inline const char* LightingName(Lighting lighting) {
    switch (lighting) {
    case Lighting::Stage:
        return "Stage Lights";
    case Lighting::Camera:
        return "Camera Light";
    case Lighting::Default:
        break;
    }
    return "Default Light";
}

/// @brief How captures are named and where they go, as the Capture panel sets them.
struct CaptureSettings {
    std::filesystem::path folder;
    /// `ExpandCaptureName`'s pattern.
    std::string name = "{scene}_{camera}_{date}_{time}";
    VideoFormat video = VideoFormat::Mp4;
    /// Whether a sequence's frames stay once encoded into video.
    bool keepFrames = true;
};

/// @brief A sequence to record: the time codes from `start` to `end`, every `every`th frame.
struct SequenceRequest {
    double start = 0.0;
    double end = 0.0;
    int every = 1;
    VideoFormat video = VideoFormat::None;
    bool keepFrames = true;
};

/// @brief What the viewport shows over its image, as its eye menu toggles it, and how it is lit.
struct ViewportDisplay {
    bool toolbar = true;
    bool hud = true;
    bool hudFps = true;
    bool hudDeviceMemory = true;
    bool hudHostMemory = true;
    bool hudResolution = true;
    bool hudCamera = false;
    Lighting lighting = Lighting::Camera;
    /// The timeline strip along the viewport's bottom.
    bool timeline = true;
    /// Eco mode, after Omniverse's: the viewport stops rendering once nothing has changed for
    /// `ecoFrames` frames, and the UI keeps running over its last image.
    bool eco = true;
    int ecoFrames = 500;
};

struct ShellOptions {
    Theme theme;
    DockSizes docks;
    /// Record profiler zones every frame, not only while the Profiler panel shows.
    bool profile = false;
    /// What the viewport shows over its image to begin with.
    ViewportDisplay display;
};

/// @brief The panels and controls over a scene. Panels read the scene and the menus edit it
/// through an undo history; its owner keeps it updated (`Scene::Update`) between frames.
///
/// The menus: File opens and saves scene files; Edit undoes, redoes, duplicates, deletes and
/// frames; Create defines new prims with defaults that show, under the selected Xform or Scope;
/// Window shows panels; Help links to the documentation and shows the controls and About.
///
/// Panels dock around the dock space's central node, which the viewport shows through. The default
/// layout is built on the first frame and after Window > Reset Layout; panels shown later dock
/// where they ask. Over the viewport, a left drag orbits the camera, a right drag pans along the
/// ground, a middle drag pans across the view and the wheel zooms; a drag that starts over the
/// viewport keeps control until its button is released.
class Shell {
  public:
    /// @param device The device in use, for the Devices panel; null for none.
    Shell(Scene& scene, const DeviceContext* device, const ShellOptions& options = {});

    /// @brief Draws this frame's UI, between `ImGui::NewFrame` and `ImGui::Render`.
    void Draw();
    /// @return Whether the viewer should keep running: false once the user quits.
    bool Open() const {
        return open_;
    }
    /// @brief Aims the camera at the selection, or at the whole scene when nothing is selected.
    void Frame();

    /// @return The viewport's position and size on the screen: the dock space's central node, less
    /// the timeline's strip along its bottom while it shows.
    std::pair<ImVec2, ImVec2> Viewport() const;
    /// @return The panel titled `title`, or null.
    Panel* FindPanel(std::string_view title) const;
    MessageLog& Log() {
        return log_;
    }
    Selection& Selected() {
        return selection_;
    }
    OrbitCamera& Camera() {
        return camera_;
    }
    const Theme& GetTheme() const {
        return theme_;
    }
    EditHistory& History() {
        return history_;
    }
    ViewportDisplay& Display() {
        return display_;
    }
    /// @brief Looks through a scene camera: the orbit takes its pose and field of view, and
    /// navigating moves it. An empty path looks through the viewer's own perspective camera.
    /// @throws std::invalid_argument for a path that is no Camera.
    void LookThrough(const Path& camera);
    /// @return The scene camera looked through; empty for the viewer's own.
    const Path& ActiveCamera() const {
        return activeCamera_;
    }
    /// @brief Shows whether eco mode paused the viewport, as the viewer decides it.
    void SetEcoPaused(bool paused) {
        ecoPaused_ = paused;
    }

    app::Timeline& Timeline() {
        return timeline_;
    }

    /// @brief Replaces the scene with a demo's, forgetting the history and the selection.
    /// @throws what the demo's constructor throws, leaving the scene empty.
    void LoadDemo(const DemoInfo& demo);
    /// @return The demo running, or null.
    Demo* CurrentDemo() const {
        return demo_.get();
    }
    /// @return Its registration, or null.
    const DemoInfo* CurrentDemoInfo() const {
        return demo_ ? &demoInfo_ : nullptr;
    }
    /// Whether the demo holds still; `StepDemo` moves it a step anyway.
    bool demoPaused = false;
    void StepDemo();
    /// @brief Builds the demo again from scratch.
    void ResetDemo();

    CaptureSettings& Capture() {
        return capture_;
    }
    /// @return This frame's screenshot request, once.
    bool TakeScreenshot() {
        return std::exchange(screenshot_, false);
    }
    void RequestScreenshot() {
        screenshot_ = true;
    }
    /// @return This frame's sequence request, once.
    std::optional<SequenceRequest> TakeSequence() {
        return std::exchange(sequence_, std::nullopt);
    }
    void RequestSequence(const SequenceRequest& sequence) {
        sequence_ = sequence;
    }
    /// @brief Shows a sequence's progress, as the viewer records it: frames done of total, or
    /// none once it stops; and whether its video is still encoding.
    void SetRecording(std::optional<std::pair<std::int64_t, std::int64_t>> progress,
                      bool encoding) {
        recording_ = progress;
        encoding_ = encoding;
    }
    std::optional<std::pair<std::int64_t, std::int64_t>> Recording() const {
        return recording_;
    }
    bool Encoding() const {
        return encoding_;
    }
    /// Whether the Capture panel's Stop was pressed this frame.
    bool stopRecording = false;
    /// @brief Lists a file a capture wrote, for the Capture panel's Recent list.
    void AddRecentCapture(const std::filesystem::path& path);
    const std::vector<std::filesystem::path>& RecentCaptures() const {
        return recentCaptures_;
    }
    /// @return What a capture's name pattern names now.
    CaptureNameFields CaptureFields() const;
    /// @return The heads-up display's lines, as "label: value", as last refreshed.
    const std::vector<std::string>& HudLines() const {
        return hudLines_;
    }

    /// @brief Defines a prim of a type the Create menu lists, with its defaults, under the selected
    /// Xform or Scope or else its default parent, as one undo step, and selects it.
    /// @return Its path.
    /// @throws std::invalid_argument for a type the menu does not list.
    Path Create(std::string_view type);
    /// @brief Copies each selected prim beside itself, as one undo step, selecting the copies.
    void DuplicateSelection();
    /// @brief Removes the selected prims, as one undo step.
    void DeleteSelection();
    /// @brief Replaces the scene with a scene file's, forgetting the history and the selection.
    /// @throws std::runtime_error as `LoadSceneFile`, leaving the scene as it was.
    void OpenScene(const std::filesystem::path& path);
    /// @throws std::runtime_error as `SaveSceneFile`.
    void SaveScene(const std::filesystem::path& path);

    /// @brief A click over the viewport asking what is under it.
    struct Pick {
        /// Where, as fractions of the viewport's width and height from its top-left corner.
        glm::vec2 at{0.0f};
        /// Ctrl held: the prim joins or leaves the selection instead of replacing it.
        bool toggle = false;
    };
    /// @return This frame's pick, once; none when nothing was clicked.
    std::optional<Pick> TakePick();
    /// @brief Selects what a pick found: the prim, or nothing, which clears the selection unless
    /// toggling.
    void Picked(const std::optional<Path>& path, bool toggle);

  private:
    /// @brief What the menus and shortcuts ask for this frame.
    struct MenuChoices {
        bool quit = false;
        bool resetLayout = false;
        bool frame = false;
        bool clearSelection = false;
        bool undo = false;
        bool redo = false;
        bool duplicate = false;
        bool remove = false;
        /// A type to create, from the Create menu.
        std::string_view create;
    };
    /// @brief The windows the menus show, until closed.
    struct Windows {
        bool controls = false;
        bool about = false;
        /// Asking for a scene file to open or to save to.
        bool open = false;
        bool save = false;
    };

    void DrawMenuBar();
    void DrawCreateMenu();
    void DrawHelpMenu();
    /// Draws the window asking for the path of a scene file to open or save, while one is asked.
    void DrawFileWindow();
    void TakeShortcuts();
    /// Runs an edit, logging what it throws instead.
    template <typename Edit> void Attempt(const char* what, Edit&& edit);
    void DrawDockSpace();
    void BuildLayout();
    void DrawPanels();
    void SelectLogTab();
    void Navigate();
    /// Draws the toolbar's pills at the viewport's top left: settings, what shows, the camera.
    void DrawToolbar();
    void DrawSettingsMenu();
    void DrawShowMenu();
    void DrawCameraMenu();
    void DrawLightingMenu();
    /// Opens a borderless window of pills at `at`, its corner `pivot` there, in the pills' style.
    bool BeginPills(const char* id, ImVec2 at, ImVec2 pivot);
    void EndPills();
    /// Draws a pill whose menu `items` draws.
    void Pill(const char* id, const std::string& label, const char* tip, void (Shell::*items)());
    /// Draws the timeline strip under the viewport, as Kit lays it out: the range's start, the
    /// ruler, its end, the current time code, then the transport.
    void DrawTimeline();
    /// @return The timeline strip's height; 0 while it is hidden.
    float TimelineHeight() const;
    /// Draws the ruler across the range, which a click or drag scrubs, to whole time codes.
    void DrawRuler(float width);
    /// Draws the transport: first frame, a frame back, play, stop, a frame on, last frame, loop
    /// and the timeline's settings.
    void DrawTransport();
    void DrawTimelineSettings();
    /// Draws the heads-up display at the viewport's top right, over its image.
    void DrawHud();
    void RefreshHud();
    /// Follows the active camera's edits made elsewhere into the orbit.
    void PullActiveCamera();
    /// Writes the orbit into the active camera once navigation moved it.
    void PushActiveCamera();
    void Apply();

    Scene& scene_;
    const DeviceContext* device_;
    Theme theme_;
    DockSizes sizes_;
    MessageLog log_;
    Selection selection_;
    EditHistory history_;
    OrbitCamera camera_;
    MenuChoices choices_;
    Windows windows_;
    /// The path the file window edits.
    std::string filePath_ = "scene.json";
    std::vector<std::unique_ptr<Panel>> panels_;
    /// The default layout's nodes; empty until it is built.
    std::map<Dock, ImGuiID> docks_;
    bool layoutBuilt_ = false;
    /// Whether the bottom dock still has to open on the Log, once its tab bar exists.
    bool selectLog_ = false;
    /// Whether a mouse drag that started over the viewport moves the camera.
    bool dragging_ = false;
    std::optional<Pick> pick_;
    ViewportDisplay display_;
    /// What the viewport showed to begin with, which Reset To Defaults goes back to.
    ViewportDisplay initialDisplay_;
    app::Timeline timeline_;
    std::unique_ptr<Demo> demo_;
    DemoInfo demoInfo_;
    CaptureSettings capture_;
    bool screenshot_ = false;
    std::optional<SequenceRequest> sequence_;
    std::optional<std::pair<std::int64_t, std::int64_t>> recording_;
    bool encoding_ = false;
    std::vector<std::filesystem::path> recentCaptures_;
    /// Whether eco mode paused the viewport this frame, as the viewer reports it.
    bool ecoPaused_ = false;
    Path activeCamera_;
    /// The active camera's world transform as last read or written, to tell its edits apart.
    glm::dmat4 activeWorld_{1.0};
    /// Whether H hid everything drawn over the viewport.
    bool overlaysHidden_ = false;
    /// The heads-up display's lines, refreshed a few times a second so they stay readable.
    std::vector<std::string> hudLines_;
    double hudRefreshed_ = -1.0;
    bool open_ = true;
};

// Presenters

/// @brief An image to draw one frame into, and the layout to leave it in.
struct FrameTarget {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    glm::uvec2 extent{0};
    VkImageLayout finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    /// For the frame's submission: waiting until the image is free to draw into, and signalling
    /// it drawn for presenting; none offscreen.
    BinarySemaphores semaphores;
};

/// @brief Supplies the images frames are drawn into and the UI's platform input, then shows
/// them.
class Presenter {
  public:
    virtual ~Presenter() = default;

    virtual VkFormat Format() const = 0;
    /// @return How many images it cycles through, at least 2.
    virtual std::uint32_t ImageCount() const = 0;
    /// @brief Takes the window system's events.
    virtual void PollEvents() = 0;
    /// @return Whether the user asked to close it.
    virtual bool CloseRequested() const = 0;
    /// @brief Starts the UI frame's platform part: display size, time step and input.
    virtual void BeginUiFrame(double delta) = 0;
    /// @return The next image, or none when there is nothing to draw into, as while minimised.
    virtual std::optional<FrameTarget> Acquire() = 0;
    /// @brief Shows the image last acquired once `drawn` is done, or once its target's semaphore
    /// is signalled.
    virtual void Present(Ticket drawn) = 0;
};

/// @brief Draws every frame into one image of a fixed size, for tests and headless runs; the UI
/// gets no input.
class OffscreenPresenter final : public Presenter {
  public:
    OffscreenPresenter(DeviceContext& context, glm::uvec2 extent);

    VkFormat Format() const override {
        return ToVkFormat(image_.Description().format);
    }
    std::uint32_t ImageCount() const override {
        return 2;
    }
    void PollEvents() override {}
    bool CloseRequested() const override {
        return false;
    }
    void BeginUiFrame(double delta) override;
    std::optional<FrameTarget> Acquire() override {
        // The previous frame's drawing is ordered before this one's on the same queue.
        return FrameTarget{image_.Handle(), image_.View(), glm::uvec2(image_.Description().extent),
                           VK_IMAGE_LAYOUT_GENERAL};
    }
    void Present(Ticket drawn) override {
        drawn_ = drawn;
    }

    /// @return The image frames are drawn into, in the general layout.
    const DeviceImage& Image() const {
        return image_;
    }
    /// @return The ticket of the last frame drawn.
    Ticket Drawn() const {
        return drawn_;
    }

  private:
    DeviceImage image_;
    Ticket drawn_;
};

#if ERNIE_VIEWER_WINDOW
/// @brief A resizable desktop window through GLFW, with no graphics API of its own. GLFW is
/// initialised by the first window and terminated with the last.
class Window {
  public:
    /// @throws std::runtime_error when there is no display or the window cannot be made.
    Window(const std::string& title, glm::uvec2 size);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    /// @return Whether a window could open: GLFW initialises, finding a display, and Vulkan can
    /// present to it.
    static bool Available();
    /// @return The instance extensions presenting to a window needs.
    static std::vector<std::string> InstanceExtensions();

    /// @return The drawable size in pixels; zero while minimised.
    glm::uvec2 FramebufferSize() const;
    /// @return The display's scale over 96 dpi, at least 1.
    float ContentScale() const;
    GLFWwindow* Handle() const {
        return window_;
    }

  private:
    GLFWwindow* window_ = nullptr;
};

/// @brief Presents to a swapchain of the window's surface, made again whenever the window's size
/// changes, and feeds the window's input to ImGui, whose context must outlive it. The host waits
/// for each acquired image and each drawn frame, so presenting needs no semaphores beside the
/// main queue's timeline.
class WindowPresenter final : public Presenter {
  public:
    /// @param window Made before the device context, which needs its instance extensions.
    /// @throws DeviceError when the device cannot present to the window.
    WindowPresenter(DeviceContext& context, std::unique_ptr<Window> window, bool vsync);
    ~WindowPresenter() override;

    VkFormat Format() const override {
        return format_.format;
    }
    std::uint32_t ImageCount() const override {
        return static_cast<std::uint32_t>(images_.size());
    }
    void PollEvents() override;
    bool CloseRequested() const override;
    void BeginUiFrame(double delta) override;
    std::optional<FrameTarget> Acquire() override;
    void Present(Ticket drawn) override;

  private:
    /// @brief Makes the swapchain, again when there is one, at `extent`, with its views and
    /// semaphores.
    void CreateSwapchain(glm::uvec2 extent);
    /// @brief Destroys the images' views and semaphores, once the queue has finished with them.
    void DestroyFrames();

    DeviceContext& context_;
    std::unique_ptr<Window> window_;
    VkSurfaceKHR surface_;
    bool vsync_;
    VkSurfaceFormatKHR format_{};
    VkExtent2D extent_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    /// Signalled by acquiring, one more than the images, used in turn: a frame reuses one only
    /// after the frames in flight before it are done.
    std::vector<VkSemaphore> acquiredSemaphores_;
    std::size_t nextAcquired_ = 0;
    /// Signalled when an image is drawn, one an image, for presenting it to wait on.
    std::vector<VkSemaphore> drawnSemaphores_;
    /// The image acquired and not yet presented.
    std::optional<std::uint32_t> acquired_;
    /// Whether the swapchain no longer matches the surface.
    bool stale_ = false;
};
#endif

}  // namespace ernie::app
