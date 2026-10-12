// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/app/file_dialog.h"
#include "ernie/app/panel.h"
#include "ernie/app/theme.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/scene/edit_history.h"
#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/session/delegate.h"
#include "ernie/session/frame_params.h"

#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::app {

/// How the viewport lights the scene. The viewer opens with `Stage`
/// when the scene has lights and `Default` otherwise.
enum class Lighting {
    Stage,    ///< The scene's own lights, as a headless render draws
              ///< them; black without any.
    Camera,   ///< A distant light along the view direction, ignoring
              ///< the scene's lights.
    Default,  ///< A white sun from above with shadows, ignoring the
              ///< scene's lights.
};

/// The default layout's docks as fractions of the window: what each
/// takes of the space left once the docks before it are placed (left,
/// then the right column, then the bottom).
struct DockSizes {
    float left = 0.24f;   ///< `Dock::Left`'s share of the width.
    float right = 0.28f;  ///< The right column's share of what the
                          ///< left dock leaves.
    float rightLower = 0.62f;  ///< `Dock::RightLower`'s share of the
                               ///< right column's height.
    float bottom = 0.25f;  ///< `Dock::Bottom`'s share of the height
                           ///< beside the side docks.
};

/// Options for opening a `Viewer`.
struct ViewerOptions {
    std::string title = "Ernie";  ///< Window title.
    std::uint32_t width =
        1600;  ///< Window width in screen coordinates.
    std::uint32_t height =
        900;  ///< Window height in screen coordinates.
    bool hidden =
        false;  ///< Render without showing the window, for tests.
    bool validation = false;  ///< Enable the Vulkan validation layer
                              ///< when installed.
    /// Wait for the display's refresh; otherwise draw frames as fast
    /// as they render.
    bool vsync = false;
    std::string device;  ///< GPU name to prefer; empty picks the
                         ///< first discrete.
    std::string renderer =
        "raster";  ///< Render backend, one of `Viewer::Renderers()`.
    Theme theme;   ///< Colours and metrics of the UI.
    std::filesystem::path
        layout;  ///< Docking layout (ImGui .ini) to load; empty uses
                 ///< the built-in default layout.
    DockSizes docks;  ///< The default layout's dock sizes, clamped to
                      ///< (0, 1).
    Path settings;  ///< RenderSettings prim the viewport starts with;
                    ///< empty picks the first one, as
                    ///< `ResolveRenderPrims`.
    /// Records the profiler's zones every frame (demos' `--profile`),
    /// not only while the Profiler panel shows.
    bool profile = false;
};

/// What the viewport draws over the scene, as Kit's viewport "show by
/// type" menu (the eye) and heads-up display settings choose. None of
/// it renders into the scene's own render products.
struct ViewportDisplay {
    bool lights = true;   ///< Every light's gizmo: its shape and the
                          ///< direction it shines.
    bool cameras = true;  ///< Every scene camera's frustum.
    bool grid = false;    ///< A grid on the ground (z = 0) with the x
                          ///< and y axes.
    bool selection = true;  ///< The selection's outline.
    bool overlays =
        true;  ///< The scene's overlays (`Scene::GetOverlay`).
    /// Scene overlays the viewport leaves out while `overlays` shows
    /// the others, by name: the eye menu's Scene Overlays submenu
    /// toggles each.
    std::set<std::string, std::less<>> hiddenOverlays;
    bool hud = true;  ///< The heads-up display at the top left, with
                      ///< the items below.
    bool hudFps = true;  ///< Frames per second and frame time.
    bool hudDeviceMemory = true;  ///< GPU memory in use, the
                                  ///< renderer's share and the total.
    bool hudHostMemory = false;   ///< The process's resident memory.
    bool hudResolution = true;    ///< The viewport image's size.
    bool hudExposure = false;     ///< The metered exposure value and
                                  ///< whether it is automatic.
    bool hudTriangles = false;    ///< Triangles the last frame drew.
    /// The viewport toolbar's menus, which applications may hide; the
    /// settings and eye menus always show, and the eye menu's Toolbar
    /// submenu shows these again.
    bool toolbarRenderer =
        true;  ///< The renderer, rendering mode and render settings.
    bool toolbarVar = true;  ///< The render var the viewport shows.
    bool toolbarCamera =
        true;  ///< The camera the viewport looks through.
    bool toolbarLighting =
        true;  ///< How the viewport lights the scene.
    /// The viewport's own UI: its toolbar and menus, heads-up
    /// display, tool strip, axis indicator and transform gizmo. Off
    /// (the H key toggles it), only the image and the scene's
    /// overlays show.
    bool ui = true;

    bool operator==(const ViewportDisplay&) const = default;
};

/// How serious a log message is.
enum class LogLevel { Info, Warning, Error };

/// One message of the viewer's log (`Viewer::Log`).
struct LogEntry {
    double time = 0.0;  ///< Seconds since the viewer opened.
    LogLevel level = LogLevel::Info;
    std::string text;
};

/// Per-frame information passed to `OnFrame`, `OnPreRender` and
/// `OnPostRender` callbacks.
struct FrameInfo {
    std::uint64_t frame = 0;  ///< Frames drawn before this one.
    double time = 0.0;        ///< Seconds since the viewer opened.
    double delta = 0.0;       ///< Seconds since the previous frame.
};

/// An interactive window onto an Ernie `Scene`: a docked viewport
/// rendering the scene with a `RenderDelegate` (the rasterizer), plus
/// Stage, Property, Render Settings, Sensor, Stats and Log panels.
///
/// The viewer keeps no scene state of its own: panels read the scene
/// each frame and write edits back through `Scene::Set`, so edits
/// from code and from the GUI are one history, seen by both. What
/// only the GUI needs (its cameras, the viewport's render product,
/// lighting presets) is defined under `SessionRoot()`, which panels
/// hide unless asked to show it.
///
/// The viewer defines its own perspective camera on opening, framing
/// the scene (`PerspectiveCamera()`, the default view). The viewport
/// looks through it or any Camera prim of the scene; navigating moves
/// whichever camera it looks through.
///
/// Drive it in one of two ways, both on the thread that created it
/// (GLFW needs the main thread on macOS):
/// - `Run()` loops until the window closes;
/// - `Step()` draws one frame and returns, for callers that own the
///   loop, such as a simulation.
///
/// Each frame runs, in order: `OnFrame` callbacks; the UI, whose
/// panels may edit the scene, then `OnUi` callbacks; `OnPreRender`
/// callbacks; the scene's sync and render, and presentation;
/// `OnPostRender` callbacks. Callbacks run on the viewer's thread; an
/// exception from one ends that frame and propagates out of `Step` or
/// `Run`, and the viewer can step again afterwards.
///
/// Applications extend it with panels (`AddPanel`), menu items
/// (`AddMenuItem`) and callbacks; TerrainGen's viewer is one such
/// application, adding its terrain configuration panels.
class Viewer {
  public:
    /// Opens the window and initializes Vulkan, ImGui and the render
    /// delegate. `scene` must outlive the viewer.
    /// @throws std::runtime_error if the window or GPU cannot be
    /// initialized.
    explicit Viewer(Scene& scene, const ViewerOptions& options = {});
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    /// Draws frames until the window closes or `Close` is called.
    void Run();

    /// Processes window events, syncs the scene, renders the viewport
    /// and the UI and presents one frame. Returns whether the window
    /// is still open.
    bool Step();

    /// Whether the window is open.
    bool IsOpen() const;

    /// Asks the window to close; `Run` returns and `Step` returns
    /// false after the current frame.
    void Close();

    /// Adds a callback run at the start of every frame, before the UI
    /// and the render, so its edits show in that frame. Returns an id
    /// for `RemoveCallback`.
    std::uint64_t
    OnFrame(std::function<void(Viewer&, const FrameInfo&)> callback);

    /// Adds a callback run during the UI, after the panels, inside
    /// the Dear ImGui frame: for keyboard shortcuts
    /// (`ImGui::IsKeyPressed`) and overlays. Returns an id for
    /// `RemoveCallback`.
    std::uint64_t
    OnUi(std::function<void(Viewer&, const FrameInfo&)> callback);

    /// Adds a callback run after the UI and just before the scene is
    /// synced and rendered: the scene then holds this frame's GUI
    /// edits, and the callback's own edits show in this frame.
    /// Returns an id for `RemoveCallback`.
    std::uint64_t OnPreRender(
        std::function<void(Viewer&, const FrameInfo&)> callback);

    /// Adds a callback run after the frame is submitted and
    /// presented; the GPU may still be rendering it. Edits show in
    /// the next frame. Returns an id for `RemoveCallback`.
    std::uint64_t OnPostRender(
        std::function<void(Viewer&, const FrameInfo&)> callback);

    /// Adds a callback run when the selection changes, by click or by
    /// `SetSelection`.
    std::uint64_t OnSelection(
        std::function<void(Viewer&, std::span<const Path>)> callback);

    /// Removes the frame, UI, pre-render, post-render or selection
    /// callback with id `id`; unknown ids are ignored.
    void RemoveCallback(std::uint64_t id);

    /// Registration scopes, for applications that host others' code,
    /// as the demo runner hosts demos: what is added (callbacks,
    /// panels, menu items, viewport toggles, heads-up display and
    /// viewport menu items) belongs to the scope current when it is
    /// added, and `RemoveScope` removes all of it at once. Scope 0,
    /// current until `SetScope`, is the application's own. `NewScope`
    /// returns a scope not used before, without making it current.
    std::uint64_t NewScope();
    std::uint64_t Scope() const;
    void SetScope(std::uint64_t scope);

    /// Removes everything added in `scope`; built-in panels are only
    /// hidden, as `RemovePanel`.
    /// @throws std::invalid_argument for scope 0.
    void RemoveScope(std::uint64_t scope);

    /// Logging, after Newton's viewers' `log_scalar`, `log_array` and
    /// `log_image`: values a simulation or sensor produces, shown
    /// live as they change. Each belongs to the registration scope
    /// current when first logged (`SetScope`), so a demo's logs go
    /// with it.
    ///
    /// Logs `value` as the next sample of the scalar signal `name`,
    /// drawn as a line of its last thousand samples in the Plots
    /// panel. `clear` drops the samples before; `smoothing` averages
    /// that many values into each sample.
    void LogScalar(std::string_view name, double value,
                   bool clear = false, int smoothing = 1);

    /// The samples of scalar signal `name` so far, oldest first;
    /// empty if none.
    std::vector<double> LoggedScalars(std::string_view name) const;

    /// Logs `array`, (n) or (rows, cols) numbers on the host or the
    /// GPU, as a heatmap in the Plots panel, replacing the last; a
    /// GPU array is downloaded after the work queued on compute
    /// stream `stream`.
    /// @throws std::invalid_argument for another shape or type.
    void LogArray(std::string_view name, const Tensor& array,
                  std::uintptr_t stream = 1);

    /// Logs `image` in a panel of its own titled `name`, replacing
    /// the last: (rows, cols), (rows, cols, channels) or a batch
    /// (views, rows, cols, channels), with 1, 3 or 4 channels, uint8
    /// (0 to 255, shown as they are, so sRGB) or float32 (0 to 1). A
    /// host image is uploaded with the next frame. An image on the
    /// viewer's GPU (`Delegate().Compute()`) is copied device to
    /// device into the texture after the work queued on compute
    /// stream `stream`, without the host; to skip even that copy,
    /// write the texture itself (`MapImage`).
    /// @throws std::invalid_argument for another shape or type, a
    /// device other than the host or the viewer's GPU, or a name
    /// another panel has.
    void LogImage(std::string_view name, const Tensor& image,
                  std::uintptr_t stream = 1);

    /// The texture of the image panel `name` itself, as a tensor of
    /// `shape` and `type` (as `LogImage` takes them, without 3
    /// channels) on the viewer's GPU, its rows the texture's pitch
    /// apart: write it on compute stream `stream` and `UnmapImage`,
    /// and the panel shows it with no copy at all. Map again for each
    /// new image; the writes on `stream` follow the frames that
    /// sampled the last one.
    /// @throws std::runtime_error without GPU interop, or if the GPU
    /// cannot sample the linear images that need
    /// (`VulkanContext::SupportsLinearSampling`);
    /// std::invalid_argument as `LogImage`.
    Tensor MapImage(std::string_view name,
                    std::vector<std::int64_t> shape, DType type,
                    std::uintptr_t stream = 1);

    /// Ends writing image `name` (`MapImage`): frames sample it after
    /// the work queued on `stream`.
    void UnmapImage(std::string_view name, std::uintptr_t stream = 1);

    /// Forgets the scalar signal, array or image `name`.
    void RemoveLogged(std::string_view name);

    Scene& GetScene();
    VulkanContext& Context();
    const VulkanContext& Context() const;

    /// The render backends the viewer can switch between, such as
    /// "raster".
    static std::vector<std::string> Renderers();

    /// The render backend the viewport draws with.
    const std::string& Renderer() const;

    /// Switches the viewport to the render backend `name`, waiting
    /// for the GPU; the new backend syncs the whole scene on its next
    /// frame. References from `Delegate()` become invalid.
    /// @throws std::invalid_argument for a backend not in
    /// `Renderers()`, or std::runtime_error if it cannot start, in
    /// which case the current backend stays.
    void SetRenderer(const std::string& name);

    /// The reflectance models of the current backend, for material
    /// editors.
    const BrdfRegistry& Brdfs() const;

    /// What the current backend holds and drew last frame: GPU
    /// memory, triangles and its own counters.
    RenderStats Stats() const;

    /// Syncs the scene into the current backend now, rather than at
    /// the next frame, to check newly authored data.
    /// @throws std::invalid_argument naming the prim and attribute of
    /// data the backend cannot use.
    void Sync();

    /// Starts over on the scene's present content, once an
    /// application has replaced it (as `OpenScene` and the demo
    /// runner do): clears the selection and undo history, and picks
    /// the RenderSettings, the camera (`PerspectiveCamera()`, framing
    /// the scene) and the lighting as on opening, except those the
    /// application chose (`SetSettings`, `SetActiveCamera`,
    /// `SetLighting`) since the scene was cleared (`ClearScene`) or
    /// the viewer opened.
    /// @throws std::invalid_argument as `Sync`.
    void ResetSession();

    /// Empties the stage for new content: removes every prim but the
    /// viewer's own and every overlay but the viewer's (named with a
    /// leading "__"), clears the selection and undo history, and
    /// forgets the settings, camera and lighting the application
    /// chose. Call `ResetSession` once the new content is in.
    void ClearScene();

    /// Replaces the stage with the scene the JSON file at `path`
    /// describes (`LoadSceneFile`, ernie/scene/scene_json.h), as File
    /// > Open Scene does: empties the stage (`ClearScene`), loads the
    /// file and starts the session over (`ResetSession`). The file is
    /// checked first, by loading it into a scene of its own, so a
    /// file that does not load leaves the stage as it was.
    /// @throws std::invalid_argument for a file that cannot be read
    /// or does not load, or that describes prims under
    /// `SessionRoot()`; or as `ResetSession`.
    void OpenScene(const std::filesystem::path& path);

    /// Writes the stage, every prim but the viewer's own, to `path`
    /// as JSON (`SaveSceneFile`), as File > Save Scene As does.
    /// @throws std::invalid_argument as `SaveSceneFile`.
    void SaveScene(const std::filesystem::path& path);

    /// The current backend itself, for what the viewer does not wrap,
    /// such as CUDA bindings.
    RenderDelegate& Delegate();

    /// Root of the prims the viewer defines for its own use,
    /// `/__session`.
    static const Path& SessionRoot();

    /// The viewer's perspective camera, `/__session/Perspective`,
    /// framing the scene's bounds when the viewer opens.
    static const Path& PerspectiveCamera();

    /// The viewer's own RenderSettings, `/__session/Settings`: what
    /// the viewport renders with while the scene has none, edited in
    /// the Settings panel as a scene's are, and always on hand to
    /// choose besides the scene's.
    static const Path& SessionSettings();

    const std::vector<Path>& Selection() const;
    void SetSelection(std::vector<Path> paths);

    /// Sets `attribute` of the prim at `path` as a GUI edit, through
    /// the viewer's undo history (`History`). With `open`, a
    /// continuing edit (a drag) merges into one undo step, which ends
    /// once no widget or gizmo is in use.
    /// @throws std::invalid_argument as `Scene::Set`.
    void Edit(const Path& path, std::string_view attribute,
              Value value, bool open = false);

    /// Gives the prim at `path` the local transform `local` as a GUI
    /// edit, as `Edit` does, through its own transform ops where they
    /// can represent it (`LocalTransformEdits`).
    void EditLocalTransform(const Path& path, const glm::dmat4& local,
                            bool open = false);

    /// Clears `attribute` of the prim at `path`, back to its
    /// fallback, as an undoable GUI edit.
    void ClearAttribute(const Path& path, std::string_view attribute);

    /// Undoes or redoes the last GUI edit (Ctrl+Z, Ctrl+Shift+Z or
    /// Ctrl+Y); returns whether there was one.
    bool Undo();
    bool Redo();

    /// The GUI edits, for undo and redo.
    EditHistory& History();

    /// Creates a prim of `type`, any of the Create menu's
    /// (`CreatableTypes`), with defaults that make it show
    /// (`AuthorDefaultPrim`): under the selected Xform or Scope if
    /// one is, else its type's usual parent, at the view's target and
    /// scale. A material is bound to the selected geometry. One undo
    /// step; selects the new prim and returns its path.
    /// @throws std::invalid_argument for a type the menu does not
    /// offer.
    Path Create(std::string_view type);

    /// Deletes the prims at `paths` and their descendants as one undo
    /// step and clears the selection; the viewer's own prims are left
    /// alone.
    void Delete(std::span<const Path> paths);

    /// Copies the prims at `paths` with their descendants and
    /// attributes (sharing tensors) beside themselves, named
    /// `<name>_01`, `<name>_02`... with the first free number, as one
    /// undo step (Ctrl+D); selects and returns the copies. The
    /// viewer's own prims are left alone.
    std::vector<Path> Duplicate(std::span<const Path> paths);

    /// Renames the prim at `path` to `name`, moving its descendants
    /// with it, as one undo step (F2 in the Stage), keeping the
    /// selection on it; returns its new path. Relationships to it
    /// (material bindings, cameras) still name the old path.
    /// @throws std::invalid_argument for an invalid name, a name
    /// taken by a sibling, the root or a viewer prim.
    Path Rename(const Path& path, std::string_view name);

    /// What the transform gizmo on a single selected prim does: none,
    /// or translate, rotate or scale (the W, E and R keys), in world
    /// or local axes.
    enum class Gizmo { None, Translate, Rotate, Scale };
    Gizmo GetGizmo() const;
    void SetGizmo(Gizmo gizmo);
    bool GizmoLocal() const;
    void SetGizmoLocal(bool local);

    /// The prim the viewport shows at pixel (`x`, `y`) of its image
    /// (`RenderDelegate::Pick`), as a click selects; the empty path
    /// for the background or a session prim.
    Path Pick(std::uint32_t x, std::uint32_t y);

    /// Adds a message to the viewer's log, shown in the Log panel;
    /// the viewer itself logs render errors as they start. The log
    /// keeps the latest few thousand messages.
    void Log(LogLevel level, std::string text);
    const std::deque<LogEntry>& LogEntries() const;
    void ClearLog();

    /// The cameras the viewport can look through: the viewer's own,
    /// then the scene's Camera prims by path.
    std::vector<Path> Cameras() const;

    /// The Camera prim the viewport looks through;
    /// `PerspectiveCamera()` until changed. If the scene's camera is
    /// removed, the viewport falls back to `PerspectiveCamera()`.
    const Path& ActiveCamera() const;

    /// Looks through the Camera prim at `camera`, the viewer's or the
    /// scene's.
    /// @throws std::invalid_argument if `camera` is not a Camera
    /// prim.
    void SetActiveCamera(Path camera);

    /// Frames `paths` (the whole scene when empty) with the active
    /// camera (the F key).
    void Frame(std::span<const Path> paths = {});

    /// Defines a Camera prim at `path` in the scene with the active
    /// camera's transform and lens, and returns it.
    /// @throws std::invalid_argument if `path` exists or is under
    /// `SessionRoot()`.
    Path CameraFromView(const Path& path);

    /// The render var the viewport shows, `LdrColor` until changed:
    /// any of the backend's `RendererInfo::viewVars`, such as
    /// `DistanceToCamera`, `Normal` or `PrimId`, each drawn as a
    /// display image (see `RasterRenderer::CanPresent`). Switching to
    /// a backend that cannot show it goes back to `LdrColor`.
    const std::string& ViewVar() const;

    /// @throws std::invalid_argument for a name not in the backend's
    /// `RendererInfo::viewVars`.
    void SetViewVar(std::string name);

    /// The spectral band the viewport shows while it shows
    /// SpectralRadiance, or -1 (the default) for the mean over the
    /// camera's bands (`RenderDelegate::SetViewBand`).
    int ViewBand() const;
    void SetViewBand(int band);

    Lighting GetLighting() const;
    void SetLighting(Lighting lighting);

    /// Whether the sensor clock (`RenderDelegate::AdvanceTime`)
    /// stands still between frames; by default it runs with them, a
    /// frame's seconds a frame, so pulses and pattern sequences play
    /// out.
    bool ClockPaused() const;
    void SetClockPaused(bool paused);

    /// Advances the sensor clock by `seconds` at the next frame,
    /// paused or not, as a single step does.
    void StepClock(double seconds);

    /// How the viewport exposes its image: `Auto` keys the scene's
    /// metered average luminance to middle grey (adapting over a
    /// fraction of a second), `Camera` (the default) takes the
    /// camera's own exposure attributes as they are, and `Default` is
    /// automatic through the viewer's own camera and the camera's own
    /// through a scene camera.
    enum class Exposure { Default, Auto, Camera };
    Exposure GetExposure() const;
    void SetExposure(Exposure exposure);

    /// Stops of exposure compensation on top of either mode
    /// (`tokens::kExposureCompensation`).
    double ExposureCompensation() const;
    void SetExposureCompensation(double stops);

    /// The EV100 of the scene's average luminance as last metered (EV
    /// = log2(L x 100 / 12.5)), if the backend meters.
    std::optional<double> MeteredEv() const;

    /// Whether the viewport currently exposes automatically
    /// (`Exposure`).
    bool AutoExposing() const;

    /// Whether frames wait for the display's refresh
    /// (`ViewerOptions::vsync`); without it the viewer draws as fast
    /// as it renders, without tearing where the display allows.
    /// Always false for a hidden viewer.
    bool GetVsync() const;
    void SetVsync(bool vsync);

    /// The viewport's render resolution, as Kit's Viewport menu sets
    /// it: `size` (width, height) or none for the viewport panel's
    /// own, times `scale`; `fill` fits the image to the panel,
    /// letterboxed, else it shows a texel per pixel, centred. The
    /// camera's aspect follows the resolution.
    void SetRenderResolution(
        std::optional<std::pair<std::uint32_t, std::uint32_t>> size,
        float scale = 1.0f, bool fill = true);

    /// How the viewport draws triangle edges: a mode overriding the
    /// render settings' own (`ernie:wireframe`) in the viewport only,
    /// or none (the default) to follow them. The Z key cycles through
    /// off, overlay and hidden line.
    std::optional<Wireframe> GetWireframe() const;
    void SetWireframe(std::optional<Wireframe> mode);

    /// Whether the viewport shades minimally (`ernie:minimal:mode`,
    /// Kit's "RTX - Minimal"): true for the surfaces' diffuse
    /// colours, false for full shading, overriding the render
    /// settings in the viewport only; or none (the default) to follow
    /// them.
    std::optional<bool> GetMinimal() const;
    void SetMinimal(std::optional<bool> minimal);
    /// Whether the viewport shades minimally now, by `GetMinimal` or
    /// else the render settings.
    bool MinimalShown() const;

    /// What the viewport draws over the scene and its heads-up
    /// display lists.
    const ViewportDisplay& GetViewportDisplay() const;
    void SetViewportDisplay(const ViewportDisplay& display);

    /// Adds a toggle to the viewport's eye menu, after the built-in
    /// ones, for what an application draws over the scene, such as
    /// physics contacts or labels. With `overlay`, the toggle shows
    /// or hides that scene overlay in the viewport (whatever the
    /// Scene Overlays toggle says); either way `changed`, if given,
    /// is told when the user switches it. Returns an id for
    /// `ViewportToggleShown`, `SetViewportToggleShown` and
    /// `RemoveViewportItem`.
    /// @throws std::invalid_argument for an empty label.
    std::uint64_t AddViewportToggle(
        std::string label, bool shown = true,
        std::string overlay = {},
        std::function<void(Viewer&, bool)> changed = {});
    /// @throws std::invalid_argument for an id that is no toggle.
    bool ViewportToggleShown(std::uint64_t id) const;
    void SetViewportToggleShown(std::uint64_t id, bool shown);

    /// Adds a line to the viewport's heads-up display, "label: value"
    /// with the text `value` returns (asked a few times a second),
    /// such as a physics step's time; the eye menu's Heads Up Display
    /// submenu toggles it. Returns an id for `HudItemShown`,
    /// `SetHudItemShown` and `RemoveViewportItem`.
    /// @throws std::invalid_argument for an empty label or no
    /// `value`.
    std::uint64_t
    AddHudItem(std::string label,
               std::function<std::string(Viewer&)> value,
               bool shown = true);
    /// @throws std::invalid_argument for an id that is no heads-up
    /// display item.
    bool HudItemShown(std::uint64_t id) const;
    void SetHudItemShown(std::uint64_t id, bool shown);

    /// Adds `label` to the viewport's settings menu (its leftmost
    /// pill), running `action` when chosen; with `checked`, the item
    /// shows a check mark while it returns true. Returns an id for
    /// `RemoveViewportItem`.
    /// @throws std::invalid_argument for an empty label or no
    /// `action`.
    std::uint64_t AddViewportMenuItem(
        std::string label, std::function<void(Viewer&)> action,
        std::function<bool(const Viewer&)> checked = {});

    /// Calls `clicked` with the overlay's name and the item's id when
    /// a click in the viewport lands on a screen overlay item with an
    /// id (`OverlayStyle::id`, such as a button's,
    /// `Overlay::Button`), in place of picking the scene there. While
    /// the pointer is over such an item, its overlay marks it hovered
    /// (`Overlay::SetHovered`). Returns an id for
    /// `RemoveViewportItem`.
    /// @throws std::invalid_argument for no `clicked`.
    std::uint64_t AddOverlayClickHandler(
        std::function<void(Viewer&, const std::string&,
                           std::uint32_t)>
            clicked);

    /// Removes the viewport toggle, heads-up display item, viewport
    /// menu item or overlay click handler with id `id`; unknown ids
    /// are ignored.
    void RemoveViewportItem(std::uint64_t id);

    /// The RenderSettings prim the viewport renders with: the scene's
    /// first unless chosen, else `SessionSettings()`.
    const Path& Settings() const;

    /// @throws std::invalid_argument if `settings` is not one of the
    /// scene's RenderSettings prims or `SessionSettings()`.
    void SetSettings(Path settings);

    /// Writes the next viewport frame to `path` as PNG, showing `var`
    /// (the shown var when empty) as the viewport draws it, and on
    /// screen for that frame. For a var's own values, render it
    /// headless (`RenderDelegate::StepAsync`).
    /// @throws std::invalid_argument as `SetViewVar`.
    void Capture(const std::filesystem::path& path,
                 std::string var = {});

    /// Writes the whole window as last drawn, UI and all, to `path`
    /// as PNG: for screenshot tests of the interface. Only a hidden
    /// viewer (`ViewerOptions::hidden`) can.
    /// @throws std::logic_error for a viewer with a window, or
    /// std::runtime_error if the file cannot be written.
    void CaptureWindow(const std::filesystem::path& path);

    /// Adds a dockable panel, shown under Window in the menu bar.
    /// Returns the panel.
    /// @throws std::invalid_argument for a null panel or a title in
    /// use or reserved ("Viewport").
    Panel& AddPanel(std::unique_ptr<Panel> panel);

    /// Constructs a `T` (a `Panel` subclass) from `args` and adds it;
    /// returns it.
    template <typename T, typename... Args>
    T& AddPanel(Args&&... args) {
        return static_cast<T&>(AddPanel(
            std::make_unique<T>(std::forward<Args>(args)...)));
    }

    /// Adds a panel titled `title` that calls `draw` to draw its
    /// contents with Dear ImGui; for panels that need no class of
    /// their own. Returns the panel.
    Panel& AddPanel(std::string title, Dock dock,
                    std::function<void(Viewer&)> draw);

    /// The panel titled `title`, built-in ("Stats") or added; null if
    /// there is none.
    Panel* FindPanel(std::string_view title);

    /// Shows the panel titled `title` and brings it to the front:
    /// where it shares a dock with other panels its tab becomes the
    /// selected one, and it takes keyboard focus. Takes effect on the
    /// next frame in which the panel has already been drawn once, so
    /// a panel focused before the first frame (such as the Log over
    /// Stats at the bottom) is in front once its docked place exists.
    /// The last panel focused in a dock wins.
    /// @throws std::invalid_argument if there is no panel titled
    /// `title`.
    void FocusPanel(std::string_view title);

    /// Adds a `WidgetPanel` titled `title`, for widgets added without
    /// ImGui (Python's panels).
    WidgetPanel& AddWidgetPanel(std::string title,
                                Dock dock = Dock::Right);

    /// Removes the panel titled `title`; built-in panels are only
    /// hidden.
    void RemovePanel(const std::string& title);

    /// Opens the viewer's file dialog (`FileDialog`), modal over the
    /// window, to choose a file to open or to save as `options` says.
    /// `done` runs on the viewer's thread with the chosen path once
    /// the user confirms, and never if they cancel; a dialog already
    /// open is replaced.
    void OpenFileDialog(
        FileDialogOptions options,
        std::function<void(Viewer&, const std::filesystem::path&)>
            done);

    /// The file dialog, open or not: for choosing through it from
    /// code, as tests do.
    FileDialog& GetFileDialog();

    /// Adds `label` to the menu bar's `menu` (created if needed),
    /// running `action` when chosen.
    void AddMenuItem(const std::string& menu,
                     const std::string& label,
                     std::function<void(Viewer&)> action);

    void SaveLayout(const std::filesystem::path& path) const;

    /// @throws std::runtime_error if the file cannot be read.
    void LoadLayout(const std::filesystem::path& path);

    const Theme& GetTheme() const;
    void SetTheme(const Theme& theme);

  private:
    /// Adds the Plots panel the first time a scalar or array is
    /// logged.
    void EnsurePlotsPanel();

    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace ernie::app
