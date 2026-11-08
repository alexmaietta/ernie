// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>

#include "ernie/app/viewer/overlays/hud.h"
#include "ernie/app/viewer/presenter/presenter.h"
#include "ernie/app/viewer/shell/editor.h"
#include "ernie/app/viewer/shell/selection.h"
#include "ernie/app/viewer/theme/theme.h"
#include "ernie/app/viewer/viewer.h"
#include "ernie/app/viewer/viewport/gizmo.h"
#include "ernie/app/viewer/viewport/orbit_camera.h"
#include "ernie/device/tensor.h"
#include "ernie/render/ray_cast/ray_cast.h"
#include "ernie/scene/scene.h"

namespace ernie::app {

class Toolbar;

inline constexpr const char* kHudItemNames[] = {"FPS",         "UI FPS",     "Device Memory",
                                                "Host Memory", "Resolution", "Camera",
                                                "Renderer",    "Selection"};

inline constexpr std::string_view kSessionRoot = "/__session";

enum class AovKind : uint8_t { Color, Hdr, Scalar, Normal, Id };

struct AovInfo {
    std::string_view name;
    AovKind kind;
    const char* group;
    const char* units;
};

inline constexpr AovInfo kAovs[] = {
    {"LdrColor", AovKind::Color, "Color", ""},
    {"HdrColor", AovKind::Hdr, "Color", ""},
    {"DiffuseAlbedo", AovKind::Color, "Color", ""},
    {"DistanceToCamera", AovKind::Scalar, "Geometry", "m"},
    {"DistanceToImagePlane", AovKind::Scalar, "Geometry", "m"},
    {"Normal", AovKind::Normal, "Geometry", ""},
    {"NonStableInstanceSegmentation", AovKind::Id, "Segmentation", ""},
    {"SemanticSegmentation", AovKind::Id, "Segmentation", ""},
    {"SemanticInstanceSegmentation", AovKind::Id, "Segmentation", ""},
    {"StableIdSegmentation", AovKind::Id, "Segmentation", ""},
    {"Intensity", AovKind::Scalar, "Sensor", ""},
    {"RadialVelocity", AovKind::Scalar, "Sensor", "m/s"},
};

const AovInfo* FindAov(std::string_view name);

struct SessionWorld {
    std::string root;
    ComposedWorld composed;
    WorldDisplay display;
    LightingMode lighting = LightingMode::Stage;
    std::optional<Quatd> aim;
    std::vector<std::pair<std::string, uint32_t>> worldPrims;
    std::vector<uint32_t> shown;
    std::vector<Double3> offsets;
    std::vector<Box> boxes;
    std::vector<Double3> origins;
    std::optional<device::Download> boxRead;
    std::optional<device::Download> originRead;
    uint64_t structure = ~uint64_t(0);
    std::vector<std::string> selection;
    double requested = -1;
};

using SessionWorlds = std::map<std::string, std::weak_ptr<SessionWorld>, std::less<>>;

struct AovDisplay {
    ColorizeOptions colors;
    float exposure = 0;

    bool operator==(const AovDisplay&) const = default;
};

void CheckPixels(const device::Tensor& pixels);

struct GridGuide {
    bool shown = false;
    bool fixed = false;
    float spacing = 1;
    uint32_t divisions = 10;
    float reach = 40;
    float minorBrightness = 0.22f;
    float majorBrightness = 0.5f;
    float height = 0;

    bool operator==(const GridGuide&) const = default;
};

struct RenderResolution {
    bool custom = false;
    uint32_t width = 1280;
    uint32_t height = 720;
    bool lockAspect = true;
    float scale = 1;
    bool fill = true;
};

struct ViewportGuides {
    GridGuide grid;
    bool cameras = true;
    bool lights = false;
    bool outline = true;
    bool bounds = false;
    bool allBounds = false;
    bool frames = false;
    bool axes = true;
    Float4 outlineColor = {1.0f, 0.6f, 0.1f, 1.0f};

    bool operator==(const ViewportGuides&) const = default;
};

class Viewport {
  public:
    using Clock = std::chrono::steady_clock;

    Viewport(ViewportDesc desc, Ref<Scene> scene, Ref<device::Device> device, Selection& selection,
             const Theme& theme, std::function<void(const std::string&)> inspect = {},
             Editor* editor = nullptr, SessionWorlds* sessions = nullptr);
    ~Viewport();
    Viewport(const Viewport&) = delete;
    Viewport& operator=(const Viewport&) = delete;

    const std::string& Name() const {
        return m_desc.name;
    }

    const std::string& Product() const {
        return m_desc.product;
    }

    const std::string& RenderedProduct() const {
        return m_product;
    }

    LightingMode Lighting() const {
        return m_session ? m_session->lighting : LightingMode::Stage;
    }

    void SetLighting(LightingMode lighting);

    std::optional<Quatd> ViewOrient() const {
        return m_view ? std::optional(m_view->orient) : std::nullopt;
    }

    const std::vector<std::string>& Views() const {
        return m_views;
    }

    const std::shared_ptr<RenderDelegate>& Delegate() const {
        return m_delegate;
    }

    const std::shared_ptr<Renderer>& GetRenderer() const {
        return m_renderer;
    }

    void ShowSceneOverlays(bool shown);

    Gizmo* GetGizmo() const {
        return m_gizmo.get();
    }

    std::optional<Float2> WorldToScreen(const Double3& point) const;
    std::optional<OverlayRay> ScreenRay(Float2 at) const;

    bool SceneOverlaysShown() const {
        return m_sceneOverlays;
    }

    std::vector<std::string> SceneOverlayNames() const;

    void Advance(const device::Operation& lastDraw);
    void Show(const device::Tensor& image, const device::Operation& ready,
              const device::Tensor& overlay = {});
    void Draw(Presenter& presenter, std::optional<ViewportRect> fixed, bool base);

    const device::Tensor* Shown() const {
        if (m_leader) {
            return m_leader->Shown();
        }
        return m_shown ? &*m_shown : nullptr;
    }

    const device::Tensor* ShownOverlay() const {
        if (m_leader) {
            return m_leader->ShownOverlay();
        }
        return m_shownOverlay.Empty() ? nullptr : &m_shownOverlay;
    }

    uint64_t ShownFrame() const {
        return m_leader ? m_leader->ShownFrame() : m_shownFrame;
    }

    const device::Operation& ShownReady() const {
        return m_leader ? m_leader->ShownReady() : m_shownReady;
    }

    void Follow(const Viewport* leader) {
        m_leader = leader == this ? nullptr : leader;
    }

    bool Interacted() const {
        return m_interacted;
    }

    Extent Size() const {
        return m_size;
    }

    const ViewportRect& Rect() const {
        return m_rect;
    }

    bool Hovered() const {
        return m_hovered;
    }

    double RenderFps() const {
        return m_leader ? m_leader->RenderFps() : m_renderFps;
    }

    void BlockInput(bool blocked) {
        m_blocked = blocked;
    }

    void Frame(bool selected);
    void LookThrough(const std::string& camera);
    void Inspect(const std::string& path);

    const std::string& Aov() const {
        return m_aov;
    }

    void SetAov(const std::string& aov);
    std::vector<const AovInfo*> Aovs() const;
    AovDisplay& Display(std::string_view aov = {});

    const std::optional<std::array<float, 2>>& AovRange() const {
        return m_range;
    }

    void ReadRange(const device::Tensor& bounds);

    bool ShowsWorlds() const {
        return m_session != nullptr;
    }

    const WorldDisplay& Worlds() const {
        static const WorldDisplay kNone;
        return m_session ? m_session->display : kNone;
    }

    void SetWorlds(WorldDisplay display);

    const std::vector<std::pair<std::string, uint32_t>>& WorldPrims() const {
        static const std::vector<std::pair<std::string, uint32_t>> kNone;
        return m_session ? m_session->worldPrims : kNone;
    }

    const OrbitCamera& Orbit() const {
        return m_orbit;
    }

    const std::string& CameraPath() const {
        return m_camera;
    }

    std::optional<float> FocalLength() const {
        return m_view && !m_view->orthographic ? std::optional(m_view->focal) : std::nullopt;
    }

    void AddOverlay(std::function<void(const ViewportRect&)> draw);
    void AddText(uint64_t id, std::string text, const TextStyle& style);
    bool SetText(uint64_t id, std::string text);
    void AddImage(uint64_t id, device::Tensor pixels, device::Operation ready,
                  const Placement& placement, float opacity);
    void AddColorbar(uint64_t id, const ColorbarDesc& desc);
    bool SetColorbar(uint64_t id, const ColorbarDesc& desc);
    bool RemoveOverlay(uint64_t id);
    void AddHudItem(std::string label, std::function<std::string()> value);

    bool visible = true;
    ViewportGuides guides;
    RenderResolution resolution;
    std::function<bool(const OverlayEvent&)> overlayEvents;
    bool hudShown = true;
    bool toolbarShown = true;
    double targetRenderFps = 0;
    std::array<bool, std::size(kHudItemNames)> hudItems{};
    std::set<std::string, std::less<>> hiddenOverlays;

  private:
    struct CameraView {
        uint32_t world = 0;
        Double3 translate;
        Quatd orient;
        WorldTransform frame{};
        float focal = 0;
        float width = 0;
        float height = 0;
        float right = 0;
        float up = 0;
        Float2 clipping;
        bool orthographic = false;
        std::optional<Box> bounds;
    };

    struct Pending {
        device::Tensor pixels;
        device::Operation ready;
        device::Tensor overlay;
    };

    struct Retired {
        device::Tensor pixels;
        device::Operation drawn;
        device::Tensor overlay;
    };

    struct TextItem {
        uint64_t id = 0;
        std::string text;
        TextStyle style;
    };

    struct ImageItem {
        uint64_t id = 0;
        device::Tensor pixels;
        device::Operation ready;
        Placement placement;
        float opacity = 1;
    };

    struct ColorbarItem {
        uint64_t id = 0;
        ColorbarDesc desc;
    };

    struct HudEntry {
        std::string label;
        std::function<std::string()> value;
    };

    struct GuideKey {
        uint64_t structure = 0;
        std::vector<std::string> selection;
        ViewportGuides guides;
        double gridStep = 0;
        double gridFade = 0;
        Double3 gridCentre;

        bool operator==(const GuideKey&) const = default;
    };

    void DefineSession();
    void RefreshWorlds();
    void ShowWorlds();
    Pending Receive(const device::Tensor& pixels, const device::Operation& ready,
                    const device::Tensor& overlay);
    device::Tensor Staging(Span<const int64_t> shape, uint64_t taken = 0);
    bool OwnsCamera() const;
    bool Fit(Extent size);
    void Refresh();
    bool StartOrbit();
    void MoveCamera();
    void StartPick(double u, double v, bool toggle);
    void LandPick();
    void LandFraming();
    void SyncOverlays();
    void UpdateGuides();
    std::optional<GizmoCamera> GizmoView(ImVec2 min, ImVec2 size) const;
    void TrackGestures(ImVec2 imageMin, ImVec2 imageSize, bool pressed);
    bool SendGesture(OverlayGesture gesture, const OverlayHit& hit, ImVec2 mouse, ImVec2 imageMin,
                     ImVec2 imageSize);
    void DrawItems(Presenter& presenter, ImDrawList& list, ImVec2 min, ImVec2 max);
    void DrawColorbar(ImDrawList& list, ImVec2 min, ImVec2 max, const ColorbarDesc& desc);
    void DrawAxes(ImDrawList& list, ImVec2 min, ImVec2 max) const;
    std::vector<std::string> HudLines() const;

    ViewportDesc m_desc;
    Ref<Scene> m_scene;
    Ref<device::Device> m_device;
    Selection& m_selection;
    const Theme& m_theme;
    std::shared_ptr<RenderDelegate> m_delegate;
    std::shared_ptr<Renderer> m_renderer;
    bool m_sceneOverlays = true;
    std::vector<std::string> m_shownOverlays;
    std::optional<GuideKey> m_guideKey;
    std::function<void(const std::string&)> m_inspect;
    SessionWorlds* m_sessions = nullptr;
    std::shared_ptr<SessionWorld> m_session;
    std::string m_product;
    std::string m_shownVar;
    std::string m_aov;
    std::map<std::string, AovDisplay, std::less<>> m_displays;
    bool m_framingWorlds = false;
    bool m_framingPrims = false;
    double m_framingMargin = 1.1;
    std::optional<device::BuildProgress> m_builds;
    std::optional<device::Download> m_bounds;
    std::optional<std::array<float, 2>> m_range;
    std::vector<std::string> m_views;
    std::set<std::string> m_unframed;
    Double3 m_orthoEye;
    float m_orthoWidth = 0;
    float m_orthoHeight = 0;

    std::optional<RenderFrame> m_rendering;
    std::optional<Pending> m_pending;
    std::optional<device::Tensor> m_shown;
    device::Tensor m_shownOverlay;
    device::Operation m_shownReady;
    uint64_t m_shownFrame = 0;
    std::vector<Retired> m_retired;
    std::vector<device::Tensor> m_staging;
    Clock::time_point m_stepStarted;
    Clock::time_point m_lastRendered;
    double m_renderFps = 0;
    double m_stepMs = 0;
    Extent m_renderSize;

    std::unique_ptr<RayCaster> m_caster;
    uint64_t m_structure = ~uint64_t(0);
    std::string m_camera;
    uint32_t m_cameraWorld = 0;
    std::vector<device::Download> m_reads;
    std::optional<CameraView> m_view;
    RayHits m_hits;
    std::optional<device::Download> m_pick;
    std::vector<device::Download> m_framing;
    bool m_pickToggles = false;
    bool m_orbiting = false;
    OrbitCamera m_orbit;
    Extent m_size;
    Extent m_resolution;
    bool m_fitted = false;
    bool m_blocked = false;
    bool m_hovered = false;
    bool m_interacted = false;
    const Viewport* m_leader = nullptr;
    ViewportRect m_rect;

    Hud m_hud;
    std::unique_ptr<Toolbar> m_toolbar;
    std::unique_ptr<Gizmo> m_gizmo;
    bool m_inputClaimed = false;
    std::optional<OverlayPick> m_hoverPick;
    std::optional<OverlayHit> m_hoverHit;
    std::optional<OverlayHit> m_pressedHit;
    ImVec2 m_pressedMouse;
    ImVec2 m_imageMin;
    ImVec2 m_imageSize;
    ImVec2 m_draggedMouse;
    std::vector<HudEntry> m_hudEntries;
    std::vector<std::function<void(const ViewportRect&)>> m_overlays;
    std::vector<TextItem> m_texts;
    std::vector<ImageItem> m_images;
    std::vector<ColorbarItem> m_colorbars;
};

}
