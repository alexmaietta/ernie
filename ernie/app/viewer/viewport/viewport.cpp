// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/viewport/viewport.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iterator>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ernie/app/viewer/overlays/toolbar.h"
#include "ernie/app/viewer/widgets/widgets.h"
#include "ernie/io/image.h"
#include "ernie/render/renderer.h"
#include "ernie/util/log.h"

namespace ernie::app {
namespace {

constexpr double kOrbitSpeed = 0.005;
constexpr double kPanSpeed = 0.0015;
constexpr float kClickSlop = 4.0f;
constexpr float kNearClip = 0.01f;
constexpr float kFarClip = 1e6f;
constexpr double kOrthographicUnits = 0.1;
constexpr int kGridPieces = 16;
constexpr double kGridFadeSteps = 16;
constexpr Float3 kGridColor = {0.5f, 0.5f, 0.5f};
constexpr Float4 kGridXColor = {0.80f, 0.32f, 0.30f, 0.85f};
constexpr Float4 kGridYColor = {0.42f, 0.72f, 0.34f, 0.85f};
constexpr float kSelectionAxisPixels = 60;
constexpr double kFraming = 1.1;
constexpr Float4 kCameraGuideColor = {0.55f, 0.75f, 0.95f, 0.8f};
constexpr Float4 kLightGuideColor = {0.95f, 0.85f, 0.45f, 0.8f};
constexpr std::string_view kViewerOverlays = "viewer/";
constexpr std::string_view kGuidesOverlay = "viewer/guides";
constexpr std::string_view kSelectionOverlay = "viewer/selection";
constexpr std::string_view kCloudsOverlay = "viewer/clouds";
constexpr Float4 kCloudColors[] = {{0.30f, 0.85f, 0.95f, 1}, {0.95f, 0.55f, 0.25f, 1},
                                   {0.55f, 0.90f, 0.40f, 1}, {0.90f, 0.40f, 0.80f, 1},
                                   {0.95f, 0.90f, 0.35f, 1}, {0.50f, 0.60f, 1.00f, 1}};

struct GridLines {
    std::vector<Float3> endpoints;
    std::vector<Float4> colors;

    void Add(Float3 from, Float3 to, Float4 color, const Double3& centre, double reach) {
        for (int piece = 0; piece < kGridPieces; ++piece) {
            const float t0 = float(piece) / kGridPieces;
            const float t1 = float(piece + 1) / kGridPieces;
            const Float3 a{from.x + (to.x - from.x) * t0, from.y + (to.y - from.y) * t0, from.z};
            const Float3 b{from.x + (to.x - from.x) * t1, from.y + (to.y - from.y) * t1, from.z};
            const double r = std::hypot(0.5 * (a.x + b.x) - centre.x, 0.5 * (a.y + b.y) - centre.y);
            const double falloff = std::clamp(1 - r / reach, 0.0, 1.0);
            const float alpha = color.w * float(falloff * falloff);
            if (alpha > 0.004f) {
                endpoints.push_back(a);
                endpoints.push_back(b);
                colors.push_back({color.x, color.y, color.z, alpha});
            }
        }
    }
};

void AddGrid(Overlay& overlay, const GridGuide& grid, double minor, double fade,
             const Double3& centre) {
    const int64_t divisions = std::max<int64_t>(grid.divisions, 2);
    const double major = double(divisions) * minor;
    const double minorReach = std::max(double(grid.reach), 1.0) * minor;
    const double reach = minorReach * std::pow(double(divisions), fade);
    const auto z = float(centre.z);
    const Float4 minorColor = {kGridColor.x, kGridColor.y, kGridColor.z,
                               grid.minorBrightness * float(1 - fade)};
    const Float4 majorColor = {kGridColor.x, kGridColor.y, kGridColor.z, grid.majorBrightness};
    GridLines lines;
    const auto family = [&](double step, double extent, Float4 color, int64_t skip) {
        if (!(color.w > 0)) {
            return;
        }
        const auto count = int64_t(std::ceil(extent / step));
        const int64_t middleX = std::llround(centre.x / step);
        const int64_t middleY = std::llround(centre.y / step);
        for (int64_t k = -count; k <= count; ++k) {
            const int64_t across = middleX + k;
            const int64_t along = middleY + k;
            const double x = double(across) * step;
            const double y = double(along) * step;
            if (across != 0 && (skip == 0 || across % skip != 0)) {
                lines.Add({float(x), float(centre.y - extent), z},
                          {float(x), float(centre.y + extent), z}, color, centre, extent);
            }
            if (along != 0 && (skip == 0 || along % skip != 0)) {
                lines.Add({float(centre.x - extent), float(y), z},
                          {float(centre.x + extent), float(y), z}, color, centre, extent);
            }
        }
    };
    family(minor, minorReach, minorColor, divisions);
    family(major, reach, majorColor, 0);
    if (std::abs(centre.y) <= reach) {
        lines.Add({float(centre.x - reach), 0, z}, {float(centre.x + reach), 0, z}, kGridXColor,
                  centre, reach);
    }
    if (std::abs(centre.x) <= reach) {
        lines.Add({0, float(centre.y - reach), z}, {0, float(centre.y + reach), z}, kGridYColor,
                  centre, reach);
    }
    overlay.Lines(lines.endpoints, {.width = 1, .pickable = false}, lines.colors);
}

Float4 BoundsTint(uint32_t prim) {
    const float hue = std::fmod(float(prim) * 0.618034f, 1.0f) * 6;
    const float fall = 1 - std::abs(std::fmod(hue, 2.0f) - 1);
    const std::array<Float3, 6> sides = {Float3{1, fall, 0}, Float3{fall, 1, 0},
                                         Float3{0, 1, fall}, Float3{0, fall, 1},
                                         Float3{fall, 0, 1}, Float3{1, 0, fall}};
    const Float3 rgb = sides[size_t(hue) % 6];
    const auto lift = [](float c) { return 0.35f + 0.65f * c; };
    return {lift(rgb.x), lift(rgb.y), lift(rgb.z), 0.9f};
}

bool IsLight(const Prim& prim) {
    return (prim.IsA(SchemaId::BoundableLightBase) || prim.IsA(SchemaId::NonboundableLightBase)) &&
           !prim.IsA(SchemaId::DomeLight);
}

ImVec2 Place(const Placement& placement, ImVec2 min, ImVec2 max, ImVec2 size) {
    const int column = int(placement.anchor) % 3;
    const int row = int(placement.anchor) / 3;
    const float x = column == 0   ? min.x + placement.x
                    : column == 2 ? max.x - placement.x - size.x
                                  : 0.5f * (min.x + max.x - size.x) + placement.x;
    const float y = row == 0   ? min.y + placement.y
                    : row == 2 ? max.y - placement.y - size.y
                               : 0.5f * (min.y + max.y - size.y) + placement.y;
    return {std::round(x), std::round(y)};
}

class ErnieDelegate : public RenderDelegate {
  public:
    ErnieDelegate(std::shared_ptr<Renderer> renderer, Viewport& viewport, std::string shown)
        : m_renderer(std::move(renderer)), m_viewport(viewport), m_shown(std::move(shown)) {}

    std::optional<RenderFrame> Render(const RenderRequest& request) override {
        std::vector<std::string> products = m_viewport.CloudProducts();
        products.insert(products.begin(), std::string(request.product));
        const RenderOutput output = m_renderer->Step(products, 0);
        m_viewport.ShowClouds(output);
        const std::vector<FrameOutput>& frames = output[request.product].frames;
        if (frames.empty()) {
            return std::nullopt;
        }
        const FrameOutput& frame = frames[0];
        const RenderVarOutput& shown = frame.RenderVar(m_shown);
        const device::Tensor image = shown.tensor.Select(0);
        const device::Tensor overlay =
            frame.overlay.Empty() ? device::Tensor() : frame.overlay.Select(0);
        const AovInfo* info = FindAov(shown.sourceName);
        if (info == nullptr || info->kind == AovKind::Color) {
            return RenderFrame{image, output.Done(), overlay};
        }
        const AovDisplay& display = m_viewport.Display(shown.sourceName);
        device::Tensor colorized;
        switch (info->kind) {
        case AovKind::Hdr:
            colorized = ColorizeHdr(image, display.exposure);
            break;
        case AovKind::Normal:
            colorized = ColorizeNormals(image);
            break;
        case AovKind::Id:
            colorized = ColorizeIds(image);
            break;
        default: {
            device::Tensor bounds;
            colorized = ColorizeScalars(image, display.colors, bounds);
            m_viewport.ReadRange(bounds);
            break;
        }
        }
        return RenderFrame{colorized, image.GetDevice().Follow(Stream::None()), overlay};
    }

    std::string_view Name() const override {
        return "Ernie";
    }

  private:
    std::shared_ptr<Renderer> m_renderer;
    Viewport& m_viewport;
    std::string m_shown;
};

AovDisplay DefaultDisplay(std::string_view aov) {
    if (aov == "DistanceToCamera" || aov == "DistanceToImagePlane") {
        return {kDepthColors};
    }
    if (aov == "RadialVelocity") {
        return {{.colormap = Colormap::Turbo, .centered = true}};
    }
    return {};
}

}

const AovInfo* FindAov(std::string_view name) {
    const auto found = std::ranges::find(kAovs, name, &AovInfo::name);
    return found == std::end(kAovs) ? nullptr : &*found;
}

void CheckPixels(const device::Tensor& pixels) {
    const Span<const int64_t> shape = pixels.Shape();
    const Span<const int64_t> strides = pixels.Strides();
    if (pixels.Type() != device::DType::UInt8 || shape.size() != 3 || shape[2] != 4 ||
        (!strides.empty() && (strides[2] != 1 || strides[1] != 4 || strides[0] != shape[1] * 4))) {
        throw Error("The viewport shows packed height x width x 4 bytes of color");
    }
}

Viewport::Viewport(ViewportDesc desc, Ref<Scene> scene, Ref<device::Device> device,
                   Selection& selection, const Theme& theme,
                   std::function<void(const std::string&)> inspect, Editor* editor,
                   SessionWorlds* sessions)
    : m_desc(std::move(desc)), m_scene(std::move(scene)), m_device(std::move(device)),
      m_selection(selection), m_theme(theme), m_inspect(std::move(inspect)), m_sessions(sessions) {
    if (m_desc.delegate) {
        m_delegate = m_desc.delegate;
    } else if (m_scene) {
        DefineSession();
        m_renderer = m_desc.renderer ? m_desc.renderer : std::make_shared<Renderer>(m_scene);
        m_scene->Find(m_product)->Set(RenderProduct::kOverlays, std::vector<std::string>{"*"});
        m_delegate = std::make_shared<ErnieDelegate>(m_renderer, *this, m_shownVar);
    }
    if (m_product.empty()) {
        m_product = m_desc.product;
    }
    if (m_scene && !m_product.empty()) {
        m_caster = std::make_unique<RayCaster>(m_scene);
    }
    hudShown = m_desc.hud;
    toolbarShown = m_desc.toolbar;
    targetRenderFps = m_desc.targetRenderFps;
    for (const HudItem item : m_desc.hudItems) {
        hudItems[size_t(item)] = true;
    }
    m_toolbar = std::make_unique<Toolbar>(m_scene, m_product, *this, m_theme, editor);
    if (m_renderer && m_caster) {
        m_gizmo = std::make_unique<Gizmo>(m_scene, m_renderer, editor);
    }
}

Viewport::~Viewport() = default;

void Viewport::DefineSession() {
    std::string key;
    for (const char c : m_desc.product.empty() ? m_desc.name : m_desc.product) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            key += c;
        } else if (!key.empty()) {
            key += '_';
        }
    }
    std::string root;
    std::optional<ComposedWorld> claimed;
    for (uint32_t copy = 1;; ++copy) {
        root = std::string(kSessionRoot) + "/" + key + (copy > 1 ? fmt::format("_{}", copy) : "");
        if (m_sessions != nullptr) {
            if (const auto found = m_sessions->find(root);
                found != m_sessions->end() && !found->second.expired()) {
                break;
            }
        }
        const std::optional<Prim> existing = m_scene->Find(root);
        if (!existing) {
            break;
        }
        if (existing->IsA(SchemaId::World)) {
            try {
                claimed = m_scene->ComposeWorld(*existing);
                break;
            } catch (const Error&) {
            }
        }
    }
    m_product = root + "/Product";
    m_views = {root + "/Perspective", root + "/Top", root + "/Front", root + "/Right"};
    m_shownVar = root + "/Shown";
    m_aov = m_desc.aov;
    if (!FindAov(m_aov)) {
        throw Error("\"{}\" isn't an AOV the viewer shows", m_aov);
    }
    const auto share = [&] {
        if (m_sessions != nullptr) {
            if (const auto found = m_sessions->find(root); found != m_sessions->end()) {
                m_session = found->second.lock();
            }
        }
        if (!m_session) {
            m_session = std::make_shared<SessionWorld>(
                SessionWorld{.root = root,
                             .composed = claimed ? std::move(*claimed)
                                                 : m_scene->ComposeWorld(*m_scene->Find(root))});
            if (m_sessions != nullptr) {
                (*m_sessions)[root] = m_session;
            }
            ShowWorlds();
        }
    };
    if (m_scene->Find(m_product)) {
        share();
        return;
    }
    std::optional<Prim> source;
    if (!m_desc.product.empty()) {
        source = m_scene->Find(m_desc.product);
        if (!source || !source->IsA<RenderProduct>()) {
            throw Error("\"{}\" isn't a RenderProduct", m_desc.product);
        }
    }
    if (!m_scene->Find(kSessionRoot)) {
        m_scene->MarkSession(m_scene->Define<Scope>(kSessionRoot));
    }
    if (!m_scene->Find(root)) {
        m_scene->Define<World>(root);
    }
    share();
    m_scene->Define<RenderVar>(m_shownVar, {.sourceName = std::string(m_aov)});
    std::optional<Prim> camera;
    if (source) {
        m_scene->Paste(m_scene->Copy(*source), m_product);
        const std::vector<std::string> cameras =
            m_scene->Select(*source).Get("camera").GetTargets(0);
        camera = cameras.empty() ? std::nullopt : m_scene->Find(cameras[0]);
    } else {
        m_scene->Define<RenderProduct>(m_product, {.renderMode = RenderMode::RealTime});
    }
    if (camera && camera->IsA(SchemaId::Camera)) {
        m_scene->Paste(m_scene->Copy(*camera), m_views[0]);
    } else {
        const Double3 eye{20, 30, 18};
        m_scene->Define<Camera>(m_views[0], {.translate = eye,
                                             .orient = LookAt(eye, Double3{}),
                                             .clippingRange = {kNearClip, kFarClip}});
    }
    const Quatd orients[] = {LookAlong({0, 0, -1}, {0, 1, 0}), LookAlong({0, 1, 0}),
                             LookAlong({-1, 0, 0})};
    for (size_t view = 1; view < m_views.size(); ++view) {
        m_scene->Define<Camera>(m_views[view], {.orient = orients[view - 1],
                                                .projection = Projection::Orthographic,
                                                .clippingRange = {kNearClip, kFarClip}});
        m_unframed.insert(m_views[view]);
    }
    const View product = m_scene->Select(m_product);
    product.Get(RenderProduct::kCamera.Column()).SetTargets(0, {m_views[0]});
    product.Get(RenderProduct::kOrderedVars.Column()).SetTargets(0, {m_shownVar});
}

void Viewport::Advance(const device::Operation& lastDraw) {
    if (!m_shown && m_renderer) {
        m_builds = device::CurrentBuilds();
    } else {
        m_builds.reset();
    }
    RefreshWorlds();
    if (m_session && !m_leader && m_session->lighting == LightingMode::Camera && m_view &&
        (!m_session->aim || !(*m_session->aim == m_view->orient))) {
        m_scene->Select(m_session->root + "/Lights/Camera")
            .Get(Xformable::kOrient)
            .Set(m_view->orient);
        m_session->aim = m_view->orient;
    }
    LandFraming();
    if (m_bounds && m_bounds->Ready()) {
        std::array<float, 2> range{};
        if (m_bounds->Bytes().size() >= sizeof(range)) {
            std::memcpy(range.data(), m_bounds->Bytes().data(), sizeof(range));
            m_range = range;
        }
        m_bounds.reset();
    }
    std::erase_if(m_retired, [](const Retired& retired) { return retired.drawn.Ready(); });
    if (m_rendering && m_rendering->ready.Ready()) {
        const Clock::time_point now = Clock::now();
        m_stepMs = std::chrono::duration<double, std::milli>(now - m_stepStarted).count();
        if (m_lastRendered != Clock::time_point{}) {
            const double fps = 1.0 / std::chrono::duration<double>(now - m_lastRendered).count();
            m_renderFps = m_renderFps > 0 ? 0.9 * m_renderFps + 0.1 * fps : fps;
        }
        m_lastRendered = now;
        m_pending = Receive(m_rendering->image, m_rendering->ready, m_rendering->overlay);
        m_rendering.reset();
    }
    if (m_pending && m_pending->ready.Ready()) {
        if (m_shown) {
            m_retired.push_back({std::move(*m_shown), lastDraw, std::move(m_shownOverlay)});
        }
        m_shown = std::move(m_pending->pixels);
        m_shownOverlay = std::move(m_pending->overlay);
        m_shownReady = m_pending->ready;
        ++m_shownFrame;
        m_pending.reset();
    }
    if (m_leader || !m_delegate || m_rendering || m_pending || m_size.width == 0) {
        return;
    }
    if (targetRenderFps > 0 && m_stepStarted != Clock::time_point{} &&
        std::chrono::duration<double>(Clock::now() - m_stepStarted).count() <
            1.0 / targetRenderFps) {
        return;
    }
    const Extent base = resolution.custom ? Extent{resolution.width, resolution.height} : m_size;
    const float scale = std::clamp(resolution.scale, 0.01f, 8.0f);
    Extent size{std::max(1u, uint32_t(std::lround(float(base.width) * scale))),
                std::max(1u, uint32_t(std::lround(float(base.height) * scale)))};
    m_renderSize = size;
    if (!Fit(size)) {
        return;
    }
    if (m_builds && !m_builds->Done()) {
        return;
    }
    m_rendering = m_delegate->Render({m_product, size.width, size.height});
    m_stepStarted = Clock::now();
}

void Viewport::Show(const device::Tensor& image, const device::Operation& ready,
                    const device::Tensor& overlay) {
    m_pending = Receive(image, ready, overlay);
}

void Viewport::ShowSceneOverlays(bool shown) {
    m_sceneOverlays = shown;
    SyncOverlays();
}

std::vector<std::string> Viewport::SceneOverlayNames() const {
    std::vector<std::string> names =
        m_renderer ? m_renderer->GetOverlays()->Names() : std::vector<std::string>();
    std::erase_if(names, [](const std::string& name) { return name.starts_with(kViewerOverlays); });
    return names;
}

void Viewport::SyncOverlays() {
    if (!m_renderer) {
        return;
    }
    std::vector<std::string> shown;
    if (!m_sceneOverlays || !hiddenOverlays.empty()) {
        shown.push_back(std::string(kViewerOverlays) + "*");
    }
    if (m_sceneOverlays && !hiddenOverlays.empty()) {
        for (std::string& name : SceneOverlayNames()) {
            if (!hiddenOverlays.contains(name)) {
                shown.push_back(std::move(name));
            }
        }
    }
    if (shown == m_shownOverlays) {
        return;
    }
    m_scene->Find(m_product)->Set(RenderProduct::kOverlays,
                                  shown.empty() ? std::vector<std::string>{"*"} : shown);
    m_shownOverlays = std::move(shown);
}

Viewport::Pending Viewport::Receive(const device::Tensor& pixels, const device::Operation& ready,
                                    const device::Tensor& overlay) {
    CheckPixels(pixels);
    if (!overlay.Empty()) {
        CheckPixels(overlay);
        if (!std::ranges::equal(overlay.Shape(), pixels.Shape())) {
            throw Error("A viewport's overlay is {} x {}, and its image {} x {}",
                        overlay.Shape()[1], overlay.Shape()[0], pixels.Shape()[1],
                        pixels.Shape()[0]);
        }
    }
    Pending pending{pixels, ready, overlay};
    pending.pixels = Staging(pixels.Shape());
    pending.ready = m_device->Copy(pixels, pending.pixels, pending.ready);
    if (!overlay.Empty()) {
        pending.overlay = Staging(overlay.Shape(), pending.pixels.Address());
        pending.ready = m_device->Copy(overlay, pending.overlay, pending.ready);
    }
    return pending;
}

device::Tensor Viewport::Staging(Span<const int64_t> shape, uint64_t taken) {
    const auto held = [](const device::Tensor& tensor, uint64_t address) {
        return !tensor.Empty() && tensor.Address() == address;
    };
    const auto busy = [&](const device::Tensor& tensor) {
        const uint64_t address = tensor.Address();
        return address == taken || (m_shown && m_shown->Address() == address) ||
               held(m_shownOverlay, address) ||
               (m_pending &&
                (held(m_pending->pixels, address) || held(m_pending->overlay, address))) ||
               std::ranges::any_of(m_retired, [&](const Retired& retired) {
                   return held(retired.pixels, address) || held(retired.overlay, address);
               });
    };
    const auto fits = [&](const device::Tensor& tensor) {
        return std::ranges::equal(tensor.Shape(), shape);
    };
    std::erase_if(m_staging,
                  [&](const device::Tensor& tensor) { return !fits(tensor) && !busy(tensor); });
    for (const device::Tensor& tensor : m_staging) {
        if (fits(tensor) && !busy(tensor)) {
            return tensor;
        }
    }
    return m_staging.emplace_back(m_device->CreateTensor(
        device::DType::UInt8, {shape.begin(), shape.end()}, "viewer staging"));
}

void Viewport::Draw(Presenter& presenter, std::optional<ViewportRect> fixed, bool base) {
    Refresh();
    LandPick();
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    if (fixed) {
        ImGui::SetNextWindowPos({fixed->x, fixed->y});
        ImGui::SetNextWindowSize({fixed->width, fixed->height});
        flags |= ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground |
                 ImGuiWindowFlags_NoBringToFrontOnFocus;
    }
    if (base) {
        UpdateGuides();
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    const bool open = ImGui::Begin(m_desc.name.c_str(), fixed ? nullptr : &visible, flags);
    ImGui::PopStyleVar(2);
    m_hovered = false;
    if (open) {
        m_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 size = ImGui::GetContentRegionAvail();
        const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
        m_size = {uint32_t(std::max(size.x * scale.x, 1.0f)),
                  uint32_t(std::max(size.y * scale.y, 1.0f))};
        m_rect = {origin.x, origin.y, size.x, size.y};
        ImDrawList& list = *ImGui::GetWindowDrawList();
        const ImVec2 max{origin.x + size.x, origin.y + size.y};
        ImVec2 imageMin = origin;
        ImVec2 imageSize = size;
        if (const device::Tensor* shown = Shown()) {
            const ImVec2 texels{float(shown->Shape()[1]), float(shown->Shape()[0])};
            if (!resolution.fill && !m_leader) {
                imageSize = {texels.x / scale.x, texels.y / scale.y};
            } else {
                const float aspect = texels.x / texels.y;
                imageSize = size.x / size.y > aspect ? ImVec2{size.y * aspect, size.y}
                                                     : ImVec2{size.x, size.x / aspect};
            }
            imageMin = {origin.x + 0.5f * (size.x - imageSize.x),
                        origin.y + 0.5f * (size.y - imageSize.y)};
            presenter.Viewport(list, *shown, imageMin,
                               {imageMin.x + imageSize.x, imageMin.y + imageSize.y});
            const device::Tensor* overlay = ShownOverlay();
            if (overlay) {
                presenter.Image(list, *overlay, imageMin,
                                {imageMin.x + imageSize.x, imageMin.y + imageSize.y}, 1);
            }
        }
        m_imageMin = imageMin;
        m_imageSize = imageSize;
        if (m_gizmo && base) {
            m_gizmo->Update(m_selection, GizmoView(imageMin, imageSize));
        }
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton("##viewport", {std::max(size.x, 1.0f), std::max(size.y, 1.0f)},
                               ImGuiButtonFlags_MouseButtonLeft |
                                   ImGuiButtonFlags_MouseButtonRight |
                                   ImGuiButtonFlags_MouseButtonMiddle);
        const ImGuiIO& io = ImGui::GetIO();
        if (m_gizmo && base && !m_blocked && (ImGui::IsItemHovered() || m_gizmo->Active())) {
            const bool pressed =
                ImGui::IsItemActivated() && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
            if (m_gizmo->Input(io.MousePos, pressed, ImGui::IsMouseDown(ImGuiMouseButton_Left))) {
                m_inputClaimed = true;
            }
        }
        if (base && !m_blocked && !(m_gizmo && (m_gizmo->Active() || m_gizmo->Hovered()))) {
            TrackGestures(imageMin, imageSize,
                          ImGui::IsItemActivated() && ImGui::IsMouseClicked(ImGuiMouseButton_Left));
        }
        m_interacted = ImGui::IsItemActivated() || (ImGui::IsItemHovered() && io.MouseWheel != 0);
        if (!m_blocked && !m_inputClaimed && ImGui::IsItemDeactivated() &&
            ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
            io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Left] < kClickSlop * kClickSlop &&
            !io.KeyShift && imageSize.x > 0 && imageSize.y > 0) {
            const double u = (io.MousePos.x - imageMin.x) / imageSize.x;
            const double v = (io.MousePos.y - imageMin.y) / imageSize.y;
            if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
                StartPick(u, v, io.KeyCtrl);
            }
        }
        if (!m_blocked && !m_inputClaimed) {
            MoveCamera();
        }
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && !(m_gizmo && m_gizmo->Active())) {
            m_inputClaimed = false;
        }
        DrawItems(presenter, list, origin, max);
        if (base && guides.axes) {
            DrawAxes(list, origin, max);
        }
        if (m_builds && !m_shown) {
            const device::BuildProgress& builds = *m_builds;
            const float width = std::min(28.0f * ImGui::GetFontSize(), 0.8f * (max.x - origin.x));
            const float line = ImGui::GetTextLineHeightWithSpacing();
            ImGui::SetCursorScreenPos(
                {0.5f * (origin.x + max.x - width), 0.5f * (origin.y + max.y) - 2 * line});
            ImGui::BeginGroup();
            const float spinner = ImGui::GetTextLineHeight();
            const ImVec2 spot = ImGui::GetCursorScreenPos();
            const ImVec2 center{spot.x + 0.5f * spinner, spot.y + 0.5f * spinner};
            const float start = float(ImGui::GetTime()) * 6.0f;
            list.PathClear();
            list.PathArcTo(center, 0.4f * spinner, start, start + 4.5f, 24);
            list.PathStroke(ImGui::GetColorU32(ImGuiCol_PlotHistogram), 0.15f * spinner);
            ImGui::Dummy({spinner, spinner});
            ImGui::SameLine();
            ImGui::TextUnformatted(builds.Done() ? "Starting the renderer"
                                                 : "Preparing the renderer");
            const float done =
                builds.queued > 0 ? float(builds.finished) / float(builds.queued) : 0;
            ImGui::ProgressBar(
                done, {width, 0},
                fmt::format("{} of {} kernels", builds.finished, builds.queued).c_str());
            ImGui::PushStyleColor(ImGuiCol_Text, m_theme.textMuted);
            ImGui::TextUnformatted(
                builds.building.empty()
                    ? (builds.Done() ? "Rendering the first frame" : "Waiting for kernels")
                    : fmt::format("Compiling {}", builds.building.front()).c_str());
            if (builds.cached > 0) {
                ImGui::Text("%llu loaded from the shader cache",
                            static_cast<unsigned long long>(builds.cached));
            }
            ImGui::PopStyleColor();
            ImGui::EndGroup();
        }
        if (base && toolbarShown) {
            m_toolbar->Draw(origin);
        }
        const float dpi = ImGui::GetStyle().FontScaleDpi;
        const float pad = 8.0f * dpi;
        float right = max.x - pad;
        if (base && hudShown) {
            const float left =
                m_hud.Draw(list, {right, origin.y + pad}, m_theme, [&] { return HudLines(); });
            right = left < right ? left - pad : right;
        }
        for (const auto& overlay : m_overlays) {
            overlay(m_rect);
        }
    }
    ImGui::End();
}

void Viewport::UpdateGuides() {
    if (!m_renderer || !m_scene) {
        return;
    }
    SyncOverlays();
    const GridGuide& grid = guides.grid;
    double step = std::max(double(grid.spacing), 1e-6);
    double fade = 0;
    if (!grid.fixed) {
        const double base = std::max(double(grid.divisions), 2.0);
        const double level = std::log(std::max(0.25 * m_orbit.distance, 1e-3)) / std::log(base);
        step = std::pow(base, std::floor(level));
        fade = std::round((level - std::floor(level)) * kGridFadeSteps) / kGridFadeSteps;
    }
    const Double3 centre{std::round(m_orbit.target.x / step) * step,
                         std::round(m_orbit.target.y / step) * step, grid.height};
    GuideKey key{m_scene->StructureVersion(),
                 m_selection.paths,
                 m_session ? m_session->shown : std::vector<uint32_t>{},
                 guides,
                 grid.shown ? step : 0,
                 grid.shown ? fade : 0,
                 grid.shown ? centre : Double3{}};
    if (m_guideKey == key) {
        return;
    }
    m_guideKey = std::move(key);

    std::vector<uint32_t> cameras;
    std::vector<uint32_t> lights;
    std::vector<uint32_t> bounded;
    std::vector<uint32_t> sensors;
    std::vector<Prim> stack;
    const View roots = m_scene->Select("/*");
    for (const uint32_t root : roots.Prims()) {
        const Prim prim = m_scene->ByIndex(root);
        if (!prim.IsSession()) {
            stack.push_back(prim);
        }
    }
    while (!stack.empty()) {
        const Prim prim = stack.back();
        stack.pop_back();
        if (prim.IsA(SchemaId::Camera)) {
            cameras.push_back(prim.Index());
        } else if (prim.IsA(SchemaId::Lidar) || prim.IsA(SchemaId::Radar)) {
            sensors.push_back(prim.Index());
        } else if (IsLight(prim)) {
            lights.push_back(prim.Index());
        } else if (guides.allBounds && prim.IsA(SchemaId::Gprim)) {
            bounded.push_back(prim.Index());
        }
        for (const Prim& child : prim.Children()) {
            stack.push_back(child);
        }
    }

    std::vector<uint32_t> framed;
    std::vector<uint32_t> outlined;
    std::vector<uint32_t> groups;
    std::vector<uint32_t> chosenCameras;
    std::vector<uint32_t> chosenLights;
    std::vector<uint32_t> chosenSensors;
    std::vector<uint32_t> chosenNodes;
    for (uint32_t group = 0; group < m_selection.paths.size(); ++group) {
        const std::optional<Prim> selected = m_scene->Find(m_selection.paths[group]);
        if (!selected) {
            continue;
        }
        framed.push_back(selected->Index());
        stack.assign(1, *selected);
        while (!stack.empty()) {
            const Prim prim = stack.back();
            stack.pop_back();
            if (prim.IsA(SchemaId::Gprim)) {
                outlined.push_back(prim.Index());
                groups.push_back(group);
            } else if (prim.IsA(SchemaId::Camera)) {
                chosenCameras.push_back(prim.Index());
            } else if (prim.IsA(SchemaId::Lidar) || prim.IsA(SchemaId::Radar)) {
                chosenSensors.push_back(prim.Index());
            } else if (prim.IsA(SchemaId::SensorNode)) {
                chosenNodes.push_back(prim.Index());
            } else if (IsLight(prim)) {
                chosenLights.push_back(prim.Index());
            }
            for (const Prim& child : prim.Children()) {
                stack.push_back(child);
            }
        }
    }
    const auto unchosen = [](std::vector<uint32_t> prims, const std::vector<uint32_t>& chosen) {
        std::erase_if(
            prims, [&](uint32_t prim) { return std::ranges::find(chosen, prim) != chosen.end(); });
        return prims;
    };

    Overlay shown;
    if (grid.shown) {
        AddGrid(shown, grid, step, fade, centre);
    }
    if (!bounded.empty()) {
        std::vector<Float4> tints;
        for (const uint32_t prim : bounded) {
            tints.push_back(BoundsTint(prim));
        }
        shown.Prims(bounded, OverlayPrimShape::Bounds, {.width = 1, .pickable = false}, 1, {},
                    tints);
    }
    if (guides.cameras) {
        shown.Prims(unchosen(cameras, chosenCameras), OverlayPrimShape::Frustums,
                    {.color = kCameraGuideColor, .width = 1, .depth = OverlayDepth::XRay});
    }
    if (guides.lights) {
        shown.Prims(unchosen(lights, chosenLights), OverlayPrimShape::Lights,
                    {.color = kLightGuideColor, .width = 1, .depth = OverlayDepth::XRay});
    }
    m_renderer->GetOverlays()->Set(kGuidesOverlay, shown);

    Overlay selection;
    const OverlayStyle chosen{
        .color = guides.outlineColor, .width = 1.5f, .depth = OverlayDepth::XRay};
    if (guides.outline) {
        selection.Outline(outlined, {.color = guides.outlineColor, .width = 2});
    }
    if (guides.bounds) {
        selection.Prims(outlined, OverlayPrimShape::Bounds, chosen, 1, groups);
    }
    if (guides.frames) {
        selection.Prims(framed, OverlayPrimShape::Frames,
                        {.width = 2, .depth = OverlayDepth::XRay, .screenSized = true},
                        kSelectionAxisPixels);
    }
    selection.Prims(chosenCameras, OverlayPrimShape::Frustums, chosen);
    selection.Prims(chosenLights, OverlayPrimShape::Lights, chosen);
    m_renderer->GetOverlays()->Set(kSelectionOverlay, selection);
    if (guides.sensorClouds == SensorClouds::Off) {
        SyncSensorClouds({}, {}, {});
    } else {
        SyncSensorClouds(guides.sensorClouds == SensorClouds::Selected ? chosenSensors : sensors,
                         chosenNodes, sensors);
    }
}

namespace {

Float3 BeamDirection(float azimuth, float elevation) {
    const float a = azimuth * std::numbers::pi_v<float> / 180;
    const float e = elevation * std::numbers::pi_v<float> / 180;
    return {-std::sin(a) * std::cos(e), std::sin(e), -std::cos(a) * std::cos(e)};
}

std::vector<Float3> FieldLines(Float2 azimuths, Float2 elevations, float length) {
    constexpr int kSteps = 24;
    std::vector<Float3> lines;
    const auto at = [&](float azimuth, float elevation) {
        const Float3 direction = BeamDirection(azimuth, elevation);
        return Float3{direction.x * length, direction.y * length, direction.z * length};
    };
    const bool full = std::abs(azimuths.y - azimuths.x) >= 359.9f;
    for (const float elevation : {elevations.x, elevations.y}) {
        for (int step = 0; step < kSteps; ++step) {
            const float from = azimuths.x + (azimuths.y - azimuths.x) * float(step) / kSteps;
            const float to = azimuths.x + (azimuths.y - azimuths.x) * float(step + 1) / kSteps;
            lines.push_back(at(from, elevation));
            lines.push_back(at(to, elevation));
        }
    }
    if (!full) {
        for (const float azimuth : {azimuths.x, azimuths.y}) {
            for (const float elevation : {elevations.x, elevations.y}) {
                lines.push_back({0, 0, 0});
                lines.push_back(at(azimuth, elevation));
            }
            lines.push_back(at(azimuth, elevations.x));
            lines.push_back(at(azimuth, elevations.y));
        }
    }
    return lines;
}

}

void Viewport::SyncSensorClouds(std::vector<uint32_t> sensors, const std::vector<uint32_t>& nodes,
                                std::vector<uint32_t> candidates) {
    const std::vector<uint32_t> worlds = m_session ? m_session->shown : std::vector<uint32_t>{};
    const auto hidden = [&](uint32_t sensor) {
        const uint32_t world = m_scene->ByIndex(sensor).World();
        return !worlds.empty() && world != kGlobalWorld &&
               std::ranges::find(worlds, world) == worlds.end();
    };
    std::erase_if(sensors, hidden);
    std::erase_if(candidates, hidden);
    std::ranges::sort(sensors);
    sensors.erase(std::unique(sensors.begin(), sensors.end()), sensors.end());
    std::vector<std::pair<uint32_t, uint32_t>> taps;
    for (const uint32_t node : nodes) {
        std::optional<Prim> pipeline = m_scene->ByIndex(node).Parent();
        while (pipeline && !pipeline->IsA(SchemaId::SensorPipeline)) {
            pipeline = pipeline->Parent();
        }
        if (!pipeline) {
            continue;
        }
        for (const uint32_t sensor : candidates) {
            const std::vector<std::string> bound =
                m_scene->ByIndex(sensor).Get(Sensor::kPipelineBinding);
            if (!bound.empty() && bound.front() == pipeline->Path()) {
                taps.emplace_back(sensor, node);
            }
        }
    }
    if (sensors.size() + taps.size() > guides.mostSensorClouds) {
        Log(LogLevel::Warning,
            "{} lidar and radar clouds are shown, so the viewer draws the first {}",
            sensors.size() + taps.size(), guides.mostSensorClouds);
        sensors.resize(std::min<size_t>(sensors.size(), guides.mostSensorClouds));
        taps.resize(guides.mostSensorClouds - sensors.size());
    }
    const std::string root = m_product + "Clouds";
    std::vector<std::pair<uint32_t, uint32_t>> wanted;
    for (const uint32_t sensor : sensors) {
        wanted.emplace_back(sensor, kNoPrim);
    }
    wanted.insert(wanted.end(), taps.begin(), taps.end());
    std::vector<SensorCloud> clouds;
    for (const auto& [index, node] : wanted) {
        const Prim sensor = m_scene->ByIndex(index);
        const bool radar = sensor.IsA(SchemaId::Radar);
        const std::string product = node == kNoPrim
                                        ? std::format("{}/Cloud{}", root, index)
                                        : std::format("{}/Cloud{}Stage{}", root, index, node);
        const auto kept = std::ranges::find(m_sensorClouds, product, &SensorCloud::product);
        if (kept != m_sensorClouds.end()) {
            clouds.push_back(std::move(*kept));
            continue;
        }
        if (!m_scene->Find(root)) {
            m_scene->Define<Scope>(root);
        }
        std::vector<std::string> channels = {"Coordinates", "Distance", "Intensity"};
        if (radar) {
            channels.insert(channels.end(), {"RadialVelocity", "RadialVelocityVector"});
        }
        const std::string var = product + "Points";
        if (node == kNoPrim) {
            m_scene->Define<RenderVar>(var, {.sourceName = "PointCloud", .channels = channels});
        } else {
            m_scene->Define<RenderVar>(var, {.sourceName = "Tap",
                                             .channels = channels,
                                             .tap = {m_scene->ByIndex(node).Path()}});
        }
        m_scene->Define<RenderProduct>(product, {.camera = {sensor.Path()}, .orderedVars = {var}});
        SensorCloud cloud{
            .product = product,
            .var = var,
            .sensor = index,
            .stage = node != kNoPrim,
            .radar = radar,
            .world = sensor.World(),
            .color = kCloudColors[(index + (node == kNoPrim ? 0 : 3)) % std::size(kCloudColors)]};
        if (radar) {
            const Float2 field = sensor.Get(Radar::kFieldOfView);
            cloud.range = sensor.Get(Radar::kRange).y;
            cloud.field =
                FieldLines({field.x / 2, -field.x / 2}, {-field.y / 2, field.y / 2}, cloud.range);
        } else {
            const std::vector<float> elevations = sensor.Get(Lidar::kEmitterElevations);
            std::vector<float> azimuths = sensor.Get(Lidar::kEmitterAzimuths);
            Float2 sweep{0, 0};
            if (sensor.Get(Lidar::kScanType) == ScanType::Rotary) {
                sweep = sensor.Get(Lidar::kAzimuthRange);
            } else if (!azimuths.empty()) {
                sweep = {*std::ranges::min_element(azimuths), *std::ranges::max_element(azimuths)};
            }
            cloud.range = sensor.Get(Lidar::kRange).y;
            if (!elevations.empty()) {
                cloud.field = FieldLines(
                    sweep,
                    {*std::ranges::min_element(elevations), *std::ranges::max_element(elevations)},
                    cloud.range);
            }
        }
        clouds.push_back(std::move(cloud));
    }
    for (const SensorCloud& cloud : m_sensorClouds) {
        if (std::ranges::find(clouds, cloud.product, &SensorCloud::product) == clouds.end()) {
            for (const std::string& path : {cloud.product, cloud.var}) {
                if (const std::optional<Prim> prim = m_scene->Find(path)) {
                    m_scene->Remove(*prim);
                }
            }
        }
    }
    m_sensorClouds = std::move(clouds);
    if (m_sensorClouds.empty()) {
        m_renderer->GetOverlays()->Set(kCloudsOverlay, Overlay());
    }
}

std::vector<std::string> Viewport::CloudProducts() const {
    std::vector<std::string> products;
    for (const SensorCloud& cloud : m_sensorClouds) {
        products.push_back(cloud.product);
    }
    return products;
}

void Viewport::ShowClouds(const RenderOutput& output) {
    if (m_sensorClouds.empty()) {
        return;
    }
    constexpr float kArrowSeconds = 0.5f;
    const ColorizeOptions velocityColors{
        .colormap = Colormap::Turbo, .minimum = -10.0f, .maximum = 10.0f, .centered = true};
    Overlay shown;
    for (SensorCloud& cloud : m_sensorClouds) {
        if (output.Contains(cloud.product) && !output[cloud.product].frames.empty()) {
            const RenderVarOutput& points =
                output[cloud.product].frames.front().RenderVar(cloud.var);
            const device::Tensor& coordinates = points.channels.at("Coordinates");
            cloud.points = coordinates.Select(0);
            cloud.count = points.channels.at("Counts");
            cloud.colors = {};
            const CloudColor mode = guides.cloudColor;
            if (mode != CloudColor::Sensor && (mode != CloudColor::Velocity || cloud.radar)) {
                ColorizeOptions options{.colormap = Colormap::Turbo};
                device::Tensor values;
                switch (mode) {
                case CloudColor::Intensity:
                    values = points.channels.at("Intensity");
                    options.colormap = Colormap::Viridis;
                    options.minimum = 0;
                    options.maximum = cloud.radar ? 10.0f : 1.0f;
                    break;
                case CloudColor::Range:
                    values = points.channels.at("Distance");
                    options.minimum = 0;
                    options.maximum = cloud.range;
                    break;
                case CloudColor::Height:
                    values = coordinates;
                    options.channel = 1;
                    options.minimum = -5;
                    options.maximum = 5;
                    break;
                default:
                    values = points.channels.at("RadialVelocity");
                    options = velocityColors;
                    break;
                }
                cloud.colors = ColorizeScalars(values, options).Select(0);
            }
            cloud.vectors = {};
            cloud.vectorColors = {};
            if (cloud.radar && guides.velocityArrows) {
                cloud.vectors = points.channels.at("RadialVelocityVector").Select(0);
                cloud.vectorColors =
                    mode == CloudColor::Velocity
                        ? cloud.colors
                        : ColorizeScalars(points.channels.at("RadialVelocity"), velocityColors)
                              .Select(0);
            }
        }
        const OverlayStyle anchored{.color = cloud.color,
                                    .width = 1,
                                    .depth = guides.cloudsThroughGeometry ? OverlayDepth::XRay
                                                                          : OverlayDepth::Test,
                                    .world = cloud.world,
                                    .anchor = cloud.sensor,
                                    .pickable = false};
        if (!cloud.points.Empty()) {
            OverlayStyle style = anchored;
            style.size = cloud.stage ? 4 : 3;
            style.marker = cloud.stage ? MarkerShape::Square : MarkerShape::Circle;
            shown.Add(OverlayMarkers{.points = cloud.points,
                                     .colors = cloud.colors,
                                     .itemCount = cloud.count,
                                     .style = style});
        }
        if (cloud.stage) {
            continue;
        }
        if (guides.sensorFields && !cloud.field.empty()) {
            OverlayStyle style = anchored;
            style.color.w = 0.6f;
            style.depth = OverlayDepth::XRay;
            shown.Lines(cloud.field, style);
        }
        if (guides.rangeRings && cloud.range > 0) {
            OverlayStyle style = anchored;
            style.color.w = 0.4f;
            for (int ring = 1; ring <= 4; ++ring) {
                shown.Circle({0, 0, 0}, {0, 1, 0}, cloud.range * float(ring) / 4, style);
            }
        }
        if (cloud.radar && guides.velocityArrows && !cloud.vectors.Empty()) {
            shown.Add(OverlayArrows{.points = cloud.points,
                                    .vectors = cloud.vectors,
                                    .vectorScale = kArrowSeconds,
                                    .colors = cloud.vectorColors,
                                    .itemCount = cloud.count,
                                    .style = anchored});
        }
    }
    m_renderer->GetOverlays()->Set(kCloudsOverlay, shown);
}

void Viewport::AddOverlay(std::function<void(const ViewportRect&)> draw) {
    m_overlays.push_back(std::move(draw));
}

void Viewport::AddText(uint64_t id, std::string text, const TextStyle& style) {
    m_texts.push_back({id, std::move(text), style});
}

bool Viewport::SetText(uint64_t id, std::string text) {
    for (TextItem& item : m_texts) {
        if (item.id == id) {
            item.text = std::move(text);
            return true;
        }
    }
    return false;
}

void Viewport::AddImage(uint64_t id, device::Tensor pixels, device::Operation ready,
                        const Placement& placement, float opacity) {
    CheckPixels(pixels);
    if (&pixels.GetDevice() != m_device.Get()) {
        const device::Tensor copy = m_device->CreateTensor(
            device::DType::UInt8, {pixels.Shape()[0], pixels.Shape()[1], 4}, "viewport image");
        ready = m_device->Copy(pixels, copy, ready);
        pixels = copy;
    }
    m_images.push_back({id, std::move(pixels), ready, placement, opacity});
}

void Viewport::AddColorbar(uint64_t id, const ColorbarDesc& desc) {
    m_colorbars.push_back({id, desc});
}

bool Viewport::SetColorbar(uint64_t id, const ColorbarDesc& desc) {
    for (ColorbarItem& item : m_colorbars) {
        if (item.id == id) {
            item.desc = desc;
            return true;
        }
    }
    return false;
}

bool Viewport::RemoveOverlay(uint64_t id) {
    return std::erase_if(m_texts, [&](const TextItem& item) { return item.id == id; }) +
               std::erase_if(m_images, [&](const ImageItem& item) { return item.id == id; }) +
               std::erase_if(m_colorbars, [&](const ColorbarItem& item) { return item.id == id; }) >
           0;
}

void Viewport::AddHudItem(std::string label, std::function<std::string()> value) {
    m_hudEntries.push_back({std::move(label), std::move(value)});
}

void Viewport::DrawItems(Presenter& presenter, ImDrawList& list, ImVec2 min, ImVec2 max) {
    for (const ImageItem& image : m_images) {
        if (!image.ready.Ready()) {
            continue;
        }
        const float width = float(image.pixels.Shape()[1]);
        const float height = float(image.pixels.Shape()[0]);
        ImVec2 size{image.placement.width, image.placement.height};
        if (size.x <= 0 && size.y <= 0) {
            size = {width, height};
        } else if (size.x <= 0) {
            size.x = size.y * width / height;
        } else if (size.y <= 0) {
            size.y = size.x * height / width;
        }
        const ImVec2 corner = Place(image.placement, min, max, size);
        presenter.Image(list, image.pixels, corner, {corner.x + size.x, corner.y + size.y},
                        image.opacity);
    }
    for (const ColorbarItem& colorbar : m_colorbars) {
        DrawColorbar(list, min, max, colorbar.desc);
    }
    const AovInfo* aov = FindAov(m_aov);
    if (m_range && aov && aov->kind == AovKind::Scalar) {
        const ColorizeOptions& colors = Display().colors;
        DrawColorbar(list, min, max,
                     {.colormap = colors.colormap,
                      .minimum = (*m_range)[0],
                      .maximum = (*m_range)[1],
                      .logarithmic = colors.logarithmic,
                      .reversed = colors.reversed,
                      .label = std::string(aov->name),
                      .units = aov->units});
    }
    for (const TextItem& item : m_texts) {
        const TextStyle& style = item.style;
        ImFont* font = ImGui::GetFont();
        const float size = style.size > 0 ? style.size : ImGui::GetFontSize();
        const float pad = style.background ? 0.35f * size : 0.0f;
        const ImVec2 text = font->CalcTextSizeA(size, FLT_MAX, 0.0f, item.text.c_str());
        const ImVec2 corner =
            Place(style.placement, min, max, {text.x + 2 * pad, text.y + 2 * pad});
        const ImVec2 at{corner.x + pad, corner.y + pad};
        if (style.background) {
            list.AddRectFilled(corner, {corner.x + text.x + 2 * pad, corner.y + text.y + 2 * pad},
                               ImGui::GetColorU32(m_theme.overlay), 0.3f * size);
        }
        if (style.shadow) {
            const float offset = std::max(1.0f, std::round(size / 14));
            list.AddText(font, size, {at.x + offset, at.y + offset},
                         IM_COL32(0, 0, 0, int(160 * style.color.w)), item.text.c_str());
        }
        list.AddText(
            font, size, at,
            ImGui::GetColorU32({style.color.x, style.color.y, style.color.z, style.color.w}),
            item.text.c_str());
    }
}

void Viewport::DrawAxes(ImDrawList& list, ImVec2 min, ImVec2 max) const {
    if (!m_view) {
        return;
    }
    const auto unit = [](Float3 v) {
        const double length = std::sqrt(double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
        return length > 0 ? Double3{v.x / length, v.y / length, v.z / length} : Double3{};
    };
    const Double3 right = unit(m_view->frame.x);
    const Double3 up = unit(m_view->frame.y);
    const Double3 back = unit(m_view->frame.z);

    struct Axis {
        double x;
        double y;
        double z;
        ImU32 color;
        const char* label;
    };

    std::array<Axis, 3> axes{{{right.x, up.x, back.x, kAxisColors[0], "X"},
                              {right.y, up.y, back.y, kAxisColors[1], "Y"},
                              {right.z, up.z, back.z, kAxisColors[2], "Z"}}};
    std::ranges::sort(axes, {}, &Axis::z);
    const float dpi = ImGui::GetStyle().FontScaleDpi;
    const float radius = 28.0f * dpi;
    const ImVec2 centre{min.x + 8.0f * dpi + radius, max.y - 8.0f * dpi - radius};
    list.AddCircleFilled(centre, radius + 4.0f * dpi,
                         ImGui::GetColorU32(WithAlpha(m_theme.overlay, 0.6f)));
    const float reach = 0.68f * radius;
    for (const Axis& axis : axes) {
        const ImVec2 tip{centre.x + float(axis.x) * reach, centre.y - float(axis.y) * reach};
        const ImU32 color =
            axis.z < 0 ? (axis.color & ~IM_COL32_A_MASK) | (0x73u << IM_COL32_A_SHIFT) : axis.color;
        list.AddLine(centre, tip, color, 2.0f * dpi);
        list.AddCircleFilled(tip, 7.0f * dpi, color);
        const ImVec2 text = ImGui::CalcTextSize(axis.label);
        list.AddText({tip.x - 0.5f * text.x, tip.y - 0.5f * text.y}, IM_COL32_WHITE, axis.label);
    }
}

void Viewport::DrawColorbar(ImDrawList& list, ImVec2 min, ImVec2 max, const ColorbarDesc& desc) {
    const bool vertical = desc.placement.height >= desc.placement.width;
    const ImVec2 size{std::max(desc.placement.width, 4.0f), std::max(desc.placement.height, 4.0f)};
    const ImVec2 corner = Place(desc.placement, min, max, size);
    constexpr int kSteps = 64;
    for (int step = 0; step < kSteps; ++step) {
        const float from = float(step) / kSteps;
        const float to = float(step + 1) / kSteps;
        const auto color = [&](float t) {
            const std::array<uint8_t, 4> rgba =
                ColormapColor(desc.colormap, desc.reversed ? 1 - t : t);
            return IM_COL32(rgba[0], rgba[1], rgba[2], rgba[3]);
        };
        if (vertical) {
            const ImVec2 low{corner.x, corner.y + size.y * (1 - to)};
            const ImVec2 high{corner.x + size.x, corner.y + size.y * (1 - from)};
            list.AddRectFilledMultiColor(low, high, color(to), color(to), color(from), color(from));
        } else {
            const ImVec2 low{corner.x + size.x * from, corner.y};
            const ImVec2 high{corner.x + size.x * to, corner.y + size.y};
            list.AddRectFilledMultiColor(low, high, color(from), color(to), color(to), color(from));
        }
    }
    list.AddRect(corner, {corner.x + size.x, corner.y + size.y},
                 ImGui::GetColorU32(m_theme.overlayBorder));
    const ImU32 text = ImGui::GetColorU32(m_theme.text);
    const ImU32 shadow = IM_COL32(255, 255, 255, 170);
    const auto label = [&](ImVec2 at, const std::string& value) {
        list.AddText({at.x + 1, at.y + 1}, shadow, value.c_str());
        list.AddText(at, text, value.c_str());
    };
    const uint32_t ticks = std::max(desc.ticks, 2u);
    for (uint32_t tick = 0; tick < ticks; ++tick) {
        const float t = float(tick) / float(ticks - 1);
        const float value = desc.logarithmic && desc.minimum > 0 && desc.maximum > 0
                                ? std::exp(std::log(desc.minimum) +
                                           t * (std::log(desc.maximum) - std::log(desc.minimum)))
                                : desc.minimum + t * (desc.maximum - desc.minimum);
        const std::string number = fmt::format("{:.3g}", value);
        const ImVec2 extent = ImGui::CalcTextSize(number.c_str());
        if (vertical) {
            const float y = corner.y + size.y * (1 - t);
            list.AddLine({corner.x - 4, y}, {corner.x, y}, text);
            label({corner.x - 6 - extent.x, y - 0.5f * extent.y}, number);
        } else {
            const float x = corner.x + size.x * t;
            list.AddLine({x, corner.y + size.y}, {x, corner.y + size.y + 4}, text);
            label({x - 0.5f * extent.x, corner.y + size.y + 5}, number);
        }
    }
    std::string title = desc.label;
    if (!desc.units.empty()) {
        title += (title.empty() ? "" : " ") + ("(" + desc.units + ")");
    }
    if (!title.empty()) {
        const ImVec2 extent = ImGui::CalcTextSize(title.c_str());
        label(vertical ? ImVec2{corner.x + size.x - extent.x, corner.y - extent.y - 4}
                       : ImVec2{corner.x, corner.y - extent.y - 4},
              title);
    }
}

std::vector<std::string> Viewport::HudLines() const {
    std::vector<std::string> lines;
    if (m_shown) {
        const device::BuildProgress builds = device::CurrentBuilds();
        if (!builds.Done()) {
            lines.push_back(
                fmt::format("Compiling kernels: {} left", builds.queued - builds.finished));
        }
    }
    const auto shown = [&](HudItem item) { return hudItems[size_t(item)]; };
    const double ui = std::max(double(ImGui::GetIO().Framerate), 1e-3);
    if (shown(HudItem::Fps)) {
        lines.push_back(
            m_renderFps > 0
                ? fmt::format("FPS: {:.1f}, Step time: {:.2f} ms", m_renderFps, m_stepMs)
                : fmt::format("FPS: {:.1f}, Frame time: {:.2f} ms", ui, 1000 / ui));
    }
    if (shown(HudItem::UiFps) && m_renderFps > 0) {
        lines.push_back(fmt::format("UI FPS: {:.1f}", ui));
    }
    if (shown(HudItem::DeviceMemory)) {
        lines.push_back("Device Memory: " + FormatBytes(m_device->MemoryUsage().deviceBytes) +
                        " / " + FormatBytes(m_device->Adapter().memoryBytes));
    }
    if (shown(HudItem::HostMemory)) {
        const std::optional<uint64_t> resident = ResidentBytes();
        lines.push_back("Host Memory: " + (resident ? FormatBytes(*resident) : "unknown"));
    }
    if (shown(HudItem::Resolution)) {
        const Extent size = m_renderSize.width > 0 ? m_renderSize : m_size;
        lines.push_back(fmt::format("Resolution: {} x {}", size.width, size.height));
    }
    if (shown(HudItem::Camera) && !m_camera.empty()) {
        const std::string& path = m_camera;
        std::string line = "Camera: " + path.substr(path.rfind('/') + 1);
        if (const std::optional<float> focal = FocalLength()) {
            line += fmt::format(", {:.0f} mm", *focal);
        }
        lines.push_back(std::move(line));
    }
    if (shown(HudItem::Renderer) && m_delegate) {
        lines.push_back("Renderer: " + std::string(m_delegate->Name()));
    }
    if (shown(HudItem::Selection)) {
        const std::vector<std::string>& selected = m_selection.paths;
        lines.push_back("Selection: " + (selected.empty() ? std::string("none")
                                         : selected.size() == 1
                                             ? selected.front()
                                             : fmt::format("{} prims", selected.size())));
    }
    for (const HudEntry& entry : m_hudEntries) {
        lines.push_back(entry.label + ": " + entry.value());
    }
    return lines;
}

void Viewport::Refresh() {
    if (!m_caster) {
        return;
    }
    if (m_scene->StructureVersion() != m_structure) {
        m_structure = m_scene->StructureVersion();
        const std::vector<std::string> cameras =
            m_scene->Select(m_product).Get("camera").GetTargets(0);
        const std::optional<Prim> found =
            cameras.empty() ? std::nullopt : m_scene->Find(cameras.front());
        m_camera = found ? found->Path() : std::string();
        m_cameraWorld = found ? found->World() : 0;
        m_reads.clear();
        m_view.reset();
        m_fitted = false;
    }
    if (m_camera.empty()) {
        return;
    }
    if (!m_reads.empty()) {
        if (!std::ranges::all_of(m_reads,
                                 [](const device::Download& read) { return read.Ready(); })) {
            return;
        }
        const auto read = [&]<class T>(size_t index, T value) {
            if (m_reads[index].Bytes().size() >= sizeof(T)) {
                std::memcpy(&value, m_reads[index].Bytes().data(), sizeof(T));
            }
            return value;
        };
        CameraView view;
        view.world = m_cameraWorld;
        view.translate = read(0, Double3{});
        view.orient = read(1, Quatd{});
        view.frame = read(2, WorldTransform{});
        view.focal = read(3, 0.0f);
        view.width = read(4, 0.0f);
        view.height = read(5, 0.0f);
        view.right = read(6, 0.0f);
        view.up = read(7, 0.0f);
        view.clipping = read(8, Float2{});
        view.orthographic = read(9, uint32_t(0)) == uint32_t(Projection::Orthographic);
        const Box box = read(10, Box{{1, 1, 1}, {0, 0, 0}});
        if (m_reads[10].Bytes().size() >= sizeof(Box) && box.lower.x <= box.upper.x) {
            view.bounds = box;
        }
        m_view = view;
        m_reads.clear();
    }
    const View camera = m_scene->Select(m_camera);
    m_reads.push_back(camera.Get(Camera::kTranslate.Column()).ReadHostAsync());
    m_reads.push_back(camera.Get(Camera::kOrient.Column()).ReadHostAsync());
    m_reads.push_back(m_scene->ReadWorldTransformsAsync(std::vector<std::string>{m_camera}));
    for (const ColumnRef column :
         {Camera::kFocalLength.Column(), Camera::kHorizontalAperture.Column(),
          Camera::kVerticalAperture.Column(), Camera::kHorizontalApertureOffset.Column(),
          Camera::kVerticalApertureOffset.Column(), Camera::kClippingRange.Column(),
          Camera::kProjection.Column()}) {
        m_reads.push_back(camera.Get(column).ReadHostAsync());
    }
    m_reads.push_back(m_scene->ReadBoundsAsync());
}

bool Viewport::OwnsCamera() const {
    return std::ranges::find(m_views, m_camera) != m_views.end();
}

bool Viewport::Fit(Extent size) {
    if (!m_caster || size.width == 0 || size.height == 0) {
        return true;
    }
    if (size.width != m_resolution.width || size.height != m_resolution.height) {
        m_scene->Select(m_product).Get<Uint2>("resolution").Set(Uint2{size.width, size.height});
        m_resolution = size;
        m_fitted = false;
    }
    if (m_view && !m_camera.empty()) {
        const float width = m_view->height * float(size.width) / float(size.height);
        if (OwnsCamera() && (!m_fitted || std::abs(m_view->width - width) > 1e-4f * width)) {
            m_view->width = width;
            m_scene->Select(m_camera).Get(Camera::kHorizontalAperture).Set(width);
        }
        m_fitted = true;
    }
    return m_structure == m_scene->StructureVersion() && (m_fitted || m_camera.empty());
}

bool Viewport::StartOrbit() {
    if (!m_view) {
        return false;
    }
    std::optional<Double3> focus;
    if (const std::optional<Box>& bounds = m_view->bounds) {
        focus = Double3{(double(bounds->lower.x) + bounds->upper.x) / 2,
                        (double(bounds->lower.y) + bounds->upper.y) / 2,
                        (double(bounds->lower.z) + bounds->upper.z) / 2};
    }
    m_orbit.LookFrom(m_view->translate, m_view->orient, focus);
    return true;
}

void Viewport::Frame(bool selected) {
    if (m_camera.empty()) {
        return;
    }
    std::vector<Prim> stack;
    if (selected) {
        for (const std::string& path : m_selection.paths) {
            if (const std::optional<Prim> prim = m_scene->Find(path)) {
                stack.push_back(*prim);
            }
        }
    }
    std::vector<std::string> bounded;
    while (!stack.empty()) {
        const Prim prim = stack.back();
        stack.pop_back();
        if (prim.IsA(SchemaId::Boundable)) {
            bounded.push_back(prim.Path());
        }
        for (const Prim& child : prim.Children()) {
            stack.push_back(child);
        }
    }
    m_framing.clear();
    m_framingPrims = !bounded.empty();
    if (bounded.empty()) {
        m_framingWorlds = m_session && !m_session->shown.empty();
        m_framing.push_back(m_framingWorlds ? m_scene->ReadAllBoundsAsync()
                                            : m_scene->ReadBoundsAsync());
        return;
    }
    m_framing.push_back(m_scene->ReadPrimBoundsAsync(Span<const std::string>(bounded)));
}

void Viewport::LandFraming() {
    if (m_framing.empty() ||
        !std::ranges::all_of(m_framing,
                             [](const device::Download& read) { return read.Ready(); }) ||
        !StartOrbit()) {
        return;
    }
    Box box{{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
    const auto grow = [&](const Float3& lower, const Float3& upper) {
        if (lower.x > upper.x) {
            return;
        }
        box.lower = {std::min(box.lower.x, lower.x), std::min(box.lower.y, lower.y),
                     std::min(box.lower.z, lower.z)};
        box.upper = {std::max(box.upper.x, upper.x), std::max(box.upper.y, upper.y),
                     std::max(box.upper.z, upper.z)};
    };
    if (m_framingPrims) {
        const Span<const std::byte> bytes = m_framing[0].Bytes();
        for (size_t at = 0; at + sizeof(Box) <= bytes.size(); at += sizeof(Box)) {
            Box read;
            std::memcpy(&read, bytes.data() + at, sizeof(Box));
            if (read.lower.x != read.upper.x || read.lower.y != read.upper.y ||
                read.lower.z != read.upper.z) {
                grow(read.lower, read.upper);
            }
        }
    } else if (m_framing.size() == 1 && m_framingWorlds) {
        const Span<const std::byte> bytes = m_framing[0].Bytes();
        const auto stored = [&](size_t slot) {
            Box read{{1, 1, 1}, {0, 0, 0}};
            if ((slot + 1) * sizeof(Box) <= bytes.size()) {
                std::memcpy(&read, bytes.data() + slot * sizeof(Box), sizeof(Box));
            }
            return read;
        };
        if (m_session->display.global) {
            const Box global = stored(0);
            grow(global.lower, global.upper);
        }
        for (size_t at = 0; at < m_session->shown.size(); ++at) {
            const Box world = stored(size_t(m_session->shown[at]) + 1);
            const uint32_t index = m_session->shown[at];
            const Double3 origin =
                index < m_session->origins.size() ? m_session->origins[index] : Double3{};
            const Double3 offset =
                at < m_session->offsets.size() ? m_session->offsets[at] : Double3{};
            const Double3 shift{origin.x + offset.x, origin.y + offset.y, origin.z + offset.z};
            const Float3 by{float(shift.x), float(shift.y), float(shift.z)};
            grow({world.lower.x + by.x, world.lower.y + by.y, world.lower.z + by.z},
                 {world.upper.x + by.x, world.upper.y + by.y, world.upper.z + by.z});
        }
    } else if (m_framing.size() == 1) {
        Box whole{{1, 1, 1}, {0, 0, 0}};
        if (m_framing[0].Bytes().size() >= sizeof(Box)) {
            std::memcpy(&whole, m_framing[0].Bytes().data(), sizeof(Box));
        }
        grow(whole.lower, whole.upper);
    }
    const bool unpublished =
        !m_framingPrims && std::ranges::any_of(m_framing, [](const device::Download& read) {
            return read.Bytes().empty();
        });
    m_framing.clear();
    if (unpublished) {
        m_framing.push_back(m_framingWorlds ? m_scene->ReadAllBoundsAsync()
                                            : m_scene->ReadBoundsAsync());
        return;
    }
    if (box.lower.x > box.upper.x) {
        return;
    }
    const Double3 extent{double(box.upper.x) - box.lower.x, double(box.upper.y) - box.lower.y,
                         double(box.upper.z) - box.lower.z};
    const double radius = std::max(
        0.5 * std::sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z), 1e-3);
    const Double3 center{(double(box.lower.x) + box.upper.x) / 2,
                         (double(box.lower.y) + box.upper.y) / 2,
                         (double(box.lower.z) + box.upper.z) / 2};
    if (m_view->orthographic) {
        const Float3 back = m_view->frame.z;
        const double away = 4 * radius + 1;
        const float height = float(2 * kFraming * radius / kOrthographicUnits);
        const float width = height * m_view->width / std::max(m_view->height, 1e-6f);
        const View camera = m_scene->Select(m_camera);
        camera.Get<Double3>("translate")
            .Set(Double3{center.x + back.x * away, center.y + back.y * away,
                         center.z + back.z * away});
        camera.Get(Camera::kVerticalAperture).Set(height);
        camera.Get(Camera::kHorizontalAperture).Set(width);
        m_view->height = height;
        m_view->width = width;
        return;
    }
    const double aperture = std::max(std::min(m_view->width, m_view->height), 1e-3f);
    const double half = std::atan(0.5 * aperture / std::max(m_view->focal, 1e-3f));
    m_orbit.target = center;
    m_orbit.distance = kFraming * radius / std::sin(std::max(half, 1e-3));
    const View camera = m_scene->Select(m_camera);
    camera.Get<Double3>("translate").Set(m_orbit.Eye());
    camera.Get<Quatd>("orient").Set(m_orbit.Orient());
    m_orbiting = true;
}

void Viewport::MoveCamera() {
    if (!m_caster) {
        return;
    }
    const ImGuiIO& io = ImGui::GetIO();
    const bool active = ImGui::IsItemActive();
    const bool scrolled = ImGui::IsItemHovered() && io.MouseWheel != 0;
    if (m_view && m_view->orthographic) {
        if (ImGui::IsItemActivated() || (scrolled && !active)) {
            m_orthoEye = m_view->translate;
            m_orthoWidth = m_view->width;
            m_orthoHeight = m_view->height;
        }
        const ImVec2 drag = io.MouseDelta;
        const View camera = m_scene->Select(m_camera);
        if (active && (drag.x != 0 || drag.y != 0)) {
            const double units = m_orthoHeight * kOrthographicUnits / std::max(m_size.height, 1u);
            const Float3 right = m_view->frame.x;
            const Float3 up = m_view->frame.y;
            const double across = -drag.x * units;
            const double raised = drag.y * units;
            m_orthoEye = {m_orthoEye.x + right.x * across + up.x * raised,
                          m_orthoEye.y + right.y * across + up.y * raised,
                          m_orthoEye.z + right.z * across + up.z * raised};
            camera.Get<Double3>("translate").Set(m_orthoEye);
        }
        if (scrolled) {
            const float scale = float(std::pow(0.9, io.MouseWheel));
            m_orthoWidth *= scale;
            m_orthoHeight *= scale;
            camera.Get(Camera::kHorizontalAperture).Set(m_orthoWidth);
            camera.Get(Camera::kVerticalAperture).Set(m_orthoHeight);
        }
        return;
    }
    if (ImGui::IsItemActivated() || (scrolled && !active)) {
        m_orbiting = StartOrbit();
    }
    if (!m_orbiting) {
        return;
    }
    const ImVec2 drag = io.MouseDelta;
    const bool dragged = active && (drag.x != 0 || drag.y != 0);
    if (dragged) {
        const double step = m_orbit.distance * kPanSpeed;
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
            m_orbit.PanVertical(-drag.x * step, drag.y * step);
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right) || io.KeyShift) {
            m_orbit.Pan(-drag.x * step, drag.y * step);
        } else {
            m_orbit.Orbit(-drag.x * kOrbitSpeed, drag.y * kOrbitSpeed);
        }
    }
    if (scrolled) {
        m_orbit.Zoom(io.MouseWheel);
    }
    if (dragged || scrolled) {
        const View camera = m_scene->Select(m_camera);
        camera.Get<Double3>("translate").Set(m_orbit.Eye());
        camera.Get<Quatd>("orient").Set(m_orbit.Orient());
    }
}

std::optional<GizmoCamera> Viewport::GizmoView(ImVec2 min, ImVec2 size) const {
    if (!m_view) {
        return std::nullopt;
    }
    return GizmoCamera{.frame = m_view->frame,
                       .focal = m_view->focal,
                       .width = m_view->width,
                       .height = m_view->height,
                       .right = m_view->right,
                       .up = m_view->up,
                       .orthographic = m_view->orthographic,
                       .min = min,
                       .size = size};
}

std::optional<Float2> Viewport::WorldToScreen(const Double3& point) const {
    const std::optional<GizmoCamera> camera = GizmoView(m_imageMin, m_imageSize);
    const std::optional<ImVec2> at = camera ? ToScreen(*camera, point) : std::nullopt;
    if (!at) {
        return std::nullopt;
    }
    return Float2{at->x - m_rect.x, at->y - m_rect.y};
}

std::optional<OverlayRay> Viewport::ScreenRay(Float2 at) const {
    const std::optional<GizmoCamera> camera = GizmoView(m_imageMin, m_imageSize);
    if (!camera || m_imageSize.x <= 0 || m_imageSize.y <= 0) {
        return std::nullopt;
    }
    const GizmoRay ray = CameraRay(*camera, (at.x + m_rect.x - m_imageMin.x) / m_imageSize.x,
                                   (at.y + m_rect.y - m_imageMin.y) / m_imageSize.y);
    const double length =
        std::sqrt(ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y +
                  ray.direction.z * ray.direction.z);
    if (!(length > 0)) {
        return std::nullopt;
    }
    return OverlayRay{{float(ray.origin.x), float(ray.origin.y), float(ray.origin.z)},
                      {float(ray.direction.x / length), float(ray.direction.y / length),
                       float(ray.direction.z / length)}};
}

bool Viewport::SendGesture(OverlayGesture gesture, const OverlayHit& hit, ImVec2 mouse,
                           ImVec2 imageMin, ImVec2 imageSize) {
    if (hit.overlay.starts_with("viewer/")) {
        if (hit.prim == kNoPrim || hit.prim >= m_scene->PrimCount()) {
            return false;
        }
        if (gesture == OverlayGesture::Click) {
            m_selection.Set(m_scene->ByIndex(hit.prim).Path());
        }
        return gesture == OverlayGesture::Press;
    }
    if (!overlayEvents) {
        return false;
    }
    const ImGuiIO& io = ImGui::GetIO();
    OverlayEvent event{.gesture = gesture,
                       .viewport = m_desc.name,
                       .hit = hit,
                       .x = mouse.x - m_rect.x,
                       .y = mouse.y - m_rect.y,
                       .modifiers = {io.KeyCtrl, io.KeyShift, io.KeyAlt, io.KeySuper}};
    if (const std::optional<GizmoCamera> camera = GizmoView(imageMin, imageSize)) {
        const GizmoRay ray =
            CameraRay(*camera, (mouse.x - imageMin.x) / std::max(imageSize.x, 1.0f),
                      (mouse.y - imageMin.y) / std::max(imageSize.y, 1.0f));
        const double length =
            std::sqrt(ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y +
                      ray.direction.z * ray.direction.z);
        if (length > 0) {
            event.ray = {{float(ray.origin.x), float(ray.origin.y), float(ray.origin.z)},
                         {float(ray.direction.x / length), float(ray.direction.y / length),
                          float(ray.direction.z / length)}};
        }
    }
    return overlayEvents(event);
}

void Viewport::TrackGestures(ImVec2 imageMin, ImVec2 imageSize, bool pressed) {
    if (!m_renderer) {
        m_hoverPick.reset();
        m_hoverHit.reset();
        m_pressedHit.reset();
        return;
    }
    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    if (m_pressedHit) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (mouse.x != m_draggedMouse.x || mouse.y != m_draggedMouse.y) {
                m_draggedMouse = mouse;
                SendGesture(OverlayGesture::Drag, *m_pressedHit, mouse, imageMin, imageSize);
            }
            return;
        }
        const OverlayHit released = *m_pressedHit;
        m_pressedHit.reset();
        SendGesture(OverlayGesture::Release, released, mouse, imageMin, imageSize);
        const float dx = mouse.x - m_pressedMouse.x;
        const float dy = mouse.y - m_pressedMouse.y;
        if (dx * dx + dy * dy < kClickSlop * kClickSlop) {
            SendGesture(OverlayGesture::Click, released, mouse, imageMin, imageSize);
        }
        return;
    }
    const device::Tensor* shown = Shown();
    const bool inside = shown && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
                        mouse.x >= imageMin.x && mouse.y >= imageMin.y &&
                        mouse.x < imageMin.x + imageSize.x && mouse.y < imageMin.y + imageSize.y;
    if (m_hoverPick && m_hoverPick->Ready()) {
        const std::optional<OverlayHit> hit = inside ? m_hoverPick->Hit() : std::nullopt;
        m_hoverPick.reset();
        const auto same = [](const std::optional<OverlayHit>& a,
                             const std::optional<OverlayHit>& b) {
            return a.has_value() == b.has_value() &&
                   (!a || (a->overlay == b->overlay && a->batch == b->batch && a->item == b->item &&
                           a->id == b->id));
        };
        if (!same(hit, m_hoverHit)) {
            if (m_hoverHit) {
                SendGesture(OverlayGesture::Leave, *m_hoverHit, mouse, imageMin, imageSize);
            }
            if (hit) {
                SendGesture(OverlayGesture::Hover, *hit, mouse, imageMin, imageSize);
            }
        }
        m_hoverHit = hit;
    }
    if (!inside) {
        if (m_hoverHit) {
            SendGesture(OverlayGesture::Leave, *m_hoverHit, mouse, imageMin, imageSize);
            m_hoverHit.reset();
        }
        return;
    }
    if (!m_hoverPick) {
        const float width = float(shown->Shape()[1]);
        const float height = float(shown->Shape()[0]);
        const float scale = width / std::max(imageSize.x, 1.0f);
        m_hoverPick = m_renderer->GetOverlays()->Pick(
            m_product,
            {(mouse.x - imageMin.x) / std::max(imageSize.x, 1.0f) * width,
             (mouse.y - imageMin.y) / std::max(imageSize.y, 1.0f) * height},
            6 * scale * ImGui::GetStyle().FontScaleDpi);
    }
    if (pressed && m_hoverHit &&
        SendGesture(OverlayGesture::Press, *m_hoverHit, mouse, imageMin, imageSize)) {
        m_pressedHit = m_hoverHit;
        m_pressedMouse = mouse;
        m_draggedMouse = mouse;
        m_inputClaimed = true;
    }
}

void Viewport::StartPick(double u, double v, bool toggle) {
    if (!m_view) {
        return;
    }
    const CameraView& view = *m_view;
    const GizmoRay ray = CameraRay(*GizmoView({}, {1, 1}), u, v);
    const Double3 origin = ray.origin;
    const Double3 along = ray.direction;
    const double scale = std::sqrt(along.x * along.x + along.y * along.y + along.z * along.z);
    if (!(scale > 0)) {
        return;
    }
    std::vector<float> origins = {float(origin.x), float(origin.y), float(origin.z)};
    std::vector<float> directions = {float(along.x / scale), float(along.y / scale),
                                     float(along.z / scale)};
    int64_t shape[] = {1, 3};
    const auto import = [&](std::vector<float>& values) {
        DLTensor host{};
        host.data = values.data();
        host.device = {kDLCPU, 0};
        host.ndim = 2;
        host.dtype = {kDLFloat, 32, 1};
        host.shape = shape;
        return m_scene->GetDevice().Import(host);
    };
    Rays rays;
    rays.origins = import(origins);
    rays.directions = import(directions);
    rays.minDistance = float(view.clipping.x * scale);
    rays.maxDistance = float(view.clipping.y * scale);
    rays.world = view.world;
    m_caster->Cast(rays, m_hits, {.outputs = RayOutput::Prim, .label = "Viewer/Pick"});
    m_pick = m_hits.prim.ToHostAsync();
    m_pickToggles = toggle;
}

void Viewport::LandPick() {
    if (!m_pick || !m_pick->Ready()) {
        return;
    }
    uint32_t prim = kNoPrim;
    std::memcpy(&prim, m_pick->Bytes().data(), sizeof(prim));
    m_pick.reset();
    if (prim == kNoPrim) {
        if (!m_pickToggles) {
            m_selection.Clear();
        }
        return;
    }
    const std::string path = m_scene->ByIndex(prim).Path();
    if (m_pickToggles) {
        m_selection.Toggle(path);
    } else {
        m_selection.Set(path);
    }
}

void Viewport::LookThrough(const std::string& camera) {
    if (!m_caster || camera == m_camera) {
        return;
    }
    m_scene->Select(m_product).Get(RenderProduct::kCamera.Column()).SetTargets(0, {camera});
    m_structure = ~uint64_t(0);
    m_orbiting = false;
    m_view.reset();
    m_reads.clear();
    if (m_unframed.erase(camera) > 0) {
        m_framing = {m_scene->ReadBoundsAsync()};
    }
}

void Viewport::SetAov(const std::string& aov) {
    if (m_shownVar.empty()) {
        throw Error("Viewport \"{}\" shows its delegate's images, not AOVs", m_desc.name);
    }
    if (!FindAov(aov)) {
        throw Error("\"{}\" isn't an AOV the viewer shows", aov);
    }
    if (aov == m_aov) {
        return;
    }
    m_scene->Select(m_shownVar).Get(RenderVar::kSourceName).Set(m_scene->Token(aov));
    m_aov = aov;
    m_bounds.reset();
    m_range.reset();
}

std::vector<const AovInfo*> Viewport::Aovs() const {
    const std::optional<Prim> sensor = m_camera.empty() ? std::nullopt : m_scene->Find(m_camera);
    const bool radar = sensor && sensor->IsA(SchemaId::Radar);
    const bool returns = sensor && (radar || sensor->IsA(SchemaId::Lidar));
    std::vector<const AovInfo*> offered;
    for (const AovInfo& info : kAovs) {
        const bool shaded = info.kind == AovKind::Color || info.kind == AovKind::Hdr;
        const bool sensed = info.group == std::string_view("Sensor");
        if ((returns && shaded) || (!returns && sensed) ||
            (info.name == "RadialVelocity" && !radar) ||
            (returns && info.name == "DistanceToImagePlane")) {
            continue;
        }
        offered.push_back(&info);
    }
    return offered;
}

AovDisplay& Viewport::Display(std::string_view aov) {
    const std::string_view name = aov.empty() ? std::string_view(m_aov) : aov;
    if (const auto found = m_displays.find(name); found != m_displays.end()) {
        return found->second;
    }
    return m_displays.emplace(std::string(name), DefaultDisplay(name)).first->second;
}

void Viewport::ReadRange(const device::Tensor& bounds) {
    m_bounds = bounds.ToHostAsync();
}

void Viewport::SetWorlds(WorldDisplay display) {
    if (!m_session) {
        throw Error("Viewport \"{}\" shares its cameras with another viewport, which picks the "
                    "worlds they show",
                    m_desc.name);
    }
    m_session->display = std::move(display);
    ShowWorlds();
}

void Viewport::RefreshWorlds() {
    if (!m_session) {
        return;
    }
    bool recompose = false;
    if (m_scene->StructureVersion() != m_session->structure) {
        m_session->structure = m_scene->StructureVersion();
        m_session->worldPrims.clear();
        std::vector<Prim> stack;
        const View roots = m_scene->Select("/*");
        for (const uint32_t root : roots.Prims()) {
            const Prim prim = m_scene->ByIndex(root);
            if (!prim.IsSession()) {
                stack.push_back(prim);
            }
        }
        while (!stack.empty()) {
            const Prim prim = stack.back();
            stack.pop_back();
            if (prim.IsA(SchemaId::World)) {
                m_session->worldPrims.emplace_back(prim.Path(), prim.World());
            }
            for (const Prim& child : prim.Children()) {
                stack.push_back(child);
            }
        }
        std::ranges::sort(m_session->worldPrims, {}, &std::pair<std::string, uint32_t>::second);
        m_session->boxes.clear();
        m_session->origins.clear();
        m_session->requested = -1;
        recompose = true;
    }
    const double now = ImGui::GetTime();
    if ((m_session->boxes.empty() || m_session->origins.empty()) && !m_session->boxRead &&
        !m_session->originRead && !m_session->worldPrims.empty() &&
        (m_session->requested < 0 || now - m_session->requested > 0.5)) {
        std::vector<std::string> paths;
        for (const auto& [path, world] : m_session->worldPrims) {
            paths.push_back(path);
        }
        m_session->boxRead = m_scene->ReadAllBoundsAsync();
        m_session->originRead = m_scene->ReadWorldTransformsAsync(Span<const std::string>(paths));
        m_session->requested = now;
    }
    if (m_session->originRead && m_session->originRead->Ready()) {
        const Span<const std::byte> bytes = m_session->originRead->Bytes();
        std::vector<Double3> origins;
        for (size_t at = 0;
             at < m_session->worldPrims.size() && (at + 1) * sizeof(WorldTransform) <= bytes.size();
             ++at) {
            WorldTransform transform;
            std::memcpy(&transform, bytes.data() + at * sizeof(WorldTransform), sizeof(transform));
            const uint32_t world = m_session->worldPrims[at].second;
            origins.resize(std::max<size_t>(origins.size(), size_t(world) + 1), Double3{});
            origins[world] = {double(transform.high.x) + transform.low.x,
                              double(transform.high.y) + transform.low.y,
                              double(transform.high.z) + transform.low.z};
        }
        m_session->originRead.reset();
        if (!origins.empty()) {
            m_session->origins = std::move(origins);
            recompose = true;
        }
    }
    if (m_session->boxRead && m_session->boxRead->Ready()) {
        const Span<const std::byte> bytes = m_session->boxRead->Bytes();
        std::vector<Box> boxes(bytes.size() / sizeof(Box));
        std::memcpy(boxes.data(), bytes.data(), boxes.size() * sizeof(Box));
        m_session->boxRead.reset();
        if (std::ranges::any_of(boxes, [](const Box& box) { return box.lower.x <= box.upper.x; })) {
            m_session->boxes = std::move(boxes);
            recompose = true;
        }
    }
    if (m_session->display.pick == WorldPick::Selected &&
        m_selection.paths != m_session->selection) {
        m_session->selection = m_selection.paths;
        recompose = true;
    }
    if (recompose) {
        ShowWorlds();
    }
}

void Viewport::ShowWorlds() {
    std::vector<uint32_t> worlds;
    const WorldDisplay& display = m_session->display;
    switch (display.pick) {
    case WorldPick::Listed:
        for (const std::string& path : display.worlds) {
            if (const std::optional<Prim> prim = m_scene->Find(path);
                prim && prim->IsA(SchemaId::World)) {
                worlds.push_back(prim->World());
            }
        }
        break;
    case WorldPick::Selected:
        for (const std::string& path : m_selection.paths) {
            const std::optional<Prim> prim = m_scene->Find(path);
            if (prim && prim->World() != kGlobalWorld && !prim->IsSession()) {
                worlds.push_back(prim->World());
            }
        }
        break;
    default:
        for (const auto& [path, world] : m_session->worldPrims) {
            worlds.push_back(world);
        }
        if (display.pick == WorldPick::Random) {
            std::mt19937 random(display.seed);
            std::ranges::shuffle(worlds, random);
        }
        if (display.pick != WorldPick::All && display.count > 0 && worlds.size() > display.count) {
            worlds.resize(display.count);
        }
        break;
    }
    std::ranges::sort(worlds);
    worlds.erase(std::unique(worlds.begin(), worlds.end()), worlds.end());

    std::vector<Double3> offsets;
    if (display.layout == WorldLayout::Grid && worlds.size() > 1) {
        Double3 spacing = display.spacing.value_or(Double3{});
        if (!display.spacing) {
            double extent = 0;
            for (const uint32_t world : worlds) {
                if (size_t(world) + 1 < m_session->boxes.size()) {
                    const Box& box = m_session->boxes[size_t(world) + 1];
                    if (box.lower.x <= box.upper.x) {
                        extent = std::max({extent, double(box.upper.x) - box.lower.x,
                                           double(box.upper.y) - box.lower.y});
                    }
                }
            }
            const double step = extent > 0 ? std::ceil(extent * 1.5) : 10.0;
            spacing = {step, step, 0};
        }
        const auto side = size_t(std::ceil(std::sqrt(double(worlds.size()))));
        const size_t rows = (worlds.size() + side - 1) / side;
        for (size_t at = 0; at < worlds.size(); ++at) {
            const double column = double(at % side) - 0.5 * double(side - 1);
            const double row = double(at / side) - 0.5 * double(rows - 1);
            const Double3 origin =
                worlds[at] < m_session->origins.size() ? m_session->origins[worlds[at]] : Double3{};
            offsets.push_back({column * spacing.x - origin.x, row * spacing.y - origin.y, 0});
        }
    }
    m_session->shown = worlds;
    m_session->offsets = offsets;
    m_session->composed.Set({.worlds = std::move(worlds),
                             .offsets = std::move(offsets),
                             .global = display.global,
                             .stageLights = m_session->lighting == LightingMode::Stage});
}

void Viewport::SetLighting(LightingMode lighting) {
    if (!m_session) {
        throw Error("Viewport \"{}\" has no scene to light", m_desc.name);
    }
    const std::string lights = m_session->root + "/Lights";
    if (lighting != LightingMode::Stage && !m_scene->Find(lights)) {
        m_scene->Define<Scope>(lights);
        m_scene->Define<DistantLight>(
            lights + "/Camera",
            {.visibility = Visibility::Invisible, .visibleInPrimaryRay = false, .angle = 0});
        m_scene->Define<DistantLight>(lights + "/Default", {.visibility = Visibility::Invisible,
                                                            .orient = LookAlong({-0.35, 0.5, -1.0}),
                                                            .visibleInPrimaryRay = false});
    }
    m_session->lighting = lighting;
    m_session->aim.reset();
    for (const auto& [name, mode] : {std::pair{"/Camera", LightingMode::Camera},
                                     std::pair{"/Default", LightingMode::Default}}) {
        if (m_scene->Find(lights + name)) {
            m_scene->Select(lights + name)
                .Get(Imageable::kVisibility)
                .Set(lighting == mode ? Visibility::Inherited : Visibility::Invisible);
        }
    }
    ShowWorlds();
}

void Viewport::Inspect(const std::string& path) {
    if (m_inspect) {
        m_inspect(path);
    } else {
        m_selection.Set(path);
    }
}

}
