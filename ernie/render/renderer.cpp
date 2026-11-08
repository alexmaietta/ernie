// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/renderer_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ernie/device/commands.h"
#include "ernie/render/kernels.h"
#include "ernie/render/kernels/render/render.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr uint64_t PointCloud::* kPointFields[] = {
    &PointCloud::coordinates, &PointCloud::distance,      &PointCloud::intensity,
    &PointCloud::stableId,    &PointCloud::emitter,       &PointCloud::firing,
    &PointCloud::echo,        &PointCloud::radialVelocity};

constexpr const char* kRenderEntries[kRenderEntryCount] = {"RenderMinimalData",
                                                           "RenderMinimalDisplayColor",
                                                           "RenderMinimalTextureDiffuse",
                                                           "RenderMinimalFull",
                                                           "RenderRealTime",
                                                           "RenderRealTimeBands",
                                                           "RenderLidar",
                                                           "RenderRadar"};

constexpr size_t kHeldOutputsWarning = 8;

}

RendererImpl::RendererImpl(Ref<Scene> scene, const RendererDesc& desc)
    : m_scene(scene ? std::move(scene) : throw Error("A renderer of no scene")),
      m_maxPixelsPerLaunch(std::clamp<uint64_t>(desc.maxPixelsPerLaunch, 1, ~uint32_t(0))),
      m_overlays(desc.overlays ? desc.overlays : Overlays::Create(m_scene->GetDevice())) {
    m_renderProgram = m_scene->GetDevice().CreateProgram(
        {.modules = {"ernie.render.kernels.render.render"},
         .library = &kRenderKernels,
         .entries =
             [] {
                 std::vector<std::string> entries(std::begin(kRenderEntries),
                                                  std::end(kRenderEntries));
                 entries.emplace_back("RenderScatterPoints");
                 return entries;
             }(),
         .loadEntries = false,
         .label = "render",
         .traversal = SceneImpl::Get(*m_scene).GetBuilder().Format()});
    Log(LogLevel::Debug, "created a renderer: at most {} per launch",
        Counted(m_maxPixelsPerLaunch, "pixel"));
}

RenderOutput RendererImpl::Step(std::span<const std::string> products, double dt, Stream stream) {
    ERNIE_ZONE("Render/Step");
    const std::lock_guard stepping(m_stepMutex);
    device::Operation published = m_scene->Publish({.dt = dt}, stream);
    DropStaleProducts();
    StepPlan& plan = PlanProducts(products);
    const ProductTable& resolved = SceneImpl::Get(*m_scene).GetProducts();
    std::set<uint32_t> due;
    for (const uint32_t prim : plan.plan->products) {
        if (resolved.Find(prim)->due) {
            due.insert(prim);
        }
    }

    const SimulationClock& clock = SceneImpl::Get(*m_scene).GetClock();
    RenderOutput result;
    result.simulationStartTime = clock.start;
    result.simulationEndTime = clock.end;
    if (due.empty()) {
        result.m_products = plan.products;
        for (auto& [path, product] : result.m_products) {
            product.frames.clear();
        }
        return result;
    }

    if (!plan.plan->taxonomies.empty()) {
        m_scene->Require(plan.plan->taxonomies);
    }
    if (m_scene->Pending()) {
        published = m_scene->Publish({}, stream);
    }
    const Snapshot snapshot = m_scene->Acquire();
    if (!plan.plan->taxonomies.empty()) {
        UpdateMaps();
    }
    const OutputSet& set = AcquireSet(plan);
    result.m_products = plan.products;
    for (auto& [path, product] : result.m_products) {
        if (!due.contains(m_products.at(path).prim)) {
            product.frames.clear();
        }
        for (FrameOutput& frame : product.frames) {
            frame.startTime = result.simulationStartTime;
            frame.endTime = result.simulationEndTime;
        }
    }
    for (const OwnedVar& var : set.vars) {
        std::vector<FrameOutput>& frames = result.m_products.at(var.product).frames;
        if (frames.empty()) {
            continue;
        }
        RenderVarOutput& output = frames[0].renderVars.at(var.renderVar);
        if (var.channel.empty()) {
            output.tensor = var.tensor;
        } else {
            output.channels[var.channel] = var.tensor;
        }
    }
    for (const OwnedOverlay& overlay : set.overlays) {
        std::vector<FrameOutput>& frames = result.m_products.at(overlay.product).frames;
        if (!frames.empty()) {
            frames[0].overlay = overlay.tensor;
        }
    }
    result.m_done = plan.plan->captureCount > 0 ? Trace(set, due, snapshot) : published;
    m_scene->Release(snapshot);
    return result;
}

void RendererImpl::DropStaleProducts() {
    const ProductTable& resolved = SceneImpl::Get(*m_scene).GetProducts();
    std::vector<std::string> stale;
    for (const auto& [path, product] : m_products) {
        const ResolvedProduct* current = resolved.Find(product.prim);
        if (current == nullptr || current->version != product.version) {
            stale.push_back(path);
        }
    }
    if (stale.empty()) {
        return;
    }
    Log(LogLevel::Debug, "{} changed, so {} prepared again", Counted(stale.size(), "product"),
        stale.size() == 1 ? "it's" : "they're");
    for (const std::string& path : stale) {
        m_products.erase(path);
        std::erase_if(m_plans, [&](const StepPlan& plan) { return plan.products.contains(path); });
    }
    m_plan = nullptr;
}

RendererImpl::StepPlan& RendererImpl::PlanProducts(std::span<const std::string> products) {
    std::shared_ptr<const ProductPlan> planned =
        SceneImpl::Get(*m_scene).GetProducts().Plan(products);
    const auto cached = std::ranges::find_if(
        m_plans, [&](const StepPlan& plan) { return plan.plan->id == planned->id; });
    if (cached != m_plans.end()) {
        m_plans.splice(m_plans.begin(), m_plans, cached);
        m_plan = &m_plans.front();
        return *m_plan;
    }
    ERNIE_ZONE("Render/PlanProducts");

    StepPlan plan{planned};
    for (size_t index = 0; index < planned->paths.size(); ++index) {
        const std::string& path = planned->paths[index];
        const auto [prepared, added] = m_products.try_emplace(
            path, *SceneImpl::Get(*m_scene).GetProducts().Find(planned->products[index]));
        const ResolvedProduct& product = prepared->second;
        FrameOutput frame;
        std::string sources;
        for (const ResolvedVar& var : product.vars) {
            const std::string_view source = kRenderVarSourceTokens[uint32_t(var.source)];
            frame.renderVars[var.path] = {var.path, std::string(source), {}};
            sources += std::format("{}{}", sources.empty() ? "" : ", ", source);
        }
        if (added) {
            const ProductRecord& record = product.record;
            Log(LogLevel::Debug, "prepared {}: {} at {} x {} writing {} with {}{}", path,
                Counted(product.sensors.size(), "sensor"), record.width, record.height,
                sources.empty() ? "nothing" : sources, kRenderEntries[uint32_t(product.entry)],
                record.frameRate > 0 ? std::format(", capturing at {} Hz", record.frameRate) : "");
        }
        plan.products[path] = ProductOutput{path, {std::move(frame)}};
    }
    uint64_t launches = 0;
    for (const uint64_t pixels : planned->pixelCounts) {
        launches += (pixels + m_maxPixelsPerLaunch - 1) / m_maxPixelsPerLaunch;
    }
    Log(LogLevel::Debug, "planned {}: {}, {} in {}", Counted(plan.products.size(), "product"),
        Counted(planned->captureCount, "capture"), Counted(planned->pixelCount, "pixel"),
        Counted(launches, "launch"));
    m_plans.push_front(std::move(plan));
    m_plan = &m_plans.front();
    if (m_plans.size() > kCachedPlans) {
        m_plans.pop_back();
        const auto unused = [&](const auto& entry) {
            return std::ranges::none_of(
                m_plans, [&](const StepPlan& kept) { return kept.products.contains(entry.first); });
        };
        const size_t dropped = std::erase_if(m_products, unused);
        Log(LogLevel::Debug, "dropped the least recently stepped plan and {} only it used",
            Counted(dropped, "product"));
    }
    return *m_plan;
}

void RendererImpl::UpdateMaps() {
    SemanticTable& semantics = SceneImpl::Get(*m_scene).GetSemantics();
    for (auto& [path, planned] : m_plan->products) {
        const ResolvedProduct& product = m_products.at(path);
        for (const ResolvedVar& var : product.vars) {
            if (!var.semantic || var.source == RenderVarSource::PointCloud) {
                continue;
            }
            const SemanticMap map = semantics.Map(var.source, product.taxonomy);
            for (FrameOutput& frame : planned.frames) {
                RenderVarOutput& output = frame.renderVars.at(var.path);
                output.idMap = map.map;
                if (var.shape.empty()) {
                    output.tensor = map.packed;
                }
            }
        }
    }
}

RendererImpl::OutputSet& RendererImpl::AcquireSet(StepPlan& plan) {
    bool outlined = false;
    {
        const OverlaysImpl& overlays = OverlaysImpl::Get(*m_overlays);
        const std::lock_guard lock(overlays.mutex);
        outlined = std::ranges::any_of(overlays.layers,
                                       [](const auto& entry) { return entry.second->Outlined(); });
    }
    for (OutputSet& set : plan.sets) {
        if (set.outlined == outlined &&
            std::ranges::all_of(
                set.vars,
                [](const OwnedVar& var) { return var.tensor.GetBuffer().RefCount() == 1; }) &&
            std::ranges::all_of(set.overlays, [](const OwnedOverlay& overlay) {
                return overlay.tensor.GetBuffer().RefCount() == 1;
            })) {
            return set;
        }
    }

    ERNIE_ZONE("Render/AcquireSet");
    device::Device& device = m_scene->GetDevice();
    const device::BufferUsage usage =
        device.SharedDevice() ? device::BufferUsage::Interop : device::BufferUsage::None;
    OutputSet set;
    set.outlined = outlined;
    std::map<uint32_t, std::array<uint64_t, kRenderVarSourceCount>> addresses;
    std::map<uint32_t, uint64_t> primIds;
    for (const auto& [path, output] : plan.products) {
        const ResolvedProduct& product = m_products.at(path);
        std::array<uint64_t, kRenderVarSourceCount>& slots = addresses[product.prim];
        for (const ResolvedVar& var : product.vars) {
            if (var.source == RenderVarSource::PointCloud) {
                uint64_t& clouds = slots[uint32_t(var.source)];
                clouds = AcquireCloud(set, path, product, var, clouds);
                continue;
            }
            if (var.shape.empty()) {
                continue;
            }
            const uint64_t bytes =
                std::accumulate(var.shape.begin(), var.shape.end(),
                                uint64_t(ElementBytes(var.type)), std::multiplies<>());
            const Ref<device::Buffer> buffer =
                device.CreateBuffer({.bytes = bytes, .usage = usage, .label = var.path});
            device::Tensor tensor(buffer, 0, var.type, var.shape);
            slots[uint32_t(var.source)] = tensor.Address();
            set.vars.push_back({path, var.path, std::move(tensor), {}});
        }
        if (product.overlays.empty()) {
            continue;
        }
        const std::vector<int64_t> shape = {int64_t(product.sensors.size()), product.record.height,
                                            product.record.width, 4};
        const uint64_t pixels = uint64_t(shape[0]) * shape[1] * shape[2];
        OwnedOverlay overlay{path, device::Tensor(device.CreateBuffer({.bytes = pixels * 4,
                                                                       .usage = usage,
                                                                       .label = path + " overlay"}),
                                                  0, device::DType::UInt8, shape)};
        uint64_t& depth = slots[uint32_t(RenderVarSource::DistanceToImagePlane)];
        if (depth == 0) {
            overlay.ownDepth =
                device.CreateBuffer({.bytes = pixels * 4, .label = path + " overlay depth"});
            depth = overlay.ownDepth->Address();
        }
        overlay.depth = depth;
        if (outlined) {
            overlay.primIds =
                device.CreateBuffer({.bytes = pixels * 4, .label = path + " overlay prim ids"});
            primIds[product.prim] = overlay.primIds->Address();
        }
        set.overlays.push_back(std::move(overlay));
    }
    for (uint32_t entry = 0; entry < kRenderEntryCount; ++entry) {
        const std::vector<PlannedCapture>& planned = plan.plan->captures[entry];
        if (planned.empty()) {
            continue;
        }
        std::vector<Capture> captures;
        for (const PlannedCapture& each : planned) {
            Capture& capture = captures.emplace_back(Capture{.firstPixel = each.firstPixel,
                                                             .product = each.product,
                                                             .sensor = each.sensor,
                                                             .layer = each.layer,
                                                             .width = each.width,
                                                             .height = each.height,
                                                             .taxonomy = each.taxonomy,
                                                             .missDistance = each.missDistance,
                                                             .returns = each.returns});
            const std::array<uint64_t, kRenderVarSourceCount>& slots = addresses[each.product];
            std::copy(slots.begin(), slots.end(), std::begin(capture.vars));
            const auto found = primIds.find(each.product);
            capture.primIds = found != primIds.end() ? found->second : 0;
        }
        set.captureBuffers[entry] = device.CreateBuffer({.label = "render captures"}, captures);
        set.captures[entry] = std::move(captures);
    }
    plan.sets.push_back(std::move(set));
    if (plan.sets.size() == kHeldOutputsWarning) {
        Log(LogLevel::Warning,
            "{} render outputs are held at once, so each step allocates more; drop outputs "
            "once they're used",
            kHeldOutputsWarning);
    }
    Log(LogLevel::Debug, "allocated render output set {}", plan.sets.size());
    return plan.sets.back();
}

uint64_t RendererImpl::AcquireCloud(OutputSet& set, const std::string& path,
                                    const ResolvedProduct& product, const ResolvedVar& var,
                                    uint64_t next) {
    device::Device& device = m_scene->GetDevice();
    const device::BufferUsage usage =
        device.SharedDevice() ? device::BufferUsage::Interop : device::BufferUsage::None;
    const auto sensors = int64_t(product.sensors.size());
    const uint32_t returns = std::max(product.record.returns, 1u);
    const uint32_t perSensor = product.record.width * product.record.height * returns;
    const auto add = [&](std::string channel, device::DType type, std::vector<int64_t> shape) {
        const uint64_t bytes = std::accumulate(shape.begin(), shape.end(),
                                               uint64_t(ElementBytes(type)), std::multiplies<>());
        device::Tensor tensor(
            device.CreateBuffer(
                {.bytes = bytes, .usage = usage, .label = var.path + " " + channel}),
            0, type, std::move(shape));
        const uint64_t address = tensor.Address();
        set.vars.push_back({path, var.path, std::move(tensor), std::move(channel)});
        return address;
    };

    OwnedCloud owned{product.prim, uint32_t(sensors) * perSensor};
    owned.records = device.CreateBuffer(
        {.bytes = uint64_t(owned.slots) * sizeof(PointRecord), .label = path + " point records"});
    owned.flags =
        device.CreateBuffer({.bytes = uint64_t(owned.slots) * 4, .label = path + " point flags"});
    owned.offsets =
        device.CreateBuffer({.bytes = uint64_t(owned.slots) * 4, .label = path + " point offsets"});
    PointCloud cloud{};
    cloud.records = owned.records->Address();
    cloud.flags = owned.flags->Address();
    cloud.offsets = owned.offsets->Address();
    cloud.counts = add("Counts", device::DType::UInt32, {sensors});
    for (const PointChannel channel : var.channels) {
        const PointChannelInfo& info = kPointChannels[uint32_t(channel)];
        std::vector<int64_t> shape = {sensors, perSensor};
        if (info.lanes > 1) {
            shape.push_back(info.lanes);
        }
        cloud.*kPointFields[uint32_t(channel)] =
            add(std::string(info.name), info.type, std::move(shape));
    }
    cloud.slotsPerSensor = perSensor;
    cloud.width = product.record.width;
    cloud.returns = returns;
    cloud.sensors = uint32_t(sensors);
    cloud.world = var.frame == FrameOfReference::World ? 1 : 0;
    cloud.next = next;
    owned.cloud = device.CreateBuffer({.label = path + " point cloud"}, std::vector{cloud});
    const uint64_t address = owned.cloud->Address();
    set.clouds.push_back(std::move(owned));
    return address;
}

device::Operation RendererImpl::Trace(const OutputSet& set, const std::set<uint32_t>& due,
                                      const Snapshot& snapshot) {
    static const profiler::ZoneId traceZone = profiler::ZoneId::Named("Render/Trace");
    RenderParams params{};
    params.tables.address = snapshot.tables;

    device::Queue& queue = m_scene->GetDevice().GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.BeginZone(traceZone);
    encoder.Concurrent(true);
    for (uint32_t entry = 0; entry < kRenderEntryCount; ++entry) {
        std::vector<Capture> captures;
        uint64_t pixelCount = 0;
        for (Capture capture : set.captures[entry]) {
            if (due.contains(capture.product)) {
                capture.firstPixel = pixelCount;
                pixelCount += uint64_t(capture.width) * capture.height;
                captures.push_back(capture);
            }
        }
        if (captures.empty()) {
            continue;
        }
        params.captures.address =
            captures.size() == set.captures[entry].size()
                ? set.captureBuffers[entry]->Address()
                : queue.Upload(captures.data(), captures.size() * sizeof(Capture)).Address();
        params.captureCount = uint32_t(captures.size());
        for (uint64_t first = 0; first < pixelCount; first += m_maxPixelsPerLaunch) {
            params.firstPixel = first;
            params.pixelCount = uint32_t(std::min(m_maxPixelsPerLaunch, pixelCount - first));
            encoder.Dispatch(m_renderProgram->Entry(kRenderEntries[entry]), params,
                             {.x = device::GroupCount(params.pixelCount, kRenderGroup)});
        }
    }
    encoder.Concurrent(false);
    for (const OwnedCloud& cloud : set.clouds) {
        if (!due.contains(cloud.product)) {
            continue;
        }
        if (cloud.slots > m_scanCapacity) {
            m_scanCapacity = cloud.slots;
            m_scan = std::make_unique<algorithms::Scan>(m_scene->GetDevice(), m_scanCapacity);
        }
        m_scan->ExclusiveSum(encoder, *cloud.flags, *cloud.offsets, cloud.slots);
        ScatterParams scatter{};
        scatter.cloud.address = cloud.cloud->Address();
        scatter.tables.address = snapshot.tables;
        scatter.slotCount = cloud.slots;
        encoder.Dispatch(m_renderProgram->Entry("RenderScatterPoints"), scatter,
                         {.x = device::GroupCount(cloud.slots, kRenderGroup)});
    }
    encoder.EndZone();
    auto overlays =
        std::make_shared<std::vector<OverlayContents>>(RecordOverlays(encoder, set, due, snapshot));
    const device::Operation done = queue.Submit(encoder.Finish());
    if (!overlays->empty()) {
        queue.Retire(std::move(overlays), done);
    }
    for (const std::shared_ptr<OverlayPickState>& pick : m_servedPicks) {
        {
            const std::lock_guard lock(pick->mutex);
            pick->recorded = true;
            pick->done = done;
        }
        pick->served.notify_all();
    }
    m_servedPicks.clear();
    return done;
}

std::vector<OverlayContents> RendererImpl::RecordOverlays(device::CommandEncoder& encoder,
                                                          const OutputSet& set,
                                                          const std::set<uint32_t>& due,
                                                          const Snapshot& snapshot) {
    std::vector<std::pair<std::string, OverlayContents>> named;
    {
        OverlaysImpl& overlays = OverlaysImpl::Get(*m_overlays);
        const std::lock_guard lock(overlays.mutex);
        named.assign(overlays.layers.begin(), overlays.layers.end());
        std::vector<std::shared_ptr<OverlayPickState>> waiting;
        for (std::shared_ptr<OverlayPickState>& pick : overlays.picks) {
            const auto product = m_products.find(pick->product);
            const bool served = product != m_products.end() && due.contains(product->second.prim);
            (served ? m_servedPicks : waiting).push_back(std::move(pick));
        }
        overlays.picks = std::move(waiting);
    }
    if (set.overlays.empty()) {
        return {};
    }
    const auto matches = [](const std::vector<std::string>& patterns, std::string_view name) {
        return std::ranges::any_of(patterns, [&](const std::string& pattern) {
            return pattern.ends_with('*')
                       ? name.starts_with(std::string_view(pattern).substr(0, pattern.size() - 1))
                       : name == pattern;
        });
    };

    std::vector<OverlayContents> used;
    std::vector<std::string> names;
    std::vector<uint32_t> slots(named.size(), ~0u);
    std::vector<OverlayTarget> targets;
    std::vector<std::pair<std::string, uint32_t>> targetKeys;
    for (const OwnedOverlay& overlay : set.overlays) {
        const ResolvedProduct& product = m_products.at(overlay.product);
        if (!due.contains(product.prim)) {
            continue;
        }
        encoder.Fill(overlay.tensor.GetBuffer());
        const std::vector<std::string>& patterns = product.overlays;
        uint64_t mask = 0;
        for (size_t at = 0; at < named.size(); ++at) {
            if (!matches(patterns, named[at].first)) {
                continue;
            }
            if (slots[at] == ~0u) {
                slots[at] = uint32_t(used.size());
                used.push_back(named[at].second);
                names.push_back(named[at].first);
            }
            if (slots[at] < kMostOverlays) {
                mask |= uint64_t(1) << slots[at];
            }
        }
        for (uint32_t layer = 0; layer < product.sensors.size(); ++layer) {
            targets.push_back({.camera = product.sensors[layer],
                               .layer = layer,
                               .width = product.record.width,
                               .height = product.record.height,
                               .missDistance = product.record.missDistance,
                               .overlay = overlay.tensor.Address(),
                               .depth = overlay.depth,
                               .overlays = mask,
                               .primIds = overlay.primIds ? overlay.primIds->Address() : 0});
            targetKeys.emplace_back(overlay.product, layer);
        }
    }
    if (targets.empty() || used.empty()) {
        return {};
    }
    std::vector<OverlayCameraSource> cameraImages;
    for (const OwnedVar& var : set.vars) {
        const ResolvedProduct& product = m_products.at(var.product);
        if (!due.contains(product.prim) ||
            !IsA(SchemaId(product.record.sensorSchema), SchemaId::Camera)) {
            continue;
        }
        const auto source = std::ranges::find_if(product.vars, [&](const ResolvedVar& candidate) {
            return candidate.path == var.renderVar && candidate.source == RenderVarSource::LdrColor;
        });
        if (source == product.vars.end()) {
            continue;
        }
        for (uint32_t layer = 0; layer < product.sensors.size(); ++layer) {
            cameraImages.push_back({product.sensors[layer], var.tensor.Select(layer)});
        }
    }
    std::vector<OverlayPickRequest> picks;
    std::vector<std::shared_ptr<OverlayPickState>> answered;
    for (const std::shared_ptr<OverlayPickState>& pick : m_servedPicks) {
        const auto target = std::ranges::find(targetKeys, std::pair(pick->product, pick->layer));
        if (target == targetKeys.end()) {
            continue;
        }
        pick->readback = m_scene->GetDevice().CreateBuffer({.bytes = sizeof(OverlayPickResult),
                                                            .memory = device::Memory::Readback,
                                                            .label = "overlay pick"});
        picks.push_back(
            {uint32_t(target - targetKeys.begin()), pick->pixel, pick->radius, pick->readback});
        answered.push_back(pick);
    }
    if (!m_overlayPass) {
        m_overlayPass.emplace(m_scene->GetDevice());
    }
    m_overlayPass->Record(encoder, snapshot, m_scene->WorldCount(), targets, used, cameraImages,
                          picks);
    for (const std::shared_ptr<OverlayPickState>& pick : answered) {
        pick->names = names;
        pick->sources = m_overlayPass->BatchSources();
    }
    return used;
}

Renderer::Renderer(Ref<Scene> scene, const RendererDesc& desc)
    : m_impl(std::make_unique<RendererImpl>(std::move(scene), desc)) {}

Renderer::~Renderer() = default;
Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

RenderOutput Renderer::Step(Span<const std::string> products, double dt, Stream stream) {
    return m_impl->Step(products, dt, stream);
}

const Ref<Overlays>& Renderer::GetOverlays() const {
    return m_impl->m_overlays;
}

}
