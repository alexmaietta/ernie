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
    StepPlan& plan = PlanProducts(products);
    const ProductPlan& planned = *plan.plan;
    const ProductTable& table = SceneImpl::Get(*m_scene).GetProducts();
    std::set<uint32_t> due;
    for (const ResolvedProduct& product : planned.products) {
        if (table.Find(product.prim)->due) {
            due.insert(product.prim);
        }
    }

    const SimulationClock& clock = SceneImpl::Get(*m_scene).GetClock();
    RenderOutput result;
    result.simulationStartTime = clock.start;
    result.simulationEndTime = clock.end;
    for (const std::string& path : planned.paths) {
        result.m_products[path] = ProductOutput{path, {}};
    }
    if (due.empty()) {
        return result;
    }

    if (!planned.taxonomies.empty()) {
        m_scene->Require(planned.taxonomies);
    }
    if (m_scene->Pending()) {
        published = m_scene->Publish({}, stream);
    }
    const Snapshot snapshot = m_scene->Acquire();
    OutputSet& set = AcquireSet(plan);
    SemanticTable& semantics = SceneImpl::Get(*m_scene).GetSemantics();
    for (size_t index = 0; index < planned.products.size(); ++index) {
        const ResolvedProduct& product = planned.products[index];
        if (!due.contains(product.prim)) {
            continue;
        }
        FrameOutput& frame =
            result.m_products.at(planned.paths[index])
                .frames.emplace_back(FrameOutput{.startTime = result.simulationStartTime,
                                                 .endTime = result.simulationEndTime});
        for (const ResolvedVar& var : product.vars) {
            RenderVarOutput& output = frame.renderVars[var.path];
            output.renderVarPath = var.path;
            output.sourceName = kRenderVarSourceTokens[uint32_t(var.source)];
            if (var.semantic && var.source != RenderVarSource::PointCloud) {
                const SemanticMap map = semantics.Map(var.source, product.taxonomy);
                output.idMap = map.map;
                if (var.shape.empty()) {
                    output.tensor = map.packed;
                }
            }
        }
    }
    for (const OwnedVar& var : set.vars) {
        std::vector<FrameOutput>& frames = result.m_products.at(planned.paths[var.product]).frames;
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
        std::vector<FrameOutput>& frames =
            result.m_products.at(planned.paths[overlay.product]).frames;
        if (!frames.empty()) {
            frames[0].overlay = overlay.tensor;
        }
    }
    const uint64_t runs = SceneImpl::Get(*m_scene).GetPublisher().TableRuns();
    std::set<uint32_t> stale;
    for (const uint32_t prim : due) {
        if (const auto traced = set.traced.find(prim);
            traced == set.traced.end() || traced->second != runs) {
            stale.insert(prim);
            set.traced[prim] = runs;
        }
    }
    result.m_done = Trace(planned, set, due, stale, snapshot);
    m_scene->Release(snapshot);
    return result;
}

RendererImpl::StepPlan& RendererImpl::PlanProducts(std::span<const std::string> products) {
    std::shared_ptr<const ProductPlan> planned =
        SceneImpl::Get(*m_scene).GetProducts().Plan(products);
    std::erase_if(m_plans, [&](const StepPlan& plan) {
        return plan.plan->id != planned->id && plan.plan->paths == planned->paths;
    });
    const auto cached = std::ranges::find_if(
        m_plans, [&](const StepPlan& plan) { return plan.plan->id == planned->id; });
    if (cached != m_plans.end()) {
        m_plans.splice(m_plans.begin(), m_plans, cached);
        return m_plans.front();
    }
    uint64_t launches = 0;
    for (const uint64_t pixels : planned->pixelCounts) {
        launches += (pixels + m_maxPixelsPerLaunch - 1) / m_maxPixelsPerLaunch;
    }
    Log(LogLevel::Debug, "planned {}: {}, {} in {}", Counted(planned->paths.size(), "product"),
        Counted(planned->captureCount, "capture"), Counted(planned->pixelCount, "pixel"),
        Counted(launches, "launch"));
    m_plans.push_front({std::move(planned)});
    if (m_plans.size() > kCachedPlans) {
        m_plans.pop_back();
    }
    return m_plans.front();
}

RendererImpl::OutputSet& RendererImpl::AcquireSet(StepPlan& plan) {
    const bool outlined = OverlaysImpl::Get(*m_overlays).Outlined();
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
    const ProductPlan& planned = *plan.plan;
    for (uint32_t index = 0; index < planned.products.size(); ++index) {
        const std::string& path = planned.paths[index];
        const ResolvedProduct& product = planned.products[index];
        std::array<uint64_t, kRenderVarSourceCount>& slots = addresses[product.prim];
        for (const ResolvedVar& var : product.vars) {
            if (var.source == RenderVarSource::PointCloud) {
                uint64_t& clouds = slots[uint32_t(var.source)];
                clouds = AcquireCloud(set, index, path, product, var, clouds);
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
            set.vars.push_back({index, var.path, std::move(tensor), {}});
        }
        if (product.overlays.empty()) {
            continue;
        }
        const std::vector<int64_t> shape = {int64_t(product.sensors.size()), product.record.height,
                                            product.record.width, 4};
        const uint64_t pixels = uint64_t(shape[0]) * shape[1] * shape[2];
        OwnedOverlay overlay{
            index,
            device::Tensor(device.CreateBuffer(
                               {.bytes = pixels * 4, .usage = usage, .label = path + " overlay"}),
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
        if (planned.captures[entry].empty()) {
            continue;
        }
        std::vector<Capture> captures;
        for (const PlannedCapture& each : planned.captures[entry]) {
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

uint64_t RendererImpl::AcquireCloud(OutputSet& set, uint32_t index, const std::string& path,
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
        set.vars.push_back({index, var.path, std::move(tensor), std::move(channel)});
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

device::Operation RendererImpl::Trace(const ProductPlan& plan, const OutputSet& set,
                                      const std::set<uint32_t>& due,
                                      const std::set<uint32_t>& stale, const Snapshot& snapshot) {
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
            if (stale.contains(capture.product)) {
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
        if (!stale.contains(cloud.product)) {
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
    std::vector<std::string> stepped;
    std::vector<OverlayOutput> views;
    for (size_t index = 0; index < plan.products.size(); ++index) {
        if (due.contains(plan.products[index].prim)) {
            stepped.push_back(plan.paths[index]);
        }
    }
    for (const OwnedOverlay& overlay : set.overlays) {
        const ResolvedProduct& product = plan.products[overlay.product];
        if (!due.contains(product.prim)) {
            continue;
        }
        OverlayOutput& view = views.emplace_back(
            OverlayOutput{plan.paths[overlay.product], &product.overlays, &overlay.tensor, {}});
        for (uint32_t layer = 0; layer < product.sensors.size(); ++layer) {
            view.targets.push_back({.camera = product.sensors[layer],
                                    .layer = layer,
                                    .width = product.record.width,
                                    .height = product.record.height,
                                    .missDistance = product.record.missDistance,
                                    .overlay = overlay.tensor.Address(),
                                    .depth = overlay.depth,
                                    .primIds = overlay.primIds ? overlay.primIds->Address() : 0});
        }
    }
    OverlaysImpl& overlays = OverlaysImpl::Get(*m_overlays);
    const std::vector<std::shared_ptr<OverlayPickState>> picks = overlays.TakePicks(stepped);
    std::vector<OverlayContents> drawn;
    if (!views.empty()) {
        std::vector<OverlayCameraSource> cameraImages;
        for (const OwnedVar& var : set.vars) {
            const ResolvedProduct& product = plan.products[var.product];
            const auto source =
                std::ranges::find_if(product.vars, [&](const ResolvedVar& candidate) {
                    return candidate.path == var.renderVar &&
                           candidate.source == RenderVarSource::LdrColor;
                });
            if (!due.contains(product.prim) || source == product.vars.end() ||
                !IsA(SchemaId(product.record.sensorSchema), SchemaId::Camera)) {
                continue;
            }
            for (uint32_t layer = 0; layer < product.sensors.size(); ++layer) {
                cameraImages.push_back({product.sensors[layer], var.tensor.Select(layer)});
            }
        }
        if (!m_overlayPass) {
            m_overlayPass.emplace(m_scene->GetDevice());
        }
        drawn = m_overlayPass->Draw(encoder, snapshot, m_scene->WorldCount(), overlays, views,
                                    cameraImages, picks);
    }
    const device::Operation done = queue.Submit(encoder.Finish());
    if (!drawn.empty()) {
        queue.Retire(std::make_shared<std::vector<OverlayContents>>(std::move(drawn)), done);
    }
    OverlayPass::Serve(picks, done);
    return done;
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
