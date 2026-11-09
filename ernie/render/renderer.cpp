// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/renderer_impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ernie/device/commands.h"
#include "ernie/material/codegen.h"
#include "ernie/render/kernels.h"
#include "ernie/render/kernels/render/render.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/semantics.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr uint64_t PointCloud::* kPointFields[] = {
    &PointCloud::coordinates, &PointCloud::distance,       &PointCloud::intensity,
    &PointCloud::stableId,    &PointCloud::emitter,        &PointCloud::firing,
    &PointCloud::echo,        &PointCloud::radialVelocity, &PointCloud::instanceId};

constexpr size_t kHeldOutputsWarning = 8;

}

RendererImpl::RendererImpl(Ref<Scene> scene, const RendererDesc& desc)
    : m_scene(scene ? std::move(scene) : throw Error("A renderer of no scene")),
      m_maxPixelsPerLaunch(std::clamp<uint64_t>(desc.maxPixelsPerLaunch, 1, ~uint32_t(0))),
      m_overlays(desc.overlays ? desc.overlays : Overlays::Create(m_scene->GetDevice())) {
    const DefinitionSetRef& definitions = SceneImpl::Get(*m_scene).GetMaterials().Definitions();
    m_renderProgram = CreateRenderProgram(*definitions);
    m_renderHash = definitions->Hash();
    Log(LogLevel::Debug, "created a renderer: at most {} per launch",
        Counted(m_maxPixelsPerLaunch, "pixel"));
}

Ref<device::Program> RendererImpl::CreateRenderProgram(const DefinitionSet& definitions) const {
    const auto sources = DispatchSources(definitions);
    return m_scene->GetDevice().CreateProgram(
        {.modules = {"ernie.render.kernels.render.render"},
         .library = &kRenderKernels,
         .sources = {sources.begin(), sources.end()},
         .entries =
             [] {
                 std::vector<std::string> entries(std::begin(kRenderEntryNames),
                                                  std::end(kRenderEntryNames));
                 entries.emplace_back("RenderScatterPoints");
                 return entries;
             }(),
         .loadEntries = false,
         .label = "render",
         .traversal = SceneImpl::Get(*m_scene).GetBuilder().Format()});
}

device::Kernel& RendererImpl::RenderEntry(std::string_view name) {
    if (!m_usedEntries.contains(name)) {
        m_usedEntries.emplace(name);
    }
    if (m_pendingProgram) {
        device::Kernel& pending = m_pendingProgram->Entry(name);
        if (SceneImpl::Get(*m_scene).WaitsForKernels() || pending.Ready()) {
            try {
                pending.Wait();
                return pending;
            } catch (const std::exception&) {
            }
        }
    }
    return m_renderProgram->Entry(name);
}

void RendererImpl::UpdateRenderProgram() {
    const DefinitionSetRef& definitions = SceneImpl::Get(*m_scene).GetMaterials().Definitions();
    const uint64_t hash = definitions->Hash();
    if (hash == m_renderHash || hash == m_failedHash) {
        m_pendingProgram = nullptr;
        m_pendingDefinitions = nullptr;
    } else if (!m_pendingDefinitions || m_pendingDefinitions->Hash() != hash) {
        m_pendingProgram = CreateRenderProgram(*definitions);
        m_pendingDefinitions = definitions;
        m_pendingStart = std::chrono::steady_clock::now();
        Log(LogLevel::Debug,
            "relinking the render kernels for {}, keeping the current ones "
            "until the new ones are ready",
            Counted(definitions->Definitions().size(), "generated material definition"));
    }
    if (!m_pendingProgram) {
        return;
    }
    ++m_kernels;
    if (m_usedEntries.empty()) {
        return;
    }
    const bool wait = SceneImpl::Get(*m_scene).WaitsForKernels();
    try {
        for (const std::string& name : m_usedEntries) {
            device::Kernel& kernel = m_pendingProgram->Entry(name);
            if (wait) {
                kernel.Wait();
            } else if (!kernel.Ready()) {
                return;
            } else {
                kernel.Wait();
            }
        }
    } catch (const std::exception& error) {
        std::string message = error.what();
        for (const DefinitionModule& definition : m_pendingDefinitions->Definitions()) {
            for (const auto& [generated, origin] : definition.origins) {
                for (size_t at = message.find(generated); at != std::string::npos;
                     at = message.find(generated, at + origin.size())) {
                    const bool dotted = at >= 2 && message.compare(at - 2, 2, "./") == 0;
                    at -= dotted ? 2 : 0;
                    message.replace(at, generated.size() + (dotted ? 2 : 0), origin);
                }
            }
        }
        Log(LogLevel::Error,
            "relinking the render kernels failed, so materials keep their previous look: {}",
            message);
        m_failedHash = hash;
        m_pendingProgram = nullptr;
        m_pendingDefinitions = nullptr;
        return;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - m_pendingStart).count();
    Log(LogLevel::Info, "relinked the render kernels for {} in {:.1f} s",
        Counted(m_pendingDefinitions->Definitions().size(), "generated material definition"),
        seconds);
    m_renderProgram = std::move(m_pendingProgram);
    m_renderHash = hash;
    m_pendingDefinitions = nullptr;
}

RenderOutput RendererImpl::Step(std::span<const std::string> products, double dt, Stream stream) {
    ERNIE_ZONE("Render/Step");
    const std::lock_guard stepping(m_stepMutex);
    device::Operation published = m_scene->Publish({.dt = dt}, stream);
    StepPlan& plan = PlanProducts(products);
    const ProductPlan& planned = *plan.plan;
    const CaptureSchedule& captures = SceneImpl::Get(*m_scene).GetCaptures();
    std::set<uint32_t> due;
    for (const std::shared_ptr<const ResolvedProduct>& product : planned.products) {
        if (captures.Due(product->prim)) {
            due.insert(product->prim);
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
    UpdateRenderProgram();
    OutputSet& set = AcquireSet(plan);
    const std::vector<PipelineRun> pipelineRuns = PrepareRuns(planned, due);
    std::set<uint32_t> ran;
    for (const PipelineRun& run : pipelineRuns) {
        ran.insert(run.products.begin(), run.products.end());
    }
    SemanticTable& semantics = SceneImpl::Get(*m_scene).GetSemantics();
    for (size_t index = 0; index < planned.products.size(); ++index) {
        const ResolvedProduct& product = *planned.products[index];
        if (!due.contains(product.prim)) {
            continue;
        }
        FrameOutput& frame =
            result.m_products.at(planned.paths[index])
                .frames.emplace_back(FrameOutput{.startTime = result.simulationStartTime,
                                                 .endTime = result.simulationEndTime});
        for (const ResolvedVar& var : product.vars) {
            if (!var.output.empty() && !ran.contains(uint32_t(index))) {
                continue;
            }
            RenderVarOutput& output = frame.renderVars[var.path];
            output.renderVarPath = var.path;
            output.sourceName = var.tap ? std::string(kTapSource)
                                : var.output.empty()
                                    ? std::string(kRenderVarSourceTokens[uint32_t(var.source)])
                                    : var.output;
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
        const auto found = frames[0].renderVars.find(var.renderVar);
        if (found == frames[0].renderVars.end()) {
            continue;
        }
        RenderVarOutput& output = found->second;
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
    const std::pair<uint64_t, uint64_t> runs = {snapshot.Version(), m_kernels};
    std::set<uint32_t> stale;
    for (const uint32_t prim : due) {
        if (const auto traced = set.traced.find(prim);
            traced == set.traced.end() || traced->second != runs) {
            stale.insert(prim);
            set.traced[prim] = runs;
        }
    }
    result.m_done = Trace(planned, set, due, stale, snapshot, pipelineRuns);
    m_scene->Release(snapshot);
    return result;
}

RendererImpl::StepPlan& RendererImpl::PlanProducts(std::span<const std::string> products) {
    std::shared_ptr<const ProductPlan> planned = SceneImpl::Get(*m_scene).GetPlans().Plan(products);
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
    const PipelineTable& pipelines = SceneImpl::Get(*m_scene).GetPipelines();
    for (uint32_t index = 0; index < planned.products.size(); ++index) {
        const std::string& path = planned.paths[index];
        const ResolvedProduct& product = *planned.products[index];
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
            if (var.output.empty()) {
                slots[uint32_t(var.source)] = tensor.Address();
            }
            set.vars.push_back({index, var.path, std::move(tensor), {}, var.output});
        }
        const std::shared_ptr<const PipelineInstance> instance =
            product.pipeline == kNoPrim ? nullptr : pipelines.Find(product.pipeline);
        if (instance && std::ranges::any_of(instance->resolved.values, [](const auto& value) {
                return value.type == pipeline::SignalType::ElectronRate;
            })) {
            uint64_t& rates = slots[uint32_t(RenderVarSource::ElectronRate)];
            if (rates == 0) {
                const uint64_t bytes = uint64_t(product.sensors.size()) * product.width *
                                       product.height * std::max(product.channels, 1u) *
                                       sizeof(float);
                rates = set.rateBuffers
                            .emplace_back(
                                device.CreateBuffer({.bytes = bytes, .label = path + " rates"}))
                            ->Address();
            }
            set.rates[index] = rates;
        }
        if (product.overlays.empty()) {
            continue;
        }
        const std::vector<int64_t> shape = {int64_t(product.sensors.size()), product.height,
                                            product.width, 4};
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
            Capture& capture = captures.emplace_back(Capture{.plan = each});
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
    const uint32_t returns = std::max(product.returns, 1u);
    const uint32_t perSensor = product.width * product.height * returns;
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

    OwnedCloud owned{product.prim, uint32_t(sensors) * perSensor, index, var.output,
                     var.frame == FrameOfReference::World};
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
    cloud.width = product.width;
    cloud.returns = returns;
    cloud.sensors = uint32_t(sensors);
    cloud.world = var.frame == FrameOfReference::World ? 1 : 0;
    cloud.next = next;
    owned.cloud = device.CreateBuffer({.label = path + " point cloud"}, std::vector{cloud});
    const uint64_t address = owned.cloud->Address();
    set.clouds.push_back(std::move(owned));
    return address;
}

device::Operation RendererImpl::Trace(const ProductPlan& plan, OutputSet& set,
                                      const std::set<uint32_t>& due,
                                      const std::set<uint32_t>& stale, const Snapshot& snapshot,
                                      std::span<const PipelineRun> runs) {
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
            if (stale.contains(capture.plan.product)) {
                capture.plan.firstPixel = pixelCount;
                pixelCount += uint64_t(capture.plan.width) * capture.plan.height;
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
            encoder.Dispatch(RenderEntry(kRenderEntryNames[entry]), params,
                             {.x = device::GroupCount(params.pixelCount, kRenderGroup)});
        }
    }
    encoder.Concurrent(false);
    std::vector<PipelineRun> traced;
    for (const PipelineRun& run : runs) {
        const bool returns =
            std::ranges::any_of(run.instance->resolved.values, [](const pipeline::Value& value) {
                return value.type == pipeline::SignalType::Returns;
            });
        if (!returns || stale.contains(plan.products[run.products.front()]->prim)) {
            traced.push_back(run);
        }
    }
    RecordRuns(encoder, plan, set, traced, snapshot);
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
        encoder.Dispatch(RenderEntry("RenderScatterPoints"), scatter,
                         {.x = device::GroupCount(cloud.slots, kRenderGroup)});
    }
    encoder.EndZone();
    std::vector<std::string> stepped;
    std::vector<OverlayOutput> views;
    for (size_t index = 0; index < plan.products.size(); ++index) {
        if (due.contains(plan.products[index]->prim)) {
            stepped.push_back(plan.paths[index]);
        }
    }
    for (const OwnedOverlay& overlay : set.overlays) {
        const ResolvedProduct& product = *plan.products[overlay.product];
        if (!due.contains(product.prim)) {
            continue;
        }
        OverlayOutput& view = views.emplace_back(
            OverlayOutput{plan.paths[overlay.product], &product.overlays, &overlay.tensor, {}});
        for (uint32_t layer = 0; layer < product.sensors.size(); ++layer) {
            view.targets.push_back({.camera = product.sensors[layer],
                                    .product = product.prim,
                                    .layer = layer,
                                    .width = product.width,
                                    .height = product.height,
                                    .missDistance = product.missDistance,
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
            const ResolvedProduct& product = *plan.products[var.product];
            const auto source =
                std::ranges::find_if(product.vars, [&](const ResolvedVar& candidate) {
                    return candidate.path == var.renderVar &&
                           candidate.source == RenderVarSource::LdrColor;
                });
            if (!due.contains(product.prim) || source == product.vars.end() ||
                !IsA(product.sensor, SchemaId::Camera)) {
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

std::vector<RendererImpl::PipelineRun> RendererImpl::PrepareRuns(const ProductPlan& plan,
                                                                 const std::set<uint32_t>& due) {
    const PipelineTable& table = SceneImpl::Get(*m_scene).GetPipelines();
    const bool wait = SceneImpl::Get(*m_scene).WaitsForKernels();
    std::shared_ptr<const pipeline::PipelineSet> definitions;
    std::vector<PipelineRun> runs;
    for (const PlannedPipeline& planned : plan.pipelines) {
        PipelineRun run{.planned = &planned, .instance = table.Find(planned.pipeline)};
        for (const uint32_t product : planned.products) {
            if (due.contains(plan.products[product]->prim)) {
                run.products.push_back(product);
            }
        }
        if (run.products.empty() || !run.instance || !run.instance->error.empty()) {
            continue;
        }
        const uint32_t id = run.instance->definition;
        if (m_failedPipelines.contains(id)) {
            continue;
        }
        try {
            std::shared_ptr<const CompiledPipeline>& compiled = m_pipelinePrograms[id];
            if (!compiled) {
                definitions = definitions ? definitions : table.Definitions();
                compiled =
                    CompilePipeline(m_scene->GetDevice(), definitions->Find(id), *table.Registry());
            }
            run.compiled = compiled;
            for (const std::string& entry : compiled->generated.entries) {
                device::Kernel& kernel = compiled->program->Entry(entry);
                if (!wait && !kernel.Ready()) {
                    run.compiled = nullptr;
                    break;
                }
                kernel.Wait();
            }
        } catch (const std::exception& error) {
            Log(LogLevel::Error, "compiling {} failed, so its outputs are left out: {}",
                run.instance->path, error.what());
            m_failedPipelines.insert(id);
            m_pipelinePrograms.erase(id);
            continue;
        }
        if (!run.compiled) {
            if (m_pendingPipelines.insert(id).second) {
                Log(LogLevel::Info, "{} is compiling, so its outputs are left out until it's ready",
                    run.instance->path);
            }
            continue;
        }
        runs.push_back(std::move(run));
    }
    return runs;
}

void RendererImpl::RecordRuns(device::CommandEncoder& encoder, const ProductPlan& plan,
                              OutputSet& set, std::span<const PipelineRun> runs,
                              const Snapshot& snapshot) {
    if (runs.empty()) {
        return;
    }
    device::Device& device = m_scene->GetDevice();
    device::Queue& queue = device.GetQueue();
    const std::shared_ptr<const pipeline::NodeRegistry> nodes =
        SceneImpl::Get(*m_scene).GetPipelines().Registry();
    const pipeline::NodeRegistry& registry = *nodes;
    const Prims& prims = SceneImpl::Get(*m_scene).GetStructure().GetPrims();
    const auto layer = [&](uint32_t product, uint32_t sensor) {
        const std::vector<uint32_t>& sensors = plan.products[product]->sensors;
        return uint64_t(std::ranges::find(sensors, sensor) - sensors.begin());
    };
    const auto scratchBytes = [](const pipeline::SignalLayout& layout, uint64_t elements) {
        const uint64_t bytes = (layout.perCapture ? 1 : elements) * layout.channels * sizeof(float);
        return (bytes + pipeline::kArenaAlignment - 1) / pipeline::kArenaAlignment *
               pipeline::kArenaAlignment;
    };
    const auto recordBytes = [](uint64_t elements) {
        const uint64_t bytes = elements * (sizeof(PointRecord) + sizeof(uint32_t));
        return (bytes + pipeline::kArenaAlignment - 1) / pipeline::kArenaAlignment *
               pipeline::kArenaAlignment;
    };
    const auto cloudOf = [&](const PipelineRun& run, uint32_t product, std::string_view output) {
        const OwnedCloud* found = nullptr;
        for (const OwnedCloud& cloud : set.clouds) {
            if (cloud.index == product && !cloud.output.empty() &&
                (output.empty() || cloud.output == output) &&
                std::ranges::find(run.products, cloud.index) != run.products.end()) {
                found = &cloud;
                break;
            }
        }
        return found;
    };
    const auto read = [&](const PipelineRun& run, const std::string& name) {
        return std::ranges::any_of(set.vars, [&](const OwnedVar& var) {
            return var.output == name &&
                   std::ranges::find(run.products, var.product) != run.products.end();
        });
    };

    struct Placed {
        pipeline::ArenaLayout arena;
        uint64_t base = 0;
        uint64_t scratch = 0;
    };

    std::vector<Placed> placed;
    uint64_t arenaBytes = 0;
    for (const PipelineRun& run : runs) {
        const ResolvedProduct& product = *plan.products[run.products.front()];
        const uint64_t elements =
            uint64_t(product.width) * product.height * std::max(product.returns, 1u);
        Placed& each = placed.emplace_back(Placed{
            pipeline::LayoutArena(run.compiled->definition->segments, product.layouts, elements),
            arenaBytes});
        each.scratch = each.base + each.arena.bytes;
        arenaBytes = each.scratch;
        for (const pipeline::ResolvedNode& node : run.instance->resolved.nodes) {
            if (registry[node.def].kind == pipeline::NodeKind::Source &&
                run.instance->resolved.values[node.outputs.front()].type ==
                    pipeline::SignalType::Returns) {
                arenaBytes += recordBytes(elements);
            }
        }
        for (const pipeline::ResolvedOutput& output : run.instance->resolved.outputs) {
            const pipeline::SignalLayout& layout = product.layouts[output.value];
            if (run.instance->resolved.values[output.value].type ==
                pipeline::SignalType::PointCloud) {
                arenaBytes += recordBytes(elements);
            } else if (layout.perCapture || !read(run, output.name)) {
                arenaBytes += scratchBytes(layout, elements);
            }
        }
    }
    if (!set.arena || set.arena->Bytes() < arenaBytes) {
        set.arena =
            device.CreateBuffer({.bytes = std::max<uint64_t>(arenaBytes, pipeline::kArenaAlignment),
                                 .label = "pipeline arena"});
    }

    struct Batch {
        const CompiledPipeline* compiled = nullptr;
        std::vector<PipelineCapture> captures;
        std::vector<PipelineValue> values;
        uint64_t elements = 0;
    };

    std::map<uint32_t, Batch> batches;
    std::vector<uint32_t> words;
    std::vector<std::pair<device::BufferRange, device::BufferRange>> copies;
    std::vector<std::pair<device::BufferRange, device::BufferRange>> sources;
    for (size_t index = 0; index < runs.size(); ++index) {
        const PipelineRun& run = runs[index];
        const PlannedPipeline& planned = *run.planned;
        const uint32_t source = run.products.front();
        const ResolvedProduct& product = *plan.products[source];
        const pipeline::PipelineDefinition& definition = *run.compiled->definition;
        const pipeline::GeneratedPipeline& generated = run.compiled->generated;
        const pipeline::ResolvedPipeline& resolved = run.instance->resolved;
        const uint32_t returns = std::max(product.returns, 1u);
        const uint64_t elements = uint64_t(product.width) * product.height * returns;
        PipelineState& state = m_pipelineStates[std::tuple(planned.pipeline, planned.sensor,
                                                           product.width, product.height)];
        if (!state.state || state.definition != definition.id) {
            if (state.state) {
                Log(LogLevel::Debug, "{} restarts its state on {} with a new definition",
                    run.instance->path, m_scene->ByIndex(planned.sensor).Path());
            }
            std::vector<float> initial = pipeline::InitialState(definition, registry);
            initial.resize(std::max<size_t>(initial.size(), 1));
            state = {definition.id, 0, device.CreateBuffer({.label = "pipeline state"}, initial)};
        }

        Batch& batch = batches[definition.id];
        batch.compiled = run.compiled.get();
        const auto firstValue = uint32_t(batch.values.size());
        const auto storedCount = uint32_t(definition.segments.stored.size());
        uint64_t scratch = placed[index].scratch;
        for (uint32_t slot = 0; slot < generated.slots.size(); ++slot) {
            const uint32_t value = generated.slots[slot];
            const pipeline::SignalLayout& layout = product.layouts[value];
            PipelineValue entry{0, layout.channels, layout.mosaic ? kSignalMosaic : 0u, 0};
            const pipeline::SignalType type = resolved.values[value].type;
            const auto bindRecords = [&](const OwnedCloud& cloud, uint64_t first) {
                entry.address = cloud.records->Address() + first * sizeof(PointRecord);
                entry.returned = cloud.flags->Address() + first * sizeof(uint32_t);
                entry.flags |= kSignalRecords | (cloud.world ? kSignalWorld : 0u);
            };
            if (slot < storedCount && type == pipeline::SignalType::Returns &&
                registry[resolved.nodes[resolved.values[value].node].def].kind ==
                    pipeline::NodeKind::Source) {
                const OwnedCloud* cloud = cloudOf(run, source, {});
                if (cloud == nullptr) {
                    throw Error("{} reads lidar returns, but no point cloud of {} reads it",
                                run.instance->path, m_scene->ByIndex(product.prim).Path());
                }
                const uint64_t first = layer(source, planned.sensor) * elements;
                const device::BufferRange records =
                    device::BufferRange(*set.arena).Slice(scratch, elements * sizeof(PointRecord));
                const device::BufferRange flags =
                    device::BufferRange(*set.arena)
                        .Slice(scratch + elements * sizeof(PointRecord),
                               elements * sizeof(uint32_t));
                scratch += recordBytes(elements);
                sources.emplace_back(
                    device::BufferRange(*cloud->records)
                        .Slice(first * sizeof(PointRecord), elements * sizeof(PointRecord)),
                    records);
                sources.emplace_back(
                    device::BufferRange(*cloud->flags)
                        .Slice(first * sizeof(uint32_t), elements * sizeof(uint32_t)),
                    flags);
                entry.address = records.Address();
                entry.returned = flags.Address();
                entry.flags |= kSignalRecords | (cloud->world ? kSignalWorld : 0u);
            } else if (slot >= storedCount && type == pipeline::SignalType::PointCloud) {
                const std::string& name = resolved.outputs[slot - storedCount].name;
                const OwnedCloud* bound = nullptr;
                uint64_t boundFirst = 0;
                for (const uint32_t reader : run.products) {
                    for (const OwnedCloud& cloud : set.clouds) {
                        if (cloud.index != reader || cloud.output != name) {
                            continue;
                        }
                        const uint64_t first = layer(reader, planned.sensor) * elements;
                        if (bound == nullptr) {
                            bindRecords(cloud, first);
                            bound = &cloud;
                            boundFirst = first;
                            continue;
                        }
                        const auto slice = [&](const Ref<device::Buffer>& buffer, uint64_t at,
                                               uint64_t stride) {
                            return device::BufferRange(*buffer).Slice(at * stride,
                                                                      elements * stride);
                        };
                        copies.emplace_back(slice(bound->records, boundFirst, sizeof(PointRecord)),
                                            slice(cloud.records, first, sizeof(PointRecord)));
                        copies.emplace_back(slice(bound->flags, boundFirst, sizeof(uint32_t)),
                                            slice(cloud.flags, first, sizeof(uint32_t)));
                    }
                }
                if (!bound) {
                    entry.address = set.arena->Address() + scratch;
                    entry.returned = entry.address + elements * sizeof(PointRecord);
                    entry.flags |= kSignalRecords;
                    scratch += recordBytes(elements);
                }
            } else if (slot < storedCount &&
                       registry[resolved.nodes[resolved.values[value].node].def].kind ==
                           pipeline::NodeKind::Source) {
                const uint32_t stride = std::max(product.channels, 1u);
                entry.address = set.rates.at(source) +
                                layer(source, planned.sensor) * elements * stride * sizeof(float);
                if (layout.mosaic) {
                    entry.flags |= kSignalSampled | (stride << kSignalStrideShift);
                }
            } else if (slot < storedCount) {
                entry.address =
                    set.arena->Address() + placed[index].base + placed[index].arena.offsets[value];
                entry.flags |= layout.element == pipeline::ElementType::F16 ? kSignalHalf : 0u;
            } else {
                const std::string& name = resolved.outputs[slot - storedCount].name;
                const uint64_t bytes =
                    (layout.perCapture ? pipeline::kStatWords : elements * layout.channels) *
                    sizeof(float);
                std::optional<device::BufferRange> written;
                if (layout.perCapture || !read(run, name)) {
                    written = device::BufferRange(*set.arena).Slice(scratch, bytes);
                    scratch += scratchBytes(layout, elements);
                }
                for (const OwnedVar& var : set.vars) {
                    if (var.output != name ||
                        std::ranges::find(run.products, var.product) == run.products.end()) {
                        continue;
                    }
                    const device::BufferRange target =
                        device::BufferRange(var.tensor.GetBuffer())
                            .Slice(var.tensor.Offset() + layer(var.product, planned.sensor) * bytes,
                                   bytes);
                    if (written) {
                        copies.emplace_back(*written, target);
                    } else {
                        written = target;
                    }
                }
                entry.address = written->Address();
            }
            batch.values.push_back(entry);
        }

        const uint32_t detector = prims.FindRow(planned.sensor, SchemaId::DetectorAPI);
        const uint32_t camera = prims.FindRow(planned.sensor, SchemaId::Camera);
        batch.captures.push_back(
            {.firstElement = batch.elements,
             .words = {words.size() * sizeof(uint32_t)},
             .sensor = planned.sensor,
             .detector = detector == kNoRow ? 0 : detector,
             .width = product.width * returns,
             .height = product.height,
             .capture = state.captures++,
             .identity = HashPath(m_scene->ByIndex(planned.sensor).Path()).words[0],
             .firstValue = firstValue,
             .camera = camera == kNoRow ? 0 : camera,
             .state = {state.state->Address()}});
        words.insert(words.end(), run.instance->words.begin(), run.instance->words.end());
        words.push_back(0);
        batch.elements += elements;
    }

    const uint64_t wordAddress =
        queue.Upload(words.data(), words.size() * sizeof(uint32_t)).Address();
    encoder.Barrier();
    for (const auto& [from, to] : sources) {
        encoder.Copy(from, to);
    }
    if (!sources.empty()) {
        encoder.Barrier();
    }
    for (auto& [id, batch] : batches) {
        for (PipelineCapture& capture : batch.captures) {
            capture.words.address += wordAddress;
        }
        const PipelineParams params{
            .tables = {snapshot.tables},
            .captures = {queue
                             .Upload(batch.captures.data(),
                                     batch.captures.size() * sizeof(PipelineCapture))
                             .Address()},
            .values = {queue
                           .Upload(batch.values.data(), batch.values.size() * sizeof(PipelineValue))
                           .Address()},
            .elementCount = batch.elements,
            .captureCount = uint32_t(batch.captures.size())};
        RecordPipeline(encoder, *batch.compiled, params, uint32_t(batch.captures.size()),
                       batch.elements);
    }
    for (const auto& [from, to] : copies) {
        encoder.Copy(from, to);
    }
    if (!copies.empty()) {
        encoder.Barrier();
    }
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
