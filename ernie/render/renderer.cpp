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
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

struct RenderVarFormat {
    device::DType type;
    uint32_t channels;
};

RenderVarFormat SourceFormat(RenderVarSource source) {
    switch (source) {
    case RenderVarSource::LdrColor:
        return {device::DType::UInt8, 4};
    case RenderVarSource::HdrColor:
        return {device::DType::Float32, 4};
    case RenderVarSource::DistanceToCamera:
    case RenderVarSource::DistanceToImagePlane:
    case RenderVarSource::Intensity:
    case RenderVarSource::RadialVelocity:
        return {device::DType::Float32, 1};
    case RenderVarSource::NonStableInstanceSegmentation:
        return {device::DType::UInt32, 1};
    case RenderVarSource::Normal:
        return {device::DType::Float32, 3};
    case RenderVarSource::DiffuseAlbedo:
        return {device::DType::UInt8, 4};
    case RenderVarSource::SemanticSegmentation:
        return {device::DType::UInt32, 1};
    case RenderVarSource::SemanticInstanceSegmentation:
    case RenderVarSource::StableIdSegmentation:
    case RenderVarSource::InstancerStableId:
        return {device::DType::UInt32, 4};
    case RenderVarSource::InstanceId:
        return {device::DType::Int64, 1};
    case RenderVarSource::SemanticIdMap:
    case RenderVarSource::StableIdMap:
        return {device::DType::UInt8, 1};
    case RenderVarSource::PointCloud:
        break;
    }
    throw Error("No format for render var source {}", uint32_t(source));
}

bool PerPixel(RenderVarSource source) {
    return source != RenderVarSource::SemanticIdMap && source != RenderVarSource::StableIdMap &&
           source != RenderVarSource::PointCloud;
}

struct PointChannel {
    std::string_view name;
    device::DType type;
    int64_t lanes;
    uint64_t PointCloud::* address;
};

constexpr PointChannel kPointChannels[] = {
    {"Coordinates", device::DType::Float32, 3, &PointCloud::coordinates},
    {"Distance", device::DType::Float32, 1, &PointCloud::distance},
    {"Intensity", device::DType::Float32, 1, &PointCloud::intensity},
    {"StableId", device::DType::UInt32, 4, &PointCloud::stableId},
    {"Emitter", device::DType::UInt32, 1, &PointCloud::emitter},
    {"Firing", device::DType::UInt32, 1, &PointCloud::firing},
    {"Echo", device::DType::UInt32, 1, &PointCloud::echo},
    {"RadialVelocity", device::DType::Float32, 1, &PointCloud::radialVelocity}};

const PointChannel& FindPointChannel(std::string_view name) {
    for (const PointChannel& channel : kPointChannels) {
        if (channel.name == name) {
            return channel;
        }
    }
    throw Error("A PointCloud has no channel {}", name);
}

bool Semantic(RenderVarSource source) {
    return source == RenderVarSource::NonStableInstanceSegmentation ||
           source == RenderVarSource::SemanticSegmentation ||
           source == RenderVarSource::SemanticIdMap ||
           source == RenderVarSource::SemanticInstanceSegmentation ||
           source == RenderVarSource::StableIdSegmentation ||
           source == RenderVarSource::StableIdMap;
}

struct SensorShape {
    Uint2 resolution;
    uint32_t returns = 0;

    bool operator==(const SensorShape&) const = default;
};

SensorShape LidarShape(const Prim& lidar, Uint2) {
    const SensorShape shape{
        {lidar.Get(Lidar::kFiringsPerScan), uint32_t(lidar.Get(Lidar::kEmitterElevations).size())},
        lidar.Get(Lidar::kMaxReturns)};
    if (shape.returns == 0 || shape.returns > kMostLidarReturns) {
        throw Error("{} reports {} returns per beam, and lidars report 1 to {}", lidar.Path(),
                    shape.returns, kMostLidarReturns);
    }
    if (shape.resolution.y == 0) {
        throw Error("{} has no emitters", lidar.Path());
    }
    return shape;
}

constexpr ColumnRef kLidarInputs[] = {kLidarFiringsPerScan, kLidarEmitterElevations,
                                      kLidarMaxReturns};
constexpr ColumnRef kRadarInputs[] = {kRadarRays};

struct SensorKind {
    SchemaId schema;
    std::optional<RenderEntry> entry;
    bool shades;
    bool returns;
    bool measuresVelocity;
    SensorShape (*shape)(const Prim& sensor, Uint2 resolution);
    std::span<const ColumnRef> inputs;
};

const SensorKind kSensorKinds[] = {
    {SchemaId::Camera,
     std::nullopt,
     true,
     false,
     false,
     [](const Prim&, Uint2 resolution) { return SensorShape{resolution}; },
     {}},
    {SchemaId::Lidar, RenderEntry::Lidar, false, true, false, LidarShape, kLidarInputs},
    {SchemaId::Radar, RenderEntry::Radar, false, true, true,
     [](const Prim& radar, Uint2) { return SensorShape{radar.Get(Radar::kRays)}; }, kRadarInputs},
};

const SensorKind& FindSensorKind(SchemaId schema) {
    for (const SensorKind& kind : kSensorKinds) {
        if (IsA(schema, kind.schema)) {
            return kind;
        }
    }
    throw Error("A {} isn't a sensor the renderer renders", Info(schema).name);
}

bool Outputs(const SensorKind& kind, RenderVarSource source) {
    switch (source) {
    case RenderVarSource::DistanceToCamera:
    case RenderVarSource::Normal:
        return true;
    case RenderVarSource::Intensity:
    case RenderVarSource::PointCloud:
        return kind.returns;
    case RenderVarSource::RadialVelocity:
        return kind.measuresVelocity;
    default:
        return kind.shades || Semantic(source);
    }
}

IdEntries MapEntries(const Semantics& semantics, RenderVarSource source, Token taxonomy) {
    switch (source) {
    case RenderVarSource::SemanticSegmentation:
    case RenderVarSource::SemanticIdMap:
        return semantics.SemanticIdEntries(taxonomy);
    case RenderVarSource::NonStableInstanceSegmentation:
        return semantics.InstanceIdEntries(taxonomy);
    case RenderVarSource::SemanticInstanceSegmentation:
        return semantics.SemanticInstanceIdEntries(taxonomy);
    default:
        return semantics.StableIdEntries();
    }
}

bool Shaded(std::span<const RendererImpl::ProductVar> vars) {
    return std::ranges::any_of(vars, [](const RendererImpl::ProductVar& var) {
        return var.source == RenderVarSource::LdrColor || var.source == RenderVarSource::HdrColor ||
               var.source == RenderVarSource::DiffuseAlbedo;
    });
}

RenderEntry SelectEntry(std::span<const RendererImpl::ProductVar> vars, RenderMode renderMode,
                        MinimalMode minimalMode, SpectralMode spectralMode) {
    if (!Shaded(vars)) {
        return RenderEntry::MinimalData;
    }
    if (renderMode == RenderMode::RealTime) {
        return spectralMode == SpectralMode::Bands ? RenderEntry::RealTimeBands
                                                   : RenderEntry::RealTime;
    }
    switch (minimalMode) {
    case MinimalMode::DisplayColor:
        return RenderEntry::MinimalDisplayColor;
    case MinimalMode::TextureDiffuse:
        return RenderEntry::MinimalTextureDiffuse;
    case MinimalMode::Full:
        return RenderEntry::MinimalFull;
    default:
        return RenderEntry::MinimalData;
    }
}

constexpr ColumnRef kProductInputs[] = {
    kRenderProductCamera,      kRenderProductResolution,       kRenderProductOrderedVars,
    kRenderVarSourceName,      kRenderProductSemanticTaxonomy, kRenderProductMissDistance,
    kRenderProductRenderMode,  kRenderProductSpectralMode,     kMinimalAPIMode,
    kSensorAPISensorFrameRate, kSensorAPISensorStartTime,      kRenderVarChannels,
    kRenderVarFrameOfReference};

constexpr const char* kRenderEntries[kRenderEntryCount] = {"RenderMinimalData",
                                                           "RenderMinimalDisplayColor",
                                                           "RenderMinimalTextureDiffuse",
                                                           "RenderMinimalFull",
                                                           "RenderRealTime",
                                                           "RenderRealTimeBands",
                                                           "RenderLidar",
                                                           "RenderRadar"};

constexpr size_t kHeldOutputsWarning = 8;

constexpr double kCaptureTolerance = 1e-6;

std::string Counted(uint64_t count, std::string_view noun) {
    return fmt::format("{} {}{}", count, noun, count == 1 ? "" : "s");
}

}

RendererImpl::RendererImpl(Ref<Scene> scene, const RendererDesc& desc)
    : m_scene(scene ? std::move(scene) : throw Error("A renderer of no scene")),
      m_maxPixelsPerLaunch(std::clamp<uint64_t>(desc.maxPixelsPerLaunch, 1, ~uint32_t(0))) {
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
    if (ProductsChanged()) {
        ClearProducts();
    }
    std::map<std::string, std::vector<std::string>, std::less<>> shown;
    {
        const std::lock_guard lock(m_overlayMutex);
        for (const std::string& path : products) {
            if (const auto found = m_shownOverlays.find(path); found != m_shownOverlays.end()) {
                shown.insert(*found);
            }
        }
    }
    StepPlan& plan = PlanProducts(products);
    for (const auto& [path, names] : shown) {
        if (m_products.at(path).sensor != SchemaId::Camera) {
            throw Error("{} renders no camera, and only cameras show overlays", path);
        }
    }
    if (plan.shownOverlays != shown) {
        plan.sets.clear();
        plan.shownOverlays = std::move(shown);
    }
    std::set<uint32_t> due;
    for (const auto& [path, planned] : plan.products) {
        const Product& product = m_products.at(path);
        if (Due(path, product, dt)) {
            due.insert(product.prim);
        }
    }

    RenderOutput result;
    result.simulationStartTime = m_time;
    m_time += dt;
    result.simulationEndTime = m_time;
    if (due.empty()) {
        result.m_products = plan.products;
        for (auto& [path, product] : result.m_products) {
            product.frames.clear();
        }
        return result;
    }

    if (!plan.taxonomies.empty()) {
        m_scene->Require(plan.taxonomies);
    }
    const device::Operation published = m_scene->Publish({}, stream);
    const Snapshot snapshot = m_scene->Acquire();
    if (!plan.taxonomies.empty()) {
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
    result.m_done = plan.captureCount > 0 ? Trace(set, due, snapshot) : published;
    m_scene->Release(snapshot);
    return result;
}

void RendererImpl::Reset(double time) {
    const std::lock_guard stepping(m_stepMutex);
    Log(LogLevel::Debug, "simulation time reset from {} to {}", m_time, time);
    m_time = time;
    m_schedules.clear();
}

bool RendererImpl::Due(const std::string& path, const Product& product, double dt) {
    if (product.frameRate <= 0 || dt == 0) {
        return true;
    }
    const double rate = product.frameRate;
    Schedule& schedule = m_schedules[path];
    if (schedule.frameRate != product.frameRate || schedule.startTime != product.startTime) {
        const double first = std::ceil((m_time - product.startTime) * rate - kCaptureTolerance);
        schedule = {product.frameRate, product.startTime, std::max<int64_t>(0, int64_t(first))};
    }
    const double reached = (m_time + dt - product.startTime) * rate + kCaptureTolerance;
    if (double(schedule.next) > reached) {
        return false;
    }
    const auto last = int64_t(std::floor(reached));
    if (last > schedule.next) {
        Log(LogLevel::Debug, "{} skipped {} in a step of {} s", path,
            Counted(uint64_t(last - schedule.next), "capture"), dt);
    }
    schedule.next = last + 1;
    return true;
}

bool RendererImpl::ProductsChanged() {
    SceneImpl& impl = SceneImpl::Get(*m_scene);
    std::vector<uint64_t> writes;
    for (const ColumnRef column : kProductInputs) {
        writes.push_back(impl.GetStore().LastWrite(column));
    }
    for (const SensorKind& kind : kSensorKinds) {
        for (const ColumnRef column : kind.inputs) {
            writes.push_back(impl.GetStore().LastWrite(column));
        }
    }
    const uint64_t epoch = impl.GetCatalog().Epoch();
    const bool changed = epoch != m_epoch || writes != m_productWrites;
    if (changed && m_epoch != ~uint64_t(0)) {
        Log(LogLevel::Debug, "the scene's structure or a render product changed, so products are "
                             "prepared again");
    }
    m_epoch = epoch;
    m_productWrites = std::move(writes);
    return changed;
}

RendererImpl::StepPlan& RendererImpl::PlanProducts(std::span<const std::string> products) {
    const std::set<std::string_view> requested(products.begin(), products.end());
    for (auto cached = m_plans.begin(); cached != m_plans.end(); ++cached) {
        if (std::ranges::equal(requested, std::views::keys(cached->products))) {
            m_plans.splice(m_plans.begin(), m_plans, cached);
            m_plan = &m_plans.front();
            return *m_plan;
        }
    }
    ERNIE_ZONE("Render/PlanProducts");

    StepPlan plan;
    for (const std::string_view requestedPath : requested) {
        const std::string path(requestedPath);
        const Product& product = PrepareProduct(path);
        Capture capture{};
        capture.product = product.prim;
        capture.width = product.resolution.x;
        capture.height = product.resolution.y;
        capture.missDistance = product.missDistance;
        capture.returns = std::max(product.returns, 1u);
        if (product.semantic) {
            const auto found = std::ranges::find(plan.taxonomies, product.taxonomy);
            capture.taxonomy = uint32_t(found - plan.taxonomies.begin());
            if (found == plan.taxonomies.end()) {
                plan.taxonomies.push_back(product.taxonomy);
            }
        }

        FrameOutput frame;
        for (const ProductVar& var : product.vars) {
            frame.renderVars[var.output.renderVarPath] = var.output;
        }
        plan.products[path] = ProductOutput{path, {std::move(frame)}};

        const auto entry = uint32_t(product.entry);
        Launch& launch = plan.launches[entry];
        for (capture.layer = 0; capture.layer < product.sensors.size(); ++capture.layer) {
            capture.sensor = product.sensors[capture.layer];
            capture.firstPixel = launch.pixelCount;
            launch.captures.push_back(capture);
            launch.pixelCount += uint64_t(capture.width) * capture.height;
        }
    }

    uint64_t launches = 0;
    for (uint32_t entry = 0; entry < kRenderEntryCount; ++entry) {
        const Launch& launch = plan.launches[entry];
        plan.captureCount += uint32_t(launch.captures.size());
        plan.pixelCount += launch.pixelCount;
        launches += (launch.pixelCount + m_maxPixelsPerLaunch - 1) / m_maxPixelsPerLaunch;
    }
    Log(LogLevel::Debug, "planned {}: {}, {} in {}", Counted(plan.products.size(), "product"),
        Counted(plan.captureCount, "capture"), Counted(plan.pixelCount, "pixel"),
        Counted(launches, "launch"));
    m_plans.push_front(std::move(plan));
    m_plan = &m_plans.front();
    if (m_plans.size() > kCachedPlans) {
        m_plans.pop_back();
        const auto unused = [&](const auto& entry) {
            return std::ranges::none_of(m_plans, [&](const StepPlan& cached) {
                return cached.products.contains(entry.first);
            });
        };
        const size_t dropped = std::erase_if(m_products, unused);
        std::erase_if(m_schedules, unused);
        Log(LogLevel::Debug, "dropped the least recently stepped plan and {} only it used",
            Counted(dropped, "product"));
    }
    return *m_plan;
}

RendererImpl::Product& RendererImpl::PrepareProduct(const std::string& path) {
    if (const auto found = m_products.find(path); found != m_products.end()) {
        return found->second;
    }
    ERNIE_ZONE("Render/PrepareProduct");
    Scene& scene = *m_scene;
    const Prims& prims = SceneImpl::Get(scene).GetCatalog().GetPrims();
    const std::optional<Prim> prim = scene.Find(path);
    if (!prim || !prim->IsA<RenderProduct>()) {
        throw Error("\"{}\" isn't a RenderProduct", path);
    }
    const View view = scene.Select(*prim);
    const auto targets = [&](ColumnRef column, SchemaId schema) {
        const std::vector<uint32_t> found = view.GetArray<uint32_t>(column).Get(0);
        for (const uint32_t target : found) {
            if (!prims.Alive(target) || !IsA(prims.At(target).schema, schema)) {
                throw Error("{}'s {} targets prim {}, which isn't a {}", path, column.column->field,
                            target, Info(schema).name);
            }
        }
        return found;
    };

    Product product;
    product.prim = prim->Index();
    product.sensors = targets(kRenderProductCamera, SchemaId::Sensor);
    if (product.sensors.empty()) {
        throw Error("{} renders no sensor", path);
    }
    const Prim lead = scene.At(product.sensors.front());
    const SensorKind& kind = FindSensorKind(prims.At(lead.Index()).schema);
    product.sensor = kind.schema;
    const Uint2 resolution = prim->Get(RenderProduct::kResolution);
    const SensorShape shape = kind.shape(lead, resolution);
    product.frameRate = lead.Get(SensorAPI::kSensorFrameRate);
    product.startTime = lead.Get(SensorAPI::kSensorStartTime);
    for (const uint32_t sensor : product.sensors) {
        const Prim each = scene.At(sensor);
        if (prims.At(sensor).schema != prims.At(lead.Index()).schema) {
            throw Error("{} stacks {} and {}, which are different kinds of sensor", path,
                        lead.Path(), each.Path());
        }
        if (each.Get(SensorAPI::kSensorFrameRate) != product.frameRate ||
            each.Get(SensorAPI::kSensorStartTime) != product.startTime) {
            throw Error("{} stacks {} and {}, which capture at different times", path, lead.Path(),
                        each.Path());
        }
        if (kind.shape(each, resolution) != shape) {
            throw Error("{} stacks {} and {}, which have different shapes", path, lead.Path(),
                        each.Path());
        }
    }
    product.resolution = shape.resolution;
    product.returns = shape.returns;
    if (product.resolution.x == 0 || product.resolution.y == 0) {
        throw Error("{} has a resolution of {} x {}", path, product.resolution.x,
                    product.resolution.y);
    }
    product.taxonomy = view.Get<Token>(kRenderProductSemanticTaxonomy).Get().front();
    product.missDistance = prim->Get(RenderProduct::kMissDistance);

    std::string sources;
    for (const uint32_t var : targets(kRenderProductOrderedVars, SchemaId::RenderVar)) {
        const ProductVar& added = product.vars.emplace_back(PrepareVar(var, product));
        sources += (sources.empty() ? "" : ", ") + added.output.sourceName;
        product.semantic |= Semantic(added.source);
        product.semantic |= std::ranges::find(added.channels, "StableId") != added.channels.end();
        if (!Outputs(kind, added.source)) {
            throw Error("{} writes {}, which a {} doesn't output", path, added.output.sourceName,
                        Info(kind.schema).name);
        }
        if (!kind.measuresVelocity &&
            std::ranges::find(added.channels, "RadialVelocity") != added.channels.end()) {
            throw Error("{} asks {} for RadialVelocity, which a {} doesn't measure", path,
                        added.output.renderVarPath, Info(kind.schema).name);
        }
    }
    product.entry = kind.entry ? *kind.entry
                               : SelectEntry(product.vars, prim->Get(RenderProduct::kRenderMode),
                                             prim->Get(RenderProduct::kMode),
                                             prim->Get(RenderProduct::kSpectralMode));

    Log(LogLevel::Debug, "prepared {}: {} at {} x {} writing {} with {}{}", path,
        Counted(product.sensors.size(), "sensor"), product.resolution.x, product.resolution.y,
        sources.empty() ? "nothing" : sources, kRenderEntries[uint32_t(product.entry)],
        product.frameRate > 0 ? fmt::format(", capturing at {} Hz", product.frameRate) : "");
    return m_products.emplace(path, std::move(product)).first->second;
}

RendererImpl::ProductVar RendererImpl::PrepareVar(uint32_t var, const Product& product) {
    Scene& scene = *m_scene;
    const Prim prim = scene.At(var);
    const RenderVarSource source = prim.Get(RenderVar::kSourceName);
    RenderVarOutput output{prim.Path(), std::string(kRenderVarSourceTokens[uint32_t(source)]), {}};
    if (source == RenderVarSource::PointCloud) {
        std::vector<std::string> channels = prim.Get(RenderVar::kChannels);
        if (channels.empty()) {
            channels = {"Coordinates", "Intensity"};
        }
        for (const std::string& channel : channels) {
            FindPointChannel(channel);
        }
        return {source,
                std::move(output),
                device::DType::UInt8,
                {},
                std::move(channels),
                prim.Get(RenderVar::kFrameOfReference) == FrameOfReference::World};
    }
    if (!PerPixel(source)) {
        return {source, std::move(output)};
    }

    const RenderVarFormat format = SourceFormat(source);
    std::vector<int64_t> shape = {int64_t(product.sensors.size()), product.resolution.y,
                                  product.resolution.x};
    if (product.returns > 0) {
        shape.push_back(product.returns);
    }
    if (format.channels > 1) {
        shape.push_back(format.channels);
    }
    return {source, std::move(output), format.type, std::move(shape)};
}

void RendererImpl::ClearProducts() {
    m_products.clear();
    m_plans.clear();
    m_plan = nullptr;
}

void RendererImpl::UpdateMaps() {
    const Semantics& semantics = SceneImpl::Get(*m_scene).GetSemantics();
    for (auto& [path, planned] : m_plan->products) {
        Product& product = m_products.at(path);
        if (!product.semantic) {
            continue;
        }
        const bool stale = product.mapVersion != semantics.Runs();
        for (ProductVar& var : product.vars) {
            if (!Semantic(var.source)) {
                continue;
            }
            if (stale) {
                const IdEntries entries = MapEntries(semantics, var.source, product.taxonomy);
                IdMap map;
                std::vector<std::pair<StableId, std::string>> texts;
                for (const IdEntry& entry : entries) {
                    const StableId& id = entry.id;
                    IdLabel& label = map[{id.words[0], id.words[1], id.words[2], id.words[3]}];
                    label.path = entry.path;
                    label.labels.insert(entry.labels.begin(), entry.labels.end());
                    texts.emplace_back(id, entry.text);
                }
                var.output.idMap = std::make_shared<const IdMap>(std::move(map));
                if (!PerPixel(var.source)) {
                    const std::vector<std::byte> bytes = PackIdMap(texts);
                    const Ref<device::Buffer> buffer = m_scene->GetDevice().CreateBuffer(
                        {.label = var.output.renderVarPath}, std::span<const std::byte>(bytes));
                    var.output.tensor =
                        device::Tensor(buffer, 0, device::DType::UInt8, {int64_t(bytes.size())});
                }
            }
            for (FrameOutput& frame : planned.frames) {
                RenderVarOutput& output = frame.renderVars.at(var.output.renderVarPath);
                output.idMap = var.output.idMap;
                if (!PerPixel(var.source)) {
                    output.tensor = var.output.tensor;
                }
            }
        }
        product.mapVersion = semantics.Runs();
    }
}

RendererImpl::OutputSet& RendererImpl::AcquireSet(StepPlan& plan) {
    bool outlined = false;
    {
        const std::lock_guard lock(m_overlayMutex);
        outlined = std::ranges::any_of(
            m_overlays, [](const auto& entry) { return entry.second && entry.second->Outlined(); });
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
        const Product& product = m_products.at(path);
        std::array<uint64_t, kRenderVarSourceCount>& slots = addresses[product.prim];
        for (const ProductVar& var : product.vars) {
            if (var.source == RenderVarSource::PointCloud) {
                uint64_t& clouds = slots[uint32_t(var.source)];
                clouds = AcquireCloud(set, path, product, var, clouds);
                continue;
            }
            if (!PerPixel(var.source)) {
                continue;
            }
            const uint64_t bytes =
                std::accumulate(var.shape.begin(), var.shape.end(),
                                uint64_t(ElementBytes(var.type)), std::multiplies<>());
            const Ref<device::Buffer> buffer = device.CreateBuffer(
                {.bytes = bytes, .usage = usage, .label = var.output.renderVarPath});
            device::Tensor tensor(buffer, 0, var.type, var.shape);
            slots[uint32_t(var.source)] = tensor.Address();
            set.vars.push_back({path, var.output.renderVarPath, std::move(tensor), {}});
        }
        if (!plan.shownOverlays.contains(path)) {
            continue;
        }
        const std::vector<int64_t> shape = {int64_t(product.sensors.size()), product.resolution.y,
                                            product.resolution.x, 4};
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
        std::vector<Capture> captures = plan.launches[entry].captures;
        if (captures.empty()) {
            continue;
        }
        for (Capture& capture : captures) {
            const std::array<uint64_t, kRenderVarSourceCount>& slots = addresses[capture.product];
            std::copy(slots.begin(), slots.end(), std::begin(capture.vars));
            const auto found = primIds.find(capture.product);
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

uint64_t RendererImpl::AcquireCloud(OutputSet& set, const std::string& path, const Product& product,
                                    const ProductVar& var, uint64_t next) {
    device::Device& device = m_scene->GetDevice();
    const device::BufferUsage usage =
        device.SharedDevice() ? device::BufferUsage::Interop : device::BufferUsage::None;
    const auto sensors = int64_t(product.sensors.size());
    const uint32_t returns = std::max(product.returns, 1u);
    const uint32_t perSensor = product.resolution.x * product.resolution.y * returns;
    const auto add = [&](std::string channel, device::DType type, std::vector<int64_t> shape) {
        const uint64_t bytes = std::accumulate(shape.begin(), shape.end(),
                                               uint64_t(ElementBytes(type)), std::multiplies<>());
        device::Tensor tensor(
            device.CreateBuffer({.bytes = bytes,
                                 .usage = usage,
                                 .label = var.output.renderVarPath + " " + channel}),
            0, type, std::move(shape));
        const uint64_t address = tensor.Address();
        set.vars.push_back({path, var.output.renderVarPath, std::move(tensor), std::move(channel)});
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
    for (const std::string& name : var.channels) {
        const PointChannel& channel = FindPointChannel(name);
        std::vector<int64_t> shape = {sensors, perSensor};
        if (channel.lanes > 1) {
            shape.push_back(channel.lanes);
        }
        cloud.*channel.address = add(name, channel.type, std::move(shape));
    }
    cloud.slotsPerSensor = perSensor;
    cloud.width = product.resolution.x;
    cloud.returns = returns;
    cloud.sensors = uint32_t(sensors);
    cloud.world = var.world ? 1 : 0;
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
    params.scene.address = snapshot.sceneData;
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
        const std::lock_guard lock(m_overlayMutex);
        named.assign(m_overlays.begin(), m_overlays.end());
        std::vector<std::shared_ptr<OverlayPickState>> waiting;
        for (std::shared_ptr<OverlayPickState>& pick : m_picks) {
            const auto product = m_products.find(pick->product);
            const bool served = product != m_products.end() && due.contains(product->second.prim);
            (served ? m_servedPicks : waiting).push_back(std::move(pick));
        }
        m_picks = std::move(waiting);
    }
    if (set.overlays.empty()) {
        return {};
    }
    const auto matches = [](const std::vector<std::string>& patterns, std::string_view name) {
        return patterns.empty() || std::ranges::any_of(patterns, [&](const std::string& pattern) {
                   return pattern.ends_with('*')
                              ? name.starts_with(
                                    std::string_view(pattern).substr(0, pattern.size() - 1))
                              : name == pattern;
               });
    };

    std::vector<OverlayContents> used;
    std::vector<std::string> names;
    std::vector<uint32_t> slots(named.size(), ~0u);
    std::vector<OverlayTarget> targets;
    std::vector<std::pair<std::string, uint32_t>> targetKeys;
    for (const OwnedOverlay& overlay : set.overlays) {
        const Product& product = m_products.at(overlay.product);
        if (!due.contains(product.prim)) {
            continue;
        }
        encoder.Fill(overlay.tensor.GetBuffer());
        const std::vector<std::string>& patterns = m_plan->shownOverlays.at(overlay.product);
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
                               .width = product.resolution.x,
                               .height = product.resolution.y,
                               .missDistance = product.missDistance,
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
        const Product& product = m_products.at(var.product);
        if (!due.contains(product.prim) || product.sensor != SchemaId::Camera) {
            continue;
        }
        const auto source = std::ranges::find_if(product.vars, [&](const ProductVar& candidate) {
            return candidate.output.renderVarPath == var.renderVar &&
                   candidate.source == RenderVarSource::LdrColor;
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

void RendererImpl::SetOverlay(std::string_view name, Overlay overlay) {
    OverlayContents contents = OverlayPass::Share(overlay);
    if (contents) {
        OverlayPass::CheckDevice(*contents, m_scene->GetDevice());
    }
    OverlayContents previous;
    const std::lock_guard lock(m_overlayMutex);
    const auto found = m_overlays.find(name);
    if (found == m_overlays.end()) {
        if (contents) {
            m_overlays.emplace(std::string(name), std::move(contents));
        }
        return;
    }
    previous = std::exchange(found->second, std::move(contents));
    if (!found->second) {
        m_overlays.erase(found);
    }
}

OverlayPick RendererImpl::PickOverlay(std::string_view product, Float2 pixel, float radius,
                                      uint32_t layer) {
    if (!(radius >= 0)) {
        throw Error("An overlay pick's radius is at least 0, not {}", radius);
    }
    auto state = std::make_shared<OverlayPickState>();
    state->product = std::string(product);
    state->layer = layer;
    state->pixel = pixel;
    state->radius = radius;
    const std::lock_guard lock(m_overlayMutex);
    m_picks.push_back(state);
    return OverlayPick(std::move(state));
}

void RendererImpl::RemoveOverlay(std::string_view name) {
    SetOverlay(name, Overlay());
}

void RendererImpl::ShowOverlays(std::string_view product, std::span<const std::string> names) {
    const std::lock_guard lock(m_overlayMutex);
    m_shownOverlays.insert_or_assign(std::string(product),
                                     std::vector<std::string>(names.begin(), names.end()));
}

void RendererImpl::HideOverlays(std::string_view product) {
    const std::lock_guard lock(m_overlayMutex);
    if (const auto found = m_shownOverlays.find(product); found != m_shownOverlays.end()) {
        m_shownOverlays.erase(found);
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

void Renderer::Reset(double time) {
    m_impl->Reset(time);
}

double Renderer::Time() const {
    return m_impl->Time();
}

void Renderer::SetOverlay(std::string_view name, Overlay overlay) {
    m_impl->SetOverlay(name, std::move(overlay));
}

void Renderer::RemoveOverlay(std::string_view name) {
    m_impl->RemoveOverlay(name);
}

void Renderer::ShowOverlays(std::string_view product, Span<const std::string> names) {
    m_impl->ShowOverlays(product, names);
}

void Renderer::HideOverlays(std::string_view product) {
    m_impl->HideOverlays(product);
}

OverlayPick Renderer::PickOverlay(std::string_view product, Float2 pixel, float radius,
                                  uint32_t layer) {
    return m_impl->PickOverlay(product, pixel, radius, layer);
}

}
