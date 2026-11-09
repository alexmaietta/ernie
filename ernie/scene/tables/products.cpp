// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/products.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <initializer_list>
#include <optional>

#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/stage/views/view.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

std::vector<ColumnRef> ProductInputs() {
    std::vector<ColumnRef> inputs = {
        kLidarFiringsPerScan, kLidarEmitterElevations,     kLidarMaxReturns,
        kRadarRays,           kDetectorAPIDetectorPattern, kSensorPipelineBinding};
    for (const SchemaId schema :
         {SchemaId::RenderProduct, SchemaId::RenderVar, SchemaId::MinimalAPI,
          SchemaId::BackgroundAPI, SchemaId::RealTimeAPI, SchemaId::SensorAPI,
          SchemaId::ColorBackground, SchemaId::GradientBackground}) {
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t index = 0; index < columns.size(); ++index) {
            inputs.push_back({schema, &columns[index], index});
        }
    }
    return inputs;
}

struct Shape {
    Uint2 resolution;
    uint32_t returns = 0;

    bool operator==(const Shape&) const = default;
};

Shape SensorShape(const Prim& sensor, Uint2 resolution) {
    if (sensor.IsA<Lidar>()) {
        const Shape shape{{sensor.Get(Lidar::kFiringsPerScan),
                           uint32_t(sensor.Get(Lidar::kEmitterElevations).size())},
                          sensor.Get(Lidar::kMaxReturns)};
        if (shape.returns == 0 || shape.returns > kMostLidarReturns) {
            throw Error("{} reports {} returns per beam, and lidars report 1 to {}", sensor.Path(),
                        shape.returns, kMostLidarReturns);
        }
        if (shape.resolution.y == 0) {
            throw Error("{} has no emitters", sensor.Path());
        }
        return shape;
    }
    if (sensor.IsA<Radar>()) {
        return {sensor.Get(Radar::kRays)};
    }
    return {resolution};
}

int64_t DetectorChannels(const Prim& camera) {
    switch (camera.Get(DetectorAPI::kDetectorPattern)) {
    case DetectorPattern::Mono:
        return 1;
    case DetectorPattern::Rccc:
        return 2;
    case DetectorPattern::Bands:
        return kDetectorBands;
    default:
        return 3;
    }
}

struct VarFormat {
    device::DType type;
    int64_t channels;
};

VarFormat SourceFormat(RenderVarSource source) {
    switch (source) {
    case RenderVarSource::LdrColor:
    case RenderVarSource::DiffuseAlbedo:
        return {device::DType::UInt8, 4};
    case RenderVarSource::HdrColor:
        return {device::DType::Float32, 4};
    case RenderVarSource::DistanceToCamera:
    case RenderVarSource::DistanceToImagePlane:
    case RenderVarSource::Intensity:
    case RenderVarSource::RadialVelocity:
        return {device::DType::Float32, 1};
    case RenderVarSource::NonStableInstanceSegmentation:
    case RenderVarSource::SemanticSegmentation:
        return {device::DType::UInt32, 1};
    case RenderVarSource::Normal:
        return {device::DType::Float32, 3};
    case RenderVarSource::SemanticInstanceSegmentation:
    case RenderVarSource::StableIdSegmentation:
    case RenderVarSource::InstancerStableId:
        return {device::DType::UInt32, 4};
    case RenderVarSource::InstanceId:
        return {device::DType::Int64, 1};
    default:
        return {device::DType::UInt8, 0};
    }
}

constexpr uint64_t Sources(std::initializer_list<RenderVarSource> sources) {
    uint64_t mask = 0;
    for (const RenderVarSource source : sources) {
        mask |= uint64_t(1) << uint32_t(source);
    }
    return mask;
}

constexpr bool Includes(uint64_t sources, RenderVarSource source) {
    return (sources >> uint32_t(source) & 1) != 0;
}

constexpr uint64_t kSemanticSources =
    Sources({RenderVarSource::NonStableInstanceSegmentation, RenderVarSource::SemanticSegmentation,
             RenderVarSource::SemanticIdMap, RenderVarSource::SemanticInstanceSegmentation,
             RenderVarSource::StableIdSegmentation, RenderVarSource::StableIdMap});

constexpr uint64_t kRangeSources =
    kSemanticSources | Sources({RenderVarSource::DistanceToCamera, RenderVarSource::Normal,
                                RenderVarSource::Intensity, RenderVarSource::PointCloud});

struct SensorKind {
    SchemaId schema;
    std::optional<RenderEntry> entry;
    uint64_t outputs;
};

const SensorKind kSensorKinds[] = {
    {SchemaId::Camera, std::nullopt,
     ((uint64_t(1) << kRenderVarSourceCount) - 1) &
         ~Sources({RenderVarSource::Intensity, RenderVarSource::PointCloud,
                   RenderVarSource::RadialVelocity})},
    {SchemaId::Lidar, RenderEntry::Lidar, kRangeSources},
    {SchemaId::Radar, RenderEntry::Radar,
     kRangeSources | Sources({RenderVarSource::RadialVelocity})},
};

const SensorKind& FindSensorKind(SchemaId schema) {
    for (const SensorKind& kind : kSensorKinds) {
        if (IsA(schema, kind.schema)) {
            return kind;
        }
    }
    throw Error("A {} isn't a sensor the renderer renders", Info(schema).name);
}

struct BoundPipeline {
    std::shared_ptr<const PipelineInstance> instance;
};

std::vector<PointChannel> CloudChannels(const Prim& prim, const SensorKind& kind, bool& semantic) {
    std::vector<std::string> names = prim.Get(RenderVar::kChannels);
    if (names.empty()) {
        names = {"Coordinates", "Intensity"};
    }
    std::vector<PointChannel> channels;
    for (const std::string& channel : names) {
        const auto found = std::ranges::find(kPointChannels, channel, &PointChannelInfo::name);
        if (found == std::end(kPointChannels)) {
            throw Error("{} asks for {}, which a PointCloud has no channel for", prim.Path(),
                        channel);
        }
        const auto point = PointChannel(found - std::begin(kPointChannels));
        if ((point == PointChannel::RadialVelocity ||
             point == PointChannel::RadialVelocityVector) &&
            !Includes(kind.outputs, RenderVarSource::RadialVelocity)) {
            throw Error("{} asks for {}, which a {} doesn't measure", prim.Path(), channel,
                        Info(kind.schema).name);
        }
        semantic |= point == PointChannel::StableId;
        channels.push_back(point);
    }
    return channels;
}

ResolvedVar PipelineVar(const Prim& prim, const ResolvedProduct& product, const SensorKind& kind,
                        const BoundPipeline& bound, const std::string& name) {
    const bool tap = name == kTapSource;
    const std::string output = tap ? prim.Path() : name;
    const auto found = std::ranges::find(bound.instance->resolved.outputs, output,
                                         &pipeline::ResolvedOutput::name);
    if (found == bound.instance->resolved.outputs.end()) {
        if (tap) {
            throw Error("{} taps no node of {}", prim.Path(), bound.instance->path);
        }
        throw Error("{} reads {}, which isn't an output the renderer writes or {} makes",
                    prim.Path(), name, bound.instance->path);
    }
    const pipeline::SignalLayout& layout = product.layouts[found->value];
    ResolvedVar var{.prim = prim.Index(),
                    .path = prim.Path(),
                    .frame = prim.Get(RenderVar::kFrameOfReference),
                    .type = device::DType::Float32,
                    .output = output,
                    .tap = tap};
    const pipeline::SignalType type = bound.instance->resolved.values[found->value].type;
    if (type == pipeline::SignalType::PointCloud ||
        (type == pipeline::SignalType::Returns &&
         (kind.schema == SchemaId::Lidar || kind.schema == SchemaId::Radar))) {
        if (kind.schema != SchemaId::Lidar && kind.schema != SchemaId::Radar) {
            throw Error("{} reads the point cloud {} makes, and only lidars and radars make them "
                        "in a pipeline",
                        prim.Path(), bound.instance->path);
        }
        var.source = RenderVarSource::PointCloud;
        var.channels = CloudChannels(prim, kind, var.semantic);
        return var;
    }
    var.shape = {int64_t(product.sensors.size())};
    if (layout.perCapture) {
        var.shape.push_back(pipeline::kStatWords);
        return var;
    }
    var.shape.insert(var.shape.end(), {int64_t(product.height), int64_t(product.width)});
    if (product.returns > 0) {
        var.shape.push_back(product.returns);
    }
    if (layout.channels > 1) {
        var.shape.push_back(layout.channels);
    }
    return var;
}

ResolvedVar ResolveVar(const Prim& prim, const ResolvedProduct& product, const SensorKind& kind,
                       int64_t channels, const BoundPipeline& bound) {
    const std::string name = prim.Get(RenderVar::kSourceName);
    const std::optional<RenderVarSource> builtIn = FindRenderVarSource(name);
    if (!builtIn) {
        if (!bound.instance && name == kTapSource) {
            throw Error("{} taps a pipeline node, but its product binds no pipeline", prim.Path());
        }
        if (!bound.instance) {
            throw Error("{} reads {}, which isn't an output the renderer writes", prim.Path(),
                        name);
        }
        return PipelineVar(prim, product, kind, bound, name);
    }
    const RenderVarSource source = *builtIn;
    ResolvedVar var{.prim = prim.Index(),
                    .path = prim.Path(),
                    .source = source,
                    .frame = prim.Get(RenderVar::kFrameOfReference),
                    .semantic = Includes(kSemanticSources, source)};
    if (!Includes(kind.outputs, source)) {
        throw Error("{} writes {}, which a {} doesn't output", prim.Path(), name,
                    Info(kind.schema).name);
    }
    if (source == RenderVarSource::PointCloud) {
        var.channels = CloudChannels(prim, kind, var.semantic);
        return var;
    }
    VarFormat format = SourceFormat(source);
    if (source == RenderVarSource::ElectronRate) {
        if (channels == 0) {
            throw Error("{} writes ElectronRate from cameras whose detector patterns have "
                        "different channel counts",
                        var.path);
        }
        format = {device::DType::Float32, channels};
    }
    if (format.channels == 0) {
        return var;
    }
    var.type = format.type;
    var.shape = {int64_t(product.sensors.size()), product.height, product.width};
    if (product.returns > 0) {
        var.shape.push_back(product.returns);
    }
    if (format.channels > 1) {
        var.shape.push_back(format.channels);
    }
    return var;
}

RenderEntry SelectEntry(const ResolvedProduct& product, const SensorKind& kind) {
    if (kind.entry) {
        return *kind.entry;
    }
    const bool shaded = std::ranges::any_of(product.vars, [](const ResolvedVar& var) {
        return var.source == RenderVarSource::LdrColor || var.source == RenderVarSource::HdrColor ||
               var.source == RenderVarSource::DiffuseAlbedo ||
               var.source == RenderVarSource::ElectronRate || !var.output.empty();
    });
    if (!shaded) {
        return RenderEntry::MinimalData;
    }
    const ProductRecord& record = product.record;
    if (RenderMode(record.renderMode) == RenderMode::RealTime) {
        return SpectralMode(record.spectralMode) == SpectralMode::Bands ? RenderEntry::RealTimeBands
                                                                        : RenderEntry::RealTime;
    }
    switch (MinimalMode(record.minimalMode)) {
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

}

std::span<const ColumnRef> ProductTable::Inputs() const {
    static const std::vector<ColumnRef> inputs = ProductInputs();
    return inputs;
}

void ProductTable::Describe(RenderTables& tables) const {
    tables.products.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
}

std::shared_ptr<const ResolvedProduct> ProductTable::Find(uint32_t prim) const {
    const std::lock_guard lock(m_mutex);
    const auto found = m_products.find(prim);
    return found == m_products.end() ? nullptr : found->second;
}

std::vector<std::shared_ptr<const ResolvedProduct>> ProductTable::Products() const {
    const std::lock_guard lock(m_mutex);
    std::vector<std::shared_ptr<const ResolvedProduct>> products;
    for (const auto& [prim, product] : m_products) {
        products.push_back(product);
    }
    return products;
}

ResolvedProduct ProductTable::Resolve(uint32_t index) const {
    const Prims& prims = m_structure.GetPrims();
    ResolvedProduct product;
    product.prim = index;
    const Prim prim = m_scene.ByIndex(index);
    const std::string path = prim.Path();
    const View view(m_structure, SchemaId::RenderProduct, std::vector<uint32_t>{index});
    const auto targets = [&](ColumnRef column, SchemaId schema) {
        const std::vector<uint32_t> found = view.GetArray<uint32_t>(column).Get(0);
        for (const uint32_t target : found) {
            if (!prims.Alive(target) || !IsA(prims.Record(target).schema, schema)) {
                throw Error("{}'s {} targets prim {}, which isn't a {}", path, column.column->field,
                            target, Info(schema).name);
            }
        }
        return found;
    };
    ProductRecord& record = product.record;
    record.textureLodBias = prim.Get(RenderProduct::kTextureLodBias);
    record.maxAnisotropy = uint32_t(std::max(prim.Get(RenderProduct::kMaxAnisotropy), 1));
    record.maxBounces = uint32_t(std::max(prim.Get(RealTimeAPI::kMaxBounces), 0));
    record.pixelSamples = uint32_t(std::max(prim.Get(RealTimeAPI::kPixelSamples), 1));
    record.conform = uint32_t(prim.Get(RenderProduct::kAspectRatioConformPolicy));
    record.pixelAspect = prim.Get(RenderProduct::kPixelAspectRatio);
    if (!(record.pixelAspect > 0)) {
        record.pixelAspect = 1;
    }
    record.castShadows = prim.Get(MinimalAPI::kCastShadows) ? 1 : 0;
    record.shadowVisibility = prim.Get(MinimalAPI::kShadowVisibility);
    record.hemispheric = prim.Get(MinimalAPI::kAmbientHemispheric) ? 1 : 0;
    const float intensity = prim.Get(MinimalAPI::kAmbientLightIntensity);
    const Float3 ambient = prim.Get(MinimalAPI::kAmbientLightColor);
    const Float3 ground = prim.Get(MinimalAPI::kAmbientGroundColor);
    record.ambient = {ambient.x * intensity, ambient.y * intensity, ambient.z * intensity};
    record.ground = {ground.x * intensity, ground.y * intensity, ground.z * intensity};
    record.backgroundColor = prim.Get(BackgroundAPI::kBackgroundColor);
    const std::vector<uint32_t> backgrounds =
        view.GetArray<uint32_t>(kBackgroundAPIBackground).Get(0);
    record.background = backgrounds.empty() ? kNoPrim : backgrounds.front();
    record.backgroundKind = uint32_t(SchemaId::None);
    if (record.background != kNoPrim && prims.Alive(record.background)) {
        const Prim background = m_scene.ByIndex(record.background);
        record.backgroundKind = uint32_t(prims.Record(record.background).schema);
        if (background.IsA<ColorBackground>()) {
            record.backgroundLow = background.Get(ColorBackground::kColor);
        } else if (background.IsA<GradientBackground>()) {
            record.backgroundLow = background.Get(GradientBackground::kHorizonColor);
            record.backgroundHigh = background.Get(GradientBackground::kZenithColor);
        }
    }
    record.renderMode = uint32_t(prim.Get(RenderProduct::kRenderMode));
    record.minimalMode = uint32_t(prim.Get(MinimalAPI::kMode));
    record.spectralMode = uint32_t(prim.Get(RenderProduct::kSpectralMode));
    product.missDistance = prim.Get(RenderProduct::kMissDistance);
    record.sensors = m_structure.GetColumnStore().Array(kRenderProductCamera,
                                                        prims.Row(index, SchemaId::RenderProduct));
    record.vars = m_structure.GetColumnStore().Array(kRenderProductOrderedVars,
                                                     prims.Row(index, SchemaId::RenderProduct));
    product.taxonomy = view.Get<Token>(kRenderProductSemanticTaxonomy).Get().front();
    try {
        product.sensors = targets(kRenderProductCamera, SchemaId::Sensor);
        if (product.sensors.empty()) {
            throw Error("{} renders no sensor", path);
        }
        const Prim lead = m_scene.ByIndex(product.sensors.front());
        const SchemaId schema = prims.Record(lead.Index()).schema;
        const Uint2 resolution = prim.Get(RenderProduct::kResolution);
        const Shape shape = SensorShape(lead, resolution);
        product.sensor = schema;
        product.frameRate = lead.Get(SensorAPI::kSensorFrameRate);
        product.startTime = lead.Get(SensorAPI::kSensorStartTime);
        for (const uint32_t sensor : product.sensors) {
            const Prim each = m_scene.ByIndex(sensor);
            if (prims.Record(sensor).schema != schema) {
                throw Error("{} stacks {} and {}, which are different kinds of sensor", path,
                            lead.Path(), each.Path());
            }
            if (each.Get(SensorAPI::kSensorFrameRate) != product.frameRate ||
                each.Get(SensorAPI::kSensorStartTime) != product.startTime) {
                throw Error("{} stacks {} and {}, which capture at different times", path,
                            lead.Path(), each.Path());
            }
            if (SensorShape(each, resolution) != shape) {
                throw Error("{} stacks {} and {}, which have different shapes", path, lead.Path(),
                            each.Path());
            }
        }
        product.width = shape.resolution.x;
        product.height = shape.resolution.y;
        product.returns = shape.returns;
        if (product.width == 0 || product.height == 0) {
            throw Error("{} has a resolution of {} x {}", path, product.width, product.height);
        }
        const SensorKind& kind = FindSensorKind(schema);
        int64_t channels = 0;
        if (lead.IsA<Camera>()) {
            channels = DetectorChannels(lead);
            for (const uint32_t sensor : product.sensors) {
                channels = DetectorChannels(m_scene.ByIndex(sensor)) == channels ? channels : 0;
            }
        }
        product.channels = uint32_t(channels);
        const auto binding = [&](ColumnRef column, uint32_t at) {
            const View bound(m_structure, prims.Record(at).schema, std::vector<uint32_t>{at});
            const std::vector<uint32_t> found = bound.GetArray<uint32_t>(column).Get(0);
            return found.empty() || !prims.Alive(found.front()) ? kNoPrim : found.front();
        };
        product.pipeline = binding(kRenderProductPipelineBinding, index);
        if (product.pipeline == kNoPrim) {
            product.pipeline = binding(kSensorPipelineBinding, lead.Index());
            for (const uint32_t sensor : product.sensors) {
                if (binding(kSensorPipelineBinding, sensor) != product.pipeline) {
                    throw Error("{} stacks {} and {}, which bind different pipelines", path,
                                lead.Path(), m_scene.ByIndex(sensor).Path());
                }
            }
        }
        BoundPipeline bound;
        if (product.pipeline != kNoPrim) {
            if (!IsA(prims.Record(product.pipeline).schema, SchemaId::SensorPipeline)) {
                throw Error("{} binds {}, which isn't a SensorPipeline", path,
                            m_scene.ByIndex(product.pipeline).Path());
            }
            bound.instance = m_pipelines.Find(product.pipeline);
            if (!bound.instance) {
                throw Error("{} binds {}, which isn't resolved yet", path,
                            m_scene.ByIndex(product.pipeline).Path());
            }
            if (!bound.instance->error.empty()) {
                throw Error("{}", bound.instance->error);
            }
            const DetectorPattern pattern = lead.IsA<Camera>()
                                                ? lead.Get(DetectorAPI::kDetectorPattern)
                                                : DetectorPattern::Mono;
            const bool mosaics = pattern != DetectorPattern::Mono &&
                                 pattern != DetectorPattern::Rgb &&
                                 pattern != DetectorPattern::Bands;
            product.layouts =
                pipeline::PropagateLayouts(bound.instance->resolved, *m_pipelines.Registry(),
                                           {.channels = uint32_t(std::max<int64_t>(channels, 1)),
                                            .pattern = uint32_t(pattern),
                                            .mosaics = mosaics});
        }
        for (const uint32_t var : targets(kRenderProductOrderedVars, SchemaId::RenderVar)) {
            const ResolvedVar& added = product.vars.emplace_back(
                ResolveVar(m_scene.ByIndex(var), product, kind, channels, bound));
            product.semantic |= added.semantic;
        }
        const bool rated =
            std::ranges::any_of(product.vars,
                                [](const ResolvedVar& var) {
                                    return var.output.empty() &&
                                           var.source == RenderVarSource::ElectronRate;
                                }) ||
            (bound.instance &&
             std::ranges::any_of(bound.instance->resolved.values, [](const pipeline::Value& value) {
                 return value.type == pipeline::SignalType::ElectronRate;
             }));
        if (bound.instance &&
            std::ranges::any_of(bound.instance->resolved.values, [](const pipeline::Value& value) {
                return value.type == pipeline::SignalType::Returns;
            })) {
            std::optional<FrameOfReference> frame;
            for (const ResolvedVar& var : product.vars) {
                if (var.output.empty() || var.source != RenderVarSource::PointCloud) {
                    continue;
                }
                if (frame && *frame != var.frame) {
                    throw Error("{} reads point clouds from {} in two frames of reference, and a "
                                "pipeline's clouds share one",
                                path, bound.instance->path);
                }
                frame = var.frame;
            }
            if (kind.schema != SchemaId::Lidar && kind.schema != SchemaId::Radar) {
                throw Error("{} runs {}, which reads returns, so its sensor must be a Lidar or a "
                            "Radar",
                            path, bound.instance->path);
            }
        }
        if (rated && RenderMode(record.renderMode) != RenderMode::RealTime) {
            throw Error("{} measures electron rates, which only a RealTime render traces, so set "
                        "its renderMode to RealTime",
                        path);
        }
        product.entry = SelectEntry(product, kind);
        product.overlays = prim.Get(RenderProduct::kOverlays);
        if (!product.overlays.empty() && kind.schema != SchemaId::Camera) {
            throw Error("{} draws overlays over a {}, and only cameras draw them", path,
                        Info(kind.schema).name);
        }
    } catch (const Error& error) {
        product.error = error.what();
    }
    record.sensorSchema = uint32_t(product.sensor);
    record.width = product.width;
    record.height = product.height;
    record.returns = product.returns;
    record.frameRate = product.frameRate;
    record.startTime = product.startTime;
    record.missDistance = product.missDistance;
    return product;
}

void ProductTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Products/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    const bool edited = changes.resync || changes.structure || !changes.apis.empty() ||
                        m_pipelines.Version() != m_pipelineVersion ||
                        std::ranges::any_of(Inputs(), [&](ColumnRef input) {
                            return store.LastWrite(input) > m_seen;
                        });
    m_seen = store.WriteSequence();
    m_pipelineVersion = m_pipelines.Version();
    if (edited) {
        Resolve();
    }
}

void ProductTable::Resolve() {
    const Prims& prims = m_structure.GetPrims();
    std::map<uint32_t, std::shared_ptr<const ResolvedProduct>> resolved;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || prims.Record(prim).schema != SchemaId::RenderProduct) {
            continue;
        }
        ResolvedProduct product = Resolve(prim);
        const std::shared_ptr<const ResolvedProduct> previous = Find(prim);
        const bool same = previous && static_cast<const ProductShape&>(*previous) == product;
        product.version = same ? previous->version : ++m_versions;
        if (!same && product.error.empty()) {
            std::string sources;
            for (const ResolvedVar& var : product.vars) {
                sources += std::format(
                    "{}{}", sources.empty() ? "" : ", ",
                    var.output.empty() ? std::string(kRenderVarSourceTokens[uint32_t(var.source)])
                                       : var.output);
            }
            Log(LogLevel::Debug, "resolved {}: {} at {} x {} writing {}{}",
                m_scene.ByIndex(prim).Path(), Counted(product.sensors.size(), "sensor"),
                product.width, product.height, sources.empty() ? "nothing" : sources,
                product.frameRate > 0 ? std::format(", capturing at {} Hz", product.frameRate)
                                      : "");
        }
        m_records.Set(prims.Row(prim, SchemaId::RenderProduct), product.record);
        resolved.emplace(prim, std::make_shared<const ResolvedProduct>(std::move(product)));
    }
    {
        const std::lock_guard lock(m_mutex);
        m_products = std::move(resolved);
    }
    m_records.Reserve(m_structure.GetColumnStore().Schema(SchemaId::RenderProduct).Capacity());
    m_records.Upload(m_structure.GetColumnStore().GetDevice());
}

}
