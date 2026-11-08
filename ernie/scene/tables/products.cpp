// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/products.h"

#include <algorithm>
#include <cmath>
#include <optional>

#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/views/view.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr double kCaptureTolerance = 1e-6;

std::vector<ColumnRef> ProductInputs() {
    std::vector<ColumnRef> inputs = {kLidarFiringsPerScan, kLidarEmitterElevations,
                                     kLidarMaxReturns, kRadarRays};
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

bool SamePlan(const ResolvedProduct& a, const ResolvedProduct& b) {
    const ProductRecord& x = a.record;
    const ProductRecord& y = b.record;
    return x.sensorSchema == y.sensorSchema && x.width == y.width && x.height == y.height &&
           x.returns == y.returns && x.frameRate == y.frameRate && x.startTime == y.startTime &&
           x.missDistance == y.missDistance && x.renderMode == y.renderMode &&
           x.minimalMode == y.minimalMode && x.spectralMode == y.spectralMode &&
           a.sensors == b.sensors && a.vars == b.vars && a.taxonomy == b.taxonomy &&
           a.error == b.error;
}

}

std::span<const ColumnRef> ProductTable::Inputs() const {
    static const std::vector<ColumnRef> inputs = ProductInputs();
    return inputs;
}

void ProductTable::Describe(RenderTables& tables) const {
    tables.products.address = m_records.Buffer() ? m_records.Buffer()->Address() : 0;
}

const ResolvedProduct* ProductTable::Find(uint32_t prim) const {
    const auto found = m_products.find(prim);
    return found == m_products.end() ? nullptr : &found->second;
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
    record.missDistance = prim.Get(RenderProduct::kMissDistance);
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
        record.sensorSchema = uint32_t(schema);
        record.frameRate = lead.Get(SensorAPI::kSensorFrameRate);
        record.startTime = lead.Get(SensorAPI::kSensorStartTime);
        for (const uint32_t sensor : product.sensors) {
            const Prim each = m_scene.ByIndex(sensor);
            if (prims.Record(sensor).schema != schema) {
                throw Error("{} stacks {} and {}, which are different kinds of sensor", path,
                            lead.Path(), each.Path());
            }
            if (each.Get(SensorAPI::kSensorFrameRate) != record.frameRate ||
                each.Get(SensorAPI::kSensorStartTime) != record.startTime) {
                throw Error("{} stacks {} and {}, which capture at different times", path,
                            lead.Path(), each.Path());
            }
            if (SensorShape(each, resolution) != shape) {
                throw Error("{} stacks {} and {}, which have different shapes", path, lead.Path(),
                            each.Path());
            }
        }
        record.width = shape.resolution.x;
        record.height = shape.resolution.y;
        record.returns = shape.returns;
        if (record.width == 0 || record.height == 0) {
            throw Error("{} has a resolution of {} x {}", path, record.width, record.height);
        }
        for (const uint32_t var : targets(kRenderProductOrderedVars, SchemaId::RenderVar)) {
            const Prim each = m_scene.ByIndex(var);
            product.vars.push_back({.prim = var,
                                    .path = each.Path(),
                                    .source = each.Get(RenderVar::kSourceName),
                                    .channels = each.Get(RenderVar::kChannels),
                                    .frame = each.Get(RenderVar::kFrameOfReference)});
        }
    } catch (const Error& error) {
        product.error = error.what();
    }
    return product;
}

void ProductTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Products/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    const bool edited = changes.resync || changes.structure || !changes.apis.empty() ||
                        std::ranges::any_of(Inputs(), [&](ColumnRef input) {
                            return store.LastWrite(input) > m_seen;
                        });
    m_seen = store.WriteSequence();
    if (edited) {
        Resolve();
    }
    ScheduleCaptures();
}

void ProductTable::Resolve() {
    const Prims& prims = m_structure.GetPrims();
    std::map<uint32_t, ResolvedProduct> resolved;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || prims.Record(prim).schema != SchemaId::RenderProduct) {
            continue;
        }
        ResolvedProduct product = Resolve(prim);
        const auto previous = m_products.find(prim);
        product.version = previous != m_products.end() && SamePlan(previous->second, product)
                              ? previous->second.version
                              : ++m_versions;
        m_records.Set(prims.Row(prim, SchemaId::RenderProduct), product.record);
        resolved.emplace(prim, std::move(product));
    }
    m_products = std::move(resolved);
    m_records.Reserve(m_structure.GetColumnStore().Schema(SchemaId::RenderProduct).Capacity());
    m_records.Upload(m_structure.GetColumnStore().GetDevice());
}

void ProductTable::ScheduleCaptures() {
    m_scheduled = m_clock.steps;
    if (m_clock.resets != m_resets) {
        m_resets = m_clock.resets;
        m_schedules.clear();
    }
    std::erase_if(m_schedules,
                  [&](const auto& entry) { return !m_products.contains(entry.first); });
    const double dt = m_clock.end - m_clock.start;
    for (auto& [prim, product] : m_products) {
        const float rate = product.record.frameRate;
        if (rate <= 0 || dt == 0) {
            product.due = true;
            continue;
        }
        Schedule& schedule = m_schedules[prim];
        const float start = product.record.startTime;
        if (schedule.frameRate != rate || schedule.startTime != start) {
            const double first = std::ceil((m_clock.start - start) * rate - kCaptureTolerance);
            schedule = {rate, start, std::max<int64_t>(0, int64_t(first))};
        }
        const double reached = (m_clock.end - start) * rate + kCaptureTolerance;
        product.due = double(schedule.next) <= reached;
        if (!product.due) {
            continue;
        }
        const auto last = int64_t(std::floor(reached));
        if (last > schedule.next) {
            Log(LogLevel::Debug, "{} skipped {} in a step of {} s", m_scene.ByIndex(prim).Path(),
                Counted(uint64_t(last - schedule.next), "capture"), dt);
        }
        schedule.next = last + 1;
    }
}

}
