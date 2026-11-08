// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/lights.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>

#include "ernie/scene/kernels.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/instance_layout.h"
#include "ernie/scene/tables/kernels/lights.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr double kCmf[kSpectralRows][3] = {
    {1.299000e-04, 3.917000e-06, 6.061000e-04}, {2.321000e-04, 6.965000e-06, 1.086000e-03},
    {4.149000e-04, 1.239000e-05, 1.946000e-03}, {7.416000e-04, 2.202000e-05, 3.486000e-03},
    {1.368000e-03, 3.900000e-05, 6.450001e-03}, {2.236000e-03, 6.400000e-05, 1.054999e-02},
    {4.243000e-03, 1.200000e-04, 2.005001e-02}, {7.650000e-03, 2.170000e-04, 3.621000e-02},
    {1.431000e-02, 3.960000e-04, 6.785001e-02}, {2.319000e-02, 6.400000e-04, 1.102000e-01},
    {4.351000e-02, 1.210000e-03, 2.074000e-01}, {7.763000e-02, 2.180000e-03, 3.713000e-01},
    {1.343800e-01, 4.000000e-03, 6.456000e-01}, {2.147700e-01, 7.300000e-03, 1.039050e+00},
    {2.839000e-01, 1.160000e-02, 1.385600e+00}, {3.285000e-01, 1.684000e-02, 1.622960e+00},
    {3.482800e-01, 2.300000e-02, 1.747060e+00}, {3.480600e-01, 2.980000e-02, 1.782600e+00},
    {3.362000e-01, 3.800000e-02, 1.772110e+00}, {3.187000e-01, 4.800000e-02, 1.744100e+00},
    {2.908000e-01, 6.000000e-02, 1.669200e+00}, {2.511000e-01, 7.390000e-02, 1.528100e+00},
    {1.953600e-01, 9.098000e-02, 1.287640e+00}, {1.421000e-01, 1.126000e-01, 1.041900e+00},
    {9.564000e-02, 1.390200e-01, 8.129501e-01}, {5.795001e-02, 1.693000e-01, 6.162000e-01},
    {3.201000e-02, 2.080200e-01, 4.651800e-01}, {1.470000e-02, 2.586000e-01, 3.533000e-01},
    {4.900000e-03, 3.230000e-01, 2.720000e-01}, {2.400000e-03, 4.073000e-01, 2.123000e-01},
    {9.300000e-03, 5.030000e-01, 1.582000e-01}, {2.910000e-02, 6.082000e-01, 1.117000e-01},
    {6.327000e-02, 7.100000e-01, 7.824999e-02}, {1.096000e-01, 7.932000e-01, 5.725001e-02},
    {1.655000e-01, 8.620000e-01, 4.216000e-02}, {2.257499e-01, 9.148501e-01, 2.984000e-02},
    {2.904000e-01, 9.540000e-01, 2.030000e-02}, {3.597000e-01, 9.803000e-01, 1.340000e-02},
    {4.334499e-01, 9.949501e-01, 8.749999e-03}, {5.120501e-01, 1.000000e+00, 5.749999e-03},
    {5.945000e-01, 9.950000e-01, 3.900000e-03}, {6.784000e-01, 9.786000e-01, 2.749999e-03},
    {7.621000e-01, 9.520000e-01, 2.100000e-03}, {8.425000e-01, 9.154000e-01, 1.800000e-03},
    {9.163000e-01, 8.700000e-01, 1.650001e-03}, {9.786000e-01, 8.163000e-01, 1.400000e-03},
    {1.026300e+00, 7.570000e-01, 1.100000e-03}, {1.056700e+00, 6.949000e-01, 1.000000e-03},
    {1.062200e+00, 6.310000e-01, 8.000000e-04}, {1.045600e+00, 5.668000e-01, 6.000000e-04},
    {1.002600e+00, 5.030000e-01, 3.400000e-04}, {9.384000e-01, 4.412000e-01, 2.400000e-04},
    {8.544499e-01, 3.810000e-01, 1.900000e-04}, {7.514000e-01, 3.210000e-01, 1.000000e-04},
    {6.424000e-01, 2.650000e-01, 4.999999e-05}, {5.419000e-01, 2.170000e-01, 3.000000e-05},
    {4.479000e-01, 1.750000e-01, 2.000000e-05}, {3.608000e-01, 1.382000e-01, 1.000000e-05},
    {2.835000e-01, 1.070000e-01, 0.000000e+00}, {2.187000e-01, 8.160000e-02, 0.000000e+00},
    {1.649000e-01, 6.100000e-02, 0.000000e+00}, {1.212000e-01, 4.458000e-02, 0.000000e+00},
    {8.740000e-02, 3.200000e-02, 0.000000e+00}, {6.360000e-02, 2.320000e-02, 0.000000e+00},
    {4.677000e-02, 1.700000e-02, 0.000000e+00}, {3.290000e-02, 1.192000e-02, 0.000000e+00},
    {2.270000e-02, 8.210000e-03, 0.000000e+00}, {1.584000e-02, 5.723000e-03, 0.000000e+00},
    {1.135916e-02, 4.102000e-03, 0.000000e+00}, {8.110916e-03, 2.929000e-03, 0.000000e+00},
    {5.790346e-03, 2.091000e-03, 0.000000e+00}, {4.109457e-03, 1.484000e-03, 0.000000e+00},
    {2.899327e-03, 1.047000e-03, 0.000000e+00}, {2.049190e-03, 7.400000e-04, 0.000000e+00},
    {1.439971e-03, 5.200000e-04, 0.000000e+00}, {9.999493e-04, 3.611000e-04, 0.000000e+00},
    {6.900786e-04, 2.492000e-04, 0.000000e+00}, {4.760213e-04, 1.719000e-04, 0.000000e+00},
    {3.323011e-04, 1.200000e-04, 0.000000e+00}, {2.348261e-04, 8.480000e-05, 0.000000e+00},
    {1.661505e-04, 6.000000e-05, 0.000000e+00}, {1.174130e-04, 4.240000e-05, 0.000000e+00},
    {8.307527e-05, 3.000000e-05, 0.000000e+00}, {5.870652e-05, 2.120000e-05, 0.000000e+00},
    {4.150994e-05, 1.499000e-05, 0.000000e+00}, {2.935326e-05, 1.060000e-05, 0.000000e+00},
    {2.067383e-05, 7.465700e-06, 0.000000e+00}, {1.455977e-05, 5.257800e-06, 0.000000e+00},
    {1.025398e-05, 3.702900e-06, 0.000000e+00}, {7.221456e-06, 2.607800e-06, 0.000000e+00},
    {5.085868e-06, 1.836600e-06, 0.000000e+00}, {3.581652e-06, 1.293400e-06, 0.000000e+00},
    {2.522525e-06, 9.109300e-07, 0.000000e+00}, {1.776509e-06, 6.415300e-07, 0.000000e+00},
    {1.251141e-06, 4.518100e-07, 0.000000e+00},
};

constexpr uint32_t kDaylightRows = 54;
constexpr double kDaylight[kDaylightRows][3] = {
    {0.04, 0.02, 0.0},   {6.0, 4.5, 2.0},     {29.6, 22.4, 4.0},   {55.3, 42.0, 8.5},
    {57.3, 40.6, 7.8},   {61.8, 41.6, 6.7},   {61.5, 38.0, 5.3},   {68.8, 42.4, 6.1},
    {63.4, 38.5, 3.0},   {65.8, 35.0, 1.2},   {94.8, 43.4, -1.1},  {104.8, 46.3, -0.5},
    {105.9, 43.9, -0.7}, {96.8, 37.1, -1.2},  {113.9, 36.7, -2.6}, {125.6, 35.9, -2.9},
    {125.5, 32.6, -2.8}, {121.3, 27.9, -2.6}, {121.3, 24.3, -2.6}, {113.5, 20.1, -1.8},
    {113.1, 16.2, -1.5}, {110.8, 13.2, -1.3}, {106.5, 8.6, -1.2},  {108.8, 6.1, -1.0},
    {105.3, 4.2, -0.5},  {104.4, 1.9, -0.3},  {100.0, 0.0, 0.0},   {96.0, -1.6, 0.2},
    {95.1, -3.5, 0.5},   {89.1, -3.5, 2.1},   {90.5, -5.8, 3.2},   {90.3, -7.2, 4.1},
    {88.4, -8.6, 4.7},   {84.0, -9.5, 5.1},   {85.1, -10.9, 6.7},  {81.9, -10.7, 7.3},
    {82.6, -12.0, 8.6},  {84.9, -14.0, 9.8},  {81.3, -13.6, 10.2}, {71.9, -12.0, 8.3},
    {74.3, -13.3, 9.6},  {76.4, -12.9, 8.5},  {63.3, -10.6, 7.0},  {71.7, -11.6, 7.6},
    {77.0, -12.2, 8.0},  {65.2, -10.2, 6.7},  {47.7, -7.8, 5.2},   {68.6, -11.2, 7.4},
    {65.0, -10.4, 6.8},  {66.0, -10.6, 7.0},  {61.0, -9.7, 6.4},   {53.3, -8.3, 5.5},
    {58.9, -9.3, 6.1},   {61.9, -9.8, 6.5},
};

struct Rgb {
    double r = 0;
    double g = 0;
    double b = 0;
};

double Planck(double nm, double kelvin) {
    return 1.0 / (std::pow(nm, 5.0) * std::expm1(1.438776877e7 / (nm * kelvin)));
}

double Daylight(double nm, double kelvin) {
    const double t = std::clamp(kelvin, 4000.0, 25000.0);
    const double x = t <= 7000
                         ? -4.6070e9 / (t * t * t) + 2.9678e6 / (t * t) + 99.11 / t + 0.244063
                         : -2.0064e9 / (t * t * t) + 1.9018e6 / (t * t) + 247.48 / t + 0.237040;
    const double y = -3 * x * x + 2.870 * x - 0.275;
    const double m = 0.0241 + 0.2562 * x - 0.7341 * y;
    const double m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m;
    const double m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m;
    const double at = std::clamp((nm - 300) / 10, 0.0, double(kDaylightRows - 1));
    const uint32_t row = std::min(uint32_t(at), kDaylightRows - 2);
    const double f = at - row;
    double value = 0;
    const double weights[3] = {1, m1, m2};
    for (uint32_t component = 0; component < 3; ++component) {
        value += weights[component] *
                 std::lerp(kDaylight[row][component], kDaylight[row + 1][component], f);
    }
    return value;
}

template <class Spectrum> Rgb SpectrumRgb(const Spectrum& spectrum) {
    double x = 0;
    double y = 0;
    double z = 0;
    for (uint32_t row = 0; row < kSpectralRows; ++row) {
        const double value = spectrum(kSpectralFirstNm + row * kSpectralStepNm);
        x += value * kCmf[row][0];
        y += value * kCmf[row][1];
        z += value * kCmf[row][2];
    }
    return {(3.2404542 * x - 1.5371385 * y - 0.4985314 * z) / y,
            (-0.9692660 * x + 1.8760108 * y + 0.0415560 * z) / y,
            (0.0556434 * x - 0.2040259 * y + 1.0572252 * z) / y};
}

Float3 UnitLuminance(Rgb rgb, Rgb white) {
    rgb = {std::max(rgb.r / white.r, 0.0), std::max(rgb.g / white.g, 0.0),
           std::max(rgb.b / white.b, 0.0)};
    const double luminance = 0.2126729 * rgb.r + 0.7151522 * rgb.g + 0.0721750 * rgb.b;
    return {float(rgb.r / luminance), float(rgb.g / luminance), float(rgb.b / luminance)};
}

SpectralTables BuildSpectralTables() {
    SpectralTables tables{};
    for (uint32_t row = 0; row < kSpectralRows; ++row) {
        tables.cmf[row] = {float(kCmf[row][0]), float(kCmf[row][1]), float(kCmf[row][2])};
    }
    const Rgb white = SpectrumRgb([](double nm) { return Daylight(nm, 6504); });
    tables.white = {float(white.r), float(white.g), float(white.b)};
    for (uint32_t row = 0; row < kTemperatureRows; ++row) {
        const double mired = std::max(double(row) / (kTemperatureRows - 1) * kMaxMired, 1e-3);
        const double kelvin = 1e6 / mired;
        tables.blackbody[row] =
            UnitLuminance(SpectrumRgb([&](double nm) { return Planck(nm, kelvin); }), white);
        tables.daylight[row] =
            UnitLuminance(SpectrumRgb([&](double nm) { return Daylight(nm, kelvin); }), white);
    }
    return tables;
}

std::vector<ColumnRef> LightInputs() {
    std::vector<ColumnRef> inputs = {kRenderProductCamera};
    const SchemaId schemas[] = {SchemaId::LightAPI,
                                SchemaId::DistantLight,
                                SchemaId::SphereLight,
                                SchemaId::RectLight,
                                SchemaId::DiskLight,
                                SchemaId::CylinderLight,
                                SchemaId::DomeLight,
                                SchemaId::PortalLight,
                                SchemaId::ShapingAPI,
                                SchemaId::OmniShapingAPI,
                                SchemaId::ShadowAPI,
                                SchemaId::PhysicalLightIlluminantAPI,
                                SchemaId::PhotometricAreaLightAPI,
                                SchemaId::PhotometricDistantLightAPI,
                                SchemaId::CollectionAPI,
                                SchemaId::PointInstancer,
                                SchemaId::PrototypeAPI};
    for (const SchemaId schema : schemas) {
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t index = 0; index < columns.size(); ++index) {
            inputs.push_back({schema, &columns[index], index});
        }
    }
    return inputs;
}

}

std::span<const ColumnRef> LightTable::Inputs() const {
    static const std::vector<ColumnRef> inputs = LightInputs();
    return inputs;
}

void LightTable::Describe(RenderTables& tables) const {
    tables.lights.address = m_layout ? m_layout->Address() : 0;
    tables.collections.address = m_collections.Address();
    tables.lightShapes.address = m_shapes.Address();
}

void LightTable::Hold(std::vector<Ref<const RefCounted>>& held) const {
    HoldBuffers(held, {m_layout, m_records, m_cdf, m_maps, m_spectra, m_collections.Buffer(),
                       m_shapes.Buffer()});
}

void LightTable::Prepare() {
    Kernels();
    Spectra();
}

const Ref<device::Buffer>& LightTable::Spectra() {
    if (!m_spectra) {
        const std::vector<SpectralTables> tables = {BuildSpectralTables()};
        m_spectra = m_structure.GetColumnStore().GetDevice().CreateBuffer(
            {.label = "spectral tables"}, tables);
    }
    return m_spectra;
}

bool LightTable::Restructured(const ChangeSet& changes) const {
    if (m_structure.GetPrims().WorldCount() != m_worlds || changes.Wrote(TableId::Instances)) {
        return true;
    }
    for (const StructureEntry& entry : changes.removed) {
        if (m_lights.contains(entry.prim) || m_worldPrims.contains(entry.prim)) {
            return true;
        }
    }
    const Prims& prims = m_structure.GetPrims();
    for (const StructureEntry& entry : changes.added) {
        if (prims.Alive(entry.prim) && prims.Generation(entry.prim) == entry.generation &&
            (prims.HasApi(entry.prim, SchemaId::LightAPI) ||
             IsA(prims.Record(entry.prim).schema, SchemaId::World))) {
            return true;
        }
    }
    return std::ranges::any_of(
        changes.apis, [](const StructureEntry& entry) { return entry.api == SchemaId::LightAPI; });
}

void LightTable::Relayout() {
    ERNIE_ZONE("Lights/Relayout");
    const Prims& prims = m_structure.GetPrims();
    device::Device& device = m_structure.GetColumnStore().GetDevice();
    m_worlds = prims.WorldCount();
    m_lights.clear();
    m_worldPrims.clear();
    std::vector<std::vector<uint32_t>> lists(m_worlds + 1);
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim)) {
            continue;
        }
        const PrimRecord& record = prims.Record(prim);
        if (IsA(record.schema, SchemaId::World)) {
            m_worldPrims.insert(prim);
        }
        if (HasApi(record.apis, SchemaId::LightAPI)) {
            m_lights.insert(prim);
            if (!m_instances.Members().contains(prim)) {
                lists[TlasIndex(record.world, m_worlds)].push_back(prim);
            }
        }
    }
    std::vector<uint32_t> domes;
    for (const uint32_t prim : m_lights) {
        if (prims.Record(prim).schema == SchemaId::DomeLight) {
            domes.push_back(prim);
        }
    }
    m_domes = uint32_t(domes.size());
    std::vector<uint32_t> items;
    std::vector<uint32_t> copies;
    std::vector<uint32_t> starts = {0};
    m_counts.clear();
    m_compositions = m_structure.CompositionVersion();
    m_shownIn.clear();
    const std::map<uint32_t, ComposedWorldDesc>& composed = m_structure.Compositions();
    for (uint32_t tlas = 0; tlas <= m_worlds; ++tlas) {
        items.insert(items.end(), lists[tlas].begin(), lists[tlas].end());
        const auto composition = tlas < m_worlds ? composed.find(tlas) : composed.end();
        const bool stage = composition == composed.end() || composition->second.stageLights;
        if (composition != composed.end() && stage) {
            for (const uint32_t world : composition->second.worlds) {
                if (world < m_worlds && world != tlas) {
                    items.insert(items.end(), lists[world].begin(), lists[world].end());
                    m_shownIn[world].insert(tlas);
                }
            }
        }
        if (tlas < m_worlds && stage) {
            items.insert(items.end(), lists[m_worlds].begin(), lists[m_worlds].end());
        }
        for (const auto& [block, offset] : m_instances.LightCopies(tlas)) {
            items.push_back(kInstancerSlot | uint32_t(copies.size() / 2));
            copies.push_back(block);
            copies.push_back(offset);
        }
        starts.push_back(uint32_t(items.size()));
        m_counts.push_back(starts.back() - starts[starts.size() - 2]);
    }

    m_records = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(items.size(), 1) * sizeof(LightRecord), .label = "lights"});
    m_cdf = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(items.size(), 1) * sizeof(float), .label = "light cdf"});
    std::vector<uint32_t> words(sizeof(LightLayout) / sizeof(uint32_t));
    const auto append = [&](std::span<const uint32_t> values) {
        const uint64_t at = words.size() * sizeof(uint32_t);
        words.insert(words.end(), values.begin(), values.end());
        return at;
    };
    const uint64_t first = append(starts);
    const uint64_t listed = append(items);
    std::vector<uint32_t> slots(items.size(), kNoRow);
    for (size_t slot = 0; slot < items.size(); ++slot) {
        const auto dome = std::ranges::find(domes, items[slot]);
        if (dome != domes.end()) {
            slots[slot] = uint32_t(dome - domes.begin());
        }
    }
    const uint64_t domePrims = append(domes);
    const uint64_t domeSlots = append(slots);
    std::vector<uint32_t> bases(m_worlds + 1, kNoRow);
    std::vector<Float3> shifts;
    for (const auto& [world, desc] : composed) {
        if (world >= m_worlds) {
            continue;
        }
        bases[world] = uint32_t(shifts.size());
        shifts.resize(shifts.size() + m_worlds, Float3{0, 0, 0});
        for (size_t at = 0; at < desc.worlds.size() && at < desc.offsets.size(); ++at) {
            if (desc.worlds[at] < m_worlds) {
                const Double3& offset = desc.offsets[at];
                shifts[bases[world] + desc.worlds[at]] = {float(offset.x), float(offset.y),
                                                          float(offset.z)};
            }
        }
    }
    const uint64_t copiesStart = append(copies);
    const uint64_t offsetBases = append(bases);
    const uint64_t offsets =
        append(std::span(reinterpret_cast<const uint32_t*>(shifts.data()), shifts.size() * 3));
    m_maps = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(m_domes, 1) * kDomeMapFloats * sizeof(float),
         .label = "dome maps"});
    m_layout =
        device.CreateBuffer({.bytes = words.size() * sizeof(uint32_t), .label = "light layout"});
    LightLayout layout{};
    layout.records.address = m_records->Address();
    layout.firstLights.address = m_layout->Address() + first;
    layout.prims.address = m_layout->Address() + listed;
    layout.cdf.address = m_cdf->Address();
    layout.maps.address = m_maps->Address();
    layout.domePrims.address = m_layout->Address() + domePrims;
    layout.domeSlots.address = m_layout->Address() + domeSlots;
    layout.offsetBases.address = m_layout->Address() + offsetBases;
    layout.offsets.address = m_layout->Address() + offsets;
    layout.copies.address = m_layout->Address() + copiesStart;
    layout.worldCount = m_worlds;
    layout.domeCount = m_domes;
    std::memcpy(words.data(), &layout, sizeof(layout));
    device.GetQueue().Write(*m_layout, words.data());
}

void LightTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Lights/Run");
    m_written = 0;
    const bool relaid = changes.resync || !m_layout || Restructured(changes) ||
                        m_structure.CompositionVersion() != m_compositions;
    if (relaid) {
        Relayout();
    }
    ColumnStore& store = m_structure.GetColumnStore();
    WriteCollectionsAndShapes();
    const bool viewed = store.LastWrite(kRenderProductCamera) > m_seen;
    m_seen = store.WriteSequence();
    std::vector<uint32_t> dirty;
    const bool global = relaid || viewed || changes.Global();
    std::set<uint32_t> shown;
    for (const uint32_t world : changes.worlds) {
        if (const auto found = m_shownIn.find(world); found != m_shownIn.end()) {
            shown.insert(found->second.begin(), found->second.end());
        }
    }
    for (uint32_t tlas = 0; tlas <= m_worlds; ++tlas) {
        const bool touched =
            global || shown.contains(tlas) || std::ranges::binary_search(changes.worlds, tlas);
        if (touched && m_counts[tlas] > 0) {
            dirty.push_back(tlas);
        }
    }
    if (dirty.empty()) {
        return;
    }
    uint32_t widest = 0;
    for (const uint32_t tlas : dirty) {
        widest = std::max(widest, m_counts[tlas]);
        m_written += m_counts[tlas];
    }

    device::Device& device = store.GetDevice();
    const Ref<device::Buffer> tlases = device.CreateBuffer({.label = "dirty light lists"}, dirty);
    LightParams params{};
    params.scene.address = store.DeviceData();
    params.layout.address = m_layout->Address();
    params.instances = m_tables.instances;
    params.origins = m_tables.origins;
    params.transforms = m_tables.transforms;
    params.imageables = m_tables.imageables;
    params.textures = m_tables.textures;
    params.textureBindings = m_tables.textureBindings;
    params.tlases.address = tlases->Address();
    params.spectra.address = Spectra()->Address();
    params.tlasCount = uint32_t(dirty.size());
    params.lightLink = m_structure.GetTokens().Intern("lightLink").id;
    params.shadowLink = m_structure.GetTokens().Intern("shadowLink").id;
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    if (m_domes > 0) {
        encoder.Dispatch(Kernels().Entry("WriteDomeMaps"), params,
                         {.x = device::GroupCount(kDomeMapCells, kLightGroup), .y = m_domes});
        encoder.Barrier();
        encoder.Dispatch(Kernels().Entry("WriteDomeRows"), params,
                         {.x = device::GroupCount(kDomeMapHeight, kLightGroup), .y = m_domes});
        encoder.Barrier();
        encoder.Dispatch(Kernels().Entry("WriteDomeMarginals"), params,
                         {.x = device::GroupCount(m_domes, kLightGroup)});
        encoder.Barrier();
    }
    encoder.Dispatch(Kernels().Entry("WriteLights"), params,
                     {.x = device::GroupCount(widest, kLightGroup), .y = uint32_t(dirty.size())});
    encoder.Barrier();
    encoder.Dispatch(Kernels().Entry("WriteLightCdf"), params,
                     {.x = device::GroupCount(uint32_t(dirty.size()), kLightGroup)});
    queue.Submit(encoder.Finish());
}

void LightTable::WriteCollectionsAndShapes() {
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    LightParams params{};
    params.collectionRows = store.Schema(SchemaId::CollectionAPI).Rows();
    params.shapeRows = store.Schema(SchemaId::LightAPI).Rows();
    m_collections.Reserve(device, store.Schema(SchemaId::CollectionAPI).Capacity());
    m_shapes.Reserve(device, store.Schema(SchemaId::LightAPI).Capacity());
    params.scene.address = store.DeviceData();
    params.collections.address = m_collections.Address();
    params.shapes.address = m_shapes.Address();
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    if (params.collectionRows > 0) {
        encoder.Dispatch(Kernels().Entry("WriteCollections"), params,
                         {.x = device::GroupCount(params.collectionRows, kLightGroup)});
    }
    if (params.shapeRows > 0) {
        encoder.Dispatch(Kernels().Entry("WriteLightShapes"), params,
                         {.x = device::GroupCount(params.shapeRows, kLightGroup)});
    }
    queue.Submit(encoder.Finish());
}

device::Program& LightTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.lights"},
             .library = &kSceneKernels,
             .entries = {"WriteDomeMaps", "WriteDomeRows", "WriteDomeMarginals", "WriteLights",
                         "WriteLightCdf", "WriteCollections", "WriteLightShapes"},
             .label = "lights"});
    }
    return *m_program;
}

}
