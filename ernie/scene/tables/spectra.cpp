// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/spectra.h"

#include <algorithm>
#include <cstring>

#include "ernie/material/uplift.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

constexpr float kVisibleFirst = 380;
constexpr float kVisibleLast = 780;
constexpr float kBasisFirst = 402.5f;
constexpr float kBasisStep = 45;
constexpr Float3 kBasis[] = {{0.05f, 0.00f, 0.60f}, {0.00f, 0.05f, 1.00f}, {0.00f, 0.45f, 0.70f},
                             {0.05f, 1.00f, 0.15f}, {0.45f, 0.85f, 0.00f}, {1.00f, 0.35f, 0.00f},
                             {0.80f, 0.05f, 0.00f}, {0.25f, 0.00f, 0.00f}};

Float3 Basis(float nanometers) {
    const float at = std::clamp((nanometers - kBasisFirst) / kBasisStep, 0.0f, 7.0f);
    const uint32_t below = std::min(uint32_t(at), 6u);
    const float t = at - float(below);
    const Float3& a = kBasis[below];
    const Float3& b = kBasis[below + 1];
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
}

float Sample(std::span<const Float2> samples, float nanometers) {
    if (samples.empty()) {
        return 0;
    }
    if (nanometers <= samples.front().x) {
        return samples.front().y;
    }
    if (nanometers >= samples.back().x) {
        return samples.back().y;
    }
    const auto above = std::ranges::upper_bound(samples, nanometers, {}, &Float2::x);
    const Float2& high = *above;
    const Float2& low = *(above - 1);
    const float t = (nanometers - low.x) / std::max(high.x - low.x, 1e-6f);
    return low.y + (high.y - low.y) * t;
}

float Average(std::span<const Float2> samples, float low, float high) {
    constexpr uint32_t kSteps = 64;
    float sum = 0;
    for (uint32_t step = 0; step < kSteps; ++step) {
        sum += Sample(samples, low + (float(step) + 0.5f) * (high - low) / kSteps);
    }
    return sum / kSteps;
}

Float3 Rgb(std::span<const Float2> samples) {
    Float3 sum{};
    Float3 weight{};
    for (float nanometers = kVisibleFirst + 0.5f; nanometers < kVisibleLast; nanometers += 1) {
        const Float3 basis = Basis(nanometers);
        const float value = Sample(samples, nanometers);
        sum = {sum.x + basis.x * value, sum.y + basis.y * value, sum.z + basis.z * value};
        weight = {weight.x + basis.x, weight.y + basis.y, weight.z + basis.z};
    }
    return {sum.x / weight.x, sum.y / weight.y, sum.z / weight.z};
}

std::vector<float> Floats(ColumnStore& store, ColumnRef column, uint32_t row) {
    const ArrayRef array = store.Array(column, row);
    if (array.address == 0) {
        return {};
    }
    return Items<float>(store.Arrays(ColumnType::FloatArray).Read(array));
}

}

std::span<const ColumnRef> SpectralTable::Inputs() const {
    static const ColumnRef kInputs[] = {kSampledSpectrumWavelengths, kSampledSpectrumValues,
                                        kDetectorAPIDetectorSpectralRange};
    return kInputs;
}

void SpectralTable::Run(const ChangeSet&, TableVersions&) {
    ERNIE_ZONE("Spectra/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    const Prims& prims = m_structure.GetPrims();
    ColumnTable& spectra = store.Schema(SchemaId::SampledSpectrum);
    ColumnTable& detectors = store.Schema(SchemaId::DetectorAPI);
    m_count = spectra.Rows();
    m_detectors = detectors.Rows();
    const std::vector<Float2> ranges =
        Items<Float2>(detectors.Read(kDetectorAPIDetectorSpectralRange.index, 0, detectors.Rows()));
    const std::vector<uint32_t> owners =
        Items<uint32_t>(spectra.Read(store.PrimsColumn(SchemaId::SampledSpectrum), 0, m_count));
    std::vector<Float3> rgb(std::max(m_count, 1u));
    std::vector<float> bands(std::max<size_t>(size_t(m_count) * m_detectors * kDetectorBands, 1));
    std::vector<Uint2> spans(std::max(m_count, 1u));
    std::vector<Float2> samples;
    for (uint32_t row = 0; row < m_count; ++row) {
        if (owners[row] == kNoPrim || !prims.Alive(owners[row])) {
            continue;
        }
        const std::vector<float> wavelengths = Floats(store, kSampledSpectrumWavelengths, row);
        const std::vector<float> values = Floats(store, kSampledSpectrumValues, row);
        std::vector<Float2> curve;
        bool increasing = true;
        for (size_t i = 0; i < std::min(wavelengths.size(), values.size()); ++i) {
            increasing = increasing && (curve.empty() || wavelengths[i] > curve.back().x);
            curve.push_back({wavelengths[i], values[i]});
        }
        if (wavelengths.size() != values.size() || !increasing) {
            Log(LogLevel::Warning,
                "{} needs increasing wavelengths and one value for each; it reads as zero",
                m_structure.GetPaths().Path(owners[row]));
            curve.clear();
        }
        spans[row] = {uint32_t(samples.size()), uint32_t(curve.size())};
        samples.insert(samples.end(), curve.begin(), curve.end());
        rgb[row] = Rgb(curve);
        for (uint32_t detector = 0; detector < m_detectors; ++detector) {
            const float first = std::max(ranges[detector].x, 0.0f);
            const float width =
                std::max((ranges[detector].y - first) / float(kDetectorBands), 1e-3f);
            for (uint32_t band = 0; band < kDetectorBands; ++band) {
                bands[(size_t(detector) * m_count + row) * kDetectorBands + band] =
                    Average(curve, first + float(band) * width, first + float(band + 1) * width);
            }
        }
    }
    if (samples.empty()) {
        samples.push_back({});
    }
    if (!m_sigmoids) {
        const SigmoidTable& sigmoids = Sigmoids();
        m_sigmoids = device.CreateBuffer({.label = "sigmoid uplift"}, sigmoids.coefficients);
        m_sigmoidScales = device.CreateBuffer({.label = "sigmoid scales"}, sigmoids.scales);
    }
    m_rgb = device.CreateBuffer({.label = "spectra rgb"}, rgb);
    m_bands = device.CreateBuffer({.label = "spectra bands"}, bands);
    m_ranges = device.CreateBuffer({.label = "spectra ranges"}, spans);
    m_samples = device.CreateBuffer({.label = "spectra samples"}, samples);
    Log(LogLevel::Debug, "projected {} spectra onto {}", m_count,
        Counted(m_detectors, "detector band set"));
}

void SpectralTable::Describe(RenderTables& tables) const {
    tables.spectra.rgb.address = m_rgb ? m_rgb->Address() : 0;
    tables.spectra.bands.address = m_bands ? m_bands->Address() : 0;
    tables.spectra.ranges.address = m_ranges ? m_ranges->Address() : 0;
    tables.spectra.samples.address = m_samples ? m_samples->Address() : 0;
    tables.spectra.sigmoids.address = m_sigmoids ? m_sigmoids->Address() : 0;
    tables.spectra.sigmoidScales.address = m_sigmoidScales ? m_sigmoidScales->Address() : 0;
    tables.spectra.sigmoidResolution = m_sigmoids ? kSigmoidResolution : 0;
    tables.spectra.count = m_count;
    tables.spectra.detectors = m_detectors;
}

}
