#pragma once

#include "ernie/scene/sensor_filters.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

/// The sensor pass of a frame: the filters of its chain that its held vars need, fused into one
/// compute kernel that writes only those vars. Every filter after the lens works on one pixel, so
/// the bands stream from the lens straight into the integrators without SpectralRadiance or any
/// other intermediate image; a spatial filter later in a chain would start a kernel of its own,
/// reading what the one before wrote.
///
/// References:
/// - J. Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing Parallelism,
///   Locality, and Recomputation in Image Processing Pipelines", PLDI 2013 (fusing pointwise
///   stages and materialising only at stencils). https://doi.org/10.1145/2491956.2462176
struct SensorPlan {
    /// A filter that runs: a built-in, or a custom one with its parameters at `firstParam`.
    struct Stage {
        std::optional<SensorFilter> builtin;
        std::optional<SensorFilterModel> custom;
        std::uint32_t firstParam = 0;
    };

    std::vector<Stage> stages;          ///< In chain order, the lens first.
    std::vector<SensorFilter> filters;  ///< The built-in ones among them.
    std::vector<float> params;          ///< Every custom stage's parameters, packed in order.
    std::uint32_t holds = 0;            ///< Vars written, a bit per `kRenderVarNames` index.

    /// Identifies the kernel: plans with the same key share it (parameter values aside, which the
    /// kernel reads from a buffer).
    std::string Key() const;

    /// The Slang source of the kernel's entry point `sensorMain`, importing module `sensor` and
    /// each custom filter's module (`Modules`).
    std::string Source() const;

    /// The custom filters' modules as (name, source), for the shader compiler.
    std::vector<std::pair<std::string, std::string>> Modules() const;
};

/// The plan for `chain` (`CheckFilterChain`, custom filters from `registry`) writing `held`
/// (render var names): the lens, QE and the filters on electrons when Electrons or RawDN are
/// held, ADC and the filters on its numbers for RawDN, CIE and the filters on linear colour for
/// HdrColor or LdrColor, exposure and the filters on display colour for LdrColor, and band
/// filters whenever the bands reach a held var. A custom filter's parameters are its step's
/// inputs, or their defaults.
SensorPlan PlanSensor(std::span<const SensorFilterStep> chain,
                      std::span<const std::string_view> held, const SensorFilterRegistry& registry);

/// `PlanSensor` of a chain of built-in filters.
SensorPlan PlanSensor(std::span<const SensorFilter> chain, std::span<const std::string_view> held);

}  // namespace ernie
