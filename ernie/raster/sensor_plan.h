#pragma once

#include "ernie/scene/camera.h"
#include "ernie/scene/sensor_filters.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

/// The sensor pass of a frame: the filters of its chain that its held vars need, fused into as few
/// compute kernels ("segments") as its spatial filters allow, writing only those vars. Filters that
/// work on one pixel fuse: the bands stream from the lens straight into the integrators without
/// SpectralRadiance or any other intermediate image. A custom spatial filter reads its domain at
/// the pixels around its own, so it starts a segment: the one before stores that domain to an
/// image (`spectrumImages`, `channelImages`), and the filter heads the next, followed by what comes
/// after it on its branch of the chain (the spectrum before the integrators, then electrons and
/// digital numbers, or linear and display colour). A chain without one is a single kernel.
///
/// References:
/// - J. Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing Parallelism,
///   Locality, and Recomputation in Image Processing Pipelines", PLDI 2013 (fusing pointwise
///   stages and materialising only at stencils). https://doi.org/10.1145/2491956.2462176
struct SensorPlan {
    /// A filter that runs: a built-in, or a custom one with its parameters at `firstParam`, in
    /// kernel `segment`.
    struct Stage {
        std::optional<SensorFilter> builtin;
        std::optional<SensorFilterModel> custom;
        std::uint32_t firstParam = 0;
        std::uint32_t segment = 0;
        std::uint32_t outputBase = 0;  ///< The AOV slot of its output 0 (`AovOutputsOf`).
        std::uint32_t stateBase = 0;   ///< The state slot of its state 0.
    };

    std::vector<Stage> stages;          ///< In chain order, the lens first.
    std::vector<SensorFilter> filters;  ///< The built-in ones among them.
    std::vector<float> params;          ///< Every custom stage's parameters, packed in order.
    std::uint32_t holds = 0;            ///< Vars written, a bit per `kRenderVarNames` index.
    std::uint32_t aovs = 0;             ///< AOV slots written, a bit per `kAovSlotNames` index.
    /// State slots per pixel the temporal filters that run keep between frames, in their order.
    std::uint32_t states = 0;
    std::uint32_t segments = 1;         ///< Kernels, run in order with a barrier between.
    bool spectrumImages = false;  ///< Whether segments pass the spectrum on (spectral spatial).
    bool channelImages = false;   ///< Whether they pass other domains on.
    /// Segments whose spatial filters read their input's pyramid (`SensorFilterModel::pyramid`),
    /// a bit each: the renderer builds the channel images' pyramid before them.
    std::uint32_t pyramids = 0;
    /// The lens head's projection, a struct of lens.slang: "Pinhole", "BrownConrady", "Fisheye"
    /// or "Pushbroom" (`LensKernelOf`). Compiled into the kernel, so a pinhole runs no distortion
    /// code.
    std::string lens = "Pinhole";
    /// Whether the camera is a pushbroom, whose pixels' rays come from module `projection`'s
    /// pushbroom strip (projection_pushbroom.slang) rather than one view-projection.
    bool pushbroom = false;
    /// Whether the camera is rendered on cube faces (`NeedsCubeFaces`): the lens follows each
    /// pixel's ray (`lensRayOf`) onto its face's layer rather than through one pinhole image.
    bool cube = false;
    /// Whether the lens is a table of rays (`LensModel::rays`), which the kernel interpolates in
    /// place of its projection and distortion.
    bool table = false;

    /// Identifies the kernels: plans with the same key share them (parameter values aside, which
    /// the kernels read from a buffer).
    std::string Key() const;

    /// The Slang source of kernel `segment`'s entry point `sensorMain`, importing module `sensor`
    /// and its custom filters' modules (`Modules`). Segment `s` reads the images segment `s - 1`
    /// stored (sensor.slang's bindings 16 and 17) and stores those segment `s + 1` reads (18, 19).
    std::string Source(std::uint32_t segment = 0) const;

    /// The Slang source of module `lensProjection`, which module `sensor` imports: `lensPinholeOf`,
    /// `lensRayOf` and `lensFalloff` of projection `lens`, from module `lens`, and `kLensCube`.
    std::string LensSource() const;

    /// The custom filters' modules as (name, source), for the shader compiler.
    std::vector<std::pair<std::string, std::string>> Modules() const;
};

/// The plan for `chain` (`CheckFilterChain`, custom filters from `registry`) writing `held`
/// (render var names): the lens, QE and the filters on electrons when Electrons or RawDN are
/// held, ADC and the filters on its numbers for RawDN, CIE and the filters on linear colour for
/// HdrColor or LdrColor, exposure and the filters on display colour for LdrColor, and band
/// filters whenever the bands reach a held var. A custom filter's parameters are its step's
/// inputs, or their defaults. The chain's AOVs take slots from `aovOffset` on (a product's later
/// chains after its earlier ones').
SensorPlan PlanSensor(std::span<const SensorFilterStep> chain,
                      std::span<const std::string_view> held, const SensorFilterRegistry& registry,
                      std::uint32_t aovOffset = 0);

/// The lens projection of lens.slang that `lens` compiles into the sensor kernel
/// (`SensorPlan::lens`): "Pinhole" without distortion, "BrownConrady" with it, or "Fisheye"; for
/// a `pushbroom` camera, "Pushbroom" (its distortion across the line only).
std::string LensKernelOf(const LensModel& lens, bool pushbroom = false);

/// `PlanSensor` of a chain of built-in filters.
SensorPlan PlanSensor(std::span<const SensorFilter> chain, std::span<const std::string_view> held);

}  // namespace ernie
