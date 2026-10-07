#pragma once

#include "ernie/scene/sensor_filters.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// How a sensor filter runs in a fused kernel (sensor.slang):
/// - `Spatial`: reads whole images and produces each band of its pixel (`ISpatialSource`); it heads
///   a kernel, its input written to memory by what came before. The lens.
/// - `Band`: on each band's radiance, before integration (`IBandFilter`).
/// - `Integrator`: folds the bands into channels (`IIntegrator`): QE into electrons, CIE into
///   linear colour.
/// - `Pixel`: on one pixel's channels, in chain order (`IPixelFilter`): noise, ADC, exposure.
enum class FilterKind : std::uint8_t { Spatial, Band, Integrator, Pixel };

/// The kind of a built-in filter.
FilterKind KindOf(SensorFilter filter);

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
    std::vector<SensorFilter> filters;  ///< That run, in chain order, the lens first.
    std::uint32_t holds = 0;            ///< Vars written, a bit per `kRenderVarNames` index.

    /// Identifies the kernel: plans with the same key share it.
    std::string Key() const;

    /// The Slang source of the kernel's entry point `sensorMain`, importing module `sensor`.
    std::string Source() const;
};

/// The plan for `chain` (`CheckFilterChain`) writing `held` (render var names): the lens, QE and
/// its noise when Electrons or RawDN are held, ADC for RawDN, CIE for HdrColor or LdrColor and
/// exposure for LdrColor, each only if in `chain`.
SensorPlan PlanSensor(std::span<const SensorFilter> chain, std::span<const std::string_view> held);

}  // namespace ernie
