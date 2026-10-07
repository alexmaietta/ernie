#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// Render var names, after Omniverse's where it has the quantity.
namespace render_vars {
inline constexpr std::string_view kLdrColor = "LdrColor";  ///< uint8 sRGB (rows, cols, 4).
inline constexpr std::string_view kHdrColor = "HdrColor";  ///< float32 linear sRGB, cd/m^2.
/// float32 spectral radiance at each sensor pixel, W m^-2 sr^-1 nm^-1, (bands, rows, cols).
inline constexpr std::string_view kSpectralRadiance = "SpectralRadiance";
inline constexpr std::string_view kElectrons = "Electrons";  ///< float32 per sensor channel.
inline constexpr std::string_view kRawDn = "RawDN";          ///< uint16 per sensor channel.
inline constexpr std::string_view kDistanceToImagePlane = "DistanceToImagePlane";  ///< float32.
inline constexpr std::string_view kDistanceToCamera = "DistanceToCamera";          ///< float32.
inline constexpr std::string_view kNormal = "Normal";  ///< float32 world shading normal (xyz).
inline constexpr std::string_view kPrimId = "PrimId";  ///< uint32, 0 for background.
inline constexpr std::string_view kInstanceId = "InstanceId";  ///< uint32 instance index.
}  // namespace render_vars

/// Every render var name, in the order the rasterizer binds their images.
inline constexpr std::array<std::string_view, 10> kRenderVarNames{
    render_vars::kSpectralRadiance, render_vars::kElectrons, render_vars::kRawDn,
    render_vars::kHdrColor,         render_vars::kLdrColor,  render_vars::kDistanceToImagePlane,
    render_vars::kDistanceToCamera, render_vars::kNormal,    render_vars::kPrimId,
    render_vars::kInstanceId};

/// The built-in sensor filters a render product runs in order on its spectral radiance, as
/// Chrono::Sensor's filter lists; the rasterizer fuses a chain into one kernel (`PlanSensor`), so
/// the vars below are written only when held:
/// - `Lens`: resamples the radiance into sensor pixels through the inverse lens distortion, with
///   cos^4 vignetting; writes SpectralRadiance and the geometry vars. Always runs first.
/// - `Qe`: mean electrons per channel (camera equation, quantum efficiency, dark current); writes
///   Electrons.
/// - `Noise`: shot, read, PRNU and DSNU noise on Electrons.
/// - `Adc`: digital numbers from Electrons; writes RawDN.
/// - `Cie`: linear sRGB in cd/m^2 through the CIE observer; writes HdrColor.
/// - `Exposure`: the photographic output, HDR colour times the exposure scale, sRGB-encoded;
///   writes LdrColor.
enum class SensorFilter : std::uint8_t { Lens, Qe, Noise, Adc, Cie, Exposure };

inline constexpr std::array<SensorFilter, 6> kSensorFilters{
    SensorFilter::Lens, SensorFilter::Qe,  SensorFilter::Noise,
    SensorFilter::Adc,  SensorFilter::Cie, SensorFilter::Exposure};

/// The filter's name, as `ernie:filter:type` tokens spell it: "lens", "qe", "noise", "adc", "cie"
/// or "exposure".
std::string_view NameOf(SensorFilter filter);

/// The filter named `name`, if there is one.
std::optional<SensorFilter> SensorFilterNamed(std::string_view name);

/// Every filter in its natural order: the sensor chain then the photographic chain.
std::vector<SensorFilter> DefaultFilterChain();

/// `chain` with `Lens` prepended when it does not start with it.
/// @throws std::invalid_argument if a filter repeats or runs before what it reads (`Noise` and
/// `Adc` need `Qe` earlier, `Exposure` needs `Cie` earlier).
std::vector<SensorFilter> CheckFilterChain(std::span<const SensorFilter> chain);

/// The render vars `chain` writes.
std::vector<std::string_view> VarsWrittenBy(std::span<const SensorFilter> chain);

/// How a sensor filter runs in the fused sensor kernel (sensor.slang):
/// - `Spatial`: reads whole images and produces each band of its pixel (`ISpatialSource`); it heads
///   a kernel. The lens.
/// - `Band`: on each band's radiance, before integration (`IBandFilter`).
/// - `Integrator`: folds the bands into channels (`IIntegrator`): QE into electrons, CIE into
///   linear colour.
/// - `Pixel`: on one pixel's values of one domain, in chain order (`IPixelFilter`).
enum class FilterKind : std::uint8_t { Spatial, Band, Integrator, Pixel };

/// What a filter works on (sensor.slang's `SensorPixel`): each band's radiance (`Spectrum`), the
/// channels' electrons (after QE), their digital numbers (after ADC), linear sRGB (after CIE) or
/// the sRGB-encoded display colour (after exposure).
enum class FilterDomain : std::uint8_t { Spectrum, Electrons, Digital, Rgb, Display };

/// The kind of a built-in filter.
FilterKind KindOf(SensorFilter filter);

/// A custom filter's parameter: a float its prim sets as `inputs:<name>`.
struct SensorFilterParam {
    std::string name;  ///< An identifier: the field of the filter's `Params` struct.
    float defaultValue = 0.0f;
    std::string description;
};

/// A sensor filter written in Slang, run in the fused sensor kernel like the built-in ones: a
/// struct `typeName` in module `slang` implementing `IBandFilter` (kind `Band`, domain `Spectrum`)
/// or `IPixelFilter` (kind `Pixel`, on one of the other domains), with a nested `Params` struct of
/// one float field per parameter, in order. The module imports `sensor` for the interfaces,
/// `SensorPixel` and `PixelContext`.
///
///     import sensor;
///     struct HotPixels : IPixelFilter {
///         struct Params { float fraction; float electrons; }
///         static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
///             if (pixel.fixedRandom(0x407) < params.fraction) {
///                 value.electrons += params.electrons;
///             }
///         }
///     }
struct SensorFilterModel {
    std::string id;  ///< `ernie:filter:type` of its prims; an identifier, not a built-in's name.
    std::string name;
    std::string description;
    std::vector<std::string> references;
    FilterKind kind = FilterKind::Pixel;  ///< `Band` or `Pixel`.
    FilterDomain domain = FilterDomain::Electrons;
    std::vector<SensorFilterParam> params;
    std::string typeName;  ///< The Slang struct implementing the interface.
    std::string slang;     ///< Its module's source.
};

/// Custom sensor filters a renderer can run, by `SensorFilterModel::id`.
class SensorFilterRegistry {
  public:
    /// Most parameters a filter may declare.
    static constexpr std::size_t kMaxParams = 16;

    /// Adds `model`; its Slang is compiled when a chain first uses it (`RasterRenderer::
    /// AddSensorFilter` compiles eagerly).
    /// @throws std::invalid_argument if the id is not an identifier, names a built-in filter or is
    /// taken; the kind is not `Band` or `Pixel`, or the domain does not suit it; there are more
    /// than `kMaxParams` parameters or one is not an identifier or repeats; or the type name is
    /// not an identifier or the source does not mention it.
    void Add(SensorFilterModel model);

    /// The filter with id `id`, or null.
    const SensorFilterModel* Find(std::string_view id) const;

    std::span<const SensorFilterModel> Models() const {
        return models_;
    }

  private:
    std::vector<SensorFilterModel> models_;
};

/// One filter of a product's chain: a SensorFilter prim's `ernie:filter:type`, a built-in's name
/// or a registered filter's id, with the floats of its `inputs:*` attributes by name (without
/// "inputs:").
struct SensorFilterStep {
    std::string type;
    std::map<std::string, float, std::less<>> inputs;

    /// The built-in filter it names, if any.
    std::optional<SensorFilter> Builtin() const {
        return SensorFilterNamed(type);
    }

    bool operator==(const SensorFilterStep&) const = default;
};

/// Steps of the built-in `chain`.
std::vector<SensorFilterStep> StepsOf(std::span<const SensorFilter> chain);

/// `chain` with the lens prepended when it does not start with it, its custom filters looked up
/// in `registry`.
/// @throws std::invalid_argument as the built-in `CheckFilterChain`; for a type neither built in
/// nor in `registry`; or for a custom filter before the domain it works on exists (`Electrons`
/// after `qe`, `Digital` after `adc`, `Rgb` after `cie`, `Display` after `exposure`) or, for a
/// band filter, after `qe` or `cie`, which integrate the bands.
std::vector<SensorFilterStep> CheckFilterChain(std::span<const SensorFilterStep> chain,
                                               const SensorFilterRegistry& registry);

/// The render vars a frame holds to produce `requested`: those, in `kRenderVarNames` order, as
/// the sensor keeps what they are made from (SpectralRadiance, Electrons, HdrColor) in registers.
/// Empty `requested` means every var.
std::vector<std::string_view> RequiredVars(std::span<const std::string_view> requested);

}  // namespace ernie
