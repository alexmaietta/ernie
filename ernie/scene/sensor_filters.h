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

/// Arbitrary output variables (AOVs): what sensor filters write besides the render vars, such as
/// a mosaic before demosaicing or the charge blooming spilled. A filter model declares its outputs
/// (`SensorFilterModel::outputs`) and writes them with sensor.slang's `writeOutput`; a RenderVar
/// asks for one with `sourceName` "aov:<output>" and maps it as float32 (rows, cols, 4). A
/// product's chain lays its filters' outputs into `kMaxAovs` slots in chain order
/// (`AovOutputsOf`); a requested output's filter runs whatever else the product holds.
inline constexpr std::string_view kAovPrefix = "aov:";
inline constexpr std::size_t kMaxAovs = 8;
/// The slots' names inside a frame (its held vars and images).
inline constexpr std::array<std::string_view, kMaxAovs> kAovSlotNames{
    "Aov0", "Aov1", "Aov2", "Aov3", "Aov4", "Aov5", "Aov6", "Aov7"};

/// One output of a sensor filter.
struct SensorFilterOutput {
    std::string name;  ///< An identifier; render vars ask for "aov:<name>".
    std::string description;
};

/// A custom filter's parameter: a float its prim sets as `inputs:<name>`, or with `size` above 1
/// an array of floats (`float name[size]` in the filter's `Params`, such as a curve; sensor.slang's
/// `curveAt` reads one), which the prim sets as a float array of any length, resampled linearly
/// to `size` (a single number fills it).
struct SensorFilterParam {
    std::string name;  ///< An identifier: the field of the filter's `Params` struct.
    float defaultValue = 0.0f;
    std::string description;
    std::uint32_t size = 1;
    /// An array parameter's default, resampled to `size`; empty fills it with `defaultValue`.
    std::vector<float> defaults = {};
};

/// `values` resampled linearly to `size` evenly spaced samples, first to first and last to last;
/// one value fills them all, and none gives zeros.
std::vector<float> ResampleCurve(std::span<const float> values, std::size_t size);

/// A sensor filter written in Slang, run in the fused sensor kernel like the built-in ones: a
/// struct `typeName` in module `slang` implementing `IBandFilter` (kind `Band`, domain `Spectrum`),
/// `IPixelFilter` (kind `Pixel`, on one of the other domains), or for kind `Spatial`, which reads
/// the pixels around its own, `ISpatialBandFilter` (domain `Spectrum`) or `ISpatialPixelFilter`
/// (the other domains). It has a nested `Params` struct of one float field per parameter, in order.
/// The module imports `sensor` for the interfaces, `SensorPixel`, `PixelContext` and
/// `SpatialImage`. A spatial filter splits the kernel: what comes before it writes its domain to an
/// image, which it reads, heading a kernel of its own for what follows on its branch.
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
    FilterKind kind = FilterKind::Pixel;  ///< `Band`, `Pixel` or `Spatial`.
    FilterDomain domain = FilterDomain::Electrons;
    std::vector<SensorFilterParam> params;
    std::string typeName;  ///< The Slang struct implementing the interface.
    std::string slang;     ///< Its module's source.
    /// Its AOVs, which it writes with `writeOutput(pixel, index, value)` in this order.
    std::vector<SensorFilterOutput> outputs = {};
};

/// Custom sensor filters a renderer can run, by `SensorFilterModel::id`.
class SensorFilterRegistry {
  public:
    /// Most parameters a filter may declare, and floats they may hold together.
    static constexpr std::size_t kMaxParams = 16;
    static constexpr std::size_t kMaxParamFloats = 1024;

    /// Adds `model`; its Slang is compiled when a chain first uses it (`RasterRenderer::
    /// AddSensorFilter` compiles eagerly).
    /// @throws std::invalid_argument if the id is not an identifier, names a built-in filter or is
    /// taken; the kind is `Integrator`, or the domain does not suit it; there are more
    /// than `kMaxParams` parameters or one is not an identifier or repeats; or the type name is
    /// not an identifier or the source does not mention it.
    void Add(SensorFilterModel model);

    /// The filter with id `id`, or null.
    const SensorFilterModel* Find(std::string_view id) const;

    std::span<const SensorFilterModel> Models() const {
        return models_;
    }

    /// A registry of the standard filters (`StandardSensorFilters`), as renderers start with.
    static SensorFilterRegistry WithStandard();

  private:
    std::vector<SensorFilterModel> models_;
};

/// The standard library of sensor filters, written in Slang like custom ones and registered with
/// every renderer, for a SensorFilter prim's `ernie:filter:type`:
/// - `psf` (spatial, spectrum): the lens's point-spread, diffraction for the camera's f-number
///   per band (a Gaussian fit to the Airy disc) and a Gaussian `blur` in pixels; before `qe`.
/// - `defocus` (spatial, spectrum): thin-lens depth of field from the camera's `fStop` and
///   `focusDistance`; before `qe`.
/// - `bayer` (pixel, electrons): a Bayer colour filter mosaic, `pattern` 0 RGGB, 1 BGGR, 2 GRBG,
///   3 GBRG; best between `noise` and `adc`.
/// - `demosaic` (spatial, digital): interpolates the mosaic back, `method` 0 bilinear or
///   1 Malvar-He-Cutler; after `adc`. AOV `mosaic`: the mosaic it read.
/// - `blooming` (spatial, electrons): charge beyond the full well spilling along columns,
///   `antiBlooming` the fraction drained; after `qe`. AOV `overflow`: the charge beyond the well.
/// - `whiteBalance` (pixel, rgb): `red`, `green` and `blue` gains; after `cie`.
/// - `colorMatrix` (pixel, rgb): a 3 x 3 colour correction matrix `m00`..`m22`; after `cie`.
/// - `compand` (pixel, digital): square-root companding from `inputBits` to `outputBits`;
///   after `adc`.
/// - `toneCurve` (pixel, display): each channel through `curve` (33 samples over 0 to 1); after
///   `exposure`.
/// - `spectralTransmission` (band): each band times `curve` (48 samples, 360 to 830 nm), a
///   measured filter's transmission; before `qe` and `cie`.
/// Each file's header cites its references.
std::vector<SensorFilterModel> StandardSensorFilters();

/// One filter of a product's chain: a SensorFilter prim's `ernie:filter:type`, a built-in's name
/// or a registered filter's id, with the floats of its `inputs:*` attributes by name (without
/// "inputs:").
struct SensorFilterStep {
    std::string type;
    std::map<std::string, float, std::less<>> inputs;
    /// Its float-array `inputs:*`, for array parameters.
    std::map<std::string, std::vector<float>, std::less<>> arrays = {};

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

/// The AOVs `chain`'s custom filters write (`SensorFilterModel::outputs`), in slot order.
/// @throws std::invalid_argument as `CheckFilterChain`, for two outputs of one name, or more than
/// `kMaxAovs`.
std::vector<std::string> AovOutputsOf(std::span<const SensorFilterStep> chain,
                                      const SensorFilterRegistry& registry);

/// The render vars a frame holds to produce `requested`: those, in `kRenderVarNames` order, as
/// the sensor keeps what they are made from (SpectralRadiance, Electrons, HdrColor) in registers,
/// then the AOV slots (`kAovSlotNames`) requested. Empty `requested` means every render var.
std::vector<std::string_view> RequiredVars(std::span<const std::string_view> requested);

}  // namespace ernie
