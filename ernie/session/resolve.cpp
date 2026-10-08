#include "ernie/session/resolve.h"

#include "ernie/geometry/primitives.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/astronomy.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

bool InSession(const Path& path) {
    static const Path root{std::string(tokens::kSessionRoot)};
    return path.HasPrefix(root);
}

/// The first prim of `type` outside the session root, or the empty path.
Path FirstOutsideSession(const Scene& scene, std::string_view type) {
    for (const Path& path : scene.Prims(type)) {
        if (!InSession(path)) {
            return path;
        }
    }
    return {};
}

/// The targets of the relationship `attribute`, one path or a list; none when unset.
std::optional<std::vector<Path>> RelationshipOf(const Scene& scene, const Path& prim,
                                                std::string_view attribute) {
    const Value* value = prim.IsEmpty() ? nullptr : scene.Get(prim, attribute);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (const Path* one = value->Get<Path>()) {
        return std::vector<Path>{*one};
    }
    if (const std::vector<Path>* many = value->Get<std::vector<Path>>()) {
        return *many;
    }
    throw std::invalid_argument("'" + std::string(attribute) + "' of '" + prim.String() +
                                "' is not a relationship");
}

Path FirstVisible(const Scene& scene, std::string_view type) {
    for (const Path& path : scene.Prims(type)) {
        if (scene.IsVisible(path)) {
            return path;
        }
    }
    return {};
}

float Number(const Scene& scene, const Path& path, std::string_view attribute, float fallback) {
    return static_cast<float>(GetNumber(scene, path, attribute, fallback));
}

/// A CPU array of `components`-vectors as floats, shaped (n, components) or flat.
std::vector<float> FloatsOf(const Tensor& tensor, std::size_t components, std::string_view what) {
    const bool shaped =
        tensor.Rank() == 2 && tensor.Shape()[1] == static_cast<std::int64_t>(components);
    if (tensor.Location().kind != DeviceKind::Cpu || !(tensor.Rank() == 1 || shaped) ||
        static_cast<std::size_t>(tensor.Elements()) % components != 0) {
        throw std::invalid_argument(std::string(what) + " must be a CPU array of " +
                                    std::to_string(components) + "-vectors");
    }
    const Tensor packed = tensor.Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    if (packed.Type() == DType::Float32) {
        return {packed.As<float>(), packed.As<float>() + count};
    }
    if (packed.Type() == DType::Float64) {
        return {packed.As<double>(), packed.As<double>() + count};
    }
    throw std::invalid_argument(std::string(what) + " must be float32 or float64");
}

/// A CPU array of integers, of any shape.
std::vector<std::int64_t> IntsOf(const Tensor& tensor, std::string_view what) {
    if (tensor.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(std::string(what) + " must be a CPU array");
    }
    const Tensor packed = tensor.Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    switch (packed.Type()) {
    case DType::Int32:
        return {packed.As<std::int32_t>(), packed.As<std::int32_t>() + count};
    case DType::Int64:
        return {packed.As<std::int64_t>(), packed.As<std::int64_t>() + count};
    case DType::UInt32:
        return {packed.As<std::uint32_t>(), packed.As<std::uint32_t>() + count};
    default:
        throw std::invalid_argument(std::string(what) + " must be int32, int64 or uint32");
    }
}

/// The prim's attribute as an array: a written tensor or, for small arrays, a float array value.
std::optional<std::vector<float>> FloatAttribute(const Scene& scene, const Path& path,
                                                 std::string_view attribute,
                                                 std::size_t components) {
    if (const Tensor* tensor = scene.Read(path, attribute)) {
        return FloatsOf(*tensor, components, attribute);
    }
    return std::nullopt;
}

/// Targets of a relationship holding one path or a list of them.
std::vector<Path> TargetsOf(const Scene& scene, const Path& path, std::string_view attribute) {
    const Value* value = scene.Get(path, attribute);
    if (value == nullptr) {
        return {};
    }
    if (const auto* list = value->Get<std::vector<Path>>()) {
        return *list;
    }
    if (const auto* single = value->Get<Path>()) {
        return {*single};
    }
    return {};
}

/// The `ernie:spectrum` pairs of a light, if it has them.
const std::vector<float>* SpectrumPairs(const Scene& scene, const Path& light) {
    const Value* value = scene.Get(light, tokens::kLightSpectrum);
    return value != nullptr ? value->Get<std::vector<float>>() : nullptr;
}

/// UsdLux's `inputs:intensity` times 2^`inputs:exposure`.
double LightIntensity(const Scene& scene, const Path& light) {
    return GetNumber(scene, light, tokens::kIntensity, 1.0) *
           std::exp2(GetNumber(scene, light, tokens::kExposure, 0.0));
}

/// The black body temperature of a light with `inputs:enableColorTemperature`, else 0.
double ColorTemperature(const Scene& scene, const Path& light) {
    return GetBool(scene, light, tokens::kEnableColorTemperature, false)
               ? GetNumber(scene, light, tokens::kColorTemperature, 6500.0)
               : 0.0;
}

/// `shape` averaged over each of `bands` (eight samples per band) times `scale`.
template <typename Shape>
std::vector<float> BinShape(const Shape& shape, const SpectralBands& bands, double scale) {
    std::vector<float> values(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        double sum = 0.0;
        for (int k = 0; k < 8; ++k) {
            sum += shape(bands.Lower(i) + (k + 0.5) * bands.Width() / 8.0);
        }
        values[static_cast<std::size_t>(i)] = static_cast<float>(scale * sum / 8.0);
    }
    return values;
}

/// The relative shape of a light's spectrum over wavelength in nm (`tokens::kPhysicalIlluminant`):
/// its illuminant, else a black body at the enabled colour temperature, else flat; times the
/// shape of `inputs:color` (upsampled with `RgbToSigmoid`).
/// @throws std::invalid_argument for an unknown illuminant, a "custom" one without pairs, or
/// illuminant D outside 4000 to 25000 K.
std::function<double(double)> IlluminantShape(const Scene& scene, const Path& light) {
    using namespace tokens;
    const std::string illuminant = GetToken(scene, light, kPhysicalIlluminant, "");
    const double kelvin = GetNumber(scene, light, kColorTemperature, 6500.0);
    std::function<double(double)> base;
    if (illuminant.empty()) {
        const double temperature = ColorTemperature(scene, light);
        base = [temperature](double w) {
            return temperature > 0.0 ? PlanckRadiance(w, temperature) : 1.0;
        };
    } else if (illuminant == kWhite) {
        base = [](double) { return 1.0; };
    } else if (illuminant == kBlackbody) {
        if (!(kelvin > 0.0)) {
            throw std::invalid_argument("Light '" + light.String() +
                                        "' needs a positive colour temperature");
        }
        base = [kelvin](double w) { return PlanckRadiance(w, kelvin); };
    } else if (illuminant == kIlluminantD) {
        CieDaylight(560.0, kelvin);  // checks the temperature
        base = [kelvin](double w) { return CieDaylight(w, kelvin); };
    } else if (illuminant == kCustom) {
        const Value* value = scene.Get(light, kPhysicalCustomIlluminant);
        const auto* pairs = value != nullptr ? value->Get<std::vector<float>>() : nullptr;
        if (pairs == nullptr || pairs->size() < 4) {
            throw std::invalid_argument("Light '" + light.String() + "' has a custom illuminant "
                                        "but no '" + std::string(kPhysicalCustomIlluminant) +
                                        "' pairs");
        }
        const auto spectrum = std::make_shared<Spectrum>(Spectrum::FromPairs(*pairs));
        const double low = spectrum->Wavelengths().front();
        const double high = spectrum->Wavelengths().back();
        base = [spectrum, low, high](double w) {
            return w < low || w > high ? 0.0 : std::max((*spectrum)(w), 0.0);
        };
    } else {
        throw std::invalid_argument("Light '" + light.String() + "' has unknown illuminant '" +
                                    illuminant + "'");
    }
    const glm::dvec3 color = GetVec3(scene, light, kColor, glm::dvec3(1.0));
    if (color == glm::dvec3(1.0)) {
        return base;
    }
    const SigmoidSpectrum tint = RgbToSigmoid(color);
    return [base, tint](double w) { return base(w) * tint(w); };
}

/// Whether the light gives its brightness in photometric units (`tokens::kPhotometricPower`).
bool IsPhotometric(const Scene& scene, const Path& light) {
    using namespace tokens;
    return scene.Get(light, kPhotometricIlluminance) != nullptr ||
           scene.Get(light, kPhotometricIlluminanceDistance) != nullptr ||
           scene.Get(light, kPhotometricPower) != nullptr;
}

/// A photometric area light's luminous value over `spread` (4 pi sr for a sphere's luminous
/// intensity, pi times the area for a flat emitter's luminance): its power in lumens over the
/// spread, or, with an illuminance distance d, the illuminance at d times d^2 (a luminous
/// intensity), over the area for a flat emitter.
double PhotometricAreaValue(const Scene& scene, const Path& light, double spread, double area) {
    using namespace tokens;
    const double distance = GetNumber(scene, light, kPhotometricIlluminanceDistance, 0.0);
    if (distance > 0.0) {
        const double intensity =
            GetNumber(scene, light, kPhotometricIlluminance, 0.0) * distance * distance;
        return area > 0.0 ? intensity / area : intensity;
    }
    return GetNumber(scene, light, kPhotometricPower, 1600.0) / spread;
}

/// A light's spectrum in relative units, per band of `bands`: its shape (`IlluminantShape`)
/// scaled so that its luminous value (683 lm/W times its integral against the CIE y function) is
/// `luminous`.
std::vector<float> RelativeSpectrum(const Scene& scene, const Path& light,
                                    const SpectralBands& bands, double luminous) {
    const std::function<double(double)> shape = IlluminantShape(scene, light);
    // Midpoints of 1 nm steps, so narrow spectra integrate as wide as they are.
    double luminance = 0.0;
    for (double w = 360.5; w < 830.0; w += 1.0) {
        luminance += shape(w) * CieXyz(w).y;
    }
    if (!(luminance > 0.0)) {
        throw std::invalid_argument("Light '" + light.String() +
                                    "' has no visible light to scale to a luminous value");
    }
    return BinShape(shape, bands, luminous / (kLuminousEfficacy * luminance));
}

/// A physical lamp's spectrum per band (see `tokens::kLightPower`): its power over `spread`, which
/// is 4 pi sr for a sphere's radiant intensity and pi times the area for an area light's radiance.
std::vector<float> LampSpectrum(const Scene& scene, const Path& light, const SpectralBands& bands,
                                const std::vector<float>* pairs, double spread) {
    const bool hasPower = scene.Get(light, tokens::kLightPower) != nullptr;
    const double power = GetNumber(scene, light, tokens::kLightPower, 0.0);
    if (power < 0.0) {
        throw std::invalid_argument("Light '" + light.String() + "' has a negative power");
    }
    if (pairs != nullptr) {
        const Spectrum spectrum = Spectrum::FromPairs(*pairs);
        std::vector<float> binned = spectrum.Bin(bands);
        if (hasPower && !spectrum.Empty()) {
            const double low = std::max(360.0, spectrum.Wavelengths().front());
            const double high = std::min(830.0, spectrum.Wavelengths().back());
            const double total = high > low ? spectrum.Average(low, high) * (high - low) : 0.0;
            const double scale = total > 0.0 ? power / spread / total : 0.0;
            for (float& value : binned) {
                value = static_cast<float>(static_cast<double>(value) * scale);
            }
        }
        return binned;
    }
    const double temperature = ColorTemperature(scene, light);
    const auto shape = [temperature](double w) {
        return temperature > 0.0 ? PlanckRadiance(w, temperature) : 1.0;
    };
    double total = 0.0;
    for (double w = 360.5; w < 830.0; w += 1.0) {
        total += shape(w);
    }
    return BinShape(shape, bands, power / spread / total);
}

/// Real spherical harmonics up to order 2 at unit direction `d`, in the order (0,0), (1,-1),
/// (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2).
std::array<double, 9> ShBasis(const glm::dvec3& d) {
    return {0.282095,
            0.488603 * d.y,
            0.488603 * d.z,
            0.488603 * d.x,
            1.092548 * d.x * d.y,
            1.092548 * d.y * d.z,
            0.315392 * (3.0 * d.z * d.z - 1.0),
            1.092548 * d.x * d.z,
            0.546274 * (d.x * d.x - d.y * d.y)};
}

}  // namespace

std::string_view WireframeToken(Wireframe mode) {
    switch (mode) {
    case Wireframe::Off:
        return tokens::kOff;
    case Wireframe::Overlay:
        return tokens::kOverlay;
    case Wireframe::HiddenLine:
        return tokens::kHiddenLine;
    }
    return tokens::kOff;
}

Wireframe WireframeOf(const Scene& scene, const Path& settings) {
    const Value* value = settings.IsEmpty() ? nullptr : scene.Get(settings, tokens::kWireframe);
    if (value == nullptr) {
        return Wireframe::Off;
    }
    if (value->Is<bool>() || value->Is<int>()) {
        return GetBool(scene, settings, tokens::kWireframe, false) ? Wireframe::Overlay
                                                                   : Wireframe::Off;
    }
    const std::string token = GetToken(scene, settings, tokens::kWireframe, "");
    for (const Wireframe mode : {Wireframe::Off, Wireframe::Overlay, Wireframe::HiddenLine}) {
        if (token == WireframeToken(mode)) {
            return mode;
        }
    }
    throw std::invalid_argument("'" + std::string(tokens::kWireframe) + "' of '" +
                                settings.String() +
                                "' must be \"off\", \"overlay\" or \"hiddenLine\"");
}

Occlusion OcclusionOf(const Scene& scene, const Path& settings) {
    const std::string token = settings.IsEmpty()
                                  ? std::string(tokens::kOff)
                                  : GetToken(scene, settings, tokens::kOcclusion, tokens::kOff);
    if (token == tokens::kOff) {
        return Occlusion::Off;
    }
    if (token == tokens::kOccluders) {
        return Occlusion::Occluders;
    }
    if (token == tokens::kTwoPass) {
        return Occlusion::TwoPass;
    }
    throw std::invalid_argument("'" + std::string(tokens::kOcclusion) + "' of '" +
                                settings.String() +
                                "' must be \"off\", \"occluders\" or \"twoPass\"");
}

Culling CullingOf(const Scene& scene, const Path& settings) {
    const std::string token = settings.IsEmpty()
                                  ? std::string(tokens::kGpu)
                                  : GetToken(scene, settings, tokens::kCulling, tokens::kGpu);
    if (token == tokens::kOff) {
        return Culling::Off;
    }
    if (token == tokens::kCpu) {
        return Culling::Cpu;
    }
    if (token == tokens::kGpu) {
        return Culling::Gpu;
    }
    throw std::invalid_argument("'" + std::string(tokens::kCulling) + "' of '" + settings.String() +
                                "' must be \"off\", \"cpu\" or \"gpu\"");
}

std::optional<std::uint64_t> FrameOf(const Scene& scene, const Path& settings) {
    if (settings.IsEmpty() || scene.Get(settings, tokens::kFrame) == nullptr) {
        return std::nullopt;
    }
    const double frame = GetNumber(scene, settings, tokens::kFrame, -1.0);
    if (!(frame >= 0.0) || frame != std::floor(frame) || frame >= 0x1p64) {
        throw std::invalid_argument("'" + std::string(tokens::kFrame) + "' of '" +
                                    settings.String() + "' must be a non-negative integer");
    }
    return static_cast<std::uint64_t>(frame);
}

Path BoundMaterial(const Scene& scene, const Path& prim) {
    const Path material = GetTarget(scene, prim, tokens::kMaterialBinding);
    if (scene.Has(material) && scene.TypeOf(material) == tokens::kMaterial) {
        return material;
    }
    return {};
}

RenderPrims ResolveRenderPrims(const Scene& scene, const Path& settings) {
    RenderPrims prims;
    if (!settings.IsEmpty()) {
        if (!scene.Has(settings) || scene.TypeOf(settings) != tokens::kRenderSettings) {
            throw std::invalid_argument("'" + settings.String() + "' is not a RenderSettings prim");
        }
        prims.settings = settings;
    } else {
        prims.settings = FirstOutsideSession(scene, tokens::kRenderSettings);
    }
    if (!prims.settings.IsEmpty()) {
        prims.camera = GetTarget(scene, prims.settings, tokens::kCameraRel);
    }
    if (prims.camera.IsEmpty()) {
        prims.camera = FirstOutsideSession(scene, tokens::kCamera);
    }
    prims.terrain = FirstVisible(scene, tokens::kTerrain);
    if (!prims.terrain.IsEmpty()) {
        prims.material = BoundMaterial(scene, prims.terrain);
    }
    const std::optional<std::vector<Path>> chosen =
        RelationshipOf(scene, prims.settings, tokens::kLightsRel);
    std::vector<Path> candidates;
    if (chosen) {
        for (const Path& path : *chosen) {
            if (scene.Has(path) && scene.IsVisible(path)) {
                candidates.push_back(path);
            }
        }
    } else {
        for (const std::string_view type :
             {tokens::kDistantLight, tokens::kSphereLight, tokens::kRectLight, tokens::kDiskLight,
              tokens::kDomeLight}) {
            for (const Path& path : scene.Prims(type)) {
                if (!InSession(path) && scene.IsVisible(path)) {
                    candidates.push_back(path);
                }
            }
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (const Path& path : candidates) {
        const std::string& type = scene.TypeOf(path);
        if (type == tokens::kDistantLight && prims.light.IsEmpty()) {
            prims.light = path;
        } else if (type == tokens::kDomeLight) {
            if (prims.dome.IsEmpty()) {
                prims.dome = path;
            }
        } else if (type == tokens::kDistantLight || type == tokens::kSphereLight ||
                   type == tokens::kRectLight || type == tokens::kDiskLight) {
            prims.lights.push_back(path);
        }
    }
    prims.ground = FirstVisible(scene, tokens::kGroundPlane);
    prims.stars = FirstVisible(scene, tokens::kStarField);
    return prims;
}

namespace {

void CheckPosesShape(const Tensor& tensor, const Path& camera) {
    const bool shaped = (tensor.Rank() == 3 && tensor.Shape()[1] == 4 && tensor.Shape()[2] == 4) ||
                        (tensor.Rank() == 2 && tensor.Shape()[1] == 16);
    if (!shaped || tensor.Shape()[0] < 1) {
        throw std::invalid_argument("'" + std::string(tokens::kPoses) + "' of '" + camera.String() +
                                    "' must be shaped (views, 4, 4)");
    }
}

}  // namespace

glm::dmat4 PosesParentOf(const Scene& scene, const Path& camera) {
    const Path parent = camera.Parent();
    return parent.IsEmpty() || parent.IsRoot() || !scene.Has(parent) ? glm::dmat4(1.0)
                                                                     : scene.WorldTransform(parent);
}

const Tensor* GpuPosesOf(const Scene& scene, const Path& camera) {
    const Tensor* tensor = scene.Read(camera, tokens::kPoses);
    if (tensor == nullptr || !IsGpu(tensor->Location().kind)) {
        return nullptr;
    }
    CheckPosesShape(*tensor, camera);
    const std::string device = GetToken(scene, camera, tokens::kPosesDevice, tokens::kGpu);
    if (device == tokens::kHost) {
        return nullptr;
    }
    if (device != tokens::kGpu) {
        throw std::invalid_argument("'" + std::string(tokens::kPosesDevice) + "' of '" +
                                    camera.String() + "' must be \"gpu\" or \"host\"");
    }
    return tensor->Type() == DType::Float32 && tensor->IsContiguous() ? tensor : nullptr;
}

std::vector<glm::dmat4> ViewPosesOf(const Scene& scene, std::span<const Path> cameras,
                                    std::uint32_t width, std::uint32_t height,
                                    const Tensor* hostPoses) {
    using namespace tokens;
    for (const Path& camera : cameras) {
        if (!scene.Has(camera) || scene.TypeOf(camera) != kCamera) {
            throw std::invalid_argument("'" + camera.String() + "' is not a Camera prim");
        }
    }
    std::vector<glm::dmat4> poses;
    if (cameras.size() > 1) {
        const Camera first = CameraOf(scene, cameras.front(), width, height);
        const glm::mat4 projection = first.RasterView().projection;
        for (const Path& path : cameras) {
            const Camera camera = CameraOf(scene, path, width, height);
            const glm::mat4 other = camera.RasterView().projection;
            bool same = camera.bands.count == first.bands.count &&
                        camera.bands.minNm == first.bands.minNm &&
                        camera.bands.maxNm == first.bands.maxNm;
            for (int c = 0; c < 4 && same; ++c) {
                for (int r = 0; r < 4 && same; ++r) {
                    same = std::abs(other[c][r] - projection[c][r]) <=
                           1e-5f * std::max(1.0f, std::abs(projection[c][r]));
                }
            }
            if (!same) {
                throw std::invalid_argument(
                    "Camera '" + path.String() + "' differs in projection or bands from '" +
                    cameras.front().String() + "', so they cannot render as one batch");
            }
            poses.push_back(camera.transform);
        }
        return poses;
    }
    if (cameras.size() != 1) {
        return poses;
    }
    const Path& camera = cameras.front();
    const Tensor* tensor = hostPoses != nullptr ? hostPoses : scene.Read(camera, kPoses);
    if (tensor == nullptr) {
        return poses;
    }
    CheckPosesShape(*tensor, camera);
    if (tensor->Location().kind != DeviceKind::Cpu ||
        (tensor->Type() != DType::Float32 && tensor->Type() != DType::Float64)) {
        throw std::invalid_argument("'" + std::string(kPoses) + "' of '" + camera.String() +
                                    "' must be a CPU float32 or float64 array");
    }
    const Tensor packed = tensor->Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    std::vector<double> values;
    if (packed.Type() == DType::Float32) {
        values.assign(packed.As<float>(), packed.As<float>() + count);
    } else {
        values.assign(packed.As<double>(), packed.As<double>() + count);
    }
    const glm::dmat4 parentWorld = PosesParentOf(scene, camera);
    for (std::size_t first = 0; first < values.size(); first += 16) {
        glm::dmat4 local(1.0);
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                local[col][row] = values[first + static_cast<std::size_t>(row * 4 + col)];
            }
        }
        poses.push_back(parentWorld * local);
    }
    return poses;
}

std::int32_t WorldOf(const Scene& scene, const Path& path) {
    for (Path at = path; !at.IsEmpty() && !at.IsRoot(); at = at.Parent()) {
        if (scene.Has(at) && scene.Get(at, tokens::kWorld) != nullptr) {
            return static_cast<std::int32_t>(GetNumber(scene, at, tokens::kWorld, -1.0));
        }
    }
    return -1;
}

namespace {

/// The `ernie:worlds` of `path` if it has `count` of them, as worlds.
std::optional<std::vector<std::int32_t>> WorldsAttribute(const Scene& scene, const Path& path,
                                                         std::size_t count) {
    const Tensor* tensor = scene.Read(path, tokens::kWorlds);
    if (tensor == nullptr) {
        return std::nullopt;
    }
    const std::vector<std::int64_t> values = IntsOf(*tensor, tokens::kWorlds);
    if (values.size() != count) {
        throw std::invalid_argument("'" + std::string(tokens::kWorlds) + "' of '" + path.String() +
                                    "' has " + std::to_string(values.size()) + " worlds for " +
                                    std::to_string(count));
    }
    return std::vector<std::int32_t>(values.begin(), values.end());
}

}  // namespace

std::vector<std::int32_t> ViewWorldsOf(const Scene& scene, std::span<const Path> cameras,
                                       std::size_t views) {
    std::vector<std::int32_t> worlds;
    if (cameras.size() > 1) {
        for (const Path& camera : cameras) {
            worlds.push_back(WorldOf(scene, camera));
        }
        return worlds;
    }
    if (cameras.empty()) {
        return std::vector<std::int32_t>(views, -1);
    }
    if (scene.Read(cameras.front(), tokens::kPoses) != nullptr) {
        if (auto perView = WorldsAttribute(scene, cameras.front(), views)) {
            return *perView;
        }
    }
    return std::vector<std::int32_t>(views, WorldOf(scene, cameras.front()));
}

std::vector<std::int32_t> InstanceWorldsOf(const Scene& scene, const Path& instancer,
                                           std::size_t count) {
    if (auto perInstance = WorldsAttribute(scene, instancer, count)) {
        return *perInstance;
    }
    return std::vector<std::int32_t>(count, WorldOf(scene, instancer));
}

namespace {

/// The sensor filter chain a product or render var names in `ernie:filters`: each SensorFilter
/// prim's type, with every number or float array under its inputs:.
std::vector<SensorFilterStep> FilterStepsOf(const Scene& scene, const Path& prim) {
    using namespace tokens;
    std::vector<SensorFilterStep> steps;
    for (const Path& filter : TargetsOf(scene, prim, kFilters)) {
        if (!scene.Has(filter) || scene.TypeOf(filter) != kSensorFilter) {
            throw std::invalid_argument("'" + filter.String() + "' is not a SensorFilter prim");
        }
        SensorFilterStep step;
        step.type = GetToken(scene, filter, kFilterType, "");
        if (step.type.empty()) {
            throw std::invalid_argument("Sensor filter '" + filter.String() + "' has no '" +
                                        std::string(kFilterType) + "'");
        }
        constexpr std::string_view kInputs = "inputs:";
        for (const std::string& attribute : scene.Attributes(filter)) {
            if (!attribute.starts_with(kInputs)) {
                continue;
            }
            const std::string name = attribute.substr(kInputs.size());
            const Value* value = scene.Get(filter, attribute);
            if (const auto* list = value != nullptr ? value->Get<std::vector<float>>() : nullptr) {
                step.arrays[name] = *list;
            } else if (const Tensor* tensor = scene.Read(filter, attribute)) {
                step.arrays[name] = FloatsOf(*tensor, 1, attribute);
            } else {
                step.inputs[name] = static_cast<float>(GetNumber(scene, filter, attribute, 0.0));
            }
        }
        steps.push_back(std::move(step));
    }
    return steps;
}

}  // namespace

std::vector<RenderProductSpec>
ResolveRenderProducts(const Scene& scene, std::span<const Path> products, const Path& settings) {
    using namespace tokens;
    const Path resolved = ResolveRenderPrims(scene, settings).settings;
    const glm::dvec2 settingsResolution = GetVec2(scene, resolved, kResolution, {2048.0, 1080.0});
    std::vector<Path> paths(products.begin(), products.end());
    if (paths.empty() && !resolved.IsEmpty()) {
        paths = TargetsOf(scene, resolved, kProducts);
    }
    const auto extentOf = [](glm::dvec2 resolution) {
        if (!(resolution.x >= 1.0) || !(resolution.y >= 1.0)) {
            throw std::invalid_argument("Render resolutions must be positive");
        }
        return VkExtent2D{static_cast<std::uint32_t>(resolution.x),
                          static_cast<std::uint32_t>(resolution.y)};
    };
    std::vector<RenderProductSpec> specs;
    if (paths.empty()) {
        RenderProductSpec spec;
        spec.product = resolved.IsEmpty() ? Path::Root() : resolved;
        spec.resolution = extentOf(settingsResolution);
        spec.vars.emplace_back(spec.product.IsRoot() ? Path("/LdrColor")
                                                     : spec.product.Child("LdrColor"),
                               std::string(kLdrColor));
        spec.varFilters.push_back(std::nullopt);
        specs.push_back(std::move(spec));
        return specs;
    }
    for (const Path& path : paths) {
        if (!scene.Has(path) || scene.TypeOf(path) != kRenderProduct) {
            throw std::invalid_argument("'" + path.String() + "' is not a RenderProduct prim");
        }
        RenderProductSpec spec;
        spec.product = path;
        spec.cameras = TargetsOf(scene, path, kCameraRel);
        spec.camera = spec.cameras.empty() ? Path() : spec.cameras.front();
        spec.resolution = extentOf(GetVec2(scene, path, kResolution, settingsResolution));
        if (scene.Get(path, kOverlays) != nullptr) {
            spec.overlays = GetToken(scene, path, kOverlays, "");
        }
        bool ownChains = false;
        for (const Path& var : TargetsOf(scene, path, kOrderedVars)) {
            if (!scene.Has(var) || scene.TypeOf(var) != kRenderVar) {
                throw std::invalid_argument("'" + var.String() + "' is not a RenderVar prim");
            }
            const std::string name = GetToken(scene, var, kSourceName, kLdrColor);
            spec.vars.emplace_back(var, name);
            // A var with its own `ernie:filters` comes from that chain, not the product's.
            if (scene.Get(var, kFilters) != nullptr) {
                spec.varFilters.push_back(FilterStepsOf(scene, var));
                ownChains = true;
            } else {
                spec.varFilters.push_back(std::nullopt);
            }
        }
        spec.filters = FilterStepsOf(scene, path);
        // The built-in filters decide which vars the chain writes. A chain with custom filters
        // (which may integrate in place of qe or cie), or vars with chains of their own, are
        // checked by the renderer, which knows the custom filters (`VarsWrittenBy` with its
        // registry).
        std::vector<SensorFilter> builtins;
        bool custom = ownChains;
        for (const SensorFilterStep& step : spec.filters) {
            if (const std::optional<SensorFilter> builtin = step.Builtin()) {
                builtins.push_back(*builtin);
            } else {
                custom = true;
            }
        }
        if (custom) {
            if (spec.vars.empty()) {
                spec.vars.emplace_back(path.Child("LdrColor"), std::string(kLdrColor));
                spec.varFilters.push_back(std::nullopt);
            }
            specs.push_back(std::move(spec));
            continue;
        }
        const std::vector<SensorFilter> chain =
            CheckFilterChain(spec.filters.empty() ? DefaultFilterChain() : builtins);
        const std::vector<std::string_view> written = VarsWrittenBy(chain);
        for (const auto& [var, name] : spec.vars) {
            // AOVs ("aov:<output>") come from custom filters, which the renderer checks.
            if (name.starts_with(kAovPrefix) && name.size() > kAovPrefix.size()) {
                continue;
            }
            const auto& names = kRenderVarNames;
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                throw std::invalid_argument("Render var '" + var.String() + "' asks for '" + name +
                                            "', which Ernie does not render");
            }
            if (std::find(written.begin(), written.end(), name) == written.end()) {
                throw std::invalid_argument("Render var '" + var.String() + "' asks for '" + name +
                                            "', which the product's sensor filters do not write");
            }
        }
        if (spec.vars.empty()) {
            spec.vars.emplace_back(path.Child("LdrColor"), std::string(kLdrColor));
            spec.varFilters.push_back(std::nullopt);
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

std::vector<float> SunIrradianceOf(const Scene& scene, const Path& light,
                                   const SpectralBands& bands) {
    using namespace tokens;
    bands.Check();
    if (IsPhotometric(scene, light)) {
        return RelativeSpectrum(scene, light, bands,
                                GetNumber(scene, light, kPhotometricIlluminance, 10000.0));
    }
    const double temperature = GetBool(scene, light, kEnableColorTemperature, false)
                                   ? GetNumber(scene, light, kColorTemperature, 6500.0)
                                   : 0.0;
    const std::vector<float>* pairs = SpectrumPairs(scene, light);
    const bool physical = scene.Get(light, kLightIrradiance) != nullptr || pairs != nullptr;
    if (physical) {
        if (pairs != nullptr) {
            const Spectrum spectrum = Spectrum::FromPairs(*pairs);
            std::vector<float> binned = spectrum.Bin(bands);
            if (scene.Get(light, kLightIrradiance) != nullptr && !spectrum.Empty()) {
                const double total =
                    spectrum.Average(spectrum.Wavelengths().front(),
                                     spectrum.Wavelengths().back()) *
                    (spectrum.Wavelengths().back() - spectrum.Wavelengths().front());
                const double scale = GetNumber(scene, light, kLightIrradiance, total) / total;
                for (float& value : binned) {
                    value = static_cast<float>(static_cast<double>(value) * scale);
                }
            }
            return binned;
        }
        return BlackbodyIrradiance(bands, temperature > 0.0 ? temperature : 5772.0,
                                   GetNumber(scene, light, kLightIrradiance, 1361.0));
    }
    return RelativeSpectrum(scene, light, bands, M_PI * LightIntensity(scene, light));
}

Light LightOf(const Scene& scene, const Path& path, const SpectralBands& bands) {
    using namespace tokens;
    bands.Check();
    Light light;
    light.shadows = GetBool(scene, path, kShadowEnable, true);
    const std::string_view type = scene.TypeOf(path);
    if (type == kDistantLight) {
        light.type = LightType::Distant;
        light.direction = glm::vec3(TowardsLight(scene, path));
        light.spectrum = SunIrradianceOf(scene, path, bands);
        return light;
    }
    if (type != kSphereLight && type != kRectLight && type != kDiskLight) {
        throw std::invalid_argument("'" + path.String() +
                                    "' is not a DistantLight, SphereLight, RectLight or DiskLight");
    }
    const glm::dmat4 world = scene.WorldTransform(path);
    light.position = glm::dvec3(world[3]);
    const glm::dvec3 axis = -glm::dvec3(world[2]);
    const glm::dvec3 xAxis(world[0]);
    const glm::dvec3 yAxis(world[1]);
    if (glm::length(axis) == 0.0 || glm::length(xAxis) == 0.0 || glm::length(yAxis) == 0.0) {
        throw std::invalid_argument("Light '" + path.String() + "' has a degenerate transform");
    }
    light.direction = glm::vec3(glm::normalize(axis));
    if (type != kSphereLight) {
        const glm::dvec3 d = glm::normalize(axis);
        const glm::dvec3 x = xAxis - glm::dot(xAxis, d) * d;
        if (glm::length(x) == 0.0) {
            throw std::invalid_argument("Light '" + path.String() + "' has a degenerate transform");
        }
        light.tangent = glm::vec3(glm::normalize(x));
        double area = 0.0;
        if (type == kRectLight) {
            light.type = LightType::Rect;
            light.size = glm::vec2(GetNumber(scene, path, kWidth, 1.0) * glm::length(xAxis),
                                   GetNumber(scene, path, kHeight, 1.0) * glm::length(yAxis));
            area = static_cast<double>(light.size.x) * static_cast<double>(light.size.y);
        } else {
            light.type = LightType::Disk;
            light.radius =
                static_cast<float>(GetNumber(scene, path, kRadius, 0.5) * glm::length(xAxis));
            const double radius = light.radius;
            area = M_PI * radius * radius;
        }
        if (!(area > 0.0)) {
            throw std::invalid_argument("Light '" + path.String() + "' has no area");
        }
        const std::vector<float>* pairs = SpectrumPairs(scene, path);
        if (IsPhotometric(scene, path)) {
            light.spectrum = RelativeSpectrum(
                scene, path, bands, PhotometricAreaValue(scene, path, M_PI * area, area));
        } else {
            light.spectrum = scene.Get(path, kLightPower) != nullptr || pairs != nullptr
                                 ? LampSpectrum(scene, path, bands, pairs, M_PI * area)
                                 : RelativeSpectrum(scene, path, bands,
                                                    LightIntensity(scene, path));
        }
        return light;
    }
    light.radius =
        GetBool(scene, path, kTreatAsPoint, false) ? 0.0f : Number(scene, path, kRadius, 0.5f);
    light.coneAngle = std::clamp(Number(scene, path, kShapingConeAngle, 180.0f), 0.0f, 180.0f);
    light.coneSoftness = std::clamp(Number(scene, path, kShapingConeSoftness, 0.0f), 0.0f, 1.0f);
    if (light.radius < 0.0f) {
        throw std::invalid_argument("SphereLight '" + path.String() + "' has a negative radius");
    }
    if (IsPhotometric(scene, path)) {
        light.spectrum = RelativeSpectrum(scene, path, bands,
                                          PhotometricAreaValue(scene, path, 4.0 * M_PI, 0.0));
        return light;
    }
    const std::vector<float>* pairs = SpectrumPairs(scene, path);
    if (scene.Get(path, kLightPower) != nullptr || pairs != nullptr) {
        light.spectrum = LampSpectrum(scene, path, bands, pairs, 4.0 * M_PI);
        return light;
    }
    const double radius = light.radius;
    const double area = radius > 0.0 ? M_PI * radius * radius : 1.0;
    light.spectrum = RelativeSpectrum(scene, path, bands, LightIntensity(scene, path) * area);
    return light;
}

std::array<glm::vec3, 9> DomeIrradianceSh(const Tensor& latlong, const glm::dmat3& rotation) {
    if (latlong.Location().kind != DeviceKind::Cpu || latlong.Type() != DType::Float32 ||
        latlong.Shape().size() != 3 || latlong.Shape()[2] < 3 || latlong.Shape()[0] < 1 ||
        latlong.Shape()[1] < 1) {
        throw std::invalid_argument(
            "A dome texture must be a float32 (rows, cols, 3 or 4) array on the CPU");
    }
    const Tensor texels = latlong.Contiguous();
    const std::int64_t rows = texels.Shape()[0];
    const std::int64_t cols = texels.Shape()[1];
    const std::int64_t channels = texels.Shape()[2];
    const float* data = texels.As<float>();
    std::array<glm::dvec3, 9> radiance{};
    const double dTheta = M_PI / static_cast<double>(rows);
    const double dPhi = 2.0 * M_PI / static_cast<double>(cols);
    for (std::int64_t r = 0; r < rows; ++r) {
        const double theta = (static_cast<double>(r) + 0.5) * dTheta;
        const double solidAngle = (std::cos(static_cast<double>(r) * dTheta) -
                                   std::cos(static_cast<double>(r + 1) * dTheta)) *
                                  dPhi;
        for (std::int64_t c = 0; c < cols; ++c) {
            const double phi = (static_cast<double>(c) + 0.5) * dPhi;
            const glm::dvec3 local(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                                   std::cos(theta));
            const float* texel = data + (r * cols + c) * channels;
            const glm::dvec3 rgb(texel[0], texel[1], texel[2]);
            const std::array<double, 9> basis = ShBasis(rotation * local);
            for (std::size_t k = 0; k < 9; ++k) {
                radiance[k] += rgb * basis[k] * solidAngle;
            }
        }
    }
    constexpr std::array<double, 9> kBand{1.0,  2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25,
                                          0.25, 0.25,      0.25,      0.25};
    std::array<glm::vec3, 9> sh;
    for (std::size_t k = 0; k < 9; ++k) {
        sh[k] = glm::vec3(radiance[k] * kBand[k]);
    }
    return sh;
}

DomeLight DomeOf(const Scene& scene, const Path& path, const SpectralBands& bands) {
    using namespace tokens;
    bands.Check();
    if (scene.TypeOf(path) != kDomeLight) {
        throw std::invalid_argument("'" + path.String() + "' is not a DomeLight");
    }
    DomeLight dome;
    const glm::dmat4 world = scene.WorldTransform(path);
    const glm::dmat3 rotation(glm::normalize(glm::dvec3(world[0])),
                              glm::normalize(glm::dvec3(world[1])),
                              glm::normalize(glm::dvec3(world[2])));
    dome.toDome = glm::mat3(glm::transpose(rotation));
    dome.visible = GetBool(scene, path, kVisibleInPrimaryRay, true);
    if (const Tensor* texture = scene.Read(path, kLightTexture)) {
        dome.sh = DomeIrradianceSh(*texture, rotation);
    } else {
        dome.sh[0] = glm::vec3(static_cast<float>(std::sqrt(4.0 * M_PI)));
    }
    if (IsPhotometric(scene, path)) {
        // An upward-facing surface receives pi times the luminance times the map's irradiance
        // there (1 for a uniform dome), as its spherical harmonics give it.
        const std::array<double, 9> up = ShBasis({0.0, 0.0, 1.0});
        glm::dvec3 irradiance(0.0);
        for (std::size_t k = 0; k < up.size(); ++k) {
            irradiance += glm::dvec3(dome.sh[k]) * up[k];
        }
        const double luminance = glm::dot(irradiance, glm::dvec3(0.2126, 0.7152, 0.0722));
        if (!(luminance > 0.0)) {
            throw std::invalid_argument("DomeLight '" + path.String() +
                                        "' lights no upward surface to give an illuminance");
        }
        dome.radiance = RelativeSpectrum(
            scene, path, bands,
            GetNumber(scene, path, kPhotometricIlluminance, 10000.0) / (M_PI * luminance));
        return dome;
    }
    const std::vector<float>* pairs = SpectrumPairs(scene, path);
    dome.radiance = pairs != nullptr
                        ? Spectrum::FromPairs(*pairs).Bin(bands)
                        : RelativeSpectrum(scene, path, bands, LightIntensity(scene, path));
    return dome;
}

std::optional<Image> DomeMapOf(const Scene& scene, const Path& path) {
    const Tensor* texture = scene.Read(path, tokens::kLightTexture);
    if (texture == nullptr) {
        return std::nullopt;
    }
    try {
        return ImageOfTensor(*texture, ImageEncoding::Linear);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument("'" + std::string(tokens::kLightTexture) + "' of '" +
                                    path.String() + "': " + error.what());
    }
}

namespace {

/// The prim's float array `attribute`, `components` per element, from a tensor or a float array
/// value.
std::optional<std::vector<float>> FloatArray(const Scene& scene, const Path& path,
                                             std::string_view attribute,
                                             std::size_t components) {
    if (std::optional<std::vector<float>> values =
            FloatAttribute(scene, path, attribute, components)) {
        return values;
    }
    if (const Value* value = scene.Get(path, attribute)) {
        if (const auto* list = value->Get<std::vector<float>>()) {
            if (list->size() % components != 0) {
                throw std::invalid_argument("'" + std::string(attribute) + "' of '" +
                                            path.String() + "' needs " +
                                            std::to_string(components) + " values per element");
            }
            return *list;
        }
    }
    return std::nullopt;
}

/// The rotation part of the world transform of `path`, without scale.
glm::dmat3 WorldRotation(const Scene& scene, const Path& path) {
    const glm::dmat4 world = scene.WorldTransform(path);
    return glm::dmat3(glm::normalize(glm::dvec3(world[0])), glm::normalize(glm::dvec3(world[1])),
                      glm::normalize(glm::dvec3(world[2])));
}

}  // namespace

std::vector<Star> StarsOf(const Scene& scene, const Path& path) {
    using namespace tokens;
    if (scene.TypeOf(path) != kStarField) {
        throw std::invalid_argument("'" + path.String() + "' is not a StarField");
    }
    const auto array = [&](std::string_view attribute, std::size_t components) {
        return FloatArray(scene, path, attribute, components);
    };
    const std::optional<std::vector<float>> magnitudes = array(kStarsMagnitude, 1);
    if (!magnitudes) {
        return {};
    }
    const std::size_t count = magnitudes->size();
    const auto check = [&](const std::optional<std::vector<float>>& values,
                           std::string_view attribute, std::size_t components) {
        if (values && values->size() != count * components) {
            throw std::invalid_argument("'" + std::string(attribute) + "' of '" + path.String() +
                                        "' has " + std::to_string(values->size() / components) +
                                        " stars where '" + std::string(kStarsMagnitude) +
                                        "' has " + std::to_string(count));
        }
    };
    const std::optional<std::vector<float>> ra = array(kStarsRightAscension, 1);
    const std::optional<std::vector<float>> dec = array(kStarsDeclination, 1);
    const std::optional<std::vector<float>> direction = array(kStarsDirection, 3);
    const std::optional<std::vector<float>> temperature = array(kStarsTemperature, 1);
    const std::optional<std::vector<float>> colorIndex = array(kStarsColorIndex, 1);
    check(ra, kStarsRightAscension, 1);
    check(dec, kStarsDeclination, 1);
    check(direction, kStarsDirection, 3);
    check(temperature, kStarsTemperature, 1);
    check(colorIndex, kStarsColorIndex, 1);
    if (!(ra && dec) && !direction) {
        throw std::invalid_argument("StarField '" + path.String() + "' needs '" +
                                    std::string(kStarsRightAscension) + "' and '" +
                                    std::string(kStarsDeclination) + "', or '" +
                                    std::string(kStarsDirection) + "'");
    }
    const double zeroPoint = GetNumber(scene, path, kStarsZeroPoint, kMagnitudeZeroLux) *
                             std::exp2(GetNumber(scene, path, kExposure, 0.0));
    const double limit = GetNumber(scene, path, kStarsMagnitudeLimit, 99.0);
    std::vector<Star> stars;
    stars.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double magnitude = (*magnitudes)[i];
        if (!(magnitude <= limit)) {
            continue;
        }
        Star star;
        if (ra && dec) {
            star.rightAscension = static_cast<float>(glm::radians(static_cast<double>((*ra)[i])));
            star.declination = static_cast<float>(glm::radians(static_cast<double>((*dec)[i])));
        } else {
            const glm::dvec3 d = glm::normalize(glm::dvec3((*direction)[3 * i],
                                                           (*direction)[3 * i + 1],
                                                           (*direction)[3 * i + 2]));
            star.rightAscension = static_cast<float>(std::atan2(d.y, d.x));
            star.declination = static_cast<float>(std::asin(std::clamp(d.z, -1.0, 1.0)));
        }
        const double kelvin = temperature  ? static_cast<double>((*temperature)[i])
                              : colorIndex ? ColorIndexToTemperature((*colorIndex)[i])
                                           : 5772.0;
        star.temperature = static_cast<float>(kelvin);
        star.irradiance555 = static_cast<float>(
            BlackbodySpectralIrradiance(555.0, kelvin, StarIlluminance(magnitude, zeroPoint)));
        stars.push_back(star);
    }
    return stars;
}

StarField StarFieldOf(const Scene& scene, const Path& path) {
    if (scene.TypeOf(path) != tokens::kStarField) {
        throw std::invalid_argument("'" + path.String() + "' is not a StarField");
    }
    StarField field;
    field.toCatalogue = glm::mat3(glm::transpose(WorldRotation(scene, path)));
    return field;
}

double OverscanOf(const Camera& camera) {
    if (camera.lens.IsPinhole()) {
        return 1.0;
    }
    const double f = camera.FocalLengthPixels();
    const glm::dvec2 centre = camera.PrincipalPoint();
    const double w = camera.width;
    const double h = camera.height;
    double overscan = 1.0;
    const auto include = [&](double x, double y) {
        const glm::dvec2 u = camera.lens.Undistort({(x - centre.x) / f, (centre.y - y) / f});
        const double right = w - centre.x;
        const double up = centre.y;
        const double down = h - centre.y;
        if (u.x > 0.0) {
            overscan = std::max(overscan, u.x * f / right);
        } else if (u.x < 0.0) {
            overscan = std::max(overscan, -u.x * f / centre.x);
        }
        if (u.y > 0.0) {
            overscan = std::max(overscan, u.y * f / up);
        } else if (u.y < 0.0) {
            overscan = std::max(overscan, -u.y * f / down);
        }
    };
    constexpr int kSamples = 64;
    for (int i = 0; i <= kSamples; ++i) {
        const double t = static_cast<double>(i) / kSamples;
        include(t * w, 0.0);
        include(t * w, h);
        include(0.0, t * h);
        include(w, t * h);
    }
    return overscan * 1.01;
}

FrameParams FrameParamsOf(const Scene& scene, const RenderPrims& prims, std::uint32_t width,
                          std::uint32_t height) {
    using namespace tokens;
    if (prims.camera.IsEmpty()) {
        throw std::invalid_argument("The scene has no Camera");
    }
    FrameParams params;
    params.camera = CameraOf(scene, prims.camera, width, height);
    params.overscan = OverscanOf(params.camera);
    const Path& settings = prims.settings;
    params.ambient = Number(scene, settings, kAmbient, params.ambient);
    params.wireframe = WireframeOf(scene, settings);
    params.overlays = GetToken(scene, settings, kOverlays, "");
    params.exposure =
        static_cast<float>(std::exp2(GetNumber(scene, settings, kExposureCompensation, 0.0)));
    params.meter = GetBool(scene, settings, kMeter, false);
    params.culling = CullingOf(scene, settings);
    params.occlusion = OcclusionOf(scene, settings);
    // A pushbroom's depth pyramid holds its strip's depths, which no one view-projection tests
    // against: its instances cull by the envelope's frustum alone.
    if (params.camera.pushbroom) {
        params.occlusion = Occlusion::Off;
    }
    params.meshLodError =
        std::max(Number(scene, settings, kLodMeshError, params.meshLodError), 0.0f);
    params.clearColor = glm::vec3(GetVec3(scene, settings, kClearColor, {0.0, 0.0, 0.0}));
    ShadowSettings& shadows = params.shadows;
    shadows.cascades =
        static_cast<int>(GetNumber(scene, settings, kShadowCascades, shadows.cascades));
    shadows.resolution = static_cast<std::uint32_t>(
        GetNumber(scene, settings, kShadowResolution, shadows.resolution));
    shadows.worldResolution = static_cast<std::uint32_t>(
        GetNumber(scene, settings, kShadowWorldResolution, shadows.worldResolution));
    const std::string worlds = GetToken(scene, settings, kShadowWorlds, kSeparate);
    if (worlds == kSeparate) {
        shadows.worlds = WorldShadows::Separate;
    } else if (worlds == kShared) {
        shadows.worlds = WorldShadows::Shared;
    } else if (worlds == kMerged) {
        shadows.worlds = WorldShadows::Merged;
    } else {
        throw std::invalid_argument("'" + std::string(kShadowWorlds) + "' of '" +
                                    settings.String() +
                                    "' must be \"separate\", \"shared\" or \"merged\"");
    }
    const std::string fit = GetToken(scene, settings, kShadowWorldFit, kScene);
    if (fit != kScene && fit != kWorldFit) {
        throw std::invalid_argument("'" + std::string(kShadowWorldFit) + "' of '" +
                                    settings.String() + "' must be \"scene\" or \"world\"");
    }
    shadows.worldFit = fit == kWorldFit ? WorldShadowFit::World : WorldShadowFit::Scene;
    const std::string viewFit = GetToken(scene, settings, kShadowFit, kScene);
    if (viewFit != kScene && viewFit != kVisible) {
        throw std::invalid_argument("'" + std::string(kShadowFit) + "' of '" + settings.String() +
                                    "' must be \"scene\" or \"visible\"");
    }
    shadows.fit = viewFit == kVisible ? ShadowFit::Visible : ShadowFit::Scene;
    shadows.worldMargin =
        std::max(Number(scene, settings, kShadowWorldMargin, shadows.worldMargin), 0.0f);
    shadows.softness = Number(scene, settings, kShadowSoftness, shadows.softness);
    shadows.normalOffset = Number(scene, settings, kShadowNormalOffset, shadows.normalOffset);
    shadows.splitLambda = Number(scene, settings, kShadowSplitLambda, shadows.splitLambda);
    shadows.showCascades = GetBool(scene, settings, kShadowShowCascades, shadows.showCascades);
    if (!prims.light.IsEmpty()) {
        params.sunDirection = glm::vec3(TowardsLight(scene, prims.light));
        params.sunIrradiance = SunIrradianceOf(scene, prims.light, params.camera.bands);
        params.sunAngle = std::max(Number(scene, prims.light, kAngle, params.sunAngle), 0.0f);
        shadows.enabled = GetBool(scene, prims.light, kShadowEnable, true);
    }
    params.sun = !prims.light.IsEmpty();
    if (!prims.dome.IsEmpty()) {
        params.dome = DomeOf(scene, prims.dome, params.camera.bands);
    }
    if (!prims.stars.IsEmpty()) {
        params.stars = StarFieldOf(scene, prims.stars);
    }
    for (const Path& light : prims.lights) {
        params.lights.push_back(LightOf(scene, light, params.camera.bands));
        params.lights.back().world = WorldOf(scene, light);
    }
    return params;
}

MaterialSettings MaterialSettingsOf(const Scene& scene, const Path& material,
                                    const BrdfRegistry& brdfs) {
    using namespace tokens;
    MaterialSettings settings;
    settings.brdf = GetToken(scene, material, kBrdf, settings.brdf);
    const BrdfModel* model = brdfs.Find(settings.brdf);
    if (model == nullptr) {
        throw std::invalid_argument("Material '" + material.String() + "' names unknown BRDF '" +
                                    settings.brdf + "'");
    }
    for (const BrdfParam& param : model->params) {
        settings.brdfParams.push_back(
            Number(scene, material, "inputs:" + param.name, param.defaultValue));
    }
    settings.albedo = glm::vec3(GetVec3(scene, material, kAlbedo, glm::dvec3(settings.albedo)));
    settings.useAlbedoMap = GetBool(scene, material, kAlbedoMapEnable, true);
    DetailSettings& detail = settings.detail;
    detail.tileSize = Number(scene, material, kDetailTileSize, detail.tileSize);
    detail.albedoStrength = Number(scene, material, kDetailAlbedoStrength, detail.albedoStrength);
    detail.normalStrength = Number(scene, material, kDetailNormalStrength, detail.normalStrength);
    detail.directXNormals = GetBool(scene, material, kDetailDirectXNormals, detail.directXNormals);
    detail.hexTiling = GetBool(scene, material, kDetailHexTiling, detail.hexTiling);
    detail.hexRotation = Number(scene, material, kDetailHexRotation, detail.hexRotation);
    detail.hexContrast = Number(scene, material, kDetailHexContrast, detail.hexContrast);
    detail.hexSharpness = Number(scene, material, kDetailHexSharpness, detail.hexSharpness);
    return settings;
}

bool IsMeshType(std::string_view type) {
    using namespace tokens;
    return type == kMesh || type == kCube || type == kSphere || type == kCylinder ||
           type == kCone || type == kCapsule || type == kPlane;
}

MeshData PrimitiveMeshOf(const Scene& scene, const Path& path) {
    using namespace tokens;
    const std::string_view type = scene.TypeOf(path);
    const auto number = [&](std::string_view attribute, double fallback) {
        return static_cast<float>(GetNumber(scene, path, attribute, fallback));
    };
    const std::string axisName = GetToken(scene, path, kAxis, "Z");
    const std::optional<Axis> axis = AxisNamed(axisName);
    if (!axis) {
        throw std::invalid_argument("'" + path.String() + "' has unknown axis '" + axisName + "'");
    }
    if (type == kCube) {
        return BoxMesh(glm::vec3(number(kSize, 2.0)));
    }
    if (type == kSphere) {
        return SphereMesh(number(kGeomRadius, 1.0));
    }
    if (type == kCylinder) {
        return CylinderMesh(number(kGeomRadius, 1.0), number(kGeomHeight, 2.0), *axis);
    }
    if (type == kCone) {
        return ConeMesh(number(kGeomRadius, 1.0), number(kGeomHeight, 2.0), *axis);
    }
    if (type == kCapsule) {
        return CapsuleMesh(number(kGeomRadius, 0.5), number(kGeomHeight, 1.0), *axis);
    }
    if (type == kPlane) {
        return PlaneMesh(number(kGeomWidth, 2.0), number(kGeomLength, 2.0), *axis);
    }
    throw std::invalid_argument("'" + path.String() + "' is not a UsdGeom primitive");
}

MeshData MeshDataOf(const Scene& scene, const Path& mesh) {
    using namespace tokens;
    if (scene.TypeOf(mesh) != kMesh && IsMeshType(scene.TypeOf(mesh))) {
        return PrimitiveMeshOf(scene, mesh);
    }
    MeshData data;
    const Tensor* pointsTensor = scene.Read(mesh, kPoints);
    if (pointsTensor != nullptr && IsGpu(pointsTensor->Location().kind)) {
        if (pointsTensor->Elements() % 3 != 0) {
            throw std::invalid_argument("points must be 3-vectors");
        }
        data.points.assign(static_cast<std::size_t>(pointsTensor->Elements()), 0.0f);
    } else {
        data.points = FloatAttribute(scene, mesh, kPoints, 3).value_or(std::vector<float>());
    }
    const std::size_t count = data.points.size() / 3;
    const Tensor* counts = scene.Read(mesh, kFaceVertexCounts);
    const Tensor* indices = scene.Read(mesh, kFaceVertexIndices);
    if ((counts == nullptr) != (indices == nullptr)) {
        throw std::invalid_argument("needs both faceVertexCounts and faceVertexIndices");
    }
    if (counts != nullptr) {
        data.indices = TriangulateFaces(IntsOf(*counts, kFaceVertexCounts),
                                        IntsOf(*indices, kFaceVertexIndices), count);
    }
    // Normals and colours written on the GPU are zeros here, holding their place until the
    // renderer copies them in (as points).
    const Tensor* normalsTensor = scene.Read(mesh, kNormals);
    if (normalsTensor != nullptr && IsGpu(normalsTensor->Location().kind)) {
        if (static_cast<std::size_t>(normalsTensor->Elements()) == data.points.size()) {
            data.normals.assign(data.points.size(), 0.0f);
        }
    } else if (std::optional<std::vector<float>> normals =
                   FloatAttribute(scene, mesh, kNormals, 3)) {
        if (normals->size() == data.points.size()) {
            data.normals = std::move(*normals);
        }
    }
    if (const Tensor* colors = scene.Read(mesh, kDisplayColor)) {
        if (static_cast<std::size_t>(colors->Elements()) == data.points.size() && count > 1) {
            data.colors = IsGpu(colors->Location().kind) ? std::vector<float>(data.points.size(), 0.0f)
                                                         : FloatsOf(*colors, 3, kDisplayColor);
        }
    }
    return data;
}

InstanceData InstanceDataOf(const Scene& scene, const Path& instancer) {
    using namespace tokens;
    InstanceData data;
    data.positions = FloatAttribute(scene, instancer, kPositions, 3).value_or(std::vector<float>());
    data.orientations =
        FloatAttribute(scene, instancer, kOrientations, 4).value_or(std::vector<float>());
    data.scales = FloatAttribute(scene, instancer, kScales, 3).value_or(std::vector<float>());
    return data;
}

std::optional<Bounds3> ExtentOf(const Scene& scene, const Path& prim) {
    std::vector<float> values;
    if (const Tensor* tensor = scene.Read(prim, tokens::kExtent)) {
        if (tensor->Location().kind != DeviceKind::Cpu || tensor->Elements() != 6) {
            return std::nullopt;
        }
        values = FloatsOf(*tensor, 3, tokens::kExtent);
    } else if (const Value* value = scene.Get(prim, tokens::kExtent)) {
        if (const auto* floats = value->Get<std::vector<float>>()) {
            values = *floats;
        }
    }
    if (values.size() != 6) {
        return std::nullopt;
    }
    return Bounds3(glm::vec3(values[0], values[1], values[2]),
                   glm::vec3(values[3], values[4], values[5]));
}

std::optional<Bounds3> LocalBoundsOf(const Scene& scene, const Path& prim) {
    if (std::optional<Bounds3> extent = ExtentOf(scene, prim)) {
        return extent;
    }
    if (!IsMeshType(scene.TypeOf(prim))) {
        return std::nullopt;
    }
    MeshData mesh;
    try {
        if (scene.TypeOf(prim) == tokens::kMesh) {
            const Tensor* points = scene.Read(prim, tokens::kPoints);
            if (points == nullptr || points->Location().kind != DeviceKind::Cpu) {
                return std::nullopt;
            }
            mesh.points = FloatsOf(*points, 3, tokens::kPoints);
        } else {
            mesh = PrimitiveMeshOf(scene, prim);
        }
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
    if (mesh.points.size() < 3) {
        return std::nullopt;
    }
    Bounds3 bounds{glm::vec3(std::numeric_limits<float>::max()),
                   glm::vec3(std::numeric_limits<float>::lowest())};
    for (std::size_t i = 0; i + 2 < mesh.points.size(); i += 3) {
        const glm::vec3 p(mesh.points[i], mesh.points[i + 1], mesh.points[i + 2]);
        bounds.first = glm::min(bounds.first, p);
        bounds.second = glm::max(bounds.second, p);
    }
    return bounds;
}

std::vector<std::vector<std::uint32_t>>
InstancesByPrototype(const Scene& scene, const Path& instancer, std::size_t prototypes) {
    std::vector<std::vector<std::uint32_t>> lists(prototypes);
    const Tensor* positions = scene.Read(instancer, tokens::kPositions);
    const std::size_t count =
        positions != nullptr ? static_cast<std::size_t>(positions->Elements()) / 3 : 0;
    if (prototypes == 0) {
        return lists;
    }
    const Tensor* protoIndices = scene.Read(instancer, tokens::kProtoIndices);
    if (protoIndices == nullptr) {
        for (std::uint32_t i = 0; i < count; ++i) {
            lists[0].push_back(i);
        }
        return lists;
    }
    const std::vector<std::int64_t> indices = IntsOf(*protoIndices, tokens::kProtoIndices);
    for (std::size_t i = 0; i < std::min(count, indices.size()); ++i) {
        if (indices[i] >= 0 && static_cast<std::size_t>(indices[i]) < prototypes) {
            lists[static_cast<std::size_t>(indices[i])].push_back(static_cast<std::uint32_t>(i));
        }
    }
    return lists;
}

Image ImageOfTensor(const Tensor& tensor, ImageEncoding encoding) {
    const bool grey = tensor.Rank() == 2;
    const std::int64_t channels = grey ? 1 : (tensor.Rank() == 3 ? tensor.Shape()[2] : 0);
    const bool typed = tensor.Type() == DType::Float32 || tensor.Type() == DType::UInt8;
    if (!typed || (channels != 1 && channels != 3 && channels != 4) ||
        tensor.Location().kind != DeviceKind::Cpu || tensor.Shape()[0] < 1 ||
        tensor.Shape()[1] < 1) {
        throw std::invalid_argument(
            "expected a float32 or uint8 (rows, cols[, 1, 3 or 4]) CPU tensor");
    }
    const Tensor packed = tensor.Contiguous();
    Image image(static_cast<std::uint32_t>(tensor.Shape()[1]),
                static_cast<std::uint32_t>(tensor.Shape()[0]));
    const bool bytes = tensor.Type() == DType::UInt8;
    const auto* floats = bytes ? nullptr : packed.As<float>();
    const auto* uints = bytes ? packed.As<std::uint8_t>() : nullptr;
    const auto value = [&](std::size_t index, bool colour) {
        if (!bytes) {
            return floats[index];
        }
        const float stored = static_cast<float>(uints[index]) / 255.0f;
        return colour && encoding == ImageEncoding::Srgb ? SrgbToLinear(stored) : stored;
    };
    const auto stride = static_cast<std::size_t>(channels);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        glm::vec4& pixel = image.pixels[i];
        const std::size_t base = i * stride;
        if (channels == 1) {
            pixel = glm::vec4(glm::vec3(value(base, true)), 1.0f);
        } else {
            pixel = glm::vec4(value(base, true), value(base + 1, true), value(base + 2, true),
                              channels == 4 ? value(base + 3, false) : 1.0f);
        }
    }
    return image;
}

}  // namespace ernie
