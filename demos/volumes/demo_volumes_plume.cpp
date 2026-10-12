// SPDX-License-Identifier: BSD-3-Clause
// A rocket engine's plume in sunlight: hot gas glowing as a black
// body at the nozzle, cooling to soot that shadows and dims what is
// behind. A compute shader writes its soot and temperature into the
// renderer's grids each frame (RenderDelegate::SetFieldWriter).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/device/buffer.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "ernie/util/noise.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

/// The plume's hottest, at the nozzle.
constexpr double kPeakKelvin = 3000.0;
constexpr std::uint32_t kNx = 48;
constexpr std::uint32_t kNy = 48;
constexpr std::uint32_t kNz = 120;
constexpr double kVoxel = 0.1;
const Path kPlume("/World/Plume");

/// A rocket engine's plume `time` seconds in, from a nozzle 1.2 m
/// across at the origin down -z for 12 m, in 0.1 m voxels: widening
/// as it goes, hottest at the nozzle with shock diamonds, cooling to
/// soot downstream; turbulence (GradientNoise's fBm, its permutation
/// `perm`) streams down it and the core flickers. As the Python
/// twin's Warp kernel.
constexpr const char* kPlumeSlang = R"(
static const int kNx = 48;
static const int kNy = 48;
static const int kNz = 120;
static const float kPi = 3.14159265358979;

struct Push {
    float* out;
    uint* perm;
    float time;
    int field;
}
[[vk::push_constant]] ConstantBuffer<Push> push;

float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float gradient(uint hash, float x, float y, float z) {
    let h = hash & 15u;
    let u = h < 8u ? x : y;
    let v = h < 4u ? y : (h == 12u || h == 14u ? x : z);
    return ((h & 1u) == 0u ? u : -u) + ((h & 2u) == 0u ? v : -v);
}

float mix(float t, float a, float b) {
    return a + t * (b - a);
}

// Perlin's improved noise, as GradientNoise's.
float noise(float3 p) {
    let fl = floor(p);
    let x = uint(int(fl.x) & 255);
    let y = uint(int(fl.y) & 255);
    let z = uint(int(fl.z) & 255);
    let f = p - fl;
    let u = fade(f.x);
    let v = fade(f.y);
    let w = fade(f.z);
    let a = push.perm[x] + y;
    let aa = push.perm[a] + z;
    let ab = push.perm[a + 1] + z;
    let b = push.perm[x + 1] + y;
    let ba = push.perm[b] + z;
    let bb = push.perm[b + 1] + z;
    return mix(w,
               mix(v,
                   mix(u, gradient(push.perm[aa], f.x, f.y, f.z),
                       gradient(push.perm[ba], f.x - 1.0, f.y, f.z)),
                   mix(u, gradient(push.perm[ab], f.x, f.y - 1.0, f.z),
                       gradient(push.perm[bb], f.x - 1.0, f.y - 1.0, f.z))),
               mix(v,
                   mix(u, gradient(push.perm[aa + 1], f.x, f.y, f.z - 1.0),
                       gradient(push.perm[ba + 1], f.x - 1.0, f.y, f.z - 1.0)),
                   mix(u, gradient(push.perm[ab + 1], f.x, f.y - 1.0, f.z - 1.0),
                       gradient(push.perm[bb + 1], f.x - 1.0, f.y - 1.0, f.z - 1.0))));
}

// Fractional Brownian motion, as GradientNoise::Fbm's.
float fbm(float3 p, int octaves) {
    float sum = 0.0;
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    for (int o = 0; o < octaves; ++o) {
        sum += amplitude * noise(p * frequency);
        total += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return sum / total;
}

// The plume's soot (`field` 0) or temperature (1) at each cell, z fastest.
[shader("compute")]
[numthreads(64, 1, 1)]
void plume(uint3 id: SV_DispatchThreadID) {
    if (id.x >= uint(kNx * kNy * kNz)) {
        return;
    }
    let i = int(id.x) / (kNy * kNz);
    let j = (int(id.x) / kNz) % kNy;
    let k = int(id.x) % kNz;
    let px = -2.4 + 0.1 * (float(i) + 0.5);
    let py = -2.4 + 0.1 * (float(j) + 0.5);
    let pz = -12.0 + 0.1 * (float(k) + 0.5);
    let s = max(-pz, 0.0);
    let radius = 0.6 + 0.12 * s;
    let core = exp(-2.0 * (px * px + py * py) / (radius * radius));
    let fadeIn = clamp(s / 0.3, 0.0, 1.0) * clamp((12.0 - s) / 2.0, 0.0, 1.0);
    let turbulence = 0.5 + 0.5 * fbm(float3(1.5 * px, 1.5 * py, 0.8 * (pz + 20.0 * push.time)), 3);
    float value;
    if (push.field == 0) {
        value = core * (0.8 + 0.4 * (1.0 - exp(-s / 4.0))) * exp(-s / 14.0) * fadeIn *
                (0.4 + 1.2 * turbulence);
    } else {
        let diamonds = 1.0 + 0.35 * cos(2.0 * kPi * s / 1.4) * exp(-s / 3.0);
        let flicker = 1.0 + 0.08 * sin(2.0 * kPi * 12.0 * push.time);
        value = core * diamonds * exp(-s / 5.0) * fadeIn * flicker * (0.85 + 0.3 * turbulence);
    }
    push.out[id.x] = value > 1e-3 ? value : 0.0;
}
)";

/// The plume's shader on one renderer's device, which its writers
/// run.
class PlumeShader {
  public:
    explicit PlumeShader(const VulkanContext& context)
        : layout_(context, {}, sizeof(Push)),
          permutation_(context, 512 * sizeof(std::uint32_t),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       BufferMemory::Device) {
        const ShaderCompiler compiler;
        const std::array<ShaderModuleSource, 1> modules{
            {{"plume", kPlumeSlang}}};
        pipeline_ = std::make_unique<ComputePipeline>(
            context, layout_,
            compiler.CompileSpirv(modules, "plume"));
        const auto& table = GradientNoise(5).Permutation();
        const std::vector<std::uint32_t> words(table.begin(),
                                               table.end());
        permutation_.Write(std::as_bytes(std::span(words)));
    }

    /// Records field `field` (soot 0, temperature 1) at `time` into
    /// `target`.
    void Write(VkCommandBuffer commands,
               const FieldWriteTarget& target, int field,
               double time) const {
        const Push push{target.values, permutation_.Address(),
                        static_cast<float>(time), field};
        layout_.PushConstants(commands, &push, sizeof(push));
        pipeline_->Dispatch(commands,
                            std::span<const DescriptorResource>(),
                            (kNx * kNy * kNz + 63) / 64);
    }

  private:
    struct Push {
        VkDeviceAddress out, perm;
        float time;
        std::int32_t field;
    };

    PipelineLayout layout_;
    std::unique_ptr<ComputePipeline> pipeline_;
    GpuBuffer permutation_;
};

/// The image's mean linear RGB over columns [left, right) and rows
/// [top, bottom).
glm::dvec3 MeanIn(const Tensor& image, std::int64_t left,
                  std::int64_t right, std::int64_t top,
                  std::int64_t bottom) {
    const std::int64_t cols = image.Shape()[1];
    const std::uint8_t* pixels = image.As<std::uint8_t>();
    glm::dvec3 sum(0.0);
    for (std::int64_t r = top; r < bottom; ++r) {
        for (std::int64_t c = left; c < right; ++c) {
            const std::uint8_t* p = pixels + 4 * (r * cols + c);
            sum += demos::SrgbToLinear(glm::dvec3(p[0], p[1], p[2]) /
                                       255.0);
        }
    }
    return sum / static_cast<double>((right - left) * (bottom - top));
}

class PlumeDemo : public app::Demo {
  public:
    PlumeDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 12;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The rocket above its nozzle at the origin, 1.2 m across.
        Cylinder::Define(scene, "/World/Rocket/Body")
            .SetRadius(0.6)
            .SetHeight(6.0)
            .SetMaterialBinding("/World/Looks/Paint");
        SetXform(scene, "/World/Rocket/Body",
                 {.translate = glm::dvec3(0.0, 0.0, 3.4)});
        Cone::Define(scene, "/World/Rocket/Nose")
            .SetRadius(0.6)
            .SetHeight(1.6)
            .SetMaterialBinding("/World/Looks/Paint");
        SetXform(scene, "/World/Rocket/Nose",
                 {.translate = glm::dvec3(0.0, 0.0, 7.2)});
        Cone::Define(scene, "/World/Rocket/Nozzle")
            .SetRadius(0.4)
            .SetHeight(0.8)
            .SetMaterialBinding("/World/Looks/Metal");
        SetXform(scene, "/World/Rocket/Nozzle",
                 {.translate = glm::dvec3(0.0, 0.0, 0.4)});
        Material::Define(scene, "/World/Looks/Paint")
            .SetAlbedo(glm::dvec3(0.7));
        Material::Define(scene, "/World/Looks/Metal")
            .SetAlbedo(glm::dvec3(0.25));

        // The plume's soot and temperature, written on the GPU each
        // frame into dense fields whose first cell is at (-2.4, -2.4,
        // -12) m and 0.1 m apart.
        Volume::Define(scene, kPlume)
            .SetField("density", kPlume.Child("Density"))
            .SetField("temperature", kPlume.Child("Temperature"))
            .SetMaterialBinding("/World/Looks/Exhaust");
        for (const char* name : {"Density", "Temperature"}) {
            DenseField::Define(scene, kPlume.Child(name))
                .SetVoxelSize(kVoxel);
            SetXform(scene, kPlume.Child(name),
                     {.translate = glm::dvec3(-2.35, -2.35, -11.95)});
        }
        // Soot: mostly absorbing, so it glows as a black body where
        // hot (Kirchhoff) and scatters a little sunlight.
        Material::Define(scene, "/World/Looks/Exhaust")
            .SetDensity(1.5)
            .SetVolumeAlbedo(glm::dvec3(0.3))
            .SetAnisotropy(0.4);
        SetPeakTemperature(kPeakKelvin);

        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetShadowEnable(true)
            .SetTransform(DistantLightTransform({-0.5, -1.0, 0.6}));
        // Daylight exposure, from the side 40 m off.
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(35.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 500.0)
            .SetTransform(LookAtTransform({0.0, -40.0, -1.0},
                                          {0.0, 0.0, -1.5},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({320, 480})
            .SetAmbient(0.0);

        // Written in the frames of whichever renderers draw the
        // scene.
        Write(GetRenderer().Delegate(), GetRenderer().Context());
        Widgets().Slider(
            "Peak temperature (K)", 800.0, 3500.0, kPeakKelvin,
            [this](double kelvin) { SetPeakTemperature(kelvin); });
        if (viewer != nullptr) {
            Write(viewer->Delegate(), viewer->Context());
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        const Tensor image = GetRenderer().Render(GetScene());
        const std::int64_t rows = image.Shape()[0];
        const std::int64_t cols = image.Shape()[1];
        // The brightest spot down the plume's axis (a shock diamond),
        // and the empty sky beside it.
        glm::dvec3 core(0.0);
        for (std::int64_t row = rows / 4; row + 8 <= rows; row += 4) {
            const glm::dvec3 spot = MeanIn(
                image, cols / 2 - 4, cols / 2 + 4, row, row + 8);
            if (spot.r > core.r) {
                core = spot;
            }
        }
        const glm::dvec3 sky = MeanIn(image, 0, 16, rows - 32, rows);
        std::printf("Plume core %.4f %.4f %.4f, sky %.4f %.4f %.4f\n",
                    core.r, core.g, core.b, sky.r, sky.g, sky.b);
        Check(sky.r + sky.g + sky.b < 0.01, "space is black");
        Check(core.r > 0.2, "the hot core glows");
        Check(core.r > core.b, "the glow is warm, as a black body's");
    }

    void Step() override {
        time_ += frameDt;
    }

    /// Stops the writers and waits for the frames that use the
    /// shaders before they go.
    void Close() override {
        for (const Written& written : written_) {
            for (const char* name : {"Density", "Temperature"}) {
                written.delegate->SetFieldWriter(
                    kPlume.Child(name), {kNx, kNy, kNz}, nullptr);
            }
            written.context->WaitIdle();
        }
        written_.clear();
        shaders_.clear();
    }

  private:
    void Write(RenderDelegate& delegate,
               const VulkanContext& context) {
        shaders_.push_back(std::make_unique<PlumeShader>(context));
        const PlumeShader* shader = shaders_.back().get();
        written_.push_back({&delegate, &context});
        for (int field = 0; field < 2; ++field) {
            delegate.SetFieldWriter(
                kPlume.Child(field == 0 ? "Density" : "Temperature"),
                {kNx, kNy, kNz},
                [this, shader,
                 field](VkCommandBuffer commands,
                        const FieldWriteTarget& target) {
                    shader->Write(commands, target, field, time_);
                });
        }
    }

    void SetPeakTemperature(double kelvin) {
        Material(GetScene(), "/World/Looks/Exhaust")
            .SetTemperatureScale(kelvin);
    }

    double time_ = 0.0;
    std::vector<std::unique_ptr<PlumeShader>> shaders_;
    /// The delegates writing the plume, and their devices.
    struct Written {
        RenderDelegate* delegate;
        const VulkanContext* context;
    };
    std::vector<Written> written_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    PlumeDemo, "volumes", "plume",
    "A rocket engine's plume in sunlight: black-body glow and soot, "
    "written on the GPU each frame.");
