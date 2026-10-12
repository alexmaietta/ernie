// SPDX-License-Identifier: BSD-3-Clause
// A dust devil on Mars, seen from a rover: a hollow column of dust
// 120 m tall, widening as it rises, swirling faster near its core and
// lifting a skirt of dust at its foot, lit by the sun and the dusty
// sky. A compute shader writes its dust into the renderer's grid each
// frame (RenderDelegate::SetFieldWriter).
//
// References:
// - M. Balme and R. Greeley, "Dust devils on Earth and Mars", Reviews
//   of Geophysics 44(3), RG3003, 2006 (their hollow, widening columns
//   and skirts, tens to hundreds of metres tall on Mars).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
#include "ernie/device/buffer.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/volume_schemas.h"
#include "ernie/scene/xform.h"
#include "ernie/util/noise.h"
#include "util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr std::uint32_t kNx = 64;
constexpr std::uint32_t kNy = 64;
constexpr std::uint32_t kNz = 256;
constexpr double kVoxel = 0.625;
constexpr double kDensity = 0.2;
/// The Sun's irradiance at Mars's mean distance, 1.5237 au.
constexpr double kMarsIrradiance = 1361.0 / (1.5237 * 1.5237);
const Path kDevil("/World/Devil");

/// The dust devil's dust `time` seconds in, as the Python twin's Warp
/// kernel: in 0.625 m voxels from (-20, -20, 0) m, a hollow wall
/// about an axis that wanders, turning faster towards the core and
/// rising, its turbulence GradientNoise's fBm (its permutation
/// `perm`).
constexpr const char* kDevilSlang = R"(
static const int kNx = 64;
static const int kNy = 64;
static const int kNz = 256;

struct Push {
    float* out;
    uint* perm;
    float time;
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

// The dust at each cell, z fastest.
[shader("compute")]
[numthreads(64, 1, 1)]
void devil(uint3 id: SV_DispatchThreadID) {
    if (id.x >= uint(kNx * kNy * kNz)) {
        return;
    }
    let i = int(id.x) / (kNy * kNz);
    let j = (int(id.x) / kNz) % kNy;
    let k = int(id.x) % kNz;
    let px = -20.0 + 0.625 * (float(i) + 0.5);
    let py = -20.0 + 0.625 * (float(j) + 0.5);
    let pz = 0.625 * (float(k) + 0.5);
    // Its axis wanders, more so higher up.
    let qx = px - 1.5 * sin(0.3 * push.time + 0.02 * pz);
    let qy = py - 1.5 * cos(0.23 * push.time + 0.015 * pz);
    let r = sqrt(qx * qx + qy * qy);
    let radius = 4.0 + 0.06 * pz;
    // Turning faster towards the core, and rising.
    let spin = 1.2 * push.time * radius / max(r, 0.5 * radius);
    let c = cos(spin);
    let s = sin(spin);
    let swirled = float3(c * qx + s * qy, c * qy - s * qx, pz - 6.0 * push.time);
    let turbulence = 0.5 + 0.5 * fbm(swirled * float3(0.25, 0.25, 0.08), 4);
    let ring = (r - radius) / (0.35 * radius);
    let inside = r / radius;
    let wall = exp(-ring * ring) + 0.25 * exp(-inside * inside);
    let column = wall * exp(-pz / 70.0) * clamp((128.0 - pz) / 20.0, 0.0, 1.0);
    let foot = r / 9.0;
    let skirt = 1.5 * exp(-pz / 2.0) * exp(-foot * foot);
    let dust = (column + skirt) * clamp(1.6 * turbulence - 0.2, 0.0, 2.0);
    push.out[id.x] = dust > 1e-3 ? dust : 0.0;
}
)";

/// The dust devil's shader on one renderer's device, which its writer
/// runs.
class DevilShader {
  public:
    explicit DevilShader(const VulkanContext& context)
        : layout_(context, {}, sizeof(Push)),
          permutation_(context, 512 * sizeof(std::uint32_t),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       BufferMemory::Device) {
        const ShaderCompiler compiler;
        const std::array<ShaderModuleSource, 1> modules{
            {{"devil", kDevilSlang}}};
        pipeline_ = std::make_unique<ComputePipeline>(
            context, layout_,
            compiler.CompileSpirv(modules, "devil"));
        const auto& table = GradientNoise(7).Permutation();
        const std::vector<std::uint32_t> words(table.begin(),
                                               table.end());
        permutation_.Write(std::as_bytes(std::span(words)));
    }

    /// Records the dust at `time` into `target`.
    void Write(VkCommandBuffer commands,
               const FieldWriteTarget& target, double time) const {
        const Push push{target.values, permutation_.Address(),
                        static_cast<float>(time)};
        layout_.PushConstants(commands, &push, sizeof(push));
        pipeline_->Dispatch(commands,
                            std::span<const DescriptorResource>(),
                            (kNx * kNy * kNz + 63) / 64);
    }

  private:
    struct Push {
        VkDeviceAddress out;
        VkDeviceAddress perm;
        float time;
    };

    PipelineLayout layout_;
    std::unique_ptr<ComputePipeline> pipeline_;
    GpuBuffer permutation_;
};

/// The images' mean absolute difference, in sRGB values.
double MeanDifference(const Tensor& a, const Tensor& b) {
    const std::uint8_t* x = a.As<std::uint8_t>();
    const std::uint8_t* y = b.As<std::uint8_t>();
    const std::int64_t pixels = a.Shape()[0] * a.Shape()[1];
    double sum = 0.0;
    for (std::int64_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            sum += std::abs(int(x[4 * i + c]) - int(y[4 * i + c]));
        }
    }
    return sum / static_cast<double>(3 * pixels);
}

class DustDevilDemo : public app::Demo {
  public:
    DustDevilDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 6;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // Mars's soil and air in a clear season, the sun 25 degrees up
        // and 50 to the view's left: the dust lit from behind,
        // scattering forward, clear of the sky's glow about the sun.
        demos::DefinePlanetSurface(scene, "/World/Mars", 3389.5e3,
                                   glm::dvec3(0.33, 0.22, 0.14),
                                   "/World/Looks/Soil");
        Atmosphere::Define(scene, "/World/Air")
            .SetPreset(std::string(tokens::kMars))
            .SetDustOpticalDepth(0.3)
            .SetGroundAlbedo(0.25);
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(kMarsIrradiance)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetAngle(0.35)
            .SetShadowEnable(true)
            .SetTransform(DistantLightTransform({0.265, 0.867, 0.423}));

        // The dust, written on the GPU each frame, its first cell at
        // (-19.6875, -19.6875, 0.3125) m.
        Volume::Define(scene, kDevil)
            .SetField("density", kDevil.Child("Density"))
            .SetMaterialBinding("/World/Looks/Dust");
        DenseField::Define(scene, kDevil.Child("Density"))
            .SetVoxelSize(kVoxel);
        SetXform(
            scene, kDevil.Child("Density"),
            {.translate = glm::dvec3(-19.6875, -19.6875, 0.3125)});
        // Martian dust: reddish, scattering forward.
        Material::Define(scene, "/World/Looks/Dust")
            .SetDensity(kDensity)
            .SetVolumeAlbedo(glm::dvec3(0.9, 0.75, 0.6))
            .SetAnisotropy(0.6);

        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetSensorNoise(false)
            .SetExposureTime(1.0 / 125.0)
            .SetExposureIso(125.0)
            .SetExposureFStop(11.0)
            .SetTransform(LookAtTransform({-160.0, -70.0, 2.5},
                                          {0.0, 0.0, 35.0},
                                          {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({360, 480})
            .SetAmbient(0.0);

        Write(GetRenderer().Delegate(), GetRenderer().Context());
        Widgets().Slider("Dust density", 0.0, 0.5, kDensity,
                         [this](double density) {
                             Material(GetScene(), "/World/Looks/Dust")
                                 .SetDensity(density);
                         });
        if (viewer != nullptr) {
            Write(viewer->Delegate(), viewer->Context());
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        // The devil against none: its dust shows against the sky.
        const Tensor with = GetRenderer().Render(GetScene());
        Material(GetScene(), "/World/Looks/Dust").SetDensity(0.0);
        const Tensor without = GetRenderer().Render(GetScene());
        Material(GetScene(), "/World/Looks/Dust")
            .SetDensity(kDensity);
        const double difference = MeanDifference(with, without);
        std::printf("The dust devil changes the image by %.2f\n",
                    difference);
        Check(difference > 1.0, "the dust devil shows");
    }

    void Step() override {
        time_ += frameDt;
    }

    /// Stops the writers and waits for the frames that use the
    /// shaders before they go.
    void Close() override {
        for (const Written& written : written_) {
            written.delegate->SetFieldWriter(
                kDevil.Child("Density"), {kNx, kNy, kNz}, nullptr);
            written.context->WaitIdle();
        }
        written_.clear();
        shaders_.clear();
    }

  private:
    void Write(RenderDelegate& delegate,
               const VulkanContext& context) {
        shaders_.push_back(std::make_unique<DevilShader>(context));
        const DevilShader* shader = shaders_.back().get();
        written_.push_back({&delegate, &context});
        delegate.SetFieldWriter(
            kDevil.Child("Density"), {kNx, kNy, kNz},
            [this, shader](VkCommandBuffer commands,
                           const FieldWriteTarget& target) {
                shader->Write(commands, target, time_);
            });
    }

    double time_ = 0.0;
    std::vector<std::unique_ptr<DevilShader>> shaders_;
    /// The delegates writing the dust, and their devices.
    struct Written {
        RenderDelegate* delegate;
        const VulkanContext* context;
    };
    std::vector<Written> written_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    DustDevilDemo, "volumes", "dust_devil",
    "A dust devil on Mars from a rover: a swirling hollow column of "
    "dust written on the GPU each frame.");
