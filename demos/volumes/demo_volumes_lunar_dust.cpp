// SPDX-License-Identifier: BSD-3-Clause
// A lander hovering over the Moon, its exhaust blowing a sheet of
// dust outward: a low sun lights the dust, which shades the ground
// under it. A compute shader writes the dust, streaming out from
// under the engine, into the renderer's grid each frame
// (RenderDelegate::SetFieldWriter).

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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

/// The image's mean linear RGB over rows [top, bottom).
glm::dvec3 MeanRows(const Tensor& image, std::int64_t top,
                    std::int64_t bottom) {
    const std::int64_t cols = image.Shape()[1];
    const std::uint8_t* pixels = image.As<std::uint8_t>();
    glm::dvec3 sum(0.0);
    for (std::int64_t r = top; r < bottom; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            const std::uint8_t* p = pixels + 4 * (r * cols + c);
            sum += demos::SrgbToLinear(glm::dvec3(p[0], p[1], p[2]) /
                                       255.0);
        }
    }
    return sum / static_cast<double>((bottom - top) * cols);
}

constexpr std::uint32_t kNx = 192;
constexpr std::uint32_t kNy = 192;
constexpr std::uint32_t kNz = 24;
constexpr double kVoxel = 0.25;
const Path kDust("/World/Dust");

/// The dust's density `time` seconds in, as the Python twin's Warp
/// kernel: over a 48 m square, 6 m deep, in 0.25 m voxels, a sheet
/// streaming out from under the engine at 8 m/s, streaked along its
/// rays and billowing (GradientNoise's fBm, its permutation `perm`),
/// thinning with height and distance.
constexpr const char* kDustSlang = R"(
static const int kNx = 192;
static const int kNy = 192;
static const int kNz = 24;

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
void dust(uint3 id: SV_DispatchThreadID) {
    if (id.x >= uint(kNx * kNy * kNz)) {
        return;
    }
    let i = int(id.x) / (kNy * kNz);
    let j = (int(id.x) / kNz) % kNy;
    let k = int(id.x) % kNz;
    let px = -24.0 + 0.25 * (float(i) + 0.5);
    let py = -24.0 + 0.25 * (float(j) + 0.5);
    let pz = 0.25 * (float(k) + 0.5);
    let r = sqrt(px * px + py * py);
    let c = r > 0.0 ? px / r : 1.0;
    let s = r > 0.0 ? py / r : 0.0;
    // Out from 1.5 m, fading over 10 m; thicker further out.
    let radial = clamp((r - 1.5) / 2.0, 0.0, 1.0) * exp(-r / 10.0);
    let vertical = exp(-pz / (0.2 + 0.05 * r));
    // Streaks along the rays, and billows, streaming out.
    let travelled = 8.0 * push.time;
    let streaks = 0.5 + 0.5 * fbm(float3(6.0 * c, 6.0 * s, 0.15 * (r - travelled)), 3);
    let billows =
        0.5 + 0.5 * fbm(float3(0.6 * (px - travelled * c) + 17.0, 0.6 * (py - travelled * s) + 17.0,
                               0.6 * pz + 17.0), 4);
    let value = radial * vertical * clamp(2.5 * streaks - 1.0, 0.0, 1.0) * (0.5 + billows);
    push.out[id.x] = value > 1e-3 ? value : 0.0;
}
)";

/// The dust's shader on one renderer's device, which its writer runs.
class DustShader {
  public:
    explicit DustShader(const VulkanContext& context)
        : layout_(context, {}, sizeof(Push)),
          permutation_(context, 512 * sizeof(std::uint32_t),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       BufferMemory::Device) {
        const ShaderCompiler compiler;
        const std::array<ShaderModuleSource, 1> modules{
            {{"dust", kDustSlang}}};
        pipeline_ = std::make_unique<ComputePipeline>(
            context, layout_, compiler.CompileSpirv(modules, "dust"));
        const auto table = GradientNoise(11).Permutation();
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

class LunarDustDemo : public app::Demo {
  public:
    LunarDustDemo(app::Viewer* viewer,
                  const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 12;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Regolith");
        Material::Define(scene, "/World/Looks/Regolith")
            .SetAlbedo(glm::dvec3(0.12));
        // The lander 4 m up: its body, legs and engine bell.
        Cube::Define(scene, "/World/Lander/Body")
            .SetSize(1.0)
            .SetMaterialBinding("/World/Looks/Foil");
        SetXform(scene, "/World/Lander/Body",
                 {.translate = glm::dvec3(0.0, 0.0, 5.5),
                  .scale = glm::dvec3(3.0, 3.0, 2.0)});
        for (int leg = 0; leg < 4; ++leg) {
            const double angle =
                std::numbers::pi * (0.25 + 0.5 * leg);
            const std::string path =
                "/World/Lander/Leg" + std::to_string(leg);
            Cylinder::Define(scene, path)
                .SetRadius(0.08)
                .SetHeight(2.0)
                .SetMaterialBinding("/World/Looks/Metal");
            SetXform(scene, path,
                     {.translate =
                          glm::dvec3(2.0 * std::cos(angle),
                                     2.0 * std::sin(angle), 4.0)});
        }
        Cone::Define(scene, "/World/Lander/Bell")
            .SetRadius(0.5)
            .SetHeight(0.8)
            .SetMaterialBinding("/World/Looks/Metal");
        SetXform(scene, "/World/Lander/Bell",
                 {.translate = glm::dvec3(0.0, 0.0, 4.1)});
        Material::Define(scene, "/World/Looks/Foil")
            .SetAlbedo(glm::dvec3(0.35, 0.28, 0.12));
        Material::Define(scene, "/World/Looks/Metal")
            .SetAlbedo(glm::dvec3(0.3));

        // The dust, written on the GPU each frame, its first cell at
        // (-23.875, -23.875, 0.125) m: grains that scatter about half
        // the light, mostly back towards the sun, as regolith does
        // (Hapke's double Henyey-Greenstein lobes).
        Volume::Define(scene, kDust)
            .SetField("density", kDust.Child("Density"))
            .SetMaterialBinding("/World/Looks/Dust");
        DenseField::Define(scene, kDust.Child("Density"))
            .SetVoxelSize(kVoxel);
        SetXform(scene, kDust.Child("Density"),
                 {.translate = glm::dvec3(-23.875, -23.875, 0.125)});
        Material::Define(scene, "/World/Looks/Dust")
            .SetDensity(kDensity)
            .SetVolumeAlbedo(glm::dvec3(0.55))
            .SetPhaseFunction(
                std::string(tokens::kDoubleHenyeyGreenstein));
        scene.Set("/World/Looks/Dust", tokens::kPhaseB, 0.3);
        scene.Set("/World/Looks/Dust", tokens::kPhaseC, 0.6);

        // A low sun from the left.
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetShadowEnable(true)
            .SetTransform(DistantLightTransform({-1.0, -0.7, 0.3}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(24.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 250.0)
            .SetTransform(LookAtTransform(
                {8.0, -22.0, 2.5}, {0.0, 0.0, 2.5}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({480, 320})
            .SetAmbient(0.0);

        Write(GetRenderer().Delegate(), GetRenderer().Context());
        Widgets().Slider("Dust density (1/m)", 0.0, 2.0, kDensity,
                         [this](double density) {
                             Material(GetScene(), "/World/Looks/Dust")
                                 .SetDensity(density);
                         });
        if (viewer != nullptr) {
            Write(viewer->Delegate(), viewer->Context());
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void Step() override {
        time_ += frameDt;
    }

    /// Stops the writers and waits for the frames that use the
    /// shaders before they go.
    void Close() override {
        for (const Written& written : written_) {
            written.delegate->SetFieldWriter(
                kDust.Child("Density"), {kNx, kNy, kNz}, nullptr);
            written.context->WaitIdle();
        }
        written_.clear();
        shaders_.clear();
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        Scene& scene = GetScene();
        const Tensor dusty = GetRenderer().Render(scene);
        scene.Set("/World/Dust", tokens::kVisibility,
                  std::string(tokens::kInvisible));
        const Tensor clear = GetRenderer().Render(scene);
        scene.Set("/World/Dust", tokens::kVisibility,
                  std::string(tokens::kInherited));
        const std::int64_t rows = dusty.Shape()[0];
        // The sky above, and the ground beyond the lander, seen
        // through the dust or without it.
        const glm::dvec3 sky = MeanRows(dusty, 0, rows / 8);
        const auto band = [&](const Tensor& image) {
            return MeanRows(image, rows / 2 + rows / 16,
                            rows / 2 + rows / 8);
        };
        const glm::dvec3 with = band(dusty);
        const glm::dvec3 without = band(clear);
        std::printf("Sky %.4f, ground band with dust %.4f %.4f %.4f, "
                    "without %.4f %.4f %.4f\n",
                    sky.r + sky.g + sky.b, with.r, with.g, with.b,
                    without.r, without.g, without.b);
        Check(sky.r + sky.g + sky.b < 0.01,
              "the Moon's sky is black");
        Check(std::abs((with.r + with.g + with.b) -
                       (without.r + without.g + without.b)) > 0.01,
              "the dust shows over the ground");
    }

  private:
    static constexpr double kDensity = 0.3;

    void Write(RenderDelegate& delegate,
               const VulkanContext& context) {
        shaders_.push_back(std::make_unique<DustShader>(context));
        const DustShader* shader = shaders_.back().get();
        written_.push_back({&delegate, &context});
        delegate.SetFieldWriter(
            kDust.Child("Density"), {kNx, kNy, kNz},
            [this, shader](VkCommandBuffer commands,
                           const FieldWriteTarget& target) {
                shader->Write(commands, target, time_);
            });
    }

    double time_ = 0.0;
    std::vector<std::unique_ptr<DustShader>> shaders_;
    /// The delegates writing the dust, and their devices.
    struct Written {
        RenderDelegate* delegate;
        const VulkanContext* context;
    };
    std::vector<Written> written_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    LunarDustDemo, "volumes", "lunar_dust",
    "A lander's exhaust blowing a sheet of lunar dust in a low sun, "
    "streaming out, written on the GPU each frame.");
