// SPDX-License-Identifier: BSD-3-Clause
// A spacecraft's reaction control thrusters firing short pulses in
// sunlight: each puff of exhaust spreads in a wide cone, a shell of
// gas racing out after the valve closes, a hot flash at the nozzle
// while it fires. A compute shader writes the gas into the renderer's
// grids each frame (RenderDelegate::SetFieldWriter).
//
// References:
// - G. A. Simons, "Effect of Nozzle Boundary Layers on Rocket Exhaust
//   Plumes", AIAA Journal 10(11), 1972, 1534-1535 (a plume's density
//   falling as the square of the distance and a power of the cosine
//   off its axis).

#include "ernie/app/demo.h"
#include "ernie/app/viewer.h"
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

constexpr std::uint32_t kNx = 72;
constexpr std::uint32_t kNy = 72;
constexpr std::uint32_t kNz = 56;
constexpr double kVoxel = 0.125;
/// Each thruster's pulse, s, of every period.
constexpr double kPulse = 0.08;
const Path kGas("/World/Gas");

/// The thrusters' exhaust `time` seconds in, as the Python twin's
/// Warp kernel: in 0.125 m voxels from (-4.5, -4.5, -2) m, eight
/// thrusters in four quads around the body's top, each pulsing
/// `pulse` s in turn every 1.2 s, the gas leaving at 4 m/s.
constexpr const char* kRcsSlang = R"(
static const int kNx = 72;
static const int kNy = 72;
static const int kNz = 56;
static const float kPi = 3.14159265358979;
static const float kPeriod = 1.2;
static const float kSpeed = 4.0;

struct Push {
    float* out;
    float time;
    float pulse;
    int field;
}
[[vk::push_constant]] ConstantBuffer<Push> push;

// The gas (`field` 0) or its heat (1) at each cell, z fastest.
[shader("compute")]
[numthreads(64, 1, 1)]
void rcs(uint3 id: SV_DispatchThreadID) {
    if (id.x >= uint(kNx * kNy * kNz)) {
        return;
    }
    let i = int(id.x) / (kNy * kNz);
    let j = (int(id.x) / kNz) % kNy;
    let k = int(id.x) % kNz;
    let p = float3(-4.5 + 0.125 * (float(i) + 0.5), -4.5 + 0.125 * (float(j) + 0.5),
                   -2.0 + 0.125 * (float(k) + 0.5));
    float gas = 0.0;
    float heat = 0.0;
    for (int t = 0; t < 8; ++t) {
        // Quad t / 2 around the body; its radial thruster, then its roll one.
        let angle = 0.5 * kPi * float(t / 2);
        let out = float3(cos(angle), sin(angle), 0.0);
        let axis = (t % 2) == 0 ? out : float3(-out.y, out.x, 0.0);
        let to = p - (1.15 * out + float3(0.0, 0.0, 1.5));
        let r = max(length(to), 0.05);
        let c = dot(to, axis) / r;
        if (c <= 0.0) {
            continue;
        }
        // Firing when the gas here left the nozzle, its pulse's edges soft.
        let left = push.time - r / kSpeed - float(t) * kPeriod / 8.0;
        let phase = left - kPeriod * floor(left / kPeriod);
        let on = clamp(phase / 0.01, 0.0, 1.0) * clamp((push.pulse - phase) / 0.01, 0.0, 1.0);
        let c2 = c * c;
        let c4 = c2 * c2;
        let c8 = c4 * c4;
        let spread = c8 * c8;
        let near = 0.25 / max(r, 0.25);
        gas += on * spread * near * near;
        heat = max(heat, on * spread * exp(-r / 0.15));
    }
    let value = push.field == 0 ? gas : heat;
    push.out[id.x] = value > 1e-4 ? value : 0.0;
}
)";

/// The exhaust's shader on one renderer's device, which its writers
/// run.
class RcsShader {
  public:
    explicit RcsShader(const VulkanContext& context)
        : layout_(context, {}, sizeof(Push)) {
        const ShaderCompiler compiler;
        const std::array<ShaderModuleSource, 1> modules{
            {{"rcs", kRcsSlang}}};
        pipeline_ = std::make_unique<ComputePipeline>(
            context, layout_, compiler.CompileSpirv(modules, "rcs"));
    }

    /// Records field `field` (gas 0, heat 1) at `time` into `target`.
    void Write(VkCommandBuffer commands,
               const FieldWriteTarget& target, int field, double time,
               double pulse) const {
        const Push push{target.values, static_cast<float>(time),
                        static_cast<float>(pulse), field};
        layout_.PushConstants(commands, &push, sizeof(push));
        pipeline_->Dispatch(commands,
                            std::span<const DescriptorResource>(),
                            (kNx * kNy * kNz + 63) / 64);
    }

  private:
    struct Push {
        VkDeviceAddress out;
        float time;
        float pulse;
        std::int32_t field;
    };

    PipelineLayout layout_;
    std::unique_ptr<ComputePipeline> pipeline_;
};

/// The image's sum of sRGB values.
double Sum(const Tensor& image) {
    const std::uint8_t* pixels = image.As<std::uint8_t>();
    double sum = 0.0;
    for (std::int64_t i = 0; i < image.Shape()[0] * image.Shape()[1];
         ++i) {
        sum += pixels[4 * i] + pixels[4 * i + 1] + pixels[4 * i + 2];
    }
    return sum;
}

class RcsDemo : public app::Demo {
  public:
    RcsDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        numFrames = 30;
        Scene& scene = GetScene();
        Xform::Define(scene, "/World");
        // The spacecraft, 2 m across, its thruster quads around its
        // top.
        Cylinder::Define(scene, "/World/Craft/Body")
            .SetRadius(1.0)
            .SetHeight(4.0)
            .SetMaterialBinding("/World/Looks/Paint");
        for (int q = 0; q < 4; ++q) {
            const double angle = 0.5 * std::numbers::pi * q;
            const Path quad("/World/Craft/Quad" + std::to_string(q));
            Sphere::Define(scene, quad)
                .SetRadius(0.15)
                .SetMaterialBinding("/World/Looks/Metal");
            SetXform(scene, quad,
                     {.translate =
                          glm::dvec3(1.15 * std::cos(angle),
                                     1.15 * std::sin(angle), 1.5)});
        }
        Material::Define(scene, "/World/Looks/Paint")
            .SetAlbedo(glm::dvec3(0.8));
        Material::Define(scene, "/World/Looks/Metal")
            .SetAlbedo(glm::dvec3(0.3));

        // The exhaust and its heat, written on the GPU each frame.
        Volume::Define(scene, kGas)
            .SetField("density", kGas.Child("Density"))
            .SetField("temperature", kGas.Child("Temperature"))
            .SetMaterialBinding("/World/Looks/Exhaust");
        for (const char* name : {"Density", "Temperature"}) {
            DenseField::Define(scene, kGas.Child(name))
                .SetVoxelSize(kVoxel);
            SetXform(
                scene, kGas.Child(name),
                {.translate = glm::dvec3(-4.4375, -4.4375, -1.9375)});
        }
        // Frozen propellant droplets: bright, scattering forward, and
        // a black-body glow where hot.
        Material::Define(scene, "/World/Looks/Exhaust")
            .SetDensity(kDensity)
            .SetVolumeAlbedo(glm::dvec3(0.9))
            .SetAnisotropy(0.3)
            .SetTemperatureScale(2500.0);

        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(1361.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetShadowEnable(true)
            .SetTransform(DistantLightTransform({0.6, -0.5, 0.4}));
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(28.0)
            .SetSensorNoise(false)
            .SetExposureIso(100.0)
            .SetExposureFStop(8.0)
            .SetExposureTime(1.0 / 500.0)
            .SetTransform(LookAtTransform(
                {9.0, -9.0, 4.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({480, 320})
            .SetAmbient(0.0);

        Write(GetRenderer().Delegate(), GetRenderer().Context());
        Widgets().Slider("Pulse (ms)", 20.0, 300.0, 1000.0 * kPulse,
                         [this](double ms) { pulse_ = 1e-3 * ms; });
        if (viewer != nullptr) {
            Write(viewer->Delegate(), viewer->Context());
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    void TestFinal() override {
        // The gas against none of it: what it scatters and glows.
        const double with = Sum(GetRenderer().Render(GetScene()));
        Material(GetScene(), "/World/Looks/Exhaust").SetDensity(0.0);
        const double without = Sum(GetRenderer().Render(GetScene()));
        Material(GetScene(), "/World/Looks/Exhaust")
            .SetDensity(kDensity);
        std::printf("Exhaust adds %.0f to %.0f\n", with - without,
                    without);
        Check(with > 1.02 * without, "the exhaust shows");
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
                    kGas.Child(name), {kNx, kNy, kNz}, nullptr);
            }
            written.context->WaitIdle();
        }
        written_.clear();
        shaders_.clear();
    }

  private:
    static constexpr double kDensity = 20.0;

    void Write(RenderDelegate& delegate,
               const VulkanContext& context) {
        shaders_.push_back(std::make_unique<RcsShader>(context));
        const RcsShader* shader = shaders_.back().get();
        written_.push_back({&delegate, &context});
        for (int field = 0; field < 2; ++field) {
            delegate.SetFieldWriter(
                kGas.Child(field == 0 ? "Density" : "Temperature"),
                {kNx, kNy, kNz},
                [this, shader,
                 field](VkCommandBuffer commands,
                        const FieldWriteTarget& target) {
                    shader->Write(commands, target, field, time_,
                                  pulse_);
                });
        }
    }

    double time_ = 0.0;
    double pulse_ = kPulse;
    std::vector<std::unique_ptr<RcsShader>> shaders_;
    /// The delegates writing the exhaust, and their devices.
    struct Written {
        RenderDelegate* delegate;
        const VulkanContext* context;
    };
    std::vector<Written> written_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    RcsDemo, "volumes", "rcs",
    "A spacecraft's reaction control thrusters pulsing in sunlight, "
    "their exhaust written on the GPU each frame.");
