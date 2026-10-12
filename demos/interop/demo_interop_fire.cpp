// SPDX-License-Identifier: BSD-3-Clause
// A campfire simulated by Vulkan compute shaders inside each frame,
// as NVIDIA Flow's fire: fuel burns where it is hot, heating the gas
// and making soot, which rises, swirls and cools. Its soot and
// temperature are written straight into the renderer's volume grids
// (RenderDelegate::SetFieldWriter), glowing as a black body where
// hot. The Python twin runs the same solver in Warp.

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
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ernie;
using demos::Check;

constexpr std::uint32_t kNx = 64;
constexpr std::uint32_t kNy = 64;
constexpr std::uint32_t kNz = 128;
constexpr std::uint32_t kCells = kNx * kNy * kNz;
constexpr double kVoxel =
    0.03;  // m: a 1.9 m by 1.9 m by 3.8 m domain.
constexpr int kJacobi = 40;
constexpr float kVorticity = 0.04f;
constexpr double kKelvinScale = 2000.0;
const Path kFire("/World/Fire");

/// Flow's fire on a grid of cells, z fastest, each pass a dispatch
/// over every cell with its arrays as pointers, as the Python twin's
/// Warp kernels: combustion, vorticity confinement, a Jacobi pressure
/// projection and MacCormack advection.
///
/// References:
/// - NVIDIA Flow 2 (PhysX SDK, flow/source/nvflow/shaders/
///   AdvectionCommon.hlsli): its combustion model and constants.
/// - J. Stam, "Stable Fluids", SIGGRAPH 1999.
///   https://doi.org/10.1145/311535.311548
/// - R. Fedkiw, J. Stam and H. W. Jensen, "Visual Simulation of
/// Smoke",
///   SIGGRAPH 2001. https://doi.org/10.1145/383259.383260
/// - A. Selle et al., "An Unconditionally Stable MacCormack Method",
///   Journal of Scientific Computing 35, 2008.
constexpr const char* kSolverSlang = R"(
static const int kNx = 64;
static const int kNy = 64;
static const int kNz = 128;
static const float kIgnition = 0.05;
static const float kBurnPerTemperature = 4.0;
static const float kFuelPerBurn = 0.25;
static const float kTemperaturePerBurn = 5.0;
static const float kSmokePerBurn = 1.0;
static const float kCooling = 5.0;
static const float kBuoyancy = 25.0;

struct Push {
    float* a;
    float* b;
    float* c;
    float* d;
    float* e;
    float* f;
    float* g;
    float* h;
    float x;
    float y;
    float z;
    int n;
}
[[vk::push_constant]] ConstantBuffer<Push> push;

bool cellOf(uint id, out int3 cell) {
    cell = int3(int(id) / (kNy * kNz), (int(id) / kNz) % kNy, int(id) % kNz);
    return id < uint(kNx * kNy * kNz);
}

// `f` at cell (i, j, k), clamped into the grid.
float at(float* f, int i, int j, int k) {
    return f[(clamp(i, 0, kNx - 1) * kNy + clamp(j, 0, kNy - 1)) * kNz + clamp(k, 0, kNz - 1)];
}

float mix(float a, float b, float t) {
    return a + (b - a) * t;
}

// `f` trilinear at (x, y, z), in cells.
float sampleAt(float* f, float x, float y, float z) {
    let fx = floor(x);
    let fy = floor(y);
    let fz = floor(z);
    let i = int(fx);
    let j = int(fy);
    let k = int(fz);
    let u = x - fx;
    let v = y - fy;
    let w = z - fz;
    let c00 = mix(at(f, i, j, k), at(f, i + 1, j, k), u);
    let c10 = mix(at(f, i, j + 1, k), at(f, i + 1, j + 1, k), u);
    let c01 = mix(at(f, i, j, k + 1), at(f, i + 1, j, k + 1), u);
    let c11 = mix(at(f, i, j + 1, k + 1), at(f, i + 1, j + 1, k + 1), u);
    return mix(mix(c00, c10, v), mix(c01, c11, v), w);
}

// The least and greatest of `f`'s eight values around (x, y, z).
float2 rangeAt(float* f, float x, float y, float z) {
    let i = int(floor(x));
    let j = int(floor(y));
    let k = int(floor(z));
    var low = at(f, i, j, k);
    var high = low;
    for (int c = 1; c < 8; ++c) {
        let value = at(f, i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
        low = min(low, value);
        high = max(high, value);
    }
    return float2(low, high);
}

// A value in [0, 1) from integers, as the Python twin's.
float hash01(int x, int y, int z, int n) {
    uint h = uint(x) * 73856093u ^ uint(y) * 19349663u ^ uint(z) * 83492791u ^
             uint(n) * 2654435761u;
    h = h ^ (h >> 13u);
    h = h * 0x5BD1E995u;
    h = h ^ (h >> 15u);
    return float(h & 0xFFFFFFu) / 16777216.0;
}

// The bed feeds fuel and heat; hot fuel burns into heat and soot; the
// gas cools and the hot rises: a heat, b fuel, c smoke, d e f the flow,
// x dt, n the step.
[shader("compute")]
[numthreads(64, 1, 1)]
void burn(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    let n = id.x;
    let dt = push.x;
    let dx = float(c.x - kNx / 2);
    let dy = float(c.y - kNy / 2);
    let dz = float(c.z - 3) * 2.0;
    let r = sqrt(dx * dx + dy * dy + dz * dz);
    var t = push.a[n];
    var f = push.b[n];
    if (r < 9.0) {
        let flicker = hash01(c.x / 4, c.y / 4, c.z / 3, push.n / 4);
        let w = (1.0 - r / 9.0) * (0.4 + 0.6 * flicker);
        t = mix(t, 1.0, min(10.0 * dt, 1.0) * w);
        f = mix(f, 0.8, min(2.0 * dt, 1.0) * w);
        push.d[n] = push.d[n] + 1.5 * (hash01(c.x / 4, c.y / 4, c.z / 4, push.n / 4) - 0.5) * w;
        push.e[n] =
            push.e[n] + 1.5 * (hash01(c.x / 4 + 9, c.y / 4, c.z / 4, push.n / 4) - 0.5) * w;
    }
    var burnt = 0.0;
    if (t >= kIgnition && f > 0.0) {
        burnt = min(kBurnPerTemperature * (min(t, 1.0) - kIgnition) * dt, f / kFuelPerBurn);
    }
    f = f - kFuelPerBurn * burnt;
    t = min(t + kTemperaturePerBurn * burnt, 1.0);
    t = t - kCooling * dt * t;
    push.a[n] = t;
    push.b[n] = f;
    let edge = min(min(float(c.x), float(kNx - 1 - c.x)), min(float(c.y), float(kNy - 1 - c.y)));
    let fade = clamp(edge / 8.0, 0.0, 1.0);
    push.c[n] = (push.c[n] + kSmokePerBurn * burnt) * mix(0.9, 1.0, fade);
    push.f[n] = push.f[n] + kBuoyancy * t * dt;
}

// The flow a b c's vorticity into d e f.
[shader("compute")]
[numthreads(64, 1, 1)]
void curl(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    let i = c.x;
    let j = c.y;
    let k = c.z;
    push.d[id.x] = 0.5 * (at(push.c, i, j + 1, k) - at(push.c, i, j - 1, k) -
                          at(push.b, i, j, k + 1) + at(push.b, i, j, k - 1));
    push.e[id.x] = 0.5 * (at(push.a, i, j, k + 1) - at(push.a, i, j, k - 1) -
                          at(push.c, i + 1, j, k) + at(push.c, i - 1, j, k));
    push.f[id.x] = 0.5 * (at(push.b, i + 1, j, k) - at(push.b, i - 1, j, k) -
                          at(push.a, i, j + 1, k) + at(push.a, i, j - 1, k));
}

float spin(int i, int j, int k) {
    return length(float3(at(push.a, i, j, k), at(push.b, i, j, k), at(push.c, i, j, k)));
}

// Vorticity confinement: the vorticity a b c pushes the flow d e f, x
// the strength.
[shader("compute")]
[numthreads(64, 1, 1)]
void confine(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    let i = c.x;
    let j = c.y;
    let k = c.z;
    let n = 0.5 * float3(spin(i + 1, j, k) - spin(i - 1, j, k),
                         spin(i, j + 1, k) - spin(i, j - 1, k),
                         spin(i, j, k + 1) - spin(i, j, k - 1));
    let l = length(n);
    if (l > 1.0e-6) {
        let force = push.x * cross(n / l, float3(push.a[id.x], push.b[id.x], push.c[id.x]));
        push.d[id.x] = push.d[id.x] + force.x;
        push.e[id.x] = push.e[id.x] + force.y;
        push.f[id.x] = push.f[id.x] + force.z;
    }
}

// The flow a b c's divergence into d.
[shader("compute")]
[numthreads(64, 1, 1)]
void divergence(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    push.d[id.x] = 0.5 * (at(push.a, c.x + 1, c.y, c.z) - at(push.a, c.x - 1, c.y, c.z) +
                          at(push.b, c.x, c.y + 1, c.z) - at(push.b, c.x, c.y - 1, c.z) +
                          at(push.c, c.x, c.y, c.z + 1) - at(push.c, c.x, c.y, c.z - 1));
}

// A Jacobi iteration of the pressure a, b the divergence, into c.
[shader("compute")]
[numthreads(64, 1, 1)]
void jacobi(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    push.c[id.x] = (at(push.a, c.x + 1, c.y, c.z) + at(push.a, c.x - 1, c.y, c.z) +
                    at(push.a, c.x, c.y + 1, c.z) + at(push.a, c.x, c.y - 1, c.z) +
                    at(push.a, c.x, c.y, c.z + 1) + at(push.a, c.x, c.y, c.z - 1) -
                    push.b[id.x]) /
                   6.0;
}

// The flow b c d less the pressure a's gradient; the ground stops it.
[shader("compute")]
[numthreads(64, 1, 1)]
void project(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    let n = id.x;
    push.b[n] = push.b[n] - 0.5 * (at(push.a, c.x + 1, c.y, c.z) - at(push.a, c.x - 1, c.y, c.z));
    push.c[n] = push.c[n] - 0.5 * (at(push.a, c.x, c.y + 1, c.z) - at(push.a, c.x, c.y - 1, c.z));
    push.d[n] = push.d[n] - 0.5 * (at(push.a, c.x, c.y, c.z + 1) - at(push.a, c.x, c.y, c.z - 1));
    if (c.z == 0) {
        push.d[n] = max(push.d[n], 0.0);
    }
}

// a traced back along the flow c d e into b, x the step's cells per m/s.
[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    let n = id.x;
    push.b[n] = sampleAt(push.a, float(c.x) - push.c[n] * push.x,
                         float(c.y) - push.d[n] * push.x, float(c.z) - push.e[n] * push.x);
}

// MacCormack: a the field, b traced forward, c traced back, into d, the
// flow e f g; x cells per m/s, y the blend, z what is kept.
[shader("compute")]
[numthreads(64, 1, 1)]
void correct(uint3 id: SV_DispatchThreadID) {
    int3 c;
    if (!cellOf(id.x, c)) {
        return;
    }
    let n = id.x;
    let x = float(c.x) - push.e[n] * push.x;
    let y = float(c.y) - push.f[n] * push.x;
    let z = float(c.z) - push.g[n] * push.x;
    var value = push.b[n] + push.y * 0.5 * (push.a[n] - push.c[n]);
    let range = rangeAt(push.a, x, y, z);
    value = clamp(value, range.x, range.y);
    push.d[n] = push.z * value;
}

// a into b: a field for the renderer.
[shader("compute")]
[numthreads(64, 1, 1)]
void copyOut(uint3 id: SV_DispatchThreadID) {
    if (id.x < uint(kNx * kNy * kNz)) {
        push.b[id.x] = push.a[id.x];
    }
}
)";

/// The demo's time: the steps the solvers should have taken, and each
/// one's seconds.
struct FireState {
    int steps = 0;
    float dt = 1.0f / 60.0f;
};

/// The solver on one renderer's device: its grids and pipelines,
/// which the fire's writers step to the demo's time and copy out of.
class FireSolver {
  public:
    FireSolver(const VulkanContext& context, const FireState& state)
        : state_(state), layout_(context, {}, sizeof(Push)) {
        const ShaderCompiler compiler;
        const std::array<ShaderModuleSource, 1> modules{
            {{"fire", kSolverSlang}}};
        for (const char* entry :
             {"burn", "curl", "confine", "divergence", "jacobi",
              "project", "trace", "correct", "copyOut"}) {
            pipelines_.push_back(std::make_unique<ComputePipeline>(
                context, layout_,
                compiler.CompileSpirv(modules, entry)));
        }
        const std::vector<float> zeros(kCells, 0.0f);
        for (std::unique_ptr<GpuBuffer>& grid : grids_) {
            grid = std::make_unique<GpuBuffer>(
                context, kCells * sizeof(float),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
            grid->Write(std::as_bytes(std::span(zeros)));
        }
    }

    /// Steps to the demo's time, then writes field `field` (soot 0,
    /// temperature 1) into `target`.
    void Write(VkCommandBuffer commands,
               const FieldWriteTarget& target, int field) {
        while (done_ < state_.steps) {
            Step(commands, ++done_);
        }
        Push push{};
        push.a = Address(field == 0 ? kSmoke : kHeat);
        push.b = target.values;
        Run(commands, kCopyOut, push);
    }

    /// The field's values, for tests.
    std::vector<float> Read(int field) const {
        std::vector<float> values(kCells);
        grids_[slots_[field == 0 ? kSmoke : kHeat]]->Read(
            std::as_writable_bytes(std::span(values)));
        return values;
    }

  private:
    struct Push {
        VkDeviceAddress a, b, c, d, e, f, g, h;
        float x, y, z;
        std::int32_t n;
    };
    enum Grid {
        kHeat,
        kFuel,
        kSmoke,
        kVx,
        kVy,
        kVz,
        kWx,
        kWy,
        kWz,
        kPressure,
        kDiv,
        kA,
        kB,
        kC,
        kD,
        kE,
        kGrids
    };
    enum Pass {
        kBurn,
        kCurl,
        kConfine,
        kDivergence,
        kJacobiPass,
        kProject,
        kTrace,
        kCorrect,
        kCopyOut
    };

    VkDeviceAddress Address(Grid grid) const {
        return grids_[slots_[grid]]->Address();
    }

    void Swap(Grid a, Grid b) {
        std::swap(slots_[a], slots_[b]);
    }

    void Run(VkCommandBuffer commands, Pass pass,
             const Push& push) const {
        layout_.PushConstants(commands, &push, sizeof(push));
        pipelines_[pass]->Dispatch(
            commands, std::span<const DescriptorResource>(),
            (kCells + 63) / 64);
        ComputeBarrier(commands);
    }

    /// Carries `fields` along the flow into `outputs`, MacCormack's
    /// way, as the Python twin's `maccormack`.
    void MacCormack(VkCommandBuffer commands,
                    std::span<const Grid> fields,
                    std::span<const Grid> outputs, float blend,
                    std::span<const float> keep) {
        const float cells = state_.dt / static_cast<float>(kVoxel);
        for (std::size_t i = 0; i < fields.size(); ++i) {
            Push push{};
            push.a = Address(fields[i]);
            push.b = Address(kD);
            push.c = Address(kVx);
            push.d = Address(kVy);
            push.e = Address(kVz);
            push.x = cells;
            Run(commands, kTrace, push);
            push.a = Address(kD);
            push.b = Address(kE);
            push.x = -cells;
            Run(commands, kTrace, push);
            push = {};
            push.a = Address(fields[i]);
            push.b = Address(kD);
            push.c = Address(kE);
            push.d = Address(outputs[i]);
            push.e = Address(kVx);
            push.f = Address(kVy);
            push.g = Address(kVz);
            push.x = cells;
            push.y = blend;
            push.z = keep[i];
            Run(commands, kCorrect, push);
        }
    }

    /// Step `frame` (from 1), as the Python twin's `step`.
    void Step(VkCommandBuffer commands, int frame) {
        Push push{};
        push.a = Address(kHeat);
        push.b = Address(kFuel);
        push.c = Address(kSmoke);
        push.d = Address(kVx);
        push.e = Address(kVy);
        push.f = Address(kVz);
        push.x = state_.dt;
        push.n = frame;
        Run(commands, kBurn, push);
        push = {};
        push.a = Address(kVx);
        push.b = Address(kVy);
        push.c = Address(kVz);
        push.d = Address(kWx);
        push.e = Address(kWy);
        push.f = Address(kWz);
        Run(commands, kCurl, push);
        push = {};
        push.a = Address(kWx);
        push.b = Address(kWy);
        push.c = Address(kWz);
        push.d = Address(kVx);
        push.e = Address(kVy);
        push.f = Address(kVz);
        push.x = kVorticity;
        Run(commands, kConfine, push);
        push = {};
        push.a = Address(kVx);
        push.b = Address(kVy);
        push.c = Address(kVz);
        push.d = Address(kDiv);
        Run(commands, kDivergence, push);
        for (int i = 0; i < kJacobi; ++i) {
            push = {};
            push.a = Address(kPressure);
            push.b = Address(kDiv);
            push.c = Address(kA);
            Run(commands, kJacobiPass, push);
            Swap(kPressure, kA);
        }
        push = {};
        push.a = Address(kPressure);
        push.b = Address(kVx);
        push.c = Address(kVy);
        push.d = Address(kVz);
        Run(commands, kProject, push);
        // The flow carried by itself, from its values before.
        const std::array<Grid, 3> velocity{kVx, kVy, kVz};
        const std::array<Grid, 3> spare{kA, kB, kC};
        MacCormack(commands, velocity, spare, 0.5f,
                   std::array<float, 3>{0.999f, 0.999f, 0.999f});
        Swap(kVx, kA);
        Swap(kVy, kB);
        Swap(kVz, kC);
        // Soot fades slowly; fuel and heat carried as they are.
        const std::array<Grid, 3> scalars{kHeat, kFuel, kSmoke};
        MacCormack(commands, scalars, spare, 0.9f,
                   std::array<float, 3>{1.0f, 1.0f, 0.99f});
        Swap(kHeat, kA);
        Swap(kFuel, kB);
        Swap(kSmoke, kC);
    }

    const FireState& state_;
    PipelineLayout layout_;
    std::vector<std::unique_ptr<ComputePipeline>> pipelines_;
    std::array<std::unique_ptr<GpuBuffer>, kGrids> grids_;
    /// Which of `grids_` holds each `Grid` now, as passes swap them.
    std::array<int, kGrids> slots_ = [] {
        std::array<int, kGrids> slots{};
        for (int i = 0; i < kGrids; ++i) {
            slots[i] = i;
        }
        return slots;
    }();
    int done_ = 0;
};

class FireDemo : public app::Demo {
  public:
    FireDemo(app::Viewer* viewer, const app::DemoOptions& options)
        : Demo(viewer, options) {
        // Headless, while the twins' solvers still agree: past about
        // 60 steps their float rounding grows into a different fire.
        numFrames = 45;
        DefineScene(GetScene());
        // Solved in the frames of whichever renderers draw the scene.
        Write(GetRenderer().Delegate(), GetRenderer().Context());
        if (viewer != nullptr) {
            Write(viewer->Delegate(), viewer->Context());
            viewer->SetActiveCamera("/World/Camera");
        }
    }

    void Step() override {
        state_.steps += 1;
        state_.dt = static_cast<float>(frameDt);
    }

    void SaveOutput(const std::filesystem::path& path) override {
        app::WritePng(path, GetRenderer().Render(GetScene()));
    }

    /// Stops the writers and waits for the frames that use the
    /// solvers' grids before they go.
    void Close() override {
        for (const Written& written : written_) {
            written.delegate->SetFieldWriter(
                kFire.Child("Density"), {kNx, kNy, kNz}, nullptr);
            written.delegate->SetFieldWriter(
                kFire.Child("Heat"), {kNx, kNy, kNz}, nullptr);
            written.context->WaitIdle();
        }
        written_.clear();
        solvers_.clear();
    }

    void TestFinal() override {
        const Tensor image = GetRenderer().Render(GetScene());
        const std::vector<float> smoke = solvers_.front()->Read(0);
        const std::vector<float> heat = solvers_.front()->Read(1);
        double sum = 0.0;
        double moment = 0.0;
        for (std::uint32_t n = 0; n < kCells; ++n) {
            sum += smoke[n];
            moment += smoke[n] * (n % kNz) * kVoxel;
        }
        const double rise = moment / sum;
        const float hottest =
            *std::max_element(heat.begin(), heat.end());
        std::printf("soot's mean height %.2f m, hottest %.2f\n", rise,
                    hottest);
        Check(rise > 0.3, "the soot should rise");
        Check(hottest > 0.5f, "the fire should be hot");
        const auto* texels = image.As<std::uint8_t>();
        const std::int64_t pixels =
            image.Shape()[0] * image.Shape()[1];
        std::int64_t warm = 0;
        for (std::int64_t i = 0; i < pixels; ++i) {
            const std::uint8_t* t = texels + 4 * i;
            warm += t[0] - t[2] > 40 && t[0] > 120 ? 1 : 0;
        }
        const double glowing =
            static_cast<double>(warm) / static_cast<double>(pixels);
        std::printf("glowing pixels %.2f%%\n", 100.0 * glowing);
        Check(glowing > 0.005, "the flames should glow");
    }

  private:
    /// The ground at dusk, the fire's light, its soot and the camera,
    /// as the Python twin's.
    static void DefineScene(Scene& scene) {
        Xform::Define(scene, "/World");
        GroundPlane::Define(scene, "/World/Ground")
            .SetMaterialBinding("/World/Looks/Earth");
        Material::Define(scene, "/World/Looks/Earth")
            .SetAlbedo(glm::dvec3(0.1, 0.08, 0.06));
        DistantLight::Define(scene, "/World/Sun")
            .SetIrradiance(5.0)
            .SetPhysicalIlluminant(std::string(tokens::kSun))
            .SetTransform(DistantLightTransform({-1.0, 0.4, 0.12}));
        SphereLight::Define(scene, "/World/Glow")
            .SetRadius(0.3)
            .SetColor(glm::dvec3(1.0, 0.5, 0.15))
            .SetIntensity(60.0);
        SetXform(scene, "/World/Glow",
                 {.translate = glm::dvec3(0.0, 0.0, 0.25)});
        Volume::Define(scene, kFire)
            .SetField("density", kFire.Child("Density"))
            .SetField("temperature", kFire.Child("Heat"))
            .SetMaterialBinding("/World/Looks/Soot");
        // The grid's first cell at the domain's corner, its floor on
        // the ground; the values come from the writers.
        const glm::dvec3 low(-0.5 * kNx * kVoxel, -0.5 * kNy * kVoxel,
                             0.0);
        for (const char* name : {"Density", "Heat"}) {
            DenseField::Define(scene, kFire.Child(name))
                .SetVoxelSize(kVoxel);
            SetXform(scene, kFire.Child(name), {.translate = low});
        }
        // Soot: absorbing more than it scatters, so it glows as a
        // black body where hot and is dark grey where it has cooled.
        Material::Define(scene, "/World/Looks/Soot")
            .SetDensity(2.0)
            .SetVolumeAlbedo(glm::dvec3(0.5))
            .SetAnisotropy(0.3)
            .SetTemperature(300.0)
            .SetTemperatureScale(kKelvinScale)
            .SetScatteringOctaves(3);
        Camera::Define(scene, "/World/Camera")
            .SetFocalLength(28.0)
            .SetSensorNoise(false)
            .SetExposureIso(800.0)
            .SetExposureFStop(4.0)
            .SetExposureTime(1.0 / 60.0)
            .SetTransform(LookAtTransform(
                {0.0, -6.0, 1.4}, {0.0, 0.0, 1.3}, {0.0, 0.0, 1.0}));
        RenderSettings::Define(scene, "/Render/Settings")
            .SetCamera(Path("/World/Camera"))
            .SetResolution({480, 480})
            .SetAmbient(0.0);
    }

    void Write(RenderDelegate& delegate,
               const VulkanContext& context) {
        solvers_.push_back(
            std::make_unique<FireSolver>(context, state_));
        FireSolver* solver = solvers_.back().get();
        written_.push_back({&delegate, &context});
        for (int field = 0; field < 2; ++field) {
            delegate.SetFieldWriter(
                kFire.Child(field == 0 ? "Density" : "Heat"),
                {kNx, kNy, kNz},
                [solver, field](VkCommandBuffer commands,
                                const FieldWriteTarget& target) {
                    solver->Write(commands, target, field);
                });
        }
    }

    FireState state_;
    std::vector<std::unique_ptr<FireSolver>> solvers_;
    /// The delegates writing the fire, and their devices.
    struct Written {
        RenderDelegate* delegate;
        const VulkanContext* context;
    };
    std::vector<Written> written_;
};

}  // namespace

ERNIE_REGISTER_DEMO(
    FireDemo, "interop", "fire",
    "A campfire simulated by Vulkan compute shaders in "
    "each frame, as NVIDIA Flow's fire, written into the "
    "renderer's volume grids.");
