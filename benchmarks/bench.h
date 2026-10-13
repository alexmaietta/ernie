// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Ernie's benchmarks: scenarios (scenes and what changes each step),
// presets (bundles of render settings), output sets (render vars) and
// suites (sweeps of scenarios' parameters), each registered by name.
//
// A case is a scenario and a value for each of its parameters, run
// through one fixed lifecycle (runner.h): build and setup, batch
// calibration, warm-up samples, measured samples, validation and
// teardown. Each measured sample times a batch of steps; the result
// keeps every sample, and summarises them without dropping any.
//
// References:
// - NVIDIA, Isaac Sim `isaacsim.performance.benchmarking` (the
//   operation, subsystem and workflow scopes, the lifecycle,
//   calibrated batching, completion boundaries, regions and recorded
//   metrics, the result document),
//   https://github.com/isaac-sim/IsaacSim
// - NVIDIA, Isaac Lab `isaaclab.benchmark` and
//   scripts/benchmarks/benchmark_cameras.py (autotuning a count until
//   a resource limit, paired comparisons reported with a confidence
//   interval), https://github.com/isaac-sim/IsaacLab

#include "ernie/geometry/mesh_data.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/scene.h"
#include "ernie/session/outputs.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {
class Renderer;
}

namespace ernie::bench {

/// Times a named block as a region of the sample (or setup phase) it
/// runs in, as Isaac Sim's `benchmark.measure`: nested regions join
/// their names with "/". Outside a run it does nothing.
class Measure {
  public:
    explicit Measure(std::string name);
    ~Measure();
    Measure(const Measure&) = delete;
    Measure& operator=(const Measure&) = delete;

  private:
    bool active_ = false;
    std::uint64_t start_ = 0;
};

/// Records `value` in `unit` as a metric of the sample running, as
/// Isaac Sim's `benchmark.record`; several in a sample average.
/// Outside a measured sample it does nothing.
void Record(const std::string& name, double value,
            const std::string& unit);

/// Keeps `value`'s computation from being optimised away.
template <typename T> void KeepAlive(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

/// One benchmark case: a scenario and its parameters' values by
/// name; unset ones take the scenario's defaults.
struct Case {
    std::string scenario;
    std::map<std::string, std::string> params;

    /// "scenario/name=value/...", the parameters in the scenario's
    /// order: what results are matched by and --filter matches.
    std::string Name() const;
};

/// What a benchmark measures, as Isaac Sim's scopes: an `operation`
/// (a small call, batched until a sample is long enough to time), a
/// `subsystem` (a part of the renderer, such as syncing a scene) or a
/// `workflow` (whole frames, as an application renders them).
enum class Scope { Operation, Subsystem, Workflow };

/// Where a step counts as done: `fenced`, each step waited for on the
/// host (its latency), or `pipelined`, a sample's steps submitted
/// back to back and only its last waited for (throughput, as a
/// learner consumes frames).
enum class Completion { Fenced, Pipelined };

/// How a scenario is sampled, and what is recorded besides time.
struct Definition {
    Scope scope = Scope::Subsystem;
    Completion completion = Completion::Pipelined;
    int warmupSamples = 3;     ///< Untimed samples after setup.
    int measuredSamples = 20;  ///< Independent timed samples kept.
    /// Steps in each sample; 0 calibrates, doubling from 1 until a
    /// sample takes `minSampleMs` (at most `maxSteps`).
    int stepsPerSample = 0;
    double minSampleMs = 50.0;
    int maxSteps = 1024;
    /// "process" (CPU time and resident memory) and "gpu" (GPU time
    /// from timestamp queries, per pass, and how busy that keeps the
    /// GPU; this process's device memory and its budget from
    /// VK_EXT_memory_budget): Vulkan's, on any vendor's GPU.
    std::vector<std::string> profiles = {"process", "gpu"};
    /// "short", "medium" or "long": how long a case takes, for
    /// choosing suites.
    std::string duration = "short";
};

std::string_view NameOf(Scope scope);
std::string_view NameOf(Completion completion);
Scope ScopeNamed(std::string_view name);
Completion CompletionNamed(std::string_view name);

/// A scenario's parameter, settable with `--set name=value`.
struct Param {
    std::string name;
    std::string defaultValue;
    std::string doc;
};

/// Something a step renders a number of, such as views or pixels,
/// reported as a rate: `count` per step, divided by `scale` (1e6 for
/// mega-) in the printed rate `name`/s.
struct Unit {
    std::string name;
    double count = 0.0;
    double scale = 1.0;
};

/// The paths of the prims every scenario's scene has by convention,
/// for presets and output sets.
struct ScenePaths {
    Path camera = "/World/Camera";
    Path sun = "/World/Sun";
    Path product = "/Render/Product";
    Path settings = "/Render/Settings";
};
inline const ScenePaths kPaths;

/// An instance of prototype `proto` in `world` (-1 every world).
struct Placement {
    int proto = 0;
    int world = -1;
    glm::vec3 position{0.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
    /// Linear RGB in place of the prototype's displayColor; negative
    /// keeps it. Set for every placement of an instancer or none.
    glm::vec3 colour{-1.0f};
};

/// A turn of `radians` about z.
glm::quat Yaw(float radians);

/// A scene to benchmark, made fresh for each case: `Build` adds its
/// prims, `Setup` its GPU data, and `Step` changes it before each
/// step; `Validate` checks the last frame and `Teardown` releases
/// what `Setup` made. The runner steps `Products` with the settings.
class Scenario {
  public:
    virtual ~Scenario() = default;

    /// The parameters, in the order case names list them, with
    /// defaults.
    virtual std::vector<Param> Params() const {
        return {};
    }

    /// How the case is sampled; the runner's options may override it.
    virtual Definition Describe(const Case& c) const;

    /// Adds the case's prims to an empty `scene`.
    virtual void Build(Scene& scene, const Case& c) = 0;

    /// Uploads what steps use, after `Build`; nothing by default.
    virtual void Setup(Renderer& renderer, Scene& scene,
                       const Case& c);

    /// Changes the scene before step `step`; nothing by default.
    virtual void Step(Renderer& renderer, Scene& scene, const Case& c,
                      int step);

    /// Checks the frame of a last, fenced step (its products' outputs
    /// on the host) after the measured samples; throws, or returns a
    /// reason, for a frame that is wrong. Passes by default.
    virtual std::string
    Validate(const Case& c, const RenderProductSetOutputs& outputs);

    /// Releases what `Setup` made; nothing by default.
    virtual void Teardown(Renderer& renderer, Scene& scene,
                          const Case& c);

    /// The products each step renders: `kPaths.product` by default.
    virtual std::vector<Path> Products(const Case& c) const;

    /// What a step renders, for the rates; none by default.
    virtual std::vector<Unit> Units(const Case& c) const;

    /// What `Build` made, such as triangles, for the results.
    const std::vector<std::pair<std::string, double>>& Stats() const {
        return stats_;
    }

    /// The parameter `name` of `c`, or its default from `Params`.
    /// @throws std::invalid_argument for a name `Params` lacks.
    std::string Get(const Case& c, std::string_view name) const;
    double GetDouble(const Case& c, std::string_view name) const;
    int GetInt(const Case& c, std::string_view name) const;

  protected:
    /// Adds `value` to the stat `name` of the built scene.
    void AddStat(const std::string& name, double value);

    /// Defines a PointInstancer at `path` of `prototypes` (prims
    /// defined already, usually beneath it) placed by `placements`,
    /// and counts its instances and triangles in the stats.
    void DefineInstances(Scene& scene, const Path& path,
                         const std::vector<Path>& prototypes,
                         const std::vector<Placement>& placements);

  private:
    std::vector<std::pair<std::string, double>> stats_;
};

struct ScenarioInfo {
    std::string name;
    std::string description;
    std::function<std::unique_ptr<Scenario>()> make;
};

/// A preset: a bundle of render settings applied to a scene's
/// `kPaths` prims, such as a tier of shading.
struct PresetInfo {
    std::string name;
    std::string description;
    std::function<void(Scene&)> apply;
};

/// An output set: the render vars a product renders, by source name.
struct OutputsInfo {
    std::string name;
    std::string description;
    std::vector<std::string> vars;
};

/// A suite: a named list of cases, usually sweeps (`Sweep`).
struct SuiteInfo {
    std::string name;
    std::string description;
    std::function<std::vector<Case>()> cases;
};

/// The cases of every combination of `axes`' values for each of
/// `scenarios`; an axis a scenario lacks is left out of its cases.
struct Sweep {
    std::vector<std::string> scenarios;
    std::vector<std::pair<std::string, std::vector<std::string>>>
        axes;

    std::vector<Case> Cases() const;
};

/// Values for a `Sweep` axis from numbers.
std::vector<std::string> Values(std::initializer_list<int> values);

/// `cases` and `more`, without repeats.
void Append(std::vector<Case>& cases, const std::vector<Case>& more);

// Registries, filled at static initialization by the macros below.
std::vector<ScenarioInfo>& Scenarios();
std::vector<PresetInfo>& Presets();
std::vector<OutputsInfo>& OutputSets();
std::vector<SuiteInfo>& Suites();

/// The registered entry named `name`.
/// @throws std::invalid_argument naming the registered ones.
const ScenarioInfo& FindScenario(std::string_view name);
const PresetInfo& FindPreset(std::string_view name);
const OutputsInfo& FindOutputs(std::string_view name);
const SuiteInfo& FindSuite(std::string_view name);

/// The parameters of scenario `name`.
std::vector<Param> ParamsOf(std::string_view name);

struct Registration {
    explicit Registration(ScenarioInfo info);
    explicit Registration(PresetInfo info);
    explicit Registration(OutputsInfo info);
    explicit Registration(SuiteInfo info);
};

#define ERNIE_BENCH_CONCAT_(a, b) a##b
#define ERNIE_BENCH_CONCAT(a, b) ERNIE_BENCH_CONCAT_(a, b)
#define ERNIE_BENCH_REGISTER_(info)                                  \
    static const ::ernie::bench::Registration ERNIE_BENCH_CONCAT(    \
        ernie_bench_registration_, __LINE__)(info)

/// Registers scenario `Class` as `name`.
#define ERNIE_BENCH_SCENARIO(Class, name, description)               \
    ERNIE_BENCH_REGISTER_((::ernie::bench::ScenarioInfo{             \
        name, description, [] {                                      \
            return std::unique_ptr<::ernie::bench::Scenario>(        \
                new Class());                                        \
        }}))

/// Registers a preset: `name`, `description`, and a callable taking
/// the scene.
#define ERNIE_BENCH_PRESET(name, description, ...)                   \
    ERNIE_BENCH_REGISTER_((                                          \
        ::ernie::bench::PresetInfo{name, description, __VA_ARGS__}))

/// Registers an output set: `name`, `description`, and its vars.
#define ERNIE_BENCH_OUTPUTS(name, description, ...)                  \
    ERNIE_BENCH_REGISTER_((::ernie::bench::OutputsInfo{              \
        name, description, __VA_ARGS__}))

/// Registers a suite: `name`, `description`, and a callable
/// returning its cases.
#define ERNIE_BENCH_SUITE(name, description, ...)                    \
    ERNIE_BENCH_REGISTER_(                                           \
        (::ernie::bench::SuiteInfo{name, description, __VA_ARGS__}))

// Helpers for building scenes.

/// Defines a Mesh prim at `path` from `mesh`: the inverse of
/// `MeshDataOf`, its texture coordinates as "st".
Mesh DefineMesh(Scene& scene, const Path& path, const MeshData& mesh);

/// Fills `mesh`'s texture coordinates, `perMetre` repeats along the
/// two axes across each triangle's dominant one, for textures on
/// the generators' meshes, which have none.
void ProjectUvs(MeshData& mesh, float perMetre = 1.0f);

/// A UsdPreviewSurface material at `path` whose diffuse colour is a
/// texture of `texels` (uint8 RGBA, `size` squared) read through
/// "st".
void DefineTexturedMaterial(Scene& scene, const Path& path,
                            const std::vector<std::uint8_t>& texels,
                            int size);

/// A procedural texture, uint8 RGBA `size` squared: checker squares
/// of `a` and `b`, `cells` along a side, with value noise of
/// `noise` (0 to 1) over them from `seed`.
std::vector<std::uint8_t> CheckerTexture(int size, int cells,
                                         glm::vec3 a, glm::vec3 b,
                                         float noise,
                                         std::uint32_t seed);

/// A pose at `eye` looking at `target`, z up.
glm::dmat4 LookAt(glm::dvec3 eye, glm::dvec3 target);

/// A small deterministic generator (PCG32) for procedural scenes.
class Random {
  public:
    /// Seeded as PCG's reference does, so that nearby seeds start far
    /// apart.
    explicit Random(std::uint64_t seed) {
        Next();
        state_ += seed;
        Next();
    }
    std::uint32_t Next();
    float Uniform(float lo = 0.0f, float hi = 1.0f);
    int Int(int lo, int hiInclusive);

  private:
    std::uint64_t state_ = 0;
};

}  // namespace ernie::bench
