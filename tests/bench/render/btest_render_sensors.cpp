// SPDX-License-Identifier: BSD-3-Clause
#include <atomic>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "bench.h"
#include "ernie/device/queue.h"
#include "ernie/io/image.h"
#include "ernie/overlay/overlay.h"
#include "ernie/pipeline/builtins.h"
#include "ernie/pipeline/definition.h"
#include "ernie/render/pipelines.h"
#include "ernie/render/renderer.h"
#include "ernie/render/shaders.h"
#include "ernie/render/standalone.h"
#include "ernie/scene/rotation.h"
#include "ernie/scene/scene.h"

namespace ernie {
namespace {

constexpr uint32_t kWarmup = 5;
constexpr size_t kInflight = 2;
const std::vector<std::string> kProducts = {"/Render/Product"};

enum class Graph : int64_t { None, Raw, Isp };
enum class Returns : int64_t { Traced, Identity, Chain };
enum class Clouds : int64_t { Off, Markers, Colored };

void Props(Scene& scene) {
    scene.Define<DistantLight>("/Sun", {.orient = FromAzimuthElevation(60, 50)});
    scene.Define<DomeLight>("/Sky", {});
    scene.Define<Plane>("/Ground", {.width = 0, .length = 0});
    for (int i = 0; i < 12; ++i) {
        const double angle = i * 0.5236;
        scene.Define<Cube>(std::format("/Box{}", i),
                           {.translate = {12 * std::cos(angle), 12 * std::sin(angle), 1},
                            .displayColor = {0.3f + 0.05f * float(i), 0.5f, 0.6f},
                            .size = 2});
    }
}

Prim Node(Scene& scene, const std::string& path, const char* id, const Prim& input,
          const char* port, const char* output) {
    const Prim node = scene.Define<Shader>(path, {.id = id});
    node.Connect(port, input, output);
    return node;
}

std::string DefineCameraGraph(Scene& scene, Graph graph) {
    const Prim pipe = scene.Define<SensorPipeline>("/Pipe");
    const Prim signal = scene.Define<Shader>("/Pipe/Signal", {.id = "CameraSignal"});
    if (graph == Graph::Isp) {
        signal.Set(CameraSignal::kLayout, CameraSignalLayout::Mosaic);
    }
    const Prim integrate =
        Node(scene, "/Pipe/Integrate", "Integrate", signal, "electronRate", "electronRate");
    const Prim noise = Node(scene, "/Pipe/Noise", "Emva1288", integrate, "electrons", "electrons");
    const Prim adc = Node(scene, "/Pipe/Adc", "Adc", noise, "electrons", "electrons");
    const Prim gain = Node(scene, "/Pipe/Gain", "Gain", adc, "digital", "digital");
    if (graph == Graph::Raw) {
        pipe.Connect("raw", gain, "digital");
        return "raw";
    }
    const Prim black = Node(scene, "/Pipe/Black", "BlackLevel", gain, "digital", "digital");
    const Prim demosaic = Node(scene, "/Pipe/Demosaic", "Demosaic", black, "digital", "digital");
    const Prim color = Node(scene, "/Pipe/Color", "ToColor", demosaic, "digital", "digital");
    const Prim ccm = Node(scene, "/Pipe/Ccm", "Ccm", color, "color", "color");
    const Prim tone = Node(scene, "/Pipe/Tone", "Tone", ccm, "color", "color");
    tone.Set(Tone::kCurve, ToneCurve::Aces);
    const Prim encode = Node(scene, "/Pipe/Encode", "Encode", tone, "color", "color");
    pipe.Connect("display", encode, "display");
    return "display";
}

struct Stepper {
    Ref<Scene> scene;
    Renderer renderer;
    std::vector<std::string> products = kProducts;
    double units = 0;
};

void Warm(Stepper& stepper) {
    for (uint32_t i = 0; i < kWarmup; ++i) {
        stepper.renderer.Step(stepper.products, 1.0 / 30).Wait();
    }
}

void Nudge(Scene& scene, uint64_t step) {
    scene.Find("/Box0")->Set(Xformable::kTranslate, Double3{12, 0, 1 + 0.01 * double(step % 2)});
}

void Steps(benchmark::State& state, Stepper& stepper, const char* unit,
           const std::function<void(uint64_t)>& before = {}, bool moving = true) {
    std::deque<RenderOutput> flight;
    uint64_t step = 0;
    for (auto _ : state) {
        if (moving) {
            Nudge(*stepper.scene, step);
        }
        if (before) {
            before(step);
        }
        ++step;
        flight.push_back(stepper.renderer.Step(stepper.products, 1.0 / 30));
        if (flight.size() >= kInflight) {
            flight.front().Wait();
            flight.pop_front();
        }
    }
    for (const RenderOutput& output : flight) {
        output.Wait();
    }
    state.counters["FPS"] =
        benchmark::Counter(double(state.iterations()), benchmark::Counter::kIsRate);
    state.counters[unit] = benchmark::Counter(double(state.iterations()) * stepper.units * 1e-6,
                                              benchmark::Counter::kIsRate);
}

Stepper PrepareCamera(Graph graph, uint32_t cameras, uint32_t height) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice(), {.waitForKernels = true});
    Props(*scene);
    const std::string output =
        graph == Graph::None ? "ElectronRate" : DefineCameraGraph(*scene, graph);
    std::vector<std::string> paths;
    for (uint32_t camera = 0; camera < cameras; ++camera) {
        const std::string path = std::format("/Cam{}", camera);
        const double angle = camera * 6.2832 / cameras;
        const Double3 eye{4 * std::cos(angle), 4 * std::sin(angle), 1.5};
        const Prim prim = scene->Define<Camera>(
            path, {.translate = eye,
                   .orient = LookAt(eye, {12 * std::cos(angle), 12 * std::sin(angle), 1})});
        prim.Set(Camera::kExposureTime, 1e-4f);
        prim.Set(DetectorAPI::kDetectorPattern,
                 graph == Graph::Isp ? DetectorPattern::Rggb : DetectorPattern::Rgb);
        if (graph != Graph::None) {
            prim.Set(Camera::kPipelineBinding, std::vector<std::string>{"/Pipe"});
        }
        paths.push_back(path);
    }
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Out", {.sourceName = output});
    const uint32_t width = height * 4 / 3;
    scene->Define<RenderProduct>("/Render/Product", {.renderMode = RenderMode::RealTime,
                                                     .camera = paths,
                                                     .resolution = {width, height},
                                                     .orderedVars = {"/Render/Out"}});
    Stepper stepper{scene, Renderer(scene)};
    stepper.units = double(cameras) * width * height;
    Warm(stepper);
    return stepper;
}

void CameraSteps(benchmark::State& state) {
    const auto graph = Graph(state.range(0));
    const auto cameras = uint32_t(state.range(1));
    const auto height = uint32_t(state.range(2));
    Stepper& stepper =
        bench::Cached<Stepper>(std::format("camera/{}/{}/{}", int(graph), cameras, height),
                               [&] { return PrepareCamera(graph, cameras, height); });
    Steps(state, stepper, "Mpx/s");
}

void ValueEdits(benchmark::State& state) {
    const bool edits = state.range(0) != 0;
    const bool moving = state.range(1) != 0;
    Stepper& stepper =
        bench::Cached<Stepper>("edits", [&] { return PrepareCamera(Graph::Raw, 1, 720); });
    const Prim gain = *stepper.scene->Find("/Pipe/Gain");
    Steps(
        state, stepper, "Mpx/s",
        [&](uint64_t step) {
            if (edits) {
                gain.Set(Gain::kGain, step % 2 == 0 ? 1.5f : 2.0f);
            }
        },
        moving);
}

void Pipe(Scene& scene, Returns returns, const char* source,
          const std::vector<const char*>& chain) {
    if (returns == Returns::Traced) {
        return;
    }
    const Prim pipe = scene.Define<SensorPipeline>("/Pipe");
    Prim last = scene.Define<Shader>("/Pipe/Returns", {.id = source});
    if (returns == Returns::Chain) {
        for (const char* id : chain) {
            last = Node(scene, std::format("/Pipe/{}", id), id, last, "returns", "returns");
        }
    }
    const Prim cloud = Node(scene, "/Pipe/Cloud", "ToPointCloud", last, "returns", "returns");
    pipe.Connect("cloud", cloud, "cloud");
}

void CloudProduct(Scene& scene, const std::string& sensor, Returns returns,
                  std::vector<std::string> channels) {
    scene.Define<Scope>("/Render");
    scene.Define<RenderVar>("/Render/Points",
                            {.sourceName = returns == Returns::Traced ? "PointCloud" : "cloud",
                             .channels = std::move(channels)});
    scene.Define<RenderProduct>("/Render/Product",
                                {.camera = {sensor}, .orderedVars = {"/Render/Points"}});
}

std::vector<float> Elevations(uint32_t beams) {
    std::vector<float> elevations;
    for (uint32_t beam = 0; beam < beams; ++beam) {
        elevations.push_back(-25 + 40 * float(beam) / float(beams - 1));
    }
    return elevations;
}

Stepper PrepareLidar(Returns returns, uint32_t beams) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice(), {.waitForKernels = true});
    Props(*scene);
    Pipe(*scene, returns, "LidarReturns",
         {"RangeNoise", "IntensityNoise", "Dropout", "RangeQuantize"});
    if (returns == Returns::Chain) {
        scene->Find("/Pipe/Dropout")->Set(Dropout::kRate, 0.1f);
    }
    scene->Define<Lidar>("/Lidar", {.translate = {0, 0, 2},
                                    .orient = LookAlong({1, 0, 0}),
                                    .pipelineBinding = returns == Returns::Traced
                                                           ? std::vector<std::string>{}
                                                           : std::vector<std::string>{"/Pipe"},
                                    .firingsPerScan = 1800,
                                    .emitterElevations = Elevations(beams)});
    CloudProduct(*scene, "/Lidar", returns, {"Coordinates", "Distance", "Intensity"});
    Stepper stepper{scene, Renderer(scene)};
    stepper.units = double(beams) * 1800;
    Warm(stepper);
    return stepper;
}

void LidarSteps(benchmark::State& state) {
    const auto returns = Returns(state.range(0));
    const auto beams = uint32_t(state.range(1));
    Stepper& stepper = bench::Cached<Stepper>(std::format("lidar/{}/{}", int(returns), beams),
                                              [&] { return PrepareLidar(returns, beams); });
    Steps(state, stepper, "Mbeams/s");
}

Stepper PrepareRadar(Returns returns, uint32_t columns) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice(), {.waitForKernels = true});
    Props(*scene);
    Pipe(*scene, returns, "RadarReturns", {"RadarNoise", "Cfar"});
    if (returns == Returns::Chain) {
        scene->Find("/Pipe/RadarNoise")->Set(RadarNoise::kFloor, 0.01f);
    }
    const uint32_t rows = columns / 4;
    scene->Define<Radar>("/Radar", {.translate = {0, 0, 1},
                                    .orient = LookAlong({1, 0, 0}),
                                    .pipelineBinding = returns == Returns::Traced
                                                           ? std::vector<std::string>{}
                                                           : std::vector<std::string>{"/Pipe"},
                                    .fieldOfView = {120, 30},
                                    .rays = {columns, rows}});
    CloudProduct(*scene, "/Radar", returns,
                 {"Coordinates", "Distance", "Intensity", "RadialVelocity"});
    Stepper stepper{scene, Renderer(scene)};
    stepper.units = double(columns) * rows;
    Warm(stepper);
    return stepper;
}

void RadarSteps(benchmark::State& state) {
    const auto returns = Returns(state.range(0));
    const auto columns = uint32_t(state.range(1));
    Stepper& stepper = bench::Cached<Stepper>(std::format("radar/{}/{}", int(returns), columns),
                                              [&] { return PrepareRadar(returns, columns); });
    Steps(state, stepper, "Mrays/s");
}

struct Standalone {
    Ref<Scene> scene;
    std::unique_ptr<Pipeline> pipeline;
    PipelineRunDesc run;
    double pixels = 0;
};

Standalone PrepareStandalone(Graph graph, uint32_t height) {
    Standalone setup{Scene::Create(bench::BenchDevice())};
    DefineCameraGraph(*setup.scene, graph);
    setup.pipeline = std::make_unique<Pipeline>(setup.scene, "/Pipe");
    const uint32_t width = height * 16 / 9;
    const int64_t channels = graph == Graph::Isp ? 1 : 3;
    std::vector<float> rates(size_t(width) * height * size_t(channels));
    for (size_t at = 0; at < rates.size(); ++at) {
        rates[at] = 1e6f * float(1 + at % 97);
    }
    setup.run.inputs.emplace(
        "Signal", device::Tensor(bench::BenchDevice().CreateBuffer({.label = "rates"}, rates), 0,
                                 device::DType::Float32, {1, height, width, channels}));
    setup.run.detector.pattern = graph == Graph::Isp ? DetectorPattern::Rggb : DetectorPattern::Rgb;
    setup.run.exposureTime = 1e-3f;
    setup.pixels = double(width) * height;
    for (uint32_t i = 0; i < kWarmup; ++i) {
        for (const auto& [name, tensor] : setup.pipeline->Run(setup.run)) {
            tensor.GetDevice().GetQueue().WaitIdle();
        }
    }
    return setup;
}

void StandaloneRuns(benchmark::State& state) {
    const auto graph = Graph(state.range(0));
    const auto height = uint32_t(state.range(1));
    Standalone& setup =
        bench::Cached<Standalone>(std::format("standalone/{}/{}", int(graph), height),
                                  [&] { return PrepareStandalone(graph, height); });
    for (auto _ : state) {
        const PipelineOutputs outputs = setup.pipeline->Run(setup.run);
        outputs.begin()->second.GetDevice().GetQueue().WaitIdle();
    }
    state.counters["Mpx/s"] = benchmark::Counter(double(state.iterations()) * setup.pixels * 1e-6,
                                                 benchmark::Counter::kIsRate);
}

std::filesystem::path WriteNode() {
    static std::atomic<uint64_t> version{uint64_t(std::time(nullptr)) << 20};
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "ernie_bench_sensor_node.slang";
    std::ofstream(file) << std::format(R"(
import ernie.pipeline;
using namespace ernie;

// {}
struct BenchHotPixels : IPointNode {{
    [Input(0, 1)] float fraction = 1e-4;
    float level = 2000;

    Electrons Apply(Electrons e, PointContext c) {{
        return c.FixedUniform(0) < fraction ? e.Add(level) : e;
    }}
}}
)",
                                       version++);
    return file;
}

void Compile(benchmark::State& state) {
    const auto graph = Graph(state.range(0));
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice());
    DefineCameraGraph(*scene, graph);
    LoadShaders(*scene, WriteNode());
    const Prim hot =
        Node(*scene, "/Pipe/Hot", "BenchHotPixels", *scene->Find("/Pipe/Noise"), "e", "electrons");
    scene->Find("/Pipe/Adc")->Connect("electrons", hot, "electrons");
    PipelineRunDesc run;
    const int64_t channels = graph == Graph::Isp ? 1 : 3;
    run.inputs.emplace(
        "Signal", bench::BenchDevice().CreateTensor(device::DType::Float32, {1, 4, 4, channels}));
    run.detector.pattern = graph == Graph::Isp ? DetectorPattern::Rggb : DetectorPattern::Rgb;
    for (auto _ : state) {
        state.PauseTiming();
        LoadShaders(*scene, WriteNode());
        state.ResumeTiming();
        const Pipeline pipeline(scene, "/Pipe");
        const PipelineOutputs outputs = pipeline.Run(run);
        outputs.begin()->second.GetDevice().GetQueue().WaitIdle();
    }
    std::filesystem::remove(std::filesystem::temp_directory_path() /
                            "ernie_bench_sensor_node.slang");
}

void LoadNodes(benchmark::State& state) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice());
    for (auto _ : state) {
        benchmark::DoNotOptimize(LoadShaders(*scene, WriteNode()));
    }
    std::filesystem::remove(std::filesystem::temp_directory_path() /
                            "ernie_bench_sensor_node.slang");
}

struct Viewed {
    Stepper stepper;
    std::optional<RenderOutput> last;
};

Viewed PrepareViewed(uint32_t beams) {
    const Ref<Scene> scene = Scene::Create(bench::BenchDevice(), {.waitForKernels = true});
    Props(*scene);
    scene->Define<Lidar>("/Lidar", {.translate = {0, 0, 2},
                                    .orient = LookAlong({1, 0, 0}),
                                    .firingsPerScan = 1800,
                                    .emitterElevations = Elevations(beams)});
    const Double3 eye{20, 20, 15};
    scene->Define<Camera>("/View", {.translate = eye, .orient = LookAt(eye, {0, 0, 0})});
    scene->Define<Scope>("/Render");
    scene->Define<RenderVar>("/Render/Color", {.sourceName = "LdrColor"});
    scene->Define<RenderVar>(
        "/Render/Points", {.sourceName = "PointCloud", .channels = {"Coordinates", "Intensity"}});
    scene->Define<RenderProduct>("/Render/Product", {.camera = {"/View"},
                                                     .resolution = {1280, 720},
                                                     .orderedVars = {"/Render/Color"},
                                                     .overlays = {"*"}});
    scene->Define<RenderProduct>("/Render/Cloud",
                                 {.camera = {"/Lidar"}, .orderedVars = {"/Render/Points"}});
    Viewed viewed{Stepper{scene, Renderer(scene), {"/Render/Product", "/Render/Cloud"}}};
    viewed.stepper.units = double(beams) * 1800;
    Warm(viewed.stepper);
    return viewed;
}

void ViewerClouds(benchmark::State& state) {
    const auto clouds = Clouds(state.range(0));
    const auto beams = uint32_t(state.range(1));
    Viewed& viewed = bench::Cached<Viewed>(std::format("viewed/{}", beams),
                                           [&] { return PrepareViewed(beams); });
    Overlays& overlays = *viewed.stepper.renderer.GetOverlays();
    overlays.Remove("clouds");
    const uint32_t lidar = viewed.stepper.scene->Find("/Lidar")->Index();
    std::deque<RenderOutput> flight;
    uint64_t step = 0;
    for (auto _ : state) {
        Nudge(*viewed.stepper.scene, step++);
        if (clouds != Clouds::Off && viewed.last) {
            const RenderVarOutput& points =
                (*viewed.last)["/Render/Cloud"].frames.at(0).RenderVar("/Render/Points");
            OverlayMarkers markers{.points = points.channels.at("Coordinates").Select(0),
                                   .itemCount = points.channels.at("Counts"),
                                   .style = {.size = 3, .anchor = lidar, .pickable = false}};
            if (clouds == Clouds::Colored) {
                markers.colors =
                    ColorizeScalars(points.channels.at("Intensity"),
                                    {.colormap = Colormap::Turbo, .minimum = 0, .maximum = 1})
                        .Select(0);
            }
            Overlay overlay;
            overlay.Add(std::move(markers));
            overlays.Set("clouds", std::move(overlay));
        }
        flight.push_back(viewed.stepper.renderer.Step(viewed.stepper.products, 1.0 / 30));
        viewed.last = flight.back();
        if (flight.size() >= kInflight) {
            flight.front().Wait();
            flight.pop_front();
        }
    }
    for (const RenderOutput& output : flight) {
        output.Wait();
    }
    overlays.Remove("clouds");
    state.counters["FPS"] =
        benchmark::Counter(double(state.iterations()), benchmark::Counter::kIsRate);
}

const bool kCore = bench::Core("^Sensors/Camera/graph:[0-2]/cameras:1/height:1080/") &&
                   bench::Core("^Sensors/Camera/graph:2/cameras:16/height:240/") &&
                   bench::Core("^Sensors/ValueEdits/") &&
                   bench::Core("^Sensors/Lidar/returns:[0-2]/beams:128/") &&
                   bench::Core("^Sensors/Radar/returns:[0-2]/columns:960/") &&
                   bench::Core("^Sensors/Standalone/graph:[1-2]/height:2160/") &&
                   bench::Core("^Sensors/ViewerClouds/clouds:[0-2]/beams:128/");

BENCHMARK(CameraSteps)
    ->Name("Sensors/Camera")
    ->ArgNames({"graph", "cameras", "height"})
    ->ArgsProduct({{0, 1, 2}, {1}, {720, 1080}})
    ->ArgsProduct({{0, 1, 2}, {16}, {240}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(ValueEdits)
    ->Name("Sensors/ValueEdits")
    ->ArgNames({"edits", "moving"})
    ->ArgsProduct({{0, 1}, {0, 1}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(LidarSteps)
    ->Name("Sensors/Lidar")
    ->ArgNames({"returns", "beams"})
    ->ArgsProduct({{0, 1, 2}, {32, 128}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(RadarSteps)
    ->Name("Sensors/Radar")
    ->ArgNames({"returns", "columns"})
    ->ArgsProduct({{0, 1, 2}, {240, 960}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(StandaloneRuns)
    ->Name("Sensors/Standalone")
    ->ArgNames({"graph", "height"})
    ->ArgsProduct({{1, 2}, {720, 2160}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(Compile)
    ->Name("Sensors/Compile")
    ->ArgNames({"graph"})
    ->DenseRange(1, 2)
    ->Iterations(3)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(LoadNodes)
    ->Name("Sensors/LoadNodes")
    ->Iterations(3)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(ViewerClouds)
    ->Name("Sensors/ViewerClouds")
    ->ArgNames({"clouds", "beams"})
    ->ArgsProduct({{0, 1, 2}, {32, 128}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}
}
