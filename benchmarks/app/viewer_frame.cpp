// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer_frame.cpp
/// @brief The viewer's frame: an offscreen 1920 x 1080 viewer over a scene of a few hundred
/// prims, eco mode off, its time split by host and device profiler zone. `app.viewer_frame`
/// traces apart from the UI where the device has a compute queue; `app.viewer_frame_sync` traces
/// every frame before its UI.
#include "benchmarks/bench.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/xform.h"
#include "ernie/util/profiler.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ernie::bench {
namespace {

constexpr int kGroups = 20;
constexpr int kShapesPerGroup = 15;
/// Frames drawn before measuring, while pipelines compile and the layout settles.
constexpr std::uint64_t kWarmup = 30;
constexpr std::uint64_t kMeasured = 200;

void BuildScene(Scene& scene) {
    GroundPlane::Define(scene, "/World/Ground");
    for (int group = 0; group < kGroups; ++group) {
        const std::string root = "/World/Group" + std::to_string(group);
        Xform::Define(scene, root);
        SetXform(scene, root, {.translate = glm::dvec3(3.0 * (group % 5), 3.0 * (group / 5), 0)});
        for (int shape = 0; shape < kShapesPerGroup; ++shape) {
            const std::string path = root + "/Shape" + std::to_string(shape);
            if (shape % 2 == 0) {
                Sphere::Define(scene, path).SetRadius(0.2);
            } else {
                Cube::Define(scene, path).SetSize(0.3);
            }
            SetXform(scene, path,
                     {.translate = glm::dvec3(0.4 * (shape % 4), 0.4 * (shape / 4), 0.5)});
        }
    }
    scene.Update();
}

/// The average milliseconds a frame spends in each zone at most `deepest` deep, by name, in the
/// order they first end.
std::vector<std::pair<std::string, double>>
Average(const std::vector<profiler::Frame>& frames,
        std::vector<profiler::Zone> profiler::Frame::* zones, std::uint32_t deepest) {
    std::map<std::string, double> totals;
    std::vector<std::string> order;
    std::size_t counted = 0;
    for (const profiler::Frame& frame : frames) {
        // GPU zones of the newest frames may not have arrived.
        if ((frame.*zones).empty()) {
            continue;
        }
        ++counted;
        for (const profiler::Zone& zone : frame.*zones) {
            if (zone.thread != 0 || zone.depth > deepest) {
                continue;
            }
            const std::string name = std::string(2 * zone.depth, ' ') + zone.name;
            if (!totals.contains(name)) {
                order.push_back(name);
            }
            totals[name] += static_cast<double>(zone.end - zone.start);
        }
    }
    std::vector<std::pair<std::string, double>> averages;
    for (const std::string& name : order) {
        averages.emplace_back(name, totals[name] / static_cast<double>(counted) * 1e-6);
    }
    return averages;
}

Timing RunViewerFrame(bool async) {
    Scene scene;
    BuildScene(scene);
    app::ViewerOptions options{
        .size = {1920, 1080}, .offscreen = true, .frames = kWarmup, .asyncRender = async};
    options.shell.display.eco = false;
    // Zones every frame, not only while the Profiler panel's tab shows.
    options.shell.profile = true;
    profiler::SetHistory(kMeasured);
    app::Viewer viewer(scene, options);
    // A prim selected, as while working, so Properties draws its fields.
    viewer.GetShell().Selected().Set("/World/Group0/Shape0");
    viewer.Run();
    profiler::Clear();
    profiler::SetEnabled(true);
    viewer.RunFrames(kMeasured);
    profiler::SetEnabled(false);
    const std::vector<profiler::Frame> frames = profiler::Frames();
    std::vector<double> durations;
    for (const profiler::Frame& frame : frames) {
        durations.push_back(static_cast<double>(frame.end - frame.start) * 1e-6);
    }
    std::ranges::sort(durations);
    double frameMs = 0.0;
    for (const double ms : durations) {
        frameMs += ms;
    }
    frameMs /= static_cast<double>(durations.size());
    std::printf("  median %.3f ms, slowest %.3f ms a frame\n", durations[durations.size() / 2],
                durations.back());
    // A trace writes its pose first: the frames that started one.
    const auto traced = std::ranges::count_if(frames, [](const profiler::Frame& frame) {
        return std::ranges::any_of(frame.cpu, [](const profiler::Zone& zone) {
            return std::string_view(zone.name) == "Write the viewport pose";
        });
    });
    std::printf("  %zu frames of 1920 x 1080, %d prims, %s: %.3f ms a frame, %.0f frames and "
                "%.0f traces a second\n",
                frames.size(), kGroups * (kShapesPerGroup + 1) + 1,
                viewer.AsyncRender() ? "tracing apart from the UI" : "tracing each frame", frameMs,
                1e3 / frameMs,
                static_cast<double>(traced) / static_cast<double>(frames.size()) * 1e3 / frameMs);
    for (const auto& [title, zones, deepest] :
         {std::tuple("Host", &profiler::Frame::cpu, 2u), {"Device", &profiler::Frame::gpu, 2u}}) {
        std::printf("  %s\n", title);
        for (const auto& [name, ms] : Average(frames, zones, deepest)) {
            std::printf("    %-44s %8.3f ms\n", name.c_str(), ms);
        }
    }
    return {.ms = frameMs, .targetMs = 1.0};
}

const bool kRegistered = Register("app.viewer_frame", [] { return RunViewerFrame(true); });
const bool kRegisteredSync =
    Register("app.viewer_frame_sync", [] { return RunViewerFrame(false); });

}  // namespace
}  // namespace ernie::bench
