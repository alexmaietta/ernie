// SPDX-License-Identifier: BSD-3-Clause
/// @file pose_step.cpp
/// @brief A steady-state step of 16,384 instances on device columns: poses written in place,
/// then turned into instance transforms, with no bulk bytes through the host.
#include "tests/device/pose_step.h"
#include "benchmarks/bench.h"

#include <cstdio>
#include <memory>

namespace ernie::bench {
namespace {

const bool kPoseStep = Register("device.pose_step.16k_instances", [] {
    std::unique_ptr<DeviceContext> context;
    try {
        context = std::make_unique<DeviceContext>();
    } catch (const DeviceError& error) {
        std::printf("skipped: %s\n", error.what());
        return Timing{0.0, 1.0};
    }
    const ShaderCompiler compiler;
    test::PoseStep poses(*context, compiler, 16'384);
    std::uint32_t step = 0;
    const auto run = [&] { poses.Step(++step).Wait(); };
    run();
    const Transfers before = context->Transfers();
    const double ms = Fastest([] {}, run, 20);
    if (context->Transfers() != before) {
        std::printf("device.pose_step.16k_instances moved bytes through the host\n");
    }
    return Timing{ms, 1.0};
});

}  // namespace
}  // namespace ernie::bench
