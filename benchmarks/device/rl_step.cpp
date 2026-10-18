// SPDX-License-Identifier: BSD-3-Clause
/// @file rl_step.cpp
/// @brief A steady-state RL step of 16,384 worlds on device columns: poses written in place, then
/// turned into instance transforms, with no bulk bytes through the host.
#include "tests/device/rl_step.h"
#include "benchmarks/bench.h"

#include <cstdio>
#include <memory>

namespace ernie::bench {
namespace {

const bool kRlStep = Register("device.rl_step.poses_16k", [] {
    std::unique_ptr<DeviceContext> context;
    try {
        context = std::make_unique<DeviceContext>();
    } catch (const DeviceError& error) {
        std::printf("skipped: %s\n", error.what());
        return Timing{0.0, 1.0};
    }
    const ShaderCompiler compiler;
    test::RlStep rl(*context, compiler, 16'384);
    std::uint32_t step = 0;
    const auto run = [&] {
        const Ticket ticket = rl.Step(++step);
        ticket.Wait();
    };
    run();
    const Transfers before = context->Transfers();
    const double ms = Fastest([] {}, run, 20);
    if (context->Transfers() != before) {
        std::printf("device.rl_step.poses_16k moved bytes through the host\n");
    }
    return Timing{ms, 1.0};
});

}  // namespace
}  // namespace ernie::bench
