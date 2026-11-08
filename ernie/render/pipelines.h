// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>

#include "ernie/device/commands.h"
#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/pipeline/codegen.h"
#include "ernie/render/kernels/pipeline/pipeline.h"

namespace ernie {

struct CompiledPipeline {
    std::shared_ptr<const pipeline::PipelineDefinition> definition;
    pipeline::GeneratedPipeline generated;
    Ref<device::Program> program;
};

std::shared_ptr<const CompiledPipeline>
CompilePipeline(device::Device& device,
                std::shared_ptr<const pipeline::PipelineDefinition> definition,
                const pipeline::NodeRegistry& registry);

void RecordPipeline(device::CommandEncoder& encoder, const CompiledPipeline& compiled,
                    const PipelineParams& params, uint32_t captures, uint64_t elements);

}
