// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/pipelines.h"

#include "ernie/render/kernels.h"

namespace ernie {

std::shared_ptr<const CompiledPipeline>
CompilePipeline(device::Device& device,
                std::shared_ptr<const pipeline::PipelineDefinition> definition,
                const pipeline::NodeRegistry& registry) {
    auto compiled = std::make_shared<CompiledPipeline>();
    compiled->generated = pipeline::GeneratePipeline(*definition, registry);
    compiled->program =
        device.CreateProgram({.modules = {compiled->generated.module},
                              .library = &kRenderKernels,
                              .sources = {{compiled->generated.path, compiled->generated.source}},
                              .entries = compiled->generated.entries,
                              .label = "sensor pipeline"});
    compiled->definition = std::move(definition);
    return compiled;
}

void RecordPipeline(device::CommandEncoder& encoder, const CompiledPipeline& compiled,
                    const PipelineParams& params, uint32_t captures, uint64_t elements) {
    const pipeline::GeneratedPipeline& generated = compiled.generated;
    for (uint32_t index = 0; index < generated.entries.size(); ++index) {
        device::Kernel& kernel = compiled.program->Entry(generated.entries[index]);
        const pipeline::DispatchShape shape = generated.shapes[index];
        const uint64_t threads = shape == pipeline::DispatchShape::Elements ? elements
                                 : shape == pipeline::DispatchShape::ReduceGroups
                                     ? uint64_t(captures) * kReduceGroups * kPipelineGroup
                                     : captures;
        encoder.Dispatch(kernel, params,
                         device::FoldGroups(device::GroupCount(threads, kPipelineGroup)));
        encoder.Barrier();
    }
}

}
