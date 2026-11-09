// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/algorithms/kernels.h"
#include "ernie/bvh/kernels.h"
#include "ernie/device/compute/runtime_compiler.h"
#include "ernie/device/cuda/api.h"
#include "ernie/device/cuda/device.h"
#include "ernie/device/hip/api.h"
#include "ernie/device/hip/device.h"
#include "ernie/device/slang/compiler.h"
#include "ernie/io/kernels.h"
#include "ernie/overlay/kernels.h"
#include "ernie/render/kernels.h"
#include "ernie/scene/kernels.h"
#include "ernie/texture/kernels.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/error.h"

namespace ernie::device {
namespace {

struct RuntimeProgram {
    const KernelLibrary* library;
    std::string module;
    std::vector<const char*> entries;
};

std::vector<RuntimeProgram> RuntimePrograms() {
    return {
        {&kAlgorithmsKernels, "ernie.algorithms.kernels.radix_sort", {"CountDigits", "Scatter"}},
        {&kAlgorithmsKernels, "ernie.algorithms.kernels.scan", {"SumTiles", "ScanTiles"}},
        {&kBvhKernels, "ernie.bvh.kernels.instances", {"MaskInstances", "GatherWorlds"}},
        {&kBvhKernels,
         "ernie.bvh.kernels.lbvh",
         {"Prepare", "Morton", "Hierarchy", "FitBoxes", "Seed", "Advance", "Collapse", "Finish"}},
        {&kBvhKernels, "ernie.bvh.kernels.refit", {"RefitLeaves", "RefitNodes"}},
        {&kIoKernels,
         "ernie.io.kernels.colorize",
         {"ScalarRange", "ColorizeScalars", "ColorizeHdr", "ColorizeNormals", "ColorizeIds"}},
        {&kOverlayKernels,
         "ernie.overlay.kernels.project",
         {"ProjectOverlays", "ScatterOverlays", "FindOverlayTiles"}},
        {&kOverlayKernels,
         "ernie.overlay.kernels.groups",
         {"KeyOverlayRows", "StartOverlayGroups", "MeasureOverlayGroups"}},
        {&kOverlayKernels,
         "ernie.overlay.kernels.composite",
         {"FoldOverlayTiles", "CompositeOverlays", "BlendOverlay"}},
        {&kOverlayKernels,
         "ernie.overlay.kernels.selection",
         {"ClearOverlayBoxes", "UniteOverlayBoxes", "OutlineOverlays"}},
        {&kOverlayKernels,
         "ernie.overlay.kernels.picks",
         {"ClearOverlayPicks", "MeasureOverlayPicks", "ChooseOverlayPicks", "ResolveOverlayPicks"}},
        {&kRenderKernels, "ernie.render.kernels.query.ray_cast", {"CastClosest", "CastAny"}},
        {&kRenderKernels,
         "ernie.render.kernels.render.render",
         {"RenderMinimalData", "RenderMinimalDisplayColor", "RenderMinimalTextureDiffuse",
          "RenderMinimalFull", "RenderRealTime", "RenderRealTimeBands", "RenderLidar",
          "RenderScatterPoints", "RenderRadar"}},
        {&kSceneKernels,
         "ernie.stage.columns.kernels.cells",
         {"CopyWords", "ScatterCells", "GatherCells", "CloneRows", "FillCells", "WritePoses",
          "ReadPoses", "GatherRows"}},
        {&kSceneKernels,
         "ernie.scene.tables.kernels.bounds",
         {"PrimBounds", "SlotBounds", "InstancerBounds", "FinishBoxes"}},
        {&kSceneKernels,
         "ernie.stage.columns.kernels.changes",
         {"StampRows", "CompactChanged", "CompactWorlds"}},
        {&kSceneKernels,
         "ernie.scene.tables.kernels.meshes",
         {"SmoothNormals", "FinishShapeBounds", "FaceSizes", "FaceTriangles", "TopologySizes",
          "EmitTriangles", "WriteAdjacency", "WriteStarts"}},
        {&kSceneKernels, "ernie.scene.tables.kernels.transforms", {"ResolveRows", "WriteMotions"}},
        {&kSceneKernels, "ernie.scene.tables.kernels.instances", {"WriteInstances"}},
        {&kSceneKernels,
         "ernie.scene.tables.kernels.lights",
         {"WriteDomeMaps", "WriteDomeRows", "WriteDomeMarginals", "WriteLights", "WriteLightCdf",
          "WriteCollections", "WriteLightShapes"}},
        {&kSceneKernels, "ernie.scene.tables.kernels.materials", {"WriteMaterials"}},
        {&kSceneKernels,
         "ernie.scene.tables.kernels.sensors",
         {"WriteCameras", "WriteLidars", "WriteRadars"}},
        {&kSceneKernels, "ernie.scene.tables.kernels.surfaces", {"WriteSurfaces"}},
        {&kSceneKernels,
         "ernie.scene.tables.kernels.tangents",
         {"GroupTangents", "CornerTangents"}},
        {&kTextureKernels, "ernie.texture.kernels.mips", {"DownsampleLevels"}},
    };
}

bool EnvironmentFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string_view(value) == "1";
}

template <class Traits, class Load>
void CompileEveryKernel(Backend backend, Load load, std::span<const char* const> targets,
                        std::span<const char* const> everyKernelTargets) {
    if (!EnvironmentFlag("ERNIE_TEST_FULL")) {
        GTEST_SKIP() << "compiles for every target only when ERNIE_TEST_FULL=1";
    }
    const compute::RuntimeCompiler* compiler = nullptr;
    try {
        compiler = &load();
    } catch (const UnsupportedError& error) {
        if (EnvironmentFlag("ERNIE_TEST_REQUIRE_COMPILERS")) {
            FAIL() << error.what();
        }
        GTEST_SKIP() << error.what();
    }
    const DiskCache cache(UserCacheDirectory());
    const SourceFiles sources;
    const compute::SourceHeader prelude{kPreludeHeader, Traits::Prelude()};
    bool first = true;
    for (const RuntimeProgram& program : RuntimePrograms()) {
        const std::vector<std::string> modules = {program.module};
        for (const char* entry : program.entries) {
            SCOPED_TRACE(program.module + "." + entry);
            const KernelCode code =
                Compile(backend, Traversal::Software, Sampling::Software, false, program.library,
                        sources, modules, entry, "runtime compilers", cache);
            for (const char* target : targets) {
                if (!first && std::ranges::find(everyKernelTargets, std::string_view(target)) ==
                                  everyKernelTargets.end()) {
                    continue;
                }
                SCOPED_TRACE(target);
                std::string object;
                EXPECT_NO_THROW(
                    object = compiler
                                 ->CompileCached(
                                     code.code, {&prelude, 1},
                                     Traits::CompileOptions(target, code.shape.groupSize, false),
                                     entry, cache)
                                 .code);
                EXPECT_FALSE(object.empty());
            }
            first = false;
        }
    }
}

TEST(RuntimeCompilersTest, KernelsCompileForEveryNvidiaTarget) {
    const char* const targets[] = {"sm_75", "sm_86", "sm_90", "sm_120"};
    CompileEveryKernel<cuda::Traits>(Backend::Cuda, cuda::GetNvrtc, targets, targets);
}

TEST(RuntimeCompilersTest, KernelsCompileForEveryAmdTarget) {
    const char* const targets[] = {"gfx90a", "gfx942", "gfx950", "gfx1100"};
    const char* const everyKernel[] = {"gfx942", "gfx1100"};
    CompileEveryKernel<hip::Traits>(Backend::Hip, hip::GetHiprtc, targets, everyKernel);
}

}
}
