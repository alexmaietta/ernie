// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/shader_compiler.h"
#include "ernie/scene/brdf.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::BrdfModel;
using ernie::BrdfRegistry;
using ernie::ShaderCompiler;
using ernie::ShaderModuleSource;

namespace {

constexpr std::uint32_t kSpirvMagic = 0x07230203;

/// A compute shader evaluating `model`, the way hosts compose
/// reflectance models.
std::vector<ShaderModuleSource> ProbeModules(const BrdfModel& model) {
    std::string probe = R"(import brdf;
import MODULE;

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> results;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    BrdfInput s = {0.8, 0.9, 0.95, 1.0, float3(0.2)};
    BrdfParams p;
    p.values[0] = float4(0.1, 0.2, 0.3, 0.4);
    p.values[1] = float4(0.5, 0.3, 0.0, 0.0);
    p.values[2] = float4(0.0);
    p.values[3] = float4(0.0);
    TYPE m;
    results[0] = float4(m.radianceFactor(s, p), 1.0);
}
)";
    probe.replace(probe.find("MODULE"), 6,
                  ernie::BrdfModuleName(model));
    probe.replace(probe.find("TYPE"), 4, model.typeName);
    std::vector<ShaderModuleSource> modules =
        ernie::BrdfModules(model);
    modules.push_back({"probe", probe});
    return modules;
}

}  // namespace

TEST(ShaderCompiler, CompilesEveryBuiltinModelToSpirvAndWgsl) {
    const ShaderCompiler compiler;
    const BrdfRegistry registry = BrdfRegistry::WithBuiltins();
    for (const BrdfModel& model : registry.Models()) {
        const auto modules = ProbeModules(model);
        const std::vector<std::uint32_t> spirv =
            compiler.CompileSpirv(modules, "main");
        ASSERT_GT(spirv.size(), 5u) << model.id;
        EXPECT_EQ(spirv[0], kSpirvMagic) << model.id;
        const std::string wgsl =
            compiler.CompileWgsl(modules, "main");
        EXPECT_NE(wgsl.find("@compute"), std::string::npos)
            << model.id << "\n"
            << wgsl;
    }
}

TEST(ShaderCompiler, ReportsErrorsWithSlangDiagnostics) {
    const ShaderCompiler compiler;
    const std::vector<ShaderModuleSource> broken = {
        {"broken", "float3 f() { return undefined_symbol; }"}};
    try {
        compiler.CompileSpirv(broken, "main");
        ADD_FAILURE() << "compiled a broken module";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("broken failed to compile"),
                  std::string::npos)
            << message;
        EXPECT_NE(message.find("undefined_symbol"), std::string::npos)
            << message;
    }

    const std::vector<ShaderModuleSource> noEntry = {
        {"plain", "float twice(float x) { return 2 * x; }"}};
    EXPECT_THROW(compiler.CompileSpirv(noEntry, "main"),
                 std::invalid_argument);
    EXPECT_THROW(compiler.CompileWgsl({}, "main"),
                 std::invalid_argument);
}

TEST(ShaderCompiler, CachesSpirvOnDiskUntilTheSourcesChange) {
    namespace fs = std::filesystem;
    const fs::path cache =
        fs::temp_directory_path() /
        ("ernie_shader_cache_" +
         std::to_string(
             ::testing::UnitTest::GetInstance()->random_seed()));
    fs::remove_all(cache);
    const char* previous = std::getenv("ERNIE_SHADER_CACHE");
    const std::string saved = previous != nullptr ? previous : "";
    setenv("ERNIE_SHADER_CACHE", cache.string().c_str(), 1);
    std::vector<ShaderModuleSource> modules = {{"cached", R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<float> values;
[shader("compute")]
[numthreads(1, 1, 1)]
void main() { values[0] = 1.0; })"}};
    {
        const ShaderCompiler compiler;
        const std::vector<std::uint32_t> first =
            compiler.CompileSpirv(modules, "main");
        EXPECT_EQ(compiler.CacheHits(), 0u);
        ASSERT_EQ(std::distance(fs::directory_iterator(cache),
                                fs::directory_iterator()),
                  1);
        const ShaderCompiler again;
        EXPECT_EQ(again.CompileSpirv(modules, "main"), first);
        EXPECT_EQ(again.CacheHits(), 1u) << "a new compiler reads it";

        modules[0].source += "\n// changed";
        again.CompileSpirv(modules, "main");
        EXPECT_EQ(again.CacheHits(), 1u) << "a changed source misses";

        // A damaged entry compiles afresh.
        for (const fs::directory_entry& entry :
             fs::directory_iterator(cache)) {
            std::ofstream(entry.path(), std::ios::trunc) << "junk";
        }
        const ShaderCompiler damaged;
        modules[0].source.resize(modules[0].source.size() - 11);
        EXPECT_EQ(damaged.CompileSpirv(modules, "main"), first);
        EXPECT_EQ(damaged.CacheHits(), 0u);
    }
    setenv("ERNIE_SHADER_CACHE", "0", 1);
    {
        const ShaderCompiler off;
        off.CompileSpirv(modules, "main");
        off.CompileSpirv(modules, "main");
        EXPECT_EQ(off.CacheHits(), 0u);
    }
    if (previous != nullptr) {
        setenv("ERNIE_SHADER_CACHE", saved.c_str(), 1);
    } else {
        unsetenv("ERNIE_SHADER_CACHE");
    }
    fs::remove_all(cache);
}

TEST(ShaderCompiler, CompilesCustomModelsAgainstTheSharedModule) {
    BrdfModel custom;
    custom.id = "half_lambert";
    custom.typeName = "HalfLambert";
    custom.slang =
        "import brdf;\n"
        "public struct HalfLambert : IBrdf {\n"
        "    public float3 radianceFactor(BrdfInput s, BrdfParams p) "
        "{\n"
        "        return 0.5 * s.albedo * s.mu0 * p.get(0);\n"
        "    }\n"
        "}\n";
    const ShaderCompiler compiler;
    EXPECT_EQ(compiler.CompileSpirv(ProbeModules(custom), "main")[0],
              kSpirvMagic);

    custom.slang =
        "import brdf;\npublic struct HalfLambert : IBrdf {}\n";
    EXPECT_THROW(compiler.CompileSpirv(ProbeModules(custom), "main"),
                 std::invalid_argument)
        << "missing interface method";
}
