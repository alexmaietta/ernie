// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/program.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "support/buffers.h"
#include "support/devices.h"

namespace ernie::device {
namespace {

using test::Download;
using test::ForEachDevice;

static_assert(GroupCount(0, 64) == 0);
static_assert(GroupCount(65, 64) == 2);
static_assert(FoldGroups(10) == Size3{10, 1, 1});
static_assert(FoldGroups(kMaxGroupsPerAxis + 1) == Size3{32768, 2, 1});

struct AddParams {
    uint64_t words;
    uint32_t count;
    uint32_t add;
};

const char* const kAddSource = R"(
struct AddParams {
    uint* words;
    uint count;
    uint add;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void Add(uniform AddParams p, uint3 id: SV_DispatchThreadID) {
    if (id.x < p.count) {
        p.words[id.x] += p.add;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void Nothing() {}
)";

Ref<Program> AddProgram(Device& device) {
    RegisterFile("test_add.slang", kAddSource);
    return device.CreateProgram({.modules = {"test_add"}, .entries = {"Add"}, .label = "add"});
}

TEST(ProgramTest, DispatchesDirectlyAndIndirectly) {
    ForEachDevice([](Device& device) {
        const Ref<Program> program = AddProgram(device);
        Kernel& add = program->Entry("Add");
        add.Wait();
        EXPECT_TRUE(add.Ready());
        EXPECT_EQ(add.Name(), "Add");
        EXPECT_EQ(&add.GetProgram(), program.Get());
        EXPECT_GE(add.Stats().compileMs, 0);

        std::vector<uint32_t> words(100);
        std::iota(words.begin(), words.end(), 0u);
        const Ref<Buffer> buffer = device.CreateBuffer({.label = "words"}, words);
        const Ref<Buffer> groups = device.CreateBuffer(
            {.usage = BufferUsage::Indirect, .label = "groups"}, std::vector{1u, 1u, 1u});
        const AddParams params{buffer->Address(), 100, 3};
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        encoder.Dispatch(add, params, {.x = GroupCount(100, 64)});
        encoder.Barrier();
        encoder.Dispatch(add, params, {.x = 2}, *groups);
        device.GetQueue().Submit(encoder.Finish());

        for (uint32_t i = 0; i < words.size(); ++i) {
            words[i] += i < 64 ? 6 : 3;
        }
        EXPECT_EQ(Download<uint32_t>(device, *buffer, 100), words);
    });
}

TEST(ProgramTest, ReportsKernelsThatCannotBeBuilt) {
    ForEachDevice([](Device& device) {
        const Ref<Program> program = AddProgram(device);
        EXPECT_NO_THROW(program->Entry("Nothing").Wait());
        EXPECT_THROW(program->Entry("NoSuchEntry").Wait(), Error);
        EXPECT_THROW(device.CreateProgram({.modules = {"test_no_such_module"}}), Error);

        RegisterFile("test_broken.slang", "void Broken( {");
        Kernel& broken = device.CreateProgram({.modules = {"test_broken"}})->Entry("Broken");
        try {
            broken.Wait();
            FAIL() << "a module that doesn't parse compiled";
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find("test_broken"), std::string::npos);
        }
    });
}

}
}
