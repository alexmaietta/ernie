// SPDX-License-Identifier: BSD-3-Clause
// The `main` of each demo's own program, demo_<category>_<name>: runs
// that demo (ERNIE_DEMO_ID) with every C++ demo in its Demos panel,
// taking the options every demo takes.

#include "ernie/app/demo.h"

#include <cstdio>
#include <optional>
#include <vector>

int main(int argc, char** argv) {
    std::vector<ernie::app::DemoInfo> demos =
        ernie::app::RegisteredDemos();
    const std::optional<std::size_t> self =
        ernie::app::FindDemo(demos, ERNIE_DEMO_ID);
    if (!self) {
        std::fprintf(stderr, "The demo %s is not registered\n",
                     ERNIE_DEMO_ID);
        return 1;
    }
    return ernie::app::DemoMain(argc, argv, std::move(demos), *self);
}
