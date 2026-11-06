// SPDX-License-Identifier: BSD-3-Clause
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "ernie/viewer/cli/cli.h"

int main(int argc, char** argv) {
    ernie::app::Command command;
    try {
        command = ernie::app::ParseCommand(std::vector<std::string>(argv + 1, argv + argc));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 2;
    }
    try {
        return ernie::app::RunCommand(command);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ernie: %s\n", error.what());
        return 1;
    }
}
