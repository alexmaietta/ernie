// SPDX-License-Identifier: BSD-3-Clause
#include <string>
#include <vector>

#include "ernie/app/cli.h"

int main(int argc, char** argv) {
    return ernie::app::RunCli(std::vector<std::string>(argv + 1, argv + argc));
}
