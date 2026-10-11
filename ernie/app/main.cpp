// SPDX-License-Identifier: BSD-3-Clause
// The `ernie` program: the viewer on a scene file or a default scene,
// with a Demos panel of every demo linked in (ernie_demos), or a
// render of the scene without a window (ErnieMain).

#include "ernie/app/cli.h"

#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + (argc > 0 ? 1 : 0),
                                        argv + argc);
    return ernie::app::ErnieMain(args, ERNIE_VERSION);
}
