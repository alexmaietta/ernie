// SPDX-License-Identifier: BSD-3-Clause
/// @file first_render.cpp
/// @brief The README's first C++ render against an installed libernie, exactly as the README has
/// it (CMake takes it from README.md); it reports a skip (77) on a machine with no device that
/// ray traces.
#include "first_render.inl"

#include <cstdio>
#include <stdexcept>

int main() {
    try {
        ReadmeMain();
    } catch (const std::runtime_error& error) {
        std::printf("Skipped: %s\n", error.what());
        return 77;
    }
    std::printf("Rendered the README's first scene\n");
    return 0;
}
