// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/app/demo.h"
#include "ernie/scene/scene.h"

#include <span>
#include <string>
#include <string_view>

namespace ernie::app {

/// Defines in `scene` what the `ernie` program opens on without a
/// scene file or demo: a ground plane with a cube, a sphere and a
/// cylinder on it, lit by a sun, and a camera for renders without a
/// viewer.
void DefineDefaultScene(Scene& scene);

/// The `ernie` program, also the `ernie` command a pip install puts
/// on the path:
///
///     ernie                       the default scene, Demos panel
///     ernie scene.json            a scene file (LoadSceneFile)
///     ernie scene.json --output x.png     renders it, no window
///     ernie --demo demo_lighting_illuminants   that demo running
///     ernie --list                the demos' names
///     ernie bake-texture in.png out.dds    a texture baked
///                                 (`BakeTextureMain`)
///
/// It takes the options every demo takes (`ParseDemoOptions`), so a
/// scene file renders headless or without a viewer as a demo does,
/// and `--version`. The Demos panel lists the scene (the default or
/// the file's) and every registered demo (`RegisteredDemos`). `args`
/// is the command line without the program name; returns the exit
/// status, printing any error.
int ErnieMain(std::span<const std::string> args,
              std::string_view version);

/// `ernie bake-texture input output.dds [options]` (`args` after
/// `bake-texture`): bakes an image into a mipped, block-compressed
/// DDS file (`BakeTexture`). Returns the exit status, printing any
/// error.
int BakeTextureMain(std::span<const std::string> args);

}  // namespace ernie::app
