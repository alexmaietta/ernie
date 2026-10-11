// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/session/outputs.h"
#include "ernie/util/tensor.h"
#include "support/test_raster.h"
#include "support/vulkan_fixture.h"

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

/// Scenes and image helpers shared by the raster delegate's GPU tests
/// (raster_*_gpu_test.cpp).
namespace ernie::testing {

class RasterDelegateTest : public VulkanTest {};

/// The image size of TerrainScene and MeshScene, and of FlatScene's
/// vars.
inline constexpr std::uint32_t kSize = 48;
inline constexpr std::uint32_t kVarSize = 32;

/// `raster`'s heights as a tensor.
Tensor HeightsOf(const TestRaster& raster);

/// A terrain lit by a low sun, with render settings for a `kSize`
/// square image.
Scene TerrainScene(const TestRaster& raster);

/// A square of side 4 in the x-y plane centred on the origin, as one
/// quad face.
void DefineQuad(Scene& scene, const Path& path, float half = 2.0f);

/// Meshes seen from straight above, lit from overhead, against a
/// black background.
Scene MeshScene();

/// MeshScene plus a row of 41 instances along x every 5 m, most of
/// them outside the view.
Scene CullingScene();

/// The RGBA texel showing world point `p` in `image`.
const std::uint8_t* TexelAt(const Scene& scene, const Tensor& image,
                            glm::vec3 p);

/// A flat grey (albedo 0.5) Hillshade terrain 100 m below a camera
/// looking straight down, lit by a relative white sun overhead
/// without shadows, rendering every var at 32 x 32.
Scene FlatScene();

/// Every var of `outputs`' product mapped to the CPU.
std::map<std::string, Tensor, std::less<>>
MapVars(RenderProductSetOutputs& outputs);

/// The value at (`row`, `col`, `channel`) of a FlatScene var.
template <typename T>
T At(const Tensor& tensor, std::size_t row, std::size_t col,
     std::size_t channel = 0) {
    const std::size_t channels =
        tensor.Shape().size() == 3
            ? static_cast<std::size_t>(tensor.Shape()[2])
            : 1;
    return static_cast<const T*>(
        tensor.Data())[(row * kVarSize + col) * channels + channel];
}

/// The pixel of FlatScene's image showing ground point (x, y).
std::pair<std::size_t, std::size_t> PixelOf(const Scene& scene,
                                            float x, float y);

/// The bytes of a CPU tensor.
std::vector<std::uint8_t> Bytes(const Tensor& image);

/// How many bytes of `a` and `b` differ by more than 8.
std::size_t BytesDiffering(const std::vector<std::uint8_t>& a,
                           const std::vector<std::uint8_t>& b);

}  // namespace ernie::testing
