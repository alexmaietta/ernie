# Third-party code

## Vendored

Unmodified copies of small single-header libraries used by Ernie. Each directory keeps
the upstream license; the `.cpp` file beside the headers is ours and only compiles the
implementation. Build glue is in `cmake/third_party/<name>.cmake`.

| Library | Version | Files | License | Used for |
| --- | --- | --- | --- | --- |
| [stb](https://github.com/nothings/stb) | commit `2c980bb59875b0d32144a71867fbdebb2f77cd20` | `stb_image.h`, `stb_image_write.h` | MIT or public domain | Reading PNG (8/16-bit) and JPEG textures |
| [DLPack](https://github.com/dmlc/dlpack) | `v1.3` | `include/dlpack/dlpack.h` | Apache-2.0 | Tensor exchange with NumPy, PyTorch, JAX and Warp |

To update one, replace its headers with the files from the new upstream version, update the
version here, and rerun the image tests (`ernie_tests_unit` or, inside TerrainGen, `terraingen_tests_ernie_unit`, with `--gtest_filter='Image*:*Png*:*Exr*'`).

## Fetched

Everything else is fetched at configure time (`cmake/third_party/<name>.cmake`, which pins the version as `ERNIE_DEP_VERSION_<name>`) and built into Ernie's binaries. The packages, the wheel and `cmake --install` ship each library's license under `third_party/<name>/` (`cmake/ErnieLicenses.cmake`); add a dependency there as well as here.

| Library | Version | License | Used for |
| --- | --- | --- | --- |
| [Slang](https://github.com/shader-slang/slang) | `2026.19` (prebuilt, shared) | Apache-2.0 WITH LLVM-exception | Compiling the shaders to SPIR-V at run time |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | `vulkan-sdk-1.4.363.0` | Apache-2.0 OR MIT | Vulkan API |
| [volk](https://github.com/zeux/volk) | `vulkan-sdk-1.4.363.0` | MIT | Loading Vulkan at run time |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | `v3.4.0` | MIT | GPU memory |
| [GLM](https://github.com/g-truc/glm) | `1.0.3` | MIT | Vector and matrix math |
| [OpenEXR](https://github.com/AcademySoftwareFoundation/openexr) | `v3.5.2` | BSD-3-Clause | Reading EXR textures |
| [Imath](https://github.com/AcademySoftwareFoundation/Imath) | with OpenEXR | BSD-3-Clause | OpenEXR's math |
| [libdeflate](https://github.com/ebiggers/libdeflate), [zstd](https://github.com/facebook/zstd), [OpenJPH](https://github.com/aous72/OpenJPH) | vendored by OpenEXR | MIT, BSD-3-Clause, BSD-2-Clause | OpenEXR's compression |
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | `v1.3` | MIT | Simplified levels of detail |
| [nlohmann/json](https://github.com/nlohmann/json) | `v3.12.0` | MIT | Reading JSON |
| [Dear ImGui](https://github.com/ocornut/imgui) | `v1.92.9b-docking` | MIT | The viewer's UI (app only) |
| [ImPlot](https://github.com/epezent/implot) | `v1.0` | MIT | The viewer's plots (app only) |
| [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) | `1.10` | MIT | The viewer's gizmos (app only) |
| [GLFW](https://github.com/glfw/glfw) | `3.4` | Zlib | Windows and input (app only; a system GLFW 3.4 when found) |
| [Inter](https://github.com/rsms/inter) | `v4.1` | OFL-1.1 | The viewer's typeface (app only) |
| [Lucide](https://github.com/lucide-icons/lucide) | `1.52.0` | ISC | The viewer's icons (app only) |
| [MoltenVK](https://github.com/KhronosGroup/MoltenVK) | `v1.4.2` | Apache-2.0 | Vulkan on macOS (macOS only) |
| [nanobind](https://github.com/wjakob/nanobind) | `v3.1.0` when not installed | BSD-3-Clause | The Python module (Python only) |
| [Tracy](https://github.com/wolfpld/tracy) | `v0.14.1` | BSD-3-Clause | Profiling (`ERNIE_TRACY` builds only) |
| [GoogleTest](https://github.com/google/googletest) | `v1.18.0` | BSD-3-Clause | Tests only; not distributed |
