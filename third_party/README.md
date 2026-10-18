# Third-party dependencies

Small dependencies are vendored here, byte for byte as their upstream releases, so a clone builds them without a network. Each folder's README says which release it is and how to update it. The `.cpp` files at this level are Ernie's, compiling header-only libraries once.

| Folder | Upstream | Version | Licence |
|---|---|---|---|
| `basisu` | [BinomialLLC/basis_universal](https://github.com/BinomialLLC/basis_universal) (transcoder) | `v2_50` | Apache-2.0 |
| `bc7enc` | [richgel999/bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo) | `b9438627` | MIT or public domain |
| `dlpack` | [dmlc/dlpack](https://github.com/dmlc/dlpack) | `v1.3` | Apache-2.0 |
| `glm` | [g-truc/glm](https://github.com/g-truc/glm) | `1.0.3` | MIT |
| `json` | [nlohmann/json](https://github.com/nlohmann/json) | `v3.12.0` | MIT |
| `nanovdb` | [AcademySoftwareFoundation/openvdb](https://github.com/AcademySoftwareFoundation/openvdb) (`PNanoVDB.h`) | see its header | Apache-2.0 |
| `stb` | [nothings/stb](https://github.com/nothings/stb) | `2c980bb5` | MIT or public domain |
| `vma` | [GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | `v3.4.0` | MIT |
| `volk` | [zeux/volk](https://github.com/zeux/volk) | `vulkan-sdk-1.4.363.0` | MIT |
| `vulkan_headers` | [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | `vulkan-sdk-1.4.363.0` | Apache-2.0 or MIT |

The large ones are fetched on first configure into `.deps/` (`ERNIE_DEPS_DIR`), pinned in `cmake/third_party/`. After that, `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` configures offline.

| Dependency | Pinned in | Needed for |
|---|---|---|
| Slang (prebuilt) | `slang.cmake` | always |
| OpenEXR and Imath | `openexr.cmake` | always |
| GoogleTest | `googletest.cmake` | `ERNIE_BUILD_TESTS` |
| nanobind | `nanobind.cmake` | `ERNIE_BUILD_PYTHON`, when the target Python has none installed |
| Tracy | `tracy.cmake` | `ERNIE_TRACY` |
