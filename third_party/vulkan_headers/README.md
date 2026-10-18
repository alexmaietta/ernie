The Vulkan C headers, vendored from [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) tag `vulkan-sdk-1.4.363.0` (Apache-2.0 or MIT, see `LICENSE.md` and `LICENSES/`). Only the C headers in `include/vulkan/` and `include/vk_video/` are kept: the C++ bindings (`*.hpp`, `*.cppm`), the registry and the tests are not used.

To update, replace `include/vulkan/*.h`, `include/vk_video/`, `LICENSE.md` and `LICENSES/` from the new tag's archive, and change the tag here. Keep it on the same SDK tag as `../volk`.
