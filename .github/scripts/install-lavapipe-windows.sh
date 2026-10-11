#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# Mesa's lavapipe, the software Vulkan driver, on a Windows runner:
# mesa-dist-win's build of it and LunarG's Vulkan loader, each checked
# against its digest. The driver is registered where the loader looks
# for drivers, as the loader ignores VK_DRIVER_FILES in an elevated
# process, as the runner's are. The loader's directory goes on PATH
# (GITHUB_PATH) for later steps.
set -euo pipefail

mesa_version=26.2.3
mesa_sha256=3f3613adb43cfd0f2e665ce2400b130c275f0b3317cb3a05566320a3a67589ed
loader_version=1.4.363.0
loader_sha256=a25a927aa8b9f0371048f1861cf88ac3b9bc9b1fb332c42d897c8ab32695769a

vulkan="$RUNNER_TEMP/vulkan"
mkdir -p "$vulkan"
cd "$vulkan"
curl -fsSL -o mesa.7z "https://github.com/pal1000/mesa-dist-win/releases/download/$mesa_version/mesa3d-$mesa_version-release-msvc.7z"
curl -fsSL -o loader.zip "https://sdk.lunarg.com/sdk/download/$loader_version/windows/VulkanRT-X64-$loader_version-Components.zip"
echo "$mesa_sha256  mesa.7z" | sha256sum -c -
echo "$loader_sha256  loader.zip" | sha256sum -c -
7z x -y -omesa mesa.7z > /dev/null
7z x -y -oloader loader.zip > /dev/null
cp loader/*/x64/vulkan-1.dll mesa/x64/
cygpath -w "$vulkan/mesa/x64" >> "$GITHUB_PATH"
reg add 'HKLM\SOFTWARE\Khronos\Vulkan\Drivers' //f //t REG_DWORD //d 0 \
    //v "$(cygpath -w "$vulkan/mesa/x64/lvp_icd.x86_64.json")"
