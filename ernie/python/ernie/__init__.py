# SPDX-License-Identifier: BSD-3-Clause
"""Ernie: a physically based, spectral Vulkan ray tracer."""

from ernie import _ernie
from ernie._ernie import (
    PRIM_TYPES,
    DeviceColumn,
    DeviceContext,
    Prim,
    Renderer,
    Scene,
    SceneArray,
    VulkanDeviceInfo,
    VulkanDeviceType,
    __version__,
    distant_light_transform,
    look_at_transform,
    tokens,
    vulkan_devices,
)

# Each prim type by its name, for scene.define("/World/Rock", ernie.Sphere).
globals().update({name: name for name in PRIM_TYPES})


def cuda_allocator():
    """PyTorch's allocator for tensors Ernie shares in place with ``DeviceContext.wrap``.

    Install it before PyTorch first allocates on the GPU::

        torch.cuda.memory.change_current_allocator(ernie.cuda_allocator())

    Each tensor is then its own exportable allocation, with no caching: allocate long-lived
    simulation state this way, not temporaries.
    """
    import torch

    return torch.cuda.memory.CUDAPluggableAllocator(
        _ernie.__file__, "ernie_cuda_malloc", "ernie_cuda_free"
    )


__all__ = [
    "PRIM_TYPES",
    "DeviceColumn",
    "DeviceContext",
    "Prim",
    "Renderer",
    "Scene",
    "SceneArray",
    "VulkanDeviceInfo",
    "VulkanDeviceType",
    "__version__",
    "cuda_allocator",
    "distant_light_transform",
    "look_at_transform",
    "tokens",
    "vulkan_devices",
    *PRIM_TYPES,
]
