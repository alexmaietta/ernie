// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.texture.kernels.software;

#if ERNIE_OBJECT_TEXTURES || ERNIE_HARDWARE_TEXTURES
__exported import ernie.texture.kernels.hardware;
#if ERNIE_OBJECT_TEXTURES
typealias SampledImage = ernie::TextureObjects;
#else
typealias SampledImage = BindlessImage;
#endif
typealias SampledTexture = HardwareTexture<SampledImage>;
typealias SampledUdimSet = HardwareUdimSet<SampledImage>;
#else
typealias SampledTexture = SoftwareTexture;
typealias SampledUdimSet = UdimSet;
#endif

SampledTexture TextureFromHandle(uint64_t handle) {
#if ERNIE_OBJECT_TEXTURES || ERNIE_HARDWARE_TEXTURES
    return SampledTexture(SampledImage.FromHandle(handle));
#else
    return SoftwareTexture(Ptr<ernie::TextureDesc>(handle));
#endif
}
