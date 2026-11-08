// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.mapping;
import ernie.render.kernels.material.context;

using namespace ernie;

SurfaceTexels<Inputs> SampleNodeTextures<let Inputs : uint>(ShadeContext context,
                                                           Ptr<MappingRecord> mapping,
                                                           Ptr<TextureInput> inputs) {
    SurfaceTexels<Inputs> texels = SurfaceTexels<Inputs>(context.materials,
                                                        context.point.shadingNormal);
    texels.inputs = inputs;
    PlacementSettings settings;
    settings.uv.angle = radians(mapping.uvRotation);
    settings.uv.scale = mapping.uvScale.Vector();
    settings.uv.offset = mapping.uvOffset.Vector();
    settings.fine = false;
    settings.single = context.materials.minimal;
    settings.hex = mapping.hexTiling != 0;
    settings.hexContrast = mapping.hexTilingContrast;
    settings.hexRotation = mapping.hexTilingRotation / 180;
    settings.frameNormal = context.point.shadingNormal;
    ForEachPlacement(context.materials.hit, context.point, *mapping, settings, texels);
    return texels;
}
