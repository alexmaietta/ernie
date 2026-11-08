// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.hit;

#include "ernie/scene/kernels/textures.h"

using namespace ernie;

struct MaterialQuery {
    HitQuery hit;
    float lodBias;
    uint maxAnisotropy;
    bool minimal;

    __init(HitQuery hit, float lodBias, uint maxAnisotropy) {
        this.hit = hit;
        this.lodBias = lodBias;
        this.maxAnisotropy = maxAnisotropy;
        minimal = false;
    }

    static MaterialQuery Load(HitQuery hit, uint product) {
        RenderProductData* products = &hit.scene->schemas.renderProduct;
        const uint row = PrimRow(hit.scene, product, kSlotRenderProduct);
        return MaterialQuery(hit, products.textureLodBias[row],
                             uint(max(products.maxAnisotropy[row], 1)));
    }

    uint BoundMaterial(uint prim) {
        return PrimImageable(hit.tables, prim).material;
    }

    SchemaId Schema(uint material) {
        return material == kNoPrim ? SchemaId::None : PrimSchema(hit.tables, material);
    }

    uint Textures(uint material) {
        return material == kNoPrim ? kNoTextures : hit.tables->textures[material].textures;
    }

    Ptr<uint64_t> Bindings(uint textures, uint first) {
        return textures == kNoTextures ? Ptr<uint64_t>(0)
                                       : hit.scene->textureBindings + (textures + first);
    }

    uint64_t Texture(uint textures, uint input) {
        return textures == kNoTextures ? 0 : hit.scene->textureBindings[textures + input];
    }
}
