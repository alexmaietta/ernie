// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.hit;


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
        ProductRecord* record = &hit.tables->products[PrimRow(hit.tables, product, kSlotRenderProduct)];
        return MaterialQuery(hit, record.textureLodBias, record.maxAnisotropy);
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
                                       : hit.tables->textureBindings + (textures + first);
    }

    uint64_t Texture(uint textures, uint input) {
        return textures == kNoTextures ? 0 : hit.tables->textureBindings[textures + input];
    }
}
