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

    uint Definition(uint material) {
        return material == kNoPrim ? kDefinitionDiffuse : hit.tables->materials[material].definition;
    }

    T* Params<T>(uint material) {
        return (T*)(hit.tables->materialWords + hit.tables->materials[material].words);
    }
}
