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

    MaterialRecord Record(uint material) {
        if (material == kNoPrim) {
            MaterialRecord diffuse;
            diffuse.definition = kDefinitionDiffuse;
            diffuse.words = 0;
            diffuse.previousDefinition = kDefinitionPlaceholder;
            diffuse.previousWords = 0;
            return diffuse;
        }
        return hit.tables->materials[material];
    }

    T* Params<T>(uint words) {
        return (T*)(hit.tables->materialWords + words);
    }
}
