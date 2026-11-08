// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.surface.element;

#include "ernie/scene/records/primvars.h"

using namespace ernie;

struct PrimvarRef {
    uint64_t table;
    PrimvarEntry entry;

    bool Found() {
        return table != 0;
    }
}

PrimvarRef FindPrimvar(RenderTables* tables, uint slot, uint name) {
    return FindPrimvarIn(tables->surfaces[slot].primvars, name);
}

PrimvarRef FindPrimvarIn(ArrayRef table, uint name) {
    PrimvarRef found;
    found.table = 0;
    if (table.count == 0) {
        return found;
    }
    const Ptr<PrimvarEntry> entries = Ptr<PrimvarEntry>(table.address + sizeof(uint));
    uint low = 0;
    uint high = Ptr<uint>(table.address)[0];
    while (low < high) {
        const uint middle = (low + high) / 2;
        const PrimvarEntry entry = entries[middle];
        if (entry.name == name) {
            found.table = table.address;
            found.entry = entry;
            return found;
        }
        if (entry.name < name) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return found;
}

float4 PrimvarValue(PrimvarRef primvar, uint index) {
    const Ptr<uint> words = Ptr<uint>(primvar.table);
    const PrimvarType type = PrimvarType(primvar.entry.type);
    const uint components = PrimvarComponents(type);
    const uint at = primvar.entry.values + index * components * primvar.entry.elementSize;
    float4 value = float4(0);
    for (uint c = 0; c < components; ++c) {
        const uint word = words[at + c];
        value[c] = PrimvarIsInt(type) ? float(int(word)) : asfloat(word);
    }
    return value;
}

bool ReadPrimvar(PrimvarRef primvar, SurfaceElement element, out float4 value) {
    value = float4(0);
    if (!primvar.Found()) {
        return false;
    }
    const Interpolation interpolation = Interpolation(primvar.entry.interpolation);
    if (!element.indexed && interpolation != Interpolation::Constant) {
        return false;
    }
    uint3 indices = InterpolationIndices(element, interpolation);
    if (primvar.entry.indexCount > 0) {
        if (any(indices >= primvar.entry.indexCount)) {
            return false;
        }
        const Ptr<uint> words = Ptr<uint>(primvar.table + primvar.entry.indices * sizeof(uint));
        indices = uint3(words[indices.x], words[indices.y], words[indices.z]);
    }
    if (any(indices >= primvar.entry.valueCount)) {
        return false;
    }
    if (PrimvarIsInt(PrimvarType(primvar.entry.type))) {
        const float3 w = element.weights;
        const uint nearest =
            w.x >= w.y && w.x >= w.z ? indices.x : (w.y >= w.z ? indices.y : indices.z);
        value = PrimvarValue(primvar, nearest);
        return true;
    }
    value = PrimvarValue(primvar, indices.x) * element.weights.x +
            PrimvarValue(primvar, indices.y) * element.weights.y +
            PrimvarValue(primvar, indices.z) * element.weights.z;
    return true;
}

bool ReadInstancePrimvar(RenderTables* tables, uint slot, uint name, out float4 value) {
    value = float4(0);
    const PrimvarRef primvar = FindPrimvarIn(tables->surfaces[slot].instancerPrimvars, name);
    if (!primvar.Found()) {
        return false;
    }
    const SlotHit drawn = tables->instances->hits[slot];
    uint index = Interpolation(primvar.entry.interpolation) == Interpolation::Constant
                     ? 0
                     : drawn.instance;
    if (primvar.entry.indexCount > 0) {
        if (index >= primvar.entry.indexCount) {
            return false;
        }
        index = Ptr<uint>(primvar.table + primvar.entry.indices * sizeof(uint))[index];
    }
    if (index >= primvar.entry.valueCount) {
        return false;
    }
    value = PrimvarValue(primvar, index);
    return true;
}

float3 SurfaceDisplayColor(RenderTables* tables, uint slot, SurfaceElement element) {
    const uint name = tables->instances->displayColorName;
    float4 tint;
    if (ReadInstancePrimvar(tables, slot, name, tint)) {
        return tint.xyz;
    }
    if (ReadPrimvar(FindPrimvar(tables, slot, name), element, tint)) {
        return tint.xyz;
    }
    return tables->surfaces[slot].displayColor.Vector();
}
