// Updates of a heightfield written on the GPU, over the rectangle of cells that changed: the
// central-difference gradients and their mips that terrain shading samples, and each LOD leaf's
// min and max height (`LodQuadtree`), the same as `TerrainDrawable::SetHeightfield` computes on the
// host.
//
// Bindings: 0 the heights, row-major floats; 1 the gradient mip written (rg16f); 2 the mip below
// it (mips above 0); 3 per LOD leaf, (min, max), min above max for a leaf without valid cells.

struct UpdateConstants {
    // The rectangle at this level: first column and row, end column and row (exclusive).
    uint4 rect;
    // Raster width and height at this level, the LOD leaf's cells, leaves per side.
    uint4 size;
    // x: 0.5 over the cell size, y: the nodata value, z: 1 with a nodata value.
    float4 values;
}

[[vk::binding(0, 0)]] StructuredBuffer<float> heights;
[[vk::binding(1, 0)]] [[vk::image_format("rg16f")]] RWTexture2D<float2> gradients;
[[vk::binding(2, 0)]] [[vk::image_format("rg16f")]] RWTexture2D<float2> below;
[[vk::binding(3, 0)]] RWStructuredBuffer<float2> leaves;
[[vk::push_constant]] ConstantBuffer<UpdateConstants> constants;

float heightAt(uint col, uint row) {
    return heights[row * constants.size.x + col];
}

// Level 0: central differences, one-sided at the edges.
[shader("compute")]
[numthreads(8, 8, 1)]
void gradientsMain(uint3 id : SV_DispatchThreadID) {
    let col = constants.rect.x + id.x;
    let row = constants.rect.y + id.y;
    if (col >= constants.rect.z || row >= constants.rect.w) {
        return;
    }
    let w = constants.size.x;
    let h = constants.size.y;
    let west = col > 0 ? col - 1 : col;
    let east = col + 1 < w ? col + 1 : col;
    let north = row > 0 ? row - 1 : row;
    let south = row + 1 < h ? row + 1 : row;
    let scale = constants.values.x;
    gradients[uint2(col, row)] = float2((heightAt(east, row) - heightAt(west, row)) * scale,
                                        (heightAt(col, north) - heightAt(col, south)) * scale);
}

// Mips above 0: the mean of the 2 x 2 texels below, clamped at the edges of the level below
// (`size` its width and height).
[shader("compute")]
[numthreads(8, 8, 1)]
void gradientMipMain(uint3 id : SV_DispatchThreadID) {
    let col = constants.rect.x + id.x;
    let row = constants.rect.y + id.y;
    if (col >= constants.rect.z || row >= constants.rect.w) {
        return;
    }
    let w = constants.size.x;
    let h = constants.size.y;
    let c0 = min(2 * col, w - 1);
    let c1 = min(2 * col + 1, w - 1);
    let r0 = min(2 * row, h - 1);
    let r1 = min(2 * row + 1, h - 1);
    gradients[uint2(col, row)] = 0.25 * (below[uint2(c0, r0)] + below[uint2(c1, r0)] +
                                         below[uint2(c0, r1)] + below[uint2(c1, r1)]);
}

bool isValid(float value) {
    return isfinite(value) && !(constants.values.z > 0.5 && value == constants.values.y);
}

// Per LOD leaf in the rectangle (in leaves), the range of the valid heights at its vertices.
[shader("compute")]
[numthreads(8, 8, 1)]
void leafRangeMain(uint3 id : SV_DispatchThreadID) {
    let x = constants.rect.x + id.x;
    let y = constants.rect.y + id.y;
    if (x >= constants.rect.z || y >= constants.rect.w) {
        return;
    }
    let w = constants.size.x;
    let h = constants.size.y;
    let cells = constants.size.z;
    var range = float2(asfloat(0x7f800000u), asfloat(0xff800000u));
    let row0 = y * cells;
    let col0 = x * cells;
    if (row0 < h - 1 && col0 < w - 1) {
        let row1 = min(row0 + cells, h - 1);
        let col1 = min(col0 + cells, w - 1);
        for (uint row = row0; row <= row1; ++row) {
            for (uint col = col0; col <= col1; ++col) {
                let value = heightAt(col, row);
                if (isValid(value)) {
                    range = float2(min(range.x, value), max(range.y, value));
                }
            }
        }
    }
    leaves[y * constants.size.w + x] = range;
}
