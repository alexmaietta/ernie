// Demosaicing: the full RGB digital numbers of each pixel interpolated from a Bayer mosaic of the
// same `pattern` (`bayer`), by `method` 0 bilinear interpolation of each channel's neighbours, or
// 1 Malvar, He and Cutler's gradient-corrected linear filters, which add the local gradient of
// the pixel's own channel and avoid most of bilinear's colour fringes at edges. On digital
// numbers, after `adc`; the result stays within the ADC's range. Output 0, `mosaic`, keeps the
// raw mosaic it read.
//
// Reference: H. S. Malvar, L. He and R. Cutler, "High-quality linear interpolation for demosaicing
// of Bayer-patterned color images", IEEE ICASSP 2004, vol. 3, 485-488.
import sensor;

// The channel (0 red, 1 green, 2 blue) of the filter over pixel `at` in `pattern` (as `bayer`).
uint mosaicChannel(int2 at, uint pattern) {
    let x = uint(at.x) & 1u;
    let y = uint(at.y) & 1u;
    let tiles = uint4[4](uint4(0, 1, 1, 2), uint4(2, 1, 1, 0), uint4(1, 0, 2, 1), uint4(1, 2, 0, 1));
    return tiles[min(pattern, 3u)][y * 2u + x];
}

struct Demosaic : ISpatialPixelFilter {
    struct Params {
        float pattern;  // as `bayer`: 0 RGGB, 1 BGGR, 2 GRBG, 3 GBRG
        float method;   // 0 bilinear, 1 Malvar-He-Cutler
    }

    // The mosaic's value at `at`: the one channel its filter passed.
    static float raw(PixelContext pixel, SpatialImage input, int2 at, uint pattern) {
        return input.values(pixel, at)[mosaicChannel(at, pattern)];
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        if (pixel.uniforms().size.w != 3) {
            return;
        }
        let pattern = uint(params.pattern);
        let p = int2(pixel.id.xy);
        let own = mosaicChannel(p, pattern);
        // The 5 x 5 neighbourhood of the mosaic.
        float m[5][5];
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                m[dy + 2][dx + 2] = raw(pixel, input, p + int2(dx, dy), pattern);
            }
        }
        let c = m[2][2];
        let cross = m[1][2] + m[3][2] + m[2][1] + m[2][3];  // up, down, left, right
        let diagonal = m[1][1] + m[1][3] + m[3][1] + m[3][3];
        let horizontal = m[2][1] + m[2][3];
        let vertical = m[1][2] + m[3][2];
        float3 rgb;
        rgb[own] = c;
        if (params.method < 0.5) {
            // Bilinear: greens from the cross; at green pixels the other two from their row and
            // column pairs; at red or blue pixels the opposite colour from the diagonals.
            if (own == 1) {
                let rowChannel = mosaicChannel(p + int2(1, 0), pattern);
                let columnChannel = mosaicChannel(p + int2(0, 1), pattern);
                rgb[rowChannel] = 0.5 * horizontal;
                rgb[columnChannel] = 0.5 * vertical;
            } else {
                rgb[1] = 0.25 * cross;
                rgb[2 - own] = 0.25 * diagonal;
            }
        } else {
            // Malvar-He-Cutler, the paper's 5 x 5 kernels (scaled by 1/8).
            let farCross = m[0][2] + m[4][2] + m[2][0] + m[2][4];
            let farRow = m[2][0] + m[2][4];
            let farColumn = m[0][2] + m[4][2];
            if (own == 1) {
                // Green pixel: the colour of its row and of its column.
                let rowChannel = mosaicChannel(p + int2(1, 0), pattern);
                let columnChannel = mosaicChannel(p + int2(0, 1), pattern);
                let alongRow = (5.0 * c + 4.0 * horizontal - diagonal - farRow +
                                0.5 * farColumn) / 8.0;
                let alongColumn = (5.0 * c + 4.0 * vertical - diagonal - farColumn +
                                   0.5 * farRow) / 8.0;
                rgb[rowChannel] = alongRow;
                rgb[columnChannel] = alongColumn;
            } else {
                rgb[1] = (4.0 * c + 2.0 * cross - farCross) / 8.0;
                rgb[2 - own] = (6.0 * c + 2.0 * diagonal - 1.5 * farCross) / 8.0;
            }
        }
        let largest = sensorUniformsMaxDn(pixel);
        value.digital = uint4(uint3(clamp(rgb + 0.5, float3(0.0), float3(largest))), 0u);
        // Output 0, `mosaic`: the raw mosaic, the pixel's own channel alone.
        var mosaic = float4(0.0);
        mosaic[own] = c;
        writeOutput(pixel, 0, mosaic);
    }

    static float sensorUniformsMaxDn(PixelContext pixel) {
        return pixel.uniforms().adc.w;
    }
}
