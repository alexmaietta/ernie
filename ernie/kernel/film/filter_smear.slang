// SPDX-License-Identifier: BSD-3-Clause
// Frame-transfer smear: a frame-transfer CCD without a shutter keeps collecting light while its
// charge moves, so bright sources streak along their columns (rows with `axis` 1). Two shifts
// smear, both towards the storage area (the bottom, or the right with `axis` 1; the top or left
// with `towards` 1), each moving a whole frame in its time, `1 / rows` of it per row:
// - readout, after the exposure (`transferTime`): each pixel's charge passes under the rows
//   between it and the storage area, gaining their signal;
// - reset, the flush before the exposure (`resetTime`): the empty well that ends at a pixel comes
//   in from the far edge, passing under the rows before it.
// A pixel at row r of R gains sum over the rows i past it of S_i transferTime / (R t), and over
// the rows before it of S_i resetTime / (R t), S_i a row's electrons and t the exposure time:
// readout smear on the storage side of a source, reset smear on the other. Equal times give the
// column's sum but its own pixel, evenly. On electrons, after `qe` and before `noise`. Each pixel
// reads its whole column.
//
// References:
// - J. R. Janesick, "Scientific Charge-Coupled Devices", SPIE Press, 2001 (frame-transfer
//   architecture and image smear).
// - STAR-Dundee, "PANGU v7 Datasheet", 2024 (readout and reset smear in its camera model).
import sensor;

struct Smear : ISpatialPixelFilter {
    struct Params {
        float transferTime;  // seconds to shift the whole frame into storage after the exposure
        float axis;          // 0: the charge moves along columns, 1: along rows
        float resetTime;     // seconds to flush the whole frame before the exposure
        float towards;       // 0: storage at the bottom (right), 1: at the top (left)
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let size = input.size(pixel);
        let centre = int2(pixel.id.xy);
        let along = params.axis < 0.5 ? int2(0, 1) : int2(1, 0);
        let count = params.axis < 0.5 ? size.y : size.x;
        let start = params.axis < 0.5 ? int2(centre.x, 0) : int2(0, centre.y);
        let own = params.axis < 0.5 ? centre.y : centre.x;
        var before = float4(0.0);  // rows with a lower index
        var after = float4(0.0);
        for (int i = 0; i < count; ++i) {
            let signal = input.values(pixel, start + along * i);
            if (i < own) {
                before += signal;
            } else if (i > own) {
                after += signal;
            }
        }
        let storageLow = params.towards > 0.5;
        let readout = storageLow ? before : after;
        let reset = storageLow ? after : before;
        let perRow = 1.0 / (float(count) * max(pixel.exposureTime(), 1e-12));
        value.electrons = input.values(pixel, centre) +
                          perRow * (params.transferTime * readout + params.resetTime * reset);
    }
}
