// Frame-transfer smear: a frame-transfer CCD without a shutter keeps collecting light while it
// shifts its image into the storage area, each pixel passing under every row of its column for
// `transferTime` over the rows. A pixel gains the column's mean signal times `transferTime` over
// the exposure time: bright sources streak along their columns (rows with `axis` 1). On
// electrons, after `qe` and before `noise`. Each pixel reads its whole column.
//
// Reference: J. R. Janesick, "Scientific Charge-Coupled Devices", SPIE Press, 2001 (frame-transfer
// architecture and image smear).
import sensor;

struct Smear : ISpatialPixelFilter {
    struct Params {
        float transferTime;  // seconds to shift the whole frame into storage
        float axis;          // 0: the charge moves along columns, 1: along rows
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let size = input.size(pixel);
        let centre = int2(pixel.id.xy);
        let along = params.axis < 0.5 ? int2(0, 1) : int2(1, 0);
        let count = params.axis < 0.5 ? size.y : size.x;
        let start = params.axis < 0.5 ? int2(centre.x, 0) : int2(0, centre.y);
        var sum = float4(0.0);
        for (int i = 0; i < count; ++i) {
            sum += input.values(pixel, start + along * i);
        }
        let share = params.transferTime / max(pixel.exposureTime(), 1e-12);
        value.electrons = input.values(pixel, centre) + share * sum / float(count);
    }
}
