// Blooming: charge beyond a pixel's full well spills into its neighbours along its column, half
// up and half down, filling each to its full well before moving on, as in a CCD without
// anti-blooming: saturated stars and highlights grow vertical streaks whose length is their excess
// over the full well. `antiBlooming` is the fraction of the excess an overflow drain removes
// instead (most CMOS sensors drain nearly all). Each pixel gathers what reaches it from the
// pixels within `maxLength` above and below, from the excess before spilling (the neighbours'
// own charge is not counted against the spill: a fair approximation where streaks cross dark
// sky, as they usually do). On electrons, after `qe` (and before `noise`, as the spill happens
// while the sensor exposes); the result is clipped at the full well.
//
// Output 0, `overflow`: the electrons each pixel had beyond the full well, before spilling.
//
// Reference: J. R. Janesick, "Scientific Charge-Coupled Devices", SPIE Press, 2001, section 3.5
// (full well, blooming and anti-blooming drains).
import sensor;

struct Blooming : ISpatialPixelFilter {
    struct Params {
        float antiBlooming;  // fraction of the excess drained, 0 to 1
        float maxLength;     // the longest spill in pixels each way, bounding the cost
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let well = fullWell(pixel);
        let centre = int2(pixel.id.xy);
        let own = input.values(pixel, centre);
        let keep = 1.0 - saturate(params.antiBlooming);
        var e = min(own, float4(well));
        let reach = int(max(params.maxLength, 0.0));
        for (int k = 1; k <= reach; ++k) {
            for (int side = -1; side <= 1; side += 2) {
                let source = input.values(pixel, centre + int2(0, side * k));
                // Half the source's excess goes each way, filling k - 1 pixels before this one.
                let spill = 0.5 * keep * max(source - well, float4(0.0));
                e += clamp(spill - float(k - 1) * well, float4(0.0), float4(well));
            }
        }
        value.electrons = min(e, float4(well));
        writeOutput(pixel, 0, max(own - well, float4(0.0)));
    }
}
