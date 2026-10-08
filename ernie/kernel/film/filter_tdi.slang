// Time delay integration: a TDI line scanner's `stages` rows each expose the same strip of the
// scene for one line time while its charge moves along with the image, so a line collects the sum
// of the stages' signal, `stages` times one line's, while it is read (and its read noise added)
// once. When the charge's clocking and the image's motion are out of step, the image drifts by
// `drift` pixels along the track per stage, and the sum smears over the stages' drifted copies.
// Render each line with the exposure of one line time. On electrons, after `qe` and before
// `noise`, whose shot noise then follows the summed signal (SNR growing as the square root of the
// stages).
//
// References:
// - G. Lepage, J. Bogaerts and G. Meynants, "Time-Delay-Integration Architectures in CMOS Image
//   Sensors", IEEE Transactions on Electron Devices 56(11), 2009.
// - J. R. Janesick, "Scientific Charge-Coupled Devices", SPIE Press, 2001 (TDI clocking).
import sensor;

struct Tdi : ISpatialPixelFilter {
    struct Params {
        float stages;  // TDI stages summed, 1 to 128
        float drift;   // pixels the image drifts along the track per stage; 0 in step
        float axis;    // 0: the track runs down the rows (y), 1: along the columns (x)
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let stages = uint(clamp(params.stages, 1.0, 128.0));
        let along = params.axis < 0.5 ? float2(0.0, 1.0) : float2(1.0, 0.0);
        let centre = float2(pixel.id.xy);
        var sum = float4(0.0);
        for (uint k = 0; k < stages; ++k) {
            // Stage k sees the scene drifted by k drift pixels: linear between the two pixels
            // around it.
            let shift = float(k) * params.drift;
            let low = floor(shift);
            let t = shift - low;
            let a = input.values(pixel, int2(centre + along * low));
            let b = t > 0.0 ? input.values(pixel, int2(centre + along * (low + 1.0))) : a;
            sum += lerp(a, b, t);
        }
        value.electrons = sum;
    }
}
