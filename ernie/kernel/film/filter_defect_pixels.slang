// Defect pixels: a fixed fraction of the sensor's pixels dead (no response to light) and another
// hot (a dark current far above the rest, each its own rate between half and one and a half
// times `hotRate`), the same pixels every frame from the sensor's seed. On electrons, after `qe`
// and before `noise`, which adds the hot pixels' shot noise and every pixel's read noise.
//
// Reference: EMVA Standard 1288 Release 4.0 Linear, 2021 (defect pixel characterisation: dead,
// hot and stuck pixels).
import sensor;

struct DefectPixels : IPixelFilter {
    struct Params {
        float dead;     // fraction of pixels with no response
        float hot;      // fraction of pixels with a high dark current
        float hotRate;  // a hot pixel's typical dark current, electrons per second
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let which = pixel.fixedRandom(0x61u);
        if (which <= params.dead) {
            value.electrons = float4(0.0);
        } else if (which <= params.dead + params.hot) {
            let rate = params.hotRate * (0.5 + pixel.fixedRandom(0x62u));
            value.electrons += float4(rate * pixel.exposureTime());
        }
    }
}
