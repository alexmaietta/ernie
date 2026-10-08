// Square-root companding: digital numbers of `inputBits` mapped to `outputBits` as the square
// root of their fraction of full scale, out = round((2^outputBits - 1) sqrt(in / (2^inputBits -
// 1))), as spacecraft cameras (LROC's NAC and WAC among them) compress 12-bit samples to 8 bits
// for downlink: coarse steps where shot noise is large, fine ones in the shadows. After `adc`.
//
// Reference: M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC) Instrument
// Overview", Space Science Reviews 150:81-124, 2010 (12- to 8-bit companding).
import sensor;

struct Compand : IPixelFilter {
    struct Params {
        float inputBits;
        float outputBits;
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let inputMax = exp2(params.inputBits) - 1.0;
        let outputMax = exp2(params.outputBits) - 1.0;
        let fraction = saturate(float4(value.digital) / inputMax);
        value.digital = uint4(floor(outputMax * sqrt(fraction) + 0.5));
    }
}
