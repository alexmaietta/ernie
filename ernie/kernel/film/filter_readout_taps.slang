// Readout taps: a sensor read out through `taps` interleaved output amplifiers and ADCs, pixel x
// on tap x mod taps (even and odd pixels for two), each with its own gain and offset, so the image
// carries a fine column pattern of period `taps`, as LROC's narrow-angle cameras show between
// their even and odd pixels. On digital numbers, after `adc`; the result stays within the ADC's
// range.
//
// Reference: D. C. Humm et al., "Flight Calibration of the LROC Narrow Angle Camera", Space
// Science Reviews 200:431-473, 2016 (even and odd pixel offsets).
import sensor;

struct ReadoutTaps : IPixelFilter {
    struct Params {
        float taps;  // interleaved taps, 1 to 4
        float offset0;  // each tap's offset, DN
        float offset1;
        float offset2;
        float offset3;
        float gain0;  // each tap's gain
        float gain1;
        float gain2;
        float gain3;
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let taps = uint(clamp(params.taps, 1.0, 4.0));
        let tap = pixel.id.x % taps;
        let offsets = float4(params.offset0, params.offset1, params.offset2, params.offset3);
        let gains = float4(params.gain0, params.gain1, params.gain2, params.gain3);
        let largest = pixel.uniforms().adc.w;
        let dn = float4(value.digital) * gains[tap] + offsets[tap];
        value.digital = uint4(clamp(floor(dn + 0.5), float4(0.0), float4(largest)));
    }
}
