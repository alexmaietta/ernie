// Thermal dark current: electrons the sensor generates without light over the exposure, at a
// rate doubling every `doubling` degrees above `referenceTemperature` (where it is
// `referenceRate`), each pixel's own rate off the mean by a fixed relative `nonuniformity` (dark
// signal non-uniformity). Use it in place of the sensor's constant dark current
// (`ernie:sensor:darkCurrent` 0) to model a camera warming or cooling. On electrons, after `qe`
// and before `noise`, which adds its shot noise.
//
// References:
// - EMVA Standard 1288 Release 4.0 Linear, 2021 (dark current and its doubling temperature).
// - J. R. Janesick, "Photon Transfer", SPIE Press, 2007 (dark current's temperature dependence).
import sensor;

struct DarkCurrent : IPixelFilter {
    struct Params {
        float temperature;           // the sensor's, degrees Celsius
        float referenceTemperature;  // degrees Celsius
        float referenceRate;         // mean electrons per second there
        float doubling;              // degrees Celsius per doubling of the rate
        float nonuniformity;         // standard deviation of each pixel's rate, relative
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let rate = params.referenceRate *
                   exp2((params.temperature - params.referenceTemperature) /
                        max(params.doubling, 1e-3));
        let own = max(1.0 + params.nonuniformity * pixel.fixedNormal(pixel.id.xy, 0x63u), 0.0);
        value.electrons += float4(rate * own * pixel.exposureTime());
    }
}
