// Measured quantum efficiency: electrons per channel from the bands, through red, green and blue
// QE curves the SensorFilter prim gives (each 48 samples, 360 to 830 nm every 10 nm; float arrays
// of any length, resampled), in place of the camera's own sensor model: electrons_c = sum over
// bands of radiance times the camera equation's photons (`photonsPerRadiance`) times QE_c(lambda),
// plus dark current. For matching a real camera's published or measured curves. An integrator into
// electrons, in place of `qe`.
//
// Reference: EMVA Standard 1288 Release 4.0 Linear, 2021 (quantum efficiency and the photon
// transfer of a linear sensor).
import sensor;

struct MeasuredQe : ICustomIntegrator {
    struct Params {
        float red[48];
        float green[48];
        float blue[48];
    }

    static void accumulate(inout SensorPixel value, uint band, float radiance, PixelContext pixel,
                           Params params) {
        let t = (pixel.wavelength(band) - 360.0) / (830.0 - 360.0);
        let photons = radiance * pixel.photonsPerRadiance(band);
        value.electrons += photons * float4(max(curveAt(params.red, t), 0.0),
                                            max(curveAt(params.green, t), 0.0),
                                            max(curveAt(params.blue, t), 0.0), 0.0);
    }

    static void finish(inout SensorPixel value, PixelContext pixel, Params params) {
        var e = value.electrons + pixel.darkElectrons();
        e.w = 0.0;
        value.electrons = e;
    }
}
