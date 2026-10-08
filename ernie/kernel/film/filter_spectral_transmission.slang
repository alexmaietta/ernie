// Spectral transmission: each band's radiance times a measured transmission curve, `curve`, 48
// samples from 360 to 830 nm (every 10 nm; all 1 by default), such as an IR-cut, band-pass or
// neutral density filter, a lens's or window's transmission, or a filter wheel position; a
// SensorFilter prim sets it as a float array of any length, resampled over that range. On the
// spectrum, before `qe` and `cie`.
import sensor;

struct SpectralTransmission : IBandFilter {
    struct Params {
        float curve[48];
    }

    static float apply(float radiance, uint band, PixelContext pixel, Params params) {
        let t = (pixel.wavelength(band) - 360.0) / (830.0 - 360.0);
        return radiance * max(curveAt(params.curve, t), 0.0);
    }
}
