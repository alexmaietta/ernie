// SPDX-License-Identifier: BSD-3-Clause
// Point-spread function: each band blurred by the lens's diffraction, the Gaussian fit to the
// Airy pattern of a circular aperture, sigma = 0.42 lambda N in the image plane, so redder bands
// spread wider; combined with a Gaussian of `blur` pixels for aberrations and the sensor's
// optical crosstalk. The weights are normalised, so the filter moves light and keeps it all:
// a point source's total stays its irradiance, spread into a disc that grows with the f-number.
// Bright stars and highlights grow into soft discs as through a real lens.
//
// References:
// - G. B. Airy, "On the Diffraction of an Object-glass with Circular Aperture", Transactions of
//   the Cambridge Philosophical Society 5:283-291, 1835.
// - B. Zhang, J. Zerubia and J.-C. Olivo-Marin, "Gaussian approximations of fluorescence
//   microscope point-spread function models", Applied Optics 46(10):1819-1829, 2007 (the
//   Gaussian fit to the Airy disc, sigma = 0.42 lambda N).
import sensor;

struct Psf : ISpatialBandFilter {
    struct Params {
        float blur;       // Gaussian sigma in pixels added to diffraction's.
        float maxRadius;  // the widest kernel in pixels, bounding the cost
    }

    static float band(uint band, PixelContext pixel, SpatialImage input, Params params) {
        let wavelengthUm = pixel.wavelength(band) * 1e-3;
        let pitch = max(pixel.pixelPitchMicrons(), 1e-3);
        let diffraction = 0.42 * wavelengthUm * pixel.diffractionFNumber() / pitch;
        let sigma = sqrt(diffraction * diffraction + params.blur * params.blur);
        let centre = int2(pixel.id.xy);
        if (sigma < 0.2) {
            return input.band(pixel, centre, band);
        }
        let radius = int(min(ceil(3.0 * sigma), max(params.maxRadius, 1.0)));
        let k = -0.5 / (sigma * sigma);
        var sum = 0.0;
        var weights = 0.0;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                let w = exp(k * float(x * x + y * y));
                sum += w * input.band(pixel, centre + int2(x, y), band);
                weights += w;
            }
        }
        return sum / weights;
    }
}
