// SPDX-License-Identifier: BSD-3-Clause
// Thin-lens depth of field: a point at distance d, with the lens of focal length f and f-number N
// focused at distance s, images as a disc of diameter c = (f / N) |d - s| / d * f / (s - f) on
// the sensor (c = f^2 / (N d) when focused at infinity). Each pixel gathers the neighbours whose
// discs cover it, each weighted by one over its disc's area: the scatter of every pixel's light
// over its disc, as a gather, so light is conserved. It ignores occlusion within the blur (an
// out-of-focus background can bleed over a sharp foreground edge). Pinhole cameras (USD fStop 0)
// are untouched. With axial colour (the lens's `chromaticFocusMm`) each band's focal length
// shifts, and with it where it images each point: stars and edges in focus for one wavelength
// blur in the others (`cocRadius`).
//
// References:
// - M. Potmesil and I. Chakravarty, "A lens and aperture camera model for synthetic image
//   generation", SIGGRAPH 1981, 297-305 (the thin-lens circle of confusion).
// - J. Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH
//   2014 course (depth of field as scatter-as-gather).
import sensor;

struct Defocus : ISpatialBandFilter {
    struct Params {
        float maxRadius;  // the widest disc radius in pixels, bounding the cost
    }

    // The blur disc's radius in pixels for what pixel `at` sees in band `band`. The sensor sits
    // where the reference wavelength's lens (focal length f) images the focus distance s, at
    // v_s = f s / (s - f) (f at infinity); band `band`'s lens, f_b = f + its axial colour's
    // shift, images a point at distance d at v = f_b d / (d - f_b) (f_b at infinity), and the
    // aperture's cone of diameter f / N meets the sensor in a disc c = (f / N) |v - v_s| / v.
    static float cocRadius(PixelContext pixel, int2 at, uint band, float maxRadius) {
        let distance = sceneDistance(pixel, at);
        let f = pixel.focalLengthMm() * 1e-3;  // metres, as scene units
        let fb = f + chromaticOf(pixel.uniforms(), band).y * 1e-3;
        let n = pixel.lensFNumber();
        let s = pixel.focusDistance();
        let aperture = f / n;
        let sensor = s > 0.0 ? f * s / max(s - f, 1e-6) : f;
        let image = isinf(distance) ? fb : fb * distance / max(distance - fb, 1e-6);
        let diameter = aperture * abs(image - sensor) / max(image, 1e-9);
        let pixels = diameter / (pixel.pixelPitchMicrons() * 1e-6);
        return min(0.5 * pixels, maxRadius);
    }

    static float band(uint band, PixelContext pixel, SpatialImage input, Params params) {
        let centre = int2(pixel.id.xy);
        if (pixel.lensFNumber() <= 0.0) {
            return input.band(pixel, centre, band);
        }
        let reach = int(ceil(max(params.maxRadius, 0.0)));
        var sum = 0.0;
        for (int y = -reach; y <= reach; ++y) {
            for (int x = -reach; x <= reach; ++x) {
                let at = centre + int2(x, y);
                // Discs under a pixel keep their light in their own pixel: the area is at least 1.
                let radius = max(cocRadius(pixel, at, band, params.maxRadius), 0.5);
                if (float(x * x + y * y) <= radius * radius) {
                    sum += input.band(pixel, at, band) /
                           max(3.14159265 * radius * radius, 1.0);
                }
            }
        }
        return sum;
    }
}
