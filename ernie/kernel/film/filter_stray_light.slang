// SPDX-License-Identifier: BSD-3-Clause
// Stray light: sunlight entering the aperture from outside the field, scattered off the lens's
// surfaces (roughness, contamination), its barrel and the baffle vanes onto the sensor as a
// veiling glare over the whole image. Its share follows the optics' point source transmittance,
// PST(theta) = E_sensor / E_source, the irradiance it puts on the sensor over the irradiance the
// source puts on the entrance aperture at normal incidence, as a function of the source's angle
// theta off the optical axis: highest near the field's edge, falling steeply with the angle (a
// power of it, about theta^-2 to theta^-3 for scatter off the optics), and zero once the
// aperture no longer sees the source (`cutoff`, 90 degrees past which the sun is behind the
// entrance plane; baffles bring it in). The veiling irradiance E_sun PST(theta) reads as the
// radiance PST(theta) times `PixelContext.sunFocalRadiance` (the camera equation inverted), added
// to each band across the image: its mean, times a linear ramp towards the side of the image
// nearest the sun, 1 at the centre and 1 + `gradient` half a diagonal away towards the sun (1 -
// `gradient` away from it). Every other distant light (Earthshine, a second star) adds a veil of
// its own the same way, by its own angle and irradiance.
//
// PST(theta) = strength (theta / angle)^-slope, held below `angle`; or, when `angles` (degrees)
// increase, strength times 10^`logValues` at theta, linear in log PST between them and held
// beyond them (a measured or modelled PST curve, with strength 1). Both arrays take up to 25
// points, the same number each; given in log10, the resampling to 25 keeps the curve straight in
// log PST between the points (vertices between samples are cut a little). With `inField` 0 there is none while the sun images on the sensor
// (through the pinhole projection), where `glare` scatters the image's own light instead.
//
// How it differs from `glare`: glare spreads light already in the image (a bright source in the
// field) through the lens's wide-angle scatter (a Harvey BSDF's power-law point spread), so it
// reads only the image; stray light comes from a source the image does not hold, so it needs the
// sun's direction and irradiance from the frame, and it lands as a nearly flat veil set by the
// PST rather than as a halo around anything. On the spectrum, before `qe` and `cie`, so every
// output sees it; not vignetted. At zero strength the image is untouched.
//
// References:
// - R. P. Breault, "Control of Stray Light", in M. Bass (ed.), "Handbook of Optics", 2nd ed.,
//   vol. I, chapter 38, McGraw-Hill, 1995 (point source transmittance, baffles, critical and
//   illuminated objects).
// - E. C. Fest, "Stray Light Analysis and Control", SPIE Press, 2013 (PST curves; Harvey scatter
//   off optical surfaces).
import sensor;

struct StrayLight : IBandFilter {
    struct Params {
        float strength;       // PST at `angle`, or the table's scale
        float angle;          // degrees off axis where the power law's PST is `strength`
        float slope;          // the power law's power of the angle
        float cutoff;         // degrees off axis beyond which the aperture sees no sun
        float gradient;       // the veil's linear rise towards the sun's side, 0 to 1
        float inField;        // 1: also while the sun images on the sensor
        float angles[25];     // a PST table's angles off axis, degrees, increasing
        float logValues[25];  // log10 of its PST at them
    }

    // The PST at `theta` degrees off axis.
    static float pst(float theta, Params params) {
        if (theta > params.cutoff) {
            return 0.0;
        }
        if (params.angles[24] <= params.angles[0]) {
            let reference = max(params.angle, 1e-3);
            return params.strength * pow(max(theta, reference) / reference, -params.slope);
        }
        var logPst = params.logValues[24];
        if (theta <= params.angles[0]) {
            logPst = params.logValues[0];
        } else {
            for (uint i = 1; i < 25; ++i) {
                if (theta <= params.angles[i]) {
                    let t = (theta - params.angles[i - 1]) /
                            max(params.angles[i] - params.angles[i - 1], 1e-6);
                    logPst = lerp(params.logValues[i - 1], params.logValues[i], t);
                    break;
                }
            }
        }
        return params.strength * exp2(logPst * 3.32192809);
    }

    // The veil of one source: `source` its direction in the camera's frame (w 1 when it lights
    // the frame), `focal` its focal radiance in this band.
    static float veil(float4 source, float focal, PixelContext pixel, Params params) {
        if (source.w < 0.5 || focal == 0.0) {
            return 0.0;
        }
        let cam = pixel.uniforms();
        let size = float2(cam.size.xy);
        // The source's place in the pinhole image, from the principal point.
        let ahead = -source.z;
        let image = cam.intrinsics.x * float2(source.x, -source.y) / max(ahead, 1e-6);
        let spot = cam.intrinsics.yz + image;
        if (params.inField < 0.5 && ahead > 0.0 && all(spot >= float2(0.0)) &&
            all(spot <= size)) {
            return 0.0;
        }
        let theta = degrees(acos(clamp(ahead, -1.0, 1.0)));
        // Towards the source across the image (y down), from the image's centre.
        let side = float2(source.x, -source.y);
        let towards = length(side) > 1e-6 ? normalize(side) : float2(0.0);
        let offset = float2(pixel.id.xy) + 0.5 - 0.5 * size;
        let ramp = dot(offset, towards) / max(0.5 * length(size), 1.0);
        let profile = 1.0 + saturate(params.gradient) * ramp;
        return pst(theta, params) * profile * focal;
    }

    // The sun's veil and every other distant light's (`PixelContext.sourceInCamera`), each by
    // its own angle off the axis.
    static float apply(float radiance, uint band, PixelContext pixel, Params params) {
        if (params.strength == 0.0) {
            return radiance;
        }
        var veiled = radiance +
                     veil(pixel.sunInCamera(), pixel.sunFocalRadiance(band), pixel, params);
        for (uint s = 0; s < kStraySources; ++s) {
            veiled += veil(pixel.sourceInCamera(s), pixel.sourceFocalRadiance(s, band), pixel,
                           params);
        }
        return veiled;
    }
}
