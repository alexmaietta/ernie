// SPDX-License-Identifier: BSD-3-Clause
// Lens projections for the sensor's lens head: each maps a sensor pixel's normalised image
// coordinates (x, y) = ((u - cx) / f, (cy - v) / f) to the pinhole image plane the raster image is
// drawn in (tan of the field angle along each axis), and gives the natural fall-off of irradiance
// there. The renderer compiles the sensor kernel with one of them (`SensorPlan::lens`, the
// generated module `lensProjection`), so a pinhole camera pays for no distortion at all.
//
// `k` and `p2` are SensorUniforms' `distortion` and `lens.x`: Brown-Conrady's k1, k2, k3, p1 and
// p2, or the fisheye's k1 to k4.
//
// References:
// - D. C. Brown, "Decentering Distortion of Lenses", Photogrammetric Engineering 32(3):444-462,
//   1966 (in OpenCV's convention).
// - J. Kannala and S. S. Brandt, "A Generic Camera Model and Calibration Method for Conventional,
//   Wide-Angle, and Fish-Eye Lenses", IEEE TPAMI 28(8):1335-1340, 2006.
//   https://doi.org/10.1109/TPAMI.2006.153 (OpenCV's fisheye model).
// - B. K. P. Horn, "Robot Vision", MIT Press, 1986 (cos^4 fall-off of image irradiance).

// cos^4 of the field angle of pinhole image point `pinhole`: 1 / (1 + r^2)^2.
float cosine4(float2 pinhole) {
    let cosine2 = 1.0 / (1.0 + dot(pinhole, pinhole));
    return cosine2 * cosine2;
}

// Each projection's `rayOf` gives the unit ray, in the camera's frame (x right, y up, looking down
// -z), that lands at the sensor coordinates: what a camera rendered on cube faces follows
// (cube_faces.slang), valid past 90 degrees off axis where the pinhole image is not.

// No distortion: the sensor's image is the pinhole image.
public struct Pinhole {
    public static float2 pinholeOf(float2 sensor, float4 k, float p2) {
        return sensor;
    }

    public static float3 rayOf(float2 sensor, float4 k, float p2) {
        return normalize(float3(sensor, -1.0));
    }

    public static float falloff(float2 sensor, float2 pinhole, float4 k) {
        return cosine4(pinhole);
    }
}

// Brown-Conrady radial and tangential distortion, inverted by fixed-point iteration.
public struct BrownConrady {
    static float2 distort(float2 p, float4 k, float p2) {
        let r2 = dot(p, p);
        let radial = 1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z));
        return float2(p.x * radial + 2.0 * k.w * p.x * p.y + p2 * (r2 + 2.0 * p.x * p.x),
                      p.y * radial + k.w * (r2 + 2.0 * p.y * p.y) + 2.0 * p2 * p.x * p.y);
    }

    public static float2 pinholeOf(float2 sensor, float4 k, float p2) {
        var p = sensor;
        for (int i = 0; i < 12; ++i) {
            p += sensor - distort(p, k, p2);
        }
        return p;
    }

    public static float3 rayOf(float2 sensor, float4 k, float p2) {
        return normalize(float3(pinholeOf(sensor, k, p2), -1.0));
    }

    public static float falloff(float2 sensor, float2 pinhole, float4 k) {
        return cosine4(pinhole);
    }
}

// A pushbroom's detector line, with Brown-Conrady radial and tangential distortion: its rows are
// its readout over time, not space, so a sensor pixel is a point of the line, whose pinhole
// point (`undistort`) lies along the line and, for a line off the axis or tangential terms, off
// it. sensor.slang's `pushbroomTapAt` takes it there, onto the raster row of the pixel's own
// time, and `retimed` onto the row whose pose sees the point on the line. `k` is (k1, k2, k3,
// the line's normalised y), `tangential` (p1, p2).
public struct Pushbroom {
    static float2 distort(float2 p, float3 k, float2 tangential) {
        let r2 = dot(p, p);
        let radial = 1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z));
        return float2(p.x * radial + 2.0 * tangential.x * p.x * p.y +
                          tangential.y * (r2 + 2.0 * p.x * p.x),
                      p.y * radial + tangential.x * (r2 + 2.0 * p.y * p.y) +
                          2.0 * tangential.y * p.x * p.y);
    }

    // The pinhole image point of normalised image point `distorted`, by fixed-point iteration.
    public static float2 undistort(float2 distorted, float4 k, float2 tangential) {
        var p = distorted;
        for (int i = 0; i < 12; ++i) {
            p += distorted - distort(p, k.xyz, tangential);
        }
        return p;
    }

    // The lens projection interface, unused: sensor.slang maps a pushbroom's pixels itself.
    public static float2 pinholeOf(float2 sensor, float4 k, float p2) {
        return sensor;
    }

    // A pushbroom is never rendered on cube faces.
    public static float3 rayOf(float2 sensor, float4 k, float p2) {
        return normalize(float3(sensor, -1.0));
    }

    // cos^4 of the field angle of the pixel's ray, its pinhole point on (or off) the line.
    public static float falloff(float2 sensor, float2 pinhole, float4 k) {
        return cosine4(pinhole);
    }
}

// Kannala-Brandt fisheye: a ray at field angle theta lands at radius
// theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) in normalised
// coordinates, in the ray's azimuth.
public struct Fisheye {
    static float radius(float theta, float4 k) {
        let t2 = theta * theta;
        return theta * (1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w))));
    }

    static float slope(float theta, float4 k) {
        let t2 = theta * theta;
        return 1.0 + t2 * (3.0 * k.x + t2 * (5.0 * k.y + t2 * (7.0 * k.z + t2 * 9.0 * k.w)));
    }

    // The field angle of the ray landing at radius `rd`, by Newton's method from theta = rd.
    static float fieldAngle(float rd, float4 k) {
        var theta = rd;
        for (int i = 0; i < 8; ++i) {
            theta -= (radius(theta, k) - rd) / slope(theta, k);
        }
        return theta;
    }

    public static float2 pinholeOf(float2 sensor, float4 k, float p2) {
        let rd = length(sensor);
        if (rd < 1e-8) {
            return sensor;
        }
        return sensor * (tan(fieldAngle(rd, k)) / rd);
    }

    // At the field angle itself, which may pass 90 degrees.
    public static float3 rayOf(float2 sensor, float4 k, float p2) {
        let rd = length(sensor);
        if (rd < 1e-8) {
            return float3(0.0, 0.0, -1.0);
        }
        let theta = fieldAngle(rd, k);
        return float3(sensor / rd * sin(theta), -cos(theta));
    }

    // The irradiance a pixel collects relative to the axis: cos(theta) of the aperture's
    // foreshortening times the solid angle a unit of image area sees, sin(theta) /
    // (theta_d theta_d'), which is cos^4 for the pinhole's theta_d = tan(theta).
    public static float falloff(float2 sensor, float2 pinhole, float4 k) {
        let rd = length(sensor);
        if (rd < 1e-8) {
            return 1.0;
        }
        let theta = fieldAngle(rd, k);
        // |cos|: past 90 degrees the aperture is foreshortened as much again.
        return abs(cos(theta)) * sin(theta) / (rd * slope(theta, k));
    }
}
