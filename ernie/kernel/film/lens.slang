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

// No distortion: the sensor's image is the pinhole image.
public struct Pinhole {
    public static float2 pinholeOf(float2 sensor, float4 k, float p2) {
        return sensor;
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

    public static float falloff(float2 sensor, float2 pinhole, float4 k) {
        return cosine4(pinhole);
    }
}

// A pushbroom's detector line, with Brown-Conrady radial distortion across it: its rows are its
// readout over time, not space, so only x moves. `k` is (k1, k2, k3, the line's normalised y);
// the row's coordinate is scaled by `overscan` so the raster row is the sensor's row (the raster
// image is drawn with the focal length over `overscan`, wide enough for the undistorted line).
public struct Pushbroom {
    static float2 distort(float2 p, float4 k) {
        let r2 = dot(p, p);
        return p * (1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z)));
    }

    public static float2 pinholeOf(float2 sensor, float4 k, float overscan) {
        let distorted = float2(sensor.x, k.w);
        var p = distorted;
        for (int i = 0; i < 12; ++i) {
            p += distorted - distort(p, k);
        }
        return float2(p.x, sensor.y * overscan);
    }

    // cos^4 of the field angle along the line, where the row's ray lies.
    public static float falloff(float2 sensor, float2 pinhole, float4 k) {
        return cosine4(float2(pinhole.x, k.w));
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

    // The irradiance a pixel collects relative to the axis: cos(theta) of the aperture's
    // foreshortening times the solid angle a unit of image area sees, sin(theta) /
    // (theta_d theta_d'), which is cos^4 for the pinhole's theta_d = tan(theta).
    public static float falloff(float2 sensor, float2 pinhole, float4 k) {
        let rd = length(sensor);
        if (rd < 1e-8) {
            return 1.0;
        }
        let theta = fieldAngle(rd, k);
        return cos(theta) * sin(theta) / (rd * slope(theta, k));
    }
}
