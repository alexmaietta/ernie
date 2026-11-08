// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.camera;
__exported import ernie.render.kernels.spectrum.spectrum;

using namespace ernie;

static const float kNitSpectralRadiance = 1.0f / (683.0f * 106.857f);
static const float kPhotonsPerJouleNm = 5.0341165e15f;
static const uint kMostDetectorChannels = 8;

float CameraFalloff(float3 direction, float3 forward) {
    const float cosine = max(dot(normalize(direction), normalize(forward)), 0.0f);
    const float square = cosine * cosine;
    return square * square;
}

struct ChannelResponse {
    Ptr<DetectorRecord> detector;
    uint channels;
    float scale;

    static ChannelResponse Load(RenderTables* tables, uint sensor, CameraModel camera,
                                float2 resolution) {
        ChannelResponse response;
        response.detector = tables->detectors + ApiRow(tables, sensor, kApiDetectorAPI);
        response.channels = response.detector->channels;
        const CameraRecord record = tables->cameras[camera.lens.row];
        const float2 pixel = camera.sensor.aperture / resolution;
        const float area = pixel.x * pixel.y * 1e-6f;
        const float fNumber = max(record.fNumber, 1e-3f);
        response.scale = kNitSpectralRadiance * response.detector->bands.width *
                         kPhotonsPerJouleNm * area * kPi / (4 * fNumber * fNumber);
        return response;
    }

    float Rate(BandValues signal, uint channel) {
        const BandSet bands = detector->bands;
        float4 low = float4(0);
        float4 high = float4(0);
        [ForceUnroll]
        for (uint band = 0; band < 4; ++band) {
            low[band] = bands.first + (float(band) + 0.5f) * bands.width;
            high[band] = bands.first + (float(band) + 4.5f) * bands.width;
        }
        if (DetectorPattern(detector->pattern) == DetectorPattern::Bands) {
            const float4 qeLow = detector->qe[0].Vector();
            const float4 qeHigh = detector->qe[1].Vector();
            const float value = channel < 4 ? signal.low[channel] * qeLow[channel] * low[channel]
                                            : signal.high[channel - 4] * qeHigh[channel - 4] *
                                                  high[channel - 4];
            return value * scale;
        }
        return (dot(signal.low * low, detector->qe[2 * channel].Vector()) +
                dot(signal.high * high, detector->qe[2 * channel + 1].Vector())) *
               scale;
    }
}
