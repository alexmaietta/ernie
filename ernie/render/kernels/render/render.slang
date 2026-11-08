// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.render.kernels.integrator.minimal;
import ernie.render.kernels.integrator.realtime;
import ernie.render.kernels.random;
import ernie.render.kernels.sensors.camera.response;
import ernie.render.kernels.sensors.lidar.response;
import ernie.render.kernels.sensors.points;
import ernie.render.kernels.sensors.radar.response;

using namespace ernie;

static const uint kPixelTileWidth = 8;
static const uint kPixelTileHeight = 4;

bool FindPixel(RenderParams params, out CapturePixel pixel) {
    pixel.capture = params.captures;
    pixel.coord = uint2(0);
    pixel.index = 0;
    const uint thread = LinearThreadIndex();
    if (thread >= params.pixelCount) {
        return false;
    }
    const uint64_t flat = params.firstPixel + thread;
    uint low = 0;
    uint high = params.captureCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (params.captures[middle].plan.firstPixel <= flat) {
            low = middle;
        } else {
            high = middle;
        }
    }
    Capture* capture = params.captures + low;
    const uint local = uint(flat - capture->plan.firstPixel);
    pixel.capture = capture;
    pixel.coord = uint2(local % capture->plan.width, local / capture->plan.width);
    if (capture->plan.width % kPixelTileWidth == 0 &&
        capture->plan.height % kPixelTileHeight == 0) {
        const uint tileSize = kPixelTileWidth * kPixelTileHeight;
        const uint tilesAcross = capture->plan.width / kPixelTileWidth;
        const uint tile = local / tileSize;
        const uint lane = local % tileSize;
        pixel.coord = uint2((tile % tilesAcross) * kPixelTileWidth + lane % kPixelTileWidth,
                             (tile / tilesAcross) * kPixelTileHeight + lane / kPixelTileWidth);
    }
    pixel.index = (capture->plan.layer * capture->plan.height + pixel.coord.y) *
                      capture->plan.width +
                  pixel.coord.x;
    return true;
}

void RenderPixel<P : IPattern, R : IResponse<P>>(RenderParams params) {
    CapturePixel pixel;
    if (!FindPixel(params, pixel)) {
        return;
    }
    Capture* capture = pixel.capture;
    const SensorContext context = SensorContext(params, capture);
    const P pattern = P.Load(context);
    R response = R.Load(context, pattern);
    for (uint sample = 0; sample < pattern.SampleCount(); ++sample) {
        Rng rng = Rng(capture->plan.sensor, pixel.index, sample, capture->plan.product);
        response.Add(pattern.Generate(pixel, sample), rng);
    }
    response.Write(context, pattern, pixel);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalData(uniform RenderParams params) {
    RenderPixel<CameraPattern, CameraResponse<MinimalIntegrator<MinimalMode::Data>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalDisplayColor(uniform RenderParams params) {
    RenderPixel<CameraPattern, CameraResponse<MinimalIntegrator<MinimalMode::DisplayColor>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalTextureDiffuse(uniform RenderParams params) {
    RenderPixel<CameraPattern, CameraResponse<MinimalIntegrator<MinimalMode::TextureDiffuse>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalFull(uniform RenderParams params) {
    RenderPixel<CameraPattern, CameraResponse<MinimalIntegrator<MinimalMode::Full>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderRealTime(uniform RenderParams params) {
    RenderPixel<CameraPattern, CameraResponse<RealTimeIntegrator<RgbSpectrum>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderRealTimeBands(uniform RenderParams params) {
    RenderPixel<CameraPattern, CameraResponse<RealTimeIntegrator<BandSpectrum>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderLidar(uniform RenderParams params) {
    RenderPixel<LidarPattern, LidarResponse>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderScatterPoints(uniform ScatterParams params) {
    ScatterPoints(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderRadar(uniform RenderParams params) {
    RenderPixel<RadarPattern, RadarResponse>(params);
}
