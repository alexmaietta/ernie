// SPDX-License-Identifier: BSD-3-Clause

import ernie.render.kernels.sensors.camera.pattern;
import ernie.render.kernels.sensors.camera.response;
import ernie.render.kernels.sensors.lidar.pattern;
import ernie.render.kernels.sensors.lidar.response;
import ernie.render.kernels.sensors.points;
import ernie.render.kernels.sensors.radar.pattern;
import ernie.render.kernels.sensors.radar.response;
import ernie.device.kernels.device;
import ernie.render.kernels.render.output;
import ernie.render.kernels.render.irender;
import ernie.render.kernels.render.minimal;
import ernie.render.kernels.render.realtime;
import ernie.render.kernels.spectrum.spectrum;
import ernie.render.kernels.random;

using namespace ernie;

static const uint kPixelTileWidth = 8;
static const uint kPixelTileHeight = 4;

bool FindPixel(RenderParams params, out CapturePixel target) {
    target.capture = params.captures;
    target.pixel = uint2(0);
    target.index = 0;
    const uint thread = LinearThreadIndex();
    if (thread >= params.pixelCount) {
        return false;
    }
    const uint64_t pixel = params.firstPixel + thread;
    uint low = 0;
    uint high = params.captureCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (params.captures[middle].firstPixel <= pixel) {
            low = middle;
        } else {
            high = middle;
        }
    }
    Capture* capture = params.captures + low;
    const uint local = uint(pixel - capture->firstPixel);
    target.capture = capture;
    target.pixel = uint2(local % capture->width, local / capture->width);
    if (capture->width % kPixelTileWidth == 0 && capture->height % kPixelTileHeight == 0) {
        const uint tileSize = kPixelTileWidth * kPixelTileHeight;
        const uint tilesAcross = capture->width / kPixelTileWidth;
        const uint tile = local / tileSize;
        const uint lane = local % tileSize;
        target.pixel = uint2((tile % tilesAcross) * kPixelTileWidth + lane % kPixelTileWidth,
                             (tile / tilesAcross) * kPixelTileHeight + lane / kPixelTileWidth);
    }
    target.index = (capture->layer * capture->height + target.pixel.y) * capture->width + target.pixel.x;
    return true;
}

void Render<P : IPattern, R : IResponse>(RenderParams params) {
    CapturePixel target;
    if (!FindPixel(params, target)) {
        return;
    }
    Capture* capture = target.capture;
    SceneData* data = params.scene;
    const HitQuery query = HitQuery(data, params.layout, data->prims.world[capture->sensor]);
    const P pattern = P.Load(params, capture);
    R response = R.Load(params, capture, query);
    for (uint sample = 0; sample < pattern.SampleCount(); ++sample) {
        Rng rng = Rng(capture->sensor, target.index, sample, capture->product);
        response.Add(pattern.Generate(target, sample), rng);
    }
    response.Write(params, target);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalData(uniform RenderParams params) {
    Render<CameraPattern, CameraResponse<MinimalRender<MinimalMode::Data>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalDisplayColor(uniform RenderParams params) {
    Render<CameraPattern, CameraResponse<MinimalRender<MinimalMode::DisplayColor>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalTextureDiffuse(uniform RenderParams params) {
    Render<CameraPattern, CameraResponse<MinimalRender<MinimalMode::TextureDiffuse>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderMinimalFull(uniform RenderParams params) {
    Render<CameraPattern, CameraResponse<MinimalRender<MinimalMode::Full>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderRealTime(uniform RenderParams params) {
    Render<CameraPattern, CameraResponse<RealTimeRender<RgbSpectrum>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderRealTimeBands(uniform RenderParams params) {
    Render<CameraPattern, CameraResponse<RealTimeRender<BandSpectrum>>>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderLidar(uniform RenderParams params) {
    Render<LidarPattern, LidarResponse>(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderScatterPoints(uniform ScatterParams params) {
    ScatterPoints(params);
}

[shader("compute")]
[numthreads(kRenderGroup, 1, 1)]
void RenderRadar(uniform RenderParams params) {
    Render<RadarPattern, RadarResponse>(params);
}
