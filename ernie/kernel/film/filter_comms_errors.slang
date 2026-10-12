// SPDX-License-Identifier: BSD-3-Clause
// Communication errors: the image as received over a noisy downlink, its digital numbers sent in
// raster order in packets of `packetPixels` pixels, new errors every frame:
// - lost packets (`packetLoss`, each packet's chance): their pixels read `fill`, as the gaps a
//   ground system leaves where telemetry never arrived;
// - slips (`slips`, each packet's chance): a sample dropped or repeated, so the rest of the line
//   from that packet on moves left or right by 1 to `maxSlip` pixels, slips adding up along a
//   line; pixels moved past the image's edge read `fill`;
// - bit errors (`bitErrorRate`, each bit's chance): a pixel's number with one of the ADC's bits
//   flipped, in one of its channels.
// On digital numbers, after `adc` (and any companding). AOV `errors`: 1 where a packet was lost, 2
// where the line slipped, 4 where a bit flipped, summed.
//
// References:
// - CCSDS 133.0-B-2, "Space Packet Protocol", Blue Book, 2020 (telemetry in packets).
// - CCSDS 131.0-B-3, "TM Synchronization and Channel Coding", Blue Book, 2017 (bit errors and
//   frame synchronisation on the downlink).
// - STAR-Dundee, "PANGU v7 Datasheet", 2024 (communication errors in its camera model).
import sensor;

uint3 commsHash(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Uniform in (0, 1].
float commsUniform(uint bits) {
    return (float(bits >> 8) + 1.0) / 16777216.0;
}

struct CommsErrors : ISpatialPixelFilter {
    struct Params {
        float packetPixels;  // pixels per packet, in raster order
        float packetLoss;    // chance a packet is lost
        float slips;         // chance a packet slips the rest of its line
        float maxSlip;       // the largest slip, pixels
        float bitErrorRate;  // chance each bit flips
        float fill;          // DN of pixels without data
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let size = input.size(pixel);
        let width = uint(size.x);
        let x = pixel.id.x;
        let y = pixel.id.y;
        let perPacket = uint(max(params.packetPixels, 1.0));
        let packet = (y * width + x) / perPacket;
        let frameSeed = pixel.uniforms().noise.z;
        // The slips of the line's packets up to this one, at most 4096 back.
        let first = max((y * width) / perPacket, packet > 4096u ? packet - 4096u : 0u);
        var shift = 0;
        for (uint q = first; q <= packet; ++q) {
            let h = commsHash(uint3(q, frameSeed, 0x534c4950u));
            if (commsUniform(h.x) <= params.slips) {
                let by = 1 + int(floor(commsUniform(h.y) * max(params.maxSlip, 1.0)));
                shift += commsUniform(h.z) < 0.5 ? -by : by;
            }
        }
        var errors = shift != 0 ? 2.0 : 0.0;
        let source = int(x) + shift;
        let largest = pixel.uniforms().adc.w;
        var dn = source >= 0 && source < size.x ? input.values(pixel, int2(source, int(y)))
                                                : float4(params.fill);
        let lost = commsHash(uint3(packet, frameSeed, 0x4c4f5354u));
        if (commsUniform(lost.x) <= params.packetLoss) {
            dn = float4(params.fill);
            errors += 1.0;
        }
        // One flipped bit: bit floor(u / rate) when that is below the ADC's bits.
        let bits = uint(round(log2(largest + 1.0)));
        let u = pixel.random(0x464c4950u);
        let rate = max(params.bitErrorRate, 1e-30);
        if (u < rate * float(bits)) {
            let bit = min(uint(u / rate), bits - 1u);
            let channel = min(uint(pixel.random(0x4348414eu) * float(pixel.channels())),
                              pixel.channels() - 1u);
            dn[channel] = float(uint(dn[channel]) ^ (1u << bit));
            errors += 4.0;
        }
        var digital = uint4(clamp(dn, float4(0.0), float4(largest)));
        for (uint c = pixel.channels(); c < 4; ++c) {
            digital[c] = 0;
        }
        value.digital = digital;
        writeOutput(pixel, 0, float4(errors));
    }
}
