// White balance: gains on linear colour's red, green and blue (von Kries scaling in the output
// space), as a camera's white balance multiplies its channels so a grey under the scene's light
// comes out grey. On linear colour, after `cie` and before `exposure`.
//
// Reference: J. von Kries, "Influence of adaptation on the effects produced by luminous stimuli",
// 1905, in D. L. MacAdam (ed.), "Sources of Color Science", MIT Press, 1970 (channel scaling).
import sensor;

struct WhiteBalance : IPixelFilter {
    struct Params {
        float red;
        float green;
        float blue;
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        value.rgb *= float3(params.red, params.green, params.blue);
    }
}
