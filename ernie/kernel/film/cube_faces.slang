// SPDX-License-Identifier: BSD-3-Clause
// The cube faces a wide camera's views are rendered on (ernie/scene/wide_view.h holds the same
// conventions on the CPU): in the camera's frame (x right, y up, looking down -z), face 0 looks
// down -z, 1 along +x, 2 along -x, 3 along +y, 4 along -y and 5 (the back, for fields past 125
// degrees off axis) along +z, each a square pinhole view of 90 degrees plus a margin. A sensor
// pixel's lens ray picks the face it leaves through by its largest component, and lands on that
// face's image plane.
//
// References:
// - N. Greene, "Environment Mapping and Other Applications of World Projections", IEEE Computer
//   Graphics and Applications 6(11):21-29, 1986 (cube maps).
// - Khronos, "Vulkan Specification", 16.5.4 "Cube Map Face Selection".

// Face `face`'s axes in the camera's frame, as the rows of a matrix taking camera rays to the
// face's frame (it looks down its -z, as the camera does on face 0).
public float3x3 cubeFaceToFace(int face) {
    switch (face) {
    case 1:  // along +x
        return float3x3(float3(0, 0, 1), float3(0, 1, 0), float3(-1, 0, 0));
    case 2:  // along -x
        return float3x3(float3(0, 0, -1), float3(0, 1, 0), float3(1, 0, 0));
    case 3:  // along +y
        return float3x3(float3(1, 0, 0), float3(0, 0, 1), float3(0, -1, 0));
    case 4:  // along -y
        return float3x3(float3(1, 0, 0), float3(0, 0, -1), float3(0, 1, 0));
    case 5:  // along +z, turned about y
        return float3x3(float3(-1, 0, 0), float3(0, 1, 0), float3(0, 0, -1));
    default:
        return float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    }
}

// The face, of the first `faces` (5, or 6 with the back), whose view direction lies nearest unit
// camera ray `ray`: the largest of -z, x, -x, y, -y and z (ties to the earlier face, as on the
// CPU).
public int cubeFaceOf(float3 ray, int faces) {
    var best = 0;
    var largest = -ray.z;
    let others = float4(ray.x, -ray.x, ray.y, -ray.y);
    [unroll]
    for (int i = 0; i < 4; ++i) {
        if (others[i] > largest) {
            largest = others[i];
            best = i + 1;
        }
    }
    if (faces > 5 && ray.z > largest) {
        best = 5;
    }
    return best;
}

// Where `ray` crosses face `face`'s image plane at unit distance: tangent-plane coordinates,
// x right and y up on the face, within [-1, 1] across its 90 degrees.
public float2 cubeFacePlane(int face, float3 ray) {
    let local = mul(cubeFaceToFace(face), ray);
    return local.xy / -local.z;
}
