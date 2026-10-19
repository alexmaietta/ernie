// SPDX-License-Identifier: BSD-3-Clause
/// @file frame_count.slang
/// @brief The device scene's frame number, counted on the device: one thread adds one each
/// frame, so frames replayed from recorded commands count too, and the random numbers keyed by
/// frame differ from one to the next.
import buffer_ref;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform BufferRef<uint> frame) {
    frame[0] = frame[0] + 1;
}
