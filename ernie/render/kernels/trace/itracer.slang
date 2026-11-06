// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.trace.triangle;

interface ITracer {
    static Hit Closest<F : IHitFilter, I : IIntersector>(uint64_t accel, Ray ray, uint mask,
                                                         F filter, I intersector);

    static bool Any<F : IHitFilter, I : IIntersector>(uint64_t accel, Ray ray, uint mask,
                                                      F filter, I intersector);

    static Hit Trace<F : IHitFilter, I : IIntersector>(uint64_t accel, Ray ray, uint mask,
                                                       F filter, I intersector, bool any);
}

[Differentiable]
Hit DifferentiableClosest<T : ITracer, F : IHitFilter, I : IIntersector, C : ITriangles>(
    no_diff uint64_t accel, Ray ray, no_diff uint mask, F filter, I intersector, C triangles) {
    const Hit found = no_diff T.Closest(accel, detach(ray), mask, filter, intersector);
    if (found.procedural) {
        return intersector.Reintersect(ray, found);
    }
    return Reintersect(ray, found, triangles);
}
