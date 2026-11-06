// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.trace.software;

#if ERNIE_HARDWARE_TRAVERSAL
__exported import ernie.render.kernels.trace.hardware;
typealias Tracer = HardwareTracer;
#else
typealias Tracer = SoftwareTracer;
#endif
