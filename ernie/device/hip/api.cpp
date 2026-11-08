// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/hip/api.h"

#include <format>
#include <utility>
#include <variant>

#include "ernie/util/error.h"
#include "ernie/util/library.h"
#include "ernie/util/log.h"

namespace ernie::device::hip {
namespace {

void ResolveFunctions(SymbolResolver& resolve, Api& api) {
    resolve("hipInit", api.hipInit);
    resolve("hipRuntimeGetVersion", api.hipRuntimeGetVersion);
    resolve("hipDriverGetVersion", api.hipDriverGetVersion);
    resolve("hipGetDeviceCount", api.hipGetDeviceCount);
    resolve("hipGetErrorName", api.hipGetErrorName);
    resolve("hipGetErrorString", api.hipGetErrorString);
    resolve("hipGetLastError", api.hipGetLastError);
    resolve("hipPeekAtLastError", api.hipPeekAtLastError);
    resolve("hipDeviceGetName", api.hipDeviceGetName);
    resolve("hipDeviceGetUuid", api.hipDeviceGetUuid);
    resolve("hipDeviceTotalMem", api.hipDeviceTotalMem);
    resolve("hipDeviceGetAttribute", api.hipDeviceGetAttribute);
    resolve("hipGetDevicePropertiesR0600", api.hipGetDevicePropertiesR0600);
    resolve("hipGetDevice", api.hipGetDevice);
    resolve("hipSetDevice", api.hipSetDevice);
    resolve("hipMalloc", api.hipMalloc);
    resolve("hipFree", api.hipFree);
    resolve("hipMemGetInfo", api.hipMemGetInfo);
    resolve("hipMemPoolCreate", api.hipMemPoolCreate);
    resolve("hipMemPoolDestroy", api.hipMemPoolDestroy);
    resolve("hipMemPoolSetAttribute", api.hipMemPoolSetAttribute);
    resolve("hipMallocFromPoolAsync", api.hipMallocFromPoolAsync);
    resolve("hipFreeAsync", api.hipFreeAsync);
    resolve("hipHostMalloc", api.hipHostMalloc);
    resolve("hipHostFree", api.hipHostFree);
    resolve("hipHostGetDevicePointer", api.hipHostGetDevicePointer);
    resolve("hipStreamCreateWithFlags", api.hipStreamCreateWithFlags);
    resolve("hipStreamDestroy", api.hipStreamDestroy);
    resolve("hipStreamSynchronize", api.hipStreamSynchronize);
    resolve("hipEventCreateWithFlags", api.hipEventCreateWithFlags);
    resolve("hipEventDestroy", api.hipEventDestroy);
    resolve("hipEventQuery", api.hipEventQuery);
    resolve("hipEventSynchronize", api.hipEventSynchronize);
    resolve("hipEventRecordWithFlags", api.hipEventRecordWithFlags);
    resolve("hipStreamWaitEvent", api.hipStreamWaitEvent);
    resolve("hipLaunchHostFunc", api.hipLaunchHostFunc);
    resolve("hipImportExternalSemaphore", api.hipImportExternalSemaphore);
    resolve("hipDestroyExternalSemaphore", api.hipDestroyExternalSemaphore);
    resolve("hipImportExternalMemory", api.hipImportExternalMemory);
    resolve("hipExternalMemoryGetMappedBuffer", api.hipExternalMemoryGetMappedBuffer);
    resolve("hipDestroyExternalMemory", api.hipDestroyExternalMemory);
    resolve("hipMemcpyDtoH", api.hipMemcpyDtoH);
    resolve("hipSignalExternalSemaphoresAsync", api.hipSignalExternalSemaphoresAsync);
    resolve("hipWaitExternalSemaphoresAsync", api.hipWaitExternalSemaphoresAsync);
    resolve("hipEventElapsedTime", api.hipEventElapsedTime);
    resolve("hipMemcpyAsync", api.hipMemcpyAsync);
    resolve("hipMemsetD32Async", api.hipMemsetD32Async);
    resolve("hipMemcpyHtoDAsync", api.hipMemcpyHtoDAsync);
    resolve("hipModuleLoadData", api.hipModuleLoadData);
    resolve("hipModuleUnload", api.hipModuleUnload);
    resolve("hipModuleGetGlobal", api.hipModuleGetGlobal);
    resolve("hipMemcpyHtoD", api.hipMemcpyHtoD);
    resolve("hipModuleGetFunction", api.hipModuleGetFunction);
    resolve("hipFuncGetAttribute", api.hipFuncGetAttribute);
    resolve("hipModuleLaunchKernel", api.hipModuleLaunchKernel);
    resolve("hipStreamBeginCapture", api.hipStreamBeginCapture);
    resolve("hipStreamEndCapture", api.hipStreamEndCapture);
    resolve("hipGraphInstantiateWithFlags", api.hipGraphInstantiateWithFlags);
    resolve("hipGraphLaunch", api.hipGraphLaunch);
    resolve("hipGraphExecDestroy", api.hipGraphExecDestroy);
    resolve("hipGraphDestroy", api.hipGraphDestroy);
    resolve("hipMipmappedArrayCreate", api.hipMipmappedArrayCreate);
    resolve("hipMipmappedArrayGetLevel", api.hipMipmappedArrayGetLevel);
    resolve("hipMipmappedArrayDestroy", api.hipMipmappedArrayDestroy);
    resolve("hipMemcpyParam2DAsync", api.hipMemcpyParam2DAsync);
    resolve("hipTexObjectCreate", api.hipTexObjectCreate);
    resolve("hipTexObjectDestroy", api.hipTexObjectDestroy);
}

Loaded<Api> Initialize(Api api) {
    int version = 0;
    api.hipRuntimeGetVersion(&version);
    api.runtimeVersion = DecodeVersion(version);
    if (api.runtimeVersion < kMinimumRuntimeVersion) {
        return TooOld("the HIP runtime", api.runtimeVersion, kMinimumRuntimeVersion);
    }

    const hipError_t initialized = api.hipInit(0);
    if (initialized == hipErrorNoDevice) {
        return "the HIP runtime found no device";
    }
    if (initialized != hipSuccess) {
        return std::format("hipInit failed: {}", api.ErrorName(initialized));
    }

    api.hipDriverGetVersion(&version);
    api.driverVersion = DecodeVersion(version);
    Log(LogLevel::Debug, "the HIP runtime is version {}", api.runtimeVersion.ToString());
    return api;
}

Loaded<Api> Load() {
#if defined(_WIN32)
    const auto names = {"amdhip64_7.dll", "amdhip64_6.dll"};
    const std::initializer_list<const char*> paths = {};
#else
    const auto names = {"libamdhip64.so.7", "libamdhip64.so.6"};
    const auto paths = {"/opt/rocm/lib/libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so.6"};
#endif
    const Library library = Library::Load(names, paths);
    if (!library) {
        return "no HIP runtime was found";
    }

    SymbolResolver resolve(library);
    Api api;
    api.library = library.Name();
    ResolveFunctions(resolve, api);
    if (resolve.Missing() != nullptr) {
        return std::format("{} lacks {}", api.library, resolve.Missing());
    }
    return Initialize(std::move(api));
}

const Loaded<Api>& Cached() {
    static const Loaded<Api> loaded = LogUnavailable("HIP", Load());
    return loaded;
}

}

bool Available() {
    return std::holds_alternative<Api>(Cached());
}

const Api& GetApi() {
    return RequireLoaded(Cached());
}

const compute::RuntimeCompiler& GetHiprtc() {
#if defined(_WIN32)
    const auto names = {"hiprtc0702.dll", "hiprtc0602.dll"};
    const std::initializer_list<const char*> paths = {};
#else
    const auto names = {"libhiprtc.so.7", "libhiprtc.so.6"};
    const auto paths = {"/opt/rocm/lib/libhiprtc.so.7", "/opt/rocm/lib/libhiprtc.so.6"};
#endif
    static const Loaded<compute::RuntimeCompiler> loaded =
        LogUnavailable("hipRTC", compute::LoadRuntimeCompiler("hipRTC", "hiprtc", "Code", names,
                                                              paths, kMinimumRuntimeVersion));
    return RequireLoaded(loaded);
}

const char* Api::ErrorName(hipError_t error) const {
    const char* name = hipGetErrorName != nullptr ? hipGetErrorName(error) : nullptr;
    return name != nullptr ? name : "unknown";
}

void Api::Warn(hipError_t error, const char* what) const {
    if (error != hipSuccess) {
        Log(LogLevel::Warning, "{} failed: {}", what, ErrorName(error));
    }
}

void Api::Check(hipError_t error, const char* what) const {
    if (error == hipSuccess) {
        return;
    }
    const char* text = hipGetErrorString != nullptr ? hipGetErrorString(error) : nullptr;
    if (LostDevice(error)) {
        throw DeviceLostError("{} failed: the device was lost: {} ({}): {}", what, ErrorName(error),
                              error, text != nullptr ? text : "");
    }
    throw ApiError(what, ErrorName(error), error, text != nullptr ? text : "");
}

}
