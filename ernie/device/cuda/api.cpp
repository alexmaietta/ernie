// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/cuda/api.h"

#include <utility>
#include <variant>

#include <fmt/format.h>

#include "ernie/util/error.h"
#include "ernie/util/library.h"
#include "ernie/util/log.h"

namespace ernie::device::cuda {
namespace {

using CUdriverProcAddressQueryResult = int;
using GetProcAddress = CUresult (*)(const char* symbol, void** function, int cudaVersion,
                                    uint64_t flags, CUdriverProcAddressQueryResult* status);

void ResolveFunctions(SymbolResolver& resolve, Api& api) {
    resolve("cuInit", api.cuInit);
    resolve("cuDriverGetVersion", api.cuDriverGetVersion);
    resolve("cuDeviceGetCount", api.cuDeviceGetCount);
    resolve("cuGetErrorName", api.cuGetErrorName);
    resolve("cuGetErrorString", api.cuGetErrorString);
    resolve("cuDeviceGet", api.cuDeviceGet);
    resolve("cuDeviceGetName", api.cuDeviceGetName);
    resolve("cuDeviceGetUuid", api.cuDeviceGetUuid);
    resolve("cuDeviceTotalMem", api.cuDeviceTotalMem);
    resolve("cuDeviceGetAttribute", api.cuDeviceGetAttribute);
    resolve("cuDevicePrimaryCtxRetain", api.cuDevicePrimaryCtxRetain);
    resolve("cuDevicePrimaryCtxRelease", api.cuDevicePrimaryCtxRelease);
    resolve("cuCtxPushCurrent", api.cuCtxPushCurrent);
    resolve("cuCtxPopCurrent", api.cuCtxPopCurrent);
    resolve("cuMemAlloc", api.cuMemAlloc);
    resolve("cuMemFree", api.cuMemFree);
    resolve("cuMemGetInfo", api.cuMemGetInfo);
    resolve("cuMemPoolCreate", api.cuMemPoolCreate);
    resolve("cuMemPoolDestroy", api.cuMemPoolDestroy);
    resolve("cuMemPoolSetAttribute", api.cuMemPoolSetAttribute);
    resolve("cuMemAllocFromPoolAsync", api.cuMemAllocFromPoolAsync);
    resolve("cuMemFreeAsync", api.cuMemFreeAsync);
    resolve("cuMemHostAlloc", api.cuMemHostAlloc);
    resolve("cuMemFreeHost", api.cuMemFreeHost);
    resolve("cuMemHostGetDevicePointer", api.cuMemHostGetDevicePointer);
    resolve("cuStreamCreate", api.cuStreamCreate);
    resolve("cuStreamDestroy", api.cuStreamDestroy);
    resolve("cuStreamSynchronize", api.cuStreamSynchronize);
    resolve("cuEventCreate", api.cuEventCreate);
    resolve("cuEventDestroy", api.cuEventDestroy);
    resolve("cuEventQuery", api.cuEventQuery);
    resolve("cuEventSynchronize", api.cuEventSynchronize);
    resolve("cuEventRecordWithFlags", api.cuEventRecordWithFlags);
    resolve("cuStreamWaitEvent", api.cuStreamWaitEvent);
    resolve("cuLaunchHostFunc", api.cuLaunchHostFunc);
    resolve("cuImportExternalSemaphore", api.cuImportExternalSemaphore);
    resolve("cuDestroyExternalSemaphore", api.cuDestroyExternalSemaphore);
    resolve("cuImportExternalMemory", api.cuImportExternalMemory);
    resolve("cuExternalMemoryGetMappedBuffer", api.cuExternalMemoryGetMappedBuffer);
    resolve("cuDestroyExternalMemory", api.cuDestroyExternalMemory);
    resolve("cuMemcpyDtoH", api.cuMemcpyDtoH);
    resolve("cuSignalExternalSemaphoresAsync", api.cuSignalExternalSemaphoresAsync);
    resolve("cuWaitExternalSemaphoresAsync", api.cuWaitExternalSemaphoresAsync);
    resolve("cuEventElapsedTime", api.cuEventElapsedTime);
    resolve("cuMemcpyAsync", api.cuMemcpyAsync);
    resolve("cuMemsetD32Async", api.cuMemsetD32Async);
    resolve("cuMemcpyHtoDAsync", api.cuMemcpyHtoDAsync);
    resolve("cuModuleLoadData", api.cuModuleLoadData);
    resolve("cuModuleUnload", api.cuModuleUnload);
    resolve("cuModuleGetGlobal", api.cuModuleGetGlobal);
    resolve("cuMemcpyHtoD", api.cuMemcpyHtoD);
    resolve("cuModuleGetFunction", api.cuModuleGetFunction);
    resolve("cuFuncGetAttribute", api.cuFuncGetAttribute);
    resolve("cuLaunchKernel", api.cuLaunchKernel);
    resolve("cuStreamBeginCapture", api.cuStreamBeginCapture);
    resolve("cuStreamEndCapture", api.cuStreamEndCapture);
    resolve("cuGraphInstantiateWithFlags", api.cuGraphInstantiateWithFlags);
    resolve("cuGraphLaunch", api.cuGraphLaunch);
    resolve("cuGraphExecDestroy", api.cuGraphExecDestroy);
    resolve("cuGraphDestroy", api.cuGraphDestroy);
    resolve("cuMipmappedArrayCreate", api.cuMipmappedArrayCreate);
    resolve("cuMipmappedArrayGetLevel", api.cuMipmappedArrayGetLevel);
    resolve("cuMipmappedArrayDestroy", api.cuMipmappedArrayDestroy);
    resolve("cuMemcpy2DAsync", api.cuMemcpy2DAsync);
    resolve("cuTexObjectCreate", api.cuTexObjectCreate);
    resolve("cuTexObjectDestroy", api.cuTexObjectDestroy);
}

Loaded<Api> Initialize(Api api) {
    int version = 0;
    api.cuDriverGetVersion(&version);
    api.driverVersion = DecodeVersion(version);
    if (api.driverVersion < kMinimumDriverVersion) {
        return TooOld("the CUDA driver", api.driverVersion, kMinimumDriverVersion);
    }

    const CUresult initialized = api.cuInit(0);
    if (initialized == CUDA_ERROR_NO_DEVICE) {
        return "the CUDA driver found no device";
    }
    if (initialized != CUDA_SUCCESS) {
        return fmt::format("cuInit failed: {}", api.ErrorName(initialized));
    }

    Log(LogLevel::Debug, "the CUDA driver supports CUDA {}", api.driverVersion.ToString());
    return api;
}

Loaded<Api> Load() {
#if defined(_WIN32)
    const auto names = {"nvcuda.dll"};
#else
    const auto names = {"libcuda.so.1", "libcuda.so"};
#endif
    const Library library = Library::Load(names);
    if (!library) {
        return "no CUDA driver was found";
    }
    GetProcAddress getProcAddress = nullptr;
    if (!library.Symbol("cuGetProcAddress_v2", getProcAddress)) {
        return fmt::format("{} is older than CUDA 12.0", library.Name());
    }

    constexpr int kAbi = 12000;
    SymbolResolver resolve([getProcAddress](const char* symbol) -> void* {
        void* address = nullptr;
        CUdriverProcAddressQueryResult status = 0;
        return getProcAddress(symbol, &address, kAbi, 0, &status) == CUDA_SUCCESS ? address
                                                                                  : nullptr;
    });

    Api api;
    api.library = library.Name();
    ResolveFunctions(resolve, api);
    if (resolve.Missing() != nullptr) {
        return fmt::format("{} lacks {}", api.library, resolve.Missing());
    }
    return Initialize(std::move(api));
}

const Loaded<Api>& Cached() {
    static const Loaded<Api> loaded = LogUnavailable("CUDA", Load());
    return loaded;
}

}

bool Available() {
    return std::holds_alternative<Api>(Cached());
}

const Api& GetApi() {
    return RequireLoaded(Cached());
}

const compute::RuntimeCompiler& GetNvrtc() {
#if defined(_WIN32)
    const auto names = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll"};
#else
    const auto names = {"libnvrtc.so.13", "libnvrtc.so.12"};
#endif
    static const Loaded<compute::RuntimeCompiler> loaded =
        LogUnavailable("NVRTC", compute::LoadRuntimeCompiler("NVRTC", "nvrtc", "CUBIN", names, {},
                                                             kMinimumDriverVersion));
    return RequireLoaded(loaded);
}

const char* Api::ErrorName(CUresult result) const {
    const char* name = nullptr;
    if (cuGetErrorName == nullptr || cuGetErrorName(result, &name) != CUDA_SUCCESS ||
        name == nullptr) {
        return "unknown";
    }
    return name;
}

void Api::Warn(CUresult result, const char* what) const {
    if (result != CUDA_SUCCESS) {
        Log(LogLevel::Warning, "{} failed: {}", what, ErrorName(result));
    }
}

void Api::Check(CUresult result, const char* what) const {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* text = nullptr;
    if (cuGetErrorString == nullptr || cuGetErrorString(result, &text) != CUDA_SUCCESS ||
        text == nullptr) {
        text = "";
    }
    if (LostDevice(result)) {
        throw DeviceLostError("{} failed: the device was lost: {} ({}): {}", what,
                              ErrorName(result), result, text);
    }
    throw ApiError(what, ErrorName(result), result, text);
}

}
