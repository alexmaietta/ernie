// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include "ernie/device/queue.h"
#include "ernie/ernie.h"
#include "ernie/stage/views/layout.h"

namespace ernie::python {

SchemaId FindSchema(std::string_view name);
nanobind::object Numpy(std::vector<std::byte> bytes, const std::vector<size_t>& shape,
                       DLDataType dtype);

DLDataType DataType(device::DType type);
nanobind::tuple TensorDevice(const device::Tensor& tensor);
nanobind::object ExportTensor(const device::Tensor& tensor, std::optional<int64_t> stream,
                              std::optional<nanobind::tuple> maxVersion,
                              std::optional<nanobind::tuple> dlDevice, std::optional<bool> copy);

inline Stream ToStream(std::optional<uintptr_t> stream) {
    return stream ? Stream::Compute(reinterpret_cast<void*>(*stream)) : Stream::None();
}

struct TakenTensor {
    DLManagedTensorVersioned* managed = nullptr;
    Stream producer = Stream::None();
};

TakenTensor TakeTensor(nanobind::handle source, std::optional<uintptr_t> stream);

inline std::shared_ptr<void> Hold(DLManagedTensorVersioned* managed) {
    return std::shared_ptr<void>(managed, [](void* held) {
        auto* tensor = static_cast<DLManagedTensorVersioned*>(held);
        if (tensor->deleter) {
            tensor->deleter(tensor);
        }
    });
}

class HeldTensors {
  public:
    const DLTensor* Take(nanobind::handle value, std::optional<uintptr_t> stream) {
        if (value.is_none()) {
            return nullptr;
        }
        return &Keep(value, stream).managed->dl_tensor;
    }

    device::Tensor Import(device::Device& device, nanobind::handle value,
                          std::optional<uintptr_t> stream, std::string label) {
        const TakenTensor taken = Keep(value, stream);
        return device.Import(taken.managed->dl_tensor, taken.producer, std::move(label));
    }

    Stream Producer() const {
        return m_producer;
    }

    void Retire(Scene& scene, const device::Operation& done) {
        scene.GetDevice().GetQueue().Retire(std::move(m_held), done);
    }

  private:
    TakenTensor Keep(nanobind::handle value, std::optional<uintptr_t> stream) {
        const TakenTensor taken = TakeTensor(value, stream);
        m_held->push_back(Hold(taken.managed));
        if (!m_taken) {
            m_producer = taken.producer;
            m_taken = true;
        }
        return taken;
    }

    std::shared_ptr<std::vector<std::shared_ptr<void>>> m_held =
        std::make_shared<std::vector<std::shared_ptr<void>>>();
    Stream m_producer = Stream::None();
    bool m_taken = false;
};

void BindDevice(nanobind::module_& module);
void BindScene(nanobind::module_& module);
void BindRender(nanobind::module_& module);
void BindImport(nanobind::module_& module);
void BindIo(nanobind::module_& module);
void BindSensorPipeline(nanobind::module_& module);
void BindRotation(nanobind::module_& module);

}

namespace nanobind::detail {

template <class T> struct type_caster<ernie::Ref<T>> {
    using Caster = make_caster<T>;
    NB_TYPE_CASTER(ernie::Ref<T>, Caster::Name)

    bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept {
        Caster caster;
        if (!caster.from_python(src, flags_for_local_caster<T>(flags), cleanup)) {
            return false;
        }
        value = ernie::Ref<T>(caster.operator T*());
        return true;
    }

    static handle from_cpp(const ernie::Ref<T>& ref, rv_policy, cleanup_list* cleanup) noexcept {
        if (!ref) {
            return none().release();
        }
        if (object existing = find(*ref.Get())) {
            return existing.release();
        }
        const handle created = Caster::from_cpp(ref.Get(), rv_policy::reference, cleanup);
        if (created.is_valid()) {
            ref->AddRef();
            keep_alive_cb(created, ref.Get(),
                          [](void* held) noexcept { static_cast<T*>(held)->Release(); });
        }
        return created;
    }
};

}
