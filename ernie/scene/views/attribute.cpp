// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view_state.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

#include <fmt/format.h>

#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/attributes.h"
#include "ernie/scene/structure/packing.h"
#include "ernie/scene/views/layout.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

DLTensor Describe(const device::Tensor& tensor) {
    const device::DType type = tensor.Type();
    DLTensor described{};
    described.ndim = int32_t(tensor.Shape().size());
    described.dtype = {
        uint8_t(type == device::DType::Float32 || type == device::DType::Float64 ? kDLFloat
                : type == device::DType::Int32 || type == device::DType::Int64   ? kDLInt
                                                                                 : kDLUInt),
        uint8_t(device::ElementBytes(type) * 8), 1};
    described.shape = const_cast<int64_t*>(tensor.Shape().data());
    described.strides =
        tensor.Strides().empty() ? nullptr : const_cast<int64_t*>(tensor.Strides().data());
    return described;
}

bool IsByteBool(const DLTensor& tensor) {
    return tensor.ndim == 1 && tensor.dtype.bits == 8 && tensor.dtype.lanes == 1 &&
           (tensor.dtype.code == kDLBool || tensor.dtype.code == kDLUInt ||
            tensor.dtype.code == kDLInt);
}

bool Selected(std::span<const uint64_t> mask, uint32_t index) {
    return mask.empty() || (mask[index / 64] >> (index % 64) & 1) != 0;
}

std::vector<int64_t> Shape(uint32_t count, const ColumnLayout& layout) {
    std::vector<int64_t> shape = {int64_t(count)};
    if (layout.lanes > 1) {
        shape.push_back(layout.lanes);
    }
    return shape;
}

}

UntypedAttribute::UntypedAttribute(std::shared_ptr<ViewState> view, ColumnRef column)
    : m_view(std::move(view)), m_column(column) {
    if (!m_column) {
        throw Error("An attribute of no column");
    }
    if (m_view->Schema() != SchemaId::None && !IsA(m_view->Schema(), m_column.schema) &&
        !IsApi(m_column.schema)) {
        throw Error("A view of {} has no {}'s {}", Info(m_view->Schema()).name,
                    Info(m_column.schema).name, m_column.column->field);
    }
}

void UntypedAttribute::CheckValues(size_t count) const {
    if (count != Size()) {
        throw Error("{} values for an attribute of {} prims", count, Size());
    }
}

ColumnRef UntypedAttribute::Checked(ColumnRef column, bool held, bool arrays) {
    if (!held) {
        throw Error("{}'s {} isn't {} the type asked for", Info(column.schema).name,
                    column.column->field, arrays ? "an array of" : "held as");
    }
    return column;
}

uint32_t UntypedAttribute::Size() const {
    return m_view->Size();
}

const ViewRows& UntypedAttribute::Rows() const {
    return m_view->Rows(m_column.schema);
}

bool UntypedAttribute::Dense() const {
    return Rows().Dense();
}

Store& UntypedAttribute::GetStore() const {
    return m_view->GetStore();
}

Table& UntypedAttribute::GetTable() const {
    return GetStore().Schema(m_column.schema);
}

uint32_t UntypedAttribute::CellBytes() const {
    return ElementBytes(m_column.column->type);
}

void UntypedAttribute::CheckWritable(bool arrays) const {
    if (Has(m_column.column->flags, ColumnFlags::Internal)) {
        throw Error("{}'s {} is internal; the catalog writes it", Info(m_column.schema).name,
                    m_column.column->field);
    }
    if (!arrays && IsArray(m_column.column->type)) {
        throw UnsupportedError("{}'s {} holds arrays, which are written per prim",
                               Info(m_column.schema).name, m_column.column->field);
    }
}

void UntypedAttribute::CheckNotAttached() const {
    if (GetStore().AttachmentCount() > 0 && GetStore().Attached(m_column, Rows().rows)) {
        throw Error("{}'s {} is attached; write the attached tensor and touch the attribute",
                    Info(m_column.schema).name, m_column.column->field);
    }
}

void UntypedAttribute::StampWorlds() const {
    for (const uint32_t world : Rows().worlds) {
        GetStore().StampWorld(world);
    }
}

void UntypedAttribute::Fill(Span<const std::byte> cell) {
    CheckWritable();
    CheckNotAttached();
    if (cell.size() != CellBytes()) {
        throw Error("{} bytes for {}, whose cells are {}", cell.size(), m_column.column->field,
                    CellBytes());
    }
    const ViewRows& rows = Rows();
    for (const ViewBlock& block : rows.blocks) {
        GetStore().Fill(m_column, block.firstRow, block.count, cell);
    }
    StampWorlds();
}

void UntypedAttribute::Write(Span<const std::byte> cells) {
    ERNIE_ZONE("Scene/Write");
    CheckWritable();
    CheckNotAttached();
    StageCells(cells);
}

void UntypedAttribute::StageCells(Span<const std::byte> cells) {
    const uint32_t cell = CellBytes();
    if (cells.size() != size_t(cell) * Size()) {
        throw Error("{} bytes for {} prims' {}, whose cells are {}", cells.size(), Size(),
                    m_column.column->field, cell);
    }
    const ViewRows& rows = Rows();
    for (const ViewBlock& block : rows.blocks) {
        GetStore().Stage(m_column, block.firstRow,
                         cells.subspan(size_t(block.first) * cell, size_t(block.count) * cell));
    }
    StampWorlds();
}

device::Operation UntypedAttribute::Write(const DLTensor& source, Stream producer,
                                          const DLTensor* mask) {
    ERNIE_ZONE("Scene/Write");
    CheckWritable();
    CheckNotAttached();
    return WriteTensor(source, producer, mask);
}

device::Operation UntypedAttribute::WriteTensor(const DLTensor& source, Stream producer,
                                                const DLTensor* mask) {
    if (Size() == 0) {
        return {};
    }
    const ColumnInfo& column = *m_column.column;
    const std::byte* data = TensorBytes(source);
    if (OnHost(source.device)) {
        WaitOnHost(GetStore().GetDevice(), producer);
        std::vector<uint64_t> words;
        if (mask) {
            if (!OnHost(mask->device)) {
                throw UnsupportedError("A host tensor is written with a host mask");
            }
            const uint64_t* bits = reinterpret_cast<const uint64_t*>(TensorBytes(*mask));
            words.assign(bits, bits + (Size() + 63) / 64);
        }
        std::vector<std::byte> cells;
        if (column.type == ColumnType::Bool && IsByteBool(source) && source.shape[0] == Size()) {
            std::vector<uint32_t> words32(Size());
            for (uint32_t i = 0; i < Size(); ++i) {
                words32[i] = data[i] != std::byte{0} ? 1 : 0;
            }
            const std::span<const std::byte> bytes = std::as_bytes(std::span(words32));
            cells.assign(bytes.begin(), bytes.end());
        } else {
            if (TensorRows(source, Layout(column.type), column.field) != Size()) {
                throw Error("A tensor of {} rows for {} prims' {}", source.shape[0], Size(),
                            column.field);
            }
            cells.assign(data, data + size_t(CellBytes()) * Size());
        }
        if (words.empty()) {
            StageCells(cells);
            return {};
        }
        const uint32_t cell = CellBytes();
        const ViewRows& rows = Rows();
        for (uint32_t i = 0; i < Size(); ++i) {
            if (rows.rows[i] != kNoRow && Selected(words, i)) {
                GetStore().Stage(m_column, rows.rows[i],
                                 std::span(cells).subspan(size_t(i) * cell, cell));
            }
        }
        StampWorlds();
        return {};
    }

    if (TensorRows(source, Layout(column.type), column.field) != Size()) {
        throw Error("A tensor of {} rows for {} prims' {}", source.shape[0], Size(), column.field);
    }
    device::Device& device = GetStore().GetDevice();
    CheckReadable(device, source.device);
    uint64_t cells = uint64_t(data);
    Ref<device::Buffer> staging;
    if (!device.ReadsDirectly(source.device)) {
        staging =
            device.Stage(cells, uint64_t(CellBytes()) * Size(), producer, "attribute staging");
        cells = staging->Address();
    }
    return Scatter(cells, producer, Upload(mask, producer));
}

device::Operation UntypedAttribute::Write(const device::Tensor& source, Stream producer,
                                          const device::Tensor& mask) {
    ERNIE_ZONE("Scene/Write");
    CheckWritable();
    CheckNotAttached();
    if (Size() == 0) {
        return {};
    }
    const ColumnInfo& column = *m_column.column;
    device::Device& device = GetStore().GetDevice();
    if (source.Empty() || &source.GetDevice() != &device) {
        throw Error("{} is written from tensors on the scene's {}; Device::Copy or "
                    "Device::Import moves them",
                    column.field, device.Adapter().alias);
    }
    if (TensorRows(Describe(source), Layout(column.type), column.field) != Size()) {
        throw Error("A tensor of {} rows for {} prims' {}", source.Shape()[0], Size(),
                    column.field);
    }
    return Scatter(source.Address(), producer, Upload(mask));
}

UntypedAttribute::Mask UntypedAttribute::Upload(const DLTensor* mask, Stream producer) const {
    if (!mask) {
        return {};
    }
    const uint64_t words = (uint64_t(Size()) + 63) / 64;
    if (mask->ndim != 1 || mask->dtype.code != kDLUInt || mask->dtype.bits != 64 ||
        uint64_t(mask->shape[0]) < words) {
        throw Error("A mask is a [words] tensor of uint64 with a bit per prim; {} prims take {} "
                    "words",
                    Size(), words);
    }
    device::Device& device = GetStore().GetDevice();
    const std::byte* bits = TensorBytes(*mask);
    Mask uploaded;
    if (OnHost(mask->device)) {
        WaitOnHost(device, producer);
        uploaded.buffer = device.CreateBuffer(
            {.label = "attribute mask"},
            std::span<const uint64_t>(reinterpret_cast<const uint64_t*>(bits), words));
    } else if (device.ReadsDirectly(mask->device)) {
        uploaded.address = uint64_t(bits);
        return uploaded;
    } else {
        CheckReadable(device, mask->device);
        uploaded.buffer =
            device.Stage(uint64_t(bits), words * sizeof(uint64_t), producer, "attribute mask");
    }
    uploaded.address = uploaded.buffer->Address();
    return uploaded;
}

UntypedAttribute::Mask UntypedAttribute::Upload(const device::Tensor& mask) const {
    if (mask.Empty()) {
        return {};
    }
    const uint64_t words = (uint64_t(Size()) + 63) / 64;
    if (mask.Shape().size() != 1 || mask.Type() != device::DType::UInt64 ||
        uint64_t(mask.Shape()[0]) < words ||
        (!mask.Strides().empty() && mask.Shape()[0] > 1 && mask.Strides()[0] != 1)) {
        throw Error("A mask is a contiguous [words] tensor of uint64 with a bit per prim; {} prims "
                    "take {} words",
                    Size(), words);
    }
    device::Device& device = GetStore().GetDevice();
    if (&mask.GetDevice() != &device) {
        throw Error("A mask on {} for a scene on {}; Device::Copy or Device::Import moves it",
                    mask.GetDevice().Adapter().alias, device.Adapter().alias);
    }
    return {mask.Address(), {}};
}

device::Operation UntypedAttribute::Scatter(uint64_t cells, Stream producer, const Mask& selected) {
    Store& store = GetStore();
    store.Flush();
    store.GetDevice().GetQueue().Follow(producer);
    CellsParams params{};
    params.target = store.Stamps(m_column);
    params.column.address = GetTable().Pages(m_column.index);
    params.cells.address = cells;
    const RowSource rowSource = m_view->Source(m_column.schema);
    params.rows.address = rowSource.rows;
    params.firstRow = rowSource.firstRow;
    params.mask.address = selected.address;
    params.count = Size();
    params.words = CellBytes() / sizeof(uint32_t);
    store.Touch(m_column);
    StampWorlds();
    const device::Operation scattered =
        store.Dispatch("ScatterCells", params, uint64_t(params.count) * params.words);
    store.Reassert(m_column, Rows().rows);
    return scattered;
}

device::Operation UntypedAttribute::Stamp(Stream producer, const DLTensor* mask) {
    Store& store = GetStore();
    store.GetDevice().GetQueue().Follow(producer);
    const Mask selected = Upload(mask, producer);
    StampParams params{};
    params.target = store.Stamps(m_column);
    const RowSource rowSource = m_view->Source(m_column.schema);
    params.rows.address = rowSource.rows;
    params.firstRow = rowSource.firstRow;
    params.mask.address = selected.address;
    params.count = Size();
    store.Touch(m_column);
    StampWorlds();
    if (IsArray(m_column.column->type)) {
        store.LogArrayRows(m_column, Rows().rows);
    }
    const device::Operation stamped = store.Dispatch("StampRows", params, params.count);
    store.Reassert(m_column, Rows().rows);
    return stamped;
}

device::Operation UntypedAttribute::MarkChanged(Stream producer, const DLTensor* mask) {
    CheckWritable(true);
    if (Size() == 0) {
        return {};
    }
    if (GetStore().AttachmentCount() > 0) {
        const Attachment* attachment = GetStore().FindAttachment(m_column, Rows().rows);
        if (attachment && !attachment->inPlace) {
            const DLTensor& source = *attachment->tensor;
            return IsArray(m_column.column->type) ? CopyArrays(source, producer, mask)
                                                  : WriteTensor(source, producer, mask);
        }
    }
    return Stamp(producer, mask);
}

device::Operation UntypedAttribute::AttachArrays(const Attachment& attachment,
                                                 const DLTensor& source, Stream producer,
                                                 AttachFrom from) {
    const ColumnInfo& column = *m_column.column;
    const uint32_t count = TensorItems(source, Layout(column.type), Size(), column.field);
    if (column.type == ColumnType::PrimArray) {
        m_view->GetCatalog().MarkUntrackedTargets();
    }
    Store& store = GetStore();
    store.Flush();
    ArrayHeap& heap = store.Arrays(column.type);
    device::Device& device = store.GetDevice();
    device::Queue& queue = device.GetQueue();
    const uint64_t bytes = uint64_t(count) * heap.ItemBytes();
    std::vector<ArrayRef> old;
    for (const uint32_t row : attachment.rows) {
        old.push_back(store.Array(m_column, row));
        if (from == AttachFrom::Scene && old.back().count != count) {
            throw Error("{}'s {} attaches {} items per prim from prims whose arrays hold {}",
                        Info(m_column.schema).name, column.field, count, old.back().count);
        }
    }
    if (!attachment.inPlace && !OnHost(source.device)) {
        if (from == AttachFrom::Scene) {
            throw UnsupportedError("{} copies scene values into host tensors and ones it reads "
                                   "directly, not into memory of device type {}",
                                   device.Adapter().alias, int(source.device.device_type));
        }
        CheckReadable(device, source.device);
    }
    std::vector<ArrayRef> fresh;
    if (attachment.inPlace) {
        queue.Follow(producer);
        fresh = heap.Borrow(device.Borrow(source.device, uint64_t(TensorBytes(source)),
                                          bytes * Size(), fmt::format("attached {}", column.field)),
                            Size(), count);
    } else {
        fresh = heap.AllocateRun(std::vector<uint32_t>(Size(), count));
    }
    if (from == AttachFrom::Scene) {
        device::CommandEncoder encoder = queue.CreateEncoder();
        for (uint32_t i = 0; i < Size(); ++i) {
            encoder.Copy(heap.Range(old[i]).Slice(0, bytes), heap.Range(fresh[i]).Slice(0, bytes));
        }
        queue.Submit(encoder.Finish());
    }
    for (uint32_t i = 0; i < Size(); ++i) {
        store.SetArray(m_column, attachment.rows[i], fresh[i]);
    }
    store.Flush();
    if (from == AttachFrom::Scene) {
        if (!attachment.inPlace) {
            WaitOnHost(device, producer);
            const device::BufferRange first = heap.Range(fresh.front());
            queue.Read(first.GetBuffer().Slice(first.Offset(), bytes * Size()),
                       const_cast<std::byte*>(TensorBytes(source)));
        }
        return queue.Mark();
    }
    if (attachment.inPlace) {
        return Stamp(Stream::None(), nullptr);
    }
    return CopyArrays(source, producer, nullptr);
}

device::Operation UntypedAttribute::CopyArrays(const DLTensor& source, Stream producer,
                                               const DLTensor* mask) {
    Store& store = GetStore();
    store.Flush();
    ArrayHeap& heap = store.Arrays(m_column.column->type);
    const ArrayRef first = store.Array(m_column, Rows().rows.front());
    const device::BufferRange range = heap.Range(first);
    const device::BufferRange run =
        range.GetBuffer().Slice(range.Offset(), uint64_t(first.count) * heap.ItemBytes() * Size());
    if (OnHost(source.device)) {
        WaitOnHost(store.GetDevice(), producer);
        store.GetDevice().GetQueue().Write(run, TensorBytes(source));
    } else {
        store.GetDevice().CopyIn(run, uint64_t(TensorBytes(source)), producer);
    }
    return Stamp(producer, mask);
}

void UntypedAttribute::ReleaseArrays() {
    Store& store = GetStore();
    store.Flush();
    ArrayHeap& heap = store.Arrays(m_column.column->type);
    const std::vector<uint32_t>& rows = Rows().rows;
    const ArrayRef first = store.Array(m_column, rows.front());
    const std::vector<ArrayRef> fresh =
        heap.AllocateRun(std::vector<uint32_t>(rows.size(), first.count));
    const uint64_t bytes = uint64_t(first.count) * heap.ItemBytes() * rows.size();
    const device::BufferRange from = heap.Range(first);
    const device::BufferRange to = heap.Range(fresh.front());
    device::Queue& queue = store.GetDevice().GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Copy(from.GetBuffer().Slice(from.Offset(), bytes),
                 to.GetBuffer().Slice(to.Offset(), bytes));
    queue.Submit(encoder.Finish());
    for (uint32_t i = 0; i < rows.size(); ++i) {
        store.SetArray(m_column, rows[i], fresh[i]);
    }
    store.Flush();
}

device::Operation UntypedAttribute::WaitForReads(Stream consumer) const {
    if (!Attached()) {
        throw Error("{}'s {} isn't attached, so nothing of it waits", Info(m_column.schema).name,
                    m_column.column->field);
    }
    const device::Operation read = GetStore().GetDevice().GetQueue().Mark();
    read.Wait(consumer);
    return read;
}

bool UntypedAttribute::Attached() const {
    return GetStore().AttachmentCount() > 0 && GetStore().FindAttachment(m_column, Rows().rows);
}

device::Operation UntypedAttribute::Attach(DLManagedTensor* tensor, Stream producer,
                                           AttachFrom from) {
    return AttachTensor(tensor ? &tensor->dl_tensor : nullptr, producer, from, [tensor] {
        return std::shared_ptr<const DLTensor>(&tensor->dl_tensor, [tensor](const DLTensor*) {
            if (tensor->deleter) {
                tensor->deleter(tensor);
            }
        });
    });
}

device::Operation UntypedAttribute::Attach(DLManagedTensorVersioned* tensor, Stream producer,
                                           AttachFrom from) {
    if (tensor && (tensor->flags & DLPACK_FLAG_BITMASK_IS_COPIED)) {
        throw Error("{}'s {} attaches a copy, so later writes to the tensor wouldn't reach the "
                    "scene; export it without copying",
                    Info(m_column.schema).name, m_column.column->field);
    }
    if (tensor && from == AttachFrom::Scene && (tensor->flags & DLPACK_FLAG_BITMASK_READ_ONLY)) {
        throw Error("A read-only tensor can't take {}'s scene values of {}",
                    Info(m_column.schema).name, m_column.column->field);
    }
    return AttachTensor(tensor ? &tensor->dl_tensor : nullptr, producer, from, [tensor] {
        return std::shared_ptr<const DLTensor>(&tensor->dl_tensor, [tensor](const DLTensor*) {
            if (tensor->deleter) {
                tensor->deleter(tensor);
            }
        });
    });
}

device::Operation UntypedAttribute::AttachTensor(const DLTensor* tensor, Stream producer,
                                                 AttachFrom from, const Owner& own) {
    CheckWritable(true);
    const ColumnInfo& column = *m_column.column;
    if (tensor == nullptr) {
        throw Error("{} attaches no tensor", column.field);
    }
    if (Size() == 0) {
        throw Error("An empty view attaches no {}", column.field);
    }
    const ViewRows& rows = Rows();
    if (rows.live != Size()) {
        throw Error("A view with removed prims can't attach {}", column.field);
    }
    Store& store = GetStore();
    if (store.Attached(m_column, rows.rows)) {
        throw Error("{}'s {} is already attached for some of these prims",
                    Info(m_column.schema).name, column.field);
    }
    const DLTensor& source = *tensor;
    device::Device& device = store.GetDevice();
    device::Queue& queue = device.GetQueue();
    Attachment attachment{m_column, rows.rows, {}, device.ReadsDirectly(source.device), {}};
    device::Operation ready;
    if (IsArray(column.type)) {
        ready = AttachArrays(attachment, source, producer, from);
    } else if (TensorRows(source, Layout(column.type), column.field) != Size()) {
        throw Error("A tensor of {} rows attached for {} prims' {}", source.shape[0], Size(),
                    column.field);
    } else if (attachment.inPlace) {
        if (!Dense()) {
            throw Error("{}'s {} is attached in place only for prims allocated together; define "
                        "them with BatchDesc::attachable or clone them with CloneDesc::attachable",
                        Info(m_column.schema).name, column.field);
        }
        const uint32_t first = rows.blocks.front().firstRow;
        const uint64_t bytes = uint64_t(CellBytes()) * Size();
        Ref<device::Buffer> borrowed =
            device.Borrow(source.device, uint64_t(TensorBytes(source)), bytes,
                          fmt::format("attached {}", column.field));
        Table& table = GetTable();
        table.Flush();
        queue.Follow(producer);
        if (from == AttachFrom::Scene) {
            device::CommandEncoder encoder = queue.CreateEncoder();
            encoder.Copy(*table.Contiguous(m_column.index, first, Size()),
                         borrowed->Slice(0, bytes));
            queue.Submit(encoder.Finish());
        }
        try {
            table.Redirect(m_column.index, first, Size(), borrowed);
        } catch (const Error& error) {
            throw Error("{}'s {} can't be attached in place: {}. Define the prims with "
                        "BatchDesc::attachable or clone them with CloneDesc::attachable",
                        Info(m_column.schema).name, column.field, error.what());
        }
        ready = from == AttachFrom::Tensor ? Stamp(Stream::None(), nullptr) : queue.Mark();
    } else if (from == AttachFrom::Scene) {
        if (!OnHost(source.device)) {
            throw UnsupportedError("{} copies scene values into host tensors and ones it reads "
                                   "directly, not into memory of device type {}",
                                   device.Adapter().alias, int(source.device.device_type));
        }
        const std::vector<std::byte> cells = ReadHost();
        WaitOnHost(device, producer);
        std::copy(cells.begin(), cells.end(), const_cast<std::byte*>(TensorBytes(source)));
    } else {
        if (!OnHost(source.device)) {
            CheckReadable(device, source.device);
        }
        ready = WriteTensor(source, producer, nullptr);
    }
    attachment.tensor = own();
    store.Attach(std::move(attachment));
    if (from == AttachFrom::Scene) {
        ready.Wait(producer);
    }
    return ready;
}

device::Operation UntypedAttribute::Detach(Stream consumer) {
    Store& store = GetStore();
    const ViewRows& rows = Rows();
    Attachment attachment = store.Detach(m_column, rows.rows);
    device::Queue& queue = store.GetDevice().GetQueue();
    if (attachment.inPlace && IsArray(m_column.column->type)) {
        ReleaseArrays();
    } else if (attachment.inPlace) {
        Table& table = GetTable();
        const uint32_t first = rows.blocks.front().firstRow;
        const Ref<device::Buffer> borrowed = table.Restore(m_column.index, first);
        const device::BufferRange owned = *table.Contiguous(m_column.index, first, Size());
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(borrowed->Slice(0, owned.Bytes()), owned);
        queue.Submit(encoder.Finish());
    }
    const device::Operation ready = queue.Mark();
    if (consumer.IsCompute()) {
        ready.Wait(consumer);
        queue.Retire(std::const_pointer_cast<DLTensor>(std::move(attachment.tensor)), ready);
    } else {
        ready.Wait();
        attachment.tensor.reset();
    }
    return ready;
}

Ref<device::Buffer> UntypedAttribute::GatherCells() const {
    Store& store = GetStore();
    store.Flush();
    device::Device& device = store.GetDevice();
    const Ref<device::Buffer> cells =
        device.CreateBuffer({.bytes = std::max<uint64_t>(uint64_t(CellBytes()) * Size(), 4),
                             .usage = device::BufferUsage::Interop,
                             .label = "gathered attribute"});
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Fill(*cells);
    queue.Submit(encoder.Finish());
    CellsParams params{};
    params.column.address = GetTable().Pages(m_column.index);
    params.cells.address = cells->Address();
    const RowSource rowSource = m_view->Source(m_column.schema);
    params.rows.address = rowSource.rows;
    params.firstRow = rowSource.firstRow;
    params.count = Size();
    params.words = CellBytes() / sizeof(uint32_t);
    store.Dispatch("GatherCells", params, uint64_t(params.count) * params.words);
    return cells;
}

device::Download UntypedAttribute::ReadHostAsync() const {
    if (Size() == 0) {
        return {};
    }
    Store& store = GetStore();
    device::Queue& queue = store.GetDevice().GetQueue();
    if (Dense()) {
        store.Flush();
        return queue.ReadAsync(
            *GetTable().Contiguous(m_column.index, Rows().blocks.front().firstRow, Size()));
    }
    const Ref<device::Buffer> cells = GatherCells();
    return queue.ReadAsync(cells->Slice(0, uint64_t(CellBytes()) * Size()));
}

std::vector<std::byte> UntypedAttribute::ReadHost() const {
    ERNIE_ZONE("Scene/Read");
    const device::Download download = ReadHostAsync();
    download.Wait();
    const std::span<const std::byte> bytes = download.Bytes();
    return {bytes.begin(), bytes.end()};
}

device::Tensor UntypedAttribute::Read(Stream consumer) const {
    ERNIE_ZONE("Scene/Read");
    const Ref<device::Buffer> cells = GatherCells();
    GetStore().GetDevice().GetQueue().Mark().Wait(consumer);
    return device::Tensor(cells, 0, Layout(m_column.column->type).type,
                          Shape(Size(), Layout(m_column.column->type)));
}

void UntypedAttribute::SetArray(uint32_t index, Span<const std::byte> items) {
    ERNIE_ZONE("Scene/Write");
    CheckWritable(true);
    CheckNotAttached();
    if (!IsArray(m_column.column->type)) {
        throw Error("{}'s {} holds no arrays", Info(m_column.schema).name, m_column.column->field);
    }
    const uint32_t row = Rows().rows.at(index);
    if (row == kNoRow) {
        throw Error("Prim {} of the view was removed", index);
    }
    Store& store = GetStore();
    ArrayHeap& arrays = store.Arrays(m_column.column->type);
    if (items.size() % arrays.ItemBytes() != 0) {
        throw Error("{} bytes for items of {} bytes", items.size(), arrays.ItemBytes());
    }
    const uint32_t count = uint32_t(items.size() / arrays.ItemBytes());
    const ArrayRef current = store.Array(m_column, row);
    const bool reused = current.address != 0 && current.count == count &&
                        arrays.References(current) == 1 && !arrays.Borrowed(current);
    const ArrayRef array = reused ? current : arrays.Allocate(count);
    arrays.Stage(array, items);
    store.SetArray(m_column, row, array);
    if (m_column.column->type == ColumnType::PrimArray) {
        std::vector<uint32_t> targets(count);
        std::memcpy(targets.data(), items.data(), items.size());
        m_view->GetCatalog().MarkTargets(targets);
    }
    store.StampWorld(m_view->GetCatalog().GetPrims().At(m_view->Prims()[index]).world);
}

std::vector<std::byte> UntypedAttribute::GetArray(uint32_t index) const {
    ERNIE_ZONE("Scene/Read");
    const uint32_t row = Rows().rows.at(index);
    if (row == kNoRow) {
        throw Error("Prim {} of the view was removed", index);
    }
    Store& store = GetStore();
    store.Flush();
    const ArrayRef array = store.Array(m_column, row);
    return array.address == 0 ? std::vector<std::byte>{}
                              : store.Arrays(m_column.column->type).Read(array);
}

void UntypedAttribute::CheckArrayType(ColumnType type) const {
    if (m_column.column->type != type) {
        throw Error("{}'s {} doesn't hold that kind of array", Info(m_column.schema).name,
                    m_column.column->field);
    }
}

void UntypedAttribute::SetTargets(uint32_t index, const std::vector<std::string>& paths) {
    CheckWritable(true);
    CheckNotAttached();
    CheckArrayType(ColumnType::PrimArray);
    if (Rows().rows.at(index) == kNoRow) {
        throw Error("Prim {} of the view was removed", index);
    }
    Catalog& catalog = m_view->GetCatalog();
    const uint32_t prim = m_view->Prims()[index];
    catalog.SetTargets(prim, Rows().rows.at(index), m_column, paths);
    GetStore().StampWorld(catalog.GetPrims().At(prim).world);
}

std::vector<std::string> UntypedAttribute::GetTargets(uint32_t index) const {
    CheckArrayType(ColumnType::PrimArray);
    const Catalog& catalog = m_view->GetCatalog();
    std::vector<std::string> paths;
    for (const uint32_t prim : Items<uint32_t>(GetArray(index))) {
        if (prim != kNoPrim && catalog.GetPrims().Alive(prim)) {
            paths.push_back(catalog.GetPaths().Path(prim));
        }
    }
    return paths;
}

void UntypedAttribute::SetTokens(uint32_t index, const std::vector<std::string>& texts) {
    CheckArrayType(ColumnType::TokenArray);
    const std::vector<uint32_t> ids = InternTokens(m_view->GetCatalog().GetTokens(), texts);
    SetArray(index, AsBytes(Span<const uint32_t>(ids)));
}

std::vector<std::string> UntypedAttribute::GetTokens(uint32_t index) const {
    CheckArrayType(ColumnType::TokenArray);
    const Tokens& tokens = m_view->GetCatalog().GetTokens();
    std::vector<std::string> texts;
    for (const uint32_t id : Items<uint32_t>(GetArray(index))) {
        texts.emplace_back(tokens.Text(Token{id}));
    }
    return texts;
}

void UntypedAttribute::SetPrimvars(uint32_t index, const std::vector<Primvar>& primvars) {
    if (m_column.column->field != "primvars") {
        throw Error("{}'s {} holds no primvars", Info(m_column.schema).name,
                    m_column.column->field);
    }
    const std::vector<int32_t> table = PackPrimvars(m_view->GetCatalog().GetTokens(), primvars);
    SetArray(index, AsBytes(Span<const int32_t>(table)));
}

std::vector<Primvar> UntypedAttribute::GetPrimvars(uint32_t index) const {
    if (m_column.column->field != "primvars") {
        throw Error("{}'s {} holds no primvars", Info(m_column.schema).name,
                    m_column.column->field);
    }
    return UnpackPrimvars(m_view->GetCatalog().GetTokens(), Items<int32_t>(GetArray(index)));
}

void UntypedAttribute::MapArray(Mapping& mapping) {
    const ViewRows& rows = Rows();
    Store& store = GetStore();
    store.Flush();
    ArrayHeap& heap = store.Arrays(m_column.column->type);
    const ColumnLayout layout = Layout(m_column.column->type);
    const bool writes = mapping.m_access != MapAccess::Read;
    if (writes && m_column.column->type == ColumnType::PrimArray) {
        m_view->GetCatalog().MarkUntrackedTargets();
    }
    std::vector<ArrayRef> arrays;
    std::vector<uint32_t> offsets = {0};
    for (const uint32_t row : rows.rows) {
        if (row == kNoRow) {
            throw Error("A mapping of arrays of removed prims");
        }
        arrays.push_back(store.Array(m_column, row));
        offsets.push_back(offsets.back() + arrays.back().count);
    }
    const uint64_t item = heap.ItemBytes();
    const uint32_t count = arrays.empty() ? 0 : arrays.front().count;
    const bool equal = count > 0 && std::ranges::all_of(arrays, [&](const ArrayRef& array) {
                           return array.count == count;
                       });
    const auto first = std::ranges::find_if(arrays, [](const ArrayRef& a) { return a.count > 0; });
    uint64_t stride = uint64_t(count) * item;
    if (equal && arrays.size() > 1 && arrays[1].address > arrays[0].address) {
        stride = arrays[1].address - arrays[0].address;
    }
    bool run = first != arrays.end() && stride % device::ElementBytes(layout.type) == 0;
    if (run) {
        const device::BufferRange range = heap.Range(*first);
        const device::Buffer& buffer = range.GetBuffer();
        uint64_t expected = first->address;
        for (const ArrayRef& array : arrays) {
            if (array.count == 0) {
                continue;
            }
            run = run && array.address == expected && &heap.Range(array).GetBuffer() == &buffer &&
                  (!writes || heap.References(array) == 1);
            expected += equal ? stride : uint64_t(array.count) * item;
        }
    }

    device::Device& device = store.GetDevice();
    device::Queue& queue = device.GetQueue();
    Ref<device::Buffer> owner;
    uint64_t offset = 0;
    if (first == arrays.end()) {
        owner = device.CreateBuffer({.bytes = 4, .label = "empty array mapping"});
    } else if (run) {
        const device::BufferRange range = heap.Range(*first);
        owner = Ref<device::Buffer>(&range.GetBuffer());
        offset = range.Offset();
    } else {
        std::vector<uint32_t> counts(arrays.size());
        for (uint32_t i = 0; i < arrays.size(); ++i) {
            counts[i] = arrays[i].count;
        }
        std::vector<ArrayRef> fresh;
        if (writes) {
            fresh = heap.AllocateRun(counts);
        } else {
            owner = device.CreateBuffer({.bytes = uint64_t(offsets.back()) * item,
                                         .usage = device::BufferUsage::Interop,
                                         .label = "array mapping staging"});
        }
        device::CommandEncoder encoder = queue.CreateEncoder();
        for (uint32_t i = 0; i < arrays.size(); ++i) {
            if (arrays[i].count == 0) {
                continue;
            }
            const uint64_t bytes = uint64_t(arrays[i].count) * item;
            encoder.Copy(heap.Range(arrays[i]).Slice(0, bytes),
                         writes ? heap.Range(fresh[i]).Slice(0, bytes)
                                : owner->Slice(uint64_t(offsets[i]) * item, bytes));
        }
        queue.Submit(encoder.Finish());
        if (writes) {
            for (uint32_t i = 0; i < arrays.size(); ++i) {
                if (arrays[i].count > 0) {
                    store.SetArray(m_column, rows.rows[i], fresh[i]);
                }
            }
            store.Flush();
            const device::BufferRange range = heap.Range(fresh[first - arrays.begin()]);
            owner = Ref<device::Buffer>(&range.GetBuffer());
            offset = range.Offset();
        }
        stride = uint64_t(count) * item;
    }

    const uint64_t element = device::ElementBytes(layout.type);
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    if (equal) {
        shape = {int64_t(arrays.size()), int64_t(count)};
        strides = {int64_t(stride / element), int64_t(layout.lanes)};
    } else {
        shape = {int64_t(offsets.back())};
        strides = {int64_t(layout.lanes)};
        mapping.m_offsets = device::Tensor(device.CreateBuffer({.label = "array mapping offsets"},
                                                               std::span<const uint32_t>(offsets)),
                                           0, device::DType::UInt32, {int64_t(offsets.size())});
    }
    if (layout.lanes > 1) {
        shape.push_back(layout.lanes);
        strides.push_back(1);
    }
    mapping.m_values =
        device::Tensor(owner, offset, layout.type, std::move(shape), std::move(strides));
    mapping.m_aliased = run || writes;
    mapping.m_ready = queue.Mark();
}

Mapping UntypedAttribute::Map(MapAccess access, Stream consumer) {
    if (access != MapAccess::Read) {
        CheckWritable(true);
        const Attachment* attachment = GetStore().AttachmentCount() > 0
                                           ? GetStore().FindAttachment(m_column, Rows().rows)
                                           : nullptr;
        if (attachment == nullptr) {
            CheckNotAttached();
        } else if (!attachment->inPlace || IsArray(m_column.column->type)) {
            throw Error("{}'s {} is attached as arrays or by copy, so it doesn't map for writing; "
                        "write the attached tensor and touch the attribute",
                        Info(m_column.schema).name, m_column.column->field);
        }
    }
    Store& store = GetStore();
    store.Flush();
    const ColumnLayout layout = Layout(m_column.column->type);
    device::Queue& queue = store.GetDevice().GetQueue();
    Mapping mapping;
    mapping.m_view = m_view;
    mapping.m_column = m_column;
    mapping.m_access = access;
    if (IsArray(m_column.column->type)) {
        MapArray(mapping);
    } else if (Size() > 0 && Dense()) {
        const device::BufferRange range =
            *GetTable().Contiguous(m_column.index, Rows().blocks.front().firstRow, Size());
        mapping.m_values = device::Tensor(Ref<device::Buffer>(&range.GetBuffer()), range.Offset(),
                                          layout.type, Shape(Size(), layout));
        mapping.m_aliased = true;
        mapping.m_ready = queue.Mark();
    } else if (Size() > 0) {
        Ref<device::Buffer> staging;
        if (access == MapAccess::Write) {
            staging = store.GetDevice().CreateBuffer({.bytes = uint64_t(CellBytes()) * Size(),
                                                      .usage = device::BufferUsage::Interop,
                                                      .label = "mapping staging"});
        } else {
            staging = GatherCells();
        }
        mapping.m_values = device::Tensor(staging, 0, layout.type, Shape(Size(), layout));
        mapping.m_rows = device::Tensor(m_view->DeviceRows(m_column.schema), 0,
                                        device::DType::UInt32, {int64_t(Size())});
        mapping.m_ready = queue.Mark();
    }
    if (consumer.IsCompute()) {
        mapping.m_ready.Wait(consumer);
    }
    store.Pin(m_column.schema);
    mapping.m_open = true;
    return mapping;
}

}
