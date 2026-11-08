// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/commands.h"
#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/array_pool.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/columns/column_table.h"
#include "ernie/scene/columns/device_heap.h"
#include "ernie/scene/columns/kernels/cells.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/schema/columns.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class T> std::span<const std::byte> Bytes(const std::vector<T>& values) {
    return std::as_bytes(std::span(values));
}

template <class T> std::vector<T> Cells(std::span<const std::byte> bytes) {
    std::vector<T> cells(bytes.size() / sizeof(T));
    std::memcpy(cells.data(), bytes.data(), bytes.size());
    return cells;
}

std::vector<uint32_t> Sequence(uint32_t count, uint32_t first) {
    std::vector<uint32_t> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        values[i] = first + i;
    }
    return values;
}

void Dispatch(device::Device& device, const char* entry, const CellsParams& params) {
    const Ref<device::Program> program =
        device.CreateProgram({.modules = {"ernie.scene.columns.kernels.cells"},
                              .library = &kSceneKernels,
                              .label = "store test"});
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Dispatch(program->Entry(entry), params,
                     {.x = device::GroupCount(uint64_t(params.count) * params.words, kCellsGroup)});
    queue.Submit(encoder.Finish()).Wait();
}

std::vector<uint32_t> Gather(device::Device& device, uint64_t pages, std::span<const uint32_t> rows,
                             uint32_t words) {
    const Ref<device::Buffer> rowBuffer = device.CreateBuffer({.label = "rows"}, rows);
    const Ref<device::Buffer> cells =
        device.CreateBuffer({.bytes = rows.size() * words * sizeof(uint32_t), .label = "cells"});
    CellsParams params{};
    params.column.address = pages;
    params.cells.address = cells->Address();
    params.rows.address = rowBuffer->Address();
    params.count = uint32_t(rows.size());
    params.words = words;
    Dispatch(device, "GatherCells", params);
    std::vector<uint32_t> out(rows.size() * words);
    device.GetQueue().Read(*cells, out.data());
    return out;
}

TEST(DeviceHeapTest, FreedBlocksComeBackOnlyWhenAligned) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        const HeapBlock first = heap.Allocate(16, 16);
        const HeapBlock odd = heap.Allocate(256, 16);
        ASSERT_NE(odd.offset % 256, 0u);
        heap.Free(odd);
        const HeapBlock aligned = heap.Allocate(256, 256);
        EXPECT_EQ(aligned.offset % 256, 0u);
        EXPECT_NE(aligned.offset, odd.offset);
        const HeapBlock again = heap.Allocate(256, 16);
        EXPECT_EQ(again.offset, odd.offset);
        EXPECT_EQ(again.buffer, odd.buffer);
        heap.Free(first);
    });
}

TEST(DeviceHeapTest, ZeroBytesAllocateNothing) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        EXPECT_FALSE(heap.Allocate(0, 16));
        EXPECT_EQ(heap.Bytes(), 0u);
    });
}

TEST(TableTest, FreshRowsHoldTheirColumnsFill) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}, {4, 0xFFFFFFFF}, {8, 0}});
        table.Allocate(3);
        EXPECT_EQ(Cells<uint32_t>(table.Read(0, 0, 3)), (std::vector<uint32_t>{0, 0, 0}));
        EXPECT_EQ(Cells<uint32_t>(table.Read(1, 0, 3)),
                  (std::vector<uint32_t>{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}));
        EXPECT_EQ(Cells<uint64_t>(table.Read(2, 0, 3)), (std::vector<uint64_t>{0, 0, 0}));
    });
}

TEST(TableTest, StagedWritesLandOnFlushAndTheLastOneWins) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(4);
        table.Stage(0, 0, Bytes(std::vector<uint32_t>{1, 2, 3, 4}));
        table.Stage(0, 2, Bytes(std::vector<uint32_t>{30}));
        EXPECT_TRUE(table.Staged());
        table.Flush();
        EXPECT_FALSE(table.Staged());
        EXPECT_EQ(Cells<uint32_t>(table.Read(0, 0, 4)), (std::vector<uint32_t>{1, 2, 30, 4}));
    });
}

TEST(TableTest, GrowthNeverMovesRowsAlreadyThere) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}, {12, 0}});
        table.Allocate(10);
        table.Stage(0, 0, Bytes(Sequence(10, 100)));
        table.Flush();
        const device::BufferRange before = table.Ranges(0, 0, 10).front();
        const uint64_t address = before.Address();

        table.Allocate(5000);
        EXPECT_GE(table.Capacity(), 5010u);
        EXPECT_EQ(table.Ranges(0, 0, 10).front().Address(), address);
        EXPECT_EQ(Cells<uint32_t>(table.Read(0, 0, 10)), Sequence(10, 100));
    });
}

TEST(TableTest, ABatchIsContiguousEvenWhenItStartsANewSpan) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(200);
        const uint32_t first = table.Allocate(1000);
        ASSERT_TRUE(table.Contiguous(0, first, 1000));
        EXPECT_EQ(table.Contiguous(0, first, 1000)->Bytes(), 4000u);
        EXPECT_FALSE(table.Contiguous(0, 0, first + 1000));
        EXPECT_EQ(table.Ranges(0, 0, first + 1000).size(), 2u);
    });
}

TEST(TableTest, TailRowsLeftBehindByGrowthAreReused) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(250);
        const uint32_t batch = table.Allocate(10);
        EXPECT_GE(batch, kPageRows);
        const uint32_t reused = table.Acquire();
        EXPECT_GE(reused, 250u);
        EXPECT_LT(reused, kPageRows);
    });
}

TEST(TableTest, ReleasedRowsAreAcquiredAgain) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(4);
        table.Release(2);
        EXPECT_EQ(table.Acquire(), 2u);
        EXPECT_THROW(table.Release(9), Error);
    });
}

TEST(TableTest, WritesAcrossSpansSplitAtTheBoundary) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(300);
        table.Allocate(300);
        const uint32_t rows = table.Rows();
        table.Stage(0, 0, Bytes(Sequence(rows, 7)));
        table.Flush();
        EXPECT_EQ(Cells<uint32_t>(table.Read(0, 0, rows)), Sequence(rows, 7));
    });
}

TEST(TableTest, KernelsFindEveryRowThroughThePageTable) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}, {12, 0}});
        table.Allocate(100);
        table.Allocate(700);
        table.Allocate(3000);
        const uint32_t rows = table.Rows();
        std::vector<uint32_t> words(size_t(rows) * 3);
        for (uint32_t i = 0; i < words.size(); ++i) {
            words[i] = i * 3 + 1;
        }
        table.Stage(0, 0, Bytes(Sequence(rows, 0)));
        table.Stage(1, 0, Bytes(words));
        table.Flush();

        const std::vector<uint32_t> picks = {0, 1, 255, 256, 257, 799, 800, rows - 1, 513};
        std::vector<uint32_t> expected;
        for (const uint32_t row : picks) {
            expected.push_back(row);
        }
        EXPECT_EQ(Gather(device, table.Pages(0), picks, 1), expected);

        std::vector<uint32_t> wide;
        for (const uint32_t row : picks) {
            wide.insert(wide.end(), words.begin() + row * 3, words.begin() + row * 3 + 3);
        }
        EXPECT_EQ(Gather(device, table.Pages(1), picks, 3), wide);
    });
}

TEST(TableTest, AlignedBatchesStartAPageAndKeepTheirTail) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(10);
        const uint32_t first = table.Allocate(300, true);
        EXPECT_EQ(first, kPageRows);
        EXPECT_EQ(table.Allocate(1), 3 * kPageRows);
        EXPECT_LT(table.Acquire(), kPageRows);
        table.Release(first + 5);
        for (uint32_t i = 0; i < kPageRows; ++i) {
            const uint32_t row = table.Acquire();
            EXPECT_TRUE(row < first || row >= first + 300) << row;
        }
    });
}

TEST(TableTest, RedirectedRowsAreReadAndWrittenThroughTheirBuffer) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(5);
        const uint32_t first = table.Allocate(300, true);
        table.Stage(0, 0, Bytes(Sequence(table.Rows(), 0)));
        table.Flush();
        EXPECT_THROW(table.Redirect(0, 0, 5, device.CreateBuffer({.bytes = 20})), Error);
        EXPECT_THROW(table.Redirect(0, first, 100, device.CreateBuffer({.bytes = 400})), Error);

        const Ref<device::Buffer> buffer = device.CreateBuffer({.bytes = 300 * 4});
        device.GetQueue().Write(*buffer, Sequence(300, 1000).data());
        const uint64_t layout = table.Layout();
        table.Redirect(0, first, 300, buffer);
        EXPECT_TRUE(table.Redirected(0, first + 299, 1));
        EXPECT_FALSE(table.Redirected(0, 0, 5));
        EXPECT_GT(table.Layout(), layout);
        const std::vector<uint32_t> picks = {4, first, first + 299};
        EXPECT_EQ(Gather(device, table.Pages(0), picks, 1), (std::vector<uint32_t>{4, 1000, 1299}));

        table.Stage(0, first + 1, Bytes(std::vector<uint32_t>{7}));
        table.Flush();
        std::vector<uint32_t> held(300);
        device.GetQueue().Read(*buffer, held.data());
        EXPECT_EQ(held[1], 7u);
        EXPECT_EQ(Cells<uint32_t>(table.Read(0, first, 2)), (std::vector<uint32_t>{1000, 7}));

        EXPECT_EQ(table.Restore(0, first), buffer);
        EXPECT_EQ(Gather(device, table.Pages(0), picks, 1),
                  (std::vector<uint32_t>{4, first, first + 299}));
        EXPECT_THROW(table.Restore(0, first), Error);
    });
}

TEST(TableTest, PageTablesKeepTheirAddressUntilTheyOutgrowIt) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ColumnTable table(heap, "test", {{4, 0}});
        table.Allocate(1);
        const uint64_t layout = table.Layout();
        const uint64_t pages = table.Pages(0);
        table.Allocate(kPageRows - 1);
        EXPECT_EQ(table.Layout(), layout);
        EXPECT_EQ(table.Pages(0), pages);
        table.Allocate(100000);
        EXPECT_GT(table.Layout(), layout);
        EXPECT_EQ(Gather(device, table.Pages(0), std::vector<uint32_t>{0}, 1),
                  std::vector<uint32_t>{0});
    });
}

TEST(ArrayPoolTest, ARunPacksItsArraysBackToBack) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ArrayPool arrays(heap, "test", 12);
        const std::vector<ArrayRef> run = arrays.AllocateRun(std::vector<uint32_t>{5, 2, 0, 3});
        ASSERT_EQ(run.size(), 4u);
        EXPECT_EQ(run[1].address, run[0].address + 60);
        EXPECT_EQ(run[2].address, 0u);
        EXPECT_EQ(run[3].address, run[1].address + 24);
        EXPECT_EQ(run[3].count, 3u);
        EXPECT_EQ(arrays.ArrayCount(), 3u);
    });
}

TEST(ArrayPoolTest, ARunIsFreedOnlyAfterItsLastArray) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ArrayPool arrays(heap, "test", 4);
        const std::vector<ArrayRef> run = arrays.AllocateRun(std::vector<uint32_t>{64, 64});
        arrays.Write(run[1], Bytes(Sequence(64, 9)));
        arrays.Free(run[0]);
        const ArrayRef other = arrays.Allocate(128);
        EXPECT_NE(other.address, run[0].address);
        EXPECT_EQ(Cells<uint32_t>(arrays.Read(run[1])), Sequence(64, 9));
        arrays.Free(run[1]);
        EXPECT_THROW(arrays.Read(run[1]), Error);
        arrays.Free(other);
    });
}

TEST(ArrayPoolTest, SharedArraysLiveUntilTheirLastReference) {
    test::ForEachDevice([](device::Device& device) {
        DeviceHeap heap(device, "test");
        ArrayPool arrays(heap, "test", 4);
        const ArrayRef array = arrays.Allocate(3);
        arrays.Share(array);
        EXPECT_EQ(arrays.References(array), 2u);
        arrays.Free(array);
        EXPECT_EQ(arrays.References(array), 1u);
        arrays.Write(array, Bytes(std::vector<uint32_t>{1, 2, 3}));
        EXPECT_EQ(Cells<uint32_t>(arrays.Read(array)), (std::vector<uint32_t>{1, 2, 3}));
        arrays.Free(array);
        EXPECT_EQ(arrays.ArrayCount(), 0u);
    });
}

TEST(ColumnStoreTest, SceneDataPointsEveryColumnAtItsPageTable) {
    test::ForEachDevice([](device::Device& device) {
        ColumnStore store(device);
        ColumnTable& xformable = store.Schema(SchemaId::Xformable);
        xformable.Allocate(1);
        const AuthoredData data = store.Data();
        EXPECT_EQ(data.schemas.xformable.translate.pages.address,
                  xformable.Pages(kXformableTranslate.index));
        EXPECT_EQ(data.schemas.xformable.prim.pages.address,
                  xformable.Pages(store.PrimsColumn(SchemaId::Xformable)));
        EXPECT_EQ(data.schemas.xformable.translateStamp.pages.address,
                  xformable.Pages(store.StampColumn(kXformableTranslate)));
        EXPECT_EQ(data.prims.world.pages.address, store.Prims().Pages(kPrimWorld));
    });
}

TEST(ColumnStoreTest, ScatteredCellsStampTheirRowsAndWorlds) {
    test::ForEachDevice([](device::Device& device) {
        ColumnStore store(device);
        store.SetCounts(4, 3);
        ColumnTable& prims = store.Prims();
        prims.Allocate(4);
        prims.Stage(kPrimWorld, 0, Bytes(std::vector<uint32_t>{0, 2, 2, kGlobalWorld}));
        ColumnTable& xformable = store.Schema(SchemaId::Xformable);
        xformable.Allocate(4);
        xformable.Stage(store.PrimsColumn(SchemaId::Xformable), 0,
                        Bytes(std::vector<uint32_t>{0, 1, 2, 3}));
        store.SetStamp(5);

        const std::vector<uint32_t> rows = {1, 3};
        const std::vector<double> cells = {1, 2, 3, 4, 5, 6};
        const Ref<device::Buffer> rowBuffer = device.CreateBuffer({.label = "rows"}, rows);
        const Ref<device::Buffer> cellBuffer = device.CreateBuffer({.label = "cells"}, cells);
        CellsParams params{};
        params.target = store.Stamps(kXformableTranslate);
        params.column.address = xformable.Pages(kXformableTranslate.index);
        params.cells.address = cellBuffer->Address();
        params.rows.address = rowBuffer->Address();
        params.count = 2;
        params.words = sizeof(Double3) / 4;
        Dispatch(device, "ScatterCells", params);

        EXPECT_EQ(Cells<double>(xformable.Read(kXformableTranslate.index, 0, 4)),
                  (std::vector<double>{0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5, 6}));
        EXPECT_EQ(Cells<uint64_t>(xformable.Read(store.StampColumn(kXformableTranslate), 0, 4)),
                  (std::vector<uint64_t>{0, 5, 0, 5}));
        EXPECT_EQ(Cells<uint64_t>(store.Worlds().Read(0, 0, 4)),
                  (std::vector<uint64_t>{5, 0, 0, 5}));
    });
}

TEST(ColumnStoreTest, UntrackedColumnsStampNothing) {
    test::ForEachDevice([](device::Device& device) {
        ColumnStore store(device);
        EXPECT_FALSE(store.Tracked(kSphereRadius));
        EXPECT_EQ(store.Stamps(kSphereRadius).stamps.address, 0u);
        EXPECT_TRUE(store.Tracked(kXformableTranslate));
    });
}

TEST(ColumnStoreTest, WritesBumpOnlyTheirColumnsSequence) {
    test::ForEachDevice([](device::Device& device) {
        ColumnStore store(device);
        EXPECT_EQ(store.LastWrite(kXformableTranslate), 0u);
        store.Touch(kXformableTranslate);
        const uint64_t translate = store.LastWrite(kXformableTranslate);
        EXPECT_GT(translate, 0u);
        EXPECT_EQ(store.LastWrite(kXformableOrient), 0u);
        store.Touch(kSphereRadius);
        EXPECT_EQ(store.LastWrite(kXformableTranslate), translate);
        EXPECT_EQ(store.WriteSequence(), 2u);
    });
}

TEST(ColumnStoreTest, SceneDataIsUploadedOnlyWhenItChanges) {
    test::ForEachDevice([](device::Device& device) {
        ColumnStore store(device);
        const uint64_t address = store.DeviceData();
        const uint64_t uploaded = device.Traffic().uploadBytes;
        EXPECT_EQ(store.DeviceData(), address);
        EXPECT_EQ(device.Traffic().uploadBytes, uploaded);
        store.SetCounts(3, 1);
        store.DeviceData();
        EXPECT_GT(device.Traffic().uploadBytes, uploaded);
    });
}

}
}
