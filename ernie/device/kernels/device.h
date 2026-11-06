// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

struct DeviceGlobals {
    Ptr<uint> debug;
    uint debugCapacity;
    uint reserved;
};

ERNIE_CHECK_SIZE(DeviceGlobals, 16);

static const uint kDebugHeaderWords = 4;
static const uint kDebugRecordWords = 16;

static const uint kTakenWord = 0;
static const uint kSubmissionWord = 1;

static const uint kStampWord = 0;
static const uint kKindWord = 1;
static const uint kHashWord = 2;
static const uint kThreadWord = 3;
static const uint kValuesWord = 6;
static const uint kLinearWord = 14;
static const uint kKernelWord = 15;

static const uint kAssertRecord = 1;
static const uint kPrintRecord = 2;

static const uint kPrintedUint = 0;
static const uint kPrintedInt = 1;
static const uint kPrintedFloat = 2;
static const uint kPrintedBool = 3;

}
