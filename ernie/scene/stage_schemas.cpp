// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/stage_schemas.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(Stage, "Stage", nullptr, true,
                    "A scene's stage metadata, on its root prim: its "
                    "units, time and default prim.",
                    ERNIE_STAGE_ATTRIBUTES)

}  // namespace ernie
