// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>

#include "ernie/import/importer.h"

namespace ernie {

std::shared_ptr<Importer> CreateGltfImporter();

}
