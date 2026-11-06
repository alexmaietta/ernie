// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>

#include "ernie/device/ref.h"
#include "ernie/export.h"

namespace ernie::device {

class Device;

class ERNIE_API Resource : public RefCounted {
  public:
    ~Resource() override;

    Device& GetDevice() const {
        return *m_device;
    }

    const std::string& Label() const {
        return m_label;
    }

  protected:
    Resource(Device& device, std::string label);

  private:
    friend class Device;

    void LastReleased() noexcept override;

    Device* m_device;
    std::string m_label;
    bool m_holdsDevice = true;
};

}
