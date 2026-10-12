// SPDX-License-Identifier: BSD-3-Clause
// The casters' side of a virtual shadow map (virtual_shadow.slang): a shadow pass's fragment
// writing the level its Shadows block names (`Shadows::virtualMap`) through the page table.
// Drawables' shadow shaders give it a fragment entry of their own.
module virtual_shadow_caster;

import frame;
import virtual_shadow;

// The fragment at viewport pixel `position.xy` with reversed-Z depth `position.z`, into the
// level being drawn.
public void writeVirtualShadow(float4 position) {
    if (s.virtualMap.w != 0) {
        virtualShadowWrite(virtualShadowAt(s.virtualMap.xy), position, s.virtualMap.z);
    }
}
