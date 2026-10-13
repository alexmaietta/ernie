// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "bench.h"

#include "ernie/util/tensor.h"

#include <span>
#include <vector>

namespace ernie::bench {

/// A batch of views through one camera, `views` of each of `worlds`
/// worlds that overlap at the origin, told apart by their world index
/// (`ernie:worlds`); prims without one, such as a ground, are in
/// every world. Its parameters:
///
/// - resolution: pixels along each side of a square view
/// - worlds, views: worlds, and views of each
/// - preset: a registered preset (render settings)
/// - outputs: a registered output set (render vars)
///
/// Poses change every step, from a cycle of `kPoseSets` uploaded
/// to CUDA when the renderer has it (as a simulator hands them over),
/// and the outputs are mapped to CUDA without a copy.
class BatchScenario : public Scenario {
  public:
    static constexpr int kPoseSets = 32;

    std::vector<Param> Params() const override;

    /// Whole frames, pipelined, calibrated to samples of 50 ms.
    Definition Describe(const Case& c) const override;

    /// Every view's colour (or distances) should vary, not render
    /// flat.
    std::string
    Validate(const Case& c,
             const RenderProductSetOutputs& outputs) override;

    void Build(Scene& scene, const Case& c) final;
    void Setup(Renderer& renderer, Scene& scene,
               const Case& c) override;
    void Step(Renderer& renderer, Scene& scene, const Case& c,
              int step) override;
    std::vector<Unit> Units(const Case& c) const override;

    /// Whether poses are on the GPU, after `Setup`.
    bool PosesOnGpu() const {
        return posesOnGpu_;
    }

  protected:
    /// The scenario's own parameters, after the batch's.
    virtual std::vector<Param> WorldParams() const {
        return {};
    }

    /// Adds the worlds' prims; the camera, the sun and the render
    /// prims are there already.
    virtual void BuildWorlds(Scene& scene, const Case& c) = 0;

    /// The pose of every view at `step`, world-major: view `v` sees
    /// world `v / views`.
    virtual void Poses(const Case& c, int step,
                       std::span<glm::dmat4> poses) const = 0;

    /// The camera's focal length, mm (the aperture is 20.955 mm).
    virtual double FocalLength() const {
        return 18.0;
    }

    /// Towards the sun.
    virtual glm::dvec3 SunDirection() const {
        return {0.4, -0.6, 0.7};
    }

    /// The ambient light's luminance, cd/m^2 (`ernie:ambient`).
    virtual double Ambient() const {
        return 0.3;
    }

    int Worlds(const Case& c) const {
        return GetInt(c, "worlds");
    }
    int ViewsPerWorld(const Case& c) const {
        return GetInt(c, "views");
    }

  private:
    Tensor PoseTensor(const Case& c, int step) const;

    std::vector<Tensor> poses_;
    bool posesOnGpu_ = false;
};

}  // namespace ernie::bench
