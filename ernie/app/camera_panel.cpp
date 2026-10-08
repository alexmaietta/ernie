#include "ernie/app/builtin_panels.h"
#include "ernie/app/property_fields.h"
#include "ernie/app/session.h"
#include "ernie/app/viewer.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/tokens.h"

#include <imgui.h>
#include <implot.h>

#include <cmath>
#include <exception>
#include <string>
#include <vector>

namespace ernie::app {

namespace {

using Kind = PropertyField::Kind;

/// The camera attributes Ernie reads, with the fallbacks of a default `Camera`.
std::vector<PropertyField> LensFields() {
    using namespace tokens;
    const Camera camera;
    return {
        {kFocalLength, "Focal length", Kind::Float, camera.focalLengthMm, 1.0, 2000.0, {}, "mm"},
        {kVerticalAperture, "Vertical aperture", Kind::Float, camera.verticalApertureMm, 0.1,
         100.0, {}, "mm"},
        {kClippingRange, "Clipping range", Kind::Vec2,
         glm::dvec2(camera.nearPlane, camera.farPlane)},
        {kLensVignetting, "Vignetting", Kind::Bool, camera.lens.vignetting},
        {kLensTransmission, "Transmission", Kind::Float, camera.lens.transmission, 0.0, 1.0},
    };
}

std::vector<PropertyField> ExposureFields() {
    using namespace tokens;
    const Camera camera;
    return {
        {kCameraExposure, "Exposure", Kind::Float, camera.exposureStops, -16.0, 16.0, {},
         "Stops; scales the photographic outputs"},
        {kExposureTime, "Exposure time", Kind::Float, camera.exposureTime, 1e-6, 10.0, {},
         "Seconds"},
        {kExposureIso, "ISO", Kind::Float, camera.iso, 25.0, 102400.0},
        {kExposureFStop, "f-number", Kind::Float, camera.fStop, 0.5, 64.0},
        {kExposureResponsivity, "Responsivity", Kind::Float, camera.responsivity, 0.0, 100.0},
    };
}

std::vector<PropertyField> SensorFields() {
    using namespace tokens;
    const Camera camera;
    const SensorModel& sensor = camera.sensor;
    return {
        {kSpectralRange, "Spectral range", Kind::Vec2,
         glm::dvec2(camera.bands.minNm, camera.bands.maxNm), 0, 0, {}, "nm"},
        {kSpectralBands, "Bands", Kind::Int, camera.bands.count, 1, SpectralBands::kMaxBands},
        {kSensorType, "Type", Kind::Token, std::string("rgb"), 0, 0, {"rgb", "mono"}},
        {kSensorNoise, "Noise", Kind::Bool, sensor.noise},
        {kSensorFullWell, "Full well", Kind::Float, sensor.fullWellElectrons, 100.0, 1e6, {},
         "Electrons"},
        {kSensorReadNoise, "Read noise", Kind::Float, sensor.readNoiseElectrons, 0.0, 100.0, {},
         "Electrons"},
        {kSensorDarkCurrent, "Dark current", Kind::Float, sensor.darkCurrentElectrons, 0.0, 1000.0,
         {}, "Electrons per second"},
        {kSensorPrnu, "PRNU", Kind::Float, sensor.prnu, 0.0, 0.2},
        {kSensorDsnu, "DSNU", Kind::Float, sensor.dsnuElectrons, 0.0, 100.0, {}, "Electrons"},
        {kSensorGain, "Electrons per DN", Kind::Float, sensor.electronsPerDnAtIso100, 0.0, 1000.0,
         {}, "At ISO 100; 0 for the full well over the ADC range"},
        {kSensorAdcBits, "ADC bits", Kind::Int, sensor.adcBits, 1, 16},
        {kSensorBlackLevel, "Black level", Kind::Float, sensor.blackLevelDn, 0.0, 4096.0, {}, "DN"},
        {kSensorSeed, "Seed", Kind::Int, static_cast<int>(sensor.seed), 0, 0},
    };
}

class CameraPanel final : public Panel {
  public:
    CameraPanel() : Panel("Camera", Dock::Right) {}

    void Draw(Viewer& viewer) override {
        Scene& scene = viewer.GetScene();
        const Path& camera = viewer.ActiveCamera();
        ImGui::TextUnformatted(camera.String().c_str());
        if (IsSessionPath(camera)) {
            ImGui::SameLine();
            ImGui::TextDisabled("(the viewer's)");
        }
        if (ImGui::Button("Create Camera from View")) {
            const Path created = viewer.CameraFromView(FreePath(scene, "/World/Camera"));
            viewer.SetActiveCamera(created);
            viewer.SetSelection({created});
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("A scene camera with the view's pose and lens, looked through");
        }
        const float labelWidth = viewer.GetTheme().propertyLabelWidth * ImGui::GetFontSize() /
                                 viewer.GetTheme().fontSize;
        std::string error;
        const auto section = [&](const char* title, const std::vector<PropertyField>& fields) {
            if (ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID(title);
                const std::string failure = DrawPropertyFields(viewer, camera, fields, labelWidth);
                if (error.empty()) {
                    error = failure;
                }
                ImGui::PopID();
            }
        };
        section("Lens", lens_);
        section("Exposure", exposure_);
        section("Sensor", sensor_);
        if (ImGui::CollapsingHeader("Quantum Efficiency", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawEfficiency(scene, camera);
        }
        if (ImGui::CollapsingHeader("Filter Chain")) {
            std::string chain;
            for (const SensorFilter filter : DefaultFilterChain()) {
                chain += (chain.empty() ? "" : "  >  ") + std::string(NameOf(filter));
            }
            ImGui::TextWrapped("%s", chain.c_str());
            ImGui::TextDisabled("The viewport's chain; products set theirs with ernie:filters.");
        }
        if (!error.empty()) {
            ImGui::TextColored(ImVec4(0.89f, 0.34f, 0.32f, 1.0f), "%s", error.c_str());
        }
    }

  private:
    /// `base`, or it with the first number after it that names no prim.
    static Path FreePath(const Scene& scene, const std::string& base) {
        Path path(base);
        for (int n = 1; scene.Has(path); ++n) {
            path = Path(base + std::to_string(n));
        }
        return path;
    }

    /// Each channel's quantum efficiency over wavelength, and as binned into the camera's bands.
    static void DrawEfficiency(const Scene& scene, const Path& path) {
        Camera camera;
        try {
            camera = CameraOf(scene, path, 64, 64);
        } catch (const std::exception& error) {
            ImGui::TextDisabled("%s", error.what());
            return;
        }
        const SpectralBands& bands = camera.bands;
        if (!ImPlot::BeginPlot("##qe", ImVec2(-1.0f, 180.0f), ImPlotFlags_NoMenus)) {
            return;
        }
        ImPlot::SetupAxes("nm", "QE", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        const double low = bands.minNm - 20.0;
        const double high = bands.maxNm + 20.0;
        for (const SensorChannel& channel : camera.sensor.channels) {
            std::vector<double> wavelengths;
            std::vector<double> values;
            for (double nm = low; nm <= high; nm += 2.0) {
                wavelengths.push_back(nm);
                values.push_back(channel.quantumEfficiency(nm));
            }
            ImPlot::PlotLine(channel.name.c_str(), wavelengths.data(), values.data(),
                             static_cast<int>(values.size()));
            // What the renderer uses: the curve's mean over each band.
            const std::vector<float> binned = channel.quantumEfficiency.Bin(bands);
            std::vector<double> edges;
            std::vector<double> steps;
            for (int band = 0; band < bands.count; ++band) {
                edges.push_back(bands.Lower(band));
                steps.push_back(binned[static_cast<std::size_t>(band)]);
            }
            edges.push_back(bands.maxNm);
            steps.push_back(steps.empty() ? 0.0 : steps.back());
            ImPlotSpec spec;
            spec.FillAlpha = 0.25f;
            spec.Flags = ImPlotStairsFlags_Shaded;
            ImPlot::PlotStairs((channel.name + " bands").c_str(), edges.data(), steps.data(),
                               static_cast<int>(steps.size()), spec);
        }
        ImPlot::EndPlot();
        ImGui::TextDisabled("%d bands of %.1f nm from %.0f to %.0f nm", bands.count, bands.Width(),
                            bands.minNm, bands.maxNm);
    }

    std::vector<PropertyField> lens_ = LensFields();
    std::vector<PropertyField> exposure_ = ExposureFields();
    std::vector<PropertyField> sensor_ = SensorFields();
};

}  // namespace

std::unique_ptr<Panel> MakeCameraPanel() {
    return std::make_unique<CameraPanel>();
}

}  // namespace ernie::app
