// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/cahvore.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

std::string Fixture(const char* name) {
    return (std::filesystem::path(ERNIE_TEST_FIXTURES) / "cahvore" /
            name)
        .string();
}

const ernie::CahvoreModel&
Named(const std::vector<ernie::NamedCahvore>& models,
      const std::string& name) {
    for (const ernie::NamedCahvore& model : models) {
        if (model.name == name) {
            return model.model;
        }
    }
    throw std::invalid_argument("No camera model " + name);
}

double Degrees(const glm::dvec3& a, const glm::dvec3& b) {
    return std::acos(std::clamp(glm::dot(a, b), -1.0, 1.0)) * 180.0 /
           M_PI;
}

// A 1024 x 1024 camera looking down +z from the origin, focal length
// 512 pixels.
ernie::CahvoreModel Simple(ernie::CahvoreModel::Kind kind) {
    ernie::CahvoreModel model;
    model.kind = kind;
    model.c = {0.0, 0.0, 0.0};
    model.a = {0.0, 0.0, 1.0};
    model.h = {512.0, 0.0, 511.5};
    model.v = {0.0, 512.0, 511.5};
    model.o = model.a;
    model.width = 1024;
    model.height = 1024;
    return model;
}

}  // namespace

TEST(Cahvore, ReadsTheMars2020CameraModels) {
    const auto models = ernie::ReadCahvoreModelsFile(
        Fixture("m2020_camera_models.json"));
    ASSERT_EQ(models.size(), 6u);
    const ernie::CahvoreModel& hazcam = Named(models, "HAZFLA");
    EXPECT_EQ(hazcam.kind, ernie::CahvoreModel::Kind::Cahvore);
    EXPECT_EQ(hazcam.pupilType, 3);
    EXPECT_DOUBLE_EQ(hazcam.linearity, 0.37);
    EXPECT_DOUBLE_EQ(hazcam.e.x, -0.006127);
    EXPECT_EQ(hazcam.width, 5120u);
    EXPECT_EQ(Named(models, "SuperCam").kind,
              ernie::CahvoreModel::Kind::Cahv);
    EXPECT_EQ(Named(models, "MCAM_Z_LEFT-Z026").kind,
              ernie::CahvoreModel::Kind::Cahvor);
    EXPECT_THROW(ernie::ReadCahvoreModels("{}"),
                 std::invalid_argument);
    EXPECT_THROW(
        ernie::ReadCahvoreModels(
            R"([{"name": "x", "model": {"type": "cahvx"}}])"),
        std::invalid_argument);
}

TEST(Cahvore, ReducesToCahvWithoutDistortion) {
    // CAHVOR with no radial terms, and CAHVORE of type 1
    // (perspective) with neither radial nor pupil terms, are CAHV.
    const ernie::CahvoreModel cahv =
        Simple(ernie::CahvoreModel::Kind::Cahv);
    ernie::CahvoreModel cahvor =
        Simple(ernie::CahvoreModel::Kind::Cahvor);
    ernie::CahvoreModel cahvore =
        Simple(ernie::CahvoreModel::Kind::Cahvore);
    cahvore.pupilType = 1;
    for (const glm::dvec2 image :
         {glm::dvec2(511.5, 511.5), glm::dvec2(0.0, 0.0),
          glm::dvec2(1023.0, 300.0), glm::dvec2(100.0, 900.0)}) {
        const glm::dvec3 expected = cahv.ImageToRay(image).direction;
        EXPECT_NEAR(
            Degrees(cahvor.ImageToRay(image).direction, expected),
            0.0, 1e-9);
        EXPECT_NEAR(
            Degrees(cahvore.ImageToRay(image).direction, expected),
            0.0, 1e-6);
        // The CAHV projection: (P - C) . H / (P - C) . A.
        const glm::dvec2 back = cahv.PointToImage(3.0 * expected);
        EXPECT_NEAR(back.x, image.x, 1e-9);
        EXPECT_NEAR(back.y, image.y, 1e-9);
    }
    EXPECT_NEAR(Degrees(cahv.ImageToRay({1023.0, 511.5}).direction,
                        {0.0, 0.0, 1.0}),
                std::atan(511.5 / 512.0) * 180.0 / M_PI, 1e-9);
}

TEST(Cahvore, ProjectsPointsBackOntoTheirPixels) {
    // Every model, through its distortion and (CAHVORE) moving
    // entrance pupil: points along a pixel's ray, near and far,
    // project back to the pixel.
    for (const ernie::NamedCahvore& named :
         ernie::ReadCahvoreModelsFile(
             Fixture("m2020_camera_models.json"))) {
        const ernie::CahvoreModel& m = named.model;
        for (const double fx : {0.02, 0.3, 0.5, 0.81, 0.98}) {
            for (const double fy : {0.03, 0.5, 0.97}) {
                const glm::dvec2 image(fx * (m.width - 1),
                                       fy * (m.height - 1));
                const ernie::CahvoreModel::Ray ray =
                    m.ImageToRay(image);
                for (const double distance : {0.5, 3.0, 50.0}) {
                    const glm::dvec2 back = m.PointToImage(
                        ray.origin + distance * ray.direction);
                    EXPECT_NEAR(back.x, image.x, 1e-6)
                        << named.name << " at " << distance;
                    EXPECT_NEAR(back.y, image.y, 1e-6)
                        << named.name << " at " << distance;
                }
            }
        }
    }
}

TEST(Cahvore, Mars2020ModelsSeeThePublishedFieldsOfView) {
    // Maki et al. 2020, Table 2: the Navcams 96 x 73 degrees, the
    // Hazcams 136 x 102 (nominal; the calibrated models, edge pixel
    // centre to edge pixel centre, see the Navcam's 93.6 x 72.6 and
    // the Hazcam's 128.7 x 98.6).
    const auto models = ernie::ReadCahvoreModelsFile(
        Fixture("m2020_camera_models.json"));
    const auto field = [](const ernie::CahvoreModel& m,
                          bool horizontal) {
        const double w = m.width - 1.0;
        const double h = m.height - 1.0;
        return horizontal
                   ? Degrees(m.ImageToRay({0.0, 0.5 * h}).direction,
                             m.ImageToRay({w, 0.5 * h}).direction)
                   : Degrees(m.ImageToRay({0.5 * w, 0.0}).direction,
                             m.ImageToRay({0.5 * w, h}).direction);
    };
    EXPECT_NEAR(field(Named(models, "NAVL"), true), 96.0, 3.0);
    EXPECT_NEAR(field(Named(models, "NAVL"), false), 73.0, 3.0);
    EXPECT_NEAR(field(Named(models, "HAZFLA"), true), 136.0, 8.0);
    EXPECT_NEAR(field(Named(models, "HAZFLA"), false), 102.0, 4.0);
}

TEST(Cahvore, ConvertsToErnieCameras) {
    const auto models = ernie::ReadCahvoreModelsFile(
        Fixture("m2020_camera_models.json"));
    // Mastcam-Z's left camera at 26 mm (Bell et al. 2021, Table 5:
    // 7.4 um pixels): CAHVOR as a perspective lens.
    const ernie::CahvoreModel::ErnieCamera mastcam =
        Named(models, "MCAM_Z_LEFT-Z026").ToCamera(7.4);
    EXPECT_EQ(mastcam.camera.lens.projection,
              ernie::LensProjection::Perspective);
    EXPECT_NEAR(mastcam.camera.focalLengthMm, 26.0, 0.5);
    EXPECT_LT(mastcam.residualPixels, 0.15)
        << "its pixels 0.015% from square";
    EXPECT_FALSE(mastcam.rowsReversed);
    EXPECT_NO_THROW(mastcam.camera.Check());
    // The Navcam (6.4 um, 19.1 mm): CAHVORE as a fish-eye, its moving
    // pupil millimetres. Its distortion's axis O lies 0.08 degrees
    // off the image plane's normal, a tilt Ernie's lens cannot take:
    // a few pixels at the image's edges, of 5120.
    const ernie::CahvoreModel& navl = Named(models, "NAVL");
    const ernie::CahvoreModel::ErnieCamera navcam =
        navl.ToCamera(6.4);
    EXPECT_EQ(navcam.camera.lens.projection,
              ernie::LensProjection::Fisheye);
    EXPECT_NEAR(navcam.camera.focalLengthMm, 19.1, 0.5);
    EXPECT_LT(navcam.residualPixels, 5.0);
    EXPECT_LT(navcam.pupilShiftMax, 0.01);
    EXPECT_NO_THROW(navcam.camera.Check());
    // Its pose: the camera at C looking along A.
    EXPECT_NEAR(
        glm::length(glm::dvec3(navcam.camera.transform[3]) - navl.c),
        0.0, 1e-12);
    EXPECT_NEAR(Degrees(-glm::dvec3(navcam.camera.transform[2]),
                        glm::normalize(navl.a)),
                0.0, 1e-9);
    // The Hazcam reaches almost 80 degrees off its axis, at its
    // corners.
    const ernie::CahvoreModel::ErnieCamera hazcam =
        Named(models, "HAZFLA").ToCamera(6.4);
    EXPECT_NEAR(hazcam.fieldAngleMax * 180.0 / M_PI, 79.4, 0.5);
    EXPECT_LT(hazcam.residualPixels, 5.0);
    // SuperCam's CAHV model in this release is no physical camera:
    // its image axes 75 degrees apart, its centre far off the image.
    EXPECT_THROW(Named(models, "SuperCam").ToCamera(6.4),
                 std::invalid_argument);
}

TEST(Cahvore, FitsMirroredAndDistortedModelsExactly) {
    // A CAHVOR lens with its axis on A, and its mirror (V up the
    // image): Ernie's Brown-Conrady lens takes the distortion to a
    // hundredth of a pixel, the mirror as flipped rows.
    ernie::CahvoreModel model =
        Simple(ernie::CahvoreModel::Kind::Cahvor);
    model.r = {0.0, -0.08, 0.02};
    const ernie::CahvoreModel::ErnieCamera upright =
        model.ToCamera(5.0);
    EXPECT_FALSE(upright.rowsReversed);
    EXPECT_LT(upright.residualPixels, 0.01);
    EXPECT_NE(upright.camera.lens.k1, 0.0);
    model.v = {0.0, -512.0, 511.5};
    const ernie::CahvoreModel::ErnieCamera mirrored =
        model.ToCamera(5.0);
    EXPECT_TRUE(mirrored.rowsReversed);
    EXPECT_LT(mirrored.residualPixels, 0.01);
    // A CAHVORE fish-eye on A, without pupil movement: Ernie's
    // Kannala-Brandt lens takes it.
    ernie::CahvoreModel fisheye =
        Simple(ernie::CahvoreModel::Kind::Cahvore);
    fisheye.pupilType = 2;
    fisheye.h = {300.0, 0.0, 511.5};
    fisheye.v = {0.0, 300.0, 511.5};
    fisheye.r = {0.0, 0.01, -0.002};
    const ernie::CahvoreModel::ErnieCamera wide =
        fisheye.ToCamera(5.0);
    EXPECT_EQ(wide.camera.lens.projection,
              ernie::LensProjection::Fisheye);
    EXPECT_GT(wide.fieldAngleMax, 1.0);
    EXPECT_LT(wide.residualPixels, 0.05);
}
