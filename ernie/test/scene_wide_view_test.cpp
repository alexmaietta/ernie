#include "ernie/scene/wide_view.h"

#include "ernie/scene/cahvore.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <random>

namespace {

constexpr ernie::CubeFace kFaces[] = {ernie::CubeFace::Front, ernie::CubeFace::Right,
                                      ernie::CubeFace::Left, ernie::CubeFace::Up,
                                      ernie::CubeFace::Down};

}  // namespace

TEST(CubeFaces, AreRotationsLookingAlongTheCamerasAxes) {
    const glm::dvec3 looks[] = {{0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    for (int f = 0; f < ernie::kCubeFaces; ++f) {
        const glm::dmat3 axes = ernie::CubeFaceAxes(kFaces[f]);
        EXPECT_NEAR(glm::determinant(axes), 1.0, 1e-12) << f;
        const glm::dmat3 identity = glm::transpose(axes) * axes;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                EXPECT_NEAR(identity[i][j], i == j ? 1.0 : 0.0, 1e-12);
            }
        }
        EXPECT_NEAR(glm::distance(-axes[2], looks[f]), 0.0, 1e-12) << "face " << f;
        EXPECT_EQ(ernie::CubeFaceOf(looks[f]), kFaces[f]);
    }
}

TEST(CubeFaces, CoverEveryRayWithin125DegreesOfTheAxis) {
    std::mt19937 random(7);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    int checked = 0;
    while (checked < 20000) {
        glm::dvec3 ray(unit(random), unit(random), unit(random));
        if (glm::length(ray) > 1.0 || glm::length(ray) < 1e-3) {
            continue;
        }
        ray = glm::normalize(ray);
        const double offAxis = std::acos(-ray.z);
        if (offAxis > 125.0 * M_PI / 180.0) {
            continue;
        }
        const ernie::CubeFace face = ernie::CubeFaceOf(ray);
        const glm::dvec2 plane = ernie::CubeFacePlane(face, ray);
        ASSERT_LE(std::abs(plane.x), 1.0 + 1e-12) << offAxis;
        ASSERT_LE(std::abs(plane.y), 1.0 + 1e-12) << offAxis;
        ASSERT_NEAR(glm::distance(ernie::CubeFaceRay(face, plane), ray), 0.0, 1e-12);
        ++checked;
    }
    // Past 125.3 degrees along a diagonal, the back face would be needed.
    const glm::dvec3 behind =
        glm::normalize(glm::dvec3(std::sin(2.25) / std::sqrt(2.0), std::sin(2.25) / std::sqrt(2.0),
                                  -std::cos(2.25)));
    const glm::dvec2 plane = ernie::CubeFacePlane(ernie::CubeFaceOf(behind), behind);
    EXPECT_GT(std::max(std::abs(plane.x), std::abs(plane.y)), 1.0);
    // A face of 512 pixels with a margin of 2 reaches 2 pixels past 45 degrees.
    const double extent = ernie::CubeFaceExtent(512.0, 2.0);
    EXPECT_NEAR((extent - 1.0) * 512.0 / (2.0 * extent), 2.0, 1e-12);
}

TEST(LensRays, FollowFisheyesPastNinetyDegrees) {
    ernie::LensModel pinhole;
    EXPECT_NEAR(glm::distance(ernie::LensRay(pinhole, {0.5, -0.25}),
                              glm::normalize(glm::dvec3(0.5, -0.25, -1.0))),
                0.0, 1e-12);
    // Equidistant: the field angle is the radius, 2 radians (115 degrees) here, behind the
    // camera's image plane and on its right face.
    ernie::LensModel fisheye;
    fisheye.projection = ernie::LensProjection::Fisheye;
    const glm::dvec3 ray = ernie::LensRay(fisheye, {2.0, 0.0});
    EXPECT_NEAR(glm::distance(ray, glm::dvec3(std::sin(2.0), 0.0, -std::cos(2.0))), 0.0, 1e-9);
    EXPECT_GT(ray.z, 0.0);
    EXPECT_EQ(ernie::CubeFaceOf(ray), ernie::CubeFace::Right);
    // Brown-Conrady: the ray through the undistorted point.
    ernie::LensModel barrel;
    barrel.k1 = -0.1;
    const glm::dvec2 distorted(0.4, 0.3);
    const glm::dvec3 bent = ernie::LensRay(barrel, distorted);
    EXPECT_NEAR(glm::distance(glm::dvec2(bent) / -bent.z, barrel.Undistort(distorted)), 0.0,
                1e-12);
}

TEST(LensRays, Mars2020HazcamsNeedCubeFacesAndNavcamsDoNot) {
    const auto models = ernie::ReadCahvoreModelsFile(
        (std::filesystem::path(ERNIE_TEST_FIXTURES) / "cahvore" / "m2020_camera_models.json")
            .string());
    for (const ernie::NamedCahvore& named : models) {
        if (named.name != "HAZFLA" && named.name != "NAVL") {
            continue;
        }
        const ernie::CahvoreModel::ErnieCamera fit = named.model.ToCamera(6.4);
        // The camera's widest ray, through its fitted lens, as the model's own.
        EXPECT_NEAR(ernie::MaxFieldAngle(fit.camera), fit.fieldAngleMax, 0.01) << named.name;
        EXPECT_EQ(ernie::NeedsCubeFaces(fit.camera), named.name == "HAZFLA") << named.name;
    }
    // The Navcam's ray table, every 8 pixels, gives its exact rays to a twentieth of a pixel's
    // angle across the image (where the fitted lens is off by pixels).
    const ernie::NamedCahvore* navl = nullptr;
    for (const ernie::NamedCahvore& named : models) {
        navl = named.name == "NAVL" ? &named : navl;
    }
    ASSERT_NE(navl, nullptr);
    const ernie::LensRayTable table = navl->model.RayTable(8);
    EXPECT_EQ(table.columns, 5120u / 8 + 1);
    const ernie::CahvoreModel::ErnieCamera fit = navl->model.ToCamera(6.4);
    const glm::dmat3 toCamera = glm::transpose(glm::dmat3(fit.camera.transform));
    const double pixelAngle = 1.0 / fit.camera.FocalLengthPixels();
    double worst = 0.0;
    for (const glm::dvec2 image : {glm::dvec2(3.3, 2.1), glm::dvec2(2559.5, 1919.5),
                                   glm::dvec2(5117.0, 3800.7), glm::dvec2(1000.25, 3837.5)}) {
        const glm::dvec3 exact = toCamera * navl->model.ImageToRay(image).direction;
        const glm::dvec3 tabled = table.RayAt(image + 0.5);
        worst = std::max(worst, std::acos(std::min(1.0, glm::dot(exact, tabled))) / pixelAngle);
    }
    EXPECT_LT(worst, 0.05);

    ernie::Camera pinhole;
    pinhole.width = 640;
    pinhole.height = 480;
    pinhole.focalLengthMm = 2.0;  // a 150 degree pinhole is stretched, not a cube camera
    EXPECT_FALSE(ernie::NeedsCubeFaces(pinhole));
}
