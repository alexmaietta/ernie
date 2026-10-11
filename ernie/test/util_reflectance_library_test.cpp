// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/reflectance_library.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

using ernie::FindReflectance;
using ernie::ReflectanceLibrary;

TEST(ReflectanceLibrary, EverySpectrumIsMeasuredAcrossTheShortwave) {
    std::set<std::string> ids;
    for (const ernie::MeasuredReflectance& entry :
         ReflectanceLibrary()) {
        EXPECT_TRUE(ids.insert(std::string(entry.id)).second)
            << entry.id << " twice";
        EXPECT_FALSE(entry.name.empty());
        EXPECT_FALSE(entry.source.empty());
        const ernie::Spectrum& spectrum = entry.spectrum;
        EXPECT_LE(spectrum.Wavelengths().front(), 430.0) << entry.id;
        EXPECT_GE(spectrum.Wavelengths().back(), 2490.0) << entry.id;
        for (const double value : spectrum.Values()) {
            EXPECT_GE(value, 0.0) << entry.id;
            EXPECT_LE(value, 1.0) << entry.id;
        }
        EXPECT_EQ(FindReflectance(entry.id), &entry);
    }
    EXPECT_EQ(ids.size(), 19u);
    EXPECT_EQ(FindReflectance("unobtainium"), nullptr);
}

TEST(ReflectanceLibrary, MaterialsReflectBeyondTheVisibleAsTheyDo) {
    const auto at = [](const char* id, double nm) {
        return FindReflectance(id)->spectrum(nm);
    };
    // Green leaves: dark in the red, bright past the red edge, and
    // darker again in the water bands of the shortwave infrared.
    EXPECT_GT(at("grass", 850.0), 5.0 * at("grass", 670.0));
    EXPECT_LT(at("grass", 1950.0), 0.5 * at("grass", 1050.0));
    EXPECT_LT(at("dry_grass", 850.0) / at("dry_grass", 670.0),
              at("grass", 850.0) / at("grass", 670.0))
        << "dry grass has no red edge to speak of";
    // Snow: white in the visible, dark past 1.4 um.
    EXPECT_GT(at("fine_snow", 550.0), 0.9);
    EXPECT_LT(at("fine_snow", 1550.0), 0.2);
    // Asphalt dark throughout; aluminium a mirror in the thermal
    // infrared, which is why it looks cold there.
    EXPECT_LT(at("asphalt", 1550.0), 0.15);
    EXPECT_GT(at("aluminium", 10000.0), 0.9);
    EXPECT_LT(at("grass", 10000.0), 0.05);
}
