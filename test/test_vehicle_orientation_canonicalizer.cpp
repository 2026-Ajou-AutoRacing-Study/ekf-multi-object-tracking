#include <gtest/gtest.h>

#include "vehicle_orientation_canonicalizer.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

TEST(VehicleOrientationCanonicalizer, FlipsMovingReversedVehicle) {
    vehicle_orientation::Canonicalizer canonicalizer;
    const auto result = canonicalizer.Apply(
        7, kPi, 5.0, 0.0, true, true, 1.0,
        100.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    EXPECT_TRUE(result.flipped);
    EXPECT_NEAR(result.yaw, 0.0, 1.0e-9);
}

TEST(VehicleOrientationCanonicalizer, RetainsDecisionAtLowSpeed) {
    vehicle_orientation::Canonicalizer canonicalizer;
    canonicalizer.Apply(
        7, kPi, 5.0, 0.0, true, true, 1.0,
        100.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    const auto result = canonicalizer.Apply(
        7, kPi, 0.1, 0.0, true, true, 1.0,
        100.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    EXPECT_TRUE(result.flipped);
    EXPECT_NEAR(result.yaw, 0.0, 1.0e-9);
}

TEST(VehicleOrientationCanonicalizer, UsesHysteresisAroundNinetyDegrees) {
    vehicle_orientation::Canonicalizer canonicalizer;
    auto result = canonicalizer.Apply(
        7, 110.0 * kPi / 180.0, 5.0, 0.0, true, true, 1.0,
        100.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    EXPECT_TRUE(result.flipped);
    result = canonicalizer.Apply(
        7, 90.0 * kPi / 180.0, 5.0, 0.0, true, true, 1.0,
        100.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    EXPECT_TRUE(result.flipped);
    result = canonicalizer.Apply(
        7, 70.0 * kPi / 180.0, 5.0, 0.0, true, true, 1.0,
        100.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    EXPECT_FALSE(result.flipped);
}

TEST(VehicleOrientationCanonicalizer, DoesNotChangeNonVehicle) {
    vehicle_orientation::Canonicalizer canonicalizer;
    const auto result = canonicalizer.Apply(
        3, kPi, 5.0, 0.0, false, true, 1.0,
        100.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    EXPECT_FALSE(result.flipped);
    EXPECT_NEAR(std::abs(result.yaw), kPi, 1.0e-9);
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
