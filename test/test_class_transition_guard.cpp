#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "class_transition_guard.hpp"

namespace {

namespace guard = class_transition_guard;

constexpr int kUnknown = 0;
constexpr int kCar = 1;
constexpr int kTruck = 2;
constexpr int kPedestrian = 3;
constexpr int kBicycle = 4;
const guard::Dimensions kObstacle{{1.2, 1.3, 1.0}};
const guard::Dimensions kVehicle{{4.6, 1.8, 1.5}};

guard::Config EnabledConfig() {
    guard::Config config;
    config.enabled = true;
    return config;
}

guard::Decision Observe(guard::State& state, const int observed_class,
                        const double stamp, const guard::Config& config = EnabledConfig()) {
    return guard::Evaluate(config, state, true, kUnknown, kObstacle,
                           observed_class, kVehicle, stamp);
}

void ExpectPassThrough(const guard::Decision& decision, const int classification) {
    ASSERT_EQ(decision.vote_count, 1U);
    EXPECT_EQ(decision.class_votes[0], classification);
    EXPECT_FALSE(decision.hold_shape);
    EXPECT_FALSE(decision.pending);
    EXPECT_FALSE(decision.released);
}

void ApplyVotes(const guard::Decision& decision, std::array<double, 5>& scores) {
    for (std::size_t vote = 0U; vote < decision.vote_count; ++vote) {
        for (std::size_t label = 0U; label < scores.size(); ++label) {
            scores[label] = 0.8 * scores[label] +
                (static_cast<int>(label) == decision.class_votes[vote] ? 0.2 : 0.0);
        }
    }
}

TEST(ClassTransitionGuard, DisabledDefaultPassesThroughAndClearsHistory) {
    guard::State state;
    ASSERT_TRUE(Observe(state, kCar, 1.0).pending);
    ExpectPassThrough(Observe(state, kCar, 1.1, guard::Config{}), kCar);
    EXPECT_EQ(state.phase, guard::Phase::Idle);
    EXPECT_EQ(state.vote_count, 0U);
}

TEST(ClassTransitionGuard, DisabledGuardPreservesOriginalClassAndDimensionEmaExactly) {
    guard::State state;
    std::array<double, 5> original_scores{{1.0, 0.0, 0.0, 0.0, 0.0}};
    auto guarded_scores = original_scores;
    auto original_dimensions = kObstacle;
    auto guarded_dimensions = original_dimensions;
    for (int index = 0; index < 200; ++index) {
        const int label = (index * 7) % 5;
        const auto& dimensions = index % 3 == 0 ? kVehicle : kObstacle;
        const auto decision = guard::Evaluate(guard::Config{}, state, true, kUnknown,
            guarded_dimensions, label, dimensions, index * 0.1);
        ApplyVotes(guard::PassThrough(label), original_scores);
        ApplyVotes(decision, guarded_scores);
        for (std::size_t axis = 0U; axis < dimensions.size(); ++axis) {
            original_dimensions[axis] = 0.3 * dimensions[axis] + 0.7 * original_dimensions[axis];
            if (!decision.hold_shape) {
                guarded_dimensions[axis] = 0.3 * dimensions[axis] + 0.7 * guarded_dimensions[axis];
            }
        }
        EXPECT_EQ(original_scores, guarded_scores);
        EXPECT_EQ(original_dimensions, guarded_dimensions);
    }
}

TEST(ClassTransitionGuard, RequiresThreeAndReplaysExactFamilyVoteOrder) {
    guard::State state;
    const auto first = Observe(state, kCar, 1.0);
    const auto second = Observe(state, kTruck, 1.1);
    EXPECT_TRUE(first.pending);
    EXPECT_TRUE(first.hold_shape);
    EXPECT_EQ(first.vote_count, 0U);
    EXPECT_TRUE(second.pending);
    const auto third = Observe(state, kCar, 1.2);
    EXPECT_TRUE(third.released);
    EXPECT_FALSE(third.hold_shape);
    ASSERT_EQ(third.vote_count, 3U);
    EXPECT_EQ(third.class_votes[0], kCar);
    EXPECT_EQ(third.class_votes[1], kTruck);
    EXPECT_EQ(third.class_votes[2], kCar);
    EXPECT_EQ(state.phase, guard::Phase::Released);
    EXPECT_EQ(state.vote_count, 0U);
}

TEST(ClassTransitionGuard, GenuineStationaryCorrectionEventuallySucceedsWithOriginalEma) {
    // There is intentionally no velocity/confidence input to this policy.
    guard::State state;
    std::array<double, 5> scores{{1.0, 0.0, 0.0, 0.0, 0.0}};
    for (int index = 0; index < 3; ++index) {
        ApplyVotes(Observe(state, kCar, 1.0 + 0.1 * index), scores);
    }
    EXPECT_NEAR(scores[kCar], 0.488, 1.e-12);
    EXPECT_NEAR(scores[kUnknown], 0.512, 1.e-12);
    const auto fourth = Observe(state, kCar, 1.3);
    ExpectPassThrough(fourth, kCar);
    ApplyVotes(fourth, scores);
    EXPECT_GT(scores[kCar], scores[kUnknown]);
    EXPECT_NEAR(scores[kCar], 0.5904, 1.e-12);
    ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, kCar,
        kVehicle, kCar, kVehicle, 1.4), kCar);
    EXPECT_EQ(state.phase, guard::Phase::Idle);
}

TEST(ClassTransitionGuard, VehicleToUnknownAlsoRequiresTemporalConfirmation) {
    guard::State state;
    for (int index = 0; index < 3; ++index) {
        const auto decision = guard::Evaluate(EnabledConfig(), state, true, kTruck,
            kVehicle, kUnknown, kObstacle, 1.0 + index * 0.1);
        EXPECT_EQ(decision.pending, index < 2);
        EXPECT_EQ(decision.released, index == 2);
    }
}

TEST(ClassTransitionGuard, AlternatingClassAndShapeNeverFlips) {
    guard::State state;
    std::array<double, 5> scores{{1.0, 0.0, 0.0, 0.0, 0.0}};
    for (int index = 0; index < 100; ++index) {
        const bool vehicle = index % 2 == 0;
        const auto decision = guard::Evaluate(EnabledConfig(), state, true, kUnknown,
            kObstacle, vehicle ? kCar : kUnknown,
            vehicle ? kVehicle : kObstacle, 1.0 + index * 0.1);
        ApplyVotes(decision, scores);
        EXPECT_EQ(scores[kCar], 0.0);
        EXPECT_EQ(scores[kUnknown], 1.0);
    }
}

TEST(ClassTransitionGuard, FamilyConsistentButShapeInconsistentEvidenceRestarts) {
    guard::State state;
    const guard::Dimensions huge_vehicle{{12.0, 3.0, 3.0}};
    for (int index = 0; index < 8; ++index) {
        const auto decision = guard::Evaluate(EnabledConfig(), state, true, kUnknown,
            kObstacle, kCar, index % 2 == 0 ? kVehicle : huge_vehicle,
            1.0 + index * 0.1);
        EXPECT_TRUE(decision.pending);
        EXPECT_EQ(state.vote_count, 1U);
    }
}

TEST(ClassTransitionGuard, CarTruckPedestrianAndBicycleAreUntouched) {
    guard::State state;
    for (const int source : {kCar, kTruck, kPedestrian, kBicycle}) {
        for (const int target : {kCar, kTruck, kPedestrian, kBicycle}) {
            ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, source,
                kObstacle, target, kVehicle, 1.0), target);
        }
    }
    for (const int non_vehicle : {kPedestrian, kBicycle}) {
        ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, kUnknown,
            kObstacle, non_vehicle, kVehicle, 1.0), non_vehicle);
        ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, non_vehicle,
            kVehicle, kUnknown, kObstacle, 1.0), kUnknown);
    }
}

TEST(ClassTransitionGuard, TentativeOrSimilarSizeTransitionIsUntouched) {
    guard::State state;
    ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, false, kUnknown,
        kObstacle, kCar, kVehicle, 1.0), kCar);
    const guard::Dimensions similar{{1.4, 1.3, 1.2}};
    ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, kUnknown,
        kObstacle, kCar, similar, 1.1), kCar);
}

TEST(ClassTransitionGuard, FootprintAxisSwapIsNotDiscontinuity) {
    guard::State state;
    const guard::Dimensions swapped{{1.8, 4.6, 1.5}};
    ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, kUnknown,
        kVehicle, kCar, swapped, 1.0), kCar);
}

TEST(ClassTransitionGuard, HeightDiscontinuityIsGuarded) {
    guard::State state;
    const guard::Dimensions tall{{1.2, 1.3, 2.0}};
    EXPECT_TRUE(guard::Evaluate(EnabledConfig(), state, true, kUnknown,
        kObstacle, kCar, tall, 1.0).pending);
}

TEST(ClassTransitionGuard, LowAndNonFiniteDimensionsFallBackAndClearEvidence) {
    guard::State state;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const double invalid : {0.0, -1.0, 0.1, nan, inf}) {
        for (const bool invalid_track : {false, true}) {
            ASSERT_TRUE(Observe(state, kCar, 1.0).pending);
            auto dimensions = kVehicle;
            dimensions[2] = invalid;
            ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, kUnknown,
                invalid_track ? dimensions : kObstacle, kCar,
                invalid_track ? kVehicle : dimensions, 1.1), kCar);
            EXPECT_EQ(state.phase, guard::Phase::Idle);
        }
    }
}

TEST(ClassTransitionGuard, ExcessiveGapRestartsButBoundaryGapIsAccepted) {
    guard::State state;
    Observe(state, kCar, 1.0);
    Observe(state, kCar, 1.5);
    EXPECT_EQ(state.vote_count, 2U);
    const auto decision = Observe(state, kCar, 2.001);
    EXPECT_TRUE(decision.pending);
    EXPECT_EQ(state.vote_count, 1U);
}

TEST(ClassTransitionGuard, RepeatedOrBackwardsStampRestartsWithoutReusingVotes) {
    for (const double stamp : {1.0, 0.9}) {
        guard::State state;
        Observe(state, kCar, 1.0);
        EXPECT_TRUE(Observe(state, kCar, stamp).pending);
        EXPECT_EQ(state.vote_count, 1U);
    }
}

TEST(ClassTransitionGuard, NonFiniteStampFallsBackAndClearsEvidence) {
    for (const double stamp : {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()}) {
        guard::State state;
        Observe(state, kCar, 1.0);
        ExpectPassThrough(Observe(state, kCar, stamp), kCar);
        EXPECT_EQ(state.phase, guard::Phase::Idle);
    }
}

TEST(ClassTransitionGuard, ResetAndReconnectionDoNotInheritEvidence) {
    guard::State state;
    Observe(state, kCar, 1.0);
    Observe(state, kCar, 1.1);
    state.Reset();  // core calls this for an unassociated frame or track reset
    EXPECT_TRUE(Observe(state, kCar, 1.2).pending);
    EXPECT_EQ(state.vote_count, 1U);
    EXPECT_TRUE(Observe(state, kCar, 1.3).pending);
    EXPECT_TRUE(Observe(state, kCar, 1.4).released);
}

TEST(ClassTransitionGuard, ReleasedEvidenceExpiresAndOppositeFamilyCancelsIt) {
    guard::State state;
    for (int index = 0; index < 3; ++index) Observe(state, kCar, 1.0 + index * 0.1);
    EXPECT_TRUE(Observe(state, kCar, 1.8).pending);
    EXPECT_EQ(state.vote_count, 1U);
    Observe(state, kCar, 1.9);
    EXPECT_TRUE(Observe(state, kCar, 2.0).released);
    ExpectPassThrough(guard::Evaluate(EnabledConfig(), state, true, kUnknown,
        kObstacle, kUnknown, kObstacle, 2.1), kUnknown);
    EXPECT_EQ(state.phase, guard::Phase::Idle);
    EXPECT_TRUE(Observe(state, kCar, 2.2).pending);
}

TEST(ClassTransitionGuard, InvalidConfigFallsBackAndStorageRemainsBounded) {
    guard::State state;
    for (const unsigned int count : {0U, 1U, 33U, 1000000U}) {
        auto config = EnabledConfig();
        config.required_consecutive = count;
        ExpectPassThrough(Observe(state, kCar, 1.0, config), kCar);
        EXPECT_EQ(state.phase, guard::Phase::Idle);
    }
    for (const double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        auto config = EnabledConfig();
        config.maximum_gap_sec = invalid;
        ExpectPassThrough(Observe(state, kCar, 1.0, config), kCar);
        config = EnabledConfig();
        config.minimum_dimension_m = invalid;
        ExpectPassThrough(Observe(state, kCar, 1.0, config), kCar);
        config = EnabledConfig();
        config.shape_ratio = invalid;
        ExpectPassThrough(Observe(state, kCar, 1.0, config), kCar);
    }
    auto config = EnabledConfig();
    config.required_consecutive = guard::kMaximumPendingVotes;
    for (std::size_t index = 0; index < guard::kMaximumPendingVotes; ++index) {
        const auto decision = Observe(state, kCar, 1.0 + index * 0.1, config);
        EXPECT_EQ(decision.released, index + 1U == guard::kMaximumPendingVotes);
        EXPECT_LE(decision.vote_count, guard::kMaximumPendingVotes);
    }
}

TEST(ClassTransitionGuard, CorruptPendingCountIsRejectedSafely) {
    guard::State state;
    Observe(state, kCar, 1.0);
    state.vote_count = guard::kMaximumPendingVotes + 1U;
    ExpectPassThrough(Observe(state, kCar, 1.1), kCar);
    EXPECT_EQ(state.phase, guard::Phase::Idle);
}

TEST(ClassTransitionGuard, InvalidStoredShapeCannotReleaseOldEvidence) {
    guard::State state;
    Observe(state, kCar, 1.0);
    Observe(state, kCar, 1.1);
    state.candidate_dimensions[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(Observe(state, kCar, 1.2).pending);
    EXPECT_EQ(state.vote_count, 1U);
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
