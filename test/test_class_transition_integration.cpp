#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "algorithm/ekf_multi_object_tracking.hpp"

namespace {

namespace guard = class_transition_guard;

mc_mot::Meastruct Measurement(
        const double stamp,
        const mc_mot::ObjectClass classification,
        const bool vehicle_shape,
        const double x = 0.0,
        const double y = 0.0) {
    mc_mot::Meastruct measurement;
    measurement.state = mc_mot::ObjectState{};
    measurement.state.time_stamp = stamp;
    measurement.state.x = x;
    measurement.state.y = y;
    measurement.classification = classification;
    measurement.dimension.length = vehicle_shape ? 4.6 : 1.2;
    measurement.dimension.width = vehicle_shape ? 1.8 : 1.3;
    measurement.dimension.height = vehicle_shape ? 1.5 : 1.0;
    measurement.detection_confidence = 0.8;
    return measurement;
}

MultiClassObjectTrackingConfig Config(const bool enabled = true) {
    MultiClassObjectTrackingConfig config;
    config.class_transition_guard_config.enabled = enabled;
    config.dimension_filter_alpha = 0.3;
    config.prediction_model = mc_mot::CTRV;
    config.max_association_dist_m = 5.0;
    config.meas_noise_std_xy_m = 0.4;
    return config;
}

class Tracker {
public:
    explicit Tracker(const MultiClassObjectTrackingConfig& config = Config())
        : tracker_(new EkfMultiObjectTracking(config)) {}

    void Step(const double stamp, const std::vector<mc_mot::Meastruct>& measurements) {
        if (stamp_ > 0.0) tracker_->RunPrediction(stamp - stamp_);
        mc_mot::Meastructs frame;
        frame.time_stamp = stamp;
        frame.meas = measurements;
        tracker_->RunUpdate(frame);
        stamp_ = stamp;
    }

    void Step(const mc_mot::Meastruct& measurement) {
        Step(measurement.state.time_stamp, {measurement});
    }

    std::vector<mc_mot::TrackStruct> Active() const {
        std::vector<mc_mot::TrackStruct> active;
        for (const auto& track : tracker_->GetTrackResults().track) {
            if (track.is_init) active.push_back(track);
        }
        return active;
    }

    mc_mot::TrackStruct One() const {
        const auto active = Active();
        if (active.size() != 1U) throw std::runtime_error("expected exactly one initialized track");
        return active.front();
    }

    mc_mot::TrackStruct Slot(const std::size_t index) const {
        return tracker_->GetTrackResults().track.at(index);
    }

    void Confirm(const mc_mot::ObjectClass classification = mc_mot::UNKNOWN,
                 const bool vehicle_shape = false) {
        for (int index = 0; index < 3; ++index) {
            Step(Measurement(1.0 + 0.1 * index, classification, vehicle_shape));
        }
        ASSERT_EQ(Active().size(), 1U);
        ASSERT_TRUE(One().is_confirmed);
    }

private:
    std::unique_ptr<EkfMultiObjectTracking> tracker_;
    double stamp_{0.0};
};

void ExpectDimensionsEqual(const mc_mot::ObjectDimension& first,
                           const mc_mot::ObjectDimension& second) {
    EXPECT_EQ(first.length, second.length);
    EXPECT_EQ(first.width, second.width);
    EXPECT_EQ(first.height, second.height);
}

void ExpectExactTrack(const mc_mot::TrackStruct& first, const mc_mot::TrackStruct& second) {
    EXPECT_EQ(first.track_id, second.track_id);
    EXPECT_EQ(first.initialization_time, second.initialization_time);
    EXPECT_EQ(first.update_time, second.update_time);
    EXPECT_EQ(first.last_measurement_time, second.last_measurement_time);
    EXPECT_EQ(first.age, second.age);
    EXPECT_EQ(first.is_init, second.is_init);
    EXPECT_EQ(first.is_confirmed, second.is_confirmed);
    EXPECT_EQ(first.is_associated, second.is_associated);
    EXPECT_EQ(first.detection_confidence, second.detection_confidence);
    EXPECT_EQ(first.object_z, second.object_z);
    EXPECT_EQ(first.direction_score, second.direction_score);
    EXPECT_EQ(first.getRepClass(), second.getRepClass());
    EXPECT_EQ(first.getRepClassProb(), second.getRepClassProb());
    ExpectDimensionsEqual(first.dimension, second.dimension);
    for (int index = 0; index < CLASS_NUM; ++index) {
        EXPECT_EQ(first.class_score_arr[index], second.class_score_arr[index]);
    }
    for (int row = 0; row < 8; ++row) {
        EXPECT_EQ(first.state_vec(row), second.state_vec(row));
        for (int column = 0; column < 8; ++column) {
            EXPECT_EQ(first.state_cov(row, column), second.state_cov(row, column));
        }
    }
    for (int index = 0; index < MAX_HISTORY; ++index) {
        EXPECT_EQ(first.detection_arr[index], second.detection_arr[index]);
    }
}

TEST(ClassTransitionIntegration, DefaultOffMatchesExplicitOffAndOriginalAttributeEmaFor200Frames) {
    auto default_config = Config();
    default_config.class_transition_guard_config = guard::Config{};
    ASSERT_FALSE(default_config.class_transition_guard_config.enabled);
    auto explicit_config = default_config;
    explicit_config.class_transition_guard_config.enabled = false;
    explicit_config.class_transition_guard_config.required_consecutive = 32U;
    explicit_config.class_transition_guard_config.shape_ratio = 1.1;
    Tracker default_off(default_config);
    Tracker explicit_off(explicit_config);
    std::array<double, CLASS_NUM> scores{{1.0, 0.0, 0.0, 0.0, 0.0}};
    mc_mot::ObjectDimension dimensions;
    for (int index = 0; index < 200; ++index) {
        SCOPED_TRACE(index);
        const int phase = index % 13;
        const auto label = phase < 4 ? mc_mot::UNKNOWN : phase < 9 ? mc_mot::CAR : mc_mot::TRUCK;
        auto measurement = Measurement(1.0 + 0.1 * index, label, phase >= 4,
                                       0.03 * index, 0.1 * std::sin(0.1 * index));
        measurement.state.yaw = 0.03 * std::sin(0.1 * index);
        measurement.state.z = 0.02 * index;
        measurement.detection_confidence = 0.4 + 0.002 * index;
        default_off.Step(measurement);
        explicit_off.Step(measurement);
        const auto actual = default_off.One();
        ExpectExactTrack(actual, explicit_off.One());
        ASSERT_EQ(actual.track_id, 0);
        if (index == 0) {
            dimensions = measurement.dimension;
        } else {
            dimensions.length = 0.3 * measurement.dimension.length + 0.7 * dimensions.length;
            dimensions.width = 0.3 * measurement.dimension.width + 0.7 * dimensions.width;
            dimensions.height = 0.3 * measurement.dimension.height + 0.7 * dimensions.height;
            for (int classification = 0; classification < CLASS_NUM; ++classification) {
                scores[classification] = 0.8 * scores[classification] +
                    (classification == label ? 0.2 : 0.0);
            }
        }
        ExpectDimensionsEqual(actual.dimension, dimensions);
        for (int classification = 0; classification < CLASS_NUM; ++classification) {
            EXPECT_EQ(actual.class_score_arr[classification], scores[classification]);
        }
        EXPECT_EQ(actual.class_transition_state.phase, guard::Phase::Idle);
    }
}

TEST(ClassTransitionIntegration, StationaryConfirmedUnknownRecoversToCarWithinFourObservations) {
    Tracker tracker;
    tracker.Confirm();
    const auto original = tracker.One();
    for (int index = 0; index < 4; ++index) {
        tracker.Step(Measurement(1.3 + 0.1 * index, mc_mot::CAR, true));
        const auto track = tracker.One();
        EXPECT_EQ(track.track_id, original.track_id);
        EXPECT_TRUE(track.is_confirmed);
        EXPECT_DOUBLE_EQ(track.state_vec(S_VX), 0.0);
        EXPECT_DOUBLE_EQ(track.state_vec(S_VY), 0.0);
        if (index < 2) {
            EXPECT_EQ(track.getRepClass(), mc_mot::UNKNOWN);
            ExpectDimensionsEqual(track.dimension, original.dimension);
            EXPECT_EQ(track.class_score_arr[mc_mot::CAR], 0.0);
        }
    }
    const auto corrected = tracker.One();
    EXPECT_EQ(corrected.getRepClass(), mc_mot::CAR);
    EXPECT_NEAR(corrected.class_score_arr[mc_mot::CAR], 0.5904, 1.e-12);
    EXPECT_GT(corrected.dimension.length, original.dimension.length);
}

TEST(ClassTransitionIntegration, PendingFlickerHoldsShapeAndClassButStillUpdatesPosition) {
    Tracker tracker;
    tracker.Confirm();
    const auto original = tracker.One();
    for (int index = 0; index < 40; ++index) {
        const bool vehicle = index % 2 == 0;
        tracker.Step(Measurement(1.3 + 0.1 * index,
            vehicle ? mc_mot::CAR : mc_mot::UNKNOWN, vehicle, 0.02 * (index + 1)));
        const auto track = tracker.One();
        EXPECT_EQ(track.track_id, original.track_id);
        EXPECT_EQ(track.getRepClass(), mc_mot::UNKNOWN);
        EXPECT_EQ(track.class_score_arr[mc_mot::CAR], 0.0);
        EXPECT_EQ(track.class_score_arr[mc_mot::UNKNOWN], 1.0);
        ExpectDimensionsEqual(track.dimension, original.dimension);
        EXPECT_GT(track.state_vec(S_X), original.state_vec(S_X));
        EXPECT_EQ(track.last_measurement_time, 1.3 + 0.1 * index);
        EXPECT_EQ(track.class_transition_state.phase,
                  vehicle ? guard::Phase::Pending : guard::Phase::Idle);
    }
}

TEST(ClassTransitionIntegration, PendingGeometryDoesNotChangeTheOrdinaryEkfCorrection) {
    Tracker guarded;
    Tracker unguarded(Config(false));
    guarded.Confirm();
    unguarded.Confirm();
    for (int index = 0; index < 2; ++index) {
        const auto observation = Measurement(1.3 + index * 0.1, mc_mot::CAR, true,
                                            0.1 + index * 0.1, 0.05);
        guarded.Step(observation);
        unguarded.Step(observation);
        const auto held = guarded.One();
        const auto normal = unguarded.One();
        for (int row = 0; row < 8; ++row) {
            EXPECT_EQ(held.state_vec(row), normal.state_vec(row));
            for (int column = 0; column < 8; ++column) {
                EXPECT_EQ(held.state_cov(row, column), normal.state_cov(row, column));
            }
        }
        EXPECT_EQ(held.detection_confidence, normal.detection_confidence);
        EXPECT_EQ(held.object_z, normal.object_z);
        EXPECT_NE(held.dimension.length, normal.dimension.length);
    }
}

TEST(ClassTransitionIntegration, ConfirmedCarEventuallyAdoptsUnknownWithoutIdChange) {
    Tracker tracker;
    tracker.Confirm(mc_mot::CAR, true);
    const int id = tracker.One().track_id;
    for (int index = 0; index < 4; ++index) {
        tracker.Step(Measurement(1.3 + 0.1 * index, mc_mot::UNKNOWN, false));
        EXPECT_EQ(tracker.One().track_id, id);
        if (index < 2) {
            EXPECT_EQ(tracker.One().getRepClass(), mc_mot::CAR);
        }
    }
    EXPECT_EQ(tracker.One().getRepClass(), mc_mot::UNKNOWN);
    EXPECT_NEAR(tracker.One().class_score_arr[mc_mot::UNKNOWN], 0.5904, 1.e-12);
}

TEST(ClassTransitionIntegration, UnassociatedFrameClearsPendingEvidenceBeforeReacquisition) {
    Tracker tracker;
    tracker.Confirm();
    const int id = tracker.One().track_id;
    tracker.Step(Measurement(1.3, mc_mot::CAR, true));
    tracker.Step(Measurement(1.4, mc_mot::CAR, true));
    ASSERT_EQ(tracker.One().class_transition_state.vote_count, 2U);
    tracker.Step(1.5, {});
    ASSERT_EQ(tracker.One().track_id, id);
    EXPECT_EQ(tracker.One().class_transition_state.phase, guard::Phase::Idle);
    EXPECT_EQ(tracker.One().class_transition_state.vote_count, 0U);
    tracker.Step(Measurement(1.6, mc_mot::CAR, true));
    EXPECT_EQ(tracker.One().track_id, id);
    EXPECT_EQ(tracker.One().class_transition_state.vote_count, 1U);
    EXPECT_EQ(tracker.One().class_score_arr[mc_mot::CAR], 0.0);
    tracker.Step(Measurement(1.7, mc_mot::CAR, true));
    EXPECT_EQ(tracker.One().class_transition_state.vote_count, 2U);
    tracker.Step(Measurement(1.8, mc_mot::CAR, true));
    EXPECT_EQ(tracker.One().class_transition_state.phase, guard::Phase::Released);
}

TEST(ClassTransitionIntegration, LongMeasurementGapRestartsConfirmationWithoutDeletingTrack) {
    Tracker tracker;
    tracker.Confirm();
    const int id = tracker.One().track_id;
    tracker.Step(Measurement(1.3, mc_mot::CAR, true));
    tracker.Step(Measurement(1.4, mc_mot::CAR, true));
    tracker.Step(Measurement(2.0, mc_mot::CAR, true));
    const auto reconnected = tracker.One();
    EXPECT_EQ(reconnected.track_id, id);
    EXPECT_EQ(reconnected.class_transition_state.phase, guard::Phase::Pending);
    EXPECT_EQ(reconnected.class_transition_state.vote_count, 1U);
    EXPECT_EQ(reconnected.class_score_arr[mc_mot::CAR], 0.0);
}

TEST(ClassTransitionIntegration, InactiveTrackClassificationIsInitialized) {
    const mc_mot::TrackStruct fresh;
    EXPECT_EQ(fresh.classification, mc_mot::UNKNOWN);
    const mc_mot::TrackStruct copied = fresh;
    EXPECT_EQ(copied.classification, mc_mot::UNKNOWN);
}

TEST(ClassTransitionIntegration, TrackResetClearsPendingStateBeforeStructReuse) {
    Tracker tracker;
    tracker.Confirm();
    tracker.Step(Measurement(1.3, mc_mot::CAR, true));
    tracker.Step(Measurement(1.4, mc_mot::CAR, true));
    auto reused = tracker.One();
    ASSERT_EQ(reused.class_transition_state.vote_count, 2U);
    reused.reset();
    EXPECT_EQ(reused.class_transition_state.phase, guard::Phase::Idle);
    EXPECT_EQ(reused.class_transition_state.vote_count, 0U);
    const auto decision = guard::Evaluate(Config().class_transition_guard_config,
        reused.class_transition_state, true, mc_mot::UNKNOWN,
        {{1.2, 1.3, 1.0}}, mc_mot::CAR, {{4.6, 1.8, 1.5}}, 3.0);
    EXPECT_TRUE(decision.pending);
    EXPECT_EQ(reused.class_transition_state.vote_count, 1U);

    tracker.Step(2.5, {});  // measured age exceeds the stationary 1.0 s lifetime
    EXPECT_TRUE(tracker.Active().empty());
    const auto deleted_slot = tracker.Slot(0U);
    EXPECT_FALSE(deleted_slot.is_init);
    EXPECT_EQ(deleted_slot.class_transition_state.phase, guard::Phase::Idle);
    EXPECT_EQ(deleted_slot.class_transition_state.vote_count, 0U);
    tracker.Step(Measurement(2.6, mc_mot::CAR, true));
    EXPECT_EQ(tracker.One().class_transition_state.phase, guard::Phase::Idle);
    EXPECT_EQ(tracker.One().class_transition_state.vote_count, 0U);
}

TEST(ClassTransitionIntegration, FastPromotionCannotBypassPendingClassShapeConfirmation) {
    auto config = Config();
    config.fast_confirmed_vehicle_promotion = true;
    Tracker tracker(config);
    tracker.Confirm();
    const int id = tracker.One().track_id;
    for (int index = 0; index < 3; ++index) {
        auto observation = Measurement(1.3 + index * 0.1, mc_mot::CAR, true);
        observation.detection_confidence = 0.95;
        tracker.Step(observation);
        const auto track = tracker.One();
        EXPECT_EQ(track.track_id, id);
        if (index < 2) {
            EXPECT_EQ(track.getRepClass(), mc_mot::UNKNOWN);
            EXPECT_EQ(track.class_score_arr[mc_mot::CAR], 0.0);
            EXPECT_EQ(track.class_transition_state.phase, guard::Phase::Pending);
        } else {
            EXPECT_EQ(track.getRepClass(), mc_mot::CAR);
            EXPECT_EQ(track.class_transition_state.phase, guard::Phase::Released);
        }
    }
}

TEST(ClassTransitionIntegration, PedestrianShapeUpdatesAreIdenticalWithGuardOnOrOff) {
    Tracker guarded;
    Tracker unguarded(Config(false));
    guarded.Confirm(mc_mot::PEDESTRIAN);
    unguarded.Confirm(mc_mot::PEDESTRIAN);
    for (int index = 0; index < 20; ++index) {
        auto observation = Measurement(1.3 + 0.1 * index, mc_mot::PEDESTRIAN,
                                       index % 2 == 0, 0.03 * index);
        guarded.Step(observation);
        unguarded.Step(observation);
        const auto track = guarded.One();
        ExpectExactTrack(track, unguarded.One());
        EXPECT_EQ(track.getRepClass(), mc_mot::PEDESTRIAN);
        EXPECT_EQ(track.class_transition_state.phase, guard::Phase::Idle);
    }
}

TEST(ClassTransitionIntegration, CarTruckFamilyChangeDoesNotHoldLargeShapeUpdate) {
    Tracker tracker;
    tracker.Confirm(mc_mot::CAR, true);
    const auto before = tracker.One();
    auto truck = Measurement(1.3, mc_mot::TRUCK, true);
    truck.dimension.length = 12.0;
    tracker.Step(truck);
    const auto after = tracker.One();
    EXPECT_EQ(after.track_id, before.track_id);
    EXPECT_GT(after.dimension.length, before.dimension.length);
    EXPECT_DOUBLE_EQ(after.class_score_arr[mc_mot::TRUCK], 0.2);
    EXPECT_EQ(after.class_transition_state.phase, guard::Phase::Idle);
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
