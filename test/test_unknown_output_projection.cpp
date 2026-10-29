#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "algorithm/ekf_multi_object_tracking.hpp"

namespace {
MultiClassObjectTrackingConfig Config(bool hold) {
  MultiClassObjectTrackingConfig config;
  config.hold_unknown_output_pose = hold;
  config.global_coord_track = true;
  config.max_association_dist_m = 5.;
  config.prediction_model = mc_mot::CTRV;
  config.meas_noise_std_xy_m = .4;
  config.meas_noise_std_yaw_deg = 10.;
  config.system_noise_std_xy_m = .01;
  config.system_noise_std_vx_vy_ms = 1.;
  config.system_noise_std_yaw_rate_degs = 1.;
  return config;
}

mc_mot::Meastruct Measurement(double time, double x, mc_mot::ObjectClass label) {
  mc_mot::Meastruct measurement;
  measurement.state = {};
  measurement.state.time_stamp = time;
  measurement.state.x = x;
  measurement.classification = label;
  measurement.dimension.length = 1.2;
  measurement.dimension.width = .8;
  measurement.dimension.height = 1.;
  measurement.detection_confidence = .8;
  return measurement;
}

void Step(EkfMultiObjectTracking & tracker, double time, double x,
          mc_mot::ObjectClass label = mc_mot::UNKNOWN) {
  const double previous = tracker.GetTrackResults().time_stamp;
  if (previous > 0.) tracker.RunPrediction(time - previous);
  mc_mot::Meastructs measurements;
  measurements.time_stamp = time;
  measurements.meas = {Measurement(time, x, label)};
  tracker.RunUpdate(measurements);
}

void Train(EkfMultiObjectTracking & tracker, mc_mot::ObjectClass label = mc_mot::UNKNOWN) {
  for (int i = 0; i < 30; ++i) {
    const double time = 1. + .1 * i;
    Step(tracker, time, 2. * (time - 1.), label);
  }
}

mc_mot::TrackStruct Active(const mc_mot::TrackStructs & tracks) {
  for (const auto & track : tracks.track) if (track.is_init) return track;
  ADD_FAILURE() << "No active track";
  return {};
}

void SameInternalState(const EkfMultiObjectTracking & one,
                       const EkfMultiObjectTracking & two) {
  const auto a = one.GetTrackResults();
  const auto b = two.GetTrackResults();
  ASSERT_DOUBLE_EQ(a.time_stamp, b.time_stamp);
  ASSERT_EQ(a.track.size(), b.track.size());
  for (std::size_t i = 0; i < a.track.size(); ++i) {
    const auto & x = a.track[i];
    const auto & y = b.track[i];
    ASSERT_EQ(x.is_init, y.is_init);
    if (!x.is_init) continue;
    EXPECT_EQ(x.track_id, y.track_id);
    EXPECT_EQ(x.is_confirmed, y.is_confirmed);
    EXPECT_EQ(x.is_associated, y.is_associated);
    EXPECT_EQ(x.last_measurement_time, y.last_measurement_time);
    EXPECT_EQ(x.age, y.age);
    EXPECT_TRUE((x.state_vec.array() == y.state_vec.array()).all());
    EXPECT_TRUE((x.state_cov.array() == y.state_cov.array()).all());
    EXPECT_EQ(x.getRepClass(), y.getRepClass());
  }
}
}  // namespace

TEST(UnknownOutputProjection, CandidateDefaultsOff) {
  EXPECT_FALSE(MultiClassObjectTrackingConfig{}.hold_unknown_output_pose);
}

TEST(UnknownOutputProjection, UnknownCopyRetainsMeasurementPoseNotInternalFalseVelocity) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config(false));
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  Train(*baseline);
  Train(*candidate);
  const auto measurement = candidate->GetTrackResults();
  const auto actual = Active(measurement);
  ASSERT_GT(actual.state_vec(S_VX), 1.);
  for (double age : {.001, .03, .10, .25, .50}) {
    const double target = measurement.time_stamp + age;
    const auto original = baseline->GetPredictedTrackResults(target);
    const auto output = candidate->GetPredictedTrackResults(target);
    const auto held = Active(output);
    EXPECT_DOUBLE_EQ(output.time_stamp, original.time_stamp);
    EXPECT_DOUBLE_EQ(output.time_stamp, target);
    EXPECT_DOUBLE_EQ(held.state_vec(S_X), actual.state_vec(S_X));
    EXPECT_DOUBLE_EQ(held.state_vec(S_Y), actual.state_vec(S_Y));
    EXPECT_DOUBLE_EQ(held.state_vec(S_YAW), actual.state_vec(S_YAW));
    EXPECT_DOUBLE_EQ(held.last_measurement_time, actual.last_measurement_time);
    EXPECT_EQ(held.is_associated, actual.is_associated);
    EXPECT_EQ(held.track_id, actual.track_id);
    // Holding a pose does not pretend that uncertainty or age disappeared.
    EXPECT_TRUE((held.state_cov.array() == Active(original).state_cov.array()).all());
    EXPECT_GT(Active(original).state_vec(S_X), held.state_vec(S_X));
    SameInternalState(*baseline, *candidate);
  }
}

TEST(UnknownOutputProjection, FreshMeasurementCanStillMoveAndInternalStateIsUnchanged) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config(false));
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  Train(*baseline);
  Train(*candidate);
  const auto before = Active(candidate->GetTrackResults());
  candidate->GetPredictedTrackResults(4.2);
  Step(*baseline, 4., 6.1);
  Step(*candidate, 4., 6.1);
  SameInternalState(*baseline, *candidate);
  const auto fresh = Active(candidate->GetTrackResults());
  EXPECT_GT(fresh.state_vec(S_X), before.state_vec(S_X));
  const auto output = candidate->GetPredictedTrackResults(4.2);
  EXPECT_DOUBLE_EQ(Active(output).state_vec(S_X), fresh.state_vec(S_X));
}

TEST(UnknownOutputProjection, ReclassificationResumesProjectionWithoutResetOrVelocityLoss) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config(false));
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  Train(*baseline);
  Train(*candidate);
  const int id = Active(candidate->GetTrackResults()).track_id;
  for (int i = 0; i < 12; ++i) {
    const double time = 4. + .1 * i;
    Step(*baseline, time, 2. * (time - 1.), mc_mot::CAR);
    Step(*candidate, time, 2. * (time - 1.), mc_mot::CAR);
    SameInternalState(*baseline, *candidate);
  }
  const auto measured = Active(candidate->GetTrackResults());
  ASSERT_EQ(measured.getRepClass(), mc_mot::CAR);
  ASSERT_EQ(measured.track_id, id);
  const double target = candidate->GetTrackResults().time_stamp + .2;
  const auto output = Active(candidate->GetPredictedTrackResults(target));
  const auto reference = Active(baseline->GetPredictedTrackResults(target));
  EXPECT_TRUE((output.state_vec.array() == reference.state_vec.array()).all());
  EXPECT_TRUE((output.state_cov.array() == reference.state_cov.array()).all());
  EXPECT_GT(output.state_vec(S_X), measured.state_vec(S_X));
}

TEST(UnknownOutputProjection, CarsTrucksAndPedestriansRemainExactlyEqual) {
  for (const auto label : {mc_mot::CAR, mc_mot::TRUCK, mc_mot::PEDESTRIAN}) {
    auto baseline = std::make_unique<EkfMultiObjectTracking>(Config(false));
    auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
    Train(*baseline, label);
    Train(*candidate, label);
    SameInternalState(*baseline, *candidate);
    const auto target = candidate->GetTrackResults().time_stamp + .5;
    const auto output = Active(candidate->GetPredictedTrackResults(target));
    const auto reference = Active(baseline->GetPredictedTrackResults(target));
    EXPECT_TRUE((output.state_vec.array() == reference.state_vec.array()).all());
    EXPECT_TRUE((output.state_cov.array() == reference.state_cov.array()).all());
  }
}

TEST(UnknownOutputProjection, RelativeCoordinatesDoNotFreezeAWorldStaticObstacle) {
  auto base_config = Config(false), held_config = Config(true);
  base_config.global_coord_track = held_config.global_coord_track = false;
  auto baseline = std::make_unique<EkfMultiObjectTracking>(base_config);
  auto candidate = std::make_unique<EkfMultiObjectTracking>(held_config);
  Train(*baseline);
  Train(*candidate);
  const double target = candidate->GetTrackResults().time_stamp + .25;
  const auto output = Active(candidate->GetPredictedTrackResults(target));
  const auto reference = Active(baseline->GetPredictedTrackResults(target));
  EXPECT_TRUE((output.state_vec.array() == reference.state_vec.array()).all());
}

TEST(UnknownOutputProjection, ZeroOrPastTargetDoesNotRewriteMeasurementTimestamp) {
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  Train(*candidate);
  const auto measured = candidate->GetTrackResults();
  for (double target : {measured.time_stamp, measured.time_stamp - .1}) {
    const auto output = candidate->GetPredictedTrackResults(target);
    EXPECT_DOUBLE_EQ(output.time_stamp, measured.time_stamp);
    EXPECT_TRUE((Active(output).state_vec.array() == Active(measured).state_vec.array()).all());
  }
}

TEST(StationaryUnknownPrediction, CandidateDefaultsOff) {
  EXPECT_FALSE(MultiClassObjectTrackingConfig{}.stationary_unknown_prediction);
}

TEST(StationaryUnknownPrediction, KnownClassHistoryProtectsFastReclassifiedVehicleThroughDropout) {
  for (const auto known_class : {mc_mot::CAR, mc_mot::TRUCK, mc_mot::BICYCLE}) {
  SCOPED_TRACE(static_cast<int>(known_class));
  for (double speed : {10., 20.}) {
    auto off_config = Config(false), on_config = Config(false);
    on_config.stationary_unknown_prediction = true;
    auto baseline = std::make_unique<EkfMultiObjectTracking>(off_config);
    auto candidate = std::make_unique<EkfMultiObjectTracking>(on_config);
    for (int i = 0; i < 60; ++i) {
      const double time = 1. + i*.1;
      const auto label = i < 50 ? known_class : mc_mot::UNKNOWN;
      Step(*baseline, time, speed*(time-1.), label);
      Step(*candidate, time, speed*(time-1.), label);
      SameInternalState(*baseline, *candidate);
    }
    const auto before = Active(candidate->GetTrackResults());
    ASSERT_EQ(before.getRepClass(), mc_mot::UNKNOWN);
    ASSERT_TRUE(before.has_known_class_measurement);
    for (double gap : {.3, .5, 1.}) {
      const auto output = Active(candidate->GetPredictedTrackResults(6.9+gap));
      EXPECT_TRUE(output.has_known_class_measurement);
      const auto expected = Active(baseline->GetPredictedTrackResults(6.9+gap));
      EXPECT_TRUE((output.state_vec.array() == expected.state_vec.array()).all());
      EXPECT_TRUE((output.state_cov.array() == expected.state_cov.array()).all());
    }
    Step(*baseline, 7.4, speed*6.4, known_class);
    Step(*candidate, 7.4, speed*6.4, known_class);
    SameInternalState(*baseline, *candidate);
    EXPECT_EQ(Active(candidate->GetTrackResults()).track_id, before.track_id);
  }
  }
}

TEST(StationaryUnknownPrediction, KnownClassHistoryResetsWhenTrackSlotIsReused) {
  mc_mot::TrackStruct track;
  EXPECT_FALSE(track.has_known_class_measurement);
  track.has_known_class_measurement = true;
  track.reset();
  EXPECT_FALSE(track.has_known_class_measurement);
  for (const auto label : {mc_mot::CAR, mc_mot::TRUCK, mc_mot::PEDESTRIAN, mc_mot::BICYCLE}) {
    auto config = Config(false);
    config.stationary_unknown_prediction = true;
    auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
    Step(*tracker, 1., 0., label);
    EXPECT_TRUE(Active(tracker->GetTrackResults()).has_known_class_measurement);
    EXPECT_TRUE(Active(tracker->GetPredictedTrackResults(1.1)).has_known_class_measurement);
  }
}

TEST(StationaryUnknownPrediction, OneKnownClassMeasurementProtectsEvenBeforeClassEmaSwitch) {
  auto config = Config(false);
  config.stationary_unknown_prediction = true;
  config.detection_velocity_fusion_mode = 2;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
  for (int i=0; i<10; ++i) Step(*tracker, 1.+i*.1, 0.);
  tracker->RunPrediction(.1);
  auto known = Measurement(2., .2, mc_mot::CAR);
  known.has_velocity = true;
  known.state.v_x = 2.;
  mc_mot::Meastructs frame;
  frame.time_stamp = 2.;
  frame.meas = {known};
  tracker->RunUpdate(frame);
  const auto before = Active(tracker->GetTrackResults());
  ASSERT_EQ(before.getRepClass(), mc_mot::UNKNOWN);
  EXPECT_TRUE(before.has_known_class_measurement);
  const auto predicted = Active(tracker->GetPredictedTrackResults(2.2));
  EXPECT_GT(predicted.state_vec(S_X), before.state_vec(S_X));
}

TEST(StationaryUnknownPrediction, HiddenVelocityDoesNotAdvectPoseAndCovarianceGrows) {
  auto config = Config(false);
  config.stationary_unknown_prediction = true;
  config.detection_velocity_fusion_mode = 1;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
  auto measurement = Measurement(1., 0., mc_mot::UNKNOWN);
  measurement.has_velocity = true;
  measurement.state.v_x = 8.;
  mc_mot::Meastructs frame;
  frame.time_stamp = 1.;
  frame.meas = {measurement};
  tracker->RunUpdate(frame);
  const auto before = Active(tracker->GetTrackResults());
  ASSERT_DOUBLE_EQ(before.state_vec(S_VX), 8.);
  const auto output = tracker->GetPredictedTrackResults(1.5);
  EXPECT_DOUBLE_EQ(Active(output).state_vec(S_X), before.state_vec(S_X));
  // Latent state is not destroyed or silently interpreted as a fresh speed measurement.
  EXPECT_DOUBLE_EQ(Active(output).state_vec(S_VX), before.state_vec(S_VX));
  EXPECT_GT(Active(output).state_cov(S_X, S_X), before.state_cov(S_X, S_X) + .24);
  tracker->RunPrediction(.5);
  const auto after = Active(tracker->GetTrackResults());
  EXPECT_DOUBLE_EQ(after.state_vec(S_X), before.state_vec(S_X));
  EXPECT_DOUBLE_EQ(after.update_time, before.update_time + .5);
  EXPECT_TRUE((after.state_cov.array() == Active(output).state_cov.array()).all());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix8_8d> solver(after.state_cov);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1.e-9);
}

TEST(StationaryUnknownPrediction, FreshMeasurementsConvergeAfterAnInitialBiasedCenter) {
  auto config = Config(false);
  config.stationary_unknown_prediction = true;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
  Step(*tracker, 1., 1.2);
  const int original = Active(tracker->GetTrackResults()).track_id;
  for (int i = 1; i <= 30; ++i) Step(*tracker, 1. + i*.1, 0.);
  const auto latest = Active(tracker->GetTrackResults());
  EXPECT_EQ(latest.track_id, original);
  EXPECT_LT(std::abs(latest.state_vec(S_X)), .1);
  EXPECT_TRUE(latest.is_confirmed);
  EXPECT_DOUBLE_EQ(Active(tracker->GetPredictedTrackResults(4.2)).state_vec(S_X), latest.state_vec(S_X));
}

TEST(StationaryUnknownPrediction, DistinctNearbySmallStaticObjectsBothConfirm) {
  auto config = Config(false);
  config.stationary_unknown_prediction = true;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
  for (int i = 0; i < 20; ++i) {
    const double time = 1. + .1*i;
    auto first = Measurement(time, 0., mc_mot::UNKNOWN);
    auto second = Measurement(time, .5, mc_mot::UNKNOWN);
    first.dimension.length = first.dimension.width = .3;
    second.dimension = first.dimension;
    const double previous = tracker->GetTrackResults().time_stamp;
    if (previous > 0.) tracker->RunPrediction(time - previous);
    mc_mot::Meastructs frame;
    frame.time_stamp = time;
    frame.meas = {first, second};
    tracker->RunUpdate(frame);
  }
  int confirmed = 0;
  for (const auto & track : tracker->GetTrackResults().track) confirmed += track.is_init && track.is_confirmed;
  EXPECT_EQ(confirmed, 2);
}

TEST(StationaryUnknownPrediction, MovingUnknownCanRecoverAsCarWithoutReset) {
  auto config = Config(false);
  config.stationary_unknown_prediction = true;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
  // This deliberately violates the stationary UNKNOWN assumption: a moving
  // vehicle is initially mislabeled. Do not assert equal UNKNOWN tracking;
  // verify that fresh CAR evidence can recover the original ID and velocity.
  Train(*tracker);
  const int original = Active(tracker->GetTrackResults()).track_id;
  for (int i = 0; i < 20; ++i) {
    const double time = 4. + .1 * i;
    Step(*tracker, time, 2. * (time - 1.), mc_mot::CAR);
  }
  const auto current = Active(tracker->GetTrackResults());
  EXPECT_EQ(current.track_id, original);
  EXPECT_EQ(current.getRepClass(), mc_mot::CAR);
  EXPECT_NEAR(current.state_vec(S_VX), 2., .1);
  EXPECT_LT(std::abs(current.state_vec(S_X) - 9.8), .1);
}

TEST(StationaryUnknownPrediction, KnownVehiclesAndPedestriansRemainExactlyUnchanged) {
  for (const auto label : {mc_mot::CAR, mc_mot::TRUCK, mc_mot::PEDESTRIAN}) {
    auto plain_config = Config(false), fixed_config = Config(false);
    fixed_config.stationary_unknown_prediction = true;
    auto reference = std::make_unique<EkfMultiObjectTracking>(plain_config);
    auto candidate = std::make_unique<EkfMultiObjectTracking>(fixed_config);
    Train(*reference, label);
    Train(*candidate, label);
    SameInternalState(*reference, *candidate);
    const double target = reference->GetTrackResults().time_stamp + .5;
    const auto output = Active(candidate->GetPredictedTrackResults(target));
    const auto original = Active(reference->GetPredictedTrackResults(target));
    EXPECT_TRUE((output.state_vec.array() == original.state_vec.array()).all());
    EXPECT_TRUE((output.state_cov.array() == original.state_cov.array()).all());
  }
}

TEST(UnknownCoastingOutput, DefaultOffAndStaticHistoryReset) {
  EXPECT_FALSE(MultiClassObjectTrackingConfig{}.hold_unknown_coasting_pose);
  mc_mot::TrackStruct track;
  track.has_observed_output_pose = true;
  track.last_observed_output_pose << 1., 2., 3.;
  track.reset();
  EXPECT_FALSE(track.has_observed_output_pose);
  EXPECT_EQ(track.last_observed_output_pose.norm(), 0.);
}

TEST(UnknownCoastingOutput, DropsNoTracksAndHoldsOnlyOutputUntilNextAssociation) {
  auto base_config = Config(false), hold_config = Config(false);
  hold_config.hold_unknown_coasting_pose = true;
  auto reference = std::make_unique<EkfMultiObjectTracking>(base_config);
  auto candidate = std::make_unique<EkfMultiObjectTracking>(hold_config);
  Train(*reference);
  Train(*candidate);
  const auto initial = Active(candidate->GetTrackResults());
  for (int i = 1; i <= 3; ++i) {
    reference->RunPrediction(.1);
    candidate->RunPrediction(.1);
    mc_mot::Meastructs empty;
    empty.time_stamp = initial.last_measurement_time + i * .1;
    reference->RunUpdate(empty);
    candidate->RunUpdate(empty);
    const auto raw = Active(reference->GetTrackResults());
    const auto held = Active(candidate->GetTrackResults());
    EXPECT_GT(raw.state_vec(S_X), initial.state_vec(S_X));
    EXPECT_DOUBLE_EQ(held.state_vec(S_X), initial.state_vec(S_X));
    EXPECT_DOUBLE_EQ(held.last_measurement_time, raw.last_measurement_time);
    EXPECT_EQ(held.track_id, raw.track_id);
    EXPECT_EQ(held.is_confirmed, raw.is_confirmed);
    EXPECT_TRUE((held.state_cov.array() == raw.state_cov.array()).all());
    // Disable output-only projection to compare the untouched internal filters.
    candidate->UpdateConfig(base_config);
    SameInternalState(*reference, *candidate);
    candidate->UpdateConfig(hold_config);
  }
  Step(*reference, 4.3, 7.0);
  Step(*candidate, 4.3, 7.0);
  EXPECT_GT(Active(candidate->GetTrackResults()).state_vec(S_X), initial.state_vec(S_X));
  candidate->UpdateConfig(base_config);
  SameInternalState(*reference, *candidate);
}

TEST(UnknownCoastingOutput, KnownVehicleAndPedestrianOutputsStayIdentical) {
  for (auto label : {mc_mot::CAR, mc_mot::TRUCK, mc_mot::PEDESTRIAN}) {
    auto base_config = Config(false), hold_config = Config(false);
    hold_config.hold_unknown_coasting_pose = true;
    auto reference = std::make_unique<EkfMultiObjectTracking>(base_config);
    auto candidate = std::make_unique<EkfMultiObjectTracking>(hold_config);
    Train(*reference, label);
    Train(*candidate, label);
    reference->RunPrediction(.2);
    candidate->RunPrediction(.2);
    SameInternalState(*reference, *candidate);
  }
}

TEST(UnknownCoastingOutput, PredictedCopyPreservesCovarianceAndFreshMeasurementTime) {
  auto base_config = Config(false), hold_config = Config(false);
  hold_config.hold_unknown_coasting_pose = true;
  auto reference = std::make_unique<EkfMultiObjectTracking>(base_config);
  auto candidate = std::make_unique<EkfMultiObjectTracking>(hold_config);
  for (int i = 0; i < 30; ++i) {
    const double stamp = 1. + i * .1;
    for (auto *tracker : {reference.get(), candidate.get()}) {
      if (i) tracker->RunPrediction(.1);
      mc_mot::Meastructs frame;
      frame.time_stamp = stamp;
      auto measurement = Measurement(stamp, i * .2, mc_mot::UNKNOWN);
      measurement.state.yaw = i * .02;
      frame.meas = {measurement};
      tracker->RunUpdate(frame);
    }
  }
  const auto last = Active(candidate->GetTrackResults());
  for (auto *tracker : {reference.get(), candidate.get()}) tracker->RunPrediction(.3);
  const auto plain = Active(reference->GetPredictedTrackResults(4.4));
  const auto held = Active(candidate->GetPredictedTrackResults(4.4));
  EXPECT_TRUE((held.state_cov.array() == plain.state_cov.array()).all());
  EXPECT_DOUBLE_EQ(held.state_vec(S_X), last.state_vec(S_X));
  EXPECT_DOUBLE_EQ(held.state_vec(S_YAW), last.state_vec(S_YAW));
  EXPECT_DOUBLE_EQ(held.last_measurement_time, plain.last_measurement_time);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
