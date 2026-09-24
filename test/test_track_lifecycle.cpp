#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "algorithm/ekf_multi_object_tracking.hpp"

namespace {

mc_mot::Meastruct Measurement(
    const double timestamp,
    const double x,
    const double y,
    const mc_mot::ObjectClass classification = mc_mot::ObjectClass::UNKNOWN,
    const double length = 0.5,
    const double width = 2.1,
    const double velocity_x = 0.0) {
  mc_mot::Meastruct measurement;
  measurement.classification = classification;
  measurement.dimension.length = length;
  measurement.dimension.width = width;
  measurement.dimension.height = 1.0;
  measurement.state.time_stamp = timestamp;
  measurement.state.x = x;
  measurement.state.y = y;
  measurement.state.z = 0.0;
  measurement.state.yaw = 0.0;
  measurement.state.v_x = velocity_x;
  measurement.state.v_y = 0.0;
  measurement.has_velocity = std::abs(velocity_x) > 0.0;
  return measurement;
}

void Update(
    EkfMultiObjectTracking& tracker,
    const double timestamp,
    const std::vector<mc_mot::Meastruct>& measurements) {
  mc_mot::Meastructs frame;
  frame.time_stamp = timestamp;
  frame.meas = measurements;
  tracker.RunUpdate(frame);
}

void PredictAndUpdate(
    EkfMultiObjectTracking& tracker,
    double& previous_timestamp,
    const double timestamp,
    const std::vector<mc_mot::Meastruct>& measurements) {
  tracker.RunPrediction(timestamp - previous_timestamp);
  Update(tracker, timestamp, measurements);
  previous_timestamp = timestamp;
}

std::vector<mc_mot::TrackStruct> InitializedTracks(
    const EkfMultiObjectTracking& tracker) {
  std::vector<mc_mot::TrackStruct> tracks;
  for (const auto& track : tracker.GetTrackResults().track) {
    if (track.is_init) tracks.push_back(track);
  }
  return tracks;
}

int ConfirmedTrackId(const EkfMultiObjectTracking& tracker) {
  for (const auto& track : InitializedTracks(tracker)) {
    if (track.is_confirmed) return track.track_id;
  }
  return -1;
}

void ConfirmStationaryTrack(
    EkfMultiObjectTracking& tracker,
    double& timestamp) {
  timestamp = 1.0;
  Update(tracker, timestamp, {Measurement(timestamp, 0.0, 0.0)});
  PredictAndUpdate(
      tracker, timestamp, 1.1, {Measurement(1.1, 0.0, 0.0)});
  PredictAndUpdate(
      tracker, timestamp, 1.2, {Measurement(1.2, 0.0, 0.0)});
  ASSERT_GE(ConfirmedTrackId(tracker), 0);
}

}  // namespace

TEST(TrackLifecycle, ConfirmedStationaryTrackSurvivesBoundedGapAndReusesId) {
  MultiClassObjectTrackingConfig config;
  config.time_aware_track_lifecycle = true;
  config.confirmed_stationary_track_max_coast_time_sec = 1.0;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  double timestamp = 0.0;
  ConfirmStationaryTrack(*tracker, timestamp);
  const int original_id = ConfirmedTrackId(*tracker);

  for (const double next : {1.4, 1.6, 1.8, 2.0}) {
    PredictAndUpdate(*tracker, timestamp, next, {});
    EXPECT_EQ(ConfirmedTrackId(*tracker), original_id);
  }
  PredictAndUpdate(
      *tracker, timestamp, 2.1, {Measurement(2.1, 0.0, 0.0)});
  EXPECT_EQ(ConfirmedTrackId(*tracker), original_id);
  EXPECT_EQ(InitializedTracks(*tracker).size(), 1U);
}

TEST(TrackLifecycle, ConfirmedStationaryTrackExpiresAfterCoastLimit) {
  MultiClassObjectTrackingConfig config;
  config.time_aware_track_lifecycle = true;
  config.confirmed_stationary_track_max_coast_time_sec = 1.0;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  double timestamp = 0.0;
  ConfirmStationaryTrack(*tracker, timestamp);
  PredictAndUpdate(*tracker, timestamp, 2.21, {});
  EXPECT_TRUE(InitializedTracks(*tracker).empty());
}

TEST(TrackLifecycle, DuplicateLargeObjectBirthIsSuppressedNearMatureTrack) {
  MultiClassObjectTrackingConfig config;
  config.suppress_duplicate_track_birth = true;
  config.duplicate_birth_suppression_distance_m = 0.75;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  double timestamp = 0.0;
  ConfirmStationaryTrack(*tracker, timestamp);
  PredictAndUpdate(
      *tracker,
      timestamp,
      1.3,
      {
          // Put the unmatched duplicate first to guard against input-order
          // dependent births before the mature track is marked associated.
          Measurement(1.3, 0.30, 0.0, mc_mot::ObjectClass::UNKNOWN, 0.7, 1.5),
          Measurement(1.3, 0.0, 0.0),
      });
  EXPECT_EQ(InitializedTracks(*tracker).size(), 1U);
}

TEST(TrackLifecycle, DistinctOrSmallNearbyObjectsMayStartTracks) {
  MultiClassObjectTrackingConfig config;
  config.suppress_duplicate_track_birth = true;
  config.duplicate_birth_suppression_distance_m = 0.75;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  double timestamp = 0.0;
  ConfirmStationaryTrack(*tracker, timestamp);
  PredictAndUpdate(
      *tracker,
      timestamp,
      1.3,
      {
          Measurement(1.3, 0.0, 0.0),
          Measurement(1.3, 1.20, 0.0),
          Measurement(1.3, 0.30, 0.0, mc_mot::ObjectClass::UNKNOWN, 0.4, 0.6),
      });
  EXPECT_EQ(InitializedTracks(*tracker).size(), 3U);
}

TEST(TrackLifecycle, LegacyModeStillUsesFrameCountDeletion) {
  MultiClassObjectTrackingConfig config;
  config.time_aware_track_lifecycle = false;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  double timestamp = 0.0;
  ConfirmStationaryTrack(*tracker, timestamp);
  for (const double next : {1.3, 1.4, 1.5}) {
    PredictAndUpdate(*tracker, timestamp, next, {});
  }
  EXPECT_TRUE(InitializedTracks(*tracker).empty());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
