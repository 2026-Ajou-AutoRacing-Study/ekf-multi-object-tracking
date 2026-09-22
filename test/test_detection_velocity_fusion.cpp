#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>

#include "algorithm/ekf_multi_object_tracking.hpp"

namespace {

mc_mot::Meastruct Measurement(
    double timestamp,
    double x,
    double y,
    bool has_velocity,
    double velocity_x,
    double velocity_y) {
  mc_mot::Meastruct measurement;
  measurement.classification = mc_mot::ObjectClass::CAR;
  measurement.dimension.length = 4.5;
  measurement.dimension.width = 1.9;
  measurement.dimension.height = 1.7;
  measurement.state.time_stamp = timestamp;
  measurement.state.x = x;
  measurement.state.y = y;
  measurement.state.z = 0.0;
  measurement.state.yaw = 0.0;
  measurement.state.v_x = velocity_x;
  measurement.state.v_y = velocity_y;
  measurement.has_velocity = has_velocity;
  return measurement;
}

const mc_mot::TrackStruct& InitializedTrack(const mc_mot::TrackStructs& tracks) {
  for (const auto& track : tracks.track) {
    if (track.is_init) return track;
  }
  throw std::runtime_error("no initialized track");
}

void Update(EkfMultiObjectTracking& tracker, const mc_mot::Meastruct& measurement) {
  mc_mot::Meastructs measurements;
  measurements.time_stamp = measurement.state.time_stamp;
  measurements.meas.push_back(measurement);
  tracker.RunUpdate(measurements);
}

}  // namespace

TEST(DetectionVelocityFusion, OffPreservesZeroVelocityInitialization) {
  MultiClassObjectTrackingConfig config;
  config.detection_velocity_fusion_mode = 0;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  Update(*tracker, Measurement(1.0, 0.0, 0.0, true, 7.0, -1.0));
  const auto tracks = tracker->GetTrackResults();
  const auto& track = InitializedTrack(tracks);

  EXPECT_DOUBLE_EQ(track.state_vec(S_VX), 0.0);
  EXPECT_DOUBLE_EQ(track.state_vec(S_VY), 0.0);
}

TEST(DetectionVelocityFusion, InitializesFromValidVelocityMeasurement) {
  MultiClassObjectTrackingConfig config;
  config.detection_velocity_fusion_mode = 1;
  config.detection_velocity_noise_std_mps = 0.25;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  Update(*tracker, Measurement(1.0, 0.0, 0.0, true, 7.0, -1.0));
  const auto tracks = tracker->GetTrackResults();
  const auto& track = InitializedTrack(tracks);

  EXPECT_DOUBLE_EQ(track.state_vec(S_VX), 7.0);
  EXPECT_DOUBLE_EQ(track.state_vec(S_VY), -1.0);
  EXPECT_NEAR(track.state_cov(S_VX, S_VX), 0.0625, 1.0e-12);
  EXPECT_NEAR(track.state_cov(S_VY, S_VY), 0.0625, 1.0e-12);
}

TEST(DetectionVelocityFusion, EveryUpdateCorrectsVelocityState) {
  MultiClassObjectTrackingConfig off_config;
  off_config.detection_velocity_fusion_mode = 0;
  auto off_tracker = std::make_unique<EkfMultiObjectTracking>(off_config);
  Update(*off_tracker, Measurement(1.0, 0.0, 0.0, false, 0.0, 0.0));
  off_tracker->RunPrediction(0.1);
  Update(*off_tracker, Measurement(1.1, 0.5, 0.0, true, 5.0, 0.0));
  const double off_velocity =
      InitializedTrack(off_tracker->GetTrackResults()).state_vec(S_VX);

  MultiClassObjectTrackingConfig update_config;
  update_config.detection_velocity_fusion_mode = 2;
  update_config.detection_velocity_noise_std_mps = 0.1;
  auto update_tracker = std::make_unique<EkfMultiObjectTracking>(update_config);
  Update(*update_tracker, Measurement(1.0, 0.0, 0.0, false, 0.0, 0.0));
  update_tracker->RunPrediction(0.1);
  Update(*update_tracker, Measurement(1.1, 0.5, 0.0, true, 5.0, 0.0));
  const double fused_velocity =
      InitializedTrack(update_tracker->GetTrackResults()).state_vec(S_VX);

  EXPECT_LT(std::abs(fused_velocity - 5.0), std::abs(off_velocity - 5.0));
  EXPECT_NEAR(fused_velocity, 5.0, 0.05);
}

TEST(CenterMotionVelocityFusion, WaitsForMinimumCausalBaseline) {
  MultiClassObjectTrackingConfig off_config;
  off_config.global_coord_track = true;
  auto off_tracker = std::make_unique<EkfMultiObjectTracking>(off_config);

  MultiClassObjectTrackingConfig q0_config = off_config;
  q0_config.center_motion_velocity_fusion = true;
  q0_config.center_motion_min_baseline_sec = 0.30;
  auto q0_tracker = std::make_unique<EkfMultiObjectTracking>(q0_config);

  const auto first = Measurement(1.0, 0.0, 0.0, false, 0.0, 0.0);
  const auto second = Measurement(1.2, 2.0, 0.0, false, 0.0, 0.0);
  Update(*off_tracker, first);
  Update(*q0_tracker, first);
  off_tracker->RunPrediction(0.2);
  q0_tracker->RunPrediction(0.2);
  Update(*off_tracker, second);
  Update(*q0_tracker, second);

  const auto off_tracks = off_tracker->GetTrackResults();
  const auto q0_tracks = q0_tracker->GetTrackResults();
  const auto& off_track = InitializedTrack(off_tracks);
  const auto& q0_track = InitializedTrack(q0_tracks);
  EXPECT_EQ(q0_track.center_history.size(), 2U);
  EXPECT_DOUBLE_EQ(q0_track.state_vec(S_VX), off_track.state_vec(S_VX));
  EXPECT_DOUBLE_EQ(
      q0_track.state_cov(S_VX, S_VX), off_track.state_cov(S_VX, S_VX));
}

TEST(CenterMotionVelocityFusion, CorrectsVehicleVelocityAfterCausalBaseline) {
  MultiClassObjectTrackingConfig off_config;
  off_config.global_coord_track = true;
  auto off_tracker = std::make_unique<EkfMultiObjectTracking>(off_config);

  MultiClassObjectTrackingConfig q0_config = off_config;
  q0_config.center_motion_velocity_fusion = true;
  q0_config.center_motion_early_noise_std_mps = 0.5;
  q0_config.center_motion_mature_noise_std_mps = 0.5;
  auto q0_tracker = std::make_unique<EkfMultiObjectTracking>(q0_config);

  for (int index = 0; index <= 6; ++index) {
    const double timestamp = 1.0 + 0.1 * index;
    const auto measurement = Measurement(
        timestamp, 1.0 * index, 0.0, false, 0.0, 0.0);
    if (index > 0) {
      off_tracker->RunPrediction(0.1);
      q0_tracker->RunPrediction(0.1);
    }
    Update(*off_tracker, measurement);
    Update(*q0_tracker, measurement);
  }

  const double off_velocity =
      InitializedTrack(off_tracker->GetTrackResults()).state_vec(S_VX);
  const double q0_velocity =
      InitializedTrack(q0_tracker->GetTrackResults()).state_vec(S_VX);
  EXPECT_LT(std::abs(q0_velocity - 10.0), std::abs(off_velocity - 10.0));
  EXPECT_NEAR(q0_velocity, 10.0, 0.5);
}

TEST(CenterMotionVelocityFusion, ResetsHistoryAcrossAssociationGap) {
  MultiClassObjectTrackingConfig config;
  config.global_coord_track = true;
  config.center_motion_velocity_fusion = true;
  config.center_motion_max_gap_sec = 0.25;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  Update(*tracker, Measurement(1.0, 0.0, 0.0, false, 0.0, 0.0));
  tracker->RunPrediction(0.4);
  Update(*tracker, Measurement(1.4, 2.0, 0.0, false, 0.0, 0.0));

  const auto tracks = tracker->GetTrackResults();
  const auto& track = InitializedTrack(tracks);
  ASSERT_EQ(track.center_history.size(), 1U);
  EXPECT_DOUBLE_EQ(track.center_history.front().time_stamp, 1.4);
}

TEST(CenterMotionVelocityFusion, DoesNotOvercountOverlappingWindows) {
  MultiClassObjectTrackingConfig config;
  config.global_coord_track = true;
  config.center_motion_velocity_fusion = true;
  config.center_motion_min_baseline_sec = 0.30;
  config.center_motion_min_update_interval_sec = 0.20;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  for (int index = 0; index <= 4; ++index) {
    if (index > 0) tracker->RunPrediction(0.1);
    Update(
        *tracker,
        Measurement(
            1.0 + 0.1 * index,
            static_cast<double>(index),
            0.0,
            false,
            0.0,
            0.0));
  }

  const auto tracks = tracker->GetTrackResults();
  const auto& track = InitializedTrack(tracks);
  EXPECT_NEAR(track.last_center_motion_fusion_time, 1.3, 1.0e-12);
}

TEST(CenterMotionVelocityFusion, RejectsNonlinearCenterHistory) {
  MultiClassObjectTrackingConfig config;
  config.global_coord_track = true;
  config.center_motion_velocity_fusion = true;
  config.center_motion_max_fit_residual_m = 0.10;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);

  const double positions[] = {0.0, 1.0, 2.0, 4.0};
  for (int index = 0; index < 4; ++index) {
    if (index > 0) tracker->RunPrediction(0.1);
    Update(
        *tracker,
        Measurement(
            1.0 + 0.1 * index,
            positions[index],
            0.0,
            false,
            0.0,
            0.0));
  }

  const auto tracks = tracker->GetTrackResults();
  const auto& track = InitializedTrack(tracks);
  EXPECT_DOUBLE_EQ(track.last_center_motion_fusion_time, 0.0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
