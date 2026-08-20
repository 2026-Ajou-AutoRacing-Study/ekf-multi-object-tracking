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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
