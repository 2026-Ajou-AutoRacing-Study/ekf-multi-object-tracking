#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "algorithm/ekf_multi_object_tracking.hpp"

namespace {
MultiClassObjectTrackingConfig Config(bool shape = false) {
  MultiClassObjectTrackingConfig config;
  config.global_coord_track = true;
  config.prediction_model = mc_mot::CTRV;
  config.max_association_dist_m = 5.;
  config.system_noise_std_xy_m = .01;
  config.system_noise_std_yaw_deg = .5;
  config.system_noise_std_vx_vy_ms = 1.;
  config.system_noise_std_yaw_rate_degs = 1.;
  config.system_noise_std_ax_ay_ms2 = .6;
  config.meas_noise_std_xy_m = .4;
  config.meas_noise_std_yaw_deg = 10.;
  config.dimension_filter_alpha = .3;
  config.pedestrian_gaussian_association_ranking = true;
  config.shape_consistent_association_ranking = shape;
  return config;
}

mc_mot::Meastruct Measurement(double time, double x, double y,
    mc_mot::ObjectClass label = mc_mot::UNKNOWN,
    double length = 1.6, double width = .35, double yaw = 0.) {
  mc_mot::Meastruct result;
  result.state = {};
  result.state.time_stamp = time;
  result.state.x = x;
  result.state.y = y;
  result.state.yaw = yaw;
  result.classification = label;
  result.dimension.length = length;
  result.dimension.width = width;
  result.dimension.height = label == mc_mot::TRUCK ? 3. : 1.2;
  result.detection_confidence = .8;
  return result;
}

void Step(EkfMultiObjectTracking& tracker, double previous, double time,
          const std::vector<mc_mot::Meastruct>& objects) {
  if (previous > 0.) tracker.RunPrediction(time - previous);
  mc_mot::Meastructs measurements;
  measurements.time_stamp = time;
  measurements.meas = objects;
  tracker.RunUpdate(measurements);
}

std::vector<mc_mot::TrackStruct> Active(const EkfMultiObjectTracking& tracker) {
  std::vector<mc_mot::TrackStruct> result;
  for (const auto& track : tracker.GetTrackResults().track)
    if (track.is_init) result.push_back(track);
  return result;
}

int ConfirmedCount(const EkfMultiObjectTracking& tracker) {
  int count = 0;
  for (const auto& track : Active(tracker)) count += track.is_confirmed;
  return count;
}

int MatchedId(const EkfMultiObjectTracking& tracker, int measurement = 0) {
  for (const auto& row : tracker.GetAssociationDiagnostics().rows)
    if (row.event == "matched" && row.measurement_index == measurement)
      return row.track_id;
  return -1;
}

void EqualStates(const EkfMultiObjectTracking& a,
                 const EkfMultiObjectTracking& b) {
  const auto one = a.GetTrackResults();
  const auto two = b.GetTrackResults();
  ASSERT_DOUBLE_EQ(one.time_stamp, two.time_stamp);
  ASSERT_EQ(one.track.size(), two.track.size());
  for (std::size_t i = 0; i < one.track.size(); ++i) {
    const auto& x = one.track[i];
    const auto& y = two.track[i];
    ASSERT_EQ(x.is_init, y.is_init) << i;
    if (!x.is_init) continue;
    EXPECT_EQ(x.track_id, y.track_id);
    EXPECT_TRUE((x.state_vec.array() == y.state_vec.array()).all()) << i;
    EXPECT_TRUE((x.state_cov.array() == y.state_cov.array()).all()) << i;
    EXPECT_EQ(x.age, y.age);
    EXPECT_EQ(x.is_associated, y.is_associated);
    EXPECT_EQ(x.is_confirmed, y.is_confirmed);
    EXPECT_EQ(x.last_measurement_time, y.last_measurement_time);
    EXPECT_EQ(x.dimension.length, y.dimension.length);
    EXPECT_EQ(x.dimension.width, y.dimension.width);
    EXPECT_EQ(x.dimension.height, y.dimension.height);
    EXPECT_EQ(x.detection_confidence, y.detection_confidence);
    EXPECT_EQ(x.direction_score, y.direction_score);
    for (int k = 0; k < MAX_HISTORY; ++k)
      EXPECT_EQ(x.detection_arr[k], y.detection_arr[k]);
    for (int k = 0; k < CLASS_NUM; ++k)
      EXPECT_EQ(x.class_score_arr[k], y.class_score_arr[k]);
  }
}

void MatureBarrier(EkfMultiObjectTracking& tracker) {
  for (int i = 0; i < 30; ++i) {
    const double time = 1. + i*.1;
    Step(tracker, i ? time-.1 : 0., time, {Measurement(time, 0., 0.)});
  }
}
}  // namespace

TEST(ShapeConsistentAssociation, NewOptionsDefaultOff) {
  EXPECT_FALSE(MultiClassObjectTrackingConfig{}.shape_consistent_association_ranking);
}

TEST(ShapeConsistentAssociation, MovingTruckOutlierCannotConfirmByStealingNormalHit) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config());
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  baseline->SetAssociationDiagnosticsEnabled(true);
  candidate->SetAssociationDiagnosticsEnabled(true);
  const double speed = 13.7;
  for (int i = 0; i < 60; ++i) {
    const double time = 1. + i*.1;
    const auto detection = Measurement(time, speed*(time-1.), 0.,
                                      mc_mot::TRUCK, 6.33, 2.49);
    Step(*baseline, i ? time-.1 : 0., time, {detection});
    Step(*candidate, i ? time-.1 : 0., time, {detection});
  }
  ASSERT_EQ(Active(*baseline).size(), 1U);
  const int mature_id = Active(*baseline).front().track_id;
  ASSERT_TRUE(Active(*baseline).front().is_confirmed);
  EXPECT_NEAR(Active(*baseline).front().state_vec(S_VX), speed, .1);
  const double outlier_time = 7.;
  const auto prediction = baseline->GetPredictedTrackResults(outlier_time);
  const double predicted_x = prediction.track[mature_id].state_vec(S_X);
  auto outlier = Measurement(outlier_time, predicted_x+2.61, .10,
                             mc_mot::TRUCK, 12.109, 2.805, .056);
  outlier.detection_confidence = .51179;
  Step(*baseline, 6.9, outlier_time, {outlier});
  Step(*candidate, 6.9, outlier_time, {outlier});
  ASSERT_EQ(Active(*baseline).size(), 2U);
  ASSERT_EQ(Active(*candidate).size(), 2U);
  EXPECT_EQ(ConfirmedCount(*candidate), 1);
  bool saw_mahalanobis_rejection = false;
  for (const auto& row : candidate->GetAssociationDiagnostics().rows) {
    if (row.event == "unmatched_candidate" && row.track_id == mature_id) {
      EXPECT_LT(row.distance_m, 5.);
      EXPECT_GT(row.association_cost, 5.);
      saw_mahalanobis_rejection = row.reason == "mahalanobis_gate";
    }
  }
  EXPECT_TRUE(saw_mahalanobis_rejection);
  // Birth/gates remain unchanged: both candidates still create a tentative.
  EqualStates(*baseline, *candidate);
  Step(*baseline, outlier_time, 7.09, {});
  Step(*candidate, outlier_time, 7.09, {});
  auto normal = Measurement(7.187, outlier.state.x+.08, .02,
                            mc_mot::TRUCK, 6.457, 2.547);
  normal.detection_confidence = .86628;
  Step(*baseline, 7.09, 7.187, {normal});
  Step(*candidate, 7.09, 7.187, {normal});
  EXPECT_NE(MatchedId(*baseline), mature_id);
  EXPECT_EQ(MatchedId(*candidate), mature_id);
  EXPECT_EQ(ConfirmedCount(*baseline), 2);
  EXPECT_EQ(ConfirmedCount(*candidate), 1);
  for (int i = 1; i <= 12; ++i) {
    const double time = 7.187+i*.1;
    normal.state.time_stamp = time;
    normal.state.x += speed*.1;
    Step(*candidate, time-.1, time, {normal});
    EXPECT_EQ(ConfirmedCount(*candidate), 1);
    EXPECT_EQ(MatchedId(*candidate), mature_id);
  }
  ASSERT_EQ(Active(*candidate).size(), 1U);
  EXPECT_EQ(Active(*candidate).front().track_id, mature_id);
}

TEST(ShapeConsistentAssociation, ThinSeparateNewBarrierSurvivesMatureBarrierOcclusion) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config());
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  MatureBarrier(*baseline);
  MatureBarrier(*candidate);
  for (int i = 0; i < 20; ++i) {
    const double time = 4.+i*.1;
    // Two 1.6 x .35 m boxes, centers separated transversally by 1 m: IoU=0.
    std::vector<mc_mot::Meastruct> objects{Measurement(time, 0., 1.)};
    if (i == 0 || i >= 5) objects.push_back(Measurement(time, 0., 0.));
    Step(*baseline, time-.1, time, objects);
    Step(*candidate, time-.1, time, objects);
    EqualStates(*baseline, *candidate);
  }
  EXPECT_EQ(ConfirmedCount(*candidate), 2);
}

TEST(ShapeConsistentAssociation, ConsistentLargeUnmatchedObservationCanStillConfirm) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config());
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  for (int i = 0; i < 30; ++i) {
    const double time = 1.+i*.1;
    const auto small = Measurement(time, 0., 0., mc_mot::TRUCK, 6.33, 2.49);
    Step(*baseline, i ? time-.1 : 0., time, {small});
    Step(*candidate, i ? time-.1 : 0., time, {small});
  }
  // A persistent long observation is not a one-frame expansion followed by
  // normal-size evidence for the mature track. Do not starve it indefinitely.
  for (int i = 0; i < 4; ++i) {
    const double time = 4.+i*.1;
    const auto large = Measurement(time, 2.61, .10, mc_mot::TRUCK, 12.109, 2.805);
    Step(*baseline, time-.1, time, {large});
    Step(*candidate, time-.1, time, {large});
    EqualStates(*baseline, *candidate);
  }
  ASSERT_EQ(Active(*candidate).size(), 2U);
  EXPECT_EQ(ConfirmedCount(*candidate), 2);
  EXPECT_TRUE(Active(*candidate).back().is_associated);
}

TEST(ShapeConsistentAssociation, ThinBarrierReacquisitionPreservesBothIds) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config());
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  for (int i = 0; i < 60; ++i) {
    const double time = 1.+i*.1;
    std::vector<mc_mot::Meastruct> objects;
    if (i < 30 || i >= 35) objects.push_back(Measurement(time, 0., 0.));
    if (i < 35 || i >= 40) objects.push_back(Measurement(time, 0., 1.));
    Step(*baseline, i ? time-.1 : 0., time, objects);
    Step(*candidate, i ? time-.1 : 0., time, objects);
    EqualStates(*baseline, *candidate);
  }
  ASSERT_EQ(Active(*candidate).size(), 2U);
  EXPECT_EQ(Active(*candidate)[0].track_id, 0);
  EXPECT_EQ(Active(*candidate)[1].track_id, 1);
  EXPECT_EQ(ConfirmedCount(*candidate), 2);
}

TEST(ShapeConsistentAssociation, AdjacentLaneNewVehicleAndCutInRemainIndependent) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config());
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  for (int i = 0; i < 80; ++i) {
    const double time = 1.+i*.1;
    std::vector<mc_mot::Meastruct> objects;
    if (i < 30 || i >= 34)
      objects.push_back(Measurement(time, i*.8, 0., mc_mot::CAR, 4.8, 1.8));
    if (i >= 25) {
      const double lateral = i < 40 ? 3.5 : std::max(2., 3.5-(i-40)*.1);
      objects.push_back(Measurement(time, i*.8, lateral, mc_mot::CAR, 4.6, 1.8));
    }
    Step(*baseline, i ? time-.1 : 0., time, objects);
    Step(*candidate, i ? time-.1 : 0., time, objects);
    EqualStates(*baseline, *candidate);
  }
  EXPECT_EQ(ConfirmedCount(*candidate), 2);
}

TEST(ShapeConsistentAssociation, LargeVehiclePartialOcclusionSizeRecoveryIsUnchanged) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config());
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  for (int i = 0; i < 90; ++i) {
    const double time = 1.+i*.1;
    const double length = i >= 30 && i < 40 ? 6.3 : 12.1;
    const auto object = Measurement(time, i*.6, 0., mc_mot::TRUCK, length, 2.5);
    Step(*baseline, i ? time-.1 : 0., time, {object});
    Step(*candidate, i ? time-.1 : 0., time, {object});
    EqualStates(*baseline, *candidate);
  }
  ASSERT_EQ(Active(*candidate).size(), 1U);
  EXPECT_EQ(Active(*candidate).front().track_id, 0);
  EXPECT_NEAR(Active(*candidate).front().dimension.length, 12.1, .001);
}

TEST(ShapeConsistentAssociation, TwoClosePedestriansAreExactlyUnchanged) {
  auto baseline = std::make_unique<EkfMultiObjectTracking>(Config());
  auto candidate = std::make_unique<EkfMultiObjectTracking>(Config(true));
  for (int i = 0; i < 80; ++i) {
    const double time = 1.+i*.1;
    std::vector<mc_mot::Meastruct> objects;
    if (i%9 != 0)
      objects.push_back(Measurement(time, i*.08, 0., mc_mot::PEDESTRIAN, .5, .5));
    if (i >= 15 && i%11 != 0)
      objects.push_back(Measurement(time, i*.08, .8, mc_mot::PEDESTRIAN, .5, .5));
    Step(*baseline, i ? time-.1 : 0., time, objects);
    Step(*candidate, i ? time-.1 : 0., time, objects);
    EqualStates(*baseline, *candidate);
  }
  EXPECT_EQ(ConfirmedCount(*candidate), 2);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
