#include <gtest/gtest.h>
#include <limits>
#include "algorithm/ekf_multi_object_tracking.hpp"

namespace {
mc_mot::Meastruct Measurement(double t, double x, double y,
                             mc_mot::ObjectClass cls = mc_mot::PEDESTRIAN) {
  mc_mot::Meastruct m;
  m.state = {};
  m.state.time_stamp = t; m.state.x = x; m.state.y = y;
  m.classification = cls; m.detection_confidence = .8;
  m.dimension.length = .5; m.dimension.width = .5; m.dimension.height = 1.8;
  return m;
}
std::vector<mc_mot::TrackStruct> Tracks(const EkfMultiObjectTracking& tracker) {
  std::vector<mc_mot::TrackStruct> result;
  for (const auto& t : tracker.GetTrackResults().track) if (t.is_init) result.push_back(t);
  return result;
}
void Step(EkfMultiObjectTracking& tracker, double t, std::vector<mc_mot::Meastruct> ms) {
  if (t > 1.) tracker.RunPrediction(.1);
  mc_mot::Meastructs frame; frame.time_stamp = t; frame.meas = ms;
  tracker.RunUpdate(frame);
}
}

TEST(PedestrianRanking, CovarianceVolumePenalizesBroadTrack) {
  Eigen::Vector2d r(.3, .3);
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * .16;
  EXPECT_LT(mc_mot::GaussianXYAssociationRank(r, Eigen::Matrix2d::Identity()*.1, R, .8, 1000.),
            mc_mot::GaussianXYAssociationRank(r, Eigen::Matrix2d::Identity()*1000., R, .8, 1000.));
}
TEST(PedestrianRanking, LowConfidenceUsesMeasurementNoiseMultiplier) {
  Eigen::Vector2d r(.3, .1);
  Eigen::Matrix2d P = Eigen::Matrix2d::Identity() * .1;
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * .16;
  const double variance = .1 + 10*.16;
  EXPECT_NEAR(mc_mot::GaussianXYAssociationRank(r, P, R, .1, 7.),
              .5*(r.squaredNorm()/variance + 2*std::log(variance)), 1.e-12);
}
TEST(PedestrianRanking, InvalidCovarianceFallsBack) {
  Eigen::Matrix2d P = -Eigen::Matrix2d::Identity();
  const Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
  EXPECT_DOUBLE_EQ(mc_mot::GaussianXYAssociationRank(Eigen::Vector2d::Zero(), P, R, .8, 3.), 3.);
  P(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_DOUBLE_EQ(mc_mot::GaussianXYAssociationRank(Eigen::Vector2d::Zero(), P, R, .8, 3.), 3.);
}
TEST(PedestrianRanking, NonFiniteInnovationFallsBack) {
  Eigen::Vector2d r(0., std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(mc_mot::GaussianXYAssociationRank(r, Eigen::Matrix2d::Identity(),
                    Eigen::Matrix2d::Identity(), .8, 2.), 2.);
}
TEST(PedestrianRanking, GenericCoreDefaultsOff) {
  EXPECT_FALSE(MultiClassObjectTrackingConfig{}.pedestrian_gaussian_association_ranking);
}
TEST(PedestrianRanking, CannotAdmitOutsidePedestrianGate) {
  MultiClassObjectTrackingConfig config;
  config.max_association_dist_m = 5.;
  config.pedestrian_gaussian_association_ranking = true;
  EkfMultiObjectTracking tracker(config);
  Step(tracker, 1., {Measurement(1., 0., 0.)});
  Step(tracker, 1.1, {Measurement(1.1, 2.01, 0.)});
  ASSERT_EQ(Tracks(tracker).size(), 2U);
}
TEST(PedestrianRanking, CannotAssociatePedestrianWithVehicle) {
  MultiClassObjectTrackingConfig config;
  config.pedestrian_gaussian_association_ranking = true;
  EkfMultiObjectTracking tracker(config);
  Step(tracker, 1., {Measurement(1., 0., 0., mc_mot::CAR)});
  Step(tracker, 1.1, {Measurement(1.1, 0., 0.)});
  ASSERT_EQ(Tracks(tracker).size(), 2U);
}
TEST(PedestrianRanking, LargeNllDoesNotReplaceLegacyGate) {
  MultiClassObjectTrackingConfig config;
  config.max_association_dist_m = 5.;
  config.system_noise_std_xy_m = 1000.;
  config.pedestrian_gaussian_association_ranking = true;
  EkfMultiObjectTracking tracker(config);
  Step(tracker,1.,{Measurement(1.,0.,0.)});
  Step(tracker,1.1,{Measurement(1.1,.01,0.)});
  ASSERT_EQ(Tracks(tracker).size(),1U);
  EXPECT_TRUE(Tracks(tracker).front().is_associated);
}
TEST(PedestrianRanking, MaintainsTwoCloseActors) {
  MultiClassObjectTrackingConfig config;
  config.pedestrian_gaussian_association_ranking = true;
  EkfMultiObjectTracking tracker(config);
  for (int i=0; i<30; ++i) {
    const double t = 1. + i*.1;
    Step(tracker, t, {Measurement(t, i*.05, 0.), Measurement(t, i*.05, .8)});
  }
  const auto tracks = Tracks(tracker);
  ASSERT_EQ(tracks.size(), 2U);
  EXPECT_TRUE(tracks[0].is_confirmed); EXPECT_TRUE(tracks[1].is_confirmed);
  EXPECT_NE(tracks[0].track_id, tracks[1].track_id);
}
TEST(PedestrianRanking, NonPedestrianStatesAreUnchanged) {
  MultiClassObjectTrackingConfig off, on;
  on.pedestrian_gaussian_association_ranking = true;
  EkfMultiObjectTracking baseline(off), candidate(on);
  for (int i=0; i<30; ++i) {
    const double t=1.+i*.1;
    auto ms = std::vector<mc_mot::Meastruct>{Measurement(t,i*.1,0.,mc_mot::CAR),
                  Measurement(t,5.,3.,mc_mot::UNKNOWN)};
    Step(baseline,t,ms); Step(candidate,t,ms);
    const auto a=Tracks(baseline), b=Tracks(candidate);
    ASSERT_EQ(a.size(),b.size());
    for (size_t j=0;j<a.size();++j) {
      EXPECT_EQ(a[j].track_id,b[j].track_id);
      EXPECT_TRUE((a[j].state_vec.array()==b[j].state_vec.array()).all());
      EXPECT_TRUE((a[j].state_cov.array()==b[j].state_cov.array()).all());
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
