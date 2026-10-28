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

TEST(VehiclePromotion, ConfirmedUnknownAdoptsStrongCarWithoutChangingId) {
  MultiClassObjectTrackingConfig config;
  config.fast_confirmed_vehicle_promotion = true;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
  double timestamp = 0.0;
  ConfirmStationaryTrack(*tracker, timestamp);
  const int id = ConfirmedTrackId(*tracker);
  auto measurement = Measurement(1.3, 0.0, 0.0, mc_mot::ObjectClass::CAR);
  measurement.detection_confidence = 0.8;
  PredictAndUpdate(*tracker, timestamp, 1.3, {measurement});
  const auto tracks = InitializedTracks(*tracker);
  ASSERT_EQ(tracks.size(), 1U);
  EXPECT_EQ(tracks.front().track_id, id);
  EXPECT_EQ(tracks.front().getRepClass(), mc_mot::ObjectClass::CAR);
  EXPECT_NEAR(tracks.front().getRepClassProb(), 0.8, 1e-6);
}

TEST(VehiclePromotion, WeakCarAndDisabledPolicyKeepLegacySmoothing) {
  for (const bool enabled : {false, true}) {
    MultiClassObjectTrackingConfig config;
    config.fast_confirmed_vehicle_promotion = enabled;
    auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
    double timestamp = 0.0;
    ConfirmStationaryTrack(*tracker, timestamp);
    auto measurement = Measurement(1.3, 0.0, 0.0, mc_mot::ObjectClass::CAR);
    measurement.detection_confidence = enabled ? 0.59 : 0.8;
    PredictAndUpdate(*tracker, timestamp, 1.3, {measurement});
    EXPECT_EQ(InitializedTracks(*tracker).front().getRepClass(), mc_mot::ObjectClass::UNKNOWN);
  }
}

TEST(VehiclePromotion, TentativeTrackStillRequiresOriginalConfirmation) {
  MultiClassObjectTrackingConfig config;
  config.fast_confirmed_vehicle_promotion = true;
  auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
  double timestamp = 1.0;
  Update(*tracker, timestamp, {Measurement(timestamp, 0.0, 0.0)});
  PredictAndUpdate(*tracker, timestamp, 1.1,
      {Measurement(1.1, 0.0, 0.0, mc_mot::ObjectClass::CAR)});
  ASSERT_EQ(InitializedTracks(*tracker).size(), 1U);
  EXPECT_FALSE(InitializedTracks(*tracker).front().is_confirmed);
  EXPECT_EQ(InitializedTracks(*tracker).front().getRepClass(), mc_mot::ObjectClass::UNKNOWN);
}

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

namespace {
void ConfirmSmall(EkfMultiObjectTracking& tracker, double& time,
                  mc_mot::ObjectClass label=mc_mot::UNKNOWN, double side=.6) {
  time=1.;
  Update(tracker,time,{Measurement(time,0.,0.,label,side,side)});
  for (double next : {1.1,1.2,1.3})
    PredictAndUpdate(tracker,time,next,{Measurement(next,0.,0.,label,side,side)});
  ASSERT_EQ(InitializedTracks(tracker).size(),1U);
  ASSERT_TRUE(InitializedTracks(tracker).front().is_confirmed);
}

std::size_t SmallPairCount(double distance, double new_height=1., double z=0.,
                           double side=.6, bool enabled=true, double old_height=1.) {
  MultiClassObjectTrackingConfig config;
  config.suppress_small_unknown_duplicate_birth=enabled;
  config.duplicate_birth_minimum_max_dimension_m=2.;
  auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
  double time=1.;
  auto old=Measurement(time,0.,0.,mc_mot::UNKNOWN,side,side);
  old.dimension.height=old_height;
  Update(*tracker,time,{old});
  for (double next : {1.1,1.2,1.3}) {
    old.state.time_stamp=next;
    PredictAndUpdate(*tracker,time,next,{old});
  }
  old.state.time_stamp=1.4;
  auto added=Measurement(1.4,distance,0.,mc_mot::UNKNOWN,side,side);
  added.dimension.height=new_height;
  added.state.z=z;
  PredictAndUpdate(*tracker,time,1.4,{added,old});
  return InitializedTracks(*tracker).size();
}
}  // namespace

TEST(SmallUnknownBirthGuard, DefaultIsOffAndExistingSmallBirthBehaviorIsPreserved) {
  EXPECT_FALSE(MultiClassObjectTrackingConfig{}.suppress_small_unknown_duplicate_birth);
  EXPECT_EQ(SmallPairCount(.1,1.,0.,.6,false),2U);
}

TEST(SmallUnknownBirthGuard, CurrentRawOverlapSuppressesBirthRegardlessOfInputOrder) {
  for (bool reverse : {false,true}) {
    MultiClassObjectTrackingConfig config;
    config.suppress_small_unknown_duplicate_birth=true;
    auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
    tracker->SetAssociationDiagnosticsEnabled(true);
    double time;
    ConfirmSmall(*tracker,time);
    const int original=ConfirmedTrackId(*tracker);
    std::vector<mc_mot::Meastruct> pair{
        Measurement(1.4,.1,0.,mc_mot::UNKNOWN,.6,.6),
        Measurement(1.4,0.,0.,mc_mot::UNKNOWN,.6,.6)};
    if (reverse) std::reverse(pair.begin(),pair.end());
    PredictAndUpdate(*tracker,time,1.4,pair);
    ASSERT_EQ(InitializedTracks(*tracker).size(),1U);
    EXPECT_EQ(ConfirmedTrackId(*tracker),original);
    bool found=false;
    for (const auto& row : tracker->GetAssociationDiagnostics().rows)
      found |= std::string(row.event)=="birth_suppressed" &&
          std::string(row.reason)=="small_raw_overlap_associated_unknown";
    EXPECT_TRUE(found);
  }
}

TEST(SmallUnknownBirthGuard, AdjacentDrumsAndSmallSeparateObjectsRemainDistinct) {
  for (double distance : {.5,.75,1.0}) EXPECT_EQ(SmallPairCount(distance),2U);
  EXPECT_EQ(SmallPairCount(.2,1.,0.,.2),2U);  // touch but do not overlap
}

TEST(SmallUnknownBirthGuard, SameXYDifferentHeightOrVerticalPositionIsNotMerged) {
  EXPECT_EQ(SmallPairCount(0.,1.,.301),2U);
  EXPECT_EQ(SmallPairCount(0.,1.501),2U);
  EXPECT_EQ(SmallPairCount(0.,.149,0.,.6,true,.2),2U);
  EXPECT_EQ(SmallPairCount(.1,std::numeric_limits<double>::quiet_NaN()),2U);
  EXPECT_EQ(SmallPairCount(.1,std::numeric_limits<double>::infinity()),2U);
}

TEST(SmallUnknownBirthGuard, InclusiveGeometryBoundariesAndStrictSmallScope) {
  EXPECT_EQ(SmallPairCount(.25,1.,0.,.9),1U);
  EXPECT_EQ(SmallPairCount(.250001,1.,0.,.9),2U);
  EXPECT_EQ(SmallPairCount(.1,1.,.30),1U);
  EXPECT_EQ(SmallPairCount(.1,1.5),1U);
  EXPECT_EQ(SmallPairCount(.1,.15,0.,.6,true,.2),1U);
  const double boundary=.6*(1.-.45)/(1.+.45);
  EXPECT_EQ(SmallPairCount(boundary-1.e-6),1U);
  EXPECT_EQ(SmallPairCount(boundary+1.e-6),2U);
  // The new scope is a fixed <1m raw XY contract, not a silent expansion
  // when the independent legacy minimum dimension is raised to 2m.
  EXPECT_EQ(SmallPairCount(.1,1.,0.,.999),1U);
  EXPECT_EQ(SmallPairCount(.1,1.,0.,1.0),2U);
}

TEST(SmallUnknownBirthGuard, TwoCurrentSupportedMatchesAreAmbiguous) {
  MultiClassObjectTrackingConfig config;
  config.suppress_small_unknown_duplicate_birth=true;
  auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
  double time=1.;
  for (int i=0;i<5;++i) {
    double next=1.+i*.1;
    const std::vector<mc_mot::Meastruct> pair{
        Measurement(next,0.,0.,mc_mot::UNKNOWN,.6,.6),
        Measurement(next,.4,0.,mc_mot::UNKNOWN,.6,.6)};
    if (!i) Update(*tracker,next,pair);
    else PredictAndUpdate(*tracker,time,next,pair);
  }
  ASSERT_EQ(InitializedTracks(*tracker).size(),2U);
  PredictAndUpdate(*tracker,time,1.5,{
      Measurement(1.5,0.,0.,mc_mot::UNKNOWN,.6,.6),
      Measurement(1.5,.4,0.,mc_mot::UNKNOWN,.6,.6),
      Measurement(1.5,.2,0.,mc_mot::UNKNOWN,.6,.6)});
  EXPECT_EQ(InitializedTracks(*tracker).size(),3U);
}

TEST(SmallUnknownBirthGuard, KnownVehicleAndPedestrianEvidenceCannotSuppressUnknownBirth) {
  for (auto label : {mc_mot::CAR,mc_mot::TRUCK,mc_mot::PEDESTRIAN,mc_mot::BICYCLE}) {
    MultiClassObjectTrackingConfig config;
    config.suppress_small_unknown_duplicate_birth=true;
    auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
    double time;
    ConfirmSmall(*tracker,time,label);
    PredictAndUpdate(*tracker,time,1.4,{
        Measurement(1.4,0.,0.,label,.6,.6),
        Measurement(1.4,.1,0.,mc_mot::UNKNOWN,.6,.6)});
    EXPECT_EQ(InitializedTracks(*tracker).size(),2U);
  }
}

TEST(SmallUnknownBirthGuard, PreviouslyKnownClassRemainsProtectedAfterUnknownVotes) {
  MultiClassObjectTrackingConfig config;
  config.suppress_small_unknown_duplicate_birth=true;
  auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
  double time;
  ConfirmSmall(*tracker,time,mc_mot::CAR);
  for (int i=0;i<20;++i) {
    const double next=1.4+i*.1;
    PredictAndUpdate(*tracker,time,next,{Measurement(next,0.,0.,mc_mot::UNKNOWN,.6,.6)});
  }
  ASSERT_EQ(InitializedTracks(*tracker).front().getRepClass(),mc_mot::UNKNOWN);
  ASSERT_TRUE(InitializedTracks(*tracker).front().has_known_class_measurement);
  PredictAndUpdate(*tracker,time,3.4,{
      Measurement(3.4,0.,0.,mc_mot::UNKNOWN,.6,.6),
      Measurement(3.4,.1,0.,mc_mot::UNKNOWN,.6,.6)});
  EXPECT_EQ(InitializedTracks(*tracker).size(),2U);
}

TEST(SmallUnknownBirthGuard, MovingKnownVehicleIsNotSuppressedOrUsedAsUnknownEvidence) {
  MultiClassObjectTrackingConfig config;
  config.suppress_small_unknown_duplicate_birth=true;
  auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
  double time=1.;
  for (int i=0;i<30;++i) {
    const double next=1.+i*.1;
    auto car=Measurement(next,i*.8,0.,mc_mot::CAR,4.5,1.8);
    if (!i) Update(*tracker,next,{car});
    else PredictAndUpdate(*tracker,time,next,{car});
  }
  ASSERT_GT(InitializedTracks(*tracker).front().state_vec(S_VX),7.);
  PredictAndUpdate(*tracker,time,4.,{
      Measurement(4.,24.,0.,mc_mot::CAR,4.5,1.8),
      Measurement(4.,24.1,0.,mc_mot::UNKNOWN,.6,.6)});
  EXPECT_EQ(InitializedTracks(*tracker).size(),2U);
}

TEST(SmallUnknownBirthGuard, DoesNotSuppressSmallPedestriansOrTentativeCompetitors) {
  for (auto label : {mc_mot::UNKNOWN,mc_mot::PEDESTRIAN}) {
    MultiClassObjectTrackingConfig config;
    config.suppress_small_unknown_duplicate_birth=true;
    auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
    Update(*tracker,1.,{Measurement(1.,0.,0.,label,.6,.6)});
    double time=1.;
    PredictAndUpdate(*tracker,time,1.1,{
        Measurement(1.1,0.,0.,label,.6,.6),
        Measurement(1.1,.1,0.,label,.6,.6)});
    EXPECT_EQ(InitializedTracks(*tracker).size(),2U);
  }
}

TEST(SmallUnknownBirthGuard, RotatedAndQuarterTurnSwappedRawBoxesUseActualFootprint) {
  MultiClassObjectTrackingConfig config;
  config.suppress_small_unknown_duplicate_birth=true;
  auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
  double time=1.;
  auto old=Measurement(1.,0.,0.,mc_mot::UNKNOWN,.9,.5);
  Update(*tracker,time,{old});
  for (double next : {1.1,1.2,1.3}) {
    old.state.time_stamp=next;
    PredictAndUpdate(*tracker,time,next,{old});
  }
  old.state.time_stamp=1.4;
  auto rotated=Measurement(1.4,.1,0.,mc_mot::UNKNOWN,.5,.9);
  rotated.state.yaw=M_PI/2.;
  PredictAndUpdate(*tracker,time,1.4,{rotated,old});
  EXPECT_EQ(InitializedTracks(*tracker).size(),1U);
}

TEST(SmallUnknownBirthGuard, MasterBirthGuardSwitchStillDisablesTheComplement) {
  MultiClassObjectTrackingConfig config;
  config.suppress_small_unknown_duplicate_birth=true;
  config.suppress_duplicate_track_birth=false;
  auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
  double time;
  ConfirmSmall(*tracker,time);
  PredictAndUpdate(*tracker,time,1.4,{
      Measurement(1.4,0.,0.,mc_mot::UNKNOWN,.6,.6),
      Measurement(1.4,.1,0.,mc_mot::UNKNOWN,.6,.6)});
  EXPECT_EQ(InitializedTracks(*tracker).size(),2U);
}

TEST(SmallUnknownBirthGuard, BothCurrentRawBoxesMustBeStrictlyBelowOneMeter) {
  for (double matched_side : {.999,1.0,1.1}) {
    MultiClassObjectTrackingConfig config;
    config.suppress_small_unknown_duplicate_birth=true;
    config.duplicate_birth_minimum_max_dimension_m=2.;
    auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
    double time;
    ConfirmSmall(*tracker,time,mc_mot::UNKNOWN,matched_side);
    PredictAndUpdate(*tracker,time,1.4,{
        Measurement(1.4,0.,0.,mc_mot::UNKNOWN,matched_side,matched_side),
        Measurement(1.4,.05,0.,mc_mot::UNKNOWN,.8,.8)});
    EXPECT_EQ(InitializedTracks(*tracker).size(),matched_side<1.0 ? 1U : 2U);
  }
}

TEST(SmallUnknownBirthGuard, EitherMatchedRawAxisAtTheBoundaryExcludesSmallScope) {
  for (bool swapped : {false,true}) {
    MultiClassObjectTrackingConfig config;
    config.suppress_small_unknown_duplicate_birth=true;
    config.duplicate_birth_minimum_max_dimension_m=2.;
    auto tracker=std::make_unique<EkfMultiObjectTracking>(config);
    double time=1.;
    auto old=Measurement(time,0.,0.,mc_mot::UNKNOWN,swapped ? .8 : 1.,swapped ? 1. : .8);
    Update(*tracker,time,{old});
    for (double next : {1.1,1.2,1.3}) {
      old.state.time_stamp=next;
      PredictAndUpdate(*tracker,time,next,{old});
    }
    old.state.time_stamp=1.4;
    PredictAndUpdate(*tracker,time,1.4,{old,Measurement(1.4,.05,0.,mc_mot::UNKNOWN,.8,.8)});
    EXPECT_EQ(InitializedTracks(*tracker).size(),2U);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
