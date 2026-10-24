#include <gtest/gtest.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <sstream>

#include "association_diagnostic_json.hpp"
#include "diagnostic_subscription_gate.hpp"

TEST(AssociationDiagnosticSubscription, DefaultAutoAdvertisesWithoutCollecting) {
    EXPECT_TRUE(tracking_diagnostics::ShouldAdvertise(false, true));
    EXPECT_FALSE(tracking_diagnostics::ShouldCollect(false, true, 0));
    EXPECT_TRUE(tracking_diagnostics::ShouldCollect(false, true, 1));
    EXPECT_FALSE(tracking_diagnostics::ShouldCollect(false, true, 0));
}

TEST(AssociationDiagnosticSubscription, ExplicitDisablePreservesFullOff) {
    EXPECT_FALSE(tracking_diagnostics::ShouldAdvertise(false, false));
    EXPECT_FALSE(tracking_diagnostics::ShouldCollect(false, false, 0));
    EXPECT_FALSE(tracking_diagnostics::ShouldCollect(false, false, 5));
}

TEST(AssociationDiagnosticSubscription, LegacyExplicitEnableRequiresSubscriber) {
    EXPECT_TRUE(tracking_diagnostics::ShouldAdvertise(true, false));
    EXPECT_FALSE(tracking_diagnostics::ShouldCollect(true, false, 0));
    EXPECT_TRUE(tracking_diagnostics::ShouldCollect(true, false, 1));
    EXPECT_TRUE(tracking_diagnostics::ShouldCollect(true, true, 2));
}

namespace {
mc_mot::Meastruct Measurement(double time, double x, double y,
    mc_mot::ObjectClass label = mc_mot::ObjectClass::UNKNOWN) {
    mc_mot::Meastruct result;
    result.state = {};
    result.state.time_stamp = time;
    result.state.x = x;
    result.state.y = y;
    result.classification = label;
    result.dimension.length = 1.6;
    result.dimension.width = .35;
    result.dimension.height = 1.2;
    result.detection_confidence = .8;
    return result;
}

void Update(EkfMultiObjectTracking& tracker, double previous, double time,
            const std::vector<mc_mot::Meastruct>& objects) {
    if (previous) tracker.RunPrediction(time - previous);
    mc_mot::Meastructs measurements;
    measurements.time_stamp = time;
    measurements.meas = objects;
    tracker.RunUpdate(measurements);
}

void EqualStates(const mc_mot::TrackStructs& one, const mc_mot::TrackStructs& two) {
    ASSERT_EQ(one.time_stamp, two.time_stamp);
    ASSERT_EQ(one.track.size(), two.track.size());
    for (std::size_t i = 0; i < one.track.size(); ++i) {
        const auto& a = one.track[i];
        const auto& b = two.track[i];
        EXPECT_EQ(a.track_id, b.track_id);
        EXPECT_TRUE((a.state_vec.array() == b.state_vec.array()).all());
        EXPECT_TRUE((a.state_cov.array() == b.state_cov.array()).all());
        EXPECT_EQ(a.initialization_time, b.initialization_time);
        EXPECT_EQ(a.update_time, b.update_time);
        EXPECT_EQ(a.last_measurement_time, b.last_measurement_time);
        EXPECT_EQ(a.is_init, b.is_init);
        EXPECT_EQ(a.is_confirmed, b.is_confirmed);
        EXPECT_EQ(a.is_associated, b.is_associated);
        EXPECT_EQ(a.age, b.age);
        EXPECT_EQ(a.detection_confidence, b.detection_confidence);
        EXPECT_EQ(a.direction_score, b.direction_score);
        EXPECT_EQ(a.object_z, b.object_z);
        EXPECT_EQ(a.dimension.length, b.dimension.length);
        EXPECT_EQ(a.dimension.width, b.dimension.width);
        EXPECT_EQ(a.dimension.height, b.dimension.height);
        EXPECT_EQ(a.has_detector_velocity, b.has_detector_velocity);
        EXPECT_EQ(a.detector_velocity_x, b.detector_velocity_x);
        EXPECT_EQ(a.detector_velocity_y, b.detector_velocity_y);
        EXPECT_EQ(a.detector_velocity_time, b.detector_velocity_time);
        EXPECT_EQ(a.detector_velocity_variance_mps2, b.detector_velocity_variance_mps2);
        EXPECT_EQ(a.last_center_motion_fusion_time, b.last_center_motion_fusion_time);
        EXPECT_EQ(a.has_center_motion_velocity, b.has_center_motion_velocity);
        EXPECT_EQ(a.center_motion_velocity_x, b.center_motion_velocity_x);
        EXPECT_EQ(a.center_motion_velocity_y, b.center_motion_velocity_y);
        EXPECT_EQ(a.center_motion_velocity_noise_std_mps, b.center_motion_velocity_noise_std_mps);
        ASSERT_EQ(a.center_history.size(), b.center_history.size());
        for (std::size_t k = 0; k < a.center_history.size(); ++k) {
            EXPECT_EQ(a.center_history[k].time_stamp, b.center_history[k].time_stamp);
            EXPECT_EQ(a.center_history[k].x, b.center_history[k].x);
            EXPECT_EQ(a.center_history[k].y, b.center_history[k].y);
        }
        for (int k = 0; k < CLASS_NUM; ++k) EXPECT_EQ(a.class_score_arr[k], b.class_score_arr[k]);
        for (int k = 0; k < MAX_HISTORY; ++k) EXPECT_EQ(a.detection_arr[k], b.detection_arr[k]);
    }
}

bool Has(const EkfMultiObjectTracking& tracker, const std::string& event,
         const std::string& reason = "") {
    for (const auto& row : tracker.GetAssociationDiagnostics().rows) {
        if (row.event == event && (reason.empty() || row.reason == reason)) return true;
    }
    return false;
}

void Mature(EkfMultiObjectTracking& tracker) {
    for (int i = 0; i < 5; ++i) {
        const double time = 1. + i*.1;
        Update(tracker, i ? time-.1 : 0., time, {Measurement(time, 0., 0.)});
    }
}
}  // namespace

TEST(AssociationDiagnostics, DisabledByDefaultAndEmptyUpdateWorks) {
    auto tracker = std::make_unique<EkfMultiObjectTracking>();
    Update(*tracker, 0., 1., {Measurement(1., 0., 0.)});
    EXPECT_TRUE(tracker->GetAssociationDiagnostics().rows.empty());
    EXPECT_EQ(tracker->GetAssociationDiagnostics().rows.capacity(), 0U);
    tracker->SetAssociationDiagnosticsEnabled(true);
    Update(*tracker, 1., 1.1, {});
    EXPECT_EQ(tracker->GetAssociationDiagnostics().input_count, 0U);
    EXPECT_TRUE(Has(*tracker, "coast"));
    tracker->SetAssociationDiagnosticsEnabled(false);
    EXPECT_TRUE(tracker->GetAssociationDiagnostics().rows.empty());
}

TEST(AssociationDiagnostics, StateAndPredictionCopiesAreExactlyIdentical) {
    MultiClassObjectTrackingConfig config;
    config.global_coord_track = true;
    config.prediction_model = mc_mot::PredictionModel::CTRV;
    config.pedestrian_gaussian_association_ranking = true;
    config.dimension_filter_alpha = .3;
    config.meas_noise_std_xy_m = .4;
    auto plain = std::make_unique<EkfMultiObjectTracking>(config);
    auto traced = std::make_unique<EkfMultiObjectTracking>(config);
    for (int i = 0; i < 150; ++i) {
        const double time = 1. + i*.1;
        std::vector<mc_mot::Meastruct> objects;
        if (i < 100) objects.push_back(Measurement(time, 0., 0.));
        if (i >= 12 && i < 120) objects.push_back(Measurement(time, 0., 1.));
        if (i >= 5 && i < 80 && i%9 != 0) objects.push_back(Measurement(time, i*.08, 4., mc_mot::ObjectClass::CAR));
        if (i >= 15 && i < 100 && i%11 != 0) objects.push_back(Measurement(time, 10.-i*.04, 5., mc_mot::ObjectClass::PEDESTRIAN));
        if (i >= 30 && i < 115) objects.push_back(Measurement(time, 9.-i*.02, 5.6, mc_mot::ObjectClass::PEDESTRIAN));
        traced->SetAssociationDiagnosticsEnabled(i%17 != 0);
        Update(*plain, i ? time-.1 : 0., time, objects);
        Update(*traced, i ? time-.1 : 0., time, objects);
        EqualStates(plain->GetTrackResults(), traced->GetTrackResults());
        EqualStates(plain->GetPredictedTrackResults(time+.07), traced->GetPredictedTrackResults(time+.07));
    }
}

TEST(AssociationDiagnostics, RecordsConfirmationCoastAndTimeout) {
    auto tracker = std::make_unique<EkfMultiObjectTracking>();
    tracker->SetAssociationDiagnosticsEnabled(true);
    Update(*tracker, 0., 1., {Measurement(1., 0., 0.)});
    EXPECT_TRUE(Has(*tracker, "birth"));
    Update(*tracker, 1., 1.1, {Measurement(1.1, 0., 0.)});
    Update(*tracker, 1.1, 1.2, {Measurement(1.2, 0., 0.)});
    EXPECT_TRUE(Has(*tracker, "matched", "confirmed"));
    Update(*tracker, 1.2, 1.5, {});
    EXPECT_TRUE(Has(*tracker, "coast", "confirmed_unmatched"));
    EXPECT_NEAR(tracker->GetAssociationDiagnostics().rows.front().prior_measurement_age_sec, .3, 1.e-12);
    Update(*tracker, 1.5, 2.3, {});
    EXPECT_TRUE(Has(*tracker, "delete", "coast_timeout"));
}

TEST(AssociationDiagnostics, RecordsSuppressionWithoutChangingNearbyObjectPolicy) {
    auto tracker = std::make_unique<EkfMultiObjectTracking>();
    tracker->SetAssociationDiagnosticsEnabled(true);
    Mature(*tracker);
    Update(*tracker, 1.4, 1.5, {Measurement(1.5, 0., .3), Measurement(1.5, 0., 0.)});
    EXPECT_TRUE(Has(*tracker, "birth_suppressed"));
    EXPECT_TRUE(Has(*tracker, "unmatched_candidate", "eligible_competitor"));
    for (const auto& row : tracker->GetAssociationDiagnostics().rows) {
        if (std::string(row.event) == "unmatched_candidate") EXPECT_EQ(row.track_assigned_measurement_index, 1);
    }
    Update(*tracker, 1.5, 1.6, {Measurement(1.6, 0., 0.), Measurement(1.6, 0., 1.)});
    EXPECT_TRUE(Has(*tracker, "birth"));
    EXPECT_FALSE(Has(*tracker, "birth_suppressed"));
}

TEST(AssociationDiagnostics, PedestrianClassGateIsExplicit) {
    auto tracker = std::make_unique<EkfMultiObjectTracking>();
    tracker->SetAssociationDiagnosticsEnabled(true);
    Mature(*tracker);
    Update(*tracker, 1.4, 1.5, {Measurement(1.5, 0., 0., mc_mot::ObjectClass::PEDESTRIAN)});
    EXPECT_TRUE(Has(*tracker, "unmatched_candidate", "class_gate"));
    EXPECT_TRUE(Has(*tracker, "birth"));
    EXPECT_FALSE(Has(*tracker, "birth_suppressed"));
}

TEST(AssociationDiagnostics, MahalanobisRejectionIsDistinctFromDistanceGate) {
    MultiClassObjectTrackingConfig config;
    config.max_association_dist_m = 5.;
    config.meas_noise_std_xy_m = .4;
    config.system_noise_std_xy_m = .01;
    config.system_noise_std_vx_vy_ms = 1.;
    auto tracker = std::make_unique<EkfMultiObjectTracking>(config);
    tracker->SetAssociationDiagnosticsEnabled(true);
    for (int i = 0; i < 100; ++i) {
        const double time = 1.+i*.1;
        Update(*tracker, i ? time-.1 : 0., time, {Measurement(time, 0., 0., mc_mot::ObjectClass::CAR)});
    }
    Update(*tracker, 10.9, 11., {Measurement(11., 2.61, 0., mc_mot::ObjectClass::CAR)});
    ASSERT_TRUE(Has(*tracker, "unmatched_candidate", "mahalanobis_gate"));
    EXPECT_TRUE(Has(*tracker, "birth"));
    for (const auto& row : tracker->GetAssociationDiagnostics().rows) {
        if (std::string(row.event) != "unmatched_candidate") continue;
        EXPECT_LT(row.distance_m, 5.);
        EXPECT_GE(row.association_cost, 5.);
        EXPECT_EQ(row.track_assigned_measurement_index, -1);
    }
}

TEST(AssociationDiagnostics, RowsAreBoundedAndTruncationIsExplicit) {
    auto tracker = std::make_unique<EkfMultiObjectTracking>();
    tracker->SetAssociationDiagnosticsEnabled(true);
    std::vector<mc_mot::Meastruct> objects;
    for (int i = 0; i < 150; ++i) objects.push_back(Measurement(1., i*10., 0.));
    Update(*tracker, 0., 1., objects);
    const auto& trace = tracker->GetAssociationDiagnostics();
    EXPECT_EQ(trace.rows.size(), mc_mot::AssociationDiagnostics::kMaximumRows);
    EXPECT_EQ(trace.omitted_rows, 194U);
}

TEST(AssociationDiagnostics, JsonIsParseableEscapedFiniteAndUuidAligned) {
    mc_mot::AssociationDiagnostics trace;
    trace.measurement_stamp = 1.25;
    trace.omitted_rows = 2;
    mc_mot::AssociationDiagnosticRow row;
    row.event = "birth";
    row.track_id = 0x156;
    trace.rows.push_back(row);
    const std::string json = tracking_diagnostics::ToJson(trace, 20, 31, "map\n\"quoted", 1.5);
    boost::property_tree::ptree parsed;
    std::stringstream input(json);
    ASSERT_NO_THROW(boost::property_tree::read_json(input, parsed));
    EXPECT_EQ(parsed.get<std::string>("world_frame"), "map\n\"quoted");
    EXPECT_TRUE(parsed.get<bool>("truncated"));
    EXPECT_EQ(parsed.get_child("rows").front().second.get<std::string>("track_uuid"), "414a4f55000000000000000000000156");
    EXPECT_EQ(parsed.get_child("rows").front().second.get<std::string>("snapshot_phase"), "post_birth");
    EXPECT_EQ(parsed.get_child("rows").front().second.get<std::string>("candidate_scope"), "unmatched_measurement");
    EXPECT_NE(json.find("\"distance_m\":null"), std::string::npos);
    EXPECT_EQ(json.find(":nan"), std::string::npos);
    EXPECT_EQ(json.find(":inf"), std::string::npos);
}

TEST(AssociationDiagnostics, UnmatchedSnapshotAndAssignmentPhasesAreExplicit) {
    auto tracker = std::make_unique<EkfMultiObjectTracking>();
    tracker->SetAssociationDiagnosticsEnabled(true);
    Mature(*tracker);
    Update(*tracker, 1.4, 1.5, {Measurement(1.5, 0., .3), Measurement(1.5, 0., 0.)});
    std::stringstream input(tracking_diagnostics::ToJson(tracker->GetAssociationDiagnostics(), 1, 2, "map", 1.6));
    boost::property_tree::ptree parsed;
    boost::property_tree::read_json(input, parsed);
    bool found = false;
    for (const auto& row : parsed.get_child("rows")) {
        if (row.second.get<std::string>("event") != "unmatched_candidate") continue;
        found = true;
        EXPECT_EQ(row.second.get<std::string>("snapshot_phase"), "pre_association_update");
        EXPECT_EQ(row.second.get<std::string>("candidate_scope"), "nearest_existing_track_only");
        EXPECT_EQ(row.second.get<std::string>("assignment_index_phase"), "post_assignment");
        EXPECT_FALSE(row.second.get<bool>("associated"));
        EXPECT_EQ(row.second.get<int>("track_assigned_measurement_index"), 1);
    }
    EXPECT_TRUE(found);
    EXPECT_EQ(parsed.get<std::string>("collection_scope"), "bounded_chosen_and_nearest_only_not_exhaustive");
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
