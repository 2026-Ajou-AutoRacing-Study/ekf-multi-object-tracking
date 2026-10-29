/****************************************************************************/
// Module:      ekf_multi_object_tracking.cpp
// Description: ekf_multi_object_tracking
//
// Authors: Jaeyoung Jo (wodud3743@gmail.com)
// Version: 0.1
//
// Revision History
//      July 19, 2024: Jaeyoung Jo - Created.
//      Jan  08, 2025: Jaeyoung Jo - Public data type.
//      XXXX XX, 2023: XXXXXXX XX -
/****************************************************************************/

#include "algorithm/ekf_multi_object_tracking.hpp"

constexpr std::size_t mc_mot::AssociationDiagnostics::kMaximumRows;

namespace {
void HoldUnknownCoastingPose(mc_mot::TrackStruct &track,
                            const MultiClassObjectTrackingConfig &config) {
    if (!config.hold_unknown_coasting_pose || !config.global_coord_track ||
        !track.is_init || track.getRepClass() != mc_mot::UNKNOWN ||
        track.has_known_class_measurement || !track.has_observed_output_pose) return;
    track.state_vec(S_X) = track.last_observed_output_pose.x();
    track.state_vec(S_Y) = track.last_observed_output_pose.y();
    track.state_vec(S_YAW) = track.last_observed_output_pose.z();
}

double DimensionRatio(const mc_mot::ObjectDimension& a,
                      const mc_mot::ObjectDimension& b) {
    if (!std::isfinite(a.length) || !std::isfinite(a.width) ||
        !std::isfinite(b.length) || !std::isfinite(b.width) ||
        std::min({a.length, a.width, b.length, b.width}) < 0.25) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max({a.length / b.length, b.length / a.length,
                     a.width / b.width, b.width / a.width});
}

bool InsideFootprint(double x, double y, const mc_mot::TrackStruct& track) {
    const double dx = x - track.state_vec(S_X);
    const double dy = y - track.state_vec(S_Y);
    const double c = std::cos(track.state_vec(S_YAW));
    const double s = std::sin(track.state_vec(S_YAW));
    // A small numerical allowance, not a wider global birth/association gate.
    return std::abs(c * dx + s * dy) <= track.dimension.length * 0.5 + 0.10 &&
           std::abs(-s * dx + c * dy) <= track.dimension.width * 0.5 + 0.10;
}

bool IsShapeConflictingCompetitor(const mc_mot::Meastruct& observation,
                                 const mc_mot::TrackStruct& tentative,
                                 const mc_mot::TrackStruct& mature) {
    if (!tentative.is_init || tentative.is_confirmed ||
        !mature.is_init || !mature.is_confirmed ||
        observation.classification == mc_mot::PEDESTRIAN ||
        tentative.getRepClass() == mc_mot::PEDESTRIAN ||
        mature.getRepClass() == mc_mot::PEDESTRIAN) return false;
    const double tentative_ratio = DimensionRatio(observation.dimension, tentative.dimension);
    const double mature_ratio = DimensionRatio(observation.dimension, mature.dimension);
    if (!std::isfinite(tentative_ratio) || !std::isfinite(mature_ratio) ||
        tentative_ratio < 1.50 || mature_ratio > 1.35) return false;
    if (std::abs(std::cos(tentative.state_vec(S_YAW) - mature.state_vec(S_YAW))) < 0.70710678 ||
        std::abs(std::cos(observation.state.yaw - mature.state_vec(S_YAW))) < 0.70710678) return false;
    return InsideFootprint(tentative.state_vec(S_X), tentative.state_vec(S_Y), mature) &&
           InsideFootprint(observation.state.x, observation.state.y, mature);
}

struct FootprintPoint { double x, y; };

double Cross(const FootprintPoint& a, const FootprintPoint& b, const FootprintPoint& p) {
    return (b.x-a.x)*(p.y-a.y) - (b.y-a.y)*(p.x-a.x);
}

std::vector<FootprintPoint> RawFootprint(const mc_mot::Meastruct& box,
                                       const double origin_x, const double origin_y) {
    const double c=std::cos(box.state.yaw), s=std::sin(box.state.yaw);
    const double half_l=box.dimension.length*.5, half_w=box.dimension.width*.5;
    std::vector<FootprintPoint> corners;
    for (const auto& offset : std::array<FootprintPoint,4>{{
            {-half_l,-half_w}, {half_l,-half_w}, {half_l,half_w}, {-half_l,half_w}}}) {
        corners.push_back({box.state.x-origin_x+c*offset.x-s*offset.y,
                           box.state.y-origin_y+s*offset.x+c*offset.y});
    }
    return corners;
}

double PolygonArea(const std::vector<FootprintPoint>& polygon) {
    double twice_area=0.;
    for (std::size_t i=0; i<polygon.size(); ++i) {
        const auto& a=polygon[i];
        const auto& b=polygon[(i+1)%polygon.size()];
        twice_area += a.x*b.y-a.y*b.x;
    }
    return std::abs(twice_area)*.5;
}

double RawFootprintIoU(const mc_mot::Meastruct& one, const mc_mot::Meastruct& two) {
    // Relative coordinates avoid subtracting huge map-coordinate area terms.
    auto polygon=RawFootprint(one,one.state.x,one.state.y);
    const auto clip=RawFootprint(two,one.state.x,one.state.y);
    for (std::size_t i=0; i<clip.size(); ++i) {
        const auto source=polygon;
        polygon.clear();
        if (source.empty()) return 0.;
        const auto& a=clip[i];
        const auto& b=clip[(i+1)%clip.size()];
        auto previous=source.back();
        double before=Cross(a,b,previous);
        for (const auto& point : source) {
            const double after=Cross(a,b,point);
            if ((after>=0.) != (before>=0.)) {
                const double fraction=before/(before-after);
                polygon.push_back({previous.x+fraction*(point.x-previous.x),
                                   previous.y+fraction*(point.y-previous.y)});
            }
            if (after>=0.) polygon.push_back(point);
            previous=point;
            before=after;
        }
    }
    const double intersection=PolygonArea(polygon);
    const double area_one=one.dimension.length*one.dimension.width;
    const double area_two=two.dimension.length*two.dimension.width;
    const double denominator=area_one+area_two-intersection;
    const double iou=denominator>1.e-12 ? intersection/denominator : 0.;
    return std::isfinite(iou) ? iou : 0.;
}

bool ValidRawBox(const mc_mot::Meastruct& box) {
    return std::isfinite(box.state.x) && std::isfinite(box.state.y) &&
        std::isfinite(box.state.z) && std::isfinite(box.state.yaw) &&
        std::isfinite(box.dimension.length) && std::isfinite(box.dimension.width) &&
        std::isfinite(box.dimension.height) &&
        std::min({box.dimension.length,box.dimension.width,box.dimension.height})>=.15;
}

bool SmallRawOverlapCompatible(const mc_mot::Meastruct& one,
                               const mc_mot::Meastruct& two) {
    if (!ValidRawBox(one) || !ValidRawBox(two) ||
        std::hypot(one.state.x-two.state.x,one.state.y-two.state.y)>.25 ||
        std::abs(one.state.z-two.state.z)>.30) return false;
    const std::array<double,3> first{{std::min(one.dimension.length,one.dimension.width),
        std::max(one.dimension.length,one.dimension.width),one.dimension.height}};
    const std::array<double,3> second{{std::min(two.dimension.length,two.dimension.width),
        std::max(two.dimension.length,two.dimension.width),two.dimension.height}};
    for (std::size_t i=0; i<first.size(); ++i)
        if (std::max(first[i],second[i])/std::min(first[i],second[i])>1.5) return false;
    return RawFootprintIoU(one,two)>=.45;
}
}  // namespace

/*
    Predict all track in same dt_sec
*/
void EkfMultiObjectTracking::RunPrediction(double dt_sec) {
    for (int i = 0; i < MAX_TRACKS; i++) {
        if (all_tracks_[i].is_init == true) {
            PredictTrack(all_tracks_[i], dt_sec);
        }
    }
    recent_timestamp_ += dt_sec;
}

void EkfMultiObjectTracking::RunUpdate(const mc_mot::Meastructs &measurements) {

    recent_timestamp_ = measurements.time_stamp;

    int i_meas_num = measurements.meas.size();
    if(i_meas_num > MAX_TRACKS){
        ROS_WARN_STREAM("[EKF tracker] Too many measurements: " << i_meas_num);
        i_meas_num = MAX_TRACKS;
    }
    std::vector<mc_mot::AssociationDiagnosticRow> nearest_candidates;
    if (association_diagnostics_enabled_) {
        association_diagnostics_.measurement_stamp = measurements.time_stamp;
        association_diagnostics_.input_count = measurements.meas.size();
        association_diagnostics_.processed_count = i_meas_num;
        association_diagnostics_.omitted_rows = 0;
        association_diagnostics_.rows.clear();
        association_diagnostics_.rows.reserve(mc_mot::AssociationDiagnostics::kMaximumRows);
        nearest_candidates.resize(i_meas_num);
    }
    Eigen::MatrixXd cost_matrix(i_meas_num, MAX_TRACKS);
    cost_matrix.setConstant(std::numeric_limits<double>::max());
    Eigen::MatrixXd ranking;
    const bool use_ranking = config_.pedestrian_gaussian_association_ranking ||
        config_.shape_consistent_association_ranking;
    if (use_ranking) {
        ranking = cost_matrix;
    }

    // Initialize is_associated flag for all tracks
    for (auto &track : all_tracks_) {
        track.is_associated = false;
    }

    double l2_distance, maha_distance;
    // Calculate cost matrix: rows: measurements, columns: tracks
    for (int meas_idx = 0; meas_idx < i_meas_num; ++meas_idx) {
        for (int track_idx = 0; track_idx < MAX_TRACKS; ++track_idx) {
            l2_distance = CalculateDistance(measurements.meas[meas_idx].state, all_tracks_[track_idx].state_vec);
            maha_distance = CalculateMahalanobisDistance(measurements.meas[meas_idx].state, all_tracks_[track_idx]);
            if (all_tracks_[track_idx].is_init == false || l2_distance > config_.max_association_dist_m)
                maha_distance = 1000.0; // Increase distance for uninitialized tracks to avoid pairing

            if ((measurements.meas[meas_idx].classification == mc_mot::ObjectClass::PEDESTRIAN &&
                 all_tracks_[track_idx].getRepClass() != mc_mot::ObjectClass::PEDESTRIAN) ||
                (all_tracks_[track_idx].getRepClass() == mc_mot::ObjectClass::PEDESTRIAN &&
                 measurements.meas[meas_idx].classification != mc_mot::ObjectClass::PEDESTRIAN)) {
                maha_distance = 1000.0; // Associate only among pedestrians
            }

            if (measurements.meas[meas_idx].classification == mc_mot::ObjectClass::PEDESTRIAN && l2_distance > 2.0) {
                maha_distance = 1000.0; // Shorter association distance for pedestrians
            }

            if (association_diagnostics_enabled_ && all_tracks_[track_idx].is_init &&
                (nearest_candidates[meas_idx].track_id < 0 ||
                 l2_distance < nearest_candidates[meas_idx].distance_m)) {
                const char* reason = "eligible_competitor";
                const bool measurement_is_pedestrian = measurements.meas[meas_idx].classification == mc_mot::ObjectClass::PEDESTRIAN;
                const bool track_is_pedestrian = all_tracks_[track_idx].getRepClass() == mc_mot::ObjectClass::PEDESTRIAN;
                if (measurement_is_pedestrian != track_is_pedestrian) reason = "class_gate";
                else if (l2_distance > config_.max_association_dist_m || (measurement_is_pedestrian && l2_distance > 2.0)) reason = "distance_gate";
                else if (!(maha_distance < config_.max_association_dist_m)) reason = "mahalanobis_gate";
                auto row = AssociationSnapshot("unmatched_candidate", reason, all_tracks_[track_idx], meas_idx, &measurements.meas[meas_idx]);
                row.distance_m = l2_distance;
                row.association_cost = maha_distance;
                nearest_candidates[meas_idx] = row;
            }

            cost_matrix(meas_idx, track_idx) = maha_distance;
            if (use_ranking) {
                ranking(meas_idx, track_idx) = maha_distance;
                if (config_.pedestrian_gaussian_association_ranking &&
                    measurements.meas[meas_idx].classification == mc_mot::ObjectClass::PEDESTRIAN &&
                    all_tracks_[track_idx].is_init && maha_distance < config_.max_association_dist_m) {
                    const auto &track = all_tracks_[track_idx];
                    const auto &observation = measurements.meas[meas_idx];
                    Eigen::Vector2d innovation;
                    innovation << observation.state.x - track.state_vec(S_X),
                                  observation.state.y - track.state_vec(S_Y);
                    ranking(meas_idx, track_idx) = mc_mot::GaussianXYAssociationRank(
                        innovation, track.state_cov.block<2, 2>(S_X, S_X),
                        R_.block<2, 2>(0, 0), observation.detection_confidence, maha_distance);
                }
            }
            if (association_diagnostics_enabled_ && nearest_candidates[meas_idx].track_id == track_idx) {
                nearest_candidates[meas_idx].ranking_cost = use_ranking ? ranking(meas_idx, track_idx) : maha_distance;
            }
        }
    }

    if (config_.shape_consistent_association_ranking) {
        // Do not remove an admitted edge, suppress a birth, or always prefer
        // old IDs. Only delay a tentative competitor behind an eligible,
        // shape-consistent explanation of the same occupied footprint.
        std::vector<int> mature_indices;
        for (int t = 0; t < MAX_TRACKS; ++t) {
            if (all_tracks_[t].is_init && all_tracks_[t].is_confirmed)
                mature_indices.push_back(t);
        }
        for (int m = 0; m < i_meas_num; ++m) {
            for (int t = 0; t < MAX_TRACKS; ++t) {
                if (!(cost_matrix(m, t) < config_.max_association_dist_m) ||
                    !all_tracks_[t].is_init || all_tracks_[t].is_confirmed) continue;
                double preferred_rank = std::numeric_limits<double>::infinity();
                for (const int stable : mature_indices) {
                    if (cost_matrix(m, stable) < config_.max_association_dist_m &&
                        IsShapeConflictingCompetitor(measurements.meas[m],
                            all_tracks_[t], all_tracks_[stable])) {
                        preferred_rank = std::min(preferred_rank, ranking(m, stable));
                    }
                }
                if (std::isfinite(preferred_rank)) {
                    ranking(m, t) = std::max(ranking(m, t), preferred_rank + 1.e-6);
                }
            }
            if (association_diagnostics_enabled_ && nearest_candidates[m].track_id >= 0) {
                nearest_candidates[m].ranking_cost = ranking(m, nearest_candidates[m].track_id);
            }
        }
    }

    std::vector<int> assignment; // Initialized with i_meas_num. Stores associated track index for each element
    std::vector<int> assignment_track;
    MatchPairs(cost_matrix, assignment, assignment_track,
               use_ranking ? &ranking : nullptr);

    int init_count = 0;
    int asso_count = 0;
    // Update all associated tracks first.  Birth suppression must inspect the
    // complete set of tracks associated in this frame; processing measurements
    // in input order could otherwise create a duplicate before the matched
    // measurement later in the array marks its mature track as associated.
    for (int meas_idx = 0; meas_idx < i_meas_num; ++meas_idx) {
        int track_idx = assignment[meas_idx];

        if (track_idx != -1 && all_tracks_[track_idx].is_init == true) {
            double diagnostic_distance = 0.0;
            double diagnostic_prior_age = 0.0;
            bool was_confirmed = false;
            if (association_diagnostics_enabled_) {
                was_confirmed = all_tracks_[track_idx].is_confirmed;
                diagnostic_distance = CalculateDistance(measurements.meas[meas_idx].state, all_tracks_[track_idx].state_vec);
                diagnostic_prior_age = recent_timestamp_ - all_tracks_[track_idx].last_measurement_time;
            }
            UpdateTrack(all_tracks_[track_idx], measurements.meas[meas_idx]);

            // KF can be performed if there are at least 2 valid detections
            if (all_tracks_[track_idx].age >= 3 && all_tracks_[track_idx].countDetectionNum() >= 2) {
                all_tracks_[track_idx].is_confirmed = true;
            }
            if (association_diagnostics_enabled_) {
                const char* reason = !was_confirmed && all_tracks_[track_idx].is_confirmed
                    ? "confirmed" : (all_tracks_[track_idx].class_transition_state.phase ==
                        class_transition_guard::Phase::Pending ? "class_shape_pending" : "update");
                auto row = AssociationSnapshot("matched", reason, all_tracks_[track_idx], meas_idx, &measurements.meas[meas_idx]);
                row.distance_m = diagnostic_distance;
                row.association_cost = cost_matrix(meas_idx, track_idx);
                row.ranking_cost = use_ranking ? ranking(meas_idx, track_idx) : cost_matrix(meas_idx, track_idx);
                row.prior_measurement_age_sec = diagnostic_prior_age;
                row.track_assigned_measurement_index = meas_idx;
                AppendAssociationDiagnostic(row);
            }

            asso_count++;
        }
    }

    // Only after all matched tracks have been updated, consider unmatched
    // measurements as new-track candidates.
    for (int meas_idx = 0; meas_idx < i_meas_num; ++meas_idx) {
        const int track_idx = assignment[meas_idx];
        if (track_idx != -1 && all_tracks_[track_idx].is_init == true) {
            continue;
        }
        if (association_diagnostics_enabled_) {
            auto row = nearest_candidates[meas_idx];
            row.measurement_index = meas_idx;
            row.measurement_class = measurements.meas[meas_idx].classification;
            row.measurement_score = measurements.meas[meas_idx].detection_confidence;
            if (row.track_id < 0) {
                row.event = "unmatched_candidate";
                row.reason = "no_existing_track";
            } else {
                row.track_assigned_measurement_index = assignment_track[row.track_id];
            }
            AppendAssociationDiagnostic(row);
        }
        int blocking_track_id = -1;
        const bool legacy_duplicate = ShouldSuppressDuplicateTrackBirth(
            measurements.meas[meas_idx], association_diagnostics_enabled_ ? &blocking_track_id : nullptr);
        const bool small_raw_duplicate = !legacy_duplicate && ShouldSuppressSmallUnknownDuplicateBirth(
            measurements.meas[meas_idx], measurements, assignment_track,
            association_diagnostics_enabled_ ? &blocking_track_id : nullptr);
        if (legacy_duplicate || small_raw_duplicate) {
            if (association_diagnostics_enabled_) {
                auto row = AssociationSnapshot("birth_suppressed", small_raw_duplicate
                    ? "small_raw_overlap_associated_unknown" : "near_associated_mature_track",
                    all_tracks_[blocking_track_id], meas_idx, &measurements.meas[meas_idx]);
                row.distance_m = CalculateDistance(measurements.meas[meas_idx].state, all_tracks_[blocking_track_id].state_vec);
                AppendAssociationDiagnostic(row);
            }
            continue;
        }
        mc_mot::TrackStruct new_track;

        InitTrack(new_track, measurements.meas[meas_idx]);

        all_tracks_[cur_track_id_] = new_track;
        if (association_diagnostics_enabled_) {
            AppendAssociationDiagnostic(AssociationSnapshot("birth", "unmatched_measurement", new_track, meas_idx, &measurements.meas[meas_idx]));
        }

        UpdateTrackId(); // Increment cur_track_id_
        init_count++;
    }

    // Update information for unassociated tracks
    int i_deleted_num = 0;
    for (auto &track : all_tracks_) {
        if (track.is_associated == false) {
            track.class_transition_state.Reset();
            track.age++;
            track.updateDetectionCount(false);

            // Reset outdated tracks
            if (track.is_init == true &&
                ShouldDeleteUnassociatedTrack(track, measurements.time_stamp)) {
                if (association_diagnostics_enabled_) AppendAssociationDiagnostic(AssociationSnapshot("delete", track.is_confirmed && config_.time_aware_track_lifecycle ? "coast_timeout" : "tentative_outdated", track));
                track.reset();
                i_deleted_num++;
            }
            else if (sqrt(track.state_vec(S_VX) * track.state_vec(S_VX) +
                          track.state_vec(S_VY) * track.state_vec(S_VY)) > MAX_TRACK_VEL) {
                if (association_diagnostics_enabled_ && track.is_init) AppendAssociationDiagnostic(AssociationSnapshot("delete", "excess_velocity", track));
                // Reset tracks with excessive velocity
                track.reset();
                i_deleted_num++;
            }
            else if (sqrt(track.state_vec(S_AX) * track.state_vec(S_AX) +
                          track.state_vec(S_AY) * track.state_vec(S_AY)) > MAX_TRACK_ACC) {
                if (association_diagnostics_enabled_ && track.is_init) AppendAssociationDiagnostic(AssociationSnapshot("delete", "excess_acceleration", track));
                // Reset tracks with excessive acceleration
                track.reset();
                i_deleted_num++;
            }
            else if (association_diagnostics_enabled_ && track.is_init) {
                AppendAssociationDiagnostic(AssociationSnapshot("coast", track.is_confirmed ? "confirmed_unmatched" : "tentative_unmatched", track));
            }
        }
    }

    // FIXME:  ======= For debugging =============
    int init_track_num = 0;
    int confirmed_track_num = 0;
    for (const auto &track : all_tracks_) {
        if (track.is_init) init_track_num++;
        if (track.is_confirmed) confirmed_track_num++;
    }
    ROS_DEBUG_STREAM("[EKF tracker] det=" << i_meas_num << " associated=" << asso_count
                     << " new=" << init_count << " deleted=" << i_deleted_num
                     << " active=" << init_track_num << " confirmed=" << confirmed_track_num);

    // ======= For debugging =============
}

mc_mot::TrackStructs EkfMultiObjectTracking::GetTrackResults() const{

    mc_mot::TrackStructs o_track_results;

    o_track_results.time_stamp = recent_timestamp_;

    for (auto &track : all_tracks_) {
        mc_mot::TrackStruct o_track;
        o_track = track;
        // double track_vel = sqrt(o_track.state_vec(S_VX) * o_track.state_vec(S_VX) +
        //                         o_track.state_vec(S_VY) * o_track.state_vec(S_VY));
        // if (track_vel < 1.0 && o_track.classification != mc_mot::ObjectClass::PEDESTRIAN) {
        //     o_track.state_vec(S_VX) = 0.0;
        //     o_track.state_vec(S_VY) = 0.0;
        //     o_track.state_vec(S_YAW_RATE) = 0.0;
        //     o_track.state_vec(S_AX) = 0.0;
        //     o_track.state_vec(S_AY) = 0.0;
        // }
        HoldUnknownCoastingPose(o_track, config_);
        o_track_results.track.push_back(o_track);
    }

    return o_track_results;
}

mc_mot::TrackStructs EkfMultiObjectTracking::GetPredictedTrackResults(
        const double target_timestamp) {
    // Project raw copied state before applying output-only pose policies.
    // Starting from an already-held yaw could change predicted covariance.
    mc_mot::TrackStructs results;
    results.time_stamp = recent_timestamp_;
    for (const auto &track : all_tracks_) results.track.push_back(track);
    const double dt = target_timestamp - results.time_stamp;
    if (dt <= 0.0) {
        for (auto &track : results.track) HoldUnknownCoastingPose(track, config_);
        return results;
    }
    for (auto& track : results.track) {
        if (!track.is_init) continue;
        const bool hold_pose = config_.hold_unknown_output_pose &&
            config_.global_coord_track && track.getRepClass() == mc_mot::UNKNOWN;
        const double measured_x = track.state_vec(S_X);
        const double measured_y = track.state_vec(S_Y);
        const double measured_yaw = track.state_vec(S_YAW);
        PredictTrack(track, dt);
        if (hold_pose) {
            // Retain predicted uncertainty; only the copied pose is held.
            // The internal measurement-time filter and estimated velocity
            // remain diagnostic, and a later CAR label projects normally.
            track.state_vec(S_X) = measured_x;
            track.state_vec(S_Y) = measured_y;
            track.state_vec(S_YAW) = measured_yaw;
        }
        HoldUnknownCoastingPose(track, config_);
    }
    results.time_stamp = target_timestamp;
    return results;
}

void EkfMultiObjectTracking::UpdateConfig(const MultiClassObjectTrackingConfig config) {
    std::cout << "Update Config !" << std::endl;
    config_ = config;
    UpdateMatrix();
}

// Utils

void EkfMultiObjectTracking::PredictTrack(mc_mot::TrackStruct &track, double dt) {
    if (config_.stationary_unknown_prediction && config_.global_coord_track &&
        track.getRepClass() == mc_mot::UNKNOWN && !track.has_known_class_measurement) {
        // Identity transition: pose does not depend on a noisy hidden velocity.
        // Keep latent dynamic state/uncertainty for later class recovery, rather
        // than resetting the EKF or freezing actual measurement corrections.
        // Existing process velocity noise supplies a dt-aware random-walk
        // allowance for changes in a static object's observed center.
        Eigen::Matrix8_8d stationary_noise = Q_;
        const double displacement_std = config_.system_noise_std_vx_vy_ms * dt;
        stationary_noise(S_X, S_X) += displacement_std * displacement_std;
        stationary_noise(S_Y, S_Y) += displacement_std * displacement_std;
        track.state_cov += stationary_noise;
        track.update_time += dt;
        return;
    }
    double track_vel =
            sqrt(track.state_vec(S_VX) * track.state_vec(S_VX) + track.state_vec(S_VY) * track.state_vec(S_VY));

    // Retain only the velocity in the vehicle heading direction
    if (config_.use_kinematic_model == true) {
        double heading_align_vel =
                track.state_vec(S_VX) * cos(track.state_vec[S_YAW]) + track.state_vec(S_VY) * sin(track.state_vec[S_YAW]);
        double heading_align_vx = heading_align_vel * cos(track.state_vec[S_YAW]);
        double heading_align_vy = heading_align_vel * sin(track.state_vec[S_YAW]);

        track.state_vec(S_VX) = heading_align_vx;
        track.state_vec(S_VY) = heading_align_vy;
    }

    // State transition matrix (Jacobian)
    Eigen::Matrix8_8d F = Eigen::Matrix8_8d::Identity();

    if (track.getRepClass() == mc_mot::ObjectClass::PEDESTRIAN) {
        F(S_X, S_VX) = dt; // x' = x + vx * dt
        F(S_Y, S_VY) = dt; // y' = y + vy * dt
    }
    else if (config_.prediction_model == mc_mot::PredictionModel::CV) {
        F(S_X, S_VX) = dt;         // x' = x + vx * dt
        F(S_Y, S_VY) = dt;         // y' = y + vy * dt
        F(S_YAW, S_YAW_RATE) = dt; // yaw' = yaw + yaw_rate * dt
    }
    else if (config_.prediction_model == mc_mot::PredictionModel::CTRV) {
        double delta_theta = track.state_vec(S_YAW_RATE) * dt;

        // Calculate rotation matrix
        double cos_del_theta = std::cos(delta_theta);
        double sin_del_theta = std::sin(delta_theta);

        double vx_rotated = cos_del_theta * track.state_vec(S_VX) - sin_del_theta * track.state_vec(S_VY);
        double vy_rotated = sin_del_theta * track.state_vec(S_VX) + cos_del_theta * track.state_vec(S_VY);

        track.state_vec(S_VX) = vx_rotated;
        track.state_vec(S_VY) = vy_rotated;

        F(S_X, S_VX) = dt;         // x' = x + vx * dt
        F(S_Y, S_VY) = dt;         // y' = y + vy * dt
        F(S_YAW, S_YAW_RATE) = dt; // yaw' = yaw + yaw_rate * dt
    }
    else if (config_.prediction_model == mc_mot::PredictionModel::CA) {
        F(S_X, S_VX) = dt;            // x' = x + vx * dt
        F(S_X, S_AX) = 0.5 * dt * dt; // x' = x + 0.5 * ax * dt^2
        F(S_Y, S_VY) = dt;            // y' = y + vy * dt
        F(S_Y, S_AY) = 0.5 * dt * dt; // y' = y + 0.5 * ay * dt^2
        F(S_YAW, S_YAW_RATE) = dt;    // yaw' = yaw + yaw_rate * dt
        F(S_VX, S_AX) = dt;           // vx' = vx + ax * dt
        F(S_VY, S_AY) = dt;           // vy' = vy + ay * dt
    }
    else { // CTRA
        double delta_theta = track.state_vec(S_YAW_RATE) * dt;

        // Calculate rotation matrix
        double cos_del_theta = std::cos(delta_theta);
        double sin_del_theta = std::sin(delta_theta);

        double vx_rotated = cos_del_theta * track.state_vec(S_VX) - sin_del_theta * track.state_vec(S_VY);
        double vy_rotated = sin_del_theta * track.state_vec(S_VX) + cos_del_theta * track.state_vec(S_VY);

        track.state_vec(S_VX) = vx_rotated;
        track.state_vec(S_VY) = vy_rotated;

        F(S_X, S_VX) = dt;            // x' = x + vx * dt
        F(S_X, S_AX) = 0.5 * dt * dt; // x' = x + 0.5 * ax * dt^2
        F(S_Y, S_VY) = dt;            // y' = y + vy * dt
        F(S_Y, S_AY) = 0.5 * dt * dt; // y' = y + 0.5 * ay * dt^2
        F(S_YAW, S_YAW_RATE) = dt;    // yaw' = yaw + yaw_rate * dt
        F(S_VX, S_AX) = dt;           // vx' = vx + ax * dt
        F(S_VY, S_AY) = dt;           // vy' = vy + ay * dt
    }

    // Disable yaw prediction at low speeds
    if (track_vel < 3.0) {
        F(S_YAW, S_YAW_RATE) = 0.0;
    }

    // Add direction matrix to add larger covariance in the direction of travel
    Eigen::Matrix8_8d Q = Q_;

    // Add larger covariance in the direction of velocity
    double angle = atan2(track.state_vec(S_VY), track.state_vec(S_VX));
    Eigen::Matrix2d rot_mat;
    rot_mat << cos(angle), -sin(angle), sin(angle), cos(angle);

    Eigen::Matrix2d direction_cov;
    direction_cov << std::max(track_vel * 10, 1.0), 0.0, // Larger covariance in the direction of travel
            0.0, 1.0;                                    // Covariance in the direction perpendicular to travel

    Eigen::Matrix2d Q_skew = Q_.block<2, 2>(S_X, S_X).cwiseProduct(direction_cov);
    Eigen::Matrix2d Q_skew_rot = rot_mat * Q_skew * rot_mat.transpose();

    Q.block<2, 2>(0, 0) = Q_skew_rot;

    // Predict state vector
    track.state_vec = F * track.state_vec;
    // Predict covariance matrix
    track.state_cov = F * track.state_cov * F.transpose() + Q;

    track.update_time += dt;
}

void EkfMultiObjectTracking::UpdateTrack(mc_mot::TrackStruct &track, const mc_mot::Meastruct &measurement) {
    // Be conservative about dynamic evidence: even one associated known-class
    // measurement restores the original motion model for this track's lifetime.
    // Do not rely on the representative-class EMA, which can decay to UNKNOWN.
    track.has_known_class_measurement = track.has_known_class_measurement ||
        measurement.classification == mc_mot::CAR ||
        measurement.classification == mc_mot::TRUCK ||
        measurement.classification == mc_mot::PEDESTRIAN ||
        measurement.classification == mc_mot::BICYCLE;
    const auto class_decision = class_transition_guard::Evaluate(
        config_.class_transition_guard_config, track.class_transition_state,
        track.is_confirmed, track.getRepClass(),
        {{track.dimension.length, track.dimension.width, track.dimension.height}},
        measurement.classification,
        {{measurement.dimension.length, measurement.dimension.width, measurement.dimension.height}},
        measurement.state.time_stamp);
    Eigen::Vector3d measurement_vec;
    measurement_vec << measurement.state.x, measurement.state.y, measurement.state.yaw;

    // Calculate Kalman gain
    Eigen::Matrix3d R = R_;

    // Dot product between measurement heading and track heading (-1.0 ~ 1.0)
    // -1.0: opposite direction, 1.0: same direction
    double meas_track_yaw_inner = CalculateYawDotProduct(measurement_vec(2), track.state_vec(S_YAW));

    // Track velocity
    double track_vel =
            sqrt(track.state_vec(S_VX) * track.state_vec(S_VX) + track.state_vec(S_VY) * track.state_vec(S_VY));

    // Update direction score by comparing measurement heading and track heading
    track.direction_score = config_.dimension_filter_alpha * meas_track_yaw_inner +
                            (1.0 - config_.dimension_filter_alpha) * track.direction_score;

    if (meas_track_yaw_inner < -cos(M_PI / 4.0)) {
        // Track heading is incorrect. Flip track heading and reset direction score to 0.5.
        if (track.direction_score < 0) {
            track.state_vec(S_YAW) += M_PI;
            track.direction_score = 0.5;
        }
        else {
            // Track heading is correct but measurement heading is incorrect. Flip measurement heading.
            measurement_vec(2) += M_PI;
        }
    }

    // Increase measurement uncertainty if detection confidence is low
    if (measurement.detection_confidence < 0.5) {
        R = 10.0 * R;
    }

    Eigen::Matrix3d S = H_ * track.state_cov * H_.transpose() + R;
    Eigen::Matrix8_3d K = track.state_cov * H_.transpose() * S.inverse();

    // Yaw normalization
    while (measurement_vec(2) - track.state_vec(S_YAW) > M_PI) {
        measurement_vec(2) -= 2.0 * M_PI;
    }
    while (measurement_vec(2) - track.state_vec(S_YAW) < -M_PI) {
        measurement_vec(2) += 2.0 * M_PI;
    }

    // Update state vector
    track.state_vec += K * (measurement_vec - H_ * track.state_vec);

    // Update covariance matrix
    Eigen::Matrix8_8d I = Eigen::Matrix8_8d::Identity();
    track.state_cov = (I - K * H_) * track.state_cov;

    const double track_age_sec = std::max(
        0.0, measurement.state.time_stamp - track.initialization_time);
    const bool use_detector_velocity_update =
        config_.detection_velocity_fusion_mode == 2 ||
        (config_.detection_velocity_fusion_mode == 3 &&
         track_age_sec <=
             config_.detection_velocity_update_max_track_age_sec);
    if (measurement.has_velocity && use_detector_velocity_update) {
        FuseVelocity(
            track,
            measurement.state.v_x,
            measurement.state.v_y,
            config_.detection_velocity_noise_std_mps);
    }
    if (measurement.has_velocity &&
        (config_.detection_velocity_fusion_mode == 4 ||
         config_.detection_velocity_fusion_mode == 5) &&
        (measurement.classification == mc_mot::ObjectClass::CAR ||
         measurement.classification == mc_mot::ObjectClass::TRUCK)) {
        const double heading_x = std::cos(track.state_vec(S_YAW));
        const double heading_y = std::sin(track.state_vec(S_YAW));
        const double detector_longitudinal =
            heading_x * measurement.state.v_x +
            heading_y * measurement.state.v_y;
        const double tracker_longitudinal =
            heading_x * track.state_vec(S_VX) +
            heading_y * track.state_vec(S_VY);
        if (std::abs(detector_longitudinal - tracker_longitudinal) <=
            config_.detection_velocity_max_longitudinal_innovation_mps) {
            track.has_detector_velocity = true;
            track.detector_velocity_x = measurement.state.v_x;
            track.detector_velocity_y = measurement.state.v_y;
            track.detector_velocity_time = measurement.state.time_stamp;
            track.detector_velocity_variance_mps2 =
                measurement.velocity_variance_mps2;
        }
    }

    UpdateCenterMotionVelocity(track, measurement);

    // Update track attributes
    track.is_associated = true;
    track.updateDetectionCount(true);
    track.update_time = measurement.state.time_stamp;
    track.last_measurement_time = measurement.state.time_stamp;
    track.last_observed_output_pose = track.state_vec.head<3>();
    track.has_observed_output_pose = true;
    track.detection_confidence = config_.dimension_filter_alpha * measurement.detection_confidence +
                                 (1.0 - config_.dimension_filter_alpha) * track.detection_confidence;
    track.age++;

    // A pending cross-family shape discontinuity does not change the ordinary
    // position/velocity update. Only dimensions and class evidence are deferred.
    if (!class_decision.hold_shape) {
      track.dimension.length = config_.dimension_filter_alpha * measurement.dimension.length +
                             (1.0 - config_.dimension_filter_alpha) * track.dimension.length;
    track.dimension.width = config_.dimension_filter_alpha * measurement.dimension.width +
                            (1.0 - config_.dimension_filter_alpha) * track.dimension.width;
    track.dimension.height = config_.dimension_filter_alpha * measurement.dimension.height +
                             (1.0 - config_.dimension_filter_alpha) * track.dimension.height;
    }

    track.object_z = config_.dimension_filter_alpha * measurement.state.z +
                     (1.0 - config_.dimension_filter_alpha) * track.object_z;

    // Class Filtering
    const bool promote_vehicle = class_decision.vote_count > 0 &&
        config_.fast_confirmed_vehicle_promotion &&
        track.is_confirmed && track.getRepClass() == mc_mot::ObjectClass::UNKNOWN &&
        (measurement.classification == mc_mot::ObjectClass::CAR ||
         measurement.classification == mc_mot::ObjectClass::TRUCK) &&
        std::isfinite(measurement.detection_confidence) &&
        measurement.detection_confidence >= config_.vehicle_promotion_min_confidence;
    if (promote_vehicle) {
        // Scores remain a heuristic, not calibrated existence probabilities.
        // Preserve uncertainty instead of forcing the vehicle score to 1.0.
        const double probability = std::max(0.51, std::min(1.0,
            static_cast<double>(measurement.detection_confidence)));
        std::fill(track.class_score_arr, track.class_score_arr + CLASS_NUM, 0.0);
        track.class_score_arr[mc_mot::ObjectClass::UNKNOWN] = 1.0 - probability;
        track.class_score_arr[measurement.classification] = probability;
    } else {
        for (std::size_t vote = 0; vote < class_decision.vote_count; ++vote) {
            track.updateClassScore(class_decision.class_votes[vote]);
        }
    }

    // Yaw rate Filtering based on Kinematic Model. Only available in global track mode
    if (config_.use_yaw_rate_filtering && config_.global_coord_track == true &&
        (track.getRepClass() == mc_mot::ObjectClass::CAR || track.getRepClass() == mc_mot::ObjectClass::TRUCK)) {
        double vel_heading_dot =
                CalculateYawDotProduct(atan2(track.state_vec(S_VY), track.state_vec(S_VX)), track.state_vec(S_YAW));
        double vel_heading_cross =
                CalculateYawCrossProduct(track.state_vec(S_YAW), atan2(track.state_vec(S_VY), track.state_vec(S_VX)));
        // if vel_heading_dot > 0, forward, if vel_heading_dot < 0, backward
        // if vel_heading_cross > 0, turn left, if vel_heading_cross < 0, turn right

        double heading_align_vel_ms = vel_heading_dot * track_vel;
        double target_wheel_base = track.dimension.length * 0.7; // Assume wheelbase is 0.7 times the vehicle length
        double max_yaw_rate_rad = heading_align_vel_ms * tan(config_.max_steer_deg * M_PI / 180.0) / target_wheel_base;

        // Limit track yaw rate if it exceeds max_yaw_rate_rad
        if (fabs(track.state_vec(S_YAW_RATE)) > fabs(max_yaw_rate_rad)) {
            track.state_vec(S_YAW_RATE) *= fabs(max_yaw_rate_rad) / fabs(track.state_vec(S_YAW_RATE));
        }

        // Set yaw rate to 0 if track yaw rate direction differs from velocity direction
        if (track.state_vec(S_YAW_RATE) * vel_heading_cross < 0) {
            track.state_vec(S_YAW_RATE) = 0.0;
        }
    }
}

void EkfMultiObjectTracking::FuseVelocity(
        mc_mot::TrackStruct &track,
        const double velocity_x,
        const double velocity_y,
        const double noise_std_mps) {
        Eigen::Matrix<double, 2, 8> velocity_h =
            Eigen::Matrix<double, 2, 8>::Zero();
        velocity_h(0, S_VX) = 1.0;
        velocity_h(1, S_VY) = 1.0;
        Eigen::Vector2d velocity_measurement(velocity_x, velocity_y);
        const double variance = noise_std_mps * noise_std_mps;
        Eigen::Matrix2d velocity_r = variance * Eigen::Matrix2d::Identity();
        Eigen::Matrix2d velocity_s =
            velocity_h * track.state_cov * velocity_h.transpose() + velocity_r;
        Eigen::Matrix<double, 8, 2> velocity_k =
            track.state_cov * velocity_h.transpose() * velocity_s.inverse();
        track.state_vec += velocity_k *
            (velocity_measurement - velocity_h * track.state_vec);
        Eigen::Matrix8_8d identity = Eigen::Matrix8_8d::Identity();
        track.state_cov = (identity - velocity_k * velocity_h) * track.state_cov;
}

void EkfMultiObjectTracking::UpdateCenterMotionVelocity(
        mc_mot::TrackStruct &track,
        const mc_mot::Meastruct &measurement) {
    if (!config_.center_motion_velocity_fusion || !config_.global_coord_track ||
        (measurement.classification != mc_mot::ObjectClass::CAR &&
         measurement.classification != mc_mot::ObjectClass::TRUCK)) {
        return;
    }

    auto &history = track.center_history;
    const mc_mot::CenterObservation current{
        measurement.state.time_stamp, measurement.state.x, measurement.state.y};

    if (!history.empty()) {
        const double gap = current.time_stamp - history.back().time_stamp;
        if (gap <= 0.0 || gap > config_.center_motion_max_gap_sec) {
            history.clear();
        }
    }
    history.push_back(current);

    const double keep_sec = config_.center_motion_long_baseline_sec +
                            config_.center_motion_max_gap_sec;
    while (history.size() > 1 &&
           current.time_stamp - history.front().time_stamp > keep_sec) {
        history.pop_front();
    }

    if (history.size() < 2) return;
    const double available_span = current.time_stamp - history.front().time_stamp;
    if (available_span < config_.center_motion_min_baseline_sec) return;

    double target_baseline = 0.45;
    if (available_span >= config_.center_motion_long_baseline_sec) {
        target_baseline = config_.center_motion_long_baseline_sec;
    } else if (available_span >= config_.center_motion_medium_baseline_sec) {
        target_baseline = config_.center_motion_medium_baseline_sec;
    }

    const mc_mot::CenterObservation *reference = nullptr;
    double best_error = std::numeric_limits<double>::max();
    for (const auto &candidate : history) {
        const double baseline = current.time_stamp - candidate.time_stamp;
        if (baseline < config_.center_motion_min_baseline_sec ||
            baseline > config_.center_motion_long_baseline_sec + 1.0e-6) {
            continue;
        }
        const double error = std::abs(baseline - target_baseline);
        if (error < best_error) {
            best_error = error;
            reference = &candidate;
        }
    }
    if (reference == nullptr) return;

    std::vector<const mc_mot::CenterObservation *> fit_observations;
    for (const auto &candidate : history) {
        if (current.time_stamp - candidate.time_stamp <=
            config_.center_motion_long_baseline_sec + 1.0e-6) {
            fit_observations.push_back(&candidate);
        }
    }
    if (fit_observations.size() < 3) return;

    double mean_t = 0.0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (const auto *item : fit_observations) {
        mean_t += item->time_stamp;
        mean_x += item->x;
        mean_y += item->y;
    }
    const double count = static_cast<double>(fit_observations.size());
    mean_t /= count;
    mean_x /= count;
    mean_y /= count;

    double time_variance = 0.0;
    double time_x_covariance = 0.0;
    double time_y_covariance = 0.0;
    for (const auto *item : fit_observations) {
        const double centered_t = item->time_stamp - mean_t;
        time_variance += centered_t * centered_t;
        time_x_covariance += centered_t * (item->x - mean_x);
        time_y_covariance += centered_t * (item->y - mean_y);
    }
    if (time_variance <= 1.0e-9) return;
    const double fit_vx = time_x_covariance / time_variance;
    const double fit_vy = time_y_covariance / time_variance;
    double squared_residual_sum = 0.0;
    for (const auto *item : fit_observations) {
        const double dt = item->time_stamp - mean_t;
        const double residual_x = item->x - (mean_x + fit_vx * dt);
        const double residual_y = item->y - (mean_y + fit_vy * dt);
        squared_residual_sum +=
            residual_x * residual_x + residual_y * residual_y;
    }
    const double fit_residual = std::sqrt(squared_residual_sum / count);
    if (!std::isfinite(fit_residual) ||
        fit_residual > config_.center_motion_max_fit_residual_m) {
        return;
    }

    if (track.last_center_motion_fusion_time > 0.0 &&
        current.time_stamp - track.last_center_motion_fusion_time <
            config_.center_motion_min_update_interval_sec) {
        return;
    }

    const double baseline = current.time_stamp - reference->time_stamp;
    const double velocity_x = (current.x - reference->x) / baseline;
    const double velocity_y = (current.y - reference->y) / baseline;
    const double speed = std::hypot(velocity_x, velocity_y);
    if (!std::isfinite(speed) || speed > MAX_TRACK_VEL) {
        history.clear();
        history.push_back(current);
        return;
    }

    const double maturity = std::max(
        0.0,
        std::min(
            1.0,
            (baseline - config_.center_motion_min_baseline_sec) /
                std::max(
                    1.0e-6,
                    config_.center_motion_long_baseline_sec -
                        config_.center_motion_min_baseline_sec)));
    const double noise_std =
        config_.center_motion_early_noise_std_mps + maturity *
        (config_.center_motion_mature_noise_std_mps -
         config_.center_motion_early_noise_std_mps);
    track.has_center_motion_velocity = true;
    track.center_motion_velocity_x = velocity_x;
    track.center_motion_velocity_y = velocity_y;
    track.center_motion_velocity_noise_std_mps = noise_std;
    if (config_.center_motion_fuse_ekf_state) {
        FuseVelocity(track, velocity_x, velocity_y, noise_std);
    }
    track.last_center_motion_fusion_time = current.time_stamp;
}

void EkfMultiObjectTracking::InitTrack(mc_mot::TrackStruct &track, const mc_mot::Meastruct &measurement) {
    track.track_id = cur_track_id_;
    track.initialization_time = measurement.state.time_stamp;
    track.update_time = measurement.state.time_stamp;
    track.last_measurement_time = measurement.state.time_stamp;
    track.detection_confidence = measurement.detection_confidence;
    track.has_known_class_measurement =
        measurement.classification == mc_mot::CAR ||
        measurement.classification == mc_mot::TRUCK ||
        measurement.classification == mc_mot::PEDESTRIAN ||
        measurement.classification == mc_mot::BICYCLE;

    track.state_vec(S_X) = measurement.state.x;
    track.state_vec(S_Y) = measurement.state.y;
    track.state_vec(S_YAW) = measurement.state.yaw;
    track.last_observed_output_pose = track.state_vec.head<3>();
    track.has_observed_output_pose = true;
    if (measurement.has_velocity &&
        config_.detection_velocity_fusion_mode >= 1 &&
        config_.detection_velocity_fusion_mode <= 3) {
        track.state_vec(S_VX) = measurement.state.v_x;
        track.state_vec(S_VY) = measurement.state.v_y;
        const double variance =
            config_.detection_velocity_noise_std_mps *
            config_.detection_velocity_noise_std_mps;
        track.state_cov(S_VX, S_VX) = variance;
        track.state_cov(S_VY, S_VY) = variance;
        track.state_cov(S_VX, S_VY) = 0.0;
        track.state_cov(S_VY, S_VX) = 0.0;
    }
    if (measurement.has_velocity &&
        (config_.detection_velocity_fusion_mode == 4 ||
         config_.detection_velocity_fusion_mode == 5) &&
        (measurement.classification == mc_mot::ObjectClass::CAR ||
         measurement.classification == mc_mot::ObjectClass::TRUCK)) {
        track.has_detector_velocity = true;
        track.detector_velocity_x = measurement.state.v_x;
        track.detector_velocity_y = measurement.state.v_y;
        track.detector_velocity_time = measurement.state.time_stamp;
        track.detector_velocity_variance_mps2 =
            measurement.velocity_variance_mps2;
    }

    Eigen::Matrix3d S = H_ * track.state_cov * H_.transpose() + R_;
    Eigen::Matrix8_3d K = track.state_cov * H_.transpose() * S.inverse();

    // Update covariance matrix
    Eigen::Matrix8_8d I = Eigen::Matrix8_8d::Identity();
    track.state_cov = (I - K * H_) * track.state_cov;

    track.dimension = measurement.dimension;
    track.class_score_arr[mc_mot::ObjectClass(measurement.classification)] = 1.0;
    track.object_z = measurement.state.z;
    track.age = 0;

    track.is_init = true;
    track.is_associated = false; // TODO:

    if (config_.center_motion_velocity_fusion && config_.global_coord_track &&
        (measurement.classification == mc_mot::ObjectClass::CAR ||
         measurement.classification == mc_mot::ObjectClass::TRUCK)) {
        track.center_history.push_back(mc_mot::CenterObservation{
            measurement.state.time_stamp,
            measurement.state.x,
            measurement.state.y});
    }

    track.updateDetectionCount(true);
}

void EkfMultiObjectTracking::MatchPairs(const Eigen::MatrixXd &cost_matrix, std::vector<int> &row_indices,
                                          std::vector<int> &col_indices, const Eigen::MatrixXd *ranking) {
    // Initialize variables
    int num_rows = cost_matrix.rows(); // Meas num
    int num_cols = cost_matrix.cols(); // Track num
    row_indices.assign(num_rows, -1);
    col_indices.assign(num_cols, -1);
    std::vector<bool> row_assigned(num_rows, false);
    std::vector<bool> col_assigned(num_cols, false);

    // Create a list of all possible pairs (row, col) and sort them by cost
    struct Pair {
        int row;
        int col;
        double cost;
        bool operator<(const Pair &other) const { return cost < other.cost; }
    };

    std::vector<Pair> pairs;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            // Add only those within the distance threshold to pairs
            if (cost_matrix(i, j) < config_.max_association_dist_m) {
                pairs.push_back({i, j, ranking ? (*ranking)(i, j) : cost_matrix(i, j)});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end());

    // Greedily assign pairs with the smallest cost
    for (const auto &pair : pairs) {
        if (!row_assigned[pair.row] && !col_assigned[pair.col]) {
            row_indices[pair.row] = pair.col;
            col_indices[pair.col] = pair.row;
            row_assigned[pair.row] = true;
            col_assigned[pair.col] = true;
        }
    }
}

// Purpose: To find an available new track ID.
// If no empty track ID is found after a full loop, the current track ID will be removed and replaced as a last resort.
void EkfMultiObjectTracking::UpdateTrackId() {
    int i_cur_track_id = cur_track_id_;
    do {
        cur_track_id_++;
        if (cur_track_id_ >= MAX_TRACKS) {
            cur_track_id_ = 0;
        }

        if (cur_track_id_ == i_cur_track_id) {
            break; // If no space is found after a full loop, replace the track
        }
    } while (all_tracks_[cur_track_id_].is_init == true); // Increment track id if the next track number is occupied
}

void EkfMultiObjectTracking::UpdateMatrix() {
    Q_ = Eigen::Matrix8_8d::Zero();
    Q_.diagonal().array() = config_.system_noise_std_xy_m;
    Q_(S_X, S_X) = config_.system_noise_std_xy_m * config_.system_noise_std_xy_m;
    Q_(S_Y, S_Y) = config_.system_noise_std_xy_m * config_.system_noise_std_xy_m;
    Q_(S_YAW, S_YAW) = (config_.system_noise_std_yaw_deg * M_PI / 180.0) * 
                       (config_.system_noise_std_yaw_deg * M_PI / 180.0);
    Q_(S_VX, S_VX) = config_.system_noise_std_vx_vy_ms * config_.system_noise_std_vx_vy_ms;
    Q_(S_VY, S_VY) = config_.system_noise_std_vx_vy_ms * config_.system_noise_std_vx_vy_ms;
    Q_(S_YAW_RATE, S_YAW_RATE) = (config_.system_noise_std_yaw_rate_degs * M_PI / 180.0) *
                                 (config_.system_noise_std_yaw_rate_degs * M_PI / 180.0);
    Q_(S_AX, S_AX) = config_.system_noise_std_ax_ay_ms2 * config_.system_noise_std_ax_ay_ms2;
    Q_(S_AY, S_AY) = config_.system_noise_std_ax_ay_ms2 * config_.system_noise_std_ax_ay_ms2;

    R_ = Eigen::Matrix3d::Identity();
    R_(0, 0) = config_.meas_noise_std_xy_m * config_.meas_noise_std_xy_m; // x m
    R_(1, 1) = config_.meas_noise_std_xy_m * config_.meas_noise_std_xy_m; // y m
    R_(2, 2) = (config_.meas_noise_std_yaw_deg * M_PI / 180.0) *
               (config_.meas_noise_std_yaw_deg * M_PI / 180.0); // yaw rad

    H_ = Eigen::Matrix3_8d::Zero();
    H_(0, 0) = 1.0; // x
    H_(1, 1) = 1.0; // y
    H_(2, 2) = 1.0; // yaw
}

bool EkfMultiObjectTracking::AreLifecycleClassesCompatible(
        const mc_mot::ObjectClass measurement_class,
        const mc_mot::ObjectClass track_class) const {
    const bool measurement_is_pedestrian =
        measurement_class == mc_mot::ObjectClass::PEDESTRIAN;
    const bool track_is_pedestrian =
        track_class == mc_mot::ObjectClass::PEDESTRIAN;
    return measurement_is_pedestrian == track_is_pedestrian;
}

bool EkfMultiObjectTracking::ShouldDeleteUnassociatedTrack(
        const mc_mot::TrackStruct &track,
        const double measurement_time) const {
    if (!track.is_confirmed || !config_.time_aware_track_lifecycle) {
        return track.isOutdated();
    }

    const double speed = std::hypot(
        track.state_vec(S_VX), track.state_vec(S_VY));
    const auto representative_class =
        static_cast<mc_mot::ObjectClass>(track.getRepClass());
    const bool class_can_be_stationary_obstacle =
        representative_class == mc_mot::ObjectClass::UNKNOWN ||
        representative_class == mc_mot::ObjectClass::CAR ||
        representative_class == mc_mot::ObjectClass::TRUCK;
    const bool stationary =
        representative_class == mc_mot::ObjectClass::UNKNOWN ||
        (class_can_be_stationary_obstacle &&
         speed <= config_.lifecycle_stationary_speed_threshold_mps);
    const double maximum_coast_time = stationary
        ? config_.confirmed_stationary_track_max_coast_time_sec
        : config_.confirmed_track_max_coast_time_sec;
    return measurement_time - track.last_measurement_time > maximum_coast_time;
}

bool EkfMultiObjectTracking::ShouldSuppressDuplicateTrackBirth(
        const mc_mot::Meastruct &measurement,
        int* blocking_track_id) const {
    if (!config_.suppress_duplicate_track_birth) return false;

    const double measurement_max_dimension = std::max(
        measurement.dimension.length, measurement.dimension.width);
    if (measurement_max_dimension <
        config_.duplicate_birth_minimum_max_dimension_m) {
        return false;
    }

    for (const auto &track : all_tracks_) {
        if (!track.is_init || !track.is_confirmed || !track.is_associated) {
            continue;
        }
        if (!AreLifecycleClassesCompatible(
                measurement.classification,
                static_cast<mc_mot::ObjectClass>(track.getRepClass()))) {
            continue;
        }

        const double track_max_dimension = std::max(
            track.dimension.length, track.dimension.width);
        if (track_max_dimension <
            config_.duplicate_birth_minimum_max_dimension_m) {
            continue;
        }
        const double dimension_ratio = std::max(
            measurement_max_dimension / std::max(1.0e-6, track_max_dimension),
            track_max_dimension / std::max(1.0e-6, measurement_max_dimension));
        if (dimension_ratio >
            config_.duplicate_birth_maximum_dimension_ratio) {
            continue;
        }
        if (CalculateDistance(measurement.state, track.state_vec) <=
            config_.duplicate_birth_suppression_distance_m) {
            if (blocking_track_id) *blocking_track_id = track.track_id;
            return true;
        }
    }
    return false;
}

bool EkfMultiObjectTracking::ShouldSuppressSmallUnknownDuplicateBirth(
        const mc_mot::Meastruct &measurement,
        const mc_mot::Meastructs &measurements,
        const std::vector<int> &assignment_track,
        int* blocking_track_id) const {
    if (!config_.suppress_duplicate_track_birth || !config_.suppress_small_unknown_duplicate_birth ||
        measurement.classification!=mc_mot::UNKNOWN || !ValidRawBox(measurement) ||
        std::max(measurement.dimension.length,measurement.dimension.width)>=1.0) return false;
    int unique_track=-1;
    for (const auto& track : all_tracks_) {
        if (!track.is_init || !track.is_confirmed || !track.is_associated ||
            track.getRepClass()!=mc_mot::UNKNOWN || track.has_known_class_measurement ||
            track.track_id<0 || static_cast<std::size_t>(track.track_id)>=assignment_track.size()) continue;
        const int index=assignment_track[track.track_id];
        if (index<0 || static_cast<std::size_t>(index)>=measurements.meas.size()) continue;
        const auto& raw=measurements.meas[index];
        if (raw.classification!=mc_mot::UNKNOWN ||
            std::max(raw.dimension.length,raw.dimension.width)>=1.0 ||
            !SmallRawOverlapCompatible(measurement,raw)) continue;
        // Two current supported explanations are ambiguous. Do not choose one
        // by iteration order or erase a genuinely separate small observation.
        if (unique_track>=0) return false;
        unique_track=track.track_id;
    }
    if (unique_track<0) return false;
    if (blocking_track_id) *blocking_track_id=unique_track;
    return true;
}

mc_mot::AssociationDiagnosticRow EkfMultiObjectTracking::AssociationSnapshot(
        const char* event, const char* reason, const mc_mot::TrackStruct& track,
        const int measurement_index, const mc_mot::Meastruct* measurement) const {
    mc_mot::AssociationDiagnosticRow row;
    row.event = event;
    row.reason = reason;
    row.measurement_index = measurement_index;
    row.track_id = track.track_id;
    row.track_class = track.getRepClass();
    row.prior_measurement_age_sec = recent_timestamp_ - track.last_measurement_time;
    row.last_measurement_time = track.last_measurement_time;
    row.initialization_time = track.initialization_time;
    row.score = track.detection_confidence;
    row.confirmed = track.is_confirmed;
    row.associated = track.is_associated;
    row.detection_count = track.countDetectionNum();
    row.center = {{track.state_vec(S_X), track.state_vec(S_Y), track.object_z}};
    row.velocity = {{track.state_vec(S_VX), track.state_vec(S_VY)}};
    row.dimensions = {{track.dimension.length, track.dimension.width, track.dimension.height}};
    std::copy(track.class_score_arr, track.class_score_arr + CLASS_NUM, row.class_votes.begin());
    if (measurement) {
        row.measurement_class = measurement->classification;
        row.measurement_score = measurement->detection_confidence;
    }
    return row;
}

void EkfMultiObjectTracking::AppendAssociationDiagnostic(const mc_mot::AssociationDiagnosticRow& row) {
    if (association_diagnostics_.rows.size() < mc_mot::AssociationDiagnostics::kMaximumRows) {
        association_diagnostics_.rows.push_back(row);
    } else {
        ++association_diagnostics_.omitted_rows;
    }
}

// ======== Utils Functions ============

double EkfMultiObjectTracking::CalculateDistance(const mc_mot::ObjectState &state1,
                                                   const mc_mot::ObjectState &state2) const {
    return std::sqrt(std::pow(state1.x - state2.x, 2) + std::pow(state1.y - state2.y, 2));
}

double EkfMultiObjectTracking::CalculateDistance(const mc_mot::ObjectState &state1, const Eigen::Vector8d &state2) const {
    return std::sqrt(std::pow(state1.x - state2(0), 2) + std::pow(state1.y - state2(1), 2));
}

double EkfMultiObjectTracking::CalculateMahalanobisDistance(const mc_mot::ObjectState &state1,
                                                              const mc_mot::TrackStruct &track) {
    Eigen::Vector2d mean_diff;
    mean_diff << state1.x - track.state_vec(S_X), state1.y - track.state_vec(S_Y);

    Eigen::Matrix2d covariance;
    covariance << track.state_cov(S_X, S_X), track.state_cov(S_X, S_Y),
                track.state_cov(S_Y, S_X), track.state_cov(S_Y, S_Y); 
                // Use the top 2x2 part of state_cov as the covariance matrix

    // Calculate the inverse of the covariance matrix
    Eigen::Matrix2d inv_covariance = covariance.inverse();

    // Calculate Mahalanobis distance
    double mahalanobis_distance = std::sqrt(mean_diff.transpose() * inv_covariance * mean_diff);

    if (mahalanobis_distance > mean_diff.norm() * 3.0) mahalanobis_distance = mean_diff.norm() * 3.0;
    if (mahalanobis_distance < mean_diff.norm() / 5.0) mahalanobis_distance = mean_diff.norm() / 5.0;
    return mahalanobis_distance;
}

double EkfMultiObjectTracking::CalculateYawDotProduct(const double& yaw1, const double& yaw2) {
    // Calculate direction vectors
    const double x1 = std::cos(yaw1);
    const double y1 = std::sin(yaw1);
    const double x2 = std::cos(yaw2);
    const double y2 = std::sin(yaw2);

    // Calculate dot product
    const double dot_product = x1 * x2 + y1 * y2;
    return dot_product;
}

double EkfMultiObjectTracking::CalculateYawCrossProduct(const double& yaw1, const double& yaw2) {
    // Create unit direction vectors using the two yaw angles
    const double x1 = std::cos(yaw1);
    const double y1 = std::sin(yaw1);
    const double x2 = std::cos(yaw2);
    const double y2 = std::sin(yaw2);

    // Calculate the cross product of 2D vectors
    const double cross_product = x1 * y2 - y1 * x2;

    return cross_product;
}
