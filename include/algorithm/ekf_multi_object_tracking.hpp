/****************************************************************************/
// Module:      ekf_multi_object_tracking.hpp
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

#ifndef __EKF_MULTI_OBJECT_TRACKING_HPP__
#define __EKF_MULTI_OBJECT_TRACKING_HPP__
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <ros/ros.h>

#define MAX_TRACKS 500
#define MAX_HISTORY 7
#define MAX_HISTORY_FOR_OUTDATED 5
#define CLASS_NUM 5

#define INIT_COV_VAL 100.0

// CA MODEL STATE: X Y YAW VX VY YAW_RATE AX AY
//                 0  1  2  3  4  5  6  7
#define S_X 0
#define S_Y 1
#define S_YAW 2
#define S_VX 3
#define S_VY 4
#define S_YAW_RATE 5
#define S_AX 6
#define S_AY 7

// Maximum track velocity and acceleration
#define MAX_TRACK_VEL 60.0
#define MAX_TRACK_ACC 25.0

namespace Eigen {

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector8d = Eigen::Matrix<double, 8, 1>;

using Matrix6_6d = Eigen::Matrix<double, 6, 6>;
using Matrix8_8d = Eigen::Matrix<double, 8, 8>;

using Matrix3_8d = Eigen::Matrix<double, 3, 8>;
using Matrix8_3d = Eigen::Matrix<double, 8, 3>;

} // namespace Eigen

namespace mc_mot {

typedef enum { NONE = 0, ODOMETRY } LocalizationType;
typedef enum { CV = 0, CTRV, CA, CTRA } PredictionModel;
typedef enum { UNKNOWN = 0, CAR, TRUCK, PEDESTRIAN, BICYCLE} ObjectClass;

struct ObjectState {
    double time_stamp{0.0};
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
    double v_x;
    double v_y;
    double v_z;
    double a_x;
    double a_y;
    double a_z;
    double roll_rate;
    double pitch_rate;
    double yaw_rate;
};

struct ObjectDimension {
    double length{0.0};
    double width{0.0};
    double height{0.0};
};

struct CenterObservation {
    double time_stamp{0.0};
    double x{0.0};
    double y{0.0};
};

struct TrackStruct {
    int track_id{-1};
    double initialization_time{0.0};
    double update_time{0.0};
    bool has_detector_velocity{false};
    double detector_velocity_x{0.0};
    double detector_velocity_y{0.0};
    double detector_velocity_time{0.0};
    double detector_velocity_variance_mps2{1.0};
    double detection_confidence{0.0};
    unsigned int age{0};

    double direction_score{1.0}; // 1.0 High, -1 Low
    Eigen::Vector8d state_vec;   // X Y YAW VX VY YAWRATE AX AY
    Eigen::Matrix8_8d state_cov;
    ObjectClass classification;
    ObjectDimension dimension;
    double object_z{0.0};
    std::deque<CenterObservation> center_history;
    double last_center_motion_fusion_time{0.0};
    bool has_center_motion_velocity{false};
    double center_motion_velocity_x{0.0};
    double center_motion_velocity_y{0.0};
    double center_motion_velocity_noise_std_mps{0.0};

    bool is_init{false};
    bool is_confirmed{false};
    bool is_associated{false};

    bool detection_arr[MAX_HISTORY]{false};
    double class_score_arr[CLASS_NUM]{0.0};

    TrackStruct() {
        state_vec = Eigen::Vector8d::Zero();
        state_cov = Eigen::Matrix8_8d::Zero();
        state_cov.diagonal().array() = INIT_COV_VAL;
    }

    // Update the number of detections
    void updateDetectionCount(bool associated) {
        // Shift the array one position to the right and add the new value
        for (int i = MAX_HISTORY - 1; i > 0; --i) {
            detection_arr[i] = detection_arr[i - 1];
        }
        detection_arr[0] = associated;
    }

    // Return the number of detections in the recent MAX_HISTORY
    int countDetectionNum() {
        int detection_count = 0;
        for (int i = MAX_HISTORY - 1; i >= 0; --i) {
            if (detection_arr[i] == true) {
                detection_count++;
            }
        }
        return detection_count;
    }

    // Check if the track is outdated
    bool isOutdated() {
        // If the number of detections in the recent MAX_HISTORY is less than MAX_HISTORY_FOR_OUTDATED, it is outdated
        return countDetectionNum() < 
                MAX_HISTORY_FOR_OUTDATED - std::max(0, MAX_HISTORY - static_cast<int>(age) );
    }

    // Update the representative class probability of the track
    void updateClassScore(int cur_class) {
        double alpha = 0.2;
        for (int i = 0; i < CLASS_NUM; i++) {
            if (i == cur_class) {
                class_score_arr[i] = (alpha) * (1.0) + (1.0 - alpha) * class_score_arr[i];
            }
            else {
                class_score_arr[i] = (1.0 - alpha) * class_score_arr[i];
            }
        }
    }

    // Return the representative class of the track
    int getRepClass() {
        int rep_class = 0;
        double rep_prob = 0.0;
        for (int i = 0; i < CLASS_NUM; i++) {
            if (class_score_arr[i] > rep_prob) {
                rep_class = i;
                rep_prob = class_score_arr[i];
            }
        }
        return rep_class;
    }

    // Return the representative class probability of the track
    double getRepClassProb() {
        int rep_class = 0;
        double rep_prob = 0.0;
        for (int i = 0; i < CLASS_NUM; i++) {
            if (class_score_arr[i] > rep_prob) {
                rep_class = i;
                rep_prob = class_score_arr[i];
            }
        }
        return rep_prob;
    }

    // Reset the track
    void reset() {
        initialization_time = 0.0;
        update_time = 0.0;
        has_detector_velocity = false;
        detector_velocity_x = 0.0;
        detector_velocity_y = 0.0;
        detector_velocity_time = 0.0;
        detector_velocity_variance_mps2 = 1.0;
        detection_confidence = 0.0;
        age = 0;

        classification = ObjectClass::UNKNOWN;
        direction_score = 0.0;
        object_z = 0.0;

        state_vec = Eigen::Vector8d::Zero();
        state_cov = Eigen::Matrix8_8d::Zero();
        state_cov.diagonal().array() = INIT_COV_VAL;

        is_init = false;
        is_confirmed = false;
        is_associated = false;
        std::fill(std::begin(detection_arr), std::end(detection_arr), false);
        center_history.clear();
        last_center_motion_fusion_time = 0.0;
        has_center_motion_velocity = false;
        center_motion_velocity_x = 0.0;
        center_motion_velocity_y = 0.0;
        center_motion_velocity_noise_std_mps = 0.0;
    }
};

struct TrackStructs {
    double time_stamp{0.0};
    std::vector<TrackStruct> track;
};

struct Meastruct {
    unsigned int id{0};
    float detection_confidence{1.0};
    bool has_velocity{false};
    double velocity_variance_mps2{1.0};

    ObjectClass classification;
    ObjectDimension dimension;
    ObjectState state;
};

struct Meastructs {
    double time_stamp{0.0};
    std::vector<Meastruct> meas;
};
} // namespace mc_mot

struct MultiClassObjectTrackingConfig {
    mc_mot::LocalizationType input_localization = mc_mot::LocalizationType::NONE;
    bool global_coord_track{false};
    bool output_local_coord{false};
    bool output_period_lidar{false};
    bool output_confirmed_track{false};

    bool use_predefined_ref_point{false};
    double reference_lat{0.0};
    double reference_lon{0.0};
    double reference_height{0.0};

    bool cal_detection_individual_time{false};
    double lidar_rotation_period{0.1};
    bool lidar_sync_scan_start{true};

    double max_association_dist_m{3.0};

    mc_mot::PredictionModel prediction_model = mc_mot::PredictionModel::CV;

    double system_noise_std_xy_m{0.1};
    double system_noise_std_yaw_deg{0.1};
    double system_noise_std_vx_vy_ms{0.1};
    double system_noise_std_yaw_rate_degs{0.1};
    double system_noise_std_ax_ay_ms2{0.1};

    double meas_noise_std_xy_m{0.1};
    double meas_noise_std_yaw_deg{0.1};
    // 0: ignore detector velocity, 1: initialize new tracks only,
    // 2: initialize and update velocity whenever a valid measurement exists,
    // 3: initialize and update only during the configured early-track window,
    // 4: keep the EKF position-only and expose a gated early longitudinal
    //    detector velocity at publication time.
    // 5: keep the EKF position-only and blend longitudinal velocity at
    //    publication time according to detector/tracker uncertainty.
    int detection_velocity_fusion_mode{0};
    double detection_velocity_noise_std_mps{1.0};
    double detection_velocity_update_max_track_age_sec{1.0};
    double detection_velocity_maximum_output_age_sec{0.25};
    double detection_velocity_max_longitudinal_innovation_mps{5.0};
    double detection_velocity_longitudinal_blend{1.0};

    // Optional causal velocity observation derived from associated raw
    // detection centers in the global frame.  Kept separate from detector
    // velocity so a 7D detector can use it without changing its message.
    bool center_motion_velocity_fusion{false};
    bool center_motion_fuse_ekf_state{true};
    bool center_motion_override_output_velocity{false};
    double center_motion_maximum_output_age_sec{0.40};
    double center_motion_min_baseline_sec{0.30};
    double center_motion_medium_baseline_sec{0.65};
    double center_motion_long_baseline_sec{0.90};
    double center_motion_max_gap_sec{0.25};
    double center_motion_min_update_interval_sec{0.20};
    double center_motion_max_fit_residual_m{0.10};
    double center_motion_early_noise_std_mps{1.50};
    double center_motion_mature_noise_std_mps{0.60};

    double dimension_filter_alpha{0.1};

    bool use_kinematic_model{false};
    bool use_yaw_rate_filtering{false};

    bool canonicalize_vehicle_orientation_to_motion{true};
    double orientation_canonicalization_min_speed_mps{1.0};
    double orientation_flip_enter_error_deg{100.0};
    double orientation_flip_exit_error_deg{80.0};

    bool compensate_output_to_current_time{false};
    double maximum_output_compensation_sec{0.25};

    // 0.0 keeps detector-synchronized publication. A positive value enables
    // the node-internal scheduler, which publishes a non-mutating prediction
    // of the EKF state at fixed ROS-time intervals.
    double fixed_output_rate_hz{0.0};
    double maximum_fixed_output_prediction_age_sec{0.25};

    double max_steer_deg{30.0};

    bool visualize_mesh{false};
};

class EkfMultiObjectTracking {
public:
    explicit EkfMultiObjectTracking(const MultiClassObjectTrackingConfig &config) : config_(config) {
        UpdateMatrix();
    }

    EkfMultiObjectTracking() : EkfMultiObjectTracking(MultiClassObjectTrackingConfig{}) {}

public:
    // Public functions
    void RunPrediction(double dt_sec);
    void RunUpdate(const mc_mot::Meastructs &measurements);
    mc_mot::TrackStructs GetTrackResults() const;
    mc_mot::TrackStructs GetPredictedTrackResults(double target_timestamp);

    void UpdateConfig(const MultiClassObjectTrackingConfig config);

private:
    // Private functions
    void PredictTrack(mc_mot::TrackStruct &track, double dt);
    void UpdateTrack(mc_mot::TrackStruct &track, const mc_mot::Meastruct &measurement);
    void InitTrack(mc_mot::TrackStruct &track, const mc_mot::Meastruct &measurement);
    void UpdateCenterMotionVelocity(
        mc_mot::TrackStruct &track,
        const mc_mot::Meastruct &measurement);
    void FuseVelocity(
        mc_mot::TrackStruct &track,
        double velocity_x,
        double velocity_y,
        double noise_std_mps);

    void MatchPairs(const Eigen::MatrixXd &cost_matrix, std::vector<int> &row_indices, std::vector<int> &col_indices);
    void UpdateTrackId();
    void UpdateMatrix();

private:
    // Utils
    double CalculateDistance(const mc_mot::ObjectState &state1, const mc_mot::ObjectState &state2);
    double CalculateDistance(const mc_mot::ObjectState &state1, const Eigen::Vector8d &state2);
    double CalculateMahalanobisDistance(const mc_mot::ObjectState &state1, const mc_mot::TrackStruct &track);
    double CalculateYawDotProduct(const double &yaw1, const double &yaw2);
    double CalculateYawCrossProduct(const double &yaw1, const double &yaw2);

private:
    // Private Variables

    std::array<mc_mot::TrackStruct, MAX_TRACKS> all_tracks_;
    int cur_track_id_{0}; // Current track id

    double recent_timestamp_{0.0}; // Recent timestamp

    Eigen::Matrix8_8d Q_; // System noise covariance matrix
    Eigen::Matrix3d R_;   // Measurement noise covariance matrix
    Eigen::Matrix3_8d H_; // Measurement matrix

    // config
    MultiClassObjectTrackingConfig config_;
};

#endif // __EKF_MULTI_OBJECT_TRACKING_HPP__
