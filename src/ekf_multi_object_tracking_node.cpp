/****************************************************************************/
// Module:      ekf_multi_object_tracking_node.cpp
// Description: ekf_multi_object_tracking_node
//
// Authors: Jaeyoung Jo (wodud3743@gmail.com)
// Version: 0.1
//
// Revision History
//      July 19, 2024: Jaeyoung Jo - Created.
//      Jan  08, 2025: Jaeyoung Jo - Public data type.
//      XXXX XX, 2023: XXXXXXX XX -
/****************************************************************************/

#include "ekf_multi_object_tracking_node.hpp"

namespace {

uint8_t ToAutowareClass(const int track_class) {
    switch (track_class) {
        case mc_mot::ObjectClass::CAR:
            return autoware_perception_msgs::ObjectClassification::CAR;
        case mc_mot::ObjectClass::TRUCK:
            return autoware_perception_msgs::ObjectClassification::TRUCK;
        case mc_mot::ObjectClass::PEDESTRIAN:
            return autoware_perception_msgs::ObjectClassification::PEDESTRIAN;
        case mc_mot::ObjectClass::BICYCLE:
            return autoware_perception_msgs::ObjectClassification::BICYCLE;
        default:
            return autoware_perception_msgs::ObjectClassification::UNKNOWN;
    }
}

unique_identifier_msgs::UUID StableTrackUuid(const int track_id) {
    unique_identifier_msgs::UUID uuid;
    std::fill(uuid.uuid.begin(), uuid.uuid.end(), 0U);
    // Namespace the deterministic ID as "AJOU" and retain the tracker ID in
    // the final four bytes.  This stays stable for the life of a track.
    uuid.uuid[0] = 'A';
    uuid.uuid[1] = 'J';
    uuid.uuid[2] = 'O';
    uuid.uuid[3] = 'U';
    const uint32_t value = static_cast<uint32_t>(track_id);
    uuid.uuid[12] = static_cast<uint8_t>((value >> 24) & 0xffU);
    uuid.uuid[13] = static_cast<uint8_t>((value >> 16) & 0xffU);
    uuid.uuid[14] = static_cast<uint8_t>((value >> 8) & 0xffU);
    uuid.uuid[15] = static_cast<uint8_t>(value & 0xffU);
    return uuid;
}

}  // namespace

EkfMultiObjectTrackingNode::EkfMultiObjectTrackingNode(){

    // Data validation
    b_is_new_lidar_objects_ = false;
    b_is_new_track_objects_ = false;

    b_is_new_motion_input_ = false;

    b_is_track_init_ = false;

    last_predicted_time_ = 0.0;

    Init();
}

EkfMultiObjectTrackingNode::~EkfMultiObjectTrackingNode() {}

void EkfMultiObjectTrackingNode::Init() {
    ROS_INFO_STREAM("Init");
    // Node initialization
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    ProcessYAML();

    if (!nh.getParam("/topic_name/lidar_objects", cfg_lidar_objects_topic_)) {
        ROS_ERROR_STREAM("Failed to get param: /topic_name/lidar_objects");
        cfg_lidar_objects_topic_ = "/app/perc/lidar_objects";
    }
    if (!nh.getParam("/topic_name/odometry", cfg_odometry_topic_)) {
        ROS_ERROR_STREAM("Failed to get param: /topic_name/odometry");
        cfg_odometry_topic_ = "/odometry";
    }
    if (!nh.getParam("/topic_name/output_track_jsk", cfg_output_track_jsk_topic_)) {
        ROS_ERROR_STREAM("Failed to get param: /topic_name/output_track_jsk");
        cfg_output_track_jsk_topic_ = "/output_track_jsk";
    }
    if (!nh.getParam("/topic_name/output_track_marker", cfg_output_track_marker_topic_)) {
        ROS_ERROR_STREAM("Failed to get param: /topic_name/output_track_marker");
        cfg_output_track_marker_topic_ = "/output_track_marker";
    }

    private_nh.param<std::string>("world_frame_id", world_frame_id_, "map");
    private_nh.param<double>("transform_timeout_sec", transform_timeout_sec_, 0.1);

    // Standard object pipeline I/O.  Global names are selected by launch remaps.
    s_lidar_objects_ = private_nh.subscribe(
        "input/objects", 10, &EkfMultiObjectTrackingNode::CallbackDetectedObjects, this);

    // Localization source subscriber
    if (config_.input_localization == mc_mot::LocalizationType::ODOMETRY)
        s_odometry_ = nh.subscribe(cfg_odometry_topic_, 10, &EkfMultiObjectTrackingNode::CallbackOdometry, this);

    // Publisher init
    p_track_objects_ = private_nh.advertise<autoware_perception_msgs::TrackedObjects>(
        "output/objects", 10);
    p_center_motion_active_objects_ =
        private_nh.advertise<autoware_perception_msgs::TrackedObjects>(
            "diagnostics/center_motion_active_objects", 10);
    p_center_motion_inactive_objects_ =
        private_nh.advertise<autoware_perception_msgs::TrackedObjects>(
            "diagnostics/center_motion_inactive_objects", 10);
    // JSK and markers remain visualization-only compatibility outputs.
    p_all_track_ = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>(cfg_output_track_jsk_topic_, 10);
    p_all_track_info_ = nh.advertise<visualization_msgs::MarkerArray>(cfg_output_track_marker_topic_, 10);
    p_ego_stl_ = nh.advertise<visualization_msgs::Marker>("app/perc/vis_ego_stl", 10);

    // algorithm init
    mcot_algorithm_ = EkfMultiObjectTracking(config_);

    b_is_init_ = true;
    mcot_algorithm_.UpdateConfig(config_);
}

void EkfMultiObjectTrackingNode::Run() {
    // ----- Input -----
    ros_interface::DetectObjects3D lidar_objects;
    bool has_new_lidar_objects = false;
    {
        std::lock_guard<std::mutex> lock(mutex_lidar_objects_);
        has_new_lidar_objects = b_is_new_lidar_objects_;
        if (has_new_lidar_objects) lidar_objects = i_lidar_objects_;
    }

    // Detector-synchronized mode has no work until the next measurement.
    // Fixed-rate mode must continue so its internal output scheduler can
    // publish predicted copies between measurements.
    if (!has_new_lidar_objects && config_.fixed_output_rate_hz <= 0.0) return;

    
    std::deque<mc_mot::ObjectState> deque_lidar_state;
    if(config_.input_localization != mc_mot::LocalizationType::NONE){
        std::lock_guard<std::mutex> lock(mutex_motion_);
        if (deque_lidar_state_.size() < 1) return;

        mc_mot::ObjectState lidar_state = deque_lidar_state_.back();

        if(has_new_lidar_objects &&
           lidar_objects.header.stamp - lidar_state.time_stamp > 1.0){
            ROS_WARN_STREAM("LiDAR STATE is Old!");
            deque_lidar_state_.clear();
        }
        deque_lidar_state = deque_lidar_state_;

        // Motion Update with Input Localization
        if (b_is_new_motion_input_ == true && b_is_track_init_ == true) {
            double dt = lidar_state.time_stamp - last_predicted_time_;
            mcot_algorithm_.RunPrediction(dt);
            last_predicted_time_ = lidar_state.time_stamp;

            b_is_new_motion_input_ = false;
            b_is_new_track_objects_ = true;
        }
    }

    // Measurement Update
    if (has_new_lidar_objects) {
        if (config_.input_localization == mc_mot::LocalizationType::NONE && b_is_track_init_ == true) {
            double dt = lidar_objects.header.stamp - last_predicted_time_;
            if (dt <= 0.0) {
                ROS_WARN_THROTTLE(1.0, "[EKF tracker] Non-increasing detection stamp; dropping frame");
                b_is_new_lidar_objects_ = false;
                return;
            }
            mcot_algorithm_.RunPrediction(dt);
            last_predicted_time_ = lidar_objects.header.stamp;
        }

        mc_mot::Meastructs meas_structs;

        // when there is no localization source or when using local coordinate tracking,
        // just directly using original detection data as track update information.
        if (config_.input_localization == mc_mot::LocalizationType::NONE ||
            config_.global_coord_track == false) {
            DetectObjects2LocalMeasurements(lidar_objects, meas_structs);
        }
        else {
            // when using localization source and using global coordinate tracking.
            // convert local coordinate detection into global coordinate.
            if (DetectObjects2GlobMeasurements(lidar_objects, meas_structs) == false) {
                ROS_WARN_STREAM("CANNOT CONVERT EGO TO GLOBAL. NO LIDAR STATES");
                return;
            }
        }

        // Run measurment update in algorithm
        mcot_algorithm_.RunUpdate(meas_structs);

        b_is_new_lidar_objects_ = false;
        b_is_new_track_objects_ = true;

        if(b_is_track_init_ == false){
            b_is_track_init_ = true;
            last_predicted_time_ = lidar_objects.header.stamp;
            ROS_INFO("Init Tracker");
        }
    }

    
    bool build_output = b_is_new_track_objects_;
    double fixed_output_target_sec = 0.0;
    if (config_.fixed_output_rate_hz > 0.0) {
        // A measurement update does not publish immediately in fixed-rate
        // mode. Only the internal scheduler below owns output cadence.
        b_is_new_track_objects_ = false;
        build_output = false;

        if (b_is_track_init_) {
            const double now_sec = ros::Time::now().toSec();
            const double period_sec = 1.0 / config_.fixed_output_rate_hz;
            if (next_fixed_output_time_sec_ <= 0.0 ||
                now_sec + period_sec < next_fixed_output_time_sec_) {
                next_fixed_output_time_sec_ = now_sec;
            }
            if (now_sec + 1.0e-9 >= next_fixed_output_time_sec_) {
                fixed_output_target_sec = next_fixed_output_time_sec_;
                // Skip missed slots instead of emitting a burst of duplicate
                // messages after a slow callback or a rosbag clock jump.
                const double periods_behind =
                    std::floor((now_sec - next_fixed_output_time_sec_) / period_sec);
                next_fixed_output_time_sec_ +=
                    (std::max(0.0, periods_behind) + 1.0) * period_sec;
                build_output = true;
            }
        }
    }

    if (build_output) {
        // ----- Output -----
        mc_mot::TrackStructs mot_track_structs = mcot_algorithm_.GetTrackResults();
        if (config_.fixed_output_rate_hz > 0.0) {
            const double output_age_sec =
                fixed_output_target_sec - mot_track_structs.time_stamp;
            if (output_age_sec < -1.0e-6 ||
                output_age_sec > config_.maximum_fixed_output_prediction_age_sec) {
                ROS_DEBUG_THROTTLE(
                    1.0,
                    "[EKF tracker] Skip fixed output with state age %.3f sec",
                    output_age_sec);
                return;
            }
            if (output_age_sec > 0.0) {
                mot_track_structs = mcot_algorithm_.GetPredictedTrackResults(
                    fixed_output_target_sec);
            }
        } else {
            const double current_ros_time = ros::Time::now().toSec();
            const double output_age_sec =
                current_ros_time - mot_track_structs.time_stamp;
            if (config_.compensate_output_to_current_time &&
                output_age_sec > 0.0 &&
                output_age_sec <= config_.maximum_output_compensation_sec) {
            // Extrapolate an output copy only. The measurement-time EKF state
            // remains untouched, preserving the true dt for the next update.
                mot_track_structs = mcot_algorithm_.GetPredictedTrackResults(
                    current_ros_time);
            }
        }

        std::string o_frame_id;
        if (config_.input_localization != mc_mot::LocalizationType::NONE) {
            if(deque_lidar_state.size() <= 1){
                ROS_WARN_STREAM("CANNOT CONVERT TRACK TO LOCAL. NO LIDAR STATES");
                b_is_new_track_objects_ = false;
                return;
            }
            mc_mot::ObjectState synced_lidar_state =
                    GetSyncedLidarState(mot_track_structs.time_stamp, deque_lidar_state);
            ConvertTrackGlobalToLocal(mot_track_structs, synced_lidar_state);

            o_frame_id = world_frame_id_;
            VisualizeTrackObjects(mot_track_structs, o_frame_id);
        }
        else { // no localization. output velodyne coordinate
            o_frame_id = world_frame_id_;
            VisualizeTrackObjects(mot_track_structs, o_frame_id);
        }
        BuildTrackedObjects(mot_track_structs, o_frame_id);
        b_is_new_track_objects_ = true;
    }


}

void EkfMultiObjectTrackingNode::Publish() {
    if (b_is_new_track_objects_ == true) {
        ROS_DEBUG_STREAM("[EKF tracker] Publish " << o_tracked_objects_.objects.size()
                         << " standardized tracks");
        
        p_all_track_.publish(o_jsk_tracked_objects_);
        p_track_objects_.publish(o_tracked_objects_);
        if (publish_center_motion_diagnostics_) {
            p_center_motion_active_objects_.publish(o_center_motion_active_objects_);
            p_center_motion_inactive_objects_.publish(o_center_motion_inactive_objects_);
        }
        p_all_track_info_.publish(o_vis_track_info_);
        p_ego_stl_.publish(o_vis_ego_stl_);

        b_is_new_track_objects_ = false;
    }
}

void EkfMultiObjectTrackingNode::Terminate() { ROS_INFO_STREAM("Terminate Node"); }

void EkfMultiObjectTrackingNode::ProcessYAML() {
    ros::NodeHandle nh;

    // Read parameters from YAML
    int i_input_localization = 0;
    nh.getParam("configure/input_localization", i_input_localization);
    config_.input_localization =  mc_mot::LocalizationType(i_input_localization);

    ROS_DEBUG_STREAM("[EKF tracker] input_localization=" << i_input_localization);
    
    nh.getParam("configure/global_coord_track", config_.global_coord_track);
    nh.getParam("configure/output_local_coord", config_.output_local_coord);
    nh.getParam("configure/output_period_lidar", config_.output_period_lidar);
    nh.getParam("configure/output_confirmed_track", config_.output_confirmed_track);
    nh.getParam("configure/use_predefined_ref_point", config_.use_predefined_ref_point);
    nh.getParam("configure/reference_lat", config_.reference_lat);
    nh.getParam("configure/reference_lon", config_.reference_lon);
    nh.getParam("configure/reference_height", config_.reference_height);
    nh.getParam("configure/cal_detection_individual_time", config_.cal_detection_individual_time);
    nh.getParam("configure/lidar_rotation_period", config_.lidar_rotation_period);
    nh.getParam("configure/lidar_sync_scan_start", config_.lidar_sync_scan_start);
    nh.getParam("configure/max_association_dist_m", config_.max_association_dist_m);

    int i_prediction_model = 0;
    nh.getParam("configure/prediction_model", i_prediction_model);
    config_.prediction_model = mc_mot::PredictionModel(i_prediction_model);

    nh.getParam("configure/system_noise_std_xy_m", config_.system_noise_std_xy_m);
    nh.getParam("configure/system_noise_std_yaw_deg", config_.system_noise_std_yaw_deg);
    nh.getParam("configure/system_noise_std_vx_vy_ms", config_.system_noise_std_vx_vy_ms);
    nh.getParam("configure/system_noise_std_yaw_rate_degs", config_.system_noise_std_yaw_rate_degs);
    nh.getParam("configure/system_noise_std_ax_ay_ms2", config_.system_noise_std_ax_ay_ms2);
    nh.getParam("configure/meas_noise_std_xy_m", config_.meas_noise_std_xy_m);
    nh.getParam("configure/meas_noise_std_yaw_deg", config_.meas_noise_std_yaw_deg);
    nh.param<int>(
        "configure/detection_velocity_fusion_mode",
        config_.detection_velocity_fusion_mode,
        0);
    nh.param<double>(
        "configure/detection_velocity_noise_std_mps",
        config_.detection_velocity_noise_std_mps,
        1.0);
    nh.param<double>(
        "configure/detection_velocity_update_max_track_age_sec",
        config_.detection_velocity_update_max_track_age_sec,
        1.0);
    nh.param<bool>(
        "configure/center_motion_velocity_fusion",
        config_.center_motion_velocity_fusion,
        false);
    nh.param<bool>(
        "configure/publish_center_motion_diagnostics",
        publish_center_motion_diagnostics_,
        false);
    nh.param<bool>(
        "configure/center_motion_fuse_ekf_state",
        config_.center_motion_fuse_ekf_state,
        true);
    nh.param<bool>(
        "configure/center_motion_override_output_velocity",
        config_.center_motion_override_output_velocity,
        false);
    nh.param<double>(
        "configure/center_motion_maximum_output_age_sec",
        config_.center_motion_maximum_output_age_sec,
        0.40);
    nh.param<double>(
        "configure/center_motion_min_baseline_sec",
        config_.center_motion_min_baseline_sec,
        0.30);
    nh.param<double>(
        "configure/center_motion_medium_baseline_sec",
        config_.center_motion_medium_baseline_sec,
        0.65);
    nh.param<double>(
        "configure/center_motion_long_baseline_sec",
        config_.center_motion_long_baseline_sec,
        0.90);
    nh.param<double>(
        "configure/center_motion_max_gap_sec",
        config_.center_motion_max_gap_sec,
        0.25);
    nh.param<double>(
        "configure/center_motion_min_update_interval_sec",
        config_.center_motion_min_update_interval_sec,
        0.20);
    nh.param<double>(
        "configure/center_motion_max_fit_residual_m",
        config_.center_motion_max_fit_residual_m,
        0.10);
    nh.param<double>(
        "configure/center_motion_early_noise_std_mps",
        config_.center_motion_early_noise_std_mps,
        1.50);
    nh.param<double>(
        "configure/center_motion_mature_noise_std_mps",
        config_.center_motion_mature_noise_std_mps,
        0.60);
    nh.getParam("configure/dimension_filter_alpha", config_.dimension_filter_alpha);
    nh.getParam("configure/use_kinematic_model", config_.use_kinematic_model);
    nh.getParam("configure/use_yaw_rate_filtering", config_.use_yaw_rate_filtering);
    nh.getParam(
        "configure/canonicalize_vehicle_orientation_to_motion",
        config_.canonicalize_vehicle_orientation_to_motion);
    nh.getParam(
        "configure/orientation_canonicalization_min_speed_mps",
        config_.orientation_canonicalization_min_speed_mps);
    nh.getParam(
        "configure/orientation_flip_enter_error_deg",
        config_.orientation_flip_enter_error_deg);
    nh.getParam(
        "configure/orientation_flip_exit_error_deg",
        config_.orientation_flip_exit_error_deg);
    nh.getParam(
        "configure/compensate_output_to_current_time",
        config_.compensate_output_to_current_time);
    nh.getParam(
        "configure/maximum_output_compensation_sec",
        config_.maximum_output_compensation_sec);
    nh.getParam(
        "configure/fixed_output_rate_hz",
        config_.fixed_output_rate_hz);
    nh.getParam(
        "configure/maximum_fixed_output_prediction_age_sec",
        config_.maximum_fixed_output_prediction_age_sec);
    nh.getParam("configure/max_steer_deg", config_.max_steer_deg);
    nh.getParam("configure/visualize_mesh", config_.visualize_mesh);

    // Vehicle origin parameters
    int i_vehicle_origin;
    nh.getParam("vehicle_origin/vehicle_origin", i_vehicle_origin);

    std::string str_origin = (i_vehicle_origin == 0) ? "rear" : "cg";
    nh.getParam(str_origin + "_to_main_lidar/transform_xyz_m", cfg_vec_d_ego_to_lidar_xyz_m_);
    nh.getParam(str_origin + "_to_main_lidar/rotation_rpy_deg", cfg_vec_d_ego_to_lidar_rpy_deg_);

    std::cout << "[Config] ego_to_lidar_xyz_m: ";
    for (const auto& val : cfg_vec_d_ego_to_lidar_xyz_m_) {
        std::cout << val << " ";
    }

    std::cout << "[Config] ego_to_lidar_rpy_deg: ";
    for (const auto& val : cfg_vec_d_ego_to_lidar_rpy_deg_) {
        std::cout << val << " ";
    }

    if (cfg_vec_d_ego_to_lidar_xyz_m_.size() != 3 || cfg_vec_d_ego_to_lidar_rpy_deg_.size() != 3) {
        ROS_ERROR("[Config] Not proper calibration!");
        ros::shutdown();
    }

    std::cout << std::endl;
    ROS_WARN("[DetectObject tracking] YAML file is updated!");

    if (b_is_init_) mcot_algorithm_.UpdateConfig(config_);
}

// --------------------------------------------------

bool EkfMultiObjectTrackingNode::DetectObjects2GlobMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                                                  mc_mot::Meastructs& o_glob_lidar_measurements) {

    // Copy deque lidar state from member variable with mutex
    std::deque<mc_mot::ObjectState> deque_lidar_state;
    {
        std::lock_guard<std::mutex> lock(mutex_motion_);
        deque_lidar_state = deque_lidar_state_;
    }

    if (deque_lidar_state.size() < 3) {
        return false;
    }

    // if lidar state is order than object time, Run state prediction
    if (deque_lidar_state.back().time_stamp < lidar_objects.header.stamp) {
        int prediction_step_count = 0;
        // Run state prediction
        while (deque_lidar_state.back().time_stamp < lidar_objects.header.stamp) {
            int i_state_size = deque_lidar_state.size();
            mc_mot::ObjectState new_state =
                    PredictNextState(deque_lidar_state[i_state_size - 1], deque_lidar_state[i_state_size]);
            deque_lidar_state.push_back(new_state);
            prediction_step_count++;
        }

        ROS_WARN_STREAM("[Local To Global] Predict state " << prediction_step_count << " Count.");
    }

    int i_det_size = lidar_objects.object.size();
    o_glob_lidar_measurements.meas.resize(i_det_size);
    o_glob_lidar_measurements.time_stamp = lidar_objects.header.stamp;

    for (int i = 0; i < i_det_size; i++) {
        // Type conversion from ros interface to mc_mot interface
        ConvertDetectObjectToMeastruct(lidar_objects.object[i], o_glob_lidar_measurements.meas[i]);

        // 1. Measured angle based time compensation.
        if (config_.cal_detection_individual_time == true) {
            AngleBasedTimeCompensation(o_glob_lidar_measurements.meas[i]);
        }

        // 2. Transform LiDAR local meas to Global coordinate
        TransformMeasLiDAR2Global(o_glob_lidar_measurements.meas[i], deque_lidar_state);
    }

    return true;
}

void EkfMultiObjectTrackingNode::DetectObjects2LocalMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                                                   mc_mot::Meastructs& o_local_lidar_measurements) {

    int i_det_size = lidar_objects.object.size();
    o_local_lidar_measurements.meas.resize(i_det_size);
    o_local_lidar_measurements.time_stamp = lidar_objects.header.stamp;

    for (int i = 0; i < i_det_size; i++) {
        ConvertDetectObjectToMeastruct(lidar_objects.object[i], o_local_lidar_measurements.meas[i]);
    }
}

void EkfMultiObjectTrackingNode::ConvertDetectObjectToMeastruct(const ros_interface::DetectObject3D& detect_object, 
                                                                mc_mot::Meastruct& meas) {
    meas.id = detect_object.id;
    meas.detection_confidence = detect_object.confidence_score;
    meas.classification = mc_mot::ObjectClass(detect_object.classification);
    meas.has_velocity = detect_object.has_velocity;

    meas.state.time_stamp = detect_object.state.header.stamp;
    meas.state.x = detect_object.state.x;
    meas.state.y = detect_object.state.y;
    meas.state.z = detect_object.state.z;
    meas.state.yaw = detect_object.state.yaw;
    meas.state.v_x = detect_object.state.v_x;
    meas.state.v_y = detect_object.state.v_y;

    meas.dimension.height = detect_object.dimension.height;
    meas.dimension.width = detect_object.dimension.width;
    meas.dimension.length = detect_object.dimension.length;
}

void EkfMultiObjectTrackingNode::BuildTrackedObjects(
        const mc_mot::TrackStructs& track_structs, const std::string& frame_id) {
    o_tracked_objects_.header.frame_id = frame_id;
    o_tracked_objects_.header.stamp = ros::Time(track_structs.time_stamp);
    o_tracked_objects_.objects.clear();
    o_center_motion_active_objects_.header = o_tracked_objects_.header;
    o_center_motion_active_objects_.objects.clear();
    o_center_motion_inactive_objects_.header = o_tracked_objects_.header;
    o_center_motion_inactive_objects_.objects.clear();

    std::unordered_set<int> active_track_ids;
    int canonicalized_vehicle_count = 0;
    for (auto track : track_structs.track) {
        active_track_ids.insert(track.track_id);
        if (!IsVisualizeTrack(track)) continue;

        autoware_perception_msgs::TrackedObject output;
        output.object_id = StableTrackUuid(track.track_id);
        output.existence_probability = static_cast<float>(
            std::max(0.0, std::min(1.0, track.detection_confidence)));

        autoware_perception_msgs::ObjectClassification classification;
        classification.label = ToAutowareClass(track.getRepClass());
        classification.probability = static_cast<float>(
            std::max(0.0, std::min(1.0, track.getRepClassProb())));
        output.classification.push_back(classification);

        const bool is_vehicle =
            classification.label == autoware_perception_msgs::ObjectClassification::CAR ||
            classification.label == autoware_perception_msgs::ObjectClassification::TRUCK ||
            classification.label == autoware_perception_msgs::ObjectClassification::BUS ||
            classification.label == autoware_perception_msgs::ObjectClassification::TRAILER;
        double output_velocity_x = track.state_vec(S_VX);
        double output_velocity_y = track.state_vec(S_VY);
        const double center_motion_age =
            track_structs.time_stamp - track.last_center_motion_fusion_time;
        const bool center_motion_available =
            config_.center_motion_velocity_fusion &&
            track.has_center_motion_velocity &&
            center_motion_age >= -1.0e-6 &&
            center_motion_age <= config_.center_motion_maximum_output_age_sec;
        const bool use_center_motion_output =
            config_.center_motion_override_output_velocity &&
            center_motion_available;
        if (use_center_motion_output) {
            output_velocity_x = track.center_motion_velocity_x;
            output_velocity_y = track.center_motion_velocity_y;
        }
        const auto canonical = vehicle_orientation_canonicalizer_.Apply(
            track.track_id,
            track.state_vec(S_YAW),
            output_velocity_x,
            output_velocity_y,
            is_vehicle,
            config_.canonicalize_vehicle_orientation_to_motion,
            config_.orientation_canonicalization_min_speed_mps,
            config_.orientation_flip_enter_error_deg * M_PI / 180.0,
            config_.orientation_flip_exit_error_deg * M_PI / 180.0);
        if (canonical.flipped) ++canonicalized_vehicle_count;
        const double output_yaw = canonical.yaw;

        auto& kinematics = output.kinematics;
        auto& pose = kinematics.pose_with_covariance.pose;
        pose.position.x = track.state_vec(S_X);
        pose.position.y = track.state_vec(S_Y);
        pose.position.z = track.object_z;
        pose.orientation = tf::createQuaternionMsgFromYaw(output_yaw);

        auto& pose_cov = kinematics.pose_with_covariance.covariance;
        std::fill(pose_cov.begin(), pose_cov.end(), 0.0);
        pose_cov[0] = track.state_cov(S_X, S_X);
        pose_cov[1] = track.state_cov(S_X, S_Y);
        pose_cov[5] = track.state_cov(S_X, S_YAW);
        pose_cov[6] = track.state_cov(S_Y, S_X);
        pose_cov[7] = track.state_cov(S_Y, S_Y);
        pose_cov[11] = track.state_cov(S_Y, S_YAW);
        pose_cov[30] = track.state_cov(S_YAW, S_X);
        pose_cov[31] = track.state_cov(S_YAW, S_Y);
        pose_cov[35] = track.state_cov(S_YAW, S_YAW);

        // Autoware object twist is expressed along/across the object heading,
        // while the EKF state velocity is maintained in the map frame.
        const double cosine = std::cos(output_yaw);
        const double sine = std::sin(output_yaw);
        auto& twist = kinematics.twist_with_covariance.twist;
        twist.linear.x = cosine * output_velocity_x + sine * output_velocity_y;
        twist.linear.y = -sine * output_velocity_x + cosine * output_velocity_y;
        twist.angular.z = track.state_vec(S_YAW_RATE);

        Eigen::Matrix2d velocity_covariance;
        velocity_covariance <<
            track.state_cov(S_VX, S_VX), track.state_cov(S_VX, S_VY),
            track.state_cov(S_VY, S_VX), track.state_cov(S_VY, S_VY);
        if (use_center_motion_output) {
            const double variance =
                track.center_motion_velocity_noise_std_mps *
                track.center_motion_velocity_noise_std_mps;
            velocity_covariance = variance * Eigen::Matrix2d::Identity();
        }
        Eigen::Matrix2d world_to_body;
        world_to_body << cosine, sine, -sine, cosine;
        const Eigen::Matrix2d body_velocity_covariance =
            world_to_body * velocity_covariance * world_to_body.transpose();
        auto& twist_cov = kinematics.twist_with_covariance.covariance;
        std::fill(twist_cov.begin(), twist_cov.end(), 0.0);
        twist_cov[0] = body_velocity_covariance(0, 0);
        twist_cov[1] = body_velocity_covariance(0, 1);
        twist_cov[6] = body_velocity_covariance(1, 0);
        twist_cov[7] = body_velocity_covariance(1, 1);
        twist_cov[35] = track.state_cov(S_YAW_RATE, S_YAW_RATE);

        auto& acceleration = kinematics.acceleration_with_covariance.accel;
        acceleration.linear.x =
            cosine * track.state_vec(S_AX) + sine * track.state_vec(S_AY);
        acceleration.linear.y =
            -sine * track.state_vec(S_AX) + cosine * track.state_vec(S_AY);
        auto& acceleration_cov = kinematics.acceleration_with_covariance.covariance;
        std::fill(acceleration_cov.begin(), acceleration_cov.end(), 0.0);
        acceleration_cov[0] = track.state_cov(S_AX, S_AX);
        acceleration_cov[7] = track.state_cov(S_AY, S_AY);

        kinematics.orientation_availability =
            classification.label == autoware_perception_msgs::ObjectClassification::CAR ||
                    classification.label == autoware_perception_msgs::ObjectClassification::TRUCK
                ? autoware_perception_msgs::TrackedObjectKinematics::AVAILABLE
                : autoware_perception_msgs::TrackedObjectKinematics::SIGN_UNKNOWN;
        kinematics.is_stationary =
            std::hypot(output_velocity_x, output_velocity_y) < 0.5;

        output.shape.type = autoware_perception_msgs::Shape::BOUNDING_BOX;
        output.shape.dimensions.x = track.dimension.length;
        output.shape.dimensions.y = track.dimension.width;
        output.shape.dimensions.z = track.dimension.height;
        o_tracked_objects_.objects.push_back(output);
        if (publish_center_motion_diagnostics_) {
            // The diagnostic split describes where the causal estimator
            // passed its quality gate. This remains meaningful when output
            // override is disabled, enabling a paired baseline on the exact
            // same accepted samples.
            if (center_motion_available) {
                o_center_motion_active_objects_.objects.push_back(output);
            } else {
                o_center_motion_inactive_objects_.objects.push_back(output);
            }
        }
    }
    vehicle_orientation_canonicalizer_.RetainOnly(active_track_ids);
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "[EKF tracker] Canonicalized " << canonicalized_vehicle_count
        << " moving vehicle orientations in this frame");
}

void EkfMultiObjectTrackingNode::VisualizeTrackObjects(const mc_mot::TrackStructs& track_structs,
                                                         std::string frame_id) {
    o_jsk_tracked_objects_.header.frame_id = frame_id;
    o_jsk_tracked_objects_.header.stamp = ros::Time(track_structs.time_stamp);

    o_jsk_tracked_objects_.boxes.clear();
    o_vis_track_info_.markers.clear();

    if (config_.visualize_mesh == true) {
        // Visualize EGO STL
        o_vis_ego_stl_.header.frame_id = str_detection_frame_id_;
        o_vis_ego_stl_.ns = "ego_stl";
        o_vis_ego_stl_.id = 0;
        o_vis_ego_stl_.type = visualization_msgs::Marker::MESH_RESOURCE;
        o_vis_ego_stl_.action = visualization_msgs::Marker::ADD;

        o_vis_ego_stl_.mesh_resource = "package://ekf_multi_object_tracking/dae/Ioniq5.stl";

        o_vis_ego_stl_.pose.position.x = 0.0;
        o_vis_ego_stl_.pose.position.y = 0;
        o_vis_ego_stl_.pose.position.z = -cfg_vec_d_ego_to_lidar_xyz_m_[2];

        tf2::Quaternion quat;
        quat.setRPY(0, 0, M_PI); // Roll=0, Pitch=0, Yaw=180 degrees (π radians)
        quat.normalize();

        o_vis_ego_stl_.pose.orientation.x = quat.x();
        o_vis_ego_stl_.pose.orientation.y = quat.y();
        o_vis_ego_stl_.pose.orientation.z = quat.z();
        o_vis_ego_stl_.pose.orientation.w = quat.w();

        o_vis_ego_stl_.scale.x = 1.0;
        o_vis_ego_stl_.scale.y = 1.0;
        o_vis_ego_stl_.scale.z = 1.0;

        // Set color
        o_vis_ego_stl_.color.r = 1.0f;
        o_vis_ego_stl_.color.g = 1.0f;
        o_vis_ego_stl_.color.b = 1.0f;
        o_vis_ego_stl_.color.a = 1.0;
    }

    for (auto track : track_structs.track) {
        double track_vel = sqrt(track.state_vec(S_VX) * track.state_vec(S_VX)
                                + track.state_vec(S_VY) * track.state_vec(S_VY));
        double track_acc = sqrt(track.state_vec(S_AX) * track.state_vec(S_AX)
                                + track.state_vec(S_AY) * track.state_vec(S_AY));

        bool b_visualize_cur_track = IsVisualizeTrack(track);

        // 1. JSK Box
        if (b_visualize_cur_track == true) {
            jsk_recognition_msgs::BoundingBox o_jsk_bbox;
            o_jsk_bbox.header.frame_id = frame_id;
            // o_jsk_bbox.header.stamp = ros::Time(track_structs.time_stamp);
            o_jsk_bbox.header.stamp = ros::Time(track.update_time);

            o_jsk_bbox.label = track.getRepClass();
            o_jsk_bbox.dimensions.x = track.dimension.length;
            o_jsk_bbox.dimensions.y = track.dimension.width;
            o_jsk_bbox.dimensions.z = track.dimension.height;
            o_jsk_bbox.pose.position.x = track.state_vec(S_X);
            o_jsk_bbox.pose.position.y = track.state_vec(S_Y);
            o_jsk_bbox.pose.position.z = track.object_z;

            // bounding box
            tf2::Quaternion myQuaternion;
            myQuaternion.setRPY(0, 0, track.state_vec(S_YAW));
            myQuaternion = myQuaternion.normalize();
            o_jsk_bbox.pose.orientation.x = myQuaternion.getX();
            o_jsk_bbox.pose.orientation.y = myQuaternion.getY();
            o_jsk_bbox.pose.orientation.z = myQuaternion.getZ();
            o_jsk_bbox.pose.orientation.w = myQuaternion.getW();

            // Visualzation of velocity
            o_jsk_bbox.value = track_vel;

            o_jsk_tracked_objects_.boxes.push_back(o_jsk_bbox);
        }

        // 2. Markerarray
        if (track.track_id != -1) {
            // 1. Track Text
            visualization_msgs::Marker vis_track_info;
            vis_track_info.header.frame_id = frame_id;
            vis_track_info.header.stamp = ros::Time(track_structs.time_stamp);

            vis_track_info.ns = "track_info";
            vis_track_info.id = track.track_id;
            vis_track_info.action = visualization_msgs::Marker::ADD;
            vis_track_info.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            // vis_track_info.lifetime = ros::Duration(0.1);
           
            std::ostringstream display_text;
            display_text << "ID (" << track.track_id << ") score: " << std::fixed << std::setprecision(2)
                         << std::round(track.detection_confidence * 100.0) / 100.0
                         << "\ncls    : " << track.getRepClass() << " prob: " << std::fixed << std::setprecision(2)
                         << std::round(track.getRepClassProb() * 100.0) / 100.0 << "\nVel    : " << std::fixed
                         << std::setprecision(3) << std::round(track_vel * 1000.0) / 1000.0 << "m/s"
                         << "\nAccel  : " << std::fixed << std::setprecision(3)
                         << std::round(track_acc * 1000.0) / 1000.0 << "m/s^2"
                         << "\nYawVel : " << std::fixed << std::setprecision(3)
                         << std::round(track.state_vec(S_YAW_RATE) * 180.0 / M_PI * 1000.0) / 1000.0 << "deg/s"
                         << "\nHscore : " << std::fixed << std::setprecision(3)
                         << std::round(track.direction_score * 1000.0) / 1000.0;

            vis_track_info.text = display_text.str();

            vis_track_info.scale.z = 0.5;

            vis_track_info.color.r = 1.0f;
            vis_track_info.color.g = 1.0f;
            vis_track_info.color.b = 1.0f;
            vis_track_info.color.a = 1.0f;

            vis_track_info.pose.position.x = track.state_vec(S_X);
            vis_track_info.pose.position.y = track.state_vec(S_Y);
            vis_track_info.pose.position.z = track.object_z + 3.0;
            vis_track_info.pose.orientation = tf::createQuaternionMsgFromYaw(track.state_vec(S_YAW));

            if (b_visualize_cur_track == false) {
                vis_track_info.action = visualization_msgs::Marker::DELETE;
                vis_track_info.color.a = 0.0f;
            }

            o_vis_track_info_.markers.push_back(vis_track_info);

            // 2. Cov Cylinder
            visualization_msgs::Marker cov_marker;
            cov_marker.header.frame_id = frame_id;
            cov_marker.header.stamp = ros::Time(track_structs.time_stamp);
            cov_marker.ns = "covariance";
            cov_marker.id = track.track_id;
            cov_marker.action = visualization_msgs::Marker::ADD;
            cov_marker.type = visualization_msgs::Marker::CYLINDER;
            cov_marker.pose.position.x = track.state_vec(S_X);
            cov_marker.pose.position.y = track.state_vec(S_Y);
            cov_marker.pose.position.z = track.object_z - track.dimension.height / 2.0;
            cov_marker.pose.orientation.x = 0.0;
            cov_marker.pose.orientation.y = 0.0;
            cov_marker.pose.orientation.z = 0.0;
            cov_marker.pose.orientation.w = 1.0;

            // Calculate eigenvalues and eigenvectors of the covariance matrix
            Eigen::Matrix2d cov_matrix;
            cov_matrix << track.state_cov(S_X, S_X), track.state_cov(S_X, S_Y),
                            track.state_cov(S_Y, S_X), track.state_cov(S_Y, S_Y);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov_matrix);
            Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
            Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();

            // Set the size of the ellipse using eigenvalues
            cov_marker.scale.x = 2 * std::sqrt(eigenvalues[0]); // Major axis
            cov_marker.scale.y = 2 * std::sqrt(eigenvalues[1]); // Minor axis

            cov_marker.scale.z = 0.1; // Height of the cylinder (set small)

            cov_marker.color.r = 0.0f;
            cov_marker.color.g = 1.0f;
            cov_marker.color.b = 0.0f;
            cov_marker.color.a = 0.3f;

            // Set the direction of the ellipse using eigenvectors
            double angle = std::atan2(eigenvectors(1, 0), eigenvectors(0, 0));
            cov_marker.pose.orientation = tf::createQuaternionMsgFromYaw(angle);

            if (b_visualize_cur_track == false) {
                cov_marker.action = visualization_msgs::Marker::DELETE;
                cov_marker.color.a = 0.0f;
            }

            o_vis_track_info_.markers.push_back(cov_marker);

            // 3. Velocity Arrow
            visualization_msgs::Marker vel_arrow_marker;
            vel_arrow_marker.header.frame_id = frame_id;
            vel_arrow_marker.header.stamp = ros::Time(track_structs.time_stamp);
            vel_arrow_marker.ns = "velocity_arrow";
            vel_arrow_marker.id = track.track_id;
            vel_arrow_marker.type = visualization_msgs::Marker::ARROW;
            vel_arrow_marker.action = visualization_msgs::Marker::ADD;

            geometry_msgs::Point start, end;
            start.x = track.state_vec(S_X);
            start.y = track.state_vec(S_Y);
            start.z = track.object_z - track.dimension.height / 2.0;

            end.x = start.x + track.state_vec(S_VX); 
            end.y = start.y + track.state_vec(S_VY);
            end.z = start.z;

            vel_arrow_marker.points.push_back(start);
            vel_arrow_marker.points.push_back(end);

            vel_arrow_marker.scale.x = 0.3; 
            vel_arrow_marker.scale.y = 0.5; 
            vel_arrow_marker.scale.z = 0.5; 

            vel_arrow_marker.color.r = 1.0f;
            vel_arrow_marker.color.g = 1.0f;
            vel_arrow_marker.color.b = 1.0f;
            vel_arrow_marker.color.a = 0.6f;

            vel_arrow_marker.pose.orientation.x = 0.0;
            vel_arrow_marker.pose.orientation.y = 0.0;
            vel_arrow_marker.pose.orientation.z = 0.0;
            vel_arrow_marker.pose.orientation.w = 1.0;

            if (b_visualize_cur_track == false) {
                vel_arrow_marker.action = visualization_msgs::Marker::DELETE;
                vel_arrow_marker.color.a = 0.0f;
            }

            o_vis_track_info_.markers.push_back(vel_arrow_marker);

            // Target STL 시각화
            if (config_.visualize_mesh == true) {
                visualization_msgs::Marker track_stl;
                track_stl.header.frame_id = frame_id;
                track_stl.ns = "track_mesh";
                track_stl.id = track.track_id;
                track_stl.type = visualization_msgs::Marker::MESH_RESOURCE;
                track_stl.action = visualization_msgs::Marker::ADD;

                double stl_heading = track.state_vec(S_YAW);

                track_stl.scale.x = 1.0;
                track_stl.scale.y = 1.0;
                track_stl.scale.z = 1.0;

                switch (track.getRepClass()) {
                case mc_mot::ObjectClass::CAR:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/SimpleCar_turn.dae";
                    track_stl.scale.x = track.dimension.length / 4.6;
                    track_stl.scale.y = track.dimension.width / 2.1;
                    track_stl.scale.z = track.dimension.height / 1.59;

                    break;
                case mc_mot::ObjectClass::TRUCK:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/bus.stl";

                    track_stl.scale.x = track.dimension.length / 15.952;
                    track_stl.scale.y = track.dimension.width / 3.66;
                    track_stl.scale.z = track.dimension.height / 4.87;

                    break;
                case mc_mot::ObjectClass::PEDESTRIAN:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/pedestrian.stl";
                    stl_heading += M_PI / 2.0;

                    break;
                case mc_mot::ObjectClass::BICYCLE:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/moto_simple_1.stl";

                    track_stl.scale.x = track.dimension.length / 3.19;
                    track_stl.scale.y = track.dimension.width / 1.1;
                    track_stl.scale.z = track.dimension.height / 1.48;

                    break;
                default:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/SimpleCar_turn.dae";

                    track_stl.scale.x = track.dimension.length / 4.6;
                    track_stl.scale.y = track.dimension.width / 2.1;
                    track_stl.scale.z = track.dimension.height / 1.59;

                    break;
                }

                track_stl.pose.position.x = track.state_vec(S_X);
                track_stl.pose.position.y = track.state_vec(S_Y);
                track_stl.pose.position.z = track.object_z - track.dimension.height / 2.0;

                tf2::Quaternion quat;
                quat.setRPY(0, 0, stl_heading); // Roll=0, Pitch=0, Yaw=180 degrees (π radians)
                quat.normalize();

                track_stl.pose.orientation.x = quat.x();
                track_stl.pose.orientation.y = quat.y();
                track_stl.pose.orientation.z = quat.z();
                track_stl.pose.orientation.w = quat.w();

                // Set color
                track_stl.color.r = 0.8f;
                track_stl.color.g = 0.8f;
                track_stl.color.b = 0.8f;
                track_stl.color.a = 1.0;

                // Tracks with low detection score are gray
                if (track.detection_confidence < 0.5) {
                    track_stl.color.r = 0.4f;
                    track_stl.color.g = 0.4f;
                    track_stl.color.b = 0.4f;
                }

                if (b_visualize_cur_track == false) {
                    track_stl.action = visualization_msgs::Marker::DELETE;
                    track_stl.color.a = 0.0f;
                }

                o_vis_track_info_.markers.push_back(track_stl);
            }
        }
    }
}

void EkfMultiObjectTrackingNode::ConvertTrackGlobalToLocal(mc_mot::TrackStructs& track_structs,
                                                             mc_mot::ObjectState synced_lidar_state) {
    double cos_yaw = std::cos(synced_lidar_state.yaw);
    double sin_yaw = std::sin(synced_lidar_state.yaw);

    double glob_lidar_vx = synced_lidar_state.v_x;
    double glob_lidar_vy = synced_lidar_state.v_y;

    for (auto& track : track_structs.track) {
        if (track.is_init == true) {

            // Position in track
            double x_track = track.state_vec(S_X);
            double y_track = track.state_vec(S_Y);

            double x_lidar, y_lidar;

            double glob_rel_vx = 0.0;
            double glob_rel_vy = 0.0;   

            if(config_.global_coord_track == false){ // Local Tracker. Compensate absolute vel, acc, yawrate. No Orienation change
                // Lidar coordinate position is same as track position
                double x_lidar = x_track;
                double y_lidar = y_track;

                track.state_vec(S_X) = x_lidar;
                track.state_vec(S_Y) = y_lidar;

                // Convert Global lidar state to local coordinate
                double local_lidar_vx = cos_yaw * synced_lidar_state.v_x + sin_yaw * synced_lidar_state.v_y;
                double local_lidar_vy = -sin_yaw * synced_lidar_state.v_x + cos_yaw * synced_lidar_state.v_y;

                double local_lidar_ax = cos_yaw * synced_lidar_state.a_x + sin_yaw * synced_lidar_state.a_y;
                double local_lidar_ay = -sin_yaw * synced_lidar_state.a_x + cos_yaw * synced_lidar_state.a_y;

                // Compensate velocity by yawrate
                double yaw_rate_lidar = synced_lidar_state.yaw_rate;
                double v_add_x = -yaw_rate_lidar * y_lidar;
                double v_add_y =  yaw_rate_lidar * x_lidar;

                // Compensate velocity
                track.state_vec(S_VX) = track.state_vec(S_VX) + local_lidar_vx + v_add_x;
                track.state_vec(S_VY) = track.state_vec(S_VY) + local_lidar_vy + v_add_y;

                // Compensate acceleration
                track.state_vec(S_AX) = track.state_vec(S_AX) + local_lidar_ax;
                track.state_vec(S_AY) = track.state_vec(S_AY) + local_lidar_ay;

                // Convert yaw rate
                track.state_vec(S_YAW_RATE) = track.state_vec(S_YAW_RATE) + yaw_rate_lidar;

            }else{ // Global Tracker. Compensate orientation
                // Relative position from Lidar (convert to Lidar coordinates)
                double x_relative = x_track - synced_lidar_state.x;
                double y_relative = y_track - synced_lidar_state.y;

                // Relative position in Lidar coordinates
                double x_lidar = cos_yaw * x_relative + sin_yaw * y_relative;
                double y_lidar = -sin_yaw * x_relative + cos_yaw * y_relative;

                track.state_vec(S_X) = x_lidar;
                track.state_vec(S_Y) = y_lidar;
                track.state_vec(S_YAW) = track.state_vec(S_YAW) - synced_lidar_state.yaw;

                // Convert global velocity to Lidar coordinates
                track.state_vec(S_VX) = cos_yaw * track.state_vec(S_VX) + sin_yaw * track.state_vec(S_VY);
                track.state_vec(S_VY) = -sin_yaw * track.state_vec(S_VX) + cos_yaw * track.state_vec(S_VY);

                // Convert acceleration to Lidar coordinates
                track.state_vec(S_AX) = cos_yaw * track.state_vec(S_AX) + sin_yaw * track.state_vec(S_AY);
                track.state_vec(S_AY) = -sin_yaw * track.state_vec(S_AX) + cos_yaw * track.state_vec(S_AY);
            }
        }
    }
}

bool EkfMultiObjectTrackingNode::IsVisualizeTrack(const mc_mot::TrackStruct& track) {
    bool b_visualize_cur_track = true;
    if (config_.output_confirmed_track == true && track.is_confirmed == false) b_visualize_cur_track = false;
    if (track.is_init == false) b_visualize_cur_track = false;

    return b_visualize_cur_track;
}

// ========== Utils ==========

mc_mot::ObjectState EkfMultiObjectTrackingNode::GetSyncedLidarState(
        double object_time, const std::deque<mc_mot::ObjectState>& deque_lidar_state) {
    mc_mot::ObjectState object_synced_state = deque_lidar_state.back();
    double minimum_time_diff = FLT_MAX;

    for (auto& lidar_state : deque_lidar_state) {
        if (fabs(object_time - lidar_state.time_stamp) < minimum_time_diff) {
            minimum_time_diff = fabs(object_time - lidar_state.time_stamp);
            object_synced_state = lidar_state;
        }

        if (minimum_time_diff < FLT_MIN) {
            break;
        }
    }

    return object_synced_state;
}

mc_mot::ObjectState EkfMultiObjectTrackingNode::PredictNextState(const mc_mot::ObjectState& state_t_minus_1,
                                                                   const mc_mot::ObjectState& state_t) {
    mc_mot::ObjectState state_t_plus_1;

    // Calculate time interval
    double delta_t = state_t.time_stamp - state_t_minus_1.time_stamp;

    // Predict time
    state_t_plus_1.time_stamp = state_t.time_stamp + delta_t;

    // Predict position
    state_t_plus_1.x = state_t.x + state_t.v_x * delta_t + 0.5 * state_t.a_x * delta_t * delta_t;
    state_t_plus_1.y = state_t.y + state_t.v_y * delta_t + 0.5 * state_t.a_y * delta_t * delta_t;
    state_t_plus_1.z = state_t.z + state_t.v_z * delta_t + 0.5 * state_t.a_z * delta_t * delta_t;

    // Predict velocity
    state_t_plus_1.v_x = state_t.v_x + state_t.a_x * delta_t;
    state_t_plus_1.v_y = state_t.v_y + state_t.a_y * delta_t;
    state_t_plus_1.v_z = state_t.v_z + state_t.a_z * delta_t;

    // Maintain acceleration at T time
    state_t_plus_1.a_x = state_t.a_x;
    state_t_plus_1.a_y = state_t.a_y;
    state_t_plus_1.a_z = state_t.a_z;

    // Predict angle
    state_t_plus_1.roll = state_t.roll + state_t.roll_rate * delta_t;
    state_t_plus_1.pitch = state_t.pitch + state_t.pitch_rate * delta_t;
    state_t_plus_1.yaw = state_t.yaw + state_t.yaw_rate * delta_t;

    // Predict angular velocity
    state_t_plus_1.roll_rate = state_t.roll_rate;
    state_t_plus_1.pitch_rate = state_t.pitch_rate;
    state_t_plus_1.yaw_rate = state_t.yaw_rate;

    return state_t_plus_1;
}

void EkfMultiObjectTrackingNode::TransformMeasLiDAR2Global(mc_mot::Meastruct& i_meas,
                                                             const std::deque<mc_mot::ObjectState>& deque_lidar_state) {
    mc_mot::ObjectState object_synced_state = GetSyncedLidarState(i_meas.state.time_stamp, deque_lidar_state);

    Eigen::Affine3d world_to_lidar_affine =
            CreateTransformation(object_synced_state.x, object_synced_state.y, cfg_vec_d_ego_to_lidar_xyz_m_[2], 0, 0,
                                 object_synced_state.yaw);

    Eigen::Affine3d lidar_to_object_affine =
            CreateTransformation(i_meas.state.x, i_meas.state.y, i_meas.state.z, 0, 0, i_meas.state.yaw);

    Eigen::Affine3d world_to_object_affine = world_to_lidar_affine * lidar_to_object_affine;
    Eigen::Vector3d world_to_object_translation = world_to_object_affine.translation();
    Eigen::Matrix3d world_to_object_rotation = world_to_object_affine.rotation();

    i_meas.state.x = world_to_object_translation(0);
    i_meas.state.y = world_to_object_translation(1);
    i_meas.state.z = world_to_object_translation(2);
    i_meas.state.yaw = atan2(world_to_object_rotation(1, 0), world_to_object_rotation(0, 0));
}

void EkfMultiObjectTrackingNode::AngleBasedTimeCompensation(mc_mot::Meastruct& i_meas) {
    double object_angle_ego_ccw_rad, object_angle_behind_cw_rad;
    double dt_from_start_scan_sec;

    object_angle_ego_ccw_rad = atan2(i_meas.state.y, i_meas.state.x); // -pi ~ pi
    object_angle_behind_cw_rad = M_PI - object_angle_ego_ccw_rad;     // 0 ~ 2pi

    dt_from_start_scan_sec = config_.lidar_rotation_period * (object_angle_behind_cw_rad / (2.0 * M_PI)); // 0 ~ 0.1 sec

    if (config_.lidar_sync_scan_start == true) {
        i_meas.state.time_stamp += dt_from_start_scan_sec; // Using start packet lidar
    }
    else {
        i_meas.state.time_stamp -= (config_.lidar_rotation_period - dt_from_start_scan_sec);
    }
}

Eigen::Affine3d EkfMultiObjectTrackingNode::CreateTransformation(double x, double y, double z, double roll,
                                                                   double pitch, double yaw) {
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();

    // Set translation
    transform.translation() << x, y, z;

    // Set rotation (yaw -> pitch -> roll order)
    transform.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    transform.rotate(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    transform.rotate(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));

    return transform;
}



// --------------------------------------------------

void EkfMultiObjectTrackingNode::Exec(int num_thread) {
    boost::thread main_thread(boost::bind(&EkfMultiObjectTrackingNode::MainLoop, this));

    ros::AsyncSpinner spinner(num_thread);
    spinner.start();
    ros::waitForShutdown();

    main_thread.join();
}

void EkfMultiObjectTrackingNode::MainLoop() {
    ros::Rate loop_rate(100);
    ros::Time last_log_time = ros::Time::now();
    while (ros::ok()) {

        // Run algorithm
        auto start_time = std::chrono::high_resolution_clock::now(); // 시작 시간 기록

        Run();

        // Publish topics
        Publish();
        auto end_time = std::chrono::high_resolution_clock::now(); // 종료 시간 기록
        std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time; // 경과 시간 계산
        ROS_DEBUG_THROTTLE(1.0, "[EKF tracker] loop execution %.3f ms", elapsed_time.count());

        loop_rate.sleep();
    }
}
int main(int argc, char** argv) {
    std::string node_name = "ekf_multi_object_tracking";
    ros::init(argc, argv, node_name);

    EkfMultiObjectTrackingNode main_task;

    main_task.Exec(4);

    return 0;
}
