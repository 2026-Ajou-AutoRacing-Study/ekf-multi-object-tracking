# ekf-multi-object-tracking
EKF-based Multi Object Tracking

![ekf_multi_object_tracking](docs/tracking_example.gif)

## Features

The `ekf_multi_object_tracking` package supports the following features:

- **Low-Frequency Detection to High-Frequency Tracking**: Converts low-frequency detection data into high-frequency tracking data, ensuring smooth and accurate object tracking.

- **Multiple Localization Inputs (TODO)**: Supports various localization input sources, allowing flexibility in integrating different localization systems.

- **Multiple Prediction Models**: Offers a selection of prediction models, including Constant Velocity (CV), Constant Turn Rate and Velocity (CTRV), Constant Acceleration (CA), and Constant Turn Rate and Acceleration (CTRA), to suit different tracking scenarios.

- **Individual Time Consideration for Rotating LiDAR**: Takes into account the individual detection times of objects from rotating LiDAR sensors, improving the accuracy of motion compensation and tracking.

- **Vehicle Heading Correction**: Provides functionality to correct the vehicle's heading direction, enhancing the accuracy of the tracking system.

- **Class Correction**: Supports class correction features, allowing for improved classification accuracy of tracked objects.


These features make the package versatile and adaptable to a wide range of multi-object tracking applications.

## I/O

### Input
| Description                              | Type                      |
|------------------------------------------|---------------------------|
| Detection info (`~input/objects`)         | `autoware_perception_msgs::DetectedObjects` |
| Localization info (optional)             | `nav_msgs::Odometry` (optional)          |

### Output
| Description                              | Type                                |
|------------------------------------------|-------------------------------------|
| Tracked objects (`~output/objects`)       | `autoware_perception_msgs::TrackedObjects` |
| Debug bounding boxes                     | `jsk_recognition_msgs::BoundingBoxArray` |
| Debug markers                            | `visualization_msgs::MarkerArray`   |

The detection pose is transformed from `header.frame_id` into `map` at the
measurement timestamp before it enters the EKF.  Standard tracked output stays
in `map`; JSK boxes and markers are visualization-only compatibility outputs.

## How to use
### 1. Install ROS messages and other libraries
* for Ubuntu 20.04 (noetic)
```bash
sudo apt install ros-noetic-jsk-rviz-plugins
```

### 2. Install Code
```bash
cd ~/catkin_ws/src
git clone https://github.com/jaeyoungjo99/ekf-multi-object-tracking.git
cd ..
catkin_make
```

### 3. Launch Code
```bash
source devel/setup.bash
roslaunch ekf_multi_object_tracking ekf_multi_class_object_tracking.launch 
```

The default launch contract is:

```text
/perception/object_recognition/detection/objects
  -> /perception/object_recognition/tracking/objects
```

Both names can be changed with `input_objects:=...` and `output_objects:=...`.


## Configuration

The behavior of the `ekf_multi_object_tracking` node can be customized using the `config/config.yaml` file. Below is a description of the key configuration parameters:

### Topic Names

Standard data topics are private node interfaces and are selected by launch
remaps. The `topic_name/output_track_jsk` and
`topic_name/output_track_marker` YAML entries affect debug output only.

### Configuration Options
- **input_localization**: Selects the input localization source.
  - `0`: None
  - `1`: nav_msgs::Odometry
  - `2`: NavSatFix (TODO)
- **global_coord_track**: Whether the tracker is performed in global coordinates
  - `true`: Global coordinate tracker
  - `false`: Local coordinate tracker
- **output_local_coord**: Determines if the output is in local coordinates.
  - `true`: Output local coordinate
  - `false`: Output global coordinate
- **output_period_lidar**: Synchronizes output with LiDAR.
  - `true`: LiDAR synced output
  - `false`: Motion synced output
- **output_confirmed_track**: Outputs only confirmed tracks if set to `true`.
- **use_predefined_ref_point**: Uses a predefined reference point if set to `true`.
- **reference_lat, reference_lon, reference_height**: Predefined reference point coordinates.
- **cal_detection_individual_time**: Enables LiDAR motion compensation.
- **lidar_rotation_period**: Sets the LiDAR rotation period.
- **lidar_sync_scan_start**: Synchronizes LiDAR time to scan start if set to `true`.
- **max_association_dist_m**: Maximum distance for track association.
- **time_aware_track_lifecycle**: Uses elapsed message time, rather than a
  detector-frame count, to retire confirmed tracks. Tentative tracks keep the
  original strict confirmation/deletion behavior.
- **confirmed_track_max_coast_time_sec**: Maximum unobserved lifetime for a
  confirmed moving track (`0.50 s` by default).
- **confirmed_stationary_track_max_coast_time_sec**: Maximum unobserved
  lifetime for confirmed UNKNOWN or low-speed CAR/TRUCK tracks (`1.00 s` by
  default). Pedestrians do not use this longer stationary lifetime.
- **suppress_duplicate_track_birth**: Prevents an unmatched large-object
  measurement from creating a second ID very near a mature track already
  associated in the same frame. The distance, minimum size, and maximum size
  ratio are controlled by the corresponding `duplicate_birth_*` parameters.
- **prediction_model**: Selects the prediction model.
  - `0`: CV
  - `1`: CTRV (default in the AjouNice2026 team fork)
  - `2`: CA
  - `3`: CTRA
- **system_noise_std_xy_m, system_noise_std_yaw_deg, etc.**: System noise parameters.
- **meas_noise_std_xy_m, meas_noise_std_yaw_deg**: Measurement noise parameters.
- **dimension_filter_alpha**: Filtering parameter for object dimensions.
- **use_kinematic_model**: Aligns velocity direction to heading if set to `true`.
- **canonicalize_vehicle_orientation_to_motion**: Resolves the equivalent
  `(yaw, speed)` versus `(yaw + pi, -speed)` representation for moving
  CAR/TRUCK tracks at the standardized `TrackedObjects` output boundary.
- **orientation_canonicalization_min_speed_mps**: Below this speed the last
  orientation decision is retained instead of trusting a noisy velocity yaw.
- **orientation_flip_enter_error_deg / orientation_flip_exit_error_deg**:
  Hysteresis thresholds around the 90-degree motion-direction boundary. The
  production defaults are 100/80 degrees.
- **use_yaw_rate_filtering**: Restricts yaw rate based on velocity.
- **max_steer_deg**: Maximum steering angle.
- **visualize_mesh**: Enables mesh visualization if set to `true`.

### Vehicle Origin
- **vehicle_origin**: Defines the vehicle origin.
  - `0`: Rear Axle
  - `1`: C.G.

### Transformations
- **rear_to_main_lidar**: Transformation from rear axle to main LiDAR.
  - `parent_frame_id`: Parent frame ID
  - `child_frame_id`: Child frame ID
  - `transform_xyz_m`: Translation in meters
  - `rotation_rpy_deg`: Rotation in degrees
- **cg_to_main_lidar**: Transformation from C.G. to main LiDAR.

Users can modify these parameters in the `config/config.yaml` file to suit their specific requirements.
