cmake_minimum_required(VERSION 2.8.3)
project(ekf_multi_object_tracking)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  autoware_perception_msgs
  roscpp
  rosbag
  tf
  
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2_geometry_msgs
  tf2_ros
  unique_identifier_msgs
  eigen_conversions
  
  jsk_recognition_msgs

)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS
    autoware_perception_msgs
    geometry_msgs
    jsk_recognition_msgs
    roscpp
    tf
    tf2_geometry_msgs
    tf2_ros
    unique_identifier_msgs
    visualization_msgs
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}

)

add_executable(${PROJECT_NAME} 
    src/ekf_multi_object_tracking_node.cpp
    src/algorithm/ekf_multi_object_tracking.cpp
)

add_dependencies(${PROJECT_NAME} 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_class_transition_guard test/test_class_transition_guard.cpp)
  if(TARGET test_class_transition_guard)
    target_link_libraries(test_class_transition_guard ${catkin_LIBRARIES})
  endif()
  foreach(consistency_test test_class_transition_integration test_shape_consistent_association test_unknown_output_projection)
    catkin_add_gtest(${consistency_test}
      test/${consistency_test}.cpp src/algorithm/ekf_multi_object_tracking.cpp)
    if(TARGET ${consistency_test})
      target_link_libraries(${consistency_test} ${catkin_LIBRARIES})
    endif()
  endforeach()
  catkin_add_gtest(
    test_association_diagnostics
    test/test_association_diagnostics.cpp
    src/algorithm/ekf_multi_object_tracking.cpp
  )
  if(TARGET test_association_diagnostics)
    target_link_libraries(test_association_diagnostics ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_detection_input_receipt test/test_detection_input_receipt.cpp)
  if(TARGET test_detection_input_receipt)
    target_link_libraries(test_detection_input_receipt ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(
    test_pedestrian_association_ranking
    test/test_pedestrian_association_ranking.cpp
    src/algorithm/ekf_multi_object_tracking.cpp
  )
  if(TARGET test_pedestrian_association_ranking)
    target_link_libraries(test_pedestrian_association_ranking ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(
    test_vehicle_orientation_canonicalizer
    test/test_vehicle_orientation_canonicalizer.cpp
  )
  if(TARGET test_vehicle_orientation_canonicalizer)
    target_link_libraries(test_vehicle_orientation_canonicalizer ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(
    test_detection_velocity_fusion
    test/test_detection_velocity_fusion.cpp
    src/algorithm/ekf_multi_object_tracking.cpp
  )
  if(TARGET test_detection_velocity_fusion)
    target_link_libraries(test_detection_velocity_fusion ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(
    test_track_lifecycle
    test/test_track_lifecycle.cpp
    src/algorithm/ekf_multi_object_tracking.cpp
  )
  if(TARGET test_track_lifecycle)
    target_link_libraries(test_track_lifecycle ${catkin_LIBRARIES})
  endif()
endif()
