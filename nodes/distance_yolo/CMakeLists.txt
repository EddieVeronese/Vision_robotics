cmake_minimum_required(VERSION 3.5)
project(distance_yolo)


find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  sensor_msgs
  detection_msgs
  pcl_ros
  pcl_conversions
  final_msgs
)

catkin_package()


install(PROGRAMS
  src/distance_yolo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

