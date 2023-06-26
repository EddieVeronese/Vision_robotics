cmake_minimum_required(VERSION 3.0.2)
project(yolov5_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  detection_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES yolov5_ros
#  CATKIN_DEPENDS rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS
  src/detect.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

