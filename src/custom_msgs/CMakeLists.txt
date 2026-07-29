cmake_minimum_required(VERSION 3.8)
project(custom_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ActionChunk.msg"
  "msg/AsyncHiddenState.msg"
  "msg/ImageWithSeqNum.msg"
  "msg/LeftRightFloat32.msg"
  "msg/LeftRightInt32.msg"
  "msg/TicImageTokens.msg"
  "msg/TicKVCache.msg"
  "msg/TicLatencyMetadata.msg"
  "msg/TicPixelValues.msg"
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
