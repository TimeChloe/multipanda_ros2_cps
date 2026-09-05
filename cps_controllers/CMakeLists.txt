cmake_minimum_required(VERSION 3.8)
project(cps_controllers)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(mujoco_ros_msgs REQUIRED)
# CMake 3.22's FindBoost does not map python310 to boost/python.hpp.
set(_Boost_PYTHON310_HEADERS "boost/python.hpp")
find_package(pinocchio REQUIRED)
find_package(panda_motion_generator_msgs REQUIRED)
find_package(cps_human_workspace REQUIRED)
find_package(cps_safety_monitor REQUIRED)
find_package(cps_trajectory_generators REQUIRED)
find_package(Threads REQUIRED)

pluginlib_export_plugin_description_file(
  controller_interface
  controllers.xml
)

add_library(${PROJECT_NAME} SHARED
  src/joint_impedance_energybudget_controller.cpp
  src/nonlinear_impedance_controller.cpp
  src/reachable_cartesian_impedance/async.cpp
  src/reachable_cartesian_impedance_controller.cpp
  src/reachable_cartesian_impedance/logging.cpp
  src/reachable_cartesian_impedance/visualization.cpp
  src/free_motion_external_wrench_controller.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  controller_interface
  pluginlib
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  realtime_tools
  franka_semantic_components
  std_msgs
  geometry_msgs
  visualization_msgs
  mujoco_ros_msgs
  panda_motion_generator_msgs
  cps_human_workspace
  cps_safety_monitor
  cps_trajectory_generators
)

target_link_libraries(${PROJECT_NAME}
  pinocchio::pinocchio
  Threads::Threads
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES controllers.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  controller_interface
  pluginlib
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  realtime_tools
  franka_semantic_components
  std_msgs
  geometry_msgs
  visualization_msgs
  mujoco_ros_msgs
  pinocchio
  panda_motion_generator_msgs
  cps_human_workspace
  cps_safety_monitor
  cps_trajectory_generators
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_latest_value_mailbox
    test/test_latest_value_mailbox.cpp
  )
  target_include_directories(test_latest_value_mailbox PRIVATE include)
endif()

ament_package()
