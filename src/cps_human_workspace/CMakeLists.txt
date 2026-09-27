cmake_minimum_required(VERSION 3.8)
project(cps_human_workspace)
# rosidl_generate_interfaces creates targets under the package name, so the
# C++ model library needs a private CMake target name. It still installs and
# exports publicly as cps_human_workspace.
set(HUMAN_WORKSPACE_MODEL_TARGET human_workspace_model)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HumanReachableSet.msg"
  "msg/HumanWorkspace.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp")

add_library(${HUMAN_WORKSPACE_MODEL_TARGET} SHARED
  src/human_workspace.cpp
  src/human_workspace_message.cpp
)

set_target_properties(${HUMAN_WORKSPACE_MODEL_TARGET}
  PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
    EXPORT_NAME ${PROJECT_NAME}
)

target_include_directories(${HUMAN_WORKSPACE_MODEL_TARGET}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${HUMAN_WORKSPACE_MODEL_TARGET}
  geometry_msgs
  rclcpp
  std_msgs
)

target_link_libraries(${HUMAN_WORKSPACE_MODEL_TARGET}
  Eigen3::Eigen
  yaml-cpp
  ${cpp_typesupport_target}
)

add_executable(human_workspace_publisher
  src/human_workspace_publisher_node.cpp
)

add_executable(human_reachable_set_visualizer
  src/human_reachable_set_visualizer_node.cpp
)

ament_target_dependencies(human_workspace_publisher
  ament_index_cpp
  rclcpp
  std_msgs
)

target_link_libraries(human_workspace_publisher
  ${HUMAN_WORKSPACE_MODEL_TARGET}
  ${cpp_typesupport_target}
)

ament_target_dependencies(human_reachable_set_visualizer
  rclcpp
  visualization_msgs
)

target_link_libraries(human_reachable_set_visualizer
  ${cpp_typesupport_target}
)

install(
  TARGETS ${HUMAN_WORKSPACE_MODEL_TARGET}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    human_reachable_set_visualizer
    human_workspace_publisher
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

ament_export_include_directories(include)
ament_export_libraries(${HUMAN_WORKSPACE_MODEL_TARGET})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(
  Eigen3
  geometry_msgs
  rclcpp
  rosidl_default_runtime
  std_msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_human_workspace
    test/test_human_workspace.cpp
  )
  target_link_libraries(test_human_workspace
    ${HUMAN_WORKSPACE_MODEL_TARGET}
    Eigen3::Eigen
  )

  ament_add_gtest(test_human_workspace_message
    test/test_human_workspace_message.cpp
  )
  target_link_libraries(test_human_workspace_message
    ${HUMAN_WORKSPACE_MODEL_TARGET}
    Eigen3::Eigen
  )
endif()

ament_package()
