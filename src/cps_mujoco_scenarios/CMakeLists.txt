cmake_minimum_required(VERSION 3.8)
project(cps_mujoco_scenarios)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cps_human_workspace REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mujoco_ros_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveTableAssembly.action"
)

add_executable(table_assembly_action_server
  src/table_assembly_action_server.cpp
)

ament_target_dependencies(table_assembly_action_server
  geometry_msgs
  mujoco_ros_msgs
  rclcpp
  rclcpp_action
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)
target_link_libraries(table_assembly_action_server
  ${cpp_typesupport_target}
)

add_executable(surface_following_human_workspace
  src/surface_following_human_workspace_node.cpp
)

ament_target_dependencies(surface_following_human_workspace
  ament_index_cpp
  mujoco_ros_msgs
  rclcpp
)

target_link_libraries(surface_following_human_workspace
  cps_human_workspace::cps_human_workspace
)

install(
  TARGETS
    table_assembly_action_server
    surface_following_human_workspace
  RUNTIME DESTINATION lib/${PROJECT_NAME}
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

install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
