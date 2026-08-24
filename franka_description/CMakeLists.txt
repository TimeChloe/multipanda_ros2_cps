cmake_minimum_required(VERSION 3.5)
project(franka_description)
find_package(ament_cmake REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(${PROJECT_NAME}_urdf_tests test/urdf_tests.py)

endif()

# Install launch files.
install(DIRECTORY
  launch robots meshes rviz mujoco
  DESTINATION share/${PROJECT_NAME}/
)

# Pinocchio monitor base generated from the same calibrated Panda inertial
# parameters as the MuJoCo panda_ng model. End-effectors are attached at launch.
install(
  FILES ../model_urdf/panda_ng.urdf
  DESTINATION share/${PROJECT_NAME}/model_urdf
  RENAME panda_ng_monitor_base.urdf
)

ament_package()
