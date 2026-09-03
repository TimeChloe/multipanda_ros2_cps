cmake_minimum_required(VERSION 3.8)
project(cps_safety_monitor)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cps_human_workspace REQUIRED)
find_package(yaml-cpp REQUIRED)

# Use the same ReachLib and RobotArmReach implementation as the SaRA-Shield
# reference checkout.  Only the geometry subset is compiled here; this avoids
# pulling SaRA-Shield's planner, Python bindings, and debug executables into the
# real-time controller package.
set(SARA_SHIELD_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../reference/sara-shield/safety_shield"
  CACHE PATH "Path to the SaRA-Shield safety_shield source directory")
set(SARA_REACHLIB_SOURCE_DIR
  "${SARA_SHIELD_SOURCE_DIR}/external/SaRA/ReachLib")
if(NOT EXISTS "${SARA_REACHLIB_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "SaRA ReachLib is missing at ${SARA_REACHLIB_SOURCE_DIR}. "
    "Initialize reference/sara-shield/safety_shield/external/SaRA first.")
endif()
if(NOT EXISTS "${SARA_SHIELD_SOURCE_DIR}/external/spdlog/include/spdlog/spdlog.h")
  message(FATAL_ERROR
    "SaRA-Shield's spdlog headers are missing. Initialize the spdlog submodule.")
endif()

set(BUILD_TESTS OFF CACHE BOOL "Disable nested ReachLib tests" FORCE)
add_subdirectory(
  "${SARA_REACHLIB_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/sara_reachlib"
  EXCLUDE_FROM_ALL)

add_library(${PROJECT_NAME} SHARED
  src/reachable_safety_monitor.cpp
  src/sara_robot_reachability.cpp
  "${SARA_SHIELD_SOURCE_DIR}/src/robot_arm_reach.cc"
  "${SARA_SHIELD_SOURCE_DIR}/src/robot_reach_utils.cc"
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    "${SARA_SHIELD_SOURCE_DIR}/include"
    "${SARA_SHIELD_SOURCE_DIR}/external/spdlog/include"
)

# Keep warnings enabled for our adapter while treating the unmodified upstream
# SaRA/ReachLib implementation as third-party code.
target_compile_options(SaRA PRIVATE -w)
set_source_files_properties(
  "${SARA_SHIELD_SOURCE_DIR}/src/robot_arm_reach.cc"
  "${SARA_SHIELD_SOURCE_DIR}/src/robot_reach_utils.cc"
  PROPERTIES COMPILE_OPTIONS -w)

target_compile_options(${PROJECT_NAME} PRIVATE
  "-include${CMAKE_CURRENT_SOURCE_DIR}/include/cps_safety_monitor/sara_eigen_compat.hpp")

ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  cps_human_workspace
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  SaRA
  yaml-cpp
)

# RobotArmReach is intentionally exposed only through our provider interface,
# but the exact ReachLib shared object still has to accompany the installed
# monitor library.  Keep both next to one another so the controller works from
# an installed ROS workspace as well as directly from the build tree.
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")

install(
  TARGETS ${PROJECT_NAME} SaRA
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
  FILES
    "${SARA_SHIELD_SOURCE_DIR}/config/robot_config/robot_parameters_panda.yaml"
  DESTINATION share/${PROJECT_NAME}/config
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(ament_index_cpp Eigen3 cps_human_workspace yaml-cpp)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_reachable_safety_monitor
    test/test_reachable_safety_monitor.cpp
  )
  target_link_libraries(test_reachable_safety_monitor
    ${PROJECT_NAME}
    Eigen3::Eigen
    SaRA
  )
  target_include_directories(test_reachable_safety_monitor SYSTEM PRIVATE
    "${SARA_REACHLIB_SOURCE_DIR}/include"
  )
  target_compile_definitions(test_reachable_safety_monitor PRIVATE
    "SARA_PANDA_CONFIG_PATH=\"${SARA_SHIELD_SOURCE_DIR}/config/robot_config/robot_parameters_panda.yaml\"")
endif()

ament_package()
