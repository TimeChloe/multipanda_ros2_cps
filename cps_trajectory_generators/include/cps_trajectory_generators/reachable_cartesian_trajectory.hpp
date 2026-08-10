#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace cps_trajectory_generators {

struct CartesianTrajectorySample {
  double t{0.0};
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dp{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ddp{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d w{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dw{Eigen::Vector3d::Zero()};
};

struct LocalCartesianReplanConfig {
  int horizon_steps{200};
  double dt{0.001};
  double path_lookahead_sec{0.08};
  double max_velocity{0.08};
  double max_acceleration{0.4};
  double max_jerk{2.0};
  double max_angular_velocity{0.8};
  double max_angular_acceleration{4.0};
  double max_angular_jerk{40.0};
};

struct PathConsistentTimedPathConfig {
  int intended_steps{1};
  double dt{0.001};
  double path_lookahead_sec{0.08};
  // Reprojection is useful when reconnecting an arbitrary Cartesian state.
  // It must be disabled when the caller already owns an exact verified scalar
  // path state, otherwise nearby/crossing path branches can be skipped.
  bool project_start_to_nearest_path_state{true};
  double max_path_rate{1.0};
  double max_path_acceleration{0.5};
  double max_path_jerk{5.0};
  double target_path_rate{1.0};
  double initial_path_rate{-1.0};
};

struct TrajectoryGeneratorSettings {
  double shield_plan_dt{0.001};
  double monitor_frequency_hz{50.0};

  int local_replan_horizon_steps{500};
  double local_replan_dt{0.001};
  double local_path_lookahead_sec{0.50};
  double local_replan_max_velocity{0.5};
  double local_replan_max_acceleration{1.5};
  double local_replan_max_jerk{5.0};
  double local_replan_max_angular_velocity{0.8};
  double local_replan_max_angular_acceleration{4.0};
  double local_replan_max_angular_jerk{40.0};

  double failsafe_brake_max_velocity{0.8};
  double failsafe_brake_max_acceleration{2.0};
  double failsafe_brake_max_jerk{80.0};
  double failsafe_brake_max_angular_velocity{1.5};
  double failsafe_brake_max_angular_acceleration{10.0};
  double failsafe_brake_max_angular_jerk{500.0};

  double path_time_rate_min{0.0};
  double path_time_rate_max{1.0};
  double path_time_acc_limit{0.5};
  double path_time_rate_target{1.0};
};

std::string defaultTrajectoryGeneratorConfigPath();

TrajectoryGeneratorSettings loadTrajectoryGeneratorSettings(
    const std::string& config_path = "");

std::vector<CartesianTrajectorySample> makeSmoothViaPointCartesianTrajectory(
    const std::vector<Eigen::Vector3d>& waypoints,
    const Eigen::Quaterniond& reference_orientation,
    const LocalCartesianReplanConfig& config);

std::vector<CartesianTrajectorySample> makeSmoothViaPointCartesianTrajectory(
    const std::vector<Eigen::Vector3d>& waypoints,
    const std::vector<Eigen::Quaterniond>& waypoint_orientations,
    const LocalCartesianReplanConfig& config);

std::vector<CartesianTrajectorySample> makeLocalCartesianReplanFromTimedPath(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const LocalCartesianReplanConfig& config);

CartesianTrajectorySample makeRetimedPathState(
    const std::vector<CartesianTrajectorySample>& timed_path,
    double path_time,
    double path_rate,
    double path_acceleration);

std::vector<CartesianTrajectorySample> makePathConsistentTimedPathReplan(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const PathConsistentTimedPathConfig& config);

std::vector<CartesianTrajectorySample> makePathConsistentTimedPathIntendedPrefix(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const PathConsistentTimedPathConfig& config);

std::vector<CartesianTrajectorySample> makePathConsistentTimedPathBrake(
    double path_time,
    const CartesianTrajectorySample& brake_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const PathConsistentTimedPathConfig& config);

std::vector<CartesianTrajectorySample> makeCartesianBrakeTrajectory(
    const CartesianTrajectorySample& brake_start,
    const LocalCartesianReplanConfig& config);

}  // namespace cps_trajectory_generators
