#pragma once

#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace cps_trajectory_generators {

struct CartesianTrajectorySample {
  double t{0.0};
  // Scalar retiming state for samples produced from an existing timed path.
  // This remains well-defined at a Cartesian cusp where dp and w are zero.
  double path_rate{0.0};
  double path_acceleration{0.0};
  bool path_kinematics_valid{false};
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dp{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ddp{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d w{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dw{Eigen::Vector3d::Zero()};
};

struct LocalCartesianReplanConfig {
  double dt{0.001};
  // Consecutive intermediate waypoints inside both tolerances are merged.
  // The final target is always retained.
  double waypoint_merge_position_tolerance{0.001};
  double waypoint_merge_orientation_tolerance{0.005};
  // Euclidean-norm limits for the 3D linear and angular vectors.
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
  double max_path_rate{1.0};
  double max_path_acceleration{0.5};
  double max_path_jerk{5.0};
  double target_path_rate{1.0};
  double initial_path_rate{-1.0};
  // A finite value explicitly reanchors Ruckig to the measured scalar path
  // acceleration. NaN preserves the legacy estimate from Cartesian command
  // derivatives.
  double initial_path_acceleration{
      std::numeric_limits<double>::quiet_NaN()};
};

struct TrajectoryGeneratorSettings {
  // SaRA-Shield Panda: 5 controller samples per reachability interval.
  double shield_plan_dt{0.005};
  double monitor_frequency_hz{200.0};

  int local_replan_horizon_steps{64};
  double local_replan_dt{0.001};
  double waypoint_merge_position_tolerance{0.001};
  double waypoint_merge_orientation_tolerance{0.005};
  double local_replan_max_velocity{0.5};
  double local_replan_max_acceleration{1.5};
  double local_replan_max_jerk{5.0};
  double local_replan_max_angular_velocity{0.8};
  double local_replan_max_angular_acceleration{4.0};
  double local_replan_max_angular_jerk{40.0};

  double failsafe_brake_max_velocity{0.85};
  double failsafe_brake_max_acceleration{6.5};
  double failsafe_brake_max_jerk{3250.0};
  double failsafe_brake_max_angular_velocity{1.25};
  double failsafe_brake_max_angular_acceleration{12.5};
  double failsafe_brake_max_angular_jerk{6250.0};

  double path_time_rate_min{0.0};
  double path_time_rate_max{1.0};
  double path_time_acc_limit{0.5};
  double path_time_jerk_limit{5.0};
  double path_time_rate_target{1.0};
  double failsafe_path_time_acc_limit{10.0};
  double failsafe_path_time_jerk_limit{5000.0};
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

CartesianTrajectorySample makeRetimedPathState(
    const std::vector<CartesianTrajectorySample>& timed_path,
    double path_time,
    double path_rate,
    double path_acceleration);

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

}  // namespace cps_trajectory_generators
