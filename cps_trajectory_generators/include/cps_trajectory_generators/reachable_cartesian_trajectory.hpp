#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace cps_trajectory_generators {

enum class ReferenceTrajectoryType {
  kLine,
  kLissajous,
  kConstant,
  kViaPointsSmooth
};

struct SmoothStartProfile {
  double s{0.0};
  double ds{0.0};
  double dds{0.0};
};

struct TaskRefPose {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dp{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ddp{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
};

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

  double failsafe_brake_max_velocity{0.8};
  double failsafe_brake_max_acceleration{2.0};
  double failsafe_brake_max_jerk{80.0};

  double path_retiming_search_window_sec{0.25};
  int path_retiming_search_steps{41};
  double path_time_rate_min{0.0};
  double path_time_rate_max{1.0};
  double path_time_acc_limit{0.5};
  double path_time_rate_target{1.0};
};

std::string defaultTrajectoryGeneratorConfigPath();

TrajectoryGeneratorSettings loadTrajectoryGeneratorSettings(
    const std::string& config_path = "");

ReferenceTrajectoryType parseReferenceTrajectoryType(const std::string& name);

ReferenceTrajectoryType parseTrajectoryTypeOrDefault(
    bool use_constant_reference,
    const std::string& name);

SmoothStartProfile makeSmoothStartProfile(double t, double T);

double invertSmoothStartS(double s_target);

double estimateLinePathTimeFromZ(double z,
                                 double z_start,
                                 double z_final,
                                 double T_move);

TaskRefPose makeReferenceLine(double t,
                              const Eigen::Vector3d& p0,
                              const Eigen::Matrix3d& R0);

TaskRefPose makeReferenceLissajous(double t,
                                   const Eigen::Vector3d& p0,
                                   const Eigen::Matrix3d& R0);

TaskRefPose makeReferencePose(double t,
                              const Eigen::Vector3d& p0,
                              const Eigen::Matrix3d& R0,
                              ReferenceTrajectoryType traj_type);

std::vector<CartesianTrajectorySample> makeLocalCartesianReplan(
    double nominal_guess_time,
    const CartesianTrajectorySample& planning_start,
    const Eigen::Vector3d& reference_position,
    const Eigen::Quaterniond& reference_orientation,
    ReferenceTrajectoryType traj_type,
    const LocalCartesianReplanConfig& config);

std::vector<CartesianTrajectorySample> makeSmoothViaPointCartesianTrajectory(
    const std::vector<Eigen::Vector3d>& waypoints,
    const Eigen::Quaterniond& reference_orientation,
    const LocalCartesianReplanConfig& config);

std::vector<CartesianTrajectorySample> makeLocalCartesianReplanFromTimedPath(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const LocalCartesianReplanConfig& config);

std::vector<CartesianTrajectorySample> makeCartesianBrakeTrajectory(
    const CartesianTrajectorySample& brake_start,
    const LocalCartesianReplanConfig& config);

}  // namespace cps_trajectory_generators
