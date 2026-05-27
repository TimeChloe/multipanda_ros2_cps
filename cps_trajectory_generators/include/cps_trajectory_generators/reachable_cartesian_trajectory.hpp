#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace cps_trajectory_generators {

enum class ReferenceTrajectoryType {
  kLine,
  kLissajous,
  kConstant
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

}  // namespace cps_trajectory_generators
