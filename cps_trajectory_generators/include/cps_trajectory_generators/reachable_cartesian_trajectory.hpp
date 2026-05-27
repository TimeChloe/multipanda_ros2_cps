#pragma once

#include <string>

#include <Eigen/Dense>

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

}  // namespace cps_trajectory_generators
