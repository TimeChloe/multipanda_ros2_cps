#include "cps_trajectory_generators/reachable_cartesian_trajectory.hpp"

#include <algorithm>
#include <cmath>

namespace cps_trajectory_generators {

namespace {

constexpr double kSmoothStartDuration = 2.0;

}  // namespace

ReferenceTrajectoryType parseReferenceTrajectoryType(const std::string& name) {
  if (name == "lissajous") {
    return ReferenceTrajectoryType::kLissajous;
  }
  if (name == "constant") {
    return ReferenceTrajectoryType::kConstant;
  }
  return ReferenceTrajectoryType::kLine;
}

ReferenceTrajectoryType parseTrajectoryTypeOrDefault(
    bool use_constant_reference,
    const std::string& name) {
  if (use_constant_reference) {
    return ReferenceTrajectoryType::kConstant;
  }
  return parseReferenceTrajectoryType(name);
}

SmoothStartProfile makeSmoothStartProfile(double t, double T) {
  SmoothStartProfile out{};
  if (t <= 0.0) {
    return out;
  }
  if (t >= T) {
    out.s = 1.0;
    return out;
  }

  const double x = t / T;
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double x4 = x3 * x;
  const double x5 = x4 * x;

  out.s = 10.0 * x3 - 15.0 * x4 + 6.0 * x5;
  out.ds = (30.0 * x2 - 60.0 * x3 + 30.0 * x4) / T;
  out.dds = (60.0 * x - 180.0 * x2 + 120.0 * x3) / (T * T);
  return out;
}

double invertSmoothStartS(double s_target) {
  s_target = std::clamp(s_target, 0.0, 1.0);

  double lo = 0.0;
  double hi = 1.0;

  for (int i = 0; i < 40; ++i) {
    const double x = 0.5 * (lo + hi);
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x3 * x;
    const double x5 = x4 * x;
    const double s = 10.0 * x3 - 15.0 * x4 + 6.0 * x5;

    if (s < s_target) {
      lo = x;
    } else {
      hi = x;
    }
  }

  return 0.5 * (lo + hi);
}

double estimateLinePathTimeFromZ(double z,
                                 double z_start,
                                 double z_final,
                                 double T_move) {
  const double denom = z_final - z_start;

  if (std::abs(denom) < 1e-9) {
    return 0.0;
  }

  const double s = std::clamp((z - z_start) / denom, 0.0, 1.0);
  const double x = invertSmoothStartS(s);

  return std::clamp(x * T_move, 0.0, T_move);
}

TaskRefPose makeReferenceLine(double t,
                              const Eigen::Vector3d& p0,
                              const Eigen::Matrix3d& R0) {
  TaskRefPose ref;
  constexpr double T_move = 1.5;
  constexpr double dz = -0.55;
  const SmoothStartProfile ramp = makeSmoothStartProfile(t, T_move);
  ref.p = p0 + Eigen::Vector3d(0.0, 0.0, dz * ramp.s);
  ref.dp = Eigen::Vector3d(0.0, 0.0, dz * ramp.ds);
  ref.ddp = Eigen::Vector3d(0.0, 0.0, dz * ramp.dds);
  ref.R = R0;
  return ref;
}

TaskRefPose makeReferenceLissajous(double t,
                                   const Eigen::Vector3d& p0,
                                   const Eigen::Matrix3d& R0) {
  TaskRefPose ref;
  const double wt = 2.0 * M_PI * 0.25;
  const SmoothStartProfile ramp = makeSmoothStartProfile(t, kSmoothStartDuration);
  constexpr double Ax = 0.08;
  constexpr double Ay = 0.08;
  constexpr double Az = 0.04;
  const double ph_px = 0.0;
  const double ph_py = M_PI / 2.0;
  const double ph_pz = 0.0;

  const Eigen::Vector3d p_base(
      Ax * (std::sin(wt * t + ph_px) - std::sin(ph_px)),
      Ay * (std::sin(wt * t + ph_py) - std::sin(ph_py)),
      Az * (std::sin(0.5 * wt * t + ph_pz) - std::sin(ph_pz)));

  const Eigen::Vector3d dp_base(
      Ax * wt * std::cos(wt * t + ph_px),
      Ay * wt * std::cos(wt * t + ph_py),
      Az * 0.5 * wt * std::cos(0.5 * wt * t + ph_pz));

  const Eigen::Vector3d ddp_base(
      -Ax * wt * wt * std::sin(wt * t + ph_px),
      -Ay * wt * wt * std::sin(wt * t + ph_py),
      -Az * 0.25 * wt * wt * std::sin(0.5 * wt * t + ph_pz));

  ref.p = p0 + ramp.s * p_base;
  ref.dp = ramp.ds * p_base + ramp.s * dp_base;
  ref.ddp = ramp.dds * p_base + 2.0 * ramp.ds * dp_base + ramp.s * ddp_base;
  ref.R = R0;
  return ref;
}

TaskRefPose makeReferencePose(double t,
                              const Eigen::Vector3d& p0,
                              const Eigen::Matrix3d& R0,
                              ReferenceTrajectoryType traj_type) {
  switch (traj_type) {
    case ReferenceTrajectoryType::kLissajous:
      return makeReferenceLissajous(t, p0, R0);
    case ReferenceTrajectoryType::kConstant: {
      TaskRefPose ref;
      ref.p = p0;
      ref.dp.setZero();
      ref.ddp.setZero();
      ref.R = R0;
      return ref;
    }
    case ReferenceTrajectoryType::kLine:
    default:
      return makeReferenceLine(t, p0, R0);
  }
}

}  // namespace cps_trajectory_generators
