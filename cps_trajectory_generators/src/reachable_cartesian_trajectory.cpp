#include "cps_trajectory_generators/reachable_cartesian_trajectory.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <ruckig/ruckig.hpp>

namespace cps_trajectory_generators {

namespace {

constexpr double kMinDt = 1e-6;
constexpr double kSmallPositive = 1e-9;
constexpr double kSmoothStartDuration = 2.0;
constexpr double kLineMoveDuration = 1.5;
constexpr double kLineDeltaZ = -0.55;

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
  const SmoothStartProfile ramp = makeSmoothStartProfile(t, kLineMoveDuration);
  ref.p = p0 + Eigen::Vector3d(0.0, 0.0, kLineDeltaZ * ramp.s);
  ref.dp = Eigen::Vector3d(0.0, 0.0, kLineDeltaZ * ramp.ds);
  ref.ddp = Eigen::Vector3d(0.0, 0.0, kLineDeltaZ * ramp.dds);
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

std::vector<CartesianTrajectorySample> makeLocalCartesianReplan(
    double nominal_guess_time,
    const CartesianTrajectorySample& planning_start,
    const Eigen::Vector3d& reference_position,
    const Eigen::Quaterniond& reference_orientation,
    ReferenceTrajectoryType traj_type,
    const LocalCartesianReplanConfig& config) {
  std::vector<CartesianTrajectorySample> samples;

  const double dt = std::max(config.dt, kMinDt);
  const int horizon_steps = std::max(1, config.horizon_steps);
  const double lookahead_sec = std::max(config.path_lookahead_sec, dt);
  const double max_velocity = std::max(config.max_velocity, 1e-4);
  const double max_acceleration = std::max(config.max_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_jerk, 1e-4);

  Eigen::Quaterniond reference_q = reference_orientation;
  reference_q.normalize();
  const Eigen::Matrix3d reference_R = reference_q.toRotationMatrix();

  if (traj_type == ReferenceTrajectoryType::kLine) {
    const double z_start = reference_position.z();
    const double z_final = reference_position.z() + kLineDeltaZ;

    const double q0_raw = planning_start.p.z();
    const double q0 = std::clamp(q0_raw, z_final, z_start);

    const double t_near =
        estimateLinePathTimeFromZ(q0, z_start, z_final, kLineMoveDuration);

    double t_goal = t_near + lookahead_sec;
    t_goal = std::clamp(t_goal, 0.0, kLineMoveDuration);

    const bool near_final_position = std::abs(q0 - z_final) < 1e-3;
    const bool near_zero_velocity = std::abs(planning_start.dp.z()) < 1e-3;

    samples.reserve(static_cast<std::size_t>(horizon_steps));

    if (near_final_position && near_zero_velocity) {
      for (int i = 0; i < horizon_steps; ++i) {
        CartesianTrajectorySample s;
        s.t = static_cast<double>(i + 1) * dt;
        s.p = Eigen::Vector3d(reference_position.x(), reference_position.y(), z_final);
        s.q = reference_q;
        samples.push_back(s);
      }
      return samples;
    }

    const TaskRefPose goal_ref = makeReferenceLine(t_goal, reference_position, reference_R);

    const double v0 = std::min(planning_start.dp.z(), 0.0);
    const double a0 = planning_start.ddp.z();
    const double q1 = std::clamp(goal_ref.p.z(), z_final, q0);

    ruckig::Ruckig<1> otg;
    ruckig::InputParameter<1> input;
    ruckig::Trajectory<1> trajectory;

    input.current_position = {q0};
    input.current_velocity = {v0};
    input.current_acceleration = {a0};

    input.target_position = {q1};
    input.target_velocity = {0.0};
    input.target_acceleration = {0.0};

    input.max_velocity = {max_velocity};
    input.max_acceleration = {max_acceleration};
    input.max_jerk = {max_jerk};

    const auto result = otg.calculate(input, trajectory);
    if (result < ruckig::Result::Working) {
      return {};
    }

    const double duration = std::max(trajectory.get_duration(), dt);
    const int n_steps =
        std::min(horizon_steps, std::max(1, static_cast<int>(std::ceil(duration / dt))));

    samples.clear();
    samples.reserve(static_cast<std::size_t>(n_steps));

    double previous_z = q0;

    for (int i = 0; i < n_steps; ++i) {
      const double tau = std::min(static_cast<double>(i + 1) * dt, duration);

      std::array<double, 1> p{}, v{}, a{};
      trajectory.at_time(tau, p, v, a);

      double z_cmd = p[0];
      double dz_cmd = v[0];
      double ddz_cmd = a[0];

      z_cmd = std::min(z_cmd, previous_z);
      z_cmd = std::max(z_cmd, z_final);
      dz_cmd = std::min(dz_cmd, 0.0);

      if (z_cmd <= z_final + 1e-6) {
        z_cmd = z_final;
        dz_cmd = 0.0;
        ddz_cmd = 0.0;
      }

      CartesianTrajectorySample s;
      s.t = tau;
      s.p = Eigen::Vector3d(reference_position.x(), reference_position.y(), z_cmd);
      s.dp = Eigen::Vector3d(0.0, 0.0, dz_cmd);
      s.ddp = Eigen::Vector3d(0.0, 0.0, ddz_cmd);
      s.q = reference_q;
      samples.push_back(s);

      previous_z = z_cmd;
    }

    return samples;
  }

  const double t_near = std::max(0.0, nominal_guess_time);
  const double t_goal = t_near + lookahead_sec;

  const TaskRefPose goal_ref =
      makeReferencePose(t_goal, reference_position, reference_R, traj_type);

  Eigen::Vector3d target_velocity = goal_ref.dp;
  const double v_norm = target_velocity.norm();
  if (v_norm > max_velocity) {
    target_velocity *= max_velocity / std::max(v_norm, kSmallPositive);
  }

  ruckig::Ruckig<3> otg;
  ruckig::InputParameter<3> input;
  ruckig::Trajectory<3> trajectory;

  input.current_position = {
      planning_start.p.x(),
      planning_start.p.y(),
      planning_start.p.z()};

  input.current_velocity = {
      planning_start.dp.x(),
      planning_start.dp.y(),
      planning_start.dp.z()};

  input.current_acceleration = {
      planning_start.ddp.x(),
      planning_start.ddp.y(),
      planning_start.ddp.z()};

  input.target_position = {
      goal_ref.p.x(),
      goal_ref.p.y(),
      goal_ref.p.z()};

  input.target_velocity = {
      target_velocity.x(),
      target_velocity.y(),
      target_velocity.z()};

  input.target_acceleration = {0.0, 0.0, 0.0};

  input.max_velocity = {max_velocity, max_velocity, max_velocity};
  input.max_acceleration = {max_acceleration, max_acceleration, max_acceleration};
  input.max_jerk = {max_jerk, max_jerk, max_jerk};

  const auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    return {};
  }

  const double duration = std::max(trajectory.get_duration(), dt);
  const int n_steps =
      std::min(horizon_steps, std::max(1, static_cast<int>(std::ceil(duration / dt))));

  samples.reserve(static_cast<std::size_t>(n_steps));

  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);

    std::array<double, 3> p{}, v{}, a{};
    trajectory.at_time(tau, p, v, a);

    CartesianTrajectorySample s;
    s.t = tau;
    s.p = Eigen::Vector3d(p[0], p[1], p[2]);
    s.dp = Eigen::Vector3d(v[0], v[1], v[2]);
    s.ddp = Eigen::Vector3d(a[0], a[1], a[2]);
    s.q = Eigen::Quaterniond(goal_ref.R);
    s.q.normalize();
    samples.push_back(s);
  }

  return samples;
}

std::vector<CartesianTrajectorySample> makeCartesianBrakeTrajectory(
    const CartesianTrajectorySample& brake_start,
    const LocalCartesianReplanConfig& config) {
  std::vector<CartesianTrajectorySample> samples;

  const double dt = std::max(config.dt, kMinDt);
  const double max_velocity = std::max(config.max_velocity, 1e-4);
  const double max_acceleration = std::max(config.max_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_jerk, 1e-4);

  ruckig::Ruckig<3> otg;
  ruckig::InputParameter<3> input;
  ruckig::Trajectory<3> trajectory;

  input.control_interface = ruckig::ControlInterface::Velocity;

  input.current_position = {
      brake_start.p.x(),
      brake_start.p.y(),
      brake_start.p.z()};

  input.current_velocity = {
      brake_start.dp.x(),
      brake_start.dp.y(),
      brake_start.dp.z()};

  input.current_acceleration = {
      brake_start.ddp.x(),
      brake_start.ddp.y(),
      brake_start.ddp.z()};

  input.target_velocity = {0.0, 0.0, 0.0};
  input.target_acceleration = {0.0, 0.0, 0.0};

  input.max_velocity = {max_velocity, max_velocity, max_velocity};
  input.max_acceleration = {max_acceleration, max_acceleration, max_acceleration};
  input.max_jerk = {max_jerk, max_jerk, max_jerk};

  const auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    return {};
  }

  const double duration = std::max(trajectory.get_duration(), dt);
  const int n_steps =
      std::max(1, static_cast<int>(std::ceil(duration / dt)));

  samples.reserve(static_cast<std::size_t>(n_steps));

  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);

    std::array<double, 3> p{}, v{}, a{};
    trajectory.at_time(tau, p, v, a);

    CartesianTrajectorySample s;
    s.t = static_cast<double>(i + 1) * dt;
    s.p = Eigen::Vector3d(p[0], p[1], p[2]);
    s.dp = Eigen::Vector3d(v[0], v[1], v[2]);
    s.ddp = Eigen::Vector3d(a[0], a[1], a[2]);
    s.q = brake_start.q;
    s.q.normalize();
    if (i + 1 == n_steps) {
      s.dp.setZero();
      s.ddp.setZero();
    }
    samples.push_back(s);
  }

  return samples;
}

}  // namespace cps_trajectory_generators
