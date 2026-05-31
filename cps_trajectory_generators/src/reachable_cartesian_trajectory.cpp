#include "cps_trajectory_generators/reachable_cartesian_trajectory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ruckig/ruckig.hpp>

namespace cps_trajectory_generators {

namespace {

constexpr double kMinDt = 1e-6;
constexpr double kSmallPositive = 1e-9;
constexpr double kSmoothStartDuration = 2.0;
constexpr double kLineMoveDuration = 1.5;
constexpr double kLineDeltaZ = -0.55;
constexpr const char* kTrajectoryConfigFileName =
    "reachable_cartesian_trajectory.yaml";

std::string trimCopy(const std::string& input) {
  const auto begin = input.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = input.find_last_not_of(" \t\r\n");
  return input.substr(begin, end - begin + 1);
}

bool parseDouble(const std::string& text, double* value) {
  if (value == nullptr) {
    return false;
  }
  std::istringstream stream(text);
  double parsed = 0.0;
  stream >> parsed;
  if (!stream) {
    return false;
  }
  *value = parsed;
  return true;
}

bool parseInt(const std::string& text, int* value) {
  if (value == nullptr) {
    return false;
  }
  std::istringstream stream(text);
  int parsed = 0;
  stream >> parsed;
  if (!stream) {
    return false;
  }
  *value = parsed;
  return true;
}

}  // namespace

std::string defaultTrajectoryGeneratorConfigPath() {
  try {
    return ament_index_cpp::get_package_share_directory(
               "cps_trajectory_generators") +
           "/config/" + kTrajectoryConfigFileName;
  } catch (const std::exception&) {
    return "cps_trajectory_generators/config/" +
           std::string(kTrajectoryConfigFileName);
  }
}

TrajectoryGeneratorSettings loadTrajectoryGeneratorSettings(
    const std::string& config_path) {
  TrajectoryGeneratorSettings settings;
  const std::string path =
      config_path.empty() ? defaultTrajectoryGeneratorConfigPath() : config_path;

  std::ifstream file(path);
  if (!file.is_open()) {
    return settings;
  }

  std::string line;
  while (std::getline(file, line)) {
    const auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    const auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }
    const std::string key = trimCopy(line.substr(0, colon_pos));
    const std::string value_text = trimCopy(line.substr(colon_pos + 1));
    if (key.empty() || value_text.empty()) {
      continue;
    }

    double double_value = 0.0;
    int int_value = 0;

    if (key == "shield_plan_dt" && parseDouble(value_text, &double_value)) {
      settings.shield_plan_dt = double_value;
    } else if (key == "monitor_frequency_hz" &&
               parseDouble(value_text, &double_value)) {
      settings.monitor_frequency_hz = double_value;
    } else if (key == "local_replan_horizon_steps" &&
               parseInt(value_text, &int_value)) {
      settings.local_replan_horizon_steps = int_value;
    } else if (key == "local_replan_dt" &&
               parseDouble(value_text, &double_value)) {
      settings.local_replan_dt = double_value;
    } else if (key == "local_path_lookahead_sec" &&
               parseDouble(value_text, &double_value)) {
      settings.local_path_lookahead_sec = double_value;
    } else if (key == "local_replan_max_velocity" &&
               parseDouble(value_text, &double_value)) {
      settings.local_replan_max_velocity = double_value;
    } else if (key == "local_replan_max_acceleration" &&
               parseDouble(value_text, &double_value)) {
      settings.local_replan_max_acceleration = double_value;
    } else if (key == "local_replan_max_jerk" &&
               parseDouble(value_text, &double_value)) {
      settings.local_replan_max_jerk = double_value;
    } else if (key == "failsafe_brake_max_velocity" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_velocity = double_value;
    } else if (key == "failsafe_brake_max_acceleration" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_acceleration = double_value;
    } else if (key == "failsafe_brake_max_jerk" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_jerk = double_value;
    } else if (key == "path_retiming_search_window_sec" &&
               parseDouble(value_text, &double_value)) {
      settings.path_retiming_search_window_sec = double_value;
    } else if (key == "path_retiming_search_steps" &&
               parseInt(value_text, &int_value)) {
      settings.path_retiming_search_steps = int_value;
    } else if (key == "path_time_rate_min" &&
               parseDouble(value_text, &double_value)) {
      settings.path_time_rate_min = double_value;
    } else if (key == "path_time_rate_max" &&
               parseDouble(value_text, &double_value)) {
      settings.path_time_rate_max = double_value;
    } else if (key == "path_time_acc_limit" &&
               parseDouble(value_text, &double_value)) {
      settings.path_time_acc_limit = double_value;
    } else if (key == "path_time_rate_target" &&
               parseDouble(value_text, &double_value)) {
      settings.path_time_rate_target = double_value;
    }
  }

  return settings;
}

ReferenceTrajectoryType parseReferenceTrajectoryType(const std::string& name) {
  if (name == "lissajous") {
    return ReferenceTrajectoryType::kLissajous;
  }
  if (name == "constant") {
    return ReferenceTrajectoryType::kConstant;
  }
  if (name == "via_points_smooth" || name == "smooth_via_points" ||
      name == "smooth_waypoints" || name == "via_points" ||
      name == "viapoints" || name == "waypoints") {
    return ReferenceTrajectoryType::kViaPointsSmooth;
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
    case ReferenceTrajectoryType::kViaPointsSmooth:
    case ReferenceTrajectoryType::kLine:
    default:
      return makeReferenceLine(t, p0, R0);
  }
}

namespace {

CartesianTrajectorySample sampleTimedPathAt(
    const std::vector<CartesianTrajectorySample>& path,
    double t) {
  if (path.empty()) {
    return CartesianTrajectorySample{};
  }
  if (t <= path.front().t) {
    return path.front();
  }
  if (t >= path.back().t) {
    return path.back();
  }

  auto upper = std::lower_bound(
      path.begin(),
      path.end(),
      t,
      [](const CartesianTrajectorySample& sample, double value) {
        return sample.t < value;
      });

  if (upper == path.begin()) {
    return *upper;
  }
  const auto lower = upper - 1;
  const double span = std::max(upper->t - lower->t, kMinDt);
  const double alpha = std::clamp((t - lower->t) / span, 0.0, 1.0);

  CartesianTrajectorySample out;
  out.t = t;
  out.p = (1.0 - alpha) * lower->p + alpha * upper->p;
  out.dp = (1.0 - alpha) * lower->dp + alpha * upper->dp;
  out.ddp = (1.0 - alpha) * lower->ddp + alpha * upper->ddp;
  out.q = lower->q.slerp(alpha, upper->q);
  out.q.normalize();
  out.w = (1.0 - alpha) * lower->w + alpha * upper->w;
  out.dw = (1.0 - alpha) * lower->dw + alpha * upper->dw;
  return out;
}

struct SmoothPathSample {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dp_ds{Eigen::Vector3d::Zero()};
  Eigen::Vector3d d2p_ds2{Eigen::Vector3d::Zero()};
};

SmoothPathSample sampleCubicHermitePath(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<double>& u,
    const std::vector<Eigen::Vector3d>& tangents,
    double s) {
  SmoothPathSample out;
  if (points.empty()) {
    return out;
  }
  if (points.size() == 1 || s <= u.front()) {
    out.p = points.front();
    out.dp_ds = tangents.empty() ? Eigen::Vector3d::Zero() : tangents.front();
    return out;
  }
  if (s >= u.back()) {
    out.p = points.back();
    out.dp_ds = tangents.back();
    return out;
  }

  auto upper = std::lower_bound(u.begin(), u.end(), s);
  std::size_t i = static_cast<std::size_t>(std::distance(u.begin(), upper));
  if (i == 0) {
    i = 1;
  }
  const std::size_t i0 = i - 1;
  const std::size_t i1 = i;
  const double h = std::max(u[i1] - u[i0], kMinDt);
  const double x = std::clamp((s - u[i0]) / h, 0.0, 1.0);
  const double x2 = x * x;
  const double x3 = x2 * x;

  const double h00 = 2.0 * x3 - 3.0 * x2 + 1.0;
  const double h10 = x3 - 2.0 * x2 + x;
  const double h01 = -2.0 * x3 + 3.0 * x2;
  const double h11 = x3 - x2;

  const double dh00 = 6.0 * x2 - 6.0 * x;
  const double dh10 = 3.0 * x2 - 4.0 * x + 1.0;
  const double dh01 = -6.0 * x2 + 6.0 * x;
  const double dh11 = 3.0 * x2 - 2.0 * x;

  const double d2h00 = 12.0 * x - 6.0;
  const double d2h10 = 6.0 * x - 4.0;
  const double d2h01 = -12.0 * x + 6.0;
  const double d2h11 = 6.0 * x - 2.0;

  const Eigen::Vector3d& p0 = points[i0];
  const Eigen::Vector3d& p1 = points[i1];
  const Eigen::Vector3d& m0 = tangents[i0];
  const Eigen::Vector3d& m1 = tangents[i1];

  out.p = h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1;
  out.dp_ds =
      (dh00 * p0 + dh10 * h * m0 + dh01 * p1 + dh11 * h * m1) / h;
  out.d2p_ds2 =
      (d2h00 * p0 + d2h10 * h * m0 + d2h01 * p1 + d2h11 * h * m1) /
      (h * h);
  return out;
}

}  // namespace

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

std::vector<CartesianTrajectorySample> makeSmoothViaPointCartesianTrajectory(
    const std::vector<Eigen::Vector3d>& waypoints,
    const Eigen::Quaterniond& reference_orientation,
    const LocalCartesianReplanConfig& config) {
  std::vector<CartesianTrajectorySample> samples;
  if (waypoints.empty()) {
    return samples;
  }

  const double dt = std::max(config.dt, kMinDt);
  const double max_velocity = std::max(config.max_velocity, 1e-4);
  const double max_acceleration = std::max(config.max_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_jerk, 1e-4);

  Eigen::Quaterniond q_ref = reference_orientation;
  q_ref.normalize();

  std::vector<Eigen::Vector3d> points;
  points.reserve(waypoints.size());
  for (const auto& p : waypoints) {
    if (points.empty() || (p - points.back()).norm() > 1e-9) {
      points.push_back(p);
    }
  }

  if (points.empty()) {
    return samples;
  }
  if (points.size() == 1) {
    CartesianTrajectorySample s;
    s.t = 0.0;
    s.p = points.front();
    s.q = q_ref;
    samples.push_back(s);
    return samples;
  }

  std::vector<double> u(points.size(), 0.0);
  for (std::size_t i = 1; i < points.size(); ++i) {
    u[i] = u[i - 1] + std::max((points[i] - points[i - 1]).norm(), kMinDt);
  }
  const double path_length = std::max(u.back(), kMinDt);

  std::vector<Eigen::Vector3d> tangents(points.size(), Eigen::Vector3d::Zero());
  tangents.front() = (points[1] - points[0]) / std::max(u[1] - u[0], kMinDt);
  tangents.back() =
      (points.back() - points[points.size() - 2]) /
      std::max(u.back() - u[u.size() - 2], kMinDt);
  for (std::size_t i = 1; i + 1 < points.size(); ++i) {
    tangents[i] =
        (points[i + 1] - points[i - 1]) /
        std::max(u[i + 1] - u[i - 1], kMinDt);
  }

  ruckig::Ruckig<1> otg;
  ruckig::InputParameter<1> input;
  ruckig::Trajectory<1> trajectory;

  input.current_position = {0.0};
  input.current_velocity = {0.0};
  input.current_acceleration = {0.0};
  input.target_position = {path_length};
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
      std::max(1, static_cast<int>(std::ceil(duration / dt)));
  samples.reserve(static_cast<std::size_t>(n_steps) + 1);

  CartesianTrajectorySample first;
  first.t = 0.0;
  first.p = points.front();
  first.q = q_ref;
  samples.push_back(first);

  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
    std::array<double, 1> s_arr{}, ds_arr{}, dds_arr{};
    trajectory.at_time(tau, s_arr, ds_arr, dds_arr);

    const double s_path = std::clamp(s_arr[0], 0.0, path_length);
    const SmoothPathSample path_sample =
        sampleCubicHermitePath(points, u, tangents, s_path);

    CartesianTrajectorySample sample;
    sample.t = tau;
    sample.p = path_sample.p;
    sample.dp = path_sample.dp_ds * ds_arr[0];
    sample.ddp =
        path_sample.d2p_ds2 * ds_arr[0] * ds_arr[0] +
        path_sample.dp_ds * dds_arr[0];
    sample.q = q_ref;
    if (i + 1 == n_steps) {
      sample.p = points.back();
      sample.dp.setZero();
      sample.ddp.setZero();
    }
    samples.push_back(sample);
  }

  return samples;
}

std::vector<CartesianTrajectorySample> makeLocalCartesianReplanFromTimedPath(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const LocalCartesianReplanConfig& config) {
  std::vector<CartesianTrajectorySample> samples;
  if (timed_path.empty()) {
    return samples;
  }

  const double dt = std::max(config.dt, kMinDt);
  const int horizon_steps = std::max(1, config.horizon_steps);
  const double lookahead_sec = std::max(config.path_lookahead_sec, dt);
  const double max_velocity = std::max(config.max_velocity, 1e-4);
  const double max_acceleration = std::max(config.max_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_jerk, 1e-4);

  const double lower_time = std::clamp(
      min_path_time,
      timed_path.front().t,
      timed_path.back().t);

  auto lower_it = std::lower_bound(
      timed_path.begin(),
      timed_path.end(),
      lower_time,
      [](const CartesianTrajectorySample& sample, double value) {
        return sample.t < value;
      });

  if (lower_it == timed_path.end()) {
    lower_it = timed_path.end() - 1;
  }

  const double search_end_time =
      std::min(lower_time + lookahead_sec, timed_path.back().t);
  auto search_end_it = std::upper_bound(
      lower_it,
      timed_path.end(),
      search_end_time,
      [](double value, const CartesianTrajectorySample& sample) {
        return value < sample.t;
      });
  if (search_end_it == lower_it) {
    search_end_it = std::next(lower_it);
  }

  std::size_t nearest_idx =
      static_cast<std::size_t>(std::distance(timed_path.begin(), lower_it));
  double nearest_dist = (timed_path[nearest_idx].p - planning_start.p).squaredNorm();

  const std::size_t search_end_idx =
      static_cast<std::size_t>(std::distance(timed_path.begin(), search_end_it));
  for (std::size_t i = nearest_idx + 1; i < search_end_idx; ++i) {
    const double d = (timed_path[i].p - planning_start.p).squaredNorm();
    if (d < nearest_dist) {
      nearest_idx = i;
      nearest_dist = d;
    }
  }

  const double start_path_time =
      std::max(lower_time, timed_path[nearest_idx].t);
  const double target_path_time =
      std::min(start_path_time + lookahead_sec, timed_path.back().t);
  const CartesianTrajectorySample target =
      sampleTimedPathAt(timed_path, target_path_time);

  const bool at_end =
      target_path_time >= timed_path.back().t - dt &&
      (planning_start.p - timed_path.back().p).norm() < 1e-3 &&
      planning_start.dp.norm() < 1e-3;

  if (at_end) {
    samples.reserve(static_cast<std::size_t>(horizon_steps));
    for (int i = 0; i < horizon_steps; ++i) {
      CartesianTrajectorySample s = timed_path.back();
      s.t = timed_path.back().t;
      s.dp.setZero();
      s.ddp.setZero();
      samples.push_back(s);
    }
    return samples;
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
      target.p.x(),
      target.p.y(),
      target.p.z()};
  input.target_velocity = {
      std::clamp(target.dp.x(), -max_velocity, max_velocity),
      std::clamp(target.dp.y(), -max_velocity, max_velocity),
      std::clamp(target.dp.z(), -max_velocity, max_velocity)};
  input.target_acceleration = {
      std::clamp(target.ddp.x(), -max_acceleration, max_acceleration),
      std::clamp(target.ddp.y(), -max_acceleration, max_acceleration),
      std::clamp(target.ddp.z(), -max_acceleration, max_acceleration)};

  input.max_velocity = {max_velocity, max_velocity, max_velocity};
  input.max_acceleration = {max_acceleration, max_acceleration, max_acceleration};
  input.max_jerk = {max_jerk, max_jerk, max_jerk};

  auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    input.target_velocity = {0.0, 0.0, 0.0};
    input.target_acceleration = {0.0, 0.0, 0.0};
    result = otg.calculate(input, trajectory);
  }
  if (result < ruckig::Result::Working) {
    samples.reserve(static_cast<std::size_t>(horizon_steps));
    for (int i = 0; i < horizon_steps; ++i) {
      const double alpha =
          static_cast<double>(i + 1) / static_cast<double>(horizon_steps);
      CartesianTrajectorySample s;
      s.t = (1.0 - alpha) * start_path_time + alpha * target_path_time;
      s.p = (1.0 - alpha) * planning_start.p + alpha * target.p;
      s.dp = (target.p - planning_start.p) /
             std::max(static_cast<double>(horizon_steps) * dt, kMinDt);
      s.ddp.setZero();
      s.q = target.q;
      s.q.normalize();
      samples.push_back(s);
    }
    return samples;
  }

  const double duration = std::max(trajectory.get_duration(), dt);
  const int n_steps =
      std::min(horizon_steps, std::max(1, static_cast<int>(std::ceil(duration / dt))));

  samples.reserve(static_cast<std::size_t>(n_steps));
  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
    const double alpha = std::clamp(tau / duration, 0.0, 1.0);

    std::array<double, 3> p{}, v{}, a{};
    trajectory.at_time(tau, p, v, a);

    CartesianTrajectorySample s;
    s.t = (1.0 - alpha) * start_path_time + alpha * target_path_time;
    s.p = Eigen::Vector3d(p[0], p[1], p[2]);
    s.dp = Eigen::Vector3d(v[0], v[1], v[2]);
    s.ddp = Eigen::Vector3d(a[0], a[1], a[2]);
    s.q = target.q;
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
