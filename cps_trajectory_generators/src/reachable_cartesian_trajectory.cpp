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

Eigen::Quaterniond normalizedOrIdentity(const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = q_in;
  const double norm = q.norm();
  if (!std::isfinite(norm) || norm < 1e-12) {
    return Eigen::Quaterniond::Identity();
  }
  q.coeffs() /= norm;
  return q;
}

Eigen::Quaterniond shortestEquivalent(
    const Eigen::Quaterniond& reference,
    const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = normalizedOrIdentity(q_in);
  if (q.coeffs().dot(reference.coeffs()) < 0.0) {
    q.coeffs() *= -1.0;
  }
  return q;
}

Eigen::Vector3d quaternionToRotationVector(const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = normalizedOrIdentity(q_in);
  if (q.w() < 0.0) {
    q.coeffs() *= -1.0;
  }
  Eigen::AngleAxisd aa(q);
  if (!std::isfinite(aa.angle()) || aa.angle() < 1e-12) {
    return Eigen::Vector3d::Zero();
  }
  return aa.axis() * aa.angle();
}

Eigen::Quaterniond rotationVectorToQuaternion(const Eigen::Vector3d& r) {
  const double angle = r.norm();
  if (!std::isfinite(angle) || angle < 1e-12) {
    return Eigen::Quaterniond::Identity();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, r / angle));
}

Eigen::Vector3d rotationVectorBetween(
    const Eigen::Quaterniond& from,
    const Eigen::Quaterniond& to) {
  const Eigen::Quaterniond q_from = normalizedOrIdentity(from);
  const Eigen::Quaterniond q_to = shortestEquivalent(q_from, to);
  return quaternionToRotationVector(q_to * q_from.conjugate());
}

Eigen::Vector3d clampVectorComponents(
    const Eigen::Vector3d& v,
    double limit) {
  const double clamped_limit = std::max(limit, 0.0);
  return Eigen::Vector3d(
      std::clamp(v.x(), -clamped_limit, clamped_limit),
      std::clamp(v.y(), -clamped_limit, clamped_limit),
      std::clamp(v.z(), -clamped_limit, clamped_limit));
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
    } else if (key == "local_replan_max_angular_velocity" &&
               parseDouble(value_text, &double_value)) {
      settings.local_replan_max_angular_velocity = double_value;
    } else if (key == "local_replan_max_angular_acceleration" &&
               parseDouble(value_text, &double_value)) {
      settings.local_replan_max_angular_acceleration = double_value;
    } else if (key == "local_replan_max_angular_jerk" &&
               parseDouble(value_text, &double_value)) {
      settings.local_replan_max_angular_jerk = double_value;
    } else if (key == "failsafe_brake_max_velocity" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_velocity = double_value;
    } else if (key == "failsafe_brake_max_acceleration" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_acceleration = double_value;
    } else if (key == "failsafe_brake_max_jerk" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_jerk = double_value;
    } else if (key == "failsafe_brake_max_angular_velocity" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_angular_velocity = double_value;
    } else if (key == "failsafe_brake_max_angular_acceleration" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_angular_acceleration = double_value;
    } else if (key == "failsafe_brake_max_angular_jerk" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_brake_max_angular_jerk = double_value;
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

CartesianTrajectorySample retimeTimedPathSample(
    const std::vector<CartesianTrajectorySample>& path,
    double path_time,
    double path_rate,
    double path_accel) {
  CartesianTrajectorySample out = sampleTimedPathAt(path, path_time);
  out.t = path_time;
  const Eigen::Vector3d dp_ds = out.dp;
  const Eigen::Vector3d d2p_ds2 = out.ddp;
  const Eigen::Vector3d w_ds = out.w;
  const Eigen::Vector3d dw_ds2 = out.dw;
  out.dp = dp_ds * path_rate;
  out.ddp = d2p_ds2 * path_rate * path_rate + dp_ds * path_accel;
  out.w = w_ds * path_rate;
  out.dw = dw_ds2 * path_rate * path_rate + w_ds * path_accel;
  return out;
}

double estimatePathRateAtSample(
    const CartesianTrajectorySample& planning_start,
    const CartesianTrajectorySample& path_sample,
    double max_path_rate) {
  const double denom =
      path_sample.dp.squaredNorm() + path_sample.w.squaredNorm();
  if (denom < 1e-10) {
    return 0.0;
  }
  const double rate =
      (planning_start.dp.dot(path_sample.dp) +
       planning_start.w.dot(path_sample.w)) /
      denom;
  if (!std::isfinite(rate)) {
    return 0.0;
  }
  return std::clamp(rate, 0.0, std::max(max_path_rate, 0.0));
}

double nearestPathTimeInWindow(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    double lookahead_sec) {
  if (timed_path.empty()) {
    return 0.0;
  }
  const double lower_time =
      std::clamp(min_path_time, timed_path.front().t, timed_path.back().t);
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
      std::min(lower_time + std::max(lookahead_sec, kMinDt), timed_path.back().t);
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
  return timed_path[nearest_idx].t;
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

Eigen::Vector3d clampToNondecreasingSegmentProgress(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    double* previous_progress) {
  if (previous_progress == nullptr) {
    return p;
  }

  const Eigen::Vector3d segment = segment_end - segment_start;
  const double length_sq = segment.squaredNorm();
  if (length_sq < kSmallPositive) {
    *previous_progress = 0.0;
    return segment_end;
  }

  double progress = (p - segment_start).dot(segment) / length_sq;
  progress = std::clamp(progress, *previous_progress, 1.0);
  *previous_progress = progress;

  return segment_start + progress * segment;
}

Eigen::Vector3d clampVelocityToNonnegativeSegmentProgress(
    const Eigen::Vector3d& v,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end) {
  const Eigen::Vector3d segment = segment_end - segment_start;
  const double length_sq = segment.squaredNorm();
  if (length_sq < kSmallPositive) {
    return Eigen::Vector3d::Zero();
  }

  const double progress_rate = v.dot(segment) / length_sq;
  if (progress_rate >= 0.0) {
    return v;
  }

  return v - progress_rate * segment;
}

}  // namespace

std::vector<CartesianTrajectorySample> makeSmoothViaPointCartesianTrajectory(
    const std::vector<Eigen::Vector3d>& waypoints,
    const Eigen::Quaterniond& reference_orientation,
    const LocalCartesianReplanConfig& config) {
  return makeSmoothViaPointCartesianTrajectory(
      waypoints,
      std::vector<Eigen::Quaterniond>(
          waypoints.size(), normalizedOrIdentity(reference_orientation)),
      config);
}

std::vector<CartesianTrajectorySample> makeSmoothViaPointCartesianTrajectory(
    const std::vector<Eigen::Vector3d>& waypoints,
    const std::vector<Eigen::Quaterniond>& waypoint_orientations,
    const LocalCartesianReplanConfig& config) {
  std::vector<CartesianTrajectorySample> samples;
  if (waypoints.empty()) {
    return samples;
  }

  const double dt = std::max(config.dt, kMinDt);
  const double max_velocity = std::max(config.max_velocity, 1e-4);
  const double max_angular_velocity =
      std::max(config.max_angular_velocity, 1e-4);
  const double max_acceleration = std::max(config.max_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_jerk, 1e-4);
  const double orientation_metric_scale = max_velocity / max_angular_velocity;

  const Eigen::Quaterniond fallback_orientation =
      waypoint_orientations.empty()
          ? Eigen::Quaterniond::Identity()
          : normalizedOrIdentity(waypoint_orientations.front());

  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Quaterniond> orientations;
  points.reserve(waypoints.size());
  orientations.reserve(waypoints.size());
  for (std::size_t i = 0; i < waypoints.size(); ++i) {
    Eigen::Quaterniond q =
        i < waypoint_orientations.size()
            ? normalizedOrIdentity(waypoint_orientations[i])
            : fallback_orientation;
    if (!orientations.empty()) {
      q = shortestEquivalent(orientations.back(), q);
    }

    if (points.empty() ||
        (waypoints[i] - points.back()).norm() > 1e-9 ||
        rotationVectorBetween(orientations.back(), q).norm() > 1e-9) {
      points.push_back(waypoints[i]);
      orientations.push_back(q);
    }
  }

  if (points.empty()) {
    return samples;
  }
  if (points.size() == 1) {
    CartesianTrajectorySample s;
    s.t = 0.0;
    s.p = points.front();
    s.q = orientations.front();
    samples.push_back(s);
    return samples;
  }

  std::vector<double> u(points.size(), 0.0);
  for (std::size_t i = 1; i < points.size(); ++i) {
    const double position_step = (points[i] - points[i - 1]).norm();
    const double angular_step =
        rotationVectorBetween(orientations[i - 1], orientations[i]).norm();
    const double full_state_step = std::hypot(
        position_step, orientation_metric_scale * angular_step);
    u[i] = u[i - 1] + std::max(full_state_step, kMinDt);
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

  const Eigen::Quaterniond q_ref = orientations.front();
  std::vector<Eigen::Vector3d> rotation_vectors(
      points.size(), Eigen::Vector3d::Zero());
  for (std::size_t i = 1; i < orientations.size(); ++i) {
    rotation_vectors[i] = rotationVectorBetween(q_ref, orientations[i]);
  }

  std::vector<Eigen::Vector3d> rotation_tangents(
      rotation_vectors.size(), Eigen::Vector3d::Zero());
  rotation_tangents.front() =
      (rotation_vectors[1] - rotation_vectors[0]) /
      std::max(u[1] - u[0], kMinDt);
  rotation_tangents.back() =
      (rotation_vectors.back() -
       rotation_vectors[rotation_vectors.size() - 2]) /
      std::max(u.back() - u[u.size() - 2], kMinDt);
  for (std::size_t i = 1; i + 1 < rotation_vectors.size(); ++i) {
    rotation_tangents[i] =
        (rotation_vectors[i + 1] - rotation_vectors[i - 1]) /
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
  first.q = orientations.front();
  samples.push_back(first);

  double previous_path_s = 0.0;
  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
    std::array<double, 1> s_arr{}, ds_arr{}, dds_arr{};
    trajectory.at_time(tau, s_arr, ds_arr, dds_arr);

    const double s_path = std::clamp(s_arr[0], previous_path_s, path_length);
    const double ds_path = std::max(0.0, ds_arr[0]);
    previous_path_s = s_path;
    const SmoothPathSample path_sample =
        sampleCubicHermitePath(points, u, tangents, s_path);
    const SmoothPathSample orientation_sample =
        sampleCubicHermitePath(
            rotation_vectors, u, rotation_tangents, s_path);

    CartesianTrajectorySample sample;
    sample.t = tau;
    sample.p = path_sample.p;
    sample.dp = path_sample.dp_ds * ds_path;
    sample.ddp =
        path_sample.d2p_ds2 * ds_path * ds_path +
        path_sample.dp_ds * dds_arr[0];
    sample.q = rotationVectorToQuaternion(orientation_sample.p) * q_ref;
    sample.q.normalize();
    sample.w = orientation_sample.dp_ds * ds_path;
    sample.dw =
        orientation_sample.d2p_ds2 * ds_path * ds_path +
        orientation_sample.dp_ds * dds_arr[0];
    if (i + 1 == n_steps) {
      sample.p = points.back();
      sample.dp.setZero();
      sample.ddp.setZero();
      sample.q = orientations.back();
      sample.q.normalize();
      sample.w.setZero();
      sample.dw.setZero();
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
  const double max_angular_velocity =
      std::max(config.max_angular_velocity, 1e-4);
  const double max_angular_acceleration =
      std::max(config.max_angular_acceleration, 1e-4);
  const double max_angular_jerk =
      std::max(config.max_angular_jerk, 1e-4);

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
      planning_start.dp.norm() < 1e-3 &&
      rotationVectorBetween(planning_start.q, timed_path.back().q).norm() < 1e-3 &&
      planning_start.w.norm() < 1e-3;

  if (at_end) {
    samples.reserve(static_cast<std::size_t>(horizon_steps));
    for (int i = 0; i < horizon_steps; ++i) {
      CartesianTrajectorySample s = timed_path.back();
      s.t = timed_path.back().t;
      s.dp.setZero();
      s.ddp.setZero();
      s.w.setZero();
      s.dw.setZero();
      samples.push_back(s);
    }
    return samples;
  }

  const Eigen::Quaterniond planning_start_orientation =
      normalizedOrIdentity(planning_start.q);
  const Eigen::Vector3d target_rotation =
      rotationVectorBetween(planning_start_orientation, target.q);

  ruckig::Ruckig<6> otg;
  ruckig::InputParameter<6> input;
  ruckig::Trajectory<6> trajectory;

  input.current_position = {
      planning_start.p.x(),
      planning_start.p.y(),
      planning_start.p.z(),
      0.0,
      0.0,
      0.0};
  input.current_velocity = {
      planning_start.dp.x(),
      planning_start.dp.y(),
      planning_start.dp.z(),
      planning_start.w.x(),
      planning_start.w.y(),
      planning_start.w.z()};
  input.current_acceleration = {
      planning_start.ddp.x(),
      planning_start.ddp.y(),
      planning_start.ddp.z(),
      planning_start.dw.x(),
      planning_start.dw.y(),
      planning_start.dw.z()};

  input.target_position = {
      target.p.x(),
      target.p.y(),
      target.p.z(),
      target_rotation.x(),
      target_rotation.y(),
      target_rotation.z()};
  const Eigen::Vector3d target_w =
      clampVectorComponents(target.w, max_angular_velocity);
  const Eigen::Vector3d target_dw =
      clampVectorComponents(target.dw, max_angular_acceleration);
  input.target_velocity = {
      std::clamp(target.dp.x(), -max_velocity, max_velocity),
      std::clamp(target.dp.y(), -max_velocity, max_velocity),
      std::clamp(target.dp.z(), -max_velocity, max_velocity),
      target_w.x(),
      target_w.y(),
      target_w.z()};
  input.target_acceleration = {
      std::clamp(target.ddp.x(), -max_acceleration, max_acceleration),
      std::clamp(target.ddp.y(), -max_acceleration, max_acceleration),
      std::clamp(target.ddp.z(), -max_acceleration, max_acceleration),
      target_dw.x(),
      target_dw.y(),
      target_dw.z()};

  input.max_velocity = {
      max_velocity, max_velocity, max_velocity,
      max_angular_velocity, max_angular_velocity, max_angular_velocity};
  input.max_acceleration = {
      max_acceleration, max_acceleration, max_acceleration,
      max_angular_acceleration, max_angular_acceleration,
      max_angular_acceleration};
  input.max_jerk = {
      max_jerk, max_jerk, max_jerk,
      max_angular_jerk, max_angular_jerk, max_angular_jerk};

  auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    input.target_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    input.target_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
      s.q = planning_start_orientation.slerp(alpha, target.q);
      s.q.normalize();
      s.w = target_rotation /
            std::max(static_cast<double>(horizon_steps) * dt, kMinDt);
      s.dw.setZero();
      if (i + 1 == horizon_steps) {
        s.p = target.p;
        s.dp.setZero();
        s.q = target.q;
        s.q.normalize();
        s.w = target.w;
        s.dw = target.dw;
      }
      samples.push_back(s);
    }
    return samples;
  }

  const double duration = std::max(trajectory.get_duration(), dt);
  const int n_steps =
      std::min(horizon_steps, std::max(1, static_cast<int>(std::ceil(duration / dt))));

  samples.reserve(static_cast<std::size_t>(n_steps));
  double previous_progress = 0.0;
  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
    const double alpha = std::clamp(tau / duration, 0.0, 1.0);

    std::array<double, 6> p{}, v{}, a{};
    trajectory.at_time(tau, p, v, a);

    CartesianTrajectorySample s;
    s.t = (1.0 - alpha) * start_path_time + alpha * target_path_time;
    s.p = clampToNondecreasingSegmentProgress(
        Eigen::Vector3d(p[0], p[1], p[2]),
        planning_start.p,
        target.p,
        &previous_progress);
    s.dp = clampVelocityToNonnegativeSegmentProgress(
        Eigen::Vector3d(v[0], v[1], v[2]),
        planning_start.p,
        target.p);
    s.ddp = Eigen::Vector3d(a[0], a[1], a[2]);
    s.q =
        rotationVectorToQuaternion(Eigen::Vector3d(p[3], p[4], p[5])) *
        planning_start_orientation;
    s.q.normalize();
    s.w = Eigen::Vector3d(v[3], v[4], v[5]);
    s.dw = Eigen::Vector3d(a[3], a[4], a[5]);
    if (tau >= duration - kMinDt) {
      s.q = target.q;
      s.q.normalize();
      s.w = target.w;
      s.dw = target.dw;
    }
    samples.push_back(s);
  }

  return samples;
}

std::vector<CartesianTrajectorySample> makePathConsistentTimedPathReplan(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const PathConsistentTimedPathConfig& config) {
  std::vector<CartesianTrajectorySample> samples;
  if (timed_path.empty()) {
    return samples;
  }

  const double dt = std::max(config.dt, kMinDt);
  const int intended_steps = std::max(1, config.intended_steps);
  const double max_rate = std::max(config.max_path_rate, 1e-4);
  const double max_accel = std::max(config.max_path_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_path_jerk, 1e-4);
  const double start_path_time = nearestPathTimeInWindow(
      min_path_time, planning_start, timed_path, config.path_lookahead_sec);
  const CartesianTrajectorySample path_start =
      sampleTimedPathAt(timed_path, start_path_time);
  const double estimated_start_rate =
      estimatePathRateAtSample(planning_start, path_start, max_rate);
  const double start_rate =
      config.initial_path_rate >= 0.0
          ? std::clamp(config.initial_path_rate, 0.0, max_rate)
          : estimated_start_rate;
  const double target_path_time = std::min(
      std::max(start_path_time + std::max(config.path_lookahead_sec, dt),
               start_path_time + dt),
      timed_path.back().t);

  if (target_path_time <= start_path_time + kMinDt) {
    samples.reserve(static_cast<std::size_t>(intended_steps));
    for (int i = 0; i < intended_steps; ++i) {
      CartesianTrajectorySample s = sampleTimedPathAt(timed_path, timed_path.back().t);
      s.dp.setZero();
      s.ddp.setZero();
      s.w.setZero();
      s.dw.setZero();
      samples.push_back(s);
    }
    return samples;
  }

  ruckig::Ruckig<1> otg;
  ruckig::InputParameter<1> input;
  ruckig::Trajectory<1> trajectory;

  input.current_position = {start_path_time};
  input.current_velocity = {start_rate};
  input.current_acceleration = {0.0};
  input.target_position = {target_path_time};
  const bool target_is_path_end =
      target_path_time >= timed_path.back().t - kMinDt;
  input.target_velocity = {
      target_is_path_end
          ? 0.0
          : std::clamp(config.target_path_rate, 0.0, max_rate)};
  input.target_acceleration = {0.0};
  input.max_velocity = {max_rate};
  input.max_acceleration = {max_accel};
  input.max_jerk = {max_jerk};

  auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    input.target_velocity = {0.0};
    result = otg.calculate(input, trajectory);
  }
  if (result < ruckig::Result::Working) {
    return samples;
  }

  const double duration = std::max(trajectory.get_duration(), dt);
  const int n_steps = std::min(
      intended_steps,
      std::max(1, static_cast<int>(std::ceil(duration / dt))));
  samples.reserve(static_cast<std::size_t>(n_steps));

  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
    std::array<double, 1> s_arr{}, ds_arr{}, dds_arr{};
    trajectory.at_time(tau, s_arr, ds_arr, dds_arr);
    samples.push_back(retimeTimedPathSample(
        timed_path,
        std::clamp(s_arr[0], timed_path.front().t, timed_path.back().t),
        std::clamp(ds_arr[0], 0.0, max_rate),
        dds_arr[0]));
  }

  return samples;
}

std::vector<CartesianTrajectorySample> makePathConsistentTimedPathIntendedPrefix(
    double min_path_time,
    const CartesianTrajectorySample& planning_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const PathConsistentTimedPathConfig& config) {
  std::vector<CartesianTrajectorySample> samples;
  if (timed_path.empty()) {
    return samples;
  }

  const double dt = std::max(config.dt, kMinDt);
  const int intended_steps = std::max(1, config.intended_steps);
  const double max_rate = std::max(config.max_path_rate, 1e-4);
  const double max_accel = std::max(config.max_path_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_path_jerk, 1e-4);
  const double start_path_time = nearestPathTimeInWindow(
      min_path_time, planning_start, timed_path, config.path_lookahead_sec);
  const CartesianTrajectorySample path_start =
      sampleTimedPathAt(timed_path, start_path_time);
  const double estimated_start_rate =
      estimatePathRateAtSample(planning_start, path_start, max_rate);
  const double start_rate =
      config.initial_path_rate >= 0.0
          ? std::clamp(config.initial_path_rate, 0.0, max_rate)
          : estimated_start_rate;
  const double target_rate =
      std::clamp(config.target_path_rate, 0.0, max_rate);

  samples.reserve(static_cast<std::size_t>(intended_steps));
  if (start_path_time >= timed_path.back().t - kMinDt) {
    for (int i = 0; i < intended_steps; ++i) {
      CartesianTrajectorySample s = sampleTimedPathAt(timed_path, timed_path.back().t);
      s.dp.setZero();
      s.ddp.setZero();
      s.w.setZero();
      s.dw.setZero();
      samples.push_back(s);
    }
    return samples;
  }

  ruckig::Ruckig<1> otg;
  ruckig::InputParameter<1> input;
  ruckig::Trajectory<1> trajectory;

  input.control_interface = ruckig::ControlInterface::Velocity;
  input.current_position = {start_path_time};
  input.current_velocity = {start_rate};
  input.current_acceleration = {0.0};
  input.target_velocity = {target_rate};
  input.target_acceleration = {0.0};
  input.max_velocity = {max_rate};
  input.max_acceleration = {max_accel};
  input.max_jerk = {max_jerk};

  const auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    return {};
  }

  const double duration = std::max(trajectory.get_duration(), 0.0);
  std::array<double, 1> s_final{start_path_time};
  std::array<double, 1> ds_final{target_rate};
  std::array<double, 1> dds_final{0.0};
  if (duration > kMinDt) {
    trajectory.at_time(duration, s_final, ds_final, dds_final);
  }

  for (int i = 0; i < intended_steps; ++i) {
    const double tau = static_cast<double>(i + 1) * dt;
    std::array<double, 1> s_arr{}, ds_arr{}, dds_arr{};

    if (duration > kMinDt && tau <= duration + kMinDt) {
      trajectory.at_time(std::min(tau, duration), s_arr, ds_arr, dds_arr);
    } else {
      const double extra_time = std::max(0.0, tau - duration);
      s_arr[0] = s_final[0] + ds_final[0] * extra_time;
      ds_arr[0] = ds_final[0];
      dds_arr[0] = 0.0;
    }

    const double path_time =
        std::clamp(s_arr[0], timed_path.front().t, timed_path.back().t);
    const bool at_path_end = path_time >= timed_path.back().t - kMinDt;
    samples.push_back(retimeTimedPathSample(
        timed_path,
        path_time,
        at_path_end ? 0.0 : std::clamp(ds_arr[0], 0.0, max_rate),
        at_path_end ? 0.0 : dds_arr[0]));
  }

  return samples;
}

std::vector<CartesianTrajectorySample> makePathConsistentTimedPathBrake(
    double path_time,
    const CartesianTrajectorySample& brake_start,
    const std::vector<CartesianTrajectorySample>& timed_path,
    const PathConsistentTimedPathConfig& config) {
  std::vector<CartesianTrajectorySample> samples;
  if (timed_path.empty()) {
    return samples;
  }

  const double dt = std::max(config.dt, kMinDt);
  const double max_rate = std::max(config.max_path_rate, 1e-4);
  const double max_accel = std::max(config.max_path_acceleration, 1e-4);
  const double max_jerk = std::max(config.max_path_jerk, 1e-4);
  const double start_path_time =
      std::clamp(path_time, timed_path.front().t, timed_path.back().t);
  const CartesianTrajectorySample path_start =
      sampleTimedPathAt(timed_path, start_path_time);
  const double start_rate =
      estimatePathRateAtSample(brake_start, path_start, max_rate);

  ruckig::Ruckig<1> otg;
  ruckig::InputParameter<1> input;
  ruckig::Trajectory<1> trajectory;
  input.control_interface = ruckig::ControlInterface::Velocity;
  input.current_position = {start_path_time};
  input.current_velocity = {start_rate};
  input.current_acceleration = {0.0};
  input.target_velocity = {0.0};
  input.target_acceleration = {0.0};
  input.max_velocity = {max_rate};
  input.max_acceleration = {max_accel};
  input.max_jerk = {max_jerk};

  const auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    return samples;
  }

  const double duration = std::max(trajectory.get_duration(), dt);
  const int n_steps = std::max(1, static_cast<int>(std::ceil(duration / dt)));
  samples.reserve(static_cast<std::size_t>(n_steps));

  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
    std::array<double, 1> s_arr{}, ds_arr{}, dds_arr{};
    trajectory.at_time(tau, s_arr, ds_arr, dds_arr);
    CartesianTrajectorySample s = retimeTimedPathSample(
        timed_path,
        std::clamp(s_arr[0], timed_path.front().t, timed_path.back().t),
        std::clamp(ds_arr[0], 0.0, max_rate),
        dds_arr[0]);
    if (i + 1 == n_steps) {
      s.dp.setZero();
      s.ddp.setZero();
      s.w.setZero();
      s.dw.setZero();
    }
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
  const double max_angular_velocity =
      std::max(config.max_angular_velocity, 1e-4);
  const double max_angular_acceleration =
      std::max(config.max_angular_acceleration, 1e-4);
  const double max_angular_jerk =
      std::max(config.max_angular_jerk, 1e-4);

  const Eigen::Quaterniond brake_start_orientation =
      normalizedOrIdentity(brake_start.q);

  ruckig::Ruckig<6> otg;
  ruckig::InputParameter<6> input;
  ruckig::Trajectory<6> trajectory;

  input.control_interface = ruckig::ControlInterface::Velocity;

  input.current_position = {
      brake_start.p.x(),
      brake_start.p.y(),
      brake_start.p.z(),
      0.0,
      0.0,
      0.0};

  input.current_velocity = {
      brake_start.dp.x(),
      brake_start.dp.y(),
      brake_start.dp.z(),
      brake_start.w.x(),
      brake_start.w.y(),
      brake_start.w.z()};

  input.current_acceleration = {
      brake_start.ddp.x(),
      brake_start.ddp.y(),
      brake_start.ddp.z(),
      brake_start.dw.x(),
      brake_start.dw.y(),
      brake_start.dw.z()};

  input.target_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  input.target_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  input.max_velocity = {
      max_velocity, max_velocity, max_velocity,
      max_angular_velocity, max_angular_velocity, max_angular_velocity};
  input.max_acceleration = {
      max_acceleration, max_acceleration, max_acceleration,
      max_angular_acceleration, max_angular_acceleration,
      max_angular_acceleration};
  input.max_jerk = {
      max_jerk, max_jerk, max_jerk,
      max_angular_jerk, max_angular_jerk, max_angular_jerk};

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

    std::array<double, 6> p{}, v{}, a{};
    trajectory.at_time(tau, p, v, a);

    CartesianTrajectorySample s;
    s.t = static_cast<double>(i + 1) * dt;
    s.p = Eigen::Vector3d(p[0], p[1], p[2]);
    s.dp = Eigen::Vector3d(v[0], v[1], v[2]);
    s.ddp = Eigen::Vector3d(a[0], a[1], a[2]);
    s.q =
        rotationVectorToQuaternion(Eigen::Vector3d(p[3], p[4], p[5])) *
        brake_start_orientation;
    s.q.normalize();
    s.w = Eigen::Vector3d(v[3], v[4], v[5]);
    s.dw = Eigen::Vector3d(a[3], a[4], a[5]);
    if (i + 1 == n_steps) {
      s.dp.setZero();
      s.ddp.setZero();
      s.w.setZero();
      s.dw.setZero();
    }
    samples.push_back(s);
  }

  return samples;
}

}  // namespace cps_trajectory_generators
