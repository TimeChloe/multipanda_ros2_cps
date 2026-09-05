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

struct So3LeftJacobianCoefficients {
  double a{0.5};
  double b{1.0 / 6.0};
  double da_dx{-1.0 / 24.0};
  double db_dx{-1.0 / 120.0};
  double d2a_dx2{1.0 / 360.0};
  double d2b_dx2{1.0 / 2520.0};
};

So3LeftJacobianCoefficients so3LeftJacobianCoefficients(
    double angle_squared) {
  const double x = std::max(angle_squared, 0.0);
  So3LeftJacobianCoefficients out;

  // Express the coefficients as functions of x = |r|^2.  This avoids the
  // undefined derivatives of |r| at r = 0 and gives a stable series for the
  // small-angle region where the closed forms suffer cancellation.
  if (x < 1.0e-4) {
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x3 * x;
    out.a =
        0.5 - x / 24.0 + x2 / 720.0 - x3 / 40320.0 + x4 / 3628800.0;
    out.b =
        1.0 / 6.0 - x / 120.0 + x2 / 5040.0 - x3 / 362880.0 +
        x4 / 39916800.0;
    out.da_dx =
        -1.0 / 24.0 + x / 360.0 - x2 / 13440.0 + x3 / 907200.0;
    out.db_dx =
        -1.0 / 120.0 + x / 2520.0 - x2 / 120960.0 +
        x3 / 9979200.0;
    out.d2a_dx2 =
        1.0 / 360.0 - x / 6720.0 + x2 / 302400.0;
    out.d2b_dx2 =
        1.0 / 2520.0 - x / 60480.0 + x2 / 3326400.0;
    return out;
  }

  const double angle = std::sqrt(x);
  const double sine = std::sin(angle);
  const double cosine = std::cos(angle);
  const double angle2 = x;
  const double angle3 = angle2 * angle;
  const double angle4 = angle2 * angle2;
  const double angle5 = angle4 * angle;

  out.a = (1.0 - cosine) / angle2;
  out.b = (angle - sine) / angle3;

  const double da_dangle =
      (angle * sine - 2.0 * (1.0 - cosine)) / angle3;
  const double db_dangle =
      (3.0 * sine - 2.0 * angle - angle * cosine) / angle4;
  const double d2a_dangle2 =
      (angle2 * cosine - 4.0 * angle * sine +
       6.0 * (1.0 - cosine)) /
      angle4;
  const double d2b_dangle2 =
      (angle2 * sine - 12.0 * sine +
       6.0 * angle * (1.0 + cosine)) /
      angle5;

  out.da_dx = da_dangle / (2.0 * angle);
  out.db_dx = db_dangle / (2.0 * angle);
  out.d2a_dx2 =
      d2a_dangle2 / (4.0 * angle2) -
      da_dangle / (4.0 * angle3);
  out.d2b_dx2 =
      d2b_dangle2 / (4.0 * angle2) -
      db_dangle / (4.0 * angle3);
  return out;
}

struct So3AngularPathDerivatives {
  // For R(s) = Exp(r(s)) R_ref in world coordinates:
  //   omega = omega_ds * s_dot
  //   alpha = domega_ds2 * s_dot^2 + omega_ds * s_ddot
  //   jerk  = d2omega_ds3 * s_dot^3
  //          + 3 domega_ds2 * s_dot * s_ddot + omega_ds * s_jerk.
  Eigen::Vector3d omega_ds{Eigen::Vector3d::Zero()};
  Eigen::Vector3d domega_ds2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d d2omega_ds3{Eigen::Vector3d::Zero()};
};

So3AngularPathDerivatives rotationVectorToWorldAngularPathDerivatives(
    const Eigen::Vector3d& rotation_vector,
    const Eigen::Vector3d& dr_ds,
    const Eigen::Vector3d& d2r_ds2,
    const Eigen::Vector3d& d3r_ds3) {
  const double x = rotation_vector.squaredNorm();
  const So3LeftJacobianCoefficients coefficients =
      so3LeftJacobianCoefficients(x);
  const double dx_ds = 2.0 * rotation_vector.dot(dr_ds);
  const double d2x_ds2 =
      2.0 * (dr_ds.squaredNorm() + rotation_vector.dot(d2r_ds2));
  const double da_ds = coefficients.da_dx * dx_ds;
  const double db_ds = coefficients.db_dx * dx_ds;
  const double d2a_ds2 =
      coefficients.d2a_dx2 * dx_ds * dx_ds +
      coefficients.da_dx * d2x_ds2;
  const double d2b_ds2 =
      coefficients.d2b_dx2 * dx_ds * dx_ds +
      coefficients.db_dx * d2x_ds2;

  const Eigen::Vector3d cross_1 = rotation_vector.cross(dr_ds);
  const Eigen::Vector3d dcross_1_ds =
      rotation_vector.cross(d2r_ds2);
  const Eigen::Vector3d d2cross_1_ds2 =
      dr_ds.cross(d2r_ds2) +
      rotation_vector.cross(d3r_ds3);

  const Eigen::Vector3d cross_2 = rotation_vector.cross(cross_1);
  const Eigen::Vector3d dcross_2_ds =
      dr_ds.cross(cross_1) +
      rotation_vector.cross(dcross_1_ds);
  const Eigen::Vector3d d2cross_2_ds2 =
      d2r_ds2.cross(cross_1) +
      2.0 * dr_ds.cross(dcross_1_ds) +
      rotation_vector.cross(d2cross_1_ds2);

  So3AngularPathDerivatives out;
  out.omega_ds =
      dr_ds + coefficients.a * cross_1 + coefficients.b * cross_2;
  out.domega_ds2 =
      d2r_ds2 + da_ds * cross_1 +
      coefficients.a * dcross_1_ds + db_ds * cross_2 +
      coefficients.b * dcross_2_ds;
  out.d2omega_ds3 =
      d3r_ds3 + d2a_ds2 * cross_1 +
      2.0 * da_ds * dcross_1_ds +
      coefficients.a * d2cross_1_ds2 + d2b_ds2 * cross_2 +
      2.0 * db_ds * dcross_2_ds +
      coefficients.b * d2cross_2_ds2;
  return out;
}

double makeNonreversingInitialPathAcceleration(
    double path_rate,
    double path_acceleration,
    double max_path_acceleration,
    double max_path_jerk) {
  const double acceleration_limit =
      std::max(max_path_acceleration, 0.0);
  double acceleration = std::clamp(
      path_acceleration, -acceleration_limit, acceleration_limit);

  // With negative initial acceleration, the rate keeps falling until bounded
  // positive jerk can bring acceleration back to zero. Limit that deceleration
  // to the largest value that cannot make a nonnegative path rate reverse:
  //   delta_v = a^2 / (2 j) <= v.
  if (acceleration < 0.0) {
    const double nonreversing_deceleration = std::sqrt(
        2.0 * std::max(max_path_jerk, kSmallPositive) *
        std::max(path_rate, 0.0));
    acceleration = std::max(acceleration, -nonreversing_deceleration);
  }
  return acceleration;
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
    } else if (key == "waypoint_merge_position_tolerance" &&
               parseDouble(value_text, &double_value)) {
      settings.waypoint_merge_position_tolerance = double_value;
    } else if (key == "waypoint_merge_orientation_tolerance" &&
               parseDouble(value_text, &double_value)) {
      settings.waypoint_merge_orientation_tolerance = double_value;
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
    } else if (key == "path_time_jerk_limit" &&
               parseDouble(value_text, &double_value)) {
      settings.path_time_jerk_limit = double_value;
    } else if (key == "path_time_rate_target" &&
               parseDouble(value_text, &double_value)) {
      settings.path_time_rate_target = double_value;
    } else if (key == "failsafe_path_time_acc_limit" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_path_time_acc_limit = double_value;
    } else if (key == "failsafe_path_time_jerk_limit" &&
               parseDouble(value_text, &double_value)) {
      settings.failsafe_path_time_jerk_limit = double_value;
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
  out.path_rate = path_rate;
  out.path_acceleration = path_accel;
  out.path_kinematics_valid =
      std::isfinite(path_rate) && std::isfinite(path_accel);
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

double estimatePathAccelerationAtSample(
    const CartesianTrajectorySample& planning_start,
    const CartesianTrajectorySample& path_sample,
    double path_rate,
    double max_path_acceleration) {
  // For a retimed path x(s(t)):
  //   x_ddot = x_ss * s_dot^2 + x_s * s_ddot.
  // Recover the scalar acceleration from the exact Cartesian command that
  // will precede the new prefix.  Starting Ruckig at zero acceleration here
  // creates a seam and can force the controller onto a non-path Cartesian
  // fallback whenever planning overlaps already executing commands.
  const double denom =
      path_sample.dp.squaredNorm() + path_sample.w.squaredNorm();
  if (denom < 1e-10) {
    return 0.0;
  }

  const double rate_squared = path_rate * path_rate;
  const Eigen::Vector3d residual_linear =
      planning_start.ddp - path_sample.ddp * rate_squared;
  const Eigen::Vector3d residual_angular =
      planning_start.dw - path_sample.dw * rate_squared;
  const double acceleration =
      (path_sample.dp.dot(residual_linear) +
       path_sample.w.dot(residual_angular)) /
      denom;
  if (!std::isfinite(acceleration)) {
    return 0.0;
  }
  const double limit = std::max(max_path_acceleration, 0.0);
  return std::clamp(acceleration, -limit, limit);
}

struct SmoothPathSample {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dp_ds{Eigen::Vector3d::Zero()};
  Eigen::Vector3d d2p_ds2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d d3p_ds3{Eigen::Vector3d::Zero()};
};

SmoothPathSample sampleSepticHermitePath(
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
  const double x4 = x3 * x;
  const double x5 = x4 * x;
  const double x6 = x5 * x;
  const double x7 = x6 * x;

  // Septic Hermite with shared first derivatives and zero second and third
  // derivatives at every waypoint.  Adjacent segments are C3, preventing
  // both acceleration and jerk seams at via points.
  const double h00 =
      1.0 - 35.0 * x4 + 84.0 * x5 - 70.0 * x6 + 20.0 * x7;
  const double h10 =
      x - 20.0 * x4 + 45.0 * x5 - 36.0 * x6 + 10.0 * x7;
  const double h01 =
      35.0 * x4 - 84.0 * x5 + 70.0 * x6 - 20.0 * x7;
  const double h11 =
      -15.0 * x4 + 39.0 * x5 - 34.0 * x6 + 10.0 * x7;

  const double dh00 =
      -140.0 * x3 + 420.0 * x4 - 420.0 * x5 + 140.0 * x6;
  const double dh10 =
      1.0 - 80.0 * x3 + 225.0 * x4 - 216.0 * x5 + 70.0 * x6;
  const double dh01 =
      140.0 * x3 - 420.0 * x4 + 420.0 * x5 - 140.0 * x6;
  const double dh11 =
      -60.0 * x3 + 195.0 * x4 - 204.0 * x5 + 70.0 * x6;

  const double d2h00 =
      -420.0 * x2 + 1680.0 * x3 - 2100.0 * x4 + 840.0 * x5;
  const double d2h10 =
      -240.0 * x2 + 900.0 * x3 - 1080.0 * x4 + 420.0 * x5;
  const double d2h01 =
      420.0 * x2 - 1680.0 * x3 + 2100.0 * x4 - 840.0 * x5;
  const double d2h11 =
      -180.0 * x2 + 780.0 * x3 - 1020.0 * x4 + 420.0 * x5;

  const double d3h00 =
      -840.0 * x + 5040.0 * x2 - 8400.0 * x3 + 4200.0 * x4;
  const double d3h10 =
      -480.0 * x + 2700.0 * x2 - 4320.0 * x3 + 2100.0 * x4;
  const double d3h01 =
      840.0 * x - 5040.0 * x2 + 8400.0 * x3 - 4200.0 * x4;
  const double d3h11 =
      -360.0 * x + 2340.0 * x2 - 4080.0 * x3 + 2100.0 * x4;

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
  out.d3p_ds3 =
      (d3h00 * p0 + d3h10 * h * m0 + d3h01 * p1 + d3h11 * h * m1) /
      (h * h * h);
  return out;
}

}  // namespace

CartesianTrajectorySample makeRetimedPathState(
    const std::vector<CartesianTrajectorySample>& timed_path,
    double path_time,
    double path_rate,
    double path_acceleration) {
  return retimeTimedPathSample(
      timed_path, path_time, path_rate, path_acceleration);
}

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
  const double max_angular_acceleration =
      std::max(config.max_angular_acceleration, 1e-4);
  const double max_angular_jerk =
      std::max(config.max_angular_jerk, 1e-4);
  const double orientation_metric_scale = max_velocity / max_angular_velocity;

  const Eigen::Quaterniond fallback_orientation =
      waypoint_orientations.empty()
          ? Eigen::Quaterniond::Identity()
          : normalizedOrIdentity(waypoint_orientations.front());

  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Quaterniond> orientations;
  points.reserve(waypoints.size());
  orientations.reserve(waypoints.size());
  const double merge_position_tolerance =
      std::max(config.waypoint_merge_position_tolerance, 0.0);
  const double merge_orientation_tolerance =
      std::max(config.waypoint_merge_orientation_tolerance, 0.0);
  for (std::size_t i = 0; i < waypoints.size(); ++i) {
    Eigen::Quaterniond q =
        i < waypoint_orientations.size()
            ? normalizedOrIdentity(waypoint_orientations[i])
            : fallback_orientation;
    if (!orientations.empty()) {
      q = shortestEquivalent(orientations.back(), q);
    }

    if (points.empty()) {
      points.push_back(waypoints[i]);
      orientations.push_back(q);
      continue;
    }

    const double position_distance = (waypoints[i] - points.back()).norm();
    const double orientation_distance =
        rotationVectorBetween(orientations.back(), q).norm();
    const bool near_previous =
        position_distance <= merge_position_tolerance &&
        orientation_distance <= merge_orientation_tolerance;
    const bool is_final_target = i + 1 == waypoints.size();
    if (!near_previous) {
      points.push_back(waypoints[i]);
      orientations.push_back(q);
    } else if (is_final_target) {
      // Preserve an intentional tiny final motion. If there is an earlier
      // intermediate point, replace that redundant point with the exact final
      // target instead of creating a near-zero terminal segment.
      if (points.size() == 1 &&
          (position_distance > 1e-12 || orientation_distance > 1e-12)) {
        points.push_back(waypoints[i]);
        orientations.push_back(q);
      } else {
        points.back() = waypoints[i];
        orientations.back() = q;
      }
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

  // A waypoint that reverses the combined translational/orientational path
  // direction is a real cusp. A cross-waypoint central tangent would make the
  // spline overshoot that waypoint before returning. Force the geometric
  // derivative to zero only at these cusps; ordinary via points retain their
  // shared nonzero derivative and can be crossed without stopping.
  std::vector<bool> direction_reversals(points.size(), false);
  for (std::size_t i = 1; i + 1 < points.size(); ++i) {
    const Eigen::Vector3d linear_before = points[i] - points[i - 1];
    const Eigen::Vector3d linear_after = points[i + 1] - points[i];
    const Eigen::Vector3d angular_before =
        rotation_vectors[i] - rotation_vectors[i - 1];
    const Eigen::Vector3d angular_after =
        rotation_vectors[i + 1] - rotation_vectors[i];
    const double direction_dot =
        linear_before.dot(linear_after) +
        orientation_metric_scale * orientation_metric_scale *
            angular_before.dot(angular_after);
    if (direction_dot < 0.0) {
      direction_reversals[i] = true;
      tangents[i].setZero();
      rotation_tangents[i].setZero();
    }
  }

  // Convert the configured 3D Euclidean-norm limits to conservative scalar
  // Ruckig limits independently for every geometric waypoint segment:
  //   p_dot   = p_s s_dot
  //   p_ddot  = p_ss s_dot^2 + p_s s_ddot
  //   p_jerk  = p_sss s_dot^3 + 3 p_ss s_dot s_ddot + p_s s_jerk.
  // A pathological short segment therefore only slows itself and the speed
  // needed to enter/leave it; a following straight segment can accelerate
  // back to the configured Cartesian norm limit.
  struct ScalarSegmentLimits {
    double max_velocity{0.0};
    double max_acceleration{0.0};
    double max_jerk{0.0};
  };

  std::vector<ScalarSegmentLimits> segment_limits(points.size() - 1);
  constexpr int kDerivativeSamplesPerSegment = 128;
  constexpr double kPathLimitSamplingMargin = 0.995;
  for (std::size_t segment = 0; segment + 1 < u.size(); ++segment) {
    double max_linear_d1 = 0.0;
    double max_linear_d2 = 0.0;
    double max_linear_d3 = 0.0;
    double max_angular_d1 = 0.0;
    double max_angular_d2 = 0.0;
    double max_angular_d3 = 0.0;
    for (int j = 0; j <= kDerivativeSamplesPerSegment; ++j) {
      const double alpha =
          static_cast<double>(j) /
          static_cast<double>(kDerivativeSamplesPerSegment);
      const double path_s =
          (1.0 - alpha) * u[segment] + alpha * u[segment + 1];
      const SmoothPathSample linear_sample =
          sampleSepticHermitePath(points, u, tangents, path_s);
      const SmoothPathSample angular_sample =
          sampleSepticHermitePath(
              rotation_vectors, u, rotation_tangents, path_s);
      const So3AngularPathDerivatives angular_derivatives =
          rotationVectorToWorldAngularPathDerivatives(
              angular_sample.p,
              angular_sample.dp_ds,
              angular_sample.d2p_ds2,
              angular_sample.d3p_ds3);
      max_linear_d1 = std::max(max_linear_d1, linear_sample.dp_ds.norm());
      max_linear_d2 = std::max(max_linear_d2, linear_sample.d2p_ds2.norm());
      max_linear_d3 = std::max(max_linear_d3, linear_sample.d3p_ds3.norm());
      max_angular_d1 =
          std::max(max_angular_d1, angular_derivatives.omega_ds.norm());
      max_angular_d2 =
          std::max(max_angular_d2, angular_derivatives.domega_ds2.norm());
      max_angular_d3 =
          std::max(max_angular_d3, angular_derivatives.d2omega_ds3.norm());
    }

    double scalar_max_velocity = max_velocity;
    if (max_linear_d1 > kSmallPositive) {
      scalar_max_velocity = std::min(
          scalar_max_velocity, max_velocity / max_linear_d1);
    }
    if (max_angular_d1 > kSmallPositive) {
      scalar_max_velocity = std::min(
          scalar_max_velocity, max_angular_velocity / max_angular_d1);
    }
    scalar_max_velocity = std::max(scalar_max_velocity, 1e-4);

    auto scalarAccelerationLimit = [&](double scalar_velocity) {
      double limit = max_acceleration;
      if (max_linear_d1 > kSmallPositive) {
        limit = std::min(
            limit,
            (max_acceleration -
             max_linear_d2 * scalar_velocity * scalar_velocity) /
                max_linear_d1);
      }
      if (max_angular_d1 > kSmallPositive) {
        limit = std::min(
            limit,
            (max_angular_acceleration -
             max_angular_d2 * scalar_velocity * scalar_velocity) /
                max_angular_d1);
      }
      return limit;
    };
    auto scalarJerkLimit = [&](double scalar_velocity,
                               double scalar_acceleration) {
      double limit = max_jerk;
      if (max_linear_d1 > kSmallPositive) {
        const double used =
            max_linear_d3 * scalar_velocity * scalar_velocity *
                scalar_velocity +
            3.0 * max_linear_d2 * scalar_velocity * scalar_acceleration;
        limit = std::min(limit, (max_jerk - used) / max_linear_d1);
      }
      if (max_angular_d1 > kSmallPositive) {
        const double used =
            max_angular_d3 * scalar_velocity * scalar_velocity *
                scalar_velocity +
            3.0 * max_angular_d2 * scalar_velocity * scalar_acceleration;
        limit = std::min(
            limit, (max_angular_jerk - used) / max_angular_d1);
      }
      return limit;
    };

    double scalar_max_acceleration = 0.0;
    double scalar_max_jerk = 0.0;
    bool scalar_limits_found = false;
    for (int iteration = 0; iteration < 32; ++iteration) {
      scalar_max_acceleration =
          scalarAccelerationLimit(scalar_max_velocity);
      if (scalar_max_acceleration > 1e-4) {
        scalar_max_jerk = scalarJerkLimit(
            scalar_max_velocity, scalar_max_acceleration);
        if (scalar_max_jerk > 1e-4) {
          scalar_limits_found = true;
          break;
        }
      }
      scalar_max_velocity *= 0.8;
    }
    if (!scalar_limits_found) {
      return {};
    }

    segment_limits[segment].max_velocity = std::max(
        kPathLimitSamplingMargin * scalar_max_velocity, 1e-4);
    segment_limits[segment].max_acceleration = std::max(
        kPathLimitSamplingMargin *
            std::min(scalar_max_acceleration, max_acceleration),
        1e-4);
    segment_limits[segment].max_jerk = std::max(
        kPathLimitSamplingMargin * std::min(scalar_max_jerk, max_jerk),
        1e-4);
  }

  // Share a nonzero scalar velocity between adjacent C3 path segments. A
  // forward/backward pass makes each boundary rate acceleration-reachable.
  // At a true geometric reversal the exact waypoint is a cusp, so it must be
  // crossed at zero rate unless the caller supplies a blended route.
  std::vector<double> boundary_rates(points.size(), 0.0);
  for (std::size_t i = 1; i + 1 < points.size(); ++i) {
    boundary_rates[i] = std::min(
        segment_limits[i - 1].max_velocity,
        segment_limits[i].max_velocity);

    if (direction_reversals[i]) {
      boundary_rates[i] = 0.0;
    }
  }
  for (std::size_t i = 0; i + 1 < boundary_rates.size(); ++i) {
    const double segment_length = u[i + 1] - u[i];
    const double reachable_rate = std::sqrt(
        boundary_rates[i] * boundary_rates[i] +
        2.0 * segment_limits[i].max_acceleration * segment_length);
    boundary_rates[i + 1] =
        std::min(boundary_rates[i + 1], reachable_rate);
  }
  for (std::size_t i = boundary_rates.size() - 1; i > 0; --i) {
    const std::size_t segment = i - 1;
    const double segment_length = u[i] - u[segment];
    const double reachable_rate = std::sqrt(
        boundary_rates[i] * boundary_rates[i] +
        2.0 * segment_limits[segment].max_acceleration * segment_length);
    boundary_rates[segment] =
        std::min(boundary_rates[segment], reachable_rate);
  }

  auto makeSegmentTrajectory = [&segment_limits, &boundary_rates, &u](
                                   std::size_t segment,
                                   ruckig::Trajectory<1>* trajectory) {
    if (trajectory == nullptr) {
      return false;
    }
    ruckig::Ruckig<1> otg;
    ruckig::InputParameter<1> input;
    input.current_position = {0.0};
    input.current_velocity = {boundary_rates[segment]};
    input.current_acceleration = {0.0};
    input.target_position = {u[segment + 1] - u[segment]};
    input.target_velocity = {boundary_rates[segment + 1]};
    input.target_acceleration = {0.0};
    input.max_velocity = {segment_limits[segment].max_velocity};
    input.max_acceleration = {segment_limits[segment].max_acceleration};
    input.max_jerk = {segment_limits[segment].max_jerk};
    return otg.calculate(input, *trajectory) >= ruckig::Result::Working;
  };

  // Jerk can make an acceleration-reachable boundary pair infeasible over a
  // short section. Reduce only through-waypoint rates until all local Ruckig
  // problems are feasible. Rest-to-rest remains the deterministic fallback.
  bool all_segments_feasible = false;
  for (int attempt = 0; attempt < 32; ++attempt) {
    all_segments_feasible = true;
    for (std::size_t segment = 0; segment < segment_limits.size(); ++segment) {
      ruckig::Trajectory<1> trajectory;
      if (!makeSegmentTrajectory(segment, &trajectory)) {
        all_segments_feasible = false;
        break;
      }
    }
    if (all_segments_feasible) {
      break;
    }
    for (std::size_t i = 1; i + 1 < boundary_rates.size(); ++i) {
      boundary_rates[i] *= 0.8;
    }
  }
  if (!all_segments_feasible) {
    return {};
  }

  CartesianTrajectorySample first;
  first.t = 0.0;
  first.p = points.front();
  first.q = orientations.front();
  samples.push_back(first);

  double elapsed_time = 0.0;
  for (std::size_t segment = 0; segment < segment_limits.size(); ++segment) {
    ruckig::Trajectory<1> trajectory;
    if (!makeSegmentTrajectory(segment, &trajectory)) {
      return {};
    }
    const double duration = std::max(trajectory.get_duration(), dt);
    const int n_steps =
        std::max(1, static_cast<int>(std::ceil(duration / dt)));
    for (int i = 0; i < n_steps; ++i) {
      const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
      std::array<double, 1> local_s{}, ds_arr{}, dds_arr{};
      trajectory.at_time(tau, local_s, ds_arr, dds_arr);

      const double s_path = std::clamp(
          u[segment] + local_s[0], u[segment], u[segment + 1]);
      const double ds_path = std::max(0.0, ds_arr[0]);
      const SmoothPathSample path_sample =
          sampleSepticHermitePath(points, u, tangents, s_path);
      const SmoothPathSample orientation_sample =
          sampleSepticHermitePath(
              rotation_vectors, u, rotation_tangents, s_path);
      const So3AngularPathDerivatives angular_derivatives =
          rotationVectorToWorldAngularPathDerivatives(
              orientation_sample.p,
              orientation_sample.dp_ds,
              orientation_sample.d2p_ds2,
              orientation_sample.d3p_ds3);

      CartesianTrajectorySample sample;
      sample.t = elapsed_time + tau;
      sample.p = path_sample.p;
      sample.dp = path_sample.dp_ds * ds_path;
      sample.ddp =
          path_sample.d2p_ds2 * ds_path * ds_path +
          path_sample.dp_ds * dds_arr[0];
      sample.q = rotationVectorToQuaternion(orientation_sample.p) * q_ref;
      sample.q.normalize();
      sample.w = angular_derivatives.omega_ds * ds_path;
      sample.dw =
          angular_derivatives.domega_ds2 * ds_path * ds_path +
          angular_derivatives.omega_ds * dds_arr[0];
      const bool final_sample =
          segment + 1 == segment_limits.size() && i + 1 == n_steps;
      if (final_sample) {
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
    elapsed_time += duration;
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
  const double start_path_time = std::clamp(
      min_path_time, timed_path.front().t, timed_path.back().t);
  const CartesianTrajectorySample path_start =
      sampleTimedPathAt(timed_path, start_path_time);
  const double estimated_start_rate =
      estimatePathRateAtSample(planning_start, path_start, max_rate);
  const double start_rate =
      config.initial_path_rate >= 0.0
          ? std::clamp(config.initial_path_rate, 0.0, max_rate)
          : estimated_start_rate;
  const double estimated_start_acceleration =
      estimatePathAccelerationAtSample(
          planning_start, path_start, start_rate, max_accel);
  const double requested_start_acceleration =
      std::isfinite(config.initial_path_acceleration)
          ? config.initial_path_acceleration
          : estimated_start_acceleration;
  const double start_acceleration =
      makeNonreversingInitialPathAcceleration(
          start_rate, requested_start_acceleration, max_accel, max_jerk);
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
  input.current_acceleration = {start_acceleration};
  input.target_velocity = {target_rate};
  input.target_acceleration = {0.0};
  input.max_velocity = {max_rate};
  input.min_velocity = {0.0};
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

  double previous_path_time = start_path_time;
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

    const double path_time = std::clamp(
        std::max(previous_path_time, s_arr[0]),
        timed_path.front().t,
        timed_path.back().t);
    const bool at_path_end = path_time >= timed_path.back().t - kMinDt;
    samples.push_back(retimeTimedPathSample(
        timed_path,
        path_time,
        at_path_end ? 0.0 : std::clamp(ds_arr[0], 0.0, max_rate),
        at_path_end ? 0.0 : dds_arr[0]));
    previous_path_time = path_time;
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
  const double estimated_start_rate =
      estimatePathRateAtSample(brake_start, path_start, max_rate);
  const double start_rate =
      config.initial_path_rate >= 0.0
          ? std::clamp(config.initial_path_rate, 0.0, max_rate)
          : estimated_start_rate;
  const double estimated_start_acceleration =
      estimatePathAccelerationAtSample(
          brake_start, path_start, start_rate, max_accel);
  const double requested_start_acceleration =
      std::isfinite(config.initial_path_acceleration)
          ? config.initial_path_acceleration
          : estimated_start_acceleration;
  const double start_acceleration =
      makeNonreversingInitialPathAcceleration(
          start_rate,
          requested_start_acceleration,
          max_accel,
          max_jerk);

  ruckig::Ruckig<1> otg;
  ruckig::InputParameter<1> input;
  ruckig::Trajectory<1> trajectory;
  input.control_interface = ruckig::ControlInterface::Velocity;
  input.current_position = {start_path_time};
  input.current_velocity = {start_rate};
  input.current_acceleration = {start_acceleration};
  input.target_velocity = {0.0};
  input.target_acceleration = {0.0};
  input.max_velocity = {max_rate};
  input.min_velocity = {0.0};
  input.max_acceleration = {max_accel};
  input.max_jerk = {max_jerk};

  const auto result = otg.calculate(input, trajectory);
  if (result < ruckig::Result::Working) {
    return samples;
  }

  const double duration = std::max(trajectory.get_duration(), dt);
  const int n_steps = std::max(1, static_cast<int>(std::ceil(duration / dt)));
  samples.reserve(static_cast<std::size_t>(n_steps));

  double previous_path_time = start_path_time;
  for (int i = 0; i < n_steps; ++i) {
    const double tau = std::min(static_cast<double>(i + 1) * dt, duration);
    std::array<double, 1> s_arr{}, ds_arr{}, dds_arr{};
    trajectory.at_time(tau, s_arr, ds_arr, dds_arr);
    const double sample_path_time = std::clamp(
        std::max(previous_path_time, s_arr[0]),
        timed_path.front().t,
        timed_path.back().t);
    CartesianTrajectorySample s = retimeTimedPathSample(
        timed_path,
        sample_path_time,
        std::clamp(ds_arr[0], 0.0, max_rate),
        dds_arr[0]);
    previous_path_time = sample_path_time;
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
