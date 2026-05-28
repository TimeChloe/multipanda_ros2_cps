// Copyright (c) 2026
// Reachable Cartesian Impedance Controller
//
// SARA-style monitored execution with local error-tube verification.
// Intended prefix is supplied by the trajectory generator package.
//
// Modified version:
//   - Keeps K_runtime_ / D_runtime_ as physical Cartesian stiffness/damping.
//   - Fixes use_dynamic_consistent_impedance_ == true branch:
//       F = Lambda * (xddot_des - Jdot*dq) - K*e - D*edot
//   - Adds filtered finite-difference Jdot*dq compensation.
//   - Uses dynamically consistent nullspace projection when true branch is active.
//   - Updates monitor prediction so true branch remains consistent with physical K/D.
//
// NOTE:
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <franka/model.h>
#include <franka/robot_state.h>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>
#include <cps_controllers/reachable_cartesian_math.hpp>
#include <cps_trajectory_generators/reachable_cartesian_trajectory.hpp>

namespace {

constexpr double kMinDt = 1e-6;
constexpr double kSmallPositive = 1e-9;
constexpr const char* kDefaultErrorLogRootDir =
    "/home/developer/multipanda_ws/src/data_log";
constexpr const char* kDefaultErrorLogFileName =
    "reachable_cartesian_impedance_validation.csv";

using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;
using Matrix37d = Eigen::Matrix<double, 3, 7>;
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using CartesianTrajectorySample = cps_trajectory_generators::CartesianTrajectorySample;
using LocalCartesianReplanConfig = cps_trajectory_generators::LocalCartesianReplanConfig;
using ReferenceTrajectoryType = cps_trajectory_generators::ReferenceTrajectoryType;
using TaskRefPose = cps_trajectory_generators::TaskRefPose;
using cps_trajectory_generators::estimateLinePathTimeFromZ;
using cps_trajectory_generators::makeLocalCartesianReplan;
using cps_trajectory_generators::makeReferencePose;
using cps_trajectory_generators::parseTrajectoryTypeOrDefault;

Matrix3d skewSymmetric(const Vector3d& v) {
  Matrix3d S;
  S << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return S;
}

inline int daysInMonth(int year, int month) {
  static constexpr int kDaysByMonth[] = {
      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month == 2) {
    const bool leap_year =
        ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0);
    return leap_year ? 29 : 28;
  }
  return kDaysByMonth[month - 1];
}

inline int weekdayUtc(int year, int month, int day) {
  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon = month - 1;
  tm.tm_mday = day;
  tm.tm_hour = 12;
  time_t t = timegm(&tm);
  std::tm out{};
  gmtime_r(&t, &out);
  return out.tm_wday;
}

inline int lastSundayOfMonth(int year, int month) {
  int day = daysInMonth(year, month);
  while (weekdayUtc(year, month, day) != 0) {
    --day;
  }
  return day;
}

inline time_t utcTime(int year, int month, int day, int hour) {
  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon = month - 1;
  tm.tm_mday = day;
  tm.tm_hour = hour;
  return timegm(&tm);
}

inline int berlinUtcOffsetMinutes(std::chrono::system_clock::time_point now) {
  const time_t now_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
  gmtime_r(&now_time_t, &utc_tm);
  const int year = utc_tm.tm_year + 1900;

  const time_t dst_start =
      utcTime(year, 3, lastSundayOfMonth(year, 3), 1);
  const time_t dst_end =
      utcTime(year, 10, lastSundayOfMonth(year, 10), 1);

  return (now_time_t >= dst_start && now_time_t < dst_end) ? 120 : 60;
}

inline std::string makeBerlinTimestampForDirectoryName() {
  const auto now = std::chrono::system_clock::now();
  const int utc_offset_minutes = berlinUtcOffsetMinutes(now);
  const auto adjusted_now = now + std::chrono::minutes(utc_offset_minutes);
  const auto adjusted_time_t = std::chrono::system_clock::to_time_t(adjusted_now);
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch())
                          .count() %
                      1000000;

  std::tm tm{};
  gmtime_r(&adjusted_time_t, &tm);

  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y%m%d_%H%M%S")
     << "_" << std::setw(6) << std::setfill('0') << micros;
  return ss.str();
}

inline std::string sanitizedFileNameOrDefault(
    const std::string& file_name,
    const std::string& default_file_name) {
  if (file_name.empty()) {
    return default_file_name;
  }

  const std::filesystem::path path(file_name);
  const std::string sanitized = path.filename().string();
  return sanitized.empty() ? default_file_name : sanitized;
}

inline std_msgs::msg::ColorRGBA makeColor(float r, float g, float b, float a) {
  std_msgs::msg::ColorRGBA c;
  c.r = r; c.g = g; c.b = b; c.a = a;
  return c;
}

inline geometry_msgs::msg::Point toPoint(const Vector3d& p) {
  geometry_msgs::msg::Point msg;
  msg.x = p.x(); msg.y = p.y(); msg.z = p.z();
  return msg;
}

inline geometry_msgs::msg::Quaternion toQuatMsg(const Quaterniond& q) {
  geometry_msgs::msg::Quaternion msg;
  msg.x = q.x(); msg.y = q.y(); msg.z = q.z(); msg.w = q.w();
  return msg;
}

inline Quaterniond rotationMatrixToQuaternion(const Matrix3d& R) {
  Quaterniond q(R); q.normalize();
  return q;
}

inline Matrix3d makePlaneFrameFromNormal(const Vector3d& n_in) {
  const Vector3d n = n_in.normalized();
  Vector3d ref = std::abs(n.z()) < 0.9 ? Vector3d::UnitZ() : Vector3d::UnitX();
  Vector3d x = ref.cross(n).normalized();
  Vector3d y = n.cross(x).normalized();
  Matrix3d R;
  R.col(0) = x; R.col(1) = y; R.col(2) = n;
  return R;
}

}  // namespace

namespace cps_controllers {

ReachableCartesianImpedanceController::~ReachableCartesianImpedanceController() {
  stopSafetyMonitorWorker();
}

// ============================================================================
// Helper: matrix rate limit & runtime gain update
// ============================================================================
Matrix6d ReachableCartesianImpedanceController::applyMatrixRateLimit(
    const Matrix6d& current, const Matrix6d& target,
    double rate_limit, double dt) const {
  const double step = std::max(rate_limit, 0.0) * std::max(dt, kMinDt);
  Matrix6d out = current;
  for (int i = 0; i < out.rows(); ++i)
    for (int j = 0; j < out.cols(); ++j) {
      const double delta = std::clamp(target(i, j) - current(i, j), -step, step);
      out(i, j) = current(i, j) + delta;
    }
  return out;
}

void ReachableCartesianImpedanceController::updateRuntimeGains(const Matrix6d& K_target,
                                                               const Matrix6d& D_target,
                                                               double dt) {
  K_runtime_ = applyMatrixRateLimit(K_runtime_, K_target, k_rate_limit_, dt);
  D_runtime_ = applyMatrixRateLimit(D_runtime_, D_target, d_rate_limit_, dt);
}

Matrix6d ReachableCartesianImpedanceController::computeDampingFromStiffness(
    const Matrix6d& K,
    double pos_damping_scale,
    double rot_damping_scale) const {
  Matrix6d D = Matrix6d::Zero();

  const Matrix3d Kp = K.topLeftCorner<3, 3>();
  const Matrix3d Kr = K.bottomRightCorner<3, 3>();

  for (int i = 0; i < 3; ++i) {
    D(i, i) = pos_damping_scale *
              2.0 * std::sqrt(std::max(Kp(i, i), 0.0));
    D(i + 3, i + 3) = rot_damping_scale *
                      2.0 * std::sqrt(std::max(Kr(i, i), 0.0));
  }

  return D;
}


double ReachableCartesianImpedanceController::computeConservativeNormalAccelPositiveBound(
    double m_eff_n, double e_n_abs, double /*v_n_abs*/, const Matrix6d& K_used) const {
  const double m_safe = std::max(m_eff_n, kSmallPositive);
  const double k_n_up = human_workspace_.normalStiffness(K_used);
  const double a_from_stiffness = (k_n_up * e_n_abs) / m_safe;
  const double a_from_tau_rate = torque_to_accel_gain_ * tau_rate_limit_ * kMinDt;
  const double a_from_uncertainty = std::max(model_accel_uncertainty_, 0.0);
  return std::max(0.0, a_from_stiffness + a_from_tau_rate + a_from_uncertainty);
}

double ReachableCartesianImpedanceController::computeConservativeNormalBrakeAccelLowerBound(
    double m_eff_n, double v_n_abs, const Matrix6d& D_used) const {
  const double m_safe = std::max(m_eff_n, kSmallPositive);
  const double d_n_low = human_workspace_.normalDamping(D_used);
  const double a_from_damping = (d_n_low * v_n_abs) / m_safe;
  const double a_tau_loss = 0.5 * torque_to_accel_gain_ * tau_rate_limit_ * kMinDt;
  return std::max(0.0, a_from_damping - a_tau_loss);
}

// path retiming helpers ------------------------------------------------------
double ReachableCartesianImpedanceController::estimatePathParameterTimeFromCurrentState(
    const Vector3d& current_position, double nominal_guess_time) const {
  const Vector3d p0 = desired_position_;
  const Matrix3d R0 = desired_orientation_.toRotationMatrix();
  const auto traj_type = parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);
  const double window = std::max(path_retiming_search_window_sec_, shield_plan_dt_);
  const int n_steps = std::max(5, path_retiming_search_steps_);
  const double t_min = std::max(0.0, nominal_guess_time - window);
  const double t_max = std::max(t_min, nominal_guess_time + window);
  double best_t = nominal_guess_time;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n_steps; ++i) {
    const double alpha = static_cast<double>(i) / static_cast<double>(n_steps - 1);
    const double t = t_min + alpha * (t_max - t_min);
    const TaskRefPose ref = makeReferencePose(t, p0, R0, traj_type);
    const double cost = (ref.p - current_position).squaredNorm();
    if (cost < best_cost) { best_cost = cost; best_t = t; }
  }
  return std::max(0.0, best_t);
}

double ReachableCartesianImpedanceController::estimatePathTimeRateFromCurrentState(
    double path_time_anchor, const Vector3d& current_linear_velocity) const {
  const Vector3d p0 = desired_position_;
  const Matrix3d R0 = desired_orientation_.toRotationMatrix();
  const auto traj_type = parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);
  const TaskRefPose ref = makeReferencePose(path_time_anchor, p0, R0, traj_type);
  const double denom = std::max(ref.dp.squaredNorm(), kSmallPositive);
  double rate = 0.0;
  if (denom > 1e-8) rate = current_linear_velocity.dot(ref.dp) / denom;
  if (!std::isfinite(rate)) rate = 0.0;
  return std::clamp(rate, path_time_rate_min_, path_time_rate_max_);
}

ReachableCartesianImpedanceController::PathState
ReachableCartesianImpedanceController::propagateOnlinePathState(
    const Vector3d& current_position, const Vector3d& current_linear_velocity,
    double nominal_guess_time, double dtp) const {
  PathState out;
  out.t_path = estimatePathParameterTimeFromCurrentState(current_position, nominal_guess_time);
  const double rate0 = estimatePathTimeRateFromCurrentState(out.t_path, current_linear_velocity);
  const double rate_err = path_time_rate_target_ - rate0;
  const double accel_cmd = std::clamp(rate_err / std::max(dtp, kMinDt),
                                      -path_time_acc_limit_, path_time_acc_limit_);
  out.accel = accel_cmd;
  out.rate  = std::clamp(rate0 + dtp * accel_cmd, path_time_rate_min_, path_time_rate_max_);
  out.t_path = std::max(0.0, out.t_path + dtp * out.rate);
  return out;
}

// ============================================================================
// RViz diagnostics
// ============================================================================
void ReachableCartesianImpedanceController::publishRvizDiagnostics(
    double wall_time, const Vector3d& current_position,
    const Vector3d& desired_position_cur, const Vector6d& ee_twist,
    const MonitorResult& monitor) {
  (void)ee_twist;
  if (!rviz_enable_markers_ || !rviz_marker_pub_) return;
  ++rviz_publish_counter_;
  if ((rviz_publish_counter_ % std::max(1, rviz_marker_decimation_)) != 0) return;

  MarkerArray arr;
  const auto stamp = get_node()->now();
  const std::string& frame_id = rviz_frame_id_;
  const Vector3d n = human_workspace_.normal();
  auto setupMarker = [&](Marker& m, int id, const std::string& ns, int type) {
    m.header.frame_id = frame_id; m.header.stamp = stamp;
    m.ns = ns; m.id = id; m.type = type; m.action = Marker::ADD;
    m.lifetime = rclcpp::Duration::from_seconds(rviz_marker_lifetime_sec_);
  };

  const Matrix3d R_plane = makePlaneFrameFromNormal(n);
  const Quaterniond q_plane = rotationMatrixToQuaternion(R_plane);

  { Marker m; setupMarker(m, 0, "reachable_plane", Marker::CUBE);
    m.pose.position = toPoint(human_workspace_.center());
    m.pose.orientation = toQuatMsg(q_plane);
    m.scale.x = rviz_plane_size_; m.scale.y = rviz_plane_size_; m.scale.z = rviz_plane_thickness_;
    m.color = makeColor(0.1f, 0.6f, 1.0f, 0.12f);
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 1, "reachable_plane_normal", Marker::ARROW);
    m.scale.x = rviz_arrow_shaft_diameter_; m.scale.y = rviz_arrow_head_diameter_; m.scale.z = rviz_arrow_head_length_;
    m.color = makeColor(0.1f, 0.6f, 1.0f, 0.9f);
    const Vector3d p0 = human_workspace_.center();
    const Vector3d p1 = p0 + rviz_normal_arrow_length_ * n;
    m.points.push_back(toPoint(p0)); m.points.push_back(toPoint(p1));
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 2, "reachable_hand_motion_sphere", Marker::SPHERE);
    m.pose.position = toPoint(human_workspace_.center()); m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 2.0 * human_workspace_.motionRadius();
    m.color = makeColor(1.0f, 0.8f, 0.1f, 0.20f);
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 3, "reachable_hand_inflated_sphere", Marker::SPHERE);
    m.pose.position = toPoint(human_workspace_.center()); m.pose.orientation.w = 1.0;
    const double inflated_radius = human_workspace_.inflatedHandRadius();
    m.scale.x = m.scale.y = m.scale.z = 2.0 * inflated_radius;
    m.color = makeColor(1.0f, 0.3f, 0.2f, 0.10f);
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 4, "reachable_ee_center", Marker::SPHERE);
    m.pose.position = toPoint(current_position); m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.03;
    m.color = makeColor(0.0f, 1.0f, 0.2f, 0.9f);
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 5, "reachable_ee_radius", Marker::SPHERE);
    m.pose.position = toPoint(current_position); m.pose.orientation.w = 1.0;
    const double inflated_r = ee_collision_radius_ + monitor.current_pos_error_radius;
    m.scale.x = m.scale.y = m.scale.z = 2.0 * inflated_r;
    m.color = makeColor(0.0f, 1.0f, 0.2f, 0.15f);
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 6, "reachable_desired", Marker::SPHERE);
    m.pose.position = toPoint(desired_position_cur); m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.025;
    m.color = makeColor(1.0f, 1.0f, 0.0f, 0.85f);
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 7, "reachable_vn_now", Marker::ARROW);
    m.scale.x = rviz_arrow_shaft_diameter_; m.scale.y = rviz_arrow_head_diameter_; m.scale.z = rviz_arrow_head_length_;
    const double vn = monitor.v_n_now;
    const Vector3d p0 = current_position;
    const Vector3d p1 = p0 + rviz_velocity_arrow_scale_ * vn * n;
    m.points.push_back(toPoint(p0)); m.points.push_back(toPoint(p1));
    m.color = (vn >= 0.0) ? makeColor(1.0f, 0.5f, 0.0f, 0.95f) : makeColor(0.8f, 0.2f, 1.0f, 0.95f);
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 8, "reachable_vn_ub", Marker::ARROW);
    m.scale.x = rviz_arrow_shaft_diameter_; m.scale.y = rviz_arrow_head_diameter_; m.scale.z = rviz_arrow_head_length_;
    m.color = makeColor(1.0f, 0.0f, 0.0f, 0.95f);
    const Vector3d p0 = current_position + Vector3d(0.0, 0.0, rviz_ub_arrow_z_offset_);
    const Vector3d p1 = p0 + rviz_velocity_arrow_scale_ * monitor.worst_case_v_n_ub * n;
    m.points.push_back(toPoint(p0)); m.points.push_back(toPoint(p1));
    arr.markers.push_back(m); }
  { Marker m; setupMarker(m, 9, "reachable_nominal_contact", Marker::SPHERE);
    if (monitor.nominal_contact_sample_found) {
      m.pose.position = toPoint(monitor.nominal_contact_point_world); m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = 0.035;
      m.color = makeColor(1.0f, 0.0f, 1.0f, 0.95f);
      arr.markers.push_back(m);
    } else { m.action = Marker::DELETE; arr.markers.push_back(m); } }
  { Marker m; setupMarker(m, 10, "reachable_status_text", Marker::TEXT_VIEW_FACING);
    m.pose.position = toPoint(current_position + Vector3d(0.0, 0.0, rviz_text_z_offset_));
    m.pose.orientation.w = 1.0;
    m.scale.z = rviz_text_scale_;
    m.color = makeColor(1.0f, 1.0f, 1.0f, 0.95f);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << "mode=" << static_cast<int>(mode_)
       << "  d=" << monitor.plane_distance_now
       << "  rp=" << monitor.current_pos_error_radius
       << "  rv=" << monitor.current_vel_error_radius
       << "  Tn_nom=" << monitor.Tn_contact_nominal
       << "  Tn_ub=" << monitor.worst_case_Tn_ub
       << "  V_ub=" << monitor.worst_case_V_potential_ub
       << "  hV=" << monitor.h_clamping_energy
       << "  trig=" << static_cast<int>(monitor.predicted_trigger)
       << "  EF=" << monitor.terminal_energy_ub
       << "  hF=" << monitor.h_terminal_energy
       << "  wall_t=" << wall_time;
    m.text = ss.str();
    arr.markers.push_back(m); }

  rviz_marker_pub_->publish(arr);
}

// ============================================================================
// Interface configurations
// ============================================================================
controller_interface::InterfaceConfiguration
ReachableCartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i)
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  return config;
}

controller_interface::InterfaceConfiguration
ReachableCartesianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& name : franka_robot_model_->get_state_interface_names())
    config.names.push_back(name);
  return config;
}

// ============================================================================
// Sample factory helpers
// ============================================================================
ImpedanceSample ReachableCartesianImpedanceController::makeNominalSample(
    double nominal_time, double path_rate, double path_accel,
    const Matrix6d& K_target, const Matrix6d& D_target) const {
  const Vector3d p0 = desired_position_;
  const Matrix3d R0 = desired_orientation_.toRotationMatrix();
  const auto traj_type = parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);
  const TaskRefPose ref = makeReferencePose(nominal_time, p0, R0, traj_type);
  ImpedanceSample s;
  s.t = nominal_time;
  s.p = ref.p;
  s.dp = ref.dp * path_rate;
  s.ddp = ref.ddp * path_rate * path_rate + ref.dp * path_accel;
  s.q = Quaterniond(ref.R); s.q.normalize();
  s.w.setZero(); s.dw.setZero();
  s.K = K_target; s.D = D_target;
  s.failsafe = false;
  return s;
}

ImpedanceSample ReachableCartesianImpedanceController::makeFrozenFailsafeSample(
    double nominal_time, const ImpedanceSample& freeze_sample,
    const Matrix6d& K_target, const Matrix6d& D_target) const {
  ImpedanceSample s;
  s.t = nominal_time;
  s.p = freeze_sample.p;
  s.dp.setZero(); s.ddp.setZero();
  s.q = freeze_sample.q; s.q.normalize();
  s.w.setZero(); s.dw.setZero();
  s.K = K_target; s.D = D_target;
  s.failsafe = true;
  return s;
}

ImpedanceSample ReachableCartesianImpedanceController::makeEmergencyStopCommand(
    const Vector3d& current_position, const Quaterniond& current_orientation,
    double wall_time) const {
  ImpedanceSample emergency;
  emergency.t = wall_time;
  emergency.p = current_position;
  emergency.dp.setZero(); emergency.ddp.setZero();
  emergency.q = current_orientation; emergency.q.normalize();
  emergency.w.setZero(); emergency.dw.setZero();
  emergency.K = K_f_target_; emergency.D = D_f_target_;
  emergency.failsafe = true;
  return emergency;
}

// ============================================================================
// Cache for the verified intended prefix
// ============================================================================
ImpedanceSample ReachableCartesianImpedanceController::getNextIntendedCommandFromCache(
    bool advance_index) {
  if (!last_verified_plan_.valid || last_verified_plan_.intended.empty()) {
    return ImpedanceSample{};
  }
  ImpedanceSample cmd = last_verified_plan_.intended.front();
  (void)advance_index;
  return cmd;
}

ImpedanceSample ReachableCartesianImpedanceController::getNextFailsafeCommandFromCache(
    bool advance_index) {
  if (!last_verified_plan_.valid || last_verified_plan_.failsafe.empty())
    return ImpedanceSample{};
  const std::size_t idx = std::min(
      last_verified_plan_.failsafe_exec_index,
      last_verified_plan_.failsafe.size() - 1);
  ImpedanceSample cmd = last_verified_plan_.failsafe[idx];
  if (advance_index &&
      last_verified_plan_.failsafe_exec_index + 1 < last_verified_plan_.failsafe.size()) {
    ++last_verified_plan_.failsafe_exec_index;
  }
  return cmd;
}

// ============================================================================
// Generate intended prefix from the trajectory generator package
// ============================================================================
std::vector<ImpedanceSample>
ReachableCartesianImpedanceController::makeIntendedBufferFromReplanner(
    double nominal_guess_time,
    const ImpedanceSample& planning_start_command) const {
  const auto traj_type =
      parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);

  CartesianTrajectorySample planning_start;
  planning_start.t = planning_start_command.t;
  planning_start.p = planning_start_command.p;
  planning_start.dp = planning_start_command.dp;
  planning_start.ddp = planning_start_command.ddp;
  planning_start.q = planning_start_command.q;
  planning_start.q.normalize();
  planning_start.w = planning_start_command.w;
  planning_start.dw = planning_start_command.dw;

  LocalCartesianReplanConfig config;
  config.horizon_steps = local_replan_horizon_steps_;
  config.dt = local_replan_dt_;
  config.path_lookahead_sec = local_path_lookahead_sec_;
  config.max_velocity = local_replan_max_velocity_;
  config.max_acceleration = local_replan_max_acceleration_;
  config.max_jerk = local_replan_max_jerk_;

  const auto planned_samples = makeLocalCartesianReplan(
      nominal_guess_time,
      planning_start,
      desired_position_,
      desired_orientation_,
      traj_type,
      config);

  std::vector<ImpedanceSample> intended_buffer;
  intended_buffer.reserve(planned_samples.size());

  for (const auto& planned_sample : planned_samples) {
    ImpedanceSample s;
    s.t = planned_sample.t;
    s.p = planned_sample.p;
    s.dp = planned_sample.dp;
    s.ddp = planned_sample.ddp;
    s.q = planned_sample.q;
    s.q.normalize();
    s.w = planned_sample.w;
    s.dw = planned_sample.dw;
    s.K = K_nominal_;
    s.D = D_nominal_;
    s.failsafe = false;

    intended_buffer.push_back(s);
  }

  return intended_buffer;
}

// ============================================================================
// Refill the intended cache from the trajectory generator package
// ============================================================================
bool ReachableCartesianImpedanceController::refillIntendedBufferFromReplanner(
    double nominal_guess_time,
    const ImpedanceSample& planning_start_command) {
  intended_buffer_.clear();
  intended_buffer_index_ = 0;
  intended_buffer_valid_ = false;

  intended_buffer_ =
      makeIntendedBufferFromReplanner(nominal_guess_time, planning_start_command);

  intended_buffer_valid_ = !intended_buffer_.empty();
  return intended_buffer_valid_;
}

// ============================================================================
// Build a single-step candidate plan
// ============================================================================
VerifiedPlan ReachableCartesianImpedanceController::buildSingleStepCandidatePlan(
    double wall_time,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    const ImpedanceSample& next_intended) const {
  (void)inertia;
  (void)Jv;

  VerifiedPlan plan;
  plan.valid = false;
  plan.generated_wall_time = wall_time;
  plan.intended_exec_index = 0;
  plan.failsafe_exec_index = 0;

  plan.anchor.t = 0.0;
  plan.anchor.p = current_position;
  plan.anchor.dp = ee_twist.head<3>();
  plan.anchor.ddp.setZero();
  plan.anchor.q = current_orientation;
  plan.anchor.q.normalize();
  plan.anchor.w.setZero();
  plan.anchor.dw.setZero();
  plan.anchor.K = K_runtime;
  plan.anchor.D = D_runtime;
  plan.anchor.failsafe = false;

  ImpedanceSample step = next_intended;
  step.t = plan.anchor.t + std::max(local_replan_dt_, kMinDt);
  step.K = K_nominal_;
  step.D = D_nominal_;
  step.failsafe = false;

  plan.intended.clear();
  plan.intended.push_back(step);

  const ImpedanceSample freeze_anchor = step;
  const int N_fs = std::max(1, shield_horizon_steps_);
  const double dtp_fs = std::max(shield_plan_dt_, kMinDt);

  auto fill_failsafe_prefix = [&](const Matrix6d& K_terminal) {
    plan.failsafe.clear();
    plan.failsafe.reserve(static_cast<std::size_t>(N_fs));

    for (int i = 0; i < N_fs; ++i) {
      const double tk =
          freeze_anchor.t + static_cast<double>(i + 1) * dtp_fs;

      const double s_blend =
          static_cast<double>(i + 1) / static_cast<double>(N_fs);

      const Matrix6d K_sched =
          (1.0 - s_blend) * K_nominal_ + s_blend * K_terminal;

      const Matrix6d D_sched =
          computeDampingFromStiffness(
              K_sched,
              failsafe_pos_damping_scale_,
              failsafe_rot_damping_scale_);

      plan.failsafe.push_back(
          makeFrozenFailsafeSample(tk, freeze_anchor, K_sched, D_sched));
    }
  };

  // Unified fail-safe stiffness policy:
  // The terminal stiffness is exactly the configured failsafe stiffness.
  // No online k_budget / k_selected update is performed here.
  fill_failsafe_prefix(K_f_target_);

  plan.valid = !plan.intended.empty() && !plan.failsafe.empty();
  return plan;
}

SafetyMonitorConfig ReachableCartesianImpedanceController::makeSafetyMonitorConfig(
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime) const {
  SafetyMonitorConfig config;
  config.human_workspace = human_workspace_;
  config.K_runtime = K_runtime;
  config.D_runtime = D_runtime;
  config.k_rate_limit = k_rate_limit_;
  config.d_rate_limit = d_rate_limit_;
  config.safe_collision_energy_joule = safe_collision_energy_joule_;
  config.clamping_energy_budget_joule = clamping_energy_budget_joule_;
  config.energy_budget_margin_joule = energy_budget_margin_joule_;
  config.ee_collision_radius = ee_collision_radius_;
  config.tracking_pos_error_bound = tracking_pos_error_bound_;
  config.tracking_vel_error_bound = tracking_vel_error_bound_;
  config.use_dynamic_consistent_impedance = use_dynamic_consistent_impedance_;
  return config;
}

Vector3d ReachableCartesianImpedanceController::collisionCenterOffsetWorld(
    const Quaterniond& orientation) const {
  return orientation.normalized() * ee_collision_center_offset_;
}

Vector6d ReachableCartesianImpedanceController::twistAtCollisionCenter(
    const Quaterniond& orientation,
    const Vector6d& flange_twist) const {
  Vector6d collision_twist = flange_twist;
  const Vector3d offset_world = collisionCenterOffsetWorld(orientation);
  collision_twist.head<3>() =
      flange_twist.head<3>() + flange_twist.tail<3>().cross(offset_world);
  return collision_twist;
}

VerifiedPlan ReachableCartesianImpedanceController::makeCollisionCenterPlanForMonitor(
    const VerifiedPlan& flange_plan) const {
  VerifiedPlan plan = flange_plan;

  auto move_sample_to_collision_center = [this](ImpedanceSample& sample) {
    const Vector3d offset_world = sample.q.normalized() * ee_collision_center_offset_;
    sample.p += offset_world;
    sample.dp += sample.w.cross(offset_world);
    sample.ddp += sample.dw.cross(offset_world) +
                  sample.w.cross(sample.w.cross(offset_world));
  };

  move_sample_to_collision_center(plan.anchor);
  for (auto& sample : plan.intended) {
    move_sample_to_collision_center(sample);
  }
  for (auto& sample : plan.failsafe) {
    move_sample_to_collision_center(sample);
  }

  return plan;
}

// ============================================================================
// verifyCandidatePlan
// ============================================================================
MonitorResult ReachableCartesianImpedanceController::verifyCandidatePlan(
    const VerifiedPlan& plan,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime) const {
  const VerifiedPlan collision_center_plan =
      makeCollisionCenterPlanForMonitor(plan);
  const Vector3d collision_center =
      current_position + collisionCenterOffsetWorld(current_orientation);
  const Vector6d collision_twist =
      twistAtCollisionCenter(current_orientation, ee_twist);

  return cps_safety_monitor::verifyReachablePlan(
      collision_center_plan,
      collision_center,
      collision_twist,
      inertia,
      Jv,
      makeSafetyMonitorConfig(K_runtime, D_runtime));
}

// ============================================================================
// Shield decision
// ============================================================================
ShieldDecision ReachableCartesianImpedanceController::computeShieldDecision(
    double wall_time, double nominal_guess_time,
    const Vector3d& current_position, const Quaterniond& current_orientation,
    const Vector6d& ee_twist, const Matrix7d& inertia, const Matrix37d& Jv) {
  ShieldDecision dec;

  if (parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_) ==
      ReferenceTrajectoryType::kLine) {
    constexpr double T_move = 1.5;
    constexpr double dz = -0.55;

    const double z_start = desired_position_.z();
    const double z_final = desired_position_.z() + dz;

    commanded_path_time_ = estimateLinePathTimeFromZ(
        current_position.z(),
        z_start,
        z_final,
        T_move);
  }

  auto verify_plan = [&](const VerifiedPlan& plan) {
    return verifyCandidatePlan(
        plan, current_position, current_orientation, ee_twist, inertia, Jv, K_runtime_, D_runtime_);
  };

  auto execute_last_verified_failsafe = [&]() {
    dec.executing_last_verified_failsafe = true;
    dec.candidate_verified = false;
    if (last_verified_plan_.valid && !last_verified_plan_.failsafe.empty()) {
      dec.command = getNextFailsafeCommandFromCache(true);
    } else {
      dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
    }
  };

  auto try_one_verification_with_fresh_buffer_if_needed = [&]() -> bool {
    if (!intended_buffer_valid_ || intended_buffer_index_ >= intended_buffer_.size()) {
      ImpedanceSample planning_start_command;

      planning_start_command.t = wall_time;

      if (last_commanded_sample_valid_) {
        planning_start_command.p = last_commanded_sample_.p;
        planning_start_command.q = last_commanded_sample_.q;
      } else {
        planning_start_command.p = current_position;
        planning_start_command.q = current_orientation;
      }
      planning_start_command.q.normalize();

      planning_start_command.dp = ee_twist.head<3>();
      planning_start_command.w = ee_twist.tail<3>();

      planning_start_command.ddp.setZero();
      planning_start_command.dw.setZero();

      planning_start_command.K = K_nominal_;
      planning_start_command.D = D_nominal_;
      planning_start_command.failsafe = false;

      const bool ok = refillIntendedBufferFromReplanner(
          nominal_guess_time,
          planning_start_command);

      if (!ok) {
        return false;
      }
    }

    const ImpedanceSample next_step = intended_buffer_[intended_buffer_index_];

    candidate_plan_ = buildSingleStepCandidatePlan(
        wall_time,
        current_position,
        current_orientation,
        ee_twist,
        inertia,
        Jv,
        K_runtime_,
        D_runtime_,
        next_step);
    candidate_plan_valid_ = candidate_plan_.valid;
    if (!candidate_plan_valid_) return false;

    dec.monitor = verify_plan(candidate_plan_);
    dec.candidate_verified = !enable_safety_monitor_ || !dec.monitor.predicted_trigger;
    return true;
  };

  const bool first_attempt = try_one_verification_with_fresh_buffer_if_needed();

  if (!first_attempt) {
    mode_ = SafetyMode::kFailsafe;

    intended_buffer_valid_ = false;
    intended_buffer_index_ = 0;
    intended_buffer_.clear();

    execute_last_verified_failsafe();
    return dec;
  }

  if (dec.candidate_verified) {
    last_verified_plan_ = candidate_plan_;
    last_verified_plan_.valid = true;
    last_verified_plan_.intended_exec_index = 0;
    last_verified_plan_.failsafe_exec_index = 0;
    candidate_plan_valid_ = false;

    ++intended_buffer_index_;

    if (parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_) ==
        ReferenceTrajectoryType::kLine) {
      constexpr double T_move = 1.5;
      constexpr double dz = -0.55;

      const double z_start = desired_position_.z();
      const double z_final = desired_position_.z() + dz;

      const double z_for_path_time =
          last_commanded_sample_valid_ ? last_commanded_sample_.p.z()
                                      : current_position.z();

      commanded_path_time_ = estimateLinePathTimeFromZ(
          z_for_path_time,
          z_start,
          z_final,
          T_move);
    }

    mode_ = SafetyMode::kNominal;

    dec.executing_last_verified_failsafe = false;
    dec.command = last_verified_plan_.intended.front();
    return dec;
  }

  intended_buffer_valid_ = false;
  intended_buffer_index_ = 0;

  const bool second_attempt = try_one_verification_with_fresh_buffer_if_needed();

  if (second_attempt && dec.candidate_verified) {
    last_verified_plan_ = candidate_plan_;
    last_verified_plan_.valid = true;
    last_verified_plan_.intended_exec_index = 0;
    last_verified_plan_.failsafe_exec_index = 0;
    candidate_plan_valid_ = false;

    ++intended_buffer_index_;

    if (parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_) ==
        ReferenceTrajectoryType::kLine) {
      constexpr double T_move = 1.5;
      constexpr double dz = -0.55;

      const double z_start = desired_position_.z();
      const double z_final = desired_position_.z() + dz;

      commanded_path_time_ = estimateLinePathTimeFromZ(
          last_verified_plan_.intended.front().p.z(),
          z_start,
          z_final,
          T_move);
    }

    mode_ = SafetyMode::kNominal;

    dec.executing_last_verified_failsafe = false;
    dec.command = last_verified_plan_.intended.front();
    return dec;
  }

  mode_ = SafetyMode::kFailsafe;
  execute_last_verified_failsafe();
  return dec;
}

ShieldDecision ReachableCartesianImpedanceController::computeShieldDecisionForAsyncInput(
    const AsyncMonitorInput& input,
    AsyncPlanningState& state) const {
  ShieldDecision dec;

  auto execute_last_verified_failsafe = [&]() {
    dec.executing_last_verified_failsafe = true;
    dec.candidate_verified = false;

    if (state.last_verified_plan.valid &&
        !state.last_verified_plan.failsafe.empty()) {
      const std::size_t idx = std::min(
          state.last_verified_plan.failsafe_exec_index,
          state.last_verified_plan.failsafe.size() - 1);

      dec.command = state.last_verified_plan.failsafe[idx];

      if (state.last_verified_plan.failsafe_exec_index + 1 <
          state.last_verified_plan.failsafe.size()) {
        ++state.last_verified_plan.failsafe_exec_index;
      }
    } else {
      dec.command = makeEmergencyStopCommand(
          input.current_position,
          input.current_orientation,
          input.wall_time);
    }
  };

  auto try_one_verification = [&]() -> bool {
    if (!state.intended_buffer_valid ||
        state.intended_buffer_index >= state.intended_buffer.size()) {
      ImpedanceSample planning_start_command;
      planning_start_command.t = input.wall_time;

      if (input.last_commanded_sample_valid) {
        planning_start_command.p = input.last_commanded_sample.p;
        planning_start_command.q = input.last_commanded_sample.q;
      } else {
        planning_start_command.p = input.current_position;
        planning_start_command.q = input.current_orientation;
      }
      planning_start_command.q.normalize();

      planning_start_command.dp = input.ee_twist.head<3>();
      planning_start_command.w = input.ee_twist.tail<3>();
      planning_start_command.ddp.setZero();
      planning_start_command.dw.setZero();
      planning_start_command.K = K_nominal_;
      planning_start_command.D = D_nominal_;
      planning_start_command.failsafe = false;

      state.intended_buffer =
          makeIntendedBufferFromReplanner(
              input.nominal_guess_time,
              planning_start_command);

      state.intended_buffer_index = 0;
      state.intended_buffer_valid = !state.intended_buffer.empty();

      if (!state.intended_buffer_valid) {
        return false;
      }
    }

    const ImpedanceSample next_step =
        state.intended_buffer[state.intended_buffer_index];

    VerifiedPlan candidate_plan = buildSingleStepCandidatePlan(
        input.wall_time,
        input.current_position,
        input.current_orientation,
        input.ee_twist,
        input.inertia,
        input.Jv,
        input.K_runtime,
        input.D_runtime,
        next_step);

    if (!candidate_plan.valid) {
      return false;
    }

    dec.monitor = verifyCandidatePlan(
        candidate_plan,
        input.current_position,
        input.current_orientation,
        input.ee_twist,
        input.inertia,
        input.Jv,
        input.K_runtime,
        input.D_runtime);

    dec.candidate_verified =
        !enable_safety_monitor_ || !dec.monitor.predicted_trigger;

    if (!dec.candidate_verified) {
      return true;
    }

    state.last_verified_plan = candidate_plan;
    state.last_verified_plan.valid = true;
    state.last_verified_plan.intended_exec_index = 0;
    state.last_verified_plan.failsafe_exec_index = 0;
    ++state.intended_buffer_index;

    dec.executing_last_verified_failsafe = false;
    dec.command = state.last_verified_plan.intended.front();

    return true;
  };

  const bool first_attempt = try_one_verification();

  if (!first_attempt) {
    state.intended_buffer_valid = false;
    state.intended_buffer_index = 0;
    state.intended_buffer.clear();
    execute_last_verified_failsafe();
    return dec;
  }

  if (dec.candidate_verified) {
    return dec;
  }

  state.intended_buffer_valid = false;
  state.intended_buffer_index = 0;

  const bool second_attempt = try_one_verification();

  if (second_attempt && dec.candidate_verified) {
    return dec;
  }

  execute_last_verified_failsafe();
  return dec;
}

void ReachableCartesianImpedanceController::publishAsyncMonitorInput(
    const AsyncMonitorInput& input) {
  if (!async_safety_monitor_ || !safety_monitor_worker_running_.load()) {
    return;
  }

  if (async_input_mutex_.try_lock()) {
    latest_async_input_ = input;
    async_input_pending_ = true;
    async_input_mutex_.unlock();
    async_input_cv_.notify_one();
  }
}

bool ReachableCartesianImpedanceController::takeAsyncMonitorOutput(
    AsyncMonitorOutput* output) {
  if (output == nullptr) {
    return false;
  }

  if (!async_output_mutex_.try_lock()) {
    return false;
  }

  const bool has_new_output =
      latest_async_output_.valid &&
      latest_async_output_.sequence > last_consumed_async_output_sequence_;

  if (has_new_output) {
    *output = latest_async_output_;
    last_consumed_async_output_sequence_ = latest_async_output_.sequence;
  }

  async_output_mutex_.unlock();
  return has_new_output;
}

void ReachableCartesianImpedanceController::safetyMonitorWorkerLoop() {
  AsyncPlanningState state;

  while (safety_monitor_worker_running_.load()) {
    AsyncMonitorInput input;

    {
      std::unique_lock<std::mutex> lock(async_input_mutex_);
      async_input_cv_.wait(lock, [&]() {
        return async_input_pending_ || !safety_monitor_worker_running_.load();
      });

      if (!safety_monitor_worker_running_.load()) {
        break;
      }

      input = latest_async_input_;
      async_input_pending_ = false;
    }

    AsyncMonitorOutput output;
    output.sequence = input.sequence;
    output.input_wall_time = input.wall_time;
    output.valid = true;
    output.decision =
        computeShieldDecisionForAsyncInput(input, state);

    {
      std::lock_guard<std::mutex> lock(async_output_mutex_);
      latest_async_output_ = output;
    }
  }
}

void ReachableCartesianImpedanceController::startSafetyMonitorWorker() {
  if (!async_safety_monitor_ || safety_monitor_worker_running_.load()) {
    return;
  }

  safety_monitor_worker_running_.store(true);
  async_input_pending_ = false;
  latest_async_output_ = AsyncMonitorOutput{};
  last_consumed_async_output_sequence_ = 0;
  last_async_output_wall_time_ = -1.0;
  last_async_output_valid_ = false;

  safety_monitor_worker_thread_ =
      std::thread(&ReachableCartesianImpedanceController::safetyMonitorWorkerLoop, this);
}

void ReachableCartesianImpedanceController::stopSafetyMonitorWorker() {
  if (!safety_monitor_worker_running_.exchange(false)) {
    return;
  }

  async_input_cv_.notify_all();

  if (safety_monitor_worker_thread_.joinable()) {
    safety_monitor_worker_thread_.join();
  }
}

void ReachableCartesianImpedanceController::handleMujocoContactSensor(
    const mujoco_ros_msgs::msg::ScalarStamped::SharedPtr msg) {
  const double value = msg->value;
  latest_mujoco_contact_value_.store(value, std::memory_order_relaxed);
  latest_mujoco_contact_msg_time_.store(
      rclcpp::Time(msg->header.stamp).seconds(),
      std::memory_order_relaxed);

  const bool active = value > mujoco_contact_threshold_;
  latest_mujoco_contact_active_.store(active, std::memory_order_relaxed);

  if (active && !mujoco_first_contact_seen_.exchange(true)) {
    mujoco_first_contact_wall_time_.store(
        (get_node()->now() - start_time_).seconds(),
        std::memory_order_relaxed);
    mujoco_first_contact_msg_time_.store(
        rclcpp::Time(msg->header.stamp).seconds(),
        std::memory_order_relaxed);
  }
}

// ============================================================================
// computeImpedanceTorque -- corrected dynamic-consistent true branch
// ============================================================================
Vector7d ReachableCartesianImpedanceController::computeImpedanceTorque(
    const Vector7d& q,
    const Vector7d& dq,
    const Matrix7d& inertia,
    const Vector7d& coriolis,
    const Matrix67d& J_geo,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const ImpedanceSample& cmd,
    double dt) {
  updateRuntimeGains(cmd.K, cmd.D, dt);

  const Vector6d xdot = J_geo * dq;

  Vector6d error = Vector6d::Zero();
  error.head<3>() = current_position - cmd.p;
  error.tail<3>() = computeOrientationError(current_orientation, cmd.q);

  Vector6d xdot_des = Vector6d::Zero();
  xdot_des.head<3>() = cmd.dp;
  xdot_des.tail<3>() = cmd.w;

  Vector6d xddot_des = Vector6d::Zero();
  xddot_des.head<3>() = cmd.ddp;
  xddot_des.tail<3>() = cmd.dw;

  const Vector6d xdot_error = xdot - xdot_des;

  Vector6d Jdot_dq = Vector6d::Zero();

  if (J_geo_prev_valid_) {
    const Matrix67d Jdot =
        (J_geo - J_geo_prev_) / std::max(dt, kMinDt);

    Vector6d Jdot_dq_raw = Jdot * dq;

    const double raw_norm = Jdot_dq_raw.norm();
    if (jdot_dq_max_norm_ > 0.0 && raw_norm > jdot_dq_max_norm_) {
      Jdot_dq_raw *= jdot_dq_max_norm_ / std::max(raw_norm, kSmallPositive);
    }

    Jdot_dq_filtered_ =
        jdot_dq_filter_alpha_ * Jdot_dq_raw +
        (1.0 - jdot_dq_filter_alpha_) * Jdot_dq_filtered_;

    Jdot_dq = Jdot_dq_filtered_;
  }

  J_geo_prev_ = J_geo;
  J_geo_prev_valid_ = true;

  Vector7d tau_task = Vector7d::Zero();

  Matrix7d M_inv = Matrix7d::Zero();
  Matrix6d lambda = Matrix6d::Zero();
  bool lambda_valid = false;

  if (use_dynamic_consistent_impedance_) {
    const Eigen::LDLT<Matrix7d> inertia_ldlt(inertia);

    if (inertia_ldlt.info() == Eigen::Success) {
      M_inv = inertia_ldlt.solve(Matrix7d::Identity());

      Matrix6d lambda_inv = J_geo * M_inv * J_geo.transpose();
      lambda_inv = 0.5 * (lambda_inv + lambda_inv.transpose());
      lambda_inv.diagonal().array() += dynamic_lambda_regularization_;

      const Eigen::LDLT<Matrix6d> lambda_ldlt(lambda_inv);

      if (lambda_ldlt.info() == Eigen::Success) {
        lambda = lambda_ldlt.solve(Matrix6d::Identity());
        lambda = 0.5 * (lambda + lambda.transpose());
        lambda_valid = true;
      }
    }

    if (lambda_valid) {
      // K_runtime_ and D_runtime_ are physical Cartesian stiffness/damping.
      // The inertial feedforward is the only term multiplied by Lambda.
      const Vector6d wrench =
          lambda * (xddot_des - Jdot_dq)
          - K_runtime_ * error
          - D_runtime_ * xdot_error;

      tau_task = J_geo.transpose() * wrench;
    } else {
      tau_task =
          J_geo.transpose() *
          (-K_runtime_ * error - D_runtime_ * xdot_error);
    }
  } else {
    tau_task =
        J_geo.transpose() *
        (-K_runtime_ * error - D_runtime_ * xdot_error);
  }

  const Vector7d tau_nullspace_raw =
      n_stiffness_ * (desired_qn_ - q)
      - 2.0 * std::sqrt(std::max(n_stiffness_, 0.0)) * dq;

  Vector7d tau_nullspace = Vector7d::Zero();

  if (use_dynamic_consistent_impedance_ && lambda_valid) {
    const Eigen::Matrix<double, 7, 6> Jbar =
        M_inv * J_geo.transpose() * lambda;

    const Matrix7d N_transpose =
        Matrix7d::Identity() - J_geo.transpose() * Jbar.transpose();

    tau_nullspace = N_transpose * tau_nullspace_raw;
  } else {
    const Eigen::MatrixXd Jt_pinv = dampedPseudoInverse(J_geo.transpose());

    tau_nullspace =
        (Matrix7d::Identity() - J_geo.transpose() * Jt_pinv) *
        tau_nullspace_raw;
  }

  const Vector7d tau_nullspace_eff =
      (cmd.failsafe && disable_nullspace_in_failsafe_)
          ? Vector7d::Zero()
          : tau_nullspace;

  const Vector7d tau_des = tau_task + coriolis + tau_nullspace_eff;

  const double max_delta = torque_rate_limit_ * std::max(dt, kMinDt);
  Vector7d tau_cmd = tau_cmd_prev_;

  for (int i = 0; i < 7; ++i) {
    const double delta = std::clamp(
        tau_des(i) - tau_cmd_prev_(i),
        -max_delta,
        max_delta);

    tau_cmd(i) = tau_cmd_prev_(i) + delta;
  }

  tau_cmd_prev_ = tau_cmd;
  return tau_cmd;
}

// ============================================================================
// update()
// ============================================================================
controller_interface::return_type ReachableCartesianImpedanceController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  using Clock = std::chrono::steady_clock;
  const auto tic_total = Clock::now();

  const double dt = std::max(period.seconds(), kMinDt);
  const double wall_time = (this->get_node()->now() - start_time_).seconds();

  const Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
  const Eigen::Map<const Vector7d> dq(franka_robot_model_->getRobotState()->dq.data());

  const Matrix7d inertia = arrayToMatrix7d(franka_robot_model_->getMassMatrix());
  const Vector7d coriolis = arrayToVector7d(franka_robot_model_->getCoriolisForceVector());

  const Eigen::Map<const Matrix4d> pose(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  const Vector3d flange_position = pose.block<3, 1>(0, 3);
  const Matrix3d current_rotation = pose.block<3, 3>(0, 0);
  Quaterniond current_orientation(current_rotation); current_orientation.normalize();
  const Vector3d tcp_offset_world = current_orientation * tcp_offset_;
  const Vector3d current_position = flange_position + tcp_offset_world;

  Matrix67d J_geo(franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector).data());
  const Matrix37d Jv_flange = J_geo.topRows<3>();
  const Matrix37d Jw = J_geo.bottomRows<3>();
  J_geo.topRows<3>() = Jv_flange - skewSymmetric(tcp_offset_world) * Jw;
  const Vector6d ee_twist = J_geo * dq;
  const Matrix37d Jv = J_geo.topRows<3>();
  const Vector3d collision_center =
      current_position + collisionCenterOffsetWorld(current_orientation);
  const Vector6d ee_collision_twist =
      twistAtCollisionCenter(current_orientation, ee_twist);
  const Matrix37d Jv_collision =
      Jv - skewSymmetric(collisionCenterOffsetWorld(current_orientation)) * Jw;
  const auto toc_model = Clock::now();

  double paused_total = paused_nominal_time_sec_;
  if (failsafe_enter_wall_time_sec_ >= 0.0)
    paused_total += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
  const double nominal_guess_time = std::max(0.0, wall_time - paused_total);

  ShieldDecision shield_dec;

  if (async_safety_monitor_) {
    AsyncMonitorInput async_input;
    async_input.sequence = async_input_sequence_.fetch_add(1) + 1;
    async_input.wall_time = wall_time;
    async_input.nominal_guess_time = nominal_guess_time;
    async_input.current_position = current_position;
    async_input.current_orientation = current_orientation;
    async_input.ee_twist = ee_twist;
    async_input.inertia = inertia;
    async_input.Jv = Jv_collision;
    async_input.K_runtime = K_runtime_;
    async_input.D_runtime = D_runtime_;
    async_input.last_commanded_sample = last_commanded_sample_;
    async_input.last_commanded_sample_valid = last_commanded_sample_valid_;
    publishAsyncMonitorInput(async_input);

    AsyncMonitorOutput async_output;
    if (takeAsyncMonitorOutput(&async_output)) {
      last_shield_decision_ = async_output.decision;
      last_shield_decision_valid_ = true;
      last_async_output_wall_time_ = async_output.input_wall_time;
      last_async_output_valid_ = true;
    }

    const bool output_is_fresh =
        last_async_output_valid_ &&
        last_shield_decision_valid_ &&
        (async_plan_max_age_sec_ <= 0.0 ||
         (wall_time - last_async_output_wall_time_) <= async_plan_max_age_sec_);

    if (output_is_fresh) {
      shield_dec = last_shield_decision_;
    } else {
      shield_dec.executing_last_verified_failsafe = true;
      shield_dec.candidate_verified = false;
      shield_dec.command =
          makeEmergencyStopCommand(current_position, current_orientation, wall_time);
    }
  } else {
    const bool do_monitor =
        !last_shield_decision_valid_ ||
        ((monitor_counter_++ % std::max(1, monitor_decimation_)) == 0);

    if (do_monitor) {
      shield_dec = computeShieldDecision(wall_time, nominal_guess_time,
                                         current_position, current_orientation,
                                         ee_twist, inertia, Jv_collision);
      last_shield_decision_ = shield_dec;
      last_shield_decision_valid_ = true;
    } else {
      shield_dec = last_shield_decision_;
      if (mode_ == SafetyMode::kFailsafe || shield_dec.executing_last_verified_failsafe) {
        shield_dec.executing_last_verified_failsafe = true;
        if (last_verified_plan_.valid && !last_verified_plan_.failsafe.empty())
          shield_dec.command = getNextFailsafeCommandFromCache(true);
        else
          shield_dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
      } else {
        shield_dec.executing_last_verified_failsafe = false;
        if (last_verified_plan_.valid && !last_verified_plan_.intended.empty())
          shield_dec.command = last_verified_plan_.intended.front();
        else
          shield_dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
      }
      last_shield_decision_.command = shield_dec.command;
      last_shield_decision_.executing_last_verified_failsafe =
          shield_dec.executing_last_verified_failsafe;
    }
  }
  const auto toc_shield = Clock::now();

  if (shield_dec.executing_last_verified_failsafe) {
    if (failsafe_enter_wall_time_sec_ < 0.0) {
      failsafe_enter_wall_time_sec_ = wall_time;
      failsafe_start_time_sec_ = wall_time;
    }
    mode_ = SafetyMode::kFailsafe;
  } else {
    if (failsafe_enter_wall_time_sec_ >= 0.0) {
      paused_nominal_time_sec_ += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
      failsafe_enter_wall_time_sec_ = -1.0;
    }
    mode_ = SafetyMode::kNominal;
  }

  const Vector3d desired_position_cur = shield_dec.command.p;
  const Quaterniond desired_orientation_cur = shield_dec.command.q;
  const Vector3d desired_linear_velocity_cur = shield_dec.command.dp;

  Vector6d error = Vector6d::Zero();
  error.head<3>() = current_position - desired_position_cur;
  error.tail<3>() = computeOrientationError(current_orientation, desired_orientation_cur);
  const double robot_potential_energy_pos =
    0.5 * error.head<3>().transpose()
        * K_runtime_.topLeftCorner<3, 3>()
        * error.head<3>();

  const double robot_potential_energy_rot =
      0.5 * error.tail<3>().transpose()
          * K_runtime_.bottomRightCorner<3, 3>()
          * error.tail<3>();

  const double robot_potential_energy =
      robot_potential_energy_pos + robot_potential_energy_rot;

  MonitorResult monitor = shield_dec.monitor;
  {
    const Vector3d n = human_workspace_.normal();
    const double v_n_now_tube =
        std::abs(n.dot(ee_collision_twist.head<3>())) + monitor.current_vel_error_radius;
    const double Tn_now_tube =
        0.5 * std::max(monitor.m_eff_n, 0.0) * v_n_now_tube * v_n_now_tube;
    monitor.v_n_now_tube = v_n_now_tube;
    monitor.Tn_now_tube  = Tn_now_tube;
    if (mode_ == SafetyMode::kFailsafe) {
      if (prev_Tn_fs_valid_)
        monitor.Tn_dot_est = (Tn_now_tube - prev_Tn_fs_) / std::max(dt, kMinDt);
      else
        monitor.Tn_dot_est = 0.0;
      prev_Tn_fs_ = Tn_now_tube; prev_Tn_fs_valid_ = true;
    } else {
      monitor.Tn_dot_est = 0.0;
      prev_Tn_fs_valid_ = false;
    }
    last_v_n_fs_ = v_n_now_tube;
    last_monitor_result_ = monitor;
    last_monitor_result_valid_ = true;
    last_monitor_wall_time_ = wall_time;
  }

  last_commanded_sample_ = shield_dec.command;
  last_commanded_sample_valid_ = true;

  const Vector7d tau_cmd = computeImpedanceTorque(
      q, dq, inertia, coriolis, J_geo,
      current_position, current_orientation,
      shield_dec.command, dt);
  const auto toc_torque = Clock::now();

  for (int i = 0; i < kNumJoints; ++i) command_interfaces_[i].set_value(tau_cmd(i));

  publishRvizDiagnostics(wall_time, collision_center, desired_position_cur, ee_collision_twist, monitor);

  if (enable_error_logging_ && error_log_file_.is_open()) {
    error_log_file_ << std::fixed << std::setprecision(9)
        << wall_time << "," << nominal_guess_time << "," << paused_nominal_time_sec_ << ","
        << static_cast<int>(mode_) << ","
        << desired_position_cur(0) << "," << desired_position_cur(1) << "," << desired_position_cur(2) << ","
        << current_position(0) << "," << current_position(1) << "," << current_position(2) << ","
        << ee_twist(0) << "," << ee_twist(1) << "," << ee_twist(2) << ","
        << ee_twist(3) << "," << ee_twist(4) << "," << ee_twist(5) << ","
        << current_position(0) << "," << current_position(1) << "," << current_position(2) << ","
        << ee_twist(0) << "," << ee_twist(1) << "," << ee_twist(2) << ","
        << collision_center(0) << "," << collision_center(1) << "," << collision_center(2) << ","
        << ee_collision_twist(0) << "," << ee_collision_twist(1) << "," << ee_collision_twist(2) << ","
        << desired_linear_velocity_cur(0) << "," << desired_linear_velocity_cur(1) << "," << desired_linear_velocity_cur(2) << ","
        << error(0) << "," << error(1) << "," << error(2) << ","
        << error(3) << "," << error(4) << "," << error(5) << ","
        << 0.0 << "," << 0.0 << "," << 0.0 << "," << tau_cmd.norm() << ","
        << K_runtime_(0, 0) << "," << K_runtime_(1, 1) << "," << K_runtime_(2, 2) << ","
        << D_runtime_(0, 0) << "," << D_runtime_(1, 1) << "," << D_runtime_(2, 2) << ","
        << monitor.plane_distance_now << "," << monitor.plane_distance_min << ","
        << monitor.m_eff_n << "," << monitor.v_n_now << "," << monitor.Tn_now << "," << monitor.v_safe << ","
        << static_cast<int>(monitor.nominal_contact_sample_found) << ","
        << monitor.nominal_contact_time << "," << monitor.nominal_contact_distance << ","
        << monitor.v_n_contact_nominal << "," << monitor.Tn_contact_nominal << ","
        << static_cast<int>(monitor.worst_case_contact_found) << ","
        << monitor.worst_case_contact_time << "," << monitor.worst_case_plane_distance_at_candidate << ","
        << monitor.worst_case_nominal_forward_progress << ","
        << monitor.worst_case_v_n_ub << "," << monitor.worst_case_Tn_ub << ","
        << monitor.worst_case_a_pos << "," << monitor.worst_case_a_brake << "," << monitor.worst_case_a_net << ","
        << monitor.h_geom << "," << monitor.h_monitored_energy << ","
        << monitor.h_clamping_energy << "," << monitor.h_terminal_energy << ","
        << monitor.worst_case_V_potential_ub << "," << monitor.terminal_energy_ub << ","
        << robot_potential_energy << ","
        << robot_potential_energy_pos << ","
        << robot_potential_energy_rot << ","
        << monitor.v_n_now_tube << "," << monitor.Tn_now_tube << "," << monitor.Tn_dot_est << ","
        << monitor.current_pos_error_radius << "," << monitor.current_vel_error_radius << ","
        << monitor.worst_case_pos_error_radius << "," << monitor.worst_case_vel_error_radius << ","
        << static_cast<int>(monitor.monitored_contact_possible) << ","
        << static_cast<int>(monitor.collision_energy_unsafe) << ","
        << static_cast<int>(monitor.clamping_energy_unsafe) << ","
        << static_cast<int>(monitor.terminal_energy_unsafe) << ","
        << static_cast<int>(monitor.monitored_unsafe) << ","
        << static_cast<int>(monitor.predicted_trigger) << ","
        << latest_mujoco_contact_value_.load(std::memory_order_relaxed) << ","
        << static_cast<int>(latest_mujoco_contact_active_.load(std::memory_order_relaxed)) << ","
        << latest_mujoco_contact_msg_time_.load(std::memory_order_relaxed) << ","
        << static_cast<int>(mujoco_first_contact_seen_.load(std::memory_order_relaxed)) << ","
        << mujoco_first_contact_wall_time_.load(std::memory_order_relaxed) << ","
        << mujoco_first_contact_msg_time_.load(std::memory_order_relaxed) << "\n";
    ++log_write_counter_;
    if ((log_write_counter_ % 200) == 0) error_log_file_.flush();
  }
  const auto toc_io = Clock::now();

  const auto toc_total = Clock::now();
  const double model_ms = std::chrono::duration<double, std::milli>(toc_model - tic_total).count();
  const double shield_ms = std::chrono::duration<double, std::milli>(toc_shield - toc_model).count();
  const double torque_ms = std::chrono::duration<double, std::milli>(toc_torque - toc_shield).count();
  const double io_ms = std::chrono::duration<double, std::milli>(toc_io - toc_torque).count();
  const double exec_ms = std::chrono::duration<double, std::milli>(toc_total - tic_total).count();
  exec_sum_ms_ += exec_ms;
  exec_max_ms_ = std::max(exec_max_ms_, exec_ms);
  exec_min_ms_ = std::min(exec_min_ms_, exec_ms);
  if (exec_ms > 1.0) ++exec_overrun_1ms_count_;
  if (exec_ms > 2.0) ++exec_overrun_2ms_count_;
  prof_model_sum_ms_ += model_ms;
  prof_model_max_ms_ = std::max(prof_model_max_ms_, model_ms);
  prof_shield_sum_ms_ += shield_ms;
  prof_shield_max_ms_ = std::max(prof_shield_max_ms_, shield_ms);
  prof_torque_sum_ms_ += torque_ms;
  prof_torque_max_ms_ = std::max(prof_torque_max_ms_, torque_ms);
  prof_io_sum_ms_ += io_ms;
  prof_io_max_ms_ = std::max(prof_io_max_ms_, io_ms);
  ++loop_counter_;
  if (loop_counter_ >= static_cast<std::size_t>(profiling_stats_print_period_)) {
    const double n = static_cast<double>(loop_counter_);
    RCLCPP_INFO(get_node()->get_logger(),
                "[reachable_impedance] mode=%d avg=%.3f ms min=%.3f ms max=%.3f ms overruns>1ms=%zu >2ms=%zu "
                "model_avg/max=%.3f/%.3f shield_avg/max=%.3f/%.3f torque_avg/max=%.3f/%.3f io_avg/max=%.3f/%.3f "
                "buf=%zu/%zu plan_valid=%d",
                static_cast<int>(mode_),
                exec_sum_ms_ / n, exec_min_ms_, exec_max_ms_,
                exec_overrun_1ms_count_, exec_overrun_2ms_count_,
                prof_model_sum_ms_ / n, prof_model_max_ms_,
                prof_shield_sum_ms_ / n, prof_shield_max_ms_,
                prof_torque_sum_ms_ / n, prof_torque_max_ms_,
                prof_io_sum_ms_ / n, prof_io_max_ms_,
                intended_buffer_index_, intended_buffer_.size(),
                static_cast<int>(last_verified_plan_.valid));
    loop_counter_ = 0;
    exec_sum_ms_ = 0.0;
    exec_min_ms_ = 1e9;
    exec_max_ms_ = 0.0;
    exec_overrun_1ms_count_ = 0;
    exec_overrun_2ms_count_ = 0;
    prof_model_sum_ms_ = 0.0;
    prof_model_max_ms_ = 0.0;
    prof_shield_sum_ms_ = 0.0;
    prof_shield_max_ms_ = 0.0;
    prof_torque_sum_ms_ = 0.0;
    prof_torque_max_ms_ = 0.0;
    prof_io_sum_ms_ = 0.0;
    prof_io_max_ms_ = 0.0;
  }

  return controller_interface::return_type::OK;
}

// ============================================================================
// on_init
// ============================================================================
CallbackReturn ReachableCartesianImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<bool>("enable_error_logging", true);
    auto_declare<std::string>("error_log_path",
        "");
    auto_declare<std::string>("error_log_root_dir", kDefaultErrorLogRootDir);
    auto_declare<std::string>("error_log_file_name", kDefaultErrorLogFileName);

    auto_declare<bool>("use_constant_reference", false);
    auto_declare<std::string>("reference_trajectory_type", "lissajous");

    auto_declare<double>("nominal_pos_stiffness", 400.0);
    auto_declare<double>("nominal_rot_stiffness", 20.0);
    auto_declare<double>("failsafe_pos_stiffness", 50.0);
    auto_declare<double>("failsafe_rot_stiffness", 5.0);
    auto_declare<double>("failsafe_pos_damping_scale", 2.5);
    auto_declare<double>("failsafe_rot_damping_scale", 2.5);

    auto_declare<double>("n_stiffness", 0.0);
    auto_declare<bool>("disable_nullspace_in_failsafe", true);
    auto_declare<bool>("enable_safety_monitor", true);

    auto_declare<double>("safe_collision_energy_joule", 0.05);
    auto_declare<double>("clamping_energy_budget_joule", 0.05);
    auto_declare<double>("energy_budget_margin_joule", 0.005);
    auto_declare<double>("ee_collision_radius", 0.04);
    auto_declare<std::vector<double>>(
        "tcp_offset", std::vector<double>{0.0, 0.0, 0.0});
    auto_declare<std::vector<double>>(
        "ee_collision_center_offset", std::vector<double>{0.0, 0.0, 0.0});
    auto_declare<int>("monitor_decimation", 1);
    auto_declare<bool>("async_safety_monitor", true);
    auto_declare<double>("async_plan_max_age_sec", 0.05);

    cps_human_workspace::HumanWorkspace::declareParameters(get_node());

    auto_declare<double>("k_rate_limit", 5000.0);
    auto_declare<double>("d_rate_limit", 500.0);
    auto_declare<double>("tau_rate_limit", 1000.0);
    auto_declare<double>("torque_to_accel_gain", 8.0);
    auto_declare<double>("model_accel_uncertainty", 0.05);
    auto_declare<double>("stiffness_error_bound_m", 0.01);

    auto_declare<double>("tracking_pos_error_bound", 0.005);
    auto_declare<double>("tracking_vel_error_bound", 0.05);

    auto_declare<int>("shield_horizon_steps", 100);
    auto_declare<double>("shield_plan_dt", 0.01);

    auto_declare<double>("path_retiming_search_window_sec", 0.25);
    auto_declare<int>("path_retiming_search_steps", 41);
    auto_declare<double>("path_time_rate_min", 0.0);
    auto_declare<double>("path_time_rate_max", 1.5);
    auto_declare<double>("path_time_acc_limit", 3.0);
    auto_declare<double>("path_time_rate_target", 1.0);

    auto_declare<int>("local_replan_horizon_steps", 200);
    auto_declare<double>("local_replan_dt", 0.001);
    auto_declare<double>("local_path_lookahead_sec", 0.08);
    auto_declare<double>("local_replan_max_velocity", 0.08);
    auto_declare<double>("local_replan_max_acceleration", 0.4);
    auto_declare<double>("local_replan_max_jerk", 2.0);

    auto_declare<bool>("use_dynamic_consistent_impedance", true);
    auto_declare<double>("torque_rate_limit", 1000.0);

    // Added for corrected dynamic-consistent branch.
    auto_declare<double>("dynamic_lambda_regularization", 1.0e-6);
    auto_declare<double>("jdot_dq_filter_alpha", 0.15);
    auto_declare<double>("jdot_dq_max_norm", 5.0);

    auto_declare<bool>("rviz_enable_markers", true);
    auto_declare<std::string>("rviz_frame_id", "panda_link0");
    auto_declare<int>("rviz_marker_decimation", 10);
    auto_declare<double>("rviz_marker_lifetime_sec", 0.2);
    auto_declare<double>("rviz_plane_size", 0.8);
    auto_declare<double>("rviz_plane_thickness", 0.003);
    auto_declare<double>("rviz_normal_arrow_length", 0.20);
    auto_declare<double>("rviz_velocity_arrow_scale", 0.25);
    auto_declare<double>("rviz_arrow_shaft_diameter", 0.01);
    auto_declare<double>("rviz_arrow_head_diameter", 0.02);
    auto_declare<double>("rviz_arrow_head_length", 0.03);
    auto_declare<double>("rviz_text_scale", 0.04);
    auto_declare<double>("rviz_text_z_offset", 0.12);
    auto_declare<double>("rviz_ub_arrow_z_offset", 0.05);

    auto_declare<int>("profiling_stats_print_period", 1000);
    auto_declare<bool>("enable_mujoco_contact_logging", true);
    auto_declare<std::string>("mujoco_contact_sensor_topic", "/panda_metal_ball_touch");
    auto_declare<double>("mujoco_contact_threshold", 1.0e-6);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage: %s\n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

// ============================================================================
// on_configure
// ============================================================================
CallbackReturn ReachableCartesianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    enable_error_logging_ = get_node()->get_parameter("enable_error_logging").as_bool();
    legacy_error_log_path_ = get_node()->get_parameter("error_log_path").as_string();
    error_log_root_dir_ = get_node()->get_parameter("error_log_root_dir").as_string();
    error_log_file_name_ = get_node()->get_parameter("error_log_file_name").as_string();

    if (error_log_root_dir_.empty()) {
      error_log_root_dir_ = kDefaultErrorLogRootDir;
    }
    error_log_file_name_ =
        sanitizedFileNameOrDefault(error_log_file_name_, kDefaultErrorLogFileName);

    if (!legacy_error_log_path_.empty()) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "error_log_path is deprecated and will be ignored. Use error_log_root_dir and error_log_file_name.");
    }

    use_constant_reference_ = get_node()->get_parameter("use_constant_reference").as_bool();
    reference_trajectory_type_ = get_node()->get_parameter("reference_trajectory_type").as_string();

    const double nominal_pos_stiffness = get_node()->get_parameter("nominal_pos_stiffness").as_double();
    const double nominal_rot_stiffness = get_node()->get_parameter("nominal_rot_stiffness").as_double();
    const double failsafe_pos_stiffness = get_node()->get_parameter("failsafe_pos_stiffness").as_double();
    const double failsafe_rot_stiffness = get_node()->get_parameter("failsafe_rot_stiffness").as_double();
    failsafe_pos_damping_scale_ =
        get_node()->get_parameter("failsafe_pos_damping_scale").as_double();

    failsafe_rot_damping_scale_ =
        get_node()->get_parameter("failsafe_rot_damping_scale").as_double();

    n_stiffness_ = get_node()->get_parameter("n_stiffness").as_double();
    disable_nullspace_in_failsafe_ = get_node()->get_parameter("disable_nullspace_in_failsafe").as_bool();
    enable_safety_monitor_ = get_node()->get_parameter("enable_safety_monitor").as_bool();

    safe_collision_energy_joule_ = get_node()->get_parameter("safe_collision_energy_joule").as_double();
    clamping_energy_budget_joule_ =
        std::max(0.0, get_node()->get_parameter("clamping_energy_budget_joule").as_double());

    energy_budget_margin_joule_ =
        std::max(0.0, get_node()->get_parameter("energy_budget_margin_joule").as_double());

    ee_collision_radius_ = get_node()->get_parameter("ee_collision_radius").as_double();
    const auto tcp_offset =
        get_node()->get_parameter("tcp_offset").as_double_array();
    if (tcp_offset.size() == 3) {
      tcp_offset_ = Vector3d(tcp_offset[0], tcp_offset[1], tcp_offset[2]);
    } else {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "tcp_offset must contain 3 values. Using [0, 0, 0].");
      tcp_offset_.setZero();
    }
    const auto ee_collision_center_offset =
        get_node()->get_parameter("ee_collision_center_offset").as_double_array();
    if (ee_collision_center_offset.size() == 3) {
      ee_collision_center_offset_ = Vector3d(
          ee_collision_center_offset[0],
          ee_collision_center_offset[1],
          ee_collision_center_offset[2]);
    } else {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "ee_collision_center_offset must contain 3 values. Using [0, 0, 0].");
      ee_collision_center_offset_.setZero();
    }
    monitor_decimation_ = std::max(1, static_cast<int>(get_node()->get_parameter("monitor_decimation").as_int()));
    async_safety_monitor_ = get_node()->get_parameter("async_safety_monitor").as_bool();
    async_plan_max_age_sec_ =
        std::max(0.0, get_node()->get_parameter("async_plan_max_age_sec").as_double());

    k_rate_limit_ = std::max(0.0, get_node()->get_parameter("k_rate_limit").as_double());
    d_rate_limit_ = std::max(0.0, get_node()->get_parameter("d_rate_limit").as_double());
    tau_rate_limit_ = get_node()->get_parameter("tau_rate_limit").as_double();
    torque_to_accel_gain_ = get_node()->get_parameter("torque_to_accel_gain").as_double();
    model_accel_uncertainty_ = get_node()->get_parameter("model_accel_uncertainty").as_double();
    stiffness_error_bound_m_ = get_node()->get_parameter("stiffness_error_bound_m").as_double();

    tracking_pos_error_bound_ = std::max(0.0, get_node()->get_parameter("tracking_pos_error_bound").as_double());
    tracking_vel_error_bound_ = std::max(0.0, get_node()->get_parameter("tracking_vel_error_bound").as_double());

    shield_horizon_steps_ = std::max(1, static_cast<int>(get_node()->get_parameter("shield_horizon_steps").as_int()));
    shield_plan_dt_ = std::max(get_node()->get_parameter("shield_plan_dt").as_double(), kMinDt);

    path_retiming_search_window_sec_ = std::max(get_node()->get_parameter("path_retiming_search_window_sec").as_double(), 0.0);
    path_retiming_search_steps_ = std::max(5, static_cast<int>(get_node()->get_parameter("path_retiming_search_steps").as_int()));
    path_time_rate_min_ = get_node()->get_parameter("path_time_rate_min").as_double();
    path_time_rate_max_ = std::max(path_time_rate_min_, get_node()->get_parameter("path_time_rate_max").as_double());
    path_time_acc_limit_ = std::max(0.0, get_node()->get_parameter("path_time_acc_limit").as_double());
    path_time_rate_target_ = std::clamp(get_node()->get_parameter("path_time_rate_target").as_double(),
                                        path_time_rate_min_, path_time_rate_max_);

    local_replan_horizon_steps_ = std::max(1, static_cast<int>(get_node()->get_parameter("local_replan_horizon_steps").as_int()));
    local_replan_dt_ = std::max(get_node()->get_parameter("local_replan_dt").as_double(), kMinDt);
    local_path_lookahead_sec_ = std::max(get_node()->get_parameter("local_path_lookahead_sec").as_double(), local_replan_dt_);
    local_replan_max_velocity_ = std::max(1e-4, get_node()->get_parameter("local_replan_max_velocity").as_double());
    local_replan_max_acceleration_ = std::max(1e-4, get_node()->get_parameter("local_replan_max_acceleration").as_double());
    local_replan_max_jerk_ = std::max(1e-4, get_node()->get_parameter("local_replan_max_jerk").as_double());

    use_dynamic_consistent_impedance_ = get_node()->get_parameter("use_dynamic_consistent_impedance").as_bool();
    torque_rate_limit_ = get_node()->get_parameter("torque_rate_limit").as_double();

    dynamic_lambda_regularization_ = std::max(
        1.0e-10,
        get_node()->get_parameter("dynamic_lambda_regularization").as_double());
    jdot_dq_filter_alpha_ = std::clamp(
        get_node()->get_parameter("jdot_dq_filter_alpha").as_double(),
        0.0,
        1.0);
    jdot_dq_max_norm_ = std::max(
        0.0,
        get_node()->get_parameter("jdot_dq_max_norm").as_double());

    rviz_enable_markers_ = get_node()->get_parameter("rviz_enable_markers").as_bool();
    rviz_frame_id_ = get_node()->get_parameter("rviz_frame_id").as_string();
    rviz_marker_decimation_ = std::max(1, static_cast<int>(get_node()->get_parameter("rviz_marker_decimation").as_int()));
    rviz_marker_lifetime_sec_ = get_node()->get_parameter("rviz_marker_lifetime_sec").as_double();
    rviz_plane_size_ = get_node()->get_parameter("rviz_plane_size").as_double();
    rviz_plane_thickness_ = get_node()->get_parameter("rviz_plane_thickness").as_double();
    rviz_normal_arrow_length_ = get_node()->get_parameter("rviz_normal_arrow_length").as_double();
    rviz_velocity_arrow_scale_ = get_node()->get_parameter("rviz_velocity_arrow_scale").as_double();
    rviz_arrow_shaft_diameter_ = get_node()->get_parameter("rviz_arrow_shaft_diameter").as_double();
    rviz_arrow_head_diameter_ = get_node()->get_parameter("rviz_arrow_head_diameter").as_double();
    rviz_arrow_head_length_ = get_node()->get_parameter("rviz_arrow_head_length").as_double();
    rviz_text_scale_ = get_node()->get_parameter("rviz_text_scale").as_double();
    rviz_text_z_offset_ = get_node()->get_parameter("rviz_text_z_offset").as_double();
    rviz_ub_arrow_z_offset_ = get_node()->get_parameter("rviz_ub_arrow_z_offset").as_double();

    if (!human_workspace_.configureFromParameters(get_node(), get_node()->get_logger())) {
      return CallbackReturn::ERROR;
    }

    profiling_stats_print_period_ = std::max<int>(
        1, static_cast<int>(get_node()->get_parameter("profiling_stats_print_period").as_int()));
    enable_mujoco_contact_logging_ =
        get_node()->get_parameter("enable_mujoco_contact_logging").as_bool();
    mujoco_contact_sensor_topic_ =
        get_node()->get_parameter("mujoco_contact_sensor_topic").as_string();
    mujoco_contact_threshold_ =
        std::max(0.0, get_node()->get_parameter("mujoco_contact_threshold").as_double());

    if (enable_mujoco_contact_logging_ && !mujoco_contact_sensor_topic_.empty()) {
      mujoco_contact_sub_ =
          get_node()->create_subscription<mujoco_ros_msgs::msg::ScalarStamped>(
              mujoco_contact_sensor_topic_,
              rclcpp::QoS(10),
              std::bind(
                  &ReachableCartesianImpedanceController::handleMujocoContactSensor,
                  this,
                  std::placeholders::_1));
    } else {
      mujoco_contact_sub_.reset();
    }

    K_nominal_.setZero(); D_nominal_.setZero();
    K_f_target_.setZero(); D_f_target_.setZero();
    K_nominal_.topLeftCorner<3, 3>() = nominal_pos_stiffness * Matrix3d::Identity();
    K_nominal_.bottomRightCorner<3, 3>() = nominal_rot_stiffness * Matrix3d::Identity();
    D_nominal_.topLeftCorner<3, 3>() = 2.0 * std::sqrt(std::max(nominal_pos_stiffness, 0.0)) * Matrix3d::Identity();
    D_nominal_.bottomRightCorner<3, 3>() = 0.8 * 2.0 * std::sqrt(std::max(nominal_rot_stiffness, 0.0)) * Matrix3d::Identity();
    K_f_target_.topLeftCorner<3, 3>() = failsafe_pos_stiffness * Matrix3d::Identity();
    K_f_target_.bottomRightCorner<3, 3>() = failsafe_rot_stiffness * Matrix3d::Identity();
    D_f_target_.topLeftCorner<3, 3>() =
        failsafe_pos_damping_scale_ * 2.0 *
        std::sqrt(std::max(failsafe_pos_stiffness, 0.0)) *
        Matrix3d::Identity();

    D_f_target_.bottomRightCorner<3, 3>() =
        failsafe_rot_damping_scale_ * 2.0 *
        std::sqrt(std::max(failsafe_rot_stiffness, 0.0)) *
        Matrix3d::Identity();
    K_runtime_ = K_nominal_; D_runtime_ = D_nominal_;

    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model", arm_id_));

    if (rviz_enable_markers_)
      rviz_marker_pub_ = get_node()->create_publisher<MarkerArray>("reachable_impedance/markers", 10);

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception in on_configure: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

// ============================================================================
// on_activate / on_deactivate
// ============================================================================
CallbackReturn ReachableCartesianImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  start_time_ = this->get_node()->now();

  const Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
  desired_qn_ = q;
  const Eigen::Map<const Matrix4d> pose(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  desired_orientation_ = Quaterniond(pose.block<3, 3>(0, 0));
  desired_orientation_.normalize();
  desired_position_ =
      pose.block<3, 1>(0, 3) + desired_orientation_ * tcp_offset_;

  last_commanded_sample_ = ImpedanceSample{};
  last_commanded_sample_.t = 0.0;
  last_commanded_sample_.p = desired_position_;
  last_commanded_sample_.dp.setZero();
  last_commanded_sample_.ddp.setZero();
  last_commanded_sample_.q = desired_orientation_;
  last_commanded_sample_.q.normalize();
  last_commanded_sample_.w.setZero();
  last_commanded_sample_.dw.setZero();
  last_commanded_sample_.K = K_nominal_;
  last_commanded_sample_.D = D_nominal_;
  last_commanded_sample_.failsafe = false;
  last_commanded_sample_valid_ = true;

  mode_ = SafetyMode::kNominal;
  failsafe_start_time_sec_ = -1.0;
  failsafe_enter_wall_time_sec_ = -1.0;
  paused_nominal_time_sec_ = 0.0;
  loop_counter_ = 0; exec_sum_ms_ = 0.0; exec_min_ms_ = 1e9; exec_max_ms_ = 0.0;
  exec_overrun_1ms_count_ = 0;
  exec_overrun_2ms_count_ = 0;
  prof_model_sum_ms_ = 0.0;
  prof_model_max_ms_ = 0.0;
  prof_shield_sum_ms_ = 0.0;
  prof_shield_max_ms_ = 0.0;
  prof_torque_sum_ms_ = 0.0;
  prof_torque_max_ms_ = 0.0;
  prof_io_sum_ms_ = 0.0;
  prof_io_max_ms_ = 0.0;
  log_write_counter_ = 0;
  prev_Tn_fs_ = 0.0; prev_Tn_fs_valid_ = false; last_v_n_fs_ = 0.0;
  latest_mujoco_contact_value_.store(0.0);
  latest_mujoco_contact_msg_time_.store(-1.0);
  latest_mujoco_contact_active_.store(false);
  mujoco_first_contact_seen_.store(false);
  mujoco_first_contact_wall_time_.store(-1.0);
  mujoco_first_contact_msg_time_.store(-1.0);

  monitor_counter_ = 0;
  last_monitor_result_valid_ = false;
  last_monitor_wall_time_ = 0.0;
  last_monitor_result_ = MonitorResult{};

  rviz_publish_counter_ = 0;

  last_verified_plan_ = VerifiedPlan{};
  candidate_plan_ = VerifiedPlan{};
  candidate_plan_valid_ = false;

  commanded_path_time_ = 0.0;

  intended_buffer_.clear();
  intended_buffer_index_ = 0;
  intended_buffer_valid_ = false;

  tau_cmd_prev_.setZero();
  last_shield_decision_valid_ = false;
  async_input_sequence_.store(0);
  async_input_pending_ = false;
  latest_async_output_ = AsyncMonitorOutput{};
  last_consumed_async_output_sequence_ = 0;
  last_async_output_wall_time_ = -1.0;
  last_async_output_valid_ = false;

  J_geo_prev_.setZero();
  Jdot_dq_filtered_.setZero();
  J_geo_prev_valid_ = false;

  if (enable_error_logging_) {
    const std::filesystem::path root_dir(error_log_root_dir_);
    error_log_run_dir_ =
        (root_dir / makeBerlinTimestampForDirectoryName()).string();
    error_log_file_path_ =
        (std::filesystem::path(error_log_run_dir_) / error_log_file_name_).string();

    std::error_code ec;
    std::filesystem::create_directories(error_log_run_dir_, ec);
    if (ec) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Failed to create log run directory %s: %s",
          error_log_run_dir_.c_str(),
          ec.message().c_str());
      return CallbackReturn::ERROR;
    }

    const std::filesystem::path run_info_path =
        std::filesystem::path(error_log_run_dir_) / "run_info.txt";
    std::ofstream run_info_file(run_info_path, std::ios::out | std::ios::trunc);
    if (run_info_file.is_open()) {
      run_info_file << "run_directory: " << error_log_run_dir_ << "\n"
                    << "csv_file: " << error_log_file_path_ << "\n"
                    << "arm_id: " << arm_id_ << "\n"
                    << "reference_trajectory_type: " << reference_trajectory_type_ << "\n"
                    << "use_constant_reference: " << static_cast<int>(use_constant_reference_) << "\n"
                    << "enable_safety_monitor: " << static_cast<int>(enable_safety_monitor_) << "\n"
                    << "async_safety_monitor: " << static_cast<int>(async_safety_monitor_) << "\n"
                    << "async_plan_max_age_sec: " << async_plan_max_age_sec_ << "\n"
                    << "tcp_offset: ["
                    << tcp_offset_.x() << ", "
                    << tcp_offset_.y() << ", "
                    << tcp_offset_.z() << "]\n"
                    << "ee_collision_radius: " << ee_collision_radius_ << "\n"
                    << "ee_collision_center_offset_from_tcp: ["
                    << ee_collision_center_offset_.x() << ", "
                    << ee_collision_center_offset_.y() << ", "
                    << ee_collision_center_offset_.z() << "]\n"
                    << "enable_mujoco_contact_logging: "
                    << static_cast<int>(enable_mujoco_contact_logging_) << "\n"
                    << "mujoco_contact_sensor_topic: "
                    << mujoco_contact_sensor_topic_ << "\n"
                    << "mujoco_contact_threshold: "
                    << mujoco_contact_threshold_ << "\n"
                    << "timestamp_timezone: Europe/Berlin\n"
                    << "initial_desired_position: ["
                    << desired_position_.x() << ", "
                    << desired_position_.y() << ", "
                    << desired_position_.z() << "]\n"
                    << "human_plane_normal: ["
                    << human_workspace_.normal().x() << ", "
                    << human_workspace_.normal().y() << ", "
                    << human_workspace_.normal().z() << "]\n"
                    << "human_sphere_center: ["
                    << human_workspace_.center().x() << ", "
                    << human_workspace_.center().y() << ", "
                    << human_workspace_.center().z() << "]\n"
                    << "human_motion_radius: " << human_workspace_.motionRadius() << "\n"
                    << "human_hand_radius: " << human_workspace_.handRadius() << "\n"
                    << "human_workspace_config_path: "
                    << cps_human_workspace::HumanWorkspace::defaultConfigPath() << "\n";
    }

    error_log_file_.open(error_log_file_path_, std::ios::out | std::ios::trunc);
    if (!error_log_file_.is_open()) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Failed to open log file: %s",
          error_log_file_path_.c_str());
      return CallbackReturn::ERROR;
    }
    error_log_file_ << std::fixed << std::setprecision(9);
    error_log_file_
        << "wall_time_sec,nominal_time_sec,paused_nominal_time_sec,mode,"
        << "des_px,des_py,des_pz,cur_px,cur_py,cur_pz,"
        << "cur_vx,cur_vy,cur_vz,cur_wx,cur_wy,cur_wz,"
        << "tcp_px,tcp_py,tcp_pz,tcp_vx,tcp_vy,tcp_vz,"
        << "collision_center_px,collision_center_py,collision_center_pz,"
        << "collision_center_vx,collision_center_vy,collision_center_vz,"
        << "des_vx,des_vy,des_vz,"
        << "err_px,err_py,err_pz,err_rx,err_ry,err_rz,"
        << "tau_task_norm,tau_null_norm,tau_friction_norm,tau_cmd_norm,"
        << "Kx,Ky,Kz,Dx,Dy,Dz,"
        << "plane_distance_now,plane_distance_min,"
        << "m_eff_n,v_n_now,Tn_now,v_safe,"
        << "nominal_contact_sample_found,nominal_contact_time,nominal_contact_distance,"
        << "v_n_contact_nominal,Tn_contact_nominal,"
        << "worst_case_contact_found,worst_case_contact_time,worst_case_plane_distance_at_candidate,"
        << "worst_case_nominal_forward_progress,"
        << "worst_case_v_n_ub,worst_case_Tn_ub,"
        << "worst_case_a_pos,worst_case_a_brake,worst_case_a_net,"
        << "h_geom,h_monitored_energy,h_clamping_energy,h_terminal_energy,"
        << "worst_case_V_potential_ub,terminal_energy_ub,"
        << "robot_potential_energy,robot_potential_energy_pos,robot_potential_energy_rot,"
        << "v_n_now_tube,Tn_now_tube,Tn_dot_est,"
        << "current_pos_error_radius,current_vel_error_radius,"
        << "worst_case_pos_error_radius,worst_case_vel_error_radius,"
        << "monitored_contact_possible,collision_energy_unsafe,clamping_energy_unsafe,"
        << "terminal_energy_unsafe,monitored_unsafe,predicted_trigger,"
        << "mujoco_contact_value,mujoco_contact_active,mujoco_contact_msg_time_sec,"
        << "mujoco_first_contact_seen,mujoco_first_contact_wall_time_sec,"
        << "mujoco_first_contact_msg_time_sec\n";
    RCLCPP_INFO(
        get_node()->get_logger(),
        "Validation log enabled: %s",
        error_log_file_path_.c_str());
  }
  startSafetyMonitorWorker();
  return CallbackReturn::SUCCESS;
}

CallbackReturn ReachableCartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  stopSafetyMonitorWorker();
  if (error_log_file_.is_open()) { error_log_file_.flush(); error_log_file_.close(); }
  J_geo_prev_.setZero();
  Jdot_dq_filtered_.setZero();
  J_geo_prev_valid_ = false;
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace cps_controllers

PLUGINLIB_EXPORT_CLASS(cps_controllers::ReachableCartesianImpedanceController,
                       controller_interface::ControllerInterface)
