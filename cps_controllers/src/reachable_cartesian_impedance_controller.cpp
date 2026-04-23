// Copyright (c) 2026
// Reachable Cartesian Impedance Controller
//
// SARA-style monitored execution with local error-tube verification
//
// Core execution logic
// --------------------
// 1) In each monitor cycle, build a candidate monitored plan:
//      intended prefix + compliant failsafe suffix
// 2) Verify the candidate monitored plan against
//      nominal rollout + local error-tube rollout
// 3) If verified, execute the first intended sample and cache the whole plan
// 4) Otherwise, execute the next sample of the last successfully verified failsafe suffix
//
// Notes
// -----
// - Intended is regenerated every monitor cycle from the CURRENT measured state
//   by online time-parameterizing the SAME geometric path.
// - Failsafe remains impedance-based and path-consistent by freezing the newly
//   generated intended reference of the current cycle.
// - Tracking error is NOT assumed to be zero.
//   At each monitor refresh, the local tube is initialized from current measured error:
//      r_p(0) = ||x - x_anchor||
//      r_v(0) = ||v - v_anchor||
// - Then propagated by:
//      r_p_dot <= r_v
//      r_v_dot <= alpha * r_p - beta * r_v + gamma

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
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

namespace {

constexpr double kMinDt = 1e-6;
constexpr double kSmoothStartDuration = 2.0;
constexpr double kLambdaReg = 1e-9;
constexpr double kSmallPositive = 1e-9;

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
  Vector3d p{Vector3d::Zero()};
  Vector3d dp{Vector3d::Zero()};
  Vector3d ddp{Vector3d::Zero()};
  Matrix3d R{Matrix3d::Identity()};
};

inline std_msgs::msg::ColorRGBA makeColor(float r, float g, float b, float a) {
  std_msgs::msg::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}

inline geometry_msgs::msg::Point toPoint(const Vector3d& p) {
  geometry_msgs::msg::Point msg;
  msg.x = p.x();
  msg.y = p.y();
  msg.z = p.z();
  return msg;
}

inline geometry_msgs::msg::Quaternion toQuatMsg(const Quaterniond& q) {
  geometry_msgs::msg::Quaternion msg;
  msg.x = q.x();
  msg.y = q.y();
  msg.z = q.z();
  msg.w = q.w();
  return msg;
}

inline Quaterniond rotationMatrixToQuaternion(const Matrix3d& R) {
  Quaterniond q(R);
  q.normalize();
  return q;
}

inline Matrix3d makePlaneFrameFromNormal(const Vector3d& n_in) {
  const Vector3d n = n_in.normalized();
  Vector3d ref = std::abs(n.z()) < 0.9 ? Vector3d::UnitZ() : Vector3d::UnitX();
  Vector3d x = ref.cross(n).normalized();
  Vector3d y = n.cross(x).normalized();
  Matrix3d R;
  R.col(0) = x;
  R.col(1) = y;
  R.col(2) = n;
  return R;
}

inline ReferenceTrajectoryType parseReferenceTrajectoryType(const std::string& name) {
  if (name == "lissajous") {
    return ReferenceTrajectoryType::kLissajous;
  }
  if (name == "constant") {
    return ReferenceTrajectoryType::kConstant;
  }
  return ReferenceTrajectoryType::kLine;
}

inline ReferenceTrajectoryType parseTrajectoryTypeOrDefault(bool use_constant_reference,
                                                            const std::string& name) {
  if (use_constant_reference) {
    return ReferenceTrajectoryType::kConstant;
  }
  return parseReferenceTrajectoryType(name);
}

inline SmoothStartProfile makeSmoothStartProfile(double t, double T) {
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

inline TaskRefPose makeReferenceLine(double t,
                                     const Vector3d& p0,
                                     const Matrix3d& R0) {
  TaskRefPose ref;
  constexpr double T_move = 1.5;
  constexpr double dz = -0.55;

  const SmoothStartProfile ramp = makeSmoothStartProfile(t, T_move);

  ref.p = p0 + Vector3d(0.0, 0.0, dz * ramp.s);
  ref.dp = Vector3d(0.0, 0.0, dz * ramp.ds);
  ref.ddp = Vector3d(0.0, 0.0, dz * ramp.dds);
  ref.R = R0;
  return ref;
}

inline TaskRefPose makeReferenceLissajous(double t,
                                          const Vector3d& p0,
                                          const Matrix3d& R0) {
  TaskRefPose ref;

  const double wt = 2.0 * M_PI * 0.25;
  const SmoothStartProfile ramp = makeSmoothStartProfile(t, kSmoothStartDuration);

  constexpr double Ax = 0.08;
  constexpr double Ay = 0.08;
  constexpr double Az = 0.04;

  const double ph_px = 0.0;
  const double ph_py = M_PI / 2.0;
  const double ph_pz = 0.0;

  const Vector3d p_base(
      Ax * (std::sin(wt * t + ph_px) - std::sin(ph_px)),
      Ay * (std::sin(wt * t + ph_py) - std::sin(ph_py)),
      Az * (std::sin(0.5 * wt * t + ph_pz) - std::sin(ph_pz)));

  const Vector3d dp_base(
      Ax * wt * std::cos(wt * t + ph_px),
      Ay * wt * std::cos(wt * t + ph_py),
      Az * 0.5 * wt * std::cos(0.5 * wt * t + ph_pz));

  const Vector3d ddp_base(
      -Ax * wt * wt * std::sin(wt * t + ph_px),
      -Ay * wt * wt * std::sin(wt * t + ph_py),
      -Az * 0.25 * wt * wt * std::sin(0.5 * wt * t + ph_pz));

  ref.p = p0 + ramp.s * p_base;
  ref.dp = ramp.ds * p_base + ramp.s * dp_base;
  ref.ddp = ramp.dds * p_base + 2.0 * ramp.ds * dp_base + ramp.s * ddp_base;
  ref.R = R0;
  return ref;
}

inline TaskRefPose makeReferencePose(double t,
                                     const Vector3d& p0,
                                     const Matrix3d& R0,
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

inline Eigen::MatrixXd dampedPseudoInverse(const Eigen::MatrixXd& M, double lambda = 0.2) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto s = svd.singularValues();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().cols());
  for (int i = 0; i < s.size(); ++i) {
    S(i, i) = s(i) / (s(i) * s(i) + lambda * lambda);
  }
  return svd.matrixV() * S * svd.matrixU().transpose();
}

inline Vector3d computeOrientationError(const Quaterniond& current,
                                        const Quaterniond& desired) {
  Quaterniond q_curr = current;
  Quaterniond q_des = desired;

  if (q_des.coeffs().dot(q_curr.coeffs()) < 0.0) {
    q_curr.coeffs() << -q_curr.coeffs();
  }

  const Quaterniond q_err(q_curr * q_des.inverse());
  Eigen::AngleAxisd aa(q_err);
  return aa.axis() * aa.angle();
}

inline Matrix7d arrayToMatrix7d(const std::array<double, 49>& data) {
  Matrix7d out;
  for (size_t i = 0; i < 7; ++i) {
    for (size_t j = 0; j < 7; ++j) {
      out(static_cast<int>(i), static_cast<int>(j)) = data[i * 7 + j];
    }
  }
  return out;
}

inline Vector7d arrayToVector7d(const std::array<double, 7>& data) {
  Vector7d out;
  for (size_t i = 0; i < 7; ++i) {
    out(static_cast<int>(i)) = data[i];
  }
  return out;
}

}  // namespace

namespace cps_controllers {

double ReachableCartesianImpedanceController::computeNormalStiffnessAlongNormal(
    const Matrix6d& K) const {
  const Vector3d n = human_plane_normal_.normalized();
  return std::max((n.transpose() * K.topLeftCorner<3, 3>() * n)(0, 0), 0.0);
}

double ReachableCartesianImpedanceController::computeNormalDampingAlongNormal(
    const Matrix6d& D) const {
  const Vector3d n = human_plane_normal_.normalized();
  return std::max((n.transpose() * D.topLeftCorner<3, 3>() * n)(0, 0), 0.0);
}

double ReachableCartesianImpedanceController::computeConservativeNormalAccelPositiveBound(
    double m_eff_n,
    double e_n_abs,
    double /*v_n_abs*/,
    const Matrix6d& K_used) const {
  const double m_safe = std::max(m_eff_n, kSmallPositive);

  const double k_n_up = computeNormalStiffnessAlongNormal(K_used);
  const double a_from_stiffness = (k_n_up * e_n_abs) / m_safe;

  const double a_from_tau_rate = torque_to_accel_gain_ * tau_rate_limit_ * kMinDt;
  const double a_from_uncertainty = std::max(model_accel_uncertainty_, 0.0);

  return std::max(0.0, a_from_stiffness + a_from_tau_rate + a_from_uncertainty);
}

double ReachableCartesianImpedanceController::computeConservativeNormalBrakeAccelLowerBound(
    double m_eff_n,
    double v_n_abs,
    const Matrix6d& D_used) const {
  const double m_safe = std::max(m_eff_n, kSmallPositive);

  const double d_n_low = computeNormalDampingAlongNormal(D_used);
  const double a_from_damping = (d_n_low * v_n_abs) / m_safe;

  const double a_tau_loss = 0.5 * torque_to_accel_gain_ * tau_rate_limit_ * kMinDt;

  return std::max(0.0, a_from_damping - a_tau_loss);
}

double ReachableCartesianImpedanceController::distanceToSweptHandRegion(
    const Vector3d& x,
    const Vector3d& /*plane_normal*/,
    const Vector3d& sphere_center) const {
  const double sphere_radius = human_motion_radius_ + human_hand_radius_;
  return std::max((x - sphere_center).norm() - sphere_radius, 0.0);
}

double ReachableCartesianImpedanceController::estimatePathParameterTimeFromCurrentState(
    const Vector3d& current_position,
    double nominal_guess_time) const {
  const Vector3d p0 = desired_position_;
  const Matrix3d R0 = desired_orientation_.toRotationMatrix();
  const auto traj_type =
      parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);

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
    if (cost < best_cost) {
      best_cost = cost;
      best_t = t;
    }
  }
  return std::max(0.0, best_t);
}

double ReachableCartesianImpedanceController::estimatePathTimeRateFromCurrentState(
    double path_time_anchor,
    const Vector3d& current_linear_velocity) const {
  const Vector3d p0 = desired_position_;
  const Matrix3d R0 = desired_orientation_.toRotationMatrix();
  const auto traj_type =
      parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);

  const TaskRefPose ref = makeReferencePose(path_time_anchor, p0, R0, traj_type);
  const double denom = std::max(ref.dp.squaredNorm(), kSmallPositive);

  double rate = 0.0;
  if (denom > 1e-8) {
    rate = current_linear_velocity.dot(ref.dp) / denom;
  }

  if (!std::isfinite(rate)) {
    rate = 0.0;
  }

  return std::clamp(rate, path_time_rate_min_, path_time_rate_max_);
}

ReachableCartesianImpedanceController::PathState
ReachableCartesianImpedanceController::propagateOnlinePathState(
    const Vector3d& current_position,
    const Vector3d& current_linear_velocity,
    double nominal_guess_time,
    double dtp) const {
  PathState out;
  out.t_path = estimatePathParameterTimeFromCurrentState(current_position, nominal_guess_time);
  const double rate0 = estimatePathTimeRateFromCurrentState(out.t_path, current_linear_velocity);

  const double rate_err = path_time_rate_target_ - rate0;
  const double accel_cmd = std::clamp(rate_err / std::max(dtp, kMinDt),
                                      -path_time_acc_limit_, path_time_acc_limit_);
  out.accel = accel_cmd;
  out.rate = std::clamp(rate0 + dtp * accel_cmd, path_time_rate_min_, path_time_rate_max_);
  out.t_path = std::max(0.0, out.t_path + dtp * out.rate);
  return out;
}

void ReachableCartesianImpedanceController::publishRvizDiagnostics(
    double wall_time,
    const Vector3d& current_position,
    const Vector3d& desired_position_cur,
    const Vector6d& ee_twist,
    const MonitorResult& monitor) {
  (void)ee_twist;

  if (!rviz_enable_markers_ || !rviz_marker_pub_) {
    return;
  }

  ++rviz_publish_counter_;
  if ((rviz_publish_counter_ % std::max(1, rviz_marker_decimation_)) != 0) {
    return;
  }

  MarkerArray arr;
  const auto stamp = get_node()->now();
  const std::string& frame_id = rviz_frame_id_;
  const Vector3d n = human_plane_normal_.normalized();

  auto setupMarker = [&](Marker& m, int id, const std::string& ns, int type) {
    m.header.frame_id = frame_id;
    m.header.stamp = stamp;
    m.ns = ns;
    m.id = id;
    m.type = type;
    m.action = Marker::ADD;
    m.lifetime = rclcpp::Duration::from_seconds(rviz_marker_lifetime_sec_);
  };

  const Matrix3d R_plane = makePlaneFrameFromNormal(n);
  const Quaterniond q_plane = rotationMatrixToQuaternion(R_plane);

  {
    Marker m;
    setupMarker(m, 0, "reachable_plane", Marker::CUBE);
    m.pose.position = toPoint(human_sphere_center_);
    m.pose.orientation = toQuatMsg(q_plane);
    m.scale.x = rviz_plane_size_;
    m.scale.y = rviz_plane_size_;
    m.scale.z = rviz_plane_thickness_;
    m.color = makeColor(0.1f, 0.6f, 1.0f, 0.12f);
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 1, "reachable_plane_normal", Marker::ARROW);
    m.scale.x = rviz_arrow_shaft_diameter_;
    m.scale.y = rviz_arrow_head_diameter_;
    m.scale.z = rviz_arrow_head_length_;
    m.color = makeColor(0.1f, 0.6f, 1.0f, 0.9f);

    const Vector3d p0 = human_sphere_center_;
    const Vector3d p1 = p0 + rviz_normal_arrow_length_ * n;
    m.points.push_back(toPoint(p0));
    m.points.push_back(toPoint(p1));
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 2, "reachable_hand_motion_sphere", Marker::SPHERE);
    m.pose.position = toPoint(human_sphere_center_);
    m.pose.orientation.w = 1.0;
    m.scale.x = 2.0 * human_motion_radius_;
    m.scale.y = 2.0 * human_motion_radius_;
    m.scale.z = 2.0 * human_motion_radius_;
    m.color = makeColor(1.0f, 0.8f, 0.1f, 0.20f);
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 3, "reachable_hand_inflated_sphere", Marker::SPHERE);
    m.pose.position = toPoint(human_sphere_center_);
    m.pose.orientation.w = 1.0;
    const double inflated_radius = human_motion_radius_ + human_hand_radius_;
    m.scale.x = 2.0 * inflated_radius;
    m.scale.y = 2.0 * inflated_radius;
    m.scale.z = 2.0 * inflated_radius;
    m.color = makeColor(1.0f, 0.3f, 0.2f, 0.10f);
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 4, "reachable_ee_center", Marker::SPHERE);
    m.pose.position = toPoint(current_position);
    m.pose.orientation.w = 1.0;
    m.scale.x = 0.03;
    m.scale.y = 0.03;
    m.scale.z = 0.03;
    m.color = makeColor(0.0f, 1.0f, 0.2f, 0.9f);
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 5, "reachable_ee_radius", Marker::SPHERE);
    m.pose.position = toPoint(current_position);
    m.pose.orientation.w = 1.0;
    const double inflated_r = ee_collision_radius_ + monitor.current_pos_error_radius;
    m.scale.x = 2.0 * inflated_r;
    m.scale.y = 2.0 * inflated_r;
    m.scale.z = 2.0 * inflated_r;
    m.color = makeColor(0.0f, 1.0f, 0.2f, 0.15f);
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 6, "reachable_desired", Marker::SPHERE);
    m.pose.position = toPoint(desired_position_cur);
    m.pose.orientation.w = 1.0;
    m.scale.x = 0.025;
    m.scale.y = 0.025;
    m.scale.z = 0.025;
    m.color = makeColor(1.0f, 1.0f, 0.0f, 0.85f);
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 7, "reachable_vn_now", Marker::ARROW);
    m.scale.x = rviz_arrow_shaft_diameter_;
    m.scale.y = rviz_arrow_head_diameter_;
    m.scale.z = rviz_arrow_head_length_;

    const double vn = monitor.v_n_now;
    const Vector3d p0 = current_position;
    const Vector3d p1 = p0 + rviz_velocity_arrow_scale_ * vn * n;

    m.points.push_back(toPoint(p0));
    m.points.push_back(toPoint(p1));
    m.color = (vn >= 0.0) ? makeColor(1.0f, 0.5f, 0.0f, 0.95f)
                          : makeColor(0.8f, 0.2f, 1.0f, 0.95f);
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 8, "reachable_vn_fs_ub", Marker::ARROW);
    m.scale.x = rviz_arrow_shaft_diameter_;
    m.scale.y = rviz_arrow_head_diameter_;
    m.scale.z = rviz_arrow_head_length_;
    m.color = makeColor(1.0f, 0.0f, 0.0f, 0.95f);

    const Vector3d p0 = current_position + Vector3d(0.0, 0.0, rviz_ub_arrow_z_offset_);
    const Vector3d p1 =
        p0 + rviz_velocity_arrow_scale_ * monitor.worst_case_v_n_fs_ub * n;

    m.points.push_back(toPoint(p0));
    m.points.push_back(toPoint(p1));
    arr.markers.push_back(m);
  }

  {
    Marker m;
    setupMarker(m, 9, "reachable_nominal_contact", Marker::SPHERE);
    if (monitor.nominal_contact_sample_found) {
      m.pose.position = toPoint(monitor.nominal_contact_point_world);
      m.pose.orientation.w = 1.0;
      m.scale.x = 0.035;
      m.scale.y = 0.035;
      m.scale.z = 0.035;
      m.color = makeColor(1.0f, 0.0f, 1.0f, 0.95f);
      arr.markers.push_back(m);
    } else {
      m.action = Marker::DELETE;
      arr.markers.push_back(m);
    }
  }

  {
    Marker m;
    setupMarker(m, 10, "reachable_status_text", Marker::TEXT_VIEW_FACING);
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
       << "  Tn_fs_ub=" << monitor.worst_case_Tn_fs_ub
       << "  trig=" << static_cast<int>(monitor.predicted_trigger)
       << "  wall_t=" << wall_time;
    m.text = ss.str();
    arr.markers.push_back(m);
  }

  rviz_marker_pub_->publish(arr);
}

controller_interface::InterfaceConfiguration
ReachableCartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
ReachableCartesianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(name);
  }
  return config;
}

void ReachableCartesianImpedanceController::updateRuntimeGains(const Matrix6d& K_target,
                                                               const Matrix6d& D_target,
                                                               double dt) {
  const double dt_safe = std::max(dt, kMinDt);
  const double alpha = std::clamp(dt_safe / std::max(gain_filter_tau_, kMinDt), 0.0, 1.0);

  K_runtime_ = (1.0 - alpha) * K_runtime_ + alpha * K_target;
  D_runtime_ = (1.0 - alpha) * D_runtime_ + alpha * D_target;
}

void ReachableCartesianImpedanceController::enterFailsafe(
    double t_now,
    const Vector3d& desired_position_cur,
    const Quaterniond& desired_orientation_cur) {
  mode_ = SafetyMode::kFailsafe;
  failsafe_start_time_sec_ = t_now;
  failsafe_enter_wall_time_sec_ = t_now;
  frozen_desired_position_ = desired_position_cur;
  frozen_desired_orientation_ = desired_orientation_cur;
  frozen_desired_orientation_.normalize();
  prev_Tn_fs_valid_ = false;
}

void ReachableCartesianImpedanceController::leaveFailsafe(double t_now) {
  if (failsafe_enter_wall_time_sec_ >= 0.0) {
    paused_nominal_time_sec_ += std::max(0.0, t_now - failsafe_enter_wall_time_sec_);
  }
  mode_ = SafetyMode::kNominal;
  prev_Tn_fs_valid_ = false;
  failsafe_enter_wall_time_sec_ = -1.0;
}

void ReachableCartesianImpedanceController::buildReference(
    double nominal_time,
    Vector3d& desired_position_cur,
    Quaterniond& desired_orientation_cur,
    Vector3d& desired_linear_velocity_cur,
    Vector3d& desired_linear_acceleration_cur) {
  const Vector3d p0 = desired_position_;
  const Matrix3d R0 = desired_orientation_.toRotationMatrix();
  const auto traj_type =
      parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);
  const TaskRefPose ref = makeReferencePose(nominal_time, p0, R0, traj_type);

  desired_position_cur = ref.p;
  desired_orientation_cur = Quaterniond(ref.R);
  desired_orientation_cur.normalize();
  desired_linear_velocity_cur = ref.dp;
  desired_linear_acceleration_cur = ref.ddp;
}

ImpedanceSample ReachableCartesianImpedanceController::makeNominalSample(
    double nominal_time,
    double path_rate,
    double path_accel,
    const Matrix6d& K_cmd,
    const Matrix6d& D_cmd) const {
  const Vector3d p0 = desired_position_;
  const Matrix3d R0 = desired_orientation_.toRotationMatrix();
  const auto traj_type =
      parseTrajectoryTypeOrDefault(use_constant_reference_, reference_trajectory_type_);
  const TaskRefPose ref = makeReferencePose(nominal_time, p0, R0, traj_type);

  ImpedanceSample s;
  s.t = nominal_time;
  s.p = ref.p;
  s.dp = ref.dp * path_rate;
  s.ddp = ref.ddp * path_rate * path_rate + ref.dp * path_accel;
  s.q = Quaterniond(ref.R);
  s.q.normalize();
  s.w.setZero();
  s.dw.setZero();
  s.K = K_cmd;
  s.D = D_cmd;
  s.failsafe = false;
  return s;
}

ImpedanceSample ReachableCartesianImpedanceController::makeFrozenFailsafeSample(
    double nominal_time,
    const ImpedanceSample& freeze_sample,
    const Matrix6d& K_cmd,
    const Matrix6d& D_cmd) const {
  ImpedanceSample s;
  s.t = nominal_time;
  s.p = freeze_sample.p;
  s.dp.setZero();
  s.ddp.setZero();
  s.q = freeze_sample.q;
  s.q.normalize();
  s.w.setZero();
  s.dw.setZero();
  s.K = K_cmd;
  s.D = D_cmd;
  s.failsafe = true;
  return s;
}

VerifiedPlan ReachableCartesianImpedanceController::buildCandidatePlan(
    double wall_time,
    double nominal_guess_time,
    const Vector3d& current_position,
    const Quaterniond& /*current_orientation*/,
    const Vector6d& ee_twist) const {
  VerifiedPlan plan;
  plan.valid = false;
  plan.generated_wall_time = wall_time;
  plan.nominal_time_anchor = nominal_guess_time;
  plan.failsafe_exec_index = 0;

  const int N_fs = std::max(1, shield_horizon_steps_);
  const double dtp = std::max(shield_plan_dt_, kMinDt);
  const double alpha_plan =
      std::clamp(dtp / std::max(gain_filter_tau_, kMinDt), 0.0, 1.0);

  Matrix6d K_plan = K_runtime_;
  Matrix6d D_plan = D_runtime_;

  const double t_anchor =
      estimatePathParameterTimeFromCurrentState(current_position, nominal_guess_time);
  const double rate_anchor =
      estimatePathTimeRateFromCurrentState(t_anchor, ee_twist.head<3>());

  plan.anchor = makeNominalSample(t_anchor, rate_anchor, 0.0, K_plan, D_plan);

  const PathState path_state = propagateOnlinePathState(
      current_position, ee_twist.head<3>(), nominal_guess_time, dtp);

  K_plan = (1.0 - alpha_plan) * K_plan + alpha_plan * K_nominal_;
  D_plan = (1.0 - alpha_plan) * D_plan + alpha_plan * D_nominal_;

  const ImpedanceSample intended =
      makeNominalSample(path_state.t_path, path_state.rate, path_state.accel, K_plan, D_plan);
  plan.intended.push_back(intended);

  const ImpedanceSample freeze_anchor = intended;
  for (int i = 0; i < N_fs; ++i) {
    const double tk = intended.t + static_cast<double>(i + 1) * dtp;
    const double s = static_cast<double>(i + 1) / static_cast<double>(N_fs);
    const Matrix6d K_sched = (1.0 - s) * K_nominal_ + s * K_f_target_;
    const Matrix6d D_sched = (1.0 - s) * D_nominal_ + s * D_f_target_;

    K_plan = (1.0 - alpha_plan) * K_plan + alpha_plan * K_sched;
    D_plan = (1.0 - alpha_plan) * D_plan + alpha_plan * D_sched;

    plan.failsafe.push_back(makeFrozenFailsafeSample(tk, freeze_anchor, K_plan, D_plan));
  }

  return plan;
}

MonitorResult ReachableCartesianImpedanceController::verifyCandidatePlan(
    const VerifiedPlan& plan,
    const Vector3d& current_position,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Vector3d& plane_normal,
    const Vector3d& sphere_center) const {
  MonitorResult out;

  const Vector3d n = plane_normal.normalized();
  Vector3d x_pred = current_position;
  Vector3d v_pred = ee_twist.head<3>();

  Matrix3d lambda_v_inv = Jv * inertia.inverse() * Jv.transpose();
  lambda_v_inv.diagonal().array() += kLambdaReg;
  const Matrix3d lambda_v = lambda_v_inv.inverse();

  const double denom0 = (n.transpose() * lambda_v_inv * n)(0, 0);
  out.m_eff_n = 1.0 / std::max(denom0, kSmallPositive);

  const double v_n_now = n.dot(v_pred);
  out.v_n_now = v_n_now;
  out.Tn_now = 0.5 * out.m_eff_n * v_n_now * v_n_now;
  out.v_safe = std::sqrt(
      std::max(2.0 * safe_collision_energy_joule_ / std::max(out.m_eff_n, kSmallPositive), 0.0));

  const Vector3d p_nom0 = plan.anchor.p;
  const Vector3d v_nom0 = plan.anchor.dp;

  double r_p = (current_position - p_nom0).norm();
  double r_v = (ee_twist.head<3>() - v_nom0).norm();

  out.current_pos_error_radius = r_p;
  out.current_vel_error_radius = r_v;
  out.worst_case_pos_error_radius = r_p;
  out.worst_case_vel_error_radius = r_v;

  const double inflated_r_now = ee_collision_radius_ + r_p;
  out.plane_distance_now =
      distanceToSweptHandRegion(x_pred, plane_normal, sphere_center) - inflated_r_now;
  out.plane_distance_min_nominal = out.plane_distance_now;

  double worst_Tn_fs_ub = -std::numeric_limits<double>::infinity();

  auto propagate_error_tube = [&](double dtp) {
    const double rp_dot = r_v;
    const double rv_dot =
        std::max(error_pos_gain_alpha_, 0.0) * r_p -
        std::max(error_vel_gain_beta_, 0.0) * r_v +
        std::max(error_acc_disturbance_gamma_, 0.0);

    r_p = std::max(0.0, r_p + dtp * rp_dot);
    r_v = std::max(0.0, r_v + dtp * rv_dot);

    out.worst_case_pos_error_radius = std::max(out.worst_case_pos_error_radius, r_p);
    out.worst_case_vel_error_radius = std::max(out.worst_case_vel_error_radius, r_v);
  };

  auto eval_sample =
      [&](const ImpedanceSample& s, const Matrix6d& K_used, const Matrix6d& D_used, double dtp) {
        const double denom_step = (n.transpose() * lambda_v_inv * n)(0, 0);
        const double m_eff_step_n = 1.0 / std::max(denom_step, kSmallPositive);

        const Vector3d e_nom = s.p - x_pred;
        const Vector3d v_err = v_pred - s.dp;

        const Vector3d a_pred =
            lambda_v *
            (K_used.topLeftCorner<3, 3>() * e_nom -
             D_used.topLeftCorner<3, 3>() * v_err);

        const Vector3d a_pred_bounded =
            a_pred + std::max(model_accel_uncertainty_, 0.0) * n;

        const Vector3d x_next =
            x_pred + v_pred * dtp + 0.5 * a_pred_bounded * dtp * dtp;
        const Vector3d v_next = v_pred + a_pred_bounded * dtp;

        propagate_error_tube(dtp);

        const double inflated_r = ee_collision_radius_ + r_p;

        const double d_region_pred =
            distanceToSweptHandRegion(x_pred, plane_normal, sphere_center) - inflated_r;
        const double d_region_next =
            distanceToSweptHandRegion(x_next, plane_normal, sphere_center) - inflated_r;

        out.plane_distance_min_nominal =
            std::min(out.plane_distance_min_nominal, d_region_next);

        if (!out.nominal_contact_sample_found && d_region_pred > 0.0 && d_region_next <= 0.0) {
          const double alpha =
              std::clamp(d_region_pred / std::max(d_region_pred - d_region_next, kSmallPositive),
                         0.0, 1.0);

          const Vector3d x_contact = x_pred + alpha * (x_next - x_pred);
          const Vector3d v_contact = v_pred + alpha * (v_next - v_pred);

          const double v_n_contact = std::abs(n.dot(v_contact)) + r_v;

          out.nominal_contact_sample_found = true;
          out.nominal_contact_time = s.t;
          out.nominal_contact_distance =
              distanceToSweptHandRegion(x_contact, plane_normal, sphere_center) - inflated_r;
          out.v_n_contact_nominal = v_n_contact;
          out.Tn_contact_nominal = 0.5 * m_eff_step_n * v_n_contact * v_n_contact;
          out.nominal_contact_point_world = x_contact;
        }

        const double e_n_abs =
            std::abs(n.dot(x_pred - s.p)) +
            r_p +
            std::max(stiffness_error_bound_m_, 0.0);

        const double v_n_abs = std::abs(n.dot(v_pred)) + r_v;

        const double a_pos =
            computeConservativeNormalAccelPositiveBound(m_eff_step_n, e_n_abs, v_n_abs, K_used);

        const double brake_credit_scale = s.failsafe ? 1.0 : 0.35;
        const double a_brake =
            brake_credit_scale *
            computeConservativeNormalBrakeAccelLowerBound(m_eff_step_n, v_n_abs, D_used);

        const double a_net = a_pos - a_brake;

        double v_n_fs_ub = std::abs(n.dot(v_next)) + r_v;
        v_n_fs_ub += dtp * std::max(a_net, 0.0);
        v_n_fs_ub = std::max(0.0, v_n_fs_ub);

        const double Tn_fs_ub = 0.5 * m_eff_step_n * v_n_fs_ub * v_n_fs_ub;

        if (!out.worst_case_candidate_found || Tn_fs_ub > worst_Tn_fs_ub) {
          out.worst_case_candidate_found = true;
          worst_Tn_fs_ub = Tn_fs_ub;

          out.worst_case_candidate_time = s.t;
          out.worst_case_plane_distance_at_candidate = d_region_pred;
          out.worst_case_nominal_forward_progress = n.dot(x_pred - current_position);
          out.worst_case_v_n_fs_ub = v_n_fs_ub;
          out.worst_case_Tn_fs_ub = Tn_fs_ub;
          out.worst_case_a_pos = a_pos;
          out.worst_case_a_brake = a_brake;
          out.worst_case_a_net = a_net;
        }

        x_pred = x_next;
        v_pred = v_next;
      };

  double t_prev = plan.anchor.t;
  for (const auto& s : plan.intended) {
    const double dtp = std::max(s.t - t_prev, kMinDt);
    eval_sample(s, s.K, s.D, dtp);
    t_prev = s.t;
  }
  for (const auto& s : plan.failsafe) {
    const double dtp = std::max(s.t - t_prev, kMinDt);
    eval_sample(s, s.K, s.D, dtp);
    t_prev = s.t;
  }

  out.contact_possible_nominal = out.nominal_contact_sample_found;
  out.contact_possible_hybrid = out.worst_case_candidate_found;
  out.h_geom = out.plane_distance_min_nominal;

  if (out.nominal_contact_sample_found) {
    out.h_nominal_energy = safe_collision_energy_joule_ - out.Tn_contact_nominal;
  } else {
    out.h_nominal_energy = std::numeric_limits<double>::infinity();
  }

  if (out.worst_case_candidate_found) {
    out.h_failsafe_energy = safe_collision_energy_joule_ - out.worst_case_Tn_fs_ub;
  } else {
    out.h_failsafe_energy = std::numeric_limits<double>::infinity();
  }

  out.unsafe_contact_nominal =
      out.nominal_contact_sample_found && (out.h_nominal_energy < 0.0);

  out.unsafe_contact_hybrid =
      out.worst_case_candidate_found && (out.h_failsafe_energy < 0.0);

  out.predicted_trigger =
      out.unsafe_contact_nominal || out.unsafe_contact_hybrid;

  return out;
}

MonitorResult ReachableCartesianImpedanceController::runSafetyMonitor(
    double dt,
    double t,
    const Vector3d& current_position,
    const Vector3d& desired_position_cur,
    const Vector6d& ee_twist,
    const Vector3d& desired_linear_velocity_cur,
    const Vector3d& desired_linear_acceleration_cur,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Vector3d& plane_normal,
    const Vector3d& sphere_center) {
  (void)dt;
  (void)desired_linear_acceleration_cur;

  VerifiedPlan trivial_plan;
  trivial_plan.valid = false;
  trivial_plan.anchor.t = t;
  trivial_plan.anchor.p = desired_position_cur;
  trivial_plan.anchor.dp = desired_linear_velocity_cur;
  trivial_plan.anchor.ddp.setZero();
  trivial_plan.anchor.q = desired_orientation_;
  trivial_plan.anchor.K = K_runtime_;
  trivial_plan.anchor.D = D_runtime_;
  trivial_plan.anchor.failsafe = false;

  ImpedanceSample s0 = trivial_plan.anchor;
  trivial_plan.intended.push_back(s0);

  Matrix6d K_plan = K_runtime_;
  Matrix6d D_plan = D_runtime_;
  const double dtp = std::max(shield_plan_dt_, kMinDt);
  const double alpha_plan =
      std::clamp(dtp / std::max(gain_filter_tau_, kMinDt), 0.0, 1.0);

  for (int k = 0; k < std::max(1, shield_horizon_steps_); ++k) {
    const double tk = t + (k + 1) * dtp;
    const double s = static_cast<double>(k + 1) /
                     static_cast<double>(std::max(1, shield_horizon_steps_));
    const Matrix6d K_sched = (1.0 - s) * K_nominal_ + s * K_f_target_;
    const Matrix6d D_sched = (1.0 - s) * D_nominal_ + s * D_f_target_;
    K_plan = (1.0 - alpha_plan) * K_plan + alpha_plan * K_sched;
    D_plan = (1.0 - alpha_plan) * D_plan + alpha_plan * D_sched;
    trivial_plan.failsafe.push_back(makeFrozenFailsafeSample(tk, s0, K_plan, D_plan));
  }

  return verifyCandidatePlan(
      trivial_plan, current_position, ee_twist, inertia, Jv, plane_normal, sphere_center);
}

ShieldDecision ReachableCartesianImpedanceController::computeShieldDecision(
    double wall_time,
    double nominal_guess_time,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv) {
  ShieldDecision dec;

  const VerifiedPlan candidate = buildCandidatePlan(
      wall_time, nominal_guess_time, current_position, current_orientation, ee_twist);
  dec.monitor = verifyCandidatePlan(
      candidate, current_position, ee_twist, inertia, Jv,
      human_plane_normal_, human_sphere_center_);

  dec.candidate_verified = !enable_safety_monitor_ || !dec.monitor.predicted_trigger;

  last_path_anchor_time_ = candidate.anchor.t;
  last_path_rate_ = candidate.anchor.dp.norm() > 0.0 ? candidate.intended.front().dp.norm() /
                                                           std::max(candidate.anchor.dp.norm(), kSmallPositive)
                                                     : 0.0;
  last_intended_path_time_ = candidate.intended.empty() ? candidate.anchor.t : candidate.intended.front().t;

  if (dec.candidate_verified) {
    last_verified_plan_ = candidate;
    last_verified_plan_.valid = true;
    last_verified_plan_.failsafe_exec_index = 0;

    dec.executing_last_verified_failsafe = false;
    mode_ = SafetyMode::kNominal;

    if (!last_verified_plan_.intended.empty()) {
      dec.command = last_verified_plan_.intended.front();
    } else if (!last_verified_plan_.failsafe.empty()) {
      dec.command = last_verified_plan_.failsafe.front();
      dec.executing_last_verified_failsafe = true;
      mode_ = SafetyMode::kFailsafe;
    }
  } else {
    dec.executing_last_verified_failsafe = true;
    mode_ = SafetyMode::kFailsafe;

    if (last_verified_plan_.valid && !last_verified_plan_.failsafe.empty()) {
      dec.command = last_verified_plan_.failsafe.front();
    } else {
      ImpedanceSample emergency;
      emergency.t = wall_time;
      emergency.p = current_position;
      emergency.dp.setZero();
      emergency.ddp.setZero();
      emergency.q = current_orientation;
      emergency.q.normalize();
      emergency.w.setZero();
      emergency.dw.setZero();
      emergency.K = K_f_target_;
      emergency.D = D_f_target_;
      emergency.failsafe = true;
      dec.command = emergency;
    }
  }

  return dec;
}

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

  Vector7d tau_task = Vector7d::Zero();
  const Matrix6d& K_cmd = cmd.K;
  const Matrix6d& D_cmd = cmd.D;

  if (use_dynamic_consistent_impedance_) {
    Matrix6d lambda_inv = J_geo * inertia.inverse() * J_geo.transpose();
    lambda_inv.diagonal().array() += 1e-8;
    const Matrix6d lambda = lambda_inv.inverse();

    const Vector6d wrench =
        lambda * (xddot_des - K_cmd * error - D_cmd * xdot_error);

    tau_task = J_geo.transpose() * wrench;
  } else {
    tau_task = J_geo.transpose() * (-K_cmd * error - D_cmd * xdot_error);
  }

  const Eigen::MatrixXd Jt_pinv = dampedPseudoInverse(J_geo.transpose());
  const Vector7d tau_nullspace =
      (Matrix7d::Identity() - J_geo.transpose() * Jt_pinv) *
      (n_stiffness_ * (desired_qn_ - q) -
       (2.0 * std::sqrt(std::max(n_stiffness_, 0.0))) * dq);

  const Vector7d tau_nullspace_eff =
      (cmd.failsafe && disable_nullspace_in_failsafe_) ? Vector7d::Zero() : tau_nullspace;

  const Vector7d tau_des = tau_task + coriolis + tau_nullspace_eff;

  const double max_delta = torque_rate_limit_ * std::max(dt, kMinDt);

  Vector7d tau_cmd = tau_cmd_prev_;
  for (int i = 0; i < 7; ++i) {
    const double delta =
        std::clamp(tau_des(i) - tau_cmd_prev_(i), -max_delta, max_delta);
    tau_cmd(i) = tau_cmd_prev_(i) + delta;
  }

  tau_cmd_prev_ = tau_cmd;
  return tau_cmd;
}

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

  const Vector3d current_position = pose.block<3, 1>(0, 3);
  const Matrix3d current_rotation = pose.block<3, 3>(0, 0);
  Quaterniond current_orientation(current_rotation);
  current_orientation.normalize();

  Matrix67d J_geo(
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector).data());
  const Vector6d ee_twist = J_geo * dq;
  const Matrix37d Jv = J_geo.topRows<3>();

  double paused_total = paused_nominal_time_sec_;
  if (failsafe_enter_wall_time_sec_ >= 0.0) {
    paused_total += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
  }
  const double nominal_guess_time = std::max(0.0, wall_time - paused_total);

  ShieldDecision shield_dec;
  const bool do_monitor =
      !last_shield_decision_valid_ || ((monitor_counter_++ % std::max(1, monitor_decimation_)) == 0);

  if (do_monitor) {
    shield_dec = computeShieldDecision(
        wall_time, nominal_guess_time, current_position, current_orientation, ee_twist, inertia, Jv);
    last_shield_decision_ = shield_dec;
    last_shield_decision_valid_ = true;
  } else {
    shield_dec = last_shield_decision_;
  }

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
  const Vector3d desired_linear_acceleration_cur = shield_dec.command.ddp;

  updateRuntimeGains(shield_dec.command.K, shield_dec.command.D, dt);

  Vector6d error = Vector6d::Zero();
  error.head<3>() = current_position - desired_position_cur;
  error.tail<3>() = computeOrientationError(current_orientation, desired_orientation_cur);

  MonitorResult monitor = shield_dec.monitor;

  {
    const Vector3d n = human_plane_normal_.normalized();
    const double v_n_now_fs =
        std::abs(n.dot(ee_twist.head<3>())) + monitor.current_vel_error_radius;
    const double Tn_now_fs =
        0.5 * std::max(monitor.m_eff_n, 0.0) * v_n_now_fs * v_n_now_fs;

    monitor.v_n_now_fs = v_n_now_fs;
    monitor.Tn_now_fs = Tn_now_fs;

    if (mode_ == SafetyMode::kFailsafe) {
      if (prev_Tn_fs_valid_) {
        monitor.Tn_dot_est = (Tn_now_fs - prev_Tn_fs_) / std::max(dt, kMinDt);
      } else {
        monitor.Tn_dot_est = 0.0;
      }
      prev_Tn_fs_ = Tn_now_fs;
      prev_Tn_fs_valid_ = true;
    } else {
      monitor.Tn_dot_est = 0.0;
      prev_Tn_fs_valid_ = false;
    }

    last_v_n_fs_ = v_n_now_fs;
    last_monitor_result_ = monitor;
    last_monitor_result_valid_ = true;
    last_monitor_wall_time_ = wall_time;
  }

  const Vector7d tau_cmd = computeImpedanceTorque(
      q, dq, inertia, coriolis, J_geo,
      current_position, current_orientation,
      shield_dec.command, dt);

  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }

  publishRvizDiagnostics(
      wall_time, current_position, desired_position_cur, ee_twist, monitor);

  if (enable_error_logging_ && error_log_file_.is_open()) {
    error_log_file_ << std::fixed << std::setprecision(9)
                    << wall_time << ","
                    << nominal_guess_time << ","
                    << paused_nominal_time_sec_ << ","
                    << static_cast<int>(mode_) << ","
                    << desired_position_cur(0) << "," << desired_position_cur(1) << ","
                    << desired_position_cur(2) << ","
                    << current_position(0) << "," << current_position(1) << ","
                    << current_position(2) << ","
                    << ee_twist(0) << "," << ee_twist(1) << "," << ee_twist(2) << ","
                    << ee_twist(3) << "," << ee_twist(4) << "," << ee_twist(5) << ","
                    << desired_linear_velocity_cur(0) << "," << desired_linear_velocity_cur(1)
                    << "," << desired_linear_velocity_cur(2) << ","
                    << error(0) << "," << error(1) << "," << error(2) << ","
                    << error(3) << "," << error(4) << "," << error(5) << ","
                    << 0.0 << ","
                    << 0.0 << ","
                    << 0.0 << ","
                    << tau_cmd.norm() << ","
                    << shield_dec.command.K(0, 0) << "," << shield_dec.command.K(1, 1) << ","
                    << shield_dec.command.K(2, 2) << ","
                    << shield_dec.command.D(0, 0) << "," << shield_dec.command.D(1, 1) << ","
                    << shield_dec.command.D(2, 2) << ","
                    << monitor.plane_distance_now << ","
                    << monitor.plane_distance_min_nominal << ","
                    << monitor.m_eff_n << ","
                    << monitor.v_n_now << ","
                    << monitor.Tn_now << ","
                    << monitor.v_safe << ","
                    << static_cast<int>(monitor.nominal_contact_sample_found) << ","
                    << monitor.nominal_contact_time << ","
                    << monitor.nominal_contact_distance << ","
                    << monitor.v_n_contact_nominal << ","
                    << monitor.Tn_contact_nominal << ","
                    << static_cast<int>(monitor.worst_case_candidate_found) << ","
                    << monitor.worst_case_candidate_time << ","
                    << monitor.worst_case_plane_distance_at_candidate << ","
                    << monitor.worst_case_nominal_forward_progress << ","
                    << monitor.worst_case_v_n_fs_ub << ","
                    << monitor.worst_case_Tn_fs_ub << ","
                    << monitor.worst_case_a_pos << ","
                    << monitor.worst_case_a_brake << ","
                    << monitor.worst_case_a_net << ","
                    << monitor.h_geom << ","
                    << monitor.h_nominal_energy << ","
                    << monitor.h_failsafe_energy << ","
                    << monitor.v_n_now_fs << ","
                    << monitor.Tn_now_fs << ","
                    << monitor.Tn_dot_est << ","
                    << monitor.current_pos_error_radius << ","
                    << monitor.current_vel_error_radius << ","
                    << monitor.worst_case_pos_error_radius << ","
                    << monitor.worst_case_vel_error_radius << ","
                    << static_cast<int>(monitor.contact_possible_nominal) << ","
                    << static_cast<int>(monitor.contact_possible_hybrid) << ","
                    << static_cast<int>(monitor.unsafe_contact_nominal) << ","
                    << static_cast<int>(monitor.unsafe_contact_hybrid) << ","
                    << static_cast<int>(monitor.predicted_trigger) << "\n";

    ++log_write_counter_;
    if ((log_write_counter_ % 200) == 0) {
      error_log_file_.flush();
    }
  }

  const auto toc_total = Clock::now();
  const double exec_ms =
      std::chrono::duration<double, std::milli>(toc_total - tic_total).count();

  exec_sum_ms_ += exec_ms;
  exec_max_ms_ = std::max(exec_max_ms_, exec_ms);
  exec_min_ms_ = std::min(exec_min_ms_, exec_ms);
  ++loop_counter_;

  if (loop_counter_ >= static_cast<std::size_t>(profiling_stats_print_period_)) {
    const double n = static_cast<double>(loop_counter_);
    RCLCPP_INFO(get_node()->get_logger(),
                "[reachable_impedance] mode=%d avg_exec=%.6f ms min_exec=%.6f ms max_exec=%.6f ms plan_valid=%d last_monitor_age=%.6f s",
                static_cast<int>(mode_),
                exec_sum_ms_ / n,
                exec_min_ms_,
                exec_max_ms_,
                static_cast<int>(last_verified_plan_.valid),
                wall_time - last_monitor_wall_time_);
    loop_counter_ = 0;
    exec_sum_ms_ = 0.0;
    exec_min_ms_ = 1e9;
    exec_max_ms_ = 0.0;
  }

  return controller_interface::return_type::OK;
}

CallbackReturn ReachableCartesianImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<bool>("enable_error_logging", true);
    auto_declare<std::string>(
        "error_log_path",
        "/home/developer/multipanda_ws/src/data_log/reachable_cartesian_impedance_validation.csv");

    auto_declare<bool>("use_constant_reference", false);
    auto_declare<std::string>("reference_trajectory_type", "lissajous");

    auto_declare<double>("nominal_pos_stiffness", 400.0);
    auto_declare<double>("nominal_rot_stiffness", 20.0);
    auto_declare<double>("failsafe_pos_stiffness", 50.0);
    auto_declare<double>("failsafe_rot_stiffness", 5.0);
    auto_declare<double>("failsafe_pos_damping_scale", 2.5);
    auto_declare<double>("failsafe_rot_damping_scale", 2.5);
    auto_declare<double>("gain_filter_tau", 0.03);
    auto_declare<double>("n_stiffness", 0.0);
    auto_declare<bool>("disable_nullspace_in_failsafe", true);

    auto_declare<bool>("enable_safety_monitor", true);

    auto_declare<double>("safe_collision_energy_joule", 0.20);
    auto_declare<double>("ee_collision_radius", 0.04);
    auto_declare<double>("monitor_nominal_horizon_sec", 0.03);
    auto_declare<int>("monitor_nominal_steps", 10);
    auto_declare<int>("monitor_decimation", 10);

    auto_declare<std::vector<double>>(
        "human_plane_normal", std::vector<double>{0.0, 0.0, 1.0});
    auto_declare<std::vector<double>>(
        "human_sphere_center", std::vector<double>{0.0, 0.0, 0.2});

    auto_declare<double>("human_motion_radius", 0.10);
    auto_declare<double>("human_hand_radius", 0.04);

    auto_declare<double>("return_to_nominal_energy_margin", 0.02);
    auto_declare<double>("return_to_nominal_speed_threshold", 0.02);
    auto_declare<double>("return_to_nominal_tndot_threshold", 0.0);
    auto_declare<double>("return_to_nominal_geom_margin", 0.005);

    auto_declare<double>("k_rate_limit", 5000.0);
    auto_declare<double>("d_rate_limit", 500.0);
    auto_declare<double>("tau_rate_limit", 1000.0);
    auto_declare<double>("torque_to_accel_gain", 8.0);
    auto_declare<double>("model_accel_uncertainty", 0.05);
    auto_declare<double>("stiffness_error_bound_m", 0.01);

    auto_declare<double>("error_pos_gain_alpha", 60.0);
    auto_declare<double>("error_vel_gain_beta", 20.0);
    auto_declare<double>("error_acc_disturbance_gamma", 0.20);

    auto_declare<int>("shield_horizon_steps", 20);
    auto_declare<double>("shield_plan_dt", 0.01);

    auto_declare<double>("path_retiming_search_window_sec", 0.25);
    auto_declare<int>("path_retiming_search_steps", 41);
    auto_declare<double>("path_time_rate_min", 0.0);
    auto_declare<double>("path_time_rate_max", 1.5);
    auto_declare<double>("path_time_acc_limit", 3.0);
    auto_declare<double>("path_time_rate_target", 1.0);

    auto_declare<bool>("use_dynamic_consistent_impedance", true);
    auto_declare<double>("torque_rate_limit", 1000.0);

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
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage: %s\n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn ReachableCartesianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    enable_error_logging_ = get_node()->get_parameter("enable_error_logging").as_bool();
    error_log_path_ = get_node()->get_parameter("error_log_path").as_string();

    use_constant_reference_ = get_node()->get_parameter("use_constant_reference").as_bool();
    reference_trajectory_type_ =
        get_node()->get_parameter("reference_trajectory_type").as_string();

    const double nominal_pos_stiffness =
        get_node()->get_parameter("nominal_pos_stiffness").as_double();
    const double nominal_rot_stiffness =
        get_node()->get_parameter("nominal_rot_stiffness").as_double();
    const double failsafe_pos_stiffness =
        get_node()->get_parameter("failsafe_pos_stiffness").as_double();
    const double failsafe_rot_stiffness =
        get_node()->get_parameter("failsafe_rot_stiffness").as_double();
    const double failsafe_pos_damping_scale =
        get_node()->get_parameter("failsafe_pos_damping_scale").as_double();
    const double failsafe_rot_damping_scale =
        get_node()->get_parameter("failsafe_rot_damping_scale").as_double();

    gain_filter_tau_ = get_node()->get_parameter("gain_filter_tau").as_double();
    n_stiffness_ = get_node()->get_parameter("n_stiffness").as_double();
    disable_nullspace_in_failsafe_ =
        get_node()->get_parameter("disable_nullspace_in_failsafe").as_bool();

    enable_safety_monitor_ = get_node()->get_parameter("enable_safety_monitor").as_bool();

    safe_collision_energy_joule_ =
        get_node()->get_parameter("safe_collision_energy_joule").as_double();
    ee_collision_radius_ = get_node()->get_parameter("ee_collision_radius").as_double();
    monitor_nominal_horizon_sec_ =
        get_node()->get_parameter("monitor_nominal_horizon_sec").as_double();
    monitor_nominal_steps_ =
        static_cast<int>(get_node()->get_parameter("monitor_nominal_steps").as_int());
    monitor_decimation_ =
        std::max(1, static_cast<int>(get_node()->get_parameter("monitor_decimation").as_int()));

    return_to_nominal_energy_margin_ =
        get_node()->get_parameter("return_to_nominal_energy_margin").as_double();
    return_to_nominal_speed_threshold_ =
        get_node()->get_parameter("return_to_nominal_speed_threshold").as_double();
    return_to_nominal_tndot_threshold_ =
        get_node()->get_parameter("return_to_nominal_tndot_threshold").as_double();
    return_to_nominal_geom_margin_ =
        get_node()->get_parameter("return_to_nominal_geom_margin").as_double();

    k_rate_limit_ = get_node()->get_parameter("k_rate_limit").as_double();
    d_rate_limit_ = get_node()->get_parameter("d_rate_limit").as_double();
    tau_rate_limit_ = get_node()->get_parameter("tau_rate_limit").as_double();
    torque_to_accel_gain_ = get_node()->get_parameter("torque_to_accel_gain").as_double();
    model_accel_uncertainty_ = get_node()->get_parameter("model_accel_uncertainty").as_double();
    stiffness_error_bound_m_ = get_node()->get_parameter("stiffness_error_bound_m").as_double();

    error_pos_gain_alpha_ =
        get_node()->get_parameter("error_pos_gain_alpha").as_double();
    error_vel_gain_beta_ =
        get_node()->get_parameter("error_vel_gain_beta").as_double();
    error_acc_disturbance_gamma_ =
        get_node()->get_parameter("error_acc_disturbance_gamma").as_double();

    shield_horizon_steps_ =
        std::max(1, static_cast<int>(get_node()->get_parameter("shield_horizon_steps").as_int()));
    shield_plan_dt_ =
        std::max(get_node()->get_parameter("shield_plan_dt").as_double(), kMinDt);

    path_retiming_search_window_sec_ =
        std::max(get_node()->get_parameter("path_retiming_search_window_sec").as_double(), 0.0);
    path_retiming_search_steps_ =
        std::max(5, static_cast<int>(get_node()->get_parameter("path_retiming_search_steps").as_int()));
    path_time_rate_min_ =
        get_node()->get_parameter("path_time_rate_min").as_double();
    path_time_rate_max_ =
        std::max(path_time_rate_min_,
                 get_node()->get_parameter("path_time_rate_max").as_double());
    path_time_acc_limit_ =
        std::max(0.0, get_node()->get_parameter("path_time_acc_limit").as_double());
    path_time_rate_target_ =
        std::clamp(get_node()->get_parameter("path_time_rate_target").as_double(),
                   path_time_rate_min_, path_time_rate_max_);

    use_dynamic_consistent_impedance_ =
        get_node()->get_parameter("use_dynamic_consistent_impedance").as_bool();
    torque_rate_limit_ =
        get_node()->get_parameter("torque_rate_limit").as_double();

    rviz_enable_markers_ = get_node()->get_parameter("rviz_enable_markers").as_bool();
    rviz_frame_id_ = get_node()->get_parameter("rviz_frame_id").as_string();
    rviz_marker_decimation_ =
        std::max(1, static_cast<int>(get_node()->get_parameter("rviz_marker_decimation").as_int()));
    rviz_marker_lifetime_sec_ =
        get_node()->get_parameter("rviz_marker_lifetime_sec").as_double();
    rviz_plane_size_ = get_node()->get_parameter("rviz_plane_size").as_double();
    rviz_plane_thickness_ = get_node()->get_parameter("rviz_plane_thickness").as_double();
    rviz_normal_arrow_length_ =
        get_node()->get_parameter("rviz_normal_arrow_length").as_double();
    rviz_velocity_arrow_scale_ =
        get_node()->get_parameter("rviz_velocity_arrow_scale").as_double();
    rviz_arrow_shaft_diameter_ =
        get_node()->get_parameter("rviz_arrow_shaft_diameter").as_double();
    rviz_arrow_head_diameter_ =
        get_node()->get_parameter("rviz_arrow_head_diameter").as_double();
    rviz_arrow_head_length_ =
        get_node()->get_parameter("rviz_arrow_head_length").as_double();
    rviz_text_scale_ = get_node()->get_parameter("rviz_text_scale").as_double();
    rviz_text_z_offset_ = get_node()->get_parameter("rviz_text_z_offset").as_double();
    rviz_ub_arrow_z_offset_ = get_node()->get_parameter("rviz_ub_arrow_z_offset").as_double();

    const auto normal_vec =
        get_node()->get_parameter("human_plane_normal").as_double_array();
    const auto center_vec =
        get_node()->get_parameter("human_sphere_center").as_double_array();

    if (normal_vec.size() != 3 || center_vec.size() != 3) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "human_plane_normal and human_sphere_center must have length 3.");
      return CallbackReturn::ERROR;
    }

    human_plane_normal_ = Vector3d(normal_vec[0], normal_vec[1], normal_vec[2]);
    human_sphere_center_ = Vector3d(center_vec[0], center_vec[1], center_vec[2]);

    human_motion_radius_ = get_node()->get_parameter("human_motion_radius").as_double();
    human_hand_radius_ = get_node()->get_parameter("human_hand_radius").as_double();

    if (human_motion_radius_ < 0.0 || human_hand_radius_ < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "human_motion_radius and human_hand_radius must be nonnegative.");
      return CallbackReturn::ERROR;
    }

    if (human_plane_normal_.norm() < 1e-8) {
      RCLCPP_ERROR(get_node()->get_logger(), "human_plane_normal norm too small.");
      return CallbackReturn::ERROR;
    }
    human_plane_normal_.normalize();

    profiling_stats_print_period_ = std::max<int>(
        1, static_cast<int>(get_node()->get_parameter("profiling_stats_print_period").as_int()));

    K_nominal_.setZero();
    D_nominal_.setZero();
    K_f_target_.setZero();
    D_f_target_.setZero();

    K_nominal_.topLeftCorner<3, 3>() = nominal_pos_stiffness * Matrix3d::Identity();
    K_nominal_.bottomRightCorner<3, 3>() = nominal_rot_stiffness * Matrix3d::Identity();

    D_nominal_.topLeftCorner<3, 3>() =
        2.0 * std::sqrt(std::max(nominal_pos_stiffness, 0.0)) * Matrix3d::Identity();
    D_nominal_.bottomRightCorner<3, 3>() =
        0.8 * 2.0 * std::sqrt(std::max(nominal_rot_stiffness, 0.0)) * Matrix3d::Identity();

    K_f_target_.topLeftCorner<3, 3>() = failsafe_pos_stiffness * Matrix3d::Identity();
    K_f_target_.bottomRightCorner<3, 3>() = failsafe_rot_stiffness * Matrix3d::Identity();

    D_f_target_.topLeftCorner<3, 3>() =
        failsafe_pos_damping_scale *
        2.0 * std::sqrt(std::max(failsafe_pos_stiffness, 0.0)) * Matrix3d::Identity();
    D_f_target_.bottomRightCorner<3, 3>() =
        failsafe_rot_damping_scale *
        2.0 * std::sqrt(std::max(failsafe_rot_stiffness, 0.0)) * Matrix3d::Identity();

    K_runtime_ = K_nominal_;
    D_runtime_ = D_nominal_;

    franka_robot_model_ =
        std::make_unique<franka_semantic_components::FrankaRobotModel>(
            franka_semantic_components::FrankaRobotModel(
                arm_id_ + "/robot_model", arm_id_));

    if (rviz_enable_markers_) {
      rviz_marker_pub_ =
          get_node()->create_publisher<MarkerArray>("reachable_impedance/markers", 10);
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception in on_configure: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn ReachableCartesianImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  start_time_ = this->get_node()->now();

  const Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
  desired_qn_ = q;

  const Eigen::Map<const Matrix4d> pose(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());

  desired_position_ = pose.block<3, 1>(0, 3);
  desired_orientation_ = Quaterniond(pose.block<3, 3>(0, 0));
  desired_orientation_.normalize();

  frozen_desired_position_ = desired_position_;
  frozen_desired_orientation_ = desired_orientation_;

  mode_ = SafetyMode::kNominal;
  failsafe_start_time_sec_ = -1.0;
  failsafe_enter_wall_time_sec_ = -1.0;
  paused_nominal_time_sec_ = 0.0;

  loop_counter_ = 0;
  exec_sum_ms_ = 0.0;
  exec_min_ms_ = 1e9;
  exec_max_ms_ = 0.0;
  log_write_counter_ = 0;

  prev_Tn_fs_ = 0.0;
  prev_Tn_fs_valid_ = false;
  last_v_n_fs_ = 0.0;

  monitor_counter_ = 0;
  last_monitor_result_valid_ = false;
  last_monitor_wall_time_ = 0.0;
  last_monitor_result_ = MonitorResult{};

  rviz_publish_counter_ = 0;

  last_verified_plan_ = VerifiedPlan{};
  tau_cmd_prev_.setZero();
  last_shield_decision_valid_ = false;

  if (enable_error_logging_) {
    error_log_file_.open(error_log_path_, std::ios::out | std::ios::trunc);
    if (!error_log_file_.is_open()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to open log file: %s", error_log_path_.c_str());
      return CallbackReturn::ERROR;
    }

    error_log_file_ << std::fixed << std::setprecision(9);
    error_log_file_
        << "wall_time_sec,nominal_time_sec,paused_nominal_time_sec,mode,"
        << "des_px,des_py,des_pz,cur_px,cur_py,cur_pz,"
        << "cur_vx,cur_vy,cur_vz,cur_wx,cur_wy,cur_wz,des_vx,des_vy,des_vz,"
        << "err_px,err_py,err_pz,err_rx,err_ry,err_rz,"
        << "tau_task_norm,tau_null_norm,tau_friction_norm,tau_cmd_norm,"
        << "Kx,Ky,Kz,Dx,Dy,Dz,"
        << "plane_distance_now,plane_distance_min_nominal,"
        << "m_eff_n,v_n_now,Tn_now,v_safe,"
        << "nominal_contact_sample_found,nominal_contact_time,nominal_contact_distance,"
        << "v_n_contact_nominal,Tn_contact_nominal,"
        << "worst_case_candidate_found,worst_case_candidate_time,worst_case_plane_distance_at_candidate,"
        << "worst_case_nominal_forward_progress,"
        << "worst_case_v_n_fs_ub,worst_case_Tn_fs_ub,"
        << "worst_case_a_pos,worst_case_a_brake,worst_case_a_net,"
        << "h_geom,h_nominal_energy,h_failsafe_energy,"
        << "v_n_now_fs,Tn_now_fs,Tn_dot_est,"
        << "current_pos_error_radius,current_vel_error_radius,"
        << "worst_case_pos_error_radius,worst_case_vel_error_radius,"
        << "contact_possible_nominal,contact_possible_hybrid,"
        << "unsafe_contact_nominal,unsafe_contact_hybrid,predicted_trigger\n";

    RCLCPP_INFO(get_node()->get_logger(),
                "Validation log enabled: %s", error_log_path_.c_str());
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn ReachableCartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (error_log_file_.is_open()) {
    error_log_file_.flush();
    error_log_file_.close();
  }
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace cps_controllers

PLUGINLIB_EXPORT_CLASS(cps_controllers::ReachableCartesianImpedanceController,
                       controller_interface::ControllerInterface)