// Copyright (c) 2026
// Reachable Cartesian Impedance Controller
//
// Goal B:
//   Contact/collision is allowed, but collision-related energy
//   must remain below a prescribed safe threshold.
//
// Uses only Franka model data for:
//   - pose
//   - zero Jacobian
//   - mass matrix
//   - coriolis vector
//
// Features
// --------
// 1) Per-cycle nominal / failsafe safety filter
// 2) Freeze current reference when entering failsafe
// 3) Reference trajectory switch:
//      - constant
//      - line
//      - lissajous
// 4) Safety monitor:
//      - future nominal contact energy uses predicted contact-time normal velocity
//      - scans candidate times within the nominal horizon
//      - computes worst-case failsafe energy over the horizon
// 5) Energy-aware monitor:
//      - h_nominal_energy = E_safe - Tn_contact_nominal
//      - h_failsafe_energy = E_safe - E_fs_worst
//      - current failsafe storage V_fs and finite-difference derivative
// 6) Safety filter with hysteresis
// 7) Nominal-time pause during failsafe
//
// Controller law
// --------------
// tau = J^T(-K e - D v) + coriolis + tau_null
//
// Notes
// -----
// - Contact is allowed in principle.
// - Geometry is used only to detect whether future contact is relevant,
//   not as a hard safety violation by itself.
// - Failsafe is triggered only by energy violation predictions.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
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

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>

namespace {

constexpr int kNumJoints = 7;
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

inline ReferenceTrajectoryType parseReferenceTrajectoryType(const std::string& name) {
  if (name == "lissajous") {
    return ReferenceTrajectoryType::kLissajous;
  }
  if (name == "constant") {
    return ReferenceTrajectoryType::kConstant;
  }
  return ReferenceTrajectoryType::kLine;
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

inline ReferenceTrajectoryType parseTrajectoryTypeOrDefault(
    bool use_constant_reference, const std::string& name) {
  if (use_constant_reference) {
    return ReferenceTrajectoryType::kConstant;
  }
  return parseReferenceTrajectoryType(name);
}

// ---------------- trajectory 1: straight line ----------------
inline TaskRefPose makeReferenceLine(double t,
                                     const Vector3d& p0,
                                     const Matrix3d& R0) {
  TaskRefPose ref;

  constexpr double T_move = 3.0;
  constexpr double dz = -0.55;

  const SmoothStartProfile ramp = makeSmoothStartProfile(t, T_move);

  ref.p = p0 + Vector3d(0.0, 0.0, dz * ramp.s);
  ref.dp = Vector3d(0.0, 0.0, dz * ramp.ds);
  ref.ddp = Vector3d(0.0, 0.0, dz * ramp.dds);
  ref.R = R0;
  return ref;
}

// ---------------- trajectory 2: lissajous ----------------
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

void ReachableCartesianImpedanceController::updateRuntimeGains(double dt) {
  const double dt_safe = std::max(dt, kMinDt);
  const double alpha = std::clamp(dt_safe / std::max(gain_filter_tau_, kMinDt), 0.0, 1.0);

  const Matrix6d& K_target = (mode_ == SafetyMode::kFailsafe) ? K_f_target_ : K_nominal_;
  const Matrix6d& D_target = (mode_ == SafetyMode::kFailsafe) ? D_f_target_ : D_nominal_;

  K_runtime_ = (1.0 - alpha) * K_runtime_ + alpha * K_target;
  D_runtime_ = (1.0 - alpha) * D_runtime_ + alpha * D_target;
}

void ReachableCartesianImpedanceController::enterFailsafe(
    double t_now,
    const Vector3d& desired_position_cur,
    const Quaterniond& desired_orientation_cur) {
  if (mode_ == SafetyMode::kFailsafe) {
    return;
  }

  mode_ = SafetyMode::kFailsafe;
  failsafe_start_time_sec_ = t_now;
  failsafe_enter_wall_time_sec_ = t_now;

  frozen_desired_position_ = desired_position_cur;
  frozen_desired_orientation_ = desired_orientation_cur;
  frozen_desired_orientation_.normalize();

  prev_V_fs_valid_ = false;

  RCLCPP_WARN(get_node()->get_logger(),
              "Entering failsafe: reference frozen at t = %.6f s", t_now);
}

void ReachableCartesianImpedanceController::leaveFailsafe(double t_now) {
  if (mode_ == SafetyMode::kNominal) {
    return;
  }

  if (failsafe_enter_wall_time_sec_ >= 0.0) {
    paused_nominal_time_sec_ += std::max(0.0, t_now - failsafe_enter_wall_time_sec_);
  }

  mode_ = SafetyMode::kNominal;
  prev_V_fs_valid_ = false;
  failsafe_enter_wall_time_sec_ = -1.0;

  RCLCPP_INFO(get_node()->get_logger(),
              "Returning to nominal mode at t = %.6f s (paused_nominal_time=%.6f s)",
              t_now, paused_nominal_time_sec_);
}

void ReachableCartesianImpedanceController::buildReference(
    double nominal_time,
    Vector3d& desired_position_cur,
    Quaterniond& desired_orientation_cur,
    Vector3d& desired_linear_velocity_cur,
    Vector3d& desired_linear_acceleration_cur) {
  if (mode_ == SafetyMode::kFailsafe) {
    desired_position_cur = frozen_desired_position_;
    desired_orientation_cur = frozen_desired_orientation_;
    desired_orientation_cur.normalize();
    desired_linear_velocity_cur.setZero();
    desired_linear_acceleration_cur.setZero();
    return;
  }

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
    const Vector3d& plane_point) {
  (void)dt;
  MonitorResult out;

  const Vector3d n = plane_normal.normalized();
  const Vector3d x0 = current_position;
  const Vector3d v0 = ee_twist.head<3>();
  const Vector3d vd = desired_linear_velocity_cur;
  const Vector3d ad = desired_linear_acceleration_cur;

  Matrix3d lambda_v_inv = Jv * inertia.inverse() * Jv.transpose();
  lambda_v_inv.diagonal().array() += kLambdaReg;
  const Matrix3d lambda_v = lambda_v_inv.inverse();

  const double denom = (n.transpose() * lambda_v_inv * n)(0, 0);
  out.m_eff_n = 1.0 / std::max(denom, kSmallPositive);

  out.v_n_now = n.dot(v0);
  out.Tn_now = 0.5 * out.m_eff_n * out.v_n_now * out.v_n_now;
  out.v_safe = std::sqrt(
      std::max(2.0 * safe_collision_energy_joule_ / std::max(out.m_eff_n, kSmallPositive), 0.0));

  out.plane_distance_now = n.dot(plane_point - x0) - ee_collision_radius_;
  out.plane_distance_min_nominal = out.plane_distance_now;

  const int N = std::max(1, monitor_nominal_steps_);
  const double h = std::max(monitor_nominal_horizon_sec_ / static_cast<double>(N), kMinDt);

  Vector3d x_pred = x0;
  Vector3d v_pred = v0;

  double worst_E_fs = -std::numeric_limits<double>::infinity();

  const double k_n_fs =
      std::max((n.transpose() * K_f_target_.topLeftCorner<3, 3>() * n)(0, 0), 0.0);
  const double d_n_fs =
      std::max((n.transpose() * D_f_target_.topLeftCorner<3, 3>() * n)(0, 0), 0.0);

  for (int i = 0; i < N; ++i) {
    const double t_i = t + static_cast<double>(i) * h;

    const Vector3d e_nom = desired_position_cur - x_pred;

    const Vector3d a_pred =
        ad + lambda_v *
                 (K_runtime_.topLeftCorner<3, 3>() * e_nom +
                  D_runtime_.topLeftCorner<3, 3>() * (vd - v_pred));

    const Vector3d x_next = x_pred + v_pred * h + 0.5 * a_pred * h * h;
    const Vector3d v_next = v_pred + a_pred * h;

    const double d_plane_pred = n.dot(plane_point - x_pred) - ee_collision_radius_;
    const double d_plane_next = n.dot(plane_point - x_next) - ee_collision_radius_;

    out.plane_distance_min_nominal = std::min(out.plane_distance_min_nominal, d_plane_next);

    // Nominal predicted contact sample
    if (!out.nominal_contact_sample_found && d_plane_pred > 0.0 && d_plane_next <= 0.0) {
      const double alpha =
          std::clamp(d_plane_pred / std::max(d_plane_pred - d_plane_next, kSmallPositive),
                     0.0, 1.0);
      const Vector3d x_contact = x_pred + alpha * (x_next - x_pred);
      const Vector3d v_contact = v_pred + alpha * (v_next - v_pred);
      const double v_n_contact = n.dot(v_contact);

      out.nominal_contact_sample_found = true;
      out.nominal_contact_time = t_i + alpha * h;
      out.nominal_contact_distance = n.dot(plane_point - x_contact) - ee_collision_radius_;
      out.v_n_contact_nominal = v_n_contact;
      out.Tn_contact_nominal = 0.5 * out.m_eff_n * v_n_contact * v_n_contact;
    }

    // Worst-case failsafe energy over horizon
    const double e_n_i = n.dot(x_pred - desired_position_cur);
    const double v_n_i = n.dot(v_pred);

    const double E_fs_i =
        0.5 * out.m_eff_n * v_n_i * v_n_i + 0.5 * k_n_fs * e_n_i * e_n_i;

    const double c_rate =
        std::max(d_n_fs / std::max(out.m_eff_n, kSmallPositive), 1e-3);

    double tau_safe_i = 0.0;
    double delta_n_fs_i = 0.0;

    if (E_fs_i > safe_collision_energy_joule_) {
      tau_safe_i = (1.0 / c_rate) *
                   std::log(std::max(E_fs_i / std::max(safe_collision_energy_joule_, 1e-9), 1.0));

      const double v_bound0_i =
          std::sqrt(std::max(2.0 * E_fs_i / std::max(out.m_eff_n, kSmallPositive), 0.0));

      delta_n_fs_i =
          (2.0 / c_rate) * v_bound0_i *
          (1.0 - std::exp(-0.5 * c_rate * tau_safe_i));
    }

    const double nominal_forward_progress = n.dot(x_pred - x0);
    const double hybrid_forward_reach = nominal_forward_progress + delta_n_fs_i;
    const double plane_distance_candidate =
        n.dot(plane_point - x_pred) - ee_collision_radius_;
    const double plane_margin_after_hybrid =
        plane_distance_candidate - delta_n_fs_i;

    if (!out.worst_case_candidate_found || E_fs_i > worst_E_fs) {
      out.worst_case_candidate_found = true;
      worst_E_fs = E_fs_i;

      out.worst_case_candidate_time = t_i;
      out.worst_case_plane_distance_at_candidate = plane_distance_candidate;
      out.worst_case_nominal_forward_progress = nominal_forward_progress;
      out.worst_case_e_n = e_n_i;
      out.worst_case_v_n = v_n_i;
      out.worst_case_E_fs_1d = E_fs_i;
      out.worst_case_tau_safe = tau_safe_i;
      out.worst_case_delta_n_fs = delta_n_fs_i;
      out.worst_case_hybrid_forward_reach = hybrid_forward_reach;
      out.worst_case_plane_margin_after_hybrid = plane_margin_after_hybrid;
    }

    x_pred = x_next;
    v_pred = v_next;
  }

  out.contact_possible_nominal = out.nominal_contact_sample_found;
  out.contact_possible_hybrid = out.worst_case_candidate_found;

  // Geometry is no longer a hard unsafe condition.
  // Keep it only as diagnostic information.
  out.h_geom = out.worst_case_plane_margin_after_hybrid;

  if (out.nominal_contact_sample_found) {
    out.h_nominal_energy = safe_collision_energy_joule_ - out.Tn_contact_nominal;
  } else {
    out.h_nominal_energy = std::numeric_limits<double>::infinity();
  }

  if (out.worst_case_candidate_found) {
    out.h_failsafe_energy = safe_collision_energy_joule_ - out.worst_case_E_fs_1d;
  } else {
    out.h_failsafe_energy = std::numeric_limits<double>::infinity();
  }

  // Unsafe only by energy, not by geometry.
  out.unsafe_contact_nominal =
      out.nominal_contact_sample_found &&
      (out.h_nominal_energy < 0.0);

  out.unsafe_contact_hybrid =
      out.worst_case_candidate_found &&
      (out.h_failsafe_energy < 0.0);

  out.predicted_trigger =
      out.unsafe_contact_nominal || out.unsafe_contact_hybrid;

  return out;
}

controller_interface::return_type ReachableCartesianImpedanceController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  using Clock = std::chrono::steady_clock;
  const auto tic_total = Clock::now();

  const double dt = std::max(period.seconds(), kMinDt);
  const double wall_time = (this->get_node()->now() - start_time_).seconds();

  double nominal_time = wall_time - paused_nominal_time_sec_;
  if (mode_ == SafetyMode::kFailsafe && failsafe_enter_wall_time_sec_ >= 0.0) {
    nominal_time = failsafe_enter_wall_time_sec_ - paused_nominal_time_sec_;
  }
  nominal_time = std::max(0.0, nominal_time);

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

  Vector3d desired_position_cur = desired_position_;
  Quaterniond desired_orientation_cur = desired_orientation_;
  Vector3d desired_linear_velocity_cur = Vector3d::Zero();
  Vector3d desired_linear_acceleration_cur = Vector3d::Zero();

  buildReference(nominal_time,
                 desired_position_cur,
                 desired_orientation_cur,
                 desired_linear_velocity_cur,
                 desired_linear_acceleration_cur);

  updateRuntimeGains(dt);

  Vector6d error = Vector6d::Zero();
  error.head<3>() = current_position - desired_position_cur;
  error.tail<3>() = computeOrientationError(current_orientation, desired_orientation_cur);

  MonitorResult monitor;
  if (enable_safety_monitor_) {
    const Matrix37d Jv = J_geo.topRows<3>();
    monitor = runSafetyMonitor(dt,
                               wall_time,
                               current_position,
                               desired_position_cur,
                               ee_twist,
                               desired_linear_velocity_cur,
                               desired_linear_acceleration_cur,
                               inertia,
                               Jv,
                               human_plane_normal_,
                               human_plane_point_);

    if (auto_enter_failsafe_) {
      if (mode_ == SafetyMode::kNominal) {
        if (monitor.predicted_trigger) {
          enterFailsafe(wall_time, desired_position_cur, desired_orientation_cur);

          buildReference(nominal_time,
                         desired_position_cur,
                         desired_orientation_cur,
                         desired_linear_velocity_cur,
                         desired_linear_acceleration_cur);

          error.head<3>() = current_position - desired_position_cur;
          error.tail<3>() =
              computeOrientationError(current_orientation, desired_orientation_cur);
        }
      } else {
        const bool can_return_to_nominal =
            (monitor.h_failsafe_energy > return_to_nominal_energy_margin_) &&
            (monitor.h_nominal_energy > return_to_nominal_energy_margin_) &&
            (std::abs(monitor.v_n_now) < return_to_nominal_speed_threshold_) &&
            (monitor.V_fs_dot_est <= return_to_nominal_vdot_threshold_);

        if (can_return_to_nominal) {
          leaveFailsafe(wall_time);

          nominal_time = std::max(0.0, wall_time - paused_nominal_time_sec_);

          buildReference(nominal_time,
                         desired_position_cur,
                         desired_orientation_cur,
                         desired_linear_velocity_cur,
                         desired_linear_acceleration_cur);

          error.head<3>() = current_position - desired_position_cur;
          error.tail<3>() =
              computeOrientationError(current_orientation, desired_orientation_cur);
        }
      }
    }
  }

  {
    const Vector3d n = human_plane_normal_.normalized();
    const double k_n_fs =
        std::max((n.transpose() * K_f_target_.topLeftCorner<3, 3>() * n)(0, 0), 0.0);

    const double e_n_now_fs = n.dot(current_position - desired_position_cur);
    const double v_n_now_fs = n.dot(ee_twist.head<3>());

    const double V_fs_now =
        0.5 * monitor.m_eff_n * v_n_now_fs * v_n_now_fs +
        0.5 * k_n_fs * e_n_now_fs * e_n_now_fs;

    monitor.e_n_now_fs = e_n_now_fs;
    monitor.v_n_now_fs = v_n_now_fs;
    monitor.V_fs_now = V_fs_now;

    if (mode_ == SafetyMode::kFailsafe) {
      if (prev_V_fs_valid_) {
        monitor.V_fs_dot_est = (V_fs_now - prev_V_fs_) / std::max(dt, kMinDt);
      } else {
        monitor.V_fs_dot_est = 0.0;
      }
      prev_V_fs_ = V_fs_now;
      prev_V_fs_valid_ = true;
    } else {
      monitor.V_fs_dot_est = 0.0;
      prev_V_fs_valid_ = false;
    }

    last_e_n_fs_ = e_n_now_fs;
    last_v_n_fs_ = v_n_now_fs;
  }

  updateRuntimeGains(dt);

  error.head<3>() = current_position - desired_position_cur;
  error.tail<3>() = computeOrientationError(current_orientation, desired_orientation_cur);

  const Vector7d tau_task =
      J_geo.transpose() * (-K_runtime_ * error - D_runtime_ * ee_twist);

  const Eigen::MatrixXd Jt_pinv = dampedPseudoInverse(J_geo.transpose());
  const Vector7d tau_nullspace =
      (Matrix7d::Identity() - J_geo.transpose() * Jt_pinv) *
      (n_stiffness_ * (desired_qn_ - q) -
       (2.0 * std::sqrt(std::max(n_stiffness_, 0.0))) * dq);

  const Vector7d tau_nullspace_eff =
      (mode_ == SafetyMode::kFailsafe && disable_nullspace_in_failsafe_)
          ? Vector7d::Zero()
          : tau_nullspace;

  const Vector7d tau_cmd = tau_task + coriolis + tau_nullspace_eff;

  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }

  if (enable_error_logging_ && error_log_file_.is_open()) {
    error_log_file_ << std::fixed << std::setprecision(9)
                    << wall_time << ","
                    << nominal_time << ","
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
                    << tau_task.norm() << ","
                    << tau_nullspace_eff.norm() << ","
                    << 0.0 << ","
                    << tau_cmd.norm() << ","
                    << K_runtime_(0, 0) << "," << K_runtime_(1, 1) << "," << K_runtime_(2, 2)
                    << ","
                    << D_runtime_(0, 0) << "," << D_runtime_(1, 1) << "," << D_runtime_(2, 2)
                    << ","
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
                    << monitor.worst_case_e_n << ","
                    << monitor.worst_case_v_n << ","
                    << monitor.worst_case_E_fs_1d << ","
                    << monitor.worst_case_tau_safe << ","
                    << monitor.worst_case_delta_n_fs << ","
                    << monitor.worst_case_hybrid_forward_reach << ","
                    << monitor.worst_case_plane_margin_after_hybrid << ","
                    << monitor.h_geom << ","
                    << monitor.h_nominal_energy << ","
                    << monitor.h_failsafe_energy << ","
                    << monitor.e_n_now_fs << ","
                    << monitor.v_n_now_fs << ","
                    << monitor.V_fs_now << ","
                    << monitor.V_fs_dot_est << ","
                    << static_cast<int>(monitor.contact_possible_nominal) << ","
                    << static_cast<int>(monitor.contact_possible_hybrid) << ","
                    << static_cast<int>(monitor.unsafe_contact_nominal) << ","
                    << static_cast<int>(monitor.unsafe_contact_hybrid) << ","
                    << static_cast<int>(monitor.predicted_trigger)
                    << "\n";

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
                "[reachable_impedance] mode=%d avg_exec=%.6f ms min_exec=%.6f ms max_exec=%.6f ms",
                static_cast<int>(mode_),
                exec_sum_ms_ / n,
                exec_min_ms_,
                exec_max_ms_);
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
        "/home/developer/multipanda_ws/src/data_log/cartesian_impedance_failsafe_validation.csv");

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
    auto_declare<bool>("auto_enter_failsafe", false);

    auto_declare<double>("safe_collision_energy_joule", 0.10);
    auto_declare<double>("ee_collision_radius", 0.04);
    auto_declare<double>("monitor_nominal_horizon_sec", 0.02);
    auto_declare<int>("monitor_nominal_steps", 10);

    auto_declare<std::vector<double>>(
        "human_plane_normal", std::vector<double>{0.0, 0.0, 1.0});
    auto_declare<std::vector<double>>(
        "human_plane_point", std::vector<double>{0.0, 0.0, 0.2});

    auto_declare<double>("return_to_nominal_energy_margin", 0.02);
    auto_declare<double>("return_to_nominal_speed_threshold", 0.02);
    auto_declare<double>("return_to_nominal_vdot_threshold", 0.0);

    // kept only for log compatibility / backward compatibility
    auto_declare<double>("return_to_nominal_geom_margin", 0.005);

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
    auto_enter_failsafe_ = get_node()->get_parameter("auto_enter_failsafe").as_bool();

    safe_collision_energy_joule_ =
        get_node()->get_parameter("safe_collision_energy_joule").as_double();
    ee_collision_radius_ = get_node()->get_parameter("ee_collision_radius").as_double();
    monitor_nominal_horizon_sec_ =
        get_node()->get_parameter("monitor_nominal_horizon_sec").as_double();
    monitor_nominal_steps_ =
        static_cast<int>(get_node()->get_parameter("monitor_nominal_steps").as_int());

    return_to_nominal_energy_margin_ =
        get_node()->get_parameter("return_to_nominal_energy_margin").as_double();
    return_to_nominal_speed_threshold_ =
        get_node()->get_parameter("return_to_nominal_speed_threshold").as_double();
    return_to_nominal_vdot_threshold_ =
        get_node()->get_parameter("return_to_nominal_vdot_threshold").as_double();

    // optional backward compatibility parameter
    return_to_nominal_geom_margin_ =
        get_node()->get_parameter("return_to_nominal_geom_margin").as_double();

    const auto normal_vec =
        get_node()->get_parameter("human_plane_normal").as_double_array();
    const auto point_vec =
        get_node()->get_parameter("human_plane_point").as_double_array();

    if (normal_vec.size() != 3 || point_vec.size() != 3) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "human_plane_normal and human_plane_point must have length 3.");
      return CallbackReturn::ERROR;
    }

    human_plane_normal_ = Vector3d(normal_vec[0], normal_vec[1], normal_vec[2]);
    human_plane_point_ = Vector3d(point_vec[0], point_vec[1], point_vec[2]);

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

  prev_V_fs_ = 0.0;
  prev_V_fs_valid_ = false;
  last_e_n_fs_ = 0.0;
  last_v_n_fs_ = 0.0;

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
        << "worst_case_nominal_forward_progress,worst_case_e_n,worst_case_v_n,"
        << "worst_case_E_fs_1d,worst_case_tau_safe,worst_case_delta_n_fs,"
        << "worst_case_hybrid_forward_reach,worst_case_plane_margin_after_hybrid,"
        << "h_geom,h_nominal_energy,h_failsafe_energy,"
        << "e_n_now_fs,v_n_now_fs,V_fs_now,V_fs_dot_est,"
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