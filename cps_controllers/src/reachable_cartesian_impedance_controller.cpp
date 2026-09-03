// Copyright (c) 2026
// Reachable Cartesian Impedance Controller
//
// Monitored execution with a whole-robot Lachner-style energy budget in nominal contact.
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
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <franka/model.h>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <geometry_msgs/msg/pose_array.hpp>

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>
#include <cps_trajectory_generators/reachable_cartesian_trajectory.hpp>

#include "reachable_cartesian_impedance/math.hpp"
#include "reachable_cartesian_impedance/timing.hpp"

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
using cps_controllers::detail::nanosecondsToMilliseconds;
using cps_controllers::detail::steadyNowNanoseconds;
using cps_controllers::detail::SteadyClock;
using CartesianTrajectorySample = cps_trajectory_generators::CartesianTrajectorySample;
using PathConsistentTimedPathConfig =
    cps_trajectory_generators::PathConsistentTimedPathConfig;
using cps_trajectory_generators::loadTrajectoryGeneratorSettings;
using cps_trajectory_generators::makePathConsistentTimedPathBrake;
using cps_trajectory_generators::makePathConsistentTimedPathIntendedPrefix;
using cps_trajectory_generators::makeRetimedPathState;

std::vector<double> defaultNullspaceHomePoseParameter() {
  return {
      0.0,
      -0.7853981633974483,
      0.0,
      -2.356194490192345,
      0.0,
      1.5707963267948966,
      0.7853981633974483};
}

cps_controllers::SafetyMode nominalSafetyModeForMonitor(
    const cps_safety_monitor::MonitorResult& monitor) {
  return monitor.monitored_contact_possible
             ? cps_controllers::SafetyMode::kNominalContactPossible
             : cps_controllers::SafetyMode::kNominal;
}

cps_controllers::SafetyMode lastVerifiedSafetyModeForMonitor(
    const cps_safety_monitor::MonitorResult& monitor) {
  return monitor.monitored_contact_possible
             ? cps_controllers::SafetyMode::kLastVerifiedContactPossible
             : cps_controllers::SafetyMode::kLastVerifiedMonitored;
}

int executionModeForLog(bool fallback_execution) {
  return fallback_execution ? 1 : 0;
}

int executionModeForLog(cps_controllers::ExecutionStage stage) {
  return executionModeForLog(
      stage != cps_controllers::ExecutionStage::kCurrentVerified);
}

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

inline Vector3d clampVectorNorm(const Vector3d& v, double max_norm) {
  const double limit = std::max(max_norm, 0.0);
  const double norm = v.norm();
  if (norm <= limit || norm < 1.0e-12) {
    return v;
  }
  return v * (limit / norm);
}

inline double clamp01(double value) {
  return std::clamp(value, 0.0, 1.0);
}

inline cps_safety_monitor::ImpedanceSample interpolateImpedanceSample(
    const cps_safety_monitor::ImpedanceSample& a,
    const cps_safety_monitor::ImpedanceSample& b,
    double t) {
  const double span = std::max(b.t - a.t, kMinDt);
  const double u = clamp01((t - a.t) / span);
  const double u2 = u * u;
  const double u3 = u2 * u;

  const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
  const double h10 = u3 - 2.0 * u2 + u;
  const double h01 = -2.0 * u3 + 3.0 * u2;
  const double h11 = u3 - u2;

  const double dh00 = (6.0 * u2 - 6.0 * u) / span;
  const double dh10 = 3.0 * u2 - 4.0 * u + 1.0;
  const double dh01 = (-6.0 * u2 + 6.0 * u) / span;
  const double dh11 = 3.0 * u2 - 2.0 * u;

  cps_safety_monitor::ImpedanceSample out;
  out.t = t;
  out.nominal_path_time_valid =
      a.nominal_path_time_valid && b.nominal_path_time_valid;
  if (out.nominal_path_time_valid) {
    out.nominal_path_time =
        (1.0 - u) * a.nominal_path_time + u * b.nominal_path_time;
  }
  out.nominal_path_kinematics_valid =
      a.nominal_path_kinematics_valid && b.nominal_path_kinematics_valid;
  if (out.nominal_path_kinematics_valid) {
    out.nominal_path_rate =
        (1.0 - u) * a.nominal_path_rate + u * b.nominal_path_rate;
    out.nominal_path_acceleration =
        (1.0 - u) * a.nominal_path_acceleration +
        u * b.nominal_path_acceleration;
  }
  out.p = h00 * a.p + h10 * span * a.dp +
          h01 * b.p + h11 * span * b.dp;
  out.dp = dh00 * a.p + dh10 * a.dp +
           dh01 * b.p + dh11 * b.dp;
  out.ddp = (1.0 - u) * a.ddp + u * b.ddp;
  out.q = a.q.slerp(u, b.q);
  out.q.normalize();
  out.w = (1.0 - u) * a.w + u * b.w;
  out.dw = (1.0 - u) * a.dw + u * b.dw;
  out.K = (1.0 - u) * a.K + u * b.K;
  out.D = (1.0 - u) * a.D + u * b.D;
  out.failsafe = a.failsafe || b.failsafe;
  return out;
}

bool symmetricPositiveSemidefiniteSquareRoot(
    const Matrix6d& matrix,
    Matrix6d* matrix_sqrt) {
  if (matrix_sqrt == nullptr || !matrix.array().isFinite().all()) {
    return false;
  }

  const Matrix6d symmetric = 0.5 * (matrix + matrix.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6d> eig(symmetric);
  if (eig.info() != Eigen::Success) {
    return false;
  }

  *matrix_sqrt =
      eig.eigenvectors() *
      eig.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() *
      eig.eigenvectors().transpose();
  *matrix_sqrt = 0.5 * (*matrix_sqrt + matrix_sqrt->transpose());
  return matrix_sqrt->array().isFinite().all();
}

}  // namespace

namespace cps_controllers {

class ReachableCartesianImpedanceController::PinocchioJointDynamicsProvider
    final : public cps_safety_monitor::JointDynamicsProvider {
 public:
  PinocchioJointDynamicsProvider(const std::string& urdf_model_path,
                                 const Vector3d& tcp_offset,
                                 const Vector7d& joint_armature)
      : joint_armature_(joint_armature) {
    if (!joint_armature_.allFinite() ||
        (joint_armature_.array() < 0.0).any()) {
      throw std::runtime_error(
          "Prediction joint armature must contain finite nonnegative values");
    }
    pinocchio::urdf::buildModel(urdf_model_path, model_);
    if (model_.nq != 7 || model_.nv != 7) {
      throw std::runtime_error(
          "Reachable monitor URDF must contain exactly seven actuated joints");
    }
    data_ = std::make_unique<pinocchio::Data>(model_);
    frame_id_ = model_.getFrameId("panda_link8");
    if (frame_id_ >= static_cast<pinocchio::FrameIndex>(model_.nframes)) {
      throw std::runtime_error(
          "Reachable monitor URDF does not contain frame panda_link8");
    }
    tcp_offset_ = tcp_offset;

    for (int i = 0; i < 7; ++i) {
      limits_.position_lower(i) = panda_limits::kPositionLower[i];
      limits_.position_upper(i) = panda_limits::kPositionUpper[i];
      limits_.velocity(i) = panda_limits::kVelocity[i];
      limits_.acceleration(i) = panda_limits::kAcceleration[i];
      limits_.torque(i) = panda_limits::kTorque[i];
    }
  }

  bool evaluate(
      const Vector7d& q,
      const Vector7d& dq,
      cps_safety_monitor::JointDynamicsSample* sample) const override {
    if (sample == nullptr || !q.allFinite() || !dq.allFinite()) {
      return false;
    }

    try {
      const Vector7d ddq_zero = Vector7d::Zero();
      pinocchio::forwardKinematics(model_, *data_, q, dq, ddq_zero);
      pinocchio::computeJointJacobians(model_, *data_, q);
      pinocchio::updateFramePlacements(model_, *data_);

      Matrix67d pin_jacobian = Matrix67d::Zero();
      pinocchio::getFrameJacobian(
          model_, *data_, frame_id_,
          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
          pin_jacobian);

      // Pinocchio in this ROS distribution and the controller both use the
      // [linear; angular] 6D ordering.
      const Matrix67d frame_jacobian = pin_jacobian;

      const auto& placement = data_->oMf[frame_id_];
      const Vector3d offset_world = placement.rotation() * tcp_offset_;
      Matrix67d control_jacobian = frame_jacobian;
      control_jacobian.topRows<3>() =
          frame_jacobian.topRows<3>() -
          skewSymmetric(offset_world) * frame_jacobian.bottomRows<3>();

      const pinocchio::Motion frame_velocity =
          pinocchio::getFrameVelocity(
              model_, *data_, frame_id_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      const pinocchio::Motion frame_acceleration =
          pinocchio::getFrameClassicalAcceleration(
              model_, *data_, frame_id_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      const Vector3d omega = frame_velocity.angular();
      const Vector3d alpha = frame_acceleration.angular();
      Vector6d control_jdot_dq = Vector6d::Zero();
      control_jdot_dq.head<3>() =
          frame_acceleration.linear() + alpha.cross(offset_world) +
          omega.cross(omega.cross(offset_world));
      control_jdot_dq.tail<3>() = alpha;

      pinocchio::crba(model_, *data_, q);
      Matrix7d inertia = data_->M;
      inertia.triangularView<Eigen::StrictlyLower>() =
          inertia.transpose().triangularView<Eigen::StrictlyLower>();
      // MuJoCo armature is a constant inertia expressed directly in each
      // hinge coordinate. URDF link inertias do not contain this term.
      inertia.diagonal() += joint_armature_;
      pinocchio::computeCoriolisMatrix(model_, *data_, q, dq);

      sample->control_position =
          placement.translation() + offset_world;
      sample->control_orientation =
          Quaterniond(placement.rotation()).normalized();
      sample->control_jacobian = control_jacobian;
      sample->control_jdot_dq = control_jdot_dq;
      sample->inertia = 0.5 * (inertia + inertia.transpose());
      sample->coriolis = data_->C * dq;
      sample->valid = sample->control_position.allFinite() &&
                      sample->control_jacobian.allFinite() &&
                      sample->control_jdot_dq.allFinite() &&
                      sample->inertia.allFinite() &&
                      sample->coriolis.allFinite();
      return sample->valid;
    } catch (const std::exception&) {
      sample->valid = false;
      return false;
    }
  }

  cps_safety_monitor::JointDynamicsLimits limits() const override {
    return limits_;
  }

 private:
  pinocchio::Model model_;
  mutable std::unique_ptr<pinocchio::Data> data_;
  pinocchio::FrameIndex frame_id_{0};
  Vector3d tcp_offset_{Vector3d::Zero()};
  Vector7d joint_armature_{Vector7d::Zero()};
  cps_safety_monitor::JointDynamicsLimits limits_;
};

class ReachableCartesianImpedanceController::FrankaInterfaceJointDynamicsProvider
    final : public cps_safety_monitor::JointDynamicsProvider {
 public:
  FrankaInterfaceJointDynamicsProvider(
      franka_semantic_components::FrankaRobotModel* robot_model,
      const franka::RobotState& robot_state,
      const Vector3d& tcp_offset)
      : robot_model_(robot_model),
        F_T_EE_(robot_state.F_T_EE),
        EE_T_K_(robot_state.EE_T_K),
        I_total_(robot_state.I_total),
        m_total_(robot_state.m_total),
        F_x_Ctotal_(robot_state.F_x_Ctotal),
        tcp_offset_(tcp_offset) {
    if (robot_model_ == nullptr ||
        !robot_model_->supportsStateDependentEvaluation()) {
      throw std::runtime_error(
          "Franka model backend cannot evaluate arbitrary predicted q/dq");
    }
    for (int i = 0; i < 7; ++i) {
      limits_.position_lower(i) = panda_limits::kPositionLower[i];
      limits_.position_upper(i) = panda_limits::kPositionUpper[i];
      limits_.velocity(i) = panda_limits::kVelocity[i];
      limits_.acceleration(i) = panda_limits::kAcceleration[i];
      limits_.torque(i) = panda_limits::kTorque[i];
    }
  }

  bool evaluate(
      const Vector7d& q,
      const Vector7d& dq,
      cps_safety_monitor::JointDynamicsSample* sample) const override {
    if (sample == nullptr || robot_model_ == nullptr ||
        !q.allFinite() || !dq.allFinite()) {
      return false;
    }

    try {
      std::array<double, 7> q_array{};
      std::array<double, 7> dq_array{};
      std::copy_n(q.data(), 7, q_array.begin());
      std::copy_n(dq.data(), 7, dq_array.begin());

      const auto pose_array = robot_model_->getPoseMatrix(
          franka::Frame::kEndEffector, q_array, F_T_EE_, EE_T_K_);
      const Eigen::Map<const Matrix4d> pose(pose_array.data());
      const Matrix3d rotation = pose.block<3, 3>(0, 0);
      const Vector3d offset_world = rotation * tcp_offset_;

      const auto jacobian_array = robot_model_->getZeroJacobian(
          franka::Frame::kEndEffector, q_array, F_T_EE_, EE_T_K_);
      Matrix67d control_jacobian =
          Eigen::Map<const Matrix67d>(jacobian_array.data());
      control_jacobian.topRows<3>() -=
          skewSymmetric(offset_world) * control_jacobian.bottomRows<3>();

      // libfranka does not expose Jdot*dq. Estimate it by differentiating the
      // same libfranka Jacobian model along dq, so no Pinocchio quantity is
      // mixed into the real-robot rollout.
      constexpr double kJacobianDifferenceTime = 1.0e-4;
      std::array<double, 7> q_plus = q_array;
      std::array<double, 7> q_minus = q_array;
      for (std::size_t i = 0; i < q_plus.size(); ++i) {
        q_plus[i] += kJacobianDifferenceTime * dq_array[i];
        q_minus[i] -= kJacobianDifferenceTime * dq_array[i];
      }
      const auto jacobian_plus_array = robot_model_->getZeroJacobian(
          franka::Frame::kEndEffector, q_plus, F_T_EE_, EE_T_K_);
      const auto jacobian_minus_array = robot_model_->getZeroJacobian(
          franka::Frame::kEndEffector, q_minus, F_T_EE_, EE_T_K_);
      Matrix67d jacobian_plus =
          Eigen::Map<const Matrix67d>(jacobian_plus_array.data());
      Matrix67d jacobian_minus =
          Eigen::Map<const Matrix67d>(jacobian_minus_array.data());

      const auto pose_plus_array = robot_model_->getPoseMatrix(
          franka::Frame::kEndEffector, q_plus, F_T_EE_, EE_T_K_);
      const auto pose_minus_array = robot_model_->getPoseMatrix(
          franka::Frame::kEndEffector, q_minus, F_T_EE_, EE_T_K_);
      const Eigen::Map<const Matrix4d> pose_plus(pose_plus_array.data());
      const Eigen::Map<const Matrix4d> pose_minus(pose_minus_array.data());
      const Vector3d offset_plus =
          pose_plus.block<3, 3>(0, 0) * tcp_offset_;
      const Vector3d offset_minus =
          pose_minus.block<3, 3>(0, 0) * tcp_offset_;
      jacobian_plus.topRows<3>() -=
          skewSymmetric(offset_plus) * jacobian_plus.bottomRows<3>();
      jacobian_minus.topRows<3>() -=
          skewSymmetric(offset_minus) * jacobian_minus.bottomRows<3>();

      const auto inertia_array = robot_model_->getMassMatrix(
          q_array, I_total_, m_total_, F_x_Ctotal_);
      const auto coriolis_array = robot_model_->getCoriolisForceVector(
          q_array, dq_array, I_total_, m_total_, F_x_Ctotal_);

      sample->control_position = pose.block<3, 1>(0, 3) + offset_world;
      sample->control_orientation = Quaterniond(rotation).normalized();
      sample->control_jacobian = control_jacobian;
      sample->control_jdot_dq =
          ((jacobian_plus - jacobian_minus) /
           (2.0 * kJacobianDifferenceTime)) * dq;
      sample->inertia = Eigen::Map<const Matrix7d>(inertia_array.data());
      sample->inertia =
          0.5 * (sample->inertia + sample->inertia.transpose());
      sample->coriolis = Eigen::Map<const Vector7d>(coriolis_array.data());
      sample->valid = sample->control_position.allFinite() &&
                      sample->control_orientation.coeffs().allFinite() &&
                      sample->control_jacobian.allFinite() &&
                      sample->control_jdot_dq.allFinite() &&
                      sample->inertia.allFinite() &&
                      sample->coriolis.allFinite();
      return sample->valid;
    } catch (const std::exception&) {
      sample->valid = false;
      return false;
    }
  }

  cps_safety_monitor::JointDynamicsLimits limits() const override {
    return limits_;
  }

 private:
  franka_semantic_components::FrankaRobotModel* robot_model_{nullptr};
  std::array<double, 16> F_T_EE_{};
  std::array<double, 16> EE_T_K_{};
  std::array<double, 9> I_total_{};
  double m_total_{0.0};
  std::array<double, 3> F_x_Ctotal_{};
  Vector3d tcp_offset_{Vector3d::Zero()};
  cps_safety_monitor::JointDynamicsLimits limits_;
};

ReachableCartesianImpedanceController::~ReachableCartesianImpedanceController() {
  stopSafetyMonitorWorker();
}

// ============================================================================
// Helper: runtime gain update
// ============================================================================
void ReachableCartesianImpedanceController::updateRuntimeGains(const Matrix6d& K_target,
                                                               const Matrix6d& D_target) {
  K_runtime_ = K_target;
  D_runtime_ = D_target;
}

void ReachableCartesianImpedanceController::handleHumanWorkspaceState(
    const cps_human_workspace::msg::HumanWorkspace::SharedPtr msg) {
  if (!msg) {
    return;
  }

  cps_human_workspace::HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d(
      msg->sphere_center.x,
      msg->sphere_center.y,
      msg->sphere_center.z);
  parameters.center_velocity = Vector3d(
      msg->center_velocity.x,
      msg->center_velocity.y,
      msg->center_velocity.z);
  parameters.center_sinusoid_amplitude.setZero();
  parameters.center_sinusoid_frequency_hz = 0.0;
  parameters.center_sinusoid_phase_rad = 0.0;
  parameters.center_motion_time_offset_sec = 0.0;
  parameters.motion_radius = msg->motion_radius;
  parameters.hand_max_velocity = msg->hand_max_velocity;
  parameters.hand_max_acceleration = msg->hand_max_acceleration;
  parameters.measurement_error_position =
      msg->measurement_error_position;
  parameters.measurement_error_velocity =
      msg->measurement_error_velocity;
  parameters.measurement_delay = msg->measurement_delay;

  const bool reachability_parameters_valid =
      std::isfinite(parameters.hand_max_velocity) &&
      parameters.hand_max_velocity >= 0.0 &&
      std::isfinite(parameters.hand_max_acceleration) &&
      parameters.hand_max_acceleration > 0.0 &&
      std::isfinite(parameters.measurement_error_position) &&
      parameters.measurement_error_position >= 0.0 &&
      std::isfinite(parameters.measurement_error_velocity) &&
      parameters.measurement_error_velocity >= 0.0 &&
      std::isfinite(parameters.measurement_delay) &&
      parameters.measurement_delay >= 0.0;
  if (parameters.motion_radius < 0.0 ||
      !std::isfinite(parameters.motion_radius) ||
      !parameters.sphere_center.allFinite() ||
      !parameters.center_velocity.allFinite() ||
      !reachability_parameters_valid) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "Ignoring invalid human workspace state on '%s'.",
        human_workspace_topic_.c_str());
    return;
  }

  human_workspace_param_buffer_.writeFromNonRT(parameters);
  human_workspace_live_received_.store(true, std::memory_order_relaxed);
  const double receipt_time_sec = get_node()->now().seconds();
  const double message_time_sec =
      static_cast<double>(msg->header.stamp.sec) +
      1.0e-9 * static_cast<double>(msg->header.stamp.nanosec);
  const bool message_time_valid =
      std::isfinite(message_time_sec) && message_time_sec > 0.0 &&
      message_time_sec <= receipt_time_sec + 0.1;
  latest_human_workspace_msg_time_sec_.store(
      message_time_valid ? message_time_sec : receipt_time_sec,
      std::memory_order_relaxed);
}

bool ReachableCartesianImpedanceController::refreshHumanWorkspaceForMonitor(
    double wall_time) {
  if (!enable_safety_monitor_) {
    human_workspace_active_ = false;
    return false;
  }

  if (human_workspace_live_received_.load(std::memory_order_relaxed)) {
    const double latest_msg_time =
        latest_human_workspace_msg_time_sec_.load(std::memory_order_relaxed);
    const double age_sec = get_node()->now().seconds() - latest_msg_time;
    human_workspace_active_ =
        latest_msg_time >= 0.0 &&
        age_sec <= std::max(0.0, human_workspace_timeout_sec_);
    if (const auto* parameters = human_workspace_param_buffer_.readFromRT()) {
      auto live_parameters = *parameters;
      live_parameters.center_motion_time_offset_sec =
          wall_time - std::max(0.0, age_sec);
      human_workspace_.setParameters(live_parameters);
    }
    if (!human_workspace_active_) {
      RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(),
          *get_node()->get_clock(),
          1000,
          "Human workspace state on '%s' is stale or unavailable. "
          "Safety monitor is fail-closed and will reject new plans until "
          "fresh workspace data arrives.",
          human_workspace_topic_.c_str());
    }
    return human_workspace_active_;
  }

  human_workspace_active_ = human_workspace_configured_static_;
  if (human_workspace_active_) {
    auto observed_parameters = configured_human_workspace_source_.parameters();
    observed_parameters.sphere_center =
        configured_human_workspace_source_.centerAtTime(wall_time);
    observed_parameters.center_velocity =
        configured_human_workspace_source_.centerVelocityAtTime(wall_time);
    observed_parameters.center_sinusoid_amplitude.setZero();
    observed_parameters.center_sinusoid_frequency_hz = 0.0;
    observed_parameters.center_sinusoid_phase_rad = 0.0;
    observed_parameters.center_motion_time_offset_sec = wall_time;
    human_workspace_.setParameters(observed_parameters);
  }
  if (!human_workspace_active_) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "No human workspace state received on '%s'. "
        "Safety monitor is fail-closed and will reject new plans until the "
        "workspace provider is available.",
        human_workspace_topic_.c_str());
  }
  return human_workspace_active_;
}

bool ReachableCartesianImpedanceController::shouldRejectCandidateWithMonitor(
    const MonitorResult& monitor,
    bool human_workspace_available) const {
  if (!enable_safety_monitor_) {
    return false;
  }

  // Missing or stale workspace data means that human occupancy is unknown.
  // Do not interpret a provider failure as an empty workspace.
  if (!human_workspace_available) {
    return true;
  }

  // A predicted collision possibility alone is allowed. Reject the candidate
  // only when the predicted collision energy exceeds the budget. Joint-limit
  // prediction remains an independent rejection condition, including invalid
  // joint rollouts that cannot produce a predicted_trigger value.
  return monitor.predicted_trigger || monitor.joint_limit_unsafe;
}

bool ReachableCartesianImpedanceController::shouldRejectCandidateWithMonitor(
    const MonitorResult& monitor) const {
  return shouldRejectCandidateWithMonitor(monitor, human_workspace_active_);
}

bool ReachableCartesianImpedanceController::shouldApplyEnergyBudget(
    const MonitorResult& monitor) const {
  // The controlled energy is measured throughout execution, but the gain
  // adaptation is activated only after the current robot occupancy overlaps
  // the human workspace. This preserves the nominal gains outside the
  // collision area while using Eq. (14) for the command actually sent inside.
  return enable_safety_monitor_ &&
         human_workspace_active_ &&
         monitor.contact_relevant_for_energy;
}

bool ReachableCartesianImpedanceController::computeTaskInertia(
    const Matrix7d& inertia,
    const Matrix67d& J_geo,
    Matrix6d* lambda) const {
  if (lambda == nullptr) {
    return false;
  }

  const Eigen::LDLT<Matrix7d> inertia_ldlt(inertia);
  if (inertia_ldlt.info() != Eigen::Success) {
    return false;
  }

  const Matrix7d M_inv = inertia_ldlt.solve(Matrix7d::Identity());
  Matrix6d lambda_inv = J_geo * M_inv * J_geo.transpose();
  lambda_inv = 0.5 * (lambda_inv + lambda_inv.transpose());
  lambda_inv.diagonal().array() += kDynamicLambdaRegularization;

  const Eigen::LDLT<Matrix6d> lambda_ldlt(lambda_inv);
  if (lambda_ldlt.info() != Eigen::Success) {
    return false;
  }

  *lambda = lambda_ldlt.solve(Matrix6d::Identity());
  *lambda = 0.5 * (*lambda + lambda->transpose());
  return lambda->array().isFinite().all();
}

ImpedanceSample ReachableCartesianImpedanceController::applyEnergyBudget(
    const ImpedanceSample& command,
    double kinetic_energy,
    double potential_energy,
    double nullspace_potential_energy,
    bool energy_valid,
    bool active,
    EnergyBudgetInfo* info) const {
  EnergyBudgetInfo local_info;
  local_info.active = active;
  local_info.lambda_valid = energy_valid;
  local_info.kinetic_energy = std::max(0.0, kinetic_energy);
  local_info.potential_energy_before_scaling =
      std::max(0.0, potential_energy);
  local_info.nullspace_potential_energy_before_scaling =
      std::max(0.0, nullspace_potential_energy);
  local_info.total_energy_before_scaling =
      local_info.kinetic_energy +
      local_info.potential_energy_before_scaling +
      local_info.nullspace_potential_energy_before_scaling;
  // Without active scaling, the candidate and applied energies are equal.
  local_info.potential_energy =
      local_info.potential_energy_before_scaling;
  local_info.nullspace_potential_energy =
      local_info.nullspace_potential_energy_before_scaling;
  local_info.nullspace_stiffness = n_stiffness_;
  local_info.total_energy = local_info.total_energy_before_scaling;

  ImpedanceSample scaled_command = command;
  if (!active || !energy_valid) {
    if (info != nullptr) {
      *info = local_info;
    }
    return scaled_command;
  }

  // Lachner et al. Eq. (14): one common kappa scales U_x and U_q. This is
  // essential for a globally enabled nullspace spring; treating U_n as fixed
  // would leave part of the controlled potential outside the budget action.
  local_info.scale = cps_safety_monitor::energyBudgetStiffnessScale(
      local_info.kinetic_energy,
      local_info.potential_energy_before_scaling,
      local_info.nullspace_potential_energy_before_scaling,
      energy_budget_joule_);
  // This is the elastic energy stored by the stiffness command that will
  // actually be used in this control cycle. K is scaled linearly, therefore
  // its quadratic potential is scaled by the same factor.
  local_info.potential_energy =
      local_info.scale * local_info.potential_energy_before_scaling;
  local_info.nullspace_potential_energy =
      local_info.scale *
      local_info.nullspace_potential_energy_before_scaling;
  local_info.nullspace_stiffness = local_info.scale * n_stiffness_;
  local_info.total_energy =
      local_info.kinetic_energy + local_info.potential_energy +
      local_info.nullspace_potential_energy;

  // Lachner et al. (14) and (18): the same kappa also scales Cartesian
  // stiffness. The nullspace controller consumes local_info.nullspace_stiffness
  // and therefore obtains kappa*K_n and sqrt(kappa)*B_n as well.
  scaled_command.K =
      0.5 * local_info.scale * (command.K + command.K.transpose());
  scaled_command.D = std::sqrt(local_info.scale) * command.D;
  if (info != nullptr) {
    *info = local_info;
  }
  return scaled_command;
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

ImpedanceSample ReachableCartesianImpedanceController::makeEmergencyStopCommand(
    const Vector3d& current_position, const Quaterniond& current_orientation,
    double wall_time) const {
  ImpedanceSample emergency;
  emergency.t = wall_time;
  emergency.p = current_position;
  emergency.dp.setZero(); emergency.ddp.setZero();
  emergency.q = current_orientation; emergency.q.normalize();
  emergency.w.setZero(); emergency.dw.setZero();
  emergency.K = K_base_; emergency.D = D_base_;
  emergency.failsafe = true;
  return emergency;
}

bool ReachableCartesianImpedanceController::anchorLastCommandedSampleToPathStart() {
  std::vector<CartesianTrajectorySample> active_path;
  {
    std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
    active_path = cartesian_via_point_path_;
  }
  if (!last_commanded_sample_valid_ || active_path.empty()) {
    return false;
  }

  const CartesianTrajectorySample& path_start = active_path.front();
  constexpr double kPositionTolerance = 1.0e-8;
  constexpr double kDerivativeTolerance = 1.0e-8;
  constexpr double kOrientationTolerance = 1.0e-8;
  const Quaterniond command_orientation =
      normalizedQuaternionOrIdentity(last_commanded_sample_.q);
  const Quaterniond path_orientation =
      normalizedQuaternionOrIdentity(path_start.q);
  const bool continuous =
      (last_commanded_sample_.p - path_start.p).norm() <=
          kPositionTolerance &&
      (last_commanded_sample_.dp - path_start.dp).norm() <=
          kDerivativeTolerance &&
      (last_commanded_sample_.ddp - path_start.ddp).norm() <=
          kDerivativeTolerance &&
      (last_commanded_sample_.w - path_start.w).norm() <=
          kDerivativeTolerance &&
      (last_commanded_sample_.dw - path_start.dw).norm() <=
          kDerivativeTolerance &&
      1.0 - std::abs(command_orientation.dot(path_orientation)) <=
          kOrientationTolerance;
  if (!continuous) {
    return false;
  }

  last_commanded_sample_.nominal_path_time = path_start.t;
  last_commanded_sample_.nominal_path_time_valid = true;
  last_commanded_sample_.nominal_path_rate = path_time_rate_target_;
  last_commanded_sample_.nominal_path_acceleration = 0.0;
  last_commanded_sample_.nominal_path_kinematics_valid = true;
  return true;
}

ImpedanceSample ReachableCartesianImpedanceController::getNextFailsafeCommandFromCache(
    bool advance_index) {
  if (!last_verified_plan_.valid || last_verified_plan_.failsafe.empty())
    return ImpedanceSample{};

  const ImpedanceSample& failsafe_start =
      last_verified_plan_.intended.empty()
          ? last_verified_plan_.anchor
          : last_verified_plan_.intended.back();
  const double command_dt = std::max(local_replan_dt_, kMinDt);
  const double command_time =
      failsafe_start.t +
      static_cast<double>(last_verified_plan_.failsafe_exec_index + 1) *
          command_dt;
  last_verified_command_stage_ = 2;
  last_verified_command_index_ = last_verified_plan_.failsafe_exec_index;
  verified_command_selected_this_cycle_ = true;

  ImpedanceSample cmd;
  const auto& failsafe = last_verified_plan_.failsafe;
  if (command_time <= failsafe.front().t) {
    cmd = interpolateImpedanceSample(
        failsafe_start,
        failsafe.front(),
        command_time);
  } else if (command_time >= failsafe.back().t) {
    cmd = failsafe.back();
    cmd.t = command_time;
  } else {
    const auto upper = std::lower_bound(
        failsafe.begin(),
        failsafe.end(),
        command_time,
        [](const ImpedanceSample& sample, double value) {
          return sample.t < value;
        });
    const auto lower = upper - 1;
    cmd = interpolateImpedanceSample(*lower, *upper, command_time);
  }
  cmd.failsafe = true;

  if (advance_index) {
    ++last_verified_plan_.failsafe_exec_index;
  }
  return cmd;
}

ImpedanceSample ReachableCartesianImpedanceController::getNextVerifiedTrajectoryCommandFromCache(
    bool advance_index) {
  if (!last_verified_plan_.valid) {
    return ImpedanceSample{};
  }

  if (last_verified_plan_.intended_exec_index <
      last_verified_plan_.intended.size()) {
    const std::size_t idx = last_verified_plan_.intended_exec_index;
    last_verified_command_stage_ = 1;
    last_verified_command_index_ = idx;
    verified_command_selected_this_cycle_ = true;
    ImpedanceSample cmd = last_verified_plan_.intended[idx];
    if (advance_index) {
      ++last_verified_plan_.intended_exec_index;
    }
    return cmd;
  }

  if (!last_verified_plan_.failsafe.empty()) {
    return getNextFailsafeCommandFromCache(advance_index);
  }

  return ImpedanceSample{};
}

bool ReachableCartesianImpedanceController::getVerifiedTrajectoryCommandAtOffset(
    const VerifiedPlan& plan,
    std::size_t offset,
    ImpedanceSample* command) const {
  if (command == nullptr || !plan.valid) {
    return false;
  }

  const std::size_t intended_remaining =
      plan.intended_exec_index < plan.intended.size()
          ? plan.intended.size() - plan.intended_exec_index
          : 0;
  if (offset < intended_remaining) {
    *command = plan.intended[plan.intended_exec_index + offset];
    return true;
  }

  if (plan.failsafe.empty()) {
    return false;
  }

  const std::size_t failsafe_offset = offset - intended_remaining;
  const std::size_t failsafe_index =
      plan.failsafe_exec_index + failsafe_offset;
  const ImpedanceSample& failsafe_start =
      plan.intended.empty() ? plan.anchor : plan.intended.back();
  const double command_dt = std::max(local_replan_dt_, kMinDt);
  const double command_time =
      failsafe_start.t +
      static_cast<double>(failsafe_index + 1) * command_dt;

  if (command_time <= plan.failsafe.front().t) {
    *command = interpolateImpedanceSample(
        failsafe_start, plan.failsafe.front(), command_time);
  } else if (command_time >= plan.failsafe.back().t) {
    *command = plan.failsafe.back();
    command->t = command_time;
  } else {
    const auto upper = std::lower_bound(
        plan.failsafe.begin(),
        plan.failsafe.end(),
        command_time,
        [](const ImpedanceSample& sample, double value) {
          return sample.t < value;
        });
    *command = interpolateImpedanceSample(*(upper - 1), *upper, command_time);
  }
  command->failsafe = true;
  return true;
}

bool ReachableCartesianImpedanceController::isOneStepCommandTransitionContinuous(
    const ImpedanceSample& previous,
    const ImpedanceSample& next,
    bool allow_measured_derivative_reanchor) const {
  const double transition_dt = std::max(local_replan_dt_, kMinDt);
  const double max_acceleration = local_replan_max_acceleration_;
  const double max_jerk = local_replan_max_jerk_;
  const double max_angular_acceleration =
      local_replan_max_angular_acceleration_;
  const double max_angular_jerk = local_replan_max_angular_jerk_;
  constexpr double kPositionTolerance = 1.0e-5;
  constexpr double kVelocityTolerance = 1.0e-5;
  constexpr double kAccelerationTolerance = 1.0e-4;
  constexpr double kPathTimeTolerance = 1.0e-6;

  const bool path_state_continuous =
      previous.nominal_path_time_valid &&
      next.nominal_path_time_valid &&
      next.nominal_path_time >=
          previous.nominal_path_time - kPathTimeTolerance &&
      next.nominal_path_time - previous.nominal_path_time <=
          path_time_rate_max_ * transition_dt + kPathTimeTolerance;
  const bool pose_continuous =
      (next.p - previous.p).norm() <=
          std::max(previous.dp.norm(), next.dp.norm()) * transition_dt +
              0.5 * max_acceleration * transition_dt * transition_dt +
              kPositionTolerance &&
      computeOrientationError(previous.q, next.q).norm() <=
          std::max(previous.w.norm(), next.w.norm()) * transition_dt +
              0.5 * max_angular_acceleration * transition_dt * transition_dt +
              kPositionTolerance;
  const bool derivatives_continuous =
      (next.dp - previous.dp).norm() <=
          max_acceleration * transition_dt + kVelocityTolerance &&
      (next.ddp - previous.ddp).norm() <=
          max_jerk * transition_dt + kAccelerationTolerance &&
      (next.w - previous.w).norm() <=
          max_angular_acceleration * transition_dt + kVelocityTolerance &&
      (next.dw - previous.dw).norm() <=
          max_angular_jerk * transition_dt + kAccelerationTolerance;
  // The general replanner keeps position/path progress command-continuous but
  // intentionally reanchors derivatives to measured robot motion. Pose and
  // scalar path state must remain continuous; only this explicitly requested
  // derivative seam is exempt from command-to-command derivative comparison.
  return path_state_continuous && pose_continuous &&
         (allow_measured_derivative_reanchor || derivatives_continuous);
}

void ReachableCartesianImpedanceController::alignVerifiedPlanExecutionIndex(
    VerifiedPlan* plan,
    std::size_t elapsed_control_steps) const {
  if (plan == nullptr || !plan->valid) {
    return;
  }

  const std::size_t intended_size = plan->intended.size();
  plan->intended_exec_index = std::min(elapsed_control_steps, intended_size);
  plan->failsafe_exec_index =
      elapsed_control_steps > intended_size
          ? elapsed_control_steps - intended_size
          : 0;
}

std::size_t ReachableCartesianImpedanceController::failsafeCommandCount(
    const VerifiedPlan& plan) const {
  if (!plan.valid || plan.failsafe.empty()) {
    return 0;
  }
  const ImpedanceSample& failsafe_start =
      plan.intended.empty() ? plan.anchor : plan.intended.back();
  const double duration =
      std::max(0.0, plan.failsafe.back().t - failsafe_start.t);
  return static_cast<std::size_t>(std::max(
      1.0,
      std::ceil(duration / std::max(local_replan_dt_, kMinDt) - 1.0e-9)));
}

bool ReachableCartesianImpedanceController::calibrationExecutionComplete()
    const {
  return calibration_plan_latched_ &&
         last_verified_plan_.valid &&
         last_verified_plan_.intended_exec_index >=
             last_verified_plan_.intended.size() &&
         last_verified_plan_.failsafe_exec_index >=
             calibration_failsafe_command_count_;
}

// ============================================================================
// Generate intended prefix from the trajectory generator package
// ============================================================================
std::vector<ImpedanceSample>
ReachableCartesianImpedanceController::makeIntendedBufferFromReplanner(
    double nominal_guess_time,
    const ImpedanceSample& planning_start_command,
    double initial_path_rate,
    double target_path_rate,
    double commanded_path_time,
    bool reanchor_path_kinematics,
    double reanchor_path_rate,
    double reanchor_path_acceleration,
    PlanFailureReason* failure_reason) const {
  if (failure_reason != nullptr) {
    *failure_reason = PlanFailureReason::kNone;
  }
  // Strict path-consistent execution never reconstructs scalar progress from an
  // arbitrary Cartesian state.  These time guesses are retained in the caller
  // interface for logging/backward compatibility, but cannot authorize motion.
  (void)nominal_guess_time;
  (void)commanded_path_time;
  CartesianTrajectorySample planning_start;
  planning_start.t = planning_start_command.t;
  planning_start.p = planning_start_command.p;
  planning_start.dp = planning_start_command.dp;
  planning_start.ddp = planning_start_command.ddp;
  planning_start.q = planning_start_command.q;
  planning_start.q.normalize();
  planning_start.w = planning_start_command.w;
  planning_start.dw = planning_start_command.dw;

  std::vector<CartesianTrajectorySample> active_path;
  {
    std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
    active_path = cartesian_via_point_path_;
  }

  if (active_path.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = PlanFailureReason::kNoActivePath;
    }
    return {};
  }
  if (!planning_start_command.nominal_path_time_valid) {
    if (failure_reason != nullptr) {
      *failure_reason = PlanFailureReason::kMissingNominalPathState;
    }
    return {};
  }

  std::vector<CartesianTrajectorySample> planned_samples;
  bool planned_samples_are_path_consistent = false;
  auto has_continuous_seam =
      [&](const std::vector<CartesianTrajectorySample>& samples) {
    if (samples.empty()) {
      return false;
    }
    const auto& first = samples.front();
    const double dt = std::max(local_replan_dt_, kMinDt);
    const double linear_accel_step =
        local_replan_max_acceleration_ * dt;
    const double linear_jerk_step =
        local_replan_max_jerk_ * dt;
    const double angular_accel_step =
        local_replan_max_angular_acceleration_ * dt;
    const double angular_jerk_step =
        local_replan_max_angular_jerk_ * dt;
    constexpr double kSeamTolerance = 1.0e-6;

    const double max_linear_speed =
        std::max(planning_start.dp.norm(), first.dp.norm());
    const double max_angular_speed =
        std::max(planning_start.w.norm(), first.w.norm());
    const bool pose_continuous =
        (first.p - planning_start.p).norm() <=
            max_linear_speed * dt +
                0.5 * local_replan_max_acceleration_ * dt * dt +
                kSeamTolerance &&
        computeOrientationError(planning_start.q, first.q).norm() <=
            max_angular_speed * dt +
                0.5 * local_replan_max_angular_acceleration_ * dt * dt +
                kSeamTolerance;
    const bool derivatives_continuous =
        (first.dp - planning_start.dp).norm() <=
            linear_accel_step + kSeamTolerance &&
        (first.ddp - planning_start.ddp).norm() <=
            linear_jerk_step + kSeamTolerance &&
        (first.w - planning_start.w).norm() <=
            angular_accel_step + kSeamTolerance &&
        (first.dw - planning_start.dw).norm() <=
            angular_jerk_step + kSeamTolerance;
    return pose_continuous &&
           (reanchor_path_kinematics || derivatives_continuous);
  };

  if (!active_path.empty() &&
      planning_start_command.nominal_path_time_valid) {
    const double path_start_time =
        planning_start_command.nominal_path_time;

    // Keep the command position on its exact scalar path state, but initialize
    // the new Ruckig segment with measured along-path velocity and
    // acceleration. This preserves forward command-position continuity while
    // avoiding a restart from artificial zero command derivatives.
    if (reanchor_path_kinematics) {
      planning_start = makeRetimedPathState(
          active_path,
          path_start_time,
          std::clamp(reanchor_path_rate,
                     path_time_rate_min_,
                     path_time_rate_max_),
          std::clamp(reanchor_path_acceleration,
                     -path_time_acc_limit_,
                     path_time_acc_limit_));
      planning_start.t = planning_start_command.t;
    }

    // SaRA path-consistent mode advances a scalar progress state on the
    // long-term trajectory.  Prefer that construction so every intended
    // command stays on the requested via-point route.
    PathConsistentTimedPathConfig path_config;
    path_config.intended_steps = std::max(1, local_replan_horizon_steps_);
    path_config.dt = local_replan_dt_;
    path_config.path_lookahead_sec = local_path_lookahead_sec_;
    path_config.project_start_to_nearest_path_state = false;
    path_config.max_path_rate = std::max(path_time_rate_max_, 1e-4);
    path_config.max_path_acceleration =
        std::max(path_time_acc_limit_, 1e-4);
    path_config.max_path_jerk = std::max(path_time_jerk_limit_, 1e-4);
    path_config.target_path_rate =
        std::clamp(target_path_rate, path_time_rate_min_,
                   path_time_rate_max_);
    const bool starting_new_timed_path =
        path_start_time <= active_path.front().t + kMinDt;
    if (reanchor_path_kinematics) {
      path_config.initial_path_rate = std::clamp(
          reanchor_path_rate, path_time_rate_min_, path_time_rate_max_);
      path_config.initial_path_acceleration = std::clamp(
          reanchor_path_acceleration,
          -path_time_acc_limit_,
          path_time_acc_limit_);
    } else if (starting_new_timed_path) {
      path_config.initial_path_rate = std::clamp(
          initial_path_rate, path_time_rate_min_, path_time_rate_max_);
    } else {
      path_config.initial_path_rate = -1.0;
    }

    planned_samples = makePathConsistentTimedPathIntendedPrefix(
        path_start_time,
        planning_start,
        active_path,
        path_config);
    if (planned_samples.empty()) {
      if (failure_reason != nullptr) {
        *failure_reason = PlanFailureReason::kIntendedGenerationEmpty;
      }
    } else if (!has_continuous_seam(planned_samples)) {
      if (failure_reason != nullptr) {
        *failure_reason = PlanFailureReason::kIntendedSeamInvalid;
      }
      planned_samples.clear();
    } else {
      planned_samples_are_path_consistent = true;
    }
  }

  if (planned_samples.empty() && failure_reason != nullptr &&
      *failure_reason == PlanFailureReason::kNone) {
    *failure_reason = PlanFailureReason::kIntendedGenerationEmpty;
  }

  std::vector<ImpedanceSample> intended_buffer;
  intended_buffer.reserve(planned_samples.size());

  const std::size_t gain_blend_steps = std::max<std::size_t>(
      1, static_cast<std::size_t>(local_replan_horizon_steps_));
  for (std::size_t i = 0; i < planned_samples.size(); ++i) {
    const auto& planned_sample = planned_samples[i];
    ImpedanceSample s;
    s.t = planned_sample.t;
    if (planned_samples_are_path_consistent) {
      s.nominal_path_time = planned_sample.t;
      s.nominal_path_time_valid = true;
      s.nominal_path_rate = planned_sample.path_rate;
      s.nominal_path_acceleration = planned_sample.path_acceleration;
      s.nominal_path_kinematics_valid =
          planned_sample.path_kinematics_valid;
    }
    s.p = planned_sample.p;
    s.dp = planned_sample.dp;
    s.ddp = planned_sample.ddp;
    s.q = planned_sample.q;
    s.q.normalize();
    s.w = planned_sample.w;
    s.dw = planned_sample.dw;
    const double gain_blend = std::clamp(
        static_cast<double>(i + 1) /
            static_cast<double>(gain_blend_steps),
        0.0,
        1.0);
    s.K = (1.0 - gain_blend) * planning_start_command.K +
          gain_blend * K_base_;
    s.D = (1.0 - gain_blend) * planning_start_command.D +
          gain_blend * D_base_;
    s.failsafe = false;

    intended_buffer.push_back(s);
  }

  return intended_buffer;
}

// ============================================================================
// Build a candidate plan whose intended prefix is executed by the 1 kHz loop
// until the next monitor update arrives.
// ============================================================================
VerifiedPlan ReachableCartesianImpedanceController::buildCandidatePlan(
    double wall_time,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    const std::vector<ImpedanceSample>& intended_samples,
    std::size_t derivative_reanchor_index,
    PlanFailureReason* failure_reason) const {
  (void)inertia;
  (void)Jv;
  if (failure_reason != nullptr) {
    *failure_reason = PlanFailureReason::kNone;
  }

  VerifiedPlan plan;
  plan.valid = false;
  plan.generated_wall_time = wall_time;
  plan.intended_exec_index = 0;
  plan.failsafe_exec_index = 0;

  // Pin one geometric path for the complete candidate.  Besides avoiding a
  // mixed old-path intended/new-path brake during an action update, this
  // snapshot lets the final executable check prove that every command still
  // belongs to the path identified by its scalar progress coordinate.
  std::vector<CartesianTrajectorySample> active_path;
  {
    std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
    active_path = cartesian_via_point_path_;
  }
  if (active_path.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = PlanFailureReason::kNoActivePath;
    }
    return plan;
  }

  plan.anchor.t = 0.0;
  plan.anchor.p = current_position;
  plan.anchor.dp = ee_twist.head<3>();
  plan.anchor.ddp.setZero();
  plan.anchor.q = current_orientation;
  plan.anchor.q.normalize();
  plan.anchor.w = ee_twist.tail<3>();
  plan.anchor.dw.setZero();
  plan.anchor.K = K_runtime;
  plan.anchor.D = D_runtime;
  plan.anchor.failsafe = false;

  plan.intended.clear();
  plan.intended.reserve(intended_samples.size());
  for (std::size_t i = 0; i < intended_samples.size(); ++i) {
    ImpedanceSample step = intended_samples[i];
    step.t = plan.anchor.t + static_cast<double>(i + 1) *
                            std::max(local_replan_dt_, kMinDt);
    plan.intended.push_back(step);
  }

  if (plan.intended.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = PlanFailureReason::kIntendedGenerationEmpty;
    }
    return plan;
  }

  plan.nominal_time_anchor =
      intended_samples.back().nominal_path_time_valid
          ? intended_samples.back().nominal_path_time
          : intended_samples.back().t;
  const ImpedanceSample freeze_anchor = plan.intended.back();
  // Match SaRA-Shield's separation between the dense controller trajectory
  // and the coarser reachability interval edges: braking remains sampled at
  // the controller period, while makeSparsePlanForMonitor() groups those
  // samples into shield_plan_dt_ intervals.
  const double failsafe_plan_dt = std::max(local_replan_dt_, kMinDt);

  auto fill_failsafe_prefix = [&]() {
    plan.failsafe.clear();

    CartesianTrajectorySample brake_start;
    brake_start.t = freeze_anchor.t;
    brake_start.p = freeze_anchor.p;
    brake_start.dp = freeze_anchor.dp;
    brake_start.ddp = freeze_anchor.ddp;
    brake_start.q = freeze_anchor.q;
    brake_start.q.normalize();
    brake_start.w = freeze_anchor.w;
    brake_start.dw = freeze_anchor.dw;

    std::vector<CartesianTrajectorySample> brake_samples;
    bool path_consistent_brake = false;
    if (freeze_anchor.nominal_path_time_valid) {
      PathConsistentTimedPathConfig path_brake_config;
      path_brake_config.dt = failsafe_plan_dt;
      path_brake_config.path_lookahead_sec = local_path_lookahead_sec_;
      path_brake_config.max_path_rate = std::max(path_time_rate_max_, 1e-4);
      path_brake_config.max_path_acceleration =
          std::max(failsafe_path_time_acc_limit_, 1e-4);
      path_brake_config.max_path_jerk =
          std::max(failsafe_path_time_jerk_limit_, 1e-4);
      path_brake_config.target_path_rate = 0.0;
      if (freeze_anchor.nominal_path_kinematics_valid) {
        path_brake_config.initial_path_rate =
            freeze_anchor.nominal_path_rate;
        path_brake_config.initial_path_acceleration =
            freeze_anchor.nominal_path_acceleration;
      }

      brake_samples = makePathConsistentTimedPathBrake(
          freeze_anchor.nominal_path_time,
          brake_start,
          active_path,
          path_brake_config);

      // A path-consistent stop is admissible only if the intended endpoint is
      // actually on that path state.  This rejects the former hybrid jump
      // from a Cartesian OTG endpoint to an unrelated timed-path sample.
      if (!brake_samples.empty()) {
        const auto& first = brake_samples.front();
        const double dt = failsafe_plan_dt;
        constexpr double kBrakeSeamTolerance = 1.0e-6;
        const bool position_continuous =
            (first.p - freeze_anchor.p).norm() <=
            std::max(freeze_anchor.dp.norm(), first.dp.norm()) * dt +
                0.5 * failsafe_brake_max_acceleration_ * dt * dt +
                kBrakeSeamTolerance;
        const bool velocity_continuous =
            (first.dp - freeze_anchor.dp).norm() <=
            failsafe_brake_max_acceleration_ * dt +
                kBrakeSeamTolerance;
        const bool acceleration_continuous =
            (first.ddp - freeze_anchor.ddp).norm() <=
            failsafe_brake_max_jerk_ * dt +
                kBrakeSeamTolerance;
        const bool orientation_continuous =
            computeOrientationError(freeze_anchor.q, first.q).norm() <=
            std::max(freeze_anchor.w.norm(), first.w.norm()) * dt +
                0.5 * failsafe_brake_max_angular_acceleration_ * dt * dt +
                kBrakeSeamTolerance;
        const bool angular_velocity_continuous =
            (first.w - freeze_anchor.w).norm() <=
            failsafe_brake_max_angular_acceleration_ * dt +
                kBrakeSeamTolerance;
        const bool angular_acceleration_continuous =
            (first.dw - freeze_anchor.dw).norm() <=
            failsafe_brake_max_angular_jerk_ * dt +
                kBrakeSeamTolerance;
        path_consistent_brake =
            position_continuous && velocity_continuous &&
            acceleration_continuous && orientation_continuous &&
            angular_velocity_continuous &&
            angular_acceleration_continuous;
        if (!path_consistent_brake) {
          if (failure_reason != nullptr) {
            *failure_reason = PlanFailureReason::kFailsafeSeamInvalid;
          }
          brake_samples.clear();
        }
      }
    }

    // Strict path consistency: an unavailable or discontinuous scalar brake
    // invalidates the candidate.  Never replace it with a Cartesian shortcut.
    if (brake_samples.empty()) {
      if (failure_reason != nullptr &&
          *failure_reason == PlanFailureReason::kNone) {
        *failure_reason = PlanFailureReason::kFailsafeGenerationEmpty;
      }
      return;
    }

    const int N_fs = static_cast<int>(brake_samples.size());
    plan.failsafe.reserve(static_cast<std::size_t>(N_fs));
    for (int i = 0; i < N_fs; ++i) {
      const double tk =
          freeze_anchor.t + static_cast<double>(i + 1) * failsafe_plan_dt;

      const double s_blend =
          static_cast<double>(i + 1) / static_cast<double>(N_fs);

      const Matrix6d K_sched =
          (1.0 - s_blend) * freeze_anchor.K + s_blend * K_base_;

      const Matrix6d D_sched =
          (1.0 - s_blend) * freeze_anchor.D +
          s_blend * D_base_;

      ImpedanceSample s;
      const auto& brake = brake_samples[static_cast<std::size_t>(i)];
      s.t = tk;
      if (path_consistent_brake) {
        s.nominal_path_time = brake.t;
        s.nominal_path_time_valid = true;
        s.nominal_path_rate = brake.path_rate;
        s.nominal_path_acceleration = brake.path_acceleration;
        s.nominal_path_kinematics_valid = brake.path_kinematics_valid;
      }
      s.p = brake.p;
      s.dp = brake.dp;
      s.ddp = brake.ddp;
      s.q = brake.q;
      s.q.normalize();
      s.w = brake.w;
      s.dw = brake.dw;
      s.K = K_sched;
      s.D = D_sched;
      s.failsafe = true;
      plan.failsafe.push_back(s);
    }
  };

  // Intended, failsafe and emergency commands share one configured Cartesian
  // gain set. Keep this interpolation only to avoid a gain discontinuity when
  // the freeze anchor was scaled by the runtime energy budget.
  fill_failsafe_prefix();

  auto sample_within_limits = [&](const ImpedanceSample& sample) {
    const bool failsafe_limits = sample.failsafe;
    const double max_velocity =
        failsafe_limits ? failsafe_brake_max_velocity_
                        : local_replan_max_velocity_;
    const double max_acceleration =
        failsafe_limits ? failsafe_brake_max_acceleration_
                        : local_replan_max_acceleration_;
    const double max_angular_velocity =
        failsafe_limits ? failsafe_brake_max_angular_velocity_
                        : local_replan_max_angular_velocity_;
    const double max_angular_acceleration =
        failsafe_limits ? failsafe_brake_max_angular_acceleration_
                        : local_replan_max_angular_acceleration_;
    // Fail-safe samples are stored at the sparse monitor period but executed
    // at 1 kHz by interpolation. On a curved path the interpolated chord can
    // differ from the exact path evaluation by a few micrometres. Accept that
    // bounded numerical discrepancy while still rejecting any geometrically
    // different route.
    constexpr double kPathPoseTolerance = 1.0e-5;
    constexpr double kPathOrientationTolerance = 1.0e-8;
    const bool has_scalar_path_state =
        sample.nominal_path_time_valid &&
        std::isfinite(sample.nominal_path_time) &&
        sample.nominal_path_time >=
            active_path.front().t - kPathPoseTolerance &&
        sample.nominal_path_time <=
            active_path.back().t + kPathPoseTolerance;
    bool pose_is_on_active_path = false;
    if (has_scalar_path_state) {
      const CartesianTrajectorySample path_state = makeRetimedPathState(
          active_path, sample.nominal_path_time, 0.0, 0.0);
      const Quaterniond sample_orientation =
          normalizedQuaternionOrIdentity(sample.q);
      const Quaterniond path_orientation =
          normalizedQuaternionOrIdentity(path_state.q);
      pose_is_on_active_path =
          (sample.p - path_state.p).norm() <= kPathPoseTolerance &&
          1.0 - std::abs(sample_orientation.dot(path_orientation)) <=
              kPathOrientationTolerance;
    }
    return has_scalar_path_state && pose_is_on_active_path &&
           std::isfinite(sample.t) && sample.p.allFinite() &&
           sample.dp.allFinite() && sample.ddp.allFinite() &&
           sample.q.coeffs().allFinite() && sample.w.allFinite() &&
           sample.dw.allFinite() && sample.K.allFinite() &&
           sample.D.allFinite() &&
           sample.dp.norm() <= max_velocity + 1.0e-6 &&
           sample.ddp.norm() <= max_acceleration + 1.0e-6 &&
           sample.w.norm() <= max_angular_velocity + 1.0e-6 &&
           sample.dw.norm() <=
               max_angular_acceleration + 1.0e-6;
  };

  auto transition_within_limits =
      [&](const ImpedanceSample& previous,
          const ImpedanceSample& current,
          bool allow_measured_derivative_reanchor) {
    const bool failsafe_limits = previous.failsafe || current.failsafe;
    const double dt = std::max(current.t - previous.t, kMinDt);
    const double max_acceleration =
        failsafe_limits ? failsafe_brake_max_acceleration_
                        : local_replan_max_acceleration_;
    const double max_jerk =
        failsafe_limits ? failsafe_brake_max_jerk_
                        : local_replan_max_jerk_;
    const double max_angular_acceleration =
        failsafe_limits ? failsafe_brake_max_angular_acceleration_
                        : local_replan_max_angular_acceleration_;
    const double max_angular_jerk =
        failsafe_limits ? failsafe_brake_max_angular_jerk_
                        : local_replan_max_angular_jerk_;
    constexpr double kPositionTolerance = 1.0e-5;
    constexpr double kVelocityTolerance = 1.0e-5;
    constexpr double kAccelerationTolerance = 1.0e-4;
    constexpr double kPathTimeTolerance = 1.0e-6;
    const bool path_state_continuous =
        previous.nominal_path_time_valid &&
        current.nominal_path_time_valid &&
        current.nominal_path_time >=
            previous.nominal_path_time - kPathTimeTolerance &&
        current.nominal_path_time - previous.nominal_path_time <=
            path_time_rate_max_ * dt + kPathTimeTolerance;
    const bool pose_and_path_continuous =
        path_state_continuous &&
        (current.p - previous.p).norm() <=
            std::max(previous.dp.norm(), current.dp.norm()) * dt +
                0.5 * max_acceleration * dt * dt +
                kPositionTolerance &&
        computeOrientationError(previous.q, current.q).norm() <=
            std::max(previous.w.norm(), current.w.norm()) * dt +
                0.5 * max_angular_acceleration * dt * dt +
                kPositionTolerance;
    const bool derivatives_continuous =
        (current.dp - previous.dp).norm() <=
            max_acceleration * dt + kVelocityTolerance &&
        (current.ddp - previous.ddp).norm() <=
            max_jerk * dt + kAccelerationTolerance &&
        (current.w - previous.w).norm() <=
            max_angular_acceleration * dt + kVelocityTolerance &&
        (current.dw - previous.dw).norm() <=
            max_angular_jerk * dt + kAccelerationTolerance;
    // At the explicit measured-state reanchor, desired derivatives may differ
    // from the preceding command. Pose and scalar path progress remain
    // continuous, and the complete candidate is still rolled out from the
    // measured state by the safety monitor before it can be executed.
    return pose_and_path_continuous &&
           (allow_measured_derivative_reanchor || derivatives_continuous);
  };

  bool executable = !plan.intended.empty() && !plan.failsafe.empty();
  if (!executable && failure_reason != nullptr &&
      *failure_reason == PlanFailureReason::kNone) {
    *failure_reason = PlanFailureReason::kCandidateInvalidUnknown;
  }
  const ImpedanceSample* previous = nullptr;
  for (std::size_t i = 0; i < plan.intended.size(); ++i) {
    const auto& sample = plan.intended[i];
    const bool sample_valid = sample_within_limits(sample);
    if (!sample_valid && failure_reason != nullptr &&
        *failure_reason == PlanFailureReason::kNone) {
      *failure_reason = PlanFailureReason::kIntendedSampleInvalid;
    }
    executable = executable && sample_valid;
    if (previous != nullptr) {
      const bool transition_valid = transition_within_limits(
          *previous,
          sample,
          i == derivative_reanchor_index);
      if (!transition_valid && failure_reason != nullptr &&
          *failure_reason == PlanFailureReason::kNone) {
        *failure_reason = PlanFailureReason::kIntendedTransitionInvalid;
      }
      executable = executable && transition_valid;
    }
    previous = &sample;
  }
  for (const auto& sample : plan.failsafe) {
    const bool sample_valid = sample_within_limits(sample);
    if (!sample_valid && failure_reason != nullptr &&
        *failure_reason == PlanFailureReason::kNone) {
      *failure_reason = PlanFailureReason::kFailsafeSampleInvalid;
    }
    executable = executable && sample_valid;
    if (previous != nullptr) {
      const bool transition_valid =
          transition_within_limits(*previous, sample, false);
      if (!transition_valid && failure_reason != nullptr &&
          *failure_reason == PlanFailureReason::kNone) {
        *failure_reason = PlanFailureReason::kFailsafeTransitionInvalid;
      }
      executable = executable && transition_valid;
    }
    previous = &sample;
  }

  plan.valid = executable;
  if (plan.valid && failure_reason != nullptr) {
    *failure_reason = PlanFailureReason::kNone;
  } else if (!plan.valid && failure_reason != nullptr &&
             *failure_reason == PlanFailureReason::kNone) {
    *failure_reason = PlanFailureReason::kCandidateInvalidUnknown;
  }
  return plan;
}

SafetyMonitorConfig ReachableCartesianImpedanceController::makeSafetyMonitorConfig(
    const cps_human_workspace::HumanWorkspace& human_workspace,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    double wall_time) const {
  SafetyMonitorConfig config;
  config.human_workspace = human_workspace;
  config.K_runtime = K_runtime;
  config.D_runtime = D_runtime;
  config.wall_time_sec = wall_time;
  config.energy_budget_joule = energy_budget_joule_;
  config.kinetic_energy_error_bound_joule =
      kinetic_energy_error_bound_joule_;
  config.potential_energy_error_bound_joule =
      potential_energy_error_bound_joule_;
  config.nullspace_potential_energy_error_bound_joule =
      nullspace_potential_energy_error_bound_joule_;
  config.enable_runtime_energy_scaling =
      enable_runtime_energy_scaling_;
  config.enable_overbudget_joint_stabilization =
      enable_overbudget_joint_stabilization_;
  config.overbudget_joint_stiffness =
      overbudget_joint_stiffness_;
  config.overbudget_joint_scale_omega =
      overbudget_joint_scale_omega_;
  config.robot_reachability_provider = robot_reachability_provider_;
  config.ee_collision_radius = ee_collision_radius_;
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

double ReachableCartesianImpedanceController::estimatePathRateFromTimedPathSample(
    double path_time,
    const Vector3d& cartesian_velocity) const {
  std::vector<CartesianTrajectorySample> path;
  {
    std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
    path = cartesian_via_point_path_;
  }
  if (path.empty()) {
    return 0.0;
  }

  const double t =
      std::clamp(path_time,
                 path.front().t,
                 path.back().t);
  auto upper = std::lower_bound(
      path.begin(),
      path.end(),
      t,
      [](const CartesianTrajectorySample& sample, double value) {
        return sample.t < value;
      });

  Vector3d path_velocity = Vector3d::Zero();
  if (upper == path.begin()) {
    path_velocity = upper->dp;
  } else if (upper == path.end()) {
    path_velocity = path.back().dp;
  } else {
    const auto lower = upper - 1;
    const double span = std::max(upper->t - lower->t, kMinDt);
    const double alpha = std::clamp((t - lower->t) / span, 0.0, 1.0);
    path_velocity = (1.0 - alpha) * lower->dp + alpha * upper->dp;
  }

  const double denom = path_velocity.squaredNorm();
  if (denom < kSmallPositive) {
    return 0.0;
  }
  const double rate = cartesian_velocity.dot(path_velocity) / denom;
  if (!std::isfinite(rate)) {
    return 0.0;
  }
  return std::clamp(rate, path_time_rate_min_, path_time_rate_max_);
}

VerifiedPlan ReachableCartesianImpedanceController::makeSparsePlanForMonitor(
    const VerifiedPlan& dense_plan) const {
  const double interval_dt = std::max(shield_plan_dt_, kMinDt);
  VerifiedPlan sparse_plan;
  sparse_plan.anchor = dense_plan.anchor;
  sparse_plan.generated_wall_time = dense_plan.generated_wall_time;
  sparse_plan.nominal_time_anchor = dense_plan.nominal_time_anchor;
  sparse_plan.intended_exec_index = 0;
  sparse_plan.failsafe_exec_index = 0;

  double next_edge_time = sparse_plan.anchor.t + interval_dt;
  constexpr double kTimeEps = 1.0e-9;

  auto append_stage = [&](const std::vector<ImpedanceSample>& dense_samples,
                          std::vector<ImpedanceSample>* sparse_samples) {
    if (sparse_samples == nullptr || dense_samples.empty()) {
      return;
    }

    for (const auto& sample : dense_samples) {
      if (sample.t + kTimeEps >= next_edge_time) {
        sparse_samples->push_back(sample);
        next_edge_time += interval_dt;
      }
    }

    const ImpedanceSample& stage_end = dense_samples.back();
    if (sparse_samples->empty() ||
        std::abs(sparse_samples->back().t - stage_end.t) > kTimeEps) {
      sparse_samples->push_back(stage_end);
    }
  };

  append_stage(dense_plan.intended, &sparse_plan.intended);
  append_stage(dense_plan.failsafe, &sparse_plan.failsafe);

  sparse_plan.valid =
      dense_plan.valid && !sparse_plan.intended.empty() &&
      !sparse_plan.failsafe.empty();
  return sparse_plan;
}

VerifiedPlan ReachableCartesianImpedanceController::makeCollisionCenterPlanForMonitor(
    const VerifiedPlan& flange_plan) const {
  VerifiedPlan plan = flange_plan;
  plan.anchor = makeCollisionCenterSampleForMonitor(plan.anchor);
  for (auto& sample : plan.intended) {
    sample = makeCollisionCenterSampleForMonitor(sample);
  }
  for (auto& sample : plan.failsafe) {
    sample = makeCollisionCenterSampleForMonitor(sample);
  }

  return plan;
}

ImpedanceSample
ReachableCartesianImpedanceController::makeCollisionCenterSampleForMonitor(
    const ImpedanceSample& flange_sample) const {
  ImpedanceSample sample = flange_sample;
  const Vector3d offset_world =
      sample.q.normalized() * ee_collision_center_offset_;
  sample.p += offset_world;
  sample.dp += sample.w.cross(offset_world);
  sample.ddp += sample.dw.cross(offset_world) +
                sample.w.cross(sample.w.cross(offset_world));
  return sample;
}

// ============================================================================
// evaluateCandidatePlan
// ============================================================================
MonitorResult ReachableCartesianImpedanceController::evaluateCandidatePlan(
    const VerifiedPlan& plan,
    const Vector7d& q,
    const Vector7d& dq,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix67d& J_geo,
    const Vector7d& coriolis,
    const Vector6d& control_jdot_dq,
    const Vector7d& previous_torque_command,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    const cps_human_workspace::HumanWorkspace& human_workspace,
    bool human_workspace_active,
    bool human_workspace_assumed_clear,
    const ImpedanceSample& current_command_reference,
    bool current_command_reference_valid,
    double current_nullspace_stiffness,
    const cps_safety_monitor::OverbudgetJointStabilizationState&
        overbudget_joint_state,
    std::vector<JointPredictionSample>* joint_prediction_trace) const {
  if (joint_prediction_trace != nullptr) {
    joint_prediction_trace->clear();
  }
  const bool human_workspace_available =
      human_workspace_active || human_workspace_assumed_clear;
  if (!enable_safety_monitor_ || !human_workspace_available) {
    return MonitorResult{};
  }

  const VerifiedPlan monitor_plan = makeSparsePlanForMonitor(plan);

  SafetyMonitorConfig config = makeSafetyMonitorConfig(
      human_workspace,
      K_runtime,
      D_runtime,
      plan.generated_wall_time);
  config.assume_human_workspace_clear = human_workspace_assumed_clear;
  config.current_energy_reference_valid = current_command_reference_valid;
  if (current_command_reference_valid) {
    config.current_energy_reference = current_command_reference;
  }
  config.nullspace_reference = desired_qn_;
  config.nullspace_stiffness = n_stiffness_;
  config.current_nullspace_stiffness =
      std::max(0.0, current_nullspace_stiffness);
  config.overbudget_joint_state = overbudget_joint_state;
  config.collision_center_offset = ee_collision_center_offset_;
  config.previous_torque_command = previous_torque_command;
  config.previous_torque_command_valid = true;
  config.torque_rate_limit = panda_limits::kTorqueRateLimit;
  // SaRA evaluates robot and human occupancies on the same sparse interval
  // grid. Keep dense command generation at local_replan_dt_, but do not split
  // each shield interval back into controller-period reachable sets here.
  config.joint_rollout_max_dt = shield_plan_dt_;

  // Use the snapshot read by the 1 kHz controller for the rollout's initial
  // state. J_geo is expressed at the collision center here; shift it back to
  // the controlled TCP because collision_center_offset is applied separately
  // inside the joint monitor.
  config.current_joint_dynamics.control_position = current_position;
  config.current_joint_dynamics.control_orientation = current_orientation;
  config.current_joint_dynamics.control_jacobian = J_geo;
  const Vector3d collision_offset_world =
      current_orientation.normalized() * ee_collision_center_offset_;
  config.current_joint_dynamics.control_jacobian.topRows<3>() +=
      skewSymmetric(collision_offset_world) * J_geo.bottomRows<3>();
  config.current_joint_dynamics.control_jdot_dq = control_jdot_dq;
  config.current_joint_dynamics.inertia = inertia;
  config.current_joint_dynamics.coriolis = coriolis;
  config.current_joint_dynamics.valid =
      current_position.allFinite() &&
      current_orientation.coeffs().allFinite() &&
      config.current_joint_dynamics.control_jacobian.allFinite() &&
      control_jdot_dq.allFinite() && inertia.allFinite() &&
      coriolis.allFinite();
  config.current_joint_dynamics_valid =
      config.current_joint_dynamics.valid;
  config.enable_inertia_model_comparison = enable_prediction_logging_;

  if (monitor_joint_dynamics_provider_) {
    return cps_safety_monitor::verifyReachablePlanJointSpace(
        plan,
        q,
        dq,
        *monitor_joint_dynamics_provider_,
        config,
        joint_prediction_trace);
  }

  const VerifiedPlan collision_center_plan =
      makeCollisionCenterPlanForMonitor(monitor_plan);
  const Vector3d collision_center =
      current_position + collisionCenterOffsetWorld(current_orientation);
  const Vector6d collision_twist =
      twistAtCollisionCenter(current_orientation, ee_twist);

  return cps_safety_monitor::verifyReachablePlan(
      collision_center_plan,
      collision_center,
      current_orientation,
      collision_twist,
      inertia,
      J_geo,
      config);
}

// ============================================================================
// Shield decision
// ============================================================================
ShieldDecision ReachableCartesianImpedanceController::computeShieldDecision(
    double wall_time, double nominal_guess_time,
    const Vector7d& q, const Vector7d& dq,
    const Vector3d& current_position, const Quaterniond& current_orientation,
    const Vector6d& ee_twist, const Matrix7d& inertia, const Matrix67d& J_geo,
    const Vector7d& coriolis, const Vector6d& control_jdot_dq) {
  const auto decision_tic = SteadyClock::now();
  double planner_ms = 0.0;
  double plan_build_ms = 0.0;
  double monitor_eval_ms = 0.0;
  ShieldDecision dec;
  VerifiedPlan candidate_plan;

  auto stamp_timing = [&]() {
    dec.monitor_total_ms =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - decision_tic).count();
    dec.planner_ms = planner_ms;
    dec.plan_build_ms = plan_build_ms;
    dec.monitor_eval_ms = monitor_eval_ms;
  };

  auto evaluate_plan = [&](const VerifiedPlan& plan) {
    return evaluateCandidatePlan(
        plan,
        q,
        dq,
        current_position,
        current_orientation,
        ee_twist,
        inertia,
        J_geo,
        coriolis,
        control_jdot_dq,
        tau_cmd_prev_,
        K_runtime_,
        D_runtime_,
        human_workspace_,
        human_workspace_active_,
        false,
        last_commanded_sample_,
        last_commanded_sample_valid_,
        last_nullspace_stiffness_,
        overbudget_joint_state_,
        (enable_prediction_logging_ || enable_reachable_set_visualization_)
            ? &dec.joint_prediction_trace
            : nullptr);
  };

  auto execute_last_verified_monitored = [&](FallbackReason reason) {
    dec.executing_last_verified_monitored = true;
    dec.candidate_verified = false;
    dec.fallback_reason = reason;
    if (last_verified_plan_.valid &&
        (!last_verified_plan_.intended.empty() ||
         !last_verified_plan_.failsafe.empty())) {
      dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
    } else {
      dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
    }
  };

  auto try_one_monitor_pass = [&]() -> bool {
    PlanFailureReason plan_failure_reason = PlanFailureReason::kNone;
    ImpedanceSample planning_start_command;
    planning_start_command.t = wall_time;

    if (last_commanded_sample_valid_) {
      planning_start_command.p = last_commanded_sample_.p;
      planning_start_command.q = last_commanded_sample_.q;
      planning_start_command.dp = last_commanded_sample_.dp;
      planning_start_command.w = last_commanded_sample_.w;
      planning_start_command.ddp = last_commanded_sample_.ddp;
      planning_start_command.dw = last_commanded_sample_.dw;
      planning_start_command.K = last_commanded_sample_.K;
      planning_start_command.D = last_commanded_sample_.D;
      planning_start_command.nominal_path_time =
          last_commanded_sample_.nominal_path_time;
      planning_start_command.nominal_path_time_valid =
          last_commanded_sample_.nominal_path_time_valid;
      planning_start_command.nominal_path_rate =
          last_commanded_sample_.nominal_path_rate;
      planning_start_command.nominal_path_acceleration =
          last_commanded_sample_.nominal_path_acceleration;
      planning_start_command.nominal_path_kinematics_valid =
          last_commanded_sample_.nominal_path_kinematics_valid;
    } else {
      planning_start_command.p = current_position;
      planning_start_command.q = current_orientation;
      planning_start_command.dp = ee_twist.head<3>();
      planning_start_command.w = ee_twist.tail<3>();
      planning_start_command.ddp.setZero();
      planning_start_command.dw.setZero();
      planning_start_command.K = K_runtime_;
      planning_start_command.D = D_runtime_;
    }
    planning_start_command.q.normalize();

    planning_start_command.failsafe = false;

    const bool reanchor_path_kinematics =
        planning_start_command.nominal_path_time_valid &&
        planning_start_command.nominal_path_kinematics_valid;
    const double reanchor_path_rate =
        planning_start_command.nominal_path_rate;
    const double reanchor_path_acceleration =
        planning_start_command.nominal_path_acceleration;
    const auto planner_tic = SteadyClock::now();
    const std::vector<ImpedanceSample> intended_buffer =
        makeIntendedBufferFromReplanner(
            nominal_guess_time,
            planning_start_command,
            commanded_path_rate_,
            path_time_rate_target_,
            commanded_path_time_,
            reanchor_path_kinematics,
            reanchor_path_rate,
            reanchor_path_acceleration,
            &plan_failure_reason);
    planner_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - planner_tic).count();

    if (intended_buffer.empty()) {
      dec.plan_failure_reason = plan_failure_reason;
      return false;
    }

    const std::size_t segment_count = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(1, shield_intended_steps_)),
        intended_buffer.size());
    std::vector<ImpedanceSample> intended_segment(
        intended_buffer.begin(),
        intended_buffer.begin() + static_cast<std::ptrdiff_t>(segment_count));

    const auto build_tic = SteadyClock::now();
    candidate_plan = buildCandidatePlan(
        wall_time,
        current_position,
        current_orientation,
        ee_twist,
        inertia,
        J_geo.topRows<3>(),
        K_runtime_,
        D_runtime_,
        intended_segment,
        std::numeric_limits<std::size_t>::max(),
        &plan_failure_reason);
    plan_build_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - build_tic).count();
    if (!candidate_plan.valid) {
      dec.plan_failure_reason = plan_failure_reason;
      return false;
    }

    const auto eval_tic = SteadyClock::now();
    dec.monitor = evaluate_plan(candidate_plan);
    monitor_eval_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - eval_tic).count();
    dec.evaluated_plan = candidate_plan;
    dec.has_evaluated_plan = true;
    dec.candidate_verified = !shouldRejectCandidateWithMonitor(dec.monitor);
    return true;
  };

  const bool first_attempt = try_one_monitor_pass();

  if (!first_attempt) {
    // A planner/build failure is not proof that a contact state has ended.
    // Stay on the last verified monitored trajectory instead of allowing an
    // unverified default MonitorResult to select nominal mode.
    mode_ = isCollisionPossibleMode(mode_)
                ? SafetyMode::kLastVerifiedContactPossible
                : SafetyMode::kLastVerifiedMonitored;
    execute_last_verified_monitored(FallbackReason::kNone);
    stamp_timing();
    return dec;
  }

  if (dec.candidate_verified) {
    last_verified_plan_ = candidate_plan;
    last_verified_plan_.valid = true;
    last_verified_plan_.intended_exec_index = 0;
    last_verified_plan_.failsafe_exec_index = 0;
    ++last_verified_plan_generation_;

    mode_ = nominalSafetyModeForMonitor(dec.monitor);

    dec.executing_last_verified_monitored = false;
    dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
    stamp_timing();
    return dec;
  }

  mode_ = lastVerifiedSafetyModeForMonitor(dec.monitor);
  execute_last_verified_monitored(
      human_workspace_active_
          ? FallbackReason::kCandidatePredictionRejected
          : FallbackReason::kHumanWorkspaceUnavailable);
  stamp_timing();
  return dec;
}

ShieldDecision ReachableCartesianImpedanceController::computeShieldDecisionForAsyncInput(
    const AsyncMonitorInput& input,
    VerifiedPlan& last_verified_plan) const {
  const auto decision_tic = SteadyClock::now();
  double planner_ms = 0.0;
  double plan_build_ms = 0.0;
  double monitor_eval_ms = 0.0;
  ShieldDecision dec;

  auto stamp_timing = [&]() {
    dec.monitor_total_ms =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - decision_tic).count();
    dec.planner_ms = planner_ms;
    dec.plan_build_ms = plan_build_ms;
    dec.monitor_eval_ms = monitor_eval_ms;
  };

  auto get_next_verified_monitored_command =
      [&](bool advance_index) -> ImpedanceSample {
    if (!last_verified_plan.valid) {
      return ImpedanceSample{};
    }

    if (last_verified_plan.intended_exec_index <
        last_verified_plan.intended.size()) {
      const std::size_t idx = last_verified_plan.intended_exec_index;
      ImpedanceSample cmd = last_verified_plan.intended[idx];
      if (advance_index) {
        ++last_verified_plan.intended_exec_index;
      }
      return cmd;
    }

    if (last_verified_plan.failsafe.empty()) {
      return ImpedanceSample{};
    }

    const ImpedanceSample& failsafe_start =
        last_verified_plan.intended.empty()
            ? last_verified_plan.anchor
            : last_verified_plan.intended.back();
    const double command_dt = std::max(local_replan_dt_, kMinDt);
    const double command_time =
        failsafe_start.t +
        static_cast<double>(last_verified_plan.failsafe_exec_index + 1) *
            command_dt;

    ImpedanceSample cmd;
    const auto& failsafe = last_verified_plan.failsafe;
    if (command_time <= failsafe.front().t) {
      cmd = interpolateImpedanceSample(
          failsafe_start,
          failsafe.front(),
          command_time);
    } else if (command_time >= failsafe.back().t) {
      cmd = failsafe.back();
      cmd.t = command_time;
    } else {
      const auto upper = std::lower_bound(
          failsafe.begin(),
          failsafe.end(),
          command_time,
          [](const ImpedanceSample& sample, double value) {
            return sample.t < value;
          });
      const auto lower = upper - 1;
      cmd = interpolateImpedanceSample(*lower, *upper, command_time);
    }
    cmd.failsafe = true;

    if (advance_index) {
      ++last_verified_plan.failsafe_exec_index;
    }
    return cmd;
  };

  auto execute_last_verified_monitored = [&](FallbackReason reason) {
    dec.executing_last_verified_monitored = true;
    dec.candidate_verified = false;
    dec.fallback_reason = reason;

    if (last_verified_plan.valid &&
        (!last_verified_plan.intended.empty() ||
         !last_verified_plan.failsafe.empty())) {
      dec.command = get_next_verified_monitored_command(true);
    } else {
      dec.command = makeEmergencyStopCommand(
          input.current_position,
          input.current_orientation,
          input.wall_time);
    }
  };

  auto try_one_monitor_pass = [&]() -> bool {
    PlanFailureReason plan_failure_reason = PlanFailureReason::kNone;
    ImpedanceSample planning_start_command;
    if (!input.committed_prefix.empty()) {
      planning_start_command = input.committed_prefix.back();
    } else if (input.last_commanded_sample_valid) {
      planning_start_command.p = input.last_commanded_sample.p;
      planning_start_command.q = input.last_commanded_sample.q;
      planning_start_command.dp = input.last_commanded_sample.dp;
      planning_start_command.w = input.last_commanded_sample.w;
      planning_start_command.ddp = input.last_commanded_sample.ddp;
      planning_start_command.dw = input.last_commanded_sample.dw;
      planning_start_command.K = input.last_commanded_sample.K;
      planning_start_command.D = input.last_commanded_sample.D;
      planning_start_command.nominal_path_time =
          input.last_commanded_sample.nominal_path_time;
      planning_start_command.nominal_path_time_valid =
          input.last_commanded_sample.nominal_path_time_valid;
      planning_start_command.nominal_path_rate =
          input.last_commanded_sample.nominal_path_rate;
      planning_start_command.nominal_path_acceleration =
          input.last_commanded_sample.nominal_path_acceleration;
      planning_start_command.nominal_path_kinematics_valid =
          input.last_commanded_sample.nominal_path_kinematics_valid;
    } else {
      planning_start_command.p = input.current_position;
      planning_start_command.q = input.current_orientation;
      planning_start_command.dp = input.ee_twist.head<3>();
      planning_start_command.w = input.ee_twist.tail<3>();
      planning_start_command.ddp.setZero();
      planning_start_command.dw.setZero();
      planning_start_command.K = input.K_runtime;
      planning_start_command.D = input.D_runtime;
    }
    planning_start_command.t =
        input.wall_time +
        static_cast<double>(input.committed_prefix.size()) *
            std::max(local_replan_dt_, kMinDt);
    planning_start_command.q.normalize();

    planning_start_command.failsafe = false;

    const double nominal_advance_duration =
        static_cast<double>(input.nominal_advance_steps) *
        std::max(local_replan_dt_, kMinDt);
    const double activation_nominal_guess_time =
        input.nominal_guess_time + nominal_advance_duration;
    const double activation_commanded_path_time =
        planning_start_command.nominal_path_time_valid
            ? planning_start_command.nominal_path_time
            : input.commanded_path_time +
                  std::max(0.0, input.commanded_path_rate) *
                      nominal_advance_duration;

    const auto planner_tic = SteadyClock::now();
    const std::vector<ImpedanceSample> intended_buffer =
        makeIntendedBufferFromReplanner(
            activation_nominal_guess_time,
            planning_start_command,
            input.commanded_path_rate,
            input.target_path_rate,
            activation_commanded_path_time,
            input.reanchor_path_kinematics,
            input.reanchor_path_rate,
            input.reanchor_path_acceleration,
            &plan_failure_reason);
    planner_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - planner_tic).count();

    if (intended_buffer.empty()) {
      dec.plan_failure_reason = plan_failure_reason;
      return false;
    }

    const std::size_t segment_count = std::min<std::size_t>(
        std::max<std::size_t>(
            static_cast<std::size_t>(std::max(1, shield_intended_steps_)),
            async_verified_horizon_steps_),
        intended_buffer.size());
    std::vector<ImpedanceSample> replanned_segment(
        intended_buffer.begin(),
        intended_buffer.begin() + static_cast<std::ptrdiff_t>(segment_count));
    std::vector<ImpedanceSample> intended_segment;
    intended_segment.reserve(input.committed_prefix.size() + segment_count);
    intended_segment.insert(
        intended_segment.end(),
        input.committed_prefix.begin(),
        input.committed_prefix.end());
    intended_segment.insert(
        intended_segment.end(),
        replanned_segment.begin(),
        replanned_segment.end());

    const auto build_tic = SteadyClock::now();
    VerifiedPlan candidate_plan = buildCandidatePlan(
        input.wall_time,
        input.current_position,
        input.current_orientation,
        input.ee_twist,
        input.inertia,
        input.Jv,
        input.K_runtime,
        input.D_runtime,
        intended_segment,
        input.reanchor_path_kinematics
            ? input.committed_prefix.size()
            : std::numeric_limits<std::size_t>::max(),
        &plan_failure_reason);
    plan_build_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - build_tic).count();

    if (!candidate_plan.valid) {
      dec.plan_failure_reason = plan_failure_reason;
      return false;
    }

    const auto eval_tic = SteadyClock::now();
    dec.monitor = evaluateCandidatePlan(
        candidate_plan,
        input.q,
        input.dq,
        input.current_position,
        input.current_orientation,
        input.ee_twist,
        input.inertia,
        input.J_geo,
        input.coriolis,
        input.control_jdot_dq,
        input.previous_torque_command,
        input.K_runtime,
        input.D_runtime,
        input.human_workspace,
        input.human_workspace_active,
        input.human_workspace_assumed_clear,
        input.last_commanded_sample,
        input.last_commanded_sample_valid,
        input.last_nullspace_stiffness,
        input.overbudget_joint_state,
        (enable_prediction_logging_ || enable_reachable_set_visualization_)
            ? &dec.joint_prediction_trace
            : nullptr);
    monitor_eval_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - eval_tic).count();
    dec.evaluated_plan = candidate_plan;
    dec.has_evaluated_plan = true;

    dec.candidate_verified =
        !shouldRejectCandidateWithMonitor(
            dec.monitor,
            input.human_workspace_active ||
                input.human_workspace_assumed_clear);

    if (!dec.candidate_verified) {
      return true;
    }

    last_verified_plan = candidate_plan;
    last_verified_plan.valid = true;
    last_verified_plan.intended_exec_index = 0;
    last_verified_plan.failsafe_exec_index = 0;

    dec.executing_last_verified_monitored = false;
    dec.command = last_verified_plan.intended.front();

    return true;
  };

  const bool first_attempt = try_one_monitor_pass();

  if (!first_attempt) {
    // Failure to create/evaluate a candidate cannot authorize leaving
    // fallback execution.
    // Report execution of the last verified monitored plan so the real-time
    // side remains conservative until a new candidate is actually verified.
    execute_last_verified_monitored(FallbackReason::kNone);
    stamp_timing();
    return dec;
  }

  if (dec.candidate_verified) {
    stamp_timing();
    return dec;
  }

  execute_last_verified_monitored(
      input.human_workspace_active || input.human_workspace_assumed_clear
          ? FallbackReason::kCandidatePredictionRejected
          : FallbackReason::kHumanWorkspaceUnavailable);
  stamp_timing();
  return dec;
}

bool ReachableCartesianImpedanceController::publishAsyncMonitorInput(
    AsyncMonitorInput input) {
  if (!async_safety_monitor_ || !safety_monitor_worker_running_.load()) {
    return false;
  }

  if (async_input_mutex_.try_lock()) {
    if (async_input_pending_) {
      async_monitor_input_overwrite_count_.fetch_add(
          1, std::memory_order_relaxed);
    }
    latest_async_input_ = std::move(input);
    async_input_pending_ = true;
    async_input_mutex_.unlock();
    async_monitor_input_publish_count_.fetch_add(
        1, std::memory_order_relaxed);
    async_input_cv_.notify_one();
    return true;
  }
  return false;
}

bool ReachableCartesianImpedanceController::takeAsyncMonitorOutput(
    AsyncMonitorOutput* output) {
  const auto result = async_output_mailbox_.takeLatest(output);
  if (result.discarded_older > 0) {
    async_monitor_output_overwrite_count_.fetch_add(
        result.discarded_older, std::memory_order_relaxed);
  }
  if (result.taken) {
    async_monitor_output_consumed_count_.fetch_add(
        1, std::memory_order_relaxed);
  }
  return result.taken;
}
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
    double nullspace_stiffness,
    const Vector7d& overbudget_joint_torque,
    double dt) {
  updateRuntimeGains(cmd.K, cmd.D);

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
    if (kJdotDqMaxNorm > 0.0 && raw_norm > kJdotDqMaxNorm) {
      Jdot_dq_raw *= kJdotDqMaxNorm / std::max(raw_norm, kSmallPositive);
    }

    Jdot_dq_filtered_ =
        kJdotDqFilterAlpha * Jdot_dq_raw +
        (1.0 - kJdotDqFilterAlpha) * Jdot_dq_filtered_;

    Jdot_dq = Jdot_dq_filtered_;
  }

  J_geo_prev_ = J_geo;
  J_geo_prev_valid_ = true;

  Vector7d tau_task;

  Matrix7d M_inv = Matrix7d::Zero();
  Matrix6d lambda = Matrix6d::Zero();
  bool lambda_valid = false;

  if (use_dynamic_consistent_impedance_) {
    const Eigen::LDLT<Matrix7d> inertia_ldlt(inertia);

    if (inertia_ldlt.info() == Eigen::Success) {
      M_inv = inertia_ldlt.solve(Matrix7d::Identity());

      Matrix6d lambda_inv = J_geo * M_inv * J_geo.transpose();
      lambda_inv = 0.5 * (lambda_inv + lambda_inv.transpose());
      lambda_inv.diagonal().array() += kDynamicLambdaRegularization;

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

  const double effective_nullspace_stiffness =
      std::max(0.0, nullspace_stiffness);
  const Vector7d tau_nullspace_raw =
      effective_nullspace_stiffness * (desired_qn_ - q)
      - 2.0 * std::sqrt(effective_nullspace_stiffness) * dq;

  Vector7d tau_nullspace;

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

  // Nullspace control is not switched by execution stage or contact state.
  // Its only runtime adaptation is the common Lachner energy scale already
  // reflected in effective_nullspace_stiffness.
  const Vector7d tau_des =
      tau_task + coriolis + tau_nullspace + overbudget_joint_torque;
  last_tau_task_norm_ = tau_task.norm();
  last_tau_nullspace_raw_norm_ = tau_nullspace_raw.norm();
  last_tau_nullspace_projected_norm_ = tau_nullspace.norm();
  last_coriolis_norm_ = coriolis.norm();
  last_overbudget_joint_torque_norm_ = overbudget_joint_torque.norm();
  last_tau_desired_before_rate_limit_norm_ = tau_des.norm();

  const double max_delta = panda_limits::kTorqueRateLimit * std::max(dt, kMinDt);
  torque_rate_limited_last_ = false;
  torque_rate_max_ratio_last_ = 0.0;

  Vector7d tau_cmd = tau_cmd_prev_;

  for (int i = 0; i < 7; ++i) {
    const double desired_delta = tau_des(i) - tau_cmd_prev_(i);
    const double abs_desired_delta = std::abs(desired_delta);
    torque_rate_max_ratio_last_ =
        std::max(torque_rate_max_ratio_last_,
                 abs_desired_delta / std::max(max_delta, kSmallPositive));
    if (abs_desired_delta > max_delta) {
      torque_rate_limited_last_ = true;
    }

    const double delta = std::clamp(desired_delta, -max_delta, max_delta);

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
  const auto tic_total = SteadyClock::now();
  const std::uint64_t control_loop_sequence = ++control_update_sequence_;
  const bool previous_applied_verified_plan_valid =
      last_commanded_verified_plan_valid_;
  const std::uint64_t previous_applied_verified_plan_generation =
      last_commanded_verified_plan_generation_;
  const int previous_applied_verified_command_stage =
      last_commanded_verified_command_stage_;
  const std::size_t previous_applied_verified_command_index =
      last_commanded_verified_command_index_;
  verified_command_selected_this_cycle_ = false;
  last_verified_command_stage_ = 0;
  last_verified_command_index_ = 0;
  execution_stage_ = ExecutionStage::kCurrentVerified;
  fallback_reason_ = FallbackReason::kNone;

  const double dt = std::max(period.seconds(), kMinDt);
  const double wall_time = (this->get_node()->now() - start_time_).seconds();
  refreshHumanWorkspaceForMonitor(wall_time);

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
  Matrix67d J_collision_geo = J_geo;
  J_collision_geo.topRows<3>() = Jv_collision;

  acceptPendingCartesianViaPoints(
      current_position, current_orientation, wall_time);

  const bool human_workspace_assumed_clear =
      enable_safety_monitor_ &&
      active_cartesian_via_points_calibration_ &&
      calibration_assume_no_human_ &&
      !human_workspace_active_;
  const bool human_workspace_available =
      human_workspace_active_ || human_workspace_assumed_clear;
  if (human_workspace_assumed_clear) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        2000,
        "CALIBRATION ONLY: no fresh human workspace is available; treating "
        "human distance as +infinity while continuing dynamics and energy "
        "prediction.");
  }
  double current_workspace_distance_now =
      human_workspace_assumed_clear
          ? std::numeric_limits<double>::infinity()
          : std::numeric_limits<double>::quiet_NaN();
  int current_robot_link_index = -1;
  if (human_workspace_active_ && robot_reachability_provider_) {
    std::vector<cps_safety_monitor::RobotReachCapsule> current_capsules;
    const std::vector<double> zero_alpha(7, 0.0);
    if (robot_reachability_provider_->reachInterval(
            q, q, 0.0, zero_alpha, &current_capsules)) {
      const auto human_reach =
          human_workspace_.handReachableSetAtTime(wall_time);
      current_workspace_distance_now =
          robot_reachability_provider_->minimumSignedDistance(
              current_capsules,
              human_reach.center,
              human_reach.center,
              human_reach.radius,
              &current_robot_link_index);
    } else {
      // Invalid geometry is treated as contact, never as free space.
      current_workspace_distance_now =
          -std::numeric_limits<double>::infinity();
    }
  } else if (human_workspace_active_) {
    current_workspace_distance_now =
        human_workspace_.signedDistanceToInflatedSphere(
            collision_center,
            human_workspace_.inflatedCollisionRadius(
                ee_collision_radius_, 0.0),
            wall_time);
  }
  const auto toc_model = SteadyClock::now();

  double paused_total = paused_nominal_time_sec_;
  if (failsafe_enter_wall_time_sec_ >= 0.0)
    paused_total += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
  const double nominal_guess_time = std::max(0.0, wall_time - paused_total);

  ShieldDecision shield_dec;
  FallbackReason async_output_rejection_reason = FallbackReason::kNone;

  if (calibration_plan_latched_) {
    // This branch is reachable only through the dedicated calibration action.
    // It executes the already accepted normal-mode plan without replacement;
    // command generation and the runtime impedance controller are unchanged.
    shield_dec = last_shield_decision_;
    shield_dec.candidate_verified = true;
    shield_dec.executing_last_verified_monitored = true;
    shield_dec.fallback_reason = FallbackReason::kNone;
    shield_dec.plan_failure_reason = PlanFailureReason::kNone;

    if (!calibration_plan_complete_ && calibrationExecutionComplete()) {
      calibration_plan_complete_ = true;
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Calibration completed monitored plan generation %lu.",
          static_cast<unsigned long>(calibration_plan_generation_));
    }

    if (calibration_plan_complete_) {
      if (last_commanded_sample_valid_) {
        shield_dec.command = last_commanded_sample_;
        shield_dec.command.t = wall_time;
        shield_dec.command.dp.setZero();
        shield_dec.command.ddp.setZero();
        shield_dec.command.w.setZero();
        shield_dec.command.dw.setZero();
        shield_dec.command.failsafe = true;
      } else {
        shield_dec.command = makeEmergencyStopCommand(
            current_position, current_orientation, wall_time);
      }
    } else {
      shield_dec.command =
          getNextVerifiedTrajectoryCommandFromCache(true);
    }
  } else if (async_safety_monitor_) {
    // Consume a completed result before constructing and publishing the next
    // monitor input. The worker can finish a short rollout while this control
    // cycle is still building its input; taking the previous result first
    // prevents that new result from superseding an otherwise usable output.
    AsyncMonitorOutput async_output;
    bool async_output_available =
        takeAsyncMonitorOutput(&async_output);

    // A calibration action is a one-shot experiment: after arming its first
    // post-action monitor input, do not overwrite that input with newer ones
    // while the worker is still evaluating it.
    const bool calibration_waiting_for_target_output =
        active_cartesian_via_points_calibration_ &&
        !calibration_plan_latched_ &&
        !calibration_target_failed_ &&
        calibration_monitor_input_sequence_ != 0;
    const bool publish_monitor_input =
        control_loop_sequence >= next_async_monitor_control_sequence_ &&
        !calibration_waiting_for_target_output;

    if (publish_monitor_input) {
      AsyncMonitorInput async_input;
      async_input.sequence = async_input_sequence_.fetch_add(1) + 1;
      async_input.control_loop_sequence = control_loop_sequence;
      async_input.source_plan_generation = last_verified_plan_generation_;
      async_input.scheduled_control_loop_sequence =
          next_async_monitor_control_sequence_;
      async_input.publish_lateness_cycles =
          control_loop_sequence - next_async_monitor_control_sequence_;
      async_input.wall_time = wall_time;
      async_input.nominal_guess_time = nominal_guess_time;
      async_input.q = q;
      async_input.dq = dq;
      async_input.current_position = current_position;
      async_input.current_orientation = current_orientation;
      async_input.ee_twist = ee_twist;
      async_input.inertia = inertia;
      async_input.coriolis = coriolis;
      // This filtered finite-difference value comes from the live 1 kHz
      // Jacobian stream. Franka does not publish Jdot*dq directly.
      async_input.control_jdot_dq = Jdot_dq_filtered_;
      async_input.Jv = Jv_collision;
      async_input.J_geo = J_collision_geo;
      async_input.previous_torque_command = tau_cmd_prev_;
      async_input.K_runtime = K_runtime_;
      async_input.D_runtime = D_runtime_;
      async_input.human_workspace = human_workspace_;
      async_input.human_workspace_active = human_workspace_active_;
      async_input.human_workspace_assumed_clear =
          human_workspace_assumed_clear;
      async_input.last_commanded_sample = last_commanded_sample_;
      async_input.last_commanded_sample_valid = last_commanded_sample_valid_;
      async_input.last_nullspace_stiffness = last_nullspace_stiffness_;
      async_input.overbudget_joint_state = overbudget_joint_state_;
      async_input.commanded_path_time = commanded_path_time_;
      async_input.commanded_path_rate = commanded_path_rate_;
      // The nominal generator always requests the configured path rate.
      // Inside the collision area, energy adaptation changes impedance gains
      // and, for T > L_max, adds Eq. (16)-(17) joint stabilization. It never
      // freezes effective time.
      async_input.target_path_rate = path_time_rate_target_;
      async_input.reanchor_path_kinematics = false;
      async_input.reanchor_path_rate = 0.0;
      async_input.reanchor_path_acceleration = 0.0;
      async_input.committed_prefix.reserve(async_planning_lead_steps_);
      // While intended commands remain, do not extend a candidate's committed
      // prefix into the fail-safe merely to reach the configured lead length.
      // The worker result already carries an activation deadline equal to the
      // actual prefix length.  A shorter intended-only prefix therefore means
      // "finish verification before this intended tail is consumed".  Mixing
      // the high-deceleration fail-safe tail with a new nominal replan made the
      // C2 seam invalid and forced the robot to brake all the way to rest
      // before a candidate could be accepted.
      const bool commit_only_remaining_intended =
          last_verified_plan_.valid &&
          last_verified_plan_.intended_exec_index <
              last_verified_plan_.intended.size();
      for (std::size_t offset = 0;
           offset < async_planning_lead_steps_;
           ++offset) {
        ImpedanceSample committed_command;
        bool command_available = false;
        if (commit_only_remaining_intended) {
          const std::size_t intended_index =
              last_verified_plan_.intended_exec_index + offset;
          if (intended_index >= last_verified_plan_.intended.size()) {
            break;
          }
          committed_command = last_verified_plan_.intended[intended_index];
          command_available = true;
        } else {
          command_available = getVerifiedTrajectoryCommandAtOffset(
              last_verified_plan_, offset, &committed_command);
        }
        if (!command_available && last_commanded_sample_valid_) {
          committed_command = last_commanded_sample_;
          committed_command.dp.setZero();
          committed_command.ddp.setZero();
          committed_command.w.setZero();
          committed_command.dw.setZero();
          if (committed_command.nominal_path_time_valid) {
            committed_command.nominal_path_rate = 0.0;
            committed_command.nominal_path_acceleration = 0.0;
            committed_command.nominal_path_kinematics_valid = true;
          }
          committed_command.failsafe = true;
          command_available = true;
        }
        if (!command_available) {
          break;
        }
        async_input.committed_prefix.push_back(committed_command);
      }
      if (commit_only_remaining_intended) {
        async_input.nominal_advance_steps =
            static_cast<std::size_t>(std::count_if(
                async_input.committed_prefix.begin(),
                async_input.committed_prefix.end(),
                [](const ImpedanceSample& command) {
                  return !command.failsafe;
                }));
      }
      // Continue from the explicit scalar state of the committed command.
      // Cartesian derivatives cannot recover path rate at a legitimate cusp:
      // both dp and w are zero while nominal path time must continue through
      // the direction reversal. The finite-difference branch is retained only
      // for plans created before explicit scalar metadata was available.
      if (!async_input.reanchor_path_kinematics &&
          !async_input.committed_prefix.empty()) {
        const ImpedanceSample& committed_end =
            async_input.committed_prefix.back();
        if (committed_end.nominal_path_time_valid &&
            committed_end.nominal_path_kinematics_valid) {
          async_input.reanchor_path_kinematics = true;
          async_input.reanchor_path_rate = std::clamp(
              committed_end.nominal_path_rate,
              path_time_rate_min_,
              path_time_rate_max_);
          async_input.reanchor_path_acceleration = std::clamp(
              committed_end.nominal_path_acceleration,
              -path_time_acc_limit_,
              path_time_acc_limit_);
        } else if (committed_end.nominal_path_time_valid) {
          async_input.reanchor_path_kinematics = true;
          const double prefix_dt = std::max(local_replan_dt_, kMinDt);
          const std::size_t prefix_size =
              async_input.committed_prefix.size();
          if (prefix_size >= 2 &&
              async_input.committed_prefix[prefix_size - 2]
                  .nominal_path_time_valid) {
            const double end_rate =
                (committed_end.nominal_path_time -
                 async_input.committed_prefix[prefix_size - 2]
                     .nominal_path_time) /
                prefix_dt;
            async_input.reanchor_path_rate = std::clamp(
                end_rate, path_time_rate_min_, path_time_rate_max_);
            if (prefix_size >= 3 &&
                async_input.committed_prefix[prefix_size - 3]
                    .nominal_path_time_valid) {
              const double previous_rate =
                  (async_input.committed_prefix[prefix_size - 2]
                       .nominal_path_time -
                   async_input.committed_prefix[prefix_size - 3]
                       .nominal_path_time) /
                  prefix_dt;
              async_input.reanchor_path_acceleration = std::clamp(
                  (end_rate - previous_rate) / prefix_dt,
                  -path_time_acc_limit_,
                  path_time_acc_limit_);
            }
          } else {
            async_input.reanchor_path_rate =
                estimatePathRateFromTimedPathSample(
                    committed_end.nominal_path_time,
                    committed_end.dp);
          }
        }
      }
      // A transient mutex collision must not silently consume this 200 Hz
      // monitor slot. Leave the absolute sequence deadline unchanged so the
      // 1 kHz loop retries with a newer state snapshot on its next cycle.
      async_input.publish_steady_time_ns = steadyNowNanoseconds();
      const std::uint64_t published_monitor_input_sequence =
          async_input.sequence;
      const double published_commanded_path_time =
          commanded_path_time_;
      if (publishAsyncMonitorInput(std::move(async_input))) {
        last_async_input_publish_control_sequence_ = control_loop_sequence;
        if (active_cartesian_via_points_calibration_ &&
            !calibration_plan_latched_ &&
            !calibration_target_failed_ &&
            calibration_monitor_input_sequence_ == 0 &&
            published_commanded_path_time >=
                calibration_requested_capture_path_time_sec_) {
          calibration_monitor_input_sequence_ =
              published_monitor_input_sequence;
          calibration_actual_capture_path_time_sec_ =
              published_commanded_path_time;
          RCLCPP_INFO(
              get_node()->get_logger(),
              "Calibration armed next monitored trajectory at "
              "requested_path_time=%.6f actual_path_time=%.6f "
              "monitor_input=%lu control_sequence=%lu.",
              calibration_requested_capture_path_time_sec_,
              calibration_actual_capture_path_time_sec_,
              static_cast<unsigned long>(
                  calibration_monitor_input_sequence_),
              static_cast<unsigned long>(control_loop_sequence));
        }
        async_monitor_schedule_late_cycles_ +=
            control_loop_sequence - next_async_monitor_control_sequence_;
        const std::uint64_t period_cycles =
            std::max<std::uint64_t>(1, monitor_period_control_cycles_);
        next_async_monitor_control_sequence_ += period_cycles;
        while (next_async_monitor_control_sequence_ <=
               control_loop_sequence) {
          next_async_monitor_control_sequence_ += period_cycles;
          ++async_monitor_schedule_skipped_slots_;
        }
      }
    }

    // If no result was ready at the beginning of this cycle, make one more
    // non-blocking read after input publication. This catches a worker result
    // that completed concurrently while the controller assembled the input,
    // without ever replacing a result already selected for this cycle.
    if (!async_output_available) {
      async_output_available = takeAsyncMonitorOutput(&async_output);
    }

    if (async_output_available) {
      const std::int64_t output_take_steady_time_ns =
          steadyNowNanoseconds();
      AsyncMonitorTiming async_timing;
      async_timing.valid = true;
      async_timing.input_sequence = async_output.input.sequence;
      async_timing.input_control_loop_sequence =
          async_output.input.control_loop_sequence;
      async_timing.source_plan_generation =
          async_output.input.source_plan_generation;
      async_timing.committed_prefix_steps =
          async_output.input.committed_prefix.size();
      async_timing.scheduled_control_loop_sequence =
          async_output.input.scheduled_control_loop_sequence;
      async_timing.publish_lateness_cycles =
          async_output.input.publish_lateness_cycles;
      async_timing.worker_queue_wait_ms =
          async_output.worker_queue_wait_ms;
      async_timing.worker_compute_ms =
          async_output.worker_compute_ms;
      async_timing.output_handoff_ms = nanosecondsToMilliseconds(
          std::max<std::int64_t>(
              0,
              output_take_steady_time_ns -
                  async_output.worker_finish_steady_time_ns));
      async_timing.end_to_end_ms = nanosecondsToMilliseconds(
          std::max<std::int64_t>(
              0,
              output_take_steady_time_ns -
                  async_output.input.publish_steady_time_ns));
      const std::size_t async_plan_elapsed_steps =
          control_loop_sequence >= async_output.input.control_loop_sequence
              ? static_cast<std::size_t>(
                    control_loop_sequence -
                    async_output.input.control_loop_sequence)
              : 0;
      const bool async_output_matches_source_plan =
          async_output.input.source_plan_generation ==
          last_verified_plan_generation_;
      const bool async_output_before_activation =
          !async_output.input.committed_prefix.empty() &&
          async_plan_elapsed_steps <=
              async_output.input.committed_prefix.size();

      const bool calibration_target_armed =
          active_cartesian_via_points_calibration_ &&
          !calibration_plan_latched_ &&
          !calibration_target_failed_ &&
          calibration_monitor_input_sequence_ != 0;
      const bool calibration_target_output =
          calibration_target_armed &&
          async_output.input.sequence ==
              calibration_monitor_input_sequence_;
      const bool async_output_matches_calibration_target =
          !calibration_target_armed || calibration_target_output;

      // A scheduling spike can make a fully verified worker result arrive
      // after its committed prefix while it still contains a fresh intended
      // tail. Allow such a result to catch up only when its next command is a
      // valid one-step transition from the command actually sent by the
      // real-time loop. This preserves path and derivative continuity without
      // treating a missed activation deadline as an unsafe trajectory.
      bool async_output_late_catchup_continuous = false;
      const VerifiedPlan* late_catchup_plan = nullptr;
      if (async_output.decision.candidate_verified &&
          async_output.decision.evaluated_plan.valid) {
        late_catchup_plan = &async_output.decision.evaluated_plan;
      }
      if (!async_output_before_activation &&
          async_output.decision.candidate_verified &&
          late_catchup_plan != nullptr &&
          last_commanded_sample_valid_ &&
          async_plan_elapsed_steps <
              late_catchup_plan->intended.size()) {
        const ImpedanceSample& next =
            late_catchup_plan->intended[async_plan_elapsed_steps];
        async_output_late_catchup_continuous =
            isOneStepCommandTransitionContinuous(
                last_commanded_sample_,
                next,
                false);
      }
      // Never carry an assume-clear calibration result into normal mode, or
      // reuse a live-workspace result after the provider has become stale.
      const bool async_output_workspace_policy_matches =
          (async_output.input.human_workspace_assumed_clear &&
           human_workspace_assumed_clear) ||
          (async_output.input.human_workspace_active &&
           human_workspace_active_);
      const bool async_output_usable =
          async_output_matches_source_plan &&
          async_output_workspace_policy_matches &&
          async_output_matches_calibration_target &&
          (async_output_before_activation ||
           async_output_late_catchup_continuous);
      async_timing.source_plan_matches_at_handoff =
          async_output_matches_source_plan;
      async_timing.output_usable = async_output_usable;
      last_async_monitor_timing_ = async_timing;
      if (!async_output_matches_source_plan) {
        async_output_rejection_reason =
            FallbackReason::kAsyncOutputUnavailable;
      } else if (!async_output_workspace_policy_matches) {
        async_output_rejection_reason =
            FallbackReason::kAsyncOutputUnavailable;
      } else if (!async_output_matches_calibration_target) {
        async_output_rejection_reason =
            FallbackReason::kAsyncOutputUnavailable;
      } else if (!async_output_before_activation &&
                 !async_output_late_catchup_continuous) {
        async_output_rejection_reason =
            FallbackReason::kAsyncOutputUnavailable;
        ++async_activation_deadline_miss_count_;
      } else if (async_output_late_catchup_continuous) {
        ++async_late_activation_accept_count_;
      }
      const bool verified_output_acceptable_now =
          human_workspace_available &&
          async_output.decision.candidate_verified;
      const std::uint64_t accepted_plan_generation =
          async_output_usable &&
                  verified_output_acceptable_now &&
                  async_output.decision.evaluated_plan.valid
              ? last_verified_plan_generation_ + 1
              : 0;
      if (async_output.decision.has_evaluated_plan) {
        logShieldPredictionTrajectory(
            async_output.input.wall_time,
            async_output.input.nominal_guess_time,
            async_output.input.q,
            async_output.input.dq,
            async_output.input.current_position,
            async_output.input.current_orientation,
            async_output.input.ee_twist,
            async_output.input.inertia,
            async_output.input.Jv,
            async_output.input.K_runtime,
            async_output.input.D_runtime,
            async_output.input.human_workspace,
            async_output.input.human_workspace_active,
            async_output.input.human_workspace_assumed_clear,
            async_output.decision.evaluated_plan,
            async_output.decision.joint_prediction_trace,
            async_timing,
            async_output.decision.monitor,
            executionModeForLog(
                async_output.decision.executing_last_verified_monitored ||
                async_output.decision.command.failsafe),
            async_output.decision.candidate_verified,
            accepted_plan_generation,
            async_output.decision.executing_last_verified_monitored,
            async_output.decision.monitor_total_ms,
            async_output.decision.planner_ms,
            async_output.decision.plan_build_ms,
            async_output.decision.monitor_eval_ms,
            "async");
      }
      if (async_output_usable) {
        last_shield_decision_ = async_output.decision;
        last_shield_decision_valid_ = true;
        last_async_output_wall_time_ = async_output.input_wall_time;
        last_async_output_valid_ = true;
        if (verified_output_acceptable_now &&
            async_output.decision.evaluated_plan.valid) {
          last_verified_plan_ = async_output.decision.evaluated_plan;
          last_verified_plan_.valid = true;
          alignVerifiedPlanExecutionIndex(
              &last_verified_plan_,
              async_plan_elapsed_steps);
          ++last_verified_plan_generation_;
          if (active_cartesian_via_points_calibration_ &&
              !calibration_plan_latched_ &&
              async_output.input.sequence ==
                  calibration_monitor_input_sequence_ &&
              last_verified_plan_.intended_exec_index <
                  last_verified_plan_.intended.size() &&
              !last_verified_plan_.failsafe.empty()) {
            calibration_plan_latched_ = true;
            calibration_plan_complete_ = false;
            calibration_failsafe_command_count_ =
                failsafeCommandCount(last_verified_plan_);
            calibration_plan_generation_ =
                last_verified_plan_generation_;
            calibration_activation_control_sequence_ =
                control_loop_sequence;
            calibration_activation_intended_index_ =
                last_verified_plan_.intended_exec_index;
            calibration_activation_failsafe_index_ =
                last_verified_plan_.failsafe_exec_index;
            RCLCPP_INFO(
                get_node()->get_logger(),
                "Calibration captured next monitored plan generation=%lu "
                "monitor_input=%lu activation_control=%lu "
                "intended_index=%zu/%zu failsafe_index=%zu/%zu.",
                static_cast<unsigned long>(calibration_plan_generation_),
                static_cast<unsigned long>(
                    calibration_monitor_input_sequence_),
                static_cast<unsigned long>(
                    calibration_activation_control_sequence_),
                calibration_activation_intended_index_,
                last_verified_plan_.intended.size(),
                calibration_activation_failsafe_index_,
                calibration_failsafe_command_count_);
          }
        }
      }
      if (calibration_target_output && !calibration_plan_latched_) {
        calibration_target_failed_ = true;
        RCLCPP_WARN(
            get_node()->get_logger(),
            "Calibration target monitor_input=%lu was not executable "
            "(output_usable=%d candidate_verified=%d plan_valid=%d); no "
            "later candidate will be substituted.",
            static_cast<unsigned long>(
                calibration_monitor_input_sequence_),
            static_cast<int>(async_output_usable),
            static_cast<int>(async_output.decision.candidate_verified),
            static_cast<int>(
                async_output.decision.evaluated_plan.valid));
      }
    }

    const bool output_is_fresh =
        last_async_output_valid_ &&
        last_shield_decision_valid_ &&
        (async_plan_max_age_sec_ <= 0.0 ||
         (wall_time - last_async_output_wall_time_) <= async_plan_max_age_sec_);

    if (output_is_fresh) {
      shield_dec = last_shield_decision_;
      if (shield_dec.executing_last_verified_monitored ||
          shouldRejectCandidateWithMonitor(
              shield_dec.monitor, human_workspace_available)) {
        shield_dec.executing_last_verified_monitored = true;
        if (!human_workspace_available) {
          shield_dec.fallback_reason =
              FallbackReason::kHumanWorkspaceUnavailable;
        }
        if (last_verified_plan_.valid &&
            (!last_verified_plan_.intended.empty() ||
             !last_verified_plan_.failsafe.empty())) {
          shield_dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
        } else {
          shield_dec.command =
              makeEmergencyStopCommand(current_position, current_orientation, wall_time);
        }
      } else {
        shield_dec.executing_last_verified_monitored = false;
        if (last_verified_plan_.valid && !last_verified_plan_.intended.empty()) {
          shield_dec.command =
              getNextVerifiedTrajectoryCommandFromCache(true);
          shield_dec.executing_last_verified_monitored =
              shield_dec.command.failsafe;
        } else {
          shield_dec.command =
              makeEmergencyStopCommand(current_position, current_orientation, wall_time);
        }
      }
    } else {
      const bool need_sync_bootstrap =
          !last_verified_plan_.valid || last_verified_plan_.intended.empty();

      if (need_sync_bootstrap) {
        // The planner/monitor is intentionally never run in the 1 kHz update
        // during asynchronous operation. Until the first verified result is
        // available, keep the last command as a stationary bootstrap hold.
        shield_dec.executing_last_verified_monitored = true;
        shield_dec.fallback_reason =
            FallbackReason::kNoVerifiedPlanAvailable;
        if (last_commanded_sample_valid_) {
          shield_dec.command = last_commanded_sample_;
          shield_dec.command.t = wall_time;
          shield_dec.command.dp.setZero();
          shield_dec.command.ddp.setZero();
          shield_dec.command.w.setZero();
          shield_dec.command.dw.setZero();
          shield_dec.command.failsafe = true;
        } else {
          shield_dec.command = makeEmergencyStopCommand(
              current_position, current_orientation, wall_time);
        }
      } else {
        shield_dec = last_shield_decision_;
        shield_dec.fallback_reason =
            !human_workspace_available
                ? FallbackReason::kHumanWorkspaceUnavailable
                : async_output_rejection_reason != FallbackReason::kNone
                ? async_output_rejection_reason
                : FallbackReason::kAsyncOutputUnavailable;
        if (isLastVerifiedSafetyMode(mode_) ||
            shield_dec.executing_last_verified_monitored ||
            shouldRejectCandidateWithMonitor(
                shield_dec.monitor, human_workspace_available)) {
          shield_dec.executing_last_verified_monitored = true;
          if (last_verified_plan_.valid &&
              (!last_verified_plan_.intended.empty() ||
               !last_verified_plan_.failsafe.empty())) {
            shield_dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
          } else {
            shield_dec.command =
                makeEmergencyStopCommand(current_position, current_orientation, wall_time);
          }
        } else {
          shield_dec.executing_last_verified_monitored = false;
          shield_dec.command =
              getNextVerifiedTrajectoryCommandFromCache(true);
          shield_dec.executing_last_verified_monitored =
              shield_dec.command.failsafe;
        }
        last_shield_decision_.command = shield_dec.command;
        last_shield_decision_.executing_last_verified_monitored =
            shield_dec.executing_last_verified_monitored;
      }
    }
  } else {
    const bool do_monitor =
        !last_shield_decision_valid_ ||
        ((monitor_counter_++ % std::max(1, monitor_decimation_)) == 0);

    if (do_monitor) {
      const std::uint64_t generation_before_monitor =
          last_verified_plan_generation_;
      shield_dec = computeShieldDecision(wall_time, nominal_guess_time,
                                         q, dq,
                                         current_position, current_orientation,
                                         ee_twist, inertia, J_collision_geo,
                                         coriolis, Jdot_dq_filtered_);
      last_shield_decision_ = shield_dec;
      last_shield_decision_valid_ = true;
      const std::uint64_t accepted_plan_generation =
          shield_dec.candidate_verified &&
                  last_verified_plan_generation_ > generation_before_monitor
              ? last_verified_plan_generation_
              : 0;
      if (shield_dec.has_evaluated_plan) {
        const AsyncMonitorTiming sync_timing;
        logShieldPredictionTrajectory(
            wall_time,
            nominal_guess_time,
            q,
            dq,
            current_position,
            current_orientation,
            ee_twist,
            inertia,
            Jv_collision,
            K_runtime_,
            D_runtime_,
            human_workspace_,
            human_workspace_active_,
            false,
            shield_dec.evaluated_plan,
            shield_dec.joint_prediction_trace,
            sync_timing,
            shield_dec.monitor,
            executionModeForLog(
                shield_dec.executing_last_verified_monitored ||
                shield_dec.command.failsafe),
            shield_dec.candidate_verified,
            accepted_plan_generation,
            shield_dec.executing_last_verified_monitored,
            shield_dec.monitor_total_ms,
            shield_dec.planner_ms,
            shield_dec.plan_build_ms,
            shield_dec.monitor_eval_ms,
            "sync");
      }
    } else {
      shield_dec = last_shield_decision_;
      if (isLastVerifiedSafetyMode(mode_) ||
          shield_dec.executing_last_verified_monitored) {
        shield_dec.executing_last_verified_monitored = true;
        if (last_verified_plan_.valid &&
            (!last_verified_plan_.intended.empty() ||
             !last_verified_plan_.failsafe.empty()))
          shield_dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
        else
          shield_dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
      } else {
        shield_dec.executing_last_verified_monitored = false;
        if (last_verified_plan_.valid && !last_verified_plan_.intended.empty())
          shield_dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
        else
          shield_dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
      }
      last_shield_decision_.command = shield_dec.command;
      last_shield_decision_.executing_last_verified_monitored =
          shield_dec.executing_last_verified_monitored;
    }
  }
  const auto toc_shield = SteadyClock::now();

  MonitorResult monitor = shield_dec.monitor;
  const bool verified_failsafe_available =
      last_verified_plan_.valid && !last_verified_plan_.failsafe.empty();
  execution_stage_ =
      shield_dec.command.failsafe
          ? (verified_failsafe_available
                 ? ExecutionStage::kLastVerifiedFailsafe
                 : ExecutionStage::kHold)
          : (shield_dec.executing_last_verified_monitored
                 ? ExecutionStage::kLastVerifiedIntended
                 : ExecutionStage::kCurrentVerified);
  const bool fallback_execution =
      execution_stage_ != ExecutionStage::kCurrentVerified;
  if (fallback_execution &&
      shield_dec.fallback_reason == FallbackReason::kNone &&
      shield_dec.plan_failure_reason == PlanFailureReason::kNone &&
      execution_stage_ == ExecutionStage::kHold) {
    shield_dec.fallback_reason =
        FallbackReason::kNoVerifiedPlanAvailable;
  }
  fallback_reason_ =
      fallback_execution
          ? shield_dec.fallback_reason
          : FallbackReason::kNone;
  plan_failure_reason_ =
      fallback_execution && fallback_reason_ == FallbackReason::kNone
          ? shield_dec.plan_failure_reason
          : PlanFailureReason::kNone;

  if (shield_dec.executing_last_verified_monitored) {
    if (failsafe_enter_wall_time_sec_ < 0.0) {
      failsafe_enter_wall_time_sec_ = wall_time;
    }
  } else {
    if (failsafe_enter_wall_time_sec_ >= 0.0) {
      paused_nominal_time_sec_ += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
      failsafe_enter_wall_time_sec_ = -1.0;
    }
  }

  const std::size_t monitored_intended_steps =
      (shield_dec.has_evaluated_plan && shield_dec.evaluated_plan.valid)
          ? shield_dec.evaluated_plan.intended.size()
          : 0;
  const std::size_t monitored_failsafe_steps =
      (shield_dec.has_evaluated_plan && shield_dec.evaluated_plan.valid)
          ? shield_dec.evaluated_plan.failsafe.size()
          : 0;
  const std::size_t monitored_steps =
      monitored_intended_steps + monitored_failsafe_steps;
  const bool monitor_prediction_valid =
      shield_dec.has_evaluated_plan && shield_dec.evaluated_plan.valid &&
      monitored_steps > 0;
  // A failed or delayed rollout is not evidence that the previously predicted
  // collision possibility disappeared. Keep the last prediction classification
  // until a complete monitored trajectory replaces it.
  const bool predicted_contact_possible =
      monitor_prediction_valid
          ? monitor.monitored_contact_possible
          : isCollisionPossibleMode(mode_);
  // Prediction classifies and verifies contact geometry. Runtime Eq. (14)
  // uses the current overlap as its activation gate.
  monitor.workspace_distance_now = current_workspace_distance_now;
  monitor.current_robot_link_index = current_robot_link_index;
  monitor.robot_secure_radius = robot_reachability_provider_
      ? robot_reachability_provider_->secureRadius()
      : 0.0;
  if (!monitor_prediction_valid) {
    monitor.workspace_distance_min = current_workspace_distance_now;
  }
  monitor.monitored_contact_possible = predicted_contact_possible;
  monitor.contact_relevant_for_energy =
      current_workspace_distance_now <= 0.0;
  mode_ =
      shield_dec.executing_last_verified_monitored ||
              shield_dec.command.failsafe
          ? lastVerifiedSafetyModeForMonitor(monitor)
          : nominalSafetyModeForMonitor(monitor);

  EnergyBudgetInfo energy_info;
  const bool energy_budget_active =
      shouldApplyEnergyBudget(monitor);
  const bool apply_runtime_energy_scaling =
      enable_runtime_energy_scaling_ && energy_budget_active;
  // Lachner Eq. (12) is evaluated during the complete controller lifecycle.
  // Outside the collision area this is diagnostic/predictive measurement only:
  // it cannot change gains, add torque, or alter trajectory timing.
  const bool track_control_energy =
      enable_safety_monitor_;

  struct ControlEnergyTerms {
    bool valid{false};
    double kinetic_energy{0.0};
    double potential_energy{0.0};
    double nullspace_potential_energy{0.0};
  };

  Matrix6d budget_cartesian_task_inertia = Matrix6d::Zero();
  bool budget_cartesian_task_inertia_valid = false;
  const bool maintain_cartesian_energy_cache =
      enable_safety_monitor_;
  if (maintain_cartesian_energy_cache) {
    const double cache_period =
        std::max(0.0, cartesian_energy_lambda_update_period_sec_);
    const bool refresh_every_update =
        cache_period <= std::max(dt, kMinDt) + kMinDt;
    const bool cache_due =
        !cartesian_energy_task_inertia_cache_valid_ ||
        cartesian_energy_task_inertia_cache_wall_time_ < 0.0 ||
        wall_time < cartesian_energy_task_inertia_cache_wall_time_ ||
        refresh_every_update ||
        (wall_time - cartesian_energy_task_inertia_cache_wall_time_) >=
            cache_period;

    if (cache_due) {
      Matrix6d task_inertia = Matrix6d::Zero();
      if (computeTaskInertia(inertia, J_geo, &task_inertia)) {
        cartesian_energy_task_inertia_cache_ = task_inertia;
        cartesian_energy_task_inertia_cache_valid_ = true;
        cartesian_energy_task_inertia_cache_wall_time_ = wall_time;
      } else {
        cartesian_energy_task_inertia_cache_.setZero();
        cartesian_energy_task_inertia_cache_valid_ = false;
        cartesian_energy_task_inertia_cache_wall_time_ = -1.0;
      }
    }

    if (cartesian_energy_task_inertia_cache_valid_) {
      budget_cartesian_task_inertia =
          cartesian_energy_task_inertia_cache_;
      budget_cartesian_task_inertia_valid = true;
    }
  } else {
    cartesian_energy_task_inertia_cache_.setZero();
    cartesian_energy_task_inertia_cache_valid_ = false;
    cartesian_energy_task_inertia_cache_wall_time_ = -1.0;
  }

  auto compute_error_for_command = [&](const ImpedanceSample& command) {
    Vector6d command_error = Vector6d::Zero();
    command_error.head<3>() = current_position - command.p;
    command_error.tail<3>() =
        computeOrientationError(current_orientation, command.q);
    return command_error;
  };

  auto compute_control_energy_terms = [&](const ImpedanceSample& command,
                                          double nullspace_stiffness) {
    ControlEnergyTerms terms;
    if (!track_control_energy || !inertia.allFinite() ||
        !dq.allFinite()) {
      return terms;
    }

    terms.valid = true;

    // Runtime passivity storage uses the measured current state. The tracking
    // error tube is reserved for predictive rollout, not for the instantaneous
    // stored-energy calculation.
    // Lachner et al. Eq. (12): kinetic storage is the complete robot kinetic
    // energy.  This retains redundancy/nullspace motion that is invisible in
    // a Cartesian projection Lambda.
    const double kinetic_energy =
        0.5 * (dq.transpose() * inertia * dq)(0, 0);
    terms.kinetic_energy =
        std::max(0.0, kinetic_energy);

    // Use the same six-dimensional pose error as the impedance feedback, so
    // the stored Cartesian potential includes orientation as well as position.
    const Vector6d command_error = compute_error_for_command(command);
    Matrix6d stiffness_sqrt = Matrix6d::Zero();
    Matrix6d stiffness_storage =
        0.5 * (command.K + command.K.transpose());
    if (symmetricPositiveSemidefiniteSquareRoot(
            stiffness_storage, &stiffness_sqrt)) {
      stiffness_storage = stiffness_sqrt * stiffness_sqrt;
      stiffness_storage =
          0.5 * (stiffness_storage + stiffness_storage.transpose());
    }
    terms.potential_energy =
        std::max(
            0.0,
            0.5 * (command_error.transpose() *
                   stiffness_storage * command_error)(0, 0));
    if (nullspace_stiffness > 0.0) {
      // Lachner et al. Eqs. (5) and (12), with K_q = k_n I.
      const Vector7d nullspace_error = desired_qn_ - q;
      terms.nullspace_potential_energy =
          0.5 * nullspace_stiffness * nullspace_error.squaredNorm();
    }
    return terms;
  };

  // At the beginning of this cycle the measured state is the result of the
  // command sent in the previous cycle. Keep this same-command energy pair for
  // exact comparison with a predicted rollout endpoint at the corresponding
  // control-loop sequence. The command selected below belongs to the next
  // interval and would otherwise make potential-energy matching off by one.
  const double previous_applied_nullspace_stiffness =
      last_nullspace_stiffness_;
  const ControlEnergyTerms previous_applied_energy_terms =
      last_commanded_sample_valid_
          ? compute_control_energy_terms(
                last_commanded_sample_,
                previous_applied_nullspace_stiffness)
          : ControlEnergyTerms{};

  const ImpedanceSample candidate_budget_command = shield_dec.command;
  const Vector6d error =
      compute_error_for_command(candidate_budget_command);
  const ControlEnergyTerms energy_budget_terms =
      compute_control_energy_terms(
          candidate_budget_command,
          n_stiffness_);
  // Keep the energy-budget stiffness and damping adaptation, but do not
  // freeze or retime the verified trajectory.  Path progress therefore stays
  // governed exclusively by the verified command stream.
  shield_dec.command = applyEnergyBudget(
      candidate_budget_command,
      energy_budget_terms.kinetic_energy,
      energy_budget_terms.potential_energy,
      energy_budget_terms.nullspace_potential_energy,
      energy_budget_terms.valid,
      apply_runtime_energy_scaling,
      &energy_info);
  energy_info.lambda_valid =
      budget_cartesian_task_inertia_valid;

  // Runtime "current" energy always describes the command actually applied
  // in this loop, after stiffness scaling. The pre-scaling candidate energy is
  // retained separately in EnergyBudgetInfo for diagnostics.
  if (track_control_energy) {
    monitor.current_joint_energy_valid = energy_budget_terms.valid;
    monitor.current_joint_kinetic_energy =
        energy_info.kinetic_energy;
    monitor.current_cartesian_potential_energy =
        energy_info.potential_energy;
    monitor.current_nullspace_potential_energy =
        energy_info.nullspace_potential_energy;
    monitor.current_total_control_energy =
        energy_info.total_energy;
    if (budget_cartesian_task_inertia_valid) {
      monitor.current_cartesian_kinetic_energy = std::max(
          0.0,
          0.5 * (ee_twist.transpose() *
                 budget_cartesian_task_inertia * ee_twist)(0, 0));
      monitor.current_cartesian_control_energy =
          monitor.current_cartesian_kinetic_energy +
          monitor.current_cartesian_potential_energy;
      monitor.current_cartesian_energy_valid = true;
    }
  }

  // Advance the nominal path state only from the command that is really sent
  // this cycle.  Accepting an asynchronous plan must not jump path progress to
  // that plan's future endpoint (SaRA's verified path is advanced one executed
  // sample at a time for the same reason).
  if (shield_dec.command.nominal_path_time_valid) {
    const double next_path_time =
        std::max(commanded_path_time_,
                 shield_dec.command.nominal_path_time);
    if (shield_dec.command.nominal_path_kinematics_valid) {
      commanded_path_rate_ = std::clamp(
          shield_dec.command.nominal_path_rate,
          path_time_rate_min_,
          path_time_rate_max_);
    } else {
      commanded_path_rate_ = std::clamp(
          (next_path_time - commanded_path_time_) / std::max(dt, kMinDt),
          path_time_rate_min_,
          path_time_rate_max_);
    }
    commanded_path_time_ = next_path_time;
  } else if (shield_dec.command.failsafe) {
    commanded_path_rate_ = 0.0;
  }

  last_energy_budget_active_ = energy_info.active;
  last_energy_budget_lambda_valid_ = energy_info.lambda_valid;
  last_energy_stiffness_scale_ = energy_info.scale;
  last_nullspace_stiffness_ = energy_info.nullspace_stiffness;
  last_joint_kinetic_energy_ = energy_info.kinetic_energy;
  last_cartesian_potential_energy_before_scaling_ =
      energy_info.potential_energy_before_scaling;
  last_nullspace_potential_energy_before_scaling_ =
      energy_info.nullspace_potential_energy_before_scaling;
  last_nullspace_potential_energy_ =
      energy_info.nullspace_potential_energy;
  last_total_control_energy_before_scaling_ =
      energy_info.total_energy_before_scaling;
  last_cartesian_potential_energy_ = energy_info.potential_energy;
  last_total_control_energy_ = energy_info.total_energy;

  const cps_safety_monitor::OverbudgetJointStabilizationTerms
      overbudget_joint_terms =
          cps_safety_monitor::updateOverbudgetJointStabilization(
              q,
              energy_info.kinetic_energy,
              energy_budget_joule_,
              overbudget_joint_stiffness_,
              overbudget_joint_scale_omega_,
              enable_safety_monitor_ &&
                  enable_overbudget_joint_stabilization_ &&
                  energy_budget_active,
              &overbudget_joint_state_);
  last_overbudget_joint_potential_energy_ =
      overbudget_joint_terms.potential_energy;
  last_overbudget_joint_scale_rho_ =
      overbudget_joint_terms.scale_rho;
  last_overbudget_joint_torque_norm_ =
      overbudget_joint_terms.torque.norm();

  const Vector3d desired_position_cur = shield_dec.command.p;
  const Vector3d desired_linear_velocity_cur = shield_dec.command.dp;

  last_commanded_verified_plan_valid_ =
      verified_command_selected_this_cycle_;
  last_commanded_verified_plan_generation_ =
      verified_command_selected_this_cycle_
          ? last_verified_plan_generation_
          : 0;
  last_commanded_verified_command_stage_ =
      verified_command_selected_this_cycle_
          ? last_verified_command_stage_
          : 0;
  last_commanded_verified_command_index_ =
      verified_command_selected_this_cycle_
          ? last_verified_command_index_
          : 0;
  last_commanded_sample_ = shield_dec.command;
  last_commanded_sample_valid_ = true;

  const Vector7d tau_cmd = computeImpedanceTorque(
      q, dq, inertia, coriolis, J_geo,
      current_position, current_orientation,
      shield_dec.command,
      energy_info.nullspace_stiffness,
      overbudget_joint_terms.torque,
      dt);
  const auto toc_torque = SteadyClock::now();

  for (int i = 0; i < kNumJoints; ++i) command_interfaces_[i].set_value(tau_cmd(i));

  updateCartesianViaPointsActionStatus(
      current_position,
      current_orientation,
      wall_time);

  if (enable_error_logging_ && command_recording_active_ &&
      control_log_writer_.running()) {
    const Vector3d human_center =
        human_workspace_.handReachableSetAtTime(wall_time).center;
    const double mujoco_contact_msg_time =
        latest_mujoco_contact_msg_time_.load(std::memory_order_relaxed);
    const double mujoco_contact_sample_age =
        (mujoco_contact_msg_time >= 0.0)
            ? std::max(0.0, this->get_node()->now().seconds() - mujoco_contact_msg_time)
            : -1.0;
    const double verified_plan_age_sec =
        last_verified_plan_.valid
            ? std::max(0.0, wall_time - last_verified_plan_.generated_wall_time)
            : -1.0;

    control_log_writer_.tryEmplace(
        [&](ControlLogRecord& record) {
          std::size_t index = 0;
          auto add = [&](double value) {
            if (index < record.values.size()) {
              record.values[index] = value;
            }
            ++index;
          };

          add(wall_time);
          add(nominal_guess_time);
          add(paused_nominal_time_sec_);
          add(commanded_path_time_);
          add(shield_dec.command.nominal_path_time);
          add(static_cast<double>(shield_dec.command.nominal_path_time_valid));
          add(commanded_path_rate_);
          add(path_time_rate_target_);
          add(static_cast<double>(executionModeForLog(execution_stage_)));
          add(static_cast<double>(execution_stage_));
          add(static_cast<double>(fallback_reason_));
          add(static_cast<double>(plan_failure_reason_));
          add(static_cast<double>(shield_dec.candidate_verified));
          add(static_cast<double>(monitor_prediction_valid));
          add(static_cast<double>(monitor.predicted_trigger));
          add(static_cast<double>(monitor.collision_interval_index));
          add(static_cast<double>(predicted_contact_possible));
          add(static_cast<double>(monitor.monitored_contact_possible));
          add(static_cast<double>(human_workspace_active_));
          add(static_cast<double>(human_workspace_assumed_clear));
          add(static_cast<double>(monitor.contact_relevant_for_energy));
          add(static_cast<double>(monitor.collision_energy_unsafe));
          add(static_cast<double>(monitor.monitored_unsafe));
          add(static_cast<double>(monitor.joint_limit_unsafe));
          add(static_cast<double>(monitor.joint_limit_index));
          add(monitor.joint_position_violation);
          add(monitor.joint_velocity_violation);
          add(monitor.joint_acceleration_violation);
          add(monitor.joint_torque_violation);
          add(monitor.workspace_distance_now);
          add(monitor.workspace_distance_min);
          add(monitor.workspace_distance_margin);
          for (int i = 0; i < 7; ++i) add(q(i));
          for (int i = 0; i < 7; ++i) add(dq(i));
          for (int i = 0; i < 3; ++i) add(desired_position_cur(i));
          for (int i = 0; i < 3; ++i) add(current_position(i));
          for (int i = 0; i < 6; ++i) add(ee_twist(i));
          for (int i = 0; i < 3; ++i) add(collision_center(i));
          for (int i = 0; i < 3; ++i) add(human_center(i));
          for (int i = 0; i < 3; ++i) add(ee_collision_twist(i));
          for (int i = 0; i < 3; ++i) add(desired_linear_velocity_cur(i));
          for (int i = 0; i < 6; ++i) add(error(i));
          add(tau_cmd.norm());
          add(last_tau_task_norm_);
          add(last_tau_nullspace_raw_norm_);
          add(last_tau_nullspace_projected_norm_);
          add(last_overbudget_joint_torque_norm_);
          add(last_coriolis_norm_);
          add(last_tau_desired_before_rate_limit_norm_);
          add(static_cast<double>(torque_rate_limited_last_));
          add(torque_rate_max_ratio_last_);
          for (int i = 0; i < 3; ++i) add(K_runtime_(i, i));
          for (int i = 0; i < 3; ++i) add(D_runtime_(i, i));
          add(monitor.worst_case_contact_time);
          add(monitor.worst_case_workspace_distance_at_candidate);
          add(monitor.worst_case_cartesian_kinetic_energy_ub);
          add(monitor.worst_case_joint_kinetic_energy_ub);
          add(monitor.worst_case_cartesian_potential_energy_ub);
          add(monitor.worst_case_nullspace_potential_energy_ub);
          add(monitor.worst_case_total_control_energy_ub);
          add(monitor.terminal_energy_ub);
          add(monitor.robot_secure_radius);
          add(static_cast<double>(monitor.robot_reach_alpha_valid));
          for (int i = 0; i < 7; ++i) add(monitor.robot_reach_alpha(i));
          add(static_cast<double>(monitor.current_robot_link_index));
          add(static_cast<double>(monitor.worst_case_robot_link_index));
          add(static_cast<double>(monitor.current_cartesian_energy_valid));
          add(monitor.current_cartesian_kinetic_energy);
          add(static_cast<double>(monitor.current_joint_energy_valid));
          add(monitor.current_joint_kinetic_energy);
          add(monitor.current_cartesian_potential_energy);
          add(monitor.current_nullspace_potential_energy);
          add(monitor.current_total_control_energy);
          add(static_cast<double>(monitored_steps));
          add(static_cast<double>(monitored_intended_steps));
          add(static_cast<double>(monitored_failsafe_steps));
          add(static_cast<double>(control_loop_sequence));
          add(static_cast<double>(monitor_period_control_cycles_));
          add(static_cast<double>(next_async_monitor_control_sequence_));
          add(static_cast<double>(
              last_async_input_publish_control_sequence_));
          add(static_cast<double>(last_async_monitor_timing_.valid));
          add(static_cast<double>(
              last_async_monitor_timing_.input_sequence));
          add(static_cast<double>(
              last_async_monitor_timing_.input_control_loop_sequence));
          add(static_cast<double>(
              last_async_monitor_timing_.source_plan_generation));
          add(static_cast<double>(
              last_async_monitor_timing_.committed_prefix_steps));
          add(static_cast<double>(
              last_async_monitor_timing_.source_plan_matches_at_handoff));
          add(static_cast<double>(
              last_async_monitor_timing_.output_usable));
          add(static_cast<double>(
              last_async_monitor_timing_.scheduled_control_loop_sequence));
          add(static_cast<double>(
              last_async_monitor_timing_.publish_lateness_cycles));
          add(last_async_monitor_timing_.worker_queue_wait_ms);
          add(last_async_monitor_timing_.worker_compute_ms);
          add(last_async_monitor_timing_.output_handoff_ms);
          add(last_async_monitor_timing_.end_to_end_ms);
          add(static_cast<double>(async_monitor_input_publish_count_.load(
              std::memory_order_relaxed)));
          add(static_cast<double>(async_monitor_input_overwrite_count_.load(
              std::memory_order_relaxed)));
          add(static_cast<double>(async_monitor_worker_processed_count_.load(
              std::memory_order_relaxed)));
          add(static_cast<double>(async_monitor_output_overwrite_count_.load(
              std::memory_order_relaxed)));
          add(static_cast<double>(async_monitor_output_consumed_count_.load(
              std::memory_order_relaxed)));
          add(static_cast<double>(async_monitor_schedule_late_cycles_));
          add(static_cast<double>(async_monitor_schedule_skipped_slots_));
          add(verified_plan_age_sec);
          add(static_cast<double>(last_verified_plan_.intended_exec_index));
          add(static_cast<double>(last_verified_plan_.failsafe_exec_index));
          add(static_cast<double>(last_verified_command_stage_));
          add(static_cast<double>(last_verified_command_index_));
          add(static_cast<double>(verified_command_selected_this_cycle_));
          add(static_cast<double>(
              verified_command_selected_this_cycle_
                  ? last_verified_plan_generation_
                  : 0));
          add(static_cast<double>(previous_applied_verified_plan_valid));
          add(static_cast<double>(previous_applied_verified_plan_generation));
          add(static_cast<double>(previous_applied_verified_command_stage));
          add(static_cast<double>(previous_applied_verified_command_index));
          add(static_cast<double>(active_cartesian_via_points_calibration_));
          add(static_cast<double>(calibration_plan_latched_));
          add(static_cast<double>(calibration_plan_complete_));
          add(static_cast<double>(calibration_target_failed_));
          add(calibration_requested_capture_path_time_sec_);
          add(calibration_actual_capture_path_time_sec_);
          add(static_cast<double>(calibration_plan_generation_));
          add(static_cast<double>(calibration_monitor_input_sequence_));
          add(static_cast<double>(calibration_activation_control_sequence_));
          add(static_cast<double>(calibration_activation_intended_index_));
          add(static_cast<double>(calibration_activation_failsafe_index_));
          add(static_cast<double>(enable_runtime_energy_scaling_));
          add(static_cast<double>(last_energy_budget_active_));
          add(static_cast<double>(last_energy_budget_lambda_valid_));
          add(last_energy_stiffness_scale_);
          add(last_joint_kinetic_energy_);
          add(last_cartesian_potential_energy_before_scaling_);
          add(last_nullspace_potential_energy_before_scaling_);
          add(last_total_control_energy_before_scaling_);
          add(last_cartesian_potential_energy_);
          add(last_nullspace_potential_energy_);
          add(static_cast<double>(enable_nullspace_));
          add(n_stiffness_);
          add(last_nullspace_stiffness_);
          add(last_total_control_energy_);
          add(static_cast<double>(
              enable_overbudget_joint_stabilization_));
          add(static_cast<double>(overbudget_joint_state_.active));
          add(overbudget_joint_stiffness_);
          add(overbudget_joint_scale_omega_);
          add(last_overbudget_joint_potential_energy_);
          add(last_overbudget_joint_scale_rho_);
          for (int i = 0; i < 7; ++i) {
            add(overbudget_joint_state_.reference(i));
          }
          add(static_cast<double>(previous_applied_energy_terms.valid));
          add(previous_applied_energy_terms.kinetic_energy);
          add(previous_applied_energy_terms.potential_energy);
          add(previous_applied_energy_terms.nullspace_potential_energy);
          add(static_cast<double>(
              previous_applied_nullspace_stiffness > 0.0));
          add(previous_applied_energy_terms.kinetic_energy +
              previous_applied_energy_terms.potential_energy +
              previous_applied_energy_terms.nullspace_potential_energy);
          add(energy_budget_joule_);
          add(latest_mujoco_contact_value_.load(std::memory_order_relaxed));
          add(static_cast<double>(
              latest_mujoco_contact_active_.load(std::memory_order_relaxed)));
          add(mujoco_contact_sample_age);
          record.value_count = index;
        });
  }
  const auto toc_io = SteadyClock::now();

  const auto toc_total = SteadyClock::now();
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
                "[reachable_impedance] mode=%d stage=%d avg=%.3f ms min=%.3f ms max=%.3f ms overruns>1ms=%zu >2ms=%zu "
                "model_avg/max=%.3f/%.3f shield_avg/max=%.3f/%.3f torque_avg/max=%.3f/%.3f io_avg/max=%.3f/%.3f "
                "plan_valid=%d late_accept=%zu deadline_miss=%zu "
                "monitor_pub/proc/cons=%lu/%lu/%lu overwrite_in/out=%lu/%lu "
                "monitor_wait/compute/handoff/e2e=%.3f/%.3f/%.3f/%.3f ms "
                "log_q=%zu pred_q=%zu log_drop=%zu pred_drop=%zu log_schema_mismatch=%zu",
                executionModeForLog(execution_stage_),
                static_cast<int>(execution_stage_),
                exec_sum_ms_ / n, exec_min_ms_, exec_max_ms_,
                exec_overrun_1ms_count_, exec_overrun_2ms_count_,
                prof_model_sum_ms_ / n, prof_model_max_ms_,
                prof_shield_sum_ms_ / n, prof_shield_max_ms_,
                prof_torque_sum_ms_ / n, prof_torque_max_ms_,
                prof_io_sum_ms_ / n, prof_io_max_ms_,
                static_cast<int>(last_verified_plan_.valid),
                async_late_activation_accept_count_,
                async_activation_deadline_miss_count_,
                static_cast<unsigned long>(
                    async_monitor_input_publish_count_.load(
                        std::memory_order_relaxed)),
                static_cast<unsigned long>(
                    async_monitor_worker_processed_count_.load(
                        std::memory_order_relaxed)),
                static_cast<unsigned long>(
                    async_monitor_output_consumed_count_.load(
                        std::memory_order_relaxed)),
                static_cast<unsigned long>(
                    async_monitor_input_overwrite_count_.load(
                        std::memory_order_relaxed)),
                static_cast<unsigned long>(
                    async_monitor_output_overwrite_count_.load(
                        std::memory_order_relaxed)),
                last_async_monitor_timing_.worker_queue_wait_ms,
                last_async_monitor_timing_.worker_compute_ms,
                last_async_monitor_timing_.output_handoff_ms,
                last_async_monitor_timing_.end_to_end_ms,
                control_log_writer_.queueDepth(),
                prediction_log_writer_.queueDepth(),
                control_log_writer_.droppedCount(),
                prediction_log_writer_.droppedCount(),
                control_log_column_mismatch_count_.load(
                    std::memory_order_relaxed));
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
    async_late_activation_accept_count_ = 0;
    async_activation_deadline_miss_count_ = 0;
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
    auto_declare<bool>("enable_prediction_logging", true);
    auto_declare<std::string>(
        "prediction_log_file_name",
        "shield_prediction_trajectory.csv");
    auto_declare<int>("control_log_max_queue_size", 16384);
    auto_declare<int>("prediction_log_max_queue_size", 256);
    auto_declare<int>("log_batch_size", 512);
    auto_declare<double>("log_flush_period_sec", 1.0);

    auto_declare<std::string>(
        "cartesian_via_points_topic",
        "cartesian_via_points");
    auto_declare<std::string>(
        "startup_via_points_source",
        "yaml");
    auto_declare<std::string>(
        "cartesian_via_points_action_name",
        "~/follow_cartesian_via_points");
    auto_declare<std::string>(
        "calibration_action_name",
        "~/calibrate_monitored_trajectory");
    auto_declare<double>(
        "cartesian_via_points_action_feedback_period_sec",
        0.1);
    const std::string startup_via_points_source =
        get_node()->get_parameter("startup_via_points_source").as_string();
    if (startup_via_points_source != "action") {
      auto_declare<std::vector<double>>(
          "cartesian_via_points",
          std::vector<double>{std::numeric_limits<double>::quiet_NaN()});
      auto_declare<std::vector<double>>(
          "cartesian_via_point_quaternions",
          std::vector<double>{std::numeric_limits<double>::quiet_NaN()});
    }

    auto_declare<double>("pos_stiffness", 400.0);
    auto_declare<double>("rot_stiffness", 20.0);
    auto_declare<bool>("enable_nullspace", false);
    auto_declare<double>("n_stiffness", 0.0);
    auto_declare<std::vector<double>>(
        "nullspace_home_pose", defaultNullspaceHomePoseParameter());
    auto_declare<bool>("enable_safety_monitor", true);

    auto_declare<double>("energy_budget_joule", 0.05);
    auto_declare<double>("kinetic_energy_error_bound_joule", 0.0);
    auto_declare<double>("potential_energy_error_bound_joule", 0.0);
    auto_declare<double>(
        "nullspace_potential_energy_error_bound_joule", 0.0);
    auto_declare<bool>("enable_runtime_energy_scaling", true);
    auto_declare<bool>(
        "enable_overbudget_joint_stabilization", true);
    auto_declare<double>("overbudget_joint_stiffness", 1.0);
    auto_declare<double>("overbudget_joint_scale_omega", 40.0);
    auto_declare<bool>("enable_calibration_logging", false);
    auto_declare<bool>("calibration_assume_no_human", false);
    auto_declare<double>("calibration_capture_path_time_sec", 0.0);
    auto_declare<double>("cartesian_energy_lambda_update_period_sec", 0.001);
    auto_declare<double>("ee_collision_radius", 0.04);
    auto_declare<std::vector<double>>(
        "tcp_offset", std::vector<double>{0.0, 0.0, 0.0});
    auto_declare<std::string>(
        "monitor_urdf_model_path",
        "/home/developer/multipanda_ws/src/model_urdf/panda_ng.urdf");
    auto_declare<std::string>("monitor_joint_dynamics_source", "auto");
    auto_declare<std::vector<double>>(
        "prediction_joint_armature",
        std::vector<double>(7, 0.0));
    auto_declare<std::string>(
        "sara_robot_config_path",
        cps_safety_monitor::defaultSaraPandaRobotConfigPath());
    auto_declare<double>("sara_secure_radius", 0.02);
    auto_declare<bool>("enable_reachable_set_visualization", true);
    auto_declare<std::string>(
        "reachable_set_visualization_topic", "~/robot_reachable_sets");
    auto_declare<std::string>("reachable_set_visualization_frame_id", "");
    auto_declare<double>("reachable_set_visualization_period_sec", 0.1);
    auto_declare<double>("reachable_set_visualization_alpha", 0.3);
    auto_declare<std::vector<double>>(
        "ee_collision_center_offset", std::vector<double>{0.0, 0.0, 0.0});
    auto_declare<bool>("async_safety_monitor", true);
    auto_declare<int>("monitor_worker_cpu_affinity", -1);
    auto_declare<int>("monitor_worker_realtime_priority", 0);
    auto_declare<double>("async_plan_max_age_sec", 0.02);
    auto_declare<int>("async_planning_lead_steps", 8);
    auto_declare<int>("async_verified_horizon_steps", 20);

    cps_human_workspace::HumanWorkspace::declareParameters(get_node());
    auto_declare<std::string>("human_workspace_topic", "human_workspace/state");
    auto_declare<double>("human_workspace_timeout_sec", 0.5);

    auto_declare<bool>("use_dynamic_consistent_impedance", true);

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
    enable_prediction_logging_ =
        get_node()->get_parameter("enable_prediction_logging").as_bool();
    prediction_log_file_name_ = sanitizedFileNameOrDefault(
        get_node()->get_parameter("prediction_log_file_name").as_string(),
        "shield_prediction_trajectory.csv");
    control_log_max_queue_size_ = static_cast<std::size_t>(
        std::max<int64_t>(
            1,
            get_node()->get_parameter("control_log_max_queue_size").as_int()));
    const auto prediction_log_max_queue_size_param =
        get_node()->get_parameter("prediction_log_max_queue_size").as_int();
    prediction_log_max_queue_size_ = static_cast<std::size_t>(
        std::max<int64_t>(1, prediction_log_max_queue_size_param));
    log_batch_size_ = static_cast<std::size_t>(
        std::max<int64_t>(
            1, get_node()->get_parameter("log_batch_size").as_int()));
    log_flush_period_sec_ = std::max(
        0.05,
        get_node()->get_parameter("log_flush_period_sec").as_double());

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

    cartesian_via_points_topic_ =
        get_node()->get_parameter("cartesian_via_points_topic").as_string();
    startup_via_points_source_ =
        get_node()->get_parameter("startup_via_points_source").as_string();
    if (startup_via_points_source_ != "yaml" &&
        startup_via_points_source_ != "action") {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "startup_via_points_source must be 'yaml' or 'action'. Falling back to 'yaml'.");
      startup_via_points_source_ = "yaml";
    }
    cartesian_via_points_action_name_ =
        get_node()->get_parameter("cartesian_via_points_action_name").as_string();
    calibration_action_name_ =
        get_node()->get_parameter("calibration_action_name").as_string();
    cartesian_via_points_action_feedback_period_sec_ =
        std::max(
            0.0,
            get_node()
                ->get_parameter("cartesian_via_points_action_feedback_period_sec")
                .as_double());
    cartesian_via_points_.clear();
    cartesian_via_point_quaternions_.clear();
    std::vector<double> cartesian_via_points;
    if (startup_via_points_source_ != "action") {
      const rclcpp::Parameter via_points_param =
          get_node()->get_parameter("cartesian_via_points");
      if (via_points_param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
        cartesian_via_points = via_points_param.as_double_array();
      } else if (via_points_param.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET) {
        RCLCPP_WARN(
            get_node()->get_logger(),
            "cartesian_via_points must be a double array. Ignoring this value.");
      }
    }

    const bool via_points_are_full_states =
        !cartesian_via_points.empty() &&
        (cartesian_via_points.size() % 7) == 0;
    if (via_points_are_full_states) {
      const std::size_t via_point_count = cartesian_via_points.size() / 7;
      cartesian_via_points_.reserve(via_point_count);
      cartesian_via_point_quaternions_.reserve(via_point_count);
      for (std::size_t i = 0; i < via_point_count; ++i) {
        cartesian_via_points_.emplace_back(
            cartesian_via_points[7 * i + 0],
            cartesian_via_points[7 * i + 1],
            cartesian_via_points[7 * i + 2]);
        const Quaterniond q(
            cartesian_via_points[7 * i + 6],
            cartesian_via_points[7 * i + 3],
            cartesian_via_points[7 * i + 4],
            cartesian_via_points[7 * i + 5]);
        if (!std::isfinite(q.norm()) || q.norm() < 1.0e-12) {
          RCLCPP_WARN(
              get_node()->get_logger(),
              "cartesian_via_points[%zu] has an invalid quaternion. Using identity quaternion.",
              i);
          cartesian_via_point_quaternions_.push_back(Quaterniond::Identity());
        } else {
          cartesian_via_point_quaternions_.push_back(
              normalizedQuaternionOrIdentity(q));
        }
      }
    } else if (!cartesian_via_points.empty() &&
               (cartesian_via_points.size() % 3) == 0) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "cartesian_via_points currently expects 7 values per base-frame state [x, y, z, qx, qy, qz, qw]. Falling back to legacy 3-value base-frame positions and using the activation orientation unless cartesian_via_point_quaternions is set.");
      const std::size_t via_point_count = cartesian_via_points.size() / 3;
      cartesian_via_points_.reserve(via_point_count);
      for (std::size_t i = 0; i < via_point_count; ++i) {
        cartesian_via_points_.emplace_back(
            cartesian_via_points[3 * i + 0],
            cartesian_via_points[3 * i + 1],
            cartesian_via_points[3 * i + 2]);
      }
    } else if (!cartesian_via_points.empty()) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "cartesian_via_points must contain 7-value base-frame states [x, y, z, qx, qy, qz, qw]. Ignoring this value.");
    }

    std::vector<double> cartesian_via_point_quaternions;
    if (startup_via_points_source_ != "action") {
      const rclcpp::Parameter via_quaternions_param =
          get_node()->get_parameter("cartesian_via_point_quaternions");
      if (via_quaternions_param.get_type() ==
          rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
        cartesian_via_point_quaternions =
            via_quaternions_param.as_double_array();
      } else if (via_quaternions_param.get_type() !=
                 rclcpp::ParameterType::PARAMETER_NOT_SET) {
        RCLCPP_WARN(
            get_node()->get_logger(),
            "cartesian_via_point_quaternions must be a double array. Ignoring this value.");
      }
    }
    if (via_points_are_full_states && !cartesian_via_point_quaternions.empty()) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "cartesian_via_point_quaternions is deprecated and ignored because cartesian_via_points already contains 7-value Cartesian states.");
    } else if ((cartesian_via_point_quaternions.size() % 4) != 0) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "cartesian_via_point_quaternions must contain [x, y, z, w] groups. Ignoring the incomplete tail.");
    }
    const std::size_t via_quaternion_count =
        via_points_are_full_states ? 0 : cartesian_via_point_quaternions.size() / 4;
    if (!via_points_are_full_states) {
      cartesian_via_point_quaternions_.reserve(
          cartesian_via_point_quaternions_.size() + via_quaternion_count);
      for (std::size_t i = 0; i < via_quaternion_count; ++i) {
        const Quaterniond q(
            cartesian_via_point_quaternions[4 * i + 3],
            cartesian_via_point_quaternions[4 * i + 0],
            cartesian_via_point_quaternions[4 * i + 1],
            cartesian_via_point_quaternions[4 * i + 2]);
        const double norm = q.norm();
        if (!std::isfinite(norm) || norm < 1.0e-12) {
          RCLCPP_WARN(
              get_node()->get_logger(),
              "cartesian_via_point_quaternions[%zu] is invalid. Using identity quaternion.",
              i);
          cartesian_via_point_quaternions_.push_back(Quaterniond::Identity());
        } else {
          cartesian_via_point_quaternions_.push_back(
              normalizedQuaternionOrIdentity(q));
        }
      }
    }
    if (!cartesian_via_point_quaternions_.empty() &&
        cartesian_via_point_quaternions_.size() != cartesian_via_points_.size()) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "cartesian_via_point_quaternions count (%zu) differs from cartesian_via_points count (%zu). Missing orientations keep the activation orientation; extra orientations rotate at the last Cartesian point.",
          cartesian_via_point_quaternions_.size(),
          cartesian_via_points_.size());
    }

    const double pos_stiffness =
        get_node()->get_parameter("pos_stiffness").as_double();
    const double rot_stiffness =
        get_node()->get_parameter("rot_stiffness").as_double();
    enable_nullspace_ =
        get_node()->get_parameter("enable_nullspace").as_bool();
    const double configured_nullspace_stiffness = std::max(
        0.0, get_node()->get_parameter("n_stiffness").as_double());
    n_stiffness_ =
        enable_nullspace_ ? configured_nullspace_stiffness : 0.0;
    nullspace_home_pose_.setZero();
    nullspace_home_pose_valid_ = false;
    const auto nullspace_home_pose =
        get_node()->get_parameter("nullspace_home_pose").as_double_array();
    if (nullspace_home_pose.size() == kNumJoints) {
      bool finite_home_pose = true;
      for (int i = 0; i < kNumJoints; ++i) {
        nullspace_home_pose_(i) =
            nullspace_home_pose[static_cast<std::size_t>(i)];
        finite_home_pose =
            finite_home_pose && std::isfinite(nullspace_home_pose_(i));
      }
      if (finite_home_pose) {
        nullspace_home_pose_valid_ = true;
      } else {
        nullspace_home_pose_.setZero();
        RCLCPP_WARN(
            get_node()->get_logger(),
            "nullspace_home_pose contains non-finite values. Falling back "
            "to the activation joint state for the nullspace reference.");
      }
    } else {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "nullspace_home_pose must contain 7 joint values. Falling back "
          "to the activation joint state for the nullspace reference.");
    }
    enable_safety_monitor_ = get_node()->get_parameter("enable_safety_monitor").as_bool();

    energy_budget_joule_ =
        std::max(0.0, get_node()->get_parameter("energy_budget_joule").as_double());
    kinetic_energy_error_bound_joule_ = std::max(
        0.0,
        get_node()->get_parameter(
            "kinetic_energy_error_bound_joule").as_double());
    potential_energy_error_bound_joule_ = std::max(
        0.0,
        get_node()->get_parameter(
            "potential_energy_error_bound_joule").as_double());
    nullspace_potential_energy_error_bound_joule_ = std::max(
        0.0,
        get_node()->get_parameter(
            "nullspace_potential_energy_error_bound_joule").as_double());
    enable_runtime_energy_scaling_ =
        get_node()->get_parameter("enable_runtime_energy_scaling").as_bool();
    enable_overbudget_joint_stabilization_ =
        get_node()
            ->get_parameter("enable_overbudget_joint_stabilization")
            .as_bool();
    overbudget_joint_stiffness_ = std::max(
        0.0,
        get_node()->get_parameter("overbudget_joint_stiffness").as_double());
    overbudget_joint_scale_omega_ = std::max(
        0.0,
        get_node()->get_parameter("overbudget_joint_scale_omega").as_double());
    enable_calibration_logging_ =
        get_node()
            ->get_parameter("enable_calibration_logging")
            .as_bool();
    calibration_assume_no_human_ =
        get_node()
            ->get_parameter("calibration_assume_no_human")
            .as_bool();

    cartesian_energy_lambda_update_period_sec_ =
        std::max(0.0, get_node()->get_parameter("cartesian_energy_lambda_update_period_sec").as_double());
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
    monitor_urdf_model_path_ =
        get_node()->get_parameter("monitor_urdf_model_path").as_string();
    monitor_joint_dynamics_source_ =
        get_node()->get_parameter("monitor_joint_dynamics_source").as_string();
    robot_reach_config_path_ =
        get_node()->get_parameter("sara_robot_config_path").as_string();
    robot_secure_radius_ =
        get_node()->get_parameter("sara_secure_radius").as_double();
    enable_reachable_set_visualization_ =
        get_node()->get_parameter(
            "enable_reachable_set_visualization").as_bool();
    reachable_set_visualization_topic_ =
        get_node()->get_parameter(
            "reachable_set_visualization_topic").as_string();
    reachable_set_visualization_frame_id_ =
        get_node()->get_parameter(
            "reachable_set_visualization_frame_id").as_string();
    if (reachable_set_visualization_frame_id_.empty()) {
      reachable_set_visualization_frame_id_ = arm_id_ + "_link0";
    }
    reachable_set_visualization_period_sec_ = std::max(
        0.02,
        get_node()->get_parameter(
            "reachable_set_visualization_period_sec").as_double());
    reachable_set_visualization_alpha_ = std::clamp(
        get_node()->get_parameter(
            "reachable_set_visualization_alpha").as_double(),
        0.01,
        1.0);
    if (enable_reachable_set_visualization_ &&
        reachable_set_visualization_topic_.empty()) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "reachable_set_visualization_topic must not be empty when "
          "visualization is enabled");
      return CallbackReturn::ERROR;
    }
    if (robot_reach_config_path_.empty()) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "sara_robot_config_path must not be empty");
      return CallbackReturn::ERROR;
    }
    if (!std::isfinite(robot_secure_radius_) ||
        robot_secure_radius_ < 0.0) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "sara_secure_radius must be finite and nonnegative");
      return CallbackReturn::ERROR;
    }
    try {
      robot_reachability_provider_ =
          cps_safety_monitor::makeSaraRobotReachabilityProvider(
              robot_reach_config_path_,
              robot_secure_radius_);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Failed to initialize SaRA robot reachable-set provider: %s",
          error.what());
      return CallbackReturn::ERROR;
    }
    const auto prediction_joint_armature =
        get_node()->get_parameter(
            "prediction_joint_armature").as_double_array();
    if (prediction_joint_armature.size() != 7) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "prediction_joint_armature must contain exactly 7 values");
      return CallbackReturn::ERROR;
    }
    for (std::size_t i = 0; i < prediction_joint_armature.size(); ++i) {
      const double value = prediction_joint_armature[i];
      if (!std::isfinite(value) || value < 0.0) {
        RCLCPP_ERROR(
            get_node()->get_logger(),
            "prediction_joint_armature[%zu] must be finite and nonnegative",
            i);
        return CallbackReturn::ERROR;
      }
      prediction_joint_armature_(static_cast<Eigen::Index>(i)) = value;
    }
    std::transform(
        monitor_joint_dynamics_source_.begin(),
        monitor_joint_dynamics_source_.end(),
        monitor_joint_dynamics_source_.begin(),
        [](unsigned char value) {
          return static_cast<char>(std::tolower(value));
        });
    if (monitor_joint_dynamics_source_ != "auto" &&
        monitor_joint_dynamics_source_ != "franka_interface" &&
        monitor_joint_dynamics_source_ != "pinocchio") {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "monitor_joint_dynamics_source must be auto, franka_interface, or pinocchio (got '%s').",
          monitor_joint_dynamics_source_.c_str());
      return CallbackReturn::ERROR;
    }
    if (monitor_joint_dynamics_source_ == "pinocchio" &&
        monitor_urdf_model_path_.empty()) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "monitor_urdf_model_path must not be empty when Pinocchio monitoring is selected.");
      return CallbackReturn::ERROR;
    }
    monitor_joint_dynamics_provider_.reset();
    active_monitor_joint_dynamics_source_.clear();
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
    async_safety_monitor_ = get_node()->get_parameter("async_safety_monitor").as_bool();
    if (enable_reachable_set_visualization_ && !async_safety_monitor_) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Reachable-set visualization requires async_safety_monitor=true "
          "to keep Marker construction out of the 1 kHz control thread. "
          "Visualization is disabled for this configuration.");
      enable_reachable_set_visualization_ = false;
    }
    if (enable_calibration_logging_) {
      if (!async_safety_monitor_) {
        RCLCPP_ERROR(
            get_node()->get_logger(),
            "enable_calibration_logging requires async_safety_monitor=true so prediction endpoints have exact control-loop sequences");
        return CallbackReturn::ERROR;
      }
      if (!enable_error_logging_ || !enable_prediction_logging_) {
        RCLCPP_ERROR(
            get_node()->get_logger(),
            "enable_calibration_logging requires both enable_error_logging=true and enable_prediction_logging=true");
        return CallbackReturn::ERROR;
      }
    }
    if (calibration_assume_no_human_ && !enable_calibration_logging_) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "calibration_assume_no_human is enabled, but calibration logging "
          "is disabled; the override cannot become active.");
    }
    monitor_worker_cpu_affinity_ = static_cast<int>(
        get_node()->get_parameter("monitor_worker_cpu_affinity").as_int());
    if (monitor_worker_cpu_affinity_ < -1) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "monitor_worker_cpu_affinity must be -1 or a non-negative CPU index. Disabling affinity.");
      monitor_worker_cpu_affinity_ = -1;
    }
    const int max_realtime_priority =
        std::max(0, sched_get_priority_max(SCHED_FIFO));
    monitor_worker_realtime_priority_ = std::clamp(
        static_cast<int>(get_node()
                             ->get_parameter(
                                 "monitor_worker_realtime_priority")
                             .as_int()),
        0,
        max_realtime_priority);
    async_plan_max_age_sec_ =
        std::max(0.0, get_node()->get_parameter("async_plan_max_age_sec").as_double());

    trajectory_generator_config_path_ =
        cps_trajectory_generators::defaultTrajectoryGeneratorConfigPath();
    const auto trajectory_settings =
        loadTrajectoryGeneratorSettings(trajectory_generator_config_path_);

    shield_plan_dt_ = std::max(trajectory_settings.shield_plan_dt, kMinDt);
    monitor_frequency_hz_ = std::max(1.0, trajectory_settings.monitor_frequency_hz);
    monitor_update_period_sec_ = 1.0 / monitor_frequency_hz_;

    path_time_rate_min_ = trajectory_settings.path_time_rate_min;
    path_time_rate_max_ =
        std::max(path_time_rate_min_, trajectory_settings.path_time_rate_max);
    path_time_acc_limit_ =
        std::max(0.0, trajectory_settings.path_time_acc_limit);
    path_time_jerk_limit_ =
        std::max(1e-4, trajectory_settings.path_time_jerk_limit);
    path_time_rate_target_ =
        std::clamp(trajectory_settings.path_time_rate_target,
                   path_time_rate_min_,
                   path_time_rate_max_);
    failsafe_path_time_acc_limit_ =
        std::max(1e-4, trajectory_settings.failsafe_path_time_acc_limit);
    failsafe_path_time_jerk_limit_ =
        std::max(1e-4, trajectory_settings.failsafe_path_time_jerk_limit);
    local_replan_horizon_steps_ =
        std::max(1, trajectory_settings.local_replan_horizon_steps);
    local_replan_dt_ = std::max(trajectory_settings.local_replan_dt, kMinDt);
    shield_intended_steps_ = std::max(
        1,
        static_cast<int>(
            std::llround(std::max(monitor_update_period_sec_, local_replan_dt_) /
                         local_replan_dt_)));
    monitor_decimation_ = shield_intended_steps_;
    monitor_period_control_cycles_ = static_cast<std::uint64_t>(
        std::max(1, monitor_decimation_));
    async_planning_lead_steps_ = static_cast<std::size_t>(
        std::max<int64_t>(
            1,
            get_node()->get_parameter("async_planning_lead_steps").as_int()));
    async_verified_horizon_steps_ = static_cast<std::size_t>(
        std::max<int64_t>(
            1,
            get_node()->get_parameter("async_verified_horizon_steps").as_int()));
    local_path_lookahead_sec_ =
        std::max(trajectory_settings.local_path_lookahead_sec, local_replan_dt_);
    waypoint_merge_position_tolerance_ = std::max(
        0.0, trajectory_settings.waypoint_merge_position_tolerance);
    waypoint_merge_orientation_tolerance_ = std::max(
        0.0, trajectory_settings.waypoint_merge_orientation_tolerance);
    local_replan_max_velocity_ =
        std::max(1e-4, trajectory_settings.local_replan_max_velocity);
    local_replan_max_acceleration_ =
        std::max(1e-4, trajectory_settings.local_replan_max_acceleration);
    local_replan_max_jerk_ =
        std::max(1e-4, trajectory_settings.local_replan_max_jerk);
    local_replan_max_angular_velocity_ =
        std::max(1e-4, trajectory_settings.local_replan_max_angular_velocity);
    local_replan_max_angular_acceleration_ =
        std::max(1e-4, trajectory_settings.local_replan_max_angular_acceleration);
    local_replan_max_angular_jerk_ =
        std::max(1e-4, trajectory_settings.local_replan_max_angular_jerk);
    failsafe_brake_max_velocity_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_velocity);
    failsafe_brake_max_acceleration_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_acceleration);
    failsafe_brake_max_jerk_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_jerk);
    failsafe_brake_max_angular_velocity_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_angular_velocity);
    failsafe_brake_max_angular_acceleration_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_angular_acceleration);
    failsafe_brake_max_angular_jerk_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_angular_jerk);

    use_dynamic_consistent_impedance_ = get_node()->get_parameter("use_dynamic_consistent_impedance").as_bool();

    human_workspace_topic_ =
        get_node()->get_parameter("human_workspace_topic").as_string();
    human_workspace_timeout_sec_ = std::max(
        0.0,
        get_node()->get_parameter("human_workspace_timeout_sec").as_double());
    human_workspace_active_ = false;
    human_workspace_configured_static_ = false;
    human_workspace_live_received_.store(false, std::memory_order_relaxed);
    latest_human_workspace_msg_time_sec_.store(-1.0, std::memory_order_relaxed);

    const std::string human_workspace_config_path =
        get_node()->get_parameter("human_workspace_config_path").as_string();
    if (enable_safety_monitor_) {
      if (!human_workspace_config_path.empty()) {
        if (!configured_human_workspace_source_.configureFromConfigFile(
                human_workspace_config_path,
                get_node()->get_logger())) {
          return CallbackReturn::ERROR;
        }
        human_workspace_ = configured_human_workspace_source_;
        human_workspace_configured_static_ = true;
        human_workspace_active_ = true;
      } else {
        RCLCPP_WARN(
            get_node()->get_logger(),
            "enable_safety_monitor is true, but human_workspace_config_path is empty. "
            "Waiting for live human workspace states on '%s'.",
            human_workspace_topic_.c_str());
      }
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

    if (enable_safety_monitor_ && !human_workspace_topic_.empty()) {
      human_workspace_sub_ =
          get_node()->create_subscription<cps_human_workspace::msg::HumanWorkspace>(
              human_workspace_topic_,
              rclcpp::QoS(1).transient_local(),
              std::bind(
                  &ReachableCartesianImpedanceController::handleHumanWorkspaceState,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for human workspace states on '%s'.",
          human_workspace_topic_.c_str());
    } else {
      human_workspace_sub_.reset();
    }

    if (enable_reachable_set_visualization_) {
      reachable_set_visualization_pub_ =
          get_node()->create_publisher<visualization_msgs::msg::MarkerArray>(
              reachable_set_visualization_topic_,
              rclcpp::QoS(1).reliable().transient_local());
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Publishing SaRA robot reachable sets on '%s' in frame '%s' "
          "(safe=last interval, unsafe=first unsafe interval).",
          reachable_set_visualization_topic_.c_str(),
          reachable_set_visualization_frame_id_.c_str());
    } else {
      reachable_set_visualization_pub_.reset();
    }

    if (!cartesian_via_points_topic_.empty()) {
      cartesian_via_points_sub_ =
          get_node()->create_subscription<geometry_msgs::msg::PoseArray>(
              cartesian_via_points_topic_,
              rclcpp::QoS(1),
              std::bind(
                  &ReachableCartesianImpedanceController::handleCartesianViaPoints,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for Cartesian via points on '%s' as geometry_msgs/msg/PoseArray.",
          cartesian_via_points_topic_.c_str());
    } else {
      cartesian_via_points_sub_.reset();
    }

    if (!cartesian_via_points_action_name_.empty()) {
      cartesian_via_points_action_server_ =
          rclcpp_action::create_server<CartesianViaMotion>(
              get_node(),
              cartesian_via_points_action_name_,
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCartesianViaPointsActionGoal,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCartesianViaPointsActionCancel,
                  this,
                  std::placeholders::_1),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCartesianViaPointsActionAccepted,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for Cartesian via-point action goals on '%s'.",
          cartesian_via_points_action_name_.c_str());
    } else {
      cartesian_via_points_action_server_.reset();
    }

    if (!calibration_action_name_.empty()) {
      calibration_action_server_ =
          rclcpp_action::create_server<CartesianViaMotion>(
              get_node(),
              calibration_action_name_,
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCalibrationActionGoal,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCalibrationActionCancel,
                  this,
                  std::placeholders::_1),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCalibrationActionAccepted,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for monitored-trajectory calibration goals on '%s'.",
          calibration_action_name_.c_str());
    } else {
      calibration_action_server_.reset();
    }

    K_base_.setZero(); D_base_.setZero();
    K_base_.topLeftCorner<3, 3>() =
        pos_stiffness * Matrix3d::Identity();
    K_base_.bottomRightCorner<3, 3>() =
        rot_stiffness * Matrix3d::Identity();
    D_base_.topLeftCorner<3, 3>() =
        0.8 * 2.0 * std::sqrt(std::max(pos_stiffness, 0.0)) *
        Matrix3d::Identity();
    D_base_.bottomRightCorner<3, 3>() =
        0.8 * 2.0 * std::sqrt(std::max(rot_stiffness, 0.0)) *
        Matrix3d::Identity();
    K_runtime_ = K_base_; D_runtime_ = D_base_;

    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model", arm_id_));

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

  franka::RobotState* robot_state = franka_robot_model_->getRobotState();
  const Eigen::Map<const Vector7d> q(robot_state->q.data());
  try {
    monitor_joint_dynamics_provider_.reset();
    active_monitor_joint_dynamics_source_.clear();
    const bool interface_supports_prediction =
        franka_robot_model_->supportsStateDependentEvaluation();
    const bool use_franka_interface =
        monitor_joint_dynamics_source_ == "franka_interface" ||
        (monitor_joint_dynamics_source_ == "auto" &&
         interface_supports_prediction);

    if (use_franka_interface) {
      if (!interface_supports_prediction) {
        throw std::runtime_error(
            "selected Franka model backend does not evaluate arbitrary predicted q/dq");
      }
      monitor_joint_dynamics_provider_ =
          std::make_unique<FrankaInterfaceJointDynamicsProvider>(
              franka_robot_model_.get(), *robot_state, tcp_offset_);
      active_monitor_joint_dynamics_source_ = "franka_interface";
    } else {
      if (monitor_urdf_model_path_.empty()) {
        throw std::runtime_error(
            "Pinocchio fallback requires monitor_urdf_model_path");
      }
      monitor_joint_dynamics_provider_ =
          std::make_unique<PinocchioJointDynamicsProvider>(
              monitor_urdf_model_path_,
              tcp_offset_,
              prediction_joint_armature_);
      active_monitor_joint_dynamics_source_ = "pinocchio";
    }
    RCLCPP_INFO(
        get_node()->get_logger(),
        "Joint rollout dynamics source: %s (configured: %s)",
        active_monitor_joint_dynamics_source_.c_str(),
        monitor_joint_dynamics_source_.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to initialize joint rollout dynamics: %s", e.what());
    monitor_joint_dynamics_provider_.reset();
    franka_robot_model_->release_interfaces();
    return CallbackReturn::ERROR;
  }
  if (nullspace_home_pose_valid_) {
    desired_qn_ = nullspace_home_pose_;
  } else {
    desired_qn_ = q;
  }
  const Eigen::Map<const Matrix4d> pose(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  desired_orientation_ = Quaterniond(pose.block<3, 3>(0, 0));
  desired_orientation_.normalize();
  desired_position_ =
      pose.block<3, 1>(0, 3) + desired_orientation_ * tcp_offset_;

  const bool use_yaml_startup_via_points =
      startup_via_points_source_ == "yaml";
  const std::vector<Vector3d> startup_via_points =
      use_yaml_startup_via_points ? cartesian_via_points_
                                  : std::vector<Vector3d>{};
  const std::vector<Quaterniond> startup_via_orientations =
      use_yaml_startup_via_points ? cartesian_via_point_quaternions_
                                  : std::vector<Quaterniond>{};

  std::size_t waypoint_count = 0;
  std::vector<CartesianTrajectorySample> startup_path =
      buildCartesianViaPointPath(
          desired_position_,
          desired_orientation_,
          startup_via_points,
          startup_via_orientations,
          &waypoint_count);
  {
    std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
    cartesian_via_point_path_ = startup_path;
  }

  if (!use_yaml_startup_via_points) {
    RCLCPP_INFO(
        get_node()->get_logger(),
        "startup_via_points_source is 'action'. Holding the initial pose until a CartesianViaMotion action goal is received.");
  } else if (waypoint_count < 2) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "cartesian_via_points is empty. Holding the initial pose until a CartesianViaMotion action goal or PoseArray is received.");
  } else if (startup_path.empty()) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "Failed to time-parameterize startup cartesian_via_points. Holding the initial pose until a CartesianViaMotion action goal or PoseArray is received.");
  } else {
    RCLCPP_INFO(
        get_node()->get_logger(),
        "Startup via-point trajectory prepared: waypoints=%zu samples=%zu duration=%.3f s",
        waypoint_count,
        startup_path.size(),
        startup_path.back().t);
  }

  last_commanded_sample_ = ImpedanceSample{};
  last_commanded_sample_.t = 0.0;
  last_commanded_sample_.p = desired_position_;
  last_commanded_sample_.dp.setZero();
  last_commanded_sample_.ddp.setZero();
  last_commanded_sample_.q = desired_orientation_;
  last_commanded_sample_.q.normalize();
  last_commanded_sample_.w.setZero();
  last_commanded_sample_.dw.setZero();
  last_commanded_sample_.K = K_base_;
  last_commanded_sample_.D = D_base_;
  last_commanded_sample_.failsafe = false;
  last_commanded_sample_valid_ = true;
  if (!startup_path.empty() && !anchorLastCommandedSampleToPathStart()) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "Strict path-consistent execution could not anchor the startup path "
        "to the initial command state. Holding instead of using a Cartesian reconnect.");
  }
  commanded_path_rate_ =
      startup_path.empty() ? 0.0 : path_time_rate_target_;
  mode_ = SafetyMode::kNominal;
  execution_stage_ = ExecutionStage::kCurrentVerified;
  fallback_reason_ = FallbackReason::kNone;
  plan_failure_reason_ = PlanFailureReason::kNone;
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
  command_recording_active_ = false;
  control_log_column_mismatch_count_.store(0, std::memory_order_relaxed);
  control_update_sequence_ = 0;
  next_async_monitor_control_sequence_ = 1;
  last_async_input_publish_control_sequence_ = 0;
  async_monitor_schedule_late_cycles_ = 0;
  async_monitor_schedule_skipped_slots_ = 0;
  async_monitor_input_publish_count_.store(0, std::memory_order_relaxed);
  async_monitor_input_overwrite_count_.store(0, std::memory_order_relaxed);
  async_monitor_worker_processed_count_.store(0, std::memory_order_relaxed);
  async_monitor_output_overwrite_count_.store(0, std::memory_order_relaxed);
  async_monitor_output_consumed_count_.store(0, std::memory_order_relaxed);
  last_async_monitor_timing_ = AsyncMonitorTiming{};
  latest_mujoco_contact_value_.store(0.0);
  latest_mujoco_contact_msg_time_.store(-1.0);
  latest_mujoco_contact_active_.store(false);

  monitor_counter_ = 0;
  last_energy_budget_active_ = false;
  last_energy_budget_lambda_valid_ = false;
  last_energy_stiffness_scale_ = 1.0;
  last_nullspace_stiffness_ = 0.0;
  last_joint_kinetic_energy_ = 0.0;
  last_cartesian_potential_energy_before_scaling_ = 0.0;
  last_nullspace_potential_energy_before_scaling_ = 0.0;
  last_nullspace_potential_energy_ = 0.0;
  last_total_control_energy_before_scaling_ = 0.0;
  last_cartesian_potential_energy_ = 0.0;
  last_total_control_energy_ = 0.0;
  overbudget_joint_state_ =
      cps_safety_monitor::OverbudgetJointStabilizationState{};
  last_overbudget_joint_potential_energy_ = 0.0;
  last_overbudget_joint_scale_rho_ = 1.0;
  last_overbudget_joint_torque_norm_ = 0.0;
  last_tau_task_norm_ = 0.0;
  last_tau_nullspace_raw_norm_ = 0.0;
  last_tau_nullspace_projected_norm_ = 0.0;
  last_coriolis_norm_ = 0.0;
  last_tau_desired_before_rate_limit_norm_ = 0.0;
  cartesian_energy_task_inertia_cache_.setZero();
  cartesian_energy_task_inertia_cache_valid_ = false;
  cartesian_energy_task_inertia_cache_wall_time_ = -1.0;
  last_verified_plan_ = VerifiedPlan{};
  last_verified_plan_generation_ = 0;
  last_verified_command_stage_ = 0;
  last_verified_command_index_ = 0;
  verified_command_selected_this_cycle_ = false;
  last_commanded_verified_plan_valid_ = false;
  last_commanded_verified_plan_generation_ = 0;
  last_commanded_verified_command_stage_ = 0;
  last_commanded_verified_command_index_ = 0;
  calibration_plan_latched_ = false;
  calibration_plan_complete_ = false;
  calibration_target_failed_ = false;
  calibration_requested_capture_path_time_sec_ = 0.0;
  calibration_actual_capture_path_time_sec_ = -1.0;
  calibration_failsafe_command_count_ = 0;
  calibration_plan_generation_ = 0;
  calibration_monitor_input_sequence_ = 0;
  calibration_activation_control_sequence_ = 0;
  calibration_activation_intended_index_ = 0;
  calibration_activation_failsafe_index_ = 0;
  active_cartesian_via_points_calibration_ = false;
  commanded_path_time_ = 0.0;

  tau_cmd_prev_.setZero();
  last_shield_decision_valid_ = false;
  async_input_sequence_.store(0);
  async_input_pending_ = false;
  async_output_mailbox_.resetStopped();
  last_async_output_wall_time_ = -1.0;
  last_async_output_valid_ = false;

  J_geo_prev_.setZero();
  Jdot_dq_filtered_.setZero();
  J_geo_prev_valid_ = false;

  if (enable_error_logging_ || enable_prediction_logging_) {
    const std::filesystem::path root_dir(error_log_root_dir_);
    error_log_run_dir_ =
        (root_dir / makeBerlinTimestampForDirectoryName()).string();
    error_log_file_path_ =
        (std::filesystem::path(error_log_run_dir_) / error_log_file_name_).string();
    prediction_log_file_path_ =
        (std::filesystem::path(error_log_run_dir_) / prediction_log_file_name_).string();

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
      std::vector<CartesianTrajectorySample> path_snapshot;
      {
        std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
        path_snapshot = cartesian_via_point_path_;
      }
      std::string human_workspace_config_path =
          get_node()->get_parameter("human_workspace_config_path").as_string();
      if (human_workspace_config_path.empty()) {
        human_workspace_config_path = "(none)";
      }
      run_info_file << "run_directory: " << error_log_run_dir_ << "\n"
                    << "csv_file: " << error_log_file_path_ << "\n"
                    << "prediction_csv_file: " << prediction_log_file_path_ << "\n"
                    << "enable_error_logging: "
                    << static_cast<int>(enable_error_logging_) << "\n"
                    << "enable_prediction_logging: "
                    << static_cast<int>(enable_prediction_logging_) << "\n"
                    << "enable_inertia_model_comparison: "
                    << static_cast<int>(enable_prediction_logging_) << "\n"
                    << "inertia_model_comparison_semantics: "
                       "same_q_dq_runtime_snapshot_vs_prediction_provider\n"
                    << "logging_backend: bounded_async_csv\n"
                    << "control_log_max_queue_size: "
                    << control_log_max_queue_size_ << "\n"
                    << "prediction_log_max_queue_size: "
                    << prediction_log_max_queue_size_ << "\n"
                    << "log_batch_size: " << log_batch_size_ << "\n"
                    << "log_flush_period_sec: "
                    << log_flush_period_sec_ << "\n"
                    << "recording_start: first_valid_via_points_command\n"
                    << "arm_id: " << arm_id_ << "\n"
                    << "state_log_schema: orthogonal_execution_v10\n"
                    << "mode_legend: 0=current_verified_execution, "
                       "1=fallback_execution\n"
                    << "execution_stage_legend: 0=current_verified, "
                       "1=last_verified_intended, "
                       "2=last_verified_failsafe, "
                       "3=hold\n"
                    << "fallback_reason_legend: 0=none, "
                       "1=no_verified_plan_available, "
                       "2=candidate_prediction_rejected, "
                       "3=async_output_unavailable, "
                       "4=human_workspace_unavailable\n"
                    << "reason_invariant: at_most_one_of_fallback_reason_"
                       "and_plan_failure_reason_is_nonzero\n"
                    << "plan_failure_reason_legend: 0=none, "
                       "1=no_active_path, "
                       "2=missing_nominal_path_state, "
                       "3=intended_generation_empty, "
                       "4=intended_seam_invalid, "
                       "5=failsafe_generation_empty, "
                       "6=failsafe_seam_invalid, "
                       "7=intended_sample_invalid, "
                       "8=intended_transition_invalid, "
                       "9=failsafe_sample_invalid, "
                       "10=failsafe_transition_invalid, "
                       "11=candidate_invalid_unknown\n"
                    << "cartesian_via_points_count: "
                    << cartesian_via_points_.size() << "\n"
                    << "cartesian_via_point_quaternions_count: "
                    << cartesian_via_point_quaternions_.size() << "\n"
                    << "cartesian_via_points_topic: "
                    << cartesian_via_points_topic_ << "\n"
                    << "cartesian_via_point_path_samples: "
                    << path_snapshot.size() << "\n"
                    << "cartesian_via_point_path_duration_sec: "
                    << (path_snapshot.empty()
                            ? 0.0
                            : path_snapshot.back().t)
                    << "\n"
                    << "enable_safety_monitor: " << static_cast<int>(enable_safety_monitor_) << "\n"
                    << "async_safety_monitor: " << static_cast<int>(async_safety_monitor_) << "\n"
                    << "monitor_schedule_source: control_loop_sequence\n"
                    << "monitor_period_control_cycles: "
                    << monitor_period_control_cycles_ << "\n"
                    << "monitor_worker_cpu_affinity: "
                    << monitor_worker_cpu_affinity_ << "\n"
                    << "monitor_worker_realtime_priority: "
                    << monitor_worker_realtime_priority_ << "\n"
                    << "async_plan_max_age_sec: " << async_plan_max_age_sec_ << "\n"
                    << "async_planning_lead_steps: "
                    << async_planning_lead_steps_ << "\n"
                    << "async_verified_horizon_steps: "
                    << async_verified_horizon_steps_ << "\n"
                    << "monitor_frequency_hz: " << monitor_frequency_hz_ << "\n"
                    << "monitor_joint_dynamics_source_configured: "
                    << monitor_joint_dynamics_source_ << "\n"
                    << "monitor_joint_dynamics_source_active: "
                    << active_monitor_joint_dynamics_source_ << "\n"
                    << "prediction_joint_armature: ["
                    << prediction_joint_armature_(0) << ", "
                    << prediction_joint_armature_(1) << ", "
                    << prediction_joint_armature_(2) << ", "
                    << prediction_joint_armature_(3) << ", "
                    << prediction_joint_armature_(4) << ", "
                    << prediction_joint_armature_(5) << ", "
                    << prediction_joint_armature_(6) << "]\n"
                    << "prediction_joint_armature_applied: "
                    << static_cast<int>(
                           active_monitor_joint_dynamics_source_ ==
                           "pinocchio")
                    << "\n"
                    << "robot_reachable_set_backend: "
                    << (robot_reachability_provider_
                            ? robot_reachability_provider_->backendName()
                            : "legacy_cartesian")
                    << "\n"
                    << "sara_robot_config_path: "
                    << robot_reach_config_path_ << "\n"
                    << "sara_secure_radius: "
                    << robot_secure_radius_ << "\n"
                    << "sara_alpha_mode: dynamic_from_complete_monitored_trajectory\n"
                    << "enable_reachable_set_visualization: "
                    << static_cast<int>(
                         enable_reachable_set_visualization_) << "\n"
                    << "reachable_set_visualization_topic: "
                    << reachable_set_visualization_topic_ << "\n"
                    << "reachable_set_visualization_frame_id: "
                    << reachable_set_visualization_frame_id_ << "\n"
                    << "reachable_set_visualization_period_sec: "
                    << reachable_set_visualization_period_sec_ << "\n"
                    << "reachable_set_visualization_alpha: "
                    << reachable_set_visualization_alpha_ << "\n"
                    << "monitor_update_period_sec: " << monitor_update_period_sec_ << "\n"
                    << "monitor_decimation: " << monitor_decimation_ << "\n"
                    << "shield_intended_steps: " << shield_intended_steps_ << "\n"
                    << "trajectory_generator_config_path: "
                    << trajectory_generator_config_path_ << "\n"
                    << "shield_plan_dt: " << shield_plan_dt_ << "\n"
                    << "monitor_sparse_dt: " << shield_plan_dt_ << "\n"
                    << "monitor_joint_rollout_max_dt: "
                    << shield_plan_dt_ << "\n"
                    << "local_replan_dt: " << local_replan_dt_ << "\n"
                    << "failsafe_plan_dt: "
                    << std::max(local_replan_dt_, kMinDt) << "\n"
                    << "failsafe_command_dt: " << local_replan_dt_ << "\n"
                    << "cartesian_gain_policy: "
                       "contact_gated_shared_cartesian_nullspace_"
                       "energy_scaling\n"
                    << "pos_stiffness: " << K_base_(0, 0) << "\n"
                    << "rot_stiffness: " << K_base_(3, 3) << "\n"
                    << "pos_damping: " << D_base_(0, 0) << "\n"
                    << "rot_damping: " << D_base_(3, 3) << "\n"
                    << "enable_nullspace: "
                    << static_cast<int>(enable_nullspace_) << "\n"
                    << "nullspace_stiffness: " << n_stiffness_ << "\n"
                    << "nullspace_reference: "
                    << desired_qn_.transpose() << "\n"
                    << "energy_budget_joule: "
                    << energy_budget_joule_ << "\n"
                    << "kinetic_energy_error_bound_joule: "
                    << kinetic_energy_error_bound_joule_ << "\n"
                    << "potential_energy_error_bound_joule: "
                    << potential_energy_error_bound_joule_ << "\n"
                    << "nullspace_potential_energy_error_bound_joule: "
                    << nullspace_potential_energy_error_bound_joule_ << "\n"
                    << "enable_runtime_energy_scaling: "
                    << static_cast<int>(enable_runtime_energy_scaling_)
                    << "\n"
                    << "enable_overbudget_joint_stabilization: "
                    << static_cast<int>(
                           enable_overbudget_joint_stabilization_)
                    << "\n"
                    << "overbudget_joint_stiffness: "
                    << overbudget_joint_stiffness_ << "\n"
                    << "overbudget_joint_scale_omega: "
                    << overbudget_joint_scale_omega_ << "\n"
                    << "enable_calibration_logging: "
                    << static_cast<int>(
                           enable_calibration_logging_)
                    << "\n"
                    << "calibration_assume_no_human: "
                    << static_cast<int>(calibration_assume_no_human_)
                    << "\n"
                    << "calibration_capture_path_time_sec_default: "
                    << get_node()
                           ->get_parameter(
                               "calibration_capture_path_time_sec")
                           .as_double()
                    << "\n"
                    << "calibration_action_name: "
                    << calibration_action_name_ << "\n"
                    << "calibration_plan_policy: "
                       "dedicated_action_capture_exactly_next_monitor_input_"
                       "at_or_after_requested_path_time_"
                       "abort_if_not_executable_no_candidate_substitution_"
                       "then_execute_remaining_intended_and_failsafe\n"
                    << "calibration_prediction_sampling: "
                       "exact_runtime_command_grid_and_indices\n"
                    << "measured_energy_tracking_policy: "
                       "global_diagnostics_only_no_control_action\n"
                    << "measured_energy_tracking_feeds_scaling_when_active: "
                       "1\n"
                    << "cartesian_energy_lambda_update_period_sec: "
                    << cartesian_energy_lambda_update_period_sec_ << "\n"
                    << "energy_budget_activation: "
                       "current_workspace_overlap_only\n"
                    << "prediction_energy_scaling_policy: "
                       "same_predicted_current_workspace_overlap_gate_"
                       "as_runtime\n"
                    << "overbudget_joint_stabilization_policy: "
                       "current_workspace_overlap_and_T_gt_Lmax_only\n"
                    << "contact_relevant_for_energy_legend: "
                       "1=workspace_distance_now_nonpositive\n"
                    << "nullspace_home_pose: ["
                    << nullspace_home_pose_(0) << ", "
                    << nullspace_home_pose_(1) << ", "
                    << nullspace_home_pose_(2) << ", "
                    << nullspace_home_pose_(3) << ", "
                    << nullspace_home_pose_(4) << ", "
                    << nullspace_home_pose_(5) << ", "
                    << nullspace_home_pose_(6) << "]\n"
                    << "nullspace_home_pose_valid: "
                    << static_cast<int>(nullspace_home_pose_valid_) << "\n"
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
                    << "human_sphere_center: ["
                    << human_workspace_.center().x() << ", "
                    << human_workspace_.center().y() << ", "
                    << human_workspace_.center().z() << "]\n"
                    << "human_center_motion_enabled: "
                    << static_cast<int>(human_workspace_.hasMovingCenter()) << "\n"
                    << "human_center_linear_velocity: ["
                    << human_workspace_.centerLinearVelocity().x() << ", "
                    << human_workspace_.centerLinearVelocity().y() << ", "
                    << human_workspace_.centerLinearVelocity().z() << "]\n"
                    << "human_center_sinusoid_amplitude: ["
                    << human_workspace_.centerSinusoidAmplitude().x() << ", "
                    << human_workspace_.centerSinusoidAmplitude().y() << ", "
                    << human_workspace_.centerSinusoidAmplitude().z() << "]\n"
                    << "human_center_sinusoid_frequency_hz: "
                    << human_workspace_.centerSinusoidFrequencyHz() << "\n"
                    << "human_center_sinusoid_phase_rad: "
                    << human_workspace_.centerSinusoidPhaseRad() << "\n"
                    << "human_center_motion_time_offset_sec: "
                    << human_workspace_.centerMotionTimeOffsetSec() << "\n"
                    << "human_hand_physical_radius: "
                    << human_workspace_.motionRadius() << "\n"
                    << "human_reachability_model: "
                    << "sara_body_part_combined_single_hand\n"
                    << "human_hand_max_velocity: "
                    << human_workspace_.handMaxVelocity() << "\n"
                    << "human_hand_max_acceleration: "
                    << human_workspace_.handMaxAcceleration() << "\n"
                    << "human_measurement_error_position: "
                    << human_workspace_.measurementErrorPosition() << "\n"
                    << "human_measurement_error_velocity: "
                    << human_workspace_.measurementErrorVelocity() << "\n"
                    << "human_measurement_delay: "
                    << human_workspace_.measurementDelay() << "\n"
                    << "human_workspace_config_path: "
                    << human_workspace_config_path << "\n";
    }

    if (!startLogWriters()) {
      return CallbackReturn::ERROR;
    }
    if (enable_error_logging_) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Asynchronous validation log enabled: %s",
                  error_log_file_path_.c_str());
    }
    if (enable_prediction_logging_) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Asynchronous shield prediction log enabled: %s",
                  prediction_log_file_path_.c_str());
    }
  }
  last_reachable_set_visualization_wall_time_ = -1.0;
  last_reachable_set_visualization_marker_count_ = 0;
  startSafetyMonitorWorker();
  return CallbackReturn::SUCCESS;
}

CallbackReturn ReachableCartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  stopSafetyMonitorWorker();
  clearRobotReachableSetVisualization();
  stopLogWriters();
  plan_failure_reason_ = PlanFailureReason::kNone;
  human_workspace_active_ = false;
  verified_command_selected_this_cycle_ = false;
  last_commanded_verified_plan_valid_ = false;
  calibration_plan_latched_ = false;
  calibration_plan_complete_ = false;
  calibration_target_failed_ = false;
  calibration_requested_capture_path_time_sec_ = 0.0;
  calibration_actual_capture_path_time_sec_ = -1.0;
  active_cartesian_via_points_calibration_ = false;
  J_geo_prev_.setZero();
  Jdot_dq_filtered_.setZero();
  J_geo_prev_valid_ = false;
  monitor_joint_dynamics_provider_.reset();
  active_monitor_joint_dynamics_source_.clear();
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace cps_controllers

PLUGINLIB_EXPORT_CLASS(cps_controllers::ReachableCartesianImpedanceController,
                       controller_interface::ControllerInterface)
