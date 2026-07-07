// Copyright (c) 2026
// Reachable Cartesian Impedance Controller
//
// Monitored execution with a Cartesian Lachner-style energy budget in nominal contact.
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
#include <cstdint>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <franka/model.h>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <geometry_msgs/msg/pose_array.hpp>

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
using SteadyClock = std::chrono::steady_clock;
using CartesianViaMotionAction =
    panda_motion_generator_msgs::action::CartesianViaMotion;
using SimpleActionResult = panda_motion_generator_msgs::msg::SimpleActionResult;
using CartesianTrajectorySample = cps_trajectory_generators::CartesianTrajectorySample;
using LocalCartesianReplanConfig = cps_trajectory_generators::LocalCartesianReplanConfig;
using PathConsistentTimedPathConfig =
    cps_trajectory_generators::PathConsistentTimedPathConfig;
using cps_trajectory_generators::loadTrajectoryGeneratorSettings;
using cps_trajectory_generators::makeLocalCartesianReplanFromTimedPath;
using cps_trajectory_generators::makePathConsistentTimedPathBrake;
using cps_trajectory_generators::makePathConsistentTimedPathIntendedPrefix;
using cps_trajectory_generators::makeSmoothViaPointCartesianTrajectory;

cps_controllers::SafetyMode nominalSafetyModeForMonitor(
    const cps_safety_monitor::MonitorResult& monitor) {
  return monitor.contact_relevant_for_energy
             ? cps_controllers::SafetyMode::kNominalContactPossible
             : cps_controllers::SafetyMode::kNominal;
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

inline Vector3d normalizedOrZero(const Vector3d& v) {
  const double norm = v.norm();
  if (norm < 1.0e-9) {
    return Vector3d::Zero();
  }
  return v / norm;
}

inline Vector3d clampVectorNorm(const Vector3d& v, double max_norm) {
  const double limit = std::max(max_norm, 0.0);
  const double norm = v.norm();
  if (norm <= limit || norm < 1.0e-12) {
    return v;
  }
  return v * (limit / norm);
}

inline Quaterniond normalizedQuaternionOrIdentity(const Quaterniond& q_in) {
  Quaterniond q = q_in;
  const double norm = q.norm();
  if (!std::isfinite(norm) || norm < 1.0e-12) {
    return Quaterniond::Identity();
  }
  q.coeffs() /= norm;
  return q;
}

inline std::shared_ptr<CartesianViaMotionAction::Result>
makeCartesianViaMotionActionResult(int32_t state, const std::string& message) {
  auto result = std::make_shared<CartesianViaMotionAction::Result>();
  result->result.state = state;
  result->result.error = message;
  return result;
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

inline Vector3d fallbackMonitorDirection(
    const Vector3d& x_pred,
    const Vector3d& v_pred,
    const cps_human_workspace::HumanWorkspace& human_workspace,
    double time_sec) {
  Vector3d direction = normalizedOrZero(v_pred);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  direction = normalizedOrZero(human_workspace.centerAtTime(time_sec) - x_pred);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  return human_workspace.direction();
}

inline Vector3d commandMonitorDirection(
    const cps_safety_monitor::ImpedanceSample& sample,
    const Vector3d& previous_command_position,
    const Vector3d& x_pred,
    const Vector3d& v_pred,
    const cps_human_workspace::HumanWorkspace& human_workspace,
    double time_sec) {
  Vector3d direction = normalizedOrZero(sample.dp);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  direction = normalizedOrZero(sample.p - previous_command_position);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  direction = normalizedOrZero(sample.p - x_pred);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  return fallbackMonitorDirection(x_pred, v_pred, human_workspace, time_sec);
}

}  // namespace

namespace cps_controllers {

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

void ReachableCartesianImpedanceController::handleHumanWorkspaceState(
    const cps_human_workspace::msg::HumanWorkspace::SharedPtr msg) {
  if (!msg) {
    return;
  }

  cps_human_workspace::HumanWorkspace::Parameters parameters;
  parameters.workspace_direction = Vector3d(
      msg->workspace_direction.x,
      msg->workspace_direction.y,
      msg->workspace_direction.z);
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
  parameters.hand_radius = msg->hand_radius;

  if (parameters.workspace_direction.norm() < 1.0e-8 ||
      parameters.motion_radius < 0.0 ||
      parameters.hand_radius < 0.0 ||
      !std::isfinite(parameters.motion_radius) ||
      !std::isfinite(parameters.hand_radius)) {
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
  latest_human_workspace_msg_time_sec_.store(
      get_node()->now().seconds(),
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
          "Safety monitor is waiting for fresh workspace data.",
          human_workspace_topic_.c_str());
    }
    return human_workspace_active_;
  }

  human_workspace_active_ = human_workspace_configured_static_;
  if (!human_workspace_active_) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "No human workspace state received on '%s'. "
        "Safety monitor is waiting for the human workspace provider.",
        human_workspace_topic_.c_str());
  }
  return human_workspace_active_;
}

bool ReachableCartesianImpedanceController::shouldRejectCandidateWithMonitor(
    const MonitorResult& monitor,
    bool human_workspace_active) const {
  if (!enable_safety_monitor_ || !human_workspace_active) {
    return false;
  }

  // Use the current monitored trajectory when verification passes; fall back to
  // the last verified trajectory only when future-contact verification fails.
  return monitor.predicted_trigger;
}

bool ReachableCartesianImpedanceController::shouldRejectCandidateWithMonitor(
    const MonitorResult& monitor) const {
  return shouldRejectCandidateWithMonitor(monitor, human_workspace_active_);
}

bool ReachableCartesianImpedanceController::shouldApplyCartesianEnergyBudget(
    const MonitorResult& monitor,
    const ImpedanceSample& command,
    bool executing_last_verified_monitored) const {
  (void)command;
  return enable_safety_monitor_ &&
         human_workspace_active_ &&
         monitor.contact_relevant_for_energy &&
         !executing_last_verified_monitored;
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

ImpedanceSample ReachableCartesianImpedanceController::makeEffectiveTimeHoldSample(
    const ImpedanceSample& command) const {
  ImpedanceSample hold = command;
  const double min_pos_stiffness =
      std::max(0.0, cartesian_energy_min_pos_stiffness_);
  if (min_pos_stiffness > kSmallPositive) {
    bool below_floor = false;
    for (int i = 0; i < 3; ++i) {
      const double nominal_floor =
          K_nominal_(i, i) > kSmallPositive
              ? std::min(min_pos_stiffness, K_nominal_(i, i))
              : min_pos_stiffness;
      below_floor = below_floor || hold.K(i, i) < nominal_floor;
    }
    if (below_floor) {
      hold.K = K_nominal_;
      hold.D = D_nominal_;
    }
  }
  hold.dp.setZero();
  hold.ddp.setZero();
  hold.w.setZero();
  hold.dw.setZero();
  hold.failsafe = false;
  return hold;
}

double ReachableCartesianImpedanceController::cartesianEnergyScaleFloor(
    const Matrix6d& K_reference) const {
  const double min_pos_stiffness =
      std::max(0.0, cartesian_energy_min_pos_stiffness_);
  if (min_pos_stiffness <= kSmallPositive) {
    return 0.0;
  }

  double scale_floor = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double k_ref = K_reference(i, i);
    if (k_ref > kSmallPositive) {
      const double bounded_floor = std::min(min_pos_stiffness, k_ref);
      scale_floor = std::max(scale_floor, bounded_floor / k_ref);
    }
  }
  return clamp01(scale_floor);
}

ImpedanceSample ReachableCartesianImpedanceController::applyCartesianEnergyBudget(
    const ImpedanceSample& command,
    const Vector6d& error,
    const Vector6d& xdot,
    const Matrix6d& lambda,
    bool lambda_valid,
    bool active,
    CartesianEnergyBudgetInfo* info) const {
  CartesianEnergyBudgetInfo local_info;
  local_info.active = active;

  ImpedanceSample scaled_command = command;
  if (!active) {
    if (info != nullptr) {
      *info = local_info;
    }
    return scaled_command;
  }

  local_info.lambda_valid = lambda_valid;
  if (local_info.lambda_valid) {
    local_info.kinetic_energy =
        std::max(0.0, 0.5 * (xdot.transpose() * lambda * xdot)(0, 0));
  }

  local_info.potential_energy =
      std::max(0.0, 0.5 * (error.transpose() * command.K * error)(0, 0));
  local_info.total_energy =
      local_info.kinetic_energy + local_info.potential_energy;

  const double budget = std::max(0.0, energy_budget_joule_);
  if (local_info.kinetic_energy > budget) {
    local_info.scale = 0.0;
  } else if (local_info.total_energy <= budget) {
    local_info.scale = 1.0;
  } else if (local_info.potential_energy > kSmallPositive) {
    local_info.scale =
        clamp01((budget - local_info.kinetic_energy) /
                local_info.potential_energy);
  } else {
    local_info.scale = 0.0;
  }
  local_info.scale =
      std::max(local_info.scale, cartesianEnergyScaleFloor(command.K));

  scaled_command.K = local_info.scale * command.K;
  scaled_command.D = std::sqrt(local_info.scale) * command.D;

  if (info != nullptr) {
    *info = local_info;
  }
  return scaled_command;
}


double ReachableCartesianImpedanceController::computeConservativeNormalAccelPositiveBound(
    double m_eff_n, double e_n_abs, double /*v_n_abs*/, const Matrix6d& K_used) const {
  const double m_safe = std::max(m_eff_n, kSmallPositive);
  const double k_n_up = human_workspace_.normalStiffness(K_used);
  const double a_from_stiffness = (k_n_up * e_n_abs) / m_safe;
  return std::max(0.0, a_from_stiffness);
}

double ReachableCartesianImpedanceController::computeConservativeNormalBrakeAccelLowerBound(
    double m_eff_n, double v_n_abs, const Matrix6d& D_used) const {
  const double m_safe = std::max(m_eff_n, kSmallPositive);
  const double d_n_low = human_workspace_.normalDamping(D_used);
  const double a_from_damping = (d_n_low * v_n_abs) / m_safe;
  return std::max(0.0, a_from_damping);
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

// ============================================================================
// Generate intended prefix from the trajectory generator package
// ============================================================================
std::vector<ImpedanceSample>
ReachableCartesianImpedanceController::makeIntendedBufferFromReplanner(
    double nominal_guess_time,
    const ImpedanceSample& planning_start_command,
    double initial_path_rate,
    double commanded_path_time) const {
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
  config.max_angular_velocity = local_replan_max_angular_velocity_;
  config.max_angular_acceleration = local_replan_max_angular_acceleration_;
  config.max_angular_jerk = local_replan_max_angular_jerk_;

  std::vector<CartesianTrajectorySample> active_path;
  {
    std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
    active_path = cartesian_via_point_path_;
  }

  std::vector<CartesianTrajectorySample> planned_samples;
  if (!active_path.empty()) {
    PathConsistentTimedPathConfig path_config;
    path_config.intended_steps = std::max(1, shield_intended_steps_);
    path_config.dt = local_replan_dt_;
    path_config.path_lookahead_sec = local_path_lookahead_sec_;
    path_config.max_path_rate = std::max(path_time_rate_max_, 1e-4);
    path_config.max_path_acceleration =
        std::max(path_time_acc_limit_, 1e-4);
    path_config.max_path_jerk = std::max(local_replan_max_jerk_, 1e-4);
    path_config.target_path_rate =
        std::clamp(path_time_rate_target_, path_time_rate_min_,
                   path_time_rate_max_);
    path_config.initial_path_rate =
        std::clamp(initial_path_rate, path_time_rate_min_, path_time_rate_max_);

    const double path_start_time =
        std::max(nominal_guess_time, commanded_path_time);

    planned_samples = makePathConsistentTimedPathIntendedPrefix(
        path_start_time,
        planning_start,
        active_path,
        path_config);

    if (planned_samples.empty()) {
      planned_samples = makeLocalCartesianReplanFromTimedPath(
          path_start_time,
          planning_start,
          active_path,
          config);
    }
  }

  if (!planned_samples.empty()) {
    Vector3d previous_position = planning_start.p;
    Vector3d previous_velocity = planning_start.dp;
    for (auto& sample : planned_samples) {
      const double dt = std::max(local_replan_dt_, kMinDt);
      const Vector3d raw_velocity =
          (sample.p - previous_position) / dt;
      const Vector3d velocity =
          clampVectorNorm(raw_velocity, local_replan_max_velocity_);
      const Vector3d raw_acceleration =
          (velocity - previous_velocity) / dt;
      sample.dp = velocity;
      sample.ddp =
          clampVectorNorm(raw_acceleration, local_replan_max_acceleration_);
      previous_position = sample.p;
      previous_velocity = sample.dp;
    }
  }

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
    const std::vector<ImpedanceSample>& intended_samples) const {
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

  plan.intended.clear();
  plan.intended.reserve(intended_samples.size());
  for (std::size_t i = 0; i < intended_samples.size(); ++i) {
    ImpedanceSample step = intended_samples[i];
    step.t = plan.anchor.t + static_cast<double>(i + 1) *
                            std::max(local_replan_dt_, kMinDt);
    step.K = K_nominal_;
    step.D = D_nominal_;
    step.failsafe = false;
    plan.intended.push_back(step);
  }

  if (plan.intended.empty()) {
    return plan;
  }

  plan.nominal_time_anchor = intended_samples.back().t;
  const ImpedanceSample freeze_anchor = plan.intended.back();
  const double failsafe_plan_dt = std::max(shield_plan_dt_, local_replan_dt_);

  auto fill_failsafe_prefix = [&](const Matrix6d& K_terminal) {
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

    std::vector<CartesianTrajectorySample> active_path;
    {
      std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
      active_path = cartesian_via_point_path_;
    }

    std::vector<CartesianTrajectorySample> brake_samples;
    if (!active_path.empty()) {
      PathConsistentTimedPathConfig path_brake_config;
      path_brake_config.dt = failsafe_plan_dt;
      path_brake_config.path_lookahead_sec = local_path_lookahead_sec_;
      path_brake_config.max_path_rate = std::max(path_time_rate_max_, 1e-4);
      path_brake_config.max_path_acceleration =
          std::max(failsafe_brake_max_acceleration_, 1e-4);
      path_brake_config.max_path_jerk =
          std::max(failsafe_brake_max_jerk_, 1e-4);
      path_brake_config.target_path_rate = 0.0;

      brake_samples = makePathConsistentTimedPathBrake(
          plan.nominal_time_anchor,
          brake_start,
          active_path,
          path_brake_config);
    }

    LocalCartesianReplanConfig brake_config;
    brake_config.dt = failsafe_plan_dt;
    brake_config.max_velocity = failsafe_brake_max_velocity_;
    brake_config.max_acceleration = failsafe_brake_max_acceleration_;
    brake_config.max_jerk = failsafe_brake_max_jerk_;
    brake_config.max_angular_velocity = failsafe_brake_max_angular_velocity_;
    brake_config.max_angular_acceleration =
        failsafe_brake_max_angular_acceleration_;
    brake_config.max_angular_jerk = failsafe_brake_max_angular_jerk_;

    if (brake_samples.empty()) {
      brake_samples = makeCartesianBrakeTrajectory(brake_start, brake_config);
    }

    if (brake_samples.empty()) {
      const double tk = freeze_anchor.t + failsafe_plan_dt;
      plan.failsafe.push_back(
          makeFrozenFailsafeSample(tk, freeze_anchor, K_terminal, D_f_target_));
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
          (1.0 - s_blend) * K_nominal_ + s_blend * K_terminal;

      const Matrix6d D_sched =
          computeDampingFromStiffness(
              K_sched,
              failsafe_pos_damping_scale_,
              failsafe_rot_damping_scale_);

      ImpedanceSample s;
      const auto& brake = brake_samples[static_cast<std::size_t>(i)];
      s.t = tk;
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

  // Unified fail-safe stiffness policy:
  // The terminal stiffness is exactly the configured failsafe stiffness.
  // No online k_budget / k_selected update is performed here.
  fill_failsafe_prefix(K_f_target_);

  plan.valid = !plan.intended.empty() && !plan.failsafe.empty();
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
  config.energy_budget_margin_joule = energy_budget_margin_joule_;
  config.ee_collision_radius = ee_collision_radius_;
  config.contact_activation_margin = contact_activation_margin_;
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
  sparse_plan.valid = dense_plan.valid;
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
// evaluateCandidatePlan
// ============================================================================
MonitorResult ReachableCartesianImpedanceController::evaluateCandidatePlan(
    const VerifiedPlan& plan,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    const cps_human_workspace::HumanWorkspace& human_workspace,
    bool human_workspace_active) const {
  if (!enable_safety_monitor_ || !human_workspace_active) {
    return MonitorResult{};
  }

  const VerifiedPlan monitor_plan = makeSparsePlanForMonitor(plan);
  const VerifiedPlan collision_center_plan =
      makeCollisionCenterPlanForMonitor(monitor_plan);
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
      makeSafetyMonitorConfig(
          human_workspace,
          K_runtime,
          D_runtime,
          plan.generated_wall_time));
}

// ============================================================================
// Shield decision
// ============================================================================
ShieldDecision ReachableCartesianImpedanceController::computeShieldDecision(
    double wall_time, double nominal_guess_time,
    const Vector3d& current_position, const Quaterniond& current_orientation,
    const Vector6d& ee_twist, const Matrix7d& inertia, const Matrix37d& Jv) {
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
        current_position,
        current_orientation,
        ee_twist,
        inertia,
        Jv,
        K_runtime_,
        D_runtime_,
        human_workspace_,
        human_workspace_active_);
  };

  auto execute_last_verified_monitored = [&]() {
    dec.executing_last_verified_monitored = true;
    dec.candidate_verified = false;
    if (last_verified_plan_.valid &&
        (!last_verified_plan_.intended.empty() ||
         !last_verified_plan_.failsafe.empty())) {
      dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
    } else {
      dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
    }
  };

  auto try_one_monitor_pass = [&]() -> bool {
    ImpedanceSample planning_start_command;
    planning_start_command.t = wall_time;

    if (last_commanded_sample_valid_) {
      planning_start_command.p = last_commanded_sample_.p;
      planning_start_command.q = last_commanded_sample_.q;
      planning_start_command.dp = last_commanded_sample_.dp;
      planning_start_command.w = last_commanded_sample_.w;
      planning_start_command.ddp = last_commanded_sample_.ddp;
      planning_start_command.dw = last_commanded_sample_.dw;
    } else {
      planning_start_command.p = current_position;
      planning_start_command.q = current_orientation;
      planning_start_command.dp = ee_twist.head<3>();
      planning_start_command.w = ee_twist.tail<3>();
      planning_start_command.ddp.setZero();
      planning_start_command.dw.setZero();
    }
    planning_start_command.q.normalize();

    planning_start_command.K = K_nominal_;
    planning_start_command.D = D_nominal_;
    planning_start_command.failsafe = false;

    const auto planner_tic = SteadyClock::now();
    const std::vector<ImpedanceSample> intended_buffer =
        makeIntendedBufferFromReplanner(
            nominal_guess_time,
            planning_start_command,
            commanded_path_rate_,
            commanded_path_time_);
    planner_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - planner_tic).count();

    if (intended_buffer.empty()) return false;

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
        Jv,
        K_runtime_,
        D_runtime_,
        intended_segment);
    plan_build_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - build_tic).count();
    if (!candidate_plan.valid) return false;

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
    mode_ = SafetyMode::kNominal;
    dec.executing_last_verified_monitored = false;
    dec.candidate_verified = false;
    dec.command = makeEmergencyStopCommand(
        current_position,
        current_orientation,
        wall_time);
    stamp_timing();
    return dec;
  }

  if (dec.candidate_verified) {
    last_verified_plan_ = candidate_plan;
    last_verified_plan_.valid = true;
    last_verified_plan_.intended_exec_index = 0;
    last_verified_plan_.failsafe_exec_index = 0;

    if (!cartesian_effective_time_frozen_) {
      const double previous_path_time = commanded_path_time_;
      if (last_verified_plan_.nominal_time_anchor > 0.0) {
        commanded_path_time_ =
            std::max(commanded_path_time_, last_verified_plan_.nominal_time_anchor);
      }
      if (!last_verified_plan_.intended.empty()) {
        const double plan_duration =
            std::max(local_replan_dt_,
                     static_cast<double>(last_verified_plan_.intended.size()) *
                         local_replan_dt_);
        if (last_verified_plan_.nominal_time_anchor > previous_path_time + kMinDt) {
          commanded_path_rate_ = std::clamp(
              (last_verified_plan_.nominal_time_anchor - previous_path_time) /
                  plan_duration,
              path_time_rate_min_,
              path_time_rate_max_);
        } else {
          bool active_path_finished = false;
          {
            std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
            active_path_finished =
                !cartesian_via_point_path_.empty() &&
                commanded_path_time_ >= cartesian_via_point_path_.back().t - kMinDt;
          }
          if (active_path_finished) {
            commanded_path_rate_ = 0.0;
          }
        }
      }
    }

    mode_ = nominalSafetyModeForMonitor(dec.monitor);

    dec.executing_last_verified_monitored = false;
    dec.command =
        getNextVerifiedTrajectoryCommandFromCache(!cartesian_effective_time_frozen_);
    stamp_timing();
    return dec;
  }

  mode_ = SafetyMode::kLastVerifiedMonitored;
  execute_last_verified_monitored();
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

  auto execute_last_verified_monitored = [&]() {
    dec.executing_last_verified_monitored = true;
    dec.candidate_verified = false;

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
    ImpedanceSample planning_start_command;
    planning_start_command.t = input.wall_time;

    if (input.last_commanded_sample_valid) {
      planning_start_command.p = input.last_commanded_sample.p;
      planning_start_command.q = input.last_commanded_sample.q;
      planning_start_command.dp = input.last_commanded_sample.dp;
      planning_start_command.w = input.last_commanded_sample.w;
      planning_start_command.ddp = input.last_commanded_sample.ddp;
      planning_start_command.dw = input.last_commanded_sample.dw;
    } else {
      planning_start_command.p = input.current_position;
      planning_start_command.q = input.current_orientation;
      planning_start_command.dp = input.ee_twist.head<3>();
      planning_start_command.w = input.ee_twist.tail<3>();
      planning_start_command.ddp.setZero();
      planning_start_command.dw.setZero();
    }
    planning_start_command.q.normalize();

    planning_start_command.K = K_nominal_;
    planning_start_command.D = D_nominal_;
    planning_start_command.failsafe = false;

    const auto planner_tic = SteadyClock::now();
    const std::vector<ImpedanceSample> intended_buffer =
        makeIntendedBufferFromReplanner(
            input.nominal_guess_time,
            planning_start_command,
            input.commanded_path_rate,
            input.commanded_path_time);
    planner_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - planner_tic).count();

    if (intended_buffer.empty()) return false;

    const std::size_t segment_count = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(1, shield_intended_steps_)),
        intended_buffer.size());
    std::vector<ImpedanceSample> intended_segment(
        intended_buffer.begin(),
        intended_buffer.begin() + static_cast<std::ptrdiff_t>(segment_count));

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
        intended_segment);
    plan_build_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - build_tic).count();

    if (!candidate_plan.valid) {
      return false;
    }

    const auto eval_tic = SteadyClock::now();
    dec.monitor = evaluateCandidatePlan(
        candidate_plan,
        input.current_position,
        input.current_orientation,
        input.ee_twist,
        input.inertia,
        input.Jv,
        input.K_runtime,
        input.D_runtime,
        input.human_workspace,
        input.human_workspace_active);
    monitor_eval_ms +=
        std::chrono::duration<double, std::milli>(SteadyClock::now() - eval_tic).count();
    dec.evaluated_plan = candidate_plan;
    dec.has_evaluated_plan = true;

    dec.candidate_verified =
        !shouldRejectCandidateWithMonitor(
            dec.monitor,
            input.human_workspace_active);

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
    dec.executing_last_verified_monitored = false;
    dec.candidate_verified = false;
    dec.command = makeEmergencyStopCommand(
        input.current_position,
        input.current_orientation,
        input.wall_time);
    stamp_timing();
    return dec;
  }

  if (dec.candidate_verified) {
    stamp_timing();
    return dec;
  }

  execute_last_verified_monitored();
  stamp_timing();
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

bool ReachableCartesianImpedanceController::takePendingCartesianViaPoints(
    std::vector<Vector3d>* points,
    std::vector<Quaterniond>* orientations,
    std::shared_ptr<CartesianViaMotionGoalHandle>* goal_handle,
    std::uint64_t* sequence) {
  if (points == nullptr || orientations == nullptr ||
      goal_handle == nullptr || sequence == nullptr) {
    return false;
  }

  std::lock_guard<std::mutex> lock(pending_cartesian_via_points_mutex_);
  if (!pending_cartesian_via_points_available_) {
    return false;
  }

  *points = pending_cartesian_via_points_;
  *orientations = pending_cartesian_via_point_quaternions_;
  *goal_handle = pending_cartesian_via_points_goal_handle_;
  *sequence = pending_cartesian_via_points_sequence_;
  pending_cartesian_via_points_available_ = false;
  pending_cartesian_via_points_goal_handle_.reset();
  return true;
}

std::vector<CartesianTrajectorySample>
ReachableCartesianImpedanceController::buildCartesianViaPointPath(
    const Vector3d& start_position,
    const Quaterniond& start_orientation,
    const std::vector<Vector3d>& via_points,
    const std::vector<Quaterniond>& via_orientations,
    std::size_t* waypoint_count) const {
  std::vector<Vector3d> waypoints;
  std::vector<Quaterniond> waypoint_orientations;
  const std::size_t via_pose_count =
      std::max(via_points.size(), via_orientations.size());
  waypoints.reserve(via_pose_count + 1);
  waypoint_orientations.reserve(via_pose_count + 1);

  Quaterniond normalized_start_orientation =
      normalizedQuaternionOrIdentity(start_orientation);
  normalized_start_orientation.normalize();
  waypoints.push_back(start_position);
  waypoint_orientations.push_back(normalized_start_orientation);

  for (std::size_t i = 0; i < via_pose_count; ++i) {
    const Vector3d waypoint =
        i < via_points.size() ? via_points[i] : waypoints.back();

    Quaterniond waypoint_orientation = normalized_start_orientation;
    if (i < via_orientations.size()) {
      waypoint_orientation = normalizedQuaternionOrIdentity(via_orientations[i]);
      waypoint_orientation.normalize();
    }

    if ((waypoint - waypoints.back()).norm() > 1e-9 ||
        std::abs(waypoint_orientation.coeffs().dot(
            waypoint_orientations.back().coeffs())) < 1.0 - 1e-9) {
      waypoints.push_back(waypoint);
      waypoint_orientations.push_back(waypoint_orientation);
    }
  }

  if (waypoint_count != nullptr) {
    *waypoint_count = waypoints.size();
  }
  if (waypoints.size() < 2) {
    return {};
  }

  LocalCartesianReplanConfig via_config;
  via_config.horizon_steps = local_replan_horizon_steps_;
  via_config.dt = local_replan_dt_;
  via_config.path_lookahead_sec = local_path_lookahead_sec_;
  via_config.max_velocity = local_replan_max_velocity_;
  via_config.max_acceleration = local_replan_max_acceleration_;
  via_config.max_jerk = local_replan_max_jerk_;
  via_config.max_angular_velocity = local_replan_max_angular_velocity_;
  via_config.max_angular_acceleration = local_replan_max_angular_acceleration_;
  via_config.max_angular_jerk = local_replan_max_angular_jerk_;
  return makeSmoothViaPointCartesianTrajectory(
      waypoints,
      waypoint_orientations,
      via_config);
}

void ReachableCartesianImpedanceController::acceptPendingCartesianViaPoints(
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    double wall_time) {
  std::vector<Vector3d> via_points;
  std::vector<Quaterniond> via_orientations;
  std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle;
  std::uint64_t sequence = 0;
  if (!takePendingCartesianViaPoints(
          &via_points, &via_orientations, &goal_handle, &sequence)) {
    return;
  }

  std::size_t waypoint_count = 0;
  std::vector<CartesianTrajectorySample> path =
      buildCartesianViaPointPath(
          current_position,
          current_orientation,
          via_points,
          via_orientations,
          &waypoint_count);

  if (path.empty()) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "Received cartesian_via_points message %lu, but no valid path could be time-parameterized. Keeping the current path.",
        static_cast<unsigned long>(sequence));
    if (goal_handle && goal_handle->is_active()) {
      goal_handle->abort(makeCartesianViaMotionActionResult(
          SimpleActionResult::REJECTED,
          "No valid Cartesian via-point path could be time-parameterized."));
    }
    return;
  }

  cartesian_via_points_ = std::move(via_points);
  cartesian_via_point_quaternions_ = std::move(via_orientations);
  {
    std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
    cartesian_via_point_path_ = std::move(path);
  }

  resetViaPointExecutionState(current_position, current_orientation, wall_time);
  command_recording_active_ = true;

  std::shared_ptr<CartesianViaMotionGoalHandle> previous_active_goal;
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    previous_active_goal = active_cartesian_via_points_goal_handle_;
    active_cartesian_via_points_goal_handle_ = goal_handle;
    cartesian_via_points_action_last_feedback_wall_time_ = -1.0;
  }
  if (previous_active_goal &&
      previous_active_goal != goal_handle &&
      previous_active_goal->is_active()) {
    previous_active_goal->abort(makeCartesianViaMotionActionResult(
        SimpleActionResult::ABORTED,
        "Cartesian via-point goal was replaced by a newer via-point command."));
  }

  std::vector<CartesianTrajectorySample> path_snapshot;
  {
    std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
    path_snapshot = cartesian_via_point_path_;
  }
  RCLCPP_INFO(
      get_node()->get_logger(),
      "Accepted cartesian_via_points message %lu: poses=%zu waypoints=%zu samples=%zu duration=%.3f s",
      static_cast<unsigned long>(sequence),
      cartesian_via_points_.size(),
      waypoint_count,
      path_snapshot.size(),
      path_snapshot.empty() ? 0.0 : path_snapshot.back().t);
}

void ReachableCartesianImpedanceController::resetViaPointExecutionState(
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    double wall_time) {
  paused_nominal_time_sec_ = wall_time;
  failsafe_start_time_sec_ = -1.0;
  failsafe_enter_wall_time_sec_ = -1.0;
  commanded_path_time_ = 0.0;
  commanded_path_rate_ = 0.0;
  mode_ = SafetyMode::kNominal;

  last_verified_plan_ = VerifiedPlan{};
  last_verified_command_stage_ = 0;
  last_verified_command_index_ = 0;
  last_shield_decision_valid_ = false;
  last_async_output_valid_ = false;
  last_async_output_wall_time_ = -1.0;
  last_async_input_publish_wall_time_ = -1.0;
  cartesian_effective_time_frozen_ = false;
  cartesian_effective_time_freeze_start_wall_time_ = -1.0;
  cartesian_effective_time_hold_sample_valid_ = false;

  {
    std::lock_guard<std::mutex> input_lock(async_input_mutex_);
    async_input_pending_ = false;
  }
  {
    std::lock_guard<std::mutex> output_lock(async_output_mutex_);
    latest_async_output_ = AsyncMonitorOutput{};
    last_consumed_async_output_sequence_ = 0;
  }

  last_commanded_sample_ = ImpedanceSample{};
  last_commanded_sample_.t = 0.0;
  last_commanded_sample_.p = current_position;
  last_commanded_sample_.dp.setZero();
  last_commanded_sample_.ddp.setZero();
  last_commanded_sample_.q = normalizedQuaternionOrIdentity(current_orientation);
  last_commanded_sample_.w.setZero();
  last_commanded_sample_.dw.setZero();
  last_commanded_sample_.K = K_nominal_;
  last_commanded_sample_.D = D_nominal_;
  last_commanded_sample_.failsafe = false;
  last_commanded_sample_valid_ = true;
}

void ReachableCartesianImpedanceController::updateCartesianViaPointsActionStatus(
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    double wall_time) {
  std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle;
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    goal_handle = active_cartesian_via_points_goal_handle_;
  }

  if (!goal_handle) {
    return;
  }

  if (goal_handle->is_canceling()) {
    {
      std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
      cartesian_via_point_path_.clear();
    }
    cartesian_via_points_.clear();
    cartesian_via_point_quaternions_.clear();
    resetViaPointExecutionState(current_position, current_orientation, wall_time);
    goal_handle->canceled(makeCartesianViaMotionActionResult(
        SimpleActionResult::PREEMPTED,
        "Cartesian via-point goal was canceled."));
    {
      std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
      if (active_cartesian_via_points_goal_handle_ == goal_handle) {
        active_cartesian_via_points_goal_handle_.reset();
      }
    }
    return;
  }

  if (!goal_handle->is_active()) {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    if (active_cartesian_via_points_goal_handle_ == goal_handle) {
      active_cartesian_via_points_goal_handle_.reset();
    }
    return;
  }

  double path_duration = 0.0;
  {
    std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
    if (!cartesian_via_point_path_.empty()) {
      path_duration = cartesian_via_point_path_.back().t;
    }
  }

  if (path_duration <= kMinDt) {
    return;
  }

  const double progress =
      std::clamp(commanded_path_time_ / path_duration, 0.0, 1.0);
  bool should_publish_feedback = false;
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    should_publish_feedback =
        cartesian_via_points_action_last_feedback_wall_time_ < 0.0 ||
        wall_time - cartesian_via_points_action_last_feedback_wall_time_ >=
            cartesian_via_points_action_feedback_period_sec_;
    if (should_publish_feedback) {
      cartesian_via_points_action_last_feedback_wall_time_ = wall_time;
    }
  }

  if (should_publish_feedback) {
    auto feedback = std::make_shared<CartesianViaMotion::Feedback>();
    feedback->progress = static_cast<float>(progress);
    feedback->time_to_completion =
        static_cast<float>(std::max(0.0, path_duration - commanded_path_time_));
    goal_handle->publish_feedback(feedback);
  }

  const bool path_finished =
      commanded_path_time_ >= path_duration - kMinDt &&
      isNominalSafetyMode(mode_) &&
      !cartesian_effective_time_frozen_;
  if (!path_finished) {
    return;
  }

  goal_handle->succeed(makeCartesianViaMotionActionResult(
      SimpleActionResult::SUCCESS,
      "Cartesian via-point goal completed."));
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    if (active_cartesian_via_points_goal_handle_ == goal_handle) {
      active_cartesian_via_points_goal_handle_.reset();
    }
  }
}

rclcpp_action::GoalResponse
ReachableCartesianImpedanceController::handleCartesianViaPointsActionGoal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const CartesianViaMotion::Goal> goal) {
  (void)uuid;
  if (!goal || goal->via_poses.empty()) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "Rejecting Cartesian via-point action goal because it contains no poses.");
    return rclcpp_action::GoalResponse::REJECT;
  }

  for (std::size_t i = 0; i < goal->via_poses.size(); ++i) {
    const auto& pose = goal->via_poses[i];
    const Vector3d position(
        pose.position.x,
        pose.position.y,
        pose.position.z);
    const Quaterniond orientation(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z);
    if (!std::isfinite(position.x()) ||
        !std::isfinite(position.y()) ||
        !std::isfinite(position.z())) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Rejecting Cartesian via-point action goal because pose %zu has a non-finite position.",
          i);
      return rclcpp_action::GoalResponse::REJECT;
    }
    const double orientation_norm = orientation.norm();
    if (!std::isfinite(orientation_norm) || orientation_norm < 1.0e-12) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Rejecting Cartesian via-point action goal because pose %zu has an invalid quaternion.",
          i);
      return rclcpp_action::GoalResponse::REJECT;
    }
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
ReachableCartesianImpedanceController::handleCartesianViaPointsActionCancel(
    const std::shared_ptr<CartesianViaMotionGoalHandle> /*goal_handle*/) {
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ReachableCartesianImpedanceController::handleCartesianViaPointsActionAccepted(
    const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle) {
  if (!goal_handle) {
    return;
  }

  const auto goal = goal_handle->get_goal();
  std::vector<Vector3d> points;
  std::vector<Quaterniond> orientations;
  points.reserve(goal->via_poses.size());
  orientations.reserve(goal->via_poses.size());

  for (const auto& pose : goal->via_poses) {
    points.emplace_back(pose.position.x, pose.position.y, pose.position.z);
    orientations.push_back(normalizedQuaternionOrIdentity(Quaterniond(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z)));
  }

  std::shared_ptr<CartesianViaMotionGoalHandle> previous_pending_goal;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(pending_cartesian_via_points_mutex_);
    previous_pending_goal = pending_cartesian_via_points_goal_handle_;
    pending_cartesian_via_points_ = std::move(points);
    pending_cartesian_via_point_quaternions_ = std::move(orientations);
    pending_cartesian_via_points_goal_handle_ = goal_handle;
    sequence = ++pending_cartesian_via_points_sequence_;
    pending_cartesian_via_points_available_ = true;
  }

  std::shared_ptr<CartesianViaMotionGoalHandle> previous_active_goal;
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    previous_active_goal = active_cartesian_via_points_goal_handle_;
    if (previous_active_goal && previous_active_goal != goal_handle) {
      active_cartesian_via_points_goal_handle_.reset();
    }
  }

  if (previous_pending_goal &&
      previous_pending_goal != goal_handle &&
      previous_pending_goal->is_active()) {
    previous_pending_goal->abort(makeCartesianViaMotionActionResult(
        SimpleActionResult::ABORTED,
        "Cartesian via-point goal was replaced by a newer action goal."));
  }
  if (previous_active_goal &&
      previous_active_goal != goal_handle &&
      previous_active_goal->is_active()) {
    previous_active_goal->abort(makeCartesianViaMotionActionResult(
        SimpleActionResult::ABORTED,
        "Cartesian via-point goal was replaced by a newer action goal."));
  }

  RCLCPP_INFO(
      get_node()->get_logger(),
      "Queued Cartesian via-point action goal %lu with %zu poses.",
      static_cast<unsigned long>(sequence),
      goal->via_poses.size());
}

void ReachableCartesianImpedanceController::safetyMonitorWorkerLoop() {
  VerifiedPlan last_verified_plan;

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
    output.input = input;
    output.decision =
        computeShieldDecisionForAsyncInput(input, last_verified_plan);

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
  last_async_input_publish_wall_time_ = -1.0;

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

void ReachableCartesianImpedanceController::handleCartesianViaPoints(
    const geometry_msgs::msg::PoseArray::SharedPtr msg) {
  if (!msg) {
    return;
  }

  const std::string& frame_id = msg->header.frame_id;
  const std::string robot_base_frame_id =
      arm_id_.empty() ? "panda_link0" : arm_id_ + "_link0";
  if (!frame_id.empty() && frame_id != robot_base_frame_id) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "Received cartesian_via_points in frame '%s'. Interpreting poses as robot base frame '%s'.",
        frame_id.c_str(),
        robot_base_frame_id.c_str());
  }

  std::vector<Vector3d> points;
  std::vector<Quaterniond> orientations;
  points.reserve(msg->poses.size());
  orientations.reserve(msg->poses.size());

  for (std::size_t i = 0; i < msg->poses.size(); ++i) {
    const auto& pose = msg->poses[i];
    const Vector3d position(
        pose.position.x,
        pose.position.y,
        pose.position.z);
    const Quaterniond orientation(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z);

    if (!std::isfinite(position.x()) ||
        !std::isfinite(position.y()) ||
        !std::isfinite(position.z())) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Ignoring cartesian_via_points message because pose %zu has a non-finite position.",
          i);
      return;
    }

    const double orientation_norm = orientation.norm();
    if (!std::isfinite(orientation_norm) || orientation_norm < 1.0e-12) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Ignoring cartesian_via_points message because pose %zu has an invalid quaternion.",
          i);
      return;
    }

    points.push_back(position);
    orientations.push_back(normalizedQuaternionOrIdentity(orientation));
  }

  if (points.empty()) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "Ignoring empty cartesian_via_points PoseArray message.");
    return;
  }

  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(pending_cartesian_via_points_mutex_);
    pending_cartesian_via_points_ = std::move(points);
    pending_cartesian_via_point_quaternions_ = std::move(orientations);
    pending_cartesian_via_points_goal_handle_.reset();
    sequence = ++pending_cartesian_via_points_sequence_;
    pending_cartesian_via_points_available_ = true;
  }

  RCLCPP_INFO(
      get_node()->get_logger(),
      "Queued cartesian_via_points message %lu with %zu base-frame poses.",
      static_cast<unsigned long>(sequence),
      msg->poses.size());
}

void ReachableCartesianImpedanceController::handleMujocoContactSensor(
    const mujoco_ros_msgs::msg::ScalarStamped::SharedPtr msg) {
  const double value = msg->value;
  latest_mujoco_contact_value_.store(value, std::memory_order_relaxed);
  latest_mujoco_contact_msg_time_.store(
      rclcpp::Time(msg->header.stamp).seconds(),
      std::memory_order_relaxed);
  latest_mujoco_contact_sequence_.fetch_add(1, std::memory_order_relaxed);

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

void ReachableCartesianImpedanceController::logShieldPredictionTrajectory(
    double wall_time,
    double nominal_guess_time,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    const cps_human_workspace::HumanWorkspace& human_workspace,
    const VerifiedPlan& evaluated_plan,
    const MonitorResult& monitor,
    bool candidate_verified,
    bool executing_last_verified_monitored,
    double monitor_total_ms,
    double planner_ms,
    double plan_build_ms,
    double monitor_eval_ms,
    const char* source) {
  if (!enable_prediction_logging_ || !prediction_log_file_.is_open() ||
      !command_recording_active_ ||
      !evaluated_plan.valid) {
    return;
  }

  PredictionLogRecord record;
  record.wall_time = wall_time;
  record.nominal_guess_time = nominal_guess_time;
  record.current_position = current_position;
  record.current_orientation = current_orientation;
  record.ee_twist = ee_twist;
  record.inertia = inertia;
  record.Jv = Jv;
  record.K_runtime = K_runtime;
  record.D_runtime = D_runtime;
  record.human_workspace = human_workspace;
  record.evaluated_plan = evaluated_plan;
  record.monitor = monitor;
  record.candidate_verified = candidate_verified;
  record.executing_last_verified_monitored = executing_last_verified_monitored;
  record.monitor_total_ms = monitor_total_ms;
  record.planner_ms = planner_ms;
  record.plan_build_ms = plan_build_ms;
  record.monitor_eval_ms = monitor_eval_ms;
  record.source = source == nullptr ? "" : source;

  if (!prediction_log_mutex_.try_lock()) {
    return;
  }
  if (prediction_log_queue_.size() >= prediction_log_max_queue_size_) {
    prediction_log_queue_.pop_front();
  }
  prediction_log_queue_.push_back(std::move(record));
  prediction_log_mutex_.unlock();
  prediction_log_cv_.notify_one();
}

void ReachableCartesianImpedanceController::writeShieldPredictionTrajectory(
    double wall_time,
    double nominal_guess_time,
    const Vector3d& current_position,
    const Quaterniond& current_orientation,
    const Vector6d& ee_twist,
    const Matrix7d& inertia,
    const Matrix37d& Jv,
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    const cps_human_workspace::HumanWorkspace& human_workspace,
    const VerifiedPlan& evaluated_plan,
    const MonitorResult& monitor,
    bool candidate_verified,
    bool executing_last_verified_monitored,
    double monitor_total_ms,
    double planner_ms,
    double plan_build_ms,
    double monitor_eval_ms,
    const char* source) {
  if (!enable_prediction_logging_ || !prediction_log_file_.is_open() ||
      !evaluated_plan.valid) {
    return;
  }

  struct PredictionRow {
    const char* stage{""};
    int index{0};
    bool failsafe{false};
    double sample_t{0.0};
    double dt{0.0};
    Vector3d flange_target_p{Vector3d::Zero()};
    Vector3d collision_target_p{Vector3d::Zero()};
    Vector3d x_pred{Vector3d::Zero()};
    Vector3d v_pred{Vector3d::Zero()};
    Vector3d x_next{Vector3d::Zero()};
    Vector3d v_next{Vector3d::Zero()};
    Vector3d a_pred{Vector3d::Zero()};
    double d_pred{0.0};
    double d_next{0.0};
    double d_segment{0.0};
    Vector3d human_center_start{Vector3d::Zero()};
    Vector3d human_center_end{Vector3d::Zero()};
    bool contact_possible{false};
    double k_n{0.0};
    double e_n{0.0};
    double v_n{0.0};
    double T_n_ub{0.0};
    double V_n_ub{0.0};
    double Kx{0.0};
    double Ky{0.0};
    double Kz{0.0};
    double Dx{0.0};
    double Dy{0.0};
    double Dz{0.0};
  };

  const VerifiedPlan monitor_plan = makeSparsePlanForMonitor(evaluated_plan);
  const VerifiedPlan collision_plan =
      makeCollisionCenterPlanForMonitor(monitor_plan);
  const Vector3d collision_center =
      current_position + collisionCenterOffsetWorld(current_orientation);
  const Vector6d collision_twist =
      twistAtCollisionCenter(current_orientation, ee_twist);

  const double rho_p = std::max(0.0, tracking_pos_error_bound_);
  const double rho_v = std::max(0.0, tracking_vel_error_bound_);
  const double inflated_contact_radius =
      human_workspace.inflatedCollisionRadius(ee_collision_radius_, rho_p);

  Matrix3d task_inertia_inv = Jv * inertia.inverse() * Jv.transpose();
  task_inertia_inv.diagonal().array() += kSmallPositive;
  Matrix6d K_exec = K_runtime;
  Matrix6d D_exec = D_runtime;

  Vector3d x_pred = collision_center;
  Vector3d v_pred = collision_twist.head<3>();
  double t_prev = collision_plan.anchor.t;
  Vector3d previous_command_position = collision_plan.anchor.p;
  std::vector<PredictionRow> rows;
  rows.reserve(1 + collision_plan.intended.size() + collision_plan.failsafe.size());

  auto append_row = [&](const char* stage,
                        int index,
                        const ImpedanceSample& flange_sample,
                        const ImpedanceSample& collision_sample,
                        double dtp) {
    const double segment_start_time_sec = wall_time + t_prev;
    const double segment_end_time_sec = wall_time + collision_sample.t;
    K_exec = collision_sample.K;
    D_exec = collision_sample.D;

    const Matrix3d Kp = K_exec.topLeftCorner<3, 3>();
    const Matrix3d Dp = D_exec.topLeftCorner<3, 3>();
    const Vector3d force_pred =
        Kp * (collision_sample.p - x_pred) -
        Dp * (v_pred - collision_sample.dp);
    Vector3d a_pred = Vector3d::Zero();
    if (use_dynamic_consistent_impedance_) {
      a_pred = collision_sample.ddp + task_inertia_inv * force_pred;
    } else {
      a_pred = task_inertia_inv * force_pred;
    }

    const Vector3d x_next =
        x_pred + v_pred * dtp + 0.5 * a_pred * dtp * dtp;
    const Vector3d v_next = v_pred + a_pred * dtp;

    PredictionRow row;
    row.stage = stage;
    row.index = index;
    row.failsafe = collision_sample.failsafe;
    row.sample_t = collision_sample.t;
    row.dt = dtp;
    row.flange_target_p = flange_sample.p;
    row.collision_target_p = collision_sample.p;
    row.x_pred = x_pred;
    row.v_pred = v_pred;
    row.x_next = x_next;
    row.v_next = v_next;
    row.a_pred = a_pred;
    row.human_center_start = human_workspace.centerAtTime(segment_start_time_sec);
    row.human_center_end = human_workspace.centerAtTime(segment_end_time_sec);
    row.d_pred =
        human_workspace.signedDistanceToInflatedSphere(
            x_pred, inflated_contact_radius, segment_start_time_sec);
    row.d_next =
        human_workspace.signedDistanceToInflatedSphere(
            x_next, inflated_contact_radius, segment_end_time_sec);
    row.d_segment =
        human_workspace.signedDistanceSegmentToInflatedSphere(
            x_pred,
            x_next,
            inflated_contact_radius,
            segment_start_time_sec,
            segment_end_time_sec);
    row.contact_possible = row.d_segment <= 0.0;
    const Vector3d monitor_direction =
        commandMonitorDirection(
            collision_sample,
            previous_command_position,
            x_pred,
            v_pred,
            human_workspace,
            segment_end_time_sec);
    const double denom =
        (monitor_direction.transpose() * task_inertia_inv * monitor_direction)(0, 0);
    const double m_eff_dir = 1.0 / std::max(denom, kSmallPositive);
    row.k_n =
        std::max((monitor_direction.transpose() * Kp * monitor_direction)(0, 0), 0.0);
    row.e_n = std::abs(monitor_direction.dot(x_next - collision_sample.p)) + rho_p;
    row.v_n = std::abs(monitor_direction.dot(v_next)) + rho_v;
    row.T_n_ub = 0.5 * m_eff_dir * row.v_n * row.v_n;
    row.V_n_ub = 0.5 * row.k_n * row.e_n * row.e_n;
    row.Kx = K_exec(0, 0);
    row.Ky = K_exec(1, 1);
    row.Kz = K_exec(2, 2);
    row.Dx = D_exec(0, 0);
    row.Dy = D_exec(1, 1);
    row.Dz = D_exec(2, 2);
    rows.push_back(row);

    x_pred = x_next;
    v_pred = v_next;
    previous_command_position = collision_sample.p;
  };

  auto append_samples = [&](const char* stage,
                            const std::vector<ImpedanceSample>& flange_samples,
                            const std::vector<ImpedanceSample>& collision_samples) {
    const std::size_t count = std::min(flange_samples.size(), collision_samples.size());
    for (std::size_t i = 0; i < count; ++i) {
      const double dtp = std::max(collision_samples[i].t - t_prev, kMinDt);
      append_row(stage,
                 static_cast<int>(i),
                 flange_samples[i],
                 collision_samples[i],
                 dtp);
      t_prev = collision_samples[i].t;
    }
  };

  append_samples("intended", monitor_plan.intended, collision_plan.intended);
  append_samples("failsafe", monitor_plan.failsafe, collision_plan.failsafe);

  double max_T = 0.0;
  double max_V = 0.0;
  for (const auto& row : rows) {
    if (row.contact_possible) {
      max_T = std::max(max_T, row.T_n_ub);
      max_V = std::max(max_V, row.V_n_ub);
    }
  }

  const double actual_collision_distance =
      human_workspace.signedDistanceToInflatedSphere(
          collision_center,
          inflated_contact_radius,
          wall_time);
  const Vector3d actual_human_center = human_workspace.centerAtTime(wall_time);

  prediction_log_file_ << std::fixed << std::setprecision(9);
  for (const auto& row : rows) {
    const bool is_worst_T =
        row.contact_possible && std::abs(row.T_n_ub - max_T) <= 1.0e-9;
    const bool is_worst_V =
        row.contact_possible && std::abs(row.V_n_ub - max_V) <= 1.0e-9;

    prediction_log_file_
        << wall_time << "," << nominal_guess_time << ","
        << (source == nullptr ? "" : source) << ","
        << monitor_total_ms << ","
        << planner_ms << ","
        << plan_build_ms << ","
        << monitor_eval_ms << ","
        << (executing_last_verified_monitored ? 1 : 0) << ","
        << static_cast<int>(candidate_verified) << ","
        << static_cast<int>(executing_last_verified_monitored) << ","
        << static_cast<int>(monitor.predicted_trigger) << ","
        << static_cast<int>(monitor.monitored_contact_possible) << ","
        << monitor_plan.intended.size() << ","
        << monitor_plan.failsafe.size() << ","
        << row.stage << "," << row.index << ","
        << static_cast<int>(row.failsafe) << ","
        << row.sample_t << "," << row.dt << ","
        << current_position(0) << "," << current_position(1) << "," << current_position(2) << ","
        << collision_center(0) << "," << collision_center(1) << "," << collision_center(2) << ","
        << actual_human_center(0) << "," << actual_human_center(1) << "," << actual_human_center(2) << ","
        << actual_collision_distance << ","
        << row.flange_target_p(0) << "," << row.flange_target_p(1) << "," << row.flange_target_p(2) << ","
        << row.collision_target_p(0) << "," << row.collision_target_p(1) << "," << row.collision_target_p(2) << ","
        << row.x_pred(0) << "," << row.x_pred(1) << "," << row.x_pred(2) << ","
        << row.v_pred(0) << "," << row.v_pred(1) << "," << row.v_pred(2) << ","
        << row.x_next(0) << "," << row.x_next(1) << "," << row.x_next(2) << ","
        << row.v_next(0) << "," << row.v_next(1) << "," << row.v_next(2) << ","
        << row.a_pred(0) << "," << row.a_pred(1) << "," << row.a_pred(2) << ","
        << row.human_center_start(0) << "," << row.human_center_start(1) << "," << row.human_center_start(2) << ","
        << row.human_center_end(0) << "," << row.human_center_end(1) << "," << row.human_center_end(2) << ","
        << row.d_pred << "," << row.d_next << "," << row.d_segment << ","
        << static_cast<int>(row.contact_possible) << ","
        << row.k_n << "," << row.e_n << "," << row.v_n << ","
        << row.T_n_ub << "," << row.V_n_ub << ","
        << static_cast<int>(is_worst_T) << "," << static_cast<int>(is_worst_V) << ","
        << row.Kx << "," << row.Ky << "," << row.Kz << ","
        << row.Dx << "," << row.Dy << "," << row.Dz << ","
        << monitor.worst_case_Tn_ub << ","
        << monitor.worst_case_V_potential_ub << ","
        << monitor.terminal_energy_ub << ","
        << monitor.workspace_distance_margin << ","
        << monitor.h_clamping_energy << "\n";
  }

  ++prediction_log_write_counter_;
  if ((prediction_log_write_counter_ % 50) == 0) {
    prediction_log_file_.flush();
  }
}

void ReachableCartesianImpedanceController::predictionLoggerWorkerLoop() {
  while (true) {
    PredictionLogRecord record;
    {
      std::unique_lock<std::mutex> lock(prediction_log_mutex_);
      prediction_log_cv_.wait(lock, [&]() {
        return !prediction_log_queue_.empty() ||
               !prediction_log_worker_running_.load();
      });

      if (prediction_log_queue_.empty() &&
          !prediction_log_worker_running_.load()) {
        break;
      }

      record = std::move(prediction_log_queue_.front());
      prediction_log_queue_.pop_front();
    }

    writeShieldPredictionTrajectory(
        record.wall_time,
        record.nominal_guess_time,
        record.current_position,
        record.current_orientation,
        record.ee_twist,
        record.inertia,
        record.Jv,
        record.K_runtime,
        record.D_runtime,
        record.human_workspace,
        record.evaluated_plan,
        record.monitor,
        record.candidate_verified,
        record.executing_last_verified_monitored,
        record.monitor_total_ms,
        record.planner_ms,
        record.plan_build_ms,
        record.monitor_eval_ms,
        record.source.c_str());
  }
}

void ReachableCartesianImpedanceController::startPredictionLoggerWorker() {
  if (!enable_prediction_logging_ || !prediction_log_file_.is_open() ||
      prediction_log_worker_running_.load()) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(prediction_log_mutex_);
    prediction_log_queue_.clear();
  }

  prediction_log_worker_running_.store(true);
  prediction_log_worker_thread_ =
      std::thread(&ReachableCartesianImpedanceController::predictionLoggerWorkerLoop, this);
}

void ReachableCartesianImpedanceController::stopPredictionLoggerWorker() {
  if (!prediction_log_worker_running_.exchange(false)) {
    return;
  }

  prediction_log_cv_.notify_all();
  if (prediction_log_worker_thread_.joinable()) {
    prediction_log_worker_thread_.join();
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

  const double max_delta = panda_limits::kTorqueRateLimit * std::max(dt, kMinDt);
  torque_rate_limited_last_ = false;
  torque_rate_max_desired_delta_nm_last_ = 0.0;
  torque_rate_limit_delta_nm_last_ = max_delta;
  torque_rate_max_excess_nm_last_ = 0.0;
  torque_rate_max_ratio_last_ = 0.0;
  torque_rate_max_cmd_delta_nm_last_ = 0.0;

  Vector7d tau_cmd = tau_cmd_prev_;

  for (int i = 0; i < 7; ++i) {
    const double desired_delta = tau_des(i) - tau_cmd_prev_(i);
    const double abs_desired_delta = std::abs(desired_delta);
    torque_rate_max_desired_delta_nm_last_ =
        std::max(torque_rate_max_desired_delta_nm_last_, abs_desired_delta);
    torque_rate_max_ratio_last_ =
        std::max(torque_rate_max_ratio_last_,
                 abs_desired_delta / std::max(max_delta, kSmallPositive));
    torque_rate_max_excess_nm_last_ =
        std::max(torque_rate_max_excess_nm_last_, abs_desired_delta - max_delta);
    if (abs_desired_delta > max_delta) {
      torque_rate_limited_last_ = true;
    }

    const double delta = std::clamp(desired_delta, -max_delta, max_delta);

    tau_cmd(i) = tau_cmd_prev_(i) + delta;
    torque_rate_max_cmd_delta_nm_last_ =
        std::max(torque_rate_max_cmd_delta_nm_last_, std::abs(delta));
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
  last_verified_command_stage_ = 0;
  last_verified_command_index_ = 0;

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
  const auto toc_model = SteadyClock::now();

  acceptPendingCartesianViaPoints(current_position, current_orientation, wall_time);

  double paused_total = paused_nominal_time_sec_;
  if (failsafe_enter_wall_time_sec_ >= 0.0)
    paused_total += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
  if (cartesian_effective_time_freeze_start_wall_time_ >= 0.0)
    paused_total += std::max(
        0.0,
        wall_time - cartesian_effective_time_freeze_start_wall_time_);
  const double nominal_guess_time = std::max(0.0, wall_time - paused_total);

  ShieldDecision shield_dec;

  if (async_safety_monitor_) {
    const bool publish_monitor_input =
        last_async_input_publish_wall_time_ < 0.0 ||
        (wall_time - last_async_input_publish_wall_time_) >=
            std::max(monitor_update_period_sec_, kMinDt);

    if (publish_monitor_input) {
      AsyncMonitorInput async_input;
      async_input.sequence = async_input_sequence_.fetch_add(1) + 1;
      async_input.control_loop_sequence = control_loop_sequence;
      async_input.wall_time = wall_time;
      async_input.nominal_guess_time = nominal_guess_time;
      async_input.current_position = current_position;
      async_input.current_orientation = current_orientation;
      async_input.ee_twist = ee_twist;
      async_input.inertia = inertia;
      async_input.Jv = Jv_collision;
      async_input.K_runtime = K_runtime_;
      async_input.D_runtime = D_runtime_;
      async_input.human_workspace = human_workspace_;
      async_input.human_workspace_active = human_workspace_active_;
      async_input.last_commanded_sample = last_commanded_sample_;
      async_input.last_commanded_sample_valid = last_commanded_sample_valid_;
      async_input.commanded_path_time = commanded_path_time_;
      async_input.commanded_path_rate = commanded_path_rate_;
      publishAsyncMonitorInput(async_input);
      last_async_input_publish_wall_time_ = wall_time;
    }

    AsyncMonitorOutput async_output;
    if (takeAsyncMonitorOutput(&async_output)) {
      const std::size_t async_plan_elapsed_steps =
          control_loop_sequence >= async_output.input.control_loop_sequence
              ? static_cast<std::size_t>(
                    control_loop_sequence -
                    async_output.input.control_loop_sequence)
              : 0;
      last_shield_decision_ = async_output.decision;
      last_shield_decision_valid_ = true;
      last_async_output_wall_time_ = async_output.input_wall_time;
      last_async_output_valid_ = true;
      if (async_output.decision.has_evaluated_plan) {
        logShieldPredictionTrajectory(
            async_output.input.wall_time,
            async_output.input.nominal_guess_time,
            async_output.input.current_position,
            async_output.input.current_orientation,
            async_output.input.ee_twist,
            async_output.input.inertia,
            async_output.input.Jv,
            async_output.input.K_runtime,
            async_output.input.D_runtime,
            async_output.input.human_workspace,
            async_output.decision.evaluated_plan,
            async_output.decision.monitor,
            async_output.decision.candidate_verified,
            async_output.decision.executing_last_verified_monitored,
            async_output.decision.monitor_total_ms,
            async_output.decision.planner_ms,
            async_output.decision.plan_build_ms,
            async_output.decision.monitor_eval_ms,
            "async");
      }
      if (async_output.decision.candidate_verified &&
          async_output.decision.evaluated_plan.valid) {
        last_verified_plan_ = async_output.decision.evaluated_plan;
        last_verified_plan_.valid = true;
        alignVerifiedPlanExecutionIndex(
            &last_verified_plan_,
            async_plan_elapsed_steps);
        if (!cartesian_effective_time_frozen_) {
          const double previous_path_time = commanded_path_time_;
          if (last_verified_plan_.nominal_time_anchor > 0.0) {
            commanded_path_time_ =
                std::max(commanded_path_time_, last_verified_plan_.nominal_time_anchor);
          }
          if (!last_verified_plan_.intended.empty()) {
            const double plan_duration =
                std::max(local_replan_dt_,
                         static_cast<double>(last_verified_plan_.intended.size()) *
                             local_replan_dt_);
            if (last_verified_plan_.nominal_time_anchor > previous_path_time + kMinDt) {
              commanded_path_rate_ = std::clamp(
                  (last_verified_plan_.nominal_time_anchor - previous_path_time) /
                      plan_duration,
                  path_time_rate_min_,
                  path_time_rate_max_);
            } else {
              bool active_path_finished = false;
              {
                std::lock_guard<std::mutex> lock(cartesian_via_point_path_mutex_);
                active_path_finished =
                    !cartesian_via_point_path_.empty() &&
                    commanded_path_time_ >= cartesian_via_point_path_.back().t - kMinDt;
              }
              if (active_path_finished) {
                commanded_path_rate_ = 0.0;
              }
            }
          }
        }
      }
    }

    const bool output_is_fresh =
        last_async_output_valid_ &&
        last_shield_decision_valid_ &&
        (async_plan_max_age_sec_ <= 0.0 ||
         (wall_time - last_async_output_wall_time_) <= async_plan_max_age_sec_);

    if (output_is_fresh) {
      shield_dec = last_shield_decision_;
      const bool async_releases_failsafe =
          mode_ == SafetyMode::kLastVerifiedMonitored &&
          !shield_dec.executing_last_verified_monitored &&
          !shouldRejectCandidateWithMonitor(shield_dec.monitor);

      if (async_releases_failsafe) {
        shield_dec = computeShieldDecision(wall_time, nominal_guess_time,
                                           current_position, current_orientation,
                                           ee_twist, inertia, Jv_collision);
        last_shield_decision_ = shield_dec;
        last_shield_decision_valid_ = true;
        if (shield_dec.has_evaluated_plan) {
          logShieldPredictionTrajectory(
              wall_time,
              nominal_guess_time,
              current_position,
              current_orientation,
              ee_twist,
              inertia,
              Jv_collision,
              K_runtime_,
              D_runtime_,
              human_workspace_,
              shield_dec.evaluated_plan,
              shield_dec.monitor,
              shield_dec.candidate_verified,
              shield_dec.executing_last_verified_monitored,
              shield_dec.monitor_total_ms,
              shield_dec.planner_ms,
              shield_dec.plan_build_ms,
              shield_dec.monitor_eval_ms,
              "async_failsafe_release_sync");
        }
      } else if (shield_dec.executing_last_verified_monitored ||
                 shouldRejectCandidateWithMonitor(shield_dec.monitor)) {
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
        if (last_verified_plan_.valid && !last_verified_plan_.intended.empty()) {
          shield_dec.command =
              getNextVerifiedTrajectoryCommandFromCache(!cartesian_effective_time_frozen_);
        } else {
          shield_dec.command =
              makeEmergencyStopCommand(current_position, current_orientation, wall_time);
        }
      }
    } else {
      const bool need_sync_bootstrap =
          !last_verified_plan_.valid || last_verified_plan_.intended.empty();

      if (need_sync_bootstrap) {
        shield_dec = computeShieldDecision(wall_time, nominal_guess_time,
                                           current_position, current_orientation,
                                           ee_twist, inertia, Jv_collision);
        last_shield_decision_ = shield_dec;
        last_shield_decision_valid_ = true;
        if (shield_dec.has_evaluated_plan) {
          logShieldPredictionTrajectory(
              wall_time,
              nominal_guess_time,
              current_position,
              current_orientation,
              ee_twist,
              inertia,
              Jv_collision,
              K_runtime_,
              D_runtime_,
              human_workspace_,
              shield_dec.evaluated_plan,
              shield_dec.monitor,
              shield_dec.candidate_verified,
              shield_dec.executing_last_verified_monitored,
              shield_dec.monitor_total_ms,
              shield_dec.planner_ms,
              shield_dec.plan_build_ms,
              shield_dec.monitor_eval_ms,
              "async_bootstrap_sync");
        }
      } else {
        shield_dec = last_shield_decision_;
        if (mode_ == SafetyMode::kLastVerifiedMonitored ||
            shield_dec.executing_last_verified_monitored ||
            shouldRejectCandidateWithMonitor(shield_dec.monitor)) {
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
              getNextVerifiedTrajectoryCommandFromCache(!cartesian_effective_time_frozen_);
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
      shield_dec = computeShieldDecision(wall_time, nominal_guess_time,
                                         current_position, current_orientation,
                                         ee_twist, inertia, Jv_collision);
      last_shield_decision_ = shield_dec;
      last_shield_decision_valid_ = true;
      if (shield_dec.has_evaluated_plan) {
        logShieldPredictionTrajectory(
            wall_time,
            nominal_guess_time,
            current_position,
            current_orientation,
            ee_twist,
            inertia,
            Jv_collision,
            K_runtime_,
            D_runtime_,
            human_workspace_,
            shield_dec.evaluated_plan,
            shield_dec.monitor,
            shield_dec.candidate_verified,
            shield_dec.executing_last_verified_monitored,
            shield_dec.monitor_total_ms,
            shield_dec.planner_ms,
            shield_dec.plan_build_ms,
            shield_dec.monitor_eval_ms,
            "sync");
      }
    } else {
      shield_dec = last_shield_decision_;
      if (mode_ == SafetyMode::kLastVerifiedMonitored || shield_dec.executing_last_verified_monitored) {
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
          shield_dec.command =
              getNextVerifiedTrajectoryCommandFromCache(!cartesian_effective_time_frozen_);
        else
          shield_dec.command = makeEmergencyStopCommand(current_position, current_orientation, wall_time);
      }
      last_shield_decision_.command = shield_dec.command;
      last_shield_decision_.executing_last_verified_monitored =
          shield_dec.executing_last_verified_monitored;
    }
  }
  const auto toc_shield = SteadyClock::now();

  if (shield_dec.executing_last_verified_monitored) {
    if (failsafe_enter_wall_time_sec_ < 0.0) {
      failsafe_enter_wall_time_sec_ = wall_time;
      failsafe_start_time_sec_ = wall_time;
    }
    mode_ = SafetyMode::kLastVerifiedMonitored;
  } else {
    if (failsafe_enter_wall_time_sec_ >= 0.0) {
      paused_nominal_time_sec_ += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
      failsafe_enter_wall_time_sec_ = -1.0;
    }
    mode_ = nominalSafetyModeForMonitor(shield_dec.monitor);
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
  MonitorResult monitor = shield_dec.monitor;

  CartesianEnergyBudgetInfo cartesian_energy_info;
  const bool use_cartesian_energy_budget =
      shouldApplyCartesianEnergyBudget(
          monitor,
          shield_dec.command,
          shield_dec.executing_last_verified_monitored);
  Matrix6d cartesian_lambda = Matrix6d::Zero();
  bool cartesian_lambda_valid = false;
  if (use_cartesian_energy_budget) {
    const bool refresh_lambda =
        !cartesian_energy_lambda_cache_valid_ ||
        cartesian_energy_lambda_update_period_sec_ <= kMinDt ||
        wall_time - cartesian_energy_lambda_cache_wall_time_ >=
            cartesian_energy_lambda_update_period_sec_;
    if (refresh_lambda) {
      Matrix6d refreshed_lambda = Matrix6d::Zero();
      if (computeTaskInertia(inertia, J_geo, &refreshed_lambda)) {
        cartesian_energy_lambda_cache_ = refreshed_lambda;
        cartesian_energy_lambda_cache_valid_ = true;
        cartesian_energy_lambda_cache_wall_time_ = wall_time;
      }
    }
    if (cartesian_energy_lambda_cache_valid_) {
      cartesian_lambda = cartesian_energy_lambda_cache_;
      cartesian_lambda_valid = true;
    }
  } else {
    cartesian_energy_lambda_cache_valid_ = false;
    cartesian_energy_lambda_cache_wall_time_ = -1.0;
  }

  auto compute_error_for_command = [&](const ImpedanceSample& command) {
    Vector6d command_error = Vector6d::Zero();
    command_error.head<3>() = current_position - command.p;
    command_error.tail<3>() =
        computeOrientationError(current_orientation, command.q);
    return command_error;
  };

  const ImpedanceSample candidate_budget_command = shield_dec.command;
  const Vector6d candidate_error =
      compute_error_for_command(candidate_budget_command);
  CartesianEnergyBudgetInfo candidate_energy_info;
  ImpedanceSample candidate_scaled_command = applyCartesianEnergyBudget(
      candidate_budget_command,
      candidate_error,
      ee_twist,
      cartesian_lambda,
      cartesian_lambda_valid,
      use_cartesian_energy_budget,
      &candidate_energy_info);

  // Lachner et al. Eq. (14)-(15): when the unscaled controlled
  // energy exceeds the budget, scale the impedance and hold effective
  // trajectory time until the energy is back inside the budget.
  const bool cartesian_budget_limited =
      use_cartesian_energy_budget &&
      candidate_energy_info.scale < 1.0 - 1.0e-6;

  Vector6d error = candidate_error;
  cartesian_energy_info = candidate_energy_info;
  shield_dec.command = candidate_scaled_command;

  if (cartesian_budget_limited) {
    if (!cartesian_effective_time_frozen_) {
      cartesian_effective_time_freeze_start_wall_time_ = wall_time;
      commanded_path_rate_ = 0.0;
      const ImpedanceSample& hold_source =
          last_commanded_sample_valid_ ? last_commanded_sample_
                                       : candidate_budget_command;
      cartesian_effective_time_hold_sample_ =
          makeEffectiveTimeHoldSample(hold_source);
      cartesian_effective_time_hold_sample_valid_ = true;
    }
    cartesian_effective_time_frozen_ = true;

    if (!cartesian_effective_time_hold_sample_valid_) {
      cartesian_effective_time_hold_sample_ =
          makeEffectiveTimeHoldSample(candidate_budget_command);
      cartesian_effective_time_hold_sample_valid_ = true;
    }

    const ImpedanceSample hold_command = cartesian_effective_time_hold_sample_;
    error = compute_error_for_command(hold_command);
    shield_dec.command = applyCartesianEnergyBudget(
        hold_command,
        error,
        ee_twist,
        cartesian_lambda,
        cartesian_lambda_valid,
        use_cartesian_energy_budget,
        &cartesian_energy_info);
  } else {
    if (cartesian_effective_time_frozen_ &&
        cartesian_effective_time_freeze_start_wall_time_ >= 0.0) {
      paused_nominal_time_sec_ += std::max(
          0.0,
          wall_time - cartesian_effective_time_freeze_start_wall_time_);
      commanded_path_time_ = nominal_guess_time;
      commanded_path_rate_ = 0.0;
    }
    cartesian_effective_time_frozen_ = false;
    cartesian_effective_time_freeze_start_wall_time_ = -1.0;
    cartesian_effective_time_hold_sample_valid_ = false;
  }

  last_cartesian_energy_budget_active_ = cartesian_energy_info.active;
  last_cartesian_energy_budget_lambda_valid_ = cartesian_energy_info.lambda_valid;
  last_cartesian_energy_scale_ = cartesian_energy_info.scale;
  last_cartesian_kinetic_energy_ = cartesian_energy_info.kinetic_energy;
  last_cartesian_potential_energy_ = cartesian_energy_info.potential_energy;
  last_cartesian_control_energy_ = cartesian_energy_info.total_energy;

  const Vector3d desired_position_cur = shield_dec.command.p;
  const Quaterniond desired_orientation_cur = shield_dec.command.q;
  const Vector3d desired_linear_velocity_cur = shield_dec.command.dp;

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

  {
    ImpedanceSample current_command = shield_dec.command;
    current_command.p =
        desired_position_cur + collisionCenterOffsetWorld(desired_orientation_cur);
    const Vector3d monitor_direction =
        commandMonitorDirection(
            current_command,
            current_command.p,
            collision_center,
            ee_collision_twist.head<3>(),
            human_workspace_,
            wall_time);
    Matrix3d task_inertia_inv = Jv_collision * inertia.inverse() * Jv_collision.transpose();
    task_inertia_inv.diagonal().array() += kSmallPositive;
    const double denom =
        (monitor_direction.transpose() * task_inertia_inv * monitor_direction)(0, 0);
    const double m_eff_dir = 1.0 / std::max(denom, kSmallPositive);
    const double v_n_now_tube =
        std::abs(monitor_direction.dot(ee_collision_twist.head<3>())) +
        monitor.current_vel_error_radius;
    const double Tn_now_tube =
        0.5 * std::max(m_eff_dir, 0.0) * v_n_now_tube * v_n_now_tube;
    monitor.m_eff_n = m_eff_dir;
    monitor.v_n_now = monitor_direction.dot(ee_collision_twist.head<3>());
    monitor.Tn_now = 0.5 * std::max(m_eff_dir, 0.0) *
                     monitor.v_n_now * monitor.v_n_now;
    monitor.v_n_now_tube = v_n_now_tube;
    monitor.Tn_now_tube  = Tn_now_tube;
    if (mode_ == SafetyMode::kLastVerifiedMonitored) {
      if (prev_Tn_fs_valid_)
        monitor.Tn_dot_est = (Tn_now_tube - prev_Tn_fs_) / std::max(dt, kMinDt);
      else
        monitor.Tn_dot_est = 0.0;
      prev_Tn_fs_ = Tn_now_tube; prev_Tn_fs_valid_ = true;
    } else {
      monitor.Tn_dot_est = 0.0;
      prev_Tn_fs_valid_ = false;
    }
  }

  last_commanded_sample_ = shield_dec.command;
  last_commanded_sample_valid_ = true;

  const Vector7d tau_cmd = computeImpedanceTorque(
      q, dq, inertia, coriolis, J_geo,
      current_position, current_orientation,
      shield_dec.command, dt);
  const auto toc_torque = SteadyClock::now();

  for (int i = 0; i < kNumJoints; ++i) command_interfaces_[i].set_value(tau_cmd(i));

  updateCartesianViaPointsActionStatus(
      current_position,
      current_orientation,
      wall_time);

  if (enable_error_logging_ && command_recording_active_ && error_log_file_.is_open()) {
    const Vector3d human_center = human_workspace_.centerAtTime(wall_time);
    const double mujoco_contact_msg_time =
        latest_mujoco_contact_msg_time_.load(std::memory_order_relaxed);
    const std::uint64_t mujoco_contact_sequence =
        latest_mujoco_contact_sequence_.load(std::memory_order_relaxed);
    const bool mujoco_contact_new_sample =
        mujoco_contact_sequence != last_logged_mujoco_contact_sequence_;
    const std::uint64_t mujoco_contact_samples_since_last_log =
        (mujoco_contact_sequence >= last_logged_mujoco_contact_sequence_)
            ? (mujoco_contact_sequence - last_logged_mujoco_contact_sequence_)
            : 0;
    const double mujoco_contact_sample_age =
        (mujoco_contact_msg_time >= 0.0)
            ? std::max(0.0, this->get_node()->now().seconds() - mujoco_contact_msg_time)
            : -1.0;
    const double verified_plan_age_sec =
        last_verified_plan_.valid
            ? std::max(0.0, wall_time - last_verified_plan_.generated_wall_time)
            : -1.0;

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
        << human_center(0) << "," << human_center(1) << "," << human_center(2) << ","
        << ee_collision_twist(0) << "," << ee_collision_twist(1) << "," << ee_collision_twist(2) << ","
        << desired_linear_velocity_cur(0) << "," << desired_linear_velocity_cur(1) << "," << desired_linear_velocity_cur(2) << ","
        << error(0) << "," << error(1) << "," << error(2) << ","
        << error(3) << "," << error(4) << "," << error(5) << ","
        << 0.0 << "," << 0.0 << "," << 0.0 << "," << tau_cmd.norm() << ","
        << static_cast<int>(torque_rate_limited_last_) << ","
        << torque_rate_max_desired_delta_nm_last_ << ","
        << torque_rate_limit_delta_nm_last_ << ","
        << torque_rate_max_excess_nm_last_ << ","
        << torque_rate_max_ratio_last_ << ","
        << torque_rate_max_cmd_delta_nm_last_ << ","
        << K_runtime_(0, 0) << "," << K_runtime_(1, 1) << "," << K_runtime_(2, 2) << ","
        << D_runtime_(0, 0) << "," << D_runtime_(1, 1) << "," << D_runtime_(2, 2) << ","
        << monitor.workspace_distance_now << "," << monitor.workspace_distance_min << ","
        << monitor.m_eff_n << "," << monitor.v_n_now << "," << monitor.Tn_now << "," << monitor.v_safe << ","
        << static_cast<int>(monitor.nominal_contact_sample_found) << ","
        << monitor.nominal_contact_time << "," << monitor.nominal_contact_distance << ","
        << monitor.v_n_contact_nominal << "," << monitor.Tn_contact_nominal << ","
        << monitor.worst_case_contact_time << "," << monitor.worst_case_workspace_distance_at_candidate << ","
        << monitor.worst_case_nominal_forward_progress << ","
        << monitor.worst_case_v_n_ub << "," << monitor.worst_case_Tn_ub << ","
        << monitor.worst_case_a_pos << "," << monitor.worst_case_a_brake << "," << monitor.worst_case_a_net << ","
        << monitor.workspace_distance_margin << "," << monitor.h_monitored_energy << ","
        << monitor.h_clamping_energy << "," << monitor.h_terminal_energy << ","
        << monitor.worst_case_V_potential_ub << "," << monitor.terminal_energy_ub << ","
        << robot_potential_energy << ","
        << robot_potential_energy_pos << ","
        << robot_potential_energy_rot << ","
        << monitor.v_n_now_tube << "," << monitor.Tn_now_tube << "," << monitor.Tn_dot_est << ","
        << monitor.current_pos_error_radius << "," << monitor.current_vel_error_radius << ","
        << monitor.worst_case_pos_error_radius << "," << monitor.worst_case_vel_error_radius << ","
        << static_cast<int>(monitor.monitored_contact_possible) << ","
        << static_cast<int>(monitor.contact_relevant_for_energy) << ","
        << static_cast<int>(monitor.collision_energy_unsafe) << ","
        << static_cast<int>(monitor.clamping_energy_unsafe) << ","
        << static_cast<int>(monitor.terminal_energy_unsafe) << ","
        << static_cast<int>(monitor.monitored_unsafe) << ","
        << static_cast<int>(monitor.predicted_trigger) << ","
        << monitored_steps << ","
        << monitored_intended_steps << ","
        << monitored_failsafe_steps << ","
        << control_loop_sequence << ","
        << verified_plan_age_sec << ","
        << last_verified_command_stage_ << ","
        << last_verified_command_index_ << ","
        << last_verified_plan_.intended_exec_index << ","
        << last_verified_plan_.failsafe_exec_index << ","
        << static_cast<int>(last_cartesian_energy_budget_active_) << ","
        << static_cast<int>(cartesian_effective_time_frozen_) << ","
        << static_cast<int>(last_cartesian_energy_budget_lambda_valid_) << ","
        << last_cartesian_energy_scale_ << ","
        << last_cartesian_kinetic_energy_ << ","
        << last_cartesian_potential_energy_ << ","
        << last_cartesian_control_energy_ << ","
        << energy_budget_joule_ << ","
        << latest_mujoco_contact_value_.load(std::memory_order_relaxed) << ","
        << static_cast<int>(latest_mujoco_contact_active_.load(std::memory_order_relaxed)) << ","
        << mujoco_contact_msg_time << ","
        << mujoco_contact_sequence << ","
        << static_cast<int>(mujoco_contact_new_sample) << ","
        << mujoco_contact_samples_since_last_log << ","
        << mujoco_contact_sample_age << ","
        << static_cast<int>(mujoco_first_contact_seen_.load(std::memory_order_relaxed)) << ","
        << mujoco_first_contact_wall_time_.load(std::memory_order_relaxed) << ","
        << mujoco_first_contact_msg_time_.load(std::memory_order_relaxed) << "\n";
    last_logged_mujoco_contact_sequence_ = mujoco_contact_sequence;
    ++log_write_counter_;
    if ((log_write_counter_ % 200) == 0) error_log_file_.flush();
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
                "[reachable_impedance] mode=%d avg=%.3f ms min=%.3f ms max=%.3f ms overruns>1ms=%zu >2ms=%zu "
                "model_avg/max=%.3f/%.3f shield_avg/max=%.3f/%.3f torque_avg/max=%.3f/%.3f io_avg/max=%.3f/%.3f "
                "plan_valid=%d",
                static_cast<int>(mode_),
                exec_sum_ms_ / n, exec_min_ms_, exec_max_ms_,
                exec_overrun_1ms_count_, exec_overrun_2ms_count_,
                prof_model_sum_ms_ / n, prof_model_max_ms_,
                prof_shield_sum_ms_ / n, prof_shield_max_ms_,
                prof_torque_sum_ms_ / n, prof_torque_max_ms_,
                prof_io_sum_ms_ / n, prof_io_max_ms_,
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
    auto_declare<bool>("enable_prediction_logging", true);
    auto_declare<std::string>(
        "prediction_log_file_name",
        "shield_prediction_trajectory.csv");
    auto_declare<int>("prediction_log_max_queue_size", 256);

    auto_declare<std::string>(
        "cartesian_via_points_topic",
        "cartesian_via_points");
    auto_declare<std::string>(
        "startup_via_points_source",
        "yaml");
    auto_declare<std::string>(
        "cartesian_via_points_action_name",
        "~/follow_cartesian_via_points");
    auto_declare<double>(
        "cartesian_via_points_action_feedback_period_sec",
        0.1);
    const std::string startup_via_points_source =
        get_node()->get_parameter("startup_via_points_source").as_string();
    if (startup_via_points_source != "action") {
      auto_declare<std::vector<double>>(
          "cartesian_via_points",
          std::vector<double>{});
      auto_declare<std::vector<double>>(
          "cartesian_via_point_quaternions",
          std::vector<double>{});
    }

    auto_declare<double>("nominal_pos_stiffness", 400.0);
    auto_declare<double>("nominal_rot_stiffness", 20.0);
    auto_declare<double>("failsafe_pos_stiffness", 50.0);
    auto_declare<double>("failsafe_rot_stiffness", 5.0);
    auto_declare<double>("failsafe_pos_damping_scale", 2.5);
    auto_declare<double>("failsafe_rot_damping_scale", 2.5);

    auto_declare<double>("n_stiffness", 0.0);
    auto_declare<bool>("disable_nullspace_in_failsafe", true);
    auto_declare<bool>("enable_safety_monitor", true);

    auto_declare<double>("energy_budget_joule", 0.05);
    auto_declare<double>("energy_budget_margin_joule", 0.005);
    auto_declare<double>("cartesian_energy_min_pos_stiffness", 0.0);
    auto_declare<double>("cartesian_energy_lambda_update_period_sec", 0.004);
    auto_declare<double>("contact_activation_margin", 0.0);
    auto_declare<double>("ee_collision_radius", 0.04);
    auto_declare<std::vector<double>>(
        "tcp_offset", std::vector<double>{0.0, 0.0, 0.0});
    auto_declare<std::vector<double>>(
        "ee_collision_center_offset", std::vector<double>{0.0, 0.0, 0.0});
    auto_declare<bool>("async_safety_monitor", true);
    auto_declare<double>("async_plan_max_age_sec", 0.05);

    cps_human_workspace::HumanWorkspace::declareParameters(get_node());
    auto_declare<std::string>("human_workspace_topic", "human_workspace/state");
    auto_declare<double>("human_workspace_timeout_sec", 0.5);

    auto_declare<double>("tracking_pos_error_bound", 0.005);
    auto_declare<double>("tracking_vel_error_bound", 0.05);

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
    const auto prediction_log_max_queue_size_param =
        get_node()->get_parameter("prediction_log_max_queue_size").as_int();
    prediction_log_max_queue_size_ = static_cast<std::size_t>(
        std::max<int64_t>(1, prediction_log_max_queue_size_param));

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

    energy_budget_joule_ =
        std::max(0.0, get_node()->get_parameter("energy_budget_joule").as_double());

    energy_budget_margin_joule_ =
        std::max(0.0, get_node()->get_parameter("energy_budget_margin_joule").as_double());
    cartesian_energy_min_pos_stiffness_ =
        std::max(0.0, get_node()->get_parameter("cartesian_energy_min_pos_stiffness").as_double());
    cartesian_energy_lambda_update_period_sec_ =
        std::max(0.0, get_node()->get_parameter("cartesian_energy_lambda_update_period_sec").as_double());
    contact_activation_margin_ =
        std::max(0.0, get_node()->get_parameter("contact_activation_margin").as_double());

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
    async_safety_monitor_ = get_node()->get_parameter("async_safety_monitor").as_bool();
    async_plan_max_age_sec_ =
        std::max(0.0, get_node()->get_parameter("async_plan_max_age_sec").as_double());

    tracking_pos_error_bound_ = std::max(0.0, get_node()->get_parameter("tracking_pos_error_bound").as_double());
    tracking_vel_error_bound_ = std::max(0.0, get_node()->get_parameter("tracking_vel_error_bound").as_double());

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
    path_time_rate_target_ =
        std::clamp(trajectory_settings.path_time_rate_target,
                   path_time_rate_min_,
                   path_time_rate_max_);

    local_replan_horizon_steps_ =
        std::max(1, trajectory_settings.local_replan_horizon_steps);
    local_replan_dt_ = std::max(trajectory_settings.local_replan_dt, kMinDt);
    shield_intended_steps_ = std::max(
        1,
        static_cast<int>(
            std::llround(std::max(monitor_update_period_sec_, local_replan_dt_) /
                         local_replan_dt_)));
    monitor_decimation_ = shield_intended_steps_;
    local_path_lookahead_sec_ =
        std::max(trajectory_settings.local_path_lookahead_sec, local_replan_dt_);
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
        if (!human_workspace_.configureFromConfigFile(
                human_workspace_config_path,
                get_node()->get_logger())) {
          return CallbackReturn::ERROR;
        }
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

    K_nominal_.setZero(); D_nominal_.setZero();
    K_f_target_.setZero(); D_f_target_.setZero();
    K_nominal_.topLeftCorner<3, 3>() = nominal_pos_stiffness * Matrix3d::Identity();
    K_nominal_.bottomRightCorner<3, 3>() = nominal_rot_stiffness * Matrix3d::Identity();
    D_nominal_.topLeftCorner<3, 3>() = 0.8 * 2.0 * std::sqrt(std::max(nominal_pos_stiffness, 0.0)) * Matrix3d::Identity();
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
  last_commanded_sample_.K = K_nominal_;
  last_commanded_sample_.D = D_nominal_;
  last_commanded_sample_.failsafe = false;
  last_commanded_sample_valid_ = true;
  commanded_path_rate_ = 0.0;

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
  command_recording_active_ = false;
  prediction_log_write_counter_ = 0;
  control_update_sequence_ = 0;
  prev_Tn_fs_ = 0.0; prev_Tn_fs_valid_ = false;
  latest_mujoco_contact_value_.store(0.0);
  latest_mujoco_contact_msg_time_.store(-1.0);
  latest_mujoco_contact_active_.store(false);
  latest_mujoco_contact_sequence_.store(0);
  last_logged_mujoco_contact_sequence_ = 0;
  mujoco_first_contact_seen_.store(false);
  mujoco_first_contact_wall_time_.store(-1.0);
  mujoco_first_contact_msg_time_.store(-1.0);

  monitor_counter_ = 0;
  last_cartesian_energy_budget_active_ = false;
  last_cartesian_energy_budget_lambda_valid_ = false;
  last_cartesian_energy_scale_ = 1.0;
  last_cartesian_kinetic_energy_ = 0.0;
  last_cartesian_potential_energy_ = 0.0;
  last_cartesian_control_energy_ = 0.0;
  cartesian_energy_lambda_cache_.setZero();
  cartesian_energy_lambda_cache_valid_ = false;
  cartesian_energy_lambda_cache_wall_time_ = -1.0;
  cartesian_effective_time_frozen_ = false;
  cartesian_effective_time_freeze_start_wall_time_ = -1.0;
  cartesian_effective_time_hold_sample_ = ImpedanceSample{};
  cartesian_effective_time_hold_sample_valid_ = false;

  last_verified_plan_ = VerifiedPlan{};
  last_verified_command_stage_ = 0;
  last_verified_command_index_ = 0;

  commanded_path_time_ = 0.0;

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
                    << "enable_prediction_logging: "
                    << static_cast<int>(enable_prediction_logging_) << "\n"
                    << "prediction_log_max_queue_size: "
                    << prediction_log_max_queue_size_ << "\n"
                    << "recording_start: first_valid_via_points_command\n"
                    << "arm_id: " << arm_id_ << "\n"
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
                    << "async_plan_max_age_sec: " << async_plan_max_age_sec_ << "\n"
                    << "monitor_frequency_hz: " << monitor_frequency_hz_ << "\n"
                    << "monitor_update_period_sec: " << monitor_update_period_sec_ << "\n"
                    << "monitor_decimation: " << monitor_decimation_ << "\n"
                    << "shield_intended_steps: " << shield_intended_steps_ << "\n"
                    << "trajectory_generator_config_path: "
                    << trajectory_generator_config_path_ << "\n"
                    << "shield_plan_dt: " << shield_plan_dt_ << "\n"
                    << "monitor_sparse_dt: " << shield_plan_dt_ << "\n"
                    << "local_replan_dt: " << local_replan_dt_ << "\n"
                    << "failsafe_plan_dt: "
                    << std::max(shield_plan_dt_, local_replan_dt_) << "\n"
                    << "failsafe_command_dt: " << local_replan_dt_ << "\n"
                    << "energy_budget_joule: "
                    << energy_budget_joule_ << "\n"
                    << "cartesian_energy_min_pos_stiffness: "
                    << cartesian_energy_min_pos_stiffness_ << "\n"
                    << "cartesian_energy_lambda_update_period_sec: "
                    << cartesian_energy_lambda_update_period_sec_ << "\n"
                    << "cartesian_effective_time_freeze_enabled: 1\n"
                    << "contact_activation_margin: "
                    << contact_activation_margin_ << "\n"
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
                    << "human_workspace_direction: ["
                    << human_workspace_.direction().x() << ", "
                    << human_workspace_.direction().y() << ", "
                    << human_workspace_.direction().z() << "]\n"
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
                    << "human_motion_radius: " << human_workspace_.motionRadius() << "\n"
                    << "human_hand_radius: " << human_workspace_.handRadius() << "\n"
                    << "human_workspace_config_path: "
                    << human_workspace_config_path << "\n";
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
        << "human_center_px,human_center_py,human_center_pz,"
        << "collision_center_vx,collision_center_vy,collision_center_vz,"
        << "des_vx,des_vy,des_vz,"
        << "err_px,err_py,err_pz,err_rx,err_ry,err_rz,"
        << "tau_task_norm,tau_null_norm,tau_friction_norm,tau_cmd_norm,"
        << "torque_rate_limited,torque_rate_max_desired_delta_nm,"
        << "torque_rate_limit_delta_nm,torque_rate_max_excess_nm,"
        << "torque_rate_max_ratio,torque_rate_max_cmd_delta_nm,"
        << "Kx,Ky,Kz,Dx,Dy,Dz,"
        << "workspace_distance_now,workspace_distance_min,"
        << "m_eff_n,v_n_now,Tn_now,v_safe,"
        << "nominal_contact_sample_found,nominal_contact_time,nominal_contact_distance,"
        << "v_n_contact_nominal,Tn_contact_nominal,"
        << "worst_case_contact_time,worst_case_workspace_distance_at_candidate,"
        << "worst_case_nominal_forward_progress,"
        << "worst_case_v_n_ub,worst_case_Tn_ub,"
        << "worst_case_a_pos,worst_case_a_brake,worst_case_a_net,"
        << "workspace_distance_margin,h_monitored_energy,h_clamping_energy,h_terminal_energy,"
        << "worst_case_V_potential_ub,terminal_energy_ub,"
        << "robot_potential_energy,robot_potential_energy_pos,robot_potential_energy_rot,"
        << "v_n_now_tube,Tn_now_tube,Tn_dot_est,"
        << "current_pos_error_radius,current_vel_error_radius,"
        << "worst_case_pos_error_radius,worst_case_vel_error_radius,"
        << "monitored_contact_possible,contact_relevant_for_energy,"
        << "collision_energy_unsafe,clamping_energy_unsafe,"
        << "terminal_energy_unsafe,monitored_unsafe,predicted_trigger,"
        << "monitored_steps,monitored_intended_steps,monitored_failsafe_steps,"
        << "control_loop_sequence,verified_plan_age_sec,"
        << "verified_command_stage,verified_command_index,"
        << "verified_next_intended_exec_index,verified_next_failsafe_exec_index,"
        << "cartesian_energy_budget_active,cartesian_effective_time_frozen,"
        << "cartesian_energy_lambda_valid,"
        << "cartesian_energy_scale,cartesian_kinetic_energy,"
        << "cartesian_potential_energy,cartesian_control_energy,"
        << "energy_budget_joule,"
        << "mujoco_contact_value,mujoco_contact_active,mujoco_contact_msg_time_sec,"
        << "mujoco_contact_sample_seq,mujoco_contact_new_sample,"
        << "mujoco_contact_samples_since_last_log,mujoco_contact_sample_age_sec,"
        << "mujoco_first_contact_seen,mujoco_first_contact_wall_time_sec,"
        << "mujoco_first_contact_msg_time_sec\n";
    RCLCPP_INFO(
        get_node()->get_logger(),
        "Validation log enabled: %s",
        error_log_file_path_.c_str());

    if (enable_prediction_logging_) {
      prediction_log_file_.open(prediction_log_file_path_, std::ios::out | std::ios::trunc);
      if (!prediction_log_file_.is_open()) {
        RCLCPP_ERROR(
            get_node()->get_logger(),
            "Failed to open shield prediction log file: %s",
            prediction_log_file_path_.c_str());
        return CallbackReturn::ERROR;
      }
      prediction_log_file_ << std::fixed << std::setprecision(9);
      prediction_log_file_
          << "wall_time_sec,nominal_time_sec,source,"
          << "monitor_total_ms,planner_ms,plan_build_ms,monitor_eval_ms,"
          << "mode,candidate_verified,"
          << "executing_last_verified_monitored,predicted_trigger,monitored_contact_possible,"
          << "plan_intended_steps,plan_failsafe_steps,"
          << "stage,index,is_failsafe_sample,sample_t,dt,"
          << "actual_tcp_px,actual_tcp_py,actual_tcp_pz,"
          << "actual_collision_px,actual_collision_py,actual_collision_pz,"
          << "actual_human_center_px,actual_human_center_py,actual_human_center_pz,"
          << "actual_collision_distance,"
          << "flange_target_px,flange_target_py,flange_target_pz,"
          << "collision_target_px,collision_target_py,collision_target_pz,"
          << "pred_start_px,pred_start_py,pred_start_pz,"
          << "pred_start_vx,pred_start_vy,pred_start_vz,"
          << "pred_next_px,pred_next_py,pred_next_pz,"
          << "pred_next_vx,pred_next_vy,pred_next_vz,"
          << "pred_ax,pred_ay,pred_az,"
          << "human_center_start_px,human_center_start_py,human_center_start_pz,"
          << "human_center_end_px,human_center_end_py,human_center_end_pz,"
          << "distance_start,distance_next,distance_segment,"
          << "contact_possible,k_n,e_n,v_n,T_n_ub,V_potential_ub,"
          << "is_worst_T,is_worst_V,"
          << "Kx,Ky,Kz,Dx,Dy,Dz,"
          << "monitor_worst_case_Tn_ub,monitor_worst_case_V_potential_ub,"
          << "monitor_terminal_energy_ub,monitor_workspace_distance_margin,monitor_h_clamping_energy\n";
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Shield prediction log enabled: %s",
          prediction_log_file_path_.c_str());
    }
  }
  startPredictionLoggerWorker();
  startSafetyMonitorWorker();
  return CallbackReturn::SUCCESS;
}

CallbackReturn ReachableCartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  stopSafetyMonitorWorker();
  stopPredictionLoggerWorker();
  if (error_log_file_.is_open()) { error_log_file_.flush(); error_log_file_.close(); }
  if (prediction_log_file_.is_open()) {
    prediction_log_file_.flush();
    prediction_log_file_.close();
  }
  cartesian_effective_time_frozen_ = false;
  cartesian_effective_time_freeze_start_wall_time_ = -1.0;
  cartesian_effective_time_hold_sample_valid_ = false;
  human_workspace_active_ = false;
  J_geo_prev_.setZero();
  Jdot_dq_filtered_.setZero();
  J_geo_prev_valid_ = false;
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace cps_controllers

PLUGINLIB_EXPORT_CLASS(cps_controllers::ReachableCartesianImpedanceController,
                       controller_interface::ControllerInterface)
