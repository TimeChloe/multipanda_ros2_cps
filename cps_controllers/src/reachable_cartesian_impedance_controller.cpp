// Copyright (c) 2026
// Reachable Cartesian Impedance Controller
//
// Monitored execution with a whole-robot Lachner-style energy budget in nominal contact.
// Intended prefix is supplied by the trajectory generator package.
//
// Runtime and prediction use the same physical impedance and energy law.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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
#include <cps_trajectory_generators/reachable_cartesian_trajectory.hpp>

#include "reachable_cartesian_impedance/math.hpp"
#include "reachable_cartesian_impedance/timing.hpp"

namespace {

constexpr double kMinDt = 1e-6;
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
using cps_controllers::detail::nanosecondsToMilliseconds;
using cps_controllers::detail::SteadyClock;
using CartesianTrajectorySample = cps_trajectory_generators::CartesianTrajectorySample;
using PathConsistentTimedPathConfig =
    cps_trajectory_generators::PathConsistentTimedPathConfig;
using cps_trajectory_generators::makePathConsistentTimedPathBrake;
using cps_trajectory_generators::makePathConsistentTimedPathIntendedPrefix;
using cps_trajectory_generators::makeRetimedPathState;

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
  out.energy_recovery_exit_allowed = b.energy_recovery_exit_allowed;
  out.energy_recovery_epoch = b.energy_recovery_epoch;
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

// ============================================================================
// Helper: runtime gain update
// ============================================================================
void ReachableCartesianImpedanceController::updateRuntimeGains(const Matrix6d& K_target,
                                                               const Matrix6d& D_target) {
  K_runtime_ = K_target;
  D_runtime_ = D_target;
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

  // A predicted collision possibility alone is allowed. Reject nominal
  // predictions exceeding the budget at contact or in a recovery horizon.
  // Joint-limit
  // prediction remains an independent rejection condition, including invalid
  // joint rollouts that cannot produce a predicted_trigger value.
  return monitor.predicted_trigger || monitor.joint_limit_unsafe;
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
    bool workspace_available,
    bool current_overlap,
    bool motion_within_limits,
    bool normal_operation_verified,
    EnergyBudgetInfo* info) {
  EnergyBudgetInfo local_info;
  const auto recovery = cps_safety_monitor::updateEnergyRecovery(
      energy_valid ? kinetic_energy : std::numeric_limits<double>::quiet_NaN(),
      potential_energy, nullspace_potential_energy, energy_budget_joule_,
      energy_recovery_exit_energy_fraction_,
      enable_safety_monitor_ && enable_runtime_energy_scaling_,
      workspace_available, current_overlap, motion_within_limits,
      normal_operation_verified, &energy_recovery_state_,
      command.K.isApprox(K_base_, 1e-9) && command.D.isApprox(D_base_, 1e-9));
  local_info.active = recovery.scaling_active;
  local_info.scale = recovery.scale;
  local_info.recovery_exit_ready = recovery.exit_ready;
  local_info.recovery_exited = recovery.exited;
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
  if (!recovery.scaling_active) {
    if (info != nullptr) {
      *info = local_info;
    }
    return scaled_command;
  }

  // Lachner et al. Eq. (14): one common kappa scales U_x and U_q. This is
  // essential for a globally enabled nullspace spring; treating U_n as fixed
  // would leave part of the controlled potential outside the budget action.
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
    if (command_time > failsafe.back().t + kMinDt) {
      cmd.energy_recovery_exit_allowed = false;
    }
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
    if (command_time > plan.failsafe.back().t + kMinDt) {
      command->energy_recovery_exit_allowed = false;
    }
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
    const ImpedanceSample& planning_start_command,
    double initial_path_rate,
    double target_path_rate,
    bool reanchor_path_kinematics,
    double reanchor_path_rate,
    double reanchor_path_acceleration,
    PlanFailureReason* failure_reason) const {
  if (failure_reason != nullptr) {
    *failure_reason = PlanFailureReason::kNone;
  }
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
    path_config.intended_steps = static_cast<int>(monitor_period_steps_);
    path_config.dt = local_replan_dt_;
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
    // New intended motion uses nominal gains; the runtime energy law alone
    // adapts them. There is no separate horizon-dependent gain ramp.
    s.K = K_base_;
    s.D = D_base_;
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
    const Matrix6d& K_runtime,
    const Matrix6d& D_runtime,
    const std::vector<ImpedanceSample>& intended_samples,
    std::size_t derivative_reanchor_index,
    PlanFailureReason* failure_reason) const {
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

  const ImpedanceSample freeze_anchor = plan.intended.back();
  // Braking and joint rollout share the controller command grid. The sparse
  // view is used for ordinary prediction logging and Cartesian-only fallback.
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
      path_brake_config.max_path_rate = std::max(path_time_rate_max_, 1e-4);
      path_brake_config.max_path_acceleration =
          std::max(failsafe_path_time_acc_limit_, 1e-4);
      path_brake_config.max_path_jerk =
          std::max(failsafe_path_time_jerk_limit_, 1e-4);
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
    // Timed-path interpolation can differ from exact path evaluation by a
    // few micrometres on a curve. Allow that numerical discrepancy while
    // still rejecting a geometrically different route.
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
    double wall_time) const {
  SafetyMonitorConfig config;
  config.human_workspace = human_workspace;
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
  config.energy_recovery_exit_energy_fraction =
      energy_recovery_exit_energy_fraction_;
  config.energy_recovery_nominal_gains_valid = true;
  config.energy_recovery_nominal_stiffness = K_base_;
  config.energy_recovery_nominal_damping = D_base_;
  config.robot_reachability_provider = robot_reachability_provider_;
  config.ee_collision_radius = ee_collision_radius_;
  config.use_dynamic_consistent_impedance = use_dynamic_consistent_impedance_;
  return config;
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
    const cps_human_workspace::HumanWorkspace& human_workspace,
    bool human_workspace_active,
    bool human_workspace_assumed_clear,
    const ImpedanceSample& current_command_reference,
    bool current_command_reference_valid,
    double current_nullspace_stiffness,
    const cps_safety_monitor::EnergyRecoveryState& energy_recovery_state,
    std::uint64_t energy_recovery_epoch,
    std::vector<JointPredictionSample>* joint_prediction_trace) const {
  if (joint_prediction_trace != nullptr) {
    joint_prediction_trace->clear();
  }
  const bool human_workspace_available =
      human_workspace_active || human_workspace_assumed_clear;
  if (!enable_safety_monitor_ || !human_workspace_available) {
    return MonitorResult{};
  }

  SafetyMonitorConfig config = makeSafetyMonitorConfig(
      human_workspace, plan.generated_wall_time);
  config.assume_human_workspace_clear = human_workspace_assumed_clear;
  config.current_energy_reference_valid = current_command_reference_valid;
  if (current_command_reference_valid) {
    config.current_energy_reference = current_command_reference;
  }
  config.nullspace_reference = desired_qn_;
  config.nullspace_stiffness = n_stiffness_;
  config.current_nullspace_stiffness =
      std::max(0.0, current_nullspace_stiffness);
  config.energy_recovery_state = energy_recovery_state;
  config.energy_recovery_epoch = energy_recovery_epoch;
  config.previous_torque_command = previous_torque_command;
  config.previous_torque_command_valid = true;
  config.torque_rate_limit = panda_limits::kTorqueRateLimit;
  // Control, energy and geometry all use the same TCP and Jacobian.
  config.current_joint_dynamics.control_position = current_position;
  config.current_joint_dynamics.control_orientation = current_orientation;
  config.current_joint_dynamics.control_jacobian = J_geo;
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

  // Only the Cartesian fallback uses a sparse plan; the joint rollout above
  // consumes the original command timestamps.
  const VerifiedPlan monitor_plan = makeSparsePlanForMonitor(plan);
  return cps_safety_monitor::verifyReachablePlan(
      monitor_plan,
      current_position,
      current_orientation,
      ee_twist,
      inertia,
      J_geo,
      config);
}

// ============================================================================
// Shield decision
// ============================================================================
ShieldDecision ReachableCartesianImpedanceController::computeShieldDecisionForAsyncInput(
    const AsyncMonitorInput& input) const {
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

  auto report_rejected_candidate = [&](FallbackReason reason) {
    // Only the control thread owns and advances the verified execution stream.
    dec.executing_last_verified_monitored = true;
    dec.candidate_verified = false;
    dec.fallback_reason = reason;
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

    const auto planner_tic = SteadyClock::now();
    const std::vector<ImpedanceSample> intended_buffer =
        makeIntendedBufferFromReplanner(
            planning_start_command,
            input.commanded_path_rate,
            input.target_path_rate,
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

    // The generator produces exactly one monitor period, including terminal
    // hold samples at the path end. Never silently shorten either segment.
    if (input.committed_prefix.size() != monitor_period_steps_ ||
        intended_buffer.size() != monitor_period_steps_) {
      dec.plan_failure_reason = PlanFailureReason::kIntendedGenerationEmpty;
      return false;
    }
    std::vector<ImpedanceSample> intended_segment;
    intended_segment.reserve(2 * monitor_period_steps_);
    intended_segment.insert(intended_segment.end(),
        input.committed_prefix.begin(), input.committed_prefix.end());
    intended_segment.insert(intended_segment.end(),
        intended_buffer.begin(), intended_buffer.end());

    const auto build_tic = SteadyClock::now();
    VerifiedPlan candidate_plan = buildCandidatePlan(
        input.wall_time,
        input.current_position,
        input.current_orientation,
        input.ee_twist,
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
    // Preserve the permissions of the already committed prefix. Only newly
    // generated commands may model exit under this observation episode.
    for (std::size_t i = input.committed_prefix.size();
         i < candidate_plan.intended.size(); ++i) {
      candidate_plan.intended[i].energy_recovery_exit_allowed = true;
      candidate_plan.intended[i].energy_recovery_epoch = input.energy_recovery_epoch;
    }
    for (auto& sample : candidate_plan.failsafe) {
      sample.energy_recovery_exit_allowed = true;
      sample.energy_recovery_epoch = input.energy_recovery_epoch;
    }
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
        input.human_workspace,
        input.human_workspace_active,
        input.human_workspace_assumed_clear,
        input.last_commanded_sample,
        input.last_commanded_sample_valid,
        input.last_nullspace_stiffness,
        input.energy_recovery_state,
        input.energy_recovery_epoch,
        (enable_runtime_energy_scaling_ || enable_prediction_logging_ ||
         enable_reachable_set_visualization_)
            ? &dec.joint_prediction_trace
            : nullptr);
    cps_safety_monitor::restrictEnergyRecoveryExitPermissions(
        &candidate_plan, dec.joint_prediction_trace, input.committed_prefix.size());
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

    dec.executing_last_verified_monitored = false;
    dec.command = candidate_plan.intended.front();

    return true;
  };

  const bool first_attempt = try_one_monitor_pass();

  if (!first_attempt) {
    // Failure to create/evaluate a candidate cannot authorize leaving
    // fallback execution.
    // Report execution of the last verified monitored plan so the real-time
    // side remains conservative until a new candidate is actually verified.
    report_rejected_candidate(FallbackReason::kNone);
    stamp_timing();
    return dec;
  }

  if (dec.candidate_verified) {
    stamp_timing();
    return dec;
  }

  report_rejected_candidate(
      input.human_workspace_active || input.human_workspace_assumed_clear
          ? FallbackReason::kCandidatePredictionRejected
          : FallbackReason::kHumanWorkspaceUnavailable);
  stamp_timing();
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
    double nullspace_stiffness,
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
      tau_task + coriolis + tau_nullspace;
  last_tau_task_norm_ = tau_task.norm();
  last_tau_nullspace_raw_norm_ = tau_nullspace_raw.norm();
  last_tau_nullspace_projected_norm_ = tau_nullspace.norm();
  last_coriolis_norm_ = coriolis.norm();
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
  const std::int64_t control_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      tic_total.time_since_epoch()).count();
  const double control_start_interval_ms = previous_control_start_steady_ns_ > 0
      ? nanosecondsToMilliseconds(control_start_ns - previous_control_start_steady_ns_) : 0.0;
  previous_control_start_steady_ns_ = control_start_ns;
  bool async_output_processed_this_cycle = false;
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

  acceptPendingCartesianViaPoints(
      current_position, current_orientation, wall_time);

  const bool human_workspace_assumed_clear =
      enable_safety_monitor_ &&
      active_cartesian_via_points_calibration_ &&
      calibration_assume_no_human_ &&
      !human_workspace_active_;
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
            current_position,
            human_workspace_.inflatedCollisionRadius(
                ee_collision_radius_, 0.0),
            wall_time);
  }
  // A fresh clear/overlap/unknown transition invalidates any older permission
  // to leave recovery, including permissions in cached backup commands.
  const bool energy_workspace_available = human_workspace_assumed_clear ||
      (human_workspace_active_ && std::isfinite(current_workspace_distance_now));
  // Episode counter, not a timestamp or human-message sequence: positions may
  // change within one class without changing the epoch. A clear -> overlap ->
  // clear round trip advances it twice, so old clear-state permits stay stale.
  const int energy_environment = !energy_workspace_available ? 2 :
      (current_workspace_distance_now <= 0.0 ? 1 : 0);
  if (energy_environment != energy_recovery_environment_) {
    ++energy_recovery_epoch_;
    energy_recovery_environment_ = energy_environment;
  }
  const auto toc_model = SteadyClock::now();

  double paused_total = paused_nominal_time_sec_;
  if (failsafe_enter_wall_time_sec_ >= 0.0)
    paused_total += std::max(0.0, wall_time - failsafe_enter_wall_time_sec_);
  const double nominal_guess_time = std::max(0.0, wall_time - paused_total);

  ShieldExecutionDecision shield_dec = updateMonitoredCommand(
      wall_time, nominal_guess_time, q, dq, current_position, current_orientation,
      ee_twist, inertia, J_geo, coriolis, control_loop_sequence,
      human_workspace_assumed_clear, async_output_processed_this_cycle);
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
      (shield_dec.has_evaluated_plan && last_shield_decision_.evaluated_plan.valid)
          ? last_shield_decision_.evaluated_plan.intended.size()
          : 0;
  const std::size_t monitored_failsafe_steps =
      (shield_dec.has_evaluated_plan && last_shield_decision_.evaluated_plan.valid)
          ? last_shield_decision_.evaluated_plan.failsafe.size()
          : 0;
  const std::size_t monitored_steps =
      monitored_intended_steps + monitored_failsafe_steps;
  const bool monitor_prediction_valid =
      shield_dec.has_evaluated_plan && last_shield_decision_.evaluated_plan.valid &&
      monitored_steps > 0;
  // A failed or delayed rollout is not evidence that the previously predicted
  // collision possibility disappeared. Keep the last prediction classification
  // until a complete monitored trajectory replaces it.
  const bool predicted_contact_possible =
      monitor_prediction_valid
          ? monitor.monitored_contact_possible
          : isCollisionPossibleMode(mode_);
  // Current overlap triggers the stateful gain law; recovery can retain it
  // outside the collision area independently of the geometry diagnostic.
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
  // Lachner Eq. (12) is evaluated during the complete controller lifecycle.
  // Normal free motion uses this as diagnostic/predictive measurement only;
  // overlap and latched recovery use it to adapt gains without retiming.
  const bool track_control_energy =
      enable_safety_monitor_;

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
  cps_safety_monitor::JointDynamicsLimits recovery_motion_limits;
  for (int i = 0; i < 7; ++i) {
    recovery_motion_limits.position_lower(i) = panda_limits::kPositionLower[i];
    recovery_motion_limits.position_upper(i) = panda_limits::kPositionUpper[i];
    recovery_motion_limits.velocity(i) = panda_limits::kVelocity[i];
  }
  // Require a verified command, that exact sample's predicted exit permission,
  // and the same observed environment episode. This only grants permission:
  // applyEnergyBudget also rechecks measured energy, gains, overlap and motion.
  const bool recovery_exit_verified = verified_command_selected_this_cycle_ &&
      cps_safety_monitor::energyRecoveryExitPermitted(
          candidate_budget_command, energy_recovery_epoch_);
  // Keep the energy-budget stiffness and damping adaptation, but do not
  // freeze or retime the verified trajectory.  Path progress therefore stays
  // governed exclusively by the verified command stream.
  shield_dec.command = applyEnergyBudget(
      candidate_budget_command,
      energy_budget_terms.kinetic_energy,
      energy_budget_terms.potential_energy,
      energy_budget_terms.nullspace_potential_energy,
      energy_budget_terms.valid,
      energy_workspace_available,
      current_workspace_distance_now <= 0.0,
      cps_safety_monitor::jointStateWithinLimits(q, dq, recovery_motion_limits),
      recovery_exit_verified,
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
      dt);
  const auto toc_torque = SteadyClock::now();

  for (int i = 0; i < kNumJoints; ++i) command_interfaces_[i].set_value(tau_cmd(i));

  updateCartesianViaPointsActionStatus(
      current_position,
      current_orientation,
      wall_time);

  logControlCycle({
      wall_time, nominal_guess_time, shield_dec,
      monitor, q, dq,
      current_position, ee_twist, error,
      tau_cmd, energy_info, previous_applied_energy_terms,
      previous_applied_nullspace_stiffness, previous_applied_verified_plan_valid, previous_applied_verified_plan_generation,
      previous_applied_verified_command_stage, previous_applied_verified_command_index, human_workspace_assumed_clear,
      monitor_prediction_valid, predicted_contact_possible, recovery_exit_verified,
      async_output_processed_this_cycle, control_start_interval_ms, control_loop_sequence
  });
  const auto toc_io = SteadyClock::now();

  recordControlTiming(tic_total, toc_model, toc_shield, toc_torque, toc_io);

  previous_control_execution_ms_ =
      std::chrono::duration<double, std::milli>(SteadyClock::now() - tic_total).count();
  return controller_interface::return_type::OK;
}

}  // namespace cps_controllers

PLUGINLIB_EXPORT_CLASS(cps_controllers::ReachableCartesianImpedanceController,
                       controller_interface::ControllerInterface)
