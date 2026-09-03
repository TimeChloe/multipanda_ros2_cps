// Copyright (c) 2026
// Non-real-time logging support for ReachableCartesianImpedanceController.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>

namespace cps_controllers
{
namespace
{

constexpr double kMinDt = 1e-6;
constexpr double kSmallPositive = 1e-9;

}  // namespace

VerifiedPlan
ReachableCartesianImpedanceController::planForExecutionLogging(
  const VerifiedPlan & plan) const
{
  VerifiedPlan source = plan;
  source.intended_exec_index = 0;
  source.failsafe_exec_index = 0;

  VerifiedPlan execution = source;
  execution.failsafe.clear();
  const std::size_t command_count = failsafeCommandCount(source);
  execution.failsafe.reserve(command_count);
  for (std::size_t command_index = 0;
    command_index < command_count; ++command_index)
  {
    ImpedanceSample command;
    const std::size_t offset = source.intended.size() + command_index;
    if (!getVerifiedTrajectoryCommandAtOffset(source, offset, &command)) {
      execution.valid = false;
      execution.failsafe.clear();
      return execution;
    }
    // Its vector index now has exactly the same meaning as
    // last_verified_command_index_ in the 1 kHz runtime executor.
    execution.failsafe.push_back(command);
  }
  execution.valid = source.valid && !execution.intended.empty() &&
    !execution.failsafe.empty();
  return execution;
}

void ReachableCartesianImpedanceController::logShieldPredictionTrajectory(
  double wall_time,
  double nominal_guess_time,
  const Vector7d & current_q,
  const Vector7d & current_dq,
  const Vector3d & current_position,
  const Quaterniond & current_orientation,
  const Vector6d & ee_twist,
  const Matrix7d & inertia,
  const Matrix37d & Jv,
  const Matrix6d & K_runtime,
  const Matrix6d & D_runtime,
  const cps_human_workspace::HumanWorkspace & human_workspace,
  bool human_workspace_active,
  bool human_workspace_assumed_clear,
  const VerifiedPlan & evaluated_plan,
  const std::vector<JointPredictionSample> & joint_prediction_trace,
  const AsyncMonitorTiming & async_timing,
  const MonitorResult & monitor,
  int mode,
  bool candidate_verified,
  std::uint64_t accepted_plan_generation,
  bool executing_last_verified_monitored,
  double monitor_total_ms,
  double planner_ms,
  double plan_build_ms,
  double monitor_eval_ms,
  const char * source)
{
  if (!enable_prediction_logging_ || !prediction_log_writer_.running() ||
    !command_recording_active_ ||
    !evaluated_plan.valid)
  {
    return;
  }

  prediction_log_writer_.tryEmplace(
    [&](PredictionLogRecord & record) {
      record.wall_time = wall_time;
      record.nominal_guess_time = nominal_guess_time;
      record.current_q = current_q;
      record.current_dq = current_dq;
      record.current_position = current_position;
      record.current_orientation = current_orientation;
      record.ee_twist = ee_twist;
      record.inertia = inertia;
      record.Jv = Jv;
      record.K_runtime = K_runtime;
      record.D_runtime = D_runtime;
      record.human_workspace = human_workspace;
      record.human_workspace_active = human_workspace_active;
      record.human_workspace_assumed_clear =
        human_workspace_assumed_clear;
      record.evaluated_plan = evaluated_plan;
      record.joint_prediction_trace = joint_prediction_trace;
      record.async_timing = async_timing;
      record.monitor = monitor;
      record.mode = mode;
      record.candidate_verified = candidate_verified;
      record.accepted_plan_generation = accepted_plan_generation;
      record.executing_last_verified_monitored =
      executing_last_verified_monitored;
      record.monitor_total_ms = monitor_total_ms;
      record.planner_ms = planner_ms;
      record.plan_build_ms = plan_build_ms;
      record.monitor_eval_ms = monitor_eval_ms;
      record.async_source = source != nullptr && std::strcmp(source, "async") == 0;
    });
}

void ReachableCartesianImpedanceController::writeShieldPredictionTrajectory(
  std::ostream & output,
  double wall_time,
  double nominal_guess_time,
  const Vector7d & current_q,
  const Vector7d & current_dq,
  const Vector3d & current_position,
  const Quaterniond & current_orientation,
  const Vector6d & ee_twist,
  const Matrix7d & inertia,
  const Matrix37d & Jv,
  const Matrix6d & K_runtime,
  const Matrix6d & D_runtime,
  const cps_human_workspace::HumanWorkspace & human_workspace,
  bool human_workspace_active,
  bool human_workspace_assumed_clear,
  const VerifiedPlan & evaluated_plan,
  const std::vector<JointPredictionSample> & joint_prediction_trace,
  const AsyncMonitorTiming & async_timing,
  const MonitorResult & monitor,
  int mode,
  bool candidate_verified,
  std::uint64_t accepted_plan_generation,
  bool executing_last_verified_monitored,
  double monitor_total_ms,
  double planner_ms,
  double plan_build_ms,
  double monitor_eval_ms,
  const char * source)
{
  if (!evaluated_plan.valid) {
    return;
  }

  struct PredictionRow
  {
    const char * stage{""};
    int index{0};
    bool failsafe{false};
    double sample_t{0.0};
    double dt{0.0};
    Vector3d collision_target_p{Vector3d::Zero()};
    Vector3d x_pred{Vector3d::Zero()};
    Vector3d v_pred{Vector3d::Zero()};
    Vector3d x_next{Vector3d::Zero()};
    Vector3d v_next{Vector3d::Zero()};
    Vector3d a_pred{Vector3d::Zero()};
    bool joint_state_valid{false};
    double joint_sample_t{std::numeric_limits<double>::quiet_NaN()};
    bool expected_control_sequence_valid{false};
    std::uint64_t expected_control_sequence{0};
    std::size_t horizon_steps{0};
    bool guaranteed_committed_sample{false};
    bool joint_energy_valid{false};
    double joint_kinetic_energy{0.0};
    double cartesian_potential_energy{0.0};
    double nullspace_potential_energy{0.0};
    bool nullspace_potential_energy_active{false};
    bool energy_scaling_active{false};
    double energy_stiffness_scale{1.0};
    double applied_nullspace_stiffness{0.0};
    bool overbudget_joint_stabilization_active{false};
    double overbudget_joint_potential_energy{0.0};
    double overbudget_joint_scale_rho{1.0};
    double overbudget_joint_torque_norm{0.0};
    Vector7d joint_q{Vector7d::Constant(
        std::numeric_limits<double>::quiet_NaN())};
    Vector7d joint_dq{Vector7d::Constant(
        std::numeric_limits<double>::quiet_NaN())};
    bool robot_reach_valid{false};
    int closest_robot_link_index{-1};
    double d_segment{0.0};
    Vector3d human_center_end{Vector3d::Zero()};
    double human_reach_radius{0.0};
    bool contact_possible{false};
    double Kx{0.0};
    double Ky{0.0};
    double Kz{0.0};
    double Dx{0.0};
    double Dy{0.0};
    double Dz{0.0};
  };

  // Passive calibration keeps normal command selection unchanged.  Build the
  // logging view through the runtime offset accessor so fail-safe time stamps,
  // interpolation, and indices are identical to the commands actually sent.
  const VerifiedPlan monitor_plan =
    enable_calibration_logging_ ?
    planForExecutionLogging(evaluated_plan) :
    makeSparsePlanForMonitor(evaluated_plan);
  const VerifiedPlan collision_plan =
    makeCollisionCenterPlanForMonitor(monitor_plan);
  const Vector3d collision_center =
    current_position + collisionCenterOffsetWorld(current_orientation);
  const Vector6d collision_twist =
    twistAtCollisionCenter(current_orientation, ee_twist);

  Matrix3d task_inertia_inv = Jv * inertia.inverse() * Jv.transpose();
  task_inertia_inv.diagonal().array() += kSmallPositive;
  Matrix6d K_exec = K_runtime;
  Matrix6d D_exec = D_runtime;

  Vector3d x_pred = collision_center;
  Vector3d v_pred = collision_twist.head<3>();
  Vector7d previous_joint_q = current_q;
  double t_prev = collision_plan.anchor.t;
  std::vector<PredictionRow> rows;
  rows.reserve(1 + collision_plan.intended.size() + collision_plan.failsafe.size());
  std::size_t joint_trace_index = 0;
  std::vector<double> dynamic_robot_alpha;
  const bool dynamic_robot_alpha_valid =
    robot_reachability_provider_ &&
    robot_reachability_provider_->calculateTrajectoryAlpha(
    joint_prediction_trace, &dynamic_robot_alpha);

  auto append_row = [&](const char * stage,
      int index,
      const ImpedanceSample & collision_sample,
      double dtp) {
      const double segment_start_time_sec = wall_time + t_prev;
      const double segment_end_time_sec = wall_time + collision_sample.t;
      K_exec = collision_sample.K;
      D_exec = collision_sample.D;

      const Matrix3d Kp_raw = K_exec.topLeftCorner<3, 3>();
      const Matrix3d Dp_raw = D_exec.topLeftCorner<3, 3>();
      const Vector3d force_pred =
        Kp_raw * (collision_sample.p - x_pred) -
        Dp_raw * (v_pred - collision_sample.dp);
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
      row.collision_target_p = collision_sample.p;
      row.x_pred = x_pred;
      row.v_pred = v_pred;
      row.x_next = x_next;
      row.v_next = v_next;
      row.a_pred = a_pred;
      while (joint_trace_index + 1 < joint_prediction_trace.size() &&
        std::abs(
          joint_prediction_trace[joint_trace_index + 1].t -
          collision_sample.t) <=
        std::abs(
          joint_prediction_trace[joint_trace_index].t -
          collision_sample.t))
      {
        ++joint_trace_index;
      }
      if (joint_trace_index < joint_prediction_trace.size()) {
        const JointPredictionSample & joint_sample =
          joint_prediction_trace[joint_trace_index];
        if (std::abs(joint_sample.t - collision_sample.t) <= 1.0e-8 &&
          joint_sample.q.allFinite() && joint_sample.dq.allFinite())
        {
          row.joint_state_valid = true;
          row.joint_sample_t = joint_sample.t;
          row.joint_q = joint_sample.q;
          row.joint_dq = joint_sample.dq;
          row.joint_energy_valid = joint_sample.energy_valid;
          row.joint_kinetic_energy =
            joint_sample.joint_kinetic_energy;
          row.cartesian_potential_energy =
            joint_sample.cartesian_potential_energy;
          row.nullspace_potential_energy =
            joint_sample.nullspace_potential_energy;
          row.nullspace_potential_energy_active =
            joint_sample.nullspace_potential_energy_active;
          row.energy_scaling_active =
            joint_sample.energy_scaling_active;
          row.energy_stiffness_scale =
            joint_sample.energy_stiffness_scale;
          row.applied_nullspace_stiffness =
            joint_sample.applied_nullspace_stiffness;
          row.overbudget_joint_stabilization_active =
            joint_sample.overbudget_joint_stabilization_active;
          row.overbudget_joint_potential_energy =
            joint_sample.overbudget_joint_potential_energy;
          row.overbudget_joint_scale_rho =
            joint_sample.overbudget_joint_scale_rho;
          row.overbudget_joint_torque_norm =
            joint_sample.overbudget_joint_torque_norm;
          if (async_timing.valid) {
            row.horizon_steps = static_cast<std::size_t>(std::max<long long>(
                0,
                std::llround(
                  joint_sample.t /
                  std::max(local_replan_dt_, kMinDt))));
            row.expected_control_sequence_valid = true;
            row.expected_control_sequence =
              async_timing.input_control_loop_sequence + row.horizon_steps;
            row.guaranteed_committed_sample =
              row.horizon_steps > 0 &&
              row.horizon_steps <= async_timing.committed_prefix_steps;
          }
        }
      }
      const auto hand_reach =
        human_workspace.handReachableSetAtTime(segment_end_time_sec);
      row.human_center_end = hand_reach.center;
      row.human_reach_radius = hand_reach.radius;
      if (human_workspace_assumed_clear) {
        row.d_segment = std::numeric_limits<double>::infinity();
      } else if (robot_reachability_provider_ && row.joint_state_valid) {
        std::vector<cps_safety_monitor::RobotReachCapsule> robot_capsules;
        row.robot_reach_valid =
          dynamic_robot_alpha_valid &&
          robot_reachability_provider_->reachInterval(
          previous_joint_q,
          row.joint_q,
          dtp,
          dynamic_robot_alpha,
          &robot_capsules);
        row.d_segment = row.robot_reach_valid ?
          robot_reachability_provider_->minimumSignedDistance(
          robot_capsules,
          row.human_center_end,
          row.human_center_end,
          row.human_reach_radius,
          &row.closest_robot_link_index) :
          -std::numeric_limits<double>::infinity();
      } else {
        row.d_segment =
          human_workspace.signedDistanceSegmentToInflatedSphere(
          x_pred,
          x_next,
          human_workspace.inflatedCollisionRadius(
            ee_collision_radius_, 0.0),
          segment_start_time_sec,
          segment_end_time_sec);
      }
      row.contact_possible = row.d_segment <= 0.0;
      const double logged_energy_scale = row.joint_state_valid
        ? std::clamp(row.energy_stiffness_scale, 0.0, 1.0)
        : 1.0;
      const double logged_damping_scale = std::sqrt(logged_energy_scale);
      row.Kx = logged_energy_scale * K_exec(0, 0);
      row.Ky = logged_energy_scale * K_exec(1, 1);
      row.Kz = logged_energy_scale * K_exec(2, 2);
      row.Dx = logged_damping_scale * D_exec(0, 0);
      row.Dy = logged_damping_scale * D_exec(1, 1);
      row.Dz = logged_damping_scale * D_exec(2, 2);
      rows.push_back(row);

      x_pred = x_next;
      v_pred = v_next;
      if (row.joint_state_valid) {
        previous_joint_q = row.joint_q;
      }
    };

  auto append_samples = [&](const char * stage,
      const std::vector<ImpedanceSample> & collision_samples) {
      for (std::size_t i = 0; i < collision_samples.size(); ++i) {
        const double dtp = std::max(collision_samples[i].t - t_prev, kMinDt);
        append_row(
          stage,
          static_cast<int>(i),
          collision_samples[i],
          dtp);
        t_prev = collision_samples[i].t;
      }
    };

  append_samples("intended", collision_plan.intended);
  append_samples("failsafe", collision_plan.failsafe);

  const auto actual_hand_reach =
    human_workspace.handReachableSetAtTime(wall_time);
  const Vector3d actual_human_center = actual_hand_reach.center;
  double actual_collision_distance =
    human_workspace_assumed_clear ?
    std::numeric_limits<double>::infinity() :
    std::numeric_limits<double>::quiet_NaN();
  int actual_closest_robot_link_index = -1;
  if (human_workspace_active && robot_reachability_provider_) {
    std::vector<cps_safety_monitor::RobotReachCapsule> robot_capsules;
    const std::vector<double> zero_alpha(7, 0.0);
    if (robot_reachability_provider_->reachInterval(
        current_q, current_q, 0.0, zero_alpha, &robot_capsules))
    {
      actual_collision_distance =
        robot_reachability_provider_->minimumSignedDistance(
        robot_capsules,
        actual_human_center,
        actual_human_center,
        actual_hand_reach.radius,
        &actual_closest_robot_link_index);
    }
  } else if (human_workspace_active) {
    actual_collision_distance =
      human_workspace.signedDistanceToInflatedSphere(
      collision_center,
      human_workspace.inflatedCollisionRadius(
        ee_collision_radius_, 0.0),
      wall_time);
  }

  output << std::fixed << std::setprecision(9);
  for (const auto & row : rows) {
    output
      << wall_time << "," << nominal_guess_time << ","
      << (source == nullptr ? "" : source) << ","
      << static_cast<int>(human_workspace_active) << ","
      << static_cast<int>(human_workspace_assumed_clear) << ","
      << static_cast<int>(async_timing.valid) << ","
      << async_timing.input_sequence << ","
      << async_timing.input_control_loop_sequence << ","
      << async_timing.source_plan_generation << ","
      << async_timing.committed_prefix_steps << ","
      << static_cast<int>(
        async_timing.source_plan_matches_at_handoff) << ","
      << static_cast<int>(async_timing.output_usable) << ","
      << async_timing.scheduled_control_loop_sequence << ","
      << async_timing.publish_lateness_cycles << ","
      << async_timing.worker_queue_wait_ms << ","
      << async_timing.worker_compute_ms << ","
      << async_timing.output_handoff_ms << ","
      << async_timing.end_to_end_ms << ","
      << monitor_total_ms << ","
      << planner_ms << ","
      << plan_build_ms << ","
      << monitor_eval_ms << ","
      << mode << ","
      << static_cast<int>(candidate_verified) << ","
      << accepted_plan_generation << ","
      << static_cast<int>(executing_last_verified_monitored) << ","
      << static_cast<int>(monitor.predicted_trigger) << ","
      << monitor.collision_interval_index << ","
      << static_cast<int>(monitor.monitored_contact_possible) << ","
      << monitor_plan.intended.size() << ","
      << monitor_plan.failsafe.size() << ","
      << row.stage << "," << row.index << ","
      << static_cast<int>(row.failsafe) << ","
      << row.sample_t << "," << row.dt << ","
      << collision_center(0) << "," << collision_center(1) << "," << collision_center(2) << ","
      << actual_human_center(0) << "," << actual_human_center(1) << "," << actual_human_center(2) <<
      "," << actual_hand_reach.radius << ","
      << actual_collision_distance << ","
      << actual_closest_robot_link_index << ","
      << current_q(0) << "," << current_q(1) << "," << current_q(2) << ","
      << current_q(3) << "," << current_q(4) << "," << current_q(5) << ","
      << current_q(6) << ","
      << current_dq(0) << "," << current_dq(1) << "," << current_dq(2) << ","
      << current_dq(3) << "," << current_dq(4) << "," << current_dq(5) << ","
      << current_dq(6) << ","
      << row.collision_target_p(0) << "," << row.collision_target_p(1) << "," <<
      row.collision_target_p(2) << ","
      << row.x_pred(0) << "," << row.x_pred(1) << "," << row.x_pred(2) << ","
      << row.v_pred(0) << "," << row.v_pred(1) << "," << row.v_pred(2) << ","
      << row.x_next(0) << "," << row.x_next(1) << "," << row.x_next(2) << ","
      << row.v_next(0) << "," << row.v_next(1) << "," << row.v_next(2) << ","
      << row.a_pred(0) << "," << row.a_pred(1) << "," << row.a_pred(2) << ","
      << static_cast<int>(row.joint_state_valid) << ","
      << row.joint_sample_t << ","
      << static_cast<int>(row.expected_control_sequence_valid) << ","
      << row.expected_control_sequence << ","
      << row.horizon_steps << ","
      << static_cast<int>(row.guaranteed_committed_sample) << ","
      << static_cast<int>(row.joint_energy_valid) << ","
      << row.joint_kinetic_energy << ","
      << row.cartesian_potential_energy << ","
      << row.nullspace_potential_energy << ","
      << static_cast<int>(row.nullspace_potential_energy_active) << ","
      << static_cast<int>(row.energy_scaling_active) << ","
      << row.energy_stiffness_scale << ","
      << row.applied_nullspace_stiffness << ","
      << static_cast<int>(
        row.overbudget_joint_stabilization_active) << ","
      << row.overbudget_joint_potential_energy << ","
      << row.overbudget_joint_scale_rho << ","
      << row.overbudget_joint_torque_norm << ","
      << row.joint_kinetic_energy + row.cartesian_potential_energy +
        row.nullspace_potential_energy << ","
      << row.joint_kinetic_energy +
        std::max(0.0, kinetic_energy_error_bound_joule_) << ","
      << row.cartesian_potential_energy +
        std::max(0.0, potential_energy_error_bound_joule_) << ","
      << row.nullspace_potential_energy +
        (row.nullspace_potential_energy_active ?
          std::max(0.0, nullspace_potential_energy_error_bound_joule_) :
          0.0) << ","
      << row.joint_kinetic_energy + row.cartesian_potential_energy +
        row.nullspace_potential_energy +
        std::max(0.0, kinetic_energy_error_bound_joule_) +
        std::max(0.0, potential_energy_error_bound_joule_) +
        (row.nullspace_potential_energy_active ?
          std::max(0.0, nullspace_potential_energy_error_bound_joule_) :
          0.0) << ","
      << row.joint_q(0) << "," << row.joint_q(1) << "," << row.joint_q(2) << ","
      << row.joint_q(3) << "," << row.joint_q(4) << "," << row.joint_q(5) << ","
      << row.joint_q(6) << ","
      << row.joint_dq(0) << "," << row.joint_dq(1) << "," << row.joint_dq(2) << ","
      << row.joint_dq(3) << "," << row.joint_dq(4) << "," << row.joint_dq(5) << ","
      << row.joint_dq(6) << ","
      << row.human_center_end(0) << "," << row.human_center_end(1) << "," <<
      row.human_center_end(2) << "," << row.human_reach_radius << ","
      << static_cast<int>(row.robot_reach_valid) << ","
      << row.closest_robot_link_index << ","
      << row.d_segment << ","
      << static_cast<int>(row.contact_possible) << ","
      << row.Kx << "," << row.Ky << "," << row.Kz << ","
      << row.Dx << "," << row.Dy << "," << row.Dz << ","
      << monitor.worst_case_cartesian_kinetic_energy_ub << ","
      << monitor.worst_case_joint_kinetic_energy_ub << ","
      << monitor.worst_case_cartesian_potential_energy_ub << ","
      << monitor.worst_case_nullspace_potential_energy_ub << ","
      << monitor.worst_case_total_control_energy_ub << ","
      << monitor.terminal_energy_ub << ","
      << monitor.workspace_distance_margin << ","
      << static_cast<int>(monitor.current_cartesian_energy_valid) << ","
      << monitor.current_cartesian_kinetic_energy << ","
      << static_cast<int>(monitor.current_joint_energy_valid) << ","
      << monitor.current_joint_kinetic_energy << ","
      << monitor.current_cartesian_potential_energy << ","
      << monitor.current_nullspace_potential_energy << ","
      << monitor.current_total_control_energy << ","
      << static_cast<int>(monitor.inertia_model_comparison_valid) << ","
      << monitor.runtime_model_joint_kinetic_energy << ","
      << monitor.prediction_model_joint_kinetic_energy << ","
      << monitor.inertia_model_kinetic_energy_error << ","
      << monitor.inertia_model_difference_frobenius_norm << ","
      << monitor.inertia_model_difference_relative_frobenius_norm << ","
      << monitor.inertia_model_difference_max_abs << ","
      << monitor.inertia_model_difference_max_abs_row << ","
      << monitor.inertia_model_difference_max_abs_col << ","
      << static_cast<int>(monitor.inertia_model_energy_ratio_valid) << ","
      << monitor.inertia_model_min_energy_ratio << ","
      << monitor.inertia_model_max_energy_ratio << ","
      << static_cast<int>(monitor.joint_limit_unsafe) << ","
      << monitor.joint_limit_index << ","
      << monitor.joint_position_violation << ","
      << monitor.joint_velocity_violation << ","
      << monitor.joint_acceleration_violation << ","
      << monitor.joint_torque_violation << "\n";
  }
}

bool ReachableCartesianImpedanceController::startLogWriters()
{
  const std::string control_header =
    "wall_time_sec,nominal_time_sec,paused_nominal_time_sec,"
    "commanded_path_time_sec,command_path_time_sec,"
    "command_path_time_valid,commanded_path_rate,"
    "generator_target_path_rate,mode,execution_stage,fallback_reason,"
    "plan_failure_reason,candidate_verified,monitor_prediction_valid,"
    "predicted_trigger,collision_interval_index,predicted_contact_possible,"
    "monitored_contact_possible,human_workspace_active,"
    "human_workspace_assumed_clear,contact_relevant_for_energy,"
    "collision_energy_unsafe,monitored_unsafe,joint_limit_unsafe,"
    "joint_limit_index,joint_position_violation,joint_velocity_violation,"
    "joint_acceleration_violation,joint_torque_violation,workspace_distance_now,"
    "workspace_distance_min,workspace_distance_margin,"
    "measured_q1,measured_q2,measured_q3,measured_q4,measured_q5,"
    "measured_q6,measured_q7,measured_dq1,measured_dq2,measured_dq3,"
    "measured_dq4,measured_dq5,measured_dq6,measured_dq7,"
    "des_px,des_py,des_pz,cur_px,cur_py,cur_pz,"
    "cur_vx,cur_vy,cur_vz,cur_wx,cur_wy,cur_wz,"
    "collision_center_px,collision_center_py,collision_center_pz,"
    "human_center_px,human_center_py,human_center_pz,"
    "collision_center_vx,collision_center_vy,collision_center_vz,"
    "des_vx,des_vy,des_vz,err_px,err_py,err_pz,err_rx,err_ry,err_rz,"
    "tau_cmd_norm,tau_task_norm,tau_nullspace_raw_norm,"
    "tau_nullspace_projected_norm,tau_overbudget_joint_norm,coriolis_norm,"
    "tau_desired_before_rate_limit_norm,torque_rate_limited,"
    "torque_rate_max_ratio,Kx,Ky,Kz,Dx,Dy,Dz,"
    "worst_case_contact_time,worst_case_workspace_distance_at_candidate,"
    "worst_case_cartesian_kinetic_energy_ub,"
    "worst_case_joint_kinetic_energy_ub,"
    "worst_case_cartesian_potential_energy_ub,"
    "worst_case_nullspace_potential_energy_ub,"
    "worst_case_total_control_energy_ub,"
    "terminal_energy_ub,"
    "robot_reach_secure_radius,robot_reach_alpha_valid,"
    "robot_reach_alpha_1,robot_reach_alpha_2,robot_reach_alpha_3,"
    "robot_reach_alpha_4,robot_reach_alpha_5,robot_reach_alpha_6,"
    "robot_reach_alpha_7,current_robot_link_index,"
    "worst_case_robot_link_index,"
    "monitor_current_cartesian_energy_valid,"
    "monitor_current_cartesian_kinetic_energy,"
    "monitor_current_joint_energy_valid,"
    "monitor_current_joint_kinetic_energy,"
    "monitor_current_cartesian_potential_energy_after_scaling,"
    "monitor_current_nullspace_potential_energy,"
    "monitor_current_total_control_energy_after_scaling,monitored_steps,"
    "monitored_intended_steps,"
    "monitored_failsafe_steps,control_loop_sequence,"
    "monitor_period_control_cycles,next_async_monitor_control_sequence,"
    "last_async_input_publish_control_sequence,async_timing_valid,"
    "async_monitor_input_sequence,async_monitor_input_control_sequence,"
    "async_monitor_source_plan_generation,async_committed_prefix_steps,"
    "async_source_plan_matches_at_handoff,async_output_usable,"
    "async_monitor_scheduled_control_sequence,async_monitor_publish_lateness_cycles,"
    "async_monitor_worker_queue_wait_ms,async_monitor_worker_compute_ms,"
    "async_monitor_output_handoff_ms,async_monitor_end_to_end_ms,"
    "async_monitor_inputs_published,async_monitor_inputs_overwritten,"
    "async_monitor_worker_processed,async_monitor_outputs_overwritten,"
    "async_monitor_outputs_consumed,async_monitor_schedule_late_cycles,"
    "async_monitor_schedule_skipped_slots,verified_plan_age_sec,"
    "verified_next_intended_exec_index,verified_next_failsafe_exec_index,"
    "executed_verified_command_stage,executed_verified_command_index,"
    "executed_verified_plan_valid,executed_verified_plan_generation,"
    "previous_applied_verified_plan_valid,"
    "previous_applied_verified_plan_generation,"
    "previous_applied_verified_command_stage,"
    "previous_applied_verified_command_index,"
    "calibration_execution_active,calibration_plan_latched,"
    "calibration_plan_complete,calibration_target_failed,"
    "calibration_requested_capture_path_time_sec,"
    "calibration_actual_capture_path_time_sec,"
    "calibration_plan_generation,"
    "calibration_monitor_input_sequence,"
    "calibration_activation_control_sequence,"
    "calibration_activation_intended_index,"
    "calibration_activation_failsafe_index,"
    "runtime_energy_scaling_enabled,"
    "energy_budget_active,"
    "cartesian_energy_lambda_valid,energy_stiffness_scale,"
    "joint_kinetic_energy,cartesian_potential_energy_before_scaling,"
    "nullspace_potential_energy_before_scaling,"
    "total_control_energy_before_scaling,"
    "cartesian_potential_energy_after_scaling,"
    "nullspace_potential_energy_after_scaling,"
    "nullspace_enabled,nullspace_stiffness_before_scaling,"
    "nullspace_stiffness_after_scaling,"
    "total_control_energy_after_scaling,"
    "overbudget_joint_stabilization_enabled,"
    "overbudget_joint_stabilization_active,"
    "overbudget_joint_stiffness,overbudget_joint_scale_omega,"
    "overbudget_joint_potential_energy,overbudget_joint_scale_rho,"
    "overbudget_joint_reference_q1,overbudget_joint_reference_q2,"
    "overbudget_joint_reference_q3,overbudget_joint_reference_q4,"
    "overbudget_joint_reference_q5,overbudget_joint_reference_q6,"
    "overbudget_joint_reference_q7,"
    "previous_applied_energy_valid,"
    "previous_applied_joint_kinetic_energy,"
    "previous_applied_cartesian_potential_energy,"
    "previous_applied_nullspace_potential_energy,"
    "previous_applied_nullspace_potential_energy_active,"
    "previous_applied_total_energy,energy_budget_joule,mujoco_contact_value,"
    "mujoco_contact_active,mujoco_contact_sample_age_sec";

  const std::size_t expected_control_columns =
    1 + static_cast<std::size_t>(
    std::count(control_header.begin(), control_header.end(), ','));
  if (expected_control_columns > ControlLogRecord::kMaxValues) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Control log schema has %zu columns but the fixed record holds %zu.",
      expected_control_columns,
      ControlLogRecord::kMaxValues);
    return false;
  }

  if (enable_error_logging_) {
    if (!control_log_writer_.start(
        error_log_file_path_,
        control_header,
        control_log_max_queue_size_,
        log_batch_size_,
        log_flush_period_sec_,
        [this, expected_control_columns](
          std::ostream & output, const ControlLogRecord & record) {
          if (record.value_count != expected_control_columns) {
            control_log_column_mismatch_count_.fetch_add(
              1, std::memory_order_relaxed);
          }
          output << std::fixed << std::setprecision(9);
          const std::size_t count = std::min(
            record.value_count, ControlLogRecord::kMaxValues);
          for (std::size_t i = 0; i < count; ++i) {
            if (i != 0) {
              output << ',';
            }
            output << record.values[i];
          }
          output << '\n';
        }))
    {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to start asynchronous control logger: %s",
        error_log_file_path_.c_str());
      return false;
    }
  }

  if (enable_prediction_logging_) {
    const std::string prediction_header =
      "wall_time_sec,nominal_time_sec,source,human_workspace_active,"
      "human_workspace_assumed_clear,async_timing_valid,"
      "monitor_input_sequence,monitor_input_control_loop_sequence,"
      "monitor_source_plan_generation,committed_prefix_steps,"
      "source_plan_matches_at_handoff,async_output_usable,"
      "monitor_scheduled_control_loop_sequence,monitor_publish_lateness_cycles,"
      "worker_queue_wait_ms,worker_compute_ms,output_handoff_ms,"
      "monitor_end_to_end_ms,monitor_total_ms,planner_ms,"
      "plan_build_ms,monitor_eval_ms,mode,candidate_verified,"
      "accepted_plan_generation,"
      "executing_last_verified_monitored,predicted_trigger,"
      "collision_interval_index,monitored_contact_possible,"
      "plan_intended_steps,plan_failsafe_steps,"
      "stage,index,is_failsafe_sample,sample_t,dt,actual_collision_px,actual_collision_py,"
      "actual_collision_pz,actual_human_center_px,actual_human_center_py,"
      "actual_human_center_pz,actual_hand_reach_radius,actual_collision_distance,"
      "actual_closest_robot_link_index,"
      "measured_q1,measured_q2,measured_q3,measured_q4,measured_q5,"
      "measured_q6,measured_q7,measured_dq1,measured_dq2,measured_dq3,"
      "measured_dq4,measured_dq5,measured_dq6,measured_dq7,collision_target_px,"
      "collision_target_py,collision_target_pz,pred_start_px,pred_start_py,"
      "pred_start_pz,pred_start_vx,pred_start_vy,pred_start_vz,pred_next_px,"
      "pred_next_py,pred_next_pz,pred_next_vx,pred_next_vy,pred_next_vz,"
      "pred_ax,pred_ay,pred_az,pred_joint_state_valid,pred_joint_sample_t,"
      "expected_control_sequence_valid,expected_control_loop_sequence,"
      "prediction_horizon_steps,guaranteed_committed_sample,"
      "pred_energy_valid,pred_joint_kinetic_energy,"
      "pred_cartesian_potential_energy,pred_nullspace_potential_energy,"
      "pred_nullspace_potential_energy_active,pred_energy_scaling_active,"
      "pred_energy_stiffness_scale,"
      "pred_applied_nullspace_stiffness,"
      "pred_overbudget_joint_stabilization_active,"
      "pred_overbudget_joint_potential_energy,"
      "pred_overbudget_joint_scale_rho,pred_overbudget_joint_torque_norm,"
      "pred_total_energy,"
      "pred_joint_kinetic_energy_ub,pred_cartesian_potential_energy_ub,"
      "pred_nullspace_potential_energy_ub,"
      "pred_total_energy_ub,"
      "pred_q1,pred_q2,pred_q3,pred_q4,pred_q5,pred_q6,pred_q7,"
      "pred_dq1,pred_dq2,pred_dq3,pred_dq4,pred_dq5,pred_dq6,pred_dq7,"
      "human_center_end_px,human_center_end_py,"
      "human_center_end_pz,human_reach_radius,robot_reach_valid,"
      "closest_robot_link_index,"
      "distance_segment,contact_possible,"
      "Kx,Ky,Kz,Dx,Dy,Dz,"
      "monitor_worst_case_cartesian_kinetic_energy_ub,"
      "monitor_worst_case_joint_kinetic_energy_ub,"
      "monitor_worst_case_cartesian_potential_energy_ub,"
      "monitor_worst_case_nullspace_potential_energy_ub,"
      "monitor_worst_case_total_control_energy_ub,"
      "monitor_terminal_energy_ub,monitor_workspace_distance_margin,"
      "monitor_current_cartesian_energy_valid,"
      "monitor_current_cartesian_kinetic_energy,"
      "monitor_current_joint_energy_valid,"
      "monitor_current_joint_kinetic_energy,"
      "monitor_current_cartesian_potential_energy,"
      "monitor_current_nullspace_potential_energy,"
      "monitor_current_total_control_energy,"
      "inertia_model_comparison_valid,"
      "runtime_model_joint_kinetic_energy,"
      "prediction_model_joint_kinetic_energy,"
      "inertia_model_kinetic_energy_error,"
      "inertia_model_difference_frobenius_norm,"
      "inertia_model_difference_relative_frobenius_norm,"
      "inertia_model_difference_max_abs,"
      "inertia_model_difference_max_abs_row,"
      "inertia_model_difference_max_abs_col,"
      "inertia_model_energy_ratio_valid,"
      "inertia_model_min_energy_ratio,"
      "inertia_model_max_energy_ratio,"
      "joint_limit_unsafe,"
      "joint_limit_index,joint_position_violation,joint_velocity_violation,"
      "joint_acceleration_violation,joint_torque_violation";
    const std::size_t reserved_plan_steps = std::max<std::size_t>(
      128,
      std::max<std::size_t>(
        static_cast<std::size_t>(std::max(1, local_replan_horizon_steps_)),
        async_planning_lead_steps_ + async_verified_horizon_steps_));
    if (!prediction_log_writer_.start(
        prediction_log_file_path_,
        prediction_header,
        prediction_log_max_queue_size_,
        log_batch_size_,
        log_flush_period_sec_,
        [this](std::ostream & output,
        const PredictionLogRecord & record) {
          writeShieldPredictionTrajectory(
            output,
            record.wall_time,
            record.nominal_guess_time,
            record.current_q,
            record.current_dq,
            record.current_position,
            record.current_orientation,
            record.ee_twist,
            record.inertia,
            record.Jv,
            record.K_runtime,
            record.D_runtime,
            record.human_workspace,
            record.human_workspace_active,
            record.human_workspace_assumed_clear,
            record.evaluated_plan,
            record.joint_prediction_trace,
            record.async_timing,
            record.monitor,
            record.mode,
            record.candidate_verified,
            record.accepted_plan_generation,
            record.executing_last_verified_monitored,
            record.monitor_total_ms,
            record.planner_ms,
            record.plan_build_ms,
            record.monitor_eval_ms,
            record.async_source ? "async" : "sync");
        },
        [reserved_plan_steps](PredictionLogRecord & record) {
          record.evaluated_plan.intended.reserve(reserved_plan_steps);
          record.evaluated_plan.failsafe.reserve(reserved_plan_steps);
          record.joint_prediction_trace.reserve(4 * reserved_plan_steps + 1);
        }))
    {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to start asynchronous prediction logger: %s",
        prediction_log_file_path_.c_str());
      control_log_writer_.stop();
      return false;
    }
  }

  return true;
}

void ReachableCartesianImpedanceController::stopLogWriters()
{
  prediction_log_writer_.stop();
  control_log_writer_.stop();

  if (!enable_error_logging_ && !enable_prediction_logging_) {
    return;
  }

  const std::size_t control_enqueued = control_log_writer_.enqueuedCount();
  const std::size_t control_written = control_log_writer_.writtenCount();
  const std::size_t control_dropped = control_log_writer_.droppedCount();
  const std::size_t prediction_enqueued =
    prediction_log_writer_.enqueuedCount();
  const std::size_t prediction_written = prediction_log_writer_.writtenCount();
  const std::size_t prediction_dropped =
    prediction_log_writer_.droppedCount();
  const std::size_t schema_mismatches =
    control_log_column_mismatch_count_.load(std::memory_order_relaxed);
  const std::uint64_t monitor_inputs_published =
    async_monitor_input_publish_count_.load(std::memory_order_relaxed);
  const std::uint64_t monitor_inputs_overwritten =
    async_monitor_input_overwrite_count_.load(std::memory_order_relaxed);
  const std::uint64_t monitor_worker_processed =
    async_monitor_worker_processed_count_.load(std::memory_order_relaxed);
  const std::uint64_t monitor_outputs_overwritten =
    async_monitor_output_overwrite_count_.load(std::memory_order_relaxed);
  const std::uint64_t monitor_outputs_consumed =
    async_monitor_output_consumed_count_.load(std::memory_order_relaxed);

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Asynchronous logs drained: control=%zu/%zu dropped=%zu, "
    "prediction=%zu/%zu dropped=%zu, schema_mismatch=%zu, "
    "monitor published/processed/consumed=%lu/%lu/%lu, input/output overwrite=%lu/%lu",
    control_written,
    control_enqueued,
    control_dropped,
    prediction_written,
    prediction_enqueued,
    prediction_dropped,
    schema_mismatches,
    static_cast<unsigned long>(monitor_inputs_published),
    static_cast<unsigned long>(monitor_worker_processed),
    static_cast<unsigned long>(monitor_outputs_consumed),
    static_cast<unsigned long>(monitor_inputs_overwritten),
    static_cast<unsigned long>(monitor_outputs_overwritten));

  if (!error_log_run_dir_.empty()) {
    const std::filesystem::path run_info_path =
      std::filesystem::path(error_log_run_dir_) / "run_info.txt";
    std::ofstream run_info_file(
      run_info_path, std::ios::out | std::ios::app);
    if (run_info_file.is_open()) {
      run_info_file
        << "control_log_enqueued: " << control_enqueued << "\n"
        << "control_log_written: " << control_written << "\n"
        << "control_log_dropped: " << control_dropped << "\n"
        << "prediction_log_enqueued: " << prediction_enqueued << "\n"
        << "prediction_log_written: " << prediction_written << "\n"
        << "prediction_log_dropped: " << prediction_dropped << "\n"
        << "control_log_schema_mismatch: " << schema_mismatches << "\n"
        << "async_monitor_inputs_published: "
        << monitor_inputs_published << "\n"
        << "async_monitor_inputs_overwritten: "
        << monitor_inputs_overwritten << "\n"
        << "async_monitor_worker_processed: "
        << monitor_worker_processed << "\n"
        << "async_monitor_outputs_overwritten: "
        << monitor_outputs_overwritten << "\n"
        << "async_monitor_outputs_consumed: "
        << monitor_outputs_consumed << "\n"
        << "async_monitor_schedule_late_cycles: "
        << async_monitor_schedule_late_cycles_ << "\n"
        << "async_monitor_schedule_skipped_slots: "
        << async_monitor_schedule_skipped_slots_ << "\n";
    }
  }
}

// ============================================================================
// computeImpedanceTorque -- corrected dynamic-consistent true branch

}  // namespace cps_controllers
