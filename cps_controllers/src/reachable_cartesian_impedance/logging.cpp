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

Matrix3d positiveSemidefinitePart(const Matrix3d & matrix)
{
  const Matrix3d symmetric = 0.5 * (matrix + matrix.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix3d> eig(symmetric);
  if (eig.info() != Eigen::Success) {
    return symmetric;
  }
  Matrix3d psd =
    eig.eigenvectors() *
    eig.eigenvalues().cwiseMax(0.0).asDiagonal() *
    eig.eigenvectors().transpose();
  return 0.5 * (psd + psd.transpose());
}

double maxTrackingBlockRadius(const Matrix6d & tube, int block_start)
{
  const Matrix3d block =
    0.5 * (tube.block<3, 3>(block_start, block_start) +
    tube.block<3, 3>(block_start, block_start).transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix3d> eig(block);
  if (eig.info() != Eigen::Success) {
    return std::sqrt(std::max(0.0, block.norm()));
  }
  return std::sqrt(std::max(0.0, eig.eigenvalues().maxCoeff()));
}

Matrix6d propagateTrackingTube(
  const Matrix6d & tube,
  const Matrix3d & task_inertia_inv,
  const Matrix3d & Kp,
  const Matrix3d & Dp,
  double dt,
  double acc_error_bound)
{
  const double h = std::max(dt, kMinDt);
  Matrix6d A = Matrix6d::Identity();
  A.topRightCorner<3, 3>() = h * Matrix3d::Identity();
  A.bottomLeftCorner<3, 3>() = -h * task_inertia_inv * Kp;
  A.bottomRightCorner<3, 3>() =
    Matrix3d::Identity() - h * task_inertia_inv * Dp;

  Eigen::Matrix<double, 6, 3> B = Eigen::Matrix<double, 6, 3>::Zero();
  B.topRows<3>() = 0.5 * h * h * Matrix3d::Identity();
  B.bottomRows<3>() = h * Matrix3d::Identity();

  const double acc_bound = std::max(0.0, acc_error_bound);
  const Matrix6d propagated =
    A * tube * A.transpose() + acc_bound * acc_bound * B * B.transpose();
  return 0.5 * (propagated + propagated.transpose());
}


}  // namespace

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
  const VerifiedPlan & evaluated_plan,
  const std::vector<JointPredictionSample> & joint_prediction_trace,
  const AsyncMonitorTiming & async_timing,
  const MonitorResult & monitor,
  int mode,
  bool candidate_verified,
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
      record.evaluated_plan = evaluated_plan;
      record.joint_prediction_trace = joint_prediction_trace;
      record.async_timing = async_timing;
      record.monitor = monitor;
      record.mode = mode;
      record.candidate_verified = candidate_verified;
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
  const VerifiedPlan & evaluated_plan,
  const std::vector<JointPredictionSample> & joint_prediction_trace,
  const AsyncMonitorTiming & async_timing,
  const MonitorResult & monitor,
  int mode,
  bool candidate_verified,
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
    Vector7d joint_q{Vector7d::Constant(
        std::numeric_limits<double>::quiet_NaN())};
    Vector7d joint_dq{Vector7d::Constant(
        std::numeric_limits<double>::quiet_NaN())};
    double d_segment{0.0};
    Vector3d human_center_end{Vector3d::Zero()};
    bool contact_possible{false};
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

  const double acc_error_bound =
    std::max(0.0, tracking_acc_error_bound_);

  Matrix3d task_inertia_inv = Jv * inertia.inverse() * Jv.transpose();
  task_inertia_inv.diagonal().array() += kSmallPositive;
  Matrix6d K_exec = K_runtime;
  Matrix6d D_exec = D_runtime;

  Vector3d x_pred = collision_center;
  Vector3d v_pred = collision_twist.head<3>();
  Matrix6d tracking_tube = Matrix6d::Zero();
  double t_prev = collision_plan.anchor.t;
  std::vector<PredictionRow> rows;
  rows.reserve(1 + collision_plan.intended.size() + collision_plan.failsafe.size());
  std::size_t joint_trace_index = 0;

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
      const Matrix3d Kp = positiveSemidefinitePart(Kp_raw);
      const Matrix3d Dp = positiveSemidefinitePart(Dp_raw);
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

      const Matrix6d tracking_tube_next =
        propagateTrackingTube(
        tracking_tube,
        task_inertia_inv,
        Kp,
        Dp,
        dtp,
        acc_error_bound);
      const double rho_p_segment =
        std::max(
        maxTrackingBlockRadius(tracking_tube, 0),
        maxTrackingBlockRadius(tracking_tube_next, 0));
      const double inflated_contact_radius_segment =
        human_workspace.inflatedCollisionRadius(
        ee_collision_radius_,
        rho_p_segment);

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
        }
      }
      row.human_center_end = human_workspace.centerAtTime(segment_end_time_sec);
      row.d_segment =
        human_workspace.signedDistanceSegmentToInflatedSphere(
        x_pred,
        x_next,
        inflated_contact_radius_segment,
        segment_start_time_sec,
        segment_end_time_sec);
      row.contact_possible = row.d_segment <= 0.0;
      row.Kx = K_exec(0, 0);
      row.Ky = K_exec(1, 1);
      row.Kz = K_exec(2, 2);
      row.Dx = D_exec(0, 0);
      row.Dy = D_exec(1, 1);
      row.Dz = D_exec(2, 2);
      rows.push_back(row);

      x_pred = x_next;
      v_pred = v_next;
      tracking_tube = tracking_tube_next;
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

  const double actual_collision_distance =
    human_workspace.signedDistanceToInflatedSphere(
    collision_center,
    human_workspace.inflatedCollisionRadius(
      ee_collision_radius_,
      0.0),
    wall_time);
  const Vector3d actual_human_center = human_workspace.centerAtTime(wall_time);

  output << std::fixed << std::setprecision(9);
  for (const auto & row : rows) {
    output
      << wall_time << "," << nominal_guess_time << ","
      << (source == nullptr ? "" : source) << ","
      << static_cast<int>(async_timing.valid) << ","
      << async_timing.input_sequence << ","
      << async_timing.input_control_loop_sequence << ","
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
      << static_cast<int>(executing_last_verified_monitored) << ","
      << static_cast<int>(monitor.predicted_trigger) << ","
      << static_cast<int>(monitor.monitored_contact_possible) << ","
      << monitor_plan.intended.size() << ","
      << monitor_plan.failsafe.size() << ","
      << row.stage << "," << row.index << ","
      << static_cast<int>(row.failsafe) << ","
      << row.sample_t << "," << row.dt << ","
      << collision_center(0) << "," << collision_center(1) << "," << collision_center(2) << ","
      << actual_human_center(0) << "," << actual_human_center(1) << "," << actual_human_center(2) <<
      ","
      << actual_collision_distance << ","
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
      << row.joint_q(0) << "," << row.joint_q(1) << "," << row.joint_q(2) << ","
      << row.joint_q(3) << "," << row.joint_q(4) << "," << row.joint_q(5) << ","
      << row.joint_q(6) << ","
      << row.joint_dq(0) << "," << row.joint_dq(1) << "," << row.joint_dq(2) << ","
      << row.joint_dq(3) << "," << row.joint_dq(4) << "," << row.joint_dq(5) << ","
      << row.joint_dq(6) << ","
      << row.human_center_end(0) << "," << row.human_center_end(1) << "," <<
      row.human_center_end(2) << ","
      << row.d_segment << ","
      << static_cast<int>(row.contact_possible) << ","
      << row.Kx << "," << row.Ky << "," << row.Kz << ","
      << row.Dx << "," << row.Dy << "," << row.Dz << ","
      << monitor.worst_case_cartesian_kinetic_energy_ub << ","
      << monitor.worst_case_joint_kinetic_energy_ub << ","
      << monitor.worst_case_cartesian_potential_energy_ub << ","
      << monitor.worst_case_total_control_energy_ub << ","
      << monitor.terminal_energy_ub << ","
      << monitor.workspace_distance_margin << ","
      << monitor.h_monitored_energy << ","
      << static_cast<int>(monitor.current_cartesian_energy_valid) << ","
      << monitor.current_cartesian_kinetic_energy << ","
      << static_cast<int>(monitor.current_joint_energy_valid) << ","
      << monitor.current_joint_kinetic_energy << ","
      << monitor.current_cartesian_potential_energy << ","
      << monitor.current_total_control_energy << ","
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
    "generator_target_path_rate,measured_path_rate,"
    "measured_path_rate_valid,measured_path_acceleration,"
    "measured_path_acceleration_valid,mode,execution_stage,fallback_reason,"
    "plan_failure_reason,candidate_verified,monitor_prediction_valid,"
    "predicted_trigger,predicted_contact_possible,"
    "monitored_contact_possible,contact_relevant_for_energy,"
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
    "tau_cmd_norm,torque_rate_limited,torque_rate_max_ratio,Kx,Ky,Kz,Dx,Dy,Dz,"
    "worst_case_contact_time,worst_case_workspace_distance_at_candidate,"
    "worst_case_cartesian_kinetic_energy_ub,"
    "worst_case_joint_kinetic_energy_ub,"
    "worst_case_cartesian_potential_energy_ub,"
    "worst_case_total_control_energy_ub,"
    "h_monitored_energy,terminal_energy_ub,h_terminal_energy,"
    "worst_case_pos_error_radius,worst_case_vel_error_radius,"
    "monitor_current_cartesian_energy_valid,"
    "monitor_current_cartesian_kinetic_energy,"
    "monitor_current_joint_energy_valid,"
    "monitor_current_joint_kinetic_energy,"
    "monitor_current_cartesian_potential_energy,"
    "monitor_current_total_control_energy,monitored_steps,"
    "monitored_intended_steps,"
    "monitored_failsafe_steps,control_loop_sequence,"
    "monitor_period_control_cycles,next_async_monitor_control_sequence,"
    "last_async_input_publish_control_sequence,async_timing_valid,"
    "async_monitor_input_sequence,async_monitor_input_control_sequence,"
    "async_monitor_scheduled_control_sequence,async_monitor_publish_lateness_cycles,"
    "async_monitor_worker_queue_wait_ms,async_monitor_worker_compute_ms,"
    "async_monitor_output_handoff_ms,async_monitor_end_to_end_ms,"
    "async_monitor_inputs_published,async_monitor_inputs_overwritten,"
    "async_monitor_worker_processed,async_monitor_outputs_overwritten,"
    "async_monitor_outputs_consumed,async_monitor_schedule_late_cycles,"
    "async_monitor_schedule_skipped_slots,verified_plan_age_sec,"
    "verified_next_intended_exec_index,verified_next_failsafe_exec_index,"
    "cartesian_energy_budget_active,cartesian_effective_time_frozen,"
    "cartesian_energy_lambda_valid,cartesian_energy_scale,"
    "joint_kinetic_energy,cartesian_potential_energy,"
    "total_control_energy,energy_budget_joule,mujoco_contact_value,"
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
      "wall_time_sec,nominal_time_sec,source,async_timing_valid,"
      "monitor_input_sequence,monitor_input_control_loop_sequence,"
      "monitor_scheduled_control_loop_sequence,monitor_publish_lateness_cycles,"
      "worker_queue_wait_ms,worker_compute_ms,output_handoff_ms,"
      "monitor_end_to_end_ms,monitor_total_ms,planner_ms,"
      "plan_build_ms,monitor_eval_ms,mode,candidate_verified,"
      "executing_last_verified_monitored,predicted_trigger,"
      "monitored_contact_possible,plan_intended_steps,plan_failsafe_steps,"
      "stage,index,is_failsafe_sample,sample_t,dt,actual_collision_px,actual_collision_py,"
      "actual_collision_pz,actual_human_center_px,actual_human_center_py,"
      "actual_human_center_pz,actual_collision_distance,"
      "measured_q1,measured_q2,measured_q3,measured_q4,measured_q5,"
      "measured_q6,measured_q7,measured_dq1,measured_dq2,measured_dq3,"
      "measured_dq4,measured_dq5,measured_dq6,measured_dq7,collision_target_px,"
      "collision_target_py,collision_target_pz,pred_start_px,pred_start_py,"
      "pred_start_pz,pred_start_vx,pred_start_vy,pred_start_vz,pred_next_px,"
      "pred_next_py,pred_next_pz,pred_next_vx,pred_next_vy,pred_next_vz,"
      "pred_ax,pred_ay,pred_az,pred_joint_state_valid,pred_joint_sample_t,"
      "pred_q1,pred_q2,pred_q3,pred_q4,pred_q5,pred_q6,pred_q7,"
      "pred_dq1,pred_dq2,pred_dq3,pred_dq4,pred_dq5,pred_dq6,pred_dq7,"
      "human_center_end_px,human_center_end_py,"
      "human_center_end_pz,distance_segment,contact_possible,"
      "Kx,Ky,Kz,Dx,Dy,Dz,"
      "monitor_worst_case_cartesian_kinetic_energy_ub,"
      "monitor_worst_case_joint_kinetic_energy_ub,"
      "monitor_worst_case_cartesian_potential_energy_ub,"
      "monitor_worst_case_total_control_energy_ub,"
      "monitor_terminal_energy_ub,monitor_workspace_distance_margin,"
      "monitor_h_monitored_energy,"
      "monitor_current_cartesian_energy_valid,"
      "monitor_current_cartesian_kinetic_energy,"
      "monitor_current_joint_energy_valid,"
      "monitor_current_joint_kinetic_energy,"
      "monitor_current_cartesian_potential_energy,"
      "monitor_current_total_control_energy,joint_limit_unsafe,"
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
            record.evaluated_plan,
            record.joint_prediction_trace,
            record.async_timing,
            record.monitor,
            record.mode,
            record.candidate_verified,
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
