// Copyright (c) 2026
// Fixed-period monitor handoff and execution of the verified command stream.
#include <algorithm>
#include <cmath>
#include <limits>
#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>
#include <cps_controllers/reachable_cartesian_impedance/monitor_policy.hpp>
#include "math.hpp"
#include "timing.hpp"

namespace cps_controllers {
namespace {
constexpr double kMinDt = 1e-6;
using detail::nanosecondsToMilliseconds;
using detail::steadyNowNanoseconds;
}

ShieldExecutionDecision ReachableCartesianImpedanceController::updateMonitoredCommand(
    double wall_time, double nominal_guess_time,
    const Vector7d& q, const Vector7d& dq,
    const Vector3d& current_position, const Quaterniond& current_orientation,
    const Vector6d& ee_twist, const Matrix7d& inertia,
    const Matrix67d& J_geo, const Vector7d& coriolis,
    std::uint64_t control_loop_sequence, bool human_workspace_assumed_clear,
    bool& async_output_processed_this_cycle) {
  const Matrix37d Jv = J_geo.topRows<3>();
  const bool human_workspace_available = human_workspace_active_ || human_workspace_assumed_clear;
  ShieldExecutionDecision shield_dec;
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
  } else {
    // Finish acceptance before publishing another request. Merely taking a
    // result from the mailbox does not yet update the source-plan generation.
    AsyncMonitorOutput async_output;
    const bool async_output_available =
        takeAsyncMonitorOutput(&async_output);

    if (async_output_available) {
      async_output_processed_this_cycle = true;
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
      async_timing.worker_thread_cpu_ms = async_output.worker_thread_cpu_ms;
      async_timing.worker_non_cpu_ms = async_output.worker_non_cpu_ms;
      async_timing.worker_voluntary_context_switches = async_output.worker_voluntary_context_switches;
      async_timing.worker_involuntary_context_switches = async_output.worker_involuntary_context_switches;
      async_timing.worker_rollout_steps = async_output.decision.joint_prediction_trace.empty()
          ? 0 : async_output.decision.joint_prediction_trace.size() - 1;
      async_timing.intended_command_count = async_output.decision.evaluated_plan.intended.size();
      async_timing.failsafe_command_count = async_output.decision.evaluated_plan.failsafe.size();
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
          detail::withinCommittedPrefix(
              async_output.input.control_loop_sequence,
              control_loop_sequence,
              async_output.input.committed_prefix.size());

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

      // Never carry an assume-clear calibration result into normal mode, or
      // reuse a live-workspace result after the provider has become stale.
      const bool async_output_workspace_policy_matches =
          (async_output.input.human_workspace_assumed_clear &&
           human_workspace_assumed_clear) ||
          (async_output.input.human_workspace_active &&
           human_workspace_active_);
      const bool async_output_recovery_epoch_matches =
          async_output.input.energy_recovery_epoch == energy_recovery_epoch_;
      bool async_output_recovery_state_matches = true;
      if (enable_safety_monitor_ && enable_runtime_energy_scaling_ &&
          async_output.decision.candidate_verified) {
        const auto& trace = async_output.decision.joint_prediction_trace;
        const double handoff_time =
            async_output.decision.evaluated_plan.anchor.t +
            static_cast<double>(async_plan_elapsed_steps) * local_replan_dt_;
        const auto predicted = std::lower_bound(
            trace.begin(), trace.end(), handoff_time - 1e-9,
            [](const JointPredictionSample& sample, double t) { return sample.t < t; });
        async_output_recovery_state_matches = predicted != trace.end() &&
            std::abs(predicted->t - handoff_time) <= 1e-9 &&
            cps_safety_monitor::energyRecoveryStateMatchesPrediction(
                *predicted, energy_recovery_state_);
      }
      const bool async_output_usable =
          async_output_matches_source_plan &&
          async_output_recovery_epoch_matches &&
          async_output_recovery_state_matches &&
          async_output_workspace_policy_matches &&
          async_output_matches_calibration_target &&
          async_output_before_activation;
      async_timing.source_plan_matches_at_handoff =
          async_output_matches_source_plan;
      async_timing.recovery_epoch_matches_at_handoff =
          async_output_recovery_epoch_matches;
      async_timing.recovery_state_matches_at_handoff =
          async_output_recovery_state_matches;
      async_timing.output_usable = async_output_usable;
      async_timing.handoff_rejection_mask =
          (async_output_matches_source_plan ? 0U : 1U) |
          (async_output_recovery_epoch_matches ? 0U : 2U) |
          (async_output_recovery_state_matches ? 0U : 4U) |
          (async_output_workspace_policy_matches ? 0U : 8U) |
          (async_output_matches_calibration_target ? 0U : 16U) |
          (async_output_before_activation ? 0U : 32U);
      async_timing.candidate_verified_at_handoff = async_output.decision.candidate_verified;
      if (!async_output_usable) {
        async_output_rejection_reason = FallbackReason::kAsyncOutputUnavailable;
      }
      if (!async_output_before_activation) {
        ++async_activation_deadline_miss_count_;
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
      async_timing.plan_accepted = accepted_plan_generation != 0;
      last_async_monitor_timing_ = async_timing;
      if (async_output.decision.has_evaluated_plan) {
        logShieldPredictionTrajectory(
            async_output.input.wall_time,
            async_output.input.nominal_guess_time,
            async_output.input.q,
            async_output.input.dq,
            async_output.input.current_position,
            async_output.input.ee_twist,
            async_output.input.inertia,
            async_output.input.Jv,
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
        if (verified_output_acceptable_now &&
            async_output.decision.evaluated_plan.valid) {
          last_verified_plan_ = async_output.decision.evaluated_plan;
          last_verified_plan_.valid = true;
          alignVerifiedPlanExecutionIndex(
              &last_verified_plan_,
              async_plan_elapsed_steps);
          ++last_verified_plan_generation_;
          current_source_plan_generation_.store(
              last_verified_plan_generation_, std::memory_order_release);
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

    // Apply the previous result before capturing the next source generation
    // and committed prefix. Publishing first would immediately invalidate a
    // new request whenever this cycle accepts a completed plan.
    // A calibration action is a one-shot experiment: after arming its first
    // post-action monitor input, do not overwrite that input with newer ones
    // while the worker is still evaluating it.
    const bool calibration_waiting_for_target_output =
        active_cartesian_via_points_calibration_ &&
        !calibration_plan_latched_ &&
        !calibration_target_failed_ &&
        calibration_monitor_input_sequence_ != 0;
    const bool monitor_input_due =
        control_loop_sequence >= next_async_monitor_control_sequence_ &&
        !calibration_plan_latched_ &&
        !calibration_waiting_for_target_output;
    const bool request_slot_available = async_request_gate_.canPublish();
    if (monitor_input_due && !request_slot_available) {
      ++async_monitor_busy_deferred_cycles_;
    }
    // Keep the configured monitor phase, but never pipeline a second snapshot
    // against a plan that the outstanding result may replace. Coalesce missed
    // slots at the next submission; the servo loop never waits for this gate.
    const bool publish_monitor_input = monitor_input_due && request_slot_available;

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
      async_input.Jv = Jv;
      async_input.J_geo = J_geo;
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
      async_input.energy_recovery_state = energy_recovery_state_;
      async_input.energy_recovery_epoch = energy_recovery_epoch_;
      async_input.commanded_path_rate = commanded_path_rate_;
      // The nominal generator always requests the configured path rate.
      // Inside the collision area, energy adaptation changes impedance gains
      // according to the energy budget, including when T > L_max. It never
      // freezes effective time.
      async_input.target_path_rate = path_time_rate_target_;
      async_input.reanchor_path_kinematics = false;
      async_input.reanchor_path_rate = 0.0;
      async_input.reanchor_path_acceleration = 0.0;
      async_input.committed_prefix.reserve(monitor_period_steps_);
      // Commit the next full monitor period of the already verified stream,
      // including braking or terminal hold when its intended tail is exhausted.
      for (std::size_t offset = 0; offset < monitor_period_steps_; ++offset) {
        ImpedanceSample committed_command;
        bool command_available = getVerifiedTrajectoryCommandAtOffset(
            last_verified_plan_, offset, &committed_command);
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
          committed_command.energy_recovery_exit_allowed = false;
          command_available = true;
        }
        if (!command_available) {
          break;
        }
        async_input.committed_prefix.push_back(committed_command);
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
            std::max<std::uint64_t>(1, monitor_period_steps_);
        next_async_monitor_control_sequence_ += period_cycles;
        while (next_async_monitor_control_sequence_ <=
               control_loop_sequence) {
          next_async_monitor_control_sequence_ += period_cycles;
          ++async_monitor_schedule_skipped_slots_;
        }
      }
    }

    shield_dec = last_shield_decision_;
    if (!last_verified_plan_.valid || last_verified_plan_.intended.empty()) {
      // Bootstrap without running planning or monitoring in the control thread.
      shield_dec.executing_last_verified_monitored = true;
      shield_dec.fallback_reason = FallbackReason::kNoVerifiedPlanAvailable;
      shield_dec.command = makeEmergencyStopCommand(
          current_position, current_orientation, wall_time);
      if (last_commanded_sample_valid_) {
        shield_dec.command = last_commanded_sample_;
        shield_dec.command.t = wall_time;
        shield_dec.command.dp.setZero();
        shield_dec.command.ddp.setZero();
        shield_dec.command.w.setZero();
        shield_dec.command.dw.setZero();
        shield_dec.command.failsafe = true;
      }
    } else {
      // A finite verified stream supplies intended, braking, then terminal hold.
      // No independent age timer or late activation can extend that stream.
      shield_dec.command = getNextVerifiedTrajectoryCommandFromCache(true);
      shield_dec.executing_last_verified_monitored =
          !last_shield_decision_valid_ ||
          shield_dec.executing_last_verified_monitored ||
          shield_dec.command.failsafe ||
          async_output_rejection_reason != FallbackReason::kNone ||
          shouldRejectCandidateWithMonitor(shield_dec.monitor, human_workspace_available);
      if (!human_workspace_available) {
        shield_dec.fallback_reason = FallbackReason::kHumanWorkspaceUnavailable;
      } else if (async_output_rejection_reason != FallbackReason::kNone) {
        shield_dec.fallback_reason = async_output_rejection_reason;
      }
    }
    last_shield_decision_.command = shield_dec.command;
    last_shield_decision_.executing_last_verified_monitored =
        shield_dec.executing_last_verified_monitored;

  }
  return shield_dec;
}

}  // namespace cps_controllers
