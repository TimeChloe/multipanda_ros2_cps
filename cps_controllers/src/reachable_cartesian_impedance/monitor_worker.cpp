// Copyright (c) 2026
// Monitor worker scheduling and bounded mailbox transport.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include <rclcpp/rclcpp.hpp>

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>

#include "timing.hpp"

namespace {
using cps_controllers::detail::nanosecondsToMilliseconds;
using cps_controllers::detail::steadyNowNanoseconds;
}  // namespace

namespace cps_controllers
{

void ReachableCartesianImpedanceController::safetyMonitorWorkerLoop()
{
  bool worker_affinity_applied = false;
  if (monitor_worker_cpu_affinity_ >= 0) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if (monitor_worker_cpu_affinity_ < CPU_SETSIZE) {
      CPU_SET(monitor_worker_cpu_affinity_, &cpu_set);
      const int affinity_result = pthread_setaffinity_np(
        pthread_self(), sizeof(cpu_set), &cpu_set);
      if (affinity_result == 0) {
        worker_affinity_applied = true;
        RCLCPP_INFO(
          get_node()->get_logger(),
          "Safety monitor worker pinned to CPU %d.",
          monitor_worker_cpu_affinity_);
      } else {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Failed to pin safety monitor worker to CPU %d: %s",
          monitor_worker_cpu_affinity_,
          std::strerror(affinity_result));
      }
    } else {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "monitor_worker_cpu_affinity=%d exceeds CPU_SETSIZE=%d; affinity disabled.",
        monitor_worker_cpu_affinity_,
        CPU_SETSIZE);
    }
  }

  if (monitor_worker_realtime_priority_ > 0) {
    if (!worker_affinity_applied) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Safety monitor SCHED_FIFO priority %d was requested without a "
        "successfully applied dedicated CPU affinity; keeping the default "
        "scheduler to protect the controller update thread.",
        monitor_worker_realtime_priority_);
    } else {
      sched_param scheduling_parameters{};
      scheduling_parameters.sched_priority =
        monitor_worker_realtime_priority_;
      const int scheduling_result = pthread_setschedparam(
        pthread_self(), SCHED_FIFO, &scheduling_parameters);
      if (scheduling_result == 0) {
        RCLCPP_INFO(
          get_node()->get_logger(),
          "Safety monitor worker uses SCHED_FIFO priority %d on its dedicated CPU.",
          monitor_worker_realtime_priority_);
      } else {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Failed to set safety monitor worker SCHED_FIFO priority %d: %s. "
          "Continuing with the default scheduler.",
          monitor_worker_realtime_priority_,
          std::strerror(scheduling_result));
      }
    }
  }

  double last_visualization_request_wall_time = -1.0;

  while (safety_monitor_worker_running_.load()) {
    AsyncMonitorInput input;

    {
      std::unique_lock<std::mutex> lock(async_input_mutex_);
      async_input_cv_.wait(
        lock, [&]() {
          return async_input_pending_ || !safety_monitor_worker_running_.load();
        });

      if (!safety_monitor_worker_running_.load()) {
        break;
      }

      input = std::move(latest_async_input_);
      async_input_pending_ = false;
    }

    if (input.source_plan_generation !=
      current_source_plan_generation_.load(std::memory_order_acquire))
    {
      async_stale_before_compute_count_.fetch_add(1, std::memory_order_relaxed);
      async_request_gate_.discardedByWorker(input.sequence);
      continue;
    }
    const std::int64_t worker_start_ns = steadyNowNanoseconds();
    const auto cpu_start = detail::workerCpuSample();
    AsyncMonitorOutput output;
    output.sequence = input.sequence;
    output.valid = true;
    output.worker_start_steady_time_ns = worker_start_ns;
    output.worker_queue_wait_ms = nanosecondsToMilliseconds(
      std::max<std::int64_t>(
        0, worker_start_ns - input.publish_steady_time_ns));
    output.decision =
      computeShieldDecisionForAsyncInput(input);
    async_monitor_worker_processed_count_.fetch_add(1, std::memory_order_relaxed);
    if (input.source_plan_generation !=
      current_source_plan_generation_.load(std::memory_order_acquire))
    {
      async_stale_after_compute_count_.fetch_add(1, std::memory_order_relaxed);
      async_request_gate_.discardedByWorker(input.sequence);
      continue;
    }

    const bool reachable_set_output_due =
      enable_reachable_set_visualization_ &&
      reachable_set_visualization_pub_ &&
      human_reachable_set_pub_ &&
      (last_visualization_request_wall_time < 0.0 ||
      input.wall_time < last_visualization_request_wall_time ||
      input.wall_time - last_visualization_request_wall_time >=
      reachable_set_visualization_period_sec_);
    ReachableSetOutputSnapshot reachable_set_snapshot;
    if (reachable_set_output_due) {
      reachable_set_snapshot.wall_time = input.wall_time;
      reachable_set_snapshot.current_q = input.q;
      reachable_set_snapshot.human_workspace = input.human_workspace;
      reachable_set_snapshot.human_workspace_active =
        input.human_workspace_active;
      reachable_set_snapshot.human_workspace_assumed_clear =
        input.human_workspace_assumed_clear;
      reachable_set_snapshot.current_contact_energy_unsafe =
        output.decision.monitor.contact_relevant_for_energy &&
        output.decision.monitor.current_joint_energy_valid &&
        output.decision.monitor.current_total_control_energy >
        std::max(0.0, energy_budget_joule_);
      reachable_set_snapshot.first_contact_interval_index =
        output.decision.monitor.first_contact_interval_index;
      reachable_set_snapshot.first_energy_unsafe_contact_interval_index =
        output.decision.monitor.first_energy_unsafe_contact_interval_index;
      reachable_set_snapshot.robot_reach_alpha_valid =
        output.decision.monitor.robot_reach_alpha_valid;
      reachable_set_snapshot.robot_reach_alpha = output.decision.monitor.robot_reach_alpha;
      reachable_set_snapshot.joint_prediction_trace =
        output.decision.joint_prediction_trace;
    }

    output.input = std::move(input);
    const auto cpu_finish = detail::workerCpuSample();
    output.worker_finish_steady_time_ns = steadyNowNanoseconds();
    output.worker_compute_ms = nanosecondsToMilliseconds(
      std::max<std::int64_t>(
        0,
        output.worker_finish_steady_time_ns -
        output.worker_start_steady_time_ns));
    if (cpu_start.cpu_ns >= 0 && cpu_finish.cpu_ns >= cpu_start.cpu_ns) {
      output.worker_thread_cpu_ms = nanosecondsToMilliseconds(cpu_finish.cpu_ns - cpu_start.cpu_ns);
      output.worker_non_cpu_ms = std::max(0.0, output.worker_compute_ms - output.worker_thread_cpu_ms);
    }
    if (cpu_start.voluntary_switches >= 0 && cpu_finish.voluntary_switches >= 0) {
      output.worker_voluntary_context_switches =
        cpu_finish.voluntary_switches - cpu_start.voluntary_switches;
      output.worker_involuntary_context_switches =
        cpu_finish.involuntary_switches - cpu_start.involuntary_switches;
    }
    const std::uint64_t completed_input_sequence = output.sequence;
    const auto publish_result =
      async_output_mailbox_.publish(std::move(output));
    if (!publish_result.published) {
      async_request_gate_.discardedByWorker(completed_input_sequence);
    }
    const std::size_t discarded_outputs =
      publish_result.overwritten_ready +
      static_cast<std::size_t>(!publish_result.published);
    if (discarded_outputs > 0) {
      async_monitor_output_overwrite_count_.fetch_add(
        discarded_outputs, std::memory_order_relaxed);
    }
    // Rendering and ROS publication must not delay the next monitor request.
    if (reachable_set_output_due) {
      last_visualization_request_wall_time = reachable_set_snapshot.wall_time;
      visualization_mailbox_.publish(std::move(reachable_set_snapshot));
    }
  }
}

void ReachableCartesianImpedanceController::startSafetyMonitorWorker()
{
  if (!diagnostics_worker_running_.exchange(true)) {
    profiling_mailbox_.resetStopped();
    visualization_mailbox_.resetStopped();
    diagnostics_worker_thread_ =
      std::thread(&ReachableCartesianImpedanceController::diagnosticsWorkerLoop, this);
  }
  if (safety_monitor_worker_running_.load()) {
    return;
  }

  async_output_mailbox_.resetStopped();
  async_stale_before_compute_count_.store(0, std::memory_order_relaxed);
  async_stale_after_compute_count_.store(0, std::memory_order_relaxed);
  safety_monitor_worker_running_.store(true);
  async_input_pending_ = false;

  safety_monitor_worker_thread_ =
    std::thread(&ReachableCartesianImpedanceController::safetyMonitorWorkerLoop, this);
}

void ReachableCartesianImpedanceController::stopSafetyMonitorWorker()
{
  safety_monitor_worker_running_.store(false);
  async_input_cv_.notify_all();

  if (safety_monitor_worker_thread_.joinable()) {
    safety_monitor_worker_thread_.join();
  }
  diagnostics_worker_running_.store(false);
  if (diagnostics_worker_thread_.joinable()) {
    diagnostics_worker_thread_.join();
  }
}

bool ReachableCartesianImpedanceController::publishAsyncMonitorInput(
    AsyncMonitorInput input) {
  if (!safety_monitor_worker_running_.load() ||
      !async_request_gate_.canPublish()) {
    return false;
  }

  if (async_input_mutex_.try_lock()) {
    if (async_input_pending_) {
      async_monitor_input_overwrite_count_.fetch_add(
          1, std::memory_order_relaxed);
    }
    async_request_gate_.published(input.sequence);
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
    async_request_gate_.consumed(output->sequence);
    async_monitor_output_consumed_count_.fetch_add(
        1, std::memory_order_relaxed);
  }
  return result.taken;
}
// ============================================================================

}  // namespace cps_controllers
