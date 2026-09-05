// Copyright (c) 2026
// Action handling, command ingress, and asynchronous monitor worker.

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

#include <geometry_msgs/msg/pose_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>
#include <cps_trajectory_generators/reachable_cartesian_trajectory.hpp>

#include "math.hpp"
#include "timing.hpp"

namespace
{

constexpr double kMinDt = 1e-6;

using CartesianViaMotionAction =
  panda_motion_generator_msgs::action::CartesianViaMotion;
using SimpleActionResult = panda_motion_generator_msgs::msg::SimpleActionResult;
using CartesianTrajectorySample =
  cps_trajectory_generators::CartesianTrajectorySample;
using LocalCartesianReplanConfig =
  cps_trajectory_generators::LocalCartesianReplanConfig;
using cps_controllers::detail::nanosecondsToMilliseconds;
using cps_controllers::detail::steadyNowNanoseconds;
using cps_trajectory_generators::makeSmoothViaPointCartesianTrajectory;

inline std::shared_ptr<CartesianViaMotionAction::Result>
makeCartesianViaMotionActionResult(int32_t state, const std::string & message)
{
  auto result = std::make_shared<CartesianViaMotionAction::Result>();
  result->result.state = state;
  result->result.error = message;
  return result;
}

}  // namespace

namespace cps_controllers
{

bool ReachableCartesianImpedanceController::takePendingCartesianViaPoints(
  std::vector<Vector3d> * points,
  std::vector<Quaterniond> * orientations,
  std::shared_ptr<CartesianViaMotionGoalHandle> * goal_handle,
  std::uint64_t * sequence,
  bool * calibration_execution,
  double * calibration_capture_path_time_sec)
{
  if (points == nullptr || orientations == nullptr ||
    goal_handle == nullptr || sequence == nullptr ||
    calibration_execution == nullptr ||
    calibration_capture_path_time_sec == nullptr)
  {
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
  *calibration_execution = pending_cartesian_via_points_calibration_;
  *calibration_capture_path_time_sec =
    pending_calibration_capture_path_time_sec_;
  pending_cartesian_via_points_available_ = false;
  pending_cartesian_via_points_calibration_ = false;
  pending_calibration_capture_path_time_sec_ = 0.0;
  pending_cartesian_via_points_goal_handle_.reset();
  return true;
}

std::vector<CartesianTrajectorySample>
ReachableCartesianImpedanceController::buildCartesianViaPointPath(
  const Vector3d & start_position,
  const Quaterniond & start_orientation,
  const std::vector<Vector3d> & via_points,
  const std::vector<Quaterniond> & via_orientations,
  std::size_t * waypoint_count) const
{
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
      std::abs(
        waypoint_orientation.coeffs().dot(
          waypoint_orientations.back().coeffs())) < 1.0 - 1e-9)
    {
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
  via_config.dt = local_replan_dt_;
  via_config.waypoint_merge_position_tolerance =
    waypoint_merge_position_tolerance_;
  via_config.waypoint_merge_orientation_tolerance =
    waypoint_merge_orientation_tolerance_;
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
  const Vector3d & current_position,
  const Quaterniond & current_orientation,
  double wall_time)
{
  std::vector<Vector3d> via_points;
  std::vector<Quaterniond> via_orientations;
  std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle;
  std::uint64_t sequence = 0;
  bool calibration_execution = false;
  double calibration_capture_path_time_sec = 0.0;
  if (!takePendingCartesianViaPoints(
      &via_points, &via_orientations, &goal_handle, &sequence,
      &calibration_execution, &calibration_capture_path_time_sec))
  {
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
      goal_handle->abort(
        makeCartesianViaMotionActionResult(
          SimpleActionResult::REJECTED,
          "No valid Cartesian via-point path could be time-parameterized."));
    }
    return;
  }

  const double path_duration = path.back().t;
  if (calibration_execution &&
    calibration_capture_path_time_sec >= path_duration - kMinDt)
  {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Rejecting calibration action %lu because capture path time %.6f s is not before trajectory duration %.6f s.",
      static_cast<unsigned long>(sequence),
      calibration_capture_path_time_sec,
      path_duration);
    if (goal_handle && goal_handle->is_active()) {
      goal_handle->abort(
        makeCartesianViaMotionActionResult(
          SimpleActionResult::REJECTED,
          "calibration_capture_path_time_sec must be smaller than the generated trajectory duration."));
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
  calibration_plan_latched_ = false;
  calibration_plan_complete_ = false;
  calibration_target_failed_ = false;
  calibration_requested_capture_path_time_sec_ =
    calibration_execution ? calibration_capture_path_time_sec : 0.0;
  calibration_actual_capture_path_time_sec_ = -1.0;
  calibration_failsafe_command_count_ = 0;
  calibration_plan_generation_ = 0;
  calibration_monitor_input_sequence_ = 0;
  calibration_activation_control_sequence_ = 0;
  calibration_activation_intended_index_ = 0;
  calibration_activation_failsafe_index_ = 0;

  std::shared_ptr<CartesianViaMotionGoalHandle> previous_active_goal;
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    previous_active_goal = active_cartesian_via_points_goal_handle_;
    active_cartesian_via_points_goal_handle_ = goal_handle;
    active_cartesian_via_points_calibration_ = calibration_execution;
    cartesian_via_points_action_last_feedback_wall_time_ = -1.0;
  }
  if (previous_active_goal &&
    previous_active_goal != goal_handle &&
    previous_active_goal->is_active())
  {
    previous_active_goal->abort(
      makeCartesianViaMotionActionResult(
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
    "Accepted cartesian_via_points message %lu: calibration=%d capture_path_time=%.6f s poses=%zu waypoints=%zu samples=%zu duration=%.3f s",
    static_cast<unsigned long>(sequence),
    static_cast<int>(calibration_execution),
    calibration_requested_capture_path_time_sec_,
    cartesian_via_points_.size(),
    waypoint_count,
    path_snapshot.size(),
    path_snapshot.empty() ? 0.0 : path_snapshot.back().t);
}

void ReachableCartesianImpedanceController::resetViaPointExecutionState(
  const Vector3d & current_position,
  const Quaterniond & current_orientation,
  double wall_time)
{
  paused_nominal_time_sec_ = wall_time;
  failsafe_enter_wall_time_sec_ = -1.0;
  commanded_path_time_ = 0.0;
  // The nominal Cartesian path is already jerk-limited and starts with zero
  // Cartesian velocity and acceleration. Run its clock at the requested
  // nominal rate immediately; ramping this scalar from zero would apply a
  // second acceleration profile on top of the smooth timed path.
  commanded_path_rate_ = path_time_rate_target_;
  mode_ = SafetyMode::kNominal;

  last_verified_plan_ = VerifiedPlan{};
  ++last_verified_plan_generation_;
  last_verified_command_stage_ = 0;
  last_verified_command_index_ = 0;
  last_shield_decision_valid_ = false;
  last_async_output_valid_ = false;
  last_async_output_wall_time_ = -1.0;
  // Start a fresh monitor phase immediately for the new trajectory without
  // depending on ROS/simulation wall time.
  next_async_monitor_control_sequence_ = control_update_sequence_;
  {
    std::lock_guard<std::mutex> input_lock(async_input_mutex_);
    async_input_pending_ = false;
  }
  async_output_mailbox_.discardReady();

  last_commanded_sample_ = ImpedanceSample{};
  last_commanded_sample_.t = 0.0;
  last_commanded_sample_.p = current_position;
  last_commanded_sample_.dp.setZero();
  last_commanded_sample_.ddp.setZero();
  last_commanded_sample_.q = normalizedQuaternionOrIdentity(current_orientation);
  last_commanded_sample_.w.setZero();
  last_commanded_sample_.dw.setZero();
  last_commanded_sample_.K = K_base_;
  last_commanded_sample_.D = D_base_;
  last_commanded_sample_.failsafe = false;
  last_commanded_sample_valid_ = true;
  bool active_path_available = false;
  {
    std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
    active_path_available = !cartesian_via_point_path_.empty();
  }
  if (active_path_available && !anchorLastCommandedSampleToPathStart()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Strict path-consistent execution could not anchor the new path to "
      "the current command state. Holding until a continuous path is supplied.");
  }
}

void ReachableCartesianImpedanceController::updateCartesianViaPointsActionStatus(
  const Vector3d & current_position,
  const Quaterniond & current_orientation,
  double wall_time)
{
  std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle;
  bool calibration_action = false;
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    goal_handle = active_cartesian_via_points_goal_handle_;
    calibration_action = active_cartesian_via_points_calibration_;
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
    calibration_plan_latched_ = false;
    calibration_plan_complete_ = false;
    calibration_target_failed_ = false;
    calibration_monitor_input_sequence_ = 0;
    goal_handle->canceled(
      makeCartesianViaMotionActionResult(
        SimpleActionResult::PREEMPTED,
        "Cartesian via-point goal was canceled."));
    {
      std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
      if (active_cartesian_via_points_goal_handle_ == goal_handle) {
        active_cartesian_via_points_goal_handle_.reset();
        active_cartesian_via_points_calibration_ = false;
      }
    }
    return;
  }

  if (!goal_handle->is_active()) {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    if (active_cartesian_via_points_goal_handle_ == goal_handle) {
      active_cartesian_via_points_goal_handle_.reset();
      active_cartesian_via_points_calibration_ = false;
    }
    return;
  }

  if (calibration_action && calibration_target_failed_) {
    {
      std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
      cartesian_via_point_path_.clear();
    }
    cartesian_via_points_.clear();
    cartesian_via_point_quaternions_.clear();
    resetViaPointExecutionState(
      current_position, current_orientation, wall_time);
    goal_handle->abort(
      makeCartesianViaMotionActionResult(
        SimpleActionResult::ABORTED,
        "The next monitored trajectory after this calibration action was not safely executable; no later candidate was substituted."));
    {
      std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
      if (active_cartesian_via_points_goal_handle_ == goal_handle) {
        active_cartesian_via_points_goal_handle_.reset();
        active_cartesian_via_points_calibration_ = false;
      }
    }
    return;
  }

  if (calibration_action && calibration_plan_complete_) {
    goal_handle->succeed(
      makeCartesianViaMotionActionResult(
        SimpleActionResult::SUCCESS,
        "Calibration executed the action's next monitored trajectory through its final fail-safe sample."));
    {
      std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
      if (active_cartesian_via_points_goal_handle_ == goal_handle) {
        active_cartesian_via_points_goal_handle_.reset();
        active_cartesian_via_points_calibration_ = false;
      }
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

  if (calibration_action && calibration_plan_latched_) {
    return;
  }

  const bool path_finished =
    commanded_path_time_ >= path_duration - kMinDt &&
    isNominalSafetyMode(mode_);
  if (!path_finished) {
    return;
  }

  if (calibration_action) {
    goal_handle->abort(
      makeCartesianViaMotionActionResult(
        SimpleActionResult::ABORTED,
        "Calibration path ended before its next monitored trajectory could be evaluated."));
    {
      std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
      if (active_cartesian_via_points_goal_handle_ == goal_handle) {
        active_cartesian_via_points_goal_handle_.reset();
        active_cartesian_via_points_calibration_ = false;
      }
    }
    return;
  }

  goal_handle->succeed(
    makeCartesianViaMotionActionResult(
      SimpleActionResult::SUCCESS,
      "Cartesian via-point goal completed."));
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    if (active_cartesian_via_points_goal_handle_ == goal_handle) {
      active_cartesian_via_points_goal_handle_.reset();
      active_cartesian_via_points_calibration_ = false;
    }
  }
}

rclcpp_action::GoalResponse
ReachableCartesianImpedanceController::handleCartesianViaPointsActionGoal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const CartesianViaMotion::Goal> goal)
{
  (void)uuid;
  if (!goal || goal->via_poses.empty()) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Rejecting Cartesian via-point action goal because it contains no poses.");
    return rclcpp_action::GoalResponse::REJECT;
  }

  for (std::size_t i = 0; i < goal->via_poses.size(); ++i) {
    const auto & pose = goal->via_poses[i];
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
      !std::isfinite(position.z()))
    {
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
  const std::shared_ptr<CartesianViaMotionGoalHandle>/*goal_handle*/)
{
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ReachableCartesianImpedanceController::handleCartesianViaPointsActionAccepted(
  const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle)
{
  queueCartesianViaPointsActionGoal(goal_handle, false);
}

rclcpp_action::GoalResponse
ReachableCartesianImpedanceController::handleCalibrationActionGoal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const CartesianViaMotion::Goal> goal)
{
  if (!enable_calibration_logging_) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Rejecting calibration action because enable_calibration_logging is disabled.");
    return rclcpp_action::GoalResponse::REJECT;
  }
  const double capture_path_time_sec =
    get_node()
    ->get_parameter("calibration_capture_path_time_sec")
    .as_double();
  if (!std::isfinite(capture_path_time_sec) || capture_path_time_sec < 0.0) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Rejecting calibration action because calibration_capture_path_time_sec must be finite and nonnegative (got %.9f).",
      capture_path_time_sec);
    return rclcpp_action::GoalResponse::REJECT;
  }
  return handleCartesianViaPointsActionGoal(uuid, std::move(goal));
}

rclcpp_action::CancelResponse
ReachableCartesianImpedanceController::handleCalibrationActionCancel(
  const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle)
{
  return handleCartesianViaPointsActionCancel(goal_handle);
}

void ReachableCartesianImpedanceController::handleCalibrationActionAccepted(
  const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle)
{
  queueCartesianViaPointsActionGoal(goal_handle, true);
}

void ReachableCartesianImpedanceController::queueCartesianViaPointsActionGoal(
  const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle,
  bool calibration_execution)
{
  if (!goal_handle) {
    return;
  }

  const auto goal = goal_handle->get_goal();
  std::vector<Vector3d> points;
  std::vector<Quaterniond> orientations;
  points.reserve(goal->via_poses.size());
  orientations.reserve(goal->via_poses.size());

  for (const auto & pose : goal->via_poses) {
    points.emplace_back(pose.position.x, pose.position.y, pose.position.z);
    orientations.push_back(
      normalizedQuaternionOrIdentity(
        Quaterniond(
          pose.orientation.w,
          pose.orientation.x,
          pose.orientation.y,
          pose.orientation.z)));
  }

  std::shared_ptr<CartesianViaMotionGoalHandle> previous_pending_goal;
  std::uint64_t sequence = 0;
  double calibration_capture_path_time_sec = 0.0;
  if (calibration_execution) {
    calibration_capture_path_time_sec =
      get_node()
      ->get_parameter("calibration_capture_path_time_sec")
      .as_double();
    if (!std::isfinite(calibration_capture_path_time_sec) ||
      calibration_capture_path_time_sec < 0.0)
    {
      goal_handle->abort(
        makeCartesianViaMotionActionResult(
          SimpleActionResult::REJECTED,
          "calibration_capture_path_time_sec must be finite and nonnegative."));
      return;
    }
  }
  {
    std::lock_guard<std::mutex> lock(pending_cartesian_via_points_mutex_);
    previous_pending_goal = pending_cartesian_via_points_goal_handle_;
    pending_cartesian_via_points_ = std::move(points);
    pending_cartesian_via_point_quaternions_ = std::move(orientations);
    pending_cartesian_via_points_goal_handle_ = goal_handle;
    pending_cartesian_via_points_calibration_ = calibration_execution;
    pending_calibration_capture_path_time_sec_ =
      calibration_capture_path_time_sec;
    sequence = ++pending_cartesian_via_points_sequence_;
    pending_cartesian_via_points_available_ = true;
  }

  std::shared_ptr<CartesianViaMotionGoalHandle> previous_active_goal;
  {
    std::lock_guard<std::mutex> action_lock(cartesian_via_points_action_mutex_);
    previous_active_goal = active_cartesian_via_points_goal_handle_;
    if (previous_active_goal && previous_active_goal != goal_handle) {
      active_cartesian_via_points_goal_handle_.reset();
      active_cartesian_via_points_calibration_ = false;
    }
  }

  if (previous_pending_goal &&
    previous_pending_goal != goal_handle &&
    previous_pending_goal->is_active())
  {
    previous_pending_goal->abort(
      makeCartesianViaMotionActionResult(
        SimpleActionResult::ABORTED,
        "Cartesian via-point goal was replaced by a newer action goal."));
  }
  if (previous_active_goal &&
    previous_active_goal != goal_handle &&
    previous_active_goal->is_active())
  {
    previous_active_goal->abort(
      makeCartesianViaMotionActionResult(
        SimpleActionResult::ABORTED,
        "Cartesian via-point goal was replaced by a newer action goal."));
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Queued Cartesian via-point action goal %lu with %zu poses (calibration=%d capture_path_time=%.6f s).",
    static_cast<unsigned long>(sequence),
    goal->via_poses.size(),
    static_cast<int>(calibration_execution),
    calibration_capture_path_time_sec);
}

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

  VerifiedPlan last_verified_plan;

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

    const std::int64_t worker_start_ns = steadyNowNanoseconds();
    AsyncMonitorOutput output;
    output.sequence = input.sequence;
    output.input_wall_time = input.wall_time;
    output.valid = true;
    output.worker_start_steady_time_ns = worker_start_ns;
    output.worker_queue_wait_ms = nanosecondsToMilliseconds(
      std::max<std::int64_t>(
        0, worker_start_ns - input.publish_steady_time_ns));
    output.decision =
      computeShieldDecisionForAsyncInput(input, last_verified_plan);

    const bool reachable_set_output_due =
      enable_reachable_set_visualization_ &&
      reachable_set_visualization_pub_ &&
      human_reachable_set_pub_ &&
      (last_reachable_set_visualization_wall_time_ < 0.0 ||
      input.wall_time < last_reachable_set_visualization_wall_time_ ||
      input.wall_time - last_reachable_set_visualization_wall_time_ >=
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
      reachable_set_snapshot.joint_prediction_trace =
        output.decision.joint_prediction_trace;
    }

    output.input = std::move(input);
    output.worker_finish_steady_time_ns = steadyNowNanoseconds();
    output.worker_compute_ms = nanosecondsToMilliseconds(
      std::max<std::int64_t>(
        0,
        output.worker_finish_steady_time_ns -
        output.worker_start_steady_time_ns));
    async_monitor_worker_processed_count_.fetch_add(
      1, std::memory_order_relaxed);

    const auto publish_result =
      async_output_mailbox_.publish(std::move(output));
    const std::size_t discarded_outputs =
      publish_result.overwritten_ready +
      static_cast<std::size_t>(!publish_result.published);
    if (discarded_outputs > 0) {
      async_monitor_output_overwrite_count_.fetch_add(
        discarded_outputs, std::memory_order_relaxed);
    }
    // The safety result is handed off before reachable-set output generation.
    // This keeps visualization-related latency out of the acceptance path.
    if (reachable_set_output_due) {
      publishReachableSetOutputs(reachable_set_snapshot);
    }
  }
}

void ReachableCartesianImpedanceController::startSafetyMonitorWorker()
{
  if (!async_safety_monitor_ || safety_monitor_worker_running_.load()) {
    return;
  }

  async_output_mailbox_.resetStopped();
  safety_monitor_worker_running_.store(true);
  async_input_pending_ = false;
  last_async_output_wall_time_ = -1.0;
  last_async_output_valid_ = false;

  safety_monitor_worker_thread_ =
    std::thread(&ReachableCartesianImpedanceController::safetyMonitorWorkerLoop, this);
}

void ReachableCartesianImpedanceController::stopSafetyMonitorWorker()
{
  if (!safety_monitor_worker_running_.exchange(false)) {
    return;
  }

  async_input_cv_.notify_all();

  if (safety_monitor_worker_thread_.joinable()) {
    safety_monitor_worker_thread_.join();
  }
}

void ReachableCartesianImpedanceController::handleCartesianViaPoints(
  const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
  if (!msg) {
    return;
  }

  const std::string & frame_id = msg->header.frame_id;
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
    const auto & pose = msg->poses[i];
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
      !std::isfinite(position.z()))
    {
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
  const mujoco_ros_msgs::msg::ScalarStamped::SharedPtr msg)
{
  const double value = msg->value;
  latest_mujoco_contact_value_.store(value, std::memory_order_relaxed);
  latest_mujoco_contact_msg_time_.store(
    rclcpp::Time(msg->header.stamp).seconds(),
    std::memory_order_relaxed);

  const bool active = value > mujoco_contact_threshold_;
  latest_mujoco_contact_active_.store(active, std::memory_order_relaxed);
}

}  // namespace cps_controllers
