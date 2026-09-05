// Copyright (c) 2026 Yue
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mujoco_ros_msgs/srv/set_mocap_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "cps_mujoco_scenarios/action/move_table_assembly.hpp"

namespace cps_mujoco_scenarios
{

class TableAssemblyActionServer : public rclcpp::Node
{
public:
  using MoveTableAssembly =
    cps_mujoco_scenarios::action::MoveTableAssembly;
  using GoalHandleMoveTableAssembly =
    rclcpp_action::ServerGoalHandle<MoveTableAssembly>;
  using SetMocapState = mujoco_ros_msgs::srv::SetMocapState;

  TableAssemblyActionServer()
  : Node("table_assembly_action_server")
  {
    const std::string action_name = declare_parameter<std::string>(
      "action_name", "move_table_assembly");
    const std::string mocap_service_name = declare_parameter<std::string>(
      "mocap_service", "/set_mocap_state");
    body_name_ = declare_parameter<std::string>(
      "body_name", "table_assembly");
    const auto assembly_origin = declare_parameter<std::vector<double>>(
      "assembly_origin", std::vector<double>{0.0, 0.0, 0.0});
    if (assembly_origin.size() != 3 ||
      !std::all_of(
        assembly_origin.begin(), assembly_origin.end(),
        [](double value) {return std::isfinite(value);}))
    {
      throw std::invalid_argument(
              "assembly_origin must contain three finite values");
    }
    origin_x_ = assembly_origin[0];
    origin_y_ = assembly_origin[1];
    origin_z_ = assembly_origin[2];

    minimum_z_offset_ = declare_parameter<double>(
      "minimum_z_offset", -0.5);
    maximum_z_offset_ = declare_parameter<double>(
      "maximum_z_offset", 0.5);
    const double update_rate_hz = declare_parameter<double>(
      "update_rate_hz", 50.0);
    service_wait_timeout_sec_ = declare_parameter<double>(
      "service_wait_timeout_sec", 5.0);

    if (!std::isfinite(minimum_z_offset_) ||
      !std::isfinite(maximum_z_offset_) ||
      minimum_z_offset_ > maximum_z_offset_)
    {
      throw std::invalid_argument(
              "minimum_z_offset must be finite and no greater than maximum_z_offset");
    }
    if (!std::isfinite(update_rate_hz) || update_rate_hz <= 0.0) {
      throw std::invalid_argument("update_rate_hz must be positive");
    }
    if (!std::isfinite(service_wait_timeout_sec_) ||
      service_wait_timeout_sec_ <= 0.0)
    {
      throw std::invalid_argument(
              "service_wait_timeout_sec must be positive");
    }

    mocap_client_ = create_client<SetMocapState>(mocap_service_name);
    action_server_ = rclcpp_action::create_server<MoveTableAssembly>(
      this,
      action_name,
      std::bind(
        &TableAssemblyActionServer::handleGoal, this,
        std::placeholders::_1, std::placeholders::_2),
      std::bind(
        &TableAssemblyActionServer::handleCancel, this,
        std::placeholders::_1),
      std::bind(
        &TableAssemblyActionServer::handleAccepted, this,
        std::placeholders::_1));

    const auto update_period = std::chrono::duration<double>(
      1.0 / update_rate_hz);
    update_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(update_period),
      std::bind(&TableAssemblyActionServer::updateMotion, this));

    RCLCPP_INFO(
      get_logger(),
      "Table assembly action '%s' controls mocap body '%s' through '%s'; "
      "allowed z offset [%.3f, %.3f] m.",
      action_name.c_str(), body_name_.c_str(),
      mocap_service_name.c_str(), minimum_z_offset_, maximum_z_offset_);
  }

private:
  static double smoothStep(double value)
  {
    const double u = std::clamp(value, 0.0, 1.0);
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
  }

  rclcpp_action::GoalResponse handleGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const MoveTableAssembly::Goal> goal)
  {
    if (!goal || !std::isfinite(goal->target_z_offset) ||
      !std::isfinite(goal->duration) || goal->duration <= 0.0)
    {
      RCLCPP_WARN(
        get_logger(),
        "Rejecting table motion: target and duration must be finite and "
        "duration must be positive.");
      return rclcpp_action::GoalResponse::REJECT;
    }
    if (goal->target_z_offset < minimum_z_offset_ ||
      goal->target_z_offset > maximum_z_offset_)
    {
      RCLCPP_WARN(
        get_logger(),
        "Rejecting table target %.3f m outside [%.3f, %.3f] m.",
        goal->target_z_offset, minimum_z_offset_, maximum_z_offset_);
      return rclcpp_action::GoalResponse::REJECT;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (active_goal_) {
      RCLCPP_WARN(get_logger(), "Rejecting table motion while another goal is active.");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(
    const std::shared_ptr<GoalHandleMoveTableAssembly> goal_handle)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (goal_handle && active_goal_ == goal_handle) {
      return rclcpp_action::CancelResponse::ACCEPT;
    }
    return rclcpp_action::CancelResponse::REJECT;
  }

  void handleAccepted(
    const std::shared_ptr<GoalHandleMoveTableAssembly> goal_handle)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!goal_handle || active_goal_) {
      if (goal_handle) {
        auto result = std::make_shared<MoveTableAssembly::Result>();
        result->success = false;
        result->message = "another table motion is already active";
        result->final_z_offset = current_z_offset_;
        goal_handle->abort(result);
      }
      return;
    }

    active_goal_ = goal_handle;
    start_z_offset_ = current_z_offset_;
    target_z_offset_ = goal_handle->get_goal()->target_z_offset;
    motion_duration_sec_ = goal_handle->get_goal()->duration;
    motion_start_time_ = std::chrono::steady_clock::now();
    service_deadline_ = motion_start_time_ + std::chrono::duration_cast<
      std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(service_wait_timeout_sec_));
    ++goal_generation_;

    RCLCPP_INFO(
      get_logger(),
      "Moving table assembly from %.3f m to %.3f m in %.3f s.",
      start_z_offset_, target_z_offset_, motion_duration_sec_);
  }

  void updateMotion()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_goal_ || service_call_in_flight_) {
      return;
    }

    if (active_goal_->is_canceling()) {
      auto result = std::make_shared<MoveTableAssembly::Result>();
      result->success = false;
      result->message = "table motion canceled; current position is held";
      result->final_z_offset = current_z_offset_;
      active_goal_->canceled(result);
      RCLCPP_INFO(
        get_logger(), "Table motion canceled at z offset %.3f m.",
        current_z_offset_);
      active_goal_.reset();
      ++goal_generation_;
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!mocap_client_->service_is_ready()) {
      if (now >= service_deadline_) {
        abortActiveGoalLocked(
          "MuJoCo mocap service was not available before the timeout");
      }
      return;
    }

    const double elapsed_sec =
      std::chrono::duration<double>(now - motion_start_time_).count();
    const double progress = std::clamp(
      elapsed_sec / motion_duration_sec_, 0.0, 1.0);
    const double z_offset = start_z_offset_ +
      (target_z_offset_ - start_z_offset_) * smoothStep(progress);
    const bool final_sample = progress >= 1.0;

    auto request = std::make_shared<SetMocapState::Request>();
    request->mocap_state.name.push_back(body_name_);
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = get_clock()->now();
    pose.header.frame_id = "world";
    pose.pose.position.x = origin_x_;
    pose.pose.position.y = origin_y_;
    pose.pose.position.z = origin_z_ + z_offset;
    pose.pose.orientation.w = 1.0;
    request->mocap_state.pose.push_back(std::move(pose));

    const std::uint64_t generation = goal_generation_;
    service_call_in_flight_ = true;
    mocap_client_->async_send_request(
      request,
      [this, generation, z_offset, progress, final_sample](
        rclcpp::Client<SetMocapState>::SharedFuture future)
      {
        handleMocapResponse(
          generation, z_offset, progress, final_sample, std::move(future));
      });
  }

  void handleMocapResponse(
    std::uint64_t generation,
    double z_offset,
    double progress,
    bool final_sample,
    rclcpp::Client<SetMocapState>::SharedFuture future)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != goal_generation_) {
      return;
    }
    service_call_in_flight_ = false;
    if (!active_goal_) {
      return;
    }

    try {
      const auto response = future.get();
      if (!response || !response->success) {
        abortActiveGoalLocked("MuJoCo rejected the table assembly pose");
        return;
      }
    } catch (const std::exception & exception) {
      abortActiveGoalLocked(
        std::string("MuJoCo mocap service failed: ") + exception.what());
      return;
    }

    current_z_offset_ = z_offset;
    auto feedback = std::make_shared<MoveTableAssembly::Feedback>();
    feedback->progress = progress;
    feedback->current_z_offset = current_z_offset_;
    active_goal_->publish_feedback(feedback);

    if (!final_sample) {
      return;
    }

    auto result = std::make_shared<MoveTableAssembly::Result>();
    result->success = true;
    result->message = "table assembly reached its target and is holding position";
    result->final_z_offset = current_z_offset_;
    active_goal_->succeed(result);
    RCLCPP_INFO(
      get_logger(), "Table assembly reached z offset %.3f m.",
      current_z_offset_);
    active_goal_.reset();
    ++goal_generation_;
  }

  void abortActiveGoalLocked(const std::string & message)
  {
    if (!active_goal_) {
      return;
    }
    auto result = std::make_shared<MoveTableAssembly::Result>();
    result->success = false;
    result->message = message;
    result->final_z_offset = current_z_offset_;
    active_goal_->abort(result);
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    active_goal_.reset();
    service_call_in_flight_ = false;
    ++goal_generation_;
  }

  std::string body_name_;
  double origin_x_{0.0};
  double origin_y_{0.0};
  double origin_z_{0.0};
  double minimum_z_offset_{-0.5};
  double maximum_z_offset_{0.5};
  double service_wait_timeout_sec_{5.0};

  rclcpp_action::Server<MoveTableAssembly>::SharedPtr action_server_;
  rclcpp::Client<SetMocapState>::SharedPtr mocap_client_;
  rclcpp::TimerBase::SharedPtr update_timer_;

  std::mutex mutex_;
  std::shared_ptr<GoalHandleMoveTableAssembly> active_goal_;
  bool service_call_in_flight_{false};
  std::uint64_t goal_generation_{0};
  double current_z_offset_{0.0};
  double start_z_offset_{0.0};
  double target_z_offset_{0.0};
  double motion_duration_sec_{1.0};
  std::chrono::steady_clock::time_point motion_start_time_;
  std::chrono::steady_clock::time_point service_deadline_;
};

}  // namespace cps_mujoco_scenarios

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<cps_mujoco_scenarios::TableAssemblyActionServer>());
  rclcpp::shutdown();
  return 0;
}
