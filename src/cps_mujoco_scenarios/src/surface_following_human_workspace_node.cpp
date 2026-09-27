// Copyright (c) 2026 Yue
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Geometry>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <mujoco_ros_msgs/srv/get_body_state.hpp>
#include <rclcpp/rclcpp.hpp>

#include "cps_human_workspace/human_workspace.hpp"
#include "cps_human_workspace/human_workspace_message.hpp"
#include "cps_human_workspace/msg/human_workspace.hpp"

namespace cps_mujoco_scenarios
{

namespace
{

using HumanWorkspace = cps_human_workspace::HumanWorkspace;
using HumanWorkspaceMsg = cps_human_workspace::msg::HumanWorkspace;
using GetBodyState = mujoco_ros_msgs::srv::GetBodyState;
using Vector3d = Eigen::Vector3d;

bool finiteVector(const Vector3d & value)
{
  return value.array().isFinite().all();
}

bool finiteQuaternion(const Eigen::Quaterniond & value)
{
  return value.coeffs().array().isFinite().all() && value.norm() > 1.0e-12;
}

}  // namespace

class SurfaceFollowingHumanWorkspace : public rclcpp::Node
{
public:
  SurfaceFollowingHumanWorkspace()
  : Node("surface_following_human_workspace")
  {
    const std::string default_config_path =
      ament_index_cpp::get_package_share_directory("cps_mujoco_scenarios") +
      "/config/human_workspace_surface_following.yaml";
    const std::string config_path = declare_parameter<std::string>(
      "human_workspace_config_path", default_config_path);
    surface_body_name_ = declare_parameter<std::string>(
      "surface_body_name", "human_hand_surface");
    body_state_service_name_ = declare_parameter<std::string>(
      "body_state_service", "/get_body_state");
    admin_hash_ = declare_parameter<std::string>("admin_hash", "");
    // GetBodyState always reports MuJoCo world coordinates. Query the robot
    // base body once and express every surface sample in that body's frame so
    // the numeric values and the ROS frame_id describe the same coordinates.
    frame_id_ = declare_parameter<std::string>("frame_id", "panda_link0");
    const std::string state_topic_name = declare_parameter<std::string>(
      "state_topic", "human_workspace/state");
    const double publish_rate_hz =
      declare_parameter<double>("publish_rate", 50.0);

    if (surface_body_name_.empty() || body_state_service_name_.empty() ||
      frame_id_.empty() ||
      state_topic_name.empty())
    {
      throw std::invalid_argument(
              "surface body, service, and output topic names must not be empty");
    }
    if (!std::isfinite(publish_rate_hz) || publish_rate_hz <= 0.0) {
      throw std::invalid_argument("publish_rate must be finite and positive");
    }
    if (!workspace_template_.configureReachabilityFromConfigFile(
        config_path, get_logger()))
    {
      throw std::runtime_error(
              "could not configure surface-following human workspace");
    }

    body_state_client_ = create_client<GetBodyState>(body_state_service_name_);
    state_publisher_ = create_publisher<HumanWorkspaceMsg>(
      state_topic_name, rclcpp::QoS(1).transient_local());

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&SurfaceFollowingHumanWorkspace::requestSurfaceState, this));

    RCLCPP_INFO(
      get_logger(),
      "Following MuJoCo surface body '%s' through '%s' at %.1f Hz. "
      "Coordinates will be transformed from MuJoCo world into body frame "
      "'%s' and published as timestamped SaRA hand observations on '%s'.",
      surface_body_name_.c_str(), body_state_service_name_.c_str(),
      publish_rate_hz, frame_id_.c_str(), state_topic_name.c_str());
  }

private:
  void requestSurfaceState()
  {
    if (request_in_flight_) {
      return;
    }
    if (!body_state_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Waiting for MuJoCo body-state service '%s'.",
        body_state_service_name_.c_str());
      return;
    }

    auto request = std::make_shared<GetBodyState::Request>();
    request->name = have_world_to_output_transform_ ? surface_body_name_ : frame_id_;
    request->admin_hash = admin_hash_;
    request_in_flight_ = true;
    if (have_world_to_output_transform_) {
      body_state_client_->async_send_request(
        request,
        [this](rclcpp::Client<GetBodyState>::SharedFuture future) {
          handleSurfaceState(std::move(future));
        });
    } else {
      body_state_client_->async_send_request(
        request,
        [this](rclcpp::Client<GetBodyState>::SharedFuture future) {
          handleOutputFrameState(std::move(future));
        });
    }
  }

  void handleOutputFrameState(rclcpp::Client<GetBodyState>::SharedFuture future)
  {
    request_in_flight_ = false;

    GetBodyState::Response::SharedPtr response;
    try {
      response = future.get();
    } catch (const std::exception & exception) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "MuJoCo output-frame request failed: %s", exception.what());
      return;
    }
    if (!response || !response->success) {
      const std::string status = response ? response->status_message : "no response";
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Could not read MuJoCo output-frame body '%s': %s",
        frame_id_.c_str(), status.c_str());
      return;
    }

    const auto & pose = response->state.pose.pose;
    const Vector3d origin_world(
      pose.position.x, pose.position.y, pose.position.z);
    Eigen::Quaterniond output_to_world(
      pose.orientation.w, pose.orientation.x,
      pose.orientation.y, pose.orientation.z);
    if (!finiteVector(origin_world) || !finiteQuaternion(output_to_world)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Ignoring invalid pose for MuJoCo output-frame body '%s'.",
        frame_id_.c_str());
      return;
    }

    output_to_world.normalize();
    output_origin_world_ = origin_world;
    world_to_output_rotation_ = output_to_world.conjugate();
    have_world_to_output_transform_ = true;
    RCLCPP_INFO(
      get_logger(),
      "Resolved MuJoCo world -> %s: world origin [%.6f, %.6f, %.6f], "
      "body quaternion wxyz [%.6f, %.6f, %.6f, %.6f].",
      frame_id_.c_str(), origin_world.x(), origin_world.y(), origin_world.z(),
      output_to_world.w(), output_to_world.x(), output_to_world.y(),
      output_to_world.z());
  }

  void handleSurfaceState(rclcpp::Client<GetBodyState>::SharedFuture future)
  {
    request_in_flight_ = false;

    GetBodyState::Response::SharedPtr response;
    try {
      response = future.get();
    } catch (const std::exception & exception) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "MuJoCo surface-state request failed: %s", exception.what());
      return;
    }
    if (!response || !response->success) {
      const std::string status = response ? response->status_message : "no response";
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Could not read MuJoCo body '%s': %s",
        surface_body_name_.c_str(), status.c_str());
      return;
    }

    const auto & position = response->state.pose.pose.position;
    const Vector3d center_world(position.x, position.y, position.z);
    if (!finiteVector(center_world)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Ignoring non-finite position for MuJoCo body '%s'.",
        surface_body_name_.c_str());
      return;
    }

    const Vector3d center =
      world_to_output_rotation_ * (center_world - output_origin_world_);

    // Differentiate the transformed actual position instead of trusting body
    // twist. This includes both table_assembly mocap motion and the compliant
    // surface's local spring displacement, expressed in the robot-base frame.
    const auto sample_time = std::chrono::steady_clock::now();
    Vector3d velocity = Vector3d::Zero();
    Vector3d acceleration = Vector3d::Zero();
    if (have_previous_position_) {
      const double dt = std::chrono::duration<double>(
        sample_time - previous_sample_time_).count();
      if (std::isfinite(dt) && dt > 1.0e-6) {
        velocity = (center - previous_position_) / dt;
        acceleration = (velocity - previous_velocity_) / dt;
      }
    }
    if (!finiteVector(velocity) || !finiteVector(acceleration)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Ignoring invalid differentiated motion for MuJoCo body '%s'.",
        surface_body_name_.c_str());
      return;
    }

    previous_position_ = center;
    previous_velocity_ = velocity;
    previous_sample_time_ = sample_time;
    have_previous_position_ = true;

    const auto & parameters = workspace_template_.parameters();
    if (velocity.norm() > parameters.hand_max_velocity + 1.0e-6) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Measured surface speed %.3f m/s exceeds SaRA max_velocity %.3f m/s.",
        velocity.norm(), parameters.hand_max_velocity);
    }
    if (acceleration.norm() > parameters.hand_max_acceleration + 1.0e-6) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Measured surface acceleration %.3f m/s^2 exceeds SaRA "
        "max_acceleration %.3f m/s^2.",
        acceleration.norm(), parameters.hand_max_acceleration);
    }

    publishObservation(center, velocity);
  }

  void publishObservation(
    const Vector3d & center,
    const Vector3d & velocity)
  {
    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = frame_id_;
    state_publisher_->publish(
      cps_human_workspace::makeHumanWorkspaceMessage(
        header, center, velocity, workspace_template_.parameters()));
  }

  HumanWorkspace workspace_template_;
  std::string surface_body_name_;
  std::string body_state_service_name_;
  std::string admin_hash_;
  std::string frame_id_;

  rclcpp::Client<GetBodyState>::SharedPtr body_state_client_;
  rclcpp::Publisher<HumanWorkspaceMsg>::SharedPtr state_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool request_in_flight_{false};

  bool have_world_to_output_transform_{false};
  Vector3d output_origin_world_{Vector3d::Zero()};
  Eigen::Quaterniond world_to_output_rotation_{Eigen::Quaterniond::Identity()};

  bool have_previous_position_{false};
  Vector3d previous_position_{Vector3d::Zero()};
  Vector3d previous_velocity_{Vector3d::Zero()};
  std::chrono::steady_clock::time_point previous_sample_time_;
};

}  // namespace cps_mujoco_scenarios

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<
      cps_mujoco_scenarios::SurfaceFollowingHumanWorkspace>());
  rclcpp::shutdown();
  return 0;
}
