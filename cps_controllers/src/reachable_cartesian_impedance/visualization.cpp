// Copyright (c) 2026
// ROS 2 visualization for the SaRA robot reachable-set capsules.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>

namespace
{

constexpr double kMinimumCapsuleLength = 1.0e-9;
constexpr char kReachableSetNamespace[] = "sara_robot_reachable_set";

enum class ReachableSetContactState
{
  kClear,
  kContactEnergySafe,
  kContactEnergyUnsafe,
};

std_msgs::msg::ColorRGBA saraReachableSetColor(
  ReachableSetContactState state, double alpha)
{
  std_msgs::msg::ColorRGBA color;
  switch (state) {
    case ReachableSetContactState::kClear:
      color.r = 0.45F;
      color.g = 1.0F;
      color.b = 0.45F;
      break;
    case ReachableSetContactState::kContactEnergySafe:
      color.r = 1.0F;
      color.g = 0.65F;
      color.b = 0.0F;
      break;
    case ReachableSetContactState::kContactEnergyUnsafe:
      color.r = 1.0F;
      color.g = 0.0F;
      color.b = 0.0F;
      break;
  }
  color.a = static_cast<float>(alpha);
  return color;
}

}  // namespace

namespace cps_controllers
{

void ReachableCartesianImpedanceController::publishRobotReachableSetVisualization(
  const ReachableSetVisualizationSnapshot & snapshot)
{
  if (!enable_reachable_set_visualization_ ||
    !reachable_set_visualization_pub_ ||
    !robot_reachability_provider_)
  {
    return;
  }

  if (last_reachable_set_visualization_wall_time_ >= 0.0 &&
    snapshot.wall_time >= last_reachable_set_visualization_wall_time_ &&
    snapshot.wall_time - last_reachable_set_visualization_wall_time_ <
    reachable_set_visualization_period_sec_)
  {
    return;
  }
  last_reachable_set_visualization_wall_time_ = snapshot.wall_time;

  visualization_msgs::msg::MarkerArray marker_array;
  const rclcpp::Time stamp = get_node()->now();
  const auto marker_lifetime =
    rclcpp::Duration::from_seconds(1.0);
  int marker_id = 0;
  const auto & trace = snapshot.joint_prediction_trace;
  const std::vector<double> zero_alpha(7, 0.0);
  std::vector<double> dynamic_alpha;
  const bool dynamic_alpha_valid = trace.size() >= 2 &&
    robot_reachability_provider_->calculateTrajectoryAlpha(
    trace, &dynamic_alpha);

  auto make_marker = [&](int type, const std_msgs::msg::ColorRGBA & color) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = reachable_set_visualization_frame_id_;
      marker.header.stamp = stamp;
      marker.ns = kReachableSetNamespace;
      marker.id = marker_id++;
      marker.type = type;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.color = color;
      marker.lifetime = marker_lifetime;
      return marker;
    };

  auto append_sphere = [&](const Vector3d & center,
      double radius,
      const std_msgs::msg::ColorRGBA & color) {
      auto marker = make_marker(
        visualization_msgs::msg::Marker::SPHERE, color);
      marker.pose.position.x = center.x();
      marker.pose.position.y = center.y();
      marker.pose.position.z = center.z();
      const double diameter = 2.0 * radius;
      marker.scale.x = diameter;
      marker.scale.y = diameter;
      marker.scale.z = diameter;
      marker_array.markers.push_back(std::move(marker));
    };

  auto append_capsule = [&](const cps_safety_monitor::RobotReachCapsule & capsule,
      const std_msgs::msg::ColorRGBA & color) {
      const Vector3d axis = capsule.p2 - capsule.p1;
      const double length = axis.norm();
      append_sphere(capsule.p1, capsule.radius, color);
      if (length <= kMinimumCapsuleLength) {
        return;
      }
      append_sphere(capsule.p2, capsule.radius, color);

      auto cylinder = make_marker(
        visualization_msgs::msg::Marker::CYLINDER, color);
      const Vector3d midpoint = 0.5 * (capsule.p1 + capsule.p2);
      cylinder.pose.position.x = midpoint.x();
      cylinder.pose.position.y = midpoint.y();
      cylinder.pose.position.z = midpoint.z();
      const Eigen::Quaterniond orientation =
        Eigen::Quaterniond::FromTwoVectors(
        Vector3d::UnitZ(), axis / length).normalized();
      cylinder.pose.orientation.x = orientation.x();
      cylinder.pose.orientation.y = orientation.y();
      cylinder.pose.orientation.z = orientation.z();
      cylinder.pose.orientation.w = orientation.w();
      cylinder.scale.x = 2.0 * capsule.radius;
      cylinder.scale.y = 2.0 * capsule.radius;
      cylinder.scale.z = length;
      marker_array.markers.push_back(std::move(cylinder));
    };

  auto interval_capsules = [&](const Vector7d & start_q,
      const Vector7d & goal_q,
      double start_time,
      double goal_time,
      const std::vector<double> & alpha_i,
      std::vector<cps_safety_monitor::RobotReachCapsule> * capsules) {
      return capsules != nullptr &&
             robot_reachability_provider_->reachInterval(
          start_q,
          goal_q,
          std::max(0.0, goal_time - start_time),
          alpha_i,
          capsules);
    };

  auto capsule_intersections = [&](
      const std::vector<cps_safety_monitor::RobotReachCapsule> & capsules,
      double start_time,
      double goal_time,
      std::vector<bool> * intersections) {
      if (intersections == nullptr) {
        return false;
      }
      intersections->assign(capsules.size(), false);
      if (snapshot.human_workspace_assumed_clear ||
        !snapshot.human_workspace_active)
      {
        return true;
      }

      const Vector3d human_center_start =
        snapshot.human_workspace.centerAtTime(
        snapshot.wall_time + start_time);
      const Vector3d human_center_end =
        snapshot.human_workspace.centerAtTime(
        snapshot.wall_time + goal_time);
      const double human_radius = snapshot.human_workspace.motionRadius();
      for (std::size_t index = 0; index < capsules.size(); ++index) {
        const std::vector<cps_safety_monitor::RobotReachCapsule>
        single_capsule{capsules[index]};
        const double distance =
          robot_reachability_provider_->minimumSignedDistance(
          single_capsule,
          human_center_start,
          human_center_end,
          human_radius);
        if (!std::isfinite(distance)) {
          return false;
        }
        (*intersections)[index] = distance <= 0.0;
      }
      return true;
    };

  auto append_interval = [this, &append_capsule](
      const std::vector<cps_safety_monitor::RobotReachCapsule> & capsules,
      const std::vector<bool> & intersections,
      bool contact_energy_unsafe) {
      if (intersections.size() != capsules.size()) {
        return;
      }

      for (std::size_t index = 0; index < capsules.size(); ++index) {
        ReachableSetContactState state = ReachableSetContactState::kClear;
        if (intersections[index]) {
          state = contact_energy_unsafe
            ? ReachableSetContactState::kContactEnergyUnsafe
            : ReachableSetContactState::kContactEnergySafe;
        }
        append_capsule(
          capsules[index],
          saraReachableSetColor(state, reachable_set_visualization_alpha_));
      }
    };

  if (trace.size() < 2) {
    if (snapshot.current_q.allFinite()) {
      // SaRA continues to expose robot occupancy while the robot is holding.
      // A zero-duration [q, q] interval is the stationary reachable set.
      std::vector<cps_safety_monitor::RobotReachCapsule> capsules;
      std::vector<bool> intersections;
      if (interval_capsules(
        snapshot.current_q,
        snapshot.current_q,
        0.0,
        0.0,
        zero_alpha,
        &capsules) &&
        capsule_intersections(capsules, 0.0, 0.0, &intersections))
      {
        append_interval(
          capsules,
          intersections,
          snapshot.current_contact_energy_unsafe);
      }
    }
  } else if (dynamic_alpha_valid) {
    const std::size_t interval_count = trace.size() - 1;
    std::size_t selected_interval = interval_count - 1;
    bool selected_contact_energy_unsafe = false;
    if (snapshot.first_energy_unsafe_contact_interval_index >= 0 &&
      static_cast<std::size_t>(
        snapshot.first_energy_unsafe_contact_interval_index) <
      interval_count)
    {
      selected_interval =
        static_cast<std::size_t>(
          snapshot.first_energy_unsafe_contact_interval_index);
      selected_contact_energy_unsafe = true;
    } else if (snapshot.first_contact_interval_index >= 0 &&
      static_cast<std::size_t>(snapshot.first_contact_interval_index) <
      interval_count)
    {
      // A PFL-safe candidate may intentionally intersect the human workspace.
      // Select its first contact interval so that the intersecting capsule is
      // visible in orange instead of hiding the contact behind the terminal
      // (usually collision-free) interval.
      selected_interval =
        static_cast<std::size_t>(snapshot.first_contact_interval_index);
    }

    const auto & start = trace[selected_interval];
    const auto & goal = trace[selected_interval + 1];
    if (start.q.allFinite() && goal.q.allFinite() &&
      std::isfinite(start.t) && std::isfinite(goal.t))
    {
      std::vector<cps_safety_monitor::RobotReachCapsule> capsules;
      std::vector<bool> intersections;
      if (interval_capsules(
          start.q,
          goal.q,
          start.t,
          goal.t,
          dynamic_alpha,
          &capsules) &&
        capsule_intersections(
          capsules, start.t, goal.t, &intersections))
      {
        append_interval(
          capsules,
          intersections,
          selected_contact_energy_unsafe);
      }
    }
  }

  const std::size_t current_marker_count =
    static_cast<std::size_t>(marker_id);
  for (std::size_t id = current_marker_count;
    id < last_reachable_set_visualization_marker_count_;
    ++id)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = reachable_set_visualization_frame_id_;
    marker.header.stamp = stamp;
    marker.ns = kReachableSetNamespace;
    marker.id = static_cast<int>(id);
    marker.action = visualization_msgs::msg::Marker::DELETE;
    marker_array.markers.push_back(std::move(marker));
  }
  last_reachable_set_visualization_marker_count_ = current_marker_count;

  if (!marker_array.markers.empty()) {
    reachable_set_visualization_pub_->publish(marker_array);
  }
}

void ReachableCartesianImpedanceController::clearRobotReachableSetVisualization()
{
  if (!reachable_set_visualization_pub_) {
    return;
  }
  visualization_msgs::msg::MarkerArray marker_array;
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = reachable_set_visualization_frame_id_;
  marker.header.stamp = get_node()->now();
  marker.ns = kReachableSetNamespace;
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(std::move(marker));
  reachable_set_visualization_pub_->publish(marker_array);
  last_reachable_set_visualization_marker_count_ = 0;
}

}  // namespace cps_controllers
