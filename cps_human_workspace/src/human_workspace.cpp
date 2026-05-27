#include "cps_human_workspace/human_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

namespace cps_human_workspace {

void HumanWorkspace::declareParameters(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node) {
  if (!node->has_parameter("human_workspace_config_path")) {
    node->declare_parameter<std::string>("human_workspace_config_path", "");
  }
}

std::string HumanWorkspace::defaultConfigPath() {
  return ament_index_cpp::get_package_share_directory("cps_human_workspace") +
         "/config/human_workspace.yaml";
}

bool HumanWorkspace::configureFromDefaultConfig(const rclcpp::Logger& logger) {
  return configureFromConfigFile(defaultConfigPath(), logger);
}

bool HumanWorkspace::configureFromConfigFile(
    const std::string& config_path,
    const rclcpp::Logger& logger) {
  try {
    const YAML::Node root = YAML::LoadFile(config_path);

    const auto read_vector3 = [&](const char* key, Vector3d* out) -> bool {
      const YAML::Node value = root[key];
      if (!value || !value.IsSequence() || value.size() != 3) {
        RCLCPP_ERROR(logger, "%s must be a 3-element list in %s.", key, config_path.c_str());
        return false;
      }
      *out = Vector3d(
          value[0].as<double>(),
          value[1].as<double>(),
          value[2].as<double>());
      return true;
    };

    Parameters parameters;
    if (!read_vector3("plane_normal", &parameters.plane_normal) ||
        !read_vector3("sphere_center", &parameters.sphere_center)) {
      return false;
    }

    if (!root["motion_radius"] || !root["hand_radius"]) {
      RCLCPP_ERROR(
          logger,
          "motion_radius and hand_radius must be set in %s.",
          config_path.c_str());
      return false;
    }

    parameters.motion_radius = root["motion_radius"].as<double>();
    parameters.hand_radius = root["hand_radius"].as<double>();

    if (parameters.motion_radius < 0.0 || parameters.hand_radius < 0.0) {
      RCLCPP_ERROR(logger, "motion_radius and hand_radius must be nonnegative.");
      return false;
    }

    if (parameters.plane_normal.norm() < 1e-8) {
      RCLCPP_ERROR(logger, "plane_normal norm too small.");
      return false;
    }

    setParameters(parameters);
    RCLCPP_INFO(logger, "Loaded human workspace config: %s", config_path.c_str());
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(
        logger,
        "Failed to load human workspace config %s: %s",
        config_path.c_str(),
        e.what());
    return false;
  }
}

bool HumanWorkspace::configureFromParameters(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const rclcpp::Logger& logger) {
  std::string config_path;
  if (node->has_parameter("human_workspace_config_path")) {
    config_path = node->get_parameter("human_workspace_config_path").as_string();
  }
  return config_path.empty()
             ? configureFromDefaultConfig(logger)
             : configureFromConfigFile(config_path, logger);
}

void HumanWorkspace::setParameters(const Parameters& parameters) {
  parameters_ = parameters;
  parameters_.plane_normal.normalize();
}

double HumanWorkspace::inflatedHandRadius() const {
  return parameters_.motion_radius + parameters_.hand_radius;
}

double HumanWorkspace::inflatedCollisionRadius(
    double ee_collision_radius,
    double position_error_radius) const {
  return inflatedHandRadius() + ee_collision_radius + std::max(0.0, position_error_radius);
}

double HumanWorkspace::signedDistanceToInflatedSphere(
    const Vector3d& point,
    double inflated_radius) const {
  return (point - parameters_.sphere_center).norm() - inflated_radius;
}

double HumanWorkspace::signedDistanceSegmentToInflatedSphere(
    const Vector3d& a,
    const Vector3d& b,
    double inflated_radius,
    Vector3d* closest_point) const {
  const Vector3d closest = closestPointOnSegment(a, b, parameters_.sphere_center);
  if (closest_point != nullptr) {
    *closest_point = closest;
  }
  return (closest - parameters_.sphere_center).norm() - inflated_radius;
}

double HumanWorkspace::normalStiffness(const Matrix6d& K) const {
  const Vector3d& n = parameters_.plane_normal;
  return std::max((n.transpose() * K.topLeftCorner<3, 3>() * n)(0, 0), 0.0);
}

double HumanWorkspace::normalDamping(const Matrix6d& D) const {
  const Vector3d& n = parameters_.plane_normal;
  return std::max((n.transpose() * D.topLeftCorner<3, 3>() * n)(0, 0), 0.0);
}

Vector3d HumanWorkspace::closestPointOnSegment(
    const Vector3d& a,
    const Vector3d& b,
    const Vector3d& point) {
  const Vector3d ab = b - a;
  const double denom = ab.squaredNorm();

  if (denom < 1e-12) {
    return a;
  }

  const double alpha = std::clamp((point - a).dot(ab) / denom, 0.0, 1.0);
  return a + alpha * ab;
}

}  // namespace cps_human_workspace
