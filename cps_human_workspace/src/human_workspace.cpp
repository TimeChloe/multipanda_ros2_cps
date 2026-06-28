#include "cps_human_workspace/human_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

namespace cps_human_workspace {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

}  // namespace

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

    const auto read_vector3 =
        [&](const YAML::Node& parent, const char* key, Vector3d* out) -> bool {
      const YAML::Node value = parent[key];
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
    if (root["workspace_direction"]) {
      if (!read_vector3(root, "workspace_direction", &parameters.workspace_direction)) {
        return false;
      }
    } else if (root["plane_normal"]) {
      if (!read_vector3(root, "plane_normal", &parameters.workspace_direction)) {
        return false;
      }
    } else {
      RCLCPP_ERROR(
          logger,
          "workspace_direction must be a 3-element list in %s.",
          config_path.c_str());
      return false;
    }

    if (!read_vector3(root, "sphere_center", &parameters.sphere_center)) {
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

    const YAML::Node center_motion =
        root["center_motion"] ? root["center_motion"] : root["motion"];
    if (center_motion) {
      if (!center_motion.IsMap()) {
        RCLCPP_ERROR(
            logger,
            "center_motion must be a map in %s.",
            config_path.c_str());
        return false;
      }

      if (center_motion["velocity"] &&
          !read_vector3(center_motion, "velocity", &parameters.center_velocity)) {
        return false;
      }

      if (center_motion["sinusoid_amplitude"]) {
        if (!read_vector3(
                center_motion,
                "sinusoid_amplitude",
                &parameters.center_sinusoid_amplitude)) {
          return false;
        }
      } else if (center_motion["amplitude"]) {
        if (!read_vector3(
                center_motion,
                "amplitude",
                &parameters.center_sinusoid_amplitude)) {
          return false;
        }
      }

      if (center_motion["sinusoid_frequency_hz"]) {
        parameters.center_sinusoid_frequency_hz =
            center_motion["sinusoid_frequency_hz"].as<double>();
      } else if (center_motion["frequency_hz"]) {
        parameters.center_sinusoid_frequency_hz =
            center_motion["frequency_hz"].as<double>();
      }

      if (center_motion["sinusoid_phase_rad"]) {
        parameters.center_sinusoid_phase_rad =
            center_motion["sinusoid_phase_rad"].as<double>();
      } else if (center_motion["phase_rad"]) {
        parameters.center_sinusoid_phase_rad =
            center_motion["phase_rad"].as<double>();
      } else if (center_motion["phase"]) {
        parameters.center_sinusoid_phase_rad =
            center_motion["phase"].as<double>();
      }

      if (center_motion["time_offset_sec"]) {
        parameters.center_motion_time_offset_sec =
            center_motion["time_offset_sec"].as<double>();
      }
    }

    if (parameters.motion_radius < 0.0 || parameters.hand_radius < 0.0 ||
        parameters.center_sinusoid_frequency_hz < 0.0) {
      RCLCPP_ERROR(
          logger,
          "motion_radius, hand_radius, and center sinusoid frequency must be nonnegative.");
      return false;
    }

    if (parameters.workspace_direction.norm() < 1e-8) {
      RCLCPP_ERROR(logger, "workspace_direction norm too small.");
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
  parameters_.workspace_direction.normalize();
}

Vector3d HumanWorkspace::centerAtTime(double time_sec) const {
  const double tau = time_sec - parameters_.center_motion_time_offset_sec;
  const double phase =
      kTwoPi * parameters_.center_sinusoid_frequency_hz * tau +
      parameters_.center_sinusoid_phase_rad;
  return parameters_.sphere_center +
         parameters_.center_velocity * tau +
         parameters_.center_sinusoid_amplitude * std::sin(phase);
}

Vector3d HumanWorkspace::centerVelocityAtTime(double time_sec) const {
  const double tau = time_sec - parameters_.center_motion_time_offset_sec;
  const double omega = kTwoPi * parameters_.center_sinusoid_frequency_hz;
  const double phase = omega * tau + parameters_.center_sinusoid_phase_rad;
  return parameters_.center_velocity +
         parameters_.center_sinusoid_amplitude * omega * std::cos(phase);
}

bool HumanWorkspace::hasMovingCenter() const {
  return parameters_.center_velocity.squaredNorm() > 1.0e-18 ||
         (parameters_.center_sinusoid_amplitude.squaredNorm() > 1.0e-18 &&
          parameters_.center_sinusoid_frequency_hz > 0.0);
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

double HumanWorkspace::signedDistanceToInflatedSphere(
    const Vector3d& point,
    double inflated_radius,
    double time_sec) const {
  return (point - centerAtTime(time_sec)).norm() - inflated_radius;
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

double HumanWorkspace::signedDistanceSegmentToInflatedSphere(
    const Vector3d& a,
    const Vector3d& b,
    double inflated_radius,
    double start_time_sec,
    double end_time_sec,
    Vector3d* closest_robot_point,
    Vector3d* closest_human_center) const {
  const Vector3d center_start = centerAtTime(start_time_sec);
  const Vector3d center_end = centerAtTime(end_time_sec);

  const Vector3d rel_start = a - center_start;
  const Vector3d rel_end = b - center_end;
  const Vector3d rel_delta = rel_end - rel_start;
  const double denom = rel_delta.squaredNorm();
  const double alpha =
      denom < 1e-12
          ? 0.0
          : std::clamp(-rel_start.dot(rel_delta) / denom, 0.0, 1.0);

  const Vector3d robot_point = a + alpha * (b - a);
  const Vector3d human_center = center_start + alpha * (center_end - center_start);
  if (closest_robot_point != nullptr) {
    *closest_robot_point = robot_point;
  }
  if (closest_human_center != nullptr) {
    *closest_human_center = human_center;
  }
  return (robot_point - human_center).norm() - inflated_radius;
}

double HumanWorkspace::normalStiffness(const Matrix6d& K) const {
  const Vector3d& n = parameters_.workspace_direction;
  return std::max((n.transpose() * K.topLeftCorner<3, 3>() * n)(0, 0), 0.0);
}

double HumanWorkspace::normalDamping(const Matrix6d& D) const {
  const Vector3d& n = parameters_.workspace_direction;
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
