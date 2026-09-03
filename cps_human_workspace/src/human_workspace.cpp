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
    if (!read_vector3(root, "sphere_center", &parameters.sphere_center)) {
      return false;
    }

    if (!root["motion_radius"]) {
      RCLCPP_ERROR(
          logger,
          "motion_radius must be set in %s.",
          config_path.c_str());
      return false;
    }

    parameters.motion_radius = root["motion_radius"].as<double>();

    const YAML::Node hand_reachability = root["hand_reachability"];
    if (hand_reachability) {
      if (!hand_reachability.IsMap()) {
        RCLCPP_ERROR(
            logger,
            "hand_reachability must be a map in %s.",
            config_path.c_str());
        return false;
      }
      if (hand_reachability["model"]) {
        RCLCPP_ERROR(
            logger,
            "hand_reachability.model is obsolete in %s; this project now "
            "always uses the SaRA BodyPartCombined generator. Remove the "
            "model field.",
            config_path.c_str());
        return false;
      }
      if (hand_reachability["max_velocity"]) {
        parameters.hand_max_velocity =
            hand_reachability["max_velocity"].as<double>();
      }
      if (hand_reachability["max_acceleration"]) {
        parameters.hand_max_acceleration =
            hand_reachability["max_acceleration"].as<double>();
      }
      if (hand_reachability["measurement_error_position"]) {
        parameters.measurement_error_position =
            hand_reachability["measurement_error_position"].as<double>();
      }
      if (hand_reachability["measurement_error_velocity"]) {
        parameters.measurement_error_velocity =
            hand_reachability["measurement_error_velocity"].as<double>();
      }
      if (hand_reachability["delay"]) {
        parameters.measurement_delay =
            hand_reachability["delay"].as<double>();
      }
    }

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

    if (parameters.motion_radius < 0.0 ||
        parameters.center_sinusoid_frequency_hz < 0.0 ||
        parameters.hand_max_velocity < 0.0 ||
        parameters.hand_max_acceleration <= 0.0 ||
        parameters.measurement_error_position < 0.0 ||
        parameters.measurement_error_velocity < 0.0 ||
        parameters.measurement_delay < 0.0) {
      RCLCPP_ERROR(
          logger,
          "Hand radius, frequency, reachability limits, measurement errors "
          "and delay must be nonnegative; max_acceleration must be positive.");
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

HumanWorkspace::ReachableSphere HumanWorkspace::handReachableSetAtTime(
    double time_sec) const {
  ReachableSphere reachable;
  reachable.center = parameters_.sphere_center;
  reachable.radius = std::max(0.0, parameters_.motion_radius);

  // Exact single-point specialization of SaRA ReachLib's
  // BodyPartCombined::ry(). sphere_center and center_velocity are the latest
  // hand observation at center_motion_time_offset_sec. The physical hand
  // radius corresponds to the upstream body thickness divided by two.
  const double t =
      std::max(0.0, time_sec - parameters_.center_motion_time_offset_sec) +
      parameters_.measurement_delay;
  const Vector3d velocity = parameters_.center_velocity;
  const double speed = velocity.norm();
  const double max_velocity = parameters_.hand_max_velocity;
  const double max_acceleration = parameters_.hand_max_acceleration;

  if (speed > max_velocity) {
    // ReachLib falls back to an isotropic constant-speed ball when the
    // measured speed already exceeds the configured maximum.
    reachable.center = parameters_.sphere_center;
    reachable.radius += parameters_.measurement_error_position +
                        speed * t;
    return reachable;
  }

  Vector3d direction = Vector3d::UnitX();
  if (speed >= 1.0e-12) {
    direction = velocity / speed;
  }
  const double t_up = std::clamp(
      (max_velocity - speed) / max_acceleration, 0.0, t);
  const double t_max = std::clamp(
      max_velocity / max_acceleration, 0.0, t);
  const double t_down = std::clamp(
      (max_velocity + speed) / max_acceleration, 0.0, t);

  double reachability_radius = 0.0;
  if (t_down < t || (t_down >= t && t_up >= t)) {
    reachability_radius =
        max_acceleration * (t * t_max - 0.5 * t_max * t_max);
  } else {
    const double asymmetric_axis =
        0.5 *
        (t * (t_up + t_down) -
         0.5 * (t_up * t_up + t_down * t_down));
    const double transverse_axis =
        t * t_max - 0.5 * t_max * t_max;
    reachability_radius = max_acceleration * std::sqrt(
        asymmetric_axis * asymmetric_axis +
        transverse_axis * transverse_axis);
  }

  const double center_shift =
      t * speed +
      0.5 * max_acceleration *
          (t * (t_up - t_down) -
           0.5 * (t_up * t_up - t_down * t_down));
  reachable.center = parameters_.sphere_center + direction * center_shift;
  reachable.radius += reachability_radius +
                      parameters_.measurement_error_position +
                      parameters_.measurement_error_velocity * t;
  return reachable;
}

double HumanWorkspace::inflatedCollisionRadius(
    double ee_collision_radius,
    double position_error_radius) const {
  return parameters_.motion_radius + ee_collision_radius +
         std::max(0.0, position_error_radius);
}

double HumanWorkspace::signedDistanceToInflatedSphere(
    const Vector3d& point,
    double inflated_radius,
    double time_sec) const {
  const ReachableSphere hand_reach = handReachableSetAtTime(time_sec);
  const double reachability_inflation =
      std::max(0.0, hand_reach.radius - parameters_.motion_radius);
  return (point - hand_reach.center).norm() -
         (inflated_radius + reachability_inflation);
}

double HumanWorkspace::signedDistanceSegmentToInflatedSphere(
    const Vector3d& a,
    const Vector3d& b,
    double inflated_radius,
    double start_time_sec,
    double end_time_sec,
    Vector3d* closest_robot_point,
    Vector3d* closest_human_center) const {
  (void)start_time_sec;
  // The COMBINED ball at the end of the interval encloses all admissible
  // hand motion from the observation through this interval, matching the
  // single-point BodyPartCombined update used by SaRA-Shield.
  const ReachableSphere hand_reach = handReachableSetAtTime(end_time_sec);
  const Vector3d robot_point =
      closestPointOnSegment(a, b, hand_reach.center);
  if (closest_robot_point != nullptr) {
    *closest_robot_point = robot_point;
  }
  if (closest_human_center != nullptr) {
    *closest_human_center = hand_reach.center;
  }
  const double reachability_inflation =
      std::max(0.0, hand_reach.radius - parameters_.motion_radius);
  return (robot_point - hand_reach.center).norm() -
         (inflated_radius + reachability_inflation);
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
