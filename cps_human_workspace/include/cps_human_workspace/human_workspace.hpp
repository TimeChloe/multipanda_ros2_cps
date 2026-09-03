#pragma once

#include <string>

#include <Eigen/Dense>

#include <rclcpp/logger.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace cps_human_workspace {

using Vector3d = Eigen::Matrix<double, 3, 1>;

class HumanWorkspace {
 public:
  struct ReachableSphere {
    Vector3d center{Vector3d::Zero()};
    double radius{0.0};
  };

  struct Parameters {
    Vector3d sphere_center{Vector3d(0.0, 0.0, 0.2)};
    Vector3d center_velocity{Vector3d::Zero()};
    Vector3d center_sinusoid_amplitude{Vector3d::Zero()};
    double center_sinusoid_frequency_hz{0.0};
    double center_sinusoid_phase_rad{0.0};
    double center_motion_time_offset_sec{0.0};
    // Physical hand enclosure radius. SaRA BodyPartCombined prediction
    // uncertainty is added separately for every monitored interval.
    double motion_radius{0.14};
    double hand_max_velocity{2.0};
    double hand_max_acceleration{50.0};
    double measurement_error_position{0.0};
    double measurement_error_velocity{0.0};
    double measurement_delay{0.0};
  };

  static void declareParameters(
      const rclcpp_lifecycle::LifecycleNode::SharedPtr& node);

  static std::string defaultConfigPath();

  bool configureFromDefaultConfig(const rclcpp::Logger& logger);

  bool configureFromConfigFile(const std::string& config_path,
                               const rclcpp::Logger& logger);

  bool configureFromParameters(
      const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
      const rclcpp::Logger& logger);

  void setParameters(const Parameters& parameters);

  const Parameters& parameters() const { return parameters_; }
  const Vector3d& center() const { return parameters_.sphere_center; }
  Vector3d centerAtTime(double time_sec) const;
  Vector3d centerVelocityAtTime(double time_sec) const;
  bool hasMovingCenter() const;
  const Vector3d& centerLinearVelocity() const {
    return parameters_.center_velocity;
  }
  const Vector3d& centerSinusoidAmplitude() const {
    return parameters_.center_sinusoid_amplitude;
  }
  double centerSinusoidFrequencyHz() const {
    return parameters_.center_sinusoid_frequency_hz;
  }
  double centerSinusoidPhaseRad() const {
    return parameters_.center_sinusoid_phase_rad;
  }
  double centerMotionTimeOffsetSec() const {
    return parameters_.center_motion_time_offset_sec;
  }
  double motionRadius() const { return parameters_.motion_radius; }
  double handMaxVelocity() const { return parameters_.hand_max_velocity; }
  double handMaxAcceleration() const {
    return parameters_.hand_max_acceleration;
  }
  double measurementErrorPosition() const {
    return parameters_.measurement_error_position;
  }
  double measurementErrorVelocity() const {
    return parameters_.measurement_error_velocity;
  }
  double measurementDelay() const { return parameters_.measurement_delay; }
  ReachableSphere handReachableSetAtTime(double time_sec) const;
  double inflatedCollisionRadius(double ee_collision_radius,
                                 double position_error_radius) const;

  double signedDistanceToInflatedSphere(const Vector3d& point,
                                        double inflated_radius,
                                        double time_sec) const;

  double signedDistanceSegmentToInflatedSphere(
      const Vector3d& a,
      const Vector3d& b,
      double inflated_radius,
      double start_time_sec,
      double end_time_sec,
      Vector3d* closest_robot_point = nullptr,
      Vector3d* closest_human_center = nullptr) const;

  static Vector3d closestPointOnSegment(const Vector3d& a,
                                        const Vector3d& b,
                                        const Vector3d& point);

 private:
  Parameters parameters_;
};

}  // namespace cps_human_workspace
