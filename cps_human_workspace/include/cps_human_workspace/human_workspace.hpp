#pragma once

#include <string>

#include <Eigen/Dense>

#include <rclcpp/logger.hpp>

namespace cps_human_workspace
{

using Vector3d = Eigen::Matrix<double, 3, 1>;

class HumanWorkspace
{
public:
  struct ReachableSphere
  {
    Vector3d center{Vector3d::Zero()};
    double radius{0.0};
  };

  struct Parameters
  {
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

  bool configureFromConfigFile(
    const std::string & config_path,
    const rclcpp::Logger & logger);

  bool configureReachabilityFromConfigFile(
    const std::string & config_path,
    const rclcpp::Logger & logger);

  void setParameters(const Parameters & parameters);

  const Parameters & parameters() const {return parameters_;}
  Vector3d centerAtTime(double time_sec) const;
  Vector3d centerVelocityAtTime(double time_sec) const;
  ReachableSphere handReachableSetAtTime(double time_sec) const;
  double inflatedCollisionRadius(
    double ee_collision_radius,
    double position_error_radius) const;

  double signedDistanceToInflatedSphere(
    const Vector3d & point,
    double inflated_radius,
    double time_sec) const;

  double signedDistanceSegmentToInflatedSphere(
    const Vector3d & a,
    const Vector3d & b,
    double inflated_radius,
    double end_time_sec) const;

private:
  bool loadConfigFile(
    const std::string & config_path,
    const rclcpp::Logger & logger,
    bool load_motion_source);

  static Vector3d closestPointOnSegment(
    const Vector3d & a,
    const Vector3d & b,
    const Vector3d & point);

  Parameters parameters_;
};

}  // namespace cps_human_workspace
