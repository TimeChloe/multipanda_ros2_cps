#pragma once

#include <string>

#include <Eigen/Dense>

#include <rclcpp/logger.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace cps_human_workspace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector3d = Eigen::Matrix<double, 3, 1>;

class HumanWorkspace {
 public:
  struct Parameters {
    Vector3d plane_normal{Vector3d(0.0, 0.0, 1.0)};
    Vector3d sphere_center{Vector3d(0.0, 0.0, 0.2)};
    double motion_radius{0.10};
    double hand_radius{0.04};
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

  const Vector3d& normal() const { return parameters_.plane_normal; }
  const Vector3d& center() const { return parameters_.sphere_center; }
  double motionRadius() const { return parameters_.motion_radius; }
  double handRadius() const { return parameters_.hand_radius; }
  double inflatedHandRadius() const;
  double inflatedCollisionRadius(double ee_collision_radius,
                                 double position_error_radius) const;

  double signedDistanceToInflatedSphere(const Vector3d& point,
                                        double inflated_radius) const;

  double signedDistanceSegmentToInflatedSphere(
      const Vector3d& a,
      const Vector3d& b,
      double inflated_radius,
      Vector3d* closest_point = nullptr) const;

  double normalStiffness(const Matrix6d& K) const;
  double normalDamping(const Matrix6d& D) const;

  static Vector3d closestPointOnSegment(const Vector3d& a,
                                        const Vector3d& b,
                                        const Vector3d& point);

 private:
  Parameters parameters_;
};

}  // namespace cps_human_workspace
