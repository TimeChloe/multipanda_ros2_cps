#include "cps_safety_monitor/reachable_safety_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include "safety_shield/robot_arm_reach.h"

namespace cps_safety_monitor {

namespace {

class SaraRobotReachabilityProvider final
    : public RobotReachabilityProvider {
 public:
  SaraRobotReachabilityProvider(
      const std::string& robot_config_path,
      double secure_radius)
      : secure_radius_(secure_radius) {
    if (robot_config_path.empty()) {
      throw std::invalid_argument(
          "SaRA robot configuration path must not be empty");
    }
    if (!std::isfinite(secure_radius_) || secure_radius_ < 0.0) {
      throw std::invalid_argument(
          "SaRA secure_radius must be finite and nonnegative");
    }

    const YAML::Node config = YAML::LoadFile(robot_config_path);
    const int joint_count = config["nb_joints"].as<int>();
    if (joint_count != 7) {
      throw std::invalid_argument(
          "SaRA robot configuration must contain exactly seven joints");
    }
    const std::vector<double> transformations =
        config["transformation_matrices"].as<std::vector<double>>();
    const std::vector<double> enclosures =
        config["enclosures"].as<std::vector<double>>();
    const std::vector<double> link_masses =
        config["link_masses"].as<std::vector<double>>();
    const std::vector<double> link_inertias =
        config["link_inertias"].as<std::vector<double>>();
    const std::vector<double> link_centers_of_mass =
        config["link_centers_of_mass"].as<std::vector<double>>();

    if (transformations.size() != static_cast<std::size_t>(16 * joint_count) ||
        enclosures.size() != static_cast<std::size_t>(7 * joint_count) ||
        link_masses.size() < static_cast<std::size_t>(joint_count) ||
        link_inertias.size() < static_cast<std::size_t>(6 * joint_count) ||
        link_centers_of_mass.size() <
            static_cast<std::size_t>(6 * joint_count)) {
      throw std::invalid_argument(
          "SaRA robot configuration has inconsistent Panda array sizes");
    }

    std::unordered_map<int, std::set<int>> unclampable_enclosures;
    if (config["unclampable_enclosures"]) {
      const auto pairs = config["unclampable_enclosures"]
                             .as<std::vector<std::pair<int, int>>>();
      for (const auto& pair : pairs) {
        unclampable_enclosures[pair.first].insert(pair.second);
      }
    }

    robot_reach_ = std::make_unique<safety_shield::RobotArmReach>(
        transformations,
        joint_count,
        enclosures,
        link_masses,
        link_inertias,
        link_centers_of_mass,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        secure_radius_,
        unclampable_enclosures);
  }

  bool reachInterval(
      const Vector7d& start_q,
      const Vector7d& goal_q,
      double interval_duration_sec,
      const std::vector<double>& alpha_i,
      std::vector<RobotReachCapsule>* capsules) const override {
    if (capsules == nullptr) {
      return false;
    }
    capsules->clear();
    if (!start_q.allFinite() || !goal_q.allFinite() ||
        !std::isfinite(interval_duration_sec) ||
        interval_duration_sec < 0.0 || alpha_i.size() != 7 ||
        std::any_of(
            alpha_i.begin(), alpha_i.end(), [](double value) {
              return !std::isfinite(value) || value < 0.0;
            })) {
      return false;
    }

    const std::vector<double> start(
        start_q.data(), start_q.data() + start_q.size());
    const std::vector<double> goal(
        goal_q.data(), goal_q.data() + goal_q.size());
    try {
      const std::vector<reach_lib::Capsule> sara_capsules =
          robot_reach_->reach(
              start,
              goal,
              Eigen::Matrix4d::Identity(),
              Eigen::Matrix4d::Identity(),
              interval_duration_sec,
              alpha_i);
      capsules->clear();
      capsules->reserve(sara_capsules.size());
      for (const auto& capsule : sara_capsules) {
        RobotReachCapsule converted;
        converted.p1 = Vector3d(
            capsule.p1_.x, capsule.p1_.y, capsule.p1_.z);
        converted.p2 = Vector3d(
            capsule.p2_.x, capsule.p2_.y, capsule.p2_.z);
        converted.radius = capsule.r_;
        if (!converted.p1.allFinite() || !converted.p2.allFinite() ||
            !std::isfinite(converted.radius) || converted.radius < 0.0) {
          capsules->clear();
          return false;
        }
        capsules->push_back(converted);
      }
      return !capsules->empty();
    } catch (const std::exception&) {
      capsules->clear();
      return false;
    }
  }

  bool calculateTrajectoryAlpha(
      const std::vector<JointPredictionSample>& trajectory,
      std::vector<double>* alpha_i) const override {
    if (alpha_i == nullptr) {
      return false;
    }
    alpha_i->clear();
    if (trajectory.size() < 2) {
      return false;
    }

    std::lock_guard<std::mutex> lock(velocity_calculation_mutex_);
    std::vector<std::vector<safety_shield::RobotArmReach::CapsuleVelocity>>
        capsule_velocities;
    capsule_velocities.reserve(trajectory.size());
    try {
      for (const auto& sample : trajectory) {
        if (!std::isfinite(sample.t) || !sample.q.allFinite() ||
            !sample.dq.allFinite()) {
          alpha_i->clear();
          return false;
        }
        const std::vector<double> q(
            sample.q.data(), sample.q.data() + sample.q.size());
        const std::vector<double> dq(
            sample.dq.data(), sample.dq.data() + sample.dq.size());
        robot_reach_->calculateAllTransformationMatricesAndCapsules(
            q, Eigen::Matrix4d::Identity());
        capsule_velocities.push_back(
            robot_reach_->calculateAllCapsuleVelocities(dq));
        if (capsule_velocities.back().size() != 7) {
          alpha_i->clear();
          return false;
        }
      }
    } catch (const std::exception&) {
      alpha_i->clear();
      return false;
    }

    // This intentionally follows LongTermTraj::calculateAlphaBeta() in
    // SARA Shield: for each capsule endpoint, take the finite difference of
    // Cartesian speed magnitudes, then maximize over the entire trajectory.
    alpha_i->assign(7, 0.0);
    for (std::size_t sample_index = 1;
         sample_index < trajectory.size(); ++sample_index) {
      const double dt =
          trajectory[sample_index].t - trajectory[sample_index - 1].t;
      if (!std::isfinite(dt) || dt <= 0.0) {
        alpha_i->clear();
        return false;
      }
      for (std::size_t capsule_index = 0; capsule_index < 7;
           ++capsule_index) {
        const auto& before =
            capsule_velocities[sample_index - 1][capsule_index];
        const auto& after =
            capsule_velocities[sample_index][capsule_index];
        const double alpha_1 =
            std::abs(after.v1.v.norm() - before.v1.v.norm()) / dt;
        const double alpha_2 =
            std::abs(after.v2.v.norm() - before.v2.v.norm()) / dt;
        const double value = std::max(alpha_1, alpha_2);
        if (!std::isfinite(value)) {
          alpha_i->clear();
          return false;
        }
        (*alpha_i)[capsule_index] =
            std::max((*alpha_i)[capsule_index], value);
      }
    }
    return true;
  }

  double minimumSignedDistance(
      const std::vector<RobotReachCapsule>& robot_capsules,
      const Vector3d& human_center_start,
      const Vector3d& human_center_end,
      double human_radius,
      int* closest_robot_link_index) const override {
    if (closest_robot_link_index != nullptr) {
      *closest_robot_link_index = -1;
    }
    if (!human_center_start.allFinite() ||
        !human_center_end.allFinite() ||
        !std::isfinite(human_radius) || human_radius < 0.0) {
      return -std::numeric_limits<double>::infinity();
    }

    const reach_lib::Capsule human_capsule(
        reach_lib::Point(
            human_center_start.x(),
            human_center_start.y(),
            human_center_start.z()),
        reach_lib::Point(
            human_center_end.x(),
            human_center_end.y(),
            human_center_end.z()),
        human_radius);
    double minimum_distance = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < robot_capsules.size(); ++index) {
      const RobotReachCapsule& capsule = robot_capsules[index];
      const reach_lib::Capsule sara_capsule(
          reach_lib::Point(
              capsule.p1.x(), capsule.p1.y(), capsule.p1.z()),
          reach_lib::Point(
              capsule.p2.x(), capsule.p2.y(), capsule.p2.z()),
          capsule.radius);
      const double distance =
          reach_lib::intersections::capsule_capsule_dist(
              sara_capsule, human_capsule);
      if (distance < minimum_distance) {
        minimum_distance = distance;
        if (closest_robot_link_index != nullptr) {
          *closest_robot_link_index = static_cast<int>(index);
        }
      }
    }
    return minimum_distance;
  }

  double secureRadius() const override { return secure_radius_; }

  const char* backendName() const override {
    return "sara_robot_arm_reach";
  }

 private:
  double secure_radius_{0.0};
  mutable std::mutex velocity_calculation_mutex_;
  std::unique_ptr<safety_shield::RobotArmReach> robot_reach_;
};

}  // namespace

std::shared_ptr<const RobotReachabilityProvider>
makeSaraRobotReachabilityProvider(
    const std::string& robot_config_path,
    double secure_radius) {
  return std::make_shared<SaraRobotReachabilityProvider>(
      robot_config_path, secure_radius);
}

std::string defaultSaraPandaRobotConfigPath() {
  try {
    return ament_index_cpp::get_package_share_directory(
               "cps_safety_monitor") +
           "/config/robot_parameters_panda.yaml";
  } catch (const std::exception&) {
    return "cps_safety_monitor/config/robot_parameters_panda.yaml";
  }
}

}  // namespace cps_safety_monitor
