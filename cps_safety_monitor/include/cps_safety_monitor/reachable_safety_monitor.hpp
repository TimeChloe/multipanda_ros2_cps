#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "cps_human_workspace/human_workspace.hpp"

namespace cps_safety_monitor {

using Matrix3d = Eigen::Matrix3d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Matrix37d = Eigen::Matrix<double, 3, 7>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;

using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Quaterniond = Eigen::Quaterniond;

struct MonitorResult {
  bool monitored_contact_possible{false};
  bool contact_relevant_for_energy{false};
  bool monitored_unsafe{false};
  // Current monitored trajectory failed future-contact verification.
  bool predicted_trigger{false};

  double workspace_distance_now{0.0};
  double workspace_distance_min{0.0};

  double worst_case_contact_time{0.0};
  double worst_case_workspace_distance_at_candidate{0.0};
  // Full 6D Cartesian control-energy upper bounds at the worst monitored
  // contact sample.
  double worst_case_cartesian_kinetic_energy_ub{0.0};
  double worst_case_cartesian_potential_energy_ub{0.0};
  double worst_case_cartesian_control_energy_ub{0.0};

  double workspace_distance_margin{0.0};
  double h_monitored_energy{0.0};

  double current_cartesian_kinetic_energy{0.0};
  double current_cartesian_potential_energy{0.0};
  double current_cartesian_control_energy{0.0};
  bool current_cartesian_energy_valid{false};

  double worst_case_pos_error_radius{0.0};
  double worst_case_vel_error_radius{0.0};

  bool collision_energy_unsafe{false};

  double terminal_energy_ub{0.0};
  double h_terminal_energy{std::numeric_limits<double>::infinity()};
};

struct ImpedanceSample {
  double t{0.0};
  // Progress on the nominal timed path.  This is kept separate from t,
  // because t is retimed to the monitored/execution horizon.
  double nominal_path_time{0.0};
  bool nominal_path_time_valid{false};

  Vector3d p{Vector3d::Zero()};
  Vector3d dp{Vector3d::Zero()};
  Vector3d ddp{Vector3d::Zero()};

  Quaterniond q{Quaterniond::Identity()};
  Vector3d w{Vector3d::Zero()};
  Vector3d dw{Vector3d::Zero()};

  Matrix6d K{Matrix6d::Zero()};
  Matrix6d D{Matrix6d::Zero()};

  bool failsafe{false};
};

struct VerifiedPlan {
  bool valid{false};

  ImpedanceSample anchor;
  std::vector<ImpedanceSample> intended;
  std::vector<ImpedanceSample> failsafe;

  std::size_t intended_exec_index{0};
  std::size_t failsafe_exec_index{0};

  double generated_wall_time{0.0};
  double nominal_time_anchor{0.0};
};

struct SafetyMonitorConfig {
  cps_human_workspace::HumanWorkspace human_workspace;

  Matrix6d K_runtime{Matrix6d::Zero()};
  Matrix6d D_runtime{Matrix6d::Zero()};

  // Command that was actually being executed when the measured state used by
  // this monitor snapshot was captured. It must be expressed at the same
  // Cartesian point as current_position and J_geo.
  ImpedanceSample current_energy_reference;
  bool current_energy_reference_valid{false};

  double wall_time_sec{0.0};
  double energy_budget_joule{0.05};
  double energy_budget_margin_joule{0.005};
  double ee_collision_radius{0.04};
  double contact_activation_margin{0.0};
  double tracking_acc_error_bound{0.2};
  bool use_dynamic_consistent_impedance{true};
};

MonitorResult verifyReachablePlan(const VerifiedPlan& plan,
                                  const Vector3d& current_position,
                                  const Quaterniond& current_orientation,
                                  const Vector6d& ee_twist,
                                  const Matrix7d& inertia,
                                  const Matrix67d& J_geo,
                                  const SafetyMonitorConfig& config);

}  // namespace cps_safety_monitor
