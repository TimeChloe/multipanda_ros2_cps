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

using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Quaterniond = Eigen::Quaterniond;

struct MonitorResult {
  bool monitored_contact_possible{false};
  bool monitored_unsafe{false};
  bool predicted_trigger{false};

  double plane_distance_now{0.0};
  double plane_distance_min{0.0};

  double m_eff_n{0.0};
  double v_n_now{0.0};
  double Tn_now{0.0};
  double v_safe{0.0};

  bool nominal_contact_sample_found{false};
  double nominal_contact_time{0.0};
  double nominal_contact_distance{0.0};
  double v_n_contact_nominal{0.0};
  double Tn_contact_nominal{0.0};
  Vector3d nominal_contact_point_world{Vector3d::Zero()};

  bool worst_case_contact_found{false};
  double worst_case_contact_time{0.0};
  double worst_case_plane_distance_at_candidate{0.0};
  double worst_case_nominal_forward_progress{0.0};

  double worst_case_v_n_ub{0.0};
  double worst_case_Tn_ub{0.0};
  double worst_case_a_pos{0.0};
  double worst_case_a_brake{0.0};
  double worst_case_a_net{0.0};

  double h_geom{0.0};
  double h_monitored_energy{0.0};

  double v_n_now_tube{0.0};
  double Tn_now_tube{0.0};
  double Tn_dot_est{0.0};

  double current_pos_error_radius{0.0};
  double current_vel_error_radius{0.0};
  double worst_case_pos_error_radius{0.0};
  double worst_case_vel_error_radius{0.0};

  double worst_case_V_potential_ub{0.0};
  double h_clamping_energy{std::numeric_limits<double>::infinity()};
  bool clamping_energy_unsafe{false};
  bool collision_energy_unsafe{false};

  double terminal_energy_ub{0.0};
  double h_terminal_energy{std::numeric_limits<double>::infinity()};
  bool terminal_energy_unsafe{false};
};

struct ImpedanceSample {
  double t{0.0};

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

  double k_rate_limit{5000.0};
  double d_rate_limit{500.0};
  double safe_collision_energy_joule{0.05};
  double clamping_energy_budget_joule{0.05};
  double energy_budget_margin_joule{0.005};
  double ee_collision_radius{0.04};
  double tracking_pos_error_bound{0.005};
  double tracking_vel_error_bound{0.05};
  bool use_dynamic_consistent_impedance{true};
};

MonitorResult verifyReachablePlan(const VerifiedPlan& plan,
                                  const Vector3d& current_position,
                                  const Vector6d& ee_twist,
                                  const Matrix7d& inertia,
                                  const Matrix37d& Jv,
                                  const SafetyMonitorConfig& config);

}  // namespace cps_safety_monitor
