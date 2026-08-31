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
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Quaterniond = Eigen::Quaterniond;

// Robot-model quantities evaluated at one predicted joint state.  The safety
// monitor deliberately depends on this small interface instead of a concrete
// rigid-body library, so a controller can use either its hardware model or a
// thread-local simulation model.
struct JointDynamicsSample {
  bool valid{false};
  Vector3d control_position{Vector3d::Zero()};
  Quaterniond control_orientation{Quaterniond::Identity()};
  Matrix67d control_jacobian{Matrix67d::Zero()};
  Vector6d control_jdot_dq{Vector6d::Zero()};
  Matrix7d inertia{Matrix7d::Zero()};
  Vector7d coriolis{Vector7d::Zero()};
};

struct JointDynamicsLimits {
  Vector7d position_lower{Vector7d::Constant(
      -std::numeric_limits<double>::infinity())};
  Vector7d position_upper{Vector7d::Constant(
      std::numeric_limits<double>::infinity())};
  Vector7d velocity{Vector7d::Constant(
      std::numeric_limits<double>::infinity())};
  Vector7d acceleration{Vector7d::Constant(
      std::numeric_limits<double>::infinity())};
  Vector7d torque{Vector7d::Constant(
      std::numeric_limits<double>::infinity())};
};

// One state from the joint-space rollout.  This is an optional diagnostic
// trace: control decisions never depend on whether the trace is requested.
struct JointPredictionSample {
  double t{0.0};
  Vector7d q{Vector7d::Zero()};
  Vector7d dq{Vector7d::Zero()};
  // Nominal predicted endpoint energies before adding the configured
  // one-sided prediction-error bounds. These fields are for calibration and
  // logging; verification adds the bounds separately.
  bool energy_valid{false};
  double joint_kinetic_energy{0.0};
  double cartesian_potential_energy{0.0};
};

class JointDynamicsProvider {
 public:
  virtual ~JointDynamicsProvider() = default;
  virtual bool evaluate(const Vector7d& q,
                        const Vector7d& dq,
                        JointDynamicsSample* sample) const = 0;
  virtual JointDynamicsLimits limits() const = 0;
};

struct MonitorResult {
  bool monitored_contact_possible{false};
  // Current measured collision geometry overlaps the human workspace. This is
  // the runtime Cartesian energy-budget gate; predicted contact remains a
  // separate candidate-verification input in monitored_contact_possible.
  bool contact_relevant_for_energy{false};
  bool monitored_unsafe{false};
  // Predicted collision energy exceeds the configured budget, or a predicted
  // joint-limit violation was found. A collision possibility that remains
  // within the energy budget does not trigger candidate rejection.
  bool predicted_trigger{false};

  double workspace_distance_now{0.0};
  double workspace_distance_min{0.0};

  double worst_case_contact_time{0.0};
  double worst_case_workspace_distance_at_candidate{0.0};
  // Cartesian projection retained for diagnosis at the worst monitored
  // contact sample. Candidate gating uses joint kinetic + Cartesian potential.
  double worst_case_cartesian_kinetic_energy_ub{0.0};
  double worst_case_cartesian_potential_energy_ub{0.0};
  double worst_case_cartesian_control_energy_ub{0.0};
  // Paper-consistent total kinetic energy 1/2 dq^T M(q) dq.  Cartesian
  // kinetic values above are retained as diagnostics only.
  double worst_case_joint_kinetic_energy_ub{0.0};
  double worst_case_total_control_energy_ub{0.0};

  double workspace_distance_margin{0.0};

  double current_cartesian_kinetic_energy{0.0};
  double current_cartesian_potential_energy{0.0};
  double current_cartesian_control_energy{0.0};
  bool current_cartesian_energy_valid{false};
  double current_joint_kinetic_energy{0.0};
  double current_total_control_energy{0.0};
  bool current_joint_energy_valid{false};

  // Diagnostic comparison at exactly the same measured q/dq.  The runtime
  // matrix comes from current_joint_dynamics, while the prediction matrix is
  // evaluated by JointDynamicsProvider.  These values never affect the
  // verification decision.
  bool inertia_model_comparison_valid{false};
  double runtime_model_joint_kinetic_energy{0.0};
  double prediction_model_joint_kinetic_energy{0.0};
  double inertia_model_kinetic_energy_error{0.0};
  double inertia_model_difference_frobenius_norm{0.0};
  double inertia_model_difference_relative_frobenius_norm{0.0};
  double inertia_model_difference_max_abs{0.0};
  int inertia_model_difference_max_abs_row{-1};
  int inertia_model_difference_max_abs_col{-1};
  bool inertia_model_energy_ratio_valid{false};
  double inertia_model_min_energy_ratio{0.0};
  double inertia_model_max_energy_ratio{0.0};

  double worst_case_pos_error_radius{0.0};
  double worst_case_orientation_error_radius{0.0};
  double worst_case_vel_error_radius{0.0};

  bool collision_energy_unsafe{false};
  bool joint_limit_unsafe{false};
  int joint_limit_index{-1};
  double joint_position_violation{0.0};
  double joint_velocity_violation{0.0};
  double joint_acceleration_violation{0.0};
  double joint_torque_violation{0.0};

  double terminal_energy_ub{0.0};
};

struct ImpedanceSample {
  double t{0.0};
  // Progress on the nominal timed path.  This is kept separate from t,
  // because t is retimed to the monitored/execution horizon.
  double nominal_path_time{0.0};
  bool nominal_path_time_valid{false};
  // Explicit scalar path kinematics. Cartesian dp/ddp cannot recover these
  // at a legitimate direction-reversal cusp because the path tangent is zero.
  double nominal_path_rate{0.0};
  double nominal_path_acceleration{0.0};
  bool nominal_path_kinematics_valid{false};

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

  // Calibration-only geometry policy. When true, the robot dynamics and
  // energy rollout still run, but every human-workspace distance is treated
  // as +infinity. Runtime controllers must enable this only through an
  // explicit non-safety calibration mode; missing workspace data is not
  // evidence that no human is present.
  bool assume_human_workspace_clear{false};

  Matrix6d K_runtime{Matrix6d::Zero()};
  Matrix6d D_runtime{Matrix6d::Zero()};

  // Command that was actually being executed when the measured state used by
  // this monitor snapshot was captured. It must be expressed at the same
  // Cartesian point as current_position and J_geo.
  ImpedanceSample current_energy_reference;
  bool current_energy_reference_valid{false};

  // Live quantities captured from the hardware interface at the same instant
  // as current_q/current_dq.  Supplying them avoids recomputing the initial
  // rollout state with a separate model; the provider is still required for
  // future predicted q/dq samples.
  JointDynamicsSample current_joint_dynamics;
  bool current_joint_dynamics_valid{false};
  // Evaluate the prediction provider once at the measured q/dq and compare
  // its inertia with current_joint_dynamics.inertia. Logging code enables
  // this only when prediction diagnostics are requested.
  bool enable_inertia_model_comparison{false};

  double wall_time_sec{0.0};
  double energy_budget_joule{0.05};
  // Direct one-sided energy-model error bounds:
  // K_real <= K_pred + kinetic_energy_error_bound_joule and
  // V_real <= V_pred + potential_energy_error_bound_joule.
  double kinetic_energy_error_bound_joule{0.0};
  double potential_energy_error_bound_joule{0.0};
  double ee_collision_radius{0.04};
  // Fixed certified bounds between a predicted Cartesian state and the state
  // reached by the real robot after applying the verified command.
  double tracking_position_error_bound{0.0};
  double tracking_orientation_error_bound{0.0};
  double tracking_acc_error_bound{0.2};
  bool use_dynamic_consistent_impedance{true};
  Vector7d nullspace_reference{Vector7d::Zero()};
  double nullspace_stiffness{0.0};
  // Keep the joint rollout consistent with controllers that remove
  // nullspace torque while executing their fail-safe trajectory.
  bool disable_nullspace_in_failsafe{false};
  Vector7d previous_torque_command{Vector7d::Zero()};
  bool previous_torque_command_valid{false};
  double torque_rate_limit{1000.0};
  double joint_rollout_max_dt{0.001};
  Vector3d collision_center_offset{Vector3d::Zero()};
};

MonitorResult verifyReachablePlanJointSpace(
    const VerifiedPlan& plan,
    const Vector7d& current_q,
    const Vector7d& current_dq,
    const JointDynamicsProvider& dynamics,
    const SafetyMonitorConfig& config,
    std::vector<JointPredictionSample>* prediction_trace = nullptr);

MonitorResult verifyReachablePlan(const VerifiedPlan& plan,
                                  const Vector3d& current_position,
                                  const Quaterniond& current_orientation,
                                  const Vector6d& ee_twist,
                                  const Matrix7d& inertia,
                                  const Matrix67d& J_geo,
                                  const SafetyMonitorConfig& config);

}  // namespace cps_safety_monitor
