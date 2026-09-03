#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
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

// Lachner et al. Eq. (14). The same scale must be applied to every
// controlled elastic potential (Cartesian and joint/nullspace). Damping is
// scaled by sqrt(scale) by the impedance controller.
double energyBudgetStiffnessScale(double kinetic_energy,
                                  double cartesian_potential_energy,
                                  double nullspace_potential_energy,
                                  double energy_budget);

// Lachner et al. Eqs. (16)-(17). Callers enable this only in the human
// collision area. q_star is captured on the first enabled cycle for which
// T > L_max and retained until the budget is recovered or the area is exited.
// This additional joint potential is distinct from the normal projected
// nullspace potential U_q.
struct OverbudgetJointStabilizationState {
  bool active{false};
  Vector7d reference{Vector7d::Zero()};
};

struct OverbudgetJointStabilizationTerms {
  bool active{false};
  double potential_energy{0.0};
  double scale_rho{1.0};
  Vector7d torque{Vector7d::Zero()};
};

OverbudgetJointStabilizationTerms updateOverbudgetJointStabilization(
    const Vector7d& q,
    double kinetic_energy,
    double energy_budget,
    double joint_stiffness,
    double scale_omega,
    bool enabled,
    OverbudgetJointStabilizationState* state);

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
  double nullspace_potential_energy{0.0};
  bool nullspace_potential_energy_active{false};
  // True when Eq. (14) was active for the rollout interval ending at this
  // sample. For the initial sample it records the current-state gate.
  bool energy_scaling_active{false};
  double energy_stiffness_scale{1.0};
  double applied_nullspace_stiffness{0.0};
  bool overbudget_joint_stabilization_active{false};
  double overbudget_joint_potential_energy{0.0};
  double overbudget_joint_scale_rho{1.0};
  double overbudget_joint_torque_norm{0.0};
};

class JointDynamicsProvider {
 public:
  virtual ~JointDynamicsProvider() = default;
  virtual bool evaluate(const Vector7d& q,
                        const Vector7d& dq,
                        JointDynamicsSample* sample) const = 0;
  virtual JointDynamicsLimits limits() const = 0;
};

// Geometry-only robot reachable occupancy.  The runtime implementation is an
// adapter around SaRA-Shield's RobotArmReach and ReachLib capsules; keeping
// this narrow interface here avoids exposing the third-party headers to every
// controller that consumes cps_safety_monitor.
struct RobotReachCapsule {
  Vector3d p1{Vector3d::Zero()};
  Vector3d p2{Vector3d::Zero()};
  double radius{0.0};
};

class RobotReachabilityProvider {
 public:
  virtual ~RobotReachabilityProvider() = default;

  // Return SaRA's enclosing capsules for the joint-space interval
  // [start_q, goal_q].  interval_duration_sec is passed as SaRA's s_diff.
  virtual bool reachInterval(
      const Vector7d& start_q,
      const Vector7d& goal_q,
      double interval_duration_sec,
      const std::vector<double>& alpha_i,
      std::vector<RobotReachCapsule>* capsules) const = 0;

  // Match SARA Shield's PFL path: compute one Cartesian-acceleration value
  // per robot capsule from all q/dq samples in the complete monitored
  // trajectory. The returned vector is reused for every time interval.
  virtual bool calculateTrajectoryAlpha(
      const std::vector<JointPredictionSample>& trajectory,
      std::vector<double>* alpha_i) const = 0;

  // Signed distance between a robot reachable occupancy and the human-center
  // interval capsule.  Implementations use the same ReachLib capsule distance
  // routine as SaRA-Shield.
  virtual double minimumSignedDistance(
      const std::vector<RobotReachCapsule>& robot_capsules,
      const Vector3d& human_center_start,
      const Vector3d& human_center_end,
      double human_radius,
      int* closest_robot_link_index = nullptr) const = 0;

  virtual double secureRadius() const = 0;
  virtual const char* backendName() const = 0;
};

// Build a provider from an unmodified SaRA robot-parameter YAML file.  The
// configured secure_radius is overridden explicitly so it can be calibrated
// without forking SaRA's robot geometry file.
std::shared_ptr<const RobotReachabilityProvider>
makeSaraRobotReachabilityProvider(
    const std::string& robot_config_path,
    double secure_radius);

// Installed copy of SaRA-Shield's unmodified Panda robot parameters. This
// avoids source-tree-specific paths on both the simulator and the real robot.
std::string defaultSaraPandaRobotConfigPath();

struct MonitorResult {
  bool monitored_contact_possible{false};
  // Current measured collision geometry overlaps the human workspace. This
  // is also the runtime activation gate for Eq. (14) stiffness scaling.
  bool contact_relevant_for_energy{false};
  bool monitored_unsafe{false};
  // Predicted collision energy exceeds the configured budget, or a predicted
  // joint-limit violation was found. A collision possibility that remains
  // within the energy budget does not trigger candidate rejection.
  bool predicted_trigger{false};

  // SaRA-style index of the first monitored time interval that makes the
  // candidate unsafe. This indexes [prediction_trace[i],
  // prediction_trace[i + 1]]. It remains -1 for a verified trajectory.
  int collision_interval_index{-1};
  // First interval whose robot reachable occupancy intersects the human
  // workspace, regardless of whether its energy is within the PFL budget.
  // This is diagnostic metadata and does not affect candidate acceptance.
  int first_contact_interval_index{-1};
  // First geometrically intersecting interval whose energy exceeds the PFL
  // budget. Unlike collision_interval_index, this never denotes a joint-limit
  // or invalid-rollout failure.
  int first_energy_unsafe_contact_interval_index{-1};

  double workspace_distance_now{0.0};
  double workspace_distance_min{0.0};
  int current_robot_link_index{-1};
  int worst_case_robot_link_index{-1};
  double robot_secure_radius{0.0};
  bool robot_reach_alpha_valid{false};
  Vector7d robot_reach_alpha{Vector7d::Zero()};

  double worst_case_contact_time{0.0};
  double worst_case_workspace_distance_at_candidate{0.0};
  // Cartesian projection retained for diagnosis at the worst monitored
  // contact sample. Candidate gating uses joint kinetic + Cartesian and
  // nullspace potential energy, following Lachner et al. Eq. (12).
  double worst_case_cartesian_kinetic_energy_ub{0.0};
  double worst_case_cartesian_potential_energy_ub{0.0};
  double worst_case_nullspace_potential_energy_ub{0.0};
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
  double current_nullspace_potential_energy{0.0};
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
  // Direct one-sided energy-model error bounds. These affect energy
  // verification only; they do not enlarge the robot reachable occupancy:
  // K_real <= K_pred + kinetic_energy_error_bound_joule,
  // Vx_real <= Vx_pred + potential_energy_error_bound_joule, and
  // Vn_real <= Vn_pred + nullspace_potential_energy_error_bound_joule.
  double kinetic_energy_error_bound_joule{0.0};
  double potential_energy_error_bound_joule{0.0};
  double nullspace_potential_energy_error_bound_joule{0.0};
  // When set, all robot collision geometry is produced from the predicted
  // joint-state intervals by SaRA RobotArmReach.  Tracking/model uncertainty
  // is already contained once in its secure_radius.
  std::shared_ptr<const RobotReachabilityProvider>
      robot_reachability_provider;
  double ee_collision_radius{0.04};
  double tracking_acc_error_bound{0.2};
  bool use_dynamic_consistent_impedance{true};
  Vector7d nullspace_reference{Vector7d::Zero()};
  // Effective nominal stiffness. A value of zero means nullspace control is
  // globally disabled for the complete intended and fail-safe execution.
  double nullspace_stiffness{0.0};
  // Actual (possibly energy-scaled) stiffness that produced the measured
  // current state. Future rollout samples use nullspace_stiffness.
  double current_nullspace_stiffness{0.0};
  // Model the same contact-gated Eq. (14) stiffness adaptation used by the
  // runtime controller. Each rollout state evaluates its own workspace
  // overlap before selecting the gains for the following interval.
  bool enable_runtime_energy_scaling{false};
  // Contact-gated Eqs. (16)-(17) emergency joint stabilization state and
  // parameters. Each rollout state uses the same current-overlap gate as the
  // runtime controller.
  bool enable_overbudget_joint_stabilization{false};
  double overbudget_joint_stiffness{1.0};
  double overbudget_joint_scale_omega{40.0};
  OverbudgetJointStabilizationState overbudget_joint_state;
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

// Cartesian-only fallback for configurations without a joint dynamics
// provider. It fails closed for every configured joint-space energy feature
// because a Cartesian state cannot represent complete joint kinetic energy,
// U_q, or the Eq. (16)-(17) q_star stabilizer.
MonitorResult verifyReachablePlan(const VerifiedPlan& plan,
                                  const Vector3d& current_position,
                                  const Quaterniond& current_orientation,
                                  const Vector6d& ee_twist,
                                  const Matrix7d& inertia,
                                  const Matrix67d& J_geo,
                                  const SafetyMonitorConfig& config);

}  // namespace cps_safety_monitor
