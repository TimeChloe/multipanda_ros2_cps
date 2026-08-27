#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "cps_safety_monitor/reachable_safety_monitor.hpp"

namespace cps_safety_monitor {
namespace {

class IdentityJointDynamicsProvider final : public JointDynamicsProvider {
 public:
  bool evaluate(const Vector7d& q,
                const Vector7d& dq,
                JointDynamicsSample* sample) const override {
    ++evaluate_count_;
    (void)q;
    (void)dq;
    if (sample == nullptr) {
      return false;
    }
    sample->valid = true;
    sample->control_position = Vector3d::Zero();
    sample->control_orientation = Quaterniond::Identity();
    sample->control_jacobian.setZero();
    sample->control_jacobian.leftCols<6>() = Matrix6d::Identity();
    sample->control_jdot_dq.setZero();
    sample->inertia = inertia_scale_ * Matrix7d::Identity();
    sample->coriolis.setZero();
    return true;
  }

  JointDynamicsLimits limits() const override { return limits_; }

  JointDynamicsLimits limits_;
  double inertia_scale_{1.0};
  mutable int evaluate_count_{0};
};

TEST(ReachableSafetyMonitor, RejectsTangentialAndRotationalCartesianEnergy) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.12;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.p = Vector3d(0.141, 0.0, 0.0);
  plan.anchor.q = Quaterniond(Eigen::AngleAxisd(0.2, Vector3d::UnitZ()));

  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  sample.q = Quaterniond::Identity();
  sample.K.bottomRightCorner<3, 3>() = 10.0 * Matrix3d::Identity();
  plan.intended.push_back(sample);

  Vector6d twist = Vector6d::Zero();
  twist.head<3>() = Vector3d(-0.2, 1.0, 0.0);

  Matrix67d jacobian = Matrix67d::Zero();
  jacobian.leftCols<6>() = Matrix6d::Identity();

  const MonitorResult result = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      twist,
      Matrix7d::Identity(),
      jacobian,
      config);

  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_GT(result.workspace_distance_now, 0.0);
  EXPECT_FALSE(result.contact_relevant_for_energy);
  EXPECT_GT(result.worst_case_cartesian_kinetic_energy_ub,
            config.energy_budget_joule);
  EXPECT_GT(result.worst_case_cartesian_potential_energy_ub, 0.1);
  EXPECT_NEAR(result.worst_case_cartesian_control_energy_ub,
              result.worst_case_cartesian_kinetic_energy_ub +
                  result.worst_case_cartesian_potential_energy_ub,
              1.0e-9);
  EXPECT_TRUE(result.predicted_trigger);
}

TEST(ReachableSafetyMonitor,
     CollisionPossibleWithinCartesianBudgetDoesNotTrigger) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.01;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.p = Vector3d(0.141, 0.0, 0.0);
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.2;
  plan.intended.push_back(sample);

  Vector6d twist = Vector6d::Zero();
  twist.x() = -0.01;
  Matrix67d jacobian = Matrix67d::Zero();
  jacobian.leftCols<6>() = Matrix6d::Identity();

  const MonitorResult result = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      twist,
      Matrix7d::Identity(),
      jacobian,
      config);

  EXPECT_GT(result.workspace_distance_now, 0.0);
  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_FALSE(result.contact_relevant_for_energy);
  EXPECT_GT(result.worst_case_cartesian_control_energy_ub, 0.0);
  EXPECT_LT(result.worst_case_cartesian_control_energy_ub,
            config.energy_budget_joule);
  EXPECT_FALSE(result.collision_energy_unsafe);
  EXPECT_FALSE(result.predicted_trigger);
  EXPECT_FALSE(result.monitored_unsafe);
}

TEST(ReachableSafetyMonitor,
     CurrentWorkspaceMembershipDoesNotBypassCartesianPrediction) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.01;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.p = Vector3d(0.13, 0.0, 0.0);
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  plan.intended.push_back(sample);

  Vector6d twist = Vector6d::Zero();
  twist.x() = 1.0;
  Matrix67d jacobian = Matrix67d::Zero();
  jacobian.leftCols<6>() = Matrix6d::Identity();

  const MonitorResult result = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      twist,
      Matrix7d::Identity(),
      jacobian,
      config);

  EXPECT_TRUE(result.contact_relevant_for_energy);
  EXPECT_LT(result.workspace_distance_now, 0.0);
  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_GT(result.worst_case_cartesian_control_energy_ub,
            config.energy_budget_joule);
  EXPECT_TRUE(result.predicted_trigger);
}

TEST(ReachableSafetyMonitor,
     CurrentWorkspaceMembershipDoesNotBypassJointPrediction) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.01;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  plan.intended.push_back(sample);

  Vector7d dq = Vector7d::Zero();
  dq(0) = 1.0;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      dq,
      dynamics,
      config);

  EXPECT_TRUE(result.contact_relevant_for_energy);
  EXPECT_LT(result.workspace_distance_now, 0.0);
  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_GT(result.worst_case_total_control_energy_ub,
            config.energy_budget_joule);
  EXPECT_TRUE(result.predicted_trigger);
}

TEST(ReachableSafetyMonitor,
     CollisionPossibleWithinJointEnergyBudgetDoesNotTrigger) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.01;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  plan.intended.push_back(sample);

  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_LE(result.worst_case_total_control_energy_ub,
            config.energy_budget_joule);
  EXPECT_FALSE(result.collision_energy_unsafe);
  EXPECT_FALSE(result.joint_limit_unsafe);
  EXPECT_FALSE(result.predicted_trigger);
  EXPECT_FALSE(result.monitored_unsafe);
}

TEST(ReachableSafetyMonitor, UsesExecutedCommandForCurrentPotentialEnergy) {
  SafetyMonitorConfig config;
  config.current_energy_reference_valid = true;
  config.current_energy_reference.p = Vector3d::Zero();
  config.current_energy_reference.q = Quaterniond::Identity();
  config.current_energy_reference.K(0, 0) = 100.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.p = Vector3d(0.1, 0.0, 0.0);
  plan.anchor.q = Quaterniond::Identity();

  Matrix67d jacobian = Matrix67d::Zero();
  jacobian.leftCols<6>() = Matrix6d::Identity();

  const MonitorResult result = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      Vector6d::Zero(),
      Matrix7d::Identity(),
      jacobian,
      config);

  EXPECT_TRUE(result.current_cartesian_energy_valid);
  EXPECT_NEAR(result.current_cartesian_kinetic_energy, 0.0, 1.0e-12);
  EXPECT_NEAR(result.current_cartesian_potential_energy, 0.5, 1.0e-12);
  EXPECT_NEAR(result.current_cartesian_control_energy, 0.5, 1.0e-12);
}

TEST(ReachableSafetyMonitor, JointEnergyIncludesNullspaceMotion) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.current_energy_reference_valid = true;
  config.current_energy_reference.q = Quaterniond::Identity();

  VerifiedPlan plan;
  plan.valid = true;
  Vector7d q = Vector7d::Zero();
  Vector7d dq = Vector7d::Zero();
  dq(6) = 1.0;  // Pure Jacobian-nullspace motion in this mock model.

  const MonitorResult result = verifyReachablePlanJointSpace(
      plan, q, dq, dynamics, config);

  ASSERT_TRUE(result.current_joint_energy_valid);
  EXPECT_NEAR(result.current_joint_kinetic_energy, 0.5, 1.0e-12);
  EXPECT_NEAR(result.current_cartesian_kinetic_energy, 0.0, 1.0e-12);
  EXPECT_NEAR(result.current_total_control_energy, 0.5, 1.0e-12);
}

TEST(ReachableSafetyMonitor, UsesLiveDynamicsForInitialJointState) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.current_joint_dynamics_valid = true;
  config.current_joint_dynamics.valid = true;
  config.current_joint_dynamics.control_orientation =
      Quaterniond::Identity();
  config.current_joint_dynamics.control_jacobian.setZero();
  config.current_joint_dynamics.control_jacobian.leftCols<6>() =
      Matrix6d::Identity();
  config.current_joint_dynamics.inertia = 2.0 * Matrix7d::Identity();
  config.current_energy_reference_valid = true;
  config.current_energy_reference.q = Quaterniond::Identity();

  VerifiedPlan plan;
  plan.valid = true;
  const Vector7d dq = Vector7d::Ones();
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan, Vector7d::Zero(), dq, dynamics, config);

  EXPECT_EQ(dynamics.evaluate_count_, 0);
  EXPECT_TRUE(result.current_joint_energy_valid);
  EXPECT_NEAR(result.current_joint_kinetic_energy, 7.0, 1.0e-12);
}

TEST(ReachableSafetyMonitor, DisablesNullspaceTorqueDuringFailsafeWhenRequested) {
  IdentityJointDynamicsProvider dynamics;

  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d(10.0, 0.0, 0.0);
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.nullspace_reference(6) = 1.0;
  config.nullspace_stiffness = 100.0;
  config.disable_nullspace_in_failsafe = true;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample failsafe = plan.anchor;
  failsafe.t = 0.01;
  failsafe.q = Quaterniond::Identity();
  failsafe.failsafe = true;
  plan.failsafe.push_back(failsafe);

  std::vector<JointPredictionSample> prediction_trace;
  const MonitorResult disabled_result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_FALSE(prediction_trace.empty());
  EXPECT_NEAR(prediction_trace.back().dq(6), 0.0, 1.0e-12);
  EXPECT_FALSE(disabled_result.joint_limit_unsafe);

  config.disable_nullspace_in_failsafe = false;
  prediction_trace.clear();
  verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_FALSE(prediction_trace.empty());
  EXPECT_GT(prediction_trace.back().dq(6), 0.0);
}

TEST(ReachableSafetyMonitor, JointRolloutDetectsPositionLimit) {
  IdentityJointDynamicsProvider dynamics;
  dynamics.limits_.position_upper(0) = 1.0e-4;

  SafetyMonitorConfig config;
  config.tracking_acc_error_bound = 0.0;
  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  sample.ddp.x() = 10.0;
  sample.q = Quaterniond::Identity();
  plan.intended.push_back(sample);

  std::vector<JointPredictionSample> prediction_trace;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  EXPECT_TRUE(result.joint_limit_unsafe);
  EXPECT_EQ(result.joint_limit_index, 0);
  EXPECT_GT(result.joint_position_violation, 0.0);
  EXPECT_TRUE(result.predicted_trigger);
  ASSERT_GE(prediction_trace.size(), 2U);
  EXPECT_NEAR(prediction_trace.front().t, plan.anchor.t, 1.0e-12);
  EXPECT_TRUE(prediction_trace.front().q.isZero(1.0e-12));
  EXPECT_TRUE(prediction_trace.front().dq.isZero(1.0e-12));
  EXPECT_NEAR(prediction_trace.back().t, sample.t, 1.0e-12);
  EXPECT_GT(prediction_trace.back().q(0), 0.0);
  EXPECT_GT(prediction_trace.back().dq(0), 0.0);
}

TEST(ReachableSafetyMonitor, ContactIntervalUsesMaximumEndpointEnergy) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.10;
  config.tracking_acc_error_bound = 0.0;
  config.joint_rollout_max_dt = 0.001;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample braking_sample = plan.anchor;
  braking_sample.t = 0.001;
  braking_sample.D(0, 0) = 1000.0;
  plan.intended.push_back(braking_sample);

  Vector7d dq = Vector7d::Zero();
  dq(0) = 1.0;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      dq,
      dynamics,
      config);

  ASSERT_TRUE(result.monitored_contact_possible);
  EXPECT_NEAR(result.worst_case_contact_time, plan.anchor.t, 1.0e-12);
  EXPECT_NEAR(result.worst_case_joint_kinetic_energy_ub, 0.5, 1.0e-12);
  EXPECT_NEAR(result.worst_case_total_control_energy_ub, 0.5, 1.0e-12);
  EXPECT_TRUE(result.predicted_trigger);
}

TEST(ReachableSafetyMonitor,
     FixedTrackingPoseBoundsInflateContactButNotPotentialEnergy) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.collision_center_offset = Vector3d(0.10, 0.0, 0.0);
  config.energy_budget_joule = 0.05;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.p = Vector3d(0.151, 0.0, 0.0);
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  sample.K(0, 0) = 100.0;
  sample.K(3, 3) = 10.0;
  plan.intended.push_back(sample);

  Matrix67d jacobian = Matrix67d::Zero();
  jacobian.leftCols<6>() = Matrix6d::Identity();
  const MonitorResult nominal = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      Vector6d::Zero(),
      Matrix7d::Identity(),
      jacobian,
      config);

  EXPECT_GT(nominal.workspace_distance_now, 0.0);
  EXPECT_FALSE(nominal.monitored_contact_possible);

  config.tracking_position_error_bound = 0.012;
  config.tracking_orientation_error_bound = 0.10;
  const MonitorResult bounded = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      Vector6d::Zero(),
      Matrix7d::Identity(),
      jacobian,
      config);

  // The current measured geometry remains nominal; only future reachable
  // states are enlarged by the certified tracking bounds.
  EXPECT_NEAR(bounded.workspace_distance_now,
              nominal.workspace_distance_now,
              1.0e-12);
  EXPECT_TRUE(bounded.monitored_contact_possible);
  EXPECT_NEAR(bounded.worst_case_pos_error_radius, 0.012, 1.0e-12);
  EXPECT_NEAR(bounded.worst_case_orientation_error_radius, 0.10, 1.0e-12);
  EXPECT_NEAR(bounded.worst_case_cartesian_potential_energy_ub,
              0.0,
              1.0e-12);
  EXPECT_FALSE(bounded.predicted_trigger);

  config.potential_energy_error_bound_joule = 0.06;
  const MonitorResult energy_bounded = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      Vector6d::Zero(),
      Matrix7d::Identity(),
      jacobian,
      config);
  EXPECT_NEAR(energy_bounded.worst_case_cartesian_potential_energy_ub,
              0.06,
              1.0e-12);
  EXPECT_TRUE(energy_bounded.predicted_trigger);
}

TEST(ReachableSafetyMonitor,
     DirectKineticEnergyErrorBoundAppliesToFutureEndpoint) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  workspace_parameters.hand_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.05;
  config.kinetic_energy_error_bound_joule = 0.06;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.001;
  plan.intended.push_back(sample);

  std::vector<JointPredictionSample> prediction_trace;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_NEAR(result.worst_case_joint_kinetic_energy_ub, 0.06, 1.0e-12);
  EXPECT_NEAR(result.worst_case_total_control_energy_ub, 0.06, 1.0e-12);
  EXPECT_TRUE(result.predicted_trigger);
  ASSERT_EQ(prediction_trace.size(), 2U);
  EXPECT_TRUE(prediction_trace.back().energy_valid);
  EXPECT_NEAR(prediction_trace.back().joint_kinetic_energy, 0.0, 1.0e-12);
  EXPECT_NEAR(prediction_trace.back().cartesian_potential_energy,
              0.0,
              1.0e-12);
}

TEST(ReachableSafetyMonitor,
     ComparesRuntimeAndPredictionInertiaAtSameMeasuredState) {
  IdentityJointDynamicsProvider dynamics;
  dynamics.inertia_scale_ = 1.0;

  SafetyMonitorConfig config;
  config.enable_inertia_model_comparison = true;
  config.current_joint_dynamics_valid = true;
  config.current_joint_dynamics.valid = true;
  config.current_joint_dynamics.control_orientation = Quaterniond::Identity();
  config.current_joint_dynamics.inertia =
      2.0 * Matrix7d::Identity();
  config.current_joint_dynamics.control_jacobian.setZero();
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.001;
  plan.intended.push_back(sample);

  Vector7d dq = Vector7d::Zero();
  dq(0) = 1.0;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      dq,
      dynamics,
      config);

  ASSERT_TRUE(result.inertia_model_comparison_valid);
  EXPECT_NEAR(result.runtime_model_joint_kinetic_energy, 1.0, 1.0e-12);
  EXPECT_NEAR(result.prediction_model_joint_kinetic_energy, 0.5, 1.0e-12);
  EXPECT_NEAR(result.inertia_model_kinetic_energy_error, 0.5, 1.0e-12);
  EXPECT_NEAR(result.inertia_model_difference_frobenius_norm,
              std::sqrt(7.0),
              1.0e-12);
  EXPECT_NEAR(result.inertia_model_difference_relative_frobenius_norm,
              0.5,
              1.0e-12);
  EXPECT_NEAR(result.inertia_model_difference_max_abs, 1.0, 1.0e-12);
  EXPECT_EQ(result.inertia_model_difference_max_abs_row,
            result.inertia_model_difference_max_abs_col);
  ASSERT_TRUE(result.inertia_model_energy_ratio_valid);
  EXPECT_NEAR(result.inertia_model_min_energy_ratio, 2.0, 1.0e-12);
  EXPECT_NEAR(result.inertia_model_max_energy_ratio, 2.0, 1.0e-12);
}

}  // namespace
}  // namespace cps_safety_monitor
