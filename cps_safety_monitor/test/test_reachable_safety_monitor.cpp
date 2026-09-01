#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

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
    sample->control_position = control_position_;
    sample->control_orientation = control_orientation_;
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
  Vector3d control_position_{Vector3d::Zero()};
  Quaterniond control_orientation_{Quaterniond::Identity()};
  mutable int evaluate_count_{0};
};

class RecordingRobotReachabilityProvider final
    : public RobotReachabilityProvider {
 public:
  bool reachInterval(
      const Vector7d& start_q,
      const Vector7d& goal_q,
      double interval_duration_sec,
      const std::vector<double>& alpha_i,
      std::vector<RobotReachCapsule>* capsules) const override {
    if (capsules == nullptr || !start_q.allFinite() || !goal_q.allFinite() ||
        !std::isfinite(interval_duration_sec) || alpha_i.size() != 7) {
      return false;
    }
    const bool dynamic = std::any_of(
        alpha_i.begin(), alpha_i.end(), [](double value) {
          return value > 0.0;
        });
    if (dynamic) {
      ++dynamic_reach_count_;
      last_dynamic_alpha_ = alpha_i;
    } else {
      ++static_reach_count_;
    }
    capsules->assign(1, RobotReachCapsule{});
    return true;
  }

  bool calculateTrajectoryAlpha(
      const std::vector<JointPredictionSample>& trajectory,
      std::vector<double>* alpha_i) const override {
    alpha_trajectory_ = trajectory;
    if (!alpha_success_ || alpha_i == nullptr) {
      return false;
    }
    alpha_i->assign(7, dynamic_alpha_value_);
    return true;
  }

  double minimumSignedDistance(
      const std::vector<RobotReachCapsule>&,
      const Vector3d&,
      const Vector3d&,
      double,
      int* closest_robot_link_index) const override {
    if (closest_robot_link_index != nullptr) {
      *closest_robot_link_index = 0;
    }
    return 1.0;
  }

  double secureRadius() const override { return 0.02; }
  const char* backendName() const override { return "recording"; }

  bool alpha_success_{true};
  double dynamic_alpha_value_{2.5};
  mutable int static_reach_count_{0};
  mutable int dynamic_reach_count_{0};
  mutable std::vector<double> last_dynamic_alpha_;
  mutable std::vector<JointPredictionSample> alpha_trajectory_;
};

TEST(ReachableSafetyMonitor, RejectsTangentialAndRotationalCartesianEnergy) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
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
  EXPECT_EQ(result.collision_interval_index, 0);
  EXPECT_EQ(result.first_contact_interval_index, 0);
  EXPECT_EQ(result.first_energy_unsafe_contact_interval_index, 0);
}

TEST(ReachableSafetyMonitor,
     CollisionPossibleWithinCartesianBudgetDoesNotTrigger) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
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
  EXPECT_EQ(result.collision_interval_index, -1);
  EXPECT_EQ(result.first_contact_interval_index, 0);
  EXPECT_EQ(result.first_energy_unsafe_contact_interval_index, -1);
}

TEST(ReachableSafetyMonitor,
     CurrentWorkspaceMembershipDoesNotBypassCartesianPrediction) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
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
  EXPECT_EQ(result.collision_interval_index, 0);
}

TEST(ReachableSafetyMonitor,
     CollisionPossibleWithinJointEnergyBudgetDoesNotTrigger) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
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
  EXPECT_EQ(result.first_contact_interval_index, 0);
  EXPECT_EQ(result.first_energy_unsafe_contact_interval_index, -1);
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
  EXPECT_GE(result.collision_interval_index, 0);
  EXPECT_EQ(result.first_energy_unsafe_contact_interval_index, -1);
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
  EXPECT_EQ(result.collision_interval_index, 0);
  EXPECT_EQ(result.first_energy_unsafe_contact_interval_index, 0);
}

TEST(ReachableSafetyMonitor, ReportsFirstUnsafePredictionTraceInterval) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.001;
  config.tracking_acc_error_bound = 0.0;
  config.joint_rollout_max_dt = 0.001;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample safe_interval_end = plan.anchor;
  safe_interval_end.t = 0.001;
  plan.intended.push_back(safe_interval_end);
  ImpedanceSample unsafe_interval_end = safe_interval_end;
  unsafe_interval_end.t = 0.002;
  unsafe_interval_end.p.x() = 0.10;
  unsafe_interval_end.K(0, 0) = 1.0;
  plan.intended.push_back(unsafe_interval_end);

  std::vector<JointPredictionSample> prediction_trace;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 3U);
  EXPECT_TRUE(result.predicted_trigger);
  EXPECT_EQ(result.collision_interval_index, 1);
  EXPECT_EQ(result.first_contact_interval_index, 0);
  EXPECT_EQ(result.first_energy_unsafe_contact_interval_index, 1);
}

TEST(ReachableSafetyMonitor,
     DirectPotentialEnergyErrorBoundAppliesToFutureEndpoint) {
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d(0.151, 0.0, 0.0);
  workspace_parameters.motion_radius = 0.10;
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
  config.potential_energy_error_bound_joule = 0.06;
  const MonitorResult result = verifyReachablePlan(
      plan,
      plan.anchor.p,
      plan.anchor.q,
      Vector6d::Zero(),
      Matrix7d::Identity(),
      jacobian,
      config);

  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_NEAR(result.worst_case_cartesian_potential_energy_ub,
              0.06,
              1.0e-12);
  EXPECT_TRUE(result.predicted_trigger);
}

TEST(ReachableSafetyMonitor,
     DirectKineticEnergyErrorBoundAppliesToFutureEndpoint) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
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
     JointSpaceDirectEnergyBoundsApplyWithoutPoseParameters) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 1.0;
  config.tracking_acc_error_bound = 0.0;
  config.kinetic_energy_error_bound_joule = 0.02;
  config.potential_energy_error_bound_joule = 0.03;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.001;
  plan.intended.push_back(sample);

  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_NEAR(result.worst_case_joint_kinetic_energy_ub, 0.02, 1.0e-12);
  EXPECT_NEAR(
      result.worst_case_cartesian_potential_energy_ub, 0.03, 1.0e-12);
  EXPECT_NEAR(result.worst_case_total_control_energy_ub, 0.05, 1.0e-12);
  EXPECT_FALSE(result.predicted_trigger);
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

TEST(ReachableSafetyMonitor,
     UsesSaraRobotArmReachForPredictedJointIntervals) {
  const auto robot_reachability = makeSaraRobotReachabilityProvider(
      SARA_PANDA_CONFIG_PATH,
      0.02);
  ASSERT_NE(robot_reachability, nullptr);
  EXPECT_STREQ(robot_reachability->backendName(), "sara_robot_arm_reach");

  std::vector<RobotReachCapsule> capsules;
  const std::vector<double> zero_alpha(7, 0.0);
  ASSERT_TRUE(robot_reachability->reachInterval(
      Vector7d::Zero(),
      Vector7d::Zero(),
      0.001,
      zero_alpha,
      &capsules));
  ASSERT_EQ(capsules.size(), 7U);

  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.robot_reachability_provider = robot_reachability;
  config.tracking_acc_error_bound = 100.0;
  config.energy_budget_joule = 1.0;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = capsules.front().p1;
  workspace_parameters.motion_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);

  VerifiedPlan plan;
  plan.valid = true;
  ImpedanceSample sample;
  sample.t = 0.001;
  sample.q = Quaterniond::Identity();
  plan.intended.push_back(sample);

  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_LE(result.workspace_distance_now, 0.0);
  EXPECT_NEAR(result.robot_secure_radius, 0.02, 1.0e-12);
  EXPECT_TRUE(result.robot_reach_alpha_valid);
  EXPECT_TRUE(result.robot_reach_alpha.isZero(1.0e-12));
  EXPECT_GE(result.current_robot_link_index, 0);
  EXPECT_EQ(result.first_contact_interval_index, 0);
  // The legacy Cartesian tracking tube is bypassed when SaRA is present;
  // SaRA's secure radius is the sole geometric uncertainty inflation.
  EXPECT_NEAR(result.worst_case_pos_error_radius, 0.0, 1.0e-12);
  EXPECT_NEAR(result.worst_case_orientation_error_radius, 0.0, 1.0e-12);
  EXPECT_NEAR(result.worst_case_vel_error_radius, 0.0, 1.0e-12);
}

TEST(ReachableSafetyMonitor,
     ComputesDynamicAlphaFromCompleteIntendedAndFailsafeRollout) {
  IdentityJointDynamicsProvider dynamics;
  auto robot_reachability =
      std::make_shared<RecordingRobotReachabilityProvider>();
  SafetyMonitorConfig config;
  config.robot_reachability_provider = robot_reachability;
  config.joint_rollout_max_dt = 0.001;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample intended = plan.anchor;
  intended.t = 0.001;
  plan.intended.push_back(intended);
  ImpedanceSample failsafe = intended;
  failsafe.t = 0.002;
  failsafe.failsafe = true;
  plan.failsafe.push_back(failsafe);

  std::vector<JointPredictionSample> prediction_trace;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 3U);
  ASSERT_EQ(robot_reachability->alpha_trajectory_.size(), 3U);
  EXPECT_EQ(robot_reachability->dynamic_reach_count_, 2);
  ASSERT_EQ(robot_reachability->last_dynamic_alpha_.size(), 7U);
  EXPECT_TRUE(result.robot_reach_alpha_valid);
  EXPECT_TRUE(result.robot_reach_alpha.isConstant(2.5));
  EXPECT_FALSE(result.monitored_unsafe);
}

TEST(ReachableSafetyMonitor, DynamicAlphaFailureFailsClosed) {
  IdentityJointDynamicsProvider dynamics;
  auto robot_reachability =
      std::make_shared<RecordingRobotReachabilityProvider>();
  robot_reachability->alpha_success_ = false;
  SafetyMonitorConfig config;
  config.robot_reachability_provider = robot_reachability;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.001;
  plan.intended.push_back(sample);

  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  EXPECT_FALSE(result.robot_reach_alpha_valid);
  EXPECT_TRUE(result.joint_limit_unsafe);
  EXPECT_TRUE(result.predicted_trigger);
  EXPECT_TRUE(result.monitored_unsafe);
  EXPECT_EQ(result.collision_interval_index, 0);
  EXPECT_EQ(robot_reachability->dynamic_reach_count_, 0);
}

TEST(ReachableSafetyMonitor, SaraDynamicAlphaRejectsInvalidSampleTimes) {
  const auto robot_reachability = makeSaraRobotReachabilityProvider(
      SARA_PANDA_CONFIG_PATH,
      0.02);
  std::vector<JointPredictionSample> trajectory(2);
  trajectory[0].t = 0.001;
  trajectory[1].t = 0.001;
  std::vector<double> alpha_i(7, 123.0);

  EXPECT_FALSE(robot_reachability->calculateTrajectoryAlpha(
      trajectory, &alpha_i));
  EXPECT_TRUE(alpha_i.empty());
}

TEST(ReachableSafetyMonitor,
     AssumeClearStillRunsJointAndEnergyPrediction) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 1.0;
  config.human_workspace.setParameters(workspace_parameters);
  config.assume_human_workspace_clear = true;
  config.energy_budget_joule = 0.01;
  config.tracking_acc_error_bound = 0.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.001;
  sample.q = Quaterniond::Identity();
  plan.intended.push_back(sample);

  Vector7d dq = Vector7d::Zero();
  dq(0) = 1.0;
  std::vector<JointPredictionSample> prediction_trace;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      dq,
      dynamics,
      config,
      &prediction_trace);

  EXPECT_TRUE(std::isinf(result.workspace_distance_now));
  EXPECT_GT(result.workspace_distance_now, 0.0);
  EXPECT_TRUE(std::isinf(result.workspace_distance_min));
  EXPECT_FALSE(result.monitored_contact_possible);
  EXPECT_FALSE(result.contact_relevant_for_energy);
  EXPECT_FALSE(result.collision_energy_unsafe);
  EXPECT_FALSE(result.predicted_trigger);
  ASSERT_EQ(prediction_trace.size(), 2U);
  EXPECT_TRUE(prediction_trace.back().energy_valid);
  EXPECT_GT(prediction_trace.back().joint_kinetic_energy, 0.0);
  EXPECT_GT(dynamics.evaluate_count_, 0);
}

}  // namespace
}  // namespace cps_safety_monitor
