#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <Eigen/Geometry>

#include "cps_safety_monitor/reachable_safety_monitor.hpp"
#include "reach_lib.hpp"

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
    if (control_position_from_q0_) {
      sample->control_position.x() = q(0);
    }
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
  bool control_position_from_q0_{false};
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
      const Vector3d& human_center_start,
      const Vector3d& human_center_end,
      double human_radius,
      int* closest_robot_link_index) const override {
    last_human_center_start_ = human_center_start;
    last_human_center_end_ = human_center_end;
    maximum_human_radius_ = std::max(maximum_human_radius_, human_radius);
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
  mutable Vector3d last_human_center_start_{Vector3d::Zero()};
  mutable Vector3d last_human_center_end_{Vector3d::Zero()};
  mutable double maximum_human_radius_{0.0};
};

TEST(EnergyBudgetStiffnessScale, ImplementsLachnerEquation14) {
  EXPECT_NEAR(
      energyBudgetStiffnessScale(0.1, 0.2, 0.3, 1.0),
      1.0,
      1.0e-12);
  EXPECT_NEAR(
      energyBudgetStiffnessScale(0.2, 0.3, 0.5, 0.6),
      0.5,
      1.0e-12);
  EXPECT_NEAR(
      energyBudgetStiffnessScale(0.7, 0.1, 0.2, 0.6),
      0.0,
      1.0e-12);
}

TEST(OverbudgetJointStabilization, ImplementsLachnerEquations16And17) {
  OverbudgetJointStabilizationState state;
  const Vector7d q_capture = Vector7d::Zero();
  const auto capture = updateOverbudgetJointStabilization(
      q_capture, 0.7, 0.6, 1.0, 40.0, true, &state);

  ASSERT_TRUE(state.active);
  EXPECT_TRUE(capture.active);
  EXPECT_NEAR((state.reference - q_capture).norm(), 0.0, 1.0e-12);
  EXPECT_NEAR(capture.potential_energy, 0.0, 1.0e-12);
  EXPECT_NEAR(capture.scale_rho, 40.0 * 0.7 / 0.6, 1.0e-12);
  EXPECT_NEAR(capture.torque.norm(), 0.0, 1.0e-12);

  Vector7d q_displaced = q_capture;
  q_displaced(0) = 0.1;
  const auto displaced = updateOverbudgetJointStabilization(
      q_displaced, 0.7, 0.6, 1.0, 40.0, true, &state);
  EXPECT_TRUE(displaced.active);
  EXPECT_NEAR(displaced.potential_energy, 0.005, 1.0e-12);
  EXPECT_NEAR(
      displaced.scale_rho, 40.0 * 0.7 / 0.605, 1.0e-12);
  EXPECT_LT(displaced.torque(0), 0.0);

  const auto cleared = updateOverbudgetJointStabilization(
      q_displaced, 0.6, 0.6, 1.0, 40.0, true, &state);
  EXPECT_FALSE(state.active);
  EXPECT_FALSE(cleared.active);
  EXPECT_NEAR(cleared.torque.norm(), 0.0, 1.0e-12);
}

TEST(ReachableSafetyMonitor, RolloutAppliesEnergyScalingInsideCollisionArea) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.1;
  config.enable_runtime_energy_scaling = true;
  config.joint_rollout_max_dt = 0.01;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  sample.p.x() = 0.1;
  sample.K(0, 0) = 100.0;
  plan.intended.push_back(sample);

  std::vector<JointPredictionSample> prediction_trace;
  verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 2U);
  EXPECT_TRUE(prediction_trace.front().energy_scaling_active);
  EXPECT_TRUE(prediction_trace.back().energy_scaling_active);
  EXPECT_NEAR(
      prediction_trace.back().energy_stiffness_scale, 0.2, 1.0e-12);
  EXPECT_NEAR(
      prediction_trace.back().cartesian_potential_energy,
      0.1,
      1.0e-9);
}

TEST(ReachableSafetyMonitor, RolloutKeepsNominalGainsOutsideCollisionArea) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.1;
  config.enable_runtime_energy_scaling = true;
  config.joint_rollout_max_dt = 0.01;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d(10.0, 0.0, 0.0);
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  sample.p.x() = 0.1;
  sample.K(0, 0) = 100.0;
  plan.intended.push_back(sample);

  std::vector<JointPredictionSample> prediction_trace;
  verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 2U);
  EXPECT_FALSE(prediction_trace.front().energy_scaling_active);
  EXPECT_FALSE(prediction_trace.back().energy_scaling_active);
  EXPECT_NEAR(
      prediction_trace.back().energy_stiffness_scale, 1.0, 1.0e-12);
  EXPECT_NEAR(
      prediction_trace.back().cartesian_potential_energy,
      0.5,
      1.0e-9);
}

TEST(ReachableSafetyMonitor, RolloutActivatesScalingAfterEnteringCollisionArea) {
  IdentityJointDynamicsProvider dynamics;
  dynamics.control_position_from_q0_ = true;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.1;
  config.enable_runtime_energy_scaling = true;
  config.joint_rollout_max_dt = 0.01;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d(0.04025, 0.0, 0.0);
  workspace_parameters.motion_radius = 0.0;
  config.human_workspace.setParameters(workspace_parameters);

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample first = plan.anchor;
  first.t = 0.01;
  first.p.x() = 0.1;
  first.K(0, 0) = 100.0;
  plan.intended.push_back(first);
  ImpedanceSample second = first;
  second.t = 0.02;
  plan.intended.push_back(second);

  std::vector<JointPredictionSample> prediction_trace;
  verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 3U);
  EXPECT_FALSE(prediction_trace[0].energy_scaling_active);
  EXPECT_FALSE(prediction_trace[1].energy_scaling_active);
  EXPECT_NEAR(prediction_trace[1].energy_stiffness_scale, 1.0, 1.0e-12);
  EXPECT_TRUE(prediction_trace[2].energy_scaling_active);
  EXPECT_GT(prediction_trace[2].energy_stiffness_scale, 0.0);
  EXPECT_LT(prediction_trace[2].energy_stiffness_scale, 1.0);
}

TEST(ReachableSafetyMonitor, RolloutTracksOverbudgetJointStabilization) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.1;
  config.enable_overbudget_joint_stabilization = true;
  config.overbudget_joint_stiffness = 1.0;
  config.overbudget_joint_scale_omega = 40.0;
  config.joint_rollout_max_dt = 0.01;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample first = plan.anchor;
  first.t = 0.01;
  plan.intended.push_back(first);
  ImpedanceSample second = first;
  second.t = 0.02;
  plan.intended.push_back(second);

  Vector7d dq = Vector7d::Zero();
  dq(0) = 1.0;
  std::vector<JointPredictionSample> prediction_trace;
  verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      dq,
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 3U);
  EXPECT_TRUE(
      prediction_trace.front().overbudget_joint_stabilization_active);
  EXPECT_TRUE(
      prediction_trace.back().overbudget_joint_stabilization_active);
  EXPECT_GT(
      prediction_trace.back().overbudget_joint_potential_energy, 0.0);
  EXPECT_GT(prediction_trace.back().overbudget_joint_torque_norm, 0.0);
}

TEST(ReachableSafetyMonitor,
     RolloutDisablesOverbudgetJointStabilizationOutsideCollisionArea) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.1;
  config.enable_overbudget_joint_stabilization = true;
  config.overbudget_joint_stiffness = 1.0;
  config.overbudget_joint_scale_omega = 40.0;
  config.joint_rollout_max_dt = 0.01;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d(10.0, 0.0, 0.0);
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);
  config.overbudget_joint_state.active = true;
  config.overbudget_joint_state.reference = Vector7d::Ones();

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample first = plan.anchor;
  first.t = 0.01;
  plan.intended.push_back(first);
  ImpedanceSample second = first;
  second.t = 0.02;
  plan.intended.push_back(second);

  Vector7d dq = Vector7d::Zero();
  dq(0) = 1.0;
  std::vector<JointPredictionSample> prediction_trace;
  verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      dq,
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 3U);
  for (const auto& sample : prediction_trace) {
    EXPECT_FALSE(sample.overbudget_joint_stabilization_active);
    EXPECT_NEAR(sample.overbudget_joint_potential_energy, 0.0, 1.0e-12);
    EXPECT_NEAR(sample.overbudget_joint_torque_norm, 0.0, 1.0e-12);
  }
}

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

TEST(ReachableSafetyMonitor,
     CartesianFallbackFailsClosedWithConfiguredNullspaceSpring) {
  SafetyMonitorConfig config;
  config.nullspace_stiffness = 20.0;

  VerifiedPlan plan;
  plan.valid = true;

  const MonitorResult result = verifyReachablePlan(
      plan,
      Vector3d::Zero(),
      Quaterniond::Identity(),
      Vector6d::Zero(),
      Matrix7d::Identity(),
      Matrix67d::Zero(),
      config);

  EXPECT_TRUE(result.joint_limit_unsafe);
  EXPECT_TRUE(result.predicted_trigger);
  EXPECT_TRUE(result.monitored_unsafe);
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

TEST(ReachableSafetyMonitor, CurrentEnergyIncludesNullspacePotential) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.current_energy_reference_valid = true;
  config.current_energy_reference.q = Quaterniond::Identity();
  config.nullspace_reference(6) = 0.1;
  config.nullspace_stiffness = 20.0;
  config.current_nullspace_stiffness = 20.0;

  VerifiedPlan plan;
  plan.valid = true;

  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  ASSERT_TRUE(result.current_joint_energy_valid);
  EXPECT_NEAR(result.current_joint_kinetic_energy, 0.0, 1.0e-12);
  EXPECT_NEAR(
      result.current_nullspace_potential_energy, 0.1, 1.0e-12);
  EXPECT_NEAR(result.current_total_control_energy, 0.1, 1.0e-12);
}

TEST(ReachableSafetyMonitor,
     NullspacePotentialCanMakePredictedContactEnergyUnsafe) {
  IdentityJointDynamicsProvider dynamics;
  dynamics.control_position_from_q0_ = true;

  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d(0.1404, 0.0, 0.0);
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.05;
  config.tracking_acc_error_bound = 0.0;
  config.joint_rollout_max_dt = 0.01;
  config.nullspace_reference(6) = 0.1;
  config.nullspace_stiffness = 20.0;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.01;
  sample.ddp.x() = 10.0;
  plan.intended.push_back(sample);

  std::vector<JointPredictionSample> prediction_trace;
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_EQ(prediction_trace.size(), 2U);
  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_GT(result.worst_case_nullspace_potential_energy_ub, 0.09);
  EXPECT_GT(result.worst_case_total_control_energy_ub,
            config.energy_budget_joule);
  EXPECT_TRUE(result.predicted_trigger);
  EXPECT_TRUE(
      prediction_trace.back().nullspace_potential_energy_active);
  EXPECT_GT(prediction_trace.back().nullspace_potential_energy, 0.09);
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

TEST(ReachableSafetyMonitor,
     NullspaceRemainsActiveDuringContactAndFailsafe) {
  IdentityJointDynamicsProvider dynamics;

  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 100.0;
  config.nullspace_reference(6) = 1.0;
  config.nullspace_stiffness = 100.0;
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
  const MonitorResult result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config,
      &prediction_trace);

  ASSERT_FALSE(prediction_trace.empty());
  EXPECT_TRUE(result.monitored_contact_possible);
  EXPECT_GT(prediction_trace.back().dq(6), 0.0);
  EXPECT_TRUE(
      prediction_trace.back().nullspace_potential_energy_active);
  EXPECT_GT(prediction_trace.back().nullspace_potential_energy, 0.0);
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
     DirectNullspacePotentialErrorBoundFollowsGlobalSpringMode) {
  IdentityJointDynamicsProvider dynamics;
  dynamics.control_position_from_q0_ = true;

  SafetyMonitorConfig config;
  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d(0.1404, 0.0, 0.0);
  workspace_parameters.motion_radius = 0.10;
  config.human_workspace.setParameters(workspace_parameters);
  config.ee_collision_radius = 0.04;
  config.energy_budget_joule = 0.05;
  config.tracking_acc_error_bound = 0.0;
  config.joint_rollout_max_dt = 0.01;
  config.nullspace_stiffness = 1.0;
  config.nullspace_potential_energy_error_bound_joule = 0.06;

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample intended = plan.anchor;
  intended.t = 0.01;
  intended.ddp.x() = 10.0;
  plan.intended.push_back(intended);

  const MonitorResult active_result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  EXPECT_TRUE(active_result.monitored_contact_possible);
  EXPECT_GE(
      active_result.worst_case_nullspace_potential_energy_ub, 0.06);
  EXPECT_TRUE(active_result.predicted_trigger);

  plan.intended.clear();
  ImpedanceSample failsafe = plan.anchor;
  failsafe.t = 0.01;
  failsafe.ddp.x() = 10.0;
  failsafe.failsafe = true;
  plan.failsafe.push_back(failsafe);
  config.nullspace_stiffness = 0.0;

  const MonitorResult disabled_result = verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  EXPECT_TRUE(disabled_result.monitored_contact_possible);
  EXPECT_NEAR(
      disabled_result.worst_case_nullspace_potential_energy_ub,
      0.0,
      1.0e-12);
  EXPECT_FALSE(disabled_result.predicted_trigger);
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
     UsesSingleHandCombinedReachableBallForEveryRobotInterval) {
  IdentityJointDynamicsProvider dynamics;
  auto robot_reachability =
      std::make_shared<RecordingRobotReachabilityProvider>();
  SafetyMonitorConfig config;
  config.robot_reachability_provider = robot_reachability;
  config.joint_rollout_max_dt = 0.1;

  cps_human_workspace::HumanWorkspace::Parameters workspace_parameters;
  workspace_parameters.sphere_center = Vector3d::Zero();
  workspace_parameters.center_velocity.setZero();
  workspace_parameters.motion_radius = 0.01;
  workspace_parameters.hand_max_velocity = 2.0;
  workspace_parameters.hand_max_acceleration = 10.0;
  config.human_workspace.setParameters(workspace_parameters);

  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.1;
  plan.intended.push_back(sample);

  verifyReachablePlanJointSpace(
      plan,
      Vector7d::Zero(),
      Vector7d::Zero(),
      dynamics,
      config);

  // Single-point SaRA BodyPartCombined: physical 0.01 m + 0.5*a*t^2.
  EXPECT_NEAR(robot_reachability->maximum_human_radius_, 0.06, 1.0e-12);
  EXPECT_TRUE(robot_reachability->last_human_center_start_.isApprox(
      robot_reachability->last_human_center_end_, 1.0e-12));
}

TEST(ReachableSafetyMonitor,
     SingleHandCombinedReachableBallMatchesUpstreamReachLib) {
  cps_human_workspace::HumanWorkspace workspace;
  cps_human_workspace::HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d(0.4, -0.2, 0.8);
  parameters.center_velocity = Vector3d(0.7, 0.1, -0.2);
  parameters.center_motion_time_offset_sec = 3.0;
  parameters.motion_radius = 0.103;
  parameters.hand_max_velocity = 2.0;
  parameters.hand_max_acceleration = 50.0;
  parameters.measurement_error_position = 0.004;
  parameters.measurement_error_velocity = 0.1;
  parameters.measurement_delay = 0.015;
  workspace.setParameters(parameters);

  constexpr double kPredictionTime = 0.064;
  const auto generated = workspace.handReachableSetAtTime(
      parameters.center_motion_time_offset_sec + kPredictionTime);

  const reach_lib::Point position(
      parameters.sphere_center.x(),
      parameters.sphere_center.y(),
      parameters.sphere_center.z());
  const reach_lib::Point velocity(
      parameters.center_velocity.x(),
      parameters.center_velocity.y(),
      parameters.center_velocity.z());
  reach_lib::BodyPartCombined upstream(
      "hand",
      2.0 * parameters.motion_radius,
      parameters.hand_max_velocity,
      parameters.hand_max_velocity,
      parameters.hand_max_acceleration,
      parameters.hand_max_acceleration);
  upstream.update(
      {position, position},
      {velocity, velocity},
      0.0,
      kPredictionTime,
      parameters.measurement_error_position,
      parameters.measurement_error_velocity,
      parameters.measurement_delay);
  const reach_lib::Capsule expected = upstream.get_occupancy();

  EXPECT_NEAR(generated.center.x(), expected.p1_.x, 1.0e-12);
  EXPECT_NEAR(generated.center.y(), expected.p1_.y, 1.0e-12);
  EXPECT_NEAR(generated.center.z(), expected.p1_.z, 1.0e-12);
  EXPECT_NEAR(generated.radius, expected.r_, 1.0e-12);
  EXPECT_EQ(expected.p1_, expected.p2_);
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
