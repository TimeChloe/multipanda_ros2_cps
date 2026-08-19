#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "cps_safety_monitor/reachable_safety_monitor.hpp"

namespace cps_safety_monitor {
namespace {

class IdentityJointDynamicsProvider final : public JointDynamicsProvider {
 public:
  bool evaluate(const Vector7d& q,
                const Vector7d& dq,
                JointDynamicsSample* sample) const override {
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
    sample->inertia = Matrix7d::Identity();
    sample->coriolis.setZero();
    return true;
  }

  JointDynamicsLimits limits() const override { return limits_; }

  JointDynamicsLimits limits_;
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
  config.energy_budget_margin_joule = 0.0;
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
  EXPECT_GT(result.worst_case_cartesian_kinetic_energy_ub,
            config.energy_budget_joule);
  EXPECT_GT(result.worst_case_cartesian_potential_energy_ub, 0.1);
  EXPECT_NEAR(result.worst_case_cartesian_control_energy_ub,
              result.worst_case_cartesian_kinetic_energy_ub +
                  result.worst_case_cartesian_potential_energy_ub,
              1.0e-9);
  EXPECT_TRUE(result.predicted_trigger);
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

}  // namespace
}  // namespace cps_safety_monitor
