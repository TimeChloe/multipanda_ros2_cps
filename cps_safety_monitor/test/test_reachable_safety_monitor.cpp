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

TEST(EnergyRecovery, RemovalRetainsBudgetForLargeReferenceError) {
  EnergyRecoveryState state;
  // K=1500 N/m, e=0.2 m: nominal potential is 30 J, budget 0.12 J.
  const auto contact = updateEnergyRecovery(
      0.0, 30.0, 0.0, 0.12, 0.95, true, true, true, true, true, &state);
  EXPECT_EQ(state.phase, EnergyControlPhase::kLimited);
  EXPECT_NEAR(contact.scale, 0.004, 1e-12);
  for (int cycle = 0; cycle < 100; ++cycle) {
    const auto released = updateEnergyRecovery(
        0.0, 30.0, 0.0, 0.12, 0.95, true, true, false, true, true, &state);
    EXPECT_EQ(state.phase, EnergyControlPhase::kRecovering);
    EXPECT_NEAR(released.scale, contact.scale, 1e-12);
    EXPECT_FALSE(released.exit_ready);
  }
}

TEST(EnergyRecovery, ExitRequiresRestoredGainAndVerifiedNominalEnergy) {
  EnergyRecoveryState state{EnergyControlPhase::kRecovering, 0.004};
  auto terms = updateEnergyRecovery(
      0.01, 0.09, 0.0, 0.12, 0.95, true, true, false, true, true, &state);
  EXPECT_DOUBLE_EQ(terms.scale, 1.0);
  EXPECT_FALSE(terms.exited);  // The preceding command was still scaled.
  terms = updateEnergyRecovery(
      0.01, 0.09, 0.0, 0.12, 0.95, true, true, false, true, false, &state);
  EXPECT_TRUE(terms.exit_ready);
  EXPECT_FALSE(terms.exited);  // Full gain alone is not verification.
  terms = updateEnergyRecovery(
      0.01, 0.09, 0.0, 0.12, 0.95, true, true, false, false, true, &state);
  EXPECT_FALSE(terms.exited);  // Invalid motion state prevents release.
  terms = updateEnergyRecovery(
      0.01, 0.109, 0.0, 0.12, 0.95, true, true, false, true, true, &state);
  EXPECT_DOUBLE_EQ(terms.scale, 1.0);
  EXPECT_FALSE(terms.exited);  // Hysteresis uses nominal energy.
  terms = updateEnergyRecovery(
      0.01, 0.09, 0.0, 0.12, 0.95, true, true, false, true, true, &state);
  EXPECT_TRUE(terms.exited);
  EXPECT_DOUBLE_EQ(terms.scale, 1.0);
  EXPECT_EQ(state.phase, EnergyControlPhase::kNormal);
}

TEST(EnergyRecovery, MissingObservationsCannotRestoreGainsOrRelease) {
  EnergyRecoveryState state{EnergyControlPhase::kLimited, 0.004};
  auto terms = updateEnergyRecovery(
      0.0, 30.0, 0.0, 0.12, 0.95, true, false, false, true, true, &state);
  EXPECT_NEAR(terms.scale, 0.004, 1e-12);
  EXPECT_EQ(state.phase, EnergyControlPhase::kRecovering);
  for (int i = 0; i < 3; ++i) {
    terms = updateEnergyRecovery(
        0.0, 0.01, 0.0, 0.12, 0.95, true, false, false, true, true, &state);
    EXPECT_FALSE(terms.exited);
    EXPECT_TRUE(terms.scaling_active);
  }
  // Missing observations at startup must also engage protection.
  state = EnergyRecoveryState{};
  terms = updateEnergyRecovery(
      0.0, 30.0, 0.0, 0.12, 0.95, true, false, false, true, false, &state);
  EXPECT_NEAR(terms.scale, 0.004, 1e-12);
}

TEST(EnergyRecovery, UnitScaleDuringScheduledGainBlendDoesNotMeanFullStiffness) {
  EnergyRecoveryState state{EnergyControlPhase::kRecovering, 1.0};
  auto terms = updateEnergyRecovery(
      0.0, 0.01, 0.0, 0.12, 0.95, true, true, false, true, true, &state, false);
  EXPECT_DOUBLE_EQ(terms.scale, 1.0);
  EXPECT_FALSE(terms.exited);
  terms = updateEnergyRecovery(
      0.0, 0.01, 0.0, 0.12, 0.95, true, true, false, true, true, &state, true);
  EXPECT_FALSE(terms.exited);  // The preceding scheduled gains were lower.
  terms = updateEnergyRecovery(
      0.0, 0.01, 0.0, 0.12, 0.95, true, true, false, true, true, &state, true);
  EXPECT_TRUE(terms.exited);
}

TEST(EnergyRecovery, ReentryAndHighKineticEnergyKeepProtection) {
  EnergyRecoveryState state{EnergyControlPhase::kRecovering, 1.0};
  auto terms = updateEnergyRecovery(
      0.0, 0.01, 0.0, 0.12, 0.95, true, true, true, true, true, &state);
  EXPECT_EQ(state.phase, EnergyControlPhase::kLimited);
  EXPECT_FALSE(terms.exited);
  terms = updateEnergyRecovery(
      0.2, 0.01, 0.0, 0.12, 0.95, true, true, false, true, true, &state);
  EXPECT_EQ(state.phase, EnergyControlPhase::kRecovering);
  EXPECT_DOUBLE_EQ(terms.scale, 0.0);
  EXPECT_FALSE(terms.exited);
}

TEST(EnergyRecovery, NullspaceAndInvalidEnergyCannotBypassRecovery) {
  EnergyRecoveryState state{EnergyControlPhase::kRecovering, 1.0};
  auto terms = updateEnergyRecovery(
      0.01, 0.01, 1.0, 0.12, 0.95, true, true, false, true, true, &state);
  EXPECT_LT(terms.scale, 1.0);
  EXPECT_FALSE(terms.exited);
  terms = updateEnergyRecovery(
      std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.12,
      0.95, true, true, false, true, true, &state);
  EXPECT_DOUBLE_EQ(terms.scale, 0.0);
  EXPECT_FALSE(terms.exited);
}

TEST(EnergyRecovery, FreeMotionAndCalibrationRetainExistingGainPolicy) {
  EnergyRecoveryState state;
  auto terms = updateEnergyRecovery(
      0.0, 30.0, 0.0, 0.12, 0.95, true, true, false, true, true, &state);
  EXPECT_FALSE(terms.scaling_active);
  EXPECT_DOUBLE_EQ(terms.scale, 1.0);
  state.phase = EnergyControlPhase::kLimited;
  terms = updateEnergyRecovery(
      0.0, 30.0, 0.0, 0.12, 0.95, false, false, true, true, false, &state);
  EXPECT_FALSE(terms.scaling_active);
  EXPECT_EQ(state.phase, EnergyControlPhase::kNormal);
}

TEST(EnergyRecovery, OldOrUnverifiedCommandsCannotAuthorizeExit) {
  ImpedanceSample command;
  command.energy_recovery_epoch = 2;
  EXPECT_FALSE(energyRecoveryExitPermitted(command, 2));
  command.energy_recovery_exit_allowed = true;
  EXPECT_TRUE(energyRecoveryExitPermitted(command, 2));
  EXPECT_FALSE(energyRecoveryExitPermitted(command, 3));
}

TEST(EnergyRecovery, HandoffRejectsDifferentTransitionState) {
  EnergyRecoveryState actual{EnergyControlPhase::kRecovering, 0.2};
  JointPredictionSample prediction;
  prediction.energy_control_phase = EnergyControlPhase::kNormal;
  EXPECT_FALSE(energyRecoveryStateMatchesPrediction(prediction, actual));
  prediction.energy_control_phase = EnergyControlPhase::kRecovering;
  prediction.energy_recovery_runtime_scale = 0.3;
  EXPECT_TRUE(energyRecoveryStateMatchesPrediction(prediction, actual));
  prediction.energy_recovery_runtime_scale = 1.0;
  EXPECT_FALSE(energyRecoveryStateMatchesPrediction(prediction, actual));
  actual.last_scale = 1.0;
  EXPECT_TRUE(energyRecoveryStateMatchesPrediction(prediction, actual));
  actual.last_nominal_gains_restored = false;
  EXPECT_FALSE(energyRecoveryStateMatchesPrediction(prediction, actual));
}

TEST(EnergyRecovery, VerifiedPlanOnlyAuthorizesThePredictedExitCommand) {
  VerifiedPlan plan;
  ImpedanceSample command;
  command.energy_recovery_exit_allowed = true;
  command.energy_recovery_epoch = 7;
  command.t = 0.001;
  plan.intended.push_back(command);
  command.t = 0.002;
  plan.intended.push_back(command);
  command.t = 0.003;
  plan.failsafe.push_back(command);
  std::vector<JointPredictionSample> trace(3);
  for (std::size_t i = 0; i < trace.size(); ++i) {
    trace[i].t = (i + 1) * 0.001;
    trace[i].energy_valid = true;
    trace[i].energy_control_phase = EnergyControlPhase::kRecovering;
  }
  trace[1].energy_control_phase = EnergyControlPhase::kNormal;
  trace[1].energy_recovery_exited = true;
  trace[2].energy_control_phase = EnergyControlPhase::kNormal;

  restrictEnergyRecoveryExitPermissions(&plan, trace);
  EXPECT_FALSE(plan.intended[0].energy_recovery_exit_allowed);
  EXPECT_TRUE(plan.intended[1].energy_recovery_exit_allowed);
  EXPECT_FALSE(plan.failsafe[0].energy_recovery_exit_allowed);

  // Even if measured energy becomes small earlier than predicted, a safe
  // candidate must not authorize an earlier, unmodelled control-law switch.
  EnergyRecoveryState actual{EnergyControlPhase::kRecovering, 1.0};
  auto terms = updateEnergyRecovery(
      0.0, 0.01, 0.0, 0.12, 0.95, true, true, false, true,
      energyRecoveryExitPermitted(plan.intended[0], 7), &actual);
  EXPECT_TRUE(terms.exit_ready);
  EXPECT_FALSE(terms.exited);
  terms = updateEnergyRecovery(
      0.0, 0.01, 0.0, 0.12, 0.95, true, true, false, true,
      energyRecoveryExitPermitted(plan.intended[1], 7), &actual);
  EXPECT_TRUE(terms.exited);
}

TEST(EnergyRecovery, ExitAuthorizationPreservesCommitmentsAndRequiresValidEndpoint) {
  VerifiedPlan plan;
  ImpedanceSample command;
  command.energy_recovery_exit_allowed = true;
  command.energy_recovery_epoch = 4;
  command.t = 0.001;
  plan.intended.push_back(command);
  command.energy_recovery_epoch = 5;
  command.t = 0.002;
  plan.intended.push_back(command);
  command.t = 0.003;
  plan.failsafe.push_back(command);
  JointPredictionSample endpoint;
  endpoint.t = 0.003;
  endpoint.energy_valid = true;
  endpoint.energy_recovery_exited = true;

  restrictEnergyRecoveryExitPermissions(&plan, {endpoint}, 1);
  EXPECT_TRUE(plan.intended[0].energy_recovery_exit_allowed);
  EXPECT_EQ(plan.intended[0].energy_recovery_epoch, 4U);
  EXPECT_FALSE(plan.intended[1].energy_recovery_exit_allowed);  // Missing endpoint.
  EXPECT_TRUE(plan.failsafe[0].energy_recovery_exit_allowed);
  EXPECT_EQ(plan.failsafe[0].energy_recovery_epoch, 5U);

  endpoint.energy_valid = false;
  restrictEnergyRecoveryExitPermissions(&plan, {endpoint}, 1);
  EXPECT_FALSE(plan.failsafe[0].energy_recovery_exit_allowed);
  restrictEnergyRecoveryExitPermissions(&plan, {}, 0);
  EXPECT_FALSE(plan.intended[0].energy_recovery_exit_allowed);
}

TEST(EnergyRecovery, JointStateGateChecksPositionsVelocitiesAndFiniteValues) {
  JointDynamicsLimits limits;
  limits.position_lower.setConstant(-1.0);
  limits.position_upper.setConstant(1.0);
  limits.velocity.setConstant(2.0);
  Vector7d q = Vector7d::Zero();
  Vector7d dq = Vector7d::Zero();
  EXPECT_TRUE(jointStateWithinLimits(q, dq, limits));
  q(6) = 1.1;
  EXPECT_FALSE(jointStateWithinLimits(q, dq, limits));
  q(6) = 0.0;
  dq(4) = -2.1;
  EXPECT_FALSE(jointStateWithinLimits(q, dq, limits));
  dq(4) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(jointStateWithinLimits(q, dq, limits));
}

TEST(ReachableSafetyMonitor, RecoveryPersistsThroughIntendedAndFailsafeOutsideHuman) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.12;
  config.enable_runtime_energy_scaling = true;
  config.energy_recovery_state = {EnergyControlPhase::kLimited, 0.004};
  config.assume_human_workspace_clear = true;
  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.001;
  sample.p.x() = 0.2;
  sample.K(0, 0) = 1500.0;
  sample.energy_recovery_exit_allowed = true;
  plan.intended.push_back(sample);
  sample.t = 0.002;
  sample.failsafe = true;
  plan.failsafe.push_back(sample);
  std::vector<JointPredictionSample> trace;
  const auto result = verifyReachablePlanJointSpace(
      plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config, &trace);
  EXPECT_FALSE(result.monitored_contact_possible);
  ASSERT_EQ(trace.size(), 3U);
  for (std::size_t i = 1; i < trace.size(); ++i) {
    EXPECT_EQ(trace[i].energy_control_phase, EnergyControlPhase::kRecovering);
    EXPECT_FALSE(trace[i].energy_scaling_active);
    EXPECT_DOUBLE_EQ(trace[i].energy_stiffness_scale, 1.0);
    EXPECT_LE(trace[i].energy_recovery_runtime_scale, 0.004 + 1e-12);
  }
  EXPECT_TRUE(result.recovery_energy_check_active);
  EXPECT_TRUE(result.predicted_trigger);
  EXPECT_GE(result.worst_case_total_control_energy_ub, 30.0);
  const auto recovery_trace = trace;
  // The latch changes acceptance, never nominal rollout dynamics.
  config.energy_recovery_state = EnergyRecoveryState{};
  verifyReachablePlanJointSpace(
      plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config, &trace);
  EXPECT_DOUBLE_EQ(trace[1].energy_stiffness_scale, 1.0);
  EXPECT_NEAR(trace[1].dq(0), 0.3, 1e-12);
  EXPECT_TRUE(trace.back().dq.isApprox(recovery_trace.back().dq, 1e-12));
}

TEST(ReachableSafetyMonitor, NominalGainsOverrideReducedAnchorAndCommandGains) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.assume_human_workspace_clear = true;
  config.enable_runtime_energy_scaling = true;
  config.energy_recovery_nominal_gains_valid = true;
  config.energy_recovery_nominal_stiffness(0, 0) = 1500.0;
  config.energy_recovery_nominal_damping(0, 0) = 20.0;
  config.nullspace_stiffness = 10.0;
  config.current_nullspace_stiffness = 0.1;
  config.nullspace_reference(6) = 0.1;
  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.p.x() = 0.2;
  plan.anchor.K(0, 0) = 1.0;
  config.current_energy_reference = plan.anchor;
  config.current_energy_reference_valid = true;
  auto command = plan.anchor;
  command.t = 0.001;
  plan.intended.push_back(command);
  Vector7d dq = Vector7d::Zero();
  dq(0) = 0.1;
  for (const auto phase : {EnergyControlPhase::kNormal,
                          EnergyControlPhase::kLimited,
                          EnergyControlPhase::kRecovering}) {
    config.energy_recovery_state = {phase, 0.001, false};
    std::vector<JointPredictionSample> trace;
    const auto result = verifyReachablePlanJointSpace(
        plan, Vector7d::Zero(), dq, dynamics, config, &trace);
    ASSERT_EQ(trace.size(), 2U);
    EXPECT_NEAR(trace.front().cartesian_potential_energy, 30.0, 1e-12);
    EXPECT_NEAR(result.current_cartesian_potential_energy, 30.0, 1e-12);
    EXPECT_NEAR(result.current_nullspace_potential_energy, 0.05, 1e-12);
    EXPECT_NEAR(trace.back().dq(0), 0.398, 1e-12);  // 300 N - 20 * 0.1 N.
    EXPECT_DOUBLE_EQ(trace.back().applied_nullspace_stiffness, 10.0);
    EXPECT_EQ(result.predicted_trigger, phase != EnergyControlPhase::kNormal);
  }
}

TEST(ReachableSafetyMonitor, RecoveryGatesIntendedAndFailsafeWithClearSaraGeometry) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.12;
  config.enable_runtime_energy_scaling = true;
  config.energy_recovery_state = {EnergyControlPhase::kRecovering, 1.0};
  config.robot_reachability_provider =
      std::make_shared<RecordingRobotReachabilityProvider>();
  VerifiedPlan plan;
  plan.valid = true;
  auto command = plan.anchor;
  command.K(0, 0) = 1500.0;
  command.p.x() = 0.001;
  command.energy_recovery_exit_allowed = true;
  command.t = 0.001;
  plan.intended.push_back(command);
  command.t = 0.002;
  command.p.x() = 0.2;
  command.failsafe = true;
  plan.failsafe.push_back(command);
  for (const bool assume_clear : {false, true}) {
    config.assume_human_workspace_clear = assume_clear;
    const auto unsafe = verifyReachablePlanJointSpace(
        plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config);
    EXPECT_FALSE(unsafe.monitored_contact_possible);
    EXPECT_TRUE(unsafe.recovery_energy_check_active);
    EXPECT_TRUE(unsafe.predicted_trigger);
    EXPECT_EQ(unsafe.first_contact_interval_index, -1);
    EXPECT_EQ(unsafe.first_energy_unsafe_contact_interval_index, 1);
    EXPECT_GE(unsafe.worst_case_total_control_energy_ub, 30.0);
    // A hypothetical exit in intended must not un-gate the failsafe tail.
    auto safe_plan = plan;
    safe_plan.failsafe[0].p.x() = 0.001;
    const auto safe = verifyReachablePlanJointSpace(
        safe_plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config);
    EXPECT_TRUE(safe.recovery_energy_check_active);
    EXPECT_FALSE(safe.predicted_trigger);
    EXPECT_FALSE(safe.joint_limit_unsafe);
  }
}

TEST(ReachableSafetyMonitor, RecoveryExitIsPredictedAfterCommittedPrefix) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.energy_budget_joule = 0.12;
  config.enable_runtime_energy_scaling = true;
  config.energy_recovery_state = {EnergyControlPhase::kRecovering, 1.0};
  config.energy_recovery_epoch = 7;
  config.assume_human_workspace_clear = true;
  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.K(0, 0) = 1500.0;
  sample.p.x() = 0.01;
  sample.energy_recovery_exit_allowed = true;
  sample.energy_recovery_epoch = 6;  // Previously committed command.
  sample.t = 0.001;
  plan.intended.push_back(sample);
  sample.energy_recovery_epoch = 7;
  sample.t = 0.002;
  plan.intended.push_back(sample);
  std::vector<JointPredictionSample> trace;
  verifyReachablePlanJointSpace(
      plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config, &trace);
  ASSERT_EQ(trace.size(), 3U);
  EXPECT_EQ(trace[1].energy_control_phase, EnergyControlPhase::kRecovering);
  EXPECT_FALSE(trace[1].energy_recovery_exited);
  EXPECT_EQ(trace[2].energy_control_phase, EnergyControlPhase::kNormal);
  EXPECT_TRUE(trace[2].energy_recovery_exited);
  EXPECT_DOUBLE_EQ(trace[1].energy_stiffness_scale, 1.0);
  EXPECT_DOUBLE_EQ(trace[2].energy_stiffness_scale, 1.0);

  // Freezing the exit event must preserve the rollout that was verified,
  // including the committed prefix's existing permission and epoch.
  const auto original_trace = trace;
  restrictEnergyRecoveryExitPermissions(&plan, trace, 1);
  verifyReachablePlanJointSpace(
      plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config, &trace);
  ASSERT_EQ(trace.size(), original_trace.size());
  for (std::size_t i = 0; i < trace.size(); ++i) {
    EXPECT_EQ(trace[i].energy_control_phase, original_trace[i].energy_control_phase);
    EXPECT_EQ(trace[i].energy_recovery_exited, original_trace[i].energy_recovery_exited);
    EXPECT_DOUBLE_EQ(trace[i].energy_stiffness_scale, original_trace[i].energy_stiffness_scale);
    EXPECT_TRUE(trace[i].q.isApprox(original_trace[i].q, 1e-12));
    EXPECT_TRUE(trace[i].dq.isApprox(original_trace[i].dq, 1e-12));
  }
}

TEST(ReachableSafetyMonitor, RecoveryDoesNotExitDuringScheduledGainRestoration) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.enable_runtime_energy_scaling = true;
  config.energy_recovery_state = {EnergyControlPhase::kRecovering, 1.0, false};
  config.assume_human_workspace_clear = true;
  config.energy_recovery_nominal_gains_valid = true;
  config.energy_recovery_nominal_stiffness(0, 0) = 1500.0;
  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.energy_recovery_exit_allowed = true;
  sample.K(0, 0) = 100.0;
  sample.t = 0.001;
  plan.intended.push_back(sample);
  sample.K = config.energy_recovery_nominal_stiffness;
  sample.t = 0.002;
  plan.intended.push_back(sample);
  sample.t = 0.003;
  plan.intended.push_back(sample);
  std::vector<JointPredictionSample> trace;
  verifyReachablePlanJointSpace(
      plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config, &trace);
  ASSERT_EQ(trace.size(), 4U);
  EXPECT_EQ(trace[1].energy_control_phase, EnergyControlPhase::kRecovering);
  EXPECT_EQ(trace[2].energy_control_phase, EnergyControlPhase::kRecovering);
  EXPECT_EQ(trace[3].energy_control_phase, EnergyControlPhase::kNormal);
  EXPECT_FALSE(trace[1].energy_nominal_gains_restored);
  EXPECT_TRUE(trace[2].energy_nominal_gains_restored);
}

TEST(ReachableSafetyMonitor, LeavingOverlapWithinRolloutStartsRecovery) {
  IdentityJointDynamicsProvider dynamics;
  dynamics.control_position_from_q0_ = true;
  SafetyMonitorConfig config;
  config.enable_runtime_energy_scaling = true;
  config.energy_budget_joule = 0.012;
  config.ee_collision_radius = 0.0;
  config.tracking_acc_error_bound = 0.0;
  cps_human_workspace::HumanWorkspace::Parameters human;
  human.sphere_center.setZero();
  human.motion_radius = 1e-6;
  human.hand_max_velocity = 0.0;
  // ReachLib requires positive acceleration. Zero maximum velocity keeps the
  // hand stationary without creating 0/0 in the reachable-set calculation.
  human.hand_max_acceleration = 1.0;
  config.human_workspace.setParameters(human);
  ASSERT_TRUE(config.human_workspace.handReachableSetAtTime(0.0).center.allFinite());
  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.p.x() = 0.2;
  sample.K(0, 0) = 1000.0;
  sample.energy_recovery_exit_allowed = true;
  sample.t = 0.001;
  plan.intended.push_back(sample);
  sample.t = 0.002;
  sample.failsafe = true;
  plan.failsafe.push_back(sample);
  Vector7d dq = Vector7d::Zero();
  dq(0) = 0.1;
  std::vector<JointPredictionSample> trace;
  verifyReachablePlanJointSpace(plan, Vector7d::Zero(), dq, dynamics, config, &trace);
  ASSERT_EQ(trace.size(), 3U);
  EXPECT_EQ(trace[1].energy_control_phase, EnergyControlPhase::kLimited);
  EXPECT_EQ(trace[2].energy_control_phase, EnergyControlPhase::kRecovering);
  EXPECT_LT(trace[2].energy_recovery_runtime_scale, 0.001);
}

TEST(ReachableSafetyMonitor, InvalidWorkspaceGeometryCannotVerifyRecoveryExit) {
  IdentityJointDynamicsProvider dynamics;
  SafetyMonitorConfig config;
  config.enable_runtime_energy_scaling = true;
  config.energy_recovery_state = {EnergyControlPhase::kRecovering, 1.0};
  cps_human_workspace::HumanWorkspace::Parameters human;
  human.sphere_center.x() = std::numeric_limits<double>::quiet_NaN();
  config.human_workspace.setParameters(human);
  VerifiedPlan plan;
  plan.valid = true;
  plan.anchor.q = Quaterniond::Identity();
  ImpedanceSample sample = plan.anchor;
  sample.t = 0.001;
  sample.energy_recovery_exit_allowed = true;
  plan.intended.push_back(sample);
  std::vector<JointPredictionSample> trace;
  const auto result = verifyReachablePlanJointSpace(
      plan, Vector7d::Zero(), Vector7d::Zero(), dynamics, config, &trace);
  EXPECT_TRUE(result.monitored_unsafe);
  EXPECT_TRUE(result.predicted_trigger);
  EXPECT_TRUE(trace.empty());
}

TEST(ReachableSafetyMonitor, RolloutUsesNominalGainsDespiteRuntimeScalingInsideCollisionArea) {
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
  EXPECT_FALSE(prediction_trace.front().energy_scaling_active);
  EXPECT_FALSE(prediction_trace.back().energy_scaling_active);
  EXPECT_DOUBLE_EQ(prediction_trace.back().energy_stiffness_scale, 1.0);
  EXPECT_NEAR(prediction_trace.back().dq(0), 0.1, 1e-12);
  EXPECT_NEAR(
      prediction_trace.back().energy_recovery_runtime_scale, 0.2, 1.0e-12);
  EXPECT_NEAR(
      prediction_trace.back().cartesian_potential_energy,
      0.5,
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

TEST(ReachableSafetyMonitor, RolloutKeepsNominalGainsAfterEnteringCollisionArea) {
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
  EXPECT_FALSE(prediction_trace[2].energy_scaling_active);
  EXPECT_DOUBLE_EQ(prediction_trace[2].energy_stiffness_scale, 1.0);
  EXPECT_GT(prediction_trace[2].energy_recovery_runtime_scale, 0.0);
  EXPECT_LT(prediction_trace[2].energy_recovery_runtime_scale, 1.0);
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
