// Public data types shared by the reachable Cartesian impedance controller.
#pragma once

#include <vector>

#include <Eigen/Dense>

#include "cps_safety_monitor/reachable_safety_monitor.hpp"

namespace cps_controllers
{

using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;
using Matrix37d = Eigen::Matrix<double, 3, 7>;

using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Quaterniond = Eigen::Quaterniond;

enum class SafetyMode
{
  // Orthogonal state encoded for compact logging:
  //   contact-energy constraint inactive/active x nominal/fallback execution.
  kNominal = 0,
  kLastVerifiedMonitored = 1,
  kNominalContactPossible = 2,
  kLastVerifiedContactPossible = 3
};

enum class ExecutionStage
{
  // Actual command source. This is deliberately independent of SafetyMode.
  kNominalVerified = 0,
  kLastVerifiedIntended = 1,
  kFailsafe = 2,
  kEnergyHold = 3,
  kContactVerificationHold = 4,
  // The latest path-consistent intended stream is deliberately executable
  // inside the current workspace even when its predictive monitor result is
  // rejected. The 1 kHz energy governor is the safety mechanism there.
  kContactEnergyIntended = 5
};

enum class FallbackReason
{
  // Why the current command stream could not remain nominal. This is kept
  // separate from ExecutionStage: a single cause can first consume the
  // last-verified intended prefix and only later reach its fail-safe tail.
  kNone = 0,
  kBootstrapNoVerifiedPlan = 1,
  kCandidatePredictedUnsafe = 2,
  kPlannerOrPlanBuildFailure = 3,
  kAsyncOutputStale = 4,
  kSourcePlanGenerationMismatch = 5,
  kActivationDeadlineMissed = 6,
  kVerifiedIntendedExhausted = 7,
  kEmergencyStopNoCommand = 8
};

enum class PlanFailureReason
{
  // Detailed reason behind FallbackReason::kPlannerOrPlanBuildFailure.
  kNone = 0,
  kNoActivePath = 1,
  kMissingNominalPathState = 2,
  kIntendedGenerationEmpty = 3,
  kIntendedSeamInvalid = 4,
  kFailsafeGenerationEmpty = 5,
  kFailsafeSeamInvalid = 6,
  kIntendedSampleInvalid = 7,
  kIntendedTransitionInvalid = 8,
  kFailsafeSampleInvalid = 9,
  kFailsafeTransitionInvalid = 10,
  kCandidateInvalidUnknown = 11
};

inline bool isNominalSafetyMode(SafetyMode mode)
{
  return mode == SafetyMode::kNominal ||
         mode == SafetyMode::kNominalContactPossible;
}

inline bool isContactEnergyMode(SafetyMode mode)
{
  return mode == SafetyMode::kNominalContactPossible ||
         mode == SafetyMode::kLastVerifiedContactPossible;
}

inline bool isLastVerifiedSafetyMode(SafetyMode mode)
{
  return mode == SafetyMode::kLastVerifiedMonitored ||
         mode == SafetyMode::kLastVerifiedContactPossible;
}

using MonitorResult = cps_safety_monitor::MonitorResult;
using ImpedanceSample = cps_safety_monitor::ImpedanceSample;
using VerifiedPlan = cps_safety_monitor::VerifiedPlan;
using SafetyMonitorConfig = cps_safety_monitor::SafetyMonitorConfig;
using JointPredictionSample = cps_safety_monitor::JointPredictionSample;

struct ShieldDecision
{
  bool candidate_verified{false};
  bool executing_last_verified_monitored{false};
  bool has_evaluated_plan{false};
  bool has_contact_intended_plan{false};
  FallbackReason fallback_reason{FallbackReason::kNone};
  PlanFailureReason plan_failure_reason{PlanFailureReason::kNone};

  ImpedanceSample command;
  MonitorResult monitor;
  VerifiedPlan evaluated_plan;
  // Filled only when prediction logging is enabled. It is produced by the
  // same joint rollout that made the monitor decision.
  std::vector<JointPredictionSample> joint_prediction_trace;
  // A path-consistent intended stream may still be valid when construction of
  // the separate fail-safe reserve fails. It is executable only in measured
  // contact, under the 1 kHz Cartesian energy governor.
  VerifiedPlan contact_intended_plan;
  double monitor_total_ms{0.0};
  double planner_ms{0.0};
  double plan_build_ms{0.0};
  double monitor_eval_ms{0.0};
};

}  // namespace cps_controllers
