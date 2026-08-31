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
  // Internal orthogonal state used to retain the last collision prediction:
  //   predicted collision clear/possible x nominal/fallback execution.
  // The external `mode` log deliberately collapses this to current-verified
  // execution (0) versus fallback execution (1); collision possibility is
  // logged separately.
  kNominal = 0,
  kLastVerifiedMonitored = 1,
  kNominalContactPossible = 2,
  kLastVerifiedContactPossible = 3
};

enum class ExecutionStage
{
  // Actual command source. This is deliberately independent of SafetyMode and
  // of every failure-reason field.
  kCurrentVerified = 0,
  kLastVerifiedIntended = 1,
  kLastVerifiedFailsafe = 2,
  kHold = 3
};

enum class FallbackReason
{
  // Mutually exclusive non-planning cause of fallback execution. Command
  // source and intended/failsafe phase belong exclusively to ExecutionStage;
  // candidate-generation failures belong exclusively to PlanFailureReason.
  // kNone is valid for either a natural verified-failsafe transition or a
  // nonzero PlanFailureReason.
  kNone = 0,
  kNoVerifiedPlanAvailable = 1,
  kCandidatePredictionRejected = 2,
  kAsyncOutputUnavailable = 3,
  kHumanWorkspaceUnavailable = 4
};

enum class PlanFailureReason
{
  // Standalone, mutually exclusive reason that candidate generation failed.
  // When this is nonzero, FallbackReason must be kNone.
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

inline bool isCollisionPossibleMode(SafetyMode mode)
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
  FallbackReason fallback_reason{FallbackReason::kNone};
  PlanFailureReason plan_failure_reason{PlanFailureReason::kNone};

  ImpedanceSample command;
  MonitorResult monitor;
  VerifiedPlan evaluated_plan;
  // Filled only when prediction logging is enabled. It is produced by the
  // same joint rollout that made the monitor decision.
  std::vector<JointPredictionSample> joint_prediction_trace;
  double monitor_total_ms{0.0};
  double planner_ms{0.0};
  double plan_build_ms{0.0};
  double monitor_eval_ms{0.0};
};

}  // namespace cps_controllers
