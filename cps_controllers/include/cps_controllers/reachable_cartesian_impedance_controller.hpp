#pragma once

#include <atomic>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <mujoco_ros_msgs/msg/scalar_stamped.hpp>
#include <panda_motion_generator_msgs/action/cartesian_via_motion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <realtime_tools/realtime_buffer.hpp>

#include "cps_human_workspace/human_workspace.hpp"
#include "cps_human_workspace/msg/human_workspace.hpp"
#include "cps_controllers/bounded_async_file_writer.hpp"
#include "cps_safety_monitor/reachable_safety_monitor.hpp"
#include "cps_trajectory_generators/reachable_cartesian_trajectory.hpp"
#include "cps_controllers/panda_control_limits.hpp"
#include "franka_semantic_components/franka_robot_model.hpp"

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace cps_controllers {

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

enum class SafetyMode {
  // Orthogonal state encoded for compact logging:
  //   contact-energy constraint inactive/active x nominal/fallback execution.
  kNominal = 0,
  kLastVerifiedMonitored = 1,
  kNominalContactPossible = 2,
  kLastVerifiedContactPossible = 3
};

enum class ExecutionStage {
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

enum class FallbackReason {
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

enum class PlanFailureReason {
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

inline bool isNominalSafetyMode(SafetyMode mode) {
  return mode == SafetyMode::kNominal ||
         mode == SafetyMode::kNominalContactPossible;
}

inline bool isContactEnergyMode(SafetyMode mode) {
  return mode == SafetyMode::kNominalContactPossible ||
         mode == SafetyMode::kLastVerifiedContactPossible;
}

inline bool isLastVerifiedSafetyMode(SafetyMode mode) {
  return mode == SafetyMode::kLastVerifiedMonitored ||
         mode == SafetyMode::kLastVerifiedContactPossible;
}

using MonitorResult = cps_safety_monitor::MonitorResult;
using ImpedanceSample = cps_safety_monitor::ImpedanceSample;
using VerifiedPlan = cps_safety_monitor::VerifiedPlan;
using SafetyMonitorConfig = cps_safety_monitor::SafetyMonitorConfig;
using JointPredictionSample = cps_safety_monitor::JointPredictionSample;

struct ShieldDecision {
  bool candidate_verified{false};
  bool executing_last_verified_monitored{false};
  bool has_evaluated_plan{false};
  bool has_contact_intended_plan{false};
  FallbackReason fallback_reason{FallbackReason::kNone};
  PlanFailureReason plan_failure_reason{PlanFailureReason::kNone};

  ImpedanceSample command;
  MonitorResult monitor;
  VerifiedPlan evaluated_plan;
  // Filled only when prediction logging is enabled.  It is produced by the
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

class ReachableCartesianImpedanceController
    : public controller_interface::ControllerInterface {
 public:
  using CartesianViaMotion = panda_motion_generator_msgs::action::CartesianViaMotion;
  using CartesianViaMotionGoalHandle =
      rclcpp_action::ServerGoalHandle<CartesianViaMotion>;

  ~ReachableCartesianImpedanceController() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

  CallbackReturn on_init() override;

  CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;

  CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr int kNumJoints = 7;
  static constexpr double kDynamicLambdaRegularization = 1.0e-6;
  static constexpr double kJdotDqFilterAlpha = 0.15;
  static constexpr double kJdotDqMaxNorm = 5.0;

  void updateRuntimeGains(const Matrix6d& K_target,
                          const Matrix6d& D_target);

  void handleHumanWorkspaceState(
      const cps_human_workspace::msg::HumanWorkspace::SharedPtr msg);

  bool refreshHumanWorkspaceForMonitor(double wall_time);

  ImpedanceSample makeEmergencyStopCommand(
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      double wall_time) const;

  bool anchorLastCommandedSampleToPathStart();

  std::vector<ImpedanceSample> makeIntendedBufferFromReplanner(
      double nominal_guess_time,
      const ImpedanceSample& planning_start_command,
      double initial_path_rate,
      double target_path_rate,
      double commanded_path_time,
      bool reanchor_path_kinematics,
      double reanchor_path_rate,
      double reanchor_path_acceleration,
      PlanFailureReason* failure_reason = nullptr) const;

  VerifiedPlan buildCandidatePlan(
      double wall_time,
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      const Vector6d& ee_twist,
      const Matrix7d& inertia,
      const Matrix37d& Jv,
      const Matrix6d& K_runtime,
      const Matrix6d& D_runtime,
      const std::vector<ImpedanceSample>& intended_samples,
      std::size_t derivative_reanchor_index,
      PlanFailureReason* failure_reason = nullptr) const;

  MonitorResult evaluateCandidatePlan(const VerifiedPlan& plan,
                                      const Vector7d& q,
                                      const Vector7d& dq,
                                      const Vector3d& current_position,
                                      const Quaterniond& current_orientation,
                                      const Vector6d& ee_twist,
                                      const Matrix7d& inertia,
                                      const Matrix67d& J_geo,
                                      const Vector7d& previous_torque_command,
                                      const Matrix6d& K_runtime,
                                      const Matrix6d& D_runtime,
                                      const cps_human_workspace::HumanWorkspace& human_workspace,
                                      bool human_workspace_active,
                                      const ImpedanceSample& current_command_reference,
                                      bool current_command_reference_valid,
                                      std::vector<JointPredictionSample>*
                                          joint_prediction_trace = nullptr) const;

  Vector3d collisionCenterOffsetWorld(const Quaterniond& orientation) const;

  Vector6d twistAtCollisionCenter(const Quaterniond& orientation,
                                  const Vector6d& flange_twist) const;

  VerifiedPlan makeSparsePlanForMonitor(const VerifiedPlan& dense_plan) const;

  VerifiedPlan makeCollisionCenterPlanForMonitor(const VerifiedPlan& flange_plan) const;

  ImpedanceSample makeCollisionCenterSampleForMonitor(
      const ImpedanceSample& flange_sample) const;

  double estimatePathRateFromTimedPathSample(double path_time,
                                             const Vector3d& cartesian_velocity) const;

  ShieldDecision computeShieldDecision(double wall_time,
                                       double nominal_guess_time,
                                       const Vector7d& q,
                                       const Vector7d& dq,
                                       const Vector3d& current_position,
                                       const Quaterniond& current_orientation,
                                       const Vector6d& ee_twist,
                                       const Matrix7d& inertia,
                                       const Matrix67d& J_geo);

  SafetyMonitorConfig makeSafetyMonitorConfig(
      const cps_human_workspace::HumanWorkspace& human_workspace,
      const Matrix6d& K_runtime,
      const Matrix6d& D_runtime,
      double wall_time) const;

  struct AsyncMonitorInput {
    std::uint64_t sequence{0};
    std::uint64_t control_loop_sequence{0};
    std::uint64_t source_plan_generation{0};
    double wall_time{0.0};
    double nominal_guess_time{0.0};

    Vector7d q{Vector7d::Zero()};
    Vector7d dq{Vector7d::Zero()};
    Vector3d current_position{Vector3d::Zero()};
    Quaterniond current_orientation{Quaterniond::Identity()};
    Vector6d ee_twist{Vector6d::Zero()};
    Matrix7d inertia{Matrix7d::Zero()};
    Matrix37d Jv{Matrix37d::Zero()};
    Matrix67d J_geo{Matrix67d::Zero()};
    Vector7d previous_torque_command{Vector7d::Zero()};
    Matrix6d K_runtime{Matrix6d::Zero()};
    Matrix6d D_runtime{Matrix6d::Zero()};
    cps_human_workspace::HumanWorkspace human_workspace;
    bool human_workspace_active{false};

    ImpedanceSample last_commanded_sample;
    bool last_commanded_sample_valid{false};
    double commanded_path_time{0.0};
    double commanded_path_rate{0.0};
    double target_path_rate{1.0};
    bool reanchor_path_kinematics{false};
    double reanchor_path_rate{0.0};
    double reanchor_path_acceleration{0.0};
    std::size_t nominal_advance_steps{0};
    std::vector<ImpedanceSample> committed_prefix;
  };

  struct AsyncMonitorOutput {
    std::uint64_t sequence{0};
    double input_wall_time{0.0};
    bool valid{false};
    AsyncMonitorInput input;
    ShieldDecision decision;
  };

  ShieldDecision computeShieldDecisionForAsyncInput(
      const AsyncMonitorInput& input,
      VerifiedPlan& last_verified_plan) const;

  bool publishAsyncMonitorInput(AsyncMonitorInput input);
  bool takeAsyncMonitorOutput(AsyncMonitorOutput* output);
  bool takePendingCartesianViaPoints(
      std::vector<Vector3d>* points,
      std::vector<Quaterniond>* orientations,
      std::shared_ptr<CartesianViaMotionGoalHandle>* goal_handle,
      std::uint64_t* sequence);
  std::vector<cps_trajectory_generators::CartesianTrajectorySample>
  buildCartesianViaPointPath(
      const Vector3d& start_position,
      const Quaterniond& start_orientation,
      const std::vector<Vector3d>& via_points,
      const std::vector<Quaterniond>& via_orientations,
      std::size_t* waypoint_count) const;
  void acceptPendingCartesianViaPoints(
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      double wall_time);
  void resetViaPointExecutionState(const Vector3d& current_position,
                                   const Quaterniond& current_orientation,
                                   double wall_time);
  void updateCartesianViaPointsActionStatus(
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      double wall_time);
  rclcpp_action::GoalResponse handleCartesianViaPointsActionGoal(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const CartesianViaMotion::Goal> goal);
  rclcpp_action::CancelResponse handleCartesianViaPointsActionCancel(
      const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle);
  void handleCartesianViaPointsActionAccepted(
      const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle);
  void safetyMonitorWorkerLoop();
  void startSafetyMonitorWorker();
  void stopSafetyMonitorWorker();
  void handleCartesianViaPoints(
      const geometry_msgs::msg::PoseArray::SharedPtr msg);
  void handleMujocoContactSensor(
      const mujoco_ros_msgs::msg::ScalarStamped::SharedPtr msg);
  void logShieldPredictionTrajectory(
      double wall_time,
      double nominal_guess_time,
      const Vector7d& current_q,
      const Vector7d& current_dq,
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      const Vector6d& ee_twist,
      const Matrix7d& inertia,
      const Matrix37d& Jv,
      const Matrix6d& K_runtime,
      const Matrix6d& D_runtime,
      const cps_human_workspace::HumanWorkspace& human_workspace,
      const VerifiedPlan& evaluated_plan,
      const std::vector<JointPredictionSample>& joint_prediction_trace,
      const MonitorResult& monitor,
      int mode,
      bool candidate_verified,
      bool executing_last_verified_monitored,
      double monitor_total_ms,
      double planner_ms,
      double plan_build_ms,
      double monitor_eval_ms,
      const char* source);

  struct PredictionLogRecord {
    double wall_time{0.0};
    double nominal_guess_time{0.0};
    Vector7d current_q{Vector7d::Zero()};
    Vector7d current_dq{Vector7d::Zero()};
    Vector3d current_position{Vector3d::Zero()};
    Quaterniond current_orientation{Quaterniond::Identity()};
    Vector6d ee_twist{Vector6d::Zero()};
    Matrix7d inertia{Matrix7d::Zero()};
    Matrix37d Jv{Matrix37d::Zero()};
    Matrix6d K_runtime{Matrix6d::Zero()};
    Matrix6d D_runtime{Matrix6d::Zero()};
    cps_human_workspace::HumanWorkspace human_workspace;
    VerifiedPlan evaluated_plan{};
    std::vector<JointPredictionSample> joint_prediction_trace;
    MonitorResult monitor{};
    int mode{0};
    bool candidate_verified{false};
    bool executing_last_verified_monitored{false};
    double monitor_total_ms{0.0};
    double planner_ms{0.0};
    double plan_build_ms{0.0};
    double monitor_eval_ms{0.0};
    bool async_source{true};
  };

  void writeShieldPredictionTrajectory(
      std::ostream& output,
      double wall_time,
      double nominal_guess_time,
      const Vector7d& current_q,
      const Vector7d& current_dq,
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      const Vector6d& ee_twist,
      const Matrix7d& inertia,
      const Matrix37d& Jv,
      const Matrix6d& K_runtime,
      const Matrix6d& D_runtime,
      const cps_human_workspace::HumanWorkspace& human_workspace,
      const VerifiedPlan& evaluated_plan,
      const std::vector<JointPredictionSample>& joint_prediction_trace,
      const MonitorResult& monitor,
      int mode,
      bool candidate_verified,
      bool executing_last_verified_monitored,
      double monitor_total_ms,
      double planner_ms,
      double plan_build_ms,
      double monitor_eval_ms,
      const char* source);

  struct ControlLogRecord {
    static constexpr std::size_t kMaxValues = 128;
    std::array<double, kMaxValues> values{};
    std::size_t value_count{0};
  };

  bool startLogWriters();
  void stopLogWriters();

  Vector7d computeImpedanceTorque(const Vector7d& q,
                                  const Vector7d& dq,
                                  const Matrix7d& inertia,
                                  const Vector7d& coriolis,
                                  const Matrix67d& J_geo,
                                  const Vector3d& current_position,
                                  const Quaterniond& current_orientation,
                                  const ImpedanceSample& cmd,
                                  bool cartesian_energy_budget_active,
                                  double dt);

  ImpedanceSample getNextFailsafeCommandFromCache(bool advance_index);

  ImpedanceSample getNextVerifiedTrajectoryCommandFromCache(bool advance_index);

  bool getVerifiedTrajectoryCommandAtOffset(
      const VerifiedPlan& plan,
      std::size_t offset,
      ImpedanceSample* command) const;

  bool getContactIntendedCommandAtOffset(
      std::uint64_t control_loop_sequence,
      std::size_t offset,
      ImpedanceSample* command) const;

  bool isOneStepCommandTransitionContinuous(
      const ImpedanceSample& previous,
      const ImpedanceSample& next,
      bool allow_measured_derivative_reanchor = false) const;

  void alignVerifiedPlanExecutionIndex(
      VerifiedPlan* plan,
      std::size_t elapsed_control_steps) const;

  Matrix6d computeDampingFromStiffness(
      const Matrix6d& K,
      double pos_damping_scale,
      double rot_damping_scale) const;

  struct CartesianEnergyBudgetInfo {
    bool active{false};
    bool lambda_valid{false};
    double scale{1.0};
    double kinetic_energy{0.0};
    double potential_energy{0.0};
    double total_energy{0.0};
  };

  bool shouldRejectCandidateWithMonitor(const MonitorResult& monitor) const;
  bool shouldRejectCandidateWithMonitor(const MonitorResult& monitor,
                                        bool human_workspace_active) const;

  bool shouldApplyCartesianEnergyBudget(
      const MonitorResult& monitor) const;

  bool computeTaskInertia(const Matrix7d& inertia,
                          const Matrix67d& J_geo,
                          Matrix6d* lambda) const;

  ImpedanceSample makeEffectiveTimeHoldSample(
      const ImpedanceSample& command) const;
  double cartesianEnergyScaleFloor(const Matrix6d& K_reference) const;

  ImpedanceSample applyCartesianEnergyBudget(
      const ImpedanceSample& command,
      double kinetic_energy,
      double potential_energy,
      bool energy_valid,
      bool active,
      const Matrix6d& cartesian_task_inertia_sqrt,
      bool cartesian_task_inertia_valid,
      CartesianEnergyBudgetInfo* info) const;

  bool enable_error_logging_{false};
  std::string error_log_root_dir_{"/home/developer/multipanda_ws/src/data_log"};
  std::string error_log_file_name_{"reachable_cartesian_impedance_validation.csv"};
  std::string legacy_error_log_path_;
  std::string error_log_run_dir_;
  std::string error_log_file_path_;
  bool command_recording_active_{false};
  bool enable_prediction_logging_{false};
  std::string prediction_log_file_name_{"shield_prediction_trajectory.csv"};
  std::string prediction_log_file_path_;
  std::size_t control_log_max_queue_size_{16384};
  std::size_t prediction_log_max_queue_size_{256};
  std::size_t log_batch_size_{512};
  double log_flush_period_sec_{1.0};
  BoundedAsyncFileWriter<ControlLogRecord> control_log_writer_;
  BoundedAsyncFileWriter<PredictionLogRecord> prediction_log_writer_;
  std::atomic<std::size_t> control_log_column_mismatch_count_{0};

  std::string arm_id_;
  std::vector<Vector3d> cartesian_via_points_;
  std::vector<Quaterniond> cartesian_via_point_quaternions_;
  std::vector<cps_trajectory_generators::CartesianTrajectorySample>
      cartesian_via_point_path_;
  mutable std::mutex cartesian_via_point_path_mutex_;
  std::string cartesian_via_points_topic_{"cartesian_via_points"};
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr
      cartesian_via_points_sub_;
  std::string startup_via_points_source_{"yaml"};
  std::string cartesian_via_points_action_name_{"~/follow_cartesian_via_points"};
  rclcpp_action::Server<CartesianViaMotion>::SharedPtr
      cartesian_via_points_action_server_;
  std::mutex pending_cartesian_via_points_mutex_;
  std::vector<Vector3d> pending_cartesian_via_points_;
  std::vector<Quaterniond> pending_cartesian_via_point_quaternions_;
  std::shared_ptr<CartesianViaMotionGoalHandle>
      pending_cartesian_via_points_goal_handle_;
  std::uint64_t pending_cartesian_via_points_sequence_{0};
  bool pending_cartesian_via_points_available_{false};
  std::mutex cartesian_via_points_action_mutex_;
  std::shared_ptr<CartesianViaMotionGoalHandle>
      active_cartesian_via_points_goal_handle_;
  double cartesian_via_points_action_last_feedback_wall_time_{-1.0};
  double cartesian_via_points_action_feedback_period_sec_{0.1};

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  class PinocchioJointDynamicsProvider;
  std::unique_ptr<PinocchioJointDynamicsProvider>
      monitor_joint_dynamics_provider_;
  std::string monitor_urdf_model_path_;

  rclcpp::Time start_time_;

  Quaterniond desired_orientation_;
  Vector3d desired_position_;
  Vector7d desired_qn_;
  Vector7d nullspace_home_pose_{Vector7d::Zero()};
  bool nullspace_home_pose_valid_{false};

  SafetyMode mode_{SafetyMode::kNominal};
  ExecutionStage execution_stage_{ExecutionStage::kNominalVerified};
  FallbackReason fallback_reason_{FallbackReason::kNone};
  PlanFailureReason plan_failure_reason_{PlanFailureReason::kNone};
  double failsafe_start_time_sec_{-1.0};
  double failsafe_enter_wall_time_sec_{-1.0};
  double paused_nominal_time_sec_{0.0};

  Matrix6d K_nominal_{Matrix6d::Zero()};
  Matrix6d D_nominal_{Matrix6d::Zero()};
  Matrix6d K_f_target_{Matrix6d::Zero()};
  Matrix6d D_f_target_{Matrix6d::Zero()};
  Matrix6d K_runtime_{Matrix6d::Zero()};
  Matrix6d D_runtime_{Matrix6d::Zero()};

  double n_stiffness_{0.0};
  bool disable_nullspace_in_failsafe_{true};

  bool enable_safety_monitor_{true};

  double energy_budget_joule_{0.05};
  double ee_collision_radius_{0.04};
  Vector3d tcp_offset_{Vector3d::Zero()};
  Vector3d ee_collision_center_offset_{Vector3d::Zero()};
  int monitor_decimation_{1};
  bool async_safety_monitor_{true};
  double async_plan_max_age_sec_{0.02};
  // The lead is the maximum already-verified command prefix executed while
  // the worker runs. In every mode it is truncated at the intended/failsafe
  // boundary while intended commands remain, so a new candidate never
  // promises to brake just to fill it.
  // The horizon is the fresh intended tail available after activation.
  std::size_t async_planning_lead_steps_{8};
  std::size_t async_verified_horizon_steps_{20};

  cps_human_workspace::HumanWorkspace human_workspace_;
  realtime_tools::RealtimeBuffer<cps_human_workspace::HumanWorkspace::Parameters>
      human_workspace_param_buffer_;
  rclcpp::Subscription<cps_human_workspace::msg::HumanWorkspace>::SharedPtr
      human_workspace_sub_;
  std::string human_workspace_topic_{"human_workspace/state"};
  double human_workspace_timeout_sec_{0.5};
  bool human_workspace_active_{false};
  bool human_workspace_configured_static_{false};
  std::atomic_bool human_workspace_live_received_{false};
  std::atomic<double> latest_human_workspace_msg_time_sec_{-1.0};

  // 替换了旧的 error_pos_gain_alpha_ 等常数，改用固定的误差管道边界
  double tracking_acc_error_bound_{0.2};
  double joint_velocity_error_bound_{0.0};

  double shield_plan_dt_{0.01};
  int shield_intended_steps_{1};
  double monitor_frequency_hz_{200.0};
  double monitor_update_period_sec_{0.01};

  double path_time_rate_min_{0.0};
  double path_time_rate_max_{1.5};
  double path_time_acc_limit_{3.0};
  double path_time_rate_target_{1.0};

  int local_replan_horizon_steps_{64};
  double local_replan_dt_{0.001};
  double local_path_lookahead_sec_{0.08};
  double local_replan_max_velocity_{0.08};
  double local_replan_max_acceleration_{0.4};
  double local_replan_max_jerk_{2.0};
  double local_replan_max_angular_velocity_{0.8};
  double local_replan_max_angular_acceleration_{4.0};
  double local_replan_max_angular_jerk_{40.0};
  double failsafe_brake_max_velocity_{1.0};
  double failsafe_brake_max_acceleration_{4.0};
  double failsafe_brake_max_jerk_{80.0};
  double failsafe_brake_max_angular_velocity_{1.5};
  double failsafe_brake_max_angular_acceleration_{10.0};
  double failsafe_brake_max_angular_jerk_{500.0};
  std::string trajectory_generator_config_path_;

  bool use_dynamic_consistent_impedance_{true};
  bool torque_rate_limited_last_{false};
  double torque_rate_max_desired_delta_nm_last_{0.0};
  double torque_rate_limit_delta_nm_last_{0.0};
  double torque_rate_max_excess_nm_last_{0.0};
  double torque_rate_max_ratio_last_{0.0};
  double torque_rate_max_cmd_delta_nm_last_{0.0};
  Matrix67d J_geo_prev_{Matrix67d::Zero()};
  Vector6d Jdot_dq_filtered_{Vector6d::Zero()};
  bool J_geo_prev_valid_{false};

  std::size_t monitor_counter_{0};

  bool last_shield_decision_valid_{false};
  ShieldDecision last_shield_decision_{};

  std::thread safety_monitor_worker_thread_;
  std::atomic<bool> safety_monitor_worker_running_{false};
  std::atomic<std::uint64_t> async_input_sequence_{0};
  std::uint64_t control_update_sequence_{0};

  std::mutex async_input_mutex_;
  std::condition_variable async_input_cv_;
  AsyncMonitorInput latest_async_input_{};
  bool async_input_pending_{false};

  std::mutex async_output_mutex_;
  AsyncMonitorOutput latest_async_output_{};
  std::uint64_t last_consumed_async_output_sequence_{0};
  double last_async_output_wall_time_{-1.0};
  bool last_async_output_valid_{false};
  double last_async_input_publish_wall_time_{-1.0};
  std::size_t async_late_activation_accept_count_{0};
  std::size_t async_activation_deadline_miss_count_{0};

  VerifiedPlan last_verified_plan_{};
  std::uint64_t last_verified_plan_generation_{0};
  int last_verified_command_stage_{0};
  std::size_t last_verified_command_index_{0};

  // Latest path-consistent intended trajectory produced by the async worker.
  // It is separate from last_verified_plan_: an energy-unsafe prediction may
  // be executed only while the measured EE is currently inside the workspace,
  // whereas last_verified_plan_ remains the safe reserve for leaving it.
  VerifiedPlan contact_intended_plan_{};
  std::uint64_t contact_intended_input_control_sequence_{0};
  double contact_intended_input_wall_time_{-1.0};

  // Last command actually sent to impedance controller.
  // Used as replanning start position to avoid discontinuous replans
  // from noisy or lagged measured EE position.
  ImpedanceSample last_commanded_sample_{};
  bool last_commanded_sample_valid_{false};
  double commanded_path_time_{0.0};
  double commanded_path_rate_{0.0};

  Vector7d tau_cmd_prev_{Vector7d::Zero()};

  int profiling_stats_print_period_{1000};

  bool enable_mujoco_contact_logging_{true};
  std::string mujoco_contact_sensor_topic_{"/panda_metal_ball_touch"};
  double mujoco_contact_threshold_{1.0e-6};
  rclcpp::Subscription<mujoco_ros_msgs::msg::ScalarStamped>::SharedPtr
      mujoco_contact_sub_;
  std::atomic<double> latest_mujoco_contact_value_{0.0};
  std::atomic<double> latest_mujoco_contact_msg_time_{-1.0};
  std::atomic<bool> latest_mujoco_contact_active_{false};

  std::size_t loop_counter_{0};
  double exec_sum_ms_{0.0};
  double exec_min_ms_{1e9};
  double exec_max_ms_{0.0};
  std::size_t exec_overrun_1ms_count_{0};
  std::size_t exec_overrun_2ms_count_{0};
  double prof_model_sum_ms_{0.0};
  double prof_model_max_ms_{0.0};
  double prof_shield_sum_ms_{0.0};
  double prof_shield_max_ms_{0.0};
  double prof_torque_sum_ms_{0.0};
  double prof_torque_max_ms_{0.0};
  double prof_io_sum_ms_{0.0};
  double prof_io_max_ms_{0.0};

  double energy_budget_margin_joule_{0.005};
  double cartesian_energy_min_pos_stiffness_{0.0};
  double cartesian_energy_lambda_update_period_sec_{0.001};
  double cartesian_energy_damping_ratio_{0.8};
  Matrix6d cartesian_energy_task_inertia_cache_{Matrix6d::Zero()};
  Matrix6d cartesian_energy_task_inertia_sqrt_cache_{Matrix6d::Zero()};
  bool cartesian_energy_task_inertia_cache_valid_{false};
  double cartesian_energy_task_inertia_cache_wall_time_{-1.0};
  bool last_cartesian_energy_budget_active_{false};
  bool last_cartesian_energy_budget_lambda_valid_{false};
  double last_cartesian_energy_scale_{1.0};
  double last_joint_kinetic_energy_{0.0};
  double last_cartesian_potential_energy_{0.0};
  double last_cartesian_control_energy_{0.0};
  bool cartesian_effective_time_frozen_{false};
  bool contact_verification_hold_active_{false};
  FallbackReason contact_verification_hold_reason_{FallbackReason::kNone};
  PlanFailureReason contact_verification_hold_plan_failure_reason_{
      PlanFailureReason::kNone};
  double cartesian_effective_time_freeze_start_wall_time_{-1.0};
  ImpedanceSample cartesian_effective_time_hold_sample_{};
  bool cartesian_effective_time_hold_sample_valid_{false};
  // Fixed-size state used by the 1 kHz loop to project the measured TCP twist
  // onto the original path tangent while an energy hold is active. No path
  // search, allocation, or mutex is needed in the real-time loop.
  Vector3d cartesian_energy_hold_dp_ds_{Vector3d::Zero()};
  Vector3d cartesian_energy_hold_w_ds_{Vector3d::Zero()};
  bool cartesian_energy_hold_tangent_valid_{false};
  // The nominal replanner is anchored at the last command position, while its
  // scalar path velocity and acceleration come from the measured TCP motion.
  // These states deliberately contain no energy-budget information.
  double measured_path_rate_{0.0};
  bool measured_path_rate_valid_{false};
  double measured_path_acceleration_{0.0};
  bool measured_path_acceleration_valid_{false};
  // Preserve the monitor state that caused the Lachner energy hold. The
  // controller remains in mode 2 until a newly verified candidate explicitly
  // proves that the robot has left the human workspace.
  MonitorResult cartesian_effective_time_hold_monitor_{};
  double contact_activation_margin_{0.0};
  double failsafe_min_pos_stiffness_{5.0};

  double failsafe_pos_damping_scale_{2.5};
  double failsafe_rot_damping_scale_{2.5};
};

}  // namespace cps_controllers
