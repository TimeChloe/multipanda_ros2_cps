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
#include "cps_controllers/latest_value_mailbox.hpp"
#include "cps_controllers/reachable_cartesian_impedance/types.hpp"
#include "cps_safety_monitor/reachable_safety_monitor.hpp"
#include "cps_trajectory_generators/reachable_cartesian_trajectory.hpp"
#include "cps_controllers/panda_control_limits.hpp"
#include "franka_semantic_components/franka_robot_model.hpp"

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace cps_controllers {

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
                                      const Vector7d& coriolis,
                                      const Vector6d& control_jdot_dq,
                                      const Vector7d& previous_torque_command,
                                      const Matrix6d& K_runtime,
                                      const Matrix6d& D_runtime,
                                      const cps_human_workspace::HumanWorkspace& human_workspace,
                                      bool human_workspace_active,
                                      bool human_workspace_assumed_clear,
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
                                       const Matrix67d& J_geo,
                                       const Vector7d& coriolis,
                                       const Vector6d& control_jdot_dq);

  SafetyMonitorConfig makeSafetyMonitorConfig(
      const cps_human_workspace::HumanWorkspace& human_workspace,
      const Matrix6d& K_runtime,
      const Matrix6d& D_runtime,
      double wall_time) const;

  struct AsyncMonitorInput {
    std::uint64_t sequence{0};
    std::uint64_t control_loop_sequence{0};
    std::uint64_t source_plan_generation{0};
    std::uint64_t scheduled_control_loop_sequence{0};
    std::uint64_t publish_lateness_cycles{0};
    std::int64_t publish_steady_time_ns{0};
    double wall_time{0.0};
    double nominal_guess_time{0.0};

    Vector7d q{Vector7d::Zero()};
    Vector7d dq{Vector7d::Zero()};
    Vector3d current_position{Vector3d::Zero()};
    Quaterniond current_orientation{Quaterniond::Identity()};
    Vector6d ee_twist{Vector6d::Zero()};
    Matrix7d inertia{Matrix7d::Zero()};
    Vector7d coriolis{Vector7d::Zero()};
    Vector6d control_jdot_dq{Vector6d::Zero()};
    Matrix37d Jv{Matrix37d::Zero()};
    Matrix67d J_geo{Matrix67d::Zero()};
    Vector7d previous_torque_command{Vector7d::Zero()};
    Matrix6d K_runtime{Matrix6d::Zero()};
    Matrix6d D_runtime{Matrix6d::Zero()};
    cps_human_workspace::HumanWorkspace human_workspace;
    bool human_workspace_active{false};
    bool human_workspace_assumed_clear{false};

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

  struct AsyncMonitorTiming {
    bool valid{false};
    std::uint64_t input_sequence{0};
    std::uint64_t input_control_loop_sequence{0};
    std::uint64_t source_plan_generation{0};
    std::size_t committed_prefix_steps{0};
    bool source_plan_matches_at_handoff{false};
    bool output_usable{false};
    std::uint64_t scheduled_control_loop_sequence{0};
    std::uint64_t publish_lateness_cycles{0};
    double worker_queue_wait_ms{0.0};
    double worker_compute_ms{0.0};
    double output_handoff_ms{0.0};
    double end_to_end_ms{0.0};
  };

  struct AsyncMonitorOutput {
    std::uint64_t sequence{0};
    double input_wall_time{0.0};
    bool valid{false};
    AsyncMonitorInput input;
    ShieldDecision decision;
    std::int64_t worker_start_steady_time_ns{0};
    std::int64_t worker_finish_steady_time_ns{0};
    double worker_queue_wait_ms{0.0};
    double worker_compute_ms{0.0};
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
      std::uint64_t* sequence,
      bool* calibration_execution,
      double* calibration_capture_path_time_sec);
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
  rclcpp_action::GoalResponse handleCalibrationActionGoal(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const CartesianViaMotion::Goal> goal);
  rclcpp_action::CancelResponse handleCalibrationActionCancel(
      const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle);
  void handleCalibrationActionAccepted(
      const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle);
  void queueCartesianViaPointsActionGoal(
      const std::shared_ptr<CartesianViaMotionGoalHandle> goal_handle,
      bool calibration_execution);
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
      bool human_workspace_active,
      bool human_workspace_assumed_clear,
      const VerifiedPlan& evaluated_plan,
      const std::vector<JointPredictionSample>& joint_prediction_trace,
      const AsyncMonitorTiming& async_timing,
      const MonitorResult& monitor,
      int mode,
      bool candidate_verified,
      std::uint64_t accepted_plan_generation,
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
    bool human_workspace_active{false};
    bool human_workspace_assumed_clear{false};
    VerifiedPlan evaluated_plan{};
    std::vector<JointPredictionSample> joint_prediction_trace;
    AsyncMonitorTiming async_timing{};
    MonitorResult monitor{};
    int mode{0};
    bool candidate_verified{false};
    std::uint64_t accepted_plan_generation{0};
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
      bool human_workspace_active,
      bool human_workspace_assumed_clear,
      const VerifiedPlan& evaluated_plan,
      const std::vector<JointPredictionSample>& joint_prediction_trace,
      const AsyncMonitorTiming& async_timing,
      const MonitorResult& monitor,
      int mode,
      bool candidate_verified,
      std::uint64_t accepted_plan_generation,
      bool executing_last_verified_monitored,
      double monitor_total_ms,
      double planner_ms,
      double plan_build_ms,
      double monitor_eval_ms,
      const char* source);

  struct ControlLogRecord {
    static constexpr std::size_t kMaxValues = 176;
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

  bool isOneStepCommandTransitionContinuous(
      const ImpedanceSample& previous,
      const ImpedanceSample& next,
      bool allow_measured_derivative_reanchor = false) const;

  void alignVerifiedPlanExecutionIndex(
      VerifiedPlan* plan,
      std::size_t elapsed_control_steps) const;

  std::size_t failsafeCommandCount(const VerifiedPlan& plan) const;

  VerifiedPlan planForExecutionLogging(const VerifiedPlan& plan) const;

  bool calibrationExecutionComplete() const;

  struct CartesianEnergyBudgetInfo {
    bool active{false};
    bool lambda_valid{false};
    double scale{1.0};
    double kinetic_energy{0.0};
    double potential_energy_before_scaling{0.0};
    double total_energy_before_scaling{0.0};
    double potential_energy{0.0};
    double total_energy{0.0};
  };

  bool shouldRejectCandidateWithMonitor(const MonitorResult& monitor) const;
  bool shouldRejectCandidateWithMonitor(const MonitorResult& monitor,
                                        bool human_workspace_available) const;

  bool shouldApplyCartesianEnergyBudget(
      const MonitorResult& monitor) const;

  bool computeTaskInertia(const Matrix7d& inertia,
                          const Matrix67d& J_geo,
                          Matrix6d* lambda) const;

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
  std::string calibration_action_name_{"~/calibrate_monitored_trajectory"};
  rclcpp_action::Server<CartesianViaMotion>::SharedPtr
      cartesian_via_points_action_server_;
  rclcpp_action::Server<CartesianViaMotion>::SharedPtr
      calibration_action_server_;
  std::mutex pending_cartesian_via_points_mutex_;
  std::vector<Vector3d> pending_cartesian_via_points_;
  std::vector<Quaterniond> pending_cartesian_via_point_quaternions_;
  std::shared_ptr<CartesianViaMotionGoalHandle>
      pending_cartesian_via_points_goal_handle_;
  std::uint64_t pending_cartesian_via_points_sequence_{0};
  bool pending_cartesian_via_points_available_{false};
  bool pending_cartesian_via_points_calibration_{false};
  double pending_calibration_capture_path_time_sec_{0.0};
  std::mutex cartesian_via_points_action_mutex_;
  std::shared_ptr<CartesianViaMotionGoalHandle>
      active_cartesian_via_points_goal_handle_;
  bool active_cartesian_via_points_calibration_{false};
  double cartesian_via_points_action_last_feedback_wall_time_{-1.0};
  double cartesian_via_points_action_feedback_period_sec_{0.1};

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  class PinocchioJointDynamicsProvider;
  class FrankaInterfaceJointDynamicsProvider;
  std::unique_ptr<cps_safety_monitor::JointDynamicsProvider>
      monitor_joint_dynamics_provider_;
  std::string monitor_joint_dynamics_source_{"auto"};
  std::string active_monitor_joint_dynamics_source_;
  std::string monitor_urdf_model_path_;
  // Constant joint-space inertia added only by the Pinocchio prediction
  // backend. This represents simulator armature/rotor inertia that is absent
  // from the URDF. The real Franka model interface ignores this value.
  Vector7d prediction_joint_armature_{Vector7d::Zero()};

  rclcpp::Time start_time_;

  Quaterniond desired_orientation_;
  Vector3d desired_position_;
  Vector7d desired_qn_;
  Vector7d nullspace_home_pose_{Vector7d::Zero()};
  bool nullspace_home_pose_valid_{false};

  SafetyMode mode_{SafetyMode::kNominal};
  ExecutionStage execution_stage_{ExecutionStage::kCurrentVerified};
  FallbackReason fallback_reason_{FallbackReason::kNone};
  PlanFailureReason plan_failure_reason_{PlanFailureReason::kNone};
  double failsafe_start_time_sec_{-1.0};
  double failsafe_enter_wall_time_sec_{-1.0};
  double paused_nominal_time_sec_{0.0};

  // The single configured Cartesian gain set before optional runtime
  // energy-budget scaling.
  Matrix6d K_base_{Matrix6d::Zero()};
  Matrix6d D_base_{Matrix6d::Zero()};
  Matrix6d K_runtime_{Matrix6d::Zero()};
  Matrix6d D_runtime_{Matrix6d::Zero()};

  double n_stiffness_{0.0};
  bool disable_nullspace_in_failsafe_{true};

  bool enable_safety_monitor_{true};

  double energy_budget_joule_{0.05};
  double kinetic_energy_error_bound_joule_{0.0};
  double potential_energy_error_bound_joule_{0.0};
  bool enable_runtime_energy_scaling_{true};
  // Passive calibration logging only. This never changes scheduling, plan
  // acceptance, command selection, gains, or the normal safety state machine.
  bool enable_calibration_logging_{false};
  // Explicit calibration-only override. Missing/stale human data remains a
  // fail-closed condition in normal operation.
  bool calibration_assume_no_human_{false};
  bool calibration_plan_latched_{false};
  bool calibration_plan_complete_{false};
  bool calibration_target_failed_{false};
  double calibration_requested_capture_path_time_sec_{0.0};
  double calibration_actual_capture_path_time_sec_{-1.0};
  std::size_t calibration_failsafe_command_count_{0};
  std::uint64_t calibration_plan_generation_{0};
  std::uint64_t calibration_monitor_input_sequence_{0};
  std::uint64_t calibration_activation_control_sequence_{0};
  std::size_t calibration_activation_intended_index_{0};
  std::size_t calibration_activation_failsafe_index_{0};
  double ee_collision_radius_{0.04};
  Vector3d tcp_offset_{Vector3d::Zero()};
  Vector3d ee_collision_center_offset_{Vector3d::Zero()};
  int monitor_decimation_{1};
  bool async_safety_monitor_{true};
  int monitor_worker_cpu_affinity_{-1};
  int monitor_worker_realtime_priority_{0};
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

  // Certified fixed pose error bounds plus the propagated acceleration-error
  // tube used by the verifier.
  double tracking_position_error_bound_{0.0};
  double tracking_orientation_error_bound_{0.0};
  double tracking_acc_error_bound_{0.2};

  double shield_plan_dt_{0.01};
  int shield_intended_steps_{1};
  double monitor_frequency_hz_{200.0};
  double monitor_update_period_sec_{0.01};

  double path_time_rate_min_{0.0};
  double path_time_rate_max_{1.5};
  double path_time_acc_limit_{3.0};
  double path_time_jerk_limit_{5.0};
  double path_time_rate_target_{1.0};
  double failsafe_path_time_acc_limit_{10.0};
  double failsafe_path_time_jerk_limit_{5000.0};

  int local_replan_horizon_steps_{64};
  double local_replan_dt_{0.001};
  double local_path_lookahead_sec_{0.08};
  double waypoint_merge_position_tolerance_{0.001};
  double waypoint_merge_orientation_tolerance_{0.005};
  double local_replan_max_velocity_{0.08};
  double local_replan_max_acceleration_{0.4};
  double local_replan_max_jerk_{2.0};
  double local_replan_max_angular_velocity_{0.8};
  double local_replan_max_angular_acceleration_{4.0};
  double local_replan_max_angular_jerk_{40.0};
  double failsafe_brake_max_velocity_{0.85};
  double failsafe_brake_max_acceleration_{6.5};
  double failsafe_brake_max_jerk_{3250.0};
  double failsafe_brake_max_angular_velocity_{1.25};
  double failsafe_brake_max_angular_acceleration_{12.5};
  double failsafe_brake_max_angular_jerk_{6250.0};
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
  std::uint64_t monitor_period_control_cycles_{1};
  std::uint64_t next_async_monitor_control_sequence_{1};
  std::uint64_t last_async_input_publish_control_sequence_{0};
  std::uint64_t async_monitor_schedule_late_cycles_{0};
  std::uint64_t async_monitor_schedule_skipped_slots_{0};

  std::mutex async_input_mutex_;
  std::condition_variable async_input_cv_;
  AsyncMonitorInput latest_async_input_{};
  bool async_input_pending_{false};
  std::atomic<std::uint64_t> async_monitor_input_publish_count_{0};
  std::atomic<std::uint64_t> async_monitor_input_overwrite_count_{0};
  std::atomic<std::uint64_t> async_monitor_worker_processed_count_{0};

  // The worker publishes into fixed lock-free slots. The real-time loop takes
  // the newest completed result without contending on a mutex.
  LatestValueMailbox<AsyncMonitorOutput, 3> async_output_mailbox_;
  double last_async_output_wall_time_{-1.0};
  bool last_async_output_valid_{false};
  std::atomic<std::uint64_t> async_monitor_output_overwrite_count_{0};
  std::atomic<std::uint64_t> async_monitor_output_consumed_count_{0};
  AsyncMonitorTiming last_async_monitor_timing_{};
  std::size_t async_late_activation_accept_count_{0};
  std::size_t async_activation_deadline_miss_count_{0};

  VerifiedPlan last_verified_plan_{};
  std::uint64_t last_verified_plan_generation_{0};
  int last_verified_command_stage_{0};
  std::size_t last_verified_command_index_{0};
  bool verified_command_selected_this_cycle_{false};
  bool last_commanded_verified_plan_valid_{false};
  std::uint64_t last_commanded_verified_plan_generation_{0};
  int last_commanded_verified_command_stage_{0};
  std::size_t last_commanded_verified_command_index_{0};

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
  double last_cartesian_potential_energy_before_scaling_{0.0};
  double last_cartesian_control_energy_before_scaling_{0.0};
  double last_cartesian_potential_energy_{0.0};
  double last_cartesian_control_energy_{0.0};
};

}  // namespace cps_controllers
