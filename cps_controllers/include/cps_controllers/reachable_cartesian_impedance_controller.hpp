#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <mujoco_ros_msgs/msg/scalar_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "cps_human_workspace/human_workspace.hpp"
#include "cps_safety_monitor/reachable_safety_monitor.hpp"
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
  kNominal = 0,
  kFailsafe = 1
};

using MonitorResult = cps_safety_monitor::MonitorResult;
using ImpedanceSample = cps_safety_monitor::ImpedanceSample;
using VerifiedPlan = cps_safety_monitor::VerifiedPlan;
using SafetyMonitorConfig = cps_safety_monitor::SafetyMonitorConfig;

struct ShieldDecision {
  bool candidate_verified{false};
  bool executing_last_verified_failsafe{false};

  ImpedanceSample command;
  MonitorResult monitor;
};

class ReachableCartesianImpedanceController
    : public controller_interface::ControllerInterface {
 public:
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

  struct PathState {
    double t_path{0.0};
    double rate{0.0};
    double accel{0.0};
  };

  Matrix6d applyMatrixRateLimit(const Matrix6d& current,
                                const Matrix6d& target,
                                double rate_limit,
                                double dt) const;

  void updateRuntimeGains(const Matrix6d& K_target,
                          const Matrix6d& D_target,
                          double dt);

  double computeConservativeNormalAccelPositiveBound(
      double m_eff_n,
      double e_n_abs,
      double v_n_abs,
      const Matrix6d& K_used) const;

  double computeConservativeNormalBrakeAccelLowerBound(
      double m_eff_n,
      double v_n_abs,
      const Matrix6d& D_used) const;

  double estimatePathParameterTimeFromCurrentState(
      const Vector3d& current_position,
      double nominal_guess_time) const;

  double estimatePathTimeRateFromCurrentState(
      double path_time_anchor,
      const Vector3d& current_linear_velocity) const;

  PathState propagateOnlinePathState(const Vector3d& current_position,
                                     const Vector3d& current_linear_velocity,
                                     double nominal_guess_time,
                                     double dtp) const;

  void publishRvizDiagnostics(double wall_time,
                              const Vector3d& current_position,
                              const Vector3d& desired_position_cur,
                              const Vector6d& ee_twist,
                              const MonitorResult& monitor);

  ImpedanceSample makeNominalSample(double nominal_time,
                                    double path_rate,
                                    double path_accel,
                                    const Matrix6d& K_target,
                                    const Matrix6d& D_target) const;

  ImpedanceSample makeFrozenFailsafeSample(
      double nominal_time,
      const ImpedanceSample& freeze_sample,
      const Matrix6d& K_target,
      const Matrix6d& D_target) const;

  ImpedanceSample makeEmergencyStopCommand(
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      double wall_time) const;

  std::vector<ImpedanceSample> makeIntendedBufferFromReplanner(
      double nominal_guess_time,
      const ImpedanceSample& planning_start_command) const;

  bool refillIntendedBufferFromReplanner(
      double nominal_guess_time,
      const ImpedanceSample& planning_start_command);

  // 新增：构建单步执行且 path-consistent 锚点的备选计划
  VerifiedPlan buildSingleStepCandidatePlan(
      double wall_time,
      const Vector3d& current_position,
      const Quaterniond& current_orientation,
      const Vector6d& ee_twist,
      const Matrix7d& inertia,
      const Matrix37d& Jv,
      const Matrix6d& K_runtime,
      const Matrix6d& D_runtime,
      const ImpedanceSample& next_intended) const;

  MonitorResult verifyCandidatePlan(const VerifiedPlan& plan,
                                    const Vector3d& current_position,
                                    const Quaterniond& current_orientation,
                                    const Vector6d& ee_twist,
                                    const Matrix7d& inertia,
                                    const Matrix37d& Jv,
                                    const Matrix6d& K_runtime,
                                    const Matrix6d& D_runtime) const;

  Vector3d collisionCenterOffsetWorld(const Quaterniond& orientation) const;

  Vector6d twistAtCollisionCenter(const Quaterniond& orientation,
                                  const Vector6d& flange_twist) const;

  VerifiedPlan makeCollisionCenterPlanForMonitor(const VerifiedPlan& flange_plan) const;

  ShieldDecision computeShieldDecision(double wall_time,
                                       double nominal_guess_time,
                                       const Vector3d& current_position,
                                       const Quaterniond& current_orientation,
                                       const Vector6d& ee_twist,
                                       const Matrix7d& inertia,
                                       const Matrix37d& Jv);

  SafetyMonitorConfig makeSafetyMonitorConfig(const Matrix6d& K_runtime,
                                              const Matrix6d& D_runtime) const;

  struct AsyncMonitorInput {
    std::uint64_t sequence{0};
    double wall_time{0.0};
    double nominal_guess_time{0.0};

    Vector3d current_position{Vector3d::Zero()};
    Quaterniond current_orientation{Quaterniond::Identity()};
    Vector6d ee_twist{Vector6d::Zero()};
    Matrix7d inertia{Matrix7d::Zero()};
    Matrix37d Jv{Matrix37d::Zero()};
    Matrix6d K_runtime{Matrix6d::Zero()};
    Matrix6d D_runtime{Matrix6d::Zero()};

    ImpedanceSample last_commanded_sample;
    bool last_commanded_sample_valid{false};
  };

  struct AsyncMonitorOutput {
    std::uint64_t sequence{0};
    double input_wall_time{0.0};
    bool valid{false};
    ShieldDecision decision;
  };

  struct AsyncPlanningState {
    std::vector<ImpedanceSample> intended_buffer;
    std::size_t intended_buffer_index{0};
    bool intended_buffer_valid{false};
    VerifiedPlan last_verified_plan;
  };

  ShieldDecision computeShieldDecisionForAsyncInput(
      const AsyncMonitorInput& input,
      AsyncPlanningState& state) const;

  void publishAsyncMonitorInput(const AsyncMonitorInput& input);
  bool takeAsyncMonitorOutput(AsyncMonitorOutput* output);
  void safetyMonitorWorkerLoop();
  void startSafetyMonitorWorker();
  void stopSafetyMonitorWorker();
  void handleMujocoContactSensor(
      const mujoco_ros_msgs::msg::ScalarStamped::SharedPtr msg);

  Vector7d computeImpedanceTorque(const Vector7d& q,
                                  const Vector7d& dq,
                                  const Matrix7d& inertia,
                                  const Vector7d& coriolis,
                                  const Matrix67d& J_geo,
                                  const Vector3d& current_position,
                                  const Quaterniond& current_orientation,
                                  const ImpedanceSample& cmd,
                                  double dt);

  ImpedanceSample getNextIntendedCommandFromCache(bool advance_index);

  ImpedanceSample getNextFailsafeCommandFromCache(bool advance_index);

  Matrix6d computeDampingFromStiffness(
      const Matrix6d& K,
      double pos_damping_scale,
      double rot_damping_scale) const;

  bool enable_error_logging_{false};
  std::string error_log_root_dir_{"/home/developer/multipanda_ws/src/data_log"};
  std::string error_log_file_name_{"reachable_cartesian_impedance_validation.csv"};
  std::string legacy_error_log_path_;
  std::string error_log_run_dir_;
  std::string error_log_file_path_;
  std::ofstream error_log_file_;
  std::size_t log_write_counter_{0};

  std::string arm_id_;
  std::string reference_trajectory_type_{"line"};
  bool use_constant_reference_{false};

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  rclcpp::Time start_time_;

  Quaterniond desired_orientation_;
  Vector3d desired_position_;
  Vector7d desired_qn_;

  SafetyMode mode_{SafetyMode::kNominal};
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

  double safe_collision_energy_joule_{0.05};
  double ee_collision_radius_{0.04};
  Vector3d tcp_offset_{Vector3d::Zero()};
  Vector3d ee_collision_center_offset_{Vector3d::Zero()};
  int monitor_decimation_{1};
  bool async_safety_monitor_{true};
  double async_plan_max_age_sec_{0.05};

  cps_human_workspace::HumanWorkspace human_workspace_;

  double k_rate_limit_{5000.0};
  double d_rate_limit_{500.0};
  double tau_rate_limit_{1000.0};
  double torque_to_accel_gain_{8.0};
  double model_accel_uncertainty_{0.05};
  double stiffness_error_bound_m_{0.00};

  // 替换了旧的 error_pos_gain_alpha_ 等常数，改用固定的误差管道边界
  double tracking_pos_error_bound_{0.000};
  double tracking_vel_error_bound_{0.00};

  int shield_horizon_steps_{100};
  double shield_plan_dt_{0.01};

  double path_retiming_search_window_sec_{0.25};
  int path_retiming_search_steps_{41};
  double path_time_rate_min_{0.0};
  double path_time_rate_max_{1.5};
  double path_time_acc_limit_{3.0};
  double path_time_rate_target_{1.0};

  int local_replan_horizon_steps_{200};
  double local_replan_dt_{0.001};
  double local_path_lookahead_sec_{0.08};
  double local_replan_max_velocity_{0.08};
  double local_replan_max_acceleration_{0.4};
  double local_replan_max_jerk_{2.0};

  bool use_dynamic_consistent_impedance_{true};
  double torque_rate_limit_{1000.0};
  double dynamic_lambda_regularization_{1.0e-6};
  double jdot_dq_filter_alpha_{0.15};
  double jdot_dq_max_norm_{5.0};
  Matrix67d J_geo_prev_{Matrix67d::Zero()};
  Vector6d Jdot_dq_filtered_{Vector6d::Zero()};
  bool J_geo_prev_valid_{false};

  double prev_Tn_fs_{0.0};
  bool prev_Tn_fs_valid_{false};
  double last_v_n_fs_{0.0};

  std::size_t monitor_counter_{0};
  bool last_monitor_result_valid_{false};
  double last_monitor_wall_time_{0.0};
  MonitorResult last_monitor_result_{};

  bool last_shield_decision_valid_{false};
  ShieldDecision last_shield_decision_{};

  std::thread safety_monitor_worker_thread_;
  std::atomic<bool> safety_monitor_worker_running_{false};
  std::atomic<std::uint64_t> async_input_sequence_{0};

  std::mutex async_input_mutex_;
  std::condition_variable async_input_cv_;
  AsyncMonitorInput latest_async_input_{};
  bool async_input_pending_{false};

  std::mutex async_output_mutex_;
  AsyncMonitorOutput latest_async_output_{};
  std::uint64_t last_consumed_async_output_sequence_{0};
  double last_async_output_wall_time_{-1.0};
  bool last_async_output_valid_{false};

  VerifiedPlan last_verified_plan_{};
  VerifiedPlan candidate_plan_{};
  bool candidate_plan_valid_{false};

  std::vector<ImpedanceSample> intended_buffer_;
  std::size_t intended_buffer_index_{0};
  bool intended_buffer_valid_{false};

  // Last command actually sent to impedance controller.
  // Used as replanning start position to avoid discontinuous replans
  // from noisy or lagged measured EE position.
  ImpedanceSample last_commanded_sample_{};
  bool last_commanded_sample_valid_{false};
  double commanded_path_time_{0.0};

  Vector7d tau_cmd_prev_{Vector7d::Zero()};

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr rviz_marker_pub_;

  bool rviz_enable_markers_{true};
  std::string rviz_frame_id_{"panda_link0"};
  int rviz_marker_decimation_{10};
  std::size_t rviz_publish_counter_{0};

  double rviz_marker_lifetime_sec_{0.2};

  double rviz_plane_size_{0.8};
  double rviz_plane_thickness_{0.003};
  double rviz_normal_arrow_length_{0.20};
  double rviz_velocity_arrow_scale_{0.25};

  double rviz_arrow_shaft_diameter_{0.01};
  double rviz_arrow_head_diameter_{0.02};
  double rviz_arrow_head_length_{0.03};

  double rviz_text_scale_{0.04};
  double rviz_text_z_offset_{0.12};
  double rviz_ub_arrow_z_offset_{0.05};

  int profiling_stats_print_period_{1000};

  bool enable_mujoco_contact_logging_{true};
  std::string mujoco_contact_sensor_topic_{"/panda_metal_ball_touch"};
  double mujoco_contact_threshold_{1.0e-6};
  rclcpp::Subscription<mujoco_ros_msgs::msg::ScalarStamped>::SharedPtr
      mujoco_contact_sub_;
  std::atomic<double> latest_mujoco_contact_value_{0.0};
  std::atomic<double> latest_mujoco_contact_msg_time_{-1.0};
  std::atomic<bool> latest_mujoco_contact_active_{false};
  std::atomic<bool> mujoco_first_contact_seen_{false};
  std::atomic<double> mujoco_first_contact_wall_time_{-1.0};
  std::atomic<double> mujoco_first_contact_msg_time_{-1.0};

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

  double clamping_energy_budget_joule_{0.05};
  double energy_budget_margin_joule_{0.005};
  double failsafe_min_pos_stiffness_{5.0};

  double failsafe_pos_damping_scale_{2.5};
  double failsafe_rot_damping_scale_{2.5};
};

}  // namespace cps_controllers
