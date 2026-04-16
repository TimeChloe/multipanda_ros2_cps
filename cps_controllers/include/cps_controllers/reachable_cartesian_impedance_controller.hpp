#pragma once

#include <fstream>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

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

struct MonitorResult {
  bool contact_possible_nominal{false};
  bool contact_possible_hybrid{false};
  bool unsafe_contact_nominal{false};
  bool unsafe_contact_hybrid{false};
  bool predicted_trigger{false};

  double plane_distance_now{0.0};
  double plane_distance_min_nominal{0.0};

  double m_eff_n{0.0};
  double v_n_now{0.0};
  double Tn_now{0.0};
  double v_safe{0.0};

  // First predicted nominal contact sample
  bool nominal_contact_sample_found{false};
  double nominal_contact_time{0.0};
  double nominal_contact_distance{0.0};
  double v_n_contact_nominal{0.0};
  double Tn_contact_nominal{0.0};

  // Worst-case hybrid candidate over the whole future horizon
  bool worst_case_candidate_found{false};
  double worst_case_candidate_time{0.0};
  double worst_case_plane_distance_at_candidate{0.0};
  double worst_case_nominal_forward_progress{0.0};
  double worst_case_e_n{0.0};
  double worst_case_v_n{0.0};
  double worst_case_E_fs_1d{0.0};
  double worst_case_tau_safe{0.0};
  double worst_case_delta_n_fs{0.0};
  double worst_case_hybrid_forward_reach{0.0};
  double worst_case_plane_margin_after_hybrid{0.0};

  // Safety-function view
  double h_geom{0.0};
  double h_nominal_energy{0.0};
  double h_failsafe_energy{0.0};

  // Current failsafe storage
  double e_n_now_fs{0.0};
  double v_n_now_fs{0.0};
  double V_fs_now{0.0};
  double V_fs_dot_est{0.0};
};

class ReachableCartesianImpedanceController
    : public controller_interface::ControllerInterface {
 public:
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr int kNumJoints = 7;

  // --------------------------------------------------------------------------
  // helpers
  // --------------------------------------------------------------------------
  void updateRuntimeGains(double dt);

  void buildReference(double t,
                      Vector3d& desired_position_cur,
                      Quaterniond& desired_orientation_cur,
                      Vector3d& desired_linear_velocity_cur,
                      Vector3d& desired_linear_acceleration_cur);

  void enterFailsafe(double t_now,
                     const Vector3d& desired_position_cur,
                     const Quaterniond& desired_orientation_cur);

  void leaveFailsafe(double t_now);

  MonitorResult runSafetyMonitor(double dt,
                                 double t,
                                 const Vector3d& current_position,
                                 const Vector3d& desired_position_cur,
                                 const Vector6d& ee_twist,
                                 const Vector3d& desired_linear_velocity_cur,
                                 const Vector3d& desired_linear_acceleration_cur,
                                 const Matrix7d& inertia,
                                 const Matrix37d& Jv,
                                 const Vector3d& plane_normal,
                                 const Vector3d& plane_point);

  // --------------------------------------------------------------------------
  // logging
  // --------------------------------------------------------------------------
  bool enable_error_logging_{false};
  std::string error_log_path_{
      "/home/developer/multipanda_ws/src/data_log/cartesian_impedance_failsafe_validation.csv"};
  std::ofstream error_log_file_;
  std::size_t log_write_counter_{0};

  // --------------------------------------------------------------------------
  // basic configuration
  // --------------------------------------------------------------------------
  std::string arm_id_;
  std::string reference_trajectory_type_{"line"};

  bool use_constant_reference_{false};

  // --------------------------------------------------------------------------
  // robot / model handles
  // --------------------------------------------------------------------------
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  // --------------------------------------------------------------------------
  // runtime state
  // --------------------------------------------------------------------------
  rclcpp::Time start_time_;

  Quaterniond desired_orientation_;
  Vector3d desired_position_;
  Vector7d desired_qn_;

  // frozen reference used in failsafe
  Quaterniond frozen_desired_orientation_;
  Vector3d frozen_desired_position_;

  // mode
  SafetyMode mode_{SafetyMode::kNominal};
  double failsafe_start_time_sec_{-1.0};

  // nominal-time pause bookkeeping
  double failsafe_enter_wall_time_sec_{-1.0};
  double paused_nominal_time_sec_{0.0};

  // gains
  Matrix6d K_nominal_{Matrix6d::Zero()};
  Matrix6d D_nominal_{Matrix6d::Zero()};
  Matrix6d K_f_target_{Matrix6d::Zero()};
  Matrix6d D_f_target_{Matrix6d::Zero()};
  Matrix6d K_runtime_{Matrix6d::Zero()};
  Matrix6d D_runtime_{Matrix6d::Zero()};

  double gain_filter_tau_{0.03};
  double n_stiffness_{0.0};
  bool disable_nullspace_in_failsafe_{true};

  // --------------------------------------------------------------------------
  // safety monitor configuration
  // --------------------------------------------------------------------------
  bool enable_safety_monitor_{true};
  bool auto_enter_failsafe_{false};

  double safe_collision_energy_joule_{0.10};
  double ee_collision_radius_{0.3};

  double monitor_nominal_horizon_sec_{0.02};
  int monitor_nominal_steps_{10};

  Vector3d human_plane_normal_{Vector3d(0.0, 0.0, 1.0)};
  Vector3d human_plane_point_{Vector3d(0.0, 0.0, 0.2)};

  // hysteresis for returning from failsafe to nominal
  double return_to_nominal_geom_margin_{0.005};
  double return_to_nominal_energy_margin_{0.02};

  // --------------------------------------------------------------------------
  // current failsafe storage tracking
  // --------------------------------------------------------------------------
  double prev_V_fs_{0.0};
  bool prev_V_fs_valid_{false};
  double last_e_n_fs_{0.0};
  double last_v_n_fs_{0.0};

  // --------------------------------------------------------------------------
  // profiling statistics
  // --------------------------------------------------------------------------
  int profiling_stats_print_period_{1000};

  std::size_t loop_counter_{0};

  double exec_sum_ms_{0.0};
  double exec_min_ms_{1e9};
  double exec_max_ms_{0.0};
};

}  // namespace cps_controllers