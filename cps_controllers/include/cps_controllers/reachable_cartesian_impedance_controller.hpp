#pragma once

#include <fstream>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

  bool nominal_contact_sample_found{false};
  double nominal_contact_time{0.0};
  double nominal_contact_distance{0.0};
  double v_n_contact_nominal{0.0};
  double Tn_contact_nominal{0.0};

  bool worst_case_candidate_found{false};
  double worst_case_candidate_time{0.0};
  double worst_case_plane_distance_at_candidate{0.0};
  double worst_case_nominal_forward_progress{0.0};

  double worst_case_v_n_fs_ub{0.0};
  double worst_case_Tn_fs_ub{0.0};
  double worst_case_a_pos{0.0};
  double worst_case_a_brake{0.0};
  double worst_case_a_net{0.0};

  double h_geom{0.0};
  double h_nominal_energy{0.0};
  double h_failsafe_energy{0.0};

  double v_n_now_fs{0.0};
  double Tn_now_fs{0.0};
  double Tn_dot_est{0.0};

  Vector3d nominal_contact_point_world{Vector3d::Zero()};
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

  void updateRuntimeGains(double dt);

  void buildReference(double nominal_time,
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
                                 const Vector3d& sphere_center);

  double computeNormalStiffnessUpperBound() const;
  double computeNormalDampingLowerBound() const;

  double computeConservativeNormalAccelPositiveBound(double m_eff_n,
                                                     double e_n_abs,
                                                     double v_n_abs) const;

  double computeConservativeNormalBrakeAccelLowerBound(double m_eff_n,
                                                       double v_n_abs) const;

  double distanceToSweptHandRegion(const Vector3d& x,
                                   const Vector3d& plane_normal,
                                   const Vector3d& sphere_center) const;

  void publishRvizDiagnostics(double wall_time,
                              const Vector3d& current_position,
                              const Vector3d& desired_position_cur,
                              const Vector6d& ee_twist,
                              const MonitorResult& monitor);

  bool enable_error_logging_{false};
  std::string error_log_path_{
      "/home/developer/multipanda_ws/src/data_log/reachable_cartesian_impedance_validation.csv"};
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

  Quaterniond frozen_desired_orientation_;
  Vector3d frozen_desired_position_;

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

  double gain_filter_tau_{0.03};
  double n_stiffness_{0.0};
  bool disable_nullspace_in_failsafe_{true};

  bool enable_safety_monitor_{true};
  bool auto_enter_failsafe_{false};

  double safe_collision_energy_joule_{0.10};
  double ee_collision_radius_{0.04};

  double monitor_nominal_horizon_sec_{0.03};
  int monitor_nominal_steps_{10};
  int monitor_decimation_{10};

  // Projection direction still used for kinetic-energy monitoring
  Vector3d human_plane_normal_{Vector3d(0.0, 0.0, 1.0)};

  // Center of spherical human reachable region
  Vector3d human_sphere_center_{Vector3d(0.0, 0.0, 0.2)};

  // Sphere model:
  // effective human region radius = human_motion_radius + human_hand_radius
  double human_motion_radius_{0.10};
  double human_hand_radius_{0.04};

  double return_to_nominal_energy_margin_{0.02};
  double return_to_nominal_speed_threshold_{0.02};
  double return_to_nominal_tndot_threshold_{0.0};

  double return_to_nominal_geom_margin_{0.005};

  double k_rate_limit_{5000.0};
  double d_rate_limit_{500.0};
  double tau_rate_limit_{1000.0};
  double torque_to_accel_gain_{8.0};
  double model_accel_uncertainty_{0.5};
  double stiffness_error_bound_m_{0.02};

  double prev_Tn_fs_{0.0};
  bool prev_Tn_fs_valid_{false};
  double last_v_n_fs_{0.0};

  std::size_t monitor_counter_{0};
  bool last_monitor_result_valid_{false};
  double last_monitor_wall_time_{0.0};
  MonitorResult last_monitor_result_{};

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

  std::size_t loop_counter_{0};
  double exec_sum_ms_{0.0};
  double exec_min_ms_{1e9};
  double exec_max_ms_{0.0};
};

}  // namespace cps_controllers