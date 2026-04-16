#pragma once

#include <array>
#include <fstream>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include "franka_semantic_components/franka_robot_model.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace cps_controllers {

using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;

using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Quaterniond = Eigen::Quaterniond;

class NonlinearImpedanceController
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

  // helpers
  Vector6d computeTaskPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) const;
  Vector3d wrapEulerError(const Vector3d& e) const;
  Eigen::Matrix<double, 6, 7> computeAnalyticJacobian(
      const Eigen::Matrix<double, 6, 7>& J_geo,
      const Eigen::Vector3d& rpy) const;
  Eigen::Matrix<double, 6, 7> computeAnalyticJacobianDotNumerical(
      const Vector7d& q,
      const Vector7d& dq,
      const Eigen::Vector3d& rpy,
      double dt);

  // --------------------------------------------------------------------------
  // logging
  // --------------------------------------------------------------------------
  bool enable_error_logging_{false};
  std::string error_log_path_{
      "/home/developer/multipanda_ws/src/data_log/cartesian_pose_error.csv"};
  std::ofstream error_log_file_;
  std::size_t log_write_counter_{0};

  // --------------------------------------------------------------------------
  // basic configuration
  // --------------------------------------------------------------------------
  std::string arm_id_;
  std::string urdf_model_path_;

  // cached runtime parameters
  std::string reference_trajectory_type_{"lissajous"};

  bool use_constant_reference_{true};
  bool use_nonlinear_feedforward_{true};
  bool use_franka_model_for_dynamics_{true};

  bool use_friction_compensation_{false};
  bool friction_use_reference_velocity_{true};
  double friction_scale_{1.0};

  std::array<double, kNumJoints> friction_coulomb_{};
  std::array<double, kNumJoints> friction_viscous_{};
  std::array<double, kNumJoints> friction_velocity_scale_{
      {0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05}};
  std::array<double, kNumJoints> friction_offset_{};

  // --------------------------------------------------------------------------
  // robot / model handles
  // --------------------------------------------------------------------------
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;
  pinocchio::FrameIndex ee_frame_id_{0};

  // --------------------------------------------------------------------------
  // runtime state
  // --------------------------------------------------------------------------
  rclcpp::Time start_time_;

  Quaterniond desired_orientation_;
  Vector3d desired_position_;
  Vector7d desired_qn_;

  Matrix6d K_m_;
  Matrix6d D_m_;
  double n_stiffness_{10.0};

  // --------------------------------------------------------------------------
  // loop timing statistics
  // --------------------------------------------------------------------------
  int loop_stats_print_period_{1000};
  double loop_dt_warn_threshold_sec_{0.0012};
  double loop_exec_warn_threshold_ms_{1.0};

  std::size_t loop_counter_{0};

  double dt_sum_{0.0};
  double dt_min_{1e9};
  double dt_max_{0.0};

  double exec_sum_ms_{0.0};
  double exec_min_ms_{1e9};
  double exec_max_ms_{0.0};

  std::size_t deadline_miss_count_{0};
  std::size_t exec_overrun_count_{0};


  // --------------------------------------------------------------------------
  // profiling statistics
  // --------------------------------------------------------------------------
  int profiling_stats_print_period_{1000};

  std::size_t profiling_counter_{0};

  double prof_ref_sum_ms_{0.0};
  double prof_model_sum_ms_{0.0};
  double prof_task_sum_ms_{0.0};
  double prof_solve_sum_ms_{0.0};
  double prof_torque_sum_ms_{0.0};
  double prof_log_sum_ms_{0.0};
  double prof_total_sum_ms_{0.0};

  double prof_ref_max_ms_{0.0};
  double prof_model_max_ms_{0.0};
  double prof_task_max_ms_{0.0};
  double prof_solve_max_ms_{0.0};
  double prof_torque_max_ms_{0.0};
  double prof_log_max_ms_{0.0};
  double prof_total_max_ms_{0.0};
};

}  // namespace cps_controllers