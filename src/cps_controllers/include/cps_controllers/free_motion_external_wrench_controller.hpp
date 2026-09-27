#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <franka/model.h>
#include <franka/robot_state.h>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace cps_controllers {

class FreeMotionExternalWrenchController
    : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix4d = Eigen::Matrix<double, 4, 4>;
  using Matrix3d = Eigen::Matrix3d;
  using Vector3d = Eigen::Vector3d;
  using Matrix67d = Eigen::Matrix<double, 6, 7>;

  enum class TrajectoryType {
    kConstant = 0,
    kJointSine = 1,
    kCartesianLine = 2,
    kCartesianLissajous = 3
  };

  struct ExperimentSegment {
    TrajectoryType type{TrajectoryType::kConstant};
    double duration{0.0};

    int joint_index{0};
    double joint_amplitude{0.0};
    double joint_frequency{0.0};

    double cartesian_amplitude_x{0.0};
    double cartesian_amplitude_y{0.0};
    double cartesian_amplitude_z{0.0};
    double cartesian_frequency{0.0};

    double joint_track_stiffness{0.0};
    double cartesian_track_stiffness{0.0};
    double cartesian_track_damping{0.0};

    std::string name;
  };

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(
      const rclcpp::Time& time,
      const rclcpp::Duration& period) override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr int kNumJoints = 7;
  static constexpr double kMinDt = 1e-6;

  static TrajectoryType parse_trajectory_type(const std::string& name);

  void publish_msgs(const rclcpp::Time& stamp);
  void update_bias_estimation(double dt, const Vector6d& wrench_raw);

  void build_experiment_schedule();
  void update_active_segment();

  void update_reference(double segment_time);
  Vector7d compute_trajectory_torque();

  std::string arm_id_;

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  Vector7d joint_damping_{Vector7d::Zero()};
  Vector7d dq_filtered_{Vector7d::Zero()};
  double velocity_filter_alpha_{0.99};

  bool estimate_bias_{true};
  double bias_estimation_duration_{3.0};
  double bias_velocity_threshold_{0.02};
  double bias_elapsed_time_{0.0};
  std::size_t bias_sample_count_{0};
  Vector6d wrench_bias_{Vector6d::Zero()};
  Vector6d wrench_bias_sum_{Vector6d::Zero()};

  Vector6d wrench_raw_base_{Vector6d::Zero()};
  Vector6d wrench_debiased_base_{Vector6d::Zero()};
  Vector6d wrench_raw_k_{Vector6d::Zero()};
  Vector7d tau_ext_hat_filtered_{Vector7d::Zero()};
  Vector7d q_{Vector7d::Zero()};
  Vector7d dq_{Vector7d::Zero()};

  Vector7d q_start_{Vector7d::Zero()};
  Vector7d q_ref_{Vector7d::Zero()};

  Vector3d p_start_{Vector3d::Zero()};
  Matrix3d R_start_{Matrix3d::Identity()};
  Vector3d p_ref_{Vector3d::Zero()};

  double publish_rate_{100.0};
  double publish_accumulator_{0.0};

  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_raw_pub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_debiased_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tau_ext_pub_;

  bool enable_csv_logging_{false};
  std::string csv_log_path_;
  std::ofstream csv_log_file_;
  std::size_t csv_flush_counter_{0};

  bool auto_run_experiments_{true};

  double experiment_elapsed_time_{0.0};
  bool experiment_finished_{false};

  int current_segment_index_{-1};
  double current_segment_elapsed_time_{0.0};

  std::vector<ExperimentSegment> experiment_segments_;

  TrajectoryType active_trajectory_type_{TrajectoryType::kConstant};

  int active_joint_index_{0};
  double active_joint_amplitude_{0.0};
  double active_joint_frequency_{0.0};

  double active_cartesian_amplitude_x_{0.0};
  double active_cartesian_amplitude_y_{0.0};
  double active_cartesian_amplitude_z_{0.0};
  double active_cartesian_frequency_{0.0};

  double active_joint_track_stiffness_{0.0};
  double active_cartesian_track_stiffness_{0.0};
  double active_cartesian_track_damping_{0.0};

  double baseline_duration_{10.0};
  double joint_sine_duration_{30.0};
  double cartesian_line_duration_{30.0};
  double cartesian_lissajous_duration_{60.0};

  std::vector<int64_t> joint_sine_indices_;

  double default_joint_amplitude_{0.03};
  double default_joint_frequency_{0.10};

  double default_cartesian_amplitude_x_{0.03};
  double default_cartesian_amplitude_y_{0.03};
  double default_cartesian_amplitude_z_{0.02};
  double default_cartesian_frequency_{0.10};

  double default_joint_track_stiffness_{3.0};
  double default_cartesian_track_stiffness_{15.0};
  double default_cartesian_track_damping_{6.0};

  TrajectoryType manual_trajectory_type_{TrajectoryType::kConstant};

  double manual_traj_joint_amplitude_{0.05};
  double manual_traj_joint_frequency_{0.2};
  int manual_traj_joint_index_{0};

  double manual_traj_cartesian_amplitude_x_{0.03};
  double manual_traj_cartesian_amplitude_y_{0.03};
  double manual_traj_cartesian_amplitude_z_{0.02};
  double manual_traj_cartesian_frequency_{0.2};

  double manual_joint_track_stiffness_{5.0};
  double manual_cartesian_track_stiffness_{20.0};
  double manual_cartesian_track_damping_{8.0};
};

}  // namespace cps_controllers