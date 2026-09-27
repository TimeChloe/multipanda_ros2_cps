#include "cps_controllers/free_motion_external_wrench_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <pluginlib/class_list_macros.hpp>
#include <string>
#include <vector>

namespace {

using Vector7d = Eigen::Matrix<double, 7, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

inline Vector7d arrayToVector7d(const std::array<double, 7>& data) {
  Vector7d out;
  for (size_t i = 0; i < 7; ++i) {
    out(static_cast<int>(i)) = data[i];
  }
  return out;
}

inline Vector6d arrayToVector6d(const std::array<double, 6>& data) {
  Vector6d out;
  for (size_t i = 0; i < 6; ++i) {
    out(static_cast<int>(i)) = data[i];
  }
  return out;
}

}  // namespace

namespace cps_controllers {

FreeMotionExternalWrenchController::TrajectoryType
FreeMotionExternalWrenchController::parse_trajectory_type(const std::string& name) {
  if (name == "joint_sine") {
    return TrajectoryType::kJointSine;
  }
  if (name == "cartesian_line") {
    return TrajectoryType::kCartesianLine;
  }
  if (name == "cartesian_lissajous") {
    return TrajectoryType::kCartesianLissajous;
  }
  return TrajectoryType::kConstant;
}

controller_interface::InterfaceConfiguration
FreeMotionExternalWrenchController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
FreeMotionExternalWrenchController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto& name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(name);
  }
  return config;
}

CallbackReturn FreeMotionExternalWrenchController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");

    auto_declare<std::vector<double>>(
        "joint_damping",
        std::vector<double>{0.3, 0.3, 0.25, 0.20, 0.12, 0.10, 0.08});
    auto_declare<double>("velocity_filter_alpha", 0.99);

    auto_declare<bool>("estimate_bias", true);
    auto_declare<double>("bias_estimation_duration", 3.0);
    auto_declare<double>("bias_velocity_threshold", 0.02);

    auto_declare<double>("publish_rate", 100.0);

    auto_declare<bool>("enable_csv_logging", true);
    auto_declare<std::string>(
        "csv_log_path",
        "/home/developer/multipanda_ws/src/data_log/free_motion_external_wrench_auto.csv");

    auto_declare<bool>("auto_run_experiments", true);

    auto_declare<double>("baseline_duration", 10.0);
    auto_declare<double>("joint_sine_duration", 30.0);
    auto_declare<double>("cartesian_line_duration", 30.0);
    auto_declare<double>("cartesian_lissajous_duration", 60.0);

    auto_declare<std::vector<int64_t>>("joint_sine_indices", std::vector<int64_t>{3, 4, 5});

    auto_declare<double>("default_joint_amplitude", 0.03);
    auto_declare<double>("default_joint_frequency", 0.10);

    auto_declare<double>("default_cartesian_amplitude_x", 0.03);
    auto_declare<double>("default_cartesian_amplitude_y", 0.03);
    auto_declare<double>("default_cartesian_amplitude_z", 0.02);
    auto_declare<double>("default_cartesian_frequency", 0.10);

    auto_declare<double>("default_joint_track_stiffness", 3.0);
    auto_declare<double>("default_cartesian_track_stiffness", 15.0);
    auto_declare<double>("default_cartesian_track_damping", 6.0);

    auto_declare<std::string>("trajectory_type", "constant");
    auto_declare<double>("traj_joint_amplitude", 0.05);
    auto_declare<double>("traj_joint_frequency", 0.2);
    auto_declare<int>("traj_joint_index", 0);
    auto_declare<double>("traj_cartesian_amplitude_x", 0.03);
    auto_declare<double>("traj_cartesian_amplitude_y", 0.03);
    auto_declare<double>("traj_cartesian_amplitude_z", 0.02);
    auto_declare<double>("traj_cartesian_frequency", 0.2);
    auto_declare<double>("joint_track_stiffness", 5.0);
    auto_declare<double>("cartesian_track_stiffness", 20.0);
    auto_declare<double>("cartesian_track_damping", 8.0);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage: %s\n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

void FreeMotionExternalWrenchController::build_experiment_schedule() {
  experiment_segments_.clear();

  if (!auto_run_experiments_) {
    ExperimentSegment seg;
    seg.type = manual_trajectory_type_;
    seg.duration = std::numeric_limits<double>::infinity();
    seg.joint_index = manual_traj_joint_index_;
    seg.joint_amplitude = manual_traj_joint_amplitude_;
    seg.joint_frequency = manual_traj_joint_frequency_;
    seg.cartesian_amplitude_x = manual_traj_cartesian_amplitude_x_;
    seg.cartesian_amplitude_y = manual_traj_cartesian_amplitude_y_;
    seg.cartesian_amplitude_z = manual_traj_cartesian_amplitude_z_;
    seg.cartesian_frequency = manual_traj_cartesian_frequency_;
    seg.joint_track_stiffness = manual_joint_track_stiffness_;
    seg.cartesian_track_stiffness = manual_cartesian_track_stiffness_;
    seg.cartesian_track_damping = manual_cartesian_track_damping_;
    seg.name = "manual_mode";
    experiment_segments_.push_back(seg);
    return;
  }

  ExperimentSegment baseline_seg;
  baseline_seg.type = TrajectoryType::kConstant;
  baseline_seg.duration = baseline_duration_;
  baseline_seg.name = "baseline_constant";
  experiment_segments_.push_back(baseline_seg);

  for (const auto idx64 : joint_sine_indices_) {
    const int idx = static_cast<int>(idx64);
    ExperimentSegment seg;
    seg.type = TrajectoryType::kJointSine;
    seg.duration = joint_sine_duration_;
    seg.joint_index = idx;
    seg.joint_amplitude = default_joint_amplitude_;
    seg.joint_frequency = default_joint_frequency_;
    seg.joint_track_stiffness = default_joint_track_stiffness_;
    seg.name = "joint_sine_j" + std::to_string(idx + 1);
    experiment_segments_.push_back(seg);
  }

  ExperimentSegment line_seg;
  line_seg.type = TrajectoryType::kCartesianLine;
  line_seg.duration = cartesian_line_duration_;
  line_seg.cartesian_amplitude_x = 0.0;
  line_seg.cartesian_amplitude_y = 0.0;
  line_seg.cartesian_amplitude_z = default_cartesian_amplitude_z_;
  line_seg.cartesian_frequency = default_cartesian_frequency_;
  line_seg.cartesian_track_stiffness = default_cartesian_track_stiffness_;
  line_seg.cartesian_track_damping = default_cartesian_track_damping_;
  line_seg.name = "cartesian_line_z";
  experiment_segments_.push_back(line_seg);

  ExperimentSegment lis_seg;
  lis_seg.type = TrajectoryType::kCartesianLissajous;
  lis_seg.duration = cartesian_lissajous_duration_;
  lis_seg.cartesian_amplitude_x = default_cartesian_amplitude_x_;
  lis_seg.cartesian_amplitude_y = default_cartesian_amplitude_y_;
  lis_seg.cartesian_amplitude_z = default_cartesian_amplitude_z_;
  lis_seg.cartesian_frequency = default_cartesian_frequency_;
  lis_seg.cartesian_track_stiffness = default_cartesian_track_stiffness_;
  lis_seg.cartesian_track_damping = default_cartesian_track_damping_;
  lis_seg.name = "cartesian_lissajous";
  experiment_segments_.push_back(lis_seg);
}

void FreeMotionExternalWrenchController::update_active_segment() {
  if (experiment_segments_.empty()) {
    experiment_finished_ = true;
    current_segment_index_ = -1;
    current_segment_elapsed_time_ = 0.0;
    active_trajectory_type_ = TrajectoryType::kConstant;
    return;
  }

  double accumulated = 0.0;
  int found_index = -1;
  double local_time = 0.0;

  for (size_t i = 0; i < experiment_segments_.size(); ++i) {
    const double end_time = accumulated + experiment_segments_[i].duration;
    if (experiment_elapsed_time_ < end_time) {
      found_index = static_cast<int>(i);
      local_time = experiment_elapsed_time_ - accumulated;
      break;
    }
    accumulated = end_time;
  }

  if (found_index < 0) {
    experiment_finished_ = true;
    current_segment_index_ = static_cast<int>(experiment_segments_.size()) - 1;
    current_segment_elapsed_time_ = experiment_segments_.back().duration;
    active_trajectory_type_ = TrajectoryType::kConstant;

    active_joint_index_ = 0;
    active_joint_amplitude_ = 0.0;
    active_joint_frequency_ = 0.0;

    active_cartesian_amplitude_x_ = 0.0;
    active_cartesian_amplitude_y_ = 0.0;
    active_cartesian_amplitude_z_ = 0.0;
    active_cartesian_frequency_ = 0.0;

    active_joint_track_stiffness_ = 0.0;
    active_cartesian_track_stiffness_ = 0.0;
    active_cartesian_track_damping_ = 0.0;
    return;
  }

  experiment_finished_ = false;
  current_segment_index_ = found_index;
  current_segment_elapsed_time_ = local_time;

  const auto& seg = experiment_segments_[current_segment_index_];
  active_trajectory_type_ = seg.type;

  active_joint_index_ = seg.joint_index;
  active_joint_amplitude_ = seg.joint_amplitude;
  active_joint_frequency_ = seg.joint_frequency;

  active_cartesian_amplitude_x_ = seg.cartesian_amplitude_x;
  active_cartesian_amplitude_y_ = seg.cartesian_amplitude_y;
  active_cartesian_amplitude_z_ = seg.cartesian_amplitude_z;
  active_cartesian_frequency_ = seg.cartesian_frequency;

  active_joint_track_stiffness_ = seg.joint_track_stiffness;
  active_cartesian_track_stiffness_ = seg.cartesian_track_stiffness;
  active_cartesian_track_damping_ = seg.cartesian_track_damping;
}

CallbackReturn FreeMotionExternalWrenchController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();

    const auto damping = get_node()->get_parameter("joint_damping").as_double_array();
    if (damping.size() != static_cast<size_t>(kNumJoints)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "joint_damping must have size %d, got %zu",
                   kNumJoints, damping.size());
      return CallbackReturn::ERROR;
    }
    for (int i = 0; i < kNumJoints; ++i) {
      joint_damping_(i) = damping[i];
    }

    velocity_filter_alpha_ =
        get_node()->get_parameter("velocity_filter_alpha").as_double();

    estimate_bias_ =
        get_node()->get_parameter("estimate_bias").as_bool();
    bias_estimation_duration_ =
        get_node()->get_parameter("bias_estimation_duration").as_double();
    bias_velocity_threshold_ =
        get_node()->get_parameter("bias_velocity_threshold").as_double();

    publish_rate_ =
        get_node()->get_parameter("publish_rate").as_double();

    enable_csv_logging_ =
        get_node()->get_parameter("enable_csv_logging").as_bool();
    csv_log_path_ =
        get_node()->get_parameter("csv_log_path").as_string();

    auto_run_experiments_ =
        get_node()->get_parameter("auto_run_experiments").as_bool();

    baseline_duration_ =
        get_node()->get_parameter("baseline_duration").as_double();
    joint_sine_duration_ =
        get_node()->get_parameter("joint_sine_duration").as_double();
    cartesian_line_duration_ =
        get_node()->get_parameter("cartesian_line_duration").as_double();
    cartesian_lissajous_duration_ =
        get_node()->get_parameter("cartesian_lissajous_duration").as_double();

    joint_sine_indices_ =
        get_node()->get_parameter("joint_sine_indices").as_integer_array();

    default_joint_amplitude_ =
        get_node()->get_parameter("default_joint_amplitude").as_double();
    default_joint_frequency_ =
        get_node()->get_parameter("default_joint_frequency").as_double();

    default_cartesian_amplitude_x_ =
        get_node()->get_parameter("default_cartesian_amplitude_x").as_double();
    default_cartesian_amplitude_y_ =
        get_node()->get_parameter("default_cartesian_amplitude_y").as_double();
    default_cartesian_amplitude_z_ =
        get_node()->get_parameter("default_cartesian_amplitude_z").as_double();
    default_cartesian_frequency_ =
        get_node()->get_parameter("default_cartesian_frequency").as_double();

    default_joint_track_stiffness_ =
        get_node()->get_parameter("default_joint_track_stiffness").as_double();
    default_cartesian_track_stiffness_ =
        get_node()->get_parameter("default_cartesian_track_stiffness").as_double();
    default_cartesian_track_damping_ =
        get_node()->get_parameter("default_cartesian_track_damping").as_double();

    manual_trajectory_type_ = parse_trajectory_type(
        get_node()->get_parameter("trajectory_type").as_string());
    manual_traj_joint_amplitude_ =
        get_node()->get_parameter("traj_joint_amplitude").as_double();
    manual_traj_joint_frequency_ =
        get_node()->get_parameter("traj_joint_frequency").as_double();
    manual_traj_joint_index_ =
        static_cast<int>(get_node()->get_parameter("traj_joint_index").as_int());
    manual_traj_cartesian_amplitude_x_ =
        get_node()->get_parameter("traj_cartesian_amplitude_x").as_double();
    manual_traj_cartesian_amplitude_y_ =
        get_node()->get_parameter("traj_cartesian_amplitude_y").as_double();
    manual_traj_cartesian_amplitude_z_ =
        get_node()->get_parameter("traj_cartesian_amplitude_z").as_double();
    manual_traj_cartesian_frequency_ =
        get_node()->get_parameter("traj_cartesian_frequency").as_double();
    manual_joint_track_stiffness_ =
        get_node()->get_parameter("joint_track_stiffness").as_double();
    manual_cartesian_track_stiffness_ =
        get_node()->get_parameter("cartesian_track_stiffness").as_double();
    manual_cartesian_track_damping_ =
        get_node()->get_parameter("cartesian_track_damping").as_double();

    for (const auto idx64 : joint_sine_indices_) {
      const int idx = static_cast<int>(idx64);
      if (idx < 0 || idx >= kNumJoints) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "joint_sine_indices contains invalid index %d", idx);
        return CallbackReturn::ERROR;
      }
    }

    if (manual_traj_joint_index_ < 0 || manual_traj_joint_index_ >= kNumJoints) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "traj_joint_index must be in [0, %d], got %d",
                   kNumJoints - 1, manual_traj_joint_index_);
      return CallbackReturn::ERROR;
    }

    build_experiment_schedule();

    franka_robot_model_ =
        std::make_unique<franka_semantic_components::FrankaRobotModel>(
            franka_semantic_components::FrankaRobotModel(
                arm_id_ + "/robot_model", arm_id_));

    wrench_raw_pub_ =
        get_node()->create_publisher<geometry_msgs::msg::WrenchStamped>(
            "free_motion_external_wrench/raw_base", 10);

    wrench_debiased_pub_ =
        get_node()->create_publisher<geometry_msgs::msg::WrenchStamped>(
            "free_motion_external_wrench/debiased_base", 10);

    tau_ext_pub_ =
        get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
            "free_motion_external_wrench/tau_ext_hat_filtered", 10);

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Exception in on_configure: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn FreeMotionExternalWrenchController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  experiment_elapsed_time_ = 0.0;
  experiment_finished_ = false;
  current_segment_index_ = -1;
  current_segment_elapsed_time_ = 0.0;
  publish_accumulator_ = 0.0;
  csv_flush_counter_ = 0;

  const auto* robot_state = franka_robot_model_->getRobotState();
  if (robot_state == nullptr) {
    RCLCPP_ERROR(get_node()->get_logger(), "Robot state pointer is null.");
    return CallbackReturn::ERROR;
  }

  q_ = arrayToVector7d(robot_state->q);
  dq_ = arrayToVector7d(robot_state->dq);
  dq_filtered_ = dq_;

  q_start_ = q_;
  q_ref_ = q_;

  const Eigen::Map<const Matrix4d> pose(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  p_start_ = pose.block<3, 1>(0, 3);
  R_start_ = pose.block<3, 3>(0, 0);
  p_ref_ = p_start_;

  wrench_raw_base_ = arrayToVector6d(robot_state->O_F_ext_hat_K);
  wrench_raw_k_ = arrayToVector6d(robot_state->K_F_ext_hat_K);
  tau_ext_hat_filtered_ = arrayToVector7d(robot_state->tau_ext_hat_filtered);

  wrench_bias_.setZero();
  wrench_bias_sum_.setZero();
  wrench_debiased_base_ = wrench_raw_base_;
  bias_elapsed_time_ = 0.0;
  bias_sample_count_ = 0;

  update_active_segment();

  if (enable_csv_logging_) {
    csv_log_file_.open(csv_log_path_, std::ios::out | std::ios::trunc);
    if (!csv_log_file_.is_open()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to open csv log file: %s", csv_log_path_.c_str());
      return CallbackReturn::ERROR;
    }

    csv_log_file_ << std::fixed << std::setprecision(9);
    csv_log_file_
        << "time_sec,experiment_time_sec,segment_time_sec,segment_index,traj_mode,segment_name,"
        << "fx_raw,fy_raw,fz_raw,tx_raw,ty_raw,tz_raw,"
        << "fx_debias,fy_debias,fz_debias,tx_debias,ty_debias,tz_debias,"
        << "force_norm,torque_norm,"
        << "fx_k,fy_k,fz_k,tx_k,ty_k,tz_k,"
        << "ref_px,ref_py,ref_pz,"
        << "ref_q1,ref_q2,ref_q3,ref_q4,ref_q5,ref_q6,ref_q7,"
        << "tau_ext_1,tau_ext_2,tau_ext_3,tau_ext_4,tau_ext_5,tau_ext_6,tau_ext_7,"
        << "q1,q2,q3,q4,q5,q6,q7,"
        << "dq1,dq2,dq3,dq4,dq5,dq6,dq7\n";

    RCLCPP_INFO(get_node()->get_logger(),
                "CSV logging enabled: %s", csv_log_path_.c_str());
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn FreeMotionExternalWrenchController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (csv_log_file_.is_open()) {
    csv_log_file_.flush();
    csv_log_file_.close();
  }

  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

void FreeMotionExternalWrenchController::update_bias_estimation(
    double dt, const Vector6d& wrench_raw) {
  if (!estimate_bias_) {
    wrench_bias_.setZero();
    return;
  }

  if (bias_elapsed_time_ >= bias_estimation_duration_) {
    return;
  }

  if (dq_filtered_.norm() < bias_velocity_threshold_) {
    wrench_bias_sum_ += wrench_raw;
    ++bias_sample_count_;
    wrench_bias_ = wrench_bias_sum_ / static_cast<double>(bias_sample_count_);
    bias_elapsed_time_ += dt;
  }
}

void FreeMotionExternalWrenchController::update_reference(double segment_time) {
  q_ref_ = q_start_;
  p_ref_ = p_start_;

  const double wc = 2.0 * M_PI * active_cartesian_frequency_;

  switch (active_trajectory_type_) {
    case TrajectoryType::kConstant:
      break;

    case TrajectoryType::kJointSine: {
      const double wj = 2.0 * M_PI * active_joint_frequency_;
      q_ref_(active_joint_index_) =
          q_start_(active_joint_index_) +
          active_joint_amplitude_ * std::sin(wj * segment_time);
      break;
    }

    case TrajectoryType::kCartesianLine: {
      p_ref_(2) =
          p_start_(2) + active_cartesian_amplitude_z_ * std::sin(wc * segment_time);
      break;
    }

    case TrajectoryType::kCartesianLissajous: {
      p_ref_(0) =
          p_start_(0) + active_cartesian_amplitude_x_ * std::sin(wc * segment_time);
      p_ref_(1) =
          p_start_(1) + active_cartesian_amplitude_y_ * std::sin(2.0 * wc * segment_time + M_PI / 2.0);
      p_ref_(2) =
          p_start_(2) + active_cartesian_amplitude_z_ * std::sin(0.5 * wc * segment_time);
      break;
    }
  }
}

FreeMotionExternalWrenchController::Vector7d
FreeMotionExternalWrenchController::compute_trajectory_torque() {
  Vector7d tau_traj = Vector7d::Zero();

  if (experiment_finished_) {
    return tau_traj;
  }

  if (active_trajectory_type_ == TrajectoryType::kConstant) {
    return tau_traj;
  }

  if (active_trajectory_type_ == TrajectoryType::kJointSine) {
    tau_traj = active_joint_track_stiffness_ * (q_ref_ - q_);
    return tau_traj;
  }

  const Eigen::Map<const Matrix4d> pose(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  const Vector3d p_cur = pose.block<3, 1>(0, 3);

  const Eigen::Map<const Matrix67d> jacobian(
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector).data());
  const Vector3d v_cur = jacobian.topRows<3>() * dq_;

  const Vector3d f_cmd =
      active_cartesian_track_stiffness_ * (p_ref_ - p_cur) -
      active_cartesian_track_damping_ * v_cur;

  tau_traj = jacobian.topRows<3>().transpose() * f_cmd;
  return tau_traj;
}

void FreeMotionExternalWrenchController::publish_msgs(const rclcpp::Time& stamp) {
  geometry_msgs::msg::WrenchStamped raw_msg;
  raw_msg.header.stamp = stamp;
  raw_msg.header.frame_id = arm_id_ + "_link0";
  raw_msg.wrench.force.x = wrench_raw_base_(0);
  raw_msg.wrench.force.y = wrench_raw_base_(1);
  raw_msg.wrench.force.z = wrench_raw_base_(2);
  raw_msg.wrench.torque.x = wrench_raw_base_(3);
  raw_msg.wrench.torque.y = wrench_raw_base_(4);
  raw_msg.wrench.torque.z = wrench_raw_base_(5);
  wrench_raw_pub_->publish(raw_msg);

  geometry_msgs::msg::WrenchStamped debias_msg;
  debias_msg.header = raw_msg.header;
  debias_msg.wrench.force.x = wrench_debiased_base_(0);
  debias_msg.wrench.force.y = wrench_debiased_base_(1);
  debias_msg.wrench.force.z = wrench_debiased_base_(2);
  debias_msg.wrench.torque.x = wrench_debiased_base_(3);
  debias_msg.wrench.torque.y = wrench_debiased_base_(4);
  debias_msg.wrench.torque.z = wrench_debiased_base_(5);
  wrench_debiased_pub_->publish(debias_msg);

  std_msgs::msg::Float64MultiArray tau_msg;
  tau_msg.data.resize(kNumJoints);
  for (int i = 0; i < kNumJoints; ++i) {
    tau_msg.data[i] = tau_ext_hat_filtered_(i);
  }
  tau_ext_pub_->publish(tau_msg);
}

controller_interface::return_type
FreeMotionExternalWrenchController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {
  const double dt = std::max(period.seconds(), kMinDt);
  experiment_elapsed_time_ += dt;

  update_active_segment();

  const auto* robot_state = franka_robot_model_->getRobotState();
  if (robot_state == nullptr) {
    RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(),
                          2000, "Robot state pointer is null.");
    return controller_interface::return_type::ERROR;
  }

  q_ = arrayToVector7d(robot_state->q);
  dq_ = arrayToVector7d(robot_state->dq);
  tau_ext_hat_filtered_ = arrayToVector7d(robot_state->tau_ext_hat_filtered);
  wrench_raw_base_ = arrayToVector6d(robot_state->O_F_ext_hat_K);
  wrench_raw_k_ = arrayToVector6d(robot_state->K_F_ext_hat_K);

  dq_filtered_ =
      (1.0 - velocity_filter_alpha_) * dq_filtered_ + velocity_filter_alpha_ * dq_;

  update_bias_estimation(dt, wrench_raw_base_);
  wrench_debiased_base_ = wrench_raw_base_ - wrench_bias_;

  update_reference(current_segment_elapsed_time_);
  const Vector7d tau_traj = compute_trajectory_torque();
  const Vector7d coriolis =
      arrayToVector7d(franka_robot_model_->getCoriolisForceVector());

  const Vector7d tau_cmd =
      coriolis - joint_damping_.cwiseProduct(dq_filtered_) + tau_traj;

  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }

  publish_accumulator_ += dt;
  if (publish_accumulator_ >= 1.0 / std::max(1.0, publish_rate_)) {
    publish_msgs(time);
    publish_accumulator_ = 0.0;
  }

  if (enable_csv_logging_ && csv_log_file_.is_open()) {
    const double force_norm = wrench_debiased_base_.head<3>().norm();
    const double torque_norm = wrench_debiased_base_.tail<3>().norm();
    const int traj_mode = static_cast<int>(active_trajectory_type_);
    const std::string segment_name =
        (current_segment_index_ >= 0 &&
         current_segment_index_ < static_cast<int>(experiment_segments_.size()))
            ? experiment_segments_[current_segment_index_].name
            : "none";

    csv_log_file_ << std::fixed << std::setprecision(9)
                  << time.seconds() << ","
                  << experiment_elapsed_time_ << ","
                  << current_segment_elapsed_time_ << ","
                  << current_segment_index_ << ","
                  << traj_mode << ","
                  << segment_name << ","
                  << wrench_raw_base_(0) << "," << wrench_raw_base_(1) << "," << wrench_raw_base_(2) << ","
                  << wrench_raw_base_(3) << "," << wrench_raw_base_(4) << "," << wrench_raw_base_(5) << ","
                  << wrench_debiased_base_(0) << "," << wrench_debiased_base_(1) << "," << wrench_debiased_base_(2) << ","
                  << wrench_debiased_base_(3) << "," << wrench_debiased_base_(4) << "," << wrench_debiased_base_(5) << ","
                  << force_norm << "," << torque_norm << ","
                  << wrench_raw_k_(0) << "," << wrench_raw_k_(1) << "," << wrench_raw_k_(2) << ","
                  << wrench_raw_k_(3) << "," << wrench_raw_k_(4) << "," << wrench_raw_k_(5) << ","
                  << p_ref_(0) << "," << p_ref_(1) << "," << p_ref_(2) << ",";

    for (int i = 0; i < kNumJoints; ++i) {
      csv_log_file_ << q_ref_(i) << ",";
    }
    for (int i = 0; i < kNumJoints; ++i) {
      csv_log_file_ << tau_ext_hat_filtered_(i) << ",";
    }
    for (int i = 0; i < kNumJoints; ++i) {
      csv_log_file_ << q_(i) << ",";
    }
    for (int i = 0; i < kNumJoints; ++i) {
      csv_log_file_ << dq_(i);
      if (i != kNumJoints - 1) {
        csv_log_file_ << ",";
      }
    }
    csv_log_file_ << "\n";

    if ((++csv_flush_counter_ % 200) == 0) {
      csv_log_file_.flush();
    }
  }

  return controller_interface::return_type::OK;
}

}  // namespace cps_controllers

PLUGINLIB_EXPORT_CLASS(
    cps_controllers::FreeMotionExternalWrenchController,
    controller_interface::ControllerInterface)