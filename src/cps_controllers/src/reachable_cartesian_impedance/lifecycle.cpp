// Copyright (c) 2026
// ROS lifecycle, configuration, model adapters and recording setup.
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>


#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <franka/model.h>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <geometry_msgs/msg/pose_array.hpp>

#include <cps_controllers/reachable_cartesian_impedance_controller.hpp>
#include <cps_human_workspace/human_workspace_message.hpp>
#include <cps_trajectory_generators/reachable_cartesian_trajectory.hpp>

#include "math.hpp"
#include <cps_controllers/reachable_cartesian_impedance/monitor_policy.hpp>

namespace {
constexpr double kMinDt = 1e-6;
constexpr const char* kDefaultErrorLogRootDir =
    "/home/developer/multipanda_ws/src/data_log";
constexpr const char* kDefaultErrorLogFileName =
    "reachable_cartesian_impedance_validation.csv";

using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;
using Matrix37d = Eigen::Matrix<double, 3, 7>;
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;
using CartesianTrajectorySample = cps_trajectory_generators::CartesianTrajectorySample;
using cps_trajectory_generators::loadTrajectoryGeneratorSettings;

std::vector<double> defaultNullspaceHomePoseParameter() {
  return {
      0.0,
      -0.7853981633974483,
      0.0,
      -2.356194490192345,
      0.0,
      1.5707963267948966,
      0.7853981633974483};
}

inline int daysInMonth(int year, int month) {
  static constexpr int kDaysByMonth[] = {
      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month == 2) {
    const bool leap_year =
        ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0);
    return leap_year ? 29 : 28;
  }
  return kDaysByMonth[month - 1];
}

inline int weekdayUtc(int year, int month, int day) {
  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon = month - 1;
  tm.tm_mday = day;
  tm.tm_hour = 12;
  time_t t = timegm(&tm);
  std::tm out{};
  gmtime_r(&t, &out);
  return out.tm_wday;
}

inline int lastSundayOfMonth(int year, int month) {
  int day = daysInMonth(year, month);
  while (weekdayUtc(year, month, day) != 0) {
    --day;
  }
  return day;
}

inline time_t utcTime(int year, int month, int day, int hour) {
  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon = month - 1;
  tm.tm_mday = day;
  tm.tm_hour = hour;
  return timegm(&tm);
}

inline int berlinUtcOffsetMinutes(std::chrono::system_clock::time_point now) {
  const time_t now_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
  gmtime_r(&now_time_t, &utc_tm);
  const int year = utc_tm.tm_year + 1900;

  const time_t dst_start =
      utcTime(year, 3, lastSundayOfMonth(year, 3), 1);
  const time_t dst_end =
      utcTime(year, 10, lastSundayOfMonth(year, 10), 1);

  return (now_time_t >= dst_start && now_time_t < dst_end) ? 120 : 60;
}

inline std::string makeBerlinTimestampForDirectoryName() {
  const auto now = std::chrono::system_clock::now();
  const int utc_offset_minutes = berlinUtcOffsetMinutes(now);
  const auto adjusted_now = now + std::chrono::minutes(utc_offset_minutes);
  const auto adjusted_time_t = std::chrono::system_clock::to_time_t(adjusted_now);
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch())
                          .count() %
                      1000000;

  std::tm tm{};
  gmtime_r(&adjusted_time_t, &tm);

  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y%m%d_%H%M%S")
     << "_" << std::setw(6) << std::setfill('0') << micros;
  return ss.str();
}

inline std::string sanitizedFileNameOrDefault(
    const std::string& file_name,
    const std::string& default_file_name) {
  if (file_name.empty()) {
    return default_file_name;
  }

  const std::filesystem::path path(file_name);
  const std::string sanitized = path.filename().string();
  return sanitized.empty() ? default_file_name : sanitized;
}

}  // namespace

namespace cps_controllers {

class ReachableCartesianImpedanceController::PinocchioJointDynamicsProvider
    final : public cps_safety_monitor::JointDynamicsProvider {
 public:
  PinocchioJointDynamicsProvider(const std::string& urdf_model_path,
                                 const Vector3d& tcp_offset,
                                 const Vector7d& joint_armature)
      : joint_armature_(joint_armature) {
    if (!joint_armature_.allFinite() ||
        (joint_armature_.array() < 0.0).any()) {
      throw std::runtime_error(
          "Prediction joint armature must contain finite nonnegative values");
    }
    pinocchio::urdf::buildModel(urdf_model_path, model_);
    if (model_.nq != 7 || model_.nv != 7) {
      throw std::runtime_error(
          "Reachable monitor URDF must contain exactly seven actuated joints");
    }
    data_ = std::make_unique<pinocchio::Data>(model_);
    frame_id_ = model_.getFrameId("panda_link8");
    if (frame_id_ >= static_cast<pinocchio::FrameIndex>(model_.nframes)) {
      throw std::runtime_error(
          "Reachable monitor URDF does not contain frame panda_link8");
    }
    tcp_offset_ = tcp_offset;

    for (int i = 0; i < 7; ++i) {
      limits_.position_lower(i) = panda_limits::kPositionLower[i];
      limits_.position_upper(i) = panda_limits::kPositionUpper[i];
      limits_.velocity(i) = panda_limits::kVelocity[i];
      limits_.acceleration(i) = panda_limits::kAcceleration[i];
      limits_.torque(i) = panda_limits::kTorque[i];
    }
  }

  bool evaluate(
      const Vector7d& q,
      const Vector7d& dq,
      cps_safety_monitor::JointDynamicsSample* sample) const override {
    if (sample == nullptr || !q.allFinite() || !dq.allFinite()) {
      return false;
    }

    try {
      const Vector7d ddq_zero = Vector7d::Zero();
      pinocchio::forwardKinematics(model_, *data_, q, dq, ddq_zero);
      pinocchio::computeJointJacobians(model_, *data_, q);
      pinocchio::updateFramePlacements(model_, *data_);

      Matrix67d pin_jacobian = Matrix67d::Zero();
      pinocchio::getFrameJacobian(
          model_, *data_, frame_id_,
          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
          pin_jacobian);

      // Pinocchio in this ROS distribution and the controller both use the
      // [linear; angular] 6D ordering.
      const Matrix67d frame_jacobian = pin_jacobian;

      const auto& placement = data_->oMf[frame_id_];
      const Vector3d offset_world = placement.rotation() * tcp_offset_;
      Matrix67d control_jacobian = frame_jacobian;
      control_jacobian.topRows<3>() =
          frame_jacobian.topRows<3>() -
          skewSymmetric(offset_world) * frame_jacobian.bottomRows<3>();

      const pinocchio::Motion frame_velocity =
          pinocchio::getFrameVelocity(
              model_, *data_, frame_id_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      const pinocchio::Motion frame_acceleration =
          pinocchio::getFrameClassicalAcceleration(
              model_, *data_, frame_id_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      const Vector3d omega = frame_velocity.angular();
      const Vector3d alpha = frame_acceleration.angular();
      Vector6d control_jdot_dq = Vector6d::Zero();
      control_jdot_dq.head<3>() =
          frame_acceleration.linear() + alpha.cross(offset_world) +
          omega.cross(omega.cross(offset_world));
      control_jdot_dq.tail<3>() = alpha;

      pinocchio::crba(model_, *data_, q);
      Matrix7d inertia = data_->M;
      inertia.triangularView<Eigen::StrictlyLower>() =
          inertia.transpose().triangularView<Eigen::StrictlyLower>();
      // MuJoCo armature is a constant inertia expressed directly in each
      // hinge coordinate. URDF link inertias do not contain this term.
      inertia.diagonal() += joint_armature_;
      pinocchio::computeCoriolisMatrix(model_, *data_, q, dq);

      sample->control_position =
          placement.translation() + offset_world;
      sample->control_orientation =
          Quaterniond(placement.rotation()).normalized();
      sample->control_jacobian = control_jacobian;
      sample->control_jdot_dq = control_jdot_dq;
      sample->inertia = 0.5 * (inertia + inertia.transpose());
      sample->coriolis = data_->C * dq;
      sample->valid = sample->control_position.allFinite() &&
                      sample->control_jacobian.allFinite() &&
                      sample->control_jdot_dq.allFinite() &&
                      sample->inertia.allFinite() &&
                      sample->coriolis.allFinite();
      return sample->valid;
    } catch (const std::exception&) {
      sample->valid = false;
      return false;
    }
  }

  cps_safety_monitor::JointDynamicsLimits limits() const override {
    return limits_;
  }

 private:
  pinocchio::Model model_;
  mutable std::unique_ptr<pinocchio::Data> data_;
  pinocchio::FrameIndex frame_id_{0};
  Vector3d tcp_offset_{Vector3d::Zero()};
  Vector7d joint_armature_{Vector7d::Zero()};
  cps_safety_monitor::JointDynamicsLimits limits_;
};

class ReachableCartesianImpedanceController::FrankaInterfaceJointDynamicsProvider
    final : public cps_safety_monitor::JointDynamicsProvider {
 public:
  FrankaInterfaceJointDynamicsProvider(
      franka_semantic_components::FrankaRobotModel* robot_model,
      const franka::RobotState& robot_state,
      const Vector3d& tcp_offset)
      : robot_model_(robot_model),
        F_T_EE_(robot_state.F_T_EE),
        EE_T_K_(robot_state.EE_T_K),
        I_total_(robot_state.I_total),
        m_total_(robot_state.m_total),
        F_x_Ctotal_(robot_state.F_x_Ctotal),
        tcp_offset_(tcp_offset) {
    if (robot_model_ == nullptr ||
        !robot_model_->supportsStateDependentEvaluation()) {
      throw std::runtime_error(
          "Franka model backend cannot evaluate arbitrary predicted q/dq");
    }
    for (int i = 0; i < 7; ++i) {
      limits_.position_lower(i) = panda_limits::kPositionLower[i];
      limits_.position_upper(i) = panda_limits::kPositionUpper[i];
      limits_.velocity(i) = panda_limits::kVelocity[i];
      limits_.acceleration(i) = panda_limits::kAcceleration[i];
      limits_.torque(i) = panda_limits::kTorque[i];
    }
  }

  bool evaluate(
      const Vector7d& q,
      const Vector7d& dq,
      cps_safety_monitor::JointDynamicsSample* sample) const override {
    if (sample == nullptr || robot_model_ == nullptr ||
        !q.allFinite() || !dq.allFinite()) {
      return false;
    }

    try {
      std::array<double, 7> q_array{};
      std::array<double, 7> dq_array{};
      std::copy_n(q.data(), 7, q_array.begin());
      std::copy_n(dq.data(), 7, dq_array.begin());

      const auto pose_array = robot_model_->getPoseMatrix(
          franka::Frame::kEndEffector, q_array, F_T_EE_, EE_T_K_);
      const Eigen::Map<const Matrix4d> pose(pose_array.data());
      const Matrix3d rotation = pose.block<3, 3>(0, 0);
      const Vector3d offset_world = rotation * tcp_offset_;

      const auto jacobian_array = robot_model_->getZeroJacobian(
          franka::Frame::kEndEffector, q_array, F_T_EE_, EE_T_K_);
      Matrix67d control_jacobian =
          Eigen::Map<const Matrix67d>(jacobian_array.data());
      control_jacobian.topRows<3>() -=
          skewSymmetric(offset_world) * control_jacobian.bottomRows<3>();

      // libfranka does not expose Jdot*dq. Estimate it by differentiating the
      // same libfranka Jacobian model along dq, so no Pinocchio quantity is
      // mixed into the real-robot rollout.
      constexpr double kJacobianDifferenceTime = 1.0e-4;
      std::array<double, 7> q_plus = q_array;
      std::array<double, 7> q_minus = q_array;
      for (std::size_t i = 0; i < q_plus.size(); ++i) {
        q_plus[i] += kJacobianDifferenceTime * dq_array[i];
        q_minus[i] -= kJacobianDifferenceTime * dq_array[i];
      }
      const auto jacobian_plus_array = robot_model_->getZeroJacobian(
          franka::Frame::kEndEffector, q_plus, F_T_EE_, EE_T_K_);
      const auto jacobian_minus_array = robot_model_->getZeroJacobian(
          franka::Frame::kEndEffector, q_minus, F_T_EE_, EE_T_K_);
      Matrix67d jacobian_plus =
          Eigen::Map<const Matrix67d>(jacobian_plus_array.data());
      Matrix67d jacobian_minus =
          Eigen::Map<const Matrix67d>(jacobian_minus_array.data());

      const auto pose_plus_array = robot_model_->getPoseMatrix(
          franka::Frame::kEndEffector, q_plus, F_T_EE_, EE_T_K_);
      const auto pose_minus_array = robot_model_->getPoseMatrix(
          franka::Frame::kEndEffector, q_minus, F_T_EE_, EE_T_K_);
      const Eigen::Map<const Matrix4d> pose_plus(pose_plus_array.data());
      const Eigen::Map<const Matrix4d> pose_minus(pose_minus_array.data());
      const Vector3d offset_plus =
          pose_plus.block<3, 3>(0, 0) * tcp_offset_;
      const Vector3d offset_minus =
          pose_minus.block<3, 3>(0, 0) * tcp_offset_;
      jacobian_plus.topRows<3>() -=
          skewSymmetric(offset_plus) * jacobian_plus.bottomRows<3>();
      jacobian_minus.topRows<3>() -=
          skewSymmetric(offset_minus) * jacobian_minus.bottomRows<3>();

      const auto inertia_array = robot_model_->getMassMatrix(
          q_array, I_total_, m_total_, F_x_Ctotal_);
      const auto coriolis_array = robot_model_->getCoriolisForceVector(
          q_array, dq_array, I_total_, m_total_, F_x_Ctotal_);

      sample->control_position = pose.block<3, 1>(0, 3) + offset_world;
      sample->control_orientation = Quaterniond(rotation).normalized();
      sample->control_jacobian = control_jacobian;
      sample->control_jdot_dq =
          ((jacobian_plus - jacobian_minus) /
           (2.0 * kJacobianDifferenceTime)) * dq;
      sample->inertia = Eigen::Map<const Matrix7d>(inertia_array.data());
      sample->inertia =
          0.5 * (sample->inertia + sample->inertia.transpose());
      sample->coriolis = Eigen::Map<const Vector7d>(coriolis_array.data());
      sample->valid = sample->control_position.allFinite() &&
                      sample->control_orientation.coeffs().allFinite() &&
                      sample->control_jacobian.allFinite() &&
                      sample->control_jdot_dq.allFinite() &&
                      sample->inertia.allFinite() &&
                      sample->coriolis.allFinite();
      return sample->valid;
    } catch (const std::exception&) {
      sample->valid = false;
      return false;
    }
  }

  cps_safety_monitor::JointDynamicsLimits limits() const override {
    return limits_;
  }

 private:
  franka_semantic_components::FrankaRobotModel* robot_model_{nullptr};
  std::array<double, 16> F_T_EE_{};
  std::array<double, 16> EE_T_K_{};
  std::array<double, 9> I_total_{};
  double m_total_{0.0};
  std::array<double, 3> F_x_Ctotal_{};
  Vector3d tcp_offset_{Vector3d::Zero()};
  cps_safety_monitor::JointDynamicsLimits limits_;
};

ReachableCartesianImpedanceController::~ReachableCartesianImpedanceController() {
  stopSafetyMonitorWorker();
}

// ============================================================================
// Interface configurations
// ============================================================================
controller_interface::InterfaceConfiguration
ReachableCartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i)
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  return config;
}

controller_interface::InterfaceConfiguration
ReachableCartesianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& name : franka_robot_model_->get_state_interface_names())
    config.names.push_back(name);
  return config;
}

// ============================================================================
// on_init
// ============================================================================
CallbackReturn ReachableCartesianImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<bool>("enable_error_logging", true);
    auto_declare<std::string>("error_log_root_dir", kDefaultErrorLogRootDir);
    auto_declare<std::string>("error_log_file_name", kDefaultErrorLogFileName);
    auto_declare<bool>("enable_prediction_logging", true);
    auto_declare<std::string>(
        "prediction_log_file_name",
        "shield_prediction_trajectory.csv");
    auto_declare<int>("control_log_max_queue_size", 16384);
    auto_declare<int>("prediction_log_max_queue_size", 256);
    auto_declare<int>("log_batch_size", 512);
    auto_declare<double>("log_flush_period_sec", 1.0);

    auto_declare<std::string>(
        "cartesian_via_points_topic",
        "cartesian_via_points");
    auto_declare<std::string>(
        "startup_via_points_source",
        "yaml");
    auto_declare<std::string>(
        "cartesian_via_points_action_name",
        "~/follow_cartesian_via_points");
    auto_declare<std::string>(
        "calibration_action_name",
        "~/calibrate_monitored_trajectory");
    auto_declare<double>(
        "cartesian_via_points_action_feedback_period_sec",
        0.1);
    const std::string startup_via_points_source =
        get_node()->get_parameter("startup_via_points_source").as_string();
    if (startup_via_points_source != "action") {
      auto_declare<std::vector<double>>(
          "cartesian_via_points",
          std::vector<double>{std::numeric_limits<double>::quiet_NaN()});
    }

    auto_declare<double>("pos_stiffness", 400.0);
    auto_declare<double>("rot_stiffness", 20.0);
    auto_declare<bool>("enable_nullspace", false);
    auto_declare<double>("n_stiffness", 0.0);
    auto_declare<std::vector<double>>(
        "nullspace_home_pose", defaultNullspaceHomePoseParameter());
    auto_declare<bool>("enable_safety_monitor", true);

    auto_declare<double>("energy_budget_joule", 0.05);
    auto_declare<double>("kinetic_energy_error_bound_joule", 0.0);
    auto_declare<double>("potential_energy_error_bound_joule", 0.0);
    auto_declare<double>(
        "nullspace_potential_energy_error_bound_joule", 0.0);
    auto_declare<bool>("enable_runtime_energy_scaling", true);
    auto_declare<double>("energy_recovery_exit_energy_fraction", 1.0);
    auto_declare<bool>("enable_calibration_logging", false);
    auto_declare<bool>("calibration_assume_no_human", false);
    auto_declare<double>("calibration_capture_path_time_sec", 0.0);
    auto_declare<double>("cartesian_energy_lambda_update_period_sec", 0.001);
    auto_declare<double>("ee_collision_radius", 0.04);
    auto_declare<std::vector<double>>(
        "tcp_offset", std::vector<double>{0.0, 0.0, 0.0});
    auto_declare<std::string>(
        "monitor_urdf_model_path",
        "/home/developer/multipanda_ws/src/model_urdf/panda_ng.urdf");
    auto_declare<std::string>("monitor_joint_dynamics_source", "auto");
    auto_declare<std::vector<double>>(
        "prediction_joint_armature",
        std::vector<double>(7, 0.0));
    auto_declare<std::string>(
        "sara_robot_config_path",
        cps_safety_monitor::defaultSaraPandaRobotConfigPath());
    auto_declare<double>("sara_secure_radius", 0.02);
    auto_declare<bool>("enable_reachable_set_visualization", true);
    auto_declare<std::string>(
        "reachable_set_visualization_topic", "~/robot_reachable_sets");
    auto_declare<std::string>(
        "human_reachable_set_topic",
        "/human_workspace/reachable_set");
    auto_declare<std::string>("reachable_set_visualization_frame_id", "");
    auto_declare<double>("reachable_set_visualization_period_sec", 0.1);
    auto_declare<double>("reachable_set_visualization_alpha", 0.3);
    auto_declare<int>("monitor_worker_cpu_affinity", -1);
    auto_declare<int>("monitor_worker_realtime_priority", 0);

    auto_declare<std::string>("human_workspace_config_path", "");
    auto_declare<std::string>("human_workspace_topic", "human_workspace/state");
    auto_declare<double>("human_workspace_timeout_sec", 0.5);

    auto_declare<bool>("use_dynamic_consistent_impedance", true);

    auto_declare<int>("profiling_stats_print_period", 1000);
    auto_declare<bool>("enable_mujoco_contact_logging", true);
    auto_declare<std::string>("mujoco_contact_sensor_topic", "/panda_metal_ball_touch");
    auto_declare<double>("mujoco_contact_threshold", 1.0e-6);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage: %s\n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

// ============================================================================
// on_configure
// ============================================================================
CallbackReturn ReachableCartesianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    enable_error_logging_ = get_node()->get_parameter("enable_error_logging").as_bool();
    error_log_root_dir_ = get_node()->get_parameter("error_log_root_dir").as_string();
    error_log_file_name_ = get_node()->get_parameter("error_log_file_name").as_string();
    enable_prediction_logging_ =
        get_node()->get_parameter("enable_prediction_logging").as_bool();
    prediction_log_file_name_ = sanitizedFileNameOrDefault(
        get_node()->get_parameter("prediction_log_file_name").as_string(),
        "shield_prediction_trajectory.csv");
    control_log_max_queue_size_ = static_cast<std::size_t>(
        std::max<int64_t>(
            1,
            get_node()->get_parameter("control_log_max_queue_size").as_int()));
    const auto prediction_log_max_queue_size_param =
        get_node()->get_parameter("prediction_log_max_queue_size").as_int();
    prediction_log_max_queue_size_ = static_cast<std::size_t>(
        std::max<int64_t>(1, prediction_log_max_queue_size_param));
    log_batch_size_ = static_cast<std::size_t>(
        std::max<int64_t>(
            1, get_node()->get_parameter("log_batch_size").as_int()));
    log_flush_period_sec_ = std::max(
        0.05,
        get_node()->get_parameter("log_flush_period_sec").as_double());

    if (error_log_root_dir_.empty()) {
      error_log_root_dir_ = kDefaultErrorLogRootDir;
    }
    error_log_file_name_ =
        sanitizedFileNameOrDefault(error_log_file_name_, kDefaultErrorLogFileName);

    cartesian_via_points_topic_ =
        get_node()->get_parameter("cartesian_via_points_topic").as_string();
    startup_via_points_source_ =
        get_node()->get_parameter("startup_via_points_source").as_string();
    if (startup_via_points_source_ != "yaml" &&
        startup_via_points_source_ != "action") {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "startup_via_points_source must be 'yaml' or 'action'. Falling back to 'yaml'.");
      startup_via_points_source_ = "yaml";
    }
    cartesian_via_points_action_name_ =
        get_node()->get_parameter("cartesian_via_points_action_name").as_string();
    calibration_action_name_ =
        get_node()->get_parameter("calibration_action_name").as_string();
    cartesian_via_points_action_feedback_period_sec_ =
        std::max(
            0.0,
            get_node()
                ->get_parameter("cartesian_via_points_action_feedback_period_sec")
                .as_double());
    cartesian_via_points_.clear();
    cartesian_via_point_quaternions_.clear();
    std::vector<double> cartesian_via_points;
    if (startup_via_points_source_ != "action") {
      const rclcpp::Parameter via_points_param =
          get_node()->get_parameter("cartesian_via_points");
      if (via_points_param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
        cartesian_via_points = via_points_param.as_double_array();
      } else if (via_points_param.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET) {
        RCLCPP_WARN(
            get_node()->get_logger(),
            "cartesian_via_points must be a double array. Ignoring this value.");
      }
    }

    if (!cartesian_via_points.empty() &&
        (cartesian_via_points.size() % 7) == 0) {
      const std::size_t via_point_count = cartesian_via_points.size() / 7;
      cartesian_via_points_.reserve(via_point_count);
      cartesian_via_point_quaternions_.reserve(via_point_count);
      for (std::size_t i = 0; i < via_point_count; ++i) {
        cartesian_via_points_.emplace_back(
            cartesian_via_points[7 * i + 0],
            cartesian_via_points[7 * i + 1],
            cartesian_via_points[7 * i + 2]);
        const Quaterniond q(
            cartesian_via_points[7 * i + 6],
            cartesian_via_points[7 * i + 3],
            cartesian_via_points[7 * i + 4],
            cartesian_via_points[7 * i + 5]);
        if (!std::isfinite(q.norm()) || q.norm() < 1.0e-12) {
          RCLCPP_WARN(
              get_node()->get_logger(),
              "cartesian_via_points[%zu] has an invalid quaternion. Using identity quaternion.",
              i);
          cartesian_via_point_quaternions_.push_back(Quaterniond::Identity());
        } else {
          cartesian_via_point_quaternions_.push_back(
              normalizedQuaternionOrIdentity(q));
        }
      }
    } else if (!cartesian_via_points.empty()) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "cartesian_via_points must contain 7-value base-frame states [x, y, z, qx, qy, qz, qw]. Ignoring this value.");
    }

    const double pos_stiffness =
        get_node()->get_parameter("pos_stiffness").as_double();
    const double rot_stiffness =
        get_node()->get_parameter("rot_stiffness").as_double();
    enable_nullspace_ =
        get_node()->get_parameter("enable_nullspace").as_bool();
    const double configured_nullspace_stiffness = std::max(
        0.0, get_node()->get_parameter("n_stiffness").as_double());
    n_stiffness_ =
        enable_nullspace_ ? configured_nullspace_stiffness : 0.0;
    nullspace_home_pose_.setZero();
    nullspace_home_pose_valid_ = false;
    const auto nullspace_home_pose =
        get_node()->get_parameter("nullspace_home_pose").as_double_array();
    if (nullspace_home_pose.size() == kNumJoints) {
      bool finite_home_pose = true;
      for (int i = 0; i < kNumJoints; ++i) {
        nullspace_home_pose_(i) =
            nullspace_home_pose[static_cast<std::size_t>(i)];
        finite_home_pose =
            finite_home_pose && std::isfinite(nullspace_home_pose_(i));
      }
      if (finite_home_pose) {
        nullspace_home_pose_valid_ = true;
      } else {
        nullspace_home_pose_.setZero();
        RCLCPP_WARN(
            get_node()->get_logger(),
            "nullspace_home_pose contains non-finite values. Falling back "
            "to the activation joint state for the nullspace reference.");
      }
    } else {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "nullspace_home_pose must contain 7 joint values. Falling back "
          "to the activation joint state for the nullspace reference.");
    }
    enable_safety_monitor_ = get_node()->get_parameter("enable_safety_monitor").as_bool();

    energy_budget_joule_ =
        std::max(0.0, get_node()->get_parameter("energy_budget_joule").as_double());
    kinetic_energy_error_bound_joule_ = std::max(
        0.0,
        get_node()->get_parameter(
            "kinetic_energy_error_bound_joule").as_double());
    potential_energy_error_bound_joule_ = std::max(
        0.0,
        get_node()->get_parameter(
            "potential_energy_error_bound_joule").as_double());
    nullspace_potential_energy_error_bound_joule_ = std::max(
        0.0,
        get_node()->get_parameter(
            "nullspace_potential_energy_error_bound_joule").as_double());
    enable_runtime_energy_scaling_ =
        get_node()->get_parameter("enable_runtime_energy_scaling").as_bool();
    energy_recovery_exit_energy_fraction_ = get_node()->get_parameter(
        "energy_recovery_exit_energy_fraction").as_double();
    if (!std::isfinite(energy_recovery_exit_energy_fraction_) ||
        energy_recovery_exit_energy_fraction_ <= 0.0 ||
        energy_recovery_exit_energy_fraction_ > 1.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
          "energy_recovery_exit_energy_fraction must be in (0, 1].");
      return CallbackReturn::ERROR;
    }
    enable_calibration_logging_ =
        get_node()
            ->get_parameter("enable_calibration_logging")
            .as_bool();
    calibration_assume_no_human_ =
        get_node()
            ->get_parameter("calibration_assume_no_human")
            .as_bool();

    cartesian_energy_lambda_update_period_sec_ =
        std::max(0.0, get_node()->get_parameter("cartesian_energy_lambda_update_period_sec").as_double());
    ee_collision_radius_ = get_node()->get_parameter("ee_collision_radius").as_double();
    if (!std::isfinite(ee_collision_radius_) || ee_collision_radius_ <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "ee_collision_radius must be positive and finite");
      return CallbackReturn::ERROR;
    }
    const auto tcp_offset =
        get_node()->get_parameter("tcp_offset").as_double_array();
    if (tcp_offset.size() == 3 &&
        std::all_of(tcp_offset.begin(), tcp_offset.end(), [](double x) { return std::isfinite(x); })) {
      tcp_offset_ = Vector3d(tcp_offset[0], tcp_offset[1], tcp_offset[2]);
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "tcp_offset must contain 3 finite values");
      return CallbackReturn::ERROR;
    }
    monitor_urdf_model_path_ =
        get_node()->get_parameter("monitor_urdf_model_path").as_string();
    monitor_joint_dynamics_source_ =
        get_node()->get_parameter("monitor_joint_dynamics_source").as_string();
    robot_reach_config_path_ =
        get_node()->get_parameter("sara_robot_config_path").as_string();
    robot_secure_radius_ =
        get_node()->get_parameter("sara_secure_radius").as_double();
    enable_reachable_set_visualization_ =
        get_node()->get_parameter(
            "enable_reachable_set_visualization").as_bool();
    reachable_set_visualization_topic_ =
        get_node()->get_parameter(
            "reachable_set_visualization_topic").as_string();
    human_reachable_set_topic_ =
        get_node()->get_parameter(
            "human_reachable_set_topic").as_string();
    reachable_set_visualization_frame_id_ =
        get_node()->get_parameter(
            "reachable_set_visualization_frame_id").as_string();
    if (reachable_set_visualization_frame_id_.empty()) {
      reachable_set_visualization_frame_id_ = arm_id_ + "_link0";
    }
    reachable_set_visualization_period_sec_ = std::max(
        0.02,
        get_node()->get_parameter(
            "reachable_set_visualization_period_sec").as_double());
    reachable_set_visualization_alpha_ = std::clamp(
        get_node()->get_parameter(
            "reachable_set_visualization_alpha").as_double(),
        0.01,
        1.0);
    if (enable_reachable_set_visualization_ &&
        (reachable_set_visualization_topic_.empty() ||
         human_reachable_set_topic_.empty())) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Robot marker and human reachable-set output topics must not be "
          "empty when reachable-set output is enabled");
      return CallbackReturn::ERROR;
    }
    if (robot_reach_config_path_.empty()) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "sara_robot_config_path must not be empty");
      return CallbackReturn::ERROR;
    }
    if (!std::isfinite(robot_secure_radius_) ||
        robot_secure_radius_ < 0.0) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "sara_secure_radius must be finite and nonnegative");
      return CallbackReturn::ERROR;
    }
    try {
      robot_reachability_provider_ =
          cps_safety_monitor::makeSaraRobotReachabilityProvider(
              robot_reach_config_path_,
              robot_secure_radius_, tcp_offset_, ee_collision_radius_);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Failed to initialize SaRA robot reachable-set provider: %s",
          error.what());
      return CallbackReturn::ERROR;
    }
    const auto prediction_joint_armature =
        get_node()->get_parameter(
            "prediction_joint_armature").as_double_array();
    if (prediction_joint_armature.size() != 7) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "prediction_joint_armature must contain exactly 7 values");
      return CallbackReturn::ERROR;
    }
    for (std::size_t i = 0; i < prediction_joint_armature.size(); ++i) {
      const double value = prediction_joint_armature[i];
      if (!std::isfinite(value) || value < 0.0) {
        RCLCPP_ERROR(
            get_node()->get_logger(),
            "prediction_joint_armature[%zu] must be finite and nonnegative",
            i);
        return CallbackReturn::ERROR;
      }
      prediction_joint_armature_(static_cast<Eigen::Index>(i)) = value;
    }
    std::transform(
        monitor_joint_dynamics_source_.begin(),
        monitor_joint_dynamics_source_.end(),
        monitor_joint_dynamics_source_.begin(),
        [](unsigned char value) {
          return static_cast<char>(std::tolower(value));
        });
    if (monitor_joint_dynamics_source_ != "auto" &&
        monitor_joint_dynamics_source_ != "franka_interface" &&
        monitor_joint_dynamics_source_ != "pinocchio") {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "monitor_joint_dynamics_source must be auto, franka_interface, or pinocchio (got '%s').",
          monitor_joint_dynamics_source_.c_str());
      return CallbackReturn::ERROR;
    }
    if (monitor_joint_dynamics_source_ == "pinocchio" &&
        monitor_urdf_model_path_.empty()) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "monitor_urdf_model_path must not be empty when Pinocchio monitoring is selected.");
      return CallbackReturn::ERROR;
    }
    monitor_joint_dynamics_provider_.reset();
    active_monitor_joint_dynamics_source_.clear();
    if (enable_calibration_logging_) {
      if (!enable_error_logging_ || !enable_prediction_logging_) {
        RCLCPP_ERROR(
            get_node()->get_logger(),
            "enable_calibration_logging requires both enable_error_logging=true and enable_prediction_logging=true");
        return CallbackReturn::ERROR;
      }
    }
    if (calibration_assume_no_human_ && !enable_calibration_logging_) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "calibration_assume_no_human is enabled, but calibration logging "
          "is disabled; the override cannot become active.");
    }
    monitor_worker_cpu_affinity_ = static_cast<int>(
        get_node()->get_parameter("monitor_worker_cpu_affinity").as_int());
    if (monitor_worker_cpu_affinity_ < -1) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "monitor_worker_cpu_affinity must be -1 or a non-negative CPU index. Disabling affinity.");
      monitor_worker_cpu_affinity_ = -1;
    }
    const int max_realtime_priority =
        std::max(0, sched_get_priority_max(SCHED_FIFO));
    monitor_worker_realtime_priority_ = std::clamp(
        static_cast<int>(get_node()
                             ->get_parameter(
                                 "monitor_worker_realtime_priority")
                             .as_int()),
        0,
        max_realtime_priority);

    trajectory_generator_config_path_ =
        cps_trajectory_generators::defaultTrajectoryGeneratorConfigPath();
    const auto trajectory_settings =
        loadTrajectoryGeneratorSettings(trajectory_generator_config_path_);

    shield_plan_dt_ = std::max(trajectory_settings.shield_plan_dt, kMinDt);
    monitor_frequency_hz_ = trajectory_settings.monitor_frequency_hz;

    path_time_rate_min_ = trajectory_settings.path_time_rate_min;
    path_time_rate_max_ =
        std::max(path_time_rate_min_, trajectory_settings.path_time_rate_max);
    path_time_acc_limit_ =
        std::max(0.0, trajectory_settings.path_time_acc_limit);
    path_time_jerk_limit_ =
        std::max(1e-4, trajectory_settings.path_time_jerk_limit);
    path_time_rate_target_ =
        std::clamp(trajectory_settings.path_time_rate_target,
                   path_time_rate_min_,
                   path_time_rate_max_);
    failsafe_path_time_acc_limit_ =
        std::max(1e-4, trajectory_settings.failsafe_path_time_acc_limit);
    failsafe_path_time_jerk_limit_ =
        std::max(1e-4, trajectory_settings.failsafe_path_time_jerk_limit);
    local_replan_dt_ = trajectory_settings.local_replan_dt;
    monitor_period_steps_ = detail::monitorPeriodSteps(
        local_replan_dt_, monitor_frequency_hz_);
    waypoint_merge_position_tolerance_ = std::max(
        0.0, trajectory_settings.waypoint_merge_position_tolerance);
    waypoint_merge_orientation_tolerance_ = std::max(
        0.0, trajectory_settings.waypoint_merge_orientation_tolerance);
    local_replan_max_velocity_ =
        std::max(1e-4, trajectory_settings.local_replan_max_velocity);
    local_replan_max_acceleration_ =
        std::max(1e-4, trajectory_settings.local_replan_max_acceleration);
    local_replan_max_jerk_ =
        std::max(1e-4, trajectory_settings.local_replan_max_jerk);
    local_replan_max_angular_velocity_ =
        std::max(1e-4, trajectory_settings.local_replan_max_angular_velocity);
    local_replan_max_angular_acceleration_ =
        std::max(1e-4, trajectory_settings.local_replan_max_angular_acceleration);
    local_replan_max_angular_jerk_ =
        std::max(1e-4, trajectory_settings.local_replan_max_angular_jerk);
    failsafe_brake_max_velocity_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_velocity);
    failsafe_brake_max_acceleration_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_acceleration);
    failsafe_brake_max_jerk_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_jerk);
    failsafe_brake_max_angular_velocity_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_angular_velocity);
    failsafe_brake_max_angular_acceleration_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_angular_acceleration);
    failsafe_brake_max_angular_jerk_ =
        std::max(1e-4, trajectory_settings.failsafe_brake_max_angular_jerk);

    use_dynamic_consistent_impedance_ = get_node()->get_parameter("use_dynamic_consistent_impedance").as_bool();

    human_workspace_topic_ =
        get_node()->get_parameter("human_workspace_topic").as_string();
    human_workspace_timeout_sec_ = std::max(
        0.0,
        get_node()->get_parameter("human_workspace_timeout_sec").as_double());
    human_workspace_active_ = false;
    human_workspace_configured_static_ = false;
    human_workspace_live_received_.store(false, std::memory_order_relaxed);
    latest_human_workspace_msg_time_sec_.store(-1.0, std::memory_order_relaxed);

    const std::string human_workspace_config_path =
        get_node()->get_parameter("human_workspace_config_path").as_string();
    if (enable_safety_monitor_) {
      if (!human_workspace_config_path.empty()) {
        if (!configured_human_workspace_source_.configureFromConfigFile(
                human_workspace_config_path,
                get_node()->get_logger())) {
          return CallbackReturn::ERROR;
        }
        human_workspace_ = configured_human_workspace_source_;
        human_workspace_configured_static_ = true;
        human_workspace_active_ = true;
      } else {
        RCLCPP_WARN(
            get_node()->get_logger(),
            "enable_safety_monitor is true, but human_workspace_config_path is empty. "
            "Waiting for live human workspace states on '%s'.",
            human_workspace_topic_.c_str());
      }
    }

    profiling_stats_print_period_ = std::max<int>(
        1, static_cast<int>(get_node()->get_parameter("profiling_stats_print_period").as_int()));
    enable_mujoco_contact_logging_ =
        get_node()->get_parameter("enable_mujoco_contact_logging").as_bool();
    mujoco_contact_sensor_topic_ =
        get_node()->get_parameter("mujoco_contact_sensor_topic").as_string();
    mujoco_contact_threshold_ =
        std::max(0.0, get_node()->get_parameter("mujoco_contact_threshold").as_double());

    if (enable_mujoco_contact_logging_ && !mujoco_contact_sensor_topic_.empty()) {
      mujoco_contact_sub_ =
          get_node()->create_subscription<mujoco_ros_msgs::msg::ScalarStamped>(
              mujoco_contact_sensor_topic_,
              rclcpp::QoS(10),
              std::bind(
                  &ReachableCartesianImpedanceController::handleMujocoContactSensor,
                  this,
                  std::placeholders::_1));
    } else {
      mujoco_contact_sub_.reset();
    }

    if (enable_safety_monitor_ && !human_workspace_topic_.empty()) {
      human_workspace_sub_ =
          get_node()->create_subscription<cps_human_workspace::msg::HumanWorkspace>(
              human_workspace_topic_,
              rclcpp::QoS(1).transient_local(),
              std::bind(
                  &ReachableCartesianImpedanceController::handleHumanWorkspaceState,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for human workspace states on '%s'.",
          human_workspace_topic_.c_str());
    } else {
      human_workspace_sub_.reset();
    }

    if (enable_reachable_set_visualization_) {
      reachable_set_visualization_pub_ =
          get_node()->create_publisher<visualization_msgs::msg::MarkerArray>(
              reachable_set_visualization_topic_,
              rclcpp::QoS(1).reliable().transient_local());
      human_reachable_set_pub_ =
          get_node()->create_publisher<
              cps_human_workspace::msg::HumanReachableSet>(
              human_reachable_set_topic_,
              rclcpp::QoS(1).reliable().transient_local());
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Publishing SaRA robot markers on '%s' and the selected human "
          "reachable-set data on '%s' in frame '%s' (safe=last interval, "
          "contact=first contact interval, unsafe=first energy-unsafe "
          "interval).",
          reachable_set_visualization_topic_.c_str(),
          human_reachable_set_topic_.c_str(),
          reachable_set_visualization_frame_id_.c_str());
    } else {
      reachable_set_visualization_pub_.reset();
      human_reachable_set_pub_.reset();
    }

    if (!cartesian_via_points_topic_.empty()) {
      cartesian_via_points_sub_ =
          get_node()->create_subscription<geometry_msgs::msg::PoseArray>(
              cartesian_via_points_topic_,
              rclcpp::QoS(1),
              std::bind(
                  &ReachableCartesianImpedanceController::handleCartesianViaPoints,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for Cartesian via points on '%s' as geometry_msgs/msg/PoseArray.",
          cartesian_via_points_topic_.c_str());
    } else {
      cartesian_via_points_sub_.reset();
    }

    if (!cartesian_via_points_action_name_.empty()) {
      cartesian_via_points_action_server_ =
          rclcpp_action::create_server<CartesianViaMotion>(
              get_node(),
              cartesian_via_points_action_name_,
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCartesianViaPointsActionGoal,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCartesianViaPointsActionCancel,
                  this,
                  std::placeholders::_1),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCartesianViaPointsActionAccepted,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for Cartesian via-point action goals on '%s'.",
          cartesian_via_points_action_name_.c_str());
    } else {
      cartesian_via_points_action_server_.reset();
    }

    if (!calibration_action_name_.empty()) {
      calibration_action_server_ =
          rclcpp_action::create_server<CartesianViaMotion>(
              get_node(),
              calibration_action_name_,
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCalibrationActionGoal,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCalibrationActionCancel,
                  this,
                  std::placeholders::_1),
              std::bind(
                  &ReachableCartesianImpedanceController::
                      handleCalibrationActionAccepted,
                  this,
                  std::placeholders::_1));
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Listening for monitored-trajectory calibration goals on '%s'.",
          calibration_action_name_.c_str());
    } else {
      calibration_action_server_.reset();
    }

    K_base_.setZero(); D_base_.setZero();
    K_base_.topLeftCorner<3, 3>() =
        pos_stiffness * Matrix3d::Identity();
    K_base_.bottomRightCorner<3, 3>() =
        rot_stiffness * Matrix3d::Identity();
    D_base_.topLeftCorner<3, 3>() =
        0.8 * 2.0 * std::sqrt(std::max(pos_stiffness, 0.0)) *
        Matrix3d::Identity();
    D_base_.bottomRightCorner<3, 3>() =
        0.8 * 2.0 * std::sqrt(std::max(rot_stiffness, 0.0)) *
        Matrix3d::Identity();
    K_runtime_ = K_base_; D_runtime_ = D_base_;

    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model", arm_id_));

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception in on_configure: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

// ============================================================================
// on_activate / on_deactivate
// ============================================================================
CallbackReturn ReachableCartesianImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Compare the configured control rate, never a jittery wall-time estimate.
  const unsigned int control_rate = get_update_rate();
  if (control_rate > 0 && std::abs(control_rate * local_replan_dt_ - 1.0) > 1e-6) {
    RCLCPP_ERROR(get_node()->get_logger(),
        "Control update_rate=%u Hz does not match local_replan_dt=%.9f s. "
        "The command grid and control rate must agree.", control_rate, local_replan_dt_);
    return CallbackReturn::ERROR;
  }
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  start_time_ = this->get_node()->now();

  franka::RobotState* robot_state = franka_robot_model_->getRobotState();
  const Eigen::Map<const Vector7d> q(robot_state->q.data());
  try {
    monitor_joint_dynamics_provider_.reset();
    active_monitor_joint_dynamics_source_.clear();
    const bool interface_supports_prediction =
        franka_robot_model_->supportsStateDependentEvaluation();
    const bool use_franka_interface =
        monitor_joint_dynamics_source_ == "franka_interface" ||
        (monitor_joint_dynamics_source_ == "auto" &&
         interface_supports_prediction);

    if (use_franka_interface) {
      if (!interface_supports_prediction) {
        throw std::runtime_error(
            "selected Franka model backend does not evaluate arbitrary predicted q/dq");
      }
      monitor_joint_dynamics_provider_ =
          std::make_unique<FrankaInterfaceJointDynamicsProvider>(
              franka_robot_model_.get(), *robot_state, tcp_offset_);
      active_monitor_joint_dynamics_source_ = "franka_interface";
    } else {
      if (monitor_urdf_model_path_.empty()) {
        throw std::runtime_error(
            "Pinocchio fallback requires monitor_urdf_model_path");
      }
      monitor_joint_dynamics_provider_ =
          std::make_unique<PinocchioJointDynamicsProvider>(
              monitor_urdf_model_path_,
              tcp_offset_,
              prediction_joint_armature_);
      active_monitor_joint_dynamics_source_ = "pinocchio";
    }
    RCLCPP_INFO(
        get_node()->get_logger(),
        "Joint rollout dynamics source: %s (configured: %s)",
        active_monitor_joint_dynamics_source_.c_str(),
        monitor_joint_dynamics_source_.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to initialize joint rollout dynamics: %s", e.what());
    monitor_joint_dynamics_provider_.reset();
    franka_robot_model_->release_interfaces();
    return CallbackReturn::ERROR;
  }
  if (nullspace_home_pose_valid_) {
    desired_qn_ = nullspace_home_pose_;
  } else {
    desired_qn_ = q;
  }
  const Eigen::Map<const Matrix4d> pose(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  desired_orientation_ = Quaterniond(pose.block<3, 3>(0, 0));
  desired_orientation_.normalize();
  desired_position_ =
      pose.block<3, 1>(0, 3) + desired_orientation_ * tcp_offset_;

  const bool use_yaml_startup_via_points =
      startup_via_points_source_ == "yaml";
  const std::vector<Vector3d> startup_via_points =
      use_yaml_startup_via_points ? cartesian_via_points_
                                  : std::vector<Vector3d>{};
  const std::vector<Quaterniond> startup_via_orientations =
      use_yaml_startup_via_points ? cartesian_via_point_quaternions_
                                  : std::vector<Quaterniond>{};

  std::size_t waypoint_count = 0;
  std::vector<CartesianTrajectorySample> startup_path =
      buildCartesianViaPointPath(
          desired_position_,
          desired_orientation_,
          startup_via_points,
          startup_via_orientations,
          &waypoint_count);
  {
    std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
    cartesian_via_point_path_ = startup_path;
  }

  if (!use_yaml_startup_via_points) {
    RCLCPP_INFO(
        get_node()->get_logger(),
        "startup_via_points_source is 'action'. Holding the initial pose until a CartesianViaMotion action goal is received.");
  } else if (waypoint_count < 2) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "cartesian_via_points is empty. Holding the initial pose until a CartesianViaMotion action goal or PoseArray is received.");
  } else if (startup_path.empty()) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "Failed to time-parameterize startup cartesian_via_points. Holding the initial pose until a CartesianViaMotion action goal or PoseArray is received.");
  } else {
    RCLCPP_INFO(
        get_node()->get_logger(),
        "Startup via-point trajectory prepared: waypoints=%zu samples=%zu duration=%.3f s",
        waypoint_count,
        startup_path.size(),
        startup_path.back().t);
  }

  last_commanded_sample_ = ImpedanceSample{};
  last_commanded_sample_.t = 0.0;
  last_commanded_sample_.p = desired_position_;
  last_commanded_sample_.dp.setZero();
  last_commanded_sample_.ddp.setZero();
  last_commanded_sample_.q = desired_orientation_;
  last_commanded_sample_.q.normalize();
  last_commanded_sample_.w.setZero();
  last_commanded_sample_.dw.setZero();
  last_commanded_sample_.K = K_base_;
  last_commanded_sample_.D = D_base_;
  last_commanded_sample_.failsafe = false;
  last_commanded_sample_valid_ = true;
  if (!startup_path.empty() && !anchorLastCommandedSampleToPathStart()) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "Strict path-consistent execution could not anchor the startup path "
        "to the initial command state. Holding instead of using a Cartesian reconnect.");
  }
  commanded_path_rate_ =
      startup_path.empty() ? 0.0 : path_time_rate_target_;
  mode_ = SafetyMode::kNominal;
  execution_stage_ = ExecutionStage::kCurrentVerified;
  fallback_reason_ = FallbackReason::kNone;
  plan_failure_reason_ = PlanFailureReason::kNone;
  failsafe_enter_wall_time_sec_ = -1.0;
  paused_nominal_time_sec_ = 0.0;
  loop_counter_ = 0; exec_sum_ms_ = 0.0; exec_min_ms_ = 1e9; exec_max_ms_ = 0.0;
  exec_overrun_1ms_count_ = 0;
  exec_overrun_2ms_count_ = 0;
  prof_model_sum_ms_ = 0.0;
  prof_model_max_ms_ = 0.0;
  prof_shield_sum_ms_ = 0.0;
  prof_shield_max_ms_ = 0.0;
  prof_torque_sum_ms_ = 0.0;
  prof_torque_max_ms_ = 0.0;
  prof_io_sum_ms_ = 0.0;
  prof_io_max_ms_ = 0.0;
  command_recording_active_ = false;
  control_log_column_mismatch_count_.store(0, std::memory_order_relaxed);
  control_update_sequence_ = 0;
  previous_control_start_steady_ns_ = 0;
  previous_control_execution_ms_ = 0.0;
  next_async_monitor_control_sequence_ = 1;
  last_async_input_publish_control_sequence_ = 0;
  async_monitor_schedule_late_cycles_ = 0;
  async_monitor_schedule_skipped_slots_ = 0;
  async_monitor_input_publish_count_.store(0, std::memory_order_relaxed);
  async_monitor_input_overwrite_count_.store(0, std::memory_order_relaxed);
  async_monitor_worker_processed_count_.store(0, std::memory_order_relaxed);
  async_monitor_output_overwrite_count_.store(0, std::memory_order_relaxed);
  async_monitor_output_consumed_count_.store(0, std::memory_order_relaxed);
  last_async_monitor_timing_ = AsyncMonitorTiming{};
  latest_mujoco_contact_value_.store(0.0);
  latest_mujoco_contact_msg_time_.store(-1.0);
  latest_mujoco_contact_active_.store(false);

  last_energy_budget_active_ = false;
  last_energy_budget_lambda_valid_ = false;
  last_energy_stiffness_scale_ = 1.0;
  energy_recovery_state_ = cps_safety_monitor::EnergyRecoveryState{};
  energy_recovery_epoch_ = 0;
  energy_recovery_environment_ = -1;
  last_nullspace_stiffness_ = 0.0;
  last_joint_kinetic_energy_ = 0.0;
  last_cartesian_potential_energy_before_scaling_ = 0.0;
  last_nullspace_potential_energy_before_scaling_ = 0.0;
  last_nullspace_potential_energy_ = 0.0;
  last_total_control_energy_before_scaling_ = 0.0;
  last_cartesian_potential_energy_ = 0.0;
  last_total_control_energy_ = 0.0;
  last_tau_task_norm_ = 0.0;
  last_tau_nullspace_raw_norm_ = 0.0;
  last_tau_nullspace_projected_norm_ = 0.0;
  last_coriolis_norm_ = 0.0;
  last_tau_desired_before_rate_limit_norm_ = 0.0;
  cartesian_energy_task_inertia_cache_.setZero();
  cartesian_energy_task_inertia_cache_valid_ = false;
  cartesian_energy_task_inertia_cache_wall_time_ = -1.0;
  last_verified_plan_ = VerifiedPlan{};
  last_verified_plan_generation_ = 0;
  current_source_plan_generation_.store(
      last_verified_plan_generation_, std::memory_order_release);
  last_verified_command_stage_ = 0;
  last_verified_command_index_ = 0;
  verified_command_selected_this_cycle_ = false;
  last_commanded_verified_plan_valid_ = false;
  last_commanded_verified_plan_generation_ = 0;
  last_commanded_verified_command_stage_ = 0;
  last_commanded_verified_command_index_ = 0;
  calibration_plan_latched_ = false;
  calibration_plan_complete_ = false;
  calibration_target_failed_ = false;
  calibration_requested_capture_path_time_sec_ = 0.0;
  calibration_actual_capture_path_time_sec_ = -1.0;
  calibration_failsafe_command_count_ = 0;
  calibration_plan_generation_ = 0;
  calibration_monitor_input_sequence_ = 0;
  calibration_activation_control_sequence_ = 0;
  calibration_activation_intended_index_ = 0;
  calibration_activation_failsafe_index_ = 0;
  active_cartesian_via_points_calibration_ = false;
  commanded_path_time_ = 0.0;

  tau_cmd_prev_.setZero();
  last_shield_decision_valid_ = false;
  async_input_sequence_.store(0);
  async_request_gate_.resetStopped();
  async_monitor_busy_deferred_cycles_ = 0;
  async_input_pending_ = false;
  async_output_mailbox_.resetStopped();

  J_geo_prev_.setZero();
  Jdot_dq_filtered_.setZero();
  J_geo_prev_valid_ = false;

  if (enable_error_logging_ || enable_prediction_logging_) {
    const std::filesystem::path root_dir(error_log_root_dir_);
    error_log_run_dir_ =
        (root_dir / makeBerlinTimestampForDirectoryName()).string();
    error_log_file_path_ =
        (std::filesystem::path(error_log_run_dir_) / error_log_file_name_).string();
    prediction_log_file_path_ =
        (std::filesystem::path(error_log_run_dir_) / prediction_log_file_name_).string();

    std::error_code ec;
    std::filesystem::create_directories(error_log_run_dir_, ec);
    if (ec) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Failed to create log run directory %s: %s",
          error_log_run_dir_.c_str(),
          ec.message().c_str());
      return CallbackReturn::ERROR;
    }

    const std::filesystem::path run_info_path =
        std::filesystem::path(error_log_run_dir_) / "run_info.txt";
    std::ofstream run_info_file(run_info_path, std::ios::out | std::ios::trunc);
    if (run_info_file.is_open()) {
      std::vector<CartesianTrajectorySample> path_snapshot;
      {
        std::lock_guard<std::mutex> path_lock(cartesian_via_point_path_mutex_);
        path_snapshot = cartesian_via_point_path_;
      }
      std::string human_workspace_config_path =
          get_node()->get_parameter("human_workspace_config_path").as_string();
      if (human_workspace_config_path.empty()) {
        human_workspace_config_path = "(none)";
      }
      const auto& human_parameters = human_workspace_.parameters();
      const bool human_center_motion_enabled =
          human_parameters.center_velocity.squaredNorm() > 1.0e-18 ||
          (human_parameters.center_sinusoid_amplitude.squaredNorm() > 1.0e-18 &&
           human_parameters.center_sinusoid_frequency_hz > 0.0);
      run_info_file << "run_directory: " << error_log_run_dir_ << "\n"
                    << "csv_file: " << error_log_file_path_ << "\n"
                    << "prediction_csv_file: " << prediction_log_file_path_ << "\n"
                    << "enable_error_logging: "
                    << static_cast<int>(enable_error_logging_) << "\n"
                    << "enable_prediction_logging: "
                    << static_cast<int>(enable_prediction_logging_) << "\n"
                    << "enable_inertia_model_comparison: "
                    << static_cast<int>(enable_prediction_logging_) << "\n"
                    << "inertia_model_comparison_semantics: "
                       "same_q_dq_runtime_snapshot_vs_prediction_provider\n"
                    << "logging_backend: bounded_async_csv\n"
                    << "control_log_max_queue_size: "
                    << control_log_max_queue_size_ << "\n"
                    << "prediction_log_max_queue_size: "
                    << prediction_log_max_queue_size_ << "\n"
                    << "log_batch_size: " << log_batch_size_ << "\n"
                    << "log_flush_period_sec: "
                    << log_flush_period_sec_ << "\n"
                    << "recording_start: first_valid_via_points_command\n"
                    << "arm_id: " << arm_id_ << "\n"
                    << "state_log_schema: orthogonal_execution_v21\n"
                    << "monitor_gain_policy: nominal_stiffness_and_damping\n"
                    << "monitor_recovery_energy_policy: retain_contact_energy_gate_full_horizon\n"
                    << "energy_control_phase_legend: 0=normal, 1=limited, 2=recovering\n"
                    << "energy_recovery_environment_legend: 0=clear, 1=overlap, 2=unknown\n"
                    << "mode_legend: 0=current_verified_execution, "
                       "1=fallback_execution\n"
                    << "execution_stage_legend: 0=current_verified, "
                       "1=last_verified_intended, "
                       "2=last_verified_failsafe, "
                       "3=hold\n"
                    << "fallback_reason_legend: 0=none, "
                       "1=no_verified_plan_available, "
                       "2=candidate_prediction_rejected, "
                       "3=async_output_unavailable, "
                       "4=human_workspace_unavailable\n"
                    << "reason_invariant: at_most_one_of_fallback_reason_"
                       "and_plan_failure_reason_is_nonzero\n"
                    << "plan_failure_reason_legend: 0=none, "
                       "1=no_active_path, "
                       "2=missing_nominal_path_state, "
                       "3=intended_generation_empty, "
                       "4=intended_seam_invalid, "
                       "5=failsafe_generation_empty, "
                       "6=failsafe_seam_invalid, "
                       "7=intended_sample_invalid, "
                       "8=intended_transition_invalid, "
                       "9=failsafe_sample_invalid, "
                       "10=failsafe_transition_invalid, "
                       "11=candidate_invalid_unknown\n"
                    << "cartesian_via_points_count: "
                    << cartesian_via_points_.size() << "\n"
                    << "cartesian_via_point_quaternions_count: "
                    << cartesian_via_point_quaternions_.size() << "\n"
                    << "cartesian_via_points_topic: "
                    << cartesian_via_points_topic_ << "\n"
                    << "cartesian_via_point_path_samples: "
                    << path_snapshot.size() << "\n"
                    << "cartesian_via_point_path_duration_sec: "
                    << (path_snapshot.empty()
                            ? 0.0
                            : path_snapshot.back().t)
                    << "\n"
                    << "enable_safety_monitor: " << static_cast<int>(enable_safety_monitor_) << "\n"
                    << "monitor_execution: asynchronous\n"
                    << "monitor_segment_policy: equal_control_periods\n"
                    << "async_handoff_policy: strict_committed_prefix_deadline\n"
                    << "monitor_schedule_source: control_loop_sequence\n"
                    << "async_pipeline_policy: single_outstanding_until_consumed_or_discarded_accept_before_publish\n"
                    << "async_handoff_rejection_mask_legend: 1=source_generation, 2=recovery_epoch, 4=recovery_state, 8=workspace_policy, 16=calibration_target, 32=activation_window_or_continuity\n"
                    << "monitor_cpu_time_source: CLOCK_THREAD_CPUTIME_ID\n"
                    << "monitor_non_cpu_ms_semantics: elapsed_minus_thread_cpu_includes_preemption_and_blocking_and_small_sampling_overhead\n"
                    << "monitor_context_switch_source: getrusage_RUSAGE_THREAD_delta\n"
                    << "monitor_rollout_steps_semantics: actual_joint_prediction_trace_size_minus_one_zero_if_trace_unavailable\n"
                    << "monitor_command_count_semantics: dense_candidate_counts_not_sparse_csv_rows\n"
                    << "async_busy_deferred_cycles_semantics: due_control_cycles_without_free_request_slot_not_number_of_worker_requests\n"
                    << "profiling_output_execution: independent_diagnostics_worker\n"
                    << "monitor_visualization_execution: independent_diagnostics_worker_reuses_monitor_alpha\n"
                    << "monitor_period_control_cycles: "
                    << monitor_period_steps_ << "\n"
                    << "monitor_worker_cpu_affinity: "
                    << monitor_worker_cpu_affinity_ << "\n"
                    << "monitor_worker_realtime_priority: "
                    << monitor_worker_realtime_priority_ << "\n"
                    << "committed_steps: "
                    << monitor_period_steps_ << "\n"
                    << "fresh_intended_steps: "
                    << monitor_period_steps_ << "\n"
                    << "monitor_frequency_hz: " << monitor_frequency_hz_ << "\n"
                    << "monitor_joint_dynamics_source_configured: "
                    << monitor_joint_dynamics_source_ << "\n"
                    << "monitor_joint_dynamics_source_active: "
                    << active_monitor_joint_dynamics_source_ << "\n"
                    << "prediction_joint_armature: ["
                    << prediction_joint_armature_(0) << ", "
                    << prediction_joint_armature_(1) << ", "
                    << prediction_joint_armature_(2) << ", "
                    << prediction_joint_armature_(3) << ", "
                    << prediction_joint_armature_(4) << ", "
                    << prediction_joint_armature_(5) << ", "
                    << prediction_joint_armature_(6) << "]\n"
                    << "prediction_joint_armature_applied: "
                    << static_cast<int>(
                           active_monitor_joint_dynamics_source_ ==
                           "pinocchio")
                    << "\n"
                    << "robot_reachable_set_backend: "
                    << (robot_reachability_provider_
                            ? robot_reachability_provider_->backendName()
                            : "legacy_cartesian")
                    << "\n"
                    << "sara_robot_config_path: "
                    << robot_reach_config_path_ << "\n"
                    << "sara_secure_radius: "
                    << robot_secure_radius_ << "\n"
                    << "sara_alpha_mode: dynamic_from_complete_monitored_trajectory\n"
                    << "enable_reachable_set_visualization: "
                    << static_cast<int>(
                         enable_reachable_set_visualization_) << "\n"
                    << "reachable_set_visualization_topic: "
                    << reachable_set_visualization_topic_ << "\n"
                    << "human_reachable_set_topic: "
                    << human_reachable_set_topic_ << "\n"
                    << "reachable_set_visualization_frame_id: "
                    << reachable_set_visualization_frame_id_ << "\n"
                    << "reachable_set_visualization_period_sec: "
                    << reachable_set_visualization_period_sec_ << "\n"
                    << "reachable_set_visualization_alpha: "
                    << reachable_set_visualization_alpha_ << "\n"
                    << "monitor_update_period_sec: " << (1.0 / monitor_frequency_hz_) << "\n"
                    << "trajectory_generator_config_path: "
                    << trajectory_generator_config_path_ << "\n"
                    << "shield_plan_dt: " << shield_plan_dt_ << "\n"
                    << "monitor_sparse_dt: " << shield_plan_dt_ << "\n"
                    << "monitor_joint_rollout_grid: candidate_command_timestamps\n"
                    << "local_replan_dt: " << local_replan_dt_ << "\n"
                    << "failsafe_plan_dt: "
                    << std::max(local_replan_dt_, kMinDt) << "\n"
                    << "failsafe_command_dt: " << local_replan_dt_ << "\n"
                    << "cartesian_gain_policy: "
                       "overlap_triggered_recovery_latched_cartesian_nullspace_"
                       "energy_scaling\n"
                    << "pos_stiffness: " << K_base_(0, 0) << "\n"
                    << "rot_stiffness: " << K_base_(3, 3) << "\n"
                    << "pos_damping: " << D_base_(0, 0) << "\n"
                    << "rot_damping: " << D_base_(3, 3) << "\n"
                    << "enable_nullspace: "
                    << static_cast<int>(enable_nullspace_) << "\n"
                    << "nullspace_stiffness: " << n_stiffness_ << "\n"
                    << "nullspace_reference: "
                    << desired_qn_.transpose() << "\n"
                    << "energy_budget_joule: "
                    << energy_budget_joule_ << "\n"
                    << "kinetic_energy_error_bound_joule: "
                    << kinetic_energy_error_bound_joule_ << "\n"
                    << "potential_energy_error_bound_joule: "
                    << potential_energy_error_bound_joule_ << "\n"
                    << "nullspace_potential_energy_error_bound_joule: "
                    << nullspace_potential_energy_error_bound_joule_ << "\n"
                    << "enable_runtime_energy_scaling: "
                    << static_cast<int>(enable_runtime_energy_scaling_)
                    << "\n"
                    << "energy_recovery_exit_energy_fraction: "
                    << energy_recovery_exit_energy_fraction_ << "\n"
                    << "energy_recovery_exit_policy: nominal_energy_below_fraction_"
                       "previous_and_current_scale_one_clear_valid_workspace_"
                       "joint_position_velocity_limits_current_epoch_verified_command\n"
                    << "enable_calibration_logging: "
                    << static_cast<int>(
                           enable_calibration_logging_)
                    << "\n"
                    << "calibration_assume_no_human: "
                    << static_cast<int>(calibration_assume_no_human_)
                    << "\n"
                    << "calibration_capture_path_time_sec_default: "
                    << get_node()
                           ->get_parameter(
                               "calibration_capture_path_time_sec")
                           .as_double()
                    << "\n"
                    << "calibration_action_name: "
                    << calibration_action_name_ << "\n"
                    << "calibration_plan_policy: "
                       "dedicated_action_capture_exactly_next_monitor_input_"
                       "at_or_after_requested_path_time_"
                       "abort_if_not_executable_no_candidate_substitution_"
                       "then_execute_remaining_intended_and_failsafe\n"
                    << "calibration_prediction_sampling: "
                       "exact_runtime_command_grid_and_indices\n"
                    << "measured_energy_tracking_policy: "
                       "global_diagnostics_only_no_control_action\n"
                    << "measured_energy_tracking_feeds_scaling_when_active: "
                       "1\n"
                    << "cartesian_energy_lambda_update_period_sec: "
                    << cartesian_energy_lambda_update_period_sec_ << "\n"
                    << "energy_budget_activation: "
                       "current_overlap_or_latched_recovery_or_unknown_workspace\n"
                    << "prediction_energy_scaling_policy: "
                       "disabled_nominal_rollout_with_shadow_recovery_"
                       "state_and_per_command_exit_permission\n"
                    << "contact_relevant_for_energy_legend: "
                       "1=workspace_distance_now_nonpositive\n"
                    << "nullspace_home_pose: ["
                    << nullspace_home_pose_(0) << ", "
                    << nullspace_home_pose_(1) << ", "
                    << nullspace_home_pose_(2) << ", "
                    << nullspace_home_pose_(3) << ", "
                    << nullspace_home_pose_(4) << ", "
                    << nullspace_home_pose_(5) << ", "
                    << nullspace_home_pose_(6) << "]\n"
                    << "nullspace_home_pose_valid: "
                    << static_cast<int>(nullspace_home_pose_valid_) << "\n"
                    << "tcp_offset: ["
                    << tcp_offset_.x() << ", "
                    << tcp_offset_.y() << ", "
                    << tcp_offset_.z() << "]\n"
                    << "ee_collision_radius: " << ee_collision_radius_ << "\n"
                    << "control_reference_point: tcp_ball_center\n"
                    << "collision_reference_point: tcp\n"
                    << "robot_reach_capsule_layout: indices_0_to_6_arm_index_7_tcp_sphere\n"
                    << "enable_mujoco_contact_logging: "
                    << static_cast<int>(enable_mujoco_contact_logging_) << "\n"
                    << "mujoco_contact_sensor_topic: "
                    << mujoco_contact_sensor_topic_ << "\n"
                    << "mujoco_contact_threshold: "
                    << mujoco_contact_threshold_ << "\n"
                    << "timestamp_timezone: Europe/Berlin\n"
                    << "initial_desired_position: ["
                    << desired_position_.x() << ", "
                    << desired_position_.y() << ", "
                    << desired_position_.z() << "]\n"
                    << "human_sphere_center: ["
                    << human_parameters.sphere_center.x() << ", "
                    << human_parameters.sphere_center.y() << ", "
                    << human_parameters.sphere_center.z() << "]\n"
                    << "human_center_motion_enabled: "
                    << static_cast<int>(human_center_motion_enabled) << "\n"
                    << "human_center_linear_velocity: ["
                    << human_parameters.center_velocity.x() << ", "
                    << human_parameters.center_velocity.y() << ", "
                    << human_parameters.center_velocity.z() << "]\n"
                    << "human_center_sinusoid_amplitude: ["
                    << human_parameters.center_sinusoid_amplitude.x() << ", "
                    << human_parameters.center_sinusoid_amplitude.y() << ", "
                    << human_parameters.center_sinusoid_amplitude.z() << "]\n"
                    << "human_center_sinusoid_frequency_hz: "
                    << human_parameters.center_sinusoid_frequency_hz << "\n"
                    << "human_center_sinusoid_phase_rad: "
                    << human_parameters.center_sinusoid_phase_rad << "\n"
                    << "human_center_motion_time_offset_sec: "
                    << human_parameters.center_motion_time_offset_sec << "\n"
                    << "human_hand_physical_radius: "
                    << human_parameters.motion_radius << "\n"
                    << "human_reachability_model: "
                    << "sara_body_part_combined_single_hand\n"
                    << "human_hand_max_velocity: "
                    << human_parameters.hand_max_velocity << "\n"
                    << "human_hand_max_acceleration: "
                    << human_parameters.hand_max_acceleration << "\n"
                    << "human_measurement_error_position: "
                    << human_parameters.measurement_error_position << "\n"
                    << "human_measurement_error_velocity: "
                    << human_parameters.measurement_error_velocity << "\n"
                    << "human_measurement_delay: "
                    << human_parameters.measurement_delay << "\n"
                    << "human_workspace_config_path: "
                    << human_workspace_config_path << "\n";
    }

    if (!startLogWriters()) {
      return CallbackReturn::ERROR;
    }
    if (enable_error_logging_) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Asynchronous validation log enabled: %s",
                  error_log_file_path_.c_str());
    }
    if (enable_prediction_logging_) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Asynchronous shield prediction log enabled: %s",
                  prediction_log_file_path_.c_str());
    }
  }
  last_reachable_set_visualization_wall_time_ = -1.0;
  last_reachable_set_visualization_marker_count_ = 0;
  startSafetyMonitorWorker();
  return CallbackReturn::SUCCESS;
}

CallbackReturn ReachableCartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  stopSafetyMonitorWorker();
  clearReachableSetOutputs();
  stopLogWriters();
  plan_failure_reason_ = PlanFailureReason::kNone;
  human_workspace_active_ = false;
  verified_command_selected_this_cycle_ = false;
  last_commanded_verified_plan_valid_ = false;
  calibration_plan_latched_ = false;
  calibration_plan_complete_ = false;
  calibration_target_failed_ = false;
  calibration_requested_capture_path_time_sec_ = 0.0;
  calibration_actual_capture_path_time_sec_ = -1.0;
  active_cartesian_via_points_calibration_ = false;
  J_geo_prev_.setZero();
  Jdot_dq_filtered_.setZero();
  J_geo_prev_valid_ = false;
  monitor_joint_dynamics_provider_.reset();
  active_monitor_joint_dynamics_source_.clear();
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace cps_controllers
