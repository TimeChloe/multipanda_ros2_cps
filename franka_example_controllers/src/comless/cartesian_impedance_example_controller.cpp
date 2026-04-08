// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <franka_example_controllers/comless/cartesian_impedance_example_controller.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <franka/model.h>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace {

constexpr double kMinDt = 1e-6;
constexpr double kPitchCosEps = 1e-6;
constexpr double kSmoothStartDuration = 2.0;  // s
constexpr double kDynInvReg = 1e-9;

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;
using Matrix76d = Eigen::Matrix<double, 7, 6>;

enum class ReferenceTrajectoryType {
  kLine,
  kLissajous
};

struct SmoothStartProfile {
  double s;
  double ds;
  double dds;
};

struct TaskRefZYX {
  Eigen::Vector3d p, dp, ddp;
  Eigen::Vector3d rpy, drpy, ddrpy;
  Eigen::Matrix3d R;
};

inline ReferenceTrajectoryType parseReferenceTrajectoryType(const std::string& name) {
  if (name == "lissajous") {
    return ReferenceTrajectoryType::kLissajous;
  }
  return ReferenceTrajectoryType::kLine;
}

inline SmoothStartProfile makeSmoothStartProfile(double t, double T) {
  SmoothStartProfile out{0.0, 0.0, 0.0};

  if (t <= 0.0) {
    return out;
  }
  if (t >= T) {
    out.s = 1.0;
    return out;
  }

  const double x = t / T;
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double x4 = x3 * x;
  const double x5 = x4 * x;

  out.s = 10.0 * x3 - 15.0 * x4 + 6.0 * x5;
  out.ds = (30.0 * x2 - 60.0 * x3 + 30.0 * x4) / T;
  out.dds = (60.0 * x - 180.0 * x2 + 120.0 * x3) / (T * T);

  return out;
}

inline Eigen::Matrix3d rpyToRotationMatrixZYX(const Eigen::Vector3d& rpy) {
  const Eigen::Matrix3d Rz =
      Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d Ry =
      Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d Rx =
      Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX()).toRotationMatrix();
  return Rz * Ry * Rx;
}

// ZYX RPY from rotation matrix
inline Eigen::Vector3d rotationMatrixToRpy(const Eigen::Matrix3d& R) {
  Eigen::Vector3d rpy;
  const double sy = std::hypot(R(0, 0), R(1, 0));
  const bool singular = sy < 1e-8;

  if (!singular) {
    rpy(0) = std::atan2(R(2, 1), R(2, 2));   // roll
    rpy(1) = std::atan2(-R(2, 0), sy);       // pitch
    rpy(2) = std::atan2(R(1, 0), R(0, 0));   // yaw
  } else {
    rpy(0) = std::atan2(-R(1, 2), R(1, 1));
    rpy(1) = std::atan2(-R(2, 0), sy);
    rpy(2) = 0.0;
  }

  return rpy;
}

// T^{-1}(rpy): omega_world -> rpy_dot
inline Eigen::Matrix3d worldOmegaToRpyRatesMatrixZYX(const Eigen::Vector3d& rpy) {
  const double pitch = rpy(1);
  const double yaw = rpy(2);

  double ctheta = std::cos(pitch);
  const double stheta = std::sin(pitch);
  const double cpsi = std::cos(yaw);
  const double spsi = std::sin(yaw);

  if (std::abs(ctheta) < kPitchCosEps) {
    ctheta = (ctheta >= 0.0 ? kPitchCosEps : -kPitchCosEps);
  }

  const double tan_theta = stheta / ctheta;

  Eigen::Matrix3d Tinv;
  Tinv << cpsi / ctheta,      spsi / ctheta,      0.0,
         -spsi,               cpsi,               0.0,
          cpsi * tan_theta,   spsi * tan_theta,   1.0;
  return Tinv;
}

// d/dt[T^{-1}(rpy)]
inline Eigen::Matrix3d worldOmegaToRpyRatesMatrixDotZYX(
    const Eigen::Vector3d& rpy,
    const Eigen::Vector3d& rpy_dot) {
  const double pitch = rpy(1);
  const double yaw = rpy(2);

  double ctheta = std::cos(pitch);
  const double stheta = std::sin(pitch);
  const double cpsi = std::cos(yaw);
  const double spsi = std::sin(yaw);

  if (std::abs(ctheta) < kPitchCosEps) {
    ctheta = (ctheta >= 0.0 ? kPitchCosEps : -kPitchCosEps);
  }

  const double sec_theta = 1.0 / ctheta;
  const double tan_theta = stheta / ctheta;
  const double sec2_theta = sec_theta * sec_theta;

  const double pitch_dot = rpy_dot(1);
  const double yaw_dot = rpy_dot(2);

  Eigen::Matrix3d dT_dtheta = Eigen::Matrix3d::Zero();
  dT_dtheta << cpsi * sec_theta * tan_theta,  spsi * sec_theta * tan_theta, 0.0,
               0.0,                           0.0,                          0.0,
               cpsi * sec2_theta,             spsi * sec2_theta,            0.0;

  Eigen::Matrix3d dT_dyaw = Eigen::Matrix3d::Zero();
  dT_dyaw << -spsi * sec_theta,   cpsi * sec_theta,   0.0,
             -cpsi,              -spsi,               0.0,
             -spsi * tan_theta,   cpsi * tan_theta,   0.0;

  return dT_dtheta * pitch_dot + dT_dyaw * yaw_dot;
}

inline Matrix76d dynamicallyConsistentRightInverse(
    const Matrix67d& J,
    const Eigen::Matrix<double, 7, 7>& M,
    double reg = kDynInvReg) {
  const Matrix76d M_inv_JT = M.ldlt().solve(J.transpose());

  Matrix6d lambda_inv = J * M_inv_JT;
  lambda_inv.diagonal().array() += reg;

  return M_inv_JT *
         lambda_inv.ldlt().solve(Matrix6d::Identity());
}

inline Eigen::Matrix<double, 7, 7> arrayToMatrix7d(
    const std::array<double, 49>& data) {
  Eigen::Matrix<double, 7, 7> out;
  for (size_t i = 0; i < 7; ++i) {
    for (size_t j = 0; j < 7; ++j) {
      out(static_cast<int>(i), static_cast<int>(j)) = data[i * 7 + j];
    }
  }
  return out;
}

inline Eigen::Matrix<double, 7, 1> arrayToVector7d(
    const std::array<double, 7>& data) {
  Eigen::Matrix<double, 7, 1> out;
  for (size_t i = 0; i < 7; ++i) {
    out(static_cast<int>(i)) = data[i];
  }
  return out;
}

inline std::array<double, 7> vectorToArray7(
    const std::vector<double>& values,
    double default_value = 0.0) {
  std::array<double, 7> out{};
  out.fill(default_value);
  const size_t n = std::min<size_t>(7, values.size());
  for (size_t i = 0; i < n; ++i) {
    out[i] = values[i];
  }
  return out;
}

inline Eigen::Matrix<double, 7, 1> computeSmoothJointFrictionTorque(
    const Eigen::Matrix<double, 7, 1>& dq_like,
    const std::array<double, 7>& friction_coulomb,
    const std::array<double, 7>& friction_viscous,
    const std::array<double, 7>& friction_velocity_scale,
    const std::array<double, 7>& friction_offset,
    double friction_scale) {
  Eigen::Matrix<double, 7, 1> tau_friction;
  tau_friction.setZero();

  for (int i = 0; i < 7; ++i) {
    const double vs = std::max(std::abs(friction_velocity_scale[i]), 1e-6);
    const double dq_i = dq_like(i);
    const double smooth_sign = std::tanh(dq_i / vs);

    tau_friction(i) =
        friction_scale *
        (friction_coulomb[i] * smooth_sign +
         friction_viscous[i] * dq_i +
         friction_offset[i]);
  }

  return tau_friction;
}

// ---------------- trajectory 1: straight line ----------------
inline TaskRefZYX makeReferenceZYXLine(
    double t,
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& rpy0) {
  TaskRefZYX ref;

  constexpr double T_move = 3.0;   // total motion duration [s]
  constexpr double dz = -0.40;     // move downward by 40 cm

  const SmoothStartProfile ramp = makeSmoothStartProfile(t, T_move);

  ref.p = p0 + Eigen::Vector3d(0.0, 0.0, dz * ramp.s);
  ref.dp = Eigen::Vector3d(0.0, 0.0, dz * ramp.ds);
  ref.ddp = Eigen::Vector3d(0.0, 0.0, dz * ramp.dds);

  ref.rpy = rpy0;
  ref.drpy = Eigen::Vector3d::Zero();
  ref.ddrpy = Eigen::Vector3d::Zero();

  ref.R = rpyToRotationMatrixZYX(ref.rpy);
  return ref;
}

// ---------------- trajectory 2: Lissajous-like trajectory ----------------
inline TaskRefZYX makeReferenceZYXLissajous(
    double t,
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& rpy0) {
  TaskRefZYX ref;

  const double wt = 2.0 * M_PI * 0.25;  // 0.25 Hz
  const double wr = 2.0 * M_PI * 0.20;  // 0.20 Hz

  const double ph_px = 0.0;
  const double ph_py = M_PI / 2.0;
  const double ph_pz = 0.0;

  const double ph_rx = 0.0;
  const double ph_ry = M_PI / 3.0;
  const double ph_rz = 2.0 * M_PI / 3.0;

  const SmoothStartProfile ramp = makeSmoothStartProfile(t, kSmoothStartDuration);

  // translational amplitudes
  constexpr double Ax = 0.08;  // 8 cm
  constexpr double Ay = 0.08;  // 8 cm
  constexpr double Az = 0.04;  // 4 cm

  // rotational amplitudes
  constexpr double Aroll  = 0.06981317008;   // 4 deg
  constexpr double Apitch = 0.05235987756;   // 3 deg
  constexpr double Ayaw   = 0.08726646260;   // 5 deg

  const Eigen::Vector3d p_base(
      Ax * (std::sin(wt * t + ph_px) - std::sin(ph_px)),
      Ay * (std::sin(wt * t + ph_py) - std::sin(ph_py)),
      Az * (std::sin(0.5 * wt * t + ph_pz) - std::sin(ph_pz)));

  const Eigen::Vector3d dp_base(
      Ax * wt * std::cos(wt * t + ph_px),
      Ay * wt * std::cos(wt * t + ph_py),
      Az * 0.5 * wt * std::cos(0.5 * wt * t + ph_pz));

  const Eigen::Vector3d ddp_base(
      -Ax * wt * wt * std::sin(wt * t + ph_px),
      -Ay * wt * wt * std::sin(wt * t + ph_py),
      -Az * 0.25 * wt * wt * std::sin(0.5 * wt * t + ph_pz));

  const Eigen::Vector3d rpy_base(
      Aroll  * (std::sin(wr * t + ph_rx) - std::sin(ph_rx)),
      Apitch * (std::sin(wr * t + ph_ry) - std::sin(ph_ry)),
      Ayaw   * (std::sin(wr * t + ph_rz) - std::sin(ph_rz)));

  const Eigen::Vector3d drpy_base(
      Aroll  * wr * std::cos(wr * t + ph_rx),
      Apitch * wr * std::cos(wr * t + ph_ry),
      Ayaw   * wr * std::cos(wr * t + ph_rz));

  const Eigen::Vector3d ddrpy_base(
      -Aroll  * wr * wr * std::sin(wr * t + ph_rx),
      -Apitch * wr * wr * std::sin(wr * t + ph_ry),
      -Ayaw   * wr * wr * std::sin(wr * t + ph_rz));

  ref.p = p0 + ramp.s * p_base;
  ref.dp = ramp.ds * p_base + ramp.s * dp_base;
  ref.ddp = ramp.dds * p_base + 2.0 * ramp.ds * dp_base + ramp.s * ddp_base;

  ref.rpy = rpy0 + ramp.s * rpy_base;
  ref.drpy = ramp.ds * rpy_base + ramp.s * drpy_base;
  ref.ddrpy = ramp.dds * rpy_base + 2.0 * ramp.ds * drpy_base + ramp.s * ddrpy_base;

  ref.R = rpyToRotationMatrixZYX(ref.rpy);
  return ref;
}

inline TaskRefZYX makeReferenceZYX(
    double t,
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& rpy0,
    const ReferenceTrajectoryType traj_type) {
  switch (traj_type) {
    case ReferenceTrajectoryType::kLissajous:
      return makeReferenceZYXLissajous(t, p0, rpy0);
    case ReferenceTrajectoryType::kLine:
    default:
      return makeReferenceZYXLine(t, p0, rpy0);
  }
}

inline void buildTaskReferenceFromSamePoseTrajectory(
    double t,
    const Eigen::Vector3d& p0,
    const Eigen::Matrix3d& R0,
    const ReferenceTrajectoryType traj_type,
    Eigen::Matrix<double, 6, 1>& r_d,
    Eigen::Matrix<double, 6, 1>& dr_d,
    Eigen::Matrix<double, 6, 1>& ddr_d,
    Eigen::Vector3d& desired_position_cur,
    Eigen::Quaterniond& desired_orientation_cur) {
  const Eigen::Vector3d rpy0 = rotationMatrixToRpy(R0);
  const TaskRefZYX ref = makeReferenceZYX(t, p0, rpy0, traj_type);

  r_d.head<3>() = ref.p;
  dr_d.head<3>() = ref.dp;
  ddr_d.head<3>() = ref.ddp;

  r_d.tail<3>() = ref.rpy;
  dr_d.tail<3>() = ref.drpy;
  ddr_d.tail<3>() = ref.ddrpy;

  desired_position_cur = ref.p;
  desired_orientation_cur = Eigen::Quaterniond(ref.R);
  desired_orientation_cur.normalize();
}

inline void buildConstantTaskReference(
    const Eigen::Vector3d& p0,
    const Eigen::Matrix3d& R0,
    Eigen::Matrix<double, 6, 1>& r_d,
    Eigen::Matrix<double, 6, 1>& dr_d,
    Eigen::Matrix<double, 6, 1>& ddr_d,
    Eigen::Vector3d& desired_position_cur,
    Eigen::Quaterniond& desired_orientation_cur) {
  r_d.setZero();
  dr_d.setZero();
  ddr_d.setZero();

  r_d.head<3>() = p0;
  r_d.tail<3>() = rotationMatrixToRpy(R0);

  desired_position_cur = p0;
  desired_orientation_cur = Eigen::Quaterniond(rpyToRotationMatrixZYX(r_d.tail<3>()));
  desired_orientation_cur.normalize();
}

}  // namespace

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }

  return config;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto& name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(name);
  }

  return config;
}

Vector6d CartesianImpedanceExampleController::computeTaskPose(
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& p) const {
  Vector6d r;
  r.head<3>() = p;
  r.tail<3>() = rotationMatrixToRpy(R);
  return r;
}

Vector3d CartesianImpedanceExampleController::wrapEulerError(const Vector3d& e) const {
  Vector3d out = e;
  for (int i = 0; i < 3; ++i) {
    while (out(i) > M_PI) {
      out(i) -= 2.0 * M_PI;
    }
    while (out(i) < -M_PI) {
      out(i) += 2.0 * M_PI;
    }
  }
  return out;
}

Eigen::Matrix<double, 6, 7>
CartesianImpedanceExampleController::computeAnalyticJacobian(
    const Eigen::Matrix<double, 6, 7>& J_geo,
    const Eigen::Vector3d& rpy) const {
  Matrix6d T = Matrix6d::Identity();
  T.bottomRightCorner<3, 3>() = worldOmegaToRpyRatesMatrixZYX(rpy);
  return T * J_geo;
}

// 保留定义，避免与你已有的 hpp 声明不匹配
Eigen::Matrix<double, 6, 7>
CartesianImpedanceExampleController::computeAnalyticJacobianDotNumerical(
    const Vector7d& q,
    const Vector7d& dq,
    const Eigen::Vector3d& rpy,
    double dt) {
  const double safe_dt = std::max(dt, kMinDt);
  const Vector7d q_next = q + dq * safe_dt;

  pinocchio::Data data_now(pin_model_);
  pinocchio::Data data_next(pin_model_);

  pinocchio::forwardKinematics(pin_model_, data_now, q);
  pinocchio::computeJointJacobians(pin_model_, data_now, q);
  pinocchio::updateFramePlacements(pin_model_, data_now);

  Matrix67d J_geo_now = Matrix67d::Zero();
  pinocchio::getFrameJacobian(
      pin_model_, data_now, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_geo_now);

  const Matrix67d J_r_now = computeAnalyticJacobian(J_geo_now, rpy);

  pinocchio::forwardKinematics(pin_model_, data_next, q_next);
  pinocchio::computeJointJacobians(pin_model_, data_next, q_next);
  pinocchio::updateFramePlacements(pin_model_, data_next);

  const Eigen::Matrix3d R_next = data_next.oMf[ee_frame_id_].rotation();
  const Eigen::Vector3d rpy_next = rotationMatrixToRpy(R_next);

  Matrix67d J_geo_next = Matrix67d::Zero();
  pinocchio::getFrameJacobian(
      pin_model_, data_next, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_geo_next);

  const Matrix67d J_r_next = computeAnalyticJacobian(J_geo_next, rpy_next);

  return (J_r_next - J_r_now) / safe_dt;
}

controller_interface::return_type CartesianImpedanceExampleController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  const double dt = std::max(period.seconds(), kMinDt);
  const double t = (this->get_node()->now() - start_time_).seconds();

  const std::string reference_trajectory_type =
      get_node()->get_parameter("reference_trajectory_type").as_string();
  const ReferenceTrajectoryType traj_type =
      parseReferenceTrajectoryType(reference_trajectory_type);

  const bool use_franka_model_for_dynamics =
      get_node()->get_parameter("use_franka_model_for_dynamics").as_bool();

  const bool use_friction_compensation =
      get_node()->get_parameter("use_friction_compensation").as_bool();
  const bool friction_use_reference_velocity =
      get_node()->get_parameter("friction_use_reference_velocity").as_bool();
  const double friction_scale =
      get_node()->get_parameter("friction_scale").as_double();

  const std::array<double, 7> friction_coulomb =
      vectorToArray7(get_node()->get_parameter("friction_coulomb").as_double_array(), 0.0);
  const std::array<double, 7> friction_viscous =
      vectorToArray7(get_node()->get_parameter("friction_viscous").as_double_array(), 0.0);
  const std::array<double, 7> friction_velocity_scale =
      vectorToArray7(get_node()->get_parameter("friction_velocity_scale").as_double_array(), 0.05);
  const std::array<double, 7> friction_offset =
      vectorToArray7(get_node()->get_parameter("friction_offset").as_double_array(), 0.0);

  // measured state
  Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
  Eigen::Map<const Vector7d> dq(franka_robot_model_->getRobotState()->dq.data());

  const Eigen::Vector3d p0 = desired_position_;
  const Eigen::Matrix3d R0 = desired_orientation_.toRotationMatrix();

  Vector6d r_d = Vector6d::Zero();
  Vector6d dr_d = Vector6d::Zero();
  Vector6d ddr_d = Vector6d::Zero();

  Eigen::Vector3d desired_position_cur = p0;
  Eigen::Quaterniond desired_orientation_cur = desired_orientation_;
  desired_orientation_cur.normalize();

  if (use_constant_reference_) {
    buildConstantTaskReference(
        p0, R0, r_d, dr_d, ddr_d, desired_position_cur, desired_orientation_cur);
  } else {
    buildTaskReferenceFromSamePoseTrajectory(
        t, p0, R0, traj_type, r_d, dr_d, ddr_d, desired_position_cur, desired_orientation_cur);
  }

  // Pinocchio kinematics + dynamics
  pinocchio::forwardKinematics(pin_model_, *pin_data_, q, dq);
  pinocchio::computeJointJacobians(pin_model_, *pin_data_, q);
  pinocchio::computeJointJacobiansTimeVariation(pin_model_, *pin_data_, q, dq);
  pinocchio::updateFramePlacements(pin_model_, *pin_data_);

  pinocchio::crba(pin_model_, *pin_data_, q);
  pin_data_->M.template triangularView<Eigen::StrictlyLower>() =
      pin_data_->M.transpose().template triangularView<Eigen::StrictlyLower>();

  // Pinocchio quantities are always available
  const Matrix7d M_pin = pin_data_->M;
  const Matrix7d C_pin = pinocchio::computeCoriolisMatrix(pin_model_, *pin_data_, q, dq);

  // Hybrid model selection:
  // - Use libfranka/franka_semantic_components mass/coriolis if available
  // - Fall back to Pinocchio otherwise
  // - Keep Pinocchio Coriolis matrix for screenshot-like C(q,dq) * dq_ref term
  Matrix7d M_model = M_pin;
  Vector7d coriolis_current_vec = C_pin * dq;  // default fallback
  bool using_franka_dynamics = false;

  if (use_franka_model_for_dynamics) {
    try {
      M_model = arrayToMatrix7d(franka_robot_model_->getMassMatrix());
      coriolis_current_vec = arrayToVector7d(franka_robot_model_->getCoriolisForceVector());
      using_franka_dynamics = true;
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(),
          *get_node()->get_clock(),
          2000,
          "Failed to read Franka dynamics model, falling back to Pinocchio: %s",
          e.what());
      M_model = M_pin;
      coriolis_current_vec = C_pin * dq;
      using_franka_dynamics = false;
    }
  }

  const Eigen::Vector3d p = pin_data_->oMf[ee_frame_id_].translation();
  const Eigen::Matrix3d R = pin_data_->oMf[ee_frame_id_].rotation();

  const Eigen::Vector3d current_position = p;
  Eigen::Quaterniond current_orientation(R);
  current_orientation.normalize();

  const Vector6d r = computeTaskPose(R, p);

  // geometric Jacobian and its time variation
  Matrix67d J_geo = Matrix67d::Zero();
  Matrix67d J_geo_dot = Matrix67d::Zero();

  pinocchio::getFrameJacobian(
      pin_model_, *pin_data_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_geo);

  pinocchio::getFrameJacobianTimeVariation(
      pin_model_, *pin_data_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_geo_dot);

  const Eigen::Matrix3d Tinv = worldOmegaToRpyRatesMatrixZYX(r.tail<3>());
  Matrix6d T = Matrix6d::Identity();
  T.bottomRightCorner<3, 3>() = Tinv;

  const Matrix67d J_r = T * J_geo;
  const Vector6d dr = J_r * dq;

  const Eigen::Matrix3d Tinv_dot =
      worldOmegaToRpyRatesMatrixDotZYX(r.tail<3>(), dr.tail<3>());
  Matrix6d Tdot = Matrix6d::Zero();
  Tdot.bottomRightCorner<3, 3>() = Tinv_dot;

  const Matrix67d J_r_dot = Tdot * J_geo + T * J_geo_dot;

  const Eigen::Vector3d desired_rpy = r_d.tail<3>();
  const Eigen::Vector3d current_rpy = r.tail<3>();
  const double task_velocity_norm = dr.norm();

  // tracking error: e = r_d - r, de = dr_d - dr
  Vector6d e = r_d - r;
  e.tail<3>() = wrapEulerError(e.tail<3>());
  const Vector6d de = dr_d - dr;

  // dynamically consistent right inverse and task-space inertia inverse
  const Matrix76d M_inv_JT = M_model.ldlt().solve(J_r.transpose());

  Matrix6d lambda_inv = J_r * M_inv_JT;
  lambda_inv.diagonal().array() += kDynInvReg;

  const Matrix76d J_r_pinv =
      M_inv_JT * lambda_inv.ldlt().solve(Matrix6d::Identity());

  const Matrix6d task_projection = J_r * J_r_pinv;
  const double pinv_projection_error_norm =
      (task_projection - Matrix6d::Identity()).norm();

  // Screenshot-like reference realization:
  // qdot_ref = J_r^{-1} rdot_d
  // qddot_ref = J_r^{-1} (rddot_d - Jdot_r qdot_ref)
  const Vector7d dq_ref = J_r_pinv * dr_d;
  const Vector7d ddq_ref = J_r_pinv * (ddr_d - J_r_dot * dq_ref);

  Vector7d tau_friction_current = Vector7d::Zero();
  Vector7d tau_friction_ref = Vector7d::Zero();
  Vector7d tau_friction_nonlinear = Vector7d::Zero();

  if (use_friction_compensation) {
    tau_friction_current = computeSmoothJointFrictionTorque(
        dq,
        friction_coulomb,
        friction_viscous,
        friction_velocity_scale,
        friction_offset,
        friction_scale);

    tau_friction_ref = computeSmoothJointFrictionTorque(
        dq_ref,
        friction_coulomb,
        friction_viscous,
        friction_velocity_scale,
        friction_offset,
        friction_scale);

    tau_friction_nonlinear =
        friction_use_reference_velocity ? tau_friction_ref : tau_friction_current;
  }

  const Vector6d vel_residual = J_r * dq_ref - dr_d;
  const Vector6d acc_residual = J_r * ddq_ref + J_r_dot * dq_ref - ddr_d;

  const double vel_residual_norm = vel_residual.norm();
  const double acc_residual_norm = acc_residual.norm();

  // impedance-like task feedback wrench (kept for logging / comparison branch)
  const Vector7d tau_impedance =
      J_r.transpose() * (D_m_ * de + K_m_ * e);

  // ------------------------------------------------------------
  // Branch A: impedance torque form (comparison branch)
  // ------------------------------------------------------------
  const Matrix7d N_tau =
      Matrix7d::Identity() - J_r.transpose() * J_r_pinv.transpose();

  const Vector7d tau_null_impedance =
      N_tau * (n_stiffness_ * (desired_qn_ - q) -
               2.0 * std::sqrt(n_stiffness_) * dq);

  // Current-velocity Coriolis term for the comparison branch
  const Vector7d tau_impedance_total =
      tau_impedance + coriolis_current_vec + tau_friction_current + tau_null_impedance;

  // ------------------------------------------------------------
  // Branch B: nonlinear model-based law, implemented to be
  // as close as possible to the screenshot:
  //
  // tau = M(q) qdd_ref + C(q,dq) qd_ref + tau_f(qd_ref or qd)
  //       + J_r^T [ D_m (rd_d - rd) + K_m (r_d - r) ]
  //
  // Gravity is NOT added here because the effort interface already
  // provides gravity compensation at the low level.
  // ------------------------------------------------------------

  // For screenshot alignment, this term should be C(q,dq) * qdot_ref.
  // libfranka semantic component usually exposes the Coriolis vector
  // c(q,dq), not the full C(q,dq) matrix, so we keep Pinocchio here.
  const Vector7d coriolis_ref_vec = C_pin * dq_ref;

  const Vector7d tau_task_id =
      M_model * ddq_ref + coriolis_ref_vec + tau_friction_nonlinear + tau_impedance;

  const Vector7d tau_id_total =
      tau_task_id + tau_null_impedance;

  const Vector7d tau_null_id = tau_null_impedance;

  // keep same variable name for logging compatibility
  const Vector7d tau_ff = tau_id_total;

  Vector7d tau = tau_impedance_total;
  Vector7d tau_null = tau_null_impedance;

  if (use_nonlinear_feedforward_) {
    tau = tau_id_total;
    tau_null = tau_null_id;
  }

  const double tau_norm = tau.norm();
  const double tau_impedance_norm = tau_impedance.norm();
  const double tau_ff_norm = tau_ff.norm();
  const double tau_null_norm = tau_null.norm();

  const double dq_ref_norm = dq_ref.norm();
  const double ddq_ref_norm = ddq_ref.norm();
  const double J_r_dot_norm = J_r_dot.norm();
  const double M_norm = M_model.norm();
  const double C_norm = C_pin.norm();

  // useful extra diagnostics for analysis / papers
  const double coriolis_current_norm = coriolis_current_vec.norm();
  const double coriolis_ref_norm = coriolis_ref_vec.norm();
  const double friction_current_norm = tau_friction_current.norm();
  const double friction_ref_norm = tau_friction_ref.norm();
  const double friction_nonlinear_norm = tau_friction_nonlinear.norm();
  const int using_franka_dynamics_int = using_franka_dynamics ? 1 : 0;
  const int using_friction_compensation_int = use_friction_compensation ? 1 : 0;

  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau(i));
  }

  if (enable_error_logging_ && error_log_file_.is_open()) {
    const double pos_error_norm = e.head<3>().norm();
    const double rot_error_norm = e.tail<3>().norm();

    error_log_file_
        << std::fixed << std::setprecision(9)
        << t << ","
        // desired position
        << desired_position_cur(0) << "," << desired_position_cur(1) << "," << desired_position_cur(2) << ","
        // current position
        << current_position(0) << "," << current_position(1) << "," << current_position(2) << ","
        // desired quaternion [x y z w]
        << desired_orientation_cur.x() << "," << desired_orientation_cur.y() << ","
        << desired_orientation_cur.z() << "," << desired_orientation_cur.w() << ","
        // current quaternion [x y z w]
        << current_orientation.x() << "," << current_orientation.y() << ","
        << current_orientation.z() << "," << current_orientation.w() << ","
        // desired rpy
        << desired_rpy(0) << "," << desired_rpy(1) << "," << desired_rpy(2) << ","
        // current rpy
        << current_rpy(0) << "," << current_rpy(1) << "," << current_rpy(2) << ","
        // task error
        << e(0) << "," << e(1) << "," << e(2) << ","
        << e(3) << "," << e(4) << "," << e(5) << ","
        // current task velocity
        << dr(0) << "," << dr(1) << "," << dr(2) << ","
        << dr(3) << "," << dr(4) << "," << dr(5) << ","
        // reference realization residuals
        << vel_residual(0) << "," << vel_residual(1) << "," << vel_residual(2) << ","
        << vel_residual(3) << "," << vel_residual(4) << "," << vel_residual(5) << ","
        << acc_residual(0) << "," << acc_residual(1) << "," << acc_residual(2) << ","
        << acc_residual(3) << "," << acc_residual(4) << "," << acc_residual(5) << ","
        // norms
        << pos_error_norm << "," << rot_error_norm << ","
        << task_velocity_norm << ","
        << vel_residual_norm << "," << acc_residual_norm << ","
        << pinv_projection_error_norm << ","
        << tau_impedance_norm << ","
        << tau_ff_norm << ","
        << tau_null_norm << ","
        << dq_ref_norm << ","
        << ddq_ref_norm << ","
        << J_r_dot_norm << ","
        << M_norm << ","
        << C_norm << ","
        << coriolis_current_norm << ","
        << coriolis_ref_norm << ","
        << friction_current_norm << ","
        << friction_ref_norm << ","
        << friction_nonlinear_norm << ","
        << using_franka_dynamics_int << ","
        << using_friction_compensation_int << ","
        << (use_nonlinear_feedforward_ ? 1 : 0) << ","
        << tau_norm
        << "\n";

    ++log_write_counter_;
    if (log_write_counter_ % 200 == 0) {
      error_log_file_.flush();
    }
  }

  (void)dt;
  return controller_interface::return_type::OK;
}

CallbackReturn CartesianImpedanceExampleController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<bool>("enable_error_logging", false);
    auto_declare<bool>("use_constant_reference", true);
    auto_declare<bool>("use_nonlinear_feedforward", true);
    auto_declare<bool>("use_franka_model_for_dynamics", true);

    auto_declare<bool>("use_friction_compensation", false);
    auto_declare<bool>("friction_use_reference_velocity", true);
    auto_declare<double>("friction_scale", 1.0);
    auto_declare<std::vector<double>>("friction_coulomb", std::vector<double>(7, 0.0));
    auto_declare<std::vector<double>>("friction_viscous", std::vector<double>(7, 0.0));
    auto_declare<std::vector<double>>("friction_velocity_scale", std::vector<double>(7, 0.05));
    auto_declare<std::vector<double>>("friction_offset", std::vector<double>(7, 0.0));

    auto_declare<std::string>("reference_trajectory_type", "lissajous"); //line lissajous
    auto_declare<std::string>(
        "error_log_path",
        "/home/developer/multipanda_ws/src/data_log/cartesian_pose_error.csv");
    auto_declare<std::string>(
        "urdf_model_path",
        "/home/developer/multipanda_ws/src/model_urdf/panda.urdf");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage: %s\n", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    use_constant_reference_ = get_node()->get_parameter("use_constant_reference").as_bool();
    use_nonlinear_feedforward_ =
        get_node()->get_parameter("use_nonlinear_feedforward").as_bool();
    enable_error_logging_ = get_node()->get_parameter("enable_error_logging").as_bool();
    error_log_path_ = get_node()->get_parameter("error_log_path").as_string();
    urdf_model_path_ = get_node()->get_parameter("urdf_model_path").as_string();

    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model", arm_id_));

    pinocchio::urdf::buildModel(urdf_model_path_, pin_model_);
    pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

    ee_frame_id_ = static_cast<pinocchio::FrameIndex>(pin_model_.nframes);

    const std::array<std::string, 4> ee_candidates = {
        "panda_hand_tcp",
        "panda_hand",
        "panda_link8",
        "panda_grasptarget"};

    for (const auto& frame_name : ee_candidates) {
      const auto frame_id = pin_model_.getFrameId(frame_name);
      if (frame_id < pin_model_.nframes) {
        ee_frame_id_ = frame_id;
        RCLCPP_INFO(get_node()->get_logger(),
                    "Using end-effector frame: %s",
                    frame_name.c_str());
        break;
      }
    }

    if (ee_frame_id_ >= pin_model_.nframes) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to find a valid end-effector frame in URDF: %s",
                   urdf_model_path_.c_str());
      return CallbackReturn::ERROR;
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Exception in on_configure: %s",
                 e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  start_time_ = this->get_node()->now();

  const Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
  desired_qn_ = q;

  pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
  pinocchio::updateFramePlacements(pin_model_, *pin_data_);

  desired_position_ = pin_data_->oMf[ee_frame_id_].translation();
  desired_orientation_ = Quaterniond(pin_data_->oMf[ee_frame_id_].rotation());
  desired_orientation_.normalize();

  const double pos_stiff = 400.0;
  const double rot_stiff = 20.0;

  K_m_.setZero();
  D_m_.setZero();

  K_m_.topLeftCorner<3, 3>() = pos_stiff * Matrix3d::Identity();
  K_m_.bottomRightCorner<3, 3>() = rot_stiff * Matrix3d::Identity();

  D_m_.topLeftCorner<3, 3>() = 2.0 * std::sqrt(pos_stiff) * Matrix3d::Identity();
  D_m_.bottomRightCorner<3, 3>() =
      0.8 * 2.0 * std::sqrt(rot_stiff) * Matrix3d::Identity();

  n_stiffness_ = 0.0;
  log_write_counter_ = 0;

  if (enable_error_logging_) {
    error_log_file_.open(error_log_path_, std::ios::out | std::ios::trunc);
    if (!error_log_file_.is_open()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to open error log file: %s",
                   error_log_path_.c_str());
      return CallbackReturn::ERROR;
    }

    error_log_file_ << std::fixed << std::setprecision(9);
    error_log_file_
        << "time_sec,"
        << "des_px,des_py,des_pz,"
        << "cur_px,cur_py,cur_pz,"
        << "des_qx,des_qy,des_qz,des_qw,"
        << "cur_qx,cur_qy,cur_qz,cur_qw,"
        << "des_rpy_r,des_rpy_p,des_rpy_y,"
        << "cur_rpy_r,cur_rpy_p,cur_rpy_y,"
        << "err_px,err_py,err_pz,"
        << "err_rx,err_ry,err_rz,"
        << "cur_rdot_px,cur_rdot_py,cur_rdot_pz,"
        << "cur_rdot_rx,cur_rdot_ry,cur_rdot_rz,"
        << "vel_res_px,vel_res_py,vel_res_pz,"
        << "vel_res_rx,vel_res_ry,vel_res_rz,"
        << "acc_res_px,acc_res_py,acc_res_pz,"
        << "acc_res_rx,acc_res_ry,acc_res_rz,"
        << "pos_err_norm,rot_err_norm,"
        << "task_velocity_norm,vel_residual_norm,acc_residual_norm,"
        << "pinv_projection_error_norm,"
        << "tau_impedance_norm,tau_ff_norm,tau_null_norm,"
        << "dq_ref_norm,ddq_ref_norm,J_r_dot_norm,M_norm,C_norm,"
        << "coriolis_current_norm,coriolis_ref_norm,"
        << "friction_current_norm,friction_ref_norm,friction_nonlinear_norm,"
        << "using_franka_dynamics,using_friction_compensation,"
        << "use_nonlinear_feedforward,tau_norm\n";

    RCLCPP_INFO(get_node()->get_logger(),
                "Cartesian pose error logging enabled, writing to: %s",
                error_log_path_.c_str());
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceExampleController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (error_log_file_.is_open()) {
    error_log_file_.flush();
    error_log_file_.close();
  }

  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController,
                       controller_interface::ControllerInterface)