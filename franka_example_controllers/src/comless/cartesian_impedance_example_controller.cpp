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

#include <Eigen/Dense>

#include <franka/model.h>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace {

constexpr double kPseudoInverseDamping = 1e-4;
constexpr double kMinDt = 1e-6;
constexpr double kPitchCosEps = 1e-6;
constexpr double kSmoothStartDuration = 2.0;  // seconds

struct TaskRef6D {
  Eigen::Vector3d p, dp, ddp;
  Eigen::Vector3d phi, dphi, ddphi;
  Eigen::Matrix3d R;
};

struct SmoothStartProfile {
  double s;    // position scaling
  double ds;   // velocity scaling
  double dds;  // acceleration scaling
};

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(),  v.y(),
       v.z(),  0.0, -v.x(),
      -v.y(),  v.x(), 0.0;
  return S;
}

inline Eigen::Matrix3d expMapSO3(const Eigen::Vector3d& phi) {
  const double theta = phi.norm();
  if (theta < 1e-10) {
    return Eigen::Matrix3d::Identity() + skew(phi);
  }
  Eigen::AngleAxisd aa(theta, phi / theta);
  return aa.toRotationMatrix();
}

// quintic smooth start:
// t <= 0     : s = 0
// 0 < t < T  : s = 10x^3 - 15x^4 + 6x^5, x=t/T
// t >= T     : s = 1
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

// 新版参考轨迹：
// 保留原来的周期轨迹形状，但加一个 smooth-start 包络，保证
// t=0 时 offset=0, velocity=0, acceleration=0
TaskRef6D makeReference(double t,
                        const Eigen::Vector3d& p0,
                        const Eigen::Matrix3d& R0) {
  TaskRef6D ref;

  const double wt = 2.0 * M_PI * 0.25;
  const double wr = 2.0 * M_PI * 0.20;

  // 保留原来相位，保持轨迹“丰富度”
  const double ph_px = 0.0;
  const double ph_py = M_PI / 2.0;
  const double ph_pz = 0.0;

  const double ph_rx = 0.0;
  const double ph_ry = M_PI / 3.0;
  const double ph_rz = 2.0 * M_PI / 3.0;

  const SmoothStartProfile ramp = makeSmoothStartProfile(t, kSmoothStartDuration);

  // ---------- base translational offsets (before smooth-start envelope) ----------
  // enlarged amplitudes for clearer controller validation
  constexpr double Ax = 0.08;   // 8 cm
  constexpr double Ay = 0.08;   // 8 cm
  constexpr double Az = 0.04;   // 4 cm

  constexpr double Arx = 0.1396;  // 8 deg
  constexpr double Ary = 0.1047;  // 6 deg
  constexpr double Arz = 0.1745;  // 10 deg

  // constexpr double Arx = 0;  // 8 deg
  // constexpr double Ary = 0;  // 6 deg
  // constexpr double Arz = 0;  // 10 deg

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

  const Eigen::Vector3d phi_base(
      Arx * (std::sin(wr * t + ph_rx) - std::sin(ph_rx)),
      Ary * (std::sin(wr * t + ph_ry) - std::sin(ph_ry)),
      Arz * (std::sin(wr * t + ph_rz) - std::sin(ph_rz)));

  const Eigen::Vector3d dphi_base(
      Arx * wr * std::cos(wr * t + ph_rx),
      Ary * wr * std::cos(wr * t + ph_ry),
      Arz * wr * std::cos(wr * t + ph_rz));

  const Eigen::Vector3d ddphi_base(
      -Arx * wr * wr * std::sin(wr * t + ph_rx),
      -Ary * wr * wr * std::sin(wr * t + ph_ry),
      -Arz * wr * wr * std::sin(wr * t + ph_rz));

  // ---------- apply smooth-start envelope ----------
  const Eigen::Vector3d p_offset = ramp.s * p_base;
  const Eigen::Vector3d dp_offset = ramp.ds * p_base + ramp.s * dp_base;
  const Eigen::Vector3d ddp_offset =
      ramp.dds * p_base + 2.0 * ramp.ds * dp_base + ramp.s * ddp_base;

  const Eigen::Vector3d phi = ramp.s * phi_base;
  const Eigen::Vector3d dphi = ramp.ds * phi_base + ramp.s * dphi_base;
  const Eigen::Vector3d ddphi =
      ramp.dds * phi_base + 2.0 * ramp.ds * dphi_base + ramp.s * ddphi_base;

  ref.p = p0 + p_offset;
  ref.dp = dp_offset;
  ref.ddp = ddp_offset;

  ref.phi = phi;
  ref.dphi = dphi;
  ref.ddphi = ddphi;

  ref.R = R0 * expMapSO3(ref.phi);
  return ref;
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

inline Eigen::Vector3d unwrapToNearby(const Eigen::Vector3d& ref,
                                      const Eigen::Vector3d& x) {
  Eigen::Vector3d out = x;
  for (int i = 0; i < 3; ++i) {
    while (out(i) - ref(i) > M_PI) {
      out(i) -= 2.0 * M_PI;
    }
    while (out(i) - ref(i) < -M_PI) {
      out(i) += 2.0 * M_PI;
    }
  }
  return out;
}

// omega_world = E(rpy) * rpy_dot
inline Eigen::Matrix3d rpyRatesToWorldOmegaMatrix(const Eigen::Vector3d& rpy) {
  const double pitch = rpy(1);
  const double yaw = rpy(2);

  const double ctheta = std::cos(pitch);
  const double stheta = std::sin(pitch);
  const double cpsi = std::cos(yaw);
  const double spsi = std::sin(yaw);

  Eigen::Matrix3d E;
  E << cpsi * ctheta, -spsi, 0.0,
       spsi * ctheta,  cpsi, 0.0,
       -stheta,        0.0,  1.0;
  return E;
}

// rpy_dot = E^{-1}(rpy) * omega_world
inline Eigen::Matrix3d worldOmegaToRpyRatesMatrix(const Eigen::Vector3d& rpy) {
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

  Eigen::Matrix3d E_inv;
  E_inv << cpsi / ctheta,      spsi / ctheta,      0.0,
          -spsi,               cpsi,               0.0,
           cpsi * tan_theta,   spsi * tan_theta,   1.0;
  return E_inv;
}

inline Eigen::Matrix<double, 7, 6> dampedRightPseudoInverse(
    const Eigen::Matrix<double, 6, 7>& J,
    double lambda = kPseudoInverseDamping) {
  const Eigen::Matrix<double, 6, 6> JJt =
      J * J.transpose() + lambda * Eigen::Matrix<double, 6, 6>::Identity();
  return J.transpose() * JJt.inverse();
}

// 从与旧控制器完全相同的 pose reference 中，构造新控制器使用的
// r_d = [p_d; rpy_d], dr_d, ddr_d
inline void buildTaskReferenceFromSamePoseTrajectory(
    double t,
    double dt,
    const Eigen::Vector3d& p0,
    const Eigen::Matrix3d& R0,
    Eigen::Matrix<double, 6, 1>& r_d,
    Eigen::Matrix<double, 6, 1>& dr_d,
    Eigen::Matrix<double, 6, 1>& ddr_d) {
  const double h = std::max(dt, 1e-3);

  const TaskRef6D ref = makeReference(t, p0, R0);

  r_d.head<3>() = ref.p;
  dr_d.head<3>() = ref.dp;
  ddr_d.head<3>() = ref.ddp;

  const Eigen::Vector3d rpy_cur = rotationMatrixToRpy(ref.R);

  // 姿态部分：仍来自同一个 R_d(t)，这里只是映射到 rpy 坐标
  Eigen::Vector3d rpy_prev, rpy_next, rpy_next2;

  if (t < h) {
    const TaskRef6D ref_next  = makeReference(t + h,     p0, R0);
    const TaskRef6D ref_next2 = makeReference(t + 2.0 * h, p0, R0);

    rpy_next  = rotationMatrixToRpy(ref_next.R);
    rpy_next2 = rotationMatrixToRpy(ref_next2.R);

    rpy_next  = unwrapToNearby(rpy_cur, rpy_next);
    rpy_next2 = unwrapToNearby(rpy_next, rpy_next2);

    r_d.tail<3>() = rpy_cur;
    dr_d.tail<3>() = (rpy_next - rpy_cur) / h;
    ddr_d.tail<3>() = (rpy_next2 - 2.0 * rpy_next + rpy_cur) / (h * h);
  } else {
    const TaskRef6D ref_prev = makeReference(t - h, p0, R0);
    const TaskRef6D ref_next = makeReference(t + h, p0, R0);

    rpy_prev = rotationMatrixToRpy(ref_prev.R);
    rpy_next = rotationMatrixToRpy(ref_next.R);

    rpy_prev = unwrapToNearby(rpy_cur, rpy_prev);
    rpy_next = unwrapToNearby(rpy_cur, rpy_next);

    r_d.tail<3>() = rpy_cur;
    dr_d.tail<3>() = (rpy_next - rpy_prev) / (2.0 * h);
    ddr_d.tail<3>() = (rpy_next - 2.0 * rpy_cur + rpy_prev) / (h * h);
  }
}

inline void buildConstantTaskReference(
    const Eigen::Vector3d& p0,
    const Eigen::Matrix3d& R0,
    Eigen::Matrix<double, 6, 1>& r_d,
    Eigen::Matrix<double, 6, 1>& dr_d,
    Eigen::Matrix<double, 6, 1>& ddr_d) {
  r_d.setZero();
  dr_d.setZero();
  ddr_d.setZero();

  r_d.head<3>() = p0;
  r_d.tail<3>() = rotationMatrixToRpy(R0);
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
  Eigen::Matrix<double, 6, 6> T_inv = Eigen::Matrix<double, 6, 6>::Identity();
  T_inv.bottomRightCorner<3, 3>() = worldOmegaToRpyRatesMatrix(rpy);
  return T_inv * J_geo;
}

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

  Eigen::Matrix<double, 6, 7> J_geo_now = Eigen::Matrix<double, 6, 7>::Zero();
  pinocchio::getFrameJacobian(
      pin_model_, data_now, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_geo_now);

  const Eigen::Matrix<double, 6, 7> J_r_now = computeAnalyticJacobian(J_geo_now, rpy);

  pinocchio::forwardKinematics(pin_model_, data_next, q_next);
  pinocchio::computeJointJacobians(pin_model_, data_next, q_next);
  pinocchio::updateFramePlacements(pin_model_, data_next);

  const Eigen::Matrix3d R_next = data_next.oMf[ee_frame_id_].rotation();
  const Eigen::Vector3d rpy_next = rotationMatrixToRpy(R_next);

  Eigen::Matrix<double, 6, 7> J_geo_next = Eigen::Matrix<double, 6, 7>::Zero();
  pinocchio::getFrameJacobian(
      pin_model_, data_next, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_geo_next);

  const Eigen::Matrix<double, 6, 7> J_r_next = computeAnalyticJacobian(J_geo_next, rpy_next);

  return (J_r_next - J_r_now) / safe_dt;
}

controller_interface::return_type CartesianImpedanceExampleController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  const double dt = std::max(period.seconds(), kMinDt);
  const double t = (this->get_node()->now() - start_time_).seconds();

  // measured robot state always from native Franka state
  Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
  Eigen::Map<const Vector7d> dq(franka_robot_model_->getRobotState()->dq.data());

  // desired reference anchor (must use the same EE frame as control/logging)
  const Eigen::Vector3d p0 = desired_position_;
  const Eigen::Matrix3d R0 = desired_orientation_.toRotationMatrix();

  // desired task reference
  Vector6d r_d = Vector6d::Zero();
  Vector6d dr_d = Vector6d::Zero();
  Vector6d ddr_d = Vector6d::Zero();

  Eigen::Vector3d desired_position_cur = p0;
  Eigen::Quaterniond desired_orientation_cur = desired_orientation_;
  desired_orientation_cur.normalize();

  if (use_constant_reference_) {
    buildConstantTaskReference(p0, R0, r_d, dr_d, ddr_d);
  } else {
    buildTaskReferenceFromSamePoseTrajectory(t, dt, p0, R0, r_d, dr_d, ddr_d);

    const TaskRef6D ref_pose = makeReference(t, p0, R0);
    desired_position_cur = ref_pose.p;
    desired_orientation_cur = Eigen::Quaterniond(ref_pose.R);
    desired_orientation_cur.normalize();
  }

  // Pinocchio kinematics + dynamics, all in one consistent model/frame
  pinocchio::forwardKinematics(pin_model_, *pin_data_, q, dq);
  pinocchio::computeJointJacobians(pin_model_, *pin_data_, q);
  pinocchio::updateFramePlacements(pin_model_, *pin_data_);

  pinocchio::crba(pin_model_, *pin_data_, q);
  pin_data_->M.template triangularView<Eigen::StrictlyLower>() =
      pin_data_->M.transpose().template triangularView<Eigen::StrictlyLower>();

  const Matrix7d C = pinocchio::computeCoriolisMatrix(pin_model_, *pin_data_, q, dq);
  const Matrix7d M = pin_data_->M;

  const Eigen::Vector3d p = pin_data_->oMf[ee_frame_id_].translation();
  const Eigen::Matrix3d R = pin_data_->oMf[ee_frame_id_].rotation();

  const Eigen::Vector3d current_position = p;
  Eigen::Quaterniond current_orientation(R);
  current_orientation.normalize();

  const Vector6d r = computeTaskPose(R, p);

  Eigen::Matrix<double, 6, 7> J_geo = Eigen::Matrix<double, 6, 7>::Zero();
  pinocchio::getFrameJacobian(
      pin_model_, *pin_data_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_geo);

  const Eigen::Matrix<double, 6, 7> J_r = computeAnalyticJacobian(J_geo, r.tail<3>());
  const Eigen::Matrix<double, 6, 7> J_r_dot =
      computeAnalyticJacobianDotNumerical(q, dq, r.tail<3>(), dt);

  const Vector6d dr = J_r * dq;
  const Eigen::Vector3d desired_rpy = r_d.tail<3>();
  const Eigen::Vector3d current_rpy = r.tail<3>();
  const double task_velocity_norm = dr.norm();

  // tracking error: e = r_d - r, de = dr_d - dr
  Vector6d e = r_d - r;
  e.tail<3>() = wrapEulerError(e.tail<3>());
  const Vector6d de = dr_d - dr;

  const Eigen::Matrix<double, 7, 6> J_r_pinv = dampedRightPseudoInverse(J_r);

  // zero-tracking-error style reference joint motion:
  // qdot_r = J# * rdot_d
  // qddot_r = J# * (rddot_d - Jdot * qdot_r)
  const Vector7d dq_ref = J_r_pinv * dr_d;
  const Vector7d ddq_ref = J_r_pinv * (ddr_d - J_r_dot * dq_ref);

  const Vector6d vel_residual = J_r * dq_ref - dr_d;
  const Vector6d acc_residual = J_r * ddq_ref + J_r_dot * dq_ref - ddr_d;

  const double vel_residual_norm = vel_residual.norm();
  const double acc_residual_norm = acc_residual.norm();

  const Eigen::Matrix<double, 6, 6> task_projection =
      J_r * J_r_pinv;
  const double pinv_projection_error_norm =
      (task_projection - Eigen::Matrix<double, 6, 6>::Identity()).norm();

  // nonlinear impedance law matching the slide structure:
  // tau = M(q) qddot_r + C(q,dq) qdot_r + J^T [ D (rdot_d-rdot) + K (r_d-r) ]
  // NOTE: gravity is NOT added here because the effort interface already compensates gravity.
  Vector7d tau_impedance =
      J_r.transpose() * (D_m_ * de + K_m_ * e);

  Vector7d tau_ff =
      M * ddq_ref +
      C * dq_ref;

  Vector7d tau = tau_impedance;

  if (use_nonlinear_feedforward_) {
    tau += tau_ff;
  }

  // torque-space nullspace projector
  const Matrix7d N_tau =
      Matrix7d::Identity() - J_r.transpose() * J_r_pinv.transpose();

  const Vector7d tau_null =
      N_tau * (n_stiffness_ * (desired_qn_ - q) - 2.0 * std::sqrt(n_stiffness_) * dq);

  tau += tau_null;

  const double tau_norm = tau.norm();

  const double tau_impedance_norm = tau_impedance.norm();
  const double tau_ff_norm = tau_ff.norm();
  const double tau_null_norm = tau_null.norm();

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
        << (use_nonlinear_feedforward_ ? 1 : 0) << ","
        << tau_norm
        << "\n";

    ++log_write_counter_;
    if (log_write_counter_ % 200 == 0) {
      error_log_file_.flush();
    }
  }

  return controller_interface::return_type::OK;
}

CallbackReturn CartesianImpedanceExampleController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<bool>("enable_error_logging", false);
    auto_declare<bool>("use_constant_reference", true);
    auto_declare<bool>("use_nonlinear_feedforward", true);
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
    use_nonlinear_feedforward_ = get_node()->get_parameter("use_nonlinear_feedforward").as_bool();
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

  // measured q from native state
  const Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
  desired_qn_ = q;

  // initialize desired anchor using the SAME Pinocchio EE frame as control/logging
  pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
  pinocchio::updateFramePlacements(pin_model_, *pin_data_);

  desired_position_ = pin_data_->oMf[ee_frame_id_].translation();
  desired_orientation_ = Quaterniond(pin_data_->oMf[ee_frame_id_].rotation());
  desired_orientation_.normalize();

  // same stiffness settings as base impedance controller
  const double pos_stiff = 400.0;
  const double rot_stiff = 20.0;

  K_m_.setZero();
  D_m_.setZero();

  K_m_.topLeftCorner<3, 3>() = pos_stiff * Matrix3d::Identity();
  K_m_.bottomRightCorner<3, 3>() = rot_stiff * Matrix3d::Identity();

  D_m_.topLeftCorner<3, 3>() = 2.0 * std::sqrt(pos_stiff) * Matrix3d::Identity();
  D_m_.bottomRightCorner<3, 3>() =
      0.8 * 2.0 * std::sqrt(rot_stiff) * Matrix3d::Identity();

  // first debug the main task controller without nullspace interference
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