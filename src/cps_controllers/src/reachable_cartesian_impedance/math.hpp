// Private math helpers for the reachable Cartesian impedance implementation.
#pragma once

#include <array>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <cps_controllers/reachable_cartesian_impedance/types.hpp>

namespace cps_controllers {

using Matrix3d = Eigen::Matrix3d;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;
using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Quaterniond = Eigen::Quaterniond;

inline cps_controllers::SafetyMode nominalSafetyModeForMonitor(
    const cps_safety_monitor::MonitorResult& monitor) {
  return monitor.monitored_contact_possible
             ? cps_controllers::SafetyMode::kNominalContactPossible
             : cps_controllers::SafetyMode::kNominal;
}

inline cps_controllers::SafetyMode lastVerifiedSafetyModeForMonitor(
    const cps_safety_monitor::MonitorResult& monitor) {
  return monitor.monitored_contact_possible
             ? cps_controllers::SafetyMode::kLastVerifiedContactPossible
             : cps_controllers::SafetyMode::kLastVerifiedMonitored;
}

inline int executionModeForLog(bool fallback_execution) {
  return fallback_execution ? 1 : 0;
}

inline int executionModeForLog(cps_controllers::ExecutionStage stage) {
  return executionModeForLog(
      stage != cps_controllers::ExecutionStage::kCurrentVerified);
}

inline Matrix3d skewSymmetric(const Vector3d& v) {
  Matrix3d S;
  S << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return S;
}


inline Eigen::MatrixXd dampedPseudoInverse(const Eigen::MatrixXd& M,
                                           double lambda = 0.2) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto s = svd.singularValues();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().cols());
  for (int i = 0; i < s.size(); ++i) {
    S(i, i) = s(i) / (s(i) * s(i) + lambda * lambda);
  }
  return svd.matrixV() * S * svd.matrixU().transpose();
}

inline Vector3d computeOrientationError(const Quaterniond& current,
                                        const Quaterniond& desired) {
  Quaterniond q_curr = current;
  Quaterniond q_des = desired;
  if (q_des.coeffs().dot(q_curr.coeffs()) < 0.0) {
    q_curr.coeffs() << -q_curr.coeffs();
  }
  const Quaterniond q_err(q_curr * q_des.inverse());
  Eigen::AngleAxisd aa(q_err);
  return aa.axis() * aa.angle();
}

inline Quaterniond normalizedQuaternionOrIdentity(const Quaterniond& input) {
  Quaterniond normalized = input;
  const double norm = normalized.norm();
  if (!std::isfinite(norm) || norm < 1.0e-12) {
    return Quaterniond::Identity();
  }
  normalized.coeffs() /= norm;
  return normalized;
}

inline Matrix7d arrayToMatrix7d(const std::array<double, 49>& data) {
  Matrix7d out;
  for (size_t i = 0; i < 7; ++i) {
    for (size_t j = 0; j < 7; ++j) {
      out(static_cast<int>(i), static_cast<int>(j)) = data[i * 7 + j];
    }
  }
  return out;
}

inline Vector7d arrayToVector7d(const std::array<double, 7>& data) {
  Vector7d out;
  for (size_t i = 0; i < 7; ++i) {
    out(static_cast<int>(i)) = data[i];
  }
  return out;
}

}  // namespace cps_controllers
