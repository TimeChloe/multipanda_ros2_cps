#pragma once

#include <array>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace cps_controllers {

using Matrix3d = Eigen::Matrix3d;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Matrix67d = Eigen::Matrix<double, 6, 7>;
using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Quaterniond = Eigen::Quaterniond;

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
