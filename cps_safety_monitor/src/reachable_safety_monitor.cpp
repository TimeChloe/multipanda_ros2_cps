#include "cps_safety_monitor/reachable_safety_monitor.hpp"

#include <algorithm>
#include <cmath>

namespace cps_safety_monitor {

namespace {

constexpr double kMinDt = 1.0e-6;
constexpr double kLambdaReg = 1.0e-9;
constexpr double kSmallPositive = 1.0e-9;

Matrix6d symmetrized(const Matrix6d& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

Matrix3d positiveSemidefinitePart(const Matrix3d& matrix) {
  const Matrix3d symmetric = 0.5 * (matrix + matrix.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix3d> eig(symmetric);
  if (eig.info() != Eigen::Success) {
    return symmetric;
  }
  Matrix3d psd =
      eig.eigenvectors() *
      eig.eigenvalues().cwiseMax(0.0).asDiagonal() *
      eig.eigenvectors().transpose();
  return 0.5 * (psd + psd.transpose());
}

Matrix6d positiveSemidefinitePart(const Matrix6d& matrix) {
  const Matrix6d symmetric = symmetrized(matrix);
  const Eigen::SelfAdjointEigenSolver<Matrix6d> eig(symmetric);
  if (eig.info() != Eigen::Success) {
    return symmetric;
  }
  Matrix6d psd =
      eig.eigenvectors() *
      eig.eigenvalues().cwiseMax(0.0).asDiagonal() *
      eig.eigenvectors().transpose();
  return symmetrized(psd);
}

Vector3d orientationError(const Quaterniond& current,
                          const Quaterniond& desired) {
  Quaterniond q_current = current.normalized();
  Quaterniond q_desired = desired.normalized();
  if (q_desired.coeffs().dot(q_current.coeffs()) < 0.0) {
    q_current.coeffs() *= -1.0;
  }
  const Quaterniond q_error(q_current * q_desired.inverse());
  const Eigen::AngleAxisd angle_axis(q_error);
  return angle_axis.axis() * angle_axis.angle();
}

Quaterniond integrateWorldAngularVelocity(const Quaterniond& orientation,
                                           const Vector3d& angular_velocity,
                                           double dt) {
  const double angle = angular_velocity.norm() * std::max(dt, 0.0);
  if (angle <= kSmallPositive) {
    return orientation.normalized();
  }
  const Quaterniond delta(
      Eigen::AngleAxisd(angle, angular_velocity.normalized()));
  return (delta * orientation).normalized();
}

double quadraticEnergyUpperBound(const Vector6d& state,
                                 const Matrix6d& metric_psd,
                                 double translational_error_radius) {
  const double nominal_metric_norm = std::sqrt(std::max(
      0.0, (state.transpose() * metric_psd * state)(0, 0)));
  // The reachable tube currently bounds translational position/velocity only.
  // Do not pretend that this radius is also an angular uncertainty: that would
  // be dimensionally wrong and needlessly conservative. For a disturbance
  // delta=[delta_translation, 0], only the leading 3x3 metric block enters
  // delta^T metric delta; the triangle inequality then gives this upper bound.
  const Matrix3d translational_metric = positiveSemidefinitePart(
      Matrix3d(metric_psd.topLeftCorner<3, 3>()));
  const Eigen::SelfAdjointEigenSolver<Matrix3d> eig(translational_metric);
  const double max_eigenvalue =
      eig.info() == Eigen::Success
          ? std::max(0.0, eig.eigenvalues().maxCoeff())
          : std::max(0.0, translational_metric.norm());
  const double uncertainty_metric_norm =
      std::sqrt(max_eigenvalue) *
      std::max(0.0, translational_error_radius);
  const double upper_metric_norm =
      nominal_metric_norm + uncertainty_metric_norm;
  return 0.5 * upper_metric_norm * upper_metric_norm;
}

double maxBlockRadius(const Matrix6d& tube, int block_start) {
  const Matrix3d block =
      0.5 * (tube.block<3, 3>(block_start, block_start) +
             tube.block<3, 3>(block_start, block_start).transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix3d> eig(block);
  if (eig.info() != Eigen::Success) {
    return std::sqrt(std::max(0.0, block.norm()));
  }
  return std::sqrt(std::max(0.0, eig.eigenvalues().maxCoeff()));
}

Matrix6d propagateTrackingTube(
    const Matrix6d& tube,
    const Matrix3d& task_inertia_inv,
    const Matrix3d& Kp,
    const Matrix3d& Dp,
    double dt,
    double acc_error_bound) {
  const double h = std::max(dt, kMinDt);
  Matrix6d A = Matrix6d::Identity();
  A.topRightCorner<3, 3>() = h * Matrix3d::Identity();
  A.bottomLeftCorner<3, 3>() = -h * task_inertia_inv * Kp;
  A.bottomRightCorner<3, 3>() =
      Matrix3d::Identity() - h * task_inertia_inv * Dp;

  Eigen::Matrix<double, 6, 3> B = Eigen::Matrix<double, 6, 3>::Zero();
  B.topRows<3>() = 0.5 * h * h * Matrix3d::Identity();
  B.bottomRows<3>() = h * Matrix3d::Identity();

  const double acc_bound = std::max(0.0, acc_error_bound);
  Matrix6d propagated =
      A * tube * A.transpose() + acc_bound * acc_bound * B * B.transpose();
  return symmetrized(propagated);
}

}  // namespace

MonitorResult verifyReachablePlan(const VerifiedPlan& plan,
                                  const Vector3d& current_position,
                                  const Quaterniond& current_orientation,
                                  const Vector6d& ee_twist,
                                  const Matrix7d& inertia,
                                  const Matrix67d& J_geo,
                                  const SafetyMonitorConfig& config) {
  MonitorResult out;

  Vector3d x_pred = current_position;
  Vector3d v_pred = ee_twist.head<3>();
  Quaterniond q_pred = current_orientation.normalized();
  Vector3d w_pred = ee_twist.tail<3>();

  const Eigen::LDLT<Matrix7d> inertia_ldlt(inertia);
  Matrix7d inertia_inv = Matrix7d::Zero();
  if (inertia_ldlt.info() == Eigen::Success) {
    inertia_inv = inertia_ldlt.solve(Matrix7d::Identity());
  } else {
    inertia_inv = inertia.inverse();
  }
  Matrix6d cartesian_inertia_inv = J_geo * inertia_inv * J_geo.transpose();
  cartesian_inertia_inv = symmetrized(cartesian_inertia_inv);
  Matrix6d cartesian_inertia_inv_regularized = cartesian_inertia_inv;
  cartesian_inertia_inv_regularized.diagonal().array() += kLambdaReg;
  const Eigen::LDLT<Matrix6d> cartesian_inertia_ldlt(
      cartesian_inertia_inv_regularized);
  Matrix6d cartesian_inertia = Matrix6d::Identity() / kLambdaReg;
  if (cartesian_inertia_ldlt.info() == Eigen::Success) {
    cartesian_inertia = symmetrized(
        cartesian_inertia_ldlt.solve(Matrix6d::Identity()));
  }
  cartesian_inertia = positiveSemidefinitePart(cartesian_inertia);
  Matrix3d task_inertia_inv =
      cartesian_inertia_inv.topLeftCorner<3, 3>();
  task_inertia_inv.diagonal().array() += kLambdaReg;

  const double acc_error_bound =
      std::max(0.0, config.tracking_acc_error_bound);
  Matrix6d tracking_tube = Matrix6d::Zero();

  out.worst_case_pos_error_radius = maxBlockRadius(tracking_tube, 0);
  out.worst_case_vel_error_radius = maxBlockRadius(tracking_tube, 3);

  if (config.current_energy_reference_valid) {
    const ImpedanceSample& current_reference =
        config.current_energy_reference;
    Vector6d current_error = Vector6d::Zero();
    current_error.head<3>() = current_position - current_reference.p;
    current_error.tail<3>() =
        orientationError(current_orientation, current_reference.q);
    const Matrix6d current_stiffness =
        positiveSemidefinitePart(current_reference.K);
    out.current_cartesian_kinetic_energy = quadraticEnergyUpperBound(
        ee_twist, cartesian_inertia, 0.0);
    out.current_cartesian_potential_energy = quadraticEnergyUpperBound(
        current_error, current_stiffness, 0.0);
    out.current_cartesian_control_energy =
        out.current_cartesian_kinetic_energy +
        out.current_cartesian_potential_energy;
    out.current_cartesian_energy_valid = true;
  }

  const double inflated_contact_radius_now =
      config.human_workspace.inflatedCollisionRadius(
          config.ee_collision_radius,
          0.0);

  out.workspace_distance_now =
      config.human_workspace.signedDistanceToInflatedSphere(
          current_position,
          inflated_contact_radius_now,
          config.wall_time_sec);

  out.workspace_distance_min = out.workspace_distance_now;

  Matrix6d K_exec = config.K_runtime;
  Matrix6d D_exec = config.D_runtime;

  double E_contact_max = 0.0;

  double terminal_T_ub = 0.0;
  double terminal_V_ub = 0.0;
  bool terminal_sample_found = false;
  double t_prev = plan.anchor.t;

  auto eval_sample = [&](const ImpedanceSample& s, double dtp) {
    const double segment_start_time_sec = config.wall_time_sec + t_prev;
    const double segment_end_time_sec = config.wall_time_sec + s.t;

    K_exec = s.K;
    D_exec = s.D;

    const Matrix6d K_cartesian = positiveSemidefinitePart(K_exec);
    const Matrix3d Kp_raw = K_exec.topLeftCorner<3, 3>();
    const Matrix3d Dp_raw = D_exec.topLeftCorner<3, 3>();
    const Matrix3d Kp = positiveSemidefinitePart(Kp_raw);
    const Matrix3d Dp = positiveSemidefinitePart(Dp_raw);

    Vector6d pose_error = Vector6d::Zero();
    pose_error.head<3>() = x_pred - s.p;
    pose_error.tail<3>() = orientationError(q_pred, s.q);
    Vector6d twist_pred = Vector6d::Zero();
    twist_pred.head<3>() = v_pred;
    twist_pred.tail<3>() = w_pred;
    Vector6d desired_twist = Vector6d::Zero();
    desired_twist.head<3>() = s.dp;
    desired_twist.tail<3>() = s.w;
    Vector6d desired_acceleration = Vector6d::Zero();
    desired_acceleration.head<3>() = s.ddp;
    desired_acceleration.tail<3>() = s.dw;
    const Vector6d wrench_pred =
        -K_exec * pose_error - D_exec * (twist_pred - desired_twist);
    Vector6d acceleration_pred = cartesian_inertia_inv * wrench_pred;
    if (config.use_dynamic_consistent_impedance) {
      acceleration_pred += desired_acceleration;
    }
    const Vector3d a_pred = acceleration_pred.head<3>();
    const Vector3d alpha_pred = acceleration_pred.tail<3>();

    const Vector3d x_next =
        x_pred + v_pred * dtp + 0.5 * a_pred * dtp * dtp;

    const Vector3d v_next = v_pred + a_pred * dtp;
    const Vector3d w_next = w_pred + alpha_pred * dtp;
    const Quaterniond q_next = integrateWorldAngularVelocity(
        q_pred, 0.5 * (w_pred + w_next), dtp);

    const Matrix6d tracking_tube_next =
        propagateTrackingTube(
            tracking_tube,
            task_inertia_inv,
            Kp,
            Dp,
            dtp,
            acc_error_bound);
    const double rho_p_segment =
        std::max(maxBlockRadius(tracking_tube, 0),
                 maxBlockRadius(tracking_tube_next, 0));
    const double inflated_contact_radius_segment =
        config.human_workspace.inflatedCollisionRadius(
            config.ee_collision_radius,
            rho_p_segment);

    const double d_segment =
        config.human_workspace.signedDistanceSegmentToInflatedSphere(
            x_pred,
            x_next,
            inflated_contact_radius_segment,
            segment_start_time_sec,
            segment_end_time_sec);

    out.workspace_distance_min = std::min(out.workspace_distance_min, d_segment);

    Vector6d next_twist = Vector6d::Zero();
    next_twist.head<3>() = v_next;
    next_twist.tail<3>() = w_next;
    Vector6d next_pose_error = Vector6d::Zero();
    next_pose_error.head<3>() = x_next - s.p;
    next_pose_error.tail<3>() = orientationError(q_next, s.q);
    const double cartesian_T_ub = quadraticEnergyUpperBound(
        next_twist,
        cartesian_inertia,
        maxBlockRadius(tracking_tube_next, 3));
    const double cartesian_V_ub = quadraticEnergyUpperBound(
        next_pose_error,
        K_cartesian,
        maxBlockRadius(tracking_tube_next, 0));

    const bool contact_possible_step = d_segment <= 0.0;
    // Candidate acceptance is based on the maximum complete Cartesian control
    // energy over every monitored sample that may intersect the workspace.
    // The normal-projected energy remains available only as a collision
    // diagnostic and must not authorize a 6D-energetic trajectory.
    const double E_contact_ub = cartesian_T_ub + cartesian_V_ub;
    if (contact_possible_step && E_contact_ub > E_contact_max) {
      E_contact_max = E_contact_ub;

      out.worst_case_contact_time = s.t;
      out.worst_case_workspace_distance_at_candidate = d_segment;
      out.worst_case_cartesian_kinetic_energy_ub = cartesian_T_ub;
      out.worst_case_cartesian_potential_energy_ub = cartesian_V_ub;
      out.worst_case_cartesian_control_energy_ub = E_contact_ub;
    }

    if (s.failsafe) {
      terminal_T_ub = cartesian_T_ub;
      terminal_V_ub = cartesian_V_ub;
      terminal_sample_found = true;
    }

    x_pred = x_next;
    v_pred = v_next;
    q_pred = q_next;
    w_pred = w_next;
    tracking_tube = tracking_tube_next;
    out.worst_case_pos_error_radius =
        std::max(out.worst_case_pos_error_radius,
                 maxBlockRadius(tracking_tube, 0));
    out.worst_case_vel_error_radius =
        std::max(out.worst_case_vel_error_radius,
                 maxBlockRadius(tracking_tube, 3));
  };

  for (const auto& s : plan.intended) {
    const double dtp = std::max(s.t - t_prev, kMinDt);
    eval_sample(s, dtp);
    t_prev = s.t;
  }

  for (const auto& s : plan.failsafe) {
    const double dtp = std::max(s.t - t_prev, kMinDt);
    eval_sample(s, dtp);
    t_prev = s.t;
  }

  if (!terminal_sample_found) {
    terminal_T_ub = 0.0;
    terminal_V_ub = 0.0;
  }

  out.monitored_contact_possible = out.workspace_distance_min <= 0.0;
  out.workspace_distance_margin = out.workspace_distance_min;
  out.worst_case_cartesian_control_energy_ub = E_contact_max;
  out.terminal_energy_ub = terminal_T_ub + terminal_V_ub;

  const double energy_budget_eff =
      std::max(0.0, config.energy_budget_joule -
                        config.energy_budget_margin_joule);

  // SARA/PFL-style split: future predicted interaction is verified on the
  // monitored trajectory; actual current interaction is handled by the Cartesian
  // energy budget instead of rejecting the trajectory here.
  out.contact_relevant_for_energy = out.workspace_distance_now <= 0.0;
  const bool predicted_contact_requires_verification =
      out.monitored_contact_possible && !out.contact_relevant_for_energy;

  const bool current_collision_energy_unsafe =
      out.contact_relevant_for_energy &&
      out.current_cartesian_energy_valid &&
      out.current_cartesian_control_energy > energy_budget_eff;
  const bool predicted_contact_energy_unsafe =
      predicted_contact_requires_verification &&
      out.worst_case_cartesian_control_energy_ub > energy_budget_eff;

  out.predicted_trigger = predicted_contact_energy_unsafe;

  const double monitored_contact_energy_ub =
      std::max(out.worst_case_cartesian_control_energy_ub,
               out.contact_relevant_for_energy
                       && out.current_cartesian_energy_valid
                   ? out.current_cartesian_control_energy
                   : 0.0);

  out.h_monitored_energy = energy_budget_eff - monitored_contact_energy_ub;
  out.h_terminal_energy = energy_budget_eff - out.terminal_energy_ub;

  out.collision_energy_unsafe =
      current_collision_energy_unsafe || predicted_contact_energy_unsafe;
  out.monitored_unsafe =
      out.predicted_trigger ||
      (out.contact_relevant_for_energy && current_collision_energy_unsafe);

  return out;
}

}  // namespace cps_safety_monitor
