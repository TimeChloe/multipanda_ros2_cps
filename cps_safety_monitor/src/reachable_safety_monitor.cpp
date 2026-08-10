#include "cps_safety_monitor/reachable_safety_monitor.hpp"

#include <algorithm>
#include <cmath>

namespace cps_safety_monitor {

namespace {

constexpr double kMinDt = 1.0e-6;
constexpr double kLambdaReg = 1.0e-9;
constexpr double kSmallPositive = 1.0e-9;

Vector3d normalizedOrZero(const Vector3d& v) {
  const double norm = v.norm();
  if (norm < 1.0e-9) {
    return Vector3d::Zero();
  }
  return v / norm;
}

Vector3d fallbackDirection(const Vector3d& point,
                           const Vector3d& velocity,
                           const cps_human_workspace::HumanWorkspace& human_workspace,
                           double time_sec) {
  Vector3d direction =
      normalizedOrZero(point - human_workspace.centerAtTime(time_sec));
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  direction = normalizedOrZero(velocity);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  return human_workspace.direction();
}

Vector3d contactNormalDirection(
    const Vector3d& robot_point,
    const Vector3d& human_center,
    const Vector3d& fallback_point,
    const Vector3d& fallback_velocity,
    const cps_human_workspace::HumanWorkspace& human_workspace,
    double time_sec) {
  Vector3d direction = normalizedOrZero(robot_point - human_center);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }
  return fallbackDirection(
      fallback_point, fallback_velocity, human_workspace, time_sec);
}

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

double directionalBlockRadius(
    const Matrix6d& tube,
    int block_start,
    const Vector3d& direction) {
  const Vector3d unit_direction = normalizedOrZero(direction);
  if (unit_direction.squaredNorm() <= 0.0) {
    return maxBlockRadius(tube, block_start);
  }
  const Matrix3d block =
      0.5 * (tube.block<3, 3>(block_start, block_start) +
             tube.block<3, 3>(block_start, block_start).transpose());
  const double radius_squared =
      (unit_direction.transpose() * block * unit_direction)(0, 0);
  return std::sqrt(std::max(0.0, radius_squared));
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
                                  const Vector6d& ee_twist,
                                  const Matrix7d& inertia,
                                  const Matrix37d& Jv,
                                  const SafetyMonitorConfig& config) {
  MonitorResult out;

  Vector3d x_pred = current_position;
  Vector3d v_pred = ee_twist.head<3>();

  Matrix3d task_inertia_inv = Jv * inertia.inverse() * Jv.transpose();
  task_inertia_inv.diagonal().array() += kLambdaReg;

  const Vector3d direction_now =
      contactNormalDirection(
          current_position,
          config.human_workspace.centerAtTime(config.wall_time_sec),
          current_position,
          v_pred,
          config.human_workspace,
          config.wall_time_sec);
  const double denom0 = (direction_now.transpose() * task_inertia_inv * direction_now)(0, 0);
  out.m_eff_n = 1.0 / std::max(denom0, kSmallPositive);

  const double acc_error_bound =
      std::max(0.0, config.tracking_acc_error_bound);
  Matrix6d tracking_tube = Matrix6d::Zero();

  // The current state is measured, so the instantaneous contact check should
  // use the measured collision point and twist directly. The tracking tube is
  // reserved for future prediction uncertainty around the measured-state
  // rollout.
  out.current_pos_error_radius = 0.0;
  out.current_vel_error_radius = 0.0;
  out.worst_case_pos_error_radius = maxBlockRadius(tracking_tube, 0);
  out.worst_case_vel_error_radius = maxBlockRadius(tracking_tube, 3);

  const double v_n_now_raw = direction_now.dot(v_pred);
  out.v_n_now = v_n_now_raw;
  out.Tn_now = 0.5 * out.m_eff_n * v_n_now_raw * v_n_now_raw;
  out.v_n_now_tube = std::abs(v_n_now_raw);
  out.Tn_now_tube = 0.5 * out.m_eff_n * out.v_n_now_tube * out.v_n_now_tube;

  out.v_safe = std::sqrt(
      std::max(
          2.0 * config.energy_budget_joule /
              std::max(out.m_eff_n, kSmallPositive),
          0.0));

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

    const Matrix3d Kp_raw = K_exec.topLeftCorner<3, 3>();
    const Matrix3d Dp_raw = D_exec.topLeftCorner<3, 3>();
    const Matrix3d Kp = positiveSemidefinitePart(Kp_raw);
    const Matrix3d Dp = positiveSemidefinitePart(Dp_raw);

    const Vector3d force_pred =
        Kp_raw * (s.p - x_pred) - Dp_raw * (v_pred - s.dp);

    Vector3d a_pred = Vector3d::Zero();

    if (config.use_dynamic_consistent_impedance) {
      a_pred = s.ddp + task_inertia_inv * force_pred;
    } else {
      a_pred = task_inertia_inv * force_pred;
    }

    const Vector3d x_next =
        x_pred + v_pred * dtp + 0.5 * a_pred * dtp * dtp;

    const Vector3d v_next = v_pred + a_pred * dtp;

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

    const double d_pred =
        config.human_workspace.signedDistanceToInflatedSphere(
            x_pred,
            inflated_contact_radius_segment,
            segment_start_time_sec);

    Vector3d closest_predicted_point = Vector3d::Zero();
    Vector3d closest_human_center = Vector3d::Zero();
    const double d_segment =
        config.human_workspace.signedDistanceSegmentToInflatedSphere(
            x_pred,
            x_next,
            inflated_contact_radius_segment,
            segment_start_time_sec,
            segment_end_time_sec,
            &closest_predicted_point,
            &closest_human_center);

    out.workspace_distance_min = std::min(out.workspace_distance_min, d_segment);

    const Vector3d direction =
        contactNormalDirection(
            closest_predicted_point,
            closest_human_center,
            x_next,
            v_next,
            config.human_workspace,
            segment_end_time_sec);
    const double denom = (direction.transpose() * task_inertia_inv * direction)(0, 0);
    const double m_eff = 1.0 / std::max(denom, kSmallPositive);
    const double k_n =
        std::max((direction.transpose() * Kp * direction)(0, 0), 0.0);
    const double rho_p_dir =
        directionalBlockRadius(tracking_tube_next, 0, direction);
    const double rho_v_dir =
        directionalBlockRadius(tracking_tube_next, 3, direction);
    const double e_n = std::abs(direction.dot(x_next - s.p)) + rho_p_dir;
    const double v_n = std::abs(direction.dot(v_next)) + rho_v_dir;

    const double T_n_ub = 0.5 * m_eff * v_n * v_n;
    const double V_n_ub = 0.5 * k_n * e_n * e_n;

    const bool contact_possible_step = d_segment <= 0.0;
    if (contact_possible_step && !out.nominal_contact_sample_found) {
      const bool contact_at_segment_start = d_pred <= 0.0;
      out.nominal_contact_sample_found = true;
      out.nominal_contact_time = contact_at_segment_start ? t_prev : s.t;
      out.nominal_contact_distance = d_segment;
      out.v_n_contact_nominal = v_n;
      out.Tn_contact_nominal = T_n_ub;
      out.nominal_contact_point_world =
          contact_at_segment_start ? x_pred : closest_predicted_point;
    }

    const double E_contact_ub = T_n_ub + V_n_ub;
    if (contact_possible_step && E_contact_ub > E_contact_max) {
      E_contact_max = E_contact_ub;

      out.worst_case_contact_time = s.t;
      out.worst_case_workspace_distance_at_candidate = d_segment;
      out.worst_case_nominal_forward_progress = direction.dot(x_next - current_position);
      out.worst_case_v_n_ub = v_n;
      out.worst_case_Tn_ub = T_n_ub;
      out.worst_case_V_potential_ub = V_n_ub;
      out.worst_case_contact_energy_ub = E_contact_ub;

      out.worst_case_a_pos = 0.0;
      out.worst_case_a_brake = 0.0;
      out.worst_case_a_net = 0.0;
    }

    if (s.failsafe) {
      terminal_T_ub = T_n_ub;
      terminal_V_ub = V_n_ub;
      terminal_sample_found = true;
    }

    x_pred = x_next;
    v_pred = v_next;
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
  out.worst_case_contact_energy_ub = E_contact_max;
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
      out.contact_relevant_for_energy && out.Tn_now_tube > energy_budget_eff;
  const bool predicted_contact_energy_unsafe =
      predicted_contact_requires_verification &&
      out.worst_case_contact_energy_ub > energy_budget_eff;

  out.predicted_trigger = predicted_contact_energy_unsafe;

  const double monitored_contact_energy_ub =
      std::max(out.worst_case_contact_energy_ub,
               out.contact_relevant_for_energy ? out.Tn_now_tube : 0.0);

  out.h_monitored_energy = energy_budget_eff - monitored_contact_energy_ub;
  out.h_clamping_energy = energy_budget_eff - out.worst_case_V_potential_ub;
  out.h_terminal_energy = energy_budget_eff - out.terminal_energy_ub;

  out.collision_energy_unsafe =
      current_collision_energy_unsafe || predicted_contact_energy_unsafe;
  out.clamping_energy_unsafe = false;
  out.terminal_energy_unsafe = false;

  out.monitored_unsafe =
      out.predicted_trigger ||
      (out.contact_relevant_for_energy && current_collision_energy_unsafe);

  return out;
}

}  // namespace cps_safety_monitor
