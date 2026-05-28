#include "cps_safety_monitor/reachable_safety_monitor.hpp"

#include <algorithm>
#include <cmath>

namespace cps_safety_monitor {

namespace {

constexpr double kMinDt = 1.0e-6;
constexpr double kLambdaReg = 1.0e-9;
constexpr double kSmallPositive = 1.0e-9;

Matrix6d applyMatrixRateLimit(const Matrix6d& current,
                              const Matrix6d& target,
                              double rate_limit,
                              double dt) {
  const double step = std::max(rate_limit, 0.0) * std::max(dt, kMinDt);
  Matrix6d out = current;

  for (int i = 0; i < out.rows(); ++i) {
    for (int j = 0; j < out.cols(); ++j) {
      const double delta =
          std::clamp(target(i, j) - current(i, j), -step, step);
      out(i, j) = current(i, j) + delta;
    }
  }

  return out;
}

}  // namespace

MonitorResult verifyReachablePlan(const VerifiedPlan& plan,
                                  const Vector3d& current_position,
                                  const Vector6d& ee_twist,
                                  const Matrix7d& inertia,
                                  const Matrix37d& Jv,
                                  const SafetyMonitorConfig& config) {
  MonitorResult out;

  const Vector3d n = config.human_workspace.normal();

  Vector3d x_pred = current_position;
  Vector3d v_pred = ee_twist.head<3>();

  Matrix3d task_inertia_inv = Jv * inertia.inverse() * Jv.transpose();
  task_inertia_inv.diagonal().array() += kLambdaReg;

  const double denom0 = (n.transpose() * task_inertia_inv * n)(0, 0);
  out.m_eff_n = 1.0 / std::max(denom0, kSmallPositive);

  const double rho_p = std::max(0.0, config.tracking_pos_error_bound);
  const double rho_v = std::max(0.0, config.tracking_vel_error_bound);

  out.current_pos_error_radius = rho_p;
  out.current_vel_error_radius = rho_v;
  out.worst_case_pos_error_radius = rho_p;
  out.worst_case_vel_error_radius = rho_v;

  const double v_n_now_raw = n.dot(v_pred);
  out.v_n_now = v_n_now_raw;
  out.Tn_now = 0.5 * out.m_eff_n * v_n_now_raw * v_n_now_raw;

  out.v_safe = std::sqrt(
      std::max(
          2.0 * config.safe_collision_energy_joule /
              std::max(out.m_eff_n, kSmallPositive),
          0.0));

  const double inflated_contact_radius =
      config.human_workspace.inflatedCollisionRadius(
          config.ee_collision_radius,
          rho_p);

  out.plane_distance_now =
      config.human_workspace.signedDistanceToInflatedSphere(
          current_position,
          inflated_contact_radius);

  out.plane_distance_min = out.plane_distance_now;

  Matrix6d K_exec = config.K_runtime;
  Matrix6d D_exec = config.D_runtime;

  double T_n_max = 0.0;
  double V_n_max = 0.0;

  double terminal_T_ub = 0.0;
  double terminal_V_ub = 0.0;
  bool terminal_sample_found = false;

  auto eval_sample = [&](const ImpedanceSample& s, double dtp) {
    K_exec = applyMatrixRateLimit(K_exec, s.K, config.k_rate_limit, dtp);
    D_exec = applyMatrixRateLimit(D_exec, s.D, config.d_rate_limit, dtp);

    const Matrix3d Kp = K_exec.topLeftCorner<3, 3>();
    const Matrix3d Dp = D_exec.topLeftCorner<3, 3>();

    const Vector3d force_pred =
        Kp * (s.p - x_pred) - Dp * (v_pred - s.dp);

    Vector3d a_pred = Vector3d::Zero();

    if (config.use_dynamic_consistent_impedance) {
      a_pred = s.ddp + task_inertia_inv * force_pred;
    } else {
      a_pred = task_inertia_inv * force_pred;
    }

    const Vector3d x_next =
        x_pred + v_pred * dtp + 0.5 * a_pred * dtp * dtp;

    const Vector3d v_next = v_pred + a_pred * dtp;

    const double d_pred =
        config.human_workspace.signedDistanceToInflatedSphere(
            x_pred,
            inflated_contact_radius);

    const double d_next =
        config.human_workspace.signedDistanceToInflatedSphere(
            x_next,
            inflated_contact_radius);

    const double d_segment =
        config.human_workspace.signedDistanceSegmentToInflatedSphere(
            x_pred,
            x_next,
            inflated_contact_radius);

    out.plane_distance_min = std::min(out.plane_distance_min, d_segment);

    const bool contact_possible_step = d_segment <= 0.0;

    const double k_n = std::max((n.transpose() * Kp * n)(0, 0), 0.0);
    const double e_n = std::abs(n.dot(x_next - s.p)) + rho_p;
    const double v_n = std::abs(n.dot(v_next)) + rho_v;

    const double T_n_ub = 0.5 * out.m_eff_n * v_n * v_n;
    const double V_n_ub = 0.5 * k_n * e_n * e_n;

    if (contact_possible_step) {
      out.monitored_contact_possible = true;

      if (T_n_ub > T_n_max) {
        T_n_max = T_n_ub;

        out.worst_case_contact_found = true;
        out.worst_case_contact_time = s.t;
        out.worst_case_plane_distance_at_candidate = std::min(d_pred, d_next);
        out.worst_case_nominal_forward_progress = n.dot(x_next - current_position);
        out.worst_case_v_n_ub = v_n;
        out.worst_case_Tn_ub = T_n_ub;

        out.worst_case_a_pos = 0.0;
        out.worst_case_a_brake = 0.0;
        out.worst_case_a_net = 0.0;
      }

      if (V_n_ub > V_n_max) {
        V_n_max = V_n_ub;
        out.worst_case_V_potential_ub = V_n_ub;
      }

      if (!s.failsafe && !out.nominal_contact_sample_found) {
        Vector3d x_contact = Vector3d::Zero();
        config.human_workspace.signedDistanceSegmentToInflatedSphere(
            x_pred,
            x_next,
            inflated_contact_radius,
            &x_contact);

        out.nominal_contact_sample_found = true;
        out.nominal_contact_time = s.t;
        out.nominal_contact_distance = d_segment;
        out.v_n_contact_nominal = v_n;
        out.Tn_contact_nominal = T_n_ub;
        out.nominal_contact_point_world = x_contact;
      }
    }

    if (s.failsafe) {
      terminal_T_ub = T_n_ub;
      terminal_V_ub = V_n_ub;
      terminal_sample_found = true;
    }

    x_pred = x_next;
    v_pred = v_next;
  };

  double t_prev = plan.anchor.t;

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

  out.h_geom = out.plane_distance_min;
  out.worst_case_Tn_ub = T_n_max;
  out.worst_case_V_potential_ub = V_n_max;
  out.terminal_energy_ub = terminal_T_ub + terminal_V_ub;

  const double L_TF_eff =
      std::max(0.0, config.safe_collision_energy_joule -
                        config.energy_budget_margin_joule);

  const double L_QS_eff =
      std::max(0.0, config.clamping_energy_budget_joule -
                        config.energy_budget_margin_joule);

  const double L_F_eff = std::min(L_TF_eff, L_QS_eff);

  out.h_monitored_energy = L_TF_eff - out.worst_case_Tn_ub;
  out.h_clamping_energy = L_QS_eff - out.worst_case_V_potential_ub;
  out.h_terminal_energy = L_F_eff - out.terminal_energy_ub;

  out.collision_energy_unsafe = out.worst_case_Tn_ub > L_TF_eff;
  out.clamping_energy_unsafe = out.worst_case_V_potential_ub > L_QS_eff;
  out.terminal_energy_unsafe = out.terminal_energy_ub > L_F_eff;

  out.monitored_unsafe =
      out.collision_energy_unsafe ||
      out.clamping_energy_unsafe ||
      out.terminal_energy_unsafe;

  out.predicted_trigger = out.monitored_unsafe;

  return out;
}

}  // namespace cps_safety_monitor
