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

Vector3d fallbackDirection(const Vector3d& x_pred,
                           const Vector3d& v_pred,
                           const cps_human_workspace::HumanWorkspace& human_workspace,
                           double time_sec) {
  Vector3d direction = normalizedOrZero(v_pred);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  direction = normalizedOrZero(human_workspace.centerAtTime(time_sec) - x_pred);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  return human_workspace.direction();
}

Vector3d commandDirection(const ImpedanceSample& s,
                          const Vector3d& previous_command_position,
                          const Vector3d& x_pred,
                          const Vector3d& v_pred,
                          const cps_human_workspace::HumanWorkspace& human_workspace,
                          double time_sec) {
  Vector3d direction = normalizedOrZero(s.dp);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  direction = normalizedOrZero(s.p - previous_command_position);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  direction = normalizedOrZero(s.p - x_pred);
  if (direction.squaredNorm() > 0.0) {
    return direction;
  }

  return fallbackDirection(x_pred, v_pred, human_workspace, time_sec);
}

Vector3d initialCommandDirection(const VerifiedPlan& plan,
                                 const Vector3d& current_position,
                                 const Vector3d& current_velocity,
                                 const cps_human_workspace::HumanWorkspace& human_workspace,
                                 double time_sec) {
  if (!plan.intended.empty()) {
    const double sample_time_sec = time_sec + plan.intended.front().t;
    return commandDirection(plan.intended.front(),
                            plan.anchor.p,
                            current_position,
                            current_velocity,
                            human_workspace,
                            sample_time_sec);
  }
  if (!plan.failsafe.empty()) {
    const double sample_time_sec = time_sec + plan.failsafe.front().t;
    return commandDirection(plan.failsafe.front(),
                            plan.anchor.p,
                            current_position,
                            current_velocity,
                            human_workspace,
                            sample_time_sec);
  }
  return fallbackDirection(current_position, current_velocity, human_workspace, time_sec);
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
      initialCommandDirection(
          plan,
          current_position,
          v_pred,
          config.human_workspace,
          config.wall_time_sec);
  const double denom0 = (direction_now.transpose() * task_inertia_inv * direction_now)(0, 0);
  out.m_eff_n = 1.0 / std::max(denom0, kSmallPositive);

  const double rho_p = std::max(0.0, config.tracking_pos_error_bound);
  const double rho_v = std::max(0.0, config.tracking_vel_error_bound);

  out.current_pos_error_radius = rho_p;
  out.current_vel_error_radius = rho_v;
  out.worst_case_pos_error_radius = rho_p;
  out.worst_case_vel_error_radius = rho_v;

  const double v_n_now_raw = direction_now.dot(v_pred);
  out.v_n_now = v_n_now_raw;
  out.Tn_now = 0.5 * out.m_eff_n * v_n_now_raw * v_n_now_raw;
  out.v_n_now_tube = std::abs(v_n_now_raw) + rho_v;
  out.Tn_now_tube = 0.5 * out.m_eff_n * out.v_n_now_tube * out.v_n_now_tube;

  out.v_safe = std::sqrt(
      std::max(
          2.0 * config.energy_budget_joule /
              std::max(out.m_eff_n, kSmallPositive),
          0.0));

  const double inflated_contact_radius =
      config.human_workspace.inflatedCollisionRadius(
          config.ee_collision_radius,
          rho_p);

  out.workspace_distance_now =
      config.human_workspace.signedDistanceToInflatedSphere(
          current_position,
          inflated_contact_radius,
          config.wall_time_sec);

  out.workspace_distance_min = out.workspace_distance_now;

  Matrix6d K_exec = config.K_runtime;
  Matrix6d D_exec = config.D_runtime;

  double T_n_contact_max = 0.0;
  double V_n_contact_max = 0.0;

  double terminal_T_ub = 0.0;
  double terminal_V_ub = 0.0;
  bool terminal_sample_found = false;
  Vector3d previous_command_position = plan.anchor.p;

  double t_prev = plan.anchor.t;

  auto eval_sample = [&](const ImpedanceSample& s, double dtp) {
    const double segment_start_time_sec = config.wall_time_sec + t_prev;
    const double segment_end_time_sec = config.wall_time_sec + s.t;

    K_exec = s.K;
    D_exec = s.D;

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
            inflated_contact_radius,
            segment_start_time_sec);

    Vector3d closest_predicted_point = Vector3d::Zero();
    const double d_segment =
        config.human_workspace.signedDistanceSegmentToInflatedSphere(
            x_pred,
            x_next,
            inflated_contact_radius,
            segment_start_time_sec,
            segment_end_time_sec,
            &closest_predicted_point);

    out.workspace_distance_min = std::min(out.workspace_distance_min, d_segment);

    const Vector3d direction =
        commandDirection(
            s,
            previous_command_position,
            x_pred,
            v_pred,
            config.human_workspace,
            segment_end_time_sec);
    const double denom = (direction.transpose() * task_inertia_inv * direction)(0, 0);
    const double m_eff = 1.0 / std::max(denom, kSmallPositive);
    const double k_n =
        std::max((direction.transpose() * Kp * direction)(0, 0), 0.0);
    const double e_n = std::abs(direction.dot(x_next - s.p)) + rho_p;
    const double v_n = std::abs(direction.dot(v_next)) + rho_v;

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

    if (contact_possible_step && T_n_ub > T_n_contact_max) {
      T_n_contact_max = T_n_ub;

      out.worst_case_contact_time = s.t;
      out.worst_case_workspace_distance_at_candidate = d_segment;
      out.worst_case_nominal_forward_progress = direction.dot(x_next - current_position);
      out.worst_case_v_n_ub = v_n;
      out.worst_case_Tn_ub = T_n_ub;

      out.worst_case_a_pos = 0.0;
      out.worst_case_a_brake = 0.0;
      out.worst_case_a_net = 0.0;
    }

    if (contact_possible_step && V_n_ub > V_n_contact_max) {
      V_n_contact_max = V_n_ub;
      out.worst_case_V_potential_ub = V_n_ub;
    }

    if (s.failsafe) {
      terminal_T_ub = T_n_ub;
      terminal_V_ub = V_n_ub;
      terminal_sample_found = true;
    }

    x_pred = x_next;
    v_pred = v_next;
    previous_command_position = s.p;
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
  out.worst_case_Tn_ub = T_n_contact_max;
  out.worst_case_V_potential_ub = V_n_contact_max;
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
  const bool predicted_collision_energy_unsafe =
      predicted_contact_requires_verification &&
      out.worst_case_Tn_ub > energy_budget_eff;
  const bool predicted_clamping_energy_unsafe =
      predicted_contact_requires_verification &&
      out.worst_case_V_potential_ub > energy_budget_eff;
  const bool predicted_terminal_energy_unsafe =
      predicted_contact_requires_verification &&
      out.terminal_energy_ub > energy_budget_eff;

  out.predicted_trigger =
      predicted_collision_energy_unsafe ||
      predicted_clamping_energy_unsafe ||
      predicted_terminal_energy_unsafe;

  const double monitored_collision_energy_ub =
      std::max(out.worst_case_Tn_ub,
               out.contact_relevant_for_energy ? out.Tn_now_tube : 0.0);

  out.h_monitored_energy = energy_budget_eff - monitored_collision_energy_ub;
  out.h_clamping_energy = energy_budget_eff - out.worst_case_V_potential_ub;
  out.h_terminal_energy = energy_budget_eff - out.terminal_energy_ub;

  out.collision_energy_unsafe =
      current_collision_energy_unsafe || predicted_collision_energy_unsafe;
  out.clamping_energy_unsafe = predicted_clamping_energy_unsafe;
  out.terminal_energy_unsafe = predicted_terminal_energy_unsafe;

  out.monitored_unsafe =
      out.predicted_trigger ||
      (out.contact_relevant_for_energy && current_collision_energy_unsafe);

  return out;
}

}  // namespace cps_safety_monitor
