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
    out.current_total_control_energy =
        out.current_cartesian_control_energy;
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
  out.worst_case_total_control_energy_ub = E_contact_max;
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

MonitorResult verifyReachablePlanJointSpace(
    const VerifiedPlan& plan,
    const Vector7d& current_q,
    const Vector7d& current_dq,
    const JointDynamicsProvider& dynamics,
    const SafetyMonitorConfig& config,
    std::vector<JointPredictionSample>* prediction_trace) {
  MonitorResult out;

  if (prediction_trace != nullptr) {
    prediction_trace->clear();
    prediction_trace->reserve(
        1 + plan.intended.size() + plan.failsafe.size());
  }

  auto symmetrize7 = [](const Matrix7d& matrix) {
    return 0.5 * (matrix + matrix.transpose());
  };
  auto jointKineticUpperBound = [&](const Vector7d& dq,
                                    const Matrix7d& inertia) {
    const Matrix7d metric = symmetrize7(inertia);
    const double nominal_metric_norm = std::sqrt(std::max(
        0.0, (dq.transpose() * metric * dq)(0, 0)));
    const Eigen::SelfAdjointEigenSolver<Matrix7d> eig(metric);
    const double max_eigenvalue =
        eig.info() == Eigen::Success
            ? std::max(0.0, eig.eigenvalues().maxCoeff())
            : std::max(0.0, metric.norm());
    const double uncertainty_metric_norm =
        std::sqrt(max_eigenvalue) *
        std::max(0.0, config.joint_velocity_error_bound);
    const double upper_metric_norm =
        nominal_metric_norm + uncertainty_metric_norm;
    return 0.5 * upper_metric_norm * upper_metric_norm;
  };
  auto taskInertia = [&](const Matrix7d& inertia,
                        const Matrix67d& jacobian,
                        Matrix6d* lambda,
                        Matrix7d* inertia_inv) {
    if (lambda == nullptr || inertia_inv == nullptr ||
        !inertia.allFinite() || !jacobian.allFinite()) {
      return false;
    }
    const Eigen::LDLT<Matrix7d> inertia_ldlt(symmetrize7(inertia));
    if (inertia_ldlt.info() != Eigen::Success) {
      return false;
    }
    *inertia_inv = inertia_ldlt.solve(Matrix7d::Identity());
    Matrix6d lambda_inv =
        jacobian * (*inertia_inv) * jacobian.transpose();
    lambda_inv = symmetrized(lambda_inv);
    lambda_inv.diagonal().array() += kLambdaReg;
    const Eigen::LDLT<Matrix6d> lambda_ldlt(lambda_inv);
    if (lambda_ldlt.info() != Eigen::Success) {
      return false;
    }
    *lambda = symmetrized(lambda_ldlt.solve(Matrix6d::Identity()));
    return lambda->allFinite() && inertia_inv->allFinite();
  };
  auto collisionPosition = [&](const JointDynamicsSample& state) {
    return state.control_position +
           state.control_orientation.normalized() *
               config.collision_center_offset;
  };
  auto poseError = [](const JointDynamicsSample& state,
                      const ImpedanceSample& desired) {
    Vector6d error = Vector6d::Zero();
    error.head<3>() = state.control_position - desired.p;
    error.tail<3>() =
        orientationError(state.control_orientation, desired.q);
    return error;
  };

  Vector7d q_pred = current_q;
  Vector7d dq_pred = current_dq;
  JointDynamicsSample state;
  if (!q_pred.allFinite() || !dq_pred.allFinite() ||
      !dynamics.evaluate(q_pred, dq_pred, &state) || !state.valid) {
    out.monitored_unsafe = true;
    out.joint_limit_unsafe = true;
    return out;
  }
  if (prediction_trace != nullptr) {
    prediction_trace->push_back(
        JointPredictionSample{plan.anchor.t, q_pred, dq_pred});
  }

  const double inflated_contact_radius_now =
      config.human_workspace.inflatedCollisionRadius(
          config.ee_collision_radius, 0.0);
  const Vector3d collision_position_now = collisionPosition(state);
  out.workspace_distance_now =
      config.human_workspace.signedDistanceToInflatedSphere(
          collision_position_now,
          inflated_contact_radius_now,
          config.wall_time_sec);
  out.workspace_distance_min = out.workspace_distance_now;

  Matrix6d current_lambda = Matrix6d::Zero();
  Matrix7d current_inertia_inv = Matrix7d::Zero();
  const bool current_lambda_valid = taskInertia(
      state.inertia, state.control_jacobian,
      &current_lambda, &current_inertia_inv);
  if (config.current_energy_reference_valid) {
    const Vector6d current_error =
        poseError(state, config.current_energy_reference);
    const Matrix6d current_stiffness = positiveSemidefinitePart(
        config.current_energy_reference.K);
    const Vector6d current_twist = state.control_jacobian * dq_pred;
    if (current_lambda_valid) {
      out.current_cartesian_kinetic_energy = quadraticEnergyUpperBound(
          current_twist, positiveSemidefinitePart(current_lambda), 0.0);
    }
    out.current_cartesian_potential_energy = quadraticEnergyUpperBound(
        current_error, current_stiffness, 0.0);
    out.current_cartesian_control_energy =
        out.current_cartesian_kinetic_energy +
        out.current_cartesian_potential_energy;
    out.current_cartesian_energy_valid = current_lambda_valid;
    out.current_joint_kinetic_energy =
        jointKineticUpperBound(dq_pred, state.inertia);
    out.current_total_control_energy =
        out.current_joint_kinetic_energy +
        out.current_cartesian_potential_energy;
    out.current_joint_energy_valid = true;
  }

  const JointDynamicsLimits limits = dynamics.limits();
  Matrix6d tracking_tube = Matrix6d::Zero();
  double total_contact_energy_max = 0.0;
  double terminal_total_energy = 0.0;
  bool terminal_sample_found = false;
  double t_prev = plan.anchor.t;
  Vector7d previous_torque_command = config.previous_torque_command;
  bool previous_torque_command_valid =
      config.previous_torque_command_valid;

  auto recordLimitViolation = [&](const Vector7d& q,
                                  const Vector7d& dq,
                                  const Vector7d& ddq,
                                  const Vector7d& tau) {
    for (int i = 0; i < 7; ++i) {
      const double q_violation = std::max(
          {limits.position_lower(i) - q(i),
           q(i) - limits.position_upper(i), 0.0});
      const double dq_violation =
          std::max(0.0, std::abs(dq(i)) - limits.velocity(i));
      const double ddq_violation =
          std::max(0.0, std::abs(ddq(i)) - limits.acceleration(i));
      const double tau_violation =
          std::max(0.0, std::abs(tau(i)) - limits.torque(i));
      // Position, velocity and actuator torque are hard predicted-state
      // constraints. Acceleration is retained as a diagnostic: Panda's
      // published acceleration values constrain motion generators, whereas
      // an effort-controlled impedance loop can legitimately exceed them for
      // a short disturbance response.
      const double largest_hard = std::max(
          {q_violation, dq_violation, tau_violation});
      if (largest_hard > 0.0) {
        out.joint_limit_unsafe = true;
        if (out.joint_limit_index < 0 ||
            largest_hard > std::max(
                {out.joint_position_violation,
                 out.joint_velocity_violation,
                 out.joint_torque_violation})) {
          out.joint_limit_index = i;
        }
        out.joint_position_violation =
            std::max(out.joint_position_violation, q_violation);
        out.joint_velocity_violation =
            std::max(out.joint_velocity_violation, dq_violation);
        out.joint_acceleration_violation =
            std::max(out.joint_acceleration_violation, ddq_violation);
        out.joint_torque_violation =
            std::max(out.joint_torque_violation, tau_violation);
      }
    }
  };

  auto evalSample = [&](const ImpedanceSample& desired, double dt) {
    Matrix6d lambda = Matrix6d::Zero();
    Matrix7d inertia_inv = Matrix7d::Zero();
    const bool lambda_valid = taskInertia(
        state.inertia, state.control_jacobian, &lambda, &inertia_inv);
    if (!lambda_valid) {
      out.joint_limit_unsafe = true;
      return false;
    }

    const Vector6d error = poseError(state, desired);
    const Vector6d twist = state.control_jacobian * dq_pred;
    Vector6d desired_twist = Vector6d::Zero();
    desired_twist.head<3>() = desired.dp;
    desired_twist.tail<3>() = desired.w;
    Vector6d desired_acceleration = Vector6d::Zero();
    desired_acceleration.head<3>() = desired.ddp;
    desired_acceleration.tail<3>() = desired.dw;

    Vector6d wrench =
        -desired.K * error - desired.D * (twist - desired_twist);
    if (config.use_dynamic_consistent_impedance) {
      wrench += lambda *
                (desired_acceleration - state.control_jdot_dq);
    }
    const Vector7d tau_task = state.control_jacobian.transpose() * wrench;

    Vector7d tau_nullspace = Vector7d::Zero();
    const Vector3d collision_start = collisionPosition(state);
    const double start_distance =
        config.human_workspace.signedDistanceToInflatedSphere(
            collision_start,
            config.human_workspace.inflatedCollisionRadius(
                config.ee_collision_radius, 0.0),
            config.wall_time_sec + t_prev);
    const bool nullspace_enabled_for_sample =
        config.nullspace_stiffness > 0.0 && start_distance > 0.0 &&
        !(desired.failsafe && config.disable_nullspace_in_failsafe);
    if (nullspace_enabled_for_sample) {
      const Vector7d tau_nullspace_raw =
          config.nullspace_stiffness *
              (config.nullspace_reference - q_pred) -
          2.0 * std::sqrt(config.nullspace_stiffness) * dq_pred;
      const Eigen::Matrix<double, 7, 6> jbar =
          inertia_inv * state.control_jacobian.transpose() * lambda;
      const Matrix7d nullspace_projector_transpose =
          Matrix7d::Identity() -
          state.control_jacobian.transpose() * jbar.transpose();
      tau_nullspace =
          nullspace_projector_transpose * tau_nullspace_raw;
    }

    const double h = std::max(dt, kMinDt);
    const Vector7d desired_torque_command =
        tau_task + tau_nullspace + state.coriolis;
    Vector7d torque_command = desired_torque_command;
    if (previous_torque_command_valid) {
      const double max_delta =
          std::max(0.0, config.torque_rate_limit) * h;
      for (int i = 0; i < 7; ++i) {
        torque_command(i) = previous_torque_command(i) + std::clamp(
            desired_torque_command(i) - previous_torque_command(i),
            -max_delta,
            max_delta);
      }
    }
    const Vector7d generalized_control =
        torque_command - state.coriolis;
    const Vector7d ddq = inertia_inv * generalized_control;
    const Vector7d dq_next = dq_pred + ddq * h;
    const Vector7d q_next = q_pred + 0.5 * (dq_pred + dq_next) * h;

    recordLimitViolation(q_next, dq_next, ddq, torque_command);
    if (prediction_trace != nullptr &&
        q_next.allFinite() && dq_next.allFinite()) {
      prediction_trace->push_back(
          JointPredictionSample{desired.t, q_next, dq_next});
    }
    if (!q_next.allFinite() || !dq_next.allFinite()) {
      out.joint_limit_unsafe = true;
      return false;
    }

    JointDynamicsSample next_state;
    if (!dynamics.evaluate(q_next, dq_next, &next_state) ||
        !next_state.valid) {
      out.joint_limit_unsafe = true;
      return false;
    }

    const Matrix3d task_inertia_inv =
        (state.control_jacobian * inertia_inv *
         state.control_jacobian.transpose()).topLeftCorner<3, 3>();
    const Matrix6d tracking_tube_next = propagateTrackingTube(
        tracking_tube,
        task_inertia_inv,
        positiveSemidefinitePart(
            Matrix3d(desired.K.topLeftCorner<3, 3>())),
        positiveSemidefinitePart(
            Matrix3d(desired.D.topLeftCorner<3, 3>())),
        h,
        config.tracking_acc_error_bound);
    const double rho_p = std::max(
        maxBlockRadius(tracking_tube, 0),
        maxBlockRadius(tracking_tube_next, 0));
    const Vector3d collision_end = collisionPosition(next_state);
    const double segment_distance =
        config.human_workspace.signedDistanceSegmentToInflatedSphere(
            collision_start,
            collision_end,
            config.human_workspace.inflatedCollisionRadius(
                config.ee_collision_radius, rho_p),
            config.wall_time_sec + t_prev,
            config.wall_time_sec + desired.t);
    out.workspace_distance_min =
        std::min(out.workspace_distance_min, segment_distance);

    const Vector6d next_error = poseError(next_state, desired);
    Matrix6d next_lambda = Matrix6d::Zero();
    Matrix7d next_inertia_inv = Matrix7d::Zero();
    const bool next_lambda_valid = taskInertia(
        next_state.inertia, next_state.control_jacobian,
        &next_lambda, &next_inertia_inv);
    const Vector6d next_twist =
        next_state.control_jacobian * dq_next;
    const double cartesian_kinetic = next_lambda_valid
        ? quadraticEnergyUpperBound(
              next_twist,
              positiveSemidefinitePart(next_lambda),
              maxBlockRadius(tracking_tube_next, 3))
        : 0.0;
    const double joint_kinetic =
        jointKineticUpperBound(dq_next, next_state.inertia);
    const double potential = quadraticEnergyUpperBound(
        next_error,
        positiveSemidefinitePart(desired.K),
        maxBlockRadius(tracking_tube_next, 0));
    const double total_energy = joint_kinetic + potential;

    if (segment_distance <= 0.0 &&
        total_energy > total_contact_energy_max) {
      total_contact_energy_max = total_energy;
      out.worst_case_contact_time = desired.t;
      out.worst_case_workspace_distance_at_candidate = segment_distance;
      out.worst_case_cartesian_kinetic_energy_ub = cartesian_kinetic;
      out.worst_case_joint_kinetic_energy_ub = joint_kinetic;
      out.worst_case_cartesian_potential_energy_ub = potential;
      // Kept for CSV/API compatibility; this now denotes the kinetic metric
      // that actually gates the paper energy budget plus Cartesian potential.
      out.worst_case_cartesian_control_energy_ub = total_energy;
      out.worst_case_total_control_energy_ub = total_energy;
    }
    if (desired.failsafe) {
      terminal_total_energy = total_energy;
      terminal_sample_found = true;
    }

    q_pred = q_next;
    dq_pred = dq_next;
    previous_torque_command = torque_command;
    previous_torque_command_valid = true;
    state = next_state;
    tracking_tube = tracking_tube_next;
    out.worst_case_pos_error_radius = std::max(
        out.worst_case_pos_error_radius,
        maxBlockRadius(tracking_tube, 0));
    out.worst_case_vel_error_radius = std::max(
        out.worst_case_vel_error_radius,
        maxBlockRadius(tracking_tube, 3));
    return true;
  };

  auto interpolateSample = [](const ImpedanceSample& start,
                              const ImpedanceSample& end,
                              double alpha) {
    const double u = std::clamp(alpha, 0.0, 1.0);
    ImpedanceSample sample;
    sample.t = (1.0 - u) * start.t + u * end.t;
    sample.nominal_path_time =
        (1.0 - u) * start.nominal_path_time +
        u * end.nominal_path_time;
    sample.nominal_path_time_valid =
        start.nominal_path_time_valid && end.nominal_path_time_valid;
    sample.p = (1.0 - u) * start.p + u * end.p;
    sample.dp = (1.0 - u) * start.dp + u * end.dp;
    sample.ddp = (1.0 - u) * start.ddp + u * end.ddp;
    Quaterniond q_start = start.q.normalized();
    Quaterniond q_end = end.q.normalized();
    if (q_start.coeffs().dot(q_end.coeffs()) < 0.0) {
      q_end.coeffs() *= -1.0;
    }
    sample.q = q_start.slerp(u, q_end).normalized();
    sample.w = (1.0 - u) * start.w + u * end.w;
    sample.dw = (1.0 - u) * start.dw + u * end.dw;
    sample.K = (1.0 - u) * start.K + u * end.K;
    sample.D = (1.0 - u) * start.D + u * end.D;
    sample.failsafe = end.failsafe;
    return sample;
  };

  ImpedanceSample previous_desired = plan.anchor;
  auto evaluateStage = [&](const std::vector<ImpedanceSample>& stage) {
    for (const auto& desired : stage) {
      const double segment_dt =
          std::max(desired.t - previous_desired.t, kMinDt);
      const double max_step =
          std::max(config.joint_rollout_max_dt, kMinDt);
      const int substeps = std::max(
          1, static_cast<int>(std::ceil(segment_dt / max_step)));
      for (int substep = 1; substep <= substeps; ++substep) {
        const double alpha =
            static_cast<double>(substep) /
            static_cast<double>(substeps);
        const ImpedanceSample interpolated =
            interpolateSample(previous_desired, desired, alpha);
        const double dt = std::max(interpolated.t - t_prev, kMinDt);
        if (!evalSample(interpolated, dt)) {
          return false;
        }
        t_prev = interpolated.t;
      }
      previous_desired = desired;
    }
    return true;
  };

  const bool intended_complete = evaluateStage(plan.intended);
  if (intended_complete && !out.joint_limit_unsafe) {
    evaluateStage(plan.failsafe);
  }

  out.monitored_contact_possible = out.workspace_distance_min <= 0.0;
  out.workspace_distance_margin = out.workspace_distance_min;
  out.terminal_energy_ub =
      terminal_sample_found ? terminal_total_energy : 0.0;
  out.contact_relevant_for_energy = out.workspace_distance_now <= 0.0;

  const double energy_budget_eff = std::max(
      0.0,
      config.energy_budget_joule - config.energy_budget_margin_joule);
  const bool predicted_contact_requires_verification =
      out.monitored_contact_possible && !out.contact_relevant_for_energy;
  const bool current_collision_energy_unsafe =
      out.contact_relevant_for_energy &&
      out.current_joint_energy_valid &&
      out.current_total_control_energy > energy_budget_eff;
  const bool predicted_contact_energy_unsafe =
      predicted_contact_requires_verification &&
      total_contact_energy_max > energy_budget_eff;

  out.predicted_trigger =
      predicted_contact_energy_unsafe || out.joint_limit_unsafe;
  const double monitored_contact_energy_ub = std::max(
      total_contact_energy_max,
      out.contact_relevant_for_energy && out.current_joint_energy_valid
          ? out.current_total_control_energy
          : 0.0);
  out.h_monitored_energy =
      energy_budget_eff - monitored_contact_energy_ub;
  out.h_terminal_energy = energy_budget_eff - out.terminal_energy_ub;
  out.collision_energy_unsafe =
      current_collision_energy_unsafe || predicted_contact_energy_unsafe;
  out.monitored_unsafe =
      out.predicted_trigger ||
      (out.contact_relevant_for_energy && current_collision_energy_unsafe);
  return out;
}

}  // namespace cps_safety_monitor
