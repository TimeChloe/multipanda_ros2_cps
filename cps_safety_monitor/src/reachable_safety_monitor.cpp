#include "cps_safety_monitor/reachable_safety_monitor.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>

namespace cps_safety_monitor {

double energyBudgetStiffnessScale(double kinetic_energy,
                                  double cartesian_potential_energy,
                                  double nullspace_potential_energy,
                                  double energy_budget) {
  if (!std::isfinite(kinetic_energy) ||
      !std::isfinite(cartesian_potential_energy) ||
      !std::isfinite(nullspace_potential_energy) ||
      !std::isfinite(energy_budget)) {
    return 0.0;
  }

  const double kinetic = std::max(0.0, kinetic_energy);
  const double cartesian_potential =
      std::max(0.0, cartesian_potential_energy);
  const double nullspace_potential =
      std::max(0.0, nullspace_potential_energy);
  const double budget = std::max(0.0, energy_budget);
  const double total_potential =
      cartesian_potential + nullspace_potential;

  if (kinetic > budget) {
    return 0.0;
  }
  if (kinetic + total_potential <= budget) {
    return 1.0;
  }
  if (total_potential <= 0.0) {
    return 0.0;
  }
  return std::clamp((budget - kinetic) / total_potential, 0.0, 1.0);
}

OverbudgetJointStabilizationTerms updateOverbudgetJointStabilization(
    const Vector7d& q,
    double kinetic_energy,
    double energy_budget,
    double joint_stiffness,
    double scale_omega,
    bool enabled,
    OverbudgetJointStabilizationState* state) {
  OverbudgetJointStabilizationTerms terms;
  if (state == nullptr) {
    return terms;
  }

  const bool inputs_valid =
      q.allFinite() && std::isfinite(kinetic_energy) &&
      std::isfinite(energy_budget) && std::isfinite(joint_stiffness) &&
      std::isfinite(scale_omega);
  if (!enabled || !inputs_valid || joint_stiffness <= 0.0 ||
      scale_omega <= 0.0) {
    *state = OverbudgetJointStabilizationState{};
    return terms;
  }

  const double kinetic = std::max(0.0, kinetic_energy);
  const double budget = std::max(0.0, energy_budget);
  if (kinetic <= budget) {
    *state = OverbudgetJointStabilizationState{};
    return terms;
  }

  if (!state->active || !state->reference.allFinite()) {
    state->active = true;
    state->reference = q;
  }

  const double stiffness = std::max(0.0, joint_stiffness);
  const Vector7d displacement = state->reference - q;
  terms.active = true;
  terms.potential_energy =
      0.5 * stiffness * displacement.squaredNorm();
  terms.scale_rho =
      std::max(0.0, scale_omega) * kinetic /
      std::max(1.0e-12, budget + terms.potential_energy);
  terms.torque = terms.scale_rho * stiffness * displacement;
  return terms;
}

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

double quadraticEnergy(const Vector6d& state,
                       const Matrix6d& metric_psd) {
  return std::max(
      0.0, 0.5 * (state.transpose() * metric_psd * state)(0, 0));
}

double nullspacePotentialEnergy(const Vector7d& q,
                                const Vector7d& reference,
                                double stiffness) {
  if (!q.allFinite() || !reference.allFinite() ||
      !std::isfinite(stiffness)) {
    return 0.0;
  }
  const Vector7d error = reference - q;
  return 0.5 * std::max(0.0, stiffness) * error.squaredNorm();
}

double orientationInducedPositionError(double offset_norm,
                                       double orientation_error_bound) {
  constexpr double kPi = 3.14159265358979323846;
  const double angle = std::clamp(
      std::max(0.0, orientation_error_bound), 0.0, kPi);
  return 2.0 * std::max(0.0, offset_norm) * std::sin(0.5 * angle);
}

double trackingGeometryInflation(double position_error_bound,
                                 double orientation_error_bound,
                                 double collision_center_offset_norm) {
  return std::max(0.0, position_error_bound) +
         orientationInducedPositionError(
             collision_center_offset_norm,
             orientation_error_bound);
}

struct EnergyUpperBound {
  double kinetic{0.0};
  double cartesian_potential{0.0};
  double nullspace_potential{0.0};

  double total() const {
    return kinetic + cartesian_potential + nullspace_potential;
  }
};

EnergyUpperBound addOneSidedEnergyErrorBounds(
    double nominal_kinetic,
    double nominal_cartesian_potential,
    double nominal_nullspace_potential,
    double kinetic_error_bound,
    double cartesian_potential_error_bound,
    double nullspace_potential_error_bound) {
  // Tracking-pose tubes intentionally do not enter this function. They are
  // geometry bounds, whereas beta_K and beta_V directly bound the residuals
  // of their corresponding predicted energy terms.
  return EnergyUpperBound{
      std::max(0.0, nominal_kinetic) +
          std::max(0.0, kinetic_error_bound),
      std::max(0.0, nominal_cartesian_potential) +
          std::max(0.0, cartesian_potential_error_bound),
      std::max(0.0, nominal_nullspace_potential) +
          std::max(0.0, nullspace_potential_error_bound)};
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

  // This fallback has no joint trajectory and therefore cannot evaluate the
  // complete T, U_q, contact-gated Eq. (14) scaling, or Eqs. (16)-(17) joint
  // torque.
  // Never authorize a plan by silently omitting any configured joint-space
  // energy behavior; use verifyReachablePlanJointSpace instead.
  if (config.nullspace_stiffness > 0.0 ||
      config.current_nullspace_stiffness > 0.0 ||
      config.enable_runtime_energy_scaling ||
      config.enable_overbudget_joint_stabilization) {
    out.joint_limit_unsafe = true;
    out.predicted_trigger = true;
    out.monitored_unsafe = true;
    return out;
  }

  Vector3d x_pred = current_position;
  Vector3d v_pred = ee_twist.head<3>();
  Quaterniond q_pred = current_orientation.normalized();
  Vector3d w_pred = ee_twist.tail<3>();

  const Eigen::LDLT<Matrix7d> inertia_ldlt(inertia);
  Matrix7d inertia_inv;
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
  const double kinetic_energy_error_bound =
      std::max(0.0, config.kinetic_energy_error_bound_joule);
  const double potential_energy_error_bound =
      std::max(0.0, config.potential_energy_error_bound_joule);
  const double energy_budget_eff =
      std::max(0.0, config.energy_budget_joule);
  Matrix6d tracking_tube = Matrix6d::Zero();

  out.worst_case_pos_error_radius = 0.0;
  out.worst_case_orientation_error_radius = 0.0;
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
    out.current_cartesian_kinetic_energy =
        quadraticEnergy(ee_twist, cartesian_inertia);
    out.current_cartesian_potential_energy =
        quadraticEnergy(current_error, current_stiffness);
    out.current_cartesian_control_energy =
        out.current_cartesian_kinetic_energy +
        out.current_cartesian_potential_energy;
    out.current_total_control_energy =
        out.current_cartesian_control_energy;
    out.current_cartesian_energy_valid = true;
  }

  if (config.assume_human_workspace_clear) {
    out.workspace_distance_now = std::numeric_limits<double>::infinity();
  } else {
    const double inflated_contact_radius_now =
        config.human_workspace.inflatedCollisionRadius(
            config.ee_collision_radius,
            0.0);
    out.workspace_distance_now =
        config.human_workspace.signedDistanceToInflatedSphere(
            current_position,
            inflated_contact_radius_now,
            config.wall_time_sec);
  }

  out.workspace_distance_min = out.workspace_distance_now;

  Matrix6d K_exec = config.K_runtime;
  Matrix6d D_exec = config.D_runtime;

  double E_contact_max = 0.0;

  double terminal_T_ub = 0.0;
  double terminal_V_ub = 0.0;
  bool terminal_sample_found = false;
  double t_prev = plan.anchor.t;

  Vector6d previous_edge_twist = ee_twist;
  Vector6d previous_edge_error = Vector6d::Zero();
  previous_edge_error.head<3>() = current_position - plan.anchor.p;
  previous_edge_error.tail<3>() =
      orientationError(current_orientation, plan.anchor.q);
  double previous_edge_T_ub =
      quadraticEnergy(previous_edge_twist, cartesian_inertia);
  double previous_edge_V_ub = quadraticEnergy(
      previous_edge_error, positiveSemidefinitePart(plan.anchor.K));
  double previous_edge_energy_ub =
      previous_edge_T_ub + previous_edge_V_ub;
  double previous_edge_time = plan.anchor.t;
  double previous_position_error_radius = 0.0;
  double previous_orientation_error_radius = 0.0;
  int monitored_interval_index = 0;

  auto eval_sample = [&](const ImpedanceSample& s, double dtp) {
    const int interval_index = monitored_interval_index++;
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
    const double next_position_error_radius =
        maxBlockRadius(tracking_tube_next, 0);
    const double next_orientation_error_radius = 0.0;
    const double segment_position_error_radius =
        std::max(previous_position_error_radius,
                 next_position_error_radius);
    const double segment_orientation_error_radius =
        std::max(previous_orientation_error_radius,
                 next_orientation_error_radius);
    const double rho_p_segment = trackingGeometryInflation(
        segment_position_error_radius,
        segment_orientation_error_radius,
        config.collision_center_offset.norm());
    double d_segment = std::numeric_limits<double>::infinity();
    if (!config.assume_human_workspace_clear) {
      const double inflated_contact_radius_segment =
          config.human_workspace.inflatedCollisionRadius(
              config.ee_collision_radius,
              rho_p_segment);
      d_segment =
          config.human_workspace.signedDistanceSegmentToInflatedSphere(
              x_pred,
              x_next,
              inflated_contact_radius_segment,
              segment_start_time_sec,
              segment_end_time_sec);
    }

    out.workspace_distance_min = std::min(out.workspace_distance_min, d_segment);

    Vector6d next_twist = Vector6d::Zero();
    next_twist.head<3>() = v_next;
    next_twist.tail<3>() = w_next;
    Vector6d next_pose_error = Vector6d::Zero();
    next_pose_error.head<3>() = x_next - s.p;
    next_pose_error.tail<3>() = orientationError(q_next, s.q);
    // Pose tracking bounds enlarge collision geometry only. Energy-model
    // mismatch is represented directly by the independent one-sided K/V
    // error bounds, so it is not counted a second time through the pose tube.
    const EnergyUpperBound energy_ub = addOneSidedEnergyErrorBounds(
        quadraticEnergy(next_twist, cartesian_inertia),
        quadraticEnergy(next_pose_error, K_cartesian),
        0.0,
        kinetic_energy_error_bound,
        potential_energy_error_bound,
        0.0);
    const double cartesian_T_ub = energy_ub.kinetic;
    const double cartesian_V_ub = energy_ub.cartesian_potential;

    const bool contact_possible_step = d_segment <= 0.0;
    if (contact_possible_step && out.first_contact_interval_index < 0) {
      out.first_contact_interval_index = interval_index;
    }
    // Candidate acceptance is based on the maximum complete Cartesian control
    // energy over every monitored sample that may intersect the workspace.
    // The normal-projected energy remains available only as a collision
    // diagnostic and must not authorize a 6D-energetic trajectory.
    const double next_edge_energy_ub = energy_ub.total();
    const bool previous_edge_is_worst =
        previous_edge_energy_ub >= next_edge_energy_ub;
    const double interval_energy_ub =
        std::max(previous_edge_energy_ub, next_edge_energy_ub);
    if (contact_possible_step && interval_energy_ub > energy_budget_eff &&
        out.collision_interval_index < 0) {
      out.collision_interval_index = interval_index;
    }
    if (contact_possible_step && interval_energy_ub > energy_budget_eff &&
        out.first_energy_unsafe_contact_interval_index < 0) {
      out.first_energy_unsafe_contact_interval_index = interval_index;
    }
    if (contact_possible_step && interval_energy_ub > E_contact_max) {
      E_contact_max = interval_energy_ub;

      out.worst_case_contact_time =
          previous_edge_is_worst ? previous_edge_time : s.t;
      out.worst_case_workspace_distance_at_candidate = d_segment;
      out.worst_case_cartesian_kinetic_energy_ub =
          previous_edge_is_worst ? previous_edge_T_ub : cartesian_T_ub;
      out.worst_case_cartesian_potential_energy_ub =
          previous_edge_is_worst ? previous_edge_V_ub : cartesian_V_ub;
      out.worst_case_cartesian_control_energy_ub = interval_energy_ub;
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
    previous_edge_T_ub = cartesian_T_ub;
    previous_edge_V_ub = cartesian_V_ub;
    previous_edge_energy_ub = next_edge_energy_ub;
    previous_edge_time = s.t;
    previous_position_error_radius = next_position_error_radius;
    previous_orientation_error_radius = next_orientation_error_radius;
    out.worst_case_pos_error_radius =
        std::max(out.worst_case_pos_error_radius,
                 next_position_error_radius);
    out.worst_case_orientation_error_radius =
        std::max(out.worst_case_orientation_error_radius,
                 next_orientation_error_radius);
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

  // Current overlap is both a contact-risk diagnostic and the runtime
  // activation gate for Eq. (14) stiffness scaling.
  out.contact_relevant_for_energy = out.workspace_distance_now <= 0.0;
  const bool predicted_contact_requires_verification =
      out.monitored_contact_possible;

  const bool current_collision_energy_unsafe =
      out.contact_relevant_for_energy &&
      out.current_cartesian_energy_valid &&
      out.current_cartesian_control_energy > energy_budget_eff;
  const bool predicted_contact_energy_unsafe =
      predicted_contact_requires_verification &&
      out.worst_case_cartesian_control_energy_ub > energy_budget_eff;

  out.predicted_trigger = predicted_contact_energy_unsafe;

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
  auto jointKineticEnergy = [&](const Vector7d& dq,
                                const Matrix7d& inertia) {
    const Matrix7d metric = symmetrize7(inertia);
    return std::max(
        0.0, 0.5 * (dq.transpose() * metric * dq)(0, 0));
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
  auto collisionPosition = [&](const JointDynamicsSample& state) -> Vector3d {
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
  if (config.current_joint_dynamics_valid) {
    state = config.current_joint_dynamics;
  } else if (!dynamics.evaluate(q_pred, dq_pred, &state)) {
    state.valid = false;
  }
  if (!q_pred.allFinite() || !dq_pred.allFinite() || !state.valid) {
    out.monitored_unsafe = true;
    out.joint_limit_unsafe = true;
    return out;
  }

  if (config.enable_inertia_model_comparison &&
      config.current_joint_dynamics_valid) {
    JointDynamicsSample prediction_model_state;
    if (dynamics.evaluate(
            current_q, current_dq, &prediction_model_state) &&
        prediction_model_state.valid &&
        prediction_model_state.inertia.allFinite() &&
        config.current_joint_dynamics.inertia.allFinite()) {
      const Matrix7d runtime_inertia =
          symmetrize7(config.current_joint_dynamics.inertia);
      const Matrix7d prediction_inertia =
          symmetrize7(prediction_model_state.inertia);
      const Matrix7d inertia_difference =
          runtime_inertia - prediction_inertia;

      out.runtime_model_joint_kinetic_energy =
          jointKineticEnergy(current_dq, runtime_inertia);
      out.prediction_model_joint_kinetic_energy =
          jointKineticEnergy(current_dq, prediction_inertia);
      out.inertia_model_kinetic_energy_error =
          out.runtime_model_joint_kinetic_energy -
          out.prediction_model_joint_kinetic_energy;
      out.inertia_model_difference_frobenius_norm =
          inertia_difference.norm();
      out.inertia_model_difference_relative_frobenius_norm =
          inertia_difference.norm() /
          std::max(runtime_inertia.norm(), kSmallPositive);
      Eigen::Index max_row = -1;
      Eigen::Index max_col = -1;
      out.inertia_model_difference_max_abs =
          inertia_difference.cwiseAbs().maxCoeff(&max_row, &max_col);
      out.inertia_model_difference_max_abs_row =
          static_cast<int>(max_row);
      out.inertia_model_difference_max_abs_col =
          static_cast<int>(max_col);
      out.inertia_model_comparison_valid =
          std::isfinite(out.runtime_model_joint_kinetic_energy) &&
          std::isfinite(out.prediction_model_joint_kinetic_energy) &&
          std::isfinite(out.inertia_model_kinetic_energy_error) &&
          std::isfinite(out.inertia_model_difference_frobenius_norm) &&
          std::isfinite(
              out.inertia_model_difference_relative_frobenius_norm) &&
          std::isfinite(out.inertia_model_difference_max_abs);

      // The generalized eigenvalues of (M_runtime, M_prediction) bound the
      // kinetic-energy ratio for every nonzero velocity direction:
      // lambda_min <= K_runtime/K_prediction <= lambda_max.
      const Eigen::SelfAdjointEigenSolver<Matrix7d>
          prediction_inertia_eigenvalues(prediction_inertia);
      if (prediction_inertia_eigenvalues.info() == Eigen::Success &&
          prediction_inertia_eigenvalues.eigenvalues().minCoeff() >
              kSmallPositive) {
        const Eigen::GeneralizedSelfAdjointEigenSolver<Matrix7d>
            energy_ratio_solver(runtime_inertia, prediction_inertia);
        if (energy_ratio_solver.info() == Eigen::Success &&
            energy_ratio_solver.eigenvalues().allFinite()) {
          out.inertia_model_min_energy_ratio =
              energy_ratio_solver.eigenvalues().minCoeff();
          out.inertia_model_max_energy_ratio =
              energy_ratio_solver.eigenvalues().maxCoeff();
          out.inertia_model_energy_ratio_valid =
              std::isfinite(out.inertia_model_min_energy_ratio) &&
              std::isfinite(out.inertia_model_max_energy_ratio);
        }
      }
    }
  }

  const bool use_sara_robot_reach =
      static_cast<bool>(config.robot_reachability_provider);
  // A zero-duration occupancy is independent of alpha. Supplying an explicit
  // zero vector keeps the provider API uniform without inventing a fixed
  // acceleration bound before the complete trajectory has been rolled out.
  const std::vector<double> zero_robot_alpha(7, 0.0);
  out.robot_secure_radius = use_sara_robot_reach
      ? config.robot_reachability_provider->secureRadius()
      : 0.0;
  const Vector3d collision_position_now = collisionPosition(state);
  const auto human_reach_now =
      config.human_workspace.handReachableSetAtTime(config.wall_time_sec);
  if (config.assume_human_workspace_clear) {
    out.workspace_distance_now = std::numeric_limits<double>::infinity();
  } else if (use_sara_robot_reach) {
    std::vector<RobotReachCapsule> current_capsules;
    if (!config.robot_reachability_provider->reachInterval(
            q_pred, q_pred, 0.0, zero_robot_alpha, &current_capsules)) {
      out.joint_limit_unsafe = true;
      out.predicted_trigger = true;
      out.monitored_unsafe = true;
      return out;
    }
    out.workspace_distance_now =
        config.robot_reachability_provider->minimumSignedDistance(
            current_capsules,
            human_reach_now.center,
            human_reach_now.center,
            human_reach_now.radius,
            &out.current_robot_link_index);
  } else {
    const double inflated_contact_radius_now =
        config.human_workspace.inflatedCollisionRadius(
            config.ee_collision_radius, 0.0);
    out.workspace_distance_now =
        config.human_workspace.signedDistanceToInflatedSphere(
            collision_position_now,
            inflated_contact_radius_now,
            config.wall_time_sec);
  }
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
      out.current_cartesian_kinetic_energy = quadraticEnergy(
          current_twist, positiveSemidefinitePart(current_lambda));
    }
    out.current_cartesian_potential_energy =
        quadraticEnergy(current_error, current_stiffness);
    out.current_cartesian_control_energy =
        out.current_cartesian_kinetic_energy +
        out.current_cartesian_potential_energy;
    out.current_cartesian_energy_valid = current_lambda_valid;
    out.current_joint_kinetic_energy =
        jointKineticEnergy(dq_pred, state.inertia);
    out.current_nullspace_potential_energy = nullspacePotentialEnergy(
        q_pred,
        config.nullspace_reference,
        config.current_nullspace_stiffness);
    out.current_total_control_energy =
        out.current_joint_kinetic_energy +
        out.current_cartesian_potential_energy +
        out.current_nullspace_potential_energy;
    out.current_joint_energy_valid = true;
  }

  const JointDynamicsLimits limits = dynamics.limits();
  Matrix6d tracking_tube = Matrix6d::Zero();
  const double kinetic_energy_error_bound =
      std::max(0.0, config.kinetic_energy_error_bound_joule);
  const double potential_energy_error_bound =
      std::max(0.0, config.potential_energy_error_bound_joule);
  const double nullspace_potential_energy_error_bound =
      std::max(
          0.0,
          config.nullspace_potential_energy_error_bound_joule);
  const double energy_budget_eff =
      std::max(0.0, config.energy_budget_joule);
  double total_contact_energy_max = 0.0;
  double terminal_total_energy = 0.0;
  bool terminal_sample_found = false;
  double t_prev = plan.anchor.t;
  Vector7d previous_torque_command = config.previous_torque_command;
  bool previous_torque_command_valid =
      config.previous_torque_command_valid;

  const Vector6d previous_edge_twist_initial =
      state.control_jacobian * dq_pred;
  const Vector6d previous_edge_error_initial =
      poseError(state, plan.anchor);
  double previous_edge_cartesian_kinetic = current_lambda_valid
      ? quadraticEnergy(
            previous_edge_twist_initial,
            positiveSemidefinitePart(current_lambda))
      : 0.0;
  double previous_edge_joint_kinetic =
      jointKineticEnergy(dq_pred, state.inertia);
  double previous_edge_potential = quadraticEnergy(
      previous_edge_error_initial,
      positiveSemidefinitePart(plan.anchor.K));
  double previous_edge_nullspace_potential = nullspacePotentialEnergy(
      q_pred,
      config.nullspace_reference,
      config.current_nullspace_stiffness);
  double previous_edge_total_energy =
      previous_edge_joint_kinetic + previous_edge_potential +
      previous_edge_nullspace_potential;
  double previous_edge_time = plan.anchor.t;
  double previous_position_error_radius = 0.0;
  double previous_orientation_error_radius = 0.0;
  OverbudgetJointStabilizationState overbudget_joint_state =
      config.overbudget_joint_state;
  OverbudgetJointStabilizationTerms overbudget_joint_terms =
      updateOverbudgetJointStabilization(
          q_pred,
          previous_edge_joint_kinetic,
          energy_budget_eff,
          config.overbudget_joint_stiffness,
          config.overbudget_joint_scale_omega,
          config.enable_overbudget_joint_stabilization &&
              out.workspace_distance_now <= 0.0,
          &overbudget_joint_state);

  // SaRA PFL computes one alpha_i vector from the complete monitored
  // trajectory and then reuses it for every interval. Therefore the rollout
  // trace must exist even when the caller does not request it as an output.
  std::vector<JointPredictionSample> rollout_trace;
  rollout_trace.reserve(1 + plan.intended.size() + plan.failsafe.size());
  JointPredictionSample initial_prediction;
  initial_prediction.t = plan.anchor.t;
  initial_prediction.q = q_pred;
  initial_prediction.dq = dq_pred;
  initial_prediction.energy_valid = true;
  initial_prediction.joint_kinetic_energy = previous_edge_joint_kinetic;
  initial_prediction.cartesian_potential_energy = previous_edge_potential;
  initial_prediction.nullspace_potential_energy =
      previous_edge_nullspace_potential;
  initial_prediction.nullspace_potential_energy_active =
      config.current_nullspace_stiffness > 0.0;
  initial_prediction.energy_scaling_active =
      config.enable_runtime_energy_scaling &&
      out.workspace_distance_now <= 0.0;
  initial_prediction.applied_nullspace_stiffness =
      config.current_nullspace_stiffness;
  initial_prediction.overbudget_joint_stabilization_active =
      overbudget_joint_terms.active;
  initial_prediction.overbudget_joint_potential_energy =
      overbudget_joint_terms.potential_energy;
  initial_prediction.overbudget_joint_scale_rho =
      overbudget_joint_terms.scale_rho;
  initial_prediction.overbudget_joint_torque_norm =
      overbudget_joint_terms.torque.norm();
  rollout_trace.push_back(initial_prediction);

  struct PredictedReachInterval {
    int index{-1};
    double start_time{0.0};
    double end_time{0.0};
    Vector7d start_q{Vector7d::Zero()};
    Vector7d end_q{Vector7d::Zero()};
    double start_cartesian_kinetic{0.0};
    double start_joint_kinetic{0.0};
    double start_potential{0.0};
    double start_nullspace_potential{0.0};
    double start_total_energy{0.0};
    double end_cartesian_kinetic{0.0};
    double end_joint_kinetic{0.0};
    double end_potential{0.0};
    double end_nullspace_potential{0.0};
    double end_total_energy{0.0};
  };
  std::vector<PredictedReachInterval> predicted_reach_intervals;
  predicted_reach_intervals.reserve(
      plan.intended.size() + plan.failsafe.size());

  int monitored_interval_index = 0;
  auto markIntervalUnsafe = [&](int interval_index) {
    if (out.collision_interval_index < 0) {
      out.collision_interval_index = interval_index;
    }
  };

  auto recordLimitViolation = [&](const Vector7d& q,
                                  const Vector7d& dq,
                                  const Vector7d& ddq,
                                  const Vector7d& tau,
                                  int interval_index) {
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
        markIntervalUnsafe(interval_index);
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
    const int interval_index = monitored_interval_index++;
    Matrix6d lambda = Matrix6d::Zero();
    Matrix7d inertia_inv = Matrix7d::Zero();
    const bool lambda_valid = taskInertia(
        state.inertia, state.control_jacobian, &lambda, &inertia_inv);
    if (!lambda_valid) {
      markIntervalUnsafe(interval_index);
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

    // Reproduce the runtime's current-state overlap gate before calculating
    // rollout torque. Normal Cartesian and nullspace gains remain unchanged
    // outside the human collision area; Eq. (14) acts on both inside it.
    const Vector3d collision_start = collisionPosition(state);
    double start_distance = std::numeric_limits<double>::infinity();
    if (!config.assume_human_workspace_clear && use_sara_robot_reach) {
      std::vector<RobotReachCapsule> start_capsules;
      if (!config.robot_reachability_provider->reachInterval(
              q_pred,
              q_pred,
              0.0,
              zero_robot_alpha,
              &start_capsules)) {
        markIntervalUnsafe(interval_index);
        out.joint_limit_unsafe = true;
        return false;
      }
      const auto human_reach =
          config.human_workspace.handReachableSetAtTime(
              config.wall_time_sec + t_prev);
      start_distance =
          config.robot_reachability_provider->minimumSignedDistance(
              start_capsules,
              human_reach.center,
              human_reach.center,
              human_reach.radius);
      if (!std::isfinite(start_distance)) {
        markIntervalUnsafe(interval_index);
        out.joint_limit_unsafe = true;
        return false;
      }
    } else if (!config.assume_human_workspace_clear) {
      start_distance =
          config.human_workspace.signedDistanceToInflatedSphere(
              collision_start,
              config.human_workspace.inflatedCollisionRadius(
                  config.ee_collision_radius, 0.0),
              config.wall_time_sec + t_prev);
    }
    const bool collision_area_active_for_sample = start_distance <= 0.0;
    const bool energy_scaling_active_for_sample =
        config.enable_runtime_energy_scaling &&
        collision_area_active_for_sample;

    const Matrix6d nominal_stiffness =
        positiveSemidefinitePart(desired.K);
    const double nominal_start_joint_kinetic =
        jointKineticEnergy(dq_pred, state.inertia);
    const double nominal_start_potential =
        quadraticEnergy(error, nominal_stiffness);
    const double nominal_start_nullspace_potential =
        nullspacePotentialEnergy(
            q_pred,
            config.nullspace_reference,
            config.nullspace_stiffness);
    const double energy_stiffness_scale =
        energy_scaling_active_for_sample
            ? energyBudgetStiffnessScale(
                  nominal_start_joint_kinetic,
                  nominal_start_potential,
                  nominal_start_nullspace_potential,
                  energy_budget_eff)
            : 1.0;
    const Matrix6d applied_stiffness =
        energy_stiffness_scale * nominal_stiffness;
    const Matrix6d applied_damping =
        std::sqrt(energy_stiffness_scale) * desired.D;
    const double applied_nullspace_stiffness =
        energy_stiffness_scale *
        std::max(0.0, config.nullspace_stiffness);

    overbudget_joint_terms = updateOverbudgetJointStabilization(
        q_pred,
        nominal_start_joint_kinetic,
        energy_budget_eff,
        config.overbudget_joint_stiffness,
        config.overbudget_joint_scale_omega,
        config.enable_overbudget_joint_stabilization &&
            collision_area_active_for_sample,
        &overbudget_joint_state);

    Vector6d wrench =
        -applied_stiffness * error -
        applied_damping * (twist - desired_twist);
    if (config.use_dynamic_consistent_impedance) {
      wrench += lambda *
                (desired_acceleration - state.control_jdot_dq);
    }
    const Vector7d tau_task = state.control_jacobian.transpose() * wrench;

    Vector7d tau_nullspace = Vector7d::Zero();
    // Normal nullspace control is a global optional mode. Its stiffness and
    // damping receive the same Eq. (14) scale as Cartesian impedance whenever
    // the predicted state is in the human collision area.
    const bool nullspace_enabled_for_sample =
        applied_nullspace_stiffness > 0.0;
    previous_edge_cartesian_kinetic =
        quadraticEnergy(twist, positiveSemidefinitePart(lambda));
    previous_edge_joint_kinetic =
        nominal_start_joint_kinetic + kinetic_energy_error_bound;
    previous_edge_potential =
        quadraticEnergy(error, applied_stiffness) +
        potential_energy_error_bound;
    previous_edge_nullspace_potential = nullspace_enabled_for_sample
        ? nullspacePotentialEnergy(
              q_pred,
              config.nullspace_reference,
              applied_nullspace_stiffness) +
              nullspace_potential_energy_error_bound
        : 0.0;
    previous_edge_total_energy =
        previous_edge_joint_kinetic + previous_edge_potential +
        previous_edge_nullspace_potential;
    if (nullspace_enabled_for_sample) {
      const Vector7d tau_nullspace_raw =
          applied_nullspace_stiffness *
              (config.nullspace_reference - q_pred) -
          2.0 * std::sqrt(applied_nullspace_stiffness) * dq_pred;
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
        tau_task + tau_nullspace + overbudget_joint_terms.torque +
        state.coriolis;
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

    recordLimitViolation(
        q_next, dq_next, ddq, torque_command, interval_index);
    if (!q_next.allFinite() || !dq_next.allFinite()) {
      markIntervalUnsafe(interval_index);
      out.joint_limit_unsafe = true;
      return false;
    }

    JointDynamicsSample next_state;
    if (!dynamics.evaluate(q_next, dq_next, &next_state) ||
        !next_state.valid) {
      markIntervalUnsafe(interval_index);
      out.joint_limit_unsafe = true;
      return false;
    }

    Matrix6d tracking_tube_next = tracking_tube;
    double next_position_error_radius = 0.0;
    double next_orientation_error_radius = 0.0;
    double segment_distance = std::numeric_limits<double>::infinity();
    int segment_robot_link_index = -1;
    // SaRA intervals are intentionally deferred until the complete intended +
    // failsafe q/dq rollout is available and dynamic alpha_i has been derived.
    if (!config.assume_human_workspace_clear && !use_sara_robot_reach) {
      const Matrix3d task_inertia_inv =
          (state.control_jacobian * inertia_inv *
           state.control_jacobian.transpose()).topLeftCorner<3, 3>();
      tracking_tube_next = propagateTrackingTube(
          tracking_tube,
          task_inertia_inv,
          positiveSemidefinitePart(
              Matrix3d(applied_stiffness.topLeftCorner<3, 3>())),
          positiveSemidefinitePart(
              Matrix3d(applied_damping.topLeftCorner<3, 3>())),
          h,
          config.tracking_acc_error_bound);
      next_position_error_radius = maxBlockRadius(tracking_tube_next, 0);
      next_orientation_error_radius = 0.0;
      const double rho_p = trackingGeometryInflation(
          std::max(previous_position_error_radius,
                   next_position_error_radius),
          std::max(previous_orientation_error_radius,
                   next_orientation_error_radius),
          config.collision_center_offset.norm());
      const Vector3d collision_end = collisionPosition(next_state);
      segment_distance =
          config.human_workspace.signedDistanceSegmentToInflatedSphere(
              collision_start,
              collision_end,
              config.human_workspace.inflatedCollisionRadius(
                  config.ee_collision_radius, rho_p),
              config.wall_time_sec + t_prev,
              config.wall_time_sec + desired.t);
    }
    if (segment_distance < out.workspace_distance_min) {
      out.workspace_distance_min = segment_distance;
      out.worst_case_robot_link_index = segment_robot_link_index;
    }

    const Vector6d next_error = poseError(next_state, desired);
    Matrix6d next_lambda = Matrix6d::Zero();
    Matrix7d next_inertia_inv = Matrix7d::Zero();
    const bool next_lambda_valid = taskInertia(
        next_state.inertia, next_state.control_jacobian,
        &next_lambda, &next_inertia_inv);
    const Vector6d next_twist =
        next_state.control_jacobian * dq_next;
    const double cartesian_kinetic = next_lambda_valid
        ? quadraticEnergy(
              next_twist, positiveSemidefinitePart(next_lambda))
        : 0.0;
    const double nominal_joint_kinetic =
        jointKineticEnergy(dq_next, next_state.inertia);
    const double nominal_potential = quadraticEnergy(
        next_error, applied_stiffness);
    const double nominal_nullspace_potential =
        nullspace_enabled_for_sample
            ? nullspacePotentialEnergy(
                  q_next,
                  config.nullspace_reference,
                  applied_nullspace_stiffness)
            : 0.0;
    const OverbudgetJointStabilizationTerms endpoint_overbudget_terms =
        updateOverbudgetJointStabilization(
            q_next,
            nominal_joint_kinetic,
            energy_budget_eff,
            config.overbudget_joint_stiffness,
            config.overbudget_joint_scale_omega,
            config.enable_overbudget_joint_stabilization &&
                collision_area_active_for_sample,
            &overbudget_joint_state);
    const EnergyUpperBound energy_ub = addOneSidedEnergyErrorBounds(
        nominal_joint_kinetic,
        nominal_potential,
        nominal_nullspace_potential,
        kinetic_energy_error_bound,
        potential_energy_error_bound,
        nullspace_enabled_for_sample
            ? nullspace_potential_energy_error_bound
            : 0.0);
    const double joint_kinetic = energy_ub.kinetic;
    const double potential = energy_ub.cartesian_potential;
    const double nullspace_potential = energy_ub.nullspace_potential;
    const double total_energy = energy_ub.total();

    JointPredictionSample endpoint_prediction;
    endpoint_prediction.t = desired.t;
    endpoint_prediction.q = q_next;
    endpoint_prediction.dq = dq_next;
    endpoint_prediction.energy_valid = true;
    endpoint_prediction.joint_kinetic_energy = nominal_joint_kinetic;
    endpoint_prediction.cartesian_potential_energy = nominal_potential;
    endpoint_prediction.nullspace_potential_energy =
        nominal_nullspace_potential;
    endpoint_prediction.nullspace_potential_energy_active =
        nullspace_enabled_for_sample;
    endpoint_prediction.energy_scaling_active =
        energy_scaling_active_for_sample;
    endpoint_prediction.energy_stiffness_scale =
        energy_stiffness_scale;
    endpoint_prediction.applied_nullspace_stiffness =
        applied_nullspace_stiffness;
    endpoint_prediction.overbudget_joint_stabilization_active =
        endpoint_overbudget_terms.active;
    endpoint_prediction.overbudget_joint_potential_energy =
        endpoint_overbudget_terms.potential_energy;
    endpoint_prediction.overbudget_joint_scale_rho =
        endpoint_overbudget_terms.scale_rho;
    endpoint_prediction.overbudget_joint_torque_norm =
        endpoint_overbudget_terms.torque.norm();
    rollout_trace.push_back(endpoint_prediction);

    const bool previous_edge_is_worst =
        previous_edge_total_energy >= total_energy;
    const double interval_energy =
        std::max(previous_edge_total_energy, total_energy);
    if (use_sara_robot_reach) {
      PredictedReachInterval interval;
      interval.index = interval_index;
      interval.start_time = previous_edge_time;
      interval.end_time = desired.t;
      interval.start_q = q_pred;
      interval.end_q = q_next;
      interval.start_cartesian_kinetic =
          previous_edge_cartesian_kinetic;
      interval.start_joint_kinetic = previous_edge_joint_kinetic;
      interval.start_potential = previous_edge_potential;
      interval.start_nullspace_potential =
          previous_edge_nullspace_potential;
      interval.start_total_energy = previous_edge_total_energy;
      interval.end_cartesian_kinetic = cartesian_kinetic;
      interval.end_joint_kinetic = joint_kinetic;
      interval.end_potential = potential;
      interval.end_nullspace_potential = nullspace_potential;
      interval.end_total_energy = total_energy;
      predicted_reach_intervals.push_back(interval);
    } else {
      if (segment_distance <= 0.0 &&
          out.first_contact_interval_index < 0) {
        out.first_contact_interval_index = interval_index;
      }
      if (segment_distance <= 0.0 && interval_energy > energy_budget_eff) {
        if (out.first_energy_unsafe_contact_interval_index < 0) {
          out.first_energy_unsafe_contact_interval_index = interval_index;
        }
        markIntervalUnsafe(interval_index);
      }
      if (segment_distance <= 0.0 &&
          interval_energy > total_contact_energy_max) {
        total_contact_energy_max = interval_energy;
        out.worst_case_contact_time =
            previous_edge_is_worst ? previous_edge_time : desired.t;
        out.worst_case_workspace_distance_at_candidate = segment_distance;
        out.worst_case_robot_link_index = segment_robot_link_index;
        out.worst_case_cartesian_kinetic_energy_ub =
            previous_edge_is_worst
                ? previous_edge_cartesian_kinetic
                : cartesian_kinetic;
        out.worst_case_joint_kinetic_energy_ub =
            previous_edge_is_worst
                ? previous_edge_joint_kinetic
                : joint_kinetic;
        out.worst_case_cartesian_potential_energy_ub =
            previous_edge_is_worst ? previous_edge_potential : potential;
        out.worst_case_nullspace_potential_energy_ub =
            previous_edge_is_worst
                ? previous_edge_nullspace_potential
                : nullspace_potential;
        // Kept for CSV/API compatibility; this now denotes the complete
        // T + U_x + U_n metric that gates the paper energy budget.
        out.worst_case_cartesian_control_energy_ub = interval_energy;
        out.worst_case_total_control_energy_ub = interval_energy;
      }
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
    previous_edge_cartesian_kinetic = cartesian_kinetic;
    previous_edge_joint_kinetic = joint_kinetic;
    previous_edge_potential = potential;
    previous_edge_nullspace_potential = nullspace_potential;
    previous_edge_total_energy = total_energy;
    previous_edge_time = desired.t;
    if (!use_sara_robot_reach) {
      previous_position_error_radius = next_position_error_radius;
      previous_orientation_error_radius = next_orientation_error_radius;
      out.worst_case_pos_error_radius = std::max(
          out.worst_case_pos_error_radius,
          next_position_error_radius);
      out.worst_case_orientation_error_radius = std::max(
          out.worst_case_orientation_error_radius,
          next_orientation_error_radius);
      out.worst_case_vel_error_radius = std::max(
          out.worst_case_vel_error_radius,
          maxBlockRadius(tracking_tube, 3));
    }
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
    sample.nominal_path_kinematics_valid =
        start.nominal_path_kinematics_valid &&
        end.nominal_path_kinematics_valid;
    if (sample.nominal_path_kinematics_valid) {
      sample.nominal_path_rate =
          (1.0 - u) * start.nominal_path_rate +
          u * end.nominal_path_rate;
      sample.nominal_path_acceleration =
          (1.0 - u) * start.nominal_path_acceleration +
          u * end.nominal_path_acceleration;
    }
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
      const double segment_dt = desired.t - previous_desired.t;
      if (!std::isfinite(segment_dt) || segment_dt <= 0.0) {
        markIntervalUnsafe(monitored_interval_index);
        out.joint_limit_unsafe = true;
        return false;
      }
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
  bool failsafe_complete = false;
  if (intended_complete && !out.joint_limit_unsafe) {
    failsafe_complete = evaluateStage(plan.failsafe);
  }
  const bool complete_monitored_rollout =
      intended_complete && failsafe_complete && !out.joint_limit_unsafe;

  if (prediction_trace != nullptr) {
    *prediction_trace = rollout_trace;
  }

  if (use_sara_robot_reach && complete_monitored_rollout) {
    std::vector<double> dynamic_alpha;
    const bool interval_trace_consistent =
        rollout_trace.size() == predicted_reach_intervals.size() + 1;
    if (!interval_trace_consistent ||
        !config.robot_reachability_provider->calculateTrajectoryAlpha(
            rollout_trace, &dynamic_alpha) ||
        dynamic_alpha.size() != 7 ||
        std::any_of(
            dynamic_alpha.begin(), dynamic_alpha.end(), [](double value) {
              return !std::isfinite(value) || value < 0.0;
            })) {
      // Fail closed when SaRA's dynamic alpha cannot be derived from exactly
      // the q/dq samples that define the monitored intervals.
      markIntervalUnsafe(0);
      out.joint_limit_unsafe = true;
    } else {
      out.robot_reach_alpha_valid = true;
      for (int i = 0; i < 7; ++i) {
        out.robot_reach_alpha(i) = dynamic_alpha[static_cast<std::size_t>(i)];
      }

      if (!config.assume_human_workspace_clear) {
        for (const auto& interval : predicted_reach_intervals) {
          const double interval_duration =
              interval.end_time - interval.start_time;
          std::vector<RobotReachCapsule> robot_capsules;
          if (!std::isfinite(interval_duration) ||
              interval_duration <= 0.0 ||
              !config.robot_reachability_provider->reachInterval(
                  interval.start_q,
                  interval.end_q,
                  interval_duration,
                  dynamic_alpha,
                  &robot_capsules)) {
            markIntervalUnsafe(interval.index);
            out.joint_limit_unsafe = true;
            break;
          }

          int segment_robot_link_index = -1;
          const auto human_reach =
              config.human_workspace.handReachableSetAtTime(
                  config.wall_time_sec + interval.end_time);
          const double segment_distance =
              config.robot_reachability_provider->minimumSignedDistance(
                  robot_capsules,
                  human_reach.center,
                  human_reach.center,
                  human_reach.radius,
                  &segment_robot_link_index);
          if (!std::isfinite(segment_distance)) {
            markIntervalUnsafe(interval.index);
            out.joint_limit_unsafe = true;
            break;
          }
          if (segment_distance < out.workspace_distance_min) {
            out.workspace_distance_min = segment_distance;
            out.worst_case_robot_link_index = segment_robot_link_index;
          }
          if (segment_distance <= 0.0 &&
              out.first_contact_interval_index < 0) {
            out.first_contact_interval_index = interval.index;
          }

          const bool start_is_worst =
              interval.start_total_energy >= interval.end_total_energy;
          const double interval_energy = std::max(
              interval.start_total_energy, interval.end_total_energy);
          if (segment_distance <= 0.0 &&
              interval_energy > energy_budget_eff) {
            if (out.first_energy_unsafe_contact_interval_index < 0) {
              out.first_energy_unsafe_contact_interval_index = interval.index;
            }
            markIntervalUnsafe(interval.index);
          }
          if (segment_distance <= 0.0 &&
              interval_energy > total_contact_energy_max) {
            total_contact_energy_max = interval_energy;
            out.worst_case_contact_time = start_is_worst
                ? interval.start_time
                : interval.end_time;
            out.worst_case_workspace_distance_at_candidate =
                segment_distance;
            out.worst_case_robot_link_index = segment_robot_link_index;
            out.worst_case_cartesian_kinetic_energy_ub = start_is_worst
                ? interval.start_cartesian_kinetic
                : interval.end_cartesian_kinetic;
            out.worst_case_joint_kinetic_energy_ub = start_is_worst
                ? interval.start_joint_kinetic
                : interval.end_joint_kinetic;
            out.worst_case_cartesian_potential_energy_ub = start_is_worst
                ? interval.start_potential
                : interval.end_potential;
            out.worst_case_nullspace_potential_energy_ub = start_is_worst
                ? interval.start_nullspace_potential
                : interval.end_nullspace_potential;
            out.worst_case_cartesian_control_energy_ub = interval_energy;
            out.worst_case_total_control_energy_ub = interval_energy;
          }
        }
      }
    }
  } else if (use_sara_robot_reach && !out.joint_limit_unsafe) {
    // A partial intended/failsafe trajectory must never be verified with an
    // alpha vector computed from only the available prefix.
    markIntervalUnsafe(0);
    out.joint_limit_unsafe = true;
  }

  out.monitored_contact_possible = out.workspace_distance_min <= 0.0;
  out.workspace_distance_margin = out.workspace_distance_min;
  out.terminal_energy_ub =
      terminal_sample_found ? terminal_total_energy : 0.0;
  // Current overlap is both a contact-risk diagnostic and the runtime
  // activation gate for Eq. (14) stiffness scaling.
  out.contact_relevant_for_energy = out.workspace_distance_now <= 0.0;

  const bool predicted_contact_requires_verification =
      out.monitored_contact_possible;
  const bool current_collision_energy_unsafe =
      out.contact_relevant_for_energy &&
      out.current_joint_energy_valid &&
      out.current_total_control_energy > energy_budget_eff;
  const bool predicted_contact_energy_unsafe =
      predicted_contact_requires_verification &&
      total_contact_energy_max > energy_budget_eff;

  out.predicted_trigger =
      predicted_contact_energy_unsafe || out.joint_limit_unsafe;
  out.collision_energy_unsafe =
      current_collision_energy_unsafe || predicted_contact_energy_unsafe;
  out.monitored_unsafe =
      out.predicted_trigger ||
      (out.contact_relevant_for_energy && current_collision_energy_unsafe);
  return out;
}

}  // namespace cps_safety_monitor
