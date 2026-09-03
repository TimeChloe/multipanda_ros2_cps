#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "cps_trajectory_generators/reachable_cartesian_trajectory.hpp"

namespace {

using cps_trajectory_generators::CartesianTrajectorySample;
using cps_trajectory_generators::LocalCartesianReplanConfig;
using cps_trajectory_generators::PathConsistentTimedPathConfig;
using cps_trajectory_generators::TrajectoryGeneratorSettings;
using cps_trajectory_generators::makeCartesianBrakeTrajectory;
using cps_trajectory_generators::makePathConsistentTimedPathBrake;
using cps_trajectory_generators::makePathConsistentTimedPathIntendedPrefix;
using cps_trajectory_generators::makeRetimedPathState;
using cps_trajectory_generators::makeSmoothViaPointCartesianTrajectory;

TEST(TrajectoryGeneratorSettings, UsesSaraPandaFiveSampleReachabilityWindow) {
  const TrajectoryGeneratorSettings settings;
  EXPECT_DOUBLE_EQ(settings.local_replan_dt, 0.001);
  EXPECT_DOUBLE_EQ(settings.shield_plan_dt, 5.0 * settings.local_replan_dt);
}

Eigen::Quaterniond quaternionFromRotationVector(
    const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  if (angle < 1.0e-12) {
    return Eigen::Quaterniond::Identity();
  }
  return Eigen::Quaterniond(
      Eigen::AngleAxisd(angle, rotation_vector / angle));
}

Eigen::Vector3d rotationVectorFromQuaternion(
    const Eigen::Quaterniond& quaternion) {
  Eigen::Quaterniond normalized = quaternion.normalized();
  if (normalized.w() < 0.0) {
    normalized.coeffs() *= -1.0;
  }
  const Eigen::AngleAxisd angle_axis(normalized);
  if (angle_axis.angle() < 1.0e-12) {
    return Eigen::Vector3d::Zero();
  }
  return angle_axis.axis() * angle_axis.angle();
}

std::vector<CartesianTrajectorySample> makeStraightTimedPath() {
  std::vector<CartesianTrajectorySample> path;
  path.reserve(2001);
  for (int i = 0; i <= 2000; ++i) {
    const double path_time = static_cast<double>(i) * 0.001;
    CartesianTrajectorySample sample;
    sample.t = path_time;
    sample.p = Eigen::Vector3d(path_time, 0.0, 0.0);
    sample.dp = Eigen::Vector3d::UnitX();
    sample.q = Eigen::Quaterniond::Identity();
    path.push_back(sample);
  }
  return path;
}

PathConsistentTimedPathConfig makeConfig() {
  PathConsistentTimedPathConfig config;
  config.intended_steps = 100;
  config.dt = 0.001;
  config.project_start_to_nearest_path_state = false;
  config.max_path_rate = 1.0;
  config.max_path_acceleration = 1.0;
  config.max_path_jerk = 10.0;
  config.target_path_rate = 0.5;
  return config;
}

TEST(PathConsistentTimedPath, NeverMovesBackwardFromMeasuredDeceleration) {
  const auto path = makeStraightTimedPath();
  constexpr double kStartPathTime = 1.0;

  auto config = makeConfig();
  config.initial_path_rate = 0.02;
  config.initial_path_acceleration = -1.0;
  const CartesianTrajectorySample planning_start = makeRetimedPathState(
      path,
      kStartPathTime,
      config.initial_path_rate,
      config.initial_path_acceleration);

  const auto samples = makePathConsistentTimedPathIntendedPrefix(
      kStartPathTime, planning_start, path, config);

  ASSERT_EQ(samples.size(),
            static_cast<std::size_t>(config.intended_steps));
  double previous_path_time = kStartPathTime;
  for (const auto& sample : samples) {
    EXPECT_TRUE(std::isfinite(sample.t));
    EXPECT_GE(sample.t + 1.0e-12, previous_path_time);
    EXPECT_GE(sample.dp.x(), -1.0e-12);
    EXPECT_NEAR(sample.p.x(), sample.t, 1.0e-9);
    previous_path_time = sample.t;
  }
}

TEST(PathConsistentTimedPath, StartsFromMeasuredRateAndAcceleration) {
  const auto path = makeStraightTimedPath();
  constexpr double kStartPathTime = 1.0;

  auto config = makeConfig();
  config.initial_path_rate = 0.4;
  config.initial_path_acceleration = 0.3;
  config.target_path_rate = 0.8;
  const CartesianTrajectorySample planning_start = makeRetimedPathState(
      path,
      kStartPathTime,
      config.initial_path_rate,
      config.initial_path_acceleration);

  const auto samples = makePathConsistentTimedPathIntendedPrefix(
      kStartPathTime, planning_start, path, config);

  ASSERT_FALSE(samples.empty());
  const auto& first = samples.front();
  EXPECT_GT(first.t, kStartPathTime);
  EXPECT_NEAR(first.dp.x(), config.initial_path_rate, 1.0e-3);
  EXPECT_NEAR(first.ddp.x(), config.initial_path_acceleration, 2.0e-2);
}

TEST(PathConsistentTimedPath, AcceleratesForwardFromMeasuredRest) {
  const auto path = makeStraightTimedPath();
  constexpr double kStartPathTime = 0.75;

  auto config = makeConfig();
  config.initial_path_rate = 0.0;
  config.initial_path_acceleration = 0.0;
  config.target_path_rate = 1.0;
  const CartesianTrajectorySample planning_start = makeRetimedPathState(
      path,
      kStartPathTime,
      config.initial_path_rate,
      config.initial_path_acceleration);

  const auto samples = makePathConsistentTimedPathIntendedPrefix(
      kStartPathTime, planning_start, path, config);

  ASSERT_EQ(samples.size(),
            static_cast<std::size_t>(config.intended_steps));
  EXPECT_NEAR(samples.front().dp.x(), 0.0, 1.0e-3);
  EXPECT_GT(samples.back().dp.x(), samples.front().dp.x());

  double previous_path_time = kStartPathTime;
  for (const auto& sample : samples) {
    EXPECT_GE(sample.t + 1.0e-12, previous_path_time);
    EXPECT_GE(sample.dp.x(), -1.0e-12);
    previous_path_time = sample.t;
  }
}

TEST(PathConsistentTimedPath, ExplicitScalarStateCrossesZeroTangentCusp) {
  LocalCartesianReplanConfig geometry_config;
  geometry_config.dt = 0.001;
  geometry_config.waypoint_merge_position_tolerance = 0.0;
  geometry_config.waypoint_merge_orientation_tolerance = 0.0;
  geometry_config.max_velocity = 0.3;
  geometry_config.max_acceleration = 2.6;
  geometry_config.max_jerk = 100.0;

  const std::vector<Eigen::Vector3d> waypoints{
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.0, 0.0, 0.05),
      Eigen::Vector3d(0.0, 0.0, -0.5)};
  const auto timed_path = makeSmoothViaPointCartesianTrajectory(
      waypoints, Eigen::Quaterniond::Identity(), geometry_config);
  ASSERT_GT(timed_path.size(), 2u);

  const auto cusp = std::max_element(
      timed_path.begin(),
      timed_path.end(),
      [](const CartesianTrajectorySample& lhs,
         const CartesianTrajectorySample& rhs) {
        return lhs.p.z() < rhs.p.z();
      });
  ASSERT_NE(cusp, timed_path.end());
  EXPECT_NEAR(cusp->p.z(), 0.05, 1.0e-12);
  EXPECT_NEAR(cusp->dp.norm(), 0.0, 1.0e-12);

  const CartesianTrajectorySample planning_start = makeRetimedPathState(
      timed_path, cusp->t, 1.0, 0.0);
  ASSERT_TRUE(planning_start.path_kinematics_valid);
  EXPECT_DOUBLE_EQ(planning_start.path_rate, 1.0);

  auto config = makeConfig();
  config.intended_steps = 20;
  config.initial_path_rate = planning_start.path_rate;
  config.initial_path_acceleration = planning_start.path_acceleration;
  config.target_path_rate = 1.0;
  const auto intended = makePathConsistentTimedPathIntendedPrefix(
      cusp->t, planning_start, timed_path, config);

  ASSERT_EQ(intended.size(), 20u);
  EXPECT_GT(intended.front().t, cusp->t);
  EXPECT_GT(intended.back().t, cusp->t + 0.019);
  for (const auto& sample : intended) {
    EXPECT_TRUE(sample.path_kinematics_valid);
    EXPECT_NEAR(sample.path_rate, 1.0, 1.0e-12);
  }
}

TEST(PathConsistentTimedPath, BrakeUsesExplicitScalarStateAtZeroTangentCusp) {
  LocalCartesianReplanConfig geometry_config;
  geometry_config.dt = 0.001;
  geometry_config.waypoint_merge_position_tolerance = 0.0;
  geometry_config.waypoint_merge_orientation_tolerance = 0.0;
  geometry_config.max_velocity = 0.3;
  geometry_config.max_acceleration = 2.6;
  geometry_config.max_jerk = 100.0;

  const std::vector<Eigen::Vector3d> waypoints{
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.0, 0.0, 0.05),
      Eigen::Vector3d(0.0, 0.0, -0.5)};
  const auto timed_path = makeSmoothViaPointCartesianTrajectory(
      waypoints, Eigen::Quaterniond::Identity(), geometry_config);
  ASSERT_GT(timed_path.size(), 2u);
  const auto cusp = std::max_element(
      timed_path.begin(),
      timed_path.end(),
      [](const CartesianTrajectorySample& lhs,
         const CartesianTrajectorySample& rhs) {
        return lhs.p.z() < rhs.p.z();
      });
  ASSERT_NE(cusp, timed_path.end());

  const CartesianTrajectorySample brake_start = makeRetimedPathState(
      timed_path, cusp->t, 1.0, 0.0);
  auto config = makeConfig();
  config.dt = 0.001;
  config.initial_path_rate = brake_start.path_rate;
  config.initial_path_acceleration = brake_start.path_acceleration;
  config.max_path_acceleration = 10.0;
  config.max_path_jerk = 5000.0;
  const auto brake = makePathConsistentTimedPathBrake(
      cusp->t, brake_start, timed_path, config);

  ASSERT_FALSE(brake.empty());
  EXPECT_GT(brake.back().t, cusp->t);
  EXPECT_TRUE(brake.front().path_kinematics_valid);
  EXPECT_GT(brake.front().path_rate, 0.0);
  EXPECT_NEAR(brake.back().path_rate, 0.0, 1.0e-12);
}

TEST(SmoothViaPointCartesianTrajectory, UsesLinearLimitsAsThreeDimensionalNorms) {
  LocalCartesianReplanConfig config;
  config.dt = 0.001;
  config.max_velocity = 0.3;
  config.max_acceleration = 0.8;
  config.max_jerk = 4.0;
  config.max_angular_velocity = 2.0;
  config.max_angular_acceleration = 10.0;
  config.max_angular_jerk = 100.0;

  const std::vector<Eigen::Vector3d> waypoints{
      Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 1.0, 1.0)};
  const auto samples = makeSmoothViaPointCartesianTrajectory(
      waypoints, Eigen::Quaterniond::Identity(), config);

  ASSERT_GT(samples.size(), 2u);
  double maximum_speed = 0.0;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    maximum_speed = std::max(maximum_speed, samples[i].dp.norm());
    EXPECT_LE(samples[i].dp.norm(), config.max_velocity + 1.0e-8);
    EXPECT_LE(samples[i].ddp.norm(), config.max_acceleration + 1.0e-6);
    if (i > 0) {
      const double sample_dt = samples[i].t - samples[i - 1].t;
      ASSERT_GT(sample_dt, 0.0);
      const double sampled_jerk =
          (samples[i].ddp - samples[i - 1].ddp).norm() / sample_dt;
      EXPECT_LE(sampled_jerk, config.max_jerk + 1.0e-3);
    }
  }

  // A per-axis interpretation would permit sqrt(3) times this speed. The
  // scalar path parameterization instead reaches the configured vector norm.
  EXPECT_GT(maximum_speed, 0.99 * config.max_velocity);
}

TEST(SmoothViaPointCartesianTrajectory,
     WorldAngularDerivativesMatchQuaternionForChangingRotationAxis) {
  LocalCartesianReplanConfig config;
  config.dt = 0.001;
  config.waypoint_merge_position_tolerance = 0.0;
  config.waypoint_merge_orientation_tolerance = 0.0;
  config.max_velocity = 0.3;
  config.max_acceleration = 2.0;
  config.max_jerk = 100.0;
  config.max_angular_velocity = 0.4;
  config.max_angular_acceleration = 2.0;
  config.max_angular_jerk = 20.0;

  const std::vector<Eigen::Vector3d> waypoints{
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero()};
  const std::vector<Eigen::Quaterniond> orientations{
      quaternionFromRotationVector(Eigen::Vector3d::Zero()),
      quaternionFromRotationVector(Eigen::Vector3d(0.8, 0.0, 0.0)),
      quaternionFromRotationVector(Eigen::Vector3d(1.0, 0.7, 0.2))};
  const auto samples = makeSmoothViaPointCartesianTrajectory(
      waypoints, orientations, config);

  ASSERT_GT(samples.size(), 10u);
  double maximum_velocity_error = 0.0;
  double maximum_acceleration_error = 0.0;
  double maximum_coordinate_velocity_difference = 0.0;
  double maximum_sampled_angular_jerk = 0.0;
  for (std::size_t i = 1; i + 1 < samples.size(); ++i) {
    const double centered_dt = samples[i + 1].t - samples[i - 1].t;
    ASSERT_GT(centered_dt, 0.0);

    const Eigen::Quaterniond relative_rotation =
        samples[i + 1].q * samples[i - 1].q.conjugate();
    const Eigen::Vector3d finite_difference_omega =
        rotationVectorFromQuaternion(relative_rotation) / centered_dt;
    const Eigen::Vector3d finite_difference_alpha =
        (samples[i + 1].w - samples[i - 1].w) / centered_dt;
    const Eigen::Vector3d previous_rotation_vector =
        rotationVectorFromQuaternion(samples[i - 1].q);
    const Eigen::Vector3d next_rotation_vector =
        rotationVectorFromQuaternion(samples[i + 1].q);
    const Eigen::Vector3d finite_difference_rotation_vector_rate =
        (next_rotation_vector - previous_rotation_vector) / centered_dt;

    maximum_velocity_error = std::max(
        maximum_velocity_error,
        (samples[i].w - finite_difference_omega).norm());
    maximum_acceleration_error = std::max(
        maximum_acceleration_error,
        (samples[i].dw - finite_difference_alpha).norm());
    maximum_coordinate_velocity_difference = std::max(
        maximum_coordinate_velocity_difference,
        (samples[i].w - finite_difference_rotation_vector_rate).norm());
    maximum_sampled_angular_jerk = std::max(
        maximum_sampled_angular_jerk,
        (samples[i + 1].dw - samples[i - 1].dw).norm() / centered_dt);

    EXPECT_LE(samples[i].w.norm(), config.max_angular_velocity + 1.0e-8);
    EXPECT_LE(
        samples[i].dw.norm(), config.max_angular_acceleration + 1.0e-6);
  }

  EXPECT_LT(maximum_velocity_error, 2.0e-4);
  EXPECT_LT(maximum_acceleration_error, 2.0e-2);
  EXPECT_LE(
      maximum_sampled_angular_jerk, config.max_angular_jerk + 2.0e-2);
  // Ensure this trajectory exercises the non-Euclidean case. With the old
  // implementation sample.w was the rotation-vector rate, so this difference
  // was approximately zero while the quaternion finite difference disagreed.
  EXPECT_GT(maximum_coordinate_velocity_difference, 1.0e-2);
}

TEST(SmoothViaPointCartesianTrajectory,
     MergesRedundantIntermediatePointWithoutSlowingWholePath) {
  LocalCartesianReplanConfig config;
  config.dt = 0.001;
  config.waypoint_merge_position_tolerance = 0.001;
  config.waypoint_merge_orientation_tolerance = 0.005;
  config.max_velocity = 0.34;
  config.max_acceleration = 2.6;
  config.max_jerk = 1300.0;
  config.max_angular_velocity = 0.5;
  config.max_angular_acceleration = 5.0;
  config.max_angular_jerk = 2500.0;

  constexpr double kStartZ = 0.69991234;
  const std::vector<Eigen::Vector3d> waypoints{
      Eigen::Vector3d(0.0, 0.0, kStartZ),
      Eigen::Vector3d(0.0, 0.0, 0.7),
      Eigen::Vector3d(0.0, 0.0, 0.1)};
  const auto samples = makeSmoothViaPointCartesianTrajectory(
      waypoints, Eigen::Quaterniond::Identity(), config);

  ASSERT_GT(samples.size(), 2u);
  double maximum_speed = 0.0;
  for (const auto& sample : samples) {
    maximum_speed = std::max(maximum_speed, sample.dp.norm());
    EXPECT_LE(sample.p.z(), kStartZ + 1.0e-10);
    EXPECT_LE(sample.dp.norm(), config.max_velocity + 1.0e-8);
    EXPECT_LE(sample.ddp.norm(), config.max_acceleration + 1.0e-6);
  }
  EXPECT_GT(maximum_speed, 0.99 * config.max_velocity);
  EXPECT_NEAR(samples.back().p.z(), 0.1, 1.0e-12);
}

TEST(SmoothViaPointCartesianTrajectory,
     LocalizesARealReversalInsteadOfSlowingFollowingSegment) {
  LocalCartesianReplanConfig config;
  config.dt = 0.001;
  config.waypoint_merge_position_tolerance = 0.0;
  config.waypoint_merge_orientation_tolerance = 0.0;
  config.max_velocity = 0.3;
  config.max_acceleration = 2.6;
  config.max_jerk = 100.0;
  config.max_angular_velocity = 0.5;
  config.max_angular_acceleration = 5.0;
  config.max_angular_jerk = 100.0;

  const std::vector<Eigen::Vector3d> waypoints{
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.0, 0.0, 0.05),
      Eigen::Vector3d(0.0, 0.0, -1.0)};
  const auto samples = makeSmoothViaPointCartesianTrajectory(
      waypoints, Eigen::Quaterniond::Identity(), config);

  ASSERT_GT(samples.size(), 2u);
  double maximum_following_speed = 0.0;
  double maximum_z = -std::numeric_limits<double>::infinity();
  double closest_reversal_distance = std::numeric_limits<double>::infinity();
  double speed_at_closest_reversal = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto& sample = samples[i];
    maximum_z = std::max(maximum_z, sample.p.z());
    const double reversal_distance = std::abs(sample.p.z() - 0.05);
    if (reversal_distance < closest_reversal_distance) {
      closest_reversal_distance = reversal_distance;
      speed_at_closest_reversal = sample.dp.norm();
    }
    if (sample.p.z() < -0.2) {
      maximum_following_speed =
          std::max(maximum_following_speed, sample.dp.norm());
    }
    EXPECT_LE(sample.dp.norm(), config.max_velocity + 1.0e-8);
    EXPECT_LE(sample.ddp.norm(), config.max_acceleration + 1.0e-6);
    if (i > 0) {
      const double sample_dt = samples[i].t - samples[i - 1].t;
      ASSERT_GT(sample_dt, 0.0);
      const double sampled_jerk =
          (sample.ddp - samples[i - 1].ddp).norm() / sample_dt;
      EXPECT_LE(sampled_jerk, config.max_jerk + 1.0e-3);
    }
  }
  EXPECT_LE(maximum_z, 0.05 + 1.0e-10);
  EXPECT_LE(closest_reversal_distance, 1.0e-12);
  EXPECT_LE(speed_at_closest_reversal, 1.0e-12);
  EXPECT_GT(maximum_following_speed, 0.99 * config.max_velocity);
  EXPECT_NEAR(samples.back().p.z(), -1.0, 1.0e-12);
}

TEST(SmoothViaPointCartesianTrajectory,
     RetainsAnIntentionalTinyFinalTarget) {
  LocalCartesianReplanConfig config;
  config.dt = 0.001;
  config.waypoint_merge_position_tolerance = 0.001;
  config.waypoint_merge_orientation_tolerance = 0.005;
  config.max_velocity = 0.34;
  config.max_acceleration = 2.6;
  config.max_jerk = 1300.0;

  const std::vector<Eigen::Vector3d> waypoints{
      Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.0001)};
  const auto samples = makeSmoothViaPointCartesianTrajectory(
      waypoints, Eigen::Quaterniond::Identity(), config);

  ASSERT_GT(samples.size(), 1u);
  EXPECT_NEAR(samples.front().p.z(), 0.0, 1.0e-12);
  EXPECT_NEAR(samples.back().p.z(), 0.0001, 1.0e-12);
}

TEST(CartesianBrakeTrajectory, SupportsFullNormAlongAxisAndDiagonal) {
  LocalCartesianReplanConfig config;
  config.dt = 0.001;
  config.max_velocity = 0.3;
  config.max_acceleration = 0.8;
  config.max_jerk = 4.0;
  config.max_angular_velocity = 2.0;
  config.max_angular_acceleration = 10.0;
  config.max_angular_jerk = 100.0;

  const std::vector<Eigen::Vector3d> directions{
      Eigen::Vector3d::UnitX(), Eigen::Vector3d::Ones().normalized()};
  for (const auto& direction : directions) {
    CartesianTrajectorySample start;
    start.dp = config.max_velocity * direction;
    const auto samples = makeCartesianBrakeTrajectory(start, config);

    ASSERT_FALSE(samples.empty());
    EXPECT_GT(samples.front().dp.norm(), 0.99 * config.max_velocity);
    Eigen::Vector3d previous_acceleration = start.ddp;
    for (const auto& sample : samples) {
      EXPECT_LE(sample.dp.norm(), config.max_velocity + 1.0e-8);
      EXPECT_LE(sample.ddp.norm(), config.max_acceleration + 1.0e-6);
      const double sampled_jerk =
          (sample.ddp - previous_acceleration).norm() / config.dt;
      EXPECT_LE(sampled_jerk, config.max_jerk + 1.0e-3);
      previous_acceleration = sample.ddp;
    }
  }
}

}  // namespace
