#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "cps_trajectory_generators/reachable_cartesian_trajectory.hpp"

namespace {

using cps_trajectory_generators::CartesianTrajectorySample;
using cps_trajectory_generators::PathConsistentTimedPathConfig;
using cps_trajectory_generators::makePathConsistentTimedPathIntendedPrefix;
using cps_trajectory_generators::makeRetimedPathState;

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

}  // namespace
