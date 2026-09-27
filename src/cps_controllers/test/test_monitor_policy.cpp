// Copyright (c) 2026
#include <limits>
#include <gtest/gtest.h>
#include <cps_controllers/reachable_cartesian_impedance/monitor_policy.hpp>
#include <cps_trajectory_generators/reachable_cartesian_trajectory.hpp>

using cps_controllers::detail::monitorPeriodSteps;
using cps_controllers::detail::withinCommittedPrefix;

TEST(MonitorPolicy, UsesTheSamePeriodForBothSegments) {
  EXPECT_EQ(monitorPeriodSteps(0.001, 200.0), 5U);
  EXPECT_EQ(monitorPeriodSteps(0.001, 100.0), 10U);
  EXPECT_EQ(monitorPeriodSteps(0.002, 100.0), 5U);
  EXPECT_EQ(monitorPeriodSteps(0.001, 10.0), 100U);
}

TEST(MonitorPolicy, RejectsRatesThatCannotShareTheControlGrid) {
  EXPECT_THROW(monitorPeriodSteps(0.001, 300.0), std::invalid_argument);
  EXPECT_THROW(monitorPeriodSteps(0.001, 2000.0), std::invalid_argument);
}

TEST(MonitorPolicy, RejectsInvalidAndUnrepresentablePeriods) {
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(monitorPeriodSteps(invalid, 100.0), std::invalid_argument);
    EXPECT_THROW(monitorPeriodSteps(0.001, invalid), std::invalid_argument);
  }
  EXPECT_THROW(monitorPeriodSteps(0.001, 1e-20), std::invalid_argument);
}

TEST(MonitorPolicy, AcceptsTheDeadlineButNeverCatchesUpAfterIt) {
  EXPECT_TRUE(withinCommittedPrefix(100, 102, 10));
  EXPECT_TRUE(withinCommittedPrefix(100, 110, 10));
  EXPECT_FALSE(withinCommittedPrefix(100, 111, 10));
  EXPECT_FALSE(withinCommittedPrefix(100, 119, 10));
}

TEST(MonitorPolicy, RejectsEmptyPrefixesAndRegressingControlSequences) {
  EXPECT_FALSE(withinCommittedPrefix(100, 100, 0));
  EXPECT_FALSE(withinCommittedPrefix(100, 99, 10));
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  EXPECT_TRUE(withinCommittedPrefix(maximum - 10, maximum, 10));
  EXPECT_FALSE(withinCommittedPrefix(maximum - 10, 0, 10));
}

TEST(MonitorPolicy, GeneratesAFullPeriodEvenAtThePathEnd) {
  using namespace cps_trajectory_generators;
  CartesianTrajectorySample start;
  start.t = 0.0;
  start.q = Eigen::Quaterniond::Identity();
  CartesianTrajectorySample end = start;
  end.t = 1.0;
  end.p.x() = 0.1;
  const std::vector<CartesianTrajectorySample> path{start, end};
  for (double frequency : {200.0, 100.0, 10.0}) {
    PathConsistentTimedPathConfig config;
    config.dt = 0.001;
    config.intended_steps = static_cast<int>(monitorPeriodSteps(config.dt, frequency));
    const auto samples = makePathConsistentTimedPathIntendedPrefix(
        end.t, end, path, config);
    ASSERT_EQ(samples.size(), static_cast<std::size_t>(config.intended_steps));
    for (const auto& sample : samples) {
      EXPECT_TRUE(sample.p.isApprox(end.p));
      EXPECT_TRUE(sample.dp.isZero());
      EXPECT_TRUE(sample.ddp.isZero());
    }
  }
}
