#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "reachable_cartesian_impedance/timing.hpp"

TEST(WorkerTiming, ThreadCpuClockDoesNotChargeSleepingTime)
{
  using namespace cps_controllers::detail;
  const auto wall_start = steadyNowNanoseconds();
  const auto start = workerCpuSample();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const auto finish = workerCpuSample();
  const auto wall_end = steadyNowNanoseconds();
  ASSERT_GE(start.cpu_ns, 0);
  ASSERT_GE(finish.cpu_ns, start.cpu_ns);
  const double cpu_ms = nanosecondsToMilliseconds(finish.cpu_ns - start.cpu_ns);
  const double wall_ms = nanosecondsToMilliseconds(wall_end - wall_start);
  EXPECT_GE(wall_ms, 20.0);
  EXPECT_LT(cpu_ms, wall_ms / 2.0);
  ASSERT_GE(start.voluntary_switches, 0);
  EXPECT_GT(finish.voluntary_switches, start.voluntary_switches);
  EXPECT_GE(finish.involuntary_switches, start.involuntary_switches);
}
