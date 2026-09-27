// Private timing helpers for the reachable Cartesian impedance implementation.
#pragma once

#include <chrono>
#include <cstdint>
#include <sys/resource.h>
#include <time.h>

namespace cps_controllers::detail
{

using SteadyClock = std::chrono::steady_clock;

inline std::int64_t steadyNowNanoseconds()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    SteadyClock::now().time_since_epoch())
         .count();
}

inline double nanosecondsToMilliseconds(std::int64_t nanoseconds)
{
  return static_cast<double>(nanoseconds) * 1.0e-6;
}

// Read only on the monitor worker, never on the servo thread. Elapsed wall
// time includes preemption and blocking; this clock counts the worker's CPU
// execution only. Failed OS measurements stay explicitly invalid.
struct WorkerCpuSample
{
  std::int64_t cpu_ns{-1};
  std::int64_t voluntary_switches{-1};
  std::int64_t involuntary_switches{-1};
};

inline WorkerCpuSample workerCpuSample()
{
  WorkerCpuSample sample;
  timespec time{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0) {
    sample.cpu_ns = static_cast<std::int64_t>(time.tv_sec) * 1000000000 + time.tv_nsec;
  }
  rusage usage{};
  if (getrusage(RUSAGE_THREAD, &usage) == 0) {
    sample.voluntary_switches = usage.ru_nvcsw;
    sample.involuntary_switches = usage.ru_nivcsw;
  }
  return sample;
}

}  // namespace cps_controllers::detail
