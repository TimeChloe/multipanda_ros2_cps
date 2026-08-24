// Private timing helpers for the reachable Cartesian impedance implementation.
#pragma once

#include <chrono>
#include <cstdint>

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

}  // namespace cps_controllers::detail
