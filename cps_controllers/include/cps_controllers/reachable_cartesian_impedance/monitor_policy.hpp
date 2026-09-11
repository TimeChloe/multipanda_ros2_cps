// Copyright (c) 2026
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cps_controllers::detail {

// A monitor cycle contains one committed period and one fresh intended period.
// Reject incompatible rates instead of silently changing the requested frequency.
inline std::size_t monitorPeriodSteps(double control_dt, double monitor_frequency_hz) {
  if (!std::isfinite(control_dt) || control_dt <= 0.0 ||
      !std::isfinite(monitor_frequency_hz) || monitor_frequency_hz <= 0.0) {
    throw std::invalid_argument("Control step and monitor frequency must be finite and positive");
  }
  const double ratio = 1.0 / control_dt / monitor_frequency_hz;
  const double steps = std::round(ratio);
  if (!std::isfinite(ratio) || steps < 1.0 ||
      steps > static_cast<double>(std::numeric_limits<int>::max()) ||
      std::abs(ratio - steps) > 1e-9 * steps) {
    throw std::invalid_argument("Control frequency must be an integer multiple of monitor frequency");
  }
  return static_cast<std::size_t>(steps);
}

inline bool withinCommittedPrefix(std::uint64_t published_cycle,
                                  std::uint64_t current_cycle,
                                  std::size_t committed_steps) {
  return committed_steps > 0 && current_cycle >= published_cycle &&
         current_cycle - published_cycle <= committed_steps;
}

}  // namespace cps_controllers::detail
