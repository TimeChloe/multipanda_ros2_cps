#pragma once

namespace cps_controllers::panda_limits {

// Matches libfranka::kMaxTorqueRate for Panda joint torque commands.
inline constexpr double kTorqueRateLimit = 1000.0;  // Nm/s

}  // namespace cps_controllers::panda_limits
