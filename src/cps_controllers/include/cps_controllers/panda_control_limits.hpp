#pragma once

#include <array>

namespace cps_controllers::panda_limits {

// Matches libfranka::kMaxTorqueRate for Panda joint torque commands.
inline constexpr double kTorqueRateLimit = 1000.0;  // Nm/s

// Limits used by the MuJoCo Panda model in this workspace.  Keeping these in
// one place makes the asynchronous joint rollout agree with the plant instead
// of relying on the slightly wider generic URDF safety range for joint 4.
inline constexpr std::array<double, 7> kPositionLower = {
    -2.8973, -1.7628, -2.8973, -3.0421, -2.8973, -0.0175, -2.8973};
inline constexpr std::array<double, 7> kPositionUpper = {
    2.8973, 1.7628, 2.8973, -0.1518, 2.8973, 3.7525, 2.8973};
inline constexpr std::array<double, 7> kVelocity = {
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100};
inline constexpr std::array<double, 7> kAcceleration = {
    15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0};
inline constexpr std::array<double, 7> kTorque = {
    87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0};

}  // namespace cps_controllers::panda_limits
