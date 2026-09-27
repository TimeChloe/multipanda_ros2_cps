#include "cps_human_workspace/human_workspace_message.hpp"

#include <cmath>

namespace cps_human_workspace
{

msg::HumanWorkspace makeHumanWorkspaceMessage(
  const std_msgs::msg::Header & header,
  const Vector3d & center,
  const Vector3d & velocity,
  const HumanWorkspace::Parameters & reachability_parameters)
{
  msg::HumanWorkspace message;
  message.header = header;
  message.sphere_center.x = center.x();
  message.sphere_center.y = center.y();
  message.sphere_center.z = center.z();
  message.center_velocity.x = velocity.x();
  message.center_velocity.y = velocity.y();
  message.center_velocity.z = velocity.z();
  message.motion_radius = reachability_parameters.motion_radius;
  message.hand_max_velocity = reachability_parameters.hand_max_velocity;
  message.hand_max_acceleration = reachability_parameters.hand_max_acceleration;
  message.measurement_error_position =
    reachability_parameters.measurement_error_position;
  message.measurement_error_velocity =
    reachability_parameters.measurement_error_velocity;
  message.measurement_delay = reachability_parameters.measurement_delay;
  return message;
}

bool humanWorkspaceParametersFromMessage(
  const msg::HumanWorkspace & message,
  HumanWorkspace::Parameters * parameters)
{
  if (parameters == nullptr) {
    return false;
  }

  HumanWorkspace::Parameters parsed;
  parsed.sphere_center = Vector3d(
    message.sphere_center.x,
    message.sphere_center.y,
    message.sphere_center.z);
  parsed.center_velocity = Vector3d(
    message.center_velocity.x,
    message.center_velocity.y,
    message.center_velocity.z);
  parsed.center_sinusoid_amplitude.setZero();
  parsed.center_sinusoid_frequency_hz = 0.0;
  parsed.center_sinusoid_phase_rad = 0.0;
  parsed.center_motion_time_offset_sec = 0.0;
  parsed.motion_radius = message.motion_radius;
  parsed.hand_max_velocity = message.hand_max_velocity;
  parsed.hand_max_acceleration = message.hand_max_acceleration;
  parsed.measurement_error_position = message.measurement_error_position;
  parsed.measurement_error_velocity = message.measurement_error_velocity;
  parsed.measurement_delay = message.measurement_delay;

  const bool valid =
    parsed.sphere_center.allFinite() &&
    parsed.center_velocity.allFinite() &&
    std::isfinite(parsed.motion_radius) && parsed.motion_radius >= 0.0 &&
    std::isfinite(parsed.hand_max_velocity) &&
    parsed.hand_max_velocity >= 0.0 &&
    std::isfinite(parsed.hand_max_acceleration) &&
    parsed.hand_max_acceleration > 0.0 &&
    std::isfinite(parsed.measurement_error_position) &&
    parsed.measurement_error_position >= 0.0 &&
    std::isfinite(parsed.measurement_error_velocity) &&
    parsed.measurement_error_velocity >= 0.0 &&
    std::isfinite(parsed.measurement_delay) && parsed.measurement_delay >= 0.0;
  if (!valid) {
    return false;
  }

  *parameters = parsed;
  return true;
}

}  // namespace cps_human_workspace
