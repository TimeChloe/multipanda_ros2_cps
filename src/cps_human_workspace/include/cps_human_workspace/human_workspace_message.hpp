#pragma once

#include <std_msgs/msg/header.hpp>

#include "cps_human_workspace/human_workspace.hpp"
#include "cps_human_workspace/msg/human_workspace.hpp"

namespace cps_human_workspace
{

// Convert any measured hand observation into the single message contract used
// by static, synthetic-dynamic, and sensor-backed workspace providers.
msg::HumanWorkspace makeHumanWorkspaceMessage(
  const std_msgs::msg::Header & header,
  const Vector3d & center,
  const Vector3d & velocity,
  const HumanWorkspace::Parameters & reachability_parameters);

// Parse and validate the common observation contract. Motion-source fields are
// reset because a message is already a timestamped position/velocity sample;
// future motion must be generated conservatively by BodyPartCombined.
bool humanWorkspaceParametersFromMessage(
  const msg::HumanWorkspace & message,
  HumanWorkspace::Parameters * parameters);

}  // namespace cps_human_workspace
