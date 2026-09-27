#include <limits>

#include <gtest/gtest.h>

#include "cps_human_workspace/human_workspace_message.hpp"

namespace
{

using cps_human_workspace::HumanWorkspace;
using cps_human_workspace::Vector3d;

HumanWorkspace::Parameters validParameters()
{
  HumanWorkspace::Parameters parameters;
  parameters.motion_radius = 0.103;
  parameters.hand_max_velocity = 2.0;
  parameters.hand_max_acceleration = 50.0;
  parameters.measurement_error_position = 0.01;
  parameters.measurement_error_velocity = 0.1;
  parameters.measurement_delay = 0.02;
  return parameters;
}

TEST(HumanWorkspaceMessage, RoundTripUsesOneObservationContract)
{
  std_msgs::msg::Header header;
  header.stamp.sec = 12;
  header.stamp.nanosec = 345;
  header.frame_id = "panda_link0";
  const Vector3d center(0.3, -0.2, 0.4);
  const Vector3d velocity(0.1, 0.2, -0.1);
  const auto source = validParameters();

  const auto message = cps_human_workspace::makeHumanWorkspaceMessage(
    header, center, velocity, source);
  HumanWorkspace::Parameters parsed;
  ASSERT_TRUE(
    cps_human_workspace::humanWorkspaceParametersFromMessage(
      message, &parsed));

  EXPECT_EQ(message.header.frame_id, header.frame_id);
  EXPECT_TRUE(parsed.sphere_center.isApprox(center, 1.0e-12));
  EXPECT_TRUE(parsed.center_velocity.isApprox(velocity, 1.0e-12));
  EXPECT_DOUBLE_EQ(parsed.motion_radius, source.motion_radius);
  EXPECT_DOUBLE_EQ(parsed.hand_max_velocity, source.hand_max_velocity);
  EXPECT_DOUBLE_EQ(parsed.hand_max_acceleration, source.hand_max_acceleration);
  EXPECT_DOUBLE_EQ(
    parsed.measurement_error_position,
    source.measurement_error_position);
  EXPECT_DOUBLE_EQ(
    parsed.measurement_error_velocity,
    source.measurement_error_velocity);
  EXPECT_DOUBLE_EQ(parsed.measurement_delay, source.measurement_delay);
  EXPECT_TRUE(parsed.center_sinusoid_amplitude.isZero());
  EXPECT_DOUBLE_EQ(parsed.center_sinusoid_frequency_hz, 0.0);
  EXPECT_DOUBLE_EQ(parsed.center_motion_time_offset_sec, 0.0);
}

TEST(HumanWorkspaceMessage, ParsedObservationPreservesCombinedReachability)
{
  std_msgs::msg::Header header;
  const Vector3d center(0.4, -0.1, 0.3);
  const Vector3d velocity(0.2, 0.05, 0.0);
  auto expected_parameters = validParameters();
  expected_parameters.sphere_center = center;
  expected_parameters.center_velocity = velocity;
  expected_parameters.center_sinusoid_amplitude.setZero();
  expected_parameters.center_sinusoid_frequency_hz = 0.0;
  expected_parameters.center_sinusoid_phase_rad = 0.0;
  expected_parameters.center_motion_time_offset_sec = 0.0;

  const auto message = cps_human_workspace::makeHumanWorkspaceMessage(
    header, center, velocity, expected_parameters);
  HumanWorkspace::Parameters parsed_parameters;
  ASSERT_TRUE(
    cps_human_workspace::humanWorkspaceParametersFromMessage(
      message, &parsed_parameters));

  HumanWorkspace expected_workspace;
  expected_workspace.setParameters(expected_parameters);
  HumanWorkspace parsed_workspace;
  parsed_workspace.setParameters(parsed_parameters);
  for (const double prediction_time : {0.0, 0.005, 0.1, 0.5}) {
    const auto expected =
      expected_workspace.handReachableSetAtTime(prediction_time);
    const auto parsed =
      parsed_workspace.handReachableSetAtTime(prediction_time);
    EXPECT_TRUE(parsed.center.isApprox(expected.center, 1.0e-12));
    EXPECT_NEAR(parsed.radius, expected.radius, 1.0e-12);
  }
}

TEST(HumanWorkspaceMessage, RejectsInvalidObservationForEveryProvider)
{
  std_msgs::msg::Header header;
  const Vector3d center = Vector3d::Zero();
  const Vector3d velocity = Vector3d::Zero();
  auto message = cps_human_workspace::makeHumanWorkspaceMessage(
    header, center, velocity, validParameters());
  HumanWorkspace::Parameters parsed;

  message.motion_radius = -1.0;
  EXPECT_FALSE(
    cps_human_workspace::humanWorkspaceParametersFromMessage(
      message, &parsed));

  message = cps_human_workspace::makeHumanWorkspaceMessage(
    header, center, velocity, validParameters());
  message.center_velocity.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
    cps_human_workspace::humanWorkspaceParametersFromMessage(
      message, &parsed));
  EXPECT_FALSE(
    cps_human_workspace::humanWorkspaceParametersFromMessage(
      message, nullptr));
}

}  // namespace
