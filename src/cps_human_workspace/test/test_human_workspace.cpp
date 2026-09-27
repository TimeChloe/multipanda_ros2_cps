#include <gtest/gtest.h>

#include "cps_human_workspace/human_workspace.hpp"

namespace cps_human_workspace
{
namespace
{

TEST(HumanWorkspaceReachability, ZeroPredictionKeepsPhysicalHandSphere) {
  HumanWorkspace workspace;
  HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d(1.0, 2.0, 3.0);
  parameters.center_velocity.setZero();
  parameters.center_motion_time_offset_sec = 1.0;
  parameters.motion_radius = 0.103;
  workspace.setParameters(parameters);

  const auto reach = workspace.handReachableSetAtTime(1.0);
  EXPECT_TRUE(reach.center.isApprox(parameters.sphere_center, 1.0e-12));
  EXPECT_DOUBLE_EQ(reach.radius, 0.103);
}

TEST(HumanWorkspaceReachability, ConfiguredImmobileHandStaysFixedForEveryHorizon) {
  HumanWorkspace workspace;
  HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d(1.0, 2.0, 3.0);
  parameters.center_velocity.setZero();
  parameters.center_motion_time_offset_sec = 1.0;
  parameters.motion_radius = 0.103;
  parameters.hand_max_velocity = 0.0;
  parameters.hand_max_acceleration = 50.0;
  parameters.measurement_error_position = 0.0;
  parameters.measurement_error_velocity = 0.0;
  workspace.setParameters(parameters);

  for (const double prediction_time : {0.0, 0.005, 0.1, 1.0, 10.0}) {
    const auto reach =
      workspace.handReachableSetAtTime(1.0 + prediction_time);
    EXPECT_TRUE(reach.center.isApprox(parameters.sphere_center, 1.0e-12));
    EXPECT_DOUBLE_EQ(reach.radius, parameters.motion_radius);
  }
}

TEST(HumanWorkspaceReachability, CombinedZeroVelocityMatchesSaraReachLib) {
  HumanWorkspace workspace;
  HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d(1.0, 2.0, 3.0);
  parameters.center_velocity.setZero();
  parameters.center_motion_time_offset_sec = 1.0;
  parameters.motion_radius = 0.103;
  parameters.hand_max_velocity = 2.0;
  parameters.hand_max_acceleration = 50.0;
  parameters.measurement_error_position = 0.01;
  parameters.measurement_error_velocity = 0.1;
  workspace.setParameters(parameters);

  const double prediction_time = 0.2;
  const auto reach =
    workspace.handReachableSetAtTime(1.0 + prediction_time);
  const double time_to_max_velocity =
    parameters.hand_max_velocity / parameters.hand_max_acceleration;
  const double expected_motion_radius =
    parameters.hand_max_acceleration *
    (prediction_time * time_to_max_velocity -
    0.5 * time_to_max_velocity * time_to_max_velocity);
  const double expected_radius =
    parameters.motion_radius + expected_motion_radius +
    parameters.measurement_error_position +
    parameters.measurement_error_velocity * prediction_time;

  EXPECT_TRUE(reach.center.isApprox(parameters.sphere_center, 1.0e-12));
  EXPECT_NEAR(reach.radius, expected_radius, 1.0e-12);
}

TEST(HumanWorkspaceReachability, CombinedMovingObservationMatchesSaraReachLib) {
  HumanWorkspace workspace;
  HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d(1.0, 2.0, 3.0);
  parameters.center_velocity = Vector3d(0.5, 0.0, 0.0);
  parameters.center_motion_time_offset_sec = 0.0;
  parameters.motion_radius = 0.103;
  parameters.hand_max_velocity = 2.0;
  parameters.hand_max_acceleration = 50.0;
  parameters.measurement_error_position = 0.0;
  parameters.measurement_error_velocity = 0.1;
  workspace.setParameters(parameters);

  const double prediction_time = 0.005;
  const auto reach = workspace.handReachableSetAtTime(prediction_time);
  const double expected_center_shift =
    prediction_time * parameters.center_velocity.norm();
  const double expected_motion_radius =
    0.5 * parameters.hand_max_acceleration *
    prediction_time * prediction_time;
  const double expected_radius =
    parameters.motion_radius + expected_motion_radius +
    parameters.measurement_error_velocity * prediction_time;

  EXPECT_TRUE(
    reach.center.isApprox(
      parameters.sphere_center +
      Vector3d(expected_center_shift, 0.0, 0.0),
      1.0e-12));
  EXPECT_NEAR(reach.radius, expected_radius, 1.0e-12);
}

TEST(HumanWorkspaceReachability, CombinedModelIncludesConfiguredDelay) {
  HumanWorkspace workspace;
  HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d::Zero();
  parameters.center_velocity.setZero();
  parameters.center_motion_time_offset_sec = 4.0;
  parameters.motion_radius = 0.05;
  parameters.hand_max_velocity = 2.0;
  parameters.hand_max_acceleration = 10.0;
  parameters.measurement_delay = 0.1;
  workspace.setParameters(parameters);

  const auto reach = workspace.handReachableSetAtTime(4.0);
  EXPECT_NEAR(
    reach.radius, 0.05 + 0.5 * 10.0 * 0.1 * 0.1,
    1.0e-12);
}

TEST(HumanWorkspaceReachability, IntervalDistanceUsesReachableHandBall) {
  HumanWorkspace workspace;
  HumanWorkspace::Parameters parameters;
  parameters.sphere_center = Vector3d::Zero();
  parameters.center_velocity.setZero();
  parameters.motion_radius = 0.01;
  parameters.hand_max_velocity = 2.0;
  parameters.hand_max_acceleration = 10.0;
  workspace.setParameters(parameters);

  const Vector3d robot_point(0.055, 0.0, 0.0);
  const double distance = workspace.signedDistanceSegmentToInflatedSphere(
    robot_point,
    robot_point,
    workspace.inflatedCollisionRadius(0.0, 0.0),
    0.1);

  // At 0.1 s, r = 0.01 m hand radius + 0.5*a*t^2 = 0.06 m.
  EXPECT_NEAR(distance, -0.005, 1.0e-12);
}

}  // namespace
}  // namespace cps_human_workspace
