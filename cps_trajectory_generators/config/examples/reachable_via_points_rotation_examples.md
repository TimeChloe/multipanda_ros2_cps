# Rotation Via Point Examples

These examples are `geometry_msgs/msg/PoseArray` payloads. They use robot base
frame (`panda_link0`) poses with one pose per via point.

They are based on the default sim activation TCP pose:

- position: `[0.306957, 0.000000, 0.699912]`
- orientation: `[0.923956, -0.382499, 0.000000, 0.000000]`

## Examples

- `reachable_via_points_rotate_in_place_example.yaml`: keeps the TCP position
  fixed and tilts the TCP orientation, then returns to the default orientation.
- `reachable_via_points_move_and_rotate_example.yaml`: moves through a short
  descending path while changing the TCP orientation.

Send an example action goal to start motion:

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_rotate_in_place_action_goal.yaml)"
```

The same poses are also available as `PoseArray` payloads for the debug topic:

```bash
ros2 topic pub --once /cartesian_via_points geometry_msgs/msg/PoseArray "$(cat cps_trajectory_generators/config/examples/reachable_via_points_rotate_in_place_example.yaml)"
```

Quaternion entries use ROS message order `{x, y, z, w}`.
