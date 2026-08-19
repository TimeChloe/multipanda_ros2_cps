# Table Corner Approach Via Point Scenarios

These scenarios are intentionally mild. Each path first moves to an upper
corner in the robot base frame, then approaches the table/contact area with a
smooth inward descent.

All examples are `geometry_msgs/msg/PoseArray` payloads and use:

- robot base frame (`panda_link0`) coordinates
- one pose per via point, with position and quaternion orientation
- the default sim activation orientation
  `[0.923956, -0.382499, 0.000000, 0.000000]`
- a final approach point near `[0.306957, 0.000000, 0.279912]`

Start MuJoCo with the table, visual spring, and compliant hand-surface pad:

```bash
ros2 launch franka_bringup franka_sim.launch.py scene:=table_spring
```

The backward-compatible robot-and-floor scene remains available with
`scene:=no_table`, which is also the default.

## Scenarios

- `reachable_via_points_table_corner_front_left.yaml`
- `reachable_via_points_table_corner_front_right.yaml`
- `reachable_via_points_table_corner_back_left.yaml`
- `reachable_via_points_table_corner_back_right.yaml`
- `reachable_via_points_table_center_high.yaml`

Send a scenario action goal to start motion:

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_table_corner_front_left_action_goal.yaml)"
```

The same poses are also available as `PoseArray` payloads for the debug topic:

```bash
ros2 topic pub --once /cartesian_via_points geometry_msgs/msg/PoseArray "$(cat cps_trajectory_generators/config/examples/reachable_via_points_table_corner_front_left.yaml)"
```

Quaternion entries use ROS message order `{x, y, z, w}`.
