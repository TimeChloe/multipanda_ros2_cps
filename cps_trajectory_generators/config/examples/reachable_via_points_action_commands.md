# Cartesian Via-Point Action Commands

These files are `panda_motion_generator_msgs/action/CartesianViaMotion` goal
payloads. Use them with the reachable Cartesian impedance controller action:

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_line_action_goal.yaml)"
```

## Available Goals

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_line_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_loop_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_rotate_in_place_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_move_and_rotate_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_table_corner_front_left_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_table_corner_front_right_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_table_corner_back_left_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_table_corner_back_right_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_table_center_high_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_dynamic_human_crossing_action_goal.yaml)"
```

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_static_down_collision_action_goal.yaml)"
```

## Static Down Collision Disappear Test

Launch the original static human workspace provider first:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_config:=human_workspace.yaml \
  use_sim_time:=true
```

Then run this helper. It unpauses MuJoCo, sends the straight-down goal above,
watches `/panda_metal_ball_touch`, waits 3 seconds after first contact, then
uses `/set_geom_properties` to shrink the table, spring, and hand-surface geoms
so they disappear without reloading the MuJoCo model:

```bash
python3 tools/static_down_collision_disappear_test.py
```
