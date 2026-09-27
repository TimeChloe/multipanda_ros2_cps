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
sends `/move_table_assembly` to lower the complete table, spring, and
hand-surface assembly 0.5 m below the floor:

```bash
python3 tools/static_down_collision_disappear_test.py
```

Restore the complete assembly later with a second action goal. It rises 0.5 m
over 5 seconds and then holds its original pose:

```bash
ros2 action send_goal /move_table_assembly \
  cps_mujoco_scenarios/action/MoveTableAssembly \
  "{target_z_offset: 0.0, duration: 5.0}" --feedback
```
