# Dynamic Human Moving Scenario

This example keeps the reachable Cartesian impedance controller unchanged and
makes the human reachable set move instead.

The SaRA-Shield reference computes robot and human reachable capsules over
time intervals, then checks interval collisions. In this repo the monitor
already uses a simpler end-effector collision-center versus human sphere model,
so the matching approach is a time-indexed sphere center:

- human reachable set:
  `cps_human_workspace/config/human_workspace_dynamic_crossing.yaml`
- robot via-point action:
  `cps_trajectory_generators/config/examples/reachable_via_points_dynamic_human_crossing_action_goal.yaml`
- robot via-point topic payload:
  `cps_trajectory_generators/config/examples/reachable_via_points_dynamic_human_crossing.yaml`

Start the simulator separately:

```bash
ros2 launch franka_bringup franka_sim.launch.py
```

Start the dynamic human workspace visualization separately:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_config:=human_workspace_dynamic_crossing.yaml
```

Load the controller. It subscribes to `human_workspace/state`, so it can start
before or after the workspace provider:

```bash
ros2 control load_controller --set-state active reachable_cartesian_impedance_controller
```

Then send the path:

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_dynamic_human_crossing_action_goal.yaml)"
```

For a different workspace, create a config-only ROS package that installs its
YAML files under `config/`, then pass `human_workspace_package` and
`human_workspace_config` to the standalone human workspace launch:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_package:=my_lab_workspace \
  human_workspace_config:=table_corner.yaml
```

The validation CSV includes `human_center_px`, `human_center_py`, and
`human_center_pz`. The prediction CSV includes the actual human center plus
the start and end human centers for each monitored interval, so
`distance_segment` and `contact_possible` can be checked against the moving
reachable set along the path.
