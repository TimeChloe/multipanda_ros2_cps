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

Run the controller with the dynamic workspace config before the controller is
configured or activated, for example by setting the reachable controller
parameter in the controller YAML or with a parameter override:

```bash
ros2 param set /reachable_cartesian_impedance_controller human_workspace_config_path \
  "$(ros2 pkg prefix cps_human_workspace)/share/cps_human_workspace/config/human_workspace_dynamic_crossing.yaml"
```

If the controller was already active, deactivate/configure/activate it again
after changing this parameter. Then send the path:

```bash
ros2 action send_goal /reachable_cartesian_impedance_controller/follow_cartesian_via_points \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_dynamic_human_crossing_action_goal.yaml)"
```

For RViz-only visualization of the moving human workspace:

```bash
ros2 run cps_human_workspace human_workspace_visualizer --ros-args \
  -p human_workspace_config_path:="$(ros2 pkg prefix cps_human_workspace)/share/cps_human_workspace/config/human_workspace_dynamic_crossing.yaml"
```

The validation CSV includes `human_center_px`, `human_center_py`, and
`human_center_pz`. The prediction CSV includes the actual human center plus
the start and end human centers for each monitored interval, so
`distance_segment` and `contact_possible` can be checked against the moving
reachable set along the path.
