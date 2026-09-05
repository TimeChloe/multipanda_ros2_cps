# CPS MuJoCo Scenarios

`table_assembly_action_server` moves the complete MuJoCo table fixture as one
kinematic assembly. The table, visual spring, and compliant surface retain
their relative transforms and the surface keeps its local spring joint.
The pose loaded from the MuJoCo model is the zero-offset pose. By default, the
accepted absolute Z-offset range is `[-0.5, 0.5]` m.

Start the table scene; the action server starts automatically:

```bash
ros2 launch franka_bringup franka_sim.launch.py scene:=table_spring
```

Move the assembly 0.5 m down over 3 seconds so it disappears below the floor:

```bash
ros2 action send_goal /move_table_assembly \
  cps_mujoco_scenarios/action/MoveTableAssembly \
  "{target_z_offset: -0.5, duration: 3.0}" --feedback
```

Later, move it smoothly back up by 0.5 m over 5 seconds and hold it at the
loaded pose:

```bash
ros2 action send_goal /move_table_assembly \
  cps_mujoco_scenarios/action/MoveTableAssembly \
  "{target_z_offset: 0.0, duration: 5.0}" --feedback
```

Move the complete assembly 0.5 m above its loaded pose:

```bash
ros2 action send_goal /move_table_assembly \
  cps_mujoco_scenarios/action/MoveTableAssembly \
  "{target_z_offset: 0.5, duration: 5.0}" --feedback
```

The target is an absolute offset from the loaded pose, not a relative step.
This makes both commands idempotent and prevents repeated goals from drifting
the fixture. The interpolation is quintic with zero velocity and acceleration
at both endpoints.

## Surface-following human workspace

The scenario also provides a live human-workspace source whose measured hand
center is exactly the MuJoCo `human_hand_surface` body center. Start it after
the `table_spring` scene:

```bash
ros2 launch cps_mujoco_scenarios \
  surface_following_human_workspace.launch.py
```

The provider samples the actual surface body, so table-assembly motion and
local spring deflection are both included. MuJoCo's `GetBodyState` response is
in `world`; the provider also reads the `panda_link0` MuJoCo body pose once and
transforms the surface position into `panda_link0` before publishing. This is
required because this model's robot base is rotated 180 degrees around Z in
MuJoCo world. It publishes `human_workspace/state`, containing the current
surface-center position and differentiated velocity for the reachable
controller.

SaRA `BodyPartCombined` consumes the timestamped position and velocity
observation plus configured maximum velocity/acceleration and measurement
errors. It deliberately does not assume that the future table trajectory is
known. The provider uses `cps_human_workspace`'s same observation adapter as the
static and synthetic-dynamic YAML source. Measured acceleration is checked
locally against the configured limit; `max_acceleration` remains the
conservative SaRA future-motion bound. Run only one publisher for
`human_workspace/state` at a time. This source does not publish a separate
fixed-duration RViz sphere. The launch reuses `cps_human_workspace`'s
independent visualizer: the reachable controller publishes the selected
`HumanReachableSet` data, and that visualizer alone publishes the blue RViz
Marker.
