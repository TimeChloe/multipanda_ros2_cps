# Unified end-effector tool format

Put one `tool.yaml` beside its visual and collision meshes. Relative mesh paths
are resolved from the directory containing the YAML, so the complete directory
can be moved or uploaded as one unit.

```text
my_tool/
├── tool.yaml
└── meshes/
    ├── visual.stl
    └── collision.stl
```

Use `metal_ball.yaml` as the schema-version-1 example. For a mesh geometry:

```yaml
visual:
  type: mesh
  mesh: meshes/visual.stl
  scale: [1.0, 1.0, 1.0]
  rgba: [0.7, 0.7, 0.7, 1.0]

collision:
  type: mesh
  mesh: meshes/collision.stl
  scale: [1.0, 1.0, 1.0]
```

The YAML is also the authoritative source for the mounting pose, TCP, mass,
center of mass, inertia at the center of mass, conservative safety sphere and
MuJoCo contact settings. Do not infer inertial values from a visual mesh; enter
values from CAD or an identified payload model.

Schema version 1 supports one rigid tool fixed to `panda_link8`, one safety
bounding sphere, and a TCP translated from `panda_link8` without a relative
rotation. The TCP coincides with the safety sphere center. Omit `tcp` to derive
it from `mount` and `safety.bounding_sphere.center`; an explicit different TCP
is rejected. For a spherical tool this is also the physical ball center.
Visual and collision geometry may independently be a sphere, box,
cylinder or mesh.

For `metal_ball.yaml`, edit `ball_radius` once: its YAML anchor supplies the
visual, physical and safety radii. `inertial.inertia: solid_sphere` derives the
inertia from the configured mass and radius. Mesh/box/cylinder tools must use
an explicit inertia mapping instead. The default ball-center TCP is 0.03 m
along `panda_link8`'s local Z axis. It was a surface point at 0.06 m before v14;
old contact goals need conversion to ball-center coordinates.

The reachable controller appends the configured TCP sphere to the seven SaRA
arm occupancies. Radius changes therefore affect tool reachability as well as
MuJoCo collision geometry. Re-launch to regenerate models after changing the
YAML. See [TCP behavior and migration](../../../docs/experiments/tcp_ball_center.md).

Validate a file before launching:

```bash
ros2 run franka_bringup tool_model_cli /absolute/path/to/tool.yaml
```

Simulation uses one YAML to generate the robot-description URDF, Pinocchio
monitor URDF, MuJoCo MJCF and controller geometry:

```bash
ros2 launch franka_bringup franka_sim.launch.py \
  scene:=no_table tool_config:=/absolute/path/to/tool.yaml
```

For a real robot, the validator compares the first available libfranka load
against the YAML once and never writes robot parameters:

```bash
ros2 launch franka_bringup franka.launch.py \
  robot_ip:=ROBOT_IP tool_config:=/absolute/path/to/tool.yaml
```

If the mass, center of mass or inertia differs beyond its configured tolerance,
or no Franka state arrives before the startup deadline, the complete real-robot
launch is stopped. Correct the payload configuration outside this launch and
restart it; there is no automatic `set_load` path. After a successful check, the
validator cancels its timer and state subscription and remains idle for the rest
of the launch.
