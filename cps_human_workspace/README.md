# CPS Human Workspace

This package owns the reusable human-workspace model used by the reachable
Cartesian controller and safety monitor.

The intended split is:

- `cps_human_workspace`: C++ BodyPartCombined model, YAML parser, common ROS
  observation adapter, observation publisher, human reachable-set visualizer,
  and default example configs.
- Scenario or lab packages: config-only ROS packages that install one or more
  human workspace YAML files under `config/`.

This follows the same separation used in the SaRA-Shield reference: reusable
reachability logic is kept apart from human, robot, and scenario config files.

## Config Format

```yaml
sphere_center: [0.3, 0.0, 0.25]
motion_radius: 0.103

hand_reachability:
  max_velocity: 2.0
  max_acceleration: 50.0
  measurement_error_position: 0.0
  measurement_error_velocity: 0.1
  delay: 0.0

center_motion:
  velocity: [0.0, 0.0, 0.0]
  sinusoid_amplitude: [0.0, 0.16, 0.0]
  sinusoid_frequency_hz: 0.16
  sinusoid_phase_rad: -1.57079632679
  time_offset_sec: 0.0
```

The package has one human reachability model: a single-hand specialization of
SaRA ReachLib's `BodyPartCombined`. `motion_radius` is the physical hand
enclosure radius; bounded future motion and measurement uncertainty are added
for every monitor interval. The dynamic example above follows the TUM
SaRA-Shield hand values: 0.206 m thickness, 2 m/s maximum speed, 50 m/s^2
maximum acceleration, and 0.1 m/s velocity uncertainty. There is no model
selector.

The default `human_workspace.yaml` is instead an explicit stationary-hand
assumption. It uses zero maximum velocity and zero velocity measurement error,
so the same BodyPartCombined generator returns the fixed physical hand sphere
for every prediction horizon. Use that assumption only when the hand is known
not to move.

`center_motion` is only a repeatable synthetic measurement source. The live
message carries the current hand position and velocity; the controller does
not assume that the configured future sinusoid is known.

## Live Workspace State Topic

Runtime providers publish `cps_human_workspace/msg/HumanWorkspace` on
`human_workspace/state`. For camera or Vicon input, publish the measured hand
position and velocity in the robot base frame, its physical enclosure radius,
and the reachability limits/error bounds. The controller treats each message
as a timestamped observation and generates a SaRA BodyPartCombined reachable ball for
each monitor interval until `human_workspace_timeout_sec` expires. A zero or
invalid timestamp falls back to receipt time.

All providers use the public `makeHumanWorkspaceMessage()` adapter. The
controller uses the matching `humanWorkspaceParametersFromMessage()` parser,
which applies one validation policy and removes any source-specific future
motion. Consequently the static YAML source, synthetic crossing source, and
MuJoCo surface source enter exactly the same BodyPartCombined calculation:

```text
static/dynamic YAML ─┐
                    ├─ common HumanWorkspace message ─ BodyPartCombined ─ monitor
MuJoCo observation ─┘                                      │
                                                           └─ HumanReachableSet
                                                                      │
                                  cps_human_workspace visualizer ─────┘
                                                │
                                                └─ blue RViz sphere
```

There is no provider-specific reachable-set marker or calculation. The source
only determines the latest measured center and velocity plus the configured
physical/error bounds.

## Start An Observation Provider

Use the built-in configs:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_config:=human_workspace_dynamic_crossing.yaml
```

This launch starts two separate nodes. `human_workspace_publisher` publishes
only timestamped hand observations. The reachable controller evaluates
successive 5 ms robot/human intervals over the complete intended + failsafe
trajectory, publishes its robot markers, and publishes the selected human
occupancy as `cps_human_workspace/msg/HumanReachableSet`. The independent
`human_reachable_set_visualizer` in this package converts that result into the
blue RViz sphere on `/human_workspace/markers`; no human Marker is constructed
inside the controller.

The selected interval remains synchronized with the robot display: the final
interval for a clear trajectory, the first contact interval for an energy-safe
contact, or the first energy-unsafe contact interval. The visualizer does not
run another reachability model and therefore cannot diverge from the sphere
published by the controller.

The robot and human generators remain independent, as in SaRA: RobotArmReach
builds robot capsules from joint states, while BodyPartCombined builds the
human sphere from the timestamped hand observation. Sharing an interval for
calculation and visualization does not mix the two models.

To start only the presentation node, for example when the observation provider
is launched by a scenario package:

```bash
ros2 launch cps_human_workspace human_reachable_set_visualizer.launch.py
```

Its input is `/human_workspace/reachable_set`, and its only output is the
`/human_workspace/markers` MarkerArray consumed by RViz.

Use a config from another ROS package:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_package:=my_lab_workspace \
  human_workspace_config:=table_corner.yaml
```

## Create A New Workspace Package

A config-only package only needs to install its `config/` directory.

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_lab_workspace)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
```

Start this package separately from the robot simulator:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_package:=my_lab_workspace \
  human_workspace_config:=table_corner.yaml
```

The Franka simulator launch intentionally does not start or configure the human
workspace. For the current config-file based controller, set
`human_workspace_config_path` before configuring or activating
`reachable_cartesian_impedance_controller`. If that parameter is empty, the
controller waits for live workspace states on `human_workspace/state`.

## Controller Startup With Current Config Files

Start the simulator in one terminal:

```bash
ros2 launch franka_bringup franka_sim.launch.py
```

Start the controller. It can start before the human workspace provider; it will
warn until messages arrive on `human_workspace/state`.

```bash
ros2 control load_controller --set-state active reachable_cartesian_impedance_controller
```

Then start a workspace provider, for example the dynamic config-backed provider:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_config:=human_workspace_dynamic_crossing.yaml
```

For the static config as a startup-only controller source, load the controller
first, set its config parameter, then configure and activate it:

```bash
ros2 control load_controller reachable_cartesian_impedance_controller
ros2 param set /reachable_cartesian_impedance_controller human_workspace_config_path \
  "$(ros2 pkg prefix cps_human_workspace)/share/cps_human_workspace/config/human_workspace.yaml"
ros2 control set_controller_state reachable_cartesian_impedance_controller inactive
ros2 control set_controller_state reachable_cartesian_impedance_controller active
```
