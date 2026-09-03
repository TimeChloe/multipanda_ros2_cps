# CPS Human Workspace

This package owns the reusable human-workspace model used by the reachable
Cartesian controller and safety monitor.

The intended split is:

- `cps_human_workspace`: C++ model, YAML parser, RViz visualizer, and default
  example configs.
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

## Start The Visualizer

Use the built-in configs:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_config:=human_workspace_dynamic_crossing.yaml
```

Markers expire automatically after `marker_lifetime_sec` seconds once the
visualizer stops publishing. Set `marker_lifetime_sec:=0.0` if you want RViz to
keep the last marker forever.

The `Human Hand Reachable Set` RViz display shows only the cyan preview of one
SaRA BodyPartCombined interval. It intentionally contains no separate physical-hand
sphere or text label. SaRA's Panda trajectory configuration uses
`reachability_set_interval_size: 5`; with this project's `1 ms` sample time,
the interval is therefore `5 ms`:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_config:=human_workspace_dynamic_crossing.yaml \
  reachability_set_interval_sec:=0.005
```

The preview is not the complete braking horizon. The controller evaluates
successive 5 ms intervals over its complete intended + failsafe trajectory.
The `SaRA Robot Reachable Sets` RViz display publishes only the selected robot
interval. Human markers remain owned by the `Human Hand Reachable Set` display;
the safety monitor is the layer that intersects the independently generated
robot and human occupancies.

Use a config from another ROS package:

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_package:=my_lab_workspace \
  human_workspace_config:=table_corner.yaml
```

The default EE collision marker is centered on `panda_metal_ball_link`, which
is the metal ball center in the simulated Panda model. `ee_collision_radius`
sets the radius expanded around that center. If you visualize from
`panda_link8` instead, pass `ee_frame_id:=panda_link8` and
`ee_collision_center_offset_z:=0.03`.

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
