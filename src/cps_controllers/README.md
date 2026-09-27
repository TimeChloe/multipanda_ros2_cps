# Reachable Cartesian impedance controller

The controller uses fixed-period asynchronous monitoring. Configure the monitor
frequency in `cps_trajectory_generators/config/reachable_cartesian_trajectory.yaml`:

```yaml
monitor_frequency_hz: 100.0
local_replan_dt: 0.001
```

The command grid must match the configured control update rate. Every monitor
period has exactly

```
N = control_frequency / monitor_frequency
  = 1 / (local_replan_dt * monitor_frequency_hz)
```

control commands. Both the committed prefix and the fresh intended segment use
this same N. At 1 kHz control, 100 Hz monitoring gives 10/10; 200 Hz gives 5/5.
Nonpositive, nonfinite or nonintegral ratios are configuration errors; the
controller never silently rounds a requested frequency. When the ROS controller
interface exposes a nonzero configured update rate, activation also checks that
it agrees with the command grid. Wall-time slowdown of the simulator does not
change N.

## Execution rule

1. Consume any completed monitor result before publishing another request.
2. Accept a verified candidate only while its committed prefix has not expired
   and its source plan, recovery state/epoch, workspace policy and calibration
   target still match. The last committed control boundary is inclusive; later
   results are rejected, even if the candidate has unused intended commands.
3. Every N control cycles, copy the next N commands of the existing verified
   stream. The prefix can include verified braking or terminal hold; it is not
   shortened at the intended/braking boundary.
4. The worker generates exactly N fresh intended commands from the committed
   endpoint, appends a complete braking continuation and verifies the candidate.
   New intended commands use nominal gains; the existing runtime energy law
   applies measured-state gain adaptation. There is no separate gain-ramp
   horizon. At the path end, terminal samples still fill the complete period.
5. The control thread advances the verified stream by one command per update.
   An absent, rejected or late result does not authorize fresh motion. The
   existing stream continues through its remaining intended commands, braking
   and terminal hold. Without an initial verified plan, startup holds position.

Only the control thread owns the accepted plan and its execution index. The
worker owns no parallel execution cache. Accepted results are aligned to the
already elapsed control steps, so committed commands are never replayed.

The single-outstanding-request gate and accept-before-publish order prevent a
new request from referring to a plan replaced later in the same control cycle.
Those protocol invariants remain necessary even though late catch-up is gone.

## Source layout

| File | Responsibility |
|---|---|
| `src/reachable_cartesian_impedance_controller.cpp` | Path-consistent candidate generation, verification calls, verified command execution, runtime energy law, impedance torque and control update |
| `include/cps_controllers/reachable_cartesian_impedance/monitor_policy.hpp` | Equal-period derivation and strict committed-prefix deadline |
| `src/reachable_cartesian_impedance/monitor_execution.cpp` | Result acceptance, request snapshots, calibration execution and command selection |
| `src/reachable_cartesian_impedance/monitor_worker.cpp` | Worker thread lifecycle, CPU scheduling and mailbox transport |
| `src/reachable_cartesian_impedance/commands.cpp` | ROS actions, via-point commands and observation/contact callbacks |
| `src/reachable_cartesian_impedance/lifecycle.cpp` | ROS configuration, dynamics-provider adapters, activation and recording setup |
| `src/reachable_cartesian_impedance/logging.cpp` | Bounded log capture, serialization, timing summaries and diagnostic worker |
| `src/reachable_cartesian_impedance/visualization.cpp` | Reachable-set marker construction and publication |

Logging and visualization remain available. Moving their implementation out of
the main file does not eliminate the real-time cost of capturing bounded log
records. Per-request candidate generation and monitoring execute on the monitor worker;
CSV serialization and marker publication execute outside the control loop.
ROS goal ingress still constructs a new long-term via-point path when the control
loop accepts a goal; this refactor does not claim an allocation-free or lock-free
control update.

## Removed configuration and behavior

Remove these keys from custom configurations:

- `async_planning_lead_steps`
- `async_verified_horizon_steps`
- `async_plan_max_age_sec`
- `async_safety_monitor`
- `local_replan_horizon_steps` (trajectory settings)

Monitoring is always asynchronous. Both segment lengths come from N, and the
planner directly produces N samples instead of building a separate 64-step
buffer and truncating it. No wall-clock cache-expiry setting or late-catch-up
acceptance path remains. The finite verified command stream determines braking.

Run metadata reports `monitor_period_control_cycles`, `committed_steps`,
`fresh_intended_steps`, `monitor_execution: asynchronous`,
`monitor_segment_policy: equal_control_periods`, and
`async_handoff_policy: strict_committed_prefix_deadline` as derived facts.
The existing CSV columns are preserved. Rejection mask bit 32 now denotes a
strict committed-prefix deadline failure; historical files retain their original
`activation_window_or_continuity` description.

## Build and check

In the existing development container/workspace, retaining its normal dependency
environment and CMake cache:

```bash
colcon build --packages-select cps_trajectory_generators cps_controllers franka_bringup
source install/setup.bash
colcon test --packages-select cps_trajectory_generators cps_controllers \
  --ctest-args -R '^test_' --output-on-failure
colcon test-result --verbose
```

Restart the simulator and controller after rebuilding. For a new CMake cache,
the existing environment additionally uses:

```bash
--cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MODULE_PATH=/home/developer/multipanda_ws/src/mujoco_ros_pkgs/mujoco_ros/cmake \
  -DMUJOCO_DIR=/home/developer/Libraries/mujoco \
  -DFranka_DIR=/opt/libfranka/lib/cmake/Franka
```

Compare new logs against the earlier runs using the effective metadata. Fixed
prefixes that cross an existing braking segment may postpone resumption if a new
trajectory fails continuity checks. Strict deadline rejection can also cause
more braking under scheduling overruns. Neither case permits skipping the
energy, recovery or trajectory-continuity conditions.
