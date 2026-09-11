# MuJoCo realtime changes and test procedure

## Behavior changes

- The event thread now sleeps until its 16.6 ms deadline, outside the physics
  mutex. Previously it subtracted the clock epoch duration, producing a negative
  sleep and repeatedly contending for the mutex.
- `physics_wall_pacing:=true` (default) executes one physical step per outer
  iteration, using the model timestep and requested realtime factor. Since the
  2026-09-11 fix, short overruns retain the absolute schedule, with at most four
  periods of timing debt and a half-period minimum between step starts. This
  allows bounded recovery without holding the physics mutex across a batch.
  Pause/resume and speed changes reset the schedule; simulation steps are never
  dropped. Sustained overload can still slow the simulation. This is not a hard
  realtime guarantee. See `mujoco_pacing_fix_20260911.md` for measured results.
- `physics_wall_pacing:=false` retains the legacy catch-up step loop for comparison.
  `realtime:=-1.0` also uses legacy unlimited stepping. Both still receive the
  event-thread, executor and nonblocking-render fixes; this is not a complete
  restoration of the old implementation.
- Paused idle waiting is outside the physics mutex. Explicit Step actions retain
  exact step counts, with one step per outer iteration in wall-paced mode.
- Offscreen submission tries the render mutex once. If a snapshot is pending or
  the renderer is busy, physics continues without modifying that snapshot. The
  renderer retains ownership until rendering/publication finishes. Scene capture
  still executes synchronously under the physics mutex and is timed separately.
  Image delivery is best effort: rendering/publication can run below its requested
  rate when overloaded. Sensor consumers must use actual image timestamps.
- GUI mode uses a continuously spinning ROS executor rather than 10 ms polling.
  Headless finite-step and service shutdown wake the executor and join its threads.
- The control plugin's URDF lookup uses a non-auto-associated callback group and
  a local executor. It must not spin the shared executor again while that executor
  is running. Requests are bounded and cancelled on simulation/ROS shutdown.
- `no_render` correctly overrides both internal render settings. A default false
  deprecated `no_x` no longer cancels an explicit `no_render:=true`.
- Optional timing uses preallocated, bounded queues and a separate CSV writer.
  Full queues or concurrent producers drop telemetry instead of blocking physics.
- Optional `physics_cpu_affinity` pins only the physics thread. Default `-1`
  preserves existing affinity. No scheduling priority or system-wide affinity is
  changed. Existing monitor CPU/priority settings remain as configured.

All new parameters are startup parameters: restart the simulator between cases.
The single-arm launch explicitly forwards them. Server-level launch files also
accept them, including debug/valgrind branches.

## Build and basic regression tests

Use the same ROS environment/container and MuJoCo/libfranka dependency paths as
your previous successful robot build. From this repository root:

```bash
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select mujoco_ros --event-handlers console_direct+
colcon test --packages-select mujoco_ros2_control \
  --ctest-args -R robot_description_client_test --output-on-failure
colcon test-result --verbose
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tools/test/test_mujoco_timing.py \
  tools/test/test_async_monitor_timing.py \
  tools/test/test_shield_log_schema.py
python3 tools/test_mujoco_realtime_smoke.py
```

Rebuild MuJoCo-dependent plugins, including `mujoco_ros2_control`, together with
the core library: `MujocoEnv`'s layout changed. A full workspace build avoids
loading an old plugin against the new header layout. Do not use `--packages-select
mujoco_ros` alone and then run an old controller plugin.

The smoke tool launches only the repository's pendulum world, without robot
plugins, in localhost ROS domain 175. It checks paced/legacy/unlimited modes,
`no_render` overrides, exactly 1000 physical steps and clean automatic exit.
It keeps CSV/log files in `/tmp/mujoco-realtime-smoke`. This is a correctness test,
not a robot performance benchmark. Stop any previous smoke process in that
domain before rerunning. Override `--domain-id` if needed.

If the workspace was moved and CMake reports an old absolute source directory,
use a fresh build/install pair rather than reusing that cache:

```bash
colcon build --build-base build_rt --install-base install_rt \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install_rt/setup.bash
```

Use that same overlay in every terminal. Preserve the dependency environment
from your normal build; do not replace a working MuJoCo installation with a newer
version for these tests. Local validation used MuJoCo **3.2.0**. Existing Viewer
code uses fields removed in 3.2.7, so a newer 3.x library is not automatically
source-compatible.

## Robot A/B runs

Keep the robot scene, tool, human trajectory, controller configuration, recording
options, and commanded robot trajectory identical. Keep the monitor at 200 Hz.
Use the 30 ms human observation period in every new run. The older 18:42/18:43
logs had different live human workspaces and are not a controlled A/B baseline.

Create the output directory:

```bash
mkdir -p data_log/mujoco_realtime
```

Start one simulator case at a time. The launch remains paused initially so you
can start the controller and human provider before advancing the simulation.
Append your existing scene/tool/initial-position arguments to every command.

**A: paced, no rendering — recommended first run**

```bash
ros2 launch franka_bringup franka_sim.launch.py \
  headless:=true render_offscreen:=false use_rviz:=false \
  physics_wall_pacing:=true realtime:=1.0 \
  physics_timing_path:="$PWD/data_log/mujoco_realtime/A.csv"
```

In separate sourced terminals:

```bash
ros2 control load_controller --set-state active reachable_cartesian_impedance_controller
```

```bash
ros2 launch cps_human_workspace human_workspace_visualizer.launch.py \
  human_workspace_config:=human_workspace_dynamic_crossing.yaml \
  publish_rate:=33.333333333333336
```

Use your existing human configuration instead of this example if it differs;
keep that choice identical throughout the comparison. The provider creates its
timer at startup, so restart it to apply the 30 ms period.

Unpause after the controller and provider are ready:

```bash
ros2 service call /set_pause mujoco_ros_msgs/srv/SetPause \
  "{paused: false, admin_hash: ''}"
```

Send the same robot trajectory and start the same controller data recording
procedure you used for the previous logs. Record 30–60 seconds, stop recording,
then exit the simulator normally with Ctrl-C so pending timing records drain.
Restart the complete case before each repeat. Use distinct paths such as
`A1.csv`, `A2.csv`, `A3.csv`; **an existing timing file is overwritten at startup**.

**B: legacy pacing, no rendering — isolates pacing policy**

```bash
ros2 launch franka_bringup franka_sim.launch.py \
  headless:=true render_offscreen:=false use_rviz:=false \
  physics_wall_pacing:=false realtime:=1.0 \
  physics_timing_path:="$PWD/data_log/mujoco_realtime/B.csv"
```

**C: paced with MuJoCo window — isolates viewer overhead relative to A**

```bash
ros2 launch franka_bringup franka_sim.launch.py \
  headless:=false render_offscreen:=false use_rviz:=false \
  physics_wall_pacing:=true realtime:=1.0 \
  physics_timing_path:="$PWD/data_log/mujoco_realtime/C.csv"
```

**D: paced with window and offscreen camera rendering — compare with C**

```bash
ros2 launch franka_bringup franka_sim.launch.py \
  headless:=false render_offscreen:=true use_rviz:=false \
  physics_wall_pacing:=true realtime:=1.0 \
  physics_timing_path:="$PWD/data_log/mujoco_realtime/D.csv"
```

D tests camera rendering only if the model contains cameras and rendering
requests actually occur; check `render_requests` and the `render` channel. Keep
the same image subscribers between repeats. An additional D run with
`use_rviz:=true` measures the extra visualization load separately.

Run each case at least three times. Startup output includes the effective
`wall_pacing`, `headless`, `offscreen`, physics CPU and timing path settings.

## Analyze each matched pair of logs

```bash
python3 tools/analyze_mujoco_timing.py data_log/mujoco_realtime/A.csv \
  --skip-seconds 2 --output data_log/mujoco_realtime/A_summary.json
python3 tools/analyze_async_monitor_timing.py data_log/YOUR_MATCHING_CONTROLLER_RUN \
  --output data_log/mujoco_realtime/A_monitor_summary.json
```

Replace `YOUR_MATCHING_CONTROLLER_RUN` with the actual timestamped controller
directory from A. Keep each outer-loop CSV together with its matching controller
run ID. The second script without a directory selects the latest complete run,
but an explicit path avoids accidentally analyzing a later run.

Compare these fields:

| Metric | Interpretation / desired result |
|---|---|
| Monitor `async_monitor_output_handoff_ms` p99 and max | Primary outcome; initial experiment target p99 below 2 ms, not a guarantee |
| Controller `control_start_interval_ms` p99 and max | Long invocation gaps should decrease together with handoff |
| Physics `interval_ms`, counts above 2/5/10 ms | Outer physical invocation gaps |
| Physics `wake_late_ms` | Actual wake time minus that iteration's requested deadline; retries have separate deadlines |
| Physics `lock_misses`, `lock_wait_ms` | Failed try-locks and elapsed span from first failure to success, including retry sleeps/scheduling |
| Event/Viewer `elapsed_ms` | Time in the instrumented physics-lock critical section |
| Event/Viewer `lock_wait_ms` | Time waiting to acquire the physics mutex, not time blocking physics |
| Physics `step_ms` | `mj_step` wall time, including embedded controller callbacks |
| `clock_publish_ms`, `callbacks_ms`, `render_snapshot_ms` | Other synchronous work after `mj_step` |
| Per-channel `cpu_ms` versus `elapsed_ms` | Distinguishes on-CPU work from elapsed time off CPU; not a proof of which scheduler/lock caused it |
| `physics_steps_per_iteration` | One in paced running mode; potentially multiple in legacy mode |
| `simulation_seconds_per_wall_second` | Check that lower handoff does not conceal substantial simulation slowdown |
| `event_observed_hz` | Approximately 60 Hz, allowing lock contention and scheduling |
| `render_busy_physics_iterations` | Number of busy submission attempts summed over iterations; not an exact camera-frame drop count |
| `dropped_total` | Should remain zero; nonzero means telemetry coverage is incomplete |
| Verified rejection masks | Source-generation errors should remain zero; recovery-state and activation failures must be counted separately |

Timing semantics: `interval_ms` is start-to-start, while `elapsed_ms` and phase
times describe the **current** iteration. Inspect the previous iteration's work
to explain the next long interval. `mj_step` already includes controller work;
do not add controller execution a second time. Lock-wait spans and wake lateness
can overlap. CSV channels are drained separately and file order is not global
time order: align records with `steady_ns`.

The timing CSV covers the simulator's whole lifetime. For direct comparison with
a shorter controller recording, select the overlapping steady-time range and
exclude pauses/reloads; `--skip-seconds 2` only removes initial warmup. The
analyzer's progress ratio excludes pauses and missing consecutive records.

## Optional affinity experiment, after A–D

```bash
lscpu -e=CPU,CORE,SOCKET,ONLINE
pgrep -a mujoco_node
```

Choose a physical core different from the monitor's core (currently configured
as logical CPU 22), accounting for SMT siblings. Add
`physics_cpu_affinity:=N` to case A, replacing N with that CPU index. Check the
actual runtime thread state using the simulator PID:

```bash
ps -L -p PID -o pid,tid,psr,cls,rtprio,comm
cat /proc/PID/task/TID/status
```

Find `mj_physics` and inspect `Cpus_allowed_list`; `PSR` alone is only the most
recent CPU, not its affinity mask. Thread names also include `mj_events`,
`mj_viewer`, `mj_render`, and `mj_timing`. Assigning physics to a CPU does not
reserve that CPU against other threads/processes. Avoid pinning the entire
MuJoCo process to one core. Keep priority changes out of the first comparison.

## Validation performed here

- Full GLFW-backend `mujoco_ros` library and executable compiled and linked in
  an isolated `/tmp` build with MuJoCo 3.2.0 and regenerated ROS messages.
- Seven C++ regression tests passed, including render ownership under contention,
  deadline behavior, concurrent queue correctness and final CSV drain.
- After a user startup report exposed the plugin's nested-spin call, the full
  `mujoco_ros2_control` plugin was rebuilt and six ROS parameter-client regression
  tests were added. They cover an already-spinning shared executor, repeated
  lookup, delayed/invalid responses, and cancellation. The earlier pendulum smoke
  tests did not load the control plugin and therefore did not cover that path.
- Fourteen Python analyzer/schema tests passed; launch Python/XML parsed and
  all server parameter branches were checked.
- Four headless pendulum modes passed: paced, legacy, unlimited and `no_render`
  override, each with exactly 1000 steps and automatic exit.
- A live ROS Step action completed exactly 20 paused steps; unpause, pause and
  service shutdown completed successfully.
- One no-render pendulum sample after 0.1 s warmup measured approximately 59.8 Hz
  event processing, 1.004 ms mean physics interval, 1.28 ms p99, and 0.996
  simulation seconds per wall second. These are lightweight smoke measurements,
  **not** reachable-controller or GPU-rendering performance results.

The full robot workload, real GUI/offscreen image delivery, and post-change
monitor handoff still require the A–D runs above. No energy limit or recovery
rejection guard was relaxed by this change.
