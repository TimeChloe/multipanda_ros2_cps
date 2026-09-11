# MuJoCo GUI pacing diagnosis and fix — 2026-09-11

## Latest user recordings

The two new timing files are `GUI_RVIZ_clean_100Hz_20260911_143754.csv`
and `GUI_RVIZ_clean_100Hz_20260911_143823.csv` under
`data_log/mujoco_realtime`. Statistics exclude the first two **wall-clock**
seconds after the first advancing, unpaused physics sample.

| Metric | 14:37:54 | 14:38:23 |
|---|---:|---:|
| Simulation / wall progress | 86.13% | 87.69% |
| Mean physical iteration work | 0.141 ms | 0.133 ms |
| Mean physical start interval | 1.161 ms | 1.140 ms |
| Start interval p99 | 4.949 ms | 4.148 ms |
| Start interval maximum | 9.678 ms | 8.824 ms |
| Intervals over 2 ms | 820 | 181 |
| Of these, wake lateness over 1 ms | 708 (86.3%) | 161 (89.0%) |
| Of these, preceding iteration work over 1 ms | 101 | 18 |
| Physics mutex acquisition misses | 19 | 4 |

The final two explanatory counts can overlap. The iteration work includes
MuJoCo integration, ROS clock publication and callbacks. It is not controller
compute time alone. Offscreen render submission time is zero in both runs.
Viewer synchronization sections are short, but do not measure full drawing.

MuJoCo's displayed actual percentage is simulation progress divided by wall
progress, measured over approximately 100 ms windows. `realtime:=1.0` specifies
a target, not a reserved CPU budget or a guarantee. A 100 Hz monitor does not
change the simulator's 1 ms integration/control step.

## Two interacting causes

1. **Software rendering competes for CPU.** A GLFW/OpenGL probe inside the
   actual container returned `Mesa`, `llvmpipe (LLVM 15.0.7, 256 bits)`, and
   Mesa 23.2.1. Both the probe and RViz report failure to load `nouveau`.
   The MuJoCo process has many active `llvmpipe-*` threads. The container has
   NVIDIA device nodes but no `libGLX_nvidia` in its normal GLX library directory.
   Docker inspection reports runtime `runc`, no DeviceRequests, and no configured
   `NVIDIA_DRIVER_CAPABILITIES`. CPU quota is unlimited (`cpu.max: max 100000`).
   Merely requesting NVIDIA PRIME offload in a temporary process still returned
   llvmpipe in this container.
2. **The old pacing policy discarded progress after a missed deadline.**
   `nextDeadline` used `previous + period` when still in the future, otherwise
   `finish + period`. A scheduling stall was therefore followed by another full
   wait, with no way to recover lost simulation progress. Low average step cost
   could coexist with persistently slow simulation.

Software rendering and delayed wakeups are observed. The experiments below
support CPU contention as a contributor; they do not attribute every delayed
wake to one graphics thread or rule out other operating-system interference.

## Code change

Only MuJoCo scheduling changes; the reachable controller's methodology,
100 Hz configuration, 10/10 segments and rejection checks are unchanged.

`mujoco_ros/include/mujoco_ros/realtime_support.hpp` adds `PhysicsPacer`.
`mujoco_ros/src/physics.cpp` uses it only during finite-speed, running,
wall-paced simulation:

- Preserve an absolute target across short overruns.
- Keep at most four wall periods of scheduling debt after a long stall.
- Schedule the next start no earlier than half a wall period after this start
  and never before the current work finishes.
- Perform one physical step per outer iteration and release the mutex each time.
- Reset timing on resume, speed change and detected simulation reset.
- Preserve paused/manual-step and legacy/unlimited behavior.

At the current 1 ms wall period the lower start-spacing bound is 0.5 ms.
This is bounded simulation speed recovery, **not** controller `late_catchup`:
no expired monitored result receives permission to execute. No physical steps
are discarded and no integration timestep is enlarged.

There is a tradeoff: while recovering, ten simulation/control steps can occupy
approximately 5 ms of wall time. The asynchronous worker has less wall time to
meet the same ten-step deadline. Existing acceptance guards and verified
braking remain essential. Average 100% speed does not imply uniform 1 ms calls
or a hard real-time guarantee. Sustained overload or long suspensions can still
reduce achieved speed.

## Validation

An isolated Release build was made under `/tmp/mujoco-pacing-{build,install}`
in the container. The normal installed workspace was not replaced. This change
does not alter `MujocoEnv` layout or its plugin ABI.

- 11 C++ tests pass, including phase preservation, bounded recovery after a
  short/long stall, reset, overloaded computation, render snapshots and logging.
- Four pendulum integration smoke cases pass: paced, legacy, unlimited and
  `no_render` override. Each advances exactly 1000 steps / 1 s and exits cleanly.
- Robot GUI tests exercise pause, resume, controller activation/deactivation,
  human observations and Cartesian via points. Full robot shutdown retains the
  previously observed executor teardown issue; runtime results are assessed
  before shutdown and do not imply that separate issue is fixed.

The comparison script is `/tmp/mujoco_pacing_compare.py` in the container.
Each test has two separately measured 15-second wall windows: simulator-only
(MuJoCo + RViz + joint-state broadcaster), then active reachable controller with
human observations. The same isolated controller build and scene are used;
these are sequential single trials, not a statistical performance guarantee.
`phases.json` stores steady-clock boundaries. Results, timing CSVs and controller
summaries are copied to `data_log/mujoco_realtime/analysis_20260911_pacing`.

### Measured comparison

| Configuration | Simulator only | Controller active | Active physics interval p99 |
|---|---:|---:|---:|
| Old pacing, default software renderer | 86.30% | 86.47% | 3.910 ms |
| Fixed pacing, default software renderer | 99.96% | 99.96% | 3.724 ms |
| Old pacing, `LP_NUM_THREADS=4` | 95.30% | 96.73% | 2.295 ms |
| Fixed pacing, `LP_NUM_THREADS=4` | 99.99% | 100.00% | 1.785 ms |

The complete fixed/default controller run consumed 1782 results, all accepted.
The fixed/four-thread run consumed 1833 results, with 1832 accepted and one
unverified; every handoff rejection mask was zero. Neither run rejected a
verified result. All four runs have zero control/prediction CSV drops and no
schema mismatch.

For controller timing, discard rows with controller ROS elapsed time below
2 s; this is a different window from the 15-second steady-time physics phases.

| Controller timing | Old/default | Fixed/default | Fixed/four rendering threads |
|---|---:|---:|---:|
| Worker mean | 2.407 ms | 2.196 ms | 2.037 ms |
| Handoff p99 | 3.680 ms | 3.386 ms | 1.390 ms |
| Handoff maximum | 5.713 ms | 5.411 ms | 3.009 ms |
| End-to-end p99 | 6.802 ms | 6.345 ms | 4.065 ms |
| End-to-end maximum | 8.130 ms | 7.866 ms | 5.051 ms |
| Maximum elapsed control steps to receipt | 7 | 9 | 6 |

The fixed/default case illustrates why speed ratio and monitor timing must be
checked separately: it recovers average progress but some outputs use nine of
ten committed steps. Reducing software-renderer contention improves the margin
in this trial. Rendering remains enabled in both windows. Limiting threads can
change GUI frame rate; these tests did not establish a minimum display FPS.

## Running the fix

In the existing container:

```bash
cd /home/developer/multipanda_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select mujoco_ros --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Keep your launch arguments, with a new timing filename. While this container
uses llvmpipe, prefix the launch with `LP_NUM_THREADS=4`. This affects rendering
threads in the child GUI processes, not the monitor period or control timestep.
It is a measured mitigation, not hardware GPU rendering.

```bash
mkdir -p /home/developer/multipanda_ws/src/data_log/mujoco_realtime
LP_NUM_THREADS=4 ros2 launch franka_bringup franka_sim.launch.py \
  scene:=no_table \
  tool_config:=/home/developer/multipanda_ws/src/franka_bringup/config/tools/metal_ball.yaml \
  headless:=false render_offscreen:=false use_rviz:=true \
  physics_wall_pacing:=true realtime:=1.0 unpause:=false \
  physics_timing_path:=/home/developer/multipanda_ws/src/data_log/mujoco_realtime/GUI_RVIZ_pacing_fixed_$(date +%Y%m%d_%H%M%S).csv
```

Activate the usual controller and human provider, then:

```bash
ros2 service call /set_pause mujoco_ros_msgs/srv/SetPause \
  "{paused: false, admin_hash: ''}"
```

Record 30–60 seconds, including the usual commanded motion, and compare both
simulation/wall ratio and controller monitor latency/rejection masks. Occasional
UI dips are possible; do not treat a single displayed percentage as the whole
measurement.

## Hardware-rendering follow-up

The long-term graphics fix is to launch a container with NVIDIA Container
Toolkit graphics/display libraries, preserving existing workspace and data.
Relevant creation options are `--gpus all` and
`-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display`, plus the existing
X11 access and mounts. Do not delete the existing container: its workspace-level
`data_log` and installed dependencies may live in its writable layer.
Setting that environment variable in a shell inside an already created container
does not inject missing driver libraries. Verify actual OpenGL renderer after
setup; `nvidia-smi` alone does not establish OpenGL acceleration. This session
did not recreate the container or modify system graphics drivers.

References:
- [Mesa LLVMpipe](https://docs.mesa3d.org/drivers/llvmpipe.html): CPU software rasterizer.
- [Mesa environment variables](https://docs.mesa3d.org/envvars.html): `LP_NUM_THREADS` controls rendering threads.
- [NVIDIA container driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html): graphics/display library injection.
