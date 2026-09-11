# Reachable controller methodology cleanup — 2026-09-10

## Result

The controller now uses one derived monitor-period step count for scheduling,
committed commands and fresh intended commands:

`N = 1 / (local_replan_dt * monitor_frequency_hz)`.

Current configuration is 100 Hz and 1 ms, giving 10/10. Numeric ratios that are
not positive integral control periods are rejected. No independent committed,
fresh-intended or 64-step planning-horizon setting remains. A nonzero configured
ROS controller update rate must match the command grid at activation.

Removed behavior:

- Late-catch-up acceptance and its duplicated one-step transition helper/counter.
- Automatic committed-prefix truncation at the intended/braking boundary.
- The separate wall-clock expiry of cached monitor decisions.
- Synchronous planning/monitoring mode in the control update.
- Generation of a 64-step intended buffer followed by truncation; the generator
  now directly generates one full N-step period, including at the path end.
- Horizon-dependent nominal-gain blending. Fresh intended commands use nominal
  gains, with the existing runtime energy law remaining responsible for gain
  adaptation.
- The worker's duplicate verified-plan execution cache. Only the control thread
  owns and advances the accepted stream.

The full candidate is N committed commands + N fresh intended commands + complete
braking. Existing source-generation, recovery epoch/state, workspace, calibration,
energy, joint-limit and trajectory-continuity guards remain. Results arriving
at the final committed boundary are admissible; results arriving later are
rejected. The controller retains its previous verified stream and braking after
an unavailable/rejected result.

## Structure

The main implementation went from 5,329 to 2,180 lines. Core path/monitor/energy/
impedance functions remain there. ROS lifecycle and model adapters are in
`lifecycle.cpp`; asynchronous handoff is in `monitor_execution.cpp`; thread and
mailbox support is in `monitor_worker.cpp`; ROS callbacks are in `commands.cpp`;
logging/profiling capture and serialization are in `logging.cpp`; visualization
continues in `visualization.cpp`. The former mixed `async.cpp` was removed.

This is a structural and policy simplification, not a claim that every real-time
allocation or lock has been eliminated. Log capture still has bounded producer
cost, and accepting a new ROS goal still constructs its long-term via-point path.
Visualizations, recordings, calibration actions, runtime energy adaptation and
hardware torque limits remain available.

## Validation

Built `cps_trajectory_generators`, `cps_controllers` and `franka_bringup` in the
user's development container with isolated build/install directories under
`/tmp/cps-methodology-cleanup-*`. The normal installed controller was not replaced.
The fresh build required the same MuJoCo module/SDK and libfranka paths present
in the existing workspace CMake cache. Final controller compilation/linking passed.

- 26 C++ test cases passed: 6 monitor-policy cases, 5 request-gate cases, 3 mailbox
  cases, 1 worker-timing case and 11 trajectory cases.
- New tests cover 200/100 Hz derivation, a changed control step, periods over the
  former 64-step horizon, invalid/nonintegral rates, inclusive/expired committed
  deadlines, regressing/wrapping sequence values and full-length terminal holds.
- 14 Python timing-analysis and log-schema regression tests passed. Python pytest
  plugin auto-loading was disabled because the host's Anaconda interpreter tries
  to load unrelated ROS launch-testing plugins with unavailable dependencies.
- Final whitespace/diff checks passed.

Two isolated MuJoCo controller runs on ROS domain 175 successfully loaded,
activated, accepted motion, deactivated and drained all controller logs:

| Run | Control rows | Monitor results | Accepted candidates | CSV drops/schema mismatch |
|---|---:|---:|---:|---:|
| `/tmp/cps-methodology-smoke` | 6,825 | 682 | 657 | 0 |
| `/tmp/cps-methodology-smoke-service` | 6,811 | 681 | 656 | 0 |

Both verified effective 100 Hz, 10 committed commands in every processed result,
20 total intended-container commands for accepted candidates, and acceptance
within 10 control steps. These are short functional tests, not GUI performance
benchmarks or worst-case latency guarantees. Observed handoff masks were 0 and 8
(workspace policy); the tests included initial observation startup.

Simulator process teardown did not pass: sending SIGINT to the whole process
group produced an invalid-publisher-context exception; using `/shutdown` produced
`Node needs to be associated with an executor` after controller manager shutdown.
The latter was also reproduced with the unchanged installed controller (its
baseline used 100 Hz with the older 8/20 configuration). In all cases the reachable
controller had already deactivated and drained its logs successfully. The baseline
harness's final assertions initially used the wrong log directory; its recorded
startup/deactivation and executor exception were checked directly. That one
baseline recording was moved out of normal `data_log` into
`/tmp/cps-methodology-baseline/data_log/20260910_213637_331475` so it cannot be
mistaken for a user experiment. No MuJoCo code was changed by this cleanup.

## User rebuild and next run

Use the normal container workspace and its existing dependency environment:

```bash
cd /home/developer/multipanda_ws
colcon build --packages-select cps_trajectory_generators cps_controllers franka_bringup
source install/setup.bash
```

Restart the simulator and controller with the same GUI/RViz launch command.
Confirm the new `run_info.txt` contains:

```text
monitor_frequency_hz: 100
monitor_period_control_cycles: 10
committed_steps: 10
fresh_intended_steps: 10
monitor_execution: asynchronous
monitor_segment_policy: equal_control_periods
async_handoff_policy: strict_committed_prefix_deadline
```

Only `monitor_frequency_hz` needs changing for subsequent monitor-frequency
experiments at the same control rate. A new runtime recording is necessary to
measure the behavioral/performance effect of strict deadlines and full prefixes
through braking; past 100 Hz/10/10 recordings used the earlier implementation.
