# Reachable controller: 18:03 run and realtime review

The latest analyzed run is `data_log/20260910_180352_224792`, schema
`orthogonal_execution_v15`. It agrees with the controller version in this
checkout before these edits. The earlier 17:49 run used v20 and already described
several background-worker changes absent from this checkout. Its much lower
rejection count must not be substituted for the new run's results.

## What was rejected

Count one prediction record per `monitor_input_sequence`, not every trajectory
sample or every control cycle. The control log retains the previous usable
decision when a new result fails handoff, so its `candidate_verified` field is
not the verdict of every newly received result.

| Outcome | Distinct evaluated plans |
|---|---:|
| Verified and accepted | 3,444 |
| Verified but rejected at handoff | 409 |
| Not verified | 257 |
| Total logged evaluated plans | 4,110 |

All **409** verified rejections have `source_plan_matches_at_handoff=0`;
their recovery epoch and recovery state both match. This is **10.62% of the
3,853 verified results**, and **61.41% of all 666 unaccepted evaluated plans**.
Their worst-case predicted energy remains below 0.12 J. This supports the
reported concern about communication, specifically internal request/result
coordination rather than evidence of ROS/DDS packet loss.

All **257** unverified plans have predicted contact, no joint-limit violation,
and worst-case predicted energy above the **0.12 J** budget. The mean bound is
0.124976 J and maximum is 0.168408 J. These are actual energy-check rejections.
The check covers the monitored horizon and includes the configured 0.027 J
potential-energy uncertainty, so a safe current measured energy alone does not
establish that the entire candidate is safe.

The 22,304 control rows contain 1,372 cycles with candidate-prediction fallback
and only nine with async-output-unavailable fallback. That does not contradict
409 rejected verified results: the controller can continue a previous usable
verified plan after rejecting a newer result. Result rejection and fallback
execution are different events.

## Why source generations become obsolete

The old order within `update()` was:

1. Take a completed output from the mailbox.
2. Publish another input using the current plan generation and committed prefix.
3. Accept the taken output, replacing the plan and incrementing its generation.

Step 3 invalidates the input just published in step 2. For **235 of the 409**
rejected verified results, the executed generation at the end of the publishing
control cycle already exceeds the input's source generation. The other 174
become obsolete later while requests/results overlap in flight.

A concrete example from this run:

| Control sequence | Event |
|---|---|
| 3,595 | Executing generation 35 |
| 3,596 | Input 721 captures generation 35; output 720 is accepted as generation 36 |
| 3,600 | Verified output 721 arrives with source 35 and is rejected against generation 36 |
| 3,603 | Output 722, based on generation 36, is accepted as generation 37 |

The generation check is necessary: a candidate was verified against a specific
committed command prefix. Removing that guard would conceal the handoff error
and could execute a trajectory against different commands from those verified.

## Can the 200 Hz worker handle the load?

For the 4,110 logged evaluations, steady-clock timings are:

| Duration | Mean | p99 | Maximum |
|---|---:|---:|---:|
| Input queue wait | 0.241 ms | 1.643 ms | 3.485 ms |
| Worker computation | 1.398 ms | 2.720 ms | 4.566 ms |
| Output handoff | 1.910 ms | 10.170 ms | 22.912 ms |
| End to end | 3.548 ms | 11.819 ms | 24.095 ms |

The recorded computation fits a steady 5 ms request period. This does not
guarantee the activation deadline or controller responsiveness. In v15,
handoff timing starts at worker completion and ends when the control thread
records consumption; it includes publication and controller-side delay. The
log does not separate mailbox waiting, acceptance, or control execution time.

The monitor schedule is every five **control updates**, not a wall-clock 200 Hz
timer. `MujocoRos2ControlPlugin::PassiveCallback()` runs the controller from the
physics callback. `MujocoEnv::SimUnpausedPhysics()` catches simulation time up
with CPU time by executing several steps in a loop. Thus eight control steps
of planning lead are not necessarily eight milliseconds of CPU time.
Rendering waits and the physics mutex are additional possible contributors;
this v15 log cannot identify their individual contribution. The earlier v20
run's uneven control start intervals support the burst explanation, but are
not timing measurements of this new run.

Whole-run metadata also reports 83 overwritten inputs and 268 overwritten
outputs, with zero CSV drops. These counts include the pre-command period;
the prediction CSV contains only logged valid evaluated plans after recording
starts. They must not be equated directly with energy or handoff rejections.

## Changes in this checkout

- Allow only one outstanding request until its result is consumed or the
  worker explicitly discards it. Computation completion alone does not release
  the request slot. This addresses the additional cross-cycle case: publishing
  B while A is in flight would let accepting A invalidate B, even with correct
  same-cycle ordering. Path changes reset the control-side slot, and late
  completions from the old path cannot release a new request. The 5-cycle
  target phase is retained; busy slots are coalesced, never queued or waited
  on by the servo. Actual submission frequency can be below 200 Hz when the
  complete request/acceptance cycle takes longer than 5 control steps.
- Complete output acceptance before constructing the next request, so its
  source generation and committed prefix refer to the newly accepted plan.
  Suppress further input publication when calibration latches in that cycle.
- Discard obsolete source generations in the worker before and after
  computation. Keep controller-side acceptance checks for races after those
  checks. Record both discard counters separately in `run_info.txt`; these
  discards save obsolete work, not evidence of additional accepted plans.
- Move periodic profiling formatting and ROS console output from `update()`
  to a diagnostics thread using a fixed-size latest-value mailbox.
- Move reachable-set rendering/publication off the monitor worker. Reuse the
  exact monitor alpha bound so diagnostics do not contend for SaRA's mutable
  trajectory-alpha calculator. Visualization and profiling may coalesce
  intermediate snapshots rather than blocking their producers.
- Copy only fixed-size command/verdict fields on ordinary servo cycles.
  Keep cached trajectory and joint-prediction vectors outside that copy.
- Add `tools/analyze_async_monitor_timing.py`, supporting v15 prediction
  events and v20/v21 explicit control events, with regression tests for event
  counting, rejection classification, and workload grouping.

## Why computation varies: additional analysis

The worker's elapsed computation ranges from **0.510 to 4.566 ms**. Its
constituent phases, weighted across all 4,110 recorded evaluations, are:

| Phase | Mean | p99 | Fraction of worker elapsed time |
|---|---:|---:|---:|
| Intended planning | 0.127 ms | 0.299 ms | 9.09% |
| Candidate construction | 0.074 ms | 0.146 ms | 5.28% |
| Safety evaluation | 1.185 ms | 2.399 ms | 84.77% |

The monitored horizon changes with the braking trajectory. The final logged
control horizon ranges from **29 to 131 steps**; its Pearson correlation with
worker elapsed time is **0.859**. `verifyReachablePlanJointSpace()` evaluates
each dense intended/failsafe command and then performs reachability checks
using the full rollout. A longer brake therefore entails more dynamics,
energy, and geometry evaluations, despite the same 200 Hz request frequency.

| Final logged control horizon | Evaluations | Mean worker time | Maximum |
|---|---:|---:|---:|
| 29 steps | 587 | 0.733 ms | 1.537 ms |
| 80 steps | 55 | 1.628 ms | 4.139 ms |
| 130 steps | 61 | 2.400 ms | 3.353 ms |

In v15 normal-mode prediction CSVs, `plan_intended_steps` and
`plan_failsafe_steps` describe the **sparse logging view**, not the dense
integration count. This analysis groups by the maximum
`prediction_horizon_steps` across each input's rows instead. The new schema
also records the actual trace-interval count and dense candidate counts.

Different horizon lengths explain a substantial part of the variation, but
not the same-length spikes. For example, input 2257 takes 4.139 ms at an
80-step horizon, with 3.931 ms inside safety evaluation. The old elapsed-clock
timers cannot distinguish CPU execution from descheduling or a blocked lock.
It would be unsupported to identify those spikes specifically as communication
delay or CPU overload from this log alone. Even on-CPU time can change with
state-dependent work, allocation/cache behavior, and CPU frequency.

Schema `orthogonal_execution_v21` adds the following measurements:

- `worker_thread_cpu_ms`: `CLOCK_THREAD_CPUTIME_ID` on the worker.
- `worker_non_cpu_ms`: elapsed minus thread CPU time, including preemption,
  blocking, and small sampling overhead; it is not a network latency measure.
- `worker_voluntary_context_switches` and
  `worker_involuntary_context_switches`: per-request `getrusage(RUSAGE_THREAD)`
  deltas. These help distinguish blocking from involuntary scheduling but do
  not by themselves identify the responsible lock or task.
- `worker_rollout_steps`, `intended_command_count`, `failsafe_command_count`:
  actual available trace intervals and dense candidate lengths.
- Explicit processed-event, verification, acceptance, and rejection-mask
  fields, so old cached control verdicts cannot inflate rejection statistics.
- Controller start interval and previous execution duration, plus deferred
  submission cycles and pending request sequence.

CPU/switch measurements run only on the monitor worker, with `-1` indicating
an unavailable measurement. A rollout count of zero means no trace was
available. No safety horizon, generation check, or energy threshold is relaxed
to make computation look more uniform.

## Remaining realtime work identified

These changes do not make the controller fully allocation-free or hard realtime.

| Location | Remaining work and suitable treatment |
|---|---|
| `acceptPendingCartesianViaPoints()` | Path time parameterization, vector copies, blocking mutexes, action results, and console output still run at command ingress. Prepare immutable paths in a background worker and validate start continuity when handing them to the servo loop. |
| `updateCartesianViaPointsActionStatus()` | Action-state mutexes, feedback allocation/publication, terminal results. Export fixed snapshots and sticky terminal events to a ROS callback; retain command cancellation transitions in the servo loop. |
| `logShieldPredictionTrajectory()` | CSV formatting is asynchronous, but plan/trace copies happen during submission. Prepare a bounded reusable payload on the monitor worker, with reclamation outside the servo thread. |
| Async input and accepted-plan handling | Committed-prefix vectors, accepted-plan copies, and destruction of output vectors still allocate/free. Use bounded storage or explicit slot ownership; mailbox atomics alone do not make payload handling allocation-free. |
| Current collision geometry and path sampling | Current SaRA reach calls allocate, and fallback path-rate estimation copies the full path under a mutex. Use preallocated current-geometry scratch storage and immutable path snapshots. Current overlap is part of energy recovery and must not simply be replaced by an old asynchronous distance. |
| MuJoCo physics/render scheduling | Measure start intervals and physics/render waits separately. Faster controller code alone cannot prevent simulation catch-up bursts. |

The data does not justify increasing the energy limit, weakening generation
checks, or assuming a lower monitor frequency fixes this ordering bug.

## Validation and reproduction

```bash
python3 tools/analyze_async_monitor_timing.py data_log/20260910_180352_224792
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tools/test/test_async_monitor_timing.py tools/test/test_shield_log_schema.py
```

Ten Python tests and nine C++ tests pass. The C++ tests include variable
computation/handoff delays over 2,000 simulated protocol cycles, energy-verdict
rejection without a plan change, cancellation with late old-path completion,
worker discard, restart, mailbox concurrency, and a thread-CPU clock check
that excludes sleeping time. All four
modified controller translation units pass C++17 syntax checks against the
available ROS headers and locally generated current HumanWorkspace interfaces.
The installed workspace/build contains old absolute paths and stale generated
artifacts, so this is not a linked controller build or a simulator validation.
A rebuilt simulation run is still needed to measure the changes' effect on
accepted results and realtime latency. No hardware run was performed.
