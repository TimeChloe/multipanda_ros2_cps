# Latest two GUI/RViz experiments — 2026-09-14

## Inputs and measurement method

| Label | Controller run | Physical timing CSV |
|---|---|---|
| A (14:10) | `20260914_141049_198691` | `GUI_RVIZ_pacing_fixed_20260914_121045.csv` |
| B (14:11) | `20260914_141146_821838` | `GUI_RVIZ_pacing_fixed_20260914_121140.csv` |

Both controller metadata files confirm asynchronous monitoring at 100 Hz,
1 ms nominal control timestep, 10 committed and 10 fresh intended steps,
strict prefix deadline, single outstanding request, and accept-before-publish.
The user supplied a GUI/RViz launch without an explicit `LP_NUM_THREADS` setting.
The CSVs do not identify the active OpenGL renderer or inherited environment;
absence of the prefix does not establish either GPU or software rendering.

Controller `wall_time_sec` is elapsed ROS time. Physics `sim_time` has its own
origin, and paused controller calls still increment the control sequence.
Neither numeric time equality nor control sequence multiplied by 1 ms is a
reliable cross-log mapping in all phases.

We aligned the complete controller and physical start-interval sequences.
Alignment correlations are 0.9300945 (A) and 0.9998878 (B); median absolute
interval discrepancies are 0.004361 and 0.004887 ms. Physical and control
sequences have no missing rows. The complete controller windows map to:

- A: controller ROS time 0.322–43.067 s, physics simulation 0.323–43.068 s;
  42.746057 wall seconds and 42,746 control rows, all in continuous running.
- B: controller ROS time 0–19.806 s, physics simulation 0–19.807 s;
  21.237800 wall seconds and 19,850 control rows, initially paused.

Running-only statistics require consecutive advancing physics calls and, for
result events, an unbroken advancing physical interval from request publication
to receipt. This excludes 43 paused rows and the first resume-boundary row in B,
leaving 19,806 control rows and 1,980 result events over 19.805001 wall seconds.
No arbitrary slow-sample filtering or additional warmup trimming is applied.
P99 uses nearest-rank quantiles. Maxima of separate phases need not belong to
the same result and must not be added.

## Running latency (milliseconds)

| Metric | A mean | A P99 | A max | B mean | B P99 | B max |
|---|---:|---:|---:|---:|---:|---:|
| Publication → worker begins | 0.145 | 0.232 | 0.451 | 0.145 | 0.231 | 0.475 |
| Worker computation | 0.898 | 1.951 | 2.408 | 1.142 | 1.993 | 2.084 |
| Worker finish → receipt (handoff) | 0.544 | 1.043 | 5.588 | 0.536 | 1.000 | 1.367 |
| Publication → receipt (end-to-end) | 1.587 | 3.025 | 6.038 | 1.824 | 3.023 | 3.117 |
| Control call interval | 1.000 | 1.219 | 6.035 | 1.000 | 1.191 | 2.008 |
| Controller execution | 0.024 | 0.078 | 2.863 | 0.022 | 0.076 | 0.428 |

All received results arrive within 1–4 control steps in A and 1–3 in B,
inside the 10-step prefix. A has two running handoffs above 2 ms, only one above
5 ms. B has no running handoff above 2 ms.

Worker mean non-CPU time is 0.001266 ms (A) and 0.001175 ms (B); maxima are
0.004126 and 0.002414 ms. Computation varies mainly with workload: mean rollout
length is 52.90 versus 65.55 intervals, and rollout-length/compute correlations
are 0.9877 and 0.9966. B's larger average computation is not evidence of worse
communication. Monitor evaluation accounts for about 85.2% and 86.7% of worker
elapsed time, respectively.

These durations stop at receipt. They do not include waiting for a human
observation, waiting for the next monitor request, or executing the remaining
committed prefix before fresh intended commands begin.

## Why B has approximately 33 ms values in the raw CSV

Five result events (input sequences 60–64) occur while the matched physics rows
have `running=0`, `steps=0`, and simulation time zero. Their end-to-end latencies
are 30.617, 33.301, 33.414, 33.297 and 33.429 ms; handoff reaches 32.675 ms.
All have rejection mask zero. Each waits for the next paused update, which runs
approximately once per 33.3 ms; the worker itself is not taking 33 ms.

The raw request interval also reaches 333.057 ms: ten paused control calls take
approximately ten times 33.3 ms. This is consistent with control-sequence-driven
monitor scheduling. Raw output throughput over the whole recording is 93.47 Hz
because the window includes the pause. It should not be reported as running
simulation speed or monitor throughput.

After requiring advancing physics, B's maximum end-to-end is 3.117 ms,
maximum handoff 1.367 ms, and measured throughput is approximately 99.97 Hz.
The first resume-boundary physical interval (33.300 ms) is also excluded from
steady running call-spacing statistics.

## A's real 6 ms tail: delayed clock-publication stage

At controller elapsed 19.383 s, input 1956 is received after only one control
step, with these measured durations:

- Worker queue wait: 0.005719 ms.
- Worker computation: 0.444301 ms.
- Handoff: 5.587720 ms.
- End-to-end: 6.037740 ms.
- Rejection mask: zero; the result is accepted.

The preceding physical iteration, sequence 19638 at simulation time 19.383 s,
spends 6.029597 ms in work, of which 5.967595 ms is in the `PublishSimTime`
timing stage. `mj_step` takes only 0.047619 ms and later callbacks 0.014063 ms.
The following physical start interval is 6.030864 ms. No physics mutex miss is
recorded. The preceding iteration's total thread CPU time is only 0.088968 ms.

This explains the delayed receipt: the controller cannot consume the ready
worker result until its next call. It identifies a non-CPU stall inside the
clock-publication timing bracket, which includes allocation and ROS publication.
It does **not** distinguish DDS/publisher blocking from OS preemption during that
bracket. A claim that DDS itself blocked for exactly 5.97 ms would exceed the
instrumentation. Source: `mujoco_ros/src/ros_two/ros_api.cpp`, `PublishSimTime`;
phase boundaries: `mujoco_ros/src/physics.cpp`, `StepPhysics`.

The other handoff above 2 ms occurs at controller time 15.883 s; its physical
call interval is 2.681 ms and wake lateness 1.701 ms. It is also accepted.

If tail latency needs further work, instrument CPU time/context switches inside
`PublishSimTime` and separate allocation from `publish`, then examine middleware
and scheduler traces. These two recorded events do not justify changing QoS or
moving `/clock` publication asynchronously without checking simulation-clock
ordering and controller timestamp behavior.

## Delivery and rejection accounting

| Full lifecycle counter | A | B |
|---|---:|---:|
| Inputs published | 4325 | 2044 |
| Worker processed | 4325 | 2044 |
| Outputs consumed | 4325 | 2044 |
| Input/output overwrites | 0 | 0 |
| Stale-before/after-computation drops | 0 | 0 |
| Busy-deferred cycles / late schedules / skipped slots | 0 | 0 |
| Control/prediction CSV drops or schema mismatches | 0 | 0 |

These counters cover the complete controller lifecycle. The smaller event counts
below cover command-triggered recording. A has 49 earlier unrecorded events;
B has 59. Every raw recorded event in B includes its five paused results.

| Raw recorded events | A | B |
|---|---:|---:|
| Received results | 4276 | 1985 |
| Verified and accepted | 4212 | 1943 |
| Verified but rejected | 2 | 4 |
| Unverified | 62 | 38 |
| Source-plan version mismatch | 0 | 0 |
| Strict-prefix deadline rejection | 0 | 0 |

All six verified-result rejections are mask 2 (`recovery_epoch`). Each arrives
after two control calls, with 1.942–2.102 ms end-to-end latency. Recovery state
changes while the worker is computing, invalidating its earlier context:

- A: controller times 29.556 s (epoch 4→5, overlap→clear/recovering) and
  31.116 s (5→6, clear→overlap/limited).
- B: 7.649 s (16→17), 11.559 s (24→25), 11.629 s (30→31),
  12.949 s (34→35), all entering new limited/overlap contexts.

These are expected invalidations under the current recovery policy, not source
publication ordering failures or late delivery. The logs do not by themselves
prove that every rapid recovery transition is desirable; changing that policy
would be a separate controller-methodology decision.

Prediction CSV cross-check:
- A: 60 unverified predictions with possible contact and energy upper bound
  0.120021754–0.170123079 J, all above the 0.12 J budget; no joint-limit-unsafe
  verdicts. Two additional unverified outputs fail candidate construction with
  reason 9, `failsafe_sample_invalid`, at 27.155 and 27.165 s (inputs 2734/2735).
  They have no prediction rows because evaluation is not reached, not because
  a log or communication message was lost.
- B: all 38 unverified predictions have possible contact and energy upper bound
  0.120250659–0.193012813 J; no joint-limit-unsafe verdicts or missing candidates.

## Running simulation progress and assessment

| Running metric | A | B |
|---|---:|---:|
| Simulation / wall progress | 99.9975% | 100.0000% |
| Received results per wall second (approximate) | 100.03 Hz | 99.97 Hz |
| Request wall interval mean | 9.998 ms | 10.000 ms |
| Request wall interval P99 | 10.172 ms | 10.153 ms |
| Request wall interval max | 12.995 ms | 10.773 ms |

A has one shorter request phase after a new trajectory at controller time
19.447 s; commanded path time resets to zero. This is the existing intentional
command-triggered rescheduling, not a skipped monitor slot.

The previous source-version race and late-result rejection problem do not recur
in these two experiments. Result delivery is complete, and normal running timing
meets the observed committed-prefix deadline. There is still one significant
ROS clock-publication-stage tail in A. B's larger raw maximum is fully explained
by its initial pause. Thus “monitor pipeline healthy in these runs” is supported;
“all ROS communication is latency-free” is not.

Keep 100 Hz and automatic 10/10 segmentation. The current launch achieves near
1x progress with visualization enabled; this evidence does not require further
monitor-frequency reduction or longer committed horizons. Remaining follow-ups
are the rare clock-publication stall and A's two invalid braking samples.
No controller, simulator, launch or container configuration was changed during
this analysis.

Detailed JSON, alignment diagnostics, rejection context and analysis scripts:
`data_log/mujoco_realtime/analysis_20260914_latest_two/`.
