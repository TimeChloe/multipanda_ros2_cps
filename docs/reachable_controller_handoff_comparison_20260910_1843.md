# Latest two runs: handoff latency after the request-gate fix

Analyzed `20260910_184254_949684` and `20260910_184329_402608` under
`data_log`. Both report schema v21 and the single-outstanding-request policy.
Counts below use explicit processed-output events; prediction records are
independently deduplicated by monitor input sequence.

## Acceptance results

| Metric | 18:42:54 | 18:43:29 |
|---|---:|---:|
| Processed outputs during recording | 4,792 | 4,875 |
| Verified outputs | 4,540 | 4,647 |
| Verified and accepted | 4,538 | 4,507 |
| Verified but rejected | 2 | 140 |
| Verified acceptance rate | 99.96% | 96.99% |
| Source-generation mismatches | **0** | **0** |
| Inputs / outputs overwritten (whole run) | **0 / 0** | **0 / 0** |
| Worker stale-before / stale-after drops | **0 / 0** | **0 / 0** |
| Unverified energy candidates | 252 | 228 |

The earlier 18:03 run had 409 verified source-generation rejections. The
latest two runs have none, without hiding them through worker stale drops or
mailbox overwrites. The generation-coordination fix is effective in these runs.

The two verified rejections in 18:42:54 are activation-window/continuity
failures. The 140 in 18:43:29 comprise 108 recovery-epoch-only mismatches,
9 recovery-state-only mismatches, 19 with both, and 4 activation-window/
continuity failures. Epoch and state totals must not be added without removing
the 19 overlapping events.

## Handoff is still slow

All durations below are milliseconds, using each consumed result once.

| Timing | 18:42:54 mean / p99 / max | 18:43:29 mean / p99 / max |
|---|---|---|
| Input queue wait | 0.148 / 0.423 / 0.738 | 0.141 / 0.386 / 0.889 |
| Worker elapsed computation | 1.344 / 2.930 / 5.752 | 1.396 / 3.044 / 6.167 |
| Worker CPU execution | 1.339 / 2.902 / 5.747 | 1.390 / 3.025 / 5.125 |
| Worker non-CPU time | 0.0053 / 0.0170 / 2.586 | 0.0052 / 0.0115 / 4.269 |
| **Output handoff** | **1.892 / 9.451 / 39.927** | **1.921 / 9.811 / 23.107** |
| End to end | 3.384 / 10.884 / 40.997 | 3.458 / 11.343 / 24.552 |
| Controller start interval | 1.040 / 7.748 / 40.977 | 1.000 / 7.864 / 24.507 |
| Measured controller execution | 0.041 / 0.311 / 3.778 | 0.045 / 0.356 / 7.417 |

The 39.927 ms handoff in the first run occurs while logged simulation time
is still zero. Excluding those initial zero-time events, its handoff is
**1.852 ms mean, 9.029 ms p99, 16.860 ms max**. The second run's 23.107 ms
maximum occurs during advancing simulation time.

Here, handoff means worker-finish to controller-take timestamp. It includes
mailbox publication, waiting for the controller to run, and the beginning of
that control update before the take. It does not mean that a lock-free mailbox
copy itself consumes 10 ms. These logs do not separately timestamp the mailbox
release-store or every outer physics/render operation.

The association with controller invocation gaps is strong:

- Handoff versus current control-start interval Pearson r: **0.970 / 0.962**.
- For handoffs above 5 ms, the current start interval alone exceeds the whole
  handoff in **354/359 (98.6%) / 385/392 (98.2%)** cases.
- Those same long-handoff updates execute in only **0.102 / 0.105 ms mean**,
  using the following row's previous-execution field to align the measurement
  with the consumed event.
- Median controller start intervals are **0.169 / 0.165 ms**, despite a
  nominal 1 ms period: long gaps are followed by rapid catch-up updates.

A concrete second-run example at simulation time **16.699 s**:

| Observation | Value |
|---|---:|
| Consecutive controller sequences | 16,943 -> 16,944 |
| Real elapsed time between their starts | 24.507 ms |
| Worker computation | 1.371 ms |
| Worker non-CPU time | 0.0014 ms |
| Result handoff wait | 23.107 ms |
| Result accepted | yes |

The result was ready long before the next control invocation. Speeding up the
worker cannot remove the outer control-invocation gap.

The source explains a plausible mechanism: the MuJoCo control manager runs
inside the physics callback. `PhysicsLoop()` sleeps/yields and retries the
physics mutex; `SimUnpausedPhysics()` runs catch-up steps and can wait for
offscreen rendering. `SimPausedPhysics()` runs at a much slower render-related
cadence. The data locates the dominant delay outside the measured update body,
but does not distinguish rendering, mutex contention, sleep/wakeup latency,
or scheduling as the individual cause of every gap.

Rare activation failures need not have large wall-clock latency. First-run
input 686 takes only 2.995 ms end to end but advances nine control steps
against an eight-step prefix. Input 967 advances three steps against a
one-step remaining prefix. Long wall-clock waiting and exhausted simulated
command windows are different failure mechanisms.

## Second-run safety-state changes are a separate issue

The live human workspace differs substantially between the runs despite
matching activation-time metadata. The first run's logged actual human reach
radius is a constant **10 m**, with overlap and limited-energy state throughout
recording. The second run's actual reach radius varies **0.103–0.309 m**, with
moving human coordinates. These are not identical physical test conditions.

The second run has **185** environment changes: 93 clear-to-overlap and 92
overlap-to-clear. All 92 transitions back to clear coincide with a change in
the logged human center, and the signed distance jumps upward by 0.165 m on
average. The prediction snapshots show **252 radius drops above 4 cm**,
spaced **0.100016 s on average** (median 0.100 s).

This is consistent with an approximately 10 Hz observation refresh: the human
reachable set expands with observation age, then shrinks when a new observation
arrives. The repository's human publisher defaults to 10 Hz, and
`handReachableSetAtTime()` explicitly grows the reach bound with age. The logs
do not record the actual publisher parameter, so the rate is inferred from
the radius-reset pattern, not asserted from the default alone.

Thus, many second-run results cross an environment/recovery transition while
in flight. The rejection guards preserve the assumptions under which recovery
exit was verified. Their remaining rejection count is not a recurrence of the
old source-generation bug. Fresher human observations and consistent timestamp/
parameter snapshots merit attention; removing recovery guards is not justified.

## What the CPU measurements add

About 99.6% of aggregate worker elapsed time is CPU execution in both runs.
The worker is usually not blocked/descheduled for a substantial fraction of
its evaluation. Most computation variability is therefore on-CPU variability,
including variable rollout length and state/memory/cache/frequency effects.

There are exceptions: second-run input 1539 takes 6.167 ms elapsed but only
1.898 ms CPU, with 4.269 ms non-CPU time. First-run input 2300 takes 5.752 ms
elapsed and 5.747 ms CPU. Those two peaks have different explanations; labeling
all peaks as scheduling or all as computation would be inaccurate.

## Follow-up and reproducibility

The next timing investigation belongs primarily in MuJoCo's outer loop:
separate physics-mutex acquisition, sleep/wakeup, `mj_step`, render waits, and
controller callback timestamps. A controlled run with offscreen rendering
disabled would help distinguish rendering from other scheduling sources.
Keep human-workspace configuration fixed when comparing latency changes.

The analysis utility now includes the v21
`async_monitor_output_handoff_ms` column in its summary (previously omitted
from the control-event summary even though the raw data was available).

```bash
python3 tools/analyze_async_monitor_timing.py data_log/20260910_184254_949684
python3 tools/analyze_async_monitor_timing.py data_log/20260910_184329_402608
```

The ten analysis/schema regression tests pass after this reporting correction.
This comparison analyzes the supplied completed recordings; it is not a new
simulation or hardware measurement.
