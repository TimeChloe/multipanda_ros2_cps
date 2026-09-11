# Latest pacing-fixed experiment — 2026-09-11 17:04

Inputs:
- `data_log/20260911_170453_813347/run_info.txt`
- Controller validation and shield prediction CSVs in that directory.
- `data_log/mujoco_realtime/GUI_RVIZ_pacing_fixed_20260911_150441.csv`.

The timing filename uses container UTC; the controller directory uses Europe/Berlin.
Metadata confirms asynchronous monitoring, 100 Hz, 1 ms control grid, 10 committed
and 10 fresh intended steps, strict prefix deadline, and accept-before-publish
with a single outstanding request. The user supplied the `LP_NUM_THREADS=4`
launch invocation; that environment variable is not independently recorded in
these CSVs. Physical starts have a 0.501 ms minimum in the matched window,
consistent with the repaired half-period pacing floor.

## Clock alignment and measurement window

Controller `wall_time_sec` is ROS time since controller activation, not absolute
simulation time or steady wall time. Here activation occurs approximately
10.280 simulation seconds after simulation starts. Matching the CSVs directly
on their numeric time columns would select the wrong physical interval.

We aligned all 42,148 consecutive control rows with consecutive physics rows
using their independently measured start-interval sequences. The alignment
correlation is 0.975898; the next-highest competing offset is 0.033722. Median
absolute interval difference is 0.005131 ms. Differences are expected because
controller entry occurs inside `mj_step`, after physics iteration entry.

The recorded window is controller ROS elapsed 5.561–47.708 s, corresponding to
physics simulation time 15.841–57.988 s and approximately 42.153217 wall seconds.
Logging starts at the first valid via-point command, so no extra warmup trimming
is applied to this already advancing recording. Repeated ROS clock timestamps
are retained. Small endpoint offsets do not affect the conclusions below.

## Result delivery and rejection accounting

Full controller-lifecycle counters:
- Inputs published = worker processed = outputs consumed = **4772**.
- Input/output overwrites, stale-before/after-compute drops, busy deferred cycles,
  late scheduling cycles and skipped schedule slots: **0**.
- Control/prediction CSV drops and schema mismatches: **0**.

The command recording contains **4216** received result events; the preceding
556 lifecycle results occurred before command-triggered recording began.

| Recorded result category | Count |
|---|---:|
| Verified and accepted | 4129 |
| Verified but not accepted | 1 |
| Unverified | 86 |
| Total | 4216 |

Of 4130 verified results, 99.9758% were accepted. The only handoff rejection is
mask 2, `recovery_epoch`, at controller elapsed 36.388 s, input sequence 3640.
Its request was published at control cycle 36386; receipt was cycle 36388.
End-to-end latency was 1.991965 ms. The energy-recovery epoch changes from 10 to
11 at receipt, with the environment changing from clear to overlap and control
phase entering limited mode. The output was verified for the earlier recovery
context, so the controller invalidates it. This is state-consistency protection,
not late delivery or source-plan publication ordering.

Source-generation rejection count is **0**. Strict-prefix deadline rejection
count is **0**. All received results arrive after 1–6 control cycles, within the
10-step committed prefix. The previous source-version race and deadline-related
handoff failures are not observed in this run. This is evidence of a functioning
pipeline in this experiment, not a guarantee under all future loads. These
counters describe the controller/worker mailboxes, not every ROS topic or network
transport in the system.

Prediction logging independently contains 4212 unique candidates:
- 82 unverified candidates all have possible contact and energy upper bound
  **greater than 0.12 J**: range 0.120075525–0.223031010 J.
- None of those 82 has a joint-limit-unsafe verdict.
- The other four unverified received outputs have no prediction rows because
  candidate construction failed before evaluation: reason 9,
  `failsafe_sample_invalid`. Their input sequences are 871, 872, 3270 and 3271,
  at controller times 8.702, 8.712, 32.687 and 32.697 s. Their result latencies
  are 0.900–1.199 ms and all handoff masks are zero. This is a braking-sample
  validity/limit-check issue, not lost communication. The current logging does
  not identify the exact failing sample/component; that requires a separate
  planner diagnosis. Do not classify these four as energy exceedances.

## Latency

All entries below measure the recorded result events in milliseconds. P99 uses
nearest-rank quantiles; maxima from different rows should not be added together.

| Phase | Mean | P99 | Maximum |
|---|---:|---:|---:|
| Input publication → worker begins | 0.140 | 0.404 | 1.878 |
| Worker calculation | 1.528 | 2.764 | 3.592 |
| Worker finishes → controller receives (handoff) | 0.503 | 2.124 | 3.800 |
| Request publication → controller receipt | 2.171 | 4.001 | 5.529 |

There are 47 handoffs above 2 ms, none above 5 ms. Eleven end-to-end results
exceed 5 ms, none exceed 10 ms. The actual acceptance deadline remains expressed
in control steps; all requests are within it even during bounded pace recovery.

Worker mean thread CPU time is 1.526726 ms against elapsed 1.528312 ms. Mean
non-CPU time is 0.001585 ms, maximum 0.006819 ms. In this run worker calculation
variability is dominated by computation rather than preemption/blocking:
rollouts span 21–123 intervals, with correlation 0.966 between rollout length
and compute time. Reachability/monitor evaluation accounts for about 79.5% of
worker elapsed time, planning 11.9%, plan assembly 7.7%.

These timings stop at result receipt. Human observation acquisition/publication,
waiting for the next monitor request, and waiting for committed commands before
fresh intended motion are separate contributions to observation-to-action delay.

## Simulation and control scheduling

- Matched simulation/wall progress: **99.9853%**.
- Received monitor outputs per wall second: approximately **100.02 Hz**.
- Actual request intervals: mean 10.0003 ms, P99 12.0613 ms, maximum 16.6627 ms.
  Cadence is nominally 10 control steps. One five-step interval accompanies a
  new trajectory resetting commanded path time at cycle 25976; the command
  ingress code intentionally starts a fresh monitor phase. No schedule slots
  were skipped.
- Physical start intervals: mean 1.000146 ms, P99 2.171594 ms, max 5.897200 ms.
- Controller start intervals: mean 1.000146 ms, P99 2.222411 ms, max 5.970853 ms.
- Controller execution: mean 0.034289 ms, P99 0.123168 ms, max 5.711695 ms.

Average real-time speed has recovered, while isolated scheduling/execution
outliers remain. The half-period floor applies to physical iteration starts;
controller callbacks occur inside those iterations, so their measured interval
can occasionally be shorter than 0.5 ms when entry offsets change.

Across the larger physical recording, excluding only the first two running
wall seconds, progress is 99.9498%. A physical interval of 18.595539 ms occurs
at simulation time 10.281 s, before the command recording and near controller
activation. The preceding iteration spends 18.590023 ms in work (18.476256 ms
in `mj_step`). This must not be reported as a monitor handoff stall in the
later recording; it also means the entire launch is not free of long outliers.

The recorded execution has 887 braking-stage cycles, 637 prior-intended cycles,
40621 current-verified cycles and 3 hold cycles. Stage labels alone do not imply
communication failure; use the actual rejection and failure diagnostics above.

## Assessment

Keep the current 100 Hz and automatically derived 10/10 settings. This run
supports that the previous version-ordering and delayed-handoff issues are
resolved for the tested workload. Remaining items are scheduling tails and the
four invalid braking candidates, not a need to lengthen the controller horizon
or reduce monitor frequency again. No controller or simulator source was changed
for this analysis.

Full statistics and rejection details:
`data_log/mujoco_realtime/analysis_20260911_170453/result.json`.
