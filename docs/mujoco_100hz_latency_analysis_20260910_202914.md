# 100 Hz, 10/10 latency analysis — 2026-09-10 20:29

## Inputs and method

New controller recording: `data_log/20260910_202914_052659/`.
Matched outer timing: `data_log/mujoco_realtime/A_100Hz_10_10_20260910_182906.csv`.
Previous 200 Hz, 5/5 recording: `20260910_201652_477616`, paired with `A_5_5_20260910_181645.csv`.

The new run footer confirms 100 Hz, monitor period 10 control cycles, committed 10, fresh intended 10, and dense step 1 ms. Post-warmup result rows all contain a 10-command prefix and 20-command total intended container. Outer records contain only physics/event channels, no viewer/render samples or rendering requests, and one physics step per running iteration. This supports headless operation; external RViz state is not logged.

Timing below excludes startup pause and simulation time below 2 seconds. New matched interval: 2.000–18.752 simulation seconds, 16,752 control rows, 1,676 processed result events, approximately 16.762 wall seconds. Previous interval: 2.000–17.804, 15,804 control rows and 3,161 events. Positive repeated ROS timestamps are retained (new: 2, previous: 0). Controller timestamps are ROS elapsed time; outer matching endpoints are approximate within clock-delivery latency. Quantiles use nearest rank. Rejection totals explicitly identified as full-recording totals include startup.

## Latency decomposition, new run

| Phase | Mean ms | p99 ms | Maximum ms |
|---|---:|---:|---:|
| Input publication to worker start | 0.136181 | 0.209379 | 0.247075 |
| Worker computation | 1.199666 | 2.022202 | 2.384183 |
| Output ready to controller consumption | 0.580732 | 1.023937 | 1.155945 |
| Input publication to result consumption | 1.916579 | 3.019760 | 3.264942 |

The three phase means sum to end-to-end mean. Phase quantiles/maxima belong to different events and must not be added. The slowest end-to-end event, at simulation time 9.794 s, consists of 0.128289 ms queue wait + 2.310679 ms computation + 0.825974 ms handoff = 3.264942 ms; it has no rejection mask.

All 1,676 results were consumed within 1–3 control steps: 294 within one, 1,233 within two, and 149 within three. None exceeded the actual 10-step committed prefix; none exceeded even 5 ms end-to-end. There were no handoffs above 2 ms. The observed maximum leaves substantial margin relative to the nominal 10 ms prefix, but is not a worst-case execution-time guarantee.

Mean worker non-CPU time was 0.001273 ms, maximum 0.004875 ms. Worker waiting/preemption is not a major component of these measured compute durations. Handoff remains approximately half a 1 ms control interval on average, consistent with result consumption by the next controller callback, rather than consumption only once per 10 ms monitor period.

## Comparison with 200 Hz, 5/5

| Metric | Previous 200 Hz, 5/5 | New 100 Hz, 10/10 |
|---|---:|---:|
| Worker compute mean | 1.013 ms | 1.200 ms |
| Worker compute p99 | 1.788 ms | 2.022 ms |
| Handoff mean | 0.498 ms | 0.581 ms |
| Handoff p99 | 1.007 ms | 1.024 ms |
| Handoff maximum | 2.009 ms | 1.156 ms |
| Request-to-consumption mean | 1.653 ms | 1.917 ms |
| Request-to-consumption p99 | 2.835 ms | 3.020 ms |
| Request-to-consumption maximum | 3.173 ms | 3.265 ms |
| Control start interval p99 | 1.162 ms | 1.182 ms |
| Control start interval maximum | 3.291 ms | 4.166 ms |
| Processed results per wall second | 199.94 | 99.99 |
| Simulation/wall progress ratio | 0.99957 | 0.99933 |

Reducing request frequency did not lower per-request latency in this trial. Average complete prediction workload grew from 59.84 to 68.11 steps across the full recordings, consistent with a larger intended container and state-dependent braking length. New rollout-length/compute correlation is 0.99252. This supports workload as a major cause of compute variation, without proving identical cost at identical robot states.

Estimated worker-compute CPU consumption per wall second is approximately 119.8 ms versus 202.3 ms previously, a reduction of about 41%. This is the measured computation portion of one worker, equivalent to about 12.0% versus 20.2% of one CPU; it is not total process CPU utilization and excludes work outside the instrumented compute interval.

Each condition has only one run. Human center ranges and final commanded path time (8.558708292 s) match closely, but clear-environment cycle shares differ (new ~34.4%, previous ~25.1%). Small latency tails and rejection differences cannot be attributed solely to the frequency change.

## Request cadence and remaining outer-loop delay

All 1,675 consecutive post-warmup request gaps are exactly 10 control cycles. Reconstructing wall-time spacing by summing logged control-start intervals between publication rows gives:

- Mean 10.006667 ms; median 9.999952 ms.
- p99 10.168100 ms; maximum 13.455933 ms.
- Minimum 9.222525 ms, consistent with small pacing corrections.

The maximum request gap spans simulation time 9.721–9.731 s and contains the largest outer physics start interval, 4.114901 ms at 9.727 s. That outer row reports wakeup lateness 3.173843 ms, no lock misses, and only 0.091643 ms work in the previous iteration. The next two longest reconstructed request gaps also span observed late-wakeup events.

There are four matched physical start intervals over 2 ms, none over 5 ms. Three have wakeup lateness above 1 ms; all four have zero lock misses. The fourth (2.049567 ms at 2.453 s) follows an iteration taking 0.990386 ms, of which clock publication took 0.909068 ms. Exact attribution of that fourth interval requires the deadline/pacing path as well as the current trace; it should not be called a worker compute stall.

Recorded controller previous-execution time averages 0.023326 ms, p99 0.071641 ms, maximum 0.633117 ms. Matched outer physics iteration work maximum is 0.990386 ms. The measured controller computation does not explain the 4.166 ms controller call-spacing maximum. Residual tails are primarily external call scheduling/physics pacing in this trace; per-core frequency, operating-system causes, and other competing processes are not logged.

Thus zero late schedule counters mean no missed control-sequence slots, not a guarantee that each request occurs exactly every 10 wall milliseconds. The average 100 Hz rate coexists with occasional wall-time jitter.

## Receipt latency versus fresh-command timing

The 1.917 ms mean end-to-end measurement ends when the controller consumes the result. It does not measure sensor-to-action latency. With a 10-step prefix, fresh intended commands begin approximately 10 control steps after request publication, even if the result is received after two steps; remaining prefix commands still execute first. Previously that prefix was five steps.

In addition, an observation arriving between scheduled requests can wait until the next request to be included. These are architectural timing effects, not measured communication failure, and do not mean runtime energy adaptation stops between monitor requests. Actual sensor-to-action latency needs observation and command timestamp instrumentation in a consistent clock domain.

## Rejections and fallback cross-check

Full new recording: 1,880 result events, 1,844 verified, 1,836 verified and accepted, 8 verified and rejected, 36 unverified. All 8 verified rejections are recovery-epoch mismatch (mask 2), and all arrived in approximately 1.986–2.032 ms. Source-generation and activation-window/continuity rejections are both zero.

Independent prediction rows confirm all 36 unverified candidates have possible contact, energy upper bound above the unchanged 0.12 J threshold (0.120011002–0.181974368 J), and no joint-limit violation. The full-recording unverified rate is 1.91%, versus 1.69% previously; lower raw counts alone are not evidence of fewer failures when request frequency is halved.

After warmup, failsafe episodes are fewer (42 versus 63), but average episode duration grows from 5.08 to 10.48 control cycles, maximum from 10 to 30, and total failsafe cycle share from 2.02% to 2.63%. There are no hold cycles. These are stage-label durations, not evidence of complete physical stops; neither rejection nor braking figures indicate communication deadline failures.

The lifecycle footer reports zero CSV drops/schema mismatches, mailbox overwrites, stale-compute drops, busy deferrals, schedule-late cycles or skipped slots; all 1,886 lifecycle requests were processed and consumed. The CSV contains 1,880 because its recording starts at the first valid via-points command.

## Startup caveat and conclusion

The full recording includes 40 control rows at simulation time zero. Full unfiltered handoff maximum is 32.456497 ms and end-to-end maximum 33.252572 ms, from the paused startup phase. Running-time maxima reported above exclude that phase. Comparing unfiltered p99 against the previous run would be misleading because startup pause contributes a different fraction of result events.

The new configuration successfully runs near 100 Hz and consumes all measured outputs well within its 10-step prefix. It reduces aggregate worker compute load but does not demonstrate faster response than 200 Hz, 5/5. Remaining wall-time jitter appears in physical/controller call spacing, while verified rejections are recovery-state invalidations. Keep the current configuration for repeated trials if reduced CPU load is the objective; use repeated paired recordings to decide whether that tradeoff is preferable to the shorter update/prefix timing of 200 Hz.

Machine-readable statistics and event details: `data_log/mujoco_realtime/analysis_20260910_202914/{new_100Hz,previous_200Hz,latency_details}.json`.
