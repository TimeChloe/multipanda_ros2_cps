# A–D MuJoCo and controller timing analysis — 2026-09-10

## Files and comparison method

The original A–D files were in the container at `/home/developer/multipanda_ws/data_log/mujoco_realtime`. Only `/home/developer/multipanda_ws/src` is mounted to this repository. They have been copied to `data_log/mujoco_realtime/` without changing the container copies.

| Case | Controller run | Observed mode |
|---|---|---|
| A | `20260910_194704_814662` | One step/iteration; no viewer/render samples |
| B | `20260910_194832_554179` | Legacy batches, up to five steps; no viewer/render samples |
| C | `20260910_194920_771891` | One step/iteration; viewer samples |
| D | `20260910_195024_130330` | One step/iteration; viewer samples; no offscreen requests |

Mapping uses the outer steady-clock time spans converted to host local time, the controller directory times, and the observed pacing/viewer signatures. The extra 19:44:10 run predates A and is excluded.

Timing statistics below exclude initial simulation time below 2 seconds and startup pause. Outer rows use `running=1`; the end is clipped to the matching controller recording. Controller `wall_time_sec` actually comes from the ROS node clock. Repeated positive stamps are retained: they do not prove a physical pause. Boundary alignment is approximate within ROS clock-delivery latency. Raw outer summaries using the original 2-wall-second warmup are kept separately.

## Main results

| Metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Handoff mean (ms) | 0.561 | 0.586 | 0.662 | 0.671 |
| Handoff p99 (ms) | 0.984 | 1.270 | 3.557 | 3.473 |
| Handoff max (ms) | 1.683 | 1.605 | 5.585 | 5.858 |
| Control start interval p99 (ms) | 1.199 | 1.432 | 3.831 | 3.872 |
| Worker compute mean (ms) | 1.297 | 1.184 | 1.542 | 1.611 |
| Worker compute p99 (ms) | 2.084 | 2.060 | 3.642 | 3.683 |
| End-to-end p99 (ms) | 3.073 | 2.780 | 5.586 | 5.602 |
| Simulation seconds / wall second | 0.9990 | 1.0000 | 0.8896 | 0.8854 |
| Handoffs > 2 ms | 0 | 0 | 195 | 180 |
| Handoffs > 5 ms | 0 | 0 | 2 | 5 |
| Actual processed outputs / wall second | 199.8 | 200.0 | 177.9 | 177.1 |

A provides the most uniform control and the lowest handoff p99. B still meets approximately realtime speed but retains short catch-up intervals (minimum controller interval 0.043 ms); its lower mean worker/end-to-end time does not demonstrate a faster implementation, because its rollout workload is smaller. All new cases include the event-thread/executor/render-submission fixes, so A versus B cannot isolate the old event-spin bug alone.

Compared descriptively with the older 18:43 run (handoff mean 1.921 ms, p99 9.811 ms), A is about 71% lower in mean and 90% lower in p99. The earlier human-observation conditions differ, so this is not a controlled estimate of an individual patch effect.

## Startup pause is not running-time handoff

The unfiltered A log reports handoff p99 32.296 ms. There are 518 controller records at simulation time zero and 104 processed outputs above 5 ms in that phase. The paused physics path executes `mj_forward` at roughly 30 Hz, so the controller can consume results approximately every 33 ms even though physics is not advancing. After startup, A and B have no handoffs above 5 ms. C and D retain only 2 and 5 such events respectively in the measured running interval.

## GUI cases: wakeup latency dominates the remaining stalls

| Running physics interval > 2 ms | A | B | C | D |
|---|---:|---:|---:|---:|
| Count | 5 | 10 | 807 | 770 |
| Fraction of running outer samples | 0.032% | 0.064% | 4.583% | 4.714% |
| Also wake_late > 1 ms | 2 | 0 | 706 | 658 |
| Also had physics lock misses | 0 | 0 | 1 | 2 |
| Previous outer execution > 1 ms | 3 | 3 | 94 | 101 |

For C, 706/807 (87.5%) long physical intervals have wakeup lateness above 1 ms. For D it is 658/770 (85.5%). Only 1/2 of these intervals have any physics-mutex acquisition failures. Viewer instrumented lock-hold maxima are 0.042/0.054 ms. This strongly locates most residual long intervals in late wakeup rather than the measured Viewer critical section. The category counts overlap and must not be added as disjoint causes. Legacy B does not instrument relative-sleep overshoot, so its zero wake_late values do not establish zero scheduling jitter.

C/D simulate at approximately 0.890/0.885 real time: 10 simulation seconds need about 11.24/11.29 wall seconds. Their actual monitor processing rates are about 178/177 Hz because monitor scheduling is every five control cycles, not an independent 200 Hz wall timer. This is an outer pacing/scheduling limitation, not evidence that the worker is generally unable to finish a 5 ms task.

The experiment associates opening the viewer with late wakeups and slower work. The trace does not time GUI drawing/SwapBuffers or record per-core frequency/GPU-driver scheduling, so it cannot determine which of those mechanisms dominates. Each condition was run once; use repeated trials before attributing small A/B or C/D differences.

## D did not exercise offscreen camera rendering

All four CSVs have zero render requests and zero busy submissions. D has tiny snapshot-check timings but no render-channel records. Therefore D does not measure actual offscreen camera rendering load, and the C/D similarity cannot establish that offscreen rendering is free or prove the new submission policy under real camera load. A model with active cameras and actual requests is needed for that test.

## Worker computation variability

Worker CPU time accounts for approximately 99.9% of aggregate worker elapsed time. Mean non-CPU time is roughly 0.0012–0.0019 ms. The higher GUI-case computation tails are therefore mostly on-CPU variation, not time waiting to receive a result. There are rare exceptions, such as D non-CPU maximum 0.723 ms.

Full-recording computation at the same 130-step rollout length:

| Case | Samples | Mean compute (ms) | p99 compute (ms) |
|---|---:|---:|---:|
| A | 321 | 2.064 | 2.144 |
| B | 262 | 2.049 | 2.620 |
| C | 279 | 2.475 | 4.292 |
| D | 249 | 2.531 | 5.298 |

This controls rollout length but not all state-dependent work. CPU frequency, cache/memory interference and other on-CPU differences remain hypotheses, not measured causes. B has a smaller average full-recording rollout (71.3 steps versus A 78.3), explaining why average compute alone is an unreliable A/B ranking.

## Rejections: full controller recordings

| Case | Verified | Accepted verified | Rejected verified | Unverified energy candidates |
|---|---:|---:|---:|---:|
| A | 3297 | 3268 | 29 | 304 |
| B | 3911 | 3901 | 10 | 281 |
| C | 3655 | 3635 | 20 | 277 |
| D | 3396 | 3377 | 19 | 277 |

All four have zero source-generation rejections and zero activation-window/continuity rejections. The remaining verified rejections are recovery-epoch/state changes: A masks 2:24, 6:5; B 2:10; C 2:14, 4:2, 6:4; D 2:13, 6:6. Mask 6 combines epoch and state and must not be counted twice. The 409-source-version failure is not recurring.

Prediction logs independently confirm that every unverified candidate in these four recordings has contact possible, no joint-limit violation, and worst-case energy above 0.12 J. These are separate from the already-verified results rejected during handoff.

Human movement ranges are broadly consistent: x=0.45, z=0.30, y about -0.32 to 0.0 m. Environment phase occupancy and rollout workloads differ; raw rejection totals cannot alone rank scheduling policies. The actual publisher period is not recorded in these CSVs.

## Additional clock-consistency observation

The controller ignores the supplied update timestamp and uses `get_node()->now()` for its internal `wall_time`. In simulation that clock is delivered asynchronously via `/clock`. After the 2-second cutoff, A/B/C/D contain 4/473/462/391 repeated positive clock stamps. These records are retained in the analysis. A follow-up should review use of the controller callback time for control progression while keeping observation timestamps in a consistent clock domain; simply changing one clock field without auditing consumers is not justified by this analysis.

## Recommended next steps

1. Use A for realtime controller benchmarking. It maintains approximately realtime progression and handoff p99 below 1 ms in this run.
2. If the viewer is required, repeat C with physics on a separate physical CPU core from the monitor, verify actual affinity, and add GUI Render/SwapBuffers wall and CPU timing. Do not infer actual core isolation from a configured CPU number alone.
3. For A residual outliers, inspect synchronous /clock publication: its maximum is about 3.45 ms, and three of the five >2 ms physical intervals follow >1 ms outer execution. Most calls are short; tail timing matters.
4. Repeat D with actual camera requests and confirm nonzero render-channel samples before comparing offscreen performance.
5. Repeat each case three times with the same human and robot input and retain matched raw controller/outer logs.

To save future CSVs directly into the local repository from the container:

```bash
cd /home/developer/multipanda_ws/src
mkdir -p data_log/mujoco_realtime
```

Use `$PWD/data_log/mujoco_realtime/...csv` in the launch timing path from that directory. Detailed machine-readable results are under `data_log/mujoco_realtime/analysis_20260910/`.
