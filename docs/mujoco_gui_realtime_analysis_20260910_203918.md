# GUI/RViz realtime slowdown — 2026-09-10 20:39

Inputs: controller `data_log/20260910_203918_195954`, outer `data_log/mujoco_realtime/GUI_RVIZ_100Hz_10_10_20260910_183912.csv`. Baseline: headless 100 Hz run `20260910_202914_052659` and its paired CSV. Configuration confirms 100 Hz, 10 committed and 10 fresh intended, dense control step 1 ms. Viewer records are present; offscreen requests are zero. RViz presence is based on the selected experiment, not independently recorded by the CSV.

Timing excludes startup pause and simulation time below 2 seconds. New interval 2.000–20.031 s, approximately 20.395 wall seconds; 18,031 controller rows and 1,803 result events. Outer matching uses simulation time; controller ROS clock delivery makes endpoint alignment approximate. Retain repeated positive ROS timestamps (379 new versus 2 baseline).

| Metric | Headless 100 Hz | GUI/RViz 100 Hz |
|---|---:|---:|
| Simulation progress / wall progress | 99.93% | 88.41% |
| Actual processed outputs per wall second | 99.99 | 88.41 |
| Control start interval mean | 1.001 ms | 1.131 ms |
| Control start interval p99 | 1.182 ms | 4.075 ms |
| Control start interval max | 4.166 ms | 8.613 ms |
| Worker compute mean | 1.200 ms | 1.405 ms |
| Worker compute p99 | 2.022 ms | 3.279 ms |
| Output handoff p99 | 1.024 ms | 3.864 ms |
| Output handoff max | 1.156 ms | 5.725 ms |
| Input-to-consumption p99 | 3.020 ms | 5.913 ms |
| Input-to-consumption max | 3.265 ms | 7.720 ms |

The GUI percentage measures simulated-time progress divided by wall-time progress, not worker CPU utilization or percentage of monitor requests accepted. `viewer.cpp:2518` computes actual realtime as `100 / measured_slowdown`. Paced `physics.cpp:142` updates slowdown over windows at least 100 ms long. Approximately reconstructed 100 ms windows in the matched CSV range from 77.89% to 98.52%, consistent with occasional ~80% UI values. These reconstructed windows do not share exact UI boundaries and do not verify the specific reported 75% instant.

Lowering monitor frequency reduces one worker's request rate; physical integration and control still require a step per nominal 1 ms. The monitor is scheduled every 10 control cycles, not from an independent 100 Hz wall timer. With simulated/control progress at 88.4%, wall-time request throughput is correspondingly about 88.4 Hz. At 80% progress it would be approximately 80 Hz over a comparable interval.

Among 809 physical start intervals above 2 ms, 690 (85.3%) have wakeup lateness above 1 ms. Only two include physics-lock misses. In 108 cases the previous iteration's work exceeded 1 ms; categories overlap. This locates much of the slowdown in late physics wakeups, with some long iteration work also contributing. Mean physical iteration work is 0.144 ms, p99 0.648 ms, max 4.913 ms; mean controller previous-execution time is 0.033 ms, max 2.322 ms. Low average compute cost therefore does not ensure uniform 1 ms call spacing. Viewer trace covers its synchronization section, not complete draw/GPU/SwapBuffers time; these data do not distinguish RViz, MuJoCo drawing, GPU-driver effects, OS scheduling or CPU-frequency/cache interference as the ultimate cause.

A relevant pacing choice exists in `realtime_support.hpp:41`: next deadline is `previous + period` if still in the future, otherwise `now + period`. The current paced path executes one physics step and does not immediately batch catch-up steps after an overrun. When a late wakeup or long iteration has passed the next deadline, the code schedules another full period after completion. This avoids catch-up bursts but permits lost simulation progress to persist. It helps explain why scheduling jitter reduces realtime ratio despite spare average CPU capacity. Any change here must evaluate both realtime ratio and control-call spacing; increasing average speed alone could reintroduce bursty controller calls.

All 1,803 measured results arrived within 1–6 control steps, within their 10-step prefix, with maximum end-to-end under 10 ms. There are 43 end-to-end events over 5 ms and three handoffs over 5 ms. Thus this run meets the observed prefix deadline while failing to maintain 1x simulation speed. These are separate properties.

Full controller recording has 2,011 result events, 1,977 verified, 1,970 accepted verified, seven verified rejections and 34 unverified candidates. Verified rejection masks are 2:5, 4:1, 6:1 (recovery epoch/state); none are source-generation or activation-window/continuity failures. All unverified predictions have possible contact and energy upper bound over 0.12 J, with no joint-limit violations. No CSV drops, mailbox overwrites, stale-worker drops or skipped control-sequence schedule slots were recorded. Footer processed count exceeds consumed count by one at shutdown; this alone is not evidence of runtime packet loss.

Next diagnosis should focus on physics-thread scheduling, visualization refresh workload, and the overrun deadline policy. A MuJoCo-window-only comparison can isolate the added RViz effect. The current experiment does not justify further lowering monitor frequency as the primary remedy or assert that affinity alone will solve it. Preserve visualization as a required product capability while testing these factors separately.

Detailed statistics: `data_log/mujoco_realtime/analysis_20260910_203918/gui_rviz.json` and `headless.json`. No control or physics code was modified during this analysis.
