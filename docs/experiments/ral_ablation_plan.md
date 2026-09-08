# RA-L experiment plan for the reachable Cartesian impedance controller

Prepared from the workspace and local reference papers on 2026-09-07. This is a proposed protocol, not completed experimental evidence. No controller parameters were changed and no robot motions were executed during this review.

The implementation observations below describe the pre-fix snapshot. On 2026-09-08, the controller gained a stateful energy-recovery phase; see [recovery behavior and validation](energy_recovery.md) before collecting new experiments.

## Recommendation and intended contribution

Start with energy accounting and timing validation, then dedicated prediction calibration, independent validation, and finally ablations. Calibration is necessary, but it cannot repair an incorrect energy definition, a misaligned command/state pair, or incorrect collision geometry.

A useful research question is: **Can predictive verification of total controlled energy preserve fast free motion while enabling compliant contact and reliable recovery under model and timing uncertainty?** Treat this as a hypothesis to test.

SaRA-shield already combines reachability, monitored intended-plus-failsafe trajectories, contact-energy limits, and contact-type classification. Reachability plus an energy threshold alone is therefore insufficient to distinguish this work. See the supplied paper and [SaRA-shield publication](https://arxiv.org/abs/2412.10180).

Lachner et al. already control kinetic plus impedance potential energy through gain scaling and effective trajectory time. See the supplied paper, particularly Eqs. (12)–(18), and [the published article](https://journals.sagepub.com/doi/10.1177/02783649211011639).

The present implementation offers a possible contribution in predicting the closed-loop Cartesian impedance dynamics, verifying total stored energy along a path-consistent backup, and handling asynchronous execution with matched prediction provenance. Novelty relative to other literature still requires a separate related-work assessment. Do not claim that either source paper's proofs automatically transfer to this implementation.

## What is available now

| Component | Current implementation and implication |
|---|---|
| Controller | `cps_controllers/src/reachable_cartesian_impedance_controller.cpp`; 1 kHz configured control, 200 Hz asynchronous monitor. |
| Verification | `cps_safety_monitor/src/reachable_safety_monitor.cpp`; joint-space dynamics rollout, total joint kinetic plus Cartesian/nullspace potential energy, SaRA link capsules and single-hand reachable occupancy. |
| Backup | Intended plus path-consistent failsafe motion; existing cache records generation, stage and command index. |
| Energy intervention | Runtime gain scaling is gated by **current robot/human-workspace overlap**. Predicted future overlap alone does not activate it. |
| Trajectory timing | Verified command progression controls time; the runtime scaling call explicitly does not implement Lachner Eq. (15)'s energy-triggered effective-time freeze. This difference belongs in the method description and baseline design. |
| Current source YAML | Budget 0.12 J, Cartesian gains 1500/500, runtime scaling and recovery enabled, nullspace disabled, beta_K=0, beta_V=0.027 J, SaRA secure radius=0. |
| Calibration | Dedicated `calibrate_monitored_trajectory` action, independently enabled dense logging, capture at a requested path time, then execution of the captured plan's remaining intended segment and failsafe. |
| Analysis | `tools/estimate_energy_error_bounds.py` matches exact sequences and plan provenance. `tools/check_latest_shield_data.py` provides exploratory trajectory diagnostics; its nearest-time matching is not a replacement for exact calibration matching. |
| Scenarios | Free-space line/loop/rotation/mixed motion, table-corner approaches, synthetic moving-hand crossing, compliant table contact, table removal and return. |
| Tool/model | Launch generates tool-specific MuJoCo, monitor URDF and controller YAML from `franka_bringup/config/tools/metal_ball.yaml`. Archive generated artifacts as well as source configuration. |

The current monitor uses one energy budget. I did not find the source paper's complete environment/self-clamping contact classifier in this monitor. Describe it as a SaRA-based implementation unless the full original method is separately integrated and tested.

## Existing data: reasons to validate before ablation

The latest timestamped run inspected was `data_log/20260905_120613_343284`. Its `run_info.txt` records beta_K=beta_V=0, not the current source YAML's beta_V=0.027 J. Thus source settings alone do not establish what generated an old result.

The latest old calibration run, `20260831_124455_817518`, cannot be processed by today's estimator: it lacks `previous_applied_nullspace_potential_energy` and `previous_applied_nullspace_potential_energy_active`. Do not silently fill missing fields and merge old/new calibration data; collect a fresh set with the final implementation.

An offline inspection of the newest ordinary run found:

| Diagnostic | Observed value |
|---|---:|
| Control rows | 55,038 |
| Logged ROS-time span | 54.735 s |
| Control-sequence gaps | 0 |
| Repeated ROS timestamps | 8,558 |
| Maximum current-command post-scaling energy while contact flag active | 0.120000 J |
| Maximum previous-command energy at measured state while contact flag active | 0.139763 J |
| Maximum joint kinetic energy during workspace overlap | 0.450781 J |
| Maximum previous-command energy during workspace overlap | 29.923983 J |
| Maximum cached contact sample age | 37 ms |
| Deduplicated monitor end-to-end latency, median / p99 / max | 1.55 / 9.69 / 22.98 ms |

These are diagnostic values, not verified injury or physical energy-transfer measurements. The contact flag is cached; energy depends on the dynamics model and impedance reference. The large previous-command value occurs around a workspace gate transition, and needs investigation with the actual command, gains, reference error, physical contact and timestamps. Its cause is not established by this audit.

The current-command energy is recomputed after choosing new gains. The previous-command energy evaluates the measured endpoint with the gains/reference that acted during the preceding interval. Plotting only the former can conceal sampled execution deviations. The same-command pairing is already documented in the controller around `previous_applied_energy_terms`.

See [pilot figure](pilot_energy_accounting.png), [vector PDF](pilot_energy_accounting.pdf), and [numerical summary](pilot_energy_accounting_summary.json). Row counts are not independent trials or physical durations. Repeated ROS timestamps mean a control-loop index must not automatically be treated as a distinct 1 ms simulator step.

## Stage 0: accounting, geometry and timing validation

First reproduce a short free-space line, then a controlled approach/contact/release in simulation. Retain current gains, dynamics and tool throughout this initial diagnosis.

1. Validate kinetic energy against simulator mass matrix and measured joint velocity, including tool inertia and armature. Save the actual model artifacts. On hardware this remains a model-based estimate, not a directly measured energy.
2. Validate potential energy using the **same reference and applied gains** as the prediction. Include orientation. Compare pre-scaling, current-command post-scaling, and previous-applied energies separately. Do not treat a gain reduction's algebraic decrease in virtual storage as measured dissipated contact energy.
3. Align control sequence, simulator time, actual controller period, ROS observation stamps, applied-command provenance and prediction endpoint. Investigate repeated ROS times and sample age before assigning milliseconds to the discrepancies. Add actual period and simulator-step identifiers if absent.
4. Verify robot/tool enclosure coverage against actual MuJoCo geometry across configurations. The tool model generator updates the URDF and TCP-related parameters, but the SaRA enclosure configuration is a separate artifact. Verify that its final capsule covers the ball. A zero secure radius needs demonstrated containment; energy margins do not cover geometric error.
5. Use one timestamped human-workspace publisher. For table contact/removal, use the surface-following provider, which follows the actual moving/deforming pad and transforms world coordinates into `panda_link0`. The static YAML source does not establish that the physical pad moved with its virtual hand.
6. Plot overlap-entry and overlap-exit events with gain restoration, pose error, T, previous-applied energy, contact sensor and backup stage. Check whether restoring nominal stiffness with a large remaining error causes stored-energy jumps or repeated switching. Resolve or explicitly model any discovered mechanism before freezing the method.

Acceptance: explanations for the observed discrepancies, exact state/command pairing, verified physical/tool geometry, and independent timestamps/contact observations adequate for the proposed claims. Do not hide deviations by enlarging beta or the robot radius without identifying what uncertainty each parameter represents.

## Stage 1: dedicated calibration

Use `scene:=no_table`, with no physical contact, for the first calibration. Keep the safety monitor enabled because disabling it also disables energy diagnostics and the runtime energy gate.

Set these values **before configuring the controller** in an experiment configuration; retain other final gains, model, tool and scheduling settings:

```yaml
enable_safety_monitor: true
async_safety_monitor: true
enable_error_logging: true
enable_prediction_logging: true
enable_calibration_logging: true
enable_runtime_energy_scaling: false
calibration_assume_no_human: true
calibration_capture_path_time_sec: 0.0
enable_nullspace: false
n_stiffness: 0.0
```

The assume-clear override is appropriate here only because this is deliberately empty simulation. It applies to the dedicated calibration action when no fresh human workspace is available; it does not override a fresh publisher. Alternatively keep it false and publish a verified distant hand. Normal actions remain fail-closed without workspace data.

Most settings are read during configuration. A successful `ros2 param set` response does not establish that their cached controller values changed. Restart/reconfigure and verify `run_info.txt`. The capture-time parameter is specifically read when a new calibration goal is accepted and may be set between goals.

From the sourced workspace, after configuration and controller activation:

```bash
ros2 param set /reachable_cartesian_impedance_controller \
  calibration_capture_path_time_sec 0.0

ros2 action send_goal \
  /reachable_cartesian_impedance_controller/calibrate_monitored_trajectory \
  panda_motion_generator_msgs/action/CartesianViaMotion \
  "$(cat cps_trajectory_generators/config/examples/reachable_via_points_line_action_goal.yaml)" \
  --feedback
```

These are instructions for a future experiment; they were not executed during this review. Wait for successful completion, then let the logger flush. A calibration action captures one monitored plan, not the entire original long trajectory. Repeat from a controlled initial pose at capture times covering acceleration, cruise, a corner/rotation transition and braking. Choose times inside each generated path's duration; the same absolute time is not suitable for every path.

Pilot coverage: line, loop, rotate-in-place and move-and-rotate; three speed settings inside the declared operating envelope; at least three capture phases. Start with 12 low/medium-speed captures to validate the pipeline, then expand to at least 36 captures and additional initial configurations as needed. Hold out entire runs/configurations/phases from fitting. Use low speed first; these counts are planning values, not statistical guarantees.

Process training runs explicitly with full-plan scope:

```bash
python3 tools/estimate_energy_error_bounds.py \
  data_log/TRAIN_RUN_1 data_log/TRAIN_RUN_2 \
  --scope full --guard-factor 1.2 \
  --plot-dir /tmp/cps_energy_calibration_training
```

Replace the uppercase directory placeholders. The estimator requires each supplied directory to contain a complete matched calibration; do not pass every historical run indiscriminately. `--scope auto` may fall back to a committed prefix, which is insufficient to claim full backup-horizon calibration.

For matched horizon h, calculate signed residuals:

\[
r_K(h)=T_{\mathrm{measured,previous}}-\widehat T(h),\qquad
r_V(h)=V_{\mathrm{measured,previous}}-\widehat V(h).
\]

The existing estimator proposes beta_K = 1.2 max(0, max r_K), and analogously beta_V. Its guard factor is a heuristic, not a confidence bound. A zero observed positive maximum does not establish zero uncertainty. Preserve negative residuals in analysis because they reveal conservative predictions.

Inspect residual versus horizon and stage, q/dq prediction errors, matched coverage, missing rows, failed captures and accepted generation identities. Independently check whether measured link occupancy lies inside each predicted enclosure. Position-error norms alone do not prove capsule containment.

Freeze provisional margins after training, then evaluate the inequalities r_K <= beta_K and r_V <= beta_V on held-out complete plans without refitting. Report both component violations and simultaneous total-energy envelope violations, their maxima, and the fraction of independent trials with any violation. The current script estimates new margins; it does not implement a frozen-bound validation report. Use its matched CSV exports to implement that separate comparison.

If a held-out run fails, expand the model or envelope and start a new held-out evaluation. Do not tune on the test set and continue calling it held out. Free-space calibration does not validate external-contact dynamics or gain switching; those require Stage 2.

## Stage 2: validate the full controller before comparative trials

Restore `enable_runtime_energy_scaling: true`, `calibration_assume_no_human: false`, use the normal `follow_cartesian_via_points` action and install the frozen margins. Keep nullspace disabled for the main study. Keep the final scheduling and gain settings fixed.

Run contact onset, sustained contact, table removal/return and dynamic crossing. Verify actual execution against the energy envelope and geometry under scaled gains. The current calibration estimator deliberately filters out runtime-scaling-enabled rows, so normal-mode validation needs a corresponding exact-provenance analysis extension; do not merely delete that filter without accounting for actual scaled references/gains and logged sampling density.

In particular distinguish (a) energy at predicted possible contact, (b) energy during current conservative occupancy overlap, (c) energy during synchronized physical contact and (d) external energy injected into the robot. Lowering stiffness cannot instantaneously remove kinetic energy already above the budget. Define the domain of each claim and report overshoot/recovery when external disturbances violate its preconditions.

The verifier compares interval endpoint energy maxima. Check intermediate simulator/control samples too. A continuous-time upper-bound claim requires a justified intersample bound or sufficiently conservative validated treatment, beyond agreement at selected endpoints.

## Stage 3: comparisons and ablations

Keep robot/tool, path, initial state, human trace, contact mechanics, budget definition, torque limits and base gains identical unless they are the ablated factor. Compare closed-loop executions; replay the exogenous human trajectory for each controller, not one controller's recorded robot state.

| ID | Variant | Question | Availability |
|---|---|---|---|
| F | Full current method after validation | Reference method | Existing configuration |
| A1 | F with beta_K=beta_V=0 | Do residual margins prevent underestimation at a productivity cost? | Existing parameters; diagnostic simulation first |
| A2 | F with runtime energy scaling off; keep predictive energy verification | What does local compliant adaptation add to predictive rejection/backup? | Existing `enable_runtime_energy_scaling`; disables runtime scaling and its recovery latch |
| B1 | Always-active Lachner-style Cartesian energy controller, no predictive intervention | Does predictive/spatial gating improve productivity relative to a global budget? | Requires separate Cartesian baseline preserving same task, dynamics and logging |
| B2 | Reachability-based stop-before-overlap controller | What productivity/contact benefit follows from allowing bounded-energy contact? | Requires explicit candidate rejection on possible overlap |
| A3 | F without predictive rejection, retain current-overlap energy scaling and diagnostics | What does lookahead add beyond reactive contact gating? | Requires decoupled feature switch |
| A4 | Kinetic-only predictive check, retain unchanged total-energy runtime law | Does predicted impedance potential improve anticipation? | Requires explicit prediction-metric switch; internal ablation, not full SaRA-shield |
| B3 | Original SaRA-shield | Comparison to the named prior framework | Reference source exists, but this is not a YAML switch in the present controller |

For A2, report that disabling runtime scaling also disables its recovery latch and retained recovery energy gate. Nominal-gain predictive energy verification at actual geometric contact remains active. This is a combined runtime-adaptation ablation, not an isolated recovery ablation.

Do not use `enable_safety_monitor: false` as an energy-only baseline: `shouldApplyEnergyBudget` and energy tracking depend on it. The separate joint-energy controller also changes task/control coordinates, so it is not automatically a fair Cartesian baseline. B1 is a task-matched gain-scaling/effective-time baseline, not a reproduction of every mechanism in the original paper. Document its exact control law.

Do not emulate B2 by setting the energy budget to zero, which changes both verification and impedance behavior. Do not call A4 original SaRA-shield; the original has link/contact-energy definitions and contact classification absent from this isolated change.

Prioritize F, A1 and A2 for the first simulation ablation, then B1 and B2 for a stronger RA-L comparison. A3/A4 are targeted additions if the primary results leave the prediction or potential-energy contribution unclear.

## Scenario matrix

| Scenario | Existing starting point | What to vary | Primary outcomes |
|---|---|---|---|
| S0: hand absent/distant | Line, loop, rotation and mixed goals in `scene:=no_table` | Speed and posture | Tracking error, completed progress, overhead, unnecessary braking |
| S1: moving hand crossing | `reachable_via_points_dynamic_human_crossing_action_goal.yaml` and dynamic crossing workspace YAML | Hand phase and speed, robot speed | Progress over a fixed time, anticipation, rejected plans, recovery |
| S2: approach and sustained contact | `scene:=table_spring`, static-down and table-corner goals | Approach direction/speed and pad stiffness | Impact peak, impulse, sustained force, energy exceedance, contact duration |
| S3: release and re-entry | `/move_table_assembly`, existing disappear helper | Release phase, fixed dwell, removal/return rate | Gain restoration, kinetic-energy burst, overshoot, time to resume task |
| S4: bounded uncertainty and latency | S1/S2 plus controlled fault injection | Observation age/noise, known model mismatch, worker delay | Envelope coverage, false-safe decisions, safe backup completion, deadline losses |

S1's synthetic hand is a kinematic observation source; it is not by itself a physical colliding hand in MuJoCo. Use it for reachability and productivity. Use S2/S3 with the physical fixture for contact-force claims. For those scenarios start `cps_mujoco_scenarios/launch/surface_following_human_workspace.launch.py` and only one `human_workspace/state` publisher.

The table pad currently has stiffness 200,000 N/m and damping 200 N s/m in `franka_description/mujoco/franka/table.xml`. Validate its displacement/force relation, timestep sensitivity and joint-limit engagement. Robot energy, transferred energy and peak contact force are different quantities. Do not interpret a chosen 0.12 J budget as an injury threshold without application-specific physical validation.

Use the current budget as the pilot nominal value, then a small sensitivity sweep such as 0.06/0.12/0.24 J in simulation, provided each is meaningfully above the frozen aggregate uncertainty allowance and below relevant model/actuator limits. These are proposed research settings, not human-safety recommendations. Vary one factor at a time first; separate robustness within declared bounds from deliberate out-of-bound stress tests.

For a first comparison use 3 variants (F/A1/A2) x 3 scenarios (S1/S2/S3) x 10 paired initial-condition/phase cases = 90 runs. For the main study plan approximately 20–30 independent paired cases per primary scenario/variant, adjusting after pilot variability and the desired effect size are known. Deterministic identical repeats do not supply independent behavioral evidence. Randomize variant order, reset initial robot/fixture state and human phase, and retain failures and timeouts.

Define a task horizon in advance. Report progress at that horizon for all runs, plus completion rate and completion time where applicable. Do not drop timed-out runs from throughput comparisons. Hold force and energy comparisons to the same physical task phase; a stop-before-contact baseline should not appear to have good contact regulation simply because it avoided contact.

For real-robot evidence, rerun calibration with the real dynamics source/tool and a physical surrogate or instrumented fixture. Prioritize replication of S0, S2 and S3 with the full method and the relevant validated comparison. Keep deliberately weakened safety variants and aggressive stress tests in simulation. Hardware validation is a recommendation for this contribution, not a claim that RA-L universally requires hardware.

## Data to retain and analyse

Always retain `run_info.txt`, both CSVs, action goal/result, exact source commit and dirty diff, submodule revisions, source/generated configs and models, initial state, seed/phase, trial ID, and scenario events. The current run metadata is useful but does not replace a time-stamped record of later action goals and live human observations.

| Analysis | Existing signals | Additional evidence needed |
|---|---|---|
| Energy accounting | `joint_kinetic_energy`, potentials before/after scaling, `previous_applied_*`, budget, scale, recovery state | Frozen definition, exact applied references/gains, external energy when claiming disturbance rejection |
| Prediction accuracy | `pred_*`, `_ub`, expected sequence, generation/stage/index, q/dq | Frozen-bound validator; scaled-mode matching; dense intersample checks |
| Reachability | `workspace_distance_now/min`, interval `distance_segment`, `contact_possible`, link index, human reach radius, alpha, secure radius | Simulator truth for robot/tool geometry and human pose; full capsules/observation parameters if replay cannot reconstruct them |
| Task performance | Current/desired pose and velocity, errors, commanded path time/rate, execution stage | Original nominal path and synchronized start, action success/tolerance, timeout, actual completed task progress |
| Contact | `mujoco_contact_value/active/sample_age_sec` | Raw timestamped force at simulation rate, verified units, contact pair/location, pad displacement/velocity, calibrated fixture on hardware |
| Timing | Worker queue/compute/handoff/end-to-end, plan age, input sequence, unusable outputs, counters | Actual control period, simulator step/time, raw observation stamp/age, deadline and backup state at rejection |

The touch value is a sensor output, not a contact-pressure map or vector wrench. Repeated cached values in a 1 kHz CSV do not create 1 kHz force measurements. Measure transferred work only with synchronized force and contact-point velocity/displacement; force alone cannot yield it.

Predefine the following per-trial metrics:

- Energy: maximum positive exceedance, physical duration above budget, and integral of positive exceedance (J s), separately for overlap and physical contact, and for previous-applied versus current-command accounting. Do not call that integral transferred energy.
- Calibration: worst positive component residual, simultaneous total-envelope failure, maximum q/dq errors and geometry escape; stratify by horizon/stage and aggregate by complete trial.
- Contact: peak force, impulse, average/max sustained force in a predeclared post-impact window, duration above an application-validated threshold, peak fixture compression, recovery after removal.
- Performance: normalized completed path/task progress over fixed time, task completion and timeout rates, completion time, geometric path deviation and error against the original nominal trajectory. Error only against the slowed commanded reference can conceal lost productivity.
- Intervention: time in backup, stop count and dwell, time with scaled gains, recovery latency, rejection causes and minimum measured clearance where collision avoidance is the objective.
- Computation: median/p95/p99/max latency, actual activation deadline misses, unusable-plan fraction and log drops. Deduplicate monitor inputs; predictions contain many rows per rollout.

Use paired effect sizes and confidence intervals across independent trials, with trial-level paired bootstrap as one suitable approach. Predeclare a small set of comparisons and adjust multiple tests if reporting p-values. Do not use thousands of correlated control rows as the statistical sample size. Zero failures in N independent trials is observed evidence only; under an independent Bernoulli model its rough 95% upper failure-rate bound is 3/N, not zero.

## Figures and tables for the paper

| Figure | Content | Claim it tests |
|---|---|---|
| 1 | Setup plus block diagram: intended/backup generation, dynamics rollout, occupancy overlap, energy verification, execution/gain adaptation | Defines the actual method and timing/geometry assumptions |
| 2 | Held-out residual versus horizon; separate T/V and intended/failsafe; frozen beta lines and per-trial maxima | Prediction margins cover the declared operating envelope |
| 3 | One preselected representative approach/contact/release trial: aligned distance, predicted bound, previous/current energy, gain scale, force and path progress; switch events marked | Explains the causal interaction of prediction, compliance and recovery |
| 4 | Paired distributions of normalized progress, energy/force exceedance and recovery for the primary variants | Quantifies ablation effects and safety/performance tradeoffs |
| 5, if space | Tradeoff curve over budget or uncertainty; latency tail/deadline-failure inset | Shows robustness and deployability beyond one tuned point |

Use a compact table for the variant definitions and another for aggregate results including failures/timeouts and runtime. Put repeated-trial evidence in the main paper, not just one appealing trace. Select the illustrative trace by a predeclared rule, such as the median full-method progress trial, and show the largest violation separately when relevant.

RA-L currently permits six pages plus up to two overlength pages including figures/references. It welcomes multimedia/data/code, but does not permit extra supplemental text/figures to evade the page limit. Plan for four main figures and compact tables, with video showing synchronized physical behavior and reproducible data/code links. See [RA-L author instructions](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/).

## First practical session

1. Archive the actual installed/generated configuration and reproduce the short energy-accounting diagnostic, especially previous-command versus post-scaling energy and overlap transitions.
2. Verify simulator/control/observation timing and tool/capsule geometry.
3. Run a new no-contact calibration action on a line; require exact full intended/failsafe matches and successful action completion.
4. Expand calibration across motion phases, fit provisional margins, and validate on untouched runs.
5. Validate scaled contact and release behavior using synchronized physical observations.
6. Freeze method, margins and metrics, then run the 90-run F/A1/A2 pilot matrix before investing in the full comparison campaign.
