# 接触限制解除后的能量恢复阶段

2026-09-08。适用于 `reachable_cartesian_impedance_controller`。

## 本次修正：名义刚度预测，恢复期间保留能量检查

`20260908_115238_543872` 日志暴露了旧版 v11 的问题：移开人体工作空间后，runtime 仍在恢复，但 monitor 按缩放刚度预测，且碰撞能量拒绝条件随重叠解除而失效。候选计划重新通过，参考 z 从约 0.223 m 继续降到 0.100 m，实际末端没有跟上。

现在 monitor 始终使用配置的名义刚度和名义阻尼，计算预测力矩、关节运动、可达集和能量；初始参考、已提交前缀、intended、failsafe 均遵守此规则。零空间启用时也使用未缩放的名义零空间刚度。独立调用安全监测库而未提供配置名义增益时，调用方须在命令中提供名义增益。

runtime 继续根据实际测量状态执行能量缩放。monitor 内仍计算恢复状态和假设的 runtime 缩放系数，但它们仅用于恢复退出资格与异步交接检查，不参与预测力矩。名义预测与实际缩放控制不是同一控制律，不能把前者直接当成后者的实际轨迹，更不能仅由“名义刚度更大”推导实际可达集一定被包含。

若 monitor 快照处于 Limited / Recovering，或快照当前已经重叠，则在启用 runtime 能量控制时，对完整候选 intended + failsafe 区间继续执行同一能量预算检查。即使人体几何已经清空、显式 assume-clear，或名义预测中途满足退出条件，也不能解除这一候选的能量检查。后续实际进入 Normal 的快照才回到正常的几何门控策略。

因此恢复阶段，大误差造成的名义能量超限会拒绝新候选，继续现有的备份制动/末端保持流程。恢复不是无条件冻结参考：满足完整名义预测能量和关节限制的候选仍可通过。备份制动过程本身也可能有有限参考位移。最终任务目标不被重写。

真实人体几何保持原样：`workspace_distance_now`、`workspace_distance_min`、`monitored_contact_possible`、`contact_relevant_for_energy` 仍表示真实几何；新增的 `monitor_recovery_energy_check_active` 表示恢复期间保留了能量检查。因此该字段为 1 时，即使距离为正、contact_possible 为 0，仍可出现能量拒绝。旧名称 `first_energy_unsafe_contact_interval_index` 和 `worst_case_contact_time` 在恢复期间也可指非几何重叠、但受保留能量检查的区间。

## 状态与退出条件

| `energy_control_phase` | 含义 |
|---|---|
| 0 | 正常运行 |
| 1 | 当前工作空间重叠，能量控制启用 |
| 2 | 恢复阶段；重叠解除或人体观测未知，能量控制继续启用 |

runtime 恢复退出同时要求：

1. 人体工作空间观测有效且当前几何距离为正，或显式无人物标定的 assume-clear 策略有效。缺失/过期观测不会视为空场景。
2. 当前未缩放命令的 `T + Vx + Vn <= energy_recovery_exit_energy_fraction * energy_budget_joule`。
3. 当前和上一周期的能量缩放系数都为 1，且计划刚度、阻尼均已恢复到配置的名义值。
4. 测量关节位置、速度有限且满足 Panda 限制。
5. 正在执行的指令来自已验证计划，具有当前环境 epoch 的退出资格，并对应名义预测明确允许的退出时刻。

退出本身不会额外改变参考、刚度或阻尼。没有新增“必须静止”的要求。默认 `energy_recovery_exit_energy_fraction: 0.95`，合法范围 `(0, 1]`；0.12 J 预算对应 0.114 J 的 runtime 退出阈值。预测能量另外叠加配置的单侧模型误差裕量。

`enable_runtime_energy_scaling: false` 关闭 runtime 缩放、恢复状态和恢复能量保留，用于原有标定/消融；monitor 仍用名义增益，实际几何重叠处的预测能量检查仍然有效。保留既有计划增益插值和力矩变化率限制，没有新增独立的刚度变化率限制。

## 退出资格与异步交接

clear / overlap / unknown 变化会增加 `energy_recovery_epoch`。旧 epoch 指令不得批准恢复退出。已提交指令前缀保留原资格，新候选通过 `restrictEnergyRecoveryExitPermissions()` 只保留名义预测明确发生退出的指令资格；运行时还必须满足当周期的测量条件。

异步结果仍检查来源计划、时效、参考连续性、人体策略、epoch，以及交接时刻的假设恢复状态。恢复状态比较使用 `energy_recovery_runtime_scale` 是否等于 1，而不是名义预测固定为 1 的实际预测增益系数。名义预测与 runtime 偏差可使该交接检查保守地拒绝结果，不能把其状态一致视为动力学轨迹一致。

备份末端保持超出验证时域后，退出资格失效。新任务、取消任务不清除正在进行的恢复。

## 日志 v13

`run_info.txt`：

- `state_log_schema: orthogonal_execution_v13`
- `monitor_gain_policy: nominal_stiffness_and_damping`
- `monitor_recovery_energy_policy: retain_contact_energy_gate_full_horizon`

控制 CSV 包含实际缩放、实际刚度和实际能量，以及恢复状态/退出/epoch/交接字段和 `monitor_recovery_energy_check_active`。当前版本删除了额外关节弹簧的控制与预测实现、状态、参数和记录字段；它不属于本项目的方法或消融项。历史 CSV 和历史诊断图对应采集时的实现，不改写原始实验数据。

当前控制输入由 Cartesian 阻抗、可选普通零空间控制和 Coriolis 补偿组成。若动能本身超过预算，现有缩放律仍会令刚度和阻尼为零；删除额外关节控制不构成耗能制动，也不保证强外力下的闭环稳定性。

预测 CSV 的 `pred_energy_scaling_active` 固定为 0、`pred_energy_stiffness_scale` 固定为 1；预测刚度/阻尼为配置名义值。新增 `pred_energy_recovery_runtime_scale` 记录沿名义轨迹计算的假设 runtime 缩放系数，并新增 `monitor_recovery_energy_check_active`。

预测 CSV 的能量和 q/dq 属于名义控制 rollout，控制 CSV 的 `previous_applied_*` / `*_after_scaling` 属于实际缩放控制。进行预测误差标定时应使用无缩放、控制律一致的数据，不能把恢复期间二者差值直接作为相同控制律的模型误差。控制 CSV 原有 `monitor_current_*_after_scaling` 在实时循环中仍被填为实际缩放后能量；预测 CSV 的 monitor current energy 使用名义增益，二者语义不同。

## 验证与复现实验

回归测试覆盖 runtime 恢复状态机、名义预测不受缩放影响、低增益命令/锚点不能降低验证刚度、名义阻尼和零空间势能、恢复期间几何清空、SaRA 与球形 fallback、intended 和 failsafe 超预算拒绝、低能量恢复退出、退出中途不得解除整段能量检查，以及观测/epoch/交接检查。

编译和测试使用开发容器内独立目录 `/tmp/cps_recovery_build`、`/tmp/cps_recovery_install`，不启动机器人。运行环境需要重新构建、加载对应库：

```bash
colcon build --packages-select \
  cps_human_workspace cps_safety_monitor cps_controllers franka_bringup
source install/setup.bash
```

重新加载后核对 run_info 的 v13 和两个 monitor 策略字段。仿真重放此前的大误差接触释放场景，检查：

1. 人体离开后 phase=2，真实距离转正，但 monitor_recovery_energy_check_active=1。
2. 名义预算不通过的候选保持 candidate_verified=0，轨迹按既有备份制动并保持，不因几何清空就继续推进。
3. 名义预测通过、实测增益/能量等退出条件满足后，才能返回 Normal；新的 Normal 快照才解除恢复能量检查。
4. 同步观察实际/参考位姿、名义预测能量上界、runtime 缩放后能量、实际刚度、力矩变化率和退出资格。

这项修正解决恢复阶段提前放行参考的问题，不保证低预算和摩擦条件下有限时间到达目标；如果保持参考时仍长期无法接近，需要单独分析恢复运动能力。Action 只根据参考轨迹时间判完成的既有问题也不在本次 monitor 修改范围内。

本次删除后的验证：控制器、安全监测库、人体工作空间库、场景依赖及启动配置包编译通过；剩余 50 项安全监测 GTest、3 项异步邮箱 GTest 和工具模型配置生成测试通过。新动态库加载成功且不再导出被删除功能的符号；新安装的 YAML 不含相关参数。控制 CSV 的表头/写入字段均为 198 列，预测 CSV 均为 163 列；`git diff --check` 通过。构建产物位于上述独立目录，未替换运行中的控制器，未启动仿真或机器人。此前发现的仓库格式检查问题未在本次做全库重排。
