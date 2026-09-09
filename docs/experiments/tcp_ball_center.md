当前控制参考统一为金属球球心 TCP（日志版本 `orthogonal_execution_v15`）。位置目标、位置误差、Cartesian 势能、雅可比和工具球几何使用同一点，不再配置独立的 `ee_collision_center_offset`。

在 [metal_ball.yaml](../../franka_bringup/config/tools/metal_ball.yaml) 中配置：

```yaml
ball_radius: &ball_radius 0.03
mount:
  xyz: [0.0, 0.0, 0.03]
  rpy: [0.0, 0.0, 0.0]
```

`mount.xyz` 是球心相对于 `panda_link8` 的位置。省略独立 `tcp` 配置后，工具生成器从 bounding-sphere center 自动得到 TCP。此工具的 bounding-sphere center 为工具局部原点，因此 TCP 就是 mount 的平移位置。

`ball_radius` 的 YAML anchor 同时供 visual、MuJoCo physical collision 和 safety sphere 使用。`inertia: solid_sphere` 按当前质量和物理半径生成 `I = 2 m r² / 5`，因此修改半径时不需要手动修改惯量矩阵。质量保持单独配置；这里假设均匀实心球。

默认 TCP offset 为 `[0, 0, 0.03]` m，半径为 0.03 m。控制器参数仍使用 `tcp_offset` 和 `ee_collision_radius`，正常 launch 从工具 YAML 自动生成它们。用户提供显式 TCP 时，生成器要求其与 safety sphere center 重合；不一致的旧球面 TCP 配置会报错。

SaRA 检查保留原机械臂七个胶囊，并增加一个以 TCP 为中心的工具球占据集：

- 包络索引 0–6：机械臂原有胶囊；索引 7：TCP 工具球。
- 当前零时长占据的工具球中心为 TCP，半径为 `ee_collision_radius + sara_secure_radius`。
- 预测区间 `[t0, t1]` 使用 SaRA 球形特例：中心取两个 TCP 端点的中点，半径增加 TCP 端点距离的一半及 `alpha[6] * dt² / 8`。
- `alpha[6]` 同时取原第七胶囊和 TCP 的速度模长变化率上界，沿完整预测时域取最大值。保持七个 alpha 条目，输出八个占据集。
- 新增工具球参与实时重叠判断、预测区间判断及现有可视化。机械臂检测仍然启用。

本项目的 SaRA Panda 配置安装自 `cps_safety_monitor/config/robot_parameters_panda.yaml`。其第七关节平移使用 0.088 m，与当前 URDF/MuJoCo 一致；reference checkout 使用 0.080 m。包络形状沿用参考配置。TCP 从第七关节坐标系通过 Panda 固定 joint8 的 `[0,0,0.107]` m 平移和 `tcp_offset` 得到。

能量中的弹性项为球心 TCP 的六维位姿误差 `V = eᵀ K e / 2`。动能仍为完整关节动能 `T = dqᵀ M dq / 2`，不会因为选定球心而改成只有工具球的动能。半径通过工具惯量和接触包络发挥作用，不是能量公式额外乘上的系数。

v15 控制 CSV 删除重复的 `collision_center_p*`、`collision_center_v*` 六列，共 192 列。`cur_p*` 是实测 TCP 位置，`cur_v*`/`cur_w*` 是 TCP 线速度/角速度；`des_p*`/`des_v*` 是 TCP 指令位置/线速度。每个物理量只记录一份。

预测 CSV 共 163 列：`actual_collision_p*` 改名为 `actual_tcp_p*`，`collision_target_p*` 改名为 `tcp_target_p*`。`actual_collision_distance` 仍表示机器人与人体包络间的距离，因此保留碰撞含义。预测位置列仍属于日志函数的笛卡尔近似预测；预测关节状态和能量来自关节 rollout，此次没有改变这一来源区别。

`tools/check_latest_shield_data.py` 在读取入口统一不同版本的字段，分析输出使用 `measured_pz`、`target_pz` 等通用名字，图例不再使用旧碰撞中心字段名。对 v15/v14，预测参考点是统一 TCP；对更早的双中心日志，预测误差仍使用当时的碰撞点，实际指令误差仍使用当时的 TCP，不能把两者强行当作同一点。历史 CSV 不回写。

预测日志队列删除的仅是 `PredictionLogRecord.current_orientation`、`K_runtime`、`D_runtime` 三个未被写入计算使用的副本，以及 `logShieldPredictionTrajectory()` / `writeShieldPredictionTrajectory()` 的对应参数。异步 monitor 输入 `AsyncMonitorInput` 中同名字段仍保留，用于候选计划构建和预测。日志函数继续用 `K_base_`/`D_base_` 绘制标称预测，控制日志继续记录实际的 `Kx/Ky/Kz/Dx/Dy/Dz` 和缩放系数。

`run_info.txt` 增加：

```text
state_log_schema: orthogonal_execution_v15
control_reference_point: tcp_ball_center
collision_reference_point: tcp
robot_reach_capsule_layout: indices_0_to_6_arm_index_7_tcp_sphere
```

重新构建 `cps_safety_monitor`、`cps_controllers`、`franka_bringup` 和 `franka_description`，加载新安装环境并重新启动 launch，才能让控制库、工具模型和生成配置同时生效。只在 YAML 改半径，也需要重新启动 launch 生成模型，不能把它当作运行时在线参数更新。

旧 action 目标的位置原来表示球面点，现在表示球心。不要直接把过去的球面接触高度当成新的球心高度。例如，水平表面高度为 `z_surface`，球位于表面上方时，无穿透接触对应 `TCP_z = z_surface + radius`。此迁移不自动改写已有任务目标。

新增包络和坐标统一不改变能量缩放、恢复退出或标称增益预测策略，也不构成对模型误差或连续时间闭环安全的额外证明。

默认单臂仿真实验的 TCP 基准为 `panda_link0` 中的 X=0.40 m。`franka_sim.launch.py` 的初始关节角为 `[0, -0.320210883515, 0, -1.544505894256, 0, 1.224295010741, 0.785]` rad；球心 TCP 为约 `[0.400000000, 0, 0.729912340]` m。相对原始 X≈0.306957475 m 的姿态，前移约 0.093042525 m，高度和朝向保持不变，基座位置不动。显式传入 `initial_positions` 仍会覆盖该默认值。

当前 MuJoCo 基座的 `quat="0 0 0 1"` 使用 wxyz 顺序，表示绕世界 Z 旋转 180°，因此基座 +X 对应世界 −X。桌子中心的世界 X 为 −0.60 m；弹簧和接触面中心的世界 X 为 −0.40 m，与初始 TCP 的 XY 对齐。`table_assembly` 的 mocap 原点仍为零，升降/移走动作不会覆盖 XY 布局。

静态人体中心为 `[0.4, 0, 0.25]` m；动态人体运动中心为 `[0.45, -0.16, 0.30]` m，按要求保留相对 TCP 路径的 5 cm +X 偏移。动态幅度、频率、所有人体半径和可达性参数保持原值。surface-following 人体发布器直接读取接触面位置，会自动跟随新布局，无需再次添加偏移。

配套直线 via-point 示例和 `tools/static_down_collision_disappear_test.py` 的目标 X 为 0.40 m；循环、移动加旋转和桌角示例以 X=0.40 m 为基准保留原来的 X 偏移和路径形状。所有目标 Y/Z、姿态和动作设置保持原值；外部手动发送的绝对目标不会自动修改。这是初始布局和示例目标配置，不会把控制器永久锁定在 X=0.40 m。

布局验证包括 10 项工具/姿态 pytest、98 个示例目标的平移对照，以及生成后 MuJoCo 模型的离线正运动学、初始接触检查和桌面移走时的 XY 保持检查；未运行闭环接触实验。

验证于 2026-09-09 完成：开发容器内四个相关包编译通过；53 项 monitor GTest、3 项异步邮箱 GTest、9 项工具模型 pytest 全部通过。工具测试涵盖 0.03/0.05 m 半径生成、惯量更新、URDF/SaRA 关节坐标一致性及 real/sim xacro 球心 TCP。构建产物位于 `/tmp/cps_recovery_build` 和 `/tmp/cps_recovery_install`；未启动仿真或机器人，未验证接触闭环的实际运行表现。
