# HOP-DDP 数值稳定性调优指南

## 目录

1. [问题背景](#问题背景)
2. [原始 public tutorial 的 HOP 端诊断](#原始-public-tutorial-的-hop-端诊断)
3. [核心机制：复合递推、Fbar 秩和 P0 正定性](#核心机制复合递推fbar-秩和-p0-正定性)
4. [修改一：状态重标定 (Plan C)](#修改一状态重标定-plan-c)
5. [修改二：自适应 Tikhonov 正则化](#修改二自适应-tikhonov-正则化)
6. [修改三：递归稳定化](#修改三递归稳定化)
7. [修改四：有效性过滤器](#修改四有效性过滤器)
8. [参数调优规则](#参数调优规则)
9. [诊断输出阅读指南](#诊断输出阅读指南)
10. [常见问题速查](#常见问题速查)

---

## 问题背景

HOP-DDP 在四旋翼无人机非线性控制中，HOP-LQR 查询阶段可能出现两类不同的结构性问题。原始 public tutorial 结果中，`P0/temp/M` 没有先坏，首个异常是复合映射 `Fbar` 从 `T=5` 开始秩下降；在后续加入稳定化、防护和 contact 场景后，问题才转移到 `P0 = bar_E - temp` 正定性和 `M` 病态上。

**根因链路**：

```
原始 public tutorial:
Q_aug 正定但存在弱方向
  → E = Q_aug^{-1} 和 F = E A^T 可计算，单步 E/G/F 都未坏
    → 复合递推 Fbar_k = Fbar_{k-1} W_k F_k 快速丢失有效秩
      → 从 T=5 开始 rank(Fbar) 下降，valid horizon 只剩 4/160
        → P0/temp/M 仍保持健康，J 全部有限
          → T*=52 仍与 brute-force 一致，但 HOP 代理在很多非最优 horizon 上严重低估真实 rollout 成本

加防护后的 contact run:
Fbar 秩塌陷被压住
  → 问题转移到 temp = bar_F M^{-1} bar_F^T 相对 bar_E 过大
    → P0 = bar_E - temp 失去正定性
      → bad_P0 / bad_temp / bad_M 大量出现
        → HOP-LQR 代理排序可信度下降，T* shrink 开始不可靠
```

关键文件：
- [quadrotor_config.py](dynamics/quadrotor_config.py) — 模型配置、代价矩阵、状态重标定
- [quadrotor_dynamics.py](dynamics/quadrotor_dynamics.py) — 四旋翼动力学
- [quadrotor_hop_ddp.py](scripts/quadrotor_hop_ddp.py) — 测试脚本
- [hop_lqr_solver.py](hop_lib/hop_lqr_solver.py) — HOP-LQR 求解器（Riccati 递归、稳定化）
- [hop_ddp_solver.py](hop_lib/hop_ddp_solver.py) — HOP-DDP 求解器（外层迭代、T* 选择）

---

## 原始 public tutorial 的 HOP 端诊断

参考运行：`src/public_HOP_horizon_optimal_tutorial/run_quadrotor.py`。

这份原始结果不是失败案例：HOP 和 brute-force 都选择 `T*=52`，且最终轨迹质量很好。

```text
Running Brute-Force...
  T*=52, J*=3600.33

Running HOP...
  T*=52, J*=3600.33

final quality: pos_err=0.0094, state_err=0.0183
top_horizons=[52, 51, 53, 50, 54]
```

但这个结果暴露了一个更早的 HOP 结构问题：**T* 选对了，不代表 HOP 复合映射在所有 horizon 上都健康**。

### 直接症状

```text
finite_J=160/160, nan=0, inf=0
input matrices: nonfinite=False
bad_M=0, bad_P0=0, bad_temp=0
bad_single_E=0, bad_single_G=0, bad_single_F=0, bad_recursive_W=0
rank_drop_F=156
valid=4/160, last_valid=4
first structural breaks: P0@None, temp@None, F@5, M@None, W@None
```

也就是说，原始结果中最早坏掉的不是 `P0`、`temp` 或 `M`，而是 **复合后的 `Fbar` 从第 5 个 horizon 开始数值秩下降**。这条证据比后续 contact run 的 `P0@89` 更靠近原始问题源头。

选中和最优附近的 horizon 虽然都被审计为 rank deficient，但 `P0_inv` 和 `temp/Ebar` 仍然健康：

```text
T=52: rank(Fbar)=9, sv(Fbar)=[2.343e-23, 9.498e-02],
      minEig(P0_inv)=1.360e-04, temp/Ebar=3.774e-04
```

这里的 `rank(Fbar)=9` 不是单步 `F` 坏；日志同时显示 `bad_single_F=0`。它说明问题发生在复合递推

```text
Fbar_k = Fbar_{k-1} W_k F_k
```

之后，某些方向的奇异值被乘到接近机器精度，例如 `T=52` 的最小奇异值已经是 `2.343e-23`。

### HOP 估计与真实 rollout 的差异

虽然 `T*=52` 选对了，但 HOP 估计代价对很多 horizon 有明显低估：

```text
best HOP-estimated: T=52, J_est=3600.22561546
best true rollout:  T=52, J_true=3600.33222908

T=20:  J_est=3824.26, J_true=5863.20,   rel_gap=0.3478
T=52:  J_est=3600.23, J_true=3600.33,   rel_gap=2.961e-05
T=160: J_est=3607.34, J_true=328303.09, rel_gap=0.9890
```

因此原始结果的准确结论应该是：

1. HOP 的最终 `T*=52` 是正确的，和 brute-force 完全一致。
2. 原始结构性首坏点是 `Fbar@5`，不是 `P0/temp/M`。
3. `J` 全部有限，因此不能用 NaN/Inf 来发现问题。
4. HOP 代理在最优附近非常准，但在长 horizon 上严重低估真实 rollout 成本。
5. 后续加入防护后，`Fbar` 秩塌陷被压住，才观察到 `P0/temp/M` 成为新的主矛盾。

### 诊断结论

给导师汇报时，原始问题建议表述为：

> 原始 HOP 四旋翼例子中，算法最终选对了 `T*=52`，但结构审计显示复合映射 `Fbar` 从 `T=5` 开始秩下降，导致只有前 4 个 horizon 被判定为结构有效。`P0/temp/M` 没有先坏，说明最早问题不是 Schur complement 非正定，而是复合 LFT 递推中的 `Fbar` 奇异值塌到机器精度。这个秩塌陷没有改变本例最优 `T*`，但解释了为什么 HOP 估计代价在非最优 horizon 上和真实 rollout 代价差距很大。

后续优先排查方向应按阶段区分：

- 原始阶段：优先解释和修复 `Fbar` 复合递推秩塌陷。
- 加防护阶段：再分析 `P0/temp/M` 病态和 `T*` shrink guard。

---

## 核心机制：复合递推、Fbar 秩和 P0 正定性

### 增广 Q 矩阵与 Schur 补

HOP-LQR 在增广状态空间 `(x, z)` 上工作，增广后的代价矩阵为：

```
Q_aug = [Q_k   q_tilde]
        [q_tilde^T  c  ]
```

其中 `q_tilde` 耦合了状态和标量辅助变量的代价。原始 public tutorial 直接计算 `E_k = Q_aug,k^{-1}`、`F_k = E_k A_k^T`；增强版代码会先对 `Q_k` 加自适应 Tikhonov 正则化得到 `Q_reg`，再计算 `E_k = solve(Q_reg, I)` 和 `F_k = solve(Q_reg, A_k^T)`。两者的共同风险是：`Q_aug` 或 `Q_reg` 的弱方向会进入 `E/F`，再被复合递推放大或压到机器精度。

### 复合映射递归

```
bar_E_0 = E_0,  bar_F_0 = F_0,  bar_G_0 = G_0

bar_E_k = bar_E_{k-1} - temp_E
bar_F_k = bar_F_{k-1} @ Z2
bar_G_k = G_k - F_k^T @ Z2
```

其中 `temp_E = bar_F_{k-1} @ Z1`，`Z1`, `Z2` 由 `E_k`, `F_k` 通过求解线性系统得到。

这里有两个需要分开的健康条件：

1. `Fbar` 应保持足够的有效秩。原始结果的首个坏点就是 `Fbar@5`，说明复合映射在很多方向上的奇异值已经塌到机器精度。
2. 查询阶段的 `P0 = bar_E - temp` 必须正定。如果 `temp = bar_F M^{-1} bar_F^T` 在任意方向上超过 `bar_E`，P0 会失去正定性。这是加入防护后的 contact run 中更明显的后续问题。

### 为什么姿态/角速度通道是问题所在

- 姿态和角速度在动力学 Jacobian 中的耦合会把弱方向效应带入 `F = E A^T`（增强版中是 `F = Q_reg^{-1} A^T`）
- 原始 public tutorial 中，单步 `F` 没有坏，但复合后的 `Fbar` 在递推中快速失去有效秩
- 加入 SVD floor、growth cap 等防护后，`Fbar` 秩塌陷可以被压住，此时需要继续看查询阶段的 `temp = bar_F M^{-1} bar_F^T`
- 当 `bar_E^{-1/2} bar_F M^{-1/2}` 的最大奇异值超过 1 时，`temp` 相对 `bar_E` 的广义特征值会超过 1，从而"消耗" `bar_E` 的正定裕度，使 `P0` 非正定

---

## 修改一：状态重标定 (Plan C)

### 原理

用对角矩阵 D 将状态向量从物理单位变换到数值均衡的坐标系。

```
D = diag([1, 1, 1,  1, 1, 1,  r2d, r2d, r2d,  r2d, r2d, r2d])
其中 r2d = 180/π ≈ 57.2958
```

- 前 6 个分量（位置、速度）保持 SI 单位
- 后 6 个分量（姿态、角速度）从弧度/弧度每秒转换为度/度每秒

### 变换关系

```
x̃ = D @ x           (物理 → 缩放)
x = D^{-1} @ x̃      (缩放 → 物理)
f̃(x̃, u) = D @ f(D^{-1} @ x̃, u)     (动力学)
Q̃ = D^{-T} @ Q @ D^{-1}             (代价 Hessian)
```

JAX 的 `jacfwd` 自动处理变换链：`Ã = D @ A @ D^{-1}`, `B̃ = D @ B`。

### 效果

| 矩阵 | 变换前条件数 | 变换后条件数 |
|------|------------|------------|
| Q | ~1313 | ~10-25 |
| Q_T | ~3283 | ~10 |

### ⚠️ 关键细节

**角速度通道必须包含在缩放中**。`state_scale` 的最后 3 个元素必须是 `r2d`，而不是 `1.0`。

错误写法（曾导致 A 矩阵爆炸）：
```python
# 错误：索引 9,10,11 为 1.0 — A 矩阵出现 57.3x 的交叉项
state_scale = [1,1,1, 1,1,1, r2d,r2d,r2d, 1,1,1]
```

正确写法：
```python
# 正确：所有 6 个姿态/角速度分量都缩放
state_scale = [1,1,1, 1,1,1, r2d,r2d,r2d, r2d,r2d,r2d]
```

---

## 修改二：自适应 Tikhonov 正则化

### 原理

在求 `E_k = Q_reg^{-1}` 之前，对增广 `Q_k` 添加 Tikhonov 正则化：

```python
q_tilde = Q_k[:-1, -1]       # 耦合项
c = Q_k[-1, -1]              # 标量角
correction = ||q_tilde||² / max(|c|, 1e-10)
adaptive_reg = efg_q_reg + efg_q_reg_adaptive_scale * correction
Q_k_reg = Q_k + adaptive_reg * I
E_k = solve(Q_k_reg, I)
F_k = solve(Q_k_reg, A_k^T)
```

- `efg_q_reg`：基础正则化（下限）
- `efg_q_reg_adaptive_scale`：**关键调优参数**——控制对 Schur 补修正量的响应强度
- `correction`：当 `q_tilde` 大时自动增大，保护 E 不被放大

### 调优规则

- **增大 `efg_q_reg_adaptive_scale`** → 更强的正则化 → E 更小 → F 中的弱方向放大更少 → bar_F 相对增益增长更慢 → bad_P0 更少
- **副作用**：过度正则化会使 J 查询代理偏离真实 rollout 成本，导致次优的 T* 选择
- **当前最优值**：`efg_q_reg=0.02, scale=0.25`（配合 `q_att=q_omega=0.05`）

---

## 修改三：递归稳定化

### temp 相对上限 (`recursive_temp_relative_cap`)

限制 `temp_E` 不能超过 `bar_E` 的一定比例：

```python
# 将 temp_E 相对于 bar_E 的广义特征值裁剪到 [0, cap]
temp_E = clip_symmetric_relative_to_base(temp_E, bar_E, 0.0, cap)
```

- `cap=0.25` 表示 temp 最多消耗 bar_E 最小特征值方向的 25%
- **更小的值 → 更保守 → bad_P0 更少，但可能过于保守**
- **当前最优值**：0.25

### bar_F 范数裁剪 (`_stabilize_bar_f`)

```python
# 限制 bar_F 奇异值不超过上一步最大奇异值的 max_growth 倍
max_allowed = max(atol, recursive_bar_f_max_growth * prev_max_sv)
s_clipped = min(s, max_allowed)
```

- `recursive_bar_f_max_growth=1.1`：bar_F 每步最多增长 10%
- **当前最优值**：1.1

### bar_F 增益上限 (query-time)

在查询阶段，限制 bar_F 在 bar_E 度量下的最大奇异值：

```python
# 将 bar_F 的归一化奇异值裁剪到 max_gain
normalized = L_E^{-1} @ bar_F @ L_M^{-T}
s_clipped = min(s, query_bar_f_gain_cap)  # 默认 0.95
```

- `query_bar_f_gain_cap=0.95`：防止 temp 以超过 95% 的幅度消耗 bar_E
- **当前最优值**：0.95

---

## 修改四：有效性过滤器

在 HOP-LQR 查询阶段，过滤掉数值不健康的 horizon：

| 过滤条件 | 参数 | 含义 |
|---------|------|------|
| `bad_P0` | `validity_filter_use_bad_p0` | P0 的最小特征值 ≤ 0 |
| `bad_M` | `validity_filter_use_bad_m` | M 矩阵的最小特征值 ≤ 0 |
| `bad_temp` | `validity_filter_use_bad_temp` | temp 超过 bar_E |
| `rank_drop_F` | `validity_filter_use_rank_deficient_f` | F 矩阵秩不足 |

被过滤的 horizon 不计入 T* 选择。这是最后一道防线，但它只能过滤已经被判定为 bad 的单个 horizon；如果最低 selection-cost 的候选位于首个坏点之前，仍可能被选中。因此大量 horizon 被过滤（bad_P0 > 50）时，应优先调整修改一至三，并额外检查最低 selection-cost horizons 与 first_bad_P0/M 的相对位置。

---

## 参数调优规则

### 规则 1：Q 越大，E 越小，递归越稳定

```
更大 Q / Q_reg → 更小 E=Q_reg^{-1}
  → F=Q_reg^{-1}A^T 的弱方向放大更少
    → bar_F 相对增益增长更慢
      → temp 更不容易超过 bar_E
        → bad_P0 更少
```

**实践**：
- 如果 bad_P0 持续出现 → **增大** `q_att` 和 `q_omega`
- 如果 bad_P0=0 但收敛太慢 → 可以略微减小
- **不要**使 Q 条件数超过 ~50（当前 `q_att=q_omega=0.05, q_pos=0.5` → cond=10）

### 规则 2：自适应 scale 与 Q 成反比

Q 越小 → 需要更大的 `efg_q_reg_adaptive_scale` 来补偿。
Q 越大 → E 已经较小 → 只需较小的 scale。

| q_att/q_omega | 推荐 scale | 说明 |
|--------------|-----------|------|
| 0.02 | 0.25 | 首次调用 bad_P0=0，但二次调用退化 |
| 0.05 | 0.25 | 当前最优：所有调用 bad_P0=0 |

### 规则 3：递归 cap 值越小越保守

| cap 值 | 效果 |
|--------|------|
| 0.98（默认） | 几乎不限制，依赖自然稳定性 |
| 0.35 | 适度保守，阻挡极端 temp |
| 0.25 | 保守，确保 temp 远小于 bar_E |
| 0.15 | 非常保守，可能导致过度正则化 |

### 规则 4：首次 vs 后续调用的差异

首次 HOP-LQR 调用时轨迹接近悬停（`delta_u ≈ 0`），Jacobian 一致且温和。
后续调用轨迹有非零控制输入，Jacobian 发生变化，某些时间步的 `q_tilde` 可能大得多。

如果**首次调用完美但后续退化**：
→ `q_att/q_omega` 过低，增大它们（规则 1）
→ 或增大 `efg_q_reg_adaptive_scale`

如果**首次调用就有大量 bad_P0**：
→ 检查 state_scale 是否覆盖角速度通道
→ 增大 `efg_q_reg_adaptive_scale`
→ 减小 `recursive_temp_relative_cap`

### 规则 5：T* 跳跃裁剪

`solver.enable_tstar_jump_clip = True` + `max_tstar_jump = 15`
防止迭代间 T* 变化超过 15 步。当 J 查询预测不准确时，此机制可防止灾难性 rollout。

### 当前最优参数组合

```python
# quadrotor_config.py — CostMatrices
q_pos: float = 0.5
q_vel: float = 0.05
q_att: float = 0.05    # 增大以压缩 E，防止 bar_F 增长
q_omega: float = 0.05   # 同上
r_u: float = 0.2

# quadrotor_hop_ddp.py — 稳定化参数
solver.hop_lqr.efg_q_reg = 0.02
solver.hop_lqr.efg_q_reg_adaptive_scale = 0.25
solver.hop_lqr.recursive_temp_relative_cap = 0.25
solver.hop_lqr.recursive_bar_f_relative_cap = 0.35
solver.hop_lqr.recursive_bar_f_max_growth = 1.1
solver.hop_lqr.query_bar_f_gain_cap = 0.95
solver.max_tstar_jump = 15
```

---

## 诊断输出阅读指南

### 每次 HOP-LQR 调用后的摘要行

```
[diag] HOP-LQR summary:
  finite_J=100/100      ← 所有 horizon J 值均为有限值（好）
  bad_M=0               ← M 矩阵全部正定（好）
  rank_drop_F=0         ← F 全部满秩（好）
  bad_P0=0              ← ★ 最关键：P0 全部正定
  bad_temp=0            ← temp 未超过 bar_E
  minEig(P0)=2.488e-03  ← ★ 必须 > 0，越大越好
  maxRelEig(temp|bar_E)=9.996e-01  ← ★ 必须 < 1.0
```

### 需要关注的红旗信号

| 指标 | 含义 | 应对 |
|------|------|------|
| `bad_P0 > 0` | Riccati 解不正定 | 增大 q_att/q_omega 或 增大 scale |
| `maxRelEig(temp\|bar_E) > 1.0` | temp 在某些方向超过 bar_E | 减小 recursive_temp_relative_cap |
| `maxRelEig(temp\|bar_E) ≈ 1.0` | temp 边缘情况 | 同上，或增大 query_bar_f_gain_cap |
| `minEig(P0) < 1e-6` | 接近正定边界 | 同 bad_P0 的应对 |
| `bad_M > 0` | M 矩阵退化 | 通常跟随 bad_P0，先修复 bad_P0 |
| `rank_drop_F > 0` | F 秩不足 | 检查 state_scale，增大 efg_q_reg |
| `first_bad_P0 = 小数字` | 早期层级已退化 | 递归 cap 太松，或 Q 太小 |
| `first_bad_P0 = 大数字` | 后期层级退化 | 自适应 scale 可能不够 |

### P0 方向诊断

```
[diag] HOP-LQR p0-dir t=1:
  eig(bar_E)=[1.55e-02, 1.56e+00, 6.51e+00]
  eig(temp)=[3.92e-05, 3.92e-05, 6.50e+00]
  temp_on_barE_weak=1.54e-02    ← temp 在 bar_E 最弱方向上的投影
  barE_on_temp_strong=6.51e+00  ← bar_E 在 temp 最强方向上的投影
  weak_margin=8.45e-05          ← bar_E 最弱方向被 temp 消耗后剩余多少
```

- `weak_margin > 0`：P0 在此层级正定
- `weak_margin < 0`：P0 在此层级不正定
- `|<vE_min,vT_max>|`：bar_E 最弱方向与 temp 最强方向的夹角余弦——越接近 1 越危险

### rollout 预测 vs 实际

```
[diag] HOP-LQR predicted-vs-rollout:
  selected T*=85, best_rollout_T=100
  T=85: predJ=109.8, rolloutCost=1770.2, pos=0.07
  T=100: predJ=110.9, rolloutCost=1708.4, pos=0.01
```

- `predJ` ≠ `rolloutCost` 是正常现象（稳定化后的 J 是代理）
- 但 `predJ` 的**相对排序**应大致匹配 `rolloutCost` 的排序
- 如果排序完全颠倒（如 T*=10 的 predJ 最小但 rollout 灾难），说明稳定化扭曲了代理

---

## 常见问题速查

### Q: bad_P0 大量出现 (>50)，T*=0 被选中

**检查顺序**：
1. `state_scale` 最后 3 个元素是 `r2d` 还是 `1.0`？
2. `q_att`, `q_omega` 是否 ≥ 0.02？
3. `efg_q_reg_adaptive_scale` 是否 ≥ 0.2？
4. `recursive_temp_relative_cap` 是否 ≤ 0.35？

### Q: 第一次 HOP-LQR 完美，第二次开始退化

`q_att`/`q_omega` 太小。从 0.02 增大到 0.05 或更高。参考[规则 4](#规则-4首次-vs-后续调用的差异)。

### Q: 求解器收敛但 T* 在 85 和 100 之间振荡

J 查询代理与实际 rollout 之间存在偏差。可以尝试：
- 减小 `efg_q_reg_adaptive_scale`（如从 0.25 → 0.15）
- 放宽 `recursive_temp_relative_cap`（如从 0.25 → 0.35）
- 增大 `max_tstar_jump` 给求解器更多灵活性

### Q: 想从零开始调优新模型

1. 先做状态重标定，确保 Q 条件数 < 50
2. 从 `efg_q_reg=0.02, scale=0.2` 开始
3. 运行 30 次迭代，检查首次 HOP-LQR 调用后的 `bad_P0`
4. 如果 bad_P0 > 0：增大 scale（+0.05 步长）
5. 如果 bad_P0=0 但后续调用退化：增大 q_att/q_omega
6. 最后微调 `recursive_temp_relative_cap`

---

## 调优历史

| 步骤 | 改动 | 效果 |
|------|------|------|
| 1 | 修复 state_scale 角速度通道 | bad_P0 从 99 → 0（首次调用） |
| 2 | scale=0.25 + recursive_temp_cap=0.25 | 首次调用完美，二次调用 bad_P0=3 |
| 3 | q_att/q_omega = 0.02 → 0.05 | **二次调用退化完全消除**，全部调用 bad_P0=0 |
| 4 | q_att/q_omega = 0.02 → 0.002（尝试降低） | 恶化：bad_P0=24，已回退。降低 Q = 增大 E = 加速 bar_F 增长 |
| 5 | flat 模型 Q rescaling | 暂停——用户转向 12 状态模型 |

**核心教训**：对抗 Riccati 递归不稳定的方向是**增大** Q 姿态/角速度权重（抑制 E 和 bar_F），而不是减小它们。
