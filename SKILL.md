---
name: hop-ddp-quadrotor-debug
description: Use when debugging the quadrotor HOP-DDP/HOP-LQR implementation in this repository, especially when T* becomes untrustworthy, J_values contain NaN/Inf, HOP-LQR matrices become ill-conditioned, or the solver collapses to very short horizons. Focus on the diagnostic workflow, failure signatures, and validated stabilization changes already discovered in this codebase.
---

# HOP-DDP 四旋翼排障技能

这个技能用于本仓库里四旋翼版本的 `HOP-DDP / HOP-LQR` 联调与数值排障。

适用场景：
- `J_values` 出现 `nan` / `inf`
- `T*` 长期被选成 `1` 或极短时域
- `HOP-LQR` 诊断里出现 `bad_M`、`bad_P0`、`rank_drop_F`
- 四旋翼明显比 `cartpole` 更容易病态
- 已经能跑，但时域选择不可信，或者短时域把 DDP 主问题带崩

这个技能不追求“论文复述”，而是总结目前在这个代码库里已经验证过的结论、诊断顺序和修改策略。

## 关键结论

### 1. 这套实现里，真正需要先分清的是“DDP 子问题坏了”还是“HOP-LQR 选时域坏了”

不要一开始就盯着最终轨迹。

先区分两件事：
- 固定 `T_max` 时，DDP 是否能把轨迹逐步优化到接近目标
- 打开 horizon-optimal 后，是否是 HOP-LQR 把时域过早压短

如果固定 `T_max` 时 DDP 本身能明显改善轨迹，而打开 HOP-LQR 后开始劣化，优先排查 HOP-LQR 侧，不要先怀疑 DDP 主框架。

### 2. 到目前为止，四旋翼上的主病灶经历了三个阶段

已观察到的演化链条：
- 早期：`P0` 很早坏，常见是 `P0@2`
- 中期：`bar_F` / `F` 在 `t≈9~10` 开始秩亏缺
- 当前：`F` 的秩塌陷大体已经压住，后续更常见的是 `M` 在中后段变坏，或者 HOP-LQR 在局部模型下真的偏好极短时域

也就是说，排障顺序不是固定的。要以当前输出为准。

### 3. “终端代价不起作用”通常不是因为 Q_T 没加进去，而是因为局部子问题的短时域代价在当前名义轨迹附近真的更便宜

特别是在四旋翼上，如果当前名义轨迹已经允许短时域“近似不动”并避免大控制开销，那么即使 `Q_T` 存在，HOP-LQR 也可能仍然偏好很短的 horizon。

这不一定是公式实现错了，可能是：
- 当前线性化点附近的局部模型在鼓励短时域
- 当前 nominal trajectory 让短时域 surrogate 过于乐观
- 时域缩短后，DDP 前向更新把主问题带回“几乎不动”的轨迹

### 4. 四旋翼比 cartpole 更敏感，不代表问题是完全模型错误

四旋翼更容易触发病态，主要因为：
- 状态维更高
- 增广后矩阵规模更大
- `Q^{-1} A^T` 这条支路更容易把 `F` 做尖
- `bar_F` 递推对奇异值塌陷特别敏感
- 控制、姿态、速度等量纲耦合更强

cartpole 只能作为对比参考，不能直接照搬结论。

## 已验证过的实现改动

以下修改已经在仓库里做过，并且是有针对性的，不是随便试参。

### A. `T*` 选择忽略 `NaN/Inf`，但不修改 `J_values` 原始打印

目的：
- 保留原始数值问题用于诊断
- 避免 `argmin` 被非有限值直接污染

结论：
- 这是必要的基础保护，但它本身不能解决矩阵病态

### B. 四旋翼控制改成相对悬停推力 `delta_u`

含义：
- 优化变量不是绝对推力 `u`
- 而是围绕 `u_hover` 的偏差 `delta_u`

目的：
- 让 nominal control 更接近 0
- 减轻线性化和成本缩放的不平衡

结论：
- 原理上是合理的，没有明显漏洞
- 但它不是万能药，不能单独修复 HOP-LQR 的病态

### C. 先用固定 `T_max` 做 warm-start，再切回 HOP-LQR 选时域

目的：
- 避免刚开始名义轨迹很差时，HOP-LQR 直接判定 `T=1`

结论：
- 很有帮助
- 但如果后续局部子问题仍然偏好短时域，还是会塌回去

### D. 给 HOP-LQR 加详细诊断

当前已加入过的诊断包括：
- summary diagnostics
- horizon breakdown
- early horizon balance
- recursive growth
- `P0` direction diagnostics
- `F / bar_F` chain diagnostics
- validity filter

这些诊断已经足够定位当前主病灶，不要再盲目加更多噪声打印。

### E. 对 `bar_F` 增加相对约束与 SVD floor

目的：
- 防止 `bar_F` 的最小奇异值掉到机器精度附近
- 避免在 `t≈9~10` 左右就出现 rank collapse

结论：
- 这是有效的
- 最新输出里 `rank_drop_F=0` 就说明这一步基本压住了 `F` 的秩塌陷

### F. 将 `rank-deficient F` 纳入 validity filter

目的：
- 避免表面上 `J` 还是有限，但对应 horizon 已经不可信

结论：
- 逻辑上正确
- 但如果 `F` 已经修好，后续主要矛盾就不再是 `rank_drop_F`

补充注意：
- `summary diagnostics`、`validity filter`、`F-chain diagnostics` 必须共用同一套 rank 判据
- 否则会出现一种非常误导的情况：
  - summary 显示 `rank_drop_F=0`
  - validity 却显示大量 `reasons=[rank_F]`
- 一旦出现这种冲突，先统一 rank 计算逻辑，再继续解释数值现象

### G. 增加 horizon shrink guard

目的：
- 当 HOP-LQR 提议把 `T*` 大幅缩短时，先用任务质量判断这个缩短是否可信
- 避免 `T*=1` 把主问题直接带进坏循环

结论：
- 这是“主问题保护”，不是矩阵修复
- 当矩阵已经不坏，但局部 horizon preference 仍然过短时，非常有用

补充：
- 当系统已经进入稳定工作区后，不要只用“拒绝缩短”这一种硬 guard
- 更好的方式是改成 staged shrink：
  - 如果 HOP-LQR 想把 `T*` 从 `20` 直接缩到 `10`
  - 且当前任务误差还不够小
  - 那么先只允许缩到一个中间值，比如 `18`、`15` 或 `12`
- 这样可以避免一直死守在旧时域，也能避免一口气掉进过短 horizon

当前仓库里的进一步实现：
- `staged-hop-lqr` 已经不是固定大步长，而是分三级收缩
  - 大步：`5`
  - 中步：`3`
  - 小步：`2`
- 并且把 `first_bad_M` 接到外层收缩逻辑里：
  - 如果候选 horizon 对应的 `first_bad_M` 已经前移到较早位置
  - 就自动把最大收缩步长进一步压到更小

这一步的意义是：
- 不只是“防止 `T*=1`”
- 而是防止出现这种回退链：
  - 这一轮 `T*=20`
  - 下一轮被 HOP-LQR 提议成 `10`
  - 再下一轮矩阵坏点从 `M@45` 快速前移到 `M@19`

判断这一步是否有效，不要只看 `T*` 是否变大，要同时看：
- `first_bad_M` 是否停止快速前移
- `validity span` 是否不再突然塌到很短
- DDP 主问题是否还保持可接受的轨迹质量

### G2. 当矩阵已经健康时，要把问题转成“是否允许继续缩短 horizon”

典型信号：
- `bad_M=0`
- `bad_P0=0`
- `rank_drop_F=0` 或只在 very late tail 才出现
- `validity span` 很长
- 但 `raw T*` 仍持续偏向 `10~20` 一类较短 horizon

这时主矛盾已经不是矩阵病态，而是：
- HOP-LQR 的局部 surrogate 确实更偏好短时域
- 但外层 DDP 的真实任务轨迹还没有收敛到适合继续缩短的阶段

当前仓库里的处理方式：
- 在外层增加 `horizon freeze / shrink gate`
- 如果参考轨迹还没达到足够小的误差阈值，就直接冻结继续缩短：
  - `ref_pos_err > 0.20` 时冻结
  - 或 `ref_state_err > 1.00` 时冻结
- 即使误差已经低于阈值，也要看最近几轮是否进入平台期：
  - 如果最近几轮 `pos_err/state_err` 仍在明显下降，继续冻结
  - 只有任务质量改善趋缓，才允许 staged shrink 继续缩短 horizon

这一步的目标不是证明 HOP-LQR 错了，而是承认：
- “局部 horizon 偏好” 和 “当前 DDP 主问题最需要的 horizon” 不一定一致
- 在这种不一致阶段，优先保住稳定优化，而不是急着采纳更短的 `T*`

诊断读法：
- `reason=error_above_threshold`：误差还没低到允许缩短的水平
- `reason=task_still_improving`：误差已经不太高，但长时域 DDP 还在明显改善，暂时不要缩短
- 没有 freeze 而进入 `staged-hop-lqr`：说明任务质量已足够好，且近期改善接近平台期，可以尝试缩短

如果输出出现：
- `matrix diagnostics` 已基本全绿
- 但一旦继续缩短，`current quality` 开始恶化

那么下一步不该继续优先调 `_compute_composite_maps_impl`，而应该先加强这层 freeze / shrink 策略。

### G4. staged shrink 先通过历史最佳任务质量 precheck，再用 trial rollout 决定是否接受

最新补充：
- 仅靠 `horizon_guard_min_ratio` 不够
- 因为 `50 -> 45`、`34 -> 29` 这类缩短比例并不大，但仍可能把终端点切到名义轨迹上很差的位置
- 当前仓库新增了 `enable_horizon_shrink_best_quality_guard`
- 当前仓库进一步新增了 `enable_trial_horizon_shrink`

触发条件：
- 只要 `T*` 相对上一轮变短
- 就把缩短后参考轨迹的 `pos_err/state_err` 和历史最佳 `best task quality` 比较
- 如果差得过多，先进入 `trial-shrink`，让 DDP 用这个短 horizon 做一次真实 rollout
- 只有 rollout 后的 `candidate quality` 仍然差得过多，才拒绝缩短并保持上一轮 `T*`

precheck 默认阈值：
- `pos_err > max(0.08, 3.0 * best_pos_err)`
- `state_err > max(0.60, 3.0 * best_state_err)`

trial rollout 默认阈值：
- `candidate_pos_err > max(0.10, 5.0 * best_pos_err)`
- `candidate_state_err > max(0.30, 5.0 * best_state_err)`
- `candidate_total_cost <= previous_accepted_total_cost + max(1e-6, 1e-4 * abs(previous_accepted_total_cost))`

成本检查的原因：
- trial shrink 的 `ΔCost` 是在新的短 horizon 内部计算的
- 它不保证新的短 horizon 解比上一轮已接受的长 horizon 解更好
- 因为目标函数已经包含 `w*T`，不同 horizon 的 total cost 可以直接比较
- 如果 candidate 质量看起来能接受，但 total cost 仍然比上一轮 accepted cost 高，就应该拒绝这次 trial shrink

对应输出：
- `[guard] trial horizon shrink: ... pre_reason=...`
- `[guard] reject trial horizon shrink after rollout: ... post_reason=...`
- `[guard] trial horizon cooldown: ... keep T*=...`

读法：
- 只看到 `trial horizon shrink` 不代表失败，只代表截断名义轨迹看起来不够好
- 如果随后没有 `reject trial horizon shrink after rollout`，并且本轮 `Step=accepted, HorizonSource=trial-shrink`，说明短 horizon 通过了真实 rollout 检查
- 如果出现 `trial-rejected-hop-lqr`，说明短 horizon 的真实 candidate 破坏任务质量，或者 total cost 相比上一轮 accepted solution 倒退
- 如果 `post_reason=cost_backslide`，说明短 horizon 在自身子问题内下降了，但跨 horizon 总成本不如上一轮
- 如果出现 `trial-cooldown-hop-lqr`，说明刚刚有 trial shrink 被拒，接下来几轮先保持旧 horizon 继续优化，避免重复撞同一个坏缩短

这一步专门防止这种模式：
- 长时域已经达到 `best task quality: pos_err≈0.01, state_err≈0.04`
- HOP-LQR 仍偏好短 horizon
- staged shrink 允许 `34 -> 29`
- 截断后的参考终端质量突然变成 `pos_err≈0.76, state_err≈6.25`
- 后续 DDP 虽然会慢慢救回来，但已经丢掉最佳轨迹区间

为什么要用 trial：
- 纯 hard guard 太保守，会把“截断点差但 DDP 可修复”的 horizon 也拒绝掉
- 直接放开又太危险，会让短 horizon 把主问题带回坏 nominal
- trial 是折中：先允许短 horizon 进入真实 DDP 子问题，再用 rollout 质量验收
- trial reject 后默认 cooldown `3` 轮，这样不会在同一个 nominal 附近反复浪费迭代

### G1. 在 `J_value` 候选集合里临时忽略最前面的若干个 horizon

适用场景：
- 已经确认 `t=1` 或 `t=1..5` 的 `J_value` 明显异常偏小
- 但当前不想立刻继续增加 surrogate 一致性修复的复杂度
- 希望先把 `T*` 选择从“极短时域吸引子”里拉出来，再单独处理矩阵质量

当前仓库里的做法：
- 在 `find_optimal_horizon(...)` 里保留原始 `J_values`
- 但在最终 `argmin` 之前，把 `t < min_candidate_horizon` 的候选临时屏蔽为 `inf`
- 当已经确认 `t=6` 仍会把主问题拖入坏区时，可继续抬到 `min_candidate_horizon = 10`
- 也就是先忽略 `t=1..9`，观察系统是否会稳定在 `10~20` 一带

注意：
- 这只是候选集合护栏，不是 HOP-LQR 矩阵质量修复
- 如果后续 `bad_M / bad_P0 / rank_drop_F` 仍然存在，说明本体问题还在，只是没有再让最前面的异常短时域直接主导 `T*`

### G3. 用 prefix invariance 检查确认“后缀坏矩阵是否污染前缀 J”

适用场景：
- 用户怀疑 HOP-LQR 虽然表面上 `t=2~5` 矩阵健康，但后面 `M@26` 或 `F@36` 的坏点通过某种 backward 依赖影响了前面的 `J_values`
- 需要区分“后缀矩阵崩坏导致前缀 J 算错”和“前缀 surrogate 自己就偏好短 horizon”

当前仓库里的诊断：
- `enable_hop_lqr_prefix_invariance_diagnostics = True`
- 默认检查 `prefix=12`
- 打印：
  - `[diag] HOP-LQR prefix invariance: status=PASS/FAIL, prefix=..., max_abs_diff=..., max_rel_diff=...`

读法：
- `PASS` 且差异接近机器精度：
  - 说明 `J(1..prefix)` 不受后缀数据影响
  - 后面的 `M@26`、`F@36` 不是前面 `J(2..5)` 偏小的直接原因
  - 此时优先判断为“局部 surrogate 短时域偏好”
- `FAIL`：
  - 说明实现里存在非局部副作用
  - 优先检查 `_compute_composite_maps_impl(...)` 的递推是否用了全局统计量、全局裁剪或其他会让后缀影响前缀的逻辑

注意：
- HOP-LQR 的查询形式确实包含等价 backward 信息，但本实现中 `bar_E/bar_F/bar_G` 是前缀复合量
- 理论上 `J(t)` 只应该依赖 `0..t-1` 的前缀和共享 terminal surrogate
- 所以 prefix invariance 是判断“后缀是否污染前缀”的直接实验

### H. 对 `bar_G / M` 链路增加广义特征值裁剪

目的：
- 在 `bar_F` 基本稳住之后，当前主病灶往往转移到 `M = P_T + bar_G_t`
- 单纯限制谱范数只能限制“大小”，不能直接限制 `bar_G` 相对正定基准引入了多少负曲率

做法：
- 递推阶段：对 `bar_G_k` 相对 `E_k` 做广义特征值裁剪
- 查询阶段：对 `bar_G_t` 相对 `P_T` 再做一层广义特征值裁剪

直观理解：
- 不是把 `bar_G` 粗暴缩小
- 而是在白化坐标里限制它的广义特征值范围
- 这样可以更直接地延缓 `M` 失去正定性的时间点

结论：
- 这是针对 `bad_M` 的“矩阵质量修复”
- 如果有效，应该表现为 `first bad M horizons` 后移，而不是仅仅靠 `T*` guard 避免坏 horizon 被采用
- 如果输出出现“某一轮 `bad_M=0`，但后续迭代又反弹”，通常说明方向是对的，但裁剪强度还不够稳，需要继续收紧 `bar_G` 的广义特征值上下界

### I. 当 `M` 基本修好后，下一步要转向 `P0 = bar_E - temp`

典型信号：
- `rank_drop_F=0`
- `bad_M` 已经很少，或者首次坏点明显后移
- `bad_P0` 开始只在中后段零星出现，例如 `P0@30~33`

这说明当前主矛盾已经从 `bar_G / M` 转移到 `bar_E - temp` 的差分质量。

优先动作：
- 提高 `bar_E` / `P0` 的最小特征值地板
- 收紧 `temp` 相对 `bar_E` 的 Loewner cap

目标现象：
- `first bad P0 horizons` 继续后移
- `validity span` 不再被 `P0` 在中段截断

如果输出已经呈现：
- `bad_M=0` 或只在很后面出现
- 但 `P0@30~33` 仍然反复出现

那么下一步就不该再优先调 `bar_G / M`，而应该继续收紧 `temp` 相对 `bar_E` 的约束，并适度抬高 `P0` floor。

反过来，如果输出呈现：
- `rank_drop_F=0`
- `first bad P0 horizons` 只在很后面零星出现，或者只是偶发单点
- 但 `first bad M horizons` 稳定卡在较早位置，例如 `M@10~12`

那么当前主病灶仍然是 `bar_G -> M`，而不是 `temp -> P0`。

这种情况下最优先的动作是：
- 继续收紧 `recursive/query bar_G` 的广义特征值上下界
- 目标不是让所有 `M` 都完美正定，而是先把最早坏点 `M@k` 稳定往后推

只要 `M` 还是最早坏点，就不要急着回去再调 `temp_relative_cap`，否则很容易把次要问题当主问题处理。

### J. 固定 `temp` cap 不够时，要切到“弱方向自适应保护”

典型信号：
- `rank_drop_F` 已经不再是最早坏点
- `first bad M horizons` 明显后移，甚至某些迭代里已经没有 `bad_M`
- 但 `first bad P0 horizons` 还是很早，或者反复卡在相近位置

这时说明问题不再是“整体量级过大”，而是：
- `temp` 在某个很薄弱的方向上，已经快把 `bar_E` 的对应方向吃空
- 继续只调一个固定的 `temp_relative_cap`，往往会出现两难：
  - 设大了，弱方向保护不住
  - 设小了，所有健康迭代都被过度压扁

更合理的方式是：
- 先在 `bar_E` 的白化坐标里检查 `temp` 的最大广义特征值
- 只有当它逼近 `1`、也就是 `P0 ~ I - C` 的最弱方向快塌掉时
- 才自动把 `temp` 的相对 cap 收紧到更小的值

这一步的目标不是“一直更保守”，而是：
- 健康迭代尽量保留原本自由度
- 快坏掉的迭代只在最危险时被额外收紧

如果这一步有效，常见表现是：
- `first bad P0 horizons` 后移
- `validity span` 不再总是只剩极短的一段
- 对 `T*` 的保护不再主要依赖外层 clip/guard

### K. 如果矩阵“先变好、后变坏”，要在 DDP 外层加跨迭代矩阵质量保护

典型信号：
- 前几轮里一度出现
  - `bad_M=0`
  - `bad_P0=0`
  - 或者 `first bad M horizons` / `first bad P0 horizons` 已经明显后移
- 但随着 DDP 继续优化，坏点又重新前移
  - 例如 `M@30 -> M@20 -> M@12`
  - 或 `validity span` 从 `last_valid=30+` 缩回到 `10` 左右

这时不要只在 HOP-LQR 内部继续硬裁剪。

## “矩阵健康”和“轨迹健康”要分开判断

这是当前四旋翼排障里最容易混淆的一点。

不要把下面两件事混为一谈：
- `HOP-LQR` 局部矩阵当前是否健康
- 当前被选出来的 `T*` / DDP 更新是否会把 nominal trajectory 带进坏区

### 1. 什么叫“矩阵健康”

这里说的矩阵健康，专指当前这一次 HOP-LQR 查询里，以下对象在当前 nominal 轨迹附近是否数值可用：
- `bar_F`
- `M = P_T + bar_G_t`
- `P0 = bar_E_t - temp_t`

一个比较实用的“局部健康”判据是：
- `bad_M=0`
- `bad_P0=0`
- `rank_drop_F=0`
- `validity causes: bad_M=0, bad_P0=0, rank_F=0`
- `validity span` 覆盖全部或几乎全部 horizon

如果这些条件在某一轮成立，说明：
- 当前这条 nominal trajectory 附近的 HOP-LQR 局部模型是健康的
- 这不代表“以后每轮都健康”，只代表“这轮查询本身是可信的”

### 2. 什么叫“轨迹/时域健康”

轨迹/时域健康指的是：
- 当前 HOP-LQR 选出来的 `T*` 是否合理
- 当前 DDP 子问题在这个 `T*` 下是否还会产生有意义的更新
- 下一轮 nominal trajectory 是否仍留在健康区域

典型坏信号：
- `T*` 继续向极短时域塌缩，例如 `40 -> 30 -> 20 -> 10 -> 6`
- 一旦缩到很短，`candidate quality` 明显恶化
- 再下一轮突然出现：
  - `finite_J=0/N`
  - `bad_M=N`
  - `bad_P0=N`
  - `first structural breaks: P0@1 -> M@1`

这说明：
- 不是矩阵先坏导致选错 `T*`
- 而是 `T*` 先缩得过短，把 DDP nominal trajectory 带坏了
- 再下一轮 HOP-LQR 才在坏 nominal 上整体坍塌

### 3. 最重要的判断逻辑

当你看到：
- 前几轮 `bad_M=0, bad_P0=0, rank_drop_F=0`
- 但 `T*` 仍不断缩短
- 然后若干轮后才出现 `M@1 / P0@1`

要得出的结论是：
- “矩阵曾经是健康的”
- “真正先出问题的是时域选择/主问题更新”
- “后续矩阵崩塌是短时域 nominal 把系统带进坏区后的二次现象”

不要误判成：
- “因为最后矩阵坏了，所以最开始矩阵就不健康”

### 4. 当前四旋翼项目里的推荐口径

建议把状态分成三档来描述：

#### A. 病态

特征：
- 很早就出现 `P0@2`、`M@10`、`F@9`
- `validity span` 很短
- `T*` 明显不可信

结论：
- 这是 HOP-LQR 本体数值病态，优先修矩阵

#### B. 条件性健康

特征：
- 在若干轮里出现 `bad_M=0, bad_P0=0, rank_drop_F=0`
- `validity span` 很长
- 但后续如果 `T*` 缩得过短，还是会触发整体坍塌

结论：
- 这说明矩阵在“正常 nominal 附近”是健康的
- 但系统还不够鲁棒，时域策略仍可能把 nominal 推入坏区

这就是当前四旋翼输出最常见的状态。

#### C. 鲁棒健康

特征：
- 多轮连续保持 `bad_M=0, bad_P0=0, rank_drop_F=0`
- `validity span` 稳定覆盖几乎全部 horizon
- 即使 `T*` 变化，也不会轻易回退到 `M@1 / P0@1`

结论：
- 这时才能更有把握地说“矩阵真的健康了”

### 5. 实操时先问哪一个问题

每次看输出，先按下面顺序问：

1. 这一轮 HOP-LQR 矩阵本体健康吗？
   看 `bad_M / bad_P0 / rank_drop_F / validity causes`

2. 如果健康，为什么 `T*` 还在缩？
   看 `J_values`、early surrogate gap、candidate floor、任务质量

3. 如果后面矩阵又崩了，是不是短时域 nominal 先把系统带坏了？
   看是否先有 `T*` 持续缩短，再有 `M@1 / P0@1`

这样就能避免把“矩阵问题”和“时域策略问题”混在一起。

因为问题已经不只是“单轮查询矩阵太差”，而是：
- 当前 DDP 接受的新 nominal trajectory
- 会把下一轮线性化重新带回一个更坏的局部模型

更合理的做法是：
- 把 HOP-LQR 的结构诊断结果保存下来
- 在 DDP 外层比较“本轮 proposal 的矩阵质量”与“历史最好矩阵质量”
- 如果出现明显 backslide，就拒绝这次 horizon 更新

这里要特别区分两类 guard：
- `horizon guard`：拒绝一个不可信的 `T*` 提议，但不直接阻止 DDP 接受新的 nominal trajectory
- `accept-step matrix guard`：当当前 local HOP-LQR 的可信窗口已经塌到很短时，直接拒绝这次新的轨迹更新，避免下一轮线性化继续恶化

如果输出里已经出现：
- `validity span: first_valid=1, last_valid=1`
- 或 `first bad M horizons` 提前到 `10` 左右

那么只做 `T* clip / horizon guard` 往往不够，因为：
- 你虽然没有真的采用更短的 `T*`
- 但 DDP 仍然在接受一个会把下一轮线性化继续带坏的新 nominal trajectory

这时必须再加一层 `accept-step matrix guard`：
- 当 `first_bad_M` 太早，或者 `last_valid` 太短
- 就拒绝这次新轨迹，保留当前 nominal trajectory，并增大正则化

这样做的目标不是“让优化停住”，而是：
- 把 DDP 留在最近一次还拥有较长可信 horizon 窗口的局部模型附近
- 防止已经修好的 `F/M/P0` 结构在后续迭代中再次前移塌陷

### L. 当 `M/F` 已经健康，但尾段 `P0` 仍反复变坏时，要回到 `_compute_composite_maps_impl`

典型信号：
- `bad_M=0`
- `rank_drop_F=0`
- 但 `first bad P0 horizons` 仍固定出现在尾段，例如 `47~50`

这说明 query 端保护已经不是主战场。

此时更该做的是在 `_compute_composite_maps_impl` 的递推本体里，直接保护

`bar_E_k = bar_E_{k-1} - temp_E`

这一步的最弱几个方向。

推荐做法：
- 保留已有的 Loewner cap，继续控制 `temp_E <= c * bar_E_prev`
- 再额外对 `bar_E_prev` 的最弱子空间做 block-level 预算约束
- 确保这些最危险方向在减法后仍保留一个明确正裕度，而不是只靠最后统一抬地板

这一步修的是“复合映射本体”，不是 query 端的末端止血。

### K. 如果 `bad_M=0`、`rank_drop_F=0`，但尾段仍然出现 `P0@48` 之类的坏点

这时通常说明：
- `bar_G / M` 已经基本修好
- `bar_F` 的秩塌陷也已经压住
- 剩下的问题不是“整体爆炸”，而是 `bar_E - temp` 在尾段最弱几个方向上被慢慢侵蚀到穿零

这种情况下，优先改 `_compute_composite_maps_impl`，不要再主要依赖 query 端过滤。

推荐做法：
- 保留已有的 Loewner cap，继续控制 `temp_E <= c * bar_E_prev`
- 再对 `bar_E_prev` 的最弱 `2~3` 个特征方向单独做预算保护
- 不要用带较大 floor 的 `budget_pd` 近似这个预算，否则会在弱方向留下“看起来很小、但足够把 P0 推到负值”的泄漏
- 更稳的方式是：
  - 先构造弱方向预算 `weak_budget = weak_eig - target_margin`
  - 再在弱子空间里按 `weak_block` 相对 `weak_budget` 的最大广义特征值做统一缩放
  - 让这个最大广义特征值明显小于 `1`，而不是只做到“勉强不超过 1”

经验上，如果输出已经变成：
- `bad_M=0`
- `rank_drop_F=0`
- `first bad P0 horizons: [48, 49, 50]`

那么这是一个好信号，说明问题已经从“结构性崩坏”收敛成“尾段弱方向裕度不足”。
下一步就该继续加强递推里的 weak-subspace budget guard，而不是回头优先改 `bar_F` 或 `M`。

### L. 如果输出重新出现 `M@13~25`，而 `rank_drop_F=0`

这说明问题不再只是尾段 `P0`，而是 `bar_G_k` 和当前递推出来的 `bar_E_k` 开始失配。

典型信号：
- `rank_drop_F=0`
- `first structural breaks` 变成 `M@...` 或 `P0@... -> M@...`
- `last_valid` 又回到十几、二十左右

这时仅仅把 `bar_G_k` 相对单步 `E_k` 做裁剪不够，因为：
- `E_k` 只反映当前一步的单步尺度
- `bar_G_k` 实际上已经是复合映射量
- 如果它和当前的 `bar_E_k` 失去相对平衡，后续的递推分母和 query 端的 `M = P_T + bar_G_t` 都会重新变差

更稳的做法是：
- 先保留 `bar_G_k` 相对 `E_k` 的裁剪
- 再额外让 `bar_G_k` 相对当前 `bar_E_k` 做一次更紧的谱裁剪

这一步的作用不是“硬限制 horizon”，而是修复复合映射内部的尺度一致性。
如果这一步有效，通常会看到：
- `M` 的最早坏点重新后移
- `last_valid` 从十几逐步回升
- 同时 `rank_drop_F` 仍保持为 0

最值得拿来做 guard 的指标：
- `first_bad_M`
- `last_valid`

原因：
- `first_bad_M` 代表最早的结构失真开始点
- `last_valid` 代表当前真正还能信的 horizon 窗口长度

如果这两个指标持续前移/缩短，即使当前任务误差还在下降，也要警惕：
- 你正在用一个更差的局部模型继续优化
- 后续很容易又把 `T*` 选择拖回不可信的短时域

## 当前最重要的诊断读法

优先看下面这几行，不要先看整页输出。

### 1. `first structural breaks`

这是当前最核心的一行。

常见模式：
- `P0@2 -> F@9 -> M@29`
  表示最早是 `P0` 坏
- `F@9 -> M@10`
  表示最早是 `F / bar_F` 递推坏
- `M@24`
  表示 `P0` 和 `F` 暂时都稳住了，当前最早坏点已经变成 `M`

### 2. `rank_drop_F`

如果是：
- 很大，且最早坏点在 `t≈9~10`
  先修 `bar_F`
- 已经是 `0`
  就不要再把主要精力放在 `F` 上

### 3. `bad_P0`

如果是：
- 早期就大量出现，并且最早坏点是 `P0@2`
  说明 `P0` 仍然是主矛盾
- 已经长期为 `0`
  说明 `P0` 不是当前主问题

### 4. `bad_M`

如果：
- 只有中后段坏，如 `M@24`、`M@28`
  说明短 horizon 仍可信，长 horizon 开始坏
- 很早就坏，如 `M@9`
  通常意味着整体局部模型开始失真，或者某一轮时域缩短后把名义轨迹带偏了

如果已经打开了 `bar_G` 的广义特征值裁剪，还要额外看：
- `first bad M horizons` 是否整体后移
- `validity span` 是否不再只剩 `t=1`

如果这两点没有改善，说明 `bar_G` 链路仍然是当前主病灶。

如果这两点已经改善了，但：
- `first bad P0 horizons` 仍然很早
- `P0 direction diagnostics` 里的 `weak_margin` 很快转负

那就说明下一步该盯的是 `temp` 在弱方向上的侵占，而不是继续优先调 `M`。

### 5. `validity filter`

只看 `valid=...` 不够，要看：
- `valid` 的数量
- `last_valid`
- `validity causes`
- `validity cutoff`
- 当前被选出的 `T*`

如果出现：
- `valid=8/50`
- `last_valid=9`
- 然后 `T*=1`

这通常说明当前 HOP-LQR 在“仍被允许的短时域里”选择了最短那个，而不一定是矩阵坏了。

新增读法：
- `validity causes` 用来看三类规则各自一共拦掉了多少个 horizon
- `validity cutoff: first_invalid=..., reasons=[...]` 用来看“连续有效区间”是被哪条规则先截断的

如果 `first bad ... horizons` 已经很干净，但 `last_valid` 还是短，就优先看 `validity cutoff`，不要只盯 summary 总数。

## `F-chain diagnostics` 的读法

关注点不是 `||bar_F||` 爆不爆，而是两件事：
- `rank(bar_F)` 是否开始下降
- 最小奇异值是否快速掉到极小量级

经验判断：
- 如果 `rank(bar_F)` 从 13 开始掉到 12/11/10，并且最小奇异值接近机器精度
  说明是典型的奇异值塌陷
- 如果 `rank(bar_F)=13` 一直保持，最小奇异值也被 floor 住了
  说明 `F` 链条当前已经不是主矛盾

## 建议的标准排障顺序

每次只按这个顺序走，不要跳步骤。

### 第一步：先确认 DDP 固定时域是否正常

做法：
- 固定 `T*=T_max`
- 看每轮 `candidate quality`

判据：
- 如果位置误差、状态误差明显持续下降
  说明 DDP 主框架是能工作的

### 第二步：打开 HOP-LQR，但只看结构性诊断

重点看：
- `first structural breaks`
- `rank_drop_F`
- `bad_P0`
- `bad_M`

不要先盯着最终轨迹图。

### 第三步：判断当前主病灶属于哪一类

三种常见分支：

#### 分支 A：`P0` 最先坏

信号：
- `bad_P0` 很早出现
- `first structural breaks` 里最先是 `P0`

优先动作：
- 继续看 `P0 direction diagnostics`
- 检查 `temp` 是否在弱方向上吃掉了 `bar_E`

#### 分支 B：`F / bar_F` 最先坏

信号：
- `rank_drop_F` 明显
- `first structural breaks` 最先是 `F@...`

优先动作：
- 继续稳 `bar_F`
- 不要先改 `P0`

#### 分支 C：`F` 和 `P0` 都稳，但 `M` 开始坏

信号：
- `rank_drop_F=0`
- `bad_P0=0`
- 最早坏点变成 `M@...`

优先动作：
- 把重点放到 `M = P_T + bar_G_t`
- 优先尝试 `bar_G` 的广义特征值裁剪，而不是继续只靠限制 `T*`
- 检查时域缩短后 nominal trajectory 是否失真

#### 分支 D：`M` 已经大体稳住，但 `P0` 在中后段开始坏

信号：
- `rank_drop_F=0`
- `bad_M` 很少甚至为 `0`
- `first bad P0 horizons` 出现在中后段，例如 `30+`

优先动作：
- 收紧 `recursive_temp_relative_cap`
- 收紧 `query_temp_relative_cap`
- 适度抬高 `bar_E / P0` 的 eigen floor

### 第四步：如果矩阵都不坏但 `T*=1`

这是一个独立问题。

说明：
- 数值上已经“能算”
- 但局部 horizon-optimal surrogate 在当前 nominal trajectory 上偏好极短时域

优先动作：
- 用 horizon shrink guard
- 检查短时域下的任务质量是否明显恶化
- 不要再一味加矩阵正则

## 对“减小总步数会不会更好”的结论

答案是：**有可能暂时更好，但它更像诊断工具，不是根治方案。**

### 为什么减小 `tsteps` 有时看起来有效

因为：
- 复合映射递推长度变短
- `bar_F` / `bar_G` 累积病态的机会变少
- `M` 中后段坏点会被推迟，甚至直接不出现

### 但它不能回答真正的问题

如果减小 `tsteps` 后效果变好，只能说明：
- 当前病态和递推深度相关

不能说明：
- 原始 `T_max` 设定一定错
- horizon-optimal 思想本身不适合四旋翼

所以，减小步数更适合作为：
- 病态是否随 horizon 深度累积的验证实验

而不是最终解决方案。

## 不建议重复做的事情

下面这些在当前阶段不要反复做：

### 1. 不要再只盯着 `Q_T` 继续无脑调大

原因：
- 会让条件数继续恶化
- 不一定能真正改变局部短时域偏好

如果确实要测试终端代价强度，优先使用 `QuadrotorConfig.CostMatrices.terminal_cost_scale` 做整体倍率实验。

建议顺序：
- 先跑 `1.0`
- 再试 `2.0`
- 最多短测 `5.0`

观察重点：
- `early surrogate gap` 里短 horizon 的 `raw_total` 是否被明显抬高
- `raw T*` 是否离开最短候选区间
- `bad_M / bad_P0 / rank_drop_F` 是否因为 Q_T 变大重新恶化
- 推力是否更早接近饱和

### 2. 不要在 `F` 已经稳住后继续把所有精力花在 `bar_F`

如果输出已经显示：
- `rank_drop_F=0`
- `minRank(F)=13`

就说明主病灶已经转移了。

### 3. 不要把 `cartpole` 的现象直接套到四旋翼

它只能用于提供“通用病态模式”的参考，不能直接指导四旋翼调参。

### 4. 不要同时改很多地方再看结果

建议每次只改一类：
- 矩阵稳定性
- validity filter
- horizon shrink guard
- 代价权重

否则输出很难解释。

## 当前阶段的优先级建议

如果今天继续改，优先级如下：

### 优先级 1：确认 `M` 为什么成为新的最早坏点

当输出已经变成：
- `rank_drop_F=0`
- `bad_P0=0`
- `first structural breaks: M@...`

下一步就应该围绕 `M` 展开，而不是继续修 `F`

## 如何判断“已经没有明显矩阵问题了”

至少要同时满足下面几条，才比较能说 HOP-LQR 已经不再主要受矩阵病态支配：

1. `rank_drop_F=0`，并且 `minRank(F)` 长时间保持满秩。
2. `bad_P0=0`，或者只在很靠后的 horizon 零星出现。
3. `first bad M horizons` 明显后移，不再早早卡在很小的 `t`。
4. `validity filter` 不再长期只有 `valid=1/..`，而是能保留一段连续有效区间。

只要第 3 条和第 4 条还明显不满足，就仍然应该把它视为“矩阵质量问题尚未真正解决”。

补充经验：
- 如果某些迭代里已经达到 `bad_M=0, bad_P0=0, rank_drop_F=0`，但下一轮又退化，说明系统处在“临界稳定”区，不是完全没救，而是需要更强一点的矩阵约束来提高跨迭代鲁棒性。

### 优先级 2：阻止“局部时域选择把主问题带进 T=1”

即使矩阵都不坏，只要 `T*=1` 反复出现，就说明主问题需要额外保护。

### 优先级 3：再考虑代价和建模层面的改进

包括：
- 是否需要更合理的 horizon surrogate
- 是否需要调整时域惩罚 `w`
- 是否要重看短时域下 terminal surrogate 的含义

## 相关代码位置

重点文件：
- `src/hop_ddp/hop_lib/hop_lqr_solver.py`
- `src/hop_ddp/hop_lib/hop_ddp_solver.py`
- `src/hop_ddp/scripts/quadrotor_hop_ddp.py`
- `src/hop_ddp/dynamics/quadrotor_config.py`

定位建议：
- HOP-LQR 数值病态：先看 `hop_lqr_solver.py`
- DDP 如何使用 `T*`：看 `hop_ddp_solver.py`
- 四旋翼诊断开关：看 `quadrotor_config.py`
- 四旋翼实验 wiring：看 `quadrotor_hop_ddp.py`

## 处理新输出时的标准回答模板

拿到一份新的 `output.txt` 后，按这个顺序判断：

1. `DDP 固定长时域时是否在变好？`
2. `first structural breaks` 是什么？
3. `rank_drop_F / bad_P0 / bad_M` 各自是不是当前主矛盾？
4. `validity filter` 是否仍允许不可信 horizon？
5. `T*=1` 是矩阵坏掉导致的，还是局部时域选择导致的？

回答时优先给出：
- 当前主病灶
- 它是否比上一轮更靠后
- 下一步只该改哪一类东西

避免泛泛而谈。

## 当前仓库下的最简经验法则

一句话总结：

先用固定 `T_max` 证明 DDP 没坏，再用 `first structural breaks` 判断 HOP-LQR 当前是 `P0`、`F` 还是 `M` 在先坏；如果矩阵都不坏却还掉到 `T=1`，那就不是单纯数值病态，而是局部短时域偏好在把主问题带偏。
