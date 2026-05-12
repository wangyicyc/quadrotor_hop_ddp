# HOP-DDP 四旋翼项目索引

这个目录是四旋翼 HOP-DDP 实验的 ROS/Python 包目录。当前最终运行入口是：

```bash
cd /home/cyc/quadrotor_hop_ddp/src/hop_ddp
python scripts/quadrotor_hop_ddp.py
```

运行结束后，结果会保存到 `datas/YYYY-MM-DD_HH-MM-SS/`，主要包括：

- `console.log`：本次终端输出备份
- `hop_*.log`：求解器日志
- `iter_*.png`：每轮迭代轨迹图
- `quadrotor_final_trajectory.png`：最终轨迹图
- `quadrotor_tracking_errors.png`：跟踪误差图

## 目录结构

```text
hop_ddp/
├── README.md
├── CMakeLists.txt
├── package.xml
├── solver.py
├── config/
│   ├── __init__.py
│   └── quadrotor_config.py
├── dynamics/
│   └── quadrotor_dynamics.py
├── hop_lib/
│   ├── hop_ddp_solver.py
│   ├── hop_lqr_solver.py
│   ├── plot_utils.py
│   └── utils.py
├── scripts/
│   └── quadrotor_hop_ddp.py
├── datas/
├── include/
├── launch/
└── src/
```

## 运行入口

### `scripts/quadrotor_hop_ddp.py`

最终应运行这个文件。

它负责组装完整实验流程：

- 创建 `QuadrotorConfig`
- 创建 `QuadrotorDynamics`
- 构造阶段代价和终端代价
- 创建 `HOPDDPSolver`
- 调用 `solver.solve(...)`
- 保存最终轨迹图和误差图

常用运行命令：

```bash
python scripts/quadrotor_hop_ddp.py
```

## 核心代码

### `config/quadrotor_config.py`

四旋翼实验配置文件。这里集中定义：

- 物理参数：质量、重力、转动惯量、阻尼等
- 初始状态 `x0`
- 目标状态 `x_target`
- 时间步长 `dt`
- 最大步数 `tsteps`
- 控制边界
- 代价矩阵权重
- HOP-LQR 数值保护和时域过滤参数
- DDP 外层迭代参数

如果要改目标点、代价权重、时域长度或求解器参数，优先看这个文件。

### `dynamics/quadrotor_dynamics.py`

四旋翼动力学模型。这里定义：

- 状态维度和控制维度
- 连续/离散动力学
- Euler / RK4 离散化
- 悬停控制 `hover_control()`
- 控制输入与推力/力矩限制相关逻辑

### `hop_lib/hop_ddp_solver.py`

HOP-DDP 外层非线性求解器。主要职责：

- rollout 初始轨迹
- 对非线性动力学和代价进行局部线性化/二次化
- 构造 HOP-LQR 需要的增广系统
- 调用 HOP-LQR 选择 `T*`
- 执行 DDP backward pass
- 执行 forward pass 和 line search
- 返回最终轨迹、控制序列、总代价和时域

### `hop_lib/hop_lqr_solver.py`

HOP-LQR 子问题求解器。主要职责：

- 预计算 LFT 复合映射
- 快速评估不同候选 horizon 的局部二次代价
- 选择最优时域 `T*`
- 应用必要的数值稳定化和 horizon validity filter

这个文件是 HOP-DDP 里选择时域的核心模块。

### `hop_lib/plot_utils.py`

可视化工具。用于保存：

- 四旋翼轨迹图
- 状态/位置跟踪误差图
- 每轮迭代的轨迹图

### `hop_lib/utils.py`

通用工具。当前主要包括：

- 运行输出目录 `get_run_dir()`
- 日志工具 `get_logger()`
- 矩阵正定化辅助函数

## 其他文件