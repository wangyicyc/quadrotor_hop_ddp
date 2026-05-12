# 四旋翼无人机HOP-DDP项目

本项目实现了基于HOP-DDP算法的四旋翼无人机最优控制。

**注意：LQR求解器作为DDP的理论基础包，不单独用于测试。DDP算法在非线性系统的局部线性化基础上应用LQR方法。**

## 项目结构

```
hop_ddp/
├── hop-lib/                          # 核心库文件
│   ├── dynamics/                     # 动力学模型文件夹
│   │   ├── quadrotor_dynamics.py     # 四旋翼无人机动力学模型
│   │   └── quadrotor_config.py       # 四旋翼无人机参数配置
│   ├── hop_lqr_solver.py            # HOP-LQR求解器（DDP的理论基础）
│   ├── hop_ddp_solver.py            # HOP-DDP求解器（主要求解器）
│   └── class_types.py               # 数据类型定义
├── scripts/                         # 测试脚本
│   ├── test_quadrotor_dynamics.py    # 动力学模型测试
│   └── test_quadrotor_hop_ddp.py    # HOP-DDP测试
└── QUADROTOR_README.md              # 本文件
```

## 四旋翼无人机动力学模型

### 状态向量 (12维)
```
x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
```
- `px, py, pz`: 位置 (m)
- `vx, vy, vz`: 速度 (m/s)
- `phi, theta, psi`: 滚转、俯仰、偏航角 (rad)
- `p, q, r`: 机体坐标系下的角速度 (rad/s)

### 控制向量 (4维)
```
u = [f1, f2, f3, f4]
```
- `f1, f2, f3, f4`: 四个电机的推力 (N)

### 物理参数
- 质量: 1.0 kg
- 重力加速度: 9.81 m/s²
- 转动惯量: Ixx=0.0023, Iyy=0.0023, Izz=0.0040 kg·m²
- 机臂长度: 0.25 m
- 偏航力矩系数: 0.1 N·m

## 使用方法

### 1. 测试动力学模型

```bash
cd /home/cyc/Contact-Plan/src/hop_ddp/scripts
python test_quadrotor_dynamics.py
```

这个脚本测试四旋翼无人机动力学模型的基本功能，包括：
- 悬停控制
- 线性化动力学
- 上升控制
- 前进控制
- 连续时间动力学

### 2. 测试HOP-DDP求解器

```bash
cd /home/cyc/Contact-Plan/src/hop_ddp/scripts
python test_quadrotor_hop_ddp.py
```

这个脚本测试HOP-DDP求解器在四旋翼无人机非线性控制中的应用，包括：
- 悬停控制
- 直线轨迹跟踪
- 圆形轨迹跟踪

## 配置说明

### 悬停配置 (HoverConfig)
- 初始位置: (0, 0, 1) m
- 目标位置: (0, 0, 1) m
- 适用场景: 保持悬停

### 轨迹跟踪配置 (TrajectoryTrackingConfig)
- 初始位置: (0, 0, 0) m
- 目标位置: (1, 1, 1) m
- 适用场景: 点到点导航

### 圆形轨迹跟踪配置 (CircleTrackingConfig)
- 圆半径: 1.0 m
- 圆高度: 1.0 m
- 圆周速度: 0.5 m/s
- 适用场景: 圆形轨迹跟踪

## 代价函数

### 阶段代价
```
L(x, u) = 0.5 * (x^T Q x + u^T R u)
```

### 终端代价
```
L_f(x) = 0.5 * (x^T Q_T x)
```

### 权重矩阵
- 状态权重矩阵 Q (12×12):
  - 位置权重: 10.0
  - 速度权重: 1.0
  - 姿态权重: 5.0 (滚转、俯仰), 1.0 (偏航)
  - 角速度权重: 0.1

- 控制权重矩阵 R (4×4):
  - 电机推力权重: 0.1

## 算法说明

### HOP-LQR（理论基础）
- 基于线性二次调节器的最优控制
- 作为DDP算法的理论基础
- 在DDP的反向传播步骤中使用Riccati方程
- 提供局部线性化系统的最优控制解

### HOP-DDP（主要求解器）
- 基于微分动态规划的非线性最优控制
- 在非线性系统的局部线性化基础上应用LQR方法
- 通过二阶泰勒展开获得更好的局部近似
- 使用牛顿法在函数空间中进行优化
- 反向传播计算最优控制策略
- 适用于强非线性系统的轨迹优化

## 依赖项

- JAX: 用于自动微分和数值计算
- NumPy: 数值计算
- Python 3.7+

## 注意事项

1. 确保在正确的Python环境中运行（需要安装JAX）
2. 控制输入需要满足物理限制（电机推力范围）
3. 初始状态和目标状态需要在合理范围内
4. 对于复杂的轨迹，可能需要调整权重矩阵

## 扩展建议

1. 添加障碍物避障功能
2. 实现多无人机协同控制
3. 添加风扰动模型
4. 实现传感器噪声和状态估计
5. 添加可视化功能

## 参考文献

- 论文: 《Horizon Optimal Planning with Differential Dynamic Programming》
- HOP-LQR: 有限时间范围优化的线性二次调节器
- HOP-DDP: 微分动态规划的非线性最优控制方法
