"""
四旋翼无人机平坦输出配置 (Flat Output Configuration)
基于微分平坦性重构的状态空间配置
"""
import inspect
from typing import Tuple

import jax.numpy as jnp

# 复用原有的 BaseConfig 和转换函数 (此处省略重复代码，实际使用时请保留原文件中的 BaseConfig 和 class_to_dict)
# 为简洁起见，这里假设 BaseConfig 和 class_to_dict 已存在或从原文件导入
# 在实际项目中，你可以直接 import 原来的类

class BaseConfig:
    def __init__(self) -> None:
        self.init_member_classes(self)
    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key == "__class__": continue
            var = getattr(obj, key)
            if inspect.isclass(var):
                i_var = var()
                setattr(obj, key, i_var)
                BaseConfig.init_member_classes(i_var)

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"): return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"): continue
        val = getattr(obj, key)
        if isinstance(val, list):
            result[key] = [class_to_dict(item) for item in val]
        else:
            result[key] = class_to_dict(val)
    return result

class QuadrotorFlatConfig(BaseConfig):
    """四旋翼无人机平坦输出配置"""
    
    # --- 物理参数 (保持不变) ---
    mass: float = 1.0          # 质量 (kg)
    g: float = 9.81            # 重力加速度 (m/s^2)
    Ixx: float = 0.0023        # 转动惯量
    Iyy: float = 0.0023
    Izz: float = 0.0040
    arm_length: float = 0.25    
    c: float = 0.1             # 偏航力矩系数
    
    # --- 算法参数 ---
    dt: float = 0.05           # 建议更小的时间步长以捕捉高阶动态
    tsteps: int = 100          # 时间步数
    
    # --- 状态维度定义 ---
    # 平坦输出 sigma = [x, y, z, psi] (4维)
    # 状态向量 x_flat = [sigma, sigma_dot, sigma_ddot, sigma_dddot] (4 * 4 = 16维)
    nx_flat: int = 16
    nu_flat: int = 4           # 控制输入 u_flat = [snap_x, snap_y, snap_z, snap_psi]
    
    # 初始状态 (16,) - 需要构造高阶导数为0的悬停状态
    # [x, y, z, psi, dx, dy, dz, dpsi, ddx, ddy, ddz, ddpsi, dddx, dddy, dddz, dddpsi]
    x0_flat: jnp.ndarray = jnp.array([
        0.0, 0.0, 1.0, 0.0,      # 位置 + 偏航
        0.0, 0.0, 0.0, 0.0,      # 速度 + 偏航率
        0.0, 0.0, 0.0, 0.0,      # 加速度 + 偏航加速度 (悬停时为0)
        0.0, 0.0, 0.0, 0.0       # Jerk + 偏航Jerk (悬停时为0)
    ])
    
    # 目标状态 (16,)
    x_target_flat: jnp.ndarray = jnp.array([
        10.0, 10.0, 2.0, 0.0,      # 目标位置
        0.0, 0.0, 0.0, 0.0,      # 目标速度
        0.0, 0.0, 0.0, 0.0,      # 目标加速度
        0.0, 0.0, 0.0, 0.0       # 目标Jerk
    ])
    w: float = 0.1
    class CostMatrices:
        """代价矩阵配置 (针对平坦状态)"""
        Q_flat: jnp.ndarray = jnp.diag(jnp.array([
            1.0, 1.0, 1.0, 0.1,   # sigma (pos, yaw)
            0.1, 0.1, 0.1, 0.01,  # dot (vel, yaw_rate)
            0.01, 0.01, 0.01, 0.001, # ddot (acc, yaw_acc)
            0.001, 0.001, 0.001, 0.0001 # dddot (jerk, yaw_jerk)
        ]))

        Q_T_flat: jnp.ndarray = 50.0 * 100 * Q_flat
        R_flat: jnp.ndarray = jnp.diag(jnp.array([0.0001, 0.0001, 0.0001, 0.00001])) # 惩罚 Snap

    class ControlLimits:
        """物理控制限制 (用于映射后的检查)"""
        f_min: float = 0.0
        f_max_factor: float = 2.5  # 最大推力倍数
        max_tilt: float = jnp.pi / 3 # 最大倾斜角 60度

def get_flat_cost_matrices(config: QuadrotorFlatConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return config.CostMatrices.Q_flat, config.CostMatrices.R_flat

def get_flat_terminal_cost_matrix(config: QuadrotorFlatConfig) -> jnp.ndarray:
    return config.CostMatrices.Q_T_flat

def get_physical_control_limits(config: QuadrotorFlatConfig) -> Tuple[float, float, float]:
    f_max = config.ControlLimits.f_max_factor * config.mass * config.g
    return config.ControlLimits.f_min, f_max, config.ControlLimits.max_tilt
