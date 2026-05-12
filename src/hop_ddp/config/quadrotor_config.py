"""
四旋翼无人机参数配置

包含不同场景的参数配置，用于HOP-LQR和HOP-DDP求解器
"""

import jax.numpy as jnp
from typing import Tuple, Optional
import inspect


class BaseConfig:
    def __init__(self) -> None:
        """初始化所有成员类递归地。忽略所有以'__'开头的名称（内置方法）。"""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # 遍历所有属性名
        for key in dir(obj):  # dir() 返回当前局部作用域中的名称列表
            # 忽略内置属性
            if key == "__class__":  # __class__ 是一个内置属性，表示对象的类
                continue
            # 获取对应的属性对象
            var = getattr(obj, key)
            # 检查属性是否为类
            if inspect.isclass(var):
                # 实例化该类
                i_var = var()
                # 将属性设置为实例而不是类型
                setattr(obj, key, i_var)
                # 递归初始化属性的成员
                BaseConfig.init_member_classes(i_var)


def class_to_dict(obj) -> dict:
    """
    将一个Python对象(类实例)转换为字典形式，递归地处理对象的属性
    
    参数:
        obj: 要转换的对象
    
    返回:
        dict: 转换后的字典
    """
    # 如果对象没有 __dict__ 属性(意味着它不是类实例)，直接返回对象本身
    # 这处理了基本数据类型(int, str等)和非类实例的情况
    if not hasattr(obj, "__dict__"):
        return obj
    
    result = {}  # 初始化结果字典

    # 遍历对象的所有属性: dir(obj) 获取对象的所有属性和方法名
    for key in dir(obj):
        if key.startswith("_"):  # 忽略以"_"开头的属性(通常是私有属性或特殊方法)
            continue
        element = []
        val = getattr(obj, key)  # 获取属性值
        if isinstance(val, list):  # 如果值是列表，递归处理列表中的每个元素
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)  # 如果不是列表，递归处理该值
        result[key] = element  # 将处理后的值存入结果字典
    return result

class QuadrotorConfig(BaseConfig):
    """四旋翼无人机配置"""
    
    # 物理参数
    mass: float = 1.0          # 质量 (kg)
    g: float = 9.81            # 重力加速度 (m/s^2)
    Ixx: float = 0.023        # 绕X轴的转动惯量 (kg·m^2)，对齐 public tutorial
    Iyy: float = 0.023        # 绕Y轴的转动惯量 (kg·m^2)，对齐 public tutorial
    Izz: float = 0.046        # 绕Z轴的转动惯量 (kg·m^2)
    arm_length: float = 0.25   # 机臂长度 (m) — 几何参数，用于碰撞检测
    yaw_moment_coeff: float = 0.1  # 电机升力到偏航力矩的系数
    kv: float = 0.05           # 线速度阻尼系数
    kw: float = 0.01           # 角速度阻尼系数

    # 控制参数: u = [F_total, tau_phi, tau_theta, tau_psi]
    # F_total: 总推力 (N)，沿机体Z轴方向
    # tau_phi/tau_theta/tau_psi: 滚转/俯仰/偏航力矩 (N·m)
    dt: float = 0.03           # 离散化时间步长 (s)，对齐 public tutorial
    tsteps: int = 400          # 时间步数，对齐 public tutorial
    @property
    def u_limit(self) -> tuple:
        return get_control_limits(self)
    # 初始状态 (12,)
    # [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
    x0: jnp.ndarray = jnp.array([
        0.0, 0.0, 2.0,      # 位置 (m) - 在1m高度悬停
        0.0, 0.0, 0.0,      # 速度
        0.0, 0.0, 0.0,      # 欧拉角
        0.0, 0.0, 0.0       # 角速度
    ])
    
    # 目标状态 (12,)
    # [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
    x_target: jnp.ndarray = jnp.array([
        10.0, 10.0, 2.5,      # 位置 (m) - 更接近初始位置
        0.0, 0.0, 0.0,      # 速度
        0.0, 0.0, 0.0,      # 欧拉角
        0.0, 0.0, 0.0       # 角速度
    ])
    
    # 时域惩罚权重。
    # HOP-DDP 内部的时域项是 w * T，其中 T 是离散步数。
    # 因此这里用“每秒惩罚”w_base 生成“每步惩罚”w = w_base * dt，
    # 避免细化 dt 并增加 tsteps 时，同一段物理时间的惩罚被重复放大。
    w_base: float = 1.0

    @property
    def w(self) -> float:
        return self.w_base * self.dt

    pass

    class CostMatrices:
        """Physical-coordinate weights matching the public tutorial."""
        # Public tutorial physical-coordinate weights:
        # Q = diag([5,5,5, 1,1,1, 20,20,10, 1,1,1])
        # R = diag([1e-3, 1e-2, 1e-2, 1e-2])
        # Q_T = 300 * I.
        q_pos: float = 5.0
        q_vel: float = 1.0
        q_roll: float = 20.0
        q_pitch: float = 20.0
        q_yaw: float = 10.0
        q_omega: float = 1.0
        r_thrust: float = 1e-3
        r_tau_phi: float = 1e-2
        r_tau_theta: float = 1e-2
        r_tau_psi: float = 1e-2
        q_pos_T: float = 300.0
        q_vel_T: float = 300.0
        q_att_T: float = 300.0
        q_omega_T: float = 300.0
        terminal_cost_scale: float = 1.0
    
    class ControlLimits:
        """单电机升力限制；运行时转换为 [F_total, tau_phi, tau_theta, tau_psi] 的盒约束。"""
        motor_f_min: float = 0.0
        # 单电机最大升力 = motor_f_max_factor * hover_per_motor。
        motor_f_max_factor: float = 4.0
        motor_f_max: Optional[float] = None

        u_min: Optional[jnp.ndarray] = None
        u_max: Optional[jnp.ndarray] = None
    
    class StateLimits:
        """状态限制"""
        # 位置限制
        pos_min: float = -10.0
        pos_max: float = 10.0
        
        # 速度限制
        vel_min: float = -5.0
        vel_max: float = 5.0
        
        # 姿态限制 (欧拉角)
        att_min: float = -jnp.pi / 4  # -45度
        att_max: float = jnp.pi / 4   # 45度
        
        # 角速度限制
        omega_min: float = -5.0
        omega_max: float = 5.0
        
        x_min: Optional[jnp.ndarray] = None
        x_max: Optional[jnp.ndarray] = None

    # 轨迹类型
    trajectory_type: str = "hover" # 悬停控制，可选 "line" 线轨迹跟踪

    class Diagnostics:
        """仅用于四旋翼实验的诊断开关"""
        enable_detailed_hop_lqr: bool = True
        enable_hop_lqr_summary_diagnostics: bool = True
        enable_hop_lqr_horizon_cost_breakdown: bool = False
        enable_hop_lqr_early_horizon_balance: bool = False
        enable_hop_lqr_recursive_growth_diagnostics: bool = True
        enable_hop_lqr_p0_direction_diagnostics: bool = True
        enable_hop_lqr_bar_f_matrix_diagnostics: bool = True
        enable_hop_lqr_prefix_invariance_diagnostics: bool = False
        hop_lqr_prefix_invariance_horizon: int = 12
        detailed_horizons: tuple = (1, 2, 3, 4)
        enable_hop_lqr_composite_map_diagnostics: bool = False
        enable_hop_lqr_single_step_map_diagnostics: bool = False
        enable_hop_lqr_query_stabilization: bool = False
        hop_lqr_bar_e_min_eig: float = 2e-4
        hop_lqr_temp_max_scale: float = 0.10
        hop_lqr_temp_min_cap: float = 1e-2
        hop_lqr_p0_min_eig: float = 5e-4
        hop_lqr_query_temp_relative_cap: float = 0.65
        enable_hop_lqr_query_adaptive_temp_relative_cap: bool = False
        hop_lqr_query_temp_relative_cap_min: float = 0.22
        hop_lqr_query_temp_relative_cap_trigger: float = 0.72
        enable_hop_lqr_validity_filter: bool = False
        hop_lqr_validity_filter_use_bad_m: bool = True
        hop_lqr_validity_filter_use_bad_p0: bool = True
        hop_lqr_validity_filter_use_bad_temp: bool = True
        hop_lqr_validity_filter_use_rank_deficient_f: bool = True
        hop_lqr_validity_eig_floor: float = 1e-10
        hop_lqr_validity_rank_rtol: float = 1e-8
        hop_lqr_min_candidate_horizon: int = 10
        enable_hop_lqr_recursive_bar_f_relative_cap: bool = False
        hop_lqr_recursive_bar_f_relative_cap: float = 0.35
        enable_hop_lqr_recursive_bar_g_eig_clip: bool = False
        hop_lqr_recursive_bar_g_min_relative_eig: float = -0.25
        hop_lqr_recursive_bar_g_max_relative_eig: float = 2.5
        enable_hop_lqr_recursive_bar_f_svd_floor: bool = False
        hop_lqr_recursive_bar_f_svd_rtol: float = 1e-3
        hop_lqr_recursive_bar_f_svd_atol: float = 1e-10
        hop_lqr_recursive_bar_f_max_growth: float = 1.1
        enable_hop_lqr_recursive_p0_stabilization: bool = False
        hop_lqr_recursive_bar_e_min_eig: float = 5e-4
        hop_lqr_recursive_temp_max_scale: float = 0.10
        hop_lqr_recursive_temp_min_cap: float = 1e-2
        hop_lqr_recursive_temp_relative_cap: float = 0.25
        enable_hop_lqr_recursive_adaptive_temp_relative_cap: bool = False
        hop_lqr_recursive_temp_relative_cap_min: float = 0.18
        hop_lqr_recursive_temp_relative_cap_trigger: float = 0.70
        enable_hop_lqr_query_bar_g_eig_clip: bool = False
        hop_lqr_query_bar_g_min_relative_eig: float = -0.30
        hop_lqr_query_bar_g_max_relative_eig: float = 3.0
        enable_hop_lqr_query_temp_relative_cap: bool = False
        enable_hop_lqr_query_bar_f_gain_cap: bool = False
        hop_lqr_query_bar_f_gain_cap: float = 0.95

    class Settings:
        """DDP solver tuning"""
        max_iter: int = 15
        tol: float = 1e-6
        warm_start_fixed_horizon_iters: int = 0
        enable_tstar_jump_clip: bool = False
        max_tstar_jump: int = 60
        T_min: int = 40  # HOP-LQR horizon search lower bound


def get_cost_matrices(config: QuadrotorConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    获取代价函数矩阵 (Q, R)
    
    参数:
        config: 四旋翼无人机配置
    
    返回:
        Q: 状态权重矩阵 (12, 12)
        R: 控制权重矩阵 (4, 4)
    """
    cost_matrices = config.CostMatrices
    # 组合状态权重矩阵
    Q_blocks = {
        "Q_pos": jnp.diag(jnp.array([cost_matrices.q_pos] * 3)),
        "Q_vel": jnp.diag(jnp.array([cost_matrices.q_vel] * 3)),
        "Q_att": jnp.diag(jnp.array([
            cost_matrices.q_roll,
            cost_matrices.q_pitch,
            cost_matrices.q_yaw,
        ])),
        "Q_omega": jnp.diag(jnp.array([cost_matrices.q_omega] * 3)),
    }
    Q = jnp.block([
        [Q_blocks["Q_pos"], jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), Q_blocks["Q_vel"], jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), Q_blocks["Q_att"], jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), Q_blocks["Q_omega"]]
    ])
    
    R = jnp.diag(jnp.array([
        cost_matrices.r_thrust,
        cost_matrices.r_tau_phi,
        cost_matrices.r_tau_theta,
        cost_matrices.r_tau_psi,
    ]))
    q_pos_T = cost_matrices.q_pos_T
    q_vel_T = cost_matrices.q_vel_T
    q_att_T = cost_matrices.q_att_T
    q_omega_T = cost_matrices.q_omega_T
    
    Q_T_blocks = {
        "Q_pos": jnp.diag(jnp.array([q_pos_T, q_pos_T, q_pos_T])),
        "Q_vel": jnp.diag(jnp.array([q_vel_T, q_vel_T, q_vel_T])),
        "Q_att": jnp.diag(jnp.array([q_att_T, q_att_T, q_att_T])),
        "Q_omega": jnp.diag(jnp.array([q_omega_T, q_omega_T, q_omega_T])),
    }
    Q_T_base = jnp.block([
        [Q_T_blocks["Q_pos"], jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), Q_T_blocks["Q_vel"], jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), Q_T_blocks["Q_att"], jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), Q_T_blocks["Q_omega"]]
    ])
    Q_T = cost_matrices.terminal_cost_scale * Q_T_base
    
    return Q, R, Q_T


def get_motor_thrust_limits(config: QuadrotorConfig) -> Tuple[float, float]:
    """Return per-motor thrust limits."""
    limits = config.ControlLimits
    f_min = float(limits.motor_f_min)
    hover_per_motor = config.mass * config.g / 4.0
    f_max = float(limits.motor_f_max_factor * hover_per_motor)
    return f_min, f_max


def get_thrust_moment_limits(config: QuadrotorConfig) -> Tuple[float, float, float, float, float]:
    """Return X-frame [T, tau_phi, tau_theta, tau_psi] box limits.

    The optimizer control is [F_total, tau_phi, tau_theta, tau_psi], while
    the physical limits are specified as per-motor thrust bounds. For an X
    quadrotor:

        tau_phi   = L/sqrt(2) * (-f1 + f2 + f3 - f4)
        tau_theta = L/sqrt(2) * (-f1 - f2 + f3 + f4)
        tau_psi   = c         * (-f1 + f2 - f3 + f4)

    This returns the independent box bounds used by jnp.clip in [T, tau].
    """
    f_min, f_max = get_motor_thrust_limits(config)
    thrust_span = f_max - f_min
    arm_factor = config.arm_length / jnp.sqrt(2.0)
    F_min = 4.0 * f_min  # 4.0 表示四个螺旋桨的总推力
    F_max = 4.0 * f_max  # 4.0 表示四个螺旋桨的总推力
    tau_phi_max = 2.0 * arm_factor * thrust_span  # 2.0 表示两对螺旋桨的总力矩
    tau_theta_max = 2.0 * arm_factor * thrust_span  # 2.0 表示两对螺旋桨的总力矩
    tau_psi_max = 2.0 * config.yaw_moment_coeff * thrust_span
    return F_min, F_max, tau_phi_max, tau_theta_max, tau_psi_max


def get_control_limits(config: QuadrotorConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    获取控制输入限制
    参数:
        config: 四旋翼无人机配置
    返回:
        u_min: 最小控制输入 (4,) — [F_min, -tau_phi_max, -tau_theta_max, -tau_psi_max]
        u_max: 最大控制输入 (4,) — [F_max,  tau_phi_max,  tau_theta_max,  tau_psi_max]
    """
    limits = config.ControlLimits

    if limits.u_min is None:
        F_min, F_max, tau_phi_max, tau_theta_max, tau_psi_max = get_thrust_moment_limits(config)
        limits.u_min = jnp.array([F_min, -tau_phi_max, -tau_theta_max, -tau_psi_max])
        limits.u_max = jnp.array([F_max,  tau_phi_max,  tau_theta_max,  tau_psi_max])
    return limits.u_min, limits.u_max


def get_state_limits(config: QuadrotorConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    获取状态限制
    
    参数:
        config: 四旋翼无人机配置
    
    返回:
        x_min: 最小状态 (12,)
        x_max: 最大状态 (12,)
    """
    state_limits = config.StateLimits
    
    if state_limits.x_min is None:
        state_limits.x_min = jnp.array([
            state_limits.pos_min, state_limits.pos_min, state_limits.pos_min,  # 位置
            state_limits.vel_min, state_limits.vel_min, state_limits.vel_min,  # 速度
            state_limits.att_min, state_limits.att_min, state_limits.att_min,  # 姿态
            state_limits.omega_min, state_limits.omega_min, state_limits.omega_min  # 角速度
        ])
    
    if state_limits.x_max is None:
        state_limits.x_max = jnp.array([
            state_limits.pos_max, state_limits.pos_max, state_limits.pos_max,  # 位置
            state_limits.vel_max, state_limits.vel_max, state_limits.vel_max,  # 速度
            state_limits.att_max, state_limits.att_max, state_limits.att_max,  # 姿态
            state_limits.omega_max, state_limits.omega_max, state_limits.omega_max  # 角速度
        ])
    
    return state_limits.x_min, state_limits.x_max
