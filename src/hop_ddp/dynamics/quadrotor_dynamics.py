"""
四旋翼无人机动力学模型

基于12状态、4控制的四旋翼无人机动力学模型
状态 x: [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
控制 u: [F_total, tau_phi, tau_theta, tau_psi] 总推力 + 三轴力矩
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from config.quadrotor_config import QuadrotorConfig

class QuadrotorDynamics:
    """
    四旋翼无人机动力学模型类
    状态维度: 12
    控制维度: 4
    状态向量 x:
        [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        px, py, pz: 位置 (m)
        vx, vy, vz: 速度 (m/s)
        phi, theta, psi: 滚转、俯仰、偏航角 (rad)
        p, q, r: 机体坐标系下的角速度 (rad/s)
    
    控制向量 u:
        [F_total, tau_phi, tau_theta, tau_psi]
        F_total: 总推力 (N)，沿机体Z轴
        tau_phi/tau_theta/tau_psi: 滚转/俯仰/偏航力矩 (N·m)
    """
    
    def __init__(self, config: QuadrotorConfig):
        """
        初始化四旋翼无人机动力学模型
        
        参数:
            params: 四旋翼无人机物理参数
            dt: 离散化时间步长 (s)
        """
        self.dt = config.dt
        
        # 状态和控制维度
        self.nx = 12  # 状态维度
        self.nu = 4   # 控制维度
        
        # 物理参数
        self.mass = config.mass
        self.g = config.g
        self.Ixx = config.Ixx
        self.Iyy = config.Iyy
        self.Izz = config.Izz
        self.arm_length = config.arm_length
        self.kv = config.kv
        self.kw = config.kw
        
        # 转动惯量矩阵
        self.I = jnp.array([self.Ixx, self.Iyy, self.Izz])
        self.I_mat = jnp.diag(self.I)
    
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        四旋翼无人机连续时间动力学方程
        参数:
            x: 状态向量 (12,)
            u: 控制向量 (4,)
        返回:
            x_dot: 状态导数 (12,)
        """
        # 从状态向量中提取变量
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]
        # 控制量直接为 [F_total, tau_phi, tau_theta, tau_psi]
        F = u[0]
        tau = u[1:4]
        # 2. 构建旋转矩阵 (从机体坐标系到世界坐标系)
        # 使用Z-Y-X欧拉角 (yaw-pitch-roll) 顺序
        Rz = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                        [jnp.sin(psi), jnp.cos(psi), 0],
                        [0, 0, 1]])
        Ry = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                        [0, 1, 0],
                        [-jnp.sin(theta), 0, jnp.cos(theta)]])
        Rx = jnp.array([[1, 0, 0],
                        [0, jnp.cos(phi), -jnp.sin(phi)],
                        [0, jnp.sin(phi), jnp.cos(phi)]])
        R = Rz @ Ry @ Rx
        # 3. 计算线加速度 (在世界坐标系中)
        thrust_body = jnp.array([0.0, 0.0, F])
        thrust_world = R @ thrust_body
        gravity_world = jnp.array([0.0, 0.0, -self.mass * self.g])
        acc_world = (thrust_world + gravity_world) / self.mass - self.kv * jnp.array([vx, vy, vz])
        
        # 4. 计算角加速度 (在机体坐标系中)
        omega = jnp.array([p, q, r])
        I_omega = self.I_mat @ omega
        omega_cross_I_omega = jnp.cross(omega, I_omega)
        omega_dot = jnp.linalg.solve(self.I_mat, tau - omega_cross_I_omega) - self.kw * omega
        
        # 5. 角速度到欧拉角导数的变换 (运动学方程)
        sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
        tan_theta, sec_theta = jnp.tan(theta), 1.0 / jnp.cos(theta)
        
        phi_dot = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
        theta_dot = q * cos_phi - r * sin_phi
        psi_dot = q * sin_phi * sec_theta + r * cos_phi * sec_theta
        
        # 6. 组装状态导数
        x_dot = jnp.array([
            vx, vy, vz,                    # 位置导数 = 速度
            acc_world[0], acc_world[1], acc_world[2],  # 速度导数 = 加速度
            phi_dot, theta_dot, psi_dot,   # 欧拉角导数
            omega_dot[0], omega_dot[1], omega_dot[2]   # 角加速度
        ])
        
        return x_dot
    
    def discrete_dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        四旋翼无人机离散时间动力学方程 (前向欧拉离散化)
        参数:
            x: 当前状态 (12,)
            u: 控制输入 (4,)
        
        返回:
            x_next: 下一个状态 (12,)
        """
        x_dot = self.dynamics(x, u)
        return x + x_dot * self.dt

    def discrete_dynamics_rk4(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        四旋翼无人机离散时间动力学方程 (RK4 离散化)

        参数:
            x: 当前状态 (12,)
            u: 控制输入 (4,)

        返回:
            x_next: 下一个状态 (12,)
        """
        dt = self.dt
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    def get_linearized_dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        获取线性化动力学矩阵 (A, B)
        
        使用JAX自动微分计算雅可比矩阵
        
        参数:
            x: 状态向量 (12,)
            u: 控制向量 (4,)
        
        返回:
            A: 状态矩阵 (12, 12)
            B: 控制矩阵 (12, 4)
        """
        # 使用JAX自动微分计算雅可比矩阵
        A = jax.jacfwd(self.dynamics, argnums=0)(x, u)
        B = jax.jacfwd(self.dynamics, argnums=1)(x, u)
        
        return A, B
    
    def get_discrete_linearized_dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        获取离散时间线性化动力学矩阵 (A_d, B_d)
        
        参数:
            x: 状态向量 (12,)
            u: 控制向量 (4,)
        
        返回:
            A_d: 离散状态矩阵 (12, 12)
            B_d: 离散控制矩阵 (12, 4)
        """
        # 使用JAX自动微分计算离散动力学雅可比矩阵
        A_d = jax.jacfwd(self.discrete_dynamics, argnums=0)(x, u)
        B_d = jax.jacfwd(self.discrete_dynamics, argnums=1)(x, u)
        
        return A_d, B_d
    
    def hover_control(self) -> jnp.ndarray:
        """
        计算悬停时的控制输入: [F = mg, tau = 0, 0, 0]

        返回:
            u_hover: 悬停控制输入 (4,)
        """
        return jnp.array([self.mass * self.g, 0.0, 0.0, 0.0])

    def delta_to_absolute_control(self, delta_u: jnp.ndarray) -> jnp.ndarray:
        """偏差控制 → 绝对控制"""
        return self.hover_control() + delta_u

    def absolute_to_delta_control(self, u_abs: jnp.ndarray) -> jnp.ndarray:
        """绝对控制 → 偏差控制"""
        return u_abs - self.hover_control()

    def linear_trajectory_control(self, x0: jnp.ndarray, x_target: jnp.ndarray,
                             tsteps: int, dt: float) -> jnp.ndarray:
        """生成初始控制轨迹（delta_u = 0，即悬停）。"""
        return jnp.zeros((tsteps, 4))
