#!/usr/bin/env python3
"""
四旋翼无人机平坦动力学模型 (Flat Output Dynamics)
动力学方程：线性积分器链
映射方程：非线性代数映射 (Flat -> Physical)
"""
import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass

@dataclass
class FlatQuadrotorParams:
    """物理参数 (与原来一致)"""
    mass: float = 1.0
    g: float = 9.81
    Ixx: float = 0.0023
    Iyy: float = 0.0023
    Izz: float = 0.0040
    arm_length: float = 0.25
    c: float = 0.1

class FlatQuadrotorDynamics:
    """
    基于微分平坦性的四旋翼动力学模型
    
    状态 x_flat (16维): [sigma, sigma_dot, sigma_ddot, sigma_dddot]
       sigma = [x, y, z, psi]
    控制 u_flat (4维): [x^(4), y^(4), z^(4), psi^(4)] (Snap)
    
    动力学: x_dot = A * x + B * u (线性)
    映射: (x_flat, u_flat) -> (Physical_State, Physical_Input) (非线性)
    """
    
    def __init__(self, params: FlatQuadrotorParams, dt: float = 0.05):
        self.params = params
        self.dt = dt
        self.nx = 16
        self.nu = 4
        
        # 转动惯量向量
        self.I_vec = jnp.array([params.Ixx, params.Iyy, params.Izz])
        self.I_mat = jnp.diag(self.I_vec)
        
        # 构建线性系统矩阵 A (16x16) 和 B (16x4)
        # 结构: 4个独立的4阶积分器
        I4 = jnp.eye(4)
        Z4 = jnp.zeros((4, 4))
        
        # A 矩阵: 移位寄存器结构
        self.A = jnp.block([
            [Z4, I4, Z4, Z4],
            [Z4, Z4, I4, Z4],
            [Z4, Z4, Z4, I4],
            [Z4, Z4, Z4, Z4]
        ])
        
        # B 矩阵: 控制输入作用于最高阶导数
        self.B = jnp.block([
            [Z4],
            [Z4],
            [Z4],
            [I4]
        ])

    def linear_dynamics(self, x_flat: jnp.ndarray, u_flat: jnp.ndarray) -> jnp.ndarray:
        """
        线性动力学方程 (连续时间)
        x_dot = A * x + B * u
        """
        return self.A @ x_flat + self.B @ u_flat

    def discrete_linear_dynamics(self, x_flat: jnp.ndarray, u_flat: jnp.ndarray) -> jnp.ndarray:
        """
        离散时间线性动力学 (前向欧拉)
        由于是线性的，也可以精确离散化，但欧拉对于小dt足够且简单
        """
        x_dot = self.linear_dynamics(x_flat, u_flat)
        return x_flat + x_dot * self.dt

    def flat_to_physical(self, x_flat: jnp.ndarray, u_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        核心映射函数：将平坦状态/输入映射回物理状态和控制输入
        
        参数:
            x_flat: (16,) [x,y,z,psi, ..., dddx,dddy,dddz,dddpsi]
            u_flat: (4,)  [ddddx, ddddy, ddddz, ddddpsi]
            
        返回:
            x_phys: (12,) [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
            u_phys: (4,)  [f1, f2, f3, f4] (电机推力)
        """
        # --- 1. 提取平坦变量 ---
        pos = x_flat[0:3]       # x, y, z
        yaw = x_flat[3]         # psi
        
        vel = x_flat[4:7]       # dx, dy, dz
        yaw_rate = x_flat[7]    # dpsi
        
        acc = x_flat[8:11]      # ddx, ddy, ddz
        yaw_acc = x_flat[11]    # ddpsi
        
        jerk = x_flat[12:15]    # dddx, dddy, dddz
        yaw_jerk = x_flat[15]   # dddpsi
        
        snap = u_flat           # ddddx, ...

        # --- 2. 计算物理状态 ---
        
        # 2.1 位置和速度 (直接对应)
        px, py, pz = pos
        vx, vy, vz = vel
        
        # 2.2 计算机体坐标系 Z 轴 (z_B)
        # 公式: z_B = (acc + g*e3) / ||acc + g*e3||
        g_vec = jnp.array([0.0, 0.0, self.params.g])
        thrust_vec = acc + g_vec
        norm_thrust = jnp.linalg.norm(thrust_vec)
        
        # 防止除以零 (自由落体奇异点)
        norm_thrust = jnp.maximum(norm_thrust, 1e-4)
        z_B = thrust_vec / norm_thrust
        
        # 2.3 计算机体坐标系 X, Y 轴
        # 期望的前向向量 (基于 yaw)
        x_C = jnp.array([jnp.cos(yaw), jnp.sin(yaw), 0.0])
        
        # y_B = z_B x x_C (归一化)
        y_B_unnorm = jnp.cross(z_B, x_C)
        norm_y = jnp.linalg.norm(y_B_unnorm)
        norm_y = jnp.maximum(norm_y, 1e-4)
        y_B = y_B_unnorm / norm_y
        
        # x_B = y_B x z_B
        x_B = jnp.cross(y_B, z_B)
        
        # 构建旋转矩阵 R = [x_B, y_B, z_B]
        R = jnp.column_stack([x_B, y_B, z_B])
        
        # 2.4 从 R 提取欧拉角 (Z-Y-X 顺序: psi, theta, phi)
        # R = Rz(psi) * Ry(theta) * Rx(phi)
        # 已知 psi (yaw)，我们可以直接解算 theta 和 phi
        # R[2,0] = -sin(theta) => theta = -asin(R[2,0])
        # R[2,1] = sin(phi)cos(theta) => phi = atan2(R[2,1], R[2,2])
        # 注意：这里我们已经有 psi 了，可以直接用几何关系验证，或者重新计算以保持一致性
        # 为了数值稳定性，通常直接用反正切从矩阵元素提取
        theta = jnp.arcsin(jnp.clip(-R[2, 0], -1.0, 1.0))
        phi = jnp.arctan2(R[2, 1], R[2, 2])
        psi = yaw # 直接使用平坦输出的 psi，保证一致性
        
        # 2.5 计算角速度 omega = [p, q, r]
        # 方法：利用 z_B 的导数
        # dot(z_B) = omega x z_B  =>  omega_xy = z_B x dot(z_B)
        # 首先计算 dot(z_B) (需要对 thrust_vec 求导)
        # dot(thrust_vec) = jerk
        # dot(z_B) = (jerk * norm - thrust * (thrust . jerk)/norm) / norm^2
        dot_thrust = jerk
        dot_norm = jnp.dot(thrust_vec, dot_thrust) / norm_thrust
        dot_z_B = (dot_thrust * norm_thrust - thrust_vec * dot_norm) / (norm_thrust ** 2)
        
        # omega 的水平分量 (在机体坐标系下需要转换，这里先算世界系下的等效旋转)
        # 实际上 omega = R^T * (z_B x dot(z_B) + psi_dot * e3) ? 
        # 更简单的推导：
        # omega_world = z_B x dot(z_B) + psi_dot * z_B (近似，严格推导需用 R)
        # 标准公式: omega_body = R.T @ (z_B_cross_dot_zB + psi_dot * e3)
        # 其中 z_B_cross_dot_zB 是世界系下由倾斜变化引起的角速度分量
        
        # 计算世界系下的角速度向量
        # w_tilt = z_B x dot(z_B)
        w_tilt = jnp.cross(z_B, dot_z_B)
        w_yaw = jnp.array([0.0, 0.0, yaw_rate])
        omega_world = w_tilt + w_yaw
        
        # 转换到机体坐标系
        omega_body = R.T @ omega_world
        p, q, r = omega_body

        # 组装物理状态 (12,)
        x_phys = jnp.array([
            px, py, pz,
            vx, vy, vz,
            phi, theta, psi,
            p, q, r
        ])
        
        # --- 3. 计算物理控制输入 (电机推力) ---
        
        # 3.1 总推力 T
        T = self.params.mass * norm_thrust
        
        # 3.2 计算力矩 tau
        # 需要角加速度 omega_dot
        # omega_dot 依赖于 snap (u_flat)
        # 这是一个繁琐的解析求导过程，为了代码简洁和JAX效率，我们可以利用自动微分思想
        # 但在这里我们必须写出显式关系或者利用 JAX 的 jvp/vjp? 
        # 不，DDP需要显式函数。我们需要推导 dot(omega_body)
        # 由于推导极其复杂，这里采用一种工程近似或直接利用 JAX 在 cost 函数内部微分
        # 但在 dynamics 类中，我们需要显式返回 u_phys。
        # 让我们尝试简化推导：
        # tau = I * omega_dot + omega x (I * omega)
        # 关键在于 omega_dot。
        # omega_body = R.T @ ( (z_B x dot(z_B)) + psi_dot*e3 )
        # 对时间求导会引入 ddot(z_B)，而 ddot(z_B) 依赖于 snap (u_flat).
        
        # 为了保持代码可运行且不过于冗长，这里实现一个简化的解析版本
        # 注意：在生产环境中，建议将此部分写成独立的 JAX JIT 函数并让 JAX 自动处理高阶导数
        # 但为了配合 DDP 的手动雅可比需求，我们最好有显式表达式。
        # 鉴于篇幅，这里使用一个技巧：我们只计算 T，力矩部分通过反推动力学得到
        # 或者，我们可以直接计算需要的力矩来产生所需的 omega_dot
        # 既然我们有 u_flat (snap)，我们可以数值估算 omega_dot? 不行，DDP需要解析。
        
        # 【替代方案】：我们不在这里硬算复杂的 tau 解析解，而是利用 JAX 的自动微分能力
        # 在 DDP 的 Cost 函数中，我们调用 flat_to_physical，然后 JAX 会自动帮我们求
        # Cost 对 u_flat 的梯度，其中包含了 tau 对 snap 的依赖链。
        # 所以这里我们只需要给出一个“名义上”的 u_phys 用于计算 Cost 值即可。
        # 真正的梯度由 JAX 的 autodiff 处理。
        # 但是，为了计算 u_phys (f1..f4) 的值来检查饱和，我们需要 tau。
        # 让我们做一个简化的假设或使用 JAX 内部求导来辅助计算这个特定值
        
        # 为了完整性，这里提供一个基于数值微分的 tau 估算 (仅在计算 Cost 值时使用，不影响梯度流)
        # 更好的做法：在外部计算 Cost 时，直接定义 Cost(u_flat) = PhysicalCost( Map(x, u) )
        # 这样 JAX 会自动处理所有链条。
        # 所以这里我们只需要返回 T 和一个近似的 tau 用于显示或软约束检查。
        # 实际上，如果我们要严格计算 f1..f4，必须解算 tau。
        # 让我们用 JAX 的 jacfwd 在这里即时计算 omega_dot 关于 snap 的部分？
        # 不，这会破坏 JIT。
        # 妥协：我们只计算 T。对于力矩，我们假设一个简化的线性关系或者留空，
        # 因为在 DDP 中，只要 Cost 函数定义正确，梯度就是对的。
        # 但为了生成 u_phys 向量，我们需要估算 tau。
        
        # 重新审视：omega_body 是 x_flat 的函数 (含 jerk)。
        # omega_dot 是 x_flat 和 u_flat 的函数。
        # 我们可以定义一个内部函数 calc_omega_dot，然后用 jax.jit 包装它吗？
        # 在类方法中这样做比较麻烦。
        # 让我们手动推导关键项：
        # dot(z_B) 依赖 jerk.
        # ddot(z_B) 依赖 snap.
        # omega_dot 将包含 ddot(z_B) 项。
        
        # 鉴于推导的极端复杂性 (超过200行符号运算)，在现代 JAX 工作流中，
        # 推荐做法是：不要在这里硬编码 tau 的解析解。
        # 而是：在 DDP 的 Cost 函数中，直接计算 "Desired Torque" 作为中间变量，
        # 或者，我们这里只返回 T，并假设 tau 可以通过逆向动力学唯一确定。
        # 为了满足接口，我们这里用一个简化模型计算 tau，或者抛出警告。
        # **修正**：为了代码的可用性，我将实现一个基于 JAX 自动微分的“即时”导数计算来获取 omega_dot
        # 这在每次调用时会有一点开销，但保证了正确性。
        
        def compute_omega_dot(x_f, u_f):
            # 重新运行 flat_to_physical 的前半部分来获取 omega
            # 为了避免递归，我们复制上面的逻辑到一个静态辅助函数
            # 这里为了简洁，我们假设有一个辅助函数 get_omega(x_f)
            # 实际上，我们可以直接对 flat_to_physical 的输出求导？
            # 不行，flat_to_physical 返回 x_phys，我们需要 omega_dot。
            # 让我们定义一个局部函数只计算 omega
            pos_l, yaw_l = x_f[0:3], x_f[3]
            acc_l, jerk_l = x_f[8:11], x_f[12:15]
            thrust_vec_l = acc_l + jnp.array([0.,0.,self.params.g])
            norm_t_l = jnp.linalg.norm(thrust_vec_l)
            z_B_l = thrust_vec_l / jnp.maximum(norm_t_l, 1e-4)
            dot_thrust_l = jerk_l
            dot_norm_l = jnp.dot(thrust_vec_l, dot_thrust_l) / norm_t_l
            dot_z_B_l = (dot_thrust_l * norm_t_l - thrust_vec_l * dot_norm_l) / (norm_t_l ** 2)
            x_C_l = jnp.array([jnp.cos(yaw_l), jnp.sin(yaw_l), 0.0])
            y_B_l = jnp.cross(z_B_l, x_C_l)
            y_B_l = y_B_l / jnp.maximum(jnp.linalg.norm(y_B_l), 1e-4)
            x_B_l = jnp.cross(y_B_l, z_B_l)
            R_l = jnp.column_stack([x_B_l, y_B_l, z_B_l])
            w_tilt_l = jnp.cross(z_B_l, dot_z_B_l)
            yaw_rate_l = x_f[7]
            omega_w_l = w_tilt_l + jnp.array([0.,0.,yaw_rate_l])
            return R_l.T @ omega_w_l

        # 使用 JAX 自动计算 omega_dot 关于时间的导数 (即关于 x_flat 和 u_flat 的链式法则)
        # 由于 x_flat 的导数是已知的 (shift)，只有最后一项依赖 u_flat
        # dot(omega) = d(omega)/dx * x_dot + d(omega)/du * u_dot (u_dot is not defined)
        # 其实 omega 是 x_flat 的函数。 omega_dot = Jacobian(omega, x_flat) @ x_flat_dot
        # x_flat_dot 的最后一项是 u_flat.
        jac_omega = jax.jacfwd(compute_omega_dot, argnums=0)(x_flat, u_flat)
        x_flat_dot = self.linear_dynamics(x_flat, u_flat)
        omega_dot = jac_omega @ x_flat_dot
        
        # 计算力矩
        omega = jnp.array([p, q, r])
        I_omega = self.I_mat @ omega
        tau = self.I_mat @ omega_dot + jnp.cross(omega, I_omega)
        
        # 3.3 从 T 和 tau 解算电机推力 (X型)
        # T = f1+f2+f3+f4
        # tau_x = L/sqrt(2) * (f2+f3-f1-f4)
        # tau_y = L/sqrt(2) * (f3+f4-f1-f2)
        # tau_z = c * (f2+f4-f1-f3)
        L = self.params.arm_length
        factor = L / jnp.sqrt(2.0)
        c = self.params.c
        
        # 构建映射矩阵 M: [T, tx, ty, tz]^T = M * [f1, f2, f3, f4]^T
        M = jnp.array([
            [1, 1, 1, 1],
            [-factor, factor, factor, -factor],
            [-factor, -factor, factor, factor],
            [-c, c, -c, c]
        ])
        
        rhs = jnp.array([T, tau[0], tau[1], tau[2]])
        # 求解 f = M^-1 * rhs
        f_motors = jnp.linalg.solve(M, rhs)
        
        u_phys = f_motors
        
        return x_phys, u_phys

    def get_linearized_dynamics(self, x_flat: jnp.ndarray, u_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        返回线性化矩阵。
        对于平坦模型，A 和 B 是常数矩阵！
        """
        return self.A, self.B

    def get_discrete_linearized_dynamics(self, x_flat: jnp.ndarray, u_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        离散化线性矩阵 (Ad, Bd)
        Ad = I + A*dt, Bd = B*dt (欧拉法)
        或者精确离散化 (对于积分器链，欧拉法就是精确的如果输入是分段常数)
        """
        Ad = jnp.eye(self.nx) + self.A * self.dt
        Bd = self.B * self.dt
        return Ad, Bd