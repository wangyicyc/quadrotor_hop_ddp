"""
HOP-DDP (Horizon Optimal Planning with Differential Dynamic Programming) 求解器

基于论文《HOP: Fast Differential Dynamic Programming for Horizon-Optimal Trajectory Planning》的 Algorithm 2
该方法将非线性时优控制问题转化为时变线性二次子问题，并使用 HOP-LQR 高效选择最优时域

核心贡献：
1. 将非线性动力学线性化、成本函数二次化
2. 通过状态扩充将问题转化为 HOP-LQR 可解的标准形式
3. 迭代优化轨迹和时域，直至收敛
"""
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT)) 

import jax
# 必须在任何 jnp 操作之前设置
jax.config.update("jax_enable_x64", True) 
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax import vmap, jit, grad, hessian, jacrev, lax, jacfwd
from jax.lax import scan
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import time
from hop_lib.hop_lqr_solver import HOPLQRSolver
from hop_lib.utils import get_logger

_logger = get_logger('hop')

import numpy as np


@dataclass
class DDPResult:
    """DDP 求解结果容器"""
    u_opt: jnp.ndarray      # 最优控制序列
    x_opt: jnp.ndarray      # 最优状态轨迹
    T_star: int             # 最终确定的最优时域
    total_cost: float       # 总成本（包含时域惩罚）
    control_cost: float     # 控制成本（不含时域惩罚）
    iteration_costs: list   # 每次迭代的成本历史
    iteration_T_stars: list # 每次迭代选择的最优时域


class HOPDDPSolver:
    """
    核心功能：高效求解非线性时优控制问题
    问题定义：
        min_{U_T, T} J = φ(x_T) + Σ_{k=0}^{T-1} ℓ(x_k, u_k) + wT
        s.t. x_{k+1} = f(x_k, u_k)
             T ∈ {1, 2, ..., N}
    
    算法流程：
    1. 线性化和增广：在当前轨迹附近线性化动力学，二次化成本
    2. 时域选择：使用 HOP-LQR 选择最优时域 T*
    3. 截断的反向传播：在时域 T* 上执行 DDP 反向传播
    4. 前向推演和更新：线搜索，更新轨迹
    """
    def __init__(self, f: Callable, l: Callable, phi: Callable, 
                 n: int, m: int, u_limit=None, trajectory_generator=None,
                 x_target=None, iteration_callback: Optional[Callable] = None,
                 config=None, f_linearization: Optional[Callable] = None,
                 linearization_method: str = "autodiff",
                 finite_difference_eps: float = 1e-5,
                 wrap_indices: Optional[Tuple[int, ...]] = None):
        """
        初始化 HOP-DDP 求解器
        参数:
            f: 离散动力学函数，f(x, u) -> 下一状态
            l: 阶段成本函数，l(x, u) -> 标量成本
            phi: 终端成本函数，phi(x) -> 标量成本
            n: 状态维度
            m: 控制维度
            T_max: 最大候选时域；传入 config 时可省略
            u_limit: 控制限制，可以是单个数值（对称限制）或元组 (u_min, u_max)
            trajectory_generator: 可选的轨迹生成函数，trajectory_generator(x0, x_target, tsteps, dt) -> control_trajectory
            x_target: 目标状态
            iteration_callback: 可选的每轮迭代回调，签名为
                callback(iteration, x_traj, u_traj, T_star, cost, accepted_step)
            config: 可选配置对象。若提供，则读取 dt、w、tsteps、DDPSettings
                和 Diagnostics。
            f_linearization: 可选的 HOP-LQR 局部线性化动力学。若提供，
                forward rollout、trajectory cost 和 DDP backward 仍使用 f，
                只有 _compute_augmented_system 构造 HOP-LQR 增广系统时使用
                f_linearization。
        """
        self.f = f
        self.f_linearization = f if f_linearization is None else f_linearization
        self.l = l
        self.phi = phi
        self.n = n
        self.m = m
        self.trajectory_generator = trajectory_generator
        self.x_target = x_target
        self.iteration_callback = iteration_callback
        self.u_min, self.u_max = u_limit
        self.wrap_indices = tuple(wrap_indices or ())
        self.linearization_method = linearization_method
        self.finite_difference_eps = float(finite_difference_eps)
        if linearization_method == "finite_difference":
            eps = self.finite_difference_eps
            eye_x = jnp.eye(n)
            eye_u = jnp.eye(m)

            def finite_diff_x(f_eval):
                def jac_x(x, u):
                    cols = vmap(lambda e: (f_eval(x + eps * e, u) - f_eval(x - eps * e, u)) / (2.0 * eps))(eye_x)
                    return cols.T
                return jac_x

            def finite_diff_u(f_eval):
                def jac_u(x, u):
                    cols = vmap(lambda e: (f_eval(x, u + eps * e) - f_eval(x, u - eps * e)) / (2.0 * eps))(eye_u)
                    return cols.T
                return jac_u

            self.f_x = jit(finite_diff_x(f))
            self.f_u = jit(finite_diff_u(f))
            self.f_linearization_x = jit(finite_diff_x(self.f_linearization))
            self.f_linearization_u = jit(finite_diff_u(self.f_linearization))
        elif linearization_method == "autodiff":
            self.f_x = jit(jacfwd(f, 0))  # 关于 x 的雅可比矩阵，形状 (n, n)
            self.f_u = jit(jacfwd(f, 1))  # 关于 u 的雅可比矩阵，形状 (n, m)
            self.f_linearization_x = jit(jacfwd(self.f_linearization, 0))
            self.f_linearization_u = jit(jacfwd(self.f_linearization, 1))
        else:
            raise ValueError(f"Unsupported linearization_method: {linearization_method}")
        # 成本函数 l 输出标量，可以使用 grad
        self.l_x = jit(grad(l, 0))  # 关于 x 的梯度，形状 (n,)
        self.l_u = jit(grad(l, 1))  # 关于 u 的梯度，形状 (m,)
        self.l_xx = jit(hessian(l, 0))  # 关于 x 的 Hessian 矩阵，形状 (n, n)
        self.l_uu = jit(hessian(l, 1))  # 关于 u 的 Hessian 矩阵，形状 (m, m)
        # 计算混合二阶导数 l_ux
        # 先对 u 求梯度，再对 x 求雅可比
        self.l_ux = jit(jacrev(grad(l, 1), 0))  # 形状 (m, n)
        # 终端成本函数 phi 输出标量
        self.phi_x = jit(grad(phi))  # 关于 x 的梯度，形状 (n,)
        self.phi_xx = jit(hessian(phi))  # 关于 x 的 Hessian 矩阵，形状 (n, n)
        # 初始化 HOP-LQR 求解器
        self.hop_lqr = HOPLQRSolver(n+1, m)  # 注意：增广状态维度是 n+1
        # Match the public tutorial iLQR backward-pass regularization.
        self.reg_min = 1e-6
        self.reg_max = 1e3
        self.reg_factor = 10.0
        self.reg = 1e-3
        self.cost_tol = 1e-4
        # Debug 默认优先观察原始 HOP-LQR 选时域，不额外裁剪 T*。
        self.matrix_cond_warn = 1e12
        # DDP extension: allow each candidate horizon to use its own terminal
        # quadratic surrogate Q_T(x_t) instead of sharing Q_T(x_N).
        # Set False to recover the paper-style shared terminal matrix.
        self.use_per_horizon_terminal_surrogate = True
        # Debug: compare HOP-LQR predicted J with real DDP rollout cost on a
        # compact set of candidate horizons. This prints evidence but does not
        # change the chosen T*.
        self.enable_predicted_vs_rollout_diagnostics = True
        self.predicted_vs_rollout_top_k = 5
        self.predicted_vs_rollout_neighbor_step = 10
        self.enable_tstar_jump_clip = True
        self.max_tstar_jump = 60
        if config is not None:
            self.apply_config(config)

    def _state_difference(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """State difference with optional angle wrapping, matching the public tutorial _wrap helper."""
        diff = x - y
        for idx in self.wrap_indices:
            diff = diff.at[idx].set((diff[idx] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi)
        return diff

    def apply_config(self, config) -> None:
        """从配置对象批量读取 DDP 和 HOP-LQR 参数。"""
        ddp = config.Settings
        self.dt = config.dt
        self.w = config.w
        self.T_max = config.tsteps
        self.warm_start_iters = ddp.warm_start_fixed_horizon_iters
        self.enable_tstar_jump_clip = ddp.enable_tstar_jump_clip
        self.max_tstar_jump = ddp.max_tstar_jump
        self.hop_lqr.apply_settings(config.Diagnostics)

    @partial(jit, static_argnums=(0,))
    def _compute_augmented_system(self, x_traj: jnp.ndarray, u_traj: jnp.ndarray) -> tuple:
        """
        计算增广系统矩阵 - 对应论文公式(22)-(30)
        
        参数:
            x_traj: 当前状态轨迹 (N+1, n)，其中 N 是控制步数
            u_traj: 当前控制轨迹 (N, m)
        
        返回:
            A_aug: 增广状态矩阵 (N, n+1, n+1) - 对应每个控制步
            B_aug: 增广控制矩阵 (N, n+1, m)
            Q_aug: 增广状态成本矩阵 (N, n+1, n+1)
            R: 控制成本矩阵 (N, m, m)
            a_k: 动力学偏移量 (N, n)
        """
        N = len(u_traj)
        n, m = self.n, self.m
        # 初始化 (n+1, n+1)
        zeros_nn = jnp.zeros((n, n))
        zeros_n1 = jnp.zeros((n, 1))
        zeros_1n = jnp.zeros((1, n))
        # 1. 准备数据切片 (Batch 维度为 N)
        x_curr = x_traj[:-1]  # (N, n)
        x_next = x_traj[1:]   # (N, n)
        # 2. 定义单步处理函数 (输入是单个时间步的数据，不是索引)
        def compute_step_k(x_k, u_k, x_next_k):
            # 动力学计算
            f_val = self.f(x_k, u_k)
            A = self.f_linearization_x(x_k, u_k)
            B = self.f_linearization_u(x_k, u_k)
            # 偏移量 a_k
            a = f_val - x_next_k
            # 成本导数
            l_val = self.l(x_k, u_k)
            l_x_val = self.l_x(x_k, u_k)
            l_u_val = self.l_u(x_k, u_k)
            l_xx_val = self.l_xx(x_k, u_k)
            l_uu_val = self.l_uu(x_k, u_k)
            l_ux_val = self.l_ux(x_k, u_k)
            # 【添加打印诊断】
            # 检查是否出现巨大的数值
            # _logger.info(f"Step: l_val={l_val}, max(l_xx)={jnp.max(jnp.abs(l_xx_val))}, max(l_uu)={jnp.max(jnp.abs(l_uu_val))}")
            # HOP-LQR augmented problem must be built from one consistent
            # quadratic model. Do not use the DDP backward regularization here
            # unless R is regularized in exactly the same way.
            l_uu_reg = l_uu_val
            def solve_chol():
                L = jnp.linalg.cholesky(l_uu_reg)
                return jnp.linalg.solve(L.T, jnp.linalg.solve(L, jnp.eye(m)))
            l_uu_inv = solve_chol()
            # 中间项计算 (利用 @ 进行批量矩阵乘法，vmap 会自动处理 batch 维)            
            A_tilde = A - B @ l_uu_inv @ l_ux_val
            a_tilde = a - B @ l_uu_inv @ l_u_val
            Q_tilde = l_xx_val - l_ux_val.T @ l_uu_inv @ l_ux_val
            q_tilde = l_x_val - l_ux_val.T @ l_uu_inv @ l_u_val
            Q_tilde = Q_tilde + 1e-9 * jnp.eye(n, dtype=Q_tilde.dtype)
            constant_term = (
                2.0 * (l_val + self.w - 0.5 * l_u_val.T @ l_uu_inv @ l_u_val)
                + 1e-12
            )
            
            # 构建增广矩阵            
            # A_aug
            # [ A_tilde, a_tilde ]
            # [ 0,       1       ]
            row1 = jnp.hstack([A_tilde, a_tilde.reshape(-1, 1)])
            row2 = jnp.hstack([zeros_1n, jnp.ones((1, 1))])
            A_aug_k = jnp.vstack([row1, row2])
            
            # B_aug
            # [ B ]
            # [ 0 ]
            B_aug_k = jnp.vstack([B, jnp.zeros((1, m))])
            
            # Q_aug
            # [ Q_tilde, q_tilde ]
            # [ q_tilde.T, const ]
            row1_q = jnp.hstack([Q_tilde, q_tilde.reshape(-1, 1)])
            row2_q = jnp.hstack([q_tilde.reshape(1, -1), jnp.array([[constant_term]])])
            Q_aug_k = jnp.vstack([row1_q, row2_q])
            
            return A_aug_k, B_aug_k, Q_aug_k, l_uu_val, a
        # 3. 执行 vmap
        # in_axes: 对 x_curr, u_k, x_next_k 的第 0 维进行映射
        vmap_compute = vmap(compute_step_k, in_axes=(0, 0, 0))
        A_aug, B_aug, Q_aug, R, a_k = vmap_compute(x_curr, u_traj, x_next)
        return A_aug, B_aug, Q_aug, R, a_k
    
    def _compute_terminal_augmented_cost(self, x_N: jnp.ndarray) -> jnp.ndarray:
        """
        计算增广终端成本矩阵 - 对应论文公式(32)
        参数:
            x_N: 终端状态
        
        返回:
            Q_T_aug: 增广终端成本矩阵 (n+1, n+1)
        """
        n = self.n
        
        # 计算终端成本的导数
        phi_val = self.phi(x_N)
        phi_x_val = self.phi_x(x_N)
        phi_xx_val = self.phi_xx(x_N)
        
        # 构建增广终端成本矩阵
        Q_T_aug = jnp.zeros((n+1, n+1))
        Q_T_aug = Q_T_aug.at[:n, :n].set(phi_xx_val)
        Q_T_aug = Q_T_aug.at[:n, n].set(phi_x_val)
        Q_T_aug = Q_T_aug.at[n, :n].set(phi_x_val)
        Q_T_aug = Q_T_aug.at[n, n].set(2 * phi_val)

        # 参考实现仅在 Q_T_aug[-1,-1] 加 1e-12 防零特征值。
        # 不使用大 eps 的 make_pd，避免改变 HOP-LQR 看到的终端 surrogate。
        Q_T_aug = Q_T_aug.at[n, n].add(1e-12)
        Q_T_aug = 0.5 * (Q_T_aug + Q_T_aug.T)
        return Q_T_aug

    @partial(jit, static_argnums=(0,))
    def _ddp_backward_pass(self, x_traj: jnp.ndarray, u_traj: jnp.ndarray, 
                       T_star: int) -> Tuple:
        """
        DDP 反向传递 (JIT 编译版，支持动态 T_star)。
        策略：
        1. 不改变数组形状，始终对全长 (T_max) 进行 scan。
        2. 在 scan_step 内部，通过 lax.cond(k < T_star) 判断是否执行计算。
        3. 若 k >= T_star，直接透传 Carry (V_x, V_xx) 并返回零增益。
        """
        n, m = self.n, self.m        
        # 1. 准备全尺寸输入数据
        # x_traj: (T_max+1, n), u_traj: (T_max, m)
        x_curr = x_traj[:-1]  # (T_max, n)
        x_next = x_traj[1:]   # (T_max, n)
        u_curr = u_traj       # (T_max, m)
        
        # 2. 生成时间索引序列 [0, 1, ..., T_max-1]
        # 这将作为 scan 的输入之一，以便在每一步知道当前的 k
        indices = jnp.arange(self.T_max)
                
        # 3. 初始化终端条件 (在 T_star 时刻)
        # 使用 jnp.take 动态获取 T_star 时刻的状态
        x_T_star = jnp.take(x_traj, T_star, axis=0)
        value_dtype = x_traj.dtype
        V_x_term = self.phi_x(x_T_star).astype(value_dtype)
        V_xx_term = self.phi_xx(x_T_star).astype(value_dtype)
        V_xx_term = 0.5 * (V_xx_term + V_xx_term.T)

        
        def scan_step(carry, inputs_step):
            V_x_next, V_xx_next = carry
            x_k, x_next_k, u_k, k = inputs_step
            # 核心逻辑：判断当前步 k 是否在有效时域内 [0, T_star-1]
            # 注意：scan 是 reverse=True，所以 k 从 T_max-1 递减到 0
            is_valid = k < T_star
            # --- 分支 A: 无效步骤 (k >= T_star) ---
            # 策略：不更新值函数，直接透传 Carry，增益设为 0
            def invalid_step():
                # Carry 不变
                # 输出增益 (K, k_vec, V_x, V_xx) 设为 0 或占位符
                # 注意：输出形状必须与 valid_step 一致
                dtype = V_x_next.dtype
                zeros_K = jnp.zeros((m, n), dtype=dtype)
                zeros_k = jnp.zeros(m, dtype=dtype)
                zeros_Vx = jnp.zeros(n, dtype=dtype)
                zeros_Vxx = jnp.zeros((n, n), dtype=dtype)
                return carry, (zeros_K, zeros_k, zeros_Vx, zeros_Vxx)
            
            # --- 分支 B: 有效步骤 (k < T_star) ---
            # 策略：执行标准的 DDP 反向递推公式
            def valid_step():
                # 1. 动力学一阶导数
                A = self.f_x(x_k, u_k)
                B = self.f_u(x_k, u_k)
                
                # 2. 动力学二阶导数 (保持原有逻辑)
                def f_x_only(x, u): return self.f(x, u)
                f_xx = jacrev(jacrev(f_x_only))(x_k, u_k)
                
                def f_combined(z):
                    return self.f(z[:n], z[n:])
                z_k = jnp.concatenate([x_k, u_k])
                H_full = jacrev(jacrev(f_combined))(z_k)
                f_ux = H_full[:, n:n+m, :n]
                f_uu = H_full[:, n:n+m, n:n+m]
                
                # 3. 动力学残差
                f_val = self.f(x_k, u_k)
                a_k = f_val - x_next_k
                
                # 4. Q 函数梯度 (一阶)
                Q_x = self.l_x(x_k, u_k) + A.T @ V_x_next
                Q_u = self.l_u(x_k, u_k) + B.T @ V_x_next
                # DDP 修正项(选用DDP时可用): V_xx @ a_k
                # V_xx_a = V_xx_next @ a_k
                # Q_x = Q_x + A.T @ V_xx_a
                # Q_u = Q_u + B.T @ V_xx_a 
                # 5. Q 函数 Hessian (二阶)
                Q_xx = self.l_xx(x_k, u_k) + A.T @ V_xx_next @ A
                Q_ux = self.l_ux(x_k, u_k) + B.T @ V_xx_next @ A
                Q_uu = self.l_uu(x_k, u_k) + B.T @ V_xx_next @ B
                # DDP 修正项(选用DDP时可用):
                # Q_xx += jnp.einsum('i,ijk->jk', V_x_next, f_xx)
                # Q_uu += jnp.einsum('i,ijk->jk', V_x_next, f_uu)
                # Q_ux += jnp.einsum('i,ijk->jk', V_x_next, f_ux)
                # 对称化
                Q_xx = 0.5 * (Q_xx + Q_xx.T)
                Q_uu = 0.5 * (Q_uu + Q_uu.T)
                
                # 6. 正则化与求解增益
                Q_uu_reg = Q_uu + self.reg * jnp.eye(m)
                L = jnp.linalg.cholesky(Q_uu_reg)
                inv_Q_uu = jnp.linalg.solve(L.T, jnp.linalg.solve(L, jnp.eye(m)))
                k_vec_k = -inv_Q_uu @ Q_u
                K_k = -inv_Q_uu @ Q_ux
                
                # 7. 更新值函数 (Bellman 方程)
                # Use the standard iLQR value update. K/k are computed with the
                # regularized Quu for the local step, but the Bellman model itself
                # uses the original Quu, matching the reference implementation.
                V_x_k = Q_x + K_k.T @ Q_u + Q_ux.T @ k_vec_k + K_k.T @ Q_uu @ k_vec_k
                V_xx_k = Q_xx + K_k.T @ Q_ux + Q_ux.T @ K_k + K_k.T @ Q_uu @ K_k
                V_xx_k = 0.5 * (V_xx_k + V_xx_k.T)
                dtype = V_x_next.dtype
                K_k = K_k.astype(dtype)
                k_vec_k = k_vec_k.astype(dtype)
                V_x_k = V_x_k.astype(dtype)
                V_xx_k = V_xx_k.astype(dtype)
                
                return (V_x_k, V_xx_k), (K_k, k_vec_k, V_x_k, V_xx_k)

            # 根据 is_valid 选择执行哪个分支
            new_carry, outputs = lax.cond(is_valid, valid_step, invalid_step)
            return new_carry, outputs

        # 执行反向 Scan
        # reverse=True: 索引顺序为 T_max-1, T_max-2, ..., 0
        # 当 k >= T_star 时，invalid_step 被调用，Carry 保持不变 (仍为终端值)
        # 当 k == T_star-1 时，valid_step 被调用，使用终端值计算 T_star-1 的值函数
        # 初始 Carry: (V_x(T_star), V_xx(T_star))
        carry_init = (V_x_term, V_xx_term)
        # 打包输入：(x_k, x_next_k, u_k, k)
        inputs = (x_curr, x_next, u_curr, indices)
        _, (K_seq, k_seq, V_x_seq, V_xx_seq) = scan(
            scan_step, carry_init, inputs, reverse=True
        )
        return K_seq, k_seq, V_x_seq, V_xx_seq

    @partial(jit, static_argnums=(0,))
    def _forward_pass(self, x0, x_traj_ref, u_traj_ref, K, k_vec, alpha, T_star):
        """
        前向推演 (JIT 编译版，支持动态 T_star)。
        策略：接收全尺寸数组 (T_max)，在 scan 内部通过掩码逻辑只更新前 T_star 步。
        参数:
            x_traj_ref: 参考状态轨迹 (T_max+1, n)
            u_traj_ref: 参考控制轨迹 (T_max, m)
            T_star: 动态时域 (int scalar)
        """
        T_max = u_traj_ref.shape[0]
        # 生成全尺寸索引 [0, ..., T_max-1]
        indices = jnp.arange(T_max)
        def step_fn(carry, k):
            x_k, total_cost = carry
            # 判断当前步是否有效
            is_valid = k < T_star
            def do_valid_step():
                # 计算控制更新
                # 注意：这里直接使用全尺寸数组的索引 k，因为 k < T_max
                delta_x = self._state_difference(x_k, x_traj_ref[k])
                delta_u = K[k] @ delta_x + alpha * k_vec[k]
                u_k = u_traj_ref[k] + delta_u
                # u_k = jnp.clip(u_k, self.u_min, self.u_max)
                # 动力学推进
                x_next = self.f(x_k, u_k)
                # 累加成本
                cost = self.l(x_k, u_k)
                return (x_next, total_cost + cost), (x_next, u_k)
            
            def do_invalid_step():
                # 无效步：用参考控制继续模拟动力学，保持状态连续推进。
                # 不冻结状态，避免尾部在未优化的截断点产生离目标很远
                # 的僵死状态，进而污染下一轮 HOP-LQR 增强 Q 矩阵的条件数。
                u_k = u_traj_ref[k]
                x_next = self.f(x_k, u_k)
                return (x_next, total_cost), (x_next, u_k)
            
            next_carry, outputs = lax.cond(is_valid, do_valid_step, do_invalid_step)
            return next_carry, outputs

        init_carry = (x0, 0.0)
        
        # 执行全尺寸 Scan
        (x_final_raw, running_cost), (x_traj_body, u_traj_body) = scan(
            step_fn, init_carry, indices
        )
        # --- 后处理：提取真实终端状态和成本 ---
        x_traj_full = jnp.concatenate([x0[None, :], x_traj_body]) # (T_max+1, n)
        # 获取 T_star 时刻的状态
        x_T_star = jnp.take(x_traj_full, T_star, axis=0)
        # 计算终端成本
        terminal_cost = self.phi(x_T_star)
        # 总成本 = 阶段成本和 + 终端成本 + 时域惩罚
        # 注意：running_cost 已经是 sum(l(x_k, u_k)) for k in 0..T_star-1
        total_cost = running_cost + terminal_cost + self.w * T_star
        
        return x_traj_full, u_traj_body, total_cost

    @partial(jit, static_argnums=(0,))
    def _linesearch(self, x0: jnp.ndarray, x_traj: jnp.ndarray, u_traj: jnp.ndarray,
                    K: jnp.ndarray, k_vec: jnp.ndarray, T_star: int,
                    current_cost: float) -> tuple:
        """
        并行线搜索 (JIT 编译版)。
        注意：输入必须是全尺寸数组 (T_max)。
        """
        steps_arr = jnp.array([1.0 * 0.7**i for i in range(15)])
        def loss_func(alpha):
            x_new, u_new, cost_new = self._forward_pass(
                x0, x_traj, u_traj, K, k_vec, alpha, T_star
            )
            return cost_new
        # 并行计算所有步长的损失
        # vmap 会自动将 steps_arr 映射到 loss_func
        loss_arr = vmap(loss_func)(steps_arr)
        loss_arr_safe = jnp.where(jnp.isnan(loss_arr), jnp.inf, loss_arr)
        min_loss_idx = jnp.argmin(loss_arr_safe)
        all_inf = jnp.all(jnp.isinf(loss_arr_safe))
        best_alpha = jnp.where(all_inf, 0.0, steps_arr[min_loss_idx])
        best_cost = jnp.where(all_inf, current_cost, loss_arr_safe[min_loss_idx])
        best_x_new, best_u_new, _ = self._forward_pass(
            x0, x_traj, u_traj, K, k_vec, best_alpha, T_star
        )
        best_x_new = jnp.where(all_inf, x_traj, best_x_new)
        best_u_new = jnp.where(all_inf, u_traj, best_u_new)
        return best_x_new, best_u_new, best_cost
    
    @partial(jit, static_argnums=(0,))
    def _compute_trajectory_cost(self, x_traj: jnp.ndarray, u_traj: jnp.ndarray, T: int) -> float:
        """
        计算给定轨迹的总成本 (支持动态 T)
        """
        # 创建一个掩码，标记出 0 到 T-1 的位置
        # indices: [0, 1, ..., T_max-1]
        indices = jnp.arange(self.T_max)
        # mask: True if index < T else False
        mask = indices < T
        # 计算所有步长的成本
        all_stage_costs = jax.vmap(self.l)(x_traj[:-1], u_traj) # (T_max,)
        # 只累加有效部分
        stage_cost_sum = jnp.sum(all_stage_costs * mask)
        # 终端状态：取索引为 T 的状态
        # 使用 jnp.take 支持动态索引
        x_T = jnp.take(x_traj, T, axis=0)
        terminal_cost = self.phi(x_T)
        horizon_penalty = self.w * T
        
        return stage_cost_sum + terminal_cost + horizon_penalty

    @partial(jit, static_argnums=(0,))
    def traj_sim(self, x0: jnp.ndarray, u_traj: jnp.ndarray) -> jnp.ndarray:
        """
        [优化版] 生成初始轨迹 - 通过仿真动力学模型
        使用 jax.lax.scan 替代 Python 循环，支持 JIT 编译。
        
        参数:
            x0: 初始状态 (n,)
            u_traj: 控制轨迹 (T, m) - 必须长度匹配 T
            T: 仿真时域长度 (静态参数)
        
        返回:
            x_traj: 状态轨迹 (T+1, n)
            u_traj_eff: 有效的控制轨迹 (T, m)
        """
        # 1. 定义单步动力学函数 (Carry function)
        def step_fn(carry_x, u_k):
            next_x = self.f(carry_x, u_k)
            return next_x, next_x

        # 2. 执行 scan
        # initial: x0
        # xs: u_traj (形状必须是 (T, m))
        # 返回值: (final_state, all_states)
        # all_states 的形状将是 (T, n)，包含了 x_1 到 x_T
        _, x_traj_body = scan(step_fn, x0, u_traj)

        # 3. 拼接初始状态 x0
        # x_traj_body 是 [x_1, ..., x_T]
        # 我们需要 [x_0, x_1, ..., x_T]
        x_traj = jnp.vstack([x0[jnp.newaxis, :], x_traj_body])
        return x_traj
    
    def solve(self, x0: jnp.ndarray, u_traj: jnp.ndarray, 
              T_min: int = 1, T_max: Optional[int] = None,
              max_iter: Optional[int] = None, tol: Optional[float] = None) -> DDPResult:
        """
        参数:
            x0: 初始状态
            u_traj: 初始控制轨迹
            T_min: 最小允许时域
            T_max: 最大允许时域
            max_iter: 最大迭代次数
            tol: 收敛容差
        返回:
            DDPResult: 求解结果
        """
        # 设置参数
        if T_max is None:
            self.T_max = len(u_traj)
            T_max = self.T_max
        else:
            self.T_max = T_max
        self.pre_T_star = T_max

        if tol is None:
            tol = self.cost_tol
        n, m, w = self.n, self.m, self.w
        # 生成初始轨迹
        x_traj = self.traj_sim(x0, u_traj)
        # 初始化最优轨迹
        opt_x_traj = x_traj
        opt_u_traj = u_traj
        # 初始化
        iteration = 0
        cost_decrease = float('inf')
        # 存储迭代历史
        iteration_costs = []
        iteration_T_stars = []
        
        # 主迭代循环
        while iteration <= max_iter:
            iteration_start = time.time()
    
            # === 步骤1: 线性化和增广 ===
            # _logger.info(f"迭代 {iteration+1}: 开始计算增广系统")
            A_aug, B_aug, Q_aug, R, _ = self._compute_augmented_system(x_traj, u_traj)
            delta_x0 = self._state_difference(x0, x_traj[0])
            z0 = jnp.concatenate([delta_x0, jnp.array([1.0])])
            if iteration == 0:
                _logger.info(
                    "[diag] HOP-DDP local initial deviation:"
                    f" ||delta_x0||={float(jnp.linalg.norm(delta_x0)):.3e},"
                    f" z0_tail={float(z0[-1]):.1f}"
                )
            # _logger.info(f"迭代 {iteration+1}: z0 - min: {jnp.min(z0):.2e}, max: {jnp.max(z0):.2e}, NaN: {jnp.any(jnp.isnan(z0))}, Inf: {jnp.any(jnp.isinf(z0))}")
            
            x_N = x_traj[-1]
            Q_T_aug = vmap(self._compute_terminal_augmented_cost)(x_traj[1 : T_max + 1])
            if iteration == 0:
                _logger.info(
                    "[diag] terminal surrogate: per-horizon Q_T_aug(x_t), "
                    f"shape={tuple(Q_T_aug.shape)}"
                )
            
            use_warm_start_horizon = iteration < self.warm_start_iters
            if use_warm_start_horizon:
                horizon_source = "warm-start"
                T_star = T_max
                _logger.info(
                    f"[warm-start] iter={iteration+1}: "
                    f"force T*=T_max={T_max} "
                    f"({iteration+1}/{self.warm_start_iters})"
                )
            else:
                horizon_source = "hop-lqr"
                # _logger.info(f"迭代 {iteration+1}: 调用HOP-LQR求解...")
                # 使用 HOP-LQR 选择最优时域
                T_star = self.hop_lqr.solve(
                    z0, A_aug, B_aug, Q_aug, R, Q_T_aug, w, T_min, T_max
                )
            # === 步骤3: 截断的反向传播 ===
            # _logger.info(f"迭代 {iteration+1}: 开始反向传播，T_star = {T_star}")
            K, k_vec, _, _ = self._ddp_backward_pass(
                x_traj, u_traj, T_star
            )
            # === 步骤4: 前向推演和线搜索 ===
            current_cost = self._compute_trajectory_cost(
                x_traj, u_traj, T_star
            )
            # _logger.info(f"迭代 {iteration+1}: 当前成本 = {current_cost:.6f}, NaN: {jnp.isnan(current_cost)}, Inf: {jnp.isinf(current_cost)}")
            # 使用并行线搜索
            # 传全尺寸，避免因 T_star 变化触发 JIT 重新编译
            best_x_new, best_u_new, best_cost = self._linesearch(
                x0, x_traj, u_traj, K, k_vec, T_star, current_cost
            )
            # _logger.info(f"迭代 {iteration+1}: 线搜索完成 - 最优成本 = {best_cost:.6f}, NaN: {jnp.isnan(best_cost)}, Inf: {jnp.isinf(best_cost)}")
            # 检查是否接受新轨迹
            cost_decrease = current_cost - best_cost
            accepted_step = (cost_decrease > 0)
            if cost_decrease > 0:
                # 接受新轨迹
                # best_x_new / best_u_new 已经是 _forward_pass 算好的全长结果，
                # 其中 T_star 内为优化步，T_star 外使用参考控制继续 rollout。
                x_traj = best_x_new
                u_traj = best_u_new
                current_cost = best_cost
                iteration_costs.append(current_cost)
                iteration_T_stars.append(T_star)
                self.pre_T_star = T_star
                opt_x_traj = x_traj
                opt_u_traj = u_traj


                if iteration >= 2:
                    cost_change = abs(iteration_costs[-1] - iteration_costs[-2])
                    cost_converged = cost_change < tol
                    if cost_converged:
                        _logger.info(f"HOP-DDP 收敛: 成本变化 {cost_change:.6f} < {tol}")
                        break
            else:
                _logger.info("reject new traj.")
                x_traj = opt_x_traj
                u_traj = opt_u_traj

            # 记录迭代信息
            iteration_time = time.time() - iteration_start
            step_status = "accepted" if accepted_step else "rejected"
            _logger.info(f"Iteration {iteration+1}: T*={T_star}, Cost={current_cost:.6f}, "
                  f"ΔCost={cost_decrease:.6f}, Reg={self.reg:.2e}, Time={iteration_time:.3f}s, "
                  f"Step={step_status}, HorizonSource={horizon_source}")
            if self.iteration_callback is not None:
                try:
                    self.iteration_callback(
                        iteration + 1,
                        x_traj,
                        u_traj,
                        T_star,
                        float(current_cost),
                        accepted_step,
                    )
                except Exception as exc:
                    _logger.info(f"[warn] Iteration callback failed at iter={iteration+1}: {exc}")
            
            iteration += 1
        
        # iLQR风格的收敛诊断
        final_T = iteration_T_stars[-1] if iteration_T_stars else T_max
        control_cost = iteration_costs[-1] - w * final_T
        result = DDPResult(
            u_opt=opt_u_traj[:final_T],
            x_opt=opt_x_traj[:final_T+1],
            T_star=final_T,
            total_cost=iteration_costs[-1],
            control_cost=control_cost,
            iteration_costs=iteration_costs,
            iteration_T_stars=iteration_T_stars
        )
        
        _logger.info("\n" + "="*60)
        _logger.info("HOP-DDP 求解完成")
        _logger.info(f"总迭代次数: {iteration}")
        _logger.info(f"最终时域: T* = {final_T}")
        _logger.info(f"总成本: {iteration_costs[-1]:.6f} (二次成本: {control_cost:.6f} + 时域惩罚: {w*final_T:.6f})")
        _logger.info(f"收敛: {abs(cost_decrease) < tol and cost_decrease >= 0}")
        _logger.info("="*60)
        return result
