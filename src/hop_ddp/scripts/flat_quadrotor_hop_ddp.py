#!/usr/bin/env python3
"""
四旋翼无人机平坦输出 HOP-DDP 测试脚本
演示如何利用平坦性简化 DDP 求解
"""
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT)) 

import jax.numpy as jnp
from hop_lib.hop_ddp_solver import HOPDDPSolver # 假设你的库路径不变
from hop_lib.plot_utils import visualize_quadrotor_results, plot_tracking_errors

# 导入新编写的平坦模型
from dynamics.flat_quadrotor_dynamics import FlatQuadrotorDynamics, FlatQuadrotorParams
from config.flat_quadrotor_config import (
    QuadrotorFlatConfig,
    get_flat_cost_matrices,
    get_flat_terminal_cost_matrix,
)

def stage_cost_flat(x_flat: jnp.ndarray, u_flat: jnp.ndarray, 
                    x_target_flat: jnp.ndarray, Q_flat: jnp.ndarray,
                    R_flat: jnp.ndarray) -> float:
    """
    阶段代价函数：平坦空间中的标准二次型
    """
    dx = x_flat - x_target_flat
    cost_state = 0.5 * dx @ Q_flat @ dx
    cost_control = 0.5 * u_flat @ R_flat @ u_flat
    return cost_state + cost_control

def terminal_cost_flat(x_flat: jnp.ndarray, x_target_flat: jnp.ndarray, 
                       Q_T_flat: jnp.ndarray) -> float:
    """终端代价：平坦空间中的标准二次型"""
    dx = x_flat - x_target_flat
    return 0.5 * dx @ Q_T_flat @ dx


def map_flat_traj_to_physical(
    dynamics: FlatQuadrotorDynamics,
    x_traj_flat: jnp.ndarray,
    u_traj_flat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    将 flat 轨迹批量映射为物理状态轨迹和电机推力轨迹，便于复用原四旋翼可视化。
    """
    x_phys_list = []
    u_phys_list = []

    horizon = u_traj_flat.shape[0]
    for k in range(horizon):
        x_phys_k, u_phys_k = dynamics.flat_to_physical(x_traj_flat[k], u_traj_flat[k])
        x_phys_list.append(x_phys_k)
        u_phys_list.append(u_phys_k)

    if x_traj_flat.shape[0] > 0:
        u_tail = jnp.zeros(dynamics.nu) if horizon == 0 else u_traj_flat[-1]
        x_phys_terminal, _ = dynamics.flat_to_physical(x_traj_flat[-1], u_tail)
        x_phys_list.append(x_phys_terminal)

    x_phys = jnp.asarray(x_phys_list)
    u_phys = jnp.asarray(u_phys_list)
    return x_phys, u_phys

def test_flat_ddp():
    print("="*60)
    print("测试：基于平坦输出的四旋翼 HOP-DDP (状态重标定)")
    print("="*60)

    # 1. 配置
    config = QuadrotorFlatConfig()
    params = FlatQuadrotorParams(
        mass=config.mass, g=config.g,
        Ixx=config.Ixx, Iyy=config.Iyy, Izz=config.Izz,
        arm_length=config.arm_length, c=config.c
    )

    dynamics = FlatQuadrotorDynamics(params, config.dt)
    Q_flat, R_flat = get_flat_cost_matrices(config)
    Q_T_flat = get_flat_terminal_cost_matrix(config)
    iter_plot_dir = REPO_ROOT / "iteration_plots_flat"
    iter_plot_dir.mkdir(parents=True, exist_ok=True)

    # --- 状态重标定：x̃ = D @ x, 使 Q̃ = I ---
    q_diag = jnp.diag(Q_flat)
    q_diag_sqrt = jnp.sqrt(jnp.maximum(q_diag, 1e-12))
    D = jnp.diag(q_diag_sqrt)
    D_inv = jnp.diag(1.0 / q_diag_sqrt)

    # 验证条件数改善
    Q_scaled = D_inv.T @ Q_flat @ D_inv
    Q_T_scaled = D_inv.T @ Q_T_flat @ D_inv
    q_eig = jnp.linalg.eigvalsh(Q_flat)
    qs_eig = jnp.linalg.eigvalsh(Q_scaled)
    print(f"\nQ 条件数: {q_eig[-1]/q_eig[0]:.1f} → {qs_eig[-1]/qs_eig[0]:.1f}")
    print(f"Q_T 条件数: {jnp.linalg.eigvalsh(Q_T_flat)[-1]/jnp.linalg.eigvalsh(Q_T_flat)[0]:.1f}"
          f" → {jnp.linalg.eigvalsh(Q_T_scaled)[-1]/jnp.linalg.eigvalsh(Q_T_scaled)[0]:.1f}")

    x0_scaled = D @ config.x0_flat
    x_target_scaled = D @ config.x_target_flat

    # 变换后的动力学: x̃_{k+1} = D @ f(D^{-1} @ x̃_k, u)
    def f_scaled(x_scaled, u):
        x_flat = D_inv @ x_scaled
        x_next_flat = dynamics.discrete_linear_dynamics(x_flat, u)
        return D @ x_next_flat

    # 变换后的代价 (Q_scaled ≈ I)
    def l_scaled(x, u):
        dx = x - x_target_scaled
        return 0.5 * dx @ Q_scaled @ dx + 0.5 * u @ R_flat @ u

    def phi_scaled(x):
        dx = x - x_target_scaled
        return 0.5 * dx @ Q_T_scaled @ dx

    x0_phys, _ = dynamics.flat_to_physical(config.x0_flat, jnp.zeros(dynamics.nu))
    x_target_phys, _ = dynamics.flat_to_physical(config.x_target_flat, jnp.zeros(dynamics.nu))
    config.x0 = x0_phys
    config.x_target = x_target_phys

    def iteration_visualizer(iteration, x_traj_iter, u_traj_iter, T_star_iter, cost_iter, accepted_step):
        # x_traj_iter 在标定空间中，需要反变换回平坦空间
        x_vis_scaled = jnp.asarray(x_traj_iter[:T_star_iter + 1])
        x_vis_flat = x_vis_scaled @ D_inv
        u_vis_flat = jnp.asarray(u_traj_iter[:T_star_iter])
        x_vis_phys, u_vis_phys = map_flat_traj_to_physical(dynamics, x_vis_flat, u_vis_flat)
        step_tag = "accepted" if accepted_step else "rejected"
        save_path = iter_plot_dir / f"iter_{iteration:03d}_T{T_star_iter:03d}_{step_tag}.png"
        visualize_quadrotor_results(
            x_vis_phys,
            u_vis_phys,
            config,
            save_path=str(save_path),
            show_plot=False,
            close_after=True,
        )

    # 3. 创建求解器
    solver = HOPDDPSolver(
        f=f_scaled,
        l=l_scaled,
        phi=phi_scaled,
        n=dynamics.nx,
        m=dynamics.nu,
        dt=config.dt,
        w=config.w,
        u_limit = (
            -jnp.inf * jnp.ones(dynamics.nu),
            jnp.inf * jnp.ones(dynamics.nu),
        ),
        x_target=x_target_scaled,
        iteration_callback=iteration_visualizer,
    )
    # 自适应正则化：Q 已重标定 (cond=1)，但增广系统 Schur 补仍需正则化
    solver.hop_lqr.efg_q_reg = 0.01
    solver.hop_lqr.efg_q_reg_adaptive_scale = 0.15

    # 4. 初始猜测 (标定空间)
    x0 = x0_scaled
    u_init = jnp.zeros(config.nu_flat)
    u_traj = jnp.tile(u_init[None, :], (config.tsteps, 1))

    print("正在求解...")
    try:
        result = solver.solve(x0, u_traj, max_iter=10, tol=1e-5)

        print(f"求解成功! 总代价: {result.total_cost:.4f}")

        # 5. 结果反变换回平坦空间
        x_opt_flat = result.x_opt @ D_inv
        u_opt_flat = result.u_opt
        x_opt_phys, u_opt_phys = map_flat_traj_to_physical(dynamics, x_opt_flat, u_opt_flat)

        xp_end = x_opt_phys[-1]
        print(f"\n最终位置: {xp_end[0:3]}")
        print(f"目标位置: {config.x_target[0:3]}")
        print(f"误差: {jnp.linalg.norm(xp_end[0:3] - config.x_target[0:3]):.4f}")

        print("\n正在生成可视化图表...")
        visualize_quadrotor_results(x_opt_phys, u_opt_phys, config)
        plot_tracking_errors(x_opt_phys, config.x_target, config)

    except Exception as e:
        print(f"求解失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flat_ddp()
