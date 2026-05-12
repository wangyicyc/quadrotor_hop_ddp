#!/usr/bin/env python3
"""
Cart-pole HOP-DDP test script.
参考 `unicycle_hop_ddp_test.py` 的结构。
"""
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT))

from dataclasses import dataclass
import jax.numpy as jnp

from dynamics.cartpole_dynamics import CartPoleDynamics
from hop_lib.hop_ddp_solver import HOPDDPSolver
from typing import Tuple

@dataclass
class CartPoleConfig:
    """Configuration for cart-pole stabilization problem."""
    dt: float = 0.02
    tsteps: int = 100

    # Cost weights for state [x, x_dot, theta, theta_dot]
    q_x: float = 1.0
    q_x_dot: float = 0.1
    q_theta: float = 20.0
    q_theta_dot: float = 1.0

    # Control cost weight for force
    r_u: float = 0.1

    # Dynamics parameters
    m_c: float = 1.0
    m_p: float = 0.1
    l: float = 0.5
    g: float = 9.81
    u_limit: Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([-10.0]), jnp.array([10.0]))
    def x0(self) -> jnp.ndarray:
        # Slightly perturbed from upright
        return jnp.array([-1.0, 0.2, 0.2, 0.01])

    def x_target(self) -> jnp.ndarray:
        # Upright at origin
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    # Optional control limit (not enforced by solver unless you clamp it)
    u_max: float = 10.0


def get_cost_matrices(cfg: CartPoleConfig):
    Q = jnp.diag(jnp.array([cfg.q_x, cfg.q_x_dot, cfg.q_theta, cfg.q_theta_dot]))
    R = jnp.diag(jnp.array([cfg.r_u]))
    return Q, R


def stage_cost(x: jnp.ndarray, u: jnp.ndarray, x_ref: jnp.ndarray,
               Q: jnp.ndarray, R: jnp.ndarray) -> float:
    """Running cost: 1/2 * ( (x-x_ref)^T Q (x-x_ref) + u^T R u )."""
    x_err = x - x_ref
    return 0.5 * (x_err @ Q @ x_err + u @ R @ u)


def terminal_cost(x: jnp.ndarray, x_ref: jnp.ndarray, Q_T: jnp.ndarray) -> float:
    """Terminal cost: 1/2 * (x-x_ref)^T Q_T (x-x_ref)."""
    x_err = x - x_ref
    return 0.5 * (x_err @ Q_T @ x_err)


def _generate_initial_control_guess(tsteps: int) -> jnp.ndarray:
    """Simple zero control initial guess."""
    return jnp.zeros((tsteps, 1))


def trajectory_generator(x_current: jnp.ndarray, x_target: jnp.ndarray, tsteps: int, dt: float) -> jnp.ndarray:
    """轨迹生成函数，用于在DDP迭代过程中填充剩余轨迹"""
    # 为cartpole系统生成从当前状态到目标状态的控制轨迹
    # 简单地返回零控制
    return jnp.zeros((tsteps, 1))


def test_cartpole_stabilization_ddp():
    print("\n" + "=" * 60)
    print("Test: Cart-pole stabilization (HOP-DDP)")
    print("=" * 60)

    cfg = CartPoleConfig()
    dynamics = CartPoleDynamics(
        dt=cfg.dt,
        m_c=cfg.m_c,
        m_p=cfg.m_p,
        l=cfg.l,
        g=cfg.g,
    )
    Q, R = get_cost_matrices(cfg)
    Q_T = 10.0 * Q

    x0 = cfg.x0()
    x_target = cfg.x_target()

    print(f"Initial state: {x0}")
    print(f"Target state:  {x_target}")
    print(f"Time horizon: {cfg.tsteps * cfg.dt:.2f}s ({cfg.tsteps} steps @ dt={cfg.dt})")

    def l_func(x, u):
        return stage_cost(x, u, x_target, Q, R)

    def phi_func(x):
        return terminal_cost(x, x_target, Q_T)

    solver = HOPDDPSolver(
        f=dynamics.discrete_dynamics,
        l=l_func,
        phi=phi_func,
        n=dynamics.nx,
        m=dynamics.nu,
        u_limit=cfg.u_limit,
        trajectory_generator=trajectory_generator,
        x_target=x_target,
        config=cfg,
    )

    u_traj_guess = _generate_initial_control_guess(cfg.tsteps)
    print(f"\nInitial guess: zero u (shape: {u_traj_guess.shape})")

    print("\nSolving with DDP...")
    try:
        result = solver.solve(x0, u_traj_guess, max_iter=100, tol=1e-6)
        u_opt = result.u_opt
        x_opt = result.x_opt
        cost = result.total_cost
        
        # 显示结果（按照四旋翼模型格式）
        print(f"\nHOP-DDP求解完成:")
        print(f"最优时域: {result.T_star}")
        print(f"总代价: {cost:.6f}")
        print(f"最优控制轨迹形状: {u_opt.shape}")
        print(f"最优状态轨迹形状: {x_opt.shape}")
        
        # 计算跟踪误差
        final_state = x_opt[-1]
        tracking_error = jnp.linalg.norm(final_state - x_target)
        print(f"\n最终状态误差: {tracking_error:.6f}")
        
        # 显示所有时间步的结果（按照时间步0:state,input的格式）
        print(f"\n所有时间步的状态和控制输入:")
        for i in range(5,-1,-1):
            # 状态向量：[x, x_dot, theta, theta_dot]
            state_str = f"x={x_opt[i][0]:.4f}, x_dot={x_opt[i][1]:.4f}, theta={x_opt[i][2]:.4f}, theta_dot={x_opt[i][3]:.4f}"
            # 控制输入：力
            control_str = f"force={u_opt[i][0]:.4f}"
            print(f"时间步 {i}: state=[{state_str}], input=[{control_str}]")
        # 打印最终状态
        final_state_str = f"x={x_opt[-1][0]:.4f}, x_dot={x_opt[-1][1]:.4f}, theta={x_opt[-1][2]:.4f}, theta_dot={x_opt[-1][3]:.4f}"
        print(f"\nFinal state: [{final_state_str}]")
        # 目标状态
        target_state_str = f"x={x_target[0]:.4f}, x_dot={x_target[1]:.4f}, theta={x_target[2]:.4f}, theta_dot={x_target[3]:.4f}"
        print(f"Target state: [{target_state_str}]")

        return result
    except Exception as e:
        print(f"\nDDP solve failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_dynamics():
    """Simple verification of the dynamics and linearization."""
    print("\n" + "=" * 60)
    print("Verifying CartPoleDynamics")
    print("=" * 60)
    dyn = CartPoleDynamics(dt=0.02)
    x_test = jnp.array([0.0, 0.0, 0.1, 0.0])
    u_test = jnp.array([1.0])

    x_dot = dyn.dynamics(x_test, u_test)
    print(f"Continuous dynamics f(x,u): {x_dot}")

    x_next = dyn.discrete_dynamics(x_test, u_test)
    print(f"Discrete step (Euler): {x_next}")

    A_ana, B_ana = dyn.get_linearized_dynamics(x_test, u_test)
    print(f"\nAutodiff A:\n{A_ana}")
    print(f"Autodiff B:\n{B_ana}")

    A_auto, B_auto = dyn.get_linearized_dynamics_autodiff(x_test, u_test)
    diff_A = jnp.max(jnp.abs(A_ana - A_auto))
    diff_B = jnp.max(jnp.abs(B_ana - B_auto))
    print(f"\nMax difference in A: {diff_A:.2e}")
    print(f"Max difference in B: {diff_B:.2e}")
    if diff_A < 1e-10 and diff_B < 1e-10:
        print("✓ Autodiff linearization match.")
    else:
        print("✗ Mismatch in linearization!")


def main():
    print("CartPole Model Test")
    print("=" * 60)
    try:
        # 1. Verify dynamics implementation
        # verify_dynamics()

        # 2. Test stabilization problem (with solver)
        test_cartpole_stabilization_ddp()

        print("\n" + "=" * 60)
        print("All verification steps completed.")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()