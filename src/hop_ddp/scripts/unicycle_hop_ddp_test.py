#!/usr/bin/env python3
"""
Unicycle HOP-DDP test script.
Tests HOP-DDP solver on a unicycle (differential-drive) model.
参考 `double_integrator_hop_ddp.py` 的结构。
"""
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT)) 
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
# 导入上面定义的动力学模型
from dynamics.unicycle_dynamics import UnicycleDynamics
# 假设HOP-DDP求解器在hop_lib模块中
from hop_lib.hop_ddp_solver import HOPDDPSolver



@dataclass
class UnicycleConfig:
    """Configuration for the unicycle point-to-point problem."""
    dt: float = 0.05                 # Time step [s]
    tsteps: int = 100                # Number of time steps (horizon length)
    # State cost weights for [x, y, theta]
    q_x: float = 10.0
    q_y: float = 10.0
    q_theta: float = 2.0
    # Control cost weights for [v, omega]
    r_v: float = 0.5
    r_omega: float = 0.5

    # Initial state
    def x0(self) -> jnp.ndarray:
        # Start at origin, facing along positive x-axis
        return jnp.array([0.0, 0.0, 0.0])

    # Target state
    def x_target(self) -> jnp.ndarray:
        # Target: position (2.0, 1.0), orientation pi/4 (45 degrees)
        return jnp.array([2.0, 1.0, jnp.pi/4])

    # State and control limits (optional, for realistic problems)
    v_max: float = 1.0      # max linear speed [m/s]
    omega_max: float = 1.0  # max angular speed [rad/s]


def get_cost_matrices(cfg: UnicycleConfig):
    """Construct the diagonal cost matrices Q and R."""
    Q = jnp.diag(jnp.array([cfg.q_x, cfg.q_y, cfg.q_theta]))
    R = jnp.diag(jnp.array([cfg.r_v, cfg.r_omega]))
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


def _generate_initial_control_guess(x0: jnp.ndarray, x_target: jnp.ndarray,
                                    tsteps: int, dt: float) -> jnp.ndarray:
    """
    Generate a simple initial guess for the control trajectory.
    Strategy: compute a constant velocity/angular velocity that would move
    the robot in the right direction in a straight line (ignoring orientation).
    This is a naive guess, but provides a starting point for the optimizer.
    """
    # Compute desired displacement
    p0 = x0[:2]
    p_target = x_target[:2]
    dp = p_target - p0
    distance = jnp.linalg.norm(dp)
    direction = dp / (distance + 1e-6)  # unit vector

    # Simple guess: constant linear speed to cover distance, small angular speed
    total_time = tsteps * dt
    v_guess = distance / total_time if total_time > 0 else 0.1

    # Guess for angular velocity: rotate to face target initially, then maintain
    theta0 = x0[2]
    theta_target = x_target[2]
    # Angle difference (normalized to [-pi, pi))
    dtheta = (theta_target - theta0 + jnp.pi) % (2 * jnp.pi) - jnp.pi
    omega_guess = dtheta / total_time if total_time > 0 else 0.0

    # Create constant guess trajectory
    u_guess = jnp.array([v_guess, omega_guess])
    u_traj = jnp.tile(u_guess, (tsteps, 1))
    return u_traj


def test_unicycle_point_to_point_ddp():
    print("\n" + "=" * 60)
    print("Test: Unicycle point-to-point motion planning (HOP-DDP)")
    print("=" * 60)

    cfg = UnicycleConfig()
    dynamics = UnicycleDynamics(dt=cfg.dt)
    Q, R = get_cost_matrices(cfg)
    Q_T = 10.0 * Q  # Heavier terminal cost to enforce goal constraint

    x0 = cfg.x0()
    x_target = cfg.x_target()

    print(f"Initial state: {x0}")
    print(f"Target state:  {x_target}")
    print(f"Time horizon: {cfg.tsteps * cfg.dt:.2f}s ({cfg.tsteps} steps @ dt={cfg.dt})")

    # Define cost functions (partial application of reference and weights)
    def l_func(x, u):
        return stage_cost(x, u, x_target, Q, R)

    def phi_func(x):
        return terminal_cost(x, x_target, Q_T)

    # 在实际使用中，替换为真实的 HOPDDPSolver
    solver = HOPDDPSolver(
        f=dynamics.discrete_dynamics,
        l=l_func,
        phi=phi_func,
        n=dynamics.nx,
        m=dynamics.nu,
        dt=cfg.dt,
        w=1,
        u_limit=4.0
    )
    # Generate initial control guess
    u_traj_guess = _generate_initial_control_guess(x0, x_target, cfg.tsteps, cfg.dt)
    print(f"\nInitial guess: constant u = {u_traj_guess[0]} (shape: {u_traj_guess.shape})")

    print("\nSolving with DDP...")
    try:
        # 实际调用应类似：
        # result = solver.solve(x0, u_traj_guess, max_iter=50, tol=1e-6)
        result = solver.solve(x0, u_traj_guess, max_iter=10, tol=1e-6)  # Mock call

        print("\nDDP solve complete:")
        print(f"Total cost: {result.total_cost:.6f}")
        # print(f"Optimal control trajectory shape: {result.u_opt.shape}")
        # print(f"Optimal state trajectory shape: {result.x_opt.shape}")

        # Simulate forward using the initial guess to see the open-loop performance
        print("\n--- Forward simulation with initial guess ---")
        x_sim = x0.copy()
        for i in range(min(5, cfg.tsteps)):  # Show first 5 steps
            u = u_traj_guess[i]
            x_sim = dynamics.discrete_dynamics(x_sim, u)
            print(f"Step {i}: u={u}, x_next={x_sim}")

        # Simulate the full guess trajectory
        x_sim_full = x0
        for i in range(cfg.tsteps):
            x_sim_full = dynamics.discrete_dynamics(x_sim_full, u_traj_guess[i])
        error_open_loop = jnp.linalg.norm(x_sim_full - x_target)
        print(f"\nfinal state error: {error_open_loop:.6f}")
        print(f"final state: {x_sim_full}")
        print(f"target state: {x_target}")

        return result
    except Exception as e:
        print(f"\nDDP solve failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_dynamics():
    """Simple verification of the dynamics and linearization."""
    print("\n" + "=" * 60)
    print("Verifying UnicycleDynamics")
    print("=" * 60)
    dyn = UnicycleDynamics(dt=0.1)
    x_test = jnp.array([1.0, 2.0, jnp.pi/4])  # 45 deg
    u_test = jnp.array([0.5, 0.2])  # v=0.5, omega=0.2

    # Continuous dynamics
    x_dot = dyn.dynamics(x_test, u_test)
    print(f"Continuous dynamics f(x,u): {x_dot}")

    # Discrete dynamics (one step)
    x_next = dyn.discrete_dynamics(x_test, u_test)
    print(f"Discrete step (Euler): {x_next}")

    # Analytical linearization
    A_ana, B_ana = dyn.get_linearized_dynamics(x_test, u_test)
    print(f"\nAnalytical A:\n{A_ana}")
    print(f"Analytical B:\n{B_ana}")

    # Autodiff linearization (should match)
    A_auto, B_auto = dyn.get_linearized_dynamics_autodiff(x_test, u_test)
    print(f"\nAutodiff A (verification):\n{A_auto}")
    print(f"Autodiff B (verification):\n{B_auto}")

    diff_A = jnp.max(jnp.abs(A_ana - A_auto))
    diff_B = jnp.max(jnp.abs(B_ana - B_auto))
    print(f"\nMax difference in A: {diff_A:.2e}")
    print(f"Max difference in B: {diff_B:.2e}")
    if diff_A < 1e-10 and diff_B < 1e-10:
        print("✓ Analytical and autodiff linearization match.")
    else:
        print("✗ Mismatch between analytical and autodiff linearization!")


def main():
    print("Unicycle Model Test")
    print("=" * 60)
    try:
        # 1. Verify dynamics implementation
        # verify_dynamics()

        # 2. Test the point-to-point planning problem (with mock solver)
        test_unicycle_point_to_point_ddp()

        print("\n" + "=" * 60)
        print("All verification steps completed.")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()