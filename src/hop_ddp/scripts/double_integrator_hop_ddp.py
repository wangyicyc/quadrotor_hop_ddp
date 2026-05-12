#!/usr/bin/env python3
"""
Double integrator HOP-DDP test script.

Tests HOP-DDP solver on a double integrator model.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT))

from dataclasses import dataclass
import jax.numpy as jnp
from hop_lib.hop_ddp_solver import HOPDDPSolver
from dynamics.double_integrator_dynamics import DoubleIntegratorDynamics


@dataclass
class DoubleIntegratorConfig:
    dim: int = 2
    dt: float = 0.05
    tsteps: int = 60
    # Cost weights
    q_pos: float = 10.0
    q_vel: float = 1.0
    r_acc: float = 0.1

    def x0(self) -> jnp.ndarray:
        # State: [p, v] with p,v in R^dim
        p0 = jnp.zeros(self.dim)
        v0 = jnp.zeros(self.dim)
        return jnp.concatenate([p0, v0], axis=0)

    def x_target(self) -> jnp.ndarray:
        pT = jnp.ones(self.dim)
        vT = jnp.zeros(self.dim)
        return jnp.concatenate([pT, vT], axis=0)


def get_cost_matrices(cfg: DoubleIntegratorConfig):
    Q_pos = cfg.q_pos * jnp.eye(cfg.dim)
    Q_vel = cfg.q_vel * jnp.eye(cfg.dim)
    Q = jnp.block(
        [
            [Q_pos, jnp.zeros((cfg.dim, cfg.dim))],
            [jnp.zeros((cfg.dim, cfg.dim)), Q_vel],
        ]
    )
    R = cfg.r_acc * jnp.eye(cfg.dim)
    return Q, R


def stage_cost(x: jnp.ndarray, u: jnp.ndarray, x_ref: jnp.ndarray,
               Q: jnp.ndarray, R: jnp.ndarray) -> float:
    x_err = x - x_ref
    return 0.5 * (x_err @ Q @ x_err + u @ R @ u)


def terminal_cost(x: jnp.ndarray, x_ref: jnp.ndarray, Q_T: jnp.ndarray) -> float:
    x_err = x - x_ref
    return 0.5 * (x_err @ Q_T @ x_err)


def _constant_accel_guess(x0: jnp.ndarray, xT: jnp.ndarray, tsteps: int, dt: float, dim: int):
    p0 = x0[:dim]
    v0 = x0[dim:]
    pT = xT[:dim]
    T = tsteps * dt
    # Simple constant-acceleration guess to move position; not necessarily zero terminal velocity.
    if T <= 0:
        a = jnp.zeros(dim)
    else:
        a = 2.0 * (pT - p0 - v0 * T) / (T * T)
    return jnp.tile(a[None, :], (tsteps, 1))


def test_point_to_point_ddp():
    print("\n" + "=" * 60)
    print("Test 2: double integrator point-to-point (HOP-DDP)")
    print("=" * 60)

    cfg = DoubleIntegratorConfig()
    dynamics = DoubleIntegratorDynamics(dim=cfg.dim, dt=cfg.dt)
    Q, R = get_cost_matrices(cfg)
    Q_T = 10.0 * Q

    x0 = cfg.x0()
    x_target = cfg.x_target()

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
        dt=cfg.dt,
        w=1,
        u_limit=4.0
    )

    u_traj = _constant_accel_guess(x0, x_target, cfg.tsteps, cfg.dt, cfg.dim)

    print("Solving HOP-DDP...")
    try:
        result = solver.solve(x0, u_traj, max_iter=10, tol=1e-6)
        print("\nHOP-DDP solved:")
        print(f"T*: {result.T_star}")
        print(f"Total cost: {result.total_cost:.6f}")
        print(f"u shape: {result.u_opt.shape}")
        print(f"x shape: {result.x_opt.shape}")

        final_state = result.x_opt[-1]
        err = jnp.linalg.norm(final_state - x_target)
        print(f"Final state error: {err:.6f}")

        print("\nFirst 5 steps (u):")
        for i in range(min(5, len(result.u_opt))):
            print(f"step {i}: u = {result.u_opt[i]}")

        print("\nFirst 5 steps (x):")
        for i in range(min(5, len(result.x_opt))):
            print(f"step {i}: x = {result.x_opt[i]}")

        print(f"\nFinal state: {final_state}")
        print(f"Target state: {x_target}")
        return result
    except Exception as e:
        print(f"HOP-DDP failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("Double integrator HOP-DDP test")
    print("=" * 60)
    try:
        # Test 2: point-to-point
        test_point_to_point_ddp()

        print("\n" + "=" * 60)
        print("All tests completed")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
