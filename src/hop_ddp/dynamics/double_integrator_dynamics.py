"""
Double integrator dynamics model for testing.

State x = [p, v] in R^(2*dim)
Control u = a in R^dim
Continuous-time dynamics:
    p_dot = v
    v_dot = u
Discrete-time dynamics (zero-order hold, exact for constant acceleration):
    p_next = p + v*dt + 0.5*u*dt^2
    v_next = v + u*dt
"""

from typing import Tuple
import jax
import jax.numpy as jnp


class DoubleIntegratorDynamics:
    """Double integrator dynamics with configurable dimension."""

    def __init__(self, dim: int = 1, dt: float = 0.01):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dim = int(dim)
        self.dt = float(dt)

        # State and control dimensions
        self.nx = 2 * self.dim
        self.nu = self.dim
        # Uppercase aliases for compatibility with some solvers
        self.Nx = self.nx
        self.Nu = self.nu

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time dynamics."""
        p = x[: self.dim]
        v = x[self.dim :]
        x_dot = jnp.concatenate([v, u], axis=0)
        return x_dot

    def dxdt(self, xt: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Alias used by some integrators."""
        return self.dynamics(xt, u)

    def discrete_dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Discrete-time dynamics with zero-order hold on u."""
        p = x[: self.dim]
        v = x[self.dim :]
        dt = self.dt
        p_next = p + v * dt + 0.5 * u * dt * dt
        v_next = v + u * dt
        return jnp.concatenate([p_next, v_next], axis=0)

    def get_linearized_dynamics(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Continuous-time linearized dynamics (A, B)."""
        _ = x
        _ = u
        dim = self.dim
        zeros = jnp.zeros((dim, dim))
        eye = jnp.eye(dim)
        A = jnp.block([[zeros, eye], [zeros, zeros]])
        B = jnp.vstack([zeros, eye])
        return A, B

    def get_discrete_linearized_dynamics(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Discrete-time linearized dynamics (A_d, B_d)."""
        _ = x
        _ = u
        dim = self.dim
        dt = self.dt
        zeros = jnp.zeros((dim, dim))
        eye = jnp.eye(dim)
        A_d = jnp.block([[eye, dt * eye], [zeros, eye]])
        B_d = jnp.vstack([0.5 * dt * dt * eye, dt * eye])
        return A_d, B_d

    def getAt(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time A matrix for compatibility with iLQR template."""
        A, _ = self.get_linearized_dynamics(x, u)
        return A

    def getBt(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time B matrix for compatibility with iLQR template."""
        _, B = self.get_linearized_dynamics(x, u)
        return B

    def getAd(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Discrete-time A_d matrix."""
        A_d, _ = self.get_discrete_linearized_dynamics(x, u)
        return A_d

    def getBd(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Discrete-time B_d matrix."""
        _, B_d = self.get_discrete_linearized_dynamics(x, u)
        return B_d

    def get_linearized_dynamics_autodiff(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Autodiff linearization for verification."""
        A = jax.jacfwd(self.dynamics, argnums=0)(x, u)
        B = jax.jacfwd(self.dynamics, argnums=1)(x, u)
        return A, B

    def get_discrete_linearized_dynamics_autodiff(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Autodiff linearization for the discrete dynamics."""
        A_d = jax.jacfwd(self.discrete_dynamics, argnums=0)(x, u)
        B_d = jax.jacfwd(self.discrete_dynamics, argnums=1)(x, u)
        return A_d, B_d
