"""
Cart-pole dynamics model.

State x = [x, x_dot, theta, theta_dot] in R^4
Control u = [force] in R^1

theta is measured from the upright (theta=0 means pole is vertical/up).
Positive theta corresponds to the pole rotating clockwise (towards +x).

Continuous-time dynamics (standard CartPole equations):
    x_dot = x_dot
    x_ddot = f_x(x, u)
    theta_dot = theta_dot
    theta_ddot = f_theta(x, u)

Discrete-time dynamics (Euler integration):
    x_next = x + x_dot * dt
    x_dot_next = x_dot + x_ddot * dt
    theta_next = theta + theta_dot * dt
    theta_dot_next = theta_dot + theta_ddot * dt
"""

from typing import Tuple
import jax
import jax.numpy as jnp


class CartPoleDynamics:
    """Cart-pole dynamics with standard parameters and Euler discretization."""

    def __init__(
        self,
        dt: float = 0.02,
        m_c: float = 1.0,
        m_p: float = 0.1,
        l: float = 0.5,
        g: float = 9.81,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if m_c <= 0.0 or m_p <= 0.0:
            raise ValueError("m_c and m_p must be positive")
        if l <= 0.0:
            raise ValueError("l must be positive")
        self.dt = float(dt)
        self.m_c = float(m_c)
        self.m_p = float(m_p)
        self.l = float(l)
        self.g = float(g)

        # State and control dimensions
        # State: [x, x_dot, theta, theta_dot]
        # Control: [force]
        self.nx = 4
        self.nu = 1
        # Uppercase aliases for compatibility
        self.Nx = self.nx
        self.Nu = self.nu

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time dynamics: x_dot = f(x, u)."""
        # Unpack state
        x_pos = x[0]
        x_vel = x[1]
        theta = x[2]
        theta_dot = x[3]

        # Control (force on cart)
        force = u[0]

        # Parameters
        m_c = self.m_c
        m_p = self.m_p
        l = self.l
        g = self.g

        total_mass = m_c + m_p
        polemass_length = m_p * l

        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        # Intermediate term
        temp = (force + polemass_length * theta_dot * theta_dot * sin_theta) / total_mass

        # Angular acceleration
        denom = l * (4.0 / 3.0 - (m_p * cos_theta * cos_theta) / total_mass)
        theta_ddot = (g * sin_theta - cos_theta * temp) / denom

        # Linear acceleration
        x_ddot = temp - (polemass_length * theta_ddot * cos_theta) / total_mass

        # State derivatives
        return jnp.array([x_vel, x_ddot, theta_dot, theta_ddot])

    def dxdt(self, xt: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Alias used by some integrators."""
        return self.dynamics(xt, u)

    def discrete_dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Discrete-time dynamics via Euler integration."""
        dt = self.dt
        x_dot = self.dynamics(x, u)
        return x + dt * x_dot

    def get_linearized_dynamics(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Continuous-time linearized dynamics (A, B) using autodiff."""
        A = jax.jacfwd(self.dynamics, argnums=0)(x, u)
        B = jax.jacfwd(self.dynamics, argnums=1)(x, u)
        return A, B

    def get_discrete_linearized_dynamics(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Discrete-time linearized dynamics (A_d, B_d) via Euler approximation."""
        A, B = self.get_linearized_dynamics(x, u)
        A_d = jnp.eye(self.nx) + self.dt * A
        B_d = self.dt * B
        return A_d, B_d

    # Compatibility helpers for solvers
    def getAt(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time A matrix for compatibility."""
        A, _ = self.get_linearized_dynamics(x, u)
        return A

    def getBt(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time B matrix for compatibility."""
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
        """Autodiff linearization for verification (same as get_linearized_dynamics)."""
        return self.get_linearized_dynamics(x, u)

    def get_discrete_linearized_dynamics_autodiff(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Autodiff linearization for the discrete dynamics."""
        A_d = jax.jacfwd(self.discrete_dynamics, argnums=0)(x, u)
        B_d = jax.jacfwd(self.discrete_dynamics, argnums=1)(x, u)
        return A_d, B_d
