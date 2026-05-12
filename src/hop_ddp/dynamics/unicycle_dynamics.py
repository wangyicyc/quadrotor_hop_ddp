"""
Unicycle dynamics model for a 2D mobile robot.

State x = [x, y, theta] in R^3
Control u = [v, omega] in R^2
Continuous-time dynamics:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = omega
Discrete-time dynamics (Euler integration):
    x_next = x + v * cos(theta) * dt
    y_next = y + v * sin(theta) * dt
    theta_next = theta + omega * dt
"""

from typing import Tuple
import jax
import jax.numpy as jnp


class UnicycleDynamics:
    """Unicycle dynamics for a 2D mobile robot."""

    def __init__(self, dt: float = 0.01):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)

        # State and control dimensions
        # State: [x, y, theta]
        # Control: [v, omega]
        self.nx = 3
        self.nu = 2
        # Uppercase aliases for compatibility
        self.Nx = self.nx
        self.Nu = self.nu

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time dynamics: x_dot = f(x, u)."""
        v, omega = u[0], u[1]
        theta = x[2]
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)
        theta_dot = omega
        return jnp.array([x_dot, y_dot, theta_dot])

    def dxdt(self, xt: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Alias used by some integrators."""
        return self.dynamics(xt, u)

    def discrete_dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Discrete-time dynamics via Euler integration."""
        v, omega = u[0], u[1]
        theta = x[2]
        dt = self.dt
        x_next = x[0] + v * jnp.cos(theta) * dt
        y_next = x[1] + v * jnp.sin(theta) * dt
        theta_next = x[2] + omega * dt
        # 可选：将角度规整到 [-pi, pi) 区间
        # theta_next = (theta_next + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.array([x_next, y_next, theta_next])

    def get_linearized_dynamics(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Continuous-time linearized dynamics (A, B) around (x, u)."""
        # 解析计算雅可比矩阵
        v, omega = u[0], u[1]
        theta = x[2]
        # A = df/dx
        A = jnp.array([
            [0, 0, -v * jnp.sin(theta)],
            [0, 0,  v * jnp.cos(theta)],
            [0, 0, 0]
        ])
        # B = df/du
        B = jnp.array([
            [jnp.cos(theta), 0],
            [jnp.sin(theta), 0],
            [0, 1]
        ])
        return A, B

    def get_discrete_linearized_dynamics(
        self, x: jnp.ndarray, u: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Discrete-time linearized dynamics (A_d, B_d) via Euler approximation."""
        A, B = self.get_linearized_dynamics(x, u)
        A_d = jnp.eye(self.nx) + self.dt * A
        B_d = self.dt * B
        return A_d, B_d

    # 以下方法用于与iLQR/HOP-DDP等求解器模板兼容
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