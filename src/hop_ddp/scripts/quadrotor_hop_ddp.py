#!/usr/bin/env python3
"""
四旋翼无人机HOP-DDP测试脚本

测试HOP-DDP求解器在四旋翼无人机非线性控制中的应用
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT)) 
import atexit

import jax.numpy as jnp
from hop_lib.hop_ddp_solver import HOPDDPSolver
from hop_lib.utils import get_run_dir
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from config.quadrotor_config import (
    QuadrotorConfig,
    get_cost_matrices,
    get_motor_thrust_limits,
    get_thrust_moment_limits,
)

# 导入可视化工具
from hop_lib.plot_utils import visualize_quadrotor_results, plot_tracking_errors


class _TeeStream:
    """Write console output to both the terminal and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, text):
        self._stream.write(text)
        if not self._log_file.closed:
            self._log_file.write(text)
        self.flush()

    def flush(self):
        self._stream.flush()
        if not self._log_file.closed:
            self._log_file.flush()


_console_log_file = None
_original_stdout = None
_original_stderr = None


def _close_console_log():
    global _console_log_file
    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr
    if _console_log_file is not None and not _console_log_file.closed:
        _console_log_file.close()


def _setup_console_log(run_dir: Path) -> Path:
    global _console_log_file, _original_stdout, _original_stderr
    if _console_log_file is not None:
        return run_dir / "console.log"

    console_log_path = run_dir / "console.log"
    _console_log_file = open(console_log_path, "a", encoding="utf-8")
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = _TeeStream(_original_stdout, _console_log_file)
    sys.stderr = _TeeStream(_original_stderr, _console_log_file)
    atexit.register(_close_console_log)
    return console_log_path


def state_error_scaled(x: jnp.ndarray, x_ref: jnp.ndarray) -> jnp.ndarray:
    """State error in current coordinates; Euler angles are stored in radians."""
    x_err = x - x_ref
    angle_err = (x_err[6:9] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    return x_err.at[6:9].set(angle_err)


def stage_cost(x: jnp.ndarray, u: jnp.ndarray, x_ref: jnp.ndarray,
               u_ref: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray) -> float:
    """
    阶段代价函数
    参数:
        x: 当前状态 (12,)
        u: 当前控制 (4,)
        x_ref: 参考状态 (12,)
        Q: 状态权重矩阵 (12, 12)
        R: 控制权重矩阵 (4, 4)
    返回:
        cost: 阶段代价
    """
    x_err = state_error_scaled(x, x_ref)
    u_err = u - u_ref
    cost = 0.5 * (x_err @ Q @ x_err + u_err @ R @ u_err)
    return cost


def terminal_cost(x: jnp.ndarray, x_ref: jnp.ndarray, 
                  Q_T: jnp.ndarray) -> float:
    """
    终端代价函数
    
    参数:
        x: 终端状态 (12,)
        x_ref: 参考状态 (12,)
        Q_T: 终端状态权重矩阵 (12, 12)
    
    返回:
        cost: 终端代价
    """
    x_err = state_error_scaled(x, x_ref)
    cost = 0.5 * (x_err @ Q_T @ x_err)
    return cost


def make_absolute_dynamics(dynamics, integrator: str = "rk4"):
    """Build absolute-control dynamics function."""
    if integrator not in {"rk4", "euler"}:
        raise ValueError(f"Unsupported quadrotor integrator: {integrator}")
    if integrator == "euler":
        return dynamics.discrete_dynamics
    else:
        return dynamics.discrete_dynamics_rk4


def make_quadrotor_cost_functions(dynamics, Q, R, Q_T, x_target):
    """Build stage and terminal costs for the quadrotor."""
    u_ref = dynamics.hover_control()

    def l_func(x, u):
        return stage_cost(x, u, x_target, u_ref, Q, R)

    def phi_func(x):
        return terminal_cost(x, x_target, Q_T)

    return l_func, phi_func


def test_trajectory_tracking_ddp():
    """测试轨迹跟踪（HOP-DDP）"""
    print("\n" + "="*60)
    print("测试2: 四旋翼无人机轨迹跟踪（HOP-DDP）")
    print("="*60)
    
    # 创建配置并设置为轨迹跟踪
    config = QuadrotorConfig()
    # 创建动力学模型
    dynamics = QuadrotorDynamics(config)
    u_hover = dynamics.hover_control()
    motor_f_min, motor_f_max = get_motor_thrust_limits(config)
    thrust_moment_limits = get_thrust_moment_limits(config)
    
    # 获取代价矩阵（已在缩放后坐标系中，无 3283x 膨胀因子）
    Q, R, Q_T = get_cost_matrices(config)
    run_dir = Path(get_run_dir())
    iter_plot_dir = run_dir
    iter_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] Run artifacts directory: {run_dir}")
    print(
        "[diag] Loaded quadrotor cost scales:"
        f" q_pos={config.CostMatrices.q_pos:.3e},"
        f" q_vel={config.CostMatrices.q_vel:.3e},"
        f" q_att=[{config.CostMatrices.q_roll:.3e},"
        f"{config.CostMatrices.q_pitch:.3e},"
        f"{config.CostMatrices.q_yaw:.3e}],"
        f" q_omega={config.CostMatrices.q_omega:.3e},"
        f" R_diag=[{config.CostMatrices.r_thrust:.3e},"
        f"{config.CostMatrices.r_tau_phi:.3e},"
        f"{config.CostMatrices.r_tau_theta:.3e},"
        f"{config.CostMatrices.r_tau_psi:.3e}],"
        f" terminal_scale={config.CostMatrices.terminal_cost_scale:.3e}"
    )
    print(
        "[diag] Control bounds (abs):"
        f" u_min={jnp.asarray(config.u_limit[0])},"
        f" u_max={jnp.asarray(config.u_limit[1])}"
    )
    print(
        "[diag] Motor thrust bounds:"
        f" f_min={motor_f_min:.3e},"
        f" f_max={motor_f_max:.3e},"
        f" thrust_moment_limits={thrust_moment_limits}"
    )
    print(
        "[diag] Hover control:"
        f" u_hover={jnp.asarray(u_hover)}"
    )
    print(
        "[diag] Loaded cost matrix eig ranges:"
        f" eig(Q)=[{float(jnp.min(jnp.linalg.eigvalsh(Q))):.3e}, {float(jnp.max(jnp.linalg.eigvalsh(Q))):.3e}],"
        f" eig(R)=[{float(jnp.min(jnp.linalg.eigvalsh(R))):.3e}, {float(jnp.max(jnp.linalg.eigvalsh(R))):.3e}],"
        f" eig(Q_T)=[{float(jnp.min(jnp.linalg.eigvalsh(Q_T))):.3e}, {float(jnp.max(jnp.linalg.eigvalsh(Q_T))):.3e}]"
    )

    x0 = config.x0
    f_abs = make_absolute_dynamics(dynamics, integrator="euler")
    l_func, phi_func = make_quadrotor_cost_functions(dynamics, Q, R, Q_T, config.x_target)

    def iteration_visualizer(iteration, x_traj_iter, u_traj_iter, T_star_iter, cost_iter, accepted_step):
        """迭代回调"""
        if visualize_quadrotor_results is None:
            return
        x_vis = jnp.asarray(x_traj_iter[:T_star_iter + 1])
        u_vis_abs = jnp.asarray(u_traj_iter[:T_star_iter])
        step_tag = "accepted" if accepted_step else "rejected"
        save_path = iter_plot_dir / f"iter_{iteration:03d}_T{T_star_iter:03d}_{step_tag}.png"
        visualize_quadrotor_results(
            x_vis,
            u_vis_abs,
            config,
            save_path=str(save_path),
            show_plot=False,
            close_after=True,
        )
    # 创建HOP-DDP求解器 - 使用缩放后的状态空间
    solver = HOPDDPSolver(
        f=f_abs,
        l=l_func,
        phi=phi_func,
        n=dynamics.nx,
        m=dynamics.nu,
        config=config,
        u_limit=config.u_limit,
        trajectory_generator=dynamics.linear_trajectory_control,
        x_target=config.x_target,
        iteration_callback=iteration_visualizer,
        # linearization_method="finite_difference",
        finite_difference_eps=1e-5,
        wrap_indices=(6, 7, 8),
    )
    # 关闭未暴露到 QuadrotorConfig 的 HOP-LQR 数值保护，用于复现原始失效链路。
    # solver.hop_lqr.enable_conditioned_factorization = False
    # solver.hop_lqr.enable_recursive_bar_f_growth_cap = False
    # solver.hop_lqr.enable_recursive_early_bar_f_gain_cap = False
    # solver.hop_lqr.enable_recursive_temp_relative_cap = False

    # 初始状态
    x0 = config.x0
    # 初始控制猜测：public tutorial 使用绝对控制，初值为 hover control。
    u_traj = jnp.tile(u_hover[None, :], (config.tsteps, 1))
    # 求解HOP-DDP问题
    print("正在求解HOP-DDP问题...")

    result = solver.solve(x0, u_traj, T_min=config.Settings.T_min, max_iter=config.Settings.max_iter, tol=config.Settings.tol)
    u_opt = result.u_opt
    u_opt_abs = u_opt
    # 将最优状态轨迹从缩放空间反变换回物理单位
    x_opt = jnp.asarray(result.x_opt)
    cost = result.total_cost
    # 调用可视化函数
    if visualize_quadrotor_results is not None:
        print("\n正在生成可视化图表...")
        trajectory_path = run_dir / "quadrotor_final_trajectory.png"
        error_path = run_dir / "quadrotor_tracking_errors.png"
        visualize_quadrotor_results(
            x_opt,
            u_opt_abs,
            config,
            save_path=str(trajectory_path),
            show_plot=False,
            close_after=True,
        )
        # 绘制跟踪误差
        plot_tracking_errors(
            x_opt,
            config.x_target,
            config,
            save_path=str(error_path),
            show_plot=False,
            close_after=True,
        )
        print(f"最终轨迹图已保存至: {trajectory_path}")
        print(f"跟踪误差图已保存至: {error_path}")
        
def main():
    """主测试函数"""
    run_dir = Path(get_run_dir())
    console_log_path = _setup_console_log(run_dir)
    print("四旋翼无人机HOP-DDP求解器测试")
    print("="*60)
    print(f"[diag] Console log: {console_log_path}")
    
    # 测试: 轨迹跟踪
    test_trajectory_tracking_ddp()
        
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()
