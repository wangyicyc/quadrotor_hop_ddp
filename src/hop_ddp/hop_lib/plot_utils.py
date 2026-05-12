#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Optional


def visualize_quadrotor_results(x_opt: jnp.ndarray, 
                              u_opt: jnp.ndarray, 
                              config, 
                              save_path: Optional[str] = None,
                              show_plot: bool = True,
                              close_after: bool = False):
    """
    可视化四旋翼无人机的仿真结果
    
    参数:
        x_opt: 最优状态轨迹 (T+1, 12)
        u_opt: 最优控制轨迹 (T, 4)
        config: 四旋翼配置
        save_path: 图片保存路径（可选）
        show_plot: 是否立即显示图像
        close_after: 是否在保存/显示后关闭图像
    """
    T_state = x_opt.shape[0]  # 状态序列长度 (T+1)
    T_control = u_opt.shape[0]  # 控制序列长度 (T)
    
    # 时间轴
    dt = config.dt
    time_states = np.arange(T_state) * dt
    time_controls = np.arange(T_control) * dt
    
    # 提取状态变量
    pos_x = np.array(x_opt[:, 0])  # x位置
    pos_y = np.array(x_opt[:, 1])  # y位置
    pos_z = np.array(x_opt[:, 2])  # z位置
    
    vel_x = np.array(x_opt[:, 3])  # x速度
    vel_y = np.array(x_opt[:, 4])  # y速度
    vel_z = np.array(x_opt[:, 5])  # z速度
    
    att_roll = np.array(x_opt[:, 6])   # 滚转角
    att_pitch = np.array(x_opt[:, 7])  # 俯仰角
    att_yaw = np.array(x_opt[:, 8])    # 偏航角
    
    omega_x = np.array(x_opt[:, 9])   # x轴角速度
    omega_y = np.array(x_opt[:, 10])  # y轴角速度
    omega_z = np.array(x_opt[:, 11])  # z轴角速度
    
    # 提取控制变量
    thrust1 = np.array(u_opt[:, 0])
    thrust2 = np.array(u_opt[:, 1])
    thrust3 = np.array(u_opt[:, 2])
    thrust4 = np.array(u_opt[:, 3])
    
    # 目标状态
    target_pos_x = config.x_target[0]
    target_pos_y = config.x_target[1]
    target_pos_z = config.x_target[2]
    
    # 创建图形：3D 轨迹作为主视图，其他时序图作为辅助诊断视图
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.45, 1.45, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.38,
        hspace=0.36,
    )
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(gs[:, :2], projection='3d')
    # 绘制轨迹线，增加透明度和美化
    ax1.plot(pos_x, pos_y, pos_z, color='#1f77b4', linewidth=3, alpha=0.8, label='Trajectory')
    # 起始点，使用更明显的标记
    ax1.scatter([config.x0[0]], [config.x0[1]], [config.x0[2]], color='#d62728', s=150, alpha=0.9, label='Start', marker='o', edgecolors='black', linewidth=0.5)
    # 目标点，使用更明显的标记
    ax1.scatter([target_pos_x], [target_pos_y], [target_pos_z], color='#2ca02c', s=150, alpha=0.9, label='Target', marker='^', edgecolors='black', linewidth=0.5)
    # 设置坐标轴标签
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m)', fontsize=11)
    # 设置标题
    ax1.set_title('3D Trajectory', fontsize=16, fontweight='bold')
    # 设置网格
    ax1.grid(True, alpha=0.3)
    # 设置图例
    ax1.legend(prop={'size': 11})
    ax1.view_init(elev=24, azim=-58)
    
    # 2. 位置子图
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time_states, pos_x, color='#1f77b4', label='X Position', linewidth=2)
    ax2.plot(time_states, pos_y, color='#ff7f0e', label='Y Position', linewidth=2)
    ax2.plot(time_states, pos_z, color='#2ca02c', label='Z Position', linewidth=2)
    ax2.axhline(y=target_pos_x, color='#1f77b4', linestyle='--', label='Target X')
    ax2.axhline(y=target_pos_y, color='#ff7f0e', linestyle='--', label='Target Y')
    ax2.axhline(y=target_pos_z, color='#2ca02c', linestyle='--', label='Target Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.grid(True)
    ax2.legend(fontsize=8)
    
    # 3. 速度子图
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(time_states, vel_x, label='X Velocity', linewidth=2)
    ax3.plot(time_states, vel_y, label='Y Velocity', linewidth=2)
    ax3.plot(time_states, vel_z, label='Z Velocity', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.grid(True)
    ax3.legend(fontsize=8)
    
    # 4. 姿态子图
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(time_states, att_roll, label='Roll', linewidth=2)
    ax4.plot(time_states, att_pitch, label='Pitch', linewidth=2)
    ax4.plot(time_states, att_yaw, label='Yaw', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Attitude (rad)')
    ax4.set_title('Attitude vs Time')
    ax4.grid(True)
    ax4.legend(fontsize=8)
    
    # 5. 角速度子图
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.plot(time_states[:-1], omega_x[:-1], label='X Angular Vel', linewidth=2)  # 注意这里时间点比控制少1
    ax5.plot(time_states[:-1], omega_y[:-1], label='Y Angular Vel', linewidth=2)
    ax5.plot(time_states[:-1], omega_z[:-1], label='Z Angular Vel', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Velocity (rad/s)')
    ax5.set_title('Angular Velocity vs Time')
    ax5.grid(True)
    ax5.legend(fontsize=8)
    
    # 6. 控制输入子图
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.plot(time_controls, thrust1, label='Thrust 1', linewidth=2)
    ax6.plot(time_controls, thrust2, label='Thrust 2', linewidth=2)
    ax6.plot(time_controls, thrust3, label='Thrust 3', linewidth=2)
    ax6.plot(time_controls, thrust4, label='Thrust 4', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Thrust (N)')
    ax6.set_title('Control Inputs vs Time')
    ax6.grid(True)
    ax6.legend(fontsize=8, ncol=4)
    
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.94)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"图片已保存至: {save_path}")

    if show_plot:
        plt.show()

    if close_after:
        plt.close(fig)


def plot_cost_convergence(cost_history: list, title: str = "Cost Convergence"):
    """
    绘制代价收敛历史
    
    参数:
        cost_history: 代价历史列表
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_tracking_errors(x_opt: jnp.ndarray, x_target: jnp.ndarray, config,
                         save_path: Optional[str] = None,
                         show_plot: bool = True,
                         close_after: bool = False):
    """
    绘制跟踪误差
    
    参数:
        x_opt: 最优状态轨迹
        x_target: 目标状态
        config: 配置对象
    """
    T_state = x_opt.shape[0]
    dt = config.dt
    time_states = np.arange(T_state) * dt
    
    # 计算位置误差
    pos_error = np.linalg.norm(
        x_opt[:, :3] - np.array(x_target[:3]), axis=1
    )
    
    # 计算速度误差
    vel_error = np.linalg.norm(
        x_opt[:, 3:6] - np.array(x_target[3:6]), axis=1
    )
    
    # 计算姿态误差
    att_delta = x_opt[:, 6:9] - np.array(x_target[6:9])
    att_delta = (att_delta + np.pi) % (2 * np.pi) - np.pi
    att_error = np.linalg.norm(
        att_delta, axis=1
    )
    
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(time_states, pos_error, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Tracking Error')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(time_states, vel_error, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.title('Velocity Tracking Error')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(time_states, att_error, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Attitude Error (rad)')
    plt.title('Attitude Tracking Error')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    if close_after:
        plt.close(fig)
