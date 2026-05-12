import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys 
sys.path.append('..')
import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian, value_and_grad, lax
from jax.lax import scan
import jax.numpy as jnp
import numpy as np
from scipy.special import comb
from multiRobots_lib.ergodic_metric import ErgodicMetric # 计算遍历度指标
# 增广拉格朗日法优化器
from multiRobots_lib.solver import al_iLQR 
from multiRobots_lib.augument_lagrange_func import loss_traj_multi, eq_constr, ineq_constr_multi, loss_compare_multi
from multiRobots_lib.data_collect import save_ergodic_metrics_to_excel, append_metric
import yaml
import pickle as pkl
import copy
from openpyxl import Workbook, load_workbook
import logging
logging.basicConfig(
    filename='log/app.log',          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
from multiRobots_lib.hyper_params import (
    opt_args,
    target_distr,
    robot_distr,
    robot_number,
    init_state,
    sol_trajs,
    betas,
    robot_model,
)
update_map_freq = opt_args["update_map_freq"]
map_merge_freq = opt_args["map_merge_freq"] 
state_dim = robot_model.nx
robot_number = opt_args["robot_number"]
tsteps = opt_args["tsteps"]

# 双机器人初始化
start_pos = opt_args["x0"]
end_pos = opt_args["xf"]
# 假设你想创建包含 n 个 al_iLQR 结果的数组
n = robot_number  # 例如，创建 10 个结果
jit_ineq_constr_multi = jit(partial(ineq_constr_multi, warm_up=False))
warm_up_ineq_constr = jit(partial(ineq_constr_multi, warm_up=True))
traj_solver = []
traj_warmUp = []
for _id in range(n):
    solver = al_iLQR(
        args=opt_args,
        objective=loss_traj_multi,
        dynamics=robot_model,
        inequality=jit_ineq_constr_multi,
        equality=eq_constr,
        target_distr=robot_distr[_id].evals,
        robot_id=_id,
    )
    warmup = al_iLQR(
        args=opt_args,
        objective=loss_traj_multi,
        dynamics=robot_model,
        inequality=warm_up_ineq_constr,
        equality=eq_constr,
        target_distr=robot_distr[_id].evals,
        robot_id=_id,
    )
    solver.update_dynamics(robot_model, robot_number)
    warmup.update_dynamics(robot_model, robot_number)
    traj_solver.append(solver)
    traj_warmUp.append(warmup)


from multiRobots_lib.plot_utils import plot_trajs, plot_line_graph
from multiRobots_lib.tools import find_first_connected, exchange_info, optimize_trajs, update_accumulated_time
from multiRobots_lib.decay_utils import update_beta
from IPython.display import clear_output
itera = 3
itera_cnt = 0
connection_threshold = opt_args["connect_threshold"]
connection_threshold = connection_threshold**2
map_merge_cnt = 0
decay_type = 'linear'
# decay_type = 'nodecay'
init_dual = True
save_path = '../figures/probabilistic_connection.png'
involved_robots = list(range(robot_number))
involved_robots = set(involved_robots)
flag = True
accumulated_time = 0
be_num = 1
object_value = []
global_metric = {
        'time': [],
        'values': [],
    }
last_exchange_time = {}
# 迭代优化并动态可视化轨迹与障碍物分布
warm_up = True
logging.info('prob connect')
while itera_cnt < itera:
    if warm_up == True:
        sol_trajs, to_remove = optimize_trajs(involved_robots, sol_trajs, betas, traj_warmUp, init_state, init_dual, r_eps = 0.05, loss_eps = 1e-5)
    else:
        sol_trajs, to_remove = optimize_trajs(involved_robots, sol_trajs, betas, traj_solver, init_state, init_dual, r_eps = 0.10, loss_eps = 1e-5)
    involved_robots -= to_remove
    clear_output(wait=True)                   # 清除上一次输出，动态刷新
    init_dual = False
    itera_cnt += 1   
    if not involved_robots or itera_cnt == itera - 1:
        if warm_up == True and accumulated_time <= 60:
            warm_up = False
            involved_robots = set(list(range(robot_number)))
            init_dual = True
            itera_cnt = 0
            logging.info("have warm up")
            continue
        # ================================== 获取replan 信息 =========================================================
        current_time, robot_pair = find_first_connected(sol_trajs, connection_threshold, last_exchange_time)
        current_time, accumulated_time = update_accumulated_time(current_time, accumulated_time, be_num)
        map_merge_cnt += current_time
        if map_merge_cnt >= map_merge_freq:
            accumulated_time -= (map_merge_cnt - map_merge_freq)
            current_time -= (map_merge_cnt - map_merge_freq)
            map_merge_cnt = 0
            if accumulated_time >= tsteps:
                append_metric(global_metric, loss_compare_multi(sol_trajs, target_distr.evals, current_time))
                break
            robot_pair = []
            involved_robots = set(list(range(robot_number)))
            for r_id in range(robot_number):
                # print(f"current_time = {connect_timestesp}, type = {type(current_time)}")
                u_t = sol_trajs[r_id]['x'][current_time, r_id * state_dim: r_id * state_dim + 2]
                robot_distr[r_id].bayes_filter_reset(target_distr.evals[0], u_t)
                traj_solver[r_id].update_distribution(robot_distr[r_id].evals)
        # ================================ 为replan准备 ==============================================================  
        if accumulated_time == update_map_freq * be_num:
            append_metric(global_metric, loss_compare_multi(sol_trajs, target_distr.evals, current_time))
            if accumulated_time >= tsteps:
                break
            logging.info("update map")
            robot_pair = []
            target_distr.update_map(accumulated_time, "reset", "read")
            be_num += 1

        sol_trajs, connected_pairs = exchange_info(sol_trajs, robot_pair, current_time, robot_distr)
        betas = update_beta(sol_trajs, betas, decay_type)
        if not involved_robots:
            involved_robots = connected_pairs

        init_state = [traj['x'][0, :] for traj in sol_trajs]     
        init_dual = True
        itera_cnt = 0
        warm_up = True
        logging.warning(f"connect time:{current_time}, accumulated time:{accumulated_time}, and the robot pair:{robot_pair}")
    # plot_trajs(start_pos, end_pos, sol_trajs, betas, False, robot_distr, save_path)    
plot_trajs(start_pos, end_pos, sol_trajs, betas, True, robot_distr, save_path)

save_ergodic_metrics_to_excel(robot_number, global_metric, decay_type = 'prob_connect')