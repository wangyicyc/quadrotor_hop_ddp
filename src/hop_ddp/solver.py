import jax
import numpy as np  
from functools import partial
from jax import value_and_grad, grad, jacfwd, vmap, jit, hessian
import jax.numpy as jnp
from multiRobots_lib.integrator import rk4 as int_func
from jax.lax import scan
from multiRobots_lib.class_types import *
import logging
from datetime import datetime
# 生成包含年月日_时分的日志文件名，例如：app_2026-01-16_15-06.log
log_filename = f"../datas/logs/app_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
logging.basicConfig(
    filename=log_filename,          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
import yaml
with open("../datas/config/config.yaml", "r") as f:
    loaded = yaml.safe_load(f)
# 提取 opt_args 子字典
config = loaded["opt_args"]
robot_number = config["robot_number"]

# see https://github.com/MurpheyLab/ergodic-control-sandbox/blob/main/notebooks/ilqr_ergodic_control.ipynb
# I speed up the algorithm
class iLQR_template:
    def __init__(self, dt, tsteps, Q_z, R_v, dynamics: callable) -> None:
        self.dt = dt
        self.tsteps = tsteps
        self.tf = dt * tsteps
        self.x_dim = getattr(dynamics, "Nx", getattr(dynamics, "nx", None))
        self.u_dim = getattr(dynamics, "Nu", getattr(dynamics, "nu", None))
        self.dynamics = dynamics
        self.Q_z = Q_z
        self.Q_z_inv = jnp.linalg.inv(Q_z)
        self.R_v = R_v
        self.R_v_inv = jnp.linalg.inv(R_v)
        self.dyn_step = jit(partial(int_func, dxdt=self.dynamics.dxdt, dt=self.dt))
        def dyn_step_fn(x, u):
            return self.dyn_step(xt=x, u=u), x
        self.dyn_step_fn = jit(dyn_step_fn)
        # the following functions are utilities for solving the Riccati equation
        # P
        def P_dyn_rev(Pt, At, Bt, at, bt):
            return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_v_inv @ Bt.T @ Pt + self.Q_z
        self.P_dyn_step = jit(partial(int_func, dxdt=P_dyn_rev, dt=self.dt))
        def P_dyn_step_fn(Pt, inputs: LinearizedDynamics):
            At, Bt, at, bt = inputs.At, inputs.Bt, inputs.at, inputs.bt
            return self.P_dyn_step(xt=Pt, At=At, Bt=Bt, at=at, bt=bt), Pt
        self.P_dyn_step_fn = jit(P_dyn_step_fn)
        # r
        def r_dyn_rev(rt, Pt, At, Bt, at, bt):
            return (
                (At - Bt @ self.R_v_inv @ Bt.T @ Pt).T @ rt
                + at
                - Pt @ Bt @ self.R_v_inv @ bt
            )
        self.r_dyn_step = jit(partial(int_func, dxdt=r_dyn_rev, dt=self.dt))
        
        def r_dyn_step_fn(rt, inputs: RiccatiInput):
            return self.r_dyn_step(xt=rt, Pt=inputs.Pt, At=inputs.At, Bt=inputs.Bt, at=inputs.at, bt=inputs.bt), rt
        self.r_dyn_step_fn = jit(r_dyn_step_fn)

        # z /delta
        def z2v(zt, Pt, rt, Bt, bt):
            return (
                -self.R_v_inv @ Bt.T @ Pt @ zt
                - self.R_v_inv @ Bt.T @ rt
                - self.R_v_inv @ bt
            )

        self.z2v = jit(z2v)

        def z_dyn(zt, Pt, rt, At, Bt, bt):
            return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)

        self.z_dyn_step = jit(partial(int_func, dxdt=z_dyn, dt=self.dt))

        def z_dyn_step_fn(zt, inputs: ZDynamicsInput):
            return self.z_dyn_step(xt=zt, Pt=inputs.Pt,
                rt=inputs.rt,   # ⚠ 如果你把 rt 存在 at，这里注意检查
                At=inputs.At, Bt=inputs.Bt, bt=inputs.bt), zt

        self.z_dyn_step_fn = jit(z_dyn_step_fn)

        # self.temp = {'A_traj':[], 'B_traj':[], 'a_traj':[], 'b_traj':[], 'P_traj':[], 'r_traj':[], 'z_traj':[], 'v_traj':[], 'x_traj':[], 'u_traj':[]}

    def loss(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_at_vec(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_bt_vec(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_at_bt_traj(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def traj_sim(self, x0, u_traj):
        xN, x_traj = scan(self.dyn_step_fn, x0, u_traj)
        return x_traj

    def get_descent(self, x0, u_traj, past_traj, target_distribution, dual_solution, r_penalty):  # 添加参数
        # forward simulate the trajectory
        xN, x_traj = scan(self.dyn_step_fn, x0, u_traj)
        # sovle the Riccati equation backward in time
        A_traj = vmap(self.dynamics.getAt)(x_traj, u_traj)
        B_traj = vmap(self.dynamics.getBt)(x_traj, u_traj)
        a_traj, b_traj = self.get_at_bt_traj(
            TrajectorySolution(x=x_traj, u=u_traj, px=past_traj),
            target_distribution, dual_solution, r_penalty
        )
        PN = jnp.zeros((self.x_dim, self.x_dim))
        P0, P_traj = scan(
            f=self.P_dyn_step_fn,
            init=PN,
            reverse=True,
            xs = LinearizedDynamics(At=A_traj, Bt=B_traj, at=a_traj, bt=b_traj)
        )
        P_traj = jnp.vstack([P0[jnp.newaxis, :], P_traj])[:-1]
        rN = jnp.zeros(self.x_dim)
        r0, r_traj = scan(
            f=self.r_dyn_step_fn,
            init=rN,
            reverse=True,
            xs = RiccatiInput(Pt=P_traj, At=A_traj, Bt=B_traj, at=a_traj, bt=b_traj)
        )
        r_traj = jnp.vstack([r0[jnp.newaxis, :], r_traj])[:-1]
        z0 = jnp.zeros(self.x_dim)
        zN, z_traj = scan(
            f=self.z_dyn_step_fn,
            init=z0,
            xs = ZDynamicsInput(Pt=P_traj, At=A_traj, Bt=B_traj, rt=r_traj, bt=b_traj)
        )
        # compute the descent direction
        v_traj = vmap(self.z2v)(z_traj, P_traj, r_traj, B_traj, b_traj)
        return v_traj
    def solve(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

class al_iLQR(iLQR_template):
    def __init__(
        self, args: dict, objective: callable, dynamics: callable, inequality: callable
    , equality: callable, target_distr, robot_id: int) -> None:
        super().__init__(
            dt=args["dt"],
            tsteps=args["tsteps"],
            Q_z = jnp.diag(args["Q_z"]),
            R_v=jnp.diag(args["R_v"]),
            dynamics=dynamics,
        )
        self.args = args
        self.objective = jit(objective)#, static_argnames=['robot_id'])
        self.inequality = jit(inequality)
        self.equality = jit(equality)
        self.R = jnp.diag(args["R"])
        self.robot_id = robot_id
        self.target_distribution = target_distr 
        self.U_min = args["U_min"]
        self.U_max = args["U_max"]

        self.r_penalty = 1.0
        self.dual_solution = None
        self.init_state = None
        self.solution = None
        self.beta = None
        def lagrangian(solution:TrajectorySolution, dual_solution, r, target_distribution):
            mu = dual_solution.mu
            lam = dual_solution.lam
            _objective = self.objective(solution, self.beta, target_distribution)
            _ineq_constr = self.inequality(solution)
            _eq_constr = self.equality(solution)
            return _objective + (0.5 / r) * jnp.sum(
                jnp.maximum(0.0, mu + r * _ineq_constr) ** 2 - mu**2
                + jnp.sum(lam * _eq_constr + r*0.5 * (_eq_constr)**2)
            )
            # + jnp.sum(lam * _eq_constr + r*0.5 * (_eq_constr)**2)
        self.lagrangian = jit(lagrangian)
        self.lagrangian_grad = jit(grad(lagrangian, argnums=0))

        def _loss_func(_step, solution, _u_direct, dual, penalty, target_distribution):
            ctrl = solution.u + _step * _u_direct
            x_traj = self.traj_sim(self.init_state, ctrl)
            new_sol = TrajectorySolution(x=x_traj, u=ctrl, px=solution.px)  # ✅ 构造 NamedTuple
            return self.lagrangian(new_sol, dual, penalty, target_distribution)
        self.loss_func4linesearch = jit(_loss_func)

    def get_at_bt_traj(self, solution, target_distribution, dual_solution, r_penalty):  # 添加参数
        grad_val = self.lagrangian_grad(solution, dual_solution, r_penalty, target_distribution)  # 使用参数
        return grad_val.x, grad_val.u

    def update_multipliers(self):
        new_mu = jnp.maximum(0, self.dual_solution.mu + self.r_penalty * self.inequality(self.solution))
        new_lam = self.dual_solution.lam + self.r_penalty * self.equality(self.solution)

        self.dual_solution = self.dual_solution._replace(mu=new_mu, lam = new_lam)
        
    def linesearch(
        self, solution, dual_solution, u_direct, target_distribution, r_penalty, max_iter=50, initial_step=1.0, gamma=0.8
    ):
        steps_arr = jnp.array([initial_step * gamma**i for i in range(max_iter)])
        loss_arr = vmap(
        jit(
                partial(
                    self.loss_func4linesearch,
                    solution=solution,
                    _u_direct=u_direct,
                    dual=dual_solution,
                    penalty=r_penalty,
                    target_distribution=target_distribution
                )
            )
        )(_step=steps_arr)
        min_loss_idx = jnp.argmin(loss_arr)
        min_step = steps_arr[min_loss_idx]
        min_loss = loss_arr[min_loss_idx]
        ctrl = jax.lax.cond(
            min_loss <= self.lagrangian(
            solution, dual_solution, r_penalty, target_distribution),
            lambda _: solution.u + min_step * u_direct,
            lambda _: solution.u,  # 如果条件不满足，返回原控制输入
            operand=None
        )
        return ctrl
    def update_distribution(self, new_distribution):
        self.target_distribution = new_distribution

    def solve(self, x0, init_sol, beta, init_dual=True, max_iter=100, r_eps = 0.1, loss_eps = 1e-6, decay_eps=0.05, if_print=True):
        self.init_state = x0
        self.solution = TrajectorySolution(
            x=jnp.array(init_sol["x"]),
            u=jnp.array(init_sol["u"]),
            px=init_sol["px"])
        
        beta_x_raw = jnp.asarray(beta["x"]) 
        total_norm = (
            sum(jnp.sum(px) for px in beta["px"]) +
            jnp.sum(beta_x_raw) +
            1e-8
        )
        self.beta = BetaCoefficients(
            x = beta_x_raw / total_norm,
            px = [px / total_norm for px in beta["px"]]
        )
        self.beta = BetaCoefficients(x=jnp.asarray(beta["x"]), px=beta["px"])
        # if init_dual is True:
        self.get_descent_jit = jit(self.get_descent)
        self.linesearch_jit = jit(self.linesearch)
            # self.dual_solution = {"mu": jnp.zeros_like(self.inequality(self.solution)),\
            #     "lam": jnp.zeros_like(self.equality(self.solution))}
        self.dual_solution = DualVariables(mu=jnp.zeros_like(self.inequality(self.solution)), lam=jnp.zeros_like(self.equality(self.solution)))
        self.update_multipliers()
        self.r_penalty = 1.0

        loss_val = [self.objective(self.solution, self.beta, self.target_distribution)]
        _func_get_violation = jit(
            lambda sol: jnp.maximum(0, self.inequality(sol)).sum()
            + jnp.abs(self.equality(sol)).sum()
            # lambda sol: jnp.abs(self.equality(sol, self.args, self.robot_id)).sum()
        )
        violations = [_func_get_violation(self.solution)]
        # iterative optimization
        for i in range(max_iter):
            # solver LQR Problem
            v_traj = self.get_descent_jit(
                self.init_state, self.solution.u,
                self.solution.px, self.target_distribution, 
                self.dual_solution, self.r_penalty  # 传入当前状态
            )
            # line search
            _u_traj = self.linesearch_jit(self.solution, self.dual_solution, v_traj, self.target_distribution, self.r_penalty)
            self.solution = self.solution._replace(u=_u_traj, x=self.traj_sim(self.init_state, _u_traj))
            # loss_val.append(
            #     self.lagrangian(self.solution, self.dual_solution, self.r_penalty, self.target_distribution)
            # )
            loss_val.append(
                self.objective(self.solution, self.beta, self.target_distribution)
            )
            violations.append(_func_get_violation(self.solution))
            if if_print and (i+1) % 40 == 0:
                print(
                    "robot_id:{:d}\titer: {:d}\tobjective: {:.6f}\tlagrangian: {:.6f}\tviolation: {:.6f}\tpenalty: {:.6f}".format(
                        self.robot_id,
                        i,
                        self.objective(self.solution, self.beta, self.target_distribution),
                        self.lagrangian(self.solution, self.dual_solution, self.r_penalty, self.target_distribution),
                        violations[-1],
                        self.r_penalty,
                    )
                )
            # if (i+1) % 50 == 0:
                # logging.info(f"all violations:{violations}")
                # logging.info(f"id = {self.robot_id}, mu mean  = {jnp.mean(self.dual_solution.mu)}")
                # logging.info(f"id = {self.robot_id}, mu norm  = {jnp.linalg.norm(self.dual_solution.mu)}")
                # logging.info(f"id = {self.robot_id}, violations  = {violations[-1]}, r = {self.r_penalty}")
                # logging.info("r:{:6f} and violations:{:.6f} and loss{:.6f}".format(self.r_penalty, violations[-1], loss_val[-1]))
            self.update_multipliers()
            if (loss_val[-2] - loss_val[-1]) < decay_eps and jnp.abs(violations[-1]) > r_eps:
                self.r_penalty = jnp.clip(self.r_penalty * 1.05, 1e-10, 1e5)
                decay_eps *= 0.95            
            if (jnp.abs(loss_val[-1] - loss_val[-2]) < loss_eps) and jnp.abs(
                violations[-1]) < r_eps:
                logging.info("iter:{:d}, id:{:d}, r:{:.3f} and violateion:{:.3f}".format(i, self.robot_id, self.r_penalty, violations[-1]))
                return {
                    "x": np.array(self.solution.x),    # ← 转为可变的 NumPy array
                    "u": np.array(self.solution.u),
                    "px": [np.array(arr) for arr in self.solution.px]
                }, True

        if jnp.abs(violations[-1]) > r_eps:
            logging.info("failed to satisfy constraint, id: {:d} and r_penalty: {:.3f} and violations:{:.3f}:".format(self.robot_id, self.r_penalty, violations[-1]))
        else:
            logging.info("satisfy constraint, but not converge, id: {:d} and r_penalty: {:.3f} and violations:{:.3f}:".format(self.robot_id, self.r_penalty, violations[-1]))
        return {
            "x": np.array(self.solution.x),    # ← 转为可变的 NumPy array
            "u": np.array(self.solution.u),
            "px": [np.array(arr) for arr in self.solution.px]
        }, True