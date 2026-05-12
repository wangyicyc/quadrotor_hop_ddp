"""
HOP-LQR (Horizon Optimal Planning with LQR) 求解器

基于论文《HOP: Fast Differential Dynamic Programming for Horizon-Optimal Trajectory Planning》的 Algorithm 1
该方法能够高效求解时变线性系统的时优控制问题

核心贡献：
1. 将 Riccati 递推改写为线性分式变换(LFT)形式
2. 通过一次前向递归计算复合映射参数
3. 实现快速时域查询，复杂度从 O(N²n³) 降至 O(Nn³)
"""
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT)) 

import jax
jax.config.update("jax_enable_x64", True) 
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax import debug as jax_debug # 在文件顶部导入
from jax.lax import cond, scan
from jax import vmap, jit
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from hop_lib.utils import make_pd, get_logger

_logger = get_logger('hop')


@dataclass
class HOPLQRResult:
    """HOP-LQR 求解结果容器"""
    u_opt: jnp.ndarray      # 最优控制序列 (T_star, m)
    x_opt: jnp.ndarray      # 最优状态轨迹 (T_star, n)
    T_star: int             # 找到的最优时域
    J_values: jnp.ndarray   # 所有可能时域 t=1..N 对应的总成本
    optimal_cost: float     # 最优时域对应的总成本 (J_{T*} = 控制成本 + w*T*)


@dataclass
class HOPCompositeMaps:
    """存储复合映射参数的容器"""
    E: jnp.ndarray  # 形状 (N, n, n) - 单步映射参数
    F: jnp.ndarray  # 形状 (N, n, n)
    G: jnp.ndarray  # 形状 (N, n, n)
    bar_E: jnp.ndarray  # 形状 (N, n, n) - 复合映射参数
    bar_F: jnp.ndarray  # 形状 (N, n, n)
    bar_G: jnp.ndarray  # 形状 (N, n, n)


class HOPLQRSolver:
    """
    基于论文 Algorithm 1 的 HOP-LQR 求解器
    
    核心功能：高效求解 Horizon-Optimal Time-Varying LQR 问题
    问题定义：
        min_{U_T, T} J = ½x_T^T Q_T x_T + Σ_{k=0}^{T-1} ½(x_k^T Q_k x_k + u_k^T R_k u_k) + wT
        s.t. x_{k+1} = A_k x_k + B_k u_k, k=0,...,T-1
             T ∈ {1, 2, ..., N}
    
    算法优势：
    1. 将计算复杂度从 O(N²n³) 降低到 O(Nn³)
    2. 可处理时变系统矩阵
    3. 与暴力搜索基线得到相同的最优解
    """
    
    def __init__(self, n: int, m: int):
        """
        初始化 HOP-LQR 求解器
        
        参数:
            n: 状态维度
            m: 控制维度
            min_eig: Q/R 最小特征值下界，用于数值稳定
        """
        self.n = n
        self.m = m
        # 全局数值下界。_make_pd 和 Cholesky/solve 失败前会用到这些兜底项；
        # min_eig=0 时基本只做半正定投影，正数则会主动抬高最小特征值。
        self.min_eig: float = 1e-6
        self.chol_jitter: float = 1e-9

        # ---- 仅诊断：这些开关只打印证据，不修改 HOP-LQR 矩阵，
        # 也不改变用于选择 T* 的 J_values。----
        self.enable_detailed_diagnostics = False
        self.diagnostic_horizons: Tuple[int, ...] = tuple()
        self.diagnostic_max_listed = 8
        self.enable_query_cost_diagnostics = True
        self.enable_raw_efg_diagnostics = True
        self.raw_efg_diagnostic_horizons: Tuple[int, ...] = (1, 2, 3, 4, 8, 13, 20, 30, 40)
        self.enable_query_raw_vs_stabilized_diagnostics = True
        self.enable_raw_j_values_print = False
        self.enable_first_bad_p0_diagnostics = True
        self.enable_selected_horizon_matrix_diagnostics = True
        self.selected_horizon_diag_top_k = 5
        self.selected_horizon_diag_neighbor_radius = 1
        self.enable_summary_diagnostics = True
        self.enable_horizon_cost_breakdown_diagnostics = False
        self.enable_early_surrogate_gap_diagnostics = False
        self.enable_terminal_p0_norm_diagnostics = False
        self.terminal_p0_norm_diagnostic_horizons: Tuple[int, ...] = (1, 5, 30)
        self.enable_early_horizon_balance_diagnostics = False
        self.enable_recursive_growth_diagnostics = False
        self.enable_p0_direction_diagnostics = False
        self.enable_bar_f_matrix_diagnostics = False
        self.enable_prefix_invariance_diagnostics = False
        self.prefix_invariance_horizon = 12
        self.prefix_invariance_atol = 1e-8
        self.prefix_invariance_rtol = 1e-8

        # ---- 有效性判据阈值：用于标记/过滤病态 horizon。诊断总会计算这些量；
        # 只有 enable_horizon_validity_filter=True 时才会影响 T* 选择。----
        self.validity_eig_floor = 1e-10
        self.validity_rank_rtol = 1e-10

        # 复合映射递推中的分解保护。递推 bar_E/bar_F/bar_G 时需要分解
        # W = E_k + bar_G_prev；如果 W 接近奇异，这个下界会让 Cholesky
        # 分解可执行，避免某一步病态污染后续所有 horizon。
        self.enable_conditioned_factorization = True
        self.factorization_min_eig = 1e-12

        # 查询 J(t) 时的正定化保护。它会直接修改用于计算 J_values 的矩阵：
        # M = P_T + bar_G，以及 P0 = bar_E - F M^{-1} F^T。
        # 这个保护能让 J_values 更容易保持 finite/稳定，但也会改变原始 HOP 代理，
        # 所以做消融实验时需要明确记录是否开启。
        self.enable_query_stabilization = True
        self.query_min_eig_M = 1e-12
        self.query_min_eig_P0 = 1e-12

        # horizon 有效性过滤器。它不修复矩阵，只是把结构上可疑的 horizon
        # 的 selection cost 置为 inf，防止被选中；这是选择保护，不是数值修补。
        self.enable_horizon_validity_filter = True
        self.validity_filter_use_bad_m = True
        self.validity_filter_use_bad_p0 = True
        self.validity_filter_use_bad_temp = True
        self.validity_filter_use_rank_deficient_f = True
        self.validity_temp_rel_eig_max = 1.0
        self.min_candidate_horizon = 1

        # 递推阶段的 bar_F 奇异值下界。bar_F 可能在少数几次 LFT 复合后就数值秩亏；
        # 这里抬高过小奇异值，避免后续 P0/M 计算中的 rank collapse。
        self.enable_recursive_bar_f_svd_floor = True
        self.recursive_bar_f_svd_rtol = 1e-10
        self.recursive_bar_f_svd_atol = 1e-10

        # 递推阶段的 bar_F 增长上界。限制 bar_F 奇异值在相邻 horizon 间过快增长，
        # 从而减轻 Schur complement 中 F M^{-1} F^T 爆炸的问题。
        self.enable_recursive_bar_f_growth_cap = True
        self.recursive_bar_f_max_growth = 2.0

        # 早期 horizon 的归一化 bar_F 增益上界。P0 往往在前几次复合时最脆弱；
        # 这里只在早期 k 限制 E^{-1/2} bar_F W^{-1/2} 的增益。
        self.enable_recursive_early_bar_f_gain_cap = True
        self.recursive_early_bar_f_gain_cap = 1.02
        self.recursive_early_bar_f_gain_base_floor = 1e-10
        self.recursive_early_bar_f_gain_max_k = 4
        # 在 E/F/G 计算中对 Q 施加 Tikhonov 正则化，防止 Q_aug 因
        # q_tilde 耦合产生接近零的特征值传导至 E = Q^{-1} 后放大 F 的
        # 奇异值，最终导致复合映射递归中发生灾难性抵消。
        self.efg_q_reg: float = 0.02
        # 自适应 efg_q_reg：当增广 Q_k 的 Schur 补修正量大时（q_tilde 大，即
        # 局部状态偏差大），按比例增强 Tikhonov 正则化，防止 E=Q^{-1} 产生
        # 巨大特征值。修正量 = ||q_tilde||² / max(|c|, 1e-10)。
        # scale=0 退化为固定 reg。
        self.efg_q_reg_adaptive_scale = 0.25

        # 递推阶段的 bar_E/P0 稳定化。减去 recursive temp 后，bar_E 可能贴近
        # 半正定边界；这里在 bar_E_k 被后续 horizon 继续使用前抬高其特征值。
        self.enable_recursive_p0_stabilization = True
        self.recursive_bar_e_min_eig = 5e-4

        # 递推阶段的 bar_G 相对裁剪。bar_G 之后会进入 M = P_T + bar_G；
        # 将 bar_G 相对于 bar_E 的广义特征值限制在给定范围内，
        # 可以让 bar_G 与递推累计的状态度量保持兼容。
        self.enable_recursive_bar_g_relative_clip = True
        self.recursive_bar_g_min_relative_eig = -0.5
        self.recursive_bar_g_max_relative_eig = 1e4
        self.recursive_bar_g_base_floor = 1e-10

        # 查询阶段的 bar_G 相对裁剪。思路同上，但只在计算 J(t) 时启用，
        # 并且是相对于终端矩阵 P_T 做广义特征值裁剪。
        self.enable_query_bar_g_relative_clip = True
        self.query_bar_g_min_relative_eig = -0.5
        self.query_bar_g_max_relative_eig = 1e4
        self.query_bar_g_base_floor = 1e-10

        # 查询阶段的 bar_F 增益上界。开启后会限制
        # bar_E^{-1/2} bar_F M^{-1/2} 的奇异值，从而直接限制
        # temp = bar_F M^{-1} bar_F^T 对 bar_E 的“消耗”。
        self.enable_query_bar_f_gain_cap = False
        self.query_bar_f_gain_cap = 0.999
        self.query_bar_f_gain_base_floor = 1e-10

        # 递推阶段的 temp 裁剪，用于保护复合 Schur complement：
        # bar_E_k = bar_E_prev - temp_E。较宽松的上界可以防止 bar_E 接近奇异，
        # 同时不直接修改后续查询阶段的 J(t) 代价。
        self.enable_recursive_temp_relative_cap = True
        self.recursive_temp_relative_cap = 0.98
        self.recursive_temp_base_floor = 1e-10

        # 查询阶段的 temp 裁剪。开启后，在计算 J(t) 时会直接把
        # temp = bar_F M^{-1} bar_F^T 相对于 bar_E 的大小裁掉。
        # 这个保护很强，但会改变被查询的代理代价。
        self.enable_query_temp_relative_cap = False
        self.query_temp_relative_cap = 0.95
        self.query_temp_base_floor = 1e-10

        # 最近一次 solve/query 缓存下来的诊断结果。
        self.enable_composite_map_diagnostics = False
        self._debug_x0_for_query = None
        self.last_horizon_diagnostics = {}
        self.last_validity_info = {}
        self.last_J_values = None
        self.last_selection_costs = None
        self.last_selected_horizon = None
        
        # 编译关键计算函数
        self._compute_efg = jit(self._compute_efg_impl)
        self._compute_composite_maps = jit(self._compute_composite_maps_impl)
        self._fast_horizon_query = jit(self._fast_horizon_query_impl)

    def apply_diagnostics(self, diag) -> None:
        """从配置对象批量读取诊断/稳定化参数。传入对象需有与 QuadrotorConfig.Diagnostics 相同的属性名。"""
        self.enable_detailed_diagnostics = diag.enable_detailed_hop_lqr
        self.enable_summary_diagnostics = diag.enable_hop_lqr_summary_diagnostics
        self.enable_horizon_cost_breakdown_diagnostics = diag.enable_hop_lqr_horizon_cost_breakdown
        self.enable_early_horizon_balance_diagnostics = diag.enable_hop_lqr_early_horizon_balance
        self.enable_recursive_growth_diagnostics = diag.enable_hop_lqr_recursive_growth_diagnostics
        self.enable_p0_direction_diagnostics = diag.enable_hop_lqr_p0_direction_diagnostics
        self.enable_bar_f_matrix_diagnostics = diag.enable_hop_lqr_bar_f_matrix_diagnostics
        self.enable_prefix_invariance_diagnostics = diag.enable_hop_lqr_prefix_invariance_diagnostics
        self.prefix_invariance_horizon = diag.hop_lqr_prefix_invariance_horizon
        self.diagnostic_horizons = tuple(diag.detailed_horizons)
        self.enable_composite_map_diagnostics = diag.enable_hop_lqr_composite_map_diagnostics
        self.enable_single_step_map_diagnostics = diag.enable_hop_lqr_single_step_map_diagnostics
        # query stabilization
        self.enable_query_stabilization = diag.enable_hop_lqr_query_stabilization
        self.query_bar_e_min_eig = diag.hop_lqr_bar_e_min_eig
        self.query_temp_max_scale = diag.hop_lqr_temp_max_scale
        self.query_temp_min_cap = diag.hop_lqr_temp_min_cap
        self.query_p0_min_eig = diag.hop_lqr_p0_min_eig
        self.query_temp_relative_cap = diag.hop_lqr_query_temp_relative_cap
        self.enable_query_adaptive_temp_relative_cap = diag.enable_hop_lqr_query_adaptive_temp_relative_cap
        self.query_temp_relative_cap_min = diag.hop_lqr_query_temp_relative_cap_min
        self.query_temp_relative_cap_trigger = diag.hop_lqr_query_temp_relative_cap_trigger
        # validity filter
        self.enable_horizon_validity_filter = diag.enable_hop_lqr_validity_filter
        self.validity_filter_use_bad_m = diag.hop_lqr_validity_filter_use_bad_m
        self.validity_filter_use_bad_p0 = diag.hop_lqr_validity_filter_use_bad_p0
        self.validity_filter_use_rank_deficient_f = diag.hop_lqr_validity_filter_use_rank_deficient_f
        self.validity_filter_use_bad_temp = diag.hop_lqr_validity_filter_use_bad_temp
        self.validity_eig_floor = diag.hop_lqr_validity_eig_floor
        self.validity_rank_rtol = diag.hop_lqr_validity_rank_rtol
        self.min_candidate_horizon = diag.hop_lqr_min_candidate_horizon
        # recursive bar_F
        self.enable_recursive_bar_f_relative_cap = diag.enable_hop_lqr_recursive_bar_f_relative_cap
        self.recursive_bar_f_relative_cap = diag.hop_lqr_recursive_bar_f_relative_cap
        self.recursive_bar_f_max_growth = diag.hop_lqr_recursive_bar_f_max_growth
        self.enable_recursive_bar_f_svd_floor = diag.enable_hop_lqr_recursive_bar_f_svd_floor
        self.recursive_bar_f_svd_rtol = diag.hop_lqr_recursive_bar_f_svd_rtol
        self.recursive_bar_f_svd_atol = diag.hop_lqr_recursive_bar_f_svd_atol
        # recursive bar_G
        self.enable_recursive_bar_g_relative_clip = diag.enable_hop_lqr_recursive_bar_g_eig_clip
        self.recursive_bar_g_min_relative_eig = diag.hop_lqr_recursive_bar_g_min_relative_eig
        self.recursive_bar_g_max_relative_eig = diag.hop_lqr_recursive_bar_g_max_relative_eig
        # recursive P0 / temp
        self.enable_recursive_p0_stabilization = diag.enable_hop_lqr_recursive_p0_stabilization
        self.recursive_bar_e_min_eig = diag.hop_lqr_recursive_bar_e_min_eig
        self.recursive_temp_max_scale = diag.hop_lqr_recursive_temp_max_scale
        self.recursive_temp_min_cap = diag.hop_lqr_recursive_temp_min_cap
        self.recursive_temp_relative_cap = diag.hop_lqr_recursive_temp_relative_cap
        self.enable_recursive_adaptive_temp_relative_cap = diag.enable_hop_lqr_recursive_adaptive_temp_relative_cap
        self.recursive_temp_relative_cap_min = diag.hop_lqr_recursive_temp_relative_cap_min
        self.recursive_temp_relative_cap_trigger = diag.hop_lqr_recursive_temp_relative_cap_trigger
        # query bar_G / bar_F / temp
        self.enable_query_bar_g_relative_clip = diag.enable_hop_lqr_query_bar_g_eig_clip
        self.query_bar_g_min_relative_eig = diag.hop_lqr_query_bar_g_min_relative_eig
        self.query_bar_g_max_relative_eig = diag.hop_lqr_query_bar_g_max_relative_eig
        self.enable_query_temp_relative_cap = diag.enable_hop_lqr_query_temp_relative_cap
        self.enable_query_bar_f_gain_cap = diag.enable_hop_lqr_query_bar_f_gain_cap
        self.query_bar_f_gain_cap = diag.hop_lqr_query_bar_f_gain_cap

    def _terminal_matrix_np(self, P_T_np: np.ndarray, idx: int) -> np.ndarray:
        """Return the terminal matrix for one horizon from shared or per-horizon input."""
        return P_T_np[idx] if np.asarray(P_T_np).ndim == 3 else P_T_np

    def _bar_f_rank_stats_np(self, M: np.ndarray) -> tuple[float, float, int, bool]:
        """
        Unified NumPy rank diagnostic for bar_F/F-like matrices.

        We intentionally reuse the same tolerance logic everywhere so that
        summary diagnostics, validity filtering, and detailed matrix prints do
        not disagree just because they used slightly different thresholds.
        """
        arr = np.asarray(M)
        if arr.ndim != 2 or (not np.all(np.isfinite(arr))):
            return np.nan, np.nan, -1, True
        try:
            sv = np.linalg.svd(arr, compute_uv=False)
            max_sv = float(np.max(sv))
            min_sv = float(np.min(sv))
            sv_threshold = max(1e-12, float(self.validity_rank_rtol) * max_sv)
            rank = int(np.sum(sv > sv_threshold))
            full_rank = min(arr.shape[0], arr.shape[1])
            is_rank_deficient = rank < full_rank
            return min_sv, max_sv, rank, is_rank_deficient
        except Exception:
            return np.nan, np.nan, -1, True

    def _make_pd(self, M: jnp.ndarray, min_eig: Optional[float] = None) -> jnp.ndarray:
        """
        Make a symmetric matrix positive definite by eigenvalue floor.

        This enforces min eigenvalue >= min_eig (defaults to self.min_eig), which is required for
        inverting Q_k and R_k in HOP-LQR.
        """
        eig_floor = self.min_eig if min_eig is None else float(min_eig)
        M_sym = 0.5 * (M + jnp.transpose(M))
        eigvals = jnp.linalg.eigvalsh(M_sym)
        min_eig = jnp.min(eigvals)
        shift = jnp.maximum(0.0, eig_floor - min_eig)
        return M_sym + shift * jnp.eye(M_sym.shape[0])

    def _sanitize_traj_matrices(
        self, Q_traj: jnp.ndarray, R_traj: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Ensure Q_k and R_k are symmetric positive definite."""
        make_pd_vmap = vmap(self._make_pd)
        return make_pd_vmap(Q_traj), make_pd_vmap(R_traj)

    def _make_pd_np(self, M: np.ndarray, min_eig: float) -> np.ndarray:
        """NumPy version used only for human-readable diagnostics."""
        M_sym = 0.5 * (M + M.T)
        eigvals = np.linalg.eigvalsh(M_sym)
        shift = max(0.0, float(min_eig) - float(np.min(eigvals)))
        return M_sym + shift * np.eye(M_sym.shape[0], dtype=M_sym.dtype)

    def _clip_symmetric_relative_to_base_np(
        self,
        M: np.ndarray,
        base: np.ndarray,
        min_rel: float,
        max_rel: float,
        base_floor: float,
    ) -> np.ndarray:
        """NumPy wrapper matching the JAX relative clip used by query code."""
        clipped = self._clip_symmetric_relative_to_base(
            jnp.asarray(M),
            jnp.asarray(base),
            float(min_rel),
            float(max_rel),
            float(base_floor),
        )
        return np.asarray(clipped)

    def _query_cost_breakdown_np(
        self,
        x0_np: np.ndarray,
        P_T_t: np.ndarray,
        bar_E_t: np.ndarray,
        bar_F_t: np.ndarray,
        bar_G_t: np.ndarray,
        w: float,
        t: int,
    ) -> dict:
        """Compare strict raw, raw-with-P0-floor, and stabilized query costs."""
        result = {
            "raw_total": np.nan,
            "raw_floor_total": np.nan,
            "stabilized_total": np.nan,
            "raw_min_M": np.nan,
            "raw_min_P0": np.nan,
            "stabilized_min_M": np.nan,
            "stabilized_min_P0": np.nan,
            "stabilized_barF_gain_before": np.nan,
            "stabilized_barF_gain_after": np.nan,
        }
        try:
            M_raw = 0.5 * ((P_T_t + bar_G_t) + (P_T_t + bar_G_t).T)
            result["raw_min_M"] = float(np.min(np.linalg.eigvalsh(M_raw)))
            temp_raw = bar_F_t @ np.linalg.solve(M_raw, bar_F_t.T)
            P0_raw = 0.5 * ((bar_E_t - temp_raw) + (bar_E_t - temp_raw).T)
            result["raw_min_P0"] = float(np.min(np.linalg.eigvalsh(P0_raw)))

            y_raw = np.linalg.solve(P0_raw, x0_np)
            result["raw_total"] = float(0.5 * x0_np.T @ y_raw)

            P0_floor = self._make_pd_np(P0_raw, 1e-12)
            y_floor = np.linalg.solve(P0_floor, x0_np)
            result["raw_floor_total"] = float(0.5 * x0_np.T @ y_floor)
        except Exception:
            pass

        try:
            bar_G_stab = np.asarray(bar_G_t)
            if self.enable_query_bar_g_relative_clip:
                bar_G_stab = self._clip_symmetric_relative_to_base_np(
                    bar_G_stab,
                    P_T_t,
                    self.query_bar_g_min_relative_eig,
                    self.query_bar_g_max_relative_eig,
                    self.query_bar_g_base_floor,
                )
            M_stab = 0.5 * ((P_T_t + bar_G_stab) + (P_T_t + bar_G_stab).T)
            if self.enable_query_stabilization:
                M_stab = self._make_pd_np(M_stab, self.query_min_eig_M)
            result["stabilized_min_M"] = float(np.min(np.linalg.eigvalsh(M_stab)))

            bar_F_stab = np.asarray(bar_F_t)
            result["stabilized_barF_gain_before"] = self._bar_f_relative_gain_np(
                bar_F_stab,
                bar_E_t,
                M_stab,
                self.query_bar_f_gain_base_floor,
            )
            if self.enable_query_bar_f_gain_cap:
                bar_F_stab = self._clip_bar_f_gain_relative_np(
                    bar_F_stab,
                    bar_E_t,
                    M_stab,
                    self.query_bar_f_gain_cap,
                    self.query_bar_f_gain_base_floor,
                )
            result["stabilized_barF_gain_after"] = self._bar_f_relative_gain_np(
                bar_F_stab,
                bar_E_t,
                M_stab,
                self.query_bar_f_gain_base_floor,
            )

            temp_stab = bar_F_stab @ np.linalg.solve(M_stab, bar_F_stab.T)
            temp_stab = 0.5 * (temp_stab + temp_stab.T)
            if self.enable_query_temp_relative_cap:
                temp_stab = self._clip_symmetric_relative_to_base_np(
                    temp_stab,
                    bar_E_t,
                    0.0,
                    self.query_temp_relative_cap,
                    self.query_temp_base_floor,
                )

            P0_stab = 0.5 * ((bar_E_t - temp_stab) + (bar_E_t - temp_stab).T)
            if self.enable_query_stabilization:
                P0_stab = self._make_pd_np(P0_stab, self.query_min_eig_P0)
            result["stabilized_min_P0"] = float(np.min(np.linalg.eigvalsh(P0_stab)))

            y_stab = np.linalg.solve(P0_stab, x0_np)
            result["stabilized_total"] = float(0.5 * x0_np.T @ y_stab)
        except Exception:
            pass

        return result

    def _bar_f_relative_gain_np(
        self,
        F: np.ndarray,
        E_base: np.ndarray,
        M_base: np.ndarray,
        base_floor: float,
    ) -> float:
        """Largest singular value of E^{-1/2} F M^{-1/2}."""
        try:
            E_pd = self._make_pd_np(E_base, base_floor)
            M_pd = self._make_pd_np(M_base, base_floor)
            L_E = np.linalg.cholesky(E_pd)
            L_M = np.linalg.cholesky(M_pd)
            normalized = np.linalg.solve(L_E, F)
            normalized = np.linalg.solve(L_M, normalized.T).T
            sv = np.linalg.svd(normalized, compute_uv=False)
            return float(np.max(sv))
        except Exception:
            return np.nan

    def _clip_bar_f_gain_relative_np(
        self,
        F: np.ndarray,
        E_base: np.ndarray,
        M_base: np.ndarray,
        max_gain: float,
        base_floor: float,
    ) -> np.ndarray:
        """NumPy version of the query-time bar_F relative-gain cap."""
        try:
            E_pd = self._make_pd_np(E_base, base_floor)
            M_pd = self._make_pd_np(M_base, base_floor)
            L_E = np.linalg.cholesky(E_pd)
            L_M = np.linalg.cholesky(M_pd)
            normalized = np.linalg.solve(L_E, F)
            normalized = np.linalg.solve(L_M, normalized.T).T
            U, s, Vh = np.linalg.svd(normalized, full_matrices=False)
            s_clipped = np.minimum(s, float(max_gain))
            normalized_clipped = (U * s_clipped) @ Vh
            return L_E @ normalized_clipped @ L_M.T
        except Exception:
            return np.asarray(F)

    def _stabilize_bar_f(self, F_new: jnp.ndarray, F_prev: jnp.ndarray) -> jnp.ndarray:
        """Keep the composite F map full-rank without allowing explosive growth."""
        if not (self.enable_recursive_bar_f_svd_floor or self.enable_recursive_bar_f_growth_cap):
            return F_new

        U, s, Vh = jnp.linalg.svd(F_new, full_matrices=False)
        s_stable = s

        if self.enable_recursive_bar_f_growth_cap:
            prev_s = jnp.linalg.svd(F_prev, compute_uv=False)
            prev_max = jnp.max(prev_s)
            max_allowed = jnp.maximum(
                self.recursive_bar_f_svd_atol,
                float(self.recursive_bar_f_max_growth) * prev_max,
            )
            s_stable = jnp.minimum(s_stable, max_allowed)

        if self.enable_recursive_bar_f_svd_floor:
            floor = jnp.maximum(
                self.recursive_bar_f_svd_atol,
                float(self.recursive_bar_f_svd_rtol) * jnp.max(s_stable),
            )
            s_stable = jnp.maximum(s_stable, floor)

        return (U * s_stable) @ Vh

    def _clip_symmetric_relative_to_base(
        self,
        M: jnp.ndarray,
        base: jnp.ndarray,
        min_rel: float,
        max_rel: float,
        base_floor: float,
    ) -> jnp.ndarray:
        """
        Clip generalized eigenvalues of M relative to a positive base matrix.

        If base = L L^T, this clips eig(L^{-1} M L^{-T}) and maps it back.
        It is a targeted guard for keeping bar_G compatible with the matrix
        it will later be added to.
        """
        M_sym = 0.5 * (M + M.T)
        base_pd = self._make_pd(base, min_eig=base_floor)
        L = jnp.linalg.cholesky(base_pd)
        left = jnp.linalg.solve(L, M_sym)
        whitened = jnp.linalg.solve(L, left.T).T
        whitened = 0.5 * (whitened + whitened.T)
        eigvals, eigvecs = jnp.linalg.eigh(whitened)
        eigvals_clipped = jnp.clip(eigvals, float(min_rel), float(max_rel))
        whitened_clipped = (eigvecs * eigvals_clipped) @ eigvecs.T
        clipped = L @ whitened_clipped @ L.T
        return 0.5 * (clipped + clipped.T)

    def _clip_bar_f_gain_relative(
        self,
        F: jnp.ndarray,
        E_base: jnp.ndarray,
        M_base: jnp.ndarray,
        max_gain: float,
        base_floor: float,
    ) -> jnp.ndarray:
        """
        Limit singular values of E^{-1/2} F M^{-1/2}.

        In the inverse-LFT query, temp = F M^{-1} F^T.  If the largest
        normalized singular value reaches 1, temp can consume bar_E and make
        P0 = bar_E - temp non-positive definite.
        """
        E_pd = self._make_pd(E_base, min_eig=base_floor)
        M_pd = self._make_pd(M_base, min_eig=base_floor)
        L_E = jnp.linalg.cholesky(E_pd)
        L_M = jnp.linalg.cholesky(M_pd)
        normalized = jnp.linalg.solve(L_E, F)
        normalized = jnp.linalg.solve(L_M, normalized.T).T
        U, s, Vh = jnp.linalg.svd(normalized, full_matrices=False)
        s_clipped = jnp.minimum(s, float(max_gain))
        normalized_clipped = (U * s_clipped) @ Vh
        return L_E @ normalized_clipped @ L_M.T

    def _max_relative_eig_np(
        self,
        M: np.ndarray,
        base: np.ndarray,
        base_floor: float = 1e-12,
    ) -> float:
        """Largest generalized eigenvalue of M relative to a positive base."""
        M_sym = 0.5 * (M + M.T)
        base_pd = self._make_pd_np(base, base_floor)
        L = np.linalg.cholesky(base_pd)
        left = np.linalg.solve(L, M_sym)
        whitened = np.linalg.solve(L, left.T).T
        whitened = 0.5 * (whitened + whitened.T)
        return float(np.max(np.linalg.eigvalsh(whitened)))

    def _evaluate_horizon_validity(
        self,
        P_T: np.ndarray,
        composite_maps: HOPCompositeMaps,
        eig_floor: Optional[float] = None,
    ) -> dict:
        """Evaluate which horizons remain structurally usable."""
        eig_floor_val = self.validity_eig_floor if eig_floor is None else float(eig_floor)
        P_T_np = np.asarray(P_T)
        bar_E = np.asarray(composite_maps.bar_E)
        bar_F = np.asarray(composite_maps.bar_F)
        bar_G = np.asarray(composite_maps.bar_G)

        bad_M: list[int] = []
        bad_bar_F: list[int] = []
        bad_P0: list[int] = []
        bad_temp: list[int] = []
        valid_mask = np.ones(bar_E.shape[0], dtype=bool)

        for idx in range(bar_E.shape[0]):
            t = idx + 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)

            min_M = np.nan
            try:
                min_M = float(np.min(np.linalg.eigvalsh(M_t)))
            except Exception:
                pass
            is_bad_M = (not np.isfinite(min_M)) or (min_M <= eig_floor_val)
            if is_bad_M:
                bad_M.append(t)

            _, _, rank_F, rank_deficient = self._bar_f_rank_stats_np(bar_F[idx])
            is_bad_F = rank_F < 0 or rank_deficient
            if is_bad_F:
                bad_bar_F.append(t)

            min_P0 = np.nan
            max_temp_rel = np.nan
            try:
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                temp = 0.5 * (temp + temp.T)
                max_temp_rel = self._max_relative_eig_np(
                    temp,
                    0.5 * (bar_E[idx] + bar_E[idx].T),
                    base_floor=self.query_temp_base_floor,
                )
                P0_t = 0.5 * ((bar_E[idx] - temp) + (bar_E[idx] - temp).T)
                min_P0 = float(np.min(np.linalg.eigvalsh(P0_t)))
            except Exception:
                pass
            is_bad_temp = (
                (not np.isfinite(max_temp_rel))
                or (max_temp_rel >= float(self.validity_temp_rel_eig_max))
            )
            if is_bad_temp:
                bad_temp.append(t)
            is_bad_P0 = (not np.isfinite(min_P0)) or (min_P0 <= eig_floor_val)
            if is_bad_P0:
                bad_P0.append(t)

            keep = True
            if self.enable_horizon_validity_filter:
                if self.validity_filter_use_bad_m and is_bad_M:
                    keep = False
                if self.validity_filter_use_rank_deficient_f and is_bad_F:
                    keep = False
                if self.validity_filter_use_bad_p0 and is_bad_P0:
                    keep = False
                if self.validity_filter_use_bad_temp and is_bad_temp:
                    keep = False
                if t < int(self.min_candidate_horizon):
                    keep = False
            valid_mask[idx] = keep

        valid_horizons = (np.flatnonzero(valid_mask) + 1).tolist()
        return {
            "valid_mask": valid_mask,
            "valid_horizons": valid_horizons,
            "bad_M": bad_M,
            "bad_bar_F": bad_bar_F,
            "bad_P0": bad_P0,
            "bad_temp": bad_temp,
        }
    
    def _compute_efg_impl(self, Q_k: jnp.ndarray, A_k: jnp.ndarray, B_k: jnp.ndarray, R_k: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        计算单步 LFT 参数 (E_k, F_k, G_k) - 对应论文公式(6-7)

        参数:
            Q_k: 状态成本矩阵
            A_k: 状态矩阵
            B_k: 控制矩阵
            R_k: 控制成本矩阵

        返回:
            (E_k, F_k, G_k): 单步 LFT 参数
        """
        A_k_T = jnp.transpose(A_k)

        I_n = jnp.eye(self.n, dtype=Q_k.dtype)
        # 自适应正则化：从增广 Q_k 提取 Schur 补修正量
        q_tilde = Q_k[:-1, -1]
        c = Q_k[-1, -1]
        correction = jnp.sum(q_tilde * q_tilde) / jnp.maximum(jnp.abs(c), 1e-10)
        adaptive_reg = self.efg_q_reg + self.efg_q_reg_adaptive_scale * correction
        Q_k_reg = Q_k + (adaptive_reg + self.chol_jitter) * I_n
        E_k = jnp.linalg.solve(Q_k_reg, I_n)
        F_k = jnp.linalg.solve(Q_k_reg, A_k_T)

        I_m = jnp.eye(self.m, dtype=R_k.dtype)
        R_inv_Bt = jnp.linalg.solve(R_k + self.chol_jitter * I_m, jnp.transpose(B_k))
        G_k = A_k @ F_k + B_k @ R_inv_Bt

        return E_k, F_k, G_k
    
    def _compute_composite_maps_impl(self, E: jnp.ndarray, F: jnp.ndarray, G: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        计算复合映射参数 (bar_E_k, bar_F_k, bar_G_k) - 对应论文公式(13-14)
        
        参数:
            E, F, G: 单步映射参数序列，形状均为 (N, n, n)
        
        返回:
            (bar_E, bar_F, bar_G): 复合映射参数序列
        """
        N = len(E)
        def forward_step(carry, k):
            bar_E_prev, bar_F_prev, bar_G_prev = carry
            E_k, F_k, G_k = E[k], F[k], G[k]
            bar_E_prev = 0.5 * (bar_E_prev + bar_E_prev.T)
            bar_G_prev = 0.5 * (bar_G_prev + bar_G_prev.T)
            W = 0.5 * ((E_k + bar_G_prev) + (E_k + bar_G_prev).T)
            if self.enable_conditioned_factorization:
                W = self._make_pd(W, min_eig=self.factorization_min_eig)
            else:
                W = W + self.chol_jitter * jnp.eye(self.n, dtype=W.dtype)
            L = jnp.linalg.cholesky(W)
            Y1 = jnp.linalg.solve(L, bar_F_prev.T)
            Z1 = jnp.linalg.solve(L.T, Y1)
            Y2 = jnp.linalg.solve(L, F_k)
            Z2 = jnp.linalg.solve(L.T, Y2)
            temp_E = 0.5 * ((bar_F_prev @ Z1) + (bar_F_prev @ Z1).T)
            if self.enable_recursive_temp_relative_cap:
                temp_E = self._clip_symmetric_relative_to_base(
                    temp_E,
                    bar_E_prev,
                    0.0,
                    self.recursive_temp_relative_cap,
                    self.recursive_temp_base_floor,
                )
            bar_E_k = bar_E_prev - temp_E
            bar_F_k = bar_F_prev @ Z2
            if self.enable_recursive_early_bar_f_gain_cap:
                bar_F_k = cond(
                    k <= int(self.recursive_early_bar_f_gain_max_k),
                    lambda F_new: self._clip_bar_f_gain_relative(
                        F_new,
                        bar_E_k,
                        W,
                        self.recursive_early_bar_f_gain_cap,
                        self.recursive_early_bar_f_gain_base_floor,
                    ),
                    lambda F_new: F_new,
                    bar_F_k,
                )
            bar_G_k = G_k - F_k.T @ Z2
            bar_E_k = 0.5 * (bar_E_k + bar_E_k.T)
            if getattr(self, 'enable_recursive_p0_stabilization', False):
                bar_E_k = self._make_pd(bar_E_k, min_eig=float(self.recursive_bar_e_min_eig))
            bar_F_k = self._stabilize_bar_f(bar_F_k, bar_F_prev)
            bar_G_k = 0.5 * (bar_G_k + bar_G_k.T)
            if self.enable_recursive_bar_g_relative_clip:
                bar_G_k = self._clip_symmetric_relative_to_base(
                    bar_G_k,
                    bar_E_k,
                    self.recursive_bar_g_min_relative_eig,
                    self.recursive_bar_g_max_relative_eig,
                    self.recursive_bar_g_base_floor,
                )
            return (bar_E_k, bar_F_k, bar_G_k), (bar_E_k, bar_F_k, bar_G_k)
        
        # 初始化: k=0 时，复合映射就是自身
        init_bar = (E[0], F[0], G[0])
        
        # 对 k=1 到 N-1 进行递归计算
        _, (bar_E_all, bar_F_all, bar_G_all) = scan(
            forward_step, 
            init_bar, 
            jnp.arange(1, N)
        )
        
        # 将初始参数 (k=0) 拼接到结果序列开头
        bar_E = jnp.concatenate([E[0:1], bar_E_all])
        bar_F = jnp.concatenate([F[0:1], bar_F_all])
        bar_G = jnp.concatenate([G[0:1], bar_G_all])
        
        return bar_E, bar_F, bar_G
    
    def _fast_horizon_query_impl(self, x0: jnp.ndarray, P_T_tilde: jnp.ndarray, w: float, 
                                 bar_E: jnp.ndarray, bar_F: jnp.ndarray, bar_G: jnp.ndarray) -> jnp.ndarray:
        """
        快速时域查询 - 对应论文公式(19-20)
        参数:
            x0: 初始状态
            P_T: 终端代价矩阵
            w: 时域惩罚权重
            bar_E, bar_F, bar_G: 复合映射参数
        
        返回:
            J_values: 所有时域 t=1..N 对应的总成本
        """
        N = len(bar_E)
        def query_single_horizon(t: int) -> float:
            bar_E_t = bar_E[t-1]
            bar_F_t = bar_F[t-1]
            bar_G_t = bar_G[t-1]
            P_T_t = P_T_tilde[t-1] if P_T_tilde.ndim == 3 else P_T_tilde
            if self.enable_query_bar_g_relative_clip:
                bar_G_t = self._clip_symmetric_relative_to_base(
                    bar_G_t,
                    P_T_t,
                    self.query_bar_g_min_relative_eig,
                    self.query_bar_g_max_relative_eig,
                    self.query_bar_g_base_floor,
                )
            M_t = 0.5 * ((P_T_t + bar_G_t) + (P_T_t + bar_G_t).T)
            if self.enable_query_stabilization:
                M_t = self._make_pd(M_t, min_eig=self.query_min_eig_M)
            else:
                M_t = M_t + self.chol_jitter * jnp.eye(self.n, dtype=M_t.dtype)
            if self.enable_query_bar_f_gain_cap:
                bar_F_t = self._clip_bar_f_gain_relative(
                    bar_F_t,
                    bar_E_t,
                    M_t,
                    self.query_bar_f_gain_cap,
                    self.query_bar_f_gain_base_floor,
                )
            L = jnp.linalg.cholesky(M_t)
            sol = jnp.linalg.solve(L.T, jnp.linalg.solve(L, bar_F_t.T))
            temp = 0.5 * ((bar_F_t @ sol) + (bar_F_t @ sol).T)
            if self.enable_query_temp_relative_cap:
                temp = self._clip_symmetric_relative_to_base(
                    temp,
                    bar_E_t,
                    0.0,
                    self.query_temp_relative_cap,
                    self.query_temp_base_floor,
                )
            P0_inv_inner = 0.5 * ((bar_E_t - temp) + (bar_E_t - temp).T)
            if self.enable_query_stabilization:
                P0_inv_inner = self._make_pd(P0_inv_inner, min_eig=self.query_min_eig_P0)
            else:
                P0_inv_inner = P0_inv_inner + self.chol_jitter * jnp.eye(self.n, dtype=P0_inv_inner.dtype)
            L2 = jnp.linalg.cholesky(P0_inv_inner)
            y = jnp.linalg.solve(L2.T, jnp.linalg.solve(L2, x0))
            cost_quad = 0.5 * jnp.dot(x0, y)
            return cost_quad
        horizons = jnp.arange(1, N+1)
        J_values = vmap(query_single_horizon)(horizons)
        if self.enable_raw_j_values_print:
            _logger.info(f"  raw HOP-LQR J_values: {J_values}")
        return J_values
    
    def precompute_composite_maps(self, A_traj: jnp.ndarray, B_traj: jnp.ndarray, 
                                  Q_traj: jnp.ndarray, R_traj: jnp.ndarray,
                                  sanitize: bool = True) -> HOPCompositeMaps:
        """
        预计算复合映射参数 - Algorithm 1 第一阶段
        
        参数:
            A_traj: 状态矩阵序列，形状 (N, n, n)
            B_traj: 控制矩阵序列，形状 (N, n, m)
            Q_traj: 状态权重矩阵序列，形状 (N, n, n)
            R_traj: 控制权重矩阵序列，形状 (N, m, m)
        
        返回:
            HOPCompositeMaps: 包含所有复合映射参数的容器
        """
        N = len(A_traj)
        
        # 保证 Q/R 正定再求逆
        if sanitize:
            Q_traj_pd, R_traj_pd = self._sanitize_traj_matrices(Q_traj, R_traj)
        else:
            Q_traj_pd, R_traj_pd = Q_traj, R_traj

        # 计算单步映射参数 (E_k, F_k, G_k)
        compute_efg_vmap = vmap(self._compute_efg, in_axes=(0, 0, 0, 0))
        E, F, G = compute_efg_vmap(Q_traj_pd, A_traj, B_traj, R_traj_pd)
        if self.enable_raw_efg_diagnostics:
            self._print_raw_efg_diagnostics(E, F, G)
        # 计算复合映射参数 (bar_E_k, bar_F_k, bar_G_k)
        bar_E, bar_F, bar_G = self._compute_composite_maps(E, F, G)

        composite = HOPCompositeMaps(E=E, F=F, G=G, bar_E=bar_E, bar_F=bar_F, bar_G=bar_G)
        if self.enable_composite_map_diagnostics:
            self._print_composite_map_diagnostics(composite)
        
        return composite

    def _print_prefix_invariance_diagnostics(
        self,
        x0: jnp.ndarray,
        A_traj: jnp.ndarray,
        B_traj: jnp.ndarray,
        Q_traj: jnp.ndarray,
        R_traj: jnp.ndarray,
        Q_T: jnp.ndarray,
        w: float,
        full_J_values: jnp.ndarray,
    ) -> None:
        """
        Verify that early horizon costs are invariant to later suffix data.

        In HOP-LQR, J(t) should depend only on the prefix 0..t-1 plus the shared
        terminal surrogate. If recomputing the composite maps on a truncated
        prefix changes J(1..k), a nonlocal implementation side effect is present.
        """
        N = int(A_traj.shape[0])
        prefix_len = min(max(1, int(self.prefix_invariance_horizon)), N)
        if prefix_len >= N:
            return

        prefix_composite = self.precompute_composite_maps(
            A_traj[:prefix_len],
            B_traj[:prefix_len],
            Q_traj[:prefix_len],
            R_traj[:prefix_len],
            sanitize=False,
        )
        prefix_J_values = self._fast_horizon_query(
            x0,
            Q_T,
            w,
            prefix_composite.bar_E,
            prefix_composite.bar_F,
            prefix_composite.bar_G,
        )

        full_prefix = np.asarray(full_J_values[:prefix_len])
        truncated_prefix = np.asarray(prefix_J_values)
        diff = truncated_prefix - full_prefix
        finite = np.isfinite(full_prefix) & np.isfinite(truncated_prefix)
        if np.any(finite):
            abs_diff = np.abs(diff[finite])
            max_abs = float(np.max(abs_diff))
            max_rel = float(
                np.max(abs_diff / np.maximum(1.0, np.abs(full_prefix[finite])))
            )
        else:
            max_abs = np.inf
            max_rel = np.inf

        passed = bool(
            np.allclose(
                full_prefix,
                truncated_prefix,
                atol=self.prefix_invariance_atol,
                rtol=self.prefix_invariance_rtol,
                equal_nan=True,
            )
        )
        status = "PASS" if passed else "FAIL"
        _logger.info(
            "[diag] HOP-LQR prefix invariance:"
            f" status={status}, prefix={prefix_len},"
            f" max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}"
        )

        if not passed:
            rows = []
            for idx in range(min(prefix_len, self.diagnostic_max_listed)):
                rows.append(
                    f"t={idx + 1}: full={full_prefix[idx]:.6e},"
                    f" prefix={truncated_prefix[idx]:.6e},"
                    f" diff={diff[idx]:.3e}"
                )
            _logger.info("[diag] HOP-LQR prefix invariance detail: " + "; ".join(rows))
    
    def find_optimal_horizon(self, x0: jnp.ndarray, P_T_tilde: jnp.ndarray, w: float,
                            composite_maps: HOPCompositeMaps,
                            T_min: int = 1, T_max: int = 300) -> Tuple[int, jnp.ndarray]:
        """
        找到最优时域 - Algorithm 1 第二阶段
        参数:
            x0: 初始状态
            P_T: 终端代价矩阵
            w: 时域惩罚权重
            composite_maps: 预计算的复合映射参数
            T_min: 最小时域 (inclusive, 1-indexed)
            T_max: 最大时域 (inclusive, 1-indexed)

        返回:
            T_star: 最优时域 (1-indexed)
            J_values: 所有时域对应的总成本
        """
        # 兼容 shared terminal matrix (n,n) 与 per-horizon terminal stack (N,n,n)。
        # 对终端代价矩阵进行正定化处理，确保数值稳定性
        P_T_tilde = vmap(self._make_pd)(P_T_tilde) if P_T_tilde.ndim == 3 else self._make_pd(P_T_tilde)
        
        # 缓存当前查询的初始状态，用于后续调试
        self._debug_x0_for_query = np.asarray(x0)
        
        # 快速时域查询 (传入原始 P_T_pd)
        # 通过预计算的复合映射快速评估所有时域的成本
        J_values = self._fast_horizon_query(
            x0, P_T_tilde, w, 
            composite_maps.bar_E, composite_maps.bar_F, composite_maps.bar_G
        )
        
        # 根据配置标志决定是否打印诊断信息
        if (
            self.enable_summary_diagnostics
            or self.enable_horizon_cost_breakdown_diagnostics
            or self.enable_detailed_diagnostics
        ):
            self._print_horizon_diagnostics(P_T_tilde, composite_maps, J_values, w=w)
        # 数值保护：将非有限值（NaN/Inf）替换为无穷大
        # 这样可以避免 argmin 被非有限值污染，但仍保留原始值用于诊断
        J_values_safe = jnp.where(jnp.isfinite(J_values), J_values, jnp.inf)
        # 转换为 numpy 数组进行后续处理
        J_values_safe_np = np.array(J_values_safe, copy=True)
        # 评估各时域的有效性（检查矩阵是否病态、秩亏缺等问题）
        self.last_validity_info = self._evaluate_horizon_validity(P_T_tilde, composite_maps)
        # 应用时域有效性过滤器（如果启用）
        if self.enable_horizon_validity_filter:
            # 获取有效性掩码，标记哪些时域是"可信"的
            valid_mask = np.asarray(self.last_validity_info.get("valid_mask", []), dtype=bool)
            # 检查掩码尺寸是否与 J_values 匹配
            if valid_mask.size == J_values_safe_np.size:
                # 将无效时域的成本设为无穷大，确保不会被选为最优时域
                # 这样可以在数值上排除那些因矩阵病态而不可信的时域
                J_values_safe_np = np.where(valid_mask, J_values_safe_np, np.inf)
        # 选择最优时域，限制在 [T_min, T_max] 范围内
        # 与参考实现一致：短时域因信息不足可能产生垃圾预测值
        T_min_clamped = max(1, min(T_min, len(J_values_safe_np)))
        T_max_clamped = max(T_min_clamped, min(T_max, len(J_values_safe_np)))
        J_search = J_values_safe_np[T_min_clamped - 1 : T_max_clamped]
        if bool(np.all(np.isinf(J_search))):
            T_star = T_min_clamped
        else:
            T_star = int(np.argmin(J_search) + T_min_clamped)
        
        # 保存历史数据，供后续分析使用
        self.last_J_values = np.asarray(J_values)  # 原始 J_values（包含 NaN/Inf）
        self.last_selection_costs = np.array(J_values_safe_np, copy=True)  # 实际用于选择的成本
        self.last_selected_horizon = T_star  # 最终选定的时域
        
        # 打印选定时域的详细矩阵诊断信息
        self._print_selected_horizon_matrix_diagnostics(
            P_T=P_T_tilde,
            composite_maps=composite_maps,
            J_values=J_values,
            selection_costs=J_values_safe_np,
            selected_T=T_star,
            w=w,
            eig_floor=self.validity_eig_floor,
        )
        
        # 返回最优时域和所有时域的成本值
        return T_star, J_values

    def _print_horizon_diagnostics(
        self,
        P_T: jnp.ndarray,
        composite_maps: HOPCompositeMaps,
        J_values: jnp.ndarray,
        w: float,
        eig_floor: float = 1e-10,
        max_listed: int = 8,
    ) -> None:
        """
        诊断 HOP-LQR 时域查询的数值质量。
        只打印统计信息，不改变求解行为。
        """
        P_T_np = np.asarray(P_T)
        bar_E = np.asarray(composite_maps.bar_E)
        bar_F = np.asarray(composite_maps.bar_F)
        bar_G = np.asarray(composite_maps.bar_G)
        J_np = np.asarray(J_values)

        min_eig_M = []
        cond_M = []
        rank_bar_F = []
        min_sv_bar_F = []
        max_sv_bar_F = []
        min_eig_P0 = []
        max_rel_temp = []

        bad_M = []
        bad_bar_F = []
        bad_P0 = []
        bad_temp = []

        for idx in range(bar_E.shape[0]):
            t = idx + 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)
            try:
                eigs_M = np.linalg.eigvalsh(M_t)
                min_M = float(np.min(eigs_M))
            except Exception:
                min_M = np.nan
            try:
                cond_M_t = float(np.linalg.cond(M_t))
            except Exception:
                cond_M_t = np.inf

            min_eig_M.append(min_M)
            cond_M.append(cond_M_t)
            if (not np.isfinite(min_M)) or (min_M <= eig_floor):
                bad_M.append(t)

            try:
                min_sv, max_sv, rank_F, rank_deficient = self._bar_f_rank_stats_np(bar_F[idx])
            except Exception:
                max_sv = np.nan
                min_sv = np.nan
                rank_F = -1
                rank_deficient = True
            max_sv_bar_F.append(max_sv)
            min_sv_bar_F.append(min_sv)
            rank_bar_F.append(rank_F)
            if rank_F >= 0 and rank_deficient:
                bad_bar_F.append(t)

            try:
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                temp = 0.5 * (temp + temp.T)
                rel_temp = self._max_relative_eig_np(
                    temp,
                    0.5 * (bar_E[idx] + bar_E[idx].T),
                    base_floor=self.query_temp_base_floor,
                )
                P0_t = 0.5 * ((bar_E[idx] - temp) + (bar_E[idx] - temp).T)
                min_P0 = float(np.min(np.linalg.eigvalsh(P0_t)))
            except Exception:
                min_P0 = np.nan
                rel_temp = np.nan
            max_rel_temp.append(rel_temp)
            if (not np.isfinite(rel_temp)) or (rel_temp >= float(self.validity_temp_rel_eig_max)):
                bad_temp.append(t)
            min_eig_P0.append(min_P0)
            if (not np.isfinite(min_P0)) or (min_P0 <= eig_floor):
                bad_P0.append(t)

        finite_J = int(np.isfinite(J_np).sum())
        total_J = int(J_np.size)
        min_M_val = float(np.nanmin(min_eig_M)) if min_eig_M else np.nan
        worst_cond_M = float(np.nanmax(cond_M)) if cond_M else np.nan
        min_P0_val = float(np.nanmin(min_eig_P0)) if min_eig_P0 else np.nan
        max_rel_temp_val = float(np.nanmax(max_rel_temp)) if max_rel_temp else np.nan
        min_rank_F = int(min([r for r in rank_bar_F if r >= 0], default=-1))
        min_sv_val = float(np.nanmin(min_sv_bar_F)) if min_sv_bar_F else np.nan
        max_sv_val = float(np.nanmax(max_sv_bar_F)) if max_sv_bar_F else np.nan

        _logger.info(
            "[diag] HOP-LQR summary:"
            f" finite_J={finite_J}/{total_J},"
            f" bad_M={len(bad_M)},"
            f" rank_drop_F={len(bad_bar_F)},"
            f" bad_P0={len(bad_P0)},"
            f" bad_temp={len(bad_temp)},"
            f" minEig(M)={min_M_val:.3e},"
            f" worstCond(M)={worst_cond_M:.3e},"
            f" minRank(F)={min_rank_F},"
            f" sv(F)=[{min_sv_val:.3e}, {max_sv_val:.3e}],"
            f" minEig(P0)={min_P0_val:.3e},"
            f" maxRelEig(temp|bar_E)={max_rel_temp_val:.3e}"
        )

        if bad_M:
            _logger.info(f"[diag] HOP-LQR first bad M horizons: {bad_M[:max_listed]}")
        if bad_bar_F:
            _logger.info(f"[diag] HOP-LQR first rank-deficient F horizons: {bad_bar_F[:max_listed]}")
        if bad_P0:
            _logger.info(f"[diag] HOP-LQR first bad P0 horizons: {bad_P0[:max_listed]}")
        if bad_temp:
            _logger.info(f"[diag] HOP-LQR first temp-dominant horizons: {bad_temp[:max_listed]}")

        first_bad_chain = []
        if bad_P0:
            first_bad_chain.append(f"P0@{bad_P0[0]}")
        if bad_temp:
            first_bad_chain.append(f"temp@{bad_temp[0]}")
        if bad_bar_F:
            first_bad_chain.append(f"F@{bad_bar_F[0]}")
        if bad_M:
            first_bad_chain.append(f"M@{bad_M[0]}")
        if first_bad_chain:
            _logger.info(f"[diag] HOP-LQR first structural breaks: {' -> '.join(first_bad_chain)}")

        if self.enable_first_bad_p0_diagnostics:
            self._print_first_bad_p0_diagnostics(
                P_T_np=P_T_np,
                bar_E=bar_E,
                bar_F=bar_F,
                bar_G=bar_G,
                bad_P0=bad_P0,
            )

        if self.enable_early_surrogate_gap_diagnostics:
            self._print_early_surrogate_gap_diagnostics(
                P_T_np=P_T_np,
                bar_E=bar_E,
                bar_F=bar_F,
                bar_G=bar_G,
                J_np=J_np,
                w=w,
            )

        if self.enable_terminal_p0_norm_diagnostics:
            self._print_terminal_p0_norm_diagnostics(
                P_T_np=P_T_np,
                bar_E=bar_E,
                bar_F=bar_F,
                bar_G=bar_G,
                J_np=J_np,
            )

        nan_J_idx = np.flatnonzero(~np.isfinite(J_np))
        if nan_J_idx.size > 0:
            horizons = (nan_J_idx[:max_listed] + 1).tolist()
            _logger.info(f"[diag] HOP-LQR first nonfinite J horizons: {horizons}")

        if self.enable_horizon_cost_breakdown_diagnostics:
            finite_mask = np.isfinite(J_np)
            top_horizons = []
            if np.any(finite_mask):
                ranked = np.argsort(np.where(finite_mask, J_np, np.inf))
                top_horizons = (ranked[: min(max_listed, ranked.size)] + 1).tolist()
                _logger.info(f"[diag] HOP-LQR lowest-J horizons: {top_horizons}")
            auto_focus_horizons = self._build_auto_focus_horizons(
                total_horizons=bar_E.shape[0],
                bad_M=bad_M,
                bad_bar_F=bad_bar_F,
                bad_P0=bad_P0,
                top_horizons=top_horizons,
            )
            if auto_focus_horizons:
                _logger.info(f"[diag] HOP-LQR auto-focus horizons: {auto_focus_horizons}")
                self._print_detailed_horizon_diagnostics(
                    P_T_np,
                    bar_E,
                    bar_F,
                    bar_G,
                    J_np,
                    w,
                    tuple(auto_focus_horizons),
                    eig_floor=eig_floor,
                )
            if self.enable_early_horizon_balance_diagnostics:
                self._print_early_horizon_balance_diagnostics(
                    P_T_np,
                    bar_E,
                    bar_F,
                    bar_G,
                )
            if self.enable_recursive_growth_diagnostics:
                self._print_recursive_growth_diagnostics(
                    composite_maps=composite_maps,
                    bad_M=bad_M,
                    bad_bar_F=bad_bar_F,
                    bad_P0=bad_P0,
                )

        if self.enable_p0_direction_diagnostics:
            self._print_p0_direction_diagnostics(
                P_T_np=P_T_np,
                bar_E=bar_E,
                bar_F=bar_F,
                bar_G=bar_G,
                bad_P0=bad_P0,
            )

        if self.enable_bar_f_matrix_diagnostics:
            self._print_bar_f_matrix_diagnostics(
                composite_maps=composite_maps,
                bad_bar_F=bad_bar_F,
                bad_M=bad_M,
            )

        if self.enable_detailed_diagnostics:
            self._print_detailed_horizon_diagnostics(
                P_T_np,
                bar_E,
                bar_F,
                bar_G,
                J_np,
                w,
                tuple(self.diagnostic_horizons),
                eig_floor=eig_floor,
            )

        self.last_horizon_diagnostics = {
            "finite_j": finite_J,
            "total_j": total_J,
            "bad_M_count": len(bad_M),
            "bad_F_count": len(bad_bar_F),
            "bad_P0_count": len(bad_P0),
            "bad_temp_count": len(bad_temp),
            "first_bad_m": bad_M[0] if bad_M else None,
            "first_bad_f": bad_bar_F[0] if bad_bar_F else None,
            "first_bad_p0": bad_P0[0] if bad_P0 else None,
            "first_bad_temp": bad_temp[0] if bad_temp else None,
            "min_eig_M": min_M_val,
            "worst_cond_M": worst_cond_M,
            "min_rank_F": min_rank_F,
            "min_sv_F": min_sv_val,
            "max_sv_F": max_sv_val,
            "min_eig_P0": min_P0_val,
            "max_rel_temp": max_rel_temp_val,
        }

    def _print_first_bad_p0_diagnostics(
        self,
        P_T_np: np.ndarray,
        bar_E: np.ndarray,
        bar_F: np.ndarray,
        bar_G: np.ndarray,
        bad_P0: list[int],
    ) -> None:
        """Diagnose why the first bad P0 loses positive definiteness."""
        if not bad_P0:
            return

        first_bad = int(bad_P0[0])
        horizons: list[int] = []

        def add_horizon(t: int) -> None:
            if 1 <= t <= bar_E.shape[0] and t not in horizons:
                horizons.append(t)

        add_horizon(1)
        for dt in (-1, 0, 1):
            add_horizon(first_bad + dt)

        _logger.info(f"[diag] HOP-LQR first-bad-P0 root check: first_bad={first_bad}")
        for t in horizons:
            idx = t - 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            try:
                bar_E_sym = 0.5 * (bar_E[idx] + bar_E[idx].T)
                M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                temp_sym = 0.5 * (temp + temp.T)
                P0_sym = 0.5 * ((bar_E_sym - temp_sym) + (bar_E_sym - temp_sym).T)

                eig_E = np.linalg.eigvalsh(bar_E_sym)
                eig_M = np.linalg.eigvalsh(M_t)
                eig_temp = np.linalg.eigvalsh(temp_sym)
                eig_P0, vec_P0 = np.linalg.eigh(P0_sym)

                bar_E_pd = self._make_pd_np(bar_E_sym, 1e-12)
                M_pd = self._make_pd_np(M_t, 1e-12)
                L = np.linalg.cholesky(bar_E_pd)
                Lm = np.linalg.cholesky(M_pd)
                left = np.linalg.solve(L, temp_sym)
                temp_rel = np.linalg.solve(L, left.T).T
                temp_rel = 0.5 * (temp_rel + temp_rel.T)
                eig_rel = np.linalg.eigvalsh(temp_rel)
                # relEig(temp|bar_E) equals sigma^2 of this normalized bar_F map.
                normalized_bar_F = np.linalg.solve(L, bar_F[idx])
                normalized_bar_F = np.linalg.solve(Lm, normalized_bar_F.T).T
                normalized_sv = np.linalg.svd(normalized_bar_F, compute_uv=False)
                max_gain = float(np.max(normalized_sv))
                min_gain = float(np.min(normalized_sv))

                v_bad = vec_P0[:, 0]
                ray_E = float(v_bad.T @ bar_E_sym @ v_bad)
                ray_temp = float(v_bad.T @ temp_sym @ v_bad)
                margin = ray_E - ray_temp
                top_components = np.argsort(np.abs(v_bad))[-3:][::-1]
                top_desc = ",".join(f"{int(i)}:{float(v_bad[i]):+.2e}" for i in top_components)

                norm_E = float(np.linalg.norm(bar_E_sym, ord=2))
                norm_temp = float(np.linalg.norm(temp_sym, ord=2))
                norm_ratio = norm_temp / max(norm_E, 1e-16)
                cond_M = float(np.linalg.cond(M_t))

                _logger.info(
                    f"[diag] HOP-LQR first-bad-P0 t={t}: "
                    f"minEig(M)={float(eig_M[0]):.3e}, cond(M)={cond_M:.3e}, "
                    f"eig(bar_E)=[{float(eig_E[0]):.3e}, {float(eig_E[-1]):.3e}], "
                    f"eig(temp)=[{float(eig_temp[0]):.3e}, {float(eig_temp[-1]):.3e}], "
                    f"temp/bar_E_norm={norm_ratio:.3e}, "
                    f"relEig(temp|bar_E)=[{float(eig_rel[0]):.3e}, {float(eig_rel[-1]):.3e}], "
                    f"barF_gain_sv=[{min_gain:.3e}, {max_gain:.3e}], "
                    f"barF_gain_sq={max_gain * max_gain:.3e}, "
                    f"minEig(P0)={float(eig_P0[0]):.3e}, "
                    f"bad_dir_ray=[E:{ray_E:.3e}, temp:{ray_temp:.3e}, margin:{margin:.3e}], "
                    f"bad_dir_top={top_desc}"
                )
            except Exception as exc:
                _logger.info(f"[diag] HOP-LQR first-bad-P0 t={t}: failed={type(exc).__name__}: {exc}")

    def _print_raw_efg_diagnostics(
        self,
        E: jnp.ndarray,
        F: jnp.ndarray,
        G: jnp.ndarray,
    ) -> None:
        """Print raw one-step LFT matrix quality before composite recursion."""
        E_np = np.asarray(E)
        F_np = np.asarray(F)
        G_np = np.asarray(G)

        def _safe_sym_stats(mat: np.ndarray) -> tuple[float, float, float]:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan, np.nan, np.inf
            try:
                sym = 0.5 * (arr + arr.T)
                eigs = np.linalg.eigvalsh(sym)
                return float(np.min(eigs)), float(np.max(eigs)), float(np.linalg.cond(sym))
            except Exception:
                return np.nan, np.nan, np.inf

        def _safe_svd_stats(mat: np.ndarray) -> tuple[float, float, int]:
            min_sv, max_sv, rank, _ = self._bar_f_rank_stats_np(mat)
            return min_sv, max_sv, rank

        min_E = []
        cond_E = []
        min_G = []
        cond_G = []
        min_sv_F = []
        max_sv_F = []
        rank_F = []
        bad_E = []
        bad_G = []
        bad_F = []

        for idx in range(E_np.shape[0]):
            t = idx + 1
            min_e, _, cond_e = _safe_sym_stats(E_np[idx])
            min_g, _, cond_g = _safe_sym_stats(G_np[idx])
            min_f, max_f, rank_f = _safe_svd_stats(F_np[idx])
            min_E.append(min_e)
            cond_E.append(cond_e)
            min_G.append(min_g)
            cond_G.append(cond_g)
            min_sv_F.append(min_f)
            max_sv_F.append(max_f)
            rank_F.append(rank_f)
            if (not np.isfinite(min_e)) or min_e <= self.validity_eig_floor:
                bad_E.append(t)
            if (not np.isfinite(min_g)) or min_g <= self.validity_eig_floor:
                bad_G.append(t)
            if rank_f < min(F_np[idx].shape):
                bad_F.append(t)

        _logger.info(
            "[diag] HOP-LQR raw E/F/G summary:"
            f" bad_E={len(bad_E)}, bad_G={len(bad_G)}, rank_drop_F={len(bad_F)},"
            f" minEig(E)={float(np.nanmin(min_E)):.3e},"
            f" worstCond(E)={float(np.nanmax(cond_E)):.3e},"
            f" minEig(G)={float(np.nanmin(min_G)):.3e},"
            f" worstCond(G)={float(np.nanmax(cond_G)):.3e},"
            f" minRank(F)={int(min([r for r in rank_F if r >= 0], default=-1))},"
            f" sv(F)=[{float(np.nanmin(min_sv_F)):.3e}, {float(np.nanmax(max_sv_F)):.3e}]"
        )
        if bad_E:
            _logger.info(f"[diag] HOP-LQR raw first bad E horizons: {bad_E[:self.diagnostic_max_listed]}")
        if bad_G:
            _logger.info(f"[diag] HOP-LQR raw first bad G horizons: {bad_G[:self.diagnostic_max_listed]}")
        if bad_F:
            _logger.info(f"[diag] HOP-LQR raw first rank-deficient F horizons: {bad_F[:self.diagnostic_max_listed]}")

        requested = tuple(self.raw_efg_diagnostic_horizons or ())
        horizons = [t for t in requested if 1 <= int(t) <= E_np.shape[0]]
        for t in horizons:
            idx = int(t) - 1
            min_e, max_e, cond_e = _safe_sym_stats(E_np[idx])
            min_g, max_g, cond_g = _safe_sym_stats(G_np[idx])
            min_f, max_f, rank_f = _safe_svd_stats(F_np[idx])
            _logger.info(
                f"[diag] HOP-LQR raw EFG t={int(t)}: "
                f"E[eig]=[{min_e:.3e}, {max_e:.3e}], cond(E)={cond_e:.3e}, "
                f"F[rank]={rank_f}, sv(F)=[{min_f:.3e}, {max_f:.3e}], "
                f"G[eig]=[{min_g:.3e}, {max_g:.3e}], cond(G)={cond_g:.3e}"
            )

    def _print_early_surrogate_gap_diagnostics(
        self,
        P_T_np: np.ndarray,
        bar_E: np.ndarray,
        bar_F: np.ndarray,
        bar_G: np.ndarray,
        J_np: np.ndarray,
        w: float,
        max_horizon: int = 5,
    ) -> None:
        """
        Compare early-horizon surrogate cost against the nominal terminal cost.

        This is aimed at the common quadrotor failure mode where t=1 looks
        abnormally cheap even though the nominal trajectory has not actually
        reached the goal. In that case, the local surrogate is likely too
        optimistic around z0 = [delta_x0; 1] with delta_x0 ~= 0.
        """
        if self._debug_x0_for_query is None:
            return

        x0_np = np.asarray(self._debug_x0_for_query)
        upper = min(max_horizon, bar_E.shape[0], J_np.shape[0])
        _logger.info("[diag] HOP-LQR early surrogate gap:")
        for t in range(1, upper + 1):
            idx = t - 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)

            raw_quad = np.nan
            raw_total = np.nan
            surrogate_gap = np.nan

            try:
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                raw_P0_inv = 0.5 * ((bar_E[idx] - temp) + (bar_E[idx] - temp).T)
                raw_P0_inv = self._make_pd_np(raw_P0_inv, 1e-12)
                y_raw = np.linalg.solve(raw_P0_inv, x0_np)
                raw_quad = float(0.5 * x0_np.T @ y_raw)
                raw_total = raw_quad
            except Exception:
                raw_quad = np.nan
                raw_total = np.nan

            J_t = float(J_np[idx]) if np.isfinite(J_np[idx]) else J_np[idx]
            _logger.info(
                f"[diag] HOP-LQR early t={t}: "
                f"J={J_t:.3e}, "
                f"raw_quad={raw_quad:.3e}, "
                f"raw_total={raw_total:.3e}, "
                f"phi_minus_raw={surrogate_gap:.3e}, "
            )

    def _print_terminal_p0_norm_diagnostics(
        self,
        P_T_np: np.ndarray,
        bar_E: np.ndarray,
        bar_F: np.ndarray,
        bar_G: np.ndarray,
        J_np: np.ndarray,
    ) -> None:
        """
        Print terminal-surrogate and equivalent initial Riccati norms.

        This distinguishes two different failure modes:
        - ||P_T(t)|| is already tiny: terminal surrogate construction is weak.
        - ||P_T(t)|| is large but ||P0(t)||/J(t) is small: the HOP-LQR query
          believes local dynamics can cheaply cancel the terminal error.
        """
        horizons = tuple(int(t) for t in self.terminal_p0_norm_diagnostic_horizons)
        valid_horizons = [t for t in horizons if 1 <= t <= bar_E.shape[0]]
        if not valid_horizons:
            return

        _logger.info("[diag] HOP-LQR terminal/P0 norm diagnostics:")
        for t in valid_horizons:
            idx = t - 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)
            J_t = float(J_np[idx]) if idx < J_np.shape[0] and np.isfinite(J_np[idx]) else np.nan

            try:
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                P0_inv = 0.5 * ((bar_E[idx] - temp) + (bar_E[idx] - temp).T)
                P0_inv_pd = self._make_pd_np(P0_inv, 1e-12)
                P0 = np.linalg.inv(P0_inv_pd)

                norm_P_T = float(np.linalg.norm(P_T_t, ord=2))
                norm_bar_E = float(np.linalg.norm(bar_E[idx], ord=2))
                norm_bar_G = float(np.linalg.norm(bar_G[idx], ord=2))
                norm_M = float(np.linalg.norm(M_t, ord=2))
                norm_temp = float(np.linalg.norm(0.5 * (temp + temp.T), ord=2))
                norm_P0_inv = float(np.linalg.norm(P0_inv_pd, ord=2))
                norm_P0 = float(np.linalg.norm(P0, ord=2))
                min_eig_P_T = float(np.min(np.linalg.eigvalsh(0.5 * (P_T_t + P_T_t.T))))
                min_eig_P0_inv = float(np.min(np.linalg.eigvalsh(P0_inv_pd)))
                cond_P0_inv = float(np.linalg.cond(P0_inv_pd))
            except Exception:
                norm_P_T = np.nan
                norm_bar_E = np.nan
                norm_bar_G = np.nan
                norm_M = np.nan
                norm_temp = np.nan
                norm_P0_inv = np.nan
                norm_P0 = np.nan
                min_eig_P_T = np.nan
                min_eig_P0_inv = np.nan
                cond_P0_inv = np.inf

            _logger.info(
                f"[diag] HOP-LQR norm t={t}: "
                f"J={J_t:.3e}, "
                f"||P_T||2={norm_P_T:.3e}, minEig(P_T)={min_eig_P_T:.3e}, "
                f"||bar_E||2={norm_bar_E:.3e}, ||bar_G||2={norm_bar_G:.3e}, "
                f"||M=P_T+bar_G||2={norm_M:.3e}, "
                f"||temp||2={norm_temp:.3e}, "
                f"||P0_inv||2={norm_P0_inv:.3e}, minEig(P0_inv)={min_eig_P0_inv:.3e}, "
                f"cond(P0_inv)={cond_P0_inv:.3e}, "
                f"||P0||2={norm_P0:.3e}"
            )

    def _build_auto_focus_horizons(
        self,
        total_horizons: int,
        bad_M: list[int],
        bad_bar_F: list[int],
        bad_P0: list[int],
        top_horizons: list[int],
    ) -> list[int]:
        """
        自动选择一组最值得细看的 horizon：
        1. 当前 J 最小的几个；
        2. 各类病态首次出现附近；
        3. 病态开始前后的过渡区间。
        """
        selected: list[int] = []

        def add_horizon(t: int) -> None:
            if 1 <= t <= total_horizons and t not in selected:
                selected.append(t)

        for t in top_horizons[:4]:
            add_horizon(t)

        def add_neighborhood(base: Optional[int]) -> None:
            if base is None:
                return
            for dt in (-2, -1, 0, 1, 2):
                add_horizon(base + dt)

        first_bad_M = bad_M[0] if bad_M else None
        first_bad_F = bad_bar_F[0] if bad_bar_F else None
        first_bad_P0 = bad_P0[0] if bad_P0 else None
        add_neighborhood(first_bad_P0)
        add_neighborhood(first_bad_F)
        add_neighborhood(first_bad_M)

        if not selected:
            for t in (2, 3, 4, 5, 8, 9, 10, 12):
                add_horizon(t)

        return selected[: self.diagnostic_max_listed + 8]

    def _print_selected_horizon_matrix_diagnostics(
        self,
        P_T: jnp.ndarray,
        composite_maps: HOPCompositeMaps,
        J_values: jnp.ndarray,
        selection_costs: np.ndarray,
        selected_T: int,
        w: float,
        eig_floor: float = 1e-10,
    ) -> None:
        """
        打印当前最值得怀疑的 horizon 上的 J 与矩阵质量对照。

        目标不是全面展开所有 horizon，而是帮助判断：
        - HOP-LQR 当前选中的短 horizon 是否恰好对应矩阵开始失真；
        - 最低 J 的几个候选，与 first bad P0/M/F 之间是否重合。
        """
        if not self.enable_selected_horizon_matrix_diagnostics:
            return

        total_horizons = int(np.asarray(J_values).shape[0])
        valid_info = dict(self.last_validity_info or {})
        bad_M = list(valid_info.get("bad_M", []) or [])
        bad_bar_F = list(valid_info.get("bad_bar_F", []) or [])
        bad_P0 = list(valid_info.get("bad_P0", []) or [])
        bad_temp = list(valid_info.get("bad_temp", []) or [])

        selected: list[int] = []

        def add_horizon(t: int) -> None:
            t_int = int(t)
            if 1 <= t_int <= total_horizons and t_int not in selected:
                selected.append(t_int)

        def add_neighborhood(center: Optional[int]) -> None:
            if center is None:
                return
            radius = max(0, int(self.selected_horizon_diag_neighbor_radius))
            for dt in range(-radius, radius + 1):
                add_horizon(int(center) + dt)

        add_neighborhood(selected_T)

        selection_arr = np.asarray(selection_costs)
        finite_idx = np.flatnonzero(np.isfinite(selection_arr))
        top_horizons: list[int] = []
        if finite_idx.size > 0:
            order = finite_idx[np.argsort(selection_arr[finite_idx])]
            top_horizons = (order[: max(1, int(self.selected_horizon_diag_top_k))] + 1).tolist()
            for t in top_horizons:
                add_horizon(t)

        first_bad_P0 = bad_P0[0] if bad_P0 else None
        first_bad_temp = bad_temp[0] if bad_temp else None
        first_bad_F = bad_bar_F[0] if bad_bar_F else None
        first_bad_M = bad_M[0] if bad_M else None
        add_neighborhood(first_bad_P0)
        add_neighborhood(first_bad_temp)
        add_neighborhood(first_bad_F)
        add_neighborhood(first_bad_M)

        if not selected:
            return

        _logger.info(
            "[diag] HOP-LQR selected-horizon matrix check:"
            f" selected T*={int(selected_T)},"
            f" lowest selection-cost horizons={top_horizons},"
            f" first_bad_P0={first_bad_P0},"
            f" first_bad_temp={first_bad_temp},"
            f" first_bad_F={first_bad_F},"
            f" first_bad_M={first_bad_M}"
        )
        self._print_detailed_horizon_diagnostics(
            np.asarray(P_T),
            np.asarray(composite_maps.bar_E),
            np.asarray(composite_maps.bar_F),
            np.asarray(composite_maps.bar_G),
            np.asarray(J_values),
            w,
            tuple(selected),
            eig_floor=eig_floor,
        )

    def _print_early_horizon_balance_diagnostics(
        self,
        P_T_np: np.ndarray,
        bar_E: np.ndarray,
        bar_F: np.ndarray,
        bar_G: np.ndarray,
        max_horizon: int = 12,
    ) -> None:
        """
        专门诊断早期 horizon 上 P0 为何很快变坏：
        重点看 bar_E 与 temp = F M^{-1} F^T 的量级关系。
        """
        _logger.info("[diag] HOP-LQR early-horizon balance:")
        upper = min(max_horizon, bar_E.shape[0])
        for t in range(1, upper + 1):
            idx = t - 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)

            def _safe_norm(mat: np.ndarray) -> float:
                if not np.all(np.isfinite(mat)):
                    return np.nan
                try:
                    return float(np.linalg.norm(mat, ord=2))
                except Exception:
                    return np.nan

            def _safe_max_eig(mat: np.ndarray) -> float:
                if not np.all(np.isfinite(mat)):
                    return np.nan
                try:
                    sym = 0.5 * (mat + mat.T)
                    return float(np.max(np.linalg.eigvalsh(sym)))
                except Exception:
                    return np.nan

            try:
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                temp = 0.5 * (temp + temp.T)
                bar_E_sym = 0.5 * (bar_E[idx] + bar_E[idx].T)
                diff = bar_E_sym - temp
                norm_bar_E = _safe_norm(bar_E_sym)
                norm_temp = _safe_norm(temp)
                temp_to_barE = norm_temp / max(norm_bar_E, 1e-16)
                maxeig_temp = _safe_max_eig(temp)
                mineig_diff = np.nan
                try:
                    mineig_diff = float(np.min(np.linalg.eigvalsh(0.5 * (diff + diff.T))))
                except Exception:
                    mineig_diff = np.nan
            except Exception:
                norm_bar_E = _safe_norm(0.5 * (bar_E[idx] + bar_E[idx].T))
                norm_temp = np.nan
                temp_to_barE = np.nan
                maxeig_temp = np.nan
                mineig_diff = np.nan

            _logger.info(
                f"[diag] HOP-LQR balance t={t}: "
                f"||bar_E||2={norm_bar_E:.3e}, "
                f"||temp||2={norm_temp:.3e}, "
                f"temp/barE={temp_to_barE:.3e}, "
                f"maxEig(temp)={maxeig_temp:.3e}, "
                f"minEig(barE-temp)={mineig_diff:.3e}"
            )

    def _print_recursive_growth_diagnostics(
        self,
        composite_maps: HOPCompositeMaps,
        bad_M: list[int],
        bad_bar_F: list[int],
        bad_P0: list[int],
        max_horizon: int = 12,
    ) -> None:
        """
        专门定位复合映射递推在哪一步开始把 bar_F / bar_G 放大。

        这里直接检查论文递推中的分母 W_k = E_k + bar_G_{k-1}，
        以及每一步 bar_F/bar_G 相对前一步的增长倍率。
        """
        E = np.asarray(composite_maps.E)
        F = np.asarray(composite_maps.F)
        G = np.asarray(composite_maps.G)
        bar_E = np.asarray(composite_maps.bar_E)
        bar_F = np.asarray(composite_maps.bar_F)
        bar_G = np.asarray(composite_maps.bar_G)

        def _safe_norm(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan
            try:
                return float(np.linalg.norm(arr, ord=2))
            except Exception:
                return np.nan

        def _safe_min_eig(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan
            try:
                sym = 0.5 * (arr + arr.T)
                return float(np.min(np.linalg.eigvalsh(sym)))
            except Exception:
                return np.nan

        def _safe_cond(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.inf
            try:
                return float(np.linalg.cond(arr))
            except Exception:
                return np.inf

        horizons: list[int] = []

        def add_horizon(t: int) -> None:
            if 1 <= t <= len(E) and t not in horizons:
                horizons.append(t)

        for t in range(1, min(max_horizon, len(E)) + 1):
            add_horizon(t)

        for source in (bad_P0, bad_bar_F, bad_M):
            if source:
                base = source[0]
                for dt in (-1, 0, 1):
                    add_horizon(base + dt)

        _logger.info("[diag] HOP-LQR recursive growth:")
        for t in horizons:
            idx = t - 1
            norm_E = _safe_norm(E[idx])
            norm_F = _safe_norm(F[idx])
            norm_G = _safe_norm(G[idx])
            norm_bar_E = _safe_norm(bar_E[idx])
            norm_bar_F = _safe_norm(bar_F[idx])
            norm_bar_G = _safe_norm(bar_G[idx])

            if idx == 0:
                norm_bar_F_prev = np.nan
                norm_bar_G_prev = np.nan
                growth_bar_F = np.nan
                growth_bar_G = np.nan
                min_eig_W = np.nan
                cond_W = np.nan
            else:
                norm_bar_F_prev = _safe_norm(bar_F[idx - 1])
                norm_bar_G_prev = _safe_norm(bar_G[idx - 1])
                growth_bar_F = norm_bar_F / max(norm_bar_F_prev, 1e-16)
                growth_bar_G = norm_bar_G / max(norm_bar_G_prev, 1e-16)
                W_k = 0.5 * (
                    (E[idx] + bar_G[idx - 1]) + (E[idx] + bar_G[idx - 1]).T
                )
                min_eig_W = _safe_min_eig(W_k)
                cond_W = _safe_cond(W_k)

            _logger.info(
                f"[diag] HOP-LQR recur t={t}: "
                f"||E||2={norm_E:.3e}, ||F||2={norm_F:.3e}, ||G||2={norm_G:.3e}, "
                f"||bar_E||2={norm_bar_E:.3e}, ||bar_F||2={norm_bar_F:.3e}, ||bar_G||2={norm_bar_G:.3e}, "
                f"growth(bar_F)={growth_bar_F:.3e}, growth(bar_G)={growth_bar_G:.3e}, "
                f"minEig(W)={min_eig_W:.3e}, cond(W)={cond_W:.3e}"
            )

    def _print_p0_direction_diagnostics(
        self,
        P_T_np: np.ndarray,
        bar_E: np.ndarray,
        bar_F: np.ndarray,
        bar_G: np.ndarray,
        bad_P0: list[int],
        max_horizon: int = 8,
    ) -> None:
        """
        看 P0 = bar_E - temp 为何失去正定性：
        1. bar_E 的最弱方向上，temp 的 Rayleigh quotient 有多大；
        2. temp 的最强方向是否正好对齐 bar_E 的弱方向；
        3. 两者各自的头尾特征值规模。
        """
        total_horizons = bar_E.shape[0]
        horizons: list[int] = []

        def add_horizon(t: int) -> None:
            if 1 <= t <= total_horizons and t not in horizons:
                horizons.append(t)

        for t in range(1, min(max_horizon, total_horizons) + 1):
            add_horizon(t)

        if bad_P0:
            first_bad = bad_P0[0]
            for dt in (-1, 0, 1, 2):
                add_horizon(first_bad + dt)

        _logger.info("[diag] HOP-LQR P0 direction diagnostics:")
        for t in horizons:
            idx = t - 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)

            try:
                bar_E_sym = 0.5 * (bar_E[idx] + bar_E[idx].T)
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                temp_sym = 0.5 * (temp + temp.T)
                P0_sym = 0.5 * ((bar_E_sym - temp_sym) + (bar_E_sym - temp_sym).T)

                eig_E, vec_E = np.linalg.eigh(bar_E_sym)
                eig_temp, vec_temp = np.linalg.eigh(temp_sym)
                eig_P0 = np.linalg.eigvalsh(P0_sym)

                vE_min = vec_E[:, 0]
                vT_max = vec_temp[:, -1]

                temp_on_barE_weak = float(vE_min.T @ temp_sym @ vE_min)
                barE_on_temp_strong = float(vT_max.T @ bar_E_sym @ vT_max)
                align_abs = float(abs(vE_min.T @ vT_max))

                eigE_min = float(eig_E[0])
                eigE_mid = float(eig_E[min(1, len(eig_E) - 1)])
                eigE_max = float(eig_E[-1])
                eigT_min = float(eig_temp[0])
                eigT_mid = float(eig_temp[min(1, len(eig_temp) - 1)])
                eigT_max = float(eig_temp[-1])
                eigP0_min = float(eig_P0[0])

                weak_dir_margin = eigE_min - temp_on_barE_weak
                strong_dir_margin = barE_on_temp_strong - eigT_max
            except Exception:
                eigE_min = np.nan
                eigE_mid = np.nan
                eigE_max = np.nan
                eigT_min = np.nan
                eigT_mid = np.nan
                eigT_max = np.nan
                eigP0_min = np.nan
                temp_on_barE_weak = np.nan
                barE_on_temp_strong = np.nan
                align_abs = np.nan
                weak_dir_margin = np.nan
                strong_dir_margin = np.nan

            _logger.info(
                f"[diag] HOP-LQR p0-dir t={t}: "
                f"eig(bar_E)=[{eigE_min:.3e}, {eigE_mid:.3e}, {eigE_max:.3e}], "
                f"eig(temp)=[{eigT_min:.3e}, {eigT_mid:.3e}, {eigT_max:.3e}], "
                f"minEig(P0)={eigP0_min:.3e}, "
                f"temp_on_barE_weak={temp_on_barE_weak:.3e}, "
                f"barE_on_temp_strong={barE_on_temp_strong:.3e}, "
                f"weak_margin={weak_dir_margin:.3e}, "
                f"strong_margin={strong_dir_margin:.3e}, "
                f"|<vE_min,vT_max>|={align_abs:.3e}"
            )

    def _print_bar_f_matrix_diagnostics(
        self,
        composite_maps: HOPCompositeMaps,
        bad_bar_F: list[int],
        bad_M: list[int],
        max_horizon: int = 12,
    ) -> None:
        """
        聚焦诊断 F_k 与 bar_F_k 的秩/谱变化，帮助判断：
        1. 是单步 F_k 本身就信息不足；
        2. 还是复合递推 bar_F_prev @ Z2 导致列空间逐步塌缩。
        """
        F = np.asarray(composite_maps.F)
        bar_F = np.asarray(composite_maps.bar_F)

        def _safe_svd_stats(mat: np.ndarray) -> tuple[float, float, int]:
            min_sv, max_sv, rank, _ = self._bar_f_rank_stats_np(mat)
            if rank < 0:
                return np.nan, np.nan, -1
            return min_sv, max_sv, rank

        def _safe_norm(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan
            try:
                return float(np.linalg.norm(arr, ord=2))
            except Exception:
                return np.nan

        horizons: list[int] = []

        def add_horizon(t: int) -> None:
            if 1 <= t <= F.shape[0] and t not in horizons:
                horizons.append(t)

        for t in range(1, min(max_horizon, F.shape[0]) + 1):
            add_horizon(t)

        for source in (bad_bar_F, bad_M):
            if source:
                base = source[0]
                for dt in (-2, -1, 0, 1, 2):
                    add_horizon(base + dt)

        _logger.info("[diag] HOP-LQR F-chain diagnostics:")
        for t in horizons:
            idx = t - 1
            min_sv_F, max_sv_F, rank_F = _safe_svd_stats(F[idx])
            min_sv_bar_F, max_sv_bar_F, rank_bar_F = _safe_svd_stats(bar_F[idx])
            norm_F = _safe_norm(F[idx])
            norm_bar_F = _safe_norm(bar_F[idx])

            if idx == 0:
                growth_bar_F = np.nan
                drift_bar_F = np.nan
            else:
                prev_norm = _safe_norm(bar_F[idx - 1])
                growth_bar_F = norm_bar_F / max(prev_norm, 1e-16)
                try:
                    drift_bar_F = float(
                        np.linalg.norm(bar_F[idx] - bar_F[idx - 1], ord=2) / max(prev_norm, 1e-16)
                    )
                except Exception:
                    drift_bar_F = np.nan

            _logger.info(
                f"[diag] HOP-LQR F-chain t={t}: "
                f"rank(F)={rank_F}, sv(F)=[{min_sv_F:.3e}, {max_sv_F:.3e}], ||F||2={norm_F:.3e}; "
                f"rank(bar_F)={rank_bar_F}, sv(bar_F)=[{min_sv_bar_F:.3e}, {max_sv_bar_F:.3e}], "
                f"||bar_F||2={norm_bar_F:.3e}, growth(bar_F)={growth_bar_F:.3e}, drift(bar_F)={drift_bar_F:.3e}"
            )

    def _print_detailed_horizon_diagnostics(
        self,
        P_T_np: np.ndarray,
        bar_E: np.ndarray,
        bar_F: np.ndarray,
        bar_G: np.ndarray,
        J_np: np.ndarray,
        w: float,
        requested_horizons: Tuple[int, ...],
        eig_floor: float = 1e-10,
    ) -> None:
        """
        仅在显式开启时打印少量指定 horizon 的详细数值信息。
        默认关闭，避免污染其他模型的输出。
        """
        def _safe_norm(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan
            try:
                return float(np.linalg.norm(arr, ord=2))
            except Exception:
                return np.nan

        def _safe_min_eig(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan
            try:
                sym = 0.5 * (arr + arr.T)
                return float(np.min(np.linalg.eigvalsh(sym)))
            except Exception:
                return np.nan

        def _safe_extreme_eigs(mat: np.ndarray) -> tuple[float, float]:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan, np.nan
            try:
                sym = 0.5 * (arr + arr.T)
                eigs = np.linalg.eigvalsh(sym)
                return float(np.min(eigs)), float(np.max(eigs))
            except Exception:
                return np.nan, np.nan

        def _safe_svd_stats(mat: np.ndarray) -> tuple[float, float, int]:
            min_sv, max_sv, rank, _ = self._bar_f_rank_stats_np(mat)
            if rank < 0:
                return np.nan, np.nan, -1
            return min_sv, max_sv, rank

        if requested_horizons:
            horizons = [t for t in requested_horizons if 1 <= t <= bar_E.shape[0]]
        else:
            horizons = []
            for t in (2, 3, 4, 8, 13, 20):
                if 1 <= t <= bar_E.shape[0]:
                    horizons.append(t)

        for t in horizons:
            idx = t - 1
            P_T_t = self._terminal_matrix_np(P_T_np, idx)
            M_t = 0.5 * ((P_T_t + bar_G[idx]) + (P_T_t + bar_G[idx]).T)
            try:
                eigs_M = np.linalg.eigvalsh(M_t)
                min_M = float(np.min(eigs_M))
                cond_M = float(np.linalg.cond(M_t))
            except Exception:
                min_M = np.nan
                cond_M = np.inf

            min_sv, max_sv, rank_F = _safe_svd_stats(bar_F[idx])

            try:
                temp = bar_F[idx] @ np.linalg.solve(M_t, bar_F[idx].T)
                P0_t = 0.5 * ((bar_E[idx] - temp) + (bar_E[idx] - temp).T)
                min_P0 = _safe_min_eig(P0_t)
                norm_P0 = _safe_norm(P0_t)
            except Exception:
                temp = np.full_like(bar_E[idx], np.nan)
                min_P0 = np.nan
                norm_P0 = np.nan

            bar_E_sym = 0.5 * (bar_E[idx] + bar_E[idx].T)
            temp_sym = 0.5 * (temp + temp.T)
            min_bar_E = _safe_min_eig(bar_E_sym)
            min_temp, max_temp = _safe_extreme_eigs(temp_sym)
            norm_temp = _safe_norm(temp_sym)

            raw_quad = np.nan
            raw_total = np.nan
            if self.enable_query_cost_diagnostics:
                try:
                    x0_np = np.asarray(self._debug_x0_for_query)
                    raw_P0_inv = 0.5 * ((bar_E[idx] - temp) + (bar_E[idx] - temp).T)
                    raw_P0_inv = self._make_pd_np(raw_P0_inv, 1e-12)
                    y_raw = np.linalg.solve(raw_P0_inv, x0_np)
                    raw_quad = float(0.5 * x0_np.T @ y_raw)
                    raw_total = raw_quad
                except Exception:
                    raw_quad = np.nan
                    raw_total = np.nan

            J_t = float(J_np[idx]) if np.isfinite(J_np[idx]) else J_np[idx]
            _logger.info(
                f"[diag] HOP-LQR t={t}: "
                f"J={J_t}, "
                f"minEig(M)={min_M:.3e}, cond(M)={cond_M:.3e}, "
                f"rank(F)={rank_F}, sv(F)=[{min_sv:.3e}, {max_sv:.3e}], "
                f"minEig(bar_E)={min_bar_E:.3e}, ||bar_E||2={_safe_norm(bar_E[idx]):.3e}, "
                f"||bar_G||2={_safe_norm(bar_G[idx]):.3e}, "
                f"minEig(temp)={min_temp:.3e}, maxEig(temp)={max_temp:.3e}, ||temp||2={norm_temp:.3e}, "
                f"minEig(P0)={min_P0:.3e}, ||P0||2={norm_P0:.3e}, "
                f"P0_bad={bool((not np.isfinite(min_P0)) or (min_P0 <= eig_floor))}, "
                f"raw_quad={raw_quad:.3e}, raw_total={raw_total:.3e}"
            )
            if self.enable_query_raw_vs_stabilized_diagnostics and self._debug_x0_for_query is not None:
                breakdown = self._query_cost_breakdown_np(
                    np.asarray(self._debug_x0_for_query),
                    P_T_t,
                    bar_E[idx],
                    bar_F[idx],
                    bar_G[idx],
                    w,
                    t,
                )
                stabilized_total = breakdown["stabilized_total"]
                delta_stabilized = (
                    stabilized_total - J_t
                    if np.isfinite(stabilized_total) and np.isfinite(J_t)
                    else np.nan
                )
                _logger.info(
                    f"[diag] HOP-LQR query compare t={t}: "
                    f"raw_total={breakdown['raw_total']:.3e}, "
                    f"raw_floor_total={breakdown['raw_floor_total']:.3e}, "
                    f"stabilized_total={stabilized_total:.3e}, "
                    f"J_minus_stabilized={-delta_stabilized:.3e}, "
                    f"raw_minEig(M)={breakdown['raw_min_M']:.3e}, "
                    f"raw_minEig(P0)={breakdown['raw_min_P0']:.3e}, "
                    f"stab_minEig(M)={breakdown['stabilized_min_M']:.3e}, "
                    f"stab_minEig(P0)={breakdown['stabilized_min_P0']:.3e}, "
                    f"stab_barF_gain={breakdown['stabilized_barF_gain_before']:.3e}"
                    f"->{breakdown['stabilized_barF_gain_after']:.3e}"
                )

    def _print_composite_map_diagnostics(self, composite_maps: HOPCompositeMaps) -> None:
        """
        Print recursive-map diagnostics to locate where bar_E/bar_F/bar_G begin
        to collapse or saturate. This is purely observational.
        """
        def _safe_norm(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.inf
            try:
                return float(np.linalg.norm(arr, ord=2))
            except Exception:
                return np.inf

        def _safe_diff_norm(lhs: np.ndarray, rhs: np.ndarray) -> float:
            lhs_arr = np.asarray(lhs)
            rhs_arr = np.asarray(rhs)
            if (not np.all(np.isfinite(lhs_arr))) or (not np.all(np.isfinite(rhs_arr))):
                return np.inf
            try:
                return float(np.linalg.norm(lhs_arr - rhs_arr, ord=2))
            except Exception:
                return np.inf

        def _safe_min_eig(mat: np.ndarray) -> float:
            arr = np.asarray(mat)
            if not np.all(np.isfinite(arr)):
                return np.nan
            try:
                sym = 0.5 * (arr + arr.T)
                return float(np.min(np.linalg.eigvalsh(sym)))
            except Exception:
                return np.nan

        def _safe_svd_stats(mat: np.ndarray) -> tuple[float, float, int]:
            min_sv, max_sv, rank, _ = self._bar_f_rank_stats_np(mat)
            if rank < 0:
                return np.nan, np.nan, -1
            return min_sv, max_sv, rank

        E = np.asarray(composite_maps.E)
        F = np.asarray(composite_maps.F)
        G = np.asarray(composite_maps.G)
        bar_E = np.asarray(composite_maps.bar_E)
        bar_F = np.asarray(composite_maps.bar_F)
        bar_G = np.asarray(composite_maps.bar_G)

        requested = tuple(self.diagnostic_horizons) if self.diagnostic_horizons else ()
        if requested:
            horizons = [t for t in requested if 1 <= t <= E.shape[0]]
        else:
            horizons = [t for t in (1, 2, 3, 4, 8, 13, 20, 40, 60) if 1 <= t <= E.shape[0]]

        _logger.info("[diag] HOP-LQR composite-map diagnostics:")
        for t in horizons:
            idx = t - 1
            norm_E = _safe_norm(E[idx])
            norm_F = _safe_norm(F[idx])
            norm_G = _safe_norm(G[idx])
            norm_bar_E = _safe_norm(bar_E[idx])
            norm_bar_F = _safe_norm(bar_F[idx])
            norm_bar_G = _safe_norm(bar_G[idx])

            if idx == 0:
                delta_bar_E = 0.0
                delta_bar_F = 0.0
                delta_bar_G = 0.0
            else:
                delta_bar_E = _safe_diff_norm(bar_E[idx], bar_E[idx - 1])
                delta_bar_F = _safe_diff_norm(bar_F[idx], bar_F[idx - 1])
                delta_bar_G = _safe_diff_norm(bar_G[idx], bar_G[idx - 1])

            min_bar_E = _safe_min_eig(bar_E[idx])
            min_bar_G = _safe_min_eig(bar_G[idx])
            min_sv_bar_F, max_sv_bar_F, rank_bar_F = _safe_svd_stats(bar_F[idx])

            _logger.info(
                f"[diag] HOP-LQR map t={t}: "
                f"||E||2={norm_E:.3e}, ||F||2={norm_F:.3e}, ||G||2={norm_G:.3e}, "
                f"||bar_E||2={norm_bar_E:.3e}, ||bar_F||2={norm_bar_F:.3e}, ||bar_G||2={norm_bar_G:.3e}, "
                f"dbar=[{delta_bar_E:.3e}, {delta_bar_F:.3e}, {delta_bar_G:.3e}], "
                f"minEig(bar_E)={min_bar_E:.3e}, minEig(bar_G)={min_bar_G:.3e}, "
                f"rank(bar_F)={rank_bar_F}, sv(bar_F)=[{min_sv_bar_F:.3e}, {max_sv_bar_F:.3e}]"
            )
    
    def compute_optimal_control(self, x0: jnp.ndarray, A_traj: jnp.ndarray, B_traj: jnp.ndarray,
                               Q_traj: jnp.ndarray, R_traj: jnp.ndarray, Q_T: jnp.ndarray, 
                               T_star: int) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        为指定时域 T_star 计算最优控制轨迹
        
        参数:
            x0: 初始状态
            A_traj, B_traj: 系统矩阵序列
            Q_traj, R_traj: 成本矩阵序列
            Q_T: 终端成本矩阵
            T_star: 指定的时域长度
        
        返回:
            u_opt: 最优控制序列
            x_opt: 最优状态轨迹
            control_cost: 控制成本（不含时域惩罚）
        """
        # 截取前 T_star 步
        A_trunc = A_traj[:T_star]
        B_trunc = B_traj[:T_star]
        Q_trunc = Q_traj[:T_star]
        R_trunc = R_traj[:T_star]
        
        # 使用标准 LQR 方法求解固定时域问题
        n, m = self.n, self.m
        
        # 逆向 Riccati 递推
        P = Q_T
        P_list = [P]
        K_list = []
        
        for k in range(T_star-1, -1, -1):
            A_k, B_k, Q_k, R_k = A_trunc[k], B_trunc[k], Q_trunc[k], R_trunc[k]
            B_k_T = jnp.transpose(B_k)
            
            # 计算 S_k 和 K_k
            S_k = self._make_pd(R_k + B_k_T @ P @ B_k)
            K_k = jnp.linalg.solve(S_k, B_k_T @ P @ A_k)
            
            # 更新 P
            P = Q_k + jnp.transpose(A_k) @ P @ A_k - jnp.transpose(A_k) @ P @ B_k @ K_k
            P = self._make_pd(P, min_eig=0.0)
            
            P_list.insert(0, P)
            K_list.insert(0, K_k)
        
        P_list = P_list[:-1]  # 去掉最后一个，使长度与时间步匹配
        
        # 正向模拟计算最优轨迹
        x = x0
        x_opt = [x0]
        u_opt = []
        
        for k in range(T_star):
            K_k = K_list[k]
            u = -K_k @ x
            x = A_trunc[k] @ x + B_trunc[k] @ u
            
            u_opt.append(u)
            x_opt.append(x)
        
        # 计算控制成本
        control_cost = 0.0
        for k in range(T_star):
            x_k = x_opt[k]
            u_k = u_opt[k]
            control_cost += 0.5 * (jnp.dot(x_k, jnp.dot(Q_trunc[k], x_k)) + 
                                  jnp.dot(u_k, jnp.dot(R_trunc[k], u_k)))
        
        # 加上终端成本
        x_T = x_opt[-1]
        control_cost += 0.5 * jnp.dot(x_T, jnp.dot(Q_T, x_T))
        
        return jnp.array(u_opt), jnp.array(x_opt[:-1]), float(control_cost)
    
    def solve(self, x0: jnp.ndarray, A_traj: jnp.ndarray, B_traj: jnp.ndarray,
              Q_traj: jnp.ndarray, R_traj: jnp.ndarray, Q_T: jnp.ndarray,
              w: float, T_min: int = 1, T_max: int = 300) -> int:
        """
        完整的 HOP-LQR 求解流程

        参数:
            x0: 初始状态
            A_traj: 状态矩阵序列 (N, n, n)
            B_traj: 控制矩阵序列 (N, n, m)
            Q_traj: 状态权重矩阵序列 (N, n, n)
            R_traj: 控制权重矩阵序列 (N, m, m)
            Q_T: 终端成本矩阵 (n, n)
            w: 时域惩罚权重
            T_min: 最小时域 (1-indexed, inclusive)
            T_max: 最大时域 (1-indexed, inclusive)

        返回:
            T_star: 最优时域
        """
        N = len(A_traj)
        n, m = self.n, self.m
        
        # 验证输入维度
        assert A_traj.shape == (N, n, n)
        assert B_traj.shape == (N, n, m)
        assert Q_traj.shape == (N, n, n)
        assert R_traj.shape == (N, m, m)
        assert Q_T.shape == (n, n) or Q_T.shape == (N, n, n)
        assert x0.shape == (n,)
        
        # === 阶段1: 预计算复合映射 ==
        # 将矩阵正定化，确保逆存在
        Q_traj_pd, R_traj_pd = self._sanitize_traj_matrices(Q_traj, R_traj)
        Q_T_pd = vmap(self._make_pd)(Q_T) if Q_T.ndim == 3 else self._make_pd(Q_T)
        composite_maps = self.precompute_composite_maps(
            A_traj, B_traj, Q_traj_pd, R_traj_pd, sanitize=False
        )
        
        # === 阶段2: 找到最优时域 ===
        def _inverse_pd_single(M: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
            M = 0.5 * (M + M.T)
            eigvals = jnp.linalg.eigvalsh(M)
            shift = jnp.maximum(0.0, eps - jnp.min(eigvals))
            M_pd = M + shift * jnp.eye(M.shape[0], dtype=M.dtype)
            L = jnp.linalg.cholesky(M_pd)
            return jnp.linalg.solve(L.T, jnp.linalg.solve(L, jnp.eye(M.shape[0], dtype=M.dtype)))
        
        P_T_inv = vmap(_inverse_pd_single)(Q_T_pd) if Q_T_pd.ndim == 3 else _inverse_pd_single(Q_T_pd)
        T_star, J_values = self.find_optimal_horizon(x0, P_T_inv, w, composite_maps, T_min, T_max)
        if self.enable_prefix_invariance_diagnostics:
            self._print_prefix_invariance_diagnostics(
                x0=x0,
                A_traj=A_traj,
                B_traj=B_traj,
                Q_traj=Q_traj_pd,
                R_traj=R_traj_pd,
                Q_T=Q_T_pd,
                w=w,
                full_J_values=J_values,
            )
        
        return T_star
        
        # # === 阶段3: 为最优时域计算最优控制 ===
        # u_opt, x_opt, control_cost = self.compute_optimal_control(
        #     x0, A_traj, B_traj, Q_traj_pd, R_traj_pd, Q_T_pd, T_star
        # )
        
        # # 计算总成本（控制成本 + 时域惩罚）
        # optimal_cost = control_cost + w * T_star
        
        # return HOPLQRResult(
        #     u_opt=u_opt,
        #     x_opt=x_opt,
        #     T_star=T_star,
        #     J_values=J_values,
        #     optimal_cost=optimal_cost
        # )
