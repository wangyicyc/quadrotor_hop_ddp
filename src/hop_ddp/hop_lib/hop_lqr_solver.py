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
from jax.lax import cond, scan
from jax import vmap, jit
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


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

        # ---- 有效性判据阈值：用于标记/过滤病态 horizon；
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

        # 最近一次 solve/query 缓存下来的选择结果。
        self.last_validity_info = {}
        self.last_J_values = None
        self.last_selection_costs = None
        self.last_selected_horizon = None
        
        # 编译关键计算函数
        self._compute_efg = jit(self._compute_efg_impl)
        self._compute_composite_maps = jit(self._compute_composite_maps_impl)
        self._fast_horizon_query = jit(self._fast_horizon_query_impl)

    def apply_settings(self, settings) -> None:
        """从配置对象批量读取 HOP-LQR 稳定化和时域过滤参数。"""
        # query stabilization
        self.enable_query_stabilization = settings.enable_hop_lqr_query_stabilization
        self.query_temp_relative_cap = settings.hop_lqr_query_temp_relative_cap
        # validity filter
        self.enable_horizon_validity_filter = settings.enable_hop_lqr_validity_filter
        self.validity_filter_use_bad_m = settings.hop_lqr_validity_filter_use_bad_m
        self.validity_filter_use_bad_p0 = settings.hop_lqr_validity_filter_use_bad_p0
        self.validity_filter_use_rank_deficient_f = settings.hop_lqr_validity_filter_use_rank_deficient_f
        self.validity_filter_use_bad_temp = settings.hop_lqr_validity_filter_use_bad_temp
        self.validity_eig_floor = settings.hop_lqr_validity_eig_floor
        self.validity_rank_rtol = settings.hop_lqr_validity_rank_rtol
        self.min_candidate_horizon = settings.hop_lqr_min_candidate_horizon
        # recursive bar_F
        self.recursive_bar_f_max_growth = settings.hop_lqr_recursive_bar_f_max_growth
        self.enable_recursive_bar_f_svd_floor = settings.enable_hop_lqr_recursive_bar_f_svd_floor
        self.recursive_bar_f_svd_rtol = settings.hop_lqr_recursive_bar_f_svd_rtol
        self.recursive_bar_f_svd_atol = settings.hop_lqr_recursive_bar_f_svd_atol
        # recursive bar_G
        self.enable_recursive_bar_g_relative_clip = settings.enable_hop_lqr_recursive_bar_g_eig_clip
        self.recursive_bar_g_min_relative_eig = settings.hop_lqr_recursive_bar_g_min_relative_eig
        self.recursive_bar_g_max_relative_eig = settings.hop_lqr_recursive_bar_g_max_relative_eig
        # recursive P0 / temp
        self.enable_recursive_p0_stabilization = settings.enable_hop_lqr_recursive_p0_stabilization
        self.recursive_bar_e_min_eig = settings.hop_lqr_recursive_bar_e_min_eig
        self.recursive_temp_relative_cap = settings.hop_lqr_recursive_temp_relative_cap
        # query bar_G / bar_F / temp
        self.enable_query_bar_g_relative_clip = settings.enable_hop_lqr_query_bar_g_eig_clip
        self.query_bar_g_min_relative_eig = settings.hop_lqr_query_bar_g_min_relative_eig
        self.query_bar_g_max_relative_eig = settings.hop_lqr_query_bar_g_max_relative_eig
        self.enable_query_temp_relative_cap = settings.enable_hop_lqr_query_temp_relative_cap
        self.enable_query_bar_f_gain_cap = settings.enable_hop_lqr_query_bar_f_gain_cap
        self.query_bar_f_gain_cap = settings.hop_lqr_query_bar_f_gain_cap

    def _terminal_matrix_np(self, P_T_np: np.ndarray, idx: int) -> np.ndarray:
        """Return the terminal matrix for one horizon from shared or per-horizon input."""
        return P_T_np[idx] if np.asarray(P_T_np).ndim == 3 else P_T_np

    def _bar_f_rank_stats_np(self, M: np.ndarray) -> tuple[float, float, int, bool]:
        """
        Unified NumPy rank check for bar_F/F-like matrices.

        We intentionally reuse the same tolerance logic everywhere so that
        validity filtering and related matrix checks do not disagree just
        because they used slightly different thresholds.
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
        """NumPy version used by horizon validity checks."""
        M_sym = 0.5 * (M + M.T)
        eigvals = np.linalg.eigvalsh(M_sym)
        shift = max(0.0, float(min_eig) - float(np.min(eigvals)))
        return M_sym + shift * np.eye(M_sym.shape[0], dtype=M_sym.dtype)

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
        # 计算复合映射参数 (bar_E_k, bar_F_k, bar_G_k)
        bar_E, bar_F, bar_G = self._compute_composite_maps(E, F, G)

        composite = HOPCompositeMaps(E=E, F=F, G=G, bar_E=bar_E, bar_F=bar_F, bar_G=bar_G)
        return composite

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
        
        # 快速时域查询 (传入原始 P_T_pd)
        # 通过预计算的复合映射快速评估所有时域的成本
        J_values = self._fast_horizon_query(
            x0, P_T_tilde, w, 
            composite_maps.bar_E, composite_maps.bar_F, composite_maps.bar_G
        )

        # 数值保护：将非有限值（NaN/Inf）替换为无穷大
        # 这样可以避免 argmin 被非有限值污染。
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

        # 返回最优时域和所有时域的成本值
        return T_star, J_values
    
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
        T_star, _ = self.find_optimal_horizon(x0, P_T_inv, w, composite_maps, T_min, T_max)
        
        return T_star
        