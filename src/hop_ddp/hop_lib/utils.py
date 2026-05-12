import jax.numpy as jnp
import logging
import os
from datetime import datetime

# 全局日志记录器缓存
_loggers = {}
_run_dir = None


def get_run_dir():
    """获取本次运行的输出目录：datas/YYYY-MM-DD_HH-MM-SS。"""
    global _run_dir
    if _run_dir is not None:
        return _run_dir

    env_run_dir = os.environ.get("HOP_DDP_RUN_DIR")
    if env_run_dir:
        _run_dir = os.path.abspath(env_run_dir)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        _run_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datas', timestamp)
        )

    os.makedirs(_run_dir, exist_ok=True)
    return _run_dir

def get_logger(name='hop', log_dir=None):
    """
    获取一个只写入文件（不输出到控制台）的 logger。
    日志文件命名格式: name_YYYY-MM-DD_HH-MM-SS.log
    """
    if name in _loggers:
        return _loggers[name]

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    env_log_file = os.environ.get("HOP_DDP_LOG_FILE")
    if env_log_file:
        log_filename = os.path.abspath(env_log_file)
        log_dir = os.path.dirname(log_filename)
        os.makedirs(log_dir, exist_ok=True)
    elif log_dir is None:
        log_dir = get_run_dir()
        log_filename = os.path.join(log_dir, f'{name}_{timestamp}.log')
    else:
        log_filename = os.path.join(log_dir, f'{name}_{timestamp}.log')

    logger = logging.getLogger(f'hop_{name}_{timestamp}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 不传播到根 logger，避免输出到控制台

    # 移除已有的 handler（防止重复）
    logger.handlers.clear()

    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.log_file = log_filename
    logger.run_dir = log_dir

    _loggers[name] = logger
    return logger


def make_pd(matrix, eps=1e-9):
    """
    确保矩阵是正定的（Positive Definite），通过对角线加载修复。
    针对条件数爆炸问题，它会根据矩阵的量级自适应地添加偏移。
    """
    n = matrix.shape[-1]
    # 获取矩阵的迹或最大对角线元素，用于自适应缩放 eps
    # 这样当 QT 很大时，eps 也会相应增大，保持条件数在一个可控范围
    scale = jnp.max(jnp.abs(jnp.diag(matrix)))
    adaptive_eps = jnp.maximum(eps, scale * 1e-12)
    
    # 获取最小特征值（JIT 友好方式）
    # 注意：jnp.linalg.eigvalsh 只适用于对称矩阵，Q 和 R 矩阵通常是对称的
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    min_eig = jnp.min(eigenvalues)
    
    # 如果最小特征值小于阈值，则补齐差额并额外多给一点安全垫
    delta = jnp.where(min_eig < adaptive_eps, adaptive_eps - min_eig, 0.0)
    return matrix + delta * jnp.eye(n)
