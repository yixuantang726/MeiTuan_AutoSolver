"""
tools/ilp.py
ILP（整数线性规划）求解器
全局最优，支持覆盖下限(coverage_lower_bound)和成本上限(score_upper_bound)约束

变量：x_i ∈ {0,1}，候选行 i 是否被选入方案

目标：minimize F(S) = Σ (score_i·w_i - λ·w_i·bundle_i)·x_i  + λ·40

约束：
  1. 每个骑手最多选一次：       Σ_{courier_i=k} x_i ≤ 1
  2. 覆盖下限（S赢时C用）：     Σ w_i·bundle_i·x_i ≥ coverage_lower_bound
  3. 成本上限（C赢时S用）：     Σ score_i·w_i·x_i  ≤ score_upper_bound

注：同一订单可由多个骑手覆盖（概率叠加），
覆盖公式 C(S) = Σ w_i·bundle_i·x_i 为线性近似（精确值需连乘）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import defaultdict
from typing import Dict, Any

from core import load_data, evaluate


def run_ilp(
    lam: float = 30.0,
    coverage_lower_bound: float = 0.0,
    score_upper_bound: float = float("inf"),
) -> Dict[str, Any]:
    """
    ILP求解派单方案，最小化F(S)。

    coverage_lower_bound: C_min，S赢时C传入，强制 C(S) ≥ C_min
    score_upper_bound:    S_max，C赢时S传入，强制 R(S) ≤ S_max
    两个约束可同时使用，也可只用其中一个（默认不激活）。
    """
    candidates = load_data()
    n = len(candidates)

    # ── 目标系数 ──────────────────────────────────────────────────
    # 最小化 Σ (score_i·w_i - λ·w_i·bundle_i)·x_i  （常数项λ·40不影响优化）
    c_obj = np.array([
        c["score"] * c["willingness"] - lam * c["willingness"] * c["bundle_size"]
        for c in candidates
    ], dtype=float)

    # ── 约束矩阵 ──────────────────────────────────────────────────
    A_rows, lb_list, ub_list = [], [], []

    # 约束1：每个骑手最多选一次
    courier_rows = defaultdict(list)
    for i, c in enumerate(candidates):
        courier_rows[c["courier"]].append(i)
    for idxs in courier_rows.values():
        row = np.zeros(n)
        row[idxs] = 1.0
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(1.0)

    # 约束2：每个任务T最多出现在一个选中行
    task_rows = defaultdict(list)
    for i, c in enumerate(candidates):
        for t in c["t_list"]:
            task_rows[t].append(i)
    for idxs in task_rows.values():
        row = np.zeros(n)
        row[idxs] = 1.0
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(1.0)

    # 约束3：覆盖下限
    if coverage_lower_bound > 0:
        row = np.array([
            c["willingness"] * c["bundle_size"] for c in candidates
        ], dtype=float)
        A_rows.append(row)
        lb_list.append(float(coverage_lower_bound))
        ub_list.append(np.inf)

    # 约束4：成本上限
    if score_upper_bound < float("inf"):
        row = np.array([
            c["score"] * c["willingness"] for c in candidates
        ], dtype=float)
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(float(score_upper_bound))

    A = np.vstack(A_rows)
    constraints = LinearConstraint(A, lb_list, ub_list)
    bounds = Bounds(lb=0.0, ub=1.0)
    integrality = np.ones(n)

    result = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds)

    if not result.success:
        return {
            "error": f"ILP求解失败：{result.message}",
            "coverage_lower_bound": coverage_lower_bound,
            "score_upper_bound": score_upper_bound,
        }

    selected = [candidates[i] for i in range(n) if result.x[i] > 0.5]
    metrics = evaluate(selected, lam)

    return {
        "strategy":             "ilp",
        "metrics":              metrics,
        "n_selected":           len(selected),
        "coverage_lower_bound": coverage_lower_bound,
        "score_upper_bound":    score_upper_bound,
        "sample": [
            {
                "task_key":    c["task_key"],
                "courier":     c["courier"],
                "score":       c["score"],
                "willingness": c["willingness"],
                "bundle_size": c["bundle_size"],
            }
            for c in selected[:5]
        ],
        "selected": selected,
    }


def run_coverage_ilp(coverage_lower_bound: float, lam: float = 30.0) -> Dict[str, Any]:
    """C_Agent挑战工具：S赢时用，强制 C(S) ≥ coverage_lower_bound"""
    return run_ilp(lam=lam, coverage_lower_bound=coverage_lower_bound)


def run_score_ilp(score_upper_bound: float, lam: float = 30.0) -> Dict[str, Any]:
    """S_Agent挑战工具：C赢时用，强制 R(S) ≤ score_upper_bound"""
    return run_ilp(lam=lam, score_upper_bound=score_upper_bound)
