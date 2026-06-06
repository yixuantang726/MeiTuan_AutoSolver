"""
tools/heuristic.py
启发式局部搜索策略

run_score_heuristic:    挑战者是S时用 → 从coverage_greedy解出发削成本，目标打败C的高成本greedy
run_coverage_heuristic: 挑战者是C时用 → 从score_greedy解出发补覆盖，F=619<S的787，能赢

两阶段：
  1. 贪心删/补：每步检查F是否下降，下降则接受，直到收敛
  2. 1-swap：尝试用性价比(willingness/score)更高的候选替换当前解中的弱项
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from typing import List, Dict, Any, Set
from core import load_data, evaluate, evaluate_with_redundancy
from tools.greedy import run_coverage_greedy, run_score_greedy

ALL_T = [f"T{str(i).zfill(4)}" for i in range(40)]
REDUNDANT_THRESHOLD = 0.5   # 覆盖概率低于此值的任务才补入冗余骑手
BUNDLE_BONUS = 1.5          # 合单候选的性价比加成系数（鼓励选合单）


def _key(c: Dict) -> str:
    """候选唯一标识：同一courier+task_key组合视为同一行"""
    return f"{c['courier']}|{c['task_key']}"


def _marginal_cost(c: Dict) -> float:
    return c["score"] * c["willingness"]


def _can_add(c: Dict, used_couriers: Set[str]) -> bool:
    """约束：每个骑手只能出现一次（同一订单可由多个骑手覆盖）"""
    return c["courier"] not in used_couriers


def _t_prob_none(selected: List[Dict]) -> Dict[str, float]:
    """计算每个T当前被所有骑手都拒绝的概率（1-P(T被覆盖)）"""
    prob_none = defaultdict(lambda: 1.0)
    for c in selected:
        for t in c["t_list"]:
            prob_none[t] *= (1 - c["willingness"])
    return prob_none


def _add_redundant_targeted_phase(
    selected: List[Dict],
    best_F: float,
    candidates: List[Dict],
    lam: float,
    bundle_priority: bool = False,
) -> tuple[List[Dict], float]:
    """
    规则1：专攻低概率任务的冗余派发（单个补入）
    bundle_priority=True 时合单候选优先排序。
    """
    improved = True
    max_iter = 50
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1

        prob_none = _t_prob_none(selected)
        low_prob_tasks = {t for t in ALL_T if prob_none[t] > (1 - REDUNDANT_THRESHOLD)}
        if not low_prob_tasks:
            break

        selected_keys = {_key(c) for c in selected}
        used_c = {c["courier"] for c in selected}

        def gain_ratio(c):
            gain = sum(prob_none[t] * c["willingness"]
                       for t in c["t_list"] if t in low_prob_tasks)
            bonus = BUNDLE_BONUS if (bundle_priority and c["bundle_size"] >= 2) else 1.0
            return gain * bonus / c["score"] if c["score"] > 0 else 0

        targeted = sorted(
            [c for c in candidates
             if _key(c) not in selected_keys
             and c["courier"] not in used_c
             and any(t in low_prob_tasks for t in c["t_list"])],
            key=lambda c: -gain_ratio(c),
        )

        for cand in targeted:
            trial_F = evaluate_with_redundancy(selected + [cand], lam)["F"]
            if trial_F < best_F:
                selected = selected + [cand]
                best_F = trial_F
                improved = True
                break

    return selected, best_F


def _add_redundant_group_phase(
    selected: List[Dict],
    best_F: float,
    candidates: List[Dict],
    lam: float,
    max_group: int = 5,
    bundle_priority: bool = False,
) -> tuple[List[Dict], float]:
    """
    规则2：对全部任务一次加2~max_group个冗余骑手
    bundle_priority=True 时合单候选优先排序。
    """
    from itertools import combinations

    prob_none = _t_prob_none(selected)
    selected_keys = {_key(c) for c in selected}
    used_c = {c["courier"] for c in selected}

    def gain_ratio(c):
        gain = sum(prob_none[t] * c["willingness"] for t in c["t_list"])
        bonus = BUNDLE_BONUS if (bundle_priority and c["bundle_size"] >= 2) else 1.0
        return gain * bonus / c["score"] if c["score"] > 0 else 0

    pool = sorted(
        [c for c in candidates
         if _key(c) not in selected_keys and c["courier"] not in used_c],
        key=lambda c: -gain_ratio(c),
    )[:20]

    best_group, best_group_F = None, best_F
    for n in range(2, max_group + 1):
        for group_idx in combinations(range(len(pool)), n):
            group = [pool[i] for i in group_idx]
            if len({c["courier"] for c in group}) < n:
                continue
            trial_F = evaluate_with_redundancy(selected + group, lam)["F"]
            if trial_F < best_group_F:
                best_group = group
                best_group_F = trial_F

    if best_group:
        selected = selected + best_group
        best_F = best_group_F

    return selected, best_F


def _swap_phase(
    selected: List[Dict],
    best_F: float,
    candidates: List[Dict],
    lam: float,
    sel_sort_key,
    cand_sort_key,
) -> tuple[List[Dict], float]:
    """
    通用1-swap：从selected里按sel_sort_key选项尝试删除，
    再从unselected里按cand_sort_key找替换，若F下降则接受。
    """
    improved = True
    max_iter = 50
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        selected_keys = {_key(c) for c in selected}
        unselected = sorted(
            [c for c in candidates if _key(c) not in selected_keys],
            key=cand_sort_key,
        )
        for sel_item in sorted(selected, key=sel_sort_key):
            without = [c for c in selected if _key(c) != _key(sel_item)]
            used_c = {c["courier"] for c in without}
            for cand in unselected:
                if not _can_add(cand, used_c):
                    continue
                trial_F = evaluate(without + [cand], lam)["F"]
                if trial_F < best_F:
                    selected = without + [cand]
                    best_F = trial_F
                    improved = True
                    break
            if improved:
                break
    return selected, best_F


def run_score_heuristic(lam: float = 30.0) -> Dict[str, Any]:
    """
    挑战者C用：S为擂主时削成本
    基线 = coverage_greedy，目标 = 往下砍出比S的greedy更低的F

    阶段1 贪心删除：按期望成本降序逐项尝试删，F下降则接受
    阶段2 1-swap：用性价比(willingness/score)更高的候选替换高成本项
    """
    candidates = load_data()
    selected = run_coverage_greedy(lam)["selected"][:]
    best_F = evaluate_with_redundancy(selected, lam)["F"]

    # ── 阶段1：贪心删除高成本低边际覆盖项 ──────────────────────────
    improved = True
    max_iter = 50
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for item in sorted(selected, key=lambda x: -_marginal_cost(x)):
            trial = [c for c in selected if _key(c) != _key(item)]
            trial_F = evaluate_with_redundancy(trial, lam)["F"]
            if trial_F < best_F:
                selected = trial
                best_F = trial_F
                improved = True
                break

    # ── 阶段2：专攻低概率任务，单个补入 ──────────────────────────
    selected, best_F = _add_redundant_targeted_phase(selected, best_F, candidates, lam)
    # ── 阶段3：枚举2~5个冗余骑手组合，取最优 ────────────────────
    selected, best_F = _add_redundant_group_phase(selected, best_F, candidates, lam)

    # ── 阶段3：1-swap，替换候选按性价比降序 ─────────────────────────
    selected, best_F = _swap_phase(
        selected, best_F, candidates, lam,
        sel_sort_key=lambda x: -_marginal_cost(x),
        cand_sort_key=lambda x: -x["willingness"] / x["score"],
    )

    metrics = evaluate_with_redundancy(selected, lam)
    return {
        "strategy":   "score_heuristic",
        "metrics":    metrics,
        "n_selected": len(selected),
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


def run_coverage_heuristic(lam: float = 30.0) -> Dict[str, Any]:
    """
    挑战者S用：C为擂主时补覆盖
    基线 = score_greedy，目标 = 补入高性价比项提升coverage，降F打败C的greedy

    阶段1 贪心补入：按性价比降序逐项尝试加，F下降则接受，直到无法再加
    阶段2 1-swap：用willingness更高的候选替换低willingness项
    """
    candidates = load_data()
    selected = run_score_greedy(lam)["selected"][:]
    best_F = evaluate_with_redundancy(selected, lam)["F"]

    # ── 阶段1：贪心补入高性价比项 ───────────────────────────────────
    all_by_ratio = sorted(candidates, key=lambda x: -x["willingness"] / x["score"])

    improved = True
    max_iter = 50
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        used_c = {c["courier"] for c in selected}
        selected_keys = {_key(c) for c in selected}
        for cand in all_by_ratio:
            if _key(cand) in selected_keys:
                continue
            if not _can_add(cand, used_c):
                continue
            trial = selected + [cand]
            trial_F = evaluate_with_redundancy(trial, lam)["F"]
            if trial_F < best_F:
                selected = trial
                best_F = trial_F
                improved = True
                break

    # ── 阶段2：专攻低概率任务，单个补入 ──────────────────────────
    selected, best_F = _add_redundant_targeted_phase(selected, best_F, candidates, lam)
    # ── 阶段3：枚举2~5个冗余骑手组合，取最优 ────────────────────
    selected, best_F = _add_redundant_group_phase(selected, best_F, candidates, lam)

    # ── 阶段3：1-swap，用高willingness候选替换低willingness项 ────────
    selected, best_F = _swap_phase(
        selected, best_F, candidates, lam,
        sel_sort_key=lambda x: x["willingness"],
        cand_sort_key=lambda x: -(x["willingness"] / x["score"]) * (BUNDLE_BONUS if x["bundle_size"] >= 2 else 1.0),
    )

    metrics = evaluate_with_redundancy(selected, lam)
    return {
        "strategy":   "coverage_heuristic",
        "metrics":    metrics,
        "n_selected": len(selected),
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


def run_score_heuristic_bundle(lam: float = 30.0) -> Dict[str, Any]:
    """合单优先版 score_heuristic：排序时给合单候选加 BUNDLE_BONUS 权重"""
    candidates = load_data()
    selected = run_coverage_greedy(lam)["selected"][:]
    best_F = evaluate_with_redundancy(selected, lam)["F"]

    improved = True
    max_iter = 50
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for item in sorted(selected, key=lambda x: -_marginal_cost(x)):
            trial = [c for c in selected if _key(c) != _key(item)]
            trial_F = evaluate_with_redundancy(trial, lam)["F"]
            if trial_F < best_F:
                selected = trial
                best_F = trial_F
                improved = True
                break

    selected, best_F = _add_redundant_targeted_phase(selected, best_F, candidates, lam, bundle_priority=True)
    selected, best_F = _add_redundant_group_phase(selected, best_F, candidates, lam, bundle_priority=True)
    selected, best_F = _swap_phase(
        selected, best_F, candidates, lam,
        sel_sort_key=lambda x: -_marginal_cost(x),
        cand_sort_key=lambda x: -(x["willingness"] / x["score"]) * (BUNDLE_BONUS if x["bundle_size"] >= 2 else 1.0),
    )

    metrics = evaluate_with_redundancy(selected, lam)
    return {"strategy": "score_heuristic_bundle", "metrics": metrics,
            "n_selected": len(selected), "selected": selected,
            "sample": [{"task_key": c["task_key"], "courier": c["courier"],
                        "score": c["score"], "willingness": c["willingness"],
                        "bundle_size": c["bundle_size"]} for c in selected[:5]]}


def run_coverage_heuristic_bundle(lam: float = 30.0) -> Dict[str, Any]:
    """合单优先版 coverage_heuristic：排序时给合单候选加 BUNDLE_BONUS 权重"""
    candidates = load_data()
    selected = run_score_greedy(lam)["selected"][:]
    best_F = evaluate_with_redundancy(selected, lam)["F"]

    all_by_ratio = sorted(candidates,
                          key=lambda x: -(x["willingness"] / x["score"]) * (BUNDLE_BONUS if x["bundle_size"] >= 2 else 1.0))
    improved = True
    max_iter = 50
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        used_c = {c["courier"] for c in selected}
        selected_keys = {_key(c) for c in selected}
        for cand in all_by_ratio:
            if _key(cand) in selected_keys or not _can_add(cand, used_c):
                continue
            trial_F = evaluate_with_redundancy(selected + [cand], lam)["F"]
            if trial_F < best_F:
                selected = selected + [cand]
                best_F = trial_F
                improved = True
                break

    selected, best_F = _add_redundant_targeted_phase(selected, best_F, candidates, lam, bundle_priority=True)
    selected, best_F = _add_redundant_group_phase(selected, best_F, candidates, lam, bundle_priority=True)
    selected, best_F = _swap_phase(
        selected, best_F, candidates, lam,
        sel_sort_key=lambda x: x["willingness"],
        cand_sort_key=lambda x: -(x["willingness"] / x["score"]) * (BUNDLE_BONUS if x["bundle_size"] >= 2 else 1.0),
    )

    metrics = evaluate_with_redundancy(selected, lam)
    return {"strategy": "coverage_heuristic_bundle", "metrics": metrics,
            "n_selected": len(selected), "selected": selected,
            "sample": [{"task_key": c["task_key"], "courier": c["courier"],
                        "score": c["score"], "willingness": c["willingness"],
                        "bundle_size": c["bundle_size"]} for c in selected[:5]]}
