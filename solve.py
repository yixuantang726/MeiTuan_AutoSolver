"""
solve.py - 提交用，无外部依赖
策略：score_greedy + coverage_greedy + 启发式局部搜索（含冗余派发）
λ=30.0 固定
"""

from collections import defaultdict
from typing import List, Dict, Tuple

LAM = 30.0
REDUNDANT_THRESHOLD = 0.5
BUNDLE_BONUS = 1.5
BUNDLE_F_DISCOUNT = 0.9


# ── 解析输入 ──────────────────────────────────────────────────────────

def _parse(input_text: str) -> List[Dict]:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    candidates = []
    for line in lines[start:]:
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        task_key, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        t_list = [t.strip() for t in task_key.strip().split(",")]
        candidates.append({
            "task_key":    task_key.strip(),
            "t_list":      t_list,
            "bundle_size": len(t_list),
            "courier":     courier_id.strip(),
            "score":       score,
            "willingness": willingness,
        })
    return candidates


def _all_tasks(candidates: List[Dict]) -> List[str]:
    tasks = set()
    for c in candidates:
        tasks.update(c["t_list"])
    return sorted(tasks)


# ── 评估（支持冗余派发，顺序成本模型）───────────────────────────────────

def _evaluate(selected: List[Dict], lam: float, all_t: List[str]) -> Dict:
    t_couriers = defaultdict(list)
    for c in selected:
        for t in c["t_list"]:
            t_couriers[t].append(c)

    coverage = expected_score = 0.0
    for t in all_t:
        prob_none = 1.0
        cost = 0.0
        for c in t_couriers[t]:
            cost += c["score"] * c["willingness"] * prob_none
            prob_none *= (1 - c["willingness"])
        coverage += (1 - prob_none)
        expected_score += cost

    penalty = lam * (len(all_t) - coverage)
    return {
        "F":        round(expected_score + penalty, 3),
        "coverage": round(coverage, 3),
        "n_bundle": sum(1 for c in selected if c["bundle_size"] > 1),
    }


def _key(c: Dict) -> str:
    return f"{c['courier']}|{c['task_key']}"


# ── Greedy ────────────────────────────────────────────────────────────

def _greedy(candidates, sort_key) -> List[Dict]:
    used_couriers, used_tasks, selected = set(), set(), []
    for c in sorted(candidates, key=sort_key):
        if c["courier"] in used_couriers:
            continue
        if any(t in used_tasks for t in c["t_list"]):
            continue
        used_couriers.add(c["courier"])
        used_tasks.update(c["t_list"])
        selected.append(c)
    return selected


def _score_greedy(candidates):
    return _greedy(candidates, lambda x: x["score"])


def _coverage_greedy(candidates):
    return _greedy(candidates, lambda x: -(x["willingness"] * x["bundle_size"]))


# ── 启发式辅助 ────────────────────────────────────────────────────────

def _prob_none_map(selected):
    p = defaultdict(lambda: 1.0)
    for c in selected:
        for t in c["t_list"]:
            p[t] *= (1 - c["willingness"])
    return p


def _swap_phase(selected, best_F, candidates, lam, all_t,
                sel_key, cand_key, max_iter=30) -> Tuple[List, float]:
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        sk = {_key(c) for c in selected}
        unsel = sorted([c for c in candidates if _key(c) not in sk], key=cand_key)
        for item in sorted(selected, key=sel_key):
            without = [c for c in selected if _key(c) != _key(item)]
            uc = {c["courier"] for c in without}
            for cand in unsel:
                if cand["courier"] in uc:
                    continue
                if _evaluate(without + [cand], lam, all_t)["F"] < best_F:
                    selected = without + [cand]
                    best_F = _evaluate(selected, lam, all_t)["F"]
                    improved = True
                    break
            if improved:
                break
    return selected, best_F


def _add_targeted(selected, best_F, candidates, lam, all_t,
                  bundle_priority=False, max_iter=30) -> Tuple[List, float]:
    bonus = BUNDLE_BONUS if bundle_priority else 1.0
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        pn = _prob_none_map(selected)
        low = {t for t in all_t if pn[t] > (1 - REDUNDANT_THRESHOLD)}
        if not low:
            break
        sk = {_key(c) for c in selected}
        uc = {c["courier"] for c in selected}

        def gr(c):
            g = sum(pn[t] * c["willingness"] for t in c["t_list"] if t in low)
            b = bonus if c["bundle_size"] >= 2 else 1.0
            return g * b / c["score"] if c["score"] > 0 else 0

        pool = sorted(
            [c for c in candidates
             if _key(c) not in sk and c["courier"] not in uc
             and any(t in low for t in c["t_list"])],
            key=lambda c: -gr(c),
        )
        for cand in pool:
            if _evaluate(selected + [cand], lam, all_t)["F"] < best_F:
                selected = selected + [cand]
                best_F = _evaluate(selected, lam, all_t)["F"]
                improved = True
                break
    return selected, best_F


# ── 启发式主函数 ──────────────────────────────────────────────────────

def _score_heuristic(candidates, lam, all_t, bundle_priority=False) -> List[Dict]:
    selected = _coverage_greedy(candidates)[:]
    best_F = _evaluate(selected, lam, all_t)["F"]
    bonus = BUNDLE_BONUS if bundle_priority else 1.0

    # 阶段1：删除高成本项
    improved, max_iter = True, 30
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for item in sorted(selected, key=lambda x: -(x["score"] * x["willingness"])):
            trial = [c for c in selected if _key(c) != _key(item)]
            if _evaluate(trial, lam, all_t)["F"] < best_F:
                selected = trial
                best_F = _evaluate(selected, lam, all_t)["F"]
                improved = True
                break

    # 阶段2：冗余补入（专攻低概率任务）
    selected, best_F = _add_targeted(selected, best_F, candidates, lam, all_t, bundle_priority)

    # 阶段3：swap
    selected, best_F = _swap_phase(
        selected, best_F, candidates, lam, all_t,
        sel_key=lambda x: -(x["score"] * x["willingness"]),
        cand_key=lambda x: -(x["willingness"] / x["score"]) * (bonus if x["bundle_size"] >= 2 else 1.0),
    )
    return selected


def _coverage_heuristic(candidates, lam, all_t, bundle_priority=False) -> List[Dict]:
    selected = _score_greedy(candidates)[:]
    best_F = _evaluate(selected, lam, all_t)["F"]
    bonus = BUNDLE_BONUS if bundle_priority else 1.0

    # 阶段1：补入高性价比项
    by_ratio = sorted(candidates,
                      key=lambda x: -(x["willingness"] / x["score"]) * (bonus if x["bundle_size"] >= 2 else 1.0))
    improved, max_iter = True, 30
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        uc = {c["courier"] for c in selected}
        sk = {_key(c) for c in selected}
        for cand in by_ratio:
            if _key(cand) in sk or cand["courier"] in uc:
                continue
            if _evaluate(selected + [cand], lam, all_t)["F"] < best_F:
                selected = selected + [cand]
                best_F = _evaluate(selected, lam, all_t)["F"]
                improved = True
                break

    # 阶段2：冗余补入
    selected, best_F = _add_targeted(selected, best_F, candidates, lam, all_t, bundle_priority)

    # 阶段3：swap
    selected, best_F = _swap_phase(
        selected, best_F, candidates, lam, all_t,
        sel_key=lambda x: x["willingness"],
        cand_key=lambda x: -(x["willingness"] / x["score"]) * (bonus if x["bundle_size"] >= 2 else 1.0),
    )
    return selected


# ── 输出转换 ──────────────────────────────────────────────────────────

def _to_output(selected: List[Dict]) -> list:
    return [(c["task_key"], [c["courier"]]) for c in selected]


# ── 主函数 ────────────────────────────────────────────────────────────

def solve(input_text: str) -> list:
    lam = LAM
    candidates = _parse(input_text)
    if not candidates:
        return []
    all_t = _all_tasks(candidates)

    results = {}

    sol = _score_greedy(candidates)
    results["score_greedy"] = (sol, _evaluate(sol, lam, all_t)["F"])

    sol = _coverage_greedy(candidates)
    results["coverage_greedy"] = (sol, _evaluate(sol, lam, all_t)["F"])

    sol = _score_heuristic(candidates, lam, all_t)
    results["score_heuristic"] = (sol, _evaluate(sol, lam, all_t)["F"])

    sol = _coverage_heuristic(candidates, lam, all_t)
    results["coverage_heuristic"] = (sol, _evaluate(sol, lam, all_t)["F"])

    sol = _score_heuristic(candidates, lam, all_t, bundle_priority=True)
    f = _evaluate(sol, lam, all_t)["F"]
    results["bundle_heuristic"] = (sol, f * BUNDLE_F_DISCOUNT)

    best_name = min(results, key=lambda k: results[k][1])
    return _to_output(results[best_name][0])
