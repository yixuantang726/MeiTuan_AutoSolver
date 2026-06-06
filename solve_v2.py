"""
solve_v2.py - 只保留 greedy + ILP，无启发式
"""

from collections import defaultdict
from typing import List, Dict

LAM = 30.0


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
            t_list = [t.strip() for t in task_key.strip().split(",")]
            candidates.append({
                "task_key":    task_key.strip(),
                "t_list":      t_list,
                "bundle_size": len(t_list),
                "courier":     courier_id.strip(),
                "score":       float(score_str),
                "willingness": float(willingness_str),
            })
        except ValueError:
            continue
    return candidates


def _all_tasks(candidates):
    tasks = set()
    for c in candidates:
        tasks.update(c["t_list"])
    return sorted(tasks)


def _evaluate(selected, lam, all_t):
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
    return {"F": round(expected_score + lam * (len(all_t) - coverage), 3)}


def _greedy(candidates, sort_key):
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


def _ilp(candidates, lam, all_t):
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = len(candidates)
    c_obj = np.array([
        c["score"] * c["willingness"] - lam * c["willingness"] * c["bundle_size"]
        for c in candidates
    ], dtype=float)

    rows, lb_l, ub_l = [], [], []

    courier_rows = defaultdict(list)
    for i, c in enumerate(candidates):
        courier_rows[c["courier"]].append(i)
    for idxs in courier_rows.values():
        row = np.zeros(n); row[idxs] = 1.0
        rows.append(row); lb_l.append(-np.inf); ub_l.append(1.0)

    task_rows = defaultdict(list)
    for i, c in enumerate(candidates):
        for t in c["t_list"]:
            task_rows[t].append(i)
    for idxs in task_rows.values():
        row = np.zeros(n); row[idxs] = 1.0
        rows.append(row); lb_l.append(-np.inf); ub_l.append(1.0)

    result = milp(c_obj,
                  constraints=LinearConstraint(np.vstack(rows), lb_l, ub_l),
                  integrality=np.ones(n),
                  bounds=Bounds(lb=0.0, ub=1.0))

    if not result.success:
        return None
    return [candidates[i] for i in range(n) if result.x[i] > 0.5]


def solve(input_text: str) -> list:
    lam = LAM
    candidates = _parse(input_text)
    if not candidates:
        return []
    all_t = _all_tasks(candidates)

    results = {}

    sol = _greedy(candidates, lambda x: x["score"])
    results["score_greedy"] = (sol, _evaluate(sol, lam, all_t)["F"])

    sol = _greedy(candidates, lambda x: -(x["willingness"] * x["bundle_size"]))
    results["coverage_greedy"] = (sol, _evaluate(sol, lam, all_t)["F"])

    try:
        sol = _ilp(candidates, lam, all_t)
        if sol is not None:
            results["ilp"] = (sol, _evaluate(sol, lam, all_t)["F"])
    except Exception:
        pass  # scipy不可用时退化为greedy

    best_name = min(results, key=lambda k: results[k][1])
    best_sol = results[best_name][0]
    return [(c["task_key"], [c["courier"]]) for c in best_sol]
