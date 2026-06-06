"""
core.py
数据加载 + F(S)评估函数
被 tools/ 下的所有工具调用
"""
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any

DATA_PATH = "data/delivery_data_layers.xlsx"
ALL_T_COUNT = 40


def load_data(path: str = DATA_PATH) -> List[Dict[str, Any]]:
    """从xlsx加载候选池，返回候选行列表"""
    df = pd.read_excel(path, sheet_name="层2_候选池")
    candidates = []
    for _, row in df.iterrows():
        t_list = [t.strip() for t in row["task_key"].split(",")]
        candidates.append({
            "task_key":    row["task_key"],
            "t_list":      t_list,
            "bundle_size": int(row["bundle_size"]),
            "courier":     row["courier_id"],
            "score":       float(row["score"]),
            "willingness": float(row["willingness"]),
        })
    return candidates


def evaluate(selected: List[Dict], lam: float = 30.0) -> Dict[str, float]:
    """
    计算F(S) = λ·(40 - C(S)) + R(S)

    C(S) = Σ p(t)，p(t) = 1 - Π(1-wᵢ)  冗余派发时多骑手竞争
    R(S) = Σ sᵢ·wᵢ

    返回：F, coverage, expected_score, penalty, n_couriers, n_bundle
    """
    # 每个T的被接概率（支持冗余派发：同一T可被多行覆盖）
    t_prob_none = defaultdict(lambda: 1.0)
    for c in selected:
        for t in c["t_list"]:
            t_prob_none[t] *= (1 - c["willingness"])

    all_t = [f"T{str(i).zfill(4)}" for i in range(ALL_T_COUNT)]
    coverage      = sum(1 - t_prob_none[t] for t in all_t)
    expected_score = sum(c["score"] * c["willingness"] for c in selected)
    penalty       = lam * (ALL_T_COUNT - coverage)
    F             = expected_score + penalty

    return {
        "F":              round(F, 3),
        "coverage":       round(coverage, 3),
        "expected_score": round(expected_score, 3),
        "penalty":        round(penalty, 3),
        "n_selected":     len(selected),
        "n_bundle":       sum(1 for c in selected if c["bundle_size"] > 1),
        "n_couriers":     len(set(c["courier"] for c in selected)),
    }


def evaluate_with_redundancy(selected: List[Dict], lam: float = 30.0) -> Dict[str, float]:
    """
    支持冗余派发的正确评估函数。

    coverage 计算与 evaluate() 相同：C(S) = Σ (1 - Π(1-wᵢ))

    R(S) 修正：同一个T被多个骑手覆盖时，只有一个骑手会被付钱。
    按"谁接谁付"的顺序模型计算每个T的期望成本：
      E[cost_T] = Σ_i score_i · w_i · Π_{j在i之前}(1-w_j)
    即：第i个骑手接单的概率 = 前面所有人都没接 × 自己接了

    对无冗余派发的方案（greedy/ILP），结果与 evaluate() 完全相同。
    """
    from collections import defaultdict

    # 按任务T分组，每个T收集所有覆盖它的骑手（保持插入顺序）
    t_couriers: Dict[str, List[Dict]] = defaultdict(list)
    for c in selected:
        for t in c["t_list"]:
            t_couriers[t].append(c)

    all_t = [f"T{str(i).zfill(4)}" for i in range(ALL_T_COUNT)]

    # coverage：与原来相同
    coverage = 0.0
    for t in all_t:
        prob_none = 1.0
        for c in t_couriers[t]:
            prob_none *= (1 - c["willingness"])
        coverage += (1 - prob_none)

    # R(S)：每个T只算一次期望成本（顺序模型：前面都拒绝才轮到自己）
    expected_score = 0.0
    for t in all_t:
        couriers = t_couriers[t]
        if not couriers:
            continue
        prob_none_so_far = 1.0
        for c in couriers:
            # 这个骑手接单的概率 = 前面所有人都拒绝 × 自己接受
            expected_score += c["score"] * c["willingness"] * prob_none_so_far
            prob_none_so_far *= (1 - c["willingness"])

    penalty = lam * (ALL_T_COUNT - coverage)
    F       = expected_score + penalty

    return {
        "F":              round(F, 3),
        "coverage":       round(coverage, 3),
        "expected_score": round(expected_score, 3),
        "penalty":        round(penalty, 3),
        "n_selected":     len(selected),
        "n_bundle":       sum(1 for c in selected if c["bundle_size"] > 1),
        "n_couriers":     len(set(c["courier"] for c in selected)),
    }


def get_uncovered_tasks(selected: List[Dict]) -> List[str]:
    """返回未被任何选中行覆盖的T列表"""
    covered = {t for c in selected for t in c["t_list"]}
    all_t   = {f"T{str(i).zfill(4)}" for i in range(ALL_T_COUNT)}
    return sorted(all_t - covered)


def is_better(new: List[Dict], best: List[Dict], lam: float = 30.0) -> bool:
    """接单数首要，F(S)次要"""
    n = evaluate(new, lam)
    b = evaluate(best, lam)
    return n["coverage"] > b["coverage"] or (
        n["coverage"] == b["coverage"] and n["F"] < b["F"]
    )
