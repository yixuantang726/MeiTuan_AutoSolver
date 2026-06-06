"""
tools/greedy.py
C-Agent 和 S-Agent 各自的贪心求解工具
纯Python函数，不涉及LLM，可独立测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from core import load_data, evaluate


def _greedy(candidates: List[Dict], sort_key) -> List[Dict]:
    """通用贪心框架：按sort_key排序后依次选不冲突的行"""
    sorted_c = sorted(candidates, key=sort_key)
    used_couriers, used_tasks, selected = set(), set(), []
    for c in sorted_c:
        if c["courier"] in used_couriers:
            continue
        if any(t in used_tasks for t in c["t_list"]):
            continue
        used_couriers.add(c["courier"])
        used_tasks.update(c["t_list"])
        selected.append(c)
    return selected


def run_coverage_greedy(lam: float = 30.0) -> Dict[str, Any]:
    """
    C-Agent工具：按 willingness × bundle_size 降序贪心
    锚定指标：期望覆盖T数（coverage）
    返回方案摘要和评估指标
    """
    candidates = load_data()
    selected = _greedy(
        candidates,
        sort_key=lambda x: -(x["willingness"] * x["bundle_size"])
    )
    metrics = evaluate(selected, lam)
    return {
        "strategy":  "coverage_greedy",
        "metrics":   metrics,
        "n_selected": len(selected),
        "sample":    [
            {
                "task_key":    c["task_key"],
                "courier":     c["courier"],
                "score":       c["score"],
                "willingness": c["willingness"],
                "bundle_size": c["bundle_size"],
            }
            for c in selected[:5]  # 只返回前5行供LLM阅读
        ],
        "selected":  selected,  # 完整方案供后续计算
    }


def run_score_greedy(lam: float = 30.0) -> Dict[str, Any]:
    """
    S-Agent工具：按 score 升序贪心
    锚定指标：总score（期望派单成本）
    返回方案摘要和评估指标
    """
    candidates = load_data()
    selected = _greedy(
        candidates,
        sort_key=lambda x: x["score"]
    )
    metrics = evaluate(selected, lam)
    return {
        "strategy":   "score_greedy",
        "metrics":    metrics,
        "n_selected": len(selected),
        "sample":     [
            {
                "task_key":    c["task_key"],
                "courier":     c["courier"],
                "score":       c["score"],
                "willingness": c["willingness"],
                "bundle_size": c["bundle_size"],
            }
            for c in selected[:5]
        ],
        "selected":   selected,
    }
