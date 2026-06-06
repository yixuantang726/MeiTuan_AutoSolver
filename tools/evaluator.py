"""
tools/evaluator.py
J-Agent 专用工具：F(S)计算 + 未覆盖T检查
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from core import evaluate, evaluate_with_redundancy, get_uncovered_tasks

# 全局缓存：存储本轮C和S的方案，供J-Agent调用
_session: Dict[str, Any] = {
    "c_solution": None,
    "s_solution": None,
    "best_solution": None,
}


def store_solution(agent: str, selected: List[Dict]) -> None:
    """内部函数：存储Agent方案到session"""
    _session[f"{agent}_solution"] = selected


def compare_solutions(lam: float = 30.0) -> Dict[str, Any]:
    """
    J-Agent工具：比较C和S两个方案的F(S)
    返回两个方案的完整指标对比，供J-Agent推理
    """
    c_sol = _session.get("c_solution")
    s_sol = _session.get("s_solution")

    if c_sol is None or s_sol is None:
        return {"error": "C或S方案尚未生成，请先运行贪心工具"}

    c_metrics = evaluate_with_redundancy(c_sol, lam)
    s_metrics = evaluate_with_redundancy(s_sol, lam)

    # 初步判断（供J-Agent参考，不强制）
    if c_metrics["coverage"] > s_metrics["coverage"]:
        preliminary = "C_Agent方案覆盖更多T"
    elif s_metrics["F"] < c_metrics["F"]:
        preliminary = "S_Agent方案F(S)更小"
    else:
        preliminary = "两方案接近，需综合判断"

    return {
        "C_Agent": c_metrics,
        "S_Agent": s_metrics,
        "preliminary": preliminary,
        "note": "J-Agent请综合coverage和F(S)做最终判断"
    }


def check_winner_uncovered(lam: float = 30.0) -> Dict[str, Any]:
    """
    J-Agent工具：自动检查本轮胜者方案的未覆盖T
    无需手动传入selected，自动从_session取F更低的方案
    """
    c_sol = _session.get("c_solution")
    s_sol = _session.get("s_solution")
    if c_sol is None or s_sol is None:
        return {"error": "方案尚未生成"}

    c_F = evaluate_with_redundancy(c_sol, lam)["F"]
    s_F = evaluate_with_redundancy(s_sol, lam)["F"]
    winner = "C" if c_F < s_F else "S"
    winner_sol = c_sol if winner == "C" else s_sol

    uncovered = get_uncovered_tasks(winner_sol)
    used_couriers = {c["courier"] for c in winner_sol}
    remaining_couriers = 80 - len(used_couriers)

    return {
        "winner":                winner,
        "uncovered_tasks":       uncovered,
        "uncovered_count":       len(uncovered),
        "remaining_couriers":    remaining_couriers,
        "worth_patching":        len(uncovered) > 0 and remaining_couriers > 0,
        "note": "如果worth_patching=True，可调用find_winner_bundle_options获取合单方案"
    }
