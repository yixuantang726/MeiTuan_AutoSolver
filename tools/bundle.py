"""
tools/bundle.py
M-Agent 专用工具：在当前方案基础上查找合单机会
目标：找到能释放骑手槽位、同时覆盖未覆盖T的合单方案
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from core import load_data, evaluate, get_uncovered_tasks


def find_bundle_options(
    current_selected: List[Dict],
    lam: float = 30.0,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    M-Agent工具：找出能释放骑手槽位并覆盖未覆盖T的合单方案

    逻辑：
    1. 找出当前未覆盖的T
    2. 在候选池中找包含这些T的合单行（bundle_size>=2）
    3. 检查这些合单行的骑手是否还空闲
    4. 按"覆盖未覆盖T数/score"排序，返回最优选项

    返回：可行合单方案列表，供M-Agent推理是否值得替换
    """
    candidates    = load_data()
    uncovered     = set(get_uncovered_tasks(current_selected))
    used_couriers = {c["courier"] for c in current_selected}
    used_tasks    = {t for c in current_selected for t in c["t_list"]}

    if not uncovered:
        return {
            "uncovered_tasks": [],
            "bundle_options":  [],
            "note": "当前方案已覆盖所有T，无需M-Agent介入"
        }

    # 找候选：合单行 + 包含至少一个未覆盖T + 骑手未被占用
    options = []
    for c in candidates:
        if c["bundle_size"] < 2:
            continue
        if c["courier"] in used_couriers:
            continue
        covered_uncovered = [t for t in c["t_list"] if t in uncovered]
        if not covered_uncovered:
            continue
        # 这个合单行能覆盖哪些原本已覆盖的T（需要释放）
        already_covered = [t for t in c["t_list"] if t in used_tasks]

        options.append({
            "task_key":          c["task_key"],
            "courier":           c["courier"],
            "score":             c["score"],
            "willingness":       c["willingness"],
            "bundle_size":       c["bundle_size"],
            "covers_uncovered":  covered_uncovered,   # 能覆盖的未覆盖T
            "needs_to_free":     already_covered,      # 需要释放的已占用T
            "roi":               round(len(covered_uncovered) / c["score"], 4),
        })

    # 按ROI降序排列
    options.sort(key=lambda x: -x["roi"])
    top_options = options[:top_n]

    return {
        "uncovered_tasks":    sorted(uncovered),
        "uncovered_count":    len(uncovered),
        "bundle_options":     top_options,
        "note": (
            "选择某个bundle_option时，needs_to_free中的T需要找新骑手覆盖，"
            "请确认释放后净收益（减少penalty）大于额外成本"
        )
    }


def find_winner_bundle_options(lam: float = 30.0, top_n: int = 5) -> Dict[str, Any]:
    """
    J-Agent工具：自动对本轮胜者方案查找合单机会
    无需手动传入selected，自动从_session取F更低的方案
    """
    from tools.evaluator import _session, compare_solutions
    from core import evaluate as _evaluate

    c_sol = _session.get("c_solution")
    s_sol = _session.get("s_solution")
    if c_sol is None or s_sol is None:
        return {"error": "方案尚未生成"}

    c_F = _evaluate(c_sol, lam)["F"]
    s_F = _evaluate(s_sol, lam)["F"]
    winner_sol = c_sol if c_F < s_F else s_sol

    return find_bundle_options(winner_sol, lam=lam, top_n=top_n)
