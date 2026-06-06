"""
debate.py
擂台赛主循环

状态机：
  擂主策略锁定，挑战者每轮换策略
  连续3轮失败 → 调λ
  3轮内赢了   → 更换擂主
"""
import asyncio
import time
from typing import Dict, Any, List

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

from agents.m_agent import create_m_agent
from agents.j_agent import create_j_agent
from core import evaluate, evaluate_with_redundancy
from tools.greedy import run_coverage_greedy, run_score_greedy
from tools.heuristic import (run_score_heuristic, run_coverage_heuristic,
                              run_score_heuristic_bundle, run_coverage_heuristic_bundle)
from tools.ilp import run_coverage_ilp, run_score_ilp
from tools.evaluator import _session, compare_solutions

# ── 策略列表 ─────────────────────────────────────────────────────────
STRATEGIES = ["greedy", "heuristic", "ILP", "bundle_heuristic"]
BUNDLE_F_DISCOUNT = 0.9  # 合单优先方案的F(S)打折后再与其他策略比较


def _run_best_strategy(agent: str, lam: float,
                       champ_metrics=None) -> tuple[list, str]:
    """
    挑战者专用：把所有非greedy策略全部跑一遍，返回(F最低的解, 对应策略名)
    """
    best_F   = float("inf")
    best_sol = None
    best_name = "greedy"

    for strategy in STRATEGIES:
        if strategy == "greedy":
            continue
        try:
            sol = _run_strategy(agent, strategy, lam, champ_metrics)
            f   = evaluate_with_redundancy(sol, lam)["F"]
            effective_f = f * BUNDLE_F_DISCOUNT if strategy == "bundle_heuristic" else f
            print(f"  [{agent}_Agent] {strategy}: F={f:.3f}" +
                  (f" → 折后={effective_f:.3f}" if strategy == "bundle_heuristic" else ""))
            if effective_f < best_F:
                best_F   = effective_f
                best_sol = sol
                best_name = strategy
        except Exception as e:
            print(f"  [{agent}_Agent] {strategy}: 失败({e})")
            continue

    # 如果所有非greedy都失败，退化为greedy
    if best_sol is None:
        best_sol  = _run_strategy(agent, "greedy", lam)
        best_name = "greedy"

    return best_sol, best_name

DELTA_COVERAGE = 3.0   # ILP挑战者覆盖率须超过擂主多少
SCORE_RATIO    = 0.85  # ILP挑战者成本须压到擂主R(S)的多少比例


def _run_strategy(agent: str, strategy: str, lam: float,
                  champ_metrics: Dict[str, Any] = None) -> List[Dict]:
    """
    执行指定策略，返回solution。
    champ_metrics：擂主本轮指标，ILP需要用它计算约束值。
    """
    if agent == "C":
        base = run_coverage_greedy(lam)["selected"]
    else:
        base = run_score_greedy(lam)["selected"]

    if strategy == "greedy":
        return base

    if strategy == "heuristic":
        return run_coverage_heuristic(lam)["selected"] if agent == "C" \
               else run_score_heuristic(lam)["selected"]

    if strategy == "ILP":
        if agent == "C" and champ_metrics:
            c_min  = champ_metrics["coverage"] + DELTA_COVERAGE
            result = run_coverage_ilp(coverage_lower_bound=c_min, lam=lam)
            return base if "error" in result else result["selected"]
        if agent == "S" and champ_metrics:
            s_max  = champ_metrics["expected_score"] * SCORE_RATIO
            result = run_score_ilp(score_upper_bound=s_max, lam=lam)
            return base if "error" in result else result["selected"]
        return base

    if strategy == "bundle_heuristic":
        return run_coverage_heuristic_bundle(lam)["selected"] if agent == "C" \
               else run_score_heuristic_bundle(lam)["selected"]

    return base


# ── 擂台赛状态 ───────────────────────────────────────────────────────
class ArenaState:
    def __init__(self, lam: float = 30.0):
        self.lam = lam
        self.round = 0
        self.consecutive_losses = 1        # greedy预设已判断，C连败×1
        self.global_best_F = float("inf")
        self.global_best_solution = None
        self.champion_solution = None      # 擂主锁定的赢时解，每轮直接复用

        # 初始：S为擂主（第0轮greedy对比已确定），C从非greedy策略开始
        self.champion = "S"
        self.champion_strategy = "greedy"
        self.challenger = "C"
        self.challenger_strategy_idx = 0   # 下行会覆盖
        self.history: List[Dict] = []
        self.next_challenger_strategy()    # 跳过greedy，直接选非greedy起始策略

    @property
    def challenger_strategy(self) -> str:
        return STRATEGIES[self.challenger_strategy_idx % len(STRATEGIES)]

    def next_challenger_strategy(self):
        """greedy输了之后，从非greedy策略里随机选下一个"""
        import random
        non_greedy = [i for i, s in enumerate(STRATEGIES) if s != "greedy"]
        self.challenger_strategy_idx = random.choice(non_greedy)

    def switch_champion(self, winning_solution):
        """更换擂主，并锁定赢时的解"""
        self.champion, self.challenger = self.challenger, self.champion
        self.champion_strategy = STRATEGIES[
            (self.challenger_strategy_idx) % len(STRATEGIES)
        ]
        self.champion_solution = winning_solution  # 锁定赢时的解，下轮直接复用
        self.consecutive_losses = 0      # 新挑战者从0开始计连败
        self.next_challenger_strategy()  # 跳过greedy，与初始设定一致

    def adjust_lambda(self):
        """连败3轮后调λ"""
        if self.champion == "S":
            # S一直赢说明λ太低，coverage惩罚不足
            self.lam = min(self.lam * 1.3, 100)
        else:
            # C一直赢说明λ太高，score无法竞争
            self.lam = max(self.lam * 0.8, 10)
        self.challenger_strategy_idx = 0
        self.consecutive_losses = 0

    def record(self, winner: str, c_F: float, s_F: float, best_F: float,
               consecutive_losses: int, event: str = "",
               champion: str = "", champion_strategy: str = "",
               challenger: str = "", challenger_strategy: str = "",
               lam: float = None):
        self.history.append({
            "round":               self.round,
            "champion":            champion or self.champion,
            "champion_strategy":   champion_strategy or self.champion_strategy,
            "challenger":          challenger or self.challenger,
            "challenger_strategy": challenger_strategy or self.challenger_strategy,
            "winner":              winner,
            "C_F":                 c_F,
            "S_F":                 s_F,
            "best_F":              best_F,
            "lambda":              lam if lam is not None else self.lam,
            "consecutive_losses":  consecutive_losses,
            "event":               event,
        })


def _m_agent_selector(messages) -> str | None:
    """
    只有J-Agent明确说'M_Agent请介入'时才选M_Agent；
    其他所有情况强制选J_Agent，彻底禁止LLM自行选M_Agent。
    """
    if not messages:
        return "J_Agent"
    last = messages[-1]
    source  = getattr(last, "source", "")
    content = getattr(last, "content", "") or ""
    if source == "J_Agent" and "M_Agent请介入" in content:
        return "M_Agent"
    if source == "M_Agent":
        return "J_Agent"
    return "J_Agent"


def build_team(model_client: ChatCompletionClient) -> SelectorGroupChat:
    m_agent = create_m_agent(model_client)
    j_agent = create_j_agent(model_client)

    termination = TextMentionTermination("TERMINATE", sources=["J_Agent"]) | MaxMessageTermination(6)
    return SelectorGroupChat(
        participants=[j_agent, m_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_func=_m_agent_selector,
    )


async def run_debate(
    model_client: ChatCompletionClient,
    max_rounds: int = 10,
    time_limit: float = 9.0,
    verbose: bool = True,
) -> Dict[str, Any]:

    state = ArenaState(lam=30.0)
    start_time = time.time()

    print("=" * 60)
    print("AutoSolver 擂台赛开始")
    print(f"初始λ={state.lam}，最多{max_rounds}轮，时限{time_limit}秒")
    print("=" * 60)

    # ── 第0轮：greedy基线（不调LLM，结果确定，直接计算）─────────────
    c_g_sol = run_coverage_greedy(state.lam)["selected"]
    s_g_sol = run_score_greedy(state.lam)["selected"]
    c_g_metrics = evaluate_with_redundancy(c_g_sol, state.lam)
    s_g_metrics = evaluate_with_redundancy(s_g_sol, state.lam)
    print(f"\n=== 第0轮（greedy基线，无需LLM裁判）===")
    print(f"  C_coverage_greedy F={c_g_metrics['F']:.1f} | S_score_greedy F={s_g_metrics['F']:.1f}")
    print(f"  → S_greedy F更低，确立S为初始擂主，C连败×1")
    state.global_best_F = s_g_metrics["F"]
    state.global_best_solution = s_g_sol
    state.champion_solution = s_g_sol   # 初始擂主S锁定greedy解
    print(f"  全局最优F初始化={state.global_best_F:.1f}")
    # 将第0轮写入history，供最终迭代历史展示
    state.history.append({
        "round":               0,
        "champion":            "S",
        "champion_strategy":   "greedy",
        "challenger":          "C",
        "challenger_strategy": "greedy",
        "winner":              "S",
        "C_F":                 c_g_metrics["F"],
        "S_F":                 s_g_metrics["F"],
        "best_F":              s_g_metrics["F"],
        "lambda":              state.lam,
        "consecutive_losses":  1,
        "event":               "预设基线",
    })

    for _ in range(max_rounds):
        # 时间检查
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"\n时间到（{elapsed:.1f}s），输出当前最优解")
            break

        state.round += 1
        print(f"\n=== 第{state.round}轮 | λ={state.lam:.1f} | "
              f"擂主={state.champion}_Agent({state.champion_strategy}) | "
              f"挑战者={state.challenger}_Agent(全策略取优) ===")

        # ── 执行双方策略（先擂主，再挑战者）────────────────────────
        # 擂主直接复用锁定的赢时解，挑战者全策略取优
        champ_sol   = state.champion_solution          # 直接复用，不重跑
        champ_metrics = evaluate_with_redundancy(champ_sol, state.lam)

        chall_sol, chall_strat = _run_best_strategy(state.challenger, state.lam,
                                                    champ_metrics=champ_metrics)
        # 记录挑战者本轮实际用的最优策略（换擂主时锁定用）
        state.challenger_strategy_idx = STRATEGIES.index(chall_strat)

        c_sol = champ_sol if state.champion == "C" else chall_sol
        s_sol = champ_sol if state.champion == "S" else chall_sol

        _session["c_solution"] = c_sol
        _session["s_solution"] = s_sol

        # ── 调用J-Agent进行裁判 ───────────────────────────────────
        team = build_team(model_client)
        task = TextMessage(
            content=f"""
第{state.round}轮裁判。C和S方案已由Python计算完毕并存入session。

- 擂主：{state.champion}_Agent（{state.champion_strategy}），F={champ_metrics['F']:.3f}
- 挑战者：{state.challenger}_Agent（{chall_strat}），F={evaluate_with_redundancy(chall_sol, state.lam)['F']:.3f}
- λ={state.lam}

请直接调用 compare_solutions(lam={state.lam}) 输出对比，然后按规则宣布胜者并输出TERMINATE。
            """,
            source="user"
        )

        winner = None
        async for msg in team.run_stream(
            task=[task],
            cancellation_token=CancellationToken()
        ):
            if verbose and hasattr(msg, "source") and hasattr(msg, "content"):
                print(f"\n[{msg.source}] {msg.content}")
                print("-" * 40)
            # 从J-Agent输出里解析胜者（简单字符串匹配）
            if hasattr(msg, "source") and msg.source == "J_Agent":
                if "C_Agent胜" in msg.content:
                    winner = "C"
                elif "S_Agent胜" in msg.content:
                    winner = "S"

        # ── 更新最优解 ────────────────────────────────────────────
        c_metrics = evaluate_with_redundancy(c_sol, state.lam)
        s_metrics = evaluate_with_redundancy(s_sol, state.lam)
        best_this_round = min(c_metrics["F"], s_metrics["F"])
        best_sol_this_round = c_sol if c_metrics["F"] < s_metrics["F"] else s_sol

        if best_this_round < state.global_best_F:
            state.global_best_F = best_this_round
            state.global_best_solution = best_sol_this_round
            print(f"\n✓ 全局最优更新：F={state.global_best_F:.3f}")

        # ── 擂台赛状态更新 ────────────────────────────────────────
        if winner is None:
            # J-Agent未明确输出胜者，用F(S)决定
            winner = "C" if c_metrics["F"] < s_metrics["F"] else "S"

        winner_metrics = c_metrics if winner == "C" else s_metrics
        print(f"  本轮较优解：F={winner_metrics['F']:.3f}, "
              f"coverage={winner_metrics['coverage']:.3f}, "
              f"n_bundle={winner_metrics['n_bundle']}, "
              f"n_couriers={winner_metrics['n_couriers']}")

        # 记录本轮打架时的身份和λ（状态更新前捕获）
        round_champion    = state.champion
        round_champ_strat = state.champion_strategy
        round_challenger  = state.challenger
        round_chall_strat = chall_strat
        round_lam         = state.lam  # 本轮实际使用的λ，记录用

        if winner == state.champion:
            # 挑战者输了
            state.consecutive_losses += 1
            losses = state.consecutive_losses
            print(f"\n擂主{state.champion}_Agent守擂成功，"
                  f"挑战者连败计数={losses}")
            if losses >= 3:
                print("连败3轮，调整λ")
                state.adjust_lambda()
                print(f"新λ={state.lam:.1f}，挑战者下轮全策略重跑")
                event = "λ调整"
            else:
                state.next_challenger_strategy()
                print(f"挑战者下轮换策略：{state.challenger_strategy}")
                event = ""
        else:
            # 挑战者赢了，更换擂主（传入赢时的解锁定）
            state.switch_champion(chall_sol)
            losses = state.consecutive_losses  # = 1
            print(f"\n挑战者{state.challenger}_Agent赢得擂台！更换擂主")
            print(f"新擂主：{state.champion}_Agent({state.champion_strategy})")
            print(f"新挑战者：{state.challenger}_Agent，全策略取优")
            event = "换擂主"

        state.record(winner, c_metrics["F"], s_metrics["F"], state.global_best_F,
                     consecutive_losses=losses, event=event,
                     champion=round_champion, champion_strategy=round_champ_strat,
                     challenger=round_challenger, challenger_strategy=round_chall_strat,
                     lam=round_lam)

    # ── 输出最终结果 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("擂台赛结束")
    print(f"总轮次：{state.round}")
    print(f"全局最优F(S)：{state.global_best_F:.3f}")
    print(f"总耗时：{time.time()-start_time:.1f}s")
    print("=" * 60)

    print("\n── 迭代历史 ──")
    for h in state.history:
        event_tag = f" | [{h['event']}]" if h["event"] else ""
        round_label = f"第{h['round']}轮" if h["round"] > 0 else "第0轮(预设)"
        print(f"  {round_label} | λ={h['lambda']:.1f} | "
              f"擂主={h['champion']}_Agent({h['champion_strategy']}) "
              f"挑战者={h['challenger']}_Agent({h['challenger_strategy']}) | "
              f"C_F={h['C_F']:.1f} S_F={h['S_F']:.1f} | "
              f"胜={h['winner']}_Agent | "
              f"连败={h['consecutive_losses']}{event_tag}")

    return {
        "best_F":       state.global_best_F,
        "best_solution":state.global_best_solution,
        "history":      state.history,
        "rounds":       state.round,
    }
