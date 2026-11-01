"""题目四：竞争环境下的宝藏分配。

先用 A* 求 A、B 与各宝藏两两之间的最小移动代价（距离矩阵），博弈搜索时只查表，不再重复寻路。
玩家 A（MAX）与 B（MIN）交替选一个尚未获得的宝藏：Gain = V − d(当前位置, 宝藏)，
U = Score_A − Score_B；到达搜索深度或宝藏取完时以当前 U 作为评价值。
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .astar import SearchStats, astar
from .grid import Cell, Grid, make_heuristic
from .p3_game_tree import GameStats

MAP = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
PLAYERS = ("A", "B")
MOVE_ORDERS = ("natural", "reversed", "greedy", "anti_greedy", "best", "worst")


def parse_map(grid) -> tuple[Grid, dict[str, Cell]]:
    """数字为进入代价，'X' 为障碍；A、B、T* 所在格代价为 1，并记录它们的坐标。"""
    costs, positions = [], {}
    for r, row in enumerate(grid):
        cost_row = []
        for c, v in enumerate(row):
            s = str(v).strip().upper()
            if s == "X":
                cost_row.append(None)
            elif s.isdigit():
                cost_row.append(int(s))
            else:
                positions[s] = (r, c)
                cost_row.append(1)
        costs.append(cost_row)
    return Grid(costs), positions


def treasure_labels(values: dict[str, int]) -> list[str]:
    return sorted(values, key=lambda t: int(t[1:]))


def distance_matrix(
    grid: Grid,
    positions: dict[str, Cell],
    labels: list[str],
    heuristic: str = "admissible",
    tie_break: str = "h",
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], SearchStats]]:
    """对每个有序对 (a, b) 跑一次 A*。所有标签格代价都是 1，所以矩阵应当对称（测试里会核对）。"""
    dist, stats = {}, {}
    for a in labels:
        for b in labels:
            if a == b:
                dist[a, b] = 0
                continue
            h = make_heuristic(grid, positions[b], heuristic)
            res = astar(positions[a], positions[b], grid.neighbors, h, tie_break=tie_break)
            dist[a, b] = math.inf if res.cost is None else res.cost
            stats[a, b] = res.stats
    return dist, stats


@dataclass(frozen=True)
class GameState:
    pos: tuple[str, str]  # (A 的位置, B 的位置)，用所在格的标签表示
    remaining: tuple[str, ...]  # 尚未获得的宝藏，按编号排序
    score: tuple[float, float]  # (Score_A, Score_B) 累计收益
    to_move: int  # 当前玩家：0 = A（MAX），1 = B（MIN）

    @property
    def utility(self) -> float:
        return self.score[0] - self.score[1]


class TreasureGame:
    def __init__(self, distance: Callable[[str, str], float], values: dict[str, int]):
        self.distance = distance
        self.values = values
        self.lookups = 0  # 查询 d(p, T) 的次数

    def initial_state(self) -> GameState:
        return GameState(PLAYERS, tuple(treasure_labels(self.values)), (0, 0), 0)

    def children(self, state: GameState) -> list[tuple[str, float, GameState]]:
        """(宝藏, 本步收益, 下一状态)，按 remaining 的顺序生成。"""
        p = state.to_move
        out = []
        for t in state.remaining:
            self.lookups += 1
            gain = self.values[t] - self.distance(state.pos[p], t)
            pos = (t, state.pos[1]) if p == 0 else (state.pos[0], t)
            score = (state.score[0] + gain, state.score[1]) if p == 0 else (state.score[0], state.score[1] + gain)
            remaining = tuple(x for x in state.remaining if x != t)
            out.append((t, gain, GameState(pos, remaining, score, 1 - p)))
        return out


def _terminal(state: GameState, depth: int) -> bool:
    return depth == 0 or not state.remaining


def minimax(game: TreasureGame, state: GameState, depth: int, stats: GameStats) -> tuple[float, list[str]]:
    """返回 (值, 主变例)。同值时保留先访问到的子节点。"""
    stats.nodes += 1
    if _terminal(state, depth):
        stats.leaves += 1
        return state.utility, []
    is_max = state.to_move == 0
    best_v, best_pv = None, []
    for target, _, child in game.children(state):
        v, pv = minimax(game, child, depth - 1, stats)
        if best_v is None or (v > best_v if is_max else v < best_v):
            best_v, best_pv = v, [target] + pv
    return best_v, best_pv


def alphabeta(
    game: TreasureGame,
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    stats: GameStats,
    order=None,
) -> tuple[float, list[str]]:
    stats.nodes += 1
    if _terminal(state, depth):
        stats.leaves += 1
        return state.utility, []
    is_max = state.to_move == 0
    children = game.children(state)
    if order:
        children = order(state, children, depth)
    value, best_pv = (-math.inf if is_max else math.inf), []
    for i, (target, _, child) in enumerate(children):
        v, pv = alphabeta(game, child, depth - 1, alpha, beta, stats, order)
        if v > value if is_max else v < value:
            value, best_pv = v, [target] + pv
        if is_max:
            alpha = max(alpha, value)
        else:
            beta = min(beta, value)
        if alpha >= beta:
            rest = len(children) - i - 1
            if rest:
                stats.cutoffs += 1
                stats.pruned += rest
            break
    return value, best_pv


class FutureValue:
    """从某状态起往后能拿到的 U 增量（与已累计的分数无关），按 (位置, 剩余宝藏, 当前玩家, 深度) 记忆。
    不同走法到达同一局面时直接复用，即置换表。"""

    def __init__(self, game: TreasureGame, stats: GameStats | None = None):
        self.game = game
        self.stats = stats
        self.memo: dict[tuple, float] = {}

    def __call__(self, state: GameState, depth: int) -> float:
        key = (state.pos, state.remaining, state.to_move, depth)
        if key in self.memo:
            if self.stats:
                self.stats.cache_hits += 1
            return self.memo[key]
        if self.stats:
            self.stats.nodes += 1  # 只数真正展开的节点，命中缓存的另计
        if _terminal(state, depth):
            if self.stats:
                self.stats.leaves += 1
            v = 0
        else:
            sign = 1 if state.to_move == 0 else -1
            vals = [sign * gain + self(child, depth - 1) for _, gain, child in self.game.children(state)]
            v = max(vals) if state.to_move == 0 else min(vals)
        self.memo[key] = v
        return v


def make_move_order(kind: str, game: TreasureGame):
    """greedy 按当前玩家本步收益从高到低（A 想多拿，B 拿得多就让 U 变小），只用局部信息；
    best/worst 用精确值排序，是理想上下界。"""
    if kind == "natural":
        return None
    if kind == "reversed":
        return lambda state, children, depth: children[::-1]
    if kind == "greedy":
        return lambda state, children, depth: sorted(children, key=lambda ch: -ch[1])
    if kind == "anti_greedy":
        return lambda state, children, depth: sorted(children, key=lambda ch: ch[1])
    if kind in ("best", "worst"):
        oracle = FutureValue(TreasureGame(game.distance, game.values))

        def order(state, children, depth):
            first_max = (state.to_move == 0) == (kind == "best")
            return sorted(children, key=lambda ch: ch[2].utility + oracle(ch[2], depth - 1), reverse=first_max)

        return order
    raise ValueError(f"unknown order: {kind}")


def search(game: TreasureGame, depth: int, algorithm: str = "alphabeta", order: str = "natural") -> dict:
    """在初始局面搜索 depth 层（A、B 各走一步算两层），返回最优首步、最优收益差、主变例与开销。"""
    state = game.initial_state()
    stats = GameStats()
    t0 = time.perf_counter()
    if algorithm == "minimax":
        value, pv = minimax(game, state, depth, stats)
    elif algorithm == "alphabeta":
        value, pv = alphabeta(game, state, depth, -math.inf, math.inf, stats, make_move_order(order, game))
    elif algorithm == "minimax_memo":
        future = FutureValue(game, stats)
        stats.nodes += 1
        value, pv = None, []
        for target, _, child in game.children(state):
            v = child.utility + future(child, depth - 1)
            if value is None or v > value:
                value, pv = v, [target]
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return {
        "best_target": pv[0] if pv else None,
        "best_value": value,
        "pv": pv,
        "stats": stats,
        "time_ms": (time.perf_counter() - t0) * 1000,
    }


def build_game(grid=MAP, values=VALUES, heuristic: str = "admissible") -> tuple[TreasureGame, dict, dict]:
    terrain, positions = parse_map(grid)
    labels = list(PLAYERS) + treasure_labels(values)
    dist, stats = distance_matrix(terrain, positions, labels, heuristic)
    return TreasureGame(lambda a, b: dist[a, b], values), dist, stats


def solve(depth: int, algorithm: str = "alphabeta", grid=MAP, values=VALUES) -> dict:
    """题目要求的输出格式：{"best_target": ..., "best_value": ...}。"""
    game, _, _ = build_game(grid, values)
    res = search(game, depth, algorithm)
    value = res["best_value"]
    return {"best_target": res["best_target"], "best_value": int(value) if float(value).is_integer() else value}


def replay(game: TreasureGame, pv: list[str]) -> list[dict]:
    """沿主变例逐步给出 (玩家, 宝藏, 距离, 本步收益, 累计 U)，用于报告。"""
    state, rows = game.initial_state(), []
    for target in pv:
        player = PLAYERS[state.to_move]
        _, gain, nxt = next(ch for ch in game.children(state) if ch[0] == target)
        rows.append({
            "player": player,
            "from": state.pos[state.to_move],
            "target": target,
            "distance": game.values[target] - gain,
            "gain": gain,
            "utility": nxt.utility,
        })
        state = nxt
    return rows


def random_treasure_map(
    rng: random.Random,
    k: int,
    rows: int = 9,
    cols: int = 9,
    block_p: float = 0.15,
    terrain: tuple[int, ...] = (1, 1, 1, 1, 2, 3, 6, 8, 9),
    value_range: tuple[int, int] = (5, 12),
) -> tuple[list[list[str]], dict[str, int]]:
    """随机地图：障碍概率 block_p，其余格从 terrain 里抽代价；A、B、T1..Tk 互不重叠且两两连通。"""
    labels = list(PLAYERS) + [f"T{i}" for i in range(1, k + 1)]
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    while True:
        grid = [["X" if rng.random() < block_p else str(rng.choice(terrain)) for _ in range(cols)] for _ in range(rows)]
        for label, (r, c) in zip(labels, rng.sample(cells, len(labels))):
            grid[r][c] = label
        g, positions = parse_map(grid)
        seen, queue = {positions["A"]}, deque([positions["A"]])
        while queue:
            for nxt, _ in g.neighbors(queue.popleft()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if all(positions[label] in seen for label in labels):
            values = {f"T{i}": rng.randint(*value_range) for i in range(1, k + 1)}
            return grid, values
