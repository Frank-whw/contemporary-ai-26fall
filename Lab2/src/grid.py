"""网格地图的公共部分：四邻域、进入代价、启发函数。题目一、二、四共用。"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable

Cell = tuple[int, int]

# 上、下、左、右。固定顺序让同代价路径的选择可复现。
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

HEURISTICS = ("zero", "manhattan", "admissible", "weighted")


class Grid:
    """costs[r][c] 是进入该格的代价，None 表示障碍。"""

    def __init__(self, costs: list[list[float | None]]):
        self.costs = costs
        self.rows = len(costs)
        self.cols = len(costs[0]) if costs else 0

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.costs[cell[0]][cell[1]] is not None

    def cost(self, cell: Cell) -> float | None:
        return self.costs[cell[0]][cell[1]]

    def neighbors(self, cell: Cell):
        r, c = cell
        costs, rows, cols = self.costs, self.rows, self.cols
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                w = costs[nr][nc]
                if w is not None:
                    yield (nr, nc), w

    def min_cost(self, exclude: Cell | None = None) -> float:
        vals = [
            w
            for r, row in enumerate(self.costs)
            for c, w in enumerate(row)
            if w is not None and (r, c) != exclude
        ]
        return max(min(vals), 0) if vals else 0


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def make_heuristic(grid: Grid, goal: Cell, kind: str = "admissible", weight: float = 1.0) -> Callable[[Cell], float]:
    """kind:
    - zero：h=0，A* 退化为 Dijkstra；
    - manhattan：h=曼哈顿距离（题目给的提示）；
    - admissible：h = c_min·(M−1) + c_goal（M≥1），终点处为 0。M 步里前 M−1 步至少各花 c_min，最后一步恰好花进入终点的代价；
    - weighted：weight × admissible，即加权 A*。
    """
    gr, gc = goal
    if kind == "zero":
        return lambda cell: 0
    if kind == "manhattan":
        return lambda cell: abs(cell[0] - gr) + abs(cell[1] - gc)
    if kind not in ("admissible", "weighted"):
        raise ValueError(f"unknown heuristic: {kind}")
    c_min = grid.min_cost(exclude=goal)
    c_goal = grid.cost(goal)
    w = weight if kind == "weighted" else 1.0

    def h(cell: Cell) -> float:
        m = abs(cell[0] - gr) + abs(cell[1] - gc)
        return 0 if m == 0 else w * (c_min * (m - 1) + c_goal)

    return h


def cost_to_goal(grid: Grid, goal: Cell) -> dict[Cell, float]:
    """反向 Dijkstra 求每格到终点的真实最小代价 h*，用来检查启发函数是否高估。"""
    dist = {goal: 0}
    heap = [(0, goal)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        nd = d + grid.cost(v)  # 从邻格 u 走进 v 要付 v 的进入代价
        for u, _ in grid.neighbors(v):
            if nd < dist.get(u, math.inf):
                dist[u] = nd
                heapq.heappush(heap, (nd, u))
    return dist
