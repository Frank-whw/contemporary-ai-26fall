"""题目二：加权地形。从 A 走到相邻 B 的代价 = 进入 B 的地形权值。

按题目说明，'S' 视为平原（代价 1），'E' 视为 0（"最后一步没有代价"），'X' 视为障碍。
"""

from __future__ import annotations

from .astar import SearchResult, SearchStats, astar
from .grid import Cell, Grid, make_heuristic

TERRAIN_HEURISTICS = ("zero", "manhattan", "admissible", "weighted")
SPECIAL_COSTS = {"S": 1, "E": 0}


def cell_cost(value) -> float | None:
    s = str(value).strip().upper()
    if s == "X":
        return None
    if s in SPECIAL_COSTS:
        return SPECIAL_COSTS[s]
    v = float(s)
    return int(v) if v.is_integer() else v


def parse_terrain(grid) -> Grid:
    return Grid([[cell_cost(v) for v in row] for row in grid])


def search_terrain(
    start: Cell,
    goal: Cell,
    grid,
    heuristic: str = "admissible",
    weight: float = 1.0,
    tie_break: str = "h",
    record_trace: bool = False,
) -> SearchResult:
    """默认用修正后的可采纳启发函数 h = c_min·(M−1) + c_E（见 grid.make_heuristic）。"""
    if heuristic not in TERRAIN_HEURISTICS:
        raise ValueError(f"unknown heuristic: {heuristic}")
    terrain = parse_terrain(grid)
    start, goal = tuple(start), tuple(goal)
    if not (terrain.passable(start) and terrain.passable(goal)):
        return SearchResult([], None, SearchStats())
    h = make_heuristic(terrain, goal, heuristic, weight)
    return astar(start, goal, terrain.neighbors, h, tie_break=tie_break, record_trace=record_trace)


def solve_terrain(start: Cell, goal: Cell, grid) -> int:
    """返回最小总代价；无解返回 -1。"""
    cost = search_terrain(start, goal, grid).cost
    if cost is None:
        return -1
    return int(cost) if float(cost).is_integer() else cost


def cost_breakdown(path: list[Cell], grid) -> list[dict]:
    """逐步列出进入每格的代价和累计代价，用于展示代价累计过程。"""
    terrain = parse_terrain(grid)
    rows, total = [], 0
    for i, cell in enumerate(path):
        step = 0 if i == 0 else terrain.cost(cell)
        total += step
        rows.append({"cell": cell, "step_cost": step, "cumulative": total})
    return rows
