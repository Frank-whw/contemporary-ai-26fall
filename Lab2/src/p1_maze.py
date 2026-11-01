"""题目一：迷宫寻路。'X' 为障碍，其余格子（'S'、'E'、'.'）可通行，每步代价 1。"""

from __future__ import annotations

from .astar import SearchResult, SearchStats, astar
from .grid import Cell, Grid, make_heuristic

MAZE_HEURISTICS = ("zero", "manhattan")


def parse_maze(grid) -> Grid:
    return Grid([[None if str(v).strip().upper() == "X" else 1 for v in row] for row in grid])


def search_maze(
    start: Cell,
    goal: Cell,
    grid,
    heuristic: str = "manhattan",
    tie_break: str = "h",
    record_trace: bool = False,
) -> SearchResult:
    """heuristic="zero" 时 A* 等价于 Dijkstra（单位代价下也等价于 BFS 的层序）。"""
    if heuristic not in MAZE_HEURISTICS:
        raise ValueError(f"unknown heuristic: {heuristic}")
    maze = parse_maze(grid)
    start, goal = tuple(start), tuple(goal)
    if not (maze.passable(start) and maze.passable(goal)):
        return SearchResult([], None, SearchStats())
    h = make_heuristic(maze, goal, heuristic)
    return astar(start, goal, maze.neighbors, h, tie_break=tie_break, record_trace=record_trace)


def solve_maze(start: Cell, goal: Cell, grid) -> list[Cell]:
    """返回从起点到终点的最短路径坐标序列（含两端）；无解返回 []。"""
    return search_maze(start, goal, grid).path
