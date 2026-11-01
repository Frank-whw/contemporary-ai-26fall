"""通用 A* 搜索。

- 终点只在出队时确认，生成时不返回；
- closed 用 set，同一状态只展开一次，不重开；
- 队列里同一状态的旧条目不在入队时删除，而是在出队时发现已在 closed 中就丢弃（与"展开后丢弃同状态节点"等价）。
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field

State = Hashable
Neighbors = Callable[[State], Iterable[tuple[State, float]]]
Heuristic = Callable[[State], float]

TIE_BREAKS = ("h", "fifo", "lifo")


@dataclass
class SearchStats:
    expanded: int = 0  # 出队并展开的节点数（含终点）
    generated: int = 0  # 生成的后继数
    pushed: int = 0  # 入队次数（含起点）
    stale: int = 0  # 出队时已在 closed 中、被丢弃的旧条目
    goal_pushes: int = 0  # 终点被入队的次数
    max_open: int = 0  # 队列峰值长度
    time_ms: float = 0.0


@dataclass
class SearchResult:
    path: list
    cost: float | None  # None 表示无解
    stats: SearchStats
    trace: list = field(default_factory=list)  # (状态, g, h, f)，按出队顺序；record_trace=True 时记录


def astar(
    start: State,
    goal: State,
    neighbors: Neighbors,
    heuristic: Heuristic,
    tie_break: str = "h",
    record_trace: bool = False,
) -> SearchResult:
    """返回最短路径与统计。tie_break 决定 f 相同时的出队顺序：
    "h" 先出 h 小的（离终点近），"fifo" 先入先出，"lifo" 后入先出。"""
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"unknown tie_break: {tie_break}")
    t0 = time.perf_counter()
    stats = SearchStats()
    counter = itertools.count()

    def key(f: float, h: float) -> tuple:
        n = next(counter)
        if tie_break == "h":
            return (f, h, n)
        if tie_break == "fifo":
            return (f, n)
        return (f, -n)

    g_score = {start: 0}
    parent: dict[State, State | None] = {start: None}
    closed: set[State] = set()
    h0 = heuristic(start)
    open_heap = [(key(h0, h0), start)]
    stats.pushed = 1
    stats.goal_pushes = int(start == goal)
    trace = []

    while open_heap:
        stats.max_open = max(stats.max_open, len(open_heap))
        _, node = heapq.heappop(open_heap)
        if node in closed:
            stats.stale += 1
            continue
        closed.add(node)
        stats.expanded += 1
        g_node = g_score[node]
        if record_trace:
            h = heuristic(node)
            trace.append((node, g_node, h, g_node + h))
        if node == goal:
            path = []
            cur: State | None = node
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            stats.time_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(path[::-1], g_node, stats, trace)
        for nxt, cost in neighbors(node):
            stats.generated += 1
            if nxt in closed:
                continue
            new_g = g_node + cost
            if new_g < g_score.get(nxt, math.inf):
                g_score[nxt] = new_g
                parent[nxt] = node
                h = heuristic(nxt)
                heapq.heappush(open_heap, (key(new_g + h, h), nxt))
                stats.pushed += 1
                stats.goal_pushes += nxt == goal

    stats.time_ms = (time.perf_counter() - t0) * 1000
    return SearchResult([], None, stats, trace)
