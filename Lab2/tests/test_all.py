"""全部断言。可直接运行：python tests/test_all.py（也兼容 pytest）。

除给定用例外，用不依赖 A* 的 BFS / Dijkstra 和穷举 Minimax 作为独立参照，在随机输入上核对结果。
"""

import heapq
import math
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.grid import MOVES, cost_to_goal, make_heuristic  # noqa: E402
from src.p1_maze import parse_maze, search_maze, solve_maze  # noqa: E402
from src.p2_weighted import parse_terrain, search_terrain, solve_terrain  # noqa: E402
from src.p3_game_tree import (  # noqa: E402
    ORDERS,
    GameStats,
    alphabeta,
    best_score,
    from_json,
    knuth_moore_leaves,
    make_order,
    minimax,
    random_tree,
)
from src.p4_treasure import (  # noqa: E402
    MAP,
    MOVE_ORDERS,
    TreasureGame,
    build_game,
    distance_matrix,
    parse_map,
    random_treasure_map,
    search,
    solve,
    treasure_labels,
)
from tests import cases  # noqa: E402


def bfs_len(grid, start, goal):
    maze = parse_maze(grid)
    if not (maze.passable(start) and maze.passable(goal)):
        return None
    dist, queue = {start: 0}, deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return dist[cell]
        for nxt, _ in maze.neighbors(cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return None


def dijkstra(grid_obj, start, goal):
    dist, heap = {start: 0}, [(0, start)]
    while heap:
        d, cell = heapq.heappop(heap)
        if cell == goal:
            return d
        if d > dist[cell]:
            continue
        r, c = cell
        for dr, dc in MOVES:
            nxt = (r + dr, c + dc)
            if grid_obj.passable(nxt):
                nd = d + grid_obj.cost(nxt)
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    heapq.heappush(heap, (nd, nxt))
    return None


def random_maze(rng, n, density):
    grid = [["X" if rng.random() < density else "." for _ in range(n)] for _ in range(n)]
    grid[0][0], grid[n - 1][n - 1] = "S", "E"
    return grid


def random_terrain(rng, n):
    grid = [[rng.choice([1, 1, 1, 3, 3, 8, "X"]) for _ in range(n)] for _ in range(n)]
    grid[0][0], grid[n - 1][n - 1] = "S", "E"
    return grid


def assert_valid_path(path, grid, start, goal):
    maze = parse_maze(grid)
    assert path[0] == start and path[-1] == goal, path
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)
        assert maze.passable(b), b


def test_p1_given_case():
    assert solve_maze(cases.P1_START, cases.P1_GOAL, cases.P1_GRID) == cases.P1_EXPECTED


def test_p1_both_heuristics_give_shortest_length():
    for h in ("zero", "manhattan"):
        path = search_maze(cases.P1_START, cases.P1_GOAL, cases.P1_GRID, h).path
        assert len(path) == len(cases.P1_EXPECTED), (h, path)


def test_p1_no_path_and_edge_cases():
    walled = [["S", "X", "."], ["X", "X", "."], [".", ".", "E"]]
    assert solve_maze((0, 0), (2, 2), walled) == []
    assert solve_maze((0, 0), (0, 0), cases.P1_GRID) == [(0, 0)]
    assert solve_maze((0, 3), (3, 4), cases.P1_GRID) == []  # 起点在障碍上
    assert solve_maze((0, 0), (9, 9), cases.P1_GRID) == []  # 终点越界


def test_p1_random_mazes_match_bfs():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(2, 12)
        grid = random_maze(rng, n, rng.choice([0.0, 0.2, 0.35]))
        start, goal = (0, 0), (n - 1, n - 1)
        ref = bfs_len(grid, start, goal)
        for h in ("zero", "manhattan"):
            for tb in ("h", "fifo", "lifo"):
                path = search_maze(start, goal, grid, h, tb).path
                if ref is None:
                    assert path == []
                else:
                    assert len(path) - 1 == ref, (h, tb, grid)
                    assert_valid_path(path, grid, start, goal)


def test_p2_given_case():
    assert solve_terrain(cases.P2_START, cases.P2_GOAL, cases.P2_GRID) == cases.P2_EXPECTED
    path = search_terrain(cases.P2_START, cases.P2_GOAL, cases.P2_GRID).path
    assert path == cases.P2_EXPECTED_PATH


def test_p2_no_path():
    grid = [["S", "X", 1], ["X", "X", 1], [1, 1, "E"]]
    assert solve_terrain((0, 0), (2, 2), grid) == -1


def test_p2_admissible_heuristic_never_overestimates():
    rng = random.Random(2)
    for _ in range(200):
        n = rng.randint(2, 10)
        grid = random_terrain(rng, n)
        terrain = parse_terrain(grid)
        goal = (n - 1, n - 1)
        if not terrain.passable(goal):
            continue
        true = cost_to_goal(terrain, goal)
        h = make_heuristic(terrain, goal, "admissible")
        for cell, h_star in true.items():
            assert h(cell) <= h_star + 1e-9, (cell, h(cell), h_star)
            for nxt, step in terrain.neighbors(cell):
                assert h(cell) <= step + h(nxt) + 1e-9, "不一致"


def test_p2_naive_manhattan_overestimates_next_to_e():
    terrain = parse_terrain(cases.P2_GRID)
    true = cost_to_goal(terrain, cases.P2_GOAL)
    h = make_heuristic(terrain, cases.P2_GOAL, "manhattan")
    assert h((4, 3)) == 1 and true[(4, 3)] == 0
    assert max(h(c) - v for c, v in true.items()) == 1


def test_p2_random_terrain_matches_dijkstra():
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(2, 12)
        grid = random_terrain(rng, n)
        terrain = parse_terrain(grid)
        start, goal = (0, 0), (n - 1, n - 1)
        ref = dijkstra(terrain, start, goal)
        for h in ("zero", "manhattan", "admissible"):
            cost = search_terrain(start, goal, grid, h).cost
            assert cost == ref, (h, cost, ref, grid)
        w = search_terrain(start, goal, grid, "weighted", weight=2.0).cost
        assert (w is None) == (ref is None)
        if ref is not None:
            assert ref <= w <= 2 * ref + 1e-9


def test_p3_given_json():
    assert best_score(from_json(cases.P3_JSON)) == cases.P3_EXPECTED


def test_p3_given_object_tree():
    assert best_score(cases.p3_object_tree()) == cases.P3_EXPECTED
    assert minimax(cases.p3_object_tree()) == cases.P3_EXPECTED


def test_p3_alphabeta_equals_minimax_for_every_order():
    rng = random.Random(4)
    for _ in range(200):
        root = random_tree(rng.randint(2, 4), rng.randint(1, 5), rng, leaf_value=lambda r: r.randint(0, 9))
        ref = minimax(root)
        for kind in ORDERS:
            assert alphabeta(root, order=make_order(kind, root)) == ref, kind


def test_p3_best_order_hits_knuth_moore_bound():
    rng = random.Random(5)
    for b in (2, 3, 4):
        for d in range(1, 7):
            root = random_tree(b, d, rng)
            stats = GameStats()
            alphabeta(root, stats=stats, order=make_order("best", root))
            assert stats.leaves == knuth_moore_leaves(b, d), (b, d, stats.leaves)


def test_p3_no_order_beats_knuth_moore_bound():
    rng = random.Random(6)
    for _ in range(50):
        b, d = rng.randint(2, 4), rng.randint(1, 6)
        root = random_tree(b, d, rng)
        for kind in ORDERS:
            stats = GameStats()
            alphabeta(root, stats=stats, order=make_order(kind, root))
            assert knuth_moore_leaves(b, d) <= stats.leaves <= b**d, (kind, b, d, stats.leaves)


def test_p4_given_cases():
    for depth, expected in cases.P4_EXPECTED.items():
        for algorithm in ("minimax", "alphabeta", "minimax_memo"):
            game, _, _ = build_game()
            res = search(game, depth, algorithm)
            got = {"best_target": res["best_target"], "best_value": res["best_value"]}
            assert got == expected, (depth, algorithm, got)
        assert solve(depth) == expected


def test_p4_distance_matrix_symmetric_and_optimal():
    terrain, positions = parse_map(MAP)
    labels = ["A", "B", "T1", "T2", "T3", "T4"]
    dist, _ = distance_matrix(terrain, positions, labels)
    for a in labels:
        for b in labels:
            assert dist[a, b] == dist[b, a], (a, b)
            if a != b:
                assert dist[a, b] == dijkstra(terrain, positions[a], positions[b])


def test_p4_orders_and_depths_agree_with_minimax():
    game, _, _ = build_game()
    for depth in range(1, 5):
        ref = search(game, depth, "minimax")
        for kind in MOVE_ORDERS:
            res = search(game, depth, "alphabeta", kind)
            assert res["best_value"] == ref["best_value"], (depth, kind)


def test_p4_random_maps_agree():
    rng = random.Random(7)
    for _ in range(15):
        k = rng.randint(3, 6)
        grid, values = random_treasure_map(rng, k)
        terrain, positions = parse_map(grid)
        labels = ["A", "B"] + treasure_labels(values)
        dist, _ = distance_matrix(terrain, positions, labels)
        game = TreasureGame(lambda a, b: dist[a, b], values)
        depth = rng.randint(1, k)
        ref = search(game, depth, "minimax")["best_value"]
        assert search(game, depth, "minimax_memo")["best_value"] == ref
        for kind in MOVE_ORDERS:
            assert search(game, depth, "alphabeta", kind)["best_value"] == ref


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
