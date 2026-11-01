"""统一入口：跑 PDF 给定的测试用例，打印输出、预期与搜索开销。

    python run.py --all
    python run.py --problem 1
    python run.py --problem 4 --depth 2 4
"""

import argparse
import json
import sys

from src import __version__
from src.p1_maze import search_maze
from src.p2_weighted import cost_breakdown, search_terrain
from src.p3_game_tree import GameStats, alphabeta, from_json, minimax
from src.p4_treasure import PLAYERS, build_game, replay, search, treasure_labels
from tests import cases


def check(got, expected) -> str:
    return "PASS" if got == expected else "FAIL"


def run_p1() -> bool:
    print("== 题目一：迷宫寻路 ==")
    ok = True
    for h in ("manhattan", "zero"):
        res = search_maze(cases.P1_START, cases.P1_GOAL, cases.P1_GRID, h)
        s = res.stats
        print(f"  h={h:<9} 展开 {s.expanded:>3}  入队 {s.pushed:>3}  路径长 {len(res.path) - 1}")
        if h == "manhattan":
            print(f"  输出: {res.path}")
            print(f"  预期: {cases.P1_EXPECTED}  [{check(res.path, cases.P1_EXPECTED)}]")
            ok &= res.path == cases.P1_EXPECTED
    return ok


def run_p2() -> bool:
    print("== 题目二：加权地形 ==")
    res = search_terrain(cases.P2_START, cases.P2_GOAL, cases.P2_GRID)
    print(f"  输出: {res.cost}   预期: {cases.P2_EXPECTED}  [{check(res.cost, cases.P2_EXPECTED)}]")
    steps = " -> ".join(f"{r['cell']}(+{r['step_cost']}={r['cumulative']})" for r in cost_breakdown(res.path, cases.P2_GRID))
    print(f"  路径: {steps}")
    for h in ("admissible", "manhattan", "zero"):
        r = search_terrain(cases.P2_START, cases.P2_GOAL, cases.P2_GRID, h)
        print(f"  h={h:<10} 代价 {r.cost}  展开 {r.stats.expanded:>3}  入队 {r.stats.pushed:>3}")
    return res.cost == cases.P2_EXPECTED


def run_p3() -> bool:
    print("== 题目三：极大极小博弈树 ==")
    ok = True
    for name, root in (("JSON 输入", from_json(cases.P3_JSON)), ("对象构建", cases.p3_object_tree())):
        ab, mm = GameStats(), GameStats()
        v_ab, v_mm = alphabeta(root, stats=ab), minimax(root, stats=mm)
        print(
            f"  {name}: Alpha-Beta={v_ab}  Minimax={v_mm}  预期={cases.P3_EXPECTED}  [{check(v_ab, cases.P3_EXPECTED)}]"
            f"  叶子 {ab.leaves}/{mm.leaves}  剪枝 {ab.cutoffs} 次"
        )
        ok &= v_ab == v_mm == cases.P3_EXPECTED
    return ok


def run_p4(depths) -> bool:
    print("== 题目四：竞争环境下的宝藏分配 ==")
    game, dist, _ = build_game()
    labels = list(PLAYERS) + treasure_labels(game.values)
    print("  距离矩阵（A* 计算）:")
    print("        " + "".join(f"{b:>4}" for b in labels))
    for a in labels:
        print(f"    {a:>3} " + "".join(f"{dist[a, b]:>4}" for b in labels))
    ok = True
    for depth in depths:
        for algorithm in ("minimax", "alphabeta"):
            res = search(game, depth, algorithm)
            out = {"best_target": res["best_target"], "best_value": res["best_value"]}
            expected = cases.P4_EXPECTED.get(depth)
            tag = check(out, expected) if expected else "无预期"
            s = res["stats"]
            print(f"  H={depth} {algorithm:<9} {json.dumps(out)}  [{tag}]  节点 {s.nodes:>3}  叶子 {s.leaves:>3}  剪枝 {s.cutoffs}")
            if expected:
                ok &= out == expected
        pv = " ".join(f"{r['player']}→{r['target']}({r['gain']:+})" for r in replay(game, res["pv"]))
        print(f"       主变例: {pv}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=f"实验二 搜索算法 v{__version__}")
    parser.add_argument("--problem", type=int, choices=[1, 2, 3, 4], action="append", help="可重复指定")
    parser.add_argument("--all", action="store_true", help="运行全部题目")
    parser.add_argument("--depth", type=int, nargs="+", default=[2, 4], help="题目四的搜索深度 H")
    args = parser.parse_args()
    problems = [1, 2, 3, 4] if args.all or not args.problem else args.problem
    runners = {1: run_p1, 2: run_p2, 3: run_p3, 4: lambda: run_p4(args.depth)}
    results = {}
    for p in problems:
        results[p] = runners[p]()
        print()
    print("汇总: " + "  ".join(f"题目{p} {'PASS' if v else 'FAIL'}" for p, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
