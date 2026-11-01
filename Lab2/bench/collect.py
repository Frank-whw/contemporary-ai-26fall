"""采集报告所需的全部实验数据，写到 bench/ 下的 CSV / JSON，最后调用 plot.py 出图到 report/figures/。

    python bench/collect.py           # 完整采集
    python bench/collect.py --quick   # 缩小规模，只用来检查流程
"""

import argparse
import csv
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import __version__  # noqa: E402
from src.astar import astar  # noqa: E402
from src.grid import cost_to_goal, make_heuristic  # noqa: E402
from src.p1_maze import search_maze  # noqa: E402
from src.p2_weighted import cost_breakdown, parse_terrain, search_terrain  # noqa: E402
from src.p3_game_tree import (  # noqa: E402
    GameStats,
    alphabeta,
    from_json,
    knuth_moore_leaves,
    make_order,
    minimax,
    random_tree,
)
from src.p4_treasure import (  # noqa: E402
    MAP,
    PLAYERS,
    VALUES,
    TreasureGame,
    build_game,
    distance_matrix,
    minimax as game_minimax,
    parse_map,
    random_treasure_map,
    replay,
    search,
    treasure_labels,
)
from tests import cases  # noqa: E402

OUT = ROOT / "bench"
SEED = 2026

FULL = {
    "p1_sizes": [10, 20, 40, 80, 160],
    "p1_densities": [0.0, 0.1, 0.2, 0.3],
    "p1_seeds": 20,
    "p2_sizes": [10, 20, 40, 80],
    "p2_seeds": 20,
    "p2_weights": [1.25, 1.5, 2.0, 3.0, 5.0],
    "p2_mountain": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    "p3_depths": {2: range(1, 13), 3: range(1, 10), 4: range(1, 8)},
    "p3_seeds": 20,
    "p3_sigmas": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    "p4_k": [4, 5, 6, 7, 8],
    "p4_maps": 20,
    "p4_cache_k": [3, 4, 5, 6],
    "p4_cache_maps": 10,
    "timing_repeats": 2000,
}
QUICK = {
    **FULL,
    "p1_sizes": [10, 20, 40],
    "p1_seeds": 3,
    "p2_sizes": [10, 20],
    "p2_seeds": 3,
    "p3_depths": {2: range(1, 7), 3: range(1, 6), 4: range(1, 5)},
    "p3_seeds": 3,
    "p4_k": [4, 5, 6],
    "p4_maps": 3,
    "p4_cache_k": [3, 4],
    "p4_cache_maps": 2,
    "timing_repeats": 50,
}


def rng_for(*parts) -> random.Random:
    # 字符串种子在 CPython 里走 sha512，跨进程确定
    return random.Random("-".join(map(str, (SEED, *parts))))


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  写入 {name}（{len(rows)} 行）")


def write_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    print(f"  写入 {name}")


def _jsonable(obj):
    if isinstance(obj, float) and math.isinf(obj):
        return "inf" if obj > 0 else "-inf"
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def clean(x):
    """JSON 不支持 inf，换成字符串。"""
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def median_us(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def stats_row(s) -> dict:
    return {
        "expanded": s.expanded,
        "generated": s.generated,
        "pushed": s.pushed,
        "stale": s.stale,
        "goal_pushes": s.goal_pushes,
        "max_open": s.max_open,
        "time_ms": round(s.time_ms, 4),
    }


def q(values, p):
    """分位数（线性插值），p∈[0,1]。"""
    xs = sorted(values)
    if not xs:
        return math.nan
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---------------- 题目一 ----------------


def random_maze(rng, n, density):
    grid = [["X" if rng.random() < density else "." for _ in range(n)] for _ in range(n)]
    grid[0][0], grid[n - 1][n - 1] = "S", "E"
    return grid


def solvable_maze(rng, n, density):
    for attempt in range(1, 10_000):
        grid = random_maze(rng, n, density)
        if search_maze((0, 0), (n - 1, n - 1), grid).path:
            return grid, attempt
    raise RuntimeError("没有生成出连通的迷宫")


def collect_p1(cfg) -> dict:
    print("[题目一]")
    given = {}
    for h in ("zero", "manhattan"):
        res = search_maze(cases.P1_START, cases.P1_GOAL, cases.P1_GRID, h, record_trace=True)
        given[h] = {
            "path": res.path,
            **stats_row(res.stats),
            "time_us_median": median_us(
                lambda h=h: search_maze(cases.P1_START, cases.P1_GOAL, cases.P1_GRID, h), cfg["timing_repeats"]
            ),
            "trace": [{"cell": c, "g": g, "h": hh, "f": f} for c, g, hh, f in res.trace],
        }
    write_json("p1_given.json", given)

    # 一般图上的终止条件演示：S→G 直连代价 10，S→A→G 代价 1+4。若在"生成终点"时就返回会得到 10。
    edges = {"S": [("G", 10), ("A", 1)], "A": [("G", 4)], "G": []}
    demo = astar("S", "G", lambda v: edges[v], lambda v: 0, record_trace=True)
    write_json("p1_termination_demo.json", {
        "edges": edges,
        "first_push_cost_of_goal": 10,
        "path": demo.path,
        "cost": demo.cost,
        "goal_pushes": demo.stats.goal_pushes,
        "pop_order": [{"node": v, "g": g} for v, g, _, _ in demo.trace],
    })

    methods = [("zero", "h"), ("manhattan", "h"), ("manhattan", "fifo"), ("manhattan", "lifo")]
    rows = []
    for n in cfg["p1_sizes"]:
        for density in cfg["p1_densities"]:
            for seed in range(cfg["p1_seeds"]):
                grid, attempts = solvable_maze(rng_for("p1", n, density, seed), n, density)
                for h, tb in methods:
                    res = search_maze((0, 0), (n - 1, n - 1), grid, h, tb)
                    rows.append({
                        "N": n,
                        "density": density,
                        "seed": seed,
                        "heuristic": h,
                        "tie_break": tb,
                        "path_len": len(res.path) - 1,
                        "attempts": attempts,
                        **stats_row(res.stats),
                    })
    write_csv("p1_random.csv", rows)

    n, density = 40, 0.25
    grid, _ = solvable_maze(rng_for("p1-example"), n, density)
    example = {"N": n, "density": density, "grid": ["".join(r).replace("S", ".").replace("E", ".") for r in grid]}
    for h, tb in methods[:3]:
        res = search_maze((0, 0), (n - 1, n - 1), grid, h, tb, record_trace=True)
        example[f"{h}/{tb}"] = {"expanded": [c for c, *_ in res.trace], "path": res.path, "count": res.stats.expanded}
    write_json("p1_example.json", example)
    return {"given": given, "rows": rows}


# ---------------- 题目二 ----------------


def random_terrain(rng, n, probs):
    costs, weights = zip(*probs)
    grid = [[rng.choices(costs, weights)[0] for _ in range(n)] for _ in range(n)]
    grid[0][0], grid[n - 1][n - 1] = "S", "E"
    return grid


TERRAIN = ((1, 0.6), (3, 0.25), (8, 0.15))


def collect_p2(cfg) -> dict:
    print("[题目二]")
    terrain = parse_terrain(cases.P2_GRID)
    h_star = cost_to_goal(terrain, cases.P2_GOAL)
    h_m = make_heuristic(terrain, cases.P2_GOAL, "manhattan")
    h_a = make_heuristic(terrain, cases.P2_GOAL, "admissible")
    given = {"cells": []}
    for r in range(terrain.rows):
        for c in range(terrain.cols):
            cell = (r, c)
            given["cells"].append({"cell": cell, "h_star": h_star[cell], "manhattan": h_m(cell), "admissible": h_a(cell)})
    for h in ("zero", "manhattan", "admissible"):
        res = search_terrain(cases.P2_START, cases.P2_GOAL, cases.P2_GRID, h, record_trace=True)
        given[h] = {
            "cost": res.cost,
            "path": res.path,
            **stats_row(res.stats),
            "time_us_median": median_us(
                lambda h=h: search_terrain(cases.P2_START, cases.P2_GOAL, cases.P2_GRID, h), cfg["timing_repeats"]
            ),
            "trace": [{"cell": cc, "g": g, "h": hh, "f": f} for cc, g, hh, f in res.trace],
        }
    given["breakdown"] = cost_breakdown(given["admissible"]["path"], cases.P2_GRID)
    write_json("p2_given.json", given)

    methods = [("zero", 1.0, "h"), ("manhattan", 1.0, "h"), ("manhattan", 1.0, "fifo"), ("manhattan", 1.0, "lifo"),
               ("admissible", 1.0, "h")] + [("weighted", w, "h") for w in cfg["p2_weights"]]
    rows = []
    for n in cfg["p2_sizes"]:
        for seed in range(cfg["p2_seeds"]):
            grid = random_terrain(rng_for("p2", n, seed), n, TERRAIN)
            opt = None
            for h, w, tb in methods:
                res = search_terrain((0, 0), (n - 1, n - 1), grid, h, w, tb)
                if h == "zero":
                    opt = res.cost
                rows.append({
                    "N": n,
                    "seed": seed,
                    "heuristic": h,
                    "weight": w,
                    "tie_break": tb,
                    "cost": res.cost,
                    "optimal_cost": opt,
                    "suboptimality": res.cost / opt - 1,
                    **stats_row(res.stats),
                })
    write_csv("p2_random.csv", rows)

    sweep = []
    for mountain in cfg["p2_mountain"]:
        probs = ((1, 0.75 * (1 - mountain)), (3, 0.25 * (1 - mountain)), (8, mountain))
        for seed in range(cfg["p2_seeds"]):
            grid = random_terrain(rng_for("p2-sweep", mountain, seed), 40, probs)
            for h in ("zero", "admissible"):
                res = search_terrain((0, 0), (39, 39), grid, h)
                sweep.append({"mountain": mountain, "seed": seed, "heuristic": h, "cost": res.cost, **stats_row(res.stats)})
    write_csv("p2_terrain_sweep.csv", sweep)
    return {"given": given, "rows": rows, "sweep": sweep}


# ---------------- 题目三 ----------------

P3_ORDERS = ["original", "ascending", "descending", "best", "worst"]


def collect_p3(cfg) -> dict:
    print("[题目三]")
    root = from_json(cases.P3_JSON)
    given_rows = []
    mm = GameStats()
    v = minimax(root, stats=mm)
    given_rows.append({"algorithm": "minimax", "order": "original", "value": v, **vars(mm)})
    traces = {}
    for kind in ["original", "reversed", "ascending", "descending", "best", "worst"]:
        st, trace = GameStats(), []
        v = alphabeta(root, stats=st, order=make_order(kind, root), trace=trace)
        given_rows.append({"algorithm": "alphabeta", "order": kind, "value": v, **vars(st)})
        traces[kind] = clean(trace)
    write_csv("p3_given.csv", given_rows)
    write_json("p3_trace.json", traces)

    rows = []
    for b, depths in cfg["p3_depths"].items():
        for d in depths:
            for seed in range(cfg["p3_seeds"]):
                tree = random_tree(b, d, rng_for("p3", b, d, seed))
                st = GameStats()
                minimax(tree, stats=st)
                rows.append({"b": b, "d": d, "seed": seed, "order": "minimax", **vars(st)})
                for kind in P3_ORDERS:
                    st = GameStats()
                    alphabeta(tree, stats=st, order=make_order(kind, tree))
                    rows.append({"b": b, "d": d, "seed": seed, "order": kind, **vars(st)})
    for row in rows:
        row["km_bound"] = knuth_moore_leaves(row["b"], row["d"])
        row["full"] = row["b"] ** row["d"]
    write_csv("p3_random.csv", rows)

    noise = []
    b, d = 3, 8
    for seed in range(cfg["p3_seeds"]):
        tree = random_tree(b, d, rng_for("p3-noise", seed))
        for sigma in cfg["p3_sigmas"]:
            st = GameStats()
            alphabeta(tree, stats=st, order=make_order("noisy", tree, sigma, rng_for("p3-noise-order", seed, sigma)))
            noise.append({"b": b, "d": d, "seed": seed, "sigma": sigma, **vars(st)})
        st = GameStats()
        alphabeta(tree, stats=st)
        noise.append({"b": b, "d": d, "seed": seed, "sigma": "original", **vars(st)})
    write_csv("p3_noise.csv", noise)
    return {"given": given_rows, "rows": rows, "noise": noise}


# ---------------- 题目四 ----------------

P4_METHODS = [("minimax", "natural"), ("minimax_memo", "natural")] + [
    ("alphabeta", o) for o in ("natural", "reversed", "greedy", "anti_greedy", "best", "worst")
]


def collect_p4(cfg) -> dict:
    print("[题目四]")
    terrain, positions = parse_map(MAP)
    labels = list(PLAYERS) + treasure_labels(VALUES)
    matrix, dists = {}, {}
    for heuristic in ("zero", "admissible"):
        dists[heuristic], st = distance_matrix(terrain, positions, labels, heuristic)
        matrix[heuristic] = {
            "expanded_total": sum(s.expanded for s in st.values()),
            "pushed_total": sum(s.pushed for s in st.values()),
            "expanded": {f"{a}-{b}": s.expanded for (a, b), s in st.items()},
        }
    assert dists["zero"] == dists["admissible"]
    dist = dists["admissible"]
    matrix["labels"] = labels
    matrix["positions"] = {k: positions[k] for k in labels}
    matrix["matrix"] = [[dist[a, b] for b in labels] for a in labels]
    matrix["symmetric"] = all(dist[a, b] == dist[b, a] for a in labels for b in labels)
    write_json("p4_distance_matrix.json", matrix)

    game, _, _ = build_game()
    depth_rows, pvs = [], {}
    for depth in range(1, 5):
        for algorithm, order in P4_METHODS:
            res = search(game, depth, algorithm, order)
            t = median_us(lambda: search(game, depth, algorithm, order), max(cfg["timing_repeats"] // 10, 5))
            depth_rows.append({
                "H": depth,
                "algorithm": algorithm,
                "order": order,
                "best_target": res["best_target"],
                "best_value": res["best_value"],
                "pv": " ".join(res["pv"]),
                "time_us_median": round(t, 2),
                **vars(res["stats"]),
            })
        res = search(game, depth, "minimax")
        first_moves = []
        state = game.initial_state()
        for target, gain, child in game.children(state):
            st = GameStats()
            value, pv = game_minimax(game, child, depth - 1, st)
            first_moves.append({"target": target, "gain": gain, "value": value, "reply": replay(game, [target] + pv)})
        pvs[depth] = {"pv": replay(game, res["pv"]), "first_moves": first_moves}
    write_csv("p4_depth.csv", depth_rows)
    write_json("p4_pv.json", pvs)

    rows = []
    for k in cfg["p4_k"]:
        for seed in range(cfg["p4_maps"]):
            grid, values = random_treasure_map(rng_for("p4", k, seed), k)
            t_g, pos_g = parse_map(grid)
            dist, _ = distance_matrix(t_g, pos_g, list(PLAYERS) + treasure_labels(values))
            game_k = TreasureGame(lambda a, b, dist=dist: dist[a, b], values)
            ref = None
            for algorithm, order in P4_METHODS:
                res = search(game_k, k, algorithm, order)
                ref = res["best_value"] if ref is None else ref
                rows.append({
                    "K": k,
                    "seed": seed,
                    "algorithm": algorithm,
                    "order": order,
                    "best_target": res["best_target"],
                    "best_value": res["best_value"],
                    "agrees": res["best_value"] == ref,
                    "time_ms": round(res["time_ms"], 3),
                    **vars(res["stats"]),
                })
    write_csv("p4_random.csv", rows)

    cache_rows = []
    for k in cfg["p4_cache_k"]:
        for seed in range(cfg["p4_cache_maps"]):
            grid, values = random_treasure_map(rng_for("p4-cache", k, seed), k)
            t_g, pos_g = parse_map(grid)
            labels_k = list(PLAYERS) + treasure_labels(values)
            calls = {"n": 0}

            def no_cache(a, b, t_g=t_g, pos_g=pos_g):
                calls["n"] += 1
                return astar(pos_g[a], pos_g[b], t_g.neighbors, make_heuristic(t_g, pos_g[b])).cost

            t0 = time.perf_counter()
            v_no = search(TreasureGame(no_cache, values), k, "minimax")["best_value"]
            t_no = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dist, _ = distance_matrix(t_g, pos_g, labels_k)
            v_c = search(TreasureGame(lambda a, b, dist=dist: dist[a, b], values), k, "minimax")["best_value"]
            t_c = (time.perf_counter() - t0) * 1000
            cache_rows.append({
                "K": k,
                "seed": seed,
                "astar_calls_nocache": calls["n"],
                "astar_calls_cache": len(labels_k) * (len(labels_k) - 1),
                "time_ms_nocache": round(t_no, 3),
                "time_ms_cache": round(t_c, 3),
                "agrees": v_no == v_c,
            })
    write_csv("p4_cache.csv", cache_rows)
    return {"matrix": matrix, "depth": depth_rows, "pvs": pvs, "rows": rows, "cache": cache_rows}


# ---------------- 汇总 ----------------


def summarize(p1, p2, p3, p4) -> dict:
    out = {}

    groups = defaultdict(list)
    for r in p1["rows"]:
        groups[r["N"], r["density"], r["heuristic"], r["tie_break"]].append(r)
    p1_sum = []
    for (n, density, h, tb), rs in sorted(groups.items()):
        zero = {r["seed"]: r["expanded"] for r in groups[n, density, "zero", "h"]}
        p1_sum.append({
            "N": n,
            "density": density,
            "method": f"{h}/{tb}",
            "expanded_median": q([r["expanded"] for r in rs], 0.5),
            "expanded_q25": q([r["expanded"] for r in rs], 0.25),
            "expanded_q75": q([r["expanded"] for r in rs], 0.75),
            "ratio_to_zero_median": q([r["expanded"] / zero[r["seed"]] for r in rs], 0.5),
            "time_ms_median": q([r["time_ms"] for r in rs], 0.5),
            "path_len_median": q([r["path_len"] for r in rs], 0.5),
            "attempts_mean": statistics.mean(r["attempts"] for r in rs),
        })
    out["p1"] = p1_sum

    groups = defaultdict(list)
    for r in p2["rows"]:
        groups[r["N"], r["heuristic"], r["weight"], r["tie_break"]].append(r)
    p2_sum = []
    for (n, h, w, tb), rs in sorted(groups.items()):
        adm = {r["seed"]: r["expanded"] for r in groups[n, "admissible", 1.0, "h"]}
        p2_sum.append({
            "N": n,
            "method": f"{h}/w={w}/{tb}",
            "expanded_median": q([r["expanded"] for r in rs], 0.5),
            "ratio_to_admissible_median": q([r["expanded"] / adm[r["seed"]] for r in rs], 0.5),
            "optimal_rate": sum(r["suboptimality"] < 1e-12 for r in rs) / len(rs),
            "suboptimality_mean": statistics.mean(r["suboptimality"] for r in rs),
            "suboptimality_max": max(r["suboptimality"] for r in rs),
            "time_ms_median": q([r["time_ms"] for r in rs], 0.5),
        })
    out["p2"] = p2_sum
    sweep = defaultdict(dict)
    for r in p2["sweep"]:
        sweep[r["mountain"], r["seed"]][r["heuristic"]] = r["expanded"]
    out["p2_sweep"] = [
        {
            "mountain": m,
            "zero_median": q([v["zero"] for (mm, _), v in sweep.items() if mm == m], 0.5),
            "admissible_median": q([v["admissible"] for (mm, _), v in sweep.items() if mm == m], 0.5),
            "ratio_median": q([v["admissible"] / v["zero"] for (mm, _), v in sweep.items() if mm == m], 0.5),
        }
        for m in sorted({k[0] for k in sweep})
    ]

    groups = defaultdict(list)
    for r in p3["rows"]:
        groups[r["b"], r["d"], r["order"]].append(r)
    out["p3"] = [
        {
            "b": b,
            "d": d,
            "order": o,
            "leaves_median": q([r["leaves"] for r in rs], 0.5),
            "nodes_median": q([r["nodes"] for r in rs], 0.5),
            "km_bound": rs[0]["km_bound"],
            "full": rs[0]["full"],
            "effective_branching": q([r["leaves"] for r in rs], 0.5) ** (1 / d),
        }
        for (b, d, o), rs in sorted(groups.items())
    ]
    groups = defaultdict(list)
    for r in p3["noise"]:
        groups[str(r["sigma"])].append(r["leaves"])
    out["p3_noise"] = [{"sigma": s, "leaves_median": q(v, 0.5), "leaves_q25": q(v, 0.25), "leaves_q75": q(v, 0.75)} for s, v in groups.items()]

    groups = defaultdict(list)
    for r in p4["rows"]:
        groups[r["K"], f"{r['algorithm']}/{r['order']}"].append(r)
    out["p4_random"] = [
        {
            "K": k,
            "method": m,
            "nodes_median": q([r["nodes"] for r in rs], 0.5),
            "leaves_median": q([r["leaves"] for r in rs], 0.5),
            "time_ms_median": q([r["time_ms"] for r in rs], 0.5),
            "agree_rate": sum(r["agrees"] for r in rs) / len(rs),
        }
        for (k, m), rs in sorted(groups.items())
    ]
    groups = defaultdict(list)
    for r in p4["cache"]:
        groups[r["K"]].append(r)
    out["p4_cache"] = [
        {
            "K": k,
            "astar_calls_nocache_median": q([r["astar_calls_nocache"] for r in rs], 0.5),
            "astar_calls_cache": rs[0]["astar_calls_cache"],
            "time_ms_nocache_median": q([r["time_ms_nocache"] for r in rs], 0.5),
            "time_ms_cache_median": q([r["time_ms_cache"] for r in rs], 0.5),
            "speedup_median": q([r["time_ms_nocache"] / r["time_ms_cache"] for r in rs], 0.5),
            "agree_rate": sum(r["agrees"] for r in rs) / len(rs),
        }
        for k, rs in sorted(groups.items())
    ]
    return out


def git_version() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "."))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="缩小规模，只检查流程")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL
    timings = {}
    t_all = time.perf_counter()
    results = {}
    for name, fn in (("p1", collect_p1), ("p2", collect_p2), ("p3", collect_p3), ("p4", collect_p4)):
        t0 = time.perf_counter()
        results[name] = fn(cfg)
        timings[name] = round(time.perf_counter() - t0, 1)
        print(f"  用时 {timings[name]} s")
    write_json("summary.json", summarize(results["p1"], results["p2"], results["p3"], results["p4"]))
    write_json("meta.json", {
        "version": __version__,
        "git": git_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.processor() or platform.machine(),
        "seed": SEED,
        "quick": args.quick,
        "config": {k: (list(v) if isinstance(v, range) else v) for k, v in cfg.items() if k != "p3_depths"}
        | {"p3_depths": {b: list(d) for b, d in cfg["p3_depths"].items()}},
        "section_seconds": timings,
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if not args.no_plot:
        from bench import plot

        plot.make_all()


if __name__ == "__main__":
    main()
