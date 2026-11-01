"""读取 bench/ 下的 CSV / JSON，生成报告用图到 report/figures/。不重新跑实验：

    python bench/plot.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
FIG = ROOT / "report" / "figures"

plt.rcParams.update({
    # Arial Unicode MS 同时有中文和 ²、−、⌈⌉ 等符号；其余为没有它时的退路
    "font.sans-serif": ["Arial Unicode MS", "Heiti TC", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def read_csv(name):
    with open(BENCH / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(name):
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[(n - 1) // 2] + xs[n // 2]) / 2


def quart(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def save(fig, name, rect=(0, 0, 1, 1)):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print(f"  图 {name}")


def fig_p1_nodes():
    rows = read_csv("p1_random.csv")
    by = defaultdict(list)
    for r in rows:
        by[int(r["N"]), float(r["density"]), r["heuristic"], r["tie_break"]].append(int(r["expanded"]))
    sizes = sorted({k[0] for k in by})
    densities = sorted({k[1] for k in by})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    styles = [("zero", "h", "h=0（Dijkstra）", "tab:gray"), ("manhattan", "fifo", "曼哈顿，同 f 先入先出", "tab:orange"),
              ("manhattan", "h", "曼哈顿，同 f 先出 h 小", "tab:blue")]
    density = 0.2 if 0.2 in densities else densities[-1]
    for h, tb, label, color in styles:
        vals = [by[n, density, h, tb] for n in sizes]
        ax1.plot(sizes, [med(v) for v in vals], "o-", color=color, label=label)
        ax1.fill_between(sizes, [quart(v, 0.25) for v in vals], [quart(v, 0.75) for v in vals], color=color, alpha=0.15)
    ax1.plot(sizes, [n * n for n in sizes], ":", color="black", lw=1, label="N²（全部格子）")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("迷宫边长 N")
    ax1.set_ylabel("展开节点数（中位数，阴影为四分位）")
    ax1.set_title(f"(a) 障碍密度 {density}")
    ax1.legend(fontsize=7)
    for n in sizes:
        zero = {}
        for r in rows:
            if int(r["N"]) == n and r["heuristic"] == "zero":
                zero[float(r["density"]), r["seed"]] = int(r["expanded"])
        ratios = []
        for d in densities:
            rs = [int(r["expanded"]) / zero[d, r["seed"]] for r in rows
                  if int(r["N"]) == n and float(r["density"]) == d and r["heuristic"] == "manhattan" and r["tie_break"] == "h"]
            ratios.append(med(rs))
        ax2.plot(densities, ratios, "o-", label=f"N={n}")
    ax2.set_xlabel("障碍密度")
    ax2.set_ylabel("曼哈顿 / h=0 的展开数之比（中位数）")
    ax2.set_title("(b) 启发函数节省的比例")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7)
    save(fig, "fig1_p1_nodes.png")


def fig_p1_frontier():
    ex = read_json("p1_example.json")
    n = ex["N"]
    keys = [("zero/h", "h=0（Dijkstra）"), ("manhattan/fifo", "曼哈顿，同 f 先入先出"), ("manhattan/h", "曼哈顿，同 f 先出 h 小")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (key, title) in zip(axes, keys):
        img = [[0 if ch == "X" else 1 for ch in row] for row in ex["grid"]]
        for r, c in ex[key]["expanded"]:
            img[r][c] = 2
        for r, c in ex[key]["path"]:
            img[r][c] = 3
        ax.imshow(img, cmap=ListedColormap(["#222222", "#f4f4f4", "#9ecae1", "#d62728"]), vmin=0, vmax=3)
        ax.set_title(f"{title}\n展开 {ex[key]['count']} 个节点", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"{n}×{n} 随机迷宫（障碍密度 {ex['density']}）：黑=障碍，蓝=已展开，红=最短路径", fontsize=9)
    save(fig, "fig2_p1_frontier.png", rect=(0, 0, 1, 0.93))


def fig_p2():
    rows = read_csv("p2_random.csv")
    sweep = read_csv("p2_terrain_sweep.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.6))
    by = defaultdict(list)
    for r in rows:
        by[int(r["N"]), r["heuristic"], float(r["weight"]), r["tie_break"]].append(r)
    sizes = sorted({k[0] for k in by})
    for (h, w, tb), label, color, ls in [
        (("zero", 1.0, "h"), "h=0", "tab:gray", "-"),
        (("manhattan", 1.0, "h"), "曼哈顿（E 邻格高估 1）", "tab:orange", "--"),
        (("admissible", 1.0, "h"), "修正：c_min·(M−1)+c_E", "tab:blue", ":"),
    ]:
        ax1.plot(sizes, [med([int(r["expanded"]) for r in by[n, h, w, tb]]) for n in sizes], "o", ls=ls, color=color, label=label)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(sizes, [str(n) for n in sizes])
    ax1.minorticks_off()
    ax1.set_xlabel("地图边长 N")
    ax1.set_ylabel("展开节点数（中位数）")
    ax1.set_title("(a) 三种启发函数（后两条重合）")
    ax1.legend(fontsize=7)

    pairs = defaultdict(dict)
    for r in sweep:
        pairs[float(r["mountain"]), r["seed"]][r["heuristic"]] = int(r["expanded"])
    ms = sorted({k[0] for k in pairs})
    ratio = [med([v["admissible"] / v["zero"] for (m, _), v in pairs.items() if m == mm]) for mm in ms]
    ax2.plot(ms, ratio, "o-", color="tab:blue")
    ax2.set_xlabel("高山（代价 8）占比")
    ax2.set_ylabel("A* / Dijkstra 的展开数之比")
    ax2.set_title("(b) 地形越贵，启发函数越弱（N=40）")
    ax2.set_ylim(0, 1.05)

    n = max(sizes)
    weights = [1.0] + sorted({float(r["weight"]) for r in rows if r["heuristic"] == "weighted"})
    base = {r["seed"]: int(r["expanded"]) for r in by[n, "admissible", 1.0, "h"]}
    exp_ratio, subopt = [], []
    for w in weights:
        rs = by[n, "admissible", 1.0, "h"] if w == 1.0 else by[n, "weighted", w, "h"]
        exp_ratio.append(med([int(r["expanded"]) / base[r["seed"]] for r in rs]))
        subopt.append(100 * sum(float(r["suboptimality"]) for r in rs) / len(rs))
    ax3.plot(weights, exp_ratio, "o-", color="tab:blue", label="展开数（相对 w=1）")
    ax3.set_xlabel("权重 w（h 放大倍数）")
    ax3.set_ylabel("展开数 / w=1 时的展开数", color="tab:blue")
    ax3.set_ylim(0, 1.05)
    ax3b = ax3.twinx()
    ax3b.plot(weights, subopt, "s--", color="tab:red", label="平均超出最优代价")
    ax3b.set_ylabel("平均超出最优代价（%）", color="tab:red")
    ax3b.grid(False)
    ax3.set_title(f"(c) 加权 A* 的代价—速度权衡（N={n}）")
    save(fig, "fig3_p2.png")


def fig_p3():
    rows = read_csv("p3_random.csv")
    noise = read_csv("p3_noise.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    by = defaultdict(list)
    for r in rows:
        if int(r["b"]) == 3:
            by[r["order"], int(r["d"])].append(int(r["leaves"]))
    depths = sorted({d for _, d in by})
    for order, label, color, marker in [
        ("minimax", "Minimax（无剪枝）", "black", "x"),
        ("worst", "Alpha-Beta 最差序", "tab:red", "v"),
        ("descending", "全层降序", "tab:orange", "s"),
        ("original", "原始（随机）顺序", "tab:gray", "o"),
        ("ascending", "全层升序", "tab:purple", "D"),
        ("best", "最优序（MAX 降、MIN 升）", "tab:blue", "^"),
    ]:
        ax1.plot(depths, [med(by[order, d]) for d in depths], marker=marker, color=color, label=label, ms=4)
    km = {int(r["d"]): int(r["km_bound"]) for r in rows if int(r["b"]) == 3}
    ax1.plot(depths, [km[d] for d in depths], "--", color="black", lw=0.8,
             label=r"下界 $b^{\lceil d/2\rceil}+b^{\lfloor d/2\rfloor}-1$（与最优序重合）")
    ax1.set_yscale("log")
    ax1.set_xlabel("树深 d（分支因子 b=3）")
    ax1.set_ylabel("求值叶子数（中位数）")
    ax1.set_title("(a) 访问顺序对剪枝的影响")
    ax1.legend(fontsize=7)

    sig = defaultdict(list)
    base = []
    for r in noise:
        if r["sigma"] == "original":
            base.append(int(r["leaves"]))
        else:
            sig[float(r["sigma"])].append(int(r["leaves"]))
    xs = sorted(sig)
    ax2.plot(xs, [med(sig[s]) for s in xs], "o-", color="tab:blue", label="按 值+噪声 排序")
    ax2.fill_between(xs, [quart(sig[s], 0.25) for s in xs], [quart(sig[s], 0.75) for s in xs], color="tab:blue", alpha=0.15)
    ax2.axhline(med(base), color="tab:gray", ls="--", label="原始（随机）顺序")
    b, d = int(noise[0]["b"]), int(noise[0]["d"])
    ax2.axhline(b ** ((d + 1) // 2) + b ** (d // 2) - 1, color="tab:blue", ls=":", lw=1, label="最优下界")
    ax2.set_xscale("symlog", linthresh=0.01)
    ax2.set_xlim(0, max(xs) * 1.2)
    ax2.set_xlabel("排序噪声 σ（叶值 ~ U[0,1)，σ=0 即最优序）")
    ax2.set_ylabel("求值叶子数（中位数）")
    ax2.set_title(f"(b) 排序估计越不准，剪枝越少（b={b}, d={d}）")
    ax2.legend(fontsize=7)
    save(fig, "fig4_p3.png")


def fig_p4():
    depth = read_csv("p4_depth.csv")
    rows = read_csv("p4_random.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    methods = [
        ("minimax", "natural", "Minimax", "black", "x"),
        ("alphabeta", "worst", "αβ 最差序", "tab:red", "v"),
        ("alphabeta", "natural", "αβ 自然序 T1→T4", "tab:gray", "o"),
        ("alphabeta", "greedy", "αβ 按本步收益排序", "tab:green", "s"),
        ("alphabeta", "best", "αβ 最优序", "tab:blue", "^"),
        ("minimax_memo", "natural", "Minimax+置换表", "tab:purple", "D"),
    ]
    hs = sorted({int(r["H"]) for r in depth})
    for alg, order, label, color, marker in methods:
        vals = {int(r["H"]): int(r["nodes"]) for r in depth if r["algorithm"] == alg and r["order"] == order}
        ax1.plot(hs, [vals[h] for h in hs], marker=marker, color=color, label=label, ms=4)
    ax1.set_xticks(hs)
    ax1.set_xlabel("搜索深度 H")
    ax1.set_ylabel("展开节点数")
    ax1.set_title("(a) 题目给定地图")
    ax1.legend(fontsize=7)

    by = defaultdict(list)
    for r in rows:
        by[r["algorithm"], r["order"], int(r["K"])].append(int(r["nodes"]))
    ks = sorted({int(r["K"]) for r in rows})
    for alg, order, label, color, marker in methods:
        ax2.plot(ks, [med(by[alg, order, k]) for k in ks], marker=marker, color=color, label=label, ms=4)
    ax2.set_yscale("log")
    ax2.set_xticks(ks)
    ax2.set_xlabel("宝藏数 K（搜到终局，H=K）")
    ax2.set_ylabel("展开节点数（中位数）")
    maps = len({r["seed"] for r in rows if int(r["K"]) == ks[0]})
    ax2.set_title(f"(b) 9×9 随机地图，每个 K 取 {maps} 张")
    ax2.legend(fontsize=7)
    save(fig, "fig5_p4.png")


def make_all():
    print("[出图]")
    fig_p1_nodes()
    fig_p1_frontier()
    fig_p2()
    fig_p3()
    fig_p4()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    make_all()
