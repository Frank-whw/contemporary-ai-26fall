# 实验二：搜索算法（A\*、Minimax、Alpha-Beta）

## 1. 运行环境

- Python ≥ 3.10（实测 3.12.13，macOS arm64，Apple M5，单线程）。
- 算法代码只用标准库（`heapq`、`dataclasses` 等）；`matplotlib` 只在出图时用到。
- 在仓库根目录执行 `uv sync`（根目录 `pyproject.toml` / `uv.lock` 已锁定 matplotlib 3.11.2）；
- 下文命令都在 `Lab2/` 目录下执行。用 uv 时在命令前加 `uv run`，例如 `uv run python run.py --all`。

## 2. 文件组织

```
Lab2/
├── README.md
├── requirements.txt
├── run.py                  # 统一入口：跑给定用例，打印输出、预期与搜索开销
├── src/
│   ├── astar.py            # 通用 A*：出队时确认终点、closed 用 set、可选 tie-break
│   ├── grid.py             # 网格四邻域、进入代价、启发函数、反向 Dijkstra 求真实剩余代价
│   ├── p1_maze.py          # 题目一：迷宫寻路
│   ├── p2_weighted.py      # 题目二：加权地形
│   ├── p3_game_tree.py     # 题目三：TreeNode、Minimax、Alpha-Beta、访问顺序
│   └── p4_treasure.py      # 题目四：距离矩阵、博弈状态、Minimax / Alpha-Beta / 置换表
├── tests/
│   ├── cases.py            # 给定的全部测试用例与预期输出
│   └── test_all.py         # 18 组断言，含随机输入上与 BFS / Dijkstra / Minimax 的对拍
├── bench/
│   ├── collect.py          # 采集报告用的全部实验数据
│   ├── plot.py             # 读取 bench/ 下的数据出图
│   └── *.csv, *.json       # 实验数据（见第 5 节）
└── report/
    └── figures/            # 由 plot.py 生成的图
```

| 题目 | 函数                                               | 输入                                           | 输出                                      |
| ---- | -------------------------------------------------- | ---------------------------------------------- | ----------------------------------------- |
| 一   | `src.p1_maze.solve_maze(start, goal, grid)`        | 起点、终点 `(row, col)`，`List[List[str]]`     | 最短路径坐标列表，无解 `[]`               |
| 二   | `src.p2_weighted.solve_terrain(start, goal, grid)` | 同上，格子为数字或 `'S'`/`'E'`                 | 最小总代价 `int`，无解 `-1`               |
| 三   | `src.p3_game_tree.best_score(root)`                | `TreeNode` 根节点；JSON 用 `from_json(d)` 转换 | MAX 玩家的最佳得分                        |
| 四   | `src.p4_treasure.solve(depth)`                     | 搜索深度 H（地图与宝藏价值内置）               | `{"best_target": ..., "best_value": ...}` |

## 4. 运行顺序

1. 测试，约 1 秒：

   ```bash
   python tests/test_all.py
   ```

2. 跑给定用例，打印输出、预期、PASS/FAIL 和节点数：

   ```bash
   python run.py --all                 # 全部题目
   python run.py --problem 2           # 只跑题目二，可重复指定
   python run.py --problem 4 --depth 1 2 3 4
   ```

3. 采集实验数据并出图。在 Apple M5 上完整运行约 20 秒，`--quick` 缩小规模只检查流程，`--no-plot` 不出图：

   ```bash
   python bench/collect.py
   ```

4. 只重画图，不重跑实验：

   ```bash
   python bench/plot.py
   ```

## 5. 结果位置

| 文件                                                | 内容                                                           | 报告章节 |
| --------------------------------------------------- | -------------------------------------------------------------- | -------- |
| `bench/p1_given.json`                               | 题目一给定用例：两种启发函数的路径、开销、逐个出队节点的 g/h/f | 2        |
| `bench/p1_termination_demo.json`                    | 一般图上"终点出队才确认"的演示                                 | 1        |
| `bench/p1_random.csv`                               | 随机迷宫：N × 障碍密度 × 20 个种子 × 4 种搜索方式              | 2、6     |
| `bench/p1_example.json`                             | 40×40 迷宫的已展开格子，用于图 2                               | 2        |
| `bench/p2_given.json`                               | 题目二给定用例：代价累计、各格 h\* 与两种启发值、出队轨迹      | 3        |
| `bench/p2_random.csv`、`bench/p2_terrain_sweep.csv` | 随机地形：启发函数、加权 A\*、地形代价分布扫描                 | 3、6     |
| `bench/p3_given.csv`、`bench/p3_trace.json`         | 题目三给定树：各访问顺序的开销与 Alpha-Beta 逐节点轨迹         | 4        |
| `bench/p3_random.csv`、`bench/p3_noise.csv`         | 随机博弈树：访问顺序、排序噪声                                 | 4        |
| `bench/p4_distance_matrix.json`                     | 6×6 距离矩阵与 A\* 开销                                        | 5        |
| `bench/p4_depth.csv`、`bench/p4_pv.json`            | 给定地图 H=1..4 的结果、开销与主变例                           | 5        |
| `bench/p4_random.csv`、`bench/p4_cache.csv`         | 随机地图的走法排序、置换表、距离缓存                           | 6        |
| `bench/summary.json`                                | 报告引用的中位数、比值等汇总                                   | 全文     |
| `bench/meta.json`                                   | 版本、环境、参数、用时                                         | —        |
| `report/figures/fig1–fig5`                          | 报告插图                                                       | 2–6      |

## 6. 复现性

- 随机实例的种子由 `SEED = 2026` 加实例参数拼成字符串（CPython 对字符串种子使用 SHA-512），跨进程、跨机器都确定。
- 节点数、剪枝数、代价等计数结果完全确定；本机连续两次完整运行，除耗时字段外所有 CSV / JSON 逐位一致。耗时随机器和负载变化。
