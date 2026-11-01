"""题目三：极大极小博弈树。根节点为 MAX，子节点默认从左到右访问。"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass


class TreeNode:
    """与题目给定的数据结构一致；is_maximizing_player 来自 JSON 输入，缺省时按层交替（根为 MAX）。"""

    def __init__(self, value=None, children=None, is_maximizing_player=None):
        self.value = value  # 叶子节点的得分
        self.children = children if children is not None else []  # 子节点列表
        self.is_maximizing_player = is_maximizing_player

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class GameStats:
    nodes: int = 0  # 访问的节点数（含叶子）
    leaves: int = 0  # 求值的叶子数
    cutoffs: int = 0  # 发生剪枝的次数
    pruned: int = 0  # 因剪枝没有访问的子树数
    cache_hits: int = 0  # 记忆化搜索命中缓存的次数（题目四拓展用）


Order = Callable[[TreeNode, bool], list[TreeNode]]
ORDERS = ("original", "reversed", "ascending", "descending", "best", "worst")


def from_json(data: dict) -> TreeNode:
    """接受 {"tree_structure": {...}} 或其内层 dict。"""
    if "tree_structure" in data:
        data = data["tree_structure"]
    return TreeNode(
        value=data.get("value"),
        children=[from_json(c) for c in data.get("children", [])],
        is_maximizing_player=data.get("is_maximizing_player"),
    )


def _is_max(node: TreeNode, default: bool) -> bool:
    return default if node.is_maximizing_player is None else node.is_maximizing_player


def minimax(node: TreeNode, maximizing: bool = True, stats: GameStats | None = None):
    stats = stats if stats is not None else GameStats()
    stats.nodes += 1
    if node.is_leaf:
        stats.leaves += 1
        return node.value
    is_max = _is_max(node, maximizing)
    values = [minimax(child, not is_max, stats) for child in node.children]
    return max(values) if is_max else min(values)


def alphabeta(
    node: TreeNode,
    alpha: float = -math.inf,
    beta: float = math.inf,
    maximizing: bool = True,
    stats: GameStats | None = None,
    order: Order | None = None,
    trace: list | None = None,
    _path: str = "R",
):
    """alpha 是 MAX 已能保证的下界，beta 是 MIN 已能保证的上界；alpha ≥ beta 时，
    当前节点剩下的子节点不可能影响上层选择，直接剪掉。
    trace 非空时按后序记录每个节点进入时的 (alpha, beta)、返回值和剪掉的子节点数；
    节点名 R.i.j 表示根的第 i 个孩子的第 j 个孩子（按原始左右位置编号）。"""
    stats = stats if stats is not None else GameStats()
    stats.nodes += 1
    if node.is_leaf:
        stats.leaves += 1
        if trace is not None:
            trace.append({"node": _path, "type": "leaf", "alpha": alpha, "beta": beta, "value": node.value, "pruned": 0})
        return node.value
    is_max = _is_max(node, maximizing)
    children = order(node, is_max) if order else node.children
    alpha_in, beta_in = alpha, beta
    value = -math.inf if is_max else math.inf
    rest = 0
    for i, child in enumerate(children):
        child_path = f"{_path}.{_position(node, child)}"
        v = alphabeta(child, alpha, beta, not is_max, stats, order, trace, child_path)
        if is_max:
            value = max(value, v)
            alpha = max(alpha, value)
        else:
            value = min(value, v)
            beta = min(beta, value)
        if alpha >= beta:
            rest = len(children) - i - 1
            if rest:
                stats.cutoffs += 1
                stats.pruned += rest
            break
    if trace is not None:
        trace.append({
            "node": _path,
            "type": "MAX" if is_max else "MIN",
            "alpha": alpha_in,
            "beta": beta_in,
            "value": value,
            "pruned": rest,
        })
    return value


def _position(parent: TreeNode, child: TreeNode) -> int:
    return next(i for i, c in enumerate(parent.children) if c is child)


def best_score(root: TreeNode):
    """题目要求的函数：Alpha-Beta 剪枝，返回 MAX 玩家能保证的最佳得分。"""
    return alphabeta(root)


def subtree_values(root: TreeNode, maximizing: bool = True) -> dict[int, float]:
    """每个节点的精确 minimax 值（按 id 记忆，题目示例里共享的子树只算一次）。"""
    memo: dict[int, float] = {}

    def visit(node: TreeNode, default_max: bool) -> float:
        key = id(node)
        if key not in memo:
            if node.is_leaf:
                memo[key] = node.value
            else:
                is_max = _is_max(node, default_max)
                vals = [visit(c, not is_max) for c in node.children]
                memo[key] = max(vals) if is_max else min(vals)
        return memo[key]

    visit(root, maximizing)
    return memo


def make_order(kind: str, root: TreeNode | None = None, sigma: float = 0.0, rng: random.Random | None = None) -> Order | None:
    """子节点访问顺序。ascending/descending/best/worst/noisy 依据子树的精确 minimax 值排序，
    属于"事先知道答案"的理想排序，用来测剪枝效果的上下界：
    - ascending / descending：所有层都按值升序 / 降序；
    - best：MAX 层降序、MIN 层升序（每层先看最优子节点）；worst 与之相反；
    - noisy：按 值 + N(0, sigma²) 排成 best 顺序，sigma 越大越接近随机顺序。"""
    if kind == "original":
        return None
    if kind == "reversed":
        return lambda node, is_max: node.children[::-1]
    values = subtree_values(root)

    def v(child: TreeNode) -> float:
        return values[id(child)]

    if kind == "ascending":
        return lambda node, is_max: sorted(node.children, key=v)
    if kind == "descending":
        return lambda node, is_max: sorted(node.children, key=v, reverse=True)
    if kind == "best":
        return lambda node, is_max: sorted(node.children, key=v, reverse=is_max)
    if kind == "worst":
        return lambda node, is_max: sorted(node.children, key=v, reverse=not is_max)
    if kind == "noisy":
        rng = rng or random.Random(0)
        return lambda node, is_max: sorted(node.children, key=lambda c: v(c) + rng.gauss(0, sigma), reverse=is_max)
    raise ValueError(f"unknown order: {kind}")


def random_tree(branching: int, depth: int, rng: random.Random, leaf_value=None, maximizing: bool = True) -> TreeNode:
    """均匀分支的随机博弈树；默认叶值取 U[0,1)，几乎没有相等值。"""
    if depth == 0:
        return TreeNode(value=leaf_value(rng) if leaf_value else rng.random())
    return TreeNode(
        children=[random_tree(branching, depth - 1, rng, leaf_value, not maximizing) for _ in range(branching)],
        is_maximizing_player=maximizing,
    )


def knuth_moore_leaves(branching: int, depth: int) -> int:
    """最优排序下 Alpha-Beta 求值的叶子数：b^⌈d/2⌉ + b^⌊d/2⌋ − 1。"""
    return branching ** math.ceil(depth / 2) + branching ** (depth // 2) - 1
