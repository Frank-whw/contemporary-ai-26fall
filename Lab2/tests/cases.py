"""实验二要求 PDF 中给出的全部测试用例与预期输出。"""

from src.p3_game_tree import TreeNode

P1_START = (0, 0)
P1_GOAL = (3, 4)
P1_GRID = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
P1_EXPECTED = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]

P2_START = (0, 0)
P2_GOAL = (4, 4)
P2_GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
P2_EXPECTED = 7
P2_EXPECTED_PATH = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (4, 3), (4, 4)]

P3_JSON = {
    "tree_structure": {
        "is_maximizing_player": True,
        "children": [
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                ],
            },
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                    {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]},
                ],
            },
        ],
    }
}
P3_EXPECTED = 5


def p3_object_tree() -> TreeNode:
    """PDF 第 12 页的构建代码，原样照抄（min2 复用了 max2 对象）。"""
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


P4_EXPECTED = {
    2: {"best_target": "T2", "best_value": -1},
    4: {"best_target": "T3", "best_value": 3},
}
