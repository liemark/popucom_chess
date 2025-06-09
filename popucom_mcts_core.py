#popucom_mcts_core.py

import numpy as np

# 从泡姆棋游戏逻辑文件中导入必要的常量和函数
try:
    from popucom_chess import BLACK_PLAYER, WHITE_PLAYER, GAME_RESULT_NONE, check_game_over
except ImportError:
    print("错误: 无法导入 popucom_chess.py。请确保它在同一目录下。")
    # 提供默认常量以允许文件结构完整，但功能可能受限
    BLACK_PLAYER = 'black'
    WHITE_PLAYER = 'white'
    GAME_RESULT_NONE = 'None'
    def check_game_over(board, remaining_moves, current_player):
        # 简化版，仅用于避免导入失败
        return {BLACK_PLAYER: GAME_RESULT_NONE, WHITE_PLAYER: GAME_RESULT_NONE}


class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search tree.
    Each node corresponds to a specific game state.
    This class has been moved here to resolve circular import issues.
    """

    def __init__(self, board, remaining_moves, current_player, parent=None, move=None):
        """
        Initializes an MCTS node.

        Args:
            board (np.ndarray): The game board state (NumPy array).
            remaining_moves (dict): Dictionary of remaining moves for black and white.
            current_player (str): The player whose turn it is in this state.
            parent (MCTSNode, optional): The parent node in the MCTS tree. Defaults to None (for root).
            move (tuple, optional): The (x, y) move that led to this state from the parent. Defaults to None.
        """
        self.board = board
        self.remaining_moves = remaining_moves
        self.current_player = current_player
        self.parent = parent
        self.move = move

        self.N = 0  # Visit count
        self.W = 0.0  # Total value (sum of outcomes from this node's simulations)
        self.Q = 0.0  # Average value (W / N)

        self.P = None  # Policy prior probabilities from the neural network for this node's children
        self.children = {}  # Dictionary mapping (x, y) move to MCTSNode child
        self.is_expanded = False  # True if this node's children have been generated

        # 使用从 popucom_chess 导入的 check_game_over 来判断终局状态
        game_results = check_game_over(self.board, self.remaining_moves,
                                       BLACK_PLAYER if self.current_player == WHITE_PLAYER else WHITE_PLAYER)
        self.is_terminal = (game_results[BLACK_PLAYER] != GAME_RESULT_NONE or
                            game_results[WHITE_PLAYER] != GAME_RESULT_NONE)
        self.game_result = game_results

# 示例用法 (仅供测试此文件是否能独立运行)
if __name__ == "__main__":
    from popucom_chess import initialize_board # 仅为测试导入

    # 模拟一个棋盘和玩家信息
    dummy_board = initialize_board()
    dummy_remaining_moves = {'black': 25, 'white': 25}
    dummy_current_player = BLACK_PLAYER

    # 实例化一个 MCTSNode
    node = MCTSNode(dummy_board, dummy_remaining_moves, dummy_current_player)

    print(f"MCTSNode 创建成功。")
    print(f"当前玩家: {node.current_player}")
    print(f"是否为终局状态: {node.is_terminal}")
