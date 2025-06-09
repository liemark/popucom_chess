# popucom_puct.py

import torch
import numpy as np
import math
import random
import os
import sys

from numba import jit, float64, int64

# 导入 MCTSNode from the core file
from popucom_mcts_core import MCTSNode

# 导入常量和函数 from game logic and NN interface
from popucom_chess import (
    BOARD_SIZE, BLACK_PLAYER, WHITE_PLAYER, GAME_RESULT_NONE,
    GAME_RESULT_WIN, GAME_RESULT_LOSS, GAME_RESULT_DRAW,
    do_move, check_valid_move_for_next_player, check_game_over
)
from popucom_nn_interface import transform_game_state_to_input_tensor, calculate_score_target
from popucom_nn_model import PomPomNN

# 尝试导入 Cython 编译的模块
# 确保在运行此文件之前，已成功编译 popucom_mcts_cython_utils.pyx
try:
    import popucom_mcts_cython_utils as mcts_cython

    #print("Cython module 'popucom_mcts_cython_utils' loaded successfully.")
except ImportError:
    print("WARNING: Cython module 'popucom_mcts_cython_utils' not found.")
    print("Please compile it using 'python setup.py build_ext --inplace'.")
    print("Falling back to Pure Python/Numba implementations for MCTS core logic.")
    mcts_cython = None


class _PurePythonMCTSSearcher:
    def __init__(self, model, c_puct=1.0, use_win_loss_target=False, debug_scoring=False):
        self.model = model
        self.c_puct = c_puct
        self.use_win_loss_target = use_win_loss_target
        self.debug_scoring = debug_scoring
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    @jit(float64(float64, int64, int64, float64, float64), nopython=True, cache=True)
    def _calculate_puct_score_jitted(Q_sa, N_s, N_sa, P_sa, c_puct):
        if N_s == 0:
            sqrt_N_s = 0.0
        else:
            sqrt_N_s = math.sqrt(float(N_s))
        uct_score = Q_sa + c_puct * P_sa * (sqrt_N_s / (1.0 + N_sa))
        return uct_score

    def _get_legal_moves(self, board, player):
        legal_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                # check_valid_move_for_next_player 内部已经使用了 Numba 优化
                if check_valid_move_for_next_player(board, r, c, player):
                    legal_moves.append((r, c))
        return legal_moves

    def _evaluate_node(self, node):
        nn_input = transform_game_state_to_input_tensor(
            node.board, node.remaining_moves, node.current_player
        )
        nn_input_tensor = torch.from_numpy(nn_input).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # 模型现在只返回 3 个输出 (策略、价值、所有权)
            policy_logits, value, ownership = self.model(nn_input_tensor)
        node.P = policy_logits.squeeze(0).cpu().numpy()
        return value.item()

    def _select_child(self, node):
        # 尝试使用 Cython 优化版本，如果可用
        if mcts_cython:
            legal_moves = self._get_legal_moves(node.board, node.current_player)
            if not legal_moves:
                return None, None  # No legal moves, no child to select

            # 准备 Cython 函数所需的输入数据
            # FIX: Convert policy_priors to float64 to match Cython expectation
            policy_priors = node.P.astype(np.float64)  # 策略先验
            child_N_counts = np.zeros(len(legal_moves), dtype=np.int64)
            child_Q_values = np.zeros(len(legal_moves), dtype=np.float64)

            for i, move in enumerate(legal_moves):
                child = node.children.get(move)
                if child:
                    child_N_counts[i] = child.N
                    child_Q_values[i] = child.Q

            # 调用 Cython 函数
            best_move, _ = mcts_cython.select_best_move_cy(
                policy_priors, child_N_counts, child_Q_values, node.N, self.c_puct, legal_moves
            )

            # 返回最佳移动及其对应的子节点 (如果是现有节点)
            return best_move, node.children.get(best_move)

        # 原始 Python 实现 (如果 Cython 模块不可用)
        best_uct = -float('inf')
        best_move = None
        legal_moves = self._get_legal_moves(node.board, node.current_player)
        move_to_idx = {(r, c): r * BOARD_SIZE + c for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)}
        if node.P is None:
            if not node.is_expanded:
                self._evaluate_node(node)
            else:
                raise ValueError("Node policy priors (node.P) must be set before selection.")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                move = (r, c)
                move_idx = move_to_idx[move]
                if move in legal_moves:
                    child = node.children.get(move)
                    if child:
                        Q_sa = child.Q
                        N_s = node.N
                        N_sa = child.N
                        P_sa = node.P[move_idx]
                        uct_score = self._calculate_puct_score_jitted(Q_sa, N_s, N_sa, P_sa, self.c_puct)
                    else:
                        P_sa = node.P[move_idx]
                        uct_score = self._calculate_puct_score_jitted(0.0, node.N, 0, P_sa, self.c_puct)
                    if uct_score > best_uct:
                        best_uct = uct_score
                        best_move = move
        return best_move, node.children.get(best_move)

    def _expand_node(self, node):
        node.is_expanded = True
        value_from_nn = self._evaluate_node(node)
        legal_moves = self._get_legal_moves(node.board, node.current_player)
        node.legal_moves = legal_moves
        for move in legal_moves:
            next_board, next_remaining_moves, _ = do_move(
                node.board, move[0], move[1], node.remaining_moves, node.current_player
            )
            next_player = WHITE_PLAYER if node.current_player == BLACK_PLAYER else BLACK_PLAYER
            child_node = MCTSNode(next_board, next_remaining_moves, next_player, parent=node, move=move)
            node.children[move] = child_node
        return value_from_nn

    def _backpropagate(self, node, value):
        # 尝试使用 Cython 优化版本，如果可用
        if mcts_cython:
            path_nodes = []
            current_path_node = node
            while current_path_node is not None:
                path_nodes.append(current_path_node)
                current_path_node = current_path_node.parent
            path_nodes.reverse()  # 从根到当前节点 (模拟的叶节点)

            # 准备 Cython 函数所需的输入数据
            path_len = len(path_nodes)
            N_path = np.array([n.N for n in path_nodes], dtype=np.int64)
            W_path = np.array([n.W for n in path_nodes], dtype=np.float64)
            # In popucom_puct.py, inside _backpropagate method:
            is_terminal_path = np.array([n.is_terminal for n in path_nodes], dtype=np.uint8)  # Changed dtype here

            # game_result_value_path 需要包含叶子节点的值
            game_result_value_path = np.zeros(path_len, dtype=np.float64)
            # 最后一个元素是模拟的叶子节点的值
            game_result_value_path[path_len - 1] = value

            # 调用 Cython 函数
            mcts_cython.backpropagate_cy(
                N_path, W_path, is_terminal_path, game_result_value_path, path_len, self.use_win_loss_target
            )

            # 将更新后的 N 和 W 值写回 MCTSNode 对象
            # 并且重新计算 Q 值
            for i in range(path_len):
                path_nodes[i].N = N_path[i]
                path_nodes[i].W = W_path[i]
                path_nodes[i].Q = path_nodes[i].W / path_nodes[i].N if path_nodes[i].N > 0 else 0.0
            return

        # 原始 Python 实现 (如果 Cython 模块不可用)
        current = node
        while current is not None:
            current.N += 1
            if current.is_terminal:
                if self.use_win_loss_target:
                    if current.game_result[current.current_player] == GAME_RESULT_WIN:
                        terminal_value = 1.0
                    elif current.game_result[current.current_player] == GAME_RESULT_LOSS:
                        terminal_value = -1.0
                    else:
                        terminal_value = 0.0
                    current.W += terminal_value
                else:
                    terminal_score = calculate_score_target(current.board, current.current_player,
                                                            debug_mode=self.debug_scoring)
                    current.W += terminal_score
            else:
                current.W += value

            current.Q = current.W / current.N if current.N > 0 else 0.0
            value = -value
            current = current.parent

    def run_search(self, root_node, num_simulations):
        if not root_node.is_expanded:
            self._expand_node(root_node)
        for _ in range(num_simulations):
            node = root_node
            path = [node]
            while node.is_expanded and not node.is_terminal:
                selected_move, node = self._select_child(node)
                if node is None:
                    print(f"Warning: No child selected. Breaking selection loop. Current node N: {path[-1].N}")
                    break
                path.append(node)

            if node is None:
                continue

            if node.is_terminal:
                if self.use_win_loss_target:
                    if node.game_result[node.current_player] == GAME_RESULT_WIN:
                        value = 1.0
                    elif node.game_result[node.current_player] == GAME_RESULT_LOSS:
                        value = -1.0
                    else:
                        value = 0.0
                else:
                    value = calculate_score_target(node.board, node.current_player, debug_mode=self.debug_scoring)
            else:
                value = self._expand_node(node)

            self._backpropagate(node, value)

    def get_policy_distribution(self, root_node, temperature=1.0):
        policy_dist = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        legal_moves = getattr(root_node, 'legal_moves', [])
        if not legal_moves and root_node.is_expanded:
            if legal_moves:
                uniform_prob = 1.0 / len(legal_moves)
                for r, c in legal_moves:
                    policy_dist[r, c] = uniform_prob
            return policy_dist
        elif not root_node.is_expanded:
            legal_moves_from_game_logic = self._get_legal_moves(root_node.board, root_node.current_player)
            if legal_moves_from_game_logic:
                uniform_prob = 1.0 / len(legal_moves_from_game_logic)
                for r, c in legal_moves_from_game_logic:
                    policy_dist[r, c] = uniform_prob
            return policy_dist

        q_values_for_legal_moves_for_current_player = []
        moves_in_order = []
        visit_counts_for_legal_moves = []

        for r, c in legal_moves:
            move = (r, c)
            if move in root_node.children:
                child = root_node.children[move]
                visit_counts_for_legal_moves.append(child.N)
                moves_in_order.append(move)
                q_values_for_legal_moves_for_current_player.append(-child.Q)

        if not moves_in_order:
            if legal_moves:
                uniform_prob = 1.0 / len(legal_moves)
                for r, c in legal_moves:
                    policy_dist[r, c] = uniform_prob
            return policy_dist

        if temperature == 0:
            if not q_values_for_legal_moves_for_current_player:
                if legal_moves:
                    uniform_prob = 1.0 / len(legal_moves)
                    for r, c in legal_moves:
                        policy_dist[r, c] = uniform_prob
                return policy_dist

            max_q = -float('inf')
            for q_val in q_values_for_legal_moves_for_current_player:
                if q_val > max_q:
                    max_q = q_val

            best_moves_indices_in_list = [i for i, q_val in enumerate(q_values_for_legal_moves_for_current_player) if
                                          np.isclose(q_val, max_q)]

            chosen_idx_in_list = random.choice(best_moves_indices_in_list)
            chosen_move = moves_in_order[chosen_idx_in_list]

            policy_dist[chosen_move[0], chosen_move[1]] = 1.0
            return policy_dist
        else:
            visit_counts_np = np.array(visit_counts_for_legal_moves, dtype=np.float32)
            exponentiated_visits = np.power(visit_counts_np, 1.0 / temperature)
            sum_exp_visits = np.sum(exponentiated_visits)

            if sum_exp_visits > 0:
                for i, move in enumerate(moves_in_order):
                    r, c = move
                    policy_dist[r, c] = exponentiated_visits[i] / sum_exp_visits
            else:
                if legal_moves:
                    uniform_prob = 1.0 / len(legal_moves)
                    for r, c in legal_moves:
                        policy_dist[r, c] = uniform_prob
            return policy_dist


_MCTSSearcher_Implementation = _PurePythonMCTSSearcher

MCTSSearcher = _MCTSSearcher_Implementation

if __name__ == "__main__":
    class DummyNN(PomPomNN):
        def __init__(self):
            super().__init__(num_res_blocks=1, num_filters=16)

        def forward(self, x):
            batch_size = x.size(0)
            policy_output = torch.full((batch_size, BOARD_SIZE * BOARD_SIZE), 1.0 / (BOARD_SIZE * BOARD_SIZE))
            value_output = torch.rand(batch_size, 1) * 2 - 1
            ownership_output = torch.rand(batch_size, BOARD_SIZE, BOARD_SIZE) * 2 - 1
            return policy_output, value_output, ownership_output


    dummy_model = DummyNN()
    print("Using a dummy neural network model for demonstration.")

    from popucom_chess import initialize_board as chess_initialize_board, BLACK_PLAYER, WHITE_PLAYER, GAME_RESULT_WIN, \
        GAME_RESULT_LOSS, GAME_RESULT_DRAW

    initial_board = chess_initialize_board()

    initial_board[0, 0, 0] = 1
    initial_board[0, 0, 4] = 0
    initial_board[1, 0, 1] = 1
    initial_board[1, 0, 4] = 0
    initial_board[2, 0, 0] = 1
    initial_board[2, 0, 4] = 0

    test_remaining_moves = {'black': 25, 'white': 25}
    test_current_player = BLACK_PLAYER

    print(f"\nStarting PUCT search for {test_current_player}...")
    print(f"Initial remaining moves: {test_remaining_moves}")

    searcher = MCTSSearcher(model=dummy_model, c_puct=1.0)
    root_node = MCTSNode(initial_board, test_remaining_moves, test_current_player)
    searcher.run_search(root_node, num_simulations=10)

    policy_output_distribution = searcher.get_policy_distribution(root_node, temperature=1.0)

    print("\nPUCT search completed (first run might include compilation time).")
    print("Output Policy Distribution (9x9 probabilities):\n", policy_output_distribution)
    print(f"Sum of policy distribution: {np.sum(policy_output_distribution):.4f}")

    best_move_idx = np.argmax(policy_output_distribution)
    best_move_r = best_move_idx // BOARD_SIZE
    best_move_c = best_move_idx % BOARD_SIZE
    print(f"\nBest move suggested by PUCT search: ({best_move_r}, {best_move_c})")
    print(f"Probability for this move: {policy_output_distribution[best_move_r, best_move_c]:.4f}")

    print("\n--- Testing with temperature=0 (deterministic best move) ---")
    policy_output_deterministic = searcher.get_policy_distribution(root_node, temperature=0)
    print("Output Policy Distribution (temperature=0):\n", policy_output_deterministic)
    best_move_idx_det = np.argmax(policy_output_deterministic)
    best_move_r_det = best_move_idx_det // BOARD_SIZE
    best_move_c_det = best_move_idx_det % BOARD_SIZE
    print(f"Best move suggested by PUCT search (temperature=0): ({best_move_r_det}, {best_move_c_det})")
