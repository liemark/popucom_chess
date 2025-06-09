# distutils: language_level=3
# popucom_mcts_cython_utils.pyx

import numpy as np
cimport numpy as np
cimport cython
import math # For math.inf

# 从 popucom_chess 导入常量，以便在 Cython 中使用
# 通常需要将这些常量作为参数传入 Cython 函数，或者在 Cython 中重新定义
BOARD_SIZE = 9

@cython.cdivision(True) # 允许整数除法，不检查除零
@cython.boundscheck(False) # 关闭边界检查以获得最大速度
@cython.wraparound(False) # 关闭负索引检查
# Changed N_s and N_sa to long long for consistency with np.int64
cpdef double calculate_puct_score_cy(double Q_sa, long long N_s, long long N_sa, double P_sa, double c_puct):
    cdef double sqrt_N_s
    if N_s == 0:
        sqrt_N_s = 0.0
    else:
        sqrt_N_s = math.sqrt(<double>N_s)
    return Q_sa + c_puct * P_sa * (sqrt_N_s / (1.0 + N_sa))

cpdef tuple select_best_move_cy(
    double[:] policy_priors, # node.P
    long long[:] child_N_counts,  # 对应 legal_moves 的子节点访问次数 (Changed to long long)
    double[:] child_Q_values, # 对应 legal_moves 的子节点 Q 值
    long long N_s,                # 父节点访问次数 node.N (Changed to long long)
    double c_puct,
    list legal_moves_list    # (r, c) 元组的列表
):
    cdef double best_uct = -math.inf
    cdef tuple best_move = None
    cdef int r, c, move_idx, i
    cdef double Q_sa, P_sa
    cdef long long N_sa_long_long # Use long long for N_sa here
    cdef double uct_score

    cdef int num_legal_moves = len(legal_moves_list)

    if num_legal_moves == 0:
        return (None, None)

    for i in range(num_legal_moves):
        r, c = legal_moves_list[i]
        move_idx = r * BOARD_SIZE + c

        # 检查是否已有子节点
        if child_N_counts[i] > 0: # 如果 child_N_counts[i] > 0，则认为该子节点已存在
            Q_sa = child_Q_values[i]
            N_sa_long_long = child_N_counts[i] # Assign directly to long long
            P_sa = policy_priors[move_idx]
            uct_score = calculate_puct_score_cy(Q_sa, N_s, N_sa_long_long, P_sa, c_puct)
        else:
            P_sa = policy_priors[move_idx]
            uct_score = calculate_puct_score_cy(0.0, N_s, 0, P_sa, c_puct)

        if uct_score > best_uct:
            best_uct = uct_score
            best_move = (r, c)
    return best_move, best_uct

cpdef void backpropagate_cy(
    long long[:] N_path,   # 路径上所有节点的 N 值 (引用) (Changed to long long)
    double[:] W_path, # 路径上所有节点的 W 值 (引用)
    unsigned char[:] is_terminal_path, # FIX: Changed from bint[:] to unsigned char[:]
    double[:] game_result_value_path, # 终端节点的实际游戏结果值 (对于终端节点)
    long long path_len, # Changed to long long
    bint use_win_loss_target # 是否使用输赢目标
):
    cdef long long i # Changed to long long
    cdef double current_value

    if path_len > 0:
        current_value = game_result_value_path[path_len - 1]
    else:
        return # Empty path, nothing to do

    for i in range(path_len - 1, -1, -1):
        N_path[i] += 1
        W_path[i] += current_value
        current_value = -current_value
