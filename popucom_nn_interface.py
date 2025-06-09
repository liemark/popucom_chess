import numpy as np

# 从泡姆棋游戏逻辑文件中导入必要的常量
# 假设 popucom_chess.py 文件在同一目录下
try:
    from popucom_chess import (
        BOARD_SIZE,
        BLACK_PIECE_INDEX,
        WHITE_PIECE_INDEX,
        BLACK_PAINTED_INDEX,
        WHITE_PAINTED_INDEX,
        UNPAINTED_INDEX,
        BLACK_PLAYER,
        WHITE_PLAYER,
        _count_painted_cells  # 导入用于计算涂色地板数量的内部函数
    )
except ImportError:
    print("错误: 无法导入 popucom_chess.py。请确保它在同一目录下。")
    # 提供默认常量以允许文件结构完整，但功能可能受限
    BOARD_SIZE = 9
    BLACK_PIECE_INDEX = 0
    WHITE_PIECE_INDEX = 1
    BLACK_PAINTED_INDEX = 2
    WHITE_PAINTED_INDEX = 3
    UNPAINTED_INDEX = 4
    BLACK_PLAYER = 'black'
    WHITE_PLAYER = 'white'


    # 模拟一个 _count_painted_cells 函数，以防导入失败
    def _count_painted_cells(board, player_color):
        count = 0
        painted_index = BLACK_PAINTED_INDEX if player_color == BLACK_PLAYER else WHITE_PAINTED_INDEX
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c][painted_index] == 1:
                    count += 1
        return count

# 定义神经网络输入通道的数量
NUM_INPUT_CHANNELS = 11

# 新增常量：每位玩家的最大步数，用于归一化
MAX_MOVES_PER_PLAYER = 25


def transform_game_state_to_input_tensor(board, remaining_moves, current_player):
    """
    将泡姆棋的游戏状态转换为神经网络的输入张量。

    输入张量的形状为 (BOARD_SIZE, BOARD_SIZE, NUM_INPUT_CHANNELS)。
    每个通道包含以下信息（9x9 矩阵）：
    - Channel 0: 是否为黑子 (1: 是, 0: 不存在)
    - Channel 1: 是否为白子 (1: 是, 0: 不存在)
    - Channel 2: 是否为黑地板 (1: 是, 0: 不存在)
    - Channel 3: 是否为白地板 (1: 是, 0: 不存在)
    - Channel 4: 是否为空地 (1: 是, 0: 不存在)
    - Channel 5: 是否轮到黑行动 (所有位置为 1 或 0)
    - Channel 6: 是否轮到白行动 (所有位置为 1 或 0)
    - Channel 7: 黑剩余行动力 (所有位置为黑剩余步数的值)
    - Channel 8: 白剩余行动力 (所有位置为白剩余步数的值)
    - Channel 9: 黑已占据地板数量 (所有位置为黑涂色地板数量的值)
    - Channel 10: 白已占据地板数量 (所有位置为白涂色地板数量的值)

    Args:
        board (np.ndarray): 当前棋盘状态，NumPy 数组。
        remaining_moves (dict): 字典，包含 'black' 和 'white' 玩家的剩余步数。
        current_player (str): 当前轮到行动的玩家 ('black' 或 'white')。

    Returns:
        np.ndarray: 神经网络的输入张量，形状为 (BOARD_SIZE, BOARD_SIZE, NUM_INPUT_CHANNELS)。
    """
    # 初始化一个全零的输入张量
    input_tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_INPUT_CHANNELS), dtype=np.float32)

    # 填充棋盘状态相关的通道 (0-4)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell_state = board[r, c, :]  # NumPy 数组切片
            input_tensor[r, c, BLACK_PIECE_INDEX] = cell_state[BLACK_PIECE_INDEX]
            input_tensor[r, c, WHITE_PIECE_INDEX] = cell_state[WHITE_PIECE_INDEX]
            input_tensor[r, c, BLACK_PAINTED_INDEX] = cell_state[BLACK_PAINTED_INDEX]
            input_tensor[r, c, WHITE_PAINTED_INDEX] = cell_state[WHITE_PAINTED_INDEX]
            input_tensor[r, c, UNPAINTED_INDEX] = cell_state[UNPAINTED_INDEX]

    # 填充轮到行动的通道 (5-6)
    if current_player == BLACK_PLAYER:
        input_tensor[:, :, 5] = 1.0  # 黑行动
        input_tensor[:, :, 6] = 0.0  # 白不行动
    else:  # current_player == WHITE_PLAYER
        input_tensor[:, :, 5] = 0.0  # 黑不行动
        input_tensor[:, :, 6] = 1.0  # 白行动

    # 填充剩余行动力通道 (7-8)
    # 修复：归一化因子改为 MAX_MOVES_PER_PLAYER (25)
    input_tensor[:, :, 7] = float(remaining_moves[BLACK_PLAYER]) / MAX_MOVES_PER_PLAYER
    input_tensor[:, :, 8] = float(remaining_moves[WHITE_PLAYER]) / MAX_MOVES_PER_PLAYER

    # 填充已占据地板数量通道 (9-10)
    black_painted_count = _count_painted_cells(board, BLACK_PLAYER)
    white_painted_count = _count_painted_cells(board, WHITE_PLAYER)
    input_tensor[:, :, 9] = float(black_painted_count) / (BOARD_SIZE * BOARD_SIZE)  # 归一化，最大为81
    input_tensor[:, :, 10] = float(white_painted_count) / (BOARD_SIZE * BOARD_SIZE)  # 归一化，最大为81

    return input_tensor


def calculate_ownership_target(final_board, current_player_at_step):
    """
    根据最终棋盘状态计算所有权目标。
    所有权目标编码为：当前玩家所有为 1.0，对手所有为 -1.0，未涂色/空为 0.0。

    Args:
        final_board (np.ndarray): 游戏结束时的棋盘状态。
        current_player_at_step (str): 产生此步数据时对应的玩家 ('black' 或 'white')。

    Returns:
        np.ndarray: 所有权目标张量，形状为 (BOARD_SIZE, BOARD_SIZE)，范围 [-1, 1]。
    """
    ownership_target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    black_painted_idx = BLACK_PAINTED_INDEX
    white_painted_idx = WHITE_PAINTED_INDEX

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if final_board[r, c, black_painted_idx] == 1:
                # 黑色地板
                if current_player_at_step == BLACK_PLAYER:
                    ownership_target[r, c] = 1.0  # 当前玩家所有
                else:
                    ownership_target[r, c] = -1.0  # 对手所有
            elif final_board[r, c, white_painted_idx] == 1:
                # 白色地板
                if current_player_at_step == WHITE_PLAYER:
                    ownership_target[r, c] = 1.0  # 当前玩家所有
                else:
                    ownership_target[r, c] = -1.0  # 对手所有
            else:
                # 未涂色或棋子位置 (视为中立，或根据实际游戏规则调整)
                ownership_target[r, c] = 0.0

    return ownership_target


def calculate_score_target(final_board, current_player_at_step, debug_mode=False):
    """
    根据最终棋盘状态计算分数目标。
    分数目标为当前玩家涂色地板数与对手涂色地板数的差值，并归一化到 [-1, 1] 范围。

    Args:
        final_board (np.ndarray): 游戏结束时的棋盘状态。
        current_player_at_step (str): 产生此步数据时对应的玩家 ('black' 或 'white')。
        debug_mode (bool): 是否打印详细调试信息。

    Returns:
        float: 归一化后的分数目标，范围 [-1, 1]。
    """
    black_painted_count = _count_painted_cells(final_board, BLACK_PLAYER)
    white_painted_count = _count_painted_cells(final_board, WHITE_PLAYER)

    if current_player_at_step == BLACK_PLAYER:
        score_difference = black_painted_count - white_painted_count
    else:  # current_player_at_step == WHITE_PLAYER
        score_difference = white_painted_count - black_painted_count

    # 归一化分数差。最大可能分数差为 BOARD_SIZE * BOARD_SIZE (所有格子都被一方涂色)
    max_possible_score = BOARD_SIZE * BOARD_SIZE
    normalized_score = score_difference / max_possible_score

    if debug_mode:
        print(f"    [DEBUG calculate_score_target] Player: {current_player_at_step}")
        print(f"    [DEBUG calculate_score_target] Black Painted Count: {black_painted_count}")
        print(f"    [DEBUG calculate_score_target] White Painted Count: {white_painted_count}")
        print(f"    [DEBUG calculate_score_target] Raw Score Difference: {score_difference}")
        print(f"    [DEBUG calculate_score_target] Normalized Score: {normalized_score:.8f}") # More precision for debug

    return normalized_score


def get_nn_output_shapes():
    """
    返回神经网络输出的预期形状和范围。
    现在只包含策略、价值和所有权头的形状。

    Returns:
        dict: 包含 'policy_shape', 'value_shape', 'ownership_shape' 的字典。
              - 'policy_shape': (BOARD_SIZE * BOARD_SIZE,)
              - 'value_shape': (1,)
              - 'ownership_shape': (BOARD_SIZE, BOARD_SIZE)
              - 'output_range_value_ownership': (-1, 1)
              - 'output_range_policy': (0, 1)
    """
    return {
        'policy_shape': (BOARD_SIZE * BOARD_SIZE,),  # 注意这里是展平后的形状
        'value_shape': (1,),
        'ownership_shape': (BOARD_SIZE, BOARD_SIZE),
        'output_range_value_ownership': (-1, 1),  # 价值、所有权输出范围
        'output_range_policy': (0, 1)  # 策略输出范围
    }


# 示例用法 (在实际神经网络训练中，这些将是模型的输入和输出)
if __name__ == "__main__":
    # 模拟一个棋盘状态和游戏信息
    from popucom_chess import initialize_board, BLACK_PIECE_INDEX, WHITE_PIECE_INDEX, BLACK_PAINTED_INDEX, \
        WHITE_PAINTED_INDEX, UNPAINTED_INDEX

    initial_board = initialize_board()

    # 放置一些棋子和涂色地板用于测试
    initial_board[0, 0, BLACK_PIECE_INDEX] = 1  # 黑子在 (0,0)
    initial_board[0, 0, UNPAINTED_INDEX] = 0  # 移除未涂色标记
    initial_board[1, 1, WHITE_PIECE_INDEX] = 1  # 白子在 (1,1)
    initial_board[1, 1, UNPAINTED_INDEX] = 0
    initial_board[2, 2, BLACK_PAINTED_INDEX] = 1  # 黑地板在 (2,2)
    initial_board[2, 2, UNPAINTED_INDEX] = 0
    initial_board[3, 3, WHITE_PAINTED_INDEX] = 1  # 白地板在 (3,3)
    initial_board[3, 3, UNPAINTED_INDEX] = 0
    initial_board[4, 4, BLACK_PAINTED_INDEX] = 1  # 更多黑地板
    initial_board[4, 4, UNPAINTED_INDEX] = 0
    initial_board[BOARD_SIZE - 1, BOARD_SIZE - 1, WHITE_PAINTED_INDEX] = 1  # 更多白地板
    initial_board[BOARD_SIZE - 1, BOARD_SIZE - 1, UNPAINTED_INDEX] = 0

    test_remaining_moves = {'black': 20, 'white': 22}
    test_current_player = BLACK_PLAYER

    # 转换游戏状态为输入张量
    input_tensor = transform_game_state_to_input_tensor(
        initial_board, test_remaining_moves, test_current_player
    )

    print(f"神经网络输入张量的形状: {input_tensor.shape}")
    print(f"输入张量 (前5个通道，棋盘状态):")
    # 打印棋子和地板状态
    print("黑子通道 (Channel 0):\n", input_tensor[:, :, 0])
    print("白子通道 (Channel 1):\n", input_tensor[:, :, 1])
    print("黑地板通道 (Channel 2):\n", input_tensor[:, :, 2])
    print("白地板通道 (Channel 3):\n", input_tensor[:, :, 3])
    print("空地通道 (Channel 4):\n", input_tensor[:, :, 4])

    print("\n轮到黑行动通道 (Channel 5):\n", input_tensor[:, :, 5][0, 0])  # 只取一个值即可，因为是广播的
    print("轮到白行动通道 (Channel 6):\n", input_tensor[:, :, 6][0, 0])

    print("\n黑剩余行动力通道 (Channel 7):\n", input_tensor[:, :, 7][0, 0])
    print("白剩余行动力通道 (Channel 8):\n", input_tensor[:, :, 8][0, 0])

    print("\n黑已占据地板数量通道 (Channel 9):\n", input_tensor[:, :, 9][0, 0])
    print("白已占据地板数量通道 (Channel 10):\n", input_tensor[:, :, 10][0, 0])

    # 计算所有权目标
    ownership_target = calculate_ownership_target(initial_board, test_current_player)
    print(f"\n所有权目标形状: {ownership_target.shape}")
    print(f"所有权目标示例 (以黑方视角):\n{ownership_target}")

    # 计算分数目标 (启用调试模式)
    score_target = calculate_score_target(initial_board, test_current_player, debug_mode=True)
    print(f"\n分数目标 (以黑方视角): {score_target}")

    # 获取神经网络输出的预期形状
    output_shapes = get_nn_output_shapes()
    print(f"\n神经网络策略输出预期形状: {output_shapes['policy_shape']}")
    print(f"神经网络价值输出预期形状: {output_shapes['value_shape']}")
    print(f"神经网络所有权输出预期形状: {output_shapes['ownership_shape']}")
    # print(f"神经网络分数输出预期形状: {output_shapes['score_shape']}") # 移除分数头
    print(f"神经网络价值、所有权输出预期范围: {output_shapes['output_range_value_ownership']}")
    print(f"神经网络策略输出预期范围: {output_shapes['output_range_policy']}")
