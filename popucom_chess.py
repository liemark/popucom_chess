#popucom_chess

import numpy as np
from numba import jit, int64, float64, boolean, int8, types  # 导入 Numba 和 types 模块

# 棋盘尺寸常量
BOARD_SIZE = 9

# 棋盘单元格中棋子/地板状态的索引常量
# 一个单元格表示为 [has_black_piece, has_white_piece, is_black_painted, is_white_painted, is_unpainted]
# 现在这些将是 NumPy 数组的第三维索引
BLACK_PIECE_INDEX = 0  # 黑色棋子是否存在 (1: 存在, 0: 不存在)
WHITE_PIECE_INDEX = 1  # 白色棋子是否存在 (1: 存在, 0: 不存在)
BLACK_PAINTED_INDEX = 2  # 地板是否被涂成黑色 (1: 是, 0: 否)
WHITE_PAINTED_INDEX = 3  # 地板是否被涂成白色 (1: 是, 0: 否)
UNPAINTED_INDEX = 4  # 地板是否未被涂色 (1: 是, 0: 不存在)

# 玩家常量
BLACK_PLAYER = 'black'
WHITE_PLAYER = 'white'

# 游戏结果常量
GAME_RESULT_WIN = 'Win'  # 胜利
GAME_RESULT_LOSS = 'Loss'  # 失败
GAME_RESULT_DRAW = 'Draw'  # 平局
GAME_RESULT_NONE = 'None'  # 游戏未结束


def initialize_board():
    """
    初始化一个9x9的泡姆棋盘。
    棋盘现在是一个 NumPy 数组，形状为 (BOARD_SIZE, BOARD_SIZE, 5)。
    初始时所有单元格都没有棋子，且未被涂色。

    Returns:
        np.ndarray: 初始化后的棋盘。
    """
    # 使用 dtype=np.int8 节省内存，因为值只有 0 或 1
    board = np.zeros((BOARD_SIZE, BOARD_SIZE, 5), dtype=np.int8)
    # 初始时所有地板都未涂色 (UNPAINTED_INDEX = 4)
    board[:, :, UNPAINTED_INDEX] = 1
    return board


@jit(boolean(int64, int64, int64), nopython=True, cache=True)
def _is_valid_coord_jitted(x, y, board_size):
    """
    检查给定坐标 (x, y) 是否在棋盘范围内。Numba JIT 优化版本。

    Args:
        x (int): 行坐标。
        y (int): 列坐标。
        board_size (int): 棋盘边长。

    Returns:
        bool: 如果坐标合法则返回 True，否则返回 False。
    """
    return 0 <= x < board_size and 0 <= y < board_size


# Numba 优化：为了使 _is_valid_coord_jitted 可用于类方法，我们通常将其声明为静态方法
# 但为了保持原始接口（外部可能调用 _is_valid_coord），我们仍可以有一个 Python 包装
def _is_valid_coord(x, y):
    """
    检查给定坐标 (x, y) 是否在棋盘范围内 (Python 包装版本)。
    """
    return _is_valid_coord_jitted(x, y, BOARD_SIZE)


@jit(boolean(int8[:], int64, int64, int64, int64, int64, boolean), nopython=True, cache=True)
def _check_valid_move_single_cell_jitted(cell_state, black_piece_idx, white_piece_idx, black_painted_idx,
                                         white_painted_idx, unpainted_idx, is_black_player):
    """
    检查单个单元格对于落子是否合法。Numba JIT 优化版本。
    此函数仅供 check_valid_move_for_next_player_jitted 内部使用。
    """
    # 2. 该位置没有棋子
    if cell_state[black_piece_idx] == 1 or cell_state[white_piece_idx] == 1:
        return False

    # 3. 该位置的地板颜色与玩家颜色匹配或未涂色
    if is_black_player:
        return cell_state[black_painted_idx] == 1 or cell_state[unpainted_idx] == 1
    else:
        return cell_state[white_painted_idx] == 1 or cell_state[unpainted_idx] == 1


@jit(boolean(int8[:, :, :], int64, int64, int64, int64, int64, int64, int64, int64, boolean), nopython=True, cache=True)
def check_valid_move_for_next_player_jitted(board_arr, x, y, board_size, black_piece_idx, white_piece_idx,
                                            black_painted_idx, white_painted_idx, unpainted_idx, is_black_player):
    """
    检查给定位置 (x, y) 对于当前玩家是否是合法的落子位置。Numba JIT 优化版本。

    Args:
        board_arr (np.ndarray): NumPy 棋盘数组。
        x (int): 行坐标。
        y (int): 列坐标。
        board_size (int): 棋盘边长。
        ... (其他常量索引)
        is_black_player (bool): 当前玩家是否为黑方。

    Returns:
        bool: 如果是合法落子位置则返回 True，否则返回 False。
    """
    if not _is_valid_coord_jitted(x, y, board_size):
        return False

    # 直接访问 NumPy 数组的切片作为单元格状态
    cell = board_arr[x, y, :]
    return _check_valid_move_single_cell_jitted(cell, black_piece_idx, white_piece_idx, black_painted_idx,
                                                white_painted_idx, unpainted_idx, is_black_player)


# 公共接口，外部调用时仍然是 Python 包装
def check_valid_move_for_next_player(board, x, y, player):
    is_black_player_bool = (player == BLACK_PLAYER)
    # 确保传入的是 NumPy 数组
    if not isinstance(board, np.ndarray):
        raise TypeError("Board must be a NumPy array for check_valid_move_for_next_player.")

    return check_valid_move_for_next_player_jitted(
        board, x, y, BOARD_SIZE,
        BLACK_PIECE_INDEX, WHITE_PIECE_INDEX,
        BLACK_PAINTED_INDEX, WHITE_PAINTED_INDEX,
        UNPAINTED_INDEX,
        is_black_player_bool
    )


@jit(int64(int8[:, :, :], int64), nopython=True, cache=True)
def _count_painted_cells_jitted(board_arr, painted_index):
    """
    计算棋盘上指定颜色涂色地板的数量。Numba JIT 优化版本。
    """
    count = 0
    for r in range(board_arr.shape[0]):
        for c in range(board_arr.shape[1]):
            if board_arr[r, c, painted_index] == 1:
                count += 1
    return count


def _count_painted_cells(board, player_color):
    """
    计算棋盘上指定颜色涂色地板的数量 (Python 包装版本)。
    """
    painted_index = BLACK_PAINTED_INDEX if player_color == BLACK_PLAYER else WHITE_PAINTED_INDEX
    return _count_painted_cells_jitted(board, painted_index)


def get_painted_tile_counts(board):
    """
    获取棋盘上黑方和白方各自涂色地板的数量。

    Args:
        board (np.ndarray): 当前棋盘状态 (NumPy 数组)。

    Returns:
        dict: 包含 'black' 和 'white' 涂色地板数量的字典。
    """
    black_painted_count = _count_painted_cells(board, BLACK_PLAYER)
    white_painted_count = _count_painted_cells(board, WHITE_PLAYER)
    return {BLACK_PLAYER: black_painted_count, WHITE_PLAYER: white_painted_count}


# 定义 _get_elimination_targets_jitted 函数的精确 Numba 签名
# 这样可以避免 Numba 解析器可能出现的歧义
_get_elimination_targets_jitted_sig = types.Tuple((types.boolean[:, :], types.int8[:, :]))(
    types.int8[:, :, :],  # board_arr
    types.int64,  # x
    types.int64,  # y
    types.int64,  # player_piece_index
    types.int64,  # board_size
    types.int64,  # black_piece_idx
    types.int64,  # white_piece_idx
    types.int64,  # black_painted_idx
    types.int64,  # white_painted_idx
    types.int64  # unpainted_idx
)


@jit(_get_elimination_targets_jitted_sig, nopython=True, cache=True)
def _get_elimination_targets_jitted(board_arr, x, y, player_piece_index, board_size, black_piece_idx, white_piece_idx,
                                    black_painted_idx, white_painted_idx, unpainted_idx):
    """
    检查在 (x, y) 落子后，是否有3个或更多同色棋子连成一线，并返回所有需要消除的棋子坐标
    以及触发消除的“方向”列表。Numba JIT 优化版本。
    此函数返回一个布尔 NumPy 数组，指示哪些位置应被消除，以及一个固定的二维数组表示受影响方向。
    由于 Numba 不支持列表的动态增长和集合，这里改为返回一个二维布尔数组表示消除目标，
    以及一个固定的二维数组表示受 affected_painting_directions.
    """
    # 存储要消除的棋子位置，用布尔值表示
    elimination_mask = np.zeros((board_size, board_size), dtype=np.bool_)
    # 存储受影响的涂色方向 (dx, dy)
    # 使用一个固定大小的数组，例如 max_directions * 2 for dx,dy pairs
    # 然后用一个计数器表示实际有多少个有效方向
    affected_painting_directions = np.zeros((8, 2),
                                            dtype=np.int8)  # 8个半方向 (0,1),(0,-1),(1,0),(-1,0), (1,1),(-1,-1),(1,-1),(-1,1)
    num_affected_directions = 0

    # 检查方向：水平、垂直、两个对角线
    primary_directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dx, dy in primary_directions:
        line_pieces_count = 0
        current_line_coords = np.zeros((board_size * 2 - 1, 2), dtype=np.int8)  # 存储一条线上的坐标，最多17个 (9+9-1)
        current_line_idx = 0

        # 向后遍历
        curr_x, curr_y = x - dx, y - dy
        while _is_valid_coord_jitted(curr_x, curr_y, board_size) and board_arr[curr_x, curr_y, player_piece_index] == 1:
            current_line_coords[current_line_idx, 0] = curr_x
            current_line_coords[current_line_idx, 1] = curr_y
            current_line_idx += 1
            line_pieces_count += 1
            curr_x -= dx
            curr_y -= dy

        # 添加落子点
        current_line_coords[current_line_idx, 0] = x
        current_line_coords[current_line_idx, 1] = y
        current_line_idx += 1
        line_pieces_count += 1

        # 向前遍历
        curr_x, curr_y = x + dx, y + + dy
        while _is_valid_coord_jitted(curr_x, curr_y, board_size) and board_arr[curr_x, curr_y, player_piece_index] == 1:
            current_line_coords[current_line_idx, 0] = curr_x
            current_line_coords[current_line_idx, 1] = curr_y
            current_line_idx += 1
            line_pieces_count += 1
            curr_x += dx
            curr_y += dy

        if line_pieces_count >= 3:
            # 如果达到消除条件，则将这些棋子标记为待消除
            for i in range(current_line_idx):
                r, c = current_line_coords[i, 0], current_line_coords[i, 1]
                elimination_mask[r, c] = True

            # 添加受影响的方向
            if num_affected_directions < 8:
                affected_painting_directions[num_affected_directions, 0] = dx
                affected_painting_directions[num_affected_directions, 1] = dy
                num_affected_directions += 1
            if num_affected_directions < 8:  # Also add the opposite direction
                affected_painting_directions[num_affected_directions, 0] = -dx
                affected_painting_directions[num_affected_directions, 1] = -dy
                num_affected_directions += 1

    return elimination_mask, affected_painting_directions[:num_affected_directions]  # 返回有效的部分


def _get_elimination_targets(board, x, y, player):
    """
    检查在 (x, y) 落子后，是否有3个或更多同色棋子连成一线，并返回所有需要消除的棋子坐标
    以及触发消除的“方向”列表 (Python 包装版本)。
    """
    player_piece_index = BLACK_PIECE_INDEX if player == BLACK_PLAYER else WHITE_PIECE_INDEX

    elimination_mask, affected_dirs_np = _get_elimination_targets_jitted(
        board, x, y, player_piece_index, BOARD_SIZE,
        BLACK_PIECE_INDEX, WHITE_PIECE_INDEX, BLACK_PAINTED_INDEX, WHITE_PAINTED_INDEX, UNPAINTED_INDEX
    )

    eliminated_coords = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if elimination_mask[r, c]:
                eliminated_coords.append((r, c))

    # 转换 Numba 返回的 NumPy 数组为列表的元组
    affected_painting_directions = []
    for i in range(affected_dirs_np.shape[0]):
        affected_painting_directions.append(tuple(affected_dirs_np[i]))

    return eliminated_coords, affected_painting_directions


@jit(int8[:, :, :](int8[:, :, :], boolean[:, :], int64, int64), nopython=True, cache=True)
def _perform_elimination_jitted(board_arr, elimination_mask, black_piece_idx, white_piece_idx):
    """
    从棋盘上移除指定坐标的棋子。Numba JIT 优化版本。
    直接操作 NumPy 数组。
    """
    for r in range(board_arr.shape[0]):
        for c in range(board_arr.shape[1]):
            if elimination_mask[r, c]:
                board_arr[r, c, black_piece_idx] = 0
                board_arr[r, c, white_piece_idx] = 0
    return board_arr


def _perform_elimination(board, eliminated_coords):
    """
    从棋盘上移除指定坐标的棋子 (Python 包装版本)。
    """
    elimination_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)
    for r, c in eliminated_coords:
        elimination_mask[r, c] = True
    return _perform_elimination_jitted(board, elimination_mask, BLACK_PIECE_INDEX, WHITE_PIECE_INDEX)


# Corrected signature for _perform_painting_jitted (11 arguments)
_perform_painting_jitted_sig = types.int8[:, :, :](
    types.int8[:, :, :],  # board_arr
    types.int64,  # x
    types.int64,  # y
    types.int64,  # player_painted_index
    types.int64,  # opponent_piece_index
    types.int64,  # board_size
    types.Array(types.int8, 2, 'C', True),  # painting_directions_to_use_arr (2D int8 C-contiguous array)
    types.int64,  # num_painting_directions
    types.int64,  # black_painted_idx
    types.int64,  # white_painted_idx
    types.int64  # unpainted_idx
)


@jit(_perform_painting_jitted_sig, nopython=True, cache=True)
def _perform_painting_jitted(board_arr, x, y, player_painted_index, opponent_piece_index, board_size,
                             painting_directions_to_use_arr, num_painting_directions,
                             black_painted_idx, white_painted_idx, unpainted_idx):
    """
    以 (x, y) 为中心，沿指定的方向涂色。Numba JIT 优化版本。
    直接操作 NumPy 数组。
    """
    for i in range(num_painting_directions):
        dx, dy = painting_directions_to_use_arr[i, 0], painting_directions_to_use_arr[i, 1]
        curr_x, curr_y = x, y
        while _is_valid_coord_jitted(curr_x, curr_y, board_size):
            if board_arr[curr_x, curr_y, opponent_piece_index] == 1:
                break

            board_arr[curr_x, curr_y, black_painted_idx] = 0
            board_arr[curr_x, curr_y, white_painted_idx] = 0
            board_arr[curr_x, curr_y, unpainted_idx] = 0
            board_arr[curr_x, curr_y, player_painted_index] = 1

            curr_x += dx
            curr_y += dy
    return board_arr


def _perform_painting(board, x, y, player, painting_directions_to_use):
    """
    以 (x, y) 为中心，沿指定的方向涂色 (Python 包装版本)。
    """
    # FIX: Corrected WHITE_PIECE_INDEX to WHITE_PAINTED_INDEX
    player_painted_index = BLACK_PAINTED_INDEX if player == BLACK_PLAYER else WHITE_PAINTED_INDEX
    opponent_piece_index = WHITE_PIECE_INDEX if player == BLACK_PLAYER else BLACK_PIECE_INDEX

    # 将方向列表转换为 NumPy 数组
    painting_directions_arr = np.array(painting_directions_to_use, dtype=np.int8)
    if painting_directions_arr.size == 0:  # Handle empty directions case for Numba
        painting_directions_arr = np.zeros((0, 2), dtype=np.int8)

    return _perform_painting_jitted(
        board, x, y, player_painted_index, opponent_piece_index, BOARD_SIZE,
        painting_directions_arr, painting_directions_arr.shape[0],
        BLACK_PAINTED_INDEX, WHITE_PAINTED_INDEX, UNPAINTED_INDEX
    )


def check_game_over(board, remaining_moves, current_player):
    """
    检查游戏是否结束并返回结果。
    结果以字典形式返回，包含黑白双方的胜负平状态。

    Args:
        board (np.ndarray): 当前棋盘状态 (NumPy 数组)。
        remaining_moves (dict): 字典，包含 'black' 和 'white' 玩家的剩余步数。
        current_player (str): 刚刚落子的玩家 ('black' 或 'white')。

    Returns:
        dict: 包含黑白双方游戏结果的字典，例如 {'black': 'Win', 'white': 'Loss'}。
              结果可以是 'Win', 'Loss', 'Draw', 或 'None'。
    """
    next_player = WHITE_PLAYER if current_player == BLACK_PLAYER else BLACK_PLAYER

    # 1. 检查所有步数是否用尽
    if remaining_moves[BLACK_PLAYER] == 0 and remaining_moves[WHITE_PLAYER] == 0:
        black_painted_count = _count_painted_cells(board, BLACK_PLAYER)
        white_painted_count = _count_painted_cells(board, WHITE_PLAYER)

        if black_painted_count > white_painted_count:
            return {BLACK_PLAYER: GAME_RESULT_WIN, WHITE_PLAYER: GAME_RESULT_LOSS}
        elif white_painted_count > black_painted_count:
            return {BLACK_PLAYER: GAME_RESULT_LOSS, WHITE_PLAYER: GAME_RESULT_WIN}
        else:
            return {BLACK_PLAYER: GAME_RESULT_DRAW, WHITE_PLAYER: GAME_RESULT_DRAW}

    # 2. 检查下一位玩家是否无法落子
    can_next_player_move = False
    # 使用 Numba 优化后的 check_valid_move_for_next_player
    is_next_player_black = (next_player == BLACK_PLAYER)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if check_valid_move_for_next_player_jitted(
                    board, r, c, BOARD_SIZE,
                    BLACK_PIECE_INDEX, WHITE_PIECE_INDEX,
                    BLACK_PAINTED_INDEX, WHITE_PAINTED_INDEX,
                    UNPAINTED_INDEX, is_next_player_black
            ):
                can_next_player_move = True
                break
        if can_next_player_move:
            break

    if not can_next_player_move:
        # 下一位玩家无法落子，当前玩家获胜
        if current_player == BLACK_PLAYER:
            return {BLACK_PLAYER: GAME_RESULT_WIN, WHITE_PLAYER: GAME_RESULT_LOSS}
        else:
            return {BLACK_PLAYER: GAME_RESULT_LOSS, WHITE_PLAYER: GAME_RESULT_WIN}

    # 3. 游戏尚未结束
    return {BLACK_PLAYER: GAME_RESULT_NONE, WHITE_PLAYER: GAME_RESULT_NONE}


def do_move(board, x, y, remaining_moves, player):
    """
    执行一步泡姆棋的落子操作。
    此函数将根据游戏规则更新棋盘状态、剩余步数，并判断游戏结果。
    棋盘现在是 NumPy 数组。

    Args:
        board (np.ndarray): 当前棋盘状态。
        x (int): 落子的行坐标 (0-8)。
        y (int): 落子的列坐标 (0-8)。
        remaining_moves (dict): 字典，包含 'black' 和 'white' 玩家的剩余步数。
        player (str): 当前落子的玩家 ('black' 或 'white')。

    Returns:
        tuple: (updated_board, updated_remaining_moves, game_result_for_current_player)
            updated_board (np.ndarray): 落子后棋盘的新状态。
            updated_remaining_moves (dict): 更新后的剩余步数。
            game_result_for_current_player (str): 游戏结果，'Win'/'Loss'/'Draw'/'None'。
    """
    # 创建棋盘的深拷贝，以便不修改原始传入的棋盘
    new_board = board.copy()

    # 更新剩余步数
    new_remaining_moves = remaining_moves.copy()
    if new_remaining_moves[player] > 0:
        new_remaining_moves[player] -= 1
    else:
        # 理论上不应该发生，因为游戏结束条件会处理步数用尽的情况
        print(f"警告: {player} 玩家尝试在步数用尽后落子。")

    # 1. 在指定位置放置棋子
    player_piece_index = BLACK_PIECE_INDEX if player == BLACK_PLAYER else WHITE_PIECE_INDEX
    new_board[x, y, player_piece_index] = 1

    # 2. 检查并执行消除 (三消)
    elimination_targets, affected_painting_directions = _get_elimination_targets(new_board, x, y, player)

    if elimination_targets:
        # 如果存在消除目标，则执行消除
        new_board = _perform_elimination(new_board, elimination_targets)
        # 3. 如果有消除发生，则以落子点 (x, y) 为中心，沿受影响的方向进行涂色
        new_board = _perform_painting(new_board, x, y, player, affected_painting_directions)

    # 4. 判断游戏是否结束
    game_results = check_game_over(new_board, new_remaining_moves, player)
    current_player_result = game_results[player]

    return new_board, new_remaining_moves, current_player_result
