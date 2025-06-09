#self_play_worker.py

import torch
import numpy as np
import os
import pickle
import random
import sys
import multiprocessing

# 导入必要的模块和所有常量
# 这些常量直接导入，以便在主进程和 _play_single_game 外部使用
# 在 _play_single_game 内部，我们将再次明确引用它们或通过参数传递
from popucom_chess import (
    BOARD_SIZE,
    BLACK_PLAYER, WHITE_PLAYER,
    GAME_RESULT_WIN, GAME_RESULT_LOSS, GAME_RESULT_DRAW, GAME_RESULT_NONE,
    initialize_board, do_move, check_game_over,
    BLACK_PIECE_INDEX, WHITE_PIECE_INDEX,
    BLACK_PAINTED_INDEX, WHITE_PAINTED_INDEX, UNPAINTED_INDEX,
    get_painted_tile_counts  # 用于计算最终分数
)
from popucom_nn_interface import (
    transform_game_state_to_input_tensor,
    calculate_ownership_target,
    calculate_score_target
)
from popucom_puct import MCTSSearcher, MCTSNode
from popucom_nn_model import PomPomNN

# --- Configuration Parameters ---
NUM_SELF_PLAY_GAMES = 200  # Number of self-play games to generate per iteration
MODELS_DIR = "models"
DATA_DIR = "self_play_data"

# PUCT Search Parameters
PUCT_SIMULATIONS = 200
PUCT_C_PUCT = 1.0
PUCT_TEMPERATURE_INITIAL = 0.7
PUCT_TEMPERATURE_FINAL = 0.1
PUCT_TEMPERATURE_DECAY_GAMES_RATIO = 0.5

puct_dirichlet_epsilon=0.25
puct_dirichlet_alpha=0.03

# Neural Network Model Parameters (must match PomPomNN definition)
NN_NUM_RES_BLOCKS = 6
NN_NUM_FILTERS = 96

# Multiprocessing Parameters
NUM_WORKERS = 25

# Value Target Strategy:
# If True: For non-terminal states, use MCTS Q-value as training target for value head.
# If False: For non-terminal states, use the final normalized tile difference score as training target.
USE_MCTS_Q_FOR_VALUE_TARGET = False

# New Configuration: If True, the value target for ALL states (including terminal)
# will be the win/loss/draw (1.0/-1.0/0.0) outcome, overriding other value targets.
# This is for later stage fine-tuning towards pure win/loss.
USE_WIN_LOSS_FOR_VALUE_TARGET_LATER_STAGE = False


# Helper function: Load the latest model
def load_latest_model_for_selfplay(model, models_dir, device):
    """Loads the latest model from the specified directory for self-play."""
    if not os.path.exists(models_dir):
        print(f"Model directory '{models_dir}' not found. Using randomly initialized model.")
        return model

    model_files = [f for f in os.listdir(models_dir) if f.startswith("model_iter_") and f.endswith(".pth")]

    if not model_files:
        print(f"No previous models found in '{models_dir}'. Using randomly initialized model.")
        return model

    latest_iteration = -1
    latest_model_path = None
    for f_name in model_files:
        try:
            iter_str = f_name.replace("model_iter_", "").replace(".pth", "")
            iteration = int(iter_str)
            if iteration > latest_iteration:
                latest_iteration = iteration
                latest_model_path = os.path.join(models_dir, f_name)
        except ValueError:
            continue

    if latest_model_path:
        try:
            model.load_state_dict(torch.load(latest_model_path, map_location=device), strict=False)
            print(f"Model loaded from {latest_model_path} (latest iteration: {latest_iteration})")
        except Exception as e:
            print(f"Error loading model from {latest_model_path}: {e}")
            print("Using randomly initialized model instead.")
    else:
        print(f"No valid model files found in '{models_dir}'. Using randomly initialized model.")

    return model


# --- Worker function for a single game ---
def _play_single_game(worker_args):
    """
    Plays a single game and returns its data. This function runs in a separate process.
    """
    (game_id_global, model_state_dict, puct_sims, c_puct, temp_initial, temp_final,
     temp_decay_games_ratio, num_self_play_games_total, nn_num_res_blocks, nn_num_filters,
     chess_constants_dict, use_mcts_q_for_value_target, use_win_loss_for_value_target_later_stage) = worker_args

    # --- Set random seeds for reproducibility within each worker process ---
    # This is crucial for deterministic behavior when temperature is 0 and there are ties.
    seed = game_id_global  # Use a unique seed for each game
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # --- End seed setting ---

    # Unpack all board constants to local variables
    BOARD_SIZE = chess_constants_dict['BOARD_SIZE']
    BLACK_PLAYER = chess_constants_dict['BLACK_PLAYER']
    WHITE_PLAYER = chess_constants_dict['WHITE_PLAYER']
    GAME_RESULT_WIN = chess_constants_dict['GAME_RESULT_WIN']
    GAME_RESULT_LOSS = chess_constants_dict['GAME_RESULT_LOSS']
    GAME_RESULT_DRAW = chess_constants_dict['GAME_RESULT_DRAW']
    GAME_RESULT_NONE = chess_constants_dict['GAME_RESULT_NONE']
    BLACK_PIECE_INDEX = chess_constants_dict['BLACK_PIECE_INDEX']
    WHITE_PIECE_INDEX = chess_constants_dict['WHITE_PIECE_INDEX']
    BLACK_PAINTED_INDEX = chess_constants_dict['BLACK_PAINTED_INDEX']
    WHITE_PAINTED_INDEX = chess_constants_dict['WHITE_PAINTED_INDEX']
    UNPAINTED_INDEX = chess_constants_dict['UNPAINTED_INDEX']

    # Re-import necessary functions into the subprocess's global scope
    from popucom_chess import (
        initialize_board, do_move, check_game_over
    )
    from popucom_nn_interface import (
        transform_game_state_to_input_tensor,
        calculate_ownership_target,
        calculate_score_target
    )
    from popucom_puct import MCTSSearcher, MCTSNode
    from popucom_nn_model import PomPomNN

    # Each worker needs its own model instance and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PomPomNN(num_res_blocks=nn_num_res_blocks, num_filters=nn_num_filters)

    try:
        model.load_state_dict(model_state_dict, strict=False)
    except Exception as e:
        print(f"Worker {os.getpid()} error loading model state dict: {e}")
        print(f"Worker {os.getpid()} will use random weights.")
        pass

    model.eval()
    model.to(device)

    # Pass the new flag to MCTSSearcher as well
    mcts_searcher = MCTSSearcher(model=model, c_puct=c_puct,
                                 use_win_loss_target=use_win_loss_for_value_target_later_stage,
                                 dirichlet_epsilon=puct_dirichlet_epsilon,dirichlet_alpha=puct_dirichlet_alpha
                                 )

    board = initialize_board()
    remaining_moves = {BLACK_PLAYER: 25, WHITE_PLAYER: 25}
    current_player = BLACK_PLAYER
    game_steps_data = []

    game_over = False
    game_result_dict = {BLACK_PLAYER: GAME_RESULT_NONE, WHITE_PLAYER: GAME_RESULT_NONE}
    move_count = 0

    temp_decay_games_threshold = int(num_self_play_games_total * temp_decay_games_ratio)

    def get_current_temperature_local_func(current_game_idx_in_iteration):
        if current_game_idx_in_iteration < temp_decay_games_threshold:
            return temp_initial
        else:
            progress = (current_game_idx_in_iteration - temp_decay_games_threshold) / (
                    num_self_play_games_total - temp_decay_games_threshold)
            return temp_initial - progress * (temp_initial - temp_final)

    board_states_at_steps = []

    while not game_over:
        move_count += 1
        current_temp = get_current_temperature_local_func(game_id_global % num_self_play_games_total)

        root_node = MCTSNode(board, remaining_moves, current_player)
        mcts_searcher.run_search(root_node, puct_sims)
        mcts_policy_distribution = mcts_searcher.get_policy_distribution(root_node, temperature=current_temp)

        mcts_q_value_for_this_step = root_node.Q

        flat_policy = mcts_policy_distribution.flatten()
        legal_moves = mcts_searcher._get_legal_moves(board, current_player)  # Use internal method

        legal_move_flat_indices = [r * BOARD_SIZE + c for r, c in legal_moves]

        if not legal_move_flat_indices:
            game_over = True
            game_result_dict = check_game_over(board, remaining_moves,
                                               BLACK_PLAYER if current_player == WHITE_PLAYER else WHITE_PLAYER)
            break

        legal_probs = flat_policy[legal_move_flat_indices]
        if np.sum(legal_probs) == 0:
            chosen_move_flat_idx = random.choice(legal_move_flat_indices)
        else:
            legal_probs = legal_probs / np.sum(legal_probs)  # Normalize
            chosen_move_flat_idx = np.random.choice(legal_move_flat_indices, p=legal_probs)

        chosen_x = chosen_move_flat_idx // BOARD_SIZE
        chosen_y = chosen_move_flat_idx % BOARD_SIZE

        current_input_features = transform_game_state_to_input_tensor(board, remaining_moves, current_player)

        game_steps_data.append({
            'input_features': current_input_features,
            'mcts_policy': mcts_policy_distribution,
            'player_at_step': current_player,
            'game_outcome_value': None,  # Will be set after game ends
            'mcts_q_value_at_this_step': mcts_q_value_for_this_step,
            'ownership_target': None,
            # 'score_target': None # Removed as score_head is removed and value_head handles this
        })
        board_states_at_steps.append(board.copy())

        board, remaining_moves, _ = do_move(board, chosen_x, chosen_y, remaining_moves, current_player)

        game_result_dict = check_game_over(board, remaining_moves, current_player)
        if game_result_dict[BLACK_PLAYER] != GAME_RESULT_NONE or game_result_dict[WHITE_PLAYER] != GAME_RESULT_NONE:
            game_over = True

        current_player = WHITE_PLAYER if current_player == BLACK_PLAYER else BLACK_PLAYER

    # --- Game ends, calculate and assign final target values ---
    final_board_state = board

    for i, step_data in enumerate(game_steps_data):
        player_at_step = step_data['player_at_step']

        # Value Target (game_outcome_value)
        if use_win_loss_for_value_target_later_stage:
            if game_result_dict[player_at_step] == GAME_RESULT_WIN:
                step_data['game_outcome_value'] = 1.0
            elif game_result_dict[player_at_step] == GAME_RESULT_LOSS:
                step_data['game_outcome_value'] = -1.0
            else:
                step_data['game_outcome_value'] = 0.0
        elif not game_over:
            if use_mcts_q_for_value_target:
                step_data['game_outcome_value'] = step_data['mcts_q_value_at_this_step']
            else:
                step_data['game_outcome_value'] = calculate_score_target(final_board_state, player_at_step)
        else:
            step_data['game_outcome_value'] = calculate_score_target(final_board_state, player_at_step)

        # Ownership Target (remains independent)
        step_data['ownership_target'] = calculate_ownership_target(final_board_state, player_at_step)

        # Removed score_target as it's now handled by game_outcome_value (for value_head)
        # step_data['score_target'] = calculate_score_target(final_board_state, player_at_step)

    return game_id_global, game_steps_data


class SelfPlayWorker:
    """
    Class responsible for generating self-play data.
    It uses the current neural network model and PUCT search to play games and save game data.
    """

    def __init__(self, models_dir, data_dir, puct_sims, c_puct, temp_initial, temp_final, temp_decay_games_ratio,
                 num_workers, use_mcts_q_for_value_target, use_win_loss_for_value_target_later_stage):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.puct_sims = puct_sims
        self.c_puct = c_puct
        self.temp_initial = temp_initial
        self.temp_final = temp_final
        self.temp_decay_games_ratio = temp_decay_games_ratio
        self.num_workers = num_workers
        self.use_mcts_q_for_value_target = use_mcts_q_for_value_target
        self.use_win_loss_for_value_target_later_stage = use_win_loss_for_value_target_later_stage

        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Self-play data will be saved to: {os.path.abspath(self.data_dir)}")

    def _save_game_data(self, game_data_steps, game_id):
        file_name = os.path.join(self.data_dir, f"game_{game_id:08d}.pkl")
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(game_data_steps, f)
        except Exception as e:
            print(f"Error saving game {game_id} data: {e}")

    def run_self_play(self, num_games_to_generate, start_global_game_id):
        print(f"Starting generation of {num_games_to_generate} self-play games using {self.num_workers} workers...")
        print(f"Game IDs for this round will start from {start_global_game_id}.")
        print(
            f"Value target mode: {'Win/Loss' if self.use_win_loss_for_value_target_later_stage else ('MCTS Q-value' if self.use_mcts_q_for_value_target else 'Score Difference')}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PomPomNN(num_res_blocks=NN_NUM_RES_BLOCKS, num_filters=NN_NUM_FILTERS)
        model = load_latest_model_for_selfplay(model, self.models_dir, device)
        model.eval()
        model.to(device)

        model_state_dict = model.state_dict()

        chess_constants_to_pass = {
            'BOARD_SIZE': BOARD_SIZE,
            'BLACK_PLAYER': BLACK_PLAYER,
            'WHITE_PLAYER': WHITE_PLAYER,
            'GAME_RESULT_WIN': GAME_RESULT_WIN,
            'GAME_RESULT_LOSS': GAME_RESULT_LOSS,
            'GAME_RESULT_DRAW': GAME_RESULT_DRAW,
            'GAME_RESULT_NONE': GAME_RESULT_NONE,
            'BLACK_PIECE_INDEX': BLACK_PIECE_INDEX,
            'WHITE_PIECE_INDEX': WHITE_PIECE_INDEX,
            'BLACK_PAINTED_INDEX': BLACK_PAINTED_INDEX,
            'WHITE_PAINTED_INDEX': WHITE_PAINTED_INDEX,
            'UNPAINTED_INDEX': UNPAINTED_INDEX
        }

        worker_args_list = []
        for i in range(start_global_game_id, start_global_game_id + num_games_to_generate):
            worker_args_list.append((
                i,
                model_state_dict,
                self.puct_sims,
                self.c_puct,
                self.temp_initial,
                self.temp_final,
                self.temp_decay_games_ratio,
                NUM_SELF_PLAY_GAMES,
                NN_NUM_RES_BLOCKS,
                NN_NUM_FILTERS,
                chess_constants_to_pass,
                self.use_mcts_q_for_value_target,
                self.use_win_loss_for_value_target_later_stage
            ))

        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for game_id, game_steps_data in pool.imap_unordered(_play_single_game, worker_args_list):
                self._save_game_data(game_steps_data, game_id)
                print(f"Game {game_id + 1} (global) finished and saved.")
                if (game_id - start_global_game_id + 1) % 10 == 0:
                    print(
                        f"--- Current iteration has completed {(game_id - start_global_game_id + 1)}/{num_games_to_generate} games ---")

        print(f"\nSuccessfully generated {num_games_to_generate} self-play games.")


if __name__ == "__main__":
    print("--- Pom Pom Chess Self-Play Data Generator ---")

    current_models_dir = MODELS_DIR
    num_workers_arg = NUM_WORKERS
    start_global_game_id_arg = 0
    use_mcts_q_for_value_target_arg = USE_MCTS_Q_FOR_VALUE_TARGET
    use_win_loss_for_value_target_later_stage_arg = USE_WIN_LOSS_FOR_VALUE_TARGET_LATER_STAGE

    if len(sys.argv) > 1:
        current_models_dir = sys.argv[1]
        if len(sys.argv) > 2:
            try:
                num_workers_arg = int(sys.argv[2])
                if len(sys.argv) > 3:
                    start_global_game_id_arg = int(sys.argv[3])
                    if len(sys.argv) > 4:
                        use_mcts_q_for_value_target_arg = bool(int(sys.argv[4]))
                        if len(sys.argv) > 5:
                            use_win_loss_for_value_target_later_stage_arg = bool(int(sys.argv[5]))
            except ValueError:
                print(
                    "Usage: python self_play_worker.py [models_dir] [num_workers] [start_global_game_id] [use_mcts_q_for_value_target (0 or 1)] [use_win_loss_for_value_target_later_stage (0 or 1)]")
                sys.exit(1)

    worker = SelfPlayWorker(
        models_dir=current_models_dir,
        data_dir=DATA_DIR,
        puct_sims=PUCT_SIMULATIONS,
        c_puct=PUCT_C_PUCT,
        temp_initial=PUCT_TEMPERATURE_INITIAL,
        temp_final=PUCT_TEMPERATURE_FINAL,
        temp_decay_games_ratio=PUCT_TEMPERATURE_DECAY_GAMES_RATIO,
        num_workers=num_workers_arg,
        use_mcts_q_for_value_target=use_mcts_q_for_value_target_arg,
        use_win_loss_for_value_target_later_stage=use_win_loss_for_value_target_later_stage_arg
    )

    worker.run_self_play(NUM_SELF_PLAY_GAMES, start_global_game_id_arg)
