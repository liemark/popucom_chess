import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import sys
import random  # Import random for choosing transformations

# Import NN model and interface
try:
    from popucom_nn_model import PomPomNN
    from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure 'popucom_nn_model.py' and 'popucom_nn_interface.py' are in the same directory.")
    sys.exit(1)

# --- Configuration Parameters ---
DATA_DIR = "self_play_data"  # Directory where self-play data is saved
MODELS_DIR = "models"  # Model saving directory
NUM_EPOCHS = 15  # Number of training epochs
BATCH_SIZE = 256  # Training batch size
LEARNING_RATE = 1e-5  # Learning rate for the optimizer
WEIGHT_DECAY = 1e-4  # L2 regularization for the optimizer
VALUE_LOSS_WEIGHT = 0.5  # Weight for the value loss in total loss (e.g., AlphaZero uses 1.0)
OWNERSHIP_LOSS_WEIGHT = 0.1  # Weight for the ownership loss (adjust as needed)
# SCORE_LOSS_WEIGHT is removed as score head is merged into value head
SLIDING_WINDOW_SIZE = 5000  # 滑动窗口大小，每次训练保留最新的游戏局数
# AI's neural network model parameters (used for training)
NN_NUM_RES_BLOCKS = 6
NN_NUM_FILTERS = 96


class SelfPlayDataset(Dataset):
    """
    Custom Dataset for loading self-play game data with data augmentation.
    Each item in the dataset corresponds to a single game step.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # 获取所有游戏文件，并按游戏ID排序
        all_game_files = []
        for f in os.listdir(data_dir):
            if f.startswith('game_') and f.endswith('.pkl'):
                try:
                    game_id = int(f.replace('game_', '').replace('.pkl', ''))
                    all_game_files.append((game_id, os.path.join(data_dir, f)))
                except ValueError:
                    continue  # 忽略不符合命名规范的文件

        all_game_files.sort(key=lambda x: x[0])  # 按游戏ID升序排序

        # 应用滑动窗口：只保留最新的 SLIDING_WINDOW_SIZE 局游戏
        if len(all_game_files) > SLIDING_WINDOW_SIZE:
            self.data_files = [path for game_id, path in all_game_files[-SLIDING_WINDOW_SIZE:]]
            print(f"Applying sliding window: keeping {SLIDING_WINDOW_SIZE} most recent games.")
        else:
            self.data_files = [path for game_id, path in all_game_files]

        self.all_steps = []

        print(f"Loading data from {len(self.data_files)} game files in {data_dir}...")
        if not self.data_files:
            print(
                f"Warning: No .pkl files found in {data_dir} or after applying sliding window. Please ensure self_play_worker.py has generated data.")
            return

        for file_path in self.data_files:
            try:
                with open(file_path, 'rb') as f:
                    game_steps = pickle.load(f)
                    self.all_steps.extend(game_steps)
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")
        print(f"Loaded {len(self.all_steps)} total game steps for training.")

    def __len__(self):
        return len(self.all_steps)

    @staticmethod
    def _transform_2d_array(arr_2d, transform_idx, board_size):
        """
        Applies a 2D transformation (rotation and/or flip) to a (board_size, board_size) array.
        There are 8 symmetries for a square board.
        Ensures the returned array is a contiguous copy to avoid negative stride issues.
        """
        transformed_arr = arr_2d

        # Identity
        if transform_idx == 0:
            transformed_arr = arr_2d
        # Rotations
        elif transform_idx == 1:  # 90 deg clockwise
            transformed_arr = np.rot90(arr_2d, k=-1)
        elif transform_idx == 2:  # 180 deg
            transformed_arr = np.rot90(arr_2d, k=-2)
        elif transform_idx == 3:  # 270 deg clockwise
            transformed_arr = np.rot90(arr_2d, k=-3)
        # Flips (combined with rotations for other symmetries)
        elif transform_idx == 4:  # Horizontal flip (across vertical midline)
            transformed_arr = np.fliplr(arr_2d)
        elif transform_idx == 5:  # Horizontal flip then 90 deg clockwise rot (equiv. anti-diagonal flip)
            transformed_arr = np.rot90(np.fliplr(arr_2d), k=-1)
        elif transform_idx == 6:  # Horizontal flip then 180 deg rot (equiv. vertical flip)
            transformed_arr = np.rot90(np.fliplr(arr_2d), k=-2)
        elif transform_idx == 7:  # Horizontal flip then 270 deg clockwise rot (equiv. main-diagonal flip)
            transformed_arr = np.rot90(np.fliplr(arr_2d), k=-3)
        else:
            raise ValueError(f"Invalid transform_idx: {transform_idx}")

        # IMPORTANT FIX: Ensure the returned array is a contiguous copy.
        # This resolves the "negative stride" ValueError when converting to PyTorch tensor.
        return transformed_arr.copy()

    @staticmethod
    def _transform_coords(r, c, transform_idx, board_size):
        """
        Applies the same 2D transformation to a (r, c) coordinate.
        Returns the new (r', c') coordinate.
        """
        N = board_size
        # Identity
        if transform_idx == 0:
            return r, c
        # Rotations
        elif transform_idx == 1:  # 90 deg clockwise (x,y) -> (y, N-1-x)
            return c, N - 1 - r
        elif transform_idx == 2:  # 180 deg (x,y) -> (N-1-x, N-1-y)
            return N - 1 - r, N - 1 - c
        elif transform_idx == 3:  # 270 deg clockwise (x,y) -> (N-1-y, x)
            return N - 1 - c, r
        # Flips
        elif transform_idx == 4:  # Horizontal flip (x,y) -> (x, N-1-y)
            return r, N - 1 - c
        elif transform_idx == 5:  # Horizontal flip then 90 deg clockwise rot (x,y) -> (N-1-x, y) then (y, x)
            # Apply fliplr: (r, N-1-c)
            # Then apply rot90_cw to (r, N-1-c): (N-1-c, N-1-r)
            return N - 1 - c, N - 1 - r  # This is main diagonal flip (swap r,c and reflect across N-1-r, N-1-c)
        elif transform_idx == 6:  # Horizontal flip then 180 deg rot (x,y) -> (N-1-x, y)
            # Apply fliplr: (r, N-1-c)
            # Then apply rot180 to (r, N-1-c): (N-1-r, N-(N-1-c)-1) = (N-1-r, c)
            return N - 1 - r, c  # This is vertical flip
        elif transform_idx == 7:  # Horizontal flip then 270 deg clockwise rot (x,y) -> (y, N-1-x)
            # Apply fliplr: (r, N-1-c)
            # Then apply rot270_cw to (r, N-1-c): (N-(N-1-c)-1, r) = (c, r)
            return c, r  # This is anti-diagonal flip (swap r,c and reflect only r)
        else:
            raise ValueError(f"Invalid transform_idx: {transform_idx}")

    def __getitem__(self, idx):
        step = self.all_steps[idx]

        # Choose a random transformation (0-7 for 8 symmetries)
        # In a typical AlphaZero/KataGo setup, during training, a random transform is applied to each sample.
        transform_idx = random.randint(0, 7)

        # --- Apply transformation to input_features ---
        # input_features: (BOARD_SIZE, BOARD_SIZE, NUM_INPUT_CHANNELS) numpy array
        # Note: np.zeros_like creates a new array, ensuring it's contiguous.
        # But we still need to ensure the source of the data for each channel is contiguous too.
        transformed_input_features = np.zeros_like(step['input_features'],
                                                   dtype=step['input_features'].dtype)  # Preserve original dtype
        for channel in range(step['input_features'].shape[2]):
            # Call _transform_2d_array, which now includes .copy()
            transformed_input_features[:, :, channel] = \
                self._transform_2d_array(step['input_features'][:, :, channel], transform_idx, BOARD_SIZE)

        # --- Apply transformation to mcts_policy target ---
        # mcts_policy: (BOARD_SIZE, BOARD_SIZE) numpy array (flattened when loaded)
        original_policy_2d = step['mcts_policy'].reshape(BOARD_SIZE, BOARD_SIZE)
        transformed_policy_2d = np.zeros_like(original_policy_2d, dtype=np.float32)  # Policy should be float32

        # Iterate through original coordinates, transform, and place probabilities
        for r_orig in range(BOARD_SIZE):
            for c_orig in range(BOARD_SIZE):
                r_trans, c_trans = self._transform_coords(r_orig, c_orig, transform_idx, BOARD_SIZE)
                transformed_policy_2d[r_trans, c_trans] = original_policy_2d[r_orig, c_orig]

        # Flatten the transformed policy for the policy head target
        # transformed_policy_2d is created with np.zeros_like and filled, so it should be contiguous.
        transformed_mcts_policy_target = torch.from_numpy(transformed_policy_2d).flatten().float()

        # --- Apply transformation to ownership_target ---
        # ownership_target: (BOARD_SIZE, BOARD_SIZE) numpy array
        # Call _transform_2d_array, which now includes .copy()
        transformed_ownership_target = self._transform_2d_array(step['ownership_target'], transform_idx, BOARD_SIZE)
        transformed_ownership_target_tensor = torch.from_numpy(transformed_ownership_target).float()

        # --- Value target remains unchanged (scalar) ---
        value_target = torch.tensor(step['game_outcome_value']).float().unsqueeze(0)

        # Convert transformed input features to torch tensor and adjust dimensions
        # transformed_input_features is created with np.zeros_like and filled, so it should be contiguous.
        transformed_input_tensor = torch.from_numpy(transformed_input_features).permute(2, 0, 1).float()

        # Return the transformed tensors
        return transformed_input_tensor, transformed_mcts_policy_target, value_target, transformed_ownership_target_tensor


class Trainer:
    """
    Handles the training loop for the PomPomNN model.
    """

    def __init__(self, model, data_loader, optimizer, device):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device

        # Loss functions
        # Policy loss: KLDivLoss is suitable for probability distributions
        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')
        # Value loss: MSELoss for regression (now also handles score loss)
        self.value_criterion = nn.MSELoss()
        # Ownership loss: MSELoss for regression as targets are -1 to 1
        self.ownership_criterion = nn.MSELoss()
        # Score loss is removed

        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    def train_epoch(self):
        self.model.train()  # Set model to training mode
        total_policy_loss = 0
        total_value_loss = 0  # Now accumulates unified value/score loss
        total_ownership_loss = 0
        # total_score_loss is removed
        total_loss = 0
        num_batches = 0

        # Data loader now yields inputs, policy_targets, value_targets, ownership_targets
        for batch_idx, (inputs, policy_targets, value_targets, ownership_targets) in enumerate(self.data_loader):
            inputs, policy_targets, value_targets, ownership_targets = \
                inputs.to(self.device), policy_targets.to(self.device), value_targets.to(self.device), \
                    ownership_targets.to(self.device)

            self.optimizer.zero_grad()  # Zero the gradients

            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                # Forward pass - Model now returns three outputs: policy, value, ownership
                predicted_policy_probs, predicted_value, predicted_ownership = self.model(inputs)

                # Calculate Policy Loss
                policy_loss = self.policy_criterion(torch.log(predicted_policy_probs + 1e-9),
                                                    policy_targets)  # Add epsilon for numerical stability

                # Calculate Value Loss (now unified for value/score)
                value_loss = self.value_criterion(predicted_value, value_targets)

                # Calculate Ownership Loss
                ownership_loss = self.ownership_criterion(predicted_ownership, ownership_targets)

                # Combine losses
                loss = policy_loss + \
                       VALUE_LOSS_WEIGHT * value_loss + \
                       OWNERSHIP_LOSS_WEIGHT * ownership_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_ownership_loss += ownership_loss.item()
            total_loss += loss.item()
            num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_ownership_loss = total_ownership_loss / num_batches
        # avg_score_loss is removed
        avg_total_loss = total_loss / num_batches
        return avg_policy_loss, avg_value_loss, avg_ownership_loss, avg_total_loss

    def run_training(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            # Only three loss values are returned now
            policy_loss, value_loss, ownership_loss, total_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Policy Loss: {policy_loss:.4f}, "
                  f"Value Loss: {value_loss:.4f}, "
                  f"Ownership Loss: {ownership_loss:.4f}, "
                  f"Total Loss: {total_loss:.4f}")
        print("Training finished.")


def save_model(model, models_dir, iteration_num):
    """Saves the model's state dictionary to a versioned file."""
    os.makedirs(models_dir, exist_ok=True)
    file_name = os.path.join(models_dir, f"model_iter_{iteration_num:05d}.pth")
    try:
        torch.save(model.state_dict(), file_name)
        print(f"Model saved to {file_name}")
    except Exception as e:
        print(f"Error saving model to {file_name}: {e}")


def load_latest_model(model, models_dir, device):
    """Loads the latest model from the specified directory."""
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
            iter_str = f_name.replace("model_iter_", "").replace(".pth", "")  # Corrected from .pkl to .pth
            iteration = int(iter_str)
            if iteration > latest_iteration:
                latest_iteration = iteration
                latest_model_path = os.path.join(models_dir, f_name)
        except ValueError:
            continue

    if latest_model_path:
        try:
            # 修改: 加载模型状态字典时，需要设置为 strict=False
            # 因为模型结构增加了新的头，如果直接 strict=True 会报错
            model.load_state_dict(torch.load(latest_model_path, map_location=device), strict=False)
            print(f"Model loaded from {latest_model_path} (latest iteration: {latest_iteration})")
        except Exception as e:
            print(f"Error loading model from {latest_model_path}: {e}")
            print("Using randomly initialized model instead.")
    else:
        print(f"No valid model files found in '{models_dir}'. Using randomly initialized model.")

    return model


if __name__ == "__main__":
    print("--- 泡姆棋神经网络训练器 ---")

    # 从命令行参数获取迭代编号和模型目录
    # 期望的参数格式: python train_model.py <iteration_num> [models_dir]
    iteration_num = 0
    current_models_dir = MODELS_DIR

    if len(sys.argv) > 1:
        try:
            iteration_num = int(sys.argv[1])
            if len(sys.argv) > 2:
                current_models_dir = sys.argv[2]
        except ValueError:
            print("Usage: python train_model.py <iteration_num> [models_dir]")
            sys.exit(1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 添加 cuDNN 自动调优
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("Enabled torch.backends.cudnn.benchmark for potential performance gains on GPU.")

    # 1. Load Data
    dataset = SelfPlayDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No self-play data found. Please run self_play_worker.py first to generate data.")
        sys.exit(1)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0)

    # 2. Initialize Model
    # Note: num_attention_heads defaults to 8 in PomPomNN, so no explicit passing is needed unless you want to change it.
    model = PomPomNN(num_res_blocks=6, num_filters=96)
    model.to(device)

    # Load pre-existing model if available (always load the latest)
    model = load_latest_model(model, current_models_dir, device)

    # 3. Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 4. Initialize Trainer and Run Training
    trainer = Trainer(model, data_loader, optimizer, device)
    trainer.run_training(NUM_EPOCHS)

    # 5. Save Trained Model (使用当前迭代编号保存)
    save_model(model, current_models_dir, iteration_num)
