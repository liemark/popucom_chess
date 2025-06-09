import tkinter as tk
from tkinter import messagebox, Scale
import os
import torch
import numpy as np
import random
import threading  # For running AI move in a separate thread to prevent GUI freeze
import time  # For AI vs AI sleep

# Import game logic, NN model, and PUCT search
try:
    from popucom_chess import (
        BOARD_SIZE,
        BLACK_PLAYER,
        WHITE_PLAYER,
        GAME_RESULT_WIN,
        GAME_RESULT_LOSS,
        GAME_RESULT_DRAW,
        GAME_RESULT_NONE,
        initialize_board,
        do_move,
        check_game_over,
        check_valid_move_for_next_player
    )
    from popucom_nn_model import PomPomNN
    from popucom_puct import MCTSSearcher, MCTSNode
    from train_model import NN_NUM_FILTERS, NN_NUM_RES_BLOCKS
except ImportError as e:
    messagebox.showerror("导入错误", f"无法导入所需模块。请确保所有 .py 文件在同一目录下。\n错误: {e}")
    exit()

# --- Configuration Parameters ---
MODELS_DIR = "models"  # Directory where trained models are saved

# AI's neural network model parameters (should match what was used for training)
#NN_NUM_RES_BLOCKS = 6
#NN_NUM_FILTERS = 96

# Game board and piece colors
BOARD_BACKGROUND_COLOR = "#D2B48C"  # Tan color
GRID_LINE_COLOR = "#8B4513"  # SaddleBrown
UNPAINTED_FLOOR_COLOR = "#FFF8DC"  # Cornsilk
BLACK_PAINTED_FLOOR_COLOR = "#FFDAB9"  # PeachPuff (light red)
WHITE_PAINTED_FLOOR_COLOR = "#90EE90"  # LightGreen (light green)

BLACK_PIECE_COLOR = "#FF0000"  # Red
WHITE_PIECE_COLOR = "#008000"  # Green

# Game parameters
INITIAL_REMAINING_MOVES = 25


class PomPomGame:
    def __init__(self, master):
        self.master = master
        master.title("泡姆棋")  # Changed title to be more general for both modes

        self.board_size = BOARD_SIZE
        self.cell_size = 100  # Initial cell size, increased from 50 to make the board larger
        self.canvas_width = self.board_size * self.cell_size
        self.canvas_height = self.board_size * self.cell_size

        self.game_running = False
        # New: Variable to store selected game mode (human_vs_ai, human_vs_human, ai_vs_ai)
        self.game_mode = tk.StringVar(value="human_vs_ai")
        # New: Variable for Human vs AI mode to select human's color
        self.human_player_choice = tk.StringVar(value="human_black")

        # AI related attributes (initialized but only used in human_vs_ai or ai_vs_ai mode)
        self.human_player = None  # Which player is human in HvAI mode
        self.ai_player = None  # Which player is AI in HvAI mode
        self.ai_player_1 = BLACK_PLAYER  # For AI vs AI mode, AI1 is black
        self.ai_player_2 = WHITE_PLAYER  # For AI vs AI mode, AI2 is white

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure PomPomNN is initialized with correct parameters consistent with training
        self.ai_model = PomPomNN(num_res_blocks=NN_NUM_RES_BLOCKS, num_filters=NN_NUM_FILTERS).to(self.device)
        self._load_ai_model()  # Load latest model or random if none exist
        self.ai_model.eval()
        # Initialize AI searcher (c_puct fixed for game)
        # For game play, use_win_loss_target is always False, as game is score-based
        self.ai_searcher = MCTSSearcher(model=self.ai_model, c_puct=1.0, use_win_loss_target=False)

        self.current_player = BLACK_PLAYER  # Who's turn it is now, always starts as Black
        self._last_move_coords = None  # New: Stores (r, c) of the last move for highlighting

        # Game history for undo feature
        self.game_history = []  # Stores (board, remaining_moves, current_player, _last_move_coords)

        # Analysis feature related
        self.analysis_scores = {}  # Stores scores for display: {(r, c): score}
        self.analysis_in_progress = False  # Flag to prevent multiple analysis threads
        self.highlighted_ai_analysis_move = None  # Stores AI's chosen move for analysis display

        # Initialize board and remaining moves here so they always exist
        self.board = initialize_board()
        self.remaining_moves = {BLACK_PLAYER: INITIAL_REMAINING_MOVES, WHITE_PLAYER: INITIAL_REMAINING_MOVES}

        # --- GUI Elements ---
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(side=tk.TOP, pady=10)

        # Game Mode Selection Group
        self.mode_label = tk.Label(self.control_frame, text="选择模式:", font=("Arial", 10))
        self.mode_label.pack(side=tk.LEFT, padx=5)
        self.human_ai_radio = tk.Radiobutton(self.control_frame, text="人机对战", variable=self.game_mode,
                                             value="human_vs_ai", font=("Arial", 10), command=self._on_mode_change)
        self.human_ai_radio.pack(side=tk.LEFT, padx=5)
        self.human_human_radio = tk.Radiobutton(self.control_frame, text="人人对战", variable=self.game_mode,
                                                value="human_vs_human", font=("Arial", 10),
                                                command=self._on_mode_change)
        self.human_human_radio.pack(side=tk.LEFT, padx=5)
        self.ai_ai_radio = tk.Radiobutton(self.control_frame, text="机机对战", variable=self.game_mode,
                                          value="ai_vs_ai", font=("Arial", 10), command=self._on_mode_change)
        self.ai_ai_radio.pack(side=tk.LEFT, padx=5)

        # Player Role Selection Group (visible only in Human vs AI mode)
        self.role_frame = tk.Frame(self.control_frame)  # Frame to group role radio buttons
        self.role_frame.pack(side=tk.LEFT, padx=10)
        self.human_black_radio = tk.Radiobutton(self.role_frame, text="我执红 (先手)",
                                                variable=self.human_player_choice,
                                                value="human_black", font=("Arial", 10), command=self._on_mode_change)
        self.human_black_radio.pack(side=tk.LEFT, padx=5)
        self.ai_black_radio = tk.Radiobutton(self.role_frame, text="AI执红 (先手)", variable=self.human_player_choice,
                                             value="ai_black", font=("Arial", 10), command=self._on_mode_change)
        self.ai_black_radio.pack(side=tk.LEFT, padx=5)

        # Game control buttons
        self.new_game_button = tk.Button(self.control_frame, text="新游戏", command=self._reset_and_start_new_game)
        self.new_game_button.pack(side=tk.RIGHT, padx=5)
        self.undo_button = tk.Button(self.control_frame, text="悔棋", command=self._undo_move)
        self.undo_button.pack(side=tk.RIGHT, padx=5)
        self.analyze_button = tk.Button(self.control_frame, text="分析", command=self._analyze_board)
        self.analyze_button.pack(side=tk.RIGHT, padx=5)

        # Status and Moves labels (moved outside control frame for better layout)
        self.status_label = tk.Label(master, text="初始化中...", font=("Arial", 14, "bold"))  # Added bold font
        self.status_label.pack(side=tk.TOP, pady=5)  # Placed below control_frame

        self.moves_label = tk.Label(master, text="", font=("Arial", 12))
        self.moves_label.pack(side=tk.TOP, pady=5)  # Placed below status_label

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg=BOARD_BACKGROUND_COLOR)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._handle_click)

        # AI Depth Slider and Label (visibility toggled by game mode)
        self.ai_depth_label = tk.Label(master, text="AI 搜索深度 (模拟次数):", font=("Arial", 10))
        self.ai_depth_label.pack(pady=(10, 0))
        self.ai_depth_slider = Scale(master, from_=50, to=2000, orient=tk.HORIZONTAL, resolution=50, length=300)
        self.ai_depth_slider.set(800)  # Default AI search depth
        self.ai_depth_slider.pack(pady=(0, 10))

        # New: Temperature Slider and Label
        self.temperature_label = tk.Label(master, text="AI 落子温度 (选择次优解):", font=("Arial", 10))
        self.temperature_label.pack(pady=(10, 0))
        self.temperature_slider = Scale(master, from_=0.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.1, length=300)
        self.temperature_slider.set(0.0)  # Default temperature
        self.temperature_slider.pack(pady=(0, 10))

        # Initial call to set visibility and roles based on default mode
        self._on_mode_change()
        # Initial game start without resetting board, allows _on_mode_change to determine
        self._start_game_flow()

    def _on_mode_change(self):
        """
        Adjusts GUI element visibility and sets player roles based on the selected game mode.
        """
        selected_mode = self.game_mode.get()

        # Hide role-specific elements first
        self.role_frame.pack_forget()

        # Ensure AI settings sliders are always visible (packed)
        self.ai_depth_label.pack(pady=(10, 0))
        self.ai_depth_slider.pack(pady=(0, 10))
        self.temperature_label.pack(pady=(10, 0))
        self.temperature_slider.pack(pady=(0, 10))

        if selected_mode == "human_vs_ai":
            # Show role selection for Human vs AI
            self.role_frame.pack(side=tk.LEFT, padx=10)

            # Set human/AI player based on human_player_choice variable
            choice = self.human_player_choice.get()
            if choice == "human_black":
                self.human_player = BLACK_PLAYER
                self.ai_player = WHITE_PLAYER
            else:  # ai_black
                self.human_player = WHITE_PLAYER
                self.ai_player = BLACK_PLAYER
            self.ai_player_1 = None  # Not applicable for AI1/AI2 specific roles
            self.ai_player_2 = None

        elif selected_mode == "human_vs_human":
            # In Human vs Human mode, no specific human/AI roles needed
            self.human_player = None
            self.ai_player = None
            self.ai_player_1 = None
            self.ai_player_2 = None

        elif selected_mode == "ai_vs_ai":
            # In AI vs AI mode, no specific human/AI roles needed
            self.human_player = None
            self.ai_player = None
            self.ai_player_1 = BLACK_PLAYER  # AI 1 plays black
            self.ai_player_2 = WHITE_PLAYER  # AI 2 plays white

        # Start game flow without resetting board, unless it's a new game button press
        self._start_game_flow()

    def _load_ai_model(self):
        """
        Loads the latest trained AI model from the 'models' directory.
        If no model is found, the AI will use a randomly initialized model.
        """
        model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("model_iter_") and f.endswith(".pth")]

        latest_iteration = -1
        latest_model_path = None
        for f_name in model_files:
            try:
                iter_str = f_name.replace("model_iter_", "").replace(".pth", "")
                iteration = int(iter_str)
                if iteration > latest_iteration:
                    latest_iteration = iteration
                    latest_model_path = os.path.join(MODELS_DIR, f_name)
            except ValueError:
                continue

        if latest_model_path:
            try:
                self.ai_model.load_state_dict(torch.load(latest_model_path, map_location=self.device), strict=False)
                print(f"AI 模型已从 {latest_model_path} 加载。")
            except Exception as e:
                messagebox.showwarning("模型加载警告",
                                       f"无法从 {latest_model_path} 加载 AI 模型。错误: {e}\nAI 将使用随机权重。")
        else:
            messagebox.showinfo("模型加载信息", f"在 '{MODELS_DIR}' 目录中未找到 AI 模型。AI 将使用随机权重。")
        self.ai_model.eval()  # Ensure model is in evaluation mode

    def _reset_and_start_new_game(self):
        """Resets the board and starts a new game regardless of existing state."""
        self.board = initialize_board()
        self.remaining_moves = {BLACK_PLAYER: INITIAL_REMAINING_MOVES, WHITE_PLAYER: INITIAL_REMAINING_MOVES}
        self.current_player = BLACK_PLAYER
        self._last_move_coords = None  # Clear last move highlight
        self.game_history = []  # Clear game history for new game
        self.analysis_scores = {}  # Clear analysis scores
        self.highlighted_ai_analysis_move = None  # Clear AI's suggested move
        self.game_running = True
        self._toggle_interaction_buttons(tk.NORMAL)  # Re-enable buttons
        self._start_game_flow()  # Begin the game flow with reset board

    def _start_game_flow(self):
        """Starts or continues the game flow based on the current mode and board state."""
        self.draw_board()
        self._update_status()
        self.master.bind("<Configure>", self._on_resize)

        game_results_dict = check_game_over(self.board, self.remaining_moves, self.current_player)
        if game_results_dict[BLACK_PLAYER] != GAME_RESULT_NONE or game_results_dict[WHITE_PLAYER] != GAME_RESULT_NONE:
            self.game_running = False
            self._update_status()
            self._toggle_interaction_buttons(tk.NORMAL)  # Re-enable buttons if game over
            return

        if self.game_running:
            if self.game_mode.get() == "human_vs_ai" and self.current_player == self.ai_player:
                self.master.after(500, self._ai_turn)
            elif self.game_mode.get() == "ai_vs_ai":
                self.master.after(500, self._ai_turn)

    def _on_resize(self, event):
        """Handles canvas resizing."""
        new_canvas_width = self.canvas.winfo_width()
        new_canvas_height = self.canvas.winfo_height()

        if new_canvas_width <= 1 or new_canvas_height <= 1:
            return

        min_dim = min(new_canvas_width, new_canvas_height)
        self.cell_size = min_dim // self.board_size

        self.canvas.config(width=self.board_size * self.cell_size, height=self.board_size * self.cell_size)
        self.draw_board()

    def draw_board(self):
        """Draws the current game board state on the canvas."""
        self.canvas.delete("all")

        # Draw painted floors
        for r in range(self.board_size):
            for c in range(self.board_size):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                floor_color = UNPAINTED_FLOOR_COLOR
                if self.board[r][c][2] == 1:  # is_black_painted
                    floor_color = BLACK_PAINTED_FLOOR_COLOR
                elif self.board[r][c][3] == 1:  # is_white_painted
                    floor_color = WHITE_PAINTED_FLOOR_COLOR

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=floor_color, outline="")

        # Draw grid lines
        for i in range(self.board_size + 1):
            self.canvas.create_line(0, i * self.cell_size, self.canvas.winfo_width(), i * self.cell_size,
                                    fill=GRID_LINE_COLOR)
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.canvas.winfo_height(),
                                    fill=GRID_LINE_COLOR)

        # Draw pieces and last move indicator
        for r in range(self.board_size):
            for c in range(self.board_size):
                center_x = c * self.cell_size + self.cell_size // 2
                center_y = r * self.cell_size + self.cell_size // 2
                radius = self.cell_size // 2 - 5

                piece_color = None
                if self.board[r][c][0] == 1:  # has_black_piece
                    piece_color = BLACK_PIECE_COLOR
                elif self.board[r][c][1] == 1:  # has_white_piece
                    piece_color = WHITE_PIECE_COLOR

                if piece_color:
                    self.canvas.create_oval(center_x - radius, center_y - radius,
                                            center_x + radius, center_y + radius,
                                            fill=piece_color, outline=piece_color)

                    if self._last_move_coords == (r, c):
                        dot_radius = self.cell_size // 10
                        self.canvas.create_oval(center_x - dot_radius, center_y - dot_radius,
                                                center_x + dot_radius, center_y + dot_radius,
                                                fill="black", outline="black")

        # Draw analysis scores overlay
        for (r, c), score in self.analysis_scores.items():
            center_x = c * self.cell_size + self.cell_size // 2
            center_y = r * self.cell_size + self.cell_size // 2

            score_text = f"{score * 81:.4f}"  # Display more precision for analysis scores

            # Adjust font size based on cell size
            font_size = max(8, self.cell_size // 8)  # Made font size smaller
            text_color = "blue" if score >= 0 else "red"  # Blue for positive, red for negative

            # Draw a semi-transparent background for the text for better readability
            text_bbox = self.canvas.create_text(center_x, center_y + self.cell_size // 4,  # Measure text size
                                                text=score_text, font=("Arial", font_size, "bold"),
                                                fill=text_color, tags="analysis_text")

            # Get the coordinates of the text bounding box
            x1, y1, x2, y2 = self.canvas.bbox(text_bbox)
            # Create a rectangle behind the text for background
            self.canvas.create_rectangle(x1 - 2, y1 - 2, x2 + 2, y2 + 2,
                                         fill="white", stipple="gray50", tags="analysis_text",
                                         outline="")  # stipple makes it semi-transparent

            # Bring the text to the front
            self.canvas.tag_raise(text_bbox)

        # New: Draw highlight for AI's suggested move from analysis
        if self.highlighted_ai_analysis_move:
            r, c = self.highlighted_ai_analysis_move
            x1, y1 = c * self.cell_size, r * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size
            # Draw a prominent green border around the cell
            self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                                         outline="green", width=4, tags="ai_suggested_highlight")

    def _update_status(self):
        """Updates the status and moves labels based on current game state and mode."""
        if not self.game_running:
            self.status_label.config(text="游戏结束！", fg="black")
            self.moves_label.config(text="")
            return

        current_player_color_name = "红方" if self.current_player == BLACK_PLAYER else "绿方"
        turn_text = ""
        turn_color = "black"

        black_painted_count = 0
        white_painted_count = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c][2] == 1:
                    black_painted_count += 1
                elif self.board[r][c][3] == 1:
                    white_painted_count += 1

        if self.game_mode.get() == "human_vs_ai":
            if self.current_player == self.human_player:
                turn_text = f"当前回合: {current_player_color_name} (您)"
                turn_color = BLACK_PIECE_COLOR if self.current_player == BLACK_PLAYER else WHITE_PIECE_COLOR
            else:
                turn_text = f"当前回合: {current_player_color_name} (AI)"
                turn_color = BLACK_PIECE_COLOR if self.current_player == BLACK_PLAYER else WHITE_PIECE_COLOR
        elif self.game_mode.get() == "human_vs_human":
            turn_text = f"当前回合: {current_player_color_name} (玩家)"
            turn_color = BLACK_PIECE_COLOR if self.current_player == BLACK_PLAYER else WHITE_PIECE_COLOR
        elif self.game_mode.get() == "ai_vs_ai":
            turn_text = f"当前回合: {current_player_color_name} (AI)"
            turn_color = BLACK_PIECE_COLOR if self.current_player == BLACK_PLAYER else WHITE_PIECE_COLOR

        self.status_label.config(text=turn_text, fg=turn_color)
        self.moves_label.config(
            text=f"红方剩余步数: {self.remaining_moves[BLACK_PLAYER]} | "
                 f"绿方剩余步数: {self.remaining_moves[WHITE_PLAYER]}\n"
                 f"红方涂色数量: {black_painted_count} | "
                 f"绿方涂色数量: {white_painted_count}")

    def _toggle_interaction_buttons(self, state):
        """Helper to enable/disable interaction buttons and canvas clicks."""
        self.new_game_button.config(state=state)
        self.undo_button.config(state=state)
        self.analyze_button.config(state=state)
        # Only enable canvas clicks if it's a human's turn in appropriate modes
        if self.game_mode.get() == "human_vs_human" or \
                (self.game_mode.get() == "human_vs_ai" and self.current_player == self.human_player):
            self.canvas.config(state=state)
        else:  # AI turn or AI vs AI, keep canvas disabled for clicks
            self.canvas.config(state=tk.DISABLED)

    def _handle_click(self, event):
        """Handles human player's click on the board."""
        if not self.game_running or self.canvas.cget("state") == tk.DISABLED:
            return

        if self.game_mode.get() == "ai_vs_ai":
            messagebox.showinfo("提示", "当前是机机对战模式，请观看。")
            return

        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            messagebox.showwarning("非法落子", "请点击棋盘内的有效位置。")
            return

        player_making_move = None
        if self.game_mode.get() == "human_vs_ai":
            if self.current_player != self.human_player:
                messagebox.showinfo("等待", "当前是 AI 回合，请稍候。")
                return
            player_making_move = self.human_player
        elif self.game_mode.get() == "human_vs_human":
            player_making_move = self.current_player

        if not check_valid_move_for_next_player(self.board, row, col, player_making_move):
            messagebox.showwarning("非法落子", "该位置已有棋子或地板颜色不符合落子规则。")
            return

        game_ended = self._make_move(row, col, player_making_move)

        if self.game_running and not game_ended and \
                self.game_mode.get() == "human_vs_ai" and self.current_player == self.ai_player:
            self.master.after(500, self._ai_turn)

    def _make_move(self, r, c, player):
        """Executes a move and updates game state and GUI.
        Returns True if game ended, False otherwise."""
        # Clear any existing analysis display
        self.analysis_scores = {}
        self.highlighted_ai_analysis_move = None  # Clear AI's suggested move

        # Save current state for undo BEFORE making the move
        self.game_history.append({
            'board': self.board.copy(),
            'remaining_moves': self.remaining_moves.copy(),
            'current_player': self.current_player,
            '_last_move_coords': self._last_move_coords
        })

        self.board, self.remaining_moves, game_result_for_player = do_move(
            self.board, r, c, self.remaining_moves, player
        )
        self._last_move_coords = (r, c)
        self.draw_board()
        self._update_status()

        game_results_dict = check_game_over(self.board, self.remaining_moves, player)

        if game_results_dict[BLACK_PLAYER] != GAME_RESULT_NONE or game_results_dict[WHITE_PLAYER] != GAME_RESULT_NONE:
            self.game_running = False
            self._update_status()
            messagebox.showinfo("游戏结束",
                                "红方胜利！" if game_results_dict[BLACK_PLAYER] == GAME_RESULT_WIN else
                                "绿方胜利！" if game_results_dict[WHITE_PLAYER] == GAME_RESULT_WIN else "平局！")
            self._toggle_interaction_buttons(tk.NORMAL)  # Re-enable buttons at game end
            return True

        self.current_player = WHITE_PLAYER if player == BLACK_PLAYER else BLACK_PLAYER
        self._update_status()

        if self.game_running and self.game_mode.get() == "ai_vs_ai":
            self.master.after(500, self._ai_turn)

        return False

    def _undo_move(self):
        """Reverts the game to the previous state."""
        if not self.game_history:
            messagebox.showinfo("悔棋", "无法再悔棋了。")
            return

        self._toggle_interaction_buttons(tk.DISABLED)  # Disable buttons while processing undo
        self.analysis_scores = {}  # Clear any analysis scores
        self.highlighted_ai_analysis_move = None  # Clear AI's suggested move

        previous_state = self.game_history.pop()
        self.board = previous_state['board']
        self.remaining_moves = previous_state['remaining_moves']
        self.current_player = previous_state['current_player']
        self._last_move_coords = previous_state['_last_move_coords']

        self.game_running = True  # Re-enable game flow after undo
        self.draw_board()
        self._update_status()

        # Re-enable buttons, considering if it's now AI's turn
        if self.game_mode.get() == "human_vs_ai" and self.current_player == self.ai_player:
            self.master.after(500, self._ai_turn)  # Trigger AI turn
        elif self.game_mode.get() == "ai_vs_ai":
            self.master.after(500, self._ai_turn)  # Trigger AI turn
        else:  # Human vs Human, or Human vs AI where it's human's turn after undo
            self._toggle_interaction_buttons(tk.NORMAL)  # Re-enable for human play

    def _get_legal_moves_for_player(self, board, player):
        """Helper to get all legal moves for a given player on a board."""
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if check_valid_move_for_next_player(board, r, c, player):
                    legal_moves.append((r, c))
        return legal_moves

    def _ai_turn(self):
        """Handles the AI's move using PUCT search."""

        current_mode = self.game_mode.get()

        if not self.game_running:
            return

        player_to_move = self.current_player

        self.status_label.config(
            text=f"当前回合: {'红方 (AI)' if player_to_move == BLACK_PLAYER else '绿方 (AI)'} 思考中...",
            fg=BLACK_PIECE_COLOR if player_to_move == BLACK_PLAYER else WHITE_PIECE_COLOR
        )
        self.master.update_idletasks()

        self._toggle_interaction_buttons(tk.DISABLED)  # Disable buttons during AI thinking
        self.analysis_scores = {}  # Clear analysis scores if AI starts thinking
        self.highlighted_ai_analysis_move = None  # Clear AI's suggested move

        ai_search_depth = self.ai_depth_slider.get()
        ai_temperature = self.temperature_slider.get()  # Get temperature from slider

        def _ai_logic_worker_thread():
            if current_mode == "ai_vs_ai":
                time.sleep(0.3)

            root_node = MCTSNode(self.board, self.remaining_moves, player_to_move)
            self.ai_searcher.run_search(root_node, ai_search_depth)

            # Pass the temperature to get_policy_distribution
            mcts_policy_distribution = self.ai_searcher.get_policy_distribution(root_node, temperature=ai_temperature)

            if not isinstance(mcts_policy_distribution, np.ndarray):
                print(
                    f"CRITICAL ERROR: mcts_policy_distribution is not a numpy array AFTER FIX. Type: {type(mcts_policy_distribution)}. Please investigate popucom_puct.py.")
                mcts_policy_distribution = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

            legal_moves_for_ai = self._get_legal_moves_for_player(self.board, player_to_move)

            best_move = None
            if not legal_moves_for_ai:
                best_move = None
            else:
                # Select move based on the temperature-adjusted policy distribution
                flat_policy = mcts_policy_distribution.flatten()

                # Filter policy to only legal moves and normalize
                legal_move_flat_indices = [r * BOARD_SIZE + c for r, c in legal_moves_for_ai]
                legal_probs = flat_policy[legal_move_flat_indices]

                if np.sum(legal_probs) == 0:
                    # Fallback to uniform random if all legal_probs are zero (e.g. initial untrained NN)
                    chosen_move_flat_idx = random.choice(legal_move_flat_indices)
                else:
                    legal_probs = legal_probs / np.sum(legal_probs)  # Normalize
                    chosen_move_flat_idx = np.random.choice(legal_move_flat_indices, p=legal_probs)

                best_move = (chosen_move_flat_idx // BOARD_SIZE, chosen_move_flat_idx % BOARD_SIZE)

            # --- Debugging Output for AI's actual chosen move ---
            if best_move:
                chosen_move_q_value = -root_node.children[best_move].Q if best_move in root_node.children else float(
                    'nan')
                chosen_move_policy_prob = mcts_policy_distribution[best_move[0], best_move[1]]
                print(f"\nAI ({player_to_move}) 实际选择的落子: {best_move}")
                print(f"  MCTS Q值 (当前玩家视角): {chosen_move_q_value}")  # Print raw float
                print(f"  策略概率 (Policy Probability): {chosen_move_policy_prob}")  # Print raw float
            else:
                print(f"\nAI ({player_to_move}) 没有合法的落子或无法选择落子。")
            # --- End Debugging Output ---

            self.master.after(0, _ai_move_callback, best_move)

        def _ai_move_callback(move):
            if self.game_running:
                if move is None:
                    winning_player_name = "红方" if player_to_move == WHITE_PLAYER else "绿方"
                    messagebox.showinfo("游戏结束",
                                        f"{'红方' if player_to_move == BLACK_PLAYER else '绿方'} (AI) 无合法落子。{winning_player_name} 胜利！")
                    self.game_running = False
                    self._update_status()
                    self._toggle_interaction_buttons(tk.NORMAL)  # Game over, re-enable to start new game
                else:
                    game_ended = self._make_move(move[0], move[1], player_to_move)
                    if not game_ended:
                        self._toggle_interaction_buttons(tk.NORMAL)  # Re-enable after AI move, if game still running
                    else:
                        self._toggle_interaction_buttons(tk.NORMAL)  # Re-enable for new game

        threading.Thread(target=_ai_logic_worker_thread).start()

    def _analyze_board(self):
        """Triggers MCTS analysis and displays scores on the board."""
        if self.analysis_in_progress:
            # If analysis is already running, or if it's already displayed, clear it.
            self.analysis_scores = {}
            self.highlighted_ai_analysis_move = None  # Clear AI's suggested move
            self.draw_board()
            self.analyze_button.config(state=tk.NORMAL, text="分析")
            self.analysis_in_progress = False
            return

        if not self.game_running:
            messagebox.showinfo("分析", "游戏未开始或已结束，无法进行分析。")
            return

        self.analysis_in_progress = True
        self.analyze_button.config(state=tk.DISABLED, text="分析中...")  # Disable button

        self._toggle_interaction_buttons(tk.DISABLED)  # Disable all interaction buttons during analysis

        # Run MCTS search for analysis in a separate thread
        threading.Thread(target=self._run_analysis_search_thread).start()

    def _run_analysis_search_thread(self):
        """Performs MCTS search for analysis in a background thread."""
        # Use the SAME AI search depth as the actual AI turn for consistent analysis
        analysis_simulations = self.ai_depth_slider.get()
        # Get AI temperature from slider for suggested move calculation (same as AI's actual move temp)
        ai_temperature_for_suggestion = self.temperature_slider.get()

        root_node = MCTSNode(self.board, self.remaining_moves, self.current_player)
        self.ai_searcher.run_search(root_node, analysis_simulations)

        scores_for_display = {}
        legal_moves = self._get_legal_moves_for_player(self.board, self.current_player)

        # Print all legal moves and their raw Q-values for debugging
        print(f"\n--- Analysis Raw Q-Values for {self.current_player} (Simulations: {analysis_simulations}) ---")
        for r, c in legal_moves:
            move = (r, c)
            if move in root_node.children:
                child_node = root_node.children[move]
                # Q-value from child perspective (opponent's Q for next state)
                raw_q_value_child_perspective = child_node.Q
                # Convert to current player's perspective for display
                q_value_current_player_perspective = -raw_q_value_child_perspective

                print(
                    f"  Move ({r}, {c}): Raw Q (child) = {raw_q_value_child_perspective:.6f}, Q (current player) = {q_value_current_player_perspective:.6f}")

                # Invert Q-value to reflect current player's perspective for display
                scores_for_display[move] = q_value_current_player_perspective
            else:
                # This should ideally not happen if root_node.is_expanded after run_search
                print(f"  Move ({r}, {c}): Not found in children (potential issue)")
        print("------------------------------------------------------------------")

        # Get policy distribution based on the current temperature for AI's suggested move
        mcts_policy_distribution_for_suggestion = self.ai_searcher.get_policy_distribution(root_node,
                                                                                           temperature=ai_temperature_for_suggestion)

        if not isinstance(mcts_policy_distribution_for_suggestion, np.ndarray):
            print(
                f"CRITICAL ERROR: mcts_policy_distribution_for_suggestion is not a numpy array during analysis. Type: {type(mcts_policy_distribution_for_suggestion)}. Falling back to zeros.")
            mcts_policy_distribution_for_suggestion = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        ai_suggested_move = None
        if legal_moves:
            # Determine AI's suggested move based on the policy distribution and temperature
            flat_policy = mcts_policy_distribution_for_suggestion.flatten()
            legal_move_flat_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
            legal_probs = flat_policy[legal_move_flat_indices]

            if np.sum(legal_probs) == 0:
                chosen_move_flat_idx = random.choice(legal_move_flat_indices)
            else:
                legal_probs = legal_probs / np.sum(legal_probs)  # Normalize
                chosen_move_flat_idx = np.random.choice(legal_move_flat_indices, p=legal_probs)

            ai_suggested_move = (chosen_move_flat_idx // BOARD_SIZE, chosen_move_flat_idx % BOARD_SIZE)

        # --- Debugging Output for Analysis Suggested Move ---
        if ai_suggested_move:
            suggested_move_q_value = scores_for_display.get(ai_suggested_move, float('nan'))
            suggested_move_policy_prob = mcts_policy_distribution_for_suggestion[
                ai_suggested_move[0], ai_suggested_move[1]]
            print(f"\n分析建议落子: {ai_suggested_move}")
            print(f"  MCTS Q值 (当前玩家视角): {suggested_move_q_value:.6f}")  # Print raw float
            print(f"  策略概率 (Policy Probability): {suggested_move_policy_prob:.6f}")  # Print raw float
        else:
            print("\n分析中没有发现建议的落子。")
        # --- End Debugging Output ---

        # Schedule the display update on the main Tkinter thread, passing both scores and the suggested move
        self.master.after(0, self._display_analysis_results, scores_for_display, ai_suggested_move)

    def _display_analysis_results(self, scores_for_display, ai_suggested_move):
        """Displays analysis scores and AI's suggested move on the board and re-enables buttons."""
        self.analysis_scores = scores_for_display
        self.highlighted_ai_analysis_move = ai_suggested_move  # Store suggested move
        self.draw_board()  # Redraw board to show analysis scores and suggested move
        self.analyze_button.config(state=tk.NORMAL, text="再次分析")  # Change text to clear analysis
        self.analysis_in_progress = False

        self._toggle_interaction_buttons(tk.NORMAL)  # Re-enable all interaction buttons after analysis


# --- Main Application Loop ---
if __name__ == "__main__":
    # Ensure models directory exists for AI model loading
    os.makedirs(MODELS_DIR, exist_ok=True)

    root = tk.Tk()
    game = PomPomGame(root)
    root.mainloop()
