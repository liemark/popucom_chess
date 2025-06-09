# run_pipeline.py

import subprocess
import time
import os
from self_play_worker import NUM_SELF_PLAY_GAMES, NUM_WORKERS
from train_model import NUM_EPOCHS

# --- 流水线配置参数 ---
NUM_SELF_PLAY_GAMES_PER_ITERATION = NUM_SELF_PLAY_GAMES  # 每次自对弈阶段生成的游戏数量
NUM_TRAINING_EPOCHS_PER_ITERATION = NUM_EPOCHS  # 每次训练阶段的训练轮数 (epochs)
TOTAL_ITERATIONS = 1000  # 循环重复自对弈 -> 训练的次数

MODELS_DIR = "models"
DATA_DIR = "self_play_data"

# 多进程参数
#NUM_WORKERS_SELF_PLAY = os.cpu_count() or 1  # 默认使用所有 CPU 核心
NUM_WORKERS_SELF_PLAY = NUM_WORKERS

def run_script(script_name, *args):
    """
    辅助函数：运行一个 Python 脚本。
    捕获其标准输出和标准错误，并检查其退出代码。
    """
    command = ["python", script_name] + list(args)
    print(f"\n正在运行: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8')

    if process.stdout:
        print("--- 脚本 STDOUT ---")
        print(process.stdout)

    if process.stderr:
        print("--- 脚本 STDERR ---")
        print("Error in script stderr:\n", process.stderr)

    if process.returncode != 0:
        print(f"错误: 脚本 {script_name} 以非零退出代码 {process.returncode} 退出。")
        return False
    return True


def main_pipeline():
    """
    主强化学习流水线函数。
    它将循环执行自对弈数据生成和模型训练。
    """
    print("--- 启动泡姆棋强化学习流水线 ---")
    print(
        f"每次迭代将生成 {NUM_SELF_PLAY_GAMES_PER_ITERATION} 局游戏，并训练 {NUM_TRAINING_EPOCHS_PER_ITERATION} 个 Epoch。")
    print(f"模型将保存到以下目录: {os.path.abspath(MODELS_DIR)}")
    print(f"自对弈数据将存储在: {os.path.abspath(DATA_DIR)}")
    print(f"自对弈将使用 {NUM_WORKERS_SELF_PLAY} 个工作进程。")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 检查 DATA_DIR 中已有的游戏文件，确定起始全局游戏 ID
    existing_game_files = [f for f in os.listdir(DATA_DIR) if f.startswith("game_") and f.endswith(".pkl")]
    start_global_game_id = 0
    if existing_game_files:
        # 找到最大的游戏 ID，从其下一个 ID 开始
        max_game_id = -1
        for f_name in existing_game_files:
            try:
                game_id_str = f_name.replace("game_", "").replace(".pkl", "")
                game_id = int(game_id_str)
                if game_id > max_game_id:
                    max_game_id = game_id
            except ValueError:
                continue
        start_global_game_id = max_game_id + 1

    print(f"检测到现有数据，将从全局游戏 ID {start_global_game_id} 开始生成新数据。")

    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n\n==================== 迭代 {i}/{TOTAL_ITERATIONS} ====================")

        # 步骤 1: 生成自对弈数据
        print(f"\n--- 正在生成自对弈数据 (迭代 {i}) ---")
        # 将 MODELS_DIR, NUM_WORKERS_SELF_PLAY 和 start_global_game_id 作为命令行参数传递给 self_play_worker.py
        success_self_play = run_script("self_play_worker.py", MODELS_DIR, str(NUM_WORKERS_SELF_PLAY),
                                       str(start_global_game_id))
        if not success_self_play:
            print(f"自对弈工作器在迭代 {i} 中失败。停止流水线。")
            break

        # 更新下一轮迭代的起始全局游戏 ID
        start_global_game_id += NUM_SELF_PLAY_GAMES_PER_ITERATION

        # 步骤 2: 训练模型
        print(f"\n--- 正在训练模型 (迭代 {i}) ---")
        # 将当前迭代编号和 MODELS_DIR 作为命令行参数传递给 train_model.py
        success_train = run_script("train_model.py", str(i), MODELS_DIR)
        if not success_train:
            print(f"模型训练器在迭代 {i} 中失败。停止流水线。")
            break

        # 可选: 在迭代之间添加少量延迟，以避免 CPU/GPU 过载或观察输出
        # time.sleep(5)

    print("\n--- 强化学习流水线已完成 ---")
    print(f"所有模型已保存到 '{os.path.abspath(MODELS_DIR)}' 目录。")
    print("您现在可以运行 evaluate_models.py 来比较模型实力。")


if __name__ == "__main__":
    main_pipeline()
