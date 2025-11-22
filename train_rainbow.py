"""
Training script for DQN on ATC landing scheduling.
Uses DQN with hyperparameters optimized for the ATC task.
Note: Full Rainbow DQN is not available in sb3_contrib, so we use standard DQN
with the specified hyperparameters which work well for this task.
"""
import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from atc_env import ATCEnv


def train_rainbow_dqn(
    total_timesteps: int = 1_000_000,
    save_path: str = "models/atc_rainbow.zip",
    log_dir: str = "logs/",
    render_training: bool = False
):
    """
    Train DQN agent on ATC environment with Rainbow-style hyperparameters.
    
    Args:
        total_timesteps: Total training timesteps
        save_path: Path to save the trained model
        log_dir: Directory for training logs
        render_training: Whether to render during training (slower)
    """
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = ATCEnv(render_mode="console" if render_training else None)
    
    # Create DQN model with specified hyperparameters
    # Note: Full Rainbow DQN (with prioritized replay, dueling, categorical) 
    # is not available in sb3_contrib, so we use standard DQN
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=200000,
        learning_starts=10000,  # Start learning after collecting some samples
        batch_size=32,
        tau=1.0,  # Hard update
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log=log_dir,
        verbose=1,
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "results"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="atc_rainbow"
    )
    
    print("=" * 60)
    print("Starting DQN Training (Rainbow-style hyperparameters)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Model will be saved to: {save_path}")
    print(f"Logs will be saved to: {log_dir}")
    print("=" * 60)
    print()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
        progress_bar=True
    )
    
    # Save final model
    model.save(save_path)
    print(f"\nTraining complete! Model saved to {save_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Rainbow DQN on ATC environment")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during training (slower)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/atc_rainbow.zip",
        help="Path to save trained model"
    )
    
    args = parser.parse_args()
    
    train_rainbow_dqn(
        total_timesteps=args.timesteps,
        save_path=args.model_path,
        render_training=args.render
    )

