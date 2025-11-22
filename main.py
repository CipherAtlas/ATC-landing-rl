"""
Main simulation script to run trained Rainbow DQN model.
"""
import argparse
import time
from stable_baselines3 import DQN
from stable_baselines3 import DQN
from atc_env import ATCEnv
from renderer import render_step


def run_simulation(
    model_path: str = "models/atc_rainbow.zip",
    render: bool = True,
    speed: str = "normal",
    max_steps: int = 200,
    seed: int = None
):
    """
    Load trained model and run simulation.
    
    Args:
        model_path: Path to trained model
        render: Whether to render each step
        speed: "fast", "normal", or "slow" (controls delay between steps)
        max_steps: Maximum steps to run
        seed: Random seed for reproducibility
    """
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        model = DQN.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first using train_rainbow.py")
        return
    
    # Create environment
    env = ATCEnv(render_mode="console" if render else None)
    
    # Set speed delay
    speed_delays = {
        "fast": 0.1,
        "normal": 0.5,
        "slow": 1.0
    }
    delay = speed_delays.get(speed.lower(), 0.5)
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    
    print("\n" + "=" * 60)
    print("  ATC SIMULATION - Running Trained Model")
    print("=" * 60)
    print(f"Speed: {speed}")
    print(f"Max steps: {max_steps}")
    print("=" * 60)
    print()
    
    # Run simulation
    done = False
    while not done and step_count < max_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Render
        if render:
            state_info = env.get_state_info()
            render_step(state_info)
            time.sleep(delay)
        elif step_count % 5 == 0:
            # Print summary every 5 steps if not rendering
            print(f"Step {step_count}: Reward={reward:.2f}, "
                  f"Flights Served={info.get('flights_served', 0)}, "
                  f"Waiting={info.get('waiting_flights', 0)}")
        
        done = terminated or truncated
    
    # Final summary
    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Flights Served: {info.get('flights_served', 0)}")
    print(f"Crashes: {info.get('crashes', 0)}")
    print(f"Waiting Flights: {info.get('waiting_flights', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ATC simulation with trained Rainbow DQN model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/atc_rainbow.zip",
        help="Path to trained model (default: models/atc_rainbow.zip)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render simulation (default: True)"
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable rendering"
    )
    parser.add_argument(
        "--speed",
        type=str,
        default="normal",
        choices=["fast", "normal", "slow"],
        help="Simulation speed (default: normal)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum simulation steps (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    run_simulation(
        model_path=args.model,
        render=args.render,
        speed=args.speed,
        max_steps=args.max_steps,
        seed=args.seed
    )

