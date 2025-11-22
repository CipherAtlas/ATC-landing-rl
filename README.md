# ATC Landing RL - Rainbow DQN

Air Traffic Control landing scheduling system using Rainbow DQN reinforcement learning.

## Project Structure

```
atc_rl/
├── main.py              # Run trained model simulation
├── train_rainbow.py     # Train Rainbow DQN agent
├── atc_env.py           # Gymnasium environment
├── flight_generator.py   # Flight generation logic
├── flight.py            # Flight class definition
├── renderer.py          # ASCII console renderer
├── requirements.txt     # Python dependencies
├── models/              # Saved models directory
└── logs/                # Training logs directory
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the Rainbow DQN agent:

```bash
python train_rainbow.py --timesteps 1000000
```

Options:
- `--timesteps`: Total training timesteps (default: 1,000,000)
- `--render`: Render during training (slower)
- `--model-path`: Path to save trained model (default: `models/atc_rainbow.zip`)

### Running Simulation

After training, run a simulation with the trained model:

```bash
python main.py --render --speed normal --max-steps 200
```

Options:
- `--model`: Path to trained model (default: `models/atc_rainbow.zip`)
- `--render`: Enable rendering (default: True)
- `--no-render`: Disable rendering
- `--speed`: Simulation speed - `fast`, `normal`, or `slow` (default: `normal`)
- `--max-steps`: Maximum simulation steps (default: 200)
- `--seed`: Random seed for reproducibility

## Environment Details

### State Space (17 values)
- 2 runway availability flags
- Top 5 flights: fuel, wait_time, priority (5 × 3 = 15 values)

### Action Space (11 actions)
- Actions 0-9: Land flight i (0-4) on runway j (1-2)
- Action 10: DO_NOTHING

### Rewards
- Successful landing: +10
- Emergency landing: +20
- Crash: -50 (normal) or -100 (emergency)
- Blocked runway: -15
- Invalid flight: -5
- High wait time: -2 per step
- Doing nothing when planes waiting: -10

### Episode Termination
- 200 flights served
- 1+ crash occurred
- 2000 steps reached

## Features

- **Two runways** with cooldown periods (3-5 steps)
- **Dynamic flight generation**: 1-3 flights every 5 steps
- **Emergency flights**: 5% probability
- **Fuel management**: Flights have 5-20 fuel units
- **Priority scheduling**: Emergency flights prioritized
- **ASCII visualization**: Console-based simulation display


## Lessons learnt:
- Model needs more training. 200,000 timesteps performed well but I think a million would do a better job here
- Reward function is too penalty-heavy. I think I still need to figure out appropriate model structures for simple DQN projects
- The model needs to explore more. Currently its being overly cautious

### Conclusion

- I need to implement better rewards, allow the model to explore more and train it for a lot more timesteps