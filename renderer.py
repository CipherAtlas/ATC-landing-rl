from typing import Dict, List
from flight import Flight


def render_step(env_state: Dict, render_mode: str = "console"):
    """
    Render a single step of the ATC simulation in ASCII.
    
    Args:
        env_state: Dictionary containing environment state from env.get_state_info()
        render_mode: "console" or "human"
    """
    step = env_state["step"]
    runway_cooldowns = env_state["runway_cooldowns"]
    runway_cooldown_duration = env_state["runway_cooldown_duration"]
    waiting_flights = env_state["waiting_flights"]
    flights_served = env_state["flights_served"]
    crashes = env_state["crashes"]
    last_action = env_state["last_action"]
    last_reward = env_state["last_reward"]
    
    # Header
    print("=" * 50)
    print(f"  ATC SIM STEP {step}")
    print("=" * 50)
    
    # Runway status
    print("\nRunway Status:")
    for i, cooldown in enumerate(runway_cooldowns, 1):
        if cooldown == 0:
            status = "AVAILABLE"
            cooldown_str = "0"
        else:
            status = "BUSY"
            cooldown_str = f"{cooldown}"
        print(f"  Runway {i}: {status:12} (cooldown: {cooldown_str})")
    
    # Waiting flights
    print("\nWaiting Flights:")
    if len(waiting_flights) == 0:
        print("  (No flights waiting)")
    else:
        print(f"{'ID':<8} {'Fuel':<6} {'Wait':<6} {'Priority':<10}")
        print("-" * 32)
        for flight in waiting_flights[:10]:  # Show up to 10 flights
            priority_str = "EMERGENCY" if flight.is_emergency() else "NORMAL"
            print(f"{flight.flight_id:<8} {flight.fuel:<6} {flight.wait_time:<6} {priority_str:<10}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Flights Served: {flights_served}")
    print(f"  Crashes: {crashes}")
    print(f"  Waiting: {len(waiting_flights)}")
    
    # Last action
    last_landed_flight = env_state.get("last_landed_flight")
    last_landed_runway = env_state.get("last_landed_runway")
    
    if last_action is not None:
        print(f"\nLast Action: ", end="")
        if last_action == 10:
            print("DO_NOTHING")
        elif last_landed_flight and last_landed_runway:
            print(f"Landed {last_landed_flight} on Runway {last_landed_runway}")
        else:
            flight_idx = last_action // 2
            runway_idx = last_action % 2
            print(f"Action {last_action} (Runway {runway_idx + 1})")
        
        if last_reward != 0:
            sign = "+" if last_reward > 0 else ""
            print(f"Reward: {sign}{last_reward:.1f}")
    
    print("=" * 50)
    print()


def render_episode_summary(episode_num: int, total_reward: float, info: Dict):
    """Render summary at the end of an episode."""
    print("\n" + "=" * 50)
    print(f"  EPISODE {episode_num} SUMMARY")
    print("=" * 50)
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Flights Served: {info.get('flights_served', 0)}")
    print(f"Crashes: {info.get('crashes', 0)}")
    print(f"Steps: {info.get('step', 0)}")
    print("=" * 50)
    print()

