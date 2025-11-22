import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Optional, Tuple
from flight import Flight
from flight_generator import generate_flights


class ATCEnv(gym.Env):
    """Air Traffic Control Environment for landing scheduling."""
    
    metadata = {"render_modes": ["human", "console"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super(ATCEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # Runway configuration
        self.num_runways = 2
        self.runway_cooldown = np.random.randint(3, 6)  # 3-5 steps
        self.runway_cooldowns = [0, 0]  # Cooldown timers for each runway
        
        # Flight queue
        self.flight_queue: List[Flight] = []
        self.max_queue_size = 5  # Top 5 flights in observation
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 2000
        self.flights_served = 0
        self.max_flights_served = 200
        self.crashes = 0
        self.last_action = None
        self.last_reward = 0
        
        # Observation space: [runway1, runway2, top5_fuel, top5_wait, top5_priority]
        # 2 + 5*3 = 17 values
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(17,), dtype=np.float32
        )
        
        # Action space: 11 actions (10 landing combinations + DO_NOTHING)
        # Actions 0-9: Land flight i on runway j
        # Action 10: DO_NOTHING
        self.action_space = spaces.Discrete(11)
        
        # Flight generation
        self.flight_spawn_interval = 5
        self.last_spawn_step = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.flights_served = 0
        self.crashes = 0
        self.flight_queue = []
        self.runway_cooldowns = [0, 0]
        self.runway_cooldown = np.random.randint(3, 6)
        self.last_action = None
        self.last_reward = 0
        self.last_landed_flight = None
        self.last_landed_runway = None
        self.last_spawn_step = 0
        
        # Generate initial flights
        initial_flights = generate_flights(0)
        self.flight_queue.extend(initial_flights)
        
        info = {}
        return self._get_observation(), info
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation vector."""
        obs = np.zeros(17, dtype=np.float32)
        
        # Runway availability (1 = available, 0 = busy)
        obs[0] = 1.0 if self.runway_cooldowns[0] == 0 else 0.0
        obs[1] = 1.0 if self.runway_cooldowns[1] == 0 else 0.0
        
        # Get top 5 waiting flights (sorted by priority then fuel)
        waiting_flights = [f for f in self.flight_queue if f.is_waiting()]
        waiting_flights.sort(key=lambda f: (0 if f.is_emergency() else 1, -f.fuel))
        top5 = waiting_flights[:5]
        
        # Fill in flight data (pad with zeros if fewer than 5)
        idx = 2
        for i in range(5):
            if i < len(top5):
                obs[idx] = float(top5[i].fuel)  # Fuel
                obs[idx + 5] = float(top5[i].wait_time)  # Wait time
                obs[idx + 10] = 1.0 if top5[i].is_emergency() else 0.0  # Priority
            idx += 1
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step in the environment."""
        self.current_step += 1
        
        # 1. Spawn new flights
        if self.current_step - self.last_spawn_step >= self.flight_spawn_interval:
            new_flights = generate_flights(self.current_step)
            self.flight_queue.extend(new_flights)
            self.last_spawn_step = self.current_step
        
        # 2. Decrease fuel for all waiting flights
        for flight in self.flight_queue:
            if flight.is_waiting():
                flight.fuel -= 1
        
        # 3. Increase wait time (handled in Flight.tick())
        # 4. Update runway cooldowns
        for i in range(self.num_runways):
            if self.runway_cooldowns[i] > 0:
                self.runway_cooldowns[i] -= 1
        
        # 5. Agent selects action and execute landing
        reward, landed_info = self._execute_action(action)
        self.last_action = action
        self.last_reward = reward
        if landed_info:
            self.last_landed_flight = landed_info["flight_id"]
            self.last_landed_runway = landed_info["runway"]
        
        # 6. Tick all flights (update eta and wait_time)
        for flight in self.flight_queue:
            flight.tick()
        
        # 7. Remove crashed flights
        crashed = [f for f in self.flight_queue if f.fuel < 0]
        for flight in crashed:
            self.crashes += 1
            if flight.is_emergency():
                reward -= 100  # Heavy penalty for emergency crash
            else:
                reward -= 50  # Penalty for normal crash
            self.flight_queue.remove(flight)
        
        # 8. Check termination conditions
        terminated = False
        truncated = False
        
        if self.crashes >= 1:
            terminated = True
        elif self.flights_served >= self.max_flights_served:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True
        
        # 9. Additional reward shaping
        waiting_count = len([f for f in self.flight_queue if f.is_waiting()])
        
        # Penalty for doing nothing when planes are waiting
        if action == 10 and waiting_count > 0:
            reward -= 10
        
        # Small penalty for high wait times
        for flight in self.flight_queue:
            if flight.is_waiting() and flight.wait_time > 10:
                reward -= 2
        
        # Small positive reward for keeping wait times low
        for flight in self.flight_queue:
            if flight.is_waiting() and flight.wait_time < 5:
                reward += 0.1
        
        info = {
            "flights_served": self.flights_served,
            "crashes": self.crashes,
            "waiting_flights": waiting_count,
            "step": self.current_step
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> Tuple[float, Optional[Dict]]:
        """Execute the selected action and return reward and landing info."""
        reward = 0.0
        
        if action == 10:  # DO_NOTHING
            return 0.0, None
        
        # Decode action: flight_index and runway_index
        flight_index = action // 2
        runway_index = action % 2
        
        # Get waiting flights sorted by priority
        waiting_flights = [f for f in self.flight_queue if f.is_waiting()]
        waiting_flights.sort(key=lambda f: (0 if f.is_emergency() else 1, -f.fuel))
        
        # Check if flight exists
        if flight_index >= len(waiting_flights):
            return -5.0, None  # Penalty for choosing non-existing flight
        
        # Check if runway is available
        if self.runway_cooldowns[runway_index] > 0:
            return -15.0, None  # Penalty for choosing blocked runway
        
        # Execute landing
        flight = waiting_flights[flight_index]
        
        # Remove flight from queue
        self.flight_queue.remove(flight)
        self.flights_served += 1
        
        # Set runway cooldown
        self.runway_cooldowns[runway_index] = self.runway_cooldown
        
        # Reward for successful landing
        reward += 10.0
        if flight.is_emergency():
            reward += 20.0  # Bonus for emergency landing
        
        landing_info = {
            "flight_id": flight.flight_id,
            "runway": runway_index + 1
        }
        
        return reward, landing_info
    
    def render(self):
        """Render the environment (delegates to renderer)."""
        if self.render_mode == "human" or self.render_mode == "console":
            # This will be called from renderer.py
            pass
    
    def get_state_info(self) -> Dict:
        """Get current state information for rendering."""
        waiting_flights = [f for f in self.flight_queue if f.is_waiting()]
        waiting_flights.sort(key=lambda f: (0 if f.is_emergency() else 1, -f.fuel))
        
        return {
            "step": self.current_step,
            "runway_cooldowns": self.runway_cooldowns.copy(),
            "runway_cooldown_duration": self.runway_cooldown,
            "waiting_flights": waiting_flights,
            "flights_served": self.flights_served,
            "crashes": self.crashes,
            "last_action": self.last_action,
            "last_reward": self.last_reward,
            "last_landed_flight": self.last_landed_flight,
            "last_landed_runway": self.last_landed_runway
        }

