"""Flight class definition."""
import numpy as np


class Flight:
    """Represents a single flight in the queue."""
    def __init__(self, flight_id: str, fuel: int, eta: int, priority: str):
        self.flight_id = flight_id
        self.fuel = fuel
        self.eta = eta  # Time before entering queue (0 = already waiting)
        self.priority = priority  # "normal" or "emergency"
        self.wait_time = 0
        self.distance = np.random.randint(1, 10)  # Optional distance metric
    
    def is_waiting(self) -> bool:
        """Check if flight is in the landing queue."""
        return self.eta == 0
    
    def is_emergency(self) -> bool:
        """Check if flight is emergency."""
        return self.priority == "emergency"
    
    def tick(self):
        """Update flight state each time step."""
        if self.eta > 0:
            self.eta -= 1
        else:
            self.wait_time += 1

