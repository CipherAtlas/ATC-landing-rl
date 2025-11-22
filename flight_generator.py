import random
import numpy as np
from typing import List
from flight import Flight


# Airline codes for flight IDs
AIRLINE_CODES = [
    "AF", "DL", "UA", "AA", "BA", "LH", "EK", "QF", "JL", "SQ",
    "AI", "KL", "VS", "IB", "AZ", "TK", "EY", "QR", "CX", "NH"
]


def generate_flights(step_number: int) -> List[Flight]:
    """
    Generate new flights based on spawn rate.
    
    Spawn rate: 1-3 flights every 5 steps
    Emergency probability: 5%
    Fuel range: 5-20
    """
    flights = []
    
    # Spawn 1-3 flights
    num_flights = random.randint(1, 3)
    
    for _ in range(num_flights):
        # Generate flight ID
        airline = random.choice(AIRLINE_CODES)
        flight_num = random.randint(100, 999)
        flight_id = f"{airline}{flight_num}"
        
        # Generate fuel (5-20)
        fuel = random.randint(5, 20)
        
        # Generate ETA (time before entering queue)
        # Some flights arrive immediately (eta=0), others have delay
        eta = random.randint(0, 3)
        
        # Generate priority (5% emergency)
        priority = "emergency" if random.random() < 0.05 else "normal"
        
        flight = Flight(flight_id, fuel, eta, priority)
        flights.append(flight)
    
    return flights

