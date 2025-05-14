"""
Planet logic and properties.
"""
import random
import math
from datetime import datetime
from typing import List, Dict, Set, Optional
import numpy as np

class Planet:
    PLANET_TYPES = {
        # Highly habitable planets
        "temperate": {
            "weather": ["clear", "light_rain", "cloudy", "misty"],
            "atmosphere": {"N2": 0.78, "O2": 0.21, "CO2": 0.01},
            "temp_range": (15, 25),
            "hazards": set(),
            "habitability": 1.0
        },
        "garden": {
            "weather": ["sunny", "drizzle", "gentle_breeze", "warm"],
            "atmosphere": {"N2": 0.77, "O2": 0.22, "CO2": 0.01},
            "temp_range": (20, 30),
            "hazards": set(),
            "habitability": 0.9
        },
        "oceanic": {
            "weather": ["mild_rain", "sea_breeze", "partly_cloudy"],
            "atmosphere": {"N2": 0.75, "O2": 0.23, "H2O": 0.02},
            "temp_range": (18, 28),
            "hazards": {"mild_storms"},
            "habitability": 0.8
        },
        # Moderately habitable planets
        "lush": {
            "weather": ["warm_rain", "humid", "tropical_breeze"],
            "atmosphere": {"O2": 0.25, "N2": 0.70, "H2O": 0.05},
            "temp_range": (20, 35),
            "hazards": {"high_humidity"},
            "habitability": 0.7
        },
        "forest": {
            "weather": ["light_rain", "windy", "foggy"],
            "atmosphere": {"N2": 0.76, "O2": 0.23, "CO2": 0.01},
            "temp_range": (10, 30),
            "hazards": {"dense_vegetation"},
            "habitability": 0.6
        },
        "savanna": {
            "weather": ["clear", "warm", "breezy", "dry"],
            "atmosphere": {"N2": 0.78, "O2": 0.21, "CO2": 0.01},
            "temp_range": (25, 35),
            "hazards": {"drought"},
            "habitability": 0.5
        },
        # Less habitable planets
        "desert": {
            "weather": ["clear", "sandstorm", "hot"],
            "atmosphere": {"N2": 0.75, "O2": 0.20, "CO2": 0.05},
            "temp_range": (30, 45),
            "hazards": {"dehydration", "heat"},
            "habitability": 0.4
        },
        "tundra": {
            "weather": ["snow", "clear", "cold"],
            "atmosphere": {"N2": 0.78, "O2": 0.20, "CO2": 0.02},
            "temp_range": (-10, 10),
            "hazards": {"cold", "ice"},
            "habitability": 0.3
        },
        "volcanic": {
            "weather": ["ash", "hot", "smoky"],
            "atmosphere": {"CO2": 0.50, "SO2": 0.30, "N2": 0.20},
            "temp_range": (40, 60),
            "hazards": {"lava", "toxic_gas"},
            "habitability": 0.2
        },
        # Extreme planets (rare)
        "toxic": {
            "weather": ["acid_rain", "toxic_fog", "chemical_storm"],
            "atmosphere": {"Cl2": 0.30, "SO2": 0.40, "CO2": 0.30},
            "temp_range": (30, 60),
            "hazards": {"corrosive", "poisonous"},
            "habitability": 0.1
        }
    }

    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
        self.id = f"{name}_{seed}"
        self.temperature = None
        self.gravity = None
        self.atmosphere = {}
        self.weather = None
        self.hazards = set()
        self.agents = []
        self.planet_type = None
        self.current_tick = 0
        
        # Migration tracking
        self.incoming_migrations = []
        self.outgoing_migrations = []
        
        # Coordinates
        self.coords = (
            random.randint(0, 1000),
            random.randint(0, 1000),
            random.randint(0, 1000)
        )
        
        # Resources
        self.resources = {
            "food": random.randint(500, 1000),  # Increased starting resources
            "water": random.randint(500, 1000),
            "minerals": random.randint(200, 500),
            "energy": random.randint(200, 500),
            "gases": random.randint(100, 300)
        }
        
        # History log
        self.history_log = []
        self.creation_date = datetime.now().isoformat()
        
        # Day/night cycle properties
        self.day_length = random.randint(20, 40)
        self.is_day = True
        self.temperature_variation = random.uniform(3, 8)  # Reduced temperature variation
        
        # Capacity and sustainability
        self.max_population = self._calculate_max_population()
        self.resource_regeneration_rates = {
            "food": random.uniform(0.5, 2.0),
            "water": random.uniform(0.3, 1.5),
            "minerals": random.uniform(0.1, 0.5),
            "energy": random.uniform(0.2, 1.0),
            "gases": random.uniform(0.1, 0.4)
        }
        
        # Threat tracking
        self.threat_level = 0.0
        self.threat_history = []
        self.hazard_duration = {}  # Track how long each hazard has been present
        
        self.init_environment()
        self.add_to_history("Planet initialized")

    def add_to_history(self, event):
        """Add an event to planet's history with timestamp."""
        timestamp = datetime.now().isoformat()
        self.history_log.append({
            "timestamp": timestamp,
            "tick": self.current_tick,
            "event": event
        })

    def update_resources(self):
        """Update resources based on weather and time of day."""
        # Base regeneration rates
        base_regen = {
            "food": (2, 5),
            "water": (3, 6),
            "minerals": (1, 3),
            "energy": (2, 4),
            "gases": (1, 2)
        }
        
        # Regeneration multipliers based on planet type
        type_multipliers = {
            "temperate": 1.2,
            "garden": 1.5,
            "oceanic": 1.3,
            "lush": 1.4,
            "forest": 1.2,
            "savanna": 1.1,
            "desert": 0.8,
            "tundra": 0.7,
            "volcanic": 0.6,
            "toxic": 0.4
        }
        
        multiplier = type_multipliers.get(self.planet_type, 1.0)
        
        # Additional multipliers based on conditions
        if self.is_day:
            multiplier *= 1.5  # Increased regeneration during day
        if "rain" in self.weather or "drizzle" in self.weather:
            multiplier *= 1.3  # Boost during rain
        elif "storm" in self.weather:
            multiplier *= 0.7  # Reduced during storms
        
        # Apply regeneration with multipliers
        for resource, (min_regen, max_regen) in base_regen.items():
            regen_amount = random.randint(min_regen, max_regen)
            regen_amount = int(regen_amount * multiplier)
            
            # Special cases for certain weather conditions
            if resource == "water":
                if "rain" in self.weather:
                    regen_amount *= 2
                elif "humid" in self.weather:
                    regen_amount *= 1.5
            elif resource == "energy" and self.is_day:
                if "clear" in self.weather or "sunny" in self.weather:
                    regen_amount *= 2
            
            self.resources[resource] = min(1000, self.resources[resource] + regen_amount)
        
        # Resource depletion based on hazards
        if self.hazards:
            depletion_rate = len(self.hazards) * 0.1  # 10% depletion per hazard
            for resource in self.resources:
                depletion = int(self.resources[resource] * depletion_rate)
                self.resources[resource] = max(0, self.resources[resource] - depletion)
        
        # Ensure minimum resource levels based on planet type
        min_resources = {
            "food": int(100 * multiplier),
            "water": int(100 * multiplier),
            "minerals": int(50 * multiplier),
            "energy": int(50 * multiplier),
            "gases": int(30 * multiplier)
        }
        
        for resource, min_value in min_resources.items():
            self.resources[resource] = max(min_value, self.resources[resource])

    def init_environment(self):
        random.seed(self.seed)
        
        # Select planet type with bias towards habitable planets
        planet_types = list(self.PLANET_TYPES.keys())
        weights = [self.PLANET_TYPES[pt]["habitability"] for pt in planet_types]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]  # Normalize weights
        
        self.planet_type = random.choices(planet_types, weights=weights, k=1)[0]
        planet_data = self.PLANET_TYPES[self.planet_type]
        
        # Initialize temperature closer to the middle of the range
        min_temp, max_temp = planet_data["temp_range"]
        mid_temp = (min_temp + max_temp) / 2
        variation = (max_temp - min_temp) / 4  # Reduced variation
        self.temperature = random.uniform(mid_temp - variation, mid_temp + variation)
        
        # Initialize moderate gravity
        self.gravity = random.uniform(0.8, 1.2)
        
        # Initialize weather (prefer milder weather)
        weather_list = planet_data["weather"]
        if "clear" in weather_list or "mild" in str(weather_list):
            self.weather = random.choice([w for w in weather_list if "storm" not in w])
        else:
            self.weather = random.choice(weather_list)
        
        # Initialize atmosphere
        self.atmosphere = planet_data["atmosphere"].copy()
        
        # Initialize hazards
        self.hazards = planet_data["hazards"].copy()

    def update_day_night_cycle(self):
        self.current_tick = (self.current_tick + 1) % self.day_length
        was_day = self.is_day
        self.is_day = self.current_tick < (self.day_length / 2)
        
        # Log day/night transition
        if was_day != self.is_day:
            self.add_to_history(f"Transition to {'day' if self.is_day else 'night'}")
        
        # Calculate temperature modifier based on time of day
        cycle_progress = self.current_tick / self.day_length
        temperature_modifier = math.sin(cycle_progress * 2 * math.pi) * self.temperature_variation
        
        return temperature_modifier

    def update_weather(self):
        """Update weather with bias towards mild conditions."""
        old_weather = self.weather
        if random.random() < 0.1:  # Reduced weather change probability
            weather_list = self.PLANET_TYPES[self.planet_type]["weather"]
            if "clear" in weather_list or "mild" in str(weather_list):
                mild_weather = [w for w in weather_list if "storm" not in w]
                self.weather = random.choice(mild_weather)
            else:
                self.weather = random.choice(weather_list)
            
            if old_weather != self.weather:
                self.add_to_history(f"Weather changed from {old_weather} to {self.weather}")

    def update_temperature(self):
        """Update temperature with more stability."""
        old_temp = self.temperature
        min_temp, max_temp = self.PLANET_TYPES[self.planet_type]["temp_range"]
        mid_temp = (min_temp + max_temp) / 2
        
        # Smaller temperature changes
        if "storm" in self.weather:
            temp_change = random.uniform(-3, 3)
        elif "rain" in self.weather:
            temp_change = random.uniform(-2, 2)
        else:
            temp_change = random.uniform(-1, 1)
        
        day_night_modifier = self.update_day_night_cycle()
        
        # Temperature tends to return to the middle of the range
        temp_diff = mid_temp - self.temperature
        stabilizing_factor = temp_diff * 0.1
        
        self.temperature += temp_change + day_night_modifier + stabilizing_factor
        self.temperature = max(min_temp, min(max_temp, self.temperature))
        
        # Log significant temperature changes
        if abs(self.temperature - old_temp) > 5:
            self.add_to_history(f"Temperature change: {old_temp:.1f}°C → {self.temperature:.1f}°C")

    def update_environment(self):
        self.update_weather()
        self.update_temperature()
        self.update_resources()
        
        # Update hazards based on current weather
        old_hazards = self.hazards.copy()
        self.hazards = self.PLANET_TYPES[self.planet_type]["hazards"].copy()
        
        # Add weather-based hazards
        if "storm" in self.weather:
            self.hazards.add("high_winds")
        if "rain" in self.weather:
            self.hazards.add("flooding")
        
        # Log hazard changes
        new_hazards = self.hazards - old_hazards
        if new_hazards:
            self.add_to_history(f"New hazards appeared: {', '.join(new_hazards)}")

    def tick(self):
        """Update planet state for one tick."""
        self.current_tick = (self.current_tick + 1) % self.day_length
        
        # Update day/night cycle
        if self.current_tick == 0:
            self.is_day = True
            self.add_to_history("Transition to day")
        elif self.current_tick == self.day_length // 2:
            self.is_day = False
            self.add_to_history("Transition to night")
            
        # Update temperature based on day/night
        if self.is_day:
            self.temperature = min(35, self.temperature + random.uniform(0.1, 0.5))
        else:
            self.temperature = max(15, self.temperature - random.uniform(0.1, 0.5))
            
        # Calculate population-based consumption
        alive_agents = len([a for a in self.agents if a.alive])
        consumption_rate = {
            "food": alive_agents * 3.0,     # Increased consumption
            "water": alive_agents * 2.5,    # Increased consumption
            "minerals": alive_agents * 0.8,
            "energy": alive_agents * 1.0,
            "gases": alive_agents * 0.4
        }
        
        # Update resources with both regeneration and consumption
        for resource, rate in self.resource_regeneration_rates.items():
            # Get current resource amount
            current = self.resources.get(resource, 0)
            
            # Apply consumption first, ensuring it affects the resource level
            consumed = consumption_rate.get(resource, 0)
            current = max(0, current - consumed)
            
            # Then apply regeneration, but less than consumption for key resources when population is high
            regenerated = rate
            if alive_agents > 5 and resource in ['food', 'water']:
                # Reduce regeneration when many agents are present to ensure eventual depletion
                regenerated = rate * (0.8 - (alive_agents * 0.05))
            
            # Update resource with limits
            self.resources[resource] = min(1000, current + regenerated)
            
            # Log critical resource levels
            if current < 50 and resource in ['food', 'water']:
                self.add_to_history(f"Critical {resource} level: {current:.1f}")
        
        # Update hazards
        self._update_hazards()
        
        # Update threat level
        self.update_threat_level()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.planet_type,
            "coords": self.coords,
            "temp": self.temperature,
            "gravity": self.gravity,
            "weather": self.weather,
            "hazards": list(self.hazards),
            "atmosphere": self.atmosphere,
            "resources": self.resources,
            "day_night": {
                "is_day": self.is_day,
                "day_length": self.day_length,
                "current_tick": self.current_tick
            },
            "history": {
                "creation_date": self.creation_date,
                "recent_events": self.history_log[-5:] if self.history_log else []
            }
        }

    def add_agent(self, agent):
        self.agents.append(agent)
        self.add_to_history(f"Agent {agent.name} arrived on the planet")

    def _calculate_max_population(self) -> int:
        """Calculate maximum sustainable population based on resources."""
        base_capacity = sum(self.resources.values()) / 100
        habitability_multiplier = {
            "temperate": 1.5,
            "garden": 2.0,
            "oceanic": 1.2,
            "lush": 1.3,
            "forest": 1.1,
            "desert": 0.6,
            "arctic": 0.5,
            "volcanic": 0.3
        }.get(self.planet_type, 1.0)
        
        return int(base_capacity * habitability_multiplier)
        
    def update_threat_level(self):
        """Update planet's threat level based on conditions."""
        threats = 0.0
        
        # Environmental threats
        threats += len(self.hazards) * 0.2
        threats += abs(self.temperature - 20) / 50
        
        # Resource scarcity
        for resource, amount in self.resources.items():
            if amount < len(self.agents) * 5:  # Less than 5 units per agent
                threats += 0.1
        
        # Update threat history
        self.threat_history.append({
            'timestamp': datetime.now().isoformat(),
            'tick': self.current_tick,
            'level': threats,
            'hazards': list(self.hazards),
            'temperature': self.temperature
        })
        
        # Keep only last 100 threat records
        if len(self.threat_history) > 100:
            self.threat_history.pop(0)
        
        self.threat_level = threats
        
    def get_death_rate(self, window: int = 50) -> float:
        """Calculate death rate over the last n ticks."""
        if not self.agents:
            return 0.0
            
        recent_deaths = sum(1 for agent in self.agents if not agent.alive)
        return recent_deaths / len(self.agents)
        
    def predict_resources(self, ticks_ahead: int = 10) -> Dict[str, float]:
        """Predict resource levels n ticks in the future."""
        predictions = {}
        agents_alive = len([a for a in self.agents if a.alive])
        
        for resource, current in self.resources.items():
            # Estimate consumption
            consumption_rate = agents_alive * 0.5  # Assume 0.5 units per agent per tick
            
            # Estimate regeneration
            regen_rate = self.resource_regeneration_rates[resource]
            
            # Predict future value
            future_value = current + (regen_rate - consumption_rate) * ticks_ahead
            predictions[resource] = max(0, future_value)
            
        return predictions
        
    def is_sustainable(self, population: int = None) -> bool:
        """Check if planet can sustain current or given population."""
        if population is None:
            population = len(self.agents)
            
        predictions = self.predict_resources(20)  # Look 20 ticks ahead
        
        # Check if resources will be sufficient
        min_resources_per_agent = {
            "food": 5,
            "water": 5,
            "minerals": 2,
            "energy": 2,
            "gases": 1
        }
        
        for resource, min_needed in min_resources_per_agent.items():
            if predictions.get(resource, 0) < population * min_needed:
                return False
                
        return True
        
    def remove_agent(self, agent) -> bool:
        """Remove an agent from the planet."""
        if agent in self.agents:
            self.agents.remove(agent)
            self.add_to_history(f"Agent {agent.name} left the planet")
            return True
        return False
        
    def add_agent(self, agent) -> bool:
        """Add an agent to the planet."""
        if len(self.agents) < self.max_population:
            self.agents.append(agent)
            self.add_to_history(f"Agent {agent.name} arrived on the planet")
            return True
        return False
        
    def _update_hazards(self):
        """Update hazards and their duration."""
        # Update existing hazard durations
        for hazard in list(self.hazards):
            self.hazard_duration[hazard] = self.hazard_duration.get(hazard, 0) + 1
            
            # Remove hazards that have been present too long
            if self.hazard_duration[hazard] > 20:  # Hazards last max 20 ticks
                self.hazards.remove(hazard)
                del self.hazard_duration[hazard]
                self.add_to_history(f"Hazard {hazard} dissipated")
                
        # Random chance to add new hazard
        if random.random() < 0.05:  # 5% chance per tick
            new_hazard = random.choice(["storm", "extreme_temp", "radiation", "quake"])
            if new_hazard not in self.hazards:
                self.hazards.add(new_hazard)
                self.hazard_duration[new_hazard] = 0
                self.add_to_history(f"New hazard appeared: {new_hazard}")
                
    def get_sustainability_index(self) -> float:
        """Calculate overall sustainability index (0-1)."""
        if not self.agents:
            return 1.0
            
        factors = [
            self.is_sustainable(),
            1.0 - (len(self.agents) / self.max_population),
            1.0 - (self.threat_level / 2),
            min(1.0, sum(self.resources.values()) / (len(self.agents) * 100))
        ]
        
        return sum(factors) / len(factors)

    def is_full(self) -> bool:
        """Check if planet has reached maximum population capacity."""
        return len(self.agents) >= self.max_population
        
    def is_habitable(self) -> bool:
        """Check if the planet is habitable for agents."""
        if not hasattr(self, 'planet_type'):
            return True  # Default to habitable if type not set
            
        # Define habitable planet types
        habitable_types = {
            'temperate': 1.0,  # Perfect for life
            'garden': 0.9,     # Very good
            'oceanic': 0.8,    # Good with some challenges
            'lush': 0.7,       # Decent but more challenging
            'forest': 0.6      # Challenging but livable
        }
        
        # Check if planet type is habitable
        if self.planet_type in habitable_types:
            # Check temperature range
            if -10 <= self.temperature <= 40:
                # Check resource availability
                if (self.resources.get('food', 0) > 100 and 
                    self.resources.get('water', 0) > 100):
                    return True
                    
        return False
        
    def deplete_resources(self, amount=500):
        """Artificially deplete resources for testing migration."""
        for resource in self.resources:
            self.resources[resource] = max(0, self.resources[resource] - amount)
        self.add_to_history(f"Resources depleted by {amount} units")
