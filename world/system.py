"""
Star system class representing a collection of planets.
"""
from typing import List, Dict, Optional, Tuple
import random
import numpy as np

from agents.agent import Agent
from .planet import Planet
import math

class StarSystem:
    def __init__(self, name: str, position: Tuple[float, float, float]):
        self.name = name
        self.position = position  # x, y, z coordinates in the galaxy
        self.planets: List[Planet] = []
        self.connected_systems: List['StarSystem'] = []
        self.travel_costs: Dict['StarSystem', float] = {}  # Energy cost to travel to other systems
        
        # System characteristics
        self.star_type = random.choice(['yellow_dwarf', 'red_dwarf', 'blue_giant', 'white_dwarf'])
        self.star_temperature = self._generate_star_temperature()
        self.habitable_zone = self._calculate_habitable_zone()
        
        # Migration tracking
        self.incoming_migrations = []
        self.outgoing_migrations = []
        self.migration_history = []
        
        # Generate initial planets
        self._generate_planets()
        
    def _generate_planets(self):
        """
        This method is intentionally disabled as planets are created externally 
        through the create_planets_with_spacing function in simulation/runner.py.
        Keeping this as a placeholder for API compatibility.
        """
        # No planets are generated here - planets are created by the simulation runner
        pass
        
    def _generate_star_temperature(self) -> float:
        """Generate star temperature based on type."""
        temp_ranges = {
            'yellow_dwarf': (5000, 6000),
            'red_dwarf': (2500, 4000),
            'blue_giant': (20000, 30000),
            'white_dwarf': (8000, 40000)
        }
        min_temp, max_temp = temp_ranges[self.star_type]
        return random.uniform(min_temp, max_temp)
        
    def _calculate_habitable_zone(self) -> Tuple[float, float]:
        """Calculate the habitable zone range based on star temperature."""
        # Simplified calculation based on star temperature
        base_distance = np.sqrt(self.star_temperature / 5778)  # Relative to Earth's distance
        inner_boundary = base_distance * 0.95
        outer_boundary = base_distance * 1.37
        return (inner_boundary, outer_boundary)
        
    def add_planet(self, planet: Planet) -> None:
        """Add a planet to the system."""
        planet.system = self
        self.planets.append(planet)
        
        # Update planet's habitability based on position in habitable zone
        distance_from_star = np.linalg.norm(np.array(planet.coords) - np.array(self.position))
        inner_zone, outer_zone = self.habitable_zone
        
        if inner_zone <= distance_from_star <= outer_zone:
            planet.habitability_bonus = 0.2
        else:
            planet.habitability_bonus = -0.1 * abs(distance_from_star - (inner_zone + outer_zone) / 2)
            
    def connect_system(self, other_system: 'StarSystem') -> None:
        """Establish connection with another star system."""
        if other_system not in self.connected_systems:
            self.connected_systems.append(other_system)
            # Calculate travel cost based on distance
            distance = np.linalg.norm(
                np.array(self.position) - np.array(other_system.position))
            self.travel_costs[other_system] = distance * 0.1  # Energy cost per distance unit
            
            # Reciprocal connection
            if self not in other_system.connected_systems:
                other_system.connect_system(self)
                
    def get_nearest_systems(self, max_distance: float = 1000) -> List[Tuple['StarSystem', float]]:
        """Get list of nearby systems within max_distance, sorted by distance."""
        systems_with_distances = []
        for system in self.connected_systems:
            distance = np.linalg.norm(
                np.array(self.position) - np.array(system.position))
            if distance <= max_distance:
                systems_with_distances.append((system, distance))
                
        return sorted(systems_with_distances, key=lambda x: x[1])
        
    def get_habitable_planets(self) -> List[Planet]:
        """Get list of planets in the habitable zone."""
        habitable_planets = []
        inner_zone, outer_zone = self.habitable_zone
        
        for planet in self.planets:
            distance_from_star = np.linalg.norm(
                np.array(planet.coords) - np.array(self.position))
            if inner_zone <= distance_from_star <= outer_zone:
                habitable_planets.append(planet)
                
        return habitable_planets
        
    def register_migration(self, agents: List['Agent'], 
                         target_system: 'StarSystem',
                         target_planet: Planet) -> bool:
        """Register a group migration to another system."""
        if target_system not in self.connected_systems:
            return False
            
        # Calculate total energy cost
        travel_cost = self.travel_costs[target_system]
        
        # Check if all agents can make the journey
        for agent in agents:
            if agent.energy < travel_cost:
                return False
                
        # Register migration
        migration_data = {
            'agents': agents,
            'origin_system': self,
            'target_system': target_system,
            'target_planet': target_planet,
            'progress': 0,
            'travel_cost': travel_cost
        }
        
        self.outgoing_migrations.append(migration_data)
        target_system.incoming_migrations.append(migration_data)
        
        return True
        
    def update_migrations(self) -> None:
        """Update all ongoing migrations."""
        # Update outgoing migrations
        completed_outgoing = []
        for migration in self.outgoing_migrations:
            migration['progress'] += 1
            
            # Complete migration after 10 ticks
            if migration['progress'] >= 10:
                # Apply travel costs
                for agent in migration['agents']:
                    agent.energy -= migration['travel_cost']
                    
                # Move agents to new system/planet
                for agent in migration['agents']:
                    agent.planet.remove_agent(agent)
                    migration['target_planet'].add_agent(agent)
                    agent.planet = migration['target_planet']
                    
                # Record in history
                self.migration_history.append({
                    'type': 'outgoing',
                    'num_agents': len(migration['agents']),
                    'target_system': migration['target_system'].name,
                    'target_planet': migration['target_planet'].name,
                    'success': True
                })
                
                completed_outgoing.append(migration)
                
        # Remove completed migrations
        for migration in completed_outgoing:
            self.outgoing_migrations.remove(migration)
            if migration in migration['target_system'].incoming_migrations:
                migration['target_system'].incoming_migrations.remove(migration)
                
    def get_migration_stats(self) -> Dict[str, int]:
        """Get statistics about migrations."""
        return {
            'active_outgoing': len(self.outgoing_migrations),
            'active_incoming': len(self.incoming_migrations),
            'total_migrations': len(self.migration_history),
            'successful_migrations': sum(1 for m in self.migration_history if m['success'])
        }
        
    def tick(self) -> None:
        """Update system state for one tick."""
        # Update all planets
        for planet in self.planets:
            planet.tick()
            
        # Update ongoing migrations
        self.update_migrations()
        
        # Update star conditions (could affect habitable zone)
        self._update_star_conditions()
        
    def _update_star_conditions(self) -> None:
        """Update star conditions that might affect the system."""
        # Small random fluctuations in star temperature
        self.star_temperature *= random.uniform(0.995, 1.005)
        
        # Update habitable zone based on new temperature
        self.habitable_zone = self._calculate_habitable_zone()
        
        # Update planet conditions based on star
        for planet in self.planets:
            distance_from_star = np.linalg.norm(
                np.array(planet.coords) - np.array(self.position))
            inner_zone, outer_zone = self.habitable_zone
            
            # Update planet temperature based on distance from star
            base_temp = 20 - (distance_from_star - (inner_zone + outer_zone) / 2) * 10
            planet.temperature = base_temp + random.uniform(-5, 5)

    def to_dict(self):
        """Convert system data to dictionary format."""
        return {
            "name": self.name,
            "position": self.position,
            "planets": [planet.to_dict() for planet in self.planets],
            "star_type": self.star_type,
            "star_temperature": self.star_temperature,
            "habitable_zone": self.habitable_zone,
            "migration_stats": self.get_migration_stats()
        }
