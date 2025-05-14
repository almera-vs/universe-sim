"""
Migration management system for agent movement between planets and systems.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from .planet import Planet
from .system import StarSystem
from agents.agent import Agent
import math

@dataclass
class PlanetViability:
    """Stores analysis of planet viability for migration."""
    resource_score: float
    safety_score: float
    population_score: float
    distance_score: float
    total_score: float

class PlanetAnalyzer:
    """Analyzes planets for migration suitability."""
    
    def __init__(self):
        self.resource_weight = 0.4
        self.safety_weight = 0.3
        self.population_weight = 0.2
        self.distance_weight = 0.1
    
    def analyze_planet(self, planet: Planet, origin_coords: Tuple[float, float, float]) -> PlanetViability:
        """Analyze planet viability for migration."""
        resource_score = self._calculate_resource_score(planet)
        safety_score = self._calculate_safety_score(planet)
        population_score = self._calculate_population_score(planet)
        distance_score = self._calculate_distance_score(planet.coords, origin_coords)
        
        total_score = (
            resource_score * self.resource_weight +
            safety_score * self.safety_weight +
            population_score * self.population_weight +
            distance_score * self.distance_weight
        )
        
        return PlanetViability(
            resource_score=resource_score,
            safety_score=safety_score,
            population_score=population_score,
            distance_score=distance_score,
            total_score=total_score
        )
    
    def _calculate_resource_score(self, planet: Planet) -> float:
        """Calculate resource availability score."""
        total_resources = sum(planet.resources.values())
        resource_diversity = len([r for r in planet.resources.values() if r > 0])
        return min(1.0, (total_resources / 3000) * (resource_diversity / len(planet.resources)))
    
    def _calculate_safety_score(self, planet: Planet) -> float:
        """Calculate safety score based on hazards and conditions."""
        hazard_penalty = len(planet.hazards) * 0.2
        temp_penalty = abs(planet.temperature - 20) / 50  # Optimal temp around 20°C
        return max(0.0, 1.0 - hazard_penalty - temp_penalty)
    
    def _calculate_population_score(self, planet: Planet) -> float:
        """Calculate population capacity score."""
        current_pop = len(planet.agents)
        # Prevent division by zero with a minimum value
        optimal_pop = max(1.0, sum(planet.resources.values()) / 200)
        return max(0.0, 1.0 - (current_pop / optimal_pop))
    
    def _calculate_distance_score(self, target_coords: Tuple[float, float, float], 
                                origin_coords: Tuple[float, float, float]) -> float:
        """Calculate distance-based score."""
        distance = np.sqrt(sum((t - o) ** 2 for t, o in zip(target_coords, origin_coords)))
        return max(0.0, 1.0 - (distance / 2000))  # Assume 2000 units is max reasonable distance

class PathFinder:
    """Finds optimal paths between planets and systems."""
    
    def find_path(self, start_planet: Planet, target_planet: Planet) -> List[Tuple[float, float, float]]:
        """Calculate waypoints for migration path."""
        # Simple direct path for now
        return [start_planet.coords, target_planet.coords]
    
    def calculate_energy_cost(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate energy cost for a migration path."""
        total_distance = 0
        for i in range(len(path) - 1):
            distance = np.sqrt(sum((path[i+1][j] - path[i][j]) ** 2 for j in range(3)))
            total_distance += distance
        return total_distance * 0.1  # Energy cost per unit distance

class MigrationManager:
    """Manages migration operations and coordination."""
    
    def __init__(self):
        self.planet_analyzer = PlanetAnalyzer()
        self.path_finder = PathFinder()
        self.active_migrations = []
        self.migration_history = []
    
    def evaluate_migration_need(self, planet: Planet) -> bool:
        """Determine if migration is needed based on planet conditions."""
        alive_agents = len([a for a in planet.agents if a.alive])
        total_agents = len(planet.agents)
        
        if total_agents == 0:
            return False
        
        death_rate = 1 - (alive_agents / total_agents)
        food_scarcity = planet.resources.get("food", 0) < (alive_agents * 10)
        high_threats = len(planet.hazards) >= 3
        
        return death_rate > 0.7 or food_scarcity or high_threats
    
    def find_best_destination(self, source_planet: 'Planet', candidate_planets: List['Planet']) -> Optional['Planet']:
        """Find the best destination planet for migration."""
        if not candidate_planets:
            return None
            
        best_planet = None
        best_score = 0
        
        for planet in candidate_planets:
            viability = self.planet_analyzer.analyze_planet(planet, source_planet.coords)
            if viability.total_score > best_score:
                best_score = viability.total_score
                best_planet = planet
                
        return best_planet
    
    def initiate_migration(self, agent: 'Agent', source_planet: 'Planet', target_planet: 'Planet') -> bool:
        """Initiate migration for a single agent."""
        if not self._validate_migration(agent, source_planet, target_planet):
            return False
            
        # Create migration record
        migration = {
            'agent_id': agent.id,
            'source_planet': source_planet.id,
            'target_planet': target_planet.id,
            'start_tick': source_planet.current_tick,
            'status': 'in_progress',
            'energy_cost': self._calculate_energy_cost(source_planet, target_planet)
        }
        
        # Update agent state
        agent.migration_state = migration
        agent.migration_cooldown = 20  # Set cooldown to prevent frequent migrations
        agent.migration_target = target_planet
        
        # Update planet records
        source_planet.outgoing_migrations.append(migration)
        target_planet.incoming_migrations.append(migration)
        
        # Add to active migrations
        self.active_migrations.append(migration)
        
        # Log the migration
        source_planet.add_to_history(f"Agent {agent.id} started migration to {target_planet.id}")
        target_planet.add_to_history(f"Agent {agent.id} incoming from {source_planet.id}")
        
        return True
    
    def complete_migration(self, agent: 'Agent', source_planet: 'Planet', target_planet: 'Planet') -> None:
        """Complete a migration and update all relevant records."""
        migration = agent.migration_state
        if not migration:
            return
            
        # Update migration record
        migration['status'] = 'completed'
        migration['end_tick'] = target_planet.current_tick
        
        # Remove from active migrations
        if migration in self.active_migrations:
            self.active_migrations.remove(migration)
            
        # Add to history
        self.migration_history.append(migration)
        
        # Update planet populations
        source_planet.remove_agent(agent)
        target_planet.add_agent(agent)
        
        # Clear agent migration state
        agent.migration_state = None
        agent.migration_target = None
        
        # Log completion
        source_planet.add_to_history(f"Agent {agent.id} completed migration to {target_planet.id}")
        target_planet.add_to_history(f"Agent {agent.id} arrived from {source_planet.id}")
        
    def _validate_migration(self, agent: 'Agent', source_planet: 'Planet', target_planet: 'Planet') -> bool:
        """Validate if migration is possible."""
        if agent.migration_state or agent.migration_cooldown > 0:
            return False
            
        energy_cost = self._calculate_energy_cost(source_planet, target_planet)
        if agent.energy < energy_cost:
            return False
            
        if target_planet.is_full():
            return False
            
        return True
        
    def _calculate_energy_cost(self, source_planet: 'Planet', target_planet: 'Planet') -> float:
        """Calculate energy cost for migration between planets."""
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(source_planet.coords, target_planet.coords)))
        base_cost = 50  # Base energy cost
        distance_cost = distance * 0.1  # Additional cost based on distance
        return base_cost + distance_cost
    
    def update_migrations(self):
        """Update all active migrations."""
        completed = []
        
        for migration in self.active_migrations:
            migration['progress'] += 1
            
            # Simple progress check - complete after 5 ticks
            if migration['progress'] >= 5:
                agent_id = migration['agent_id']
                source_planet_id = migration['source_planet']
                target_planet_id = migration['target_planet']
                
                # Find the agent and source planet
                for agent in source_planet_id.agents:
                    if agent.id == agent_id:
                        source_planet = agent.planet
                        break
                
                if source_planet:
                    # Find the target planet
                    for planet in target_planet_id.agents:
                        if planet.id == target_planet_id:
                            target_planet = planet
                            break
                    
                    if target_planet:
                        # Complete the migration
                        source_planet.remove_agent(agent)
                        target_planet.add_agent(agent)
                        agent.planet = target_planet
                        completed.append(migration)
        
        # Remove completed migrations
        for migration in completed:
            self.active_migrations.remove(migration) 