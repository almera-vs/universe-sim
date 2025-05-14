"""
Galaxy class managing star systems and their connections.
"""
from typing import List, Dict, Optional, Tuple
import random
import numpy as np
from .system import StarSystem
from .planet import Planet

class Galaxy:
    def __init__(self, size: float = 1000.0, name: str = "Default Galaxy", seed: Optional[int] = None):
        self.size = size
        self.name = name
        self.seed = seed
        self.systems: List[StarSystem] = []
        self.migration_history = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
    def add_system(self, system: StarSystem) -> None:
        """Add a star system to the galaxy."""
        self.systems.append(system)
        
    def generate_systems(self, num_systems: int) -> None:
        """Generate star systems with random positions."""
        for i in range(num_systems):
            position = (
                random.uniform(-self.size/2, self.size/2),
                random.uniform(-self.size/2, self.size/2),
                random.uniform(-self.size/2, self.size/2)
            )
            system = StarSystem(f"System_{i}", position)
            self.systems.append(system)
            
        # Connect nearby systems
        self._establish_connections()
        
    def _establish_connections(self, max_connections: int = 5) -> None:
        """Create connections between nearby systems."""
        for system in self.systems:
            # Find nearest systems
            distances = []
            for other in self.systems:
                if other != system:
                    distance = np.linalg.norm(
                        np.array(system.position) - np.array(other.position))
                    distances.append((other, distance))
                    
            # Connect to nearest systems
            distances.sort(key=lambda x: x[1])
            for other, distance in distances[:max_connections]:
                system.connect_system(other)
                
    def find_migration_path(self, start_system: StarSystem, 
                          target_system: StarSystem) -> List[StarSystem]:
        """Find path between two systems using simple pathfinding."""
        if start_system == target_system:
            return [start_system]
            
        # Simple BFS pathfinding
        queue = [(start_system, [start_system])]
        visited = {start_system}
        
        while queue:
            current, path = queue.pop(0)
            for next_system in current.connected_systems:
                if next_system == target_system:
                    return path + [next_system]
                if next_system not in visited:
                    visited.add(next_system)
                    queue.append((next_system, path + [next_system]))
                    
        return []  # No path found
        
    def get_system_by_name(self, name: str) -> Optional[StarSystem]:
        """Find system by name."""
        for system in self.systems:
            if system.name == name:
                return system
        return None
        
    def get_nearest_habitable_planet(self, position: Tuple[float, float, float], 
                                   max_distance: float = 1000) -> Optional[Planet]:
        """Find nearest habitable planet from a position."""
        nearest_planet = None
        min_distance = float('inf')
        
        for system in self.systems:
            distance = np.linalg.norm(np.array(position) - np.array(system.position))
            if distance > max_distance:
                continue
                
            for planet in system.get_habitable_planets():
                planet_distance = np.linalg.norm(
                    np.array(position) - np.array(planet.coords))
                if planet_distance < min_distance:
                    min_distance = planet_distance
                    nearest_planet = planet
                    
        return nearest_planet
        
    def get_migration_statistics(self) -> Dict[str, int]:
        """Get galaxy-wide migration statistics."""
        stats = {
            'total_migrations': 0,
            'active_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0,
            'systems_with_migrations': 0
        }
        
        for system in self.systems:
            system_stats = system.get_migration_stats()
            stats['total_migrations'] += system_stats['total_migrations']
            stats['active_migrations'] += (
                system_stats['active_outgoing'] + system_stats['active_incoming'])
            stats['successful_migrations'] += system_stats['successful_migrations']
            
            if (system_stats['active_outgoing'] > 0 or 
                system_stats['active_incoming'] > 0):
                stats['systems_with_migrations'] += 1
                
        stats['failed_migrations'] = (
            stats['total_migrations'] - stats['successful_migrations'])
            
        return stats
        
    def tick(self) -> None:
        """Update galaxy state for one tick."""
        # Update all star systems
        for system in self.systems:
            system.tick()
            
        # Record migration statistics
        self.migration_history.append(self.get_migration_statistics())
        
    def to_dict(self) -> Dict:
        """Convert galaxy data to dictionary format."""
        return {
            "size": self.size,
            "num_systems": len(self.systems),
            "systems": [system.to_dict() for system in self.systems],
            "migration_stats": self.get_migration_statistics(),
            "migration_history": self.migration_history[-100:]  # Keep last 100 records
        }
