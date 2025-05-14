"""
Adapter to connect the universe simulation with the visualization system.
"""
from typing import List, Dict, Any, Optional
import time

from visualization.renderer import UniverseRenderer
from world.galaxy import Galaxy
from world.system import StarSystem
from world.planet import Planet
from agents.agent import Agent

class SimulationVisualizationAdapter:
    """Adapter to connect simulation state to visualization."""
    
    def __init__(self, width=1200, height=800, update_interval=0.1):
        """Initialize the adapter.
        
        Args:
            width: Width of the visualization window
            height: Height of the visualization window
            update_interval: Time between updates in seconds
        """
        self.renderer = UniverseRenderer(width, height)
        self.update_interval = update_interval
        self.running = False
        
        # Tracked simulation components
        self.galaxy = None
        self.systems = []
        self.planets = []
        self.agents = []
        self.current_tick = 0
    
    def start(self):
        """Start visualization."""
        if not self.running:
            self.running = True
            self.renderer.start()
            return True
        return False
    
    def stop(self):
        """Stop visualization."""
        self.running = False
        self.renderer.stop()
    
    def set_galaxy(self, galaxy: Galaxy):
        """Set the galaxy to visualize."""
        self.galaxy = galaxy
        self.systems = galaxy.systems
        
        # Collect all planets
        self.planets = []
        for system in self.systems:
            self.planets.extend(system.planets)
        
        # Collect all agents
        self.agents = []
        for planet in self.planets:
            self.agents.extend(planet.agents)
        
        # Update renderer data
        self._update_visualization_data()
    
    def add_system(self, system: StarSystem):
        """Add a system to the visualization."""
        if system not in self.systems:
            self.systems.append(system)
            
            # Add planets from this system
            for planet in system.planets:
                if planet not in self.planets:
                    self.planets.append(planet)
                    
                    # Add agents from this planet
                    for agent in planet.agents:
                        if agent not in self.agents:
                            self.agents.append(agent)
            
            # Update renderer data
            self._update_visualization_data()
    
    def add_planet(self, planet: Planet):
        """Add a planet to the visualization."""
        if planet not in self.planets:
            self.planets.append(planet)
            
            # Add agents from this planet
            for agent in planet.agents:
                if agent not in self.agents:
                    self.agents.append(agent)
            
            # Update renderer data
            self._update_visualization_data()
    
    def add_agent(self, agent: Agent):
        """Add an agent to the visualization."""
        if agent not in self.agents:
            self.agents.append(agent)
            
            # Update renderer data
            self._update_visualization_data()
    
    def refresh(self):
        """Refresh all tracked components and update visualization."""
        if self.galaxy:
            self.set_galaxy(self.galaxy)
        else:
            # Rebuild collection of all agents and update only those
            all_agents = []
            for planet in self.planets:
                all_agents.extend(planet.agents)
            
            self.agents = all_agents
            self._update_visualization_data()
    
    def update_tick(self, tick):
        """Update the current tick."""
        self.current_tick = tick
        self._update_visualization_data()
    
    def _update_visualization_data(self):
        """Update the renderer with current simulation data."""
        if self.running:
            self.renderer.update_data(
                systems=self.systems,
                planets=self.planets,
                agents=self.agents,
                current_tick=self.current_tick
            )
    
    def render_and_process_events(self):
        """Process events and render a frame. Returns False if should quit."""
        if not self.running:
            return False
            
        # Process events (returns False if should quit)
        if not self.renderer.process_events():
            self.running = False
            return False
            
        # Update data and render
        self._update_visualization_data()
        self.renderer.render_frame()
        
        return True
    
    def is_paused(self):
        """Return whether the visualization is paused."""
        return self.renderer.paused

def visualize_simulation(galaxy: Galaxy, width=1200, height=800):
    """
    Create an adapter and start visualization for the simulation.
    
    Args:
        galaxy: Galaxy to visualize
        width: Width of the visualization window
        height: Height of the visualization window
        
    Returns:
        The adapter instance
    """
    adapter = SimulationVisualizationAdapter(width, height)
    adapter.set_galaxy(galaxy)
    adapter.start()
    return adapter 