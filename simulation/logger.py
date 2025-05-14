"""
JSON logging functionality for the simulation.
"""
import json
import os
from datetime import datetime

class SimulationLogger:
    def __init__(self, sim_id: str, seed: int = None, galaxy = None):
        self.sim_id = sim_id
        self.seed = seed
        self.galaxy = galaxy
        self.tick_count = 0
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("data", "logs", f"sim_{sim_id}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize log files
        self.sim_log_file = os.path.join(self.log_dir, "simulation.json")
        self.metadata_file = os.path.join(self.log_dir, "metadata.json")
        
        # Initialize empty log files
        for file in [self.sim_log_file]:
            with open(file, 'w') as f:
                json.dump([], f)
                
        # Save initial metadata
        self._save_metadata()

    def _append_to_json_file(self, file_path: str, data: dict) -> None:
        """Append data to a JSON file, creating it if it doesn't exist."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Read existing data or create empty list
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    try:
                        file_data = json.load(f)
                        if not isinstance(file_data, list):
                            file_data = []
                    except json.JSONDecodeError:
                        file_data = []
            else:
                file_data = []
                
            # Append new data
            file_data.append(data)
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(file_data, f, indent=2)
                
        except Exception as e:
            print(f"Error appending to JSON file {file_path}: {e}")

    def _save_metadata(self):
        """Save comprehensive simulation metadata."""
        metadata = {
            "simulation_id": self.sim_id,
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "galaxy": self.galaxy.to_dict() if self.galaxy else None
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _log_agent_state(self, agent, tick):
        """Log agent state for a given tick."""
        state = {
            "id": agent.id,
            "tick": tick,
            "energy": agent.energy,
            "health": agent.health,
            "age": agent.age,
            "alive": agent.alive,
            "inventory": agent.inventory.copy(),
            "planet_id": agent.planet.id,
            "last_action": agent.last_action.name if agent.last_action else None,
            "last_action_success": getattr(agent, 'last_action_success', False),
            "migration_state": agent.migration_state,
            "migration_cooldown": agent.migration_cooldown,
            "migration_target": agent.migration_target.id if agent.migration_target else None
        }
        
        # Save to agent-specific log file
        agent_log_file = os.path.join(self.log_dir, f"agent_{agent.id}.json")
        self._append_to_json_file(agent_log_file, state)

    def log_tick(self, planet, agents):
        """Log state for one simulation tick."""
        self.tick_count += 1
        
        # Log each agent's state
        for agent in agents:
            self._log_agent_state(agent, self.tick_count)
        
        # Log planet state
        planet_state = {
            "tick": self.tick_count,
            "planet_id": planet.id,
            "temperature": planet.temperature,
            "weather": planet.weather,
            "hazards": list(planet.hazards),
            "resources": planet.resources,
            "is_day": planet.is_day,
            "agents": [
                {
                    "id": agent.id,
                    "alive": agent.alive,
                    "energy": agent.energy,
                    "health": agent.health,
                    "inventory": agent.inventory,
                    "migration_state": agent.migration_state,
                    "migration_target": agent.migration_target.id if agent.migration_target else None
                }
                for agent in planet.agents
            ]
        }
        
        # Save to planet-specific log file
        planet_log_file = os.path.join(self.log_dir, f"planet_{planet.id}.json")
        self._append_to_json_file(planet_log_file, planet_state)
        
        # Save to main simulation log file
        self._append_to_json_file(self.sim_log_file, planet_state)
        
        # Prepare the main simulation log entry
        log_entry = {
            "tick": self.tick_count,
            "timestamp": datetime.now().isoformat(),
            "planet": {
                "id": planet.id,
                "name": planet.name,
                "type": planet.planet_type,
                "coords": planet.coords,
                "temp": planet.temperature,
                "weather": planet.weather,
                "gravity": planet.gravity,
                "atmosphere": planet.atmosphere,
                "hazards": list(planet.hazards),
                "resources": planet.resources,
                "day_night": {
                    "is_day": planet.is_day,
                    "day_length": planet.day_length,
                    "current_tick": planet.current_tick
                },
                "history": {
                    "recent_events": planet.history_log[-3:] if planet.history_log else []
                }
            },
            "agents": [
                {
                    "name": agent.name,
                    "energy": agent.energy,
                    "health": agent.health,
                    "age": agent.age,
                    "alive": agent.alive,
                    "position": getattr(agent, 'position', (0, 0, 0)),  # Safe access to position
                    "inventory": agent.inventory,
                    "stats": getattr(agent, 'stats', {}),  # Safe access to stats
                    "last_action": agent.last_action.name if hasattr(agent, 'last_action') and agent.last_action else None,
                    "action_success": agent.state_history[-1]["success"] if hasattr(agent, 'state_history') and agent.state_history else None
                }
                for agent in agents if agent.alive
            ],
            "summary": {
                "alive_agents": sum(1 for agent in agents if agent.alive),
                "total_agents": len(agents),
                "average_energy": sum(agent.energy for agent in agents) / len(agents) if agents else 0,
                "average_health": sum(agent.health for agent in agents) / len(agents) if agents else 0,
                "active_hazards": len(planet.hazards),
                "available_resources": sum(planet.resources.values())
            }
        }
        
        try:
            # Read existing logs
            with open(self.sim_log_file, 'r') as f:
                data = json.load(f)
            
            # Append new log entry
            data.append(log_entry)
            
            # Write back to file
            with open(self.sim_log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error logging tick: {e}") 