"""
Agent class representing entities in the simulation.
"""
from enum import Enum, auto
from datetime import datetime
import random
from typing import Optional, List, Dict, Any
from world.planet import Planet

class Action(Enum):
    REST = auto()
    FORAGE = auto()
    EXPLORE = auto()
    FLEE = auto()
    DEFEND = auto()
    HUNT = auto()
    BUILD = auto()
    GATHER = auto()
    HIDE = auto()
    ATTACK = auto()
    TRADE = auto()
    REPRODUCE = auto()
    MIGRATE = auto()  # New migration action

class AgentState:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.energy = 0
        self.health = 0
        self.age = 0
        self.planet_temp = 0
        self.planet_hazards = set()
        self.planet_resources = {}
        self.planet_weather = ""
        self.last_action = None
        self.alive = True
        self.position = (0, 0, 0)  # x, y, z coordinates
        self.inventory = {
            "food": 0,
            "water": 0,
            "materials": 0,
            "tools": 0
        }
        self.stats = {
            "successful_hunts": 0,
            "escapes": 0,
            "builds": 0,
            "trades": 0,
            "reproductions": 0,
            "attacks": 0,
            "gathers": 0,
            "explorations": 0
        }
        self.migration_state = None
        self.migration_target = None
        self.migration_group = None
        self.migration_progress = 0

class Agent:
    # Action definitions with their effects
    ACTIONS = {
        Action.REST: {
            "energy_cost": 5,
            "health_effect": 10,
            "resource_cost": {},
            "cooldown": 1,
            "success_rate": 1.0
        },
        Action.FORAGE: {
            "energy_cost": 3,  # Reduced cost
            "health_effect": 0,  # Removed health penalty
            "resource_gain": {"food": (3, 7), "water": (2, 5)},  # Increased gains
            "success_rate": 0.85,
            "cooldown": 2
        },
        Action.EXPLORE: {
            "energy_cost": 8,
            "health_effect": -1,
            "resource_gain": {"materials": (2, 5)},  # Increased materials gain
            "success_rate": 0.75,
            "cooldown": 3
        },
        Action.FLEE: {
            "energy_cost": 12,
            "health_effect": -1,  # Reduced health penalty
            "success_rate": 0.85,
            "cooldown": 2
        },
        Action.DEFEND: {
            "energy_cost": 8,
            "health_effect": -2,
            "resource_cost": {"tools": 1},
            "success_rate": 0.75,
            "cooldown": 2
        },
        Action.HUNT: {
            "energy_cost": 15,  # Reduced cost
            "health_effect": -2,  # Reduced penalty
            "resource_gain": {"food": (5, 10)},  # Increased food gain
            "resource_cost": {"tools": 1},
            "success_rate": 0.65,
            "cooldown": 3
        },
        Action.BUILD: {
            "energy_cost": 12,
            "health_effect": -1,
            "resource_cost": {"materials": 2},  # Removed tools cost
            "resource_gain": {"tools": (1, 2)},  # Added tools gain
            "success_rate": 0.75,
            "cooldown": 4
        },
        Action.GATHER: {
            "energy_cost": 6,
            "health_effect": 0,  # Removed health penalty
            "resource_gain": {"materials": (3, 6), "water": (2, 4)},  # Increased gains
            "success_rate": 0.9,
            "cooldown": 2
        },
        Action.HIDE: {
            "energy_cost": 2,
            "health_effect": 2,  # Increased health gain
            "success_rate": 0.95,
            "cooldown": 1
        },
        Action.ATTACK: {
            "energy_cost": 20,
            "health_effect": -6,
            "resource_cost": {"tools": 1},
            "resource_gain": {"food": (2, 5)},  # Added food gain from attack
            "success_rate": 0.55,
            "cooldown": 4
        },
        Action.TRADE: {
            "energy_cost": 4,
            "health_effect": 0,
            "resource_gain": {"materials": (3, 6), "tools": (1, 3)},  # Increased gains
            "resource_cost": {"food": 1, "water": 1},  # Reduced costs
            "success_rate": 0.85,
            "cooldown": 2
        },
        Action.REPRODUCE: {
            "energy_cost": 20,  # Reduced cost
            "health_effect": -10,  # Reduced penalty
            "resource_cost": {"food": 3, "water": 2},  # Reduced costs
            "success_rate": 0.5,
            "cooldown": 8  # Reduced cooldown
        },
        Action.MIGRATE: {
            "energy_cost": 30,  # Fixed from -30 to 30
            "health_effect": -5,
            "resource_cost": {
                "food": 5,
                "water": 5
            },
            "success_rate": 0.7,
            "cooldown": 10  # Significant cooldown to prevent constant migration
        }
    }

    def __init__(self, id: str, planet: 'Planet'):
        self.id = id
        self.name = id  # Keep name for backward compatibility
        self.planet = planet
        self.energy = 100
        self.health = 100
        self.age = 0
        self.alive = True
        self.position = (0, 0, 0)  # Adding position attribute
        self.inventory = {
            'food': 10,
            'water': 10,
            'materials': 0,
            'tools': 0
        }
        
        # Stats tracking
        self.stats = {
            "successful_hunts": 0,
            "escapes": 0,
            "builds": 0,
            "trades": 0,
            "reproductions": 0,
            "attacks": 0,
            "gathers": 0,
            "explorations": 0
        }
        
        # Migration related
        self.migration_state = None
        self.migration_cooldown = 0
        self.migration_target = None
        self.reproduction_partner = None  # Add reproduction partner attribute
        
        # Action tracking
        self.last_action = None
        self.action_cooldown = 0
        self.last_action_success = False
        self.action_cooldowns = {action: 0 for action in Action}
        
        # History
        self.action_history = []
        self.state_history = []
        
        # Behavior
        self.behavior = None
        
    def update(self) -> None:
        """Update agent state for one tick."""
        if not self.alive:
            return
            
        self.age += 1
        
        # Handle migration cooldown
        if self.migration_cooldown > 0:
            self.migration_cooldown -= 1
            
        # Handle action cooldown
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            
        # Basic survival costs
        self.energy = max(0, self.energy - 1)  # Basic energy cost per tick
        
        # Check if agent dies
        if self.energy <= 0 or self.health <= 0:
            self.die()
            
    def start_migration(self, target_planet: 'Planet') -> bool:
        """Start migration to target planet."""
        if self.migration_state or self.migration_cooldown > 0:
            return False
            
        success = self.planet.system.migration_manager.initiate_migration(
            self, self.planet, target_planet)
            
        if success:
            self.action_history.append({
                'tick': self.planet.current_tick,
                'type': 'migration_start',
                'target': target_planet.name
            })
            
        return success
        
    def complete_migration(self, target_planet: 'Planet') -> None:
        """Complete migration to target planet."""
        if not self.migration_state:
            return
            
        self.planet.system.migration_manager.complete_migration(
            self, self.planet, target_planet)
            
        self.action_history.append({
            'tick': target_planet.current_tick,
            'type': 'migration_complete',
            'source': self.planet.name,
            'target': target_planet.name
        })
        
    def die(self) -> None:
        """Handle agent death."""
        self.alive = False
        self.planet.add_to_history(f"Agent {self.id} died at age {self.age}")
        self.action_history.append({
            'tick': self.planet.current_tick,
            'type': 'death',
            'age': self.age
        })

    def tick(self):
        if not self.alive:
            return

        self.age += 1
        self.energy = max(0, min(100, self.energy - 1))  # Reduced base energy consumption
        
        # Update cooldowns
        for action in self.action_cooldowns:
            if self.action_cooldowns[action] > 0:
                self.action_cooldowns[action] -= 1
        
        if self.migration_cooldown > 0:
            self.migration_cooldown -= 1
        
        # Update migration if in progress
        if self.migration_state:
            self.update_migration()
        
        # Health effects from environment
        self._apply_environmental_effects()
        
        # Record state before action
        current_state = self._get_current_state()
        
        # Decide and execute action
        if self.behavior:
            action = self.behavior.decide(self)
            if action and self._can_perform_action(action):
                self.last_action = action
                success = self.execute(action)
                
                # Update behavior with success rate
                self.behavior.update_success_rate(action, success)
                
                # Record state after action
                new_state = self._get_current_state()
                self.state_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "tick": self.age,
                    "state_before": current_state,
                    "action": action.name if action else None,
                    "success": success,
                    "decision_reason": self.behavior.get_decision_reason(),
                    "state_after": new_state,
                    "migration_state": self.migration_state,
                    "migration_progress": self.migration_progress
                })
            else:
                # If action is on cooldown or can't be performed, rest instead
                self.last_action = Action.REST
                success = self.execute(Action.REST)
                
                # Record state after rest
                new_state = self._get_current_state()
                self.state_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "tick": self.age,
                    "state_before": current_state,
                    "action": "REST",
                    "success": success,
                    "decision_reason": "fallback_rest",
                    "state_after": new_state,
                    "migration_state": self.migration_state,
                    "migration_progress": self.migration_progress
                })

        # Check death conditions
        if self.energy <= 0 or self.health <= 0:
            self.die()

    def _can_perform_action(self, action):
        """Check if the agent can perform the given action."""
        if not action or not isinstance(action, Action):
            return False
            
        # Check cooldown
        if self.action_cooldowns[action] > 0:
            return False
            
        action_data = self.ACTIONS[action]
        
        # Check energy
        if self.energy < action_data["energy_cost"]:
            return False
            
        # Check resources
        if "resource_cost" in action_data:
            for resource, amount in action_data["resource_cost"].items():
                if self.inventory.get(resource, 0) < amount:
                    return False
                    
        # Special case for reproduction
        if action == Action.REPRODUCE:
            if not self.reproduction_partner:
                return False
                
        return True

    def _apply_environmental_effects(self):
        """Apply environmental effects to the agent."""
        # Temperature effects - more lenient thresholds and reduced penalties
        temp = self.planet.temperature if hasattr(self.planet, 'temperature') else 0
        if abs(temp) > 60:  # Increased threshold
            self.health -= 0.5
        elif abs(temp) > 40:  # Increased threshold
            self.energy -= 0.5
        
        # Weather effects - reduced penalties
        if hasattr(self.planet, 'weather'):
            if self.planet.weather in ["storm", "blizzard", "acid_rain"]:
                self.health -= 0.5
                self.energy -= 0.5
            elif self.planet.weather in ["rain", "snow", "fog"]:
                self.energy -= 0.25  # Further reduced energy loss
        
        # Hazard effects - reduced impact
        if hasattr(self.planet, 'hazards'):
            hazard_count = len(self.planet.hazards)
            if hazard_count > 0:
                self.health -= hazard_count * 0.25  # Further reduced hazard impact
                self.energy -= hazard_count * 0.25
        
        # Check planet resource availability
        planet_resources_depleted = False
        if hasattr(self.planet, 'resources'):
            if (self.planet.resources.get('food', 0) < 10 or 
                self.planet.resources.get('water', 0) < 10):
                planet_resources_depleted = True
                
        # Resource consumption and benefits
        if self.inventory["food"] > 0 and self.inventory["water"] > 0:
            # Consume resources less frequently
            if self.age % 2 == 0:  # Only consume every other tick
                self.inventory["food"] -= 1
                self.inventory["water"] -= 1
                self.health = min(100, self.health + 4)  # Increased health gain
                self.energy = min(100, self.energy + 4)  # Increased energy gain
        else:
            # Apply stronger penalty when inventory is empty
            self.health -= 0.5  # Increased from 0.25
            self.energy -= 0.5  # Increased from 0.25
            
            # Apply stronger penalty if planet resources are also depleted
            if planet_resources_depleted:
                self.health -= 1.0
                self.energy -= 1.0
        
        # Bonus for being on a habitable planet
        if hasattr(self.planet, 'planet_type'):
            habitability_bonus = {
                "temperate": 0.5,
                "garden": 0.5,
                "oceanic": 0.4,
                "lush": 0.3,
                "forest": 0.2
            }.get(self.planet.planet_type, 0)
            
            if habitability_bonus > 0:
                self.health = min(100, self.health + habitability_bonus)
                self.energy = min(100, self.energy + habitability_bonus)
        
        # Ensure values stay within bounds
        self.health = max(0, min(100, self.health))
        self.energy = max(0, min(100, self.energy))
        
        # Check for death conditions
        if self.energy <= 0 or self.health <= 0:
            self.die()

    def execute(self, action):
        if not action or not isinstance(action, Action):
            return False

        action_data = self.ACTIONS[action]
        
        # Check if action can be performed
        if not self._can_perform_action(action):
            return False
            
        # Check resource costs before proceeding
        if "resource_cost" in action_data:
            for resource, amount in action_data["resource_cost"].items():
                if self.inventory.get(resource, 0) < amount:
                    return False  # Not enough resources
        
        # Apply action effects
        self.energy = max(0, self.energy - action_data["energy_cost"])
        self.health = max(0, min(100, self.health + action_data["health_effect"]))
        
        # Handle resource costs
        if "resource_cost" in action_data:
            for resource, amount in action_data["resource_cost"].items():
                self.inventory[resource] = max(0, self.inventory[resource] - amount)
        
        # Determine success based on success rate
        success = random.random() < action_data["success_rate"]
        
        # Only apply resource gains if action was successful
        if success and "resource_gain" in action_data:
            for resource, (min_gain, max_gain) in action_data["resource_gain"].items():
                gain = random.randint(min_gain, max_gain)
                self.inventory[resource] = self.inventory.get(resource, 0) + gain
        
        # Update stats only on success
        if success:
            if action == Action.HUNT:
                self.stats["successful_hunts"] += 1
            elif action == Action.FLEE:
                self.stats["escapes"] += 1
            elif action == Action.BUILD:
                self.stats["builds"] += 1
            elif action == Action.TRADE:
                self.stats["trades"] += 1
            elif action == Action.REPRODUCE:
                self.stats["reproductions"] += 1
            elif action == Action.ATTACK:
                self.stats["attacks"] += 1
            elif action == Action.GATHER:
                self.stats["gathers"] += 1
            elif action == Action.EXPLORE:
                self.stats["explorations"] += 1
        
        # Set cooldown regardless of success
        self.action_cooldowns[action] = action_data["cooldown"]
        
        # Record action result
        self.last_action = action
        self.last_action_success = success
        
        return success

    def _get_current_state(self):
        """Get current state of the agent."""
        state = {
            "energy": self.energy,
            "health": self.health,
            "age": self.age,
            "alive": self.alive,
            "inventory": self.inventory.copy(),
            
            # Planet state
            "planet_temp": getattr(self.planet, 'temperature', 0),
            "planet_hazards": list(getattr(self.planet, 'hazards', [])),
            "planet_resources": getattr(self.planet, 'resources', {}),
            "planet_weather": getattr(self.planet, 'weather', ''),
            
            # Action state
            "last_action": self.last_action.name if self.last_action else None,
            "last_action_success": getattr(self, 'last_action_success', False),
            "action_cooldown": self.action_cooldown,
            
            # Migration state
            "migration_state": self.migration_state,
            "migration_cooldown": self.migration_cooldown,
            "migration_target": self.migration_target.id if self.migration_target else None
        }
        return state

    def get_state(self):
        """Get current state for decision making."""
        return self._get_current_state()

    def get_state_history(self):
        """Get the history of states and actions for ML training."""
        return self.state_history

    def start_migration(self, target_planet, group=None):
        """Initialize migration to target planet."""
        if self.migration_cooldown > 0:
            return False
            
        self.migration_state = 'preparing'
        self.migration_target = target_planet
        self.migration_group = group
        self.migration_progress = 0
        return True
        
    def update_migration(self):
        """Update migration progress."""
        if not self.migration_state or not self.migration_target:
            return
            
        if self.migration_state == 'preparing':
            # Check if group is ready
            if self.migration_group and all(a.migration_state == 'preparing' for a in self.migration_group):
                self.migration_state = 'migrating'
        
        elif self.migration_state == 'migrating':
            self.migration_progress += 20  # Progress by 20% each tick
            
            if self.migration_progress >= 100:
                self._complete_migration()
                
    def _complete_migration(self):
        """Complete the migration process."""
        if self.migration_state != 'migrating':
            return
            
        # Record migration history
        self.migration_history.append({
            'timestamp': datetime.now().isoformat(),
            'from_planet': self.planet.name,
            'to_planet': self.migration_target.name,
            'success': True,
            'group_size': len(self.migration_group) if self.migration_group else 1
        })
        
        # Update agent's planet
        old_planet = self.planet
        self.planet = self.migration_target
        
        # Update planet agent lists
        old_planet.remove_agent(self)
        self.migration_target.add_agent(self)
        
        # Reset migration state
        self.migration_state = 'arrived'
        self.migration_cooldown = 50  # Set cooldown to 50 ticks
        self.migration_target = None
        self.migration_group = None
        self.migration_progress = 0
        
    def can_migrate(self) -> bool:
        """Check if agent can initiate migration."""
        return (
            self.alive and
            self.migration_cooldown <= 0 and
            self.migration_state is None and
            self.energy >= 40 and  # Minimum energy required
            self.health >= 40      # Minimum health required
        )
        
    def join_group(self, group: List['Agent']):
        """Join a migration group."""
        if not self.can_migrate():
            return False
            
        self.group_members = set(group)
        self.group_role = 'leader' if len(group) == 1 else 'follower'
        return True
        
    def leave_group(self):
        """Leave current migration group."""
        if self.group_members:
            for member in self.group_members:
                member.group_members.remove(self)
        self.group_members = set()
        self.group_role = None
