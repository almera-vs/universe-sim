"""
Base behavior class for agent decision making.
"""
from abc import ABC, abstractmethod
from .agent import Agent, Action
import random

class Behavior(ABC):
    """Abstract base class for agent behaviors."""
    
    def __init__(self):
        """Initialize behavior."""
        self.last_action = None
        self.action_history = []
        self.success_threshold = 0.6  # Minimum success rate to consider an action viable
        self.risk_tolerance = 0.7  # Higher means more willing to take risks
        self.resource_thresholds = {
            "food": 5,
            "water": 5,
            "materials": 3,
            "tools": 2
        }
        self.decision_reason = "unknown"

    @abstractmethod
    def decide(self, agent: Agent) -> Action:
        """Decide next action for the agent."""
        pass
    
    def update_success_rate(self, action: Action, success: bool) -> None:
        """Update action history with success rate."""
        self.action_history.append((action, success))
        if len(self.action_history) > 100:
            self.action_history.pop(0)
    
    def get_decision_reason(self) -> str:
        """Get the current decision reason."""
        return self.decision_reason

class SimpleBehavior(Behavior):
    def __init__(self):
        super().__init__()
        self.success_threshold = 0.6  # Minimum success rate to consider an action viable
        self.risk_tolerance = 0.7  # Higher means more willing to take risks
        self.resource_thresholds = {
            "food": 5,
            "water": 5,
            "materials": 3,
            "tools": 2
        }
        self.decision_reason = "no_decision"  # Default reason

    def decide(self, agent):
        """Make a decision based on the agent's current state and environment."""
        state = agent.get_state()

        # Get success rates for all actions
        action_success_rates = {
            action: self.get_action_success_rate(action)
            for action in Action
        }
        
        # Emergency responses (highest priority)
        if self._should_flee(state):
            self.decision_reason = "flee_danger"
            return Action.FLEE
        if self._should_defend(state):
            self.decision_reason = "defend_self"
            return Action.DEFEND
        if self._should_hide(state):
            self.decision_reason = "hide_from_threat"
            return Action.HIDE
            
        # Resource management (high priority)
        if self._needs_resources(state):
            action = self._choose_resource_action(state, action_success_rates)
            if action:
                self.decision_reason = f"gather_{action.name.lower()}"
                return action
            
        # Health and energy management
        if self._needs_rest(state):
            self.decision_reason = "rest_recovery"
            return Action.REST
            
        # Growth and development
        if self._can_reproduce(state):
            # Check if reproduction is viable based on success rate
            if action_success_rates[Action.REPRODUCE] >= self.success_threshold:
                self.decision_reason = "reproduction_ready"
                return Action.REPRODUCE
        if self._should_build(state):
            self.decision_reason = "build_improvement"
            return Action.BUILD
            
        # Exploration and gathering
        if self._should_explore(state):
            self.decision_reason = "explore_environment"
            return Action.EXPLORE
        if self._should_gather(state):
            self.decision_reason = "gather_resources"
            return Action.GATHER
            
        # Combat and hunting
        if self._should_hunt(state):
            self.decision_reason = "hunt_food"
            return Action.HUNT
        if self._should_attack(state):
            self.decision_reason = "attack_threat"
            return Action.ATTACK
            
        # Trading
        if self._should_trade(state):
            self.decision_reason = "trade_resources"
            return Action.TRADE
            
        # Fallback behaviors
        if state["energy"] < 40:
            self.decision_reason = "fallback_rest_low_energy"
            return Action.REST
        if state["health"] < 50:
            self.decision_reason = "fallback_rest_low_health"
            return Action.REST
        if any(state["inventory"][r] < self.resource_thresholds[r] for r in self.resource_thresholds):
            self.decision_reason = "fallback_gather_resources"
            return Action.GATHER
            
        # Ultimate fallback
        self.decision_reason = "fallback_explore"
        return Action.EXPLORE

    def _should_flee(self, state):
        """Determine if the agent should flee from danger."""
        # Flee if health is low and there are hazards
        if state["health"] < 30 and state["planet_hazards"]:
            return True
        # Flee if energy is critically low
        if state["energy"] < 20:
            return True
        # Flee if temperature is extreme
        if abs(state["planet_temp"]) > 60:
            return True
        return False

    def _should_defend(self, state):
        """Determine if the agent should defend itself."""
        # Defend if health is moderate and there are hazards
        if 30 <= state["health"] <= 70 and state["planet_hazards"]:
            return True
        # Defend if we have tools and are in danger
        if state["inventory"]["tools"] > 0 and state["planet_hazards"]:
            return True
        return False

    def _should_hide(self, state):
        """Determine if the agent should hide."""
        # Hide if health is low but not critical
        if 20 <= state["health"] <= 40:
            return True
        # Hide if energy is low but not critical
        if 20 <= state["energy"] <= 40:
            return True
        # Hide if weather is severe
        if state["planet_weather"] in ["storm", "blizzard", "sandstorm"]:
            return True
        return False

    def _needs_resources(self, state):
        """Check if the agent needs to gather resources."""
        # Check if any critical resources are below thresholds
        for resource, threshold in self.resource_thresholds.items():
            if state["inventory"][resource] < threshold:
                return True
        return False

    def _choose_resource_action(self, state, success_rates):
        """Choose the most appropriate resource-gathering action based on needs and success rates."""
        actions = []
        
        # Check food needs
        if state["inventory"]["food"] < self.resource_thresholds["food"]:
            actions.append((Action.FORAGE, success_rates[Action.FORAGE]))
            actions.append((Action.HUNT, success_rates[Action.HUNT]))
            
        # Check water needs
        if state["inventory"]["water"] < self.resource_thresholds["water"]:
            actions.append((Action.GATHER, success_rates[Action.GATHER]))
            
        # Check material needs
        if state["inventory"]["materials"] < self.resource_thresholds["materials"]:
            actions.append((Action.GATHER, success_rates[Action.GATHER]))
            actions.append((Action.EXPLORE, success_rates[Action.EXPLORE]))
            
        # Sort by success rate and choose the best
        if actions:
            actions.sort(key=lambda x: x[1], reverse=True)
            return actions[0][0]
            
        return Action.EXPLORE

    def _needs_rest(self, state):
        """Determine if the agent needs to rest."""
        # Rest if energy is low
        if state["energy"] < 40:
            return True
        # Rest if health is low and we have resources
        if state["health"] < 50 and state["inventory"]["food"] > 2 and state["inventory"]["water"] > 2:
            return True
        return False

    def _can_reproduce(self, state):
        """Determine if the agent can and should reproduce."""
        # Check if we have enough resources and health
        if (state["health"] > 80 and 
            state["energy"] > 80 and 
            state["inventory"]["food"] > 10 and 
            state["inventory"]["water"] > 10 and
            state["inventory"]["materials"] > 5):  # Added materials requirement
            
            # Check if we have a partner
            if hasattr(self, 'reproduction_partner') and self.reproduction_partner:
                # Check if partner is also ready to reproduce
                partner_state = self.reproduction_partner.get_state()
                if (partner_state["health"] > 80 and 
                    partner_state["energy"] > 80 and 
                    partner_state["inventory"]["food"] > 10 and 
                    partner_state["inventory"]["water"] > 10 and
                    partner_state["inventory"]["materials"] > 5):
                    return True
                    
        return False

    def _should_build(self, state):
        """Determine if the agent should build something."""
        # Build if we have enough materials and tools
        if (state["inventory"]["materials"] > 5 and 
            state["inventory"]["tools"] > 1 and 
            state["energy"] > 60):
            return True
        return False

    def _should_explore(self, state):
        """Determine if the agent should explore."""
        # Explore if we're healthy and have enough energy
        if state["health"] > 70 and state["energy"] > 60:
            return True
        # Explore if we need to find resources
        if (state["inventory"]["food"] < 3 or 
            state["inventory"]["water"] < 3 or 
            state["inventory"]["materials"] < 2):
            return True
        return False

    def _should_gather(self, state):
        """Determine if the agent should gather resources."""
        # Gather if we're running low on any resource
        for resource, threshold in self.resource_thresholds.items():
            if state["inventory"][resource] < threshold:
                return True
        return False

    def _should_hunt(self, state):
        """Determine if the agent should hunt."""
        # Hunt if we have tools and need food
        if (state["inventory"]["tools"] > 0 and 
            state["inventory"]["food"] < self.resource_thresholds["food"] and 
            state["energy"] > 50):
            return True
        return False

    def _should_attack(self, state):
        """Determine if the agent should attack."""
        # Attack if we have tools and are healthy
        if (state["inventory"]["tools"] > 1 and 
            state["health"] > 80 and 
            state["energy"] > 70):
            return True
        return False

    def _should_trade(self, state):
        """Determine if the agent should trade."""
        # Trade if we have excess resources
        if (state["inventory"]["food"] > 10 or 
            state["inventory"]["water"] > 10 or 
            state["inventory"]["materials"] > 8):
            return True
        return False

    def get_action_success_rate(self, action):
        """Get the success rate for a specific action."""
        if not self.action_history:
            return 0.5  # Default to 50% if no history
            
        action_history = [s for a, s in self.action_history if a == action]
        if not action_history:
            return 0.5
            
        return sum(action_history) / len(action_history)
