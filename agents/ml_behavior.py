"""
ML-based behavior system using trained decision tree model.
"""
import os
import joblib
import numpy as np
import pandas as pd
from .agent import Action
from typing import Dict, List, Optional, Any
from .behavior import Behavior
from .agent import Agent, AgentState
from world.migration import MigrationManager

class MLBehavior(Behavior):
    _shared_model = None
    _shared_features = None

    def __init__(self, model_path="models/agent_behavior_model.pkl"):
        super().__init__()
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        self.decision_reason = None
        self.migration_manager = MigrationManager()
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and feature names."""
        if MLBehavior._shared_model is None:  # Add class-level cache
            try:
                if os.path.exists(self.model_path):
                    model_data = joblib.load(self.model_path)
                    MLBehavior._shared_model = model_data['model']
                    MLBehavior._shared_features = model_data['feature_names']
            except Exception as e:
                print(f"Error loading model: {e}")
        
        self.model = MLBehavior._shared_model
        self.feature_names = MLBehavior._shared_features
    
    def _prepare_features(self, state: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for model prediction."""
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
            
        # Extract basic features
        features = {
            'energy': state.get('energy', 0),
            'health': state.get('health', 0),
            'age': state.get('age', 0),
            'alive': int(state.get('alive', True)),
            'inventory_food': state.get('inventory', {}).get('food', 0),
            'inventory_water': state.get('inventory', {}).get('water', 0),
            'inventory_materials': state.get('inventory', {}).get('materials', 0),
            'inventory_tools': state.get('inventory', {}).get('tools', 0),
            'planet_temp': state.get('planet_temp', 0),
            'hazards_count': len(state.get('planet_hazards', [])),
            
            # Resource ratios
            'food_ratio': state.get('inventory', {}).get('food', 0) / 20.0,
            'water_ratio': state.get('inventory', {}).get('water', 0) / 20.0,
            'tools_ratio': state.get('inventory', {}).get('tools', 0) / 5.0,
            'materials_ratio': state.get('inventory', {}).get('materials', 0) / 10.0,
            
            # Planet resources
            'planet_food_available': state.get('planet_resources', {}).get('food', 0) / 1000.0,
            'planet_water_available': state.get('planet_resources', {}).get('water', 0) / 1000.0,
            'planet_minerals_available': state.get('planet_resources', {}).get('minerals', 0) / 1000.0,
            
            # Migration features
            'planet_threat_level': state.get('planet_threat_level', 0),
            'planet_sustainability': state.get('planet_sustainability', 1.0),
            'planet_death_rate': state.get('planet_death_rate', 0),
            'migration_cooldown': state.get('migration_cooldown', 0),
            'nearby_planets_count': state.get('nearby_planets_count', 0),
            'best_planet_score': state.get('best_planet_score', 0),
            'group_size': state.get('group_size', 0)
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all expected features
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
                
        return df[self.feature_names]
        
    def decide(self, agent: Agent) -> Action:
        """Decide next action using ML model and migration logic."""
        if self.model is None:
            return self._fallback_behavior(agent)
            
        try:
            # Get current state
            state = agent.get_state()
            
            # Add migration-specific information
            state.update(self._get_migration_features(agent))
            
            # Prepare features
            features = self._prepare_features(state)
            
            if features.empty:
                return self._fallback_behavior(agent)
                
            # Check if migration is needed
            if self._should_migrate(agent, state):
                if agent.can_migrate():
                    self.decision_reason = "migration_needed"
                    return Action.MIGRATE
                    
            # Get model prediction
            action_name = self.model.predict(features)[0]
            self.decision_reason = f"ml_prediction_{action_name.lower()}"
            
            # Check if action is feasible
            if not agent._can_perform_action(Action[action_name]):
                return self._fallback_behavior(agent)
                
            return Action[action_name]
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._fallback_behavior(agent)
            
    def _get_migration_features(self, agent: Agent) -> Dict[str, float]:
        """Get migration-related features for decision making."""
        planet = agent.planet
        
        # Get nearby planets from the system
        nearby_planets = [p for p in planet.system.planets if p != planet]
        
        # Find best potential destination
        best_planet = None
        best_score = 0
        if nearby_planets:
            best_planet = self.migration_manager.find_best_destination(planet, nearby_planets)
            if best_planet:
                viability = self.migration_manager.planet_analyzer.analyze_planet(
                    best_planet, planet.coords)
                best_score = viability.total_score
                
        return {
            'planet_threat_level': planet.threat_level,
            'planet_sustainability': planet.get_sustainability_index(),
            'planet_death_rate': planet.get_death_rate(),
            'migration_cooldown': agent.migration_cooldown,
            'nearby_planets_count': len(nearby_planets),
            'best_planet_score': best_score,
            'resource_scarcity': any(v < 100 for v in planet.resources.values())
        }
        
    def _should_migrate(self, agent: Agent, state: Dict[str, Any]) -> bool:
        """Determine if agent should migrate."""
        # Don't migrate if already migrating or on cooldown
        if agent.migration_state or agent.migration_cooldown > 0:
            return False
            
        # Don't migrate if insufficient resources for the journey
        if agent.energy < 50:  # Ensure enough energy for travel
            return False
            
        # Check resource scarcity on current planet
        planet = agent.planet
        critical_resources = {
            'food': planet.resources.get('food', 0) < 100,
            'water': planet.resources.get('water', 0) < 100,
            'minerals': planet.resources.get('minerals', 0) < 50
        }
        
        # Severe resource depletion - prioritize immediate migration
        severe_depletion = (
            planet.resources.get('food', 0) < 20 or 
            planet.resources.get('water', 0) < 20
        )
        
        # Check if any critical resources are scarce
        resources_scarce = any(critical_resources.values())
        
        # Get best alternative planet
        nearby_planets = [p for p in planet.system.planets if p != planet and not p.is_full()]
        best_planet = None
        
        if nearby_planets:
            best_planet = self.migration_manager.find_best_destination(planet, nearby_planets)
        
        # No available destination planets
        if not best_planet:
            return False
            
        # Severe depletion - migrate regardless of destination quality
        if severe_depletion:
            return True
            
        # Regular resource scarcity check
        if best_planet and resources_scarce:
            # Check if target planet has better resources
            target_resources = best_planet.resources
            if (target_resources.get('food', 0) > planet.resources.get('food', 0) * 1.5 or
                target_resources.get('water', 0) > planet.resources.get('water', 0) * 1.5):
                return True
                
        # Check death rate threshold
        if state.get('planet_death_rate', 0) > 0.3:  # Lowered threshold
            return True
            
        # Check sustainability
        if state.get('planet_sustainability', 0) < 0.4 and best_planet:  # Increased threshold
            return True
            
        return False
        
    def _fallback_behavior(self, agent: Agent) -> Action:
        """Enhanced fallback behavior with migration consideration."""
        state = agent.get_state()
        state.update(self._get_migration_features(agent))
        
        # Check migration conditions first
        if self._should_migrate(agent, state):
            if agent.can_migrate():
                self.decision_reason = "fallback_migration_needed"
                return Action.MIGRATE
                
        # Rest of existing fallback logic...
        if state["energy"] <= 25 or state["health"] <= 25:
            if state.get('inventory', {}).get('food', 0) > 0 and state.get('inventory', {}).get('water', 0) > 0:
                self.decision_reason = "fallback_emergency_rest"
                return Action.REST
            else:
                self.decision_reason = "fallback_emergency_forage"
                return Action.FORAGE
                
        # Basic survival needs
        if state.get('inventory', {}).get('food', 0) < 10 or state.get('inventory', {}).get('water', 0) < 10:
            self.decision_reason = "fallback_basic_needs"
            return Action.FORAGE
            
        # Default to rest
        self.decision_reason = "fallback_rest"
        return Action.REST
    
    def update_success_rate(self, action, success):
        """Update action history with success rate."""
        self.action_history.append((action, success))
        if len(self.action_history) > 100:
            self.action_history.pop(0)
    
    def get_decision_reason(self):
        """Get the current decision reason."""
        return self.decision_reason 