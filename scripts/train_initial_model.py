"""
Generate and save initial ML model for agent behavior.
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import joblib
import os
from typing import List, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def generate_state(n_samples: int) -> pd.DataFrame:
    """Generate random agent states."""
    return pd.DataFrame({
        'energy': np.random.uniform(0, 100, n_samples),
        'health': np.random.uniform(0, 100, n_samples),
        'age': np.random.uniform(0, 1000, n_samples),
        'alive': np.ones(n_samples),  # All agents alive in training
        'inventory_food': np.random.uniform(0, 20, n_samples),
        'inventory_water': np.random.uniform(0, 20, n_samples),
        'inventory_materials': np.random.uniform(0, 20, n_samples),
        'inventory_tools': np.random.uniform(0, 10, n_samples),
        'planet_temp': np.random.uniform(-50, 50, n_samples),
        'hazards_count': np.random.randint(0, 5, n_samples),
        # Add resource ratios
        'food_ratio': np.random.uniform(0, 1, n_samples),
        'water_ratio': np.random.uniform(0, 1, n_samples),
        'tools_ratio': np.random.uniform(0, 1, n_samples),
        'materials_ratio': np.random.uniform(0, 1, n_samples),
        # Add planet resources
        'planet_food_available': np.random.uniform(0, 1, n_samples),
        'planet_water_available': np.random.uniform(0, 1, n_samples),
        'planet_minerals_available': np.random.uniform(0, 1, n_samples)
    })

def decide_action(state: Dict[str, float]) -> str:
    """Decide action based on state using realistic rules."""
    # Calculate probabilities for each action based on state
    probs = {
        'REST': 0.15,  # Base probabilities to ensure minimum representation
        'FORAGE': 0.15,
        'EXPLORE': 0.05,
        'FLEE': 0.05,
        'DEFEND': 0.05,
        'HUNT': 0.05,
        'BUILD': 0.05,
        'GATHER': 0.15,
        'HIDE': 0.05,
        'ATTACK': 0.05,
        'TRADE': 0.05,
        'REPRODUCE': 0.05
    }
    
    # Emergency conditions
    if state['energy'] <= 20 or state['health'] <= 20:
        probs['REST'] += 0.5
        probs['FORAGE'] += 0.2 if state['energy'] > 10 else 0
        probs['HIDE'] += 0.1
    
    # Resource gathering
    elif state['inventory_food'] <= 5 or state['inventory_water'] <= 5:
        if state['energy'] >= 30:
            probs['FORAGE'] += 0.4
            probs['HUNT'] += 0.2 if state['inventory_tools'] >= 2 else 0
            probs['GATHER'] += 0.2
        else:
            probs['REST'] += 0.4
            probs['HIDE'] += 0.2
    
    # Hazard responses
    elif state['hazards_count'] >= 3:
        if state['energy'] >= 40 and state['inventory_tools'] >= 1:
            probs['DEFEND'] += 0.3
            probs['FLEE'] += 0.2
            probs['HIDE'] += 0.2
            probs['ATTACK'] += 0.1
        else:
            probs['FLEE'] += 0.4
            probs['HIDE'] += 0.3
    
    # Tool management
    elif state['inventory_tools'] <= 2 and state['inventory_materials'] >= 2:
        if state['energy'] >= 40:
            probs['BUILD'] += 0.4
            probs['GATHER'] += 0.2
            probs['REST'] += 0.1
        else:
            probs['REST'] += 0.3
            probs['GATHER'] += 0.3
    
    # Combat and hunting
    elif (state['health'] >= 80 and state['energy'] >= 70 and 
          state['inventory_tools'] >= 2):
        if state['inventory_food'] <= 8:
            probs['HUNT'] += 0.3
            probs['ATTACK'] += 0.2
            probs['FORAGE'] += 0.2
            probs['GATHER'] += 0.1
        else:
            probs['ATTACK'] += 0.3
            probs['HUNT'] += 0.2
            probs['EXPLORE'] += 0.2
            probs['DEFEND'] += 0.1
    
    # Trading
    elif (state['inventory_food'] >= 15 and state['inventory_water'] >= 15 and
          (state['inventory_materials'] <= 5 or state['inventory_tools'] <= 2)):
        probs['TRADE'] += 0.4
        probs['GATHER'] += 0.2
        probs['BUILD'] += 0.1
    
    # Reproduction
    elif (state['health'] >= 90 and state['energy'] >= 90 and
          state['inventory_food'] >= 15 and state['inventory_water'] >= 15):
        probs['REPRODUCE'] += 0.3
        probs['EXPLORE'] += 0.2
        probs['TRADE'] += 0.2
        probs['REST'] += 0.1
    
    # Exploration
    elif (state['health'] >= 70 and state['energy'] >= 70 and
          state['inventory_food'] >= 10 and state['inventory_water'] >= 10):
        probs['EXPLORE'] += 0.3
        probs['GATHER'] += 0.2
        probs['HUNT'] += 0.2
        probs['TRADE'] += 0.1
    
    # Default behavior - already set in base probabilities
    
    # Normalize probabilities
    total = sum(probs.values())
    if total > 0:
        probs = {k: v/total for k, v in probs.items()}
    
    # Choose action based on probabilities
    actions = list(probs.keys())
    probabilities = list(probs.values())
    return np.random.choice(actions, p=probabilities)

def create_initial_model():
    # Generate training data
    n_samples = 20000  # Increased sample size for better distribution
    X = generate_state(n_samples)
    feature_names = list(X.columns)
    
    # Generate actions using probabilistic rules
    y = []
    for _, state in X.iterrows():
        action = decide_action(state)
        y.append(action)
    
    # Convert to numpy array
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a more complex decision tree with balanced settings
    model = DecisionTreeClassifier(
        max_depth=15,          # Increased depth for more complex decisions
        min_samples_split=5,   # Reduced to allow more splits
        min_samples_leaf=3,    # Reduced to allow more specific rules
        class_weight='balanced',  # Use balanced class weights
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("\nTraining Set Distribution:")
    train_dist = pd.Series(y_train).value_counts()
    total_train = len(y_train)
    for action, count in train_dist.items():
        percentage = (count / total_train) * 100
        print(f"{action}: {count} ({percentage:.1f}%)")
    
    print("\nTest Set Predictions Distribution:")
    test_dist = pd.Series(test_pred).value_counts()
    total_test = len(test_pred)
    for action, count in test_dist.items():
        percentage = (count / total_test) * 100
        print(f"{action}: {count} ({percentage:.1f}%)")
    
    # Save model and feature names
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the model data
    joblib.dump(model_data, 'models/agent_behavior_model.pkl')
    print("\nModel saved successfully!")

if __name__ == "__main__":
    create_initial_model() 