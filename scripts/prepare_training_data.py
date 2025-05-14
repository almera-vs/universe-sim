"""
Script to prepare ML training data from simulation logs.
"""
import json
import os
import pandas as pd
from datetime import datetime

def load_log_file(file_path):
    """Load and parse a JSON log file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_state_features(state):
    """Extract relevant features from agent state."""
    features = {
        # Agent state
        'energy': state['energy'],
        'health': state['health'],
        'age': state['age'],
        'alive': int(state['alive']),
        
        # Environment state
        'planet_temp': state.get('planet_temp', 0),  # Using get() with default value
        'planet_weather': state.get('planet_weather', ''),
        'hazards_count': len(state.get('planet_hazards', [])),
        
        # Resources
        'inventory_food': state['inventory']['food'],
        'inventory_water': state['inventory']['water'],
        'inventory_materials': state['inventory']['materials'],
        'inventory_tools': state['inventory']['tools'],
        
        # Stats
        'successful_hunts': state['stats']['successful_hunts'],
        'escapes': state['stats']['escapes'],
        'builds': state['stats']['builds'],
        'trades': state['stats']['trades'],
        'reproductions': state['stats']['reproductions'],
        'attacks': state['stats']['attacks'],
        'gathers': state['stats']['gathers'],
        'explorations': state['stats']['explorations']
    }
    
    # Add hazard flags
    for hazard in ['radiation', 'toxic_gases', 'heat_stroke', 'freezing']:
        features[f'hazard_{hazard}'] = int(hazard in state.get('planet_hazards', []))
    
    return features

def prepare_training_data(log_dir="data/logs"):
    """Prepare training data from log files."""
    all_data = []
    
    # Process each log file
    for filename in os.listdir(log_dir):
        if not filename.endswith('_agents.json'):
            continue
            
        file_path = os.path.join(log_dir, filename)
        log_data = load_log_file(file_path)
        
        if not log_data or 'agent_logs' not in log_data:
            continue
            
        # Process each agent log
        for agent_log in log_data['agent_logs']:
            agent_data = agent_log['agent']
            
            # Extract features from current state
            features = extract_state_features(agent_data['current_state'])
            
            # Add action and outcome
            features.update({
                'action': agent_data['current_state'].get('last_action'),
                'action_success': int(agent_data['current_state'].get('last_action_success', False)),
                'decision_reason': agent_data['current_state'].get('decision_reason', 'unknown'),
                'tick': agent_log['tick'],
                'timestamp': agent_log['timestamp']
            })
            
            all_data.append(features)
    
    if not all_data:
        print("No data found in log files!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/training/train_dataset_{timestamp}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nAction Distribution:")
    print(df['action'].value_counts())
    print("\nSuccess Rates by Action:")
    success_rates = df.groupby('action')['action_success'].mean()
    print(success_rates)
    print("\nDecision Reason Distribution:")
    print(df['decision_reason'].value_counts().head(10))
    
    print(f"\nDataset saved to: {output_file}")
    return df

if __name__ == "__main__":
    df = prepare_training_data() 