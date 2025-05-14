"""
Script to train a decision tree model for agent behavior.
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def prepare_features(df):
    """Prepare features for training."""
    # Select features for training
    feature_columns = [
        # Agent state
        'energy', 'health', 'age', 'alive',
        
        # Environment
        'planet_temp', 'planet_weather', 'hazards_count',
        
        # Resources
        'inventory_food', 'inventory_water', 'inventory_materials', 'inventory_tools',
        
        # Stats
        'successful_hunts', 'escapes', 'builds', 'trades',
        'reproductions', 'attacks', 'gathers', 'explorations',
        
        # Hazards
        'hazard_radiation', 'hazard_toxic_gases', 'hazard_heat_stroke', 'hazard_freezing'
    ]
    
    # Convert weather to numeric using one-hot encoding
    weather_dummies = pd.get_dummies(df['planet_weather'], prefix='weather')
    
    # Combine features
    X = pd.concat([
        df[feature_columns],
        weather_dummies
    ], axis=1)
    
    # Target variable (action)
    y = df['action']
    
    return X, y

def train_model(X, y):
    """Train decision tree model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = DecisionTreeClassifier(
        max_depth=10,  # Limit depth to prevent overfitting
        min_samples_split=5,  # Minimum samples required to split
        min_samples_leaf=2,  # Minimum samples required in leaf node
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, X.columns

def save_model(model, feature_names, output_dir="models"):
    """Save model and feature names."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "agent_behavior_model.pkl")
    joblib.dump(model, model_path)
    
    # Save feature names
    features_path = os.path.join(output_dir, "model_features.pkl")
    joblib.dump(feature_names, features_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Feature names saved to: {features_path}")

def main():
    # Load latest training data
    training_dir = "data/training"
    csv_files = [f for f in os.listdir(training_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No training data found!")
        return
        
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(training_dir, x)))
    data_path = os.path.join(training_dir, latest_file)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    model, feature_names = train_model(X, y)
    
    # Save model
    save_model(model, feature_names)

if __name__ == "__main__":
    main() 