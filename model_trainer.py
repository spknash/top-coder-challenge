#!/usr/bin/env python3

import json
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import shutil

# Trip type classification
def classify_trip_type(days, miles, receipts):
    """Classify a trip into one of 8 types based on its characteristics."""
    miles_per_day = miles / days if days > 0 else 0
    receipts_per_day = receipts / days if days > 0 else 0
    
    # Short high-mileage trips
    if days <= 3 and (miles > 500 or miles_per_day > 200):
        return "short_high_mileage"
    
    # Short high-receipt trips
    elif days <= 3 and receipts_per_day > 300:
        return "short_high_receipt"
    
    # Medium-length balanced trips
    elif 4 <= days <= 7 and 50 <= miles_per_day <= 200 and receipts_per_day <= 200:
        return "medium_balanced"
    
    # Medium-length high-receipt trips
    elif 4 <= days <= 7 and receipts_per_day > 250:
        return "medium_high_receipt"
    
    # Long low-activity trips
    elif days >= 8 and miles_per_day < 50:
        return "long_low_activity"
    
    # Long high-efficiency trips
    elif days >= 8 and miles_per_day > 150:
        return "long_high_efficiency"
    
    # Long high-receipt trips
    elif days >= 8 and receipts_per_day > 200:
        return "long_high_receipt"
    
    # Edge case: Very low activity trips
    elif miles_per_day < 10:
        return "very_low_activity"
    
    # Default case
    else:
        return "standard"

def load_test_cases(file_path='public_cases.json'):
    """Load test cases from the JSON file."""
    with open(file_path, 'r') as f:
        cases = json.load(f)
    
    # Extract features (inputs) and target (expected output)
    X = []
    y = []
    trip_types = []
    
    for case in cases:
        input_data = case['input']
        days = input_data['trip_duration_days']
        miles = input_data['miles_traveled']
        receipts = input_data['total_receipts_amount']
        
        # Classify the trip type
        trip_type = classify_trip_type(days, miles, receipts)
        
        X.append([days, miles, receipts])
        y.append(case['expected_output'])
        trip_types.append(trip_type)
    
    return np.array(X), np.array(y), trip_types

def train_model_for_trip_type(X, y, trip_type, output_dir):
    """Train a gradient boosting model for a specific trip type."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining model for trip type: {trip_type}")
    print(f"  Training data size: {len(X_train)}")
    print(f"  Validation data size: {len(X_val)}")
    
    # If there's not enough data, use a simpler model with fewer hyperparameters
    if len(X_train) < 20:
        print(f"  Warning: Limited data ({len(X_train)} samples) for {trip_type}. Using simpler model.")
        model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        best_params = {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    else:
        # Define parameter grid for grid search - smaller grid for faster training
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        
        # Create base model
        base_model = GradientBoostingRegressor(random_state=42)
        
        # Perform grid search
        print(f"  Starting grid search for optimal hyperparameters...")
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=min(5, len(X_train) // 2) if len(X_train) > 10 else 2,  # Adjust CV based on data size
            scoring='neg_mean_absolute_error',
            n_jobs=-1  # Use all available cores
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  Best parameters: {best_params}")
    
    # Evaluate on validation set
    if len(X_val) > 0:
        val_predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_predictions)
        print(f"  Validation Mean Absolute Error: ${mae:.2f}")
        
        # Calculate percentage of exact matches (within $0.01)
        exact_matches = np.sum(np.abs(val_predictions - y_val) < 0.01)
        exact_match_percent = exact_matches / len(y_val) * 100 if len(y_val) > 0 else 0
        print(f"  Exact matches (within $0.01): {exact_match_percent:.2f}%")
        
        # Calculate percentage of close matches (within $1.00)
        close_matches = np.sum(np.abs(val_predictions - y_val) < 1.0)
        close_match_percent = close_matches / len(y_val) * 100 if len(y_val) > 0 else 0
        print(f"  Close matches (within $1.00): {close_match_percent:.2f}%")
    else:
        print("  Warning: No validation data available.")
    
    # Visualize feature importance
    visualize_feature_importance(model, output_dir)
    
    # Make predictions on all data to analyze errors
    if len(X) > 0:
        all_predictions = model.predict(X)
        errors = np.abs(all_predictions - y)
        
        # Visualize predictions vs actual
        visualize_predictions(X, y, model, output_dir)
        
        # Analyze errors
        analyze_errors(X, y, model, output_dir)
    
    # Save model
    save_model(model, os.path.join(output_dir, f"{trip_type}_model.pkl"))
    
    # Save model info and performance metrics
    with open(os.path.join(output_dir, 'model_info.txt'), 'w') as f:
        f.write(f"Model for Trip Type: {trip_type}\n")
        f.write(f"Training Data Size: {len(X_train)}\n")
        f.write(f"Validation Data Size: {len(X_val)}\n")
        f.write(f"Best Parameters: {best_params}\n")
        if len(X_val) > 0:
            f.write(f"Validation MAE: ${mae:.2f}\n")
            f.write(f"Exact Match Percentage: {exact_match_percent:.2f}%\n")
            f.write(f"Close Match Percentage: {close_match_percent:.2f}%\n")
    
    return model

def visualize_feature_importance(model, output_dir, feature_names=['trip_duration', 'miles', 'receipts']):
    """Visualize feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def visualize_predictions(X, y, model, output_dir):
    """Visualize predictions vs actual values."""
    predictions = model.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.xlabel('Actual Reimbursement')
    plt.ylabel('Predicted Reimbursement')
    plt.title('Predicted vs Actual Reimbursements')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'))
    plt.close()
    
    # Calculate errors
    errors = predictions - y
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=min(50, len(errors) // 2 + 1) if len(errors) > 0 else 10, alpha=0.75)
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()

def save_model(model, filename):
    """Save the trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved as {filename}")

def analyze_errors(X, y, model, output_dir):
    """Analyze where the model makes the largest errors."""
    predictions = model.predict(X)
    errors = np.abs(predictions - y)
    
    # Only analyze if we have enough data
    if len(errors) < 5:
        return
    
    # Get indices of largest errors
    largest_error_indices = np.argsort(errors)[-min(5, len(errors)):]
    
    # Write error analysis to file
    with open(os.path.join(output_dir, 'error_analysis.txt'), 'w') as f:
        f.write("LARGEST ERRORS ANALYSIS\n")
        f.write("======================\n\n")
        
        for idx in largest_error_indices[::-1]:  # Reverse to show largest first
            f.write(f"Case: Trip Duration={X[idx][0]} days, Miles={X[idx][1]}, Receipts=${X[idx][2]:.2f}\n")
            f.write(f"  Expected: ${y[idx]:.2f}, Predicted: ${predictions[idx]:.2f}, Error: ${errors[idx]:.2f}\n")
            
            # Calculate some ratios that might be interesting
            miles_per_day = X[idx][1] / X[idx][0] if X[idx][0] > 0 else 0
            receipts_per_day = X[idx][2] / X[idx][0] if X[idx][0] > 0 else 0
            
            f.write(f"  Miles/Day: {miles_per_day:.2f}, Receipts/Day: ${receipts_per_day:.2f}\n")
            f.write("\n")

def save_trip_type_distribution(trip_types, output_dir):
    """Save distribution of trip types to a file."""
    # Count occurrences of each trip type
    trip_type_counts = {}
    for trip_type in trip_types:
        if trip_type in trip_type_counts:
            trip_type_counts[trip_type] += 1
        else:
            trip_type_counts[trip_type] = 1
    
    # Sort by count (descending)
    sorted_counts = sorted(trip_type_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Save to file
    with open(os.path.join(output_dir, 'trip_type_distribution.txt'), 'w') as f:
        f.write("TRIP TYPE DISTRIBUTION\n")
        f.write("=====================\n\n")
        f.write(f"Total trips: {len(trip_types)}\n\n")
        
        for trip_type, count in sorted_counts:
            percentage = count / len(trip_types) * 100
            f.write(f"{trip_type}: {count} trips ({percentage:.1f}%)\n")

def main():
    # Create output directory
    output_dir = 'trip_type_models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading test cases...")
    X, y, trip_types = load_test_cases()
    print(f"Loaded {len(X)} test cases")
    
    # Save trip type distribution
    save_trip_type_distribution(trip_types, output_dir)
    
    # Group data by trip type
    trip_type_data = {}
    for i, trip_type in enumerate(trip_types):
        if trip_type not in trip_type_data:
            trip_type_data[trip_type] = {'X': [], 'y': []}
        
        trip_type_data[trip_type]['X'].append(X[i])
        trip_type_data[trip_type]['y'].append(y[i])
    
    # Train models for each trip type
    models = {}
    for trip_type, data in trip_type_data.items():
        # Convert lists to numpy arrays
        X_trip = np.array(data['X'])
        y_trip = np.array(data['y'])
        
        # Create trip type specific directory
        trip_type_dir = os.path.join(output_dir, trip_type)
        
        # Train model
        model = train_model_for_trip_type(X_trip, y_trip, trip_type, trip_type_dir)
        models[trip_type] = model
    
    print("\nAll models trained and saved!")
    print(f"Models are saved in the '{output_dir}' directory, organized by trip type.")
    print("\nTo use these models in calculate_reimbursement.py:")
    print("1. The script will classify each trip")
    print("2. Load the appropriate model based on the trip type")
    print("3. Use that model to predict the reimbursement amount")

if __name__ == "__main__":
    main() 